import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from prioritized_memory import Memory

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4              # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = Memory(BUFFER_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        
        self.qnetwork_local.eval()
        self.qnetwork_target.eval()
        with torch.no_grad():
            target_action_values = self.qnetwork_target(next_state)
            expected_action_values = self.qnetwork_local(state)
        
        self.qnetwork_local.train()
        self.qnetwork_target.train()
        
        old_val = expected_action_values[0][action]
        new_val = reward
        if not done:
            new_val += GAMMA * torch.max(target_action_values)
        error = abs(old_val - new_val)
        
        # Save experience in replay memory
        self.memory.add(error, (state, action, reward, next_state, done))
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.tree.n_entries > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(int)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        mini_batches, idxs, is_weights = experiences

        states = torch.from_numpy(np.vstack([mini_batch[0] for mini_batch in mini_batches])).float().to(device)
        actions = torch.from_numpy(np.vstack([mini_batch[1] for mini_batch in mini_batches])).long().to(device)
        rewards = torch.from_numpy(np.vstack([mini_batch[2] for mini_batch in mini_batches])).float().to(device)
        next_states = torch.from_numpy(np.vstack([mini_batch[3] for mini_batch in mini_batches])).float().to(device)
        dones = torch.from_numpy(np.vstack([int(mini_batch[4]) for mini_batch in mini_batches])).float().to(device)

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        Q_source_next = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        
        Q_target = self.qnetwork_target(next_states)
        
        Q_double_target = torch.tensor([Q_target[i][max_index] for i, max_index in enumerate(Q_source_next)]).detach().unsqueeze(1)
        
        Q_observed = rewards + (gamma * Q_double_target * (1 - dones))
        
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        errors = torch.abs(Q_expected - Q_observed).data.numpy()
        
        # update priority
        for i in range(BATCH_SIZE):
            idx = idxs[i]
            self.memory.update(idx, errors[i])
        
        loss = (torch.FloatTensor(is_weights) * F.mse_loss(Q_expected, Q_observed)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
