import numpy as np
import random

from Models.DDQN_Model import DDQN_Model
from ReplayBuffer.DDQN_Replay import DDQN_Replay

import torch
import torch.nn.functional as F
import torch.optim as optim

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

class DDQN_Agent():

    def __init__(self, state_size, action_size, seed):
        """Initialize a DQN Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #initial settings of hyper params
        self.config = dict()
        self.config['buffer_size'] = int(1e5)
        self.config['batch_size'] = 64
        self.config['gamma'] = 0.99
        self.config['tau'] = 1e-3
        self.config['lr'] = 5e-4
        self.config['update_every'] = 4
        self.config['fc1'] = 64
        self.config['fc2'] = 64
        self.config['normalize'] = True

        self.state_size = state_size
        self.action_size = action_size
        self.seed_int = seed
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DDQN_Model(self.state_size, self.action_size, self.seed_int, fc1=self.config['fc1'], fc2=self.config['fc2']).to(self.device)
        self.qnetwork_target = DDQN_Model(self.state_size, self.action_size, self.seed_int, fc1=self.config['fc1'],fc2=self.config['fc2']).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.config['lr'])

        # Replay memory
        self.memory = DDQN_Replay(self.action_size, self.config['buffer_size'], self.config['batch_size'], self.seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def update(self, params:dict):
        if len(params.keys())>0:
            for k,v in params.items():
                if k in self.config.keys():
                    self.config[params[k]] = v
            if len(set(params) & {'buffer_size', 'batch_size'}) > 0:
                self.memory = DDQN_Replay(self.action_size, self.config['buffer_size'], self.config['batch_size'], self.seed)
            if len(set(params) & {'fc1', 'fc2', 'normalize'})>0:
                self.qnetwork_local = DDQN_Model(self.state_size, self.action_size, self.seed_int,
                                                 fc1=self.config['fc1'], fc2=self.config['fc2'], normalize=self.config['normalize']).to(self.device)
                self.qnetwork_target = DDQN_Model(self.state_size, self.action_size, self.seed_int,
                                                  fc1=self.config['fc1'], fc2=self.config['fc2'], normalize=self.config['normalize']).to(self.device)
                self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.config['lr'])
            if len(set(params) & {'lr'})>0:
                self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.config['lr'])

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.config['update_every']
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.config['batch_size']:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0., train=True):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps or train==False:
            return np.argmax(action_values.cpu().data.numpy()).astype(np.int32)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # DQN Implementation
        #Q_target = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        #Q_target = rewards + (self.config['gamma'] * Q_target * (1 - dones))
        #Q = self.qnetwork_local(states).gather(1, actions)

        # Double DQN Implementation
        next_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        Q_target = self.qnetwork_target(next_states).gather(1, next_actions)
        Q_target = rewards + (self.config['gamma'] * Q_target * ( 1 -dones))
        Q = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.config['tau'] * local_param.data + (1.0 - self.config['tau']) * target_param.data)