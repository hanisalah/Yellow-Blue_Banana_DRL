import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN_Model(nn.Module):
    """DQN Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1=64, fc2=64, normalize=True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DQN_Model, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.normalize = normalize
        if self.normalize==True:
            self.norm = nn.LayerNorm(state_size)
        self.fc1 = nn.Linear(state_size,fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, action_size)

    def forward(self, x):
        """Build a network that maps state -> action values."""
        if self.normalize == True:
            x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)