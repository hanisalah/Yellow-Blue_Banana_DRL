import random
import torch
import numpy as np
from collections import namedtuple, deque

# Add SumTree Data Structure needed to store and handle priorities of experiences.
# The implementation is an adaptation from https://github.com/Howuhh/prioritized_experience_replay
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (experience tuples)
        self.tree = np.zeros(2 * self.capacity - 1)  # Total nodes in the tree
        self.data = [None] * self.capacity
        self.write = 0  # Points to the next leaf node to update
        self.n_entries = 0  # Total number of experiences added

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get_leaf(self, value):
        idx = self._retrieve(0, value)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]

    def _retrieve(self, idx, value):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def total_priority(self):
        return self.tree[0]


class PER_Replay:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.6):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float): prioritizing factor
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.alpha = alpha  # Added factor for PER
        self.epsilon = 1e-5  # Small value to avoid zero priority for PER
        # self.memory = deque(maxlen=buffer_size) #For a PER it is not a simple deque anymore
        self.tree = SumTree(self.buffer_size) # For a PER, we need a SumTree data structure instead of deque
        self.max_priority = 1.0 #Used to give new experiences high priority for PER

        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)

        #self.memory.append(e) #Not needed for PER
        # PER initial priority
        self.tree.add(self.max_priority, e)

    def sample(self, beta=0.4):
        """Randomly sample a batch of experiences from memory."""

        #experiences = random.sample(self.memory, k=self.batch_size) # Not needed for PER
        # PER Implementation
        experiences = []
        idxs = []
        segment = self.tree.total_priority() / self.batch_size
        priorities = []

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(value)
            priorities.append(priority)
            experiences.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total_priority()
        is_weight = np.power(self.tree.n_entries * (sampling_probabilities + self.epsilon), -beta)
        is_weight /= is_weight.max()
        # End of PER implementation

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        is_weight = torch.from_numpy(is_weight).float().to(self.device) #Added for PER Implementation

        # Added is_weight and idxs to the return of the function for PER.
        return (states, actions, rewards, next_states, dones, is_weight, idxs)

    def update_priorities(self, idxs, errors):
        """
        Update Priorities function used for PER
        Params
        ======
        idxs: index in tree to update the priority.
        errors: used to update priority.
        """
        for idx, error in zip(idxs, errors):
            priority = (np.abs(error) + self.epsilon) ** self.alpha
            priority = np.clip(priority, self.epsilon, self.max_priority)
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        """Return the current size of internal memory."""
        #return len(self.memory) Not used in PER
        return self.tree.n_entries #PER implementation