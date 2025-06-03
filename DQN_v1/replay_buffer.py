import torch
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        
        # Umwandlung in separate Batches ist effizienter
        batch = list(zip(*transitions))
        
        states = torch.stack(batch[0])
        actions = torch.tensor(batch[1], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(batch[2], dtype=torch.float)
        next_states = torch.stack(batch[3])
        dones = torch.tensor(batch[4], dtype=torch.float)
        
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.buffer)