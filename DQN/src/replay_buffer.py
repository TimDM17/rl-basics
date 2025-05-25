import torch
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Properly handle tensors
        states_tensor = torch.stack(states) if torch.is_tensor(states[0]) else torch.FloatTensor(states)
        next_states_tensor = torch.stack(next_states) if torch.is_tensor(next_states[0]) else torch.FloatTensor(next_states)
        
        return (
            states_tensor,
            torch.LongTensor(actions).unsqueeze(1),
            torch.FloatTensor(rewards),
            next_states_tensor,
            torch.FloatTensor(dones)
        )
        
    def __len__(self):
        return len(self.buffer)