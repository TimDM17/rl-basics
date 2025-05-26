import torch
import gymnasium as gym
import minigrid
from model import MiniGridACModel

# Create environment
env = gym.make("MiniGrid-FourRooms-v0")

obs_space = env.observation_space
action_space = env.action_space

# Create model
model = MiniGridACModel(obs_space, action_space)

obs, info = env.reset()  
obs_batch = {"image": torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0)}


# Forward pass
dist, value = model(obs_batch)
action = dist.sample().item()

print(f"Action sampled: {action}")
print(f"Value estimate: {value.item():.3f}")