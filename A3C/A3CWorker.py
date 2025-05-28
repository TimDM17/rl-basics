import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import minigrid
from collections import deque

class A3CWorker:
    def __init__(self, worker_id, global_model, local_model, optimizer, env_name,
                 gamma=0.99, n_steps=5, max_episodes=1000):
        
        self.worker_id = worker_id
        self.global_model = global_model
        self.local_model = local_model
        self.local_model = local_model
        self.optimizer = optimizer
        self.gamma = gamma
        self.n_steps = n_steps
        self.max_episodes = max_episodes


        self.env = gym.make(env_name, render_mode=None)

        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
    
    def sync_with_global(self):
        self.local_model.load_state_dict(self.global_model.state_dict())

    
    def compute_n_step_returns(self, rewards, values, last_value, done):
        
        returns = []
        advantages = []

        if done:
            last_value = 0

        current_return = last_value
        for i in reversed(range(len(rewards))):
            current_return = rewards[i] + self.gamma * current_return
            returns.insert(0, current_return)

            advantage = current_return - values[i]
            advantages.insert(0, advantage)

        return returns, advantages

    
    def collect_experience(self):

        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []


