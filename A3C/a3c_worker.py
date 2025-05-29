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
        self.optimizer = optimizer
        self.gamma = gamma
        self.n_steps = n_steps
        self.max_episodes = max_episodes


        self.env = gym.make(env_name, render_mode=None)

        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
    
    def sync_with_global(self):
        self.local_model.load_state_dict(self.global_model.state_dict())

    
    def compute_returns_advantages(self, rewards, values, last_value, done):
        
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
        
        returns.reverse()
        advantages.reverse()

        return returns, advantages

    
    def collect_experience(self):

        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []

        for steps in self.n_steps:

            state = self.env.get_obs()

            with torch.no_grad():
                state_tensor = {
                    'image': torch.FloatTensor(state['image']).unsqueeze(0)
                }
                dist, value = self.local_model(state_tensor)
            
            action = dist.sample()
            log_prob = dist.log_prob(action)

            states.append(state)
            actions.append(action.item())
            values.append(value.item())
            log_probs.append(log_prob)

            obs, reward, terminated, truncated, info = self.env.step(action.item())
            done = terminated or truncated

            rewards.append(reward)

            if done:
                self.env.reset()
                break

        return states, actions, rewards, values, log_probs, done


    def compute_loss(self, states, actions, rewards, values, log_probs, done):

        if not done and len(states) == self.n_steps:

            with torch.no_grad():
                last_state = self.env.get_obs()
                state_tensor = {
                    'image': torch.FloatTensor(last_state['image']).unsqueeze(0)
                }
                _, last_value = self.local_model(state_tensor)
                last_value = last_value.item()
        else:
            last_value = 0

        returns, advantages = self.compute_returns_advantages(rewards, values, last_value, done)

        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        log_probs = torch.stack(log_probs)


        actor_loss = -(log_probs * advantages.detach()).mean()

        values_tensor = torch.FloatTensor(values)
        critic_loss = F.mse_loss(values_tensor, returns)


        total_loss = actor_loss + 0.5 * critic_loss - 0.01 

        return total_loss, actor_loss, critic_loss
    

    def update_global_model(self, loss):

        self.optimizer.zero_grad()
        loss.backward()

        
        for local_param, global_param in zip(self.local_model.parameters(),
                                            self.global_model.parameters()):
            if global_param is None:
                global_param.grad = local_param.grad.clone()
            else:
                global_param.grad += local_param.grad

        
        self.optimizer.step()

    
    def run_episode(self):

        self.env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            
            self.sync_with_global()

            states, actions, rewards, values, log_probs, done = self.collect_experience()

            episode_reward += sum(rewards)
            episode_length += len(rewards)

            loss, actor_loss, critic_loss = self.comput_loss(states, actions, rewards, values, log_probs, done)
            self.update_global_model(loss)


            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                break
            
        return episode_reward, episode_length
    

    def run(self):

        print(f"Worker {self.worker_id} started...")

        for episode in range(self.max_episode):
            episode_reward, episode_length = self.run_episode()

            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0

                print(f"Worker {self.worker_id} - Episode {episode}: "
                      f"Reward: {episode_reward:.2f}, "
                      f"Length: {episode_length}, "
                      f"Average Reward (100): {avg_reward:.2f}, "
                      f"Average Length (100): {avg_length:.2f}")
        
        print(f"Worker {self.worker_id} beendet!")

        self.env.close()
