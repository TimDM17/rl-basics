import os
import random
import gymnasium as gym
import minigrid
import torch
import torch.nn as nn

from minigrid.wrappers import FullyObsWrapper
from DQN_model import DQNCNN
from replay_buffer import ReplayBuffer
from preprocessing import preprocess_obs


# Hyperparameters
learning_rate = 0.001
discount_factor = 0.99
target_update_freq = 10  

batch_size = 64
buffer_capacity = 1000

num_episodes = 1000
max_steps_per_episode = 100

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
environment_name = "MiniGrid-FourRooms-v0"

# Create directory for model storage
os.makedirs("models", exist_ok=True)

# Init env
env = gym.make(environment_name) # render_mode="human" for visualization
env = FullyObsWrapper(env)

obs, _ = env.reset()
processed_obs = preprocess_obs(obs)

input_shape = processed_obs.shape
num_actions = env.action_space.n

# Init models
policy_net = DQNCNN(input_shape, num_actions)
target_net = DQNCNN(input_shape, num_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Optimizer & Replay Buffer
optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(buffer_capacity)

# Epsilon‐Greedy action selection
def select_action(state, epsilon):
    # Epsilon‐Greedy
    if random.random() < epsilon:
         action = env.action_space.sample() # Exploration
    else:
        with torch.no_grad():
            q_vals = policy_net(state.unsqueeze(0))
            action = torch.argmax(q_vals).item() # Exploitation
    return action

# Optimization function
def optimize_model():
    if len(replay_buffer) < batch_size:
            return
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Compute Q-values for current states
    q_vals = policy_net(states).gather(1, actions).squeeze()

    # Compute target Q-value using the target network
    with torch.no_grad():
        max_net_q_values = target_net(next_states).max(1)[0]
        targets_q_values = rewards + discount_factor * max_net_q_values * (1 - dones)

    loss = nn.MSELoss()(q_vals, targets_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Main Training loop
rewards_per_episode = []
steps_done = 0

for episode in range(num_episodes):
    obs, _ = env.reset()
    state = preprocess_obs(obs)
    episode_reward = 0
    done = False

    # limit steps per episode
    for step in range(max_steps_per_episode):
        if done:
            break

        # Select action
        action = select_action(state, epsilon)
    
        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = preprocess_obs(next_state)

        # Store transition in replay buffer
        replay_buffer.add(state, action, reward, next_state, done)

        # Update state
        state = next_state
        episode_reward += reward
        
        # Optimize model
        optimize_model()

        # Update target network periodically
        if steps_done % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        steps_done += 1

    # Epsilon‐Decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    rewards_per_episode.append(episode_reward)


    # Print progress
    print(f"Ep {episode+1}/{num_episodes}, Reward: {episode_reward:.3f}, Eps: {epsilon:.3f}")

    if episode == 0:
        best_reward = episode_reward
        torch.save(policy_net.state_dict(), "models/best_model.pt")
        print(f"Initial model saved with reward: {best_reward:.3f}")
    elif episode_reward > best_reward:
        best_reward = episode_reward
        torch.save(policy_net.state_dict(), "models/best_model.pt")
        print(f"New best model saved at episode {episode+1} with reward: {best_reward:.3f}")


# Environment schließen
env.close()
print("Training completed!")
