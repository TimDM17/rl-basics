import os
import random
import gymnasium as gym
import minigrid
import torch
import torch.nn as nn
import argparse

from DQN_model import DQNCNN
from replay_buffer import ReplayBuffer
from preprocessing import preprocess_obs


def init_environment(environment_name):
    # Init env
    env = gym.make(environment_name)
    obs, _ = env.reset()
    processed_obs = preprocess_obs(obs)

    input_shape = processed_obs.shape
    action_space = env.action_space

    return env, input_shape, action_space


def init_models(input_shape, action_space, learning_rate):
    # Init models
    policy_net = DQNCNN(input_shape, action_space)
    target_net = DQNCNN(input_shape, action_space)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

    return policy_net, target_net, optimizer


def select_action(state, epsilon, policy_net, env):
    if random.random() < epsilon:
        action = env.action_space.sample()  # Exploration
    else:
        with torch.no_grad():
            q_vals = policy_net(state.unsqueeze(0))
            action = torch.argmax(q_vals).item()  # Exploitation
    return action


# Optimization function
def optimize_model(policy_net, target_net, optimizer, replay_buffer, batch_size, discount_factor):
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


def run(config):
    if not os.path.exists(config["save_dir"]):
        os.makedirs(config["save_dir"])
    env, input_shape, action_space = init_environment(config["environment_name"])
    policy_net, target_net, optimizer = init_models(input_shape, action_space, config["learning_rate"])
    replay_buffer = ReplayBuffer(config["buffer_capacity"])

    # Training Tracking
    rewards_per_episode = []
    steps_done = 0
    best_reward = float('-inf')

    # Main loop
    for episode in range(config["num_episodes"]):
        obs, _ = env.reset()
        state = preprocess_obs(obs)
        episode_reward = 0
        done = False

        # Going throug an Episode
        for step in range(config["max_steps_per_episode"]):
            if done:
             break

            # Select action
            epsilon = max(config["epsilon_min"],
                           config["epsilon"] * (config["epsilon_decay"] ** episode))
            action = select_action(state, epsilon, policy_net, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess_obs(next_state)

            # Store transition in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)

            # Update state
            state = next_state
            episode_reward += reward
        
            # Optimize model
            optimize_model(policy_net, target_net, optimizer, replay_buffer,
                           config["batch_size"], config["discount_factor"])

            # Update target network periodically
            if steps_done % config["target_update_freq"] == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            steps_done += 1

        # Saving Result
        rewards_per_episode.append(episode_reward)

        # Print progress
        print(f"Ep {episode+1}/{config['num_episodes']}, Reward: {episode_reward:.3f}, Eps: {epsilon:.3f}")

        # Saving best Model
        if episode == 0 or episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(policy_net.state_dict(), f"{config['save_dir']}/best_model.pt")
            print(f"New best model saved at episode {episode+1} with reward: {best_reward:.3f}")
        
    # Environment schließen
    env.close()
    print("Training completed!")

    return rewards_per_episode, policy_net


if __name__ == "__main__":
    default_config = {
        "learning_rate": 0.001,
        "discount_factor": 0.99,
        "target_update_freq": 10,  
        "batch_size": 64,
        "buffer_capacity": 1000,
        "num_episodes": 1000,
        "max_steps_per_episode": 100,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "environment_name": "MiniGrid-Empty-5x5-v0",
        "save_dir": "./results"
    }
    
    
    parser = argparse.ArgumentParser(description='DQN Training für MiniGrid-Umgebungen')
    
    parser.add_argument('--learning_rate', type=float, default=default_config['learning_rate'], 
                        help=f'Lernrate für den Optimizer (default: {default_config["learning_rate"]})')
    parser.add_argument('--discount_factor', type=float, default=default_config['discount_factor'],
                        help=f'Discount-Faktor für zukünftige Belohnungen (default: {default_config["discount_factor"]})')
    parser.add_argument('--target_update_freq', type=int, default=default_config['target_update_freq'],
                        help=f'Aktualisierungshäufigkeit für das Zielnetzwerk (default: {default_config["target_update_freq"]})')
    parser.add_argument('--batch_size', type=int, default=default_config['batch_size'],
                        help=f'Batch-Größe für das Training (default: {default_config["batch_size"]})')
    parser.add_argument('--buffer_capacity', type=int, default=default_config['buffer_capacity'],
                        help=f'Kapazität des Replay-Buffers (default: {default_config["buffer_capacity"]})')
    parser.add_argument('--num_episodes', type=int, default=default_config['num_episodes'],
                        help=f'Anzahl der Trainingsepisoden (default: {default_config["num_episodes"]})')
    parser.add_argument('--max_steps_per_episode', type=int, default=default_config['max_steps_per_episode'],
                        help=f'Maximale Schritte pro Episode (default: {default_config["max_steps_per_episode"]})')
    parser.add_argument('--epsilon', type=float, default=default_config['epsilon'],
                        help=f'Startwert für Epsilon (Exploration) (default: {default_config["epsilon"]})')
    parser.add_argument('--epsilon_min', type=float, default=default_config['epsilon_min'],
                        help=f'Minimaler Epsilon-Wert (default: {default_config["epsilon_min"]})')
    parser.add_argument('--epsilon_decay', type=float, default=default_config['epsilon_decay'],
                        help=f'Abnahmerate für Epsilon (default: {default_config["epsilon_decay"]})')
    parser.add_argument('--environment_name', type=str, default=default_config['environment_name'],
                        help=f'Name der Gymnasium-Umgebung (default: {default_config["environment_name"]})')
    parser.add_argument('--save_dir', type=str, default=default_config['save_dir'],
                        help=f'Verzeichnis zum Speichern der Ergebnisse (default: {default_config["save_dir"]})')
    
    args = parser.parse_args()
    config = vars(args)  
    

    rewards, trained_model = run(config)
















