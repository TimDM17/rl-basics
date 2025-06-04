# RL-Basics

This repository is for learning and experimenting with Reinforcement Learning (RL) methods and algorithms on the MiniGrid environment. Currently, it contains implementations of:

- **DQN** (Deep Q-Network)  
- **A3C** (Asynchronous Advantage Actor-Critic)

Future RL concepts will be added.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv rl_venv
   ```
2. Activate the environment:
   ```bash
   rl_venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- **A3C**  
  Go to the `A3C_v1` folder:
  ```bash
  cd A3C_v1
  python discrete_A3C.py [--env_name ENV] [--update_iter UPDATE] [--gamma GAMMA] [--max_ep MAX_EP] [--lr LR] ...
  ```
  You can pass these hyperparameters (and more) to customize your training. Refer to the argument definitions in *discrete_A3C.py*.

- **DQN**  
  Go to the `DQN_v1` folder:
  ```bash
  cd DQN_v1
  python train.py [--environment_name ENV] [--learning_rate LR] [--discount_factor DISCOUNT] ...
  ```
  You can pass these hyperparameters (and more) to customize your training. Refer to the argument definitions in *train.py*.
