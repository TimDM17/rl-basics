import torch 
import torch.multiprocessing as mp
import torch.nn as nn
import copy 
import time
import numpy as np
from collections import deque
import threading

import gymnasium as gym
import minigrid

from a3c_worker import A3CWorker

class A3CTrainer:
    def __init__(self, model_class, obs_space, action_space, env_name, num_workers=4, lr=0.001, gamma=0.99, n_steps=5, max_episodes_per_worker=1000):
        self.model_class = model_class
        self.obs_space = obs_space
        self.action_space = action_space
        self.env_name = env_name
        self.num_workers = num_workers
        self.lr = lr
        self.gamma = gamma
        self.n_steps = n_steps
        self.max_episodes_per_worker = max_episodes_per_worker

        
        self.global_model = model_class(obs_space, action_space)
        self.global_model.share_memory()


        self.global_optimizer = torch.optim.Adam(
            self.global_model.parameters(),
            lr=lr
        )

        

        self.global_episode_rewards = deque(maxlen=1000)
        self.global_episode_lengths = deque(maxlen=1000)
        self.stats_lock = threading.Lock()

        self.training_active = True
        self.start_time = None


    def create_local_model(self):

        local_model = self.model_class(self.obs_space, self.action_space)
        local_model.load_state_dict(self.global_model.state_dict())

        return local_model
    
    @staticmethod
    def worker_process_static(worker_id, model_class, obs_space, action_space, env_name, 
                             gamma, n_steps, max_episodes_per_worker, global_model, 
                             global_optimizer, optimizer_lock, stats_queue, control_queue):
        """Static worker process to avoid pickling issues"""
        
        # Create local model
        local_model = model_class(obs_space, action_space)
        local_model.load_state_dict(global_model.state_dict())

        worker = A3CWorker(
            worker_id=worker_id,
            global_model=global_model,
            local_model=local_model,
            optimizer=global_optimizer,
            optimizer_lock=optimizer_lock,  
            env_name=env_name,
            gamma=gamma,
            n_steps=n_steps,
            max_episodes=max_episodes_per_worker
        )

        print(f"Worker {worker_id} gestartet")

        episodes_completed = 0
        while episodes_completed < max_episodes_per_worker:
            try:
                if not control_queue.empty():
                    command = control_queue.get_nowait()
                    if command == "STOP":
                        break
                
                episode_reward, episode_length = worker.run_episode()
                episodes_completed += 1

                stats_queue.put({
                    'worker_id': worker_id,
                    'episode': episodes_completed,
                    'reward': episode_reward,
                    'length': episode_length,
                    'timestamp': time.time()
                })
            except Exception as e:
                print(f"Worker {worker_id} Fehler: {e}")
                break
        
        print(f"Worker {worker_id} beendet nach {episodes_completed} Episodes!")

    def collect_statistics(self, stats_queue):
        total_episodes = 0

        while self.training_active:
            try:
                stats = stats_queue.get(timeout=1.0)

                with self.stats_lock:
                    self.global_episode_rewards.append(stats['reward'])
                    self.global_episode_lengths.append(stats['length'])
                    total_episodes += 1

                if total_episodes % 50 == 0:
                    self.print_progress(total_episodes)
            
            except:
                continue
    
    def print_progress(self, total_episodes):
        if not self.global_episode_rewards:
            return
        
        with self.stats_lock:
            avg_reward_10 = np.mean(list(self.global_episode_rewards)[-10:])
            avg_reward_100 = np.mean(list(self.global_episode_rewards)[-100:])
            avg_length = np.mean(list(self.global_episode_lengths)[-100:])
            max_reward = max(self.global_episode_rewards)

        elapsed_time = time.time() - self.start_time
        episodes_per_second = total_episodes / elapsed_time

        print(f"\n=== TRAINING PROGRESS ===")
        print(f"Total Episodes: {total_episodes}")
        print(f"Training Time: {elapsed_time/60:.1f} minutes")
        print(f"Episodes/sec: {episodes_per_second:.2f}")
        print(f"Avg Reward (last 10): {avg_reward_10:.2f}")
        print(f"Avg Reward (last 100): {avg_reward_100:.2f}")
        print(f"Max Reward: {max_reward:.2f}")
        print(f"Avg Episode Length: {avg_length:.1f}")
        print(f"========================\n")
    
    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'optimizer_state_dict': self.global_optimizer.state_dict(),  # Fixed typo
            'training_stats': {  # Fixed typo
                'rewards': list(self.global_episode_rewards),
                'lengths': list(self.global_episode_lengths)  # Fixed typo
            }
        }, filepath)
        print(f"Model gespeichert: {filepath}")

    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.global_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model geladen: {filepath}")

    def train(self, save_interval_minutes=10):
        print(f"Starte A3C Training mit {self.num_workers} Workers!")
        print(f"Environment: {self.env_name}")
        print(f"Model Parameter: {sum(p.numel() for p in self.global_model.parameters())}")
        print(f"Hyperparameter: lr={self.lr}, gamma={self.gamma}, n_steps={self.n_steps}")

        self.start_time = time.time()

        mp.set_start_method('spawn', force=True)

        
        optimizer_lock = mp.Lock()
        
        stats_queue = mp.Queue()
        control_queues = [mp.Queue() for _ in range(self.num_workers)]
        
        worker_processes = []
        for worker_id in range(self.num_workers):
            process = mp.Process(
                target=A3CTrainer.worker_process_static,  
                args=(
                    worker_id,
                    self.model_class,
                    self.obs_space,
                    self.action_space,
                    self.env_name,
                    self.gamma,
                    self.n_steps,
                    self.max_episodes_per_worker,
                    self.global_model,
                    self.global_optimizer,
                    optimizer_lock,  
                    stats_queue,
                    control_queues[worker_id]
                )
            )
            process.start()
            worker_processes.append(process)
        
        stats_thread = threading.Thread(
            target=self.collect_statistics,
            args=(stats_queue,)
        )
        stats_thread.daemon = True
        stats_thread.start()

        last_save_time = time.time()
        try:
            while any(p.is_alive() for p in worker_processes):
                time.sleep(1)
                
                
                if time.time() - last_save_time > save_interval_minutes * 60:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    self.save_model(f"a3c_checkpoint_{timestamp}.pth")
                    last_save_time = time.time()
        
        except KeyboardInterrupt:
            print("\n Training wird gestoppt...")

            for control_queue in control_queues:  
                control_queue.put("STOP")

        finally:
            self.training_active = False

            for process in worker_processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()

            self.save_model("a3c_final_model.pth")

            total_time = time.time() - self.start_time
            print(f"\n Training beendet!")
            print(f"Gesamtzeit: {total_time/60:.1f} Minuten")
            if self.global_episode_rewards:
                print(f" Finale durchschnittliche Belohnung: {np.mean(self.global_episode_rewards):.2f}")
            print(f"Model gespeichert als: a3c_final_model.pth")