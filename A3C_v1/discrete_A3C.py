import torch
import torch.multiprocessing as mp
import gymnasium as gym
import minigrid
import os
import matplotlib.pyplot as plt
import argparse
from utils import v_wrap, push_and_pull, record
from shared_adam import SharedAdam
from model import Net

os.environ["OMP_NUM_THREADS"] = "1"


class Environment:
    
    
    def __init__(self, env_name='MiniGrid-Empty-5x5-v0'):
        self.env_name = env_name
        env = gym.make(env_name)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def create_env(self, unwrapped=True):
        
        env = gym.make(self.env_name)
        return env.unwrapped if unwrapped else env


class Worker(mp.Process):
    
    
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, config):
        super(Worker, self).__init__()
        self.name = f'w{name:02d}'
        self.g_ep = global_ep
        self.g_ep_r = global_ep_r
        self.res_queue = res_queue
        self.gnet = gnet
        self.opt = opt
        self.config = config
        
        
        self.lnet = Net(config.env.observation_space, config.env.action_space)
        self.env = config.env.create_env(unwrapped=True)

    def run(self):
        
        total_step = 1
        try:
            while self.g_ep.value < self.config.max_ep:
                s, _ = self.env.reset()  
                buffer_s, buffer_a, buffer_r = [], [], []
                ep_r = 0.
                
                while True:
                    
                    if self.name == 'w00' and self.config.render:
                        self.env.render()
                    
                    
                    s_image_normalized = s["image"][None, :] / 255.0
                    s_batch = {"image": v_wrap(s_image_normalized)}
                    a = self.lnet.choose_action(s_batch)
                    
                    
                    s_, r, terminated, truncated, info = self.env.step(a)
                    done = terminated or truncated
                    
                    
                    ep_r += r
                    buffer_a.append(a)
                    buffer_s.append(s) 
                    buffer_r.append(r)

                    
                    if total_step % self.config.update_iter == 0 or done:  
                        s_image_normalized_for_pull = s_["image"] / 255.0 if not done else None
                        push_and_pull(
                            self.opt, self.lnet, self.gnet, done, 
                            s_image_normalized_for_pull, s_, 
                            buffer_s, buffer_a, buffer_r, 
                            self.config.gamma
                        )
                        buffer_s, buffer_a, buffer_r = [], [], []

                        if done:
                            record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                            break
                            
                    s = s_
                    total_step += 1
                    
        except Exception as e:
            print(f"Worker {self.name} encountered an error: {e}")
            
        finally:
            
            self.res_queue.put(None) 


class TrainingConfig:
    
    def __init__(
        self, 
        env_name='MiniGrid-Empty-5x5-v0', 
        update_iter=5, 
        gamma=0.9, 
        max_ep=3000, 
        lr=1e-4, 
        betas=(0.92, 0.999),  
        num_workers=None,     
        render=True,          
        save_interval=500     
    ):
        self.env = Environment(env_name)
        self.update_iter = update_iter
        self.gamma = gamma
        self.max_ep = max_ep
        self.lr = lr
        self.betas = betas
        self.num_workers = num_workers if num_workers else mp.cpu_count()
        self.render = render
        self.save_interval = save_interval
        
        
        self.results_dir = "./results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.model_path = os.path.join(self.results_dir, f"a3c_{env_name.replace('-', '_')}_model.pth")
        self.plot_path = os.path.join(self.results_dir, f"a3c_{env_name.replace('-', '_')}_training_curve.png")


class A3CTrainer:
    
    
    def __init__(self, config):
        self.config = config
        
        
        self.gnet = Net(config.env.observation_space, config.env.action_space)
        self.gnet.share_memory()
        
        
        self.optimizer = SharedAdam(
            self.gnet.parameters(), 
            lr=config.lr, 
            betas=config.betas
        )
        
    def train(self):
        
        
        global_ep = mp.Value('i', 0)
        global_ep_r = mp.Value('d', 0.)
        res_queue = mp.Queue()

        
        num_workers = self.config.num_workers
        print(f"Starting training with {num_workers} workers")
        
        workers = [
            Worker(self.gnet, self.optimizer, global_ep, global_ep_r, res_queue, i, self.config) 
            for i in range(num_workers)
        ]
        
        [w.start() for w in workers]
        
        
        results = []
        active_workers = num_workers
        last_save = 0
        
        while active_workers > 0:
            r = res_queue.get()
            if r is not None:
                results.append(r)
                
                if len(results) - last_save >= self.config.save_interval:
                    self.save_model(checkpoint=True, ep=len(results))
                    last_save = len(results)
            else:
                active_workers -= 1
        
        [w.join() for w in workers]

        
        self.save_model()
        
        return results
    
    def save_model(self, checkpoint=False, ep=None):
        if checkpoint:
            save_path = self.config.model_path.replace('.pth', f'_checkpoint_{ep}.pth')
            print(f"Saving checkpoint at episode {ep}...")
        else:
            save_path = self.config.model_path
            print("Training complete. Saving global network...")
            
        torch.save(self.gnet.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, path=None):
        load_path = path if path else self.config.model_path
        if os.path.exists(load_path):
            self.gnet.load_state_dict(torch.load(load_path))
            print(f"Model loaded from {load_path}")
            return True
        else:
            print(f"No model found at {load_path}")
            return False


def plot_results(results, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(results)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.title('A3C Training Progress')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training curve saved to {save_path}")
        
    plt.show()


def main():
    
    default_config = {
        "env_name": 'MiniGrid-Empty-5x5-v0',
        "update_iter": 5,
        "gamma": 0.9,
        "max_ep": 3000,
        "lr": 1e-4,
        "beta1": 0.92,
        "beta2": 0.999,
        "num_workers": None,
        "render": True,
        "save_interval": 500
    }
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='A3C Training für MiniGrid-Umgebungen')
    
    # Add arguments for all hyperparameters
    parser.add_argument('--env_name', type=str, default=default_config['env_name'],
                        help=f'Name der Gymnasium-Umgebung (default: {default_config["env_name"]})')
    parser.add_argument('--update_iter', type=int, default=default_config['update_iter'],
                        help=f'Iterationen zwischen Netzwerk-Updates (default: {default_config["update_iter"]})')
    parser.add_argument('--gamma', type=float, default=default_config['gamma'],
                        help=f'Discount-Faktor für zukünftige Belohnungen (default: {default_config["gamma"]})')
    parser.add_argument('--max_ep', type=int, default=default_config['max_ep'],
                        help=f'Maximale Anzahl von Episoden (default: {default_config["max_ep"]})')
    parser.add_argument('--lr', type=float, default=default_config['lr'],
                        help=f'Lernrate für den Optimizer (default: {default_config["lr"]})')
    parser.add_argument('--beta1', type=float, default=default_config['beta1'],
                        help=f'Beta1 Parameter für den Adam-Optimizer (default: {default_config["beta1"]})')
    parser.add_argument('--beta2', type=float, default=default_config['beta2'],
                        help=f'Beta2 Parameter für den Adam-Optimizer (default: {default_config["beta2"]})')
    parser.add_argument('--num_workers', type=int, default=default_config['num_workers'],
                        help=f'Anzahl der Worker-Prozesse (default: Anzahl der CPU-Kerne)')
    parser.add_argument('--render', action='store_true', default=default_config['render'],
                        help=f'Umgebung rendern (default: {default_config["render"]})')
    parser.add_argument('--no_render', action='store_false', dest='render',
                        help=f'Umgebung nicht rendern')
    parser.add_argument('--save_interval', type=int, default=default_config['save_interval'],
                        help=f'Intervall zum Speichern von Checkpoints (default: {default_config["save_interval"]})')
    
   
    args = parser.parse_args()
    
    
    config = TrainingConfig(
        env_name=args.env_name,
        update_iter=args.update_iter,
        gamma=args.gamma,
        max_ep=args.max_ep,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        num_workers=args.num_workers,
        render=args.render,
        save_interval=args.save_interval
    )
    
    trainer = A3CTrainer(config)
    results = trainer.train()  
    
    plot_results(results, config.plot_path)


if __name__ == "__main__":
    main()