import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gymnasium as gym
import minigrid
import os
os.environ["OMP_NUM_THREADS"] = "1"



UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000


env = gym.make('MiniGrid-MultiRoom-N2-S4-v0')
N_S = env.observation_space
N_A = env.action_space


class Net(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()

        # Define image embedding layers
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        # Calculate embedding size 
        n = obs_space["image"].shape[0]
        m = obs_space["image"].shape[1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64


        # Actor
        self.actor = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        x = obs["image"].transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)  

        embedding = x

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value
    
    def choose_action(self, s):
        self.eval()
        dist, _ = self.forward(s)
        return dist.sample().numpy()[0]
    
    def loss_func(self, s, a, v_t):
        self.train()
        dist, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        exp_v = dist.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)
        self.env = gym.make('MiniGrid-Empty-5x5-v0').unwrapped

    def run(self):
        total_step = 1
        try:
            while self.g_ep.value < MAX_EP:
                s, _ = self.env.reset()  
                buffer_s, buffer_a, buffer_r = [], [], []
                ep_r = 0.
                while True:
                    if self.name == 'w00':
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

                    if total_step % UPDATE_GLOBAL_ITER == 0 or done:  
                        # s_ (next state) also needs normalization before being passed to push_and_pull
                        s_image_normalized_for_pull = s_["image"] / 255.0 if not done else None
                        push_and_pull(self.opt, self.lnet, self.gnet, done, s_image_normalized_for_pull, s_, buffer_s, buffer_a, buffer_r, GAMMA)
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


if __name__ == "__main__":
    gnet = Net(N_S, N_A)        
    gnet.share_memory()         
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    num_workers = mp.cpu_count()
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(num_workers)]
    [w.start() for w in workers]
    
    res = []                    
    active_workers = num_workers
    while active_workers > 0:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            active_workers -= 1
    
    [w.join() for w in workers]


    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
