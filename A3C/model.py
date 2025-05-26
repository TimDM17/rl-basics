import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MiniGridACModel(nn.Module):
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