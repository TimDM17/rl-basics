import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNCNN(nn.Module):
    def __init__(self, input_shape, action_space):
        super().__init__()

        # Define image embedding layers
        self.image_conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        # Calculate embedding size
        n = input_shape[1]
        m = input_shape[2]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        self.actor = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

    def forward(self, x):
        # x = obs["image"].transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        embedding = x
        x = self.actor(embedding)
        
        return x
