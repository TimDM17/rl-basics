import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64,      kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64,      kernel_size=3, stride=1, padding=1)
        
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self._forward_conv(dummy)
            self.flat_dim = conv_out.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flat_dim, 512)
        self.out = nn.Linear(512, num_actions)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.out(x)


