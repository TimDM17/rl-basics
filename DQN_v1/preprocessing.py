import torch
import numpy as np

def preprocess_obs(obs):
    
    img = obs['image']
    img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)
    
    return img_tensor