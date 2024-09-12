import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, Tuple

class TrainingData(Dataset):
    def __init__(self, data_file: str):
        dat = np.load(data_file)
        self.X = dat['arr_0']
        self.P = dat['arr_1']
        self.V = dat['arr_2']

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.P[idx], self.V[idx])


class ResBlock(nn.Module):
    def __init__(self, channels= 32, kernel_size= 5, stride= 1, padding= 2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = channels,
                               out_channels = channels,
                               kernel_size= kernel_size, 
                               stride= stride, 
                               padding= padding)

        self.conv2 = nn.Conv2d(in_channels = channels,
                               out_channels = channels,
                               kernel_size= kernel_size, 
                               stride= stride, 
                               padding= padding)
        
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
            
    def forward(self, x):
        residual = x
        first = F.relu(self.bn1(self.conv1(x)))
        second = self.bn2(self.conv2(first))
        out = F.relu(second + residual)
        return out
    

class PolicyHead(nn.Module):
    def __init__(self, in_channels, out_channels= 16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride= 1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_channels * 8 * 8, 1968)

    def forward(self, x):
        conv_out = F.relu(self.bn(self.conv(x)))
        lin_out = self.fc(self.flatten(conv_out))
        return lin_out #output the logits directly, no normalization


class ValueHead(nn.Module):
    def __init__(self, in_channels= 32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding= 0)
        self.bn = nn.BatchNorm2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8, 32)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = F.relu(self.fc1(self.flatten(out)))
        out = F.tanh(self.fc2(out))
        return out

""""NetV1 config keys:
    {
    input_channels: number of channels in
    tower_channels: in and out channels for residual block, 
    kernel size: for residual block,
    stride: for residual block,
    padding: for residual block,
    tower_size: how many residual blocks to stack,
    policy_channels: output channels for policy head convolution,
    }
"""

class NetV1(nn.Module):
    def __init__(self, cfg: Dict[str, int]):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels = cfg['input_channels'], 
            out_channels = cfg['tower_channels'], 
            kernel_size = 3, 
            stride = 1, 
            padding = 1
            )
        self.bn = nn.BatchNorm2d(cfg['tower_channels'])
        self.tower = nn.Sequential(
            *[ResBlock(cfg['tower_channels'], cfg['kernel_size'], cfg['stride'], cfg['padding']) \
            for _ in range(cfg['tower_size'])]
            )
        self.policy_head = PolicyHead(cfg['tower_channels'], cfg['policy_channels'])
        self.value_head = ValueHead(cfg['tower_channels'])
        

    def forward(self, x):
        h1 = F.relu(self.bn(self.conv(x)))
        h_tower = self.tower(h1)
        p_hat = self.policy_head(h_tower)
        v_hat = self.value_head(h_tower)
        return p_hat, v_hat.squeeze()
    