import torch
import torch.optim as optim
from typing import Dict, Tuple
from model import NetV1

def load_from_checkpoint(path: str, device = torch.device):
    cp = torch.load(path)

    model_config = cp['model_config']
    model = NetV1(model_config).to(device)
    model.load_state_dict(cp['model_state_dict'])
    
    optimizer = optim.SGD(model.parameters(), lr= 0.01, momentum= 0.9)
    optimizer.load_state_dict(cp['optimizer_state_dict'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    losses = cp['losses']

    epoch = cp['epoch']

    return model, optimizer, scheduler, model_config, losses, epoch

    
