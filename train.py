from model import NetV1, TrainingData
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from utils import load_from_checkpoint

#setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {'MPS' if torch.backends.mps.is_available() else 'CPU'}")

dataset = TrainingData('datasets/v1mini_25000.npz')
dataloader = DataLoader(dataset, batch_size= 16, shuffle=True)

model_config = {"input_channels": 10, 
                "tower_channels": 16,
                "kernel_size": 5,
                "padding": 2,
                "stride": 1,
                "tower_size": 4,
                "policy_channels": 16}

criterion_policy = nn.CrossEntropyLoss()
criterion_value = nn.MSELoss()

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

#loading from latest checkpoint
checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]

if checkpoints:
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    model, optimizer, scheduler, model_config, losses, start_epoch = load_from_checkpoint(checkpoint_path, device)
    print(f"Resuming from epoch {start_epoch}")
else:
    losses = []
    model = NetV1(model_config)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr= 1e-2)
    start_epoch = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode= 'min', min_lr=1e-5)
    print("No checkpoint found. Starting from scratch.")

num_epochs = 1000
for epoch in range(start_epoch, num_epochs):
    model.train() 
    total_loss = 0
    
    #tqdm for nice bar in terminal
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
    
    for batch in progress_bar:
        x, p, v = batch
        x, p, v = torch.tensor(x, dtype=torch.float32).to(device), \
                  torch.tensor(p, dtype=torch.float32).to(device), \
                  torch.tensor(v, dtype=torch.float32).to(device)
        
        p_hat, v_hat = model(x)
        
        # CrossEntropyLoss expects class indices
        p_targets = torch.argmax(p, dim=1).to(device)
        
        loss_policy = criterion_policy(p_hat, p_targets)
        loss_value = criterion_value(v_hat, v)
        loss = loss_policy + loss_value
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar with loss
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
    
    scheduler.step()
    mean_epoch_loss = total_loss/len(dataloader)
    losses.append(mean_epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {mean_epoch_loss:.4f}')

    checkpoint_path = os.path.join(checkpoint_dir, f'V1_checkpoint_epoch_{epoch+1}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler,
        'model_config': model_config,
        'losses': losses,
        'epoch': epoch + 1,
    }, checkpoint_path)
    print(losses)

print('Training complete')