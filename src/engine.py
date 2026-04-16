import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Runs one epoch of training."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        era5 = batch["era5_grids"].to(device)
        basin = batch["basin_attributes"].to(device)
        daymet = batch["past_daymet"].to(device)
        targets = batch["targets"].to(device)
        
        optimizer.zero_grad()
        preds = model(era5, basin, daymet)
        loss = criterion(preds, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / len(dataloader)


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Runs one epoch of validation."""
    model.eval()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Validating")
    with torch.no_grad():
        for batch in pbar:
            era5 = batch["era5_grids"].to(device)
            basin = batch["basin_attributes"].to(device)
            daymet = batch["past_daymet"].to(device)
            targets = batch["targets"].to(device)
            
            preds = model(era5, basin, daymet)
            loss = criterion(preds, targets)
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    return total_loss / len(dataloader)