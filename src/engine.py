"""Pure PyTorch training and validation engine for S2S forecasting."""

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    device: torch.device,
) -> float:
    """Train the model for one epoch.
    
    Args:
        model: The PyTorch model to train.
        dataloader: DataLoader containing training data.
        optimizer: Optimizer for updating model parameters.
        criterion: Loss function (e.g., nn.MSELoss).
        device: Device to run computation on (cuda/cpu/mps).
        
    Returns:
        Average loss over the epoch as a float.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Unpack batch dictionary
        era5_grids = batch["era5_grids"].to(device)
        basin_attributes = batch["basin_attributes"].to(device)
        past_daymet = batch["past_daymet"].to(device)
        targets = batch["targets"].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(era5_grids, basin_attributes, past_daymet)
        
        # Compute loss
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    device: torch.device,
) -> float:
    """Validate the model for one epoch.
    
    Args:
        model: The PyTorch model to validate.
        dataloader: DataLoader containing validation data.
        criterion: Loss function (e.g., nn.MSELoss).
        device: Device to run computation on (cuda/cpu/mps).
        
    Returns:
        Average validation loss over the epoch as a float.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Validating")
    with torch.no_grad():
        for batch in pbar:
            # Unpack batch dictionary
            era5_grids = batch["era5_grids"].to(device)
            basin_attributes = batch["basin_attributes"].to(device)
            past_daymet = batch["past_daymet"].to(device)
            targets = batch["targets"].to(device)
            
            # Forward pass
            predictions = model(era5_grids, basin_attributes, past_daymet)
            
            # Compute loss
            loss = criterion(predictions, targets)
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / num_batches
    return avg_loss