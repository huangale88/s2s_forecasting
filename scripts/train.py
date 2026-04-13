"""Training script for S2S forecasting model using Hydra configuration."""

import sys
import os
from pathlib import Path

# Add parent directory to path for clean imports from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from omegaconf import DictConfig
import hydra

from src.data.datamodule import get_dataloaders
from src.models.s2s_pipeline import DummyLinearS2S
from src.engine import train_epoch, validate_epoch


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training loop.
    
    Args:
        cfg: Hydra configuration dictionary containing all hyperparameters.
    """
    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    
    # Determine device (cuda > mps > cpu)
    if cfg.training.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif cfg.training.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Instantiate dataloaders
    train_loader, val_loader = get_dataloaders(cfg.data)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Instantiate model
    model = DummyLinearS2S(cfg.model, cfg.data)
    model = model.to(device)
    print(f"Model: {cfg.model.name}")
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.MSELoss()
    
    print(f"Starting training for {cfg.training.epochs} epochs...")
    print("-" * 60)
    
    # Training loop
    for epoch in range(cfg.training.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")
        
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        
        # Validate
        val_loss = validate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    print("-" * 60)
    print("Training complete!")


if __name__ == "__main__":
    main()