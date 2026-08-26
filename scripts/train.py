import sys
from pathlib import Path

# Add parent directory to path for clean imports from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig

from src.data.datamodule import get_dataloaders
from src.models.s2s_pipeline import DummyLinearS2S
from src.engine import train_epoch, validate_epoch


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    if cfg.training.device == "cuda" and device.type == "cpu":
        print("Warning: CUDA requested but unavailable. Falling back to CPU.")
    
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_dataloaders(cfg.data, seed=cfg.seed)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    model = DummyLinearS2S(cfg.model, cfg.data).to(device)
    print(f"Model: {cfg.model.name}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.MSELoss()
    
    print(f"Starting training for {cfg.training.epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(cfg.training.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    print("-" * 60)
    print("Training complete!")

if __name__ == "__main__":
    main()