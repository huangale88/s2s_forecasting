"""DataModule and Dataset for S2S forecasting with dummy data."""

from typing import Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig


class DummyS2SDataset(Dataset):
    """Dummy dataset that generates random tensors for S2S forecasting.
    
    This dataset yields a dictionary with four keys:
        - era5_grids: shape [seq_len_past, num_era5_vars, spatial_size, spatial_size]
        - basin_attributes: shape [num_static_vars]
        - past_daymet: shape [seq_len_past, num_daymet_vars]
        - targets: shape [seq_len_future, num_daymet_vars]
    """
    
    def __init__(self, cfg: DictConfig, mode: str = "train"):
        """Initialize the dummy dataset.
        
        Args:
            cfg: Hydra config containing data dimensions.
            mode: Dataset mode (train/val/test). Not used for dummy data.
        """
        self.cfg = cfg
        self.mode = mode
        # Store dimensions from config
        self.seq_len_past = cfg.seq_len_past
        self.seq_len_future = cfg.seq_len_future
        self.spatial_size = cfg.spatial_size
        self.num_era5_vars = cfg.num_era5_vars
        self.num_daymet_vars = cfg.num_daymet_vars
        self.num_static_vars = cfg.num_static_vars
        
    def __len__(self) -> int:
        """Return a fixed number of samples for iteration."""
        return 100  # Arbitrary number of samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate a single sample of dummy data.
        
        Args:
            idx: Sample index (not used, data is randomly generated).
            
        Returns:
            Dictionary containing era5_grids, basin_attributes, past_daymet, and targets.
        """
        # Generate random tensors with correct shapes
        era5_grids = torch.randn(
            self.seq_len_past, 
            self.num_era5_vars, 
            self.spatial_size, 
            self.spatial_size
        )
        basin_attributes = torch.randn(self.num_static_vars)
        past_daymet = torch.randn(self.seq_len_past, self.num_daymet_vars)
        targets = torch.randn(self.seq_len_future, self.num_daymet_vars)
        
        return {
            "era5_grids": era5_grids,
            "basin_attributes": basin_attributes,
            "past_daymet": past_daymet,
            "targets": targets,
        }


def get_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from config.
    
    Args:
        cfg: Hydra config containing data parameters (batch_size, num_workers, etc.).
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = DummyS2SDataset(cfg, mode="train")
    val_dataset = DummyS2SDataset(cfg, mode="val")
    
    # Use 0 workers to avoid multiprocessing memory issues in constrained environments
    num_workers = min(cfg.num_workers, 0) if cfg.num_workers > 0 else 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    
    return train_loader, val_loader