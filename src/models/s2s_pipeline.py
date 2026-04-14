"""Dummy Linear S2S Model for baseline pipeline testing."""

import torch
import torch.nn as nn
from omegaconf import DictConfig


class DummyLinearS2S(nn.Module):
    """A simple linear model for S2S forecasting baseline.
    
    This model flattens all input tensors (ERA5 grids, basin attributes, and past Daymet),
    concatenates them, and passes through a simple MLP to produce forecasts.
    """
    
    def __init__(self, cfg: DictConfig, data_cfg: DictConfig):
        """Initialize the DummyLinearS2S model.
        
        Args:
            cfg: Hydra config containing model dimensions (hidden_dim, etc.).
            data_cfg: Hydra config containing data dimensions for computing input/output sizes.
        """
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        
        # Extract data dimensions from config
        seq_len_past = data_cfg.seq_len_past
        spatial_size = data_cfg.spatial_size
        num_era5_vars = data_cfg.num_era5_vars
        num_static_vars = data_cfg.num_static_vars
        num_daymet_vars = data_cfg.num_daymet_vars
        seq_len_future = data_cfg.seq_len_future
        
        # Calculate flattened input dimension
        # ERA5: seq_len_past * num_era5_vars * spatial_size * spatial_size
        era5_flat_dim = seq_len_past * num_era5_vars * spatial_size * spatial_size
        # Basin attributes: num_static_vars
        basin_flat_dim = num_static_vars
        # Past Daymet: seq_len_past * num_daymet_vars
        daymet_flat_dim = seq_len_past * num_daymet_vars
        
        input_dim = era5_flat_dim + basin_flat_dim + daymet_flat_dim
        output_dim = seq_len_future * num_daymet_vars
        
        # Define the MLP
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, output_dim),
        )
        
        # Store dimensions for reshaping
        self.seq_len_future = seq_len_future
        self.num_daymet_vars = num_daymet_vars
        
    def forward(
        self,
        era5_grids: torch.Tensor,
        basin_attributes: torch.Tensor,
        past_daymet: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            era5_grids: ERA5 data with shape [batch, seq_len_past, num_era5_vars, spatial_size, spatial_size]
            basin_attributes: Static basin attributes with shape [batch, num_static_vars]
            past_daymet: Past Daymet data with shape [batch, seq_len_past, num_daymet_vars]
            
        Returns:
            Predictions with shape [batch, seq_len_future, num_daymet_vars]
        """
        batch_size = era5_grids.shape[0]
        
        # Flatten inputs
        era5_flat = era5_grids.view(batch_size, -1)  # [batch, seq_len_past * num_era5_vars * spatial_size^2]
        basin_flat = basin_attributes.view(batch_size, -1)  # [batch, num_static_vars]
        daymet_flat = past_daymet.view(batch_size, -1)  # [batch, seq_len_past * num_daymet_vars]
        
        # Concatenate along dim=1
        concatenated = torch.cat([era5_flat, basin_flat, daymet_flat], dim=1)
        
        # Pass through MLP
        output = self.model(concatenated)
        
        # Reshape to [batch, seq_len_future, num_daymet_vars]
        output = output.view(batch_size, self.seq_len_future, self.num_daymet_vars)
        
        return output