import torch
import torch.nn as nn
from omegaconf import DictConfig


class DummyLinearS2S(nn.Module):
    """Baseline linear MLP model for S2S forecasting."""
    
    def __init__(self, cfg: DictConfig, data_cfg: DictConfig):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.seq_len_future = data_cfg.seq_len_future
        self.num_daymet_vars = data_cfg.num_daymet_vars
        
        # Compute total input dimension for the MLP
        era5_dim = data_cfg.seq_len_past * data_cfg.num_era5_vars * (data_cfg.spatial_size ** 2)
        basin_dim = data_cfg.num_static_vars
        daymet_dim = data_cfg.seq_len_past * data_cfg.num_daymet_vars
        
        input_dim = era5_dim + basin_dim + daymet_dim
        output_dim = self.seq_len_future * self.num_daymet_vars
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, output_dim),
        )
        
    def forward(
        self,
        era5_grids: torch.Tensor,
        basin_attributes: torch.Tensor,
        past_daymet: torch.Tensor,
    ) -> torch.Tensor:
        
        # Flatten everything except the batch dimension
        x = torch.cat([
            torch.flatten(era5_grids, start_dim=1),
            torch.flatten(basin_attributes, start_dim=1),
            torch.flatten(past_daymet, start_dim=1)
        ], dim=1)
        
        out = self.net(x)
        
        # Reshape to target sequence format
        return out.view(-1, self.seq_len_future, self.num_daymet_vars)