"""Tests for S2S dataloaders and dataset."""

import pytest
import torch
from omegaconf import OmegaConf, DictConfig

from src.data.datamodule import DummyS2SDataset


@pytest.fixture
def dummy_cfg() -> DictConfig:
    """Create a mock config for testing."""
    return OmegaConf.create({
        "batch_size": 16,
        "num_workers": 2,
        "seq_len_past": 14,
        "seq_len_future": 45,
        "spatial_size": 32,
        "num_era5_vars": 5,
        "num_daymet_vars": 4,
        "num_static_vars": 5,
    })


def test_dummy_dataset_shapes(dummy_cfg: DictConfig) -> None:
    """Test that DummyS2SDataset returns tensors with correct shapes."""
    dataset = DummyS2SDataset(dummy_cfg, mode="train")
    
    # Get a single sample
    sample = dataset[0]
    
    # Check that all expected keys are present
    assert "era5_grids" in sample
    assert "basin_attributes" in sample
    assert "past_daymet" in sample
    assert "targets" in sample
    
    # Check era5_grids shape: [seq_len_past, num_era5_vars, spatial_size, spatial_size]
    expected_era5_shape = (
        dummy_cfg.seq_len_past,
        dummy_cfg.num_era5_vars,
        dummy_cfg.spatial_size,
        dummy_cfg.spatial_size,
    )
    assert sample["era5_grids"].shape == expected_era5_shape, \
        f"Expected era5_grids shape {expected_era5_shape}, got {sample['era5_grids'].shape}"
    
    # Check basin_attributes shape: [num_static_vars]
    expected_basin_shape = (dummy_cfg.num_static_vars,)
    assert sample["basin_attributes"].shape == expected_basin_shape, \
        f"Expected basin_attributes shape {expected_basin_shape}, got {sample['basin_attributes'].shape}"
    
    # Check past_daymet shape: [seq_len_past, num_daymet_vars]
    expected_daymet_shape = (dummy_cfg.seq_len_past, dummy_cfg.num_daymet_vars)
    assert sample["past_daymet"].shape == expected_daymet_shape, \
        f"Expected past_daymet shape {expected_daymet_shape}, got {sample['past_daymet'].shape}"
    
    # Check targets shape: [seq_len_future, num_daymet_vars]
    expected_targets_shape = (dummy_cfg.seq_len_future, dummy_cfg.num_daymet_vars)
    assert sample["targets"].shape == expected_targets_shape, \
        f"Expected targets shape {expected_targets_shape}, got {sample['targets'].shape}"
    
    # Check that all tensors are float tensors
    assert sample["era5_grids"].dtype == torch.float32
    assert sample["basin_attributes"].dtype == torch.float32
    assert sample["past_daymet"].dtype == torch.float32
    assert sample["targets"].dtype == torch.float32