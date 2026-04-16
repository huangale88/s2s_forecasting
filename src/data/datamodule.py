import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig


class DummyS2SDataset(Dataset):
    """Generates random noise tensors matching expected S2S data dimensions."""

    def __init__(self, cfg: DictConfig, mode: str = "train"):
        self.cfg = cfg
        self.mode = mode

    def __len__(self) -> int:
        return 100

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "era5_grids": torch.randn(
                self.cfg.seq_len_past,
                self.cfg.num_era5_vars,
                self.cfg.spatial_size,
                self.cfg.spatial_size
            ),
            "basin_attributes": torch.randn(self.cfg.num_static_vars),
            "past_daymet": torch.randn(self.cfg.seq_len_past, self.cfg.num_daymet_vars),
            "targets": torch.randn(self.cfg.seq_len_future, self.cfg.num_daymet_vars),
        }


def get_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    train_dataset = DummyS2SDataset(cfg, mode="train")
    val_dataset = DummyS2SDataset(cfg, mode="val")
    
    print(f"Config num_workers: {cfg.num_workers} | Actual DataLoader num_workers: {cfg.num_workers}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    return train_loader, val_loader