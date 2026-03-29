"""Dataset for loading generated handwriting image → stroke trajectory pairs."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class StrokeDataset(Dataset):
    """Loads the dataset.npz produced by the data generation notebook.

    Each sample is a dict with:
        image:      (3, H, W) float32 tensor, normalized to [0, 1]
        trajectory: (max_seq_len, 3) float32 tensor of (dx, dy, pen_state) deltas
        length:     int, actual trajectory length before padding
        word:       str
    """

    def __init__(self, npz_path: str, max_samples: int | None = None):
        data = np.load(npz_path, allow_pickle=True)
        n = len(data["images"]) if max_samples is None else min(max_samples, len(data["images"]))
        self.images = data["images"][:n]          # (N, H, W, 3) uint8
        self.trajectories = data["trajectories"][:n]  # (N, T, 3) float32
        self.lengths = data["lengths"][:n]         # (N,) int
        self.words = data["words"][:n]             # (N,) str

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # HWC uint8 → CHW float32 [0, 1]
        img = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float() / 255.0
        traj = torch.from_numpy(self.trajectories[idx]).float()
        length = int(self.lengths[idx])
        return {
            "image": img,
            "trajectory": traj,
            "length": length,
            "word": str(self.words[idx]),
        }


def collate_fn(batch):
    """Custom collate that stacks everything and keeps words as a list."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "trajectory": torch.stack([b["trajectory"] for b in batch]),
        "length": torch.tensor([b["length"] for b in batch], dtype=torch.long),
        "word": [b["word"] for b in batch],
    }


def create_dataloaders(
    npz_path: str,
    batch_size: int = 32,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 0,
    max_samples: int | None = None,
    seed: int = 42,
):
    """Create train/val/test dataloaders from a single npz file.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    dataset = StrokeDataset(npz_path, max_samples=max_samples)
    n = len(dataset)
    n_test = max(1, int(n * test_split))
    n_val = max(1, int(n * val_split))
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=generator,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader
