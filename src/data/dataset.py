"""
PyTorch Dataset and DataLoader construction.
File location: <project_root>/src/data/dataset.py

Reads the split CSVs produced by Milestone 3 and serves tensors to the training
loop. Prefers the cached 300x300 images from Milestone 2; falls back to
preprocessing the raw image on the fly if a cached file is missing, so a partial
cache degrades in speed rather than crashing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.preprocess import cache_dir_for, preprocess_from_config, read_image
from src.data.transforms import build_transforms
from src.utils.config import get_path
from src.utils.seed import seed_worker


class APTOSDataset(Dataset):
    """Fundus images + severity labels.

    Returns ``(image_tensor, label)``. Image ids are available as
    ``dataset.ids`` in the same order, so evaluation code can align predictions
    to filenames by using ``shuffle=False``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Dict[str, Any],
        train: bool,
        use_cache: bool = True,
    ) -> None:
        ds = cfg["dataset"]
        self.cfg = cfg
        self.id_col = ds["id_col"]
        self.label_col = ds["label_col"]
        self.ext = ds["image_ext"]
        self.transform = build_transforms(cfg, train=train)

        self.ids: List[str] = df[self.id_col].astype(str).tolist()
        self.labels: List[int] = df[self.label_col].astype(int).tolist()

        raw_dir = get_path(cfg, "aptos_dir") / ds["train_images"]
        cache_dir = cache_dir_for(cfg, get_path(cfg, "interim_dir"))

        # Resolve every path once, here, rather than calling is_file() 3,662
        # times per epoch.
        self.paths: List[Path] = []
        self.from_cache: List[bool] = []
        for image_id in self.ids:
            cached = cache_dir / f"{image_id}{self.ext}"
            if use_cache and cached.is_file():
                self.paths.append(cached)
                self.from_cache.append(True)
            else:
                self.paths.append(raw_dir / f"{image_id}{self.ext}")
                self.from_cache.append(False)

        self.n_cached = int(sum(self.from_cache))
        if self.n_cached == 0 and len(self) > 0:
            # Not fatal, but the user should know why epochs are slow.
            print(f"[APTOSDataset] WARNING: no cached images found in {cache_dir}. "
                  "Preprocessing on the fly - expect much slower epochs. "
                  "Run: python -m scripts.m2_cache_images --workers 4")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = self.paths[index]
        label = self.labels[index]

        try:
            image = read_image(path)
        except ValueError as exc:
            raise RuntimeError(
                f"Failed to read image for id '{self.ids[index]}' at {path}: {exc}"
            ) from exc

        # Cached images are already preprocessed; raw ones are not.
        if not self.from_cache[index]:
            image = preprocess_from_config(image, self.cfg)

        # OpenCV gives BGR; the model and ImageNet statistics expect RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return self.transform(image), label

    # -- helpers ----------------------------------------------------------
    def class_counts(self, num_classes: int) -> np.ndarray:
        return np.bincount(np.asarray(self.labels, dtype=int), minlength=num_classes)


# ---------------------------------------------------------------------------
def make_weighted_sampler(
    labels: List[int],
    num_classes: int,
    seed: int = 42,
) -> WeightedRandomSampler:
    """Oversample rare classes so each batch is roughly class-balanced.

    Each sample's draw probability is 1/count(its class), so in expectation the
    sampler draws every class equally often. Note this means rare images are
    seen many times per epoch and common ones sometimes not at all - which is
    why it is an alternative to weighted loss, not usually a partner for it
    (doing both double-counts the correction).
    """
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0                      # avoid divide-by-zero
    per_class_weight = 1.0 / counts
    sample_weights = per_class_weight[np.asarray(labels, dtype=int)]

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )


def build_loader(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    train: bool,
    batch_size: Optional[int] = None,
    sampler: Optional[WeightedRandomSampler] = None,
    shuffle: Optional[bool] = None,
) -> Tuple[DataLoader, APTOSDataset]:
    """Create a DataLoader (and return the dataset alongside it).

    Windows notes
    -------------
    * ``num_workers`` spawns real processes, so any script using this MUST be
      guarded by ``if __name__ == "__main__":``. All scripts here are.
    * If loading hangs at the start of the first epoch, set
      ``training.num_workers: 0`` in the config. It is slower but always works.
    """
    tr = cfg["training"]
    dataset = APTOSDataset(df, cfg, train=train)

    num_workers = int(tr.get("num_workers", 2))
    if shuffle is None:
        shuffle = train and sampler is None

    generator = torch.Generator()
    generator.manual_seed(cfg["project"]["seed"])

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size or tr["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,                # keeps BatchNorm happy on a ragged tail
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    return loader, dataset
