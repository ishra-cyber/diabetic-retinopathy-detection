"""
Reproducibility helpers.

Academic projects live or die on "can you run it again and get the same
number?". Call :func:`set_seed` at the top of *every* script that involves
randomness (splitting, augmentation, training, simulated histories).

Note: full bit-exact determinism on GPU costs speed. ``deterministic=True``
turns on cuDNN deterministic algorithms, which you should use for the final
run you report in your paper; leave it off while experimenting.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch (if installed). Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # torch is optional for pure-EDA scripts
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch >= 1.8: raises if a non-deterministic op is used.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # older torch, or op without a deterministic impl
            pass
    else:
        torch.backends.cudnn.benchmark = True

    return seed


def seed_worker(worker_id: int) -> None:  # pragma: no cover - used by DataLoader
    """DataLoader ``worker_init_fn`` so each worker is seeded reproducibly."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
