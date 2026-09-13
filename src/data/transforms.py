"""
Augmentation and tensor conversion.
File location: <project_root>/src/data/transforms.py

Built on ``torchvision.transforms.v2``, which ships with PyTorch - no extra
install, no C++ compiler. (Albumentations does the same job; it is optional in
this project because its dependency chain breaks on Windows.)

Division of labour
------------------
``preprocess.py``  : geometry and colour work on the RAW photograph
                     (crop, square, CLAHE, resize). Output: uint8 300x300.
``transforms.py``  : everything that happens per-epoch on an already-300x300
                     image - random augmentation, uint8 -> float32, ImageNet
                     normalisation.

That split is what makes image caching possible: the expensive, deterministic
part runs once to disk; the cheap, random part runs every epoch.

Why the augmentations are mild
------------------------------
Diabetic retinopathy is graded from small, colour-coded features:
microaneurysms (tiny dark red dots), haemorrhages (dark red blots), hard
exudates (pale yellow). Aggressive colour augmentation destroys exactly the
signal the model needs, so hue is left untouched and brightness/contrast/
saturation move only slightly.

Geometry is a different story. A fundus photograph has no canonical
orientation - the camera can be rotated, and left and right eyes are mirror
images - so flips and modest rotation are free, label-preserving variety.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode


# ---------------------------------------------------------------------------
def build_transforms(cfg: Dict[str, Any], train: bool) -> v2.Compose:
    """Return the transform pipeline for the train or the eval split.

    Parameters
    ----------
    cfg:
        The project config (needs ``preprocessing.mean/std`` and, for training,
        the ``augmentation`` block).
    train:
        ``True``  -> random augmentation + normalisation (training only).
        ``False`` -> normalisation only. Use this for validation, test, AND
        inference in the Streamlit app. Augmenting at inference time is a
        classic bug: predictions become non-deterministic and the Grad-CAM
        heatmap no longer corresponds to the image the user uploaded.

    Notes
    -----
    Input is expected to be an already-preprocessed uint8 image of shape
    (size, size, 3) in **RGB** order - which is what ``APTOSDataset`` provides.
    """
    pp = cfg["preprocessing"]
    mean: Sequence[float] = pp["mean"]
    std: Sequence[float] = pp["std"]

    # Shared tail: uint8 HWC -> float32 CHW in [0,1] -> ImageNet-normalised.
    # ToDtype(scale=True) does the /255; doing it manually as well is a
    # surprisingly common double-scaling bug.
    tail = [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(mean), std=list(std)),
    ]

    if not train:
        return v2.Compose([v2.ToImage(), *tail])

    aug = cfg.get("augmentation", {})

    return v2.Compose([
        v2.ToImage(),                                    # -> uint8 CHW tensor

        # -- geometry: free variety, label-preserving -------------------------
        v2.RandomHorizontalFlip(p=float(aug.get("hflip_p", 0.5))),
        v2.RandomVerticalFlip(p=float(aug.get("vflip_p", 0.5))),
        v2.RandomAffine(
            degrees=float(aug.get("rotate_degrees", 15)),
            scale=(float(aug.get("scale_min", 0.9)), float(aug.get("scale_max", 1.0))),
            interpolation=InterpolationMode.BILINEAR,
            fill=0,      # black fill matches the fundus surround, so rotation
                         # corners look like real camera border, not artefacts
        ),

        # -- photometry: deliberately gentle ---------------------------------
        v2.ColorJitter(
            brightness=float(aug.get("brightness", 0.15)),
            contrast=float(aug.get("contrast", 0.15)),
            saturation=float(aug.get("saturation", 0.10)),
            hue=float(aug.get("hue", 0.0)),
        ),

        *tail,
    ])


# ---------------------------------------------------------------------------
def denormalize(tensor: torch.Tensor, cfg: Dict[str, Any]) -> np.ndarray:
    """Undo ImageNet normalisation so an augmented tensor can be displayed.

    Takes a (3, H, W) float tensor and returns an (H, W, 3) uint8 RGB array.
    Needed for the augmentation figure in the Milestone 2 notebook - without
    it, augmented samples plot as psychedelic noise and you cannot tell whether
    your pipeline is sane.
    """
    pp = cfg["preprocessing"]
    mean = torch.tensor(pp["mean"], dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(pp["std"], dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)

    out = (tensor.detach() * std + mean).clamp(0.0, 1.0)
    return (out.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)


def describe_transforms(pipeline: v2.Compose) -> str:
    """One-line-per-step summary, for pasting into the report's Methodology."""
    lines = []
    for i, step in enumerate(pipeline.transforms, start=1):
        lines.append(f"  {i}. {step.__class__.__name__}: {step}")
    return "\n".join(lines)
