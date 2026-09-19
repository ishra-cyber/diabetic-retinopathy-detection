"""
Model construction.
File location: <project_root>/src/models/factory.py

All three architectures come from ``timm`` with ImageNet weights. ``timm``
replaces the classifier head for us when ``num_classes`` is passed, so there is
no manual surgery to get wrong.

Why transfer learning rather than training from scratch: 2,562 training images
is nowhere near enough to learn low-level visual filters. ImageNet features
(edges, textures, blob and vessel-like structures) transfer well to fundus
photography, so we only need to learn the mapping from those features to the
five severity grades.

Two head types
--------------
``classification``  five outputs, softmax, cross-entropy. The Milestone 4-6
                    setup.
``ordinal``         ONE output, a real number on the 0-4 severity scale, cut
                    into grades by thresholds fitted on validation (see
                    ``src.models.thresholds`` for why).

The backbone is identical either way - only the final linear layer's width
changes - so a run of each is a clean comparison of the head, not of two
different models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

# The three architectures being compared in Milestone 6.
SUPPORTED_BACKBONES = {
    "resnet50": "Residual connections; the standard CNN baseline (~25.6M params)",
    "efficientnet_b3": "Compound depth/width/resolution scaling (~12M params)",
    "densenet121": "Dense connectivity, heavy feature reuse (~8M params)",
}


def build_model(
    backbone: str,
    num_classes: int = 5,
    pretrained: bool = True,
    drop_rate: float = 0.0,
) -> nn.Module:
    """Create a backbone with a fresh ``num_classes``-way head.

    Raises
    ------
    ImportError
        If timm is not installed.
    ValueError
        If the backbone name is unknown to timm (the message lists close matches).
    """
    try:
        import timm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "timm is required. Install it with:  pip install timm"
        ) from exc

    if backbone not in SUPPORTED_BACKBONES:
        print(f"[factory] NOTE: '{backbone}' is not one of the three project "
              f"architectures {sorted(SUPPORTED_BACKBONES)}. Continuing anyway.")

    try:
        model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=float(drop_rate),
        )
    except RuntimeError as exc:
        raise ValueError(
            f"timm could not create '{backbone}'.\n"
            f"If this mentions a download failure, you are offline - pretrained "
            f"weights are fetched on first use and then cached in "
            f"~/.cache/huggingface.\nOriginal error: {exc}"
        ) from exc

    return model


# ---------------------------------------------------------------------------
HEAD_TYPES = ("classification", "ordinal")


def head_type(cfg: Dict[str, Any]) -> str:
    """Which head this config asks for. Defaults to the original behaviour."""
    head = str(cfg.get("training", {}).get("head", "classification")).lower()
    if head not in HEAD_TYPES:
        raise ValueError(f"training.head must be one of {list(HEAD_TYPES)}, got '{head}'")
    return head


def num_outputs(cfg: Dict[str, Any]) -> int:
    """Width of the final layer: five logits, or one regression score."""
    return 1 if head_type(cfg) == "ordinal" else int(cfg["dataset"]["num_classes"])


def build_from_config(cfg: Dict[str, Any]) -> nn.Module:
    """Build the model described by ``cfg['training']``."""
    tr = cfg["training"]
    return build_model(
        backbone=tr["backbone"],
        num_classes=num_outputs(cfg),
        pretrained=bool(tr.get("pretrained", True)),
        drop_rate=float(tr.get("drop_rate", 0.0)),
    )


# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total parameters, trainable parameters)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def last_conv_layer(model: nn.Module) -> nn.Module:
    """Return the last Conv2d module in the network.

    Used as the Grad-CAM target in Milestone 7. Walking the module tree and
    taking the final convolution works for all three of our architectures and
    does not depend on their internal attribute names, which differ
    (``layer4`` / ``conv_head`` / ``features``) and change between timm versions.
    """
    candidates: List[nn.Module] = [
        m for m in model.modules() if isinstance(m, nn.Conv2d)
    ]
    if not candidates:
        raise ValueError(
            "No Conv2d layer found - Grad-CAM as implemented needs a CNN. "
            "A vision transformer would require a different attribution method."
        )
    return candidates[-1]


def describe_model(model: nn.Module, backbone: str) -> str:
    """Short summary block for the training log and the report."""
    total, trainable = count_parameters(model)
    note = SUPPORTED_BACKBONES.get(backbone, "")
    lines = [
        f"  backbone   : {backbone}",
        f"  parameters : {total:,} total | {trainable:,} trainable",
    ]
    if note:
        lines.append(f"  note       : {note}")
    return "\n".join(lines)


@torch.no_grad()
def check_forward(model: nn.Module, size: int = 300, num_classes: int = 5,
                  device: str = "cpu", expected_outputs: int | None = None
                  ) -> Tuple[int, ...]:
    """Push one dummy batch through the model to validate the output shape.

    Cheap insurance: catches a wrong head or a size mismatch in a second,
    instead of at the end of the first epoch.
    """
    width = int(expected_outputs if expected_outputs is not None else num_classes)
    model = model.to(device).eval()
    dummy = torch.zeros(2, 3, size, size, device=device)
    out = model(dummy)
    if tuple(out.shape) != (2, width):
        raise ValueError(
            f"Model output shape {tuple(out.shape)} != expected (2, {width}). "
            "Check the head width was passed to timm.create_model "
            "(one output for the ordinal head, num_classes for classification)."
        )
    return tuple(out.shape)
