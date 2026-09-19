"""
Test-set evaluation and report figures.
File location: <project_root>/src/models/evaluate.py

The one rule
------------
The test split is evaluated **once**, after all model selection is finished.
Every choice up to this point - which epoch to checkpoint, which architecture
to prefer, which hyper-parameters to use - was made on the validation split.
If you evaluate on test, tune something, and evaluate again, your reported
number is optimistically biased and you would have to say so in your report.

What this produces per run
--------------------------
    metrics_test.json     the numbers that go in your results table
    test_predictions.npz  raw predictions, for any figure you want later

With ``--tta`` the same two artefacts are written under ``*_tta*`` names, so a
test-time-augmented evaluation never overwrites the plain one and the report can
show both rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.data.dataset import build_loader
from src.data.split import load_split
from src.models.factory import build_model
from src.models.metrics import compute_all_metrics
from src.models.thresholds import apply_thresholds, scores_to_probabilities
from src.utils.config import class_names, get_path


# ---------------------------------------------------------------------------
def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    num_classes: int = 5,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Rebuild the architecture recorded in the checkpoint and load its weights.

    The checkpoint stores the backbone name and preprocessing settings, so a
    model can be restored months later without guessing how it was built - and
    so the Streamlit app in Milestone 10 preprocesses exactly the way training
    did.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Has that experiment finished training?"
        )

    # weights_only=False: our checkpoint holds config metadata, not just tensors.
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:                      # torch < 2.0 has no weights_only
        ckpt = torch.load(checkpoint_path, map_location=device)

    backbone = ckpt.get("backbone")
    if backbone is None:
        raise ValueError(f"{checkpoint_path} has no 'backbone' field - not one of ours.")

    # pretrained=False: we are about to overwrite every weight anyway, and this
    # avoids a pointless download.
    # num_outputs is what the final layer actually had: 5 for a classification
    # head, 1 for an ordinal one. Older checkpoints predate the field and were
    # all classification, so num_classes is the right fallback.
    width = int(ckpt.get("num_outputs", ckpt.get("num_classes", num_classes)))
    model = build_model(backbone, num_classes=width, pretrained=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"[load_checkpoint] WARNING missing={list(missing)[:5]} "
              f"unexpected={list(unexpected)[:5]}")

    model.to(device).eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Test-time augmentation
# ---------------------------------------------------------------------------
# A fundus photograph has no canonical orientation: the camera can be rotated,
# and a left eye is very nearly the mirror image of a right eye. Training
# exploited that with RandomHorizontalFlip(p=0.5) and RandomVerticalFlip(p=0.5),
# so the network has already been taught that these four views carry the same
# label.
#
# TTA turns that invariance into free accuracy: run all four views, average the
# softmax outputs, and the parts of the prediction that depend on orientation -
# which is noise, since orientation is not diagnostic - average out, while the
# parts that depend on pathology reinforce.
#
# Only label-preserving views belong here. Rotation by an arbitrary angle
# introduces black corners the model never saw at inference during training, and
# colour jitter perturbs exactly the hue information that distinguishes a
# haemorrhage from an exudate. Flips are safe; those are not.
#
# Cost: 4x the inference time. On 550 test images that is seconds.
TTA_VIEWS: Dict[str, Any] = {
    "none":   lambda t: t,
    "hflip":  lambda t: torch.flip(t, dims=[3]),
    "vflip":  lambda t: torch.flip(t, dims=[2]),
    "hvflip": lambda t: torch.flip(t, dims=[2, 3]),
}

DEFAULT_TTA = ("none", "hflip", "vflip", "hvflip")


def resolve_tta(tta: Any) -> Tuple[str, ...]:
    """Normalise the ``tta`` argument into a tuple of view names.

    ``False``/``None`` -> ``("none",)``, ``True`` -> the four flips, or pass an
    explicit sequence of view names.
    """
    if tta is None or tta is False:
        return ("none",)
    if tta is True:
        return DEFAULT_TTA
    views = tuple(tta)
    unknown = [v for v in views if v not in TTA_VIEWS]
    if unknown:
        raise ValueError(f"Unknown TTA view(s) {unknown}; choose from {list(TTA_VIEWS)}")
    return views


# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_split(
    model: torch.nn.Module,
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    device: torch.device,
    batch_size: Optional[int] = None,
    tta: Any = False,
    head: str = "classification",
    thresholds: Optional[List[float]] = None,
    sigma: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Run inference over a dataframe. Returns (y_true, y_pred, probabilities, ids).

    ``shuffle=False`` throughout, so ``ids`` lines up with the prediction rows -
    which is what lets Milestone 7 find specific misclassified images.

    With ``head="ordinal"`` the model emits one score per image and ``thresholds``
    (from the checkpoint, fitted on validation) cut it into grades. The returned
    probability matrix is then derived from the score rather than produced by a
    softmax - see ``src.models.thresholds.scores_to_probabilities`` for what that
    does and does not mean.

    ``tta=True`` averages the softmax output over the four flip views described
    above. The transform pipeline stays ``train=False``: the flips are applied to
    the already-normalised tensor, deterministically, so the result is still
    reproducible - unlike feeding the random training augmentation at inference,
    which is the classic wrong way to do this.
    """
    loader, dataset = build_loader(
        df, cfg, train=False, batch_size=batch_size, shuffle=False
    )

    views = resolve_tta(tta)

    all_probs, all_true = [], []
    model.eval()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        # Classification: average probabilities, not logits - logits from
        # different views live on different scales and averaging them is not a
        # well-defined operation. Ordinal: average the scores, which DO share a
        # scale (they are all estimates of the same severity number), and which
        # is the more natural thing to average anyway.
        batch_out = None
        for view in views:
            out = model(TTA_VIEWS[view](images)).float()
            if head != "ordinal":
                out = torch.softmax(out, dim=1)
            batch_out = out if batch_out is None else batch_out + out
        batch_out = batch_out / len(views)

        all_probs.append(batch_out.cpu().numpy())
        all_true.append(labels.numpy())

    outputs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_true, axis=0)

    if head == "ordinal":
        if thresholds is None:
            raise ValueError(
                "An ordinal model needs the thresholds fitted during training. "
                "They are stored in best.pt; pass them through rather than "
                "refitting on this split - refitting on test would be tuning on "
                "the held-out data."
            )
        scores = outputs.reshape(-1)
        y_pred = apply_thresholds(scores, thresholds)
        probabilities = scores_to_probabilities(
            scores, num_classes=len(thresholds) + 1, sigma=sigma,
            thresholds=thresholds)
    else:
        probabilities = outputs
        y_pred = probabilities.argmax(axis=1)

    return y_true, y_pred, probabilities, dataset.ids


def evaluate_run(
    run_dir: Path,
    cfg: Dict[str, Any],
    device: torch.device,
    split: str = "test",
    save: bool = True,
    tta: Any = False,
) -> Dict[str, Any]:
    """Evaluate one experiment's best checkpoint on a split.

    With ``tta`` set, artefacts are written as ``metrics_<split>_tta.json`` and
    ``<split>_tta_predictions.npz``. The plain files are left untouched, so the
    two rows can sit side by side in the results table instead of one quietly
    replacing the other.
    """
    run_dir = Path(run_dir)
    model, ckpt = load_checkpoint(run_dir / "best.pt", device,
                                  num_classes=cfg["dataset"]["num_classes"])

    views = resolve_tta(tta)
    suffix = "_tta" if len(views) > 1 else ""

    head = str(ckpt.get("head", "classification"))
    thresholds = ckpt.get("thresholds")
    if head == "ordinal":
        if not thresholds:
            raise ValueError(
                f"{run_dir.name} was trained with the ordinal head but its "
                "checkpoint carries no thresholds. Re-run training - the "
                "thresholds are part of the model, not a post-hoc choice."
            )
        print(f"[evaluate_run] ordinal head, thresholds from checkpoint: "
              f"{[round(t, 3) for t in thresholds]}")

    df = load_split(get_path(cfg, "splits_dir"), split)
    y_true, y_pred, probabilities, ids = predict_split(
        model, df, cfg, device, tta=tta, head=head, thresholds=thresholds,
        sigma=float(ckpt.get("ordinal_sigma", 0.5)),
    )

    metrics = compute_all_metrics(y_true, y_pred, probabilities, class_names(cfg))
    metrics["split"] = split
    metrics["run_name"] = ckpt.get("run_name", run_dir.name)
    metrics["backbone"] = ckpt.get("backbone")
    metrics["checkpoint_epoch"] = ckpt.get("epoch")
    metrics["split_fingerprint"] = ckpt.get("split_fingerprint")
    metrics["tta_views"] = list(views)
    metrics["head"] = head
    metrics["thresholds"] = list(thresholds) if thresholds else None

    if save:
        clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
        (run_dir / f"metrics_{split}{suffix}.json").write_text(
            json.dumps(clean, indent=2), encoding="utf-8")
        np.savez_compressed(
            run_dir / f"{split}{suffix}_predictions.npz",
            y_true=y_true, y_pred=y_pred, probabilities=probabilities,
            ids=np.array(ids, dtype=object),
        )

    # Kept out of the JSON but returned, for figures.
    metrics["_y_true"], metrics["_y_pred"] = y_true, y_pred
    metrics["_probabilities"], metrics["_ids"] = probabilities, ids
    return metrics


# ---------------------------------------------------------------------------
def find_runs(experiments_dir: Path) -> List[Path]:
    """All run folders holding a finished checkpoint (smoke tests excluded)."""
    if not Path(experiments_dir).is_dir():
        return []
    return sorted(
        d for d in Path(experiments_dir).iterdir()
        if d.is_dir() and (d / "best.pt").is_file() and not d.name.startswith("smoke_")
    )


def comparison_table(all_metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per run: the table that goes straight into your results section."""
    rows = []
    for m in all_metrics:
        row = {
            "backbone": m.get("backbone"),
            "run": m.get("run_name"),
            "QWK": m["qwk"],
            "accuracy": m["accuracy"],
            "balanced_acc": m["balanced_accuracy"],
            "F1_macro": m["f1_macro"],
            "AUC_macro": m.get("auc_macro"),
            "best_epoch": m.get("checkpoint_epoch"),
        }
        # Per-class recall matters clinically: missing severe disease is the
        # dangerous error, and a macro average hides it.
        for pc in m["per_class"]:
            row[f"recall_{pc['label']}"] = pc["recall"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("QWK", ascending=False).reset_index(drop=True)


def per_class_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Precision / recall / F1 / support for one run."""
    return pd.DataFrame(metrics["per_class"])


def error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ids: List[str],
) -> Dict[str, Any]:
    """How wrong are the wrong answers?

    Because grading is ordinal, an off-by-one error is a different kind of
    mistake from an off-by-three. This is what explains a high QWK sitting
    alongside a mediocre accuracy, and it belongs in your discussion section.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    distance = np.abs(y_pred - y_true)
    wrong = distance > 0

    return {
        "n": int(y_true.size),
        "n_correct": int((~wrong).sum()),
        "n_wrong": int(wrong.sum()),
        "exact_match_pct": round(float((~wrong).mean() * 100), 2),
        "within_one_pct": round(float((distance <= 1).mean() * 100), 2),
        "within_two_pct": round(float((distance <= 2).mean() * 100), 2),
        "mean_abs_error": round(float(distance.mean()), 4),
        "error_distance_counts": {int(d): int((distance == d).sum())
                                  for d in range(0, 5)},
        "worst_examples": [
            {"id": ids[i], "true": int(y_true[i]), "pred": int(y_pred[i]),
             "distance": int(distance[i])}
            for i in np.argsort(-distance)[:10] if distance[i] > 0
        ],
    }
