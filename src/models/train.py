"""
Training loop.
File location: <project_root>/src/models/train.py

Design notes
------------
* **Model selection on validation QWK**, never on accuracy and never on the
  test split. ``best.pt`` is rewritten every time validation QWK improves, so
  even if an overnight run dies at epoch 15 you still have the best model it
  found.
* **Mixed precision (AMP)** roughly halves activation memory and speeds up
  training on any RTX card. Essential at 300x300 on 6 GB.
* **Gradient accumulation** lets a 6 GB card train at an effective batch of 16
  while only ever holding 8 images of activations. Mathematically identical to
  a batch of 16, just slower.
* **Explicit LR schedule** (linear warmup then cosine decay) computed by a
  plain function rather than a scheduler object - fewer moving parts, and the
  LR at any step is easy to reason about and log.
* **Class imbalance** is handled by weighted cross-entropy, with weights taken
  from the TRAIN split only. Using whole-dataset counts would leak test-set
  label statistics into training.
* **Two head types.** ``training.head: classification`` is the original 5-way
  softmax. ``training.head: ordinal`` trains a single regression output and
  cuts it into grades with thresholds fitted on validation at every epoch (see
  ``src.models.thresholds``). Everything else - schedule, AMP, accumulation,
  checkpointing - is shared, so the two are directly comparable.
* **Early stopping on a smoothed curve.** Validation QWK on 550 images has a
  sampling noise of roughly +/-0.03. Comparing raw epoch-to-epoch scores against
  a patience counter stops runs on noise: the Milestone 5 EfficientNet-B3 run
  halted at epoch 8 having peaked at epoch 3 of a 20-epoch cosine schedule,
  before the schedule had annealed at all. Patience is therefore counted
  against a moving average, while ``best.pt`` still tracks the best single
  epoch.

Every run writes to ``experiments/<run_name>/``:
    best.pt              model weights at the best validation QWK
    history.csv          per-epoch losses, metrics, LR, timing
    metrics_val.json     full metrics for the best epoch
    config_snapshot.yaml the exact merged config that produced this run
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import build_loader, make_weighted_sampler
from src.data.split import load_split, split_fingerprint
from src.models.factory import (
    build_from_config,
    check_forward,
    describe_model,
    head_type,
    num_outputs,
)
from src.models.metrics import compute_all_metrics, summarise_for_console
from src.models.thresholds import (
    apply_thresholds,
    describe as describe_thresholds,
    fit_thresholds,
    scores_to_probabilities,
)
from src.utils.config import class_names, get_path, save_config_snapshot
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------
def lr_at_step(step: int, total_steps: int, warmup_steps: int,
               base_lr: float, min_lr_frac: float = 0.01) -> float:
    """Linear warmup for ``warmup_steps``, then cosine decay to ``min_lr``.

    Warmup matters here: the backbone carries pretrained weights and the head is
    random, so a full-size LR on step 1 can wreck useful features before the
    head has learned anything.
    """
    min_lr = base_lr * min_lr_frac
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


# ---------------------------------------------------------------------------
# Loss and imbalance handling
# ---------------------------------------------------------------------------
def class_weights_from_labels(labels: List[int], num_classes: int) -> np.ndarray:
    """Inverse-frequency weights normalised to mean 1.0 (zero-count -> 0)."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(float)
    weights = np.zeros_like(counts)
    nonzero = counts > 0
    weights[nonzero] = counts[nonzero].sum() / (nonzero.sum() * counts[nonzero])
    if weights[nonzero].mean() > 0:
        weights[nonzero] /= weights[nonzero].mean()
    return weights


class OrdinalRegressionLoss(nn.Module):
    """Smooth L1 between a single predicted score and the integer grade.

    Why Smooth L1 rather than plain MSE: APTOS labels are graded by clinicians
    and genuinely noisy, and squared error lets one badly-labelled image
    dominate a batch's gradient. Smooth L1 is quadratic near zero (so it still
    cares about small errors, which is the whole point of going ordinal) and
    linear beyond ``beta``, which caps the influence of an outlier.

    Class weights carry over from the classification path unchanged: the same
    inverse-frequency weights, applied per sample by its true grade. Without
    them a regressor trained on data that is 49% class 0 simply learns to
    predict low numbers.
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None,
                 beta: float = 1.0) -> None:
        super().__init__()
        self.beta = float(beta)
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.empty(0),
        )

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        scores = outputs.reshape(outputs.shape[0])          # (N, 1) -> (N,)
        loss = F.smooth_l1_loss(scores, targets.float(), beta=self.beta,
                                reduction="none")
        if self.class_weights.numel():
            loss = loss * self.class_weights[targets]
        return loss.mean()


def build_criterion(
    cfg: Dict[str, Any],
    train_labels: List[int],
    device: torch.device,
    logger,
) -> Tuple[nn.Module, Optional[np.ndarray]]:
    """Cross-entropy or ordinal regression loss, optionally class-weighted."""
    strategy = cfg["training"].get("imbalance_strategy", "weighted_loss")
    num_classes = cfg["dataset"]["num_classes"]
    smoothing = float(cfg["training"].get("label_smoothing", 0.0))
    head = head_type(cfg)

    weights = None
    if strategy in ("weighted_loss", "both"):
        weights = class_weights_from_labels(train_labels, num_classes)
        logger.info("Weighted cross-entropy, weights from the TRAIN split:")
        for name, w in zip(class_names(cfg), weights):
            logger.info("    %-18s %.4f", name, w)
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        logger.info("Unweighted cross-entropy (imbalance_strategy=%s).", strategy)
        weight_tensor = None

    if head == "ordinal":
        beta = float(cfg["training"].get("ordinal_beta", 1.0))
        logger.info("Ordinal head: Smooth L1 regression loss (beta=%.2f), "
                    "thresholds fitted on validation each epoch.", beta)
        criterion = OrdinalRegressionLoss(weight_tensor, beta=beta).to(device)
    else:
        if smoothing:
            logger.info("Label smoothing: %.3f (clinician-graded labels are noisy, "
                        "so a hard one-hot target overstates what is known).", smoothing)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=smoothing)
    return criterion, weights


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    amp_ctx,
    accum_steps: int,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    max_grad_norm: float,
    logger,
    log_every: int = 50,
) -> Tuple[float, int, float]:
    """Run one training epoch. Returns (mean loss, new global_step, last lr)."""
    model.train()
    running_loss = 0.0
    n_batches = 0
    last_lr = base_lr
    n_total = len(loader)

    optimizer.zero_grad(set_to_none=True)

    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with amp_ctx():
            logits = model(images)
            loss = criterion(logits, labels)

        # Divide by accum_steps so the accumulated gradient equals the gradient
        # of the mean loss over the effective batch.
        scaler.scale(loss / accum_steps).backward()

        running_loss += float(loss.detach().item())
        n_batches += 1

        is_last = (i + 1) == n_total
        if (i + 1) % accum_steps == 0 or is_last:
            last_lr = lr_at_step(global_step, total_steps, warmup_steps, base_lr)
            set_lr(optimizer, last_lr)

            if max_grad_norm and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if log_every and (i + 1) % log_every == 0:
            logger.info("    batch %4d/%-4d  loss %.4f  lr %.2e",
                        i + 1, n_total, running_loss / max(n_batches, 1), last_lr)

    return running_loss / max(n_batches, 1), global_step, last_lr


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    amp_ctx,
    cfg: Dict[str, Any],
    thresholds: Optional[Sequence[float]] = None,
    refit_thresholds: bool = True,
) -> Dict[str, Any]:
    """Evaluate on a loader. Returns the metrics dict plus loss and raw arrays.

    For the ordinal head the thresholds are re-fitted on this loader by default.
    That is correct for the validation split - the thresholds are part of the
    model and validation is where model choices are made - and wrong for test,
    where ``refit_thresholds=False`` plus the thresholds stored in the
    checkpoint is the only honest option.
    """
    head = head_type(cfg)
    num_classes = cfg["dataset"]["num_classes"]

    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_outputs: List[np.ndarray] = []
    all_true: List[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with amp_ctx():
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += float(loss.detach().item())
        n_batches += 1

        # float() before softmax: fp16 softmax can underflow to zero.
        all_outputs.append(logits.float().cpu().numpy())
        all_true.append(labels.cpu().numpy())

    outputs = np.concatenate(all_outputs, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    threshold_fit: Optional[Dict[str, Any]] = None
    scores: Optional[np.ndarray] = None

    if head == "ordinal":
        scores = outputs.reshape(-1)
        if refit_thresholds or thresholds is None:
            # Which objective the cut-points maximise is a real choice, not a
            # detail: QWK-optimal and accuracy-optimal thresholds differ, and on
            # this dataset they trade against each other (see
            # reports/m6c_objective_tradeoff.csv). Fit the one you will report.
            threshold_fit = fit_thresholds(
                scores, y_true, num_classes=num_classes,
                metric=str(cfg["training"].get("threshold_metric", "qwk")),
            )
            edges = threshold_fit["thresholds"]
        else:
            edges = list(thresholds)
        y_pred = apply_thresholds(scores, edges)
        probabilities = scores_to_probabilities(
            scores, num_classes=num_classes,
            sigma=float(cfg["training"].get("ordinal_sigma", 0.5)),
            thresholds=edges,
        )
    else:
        edges = None
        probabilities = torch.softmax(torch.from_numpy(outputs), dim=1).numpy()
        y_pred = probabilities.argmax(axis=1)

    metrics = compute_all_metrics(y_true, y_pred, probabilities, class_names(cfg))
    metrics["loss"] = round(total_loss / max(n_batches, 1), 5)
    metrics["head"] = head
    metrics["thresholds"] = list(edges) if edges is not None else None
    metrics["threshold_fit"] = threshold_fit
    metrics["_y_true"] = y_true.tolist()
    metrics["_y_pred"] = np.asarray(y_pred).tolist()
    metrics["_probabilities"] = probabilities.tolist()
    metrics["_scores"] = scores.tolist() if scores is not None else None
    return metrics


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def train(cfg: Dict[str, Any], limit: Optional[int] = None,
          epochs_override: Optional[int] = None, save: bool = True) -> Dict[str, Any]:
    """Train one experiment end to end. Returns a summary dict."""
    tr = cfg["training"]
    seed = cfg["project"]["seed"]
    set_seed(seed)

    run_name = cfg.get("run_name") or tr["backbone"]
    run_dir = get_path(cfg, "experiments_dir") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(f"train.{run_name}", log_file=run_dir / "train.log")
    logger.info("=" * 68)
    logger.info("  RUN: %s", run_name)
    logger.info("=" * 68)

    # -- device ----------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        logger.info("Device: %s (%.1f GB VRAM)", props.name, props.total_memory / 1e9)
    else:
        device = torch.device("cpu")
        logger.warning("Device: CPU - no CUDA available. This will be extremely slow.")

    use_amp = bool(tr.get("use_amp", True)) and device.type == "cuda"
    if use_amp:
        amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)  # noqa: E731
    else:
        amp_ctx = contextlib.nullcontext
    logger.info("Mixed precision (AMP): %s", use_amp)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):        # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # -- data ------------------------------------------------------------
    splits_dir = get_path(cfg, "splits_dir")
    train_df = load_split(splits_dir, "train")
    val_df = load_split(splits_dir, "val")

    if limit:
        train_df = train_df.head(limit).copy()
        val_df = val_df.head(max(limit // 2, 8)).copy()
        logger.warning("SMOKE TEST: limited to %d train / %d val samples. "
                       "Results are meaningless - this only checks the code runs.",
                       len(train_df), len(val_df))

    label_col = cfg["dataset"]["label_col"]
    train_labels = train_df[label_col].astype(int).tolist()
    num_classes = cfg["dataset"]["num_classes"]

    strategy = tr.get("imbalance_strategy", "weighted_loss")
    sampler = None
    if strategy in ("oversample", "both"):
        sampler = make_weighted_sampler(train_labels, num_classes, seed=seed)
        logger.info("Using WeightedRandomSampler (oversampling rare classes).")

    train_loader, train_ds = build_loader(train_df, cfg, train=True, sampler=sampler)
    val_loader, val_ds = build_loader(val_df, cfg, train=False, shuffle=False)

    logger.info("Train: %d images (%d from cache) | Val: %d images (%d from cache)",
                len(train_ds), train_ds.n_cached, len(val_ds), val_ds.n_cached)
    logger.info("Train class counts: %s", train_ds.class_counts(num_classes).tolist())

    # Fingerprint the split so two runs can be proven comparable.
    all_splits_path = splits_dir / "all_splits.csv"
    fingerprint = "unknown"
    if all_splits_path.is_file():
        combined = pd.read_csv(all_splits_path)
        fingerprint = split_fingerprint(combined, cfg["dataset"]["id_col"])
    logger.info("Split fingerprint: %s", fingerprint)

    # -- model -----------------------------------------------------------
    head = head_type(cfg)
    width = num_outputs(cfg)
    logger.info("Head: %s (%d output%s)", head, width, "" if width == 1 else "s")

    model = build_from_config(cfg)
    logger.info("Model:\n%s", describe_model(model, tr["backbone"]))
    out_shape = check_forward(model, size=cfg["preprocessing"]["image_size"],
                              num_classes=num_classes, device="cpu",
                              expected_outputs=width)
    logger.info("Forward check passed: output shape %s", out_shape)
    model = model.to(device)

    criterion, weights = build_criterion(cfg, train_labels, device, logger)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr["lr"]),
        weight_decay=float(tr.get("weight_decay", 1e-4)),
    )

    epochs = int(epochs_override or tr["epochs"])
    accum_steps = max(1, int(tr.get("grad_accum_steps", 1)))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum_steps))
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(steps_per_epoch * float(tr.get("warmup_epochs", 1)))
    max_grad_norm = float(tr.get("max_grad_norm", 1.0))

    logger.info("Epochs %d | batch %d x accum %d = effective %d | %d optimiser steps/epoch",
                epochs, tr["batch_size"], accum_steps,
                tr["batch_size"] * accum_steps, steps_per_epoch)

    if save:
        save_config_snapshot(cfg, run_dir / "config_snapshot.yaml")

    # -- loop ------------------------------------------------------------
    monitor = tr.get("monitor_metric", "val_qwk")
    patience = int(tr.get("early_stopping_patience", 5))
    # Patience is counted against a moving average of the monitored metric, not
    # the raw per-epoch value - see the module docstring. Window 1 restores the
    # original behaviour exactly.
    smooth_window = max(1, int(tr.get("selection_smooth_window", 3)))
    monitor_history: List[float] = []
    best_smoothed = -float("inf")
    best_thresholds: Optional[List[float]] = None
    best_score = -float("inf")
    best_epoch = -1
    best_metrics: Dict[str, Any] = {}
    epochs_without_improvement = 0
    global_step = 0
    history: List[Dict[str, Any]] = []
    run_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        logger.info("-" * 68)
        logger.info("Epoch %d/%d", epoch, epochs)

        try:
            train_loss, global_step, last_lr = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, amp_ctx,
                accum_steps, global_step, total_steps, warmup_steps,
                float(tr["lr"]), max_grad_norm, logger,
            )
            val_metrics = evaluate(model, val_loader, criterion, device, amp_ctx, cfg)

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error("=" * 68)
                logger.error("CUDA OUT OF MEMORY on %s.", tr["backbone"])
                logger.error("Fix: in configs/experiments/%s.yaml lower batch_size and",
                             cfg.get("_experiment", tr["backbone"]))
                logger.error("raise grad_accum_steps by the same factor, e.g.")
                logger.error("    batch_size: %d  ->  %d",
                             tr["batch_size"], max(2, tr["batch_size"] // 2))
                logger.error("    grad_accum_steps: %d  ->  %d",
                             accum_steps, accum_steps * 2)
                logger.error("The effective batch and the resulting model are unchanged.")
                logger.error("=" * 68)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise SystemExit(2) from exc
            raise

        epoch_time = time.perf_counter() - epoch_start
        score = val_metrics["qwk"] if monitor == "val_qwk" else val_metrics["accuracy"]
        monitor_history.append(score)
        smoothed = float(np.mean(monitor_history[-smooth_window:]))

        if val_metrics.get("threshold_fit"):
            logger.info("Thresholds fitted on validation:\n%s",
                        describe_thresholds(val_metrics["threshold_fit"]))

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": val_metrics["loss"],
            "val_qwk": val_metrics["qwk"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_auc_macro": val_metrics.get("auc_macro", float("nan")),
            "val_qwk_smoothed": round(smoothed, 5),
            "thresholds": ("|".join(f"{t:.3f}" for t in val_metrics["thresholds"])
                           if val_metrics.get("thresholds") else ""),
            "lr": round(last_lr, 8),
            "epoch_seconds": round(epoch_time, 1),
        }
        history.append(row)

        elapsed = time.perf_counter() - run_start
        eta = (elapsed / epoch) * (epochs - epoch)
        logger.info("train_loss %.4f | val_loss %.4f | val_QWK %.4f | val_acc %.4f | "
                    "%.0fs (ETA %.0f min)",
                    train_loss, val_metrics["loss"], val_metrics["qwk"],
                    val_metrics["accuracy"], epoch_time, eta / 60)

        # history.csv is rewritten every epoch, so a crash still leaves a
        # readable training curve.
        if save:
            with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)

        # -- checkpoint on improvement ----------------------------------
        if score > best_score:
            improvement = score - best_score if best_score > -float("inf") else score
            best_score, best_epoch = score, epoch
            best_metrics = val_metrics
            best_thresholds = val_metrics.get("thresholds")
            epochs_without_improvement = 0
            logger.info("  new best %s = %.4f (+%.4f) - saving best.pt", monitor, score, improvement)

            if save:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "backbone": tr["backbone"],
                        "num_classes": num_classes,
                        "head": head,
                        "num_outputs": width,
                        # The thresholds ARE part of the ordinal model. Storing
                        # them here is what lets test evaluation apply the
                        # validation-chosen cut-points instead of refitting.
                        "thresholds": best_thresholds,
                        "ordinal_sigma": float(tr.get("ordinal_sigma", 0.5)),
                        "image_size": cfg["preprocessing"]["image_size"],
                        "epoch": epoch,
                        monitor: score,
                        "class_names": class_names(cfg),
                        "preprocessing": cfg["preprocessing"],
                        "split_fingerprint": fingerprint,
                        "run_name": run_name,
                        "seed": seed,
                    },
                    run_dir / "best.pt",
                )

        # Early stopping is judged on the smoothed curve even when best.pt was
        # just rewritten, so a single lucky epoch cannot reset patience either.
        if smoothed > best_smoothed + 1e-6:
            best_smoothed = smoothed
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            logger.info("  smoothed %s %.4f (best %.4f) - no improvement (%d/%d); "
                        "best raw %s %.4f at epoch %d",
                        monitor, smoothed, best_smoothed,
                        epochs_without_improvement, patience,
                        monitor, best_score, best_epoch)
            if epochs_without_improvement >= patience:
                logger.info("Early stopping: %d epochs without improvement in the "
                            "%d-epoch moving average.", patience, smooth_window)
                break

    total_time = time.perf_counter() - run_start

    # -- write results ---------------------------------------------------
    summary = {
        "run_name": run_name,
        "experiment": cfg.get("_experiment"),
        "backbone": tr["backbone"],
        "head": head,
        "thresholds": best_thresholds,
        "threshold_metric": str(tr.get("threshold_metric", "qwk")),
        "selection_smooth_window": smooth_window,
        "image_size": cfg["preprocessing"]["image_size"],
        "epochs_run": len(history),
        "epochs_configured": epochs,
        "best_epoch": best_epoch,
        f"best_{monitor}": round(best_score, 4) if best_score > -float("inf") else None,
        "imbalance_strategy": strategy,
        "class_weights": [round(float(w), 4) for w in weights] if weights is not None else None,
        "batch_size": tr["batch_size"],
        "grad_accum_steps": accum_steps,
        "effective_batch": tr["batch_size"] * accum_steps,
        "lr": float(tr["lr"]),
        "use_amp": use_amp,
        "split_fingerprint": fingerprint,
        "seed": seed,
        "total_minutes": round(total_time / 60, 2),
        "mean_epoch_seconds": round(float(np.mean([r["epoch_seconds"] for r in history])), 1)
        if history else None,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }

    if save:
        clean_best = {k: v for k, v in best_metrics.items() if not k.startswith("_")}
        (run_dir / "metrics_val.json").write_text(
            json.dumps({"summary": summary, "val_metrics": clean_best}, indent=2),
            encoding="utf-8",
        )
        # Raw predictions for the Milestone 6 figures.
        if best_metrics:
            # ids let the ensemble align runs by image rather than by row
            # position; the Milestone 6b module falls back to position when an
            # older file lacks them.
            np.savez_compressed(
                run_dir / "val_predictions.npz",
                y_true=np.asarray(best_metrics["_y_true"]),
                y_pred=np.asarray(best_metrics["_y_pred"]),
                probabilities=np.asarray(best_metrics["_probabilities"]),
                ids=np.array(val_ds.ids, dtype=object),
            )

    logger.info("=" * 68)
    logger.info("Finished %s in %.1f min (%d epochs, mean %.0fs/epoch)",
                run_name, total_time / 60, len(history),
                summary["mean_epoch_seconds"] or 0)
    logger.info("Best %s = %.4f at epoch %d", monitor, best_score, best_epoch)
    if best_metrics:
        logger.info("Validation metrics at the best epoch:\n%s",
                    summarise_for_console(best_metrics))
    logger.info("Artefacts in: %s", run_dir)
    logger.info("=" * 68)

    return summary
