"""
MILESTONES 4 & 5 - Train one architecture.
File location: <project_root>/scripts/m4_train.py

Objective
---------
Train one of the three architectures on the Milestone 3 split, with weighted
cross-entropy for class imbalance, and checkpoint the best model by validation
Quadratic Weighted Kappa.

Run
---
    # ALWAYS smoke-test first: ~2 minutes, proves the whole path works
    python -m scripts.m4_train --experiment efficientnet_b3 --smoke

    # then the real runs
    python -m scripts.m4_train --experiment resnet50
    python -m scripts.m4_train --experiment efficientnet_b3
    python -m scripts.m4_train --experiment densenet121

    # or all three back to back (see scripts/run_all.ps1)

Options
-------
    --experiment NAME   configs/experiments/NAME.yaml     (required)
    --smoke             2 epochs on 64 images, nothing saved
    --epochs N          override the configured epoch count
    --limit N           use only the first N training images
    --run-name NAME     override the output folder name

Outputs land in experiments/<run_name>/.

The test split is NOT touched here. It is evaluated exactly once, in
Milestone 6, after all model selection is finished. Evaluating on test and then
tuning would make your reported numbers optimistically biased.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- make `src` importable no matter how this script is launched ------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_experiment_config
from src.utils.logging_utils import get_logger

LOG = get_logger("m4")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one DR classifier.")
    parser.add_argument("--experiment", required=True,
                        help="name of a file in configs/experiments/ (without .yaml)")
    parser.add_argument("--smoke", action="store_true",
                        help="2 epochs on 64 images; saves nothing. Run this FIRST.")
    parser.add_argument("--epochs", type=int, default=None, help="override epoch count")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N training images")
    parser.add_argument("--run-name", default=None, help="override the output folder name")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override batch size (use if you hit CUDA OOM)")
    args = parser.parse_args()

    try:
        cfg = load_experiment_config(args.experiment)
    except FileNotFoundError as exc:
        LOG.error("%s", exc)
        return 1

    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
        LOG.info("batch_size overridden to %d", args.batch_size)

    limit = args.limit
    epochs = args.epochs
    save = True

    if args.smoke:
        limit = limit or 64
        epochs = epochs or 2
        save = False
        cfg["run_name"] = f"smoke_{cfg['training']['backbone']}"
        LOG.info("SMOKE TEST MODE: %d images, %d epochs, nothing saved.", limit, epochs)

    # Import here so a bad --experiment name fails fast, before torch loads
    # (which takes several seconds).
    try:
        from src.models.train import train
    except ImportError as exc:
        LOG.error("Could not import the training code: %s", exc)
        LOG.error("Is PyTorch installed? "
                  "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return 1

    try:
        summary = train(cfg, limit=limit, epochs_override=epochs, save=save)
    except SystemExit as exc:                      # raised on CUDA OOM
        return int(exc.code or 1)
    except KeyboardInterrupt:
        LOG.warning("Interrupted by user. best.pt (if any) is still on disk.")
        return 130
    except Exception as exc:  # noqa: BLE001
        LOG.error("%s: %s", type(exc).__name__, exc)
        import traceback
        traceback.print_exc()
        return 1

    if args.smoke:
        LOG.info("-" * 60)
        LOG.info("SMOKE TEST PASSED for %s.", cfg["training"]["backbone"])
        LOG.info("Mean epoch time on 64 images: %ss", summary.get("mean_epoch_seconds"))
        full = 2562
        if summary.get("mean_epoch_seconds"):
            est = summary["mean_epoch_seconds"] * full / max(limit, 1) / 60
            LOG.info("Naive estimate for a full epoch (%d images): %.1f min", full, est)
            LOG.info("So %d epochs would come to roughly %.0f min.",
                     cfg["training"]["epochs"], est * cfg["training"]["epochs"])
            # That extrapolation is linear in image count, and it is WRONG -
            # badly, and always in the pessimistic direction. A 64-image epoch
            # is nearly all fixed cost: spawning Windows DataLoader workers,
            # cuDNN autotuning the first convolution, moving the model to the
            # GPU. Those happen once per epoch regardless of size, so scaling
            # 64 images up by 40x scales the overhead by 40x too.
            #
            # For the real figure, read epoch_seconds in an existing run's
            # history.csv: the Milestone 5 DenseNet121 run averaged about 45s
            # per epoch over the full 2,562 training and 550 validation images,
            # roughly 12x faster per image than a smoke epoch. Expect a 45-epoch
            # run to take well under an hour, not the number printed above.
            LOG.info("Treat that as an upper bound only - a 64-image epoch is "
                     "mostly fixed startup cost. See epoch_seconds in any "
                     "existing experiments/*/history.csv for the real rate.")
        LOG.info("-" * 60)

    return 0


if __name__ == "__main__":
    # __main__ guard is REQUIRED on Windows: DataLoader workers re-import this
    # module, and without it each worker would recursively start a new run.
    sys.exit(main())
