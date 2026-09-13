"""
MILESTONE 3 - Stratified train / validation / test split.
File location: <project_root>/scripts/m3_split.py

Objective
---------
Partition the 3,662 labelled images into 70% train / 15% val / 15% test, with
class proportions preserved, using seed 42, and write the result to
``data/splits/{train,val,test}.csv``.

Every training run reads those CSVs. That is the whole point: all three
architectures must see byte-identical data, or the Milestone 6 comparison is
not a comparison.

The script refuses to write anything if it detects leakage or a missing class.

Run
---
    python -m scripts.m3_split
    python -m scripts.m3_split --force        # overwrite an existing split

WARNING: once you have trained on a split, DO NOT regenerate it. Re-running
with a different seed invalidates every result you have. The script will not
overwrite existing split files without --force for exactly this reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- make `src` importable no matter how this script is launched ------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.inspect import load_labels, suggested_class_weights
from src.data.split import (
    SPLIT_COL,
    split_distribution,
    split_fingerprint,
    split_from_config,
    verify_split,
    write_splits,
)
from src.utils.config import class_names, ensure_dirs, get_path, load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

LOG = get_logger("m3.split")


def plot_distribution(dist, names, out_path: Path) -> None:
    """Grouped bar chart: class share within each split, plus the overall share."""
    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (split, colour) in enumerate(
        [("train", "#4C72B0"), ("val", "#DD8452"), ("test", "#55A868")]
    ):
        ax.bar(x + (i - 1) * width, dist[f"{split}_pct"], width,
               label=f"{split} (n={int(dist[f'{split}_n'].sum())})",
               color=colour, edgecolor="black", linewidth=0.5)

    ax.plot(x, dist["overall_pct"], "k--o", markersize=4, linewidth=1.2,
            label="overall", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Share of the split (%)")
    ax.set_title("Stratified split: class proportions are preserved across all three splits")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info("Saved figure: %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the stratified data split.")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing split files (INVALIDATES trained results)")
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["project"]["seed"]
    set_seed(seed)

    ds = cfg["dataset"]
    aptos_dir = get_path(cfg, "aptos_dir")
    ensure_dirs(cfg, "splits_dir", "reports_dir", "figures_dir")
    splits_dir = get_path(cfg, "splits_dir")
    reports_dir = get_path(cfg, "reports_dir")
    figures_dir = get_path(cfg, "figures_dir")

    existing = [p for p in ("train", "val", "test") if (splits_dir / f"{p}.csv").is_file()]
    if existing and not args.force:
        LOG.error("Split files already exist: %s", ", ".join(f"{p}.csv" for p in existing))
        LOG.error("Refusing to overwrite. If you have already trained on this split, "
                  "regenerating it would invalidate your results.")
        LOG.error("If you are sure, re-run with --force.")
        return 1

    try:
        df = load_labels(aptos_dir / ds["train_csv"], ds["id_col"], ds["label_col"])
        LOG.info("Loaded %d labelled rows.", len(df))

        split_df = split_from_config(df, cfg)

        # ---- verify BEFORE writing ---------------------------------------
        problems = verify_split(split_df, ds["id_col"], ds["label_col"])
        if problems:
            LOG.error("Split verification FAILED - nothing written:")
            for p in problems:
                LOG.error("  %s", p)
            return 1
        LOG.info("Split verification passed: no leakage, full coverage, all classes present.")

        names = class_names(cfg)
        dist = split_distribution(split_df, ds["label_col"], ds["class_names"])
        LOG.info("Class distribution per split:\n%s", dist.to_string(index=False))

        worst_drift = float(dist["drift_pp"].max())
        if worst_drift > 2.0:
            LOG.warning("Largest class-share drift is %.2f pp - higher than expected for a "
                        "stratified split. Check `split.stratify` is true.", worst_drift)
        else:
            LOG.info("Largest class-share drift: %.2f pp (stratification working).", worst_drift)

        # ---- write --------------------------------------------------------
        written = write_splits(split_df, splits_dir)
        for name, path in written.items():
            LOG.info("Wrote %-4s -> %s", name, path)

        dist.to_csv(reports_dir / "m3_split_distribution.csv", index=False)
        plot_distribution(dist, names, figures_dir / "m3_split_distribution.png")

        # ---- class weights, computed on the TRAIN split only -------------
        # Using the whole dataset here would leak test-set label statistics
        # into training. A small leak, but free to avoid.
        train_df = split_df[split_df[SPLIT_COL] == "train"]
        weights = suggested_class_weights(train_df, ds["label_col"], ds["num_classes"])

        fingerprint = split_fingerprint(split_df, ds["id_col"])

        meta = {
            "seed": seed,
            "fingerprint": fingerprint,
            "stratified": cfg["split"].get("stratify", True),
            "sizes": {s: int((split_df[SPLIT_COL] == s).sum()) for s in ("train", "val", "test")},
            "class_weights_from_train": [round(float(w), 6) for w in weights],
            "class_names": names,
            "worst_class_share_drift_pp": worst_drift,
        }
        (splits_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        LOG.info("-" * 64)
        LOG.info("Sizes        : train %d | val %d | test %d",
                 meta["sizes"]["train"], meta["sizes"]["val"], meta["sizes"]["test"])
        LOG.info("Fingerprint  : %s   (every training run records this)", fingerprint)
        LOG.info("Class weights (from TRAIN split only, for weighted cross-entropy):")
        for name, w in zip(names, weights):
            LOG.info("    %-18s %.4f", name, w)
        LOG.info("-" * 64)

    except Exception as exc:  # noqa: BLE001
        LOG.error("%s: %s", type(exc).__name__, exc)
        return 1

    LOG.info("Milestone 3 complete. Next: python -m scripts.m4_train --experiment efficientnet_b3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
