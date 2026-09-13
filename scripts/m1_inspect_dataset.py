"""
MILESTONE 1, STEP 2 - Dataset inspection / EDA.
File location: <project_root>/scripts/m1_inspect_dataset.py

Objective
---------
Prove the dataset is intact and produce the first three figures that will go
straight into your report:

  reports/figures/m1_class_distribution.png
  reports/figures/m1_image_size_scatter.png
  reports/figures/m1_sample_grid.png

and two machine-readable artefacts:

  reports/m1_class_distribution.csv
  reports/m1_image_properties.csv

It also prints the inverse-frequency class weights you will reuse in
Milestone 5, so the imbalance decision is evidence-based rather than assumed.

Run
---
    python -m scripts.m1_inspect_dataset
    python -m scripts.m1_inspect_dataset --scan-all      # scan every image (slow)
    python -m scripts.m1_inspect_dataset --sample-size 800

Nothing here trains anything and nothing is random without a seed, so you can
re-run it as often as you like and get identical output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- make `src` importable no matter how this script is launched ---------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")  # headless-safe: write files, never open a window
import matplotlib.pyplot as plt
import pandas as pd

from src.data.inspect import (
    check_image_files,
    class_distribution,
    find_duplicate_ids,
    load_labels,
    pick_samples_per_class,
    scan_image_properties,
    suggested_class_weights,
    summarise_image_properties,
)
from src.utils.config import class_names, ensure_dirs, get_path, load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

LOG = get_logger("m1.inspect")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_class_distribution(dist: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of samples per severity stage, annotated with counts."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(dist["class_name"], dist["count"], color="#4C72B0", edgecolor="black", linewidth=0.6)
    for bar, count, pct in zip(bars, dist["count"], dist["percent"]):
        ax.annotate(
            f"{count}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_title("APTOS 2019 - class distribution (training labels)")
    ax.set_xlabel("Diabetic retinopathy severity stage")
    ax.set_ylabel("Number of fundus images")
    ax.set_ylim(0, dist["count"].max() * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info("Saved figure: %s", out_path)


def plot_image_sizes(props: pd.DataFrame, out_path: Path) -> None:
    """Scatter of width vs height, showing how heterogeneous the raw images are."""
    ok = props[props["readable"]]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(ok["width"], ok["height"], s=14, alpha=0.45, color="#DD8452", edgecolors="none")
    ax.set_title(f"Raw image dimensions (n={len(ok)} sampled)")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.grid(alpha=0.25, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info("Saved figure: %s", out_path)


def plot_sample_grid(
    samples: pd.DataFrame,
    images_dir: Path,
    ext: str,
    id_col: str,
    label_col: str,
    names: list[str],
    out_path: Path,
    per_class: int,
) -> None:
    """Grid of example fundus images, one row per severity stage."""
    import cv2

    n_classes = len(names)
    fig, axes = plt.subplots(n_classes, per_class, figsize=(per_class * 2.6, n_classes * 2.6))
    axes = axes.reshape(n_classes, per_class)

    for row, label in enumerate(range(n_classes)):
        subset = samples[samples[label_col] == label].reset_index(drop=True)
        for col in range(per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(subset):
                continue
            image_id = subset.loc[col, id_col]
            img = cv2.imread(str(images_dir / f"{image_id}{ext}"), cv2.IMREAD_COLOR)
            if img is None:
                ax.set_title("unreadable", fontsize=7)
                continue
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(image_id, fontsize=6)
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        axes[row, 0].set_ylabel(f"{label}: {names[label]}", fontsize=8, rotation=90, labelpad=8)

    fig.suptitle("Sample fundus images by severity stage (raw, before preprocessing)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info("Saved figure: %s", out_path)


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the APTOS 2019 dataset.")
    parser.add_argument("--sample-size", type=int, default=400,
                        help="how many images to scan for size stats (default 400)")
    parser.add_argument("--scan-all", action="store_true",
                        help="scan every training image (slow, but exact)")
    parser.add_argument("--per-class", type=int, default=4,
                        help="example images per class in the sample grid")
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["project"]["seed"]
    set_seed(seed)
    LOG.info("Seed set to %d - this script is fully reproducible.", seed)

    ds = cfg["dataset"]
    aptos_dir = get_path(cfg, "aptos_dir")
    train_csv = aptos_dir / ds["train_csv"]
    images_dir = aptos_dir / ds["train_images"]
    ensure_dirs(cfg, "reports_dir", "figures_dir")
    figures_dir = get_path(cfg, "figures_dir")
    reports_dir = get_path(cfg, "reports_dir")

    try:
        # -- 1. Labels ------------------------------------------------------
        df = load_labels(train_csv, ds["id_col"], ds["label_col"])
        LOG.info("Loaded %d labelled rows from %s", len(df), train_csv.name)

        expected = ds.get("expected_train_rows")
        if expected and len(df) != expected:
            LOG.warning("Row count %d differs from the expected %d for the public "
                        "APTOS release. Not fatal - just confirm you have the right files.",
                        len(df), expected)

        dupes = find_duplicate_ids(df, ds["id_col"])
        if dupes:
            LOG.warning("Found %d duplicate id(s), e.g. %s", len(dupes), dupes[:5])
        else:
            LOG.info("No duplicate image ids.")

        # -- 2. Files on disk ----------------------------------------------
        missing, orphans = check_image_files(df, images_dir, ds["id_col"], ds["image_ext"])
        if missing:
            LOG.error("%d id(s) in the CSV have no image file, e.g. %s",
                      len(missing), missing[:5])
        else:
            LOG.info("Every labelled id has a matching image file.")
        if orphans:
            LOG.warning("%d image file(s) on disk are not in the CSV, e.g. %s",
                        len(orphans), orphans[:5])

        # -- 3. Class distribution -----------------------------------------
        names = class_names(cfg)
        dist = class_distribution(df, ds["label_col"], ds["class_names"])
        LOG.info("Class distribution:\n%s", dist.to_string(index=False))
        dist_csv = reports_dir / "m1_class_distribution.csv"
        dist.to_csv(dist_csv, index=False)
        LOG.info("Saved table: %s", dist_csv)

        weights = suggested_class_weights(df, ds["label_col"], ds["num_classes"])
        LOG.info("Inverse-frequency class weights for Milestone 5 (mean-normalised): %s",
                 ", ".join(f"{n}={w:.3f}" for n, w in zip(names, weights)))

        # -- 4. Image properties -------------------------------------------
        sample_size = None if args.scan_all else args.sample_size
        LOG.info("Scanning image properties (%s images)...",
                 "all" if sample_size is None else sample_size)
        props = scan_image_properties(
            df[ds["id_col"]], images_dir, ds["image_ext"], sample_size=sample_size, seed=seed
        )
        unreadable = props[~props["readable"]]
        if len(unreadable):
            LOG.error("%d image(s) could not be read, e.g. %s",
                      len(unreadable), unreadable["id"].head().tolist())
        LOG.info("Image property summary:\n%s", summarise_image_properties(props).to_string())
        props_csv = reports_dir / "m1_image_properties.csv"
        props.to_csv(props_csv, index=False)
        LOG.info("Saved table: %s", props_csv)

        # -- 5. Figures -----------------------------------------------------
        plot_class_distribution(dist, figures_dir / "m1_class_distribution.png")
        plot_image_sizes(props, figures_dir / "m1_image_size_scatter.png")
        samples = pick_samples_per_class(df, ds["id_col"], ds["label_col"],
                                         per_class=args.per_class, seed=seed)
        plot_sample_grid(samples, images_dir, ds["image_ext"], ds["id_col"], ds["label_col"],
                         names, figures_dir / "m1_sample_grid.png", args.per_class)

    except Exception as exc:  # noqa: BLE001 - top-level CLI handler
        LOG.error("%s: %s", type(exc).__name__, exc)
        return 1

    LOG.info("Milestone 1 complete. Review the figures in reports/figures/, then confirm "
             "before we move to Milestone 2 (preprocessing and augmentation).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
