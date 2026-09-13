"""
Dataset inspection / EDA helpers.
File location: <project_root>/src/data/inspect.py

Pure functions only - no printing, no file writing, no argparse. The script
``scripts/m1_inspect_dataset.py`` calls these and handles all I/O. Keeping the
split makes these functions easy to unit-test later (Milestone 11).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading and integrity
# ---------------------------------------------------------------------------
def load_labels(csv_path: Path, id_col: str, label_col: str) -> pd.DataFrame:
    """Read the APTOS train.csv and validate its shape.

    Returns a DataFrame with exactly ``[id_col, label_col]``.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Label CSV not found: {csv_path}\n"
            "Did Milestone 1 step 1 finish? Run: python -m scripts.m1_download_data"
        )

    df = pd.read_csv(csv_path)

    missing = [c for c in (id_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing expected column(s) {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[[id_col, label_col]].copy()
    df[id_col] = df[id_col].astype(str)

    # Labels must be integers 0..4
    if df[label_col].isna().any():
        raise ValueError(f"{csv_path.name} contains missing labels.")
    df[label_col] = df[label_col].astype(int)

    return df


def check_image_files(
    df: pd.DataFrame,
    images_dir: Path,
    id_col: str,
    ext: str,
) -> Tuple[List[str], List[str]]:
    """Cross-check CSV rows against files on disk.

    Returns
    -------
    (missing_files, orphan_files)
        ``missing_files``  : ids listed in the CSV with no image on disk.
        ``orphan_files``   : image stems on disk that are not in the CSV.
    """
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    on_disk = {p.stem for p in images_dir.glob(f"*{ext}")}
    in_csv = set(df[id_col])

    missing = sorted(in_csv - on_disk)
    orphans = sorted(on_disk - in_csv)
    return missing, orphans


def find_duplicate_ids(df: pd.DataFrame, id_col: str) -> List[str]:
    """Return ids that appear more than once in the label file."""
    counts = df[id_col].value_counts()
    return sorted(counts[counts > 1].index.tolist())


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------
def class_distribution(
    df: pd.DataFrame,
    label_col: str,
    class_names: Dict[int, str] | Sequence[str],
) -> pd.DataFrame:
    """Per-class counts, percentages and imbalance ratio vs the largest class."""
    if isinstance(class_names, dict):
        names = {int(k): v for k, v in class_names.items()}
    else:
        names = dict(enumerate(class_names))

    counts = df[label_col].value_counts().sort_index()
    # Make sure every declared class appears, even with zero samples.
    counts = counts.reindex(sorted(names), fill_value=0)

    total = int(counts.sum())
    largest = int(counts.max()) if total else 0

    out = pd.DataFrame(
        {
            "label": counts.index.astype(int),
            "class_name": [names[int(i)] for i in counts.index],
            "count": counts.values.astype(int),
        }
    )
    out["percent"] = (out["count"] / total * 100).round(2) if total else 0.0
    # "1 image of this class for every N of the majority class"
    out["ratio_to_majority"] = (
        (largest / out["count"].replace(0, np.nan)).round(2) if largest else np.nan
    )
    return out


def suggested_class_weights(df: pd.DataFrame, label_col: str, num_classes: int) -> np.ndarray:
    """Inverse-frequency class weights, normalised to mean 1.0.

    These are the weights you will feed to ``nn.CrossEntropyLoss(weight=...)``
    in Milestone 5. Computed here so Milestone 1 already tells you how skewed
    the problem is. Classes with zero samples get weight 0.
    """
    counts = np.bincount(df[label_col].to_numpy(dtype=int), minlength=num_classes).astype(float)
    weights = np.zeros_like(counts)
    nonzero = counts > 0
    weights[nonzero] = counts[nonzero].sum() / (nonzero.sum() * counts[nonzero])
    if weights[nonzero].mean() > 0:
        weights[nonzero] /= weights[nonzero].mean()
    return weights


# ---------------------------------------------------------------------------
# Image property scan
# ---------------------------------------------------------------------------
def scan_image_properties(
    ids: Iterable[str],
    images_dir: Path,
    ext: str,
    sample_size: int | None = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """Read a sample of images and report width, height, aspect ratio, mean intensity.

    Reading all ~3.6k full-resolution APTOS images is slow (they are up to
    3388x2588), so by default we sample. Pass ``sample_size=None`` to scan all.

    Corrupt or unreadable images are reported with ``readable=False`` rather
    than raising, so one bad file never kills the whole scan.
    """
    import cv2  # imported here so the module stays importable without OpenCV

    ids = list(ids)
    if sample_size is not None and sample_size < len(ids):
        rng = np.random.default_rng(seed)
        ids = list(rng.choice(ids, size=sample_size, replace=False))

    rows: List[Dict[str, Any]] = []
    for image_id in ids:
        path = images_dir / f"{image_id}{ext}"
        record: Dict[str, Any] = {"id": image_id, "readable": False,
                                  "width": np.nan, "height": np.nan,
                                  "aspect_ratio": np.nan, "mean_intensity": np.nan,
                                  "file_kb": np.nan}
        try:
            record["file_kb"] = round(path.stat().st_size / 1024, 1)
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                rows.append(record)
                continue
            h, w = img.shape[:2]
            record.update(
                readable=True,
                width=int(w),
                height=int(h),
                aspect_ratio=round(w / h, 3),
                mean_intensity=round(float(img.mean()), 2),
            )
        except (OSError, ValueError):
            pass  # keep readable=False
        rows.append(record)

    return pd.DataFrame(rows)


def summarise_image_properties(props: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics over the scanned image properties."""
    ok = props[props["readable"]]
    if ok.empty:
        raise ValueError("No readable images in the scan - check the image directory and extension.")
    return ok[["width", "height", "aspect_ratio", "mean_intensity", "file_kb"]].describe().round(2)


def pick_samples_per_class(
    df: pd.DataFrame,
    id_col: str,
    label_col: str,
    per_class: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministically choose ``per_class`` example ids for each label."""
    rng = np.random.default_rng(seed)
    picks: List[pd.DataFrame] = []
    for label, group in df.groupby(label_col):
        n = min(per_class, len(group))
        idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
        picks.append(df.loc[idx, [id_col, label_col]])
    return pd.concat(picks).sort_values(label_col).reset_index(drop=True)
