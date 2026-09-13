"""
Stratified train / validation / test splitting.
File location: <project_root>/src/data/split.py

Why this milestone is more dangerous than it looks
--------------------------------------------------
A split mistake produces no error message. Training runs, numbers appear, and
they are wrong. Two specific failures:

1. **Unstratified split.** Severe (class 3) is ~5.3% of APTOS. A plain random
   split can hand you a validation set where that class is badly
   under-represented, so the validation metric you select your model on is
   mostly noise for the class that matters most clinically.

2. **Inconsistent splits between runs.** If ResNet50 and EfficientNet-B3 are
   evaluated on different test sets, the Milestone 6 comparison table is
   meaningless. The fix is to compute the split ONCE, write it to CSV, and have
   every run read those CSVs.

So: seeded, stratified, written to disk, and verified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_COL = "split"


# ---------------------------------------------------------------------------
def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> pd.DataFrame:
    """Split into train/val/test, returning a copy with a ``split`` column.

    ``test_size`` and ``val_size`` are fractions of the WHOLE dataset, not of
    the remainder - so (0.15, 0.15) gives 70/15/15, which is what most people
    mean and what most code gets wrong.

    Done in two stages:
        1. hold out the test set from everything
        2. split the remainder into train and val, with val_size rescaled

    Parameters
    ----------
    stratify:
        Leave this True. It exists only so the notebook can demonstrate what
        goes wrong without it.
    """
    if not 0 < test_size < 1 or not 0 < val_size < 1:
        raise ValueError(f"test_size and val_size must be in (0,1); got {test_size}, {val_size}")
    if test_size + val_size >= 1.0:
        raise ValueError(f"test_size + val_size must be < 1; got {test_size + val_size}")

    counts = df[label_col].value_counts()
    if stratify and counts.min() < 3:
        raise ValueError(
            f"Class {counts.idxmin()} has only {counts.min()} sample(s) - too few to "
            "stratify into three splits. Merge or drop that class first."
        )

    strat = df[label_col] if stratify else None

    # Stage 1: carve off the test set.
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True, stratify=strat
    )

    # Stage 2: val_size is a fraction of the ORIGINAL df, so rescale it to the
    # size of what's left. 0.15 of 100 == 0.1765 of the remaining 85.
    val_relative = val_size / (1.0 - test_size)
    strat_tv = train_val[label_col] if stratify else None
    train, val = train_test_split(
        train_val, test_size=val_relative, random_state=seed, shuffle=True, stratify=strat_tv
    )

    out = df.copy()
    out[SPLIT_COL] = ""
    out.loc[train.index, SPLIT_COL] = "train"
    out.loc[val.index, SPLIT_COL] = "val"
    out.loc[test.index, SPLIT_COL] = "test"

    if (out[SPLIT_COL] == "").any():
        raise RuntimeError("Some rows were not assigned to a split - this is a bug.")

    return out


# ---------------------------------------------------------------------------
def split_from_config(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Run :func:`stratified_split` using the project config."""
    sp = cfg["split"]
    return stratified_split(
        df,
        label_col=cfg["dataset"]["label_col"],
        test_size=sp["test_size"],
        val_size=sp["val_size"],
        seed=cfg["project"]["seed"],
        stratify=sp.get("stratify", True),
    )


# ---------------------------------------------------------------------------
def split_distribution(
    df: pd.DataFrame,
    label_col: str,
    class_names: Dict[int, str] | List[str],
) -> pd.DataFrame:
    """Per-split, per-class counts and percentages, plus the percentage drift.

    ``drift_pp`` is the largest difference, in percentage points, between a
    split's share of a class and the overall share. Under stratification it
    should be well under 1 pp; anything above ~2 pp means stratification did
    not work.
    """
    names = ({int(k): v for k, v in class_names.items()}
             if isinstance(class_names, dict) else dict(enumerate(class_names)))

    rows = []
    overall = df[label_col].value_counts(normalize=True)

    for label in sorted(names):
        row: Dict[str, Any] = {"label": label, "class_name": names[label]}
        shares = []
        for split in ("train", "val", "test"):
            subset = df[df[SPLIT_COL] == split]
            n = int((subset[label_col] == label).sum())
            pct = (n / len(subset) * 100) if len(subset) else 0.0
            row[f"{split}_n"] = n
            row[f"{split}_pct"] = round(pct, 2)
            shares.append(pct)
        row["overall_pct"] = round(float(overall.get(label, 0.0)) * 100, 2)
        row["drift_pp"] = round(max(abs(s - row["overall_pct"]) for s in shares), 2)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def verify_split(df: pd.DataFrame, id_col: str, label_col: str) -> List[str]:
    """Return a list of problems. Empty list means the split is sound.

    Checks, in order of how badly each would corrupt your results:
      1. leakage  - an id in more than one split
      2. coverage - an id in no split, or ids lost/duplicated
      3. emptiness - a split with no samples
      4. class presence - a class missing entirely from a split
    """
    problems: List[str] = []

    splits = {s: set(df.loc[df[SPLIT_COL] == s, id_col]) for s in ("train", "val", "test")}

    # 1. Leakage between every pair.
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = splits[a] & splits[b]
        if overlap:
            problems.append(
                f"LEAKAGE: {len(overlap)} id(s) appear in both {a} and {b}, "
                f"e.g. {sorted(overlap)[:3]}"
            )

    # 2. Coverage.
    total = sum(len(s) for s in splits.values())
    if total != len(df):
        problems.append(f"COVERAGE: splits hold {total} ids but the dataframe has {len(df)}")
    if df[id_col].duplicated().any():
        problems.append("COVERAGE: duplicate ids in the dataframe")

    # 3. Empty splits.
    for name, ids in splits.items():
        if not ids:
            problems.append(f"EMPTY: the {name} split has no samples")

    # 4. Every class present in every split.
    for split in ("train", "val", "test"):
        subset = df[df[SPLIT_COL] == split]
        present = set(subset[label_col].unique())
        missing = set(df[label_col].unique()) - present
        if missing:
            problems.append(f"MISSING CLASS: {split} split has no samples of class {sorted(missing)}")

    return problems


# ---------------------------------------------------------------------------
def write_splits(df: pd.DataFrame, splits_dir: Path) -> Dict[str, Path]:
    """Write train.csv / val.csv / test.csv plus a combined all_splits.csv.

    Returns a mapping of split name -> path. These CSVs are small, so COMMIT
    them to git: they are the exact data partition behind your reported numbers,
    and an examiner should be able to reproduce it without rerunning anything.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    for split in ("train", "val", "test"):
        subset = df[df[SPLIT_COL] == split].drop(columns=[SPLIT_COL]).reset_index(drop=True)
        path = splits_dir / f"{split}.csv"
        subset.to_csv(path, index=False)
        written[split] = path

    combined = splits_dir / "all_splits.csv"
    df.reset_index(drop=True).to_csv(combined, index=False)
    written["all"] = combined
    return written


def load_split(splits_dir: Path, split: str) -> pd.DataFrame:
    """Read one split CSV back. Raises a helpful error if Milestone 3 was skipped."""
    path = Path(splits_dir) / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Split file missing: {path}\n"
            "Run Milestone 3 first:  python -m scripts.m3_split"
        )
    return pd.read_csv(path)


def split_fingerprint(df: pd.DataFrame, id_col: str) -> str:
    """Short hash of the id->split assignment.

    Every training run records this. If two runs show different fingerprints,
    they were trained on different data and must not be compared - which is
    much easier to notice as a mismatched hash than by reading row counts.
    """
    import hashlib

    payload = "|".join(
        f"{row[id_col]}:{row[SPLIT_COL]}"
        for _, row in df.sort_values(id_col).iterrows()
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
