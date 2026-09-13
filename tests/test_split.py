"""
Unit tests for the Milestone 3 split and the Milestone 6 metrics.
File location: <project_root>/tests/test_split.py

Run:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.split import (
    SPLIT_COL,
    split_distribution,
    split_fingerprint,
    stratified_split,
    verify_split,
    write_splits,
)
from src.models.metrics import (
    compute_all_metrics,
    normalised_confusion,
    quadratic_weighted_kappa,
)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


@pytest.fixture()
def aptos_like() -> pd.DataFrame:
    """3,662 rows with APTOS-like class proportions."""
    rng = np.random.default_rng(42)
    n = 3662
    labels = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.493, 0.101, 0.272, 0.053, 0.081])
    return pd.DataFrame({"id_code": [f"id{i:05d}" for i in range(n)], "diagnosis": labels})


# ---------------------------------------------------------------------------
# Split mechanics
# ---------------------------------------------------------------------------
def test_split_sizes_are_fractions_of_the_whole(aptos_like):
    out = stratified_split(aptos_like, "diagnosis", test_size=0.15, val_size=0.15, seed=42)
    n = len(aptos_like)
    counts = out[SPLIT_COL].value_counts()
    # val_size and test_size are fractions of the WHOLE dataset, not of the
    # remainder - the classic off-by-a-rescale bug.
    assert abs(counts["test"] / n - 0.15) < 0.01
    assert abs(counts["val"] / n - 0.15) < 0.01
    assert abs(counts["train"] / n - 0.70) < 0.01
    assert counts.sum() == n


def test_split_has_no_leakage(aptos_like):
    out = stratified_split(aptos_like, "diagnosis", seed=42)
    assert verify_split(out, "id_code", "diagnosis") == []


def test_split_is_reproducible(aptos_like):
    a = stratified_split(aptos_like, "diagnosis", seed=42)
    b = stratified_split(aptos_like, "diagnosis", seed=42)
    pd.testing.assert_series_equal(a[SPLIT_COL], b[SPLIT_COL])
    assert split_fingerprint(a, "id_code") == split_fingerprint(b, "id_code")


def test_different_seed_gives_different_split(aptos_like):
    a = stratified_split(aptos_like, "diagnosis", seed=42)
    b = stratified_split(aptos_like, "diagnosis", seed=7)
    assert split_fingerprint(a, "id_code") != split_fingerprint(b, "id_code")


def test_stratification_preserves_class_shares(aptos_like):
    out = stratified_split(aptos_like, "diagnosis", seed=42, stratify=True)
    dist = split_distribution(out, "diagnosis", CLASS_NAMES)
    # Under stratification every class share should stay within ~1pp.
    assert dist["drift_pp"].max() < 1.0


def test_every_class_present_in_every_split(aptos_like):
    out = stratified_split(aptos_like, "diagnosis", seed=42)
    for split in ("train", "val", "test"):
        present = set(out.loc[out[SPLIT_COL] == split, "diagnosis"].unique())
        assert present == {0, 1, 2, 3, 4}, f"{split} is missing classes"


def test_rejects_impossible_fractions(aptos_like):
    with pytest.raises(ValueError, match="must be < 1"):
        stratified_split(aptos_like, "diagnosis", test_size=0.6, val_size=0.5)
    with pytest.raises(ValueError, match="must be in"):
        stratified_split(aptos_like, "diagnosis", test_size=0.0, val_size=0.15)


def test_rejects_class_too_small_to_stratify():
    df = pd.DataFrame({"id_code": ["a", "b", "c", "d"], "diagnosis": [0, 0, 0, 1]})
    with pytest.raises(ValueError, match="too few to stratify"):
        stratified_split(df, "diagnosis")


def test_verify_detects_injected_leakage(aptos_like):
    out = stratified_split(aptos_like, "diagnosis", seed=42)
    # Force one training id into the test split as well.
    train_id = out.loc[out[SPLIT_COL] == "train", "id_code"].iloc[0]
    leaked = pd.concat([out, out[out["id_code"] == train_id].assign(**{SPLIT_COL: "test"})])
    problems = verify_split(leaked, "id_code", "diagnosis")
    assert any("LEAKAGE" in p for p in problems)


def test_write_splits_roundtrip(aptos_like, tmp_path):
    out = stratified_split(aptos_like, "diagnosis", seed=42)
    written = write_splits(out, tmp_path)
    total = 0
    for split in ("train", "val", "test"):
        df = pd.read_csv(written[split])
        assert SPLIT_COL not in df.columns      # split column is implied by the file
        assert list(df.columns) == ["id_code", "diagnosis"]
        total += len(df)
    assert total == len(aptos_like)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_qwk_perfect_and_constant():
    y = np.array([0, 0, 1, 2, 3, 4, 2, 1])
    assert quadratic_weighted_kappa(y, y) == pytest.approx(1.0)
    # A constant predictor must score 0 - this is exactly why QWK is used
    # instead of accuracy on a 49%-majority dataset.
    assert quadratic_weighted_kappa(y, np.zeros_like(y)) == pytest.approx(0.0)


def test_qwk_penalises_distant_errors_more():
    y = np.array([0, 0, 0, 0])
    near = np.array([1, 1, 1, 1])       # off by one
    far = np.array([4, 4, 4, 4])        # off by four
    # With a constant truth QWK is degenerate, so use a spread truth instead.
    y = np.array([0, 1, 2, 3, 4])
    near = np.array([0, 1, 2, 3, 3])
    far = np.array([0, 1, 2, 3, 0])
    assert quadratic_weighted_kappa(y, near) > quadratic_weighted_kappa(y, far)


def test_metrics_bundle_is_complete():
    rng = np.random.default_rng(0)
    y = rng.choice([0, 1, 2, 3, 4], size=200)
    pred = np.clip(y + rng.choice([-1, 0, 0, 1], size=200), 0, 4)
    probs = rng.random((200, 5))
    probs /= probs.sum(axis=1, keepdims=True)

    m = compute_all_metrics(y, pred, probs, CLASS_NAMES)
    for key in ("qwk", "accuracy", "balanced_accuracy", "f1_macro",
                "per_class", "confusion_matrix", "auc_macro"):
        assert key in m
    assert len(m["per_class"]) == 5
    assert np.array(m["confusion_matrix"]).shape == (5, 5)
    assert np.array(m["confusion_matrix"]).sum() == 200


def test_normalised_confusion_rows_sum_to_one():
    cm = [[10, 2, 0, 0, 0], [1, 5, 1, 0, 0], [0, 2, 8, 1, 0],
          [0, 0, 1, 3, 0], [0, 0, 0, 1, 4]]
    rows = normalised_confusion(cm).sum(axis=1)
    assert np.allclose(rows, 1.0)


def test_normalised_confusion_handles_empty_row():
    """A class with zero support must give a zero row, not NaN."""
    cm = [[5, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 3, 0, 0],
          [0, 0, 0, 0, 0], [0, 0, 0, 0, 2]]
    out = normalised_confusion(cm)
    assert not np.isnan(out).any()
    assert out[1].sum() == 0.0
