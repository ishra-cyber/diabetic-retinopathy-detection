"""
Unit tests for the Milestone 1 inspection helpers.
File location: <project_root>/tests/test_inspect.py

Run:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.inspect import (
    check_image_files,
    class_distribution,
    find_duplicate_ids,
    load_labels,
    pick_samples_per_class,
    suggested_class_weights,
)

CLASS_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}


@pytest.fixture()
def toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"id_code": [f"i{i}" for i in range(10)],
         "diagnosis": [0, 0, 0, 0, 0, 1, 1, 2, 3, 4]}
    )


def test_class_distribution_covers_all_classes(toy_df):
    dist = class_distribution(toy_df, "diagnosis", CLASS_NAMES)
    assert list(dist["label"]) == [0, 1, 2, 3, 4]
    assert dist["count"].sum() == 10
    assert dist.loc[dist["label"] == 0, "count"].item() == 5
    assert pytest.approx(dist["percent"].sum(), abs=0.01) == 100.0


def test_class_weights_are_inverse_frequency(toy_df):
    w = suggested_class_weights(toy_df, "diagnosis", 5)
    assert w.shape == (5,)
    # rarest classes must get the largest weights
    assert w[3] > w[1] > w[0]
    assert pytest.approx(w.mean(), abs=1e-6) == 1.0


def test_duplicate_detection():
    df = pd.DataFrame({"id_code": ["a", "b", "a"], "diagnosis": [0, 1, 0]})
    assert find_duplicate_ids(df, "id_code") == ["a"]


def test_sample_picker_is_deterministic(toy_df):
    a = pick_samples_per_class(toy_df, "id_code", "diagnosis", per_class=2, seed=42)
    b = pick_samples_per_class(toy_df, "id_code", "diagnosis", per_class=2, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_load_labels_rejects_missing_column(tmp_path):
    csv = tmp_path / "train.csv"
    pd.DataFrame({"wrong": [1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="missing expected column"):
        load_labels(csv, "id_code", "diagnosis")


def test_check_image_files_reports_missing_and_orphans(tmp_path):
    import cv2

    images = tmp_path / "train_images"
    images.mkdir()
    for name in ("a", "orphan"):
        cv2.imwrite(str(images / f"{name}.png"), np.zeros((8, 8, 3), dtype=np.uint8))

    df = pd.DataFrame({"id_code": ["a", "b"], "diagnosis": [0, 1]})
    missing, orphans = check_image_files(df, images, "id_code", ".png")
    assert missing == ["b"]
    assert orphans == ["orphan"]
