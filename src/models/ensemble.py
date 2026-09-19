"""
Ensembling trained runs by averaging their probability outputs.
File location: <project_root>/src/models/ensemble.py

Why this exists
---------------
Milestone 6 established, with a paired bootstrap, that DenseNet121, ResNet50 and
EfficientNet-B3 are *statistically indistinguishable* on this test set - every
pairwise confidence interval except one straddles zero. That is the textbook
signal to stop ranking them and start combining them: models that score the same
but disagree on individual images carry complementary information, and averaging
their probabilities recovers some of it.

The cheap part
--------------
No GPU, no retraining, no new inference. Milestone 6 already wrote
``<run>/{split}_predictions.npz`` for every run, containing the softmax output
for every image. Ensembling is an average over files that already exist on disk.

Two combination rules
---------------------
``mean``   arithmetic mean of probabilities. The default. Forgiving: one model
           being confidently wrong is diluted rather than decisive.
``gmean``  geometric mean (equivalently, the mean of log-probabilities). Harsher:
           a probability near zero from any single model vetoes that class. Worth
           reporting as a sensitivity check; it rarely changes the headline.

Both are computed per image and then re-normalised to sum to 1, so the output is
still a valid distribution and can be fed to the calibration code unchanged.

Honesty note
------------
An ensemble is a different model from any of its members, so its test score is a
new number, not a corrected version of an old one. It is reported as its own row
in the results table - never substituted for the single-model result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
def load_run_predictions(run_dir: Path, split: str = "test") -> Dict[str, np.ndarray]:
    """Read one run's saved ``{split}_predictions.npz``.

    Returns ``{y_true, y_pred, probabilities, ids}``. Raises if the file is
    missing, because a silent skip would quietly build an ensemble of two models
    while the report claims three.
    """
    path = Path(run_dir) / f"{split}_predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"Run `python -m scripts.m6_evaluate --split {split}` first - the "
            "ensemble is built from saved predictions, not from checkpoints."
        )
    data = np.load(path, allow_pickle=True)
    # Validation predictions written by the Milestone 4/5 training loop predate
    # the id column that `evaluate_run` saves, so `ids` is optional here. When it
    # is missing the members are aligned by row position instead - see
    # `align_members`, which checks that this is safe rather than assuming it.
    ids = np.asarray(data["ids"], dtype=object) if "ids" in data.files else None
    return {
        "y_true": data["y_true"].astype(int),
        "y_pred": data["y_pred"].astype(int),
        "probabilities": data["probabilities"].astype(float),
        "ids": ids,
    }


def load_members(
    run_dirs: Sequence[Path],
    split: str = "test",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load several runs, keyed by run-folder name."""
    return {Path(d).name: load_run_predictions(d, split) for d in run_dirs}


# ---------------------------------------------------------------------------
def _align_by_position(
    members: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Fallback alignment for prediction files saved without ids."""
    reference_name = next(iter(members))
    y_true_ref = members[reference_name]["y_true"]

    for name, member in members.items():
        if member["y_true"].shape != y_true_ref.shape:
            raise ValueError(
                f"Run '{name}' has {member['y_true'].size} predictions but "
                f"'{reference_name}' has {y_true_ref.size}. These are not the same "
                "split and cannot be ensembled."
            )
        if not np.array_equal(member["y_true"], y_true_ref):
            raise ValueError(
                f"Run '{name}' and '{reference_name}' disagree about the ground-truth "
                "labels row by row. Without ids there is no way to realign them - "
                f"re-run `python -m scripts.m6_evaluate` so ids are saved."
            )

    ids = np.arange(y_true_ref.size, dtype=object)   # positional stand-in
    return ids, y_true_ref, {n: m["probabilities"] for n, m in members.items()}


# ---------------------------------------------------------------------------
def align_members(
    members: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Put every member's rows in the same order and verify they agree.

    ``predict_split`` uses ``shuffle=False``, so in practice the rows already
    line up. "In practice" is not a guarantee: a member evaluated after a split
    was regenerated would line up by position but not by patient, and the
    resulting ensemble would be silently meaningless. So the ids are sorted and
    compared explicitly, and the ground-truth vectors must match exactly
    afterwards.

    Some older prediction files carry no ids (see ``load_run_predictions``). In
    that case alignment falls back to row position, which is sound here because
    every run was evaluated with ``shuffle=False`` over the same split file - but
    it is only *accepted* after the ground-truth vectors are confirmed identical
    element by element. If two runs disagree about the label of row 17, they were
    not scored over the same rows, and the function refuses rather than producing
    a plausible-looking average of unrelated images.

    Returns ``(ids, y_true, {name: probabilities})`` in the shared order.
    """
    if not members:
        raise ValueError("No ensemble members given.")

    if any(m.get("ids") is None for m in members.values()):
        return _align_by_position(members)

    reference_name = next(iter(members))
    reference_ids = np.sort(members[reference_name]["ids"])

    aligned: Dict[str, np.ndarray] = {}
    y_true_ref: Optional[np.ndarray] = None

    for name, member in members.items():
        ids = member["ids"]
        if set(ids.tolist()) != set(reference_ids.tolist()):
            raise ValueError(
                f"Run '{name}' was evaluated on a different set of images than "
                f"'{reference_name}'. These runs cannot be ensembled - re-run "
                "Milestone 6 for both on the same split."
            )
        order = np.argsort(ids)
        aligned[name] = member["probabilities"][order]
        y_true = member["y_true"][order]

        if y_true_ref is None:
            y_true_ref = y_true
        elif not np.array_equal(y_true, y_true_ref):
            raise ValueError(
                f"Run '{name}' disagrees with '{reference_name}' about the ground "
                "truth labels. One of them was evaluated against a stale split."
            )

    return reference_ids, y_true_ref, aligned


# ---------------------------------------------------------------------------
def combine(
    probabilities: Sequence[np.ndarray],
    method: str = "mean",
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Combine member probability matrices into one (n_samples, n_classes).

    ``weights`` default to equal. Weighting members by their validation score is
    tempting and usually pointless with three models this close together; if you
    do it, fit the weights on validation and say so in the report.
    """
    stack = np.stack([np.asarray(p, dtype=float) for p in probabilities], axis=0)
    if stack.ndim != 3:
        raise ValueError(f"Expected (n_models, n_samples, n_classes), got {stack.shape}")

    if weights is None:
        w = np.ones(stack.shape[0], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != stack.shape[0]:
            raise ValueError(f"{w.shape[0]} weights for {stack.shape[0]} models")
    w = w / w.sum()
    w = w.reshape(-1, 1, 1)

    if method == "mean":
        combined = (stack * w).sum(axis=0)
    elif method == "gmean":
        # Mean of logs, then exponentiate. The floor keeps log() finite when a
        # member outputs a hard zero after float32 underflow.
        combined = np.exp((np.log(np.clip(stack, 1e-12, 1.0)) * w).sum(axis=0))
    else:
        raise ValueError(f"Unknown method '{method}' (use 'mean' or 'gmean')")

    return combined / combined.sum(axis=1, keepdims=True)


def ensemble_from_members(
    members: Dict[str, Dict[str, np.ndarray]],
    method: str = "mean",
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """End-to-end: aligned members in, ensemble predictions out.

    Returns the same shape of dict as a single run, so every downstream
    function - metrics, bootstrap, figures, calibration - accepts it unchanged.
    """
    ids, y_true, aligned = align_members(members)
    names = list(aligned)
    probabilities = combine([aligned[n] for n in names], method=method, weights=weights)

    return {
        "members": names,
        "method": method,
        "ids": ids,
        "y_true": y_true,
        "y_pred": probabilities.argmax(axis=1),
        "probabilities": probabilities,
    }


# ---------------------------------------------------------------------------
def disagreement_summary(
    members: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    """How often do the members actually disagree?

    This is the number that justifies the ensemble existing. If all three models
    agree on 95% of images, an ensemble can only ever change the remaining 5%,
    and the honest conclusion in the report is "ensembling was not worth the
    extra inference cost". If they disagree on a quarter of images, there is
    real complementary signal to recover.
    """
    ids, y_true, aligned = align_members(members)
    preds = np.stack([p.argmax(axis=1) for p in aligned.values()], axis=0)

    unanimous = (preds == preds[0]).all(axis=0)
    n = preds.shape[1]

    # Of the images where they disagree, how often is at least one member right?
    contested = ~unanimous
    recoverable = (preds[:, contested] == y_true[contested]).any(axis=0)

    return {
        "n_samples": int(n),
        "n_models": int(preds.shape[0]),
        "unanimous": int(unanimous.sum()),
        "unanimous_pct": round(float(unanimous.mean() * 100), 2),
        "contested": int(contested.sum()),
        "contested_pct": round(float(contested.mean() * 100), 2),
        "unanimous_correct_pct": round(
            float((preds[0][unanimous] == y_true[unanimous]).mean() * 100), 2
        ) if unanimous.any() else float("nan"),
        "contested_recoverable_pct": round(float(recoverable.mean() * 100), 2)
        if contested.any() else float("nan"),
    }


def member_agreement_matrix(
    members: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[List[str], np.ndarray]:
    """Pairwise "fraction of images where these two predict the same grade"."""
    _, _, aligned = align_members(members)
    names = list(aligned)
    preds = {n: aligned[n].argmax(axis=1) for n in names}

    matrix = np.eye(len(names), dtype=float)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                agreement = float((preds[a] == preds[b]).mean())
                matrix[i, j] = matrix[j, i] = round(agreement, 4)
    return names, matrix
