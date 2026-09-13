"""
Evaluation metrics for ordinal 5-class DR grading.
File location: <project_root>/src/models/metrics.py

The metric that matters
----------------------
DR grading is **ordinal**: 0 < 1 < 2 < 3 < 4. Predicting 4 when the truth is 0
is a far worse error than predicting 1. Plain accuracy treats both as equally
wrong, and on a dataset that is 49.3% class 0 a degenerate "always No DR"
classifier scores 49.3% accuracy - which looks respectable and is worthless.

**Quadratic Weighted Kappa (QWK)** penalises errors by the SQUARE of their
distance, and is 0 for a constant predictor. It is also the official APTOS
competition metric, so your numbers are comparable to published work. Model
selection, early stopping and the headline result all use QWK.

Everything else here is supporting evidence for the report:
  - per-class precision/recall/F1 (recall on class 3/4 is the clinically
    important one: missing severe disease is the dangerous error)
  - confusion matrix, raw and row-normalised
  - one-vs-rest ROC-AUC
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
def quadratic_weighted_kappa(y_true: Sequence[int], y_pred: Sequence[int],
                             num_classes: int = 5) -> float:
    """QWK. 1.0 = perfect, 0.0 = no better than chance, negative = worse."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return float("nan")
    # labels= pins the full 0..4 range so a batch missing a class still scores
    # against the same scale.
    return float(cohen_kappa_score(
        y_true, y_pred, weights="quadratic", labels=list(range(num_classes))
    ))


# ---------------------------------------------------------------------------
def per_class_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """Precision, recall, F1 and support for each class."""
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return [
        {
            "label": label,
            "class_name": class_names[label],
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    ]


# ---------------------------------------------------------------------------
def roc_auc_ovr(
    y_true: Sequence[int],
    probabilities: np.ndarray,
    num_classes: int = 5,
) -> Dict[str, float]:
    """One-vs-rest ROC-AUC: macro, weighted, and per class.

    ROC-AUC needs at least one positive AND one negative example of a class, so
    any class absent from ``y_true`` is reported as NaN rather than crashing -
    which matters on small validation folds.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    out: Dict[str, float] = {}

    present = sorted(set(y_true.tolist()))
    for label in range(num_classes):
        key = f"auc_class_{label}"
        if label not in present or len(present) < 2:
            out[key] = float("nan")
            continue
        binary = (y_true == label).astype(int)
        if binary.min() == binary.max():       # all-positive or all-negative
            out[key] = float("nan")
            continue
        out[key] = round(float(roc_auc_score(binary, probabilities[:, label])), 4)

    per_class = [v for v in out.values() if not np.isnan(v)]
    out["auc_macro"] = round(float(np.mean(per_class)), 4) if per_class else float("nan")

    # sklearn's own multiclass computation, when the data allows it
    try:
        if len(present) == num_classes:
            out["auc_weighted"] = round(float(roc_auc_score(
                y_true, probabilities, multi_class="ovr", average="weighted",
                labels=list(range(num_classes)),
            )), 4)
        else:
            out["auc_weighted"] = float("nan")
    except ValueError:
        out["auc_weighted"] = float("nan")

    return out


# ---------------------------------------------------------------------------
def compute_all_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: np.ndarray | None,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """Everything at once. This dict is what gets written to metrics.json."""
    num_classes = len(class_names)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    result: Dict[str, Any] = {
        "n_samples": int(y_true.size),
        "qwk": round(quadratic_weighted_kappa(y_true, y_pred, num_classes), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        # Balanced accuracy = mean per-class recall. On an imbalanced dataset
        # the gap between accuracy and balanced accuracy tells you how much the
        # model is coasting on the majority class.
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, labels=list(range(num_classes)),
                                         average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, labels=list(range(num_classes)),
                                            average="weighted", zero_division=0)), 4),
        "per_class": per_class_metrics(y_true, y_pred, class_names),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(num_classes))
        ).tolist(),
    }

    if probabilities is not None:
        result.update(roc_auc_ovr(y_true, probabilities, num_classes))

    return result


# ---------------------------------------------------------------------------
def normalised_confusion(cm: Sequence[Sequence[int]]) -> np.ndarray:
    """Row-normalise a confusion matrix (each row sums to 1 = per-class recall).

    Always show BOTH versions in the report. The raw matrix hides class 3
    entirely because it has so few samples; the normalised one reveals that the
    model may be getting most of them wrong.
    """
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    return out


def summarise_for_console(metrics: Dict[str, Any]) -> str:
    """Compact human-readable summary, for the end of a training run."""
    lines = [
        f"  samples            : {metrics['n_samples']}",
        f"  QWK                : {metrics['qwk']:.4f}   <-- headline metric",
        f"  accuracy           : {metrics['accuracy']:.4f}",
        f"  balanced accuracy  : {metrics['balanced_accuracy']:.4f}",
        f"  macro F1           : {metrics['f1_macro']:.4f}",
    ]
    if "auc_macro" in metrics and not np.isnan(metrics["auc_macro"]):
        lines.append(f"  macro ROC-AUC      : {metrics['auc_macro']:.4f}")
    lines.append("  per-class recall   :")
    for row in metrics["per_class"]:
        lines.append(f"      {row['label']} {row['class_name']:<18} "
                     f"recall {row['recall']:.3f}  (n={row['support']})")
    return "\n".join(lines)
