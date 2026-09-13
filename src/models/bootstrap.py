"""
Bootstrap confidence intervals and paired model comparison.
File location: <project_root>/src/models/bootstrap.py

Why this exists
---------------
Three architectures landing within 0.005 QWK of each other is not a ranking, it
is a tie. But "that looks like noise" is an assertion. This module measures it.

Method
------
**Confidence interval for one model.** Resample the test set with replacement
1,000 times, recompute the metric on each resample, and report the 2.5th and
97.5th percentiles. That interval answers: "if I had drawn a different sample of
550 patients from the same population, what range of scores would I have seen?"

**Comparing two models.** The naive approach - two separate CIs, check if they
overlap - is needlessly conservative, because both models are evaluated on the
*same* images. A **paired** bootstrap resamples the image indices ONCE per
iteration and scores both models on that same resample, so the shared
sample-difficulty cancels out. The result is the distribution of the
*difference*, which is what the question is actually about.

``prob_a_better`` is the fraction of bootstrap resamples where A beat B. Read it
as strength of evidence, not as a p-value: 0.52 means a coin flip, 0.99 means a
real difference.

Everything here is pure NumPy and scikit-learn - no GPU, no PyTorch, runs in
seconds on 550 samples.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.models.metrics import quadratic_weighted_kappa

Metric = Callable[[np.ndarray, np.ndarray], float]


# ---------------------------------------------------------------------------
def _qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return quadratic_weighted_kappa(y_true, y_pred, num_classes=5)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-class recall, skipping classes absent from this resample."""
    recalls = []
    for label in np.unique(y_true):
        mask = y_true == label
        recalls.append(float((y_pred[mask] == label).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


METRICS: Dict[str, Metric] = {
    "qwk": _qwk,
    "accuracy": _accuracy,
    "balanced_accuracy": _balanced_accuracy,
}


# ---------------------------------------------------------------------------
def bootstrap_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    metric: str | Metric = "qwk",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Percentile bootstrap CI for one model's metric.

    Returns ``{point, mean, std, ci_low, ci_high, n_boot, n_valid, alpha}``.

    ``point`` is the metric on the real data - report that as your result. The
    interval describes its uncertainty; it is not a range of plausible "true"
    values in a strict frequentist sense, but it is the standard way to express
    sampling uncertainty and examiners expect it.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("empty input")

    fn = METRICS[metric] if isinstance(metric, str) else metric
    rng = np.random.default_rng(seed)
    n = y_true.size

    scores = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)          # resample WITH replacement
        # A resample containing a single class makes QWK undefined; mark it NaN
        # and exclude it rather than letting a 0.0 drag the interval down.
        sample_true = y_true[idx]
        scores[b] = fn(sample_true, y_pred[idx]) if np.unique(sample_true).size > 1 else np.nan

    valid = scores[~np.isnan(scores)]
    if valid.size < n_boot * 0.5:
        raise ValueError(
            f"Only {valid.size}/{n_boot} bootstrap samples were valid - the split may "
            "be too small or too imbalanced for a meaningful interval."
        )

    return {
        "metric": metric if isinstance(metric, str) else fn.__name__,
        "point": round(float(fn(y_true, y_pred)), 4),
        "mean": round(float(valid.mean()), 4),
        "std": round(float(valid.std(ddof=1)), 4),
        "ci_low": round(float(np.percentile(valid, 100 * alpha / 2)), 4),
        "ci_high": round(float(np.percentile(valid, 100 * (1 - alpha / 2))), 4),
        "n_boot": n_boot,
        "n_valid": int(valid.size),
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
def paired_bootstrap_difference(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    y_pred_b: Sequence[int],
    metric: str | Metric = "qwk",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap of (metric of A) − (metric of B) on the same resamples.

    If ``ci_low`` and ``ci_high`` straddle zero, the two models are
    indistinguishable on this test set - which is the expected outcome when
    architectures differ by less than the sampling noise.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_a = np.asarray(y_pred_a, dtype=int)
    y_pred_b = np.asarray(y_pred_b, dtype=int)

    fn = METRICS[metric] if isinstance(metric, str) else metric
    rng = np.random.default_rng(seed)
    n = y_true.size

    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)          # ONE resample, both models
        sample_true = y_true[idx]
        if np.unique(sample_true).size <= 1:
            diffs[b] = np.nan
            continue
        diffs[b] = fn(sample_true, y_pred_a[idx]) - fn(sample_true, y_pred_b[idx])

    valid = diffs[~np.isnan(diffs)]
    ci_low = float(np.percentile(valid, 100 * alpha / 2))
    ci_high = float(np.percentile(valid, 100 * (1 - alpha / 2)))

    return {
        "metric": metric if isinstance(metric, str) else fn.__name__,
        "observed_difference": round(float(fn(y_true, y_pred_a) - fn(y_true, y_pred_b)), 4),
        "mean_difference": round(float(valid.mean()), 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        # Strength of evidence, not a p-value.
        "prob_a_better": round(float((valid > 0).mean()), 4),
        "significant": bool(ci_low > 0 or ci_high < 0),
        "n_boot": n_boot,
        "n_valid": int(valid.size),
    }


# ---------------------------------------------------------------------------
def bootstrap_all_runs(
    runs: Dict[str, Tuple[Sequence[int], Sequence[int]]],
    metrics: Sequence[str] = ("qwk", "accuracy", "balanced_accuracy"),
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """CI table for several runs.

    ``runs`` maps a display name to ``(y_true, y_pred)``.
    """
    rows = []
    for name, (y_true, y_pred) in runs.items():
        row: Dict[str, Any] = {"model": name}
        for metric in metrics:
            ci = bootstrap_ci(y_true, y_pred, metric=metric, n_boot=n_boot, seed=seed)
            row[metric] = ci["point"]
            row[f"{metric}_ci"] = f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
            row[f"{metric}_lo"] = ci["ci_low"]
            row[f"{metric}_hi"] = ci["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(metrics[0], ascending=False).reset_index(drop=True)


def pairwise_comparison(
    runs: Dict[str, Tuple[Sequence[int], Sequence[int]]],
    metric: str = "qwk",
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Every pair of models, paired-bootstrapped."""
    names = list(runs)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            y_true = np.asarray(runs[a][0], dtype=int)
            if not np.array_equal(y_true, np.asarray(runs[b][0], dtype=int)):
                raise ValueError(
                    f"'{a}' and '{b}' have different ground truth - they were not "
                    "evaluated on the same split, so they cannot be compared."
                )
            result = paired_bootstrap_difference(
                y_true, runs[a][1], runs[b][1], metric=metric, n_boot=n_boot, seed=seed
            )
            rows.append({
                "model_a": a, "model_b": b,
                "difference": result["observed_difference"],
                "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                "prob_a_better": result["prob_a_better"],
                "distinguishable": result["significant"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def plot_forest(
    ci_table: pd.DataFrame,
    metric: str,
    out_path,
    title: str = "",
) -> None:
    """Forest plot: point estimate with a 95% CI bar per model.

    The visual argument for "these models are tied": heavily overlapping bars.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ci_table["model"].tolist()
    points = ci_table[metric].to_numpy(dtype=float)
    lows = ci_table[f"{metric}_lo"].to_numpy(dtype=float)
    highs = ci_table[f"{metric}_hi"].to_numpy(dtype=float)
    y = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(8, 0.9 * len(models) + 2.2))
    ax.errorbar(points, y, xerr=[points - lows, highs - points],
                fmt="o", color="#4C72B0", ecolor="#4C72B0",
                capsize=5, markersize=7, linewidth=1.8)

    for i, (p, lo, hi) in enumerate(zip(points, lows, highs)):
        ax.text(hi + 0.004, i, f"{p:.4f}  [{lo:.3f}, {hi:.3f}]",
                va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric.upper()}  (point estimate with 95% bootstrap CI)")
    ax.set_title(title or f"{metric.upper()} with 95% confidence intervals")
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)

    span = highs.max() - lows.min()
    ax.set_xlim(lows.min() - 0.02 * span, highs.max() + 0.16 * span)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def interpret(ci_table: pd.DataFrame, pairs: pd.DataFrame, metric: str = "qwk") -> str:
    """A paragraph, written from the numbers, ready for the report."""
    best = ci_table.iloc[0]
    worst = ci_table.iloc[-1]
    any_distinguishable = bool(pairs["distinguishable"].any())

    lines = [
        f"{best['model']} achieved the highest {metric.upper()} at {best[metric]:.4f} "
        f"(95% CI {best[f'{metric}_ci']}).",
        f"The spread across architectures was "
        f"{best[metric] - worst[metric]:.4f} {metric.upper()}.",
    ]
    if any_distinguishable:
        sig = pairs[pairs["distinguishable"]]
        for _, row in sig.iterrows():
            lines.append(
                f"The difference between {row['model_a']} and {row['model_b']} "
                f"({row['difference']:+.4f}, 95% CI [{row['ci_low']:.3f}, "
                f"{row['ci_high']:.3f}]) excludes zero, so these two are "
                f"distinguishable on this test set."
            )
    else:
        lines.append(
            "In paired bootstrap comparisons every pairwise confidence interval "
            "included zero, so no architecture is distinguishable from another on "
            "this test set. The models should be reported as performing "
            "equivalently rather than ranked."
        )
    return " ".join(lines)
