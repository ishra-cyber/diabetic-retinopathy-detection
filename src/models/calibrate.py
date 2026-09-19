"""
Confidence calibration: temperature scaling, ECE, reliability diagrams.
File location: <project_root>/src/models/calibrate.py

The problem this fixes
----------------------
The Milestone 7 Grad-CAM table contains predictions with confidence
0.9999921321868896. No 300x300 fundus classifier trained on 2,562 images knows
anything to seven decimal places. That number is an artefact of a softmax
trained to convergence with cross-entropy: modern networks are *accurate* but
badly *calibrated*, and they systematically overstate confidence (Guo et al.,
"On Calibration of Modern Neural Networks", ICML 2017).

This matters here more than in a Kaggle notebook, because the Streamlit app
prints that number to a user who is being invited to read it as "how sure the
system is". A screening-support tool that says 99.99% when it is right 80% of
the time is misleading in exactly the direction that causes harm.

Temperature scaling
-------------------
The fix is one parameter. Divide the logits by a scalar T before the softmax:

    p_calibrated = softmax(z / T)

T > 1 softens the distribution (fixes overconfidence), T < 1 sharpens it, T = 1
changes nothing. Crucially, dividing by a positive scalar cannot change which
class has the largest value, so **accuracy, QWK and every other
threshold-free metric are mathematically unchanged**. Only the confidence
numbers move. It is the rare fix with no trade-off to declare.

T is fitted by minimising negative log-likelihood on the **validation** split.
Fitting it on test would be tuning on the held-out data.

Working from saved probabilities
--------------------------------
Milestone 6 saved softmax probabilities, not raw logits - and temperature
scaling is defined on logits. It still works without re-running the model,
because softmax is shift-invariant:

    log p = z - logsumexp(z)                     (definition of softmax)
    softmax(log p / T) = softmax((z - c) / T)
                       = softmax(z/T - c/T)
                       = softmax(z/T)            (constant shift cancels)

So ``log(p)`` is a valid stand-in for the logits. No GPU, no checkpoint reload.

What gets reported
------------------
``ECE``    Expected Calibration Error - bin predictions by confidence, and
           average |confidence - accuracy| over bins, weighted by bin size.
           0 is perfect. Report before and after.
``MCE``    the worst single bin. Sensitive to sparsely-populated bins, so read
           it alongside the counts.
``NLL``    what the fit actually minimises.
``Brier``  multi-class Brier score; a proper scoring rule, so it rewards both
           accuracy and honest uncertainty.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
def _as_log_probs(probabilities: np.ndarray) -> np.ndarray:
    """Stand-in logits, per the shift-invariance argument in the module docstring."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"Expected (n_samples, n_classes), got {p.shape}")
    return np.log(np.clip(p, EPS, 1.0))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Re-softmax the probabilities at temperature ``T``.

    ``T = 1.0`` returns the input unchanged (up to floating point).
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    scaled = _as_log_probs(probabilities) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)       # overflow guard only
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
def negative_log_likelihood(
    probabilities: np.ndarray,
    y_true: Sequence[int],
) -> float:
    """Mean NLL of the true class. Lower is better."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(y_true, dtype=int)
    return float(-np.log(np.clip(p[np.arange(y.size), y], EPS, 1.0)).mean())


def brier_score(probabilities: np.ndarray, y_true: Sequence[int]) -> float:
    """Multi-class Brier score: mean squared error against the one-hot target."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(y_true, dtype=int)
    onehot = np.zeros_like(p)
    onehot[np.arange(y.size), y] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


# ---------------------------------------------------------------------------
def fit_temperature(
    probabilities: np.ndarray,
    y_true: Sequence[int],
    lo: float = 0.05,
    hi: float = 10.0,
    coarse_steps: int = 400,
    refine_rounds: int = 4,
) -> Dict[str, Any]:
    """Find the T minimising validation NLL.

    Implemented as a log-spaced grid followed by a few rounds of local
    refinement rather than a gradient optimiser: it is one parameter over a
    bounded interval, this takes milliseconds, and it has no dependency on
    scipy's optimiser behaving the same way on the examiner's machine.

    **Fit this on validation only.** Passing the test split here would make
    every test number that follows contaminated.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(y_true, dtype=int)

    def nll(t: float) -> float:
        return negative_log_likelihood(apply_temperature(p, t), y)

    grid = np.geomspace(lo, hi, coarse_steps)
    scores = np.array([nll(float(t)) for t in grid])
    best_i = int(scores.argmin())
    best_t = float(grid[best_i])

    # Tighten the bracket around the winner and re-grid inside it.
    for _ in range(refine_rounds):
        left = float(grid[max(best_i - 1, 0)])
        right = float(grid[min(best_i + 1, grid.size - 1)])
        grid = np.linspace(left, right, 50)
        scores = np.array([nll(float(t)) for t in grid])
        best_i = int(scores.argmin())
        best_t = float(grid[best_i])

    return {
        "temperature": round(best_t, 4),
        "nll_before": round(nll(1.0), 4),
        "nll_after": round(float(scores[best_i]), 4),
        "at_bound": bool(best_t <= lo * 1.01 or best_t >= hi * 0.99),
    }


# ---------------------------------------------------------------------------
def calibration_bins(
    probabilities: np.ndarray,
    y_true: Sequence[int],
    n_bins: int = 15,
) -> Dict[str, np.ndarray]:
    """Bin predictions by top-class confidence.

    Returns per-bin edges, counts, mean confidence and empirical accuracy - the
    raw material for both the ECE number and the reliability diagram.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(y_true, dtype=int)

    confidence = p.max(axis=1)
    predicted = p.argmax(axis=1)
    correct = (predicted == y).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin by upper edge so a confidence of exactly 1.0 lands in the last bin.
    index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)

    counts = np.zeros(n_bins, dtype=int)
    mean_conf = np.full(n_bins, np.nan)
    accuracy = np.full(n_bins, np.nan)

    for b in range(n_bins):
        mask = index == b
        counts[b] = int(mask.sum())
        if counts[b]:
            mean_conf[b] = confidence[mask].mean()
            accuracy[b] = correct[mask].mean()

    return {
        "edges": edges,
        "counts": counts,
        "mean_confidence": mean_conf,
        "accuracy": accuracy,
    }


def calibration_metrics(
    probabilities: np.ndarray,
    y_true: Sequence[int],
    n_bins: int = 15,
) -> Dict[str, float]:
    """ECE, MCE, mean confidence, accuracy, NLL and Brier score in one dict."""
    bins = calibration_bins(probabilities, y_true, n_bins=n_bins)
    counts = bins["counts"]
    gaps = np.abs(bins["mean_confidence"] - bins["accuracy"])

    populated = counts > 0
    total = counts.sum()
    ece = float((counts[populated] * gaps[populated]).sum() / total) if total else float("nan")
    mce = float(np.nanmax(gaps[populated])) if populated.any() else float("nan")

    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(y_true, dtype=int)

    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "mean_confidence": round(float(p.max(axis=1).mean()), 4),
        "accuracy": round(float((p.argmax(axis=1) == y).mean()), 4),
        # Positive = overconfident, negative = underconfident. The sign is the
        # part that matters clinically.
        "overconfidence": round(
            float(p.max(axis=1).mean() - (p.argmax(axis=1) == y).mean()), 4
        ),
        "nll": round(negative_log_likelihood(p, y), 4),
        "brier": round(brier_score(p, y), 4),
        "n_bins": n_bins,
    }


def calibrate_and_report(
    val_probabilities: np.ndarray,
    val_y_true: Sequence[int],
    test_probabilities: Optional[np.ndarray] = None,
    test_y_true: Optional[Sequence[int]] = None,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """Fit T on validation, then report calibration before and after on both splits.

    The test numbers here are *applications* of a parameter chosen on
    validation, not a fit to test - which is the only way this stays honest.
    """
    fit = fit_temperature(val_probabilities, val_y_true)
    temperature = fit["temperature"]

    out: Dict[str, Any] = {
        "temperature": temperature,
        "fit": fit,
        "val_before": calibration_metrics(val_probabilities, val_y_true, n_bins),
        "val_after": calibration_metrics(
            apply_temperature(val_probabilities, temperature), val_y_true, n_bins),
    }

    if test_probabilities is not None and test_y_true is not None:
        out["test_before"] = calibration_metrics(test_probabilities, test_y_true, n_bins)
        out["test_after"] = calibration_metrics(
            apply_temperature(test_probabilities, temperature), test_y_true, n_bins)

    return out


# ---------------------------------------------------------------------------
def bootstrap_calibration_change(
    probabilities: np.ndarray,
    y_true: Sequence[int],
    temperature: float,
    n_bins: int = 15,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Is the change in ECE real, or is it binning noise?

    ECE is an estimate computed from ~15 bins over a few hundred images, and on
    a test split this size a change of +/-0.02 is easily sampling noise. Quoting
    "ECE improved from 0.046 to 0.054" as though the third decimal means
    something is exactly the kind of claim a viva question is made of.

    This bootstraps the *difference* (after - before) on the same resample, the
    same paired logic as ``src.models.bootstrap``: if the interval contains
    zero, the honest statement is that calibration was unchanged on this split.

    ``nll`` and ``brier`` deltas come along because they are proper scoring
    rules computed per-sample, with no binning - when ECE says "noise" and both
    of those improve, the improvement is real and ECE simply cannot resolve it.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(y_true, dtype=int)
    p_after = apply_temperature(p, temperature)

    rng = np.random.default_rng(seed)
    n = y.size
    deltas = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[b] = (calibration_metrics(p_after[idx], y[idx], n_bins)["ece"]
                     - calibration_metrics(p[idx], y[idx], n_bins)["ece"])

    observed = (calibration_metrics(p_after, y, n_bins)["ece"]
                - calibration_metrics(p, y, n_bins)["ece"])
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))

    return {
        "ece_delta": round(float(observed), 4),
        "ece_delta_ci_low": round(ci_low, 4),
        "ece_delta_ci_high": round(ci_high, 4),
        # True only when the interval sits entirely on one side of zero.
        "ece_change_real": bool(ci_low > 0 or ci_high < 0),
        "nll_delta": round(negative_log_likelihood(p_after, y)
                           - negative_log_likelihood(p, y), 4),
        "brier_delta": round(brier_score(p_after, y) - brier_score(p, y), 4),
        "n_boot": n_boot,
    }


# ---------------------------------------------------------------------------
def plot_reliability(
    probabilities_before: np.ndarray,
    probabilities_after: np.ndarray,
    y_true: Sequence[int],
    out_path,
    n_bins: int = 15,
    title: str = "",
    temperature: Optional[float] = None,
) -> None:
    """Reliability diagram, before and after, with the confidence histogram below.

    Reading it: points on the diagonal are perfectly calibrated. Points *below*
    the diagonal mean the model claims more confidence than it earns - which is
    what an uncalibrated network does, and what the right-hand panel should show
    being pulled back onto the line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Before (raw softmax)", probabilities_before, "#C44E52"),
        (f"After (T = {temperature:.3f})" if temperature else "After", 
         probabilities_after, "#4C72B0"),
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08, "wspace": 0.18},
    )

    for col, (label, probs, colour) in enumerate(panels):
        bins = calibration_bins(probs, y_true, n_bins=n_bins)
        metrics = calibration_metrics(probs, y_true, n_bins=n_bins)
        centres = (bins["edges"][:-1] + bins["edges"][1:]) / 2
        populated = bins["counts"] > 0

        ax = axes[0, col]
        ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.2,
                label="perfect calibration")
        ax.bar(centres[populated], bins["accuracy"][populated],
               width=1.0 / n_bins * 0.92, color=colour, alpha=0.75,
               edgecolor="white", linewidth=0.6, label="observed accuracy")
        ax.plot(bins["mean_confidence"][populated], bins["accuracy"][populated],
                "o-", color="#222222", markersize=4, linewidth=1.2,
                label="bin centroid")

        ax.set_title(f"{label}\nECE {metrics['ece']:.4f}   MCE {metrics['mce']:.4f}   "
                     f"mean conf {metrics['mean_confidence']:.3f} vs "
                     f"acc {metrics['accuracy']:.3f}", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("accuracy" if col == 0 else "")
        ax.grid(alpha=0.22, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax.legend(fontsize=8, loc="upper left")

        hist = axes[1, col]
        hist.bar(centres, bins["counts"], width=1.0 / n_bins * 0.92,
                 color=colour, alpha=0.55, edgecolor="white", linewidth=0.6)
        hist.set_xlabel("predicted confidence")
        hist.set_ylabel("images" if col == 0 else "")
        hist.set_xlim(0, 1)
        hist.grid(alpha=0.22, linestyle=":")
        hist.spines[["top", "right"]].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=11, y=0.985)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def interpret(report: Dict[str, Any], split: str = "test") -> str:
    """A paragraph written from the numbers, ready to paste into the report.

    Deliberately direction-aware: it says "reduced" only when the number went
    down. A summary generator that hard-codes the outcome it hopes for will
    eventually print "reduced ECE to 0.0536" about a rise from 0.0461, and that
    sentence goes into a report nobody re-checks.
    """
    before = report.get(f"{split}_before")
    after = report.get(f"{split}_after")
    if before is None or after is None:
        before, after, split = report["val_before"], report["val_after"], "validation"

    direction = "overconfident" if before["overconfidence"] > 0 else "underconfident"
    moved = "reduced" if after["ece"] < before["ece"] else "raised"

    base = (
        f"Before calibration the model was {direction} on the {split} split: mean "
        f"confidence {before['mean_confidence']:.3f} against an accuracy of "
        f"{before['accuracy']:.3f}, giving an expected calibration error of "
        f"{before['ece']:.4f}. Temperature scaling with T = "
        f"{report['temperature']:.3f}, fitted on the validation split only, {moved} "
        f"ECE to {after['ece']:.4f} and moved mean confidence to "
        f"{after['mean_confidence']:.3f}. Because temperature scaling divides the "
        f"logits by a positive constant it cannot reorder them, so accuracy and QWK "
        f"are unchanged at {after['accuracy']:.4f}; only the reported confidence "
        f"changes."
    )

    change = report.get(f"{split}_change")
    if not change:
        return base

    interval = (f"{change['ece_delta']:+.4f} (95% CI "
                f"[{change['ece_delta_ci_low']:+.4f}, {change['ece_delta_ci_high']:+.4f}])")

    # NLL and Brier are proper scoring rules: lower is better for both, and
    # neither depends on a bin count, so they resolve changes ECE cannot.
    scoring = []
    for label, delta in (("NLL", change["nll_delta"]), ("Brier", change["brier_delta"])):
        scoring.append(f"{label} {delta:+.4f} ({'better' if delta < 0 else 'worse'})")
    scoring_text = " and ".join(scoring)

    if change["ece_change_real"]:
        return base + (
            f" A paired bootstrap of the ECE change gives {interval}, which excludes "
            f"zero, so the change is larger than the sampling noise of this split. "
            f"The binning-free scoring rules agree in direction: {scoring_text}."
        )

    return base + (
        f" A paired bootstrap of the ECE change gives {interval}, which includes zero: "
        f"on a split of this size ECE cannot resolve a change this small, so it is "
        f"reported as unchanged rather than as an improvement. The binning-free "
        f"scoring rules give {scoring_text}."
    )
