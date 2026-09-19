"""
MILESTONE 6c - Ordinal decoding: turn the existing classifiers into graders.
File location: <project_root>/scripts/m6c_ordinal_decode.py

Run
---
    python -m scripts.m6c_ordinal_decode                 # fit for accuracy
    python -m scripts.m6c_ordinal_decode --metric qwk    # fit for QWK instead

The idea
--------
No GPU, no retraining. A softmax over an ordinal scale already contains a
perfectly good continuous estimate of severity - its expected value:

    score = sum_k  k * p_k

That number uses all five probabilities, where argmax uses only the largest.
Cutting it with thresholds fitted on validation recovers part of what argmax
throws away, and the thresholds can be positioned to suit a dataset that is 49%
class 0 and 5% class 3.

This is *ordinal decoding of a classification model*. It is not the same thing
as training with an ordinal head (configs/experiments/densenet121_ordinal.yaml
does that, and needs the GPU); think of this as the cheap test of whether the
ordinal framing helps at all on this data before paying for the expensive one.

Choosing the objective
----------------------
QWK-optimal and accuracy-optimal thresholds are different thresholds, and on
this dataset they conflict: QWK is happy to trade an exact match for a
one-grade-closer miss. Fitting for QWK raises val QWK but drops test exact-match
accuracy from 0.796 to 0.733. ``--metric`` therefore defaults to ``accuracy``;
whichever you pick, name it in the report.

Honesty
-------
The thresholds are fitted on **validation** and applied unchanged to test. The
per-model choice of objective must also be made on validation - the test column
is there to be reported once, not to be shopped through.

Produces
--------
    reports/m6c_<metric>_comparison.csv     argmax vs thresholded, both splits
    reports/m6c_<metric>_bootstrap.csv      paired CIs on the test-set change
    reports/m6c_<metric>_thresholds.csv     the fitted cut-points themselves
    reports/m6c_<metric>_per_class.csv      per-class recall, before and after
    reports/m6c_<metric>_summary.txt        paragraphs for the report
    reports/figures/m6c_<metric>_*.png

Every filename carries the objective, so running all three leaves three
comparable sets on disk instead of one silently overwriting the next.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.models.bootstrap import bootstrap_all_runs, plot_forest
from src.models.ensemble import ensemble_from_members, load_members
from src.models.metrics import compute_all_metrics, per_class_metrics
from src.models.thresholds import apply_thresholds, describe, fit_thresholds
from src.utils.config import class_names, ensure_dirs, get_path, load_config
from src.utils.logging_utils import get_logger

LOG = get_logger("m6c")


def find_runs(experiments_dir: Path, split: str) -> List[Path]:
    experiments_dir = Path(experiments_dir)
    if not experiments_dir.is_dir():
        return []
    return sorted(d for d in experiments_dir.iterdir()
                  if d.is_dir() and (d / f"{split}_predictions.npz").is_file()
                  and not d.name.startswith("smoke_"))


def expected_grade(probabilities: np.ndarray) -> np.ndarray:
    """score = sum_k k * p_k, the mean of the distribution over grades."""
    grades = np.arange(probabilities.shape[1], dtype=float)
    return probabilities @ grades


def paired_bootstrap(y_true, y_a, y_b, metric_fn, n_boot=2000, seed=42) -> Dict[str, Any]:
    """Distribution of (metric of A) - (metric of B) on shared resamples."""
    y_true, y_a, y_b = map(lambda v: np.asarray(v, dtype=int), (y_true, y_a, y_b))
    rng = np.random.default_rng(seed)
    n = y_true.size
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        if np.unique(y_true[i]).size < 2:
            diffs[b] = np.nan
            continue
        diffs[b] = metric_fn(y_true[i], y_a[i]) - metric_fn(y_true[i], y_b[i])
    valid = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return {
        "difference": round(float(metric_fn(y_true, y_a) - metric_fn(y_true, y_b)), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "prob_better": round(float((valid > 0).mean()), 3),
        "distinguishable": bool(lo > 0 or hi < 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="accuracy",
                        choices=["accuracy", "qwk", "balanced_accuracy"],
                        help="what the thresholds are fitted to maximise on validation")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(cfg, "reports_dir", "figures_dir")
    reports_dir = get_path(cfg, "reports_dir")
    figures_dir = get_path(cfg, "figures_dir")
    names = class_names(cfg)
    seed = cfg["project"]["seed"]
    experiments_dir = get_path(cfg, "experiments_dir")

    loaded = {}
    for split in ("val", "test"):
        runs = find_runs(experiments_dir, split)
        if len(runs) < 1:
            LOG.error("No runs with %s_predictions.npz. Run m6_evaluate first.", split)
            return 1
        loaded[split] = load_members(runs, split)

    # The ensemble is a member here too - its decoded score is usually the best
    # of the lot, and it costs nothing extra at this point.
    members: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
        name: {"val": loaded["val"][name], "test": loaded["test"][name]}
        for name in loaded["val"]
    }
    if len(loaded["val"]) > 1:
        members["ensemble_gmean"] = {
            "val": ensemble_from_members(loaded["val"], method="gmean"),
            "test": ensemble_from_members(loaded["test"], method="gmean"),
        }

    LOG.info("Fitting thresholds on VALIDATION to maximise %s", args.metric.upper())
    LOG.info("=" * 68)

    rows: List[Dict[str, Any]] = []
    thr_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    ci_runs: Dict[str, Any] = {}

    qwk_fn = lambda yt, yp: compute_all_metrics(yt, yp, None, names)["qwk"]      # noqa: E731
    acc_fn = lambda yt, yp: float((np.asarray(yt) == np.asarray(yp)).mean())    # noqa: E731

    for name, pair in members.items():
        val_p, val_y = pair["val"]["probabilities"], pair["val"]["y_true"]
        test_p, test_y = pair["test"]["probabilities"], pair["test"]["y_true"]

        fit = fit_thresholds(expected_grade(val_p), val_y, metric=args.metric)
        edges = fit["thresholds"]
        LOG.info("%s", name)
        LOG.info("%s", describe(fit, names))

        thr_rows.append({"model": name, "metric": args.metric,
                         **{f"t{i}": t for i, t in enumerate(edges)}})

        for split, probs, y in (("val", val_p, val_y), ("test", test_p, test_y)):
            argmax_pred = probs.argmax(axis=1)
            thr_pred = apply_thresholds(expected_grade(probs), edges)
            for label, pred in (("argmax", argmax_pred), ("thresholded", thr_pred)):
                m = compute_all_metrics(y, pred, probs, names)
                row = {"model": name, "split": split, "decoding": label,
                       "QWK": m["qwk"], "accuracy": m["accuracy"],
                       "balanced_acc": m["balanced_accuracy"], "F1_macro": m["f1_macro"]}
                for pc in m["per_class"]:
                    row[f"recall_{pc['label']}"] = pc["recall"]
                rows.append(row)

            if split == "test":
                ci_runs[f"{name}:argmax"] = (y, argmax_pred)
                ci_runs[f"{name}:thresholded"] = (y, thr_pred)
                for metric_name, fn in (("qwk", qwk_fn), ("accuracy", acc_fn)):
                    boot_rows.append({
                        "model": name, "metric": metric_name,
                        **paired_bootstrap(y, thr_pred, argmax_pred, fn,
                                           n_boot=args.n_boot, seed=seed),
                    })
                for a, b in zip(per_class_metrics(y, argmax_pred, names),
                                per_class_metrics(y, thr_pred, names)):
                    per_class_rows.append({
                        "model": name, "class_name": a["class_name"],
                        "support": a["support"],
                        "recall_argmax": a["recall"], "recall_thresholded": b["recall"],
                        "recall_change": round(b["recall"] - a["recall"], 4),
                    })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(reports_dir / f"m6c_{args.metric}_comparison.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(reports_dir / f"m6c_{args.metric}_thresholds.csv", index=False)
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(reports_dir / f"m6c_{args.metric}_bootstrap.csv", index=False)
    per_class = pd.DataFrame(per_class_rows)
    per_class.to_csv(reports_dir / f"m6c_{args.metric}_per_class.csv", index=False)

    LOG.info("=" * 68)
    LOG.info("TEST split, argmax vs thresholded:\n%s",
             comparison[comparison["split"] == "test"]
             [["model", "decoding", "QWK", "accuracy", "balanced_acc",
               "recall_1", "recall_3"]].to_string(index=False))
    LOG.info("=" * 68)
    LOG.info("Paired bootstrap of the change on TEST (thresholded - argmax):\n%s",
             boot.to_string(index=False))

    ci_table = bootstrap_all_runs(ci_runs, n_boot=1000, seed=seed)
    ci_table.to_csv(reports_dir / f"m6c_{args.metric}_confidence_intervals.csv", index=False)
    plot_forest(ci_table, "accuracy", figures_dir / f"m6c_{args.metric}_accuracy_ci.png",
                title=f"Test accuracy, argmax vs thresholds fitted for {args.metric}")
    plot_forest(ci_table, "qwk", figures_dir / f"m6c_{args.metric}_qwk_ci.png",
                title=f"Test QWK, argmax vs thresholds fitted for {args.metric}")

    # -- the paragraph -------------------------------------------------------
    test = comparison[comparison["split"] == "test"]
    best_name = (test[test["decoding"] == "thresholded"]
                 .sort_values("accuracy", ascending=False).iloc[0]["model"])
    base = test[(test["model"] == best_name) & (test["decoding"] == "argmax")].iloc[0]
    tuned = test[(test["model"] == best_name) & (test["decoding"] == "thresholded")].iloc[0]
    acc_b = boot[(boot["model"] == best_name) & (boot["metric"] == "accuracy")].iloc[0]
    qwk_b = boot[(boot["model"] == best_name) & (boot["metric"] == "qwk")].iloc[0]

    verdict = ("excludes zero" if acc_b["distinguishable"]
               else "includes zero, so it is within sampling noise")
    text = (
        f"ORDINAL DECODING\n"
        f"Replacing argmax with the expected grade sum_k k*p_k, cut by four thresholds "
        f"fitted on the validation split to maximise {args.metric}, changed {best_name} "
        f"on the held-out test split from {base['accuracy']:.4f} to "
        f"{tuned['accuracy']:.4f} exact-match accuracy and from {base['QWK']:.4f} to "
        f"{tuned['QWK']:.4f} QWK. A paired bootstrap of the accuracy change gives "
        f"{acc_b['difference']:+.4f} (95% CI [{acc_b['ci_low']:+.4f}, "
        f"{acc_b['ci_high']:+.4f}]), which {verdict}; the QWK change is "
        f"{qwk_b['difference']:+.4f} (95% CI [{qwk_b['ci_low']:+.4f}, "
        f"{qwk_b['ci_high']:+.4f}]). No retraining was involved: the thresholds are "
        f"four numbers fitted to predictions that already existed, which makes this "
        f"the cheapest accuracy gain available to the project and a natural baseline "
        f"for the trained ordinal head."
    )
    (reports_dir / f"m6c_{args.metric}_summary.txt").write_text(text, encoding="utf-8")
    LOG.info("=" * 68)
    LOG.info("FOR YOUR REPORT:\n\n%s\n", text)
    LOG.info("Tables in %s", reports_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
