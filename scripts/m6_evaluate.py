"""
MILESTONE 6 - Evaluate on the held-out test split and build the report figures.
File location: <project_root>/scripts/m6_evaluate.py

THE RULE
--------
This runs ONCE, after all model selection is finished. Every earlier decision
was made on validation. If you evaluate on test, tune something, and evaluate
again, your reported numbers are optimistically biased and you would have to
disclose it.

Run
---
    python -m scripts.m6_evaluate
    python -m scripts.m6_evaluate --split val    # sanity check without burning test
    python -m scripts.m6_evaluate --split val --tta   # measure the TTA gain on val FIRST

Produces
--------
    experiments/<run>/metrics_test.json      per run
    experiments/<run>/test_predictions.npz   per run
    reports/m6_test_comparison.csv           the results table for your report
    reports/m6_per_class_<best>.csv
    reports/m6_error_analysis.json
    reports/figures/m6_*.png                 five figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.models.evaluate import (
    comparison_table,
    error_analysis,
    evaluate_run,
    find_runs,
    per_class_table,
)
from src.models.bootstrap import (
    bootstrap_all_runs,
    interpret,
    pairwise_comparison,
    plot_forest,
)
from src.models.figures import (
    plot_confusion_matrices,
    plot_error_distance,
    plot_model_comparison,
    plot_per_class_recall,
    plot_roc_curves,
)
from src.models.metrics import summarise_for_console
from src.utils.config import class_names, ensure_dirs, get_path, load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

LOG = get_logger("m6")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained models on the test split.")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--runs", nargs="*", default=None,
                        help="specific run folder names (default: all finished runs)")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="bootstrap resamples for confidence intervals (0 = skip)")
    parser.add_argument("--tta", action="store_true",
                        help="average predictions over the four flip views. Decide "
                             "whether to use it on --split val; only then re-run on test.")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    ensure_dirs(cfg, "reports_dir", "figures_dir")
    reports_dir = get_path(cfg, "reports_dir")
    suffix = "_tta" if args.tta else ""
    figures_dir = get_path(cfg, "figures_dir")
    names = class_names(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("Device: %s", device)

    runs = find_runs(get_path(cfg, "experiments_dir"))
    if args.runs:
        runs = [r for r in runs if r.name in args.runs]
    if not runs:
        LOG.error("No finished runs found (a run needs best.pt). Train first.")
        return 1

    LOG.info("Evaluating %d run(s) on the %s split: %s",
             len(runs), args.split, ", ".join(r.name for r in runs))
    if args.tta:
        LOG.info("Test-time augmentation ON: averaging over none/hflip/vflip/hvflip "
                 "(4x inference). Outputs are written under *_tta names.")
    if args.split == "test":
        LOG.warning("This is the HELD-OUT TEST SPLIT. Report these numbers as final; "
                    "do not tune anything and re-run.")

    all_metrics = []
    for run_dir in runs:
        LOG.info("-" * 64)
        LOG.info("Run: %s", run_dir.name)
        try:
            metrics = evaluate_run(run_dir, cfg, device, split=args.split, tta=args.tta)
        except Exception as exc:  # noqa: BLE001
            LOG.error("  failed: %s: %s", type(exc).__name__, exc)
            continue
        all_metrics.append(metrics)
        LOG.info("\n%s", summarise_for_console(metrics))

    if not all_metrics:
        LOG.error("Nothing evaluated successfully.")
        return 1

    # -- all runs must share one split, or the comparison is meaningless -----
    fingerprints = {m.get("split_fingerprint") for m in all_metrics}
    if len(fingerprints) > 1:
        LOG.error("Runs used DIFFERENT splits %s - they are NOT comparable.", fingerprints)
        LOG.error("Re-train every architecture on a single split before comparing.")
    else:
        LOG.info("All runs share split fingerprint %s - comparison is valid.",
                 fingerprints.pop())

    # -- table --------------------------------------------------------------
    comparison = comparison_table(all_metrics)
    LOG.info("-" * 64)
    LOG.info("%s-split comparison:\n%s", args.split.upper(), comparison.to_string(index=False))

    csv_path = reports_dir / f"m6_{args.split}{suffix}_comparison.csv"
    comparison.to_csv(csv_path, index=False)
    LOG.info("Saved table: %s", csv_path)

    # -- bootstrap confidence intervals --------------------------------------
    # Three architectures within 0.005 QWK is not a ranking, it is a tie. This
    # measures that rather than asserting it.
    if args.n_boot > 0 and len(all_metrics) >= 1:
        LOG.info("-" * 64)
        LOG.info("Bootstrap confidence intervals (%d resamples)...", args.n_boot)
        boot_runs = {
            m.get("backbone", m.get("run_name")): (m["_y_true"], m["_y_pred"])
            for m in all_metrics
        }
        try:
            ci_table = bootstrap_all_runs(boot_runs, n_boot=args.n_boot,
                                          seed=cfg["project"]["seed"])
            LOG.info("\n%s", ci_table[
                ["model", "qwk", "qwk_ci", "accuracy", "accuracy_ci",
                 "balanced_accuracy", "balanced_accuracy_ci"]
            ].to_string(index=False))
            ci_table.to_csv(reports_dir / f"m6_{args.split}{suffix}_confidence_intervals.csv",
                            index=False)

            plot_forest(ci_table, "qwk",
                        figures_dir / f"m6_qwk_confidence_intervals_{args.split}{suffix}.png",
                        title=f"{args.split.capitalize()} QWK with 95% bootstrap CI")

            if len(all_metrics) > 1:
                pairs = pairwise_comparison(boot_runs, metric="qwk", n_boot=args.n_boot,
                                            seed=cfg["project"]["seed"])
                LOG.info("\nPaired comparisons (same resample scored by both models):\n%s",
                         pairs.to_string(index=False))
                pairs.to_csv(reports_dir / f"m6_{args.split}{suffix}_pairwise.csv", index=False)

                LOG.info("-" * 64)
                LOG.info("FOR YOUR REPORT:")
                LOG.info("  %s", interpret(ci_table, pairs, "qwk"))
                LOG.info("-" * 64)
        except Exception as exc:  # noqa: BLE001 - never let stats kill the run
            LOG.warning("Bootstrap failed (%s: %s) - continuing without CIs.",
                        type(exc).__name__, exc)

    # -- figures ------------------------------------------------------------
    best = max(all_metrics, key=lambda m: m["qwk"])
    tag = best.get("backbone", "best")
    LOG.info("Best by %s QWK: %s (%.4f)", args.split, tag, best["qwk"])

    plot_confusion_matrices(
        best["confusion_matrix"], names,
        figures_dir / f"m6_confusion_matrix_{tag}{suffix}.png",
        title=f"Confusion matrix - {tag} ({args.split} split, n={best['n_samples']})",
    )
    plot_per_class_recall(all_metrics, names, figures_dir / f"m6_per_class_recall{suffix}.png")
    plot_roc_curves(best["_y_true"], best["_probabilities"], names,
                    figures_dir / f"m6_roc_curves_{tag}{suffix}.png",
                    title=f"One-vs-rest ROC - {tag} ({args.split} split)")
    plot_model_comparison(comparison, figures_dir / f"m6_model_comparison{suffix}.png")

    errors = error_analysis(best["_y_true"], best["_y_pred"], best["_ids"])
    plot_error_distance(errors, figures_dir / f"m6_error_distance_{tag}{suffix}.png",
                        title=f"Error magnitude - {tag} ({args.split} split)")

    per_class_table(best).to_csv(reports_dir / f"m6_per_class_{tag}{suffix}.csv", index=False)
    (reports_dir / f"m6_error_analysis{suffix}.json").write_text(
        json.dumps({"best_model": tag, **errors}, indent=2), encoding="utf-8")

    # -- the sentences worth putting in the report --------------------------
    LOG.info("-" * 64)
    LOG.info("ERROR ANALYSIS (%s):", tag)
    LOG.info("  exact grade match      : %.1f%%", errors["exact_match_pct"])
    LOG.info("  within one grade       : %.1f%%  <-- why QWK is high", errors["within_one_pct"])
    LOG.info("  mean absolute error    : %.3f grades", errors["mean_abs_error"])
    LOG.info("  error distance counts  : %s", errors["error_distance_counts"])
    LOG.info("-" * 64)
    LOG.info("Figures written to %s", figures_dir)
    LOG.info("Next: python -m scripts.m7_gradcam --run %s", best.get("run_name"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
