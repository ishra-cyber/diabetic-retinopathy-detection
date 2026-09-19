"""
MILESTONE 6b - Ensembling and confidence calibration, from saved predictions.
File location: <project_root>/scripts/m6b_refine.py

Run
---
    python -m scripts.m6b_refine

What it does
------------
Two improvements that need **no GPU and no retraining**, because Milestone 6
already wrote every run's softmax output to ``<run>/{split}_predictions.npz``:

1. **Ensemble.** Average the three architectures' probabilities. Milestone 6's
   paired bootstrap showed they are statistically indistinguishable from one
   another, which is precisely the situation where averaging helps: equal skill,
   different mistakes.

2. **Calibration.** Fit a single temperature on the validation split and report
   expected calibration error before and after. The app currently prints
   confidences like 0.99999, which is not a defensible claim about a model with
   79.6% accuracy.

Where this sits relative to "touch the test split once"
-------------------------------------------------------
Milestone 6's rule is that model *selection* happens on validation. This script
keeps that rule:

  - Every decision - whether to ensemble, which combination rule, what
    temperature - is made on the **validation** split.
  - The test numbers are then computed **once**, from predictions that already
    existed, by applying those validation-chosen settings.

That is a pre-specified secondary analysis, not iterative tuning on test. The
distinction is worth one sentence in the report: the ensemble and the calibrated
confidences are reported as additional rows alongside the original single-model
results, never as replacements for them.

Produces
--------
    reports/m6b_ensemble_comparison.csv          single models vs ensembles, both splits
    reports/m6b_ensemble_confidence_intervals.csv  bootstrap CIs on test
    reports/m6b_ensemble_pairwise.csv            ensemble vs each member, paired
    reports/m6b_ensemble_agreement.csv           how often members agree
    reports/m6b_per_class_ensemble.csv           per-class precision/recall/F1
    reports/m6b_calibration.csv                  ECE/MCE/NLL before and after, per model
    reports/m6b_summary.txt                      paragraphs ready for the report
    reports/figures/m6b_*.png                    reliability diagrams + comparison
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.models.bootstrap import bootstrap_all_runs, pairwise_comparison, plot_forest
from src.models.calibrate import (
    apply_temperature,
    bootstrap_calibration_change,
    calibrate_and_report,
    interpret as interpret_calibration,
    plot_reliability,
)
from src.models.ensemble import (
    disagreement_summary,
    ensemble_from_members,
    load_members,
    member_agreement_matrix,
)
from src.models.metrics import compute_all_metrics, per_class_metrics
from src.utils.config import class_names, ensure_dirs, get_path, load_config
from src.utils.logging_utils import get_logger

LOG = get_logger("m6b")

SPLITS = ("val", "test")


def find_prediction_runs(experiments_dir: Path, split: str) -> List[Path]:
    """Run folders that have saved predictions for this split (smoke runs excluded).

    Deliberately does not import ``src.models.evaluate.find_runs``: that module
    imports torch, and nothing in this script needs it. The whole point is that
    an examiner can reproduce these numbers on a laptop with no GPU.
    """
    experiments_dir = Path(experiments_dir)
    if not experiments_dir.is_dir():
        return []
    return sorted(
        d for d in experiments_dir.iterdir()
        if d.is_dir()
        and (d / f"{split}_predictions.npz").is_file()
        and not d.name.startswith("smoke_")
    )


def _row(name: str, split: str, y_true, y_pred, probabilities, names) -> Dict[str, Any]:
    m = compute_all_metrics(y_true, y_pred, probabilities, names)
    row = {
        "model": name,
        "split": split,
        "QWK": m["qwk"],
        "accuracy": m["accuracy"],
        "balanced_acc": m["balanced_accuracy"],
        "F1_macro": m["f1_macro"],
        "AUC_macro": m.get("auc_macro"),
    }
    for pc in m["per_class"]:
        row[f"recall_{pc['label']}"] = pc["recall"]
    return row


def main() -> int:
    cfg = load_config()
    ensure_dirs(cfg, "reports_dir", "figures_dir")
    reports_dir = get_path(cfg, "reports_dir")
    figures_dir = get_path(cfg, "figures_dir")
    experiments_dir = get_path(cfg, "experiments_dir")
    names = class_names(cfg)
    seed = cfg["project"]["seed"]

    summary: List[str] = []
    rows: List[Dict[str, Any]] = []
    loaded: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    ensembles: Dict[str, Dict[str, Any]] = {}

    # -- load every member for both splits ----------------------------------
    for split in SPLITS:
        run_dirs = find_prediction_runs(experiments_dir, split)
        if len(run_dirs) < 2:
            LOG.error(
                "Need at least 2 runs with %s_predictions.npz to ensemble; found %d. "
                "Run `python -m scripts.m6_evaluate --split %s` first.",
                split, len(run_dirs), split)
            return 1
        LOG.info("%s split: %d members -> %s", split, len(run_dirs),
                 ", ".join(d.name for d in run_dirs))
        loaded[split] = load_members(run_dirs, split=split)

    if set(loaded["val"]) != set(loaded["test"]):
        LOG.error("Different runs have val and test predictions - not comparable.")
        return 1

    # -- single-model rows ---------------------------------------------------
    for split in SPLITS:
        for name, member in loaded[split].items():
            rows.append(_row(name, split, member["y_true"], member["y_pred"],
                             member["probabilities"], names))

    # -- ensembles -----------------------------------------------------------
    LOG.info("-" * 64)
    for method in ("mean", "gmean"):
        for split in SPLITS:
            ens = ensemble_from_members(loaded[split], method=method)
            ensembles[f"{method}:{split}"] = ens
            rows.append(_row(f"ensemble_{method}", split,
                             ens["y_true"], ens["y_pred"], ens["probabilities"], names))

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(["split", "QWK"], ascending=[True, False])
    comparison = comparison.reset_index(drop=True)
    comparison.to_csv(reports_dir / "m6b_ensemble_comparison.csv", index=False)
    LOG.info("Single models vs ensembles:\n%s", comparison.to_string(index=False))

    # -- which rule wins ON VALIDATION - that is the choice ------------------
    val_only = comparison[comparison["split"] == "val"]
    best_val = val_only.iloc[0]
    chosen_method = "mean"
    val_ens = val_only[val_only["model"].str.startswith("ensemble_")]
    if not val_ens.empty:
        chosen_method = val_ens.iloc[0]["model"].replace("ensemble_", "")
    LOG.info("-" * 64)
    LOG.info("Best on VALIDATION: %s (QWK %.4f). Ensemble rule chosen: %s",
             best_val["model"], best_val["QWK"], chosen_method)

    chosen_test = ensembles[f"{chosen_method}:test"]

    # -- bootstrap CIs on test, ensemble alongside its members ---------------
    LOG.info("-" * 64)
    LOG.info("Bootstrap confidence intervals on test...")
    boot_runs = {name: (m["y_true"], m["y_pred"])
                 for name, m in loaded["test"].items()}
    # align the ensemble's ordering to the members' before comparing
    ref_ids = next(iter(loaded["test"].values()))["ids"]
    order = np.argsort(np.argsort(ref_ids))          # ensemble rows are id-sorted
    boot_runs[f"ensemble_{chosen_method}"] = (
        chosen_test["y_true"][order], chosen_test["y_pred"][order])

    ci_table = bootstrap_all_runs(boot_runs, n_boot=1000, seed=seed)
    ci_table.to_csv(reports_dir / "m6b_ensemble_confidence_intervals.csv", index=False)
    LOG.info("\n%s", ci_table[["model", "qwk", "qwk_ci", "accuracy", "accuracy_ci",
                               "balanced_accuracy", "balanced_accuracy_ci"]]
             .to_string(index=False))
    plot_forest(ci_table, "qwk", figures_dir / "m6b_ensemble_confidence_intervals.png",
                title="Test QWK with 95% bootstrap CI - members and ensemble")

    pairs = pairwise_comparison(boot_runs, metric="qwk", n_boot=1000, seed=seed)
    pairs.to_csv(reports_dir / "m6b_ensemble_pairwise.csv", index=False)
    LOG.info("\nPaired comparisons:\n%s", pairs.to_string(index=False))

    # -- does the ensemble actually have anything to work with? -------------
    dis = disagreement_summary(loaded["test"])
    member_names, agreement = member_agreement_matrix(loaded["test"])
    pd.DataFrame(agreement, index=member_names, columns=member_names).to_csv(
        reports_dir / "m6b_ensemble_agreement.csv")
    LOG.info("-" * 64)
    LOG.info("Member disagreement on test: %s", json.dumps(dis, indent=2))

    # -- per-class for the chosen ensemble -----------------------------------
    pd.DataFrame(per_class_metrics(chosen_test["y_true"], chosen_test["y_pred"], names)
                 ).to_csv(reports_dir / "m6b_per_class_ensemble.csv", index=False)

    # -- calibration ---------------------------------------------------------
    LOG.info("-" * 64)
    LOG.info("Temperature scaling (fitted on validation, applied to test)...")
    calib_rows: List[Dict[str, Any]] = []
    calib_reports: Dict[str, Dict[str, Any]] = {}

    targets: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
        name: {"val": loaded["val"][name], "test": loaded["test"][name]}
        for name in loaded["val"]
    }
    targets[f"ensemble_{chosen_method}"] = {
        "val": ensembles[f"{chosen_method}:val"],
        "test": ensembles[f"{chosen_method}:test"],
    }

    for name, pair in targets.items():
        report = calibrate_and_report(
            pair["val"]["probabilities"], pair["val"]["y_true"],
            pair["test"]["probabilities"], pair["test"]["y_true"],
        )
        # Is the ECE change bigger than the noise floor of a 550-image split?
        for split in SPLITS:
            report[f"{split}_change"] = bootstrap_calibration_change(
                pair[split]["probabilities"], pair[split]["y_true"],
                report["temperature"], n_boot=2000, seed=seed,
            )

        calib_reports[name] = report
        for split in SPLITS:
            calib_rows.append({
                "model": name,
                "split": split,
                "temperature": report["temperature"],
                **{f"{k}_before": v for k, v in report[f"{split}_before"].items()},
                **{f"{k}_after": v for k, v in report[f"{split}_after"].items()},
                **report[f"{split}_change"],
            })

        plot_reliability(
            pair["test"]["probabilities"],
            apply_temperature(pair["test"]["probabilities"], report["temperature"]),
            pair["test"]["y_true"],
            figures_dir / f"m6b_reliability_{name}.png",
            title=f"Reliability - {name} (test split)",
            temperature=report["temperature"],
        )
        change = report["test_change"]
        LOG.info("  %-22s T=%.3f  ECE %.4f -> %.4f (test)  delta %+.4f "
                 "[%+.4f, %+.4f] %s  NLL %+.4f  Brier %+.4f",
                 name, report["temperature"],
                 report["test_before"]["ece"], report["test_after"]["ece"],
                 change["ece_delta"], change["ece_delta_ci_low"],
                 change["ece_delta_ci_high"],
                 "REAL " if change["ece_change_real"] else "noise",
                 change["nll_delta"], change["brier_delta"])

    calib = pd.DataFrame(calib_rows)
    calib.to_csv(reports_dir / "m6b_calibration.csv", index=False)

    # -- write the paragraphs ------------------------------------------------
    best_member = ci_table[~ci_table["model"].str.startswith("ensemble_")].iloc[0]
    ens_row = ci_table[ci_table["model"].str.startswith("ensemble_")].iloc[0]
    ens_vs_best = pairs[
        ((pairs["model_a"] == ens_row["model"]) & (pairs["model_b"] == best_member["model"]))
        | ((pairs["model_b"] == ens_row["model"]) & (pairs["model_a"] == best_member["model"]))
    ]

    summary.append(
        f"ENSEMBLE\n"
        f"The three architectures agreed unanimously on {dis['unanimous_pct']:.1f}% of test "
        f"images, leaving {dis['contested_pct']:.1f}% contested; on {dis['contested_recoverable_pct']:.1f}% "
        f"of those contested images at least one member held the correct grade, which is "
        f"the headroom an ensemble can recover. Averaging the members' softmax outputs "
        f"({chosen_method}) gave a test QWK of {ens_row['qwk']:.4f} "
        f"(95% CI {ens_row['qwk_ci']}), against {best_member['qwk']:.4f} "
        f"({best_member['qwk_ci']}) for the best single model, {best_member['model']}."
    )
    if not ens_vs_best.empty:
        r = ens_vs_best.iloc[0]
        verdict = ("excludes zero, so the ensemble is distinguishable from the best "
                   "single model on this test set"
                   if r["distinguishable"] else
                   "includes zero, so the improvement is within sampling noise and the "
                   "ensemble should be reported as no worse rather than as better")
        summary.append(
            f"A paired bootstrap of the difference gives {r['difference']:+.4f} "
            f"(95% CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]), which {verdict}. "
            f"The ensemble costs three forward passes per image instead of one, so on "
            f"this evidence it is justified by its calibration and stability rather "
            f"than by a large accuracy gain."
        )
    summary.append("CALIBRATION\n" + interpret_calibration(
        calib_reports[f"ensemble_{chosen_method}"], split="test"))

    text = "\n\n".join(summary)
    (reports_dir / "m6b_summary.txt").write_text(text, encoding="utf-8")
    LOG.info("-" * 64)
    LOG.info("FOR YOUR REPORT:\n\n%s\n", text)
    LOG.info("-" * 64)
    LOG.info("Tables in %s, figures in %s", reports_dir, figures_dir)
    LOG.info("Next, on a machine with the GPU:")
    LOG.info("  python -m scripts.m6_evaluate --split val --tta     # measure TTA on val")
    LOG.info("  python -m scripts.m6_evaluate --split test --tta    # only if val improved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
