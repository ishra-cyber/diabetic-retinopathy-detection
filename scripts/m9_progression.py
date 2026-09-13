"""
MILESTONE 9 - Progression analysis and charts.
File location: <project_root>/scripts/m9_progression.py

Run
---
    python -m scripts.m9_progression

Produces a progression chart per patient plus a summary table, from whatever is
in the database. Simulated histories are labelled as such on every chart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from src.analysis.progression import (
    analyse_progression,
    next_review_suggestion,
    progression_figure,
    progression_summary_text,
)
from src.db import dao
from src.utils.config import class_names, ensure_dirs, get_path, load_config
from src.utils.logging_utils import get_logger

LOG = get_logger("m9")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate progression charts.")
    parser.add_argument("--patient", default=None, help="only this patient")
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(cfg, "reports_dir", "figures_dir")
    figures_dir = get_path(cfg, "figures_dir")
    reports_dir = get_path(cfg, "reports_dir")
    names = class_names(cfg)

    db_path = get_path(cfg, "db_path")
    if not db_path.is_file():
        LOG.error("No database at %s. Run: python -m scripts.m8_init_db --demo", db_path)
        return 1

    conn = dao.connect(db_path)
    patients = dao.list_patients(conn)
    if patients.empty:
        LOG.error("No patients in the database. Run: python -m scripts.m8_init_db --demo")
        return 1

    if args.patient:
        patients = patients[patients["patient_id"] == args.patient]
        if patients.empty:
            LOG.error("Patient '%s' not found.", args.patient)
            return 1

    rows = []
    for patient_id in patients["patient_id"]:
        visits = dao.get_visits(conn, patient_id)
        if visits.empty:
            continue

        analysis = analyse_progression(visits)
        suggestion = (next_review_suggestion(analysis["latest_stage"], analysis["trend"])
                      if analysis["latest_stage"] is not None else {})

        LOG.info("-" * 64)
        LOG.info("Patient %s", patient_id)
        LOG.info("  %s", progression_summary_text(analysis, names))
        if suggestion:
            LOG.info("  Illustrative review interval: %s months (NOT clinical advice)",
                     suggestion["months"])

        fig = progression_figure(visits, names, patient_id)
        out = figures_dir / f"m9_progression_{patient_id}.png"
        fig.savefig(out, dpi=150)
        LOG.info("  Saved figure: %s", out.name)

        rows.append({
            "patient_id": patient_id,
            "n_visits": analysis["n_visits"],
            "first_date": analysis["first_date"],
            "latest_date": analysis["latest_date"],
            "first_stage": analysis["first_stage"],
            "latest_stage": analysis["latest_stage"],
            "stage_change": analysis["stage_change"],
            "trend": analysis["trend"],
            "slope_per_year": analysis["slope_per_year"],
            "days_observed": analysis["days_observed"],
            "mean_confidence": analysis["mean_confidence"],
            "is_simulated": analysis["any_simulated"],
        })

    if not rows:
        LOG.error("No patient had any visits.")
        return 1

    summary = pd.DataFrame(rows)
    path = reports_dir / "m9_progression_summary.csv"
    summary.to_csv(path, index=False)

    LOG.info("=" * 64)
    LOG.info("Progression summary:\n%s", summary.to_string(index=False))
    LOG.info("Saved table: %s", path)

    n_sim = int(summary["is_simulated"].sum())
    if n_sim:
        LOG.warning("%d of %d patient histories are SIMULATED. APTOS 2019 is "
                    "cross-sectional; state this clearly in your report.",
                    n_sim, len(summary))
    LOG.info("=" * 64)
    LOG.info("Next: streamlit run app/app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
