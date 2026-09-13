"""
MILESTONE 8 - Create the patient-visit database.
File location: <project_root>/scripts/m8_init_db.py

Run
---
    python -m scripts.m8_init_db
    python -m scripts.m8_init_db --demo        # also add simulated patients
    python -m scripts.m8_init_db --reset       # delete and recreate (destructive)

Creates app/assets/dr_monitor.sqlite3 with the patients, visits and meta tables.
Safe to re-run: tables are created only if absent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.progression import simulate_visit_history
from src.db import dao
from src.utils.config import get_path, load_config
from src.utils.logging_utils import get_logger

LOG = get_logger("m8")

DEMO_PATIENTS = [
    ("SIM001", 5, 0, "worsening",    "Simulated: steady progression from No DR"),
    ("SIM002", 4, 3, "improving",    "Simulated: improvement after treatment"),
    ("SIM003", 6, 2, "stable",       "Simulated: stable moderate DR"),
    ("SIM004", 5, 1, "fluctuating",  "Simulated: fluctuating grades"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialise the visit database.")
    parser.add_argument("--demo", action="store_true",
                        help="add clearly-flagged simulated patients")
    parser.add_argument("--reset", action="store_true",
                        help="DELETE the existing database first")
    args = parser.parse_args()

    cfg = load_config()
    db_path = get_path(cfg, "db_path")

    if args.reset and db_path.is_file():
        LOG.warning("Deleting existing database: %s", db_path)
        db_path.unlink()

    conn = dao.connect(db_path)
    dao.init_db(conn)
    LOG.info("Database ready: %s", db_path)

    if args.demo:
        LOG.info("-" * 60)
        LOG.info("Adding SIMULATED demonstration patients.")
        LOG.info("APTOS 2019 has one image per patient and no follow-up, so multi-visit")
        LOG.info("histories cannot be real. Every row below is flagged is_simulated=1,")
        LOG.info("shown as SIMULATED in the app, and must be described that way in the report.")
        LOG.info("-" * 60)

        for patient_id, n_visits, start_stage, pattern, note in DEMO_PATIENTS:
            existing = dao.get_visits(conn, patient_id)
            if not existing.empty:
                LOG.info("  %s already has %d visit(s) - skipping.", patient_id, len(existing))
                continue
            dao.upsert_patient(conn, patient_id, display_name=f"Simulated patient {patient_id}",
                               notes=note)
            records = simulate_visit_history(
                patient_id, n_visits=n_visits, start_stage=start_stage,
                pattern=pattern, seed=cfg["project"]["seed"] + hash(patient_id) % 1000,
            )
            for record in records:
                dao.add_visit(conn, model_version="SIMULATED", **record)
            stages = [r["predicted_stage"] for r in records]
            LOG.info("  %s: %d visits, pattern %-12s stages %s",
                     patient_id, len(records), pattern, stages)

    stats = dao.database_stats(conn)
    LOG.info("-" * 60)
    for key in ("n_patients", "n_visits", "n_real_visits", "n_simulated_visits"):
        LOG.info("  %-20s %s", key, stats[key])
    LOG.info("-" * 60)
    LOG.info("Next: python -m scripts.m9_progression   (charts)")
    LOG.info("Then: streamlit run app/app.py           (the application)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
