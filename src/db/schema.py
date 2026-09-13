"""
SQLite schema for patient visit records.
File location: <project_root>/src/db/schema.py

Why SQLite
----------
Zero configuration, a single file on disk, ships with Python, and trivially
demonstrable to an examiner ("here is the database, here are the rows"). For a
single-user screening prototype that is the right choice; a multi-user clinical
system would need something else, and saying so in your report shows you know
the difference.

Design notes
------------
* ``is_simulated`` is NOT optional bookkeeping. APTOS has no longitudinal data,
  so multi-visit histories are synthetic. Every synthetic row is flagged here,
  surfaced in the UI, and disclosed in the report. Presenting simulated
  progression as real patient data would be straightforwardly dishonest.
* ``probabilities`` holds the full 5-way softmax as JSON, not just the winning
  class. You cannot recover it later, and it is what lets the app show a
  confidence breakdown rather than a single number.
* ``model_version`` records which checkpoint produced the prediction, so a row
  predicted by an older model is identifiable after you retrain.
* CHECK constraints keep impossible rows out at the database level rather than
  relying on every caller to validate.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

CREATE_PATIENTS = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id   TEXT PRIMARY KEY,
    display_name TEXT,
    notes        TEXT,
    created_at   TEXT NOT NULL
);
"""

CREATE_VISITS = """
CREATE TABLE IF NOT EXISTS visits (
    visit_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      TEXT NOT NULL,
    visit_date      TEXT NOT NULL,                       -- ISO-8601 'YYYY-MM-DD'
    predicted_stage INTEGER NOT NULL CHECK (predicted_stage BETWEEN 0 AND 4),
    confidence      REAL    NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    probabilities   TEXT,                                -- JSON array of 5 floats
    image_path      TEXT,
    gradcam_path    TEXT,
    model_version   TEXT,
    is_simulated    INTEGER NOT NULL DEFAULT 0 CHECK (is_simulated IN (0, 1)),
    notes           TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients (patient_id) ON DELETE CASCADE
);
"""

CREATE_META = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# Visit history is always queried "for this patient, in date order", so this
# composite index covers the only access pattern that matters.
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_visits_patient_date ON visits (patient_id, visit_date);",
    "CREATE INDEX IF NOT EXISTS idx_visits_date ON visits (visit_date);",
    "CREATE INDEX IF NOT EXISTS idx_visits_simulated ON visits (is_simulated);",
]

ALL_STATEMENTS = [CREATE_PATIENTS, CREATE_VISITS, CREATE_META, *CREATE_INDEXES]
