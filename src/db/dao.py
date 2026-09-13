"""
Data-access layer for the patient-visit database.
File location: <project_root>/src/db/dao.py

Every SQL statement in the project lives here. Nothing else - not the Streamlit
app, not the analysis code - writes SQL directly. That keeps the app readable
and makes this the single file to unit-test.

All functions take an open ``sqlite3.Connection`` so a test can pass an
in-memory database and the app can hold one connection for its session.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from src.db.schema import ALL_STATEMENTS, SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
def connect(db_path: str | Path, create_dirs: bool = True) -> sqlite3.Connection:
    """Open (and if needed create) the database.

    ``check_same_thread=False`` because Streamlit runs callbacks on different
    threads; ``row_factory`` so rows behave like dicts; foreign keys are OFF by
    default in SQLite and must be enabled per connection.
    """
    db_path = Path(db_path)
    if create_dirs:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if they do not exist. Safe to call repeatedly."""
    with conn:
        for statement in ALL_STATEMENTS:
            conn.execute(statement)
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?);",
            ("schema_version", str(SCHEMA_VERSION)),
        )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _validate_date(value: str | date) -> str:
    """Accept a date or an ISO string; return 'YYYY-MM-DD'. Raise on anything else."""
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"visit_date must be YYYY-MM-DD (or a date object); got {value!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------
def upsert_patient(
    conn: sqlite3.Connection,
    patient_id: str,
    display_name: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """Create a patient, or update their details if they already exist."""
    patient_id = str(patient_id).strip()
    if not patient_id:
        raise ValueError("patient_id cannot be empty.")

    with conn:
        existing = conn.execute(
            "SELECT patient_id FROM patients WHERE patient_id = ?;", (patient_id,)
        ).fetchone()

        if existing is None:
            conn.execute(
                "INSERT INTO patients (patient_id, display_name, notes, created_at) "
                "VALUES (?, ?, ?, ?);",
                (patient_id, display_name, notes, _now()),
            )
        else:
            # COALESCE keeps the stored value when the caller passes None, so a
            # later visit does not blank out a name entered earlier.
            conn.execute(
                "UPDATE patients SET display_name = COALESCE(?, display_name), "
                "notes = COALESCE(?, notes) WHERE patient_id = ?;",
                (display_name, notes, patient_id),
            )
    return patient_id


def get_patient(conn: sqlite3.Connection, patient_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM patients WHERE patient_id = ?;", (str(patient_id),)
    ).fetchone()
    return dict(row) if row else None


def list_patients(conn: sqlite3.Connection) -> pd.DataFrame:
    """All patients with a visit summary, most recently seen first."""
    query = """
        SELECT p.patient_id,
               p.display_name,
               p.created_at,
               COUNT(v.visit_id)                AS n_visits,
               MIN(v.visit_date)                AS first_visit,
               MAX(v.visit_date)                AS last_visit,
               MAX(v.predicted_stage)           AS max_stage,
               SUM(v.is_simulated)              AS n_simulated
        FROM patients p
        LEFT JOIN visits v ON v.patient_id = p.patient_id
        GROUP BY p.patient_id
        ORDER BY last_visit DESC NULLS LAST, p.patient_id;
    """
    try:
        return pd.read_sql_query(query, conn)
    except Exception:
        # NULLS LAST needs SQLite >= 3.30; fall back for older builds.
        return pd.read_sql_query(query.replace(" NULLS LAST", ""), conn)


def delete_patient(conn: sqlite3.Connection, patient_id: str) -> int:
    """Delete a patient and (via ON DELETE CASCADE) all of their visits."""
    with conn:
        cur = conn.execute("DELETE FROM patients WHERE patient_id = ?;", (str(patient_id),))
    return cur.rowcount


# ---------------------------------------------------------------------------
# Visits
# ---------------------------------------------------------------------------
def add_visit(
    conn: sqlite3.Connection,
    patient_id: str,
    visit_date: str | date,
    predicted_stage: int,
    confidence: float,
    probabilities: Optional[Sequence[float]] = None,
    image_path: Optional[str] = None,
    gradcam_path: Optional[str] = None,
    model_version: Optional[str] = None,
    is_simulated: bool = False,
    notes: Optional[str] = None,
) -> int:
    """Record one screening visit. Returns the new visit_id.

    The patient row is created automatically if it does not exist, so the app
    never has to do it in two steps.
    """
    predicted_stage = int(predicted_stage)
    confidence = float(confidence)

    if not 0 <= predicted_stage <= 4:
        raise ValueError(f"predicted_stage must be 0-4, got {predicted_stage}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be 0-1, got {confidence}")

    visit_date = _validate_date(visit_date)

    probabilities_json = None
    if probabilities is not None:
        probabilities = [float(p) for p in probabilities]
        if len(probabilities) != 5:
            raise ValueError(f"probabilities must have 5 entries, got {len(probabilities)}")
        probabilities_json = json.dumps([round(p, 6) for p in probabilities])

    # Every argument is validated BEFORE the patient row is created, so a
    # rejected visit cannot leave an orphan patient behind.
    upsert_patient(conn, patient_id)

    with conn:
        cur = conn.execute(
            """INSERT INTO visits
               (patient_id, visit_date, predicted_stage, confidence, probabilities,
                image_path, gradcam_path, model_version, is_simulated, notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (str(patient_id), visit_date, predicted_stage, confidence, probabilities_json,
             image_path, gradcam_path, model_version, int(bool(is_simulated)), notes, _now()),
        )
    return int(cur.lastrowid)


def get_visits(
    conn: sqlite3.Connection,
    patient_id: str,
    include_simulated: bool = True,
) -> pd.DataFrame:
    """All visits for one patient, oldest first (the order progression needs)."""
    query = "SELECT * FROM visits WHERE patient_id = ?"
    params: List[Any] = [str(patient_id)]
    if not include_simulated:
        query += " AND is_simulated = 0"
    query += " ORDER BY visit_date ASC, visit_id ASC;"

    df = pd.read_sql_query(query, conn, params=params)
    if not df.empty and "probabilities" in df:
        df["probabilities"] = df["probabilities"].apply(
            lambda s: json.loads(s) if isinstance(s, str) else None
        )
    return df


def get_visit(conn: sqlite3.Connection, visit_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM visits WHERE visit_id = ?;", (int(visit_id),)).fetchone()
    if row is None:
        return None
    record = dict(row)
    if isinstance(record.get("probabilities"), str):
        record["probabilities"] = json.loads(record["probabilities"])
    return record


def delete_visit(conn: sqlite3.Connection, visit_id: int) -> int:
    with conn:
        cur = conn.execute("DELETE FROM visits WHERE visit_id = ?;", (int(visit_id),))
    return cur.rowcount


def recent_visits(conn: sqlite3.Connection, limit: int = 20) -> pd.DataFrame:
    """Most recent visits across all patients - the app's home-page feed."""
    return pd.read_sql_query(
        "SELECT visit_id, patient_id, visit_date, predicted_stage, confidence, "
        "is_simulated, model_version, created_at "
        "FROM visits ORDER BY created_at DESC LIMIT ?;",
        conn, params=[int(limit)],
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def database_stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Counts for the app's status panel and for your report's screenshots."""
    def scalar(sql: str, default: Any = 0) -> Any:
        row = conn.execute(sql).fetchone()
        return (row[0] if row and row[0] is not None else default)

    by_stage = pd.read_sql_query(
        "SELECT predicted_stage, COUNT(*) AS n FROM visits "
        "GROUP BY predicted_stage ORDER BY predicted_stage;", conn
    )

    return {
        "n_patients": int(scalar("SELECT COUNT(*) FROM patients;")),
        "n_visits": int(scalar("SELECT COUNT(*) FROM visits;")),
        "n_real_visits": int(scalar("SELECT COUNT(*) FROM visits WHERE is_simulated = 0;")),
        "n_simulated_visits": int(scalar("SELECT COUNT(*) FROM visits WHERE is_simulated = 1;")),
        "earliest_visit": scalar("SELECT MIN(visit_date) FROM visits;", None),
        "latest_visit": scalar("SELECT MAX(visit_date) FROM visits;", None),
        "visits_by_stage": dict(zip(by_stage["predicted_stage"].tolist(),
                                    by_stage["n"].tolist())) if not by_stage.empty else {},
        "schema_version": scalar("SELECT value FROM meta WHERE key = 'schema_version';", "?"),
    }


def export_all(conn: sqlite3.Connection) -> pd.DataFrame:
    """Every visit joined to its patient - for CSV export and for the report."""
    return pd.read_sql_query(
        """SELECT v.visit_id, v.patient_id, p.display_name, v.visit_date,
                  v.predicted_stage, v.confidence, v.model_version,
                  v.is_simulated, v.image_path, v.gradcam_path, v.created_at
           FROM visits v LEFT JOIN patients p ON p.patient_id = v.patient_id
           ORDER BY v.patient_id, v.visit_date;""",
        conn,
    )
