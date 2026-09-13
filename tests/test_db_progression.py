"""
Unit tests for Milestones 8 and 9.
File location: <project_root>/tests/test_db_progression.py

Run:  pytest -q
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.progression import (
    TREND_IMPROVING,
    TREND_INSUFFICIENT,
    TREND_STABLE,
    TREND_WORSENING,
    analyse_progression,
    next_review_suggestion,
    progression_summary_text,
    simulate_visit_history,
)
from src.db import dao

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


@pytest.fixture()
def conn():
    connection = dao.connect(":memory:", create_dirs=False)
    dao.init_db(connection)
    return connection


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def test_init_is_idempotent(conn):
    dao.init_db(conn)
    dao.init_db(conn)
    assert dao.database_stats(conn)["n_visits"] == 0


def test_add_visit_creates_patient_automatically(conn):
    dao.add_visit(conn, "P1", "2026-01-01", 2, 0.8)
    assert dao.get_patient(conn, "P1") is not None


def test_probabilities_round_trip(conn):
    probs = [0.05, 0.08, 0.83, 0.03, 0.01]
    vid = dao.add_visit(conn, "P1", "2026-01-01", 2, 0.83, probabilities=probs)
    stored = dao.get_visit(conn, vid)["probabilities"]
    assert isinstance(stored, list) and len(stored) == 5
    assert stored == pytest.approx(probs)


@pytest.mark.parametrize("kwargs", [
    {"predicted_stage": 5},
    {"predicted_stage": -1},
    {"confidence": 1.5},
    {"confidence": -0.1},
    {"visit_date": "01/01/2026"},
    {"probabilities": [0.5, 0.5]},
])
def test_invalid_visits_are_rejected(conn, kwargs):
    args = dict(patient_id="P1", visit_date="2026-01-01", predicted_stage=1, confidence=0.5)
    args.update(kwargs)
    with pytest.raises(ValueError):
        dao.add_visit(conn, **args)


def test_rejected_visit_leaves_no_orphan_patient(conn):
    """A failed insert must not half-create a patient row."""
    with pytest.raises(ValueError):
        dao.add_visit(conn, "GHOST", "2026-01-01", 1, 0.5, probabilities=[0.5, 0.5])
    assert dao.get_patient(conn, "GHOST") is None


def test_visits_come_back_in_date_order(conn):
    for day in ("2026-03-01", "2026-01-01", "2026-02-01"):
        dao.add_visit(conn, "P1", day, 1, 0.7)
    dates = dao.get_visits(conn, "P1")["visit_date"].tolist()
    assert dates == sorted(dates)


def test_simulated_filter(conn):
    dao.add_visit(conn, "P1", "2026-01-01", 1, 0.7, is_simulated=False)
    dao.add_visit(conn, "P1", "2026-02-01", 2, 0.7, is_simulated=True)
    assert len(dao.get_visits(conn, "P1")) == 2
    assert len(dao.get_visits(conn, "P1", include_simulated=False)) == 1


def test_delete_patient_cascades_to_visits(conn):
    dao.add_visit(conn, "P1", "2026-01-01", 1, 0.7)
    dao.add_visit(conn, "P1", "2026-02-01", 2, 0.7)
    dao.delete_patient(conn, "P1")
    assert dao.database_stats(conn)["n_visits"] == 0


def test_upsert_does_not_blank_existing_name(conn):
    dao.upsert_patient(conn, "P1", display_name="Alice")
    dao.upsert_patient(conn, "P1", notes="second contact")     # no name passed
    patient = dao.get_patient(conn, "P1")
    assert patient["display_name"] == "Alice"
    assert patient["notes"] == "second contact"


def test_stats_separate_real_and_simulated(conn):
    dao.add_visit(conn, "P1", "2026-01-01", 1, 0.7, is_simulated=False)
    for i in range(3):
        dao.add_visit(conn, "S1", f"2026-0{i+1}-01", 2, 0.7, is_simulated=True)
    stats = dao.database_stats(conn)
    assert stats["n_real_visits"] == 1
    assert stats["n_simulated_visits"] == 3
    assert stats["n_patients"] == 2


# ---------------------------------------------------------------------------
# Progression
# ---------------------------------------------------------------------------
def _history(stages, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(stages), freq="180D").strftime("%Y-%m-%d")
    return pd.DataFrame({
        "visit_date": dates,
        "predicted_stage": stages,
        "confidence": [0.8] * len(stages),
        "is_simulated": [True] * len(stages),
    })


def test_empty_history():
    assert analyse_progression(pd.DataFrame())["trend"] == TREND_INSUFFICIENT


def test_single_visit_is_not_called_stable():
    """One visit is a baseline, not a trajectory - claiming 'stable' would be wrong."""
    analysis = analyse_progression(_history([2]))
    assert analysis["trend"] == TREND_INSUFFICIENT
    assert analysis["n_visits"] == 1


@pytest.mark.parametrize("stages,expected", [
    ([0, 1, 2, 3], TREND_WORSENING),
    ([3, 2, 1, 0], TREND_IMPROVING),
    ([2, 2, 2, 2], TREND_STABLE),
    ([1, 3, 2, 1], TREND_STABLE),       # ends where it started
])
def test_trend_classification(stages, expected):
    assert analyse_progression(_history(stages))["trend"] == expected


def test_slope_sign_matches_direction():
    assert analyse_progression(_history([0, 1, 2, 3]))["slope_per_year"] > 0
    assert analyse_progression(_history([3, 2, 1, 0]))["slope_per_year"] < 0
    assert analyse_progression(_history([2, 2, 2]))["slope_per_year"] == pytest.approx(0, abs=1e-6)


def test_summary_flags_simulated_data():
    text = progression_summary_text(analyse_progression(_history([0, 2])), CLASS_NAMES)
    assert "SIMULATED" in text


def test_worsening_shortens_review_interval():
    """A progressing patient should be reviewed sooner than a stable one."""
    stable = next_review_suggestion(2, TREND_STABLE)["months"]
    worse = next_review_suggestion(2, TREND_WORSENING)["months"]
    assert worse < stable


def test_review_suggestion_carries_disclaimer():
    suggestion = next_review_suggestion(3, TREND_WORSENING)
    assert "Not clinical advice" in suggestion["disclaimer"]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def test_simulated_records_are_always_flagged():
    records = simulate_visit_history("SIM", n_visits=5, pattern="worsening")
    assert all(r["is_simulated"] for r in records)
    assert all("SIMULATED" in r["notes"] for r in records)


def test_simulation_is_reproducible():
    a = simulate_visit_history("SIM", 5, 1, "worsening", seed=42)
    b = simulate_visit_history("SIM", 5, 1, "worsening", seed=42)
    assert [r["predicted_stage"] for r in a] == [r["predicted_stage"] for r in b]


def test_simulated_stages_stay_in_range():
    for pattern in ("worsening", "improving", "stable", "fluctuating"):
        for start in range(5):
            records = simulate_visit_history("SIM", 8, start, pattern, seed=1)
            assert all(0 <= r["predicted_stage"] <= 4 for r in records)
            assert all(0.0 <= r["confidence"] <= 1.0 for r in records)


def test_simulated_probabilities_sum_to_one():
    for record in simulate_visit_history("SIM", 4, 2, "stable", seed=5):
        assert sum(record["probabilities"]) == pytest.approx(1.0, abs=0.01)


def test_unknown_pattern_rejected():
    with pytest.raises(ValueError, match="unknown pattern"):
        simulate_visit_history("SIM", 3, 1, "exploding")
