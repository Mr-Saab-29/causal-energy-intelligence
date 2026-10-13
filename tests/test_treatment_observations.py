from __future__ import annotations

import pandas as pd

from src.causal.treatment_observations import build_treatment_readiness


def test_empty_collection_is_explicitly_not_ready() -> None:
    report = build_treatment_readiness(pd.DataFrame())

    assert report["status"] == "collecting"
    assert report["observed_treatment_available"] is False
    assert report["counts"]["completed_executions"] == 0


def test_completed_decisions_are_counted_without_claiming_identification() -> None:
    frame = pd.DataFrame(
        {
            "dashboard_accessed_at_utc": [
                "2026-10-01T08:00:00Z",
                "2026-10-02T08:00:00Z",
            ],
            "baseline_source": ["dashboard_access_time", "user_planned_start_time"],
            "workload_type": ["data_center_batch", "ev_charging"],
            "selection_source": ["recommended", None],
            "user_selected_start_utc": ["2026-10-01T10:00:00Z", None],
            "actual_start_utc": ["2026-10-01T10:05:00Z", None],
            "actual_completion_utc": ["2026-10-01T11:00:00Z", None],
            "actual_energy_kwh": [900.0, None],
        }
    )

    report = build_treatment_readiness(frame)

    assert report["observed_treatment_available"] is True
    assert report["identified_estimator_ready"] is False
    assert report["counts"] == {
        "decisions": 2,
        "selected_starts": 1,
        "completed_executions": 1,
        "distinct_decision_days": 2,
    }
    assert report["quality"]["completed_energy_kwh"] == 900.0
    assert report["workload_type_counts"] == {
        "data_center_batch": 1,
        "ev_charging": 1,
    }
