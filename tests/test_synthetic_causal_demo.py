from __future__ import annotations

import pandas as pd

from src.causal.marginal_response import load_estimator_config
from src.causal.synthetic_demo import generate_synthetic_evidence, mark_synthetic


def test_synthetic_evidence_is_complete_point_in_time_and_reproducible() -> None:
    observations, mart, truth = generate_synthetic_evidence(
        decision_days=30, decisions_per_day=1, seed=41
    )
    repeated, repeated_mart, repeated_truth = generate_synthetic_evidence(
        decision_days=30, decisions_per_day=1, seed=41
    )

    assert len(observations) == 30
    assert observations["status"].eq("completed").all()
    assert observations["actual_energy_kwh"].gt(0).all()
    assert observations["dashboard_accessed_at_utc"].dt.date.nunique() == 30
    assert set(load_estimator_config()["adjustment_features"]).issubset(mart.columns)
    assert (
        pd.to_datetime(mart["metadata_snapshot_at_utc"], utc=True)
        < pd.to_datetime(mart["timestamp_utc"], utc=True)
    ).all()
    pd.testing.assert_frame_equal(observations, repeated)
    pd.testing.assert_series_equal(
        mart["outcome_interconnected_direct_emissions_kgco2e_h0"],
        repeated_mart["outcome_interconnected_direct_emissions_kgco2e_h0"],
    )
    assert truth == repeated_truth
    assert truth["production_eligible"] is False


def test_synthetic_report_cannot_be_mistaken_for_production_evidence() -> None:
    report = mark_synthetic(
        {
            "status": "ok",
            "identified_estimator_ready": True,
            "evidence_tier": "observational",
        }
    )

    assert report["status"] == "synthetic_demo"
    assert report["computational_status"] == "ok"
    assert report["production_eligible"] is False
    assert report["evidence_tier"] == "synthetic_pipeline_validation_only"
