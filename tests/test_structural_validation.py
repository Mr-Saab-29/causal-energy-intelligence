from __future__ import annotations

import numpy as np
import pandas as pd

from src.causal.structural_validation import (
    build_synthetic_dispatch_counterfactual,
    compare_structural_and_statistical,
    dispatch_emissions_delta,
    load_validation_config,
)


def test_dispatch_emissions_delta_uses_output_factor_and_snapshot_hours() -> None:
    snapshots = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    baseline = pd.DataFrame({"gas": [10.0, 10.0]}, index=snapshots)
    counterfactual = pd.DataFrame({"gas": [11.0, 10.0]}, index=snapshots)
    metadata = pd.DataFrame(
        {"carrier": ["CCGT"], "efficiency": [0.5]}, index=["gas"]
    )

    result = dispatch_emissions_delta(
        baseline,
        counterfactual,
        metadata,
        pd.Series(1.0, index=snapshots),
        {"gas": 370.0},
        {"CCGT": "gas"},
    )

    assert result["emissions_delta_kgco2e"] == 370.0
    assert result["dispatch_mapping_coverage"] == 1.0


def test_disagreement_is_retained_as_high_model_risk() -> None:
    structural = pd.DataFrame(
        {
            "run_id": ["run-1"],
            "horizon_hours": [6],
            "structural_marginal_response_kgco2e_per_mwh": [400.0],
            "dispatch_mapping_coverage": [1.0],
        }
    )
    statistical = pd.DataFrame(
        {
            "horizon_hours": [6, 6],
            "estimator": ["constrained_regression", "double_machine_learning"],
            "statistical_estimate_kgco2e_per_mwh": [410.0, -100.0],
            "ci95_lower_kgco2e_per_mwh": [350.0, np.nan],
            "ci95_upper_kgco2e_per_mwh": [450.0, np.nan],
        }
    )

    report, comparison = compare_structural_and_statistical(
        structural,
        statistical,
        load_validation_config(),
        data_origin="pypsa_eur",
    )

    assert report["status"] == "model_risk"
    assert report["model_risk"] == "high"
    assert report["causal_claim_eligible"] is False
    assert report["recommended_action"] == "do_not_treat_as_structurally_validated"
    assert comparison.loc[
        comparison["estimator"].eq("double_machine_learning"), "agreement"
    ].item() is False


def test_synthetic_dispatch_counterfactual_is_explicitly_labeled() -> None:
    rows = 30
    dataset = pd.DataFrame(
        {
            "decision_id": [f"d-{index}" for index in range(rows)],
            "state_timestamp_utc": pd.date_range(
                "2026-01-01", periods=rows, freq="h", tz="UTC"
            ),
            "horizon_hours": [6, 12, 24] * 10,
            "eligible_for_fit": True,
            "pre_load_forecast_mw": np.linspace(40_000, 60_000, rows),
            "pre_renewable_forecast_mw": np.linspace(4_000, 12_000, rows),
            "pre_planned_unavailable_capacity_mw": 500.0,
            "pre_net_scheduled_import_mw": 1_000.0,
            "pre_day_ahead_import_capacity_mw": 10_000.0,
        }
    )
    factors = {
        "nuclear": 0.0,
        "gas": 370.0,
        "coal": 820.0,
        "hydro": 0.0,
        "bioenergy": 230.0,
    }

    result = build_synthetic_dispatch_counterfactual(
        dataset, load_validation_config(), factors
    )

    assert result["structural_source"].eq(
        "synthetic_zonal_merit_order_dispatch"
    ).all()
    assert result["structural_marginal_response_kgco2e_per_mwh"].ge(0).all()
    assert set(result["horizon_hours"]) == {6, 12, 24}
