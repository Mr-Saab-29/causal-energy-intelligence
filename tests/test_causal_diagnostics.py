from __future__ import annotations

import copy

import numpy as np
import pandas as pd

import src.causal.diagnostics as diagnostics
from src.causal.diagnostics import (
    add_placebo_treatments,
    block_bootstrap_association,
    load_diagnostics_config,
    select_diagnostic_inputs,
    stability_diagnostics,
    standardized_mean_difference,
)


def test_placebo_treatments_are_shifted_without_changing_decision_dose() -> None:
    frame = pd.DataFrame(
        {
            "decision_id": ["a"] * 3 + ["b"] * 3,
            "decision_at_utc": pd.to_datetime(
                ["2025-01-01"] * 3 + ["2025-01-02"] * 3, utc=True
            ),
            "state_timestamp_utc": pd.date_range(
                "2025-01-01", periods=6, freq="h", tz="UTC"
            ),
            "treatment_mwh": [0.0, 2.0, 0.0, 3.0, 0.0, 0.0],
        }
    )

    result = add_placebo_treatments(frame)

    original = result.groupby("decision_id")["treatment_mwh"].sum()
    shifted = result.groupby("decision_id")["placebo_shifted_treatment_mwh"].sum()
    pd.testing.assert_series_equal(original, shifted, check_names=False)
    assert result.loc[result["decision_id"].eq("a"), "lead_treatment_mwh"].tolist() == [
        3.0,
        0.0,
        0.0,
    ]


def test_weighted_standardized_difference_detects_balance() -> None:
    values = np.array([0.0, 1.0, 2.0, 3.0])
    treated = np.array([True, False, True, False])

    difference = standardized_mean_difference(values, treated)

    assert difference is not None
    assert difference < 0


def test_block_bootstrap_association_returns_time_robust_interval() -> None:
    config = copy.deepcopy(load_diagnostics_config())
    config["bootstrap_iterations"] = 40
    rows = 60
    rng = np.random.default_rng(9)
    treatment = rng.normal(size=rows)
    residuals = pd.DataFrame(
        {
            "decision_id": [f"d-{index}" for index in range(rows)],
            "decision_at_utc": pd.date_range(
                "2025-01-01", periods=rows, freq="D", tz="UTC"
            ),
            "treatment_residual": treatment,
            "outcome_residual": 4 * treatment + rng.normal(0, 0.2, rows),
        }
    )

    result = block_bootstrap_association(residuals, config)

    assert result["status"] == "available"
    assert result["estimate"] > 3.5
    assert result["bootstrap_successful_iterations"] == 40
    assert result["intervals"]["ci95"]["lower"] > 0


def test_stability_diagnostics_reports_unobserved_regimes() -> None:
    class ConstantEffectModel:
        renewable_edges = (2_000.0, 4_000.0)

        @staticmethod
        def marginal_response(frame: pd.DataFrame) -> np.ndarray:
            return np.full(len(frame), 100.0)

    frame = pd.DataFrame(
        {
            "decision_id": [f"d-{index}" for index in range(12)],
            "decision_at_utc": pd.date_range(
                "2025-01-01", periods=12, freq="D", tz="UTC"
            ),
            "state_timestamp_utc": pd.date_range(
                "2025-01-01", periods=12, freq="D", tz="UTC"
            ),
            "pre_renewable_forecast_mw": 1_000.0,
            "pre_planned_unavailable_capacity_mw": 0.0,
            "pre_net_scheduled_import_mw": 100.0,
            "pre_day_ahead_import_capacity_mw": 1_000.0,
            "pre_day_ahead_export_capacity_mw": 1_000.0,
        }
    )
    config = copy.deepcopy(load_diagnostics_config())
    config["bootstrap_iterations"] = 10

    summary, _ = stability_diagnostics(
        frame, ConstantEffectModel(), config, horizon=6
    )

    assert summary["status"] == "warn"
    assert "season:summer" in summary["sparse_regimes"]
    assert "renewable_regime:high" in summary["sparse_regimes"]


def test_missing_observed_models_use_approved_synthetic_product_fallback(
    tmp_path, monkeypatch
) -> None:
    synthetic_dataset = tmp_path / "synthetic.parquet"
    synthetic_models = tmp_path / "synthetic.joblib"
    synthetic_dataset.touch()
    synthetic_models.touch()
    monkeypatch.setattr(
        diagnostics, "DEFAULT_SYNTHETIC_DATASET_PATH", synthetic_dataset
    )
    monkeypatch.setattr(
        diagnostics, "DEFAULT_SYNTHETIC_MODEL_PATH", synthetic_models
    )
    config = load_diagnostics_config()

    dataset, models, origin, blocker, reason = select_diagnostic_inputs(
        tmp_path / "missing-observed.parquet",
        tmp_path / "missing-observed.joblib",
        config,
    )

    assert dataset == synthetic_dataset
    assert models == synthetic_models
    assert origin == "synthetic_product_bootstrap"
    assert blocker is None
    assert reason is not None
