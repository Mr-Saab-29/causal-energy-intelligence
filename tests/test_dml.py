from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.causal.dml import fit_dml_horizon, load_dml_config, rolling_time_cross_fit


def test_rolling_cross_fit_uses_only_embargoed_past_decisions() -> None:
    config = dml_test_config()
    frame = synthetic_panel(40)

    residuals, folds = rolling_time_cross_fit(frame, config)

    assert len(folds) == config["cross_fit_folds"]
    assert residuals["cross_fit_fold"].nunique() == config["cross_fit_folds"]
    for fold in folds:
        train_end = pd.Timestamp(fold["train_end_utc"])
        validation_start = pd.Timestamp(fold["validation_start_utc"])
        assert (
            train_end + pd.Timedelta(hours=config["embargo_hours"]) < validation_start
        )


def test_dml_estimates_requested_heterogeneity_and_comparison() -> None:
    config = dml_test_config()

    report, decisions, heterogeneity, artifact = fit_dml_horizon(
        synthetic_panel(90), 6, config
    )

    assert report["status"] == "ok"
    assert report["cross_fit_rows"] > 0
    assert set(heterogeneity["dimension"]) == {
        "hour",
        "season",
        "renewable_regime",
        "outage_regime",
        "congestion_regime",
    }
    renewable = heterogeneity[heterogeneity["dimension"].eq("renewable_regime")]
    effects = renewable.set_index("level")["mean_marginal_response_kgco2e_per_mwh"]
    assert effects["high"] > effects["low"]
    assert report["comparison"]["status"] in {"agreement", "review"}
    assert not decisions.empty
    assert artifact is not None


def test_dml_blocks_when_cross_fit_history_is_too_short() -> None:
    config = dml_test_config()
    config["minimum_cross_fit_train_decisions"] = 100

    report, decisions, heterogeneity, artifact = fit_dml_horizon(
        synthetic_panel(40), 6, config
    )

    assert report["status"] == "blocked"
    assert report["blockers"] == ["insufficient_time_cross_fit_folds"]
    assert decisions.empty
    assert heterogeneity.empty
    assert artifact is None


def dml_test_config() -> dict:
    config = copy.deepcopy(load_dml_config())
    config.update(
        {
            "adjustment_features": [
                "pre_hour_sin",
                "pre_hour_cos",
                "pre_renewable_forecast_mw",
                "pre_planned_unavailable_capacity_mw",
                "pre_net_scheduled_import_mw",
                "pre_day_ahead_import_capacity_mw",
                "pre_day_ahead_export_capacity_mw",
            ],
            "marginal_state_features": [
                "pre_hour_sin",
                "pre_renewable_forecast_mw",
            ],
            "train_share": 0.7,
            "minimum_train_decisions": 20,
            "minimum_test_decisions": 10,
            "cross_fit_folds": 3,
            "cross_fit_initial_train_share": 0.3,
            "minimum_cross_fit_train_decisions": 10,
            "minimum_cross_fit_validation_decisions": 3,
            "minimum_treatment_residual_std_mwh": 0.001,
            "nuisance_model": {
                "max_iter": 80,
                "learning_rate": 0.08,
                "max_leaf_nodes": 10,
                "min_samples_leaf": 5,
                "l2_regularization": 0.1,
            },
        }
    )
    return config


def synthetic_panel(decisions: int) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    records = []
    for decision in range(decisions):
        decision_at = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=decision)
        treated_position = int(rng.integers(0, 4))
        for position, hour in enumerate((1, 7, 13, 19)):
            timestamp = decision_at + pd.Timedelta(hours=hour)
            renewable = 2_000 + 600 * position + 8 * decision
            treatment = float(position == treated_position)
            effect = 40 + 0.012 * renewable + 30 * (hour >= 12)
            outcome = (
                2_000
                + 80 * np.sin(2 * np.pi * hour / 24)
                + 0.15 * renewable
                + treatment * effect
                + rng.normal(0, 2)
            )
            records.append(
                {
                    "decision_id": f"d-{decision}",
                    "decision_at_utc": decision_at,
                    "state_timestamp_utc": timestamp,
                    "state_roles": "actual" if treatment else "baseline|candidate",
                    "horizon_hours": 6,
                    "actual_energy_mwh": 1.0,
                    "treatment_mwh": treatment,
                    "treatment_observed": bool(treatment),
                    "eligible_for_fit": True,
                    "outcome_interconnected_emissions_kgco2e": outcome,
                    "pre_hour_sin": np.sin(2 * np.pi * hour / 24),
                    "pre_hour_cos": np.cos(2 * np.pi * hour / 24),
                    "pre_renewable_forecast_mw": renewable,
                    "pre_planned_unavailable_capacity_mw": 500
                    if decision % 5 == 0
                    else 0,
                    "pre_net_scheduled_import_mw": 300 + position * 250,
                    "pre_day_ahead_import_capacity_mw": 1_200,
                    "pre_day_ahead_export_capacity_mw": 1_000,
                }
            )
    return pd.DataFrame(records)
