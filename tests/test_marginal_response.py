from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.causal.marginal_response import (
    ConstrainedMarginalResponseRegressor,
    build_estimator_dataset,
    load_estimator_config,
    run_estimator,
    time_block_split,
)
from src.data.causal_operational_snapshot import first_snapshot_sql


def test_constrained_response_is_nonnegative_and_state_varying() -> None:
    rng = np.random.default_rng(7)
    rows = 300
    state = rng.uniform(0, 1, rows)
    treatment = rng.uniform(0, 2, rows)
    frame = pd.DataFrame(
        {
            "pre_state": state,
            "pre_calendar": rng.normal(size=rows),
            "treatment_mwh": treatment,
            "outcome": 10 + 2 * state + treatment * (3 + 4 * state) + rng.normal(0, 0.1, rows),
        }
    )
    model = ConstrainedMarginalResponseRegressor(
        ["pre_state", "pre_calendar"], ["pre_state"], alpha=0.01
    ).fit(frame, "outcome")

    response = model.marginal_response(frame)

    assert (response >= -1e-8).all()
    assert response[state > 0.8].mean() > response[state < 0.2].mean()


def test_time_block_split_keeps_decisions_disjoint_and_embargoed() -> None:
    frame = pd.DataFrame(
        {
            "decision_id": [f"d{index}" for index in range(10) for _ in range(2)],
            "decision_at_utc": [
                pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=index)
                for index in range(10)
                for _ in range(2)
            ],
        }
    )

    train, test, metadata = time_block_split(frame, 0.6, 24)

    assert set(train["decision_id"]).isdisjoint(test["decision_id"])
    assert pd.Timestamp(test["decision_at_utc"].min()) > pd.Timestamp(metadata["test_start_utc"])


def test_estimator_blocks_without_observed_workloads() -> None:
    config = load_estimator_config()
    report, dataset, decisions, artifacts = run_estimator(
        pd.DataFrame(), pd.DataFrame(), config
    )

    assert report["status"] == "blocked"
    assert report["uses_load_forecast_error_as_treatment"] is False
    assert "no_observed_treatment_outcome_overlap" in report["blockers"]
    assert dataset.empty
    assert decisions.empty
    assert artifacts == {}


def test_dataset_uses_completed_dose_and_dynamic_response_windows() -> None:
    config = small_config()
    mart = synthetic_mart(days=5, config=config)
    observation = synthetic_observations(1).iloc[[0]]

    dataset = build_estimator_dataset(observation, mart, config)

    treated = dataset[dataset["treatment_observed"]]
    assert set(dataset["horizon_hours"]) == {6, 12, 24}
    assert treated["treatment_mwh"].eq(1.0).all()
    assert dataset.loc[~dataset["treatment_observed"], "treatment_mwh"].eq(0).all()
    assert dataset["point_in_time_eligible"].all()
    assert not any(column.startswith(("mediator_", "diagnostic_")) for column in config["adjustment_features"])


def test_dataset_rejects_snapshot_created_after_recommendation() -> None:
    config = small_config()
    mart = synthetic_mart(days=5, config=config)
    mart["metadata_snapshot_at_utc"] = pd.Timestamp("2026-01-02", tz="UTC")

    dataset = build_estimator_dataset(synthetic_observations(1), mart, config)

    assert not dataset["point_in_time_eligible"].any()


def test_operational_compaction_preserves_first_nonempty_snapshot() -> None:
    sql = first_snapshot_sql(
        "select * from source\non conflict (id) do update set value = excluded.value",
        "id",
    )

    assert sql.endswith("on conflict (id) do nothing")
    assert "do update" not in sql


def test_end_to_end_estimator_returns_all_horizons_on_eligible_data() -> None:
    config = small_config()
    config.update(
        {
            "minimum_completed_decisions": 20,
            "minimum_distinct_decision_days": 20,
            "minimum_train_decisions": 10,
            "minimum_test_decisions": 5,
            "minimum_overlap_share": 0.0,
            "bootstrap_iterations": 3,
            "minimum_bootstrap_successful_iterations": 1,
            "bootstrap_block_days": 3,
        }
    )

    report, _, decisions, artifacts = run_estimator(
        synthetic_observations(30), synthetic_mart(days=35, config=config), config
    )

    assert report["status"] == "ok"
    assert [row["horizon_hours"] for row in report["horizons"]] == [6, 12, 24]
    assert set(decisions["horizon_hours"]) == {6, 12, 24}
    assert decisions["confidently_avoided_kgco2e"].ge(0).all()
    assert set(artifacts["models"]) == {"6", "12", "24"}
    assert "ci80" in report["horizons"][0]["uncertainty_intervals"]


def small_config() -> dict:
    config = copy.deepcopy(load_estimator_config())
    config.update(
        {
            "minimum_completed_decisions": 1,
            "minimum_distinct_decision_days": 1,
            "minimum_train_decisions": 1,
            "minimum_test_decisions": 1,
            "bootstrap_iterations": 0,
            "adjustment_features": ["pre_hour_sin", "pre_hour_cos", "pre_load_forecast_mw"],
            "marginal_state_features": ["pre_hour_sin", "pre_load_forecast_mw"],
        }
    )
    return config


def synthetic_mart(days: int, config: dict) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=days * 24 + 48, freq="h", tz="UTC")
    hour = timestamps.hour.to_numpy()
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "metadata_point_in_time_eligible": True,
            "metadata_snapshot_at_utc": pd.Timestamp("2025-12-31", tz="UTC"),
            "outcome_interconnected_direct_emissions_kgco2e_h0": 1000 + hour * 8,
            "pre_hour_sin": np.sin(2 * np.pi * hour / 24),
            "pre_hour_cos": np.cos(2 * np.pi * hour / 24),
            "pre_load_forecast_mw": 40_000 + hour * 50,
        }
    )
    for feature in config["adjustment_features"]:
        if feature not in frame:
            frame[feature] = np.linspace(0, 1, len(frame))
    return frame


def synthetic_observations(count: int) -> pd.DataFrame:
    records = []
    for index in range(count):
        baseline = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=index)
        candidate_hours = [3, 6, 9, 12]
        actual = baseline + pd.Timedelta(hours=candidate_hours[index % len(candidate_hours)])
        records.append(
            {
                "decision_id": f"decision-{index}",
                "dashboard_accessed_at_utc": baseline,
                "recommendation_generated_at_utc": baseline,
                "baseline_source": "dashboard_access_time",
                "baseline_start_utc": baseline,
                "recommended_start_utc": baseline + pd.Timedelta(hours=6),
                "actual_start_utc": actual,
                "actual_completion_utc": actual + pd.Timedelta(hours=1),
                "actual_energy_kwh": 1000.0,
                "candidate_alternatives": [
                    {"timestamp_utc": (baseline + pd.Timedelta(hours=hour)).isoformat(), "rank": rank}
                    for rank, hour in enumerate(candidate_hours, start=1)
                ],
                "workload_type": "data_center_batch",
                "status": "completed",
            }
        )
    return pd.DataFrame(records)
