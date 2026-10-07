"""Generate isolated synthetic workload evidence for estimator demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.causal.dml import load_dml_config, run_dml_estimator, write_dml_artifacts
from src.causal.marginal_response import (
    load_estimator_config,
    run_estimator,
    write_estimator_artifacts,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "reports/demo"
DEFAULT_MODEL_ROOT = ROOT / "models/demo/causal"


def generate_synthetic_evidence(
    *,
    decision_days: int = 90,
    decisions_per_day: int = 1,
    seed: int = 1729,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build a plausible national grid history plus measured flexible workloads."""
    if decision_days < 30 or decisions_per_day < 1:
        raise ValueError(
            "synthetic demo requires at least 30 days and one decision per day"
        )
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01", tz="UTC")
    day_offsets = np.linspace(0, 359, decision_days, dtype=int)
    end = start + pd.Timedelta(days=int(day_offsets[-1]) + 3)
    timestamps = pd.date_range(start, end, freq="h", inclusive="left")
    hour = timestamps.hour.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()
    weekend = timestamps.dayofweek.to_numpy() >= 5

    daily = np.sin(2 * np.pi * (hour - 8) / 24)
    seasonal = np.cos(2 * np.pi * (day_of_year - 20) / 365.25)
    solar_shape = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    wind = (
        5_500
        + 2_200 * np.sin(2 * np.pi * day_of_year / 9.5)
        + rng.normal(0, 650, len(timestamps))
    )
    solar = solar_shape * (4_500 - 1_800 * seasonal)
    renewable = np.maximum(1_000, wind + solar)
    load = 49_000 + 5_500 * daily + 4_000 * seasonal - 1_200 * weekend
    load += rng.normal(0, 500, len(timestamps))
    load_error = rng.normal(0, 380, len(timestamps))
    outage = np.where((day_of_year % 17) < 3, 1_400, 0) + np.where(
        (day_of_year % 43) == 0, 900, 0
    )
    net_import = (
        1_100
        + 1_500 * seasonal
        - 0.08 * renewable
        + rng.normal(0, 280, len(timestamps))
    )
    import_capacity = 10_500 + 700 * np.sin(2 * np.pi * day_of_year / 30)
    export_capacity = 9_800 + 600 * np.cos(2 * np.pi * day_of_year / 27)
    price = 55 + 0.0015 * load - 0.0024 * renewable + 0.006 * outage
    price += rng.normal(0, 3, len(timestamps))
    temperature = 12 - 9 * seasonal + 3 * np.sin(2 * np.pi * (hour - 14) / 24)

    mart = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "metadata_point_in_time_eligible": True,
            "metadata_snapshot_at_utc": timestamps.floor("D") - pd.Timedelta(days=2),
            "pre_hour_sin": np.sin(2 * np.pi * hour / 24),
            "pre_hour_cos": np.cos(2 * np.pi * hour / 24),
            "pre_day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "pre_day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
            "pre_is_weekend": weekend.astype(int),
            "pre_load_forecast_mw": load - load_error,
            "pre_load_forecast_error_lag_1h_mw": np.roll(load_error, 1),
            "pre_load_forecast_error_lag_24h_mw": np.roll(load_error, 24),
            "pre_renewable_forecast_mw": renewable,
            "pre_planned_unavailable_capacity_mw": outage,
            "pre_planned_outage_count": (outage > 0).astype(int),
            "pre_net_scheduled_import_mw": net_import,
            "pre_day_ahead_import_capacity_mw": import_capacity,
            "pre_day_ahead_export_capacity_mw": export_capacity,
            "pre_day_ahead_price_eur_mwh": price,
            "pre_weather_temperature_forecast_c_24h": temperature,
            "pre_gas_price_usd_mmbtu_lag_2m": 9 + 1.4 * seasonal,
            "pre_coal_price_usd_mt_lag_2m": 120 + 12 * seasonal,
            "pre_eua_auction_price_eur_tco2": 75
            + 5 * np.sin(2 * np.pi * day_of_year / 120),
            "pre_hydro_storage_mwh_lag_1w": 8_000_000 - 900_000 * seasonal,
        }
    )
    base_emissions = (
        3_800_000
        + 8 * load
        - 5 * renewable
        + 20 * outage
        - 2 * net_import
        + rng.normal(0, 500, len(timestamps))
    )
    mart["outcome_interconnected_direct_emissions_kgco2e_h0"] = base_emissions
    mart = mart.set_index("timestamp_utc", drop=False)

    observations = []
    true_effects = []
    recommended_followed = 0
    candidate_offsets = (3, 11, 19, 27, 35, 43)
    workload_types = ("data_center_batch", "ev_fleet_charging")
    for day_number, day_offset in enumerate(day_offsets):
        day = start + pd.Timedelta(days=int(day_offset))
        for sequence in range(decisions_per_day):
            access = day + pd.Timedelta(minutes=sequence * 10)
            candidates = [
                day + pd.Timedelta(hours=value) for value in candidate_offsets
            ]
            candidate_effects = np.array(
                [
                    true_marginal_response(mart.loc[timestamp])
                    for timestamp in candidates
                ]
            )
            recommended_position = int(np.argmin(candidate_effects))
            if rng.random() < 0.2:
                actual_position = recommended_position
                recommended_followed += 1
            else:
                actual_position = int(rng.integers(0, len(candidates)))
            actual_start = candidates[actual_position]
            duration = int(rng.integers(1, 4))
            energy_mwh = float(np.clip(rng.lognormal(np.log(2_200), 0.35), 800, 5_200))
            effect = float(candidate_effects[actual_position])
            hourly_energy = energy_mwh / duration
            for offset in range(duration):
                timestamp = actual_start + pd.Timedelta(hours=offset)
                mart.loc[
                    timestamp, "outcome_interconnected_direct_emissions_kgco2e_h0"
                ] += hourly_energy * effect
            observations.append(
                {
                    "decision_id": f"synthetic-{day_number:03d}-{sequence}",
                    "dashboard_accessed_at_utc": access,
                    "recommendation_generated_at_utc": access,
                    "baseline_source": "dashboard_access_time",
                    "baseline_start_utc": access,
                    "recommended_start_utc": candidates[recommended_position],
                    "user_selected_start_utc": actual_start,
                    "actual_start_utc": actual_start,
                    "actual_completion_utc": actual_start
                    + pd.Timedelta(hours=duration),
                    "actual_energy_kwh": energy_mwh * 1_000,
                    "candidate_alternatives": [
                        {
                            "timestamp_utc": timestamp.isoformat(),
                            "rank": rank,
                        }
                        for rank, timestamp in enumerate(
                            sorted(
                                candidates,
                                key=lambda value: true_marginal_response(
                                    mart.loc[value]
                                ),
                            ),
                            start=1,
                        )
                    ],
                    "workload_type": workload_types[
                        (day_number + sequence) % len(workload_types)
                    ],
                    "status": "completed",
                }
            )
            true_effects.append(effect)
    mart = mart.reset_index(drop=True)
    truth = {
        "data_origin": "synthetic",
        "seed": seed,
        "synthetic_scope": "aggregated_enterprise_compute_and_ev_portfolio",
        "completed_decisions": len(observations),
        "distinct_decision_days": decision_days,
        "recommendation_adherence_share": round(
            recommended_followed / len(observations), 4
        ),
        "mean_actual_energy_mwh": round(
            float(np.mean([row["actual_energy_kwh"] for row in observations])) / 1_000,
            3,
        ),
        "true_mean_marginal_response_kgco2e_per_mwh": round(
            float(np.mean(true_effects)), 3
        ),
        "production_eligible": False,
    }
    return pd.DataFrame(observations), mart, truth


def true_marginal_response(state: pd.Series) -> float:
    load = float(state["pre_load_forecast_mw"])
    renewable = float(state["pre_renewable_forecast_mw"])
    outage = float(state["pre_planned_unavailable_capacity_mw"])
    net_import = abs(float(state["pre_net_scheduled_import_mw"]))
    import_capacity = max(float(state["pre_day_ahead_import_capacity_mw"]), 1.0)
    congestion = net_import / import_capacity
    return max(
        25.0,
        105 + 0.003 * (load - renewable) + 0.018 * outage + 45 * congestion,
    )


def mark_synthetic(report: dict[str, Any]) -> dict[str, Any]:
    output = dict(report)
    output["computational_status"] = report["status"]
    output["status"] = (
        "synthetic_demo" if report["status"] == "ok" else "synthetic_demo_failed"
    )
    output["data_origin"] = "synthetic"
    output["production_eligible"] = False
    output["evidence_tier"] = "synthetic_pipeline_validation_only"
    return output


def run_synthetic_demo(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    decision_days: int = 90,
    decisions_per_day: int = 1,
    seed: int = 1729,
) -> dict[str, Any]:
    output_root = Path(output_root)
    causal_dir = output_root / "causal"
    metrics_dir = output_root / "metrics"
    causal_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    observations, mart, truth = generate_synthetic_evidence(
        decision_days=decision_days,
        decisions_per_day=decisions_per_day,
        seed=seed,
    )
    observations.to_csv(causal_dir / "synthetic_workload_observations.csv", index=False)
    mart.to_parquet(causal_dir / "synthetic_causal_feature_mart.parquet", index=False)
    (metrics_dir / "synthetic_ground_truth.json").write_text(
        json.dumps(truth, indent=2), encoding="utf-8"
    )

    base_config = load_estimator_config()
    simple_report, dataset, simple_decisions, simple_artifacts = run_estimator(
        observations, mart, base_config
    )
    simple_report = mark_synthetic(simple_report)
    write_estimator_artifacts(
        simple_report,
        dataset,
        simple_decisions,
        simple_artifacts,
        report_path=metrics_dir / "causal_identified_estimator.json",
        estimates_path=causal_dir / "marginal_response_estimates.csv",
        decisions_path=causal_dir / "confidently_avoided_emissions.csv",
        dataset_path=causal_dir / "observed_treatment_estimator_dataset.parquet",
        model_path=DEFAULT_MODEL_ROOT / "constrained_marginal_response.joblib",
    )

    dml_config = load_dml_config()
    dml_report, dml_decisions, heterogeneity, dml_models = run_dml_estimator(
        observations, mart, dml_config
    )
    dml_report = mark_synthetic(dml_report)
    write_dml_artifacts(
        dml_report,
        dml_decisions,
        heterogeneity,
        dml_models,
        report_path=metrics_dir / "causal_dml_estimator.json",
        estimates_path=causal_dir / "dml_marginal_response_estimates.csv",
        heterogeneity_path=causal_dir / "dml_heterogeneous_effects.csv",
        comparison_path=causal_dir / "dml_estimator_comparison.csv",
        decision_effects_path=causal_dir / "dml_decision_effects.csv",
        model_path=DEFAULT_MODEL_ROOT / "time_blocked_dml.joblib",
    )
    success = (
        simple_report["computational_status"] == "ok"
        and dml_report["computational_status"] == "ok"
    )
    summary = {
        "status": "ok" if success else "fail",
        "data_origin": "synthetic",
        "production_eligible": False,
        "completed_decisions": len(observations),
        "constrained_estimator_status": simple_report["status"],
        "dml_estimator_status": dml_report["status"],
        "output_root": str(output_root),
    }
    (metrics_dir / "synthetic_demo_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate isolated synthetic evidence and run both causal estimators."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--decision-days", type=int, default=90)
    parser.add_argument("--decisions-per-day", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)
    summary = run_synthetic_demo(
        args.output_root,
        decision_days=args.decision_days,
        decisions_per_day=args.decisions_per_day,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
