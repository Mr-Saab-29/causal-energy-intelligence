"""Operational next-24-hour clean-hour recommendations."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from src.carbon.intensity import load_emission_factor_config
from src.data.local_ingest import aggregate_weather
from src.features.price_features import build_price_modeling_features
from src.models.baseline_price import (
    PRODUCTION_SIGNAL_TARGETS,
    STRICT_FORECAST_FEATURES,
    TIMESTAMP_COLUMN,
    predict_model,
    predict_signal_model,
    supply_demand_feature_columns,
)
from src.optimization.workload_shift import (
    WorkloadConstraints,
    add_recommendation_confidence,
    apply_confidence_calibration,
    apply_saved_ranking_model_overlay,
    build_scenario_rerankings,
    build_top_workload_recommendations,
    build_workload_decision_rankings,
    load_confidence_calibration,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data/processed/modeling_price_features.csv"
FUTURE_WEATHER_PATH = ROOT / "data/processed/future_weather_forecast.csv"
OUTPUT_PATH = ROOT / "reports/recommendations/future_champion_workload_recommendations.csv"
RANKING_OUTPUT_PATH = ROOT / "reports/rankings/future_workload_decision_rankings.csv"
SCENARIO_OUTPUT_PATH = ROOT / "reports/scenarios/future_workload_scenario_recommendations.csv"
METADATA_OUTPUT_PATH = ROOT / "reports/metrics/future_recommendation_metadata.json"
EMISSION_FACTORS_PATH = ROOT / "config/emission_factors.yaml"
OPERATIONAL_RECOMMENDATION_HISTORY_PATH = (
    ROOT / "reports/monitoring/operational_recommendation_history.csv"
)
OPERATIONAL_RANKING_HISTORY_PATH = ROOT / "reports/monitoring/operational_ranking_history.csv"


@dataclass(frozen=True)
class FutureRecommendationSummary:
    """Summary for the operational future recommendation export."""

    generated_at_utc: str
    horizon_hours: int
    recommendation_rows: int
    scenario_recommendation_rows: int
    first_decision_group: str | None
    last_decision_group: str | None


def build_future_recommendations(
    horizon_hours: int = 24,
    output_path: str | Path = OUTPUT_PATH,
    ranking_output_path: str | Path = RANKING_OUTPUT_PATH,
    scenario_output_path: str | Path = SCENARIO_OUTPUT_PATH,
    metadata_output_path: str | Path = METADATA_OUTPUT_PATH,
    as_of_utc: str | pd.Timestamp | None = None,
) -> FutureRecommendationSummary:
    """Score the next horizon using saved champion artifacts."""
    history = pd.read_csv(FEATURES_PATH, parse_dates=[TIMESTAMP_COLUMN])
    history[TIMESTAMP_COLUMN] = pd.to_datetime(history[TIMESTAMP_COLUMN], utc=True)
    future = build_future_feature_frame(history, horizon_hours, as_of_utc=as_of_utc)
    future = add_future_signal_predictions(future)
    future = add_future_price_predictions(future)
    hourly = build_future_hourly_decision_inputs(history, future)
    rankings = build_workload_decision_rankings(
        hourly,
        WorkloadConstraints(price_weight=0.5, carbon_weight=0.5),
    )
    rankings = apply_saved_ranking_model_overlay(rankings)
    recommendations = build_top_workload_recommendations(rankings, top_n=5)
    recommendations = add_recommendation_confidence(recommendations, rankings, top_n=5)
    recommendations = apply_confidence_calibration(
        recommendations,
        load_confidence_calibration(),
    )
    scenario_recommendations, _ = build_scenario_rerankings(rankings, top_n=5)
    recommendations["is_future_recommendation"] = True
    rankings["is_future_recommendation"] = True
    scenario_recommendations["is_future_recommendation"] = True

    write_csv(ranking_output_path, rankings)
    write_csv(output_path, recommendations)
    write_csv(scenario_output_path, scenario_recommendations)
    summary = FutureRecommendationSummary(
        generated_at_utc=datetime.now(UTC).isoformat(),
        horizon_hours=horizon_hours,
        recommendation_rows=int(len(recommendations)),
        scenario_recommendation_rows=int(len(scenario_recommendations)),
        first_decision_group=str(recommendations["decision_group"].min()) if not recommendations.empty else None,
        last_decision_group=str(recommendations["decision_group"].max()) if not recommendations.empty else None,
    )
    append_operational_history(
        recommendations,
        OPERATIONAL_RECOMMENDATION_HISTORY_PATH,
        summary.generated_at_utc,
    )
    append_operational_history(
        rankings,
        OPERATIONAL_RANKING_HISTORY_PATH,
        summary.generated_at_utc,
    )
    write_json(metadata_output_path, asdict(summary))
    return summary


def build_future_feature_frame(
    history: pd.DataFrame,
    horizon_hours: int,
    as_of_utc: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build model-ready rows for future timestamps."""
    latest = history[TIMESTAMP_COLUMN].max()
    forecast_start = calculate_future_forecast_start(latest, as_of_utc=as_of_utc)
    scoring_timestamps = pd.date_range(
        forecast_start,
        periods=horizon_hours,
        freq="h",
        tz="UTC",
    )
    feature_timestamps = pd.date_range(
        latest + pd.Timedelta(hours=1),
        scoring_timestamps.max(),
        freq="h",
        tz="UTC",
    )
    future_base = pd.DataFrame({TIMESTAMP_COLUMN: feature_timestamps})
    future_base = future_base.merge(load_future_weather_aggregates(), on=TIMESTAMP_COLUMN, how="left")
    future_base = fill_missing_future_weather(future_base, history)
    for column in required_base_columns():
        if column not in future_base:
            future_base[column] = np.nan
    future_base = fill_future_target_placeholders(future_base, history)

    combined = pd.concat([history[required_base_columns()], future_base], ignore_index=True, sort=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        features = build_price_modeling_features(combined)
    features = remove_duplicate_columns(features)
    return features[features[TIMESTAMP_COLUMN].isin(scoring_timestamps)].reset_index(drop=True)


def calculate_future_forecast_start(
    latest_history_timestamp: pd.Timestamp,
    as_of_utc: str | pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Start future scoring after both latest data and current operational time."""
    latest = pd.Timestamp(latest_history_timestamp)
    if latest.tzinfo is None:
        latest = latest.tz_localize("UTC")
    else:
        latest = latest.tz_convert("UTC")
    as_of = pd.Timestamp.now(tz="UTC") if as_of_utc is None else pd.Timestamp(as_of_utc)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")
    return max(
        latest + pd.Timedelta(hours=1),
        as_of.floor("h") + pd.Timedelta(hours=1),
    )


def remove_duplicate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with unique column names, preserving first occurrence."""
    return frame.loc[:, ~frame.columns.duplicated()].copy()


def add_future_signal_predictions(future: pd.DataFrame) -> pd.DataFrame:
    """Predict future consumption, production, and source generation."""
    output = future.copy()
    for target in ["consumption", *PRODUCTION_SIGNAL_TARGETS]:
        model_name, model = load_model_for_target(target)
        _, prefix = target_to_column_prefix(target)
        predictions = predict_signal_model(
            model_name,
            model,
            output,
            supply_demand_feature_columns(prefix),
            prefix,
        )
        if target == "consumption":
            output["forecast_consumption_mwh"] = predictions
        elif target == "production":
            output["forecast_total_production_mwh"] = predictions
        output[f"forecast_{target}_mwh"] = predictions
    output["forecast_residual_demand_mwh"] = (
        output["forecast_consumption_mwh"] - output["forecast_total_production_mwh"]
    )
    output["forecast_supply_demand_gap_mwh"] = (
        output["forecast_total_production_mwh"] - output["forecast_consumption_mwh"]
    )
    return output


def add_future_price_predictions(future: pd.DataFrame) -> pd.DataFrame:
    """Predict future prices with the saved selected price model."""
    model_name, model = load_price_model()
    output = future.copy()
    output["model"] = model_name
    output["window"] = "future_24h"
    output["predicted_price_eur_mwh"] = predict_model(model_name, model, output)
    return output


def build_future_hourly_decision_inputs(history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """Create hourly decision input rows for the next 24 hours."""
    factors = load_emission_factor_config(EMISSION_FACTORS_PATH)["direct_operational_emissions"]
    source_columns = [f"forecast_{source}_mwh" for source in PRODUCTION_SIGNAL_TARGETS[1:]]
    generation = future[source_columns].clip(lower=0)
    generation.columns = PRODUCTION_SIGNAL_TARGETS[1:]
    emissions = sum(generation[source] * factors[source] for source in generation.columns)
    total_generation = generation.sum(axis=1).replace(0, np.nan)
    previous_day = history[[TIMESTAMP_COLUMN, "price_eur_mwh"]].copy()
    previous_day[TIMESTAMP_COLUMN] = previous_day[TIMESTAMP_COLUMN] + pd.Timedelta(days=1)
    output = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: future[TIMESTAMP_COLUMN],
            "window": "future_24h",
            "model": future["model"],
            "decision_date": future[TIMESTAMP_COLUMN].dt.date.astype(str),
            "actual_price_eur_mwh": future["predicted_price_eur_mwh"],
            "predicted_price_eur_mwh": future["predicted_price_eur_mwh"],
            "actual_carbon_intensity_g_co2e_per_kwh": emissions / total_generation,
            "predicted_carbon_intensity_g_co2e_per_kwh": emissions / total_generation,
            "actual_total_emissions_kg_co2e": emissions,
            "predicted_total_emissions_kg_co2e": emissions,
        }
    )
    output = output.merge(
        previous_day.rename(columns={"price_eur_mwh": "previous_day_price_eur_mwh"}),
        on=TIMESTAMP_COLUMN,
        how="left",
    )
    return output


def load_future_weather_aggregates() -> pd.DataFrame:
    """Load future weather forecast aggregates if available."""
    if not FUTURE_WEATHER_PATH.exists():
        return pd.DataFrame(columns=[TIMESTAMP_COLUMN])
    weather = pd.read_csv(FUTURE_WEATHER_PATH, parse_dates=[TIMESTAMP_COLUMN])
    weather[TIMESTAMP_COLUMN] = pd.to_datetime(weather[TIMESTAMP_COLUMN], utc=True)
    return aggregate_weather(weather)


def fill_missing_future_weather(future: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Fill unavailable future weather fields from the latest observed values."""
    output = future.copy()
    weather_columns = [column for column in required_weather_columns() if column in output]
    latest_weather = history.sort_values(TIMESTAMP_COLUMN).iloc[-1]
    for column in weather_columns:
        if column in latest_weather:
            output[column] = output[column].fillna(latest_weather[column])
    return output


def fill_future_target_placeholders(future: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Fill unknown future targets with lag-based placeholders for feature construction."""
    output = future.copy()
    lag_lookup = history.set_index(TIMESTAMP_COLUMN)
    latest_row = history.sort_values(TIMESTAMP_COLUMN).iloc[-1]
    target_columns = [
        "price_eur_mwh",
        "consumption_mwh",
        "total_production_mwh",
        "nuclear_mwh",
        "thermal_mwh",
        "gas_mwh",
        "coal_mwh",
        "oil_mwh",
        "wind_mwh",
        "solar_mwh",
        "hydro_mwh",
        "bioenergy_mwh",
        "physical_exchanges_mwh",
    ]
    for index, row in output.iterrows():
        lag_timestamp = row[TIMESTAMP_COLUMN] - pd.Timedelta(hours=24)
        for column in target_columns:
            if column not in output:
                continue
            lag_value = lag_lookup[column].get(lag_timestamp) if column in lag_lookup else np.nan
            output.at[index, column] = lag_value if pd.notna(lag_value) else latest_row.get(column, np.nan)
    return output


def required_base_columns() -> list[str]:
    """Return base columns needed by the feature builder."""
    return [
        TIMESTAMP_COLUMN,
        "price_eur_mwh",
        "consumption_mwh",
        "total_production_mwh",
        "nuclear_mwh",
        "thermal_mwh",
        "gas_mwh",
        "coal_mwh",
        "oil_mwh",
        "wind_mwh",
        "solar_mwh",
        "hydro_mwh",
        "bioenergy_mwh",
        "physical_exchanges_mwh",
        *required_weather_columns(),
    ]


def required_weather_columns() -> list[str]:
    """Return weather aggregate columns used by feature engineering."""
    return [
        "avg_temperature_c",
        "min_temperature_c",
        "max_temperature_c",
        "avg_apparent_temperature_c",
        "avg_wind_speed_mps",
        "avg_wind_speed_80m_mps",
        "avg_shortwave_radiation_wm2",
        "avg_cloud_cover_pct",
        "avg_precipitation_mm",
        "total_precipitation_mm",
        "avg_surface_pressure_hpa",
    ]


def load_model_for_target(target: str) -> tuple[str, Any]:
    """Load the persisted selected model for a signal target."""
    matches = sorted((ROOT / "models").glob(f"*_{target}_baseline.joblib"))
    if not matches:
        raise FileNotFoundError(f"No saved model artifact found for target {target!r}")
    model_name = matches[0].name.removesuffix(f"_{target}_baseline.joblib")
    return model_name, joblib.load(matches[0])


def load_price_model() -> tuple[str, Any]:
    """Load the persisted selected price model."""
    matches = sorted((ROOT / "models").glob("*_price_baseline.joblib"))
    if not matches:
        raise FileNotFoundError("No saved price model artifact found")
    model_name = matches[0].name.removesuffix("_price_baseline.joblib")
    return model_name, joblib.load(matches[0])


def target_to_column_prefix(target: str) -> tuple[str, str]:
    """Return target column and feature prefix metadata."""
    if target == "consumption":
        return "consumption_mwh", "consumption"
    if target == "production":
        return "total_production_mwh", "total_production"
    return f"{target}_mwh", target


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    """Write CSV output with parent directories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON output with parent directories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_operational_history(
    frame: pd.DataFrame,
    path: str | Path,
    generated_at_utc: str,
) -> None:
    """Append operational forecast rows for later actual-vs-forecast monitoring."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = frame.copy()
    history["forecast_generated_at_utc"] = generated_at_utc
    history.to_csv(output, mode="a", index=False, header=not output.exists())


def main(argv: list[str] | None = None) -> None:
    """Build future recommendations from the command line."""
    parser = argparse.ArgumentParser(description="Build next-24-hour clean-hour recommendations.")
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--as-of-utc", default=None)
    args = parser.parse_args(argv)
    summary = build_future_recommendations(
        horizon_hours=args.horizon_hours,
        as_of_utc=args.as_of_utc,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
