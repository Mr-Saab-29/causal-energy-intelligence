"""Settled recommendation outcome audit for operational clean-hour decisions."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.carbon.intensity import load_emission_factor_config
from src.models.baseline_price import TIMESTAMP_COLUMN
from src.monitoring.forecast_monitor import recompute_operational_actual_ranks

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data/processed/modeling_price_features.csv"
OPERATIONAL_RECOMMENDATION_HISTORY_PATH = (
    ROOT / "reports/monitoring/operational_recommendation_history.csv"
)
OPERATIONAL_RANKING_HISTORY_PATH = ROOT / "reports/monitoring/operational_ranking_history.csv"
OUTPUT_PATH = ROOT / "reports/monitoring/recommendation_outcome_audit.csv"
METRICS_OUTPUT_PATH = ROOT / "reports/metrics/recommendation_outcome_audit.json"
EMISSION_FACTORS_PATH = ROOT / "config/emission_factors.yaml"
SOURCE_TARGETS = ("nuclear", "gas", "coal", "oil", "wind", "solar", "hydro", "bioenergy")


def build_recommendation_outcome_audit(
    features_path: str | Path = FEATURES_PATH,
    recommendation_history_path: str | Path = OPERATIONAL_RECOMMENDATION_HISTORY_PATH,
    ranking_history_path: str | Path = OPERATIONAL_RANKING_HISTORY_PATH,
    output_path: str | Path = OUTPUT_PATH,
    metrics_output_path: str | Path = METRICS_OUTPUT_PATH,
    recent_days: int | None = None,
) -> dict[str, Any]:
    """Join operational recommendations to settled actual values and write audit artifacts."""
    features = read_csv(features_path, parse_dates=[TIMESTAMP_COLUMN])
    recommendations = read_csv(recommendation_history_path, parse_dates=[TIMESTAMP_COLUMN])
    rankings = read_csv(ranking_history_path, parse_dates=[TIMESTAMP_COLUMN])
    if features.empty or recommendations.empty or rankings.empty:
        audit = pd.DataFrame()
        summary = unavailable_summary(features, recommendations, rankings)
        write_csv(output_path, audit)
        write_json(metrics_output_path, summary)
        return summary

    features = normalize_timestamps(features)
    recommendations = normalize_timestamps(recommendations)
    rankings = normalize_timestamps(rankings)
    latest_actual = features[TIMESTAMP_COLUMN].max()
    if recent_days is not None:
        cutoff = latest_actual - pd.Timedelta(days=recent_days)
        recommendations = recommendations[recommendations[TIMESTAMP_COLUMN] >= cutoff].copy()
        rankings = rankings[rankings[TIMESTAMP_COLUMN] >= cutoff].copy()

    settled_rankings = rankings[rankings[TIMESTAMP_COLUMN] <= latest_actual].copy()
    settled_recommendations = recommendations[
        recommendations[TIMESTAMP_COLUMN] <= latest_actual
    ].copy()
    if settled_rankings.empty or settled_recommendations.empty:
        audit = pd.DataFrame()
        summary = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "available": False,
            "reason": "no settled recommendations overlap actual data yet",
            "latest_actual_timestamp_utc": latest_actual.isoformat(),
            "rows": 0,
        }
        write_csv(output_path, audit)
        write_json(metrics_output_path, summary)
        return summary

    actuals = build_actual_values(features)
    settled = settled_rankings.drop(
        columns=["previous_day_price_eur_mwh_observed"],
        errors="ignore",
    )
    settled = settled.merge(actuals, on=TIMESTAMP_COLUMN, how="inner")
    if settled.empty:
        audit = pd.DataFrame()
        summary = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "available": False,
            "reason": "settled ranking rows did not join to actual rows",
            "latest_actual_timestamp_utc": latest_actual.isoformat(),
            "rows": 0,
        }
        write_csv(output_path, audit)
        write_json(metrics_output_path, summary)
        return summary

    settled = recompute_operational_actual_ranks(settled)
    audit = select_recommended_rows(settled_recommendations, settled)
    audit = add_stage_errors(audit)
    audit = audit.sort_values(
        [
            "decision_group",
            "forecast_generated_at_utc",
            "model",
            "recommendation_rank",
            TIMESTAMP_COLUMN,
        ],
        na_position="last",
    ).reset_index(drop=True)
    summary = summarize_audit(audit, latest_actual)
    write_csv(output_path, audit)
    write_json(metrics_output_path, summary)
    return summary


def build_actual_values(features: pd.DataFrame) -> pd.DataFrame:
    """Build actual price, consumption, generation, and carbon values by hour."""
    factors = load_emission_factor_config(EMISSION_FACTORS_PATH)["direct_operational_emissions"]
    output = features[
        [TIMESTAMP_COLUMN, "price_eur_mwh", "consumption_mwh", "total_production_mwh"]
    ].copy()
    output = output.rename(
        columns={
            "price_eur_mwh": "actual_price_eur_mwh_observed",
            "consumption_mwh": "actual_consumption_mwh_observed",
            "total_production_mwh": "actual_total_production_mwh_observed",
        }
    )
    emissions = pd.Series(0.0, index=features.index)
    for source in SOURCE_TARGETS:
        source_column = f"{source}_mwh"
        observed_column = f"actual_{source}_generation_mwh_observed"
        source_values = pd.to_numeric(features[source_column], errors="coerce").clip(lower=0)
        output[observed_column] = source_values
        emissions = emissions + source_values * float(factors[source])
    output["actual_total_emissions_kg_co2e_observed"] = emissions
    output["actual_carbon_intensity_g_co2e_per_kwh_observed"] = (
        emissions / output["actual_total_production_mwh_observed"].replace(0, np.nan)
    )
    previous_day = features[[TIMESTAMP_COLUMN, "price_eur_mwh"]].copy()
    previous_day[TIMESTAMP_COLUMN] = previous_day[TIMESTAMP_COLUMN] + pd.Timedelta(days=1)
    return output.merge(
        previous_day.rename(columns={"price_eur_mwh": "previous_day_price_eur_mwh_observed"}),
        on=TIMESTAMP_COLUMN,
        how="left",
    )


def select_recommended_rows(recommendations: pd.DataFrame, settled: pd.DataFrame) -> pd.DataFrame:
    """Return settled ranking rows that correspond to persisted recommendations."""
    key_columns = ["forecast_generated_at_utc", "window", "model", "decision_group", TIMESTAMP_COLUMN]
    available_keys = [column for column in key_columns if column in recommendations and column in settled]
    if not available_keys:
        return pd.DataFrame()
    recommendation_context = recommendations[
        available_keys
        + [
            column
            for column in [
                "recommendation_rank",
                "recommendation_status",
                "suppressed_by_uncertainty_guard",
                "confidence_score",
                "confidence_level",
            ]
            if column in recommendations
        ]
    ].drop_duplicates(subset=available_keys)
    return recommendation_context.merge(
        settled,
        on=available_keys,
        how="inner",
        suffixes=("", "_ranking"),
    )


def add_stage_errors(frame: pd.DataFrame) -> pd.DataFrame:
    """Add forecast-vs-actual errors for dashboard outcome inspection."""
    output = frame.copy()
    error_pairs = {
        "price": ("predicted_avg_price_eur_mwh", "actual_price_eur_mwh_observed"),
        "carbon_intensity": (
            "predicted_avg_carbon_intensity_g_co2e_per_kwh",
            "actual_carbon_intensity_g_co2e_per_kwh_observed",
        ),
        "total_emissions": (
            "predicted_total_emissions_kg_co2e",
            "actual_total_emissions_kg_co2e_observed",
        ),
        "consumption": ("predicted_consumption_mwh", "actual_consumption_mwh_observed"),
        "total_production": (
            "predicted_total_production_mwh",
            "actual_total_production_mwh_observed",
        ),
    }
    for source in SOURCE_TARGETS:
        error_pairs[f"{source}_generation"] = (
            f"predicted_{source}_generation_mwh",
            f"actual_{source}_generation_mwh_observed",
        )
    for label, (predicted_column, actual_column) in error_pairs.items():
        if predicted_column in output and actual_column in output:
            output[f"{label}_error"] = (
                pd.to_numeric(output[predicted_column], errors="coerce")
                - pd.to_numeric(output[actual_column], errors="coerce")
            )
            output[f"{label}_absolute_error"] = output[f"{label}_error"].abs()
    return output


def summarize_audit(audit: pd.DataFrame, latest_actual: pd.Timestamp) -> dict[str, Any]:
    """Build compact outcome metrics for dashboard and CI artifacts."""
    if audit.empty:
        return {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "available": False,
            "reason": "no settled recommendation rows",
            "latest_actual_timestamp_utc": latest_actual.isoformat(),
            "rows": 0,
        }
    top_1 = audit[audit["recommendation_rank"] == 1].copy()
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "available": True,
        "latest_actual_timestamp_utc": latest_actual.isoformat(),
        "rows": int(len(audit)),
        "decision_groups": int(audit["decision_group"].nunique()),
        "forecast_generations": int(audit["forecast_generated_at_utc"].nunique()),
        "top_1_hit_rate": safe_mean(top_1["is_actual_best_observed"]),
        "top_5_hit_rate": safe_mean(top_1["actual_decision_rank_observed"] <= 5),
        "mean_actual_rank_of_top_1": safe_mean(top_1["actual_decision_rank_observed"]),
        "mean_combined_regret": safe_mean(top_1["combined_regret_observed"]),
        "mean_carbon_regret_g_co2e_per_kwh": safe_mean(
            top_1["carbon_regret_g_co2e_per_kwh_observed"]
        ),
        "price_mae_eur_mwh": safe_mean(audit.get("price_absolute_error")),
        "carbon_intensity_mae_g_co2e_per_kwh": safe_mean(
            audit.get("carbon_intensity_absolute_error")
        ),
        "consumption_mae_mwh": safe_mean(audit.get("consumption_absolute_error")),
        "total_production_mae_mwh": safe_mean(audit.get("total_production_absolute_error")),
        "source_generation_smape": summarize_source_smape(audit),
        "by_day": summarize_by_day(audit),
    }


def summarize_by_day(audit: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize recommendation outcome metrics by decision group."""
    rows: list[dict[str, Any]] = []
    for decision_group, group in audit.groupby("decision_group", observed=True):
        top_1 = group[group["recommendation_rank"] == 1]
        rows.append(
            {
                "decision_group": str(decision_group),
                "rows": int(len(group)),
                "top_1_hit_rate": safe_mean(top_1["is_actual_best_observed"]),
                "top_5_hit_rate": safe_mean(top_1["actual_decision_rank_observed"] <= 5),
                "mean_actual_rank_of_top_1": safe_mean(top_1["actual_decision_rank_observed"]),
                "mean_combined_regret": safe_mean(top_1["combined_regret_observed"]),
                "mean_carbon_regret_g_co2e_per_kwh": safe_mean(
                    top_1["carbon_regret_g_co2e_per_kwh_observed"]
                ),
                "price_mae_eur_mwh": safe_mean(group.get("price_absolute_error")),
                "carbon_intensity_mae_g_co2e_per_kwh": safe_mean(
                    group.get("carbon_intensity_absolute_error")
                ),
                "consumption_mae_mwh": safe_mean(group.get("consumption_absolute_error")),
                "total_production_mae_mwh": safe_mean(
                    group.get("total_production_absolute_error")
                ),
            }
        )
    return sorted(rows, key=lambda row: row["decision_group"], reverse=True)


def summarize_source_smape(audit: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize source-generation sMAPE for settled recommendations."""
    rows: list[dict[str, Any]] = []
    for source in SOURCE_TARGETS:
        actual_column = f"actual_{source}_generation_mwh_observed"
        predicted_column = f"predicted_{source}_generation_mwh"
        if actual_column not in audit or predicted_column not in audit:
            continue
        rows.append(
            {
                "source": source,
                "smape": smape(audit[actual_column], audit[predicted_column]),
                "mae_mwh": safe_mean(
                    (
                        pd.to_numeric(audit[predicted_column], errors="coerce")
                        - pd.to_numeric(audit[actual_column], errors="coerce")
                    ).abs()
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: -1 if row["smape"] is None else row["smape"],
        reverse=True,
    )


def unavailable_summary(
    features: pd.DataFrame,
    recommendations: pd.DataFrame,
    rankings: pd.DataFrame,
) -> dict[str, Any]:
    """Return a diagnostic summary when required inputs are absent."""
    missing = []
    if features.empty:
        missing.append("actual feature rows")
    if recommendations.empty:
        missing.append("operational recommendation history")
    if rankings.empty:
        missing.append("operational ranking history")
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "available": False,
        "reason": f"missing {', '.join(missing)}",
        "rows": 0,
    }


def normalize_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamp columns used by operational audit joins."""
    output = frame.copy()
    output[TIMESTAMP_COLUMN] = pd.to_datetime(output[TIMESTAMP_COLUMN], utc=True, errors="coerce")
    output = output.dropna(subset=[TIMESTAMP_COLUMN])
    if "forecast_generated_at_utc" in output:
        output["forecast_generated_at_utc"] = pd.to_datetime(
            output["forecast_generated_at_utc"],
            utc=True,
            errors="coerce",
        ).dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    return output


def safe_mean(values: pd.Series | None) -> float | None:
    """Return rounded mean for numeric or boolean series."""
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce")
    value = numeric.mean()
    return round(float(value), 4) if not pd.isna(value) else None


def smape(actual: pd.Series, predicted: pd.Series) -> float | None:
    """Return symmetric mean absolute percentage error."""
    actual_values = pd.to_numeric(actual, errors="coerce")
    predicted_values = pd.to_numeric(predicted, errors="coerce")
    denominator = (actual_values.abs() + predicted_values.abs()) / 2
    values = (predicted_values - actual_values).abs() / denominator.replace(0, np.nan)
    value = values.mean()
    return round(float(value), 4) if not pd.isna(value) else None


def read_csv(path: str | Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV file if present."""
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path, parse_dates=parse_dates)


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    """Write a CSV output with parent directories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON output with parent directories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """Run the settled recommendation outcome audit."""
    parser = argparse.ArgumentParser(description="Build settled recommendation outcome audit.")
    parser.add_argument("--recent-days", type=int, default=None)
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    parser.add_argument("--metrics-output-path", default=str(METRICS_OUTPUT_PATH))
    args = parser.parse_args(argv)
    summary = build_recommendation_outcome_audit(
        output_path=args.output_path,
        metrics_output_path=args.metrics_output_path,
        recent_days=args.recent_days,
    )
    print(json.dumps({"available": summary["available"], "rows": summary["rows"]}, indent=2))


if __name__ == "__main__":
    main()
