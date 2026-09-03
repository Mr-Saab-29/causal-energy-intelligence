"""Causal-adjusted recommendations using a marginal-emissions proxy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.optimization.workload_shift import (
    CONFIDENCE_CALIBRATION_PATH,
    TIMESTAMP_COLUMN,
    WorkloadConstraints,
    add_recommendation_confidence,
    apply_confidence_calibration,
    build_top_workload_recommendations,
    load_confidence_calibration,
    normalized_rank,
    rank_within_group,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORICAL_RANKINGS_PATH = ROOT / "reports/rankings/workload_decision_rankings.csv"
DEFAULT_FUTURE_RANKINGS_PATH = ROOT / "reports/rankings/future_workload_decision_rankings.csv"
DEFAULT_HISTORICAL_RANKING_OUTPUT_PATH = (
    ROOT / "reports/rankings/marginal_workload_decision_rankings.csv"
)
DEFAULT_FUTURE_RANKING_OUTPUT_PATH = (
    ROOT / "reports/rankings/future_marginal_workload_decision_rankings.csv"
)
DEFAULT_HISTORICAL_RECOMMENDATION_OUTPUT_PATH = (
    ROOT / "reports/recommendations/causal_adjusted_workload_recommendations.csv"
)
DEFAULT_FUTURE_RECOMMENDATION_OUTPUT_PATH = (
    ROOT / "reports/recommendations/future_causal_adjusted_workload_recommendations.csv"
)
DEFAULT_METRICS_OUTPUT_PATH = ROOT / "reports/metrics/marginal_ranking_shift_metrics.json"
MIN_CAUSAL_COVERAGE = 0.80


def build_marginal_workload_rankings(
    rankings: pd.DataFrame,
    constraints: WorkloadConstraints | None = None,
) -> pd.DataFrame:
    """Re-rank workload candidates using marginal-carbon proxy ranks."""
    constraints = constraints or WorkloadConstraints()
    output = rankings.copy()
    output[TIMESTAMP_COLUMN] = pd.to_datetime(output[TIMESTAMP_COLUMN], utc=True)
    group_columns = ["window", "model", "decision_group"]
    output = add_marginal_proxy_columns(output, group_columns)

    output["average_predicted_decision_rank"] = output["predicted_decision_rank"]
    output["average_actual_decision_rank"] = output["actual_decision_rank"]
    output["average_predicted_carbon_rank"] = output["predicted_carbon_rank"]
    output["average_actual_carbon_rank"] = output["actual_carbon_rank"]
    output["average_predicted_combined_score"] = output["predicted_combined_score"]
    output["average_actual_combined_score"] = output["actual_combined_score"]

    output["predicted_carbon_rank"] = rank_within_group(
        output,
        group_columns,
        "predicted_marginal_carbon_intensity_g_co2e_per_kwh",
    )
    output["actual_carbon_rank"] = rank_within_group(
        output,
        group_columns,
        "actual_marginal_carbon_intensity_g_co2e_per_kwh",
    )
    output["predicted_carbon_rank_pct"] = normalized_rank(
        output["predicted_carbon_rank"],
        output["candidate_count"],
    )
    output["actual_carbon_rank_pct"] = normalized_rank(
        output["actual_carbon_rank"],
        output["candidate_count"],
    )

    weight_sum = constraints.price_weight + constraints.carbon_weight
    price_weight = constraints.price_weight / weight_sum
    carbon_weight = constraints.carbon_weight / weight_sum
    output["base_predicted_combined_score"] = (
        price_weight * output["predicted_price_rank_pct"]
        + carbon_weight * output["predicted_carbon_rank_pct"]
    )
    output["predicted_combined_score"] = (
        output["base_predicted_combined_score"]
        + output.get("uncertainty_guard_penalty", 0.0)
    )
    output["actual_combined_score"] = (
        price_weight * output["actual_price_rank_pct"]
        + carbon_weight * output["actual_carbon_rank_pct"]
    )
    output["predicted_decision_rank"] = rank_within_group(
        output,
        group_columns,
        "predicted_combined_score",
    )
    output["actual_decision_rank"] = rank_within_group(
        output,
        group_columns,
        "actual_combined_score",
    )
    output["carbon_ranking_strategy"] = "marginal_proxy"
    output["causal_adjusted_rank_shift"] = (
        output["predicted_decision_rank"] - output["average_predicted_decision_rank"]
    )
    return output.sort_values(
        group_columns + ["predicted_decision_rank", TIMESTAMP_COLUMN]
    ).reset_index(drop=True)


def add_marginal_proxy_columns(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Add marginal intensity proxy columns from total emissions and average intensity."""
    output = frame.sort_values(group_columns + [TIMESTAMP_COLUMN]).copy()
    for basis in ("predicted", "actual"):
        emissions_column = f"{basis}_total_emissions_kg_co2e"
        intensity_column = f"{basis}_avg_carbon_intensity_g_co2e_per_kwh"
        generation_column = f"{basis}_implied_generation_mwh"
        marginal_column = f"{basis}_marginal_carbon_intensity_g_co2e_per_kwh"
        if emissions_column not in output or intensity_column not in output:
            output[marginal_column] = np.nan
            continue
        output[generation_column] = (
            output[emissions_column] / output[intensity_column].replace(0, np.nan)
        )
        generation_delta = output.groupby(group_columns, observed=True)[generation_column].diff()
        emissions_delta = output.groupby(group_columns, observed=True)[emissions_column].diff()
        marginal = emissions_delta.where(emissions_delta > 0) / generation_delta.where(
            generation_delta > 0
        )
        output[f"{basis}_marginal_proxy_available"] = marginal.notna()
        output[marginal_column] = marginal.fillna(output[intensity_column])
    output["causal_adjustment_available"] = output[
        "predicted_marginal_proxy_available"
    ].fillna(False)
    output["causal_carbon_source"] = np.where(
        output["causal_adjustment_available"],
        "marginal_emissions_proxy",
        "average_carbon_fallback",
    )
    output["predicted_marginal_proxy_confidence"] = np.where(
        output["causal_adjustment_available"],
        "medium",
        "unavailable",
    )
    return output


def build_causal_adjusted_recommendations(rankings: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Export top-N causal-adjusted recommendations with marginal proxy context."""
    recommendations = build_top_workload_recommendations(rankings, top_n=top_n)
    recommendations = add_recommendation_confidence(recommendations, rankings, top_n)
    recommendations = apply_confidence_calibration(
        recommendations,
        load_confidence_calibration(CONFIDENCE_CALIBRATION_PATH),
    )
    context_columns = [
        "carbon_ranking_strategy",
        "causal_carbon_source",
        "causal_adjustment_available",
        "average_predicted_decision_rank",
        "average_actual_decision_rank",
        "average_predicted_carbon_rank",
        "average_actual_carbon_rank",
        "causal_adjusted_rank_shift",
        "predicted_marginal_carbon_intensity_g_co2e_per_kwh",
        "actual_marginal_carbon_intensity_g_co2e_per_kwh",
        "predicted_marginal_proxy_confidence",
    ]
    context = rankings[
        ["window", "model", "decision_group", TIMESTAMP_COLUMN]
        + [column for column in context_columns if column in rankings]
    ]
    output = recommendations.merge(
        context,
        on=["window", "model", "decision_group", TIMESTAMP_COLUMN],
        how="left",
    )
    float_columns = output.select_dtypes(include=["float"]).columns
    output[float_columns] = output[float_columns].round(2)
    return output


def summarize_ranking_shifts(
    average_rankings: pd.DataFrame,
    marginal_rankings: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, Any]:
    """Summarize average-carbon vs marginal-proxy ranking shifts."""
    join_columns = ["window", "model", "decision_group", TIMESTAMP_COLUMN]
    comparison = average_rankings[
        join_columns + ["predicted_decision_rank", "combined_regret"]
    ].rename(
        columns={
            "predicted_decision_rank": "average_rank",
            "combined_regret": "average_combined_regret",
        }
    ).merge(
        marginal_rankings[
            join_columns
            + ["predicted_decision_rank", "combined_regret", "causal_adjustment_available"]
        ].rename(
            columns={
                "predicted_decision_rank": "marginal_rank",
                "combined_regret": "marginal_combined_regret",
            }
        ),
        on=join_columns,
        how="inner",
    )
    comparison["absolute_rank_shift"] = (
        comparison["marginal_rank"] - comparison["average_rank"]
    ).abs()
    summary: list[dict[str, Any]] = []
    for group_key, group in comparison.groupby(["window", "model", "decision_group"], observed=True):
        average_top = set(group.loc[group["average_rank"] <= top_n, TIMESTAMP_COLUMN])
        marginal_top = set(group.loc[group["marginal_rank"] <= top_n, TIMESTAMP_COLUMN])
        average_best = group.loc[group["average_rank"].idxmin()]
        marginal_best = group.loc[group["marginal_rank"].idxmin()]
        overlap = len(average_top & marginal_top)
        summary.append(
            {
                "window": group_key[0],
                "model": group_key[1],
                "decision_group": group_key[2],
                "candidate_count": int(len(group)),
                "causal_adjustment_coverage": float(group["causal_adjustment_available"].mean()),
                "top_1_changed": bool(average_best[TIMESTAMP_COLUMN] != marginal_best[TIMESTAMP_COLUMN]),
                "top_5_overlap_count": int(overlap),
                "top_5_overlap_share": float(overlap / max(len(average_top | marginal_top), 1)),
                "mean_absolute_rank_shift": float(group["absolute_rank_shift"].mean()),
                "max_absolute_rank_shift": int(group["absolute_rank_shift"].max()),
                "top_1_average_timestamp_utc": average_best[TIMESTAMP_COLUMN].isoformat(),
                "top_1_marginal_timestamp_utc": marginal_best[TIMESTAMP_COLUMN].isoformat(),
                "top_1_regret_delta": safe_float(
                    marginal_best["marginal_combined_regret"]
                    - average_best["average_combined_regret"]
                ),
            }
        )
    aggregate = summarize_shift_aggregate(pd.DataFrame(summary))
    warnings = []
    if aggregate and aggregate["mean_causal_adjustment_coverage"] < MIN_CAUSAL_COVERAGE:
        warnings.append("low_marginal_proxy_coverage")
    return {
        "method": "marginal_proxy_mvp",
        "top_n": top_n,
        "quality_guard": {
            "min_causal_adjustment_coverage": MIN_CAUSAL_COVERAGE,
            "status": "warning" if warnings else "ok",
            "warnings": warnings,
        },
        "aggregate": sanitize_json_value(aggregate),
        "summary": sanitize_json_value(summary),
    }


def summarize_shift_aggregate(summary: pd.DataFrame) -> dict[str, Any]:
    """Return aggregate shift metrics across groups."""
    if summary.empty:
        return {}
    return {
        "groups": int(len(summary)),
        "top_1_change_share": float(summary["top_1_changed"].mean()),
        "mean_top_5_overlap_share": float(summary["top_5_overlap_share"].mean()),
        "mean_absolute_rank_shift": float(summary["mean_absolute_rank_shift"].mean()),
        "mean_causal_adjustment_coverage": float(summary["causal_adjustment_coverage"].mean()),
        "mean_top_1_regret_delta": safe_float(summary["top_1_regret_delta"].mean()),
    }


def run_causal_adjusted_recommendations(
    rankings_path: str | Path,
    ranking_output_path: str | Path,
    recommendation_output_path: str | Path,
    metrics_output_path: str | Path | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Build marginal rankings and causal-adjusted recommendations for one ranking file."""
    average_rankings = pd.read_csv(rankings_path, parse_dates=[TIMESTAMP_COLUMN])
    marginal_rankings = build_marginal_workload_rankings(average_rankings)
    recommendations = build_causal_adjusted_recommendations(marginal_rankings, top_n=top_n)
    metrics = summarize_ranking_shifts(average_rankings, marginal_rankings, top_n=top_n)
    metrics["generated_from"] = {"average_rankings": str(rankings_path)}
    metrics["outputs"] = {
        "marginal_rankings": str(ranking_output_path),
        "causal_adjusted_recommendations": str(recommendation_output_path),
    }
    write_csv(ranking_output_path, marginal_rankings)
    write_csv(recommendation_output_path, recommendations)
    if metrics_output_path:
        write_json(metrics_output_path, metrics)
    return metrics


def run_all_causal_adjusted_recommendations(top_n: int = 5) -> dict[str, Any]:
    """Build historical and future causal-adjusted recommendation artifacts."""
    historical: dict[str, Any]
    if DEFAULT_HISTORICAL_RANKINGS_PATH.exists():
        historical = run_causal_adjusted_recommendations(
            DEFAULT_HISTORICAL_RANKINGS_PATH,
            DEFAULT_HISTORICAL_RANKING_OUTPUT_PATH,
            DEFAULT_HISTORICAL_RECOMMENDATION_OUTPUT_PATH,
            top_n=top_n,
        )
    else:
        historical = {
            "status": "skipped",
            "reason": "historical_rankings_missing",
            "generated_from": {"average_rankings": str(DEFAULT_HISTORICAL_RANKINGS_PATH)},
        }
    future = run_causal_adjusted_recommendations(
        DEFAULT_FUTURE_RANKINGS_PATH,
        DEFAULT_FUTURE_RANKING_OUTPUT_PATH,
        DEFAULT_FUTURE_RECOMMENDATION_OUTPUT_PATH,
        top_n=top_n,
    )
    metrics = {
        "historical": historical,
        "future": future,
        "quality_guard": future["quality_guard"],
    }
    write_json(DEFAULT_METRICS_OUTPUT_PATH, metrics)
    return metrics


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    """Write CSV with parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write strict JSON with parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(sanitize_json_value(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def safe_float(value: Any) -> float | None:
    """Return a JSON-safe float."""
    if pd.isna(value):
        return None
    return float(value)


def sanitize_json_value(value: Any) -> Any:
    """Return a value that can be emitted as strict JSON."""
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return sanitize_json_value(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def main(argv: list[str] | None = None) -> None:
    """Run causal-adjusted recommendation generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--future-only", action="store_true")
    args = parser.parse_args(argv)
    if args.future_only:
        result = run_causal_adjusted_recommendations(
            DEFAULT_FUTURE_RANKINGS_PATH,
            DEFAULT_FUTURE_RANKING_OUTPUT_PATH,
            DEFAULT_FUTURE_RECOMMENDATION_OUTPUT_PATH,
            DEFAULT_METRICS_OUTPUT_PATH,
            top_n=args.top_n,
        )
    else:
        result = run_all_causal_adjusted_recommendations(top_n=args.top_n)
    print(json.dumps(sanitize_json_value(result.get("future", result)["aggregate"]), indent=2))


if __name__ == "__main__":
    main()
