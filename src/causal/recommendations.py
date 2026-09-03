"""Causal-adjusted recommendation artifacts from marginal-emissions proxy rankings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.carbon.marginal import run_marginal_emissions_proxy
from src.optimization.workload_shift import (
    TIMESTAMP_COLUMN,
    WorkloadConstraints,
    add_recommendation_confidence,
    annotate_price_direction,
    annotate_regret_and_savings,
    apply_confidence_calibration,
    build_confidence_calibration,
    build_top_workload_recommendations,
    normalized_rank,
    rank_within_group,
)

DEFAULT_AVERAGE_RANKINGS_PATH = "reports/rankings/workload_decision_rankings.csv"
DEFAULT_FUTURE_AVERAGE_RANKINGS_PATH = "reports/rankings/future_workload_decision_rankings.csv"
DEFAULT_MARGINAL_PROXY_PATH = "reports/carbon/marginal_emissions_proxy.csv"
DEFAULT_RANKING_OUTPUT_PATH = "reports/rankings/marginal_workload_decision_rankings.csv"
DEFAULT_FUTURE_RANKING_OUTPUT_PATH = "reports/rankings/future_marginal_workload_decision_rankings.csv"
DEFAULT_RECOMMENDATION_OUTPUT_PATH = (
    "reports/recommendations/causal_adjusted_workload_recommendations.csv"
)
DEFAULT_FUTURE_RECOMMENDATION_OUTPUT_PATH = (
    "reports/recommendations/future_causal_adjusted_workload_recommendations.csv"
)
DEFAULT_METRICS_OUTPUT_PATH = "reports/metrics/marginal_ranking_shift_metrics.json"
MIN_CAUSAL_COVERAGE = 0.80


def build_marginal_workload_rankings(
    average_rankings: pd.DataFrame,
    marginal_proxy: pd.DataFrame | None = None,
    constraints: WorkloadConstraints | None = None,
    methodology: str = "direct_operational_emissions",
) -> pd.DataFrame:
    """Re-rank workload candidates with marginal-carbon intensity instead of average carbon."""
    constraints = constraints or WorkloadConstraints(methodology=methodology)
    rankings = average_rankings.copy()
    rankings[TIMESTAMP_COLUMN] = pd.to_datetime(rankings[TIMESTAMP_COLUMN], utc=True)
    rankings["average_predicted_decision_rank"] = rankings["predicted_decision_rank"]
    rankings["average_actual_decision_rank"] = rankings["actual_decision_rank"]
    rankings["average_predicted_combined_score"] = rankings["predicted_combined_score"]
    rankings["average_actual_combined_score"] = rankings["actual_combined_score"]
    rankings["average_predicted_carbon_rank"] = rankings["predicted_carbon_rank"]
    rankings["average_actual_carbon_rank"] = rankings["actual_carbon_rank"]

    group_columns = ["window", "model", "decision_group"]
    if marginal_proxy is None or marginal_proxy.empty:
        rankings = add_marginal_proxy_columns_from_rankings(rankings, group_columns)
    else:
        proxy_values = prepare_marginal_proxy_values(marginal_proxy, methodology)
        rankings = rankings.merge(
            proxy_values,
            on=[TIMESTAMP_COLUMN, "window", "model"],
            how="left",
        )
        rankings["causal_carbon_source"] = np.where(
            rankings["predicted_marginal_carbon_intensity_g_co2e_per_kwh"].notna(),
            "marginal_emissions_proxy",
            "average_carbon_fallback",
        )
        rankings["causal_adjustment_available"] = rankings[
            "predicted_marginal_carbon_intensity_g_co2e_per_kwh"
        ].notna()
        rankings["predicted_marginal_carbon_intensity_g_co2e_per_kwh"] = rankings[
            "predicted_marginal_carbon_intensity_g_co2e_per_kwh"
        ].fillna(rankings["predicted_avg_carbon_intensity_g_co2e_per_kwh"])
        rankings["actual_marginal_carbon_intensity_g_co2e_per_kwh"] = rankings[
            "actual_marginal_carbon_intensity_g_co2e_per_kwh"
        ].fillna(rankings["actual_avg_carbon_intensity_g_co2e_per_kwh"])
        rankings["predicted_marginal_proxy_confidence"] = rankings[
            "predicted_marginal_proxy_confidence"
        ].fillna("unavailable")
        rankings["actual_marginal_proxy_confidence"] = rankings[
            "actual_marginal_proxy_confidence"
        ].fillna("unavailable")

    rankings["predicted_carbon_rank"] = rank_within_group(
        rankings,
        group_columns,
        "predicted_marginal_carbon_intensity_g_co2e_per_kwh",
    )
    rankings["actual_carbon_rank"] = rank_within_group(
        rankings,
        group_columns,
        "actual_marginal_carbon_intensity_g_co2e_per_kwh",
    )
    rankings["predicted_carbon_rank_pct"] = normalized_rank(
        rankings["predicted_carbon_rank"],
        rankings["candidate_count"],
    )
    rankings["actual_carbon_rank_pct"] = normalized_rank(
        rankings["actual_carbon_rank"],
        rankings["candidate_count"],
    )

    weight_sum = constraints.price_weight + constraints.carbon_weight
    price_weight = constraints.price_weight / weight_sum
    carbon_weight = constraints.carbon_weight / weight_sum
    rankings["base_predicted_combined_score"] = (
        price_weight * rankings["predicted_price_rank_pct"]
        + carbon_weight * rankings["predicted_carbon_rank_pct"]
    )
    rankings["predicted_combined_score"] = (
        rankings["base_predicted_combined_score"]
        + rankings.get("uncertainty_guard_penalty", 0.0)
    )
    rankings["actual_combined_score"] = (
        price_weight * rankings["actual_price_rank_pct"]
        + carbon_weight * rankings["actual_carbon_rank_pct"]
    )
    rankings["predicted_decision_rank"] = rank_within_group(
        rankings,
        group_columns,
        "predicted_combined_score",
    )
    rankings["actual_decision_rank"] = rank_within_group(
        rankings,
        group_columns,
        "actual_combined_score",
    )

    annotate_regret_and_savings(rankings, group_columns)
    annotate_price_direction(rankings)
    rankings["carbon_ranking_strategy"] = "marginal_proxy"
    rankings["causal_adjusted_rank_shift"] = (
        rankings["predicted_decision_rank"] - rankings["average_predicted_decision_rank"]
    )
    return rankings.sort_values(
        group_columns + ["predicted_decision_rank", TIMESTAMP_COLUMN]
    ).reset_index(drop=True)


def add_marginal_proxy_columns_from_rankings(
    rankings: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Add marginal proxy columns from ranking total emissions and average intensity."""
    output = rankings.sort_values(group_columns + [TIMESTAMP_COLUMN]).copy()
    for basis in ("predicted", "actual"):
        emissions_column = f"{basis}_total_emissions_kg_co2e"
        intensity_column = f"{basis}_avg_carbon_intensity_g_co2e_per_kwh"
        generation_column = f"{basis}_implied_generation_mwh"
        available_column = f"{basis}_marginal_proxy_available"
        marginal_column = f"{basis}_marginal_carbon_intensity_g_co2e_per_kwh"
        confidence_column = f"{basis}_marginal_proxy_confidence"
        if emissions_column not in output or intensity_column not in output:
            output[available_column] = False
            output[marginal_column] = np.nan
            output[confidence_column] = "unavailable"
            continue
        output[generation_column] = (
            output[emissions_column] / output[intensity_column].replace(0, np.nan)
        )
        generation_delta = output.groupby(group_columns, observed=True)[generation_column].diff()
        emissions_delta = output.groupby(group_columns, observed=True)[emissions_column].diff()
        marginal = emissions_delta.where(emissions_delta > 0) / generation_delta.where(
            generation_delta > 0
        )
        output[available_column] = marginal.notna()
        output[marginal_column] = marginal.fillna(output[intensity_column])
        output[confidence_column] = np.where(output[available_column], "medium", "unavailable")

    output["causal_adjustment_available"] = output[
        "predicted_marginal_proxy_available"
    ].fillna(False)
    output["causal_carbon_source"] = np.where(
        output["causal_adjustment_available"],
        "marginal_emissions_proxy",
        "average_carbon_fallback",
    )
    return output


def prepare_marginal_proxy_values(
    marginal_proxy: pd.DataFrame,
    methodology: str,
) -> pd.DataFrame:
    """Return one predicted/actual marginal row per timestamp/window/model."""
    frame = marginal_proxy.copy()
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    frame = frame[frame["methodology"] == methodology]
    predicted = frame[frame["basis"] == "predicted"][
        [
            TIMESTAMP_COLUMN,
            "window",
            "model",
            "marginal_carbon_intensity_g_co2e_per_kwh",
            "marginal_source",
            "marginal_proxy_confidence",
        ]
    ].rename(
        columns={
            "marginal_carbon_intensity_g_co2e_per_kwh": (
                "predicted_marginal_carbon_intensity_g_co2e_per_kwh"
            ),
            "marginal_source": "predicted_marginal_source",
            "marginal_proxy_confidence": "predicted_marginal_proxy_confidence",
        }
    )
    actual = frame[frame["basis"] == "actual"][
        [
            TIMESTAMP_COLUMN,
            "window",
            "model",
            "marginal_carbon_intensity_g_co2e_per_kwh",
            "marginal_source",
            "marginal_proxy_confidence",
        ]
    ].rename(
        columns={
            "marginal_carbon_intensity_g_co2e_per_kwh": (
                "actual_marginal_carbon_intensity_g_co2e_per_kwh"
            ),
            "marginal_source": "actual_marginal_source",
            "marginal_proxy_confidence": "actual_marginal_proxy_confidence",
        }
    )
    return predicted.merge(actual, on=[TIMESTAMP_COLUMN, "window", "model"], how="outer")


def summarize_ranking_shifts(
    average_rankings: pd.DataFrame,
    marginal_rankings: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, Any]:
    """Quantify how much marginal-carbon ranking changes average-carbon recommendations."""
    join_columns = ["window", "model", "decision_group", TIMESTAMP_COLUMN]
    average = average_rankings[join_columns + ["predicted_decision_rank", "combined_regret"]].rename(
        columns={
            "predicted_decision_rank": "average_rank",
            "combined_regret": "average_combined_regret",
        }
    )
    marginal = marginal_rankings[
        join_columns
        + [
            "predicted_decision_rank",
            "combined_regret",
            "causal_adjustment_available",
            "predicted_marginal_carbon_intensity_g_co2e_per_kwh",
            "predicted_avg_carbon_intensity_g_co2e_per_kwh",
        ]
    ].rename(
        columns={
            "predicted_decision_rank": "marginal_rank",
            "combined_regret": "marginal_combined_regret",
        }
    )
    comparison = average.merge(marginal, on=join_columns, how="inner")
    comparison["absolute_rank_shift"] = (
        comparison["marginal_rank"] - comparison["average_rank"]
    ).abs()
    comparison["marginal_minus_average_regret"] = (
        comparison["marginal_combined_regret"] - comparison["average_combined_regret"]
    )

    summary: list[dict[str, Any]] = []
    top_shift_rows: list[dict[str, Any]] = []
    for group_key, group in comparison.groupby(["window", "model", "decision_group"], observed=True):
        average_top = set(group.loc[group["average_rank"] <= top_n, TIMESTAMP_COLUMN])
        marginal_top = set(group.loc[group["marginal_rank"] <= top_n, TIMESTAMP_COLUMN])
        overlap = len(average_top & marginal_top)
        average_best = group.loc[group["average_rank"].idxmin()]
        marginal_best = group.loc[group["marginal_rank"].idxmin()]
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
        shifted = group[group["absolute_rank_shift"] > 0].sort_values(
            "absolute_rank_shift",
            ascending=False,
        )
        top_shift_rows.extend(
            shifted.head(3)[
                join_columns + ["average_rank", "marginal_rank", "absolute_rank_shift"]
            ].to_dict(orient="records")
        )

    aggregate = pd.DataFrame(summary)
    aggregate_summary = summarize_shift_aggregate(aggregate)
    warnings = []
    if (
        aggregate_summary
        and aggregate_summary["mean_causal_adjustment_coverage"] < MIN_CAUSAL_COVERAGE
    ):
        warnings.append("low_marginal_proxy_coverage")
    return {
        "method": "marginal_proxy_mvp",
        "top_n": top_n,
        "quality_guard": {
            "min_causal_adjustment_coverage": MIN_CAUSAL_COVERAGE,
            "status": "warning" if warnings else "ok",
            "warnings": warnings,
        },
        "summary": sanitize_json_value(summary),
        "aggregate": sanitize_json_value(aggregate_summary),
        "largest_rank_shifts": sanitize_json_value(top_shift_rows),
    }


def summarize_shift_aggregate(summary: pd.DataFrame) -> dict[str, Any]:
    """Return compact aggregate shift metrics across all decision groups."""
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


def build_causal_adjusted_recommendations(
    marginal_rankings: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Build top-N causal-adjusted recommendations with existing confidence fields."""
    recommendations = build_top_workload_recommendations(marginal_rankings, top_n=top_n)
    recommendations = add_recommendation_confidence(recommendations, marginal_rankings, top_n)
    recommendations = apply_confidence_calibration(
        recommendations,
        build_confidence_calibration(recommendations, top_n=top_n),
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
        "predicted_marginal_source",
        "actual_marginal_source",
        "predicted_marginal_proxy_confidence",
        "actual_marginal_proxy_confidence",
    ]
    context = marginal_rankings[
        ["window", "model", "decision_group", TIMESTAMP_COLUMN]
        + [column for column in context_columns if column in marginal_rankings]
    ]
    output = recommendations.merge(
        context,
        on=["window", "model", "decision_group", TIMESTAMP_COLUMN],
        how="left",
    )
    float_columns = output.select_dtypes(include=["float"]).columns
    output[float_columns] = output[float_columns].round(2)
    return output


def run_causal_adjusted_recommendations(
    average_rankings_path: str | Path = DEFAULT_AVERAGE_RANKINGS_PATH,
    marginal_proxy_path: str | Path = DEFAULT_MARGINAL_PROXY_PATH,
    ranking_output_path: str | Path = DEFAULT_RANKING_OUTPUT_PATH,
    recommendation_output_path: str | Path = DEFAULT_RECOMMENDATION_OUTPUT_PATH,
    metrics_output_path: str | Path = DEFAULT_METRICS_OUTPUT_PATH,
    methodology: str = "direct_operational_emissions",
    top_n: int = 5,
    ensure_marginal_proxy: bool = True,
    use_marginal_proxy_file: bool = True,
) -> dict[str, Any]:
    """Build marginal rankings, shift metrics, and causal-adjusted recommendations."""
    if use_marginal_proxy_file and ensure_marginal_proxy and not Path(marginal_proxy_path).exists():
        run_marginal_emissions_proxy(output_path=marginal_proxy_path)

    average_rankings = pd.read_csv(average_rankings_path, parse_dates=[TIMESTAMP_COLUMN])
    marginal_proxy = (
        pd.read_csv(marginal_proxy_path, parse_dates=[TIMESTAMP_COLUMN])
        if use_marginal_proxy_file and Path(marginal_proxy_path).exists()
        else pd.DataFrame()
    )
    constraints = WorkloadConstraints(methodology=methodology)
    marginal_rankings = build_marginal_workload_rankings(
        average_rankings,
        marginal_proxy,
        constraints=constraints,
        methodology=methodology,
    )
    recommendations = build_causal_adjusted_recommendations(marginal_rankings, top_n=top_n)
    metrics = summarize_ranking_shifts(average_rankings, marginal_rankings, top_n=top_n)
    metrics["generated_from"] = {
        "average_rankings": str(average_rankings_path),
        "marginal_proxy": str(marginal_proxy_path) if use_marginal_proxy_file else None,
    }
    metrics["outputs"] = {
        "marginal_rankings": str(ranking_output_path),
        "causal_adjusted_recommendations": str(recommendation_output_path),
    }

    write_csv(ranking_output_path, marginal_rankings)
    write_csv(recommendation_output_path, recommendations)
    if metrics_output_path:
        write_json(metrics_output_path, metrics)
    return metrics


def run_all_causal_adjusted_recommendations(
    top_n: int = 5,
    methodology: str = "direct_operational_emissions",
) -> dict[str, Any]:
    """Build historical and future causal-adjusted recommendation artifacts."""
    historical: dict[str, Any]
    if Path(DEFAULT_AVERAGE_RANKINGS_PATH).exists():
        historical = run_causal_adjusted_recommendations(
            average_rankings_path=DEFAULT_AVERAGE_RANKINGS_PATH,
            marginal_proxy_path=DEFAULT_MARGINAL_PROXY_PATH,
            ranking_output_path=DEFAULT_RANKING_OUTPUT_PATH,
            recommendation_output_path=DEFAULT_RECOMMENDATION_OUTPUT_PATH,
            metrics_output_path=None,
            methodology=methodology,
            top_n=top_n,
            ensure_marginal_proxy=True,
        )
    else:
        historical = {
            "status": "skipped",
            "reason": "historical_rankings_missing",
            "generated_from": {"average_rankings": DEFAULT_AVERAGE_RANKINGS_PATH},
        }

    future = run_causal_adjusted_recommendations(
        average_rankings_path=DEFAULT_FUTURE_AVERAGE_RANKINGS_PATH,
        marginal_proxy_path=DEFAULT_MARGINAL_PROXY_PATH,
        ranking_output_path=DEFAULT_FUTURE_RANKING_OUTPUT_PATH,
        recommendation_output_path=DEFAULT_FUTURE_RECOMMENDATION_OUTPUT_PATH,
        metrics_output_path=None,
        methodology=methodology,
        top_n=top_n,
        ensure_marginal_proxy=False,
        use_marginal_proxy_file=False,
    )
    metrics = {
        "historical": historical,
        "future": future,
        "quality_guard": future["quality_guard"],
    }
    write_json(DEFAULT_METRICS_OUTPUT_PATH, metrics)
    return metrics


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    """Write a CSV file, creating parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write strict JSON, creating parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(sanitize_json_value(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def safe_float(value: Any) -> float | None:
    """Return a JSON-safe float or null."""
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
    """Run causal-adjusted recommendation generation from the command line."""
    parser = argparse.ArgumentParser(
        description="Build average-vs-marginal ranking comparison and causal recommendations."
    )
    parser.add_argument("--average-rankings-path", default=DEFAULT_AVERAGE_RANKINGS_PATH)
    parser.add_argument("--marginal-proxy-path", default=DEFAULT_MARGINAL_PROXY_PATH)
    parser.add_argument("--ranking-output-path", default=DEFAULT_RANKING_OUTPUT_PATH)
    parser.add_argument("--recommendation-output-path", default=DEFAULT_RECOMMENDATION_OUTPUT_PATH)
    parser.add_argument("--metrics-output-path", default=DEFAULT_METRICS_OUTPUT_PATH)
    parser.add_argument("--methodology", default="direct_operational_emissions")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--skip-marginal-proxy", action="store_true")
    parser.add_argument("--single-run", action="store_true")
    args = parser.parse_args(argv)

    custom_single_run = any(
        [
            args.average_rankings_path != DEFAULT_AVERAGE_RANKINGS_PATH,
            args.marginal_proxy_path != DEFAULT_MARGINAL_PROXY_PATH,
            args.ranking_output_path != DEFAULT_RANKING_OUTPUT_PATH,
            args.recommendation_output_path != DEFAULT_RECOMMENDATION_OUTPUT_PATH,
        ]
    )
    if args.single_run or custom_single_run:
        result = run_causal_adjusted_recommendations(
            average_rankings_path=args.average_rankings_path,
            marginal_proxy_path=args.marginal_proxy_path,
            ranking_output_path=args.ranking_output_path,
            recommendation_output_path=args.recommendation_output_path,
            metrics_output_path=args.metrics_output_path,
            methodology=args.methodology,
            top_n=args.top_n,
            ensure_marginal_proxy=not args.skip_marginal_proxy,
        )
        aggregate = result["aggregate"]
        outputs = result["outputs"]
    else:
        result = run_all_causal_adjusted_recommendations(
            top_n=args.top_n,
            methodology=args.methodology,
        )
        aggregate = result["future"]["aggregate"]
        outputs = result["future"]["outputs"]
    print(
        json.dumps(
            {
                "status": "ok",
                "aggregate": aggregate,
                "outputs": outputs,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
