"""Forecast model training entrypoint."""

from __future__ import annotations

import argparse
from typing import Any

from src.carbon import run_carbon_accounting
from src.models.baseline_price import (
    PRODUCTION_SIGNAL_TARGETS,
    run_price_baselines,
    run_price_ranking_from_predictions,
    run_supply_demand_baselines,
)
from src.optimization.workload_shift import WorkloadConstraints, run_workload_decision_ranking

PRODUCTION_BASELINE_PREDICTIONS_PATH = "reports/predictions/production_baseline_predictions.csv"
PRODUCTION_SOURCES_BASELINE_PREDICTIONS_PATH = (
    "reports/predictions/production_sources_baseline_predictions.csv"
)


def train_forecast_model(features: list[dict[str, object]]) -> dict[str, object]:
    """Train a price/carbon forecast model and return model metadata."""
    return {"status": "not_trained", "rows": len(features)}


def train_price_baselines() -> dict[str, object]:
    """Train baseline electricity price forecasting models."""
    return run_price_baselines()


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run forecast training from the command line."""
    parser = argparse.ArgumentParser(description="Train forecasting baselines.")
    parser.add_argument(
        "--target",
        choices=[
            "all",
            "price",
            "ranking",
            "decision",
            "consumption",
            "production",
            "supply-demand",
            "carbon",
        ],
        default="all",
        help=(
            "Training block to run. 'production' trains total production plus source-level "
            "generation. 'carbon' computes emissions and carbon intensity from source forecasts."
        ),
    )
    parser.add_argument("--price-weight", type=float, default=0.5)
    parser.add_argument("--carbon-weight", type=float, default=0.5)
    parser.add_argument("--duration-hours", type=int, default=1)
    parser.add_argument("--earliest-start-utc", default=None)
    parser.add_argument("--latest-end-utc", default=None)
    parser.add_argument("--max-delay-hours", type=int, default=None)
    parser.add_argument("--methodology", default="direct_operational_emissions")
    parser.add_argument("--top-n-recommendations", type=int, default=5)
    args = parser.parse_args(argv)

    if args.target == "all":
        result = run_price_baselines()
        run_supply_demand_baselines(
            target_names=PRODUCTION_SIGNAL_TARGETS[1:],
            metrics_path="reports/metrics/production_sources_baseline_metrics.json",
            predictions_path=PRODUCTION_SOURCES_BASELINE_PREDICTIONS_PATH,
            feature_importance_path=(
                "reports/metrics/production_sources_baseline_feature_importance.csv"
            ),
        )
        run_carbon_accounting(predictions_path=PRODUCTION_SOURCES_BASELINE_PREDICTIONS_PATH)
        result = run_workload_decision_ranking()
    elif args.target == "price":
        result = run_price_baselines()
    elif args.target == "ranking":
        result = run_price_ranking_from_predictions()
    elif args.target == "decision":
        result = run_workload_decision_ranking(
            constraints=WorkloadConstraints(
                earliest_start_utc=args.earliest_start_utc,
                latest_end_utc=args.latest_end_utc,
                duration_hours=args.duration_hours,
                max_delay_hours=args.max_delay_hours,
                price_weight=args.price_weight,
                carbon_weight=args.carbon_weight,
                methodology=args.methodology,
            ),
            top_n_recommendations=args.top_n_recommendations,
        )
    elif args.target == "supply-demand":
        result = run_supply_demand_baselines()
    elif args.target == "production":
        result = run_supply_demand_baselines(
            target_names=PRODUCTION_SIGNAL_TARGETS,
            metrics_path="reports/metrics/production_baseline_metrics.json",
            predictions_path=PRODUCTION_BASELINE_PREDICTIONS_PATH,
            feature_importance_path="reports/metrics/production_baseline_feature_importance.csv",
        )
        run_carbon_accounting(predictions_path=PRODUCTION_BASELINE_PREDICTIONS_PATH)
    elif args.target == "carbon":
        result = run_carbon_accounting()
    else:
        result = run_supply_demand_baselines(
            target_names=[args.target],
            metrics_path=f"reports/metrics/{args.target}_baseline_metrics.json",
            predictions_path=f"reports/predictions/{args.target}_baseline_predictions.csv",
            feature_importance_path=f"reports/metrics/{args.target}_baseline_feature_importance.csv",
        )

    print(result["summary"])
    return result


if __name__ == "__main__":
    main()
