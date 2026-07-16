"""Forecast model training entrypoint."""

from __future__ import annotations

import argparse
from typing import Any

from src.models.baseline_price import (
    PRODUCTION_SIGNAL_TARGETS,
    run_price_baselines,
    run_price_ranking_from_predictions,
    run_supply_demand_baselines,
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
        choices=["all", "price", "ranking", "consumption", "production", "supply-demand"],
        default="all",
        help=(
            "Training block to run. 'production' trains total production plus source-level "
            "generation. 'price' trains upstream consumption/total-production forecasts before price."
        ),
    )
    args = parser.parse_args(argv)

    if args.target == "all":
        result = run_price_baselines()
        run_supply_demand_baselines(
            target_names=PRODUCTION_SIGNAL_TARGETS[1:],
            metrics_path="reports/metrics/production_sources_baseline_metrics.json",
            predictions_path="reports/predictions/production_sources_baseline_predictions.csv",
            feature_importance_path=(
                "reports/metrics/production_sources_baseline_feature_importance.csv"
            ),
        )
    elif args.target == "price":
        result = run_price_baselines()
    elif args.target == "ranking":
        result = run_price_ranking_from_predictions()
    elif args.target == "supply-demand":
        result = run_supply_demand_baselines()
    elif args.target == "production":
        result = run_supply_demand_baselines(
            target_names=PRODUCTION_SIGNAL_TARGETS,
            metrics_path="reports/metrics/production_baseline_metrics.json",
            predictions_path="reports/predictions/production_baseline_predictions.csv",
            feature_importance_path="reports/metrics/production_baseline_feature_importance.csv",
        )
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
