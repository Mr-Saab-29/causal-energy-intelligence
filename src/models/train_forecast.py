"""Forecast model training entrypoint."""

from __future__ import annotations

from src.models.baseline_price import run_price_baselines


def train_forecast_model(features: list[dict[str, object]]) -> dict[str, object]:
    """Train a price/carbon forecast model and return model metadata."""
    return {"status": "not_trained", "rows": len(features)}


def train_price_baselines() -> dict[str, object]:
    """Train baseline electricity price forecasting models."""
    return run_price_baselines()
