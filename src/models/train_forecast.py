"""Forecast model training entrypoint."""

from __future__ import annotations


def train_forecast_model(features: list[dict[str, object]]) -> dict[str, object]:
    """Train a price/carbon forecast model and return model metadata."""
    return {"status": "not_trained", "rows": len(features)}

