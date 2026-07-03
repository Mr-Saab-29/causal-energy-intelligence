"""Forecast prediction utilities."""

from __future__ import annotations


def predict_energy_signals(features: list[dict[str, object]]) -> list[dict[str, object]]:
    """Predict future electricity price and carbon intensity signals."""
    return [{"price_forecast": None, "carbon_forecast": None, **row} for row in features]

