"""Feature engineering for forecasting and causal inference."""

from __future__ import annotations

import os

from src.data.load import create_database_engine
from src.features.price_features import build_and_store_price_modeling_features


def build_features(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Create model-ready features from transformed energy records."""
    return records


def build_price_features_from_environment() -> int:
    """Build and store price modeling features using DATABASE_URL."""
    engine = create_database_engine(os.environ["DATABASE_URL"])
    features = build_and_store_price_modeling_features(engine)
    return len(features)
