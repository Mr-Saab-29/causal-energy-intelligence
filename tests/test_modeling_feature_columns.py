from __future__ import annotations

from src.features.price_features import MODELING_FEATURE_COLUMNS


def test_modeling_feature_columns_are_unique() -> None:
    assert len(MODELING_FEATURE_COLUMNS) == len(set(MODELING_FEATURE_COLUMNS))
