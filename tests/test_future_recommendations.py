from __future__ import annotations

import pandas as pd

from src.models.future_recommendations import remove_duplicate_columns


def test_remove_duplicate_columns_preserves_first_occurrence() -> None:
    frame = pd.DataFrame(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        columns=["timestamp_utc", "wind_lag_24h", "wind_lag_24h"],
    )

    result = remove_duplicate_columns(frame)

    assert result.columns.tolist() == ["timestamp_utc", "wind_lag_24h"]
    assert result["wind_lag_24h"].tolist() == [2, 5]
