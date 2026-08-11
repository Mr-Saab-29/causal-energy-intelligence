from __future__ import annotations

import pandas as pd

from scripts.build_dashboard_data import filter_future_recommendations


def test_filter_future_recommendations_drops_past_rows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_utc": [
                "2026-08-10T07:00:00+00:00",
                "2026-08-10T08:00:00+00:00",
                "2026-08-10T09:00:00+00:00",
            ],
            "recommendation_rank": [1, 2, 3],
        }
    )

    filtered = filter_future_recommendations(
        frame,
        now=pd.Timestamp("2026-08-10T08:10:00Z"),
    )

    assert filtered["timestamp_utc"].tolist() == [
        "2026-08-10T08:00:00+00:00",
        "2026-08-10T09:00:00+00:00",
    ]
