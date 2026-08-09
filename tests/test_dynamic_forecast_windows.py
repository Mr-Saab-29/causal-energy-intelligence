from __future__ import annotations

import pandas as pd

from src.models.baseline_price import (
    FINAL_TEST_WINDOW_NAME,
    TIMESTAMP_COLUMN,
    build_walk_forward_windows,
)


def test_build_walk_forward_windows_uses_latest_90_days_for_final_test() -> None:
    frame = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: pd.date_range(
                "2026-01-01T00:00:00Z",
                "2026-08-09T16:00:00Z",
                freq="h",
            )
        }
    )

    windows = build_walk_forward_windows(frame)

    final_window = windows[-1]
    assert final_window.name == FINAL_TEST_WINDOW_NAME
    assert final_window.test_start == "2026-05-11T17:00:00+00:00"
    assert final_window.test_end == "2026-08-09T16:00:00+00:00"
    assert all(window.name != "test_2026_q2" for window in windows)
