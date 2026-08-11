from __future__ import annotations

import pandas as pd

from src.data.pipeline_health import check_future_timestamp_file


def test_check_future_timestamp_file_requires_future_coverage(tmp_path) -> None:
    path = tmp_path / "future_weather.csv"
    path.write_text(
        "timestamp_utc,region,temperature_c\n"
        "2026-08-10T02:00:00+00:00,FR,18.0\n",
        encoding="utf-8",
    )
    result = {
        "critical_issues": [],
        "warnings": [],
    }

    check_future_timestamp_file(
        result=result,
        path=path,
        prefix="weather",
        now=pd.Timestamp("2026-08-10T08:00:00Z"),
        strict_freshness=True,
    )

    assert "weather_has_no_future_timestamps" in result["critical_issues"]
