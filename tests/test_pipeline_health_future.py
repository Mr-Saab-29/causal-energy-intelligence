from __future__ import annotations

import pandas as pd

from src.data import pipeline_health
from src.data.pipeline_health import (
    SourceHealthConfig,
    build_pipeline_health,
    check_future_timestamp_file,
)


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


def test_cloud_health_accepts_modeling_cache_without_local_source_csvs(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(pipeline_health, "ROOT", tmp_path)
    feature_path = tmp_path / "data/processed/modeling_price_features.csv"
    feature_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp.now(tz="UTC").floor("h").isoformat()],
            "price_eur_mwh": [50.0],
        }
    ).to_csv(feature_path, index=False)
    config = SourceHealthConfig(
        name="modeling_price_features",
        path="data/processed/modeling_price_features.csv",
        required_columns=("timestamp_utc", "price_eur_mwh"),
    )

    report = build_pipeline_health(
        output_path=tmp_path / "pipeline_health.json",
        source_configs=[config],
        include_future=False,
        mode="cloud",
    )

    assert report["mode"] == "cloud"
    assert report["critical_issue_count"] == 0
    assert set(report["sources"]) == {"modeling_price_features"}
