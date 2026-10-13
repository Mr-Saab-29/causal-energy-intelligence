from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
import httpx

from src.data.neighbor_emissions import (
    build_neighbor_emissions_readiness,
    build_neighbor_hourly_emissions,
    fetch_elexon_generation,
    load_neighbor_emissions_contract,
    normalize_generation_frame,
)
from src.data.load import build_upsert_sql


def test_elexon_generation_is_normalized_to_hourly_production() -> None:
    payload = {
        "data": [
            {
                "startTime": "2026-08-01T00:00:00Z",
                "publishTime": "2026-08-01T00:30:00Z",
                "fuelType": "CCGT",
                "generation": 100.0,
            },
            {
                "startTime": "2026-08-01T00:30:00Z",
                "publishTime": "2026-08-01T01:00:00Z",
                "fuelType": "CCGT",
                "generation": 140.0,
            },
            {
                "startTime": "2026-08-01T00:30:00Z",
                "publishTime": "2026-08-01T01:00:00Z",
                "fuelType": "INTFR",
                "generation": 500.0,
            },
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["publishDateTimeFrom"].startswith("2026-08-01")
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_elexon_generation(
            pd.Timestamp("2026-08-01", tz="UTC"),
            pd.Timestamp("2026-08-02", tz="UTC"),
            client=client,
        )

    assert len(result) == 1
    assert result.loc[0, "bidding_zone"] == "GB"
    assert result.loc[0, "generation_mwh"] == pytest.approx(120.0)
    assert result.loc[0, "source_interval_count"] == 2


def test_elexon_generation_chunks_never_exceed_seven_inclusive_days() -> None:
    request_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        window_start = pd.Timestamp(request.url.params["publishDateTimeFrom"])
        window_end = pd.Timestamp(request.url.params["publishDateTimeTo"])
        request_windows.append((window_start, window_end))
        assert window_end - window_start <= pd.Timedelta(days=7)
        return httpx.Response(200, json={"data": []})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_elexon_generation(
            pd.Timestamp("2026-08-01", tz="UTC"),
            pd.Timestamp("2026-09-01", tz="UTC"),
            client=client,
        )

    assert result.empty
    assert len(request_windows) == 5
    assert request_windows[0] == (
        pd.Timestamp("2026-08-01", tz="UTC"),
        pd.Timestamp("2026-08-08", tz="UTC"),
    )
    assert request_windows[-1] == (
        pd.Timestamp("2026-08-29", tz="UTC"),
        pd.Timestamp("2026-09-01", tz="UTC"),
    )


def test_generation_normalization_uses_only_actual_aggregated_and_hourly_mean() -> None:
    index = pd.date_range("2026-08-01", periods=4, freq="15min", tz="UTC")
    values = pd.DataFrame(
        {
            ("Fossil Gas", "Actual Aggregated"): [100.0, 120.0, 140.0, 160.0],
            ("Hydro Pumped Storage", "Actual Consumption"): [20.0, 20.0, 20.0, 20.0],
        },
        index=index,
    )

    result = normalize_generation_frame(values, "BE")

    assert len(result) == 1
    assert result.loc[0, "production_type"] == "Fossil Gas"
    assert result.loc[0, "generation_mwh"] == pytest.approx(130.0)
    assert result.loc[0, "source_interval_count"] == 4


def test_hourly_emissions_preserve_fallback_uncertainty() -> None:
    generation = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2026-08-01T00:00:00Z"] * 3),
            "bidding_zone": "BE",
            "production_type": ["Fossil Gas", "Wind Onshore", "Other"],
            "generation_mwh": [100.0, 50.0, 10.0],
            "source_interval_count": [4, 4, 4],
        }
    )
    contract = load_neighbor_emissions_contract()

    result = build_neighbor_hourly_emissions(
        generation,
        contract,
        vintage_quality="historical_final",
        compacted_at=datetime(2026, 8, 2, tzinfo=UTC),
    )

    row = result.iloc[0]
    assert row["total_generation_mwh"] == 160.0
    assert row["named_factor_generation_mwh"] == 150.0
    assert row["fallback_generation_mwh"] == 10.0
    assert row["direct_emissions_kgco2e"] == pytest.approx(40_700.0)
    assert row["direct_emissions_lower_kgco2e"] == pytest.approx(37_000.0)
    assert row["direct_emissions_upper_kgco2e"] == pytest.approx(45_200.0)
    assert row["named_factor_coverage_share"] == pytest.approx(0.9375)


def test_unknown_entsoe_production_type_fails_closed() -> None:
    generation = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2026-08-01T00:00:00Z"]),
            "bidding_zone": "BE",
            "production_type": ["New Production Type"],
            "generation_mwh": [10.0],
            "source_interval_count": [1],
        }
    )

    with pytest.raises(ValueError, match="unmapped ENTSO-E production types"):
        build_neighbor_hourly_emissions(
            generation,
            load_neighbor_emissions_contract(),
            vintage_quality="historical_final",
        )


def test_readiness_requires_all_zones_and_named_factor_coverage() -> None:
    contract = load_neighbor_emissions_contract()
    timestamps = pd.date_range("2026-08-01", periods=24, freq="h", tz="UTC")
    rows = []
    for area in contract["areas"]:
        for timestamp in timestamps:
            rows.append(
                {
                    "timestamp_utc": timestamp,
                    "bidding_zone": area,
                    "total_generation_mwh": 100.0,
                    "named_factor_generation_mwh": 100.0,
                }
            )
    hourly = pd.DataFrame(rows)

    ready = build_neighbor_emissions_readiness(
        hourly,
        contract,
        timestamps.min(),
        timestamps.max() + pd.Timedelta(hours=1),
    )
    missing_gb = build_neighbor_emissions_readiness(
        hourly[hourly["bidding_zone"] != "GB"],
        contract,
        timestamps.min(),
        timestamps.max() + pd.Timedelta(hours=1),
    )

    assert ready["interconnected_boundary_ready"] is True
    assert missing_gb["interconnected_boundary_ready"] is False
    assert "low_hourly_coverage:GB:0.0000" in missing_gb["critical_issues"]


def test_compact_upsert_uses_hour_area_and_vintage_key() -> None:
    statement = build_upsert_sql(
        "causal_neighbor_hourly_emissions",
        [
            "timestamp_utc",
            "bidding_zone",
            "vintage_quality",
            "direct_emissions_kgco2e",
        ],
        ("timestamp_utc", "bidding_zone", "vintage_quality"),
    )

    assert (
        "on conflict (timestamp_utc, bidding_zone, vintage_quality)"
        in statement
    )
    assert "direct_emissions_kgco2e = excluded.direct_emissions_kgco2e" in statement
