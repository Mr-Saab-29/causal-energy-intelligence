from datetime import UTC, datetime

import pandas as pd
import pytest
from requests import HTTPError, Request, Response

from src.data.entsoe_causal_ingest import (
    REQUIRED_CAUSAL_GRID_TABLES,
    build_window,
    main,
    validate_causal_grid_schema,
)
from src.data.sources.entsoe_causal import (
    VINTAGE_HISTORICAL,
    VINTAGE_OPERATIONAL,
    _optional_query,
    fetch_entsoe_causal_data,
    normalize_balancing_frame,
    normalize_cross_border_series,
    normalize_forecast_frame,
    normalize_outage_frame,
)

SNAPSHOT = datetime(2026, 9, 21, 2, tzinfo=UTC)


def test_normalize_cross_border_series_preserves_direction_and_resolution() -> None:
    values = pd.Series(
        [1200.0, 1300.0],
        index=pd.date_range("2026-09-21", periods=2, freq="15min", tz="UTC"),
    )

    rows = normalize_cross_border_series(
        values,
        "BE",
        "FR",
        "physical_flow",
        SNAPSHOT,
        VINTAGE_OPERATIONAL,
    )

    assert len(rows) == 2
    assert rows[0].from_bidding_zone == "BE"
    assert rows[0].to_bidding_zone == "FR"
    assert rows[0].granularity.value == "15m"
    assert rows[0].source_record_id.endswith("2026-09-21T00:00:00+00:00")


def test_operational_schedule_does_not_snapshot_already_settled_hours() -> None:
    values = pd.Series(
        [1200.0, 1300.0],
        index=pd.DatetimeIndex(
            ["2026-09-21T01:00:00Z", "2026-09-21T03:00:00Z"]
        ),
    )

    rows = normalize_cross_border_series(
        values,
        "BE",
        "FR",
        "scheduled_exchange",
        SNAPSHOT,
        VINTAGE_OPERATIONAL,
    )

    assert len(rows) == 1
    assert rows[0].timestamp_utc == datetime(2026, 9, 21, 3, tzinfo=UTC)


def test_normalize_operational_forecasts_retains_snapshot_and_horizon() -> None:
    values = pd.DataFrame(
        {"Solar": [200.0], "Wind Onshore": [500.0]},
        index=pd.DatetimeIndex(["2026-09-21T05:00:00Z"]),
    )

    rows = normalize_forecast_frame(
        values,
        "FR",
        SNAPSHOT,
        VINTAGE_OPERATIONAL,
    )

    assert {row.forecast_type for row in rows} == {"solar", "wind_onshore"}
    assert {row.forecast_horizon_hours for row in rows} == {3}
    assert all(row.forecast_generated_at_utc == SNAPSHOT for row in rows)


def test_normalize_historical_forecast_is_explicitly_final_vintage() -> None:
    values = pd.Series(
        [40_000.0],
        index=pd.DatetimeIndex(["2023-01-01T00:00:00Z"]),
    )

    row = normalize_forecast_frame(
        values,
        "FR",
        SNAPSHOT,
        VINTAGE_HISTORICAL,
        "load",
    )[0]

    assert row.forecast_generated_at_utc is None
    assert row.forecast_horizon_hours is None
    assert row.source_record_id.endswith(VINTAGE_HISTORICAL)


def test_normalize_outage_computes_unavailable_capacity_without_daily_duplicates() -> None:
    values = pd.DataFrame(
        [
            {
                "created_doc_time": "2026-09-20T12:00:00Z",
                "mrid": "outage-1",
                "revision": 2,
                "businesstype": "Planned maintenance",
                "docstatus": "A05",
                "start": "2026-09-21T01:00:00Z",
                "end": "2026-09-21T04:00:00Z",
                "resolution": "PT60M",
                "nominal_power": 100.0,
                "avail_qty": 40.0,
            },
            {
                "created_doc_time": "2026-09-20T13:00:00Z",
                "mrid": "outage-2",
                "revision": 1,
                "businesstype": "Unplanned outage",
                "docstatus": "A05",
                "start": "2026-09-21T02:00:00Z",
                "end": "2026-09-21T05:00:00Z",
                "resolution": "PT60M",
                "nominal_power": 80.0,
                "avail_qty": 20.0,
            },
        ]
    )

    rows = normalize_outage_frame(
        values,
        "FR",
        SNAPSHOT,
        VINTAGE_OPERATIONAL,
    )
    row = rows[0]

    assert row.outage_type == "planned"
    assert float(row.unavailable_capacity_mw) == 60.0
    assert row.source_record_id == "entsoe:outage:outage-1:2:operational_snapshot"
    assert rows[1].outage_type == "unplanned"


def test_normalize_balancing_frame_retains_native_resolution_and_direction() -> None:
    values = pd.DataFrame(
        {"Up": [10.0, 12.0], "Down": [-4.0, -5.0]},
        index=pd.date_range("2026-09-21", periods=2, freq="15min", tz="UTC"),
    )

    rows = normalize_balancing_frame(
        values,
        "FR",
        "imbalance_volume",
        "MWh",
        SNAPSHOT,
        VINTAGE_OPERATIONAL,
    )

    assert len(rows) == 4
    assert {row.direction for row in rows} == {"up", "down"}
    assert {row.granularity.value for row in rows} == {"15m"}


def test_plan_command_needs_no_token_or_database(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "--plan-only",
            "--start-date",
            "2026-09-01",
            "--end-date",
            "2026-09-02",
        ]
    )

    output = capsys.readouterr().out
    assert '"status": "planned"' in output
    assert '"start_utc": "2026-09-01T00:00:00+00:00"' in output
    assert '"end_utc": "2026-09-03T00:00:00+00:00"' in output


def test_build_window_rejects_reversed_dates() -> None:
    with pytest.raises(ValueError, match="end date"):
        build_window("2026-09-03", "2026-09-01", 7)


def test_fetch_rejects_naive_snapshot_before_api_calls() -> None:
    with pytest.raises(ValueError, match="snapshot_at must be timezone-aware"):
        fetch_entsoe_causal_data(
            "token",
            pd.Timestamp("2026-09-01", tz="UTC"),
            pd.Timestamp("2026-09-02", tz="UTC"),
            snapshot_at=datetime(2026, 9, 1),
        )


def test_optional_legacy_query_records_warning_without_exposing_token() -> None:
    response = Response()
    response.status_code = 400
    response.request = Request(
        "GET",
        "https://example.test/api?securityToken=do-not-print",
    ).prepare()

    def rejected_query() -> None:
        raise HTTPError(response=response)

    warnings: list[str] = []
    result = _optional_query(
        rejected_query,
        dataset="activated_energy:afrr",
        tolerate_bad_request=True,
        warnings=warnings,
    )

    assert result is None
    assert warnings == [
        "activated_energy:afrr unavailable from the legacy ENTSO-E endpoint (HTTP 400)"
    ]
    assert "do-not-print" not in warnings[0]


def test_required_query_error_does_not_include_token_bearing_url() -> None:
    response = Response()
    response.status_code = 401
    response.request = Request(
        "GET",
        "https://example.test/api?securityToken=do-not-print",
    ).prepare()

    def rejected_query() -> None:
        raise HTTPError(response=response)

    with pytest.raises(RuntimeError) as error:
        _optional_query(rejected_query, dataset="physical_flow")

    assert str(error.value) == "ENTSO-E query physical_flow failed with HTTP 401"
    assert "do-not-print" not in str(error.value)


def test_schema_preflight_reports_all_missing_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    class Inspector:
        @staticmethod
        def get_table_names(schema: str) -> list[str]:
            assert schema == "public"
            return ["cross_border_observations"]

    monkeypatch.setattr(
        "src.data.entsoe_causal_ingest.inspect",
        lambda engine: Inspector(),
    )

    with pytest.raises(RuntimeError) as error:
        validate_causal_grid_schema(object())

    message = str(error.value)
    assert "grid_forecasts" in message
    assert "generation_outages" in message
    assert "balancing_observations" in message
    assert "db/causal_grid_data.sql" in message


def test_schema_preflight_accepts_complete_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Inspector:
        @staticmethod
        def get_table_names(schema: str) -> list[str]:
            assert schema == "public"
            return list(REQUIRED_CAUSAL_GRID_TABLES)

    monkeypatch.setattr(
        "src.data.entsoe_causal_ingest.inspect",
        lambda engine: Inspector(),
    )

    validate_causal_grid_schema(object())
