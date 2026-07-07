"""Data quality checks for ingested Supabase tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class QualityCheckResult:
    """Result from one database quality check."""

    name: str
    passed: bool
    rows: list[dict[str, Any]]


def run_quality_checks(engine: Engine) -> list[QualityCheckResult]:
    """Run core quality checks against canonical ingestion tables."""
    return [
        _run_check(engine, "table_coverage", TABLE_COVERAGE_SQL, expect_rows=True),
        _run_check(engine, "duplicate_source_ids", DUPLICATE_SOURCE_IDS_SQL, expect_empty=True),
        _run_check(engine, "missing_price_hours", MISSING_PRICE_HOURS_SQL, max_count_column="missing_price_hours"),
        _run_check(engine, "missing_mix_hours", MISSING_MIX_HOURS_SQL, max_count_column="missing_mix_hours"),
        _run_check(engine, "missing_weather_hours", MISSING_WEATHER_HOURS_SQL, max_count_column="missing_weather_hours"),
        _run_check(engine, "key_null_rates", KEY_NULL_RATES_SQL, expect_rows=True),
    ]


def print_quality_report(results: list[QualityCheckResult]) -> None:
    """Print a compact quality report."""
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}: {result.name}")
        for row in result.rows[:10]:
            print(f"  {row}")
        if len(result.rows) > 10:
            print(f"  ... {len(result.rows) - 10} more rows")


def _run_check(
    engine: Engine,
    name: str,
    sql: str,
    expect_empty: bool = False,
    expect_rows: bool = False,
    max_count_column: str | None = None,
) -> QualityCheckResult:
    with engine.begin() as connection:
        rows = [dict(row._mapping) for row in connection.execute(text(sql))]

    if expect_empty:
        passed = len(rows) == 0
    elif expect_rows:
        passed = len(rows) > 0
    elif max_count_column:
        passed = all(int(row.get(max_count_column) or 0) == 0 for row in rows)
    else:
        passed = True

    return QualityCheckResult(name=name, passed=passed, rows=rows)


TABLE_COVERAGE_SQL = """
select
    'electricity_prices' as table_name,
    count(*) as row_count,
    min(timestamp_utc) as min_timestamp_utc,
    max(timestamp_utc) as max_timestamp_utc
from electricity_prices
union all
select
    'hourly_electricity_mix' as table_name,
    count(*) as row_count,
    min(timestamp_utc) as min_timestamp_utc,
    max(timestamp_utc) as max_timestamp_utc
from hourly_electricity_mix
union all
select
    'weather_observations' as table_name,
    count(*) as row_count,
    min(timestamp_utc) as min_timestamp_utc,
    max(timestamp_utc) as max_timestamp_utc
from weather_observations
"""

DUPLICATE_SOURCE_IDS_SQL = """
select table_name, source, source_record_id, duplicate_count
from (
    select 'electricity_prices' as table_name, source, source_record_id, count(*) as duplicate_count
    from electricity_prices
    group by source, source_record_id
    having count(*) > 1
    union all
    select 'hourly_electricity_mix' as table_name, source, source_record_id, count(*) as duplicate_count
    from hourly_electricity_mix
    group by source, source_record_id
    having count(*) > 1
    union all
    select 'weather_observations' as table_name, source, source_record_id, count(*) as duplicate_count
    from weather_observations
    group by source, source_record_id
    having count(*) > 1
) duplicates
"""

MISSING_PRICE_HOURS_SQL = """
with bounds as (
    select region, min(date_trunc('hour', timestamp_utc)) as min_hour, max(date_trunc('hour', timestamp_utc)) as max_hour
    from electricity_prices
    group by region
),
expected_hours as (
    select region, generate_series(min_hour, max_hour, interval '1 hour') as timestamp_utc
    from bounds
)
select expected_hours.region, count(*) as missing_price_hours
from expected_hours
left join electricity_prices
    on electricity_prices.region = expected_hours.region
   and date_trunc('hour', electricity_prices.timestamp_utc) = expected_hours.timestamp_utc
where electricity_prices.id is null
group by expected_hours.region
"""

MISSING_MIX_HOURS_SQL = """
with bounds as (
    select region, scope, min(date_trunc('hour', timestamp_utc)) as min_hour, max(date_trunc('hour', timestamp_utc)) as max_hour
    from hourly_electricity_mix
    group by region, scope
),
expected_hours as (
    select region, scope, generate_series(min_hour, max_hour, interval '1 hour') as timestamp_utc
    from bounds
)
select expected_hours.region, expected_hours.scope, count(*) as missing_mix_hours
from expected_hours
left join hourly_electricity_mix
    on hourly_electricity_mix.region = expected_hours.region
   and hourly_electricity_mix.scope = expected_hours.scope
   and date_trunc('hour', hourly_electricity_mix.timestamp_utc) = expected_hours.timestamp_utc
where hourly_electricity_mix.id is null
group by expected_hours.region, expected_hours.scope
order by expected_hours.scope, expected_hours.region
"""

MISSING_WEATHER_HOURS_SQL = """
with bounds as (
    select region, min(date_trunc('hour', timestamp_utc)) as min_hour, max(date_trunc('hour', timestamp_utc)) as max_hour
    from weather_observations
    group by region
),
expected_hours as (
    select region, generate_series(min_hour, max_hour, interval '1 hour') as timestamp_utc
    from bounds
)
select expected_hours.region, count(*) as missing_weather_hours
from expected_hours
left join weather_observations
    on weather_observations.region = expected_hours.region
   and date_trunc('hour', weather_observations.timestamp_utc) = expected_hours.timestamp_utc
where weather_observations.id is null
group by expected_hours.region
order by expected_hours.region
"""

KEY_NULL_RATES_SQL = """
select
    'hourly_electricity_mix' as table_name,
    count(*) filter (where consumption_mwh is null) as null_primary_measure_1,
    count(*) filter (where total_production_mwh is null) as null_primary_measure_2,
    count(*) filter (where carbon_intensity_gco2_kwh is null) as null_primary_measure_3
from hourly_electricity_mix
union all
select
    'weather_observations' as table_name,
    count(*) filter (where temperature_c is null) as null_primary_measure_1,
    count(*) filter (where wind_speed_mps is null) as null_primary_measure_2,
    count(*) filter (where shortwave_radiation_wm2 is null) as null_primary_measure_3
from weather_observations
"""

