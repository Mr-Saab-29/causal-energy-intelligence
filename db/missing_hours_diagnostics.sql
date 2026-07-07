-- Exact missing-hour diagnostics.

-- Missing price hours.
with bounds as (
    select
        region,
        min(date_trunc('hour', timestamp_utc)) as min_hour,
        max(date_trunc('hour', timestamp_utc)) as max_hour
    from electricity_prices
    group by region
),
expected_hours as (
    select
        region,
        generate_series(min_hour, max_hour, interval '1 hour') as timestamp_utc
    from bounds
)
select
    expected_hours.region,
    expected_hours.timestamp_utc
from expected_hours
left join electricity_prices
    on electricity_prices.region = expected_hours.region
   and date_trunc('hour', electricity_prices.timestamp_utc) = expected_hours.timestamp_utc
where electricity_prices.id is null
order by expected_hours.region, expected_hours.timestamp_utc;

-- Missing electricity-mix hours.
with bounds as (
    select
        region,
        scope,
        min(date_trunc('hour', timestamp_utc)) as min_hour,
        max(date_trunc('hour', timestamp_utc)) as max_hour
    from hourly_electricity_mix
    group by region, scope
),
expected_hours as (
    select
        region,
        scope,
        generate_series(min_hour, max_hour, interval '1 hour') as timestamp_utc
    from bounds
)
select
    expected_hours.region,
    expected_hours.scope,
    expected_hours.timestamp_utc
from expected_hours
left join hourly_electricity_mix
    on hourly_electricity_mix.region = expected_hours.region
   and hourly_electricity_mix.scope = expected_hours.scope
   and date_trunc('hour', hourly_electricity_mix.timestamp_utc) = expected_hours.timestamp_utc
where hourly_electricity_mix.id is null
order by expected_hours.scope, expected_hours.region, expected_hours.timestamp_utc;

