-- Data quality checks for ingested canonical tables.

-- 1. Row counts and timestamp coverage.
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
from weather_observations;

-- 2. Duplicate source IDs.
select
    'electricity_prices' as table_name,
    source,
    source_record_id,
    count(*) as duplicate_count
from electricity_prices
group by source, source_record_id
having count(*) > 1
union all
select
    'hourly_electricity_mix' as table_name,
    source,
    source_record_id,
    count(*) as duplicate_count
from hourly_electricity_mix
group by source, source_record_id
having count(*) > 1
union all
select
    'weather_observations' as table_name,
    source,
    source_record_id,
    count(*) as duplicate_count
from weather_observations
group by source, source_record_id
having count(*) > 1;

-- 3. Missing hourly timestamps by price region.
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
    count(*) as missing_price_hours
from expected_hours
left join electricity_prices
    on electricity_prices.region = expected_hours.region
   and date_trunc('hour', electricity_prices.timestamp_utc) = expected_hours.timestamp_utc
where electricity_prices.id is null
group by expected_hours.region;

-- 4. Missing hourly timestamps by electricity-mix region.
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
    count(*) as missing_mix_hours
from expected_hours
left join hourly_electricity_mix
    on hourly_electricity_mix.region = expected_hours.region
   and hourly_electricity_mix.scope = expected_hours.scope
   and date_trunc('hour', hourly_electricity_mix.timestamp_utc) = expected_hours.timestamp_utc
where hourly_electricity_mix.id is null
group by expected_hours.region, expected_hours.scope
order by expected_hours.scope, expected_hours.region;

-- 5. Missing hourly timestamps by weather region.
with bounds as (
    select
        region,
        min(date_trunc('hour', timestamp_utc)) as min_hour,
        max(date_trunc('hour', timestamp_utc)) as max_hour
    from weather_observations
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
    count(*) as missing_weather_hours
from expected_hours
left join weather_observations
    on weather_observations.region = expected_hours.region
   and date_trunc('hour', weather_observations.timestamp_utc) = expected_hours.timestamp_utc
where weather_observations.id is null
group by expected_hours.region
order by expected_hours.region;

-- 6. Key null rates.
select
    'hourly_electricity_mix' as table_name,
    count(*) filter (where consumption_mwh is null) as null_consumption_mwh,
    count(*) filter (where total_production_mwh is null) as null_total_production_mwh,
    count(*) filter (where carbon_intensity_gco2_kwh is null) as null_carbon_intensity
from hourly_electricity_mix;

select
    'weather_observations' as table_name,
    count(*) filter (where temperature_c is null) as null_temperature_c,
    count(*) filter (where wind_speed_mps is null) as null_wind_speed_mps,
    count(*) filter (where shortwave_radiation_wm2 is null) as null_shortwave_radiation_wm2
from weather_observations;

