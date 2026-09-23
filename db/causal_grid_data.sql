-- ENTSO-E causal-grid extension. Safe to run repeatedly in Supabase SQL editor.

create table if not exists cross_border_observations (
    id uuid primary key default gen_random_uuid(),
    source text not null,
    source_record_id text not null,
    region text not null,
    timestamp_utc timestamptz not null,
    granularity text not null,
    from_bidding_zone text not null,
    to_bidding_zone text not null,
    metric text not null check (
        metric in ('physical_flow', 'scheduled_exchange', 'day_ahead_capacity')
    ),
    value_mw numeric not null check (value_mw >= 0),
    snapshot_at_utc timestamptz not null,
    vintage_quality text not null check (
        vintage_quality in ('operational_snapshot', 'historical_final')
    ),
    ingestion_timestamp_utc timestamptz not null default now(),
    created_at timestamptz not null default now(),
    unique (source, source_record_id)
);

create index if not exists idx_cross_border_time_metric
    on cross_border_observations (timestamp_utc, metric);
create index if not exists idx_cross_border_pair_time
    on cross_border_observations (from_bidding_zone, to_bidding_zone, timestamp_utc);

create table if not exists grid_forecasts (
    id uuid primary key default gen_random_uuid(),
    source text not null,
    source_record_id text not null,
    region text not null,
    timestamp_utc timestamptz not null,
    granularity text not null,
    forecast_type text not null check (
        forecast_type in ('load', 'wind_onshore', 'wind_offshore', 'solar')
    ),
    forecast_mw numeric not null check (forecast_mw >= 0),
    snapshot_at_utc timestamptz not null,
    forecast_generated_at_utc timestamptz,
    forecast_horizon_hours integer check (
        forecast_horizon_hours is null or forecast_horizon_hours >= 0
    ),
    vintage_quality text not null check (
        vintage_quality in ('operational_snapshot', 'historical_final')
    ),
    ingestion_timestamp_utc timestamptz not null default now(),
    created_at timestamptz not null default now(),
    unique (source, source_record_id)
);

create index if not exists idx_grid_forecasts_target_vintage
    on grid_forecasts (region, forecast_type, timestamp_utc, snapshot_at_utc desc);

create table if not exists generation_outages (
    id uuid primary key default gen_random_uuid(),
    source text not null,
    source_record_id text not null,
    region text not null,
    timestamp_utc timestamptz not null,
    granularity text not null,
    outage_mrid text not null,
    revision_number integer not null check (revision_number >= 0),
    outage_type text not null check (outage_type in ('planned', 'unplanned', 'other')),
    status text,
    end_utc timestamptz not null,
    production_resource_id text,
    production_resource_name text,
    production_type text,
    nominal_capacity_mw numeric check (
        nominal_capacity_mw is null or nominal_capacity_mw >= 0
    ),
    available_capacity_mw numeric check (
        available_capacity_mw is null or available_capacity_mw >= 0
    ),
    unavailable_capacity_mw numeric check (
        unavailable_capacity_mw is null or unavailable_capacity_mw >= 0
    ),
    publication_timestamp_utc timestamptz,
    snapshot_at_utc timestamptz not null,
    vintage_quality text not null check (
        vintage_quality in ('operational_snapshot', 'historical_final')
    ),
    ingestion_timestamp_utc timestamptz not null default now(),
    created_at timestamptz not null default now(),
    unique (source, source_record_id),
    check (end_utc > timestamp_utc)
);

create index if not exists idx_generation_outages_interval
    on generation_outages (region, timestamp_utc, end_utc);
create index if not exists idx_generation_outages_vintage
    on generation_outages (outage_mrid, publication_timestamp_utc, revision_number desc);

create table if not exists balancing_observations (
    id uuid primary key default gen_random_uuid(),
    source text not null,
    source_record_id text not null,
    region text not null,
    timestamp_utc timestamptz not null,
    granularity text not null,
    metric text not null check (
        metric in (
            'activated_energy',
            'activated_energy_price',
            'imbalance_volume',
            'imbalance_price'
        )
    ),
    value numeric not null,
    unit text not null,
    direction text,
    reserve_type text,
    snapshot_at_utc timestamptz not null,
    vintage_quality text not null check (
        vintage_quality in ('operational_snapshot', 'historical_final')
    ),
    ingestion_timestamp_utc timestamptz not null default now(),
    created_at timestamptz not null default now(),
    unique (source, source_record_id)
);

create index if not exists idx_balancing_time_metric
    on balancing_observations (region, timestamp_utc, metric);

create or replace view causal_forecast_errors_hourly as
with latest_forecast as (
    select distinct on (region, forecast_type, timestamp_utc)
        region,
        forecast_type,
        timestamp_utc,
        forecast_mw,
        snapshot_at_utc,
        forecast_generated_at_utc,
        vintage_quality
    from grid_forecasts
    where vintage_quality = 'operational_snapshot'
      and forecast_generated_at_utc <= timestamp_utc
    order by
        region,
        forecast_type,
        timestamp_utc,
        snapshot_at_utc desc
),
actuals as (
    select
        timestamp_utc,
        consumption_mwh as load_actual_mw,
        onshore_wind_mwh as wind_onshore_actual_mw,
        offshore_wind_mwh as wind_offshore_actual_mw,
        solar_mwh as solar_actual_mw
    from hourly_electricity_mix
    where region = 'FR' and scope = 'national'
)
select
    forecast.region,
    forecast.timestamp_utc,
    forecast.forecast_type,
    forecast.forecast_mw,
    case
        when forecast.forecast_type = 'load' then actuals.load_actual_mw
        when forecast.forecast_type = 'wind_onshore' then actuals.wind_onshore_actual_mw
        when forecast.forecast_type = 'wind_offshore' then actuals.wind_offshore_actual_mw
        when forecast.forecast_type = 'solar' then actuals.solar_actual_mw
    end as actual_mw,
    case
        when forecast.forecast_type = 'load' then actuals.load_actual_mw
        when forecast.forecast_type = 'wind_onshore' then actuals.wind_onshore_actual_mw
        when forecast.forecast_type = 'wind_offshore' then actuals.wind_offshore_actual_mw
        when forecast.forecast_type = 'solar' then actuals.solar_actual_mw
    end - forecast.forecast_mw as forecast_error_mw,
    forecast.snapshot_at_utc,
    forecast.forecast_generated_at_utc,
    forecast.vintage_quality
from latest_forecast as forecast
join actuals on actuals.timestamp_utc = forecast.timestamp_utc;

create or replace view causal_cross_border_hourly as
with latest_cross_border as (
    select distinct on (
        metric,
        from_bidding_zone,
        to_bidding_zone,
        timestamp_utc
    )
        timestamp_utc,
        metric,
        from_bidding_zone,
        to_bidding_zone,
        value_mw,
        snapshot_at_utc,
        vintage_quality
    from cross_border_observations
    where from_bidding_zone = 'FR' or to_bidding_zone = 'FR'
    order by
        metric,
        from_bidding_zone,
        to_bidding_zone,
        timestamp_utc,
        snapshot_at_utc desc
),
border_hourly as (
    select
        date_trunc('hour', timestamp_utc) as timestamp_utc,
        metric,
        from_bidding_zone,
        to_bidding_zone,
        avg(value_mw) as value_mw,
        max(snapshot_at_utc) as latest_snapshot_at_utc,
        bool_and(vintage_quality = 'operational_snapshot') as operational_vintage_only
    from latest_cross_border
    group by
        date_trunc('hour', timestamp_utc),
        metric,
        from_bidding_zone,
        to_bidding_zone
)
select
    timestamp_utc,
    metric,
    sum(
        case
            when to_bidding_zone = 'FR' then value_mw
            when from_bidding_zone = 'FR' then -value_mw
            else 0
        end
    ) as net_import_mw,
    max(latest_snapshot_at_utc) as latest_snapshot_at_utc,
    bool_and(operational_vintage_only) as operational_vintage_only
from border_hourly
group by timestamp_utc, metric;

create or replace view causal_balancing_hourly as
select
    region,
    date_trunc('hour', timestamp_utc) as timestamp_utc,
    metric,
    direction,
    reserve_type,
    avg(value) as mean_value,
    sum(abs(value)) as absolute_value_sum,
    min(unit) as unit
from balancing_observations
group by
    region,
    date_trunc('hour', timestamp_utc),
    metric,
    direction,
    reserve_type;
