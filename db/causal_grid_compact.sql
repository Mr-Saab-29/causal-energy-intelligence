-- Compact hourly causal feature tables for the 500 MB database budget.

create table if not exists causal_cross_border_hourly_features (
    timestamp_utc timestamptz not null,
    neighbor_bidding_zone text not null,
    vintage_quality text not null check (
        vintage_quality in ('operational_snapshot', 'historical_final')
    ),
    physical_import_mw double precision,
    physical_export_mw double precision,
    net_physical_import_mw double precision,
    scheduled_import_mw double precision,
    scheduled_export_mw double precision,
    net_scheduled_import_mw double precision,
    day_ahead_import_capacity_mw double precision,
    day_ahead_export_capacity_mw double precision,
    source_interval_count integer not null check (source_interval_count >= 0),
    compacted_at_utc timestamptz not null default now(),
    primary key (timestamp_utc, neighbor_bidding_zone, vintage_quality)
);

create index if not exists idx_causal_cross_border_compact_vintage_time
    on causal_cross_border_hourly_features (vintage_quality, timestamp_utc);

create table if not exists causal_france_hourly_features (
    timestamp_utc timestamptz not null,
    vintage_quality text not null check (
        vintage_quality in ('operational_snapshot', 'historical_final')
    ),
    load_forecast_mw double precision,
    wind_onshore_forecast_mw double precision,
    wind_offshore_forecast_mw double precision,
    solar_forecast_mw double precision,
    planned_unavailable_capacity_mw double precision,
    unplanned_unavailable_capacity_mw double precision,
    planned_outage_count integer not null default 0,
    unplanned_outage_count integer not null default 0,
    imbalance_price_up_eur_mwh double precision,
    imbalance_price_down_eur_mwh double precision,
    imbalance_volume_mwh double precision,
    activated_energy_price_eur_mwh double precision,
    forecast_interval_count integer not null default 0,
    balancing_interval_count integer not null default 0,
    compacted_at_utc timestamptz not null default now(),
    primary key (timestamp_utc, vintage_quality)
);

create index if not exists idx_causal_france_compact_vintage_time
    on causal_france_hourly_features (vintage_quality, timestamp_utc);
