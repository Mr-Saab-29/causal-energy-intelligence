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

create table if not exists causal_neighbor_hourly_emissions (
    timestamp_utc timestamptz not null,
    bidding_zone text not null,
    vintage_quality text not null check (
        vintage_quality in ('operational_snapshot', 'historical_final')
    ),
    total_generation_mwh double precision not null check (total_generation_mwh >= 0),
    named_factor_generation_mwh double precision not null check (
        named_factor_generation_mwh >= 0
    ),
    fallback_generation_mwh double precision not null check (
        fallback_generation_mwh >= 0
    ),
    named_factor_coverage_share double precision check (
        named_factor_coverage_share between 0 and 1
    ),
    direct_emissions_kgco2e double precision not null check (
        direct_emissions_kgco2e >= 0
    ),
    direct_emissions_lower_kgco2e double precision not null check (
        direct_emissions_lower_kgco2e >= 0
    ),
    direct_emissions_upper_kgco2e double precision not null check (
        direct_emissions_upper_kgco2e >= 0
    ),
    direct_carbon_intensity_gco2_kwh double precision,
    production_type_count integer not null check (production_type_count >= 0),
    source_interval_count integer not null check (source_interval_count >= 0),
    compacted_at_utc timestamptz not null default now(),
    primary key (timestamp_utc, bidding_zone, vintage_quality)
);

create index if not exists idx_causal_neighbor_emissions_vintage_time
    on causal_neighbor_hourly_emissions (vintage_quality, timestamp_utc);

create table if not exists causal_pre_treatment_hourly_covariates (
    timestamp_utc timestamptz not null,
    vintage_quality text not null check (
        vintage_quality in ('operational_snapshot', 'historical_final')
    ),
    weather_temperature_forecast_c_24h double precision,
    weather_region_count integer not null default 0 check (weather_region_count >= 0),
    gas_price_usd_mmbtu_lag_2m double precision check (
        gas_price_usd_mmbtu_lag_2m is null or gas_price_usd_mmbtu_lag_2m >= 0
    ),
    coal_price_usd_mt_lag_2m double precision check (
        coal_price_usd_mt_lag_2m is null or coal_price_usd_mt_lag_2m >= 0
    ),
    eua_auction_price_eur_tco2 double precision check (
        eua_auction_price_eur_tco2 is null or eua_auction_price_eur_tco2 >= 0
    ),
    hydro_storage_mwh_lag_1w double precision check (
        hydro_storage_mwh_lag_1w is null or hydro_storage_mwh_lag_1w >= 0
    ),
    fuel_reference_month date,
    carbon_auction_at_utc timestamptz,
    hydro_storage_observed_at_utc timestamptz,
    compacted_at_utc timestamptz not null default now(),
    primary key (timestamp_utc, vintage_quality)
);

create index if not exists idx_causal_pre_treatment_vintage_time
    on causal_pre_treatment_hourly_covariates (vintage_quality, timestamp_utc);
