create extension if not exists pgcrypto;

create table if not exists raw_api_pages (
    id uuid primary key default gen_random_uuid(),
    source_name text not null,
    endpoint text not null,
    request_params jsonb not null default '{}'::jsonb,
    response_payload jsonb,
    ingested_at timestamptz not null default now()
);

create index if not exists idx_raw_api_pages_source_ingested_at
    on raw_api_pages (source_name, ingested_at desc);

create table if not exists electricity_prices (
    id uuid primary key default gen_random_uuid(),
    source text not null,
    source_record_id text,
    region text not null,
    timestamp_utc timestamptz not null,
    granularity text not null,
    market text not null,
    price_eur_mwh numeric not null,
    currency char(3) not null default 'EUR',
    ingestion_timestamp_utc timestamptz not null default now(),
    created_at timestamptz not null default now(),
    unique (source, source_record_id),
    unique (region, timestamp_utc, granularity, market, source)
);

create index if not exists idx_electricity_prices_region_timestamp
    on electricity_prices (region, timestamp_utc);

create table if not exists carbon_intensity (
    id uuid primary key default gen_random_uuid(),
    source text not null,
    source_record_id text,
    region text not null,
    timestamp_utc timestamptz not null,
    granularity text not null,
    carbon_intensity_gco2_kwh numeric not null check (carbon_intensity_gco2_kwh >= 0),
    estimation_method text,
    ingestion_timestamp_utc timestamptz not null default now(),
    created_at timestamptz not null default now(),
    unique (source, source_record_id),
    unique (region, timestamp_utc, granularity, source)
);

create index if not exists idx_carbon_intensity_region_timestamp
    on carbon_intensity (region, timestamp_utc);

create table if not exists grid_demand (
    id uuid primary key default gen_random_uuid(),
    source text not null,
    source_record_id text,
    region text not null,
    timestamp_utc timestamptz not null,
    granularity text not null,
    demand_mw numeric not null check (demand_mw >= 0),
    demand_type text not null default 'actual',
    ingestion_timestamp_utc timestamptz not null default now(),
    created_at timestamptz not null default now(),
    unique (source, source_record_id),
    unique (region, timestamp_utc, granularity, demand_type, source)
);

create index if not exists idx_grid_demand_region_timestamp
    on grid_demand (region, timestamp_utc);

create table if not exists renewable_generation (
    id uuid primary key default gen_random_uuid(),
    source text not null,
    source_record_id text,
    region text not null,
    timestamp_utc timestamptz not null,
    granularity text not null,
    generation_mw numeric not null check (generation_mw >= 0),
    technology text not null,
    ingestion_timestamp_utc timestamptz not null default now(),
    created_at timestamptz not null default now(),
    unique (source, source_record_id),
    unique (region, timestamp_utc, granularity, technology, source)
);

create index if not exists idx_renewable_generation_region_timestamp
    on renewable_generation (region, timestamp_utc);

create table if not exists hourly_electricity_mix (
    id uuid primary key default gen_random_uuid(),
    source text not null,
    source_record_id text,
    region text not null,
    scope text not null check (scope in ('national', 'regional')),
    timestamp_utc timestamptz not null,
    granularity text not null default '1h',
    consumption_mwh numeric check (consumption_mwh is null or consumption_mwh >= 0),
    total_production_mwh numeric check (total_production_mwh is null or total_production_mwh >= 0),
    nuclear_mwh numeric check (nuclear_mwh is null or nuclear_mwh >= 0),
    thermal_mwh numeric check (thermal_mwh is null or thermal_mwh >= 0),
    gas_mwh numeric check (gas_mwh is null or gas_mwh >= 0),
    coal_mwh numeric check (coal_mwh is null or coal_mwh >= 0),
    oil_mwh numeric check (oil_mwh is null or oil_mwh >= 0),
    wind_mwh numeric check (wind_mwh is null or wind_mwh >= 0),
    onshore_wind_mwh numeric check (onshore_wind_mwh is null or onshore_wind_mwh >= 0),
    offshore_wind_mwh numeric check (offshore_wind_mwh is null or offshore_wind_mwh >= 0),
    solar_mwh numeric check (solar_mwh is null or solar_mwh >= 0),
    hydro_mwh numeric check (hydro_mwh is null or hydro_mwh >= 0),
    pumped_storage_mwh numeric,
    bioenergy_mwh numeric check (bioenergy_mwh is null or bioenergy_mwh >= 0),
    battery_storage_mwh numeric,
    physical_exchanges_mwh numeric,
    carbon_intensity_gco2_kwh numeric check (carbon_intensity_gco2_kwh is null or carbon_intensity_gco2_kwh >= 0),
    ingestion_timestamp_utc timestamptz not null default now(),
    created_at timestamptz not null default now(),
    unique (source, source_record_id),
    unique (region, scope, timestamp_utc, source)
);

create index if not exists idx_hourly_electricity_mix_region_timestamp
    on hourly_electricity_mix (region, timestamp_utc);

create table if not exists weather_observations (
    id uuid primary key default gen_random_uuid(),
    source text not null,
    source_record_id text,
    region text not null,
    timestamp_utc timestamptz not null,
    granularity text not null,
    temperature_c numeric,
    apparent_temperature_c numeric,
    relative_humidity_2m_pct numeric check (relative_humidity_2m_pct is null or relative_humidity_2m_pct between 0 and 100),
    dew_point_2m_c numeric,
    precipitation_mm numeric check (precipitation_mm is null or precipitation_mm >= 0),
    rain_mm numeric check (rain_mm is null or rain_mm >= 0),
    snowfall_cm numeric check (snowfall_cm is null or snowfall_cm >= 0),
    cloud_cover_pct numeric check (cloud_cover_pct is null or cloud_cover_pct between 0 and 100),
    cloud_cover_low_pct numeric check (cloud_cover_low_pct is null or cloud_cover_low_pct between 0 and 100),
    cloud_cover_mid_pct numeric check (cloud_cover_mid_pct is null or cloud_cover_mid_pct between 0 and 100),
    cloud_cover_high_pct numeric check (cloud_cover_high_pct is null or cloud_cover_high_pct between 0 and 100),
    shortwave_radiation_wm2 numeric check (shortwave_radiation_wm2 is null or shortwave_radiation_wm2 >= 0),
    direct_radiation_wm2 numeric check (direct_radiation_wm2 is null or direct_radiation_wm2 >= 0),
    diffuse_radiation_wm2 numeric check (diffuse_radiation_wm2 is null or diffuse_radiation_wm2 >= 0),
    wind_speed_mps numeric check (wind_speed_mps is null or wind_speed_mps >= 0),
    wind_speed_80m_mps numeric check (wind_speed_80m_mps is null or wind_speed_80m_mps >= 0),
    wind_direction_10m_deg numeric check (wind_direction_10m_deg is null or wind_direction_10m_deg between 0 and 360),
    wind_direction_80m_deg numeric check (wind_direction_80m_deg is null or wind_direction_80m_deg between 0 and 360),
    wind_gusts_10m_mps numeric check (wind_gusts_10m_mps is null or wind_gusts_10m_mps >= 0),
    surface_pressure_hpa numeric check (surface_pressure_hpa is null or surface_pressure_hpa >= 0),
    weather_code integer,
    solar_irradiance_wm2 numeric check (solar_irradiance_wm2 is null or solar_irradiance_wm2 >= 0),
    humidity_pct numeric check (humidity_pct is null or humidity_pct between 0 and 100),
    ingestion_timestamp_utc timestamptz not null default now(),
    created_at timestamptz not null default now(),
    unique (source, source_record_id),
    unique (region, timestamp_utc, granularity, source)
);

create index if not exists idx_weather_observations_region_timestamp
    on weather_observations (region, timestamp_utc);

create table if not exists workload_windows (
    id uuid primary key default gen_random_uuid(),
    workload_id text not null,
    region text not null,
    earliest_start_utc timestamptz not null,
    latest_end_utc timestamptz not null,
    duration_minutes integer not null check (duration_minutes > 0),
    power_kw numeric not null check (power_kw > 0),
    max_delay_minutes integer not null check (max_delay_minutes >= 0),
    service_level text not null default 'standard',
    created_at timestamptz not null default now(),
    check (latest_end_utc > earliest_start_utc)
);

create index if not exists idx_workload_windows_region_start
    on workload_windows (region, earliest_start_utc);
