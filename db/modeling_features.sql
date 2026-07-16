create table if not exists modeling_price_features (
    timestamp_utc timestamptz primary key,
    price_eur_mwh numeric not null,
    consumption_mwh numeric not null,
    total_production_mwh numeric not null,
    nuclear_mwh numeric,
    thermal_mwh numeric,
    gas_mwh numeric,
    coal_mwh numeric,
    oil_mwh numeric,
    wind_mwh numeric,
    solar_mwh numeric,
    hydro_mwh numeric,
    bioenergy_mwh numeric,
    physical_exchanges_mwh numeric,
    renewable_mwh numeric,
    fossil_mwh numeric,
    renewable_share numeric,
    nuclear_share numeric,
    fossil_share numeric,
    residual_demand_mwh numeric,
    supply_demand_gap_mwh numeric,
    avg_temperature_c numeric,
    min_temperature_c numeric,
    max_temperature_c numeric,
    avg_apparent_temperature_c numeric,
    avg_wind_speed_mps numeric,
    avg_wind_speed_80m_mps numeric,
    avg_shortwave_radiation_wm2 numeric,
    avg_cloud_cover_pct numeric,
    avg_precipitation_mm numeric,
    total_precipitation_mm numeric,
    avg_surface_pressure_hpa numeric,
    hour integer not null,
    day_of_week integer not null,
    month integer not null,
    day_of_year integer not null,
    is_weekend boolean not null,
    is_peak_hour boolean not null,
    is_morning_ramp boolean,
    is_evening_peak boolean,
    is_overnight boolean,
    hour_sin numeric not null,
    hour_cos numeric not null,
    day_of_year_sin numeric not null,
    day_of_year_cos numeric not null,
    price_lag_1h numeric,
    price_lag_24h numeric,
    price_lag_48h numeric,
    price_lag_72h numeric,
    price_lag_168h numeric,
    price_lag_336h numeric,
    price_rolling_mean_24h numeric,
    price_rolling_mean_168h numeric,
    price_rolling_std_24h numeric,
    price_rolling_min_24h numeric,
    price_rolling_max_24h numeric,
    price_rolling_range_24h numeric,
    price_lag_1h_to_24h numeric,
    price_lag_24h_to_168h numeric,
    price_lag_168h_to_336h numeric,
    price_vs_rolling_mean_24h numeric,
    consumption_lag_1h numeric,
    consumption_lag_24h numeric,
    consumption_lag_48h numeric,
    consumption_lag_72h numeric,
    consumption_lag_168h numeric,
    consumption_lag_336h numeric,
    consumption_rolling_mean_24h numeric,
    consumption_rolling_mean_168h numeric,
    consumption_rolling_std_24h numeric,
    consumption_rolling_min_24h numeric,
    consumption_rolling_max_24h numeric,
    consumption_rolling_range_24h numeric,
    consumption_lag_1h_to_24h numeric,
    consumption_lag_24h_to_168h numeric,
    total_production_lag_1h numeric,
    total_production_lag_24h numeric,
    total_production_lag_48h numeric,
    total_production_lag_72h numeric,
    total_production_lag_168h numeric,
    total_production_lag_336h numeric,
    total_production_rolling_mean_24h numeric,
    total_production_rolling_mean_168h numeric,
    total_production_rolling_std_24h numeric,
    total_production_rolling_min_24h numeric,
    total_production_rolling_max_24h numeric,
    total_production_rolling_range_24h numeric,
    total_production_lag_1h_to_24h numeric,
    total_production_lag_24h_to_168h numeric,
    wind_lag_24h numeric,
    wind_lag_168h numeric,
    wind_lag_24h_to_168h numeric,
    solar_lag_24h numeric,
    solar_lag_168h numeric,
    solar_lag_24h_to_168h numeric,
    residual_demand_lag_24h numeric,
    residual_demand_lag_168h numeric,
    residual_demand_lag_24h_to_168h numeric,
    variable_renewable_lag_24h numeric,
    variable_renewable_lag_168h numeric,
    variable_renewable_lag_24h_to_168h numeric,
    created_at timestamptz not null default now()
);

alter table modeling_price_features add column if not exists is_morning_ramp boolean;
alter table modeling_price_features add column if not exists is_evening_peak boolean;
alter table modeling_price_features add column if not exists is_overnight boolean;
alter table modeling_price_features add column if not exists price_lag_48h numeric;
alter table modeling_price_features add column if not exists price_lag_72h numeric;
alter table modeling_price_features add column if not exists price_lag_336h numeric;
alter table modeling_price_features add column if not exists price_rolling_std_24h numeric;
alter table modeling_price_features add column if not exists price_rolling_min_24h numeric;
alter table modeling_price_features add column if not exists price_rolling_max_24h numeric;
alter table modeling_price_features add column if not exists price_rolling_range_24h numeric;
alter table modeling_price_features add column if not exists price_lag_1h_to_24h numeric;
alter table modeling_price_features add column if not exists price_lag_24h_to_168h numeric;
alter table modeling_price_features add column if not exists price_lag_168h_to_336h numeric;
alter table modeling_price_features add column if not exists price_vs_rolling_mean_24h numeric;
alter table modeling_price_features add column if not exists consumption_lag_1h numeric;
alter table modeling_price_features add column if not exists consumption_lag_48h numeric;
alter table modeling_price_features add column if not exists consumption_lag_72h numeric;
alter table modeling_price_features add column if not exists consumption_lag_168h numeric;
alter table modeling_price_features add column if not exists consumption_lag_336h numeric;
alter table modeling_price_features add column if not exists consumption_rolling_mean_24h numeric;
alter table modeling_price_features add column if not exists consumption_rolling_mean_168h numeric;
alter table modeling_price_features add column if not exists consumption_rolling_std_24h numeric;
alter table modeling_price_features add column if not exists consumption_rolling_min_24h numeric;
alter table modeling_price_features add column if not exists consumption_rolling_max_24h numeric;
alter table modeling_price_features add column if not exists consumption_rolling_range_24h numeric;
alter table modeling_price_features add column if not exists consumption_lag_1h_to_24h numeric;
alter table modeling_price_features add column if not exists consumption_lag_24h_to_168h numeric;
alter table modeling_price_features add column if not exists total_production_lag_1h numeric;
alter table modeling_price_features add column if not exists total_production_lag_24h numeric;
alter table modeling_price_features add column if not exists total_production_lag_48h numeric;
alter table modeling_price_features add column if not exists total_production_lag_72h numeric;
alter table modeling_price_features add column if not exists total_production_lag_168h numeric;
alter table modeling_price_features add column if not exists total_production_lag_336h numeric;
alter table modeling_price_features add column if not exists total_production_rolling_mean_24h numeric;
alter table modeling_price_features add column if not exists total_production_rolling_mean_168h numeric;
alter table modeling_price_features add column if not exists total_production_rolling_std_24h numeric;
alter table modeling_price_features add column if not exists total_production_rolling_min_24h numeric;
alter table modeling_price_features add column if not exists total_production_rolling_max_24h numeric;
alter table modeling_price_features add column if not exists total_production_rolling_range_24h numeric;
alter table modeling_price_features add column if not exists total_production_lag_1h_to_24h numeric;
alter table modeling_price_features add column if not exists total_production_lag_24h_to_168h numeric;
alter table modeling_price_features add column if not exists wind_lag_168h numeric;
alter table modeling_price_features add column if not exists wind_lag_24h_to_168h numeric;
alter table modeling_price_features add column if not exists solar_lag_168h numeric;
alter table modeling_price_features add column if not exists solar_lag_24h_to_168h numeric;
alter table modeling_price_features add column if not exists residual_demand_lag_24h numeric;
alter table modeling_price_features add column if not exists residual_demand_lag_168h numeric;
alter table modeling_price_features add column if not exists residual_demand_lag_24h_to_168h numeric;
alter table modeling_price_features add column if not exists variable_renewable_lag_24h numeric;
alter table modeling_price_features add column if not exists variable_renewable_lag_168h numeric;
alter table modeling_price_features add column if not exists variable_renewable_lag_24h_to_168h numeric;

do $$
declare
    feature_prefix text;
    feature_suffix text;
begin
    foreach feature_prefix in array array[
        'nuclear',
        'gas',
        'coal',
        'oil',
        'wind',
        'solar',
        'hydro',
        'bioenergy'
    ]
    loop
        foreach feature_suffix in array array[
            'lag_1h',
            'lag_24h',
            'lag_48h',
            'lag_72h',
            'lag_168h',
            'lag_336h',
            'rolling_mean_24h',
            'rolling_mean_168h',
            'rolling_std_24h',
            'rolling_min_24h',
            'rolling_max_24h',
            'rolling_range_24h',
            'lag_1h_to_24h',
            'lag_24h_to_168h'
        ]
        loop
            execute format(
                'alter table modeling_price_features add column if not exists %I numeric',
                feature_prefix || '_' || feature_suffix
            );
        end loop;
    end loop;
end $$;

create index if not exists idx_modeling_price_features_timestamp
    on modeling_price_features (timestamp_utc);
