-- Hourly feature view for forecasting and causal analysis.

create or replace view hourly_france_features
with (security_invoker = true) as
select
    mix.timestamp_utc,
    mix.region,
    mix.scope,
    price.price_eur_mwh,
    mix.consumption_mwh,
    mix.total_production_mwh,
    mix.nuclear_mwh,
    mix.thermal_mwh,
    mix.gas_mwh,
    mix.coal_mwh,
    mix.oil_mwh,
    mix.wind_mwh,
    mix.onshore_wind_mwh,
    mix.offshore_wind_mwh,
    mix.solar_mwh,
    mix.hydro_mwh,
    mix.pumped_storage_mwh,
    mix.bioenergy_mwh,
    mix.battery_storage_mwh,
    mix.physical_exchanges_mwh,
    mix.carbon_intensity_gco2_kwh,
    weather.temperature_c,
    weather.apparent_temperature_c,
    weather.relative_humidity_2m_pct,
    weather.dew_point_2m_c,
    weather.precipitation_mm,
    weather.rain_mm,
    weather.snowfall_cm,
    weather.cloud_cover_pct,
    weather.cloud_cover_low_pct,
    weather.cloud_cover_mid_pct,
    weather.cloud_cover_high_pct,
    weather.shortwave_radiation_wm2,
    weather.direct_radiation_wm2,
    weather.diffuse_radiation_wm2,
    weather.wind_speed_mps,
    weather.wind_speed_80m_mps,
    weather.wind_direction_10m_deg,
    weather.wind_direction_80m_deg,
    weather.wind_gusts_10m_mps,
    weather.surface_pressure_hpa,
    weather.weather_code
from hourly_electricity_mix as mix
left join electricity_prices as price
    on price.region = 'FR'
   and date_trunc('hour', price.timestamp_utc) = date_trunc('hour', mix.timestamp_utc)
   and price.market = 'day_ahead'
left join weather_observations as weather
    on weather.region = mix.region
   and date_trunc('hour', weather.timestamp_utc) = date_trunc('hour', mix.timestamp_utc);
