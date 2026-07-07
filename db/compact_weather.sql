create table if not exists weather_france_hourly_agg as
select
    timestamp_utc,
    avg(temperature_c) as avg_temperature_c,
    min(temperature_c) as min_temperature_c,
    max(temperature_c) as max_temperature_c,
    avg(apparent_temperature_c) as avg_apparent_temperature_c,
    avg(wind_speed_mps) as avg_wind_speed_mps,
    avg(wind_speed_80m_mps) as avg_wind_speed_80m_mps,
    avg(shortwave_radiation_wm2) as avg_shortwave_radiation_wm2,
    avg(cloud_cover_pct) as avg_cloud_cover_pct,
    avg(precipitation_mm) as avg_precipitation_mm,
    sum(precipitation_mm) as total_precipitation_mm,
    avg(surface_pressure_hpa) as avg_surface_pressure_hpa
from weather_observations
group by timestamp_utc;

create index if not exists idx_weather_france_hourly_agg_timestamp
    on weather_france_hourly_agg (timestamp_utc);

