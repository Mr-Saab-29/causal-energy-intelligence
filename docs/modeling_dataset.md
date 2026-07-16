# Modeling Dataset

The first modeling dataset targets **France day-ahead electricity spot prices**.

## Target

- `price_eur_mwh`

## Grain

- One row per UTC hour.
- National France level.
- Source target: Energy-Charts day-ahead spot price.
- Source predictors: ODRE national electricity mix and compact aggregated Open-Meteo weather from `weather_france_hourly_agg`.

## Strict Forecasting Features

The strict forecasting feature set includes time features and lagged/rolling predictors that would be available before prediction time:

- `price_lag_1h`
- `price_lag_24h`
- `price_lag_48h`
- `price_lag_72h`
- `price_lag_168h`
- `price_lag_336h`
- `price_rolling_mean_24h`
- `price_rolling_mean_168h`
- `price_rolling_std_24h`
- `price_rolling_min_24h`
- `price_rolling_max_24h`
- `price_rolling_range_24h`
- `price_lag_1h_to_24h`
- `price_lag_24h_to_168h`
- `price_lag_168h_to_336h`
- `price_vs_rolling_mean_24h`
- `consumption_lag_24h`
- `consumption_lag_48h`
- `consumption_lag_72h`
- `consumption_lag_168h`
- `consumption_lag_336h`
- consumption rolling and momentum features
- `consumption_lag_24h_to_168h`
- total-production lag, rolling, and momentum features
- source-level production lag, rolling, and momentum features for nuclear, gas, coal, oil, wind, solar, hydro, and bioenergy
- `wind_lag_24h`
- `wind_lag_168h`
- `wind_lag_24h_to_168h`
- `solar_lag_24h`
- `solar_lag_168h`
- `solar_lag_24h_to_168h`
- `residual_demand_lag_24h`
- `residual_demand_lag_168h`
- `residual_demand_lag_24h_to_168h`
- `variable_renewable_lag_24h`
- `variable_renewable_lag_168h`
- `variable_renewable_lag_24h_to_168h`
- calendar features: `hour`, `day_of_week`, `month`, `is_weekend`, `is_peak_hour`, `is_morning_ramp`, `is_evening_peak`, `is_overnight`, cyclic encodings

## Causal-Ready Features

The dataset also preserves contemporaneous electricity and weather columns for causal analysis:

- `consumption_mwh`
- `total_production_mwh`
- `nuclear_mwh`
- `thermal_mwh`
- `gas_mwh`
- `coal_mwh`
- `oil_mwh`
- `wind_mwh`
- `solar_mwh`
- `hydro_mwh`
- `bioenergy_mwh`
- `physical_exchanges_mwh`
- `renewable_mwh`
- `fossil_mwh`
- `renewable_share`
- `nuclear_share`
- `fossil_share`
- `residual_demand_mwh`
- `supply_demand_gap_mwh`
- aggregated weather variables

For strict forecasting, avoid using contemporaneous realized generation/load/weather unless those values are replaced with forecasts or lagged versions.

## Apply Schema

Run `db/modeling_features.sql` in Supabase SQL editor.

If you still have `weather_observations`, create the compact weather table before dropping it by running `db/compact_weather.sql`.

If you already dropped `weather_observations`, verify that `weather_france_hourly_agg` exists:

```sql
select count(*), min(timestamp_utc), max(timestamp_utc)
from weather_france_hourly_agg;
```

## Build Dataset

From the repository root:

```bash
set -a
source .env
set +a

python3 - <<'PY'
import os
from src.data.load import create_database_engine
from src.features.price_features import build_and_store_price_modeling_features

engine = create_database_engine(os.environ["DATABASE_URL"])
features = build_and_store_price_modeling_features(engine)
print(features.shape)
print(features["timestamp_utc"].min(), features["timestamp_utc"].max())
PY
```

This writes:

- local CSV: `data/processed/modeling_price_features.csv`
- Supabase table: `modeling_price_features`
