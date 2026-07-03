# Data Contracts

This project should ingest upstream API/CSV data into a stable internal schema before modeling. The API-specific response shape should not leak into forecasting, causal inference, or optimization modules.

## Ingestion Strategy

- Use paginated API requests with configurable `page_size`, `max_pages`, timeout, and minimum interval between requests.
- Prefer date-window extraction for historical backfills from 2023-01-01 through 2026-06-30.
- Store raw API pages in `raw_api_pages` before transformation for lineage, debugging, and replay.
- Normalize all timestamps to timezone-aware UTC.
- Use source-specific adapters to map upstream fields into canonical contracts.
- Use idempotent keys: `source + source_record_id` where available, otherwise `region + timestamp_utc + granularity + source + domain-specific dimensions`.

## Source-Specific Contracts

### ODRE éCO2mix Electricity Mix

- National historical dataset: `eco2mix-national-cons-def`.
- Regional historical dataset: `eco2mix-regional-cons-def`.
- National real-time dataset: `eco2mix-national-tr`.
- Regional real-time dataset: `eco2mix-regional-tr`.
- Native resolution: 15-minute MW.
- Canonical output: hourly MWh in `hourly_electricity_mix`.
- Aggregation rule: each 15-minute MW value contributes `MW * 0.25` MWh to the containing UTC hour.
- Scope: both France national (`FR`) and regional French records.
- Extraction strategy: 7-day date windows, then offset pagination with `limit=100`.
- Note: the `*-tr` real-time datasets do not cover the full 2023-01-01 to 2026-06-30 historical range, so historical backfills should use `*-cons-def` where available and real-time datasets for the latest provisional period.

### Open-Meteo Historical Weather

- Canonical table: `weather_observations`.
- Native resolution: hourly.
- Extraction strategy: one request per location and date range where safe; split into date windows if the request gets too large.
- Requested hourly variables:
  - `temperature_2m`
  - `apparent_temperature`
  - `relative_humidity_2m`
  - `dew_point_2m`
  - `precipitation`
  - `rain`
  - `snowfall`
  - `cloud_cover`
  - `cloud_cover_low`
  - `cloud_cover_mid`
  - `cloud_cover_high`
  - `shortwave_radiation`
  - `direct_radiation`
  - `diffuse_radiation`
  - `wind_speed_10m`
  - `wind_speed_80m`
  - `wind_direction_10m`
  - `wind_direction_80m`
  - `wind_gusts_10m`
  - `surface_pressure`
  - `weather_code`

### Energy-Charts Day-Ahead Electricity Prices

- Canonical table: `electricity_prices`.
- Bidding zone: France `FR`.
- Canonical unit: EUR/MWh.
- Extraction strategy: date windows of at most 365 days per request.
- Auth: no token required.

### ENTSO-E Day-Ahead Electricity Prices

- Canonical table: `electricity_prices`.
- Bidding zone: France `10YFR-RTE------C`.
- Document type: `A44` day-ahead prices.
- Market agreement: `A01`.
- Canonical unit: EUR/MWh.
- Extraction strategy: date windows of at most 365 days per request.

## Canonical Tables

### `raw_api_pages`

Stores raw API response pages.

| Column | Type | Meaning |
| --- | --- | --- |
| `source_name` | `text` | Upstream source identifier. |
| `endpoint` | `text` | API endpoint path. |
| `request_params` | `jsonb` | Query parameters used for the request. |
| `response_payload` | `jsonb` | Raw response body. |
| `ingested_at` | `timestamptz` | Time the page was stored. |

### `electricity_prices`

Market electricity price observations.

Required fields: `source`, `region`, `timestamp_utc`, `granularity`, `market`, `price_eur_mwh`, `currency`.

### `hourly_electricity_mix`

Hourly MWh electricity production and consumption observations derived from ODRE 15-minute MW values.

Required fields: `source`, `region`, `scope`, `timestamp_utc`, `granularity`.

Important measures: `consumption_mwh`, `total_production_mwh`, `nuclear_mwh`, `thermal_mwh`, `gas_mwh`, `coal_mwh`, `oil_mwh`, `wind_mwh`, `solar_mwh`, `hydro_mwh`, `bioenergy_mwh`, `carbon_intensity_gco2_kwh`.

### `carbon_intensity`

Grid carbon intensity observations.

Required fields: `source`, `region`, `timestamp_utc`, `granularity`, `carbon_intensity_gco2_kwh`.

### `grid_demand`

Regional electricity demand/load observations.

Required fields: `source`, `region`, `timestamp_utc`, `granularity`, `demand_mw`, `demand_type`.

### `renewable_generation`

Renewable generation observations by technology.

Required fields: `source`, `region`, `timestamp_utc`, `granularity`, `generation_mw`, `technology`.

### `weather_observations`

Weather covariates for forecasting and causal adjustment.

Supported fields: `temperature_c`, `wind_speed_mps`, `solar_irradiance_wm2`, `humidity_pct`.

### `workload_windows`

Schedulable workloads used by what-if optimization.

Required fields: `workload_id`, `region`, `earliest_start_utc`, `latest_end_utc`, `duration_minutes`, `power_kw`, `max_delay_minutes`.

## Supabase Storage

Supabase should be treated as managed Postgres for this project. Use:

- `DATABASE_URL` for backend ETL/API writes through SQLAlchemy.
- Supabase SQL editor or migrations to apply `db/schema.sql`.
- Row Level Security disabled initially for backend-only tables, or enabled later with explicit service-role access policies.

## Information Still Needed

To implement source-specific ingestion, we need:

1. Supabase `DATABASE_URL`.
2. ENTSO-E API token.
3. Weather locations for national and regional France coverage.
4. Whether regional weather should use one representative coordinate per administrative region or multiple coordinates averaged per region.

## Rate Limit Notes

- Open-Meteo free non-commercial use is documented as less than 10,000 calls/day, 5,000/hour, and 600/minute.
- ENTSO-E R3 API rate limit is documented as 400 requests/minute per API token, with each request covering up to one year and a response cap of 100 `TimeSeries` elements. This is now treated as a fallback path.
- Opendatasoft Explore API records endpoint is capped at 100 records per request without `group_by`; `offset + limit` must remain below 10,000, so this implementation uses small date windows.
