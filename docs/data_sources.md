# Data Sources

## Electricity Production and Consumption

- Provider: ODRE / Opendatasoft éCO2mix.
- Historical national dataset: `eco2mix-national-cons-def`.
- Historical regional dataset: `eco2mix-regional-cons-def`.
- Real-time national dataset: `eco2mix-national-tr`.
- Real-time regional dataset: `eco2mix-regional-tr`.
- Native unit: MW at 15-minute resolution.
- Canonical unit: hourly MWh.
- Aggregation: `MWh = MW * 0.25` for each 15-minute interval, summed by hour.

## Weather

- Provider: Open-Meteo Historical Weather API.
- Native unit: hourly weather observations.
- Canonical table: `weather_observations`.
- Regional strategy: one representative city per ODRE region.
- Location mapping: `src/data/france_regions.py`.

Representative cities:

| Region | City |
| --- | --- |
| Auvergne-Rhône-Alpes | Lyon |
| Bourgogne-Franche-Comté | Dijon |
| Bretagne | Rennes |
| Centre-Val de Loire | Orléans |
| Grand Est | Strasbourg |
| Hauts-de-France | Lille |
| Île-de-France | Paris |
| Normandie | Rouen |
| Nouvelle-Aquitaine | Bordeaux |
| Occitanie | Toulouse |
| Pays de la Loire | Nantes |
| Provence-Alpes-Côte d'Azur | Marseille |

## Electricity Prices

- Default provider: Energy-Charts API.
- Market: day-ahead prices.
- France bidding zone: `FR`.
- Canonical unit: EUR/MWh.
- Endpoint: `https://api.energy-charts.info/price?bzn=FR&start=YYYY-MM-DD&end=YYYY-MM-DD`.
- Auth: no API token required.
- Fallback provider: ENTSO-E Transparency Platform.
- Alternative source options are documented in `docs/price_source_options.md`.

## Backfill Window

- Start date: `2023-01-01`.
- End date: `2026-06-30`.
- Electricity date window: 7 days.
- Weather date window: 31 days.
- Price date window: 365 days.

## Implementation Files

- Contracts: `src/data/contracts.py`.
- Date-window planning: `src/data/date_windows.py`.
- Source constants: `src/data/source_config.py`.
- France backfill plan: `src/data/pipelines/france_backfill.py`.
- Weather city mapping: `src/data/france_regions.py`.
- ODRE adapter: `src/data/sources/odre.py`.
- Open-Meteo adapter: `src/data/sources/open_meteo.py`.
- Energy-Charts adapter: `src/data/sources/energy_charts.py`.
- ENTSO-E adapter: `src/data/sources/entsoe.py`.
- Supabase schema: `db/schema.sql`.
