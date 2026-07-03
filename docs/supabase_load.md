# Supabase Load Guide

## 1. Apply Schema

Apply `db/schema.sql` in the Supabase SQL editor before running loaders.

Required tables:

- `raw_api_pages`
- `electricity_prices`
- `hourly_electricity_mix`
- `weather_observations`

## 2. Configure Environment

Set `DATABASE_URL` to the Supabase Postgres connection string.

Example format:

```bash
export DATABASE_URL="postgresql://postgres:<password>@db.<project-ref>.supabase.co:5432/postgres"
```

Use the direct Postgres connection string for batch loading. If your Supabase project requires pooled connections, use the pooler connection string from Supabase settings.

## 3. Load Price Data

Energy-Charts is the default France day-ahead price source.

```bash
python - <<'PY'
import os
from src.data.load import create_database_engine
from src.data.pipelines.supabase_load import load_france_price_history

engine = create_database_engine(os.environ["DATABASE_URL"])
rows = load_france_price_history(engine)
print(f"upserted electricity price rows: {rows}")
PY
```

Price loading uses calendar-year windows:

- `2023-01-01` to `2023-12-31`
- `2024-01-01` to `2024-12-31`
- `2025-01-01` to `2025-12-31`
- `2026-01-01` to `2026-06-30`

If Energy-Charts returns `429 Too Many Requests`, wait several minutes and rerun the same command. The upsert is idempotent, so previously loaded rows are updated rather than duplicated.

## 4. Load Electricity Mix

ODRE historical national and regional data is converted from 15-minute MW to hourly MWh.

```bash
python - <<'PY'
import os
from src.data.load import create_database_engine
from src.data.pipelines.supabase_load import load_france_electricity_mix_history

engine = create_database_engine(os.environ["DATABASE_URL"])
rows = load_france_electricity_mix_history(engine)
print(f"upserted hourly electricity mix rows: {rows}")
PY
```

Electricity mix loading writes each 7-day window to Supabase immediately. This is slower, but safer: if the run fails, rerun the same command and already-loaded windows will be upserted.

## 5. Load Weather

Open-Meteo weather is loaded for one representative city per ODRE region.

```bash
python - <<'PY'
import os
from src.data.load import create_database_engine
from src.data.pipelines.supabase_load import load_france_weather_history

engine = create_database_engine(os.environ["DATABASE_URL"])
rows = load_france_weather_history(engine)
print(f"upserted weather rows: {rows}")
PY
```

Weather loading writes one region/date-window at a time. The current mapping uses one representative city per ODRE region.

## 6. Full Historical Load

Use this only after individual source loads work.

```bash
python - <<'PY'
import os
from src.data.pipelines.supabase_load import load_france_history_to_supabase

summary = load_france_history_to_supabase(os.environ["DATABASE_URL"])
print(summary)
PY
```

## Upsert Behavior

All canonical loaders use `ON CONFLICT (source, source_record_id) DO UPDATE`.

This makes the load idempotent as long as each adapter emits stable `source_record_id` values.

The loaders are designed to be safely rerunnable after API timeouts, rate limits, or local terminal interruptions.

## Recommended Order

1. Apply `db/schema.sql`.
2. Load Energy-Charts prices.
3. Load ODRE electricity mix.
4. Load Open-Meteo weather.
5. Query row counts and timestamp coverage in Supabase.
