# Energy-Charts Price Cleanup

Energy-Charts can return sub-hourly price points for parts of the France series. The adapter now aggregates all points within each UTC hour into one hourly EUR/MWh value before upsert.

If prices were loaded before this fix, remove old Energy-Charts rows and rerun the price loader.

## Cleanup SQL

Run this in Supabase SQL editor:

```sql
delete from electricity_prices
where source_record_id like 'energy-charts:%';

delete from ingestion_checkpoints
where source_name = 'energy_charts'
  and dataset_name = 'day_ahead_prices';
```

## Reload Prices

From the repository root:

```bash
set -a
source .env
set +a

python3 - <<'PY'
import os
from src.data.load import create_database_engine
from src.data.pipelines.supabase_load import load_france_price_history

engine = create_database_engine(os.environ["DATABASE_URL"])
rows = load_france_price_history(engine)
print(f"upserted electricity price rows: {rows}")
PY
```

## Validate

Run:

```sql
select
    count(*) as row_count,
    min(timestamp_utc) as min_timestamp_utc,
    max(timestamp_utc) as max_timestamp_utc,
    count(*) filter (
        where extract(minute from timestamp_utc) != 0
           or extract(second from timestamp_utc) != 0
    ) as non_hourly_rows
from electricity_prices
where source_record_id like 'energy-charts:%';
```

Expected:

- `non_hourly_rows = 0`

