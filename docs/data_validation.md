# Data Validation Workflow

After all three sources are ingested, validate Supabase before building models.

## 1. Apply Feature View

Run `db/feature_views.sql` in the Supabase SQL editor.

This creates:

- `hourly_france_features`

The view joins:

- `hourly_electricity_mix`
- `electricity_prices`
- `weather_observations`

## 2. Run SQL Checks

Run `db/data_quality.sql` in the Supabase SQL editor.

Checks included:

- row counts and timestamp coverage
- duplicate `source_record_id` values
- missing hourly price timestamps
- missing hourly electricity-mix timestamps
- missing hourly weather timestamps
- key null-rate checks

If missing-hour checks fail, run `db/missing_hours_diagnostics.sql` to list the exact timestamps.

If `electricity_prices` contains Energy-Charts timestamps at `:15`, `:30`, or `:45`, follow `docs/price_cleanup.md`.

## 3. Run Python Checks

From the repository root:

```bash
set -a
source .env
set +a
```

Then:

```bash
python3 - <<'PY'
import os
from src.data.load import create_database_engine
from src.data.quality import print_quality_report, run_quality_checks

engine = create_database_engine(os.environ["DATABASE_URL"])
results = run_quality_checks(engine)
print_quality_report(results)

if not all(result.passed for result in results):
    raise SystemExit("one or more data quality checks failed")
PY
```

## Interpretation

- Duplicate checks should return zero rows.
- Missing-hour checks should ideally return zero missing hours.
- Some nulls may be expected for source-specific fields, but nulls in core features should be investigated.
- If `hourly_france_features` has fewer rows than expected, inspect joins by `region` and `timestamp_utc`.

## Expected Modeling Dataset

The primary modeling table/view is:

```sql
select *
from hourly_france_features;
```

Use this view for the first forecasting baseline and causal feature audit.
