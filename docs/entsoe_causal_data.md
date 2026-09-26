# ENTSO-E Causal Grid Data

This data layer supplies the grid conditions needed to move beyond the current marginal-emissions
proxy. It does not, by itself, turn the recommendation into an identified causal estimate.

## Scope

- Primary bidding zone: France (`FR`).
- Direct neighboring zones: Belgium, Germany-Luxembourg, Switzerland, Northern Italy, Spain, and
  Great Britain.
- History begins on `2023-01-01`.
- Modeling resolution is hourly. Balancing observations retain their native resolution, commonly
  15 minutes, and are aggregated only in the analytical view.
- Carbon boundary remains direct operational emissions.

The versioned source contract is `config/entsoe_causal_data.json`.

## Datasets And Causal Roles

| Data | Stored measure | Causal role |
| --- | --- | --- |
| Day-ahead capacity | Directional MW | Pre-treatment when published before the decision |
| Scheduled exchange | Directional MW | Pre-treatment when published before the decision |
| Physical flow | Directional MW | Post-treatment mediator or outcome diagnostic |
| Load, wind, and solar forecasts | Forecast MW plus snapshot | Pre-treatment controls |
| Planned outages | Versioned unavailable MW | Pre-treatment when known before the decision |
| Unplanned outages | Versioned unavailable MW | Control only if published before the decision; otherwise a response-period shock |
| Balancing activation and imbalance | Native-resolution value and direction | Post-treatment mediator or stress diagnostic |
| Lagged forecast error | Settled actual minus forecast | Pre-treatment state variable |
| Same-period forecast error | Settled actual minus forecast | Response-period shock, not an adjustment variable |

This timing distinction matters. Controlling for physical flows or balancing activation while
estimating the total workload effect would remove part of the grid response that the estimand is
trying to measure.

Realized balancing queries stop at the retrieval hour rather than requesting future values. The
legacy activated-energy publication (`A83`) is treated as optional because ENTSO-E is transitioning
that publication to aggregated balancing-energy bids under GL EB 12.3.E. Imbalance volumes, prices,
and activated-energy prices remain independently ingested; an unavailable legacy series is reported
as a warning instead of aborting the entire bounded ingestion.

Day-ahead transfer capacity is also advisory because ENTSO-E does not publish it consistently for
every France-neighbor direction. Missing-data responses and transient upstream HTTP failures are
recorded as warnings for this series only. Required physical flows, scheduled exchanges, forecasts,
and balancing-series failures still stop the backfill.

## Forecast Vintage Limitation

Operational ingestion stores the retrieval hour in `snapshot_at_utc` and labels rows
`operational_snapshot`. Those records can support point-in-time evaluation.

Historical ENTSO-E downloads are labeled `historical_final`. They may contain the final published
version rather than the exact forecast visible at a past decision time. They are suitable for
feature exploration and sensitivity analysis, but strict causal evaluation and leakage claims must
use `operational_snapshot` rows.

## Storage And Memory

Apply `db/causal_grid_data.sql` after the existing Supabase schema. The migration creates:

- `cross_border_observations`
- `grid_forecasts`
- `generation_outages`
- `balancing_observations`
- Hourly analytical views for forecast error, net imports, and balancing

Apply `db/causal_grid_compact.sql` for the bounded analytical layer. It creates one hourly France
feature table and one hourly France-neighbor table. Native-resolution staging rows are exported as
monthly Zstandard-compressed Parquet files to a private Supabase Storage bucket. This preserves the
auditable source data without consuming the 500 MB Postgres allowance.

Raw API responses are not stored. Stable source IDs make realized observations idempotent. Forecast
and day-ahead records preserve hourly ingestion snapshots, while unchanged outage revisions are
stored once per vintage class. Operational forecasts, schedules, and capacities are retained only
for target hours at or after their retrieval time, rather than repeatedly copying already-settled
history. This keeps the database bounded while retaining the versions needed for point-in-time
analysis.

## Commands

Inspect the default bounded window without credentials or database writes:

```bash
make ingest-causal-grid-plan
```

After the ENTSO-E token is active and the migration has been applied:

```bash
make ingest-causal-grid
```

Run a historical window explicitly:

```bash
python -m src.data.entsoe_causal_ingest \
  --start-date 2023-01-01 \
  --end-date 2023-01-31 \
  --historical-backfill
```

Historical backfills should be run in bounded windows. The command is intentionally not part of the
daily GitHub workflow until the token is active and a live extraction has been validated.

## Quality Gate

Audit the previous complete UTC calendar month before extending the backfill:

```bash
make causal-data-quality
```

The report is written to `reports/metrics/causal_data_quality.json`. It checks forecast,
cross-border, and balancing coverage at the hourly decision resolution while retaining native raw
intervals for diagnostics; source-key uniqueness;
outage classification; and point-in-time temporal rules. Known limitations such as unavailable
legacy `A83` activated energy and structurally unpublished day-ahead capacity pairs are reported
separately and do not block backfill. It also projects storage for the complete historical horizon
from the audited month and blocks raw-table backfills above a conservative 250 MiB budget. This
leaves headroom inside the 500 MB database allowance for existing product tables and indexes. A
report is backfill-ready only when
`backfill_ready` is true.

Use explicit dates to audit any completed monthly window:

```bash
python -m src.data.causal_data_quality \
  --start-date 2026-08-01 \
  --end-date 2026-09-01 \
  --vintage-quality historical_final
```

## Archive And Compaction

The safe monthly sequence is: validate the complete month, write Parquet, compact hourly features,
upload to the private `causal-grid-archive` bucket, download each object to verify its SHA-256 hash,
and only then permit raw-row deletion. The ordinary target never deletes staging data:

```bash
make causal-archive-compact
```

Configure `SUPABASE_PROJECT_URL`, `SUPABASE_SERVICE_ROLE_KEY`, and optionally
`CAUSAL_ARCHIVE_BUCKET`. The service-role key is a backend secret; do not commit it or expose it to
the frontend. For a local archive and compaction test without Storage access, run:

```bash
python -m src.data.causal_archive_compact \
  --start-date 2026-08-01 \
  --end-date 2026-09-01 \
  --apply-schema \
  --skip-upload
```

After inspecting the manifest and compact row counts, the destructive command may be run for that
month. It refuses to purge unless the upload and checksum verification complete in the same run:

```bash
python -m src.data.causal_archive_compact \
  --start-date 2026-08-01 \
  --end-date 2026-09-01 \
  --apply-schema \
  --purge-raw
```

Only `historical_final` staging rows can be purged. Operational snapshots remain in Postgres for
point-in-time evaluation. Local archives are written below `data/archive/entsoe/` and remain
gitignored.

## Resumable Historical Runner

Run the complete history with one command:

```bash
make causal-backfill-history
```

By default, the runner works backward from the previous complete UTC month through January 2023.
For every month it ingests, runs the quality gate, writes and verifies the Storage archive, compacts
the analytical features, purges verified historical staging rows, and checks the durable final
state before advancing. July and August 2026, or any other completed months, are skipped only when
the Storage manifest exists, compact rows are complete, and no raw historical rows remain.

If a request or quality check fails, the runner stops immediately. Fix the issue and rerun the same
command; completed months are skipped and the failed month is safely retried. Progress is persisted
after each month in `reports/metrics/causal_historical_backfill.json`. Keep the terminal and network
connection active while it runs.

To run a smaller inclusive range, use month values in `YYYY-MM` format:

```bash
python -m src.data.causal_historical_backfill \
  --from-month 2026-01 \
  --through-month 2026-06
```
