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

Historical imbalance-price, imbalance-volume, and activation-price publications can contain source
gaps. Because these are post-treatment stress diagnostics rather than adjustment controls, limited
historical coverage is retained as a warning and missing values remain null. The same gaps are
blocking for operational snapshots, where they indicate a current ingestion or publication issue.

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

ENTSO-E does not consistently publish France offshore-wind forecasts across the earliest historical
period. Historical offshore coverage below the normal threshold is retained as an explicit warning
and missing values remain null; they are never interpreted as zero generation. Load, onshore wind,
and solar remain required historical controls. Offshore wind remains required for operational
snapshots, where a gap indicates a current pipeline problem.

## Storage And Memory

Apply `db/causal_grid_data.sql` after the existing Supabase schema. The migration creates:

- `cross_border_observations`
- `grid_forecasts`
- `generation_outages`
- `balancing_observations`
- Hourly analytical views for forecast error, net imports, and balancing

Apply `db/causal_grid_compact.sql` for the bounded analytical layer. It creates one hourly France
feature table, one hourly France-neighbor flow table, and one compact neighboring-zone emissions
table. Native-resolution staging rows are exported as
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

## Causal Feature Mart

After the checked historical backfill is complete, build the analysis-ready hourly mart:

```bash
make causal-feature-mart
```

The contract is versioned in `config/causal_feature_mart.json`. The command joins the compact
France and cross-border tables to settled national electricity actuals, day-ahead price, and
available weather observations. It writes:

- `reports/causal/causal_hourly_feature_mart.parquet`
- `reports/metrics/causal_feature_readiness.json`

Columns use causal-role prefixes. `pre_` columns are the only candidates for an adjustment set;
`treatment_proxy_` identifies the observational load-innovation proxy; `outcome_` contains France
direct operational emissions calculated from settled source generation and the versioned factors
in `config/emission_factors.yaml`; and `mediator_` plus `diagnostic_` columns are excluded from total
effect adjustment. The published ODRE carbon intensity is retained as a diagnostic, not substituted
for the declared direct-generation outcome. Forecast errors enter the adjustment set only as
one-hour and 24-hour lags. Same-hour forecast errors, physical flows, balancing, realized dispatch,
and unplanned outages are post-treatment diagnostics or mediators.

The readiness report deliberately separates two questions. `feature_mart_ready` means the hourly
artifact passes coverage and uniqueness checks. `identified_estimator_ready` remains false until
the project has point-in-time historical ENTSO-E forecast vintages and observed workload
interventions. Archived weather forecasts, fuel and carbon prices, storage controls, and connected-
zone emissions are now covered by separate leakage-safe contracts. A ready mart is therefore
permission to begin descriptive proxy analysis, not permission to make a causal savings claim.

## Pre-Treatment Covariates

Apply `db/causal_grid_compact.sql`, then backfill the compact hourly control table:

```bash
make pre-treatment-backfill
```

The versioned contract is `config/causal_pre_treatment.json`. It uses an archived GFS temperature
forecast at a fixed 24-hour lead for 12 France weather locations, free World Bank monthly European
TTF natural-gas and Australian-coal benchmark prices as regime controls, completed EEX EU ETS primary
auction prices, and France aggregate hydro storage from ENTSO-E. Fuel months are shifted by two
months, auctions are joined strictly after completion, and storage observations are delayed seven
days. These conservative availability rules prevent later information from leaking into a past
decision hour.

The command writes `reports/metrics/causal_pre_treatment_readiness.json` and refuses to publish a
window below the configured 95 percent core coverage or with any timestamp-ordering violation.
The daily workflow runs `make ingest-pre-treatment` for a bounded recent refresh.

## Interconnected Emissions Boundary

The estimand covers France plus Belgium, Germany-Luxembourg, Switzerland, Northern Italy, Spain,
and Great Britain. Continental actual generation per production type comes from the ENTSO-E A75
publication. ENTSO-E does not return the required Great Britain series, so Great Britain uses the
official Elexon Insights `FUELHH` dataset. Interconnector categories in FUELHH are excluded because
the outcome counts each zone's domestic generation; including imports again would double-count
generation already attributed to its producing zone.

Elexon FUELHH is based on operationally metered generation and underrepresents embedded renewable
generation. This limitation is recorded in the readiness report. It mainly affects reported total
generation and intensity; the direct-emissions outcome still retains wide bounds for heterogeneous
categories and must not be presented as a perfectly measured system total.

The versioned mapping and quality policy are in `config/neighbor_emissions.json`. Named production
types use the direct-operational factors in `config/emission_factors.yaml`. Heterogeneous `Other`
generation receives a point factor of 370 kgCO2e/MWh and a deliberately wide 0-820 bound. The
compact table retains point, lower, and upper emissions and the share covered by named factors.
Unknown source categories fail closed. A month passes only when every zone has at least 95 percent
hourly coverage and at least 85 percent of generation uses named factors.

Apply the updated `db/causal_grid_compact.sql`, then validate a bounded operational refresh:

```bash
make ingest-neighbor-emissions
```

Backfill the agreed history with the resumable storage-safe runner:

```bash
make neighbor-emissions-backfill
```

The runner fetches one month at a time, writes normalized generation-by-type Parquet, uploads it to
`neighbor-emissions/historical_final/YYYY-MM/` in the existing private archive bucket, verifies the
downloaded checksum, and upserts only six compact hourly rows per timestamp. This avoids consuming
the 500 MB Postgres allowance with source-level history. The mart then exposes France, neighboring,
and interconnected direct-emissions outcomes and refuses to mark the boundary complete unless all
six zones pass coverage and factor-quality checks.
