# MLOps and Daily Refresh

This project now has a lightweight MLOps path for the clean-hour dashboard MVP.

## MLflow

Forecast and recommendation runs log summary metrics and report artifacts to MLflow from
`src.models.train_forecast`.

Default local tracking:

```bash
make mlflow-ui
```

Then in another terminal:

```bash
make forecast-all
```

Default environment variables:

- `MLFLOW_TRACKING_ENABLED=true`
- `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`
- `MLFLOW_EXPERIMENT_NAME=clean-hour-scheduling`

In Docker Compose, services use:

```text
MLFLOW_TRACKING_URI=http://mlflow:5000
```

Use `--disable-mlflow` on the forecast CLI when you want a run without experiment tracking:

```bash
python -m src.models.train_forecast --target decision --disable-mlflow
```

## Dagster

Dagster definitions live in `orchestration/definitions.py`.

The scheduled ingestion-monitor refresh runs at `02:00` Europe/Paris and has
three assets:

- `ingest_latest_source_data`: fetches missing source data and rebuilds modeling features.
- `source_data_snapshot`: validates current local source artifacts.
- `forecast_monitoring_report`: compares settled forecasts with actuals and flags retraining needs.

The equivalent local command is:

```bash
make ingest-monitor
```

This path does not retrain models. It writes:

- `reports/metrics/pipeline_health.json`
- `reports/metrics/forecast_monitoring.json`

## Deployable Free Schedule

The deployable no-cost ingestion monitor lives at:

```text
.github/workflows/daily-ingestion-monitor.yml
```

It runs on GitHub Actions using:

```text
cron: 17 0 * * *
```

GitHub cron is UTC-only, so `00:17` UTC lands at `02:17` Europe/Paris during
summer time and `01:17` Europe/Paris during winter time. The guard allows
scheduled runs that start inside the `00:00-06:00` Europe/Paris window, so a
queued run can start late without being skipped as long as it is still inside
the overnight window. The workflow can also be run manually from the GitHub
Actions UI.

The workflow runs as chained jobs:

- `schedule-guard`: skips scheduled runs that start outside the `00:00-06:00` Europe/Paris window
- `ingest`: runs `make ingest-latest-cloud` and `make ingest-future-cloud`; this job has a 45-minute timeout because upstream APIs and Supabase writes can exceed the normal 15-minute fast-path during slow refreshes
- `preflight-monitor`: restores current operational state, runs health/forecast monitors, and decides whether retraining is needed
- `retrain`: runs `make train-all-gated` only when the preflight decision requests retraining
- `publish-dashboard`: runs `make operational-publish`, saves the refreshed operational cache, and deploys the dashboard

This cloud variant caps historical API ingestion to a recent 14-day lookback so
an empty GitHub Actions cache cannot accidentally trigger a full 2023-to-present
backfill. Local `make ingest-monitor` still uses the normal incremental local
refresh logic.

After ingestion and monitoring, the GitHub Actions workflow decides whether to
run a gated retrain:

- retrain when `reports/metrics/forecast_monitoring.json` has
  `retraining_recommended = true`
- retrain when required cached model artifacts are missing
- otherwise reuse the current champion artifacts and only refresh
  next-24-hour recommendations

The workflow then builds `frontend/public/data/dashboard.json` and deploys the
prebuilt `frontend/dist` output to Vercel with the Vercel CLI. This keeps
generated dashboard data out of git while still publishing fresh recommendations
after each successful scheduled run. Because retraining and publishing are
separate jobs, GitHub Actions can rerun a failed publish job without repeating a
completed retrain from the same workflow run.

Required GitHub repository secrets:

- `SUPABASE_DATABASE_URL`
- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

It restores and saves a cache for operational model/report state:

- `models`
- `reports/metrics`
- `reports/predictions`
- `reports/rankings`
- `reports/recommendations`
- `reports/scenarios`
- `reports/carbon`
- `reports/monitoring`

It uploads these operational outputs as workflow artifacts for inspection:

- `reports/metrics/pipeline_health.json`
- `reports/metrics/forecast_monitoring.json`
- `reports/metrics/model_promotion_decision.json`
- `reports/metrics/future_recommendation_metadata.json`
- `reports/metrics/marginal_ranking_shift_metrics.json`
- `reports/rankings/future_marginal_workload_decision_rankings.csv`
- `reports/recommendations/future_champion_workload_recommendations.csv`
- `reports/recommendations/future_causal_adjusted_workload_recommendations.csv`
- `reports/scenarios/future_workload_scenario_recommendations.csv`
- `frontend/public/data/dashboard.json`

If the cache is empty on the first cloud run, the workflow retrains because
required model artifacts are missing. Later runs retrain only when monitoring
recommends it.

Monitoring trigger thresholds are configured in:

```text
config/monitoring_thresholds.yaml
```

Tune this file to adjust the recent monitoring window, degradation ratios,
top-5 hit-rate drop threshold, source-production sMAPE threshold, and minimum
settled operational rows.

The full manual retraining refresh has seven assets:

- `ingest_latest_source_data`: fetches missing source data and rebuilds modeling features.
- `source_data_snapshot`: validates current local source artifacts.
- `clean_hour_forecast_artifacts`: runs `make train-all`.
- `future_exogenous_data`: runs `make ingest-future`.
- `future_recommendation_artifacts`: runs `make future-recommendations`.
- `dashboard_data_contract`: runs `make dashboard-data`.
- `frontend_static_build`: runs `make frontend-build`.

The faster development refresh is available as the Dagster job
`quick_recommendation_refresh`. It uses existing trained model artifacts and runs:

- `ingest_latest_source_data`
- `source_data_snapshot`
- `quick_future_exogenous_data`: runs `make ingest-future`.
- `quick_future_recommendation_artifacts`: runs `make future-recommendations`.
- `quick_dashboard_data_contract`: runs `make dashboard-data`.
- `quick_frontend_static_build`: runs `make frontend-build`.

The equivalent local command is:

```bash
make quick-refresh
```

For live local data, run ingestion before refreshing recommendations:

```bash
make ingest-plan
make ingest-latest
make quick-refresh
```

Or run the combined path:

```bash
make daily-local-refresh
```

For the operational dashboard, build next-24-hour forward-looking recommendations
from existing trained artifacts:

```bash
make operational-refresh
```

This fetches future exogenous weather, scores the next 24 hours with saved model
artifacts, writes `reports/recommendations/future_champion_workload_recommendations.csv`,
adds causal-adjusted MVP recommendations from the marginal-emissions proxy, and
rebuilds the dashboard JSON. Use `make train-all` when you only want to
refresh historical validation artifacts and saved model artifacts.

To retrain the entire project and refresh the future dashboard output in one command:

```bash
make forecast-all
```

This runs `make train-all`, fetches future exogenous data, builds future
recommendations, rebuilds the dashboard data contract, and builds the frontend.

`make ingest-latest` infers the refresh range from the latest timestamps in
`data/processed/electricity_prices.csv`, `data/processed/hourly_electricity_mix.csv`,
and `data/processed/weather_observations.csv`, then fetches through today's UTC date.

Pipeline health is written to:

```text
reports/metrics/pipeline_health.json
```

The report checks file existence, row counts, latest timestamps, duplicate
timestamps, missing hourly timestamps, required columns, null counts, negative
production/consumption values, dashboard JSON presence, and recommendation count.
By default, source data older than two days is a critical failure. Use
`make pipeline-health-allow-stale` only when inspecting historical demo data.

Start Dagster locally:

```bash
make dagster-dev
```

The Make target creates `.dagster` and passes it to Dagster as an absolute path.
If you override `DAGSTER_HOME`, use an existing absolute path.

Open:

```text
http://127.0.0.1:3000
```

The scheduled job is `ingestion_monitor_refresh`, configured for `02:00` Europe/Paris time.

## Docker Compose

Start the local service stack:

```bash
make docker-up
```

Services:

- API: `http://127.0.0.1:8000`
- Dashboard: `http://127.0.0.1:3000`
- Dagster: `http://127.0.0.1:3001`
- MLflow: `http://127.0.0.1:5000`
- Postgres: `127.0.0.1:5432`

Optional observability stack:

```bash
make docker-observability
```

- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3002`

## Intended Evolution

Current daily refresh uses existing local/generated artifacts. The next production step is to replace
`source_data_snapshot` with source-specific Dagster assets:

- fetch latest price data
- fetch latest electricity mix data
- fetch latest weather data
- validate freshness and missing hours
- rebuild modeling features
- run forecasts and recommendation export
- publish `frontend/public/data/dashboard.json`
