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

The full daily refresh skeleton has seven assets:

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
and rebuilds the dashboard JSON. Use `make train-all` when you only want to
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

The scheduled job is `daily_clean_hour_refresh`, configured for `05:00` Europe/Paris time.

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
