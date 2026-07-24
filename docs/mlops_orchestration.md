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

The daily refresh skeleton has four assets:

- `source_data_snapshot`: validates current local source artifacts.
- `clean_hour_forecast_artifacts`: runs `make forecast-all`.
- `dashboard_data_contract`: runs `make dashboard-data`.
- `frontend_static_build`: runs `make frontend-build`.

Start Dagster locally:

```bash
make dagster-dev
```

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
