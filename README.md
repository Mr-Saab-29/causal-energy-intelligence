# Causal Energy Intelligence Platform

Forecasting, causal inference, and what-if optimization for electricity price forecasting and carbon-aware workload shifting.

## Architecture

```text
External APIs / CSV sources
  -> Airflow ETL
  -> Cloud Postgres
  -> Feature Engineering
  -> Forecasting Model
  -> Causal Inference Engine
  -> What-if Simulator
  -> FastAPI
  -> Docker
  -> Monitoring
  -> Optional local Kubernetes
```

## Repository Layout

- `dags/` — Airflow ETL orchestration.
- `src/data/` — Extract, transform, and load utilities.
- `src/features/` — Feature engineering.
- `src/models/` — Forecast training, evaluation, and prediction.
- `src/causal/` — Causal DAGs, effect estimation, and counterfactuals.
- `src/optimization/` — Carbon-aware workload shifting.
- `src/monitoring/` — Metrics and observability helpers.
- `api/` — FastAPI application.
- `notebooks/` — EDA, forecasting, and causal analysis notebooks.
- `docker/` — Container and local compose setup.
- `k8s/` — Optional local Kubernetes manifests.
- `docs/` — Architecture, causal DAG, and project report notes.
- `db/` — Supabase/Postgres schema.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

## Current Status

This repository currently contains the platform scaffold. Implementation will proceed module by module: ETL contracts, feature schema, forecasting baseline, causal DAG, counterfactual simulation, optimization, API endpoints, and monitoring.

## Data Contracts

Canonical contracts are defined in `src/data/contracts.py` and documented in `docs/data_contracts.md`. Apply the initial Supabase/Postgres schema from `db/schema.sql`.

Source-specific extraction notes are documented in `docs/data_sources.md`.

France electricity spot-price provider tradeoffs are documented in `docs/price_source_options.md`. The default spot-price source is Energy-Charts, with ENTSO-E kept as a fallback.

Supabase loading instructions are documented in `docs/supabase_load.md`.
