# Architecture

The platform combines three analytical layers:

1. **Forecasting** — predict future electricity prices and carbon intensity.
2. **Causal inference** — estimate how interventions affect price, emissions, and workload outcomes.
3. **What-if optimization** — recommend workload shifts under cost, carbon, and operational constraints.

## Data Flow

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

## Initial Components

- **ETL:** scheduled ingestion and normalization.
- **Raw storage:** raw API pages stored before transformation for lineage.
- **Storage:** PostgreSQL-compatible database such as Supabase or Neon.
- **Features:** time, weather, demand, price, generation mix, carbon intensity, and workload features.
- **Models:** forecasting baseline, evaluation, and prediction utilities.
- **Causal:** causal graph, adjustment strategy, effect estimation, and counterfactual scenarios.
- **Optimization:** workload shifting recommendations.
- **Serving:** FastAPI endpoints for forecasts, causal insights, and what-if scenarios.
- **Monitoring:** API, data, model, and optimizer metrics.
