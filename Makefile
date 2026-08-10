PYTHON ?= .venv/bin/python
DAGSTER ?= .venv/bin/dagster

.PHONY: help ingest-latest ingest-plan ingest-repair ingest-future ingest-monitor forecast-monitor future-recommendations operational-refresh train-all forecast-all quick-refresh daily-local-refresh forecast-price forecast-ranking forecast-decision forecast-recommendations forecast-scenarios forecast-decision-example forecast-consumption forecast-production forecast-supply-demand forecast-carbon pipeline-health pipeline-health-allow-stale dashboard-data frontend-install frontend-dev frontend-build mlflow-ui dagster-dev docker-build docker-up docker-down docker-observability

help:
	@echo "Forecast training targets:"
	@echo "  make ingest-plan             Show local ingestion date range without API calls"
	@echo "  make ingest-repair           Repair local CSV timestamps and rebuild features"
	@echo "  make ingest-latest           Fetch missing source data and rebuild local features"
	@echo "  make ingest-monitor          Ingest latest data and monitor forecast drift"
	@echo "  make ingest-future           Fetch next-24h future exogenous weather"
	@echo "  make forecast-monitor        Build reports/metrics/forecast_monitoring.json"
	@echo "  make future-recommendations  Build next-24h operational recommendations"
	@echo "  make operational-refresh     Build future recommendations and dashboard"
	@echo "  make forecast-consumption    Train/evaluate consumption baselines only"
	@echo "  make forecast-production     Train/evaluate total + source production baselines"
	@echo "  make forecast-carbon         Calculate carbon outputs from saved source forecasts"
	@echo "  make forecast-supply-demand  Train/evaluate consumption + all production baselines"
	@echo "  make forecast-price          Train/evaluate supply/demand + price baselines"
	@echo "  make forecast-ranking        Build decision rankings from saved price predictions"
	@echo "  make forecast-decision       Build combined price/carbon workload rankings"
	@echo "  make forecast-recommendations  Export top 5 workload shift recommendations"
	@echo "  make forecast-scenarios      Export clean-hour scenario rerankings"
	@echo "  make forecast-decision-example  Example constrained 3-hour workload ranking"
	@echo "  make train-all               Retrain historical forecasts, carbon accounting, and rankings"
	@echo "  make forecast-all            Retrain everything, build future recommendations, and dashboard"
	@echo "  make quick-refresh           Rebuild recommendations, dashboard data, and frontend"
	@echo "  make daily-local-refresh     Ingest latest data, refresh recommendations, and build frontend"
	@echo "  make pipeline-health         Build reports/metrics/pipeline_health.json"
	@echo "  make dashboard-data          Build frontend/public/data/dashboard.json"
	@echo "  make frontend-dev            Start the dashboard dev server"
	@echo "  make frontend-build          Build the Vercel-ready dashboard"
	@echo "  make mlflow-ui               Start local MLflow tracking UI"
	@echo "  make dagster-dev             Start local Dagster webserver"
	@echo "  make docker-up               Start API, dashboard, MLflow, Dagster, Postgres"

ingest-plan:
	$(PYTHON) -m src.data.local_ingest --plan-only

ingest-latest:
	$(PYTHON) -m src.data.local_ingest

ingest-repair:
	$(PYTHON) -m src.data.local_ingest --repair-only

ingest-monitor: ingest-latest pipeline-health forecast-monitor

ingest-future:
	$(PYTHON) -m src.data.future_exogenous --horizon-hours 24

forecast-monitor:
	$(PYTHON) -m src.monitoring.forecast_monitor

future-recommendations:
	$(PYTHON) -m src.models.future_recommendations --horizon-hours 24

operational-refresh: ingest-future future-recommendations dashboard-data frontend-build

train-all:
	$(PYTHON) -m src.models.train_forecast --target all

forecast-all: train-all ingest-future future-recommendations
	$(PYTHON) scripts/build_dashboard_data.py
	npm --prefix frontend run build

quick-refresh:
	$(PYTHON) -m src.models.train_forecast --target decision
	$(PYTHON) scripts/build_dashboard_data.py
	npm --prefix frontend run build

daily-local-refresh: ingest-latest quick-refresh

forecast-price:
	$(PYTHON) -m src.models.train_forecast --target price

forecast-ranking:
	$(PYTHON) -m src.models.train_forecast --target ranking

forecast-decision:
	$(PYTHON) -m src.models.train_forecast --target decision

forecast-recommendations:
	$(PYTHON) -m src.models.train_forecast --target decision --top-n-recommendations 5
	$(PYTHON) scripts/build_dashboard_data.py

forecast-scenarios:
	$(PYTHON) -m src.models.train_forecast --target decision --top-n-recommendations 5
	$(PYTHON) scripts/build_dashboard_data.py

forecast-decision-example:
	$(PYTHON) -m src.models.train_forecast --target decision --duration-hours 3 --earliest-start-utc 2026-04-01T08:00:00+00:00 --latest-end-utc 2026-04-01T22:00:00+00:00 --price-weight 0.5 --carbon-weight 0.5

forecast-consumption:
	$(PYTHON) -m src.models.train_forecast --target consumption

forecast-production:
	$(PYTHON) -m src.models.train_forecast --target production

forecast-supply-demand:
	$(PYTHON) -m src.models.train_forecast --target supply-demand

forecast-carbon:
	$(PYTHON) -m src.models.train_forecast --target carbon

dashboard-data:
	$(PYTHON) scripts/build_dashboard_data.py

pipeline-health:
	$(PYTHON) -m src.data.pipeline_health

pipeline-health-allow-stale:
	$(PYTHON) -m src.data.pipeline_health --allow-stale

frontend-install:
	npm --prefix frontend install

frontend-dev:
	npm --prefix frontend run dev -- --host 127.0.0.1

frontend-build:
	npm --prefix frontend run build

mlflow-ui:
	mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

dagster-dev:
	mkdir -p "$${DAGSTER_HOME:-$(CURDIR)/.dagster}"
	test -f "$${DAGSTER_HOME:-$(CURDIR)/.dagster}/dagster.yaml" || touch "$${DAGSTER_HOME:-$(CURDIR)/.dagster}/dagster.yaml"
	DAGSTER_HOME="$${DAGSTER_HOME:-$(CURDIR)/.dagster}" $(DAGSTER) dev --workspace workspace.yaml

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up

docker-down:
	docker compose -f docker/docker-compose.yml down

docker-observability:
	docker compose -f docker/docker-compose.yml --profile observability up
