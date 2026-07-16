PYTHON ?= python

.PHONY: help forecast-all forecast-price forecast-ranking forecast-decision forecast-recommendations forecast-decision-example forecast-consumption forecast-production forecast-supply-demand forecast-carbon

help:
	@echo "Forecast training targets:"
	@echo "  make forecast-consumption    Train/evaluate consumption baselines only"
	@echo "  make forecast-production     Train/evaluate total + source production baselines"
	@echo "  make forecast-carbon         Calculate carbon outputs from saved source forecasts"
	@echo "  make forecast-supply-demand  Train/evaluate consumption + all production baselines"
	@echo "  make forecast-price          Train/evaluate supply/demand + price baselines"
	@echo "  make forecast-ranking        Build decision rankings from saved price predictions"
	@echo "  make forecast-decision       Build combined price/carbon workload rankings"
	@echo "  make forecast-recommendations  Export top 5 workload shift recommendations"
	@echo "  make forecast-decision-example  Example constrained 3-hour workload ranking"
	@echo "  make forecast-all            Run forecasts, carbon accounting, and decision rankings"

forecast-all:
	$(PYTHON) -m src.models.train_forecast --target all

forecast-price:
	$(PYTHON) -m src.models.train_forecast --target price

forecast-ranking:
	$(PYTHON) -m src.models.train_forecast --target ranking

forecast-decision:
	$(PYTHON) -m src.models.train_forecast --target decision

forecast-recommendations:
	$(PYTHON) -m src.models.train_forecast --target decision --top-n-recommendations 5

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
