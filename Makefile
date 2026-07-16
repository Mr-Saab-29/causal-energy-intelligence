PYTHON ?= python

.PHONY: help forecast-all forecast-price forecast-ranking forecast-consumption forecast-production forecast-supply-demand

help:
	@echo "Forecast training targets:"
	@echo "  make forecast-consumption    Train/evaluate consumption baselines only"
	@echo "  make forecast-production     Train/evaluate total + source production baselines"
	@echo "  make forecast-supply-demand  Train/evaluate consumption + all production baselines"
	@echo "  make forecast-price          Train/evaluate supply/demand + price baselines"
	@echo "  make forecast-ranking        Build decision rankings from saved price predictions"
	@echo "  make forecast-all            Train price plus source-level production baselines"

forecast-all:
	$(PYTHON) -m src.models.train_forecast --target all

forecast-price:
	$(PYTHON) -m src.models.train_forecast --target price

forecast-ranking:
	$(PYTHON) -m src.models.train_forecast --target ranking

forecast-consumption:
	$(PYTHON) -m src.models.train_forecast --target consumption

forecast-production:
	$(PYTHON) -m src.models.train_forecast --target production

forecast-supply-demand:
	$(PYTHON) -m src.models.train_forecast --target supply-demand
