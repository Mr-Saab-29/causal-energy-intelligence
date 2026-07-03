"""Backfill plan for France electricity, weather, and price data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.data.date_windows import DateWindow, iter_date_windows
from src.data.france_regions import RepresentativeLocation, get_regional_weather_locations
from src.data.source_config import FRANCE_END_DATE, FRANCE_START_DATE
from src.data.sources.entsoe import ENTSOE_WINDOW_DAYS
from src.data.sources.energy_charts import ENERGY_CHARTS_WINDOW_DAYS
from src.data.sources.odre import ODRE_WINDOW_DAYS
from src.data.sources.open_meteo import OPEN_METEO_WINDOW_DAYS


@dataclass(frozen=True)
class FranceBackfillPlan:
    """Date windows for each source in the France historical backfill."""

    electricity_windows: list[DateWindow]
    weather_windows: list[DateWindow]
    price_windows: list[DateWindow]
    weather_locations: list[RepresentativeLocation]


def build_france_backfill_plan(
    start_date: date = FRANCE_START_DATE,
    end_date: date = FRANCE_END_DATE,
) -> FranceBackfillPlan:
    """Create bounded date-window extraction plans for all France sources."""
    return FranceBackfillPlan(
        electricity_windows=list(iter_date_windows(start_date, end_date, ODRE_WINDOW_DAYS)),
        weather_windows=list(iter_date_windows(start_date, end_date, OPEN_METEO_WINDOW_DAYS)),
        price_windows=list(iter_date_windows(start_date, end_date, ENERGY_CHARTS_WINDOW_DAYS)),
        weather_locations=get_regional_weather_locations(),
    )


def build_france_entsoe_backfill_plan(
    start_date: date = FRANCE_START_DATE,
    end_date: date = FRANCE_END_DATE,
) -> FranceBackfillPlan:
    """Create a France backfill plan using ENTSO-E price request limits."""
    return FranceBackfillPlan(
        electricity_windows=list(iter_date_windows(start_date, end_date, ODRE_WINDOW_DAYS)),
        weather_windows=list(iter_date_windows(start_date, end_date, OPEN_METEO_WINDOW_DAYS)),
        price_windows=list(iter_date_windows(start_date, end_date, ENTSOE_WINDOW_DAYS)),
        weather_locations=get_regional_weather_locations(),
    )
