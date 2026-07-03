"""Source-specific constants for the initial France data ingestion scope."""

from __future__ import annotations

from datetime import date

FRANCE_START_DATE = date(2023, 1, 1)
FRANCE_END_DATE = date(2026, 6, 30)

FRANCE_ENTSOE_BIDDING_ZONE = "10YFR-RTE------C"

ODRE_BASE_URL = "https://odre.opendatasoft.com"
ODRE_NATIONAL_DATASET = "eco2mix-national-tr"
ODRE_REGIONAL_DATASET = "eco2mix-regional-tr"
ODRE_NATIONAL_HISTORICAL_DATASET = "eco2mix-national-cons-def"
ODRE_REGIONAL_HISTORICAL_DATASET = "eco2mix-regional-cons-def"

OPEN_METEO_ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com"
OPEN_METEO_HISTORICAL_ENDPOINT = "/v1/archive"
OPEN_METEO_HOURLY_FEATURES = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_direction_10m",
    "wind_direction_80m",
    "wind_gusts_10m",
    "surface_pressure",
    "weather_code",
]

ENTSOE_BASE_URL = "https://web-api.tp.entsoe.eu"
ENTSOE_ENDPOINT = "/api"
ENTSOE_DAY_AHEAD_PRICES_DOCUMENT_TYPE = "A44"
ENTSOE_MARKET_AGREEMENT_TYPE = "A01"

ENERGY_CHARTS_BASE_URL = "https://api.energy-charts.info"
ENERGY_CHARTS_PRICE_ENDPOINT = "/price"
ENERGY_CHARTS_FRANCE_BIDDING_ZONE = "FR"
