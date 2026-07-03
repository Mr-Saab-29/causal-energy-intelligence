"""Representative weather locations for France regional electricity data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepresentativeLocation:
    """Representative city and coordinates for a French region."""

    region: str
    city: str
    latitude: float
    longitude: float


FRANCE_REGIONAL_WEATHER_LOCATIONS = [
    RepresentativeLocation("Auvergne-Rhône-Alpes", "Lyon", 45.7640, 4.8357),
    RepresentativeLocation("Bourgogne-Franche-Comté", "Dijon", 47.3220, 5.0415),
    RepresentativeLocation("Bretagne", "Rennes", 48.1173, -1.6778),
    RepresentativeLocation("Centre-Val de Loire", "Orléans", 47.9029, 1.9093),
    RepresentativeLocation("Grand Est", "Strasbourg", 48.5734, 7.7521),
    RepresentativeLocation("Hauts-de-France", "Lille", 50.6292, 3.0573),
    RepresentativeLocation("Île-de-France", "Paris", 48.8566, 2.3522),
    RepresentativeLocation("Normandie", "Rouen", 49.4431, 1.0993),
    RepresentativeLocation("Nouvelle-Aquitaine", "Bordeaux", 44.8378, -0.5792),
    RepresentativeLocation("Occitanie", "Toulouse", 43.6047, 1.4442),
    RepresentativeLocation("Pays de la Loire", "Nantes", 47.2184, -1.5536),
    RepresentativeLocation("Provence-Alpes-Côte d'Azur", "Marseille", 43.2965, 5.3698),
]

FRANCE_NATIONAL_WEATHER_LOCATION = RepresentativeLocation("FR", "Paris", 48.8566, 2.3522)


def get_regional_weather_locations() -> list[RepresentativeLocation]:
    """Return representative city coordinates keyed by ODRE region label."""
    return FRANCE_REGIONAL_WEATHER_LOCATIONS

