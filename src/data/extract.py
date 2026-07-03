"""Data extraction utilities for electricity price and carbon signals."""

from __future__ import annotations

from collections.abc import Iterator

from src.data.api_client import PaginatedApiClient
from src.data.contracts import ApiPageResult


def extract_energy_data() -> list[dict[str, object]]:
    """Return raw energy records from configured external sources."""
    return []


def extract_paginated_api(
    client: PaginatedApiClient,
    endpoint: str,
    params: dict[str, object] | None = None,
) -> Iterator[ApiPageResult]:
    """Stream raw API pages without loading the whole source into memory."""
    yield from client.fetch_pages(endpoint=endpoint, params=params)
