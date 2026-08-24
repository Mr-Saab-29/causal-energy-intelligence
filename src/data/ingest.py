"""Paginated ingestion helpers."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.engine import Engine

from src.data.api_client import PaginatedApiClient
from src.data.contracts import ApiPageResult


def ingest_paginated_endpoint(
    client: PaginatedApiClient,
    engine: Engine,
    endpoint: str,
    params: dict[str, object] | None = None,
) -> Iterator[ApiPageResult]:
    """Fetch an endpoint page by page without persisting raw payloads."""
    _ = engine
    for page in client.fetch_pages(endpoint=endpoint, params=params):
        yield page
