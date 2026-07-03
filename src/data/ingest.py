"""Raw ingestion pipeline helpers."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.engine import Engine

from src.data.api_client import PaginatedApiClient
from src.data.contracts import ApiPageResult
from src.data.load import insert_raw_api_page


def ingest_paginated_endpoint(
    client: PaginatedApiClient,
    engine: Engine,
    endpoint: str,
    params: dict[str, object] | None = None,
) -> Iterator[ApiPageResult]:
    """Fetch an endpoint page by page and persist each raw response."""
    for page in client.fetch_pages(endpoint=endpoint, params=params):
        request_params = {
            "page_token": page.page_token,
            "offset": page.offset,
            "limit": page.limit,
            **(params or {}),
        }
        insert_raw_api_page(
            engine=engine,
            source_name=page.source_name,
            endpoint=page.endpoint,
            request_params=request_params,
            response_payload=page.raw_response,
        )
        yield page

