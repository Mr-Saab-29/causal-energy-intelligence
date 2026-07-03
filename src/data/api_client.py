"""HTTP extraction utilities with conservative pagination controls."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from src.data.contracts import ApiPageResult


PaginationMode = Literal["offset", "cursor"]


@dataclass(frozen=True)
class PaginationConfig:
    """Controls request pacing and pagination limits."""

    mode: PaginationMode = "offset"
    page_size: int = 500
    max_pages: int = 100
    request_timeout_seconds: float = 30.0
    min_interval_seconds: float = 0.5
    offset_param: str = "offset"
    limit_param: str = "limit"
    cursor_param: str = "cursor"
    next_cursor_field: str = "next_cursor"
    records_field: str = "data"


class PaginatedApiClient:
    """Small API client designed to avoid timeouts and excessive requests."""

    def __init__(
        self,
        base_url: str,
        source_name: str,
        headers: dict[str, str] | None = None,
        config: PaginationConfig | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.source_name = source_name
        self.headers = headers or {}
        self.config = config or PaginationConfig()

    def fetch_pages(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Iterator[ApiPageResult]:
        """Yield normalized pages from an offset- or cursor-paginated endpoint."""
        request_params = dict(params or {})
        next_cursor: str | None = None
        offset = int(request_params.get(self.config.offset_param, 0))

        with httpx.Client(
            base_url=self.base_url,
            headers=self.headers,
            timeout=self.config.request_timeout_seconds,
        ) as client:
            for _ in range(self.config.max_pages):
                if self.config.mode == "offset":
                    request_params[self.config.offset_param] = offset
                    request_params[self.config.limit_param] = self.config.page_size
                else:
                    request_params[self.config.limit_param] = self.config.page_size
                    if next_cursor:
                        request_params[self.config.cursor_param] = next_cursor

                response = client.get(endpoint, params=request_params)
                response.raise_for_status()
                payload = response.json()

                records = self._extract_records(payload)
                page_result = ApiPageResult(
                    source_name=self.source_name,
                    endpoint=endpoint,
                    page_token=next_cursor,
                    next_page_token=self._extract_next_cursor(payload),
                    offset=offset if self.config.mode == "offset" else None,
                    limit=self.config.page_size,
                    records=records,
                    raw_response=payload,
                )
                yield page_result

                if not records:
                    break

                if self.config.mode == "offset":
                    if len(records) < self.config.page_size:
                        break
                    offset += self.config.page_size
                else:
                    next_cursor = page_result.next_page_token
                    if not next_cursor:
                        break

                time.sleep(self.config.min_interval_seconds)

    def _extract_records(self, payload: Any) -> list[dict[str, object]]:
        if isinstance(payload, list):
            return [record for record in payload if isinstance(record, dict)]

        if not isinstance(payload, dict):
            return []

        records = payload.get(self.config.records_field, [])
        if isinstance(records, list):
            return [record for record in records if isinstance(record, dict)]
        return []

    def _extract_next_cursor(self, payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None

        next_cursor = payload.get(self.config.next_cursor_field)
        return str(next_cursor) if next_cursor else None

