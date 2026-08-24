"""Small retry helper for upstream API requests."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import httpx

DEFAULT_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


def get_with_retries(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    max_retries: int = 5,
    backoff_seconds: float = 30.0,
    retry_status_codes: Iterable[int] = DEFAULT_RETRY_STATUS_CODES,
) -> httpx.Response:
    """GET with retry support for transient status codes and network timeouts."""
    retry_codes = set(retry_status_codes)
    for attempt in range(max_retries + 1):
        try:
            response = client.get(url, params=params)
            if response.status_code not in retry_codes:
                response.raise_for_status()
                return response

            if attempt == max_retries:
                response.raise_for_status()

            retry_after = response.headers.get("retry-after")
            wait_seconds = (
                float(retry_after)
                if retry_after
                else backoff_seconds * (attempt + 1)
            )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError, httpx.ReadTimeout):
            if attempt == max_retries:
                raise
            wait_seconds = backoff_seconds * (attempt + 1)

        time.sleep(wait_seconds)

    raise RuntimeError("HTTP request failed after retries")
