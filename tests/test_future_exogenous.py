import httpx
import pytest

from src.data.future_exogenous import (
    OPEN_METEO_FORECAST_BASE_URL,
    fetch_open_meteo_forecast,
)


def test_fetch_open_meteo_forecast_retries_invalid_json_response() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(
                200,
                content=b"",
                headers={"content-type": "text/html"},
                request=request,
            )
        return httpx.Response(
            200,
            json={"hourly": {"time": ["2026-09-08T00:00"], "temperature_2m": [19.5]}},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=OPEN_METEO_FORECAST_BASE_URL, transport=transport) as client:
        payload = fetch_open_meteo_forecast(
            client,
            latitude=48.8566,
            longitude=2.3522,
            horizon_hours=24,
            json_retries=1,
            json_backoff_seconds=0,
        )

    assert requests == 2
    assert payload["hourly"]["time"] == ["2026-09-08T00:00"]


def test_fetch_open_meteo_forecast_raises_diagnostic_after_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="temporarily unavailable",
            headers={"content-type": "text/html"},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=OPEN_METEO_FORECAST_BASE_URL, transport=transport) as client:
        with pytest.raises(ValueError) as error:
            fetch_open_meteo_forecast(
                client,
                latitude=48.8566,
                longitude=2.3522,
                horizon_hours=24,
                json_retries=0,
                json_backoff_seconds=0,
            )

    message = str(error.value)
    assert "non-JSON or malformed JSON" in message
    assert "status=200" in message
    assert "text/html" in message
    assert "temporarily unavailable" in message
