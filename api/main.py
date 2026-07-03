"""FastAPI application for the Causal Energy Intelligence Platform."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="Causal Energy Intelligence Platform",
    description="Forecasting, causal inference, and what-if optimization API.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    """Return basic API metadata."""
    return {"service": "causal-energy-intelligence", "status": "running"}

