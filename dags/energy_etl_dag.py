"""Airflow DAG entrypoint for the energy ETL pipeline.

This file is intentionally lightweight until the data source contracts are
finalized. The production DAG should call the extract, transform, and load
functions in src.data.
"""

from __future__ import annotations

from datetime import datetime

try:
    from airflow.decorators import dag, task
except ImportError:  # Allows local imports without Airflow installed.
    dag = None
    task = None


if dag and task:

    @dag(
        dag_id="energy_etl",
        start_date=datetime(2026, 1, 1),
        schedule="@hourly",
        catchup=False,
        tags=["energy", "etl"],
    )
    def energy_etl_dag() -> None:
        """Extract, transform, and load energy market data."""

        @task
        def extract() -> str:
            return "extract-placeholder"

        @task
        def transform(raw_marker: str) -> str:
            return f"transform-placeholder:{raw_marker}"

        @task
        def load(features_marker: str) -> None:
            print(f"load-placeholder:{features_marker}")

        load(transform(extract()))

    energy_etl_dag()

