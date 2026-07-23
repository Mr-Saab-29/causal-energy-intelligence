"""Build static dashboard data from clean-hour recommendation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "frontend/public/data/dashboard.json"
PRODUCTION_MODEL_LABEL = "Production Model V1"


def main() -> None:
    """Write the dashboard JSON payload used by the frontend."""
    champion = read_json(ROOT / "reports/metrics/champion_model_selection.json")
    decision_metrics = read_json(ROOT / "reports/metrics/workload_decision_metrics.json")
    ranking_metrics = read_json(ROOT / "reports/metrics/ranking_specific_metrics.json")
    scenario_metrics = read_json(ROOT / "reports/metrics/scenario_reranking_metrics.json")
    recommendations = read_csv(
        ROOT / "reports/recommendations/champion_workload_recommendations.csv"
    )
    scenario_recommendations = read_csv(
        ROOT / "reports/scenarios/workload_scenario_recommendations.csv"
    )

    recommendation_rows = prepare_records(recommendations)
    scenario_rows = prepare_records(
        scenario_recommendations[
            scenario_recommendations["model"] == champion.get("champion_model")
        ]
    )
    payload = {
        "generated_from": {
            "champion_model_selection": "reports/metrics/champion_model_selection.json",
            "recommendations": "reports/recommendations/champion_workload_recommendations.csv",
            "scenario_recommendations": "reports/scenarios/workload_scenario_recommendations.csv",
        },
        "champion": {
            "model": champion.get("champion_model"),
            "display_model_name": PRODUCTION_MODEL_LABEL,
            "weights": champion.get("weights", {}),
            "selection_rule": champion.get("selection_rule"),
            "models": champion.get("models", []),
        },
        "summary": {
            "decision_metrics": decision_metrics.get("summary", []),
            "ranking_metrics": ranking_metrics.get("summary", []),
            "scenario_metrics": scenario_metrics.get("summary", []),
            "date_count": int(recommendations["decision_group"].nunique())
            if not recommendations.empty
            else 0,
            "recommendation_count": int(len(recommendations)),
            "average_confidence_score": safe_float(
                recommendations["confidence_score"].mean()
            )
            if "confidence_score" in recommendations
            else None,
            "high_confidence_share": safe_float(
                (recommendations["confidence_level"] == "high").mean()
            )
            if "confidence_level" in recommendations
            else None,
        },
        "filters": {
            "dates": sorted(recommendations["decision_group"].dropna().unique().tolist()),
            "scenarios": sorted(
                scenario_recommendations["scenario"].dropna().unique().tolist()
            ),
        },
        "recommendations": recommendation_rows,
        "scenario_recommendations": scenario_rows,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file if it exists, otherwise return an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def prepare_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a frame to JSON records with stable null handling."""
    if frame.empty:
        return []
    cleaned = frame.replace({pd.NA: None})
    cleaned = cleaned.where(pd.notna(cleaned), None)
    return cleaned.to_dict(orient="records")


def safe_float(value: float) -> float | None:
    """Return a rounded JSON-safe float."""
    if pd.isna(value):
        return None
    return round(float(value), 4)


if __name__ == "__main__":
    main()
