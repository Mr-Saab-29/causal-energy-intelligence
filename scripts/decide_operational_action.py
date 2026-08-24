"""Decide whether the daily operational refresh needs a retrain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.baseline_price import PRODUCTION_SIGNAL_TARGETS

FORECAST_MONITORING_PATH = ROOT / "reports/metrics/forecast_monitoring.json"


def main(argv: list[str] | None = None) -> int:
    """Write the operational action decision as JSON and optional GitHub outputs."""
    parser = argparse.ArgumentParser(description="Decide daily operational action.")
    parser.add_argument("--github-output", default=None)
    args = parser.parse_args(argv)

    decision = decide_operational_action()
    print(json.dumps(decision, indent=2))
    if args.github_output:
        write_github_outputs(Path(args.github_output), decision)
    return 0


def decide_operational_action() -> dict[str, Any]:
    """Return whether the workflow should retrain before publishing recommendations."""
    reasons: list[str] = []
    monitoring = read_json(FORECAST_MONITORING_PATH)
    if not monitoring:
        reasons.append("forecast_monitoring_missing")
    elif monitoring.get("retraining_recommended"):
        reasons.append("forecast_monitoring_retraining_recommended")

    missing_models = missing_required_model_artifacts()
    if missing_models:
        reasons.append("required_model_artifacts_missing")

    return {
        "action": "retrain" if reasons else "recommend",
        "retrain": bool(reasons),
        "reasons": reasons,
        "missing_model_artifacts": missing_models,
    }


def missing_required_model_artifacts() -> list[str]:
    """List model targets that do not have a persisted artifact."""
    missing: list[str] = []
    model_dir = ROOT / "models"
    for target in ["consumption", *PRODUCTION_SIGNAL_TARGETS]:
        if not list(model_dir.glob(f"*_{target}_baseline.joblib")):
            missing.append(target)
    if not list(model_dir.glob("*_price_baseline.joblib")):
        missing.append("price")
    return missing


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object if it exists."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_github_outputs(path: Path, decision: dict[str, Any]) -> None:
    """Append decision values to GitHub Actions output file."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"action={decision['action']}\n")
        handle.write(f"retrain={str(decision['retrain']).lower()}\n")
        handle.write(f"reasons={','.join(decision['reasons'])}\n")


if __name__ == "__main__":
    raise SystemExit(main())
