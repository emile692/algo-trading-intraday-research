"""Fetch free daily GLBX.MDP3 quality conditions for the locked B window."""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path

import databento as db


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "diagnostic_is" / "gc_b0b" / "databento_dataset_conditions.json"


def main() -> None:
    key = os.getenv("DATABENTO_API_KEY") or getpass.getpass(
        "Databento API key (input hidden, never persisted): "
    ).strip()
    if not key:
        raise RuntimeError("No API key; no metadata request made")
    rows = db.Historical(key).metadata.get_dataset_condition(
        dataset="GLBX.MDP3", start_date="2015-01-02", end_date="2023-03-16"
    )
    os.environ.pop("DATABENTO_API_KEY", None)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    degraded = [row for row in rows if str(row.get("condition", "")).lower() == "degraded"]
    print(json.dumps({"rows": len(rows), "degraded_dates": len(degraded), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
