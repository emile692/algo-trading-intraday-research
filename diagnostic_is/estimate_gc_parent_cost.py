"""Estimate, but never download, the Databento GC parent request.

This module intentionally contains no ``timeseries.get_range`` or batch call.
The end timestamp is exclusive, so 2023-03-16 is included in full.
"""

from __future__ import annotations

import json
import os
import getpass
from pathlib import Path

import databento as db


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "diagnostic_is" / "databento_gc_parent_cost_estimate.json"
API_KEY_ENV = "DATABENTO_API_KEY"

BASE = {
    "dataset": "GLBX.MDP3",
    "symbols": "GC.FUT",
    "stype_in": "parent",
    "start": "2015-01-02T00:00:00Z",
    "end": "2023-03-17T00:00:00Z",
}
REQUESTS = {
    "contract_bars": {**BASE, "schema": "ohlcv-1m"},
    "instrument_definitions": {**BASE, "schema": "definition"},
}


def main() -> None:
    key = os.getenv(API_KEY_ENV)
    if not key:
        key = getpass.getpass("Databento API key (input hidden, never persisted): ").strip()
    if not key:
        raise RuntimeError(f"No {API_KEY_ENV}; no data request was made and no estimate is available")
    client = db.Historical(key)
    estimates = {}
    for name, request in REQUESTS.items():
        estimates[name] = {
            "estimated_cost_usd": float(client.metadata.get_cost(**request)),
            "estimated_record_count": int(client.metadata.get_record_count(**request)),
            "estimated_billable_size_bytes": int(client.metadata.get_billable_size(**request)),
        }
    payload = {
        "status": "estimate_only_awaiting_user_download_approval",
        "requests": REQUESTS,
        "estimates": estimates,
        "combined_estimated_cost_usd": sum(row["estimated_cost_usd"] for row in estimates.values()),
        "paid_data_download_calls": 0,
        "processing_after_approval": [
            "retain only instrument_class FUTURE; GC.FUT parent also returns calendar spreads",
            "aggregate one-minute volume across outright GC contracts by timestamp",
            "resample to the canonical MGC five-minute left-labelled bars",
            "restrict all downstream statistics to 2015-01-02 through 2023-03-16",
        ],
        "api_key_persisted": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.environ.pop(API_KEY_ENV, None)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
