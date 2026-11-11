"""Metadata-only estimates for budget-safe GC continuous alternatives."""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path

import databento as db


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "diagnostic_is" / "databento_gc_budget_options.json"
BASE = {
    "dataset": "GLBX.MDP3",
    "stype_in": "continuous",
    "schema": "ohlcv-1m",
    "start": "2015-01-02T00:00:00Z",
    "end": "2023-03-17T00:00:00Z",
}


def main() -> None:
    key = os.getenv("DATABENTO_API_KEY") or getpass.getpass(
        "Databento API key (input hidden, never persisted): "
    ).strip()
    if not key:
        raise RuntimeError("No API key; no metadata request made")
    client = db.Historical(key)
    estimates = {}
    options = [
        ("GC.v.0", "GC.v.0", "rank 0 by traded volume; preferred liquid-contract proxy"),
        ("GC.c.0", "GC.c.0", "rank 0 by calendar expiration; nearest-expiry proxy"),
        ("GC.v.0+v.1", ["GC.v.0", "GC.v.1"], "aggregate the two highest-volume outright contracts"),
        ("GC.v.0+v.1+v.2", ["GC.v.0", "GC.v.1", "GC.v.2"], "aggregate the three highest-volume outright contracts"),
    ]
    options.extend(
        (
            f"GC.v.0..v.{rank_count - 1}",
            [f"GC.v.{rank}" for rank in range(rank_count)],
            f"aggregate the {rank_count} highest-volume outright contracts",
        )
        for rank_count in range(4, 11)
    )
    for name, symbols, meaning in options:
        request = {**BASE, "symbols": symbols}
        estimates[name] = {
            "meaning": meaning,
            "request": request,
            "estimated_cost_usd": float(client.metadata.get_cost(**request)),
            "estimated_record_count": int(client.metadata.get_record_count(**request)),
            "estimated_billable_size_bytes": int(client.metadata.get_billable_size(**request)),
        }
    payload = {
        "status": "estimate_only_awaiting_user_choice",
        "credit_cap_usd": 24.0,
        "estimates": estimates,
        "paid_data_download_calls": 0,
        "roll_control": "flag mapping changes and report roll +/-2 trading sessions separately",
        "limitation": "single mapped outright contract at each timestamp; not aggregate volume across all GC contracts",
        "api_key_persisted": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.environ.pop("DATABENTO_API_KEY", None)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
