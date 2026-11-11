"""One-shot guarded download of GC volume ranks 0 through 9.

The user explicitly approved the combined request after a metadata estimate of
$18.757453262806. The script makes exactly one paid get_range call and refuses
to overwrite an existing result.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
from pathlib import Path

import databento as db
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "downloaded" / "GC_v_0_to_v_9_1m_20150102_20230317.parquet"
META = RAW.with_suffix(".metadata.json")
SYMBOLS = [f"GC.v.{rank}" for rank in range(10)]
REQUEST = {
    "dataset": "GLBX.MDP3",
    "symbols": SYMBOLS,
    "stype_in": "continuous",
    "schema": "ohlcv-1m",
    "start": "2015-01-02T00:00:00Z",
    "end": "2023-03-17T00:00:00Z",
}
COST_GUARD_USD = 20.00


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    existing = [path for path in (RAW, META) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing files: {existing}")
    key = os.getenv("DATABENTO_API_KEY") or getpass.getpass(
        "Databento API key (input hidden, never persisted): "
    ).strip()
    if not key:
        raise RuntimeError("No API key; no request made")
    client = db.Historical(key)
    estimate = float(client.metadata.get_cost(**REQUEST))
    records_estimate = int(client.metadata.get_record_count(**REQUEST))
    size_estimate = int(client.metadata.get_billable_size(**REQUEST))
    if estimate > COST_GUARD_USD:
        raise RuntimeError(
            f"Fresh estimate ${estimate:.6f} exceeds hard guard ${COST_GUARD_USD:.2f}; no download made"
        )
    store = client.timeseries.get_range(**REQUEST, stype_out="instrument_id")
    frame = store.to_df()
    os.environ.pop("DATABENTO_API_KEY", None)
    key = ""
    if frame.empty:
        raise RuntimeError("Databento returned an empty frame")
    required = {"open", "high", "low", "close", "volume", "instrument_id"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Missing columns: {sorted(required - set(frame.columns))}")
    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is None:
        raise RuntimeError("Expected timezone-aware Databento timestamps")
    if idx.min() < pd.Timestamp(REQUEST["start"]) or idx.max() >= pd.Timestamp(REQUEST["end"]):
        raise RuntimeError(f"Out-of-request timestamp: {idx.min()} to {idx.max()}")
    duplicate_contract_minutes = int(
        frame.reset_index().duplicated([frame.index.name or "ts_event", "instrument_id"]).sum()
    )
    if duplicate_contract_minutes:
        raise RuntimeError(f"Found {duplicate_contract_minutes} duplicate timestamp/instrument rows")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(RAW)
    payload = {
        "request": REQUEST,
        "fresh_estimated_cost_usd": estimate,
        "cost_guard_usd": COST_GUARD_USD,
        "estimated_record_count": records_estimate,
        "estimated_billable_size_bytes": size_estimate,
        "paid_get_range_calls": 1,
        "rows": len(frame),
        "first_timestamp": str(idx.min()),
        "last_timestamp": str(idx.max()),
        "distinct_instrument_ids": int(frame.instrument_id.nunique()),
        "duplicate_timestamp_instrument_rows": duplicate_contract_minutes,
        "raw_file": str(RAW.relative_to(ROOT)),
        "raw_file_size_bytes": RAW.stat().st_size,
        "raw_sha256": sha256(RAW),
        "api_key_persisted": False,
    }
    META.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
