"""Download MCL.v.0 once, validate it, then atomically publish the parquet."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import databento as db
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "downloaded"
TARGET = OUTPUT_DIR / "MCL_v_0_1m_20210712_20260924.parquet"
PART = TARGET.with_suffix(".part.parquet")
METADATA = TARGET.with_suffix(".metadata.json")
DATASET, SYMBOL, STYPE_IN, SCHEMA = "GLBX.MDP3", "MCL.v.0", "continuous", "ohlcv-1m"
START, END, MAX_COST_USD = "2021-07-12T00:00:00Z", "2026-09-24T00:00:00Z", 6.60
REQUIRED = {"rtype", "publisher_id", "instrument_id", "open", "high", "low", "close", "volume", "symbol"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(f"Refusing a paid duplicate download: {TARGET}")
    if PART.exists():
        raise FileExistsError(f"Refusing ambiguous partial-file recovery: {PART}")
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY is required and is never persisted")
    client = db.Historical(key)
    request = dict(dataset=DATASET, symbols=SYMBOL, stype_in=STYPE_IN,
                   schema=SCHEMA, start=START, end=END)
    estimate = float(client.metadata.get_cost(**request))
    print(f"Validated pre-download estimate: ${estimate:.6f}", flush=True)
    if estimate > MAX_COST_USD:
        raise RuntimeError(f"Cost guard stopped request: ${estimate:.6f} > ${MAX_COST_USD:.2f}")

    # The sole paid request in this script.
    frame = client.timeseries.get_range(**request).to_df()
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise RuntimeError("Invalid or empty Databento frame")
    if not frame.index.is_monotonic_increasing or frame.index.duplicated().any():
        raise RuntimeError("Timestamp validation failed")
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")
    if sorted(frame.symbol.dropna().astype(str).unique()) != [SYMBOL]:
        raise RuntimeError("Returned symbol validation failed")
    first, last = frame.index.min().tz_convert("UTC"), frame.index.max().tz_convert("UTC")
    if first < pd.Timestamp(START) or last >= pd.Timestamp(END):
        raise RuntimeError("Returned timestamps fall outside locked range")
    if not frame[["open", "high", "low", "close"]].notna().all().all():
        raise RuntimeError("OHLC contains missing values")
    if not ((frame.high >= frame[["open", "close"]].max(axis=1)).all()
            and (frame.low <= frame[["open", "close"]].min(axis=1)).all()
            and (frame.high >= frame.low).all()):
        raise RuntimeError("OHLC geometry failed")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(PART)
    if pq.read_metadata(PART).num_rows != len(frame):
        raise RuntimeError("Parquet row count failed")
    file_hash = sha256(PART)
    PART.replace(TARGET)
    metadata = {
        "request": request, "estimated_cost_usd": estimate,
        "cost_guard_usd": MAX_COST_USD, "paid_get_range_calls": 1,
        "row_count": len(frame), "first_timestamp": first.isoformat(),
        "last_timestamp": last.isoformat(), "duplicate_timestamps": 0,
        "instrument_count": int(frame.instrument_id.nunique()),
        "instrument_transitions": int(frame.instrument_id.ne(frame.instrument_id.shift()).sum() - 1),
        "file": str(TARGET.relative_to(REPO_ROOT)), "file_size_bytes": TARGET.stat().st_size,
        "sha256": file_hash, "api_key_persisted": False,
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
