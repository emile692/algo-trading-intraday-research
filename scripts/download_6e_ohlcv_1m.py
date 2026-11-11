"""Download 6E.v.0 once, validate it, then atomically publish the parquet."""

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
TARGET = OUTPUT_DIR / "6E_v_0_1m_20150101_20260923.parquet"
PART = TARGET.with_suffix(".part.parquet")
METADATA = TARGET.with_suffix(".metadata.json")

DATASET = "GLBX.MDP3"
SYMBOL = "6E.v.0"
STYPE_IN = "continuous"
SCHEMA = "ohlcv-1m"
START = "2015-01-01T00:00:00Z"
END = "2026-09-23T00:00:00Z"  # exclusive
MAX_COST_USD = 14.90
REQUIRED_COLUMNS = {
    "rtype", "publisher_id", "instrument_id", "open", "high", "low", "close", "volume", "symbol"
}


def _publish_existing_part(estimated_cost: float) -> None:
    """Recover a fully written part file without making any network request."""
    frame = pd.read_parquet(PART)
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        raise RuntimeError("Partial parquet is not a valid Databento frame")
    if not frame.index.is_monotonic_increasing or frame.index.duplicated().any():
        raise RuntimeError("Partial parquet timestamp validation failed")
    if sorted(frame["symbol"].dropna().astype(str).unique().tolist()) != [SYMBOL]:
        raise RuntimeError("Partial parquet symbol validation failed")
    first = pd.Timestamp(frame.index.min()).tz_convert("UTC")
    last = pd.Timestamp(frame.index.max()).tz_convert("UTC")
    if first < pd.Timestamp(START) or last >= pd.Timestamp(END):
        raise RuntimeError("Partial parquet lies outside the locked request range")
    price_columns = ["open", "high", "low", "close"]
    if not frame[price_columns].notna().all().all():
        raise RuntimeError("Partial parquet OHLC contains missing values")
    if not ((frame["high"] >= frame[["open", "close"]].max(axis=1)).all()
            and (frame["low"] <= frame[["open", "close"]].min(axis=1)).all()
            and (frame["high"] >= frame["low"]).all()):
        raise RuntimeError("Partial parquet OHLC geometry failed")
    row_count = int(len(frame))
    instrument_count = int(frame["instrument_id"].nunique())
    transitions = int(frame["instrument_id"].ne(frame["instrument_id"].shift()).sum() - 1)
    del frame
    file_hash = sha256(PART)
    PART.replace(TARGET)
    metadata = {
        "request": {
            "dataset": DATASET, "symbols": SYMBOL, "stype_in": STYPE_IN,
            "schema": SCHEMA, "start": START, "end": END,
        },
        "estimated_cost_usd": estimated_cost,
        "cost_guard_usd": MAX_COST_USD,
        "paid_get_range_calls": 1,
        "recovered_from_validated_part_without_redownload": True,
        "row_count": row_count,
        "first_timestamp": first.isoformat(),
        "last_timestamp": last.isoformat(),
        "duplicate_timestamps": 0,
        "instrument_count": instrument_count,
        "instrument_transitions": transitions,
        "file": str(TARGET.relative_to(REPO_ROOT)),
        "file_size_bytes": TARGET.stat().st_size,
        "sha256": file_hash,
        "api_key_persisted": False,
    }
    METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(f"Refusing a paid duplicate download: {TARGET}")
    if PART.exists():
        print("Recovering validated .part parquet locally; no Databento request will be made.", flush=True)
        _publish_existing_part(14.825398)
        return
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY is required and is never written to disk")

    client = db.Historical(api_key)
    request = {
        "dataset": DATASET,
        "symbols": SYMBOL,
        "stype_in": STYPE_IN,
        "schema": SCHEMA,
        "start": START,
        "end": END,
    }
    estimated_cost = float(client.metadata.get_cost(**request))
    print(f"Validated pre-download estimate: ${estimated_cost:.6f}", flush=True)
    if estimated_cost > MAX_COST_USD:
        raise RuntimeError(f"Cost guard stopped request: ${estimated_cost:.6f} > ${MAX_COST_USD:.2f}")

    # This is the only paid data request in the script.
    store = client.timeseries.get_range(**request)
    frame = store.to_df()
    if frame.empty:
        raise RuntimeError("Databento returned an empty frame; no second request will be attempted")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex")
    if frame.index.tz is None:
        raise ValueError("Expected timezone-aware UTC timestamps")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Timestamps are not monotonic")
    duplicate_count = int(frame.index.duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Duplicate timestamps: {duplicate_count}")
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    symbols = sorted(frame["symbol"].dropna().astype(str).unique().tolist())
    if symbols != [SYMBOL]:
        raise ValueError(f"Unexpected returned symbols: {symbols}")
    first = pd.Timestamp(frame.index.min()).tz_convert("UTC")
    last = pd.Timestamp(frame.index.max()).tz_convert("UTC")
    if first < pd.Timestamp(START) or last >= pd.Timestamp(END):
        raise ValueError(f"Returned timestamps outside request: {first} -> {last}")
    price_columns = ["open", "high", "low", "close"]
    if not frame[price_columns].notna().all().all():
        raise ValueError("OHLC contains missing values")
    if not ((frame["high"] >= frame[["open", "close"]].max(axis=1)).all()
            and (frame["low"] <= frame[["open", "close"]].min(axis=1)).all()
            and (frame["high"] >= frame["low"]).all()):
        raise ValueError("OHLC geometry check failed")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(PART)
    parquet_metadata = pq.read_metadata(PART)
    if parquet_metadata.num_rows != len(frame):
        raise RuntimeError("Parquet row-count verification failed")
    del parquet_metadata
    part_hash = sha256(PART)
    PART.replace(TARGET)

    metadata = {
        "request": request,
        "estimated_cost_usd": estimated_cost,
        "cost_guard_usd": MAX_COST_USD,
        "paid_get_range_calls": 1,
        "row_count": int(len(frame)),
        "first_timestamp": first.isoformat(),
        "last_timestamp": last.isoformat(),
        "duplicate_timestamps": duplicate_count,
        "instrument_count": int(frame["instrument_id"].nunique()),
        "instrument_transitions": int(frame["instrument_id"].ne(frame["instrument_id"].shift()).sum() - 1),
        "file": str(TARGET.relative_to(REPO_ROOT)),
        "file_size_bytes": TARGET.stat().st_size,
        "sha256": part_hash,
        "api_key_persisted": False,
    }
    METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
