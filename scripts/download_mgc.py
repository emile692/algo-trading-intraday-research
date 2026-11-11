"""One-shot guarded restoration of the canonical MGC.v.0 one-minute bars."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from src.data.downloader import download_data, estimate_cost
from src.data.process_index import process_index_rename_to_timestamp


ROOT = Path(__file__).resolve().parents[1]
RAW_TARGET = ROOT / "data" / "downloaded" / "MGC_v_0_1m_20150101_20260923.parquet"
TARGET = ROOT / "data" / "processed" / "parquet" / "MGC_v_0_1m_20150101_20260923.parquet"
META = RAW_TARGET.with_suffix(".metadata.json")
START = "2015-01-01T00:00:00Z"
END = "2026-09-23T00:00:00Z"
SYMBOL = "MGC.v.0"
COST_GUARD_USD = 12.10
EXPECTED_SHA256 = "2bbc2d10d59409fd347779fc00a23633fa6ad6aec0cf4b52e7a6d7a278426e96"


def main() -> None:
    existing = [path for path in (RAW_TARGET, TARGET) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing files: {existing}")
    cost = estimate_cost(SYMBOL, START, END)
    if cost > COST_GUARD_USD:
        raise RuntimeError(f"Estimated cost ${cost:.6f} exceeds guard ${COST_GUARD_USD:.2f}")
    frame = download_data(SYMBOL, START, END)
    os.environ.pop("DATABENTO_API_KEY", None)
    if frame.empty or frame.index.duplicated().any():
        raise RuntimeError("Downloaded frame is empty or contains duplicate timestamps")
    required = {"open", "high", "low", "close", "volume", "instrument_id"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Missing required columns: {sorted(required - set(frame.columns))}")
    RAW_TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(RAW_TARGET)
    processed = process_index_rename_to_timestamp(frame.copy(), tz_from="UTC", tz_to="America/New_York")
    processed.to_parquet(TARGET)
    raw_digest = hashlib.sha256(RAW_TARGET.read_bytes()).hexdigest()
    digest = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    metadata = {
        "request": {
            "dataset": "GLBX.MDP3",
            "symbols": SYMBOL,
            "stype_in": "continuous",
            "schema": "ohlcv-1m",
            "start": START,
            "end": END,
        },
        "estimated_cost_usd": cost,
        "cost_guard_usd": COST_GUARD_USD,
        "paid_get_range_calls": 1,
        "row_count": len(frame),
        "first_timestamp": str(frame.index.min()),
        "last_timestamp": str(frame.index.max()),
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "instrument_count": int(frame.instrument_id.nunique()),
        "raw_file": str(RAW_TARGET.relative_to(ROOT)),
        "raw_file_size_bytes": RAW_TARGET.stat().st_size,
        "raw_sha256": raw_digest,
        "processing": "rename index to timestamp and convert UTC to America/New_York",
        "processed_file": str(TARGET.relative_to(ROOT)),
        "processed_file_size_bytes": TARGET.stat().st_size,
        "processed_sha256": digest,
        "expected_processed_sha256": EXPECTED_SHA256,
        "matches_historical_processed_sha256": digest == EXPECTED_SHA256,
        "api_key_persisted": False,
    }
    META.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    if digest != EXPECTED_SHA256:
        raise RuntimeError("Downloaded MGC file does not match the historical SHA-256")


if __name__ == "__main__":
    main()
