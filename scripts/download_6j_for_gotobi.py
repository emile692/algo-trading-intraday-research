"""One-shot guarded acquisition of 6J.v.0 minute bars for Gotobi Phase 1."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from src.data.downloader import download_data, estimate_cost


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "downloaded" / "6J_v_0_1m_20150101_20261003.parquet"
META = TARGET.with_suffix(".metadata.json")
START = "2015-01-01T00:00:00Z"
END = "2026-10-03T00:00:00Z"
SYMBOL = "6J.v.0"
COST_GUARD_USD = 14.85


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {TARGET}")
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
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(TARGET)
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
        "file": str(TARGET.relative_to(ROOT)),
        "file_size_bytes": TARGET.stat().st_size,
        "sha256": digest,
        "api_key_persisted": False,
    }
    META.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
