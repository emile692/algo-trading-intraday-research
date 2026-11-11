"""Download MCL.v.0 (launch -> 2026-11-11) and CL.v.0 (2015 -> MCL launch), validate, publish, back up.

One paid call per file. Existing targets are never re-downloaded. The API key is read
from DATABENTO_API_KEY and never persisted. Each file is copied to an archive outside
the repository and both copies are set read-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

import databento as db
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "downloaded"
ARCHIVE_DIR = Path(os.environ.get("MARKET_DATA_ARCHIVE", r"C:\Data_local\market_data_archive\databento"))
DATASET, STYPE_IN, SCHEMA = "GLBX.MDP3", "continuous", "ohlcv-1m"
JOBS = [
    ("MCL.v.0", "2021-07-12T00:00:00Z", "2026-11-11T00:00:00Z", 6.75, "MCL_v_0_1m_20210712_20261111"),
    ("CL.v.0", "2015-01-01T00:00:00Z", "2021-07-12T00:00:00Z", 8.50, "CL_v_0_1m_20150101_20210712"),
]
REQUIRED = {"instrument_id", "open", "high", "low", "close", "volume", "symbol"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def backup(path: Path) -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_DIR / path.name
    if not dest.exists():
        shutil.copy2(path, dest)
    if sha256(dest) != sha256(path):
        raise RuntimeError(f"Archive copy differs: {dest}")
    for p in (path, dest):
        p.chmod(stat.S_IREAD)


def fetch(client: db.Historical, symbol: str, start: str, end: str, guard: float, stem: str) -> None:
    target = OUTPUT_DIR / f"{stem}.parquet"
    meta_path = OUTPUT_DIR / f"{stem}.metadata.json"
    if target.exists():
        print(f"Already present, no paid call: {target}")
        backup(target); backup(meta_path)
        return
    request = dict(dataset=DATASET, symbols=symbol, stype_in=STYPE_IN, schema=SCHEMA, start=start, end=end)
    estimate = float(client.metadata.get_cost(**request))
    print(f"{symbol} estimate: ${estimate:.6f}", flush=True)
    if estimate > guard:
        raise RuntimeError(f"Cost guard stopped {symbol}: ${estimate:.6f} > ${guard:.2f}")
    frame = client.timeseries.get_range(**request).to_df()  # the sole paid call for this file
    if frame.empty or frame.index.tz is None:
        raise RuntimeError("Invalid or empty Databento frame")
    if not frame.index.is_monotonic_increasing or frame.index.duplicated().any():
        raise RuntimeError("Timestamp validation failed")
    if REQUIRED - set(frame.columns):
        raise RuntimeError(f"Missing columns: {sorted(REQUIRED - set(frame.columns))}")
    if sorted(frame.symbol.dropna().astype(str).unique()) != [symbol]:
        raise RuntimeError("Returned symbol validation failed")
    if not frame[["open", "high", "low", "close"]].notna().all().all():
        raise RuntimeError("OHLC contains missing values")
    if not ((frame.high >= frame[["open", "close"]].max(axis=1)).all()
            and (frame.low <= frame[["open", "close"]].min(axis=1)).all()):
        raise RuntimeError("OHLC geometry failed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    part = OUTPUT_DIR / f"{stem}.part.parquet"
    frame.to_parquet(part)
    if pq.read_metadata(part).num_rows != len(frame):
        raise RuntimeError("Parquet row count failed")
    part.replace(target)
    metadata = {
        "request": request, "estimated_cost_usd": estimate, "cost_guard_usd": guard,
        "row_count": len(frame), "first_timestamp": frame.index.min().isoformat(),
        "last_timestamp": frame.index.max().isoformat(),
        "instrument_count": int(frame.instrument_id.nunique()),
        "file": f"data/downloaded/{target.name}", "file_size_bytes": target.stat().st_size,
        "sha256": sha256(target), "archive_copy": str(ARCHIVE_DIR / target.name), "api_key_persisted": False,
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    backup(target); backup(meta_path)
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY is required and is never persisted")
    client = db.Historical(key)
    for job in JOBS:
        fetch(client, *job)


if __name__ == "__main__":
    main()
