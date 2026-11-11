"""Download and merge the missing MNQ/MGC data for the paper-period replay.

The API key is requested with getpass and is never persisted.  By default the
script only estimates cost; ``--execute`` is required for paid downloads.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
from pathlib import Path

import databento as db
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADED = ROOT / "data" / "downloaded"
PROCESSED = ROOT / "data" / "processed" / "parquet"
DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
MAX_COST_USD = 1.00

REQUESTS = [
    {
        "symbol": "MNQ.c.0",
        "start": "2026-03-20T00:00:00Z",
        "end": "2026-10-03T00:00:00Z",
        "raw_name": "MNQ_c_0_1m_20260320_20261003_extension.parquet",
        "existing": "MNQ_c_0_1m_20260321_094501.parquet",
        "merged": "MNQ_c_0_1m_20190505_20261003.parquet",
    },
    {
        "symbol": "MGC.v.0",
        "start": "2026-09-23T00:00:00Z",
        "end": "2026-10-03T00:00:00Z",
        "raw_name": "MGC_v_0_1m_20260923_20261003_extension.parquet",
        "existing": "MGC_v_0_1m_20150101_20260923.parquet",
        "merged": "MGC_v_0_1m_20150101_20261003.parquet",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def api_args(request: dict) -> dict:
    return {
        "dataset": DATASET,
        "symbols": request["symbol"],
        "stype_in": "continuous",
        "schema": SCHEMA,
        "start": request["start"],
        "end": request["end"],
    }


def merge_extension(existing_path: Path, extension: pd.DataFrame, merged_path: Path) -> dict:
    existing = pd.read_parquet(existing_path)
    extension = extension.copy()
    extension.index = pd.DatetimeIndex(extension.index)
    if extension.index.tz is None:
        extension.index = extension.index.tz_localize("UTC")
    extension.index = extension.index.tz_convert("America/New_York")
    extension.index.name = "timestamp"

    overlap = existing.index.intersection(extension.index)
    compare_cols = [c for c in ("open", "high", "low", "close", "volume") if c in existing and c in extension]
    overlap_max_abs_diff = 0.0
    if len(overlap) and compare_cols:
        left = existing.loc[overlap, compare_cols].astype(float)
        right = extension.loc[overlap, compare_cols].astype(float)
        overlap_max_abs_diff = float((left - right).abs().max().max())

    merged = pd.concat([existing, extension]).sort_index()
    duplicate_rows = int(merged.index.duplicated(keep="last").sum())
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.to_parquet(merged_path)
    return {
        "existing_rows": int(len(existing)),
        "extension_rows": int(len(extension)),
        "overlap_rows": int(len(overlap)),
        "overlap_max_abs_ohlcv_diff": overlap_max_abs_diff,
        "duplicates_removed": duplicate_rows,
        "merged_rows": int(len(merged)),
        "first_timestamp": str(merged.index.min()),
        "last_timestamp": str(merged.index.max()),
        "merged_sha256": sha256(merged_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Perform the paid downloads")
    args = parser.parse_args()

    key = getpass.getpass("Databento API key: ")
    client = db.Historical(key)
    estimates = [float(client.metadata.get_cost(**api_args(r))) for r in REQUESTS]
    total = sum(estimates)
    print(json.dumps({"estimates_usd": estimates, "total_usd": total, "guard_usd": MAX_COST_USD}, indent=2))
    if total > MAX_COST_USD:
        raise RuntimeError(f"Estimated cost ${total:.6f} exceeds guard ${MAX_COST_USD:.2f}")
    if not args.execute:
        return

    DOWNLOADED.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": DATASET, "schema": SCHEMA, "cost_guard_usd": MAX_COST_USD, "requests": []}
    for request, estimate in zip(REQUESTS, estimates):
        frame = client.timeseries.get_range(**api_args(request)).to_df()
        raw_path = DOWNLOADED / request["raw_name"]
        frame.to_parquet(raw_path)
        merged_path = PROCESSED / request["merged"]
        merge = merge_extension(PROCESSED / request["existing"], frame, merged_path)
        manifest["requests"].append(
            {
                "request": api_args(request),
                "estimated_cost_usd": estimate,
                "paid_get_range_calls": 1,
                "raw_path": str(raw_path.relative_to(ROOT)),
                "raw_rows": int(len(frame)),
                "raw_sha256": sha256(raw_path),
                "processed_path": str(merged_path.relative_to(ROOT)),
                **merge,
            }
        )
        print(f"Saved and merged {request['symbol']}: {len(frame):,} extension rows")

    manifest["estimated_total_cost_usd"] = total
    manifest["paid_get_range_calls"] = len(REQUESTS)
    manifest["api_key_persisted"] = False
    manifest_path = DOWNLOADED / "nine_sleeve_paper_extension_20261003.metadata.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
