"""Download the full 1-minute history of MNQ.v.0 and MES.v.0 (volume-ranked front month, unadjusted).

Purpose (2026-11-20): (1) replay the frozen MNQ engines on v.0 to measure the c.0 roll-week and switch-Monday effects
(descriptive only, no retuning); (2) the price-action study (inside-day false-break, 3-push) on micros.
Both micros started trading on 2019-05-05 (Globex evening), so the request starts 2019-05-01.

Raw files only: nothing is merged into the frozen c.0 history files.  The API key is read from DATABENTO_API_KEY or
requested with getpass, and never persisted.  Without ``--execute`` the script only prints the cost estimate.

    python scripts/download_micro_v0_full_20261120.py                         # estimate only
    python scripts/download_micro_v0_full_20261120.py --execute --max-usd 20  # paid download, guarded
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path

import databento as db
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADED = ROOT / "data" / "downloaded"
DATASET, SCHEMA = "GLBX.MDP3", "ohlcv-1m"
SYMBOLS = ("MNQ.v.0", "MES.v.0")
START = "2019-05-01T00:00:00Z"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None, help="exclusive end (ISO); default = end of the dataset's available range")
    ap.add_argument("--execute", action="store_true", help="perform the paid download")
    ap.add_argument("--max-usd", type=float, default=0.0, help="cost guard; required > estimate with --execute")
    args = ap.parse_args()

    key = os.environ.get("DATABENTO_API_KEY") or getpass.getpass("Databento API key: ")
    client = db.Historical(key)
    end = args.end or client.metadata.get_dataset_range(dataset=DATASET)["end"]
    reqs = [{"dataset": DATASET, "symbols": s, "stype_in": "continuous", "schema": SCHEMA, "start": START, "end": end}
            for s in SYMBOLS]
    estimates = [float(client.metadata.get_cost(**r)) for r in reqs]
    total = sum(estimates)
    print(json.dumps({"end": end, "estimates_usd": dict(zip(SYMBOLS, estimates)), "total_usd": round(total, 4)},
                     indent=2))
    if not args.execute:
        return
    if total > args.max_usd:
        raise SystemExit(f"Estimated cost ${total:.4f} exceeds guard --max-usd {args.max_usd:.2f}")

    tag_end = pd.Timestamp(end).strftime("%Y%m%d")
    meta = {"dataset": DATASET, "schema": SCHEMA, "purpose": __doc__.splitlines()[0], "estimated_cost_usd": total,
            "files": [], "api_key_persisted": False}
    for r, est in zip(reqs, estimates):
        df = client.timeseries.get_range(**r).to_df()
        df.index = pd.DatetimeIndex(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("America/New_York")
        df.index.name = "timestamp"
        path = DOWNLOADED / f"{r['symbols'].replace('.', '_')}_1m_20190501_{tag_end}.parquet"
        df.to_parquet(path)
        meta["files"].append({"request": r, "estimated_cost_usd": est, "path": str(path.relative_to(ROOT)),
                              "rows": int(len(df)), "first": str(df.index.min()), "last": str(df.index.max()),
                              "sha256": sha256(path)})
        print(f"Saved {r['symbols']}: {len(df):,} rows -> {path.name}")
    (DOWNLOADED / f"micro_v0_full_20190501_{tag_end}.metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
