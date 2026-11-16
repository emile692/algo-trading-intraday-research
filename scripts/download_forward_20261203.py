"""Download the forward 1-minute data (after 2026-10-02) for the prospective reads of audit_20261115.

Same continuous symbols as the frozen history: MNQ.c.0, MGC.v.0, MBT.v.0, from 2026-10-03 00:00 UTC to --end
(exclusive, a date after the last session to read).  Authorised by the user on 2026-11-15 with a 5 USD guard, to be run
only after the close of 2026-12-02 (first horizon H = 20 sessions after 2026-11-03).  Raw files only: nothing is
merged into the frozen history files.  The API key is requested with getpass and never persisted; without
``--execute`` the script only estimates the cost.

    python scripts/download_forward_20261203.py --end 2026-12-03            # estimate
    python scripts/download_forward_20261203.py --end 2026-12-03 --execute  # paid download
"""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import hashlib
import json
from pathlib import Path

import databento as db
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADED = ROOT / "data" / "downloaded"
DATASET, SCHEMA = "GLBX.MDP3", "ohlcv-1m"
SYMBOLS = ("MNQ.c.0", "MGC.v.0", "MBT.v.0")
START = "2026-10-03T00:00:00Z"
EARLIEST_END = dt.date(2026, 12, 3)
MAX_COST_USD = 5.00


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", required=True, help="exclusive end date, YYYY-MM-DD, >= 2026-12-03 and <= today")
    ap.add_argument("--execute", action="store_true", help="perform the paid download")
    args = ap.parse_args()
    end = dt.date.fromisoformat(args.end)
    if end < EARLIEST_END or end > dt.date.today():
        raise SystemExit(f"--end must be between {EARLIEST_END} and today")
    reqs = [{"dataset": DATASET, "symbols": s, "stype_in": "continuous", "schema": SCHEMA, "start": START,
             "end": f"{end.isoformat()}T00:00:00Z"} for s in SYMBOLS]
    client = db.Historical(getpass.getpass("Databento API key: "))
    estimates = [float(client.metadata.get_cost(**r)) for r in reqs]
    total = sum(estimates)
    print(json.dumps({"estimates_usd": estimates, "total_usd": round(total, 6), "guard_usd": MAX_COST_USD}, indent=2))
    if total > MAX_COST_USD:
        raise RuntimeError(f"Estimated cost ${total:.6f} exceeds guard ${MAX_COST_USD:.2f}")
    if not args.execute:
        return
    meta = {"dataset": DATASET, "schema": SCHEMA, "purpose": "audit_20261115 prospective reads (forward_check.py)",
            "estimated_cost_usd": total, "files": [], "api_key_persisted": False}
    tag = end.strftime("%Y%m%d")
    for r, est in zip(reqs, estimates):
        df = client.timeseries.get_range(**r).to_df()
        df.index = pd.DatetimeIndex(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("America/New_York")
        df.index.name = "timestamp"
        path = DOWNLOADED / f"{r['symbols'].replace('.', '_')}_1m_20261003_{tag}_forward.parquet"
        df.to_parquet(path)
        meta["files"].append({"request": r, "estimated_cost_usd": est, "path": str(path.relative_to(ROOT)),
                              "rows": int(len(df)), "first": str(df.index.min()), "last": str(df.index.max()),
                              "sha256": sha256(path)})
        print(f"Saved {r['symbols']}: {len(df):,} rows")
    (DOWNLOADED / f"forward_20261003_{tag}.metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
