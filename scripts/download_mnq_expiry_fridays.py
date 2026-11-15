"""Download MNQ.v.0 1-minute bars for the quarterly expiry Fridays missing from the MNQ.c.0 file.

The frozen file data/processed/parquet/MNQ_c_0_1m_20190505_20261003.parquet has no bar at all on 29 of the 30
quarterly expiry Fridays between 2019-05-06 and 2026-10-02, so the MNQ sleeves were never backtested on those days
(audit_20261115, gap 3).  For each missing Friday this script requests MNQ.v.0 (volume roll, i.e. the contract
production trades that day) from Thursday 00:00 UTC to Saturday 00:00 UTC: the Thursday overlap with MNQ.c.0
(still the expiring contract) measures the roll spread, the Friday session is what is missing.

Nothing is merged into the frozen files.  Output: data/downloaded/MNQ_v_0_1m_expiry_fridays_20190621_20260918.parquet
and its metadata JSON.  The API key is requested with getpass and never persisted.  By default the script only
estimates the cost; ``--execute`` performs the paid download (guard 1 USD; authorised by the user on 2026-11-15).
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
MNQ_C0 = ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20190505_20261003.parquet"
OUT = ROOT / "data" / "downloaded" / "MNQ_v_0_1m_expiry_fridays_20190621_20260918.parquet"
META = OUT.with_suffix(".metadata.json")
DATASET, SCHEMA, SYMBOL = "GLBX.MDP3", "ohlcv-1m", "MNQ.v.0"
MAX_COST_USD = 1.00


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def missing_expiry_fridays() -> list[pd.Timestamp]:
    """Third Fridays of Mar/Jun/Sep/Dec without any RTH bar in the frozen MNQ.c.0 file (timestamps only)."""
    ts = pd.DatetimeIndex(pd.read_parquet(MNQ_C0, columns=[]).index).tz_convert("America/New_York")
    minutes = ts.hour * 60 + ts.minute
    rth_days = set(ts[(minutes >= 570) & (minutes < 960)].date)
    out = []
    for year in range(2019, 2027):
        for month in (3, 6, 9, 12):
            d = pd.Timestamp(year, month, 15)
            third = d + pd.Timedelta(days=(4 - d.dayofweek) % 7)
            if pd.Timestamp("2019-05-06") <= third <= pd.Timestamp("2026-10-02") and third.date() not in rth_days:
                out.append(third)
    return out


def requests() -> list[dict]:
    return [{"dataset": DATASET, "symbols": SYMBOL, "stype_in": "continuous", "schema": SCHEMA,
             "start": (f - pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
             "end": (f + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")} for f in missing_expiry_fridays()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="perform the paid download")
    args = ap.parse_args()
    reqs = requests()
    print(f"{len(reqs)} missing expiry Fridays: {reqs[0]['start'][:10]} .. {reqs[-1]['end'][:10]}")
    client = db.Historical(getpass.getpass("Databento API key: "))
    estimates = [float(client.metadata.get_cost(**r)) for r in reqs]
    total = sum(estimates)
    print(json.dumps({"requests": len(reqs), "total_usd": round(total, 6), "guard_usd": MAX_COST_USD}, indent=2))
    if total > MAX_COST_USD:
        raise RuntimeError(f"Estimated cost ${total:.6f} exceeds guard ${MAX_COST_USD:.2f}")
    if not args.execute:
        return
    frames = []
    for r in reqs:
        df = client.timeseries.get_range(**r).to_df()
        df.index = pd.DatetimeIndex(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("America/New_York")
        df.index.name = "timestamp"
        frames.append(df)
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.to_parquet(OUT)
    meta = {"dataset": DATASET, "schema": SCHEMA, "symbol": SYMBOL, "purpose": "audit_20261115 gap 3, descriptive only",
            "fridays": [str(f.date()) for f in missing_expiry_fridays()], "requests": reqs, "estimated_cost_usd": total,
            "paid_get_range_calls": len(reqs), "rows": int(len(out)), "sha256": sha256(OUT),
            "source_c0_file": str(MNQ_C0.relative_to(ROOT)), "source_c0_sha256": sha256(MNQ_C0), "api_key_persisted": False}
    META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(out):,} rows to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
