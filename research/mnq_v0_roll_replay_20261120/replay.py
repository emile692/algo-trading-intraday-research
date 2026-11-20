"""Descriptive diagnostic: the four MNQ sleeves (S01-S04) rerun unchanged on MNQ.v.0 instead of MNQ.c.0.

Question (2026-11-20): how much of the frozen MNQ ledgers depends on the c.0 continuous series, which
  (a) keeps the expiring contract during the roll week (RTH volume ~25 % of normal on its last 4 sessions),
  (b) has no RTH bar on the quarterly expiry Fridays (measured alone in audit_20261115, gap 3: neutral),
  (c) shows an artificial gap on the day it switches contract (roll spread up to ~270 pts / 1.4 % since 2023),
      which enters the daily True Range and so the daily ATR used by S01, S02 and S04.

Variants, both cut at the last timestamp of the frozen c.0 input (2026-03-20 09:29 ET):
  V0_RAW  MNQ.v.0 as downloaded (volume roll, unadjusted): removes (a) and (b); the artificial gap of (c) is still
          there, on the v.0 switch day (about a week before expiry) instead of the Monday after expiry.
  V0_ADJ  MNQ.v.0 back-adjusted additively by the roll spread measured at each v.0 switch (median v.0 - c.0 close
          over the common minutes of the switch session, c.0 still on the old contract): removes (a), (b) and (c).
          Ranges and ATR in points are exact; price LEVELS before each roll are shifted (only matters to an engine
          that uses absolute levels).

The engines are those of audit_20261115/diag_expiry_fridays.py (S01 with the frozen IS thresholds, S02 rebuild,
S03/S04 vote freeze), called unchanged with the data path swapped.  Ledgers are compared with the frozen ones, which
the audit reproduced identically from the c.0 bars.  No parameter is changed; nothing here may be used to retune.

Command: python -m research.mnq_v0_roll_replay_20261120.replay   (about 1 hour; outputs/ next to this file)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_20261115 import diag_expiry_fridays as diag  # noqa: E402  (also puts scripts/ on sys.path)
from audit_20261115.common import load_trades, sha256, window_of  # noqa: E402
from audit_20261115.phase0_engine import KEY_COLS, _norm  # noqa: E402

OUT = HERE / "outputs"
CACHE = HERE / "cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
C0 = diag.C0
V0 = ROOT / "data" / "downloaded" / "MNQ_v_0_1m_20190501_20261120.parquet"
TICK = 0.25
ET = "America/New_York"
CODES = ["S01", "S02", "S03", "S04"]


def _conform(df: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = like["symbol"].iloc[0]
    return out[like.columns].astype(like.dtypes.to_dict())


def build_variants() -> pd.DataFrame:
    c0 = pd.read_parquet(C0)
    v0 = pd.read_parquet(V0)
    v0.index = pd.DatetimeIndex(v0.index).tz_convert(ET)
    v0 = v0.loc[:c0.index.max()]
    raw = _conform(v0, c0)
    raw.to_parquet(CACHE / "MNQ_v0_raw.parquet")

    iid = v0["instrument_id"].astype("int64")
    switches = iid.index[iid.ne(iid.shift()) & iid.shift().notna()]
    rows = []
    for t in switches:
        old_iid = int(iid.loc[:t].iloc[-2])
        window = slice(t, t + pd.Timedelta(hours=23))
        cc = c0.loc[window]
        cc = cc[cc["instrument_id"].astype("int64") == old_iid]["close"]
        vv = v0.loc[window, "close"]
        common = cc.index.intersection(vv.index)
        spread = float(np.median(vv.loc[common] - cc.loc[common])) if len(common) else np.nan
        rows.append({"switch_time": str(t), "old_instrument_id": old_iid, "new_instrument_id": int(iid.loc[t]),
                     "common_minutes": int(len(common)), "roll_spread_points": spread})
    spreads = pd.DataFrame(rows)
    if spreads["roll_spread_points"].isna().any():
        bad = spreads[spreads["roll_spread_points"].isna()]
        raise RuntimeError(f"roll spread not measurable at {bad['switch_time'].tolist()}")

    adj = pd.Series(0.0, index=v0.index)
    for t, s in zip(switches, spreads["roll_spread_points"]):
        adj.loc[adj.index < t] += s
    v0a = v0.copy()
    for col in ("open", "high", "low", "close"):
        v0a[col] = np.round((v0a[col] + adj) / TICK) * TICK
    _conform(v0a, c0).to_parquet(CACHE / "MNQ_v0_adj.parquet")
    return spreads


def compare(code: str, variant: str) -> dict:
    path = CACHE / f"engine_{variant}_{code}.csv"
    if not path.exists():
        diag.MERGED = CACHE / f"MNQ_{variant.lower()}.parquet"
        diag.RUNNERS[code]().to_csv(path, index=False)
    new, old = _norm(pd.read_csv(path)), _norm(load_trades(code))
    lo, hi = window_of(code)["full"]
    keep = lambda t: t[(t["session_date"] >= lo) & (t["session_date"] <= hi)]
    new, old = keep(new), keep(old)
    kn = set(map(tuple, new[KEY_COLS].astype(str).to_numpy()))
    ko = set(map(tuple, old[KEY_COLS].astype(str).to_numpy()))
    is_hi = window_of(code)["is"][1]

    def tot(t: pd.DataFrame, part: str) -> float:
        m = t["session_date"] <= is_hi if part == "is" else t["session_date"] > is_hi
        return round(float(t.loc[m, "net_pnl_usd"].sum()), 2)

    return {"variant": variant, "sleeve": code, "frozen_trades": len(old), "v0_trades": len(new),
            "identical_trades": len(kn & ko), "only_frozen": len(ko - kn), "only_v0": len(kn - ko),
            "frozen_pnl_usd": round(float(old["net_pnl_usd"].sum()), 2),
            "v0_pnl_usd": round(float(new["net_pnl_usd"].sum()), 2),
            "frozen_is_usd": tot(old, "is"), "v0_is_usd": tot(new, "is"),
            "frozen_post_is_usd": tot(old, "post"), "v0_post_is_usd": tot(new, "post")}


def main() -> None:
    spreads = build_variants()
    spreads.to_csv(OUT / "v0_roll_spreads.csv", index=False)
    rows = []
    for variant in ("V0_ADJ", "V0_RAW"):
        for code in CODES:
            r = compare(code, variant)
            rows.append(r)
            print(json.dumps(r), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "replay_summary.csv", index=False)
    meta = {"c0_sha256": sha256(C0), "v0_sha256": sha256(V0), "switches": int(len(spreads)),
            "spread_points_median": float(spreads["roll_spread_points"].median())}
    (OUT / "replay_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
