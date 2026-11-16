"""Phase 4b (P) — prospective reads, exactly as preregistered (preregistration.md, sections 1, 5 and 7).

Usage (once forward data exist):
    python -m audit_20261115.forward_check --daily <forward_daily.csv>

``forward_daily.csv`` holds one row per session of the union calendar strictly after 2026-11-03, a
``session_date`` column and one column per sleeve S01..S09 with the sleeve's virtual net daily PnL in USD
(0.0 on a session of its instrument without trade, empty/NaN when its instrument has no session).  It must be
produced with the frozen definitions of lock ``eight_sleeve_equal_contract_reference_20261115`` (+ S04 shadow),
S01 with the frozen ``thresholds_is.csv``; ``research/frozen_vote_forward_replay_20261101/run_replay.py`` and
``audit_20261115/phase4b_s01_oneshot.py`` are the patterns to follow.  ``baseline_fixed`` may be added as a
column ``S01_baseline_fixed``: it is reported, never judged.

Each horizon H in (20, 60, 120, 250) is read once, when every judged series has reached H sessions; the verdict is
written to outputs/forward_reads/H<H>.json and a second read of the same H is refused.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ACTIVE8, CODES, OUT, write_json

LOOKS = (20, 60, 120, 250)
START_AFTER = pd.Timestamp("2026-11-03")


def max_dd(x: np.ndarray) -> float:
    eq = np.r_[0.0, np.cumsum(x)]
    return float((eq - np.maximum.accumulate(eq)).min())


def read_horizon(daily: pd.DataFrame, H: int, crit: pd.DataFrame) -> dict:
    c = crit[(crit["block"] == 20) & (crit["H"] == H)].set_index("series")
    out = {"H": H, "series": {}}
    for code in CODES:
        s = daily[code].dropna().iloc[:H]
        pnl = float(s.sum())
        thr = float(c.loc[code, "pnl_p05"])
        out["series"][code] = {"first": str(s.index[0].date()), "last": str(s.index[-1].date()), "pnl_usd": pnl,
                               "max_dd_usd": max_dd(s.to_numpy(float)), "pnl_p05": thr,
                               "verdict": "ÉCHEC" if pnl < thr else "COHÉRENT"}
    for name, cols in (("Book 8", ACTIVE8), ("Book 9", CODES)):
        b = daily[cols].fillna(0.0).sum(axis=1).iloc[:H].to_numpy(float)
        pnl, mdd = float(b.sum()), max_dd(b)
        r = c.loc[name]
        fail = pnl < float(r.pnl_p05) or mdd < float(r.mdd_p05)
        above = pnl > float(r.pnl_p95) or mdd > float(r.mdd_p95)
        out["series"][name] = {"pnl_usd": pnl, "max_dd_usd": mdd, "pnl_p05": float(r.pnl_p05), "mdd_p05": float(r.mdd_p05),
                               "verdict": "ÉCHEC" if fail else "COHÉRENT",
                               "au_dessus_de_la_fourchette": bool(above)}
    if "S01_baseline_fixed" in daily:
        s = daily["S01_baseline_fixed"].dropna().iloc[:H]
        out["info_S01_baseline_fixed"] = {"pnl_usd": float(s.sum()), "note": "reported only, not judged"}
    return out


def main(daily_path: str, out_dir: Path | None = None, verify: bool = True) -> list[int]:
    if verify:
        from .run_all import check_prereg
        check_prereg()
    out_dir = out_dir or (OUT / "forward_reads")
    out_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(daily_path, parse_dates=["session_date"]).set_index("session_date").sort_index()
    if (daily.index <= START_AFTER).any():
        raise SystemExit("forward file contains sessions on or before 2026-11-03")
    missing = [c for c in CODES if c not in daily]
    if missing:
        raise SystemExit(f"missing sleeve columns: {missing}")
    crit = pd.read_csv(OUT / "04_criteria.csv")
    reached = min(int(daily[c].notna().sum()) for c in CODES)
    read = []
    for H in LOOKS:
        path = out_dir / f"H{H}.json"
        if H > reached or H > len(daily):
            continue
        if path.exists():
            continue  # already read once; never re-read
        res = read_horizon(daily, H, crit)
        res["read_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        write_json(path, res)
        read.append(H)
    return read


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", required=True)
    args = ap.parse_args()
    done = main(args.daily)
    print(json.dumps({"horizons_read_now": done}))
