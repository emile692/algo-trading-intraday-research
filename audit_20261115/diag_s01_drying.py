"""Descriptive diagnostic of S01's drying up (no PnL is computed, no parameter changes).

For each quarter: number of first-pass ORB candidates, ATR(18) of 1-minute bars at the signal (in points), MNQ price,
ATR/price (bp), and the share of candidates below the low band, inside the bands, above the high band, and kept by
the 18-model vote, using the frozen IS thresholds of research/signal_sheet_01_atr18/thresholds_is.csv.

It reads the long MNQ file through 2026-10-02 (the window already read once by the one-shot (O)); it is therefore NOT
part of run_all.  Command: ``python -m audit_20261115.diag_s01_drying``.  Output: outputs/s01_drying_by_quarter.csv.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd

from .common import OUT, ROOT

LONG = ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20190505_20261003.parquet"
THRESHOLDS = ROOT / "research" / "signal_sheet_01_atr18" / "thresholds_is.csv"


def main() -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location("audit_rebuild_s01_diag", ROOT / "scripts" / "rebuild_signal_sheet_01_atr18.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    th = pd.read_csv(THRESHOLDS)
    entry = m.BaselineEntryConfig()
    minute = m.prepare_minute_dataset(LONG, entry, m.ATR_PERIODS)
    minute = m.attach_daily_reference(minute, m.build_daily_reference(minute))
    candidates = m.build_candidate_universe(minute, baseline_entry=entry)
    pre = m.first_pass_signal_rows(candidates, candidates["candidate_base_pass"].fillna(False).astype(bool)).copy()
    votes = {r.model: pd.to_numeric(pre[f"atr_{int(r.atr_period)}"], errors="coerce").between(
        float(r.low_threshold_is), float(r.high_threshold_is), inclusive="both").fillna(False)
        for r in th.itertuples(index=False)}
    pre["votes"] = pd.DataFrame(votes, index=pre.index).sum(axis=1).astype(int)
    a18 = th[th.atr_period == 18]
    lo, hi = float(a18.low_threshold_is.median()), float(a18.high_threshold_is.median())
    price_col = "close" if "close" in pre.columns else "Close"
    pre["atr18"] = pd.to_numeric(pre["atr_18"], errors="coerce")
    pre["price"] = pd.to_numeric(pre[price_col], errors="coerce")
    pre["quarter"] = pd.to_datetime(pre["session_date"]).dt.to_period("Q").astype(str)
    pre["atr_bp"] = pre["atr18"] / pre["price"] * 1e4
    pre["below"] = (pre["atr18"] < lo).astype(float)
    pre["inside"] = pre["atr18"].between(lo, hi).astype(float)
    pre["above"] = (pre["atr18"] > hi).astype(float)
    pre["kept"] = (pre["votes"] >= m.MINIMUM_VOTES).astype(int)
    g = pre.groupby("quarter")
    out = pd.DataFrame({
        "candidates": g.size(),
        "atr18_median_pts": g["atr18"].median().round(2),
        "price_median": g["price"].median().round(0),
        "atr18_over_price_bp": g["atr_bp"].median().round(2),
        "share_below_low_band": g["below"].mean().round(3),
        "share_in_band": g["inside"].mean().round(3),
        "share_above_high_band": g["above"].mean().round(3),
        "kept_by_vote": g["kept"].sum(),
    })
    out.attrs.update(low_band=lo, high_band=hi)
    out.to_csv(OUT / "s01_drying_by_quarter.csv")
    print(f"ATR18 bands (median of the 6 models): {lo:.2f} - {hi:.2f} points")
    print(out.to_string())
    return out


if __name__ == "__main__":
    main()
