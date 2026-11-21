"""Tables of docs/signal_sheets/BOOK.md sections 1, 2, 4 and 5 (nine sleeves, 2024-01-02 -> 2026-03-19, union calendar).

Same computations as scripts/build_signal_sheet_heatmaps.py::build_equity_and_book (book metrics with a zero origin,
per-sleeve volatility / worst day / max drawdown, marginal variance contribution, Pearson correlation of daily PnL,
leave-one-out).  'c0' = lock 2026-11-15 + S04 of the 2026-11-01 lock (must reproduce the published BOOK.md numbers);
'v0' = lock 2026-11-21 + S04 rebuilt on MNQ v.0.  Command: python -m research.mnq_v0_relock_20261121.book_tables
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SLEEVES = ["mnq_orb_vwap_atr", "mnq_volatility_breakout", "mnq_compression_breakout", "mnq_atr_trend_flip",
           "mgc_volatility_breakout", "mgc_gap_continuation", "mbt_rolling_range_breakout",
           "mbt_trend_pullback_ma_structure", "mbt_volatility_breakout"]
SHORT = [f"S{i:02d}" for i in range(1, 10)]
WINDOW = ("2024-01-02", "2026-03-19")


def panel(variant: str) -> pd.DataFrame:
    lock = "locked_eight_sleeve_portfolio_20261115" if variant == "c0" else "locked_eight_sleeve_portfolio_20261121"
    eight = pd.read_csv(ROOT / "research" / lock / "daily_pnl.csv", parse_dates=["session_date"]).set_index("session_date")
    if variant == "c0":
        s04 = pd.read_csv(ROOT / "research" / "locked_nine_sleeve_portfolio_20261101" / "daily_pnl.csv",
                          parse_dates=["session_date"]).set_index("session_date")["mnq_atr_trend_flip"]
    else:
        s04 = pd.read_csv(ROOT / "research" / "signal_sheet_04_cluster_v0" / "daily_results.csv",
                          parse_dates=["session_date"]).groupby("session_date")["daily_pnl_usd"].sum()
        s04 = s04[(s04.index >= WINDOW[0]) & (s04.index <= WINDOW[1])]
    if not set(s04[s04 != 0].index) <= set(eight.index):
        raise RuntimeError("S04 has PnL outside the union calendar")
    data = eight.assign(mnq_atr_trend_flip=s04.reindex(eight.index).fillna(0.0))
    data = data[(data.index >= WINDOW[0]) & (data.index <= WINDOW[1])]
    return data[SLEEVES].fillna(0.0)


def book_metrics(x: pd.Series) -> dict:
    eq = np.r_[0.0, x.cumsum().to_numpy()]
    v = float(x.std(ddof=1))
    return {"net": float(x.sum()), "vol": v, "sharpe": float(np.sqrt(252) * x.mean() / v), "worst": float(x.min()),
            "maxdd": float((eq - np.maximum.accumulate(eq)).min())}


def tables(variant: str) -> dict:
    pnl = panel(variant)
    book = pnl.sum(axis=1)
    full = book_metrics(book)
    eight = book_metrics(pnl.drop(columns="mnq_atr_trend_flip").sum(axis=1))
    equity = pnl.cumsum()
    contrib = pnl.apply(lambda c: c.cov(book)) / book.var()
    per = {s: {"vol": float(pnl[s].std(ddof=1)), "worst": float(pnl[s].min()),
               "maxdd": float((equity[s] - equity[s].cummax()).min()), "risk_share": float(contrib[s])} for s in SLEEVES}
    loo = {}
    for code, s in zip(SHORT, SLEEVES):
        m = book_metrics(pnl.drop(columns=s).sum(axis=1))
        loo[code] = {**m, "d_net": m["net"] - full["net"], "d_dd": m["maxdd"] - full["maxdd"]}
    corr = pnl.corr()
    return {"sessions": int(len(pnl)), "book9": full, "book8": eight, "per_sleeve": per, "leave_one_out": loo,
            "corr": {"S02_S04": float(corr.loc["mnq_volatility_breakout", "mnq_atr_trend_flip"]),
                     "S05_S06": float(corr.loc["mgc_volatility_breakout", "mgc_gap_continuation"]),
                     "S07_S08": float(corr.loc["mbt_rolling_range_breakout", "mbt_trend_pullback_ma_structure"]),
                     "matrix": corr.round(4).values.tolist()}}


def main() -> None:
    c0 = tables("c0")
    checks = {"book9 net": (c0["book9"]["net"], 36240.96), "book9 sharpe": (round(c0["book9"]["sharpe"], 3), 1.509),
              "book9 dd": (c0["book9"]["maxdd"], -6594.74), "book8 net": (c0["book8"]["net"], 30857.96),
              "S01 vol": (round(c0["per_sleeve"]["mnq_orb_vwap_atr"]["vol"], 2), 130.74),
              "S02 share": (round(c0["per_sleeve"]["mnq_volatility_breakout"]["risk_share"], 3), 0.334),
              "S02-S04 corr": (round(c0["corr"]["S02_S04"], 2), 0.57)}
    for k, (got, exp) in checks.items():
        if abs(got - exp) > 0.006:
            raise RuntimeError(f"c.0 BOOK value not reproduced: {k} {got} vs {exp}")
    v0 = tables("v0")
    (Path(__file__).parent / "book_tables.json").write_text(json.dumps({"c0": c0, "v0": v0}, indent=2) + "\n", encoding="utf-8")
    print("c0 reproduced;", json.dumps({k: v0[k] for k in ("book9", "book8")}, indent=1))


if __name__ == "__main__":
    main()
