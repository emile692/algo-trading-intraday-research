"""Metrics of the MNQ signal sheets 01-04 (FS / IS / OOS-reused), c.0 and v.0, with the sheets' conventions.

Sharpe = sqrt(252) x mean / std (ddof 1) of the daily net PnL in USD, days without trade included, on the sleeve's own
session calendar (S01: the MNQ vote calendar of the same data, as fixed by audit_20261115 phase 0); PF = sum of
winning / |sum of losing| net trade PnL; maxDD = worst close-to-close drawdown of the cumulated daily PnL from 0;
worst day = minimum daily PnL.  Run with --check to verify the c.0 values printed in the sheets are reproduced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIRS = {"c0": {1: "signal_sheet_01_atr18", 2: "signal_sheet_02_cluster", 3: "signal_sheet_03_cluster", 4: "signal_sheet_04_cluster"},
        "v0": {1: "signal_sheet_01_atr18_v0", 2: "signal_sheet_02_cluster_v0", 3: "signal_sheet_03_cluster_v0", 4: "signal_sheet_04_cluster_v0"}}
WINDOWS = {1: {"fs": ("2019-05-05", "2026-03-20"), "is": ("2019-05-05", "2024-02-27"), "oos": ("2024-02-28", "2026-03-20")},
           **{n: {"fs": ("2019-05-06", "2026-03-19"), "is": ("2019-05-06", "2023-12-31"), "oos": ("2024-01-01", "2026-03-19")}
              for n in (2, 3, 4)}}
# c.0 values printed in the sheets (FS / IS / OOS): trades, net, sharpe, pf, maxdd
PRINTED_C0 = {1: {"fs": (701, 19416.5, 1.281, 1.339, -2148.5), "is": (514, 13852.0, 1.265, 1.322, -2148.5), "oos": (187, 5564.5, 1.325, 1.389, -875.5)},
              2: {"fs": (1085, 20259.0, 0.848, 1.239, -4790.5), "is": (726, 10980.0, 0.859, 1.221, -4790.5)},
              3: {"fs": (329, 2255.5, 0.432, 1.198, -1202.5), "is": (219, 1663.0, 0.546, 1.239, -1037.0), "oos": (110, 592.5, 0.283, 1.134, -940.5)},
              4: {"fs": (2052, 10075.0, 0.424, 1.079, -4212.5), "is": (1419, 4692.0, 0.354, 1.059, -3063.0), "oos": (633, 5383.0, 0.543, 1.114, -4212.5)}}


def load(variant: str, n: int) -> tuple[pd.Series, pd.DataFrame]:
    d = ROOT / "research" / DIRS[variant][n]
    daily = pd.read_csv(d / "daily_results.csv", parse_dates=["session_date"]).groupby("session_date")["daily_pnl_usd"].sum()
    if n == 1:
        cal = pd.read_csv(ROOT / "research" / DIRS[variant][2] / "daily_results.csv", parse_dates=["session_date"])["session_date"]
        daily = daily.reindex(pd.DatetimeIndex(cal)).fillna(0.0)
    trades = pd.read_csv(d / "trades.csv", parse_dates=["session_date"])
    return daily.astype(float), trades


def block(daily: pd.Series, trades: pd.DataFrame, a: str, b: str) -> dict:
    x = daily[(daily.index >= a) & (daily.index <= b)]
    t = trades[(trades["session_date"] >= a) & (trades["session_date"] <= b)]
    eq = np.r_[0.0, x.cumsum().to_numpy()]
    wins, losses = t.loc[t["net_pnl_usd"] > 0, "net_pnl_usd"].sum(), -t.loc[t["net_pnl_usd"] < 0, "net_pnl_usd"].sum()
    return {"trades": int(len(t)), "net_pnl_usd": float(t["net_pnl_usd"].sum()),
            "sharpe": float(np.sqrt(252) * x.mean() / x.std(ddof=1)), "pf": float(wins / losses),
            "max_dd_usd": float((eq - np.maximum.accumulate(eq)).min()), "worst_day_usd": float(x.min())}


def metrics(variant: str) -> dict:
    out = {}
    for n in (1, 2, 3, 4):
        daily, trades = load(variant, n)
        out[n] = {k: block(daily, trades, *w) for k, w in WINDOWS[n].items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        m = metrics("c0")
        for n, scopes in PRINTED_C0.items():
            for scope, (tr, net, sh, pf, dd) in scopes.items():
                got = m[n][scope]
                ok = (got["trades"] == tr and abs(got["net_pnl_usd"] - net) < 0.01 and round(got["sharpe"], 3) == sh
                      and round(got["pf"], 3) == pf and abs(got["max_dd_usd"] - dd) < 0.01)
                print(n, scope, "OK" if ok else f"MISMATCH {got}")
        return
    res = {"c0": metrics("c0"), "v0": metrics("v0")}
    (Path(__file__).parent / "sheet_metrics.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    for n in (1, 2, 3, 4):
        for scope in ("fs", "is", "oos"):
            c, v = res["c0"][n][scope], res["v0"][n][scope]
            print(n, scope, {k: (round(c[k], 3), round(v[k], 3)) for k in c})


if __name__ == "__main__":
    main()
