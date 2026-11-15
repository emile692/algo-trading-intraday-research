"""Phase 0 — reproduction of the published figures of the 2026-11-01 reference.

Three levels:
  A. metrics recomputed from the frozen result files (daily_results.csv / trades.csv / lock daily_pnl.csv);
  B. full re-execution of every sleeve from the raw minute bars with the repository engines, in memory,
     compared trade by trade with the frozen ledgers (see phase0_engine.py, cached);
  C. instrument netting (src/portfolio/aggregator.py) on 2024-01-02 -> 2026-03-19, bars cut at 2026-03-19.
Nothing in the repository is written.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .common import (ACTIVE8, BOOK_WINDOW, CODES, NAMES, OUT, ROOT, between, fr, load_daily, load_trades,
                     lock_daily_nine, max_dd, md_table, profit_factor, sharpe, window_of, write_json)

TOL_PNL, TOL_SHARPE = 1.0, 0.005

EXPECTED = {
    ("S01", "full"): dict(trades=701, pnl=19416.50, sharpe=1.156, pf=1.339, maxdd=-2148.50),
    ("S02", "full"): dict(trades=1085, pnl=20259.0, sharpe=0.873),
    ("S03", "full"): dict(trades=329, pnl=2255.50, sharpe=0.432),
    ("S04", "full"): dict(trades=2052, pnl=10075.0, sharpe=0.424),
    ("S05", "full"): dict(trades=197, pnl=3135.50, sharpe=0.370),
    ("S06", "full"): dict(trades=231, pnl=7030.50, sharpe=0.677),
    ("S07", "full"): dict(trades=461, pnl=2082.48, sharpe=0.473),
    ("S08", "full"): dict(trades=1142, pnl=3900.06, sharpe=0.642),
    ("S09", "full"): dict(trades=668, pnl=4916.24, sharpe=0.874),
    ("S09", "val"): dict(pnl=-73.22, sharpe=-0.040),
}
EXPECTED_BOOK = dict(sessions=564, pnl=37044.22, sharpe=1.539, maxdd=-6305.18, worst=-1664.46)
EXPECTED_LOO = {  # removed sleeve -> (pnl, sharpe, maxdd)
    "S01": (30715.22, 1.324, -6670.64), "S02": (27765.22, 1.595, -5704.60), "S03": (36451.72, 1.559, -6543.68),
    "S04": (31661.22, 1.751, -5425.58), "S05": (35256.22, 1.522, -6123.18), "S06": (32946.22, 1.433, -6416.78),
    "S07": (34689.18, 1.482, -6171.16), "S08": (32699.12, 1.419, -6368.38), "S09": (34169.64, 1.509, -6279.24),
}
EXPECTED_NETTING = dict(aggregated=31180.0, separate=31152.0)


def _rows_for(label: str, exp: dict, got: dict) -> list[dict]:
    rows = []
    tol = {"trades": 0, "pnl": TOL_PNL, "sharpe": TOL_SHARPE, "pf": 0.0005, "maxdd": TOL_PNL, "worst": TOL_PNL,
           "sessions": 0}
    for k, v in exp.items():
        g = got[k]
        diff = g - v
        rounding = 0.0005 if k in ("sharpe", "pf") else 0.0
        ok = abs(diff) <= tol[k] + rounding + 1e-9 if k not in ("trades", "sessions") else diff == 0
        # Published Sharpe/PF are rounded to 3 decimals: compare the rounded value as well.
        if k in ("sharpe", "pf"):
            ok = abs(round(g, 3) - v) <= tol[k] + 1e-12 if k == "sharpe" else abs(round(g, 3) - v) < 1e-9
        rows.append({"élément": label, "métrique": k, "attendu": v, "obtenu": g, "écart": diff, "ok": bool(ok)})
    return rows


def level_a() -> tuple[list[dict], dict]:
    rows, detail = [], {}
    for code in CODES:
        daily, trades = load_daily(code), load_trades(code)
        w = window_of(code)
        for scope in ("full", "is", "val"):
            d = between(daily, *w[scope])
            t = trades[trades["session_date"].between(pd.Timestamp(w[scope][0]), pd.Timestamp(w[scope][1]))]
            got = dict(trades=len(t), pnl=float(d.sum()), sharpe=sharpe(d), pf=profit_factor(t["net_pnl_usd"]),
                       maxdd=max_dd(d), sessions=len(d))
            detail[f"{code}_{scope}"] = got
            if (code, scope) in EXPECTED:
                rows += _rows_for(f"{code} {scope}", EXPECTED[(code, scope)], got)
    # S01 convention check: its source calendar contains Sunday rows (no RTH session).
    s01 = load_daily("S01")
    rth = s01[s01.index.dayofweek < 5]
    detail["S01_full_weekdays_only"] = dict(sessions=len(rth), pnl=float(rth.sum()), sharpe=sharpe(rth))
    s02_cal = load_daily("S02").index
    on_s02 = s01.reindex(s02_cal).fillna(0.0)
    detail["S01_on_S02_calendar"] = dict(sessions=len(on_s02), pnl=float(on_s02.sum()), sharpe=sharpe(on_s02),
                                         missing_trade_days=int((~s01[s01 != 0].index.isin(s02_cal)).sum()))
    return rows, detail


def book_metrics(df: pd.DataFrame, cols: list[str]) -> dict:
    p = df[cols].sum(axis=1)
    return dict(sessions=len(p), pnl=float(p.sum()), sharpe=sharpe(p), maxdd=max_dd(p), worst=float(p.min()))


def rebuild_lock_daily() -> pd.DataFrame:
    """Rebuild the nine-sleeve lock daily PnL from its declared sources (relock_nine_sleeves.py logic)."""
    six = pd.read_csv(ROOT / "research" / "portfolio_integration_phase1" / "daily_sleeve_pnl.csv",
                      parse_dates=["session_date"]).set_index("session_date")
    cal = six.index
    out = pd.DataFrame(index=cal)
    out["S01"] = six["mnq_orb_vwap_atr"]
    for code in CODES[1:]:
        out[code] = load_daily(code).reindex(cal).fillna(0.0)
    return out


def level_book() -> tuple[list[dict], dict]:
    lock = lock_daily_nine()
    rows = _rows_for("Book 9 (lock CSV)", EXPECTED_BOOK, book_metrics(lock, CODES))
    rebuilt = rebuild_lock_daily()
    diff = float((rebuilt[CODES] - lock[CODES]).abs().max().max())
    rows.append({"élément": "Book 9 reconstruit depuis les sources", "métrique": "écart quotidien absolu max vs lock",
                 "attendu": 0.0, "obtenu": diff, "écart": diff, "ok": diff < 1e-6})
    s01_atr18 = load_daily("S01").reindex(lock.index).fillna(0.0)
    d01 = float((s01_atr18 - lock["S01"]).abs().max())
    rows.append({"élément": "S01 lock vs S01 signal_sheet_01_atr18", "métrique": "écart quotidien absolu max",
                 "attendu": 0.0, "obtenu": d01, "écart": d01, "ok": d01 < 1e-6})
    for removed, (pnl, sh, dd) in EXPECTED_LOO.items():
        cols = [c for c in CODES if c != removed]
        rows += _rows_for(f"LOO sans {removed}", dict(pnl=pnl, sharpe=sh, maxdd=dd), book_metrics(lock, cols))
    detail = {"book9": book_metrics(lock, CODES), "book8": book_metrics(lock, ACTIVE8),
              "rebuilt_max_abs_diff": diff, "s01_sources_max_abs_diff": d01}
    return rows, detail


def level_netting() -> tuple[list[dict], dict]:
    from .netting import run_book, trade_sum_separate
    sep = trade_sum_separate(*BOOK_WINDOW)
    agg = run_book(*BOOK_WINDOW, lot_cap_units=50)
    rows = [
        {"élément": "Netting 2024-01→2026-03", "métrique": "PnL séparé (somme des trades)", "attendu": 31152.0,
         "obtenu": sep["total"], "écart": sep["total"] - 31152.0, "ok": abs(sep["total"] - 31152.0) < 1.0},
        {"élément": "Netting 2024-01→2026-03", "métrique": "PnL agrégé (moteur de netting, cap 5 lots)",
         "attendu": 31180.0, "obtenu": agg["total"], "écart": agg["total"] - 31180.0,
         "ok": abs(agg["total"] - 31180.0) < 1.0},
    ]
    return rows, {"separate": sep, "aggregated": {k: v for k, v in agg.items() if k != "table"}}


def main(engine: bool = True) -> dict:
    rows_a, det_a = level_a()
    rows_b, det_b = level_book()
    rows_c, det_c = level_netting()
    rows_e, det_e = [], {}
    if engine:
        from .phase0_engine import compare_all
        rows_e, det_e = compare_all()
    table = pd.DataFrame(rows_a + rows_b + rows_c + rows_e)
    table.to_csv(OUT / "00_reproduction.csv", index=False)
    result = {"all_ok": bool(table["ok"].all()), "n_checks": len(table), "n_fail": int((~table["ok"]).sum()),
              "level_a": det_a, "book": det_b, "netting": det_c, "engine": det_e}
    write_json(OUT / "00_reproduction.json", result)
    return result


if __name__ == "__main__":
    r = main(engine=False)
    print(json.dumps({k: r[k] for k in ("all_ok", "n_checks", "n_fail")}))
