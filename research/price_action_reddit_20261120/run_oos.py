"""Single out-of-sample pass (2024-01-01 -> last downloaded session) with the FROZEN default parameters.

Run once only (STRATEGIES.md frozen on 2026-11-21).  Warm-up data from 2023-10-01 gives J-1/J-2 and the ATR on the
first OOS days; no signal before 2024-01-01 is kept.  Validation criteria (decided before any computation):
PF net >= 1.2 on MNQ AND MES, >= 100 trades per instrument, positive expectancy in >= 2 of the 3 years 2024-2026.
Command: python -m research.price_action_reddit_20261120.run_oos
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.price_action.data import SPECS, load_minutes, rth_daily, signal_bars  # noqa: E402
from src.price_action.execution import CostModel, ExecConfig, minutes_by_date, simulate  # noqa: E402
from src.price_action.metrics import mc_drawdown, summarize, table  # noqa: E402
from src.price_action.signals import inside_day_false_break, three_push  # noqa: E402

DATA = {"MNQ": ROOT / "data" / "downloaded" / "MNQ_v_0_1m_20190501_20261120.parquet",
        "MES": ROOT / "data" / "downloaded" / "MES_v_0_1m_20190501_20261120.parquet"}
LOAD_START, OOS_START, OOS_END_EXCL = "2023-10-01", "2024-01-01", "2026-11-20"
FROZEN = {"A": {"tf": 5, "end": "11:00"}, "B": {"tf": 5, "end": "11:00", "n": 2}}
OUT = HERE / "outputs_oos"


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit("outputs_oos/ already exists: the OOS pass is single-shot")
    OUT.mkdir(exist_ok=True)
    cfg, costs = ExecConfig(), CostModel()
    trades = []
    for sym in ("MNQ", "MES"):
        spec = SPECS[sym]
        df = load_minutes(DATA[sym], LOAD_START, OOS_END_EXCL)
        daily, mins = rth_daily(df), minutes_by_date(df)
        bars = signal_bars(df, 5)
        sig = {"A": inside_day_false_break(bars, daily, spec.tick_size, FROZEN["A"]["end"]),
               "B": three_push(bars, spec.tick_size, n=FROZEN["B"]["n"], window_end=FROZEN["B"]["end"])}
        for setup, s in sig.items():
            s = s[s["date"] >= OOS_START]
            t = simulate(s, mins, spec, cfg, costs)
            if t.empty:
                continue
            t["setup_code"], t["instrument"] = setup, sym
            t["year"] = pd.to_datetime(t["date"]).dt.year
            t["side_label"] = t["side"].map({1: "long", -1: "short"})
            trades.append(t)
    trades = pd.concat(trades, ignore_index=True)
    trades.to_csv(OUT / "trades_oos.csv", index=False)
    table(trades, ["setup_code", "instrument"]).to_csv(OUT / "summary_oos.csv", index=False)
    table(trades, ["setup_code", "instrument", "year"]).to_csv(OUT / "by_year_oos.csv", index=False)
    table(trades, ["setup_code", "instrument", "side_label"]).to_csv(OUT / "by_side_oos.csv", index=False)
    table(trades, ["setup_code", "instrument", "year", "side_label"]).to_csv(OUT / "by_year_side_oos.csv", index=False)

    verdict, mc = {}, {}
    for setup in ("A", "B"):
        crit = {}
        for sym in ("MNQ", "MES"):
            t = trades[(trades["setup_code"] == setup) & (trades["instrument"] == sym)]
            s = summarize(t)
            f = t[t["status"] == "filled"].sort_values("signal_time") if not t.empty else t
            years = {int(y): float(g["r_net"].mean()) for y, g in f.groupby("year")} if len(f) else {}
            crit[sym] = {"trades": s.get("trades", 0), "pf_net": s.get("pf_net"),
                         "expectancy_r_by_year": years,
                         "pass_pf": bool(s.get("trades", 0) and s["pf_net"] >= 1.2),
                         "pass_count": bool(s.get("trades", 0) >= 100),
                         "pass_years": sum(v > 0 for v in years.values()) >= 2}
            if len(f):
                mc[f"{setup}_{sym}"] = {"usd_1_contract": mc_drawdown(f["pnl_net_usd"].to_numpy()),
                                        "r": mc_drawdown(f["r_net"].to_numpy())}
        ok = all(c["pass_pf"] and c["pass_count"] and c["pass_years"] for c in crit.values())
        verdict[setup] = {"retained": ok, "criteria": crit}
    (OUT / "verdict_oos.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    (OUT / "mc_oos.json").write_text(json.dumps(mc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
