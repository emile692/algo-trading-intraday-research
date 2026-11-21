"""Re-check of the catastrophe stop (k x ATR14(J-1) on the net position, decided 2026-11-15) on the MNQ v.0 book.

Same replay as audit_20261115/phase10_cat_stop.py (functions reused unchanged: replay, MGC/MBT ATR), but the MNQ
bars, the MNQ daily ATR and the S01-S03 ledgers come from MNQ v.0 back-adjusted; reference book, aggregated
execution without lot cap, MNQ/MGC 2019-05-06 -> 2026-03-19, MBT 2021-05-03 -> 2026-03-19.  The audit outputs are not
touched.  Command: python -m research.locked_eight_sleeve_portfolio_20261121.catastrophe_stop_check_v0
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

from audit_20261115 import netting as N  # noqa: E402
from audit_20261115 import phase10_cat_stop as P10  # noqa: E402
from src.portfolio.aggregator import RiskLimits, simulate_session  # noqa: E402

RS = N.RS
MNQ_V0 = "data/processed/parquet/MNQ_v_0_adj_1m_20190505_20260320.parquet"
V0_TRADES = {"S01": "research/signal_sheet_01_atr18_v0/trades.csv", "S02": "research/signal_sheet_02_cluster_v0/trades.csv",
             "S03": "research/signal_sheet_03_cluster_v0/trades.csv"}


def v0_spec() -> dict:
    s = json.loads(json.dumps(N.spec()))
    for sid, path in V0_TRADES.items():
        s["sleeves"][sid]["trades"] = path
    s["instruments"]["MNQ"]["bars_1m"] = MNQ_V0
    return s


def load_bars(spec: dict, cutoff: str) -> dict:
    import pyarrow.parquet as pq
    end = pd.Timestamp(cutoff, tz=RS.ET) + pd.Timedelta(days=1)
    out = {}
    for name in RS.INSTRUMENTS:
        path = ROOT / spec["instruments"][name]["bars_1m"]
        col = next(c for c in ("timestamp", "ts_event") if c in pq.read_schema(path).names)
        tz = str(pq.read_schema(path).field(col).type.tz)
        b = pd.read_parquet(path, columns=["open", "high", "low", "close"], filters=[(col, "<", end.tz_convert(tz))])
        b.index = b.index.tz_convert("UTC")
        out[name] = b[~b.index.duplicated(keep="last")].sort_index()
    return out


def mnq_atr_j1() -> pd.Series:
    """Same construction as phase8_regimes.daily_bars + phase10.atr_j1, on the v.0 file."""
    b = pd.read_parquet(ROOT / MNQ_V0, columns=["open", "high", "low", "close"])
    b.index = pd.DatetimeIndex(b.index).tz_convert("America/New_York")
    b = b[~b.index.duplicated(keep="last")].sort_index()
    m = b.index.hour * 60 + b.index.minute
    r = b[(m >= 570) & (m < 960) & (b.index.dayofweek < 5)]
    g = r.groupby(r.index.date)
    d = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})
    d.index = pd.to_datetime(d.index)
    d = d[d.index <= "2026-03-19"]
    tr = pd.concat([d["high"] - d["low"], (d["high"] - d["close"].shift()).abs(), (d["low"] - d["close"].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean().shift(1)


def main() -> None:
    spec = v0_spec()
    trades = RS.load_trades(spec)
    dates = RS.union_sessions(RS.load_calendars(spec)[0], RS.d(P10.WINDOW[0]), RS.d(P10.WINDOW[1]))
    cube = RS.BarCube(load_bars(spec, "2026-03-19"), dates)
    events = RS.build_events(trades, np.where(trades.sleeve == "S01", trades.qty_native, 1).astype(int))
    sp = RS.instrument_specs(spec, 10)
    res = [simulate_session(cube.session(x), events.get(x, RS.EMPTY_EVENTS), RS.SLEEVE_INSTRUMENT, sp, RiskLimits(), keep_paths=True)
           for x in dates]
    atrs = {"MNQ": mnq_atr_j1(), "MGC": P10.atr_j1("MGC"), "MBT": P10.atr_j1("MBT")}
    summary, triggers = [], []
    for k in P10.KS:
        bounds = []
        for x, r in zip(dates, res):
            ts, cs, bound = pd.Timestamp(x), cube.session(x), 0.0
            for i, n in enumerate(P10.NAMES):
                a = float(atrs[n].get(ts, np.nan))
                fl = [f for f in r.fills if f["instrument"] == i]
                if fl and np.isfinite(a):
                    p, mx = 0, 0
                    for f in sorted(fl, key=lambda f: f["minute"]):
                        p += int(f["delta"])
                        mx = max(mx, abs(p))
                    bound += mx * k * a * sp[i].point_value + mx * (sp[i].slippage_price * sp[i].point_value + sp[i].commission_per_side)
                rec = P10.replay(r, cs, i, a, k, sp)
                if rec:
                    triggers.append({"k": k, "date": str(x), "instrument": n, "atr_j1": a, **rec,
                                     "pnl_evite": rec["stopped_pnl"] - rec["actual_pnl"]})
            bounds.append(bound)
        b = pd.Series(bounds)
        row = {"k": k, "sessions": len(b), "bound_p50": float(b.median()), "bound_p99": float(b.quantile(0.99)),
               "bound_max": float(b.max())}
        for n in P10.NAMES:
            row[f"triggers_{n}"] = sum(1 for t in triggers if t["k"] == k and t["instrument"] == n)
        summary.append(row)
    audit = json.loads((ROOT / "audit_20261115" / "outputs" / "10_summary.json").read_text(encoding="utf-8"))
    out = {"book": "eight_sleeve_equal_contract_reference_20261121 (MNQ v.0 back-adjusted)", "window": list(P10.WINDOW),
           "sessions": len(dates), "net_usd": float(sum(r.final for r in res)), "summary": summary,
           "c0_audit_summary": audit["summary"], "triggers": triggers}
    (HERE / "catastrophe_stop_check_v0.json").write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
