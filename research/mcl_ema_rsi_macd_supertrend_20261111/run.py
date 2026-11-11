"""EMA / RSI / MACD / Supertrend intraday plan on MCL. One canonical spec.

    python -m research.mcl_ema_rsi_macd_supertrend_20261111.run
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from research.mcl_mudrex_strategies_20261111.run_mcl_mudrex import (
    ROOT, Sim, eia_days, hhmm, load_base, metrics, sha256, valid_sessions, wilder,
)

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json"


def supertrend(bars: pd.DataFrame, n: int, mult: float) -> tuple[np.ndarray, np.ndarray]:
    c = bars.close
    tr = pd.concat([bars.high - bars.low, (bars.high - c.shift()).abs(), (bars.low - c.shift()).abs()], axis=1).max(axis=1)
    atr = wilder(tr, n).to_numpy()
    hl2 = ((bars.high + bars.low) / 2).to_numpy()
    close = c.to_numpy()
    up, lo = hl2 + mult * atr, hl2 - mult * atr
    line, side = np.full(len(bars), np.nan), np.zeros(len(bars), dtype=int)
    fu = fl = np.nan
    d = 1
    for i in range(len(bars)):
        if np.isnan(atr[i]):
            continue
        if np.isnan(fu):
            fu, fl = up[i], lo[i]
        else:
            fu = up[i] if (up[i] < fu or close[i - 1] > fu) else fu
            fl = lo[i] if (lo[i] > fl or close[i - 1] < fl) else fl
            if d == 1 and close[i] < fl:
                d = -1
            elif d == -1 and close[i] > fu:
                d = 1
        side[i], line[i] = d, fl if d == 1 else fu
    return line, side


def build(base: pd.DataFrame, p: dict) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    b15 = base.resample("15min", label="left", closed="left").agg(agg).dropna(subset=["close"])
    e = [b15.close.ewm(span=n, adjust=False, min_periods=n).mean() for n in p["ema"]]
    t15 = pd.DataFrame({
        "trend": np.where((b15.close > e[0]) & (e[0] > e[1]) & (e[1] > e[2]), 1,
                          np.where((b15.close < e[0]) & (e[0] < e[1]) & (e[1] < e[2]), -1, 0)),
        "end15": b15.index + pd.Timedelta(minutes=15)})[e[2].notna().to_numpy()]

    b5 = base.resample("5min", label="left", closed="left").agg(agg).dropna(subset=["close"])
    c = b5.close
    d = c.diff()
    rsi = 100 - 100 / (1 + wilder(d.clip(lower=0), p["rsi_period"]) / wilder(-d.clip(upper=0), p["rsi_period"]))
    f, s, g = p["macd"]
    macd = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
    msig = macd.ewm(span=g, adjust=False).mean()
    hv = b5.volume > b5.volume.rolling(p["volume_avg_bars"]).mean().shift(1)
    line, side = supertrend(b5, *p["supertrend"])
    s5 = pd.DataFrame({
        "st_line": line, "st_side": side,
        "sig_l": ((rsi > p["rsi_long"]) & (macd > msig) & (macd.shift() <= msig.shift()) & hv & (side == 1)).to_numpy(),
        "sig_s": ((rsi < p["rsi_short"]) & (macd < msig) & (macd.shift() >= msig.shift()) & hv & (side == -1)).to_numpy(),
        "end5": b5.index + pd.Timedelta(minutes=5)}).dropna(subset=["st_line"])

    out = pd.merge_asof(base.reset_index(names="ts"), s5, left_on="ts", right_on="end5", direction="backward")
    out = pd.merge_asof(out, t15, left_on="ts", right_on="end15", direction="backward").set_index("ts")
    out["new5"] = out.end5.ne(out.end5.shift())
    return out


def run(frame: pd.DataFrame, sessions: list[date], cfg: dict, p: dict, step: int, eia: set[date]) -> pd.DataFrame:
    sim = Sim(cfg, step)
    a, z = hhmm(cfg["session"]["core_start"]), hhmm(cfg["session"]["core_end"])
    wa, wz = (hhmm(x) for x in p["entry_window"])
    ba, bz = (hhmm(x) for x in cfg["eia"]["blackout_other_strategies"])
    by_day = {d: g for d, g in frame.groupby(frame.index.date)}
    for day in sessions:
        g = by_day[day]
        rows = list(g[(g.index.time >= a) & (g.index.time < z)].itertuples())
        for j, b in enumerate(rows):
            ts, t = b.Index, b.Index.time()
            if np.isnan(b.st_line) or np.isnan(b.trend):
                continue
            if sim.pos is not None and b.new5 and b.st_side == sim.pos["side"]:  # ratchet the trailing stop
                s = sim.pos["side"]
                new = sim.rnd(b.st_line)
                sim.pos["stop"] = max(sim.pos["stop"], new) if s > 0 else min(sim.pos["stop"], new)
            if sim.pos is None and b.new5 and wa <= t < wz and not (day in eia and ba <= t < bz):
                side = 1 if (b.sig_l and b.trend == 1) else -1 if (b.sig_s and b.trend == -1) else 0
                dist = side * (b.open - b.st_line)
                if side and dist > 0:
                    sim.enter(p["name"], ts, side, b.open, dist)
                    sim.pos["stop"] = sim.rnd(b.st_line)
            if sim.pos is not None:
                sim.manage(ts, b.open, b.high, b.low)
            if sim.pos is not None and j == len(rows) - 1:
                sim.close(ts, b.close, "flatten")
    return pd.DataFrame(sim.trades)


def main() -> None:
    own = json.loads(CONFIG.read_text())
    cfg = json.loads((ROOT / own["base_config"]).read_text())
    p = own["strategy"]
    frames = {}
    for k, v in cfg["data"].items():
        if not isinstance(v, dict):
            continue
        path = ROOT / v["path"]
        if sha256(path) != v["sha256"]:
            raise SystemExit(f"Data hash mismatch: {path}")
        base, step = load_base(path)
        frame = build(base, p)
        frames[k] = (frame, step, eia_days(frame.index[0].date(), frame.index[-1].date()))
    gate, rows, passed = cfg["is_gate"], [], False
    (HERE / "trades").mkdir(exist_ok=True)
    for window, w in cfg["evaluation_windows"].items():
        if window in ("OOS", "FWD") and not passed:
            rows.append(dict(strategy=p["name"], window=window, note="not read: failed IS gate"))
            continue
        frame, step, eia = frames[w["data"]]
        sessions = valid_sessions(frame, cfg, step, *(date.fromisoformat(x) for x in w["range"]))
        trades = run(frame, sessions, cfg, p, step, eia)
        m = metrics(p["name"], window, trades, sessions)
        if window == "IS":
            passed = (m["trades"] >= gate["min_trades"] and m.get("net_usd", -1) > gate["min_net_pnl_usd"]
                      and m.get("t_stat", 0) >= gate["min_daily_t_stat"])
            m["is_gate"] = "PASS" if passed else "FAIL"
        rows.append(m)
        trades.to_csv(HERE / "trades" / f"{window.lower()}_{p['name']}.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(HERE / "results.csv", index=False)
    meta = dict(config_sha256=sha256(CONFIG), base_config_sha256=sha256(ROOT / own["base_config"]))
    (HERE / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
