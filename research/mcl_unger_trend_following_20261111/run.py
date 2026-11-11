"""Two Unger Academy CL trend-following strategies on MCL. One canonical spec each.

    python -m research.mcl_unger_trend_following_20261111.run
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from research.mcl_mudrex_strategies_20261111.run_mcl_mudrex import (
    ROOT, Sim, eia_days, hhmm, load_base, metrics, sha256, valid_sessions,
)

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json"
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def roll_times(path: Path) -> pd.DatetimeIndex:
    ids = pd.read_parquet(path, columns=["instrument_id"]).sort_index().instrument_id
    ids = ids[~ids.index.duplicated()]
    return ids.index[ids.ne(ids.shift()).to_numpy()][1:].tz_convert("America/New_York")


# ---------------------------------------------------------------- S7 multiday EMA cross

def run_s7(base: pd.DataFrame, rolls: pd.DatetimeIndex, cfg: dict, p: dict) -> tuple[pd.Series, pd.DataFrame]:
    """Returns the daily mark-to-market net PnL (by trading day) and the trade list."""
    i = cfg["instrument"]
    pv = i["point_value_usd"]
    side_cost = i["commission_per_side_usd"] + i["slippage_ticks_per_side"] * i["tick_size"] * pv
    b15 = base.resample("15min", label="left", closed="left").agg(AGG).dropna(subset=["close"])
    fast = b15.close.ewm(span=p["ema_fast"], adjust=False, min_periods=p["ema_fast"]).mean()
    slow = b15.close.ewm(span=p["ema_slow"], adjust=False, min_periods=p["ema_slow"]).mean()
    want = pd.Series(np.where(slow.isna(), 0, np.where(fast > slow, 1, np.where(fast < slow, -1, np.nan))), index=b15.index).ffill()
    want.index = want.index + pd.Timedelta(minutes=15)  # known at the 15m close
    sig = pd.merge_asof(pd.DataFrame({"ts": base.index}), want.rename("want").rename_axis("end").reset_index(),
                        left_on="ts", right_on="end", direction="backward").want.fillna(0).to_numpy()
    o, c = base.open.to_numpy(), base.close.to_numpy()
    # the position during 1m bar k is the target known at its open; fills happen at that open
    prev = np.r_[0, sig[:-1]]
    sides = np.abs(sig - prev)  # 1 = open or close, 2 = reverse
    prev_close = np.r_[o[0], c[:-1]]
    pnl = (prev * (o - prev_close) + sig * (c - o)) * pv - sides * side_cost
    is_roll = np.isin(base.index.asi8, rolls.asi8)
    pnl -= is_roll * np.abs(prev) * 2 * side_cost
    tday = (base.index + pd.Timedelta(hours=6)).date
    daily = pd.Series(pnl, index=base.index).groupby(tday).sum()

    trades, cur = [], None
    for k in np.flatnonzero(sides > 0):
        if cur is not None:
            n_rolls = int(((rolls > cur["entry_ts"]) & (rolls <= base.index[k])).sum())
            gross = cur["side"] * (o[k] - cur["entry"]) * pv
            net = gross - 2 * side_cost * (1 + n_rolls)
            trades.append(dict(strategy="s7_multiday_ema_cross", session=tday[k], entry_ts=cur["entry_ts"], exit_ts=base.index[k],
                               side=cur["side"], entry=cur["entry"], exit=o[k], reason="reverse", gross_usd=gross, net_usd=net, r=np.nan))
        cur = dict(side=int(sig[k]), entry=o[k], entry_ts=base.index[k]) if sig[k] != 0 else None
    return daily, pd.DataFrame(trades)


def metrics_s7(window: str, daily: pd.Series, trades: pd.DataFrame, a: date, z: date) -> dict:
    d = daily[(daily.index >= a) & (daily.index <= z)]
    d = d[[x.weekday() < 5 for x in d.index]]
    t = trades[(trades.session >= a) & (trades.session <= z)]
    eq = d.cumsum()
    wins, losses = t.net_usd[t.net_usd > 0].sum(), -t.net_usd[t.net_usd < 0].sum()
    sd = d.std(ddof=1)
    return dict(strategy="s7_multiday_ema_cross", window=window, sessions=len(d), trades=len(t), net_usd=d.sum(),
                gross_usd=t.gross_usd.sum(), avg_r=np.nan, win_rate=(t.net_usd > 0).mean(), profit_factor=wins / losses,
                sharpe=d.mean() / sd * np.sqrt(252), t_stat=d.mean() / sd * np.sqrt(len(d)),
                max_dd_usd=(eq - eq.cummax().clip(lower=0)).min())


# ---------------------------------------------------------------- S8 intraday Donchian

def build_s8(base: pd.DataFrame, p: dict) -> pd.DataFrame:
    b15 = base.resample("15min", label="left", closed="left").agg(AGG).dropna(subset=["close"])
    n = p["donchian_bars"]
    ch = pd.DataFrame({"up": b15.high.rolling(n).max().to_numpy(), "dn": b15.low.rolling(n).min().to_numpy(),
                       "end15": b15.index + pd.Timedelta(minutes=15)}).dropna()
    out = pd.merge_asof(base.reset_index(names="ts"), ch, left_on="ts", right_on="end15", direction="backward").set_index("ts")
    out["new15"] = out.end15.ne(out.end15.shift())
    return out


def run_s8(frame: pd.DataFrame, sessions: list[date], cfg: dict, p: dict, step: int, eia: set[date]) -> pd.DataFrame:
    sim = Sim(cfg, step)
    tick = cfg["instrument"]["tick_size"]
    a, z = hhmm(cfg["session"]["core_start"]), hhmm(cfg["session"]["core_end"])
    wa, wz = hhmm("09:00"), hhmm("14:00")
    ba, bz = (hhmm(x) for x in cfg["eia"]["blackout_other_strategies"])
    by_day = {d: g for d, g in frame.groupby(frame.index.date)}
    for day in sessions:
        g = by_day[day]
        rows = list(g[(g.index.time >= a) & (g.index.time < z)].itertuples())
        traded = False
        for j, b in enumerate(rows):
            ts, t = b.Index, b.Index.time()
            if np.isnan(b.up):
                continue
            if sim.pos is not None and b.new15:  # ratchet the trailing stop on the opposite channel side
                s = sim.pos["side"]
                sim.pos["stop"] = max(sim.pos["stop"], sim.rnd(b.dn)) if s > 0 else min(sim.pos["stop"], sim.rnd(b.up))
            if sim.pos is None and not traded and wa <= t < wz and not (day in eia and ba <= t < bz):
                buy, sell = sim.rnd(b.up + tick), sim.rnd(b.dn - tick)
                side = 1 if b.high >= buy else -1 if b.low <= sell else 0
                if side:
                    px = max(b.open, buy) if side > 0 else min(b.open, sell)
                    stop = sim.rnd(b.dn) if side > 0 else sim.rnd(b.up)
                    sim.enter(p["name"], ts, side, px, abs(px - stop))
                    sim.pos["stop"] = stop
                    traded = True
                    if j == len(rows) - 1:
                        sim.close(ts, b.close, "flatten")
                    continue  # no same-bar stop check after a stop entry: the channel is far from the entry
            if sim.pos is not None:
                sim.manage(ts, b.open, b.high, b.low)
            if sim.pos is not None and j == len(rows) - 1:
                sim.close(ts, b.close, "flatten")
    return pd.DataFrame(sim.trades)


# ---------------------------------------------------------------- main

def main() -> None:
    own = json.loads(CONFIG.read_text())
    cfg = json.loads((ROOT / own["base_config"]).read_text())
    p7 = own["strategies"]["s7_multiday_ema_cross"]
    p8 = dict(own["strategies"]["s8_intraday_donchian"], name="s8_intraday_donchian")
    data = {}
    for k, v in cfg["data"].items():
        if not isinstance(v, dict):
            continue
        path = ROOT / v["path"]
        if sha256(path) != v["sha256"]:
            raise SystemExit(f"Data hash mismatch: {path}")
        base, step = load_base(path)
        daily7, trades7 = run_s7(base, roll_times(path), cfg, p7)
        frame8 = build_s8(base, p8)
        data[k] = dict(step=step, daily7=daily7, trades7=trades7, frame8=frame8,
                       eia=eia_days(base.index[0].date(), base.index[-1].date()))
    gate, rows, passed = cfg["is_gate"], [], set()
    (HERE / "trades").mkdir(exist_ok=True)
    for window, w in cfg["evaluation_windows"].items():
        d = data[w["data"]]
        a, z = (date.fromisoformat(x) for x in w["range"])
        for name in ("s7_multiday_ema_cross", "s8_intraday_donchian"):
            if window in ("OOS", "FWD") and name not in passed:
                rows.append(dict(strategy=name, window=window, note="not read: failed IS gate"))
                continue
            if name.startswith("s7"):
                m = metrics_s7(window, d["daily7"], d["trades7"], a, z)
                trades = d["trades7"][(d["trades7"].session >= a) & (d["trades7"].session <= z)]
            else:
                sessions = valid_sessions(d["frame8"], cfg, d["step"], a, z)
                trades = run_s8(d["frame8"], sessions, cfg, p8, d["step"], d["eia"])
                m = metrics(name, window, trades, sessions)
            if window == "IS":
                ok = (m["trades"] >= gate["min_trades"] and m.get("net_usd", -1) > gate["min_net_pnl_usd"]
                      and m.get("t_stat", 0) >= gate["min_daily_t_stat"])
                m["is_gate"] = "PASS" if ok else "FAIL"
                if ok:
                    passed.add(name)
            rows.append(m)
            trades.to_csv(HERE / "trades" / f"{window.lower()}_{name}.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(HERE / "results.csv", index=False)
    meta = dict(config_sha256=sha256(CONFIG), base_config_sha256=sha256(ROOT / own["base_config"]))
    (HERE / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
