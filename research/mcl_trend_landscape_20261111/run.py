"""Exploratory Sharpe landscape of two trend rules on MCL / CL (not a pre-registered test).

    python -m research.mcl_trend_landscape_20261111.run

Rules
  s7_core   EMA fast/slow sign on 15m bars, held only 09:00-14:30 New York
  s7_day    same, held 18:00 -> 16:00 New York (flat before the Topstep 16:10 cut-off)
  s8        intraday Donchian breakout (N closed 15m bars), first breakout from `start`
Windows   PRE_CL (CL 2015-01 -> 2021-07), IS (MCL 2021-07 -> 2023-12), OOS (MCL 2024-01 -> 2026-03).
The FWD window (2026-03-20 -> 2026-11-10) is deliberately not computed.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from research.mcl_mudrex_strategies_20261111.run_mcl_mudrex import (
    ROOT, eia_days, hhmm, load_base, sha256, valid_sessions,
)

HERE = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "research/mcl_mudrex_strategies_20261111/mcl_mudrex_config.json"
S7_FAST, S7_SLOW = [5, 10, 20, 30, 50, 80], [50, 100, 150, 200, 300, 400]
S8_N, S8_START = [4, 8, 12, 16, 20, 30, 40, 60, 90], ["09:00", "09:15", "09:30", "10:00", "10:30"]
WINDOWS = ("PRE_CL", "IS", "OOS")
AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}


class Market:
    def __init__(self, path: Path, cfg: dict):
        i = cfg["instrument"]
        self.pv, self.tick = i["point_value_usd"], i["tick_size"]
        self.side_cost = i["commission_per_side_usd"] + i["slippage_ticks_per_side"] * i["tick_size"] * self.pv
        base, self.step = load_base(path)
        self.base = base
        self.o, self.h, self.l, self.c = (base[x].to_numpy() for x in ("open", "high", "low", "close"))
        idx = base.index
        self.minute = idx.hour.to_numpy() * 60 + idx.minute.to_numpy()
        self.cal = np.array(idx.date)
        self.tday = np.array((idx + pd.Timedelta(hours=6)).date)
        self.b15 = base.resample("15min", label="left", closed="left").agg(AGG).dropna(subset=["close"])
        ends = (self.b15.index + pd.Timedelta(minutes=15)).asi8
        self.i15 = np.searchsorted(ends, idx.asi8, side="right") - 1  # last closed 15m bar at each 1m open
        self.excluded = {date.fromisoformat(x) for x in cfg["session"]["explicit_excluded_sessions"]}
        self.eia = eia_days(idx[0].date(), idx[-1].date())
        self.cfg = cfg
        self._ema: dict[int, np.ndarray] = {}

    def ema(self, n: int) -> np.ndarray:
        if n not in self._ema:
            self._ema[n] = self.b15.close.ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()
        return self._ema[n]

    def on_1m(self, values15: np.ndarray) -> np.ndarray:
        out = values15[np.clip(self.i15, 0, None)].astype(float)
        out[self.i15 < 0] = np.nan
        return out


def stats(daily: pd.Series, trades: int) -> dict:
    sd = daily.std(ddof=1)
    return dict(days=len(daily), trades=trades, net_usd=daily.sum(), sharpe=daily.mean() / sd * np.sqrt(252) if sd else np.nan,
                t_stat=daily.mean() / sd * np.sqrt(len(daily)) if sd else np.nan)


def s7(m: Market, fast: int, slow: int, mode: str) -> tuple[pd.Series, pd.Series]:
    """Daily net PnL and daily entry count. Position = EMA sign while the holding window is open."""
    f, s = m.on_1m(m.ema(fast)), m.on_1m(m.ema(slow))
    want = np.where(np.isnan(s) | np.isnan(f), 0.0, np.sign(f - s))
    if mode == "core":
        open_ = (m.minute >= 9 * 60) & (m.minute < 14 * 60 + 30)
        day = m.cal
    else:
        open_ = (m.minute >= 18 * 60) | (m.minute < 16 * 60)
        day = m.tday
    ok = np.array([d.weekday() < 5 and d not in m.excluded for d in pd.unique(day)])
    allowed = pd.Series(ok, index=pd.unique(day)).reindex(day).to_numpy()
    sig = want * open_ * allowed
    prev = np.r_[0.0, sig[:-1]]
    sides = np.abs(sig - prev)
    prev_close = np.r_[m.o[0], m.c[:-1]]
    pnl = (prev * (m.o - prev_close) + sig * (m.c - m.o)) * m.pv - sides * m.side_cost
    entries = (sig != 0) & (sig != prev)
    g = pd.DataFrame({"pnl": pnl, "n": entries}).groupby(day).sum()
    g = g[[d.weekday() < 5 and d not in m.excluded for d in g.index]]
    return g.pnl, g.n


def s8(m: Market, n: int, start: str, sessions: list[date], slices: dict) -> tuple[pd.Series, int]:
    up = m.on_1m(m.b15.high.rolling(n).max().to_numpy())
    dn = m.on_1m(m.b15.low.rolling(n).min().to_numpy())
    a = hhmm(start)
    wa, wz = a.hour * 60 + a.minute, 14 * 60
    cost = 2 * m.side_cost
    out, trades = {}, 0
    for day in sessions:
        sl = slices[day]
        o, h, l, c, u, d, mn = m.o[sl], m.h[sl], m.l[sl], m.c[sl], up[sl], dn[sl], m.minute[sl]
        can = (mn >= wa) & (mn < wz) & ~np.isnan(u)
        if day in m.eia:
            can &= ~((mn >= 600) & (mn < 660))
        buy, sell = np.round(u + m.tick, 2), np.round(d - m.tick, 2)
        hit = can & ((h >= buy) | (l <= sell))
        out[day] = 0.0
        if not hit.any():
            continue
        k = int(np.argmax(hit))
        side = 1 if h[k] >= buy[k] else -1
        px = max(o[k], buy[k]) if side > 0 else min(o[k], sell[k])
        exit_px = c[-1]
        if k + 1 < len(o):
            if side > 0:
                stop = np.round(np.maximum.accumulate(d[k:]), 2)[1:]
                br = l[k + 1:] <= stop
                if br.any():
                    j = int(np.argmax(br))
                    exit_px = min(o[k + 1 + j], stop[j])
            else:
                stop = np.round(np.minimum.accumulate(u[k:]), 2)[1:]
                br = h[k + 1:] >= stop
                if br.any():
                    j = int(np.argmax(br))
                    exit_px = max(o[k + 1 + j], stop[j])
        out[day] = side * (exit_px - px) * m.pv - cost
        trades += 1
    return pd.Series(out), trades


def main() -> None:
    cfg = json.loads(BASE_CONFIG.read_text())
    markets = {}
    for k, v in cfg["data"].items():
        if isinstance(v, dict):
            path = ROOT / v["path"]
            if sha256(path) != v["sha256"]:
                raise SystemExit(f"Data hash mismatch: {path}")
            markets[k] = Market(path, cfg)
    rows = []
    for window in WINDOWS:
        w = cfg["evaluation_windows"][window]
        m = markets[w["data"]]
        a, z = (date.fromisoformat(x) for x in w["range"])
        sessions = valid_sessions(m.base, cfg, m.step, a, z)
        core = (m.minute >= 9 * 60) & (m.minute < 14 * 60 + 30)
        pos = np.flatnonzero(core)
        cuts = np.flatnonzero(m.cal[pos][1:] != m.cal[pos][:-1]) + 1
        slices = {m.cal[g[0]]: slice(g[0], g[-1] + 1) for g in np.split(pos, cuts)}
        for mode in ("core", "day"):
            for fast in S7_FAST:
                for slow in S7_SLOW:
                    if fast >= slow:
                        continue
                    pnl, n = s7(m, fast, slow, mode)
                    keep = (pnl.index >= a) & (pnl.index <= z)
                    rows.append(dict(strategy=f"s7_{mode}", p1=fast, p2=slow, window=window, **stats(pnl[keep], int(n[keep].sum()))))
        for n in S8_N:
            for start in S8_START:
                pnl, trades = s8(m, n, start, sessions, slices)
                rows.append(dict(strategy="s8", p1=n, p2=start, window=window, **stats(pnl, trades)))
        print(f"done {window}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(HERE / "landscape.csv", index=False)
    pd.set_option("display.width", 200)
    canon = res[((res.strategy.str.startswith("s7")) & (res.p1 == 50) & (res.p2 == 200)) | ((res.strategy == "s8") & (res.p1 == 20) & (res.p2 == "09:00"))]
    print(canon.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
