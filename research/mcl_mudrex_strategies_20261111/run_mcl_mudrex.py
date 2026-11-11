"""Five Mudrex crude-oil strategies transposed to MCL intraday. One canonical spec each.

Usage:
    python -m research.mcl_mudrex_strategies_20261111.run_mcl_mudrex
    python -m research.mcl_mudrex_strategies_20261111.run_mcl_mudrex --data X.parquet --out DIR --smoke

--smoke runs every strategy on whatever bars are supplied (no hash check, no IS/OOS
split, no gate). It validates the pipeline only and never produces a verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG = HERE / "mcl_mudrex_config.json"
NY = "America/New_York"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hhmm(text: str) -> time:
    return time(int(text[:2]), int(text[3:]))


def wilder(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


# ---------------------------------------------------------------- data

def load_base(path: Path) -> tuple[pd.DataFrame, int]:
    df = pd.read_parquet(path).sort_index()
    df = df[~df.index.duplicated()]
    if "instrument_id" in df.columns:  # additive back-adjustment of contract rolls
        roll = df.instrument_id.ne(df.instrument_id.shift()).to_numpy()
        roll[0] = False
        gap = np.where(roll, df.open.to_numpy() - df.close.shift().to_numpy(), 0.0)
        shift = gap[::-1].cumsum()[::-1] - gap  # sum of gaps strictly after each bar
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] + shift
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df.index = (df.index.tz_localize("UTC") if df.index.tz is None else df.index).tz_convert(NY)
    step = int(pd.Series(df.index[1:] - df.index[:-1]).mode().iloc[0].total_seconds() // 60)
    if step not in (1, 5):
        raise RuntimeError(f"base bars must be 1m or 5m, got {step}m")
    return df, step


def build_frames(base: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach to each base bar the indicators of the last *closed* 15m bar."""
    st = cfg["strategies"]
    base = base.copy()
    tday = (base.index + pd.Timedelta(hours=6)).date  # 18:00 ET roll
    base["tday"] = tday
    tp = (base.high + base.low + base.close) / 3
    pv = (tp * base.volume).groupby(tday).cumsum()
    vv = base.volume.groupby(tday).cumsum()
    base["vwap"] = (pv / vv.where(vv > 0)).ffill()

    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "vwap": "last", "tday": "last"}
    sig = base.resample("15min", label="left", closed="left").agg(agg).dropna(subset=["close"])
    c = sig.close
    s1, s2, s3, s4 = (st[k] for k in ("s1_trend_sma_pullback", "s2_vwap_rsi_pullback",
                                      "s3_breakout_retest", "s4_range_rsi_bb_reversion"))
    sig["sma_f"] = c.rolling(s1["sma_fast"]).mean()
    sig["sma_s"] = c.rolling(s1["sma_slow"]).mean()
    f, s, g = s1["macd"]
    macd = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
    msig = macd.ewm(span=g, adjust=False).mean()
    d = c.diff()
    rs = wilder(d.clip(lower=0), 14) / wilder(-d.clip(upper=0), 14)
    sig["rsi"] = 100 - 100 / (1 + rs)
    tr = pd.concat([sig.high - sig.low, (sig.high - c.shift()).abs(), (sig.low - c.shift()).abs()], axis=1).max(axis=1)
    sig["atr"] = wilder(tr, 14)
    n, k = s4["bollinger"]
    mid, sd = c.rolling(n).mean(), c.rolling(n).std(ddof=0)
    sig["bb_mid"] = mid
    vol_avg = sig.volume.rolling(20).mean().shift(1)

    daily = sig.groupby("tday").agg(h=("high", "max"), l=("low", "min")).shift(1)
    sig["pdh"] = sig.tday.map(daily.h)
    sig["pdl"] = sig.tday.map(daily.l)

    up = sig.sma_f > sig.sma_s
    dn = sig.sma_f < sig.sma_s
    sig["trend"] = np.where(up, 1, np.where(dn, -1, 0))
    # S1
    sig["s1_long"] = up & (sig.low <= sig.sma_f) & (c > sig.sma_f)
    sig["s1_short"] = dn & (sig.high >= sig.sma_f) & (c < sig.sma_f)
    sig["s1_xl"] = (c < sig.sma_f) | ((macd < msig) & (macd.shift() >= msig.shift()))
    sig["s1_xs"] = (c > sig.sma_f) | ((macd > msig) & (macd.shift() <= msig.shift()))
    # S2
    above, below = c > sig.vwap, c < sig.vwap
    same_day = sig.tday == sig.tday.shift(1)
    sig["s2_arm_l"] = above & above.shift(1, fill_value=False) & ~above.shift(2, fill_value=True) & same_day & (sig.rsi > s2["rsi_threshold"])
    sig["s2_arm_s"] = below & below.shift(1, fill_value=False) & ~below.shift(2, fill_value=True) & same_day & (sig.rsi < 100 - s2["rsi_threshold"])
    sig["s2_off_l"], sig["s2_off_s"] = below, above
    # S3
    hv = sig.volume > vol_avg
    sig["s3_arm_l"] = (c > sig.pdh) & (c.shift() <= sig.pdh) & hv
    sig["s3_arm_s"] = (c < sig.pdl) & (c.shift() >= sig.pdl) & hv
    sig["s3_off_l"], sig["s3_off_s"] = c < sig.pdh, c > sig.pdl
    # S4
    rng = (sig.sma_f - sig.sma_s).abs() <= sig.atr
    sig["s4_long"] = rng & (c < mid - k * sd) & (sig.rsi < s4["rsi_low"])
    sig["s4_short"] = rng & (c > mid + k * sd) & (sig.rsi > s4["rsi_high"])

    cols = ["atr", "bb_mid", "pdh", "pdl", "trend"] + [x for x in sig.columns if x[:3] in ("s1_", "s2_", "s3_", "s4_")]
    ready = sig[cols].copy()
    ready["sig_end"] = sig.index + pd.Timedelta(minutes=15)
    ready = ready.dropna(subset=["atr"]).reset_index(drop=True)
    out = pd.merge_asof(base.reset_index(names="ts"), ready, left_on="ts", right_on="sig_end", direction="backward")
    out = out.set_index("ts")
    out["new_sig"] = out.sig_end.ne(out.sig_end.shift())
    out["vwap_prev"] = out.vwap.shift(1)
    return out


def eia_days(start: date, end: date) -> set[date]:
    hol = set(USFederalHolidayCalendar().holidays(start=pd.Timestamp(start) - pd.Timedelta(days=7), end=pd.Timestamp(end)).date)
    days = set()
    for wed in pd.date_range(start, end, freq="W-WED").date:
        if not any((wed - pd.Timedelta(days=k)) in hol for k in range(3)):
            days.add(wed)
    return days


# ---------------------------------------------------------------- simulation

class Sim:
    def __init__(self, cfg: dict, step: int):
        i = cfg["instrument"]
        self.tick, self.pv = i["tick_size"], i["point_value_usd"]
        self.slip = i["slippage_ticks_per_side"] * i["tick_size"]
        self.comm = 2 * i["commission_per_side_usd"]
        self.step = step
        self.trades: list[dict] = []
        self.pos = None

    def rnd(self, x: float) -> float:
        return round(round(x / self.tick) * self.tick, 2)

    def enter(self, name, ts, side, price, stop_dist, target=None, target_r=None):
        stop_dist = max(self.rnd(stop_dist), self.tick)
        if target is None and target_r:
            target = price + side * target_r * stop_dist
        self.pos = dict(strategy=name, entry_ts=ts, side=side, entry=price, risk=stop_dist,
                        stop=self.rnd(price - side * stop_dist), target=None if target is None else self.rnd(target))

    def close(self, ts, price, reason):
        p = self.pos
        gross = p["side"] * (price - p["entry"]) * self.pv
        cost = self.comm + 2 * self.slip * self.pv
        self.trades.append(dict(strategy=p["strategy"], session=p["entry_ts"].date(), entry_ts=p["entry_ts"], exit_ts=ts,
                                side=p["side"], entry=p["entry"], exit=price, reason=reason, gross_usd=gross,
                                net_usd=gross - cost, r=(gross - cost) / (p["risk"] * self.pv)))
        self.pos = None

    def manage(self, ts, o, h, l, allow_target=True):
        """Intrabar stop then target. Returns True when the position was closed."""
        p = self.pos
        s = p["side"]
        if (l <= p["stop"]) if s > 0 else (h >= p["stop"]):
            self.close(ts, min(o, p["stop"]) if s > 0 else max(o, p["stop"]), "stop")
            return True
        if allow_target and p["target"] is not None and ((h >= p["target"]) if s > 0 else (l <= p["target"])):
            self.close(ts, max(o, p["target"]) if s > 0 else min(o, p["target"]), "target")
            return True
        return False


def run_strategy(name: str, frame: pd.DataFrame, sessions: list[date], cfg: dict, step: int, eia: set[date]) -> pd.DataFrame:
    p = cfg["strategies"][name]
    sim = Sim(cfg, step)
    core_a, core_z = hhmm(cfg["session"]["core_start"]), hhmm(cfg["session"]["core_end"])
    bo_a, bo_z = (hhmm(x) for x in cfg["eia"]["blackout_other_strategies"])
    win = p.get("entry_window")
    win_a, win_z = (hhmm(win[0]), hhmm(win[1])) if win else (None, None)
    k_stop, tr = p["stop_atr_mult"], p.get("target_r")
    kind = name[:2]
    by_day = {d: g for d, g in frame.groupby(frame.index.date)}

    for day in sessions:
        g = by_day[day]
        g = g[(g.index.time >= core_a) & (g.index.time < core_z)]
        is_eia = day in eia
        pend = None  # (side, level or None for VWAP)
        done = {1: 0, -1: 0}
        rows = list(g.itertuples())
        for j, b in enumerate(rows):
            ts, t, o, h, l = b.Index, b.Index.time(), b.open, b.high, b.low
            if np.isnan(b.atr):
                continue
            # 1) signal exits at the open
            if sim.pos is not None and kind == "s1" and b.new_sig:
                if (sim.pos["side"] > 0 and b.s1_xl) or (sim.pos["side"] < 0 and b.s1_xs):
                    sim.close(ts, o, "signal")
            # 2) pending limit bookkeeping on new signal bars
            if kind in ("s2", "s3") and b.new_sig:
                if pend and getattr(b, f"{kind}_off_{'l' if pend[0] > 0 else 's'}"):
                    pend = None
                if sim.pos is None and win_a <= t < win_z and not (is_eia and bo_a <= t < bo_z):
                    for side, tag in ((1, "l"), (-1, "s")):
                        if getattr(b, f"{kind}_arm_{tag}") and (kind == "s2" or done[side] == 0):
                            pend = (side, None if kind == "s2" else (b.pdh if side > 0 else b.pdl))
            # 3) entries
            entered = False
            if sim.pos is None:
                if kind == "s5":
                    if is_eia and t == hhmm("10:45") and b.trend != 0:
                        sim.enter(name, ts, int(b.trend), o, k_stop * b.atr, target_r=tr)
                        entered = True
                elif kind in ("s1", "s4"):
                    ok = b.new_sig and win_a <= t < win_z and not (is_eia and bo_a <= t < bo_z)
                    side = 1 if ok and getattr(b, f"{kind}_long") else -1 if ok and getattr(b, f"{kind}_short") else 0
                    if side:
                        target = b.bb_mid if kind == "s4" else None
                        if kind != "s4" or side * (target - o) > 0:
                            sim.enter(name, ts, side, o, k_stop * b.atr, target=target)
                            entered = True
                elif pend and win_a <= t < win_z and not (is_eia and bo_a <= t < bo_z):
                    side, level = pend
                    level = sim.rnd(b.vwap_prev if level is None else level)
                    if not np.isnan(level) and ((l <= level) if side > 0 else (h >= level)):
                        px = min(o, level) if side > 0 else max(o, level)
                        sim.enter(name, ts, side, px, k_stop * b.atr, target_r=tr)
                        done[side] += 1
                        pend = None
                        sim.manage(ts, px, h, l, allow_target=False)
                        continue
                if pend and t >= win_z:
                    pend = None
            # 4) intrabar management
            if sim.pos is not None:
                sim.manage(ts, o, h, l)
            # 5) forced flatten on the last core bar
            if sim.pos is not None and j == len(rows) - 1:
                sim.close(ts, b.close, "flatten")
    return pd.DataFrame(sim.trades)


# ---------------------------------------------------------------- reporting

def valid_sessions(frame: pd.DataFrame, cfg: dict, step: int, start: date, end: date) -> list[date]:
    s = cfg["session"]
    a, z = hhmm(s["core_start"]), hhmm(s["core_end"])
    expected = ((z.hour - a.hour) * 60 + z.minute - a.minute) // step
    core = frame[(frame.index.time >= a) & (frame.index.time < z)]
    counts = core.groupby(core.index.date).size()
    excl = {date.fromisoformat(x) for x in s["explicit_excluded_sessions"]}
    return [d for d, n in counts.items() if start <= d <= end and d.weekday() < 5 and d not in excl
            and n >= s["min_core_bar_fraction"] * expected]


def metrics(name: str, window: str, trades: pd.DataFrame, sessions: list[date]) -> dict:
    row = dict(strategy=name, window=window, sessions=len(sessions), trades=len(trades))
    if trades.empty:
        return row
    daily = trades.groupby("session").net_usd.sum().reindex(sessions, fill_value=0.0)
    eq = daily.cumsum()
    wins, losses = trades.net_usd[trades.net_usd > 0].sum(), -trades.net_usd[trades.net_usd < 0].sum()
    sd = daily.std(ddof=1)
    row.update(net_usd=trades.net_usd.sum(), gross_usd=trades.gross_usd.sum(), avg_r=trades.r.mean(),
               win_rate=(trades.net_usd > 0).mean(), profit_factor=wins / losses if losses else np.nan,
               sharpe=daily.mean() / sd * np.sqrt(252) if sd else np.nan,
               t_stat=daily.mean() / sd * np.sqrt(len(daily)) if sd else np.nan,
               max_dd_usd=(eq - eq.cummax().clip(lower=0)).min())
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(CONFIG.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "trades").mkdir(exist_ok=True)
    if args.smoke:
        sources = {"SMOKE": args.data}
        windows = {"SMOKE": {"data": "SMOKE", "range": None}}
    else:
        sources = {k: ROOT / v["path"] for k, v in cfg["data"].items() if isinstance(v, dict)}
        for k, path in sources.items():
            if not path.exists():
                raise SystemExit(f"Data not found: {path}. Restore it from {cfg['data']['archive_copy']}.")
            if sha256(path) != cfg["data"][k]["sha256"]:
                raise SystemExit(f"Data hash mismatch: {path}")
        windows = cfg["evaluation_windows"]
    frames = {}
    for k, path in sources.items():
        base, step = load_base(path)
        frame = build_frames(base, cfg)
        frames[k] = (frame, step, eia_days(frame.index[0].date(), frame.index[-1].date()))
    gate = cfg["is_gate"]
    rows, passed = [], set()
    for window, w in windows.items():
        frame, step, eia = frames[w["data"]]
        a, z = w["range"] or (str(frame.index[0].date()), str(frame.index[-1].date()))
        sessions = valid_sessions(frame, cfg, step, date.fromisoformat(a), date.fromisoformat(z))
        for name in cfg["strategies"]:
            if window in ("OOS", "FWD") and name not in passed:
                rows.append(dict(strategy=name, window=window, note="not read: failed IS gate"))
                continue
            trades = run_strategy(name, frame, sessions, cfg, step, eia)
            m = metrics(name, window, trades, sessions)
            if window == "IS":
                ok = (m["trades"] >= gate["min_trades"] and m.get("net_usd", -1) > gate["min_net_pnl_usd"]
                      and m.get("t_stat", 0) >= gate["min_daily_t_stat"])
                m["is_gate"] = "PASS" if ok else "FAIL"
                if ok:
                    passed.add(name)
            rows.append(m)
            trades.to_csv(args.out / "trades" / f"{window.lower()}_{name}.csv", index=False)
            print(f"done {window} {name}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(args.out / ("smoke_results.csv" if args.smoke else "mcl_mudrex_results.csv"), index=False)
    meta = dict(smoke=args.smoke, data={k: dict(path=str(v), sha256=sha256(v)) for k, v in sources.items()}, config_sha256=sha256(CONFIG))
    (args.out / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
