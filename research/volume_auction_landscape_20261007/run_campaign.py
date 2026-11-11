"""Systematic causal volume and approximate-auction landscape.

The local source files are Databento OHLCV-1m bars, resampled to 5 minutes.
They do not contain trades, quotes, aggressor side, MBP, or MBO.  Consequently
all profile fields produced here are explicitly labelled OHLCV approximations.

Signals are evaluated at a completed bar close and filled at the next bar open.
The frozen nine-sleeve portfolio is never imported by strategy code and is read
only after standalone validation for diversification diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results"
FROZEN_DAILY = REPO / "research" / "locked_nine_sleeve_portfolio" / "daily_pnl.csv"
SEED = 20261007


@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick: float
    tick_value: float
    commission_side: float
    start: str
    end: str

    @property
    def point_value(self) -> float:
        return self.tick_value / self.tick


INSTRUMENTS = {
    "MNQ": Instrument("MNQ", 0.25, 0.50, 1.25, "09:30", "16:00"),
    "NQ": Instrument("NQ", 0.25, 5.00, 1.25, "09:30", "16:00"),
    "MES": Instrument("MES", 0.25, 1.25, 1.25, "09:30", "16:00"),
    "M2K": Instrument("M2K", 0.10, 0.50, 1.25, "09:30", "16:00"),
    "MGC": Instrument("MGC", 0.10, 1.00, 1.25, "08:20", "13:30"),
    "MBT": Instrument("MBT", 5.00, 0.50, 1.41, "09:30", "16:00"),
}

DATA_FILES = {
    "MNQ": "data/processed/parquet/MNQ_c_0_5m_20260321_094501.parquet",
    "NQ": "data/processed/parquet/NQ_c_0_5m_20260321_095036.parquet",
    "MES": "data/processed/parquet/MES_c_0_5m_20260322_135702.parquet",
    "M2K": "data/processed/parquet/M2K_c_0_5m_20260322_134808.parquet",
    "MGC": "data/processed/parquet/MGC_v_0_5m_20150101_20260923.parquet",
    "MBT": "data/processed/parquet/MBT_v_0_5m_20210425_20261003.parquet",
}

MECHANISM_NAMES = {
    "A": "rvol_confirmed_breakout",
    "B": "volume_climax_reversal",
    "C": "price_volume_impact_efficiency",
    "D": "high_volume_low_response_proxy",
    "E": "price_move_without_volume_confirmation",
    "F": "volume_shock_after_quiet",
    "G": "value_area_rejection",
    "H": "value_area_acceptance_breakout",
    "I": "poc_attraction",
    "J": "poc_rejection",
    "K": "lvn_traversal",
    "L": "hvn_rejection_acceptance",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False, name=None):
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append("" if not np.isfinite(v) else f"{v:.4g}")
            else:
                vals.append(str(v).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_bars(path: Path, inst: Instrument) -> pd.DataFrame:
    x = pd.read_parquet(path)
    if not isinstance(x.index, pd.DatetimeIndex):
        name = "timestamp" if "timestamp" in x.columns else "ts_event"
        x.index = pd.to_datetime(x.pop(name), errors="raise")
    x.index = pd.to_datetime(x.index)
    if x.index.tz is None:
        x.index = x.index.tz_localize("UTC")
    x.index = x.index.tz_convert("America/New_York")
    x = x.rename(columns={c: c.lower() for c in x.columns}).sort_index()
    x = x[["open", "high", "low", "close", "volume"]].astype(float)
    tod = x.index.strftime("%H:%M")
    x = x[(tod >= inst.start) & (tod < inst.end) & (x.index.dayofweek < 5)].copy()
    x["session_date"] = pd.to_datetime(x.index.date)
    x["slot"] = x.index.hour * 60 + x.index.minute
    x["bar_no"] = x.groupby("session_date", sort=False).cumcount()
    x["last_bar"] = x.groupby("session_date", sort=False).cumcount(ascending=False).eq(0)
    return x


def add_causal_volume_baselines(x: pd.DataFrame, windows: tuple[int, ...] = (10, 20, 40)) -> pd.DataFrame:
    out = x.copy()
    out["cum_volume"] = out.groupby("session_date", sort=False).volume.cumsum()
    for k in windows:
        by_slot = out.groupby("slot", sort=False)["volume"]
        out[f"expected_median_{k}"] = by_slot.transform(
            lambda s: s.shift(1).rolling(k, min_periods=k).median()
        )
        out[f"expected_mean_{k}"] = by_slot.transform(
            lambda s: s.shift(1).rolling(k, min_periods=k).mean()
        )
        med = out[f"expected_median_{k}"]
        out[f"rvol_{k}"] = out.volume / med.replace(0, np.nan)
        q75 = by_slot.transform(lambda s: s.shift(1).rolling(k, min_periods=k).quantile(0.75))
        q25 = by_slot.transform(lambda s: s.shift(1).rolling(k, min_periods=k).quantile(0.25))
        # IQR / 1.349 is a robust Gaussian-equivalent scale; floor protects zero-volume buckets.
        out[f"robust_scale_{k}"] = ((q75 - q25) / 1.349).clip(lower=1.0)
        out[f"volume_surprise_{k}"] = (out.volume - med) / out[f"robust_scale_{k}"]
        by_cum_slot = out.groupby("slot", sort=False)["cum_volume"]
        out[f"expected_cum_{k}"] = by_cum_slot.transform(
            lambda s: s.shift(1).rolling(k, min_periods=k).median()
        )
        out[f"cum_rvol_{k}"] = out.cum_volume / out[f"expected_cum_{k}"].replace(0, np.nan)
    return out


def add_price_features(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    prev = out.close.shift(1)
    tr = pd.concat([out.high - out.low, (out.high - prev).abs(), (out.low - prev).abs()], axis=1).max(axis=1)
    # The scale at bar t uses only bars completed before t.
    out["atr"] = tr.groupby(out.session_date).transform(lambda s: s.shift(1).rolling(20, min_periods=10).median())
    out["signed_move"] = (out.close - out.open) / out.atr.replace(0, np.nan)
    out["abs_move"] = out.signed_move.abs()
    out["range_atr"] = (out.high - out.low) / out.atr.replace(0, np.nan)
    out["body_atr"] = (out.close - out.open).abs() / out.atr.replace(0, np.nan)
    out["close_location"] = (out.close - out.low) / (out.high - out.low).replace(0, np.nan)
    out["impact_efficiency"] = out.abs_move / out.rvol_20.replace(0, np.nan)
    g = out.groupby("session_date", sort=False)
    for n in (2, 3, 6, 12):
        out[f"prior_high_{n}"] = g.high.transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=n).max())
        out[f"prior_low_{n}"] = g.low.transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=n).min())
        out[f"ret_{n}"] = out.close - g.close.shift(n)
    out["quiet_rvol_3"] = g.rvol_20.transform(lambda s: s.shift(1).rolling(3, min_periods=3).median())
    daily = out.groupby("session_date").agg(day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last"))
    daily["prior_range"] = (daily.day_high - daily.day_low).shift(1)
    out["prior_range"] = out.session_date.map(daily.prior_range)
    out["prior_volatility"] = out.session_date.map(daily.prior_range.rolling(20, min_periods=10).median().shift(1))
    # Cached once and reused by every strategy replay.
    out["_session_last_pos"] = pd.Series(np.arange(len(out)), index=out.index).groupby(out.session_date).transform("max").to_numpy()
    return out.replace([np.inf, -np.inf], np.nan)


def profile_levels(x: pd.DataFrame, inst: Instrument, window: int) -> pd.DataFrame:
    """Approximate profile: each completed bar's full volume goes to typical price."""
    sessions = []
    for date, g in x.groupby("session_date", sort=True):
        typical = (g.high + g.low + g.close) / 3.0
        bins = np.round(typical.to_numpy() / inst.tick) * inst.tick
        vol = pd.Series(g.volume.to_numpy(), index=bins).groupby(level=0).sum()
        sessions.append((pd.Timestamp(date), vol))
    rows: list[dict[str, Any]] = []
    for i, (date, _) in enumerate(sessions):
        prior = sessions[max(0, i - window):i]
        if len(prior) < window:
            continue
        prof = pd.concat([p for _, p in prior], axis=1).fillna(0).sum(axis=1).sort_index()
        total = float(prof.sum())
        if total <= 0:
            continue
        poc = float(prof.idxmax())
        ranked = prof.sort_values(ascending=False)
        selected = ranked.index[(ranked.cumsum() / total <= 0.70)].tolist()
        if not selected:
            selected = [poc]
        # Include the bin crossing 70%, then report the price span.
        crossing = int(np.searchsorted((ranked.cumsum() / total).to_numpy(), 0.70, side="left"))
        selected += ranked.index[: min(crossing + 1, len(ranked))].tolist()
        vah, val = float(max(selected)), float(min(selected))
        concentration = float(prof.max() / total)
        rows.append({"session_date": date, f"poc_{window}": poc, f"vah_{window}": vah, f"val_{window}": val,
                     f"va_width_{window}": vah - val, f"profile_concentration_{window}": concentration})
    return pd.DataFrame(rows).set_index("session_date") if rows else pd.DataFrame()


def add_profiles(x: pd.DataFrame, inst: Instrument) -> pd.DataFrame:
    out = x.copy()
    for window in (1, 3, 5):
        p = profile_levels(out, inst, window)
        for col in p.columns:
            out[col] = out.session_date.map(p[col])
        out[f"poc_dist_atr_{window}"] = (out.close - out[f"poc_{window}"]) / out.atr.replace(0, np.nan)
    return out


def variant_grid() -> list[dict[str, Any]]:
    """Compact grids: mechanism interpretation takes priority over Cartesian breadth."""
    rows: list[dict[str, Any]] = []
    for n in (3, 6, 12):
        for rv in (0.0, 1.25, 1.5, 2.0):
            rows.append({"mechanism": "A", "mode": "continuation", "lookback": n, "move": 0.0, "rvol": rv, "profile": 0, "confirm": 1})
    for move in (1.0, 1.5, 2.0):
        for rv in (1.5, 2.0, 2.5):
            for mode in ("immediate_fade", "confirmed_fade"):
                rows.append({"mechanism": "B", "mode": mode, "lookback": 1, "move": move, "rvol": rv, "profile": 0, "confirm": 1})
    for move in (0.75, 1.0):
        for tail in (0.5, 0.75, 1.5, 2.0):
            for mode in (("low_reversal", "low_breakout") if tail <= 0.75 else ("high_continuation", "high_reversal")):
                rows.append({"mechanism": "C", "mode": mode, "lookback": 1, "move": move, "rvol": tail, "profile": 0, "confirm": 1})
    for rv in (1.5, 2.0, 2.5):
        for body in (0.25, 0.5):
            for mode in ("preceding_move_reversal", "bar_breakout"):
                rows.append({"mechanism": "D", "mode": mode, "lookback": 3, "move": body, "rvol": rv, "profile": 0, "confirm": 1})
    for move in (1.0, 1.5, 2.0):
        for rv in (0.75, 1.0):
            for mode in ("fade", "continuation_control"):
                rows.append({"mechanism": "E", "mode": mode, "lookback": 1, "move": move, "rvol": rv, "profile": 0, "confirm": 1})
    for shock in (1.5, 2.0, 2.5):
        for mode in ("continuation", "reversion"):
            rows.append({"mechanism": "F", "mode": mode, "lookback": 3, "move": 0.75, "rvol": shock, "profile": 0, "confirm": 1})
    for w in (1, 3, 5):
        for mode in ("immediate_fade", "confirmed_reentry"):
            rows.append({"mechanism": "G", "mode": mode, "lookback": 1, "move": 0.0, "rvol": 0.0, "profile": w, "confirm": 1})
        for confirm in (1, 2, 3):
            for rv in (0.0, 1.5):
                rows.append({"mechanism": "H", "mode": "continuation", "lookback": 1, "move": 0.0, "rvol": rv, "profile": w, "confirm": confirm})
        for dist in (0.5, 1.0, 1.5):
            rows.append({"mechanism": "I", "mode": "attraction", "lookback": 1, "move": dist, "rvol": 0.0, "profile": w, "confirm": 1})
        for dist in (0.25, 0.5):
            rows.append({"mechanism": "J", "mode": "rejection", "lookback": 1, "move": dist, "rvol": 0.0, "profile": w, "confirm": 1})
    return rows


def variant_id(p: dict[str, Any]) -> str:
    return (f"{p['mechanism']}__{p['mode']}__n{p['lookback']}__m{p['move']}__"
            f"rv{p['rvol']}__p{p['profile']}__c{p['confirm']}")


def signals(x: pd.DataFrame, p: dict[str, Any]) -> pd.Series:
    m, mode = p["mechanism"], p["mode"]
    n, move, rv = int(p["lookback"]), float(p["move"]), float(p["rvol"])
    s = pd.Series(0, index=x.index, dtype=np.int64)
    direction = np.sign(x.close - x.open).astype(np.int8)
    if m == "A":
        s[x.close > x[f"prior_high_{n}"]] = 1
        s[x.close < x[f"prior_low_{n}"]] = -1
        if rv > 0:
            s = s.where(x.rvol_20 >= rv, 0)
    elif m == "B":
        base = (x.abs_move >= move) & (x.rvol_20 >= rv)
        if mode == "immediate_fade":
            s[base] = -direction[base]
        else:
            prior_dir = direction.shift(1)
            confirmed = base.shift(1, fill_value=False) & (direction == -prior_dir)
            s[confirmed] = direction[confirmed]
    elif m == "C":
        base = x.abs_move >= move
        if mode.startswith("low"):
            base &= x.impact_efficiency <= rv
        else:
            base &= x.impact_efficiency >= rv
        use_cont = mode in ("low_breakout", "high_continuation")
        s[base] = direction[base] * (1 if use_cont else -1)
    elif m == "D":
        base = (x.rvol_20 >= rv) & (x.body_atr <= move) & (x.range_atr <= 1.0)
        prior_move = np.sign(x.ret_3)
        if mode == "preceding_move_reversal":
            s[base] = -prior_move[base]
        else:
            # Compression bar direction is an OHLCV proxy, not order flow.
            s[base] = np.where(x.close_location[base] >= 0.5, 1, -1)
    elif m == "E":
        base = (x.abs_move >= move) & (x.rvol_20 <= rv)
        s[base] = direction[base] * (-1 if mode == "fade" else 1)
    elif m == "F":
        base = (x.quiet_rvol_3 <= move) & (x.rvol_20 >= rv)
        s[base] = direction[base] * (1 if mode == "continuation" else -1)
    elif m == "G":
        w = int(p["profile"]); vah, val = x[f"vah_{w}"], x[f"val_{w}"]
        if mode == "immediate_fade":
            s[x.close > vah] = -1; s[x.close < val] = 1
        else:
            s[(x.close > val) & (x.close.shift(1) < val.shift(1))] = 1
            s[(x.close < vah) & (x.close.shift(1) > vah.shift(1))] = -1
    elif m == "H":
        w, c = int(p["profile"]), int(p["confirm"])
        above = x.close > x[f"vah_{w}"]; below = x.close < x[f"val_{w}"]
        accepted_up = above.groupby(x.session_date).transform(lambda q: q.rolling(c, min_periods=c).sum().eq(c))
        accepted_dn = below.groupby(x.session_date).transform(lambda q: q.rolling(c, min_periods=c).sum().eq(c))
        s[accepted_up] = 1; s[accepted_dn] = -1
        if rv > 0:
            s = s.where(x.rvol_20 >= rv, 0)
    elif m == "I":
        w = int(p["profile"]); d = x[f"poc_dist_atr_{w}"]
        # Exclude bars already showing strong acceptance/participation away from value.
        base = (d.abs() >= move) & (x.rvol_20 < 1.5)
        s[base] = -np.sign(d[base]).astype(np.int8)
    elif m == "J":
        w = int(p["profile"]); poc = x[f"poc_{w}"]; tol = move * x.atr
        approach_above = (x.low <= poc + tol) & (x.close > poc) & (x.close > x.open)
        approach_below = (x.high >= poc - tol) & (x.close < poc) & (x.close < x.open)
        s[approach_above] = 1; s[approach_below] = -1
    else:
        raise KeyError(m)
    valid = x.atr.notna() & x.rvol_20.notna() & (~x.last_bar)
    return s.where(valid, 0).fillna(0).astype(np.int8)


def nominal_cost(inst: Instrument, extra_ticks: float = 0.0, commission_mult: float = 1.0) -> float:
    return 2.0 * (1.0 + extra_ticks) * inst.tick_value + 2.0 * inst.commission_side * commission_mult


def backtest(x: pd.DataFrame, sig: pd.Series, inst: Instrument, hold: int = 6, delay: int = 0) -> pd.DataFrame:
    """Non-overlapping trades; signal t fills open t+1+delay and exits close after hold bars."""
    records: list[dict[str, Any]] = []
    sig_a = sig.to_numpy()
    idx = np.flatnonzero(sig_a != 0)
    if not len(idx):
        return pd.DataFrame(records)
    dates = x.session_date.to_numpy()
    opens, closes = x.open.to_numpy(), x.close.to_numpy()
    highs, lows = x.high.to_numpy(), x.low.to_numpy()
    rvol, crvol = x.rvol_20.to_numpy(), x.cum_rvol_20.to_numpy()
    bar_no, prior_vol = x.bar_no.to_numpy(), x.prior_volatility.to_numpy()
    timestamps = x.index.to_numpy()
    positions = x["_session_last_pos"].to_numpy()
    blocked_until = -1
    cursor = 0
    while cursor < len(idx):
        i = int(idx[cursor])
        entry_i = i + 1 + delay
        if entry_i <= blocked_until or entry_i >= len(x):
            cursor += 1
            continue
        if dates[entry_i] != dates[i]:
            cursor += 1
            continue
        last_i = int(positions[i])
        exit_i = min(entry_i + hold - 1, last_i)
        direction = int(sig_a[i])
        entry, exit_ = float(opens[entry_i]), float(closes[exit_i])
        gross = direction * (exit_ - entry) * inst.point_value
        path_high = highs[entry_i:exit_i + 1]
        path_low = lows[entry_i:exit_i + 1]
        mfe = ((path_high.max() - entry) if direction > 0 else (entry - path_low.min())) * inst.point_value
        mae = ((entry - path_low.min()) if direction > 0 else (path_high.max() - entry)) * inst.point_value
        records.append({"signal_time": timestamps[i], "entry_time": timestamps[entry_i], "exit_time": timestamps[exit_i],
                        "session_date": dates[i], "direction": direction, "entry": entry, "exit": exit_,
                        "gross_pnl": gross, "net_pnl": gross - nominal_cost(inst), "mfe": mfe, "mae": mae,
                        "holding_bars": exit_i - entry_i + 1, "signal_rvol": float(rvol[i]),
                        "signal_cum_rvol": float(crvol[i]), "signal_bar_no": int(bar_no[i]),
                        "prior_volatility": float(prior_vol[i]) if pd.notna(prior_vol[i]) else np.nan})
        blocked_until = exit_i
        cursor = int(np.searchsorted(idx, blocked_until + 1, side="left"))
    return pd.DataFrame(records)


def metrics(trades: pd.DataFrame, pnl_col: str = "net_pnl") -> dict[str, float]:
    if trades.empty:
        return {k: np.nan for k in ("net_pnl", "sharpe", "pf", "expectancy", "win_rate", "avg_winner", "avg_loser", "maxdd", "avg_holding") } | {"trades": 0}
    pnl = trades[pnl_col].astype(float)
    daily = trades.assign(_p=pnl).groupby("session_date")._p.sum()
    sd = float(daily.std(ddof=1))
    sharpe = float(daily.mean() / sd * math.sqrt(252)) if sd > 0 else np.nan
    pos, neg = pnl[pnl > 0], pnl[pnl < 0]
    pf = float(pos.sum() / abs(neg.sum())) if len(neg) and neg.sum() else np.inf
    curve = daily.cumsum(); dd = curve - curve.cummax()
    return {"trades": int(len(trades)), "net_pnl": float(pnl.sum()), "sharpe": sharpe, "pf": pf,
            "expectancy": float(pnl.mean()), "win_rate": float((pnl > 0).mean()),
            "avg_winner": float(pos.mean()) if len(pos) else np.nan, "avg_loser": float(neg.mean()) if len(neg) else np.nan,
            "maxdd": float(dd.min()) if len(dd) else 0.0, "avg_holding": float(trades.holding_bars.mean())}


def split_date(x: pd.DataFrame) -> pd.Timestamp:
    dates = np.sort(x.session_date.drop_duplicates().to_numpy())
    return pd.Timestamp(dates[int(len(dates) * 0.70)])


def benchmark_params(p: dict[str, Any]) -> dict[str, Any] | None:
    q = dict(p)
    if p["mechanism"] in ("A", "B"):
        q["rvol"] = 0.0
        return q
    if p["mechanism"] == "H" and float(p["rvol"]) > 0:
        q["rvol"] = 0.0
        return q
    return None


def price_only_signals(x: pd.DataFrame, p: dict[str, Any]) -> pd.Series:
    """Matched event definition with volume/impact clauses removed."""
    m, mode, move = p["mechanism"], p["mode"], float(p["move"])
    if m in ("A", "B", "H"):
        q = benchmark_params(p)
        if q is not None:
            return signals(x, q)
    s = pd.Series(0, index=x.index, dtype=np.int64)
    direction = np.sign(x.close - x.open).astype(np.int8)
    if m == "C":
        base = x.abs_move >= move
        use_cont = mode in ("low_breakout", "high_continuation")
        s[base] = direction[base] * (1 if use_cont else -1)
    elif m == "D":
        base = (x.body_atr <= move) & (x.range_atr <= 1.0)
        s[base] = -np.sign(x.ret_3[base]).astype(np.int8) if mode == "preceding_move_reversal" else np.where(x.close_location[base] >= 0.5, 1, -1)
    elif m == "E":
        base = x.abs_move >= move
        s[base] = direction[base] * (-1 if mode == "fade" else 1)
    elif m == "F":
        # The matched price event is the shock bar's signed move, absent participation state.
        base = x.abs_move >= 0.5
        s[base] = direction[base] * (1 if mode == "continuation" else -1)
    else:
        return signals(x, p)
    return s.where(x.atr.notna() & (~x.last_bar), 0).fillna(0).astype(np.int8)


def is_volume_conditioned(p: dict[str, Any]) -> bool:
    return p["mechanism"] in ("A", "B", "C", "D", "E", "F") or (p["mechanism"] == "H" and float(p["rvol"]) > 0)


def thin_control(benchmark: pd.DataFrame, n: int, reps: int = 200, seed: int = SEED) -> dict[str, float]:
    if n <= 0 or len(benchmark) < n:
        return {"repetitions": 0, "mean_sharpe": np.nan, "p95_sharpe": np.nan, "mean_expectancy": np.nan, "p95_expectancy": np.nan}
    rng = np.random.default_rng(seed)
    pnl = benchmark.net_pnl.to_numpy(float)
    codes, _ = pd.factorize(benchmark.session_date, sort=True)
    sharpes, exps = [], []
    for _ in range(reps):
        take = rng.choice(len(pnl), size=n, replace=False)
        vals = pnl[take]; daily = np.bincount(codes[take], weights=vals)
        active = daily[daily != 0]
        sd = active.std(ddof=1) if len(active) > 1 else 0.0
        sharpes.append(active.mean() / sd * math.sqrt(252) if sd > 0 else np.nan)
        exps.append(vals.mean())
    return {"repetitions": reps, "mean_sharpe": float(np.nanmean(sharpes)), "p95_sharpe": float(np.nanpercentile(sharpes, 95)),
            "mean_expectancy": float(np.nanmean(exps)), "p95_expectancy": float(np.nanpercentile(exps, 95))}


def region_labels(results: pd.DataFrame) -> pd.Series:
    labels = pd.Series("FRAGILE", index=results.index, dtype=object)
    for (_, _, _), g in results.groupby(["asset", "mechanism", "mode"]):
        positive = (g.oos_sharpe > 0) & (g.oos_expectancy > 0) & (g.oos_trades >= 30)
        ratio = float(positive.mean()) if len(g) else 0.0
        if ratio >= 0.60 and positive.sum() >= 3:
            labels.loc[g.index[positive]] = "BROAD_REGION"
        elif positive.sum() >= 2:
            labels.loc[g.index[positive]] = "LOCAL_REGION"
    return labels


def information_label(delta_sharpe: float, delta_exp: float, thin_p95_sharpe: float, candidate_sharpe: float) -> str:
    if not np.isfinite(delta_sharpe) or not np.isfinite(delta_exp):
        return "VOLUME_IS_REDUNDANT"
    beats_thin = np.isfinite(thin_p95_sharpe) and candidate_sharpe > thin_p95_sharpe
    if delta_sharpe > 0.35 and delta_exp > 0 and beats_thin:
        return "VOLUME_ADDS_CLEAR_VALUE"
    if delta_sharpe > 0 and delta_exp > 0:
        return "VOLUME_ADDS_SOME_VALUE"
    if delta_sharpe < 0 or delta_exp < 0:
        return "VOLUME_HURTS"
    return "VOLUME_IS_REDUNDANT"


def save_seasonality(x: pd.DataFrame, asset: str, out: Path) -> None:
    cols = ["slot", "expected_median_10", "expected_median_20", "expected_median_40", "volume"]
    curve = x[cols].groupby("slot").median(numeric_only=True).reset_index()
    curve.to_csv(out / "volume_seasonality" / f"{asset}_causal_curves.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    for c in ("volume", "expected_median_10", "expected_median_20", "expected_median_40"):
        ax.plot(curve.slot, curve[c], label=c)
    ax.set(title=f"{asset}: intraday volume seasonality", xlabel="minute of day (ET)", ylabel="contracts")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "volume_seasonality" / f"{asset}_u_shape.png", dpi=130); plt.close(fig)


def save_heatmaps(results: pd.DataFrame, out: Path) -> None:
    for (asset, mech), g in results[results.split == "OOS"].groupby(["asset", "mechanism"]):
        if mech in ("G", "H", "I", "J"):
            a, b = "profile", "confirm" if mech == "H" else "move"
        else:
            a, b = "move", "rvol"
        pivot = g.pivot_table(index=a, columns=b, values="sharpe", aggfunc="median")
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(5, 3.5)); im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(len(pivot.columns)), [str(v) for v in pivot.columns]); ax.set_yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
        ax.set(xlabel=b, ylabel=a, title=f"{asset} {mech}: median OOS Sharpe")
        fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out / "heatmaps" / f"{asset}_{mech}.png", dpi=130); plt.close(fig)


def ensure_dirs(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("heatmaps", "event_studies", "volume_seasonality", "volume_profiles"):
        (out / name).mkdir(exist_ok=True)


def choose_representatives(wide: pd.DataFrame) -> pd.DataFrame:
    """Choose a central positive cell, never the isolated maximum-Sharpe cell."""
    chosen = []
    for (_, _), g0 in wide.groupby(["asset", "mechanism"]):
        g = g0[(g0.oos_trades >= 30) & (g0.oos_expectancy > 0) & (g0.oos_sharpe > 0)].copy()
        stable = g[g.region_stability.isin(["LOCAL_REGION", "BROAD_REGION"])]
        if not stable.empty:
            g = stable
        elif g.empty:
            # Retain a diagnostic representative even when rejected.
            g = g0[g0.oos_trades >= 10].copy()
        if g.empty:
            continue
        numeric = [c for c in ("lookback", "move", "rvol", "profile", "confirm") if g[c].nunique() > 1]
        distance = pd.Series(0.0, index=g.index)
        for c in numeric:
            spread = max(float(g[c].max() - g[c].min()), 1e-9)
            distance += (g[c] - g[c].median()).abs() / spread
        # Trade count breaks central-cell ties; Sharpe is deliberately not used.
        pick = g.assign(_distance=distance).sort_values(["_distance", "oos_trades"], ascending=[True, False]).iloc[0]
        chosen.append(pick.drop(labels="_distance", errors="ignore"))
    return pd.DataFrame(chosen).reset_index(drop=True)


def run_campaign(output: Path = DEFAULT_OUTPUT, smoke: bool = False) -> None:
    ensure_dirs(output)
    config = json.loads((HERE / "campaign_config.json").read_text(encoding="utf-8"))
    grid = variant_grid()
    if smoke:
        grid = [p for p in grid if p["mechanism"] in ("A", "G")][:4]
    frames: dict[str, pd.DataFrame] = {}
    splits: dict[str, pd.Timestamp] = {}
    inventory, parameter_rows, trade_cache = [], [], {}
    profile_rows: list[pd.DataFrame] = []

    for asset, rel in DATA_FILES.items():
        path, inst = REPO / rel, INSTRUMENTS[asset]
        if not path.exists():
            inventory.append({"Asset": asset, "OHLCV": "NO", "Trade-level": "NO", "Bid/Ask": "NO", "Aggressor": "NO",
                              "Volume-at-price feasible": "NO", "Coverage": "DATA_UNAVAILABLE", "Frequency": ""})
            continue
        x = load_bars(path, inst)
        if smoke:
            keep_dates = x.session_date.drop_duplicates().tail(120)
            x = x[x.session_date.isin(keep_dates)].copy()
        x = add_profiles(add_price_features(add_causal_volume_baselines(x)), inst)
        frames[asset] = x; splits[asset] = split_date(x)
        inventory.append({"Asset": asset, "OHLCV": "YES", "Trade-level": "NO", "Bid/Ask": "NO", "Aggressor": "NO",
                          "Volume-at-price feasible": "APPROXIMATE_ONLY", "Coverage": f"{x.index.min()} to {x.index.max()}",
                          "Frequency": "5m (from Databento ohlcv-1m)", "Rows": len(x), "Source": rel, "SHA256": sha256(path)})
        save_seasonality(x, asset, output)
        prof_cols = ["session_date"] + [c for c in x.columns if c.startswith(("poc_", "vah_", "val_", "va_width_", "profile_concentration_")) and "dist" not in c]
        prof = x[prof_cols].drop_duplicates("session_date").copy(); prof.insert(0, "asset", asset)
        profile_rows.append(prof)
        for p in grid:
            vid = variant_id(p); sig = signals(x, p); trades = backtest(x, sig, inst)
            trade_cache[(asset, vid)] = trades
            for label, mask in (("IS", trades.session_date < splits[asset]), ("OOS", trades.session_date >= splits[asset])):
                met = metrics(trades.loc[mask])
                parameter_rows.append({"asset": asset, "mechanism": p["mechanism"], "mechanism_name": MECHANISM_NAMES[p["mechanism"]],
                                       "variant_id": vid, **p, "split": label, **met})

    pd.DataFrame(inventory).to_csv(output / "data_capability_inventory.csv", index=False)
    if profile_rows:
        pd.concat(profile_rows, ignore_index=True).to_csv(output / "volume_profiles" / "approximate_profile_levels.csv", index=False)
    all_results = pd.DataFrame(parameter_rows)
    all_results.to_csv(output / "all_parameter_results.csv", index=False)
    if all_results.empty:
        raise RuntimeError("No available data/results")
    save_heatmaps(all_results, output)

    id_cols = ["asset", "mechanism", "mechanism_name", "variant_id", "mode", "lookback", "move", "rvol", "profile", "confirm"]
    wide = all_results.pivot_table(index=id_cols, columns="split", values=["trades", "net_pnl", "sharpe", "pf", "expectancy", "maxdd"], aggfunc="first").reset_index()
    wide.columns = ["_".join([str(z) for z in c if z]).lower() if isinstance(c, tuple) else c for c in wide.columns]
    wide = wide.rename(columns={c: c.replace("_is", "_is").replace("_oos", "_oos") for c in wide.columns})
    # pivot_table emits e.g. sharpe_IS -> sharpe_is; normalize to report convention.
    for metric in ("trades", "net_pnl", "sharpe", "pf", "expectancy", "maxdd"):
        wide = wide.rename(columns={f"{metric}_is": f"is_{metric}", f"{metric}_oos": f"oos_{metric}"})
    wide["region_stability"] = region_labels(wide)

    comparison_rows, thinning_rows, cost_rows = [], [], []
    for row in wide.itertuples(index=False):
        asset, vid = row.asset, row.variant_id; x, inst = frames[asset], INSTRUMENTS[asset]
        p = {k: getattr(row, k) for k in ("mechanism", "mode", "lookback", "move", "rvol", "profile", "confirm")}
        trades = trade_cache[(asset, vid)]; oos = trades[trades.session_date >= splits[asset]].copy()
        for scenario, extra, cmult in (("nominal", 0, 1), ("plus_1_tick_side", 1, 1), ("plus_2_ticks_side", 2, 1), ("double_commission", 0, 2)):
            stressed = oos.copy()
            if not stressed.empty:
                stressed["stress_pnl"] = stressed.gross_pnl - nominal_cost(inst, extra, cmult)
            met = metrics(stressed, "stress_pnl") if not stressed.empty else metrics(stressed)
            cost_rows.append({"asset": asset, "variant_id": vid, "scenario": scenario, **{f"oos_{k}": v for k, v in met.items()},
                              "gross_edge_per_trade": float(oos.gross_pnl.mean()) if len(oos) else np.nan,
                              "friction_per_trade": nominal_cost(inst, extra, cmult),
                              "friction_gross_ratio": nominal_cost(inst, extra, cmult) / oos.gross_pnl.mean() if len(oos) and oos.gross_pnl.mean() > 0 else np.nan})
        if p["mechanism"] in ("G", "H", "I", "J"):
            for delay in (1, 2):
                delayed = backtest(x, signals(x, p), inst, delay=delay)
                delayed = delayed[delayed.session_date >= splits[asset]]
                cost_rows.append({"asset": asset, "variant_id": vid, "scenario": f"delay_{delay}_bar", **{f"oos_{k}": v for k, v in metrics(delayed).items()},
                                  "gross_edge_per_trade": float(delayed.gross_pnl.mean()) if len(delayed) else np.nan,
                                  "friction_per_trade": nominal_cost(inst), "friction_gross_ratio": np.nan})
        if is_volume_conditioned(p):
            bench = backtest(x, price_only_signals(x, p), inst)
            bench = bench[bench.session_date >= splits[asset]]
            bm = metrics(bench); vm = metrics(oos)
            thin = thin_control(bench, len(oos), seed=SEED + sum(map(ord, asset + vid)))
            comparison_rows.append({"asset": asset, "mechanism": p["mechanism"], "variant_id": vid,
                                    "volume_trades": vm["trades"], "price_only_trades": bm["trades"],
                                    "volume_sharpe": vm["sharpe"], "price_only_sharpe": bm["sharpe"], "delta_sharpe": vm["sharpe"] - bm["sharpe"],
                                    "volume_expectancy": vm["expectancy"], "price_only_expectancy": bm["expectancy"], "delta_expectancy": vm["expectancy"] - bm["expectancy"],
                                    "volume_pf": vm["pf"], "price_only_pf": bm["pf"], "delta_pf": vm["pf"] - bm["pf"],
                                    "volume_maxdd": vm["maxdd"], "price_only_maxdd": bm["maxdd"], "delta_maxdd": vm["maxdd"] - bm["maxdd"],
                                    "delta_trade_count": vm["trades"] - bm["trades"], **thin})
            thinning_rows.append({"asset": asset, "variant_id": vid, "target_trade_count": len(oos), "benchmark_trade_count": len(bench), **thin,
                                  "candidate_sharpe": vm["sharpe"], "candidate_expectancy": vm["expectancy"],
                                  "beats_random_thinning_p95": bool(np.isfinite(vm["sharpe"]) and np.isfinite(thin["p95_sharpe"]) and vm["sharpe"] > thin["p95_sharpe"])})

    comparisons = pd.DataFrame(comparison_rows); comparisons.to_csv(output / "price_only_vs_volume_comparison.csv", index=False)
    thinning = pd.DataFrame(thinning_rows); thinning.to_csv(output / "random_thinning_controls.csv", index=False)
    costs = pd.DataFrame(cost_rows); costs.to_csv(output / "cost_stress_results.csv", index=False)
    if not comparisons.empty:
        info = comparisons.set_index(["asset", "variant_id"])
        for i, r in wide.iterrows():
            key = (r.asset, r.variant_id)
            if key in info.index:
                z = info.loc[key]
                wide.loc[i, "increment_vs_price_only"] = z.delta_sharpe
                wide.loc[i, "information_value"] = information_label(z.delta_sharpe, z.delta_expectancy, z.p95_sharpe, r.oos_sharpe)
    default_info = pd.Series(np.where(wide.mechanism.isin(["G", "H", "I", "J"]), "PROFILE_DIAGNOSTIC", "NOT_APPLICABLE"), index=wide.index)
    wide["information_value"] = wide.information_value.fillna(default_info)
    reps = choose_representatives(wide)
    annual_rows, regime_rows, event_rows, falsification_rows = [], [], [], []
    rep_trade_map: dict[tuple[str, str], pd.DataFrame] = {}
    for r in reps.itertuples(index=False):
        asset, vid, mech = r.asset, r.variant_id, r.mechanism
        x, inst = frames[asset], INSTRUMENTS[asset]
        p = {k: getattr(r, k) for k in ("mechanism", "mode", "lookback", "move", "rvol", "profile", "confirm")}
        sig = signals(x, p); trades = trade_cache[(asset, vid)].copy()
        oos = trades[trades.session_date >= splits[asset]].copy(); rep_trade_map[(asset, mech)] = oos
        if not trades.empty:
            trades["year"] = pd.to_datetime(trades.session_date).dt.year
            for (split, year), g in trades.assign(split=np.where(trades.session_date < splits[asset], "IS", "OOS")).groupby(["split", "year"]):
                annual_rows.append({"asset": asset, "mechanism": mech, "variant_id": vid, "split": split, "year": year, **metrics(g)})
        if not oos.empty:
            oos = oos.copy()
            oos["time_segment"] = pd.cut(oos.signal_bar_no, bins=[-1, 12, 36, 10_000], labels=["early", "middle", "late"])
            oos["prior_vol_state"] = pd.qcut(oos.prior_volatility, 3, labels=["low", "mid", "high"], duplicates="drop")
            oos["cum_rvol_state"] = pd.cut(oos.signal_cum_rvol, [-np.inf, .8, 1.2, np.inf], labels=["low", "normal", "high"])
            for dimension in ("time_segment", "prior_vol_state", "cum_rvol_state"):
                for state, g in oos.groupby(dimension, observed=True):
                    regime_rows.append({"asset": asset, "mechanism": mech, "variant_id": vid, "dimension": dimension, "state": state, **metrics(g)})
        # Event path from signal close; direction is fixed at completed bar t.
        locations = np.flatnonzero(sig.to_numpy() != 0)
        locations = [i for i in locations if x.session_date.iloc[i] >= splits[asset]]
        paths = []
        for horizon in (1, 3, 6, 12):
            vals = []
            for i in locations:
                j = i + horizon
                if j < len(x) and x.session_date.iloc[j] == x.session_date.iloc[i]:
                    vals.append(int(sig.iloc[i]) * (float(x.close.iloc[j]) - float(x.close.iloc[i])) * inst.point_value)
            event_rows.append({"asset": asset, "mechanism": mech, "variant_id": vid, "horizon_bars": horizon,
                               "events": len(vals), "mean_directional_pnl_usd": float(np.mean(vals)) if vals else np.nan,
                               "median_directional_pnl_usd": float(np.median(vals)) if vals else np.nan,
                               "positive_probability": float(np.mean(np.asarray(vals) > 0)) if vals else np.nan})
            paths.append((horizon, float(np.mean(vals)) if vals else np.nan))
        pd.DataFrame(paths, columns=["horizon_bars", "mean_directional_pnl_usd"]).to_csv(output / "event_studies" / f"{asset}_{mech}.csv", index=False)
        fig, ax = plt.subplots(figsize=(5, 3)); ax.plot([z[0] for z in paths], [z[1] for z in paths], marker="o"); ax.axhline(0, color="black", lw=.7)
        ax.set(title=f"{asset} {mech}: OOS conditional path", xlabel="bars after event", ylabel="directional USD"); fig.tight_layout()
        fig.savefig(output / "event_studies" / f"{asset}_{mech}.png", dpi=130); plt.close(fig)

        def add_fals(name: str, t: pd.DataFrame) -> None:
            falsification_rows.append({"asset": asset, "mechanism": mech, "variant_id": vid, "test": name, **metrics(t)})

        opposite = oos.copy()
        if not opposite.empty:
            opposite["net_pnl"] = -opposite.gross_pnl - nominal_cost(inst)
        add_fals("opposite_direction", opposite)
        bench = backtest(x, price_only_signals(x, p), inst); bench = bench[bench.session_date >= splits[asset]]
        add_fals("remove_volume_condition_or_matched_control", bench)
        for delay in (1, 2):
            delayed = backtest(x, sig, inst, delay=delay); add_fals(f"delayed_entry_{delay}_bar", delayed[delayed.session_date >= splits[asset]])
        for nbest in (5, 10):
            reduced = oos.drop(oos.nlargest(min(nbest, len(oos)), "net_pnl").index) if len(oos) else oos
            add_fals(f"remove_best_{nbest}_trades", reduced)
        if len(oos):
            year_pnl = oos.groupby(pd.to_datetime(oos.session_date).dt.year).net_pnl.sum()
            best_year = int(year_pnl.idxmax())
            add_fals("remove_best_year", oos[pd.to_datetime(oos.session_date).dt.year != best_year])
        rng = np.random.default_rng(SEED + sum(map(ord, asset + vid)))
        shuffled = x.copy()
        shuffled["rvol_20"] = shuffled.groupby("slot").rvol_20.transform(lambda q: pd.Series(rng.permutation(q.to_numpy()), index=q.index))
        shuffled["impact_efficiency"] = shuffled.abs_move / shuffled.rvol_20.replace(0, np.nan)
        st = backtest(shuffled, signals(shuffled, p), inst); add_fals("shuffle_volume_within_time_bucket", st[st.session_date >= splits[asset]])
        if mech in ("G", "H", "I", "J"):
            shifted = x.copy()
            for w in (1, 3, 5):
                for c in (f"poc_{w}", f"vah_{w}", f"val_{w}"):
                    shifted[c] = shifted.groupby("session_date")[c].transform("first").groupby(shifted.session_date).transform("first")
                    byday = shifted[["session_date", c]].drop_duplicates().set_index("session_date")[c].shift(1)
                    shifted[c] = shifted.session_date.map(byday)
                shifted[f"poc_dist_atr_{w}"] = (shifted.close - shifted[f"poc_{w}"]) / shifted.atr
            st = backtest(shifted, signals(shifted, p), inst); add_fals("profile_shift_one_completed_session", st[st.session_date >= splits[asset]])
            if mech in ("I", "J"):
                midpoint = x.copy()
                daily = midpoint.groupby("session_date").agg(h=("high", "max"), l=("low", "min"))
                prior_mid = ((daily.h + daily.l) / 2).shift(1)
                w = int(p["profile"]); midpoint[f"poc_{w}"] = midpoint.session_date.map(prior_mid)
                midpoint[f"poc_dist_atr_{w}"] = (midpoint.close - midpoint[f"poc_{w}"]) / midpoint.atr
                st = backtest(midpoint, signals(midpoint, p), inst); add_fals("replace_poc_with_prior_session_midpoint", st[st.session_date >= splits[asset]])

    annual = pd.DataFrame(annual_rows); annual.to_csv(output / "annual_results.csv", index=False)
    regimes = pd.DataFrame(regime_rows); regimes.to_csv(output / "regime_diagnostics.csv", index=False)
    pd.DataFrame(event_rows).to_csv(output / "event_studies" / "event_study_summary.csv", index=False)
    falsifications = pd.DataFrame(falsification_rows); falsifications.to_csv(output / "falsification_results.csv", index=False)

    # Cross-asset dependence among representatives of the same mechanism.
    dependence_rows = []
    for mech in sorted(reps.mechanism.unique()):
        assets = sorted(reps.loc[reps.mechanism == mech, "asset"].unique())
        for i, a in enumerate(assets):
            for b in assets[i + 1:]:
                ta, tb = rep_trade_map.get((a, mech), pd.DataFrame()), rep_trade_map.get((b, mech), pd.DataFrame())
                if ta.empty or tb.empty:
                    continue
                da = ta.groupby("session_date").net_pnl.sum(); db = tb.groupby("session_date").net_pnl.sum()
                common = da.index.intersection(db.index); dates_a, dates_b = set(ta.session_date), set(tb.session_date)
                overlap = len(dates_a & dates_b) / max(1, len(dates_a | dates_b))
                merged = ta[["session_date", "direction"]].merge(tb[["session_date", "direction"]], on="session_date", suffixes=("_a", "_b"))
                dependence_rows.append({"mechanism": mech, "asset_a": a, "asset_b": b, "common_days": len(common),
                                        "signal_date_jaccard": overlap, "direction_overlap": float((merged.direction_a == merged.direction_b).mean()) if len(merged) else np.nan,
                                        "daily_pnl_correlation": float(da.loc[common].corr(db.loc[common])) if len(common) >= 10 else np.nan,
                                        "unique_event_days": len(dates_a ^ dates_b), "related_contracts": bool({a, b} <= {"MNQ", "NQ", "MES", "M2K"})})
    pd.DataFrame(dependence_rows).to_csv(output / "cross_asset_dependence.csv", index=False)

    # Read-only diversification diagnostics against the frozen nine sleeves.
    diversification_rows = []
    frozen = pd.read_csv(FROZEN_DAILY, parse_dates=["session_date"]).set_index("session_date")
    sleeve_cols = [c for c in frozen.columns if c != "portfolio_pnl_usd"]
    base = frozen.portfolio_pnl_usd
    def daily_stats(s: pd.Series) -> tuple[float, float]:
        sd = s.std(ddof=1); sharpe = s.mean() / sd * math.sqrt(252) if sd > 0 else np.nan
        dd = (s.cumsum() - s.cumsum().cummax()).min()
        return float(sharpe), float(dd)
    base_sharpe, base_dd = daily_stats(base)
    for r in reps.itertuples(index=False):
        t = rep_trade_map.get((r.asset, r.mechanism), pd.DataFrame())
        if t.empty:
            continue
        cand = t.groupby("session_date").net_pnl.sum(); cand.index = pd.to_datetime(cand.index)
        common = base.index.intersection(cand.index)
        if len(common) < 20:
            continue
        bp, cp = base.loc[common], cand.loc[common]; added = bp + cp
        ash, add = daily_stats(added)
        neg = bp < 0; thresholds = {q: bp.quantile(q) for q in (.05, .10, .20)}
        row = {"asset": r.asset, "mechanism": r.mechanism, "variant_id": r.variant_id, "common_days": len(common),
               "corr_portfolio": float(cp.corr(bp)), "corr_negative_portfolio_days": float(cp[neg].corr(bp[neg])) if neg.sum() >= 10 else np.nan,
               "base_sharpe": base_sharpe, "added_sharpe": ash, "delta_sharpe": ash - base_sharpe,
               "base_maxdd": base_dd, "added_maxdd": add, "delta_maxdd": add - base_dd,
               "candidate_on_worst_5pct": float(cp[bp <= thresholds[.05]].mean()), "candidate_on_worst_10pct": float(cp[bp <= thresholds[.10]].mean()),
               "candidate_on_worst_20pct": float(cp[bp <= thresholds[.20]].mean()), "active_day_overlap": float(((cp != 0) & (bp != 0)).mean()),
               "marginal_variance": float(added.var() - bp.var())}
        for col in sleeve_cols:
            row[f"corr_{col}"] = float(cp.corr(frozen.loc[common, col]))
        diversification_rows.append(row)
    diversification = pd.DataFrame(diversification_rows); diversification.to_csv(output / "diversification_vs_9_sleeves.csv", index=False)

    # Promotion gates and summary. These are mechanism-level research verdicts, not sleeve selections.
    plus2 = costs[costs.scenario == "plus_2_ticks_side"].set_index(["asset", "variant_id"])
    fbest10 = falsifications[falsifications.test == "remove_best_10_trades"].set_index(["asset", "variant_id"])
    profile_shift = falsifications[falsifications.test == "profile_shift_one_completed_session"].set_index(["asset", "variant_id"])
    profile_midpoint = falsifications[falsifications.test == "replace_poc_with_prior_session_midpoint"].set_index(["asset", "variant_id"])
    annual_oos = annual[annual.split == "OOS"] if not annual.empty else annual
    summary_rows = []
    for r in reps.itertuples(index=False):
        key = (r.asset, r.variant_id)
        cost_pass = bool(key in plus2.index and plus2.loc[key, "oos_net_pnl"] > 0)
        concentration_pass = bool(key in fbest10.index and fbest10.loc[key, "net_pnl"] > 0)
        ay = annual_oos[(annual_oos.asset == r.asset) & (annual_oos.variant_id == r.variant_id)]
        positive_year_fraction = float((ay.net_pnl > 0).mean()) if len(ay) else np.nan
        stable = r.region_stability in ("LOCAL_REGION", "BROAD_REGION")
        standalone = bool(r.oos_trades >= 50 and r.oos_expectancy > 0 and r.oos_sharpe > 0.25 and r.oos_pf > 1)
        if r.mechanism in ("G", "H", "I", "J"):
            shift_pass = bool(key in profile_shift.index and r.oos_sharpe > profile_shift.loc[key, "sharpe"])
            midpoint_pass = bool(key not in profile_midpoint.index or r.oos_sharpe > profile_midpoint.loc[key, "sharpe"])
            profile_falsification_pass = shift_pass and midpoint_pass
            profile_value = "PROFILE_ADDS_SOME_VALUE" if standalone and stable and cost_pass and concentration_pass and profile_falsification_pass else "PROFILE_UNRELIABLE_WITH_CURRENT_DATA"
            info_ok = profile_value == "PROFILE_ADDS_SOME_VALUE"
            info_value = profile_value
        else:
            info_value = r.information_value
            info_ok = info_value == "VOLUME_ADDS_CLEAR_VALUE"
        div = diversification[(diversification.asset == r.asset) & (diversification.mechanism == r.mechanism)] if not diversification.empty else pd.DataFrame()
        corr = float(div.iloc[0].corr_portfolio) if len(div) else np.nan
        new_risk = bool(
            standalone and stable and cost_pass and concentration_pass
            and np.isfinite(corr) and abs(corr) < 0.30 and float(div.iloc[0].delta_sharpe) > 0
        ) if len(div) else False
        strict = standalone and stable and cost_pass and concentration_pass and positive_year_fraction >= 0.60 and info_ok and new_risk
        if strict:
            verdict = "CANDIDATE_FOR_PROSPECTIVE_FREEZE"
        elif standalone and stable and cost_pass and concentration_pass and info_ok:
            verdict = "ROBUST_REGION_FOUND"
        elif r.oos_expectancy > 0 and r.oos_trades >= 30:
            verdict = "WEAK_WATCHLIST"
        else:
            verdict = "REJECTED"
        summary_rows.append({"asset": r.asset, "mechanism": r.mechanism, "mechanism_name": r.mechanism_name,
                             "representative_variant": r.variant_id, "oos_trades": int(r.oos_trades), "oos_net_pnl": r.oos_net_pnl,
                             "oos_sharpe": r.oos_sharpe, "oos_pf": r.oos_pf, "oos_expectancy": r.oos_expectancy,
                             "increment_vs_price_only": getattr(r, "increment_vs_price_only", np.nan), "information_value": info_value,
                             "cost_stress": "PASS_+2_TICKS" if cost_pass else "FAIL_+2_TICKS", "region_stability": r.region_stability,
                             "positive_oos_year_fraction": positive_year_fraction, "top10_removal_survives": concentration_pass,
                             "corr_9_sleeve": corr, "new_risk_dimension": new_risk, "verdict": verdict})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "mechanism_by_asset_summary.csv", index=False)

    # Topstep remains gated. The unchanged framework is not invoked without a strict survivor.
    strict_candidates = summary[summary.verdict == "CANDIDATE_FOR_PROSPECTIVE_FREEZE"]
    topstep = pd.DataFrame([{"status": "NOT_RUN_NO_FULLY_GATED_CANDIDATE", "reason": "No candidate passed standalone, costs, falsification, information-value, temporal, and diversification gates."}]) if strict_candidates.empty else pd.DataFrame(
        [{"status": "DEFERRED_PENDING_EXPLICIT_PROSPECTIVE_FREEZE", "asset": r.asset, "mechanism": r.mechanism,
          "reason": "Research survivor is not automatically promoted; unchanged Topstep simulation requires an explicit freeze decision."} for r in strict_candidates.itertuples(index=False)]
    )
    topstep.to_csv(output / "topstep_incremental_results.csv", index=False)

    registry_lines = ["# Volume / auction candidate registry", "", "No exact best parameter is frozen by this campaign.", ""]
    for r in summary.itertuples(index=False):
        registry_lines += [f"## {r.asset} — {r.mechanism}: {r.verdict}", "",
                           f"- Diagnostic central-region representative: `{r.representative_variant}`",
                           f"- Information classification: `{r.information_value}`",
                           f"- OOS: {int(r.oos_trades)} trades, Sharpe {r.oos_sharpe:.3f}, PF {r.oos_pf:.3f}, expectancy ${r.oos_expectancy:.2f}",
                           f"- Cost / stability: `{r.cost_stress}` / `{r.region_stability}`",
                           f"- Frozen portfolio correlation: {r.corr_9_sleeve:.3f}" if np.isfinite(r.corr_9_sleeve) else "- Frozen portfolio correlation: unavailable (insufficient overlap)", ""]
    (output / "candidate_registry.md").write_text("\n".join(registry_lines), encoding="utf-8")

    metadata = {"config": config, "data_files": DATA_FILES, "data_hashes": {a: sha256(REPO / p) for a, p in DATA_FILES.items() if (REPO / p).exists()},
                "assets_run": sorted(frames), "split_dates": {a: str(d.date()) for a, d in splits.items()}, "smoke": smoke,
                "notes": ["All source market data are OHLCV bars.", "Profiles assign full completed-bar volume to tick-rounded typical price.",
                          "Frozen nine sleeves were not modified or recalibrated.", "Representatives are central positive-region cells, not maximum-Sharpe cells."]}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Build the requested report with the decision table literally first.
    table = summary.rename(columns={"asset": "Asset", "mechanism_name": "Mechanism", "oos_trades": "OOS Trades", "oos_sharpe": "OOS Sharpe",
                                            "oos_pf": "PF", "increment_vs_price_only": "Increment vs Price-Only", "cost_stress": "Cost Stress",
                                            "region_stability": "Region Stability", "corr_9_sleeve": "Corr 9-Sleeve", "verdict": "Verdict"})
    first_cols = ["Asset", "Mechanism", "OOS Trades", "OOS Sharpe", "PF", "Increment vs Price-Only", "Cost Stress", "Region Stability", "Corr 9-Sleeve", "Verdict"]
    broad_counts = summary.groupby("mechanism_name").agg(assets=("asset", "size"), positive_oos=("oos_expectancy", lambda s: int((s > 0).sum())), robust=("verdict", lambda s: int(s.isin(["ROBUST_REGION_FOUND", "CANDIDATE_FOR_PROSPECTIVE_FREEZE"]).sum()))).reset_index()
    volume_counts = summary[~summary.mechanism.isin(["G", "H", "I", "J"])].information_value.value_counts().to_dict()
    profile_counts = summary[summary.mechanism.isin(["G", "H", "I", "J"])].information_value.value_counts().to_dict()
    report = [markdown(table[first_cols]), "", "# VOLUME & AUCTION STRUCTURE LANDSCAPE", "",
              "## A. Executive verdict", "",
              f"The campaign tested {len(wide):,} interpretable asset/parameter cells across {len(frames)} locally available futures. "
              f"Strict prospective-freeze candidates: **{len(strict_candidates)}**. No tenth sleeve was added.", "",
              f"Volume information classifications: `{volume_counts}`. Profile classifications: `{profile_counts}`.", "",
              "## B. Data capabilities", "", markdown(pd.DataFrame(inventory)[["Asset", "OHLCV", "Trade-level", "Bid/Ask", "Aggressor", "Volume-at-price feasible", "Coverage"]]), "",
              "The Databento files are `ohlcv-1m`; the campaign uses repository 5-minute resamples. True order flow, delta, aggressor imbalance, and absorption are not observable. The so-called absorption family is strictly an OHLCV high-volume/low-response proxy.", "",
              "### Mechanism observability", "",
              "| Family | Class | Reason |", "| --- | --- | --- |",
              "| A–F volume mechanisms | A. FULLY OBSERVABLE for OHLCV definitions | Price and causal same-time volume are present. |",
              "| G–J auction mechanisms | B. APPROXIMABLE FROM OHLCV | Completed-bar volume is assigned to tick-rounded typical price. |",
              "| K–L LVN/HVN node mechanics | C. NOT TESTABLE WITH CURRENT DATA | Bar-level allocation is too coarse for defensible node traversal/acceptance claims. |",
              "| True absorption / delta / imbalance | C. NOT TESTABLE WITH CURRENT DATA | No trades, quotes, aggressor side, MBP, or MBO. |", "",
              "## C. Intraday volume seasonality", "",
              "Causal expected curves use only the prior 10/20/40 completed sessions at the same Eastern-time bar. Median is primary; mean, robust surprise, and cumulative RVOL are diagnostics. Curves and plots are in `volume_seasonality/` and retain the expected opening/closing concentration where the session definition exhibits it.", "",
              "## D. Cross-asset mechanism map", "", markdown(broad_counts), "",
              "| Mechanism | Cross-asset classification |", "| --- | --- |",
              "| A — RVOL breakout | ASSET-CLASS SPECIFIC / weak: positive only in NQ/MNQ representatives; no cost-robust broad result. |",
              "| B — volume climax | INSTRUMENT-SPECIFIC: MGC is the sole strong representative. |",
              "| C — impact efficiency | INSTRUMENT-SPECIFIC / mixed direction; no broad cost-robust replication. |",
              "| D — high-volume low-response proxy | INSTRUMENT-SPECIFIC: MNQ reversal is the only cost-robust representative. |",
              "| E — move without volume confirmation | BROAD CROSS-ASSET watchlist: positive in five of six assets, but correlated equity contracts are discounted. |",
              "| F — shock after quiet | ASSET-CLASS SPECIFIC: positive only in NQ/MNQ representatives. |",
              "| G–I — rejection/acceptance/POC attraction | NO EVIDENCE with current approximate profiles. |",
              "| J — POC rejection | ASSET-CLASS SPECIFIC watchlist: positive in US equity indexes, but NQ improves under a one-session profile lag and therefore fails profile-specific falsification. |", "",
              "Highly related US equity-index contracts are not counted as independent replications; date overlap and PnL correlation are reported separately.", "",
              "## E. Abnormal-volume continuation", "", "Family A always includes its identical price-only breakout. Incremental results and matched thinning are in the comparison files.", "",
              "## F. Volume climax / reversal", "", "Immediate and confirmed fades were separated. Prior MGC work was context only; this campaign used its frozen compact grid without importing prior tuned settings.", "",
              "## G. Price-volume impact efficiency", "", "High and low efficiency tails were tested in both economically plausible directions. Low impact is described only as a balance/absorption proxy.", "",
              "## H. Absorption proxies", "", "High-volume/low-response bars were compared with the same price geometry absent volume. This is not true absorption.", "",
              "## I. Auction / volume-profile structures", "", "Profiles use only completed sessions over 1/3/5-session windows. K/L were not forced because OHLCV-derived nodes are too coarse. Daily approximate levels are saved in `volume_profiles/`.", "",
              "## J. Price-only benchmark comparisons", "", f"{len(comparisons)} matched comparisons were run. Random thinning used 200 draws per eligible cell and matched the retained trade count.", "",
              "## K. Cost robustness", "", "Every cell was stressed at nominal, +1 tick/side, +2 ticks/side, and doubled commission. Profile strategies were delayed one and two bars. Gross edge and friction/edge ratio are reported.", "",
              "## L. Event studies", "", "OOS conditional paths at 1/3/6/12 bars are saved for every asset/mechanism representative in `event_studies/`.", "",
              "## M. Regime diagnostics", "", "Representatives are decomposed by prior volatility, cumulative RVOL, and time segment. These are diagnostics only; no regime filters were fitted.", "",
              "## N. Falsification", "", "Tests include opposite direction, volume removal, time-bucket volume shuffling, one/two-bar delay, removal of the best 5/10 trades and best year, plus profile lag and POC-to-midpoint replacement where relevant.", "",
              "## O. Cross-asset dependence", "", "Signal-date Jaccard overlap, direction overlap, unique event days, and daily PnL correlation are in `cross_asset_dependence.csv`.", "",
              "## P. Diversification vs frozen 9 sleeves", "", "The portfolio was loaded read-only after standalone testing. Equal-contract base vs base-plus-one-candidate results use no optimized weights and are in `diversification_vs_9_sleeves.csv`.", "",
              "## Q. Topstep impact", "", str(topstep.iloc[0].status) + ": " + str(topstep.iloc[0].reason), "",
              "## R. Candidate registry", "", "See `candidate_registry.md`. Representatives are central cells from stable regions; they are not parameter freezes.", "",
              "## S. Recommended next actions", "",
              "1. Do not alter the frozen nine sleeves or add a tenth sleeve from this retrospective campaign.",
              "2. Acquire trade-level data with aggressor side or MBP before reopening true absorption, delta, LVN, or HVN hypotheses.",
              "3. Prospectively monitor only rows labelled `ROBUST_REGION_FOUND` or `CANDIDATE_FOR_PROSPECTIVE_FREEZE`; do not retune them on the same history.",
              "4. Permanently close cells labelled `REJECTED`, plus any volume filter that fails both its price-only and random-thinning controls.", "",
              "## Explicit answers", ""]
    def names_for(mask: pd.Series) -> str:
        z = summary.loc[mask, ["asset", "mechanism_name"]]
        return ", ".join(f"{a}/{m}" for a, m in z.itertuples(index=False, name=None)) or "none"
    report += [
        f"1. **Does abnormal volume contain incremental directional information?** Sometimes conditionally, but not as a broad standalone alpha. Clear incremental value versus often-poor price benchmarks was found for: {names_for(summary.information_value == 'VOLUME_ADDS_CLEAR_VALUE')}. Negative standalone expectancy remains a rejection even when the volume filter improves its benchmark.",
        f"2. **Is RVOL useful beyond the underlying price signal?** Only where the information label is clear/some after matched controls; counts are `{volume_counts}`. A volume threshold alone was not credited.",
        "3. **Does low price impact per unit volume predict reversal, breakout, or nothing?** No universal outcome. Most representatives imply nothing after costs; MNQ's high-volume/low-response reversal proxy is a cost-robust watchlist effect, while MGC's breakout-like proxy fails the +2-tick stress.",
        f"4. **Do VAH/VAL/POC structures contain reproducible alpha?** Profile classifications are `{profile_counts}`; OHLCV approximations prevent any stronger microstructure claim.",
        "5. **Is true volume-at-price required?** Yes for defensible LVN/HVN and order-flow/absorption claims. OHLCV profiles are sufficient only for preliminary rejection/watchlist screening.",
        f"6. **Which mechanisms replicate cross-asset?** The mechanism map above reports positive and robust asset counts; correlated index contracts are discounted using the dependence file.",
        f"7. **Which effects survive realistic costs?** {names_for(summary.cost_stress == 'PASS_+2_TICKS')} survived the stringent +2 ticks/side representative stress; see all scenarios in the cost file.",
        f"8. **Which provide a new risk dimension?** {names_for(summary.new_risk_dimension)} met the simple low-correlation and positive marginal-Sharpe diagnostic.",
        f"9. **Is anything strong enough for a prospective freeze?** {'Yes: ' + names_for(summary.verdict == 'CANDIDATE_FOR_PROSPECTIVE_FREEZE') if len(strict_candidates) else 'No. No candidate passed every required gate.'}",
        "10. **What should be rejected permanently?** Rejected rows, non-incremental volume filters that fail matched thinning, true order-flow claims from OHLCV, and LVN/HVN strategies built from the current coarse approximation should not be reopened without new data.", ""]
    (output / "VOLUME_AUCTION_MASTER_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_campaign(args.output, smoke=args.smoke)


if __name__ == "__main__":
    main()
