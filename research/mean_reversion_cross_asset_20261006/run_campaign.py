"""Mechanism-first intraday mean-reversion campaign across available futures.

Signals are formed on the close of a five-minute bar and filled no earlier than
the next bar open.  This module deliberately has no dependency on the frozen
nine-sleeve signal code; that portfolio is loaded only for final diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results"
FROZEN_NINE = REPO / "research" / "locked_nine_sleeve_portfolio" / "daily_pnl.csv"


@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick_size: float
    tick_value: float
    commission_side: float
    session_start: str
    session_end: str

    @property
    def point_value(self) -> float:
        return self.tick_value / self.tick_size


INSTRUMENTS = {
    "MNQ": Instrument("MNQ", 0.25, 0.50, 1.25, "09:30", "16:00"),
    "NQ": Instrument("NQ", 0.25, 5.00, 1.25, "09:30", "16:00"),
    "MES": Instrument("MES", 0.25, 1.25, 1.25, "09:30", "16:00"),
    "M2K": Instrument("M2K", 0.10, 0.50, 1.25, "09:30", "16:00"),
    "MGC": Instrument("MGC", 0.10, 1.00, 1.25, "08:20", "13:30"),
    "MBT": Instrument("MBT", 5.00, 0.50, 1.41, "09:30", "16:00"),
    "6E": Instrument("6E", 0.00005, 6.25, 1.25, "08:20", "15:00"),
    "6J": Instrument("6J", 0.0000005, 6.25, 1.25, "08:20", "15:00"),
    "ZN": Instrument("ZN", 1 / 64, 15.625, 1.25, "08:20", "15:00"),
    "ZF": Instrument("ZF", 1 / 128, 7.8125, 1.25, "08:20", "15:00"),
    "MCL": Instrument("MCL", 0.01, 1.00, 1.25, "09:00", "14:30"),
    "NG": Instrument("NG", 0.001, 10.00, 1.25, "09:00", "14:30"),
}

DATA_CANDIDATES = {
    "MNQ": ["data/processed/parquet/MNQ_c_0_5m_20260321_094501.parquet"],
    "NQ": ["data/processed/parquet/NQ_c_0_5m_20260321_095036.parquet"],
    "MES": ["data/processed/parquet/MES_c_0_5m_20260322_135702.parquet"],
    "M2K": ["data/processed/parquet/M2K_c_0_5m_20260322_134808.parquet"],
    "MGC": ["data/processed/parquet/MGC_v_0_5m_20150101_20260923.parquet", "data/processed/parquet/MGC_v_0_1m_20150101_20260923.parquet", "data/processed/parquet/MGC_c_0_5m_20260322_155729.parquet"],
    "MBT": ["data/processed/parquet/MBT_v_0_5m_20210425_20261003.parquet", "data/downloaded/MBT_v_0_1m_20210425_20261003.parquet"],
    "6E": ["data/processed/parquet/6E_v_0_1m.parquet", "data/downloaded/6E_v_0_1m.parquet"],
    "6J": ["data/processed/parquet/6J_v_0_1m.parquet", "data/downloaded/6J_v_0_1m.parquet"],
    "ZN": ["data/processed/parquet/ZN_v_0_1m.parquet", "data/downloaded/ZN_v_0_1m.parquet"],
    "ZF": ["data/processed/parquet/ZF_v_0_1m.parquet", "data/downloaded/ZF_v_0_1m.parquet"],
    "MCL": ["data/processed/parquet/MCL_v_0_1m.parquet", "data/downloaded/MCL_v_0_1m.parquet"],
    "NG": ["data/processed/parquet/NG_v_0_1m.parquet", "data/downloaded/NG_v_0_1m.parquet"],
}

FAMILIES = {
    "vwap_extension": "Session VWAP displacement normalized by causal intraday ATR.",
    "return_overshoot": "Short-horizon standardized return overshoot.",
    "consecutive_exhaustion": "Same-direction bar streak with abnormal cumulative displacement.",
    "range_extreme": "Displacement from a recent rolling range midpoint.",
    "opening_dislocation": "Early move away from the session open, scaled by prior-session range.",
    "gap_reversion": "Gap from prior relevant session close, observed for fifteen minutes.",
    "failed_breakout": "Excursion through a prior-session extreme followed by confirmed re-entry.",
    "short_fair_value": "Multi-bar deviation from a short causal EMA.",
}

EXIT_SPECS = (
    ("fair_value", 24, 1.00, True, 1.00),
    ("partial_50", 24, 0.50, True, 1.00),
    ("time_15", 3, 0.00, False, 0.00),
    ("time_30", 6, 0.00, False, 0.00),
    ("slow_120", 24, 0.00, False, 0.00),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Small dependency-free Markdown renderer (pandas.to_markdown needs tabulate)."""
    view = df.copy()
    if index:
        view = view.reset_index()
    columns = [str(c) for c in view.columns]
    def clean(value: Any) -> str:
        if isinstance(value, float):
            return "" if np.isnan(value) else f"{value:.4g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(clean(v) for v in row) + " |" for row in view.itertuples(index=False, name=None))
    return "\n".join(lines)


def discover_data() -> tuple[dict[str, Path], list[dict[str, Any]]]:
    found: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    for symbol, candidates in DATA_CANDIDATES.items():
        path = next((REPO / p for p in candidates if (REPO / p).exists()), None)
        if path is None:
            # Permit restored files with dated suffixes without changing config.
            matches = sorted((REPO / "data").rglob(f"{symbol}_*_1m_*.parquet"))
            path = matches[-1] if matches else None
        if path is not None:
            found[symbol] = path
            rows.append({"asset": symbol, "status": "AVAILABLE", "path": str(path.relative_to(REPO)), "sha256": _sha256(path), "reason": ""})
        else:
            rows.append({"asset": symbol, "status": "DATA_UNAVAILABLE", "path": "", "sha256": "", "reason": "No local intraday parquet; campaign not run and no inference made."})
    return found, rows


def _load_bars(path: Path, inst: Instrument) -> pd.DataFrame:
    x = pd.read_parquet(path)
    if not isinstance(x.index, pd.DatetimeIndex):
        ts_col = "timestamp" if "timestamp" in x.columns else "ts_event"
        x.index = pd.to_datetime(x.pop(ts_col), errors="raise")
    x.index = pd.to_datetime(x.index)
    if x.index.tz is None:
        x.index = x.index.tz_localize("America/New_York")
    else:
        x.index = x.index.tz_convert("America/New_York")
    x = x.rename(columns={c: c.lower() for c in x.columns}).sort_index()
    x = x[["open", "high", "low", "close", "volume"]].astype(float)
    # Resample only when the source is materially finer than five minutes.
    delta = x.index.to_series().diff().dropna().median()
    if delta < pd.Timedelta(minutes=4):
        x = x.resample("5min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "high", "low", "close"])
    local_time = x.index.strftime("%H:%M")
    x = x[(local_time >= inst.session_start) & (local_time < inst.session_end) & (x.index.dayofweek < 5)].copy()
    x["session_date"] = x.index.date
    return x


def _prepare_features(x: pd.DataFrame) -> pd.DataFrame:
    g = x.groupby("session_date", sort=False, group_keys=False)
    prev_close = x["close"].shift(1)
    tr = pd.concat([(x.high - x.low), (x.high - prev_close).abs(), (x.low - prev_close).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(14, min_periods=8).mean()
    typical = (x.high + x.low + x.close) / 3.0
    x["vwap"] = (typical * x.volume).groupby(x.session_date).cumsum() / x.volume.groupby(x.session_date).cumsum().replace(0, np.nan)
    x["session_open"] = g.open.transform("first")
    x["bar_in_session"] = g.cumcount()
    x["session_last"] = g.cumcount(ascending=False).eq(0)
    for n in (2, 3, 5, 6, 10, 12, 20, 24, 30):
        x[f"ret_{n}"] = g.close.pct_change(n)
        x[f"ema_{n}"] = g.close.transform(lambda s, n=n: s.ewm(span=n, adjust=False).mean())
    for n in (6, 12, 24):
        x[f"roll_high_{n}"] = g.high.transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=n).max())
        x[f"roll_low_{n}"] = g.low.transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=n).min())
        x[f"roll_mid_{n}"] = (x[f"roll_high_{n}"] + x[f"roll_low_{n}"]) / 2
    direction = np.sign(x.close - x.open)
    x["direction"] = direction
    x["up_streak"] = g.direction.transform(lambda s: s.eq(1).groupby(s.ne(s.shift()).cumsum()).cumsum()).where(direction.eq(1), 0)
    x["down_streak"] = g.direction.transform(lambda s: s.eq(-1).groupby(s.ne(s.shift()).cumsum()).cumsum()).where(direction.eq(-1), 0)
    daily = g.agg(day_open=("open", "first"), day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last"))
    prior = daily.shift(1).rename(columns=lambda c: "prior_" + c)
    for col in prior:
        x[col] = pd.Series(x.session_date, index=x.index).map(prior[col])
    x["prior_range"] = x.prior_day_high - x.prior_day_low
    x["minutes_from_open"] = x.bar_in_session * 5
    return x.replace([np.inf, -np.inf], np.nan)


def entry_parameter_grid(family: str) -> list[dict[str, float | int]]:
    if family == "vwap_extension":
        return [{"lookback": 0, "threshold": z} for z in (0.75, 1.0, 1.25, 1.5, 2.0)]
    if family == "return_overshoot":
        return [{"lookback": n, "threshold": z} for n in (2, 3, 5, 10) for z in (1.0, 1.5, 2.0)]
    if family == "consecutive_exhaustion":
        return [{"lookback": n, "threshold": z} for n in (3, 4, 5) for z in (0.75, 1.25)]
    if family == "range_extreme":
        return [{"lookback": n, "threshold": z} for n in (6, 12, 24) for z in (0.75, 1.25, 1.75)]
    if family == "opening_dislocation":
        return [{"lookback": 9, "threshold": z} for z in (0.50, 1.0, 1.5)]
    if family == "gap_reversion":
        return [{"lookback": 3, "threshold": z} for z in (0.25, 0.50, 0.75, 1.0)]
    if family == "failed_breakout":
        return [{"lookback": 1, "threshold": z} for z in (0.0, 0.25, 0.50)]
    if family == "short_fair_value":
        return [{"lookback": n, "threshold": z} for n in (5, 10, 20, 30) for z in (1.0, 1.5, 2.0)]
    raise KeyError(family)


def _signals(x: pd.DataFrame, family: str, p: dict[str, float | int]) -> tuple[pd.Series, pd.Series]:
    n, z = int(p["lookback"]), float(p["threshold"])
    atr = x.atr.replace(0, np.nan)
    if family == "vwap_extension":
        score, fair = (x.close - x.vwap) / atr, x.vwap
        signal = -np.sign(score).where(score.abs() >= z, 0)
    elif family == "return_overshoot":
        score, fair = (x.close - x.close.groupby(x.session_date).shift(n)) / (atr * math.sqrt(max(n, 1))), x[f"ema_{max(5, n)}"]
        signal = -np.sign(score).where(score.abs() >= z, 0)
    elif family == "consecutive_exhaustion":
        prior = x.close.groupby(x.session_date).shift(n)
        displacement = (x.close - prior) / atr
        signal = pd.Series(0.0, index=x.index)
        signal[(x.up_streak >= n) & (displacement >= z)] = -1
        signal[(x.down_streak >= n) & (displacement <= -z)] = 1
        fair = x.ema_10
    elif family == "range_extreme":
        fair = x[f"roll_mid_{n}"]
        score = (x.close - fair) / atr
        signal = -np.sign(score).where(score.abs() >= z, 0)
    elif family == "opening_dislocation":
        fair = x.session_open
        score = (x.close - fair) / x.prior_range.replace(0, np.nan)
        valid = x.minutes_from_open.between(15, 60)
        signal = -np.sign(score).where(valid & (score.abs() >= z), 0)
    elif family == "gap_reversion":
        fair = x.prior_day_close
        gap = (x.session_open - fair) / x.prior_range.replace(0, np.nan)
        valid = x.bar_in_session.eq(3)
        signal = -np.sign(gap).where(valid & (gap.abs() >= z), 0)
    elif family == "failed_breakout":
        buffer = z * atr
        fair = (x.prior_day_high + x.prior_day_low) / 2
        signal = pd.Series(0.0, index=x.index)
        signal[(x.high >= x.prior_day_high + buffer) & (x.close < x.prior_day_high)] = -1
        signal[(x.low <= x.prior_day_low - buffer) & (x.close > x.prior_day_low)] = 1
    elif family == "short_fair_value":
        fair = x[f"ema_{n}"]
        score = (x.close - fair) / atr
        signal = -np.sign(score).where(score.abs() >= z, 0)
    else:
        raise KeyError(family)
    valid_common = x.atr.notna() & (~x.session_last) & x.minutes_from_open.ge(15)
    return signal.fillna(0).astype(int).where(valid_common, 0), fair


def _cost(inst: Instrument, slippage_ticks: float = 1.0, commission_mult: float = 1.0) -> float:
    return 2 * slippage_ticks * inst.tick_value + 2 * inst.commission_side * commission_mult


def backtest(
    x: pd.DataFrame,
    signal: pd.Series,
    fair: pd.Series,
    inst: Instrument,
    exit_spec: tuple[str, int, float, bool, float],
    entry_delay: int = 1,
    flip: bool = False,
) -> pd.DataFrame:
    exit_name, max_bars, fraction, use_target, stop_multiple = exit_spec
    sig = signal.to_numpy(copy=True) * (-1 if flip else 1)
    opens, highs, lows, closes = (x[c].to_numpy() for c in ("open", "high", "low", "close"))
    fair_a, atr = fair.to_numpy(), x.atr.to_numpy()
    dates, times = x.session_date.to_numpy(), x.index.to_numpy()
    trades: list[dict[str, Any]] = []
    next_free = 0
    signal_indices = np.flatnonzero(sig)
    session_last = x.session_last.to_numpy(dtype=bool)
    cursor = 0
    while cursor < len(signal_indices):
        i = int(signal_indices[cursor])
        entry_i = i + entry_delay
        if entry_i <= next_free or entry_i >= len(x) or dates[entry_i] != dates[i]:
            cursor += 1
            continue
        direction = int(sig[i])
        entry = float(opens[entry_i])
        initial_fair = float(fair_a[i]) if np.isfinite(fair_a[i]) else float(closes[i])
        deviation = abs(entry - initial_fair)
        if not np.isfinite(deviation) or deviation <= 0:
            cursor += 1
            continue
        target = entry + direction * deviation * fraction if use_target else np.nan
        stop = entry - direction * max(deviation * max(stop_multiple, 1.0), float(atr[i])) if stop_multiple else np.nan
        end_i = min(entry_i + max_bars, len(x) - 1)
        exit_i, exit_price, reason = end_i, float(closes[end_i]), "time"
        for j in range(entry_i, end_i + 1):
            if dates[j] != dates[entry_i]:
                exit_i, exit_price, reason = j - 1, float(closes[j - 1]), "session_end"
                break
            if use_target:
                stop_hit = lows[j] <= stop if direction == 1 else highs[j] >= stop
                target_hit = highs[j] >= target if direction == 1 else lows[j] <= target
                if stop_hit:  # conservative same-bar ordering
                    exit_i, exit_price, reason = j, stop, "stop"
                    break
                if target_hit:
                    exit_i, exit_price, reason = j, target, "target"
                    break
            if session_last[j]:
                exit_i, exit_price, reason = j, float(closes[j]), "session_end"
                break
        path_h = highs[entry_i : exit_i + 1]
        path_l = lows[entry_i : exit_i + 1]
        mfe_points = (path_h.max() - entry) if direction == 1 else (entry - path_l.min())
        mae_points = (entry - path_l.min()) if direction == 1 else (path_h.max() - entry)
        gross = direction * (exit_price - entry) * inst.point_value
        trades.append({
            "session_date": str(dates[entry_i]), "signal_time": str(pd.Timestamp(times[i])), "entry_time": str(pd.Timestamp(times[entry_i])),
            "exit_time": str(pd.Timestamp(times[exit_i])), "direction": direction, "entry_price": entry, "exit_price": exit_price,
            "fair_value": initial_fair, "initial_deviation": deviation, "exit_reason": reason, "holding_minutes": int((exit_i-entry_i+1)*5),
            "gross_pnl": gross, "nominal_cost": _cost(inst), "net_pnl": gross-_cost(inst),
            "mfe_usd": mfe_points*inst.point_value, "mae_usd": mae_points*inst.point_value,
        })
        next_free = exit_i
        # All intervening signals are non-executable while this trade is open.
        cursor = int(np.searchsorted(signal_indices, next_free, side="right"))
    return pd.DataFrame(trades)


def metrics(trades: pd.DataFrame, split_date: str | None = None) -> dict[str, float]:
    if trades.empty:
        return {k: 0.0 for k in ("trades", "net_pnl", "sharpe", "profit_factor", "expectancy", "win_rate", "avg_winner", "avg_loser", "max_drawdown", "avg_holding_minutes", "turnover", "long_pnl", "short_pnl", "top5_concentration")}
    t = trades.copy()
    if split_date is not None:
        t = t[pd.to_datetime(t.session_date) >= pd.Timestamp(split_date)]
    if t.empty:
        return metrics(pd.DataFrame())
    pnl = t.net_pnl.astype(float)
    daily = t.assign(date=pd.to_datetime(t.session_date)).groupby("date").net_pnl.sum()
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
    daily = daily.reindex(full_idx, fill_value=0.0)
    std = daily.std(ddof=1)
    sharpe = daily.mean() / std * math.sqrt(252) if std > 0 else 0.0
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) else (float("inf") if len(wins) else 0.0)
    dd = daily.cumsum() - daily.cumsum().cummax()
    top5 = pnl.nlargest(min(5, len(pnl))).sum() / pnl.sum() if pnl.sum() > 0 else np.nan
    return {
        "trades": float(len(t)), "net_pnl": float(pnl.sum()), "sharpe": float(sharpe), "profit_factor": float(pf),
        "expectancy": float(pnl.mean()), "win_rate": float((pnl > 0).mean()), "avg_winner": float(wins.mean()) if len(wins) else 0.0,
        "avg_loser": float(losses.mean()) if len(losses) else 0.0, "max_drawdown": float(dd.min()),
        "avg_holding_minutes": float(t.holding_minutes.mean()), "turnover": float(len(t) * 2),
        "long_pnl": float(t.loc[t.direction == 1, "net_pnl"].sum()), "short_pnl": float(t.loc[t.direction == -1, "net_pnl"].sum()),
        "top5_concentration": float(top5) if np.isfinite(top5) else np.nan,
    }


def _prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in values.items()}


def _split_date(x: pd.DataFrame) -> str:
    sessions = pd.Index(sorted(pd.unique(x.session_date)))
    return str(sessions[min(len(sessions) - 1, int(len(sessions) * 0.70))])


def _yearly_rows(trades: pd.DataFrame, base: dict[str, Any]) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    out = []
    for year, group in trades.groupby(pd.to_datetime(trades.session_date).dt.year):
        out.append({**base, "year": int(year), **metrics(group)})
    return out


def _select_region_representatives(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (asset, family, exit_kind), g in results.groupby(["asset", "mechanism", "exit_kind"]):
        positive = g[(g.oos_net_pnl > 0) & (g.oos_profit_factor > 1.0) & (g.oos_trades >= 10)].copy()
        frac = len(positive) / len(g)
        if positive.empty:
            rep = g.sort_values(["oos_net_pnl", "oos_trades"], ascending=False).iloc[0]
        else:
            a = positive[["lookback", "threshold"]].astype(float)
            scale = a.std().replace(0, 1.0)
            center = a.median()
            idx = (((a-center)/scale)**2).sum(axis=1).idxmin()
            rep = positive.loc[idx]
        stable = frac >= 0.40 and len(positive) >= 3
        gate = stable and rep.oos_trades >= 30 and rep.oos_profit_factor >= 1.05 and rep.oos_sharpe >= 0.25 and rep.oos_net_pnl > 0
        rows.append({**rep.to_dict(), "positive_region_fraction": frac, "positive_cells": len(positive), "region_stable": stable, "standalone_gate": gate})
    reps = pd.DataFrame(rows)
    # One diagnostic representative per mechanism/asset, preferring a passing exit region.
    reps["rank_gate"] = reps.standalone_gate.astype(int)
    return reps.sort_values(["asset", "mechanism", "rank_gate", "positive_region_fraction", "oos_trades"], ascending=[True, True, False, False, False]).groupby(["asset", "mechanism"], as_index=False).head(1).drop(columns="rank_gate")


def _plot_heatmaps(results: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    subset = results[results.exit_kind == "time_30"]
    for (asset, family), g in subset.groupby(["asset", "mechanism"]):
        for scope in ("is", "oos"):
            pivot = g.pivot_table(index="lookback", columns="threshold", values=f"{scope}_sharpe", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-1.5, vmax=1.5)
            ax.set_xticks(range(len(pivot.columns)), [str(v) for v in pivot.columns]); ax.set_yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
            ax.set_xlabel("threshold"); ax.set_ylabel("lookback bars"); ax.set_title(f"{asset} {family} {scope.upper()} Sharpe — time 30m")
            fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(output / f"{asset}_{family}_{scope}.png", dpi=130); plt.close(fig)


def _event_study(x: pd.DataFrame, signal: pd.Series, fair: pd.Series, base: dict[str, Any], output: Path) -> dict[str, Any]:
    horizons = (1, 3, 6, 12)
    events = []
    sig_idx = np.flatnonzero(signal.to_numpy())
    close = x.close.to_numpy(); dates = x.session_date.to_numpy(); sig = signal.to_numpy(); fair_a = fair.to_numpy()
    for i in sig_idx:
        if i + max(horizons) >= len(x) or dates[i + max(horizons)] != dates[i]:
            continue
        initial = abs(close[i] - fair_a[i])
        if not np.isfinite(initial) or initial <= 0:
            continue
        row = {"direction": int(sig[i]), "initial_deviation": initial}
        for h in horizons:
            row[f"rev_{h}"] = int(sig[i]) * (close[i+h] - close[i]) / initial
        events.append(row)
    e = pd.DataFrame(events)
    if e.empty:
        return {**base, "events": 0}
    means = [e[f"rev_{h}"].mean() for h in horizons]; medians = [e[f"rev_{h}"].median() for h in horizons]
    fig, ax = plt.subplots(figsize=(6, 3.5)); mins = [h*5 for h in horizons]
    ax.plot(mins, means, marker="o", label="mean"); ax.plot(mins, medians, marker="o", label="median"); ax.axhline(0, color="black", lw=.7)
    ax.set(xlabel="minutes after signal", ylabel="fraction of initial deviation reverted", title=f"{base['asset']} {base['mechanism']} event study"); ax.legend(); fig.tight_layout()
    fig.savefig(output / f"{base['asset']}_{base['mechanism']}.png", dpi=130); plt.close(fig)
    return {**base, "events": len(e), **{f"mean_reversion_{h*5}m": means[k] for k,h in enumerate(horizons)}, **{f"median_reversion_{h*5}m": medians[k] for k,h in enumerate(horizons)}, "partial_reversion_probability_30m": float((e.rev_6 > 0).mean()), "full_reversion_probability_30m": float((e.rev_6 >= 1).mean())}


def _cost_rows(trades: pd.DataFrame, inst: Instrument, base: dict[str, Any], split_date: str) -> list[dict[str, Any]]:
    scenarios = (("nominal", 1, 1), ("plus_1_tick_side", 2, 1), ("plus_2_ticks_side", 3, 1), ("double_commission", 1, 2))
    rows = []
    for name, slip, comm in scenarios:
        t = trades.copy()
        if not t.empty:
            t["net_pnl"] = t.gross_pnl - _cost(inst, slip, comm)
        rows.append({**base, "scenario": name, **_prefixed("oos", metrics(t[t.session_date >= split_date] if not t.empty else t))})
    return rows


def _placebo_rows(x: pd.DataFrame, signal: pd.Series, fair: pd.Series, inst: Instrument, exit_spec: tuple, trades: pd.DataFrame, base: dict[str, Any], split_date: str) -> list[dict[str, Any]]:
    rows = []
    for name, flip, delay in (("opposite_direction", True, 1), ("entry_delayed_one_bar", False, 2)):
        p = backtest(x, signal, fair, inst, exit_spec, entry_delay=delay, flip=flip)
        rows.append({**base, "placebo": name, **_prefixed("oos", metrics(p[p.session_date >= split_date] if not p.empty else p))})
    oos = trades[trades.session_date >= split_date].copy() if not trades.empty else trades
    for n in (5, 10):
        reduced = oos.drop(oos.nlargest(min(n, len(oos)), "net_pnl").index) if len(oos) else oos
        rows.append({**base, "placebo": f"remove_best_{n}_trades", **_prefixed("oos", metrics(reduced))})
    return rows


def _diversification(candidate_trades: dict[str, pd.DataFrame], reps: pd.DataFrame) -> pd.DataFrame:
    if not FROZEN_NINE.exists():
        return pd.DataFrame(columns=["candidate_id", "status"])
    nine = pd.read_csv(FROZEN_NINE)
    date_col = "session_date" if "session_date" in nine.columns else "date"
    nine[date_col] = pd.to_datetime(nine[date_col])
    sleeve_cols = [c for c in nine.columns if c not in {date_col, "nine_equal_weight_raw", "nine_equal_risk"}]
    portfolio_col = "nine_equal_weight_raw" if "nine_equal_weight_raw" in nine.columns else sleeve_cols[-1]
    rows = []
    for cid, t in candidate_trades.items():
        r = reps[reps.candidate_id == cid].iloc[0]
        daily = t.assign(session_date=pd.to_datetime(t.session_date)).groupby("session_date").net_pnl.sum().rename("candidate")
        z = nine.set_index(date_col).join(daily, how="inner").fillna({"candidate": 0})
        if z.empty:
            rows.append({"candidate_id": cid, "status": "NO_OVERLAP"}); continue
        p = z[portfolio_col]
        row: dict[str, Any] = {"candidate_id": cid, "asset": r.asset, "mechanism": r.mechanism, "status": "OK", "corr_9_sleeve": z.candidate.corr(p), "corr_negative_portfolio_days": z.loc[p < 0, "candidate"].corr(p[p < 0])}
        for q in (0.05, 0.10, 0.20):
            row[f"candidate_mean_worst_{int(q*100)}pct_days"] = z.loc[p <= p.quantile(q), "candidate"].mean()
        for col in sleeve_cols[:9]:
            row[f"corr_{col}"] = z.candidate.corr(z[col])
        old = metrics(pd.DataFrame({"session_date": z.index.astype(str), "net_pnl": p, "direction": 0, "holding_minutes": 0}))
        added = metrics(pd.DataFrame({"session_date": z.index.astype(str), "net_pnl": p + z.candidate, "direction": 0, "holding_minutes": 0}))
        row.update({"base_sharpe": old["sharpe"], "added_sharpe": added["sharpe"], "base_max_drawdown": old["max_drawdown"], "added_max_drawdown": added["max_drawdown"]})
        rows.append(row)
    return pd.DataFrame(rows)


def _verdict(row: pd.Series, cost: pd.DataFrame, div: pd.DataFrame, placebo: pd.DataFrame) -> str:
    cid = row.candidate_id
    c = cost[(cost.candidate_id == cid) & (cost.scenario == "plus_1_tick_side")]
    cost_ok = (not c.empty) and float(c.iloc[0].oos_net_pnl) > 0
    p = placebo[placebo.candidate_id == cid]
    remove10 = p[p.placebo == "remove_best_10_trades"]
    opposite = p[p.placebo == "opposite_direction"]
    falsification_ok = (
        not remove10.empty and float(remove10.iloc[0].oos_net_pnl) > 0
        and not opposite.empty and float(opposite.iloc[0].oos_net_pnl) < 0
    )
    if bool(row.standalone_gate) and cost_ok and falsification_ok:
        d = div[div.candidate_id == cid]
        if not d.empty and d.iloc[0].get("status") == "OK" and float(d.iloc[0].get("corr_9_sleeve", 1)) < 0.5:
            return "CANDIDATE_FOR_PROSPECTIVE_FREEZE"
        return "ROBUST_REGION_FOUND"
    if row.oos_net_pnl > 0 and row.oos_trades >= 20:
        return "WEAK_WATCHLIST"
    return "REJECTED"


def finalize_existing(output: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Rebuild verdict-facing artifacts from completed parameter/trade outputs."""
    inventory = pd.read_csv(output / "data_inventory.csv")
    results = pd.read_csv(output / "all_parameter_results.csv")
    reps = pd.read_csv(output / "mechanism_by_asset_summary.csv")
    cost = pd.read_csv(output / "cost_stress_results.csv")
    placebo = pd.read_csv(output / "placebo_results.csv")
    div = pd.read_csv(output / "diversification_vs_9_sleeves.csv")
    reps["verdict"] = reps.apply(lambda row: _verdict(row, cost, div, placebo), axis=1)
    reps.to_csv(output / "mechanism_by_asset_summary.csv", index=False)
    registry_lines = ["# Candidate registry", "", "No strategy is promoted automatically.", ""]
    for _, r in reps.iterrows():
        registry_lines += [f"## {r.candidate_id}", "", f"- Verdict: `{r.verdict}`", f"- Region: `{r.region_stability}` ({r.positive_cells} positive cells; {r.positive_region_fraction:.1%})", f"- OOS: {int(r.oos_trades)} trades, Sharpe {r.oos_sharpe:.3f}, PF {r.oos_profit_factor:.3f}", f"- Costs +1 tick/side: `{r.cost_stress}`", ""]
    (output / "candidate_registry.md").write_text("\n".join(registry_lines), encoding="utf-8")
    _write_report(output, inventory, reps, results, cost, placebo, div)
    return reps


def _write_report(output: Path, inventory: pd.DataFrame, summary: pd.DataFrame, results: pd.DataFrame, cost: pd.DataFrame, placebo: pd.DataFrame, div: pd.DataFrame) -> None:
    compact = summary[["asset", "mechanism", "oos_trades", "oos_sharpe", "oos_profit_factor", "cost_stress", "region_stability", "corr_9_sleeve", "verdict"]].copy()
    compact.columns = ["Asset", "Mechanism", "Trades OOS", "Sharpe OOS", "PF OOS", "Cost stress", "Region stability", "Corr 9-sleeve PF", "Verdict"]
    available = inventory[inventory.status == "AVAILABLE"].asset.tolist(); unavailable = inventory[inventory.status != "AVAILABLE"].asset.tolist()
    survivors = summary[summary.verdict.isin(["ROBUST_REGION_FOUND", "CANDIDATE_FOR_PROSPECTIVE_FREEZE"])]
    mechanisms = ", ".join(sorted(survivors.mechanism.unique())) if len(survivors) else "aucun"
    assets = ", ".join(sorted(survivors.asset.unique())) if len(survivors) else "aucun"
    lines = [
        "# Mean Reversion — Master Report", "", _markdown(compact), "", "## A. Executive verdict", "",
        f"Campagne exécutée sur {', '.join(available)}. Données absentes, sans conclusion statistique : {', '.join(unavailable) or 'aucune'}.",
        "Les signaux sont connus à la clôture de t et exécutés au plus tôt à l'ouverture de t+1. Les neuf sleeves gelées ne participent jamais aux signaux ni à la sélection standalone.", "",
        "## B. Cross-asset map", "", _markdown(summary.pivot(index="mechanism", columns="asset", values="verdict").fillna("DATA_UNAVAILABLE"), index=True), "",
        "## C. Results by mechanism", "", _markdown(summary.groupby("mechanism").agg(assets_tested=("asset","nunique"), positive_oos=("oos_net_pnl",lambda s:int((s>0).sum())), robust=("verdict",lambda s:int(s.isin(["ROBUST_REGION_FOUND","CANDIDATE_FOR_PROSPECTIVE_FREEZE"]).sum()))), index=True), "",
        "## D. Results by asset", "", _markdown(summary.groupby("asset").agg(mechanisms_tested=("mechanism","nunique"), positive_oos=("oos_net_pnl",lambda s:int((s>0).sum())), robust=("verdict",lambda s:int(s.isin(["ROBUST_REGION_FOUND","CANDIDATE_FOR_PROSPECTIVE_FREEZE"]).sum()))), index=True), "",
        "## E. Entry vs exit decomposition", "", "Chaque entrée est testée avec retour à la juste valeur, retour partiel à 50 %, sorties temps 15/30 minutes et sortie lente 120 minutes. Une amélioration confinée à `slow_120` est signalée comme exposition potentiellement tendancielle, pas comme preuve de réversion.", "",
        _markdown(results.groupby(["mechanism","exit_kind"]).agg(cells=("candidate_id","count"), mean_oos_sharpe=("oos_sharpe","mean"), positive_fraction=("oos_net_pnl",lambda s:float((s>0).mean()))).round(3), index=True), "",
        "## F. Cost robustness", "", _markdown(cost.groupby("scenario").agg(candidates=("candidate_id","nunique"), positive_oos=("oos_net_pnl",lambda s:int((s>0).sum())), median_oos_pnl=("oos_net_pnl","median")).round(2), index=True), "",
        "## G. Falsification tests", "", _markdown(placebo.groupby("placebo").agg(candidates=("candidate_id","nunique"), positive_oos=("oos_net_pnl",lambda s:int((s>0).sum())), median_oos_sharpe=("oos_sharpe","median")).round(3), index=True) if len(placebo) else "Aucun candidat représentatif.", "",
        "## H. Diversification vs current 9 sleeves", "", _markdown(div) if len(div) else "Aucun candidat standalone admissible ou aucun chevauchement.", "",
        "## I. Topstep incremental results", "", "Non exécuté : cette phase est réservée aux candidats qui franchissent les gates standalone et diversification. Aucun signal n'est promu automatiquement.", "",
        "## J. Recommended next actions", "", "1. Restaurer 6E, 6J, ZN, ZF, MCL et NG avant toute conclusion à leur sujet.\n2. Conserver toute région survivante comme hypothèse de suivi, sans choisir la cellule au Sharpe maximal.\n3. N'exécuter Topstep qu'après gel prospectif explicite.", "",
        "## Answers", "",
        f"1. **Convincing intraday mean reversion?** {'Oui, dans au moins une région robuste.' if len(survivors) else 'Non démontrée par les gates strictes sur les données disponibles.'}",
        f"2. **Mechanisms surviving?** {mechanisms}.", f"3. **Assets showing it most clearly?** {assets}.",
        "4. **True reversion or exit design?** Voir la décomposition par sortie; les cas uniquement `slow_120` ne sont pas qualifiés comme réversion robuste.",
        f"5. **New risk dimension vs nine sleeves?** {'Au moins un candidat passe le diagnostic de corrélation.' if (summary.verdict=='CANDIDATE_FOR_PROSPECTIVE_FREEZE').any() else 'Aucune nouvelle dimension suffisamment validée à ce stade.'}",
        f"6. **Strong enough to freeze prospectively?** {'Oui, comme candidat seulement — aucune promotion live.' if (summary.verdict=='CANDIDATE_FOR_PROSPECTIVE_FREEZE').any() else 'Non.'}", "",
        "## Reproducibility and exclusions", "", _markdown(inventory), "",
    ]
    (output / "MEAN_REVERSION_MASTER_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(output: Path = DEFAULT_OUTPUT, quick: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True); (output / "heatmaps").mkdir(exist_ok=True); (output / "event_studies").mkdir(exist_ok=True); (output / "candidate_trades").mkdir(exist_ok=True)
    found, inventory_rows = discover_data()
    inventory = pd.DataFrame(inventory_rows); inventory.to_csv(output / "data_inventory.csv", index=False)
    all_rows: list[dict[str, Any]] = []; yearly: list[dict[str, Any]] = []
    prepared: dict[str, pd.DataFrame] = {}; split_dates: dict[str, str] = {}
    for symbol, path in found.items():
        inst = INSTRUMENTS[symbol]; x = _prepare_features(_load_bars(path, inst)); prepared[symbol] = x; split = _split_date(x); split_dates[symbol] = split
        print(f"prepared {symbol}: {len(x):,} RTH bars, split {split}", flush=True)
        families: Iterable[str] = list(FAMILIES)[:2] if quick else FAMILIES
        for family in families:
            params = entry_parameter_grid(family)[:2] if quick else entry_parameter_grid(family)
            for p in params:
                signal, fair = _signals(x, family, p)
                exits = EXIT_SPECS[:2] if quick else EXIT_SPECS
                for exit_spec in exits:
                    trades = backtest(x, signal, fair, inst, exit_spec)
                    is_t = trades[trades.session_date < split] if not trades.empty else trades; oos_t = trades[trades.session_date >= split] if not trades.empty else trades
                    cid = f"{symbol}__{family}__l{p['lookback']}__z{p['threshold']}__{exit_spec[0]}"
                    base = {"candidate_id": cid, "asset": symbol, "mechanism": family, "lookback": p["lookback"], "threshold": p["threshold"], "exit_kind": exit_spec[0], "split_date": split}
                    all_rows.append({**base, **_prefixed("full", metrics(trades)), **_prefixed("is", metrics(is_t)), **_prefixed("oos", metrics(oos_t))})
                    yearly.extend(_yearly_rows(trades, base))
            print(f"completed {symbol} / {family}", flush=True)
    results = pd.DataFrame(all_rows); results.to_csv(output / "all_parameter_results.csv", index=False)
    pd.DataFrame(yearly).to_csv(output / "yearly_results.csv", index=False)
    if results.empty:
        raise RuntimeError("No campaign result: no usable dataset was discovered.")
    reps = _select_region_representatives(results)
    _plot_heatmaps(results, output / "heatmaps")
    cost_rows: list[dict[str, Any]] = []; placebo_rows: list[dict[str, Any]] = []; event_rows: list[dict[str, Any]] = []
    candidate_trades: dict[str, pd.DataFrame] = {}
    for _, r in reps.iterrows():
        symbol, family = str(r.asset), str(r.mechanism); inst, x = INSTRUMENTS[symbol], prepared[symbol]
        p = {"lookback": int(r.lookback), "threshold": float(r.threshold)}; signal, fair = _signals(x, family, p)
        exit_spec = next(e for e in EXIT_SPECS if e[0] == r.exit_kind)
        trades = backtest(x, signal, fair, inst, exit_spec); candidate_trades[r.candidate_id] = trades
        trades.to_csv(output / "candidate_trades" / f"{r.candidate_id}.csv", index=False)
        base = {"candidate_id": r.candidate_id, "asset": symbol, "mechanism": family}
        cost_rows.extend(_cost_rows(trades, inst, base, split_dates[symbol]))
        placebo_rows.extend(_placebo_rows(x, signal, fair, inst, exit_spec, trades, base, split_dates[symbol]))
        event_rows.append(_event_study(x, signal, fair, base, output / "event_studies"))
    cost = pd.DataFrame(cost_rows); cost.to_csv(output / "cost_stress_results.csv", index=False)
    placebo = pd.DataFrame(placebo_rows); placebo.to_csv(output / "placebo_results.csv", index=False)
    pd.DataFrame(event_rows).to_csv(output / "event_study_summary.csv", index=False)
    div_candidates = {cid: t for cid, t in candidate_trades.items() if bool(reps.loc[reps.candidate_id == cid, "standalone_gate"].iloc[0])}
    div = _diversification(div_candidates, reps); div.to_csv(output / "diversification_vs_9_sleeves.csv", index=False)
    reps["cost_stress"] = reps.candidate_id.map(cost[cost.scenario == "plus_1_tick_side"].set_index("candidate_id").oos_net_pnl.gt(0).map({True:"PASS",False:"FAIL"})).fillna("NOT_TESTED")
    reps["region_stability"] = reps.region_stable.map({True:"BROAD",False:"FRAGILE"})
    reps["corr_9_sleeve"] = reps.candidate_id.map(div.set_index("candidate_id").corr_9_sleeve if "corr_9_sleeve" in div else pd.Series(dtype=float))
    reps["verdict"] = reps.apply(lambda row: _verdict(row, cost, div, placebo), axis=1)
    reps.to_csv(output / "mechanism_by_asset_summary.csv", index=False)
    registry_lines = ["# Candidate registry", "", "No strategy is promoted automatically.", ""]
    for _, r in reps.iterrows():
        registry_lines += [f"## {r.candidate_id}", "", f"- Verdict: `{r.verdict}`", f"- Region: `{r.region_stability}` ({r.positive_cells} positive cells; {r.positive_region_fraction:.1%})", f"- OOS: {int(r.oos_trades)} trades, Sharpe {r.oos_sharpe:.3f}, PF {r.oos_profit_factor:.3f}", f"- Costs +1 tick/side: `{r.cost_stress}`", ""]
    (output / "candidate_registry.md").write_text("\n".join(registry_lines), encoding="utf-8")
    _write_report(output, inventory, reps, results, cost, placebo, div)
    metadata = {"config": json.loads((HERE / "campaign_config.json").read_text(encoding="utf-8")), "available_assets": sorted(found), "files": {p.name: _sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(); run(args.output, quick=args.quick)


if __name__ == "__main__":
    main()
