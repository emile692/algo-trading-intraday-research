"""Frozen MNQ/MGC shock-recovery and overnight-to-RTH research campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT = ROOT / "research" / "conditional_shock_overnight_transfer_20261002"
SPEC_PATH = OUT / "frozen_specification.json"
START, END = pd.Timestamp("2019-05-06"), pd.Timestamp("2026-03-19")
SEED = 20261002


@dataclass(frozen=True)
class Instrument:
    symbol: str
    path: Path
    rth_start: str
    rth_end: str
    tick: float
    point_value: float


INSTRUMENTS = {
    "MNQ": Instrument("MNQ", ROOT / "data/processed/parquet/MNQ_c_0_1m_20260321_094501.parquet", "09:30", "16:00", 0.25, 2.0),
    "MGC": Instrument("MGC", ROOT / "data/processed/parquet/MGC_v_0_1m_20150101_20260923.parquet", "08:20", "13:30", 0.10, 10.0),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    def default(x: Any) -> Any:
        if isinstance(x, (pd.Timestamp, pd.Timedelta, Path)):
            return str(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return None if not np.isfinite(x) else float(x)
        raise TypeError(type(x).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=default, allow_nan=False) + "\n", encoding="utf-8")


def split_label(day: Any) -> str:
    d = pd.Timestamp(day)
    if d < pd.Timestamp("2023-01-01"):
        return "is"
    if d < pd.Timestamp("2024-01-01"):
        return "validation"
    return "oos"


def load_bars(inst: Instrument) -> pd.DataFrame:
    x = pd.read_parquet(inst.path).sort_index()
    if not isinstance(x.index, pd.DatetimeIndex):
        x = x.set_index("timestamp")
    if x.index.tz is None:
        x.index = x.index.tz_localize("UTC")
    x.index = x.index.tz_convert("America/New_York")
    x = x.loc[(x.index.date >= START.date()) & (x.index.date <= END.date())].copy()
    return x


def session_frames(bars: pd.DataFrame, inst: Instrument) -> tuple[dict[pd.Timestamp, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    local = bars.copy()
    local["session_date"] = pd.to_datetime(local.index.date)
    rth = local.between_time(inst.rth_start, inst.rth_end, inclusive="left")
    sessions: dict[pd.Timestamp, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    five_parts: list[pd.DataFrame] = []
    prior_close = np.nan
    for day, frame in rth.groupby("session_date", sort=True):
        day = pd.Timestamp(day)
        if day.weekday() >= 5 or frame.empty:
            continue
        start_ts = pd.Timestamp(f"{day.date()} {inst.rth_start}", tz="America/New_York")
        end_ts = pd.Timestamp(f"{day.date()} {inst.rth_end}", tz="America/New_York")
        has_open = start_ts in frame.index
        usable = bool(has_open and len(frame) >= 30)
        if not usable:
            continue
        frame = frame.loc[(frame.index >= start_ts) & (frame.index < end_ts)].copy()
        # Never interpret a roll jump as an intraday shock or signal.
        if "instrument_id" in frame and frame.instrument_id.nunique() != 1:
            continue
        sessions[day] = frame
        true_range = max(float(frame.high.max() - frame.low.min()), abs(float(frame.high.max()) - prior_close) if np.isfinite(prior_close) else 0.0, abs(float(frame.low.min()) - prior_close) if np.isfinite(prior_close) else 0.0)
        close = float(frame.close.iloc[-1])
        rows.append({"session_date": day, "open_ts": start_ts, "close_ts": frame.index[-1], "open": float(frame.open.iloc[0]), "high": float(frame.high.max()), "low": float(frame.low.min()), "close": close, "volume": float(frame.volume.sum()), "true_range": true_range, "bars": len(frame), "shortened": frame.index[-1] < end_ts - pd.Timedelta(minutes=2), "open_instrument_id": frame.instrument_id.iloc[0] if "instrument_id" in frame else np.nan, "close_instrument_id": frame.instrument_id.iloc[-1] if "instrument_id" in frame else np.nan})
        prior_close = close
        f = frame[["open", "high", "low", "close", "volume"]].resample("5min", origin=start_ts, label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open", "close"])
        f["session_date"] = day
        five_parts.append(f)
    daily = pd.DataFrame(rows).sort_values("session_date").reset_index(drop=True)
    daily["atr14"] = daily["true_range"].ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().shift(1)
    daily["split"] = daily.session_date.map(split_label)
    five = pd.concat(five_parts).sort_index() if five_parts else pd.DataFrame()
    five = five.merge(daily[["session_date", "atr14"]], on="session_date", how="left").set_index(five.index)
    return sessions, daily, five


def next_open(frame: pd.DataFrame, signal_end: pd.Timestamp, delay_bars: int = 0) -> tuple[pd.Timestamp | None, float | None]:
    wanted = signal_end + pd.Timedelta(minutes=delay_bars)
    if wanted not in frame.index:
        return None, None
    return wanted, float(frame.at[wanted, "open"])


def shock_events(symbol: str, sessions: dict[pd.Timestamp, pd.DataFrame], daily: pd.DataFrame, five: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    dmap = daily.set_index("session_date")
    for day, bars5 in five.groupby("session_date", sort=True):
        day = pd.Timestamp(day)
        atr = float(dmap.at[day, "atr14"])
        if not np.isfinite(atr) or atr <= 0 or day not in sessions:
            continue
        one = sessions[day]
        for lookback in (30, 60, 120):
            n = lookback // 5
            prior_high = bars5.high.rolling(n, min_periods=n).max().shift(1)
            prior_low = bars5.low.rolling(n, min_periods=n).min().shift(1)
            for side in ("down", "up"):
                intensity = (prior_high - bars5.close) / atr if side == "down" else (bars5.close - prior_low) / atr
                for threshold in (1.5, 2.0, 2.5):
                    hit = intensity.ge(threshold)
                    if not hit.any():
                        continue
                    ts = hit[hit].index[0]
                    pos = bars5.index.get_loc(ts)
                    shock_end = ts + pd.Timedelta(minutes=5)
                    origin = float(prior_high.loc[ts] if side == "down" else prior_low.loc[ts])
                    extreme = float(bars5.loc[ts, "low"] if side == "down" else bars5.loc[ts, "high"])
                    shock_close = float(bars5.loc[ts, "close"])
                    midpoint = (origin + extreme) / 2
                    base = {"instrument": symbol, "session_date": day, "split": split_label(day), "side": side, "lookback_minutes": lookback, "threshold_atr": threshold, "shock_ts": ts, "shock_end": shock_end, "intensity_atr": float(intensity.loc[ts]), "origin": origin, "extreme": extreme, "midpoint": midpoint, "shock_close": shock_close, "overlap_count": 0}
                    candidates = bars5.iloc[pos + 1 : pos + 7]
                    canonical_ts = alternative_ts = None
                    run_extreme = extreme
                    prev = bars5.iloc[pos]
                    for cts, row in candidates.iterrows():
                        no_new = row.low >= run_extreme if side == "down" else row.high <= run_extreme
                        canon = no_new and (row.close > prev.high if side == "down" else row.close < prev.low)
                        alt = no_new and (row.close > (prev.high + prev.low) / 2 if side == "down" else row.close < (prev.high + prev.low) / 2)
                        if canon and canonical_ts is None:
                            canonical_ts = cts + pd.Timedelta(minutes=5)
                        if alt and alternative_ts is None:
                            alternative_ts = cts + pd.Timedelta(minutes=5)
                        run_extreme = min(run_extreme, float(row.low)) if side == "down" else max(run_extreme, float(row.high))
                        prev = row
                    for entry_type, entry_signal in (("immediate", shock_end), ("canonical_reclaim", canonical_ts), ("midpoint_reclaim", alternative_ts)):
                        if entry_signal is None:
                            continue
                        entry_ts, entry_price = next_open(one, entry_signal)
                        if entry_ts is None:
                            continue
                        direction = 1 if side == "down" else -1
                        rec = base | {"entry_type": entry_type, "entry_ts": entry_ts, "entry_price": entry_price, "direction": direction, "recovered_midpoint": False, "recovered_origin": False}
                        path = one.loc[(one.index >= entry_ts) & (one.index <= entry_ts + pd.Timedelta(minutes=120))]
                        if path.empty:
                            continue
                        favorable = direction * (path.close.astype(float) - entry_price)
                        rec["mfe_points"] = float(favorable.max())
                        rec["mae_points"] = float(favorable.min())
                        rec["recovered_midpoint"] = bool((path.high >= midpoint).any()) if direction == 1 else bool((path.low <= midpoint).any())
                        rec["recovered_origin"] = bool((path.high >= origin).any()) if direction == 1 else bool((path.low <= origin).any())
                        for horizon in (5, 15, 30, 60, 120):
                            exit_ts = entry_ts + pd.Timedelta(minutes=horizon)
                            rec[f"fwd_{horizon}m_bps"] = direction * (float(one.at[exit_ts, "open"]) / entry_price - 1) * 10000 if exit_ts in one.index else np.nan
                        records.append(rec)
    events = pd.DataFrame(records)
    if not events.empty:
        counts = events.groupby(["instrument", "session_date", "side", "shock_ts"])["threshold_atr"].transform("size")
        events["overlap_count"] = counts
    return events


def overnight_events(symbol: str, bars: pd.DataFrame, sessions: dict[pd.Timestamp, pd.DataFrame], daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(1, len(daily)):
        prev, cur = daily.iloc[i - 1], daily.iloc[i]
        day = pd.Timestamp(cur.session_date)
        if day not in sessions or not np.isfinite(cur.atr14):
            continue
        one = sessions[day]
        overnight = bars.loc[(bars.index > prev.close_ts) & (bars.index < cur.open_ts)]
        if len(overnight) < 30:
            continue
        roll_safe = not ("instrument_id" in bars and (prev.close_instrument_id != cur.open_instrument_id or overnight.instrument_id.nunique() != 1))
        if not roll_safe:
            continue
        on_high, on_low = float(overnight.high.max()), float(overnight.low.min())
        on_close, on_open = float(overnight.close.iloc[-1]), float(prev.close)
        on_range = on_high - on_low
        vwap = float((overnight.close * overnight.volume).sum() / overnight.volume.sum()) if overnight.volume.sum() else np.nan
        rec: dict[str, Any] = {"instrument": symbol, "session_date": day, "split": split_label(day), "prior_rth_close": on_open, "rth_open": float(cur.open), "overnight_close": on_close, "overnight_high": on_high, "overnight_low": on_low, "overnight_vwap": vwap, "atr14": float(cur.atr14), "overnight_return": on_close / on_open - 1, "overnight_move_atr": (on_close - on_open) / cur.atr14, "overnight_range_atr": on_range / cur.atr14, "open_location": (cur.open - on_low) / on_range if on_range > 0 else np.nan, "open_distance_atr": (cur.open - on_open) / cur.atr14, "close_location": (on_close - on_low) / on_range if on_range > 0 else np.nan, "open_ts": cur.open_ts, "close_ts": cur.close_ts}
        for a, b, name in ((0, 15, "open_15"), (15, 30, "15_30"), (30, None, "30_noon"), (0, None, "open_close")):
            start_ts = cur.open_ts + pd.Timedelta(minutes=a)
            if b is None:
                end_ts = cur.close_ts if name == "open_close" else min(cur.close_ts, pd.Timestamp(f"{day.date()} 12:00", tz="America/New_York"))
            else:
                end_ts = cur.open_ts + pd.Timedelta(minutes=b)
            rec[f"return_{name}_bps"] = (float(one.at[end_ts, "open"]) / float(one.at[start_ts, "open"]) - 1) * 10000 if start_ts in one.index and end_ts in one.index else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        is_vals = out.loc[out.split == "is", "atr14"].dropna()
        q = is_vals.quantile([1 / 3, 2 / 3]).to_numpy()
        out["vol_regime"] = pd.cut(out.atr14, [-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"])
        out["overnight_bin"] = pd.cut(out.overnight_move_atr, [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf], labels=["<-1.5", "-1.5:-0.5", "-0.5:0.5", "0.5:1.5", ">1.5"], ordered=True)
        out["lagged_sign"] = np.sign(out.overnight_move_atr.shift(1)).fillna(0)
    return out


def simulate_trade(frame: pd.DataFrame, entry_ts: pd.Timestamp, direction: int, stop: float | None, target: float | None, exit_ts: pd.Timestamp, inst: Instrument, delay_bars: int = 0, slip_ticks: int = 1, commission_mult: float = 1.0) -> dict[str, Any] | None:
    actual_entry = entry_ts + pd.Timedelta(minutes=delay_bars)
    if actual_entry not in frame.index:
        return None
    raw_entry = float(frame.at[actual_entry, "open"])
    entry = raw_entry + direction * slip_ticks * inst.tick
    path = frame.loc[(frame.index >= actual_entry) & (frame.index <= exit_ts)]
    if path.empty:
        return None
    raw_exit, reason, actual_exit = float(path.close.iloc[-1]), "time_exit", path.index[-1]
    for ts, row in path.iloc[1:].iterrows():
        stopped = stop is not None and (row.low <= stop if direction == 1 else row.high >= stop)
        targeted = target is not None and (row.high >= target if direction == 1 else row.low <= target)
        if stopped:
            raw_exit, reason, actual_exit = float(stop), "stop", ts
            break
        if targeted:
            raw_exit, reason, actual_exit = float(target), "target", ts
            break
    exit_price = raw_exit - direction * slip_ticks * inst.tick
    gross = direction * (raw_exit - raw_entry) * inst.point_value
    commissions = 2 * 1.25 * commission_mult
    slippage = 2 * slip_ticks * inst.tick * inst.point_value
    favorable = direction * (path.loc[:actual_exit, "high"] - entry) if direction == 1 else direction * (path.loc[:actual_exit, "low"] - entry)
    adverse = direction * (path.loc[:actual_exit, "low"] - entry) if direction == 1 else direction * (path.loc[:actual_exit, "high"] - entry)
    return {"entry_ts": actual_entry, "exit_ts": actual_exit, "direction": direction, "entry_price": entry, "exit_price": exit_price, "gross_pnl_usd": gross, "commissions_usd": commissions, "slippage_usd": slippage, "net_pnl_usd": gross - commissions - slippage, "exit_reason": reason, "holding_minutes": (actual_exit - actual_entry).total_seconds() / 60, "mfe_usd": float(favorable.max() * inst.point_value), "mae_usd": float(adverse.min() * inst.point_value)}


def shock_trades(events: pd.DataFrame, sessions: dict[pd.Timestamp, pd.DataFrame], inst: Instrument) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    use = events.loc[events.entry_type == "canonical_reclaim"]
    for _, e in use.iterrows():
        frame = sessions[pd.Timestamp(e.session_date)]
        direction = int(e.direction)
        stop = float(e.extreme - inst.tick if direction == 1 else e.extreme + inst.tick)
        end = min(pd.Timestamp(e.entry_ts) + pd.Timedelta(minutes=120), frame.index[-1])
        for stress, kwargs in {"nominal": {}, "plus1tick": {"slip_ticks": 2}, "plus2ticks": {"slip_ticks": 3}, "plus3ticks": {"slip_ticks": 4}, "double_commission": {"commission_mult": 2}, "delay_one_bar": {"delay_bars": 1}}.items():
            trade = simulate_trade(frame, pd.Timestamp(e.entry_ts), direction, stop, float(e.midpoint), end, inst, **kwargs)
            if trade:
                rows.append(trade | {"instrument": inst.symbol, "mechanism": "conditional_shock_recovery", "side": "long" if direction == 1 else "short", "session_date": pd.Timestamp(e.session_date), "split": e.split, "lookback_minutes": int(e.lookback_minutes), "threshold": float(e.threshold_atr), "confirmation_minutes": 5, "stress": stress})
    return pd.DataFrame(rows)


def overnight_trades(events: pd.DataFrame, sessions: dict[pd.Timestamp, pd.DataFrame], inst: Instrument) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, e in events.iterrows():
        sign = 1 if e.overnight_move_atr > 0 else -1
        if sign == 0:
            continue
        frame = sessions[pd.Timestamp(e.session_date)]
        midpoint = (e.overnight_high + e.overnight_low) / 2
        for threshold in (0.5, 1.0, 1.5):
            if abs(e.overnight_move_atr) < threshold:
                continue
            for confirm in (15, 30):
                signal_end = pd.Timestamp(e.open_ts) + pd.Timedelta(minutes=confirm)
                if signal_end not in frame.index:
                    continue
                confirm_close = float(frame.loc[frame.index < signal_end, "close"].iloc[-1])
                acceptance = (confirm_close > e.rth_open and confirm_close > midpoint) if sign == 1 else (confirm_close < e.rth_open and confirm_close < midpoint)
                outer_open = e.open_location >= 0.75 if sign == 1 else e.open_location <= 0.25
                rejection = outer_open and ((confirm_close < midpoint) if sign == 1 else (confirm_close > midpoint))
                for mechanism, ok, direction, target, stop in (
                    ("overnight_acceptance_continuation", acceptance, sign, None, None),
                    ("overnight_rejection_fade", rejection, -sign, midpoint, e.overnight_high + inst.tick if sign == 1 else e.overnight_low - inst.tick),
                ):
                    if not ok:
                        continue
                    for stress, kwargs in {"nominal": {}, "plus1tick": {"slip_ticks": 2}, "plus2ticks": {"slip_ticks": 3}, "plus3ticks": {"slip_ticks": 4}, "double_commission": {"commission_mult": 2}, "delay_one_bar": {"delay_bars": 1}}.items():
                        trade = simulate_trade(frame, signal_end, direction, stop, target, pd.Timestamp(e.close_ts), inst, **kwargs)
                        if trade:
                            rows.append(trade | {"instrument": inst.symbol, "mechanism": mechanism, "side": "long" if direction == 1 else "short", "session_date": pd.Timestamp(e.session_date), "split": e.split, "lookback_minutes": 0, "threshold": threshold, "confirmation_minutes": confirm, "stress": stress})
    return pd.DataFrame(rows)


def metric_row(x: pd.DataFrame) -> dict[str, Any]:
    pnl = x.net_pnl_usd.astype(float) if len(x) else pd.Series(dtype=float)
    gross = x.gross_pnl_usd.astype(float) if len(x) else pd.Series(dtype=float)
    if pnl.empty:
        return {"trades": 0, "pnl_usd": 0.0, "gross_pnl_usd": 0.0, "expectancy_usd": np.nan, "sharpe": np.nan, "sortino": np.nan, "profit_factor": np.nan, "hit_rate": np.nan, "max_drawdown_usd": np.nan, "turnover": 0, "avg_holding_minutes": np.nan}
    daily = x.groupby("session_date").net_pnl_usd.sum().sort_index()
    full = pd.Series(0.0, index=pd.date_range(daily.index.min(), daily.index.max(), freq="B"))
    full.loc[pd.to_datetime(daily.index)] = daily.values
    sd = full.std(ddof=1)
    downside = full[full < 0].std(ddof=1)
    curve = full.cumsum()
    dd = curve - curve.cummax()
    losses = -pnl[pnl < 0].sum()
    return {"trades": len(x), "pnl_usd": pnl.sum(), "gross_pnl_usd": gross.sum(), "expectancy_usd": pnl.mean(), "sharpe": full.mean() / sd * math.sqrt(252) if sd > 0 else np.nan, "sortino": full.mean() / downside * math.sqrt(252) if downside > 0 else np.nan, "profit_factor": pnl[pnl > 0].sum() / losses if losses > 0 else np.inf, "hit_rate": (pnl > 0).mean(), "max_drawdown_usd": dd.min(), "turnover": 2 * len(x), "avg_holding_minutes": x.holding_minutes.mean()}


def aggregate_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    keys = ["mechanism", "instrument", "side", "lookback_minutes", "threshold", "confirmation_minutes", "stress"]
    rows = []
    for key, group in trades.groupby(keys, dropna=False):
        for split in ("all", "is", "validation", "oos"):
            part = group if split == "all" else group[group.split == split]
            rows.append(dict(zip(keys, key)) | {"period": split} | metric_row(part))
    return pd.DataFrame(rows)


def cluster_ci(frame: pd.DataFrame, value: str, group_cols: list[str], samples: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for key, g in frame.groupby(group_cols, dropna=False):
        daily = g.groupby("session_date")[value].mean().dropna().to_numpy(float)
        if not len(daily):
            continue
        means = np.array([rng.choice(daily, len(daily), replace=True).mean() for _ in range(samples)])
        t = stats.ttest_1samp(daily, 0.0) if len(daily) > 1 else None
        rows.append(dict(zip(group_cols, key if isinstance(key, tuple) else (key,))) | {"n_sessions": len(daily), "mean": daily.mean(), "ci_low": np.quantile(means, .025), "ci_high": np.quantile(means, .975), "p_value": float(t.pvalue) if t else np.nan})
    out = pd.DataFrame(rows)
    if not out.empty:
        fam = [c for c in ("mechanism", "instrument", "side") if c in out]
        out["p_bonferroni"] = out.groupby(fam).p_value.transform(lambda s: np.minimum(1.0, s * len(s))) if fam else np.minimum(1.0, out.p_value * len(out))
    return out


def event_summaries(shocks: pd.DataFrame, overnight: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shock_long = shocks.melt(id_vars=["instrument", "session_date", "split", "side", "lookback_minutes", "threshold_atr", "entry_type", "intensity_atr", "mfe_points", "mae_points", "recovered_midpoint", "recovered_origin", "overlap_count"], value_vars=[f"fwd_{h}m_bps" for h in (5, 15, 30, 60, 120)], var_name="horizon", value_name="signed_return_bps")
    shock_summary = cluster_ci(shock_long, "signed_return_bps", ["instrument", "side", "lookback_minutes", "threshold_atr", "entry_type", "horizon"])
    on_long = overnight.melt(id_vars=["instrument", "session_date", "split", "overnight_move_atr", "overnight_bin", "vol_regime", "lagged_sign"], value_vars=["return_open_15_bps", "return_15_30_bps", "return_30_noon_bps", "return_open_close_bps"], var_name="horizon", value_name="raw_return_bps")
    on_long["side"] = np.where(on_long.overnight_move_atr >= 0, "up", "down")
    on_long["continuation_bps"] = np.sign(on_long.overnight_move_atr) * on_long.raw_return_bps
    on_long["opposite_placebo_bps"] = -on_long.continuation_bps
    on_long["lagged_placebo_bps"] = on_long.lagged_sign * on_long.raw_return_bps
    on_summary = cluster_ci(on_long, "continuation_bps", ["instrument", "side", "overnight_bin", "vol_regime", "horizon"])
    return shock_summary, on_summary


def yearly_and_rolling(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nominal = trades[trades.stress == "nominal"].copy()
    nominal["year"] = pd.to_datetime(nominal.session_date).dt.year
    yearly = nominal.groupby(["mechanism", "instrument", "side", "lookback_minutes", "threshold", "confirmation_minutes", "year"]).apply(lambda x: pd.Series(metric_row(x)), include_groups=False).reset_index()
    rows = []
    for (m, i, s), g in nominal.groupby(["mechanism", "instrument", "side"]):
        d = g.groupby("session_date").net_pnl_usd.sum().sort_index()
        roll_mean = d.rolling(63, min_periods=20).mean()
        roll_sd = d.rolling(63, min_periods=20).std()
        for day in d.index:
            rows.append({"mechanism": m, "instrument": i, "side": s, "session_date": day, "rolling_expectancy_63d": roll_mean.loc[day], "rolling_sharpe_63d": roll_mean.loc[day] / roll_sd.loc[day] * math.sqrt(252) if roll_sd.loc[day] > 0 else np.nan})
    return yearly, pd.DataFrame(rows)


def diagnostic_outputs(shocks: pd.DataFrame, overnight: pd.DataFrame, trades: pd.DataFrame) -> None:
    shock = shocks.copy()
    shock["year"] = pd.to_datetime(shock.session_date).dt.year
    shock["hour"] = pd.to_datetime(shock.shock_ts).dt.hour
    shock["intensity_bin"] = pd.cut(shock.intensity_atr, [0, 1.5, 2.0, 2.5, np.inf], labels=["<1.5", "1.5:2.0", "2.0:2.5", ">=2.5"])
    shock.groupby(["instrument", "side", "entry_type", "year", "hour", "intensity_bin"], observed=True).agg(events=("session_date", "size"), mean_30m_bps=("fwd_30m_bps", "mean"), mean_mfe_points=("mfe_points", "mean"), mean_mae_points=("mae_points", "mean"), midpoint_recovery=("recovered_midpoint", "mean"), origin_recovery=("recovered_origin", "mean"), mean_overlap=("overlap_count", "mean")).reset_index().to_csv(OUT / "shock_diagnostics_by_year_hour_intensity.csv", index=False)
    on = overnight.copy()
    on["year"] = pd.to_datetime(on.session_date).dt.year
    long = on.melt(id_vars=["instrument", "session_date", "year", "overnight_move_atr", "overnight_bin", "vol_regime", "lagged_sign"], value_vars=["return_open_15_bps", "return_15_30_bps", "return_30_noon_bps", "return_open_close_bps"], var_name="horizon", value_name="raw_return_bps")
    long["side"] = np.where(long.overnight_move_atr >= 0, "up", "down")
    long["continuation_bps"] = np.sign(long.overnight_move_atr) * long.raw_return_bps
    long["opposite_placebo_bps"] = -long.continuation_bps
    long["lagged_placebo_bps"] = long.lagged_sign * long.raw_return_bps
    long.groupby(["instrument", "side", "year", "vol_regime", "horizon"], observed=True).agg(events=("session_date", "size"), continuation_bps=("continuation_bps", "mean"), opposite_placebo_bps=("opposite_placebo_bps", "mean"), lagged_placebo_bps=("lagged_placebo_bps", "mean")).reset_index().to_csv(OUT / "overnight_placebo_year_regime.csv", index=False)
    monotone = []
    for (instrument, horizon), g in long.groupby(["instrument", "horizon"]):
        means = g.groupby("overnight_bin", observed=True).raw_return_bps.mean()
        rho, p = stats.spearmanr(np.arange(len(means)), means.to_numpy(float)) if len(means) > 2 else (np.nan, np.nan)
        monotone.append({"instrument": instrument, "horizon": horizon, "spearman_rho": rho, "p_value": p, "ordered_bins": len(means)})
    pd.DataFrame(monotone).to_csv(OUT / "overnight_monotonicity.csv", index=False)
    nominal = trades[trades.stress == "nominal"].copy()
    concentration = []
    extreme = []
    keys = ["mechanism", "instrument", "side", "lookback_minutes", "threshold", "confirmation_minutes"]
    for key, g in nominal.groupby(keys):
        pnl = g.net_pnl_usd
        positive = pnl.clip(lower=0).sum()
        daily = g.groupby("session_date").net_pnl_usd.sum()
        years = g.assign(year=pd.to_datetime(g.session_date).dt.year).groupby("year").net_pnl_usd.sum()
        concentration.append(dict(zip(keys, key)) | {"top_trade_positive_pnl_share": pnl.clip(lower=0).max() / positive if positive > 0 else np.nan, "top_session_positive_pnl_share": daily.clip(lower=0).max() / daily.clip(lower=0).sum() if daily.clip(lower=0).sum() > 0 else np.nan, "top_year_positive_pnl_share": years.clip(lower=0).max() / years.clip(lower=0).sum() if years.clip(lower=0).sum() > 0 else np.nan})
        for label, drop_year in (("without_best_year", years.idxmax()), ("without_worst_year", years.idxmin())):
            part = g[pd.to_datetime(g.session_date).dt.year != drop_year]
            extreme.append(dict(zip(keys, key)) | {"scenario": label, "excluded_year": drop_year} | metric_row(part))
    pd.DataFrame(concentration).to_csv(OUT / "concentration_diagnostics.csv", index=False)
    pd.DataFrame(extreme).to_csv(OUT / "extreme_year_sensitivity.csv", index=False)


def plot_outputs(metrics: pd.DataFrame, overnight: pd.DataFrame, out: Path) -> None:
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    nom = metrics[(metrics.stress == "nominal") & (metrics.period == "all")]
    heat_metrics = {"trades": ".0f", "pnl_usd": ".0f", "sharpe": ".2f", "profit_factor": ".2f", "max_drawdown_usd": ".0f"}
    for (mechanism, instrument, side), g in nom.groupby(["mechanism", "instrument", "side"]):
        for value, fmt in heat_metrics.items():
            if mechanism == "conditional_shock_recovery":
                piv = g.pivot_table(index="lookback_minutes", columns="threshold", values=value)
            else:
                piv = g.pivot_table(index="confirmation_minutes", columns="threshold", values=value)
            if piv.empty:
                continue
            center = 0 if value in {"pnl_usd", "sharpe", "max_drawdown_usd"} else None
            fig, ax = plt.subplots(figsize=(6, 4)); sns.heatmap(piv, annot=True, fmt=fmt, cmap="RdYlGn", center=center, ax=ax); ax.set_title(f"{mechanism} {instrument} {side} — {value}"); fig.tight_layout(); fig.savefig(figdir / f"heatmap_{value}_{mechanism}_{instrument}_{side}.png", dpi=140); plt.close(fig)
    for instrument, g in overnight.groupby("instrument"):
        p = g.groupby("overnight_bin", observed=True).return_open_close_bps.agg(["mean", "count"])
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(p.index.astype(str), p["mean"], marker="o"); ax.axhline(0, color="black", lw=.8); ax.set(title=f"{instrument}: overnight bin vs open-close return", ylabel="bps"); fig.tight_layout(); fig.savefig(figdir / f"conditional_overnight_{instrument}.png", dpi=140); plt.close(fig)


def plot_shock_paths(shocks: pd.DataFrame) -> None:
    figdir = OUT / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    for (instrument, side), g in shocks.groupby(["instrument", "side"]):
        rows = []
        for entry_type, part in g.groupby("entry_type"):
            for h in (5, 15, 30, 60, 120):
                rows.append({"entry_type": entry_type, "horizon": h, "mean": part[f"fwd_{h}m_bps"].mean()})
        p = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(7, 4))
        for entry_type, part in p.groupby("entry_type"):
            ax.plot(part.horizon, part["mean"], marker="o", label=entry_type)
        ax.axhline(0, color="black", lw=.8); ax.set(title=f"{instrument} {side}: shock forward path", xlabel="minutes", ylabel="signed bps"); ax.legend(); fig.tight_layout(); fig.savefig(figdir / f"conditional_shock_{instrument}_{side}.png", dpi=140); plt.close(fig)


def portfolio_diagnostics(trades: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["mechanism", "instrument", "side", "daily_corr_max_abs", "baseline_negative_day_pnl", "equal_weight_sharpe_delta", "equal_weight_maxdd_delta", "equal_weight_es95_delta"])
    base_dir = ROOT / "research/portfolio_integration_phase1/sleeve_trades"
    base = {}
    for p in base_dir.glob("*.csv"):
        x = pd.read_csv(p, parse_dates=["session_date"])
        base[p.stem] = x.groupby("session_date").net_pnl_usd.sum()
    base_df = pd.DataFrame(base).fillna(0)
    rows = []
    for _, c in candidates.iterrows():
        mask = (trades.mechanism == c.mechanism) & (trades.instrument == c.instrument) & (trades.side == c.side) & (trades.lookback_minutes == c.lookback_minutes) & (trades.threshold == c.threshold) & (trades.confirmation_minutes == c.confirmation_minutes) & (trades.stress == "nominal")
        new = trades[mask].groupby("session_date").net_pnl_usd.sum().rename("candidate")
        common = base_df.join(new, how="left").fillna(0)
        corr = common.corr().candidate.drop("candidate").abs().max()
        old = base_df.mean(axis=1); combined = common.mean(axis=1)
        def pm(s: pd.Series) -> tuple[float, float, float]:
            sh = s.mean() / s.std() * math.sqrt(252) if s.std() > 0 else np.nan
            dd = (s.cumsum() - s.cumsum().cummax()).min()
            es = s[s <= s.quantile(.05)].mean()
            return sh, dd, es
        a, b = pm(old), pm(combined)
        rows.append(c.to_dict() | {"daily_corr_max_abs": corr, "baseline_negative_day_pnl": new.reindex(old.index, fill_value=0)[old < 0].sum(), "equal_weight_sharpe_delta": b[0] - a[0], "equal_weight_maxdd_delta": b[1] - a[1], "equal_weight_es95_delta": b[2] - a[2]})
    return pd.DataFrame(rows)


def decide(metrics: pd.DataFrame, yearly: pd.DataFrame, event_ci: pd.DataFrame) -> pd.DataFrame:
    nom = metrics[(metrics.stress == "nominal")]
    configs = nom[["mechanism", "instrument", "side", "lookback_minutes", "threshold", "confirmation_minutes"]].drop_duplicates()
    rows = []
    for _, c in configs.iterrows():
        key = np.ones(len(metrics), dtype=bool)
        for col in c.index:
            key &= metrics[col].eq(c[col]).to_numpy()
        sub = metrics[key]
        periods = sub[(sub.stress == "nominal")].set_index("period")
        full_n = periods.trades.get("all", 0); oos_n = periods.trades.get("oos", 0)
        expects = [periods.expectancy_usd.get(p, np.nan) for p in ("is", "validation", "oos")]
        stresses = sub[sub.period == "all"].set_index("stress").expectancy_usd
        cost_ok = all(stresses.get(s, np.nan) > 0 for s in ("plus1tick", "double_commission"))
        delay_ok = stresses.get("delay_one_bar", np.nan) > 0
        y = yearly[(yearly.mechanism == c.mechanism) & (yearly.instrument == c.instrument) & (yearly.side == c.side) & (yearly.lookback_minutes == c.lookback_minutes) & (yearly.threshold == c.threshold) & (yearly.confirmation_minutes == c.confirmation_minutes)]
        concentration = y.pnl_usd.clip(lower=0).max() / y.pnl_usd.clip(lower=0).sum() if y.pnl_usd.clip(lower=0).sum() > 0 else 1.0
        if full_n < 80 or oos_n < 25:
            verdict = "WATCHLIST_LOW_SAMPLE" if all(np.nan_to_num(expects, nan=-1) > 0) else "REJECTED_NO_EDGE"
        elif not all(np.nan_to_num(expects, nan=-1) > 0):
            verdict = "REJECTED_NO_EDGE"
        elif not cost_ok:
            verdict = "REJECTED_COST_SENSITIVE"
        elif not delay_ok or concentration > .5:
            verdict = "REJECTED_UNSTABLE"
        else:
            verdict = "LOCKED_SLEEVE_CANDIDATE"
        rows.append(c.to_dict() | {"trades_full": full_n, "trades_oos": oos_n, "is_expectancy": expects[0], "validation_expectancy": expects[1], "oos_expectancy": expects[2], "cost_ok": cost_ok, "delay_ok": delay_ok, "max_positive_year_share": concentration, "verdict": verdict})
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "configuration_decisions.csv", index=False)
    if detail.empty:
        return detail
    # Campaign verdict is deliberately conservative: every side is a separate test; choose a plateau representative only.
    out = []
    for (m, i), g in detail.groupby(["mechanism", "instrument"]):
        rank = {"LOCKED_SLEEVE_CANDIDATE": 4, "WATCHLIST_LOW_SAMPLE": 3, "REJECTED_COST_SENSITIVE": 2, "REJECTED_UNSTABLE": 1, "REJECTED_NO_EDGE": 0}
        ranked = g.assign(_rank=g.verdict.map(rank), _full_positive=np.where(g.is_expectancy.notna(), g.is_expectancy, -np.inf))
        best = ranked.sort_values(["_rank", "trades_full", "_full_positive", "threshold"], ascending=[False, False, False, True]).iloc[0]
        campaign_verdict = best.verdict
        if campaign_verdict == "REJECTED_NO_EDGE":
            unstable = g[(g.trades_full >= 80) & (g.trades_oos >= 25) & ((g.is_expectancy > 0) | (g.validation_expectancy > 0) | (g.oos_expectancy > 0))]
            if len(unstable) and (unstable[["is_expectancy", "validation_expectancy", "oos_expectancy"]].min(axis=1) <= 0).any():
                best = unstable.sort_values(["trades_full", "threshold"], ascending=[False, True]).iloc[0]
                campaign_verdict = "REJECTED_UNSTABLE"
        record = best.drop(labels=[x for x in ("_rank", "_full_positive") if x in best.index]).to_dict()
        record["verdict"] = campaign_verdict
        out.append(record)
    return pd.DataFrame(out)


def write_report(verdicts: pd.DataFrame, metrics: pd.DataFrame, portfolio: pd.DataFrame, audits: list[dict[str, Any]], topstep_run: bool) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_(empty)_"
        clean = frame.copy()
        for col in clean.columns:
            clean[col] = clean[col].map(lambda x: f"{x:.4g}" if isinstance(x, (float, np.floating)) else str(x))
        header = "| " + " | ".join(map(str, clean.columns)) + " |"
        rule = "|" + "|".join(["---"] * len(clean.columns)) + "|"
        body = ["| " + " | ".join(row.astype(str)) + " |" for _, row in clean.iterrows()]
        return "\n".join([header, rule] + body)

    def verdict(mech: str, inst: str) -> str:
        x = verdicts[(verdicts.mechanism == mech) & (verdicts.instrument == inst)]
        return x.verdict.iloc[0] if len(x) else "REJECTED_NO_EDGE"
    lines = ["# Final report — conditional shock recovery and overnight/RTH transfer", "", "## Résumé exécutif", "",
             f"1. `conditional_shock_recovery`: MNQ **{verdict('conditional_shock_recovery','MNQ')}**; MGC **{verdict('conditional_shock_recovery','MGC')}**.",
             f"2. `overnight_acceptance_continuation`: MNQ **{verdict('overnight_acceptance_continuation','MNQ')}**; MGC **{verdict('overnight_acceptance_continuation','MGC')}**.",
             f"3. `overnight_rejection_fade`: MNQ **{verdict('overnight_rejection_fade','MNQ')}**; MGC **{verdict('overnight_rejection_fade','MGC')}**.",
             f"4. Zone robuste: **{'oui' if (verdicts.verdict == 'LOCKED_SLEEVE_CANDIDATE').any() else 'non'}** selon les gates préspécifiés.",
             f"5. Robustesse aux coûts: **{int(verdicts.cost_ok.sum())}/{len(verdicts)}** représentants résistent à +1 tick et commission doublée.",
             f"6. Valeur marginale face aux six sleeves: **{'évaluée pour les survivants' if len(portfolio) else 'non applicable, aucun survivant standalone'}**.",
             f"7. Prochaine action: **{'validation prospective de la/les candidate(s), sans retuning' if (verdicts.verdict == 'LOCKED_SLEEVE_CANDIDATE').any() else 'clore ces familles; ne pas chercher une variante de sauvetage'}**.", "",
             "## Protocole et audit", "", "La spécification a été écrite et hashée avant le calcul. Les deux fichiers locaux ont été utilisés sans téléchargement. Les timestamps sont convertis en America/New_York; les conditions sur barres complètes sont exécutées à l’open minute suivant. Les changements d’`instrument_id` invalident les événements overnight concernés.", "", "| Instrument | Rows | First | Last | SHA-256 |", "|---|---:|---|---|---|"]
    for a in audits:
        lines.append(f"| {a['instrument']} | {a['rows']} | {a['first']} | {a['last']} | `{a['sha256']}` |")
    lines += ["", "## Décisions représentatives", "", markdown_table(verdicts), "", "## Interprétation", "", "Les hypothèses long et short ont été testées séparément. Le représentant affiché n’est pas le maximum de Sharpe: il est choisi par la hiérarchie gelée des gates puis par le seuil le plus bas en cas d’égalité. Les fichiers `strategy_metrics.csv`, `yearly_results.csv`, `rolling_metrics.csv`, `shock_event_summary.csv` et `overnight_event_summary.csv` contiennent le détail complet.", "", "Les stress tests incluent +1/+2/+3 ticks par côté, commission doublée et entrée retardée d’une minute. Les heatmaps sont dans `figures/`. Les p-values d’étude événementielle sont corrigées par Bonferroni au niveau instrument/côté.", "", "## Portefeuille et Topstep", ""]
    lines.append(markdown_table(portfolio) if len(portfolio) else "Aucun mécanisme n’a franchi les gates standalone; conformément au protocole, aucune simulation Topstep et aucune allocation marginale n’ont été lancées.")
    lines += ["", f"Simulation Topstep complète exécutée: **{topstep_run}**.", "", "## Limites", "", "Les sessions raccourcies sont reconnues par leur dernière barre observée, car le dépôt ne fournit pas un calendrier officiel unique couvrant simultanément CME Equity et COMEX sur toute la période. Elles sont signalées dans l’audit et ne servent jamais à inventer une barre. Les résultats OOS sont historiques et le dépôt documente déjà que la période 2024–2026 a été réutilisée pour certaines sleeves MGC; aucune affirmation de pristine OOS portefeuille n’est faite."]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(bootstrap_samples: int = 1000) -> None:
    if not SPEC_PATH.exists():
        raise FileNotFoundError("Frozen specification must exist before calculations")
    spec_hash_before = sha256(SPEC_PATH)
    OUT.mkdir(parents=True, exist_ok=True)
    all_shocks, all_on, all_trades, audits = [], [], [], []
    for symbol, inst in INSTRUMENTS.items():
        bars = load_bars(inst)
        sessions, daily, five = session_frames(bars, inst)
        weekday_dates = {d for d in bars.index.date if pd.Timestamp(d).weekday() < 5}
        audits.append({"instrument": symbol, "path": str(inst.path.relative_to(ROOT)), "sha256": sha256(inst.path), "rows": len(bars), "first": bars.index.min(), "last": bars.index.max(), "sessions": len(daily), "shortened_sessions": int(daily.shortened.sum()), "weekday_dates_without_usable_rth_session": int(len(weekday_dates) - len(daily))})
        shocks = shock_events(symbol, sessions, daily, five)
        on = overnight_events(symbol, bars, sessions, daily)
        all_shocks.append(shocks); all_on.append(on)
        all_trades.extend([shock_trades(shocks, sessions, inst), overnight_trades(on, sessions, inst)])
    shocks = pd.concat(all_shocks, ignore_index=True); overnight = pd.concat(all_on, ignore_index=True); trades = pd.concat(all_trades, ignore_index=True)
    shocks.to_csv(OUT / "shock_events.csv", index=False); overnight.to_csv(OUT / "overnight_events.csv", index=False); trades.to_csv(OUT / "trades.csv", index=False)
    shock_summary, on_summary = event_summaries(shocks, overnight)
    shock_summary.to_csv(OUT / "shock_event_summary.csv", index=False); on_summary.to_csv(OUT / "overnight_event_summary.csv", index=False)
    metrics = aggregate_metrics(trades); metrics.to_csv(OUT / "strategy_metrics.csv", index=False)
    yearly, rolling = yearly_and_rolling(trades); yearly.to_csv(OUT / "yearly_results.csv", index=False); rolling.to_csv(OUT / "rolling_metrics.csv", index=False)
    diagnostic_outputs(shocks, overnight, trades)
    verdicts = decide(metrics, yearly, shock_summary); verdicts.to_csv(OUT / "decision_registry.csv", index=False)
    candidates = verdicts[verdicts.verdict == "LOCKED_SLEEVE_CANDIDATE"]
    portfolio = portfolio_diagnostics(trades, candidates); portfolio.to_csv(OUT / "portfolio_marginal_value.csv", index=False)
    plot_outputs(metrics, overnight, OUT); plot_shock_paths(shocks)
    canddir = OUT / "candidate_specs"; canddir.mkdir(exist_ok=True)
    for _, row in candidates.iterrows():
        write_json(canddir / f"{row.instrument}_{row.mechanism}_{row.side}.json", row.to_dict())
    write_json(OUT / "data_audit.json", {a["instrument"]: a for a in audits})
    script_path = Path(__file__)
    inputs = [SPEC_PATH, script_path] + [x.path for x in INSTRUMENTS.values()] + list((ROOT / "research/portfolio_integration_phase1/sleeve_trades").glob("*.csv"))
    manifest = {"specification_sha256_before_and_after": [spec_hash_before, sha256(SPEC_PATH)], "inputs_and_scripts": {str(p.relative_to(ROOT)): sha256(p) for p in inputs}, "outputs": {}, "bootstrap_samples": bootstrap_samples, "seed": SEED, "topstep_run": False}
    write_report(verdicts, metrics, portfolio, audits, False)
    for p in OUT.rglob("*"):
        if p.is_file() and p.name != "manifest.json":
            manifest["outputs"][str(p.relative_to(OUT))] = sha256(p)
    write_json(OUT / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    run(args.bootstrap_samples)


if __name__ == "__main__":
    main()
