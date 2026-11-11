"""Strictly bounded TSI intraday mean-reversion research campaign.

The module contains no optimiser.  Thresholds and neighbouring TSI parameters
are fixed diagnostics, and the primary gate is evaluated before external or
Topstep work is allowed to run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import time
from typing import Iterable

import numpy as np
import pandas as pd

from src.config.settings import get_instrument_spec


THRESHOLDS = (15.0, 25.0, 35.0)
HORIZONS = (3, 6, 12, 24)  # 15/30/60/120 minutes on five-minute bars
SUBPERIODS = (("2019-2020", 2019, 2020), ("2021-2022", 2021, 2022), ("2023-2026", 2023, 2026))
TSI_NEIGHBOURHOOD = (
    (20, 10, 5), (20, 13, 7), (25, 10, 7), (25, 13, 5), (25, 13, 7),
    (25, 13, 9), (25, 16, 7), (30, 13, 7), (30, 16, 9),
)


@dataclass(frozen=True)
class CampaignSpec:
    long_span: int = 25
    short_span: int = 13
    signal_span: int = 7
    primary_threshold: float = 25.0
    atr_period: int = 14
    stop_atr: float = 1.5
    max_holding_bars: int = 12
    bootstrap_samples: int = 5000
    bootstrap_block_days: int = 10
    random_seed: int = 20261002
    positive_year_ratio: float = 0.60
    minimum_events_per_threshold: int = 30
    timezone: str = "America/New_York"
    rth_open: time = time(9, 30)
    rth_close: time = time(16, 0)


def prepare_rth_bars(bars: pd.DataFrame, spec: CampaignSpec = CampaignSpec()) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    if not required.issubset(bars.columns):
        raise ValueError(f"missing OHLC columns: {sorted(required - set(bars.columns))}")
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("bars require a timezone-aware DatetimeIndex")
    out = bars.sort_index().tz_convert(spec.timezone).copy()
    clock = out.index.time
    out = out.loc[(clock >= spec.rth_open) & (clock < spec.rth_close)].copy()
    out["session_date"] = out.index.date
    out["bar_in_session"] = out.groupby("session_date", sort=False).cumcount()
    return out


def add_indicators(
    bars: pd.DataFrame,
    *,
    long_span: int = 25,
    short_span: int = 13,
    signal_span: int = 7,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Canonical TSI: 100 * EMA(EMA(delta, 25), 13) / abs analogue."""
    out = bars.copy()
    delta = out["close"].astype(float).diff()
    num = delta.ewm(span=long_span, adjust=False, min_periods=long_span).mean()
    num = num.ewm(span=short_span, adjust=False, min_periods=short_span).mean()
    den = delta.abs().ewm(span=long_span, adjust=False, min_periods=long_span).mean()
    den = den.ewm(span=short_span, adjust=False, min_periods=short_span).mean()
    out["tsi"] = 100.0 * num / den.replace(0.0, np.nan)
    out["tsi_signal"] = out["tsi"].ewm(span=signal_span, adjust=False, min_periods=signal_span).mean()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [(out["high"] - out["low"]).abs(), (out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    out["atr14"] = tr.ewm(alpha=1.0 / atr_period, adjust=False, min_periods=atr_period).mean()
    return out


def detect_events(bars: pd.DataFrame, *, threshold: float, side: int) -> pd.DataFrame:
    """Arm after an extreme and fire once on a same-session signal crossover."""
    rows: list[dict[str, object]] = []
    for session_date, day in bars.groupby("session_date", sort=True):
        armed = False
        tsi = day["tsi"].to_numpy(float)
        signal = day["tsi_signal"].to_numpy(float)
        for pos in range(1, len(day) - 1):
            if not np.isfinite(tsi[pos]) or not np.isfinite(signal[pos]):
                continue
            if (side == 1 and tsi[pos] < -threshold) or (side == -1 and tsi[pos] > threshold):
                armed = True
            crossed = (
                side == 1 and tsi[pos - 1] <= signal[pos - 1] and tsi[pos] > signal[pos]
            ) or (
                side == -1 and tsi[pos - 1] >= signal[pos - 1] and tsi[pos] < signal[pos]
            )
            if armed and crossed:
                signal_idx, entry_idx = day.index[pos], day.index[pos + 1]
                rows.append({
                    "session_date": session_date, "signal_time": signal_idx, "entry_time": entry_idx,
                    "signal_position": int(bars.index.get_loc(signal_idx)),
                    "entry_position": int(bars.index.get_loc(entry_idx)), "side": side,
                    "direction": "long" if side == 1 else "short", "threshold": threshold,
                    "tsi": tsi[pos], "tsi_signal": signal[pos], "atr14": float(day["atr14"].iloc[pos]),
                })
                armed = False
    return pd.DataFrame(rows)


def build_event_study(bars: pd.DataFrame, events: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    instrument = get_instrument_spec(symbol)
    cost = 2.0 * (float(instrument["commission_per_side_usd"]) + float(instrument["tick_value_usd"]) * float(instrument["slippage_ticks"]))
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        entry_pos = int(event.entry_position)
        entry = float(bars.iloc[entry_pos]["open"])
        same_day = bars.iloc[entry_pos:]
        same_day = same_day.loc[same_day["session_date"].eq(event.session_date)]
        for horizon in HORIZONS:
            if len(same_day) < horizon:
                continue
            path = same_day.iloc[:horizon]
            raw_exit = float(path["close"].iloc[-1])
            gross_points = int(event.side) * (raw_exit - entry)
            mfe = max(0.0, float(path["high"].max()) - entry) if event.side == 1 else max(0.0, entry - float(path["low"].min()))
            mae = max(0.0, entry - float(path["low"].min())) if event.side == 1 else max(0.0, float(path["high"].max()) - entry)
            gross_usd = gross_points * float(instrument["point_value_usd"])
            rows.append({
                **event._asdict(), "symbol": symbol, "horizon_minutes": horizon * 5,
                "raw_entry_price": entry, "raw_exit_price": raw_exit, "gross_points": gross_points,
                "gross_return_bps": 10000.0 * gross_points / entry, "mfe_points": mfe, "mae_points": mae,
                "gross_pnl_usd": gross_usd, "net_pnl_usd": gross_usd - cost,
                "assumed_round_trip_cost_usd": cost,
            })
    return pd.DataFrame(rows)


def summarize_event_study(study: pd.DataFrame) -> pd.DataFrame:
    if study.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in study.groupby(["symbol", "direction", "threshold", "horizon_minutes"], observed=True):
        rows.append({
            "symbol": keys[0], "direction": keys[1], "threshold": keys[2], "horizon_minutes": keys[3],
            "n": len(group), "mean_gross_points": group["gross_points"].mean(),
            "mean_gross_return_bps": group["gross_return_bps"].mean(),
            "mean_net_pnl_usd": group["net_pnl_usd"].mean(), "median_mfe_points": group["mfe_points"].median(),
            "median_mae_points": group["mae_points"].median(),
            "break_even_round_trip_cost_usd": group["gross_pnl_usd"].mean(),
            "break_even_total_slippage_ticks_after_commission": max(0.0, (group["gross_pnl_usd"].mean() - 2.5) / (2.0 * get_instrument_spec(keys[0])["tick_value_usd"])),
        })
    return pd.DataFrame(rows)


def simulate_trades(bars: pd.DataFrame, events: pd.DataFrame, *, symbol: str, max_holding_bars: int = 12, stop_atr: float = 1.5) -> pd.DataFrame:
    inst = get_instrument_spec(symbol)
    slip = float(inst["tick_size"]) * float(inst["slippage_ticks"])
    commission = 2.0 * float(inst["commission_per_side_usd"])
    rows: list[dict[str, object]] = []
    unavailable_until = -1
    for event in events.sort_values("entry_position").itertuples(index=False):
        entry_pos = int(event.entry_position)
        if entry_pos <= unavailable_until or not np.isfinite(event.atr14):
            continue
        entry_bar = bars.iloc[entry_pos]
        raw_entry = float(entry_bar["open"])
        side = int(event.side)
        stop = raw_entry - side * stop_atr * float(event.atr14)
        same_day = bars.iloc[entry_pos:]
        same_day = same_day.loc[same_day["session_date"].eq(event.session_date)].iloc[:max_holding_bars]
        if same_day.empty:
            continue
        raw_exit, exit_reason, exit_idx = float(same_day["close"].iloc[-1]), "session_close", same_day.index[-1]
        for held, (idx, bar) in enumerate(same_day.iterrows(), start=1):
            stop_hit = float(bar["low"]) <= stop if side == 1 else float(bar["high"]) >= stop
            if stop_hit:
                raw_exit = min(float(bar["open"]), stop) if side == 1 else max(float(bar["open"]), stop)
                exit_reason, exit_idx = "stop", idx
                break
            zero_hit = float(bar["tsi"]) >= 0.0 if side == 1 else float(bar["tsi"]) <= 0.0
            if zero_hit:
                raw_exit, exit_reason, exit_idx = float(bar["close"]), "tsi_zero", idx
                break
            if held >= max_holding_bars:
                raw_exit, exit_reason, exit_idx = float(bar["close"]), "time_12_bars", idx
                break
        path = bars.loc[event.entry_time:exit_idx]
        executed_entry, executed_exit = raw_entry + side * slip, raw_exit - side * slip
        gross_points = side * (raw_exit - raw_entry)
        net_points = side * (executed_exit - executed_entry)
        rows.append({
            "symbol": symbol, "session_date": event.session_date, "signal_time": event.signal_time,
            "entry_time": event.entry_time, "exit_time": exit_idx, "direction": event.direction, "side": side,
            "threshold": event.threshold, "tsi_long": np.nan, "tsi_short": np.nan, "tsi_signal_span": np.nan,
            "raw_entry_price": raw_entry, "entry_price": executed_entry, "stop_price": stop,
            "raw_exit_price": raw_exit, "exit_price": executed_exit, "exit_reason": exit_reason,
            "holding_bars": int(bars.index.get_loc(exit_idx) - entry_pos + 1),
            "gross_pnl_usd": gross_points * float(inst["point_value_usd"]),
            "net_pnl_usd": net_points * float(inst["point_value_usd"]) - commission,
            "mfe_points": max(0.0, float(path["high"].max()) - raw_entry) if side == 1 else max(0.0, raw_entry - float(path["low"].min())),
            "mae_points": max(0.0, raw_entry - float(path["low"].min())) if side == 1 else max(0.0, float(path["high"].max()) - raw_entry),
        })
        unavailable_until = int(bars.index.get_loc(exit_idx))
    return pd.DataFrame(rows)


def trade_metrics(trades: pd.DataFrame, **labels: object) -> dict[str, object]:
    pnl = trades["net_pnl_usd"].astype(float) if not trades.empty else pd.Series(dtype=float)
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {**labels, "n": len(pnl), "net_expectancy_usd": pnl.mean() if len(pnl) else np.nan,
            "net_pnl_usd": pnl.sum(), "profit_factor": gains / losses if losses > 0 else np.inf,
            "win_rate": (pnl > 0).mean() if len(pnl) else np.nan}


def stability_tables(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = trades.copy()
    frame["year"] = pd.to_datetime(frame["session_date"]).dt.year
    annual = pd.DataFrame([trade_metrics(g, year=int(y)) for y, g in frame.groupby("year")])
    sub = []
    for name, start, end in SUBPERIODS:
        group = frame.loc[frame["year"].between(start, end)]
        sub.append(trade_metrics(group, subperiod=name))
    return annual, pd.DataFrame(sub)


def primary_gate(event_summary: pd.DataFrame, primary_trades: pd.DataFrame, annual: pd.DataFrame, subperiod: pd.DataFrame, spec: CampaignSpec) -> tuple[bool, dict[str, object]]:
    sixty = event_summary.loc[(event_summary.symbol == "MNQ") & (event_summary.direction == "long") & (event_summary.horizon_minutes == 60)].set_index("threshold")
    enough = all(t in sixty.index and int(sixty.loc[t, "n"]) >= spec.minimum_events_per_threshold for t in THRESHOLDS)
    vals = [float(sixty.loc[t, "mean_net_pnl_usd"]) for t in THRESHOLDS] if enough else [np.nan] * 3
    monotone = enough and vals[0] <= vals[1] <= vals[2]
    expectancy = float(primary_trades["net_pnl_usd"].mean()) if len(primary_trades) else np.nan
    positive = bool(np.isfinite(expectancy) and expectancy > 0)
    year_ratio = float((annual["net_expectancy_usd"] > 0).mean()) if len(annual) else 0.0
    stable = year_ratio >= spec.positive_year_ratio and len(subperiod) == len(SUBPERIODS) and bool((subperiod["net_expectancy_usd"] > 0).all())
    details = {"positive_after_costs": positive, "primary_net_expectancy_usd": expectancy,
               "monotone_60m_thresholds": monotone, "threshold_60m_net_expectancies_usd": dict(zip(map(str, THRESHOLDS), vals)),
               "positive_year_ratio": year_ratio, "all_subperiods_positive": bool((subperiod["net_expectancy_usd"] > 0).all()),
               "minimum_event_count_met": enough}
    return positive and monotone and stable, details


def moving_block_bootstrap(trades: pd.DataFrame, spec: CampaignSpec) -> pd.DataFrame:
    daily = trades.groupby("session_date")["net_pnl_usd"].sum().sort_index().to_numpy(float)
    if len(daily) == 0:
        return pd.DataFrame([{"statistic": "mean_daily_pnl_usd", "estimate": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}])
    rng, n, block = np.random.default_rng(spec.random_seed), len(daily), min(spec.bootstrap_block_days, len(daily))
    starts = np.arange(0, n - block + 1)
    estimates = np.empty(spec.bootstrap_samples)
    for i in range(spec.bootstrap_samples):
        sample = np.concatenate([daily[s:s + block] for s in rng.choice(starts, size=int(np.ceil(n / block)), replace=True)])[:n]
        estimates[i] = sample.mean()
    return pd.DataFrame([{"statistic": "mean_daily_pnl_usd", "estimate": daily.mean(),
                          "ci95_low": np.quantile(estimates, 0.025), "ci95_high": np.quantile(estimates, 0.975),
                          "samples": spec.bootstrap_samples, "block_days": block, "seed": spec.random_seed}])


def matched_controls(bars: pd.DataFrame, events: pd.DataFrame, *, symbol: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match each event to a non-event session at the same five-minute slot/month."""
    rng = np.random.default_rng(seed)
    event_entries = set(pd.to_datetime(events["entry_time"]))
    candidates = bars.loc[~bars.index.isin(event_entries)].copy()
    candidates["month"] = pd.to_datetime(candidates["session_date"]).map(lambda x: (x.year, x.month))
    candidates["slot"] = candidates["bar_in_session"]
    rows = []
    for event in events.itertuples(index=False):
        stamp = pd.Timestamp(event.entry_time)
        pool = candidates.loc[(candidates["month"] == (stamp.year, stamp.month)) & (candidates["slot"] == bars.loc[stamp, "bar_in_session"])]
        pool = pool.loc[pool["session_date"] != event.session_date]
        if pool.empty:
            continue
        chosen = pool.iloc[int(rng.integers(0, len(pool)))]
        entry_pos = int(bars.index.get_loc(chosen.name))
        rows.append({**event._asdict(), "session_date": chosen.session_date, "signal_time": bars.index[max(0, entry_pos - 1)],
                     "entry_time": chosen.name, "signal_position": max(0, entry_pos - 1), "entry_position": entry_pos,
                     "atr14": float(bars.iloc[max(0, entry_pos - 1)]["atr14"])})
    controls = pd.DataFrame(rows)
    control_trades = simulate_trades(bars, controls, symbol=symbol) if len(controls) else pd.DataFrame()
    return controls, control_trades


def collision_analysis(tsi_trades: pd.DataFrame, sleeve_ledgers: dict[str, pd.DataFrame], locked_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    tsi = tsi_trades.copy()
    tsi["entry_time"] = pd.to_datetime(tsi["entry_time"], utc=True)
    tsi["exit_time"] = pd.to_datetime(tsi["exit_time"], utc=True)
    for name, ledger in sleeve_ledgers.items():
        other = ledger.copy()
        other["entry_time"] = pd.to_datetime(other["entry_time"], utc=True)
        other["exit_time"] = pd.to_datetime(other["exit_time"], utc=True)
        collision_count = 0
        for trade in tsi.itertuples():
            collision_count += int(((other.entry_time <= trade.exit_time) & (other.exit_time >= trade.entry_time)).any())
        rows.append({"sleeve": name, "tsi_trade_count": len(tsi), "interval_collision_count": collision_count,
                     "interval_collision_rate": collision_count / len(tsi) if len(tsi) else np.nan,
                     "same_day_rate": tsi["session_date"].isin(set(other["session_date"].astype(str))).mean() if len(tsi) else np.nan})
    daily_tsi = tsi_trades.groupby("session_date")["net_pnl_usd"].sum()
    daily_tsi.index = pd.to_datetime(daily_tsi.index)
    daily = locked_daily.copy()
    daily.index = pd.to_datetime(daily.pop("session_date"))
    aligned = daily.join(daily_tsi.rename("tsi_mean_reversion"), how="outer").fillna(0.0)
    corr = aligned[["tsi_mean_reversion", *sleeve_ledgers]].corr().rename_axis("series").reset_index()
    return pd.DataFrame(rows), corr


def spec_dict(spec: CampaignSpec) -> dict[str, object]:
    out = asdict(spec)
    out["rth_open"], out["rth_close"] = spec.rth_open.isoformat(), spec.rth_close.isoformat()
    return out
