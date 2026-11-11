"""Frozen Phase-1 screen of a CME Bitcoin Monday intraday-trend effect."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = {
    "BTC": ROOT / "data/downloaded/BTC_v_0_1m_20171217_20261003.parquet",
    "MBT": ROOT / "data/downloaded/MBT_v_0_1m_20210425_20261003.parquet",
}
TZ = "America/New_York"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MECHANISMS = ["rolling_range_breakout", "atr_trend_flip", "volatility_breakout"]
COMMISSION_RT_USD = 2.82
TICK_POINTS = 5.0
MBT_POINT_VALUE = 0.10
TOPSTEP_FLAT_HOUR = 16
TOPSTEP_FLAT_MINUTE = 8
LAST_ENTRY_HOUR = 16
WINDOWS = {
    "reopen_early_asia": (0, 180),
    "asia": (180, 540),
    "europe": (540, 840),
    "pre_us": (840, 930),
    "us_rth": (930, 1320),
    "end_of_day": (1320, 1380),
}


@dataclass
class Prepared:
    asset: str
    minute: pd.DataFrame
    sessions: pd.DataFrame
    roll_sessions: set


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _day_label(local_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return local_index.normalize() + pd.to_timedelta((local_index.hour >= 18).astype(int), unit="D")


def prepare(asset: str, path: Path) -> Prepared:
    frame = pd.read_parquet(path).sort_index()
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    local = frame.index.tz_convert(TZ)
    trading_day = _day_label(local)
    start = trading_day - pd.Timedelta(hours=6)
    minute_no = ((local - start).total_seconds() // 60).astype(int)
    frame = frame.assign(
        local_time=local,
        trading_day=trading_day.date,
        weekday=trading_day.day_name(),
        minute_no=minute_no,
    )
    frame = frame[frame.weekday.isin(WEEKDAYS) & frame.minute_no.between(0, 1379)].copy()
    counts = frame.groupby("trading_day").instrument_id.nunique()
    roll_sessions = set(counts[counts > 1].index)
    sessions = (
        frame.groupby("trading_day", as_index=False)
        .agg(
            weekday=("weekday", "first"),
            first_timestamp=("local_time", "min"),
            last_timestamp=("local_time", "max"),
            rows=("close", "size"),
            instrument_count=("instrument_id", "nunique"),
        )
    )
    sessions["year"] = pd.to_datetime(sessions.trading_day).dt.year
    sessions["eligible"] = ~sessions.trading_day.isin(roll_sessions)
    return Prepared(asset, frame, sessions, roll_sessions)


def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous = frame.close.shift(1)
    return pd.concat(
        [(frame.high - frame.low).abs(), (frame.high - previous).abs(), (frame.low - previous).abs()], axis=1
    ).max(axis=1)


def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def signal_bars(prepared: Prepared, minutes: int) -> pd.DataFrame:
    raw = prepared.minute[~prepared.minute.trading_day.isin(prepared.roll_sessions)].copy()
    raw["bucket"] = raw.minute_no // minutes
    bars = (
        raw.groupby(["trading_day", "bucket"], as_index=False)
        .agg(
            start_time=("local_time", "first"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            instrument_id=("instrument_id", "first"),
            n_minutes=("close", "size"),
        )
    )
    bars["end_time"] = bars.start_time + pd.Timedelta(minutes=minutes)
    bars["weekday"] = pd.to_datetime(bars.trading_day).dt.day_name()
    return bars.sort_values("end_time").reset_index(drop=True)


def _flatten_timestamp(day) -> pd.Timestamp:
    return pd.Timestamp(day, tz=TZ) + pd.Timedelta(hours=TOPSTEP_FLAT_HOUR, minutes=TOPSTEP_FLAT_MINUTE)


def _fill_row(group: pd.DataFrame, when: pd.Timestamp) -> pd.Series | None:
    rows = group[group.local_time == when]
    return None if rows.empty else rows.iloc[0]


def _last_flat_row(group: pd.DataFrame, day) -> pd.Series | None:
    target = _flatten_timestamp(day)
    exact = _fill_row(group, target)
    if exact is not None:
        return exact
    rows = group[group.local_time < target]
    return None if rows.empty else rows.iloc[-1]


def _trade(asset: str, mechanism: str, day, direction: int, entry_row, exit_row, reason: str) -> dict:
    entry = float(entry_row.open)
    exit_price = float(exit_row.open if exit_row.local_time == _flatten_timestamp(day) else exit_row.open)
    gross_points = direction * (exit_price - entry)
    gross_bps = 10000.0 * gross_points / entry
    gross_usd = gross_points * MBT_POINT_VALUE
    one_tick_rt = 2.0 * TICK_POINTS * MBT_POINT_VALUE
    return {
        "asset": asset,
        "mechanism": mechanism,
        "trading_day": day,
        "weekday": pd.Timestamp(day).day_name(),
        "direction": direction,
        "entry_time": entry_row.local_time,
        "exit_time": exit_row.local_time,
        "entry_price": entry,
        "exit_price": exit_price,
        "entry_instrument_id": int(entry_row.instrument_id),
        "exit_instrument_id": int(exit_row.instrument_id),
        "exit_reason": reason,
        "gross_points": gross_points,
        "gross_bps": gross_bps,
        "gross_usd_mbt_equiv": gross_usd,
        "net_commission_usd": gross_usd - COMMISSION_RT_USD,
        "net_1tick_usd": gross_usd - COMMISSION_RT_USD - one_tick_rt,
        "net_2tick_usd": gross_usd - COMMISSION_RT_USD - 2.0 * one_tick_rt,
        "net_1tick_bps": 10000.0 * (gross_usd - COMMISSION_RT_USD - one_tick_rt) / (entry * MBT_POINT_VALUE),
    }


def run_event_strategy(prepared: Prepared, mechanism: str, events: pd.DataFrame) -> pd.DataFrame:
    trades = []
    by_day = {day: g.sort_values("local_time") for day, g in prepared.minute.groupby("trading_day")}
    for day, day_events in events.groupby("trading_day"):
        if day in prepared.roll_sessions or day not in by_day:
            continue
        group = by_day[day]
        position = None
        for event in day_events.sort_values("event_time").itertuples(index=False):
            fill = _fill_row(group, pd.Timestamp(event.event_time))
            if fill is None or fill.local_time > _flatten_timestamp(day):
                continue
            desired = int(event.direction)
            action = event.action
            if position is not None and (action == "exit" and position["direction"] == desired):
                trades.append(_trade(prepared.asset, mechanism, day, position["direction"], position["row"], fill, "signal_exit"))
                position = None
                continue
            if action == "entry":
                if position is not None and position["direction"] != desired:
                    trades.append(_trade(prepared.asset, mechanism, day, position["direction"], position["row"], fill, "reverse"))
                    position = None
                if position is None and fill.local_time.hour < LAST_ENTRY_HOUR:
                    position = {"direction": desired, "row": fill}
        if position is not None:
            flat = _last_flat_row(group, day)
            if flat is not None and flat.local_time > position["row"].local_time:
                trades.append(_trade(prepared.asset, mechanism, day, position["direction"], position["row"], flat, "topstep_flat"))
    return pd.DataFrame(trades)


def run_donchian(prepared: Prepared) -> pd.DataFrame:
    bars = signal_bars(prepared, 60)
    bars["entry_high"] = bars.high.rolling(20, min_periods=20).max().shift(1)
    bars["entry_low"] = bars.low.rolling(20, min_periods=20).min().shift(1)
    bars["exit_high"] = bars.high.rolling(10, min_periods=10).max().shift(1)
    bars["exit_low"] = bars.low.rolling(10, min_periods=10).min().shift(1)
    rows = []
    for row in bars.itertuples(index=False):
        if pd.notna(row.exit_low) and row.close < row.exit_low:
            rows.append((row.trading_day, row.end_time, "exit", 1))
        if pd.notna(row.exit_high) and row.close > row.exit_high:
            rows.append((row.trading_day, row.end_time, "exit", -1))
        if pd.notna(row.entry_high) and row.close > row.entry_high:
            rows.append((row.trading_day, row.end_time, "entry", 1))
        elif pd.notna(row.entry_low) and row.close < row.entry_low:
            rows.append((row.trading_day, row.end_time, "entry", -1))
    return run_event_strategy(prepared, "rolling_range_breakout", pd.DataFrame(rows, columns=["trading_day", "event_time", "action", "direction"]))


def _supertrend(frame: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> np.ndarray:
    atr = _wilder(_true_range(frame), period)
    midpoint = (frame.high + frame.low) / 2.0
    basic_upper = midpoint + multiplier * atr
    basic_lower = midpoint - multiplier * atr
    upper = np.full(len(frame), np.nan)
    lower = np.full(len(frame), np.nan)
    state = np.zeros(len(frame), dtype=int)
    valid = np.flatnonzero(atr.notna().to_numpy())
    if not len(valid):
        return state
    first = int(valid[0])
    for i in range(first, len(frame)):
        if i == first:
            upper[i], lower[i], state[i] = basic_upper.iloc[i], basic_lower.iloc[i], -1
            continue
        prev_close = frame.close.iloc[i - 1]
        upper[i] = basic_upper.iloc[i] if basic_upper.iloc[i] < upper[i - 1] or prev_close > upper[i - 1] else upper[i - 1]
        lower[i] = basic_lower.iloc[i] if basic_lower.iloc[i] > lower[i - 1] or prev_close < lower[i - 1] else lower[i - 1]
        state[i] = (1 if frame.close.iloc[i] > upper[i] else -1) if state[i - 1] == -1 else (-1 if frame.close.iloc[i] < lower[i] else 1)
    return state


def run_supertrend(prepared: Prepared) -> pd.DataFrame:
    bars = signal_bars(prepared, 15)
    bars["state"] = _supertrend(bars)
    prior = bars.state.shift(1)
    flips = bars[(bars.state != 0) & (prior != 0) & (bars.state != prior)].copy()
    events = pd.DataFrame({
        "trading_day": flips.trading_day,
        "event_time": flips.end_time,
        "action": "entry",
        "direction": flips.state.astype(int),
    })
    return run_event_strategy(prepared, "atr_trend_flip", events)


def run_volatility_breakout(prepared: Prepared) -> pd.DataFrame:
    minute = prepared.minute[~prepared.minute.trading_day.isin(prepared.roll_sessions)].copy()
    daily = minute.groupby("trading_day", as_index=False).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    daily["tr"] = _true_range(daily)
    daily["prior_atr"] = _wilder(daily.tr, 14).shift(1)
    atr_map = daily.set_index("trading_day").prior_atr.to_dict()
    trades = []
    for day, group in minute.groupby("trading_day"):
        atr = atr_map.get(day)
        if pd.isna(atr):
            continue
        group = group.sort_values("local_time")
        active = group[(group.minute_no >= 0) & (group.local_time < pd.Timestamp(day, tz=TZ) + pd.Timedelta(hours=LAST_ENTRY_HOUR))]
        if active.empty:
            continue
        session_open = float(active.iloc[0].open)
        upper, lower = session_open + 0.6 * float(atr), session_open - 0.6 * float(atr)
        entry = None
        direction = 0
        for row in active.itertuples(index=False):
            long_hit, short_hit = row.high >= upper, row.low <= lower
            if long_hit and short_hit:
                break
            if long_hit or short_hit:
                direction = 1 if long_hit else -1
                entry = pd.Series(row._asdict())
                entry["open"] = max(float(row.open), upper) if direction == 1 else min(float(row.open), lower)
                break
        if entry is None:
            continue
        after = group[(group.local_time >= entry.local_time) & (group.local_time <= _flatten_timestamp(day))]
        stop_hits = after[after.low <= session_open] if direction == 1 else after[after.high >= session_open]
        if not stop_hits.empty:
            exit_row = stop_hits.iloc[0].copy()
            exit_row["open"] = min(float(exit_row.open), session_open) if direction == 1 else max(float(exit_row.open), session_open)
            reason = "session_open_stop"
        else:
            exit_row = _last_flat_row(group, day)
            reason = "topstep_flat"
        if exit_row is not None and exit_row.local_time >= entry.local_time:
            trades.append(_trade(prepared.asset, "volatility_breakout", day, direction, entry, exit_row, reason))
    return pd.DataFrame(trades)


def run_asset(prepared: Prepared) -> pd.DataFrame:
    frames = [run_donchian(prepared), run_supertrend(prepared), run_volatility_breakout(prepared)]
    out = pd.concat(frames, ignore_index=True)
    out["trading_day"] = pd.to_datetime(out.trading_day).dt.date
    return out.sort_values(["entry_time", "mechanism"]).reset_index(drop=True)


def _max_drawdown(values: pd.Series) -> float:
    curve = values.fillna(0).cumsum()
    return float((curve - curve.cummax()).min()) if len(curve) else np.nan


def _pf(values: pd.Series) -> float:
    wins = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return float(wins / losses) if losses > 0 else np.inf if wins > 0 else np.nan


def _sharpe(values: pd.Series, periods: float = 52.0) -> float:
    values = values.dropna()
    return float(np.sqrt(periods) * values.mean() / values.std(ddof=1)) if len(values) > 1 and values.std(ddof=1) > 0 else np.nan


def _hac_monday(daily: pd.DataFrame, value_col: str, lag: int = 5) -> tuple[float, float, float]:
    y = daily[value_col].to_numpy(float)
    x = np.column_stack([np.ones(len(daily)), (daily.weekday == "Monday").astype(float).to_numpy()])
    if len(y) < 20:
        return np.nan, np.nan, np.nan
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    meat = np.zeros((2, 2))
    for t in range(len(y)):
        z = x[t] * resid[t]
        meat += np.outer(z, z)
    for k in range(1, min(lag, len(y) - 1) + 1):
        weight = 1.0 - k / (lag + 1.0)
        gamma = np.zeros((2, 2))
        for t in range(k, len(y)):
            gamma += np.outer(x[t] * resid[t], x[t - k] * resid[t - k])
        meat += weight * (gamma + gamma.T)
    inv = np.linalg.inv(x.T @ x)
    cov = inv @ meat @ inv
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    t_stat = float(beta[1] / se) if se > 0 else np.nan
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=max(len(y) - 2, 1))) if np.isfinite(t_stat) else np.nan
    return float(beta[1]), t_stat, p_value


def build_daily(prepared: Prepared, trades: pd.DataFrame) -> pd.DataFrame:
    calendar = prepared.sessions[prepared.sessions.eligible][["trading_day", "weekday"]].copy()
    rows = []
    for mechanism in MECHANISMS:
        values = trades[trades.mechanism == mechanism].groupby("trading_day").agg(
            trades=("gross_bps", "size"), gross_bps=("gross_bps", "sum"), net_bps=("net_1tick_bps", "sum"),
            gross_usd=("gross_usd_mbt_equiv", "sum"), net_usd=("net_1tick_usd", "sum")
        ).reset_index()
        merged = calendar.merge(values, on="trading_day", how="left").fillna({"trades": 0, "gross_bps": 0, "net_bps": 0, "gross_usd": 0, "net_usd": 0})
        merged["mechanism"] = mechanism
        rows.append(merged)
    daily = pd.concat(rows, ignore_index=True)
    wide_gross = daily.pivot(index="trading_day", columns="mechanism", values="gross_bps").sort_index()
    wide_net = daily.pivot(index="trading_day", columns="mechanism", values="net_bps").sort_index()
    vol = wide_gross.rolling(63, min_periods=20).std().shift(1)
    weights = (1.0 / vol.replace(0, np.nan)).div((1.0 / vol.replace(0, np.nan)).sum(axis=1), axis=0).fillna(1.0 / len(MECHANISMS))
    ensemble = pd.DataFrame({
        "trading_day": wide_gross.index,
        "gross_bps": (wide_gross * weights).sum(axis=1),
        "net_bps": (wide_net * weights).sum(axis=1),
    }).reset_index(drop=True)
    ensemble = ensemble.merge(calendar, on="trading_day", how="left")
    counts = daily.groupby("trading_day").trades.sum()
    ensemble["trades"] = ensemble.trading_day.map(counts).fillna(0)
    ensemble["gross_usd"] = np.nan
    ensemble["net_usd"] = np.nan
    ensemble["mechanism"] = "equal_risk_ensemble"
    return pd.concat([daily, ensemble], ignore_index=True)


def weekday_metrics(asset: str, trades: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mechanism in [*MECHANISMS, "equal_risk_ensemble"]:
        d = daily[daily.mechanism == mechanism].copy()
        t = trades[trades.mechanism == mechanism] if mechanism != "equal_risk_ensemble" else pd.DataFrame()
        beta, hac_t, hac_p = _hac_monday(d, "net_bps")
        monday_expect = t[t.weekday == "Monday"].net_1tick_usd.mean() if len(t) else d[d.weekday == "Monday"].net_bps.mean()
        other_expect = t[t.weekday != "Monday"].net_1tick_usd.mean() if len(t) else d[d.weekday != "Monday"].net_bps.mean()
        mon_sharpe = _sharpe(d[d.weekday == "Monday"].net_bps)
        other_sharpe = _sharpe(d[d.weekday != "Monday"].net_bps, 208.0)
        for weekday in WEEKDAYS:
            subd = d[d.weekday == weekday]
            subt = t[t.weekday == weekday] if len(t) else pd.DataFrame()
            net_trade = subt.net_1tick_usd if len(subt) else subd.net_bps
            rows.append({
                "asset": asset, "mechanism": mechanism, "weekday": weekday,
                "trades": int(subd.trades.sum()), "gross_pnl": float(subt.gross_usd_mbt_equiv.sum()) if len(subt) else float(subd.gross_bps.sum()),
                "net_pnl": float(subt.net_1tick_usd.sum()) if len(subt) else float(subd.net_bps.sum()),
                "expectancy_per_trade": float(net_trade.mean()) if len(net_trade) else np.nan,
                "sharpe": _sharpe(subd.net_bps), "profit_factor": _pf(net_trade),
                "hit_rate": float((net_trade > 0).mean()) if len(net_trade) else np.nan,
                "max_drawdown": _max_drawdown(subd.net_bps),
                "pnl_unit": "USD_MBT_equivalent" if mechanism != "equal_risk_ensemble" else "risk_weighted_bps",
                "monday_minus_tue_fri_expectancy": float(monday_expect - other_expect) if weekday == "Monday" else np.nan,
                "monday_minus_tue_fri_sharpe": float(mon_sharpe - other_sharpe) if weekday == "Monday" else np.nan,
                "monday_minus_tue_fri_pnl_per_trade": float(monday_expect - other_expect) if weekday == "Monday" else np.nan,
                "hac_monday_beta_net_bps": beta if weekday == "Monday" else np.nan,
                "hac_t": hac_t if weekday == "Monday" else np.nan,
                "hac_p_value": hac_p if weekday == "Monday" else np.nan,
            })
    return pd.DataFrame(rows)


def stability(asset: str, trades: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "BLOCK_start_to_2020H1": (None, pd.Timestamp("2020-06-30").date()),
        "BLOCK_2020H2_to_2022": (pd.Timestamp("2020-07-01").date(), pd.Timestamp("2022-12-31").date()),
        "BLOCK_2023_onward": (pd.Timestamp("2023-01-01").date(), None),
    }
    years = sorted({pd.Timestamp(x).year for x in daily.trading_day})
    periods.update({f"YEAR_{year}": (pd.Timestamp(f"{year}-01-01").date(), pd.Timestamp(f"{year}-12-31").date()) for year in years})
    rows = []
    for mechanism in [*MECHANISMS, "equal_risk_ensemble"]:
        d0 = daily[daily.mechanism == mechanism]
        t0 = trades[trades.mechanism == mechanism]
        for period, (start, end) in periods.items():
            d = d0.copy()
            t = t0.copy()
            if start is not None:
                d, t = d[d.trading_day >= start], t[t.trading_day >= start]
            if end is not None:
                d, t = d[d.trading_day <= end], t[t.trading_day <= end]
            mon = d[d.weekday == "Monday"]
            other = d[d.weekday != "Monday"]
            mon_t = t[t.weekday == "Monday"]
            other_t = t[t.weekday != "Monday"]
            if mechanism == "equal_risk_ensemble":
                mon_exp, other_exp = mon.net_bps.mean(), other.net_bps.mean()
                ntrades = int(mon.trades.sum())
            else:
                mon_exp, other_exp = mon_t.net_1tick_usd.mean(), other_t.net_1tick_usd.mean()
                ntrades = len(mon_t)
            rows.append({"asset": asset, "mechanism": mechanism, "period": period, "monday_sharpe": _sharpe(mon.net_bps),
                         "monday_expectancy": mon_exp, "tue_fri_expectancy": other_exp,
                         "monday_minus_tue_fri": mon_exp - other_exp, "monday_trades": ntrades})
    return pd.DataFrame(rows)


def intraday_windows(prepared: Prepared, trades: pd.DataFrame) -> pd.DataFrame:
    minute = prepared.minute[(prepared.minute.weekday == "Monday") & ~prepared.minute.trading_day.isin(prepared.roll_sessions)]
    by_day = {day: group.sort_values("local_time") for day, group in minute.groupby("trading_day")}
    rows = []
    for name, (start, end) in WINDOWS.items():
        window = minute[minute.minute_no.between(start, end - 1)]
        returns = window.groupby("trading_day").apply(lambda x: 10000.0 * (x.iloc[-1].close / x.iloc[0].open - 1.0), include_groups=False)
        for mechanism in MECHANISMS:
            sub = trades[(trades.mechanism == mechanism) & (trades.weekday == "Monday")]
            entry_minutes = ((pd.DatetimeIndex(sub.entry_time).hour + 6) % 24) * 60 + pd.DatetimeIndex(sub.entry_time).minute
            # Convert clock time to minutes since 18:00.
            entry_minutes = ((pd.DatetimeIndex(sub.entry_time).hour * 60 + pd.DatetimeIndex(sub.entry_time).minute - 1080) % 1440)
            in_window = (entry_minutes >= start) & (entry_minutes < end)
            attributed = 0.0
            active_trades = 0
            for trade in sub.itertuples(index=False):
                group = by_day.get(trade.trading_day)
                if group is None:
                    continue
                session_start = pd.Timestamp(trade.trading_day, tz=TZ) - pd.Timedelta(hours=6)
                window_start = session_start + pd.Timedelta(minutes=start)
                window_end = session_start + pd.Timedelta(minutes=end)
                overlap_start = max(pd.Timestamp(trade.entry_time), window_start)
                overlap_end = min(pd.Timestamp(trade.exit_time), window_end)
                if overlap_end <= overlap_start:
                    continue
                segment = group[(group.local_time >= overlap_start) & (group.local_time < overlap_end)]
                if segment.empty:
                    continue
                start_price = float(trade.entry_price) if overlap_start == pd.Timestamp(trade.entry_time) else float(segment.iloc[0].open)
                end_price = float(trade.exit_price) if overlap_end == pd.Timestamp(trade.exit_time) else float(segment.iloc[-1].close)
                attributed += 10000.0 * int(trade.direction) * (end_price - start_price) / float(trade.entry_price)
                active_trades += 1
            rows.append({"asset": prepared.asset, "mechanism": mechanism, "window": name,
                         "mean_btc_return_bps": float(returns.mean()), "median_btc_return_bps": float(returns.median()),
                         "trend_pnl_gross_bps": attributed, "active_trade_windows": active_trades,
                         "trades_entered": int(in_window.sum()), "monday_sessions": int(returns.size)})
    return pd.DataFrame(rows)


def weekend_gaps(prepared: Prepared, trades: pd.DataFrame) -> pd.DataFrame:
    minute = prepared.minute.sort_values("local_time")
    mondays = prepared.sessions[(prepared.sessions.weekday == "Monday") & prepared.sessions.eligible]
    local_ns = pd.DatetimeIndex(minute.local_time).tz_convert("UTC").asi8
    first_by_day = {day: group.iloc[0] for day, group in minute.groupby("trading_day", sort=False)}
    rows = []
    for day in mondays.trading_day:
        sunday_open = first_by_day.get(day)
        cutoff = (pd.Timestamp(day, tz=TZ) - pd.Timedelta(hours=6)).tz_convert("UTC").value
        prior_position = int(np.searchsorted(local_ns, cutoff, side="left") - 1)
        if sunday_open is None or prior_position < 0:
            continue
        open_px = float(sunday_open.open)
        close_px = float(minute.iloc[prior_position].close)
        gap_bps = 10000.0 * (open_px / close_px - 1.0)
        for mechanism in MECHANISMS:
            pnl = trades[(trades.trading_day == day) & (trades.mechanism == mechanism)].gross_bps.sum()
            rows.append({"asset": prepared.asset, "trading_day": day, "mechanism": mechanism, "gap_bps": gap_bps,
                         "gap_sign": int(np.sign(gap_bps)), "monday_trend_pnl_bps": pnl,
                         "gap_pnl_credited_bps": 0.0, "gap_captured": False})
    out = pd.DataFrame(rows)
    if len(out):
        out["gap_pnl_correlation"] = out.groupby(["asset", "mechanism"])["gap_bps"].transform(lambda x: x.corr(out.loc[x.index, "monday_trend_pnl_bps"]))
    return out


def btc_vs_mbt(btc: pd.DataFrame, mbt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    common_start = max(btc.trading_day.min(), mbt.trading_day.min())
    for mechanism in MECHANISMS:
        left = btc[(btc.mechanism == mechanism) & (btc.trading_day >= common_start)].copy().sort_values("entry_time")
        right = mbt[mbt.mechanism == mechanism].copy().rename(columns={"entry_time": "entry_time_mbt"}).sort_values("entry_time_mbt")
        matched = pd.merge_asof(left, right, left_on="entry_time", right_on="entry_time_mbt", by="trading_day", tolerance=pd.Timedelta("60min"), direction="nearest", suffixes=("_btc", "_mbt"))
        good = matched.dropna(subset=["direction_mbt"])
        rows.append({
            "mechanism": mechanism, "common_start": common_start, "btc_trades": len(left), "mbt_trades": len(right), "matched_trades": len(good),
            "direction_agreement": float((good.direction_btc == good.direction_mbt).mean()) if len(good) else np.nan,
            "median_abs_timing_difference_minutes": float((pd.to_datetime(good.entry_time) - pd.to_datetime(good.entry_time_mbt)).dt.total_seconds().abs().median() / 60) if len(good) else np.nan,
            "gross_pnl_bps_correlation": float(good.gross_bps_btc.corr(good.gross_bps_mbt)) if len(good) > 1 else np.nan,
            "mean_entry_fill_difference_bps": float((10000.0 * (good.entry_price_mbt / good.entry_price_btc - 1)).mean()) if len(good) else np.nan,
            "mean_net_1tick_usd_btc_signal": float(left.net_1tick_usd.mean()), "mean_net_1tick_usd_mbt": float(right.net_1tick_usd.mean()),
            "spread_data_available": False, "spread_note": "OHLCV-1m only; no bid/ask spread schema downloaded",
        })
    return pd.DataFrame(rows)


def cost_sensitivity(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scenarios = {"gross": "gross_usd_mbt_equiv", "commission_only": "net_commission_usd", "commission_plus_1tick_side": "net_1tick_usd", "commission_plus_2ticks_side": "net_2tick_usd"}
    for mechanism in MECHANISMS:
        for scope, subset in [("all_days", trades[trades.mechanism == mechanism]), ("monday", trades[(trades.mechanism == mechanism) & (trades.weekday == "Monday")])]:
            for scenario, column in scenarios.items():
                values = subset[column]
                rows.append({"mechanism": mechanism, "scope": scope, "scenario": scenario, "trades": len(subset),
                             "pnl_usd": float(values.sum()), "expectancy_usd": float(values.mean()) if len(values) else np.nan,
                             "profit_factor": _pf(values), "break_even_total_cost_per_trade_usd": float(subset.gross_usd_mbt_equiv.mean()) if len(subset) else np.nan})
    return pd.DataFrame(rows)


def decide(metrics: pd.DataFrame, stability_frame: pd.DataFrame, costs: pd.DataFrame, comparison: pd.DataFrame) -> tuple[str, dict]:
    main = metrics[(metrics.asset == "BTC") & (metrics.mechanism == "equal_risk_ensemble") & (metrics.weekday == "Monday")].iloc[0]
    mbt_cost = costs[(costs.mechanism.isin(MECHANISMS)) & (costs.scope == "monday") & (costs.scenario == "commission_plus_1tick_side")].pnl_usd.sum()
    blocks = stability_frame[(stability_frame.asset == "BTC") & (stability_frame.mechanism == "equal_risk_ensemble") & stability_frame.period.str.startswith("BLOCK_")]
    positive_blocks = int((blocks.monday_minus_tue_fri > 0).sum())
    mean_corr = float(comparison.gross_pnl_bps_correlation.mean())
    distinct = bool(main.monday_minus_tue_fri_expectancy > 0 and main.hac_monday_beta_net_bps > 0 and main.hac_p_value < 0.10)
    stable = positive_blocks >= 2
    tradable = mbt_cost > 0 and mean_corr > 0.50
    if distinct and stable and tradable:
        verdict = "PROMISING_FOR_VALIDATION"
    elif main.monday_minus_tue_fri_expectancy > 0 and mbt_cost > 0:
        verdict = "WEAK_EDGE"
    else:
        verdict = "REJECTED"
    return verdict, {"distinct_hac_10pct": distinct, "positive_blocks": positive_blocks, "mbt_monday_net_1tick_usd": mbt_cost, "mean_btc_mbt_pnl_correlation": mean_corr, "tradable": tradable}


def report(verdict: str, evidence: dict, metrics: pd.DataFrame, stability_frame: pd.DataFrame, comparison: pd.DataFrame, costs: pd.DataFrame, windows: pd.DataFrame, gaps: pd.DataFrame, prepared: dict[str, Prepared]) -> None:
    main = metrics[(metrics.asset == "BTC") & (metrics.mechanism == "equal_risk_ensemble") & (metrics.weekday == "Monday")].iloc[0]
    mon_cost = costs[(costs.scope == "monday") & (costs.scenario == "commission_plus_1tick_side")]
    lines = [
        "# Phase 1 — saisonnalité du trend intraday Bitcoin le lundi", "", f"## Verdict : {verdict}", "",
        "La campagne est indépendante. Les hypothèses 1, EIA Natural Gas et Gotobi 6J n'ont pas été rouvertes; les six sleeves principales n'ont été ni chargées ni retunées.", "",
        "## Protocole gelé", "",
        "- Source unique : futures CME Databento `GLBX.MDP3`, séries continues volume-ranked `BTC.v.0` et `MBT.v.0`, OHLCV une minute.",
        "- Trading day : 18:00 ET la veille à 17:00 ET; le lundi commence donc dimanche 18:00 ET. `America/New_York` gère automatiquement le DST.",
        "- Contrainte Topstep : aucune nouvelle entrée après 16:00 ET et flatten à 16:08 ET, avant l'échéance obligatoire de 16:10 ET.",
        "- Rolls : les sessions où `instrument_id` change intraday sont exclues; aucun saut de roll n'est crédité.",
        "- Mécanismes existants et inchangés : Donchian 20/10 sur 1 h; SuperTrend ATR(10) × 3 sur 15 min; breakout à ±0,6 ATR(14) depuis l'open de session.",
        "- Les signaux sur clôture remplissent au premier open une minute suivant. Le gap vendredi-dimanche est diagnostiqué mais jamais crédité.",
        "- Ensemble : pondération inverse de volatilité retardée 63 sessions, sans information future; poids égaux tant que l'historique est insuffisant.", "",
        "## Données et contrôles", "",
    ]
    for asset, item in prepared.items():
        path = DATA[asset]
        lines.append(f"- {asset}: {len(item.minute):,} minutes, {item.sessions.trading_day.min()} → {item.sessions.trading_day.max()}, {len(item.roll_sessions)} sessions avec roll intraday exclues; SHA-256 `{sha256(path)}`.")
    lines += ["", "Quelques journées sont marquées `degraded` par Databento. Aucune interpolation n'est effectuée; les fills exigent la minute exacte, sauf le flatten qui utilise la dernière clôture antérieure si nécessaire.", "",
              "## Test principal", "",
              f"Pour l'ensemble equal-risk BTC, l'écart lundi moins mardi-vendredi vaut **{main.monday_minus_tue_fri_expectancy:.4f}** dans l'unité de risque du portefeuille. Le coefficient HAC sur le PnL net quotidien est **{main.hac_monday_beta_net_bps:.4f} bp** (t={main.hac_t:.3f}, p={main.hac_p_value:.4f}).",
              f"Blocs naturels positifs : **{evidence['positive_blocks']}/3**. PnL MBT du lundi après commission et 1 tick/côté, somme des trois mécanismes : **{evidence['mbt_monday_net_1tick_usd']:.2f} USD**.", "",
              "Les résultats détaillés par mécanisme et weekday sont dans `weekday_metrics.csv`; les blocs et années sont dans `weekday_by_year.csv`.", "",
              "### Lundi par mécanisme", "",
              "| Actif | Mécanisme | Trades | PnL net | Expectancy | Sharpe | Lundi - autres | HAC p |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    monday_rows = metrics[metrics.weekday == "Monday"]
    for row in monday_rows.itertuples(index=False):
        lines.append(f"| {row.asset} | {row.mechanism} | {row.trades} | {row.net_pnl:.2f} | {row.expectancy_per_trade:.3f} | {row.sharpe:.3f} | {row.monday_minus_tue_fri_expectancy:.3f} | {row.hac_p_value:.4f} |")
    lines += ["", "### Stabilité de l'ensemble BTC", "", "| Bloc | Sharpe lundi | Expectancy lundi | Lundi - autres | Trades |", "|---|---:|---:|---:|---:|"]
    block_rows = stability_frame[(stability_frame.asset == "BTC") & (stability_frame.mechanism == "equal_risk_ensemble") & stability_frame.period.str.startswith("BLOCK_")]
    for row in block_rows.itertuples(index=False):
        lines.append(f"| {row.period} | {row.monday_sharpe:.3f} | {row.monday_expectancy:.3f} | {row.monday_minus_tue_fri:.3f} | {row.monday_trades} |")
    lines += ["",
              "## BTC versus MBT", ""]
    for row in comparison.itertuples(index=False):
        lines.append(f"- {row.mechanism}: {row.matched_trades} trades appariés, accord direction {row.direction_agreement:.1%}, corrélation PnL bp {row.gross_pnl_bps_correlation:.3f}. Spreads indisponibles avec OHLCV-1m.")
    lines += ["", "## Coûts MBT", "", "Commission TopstepX retenue : **2,82 USD round-turn**. Le tick MBT vaut **0,50 USD**, donc 1 tick/côté ajoute 1,00 USD par trade et 2 ticks/côté 2,00 USD. `execution_cost_sensitivity.csv` donne le brut, commission seule et les deux stress tests.", "",
              "## Localisation et weekend gap", "", "Les fenêtres horaires sont descriptives et ne deviennent jamais des règles. `intraday_window_diagnostics.csv` attribue le mark-to-market uniquement aux portions de trades réellement ouvertes dans chaque fenêtre.", ""]
    btc_windows = windows[windows.asset == "BTC"].groupby("window", as_index=False).agg(mean_btc_return_bps=("mean_btc_return_bps", "first"), trend_pnl_gross_bps=("trend_pnl_gross_bps", "sum"), trades_entered=("trades_entered", "sum"))
    lines += ["| Fenêtre | Rendement BTC moyen bp | PnL trend brut bp | Entrées |", "|---|---:|---:|---:|"]
    for row in btc_windows.itertuples(index=False):
        lines.append(f"| {row.window} | {row.mean_btc_return_bps:.3f} | {row.trend_pnl_gross_bps:.2f} | {row.trades_entered} |")
    btc_gaps = gaps[gaps.asset == "BTC"]
    lines += ["", f"Gap weekend BTC moyen : **{btc_gaps.gap_bps.mean():.3f} bp**; part créditée au PnL : **0 bp**. Les corrélations gap/PnL par mécanisme sont dans `weekend_gap_diagnostics.csv`.", "",
              "## Risque de data mining", "", "2018–2025 est une réplication historique, pas un OOS post-publication. Les observations 2026 sont isolables dans le fichier annuel mais ne sont pas qualifiées de véritable OOS sans date de publication vérifiée. Toute donnée ultérieure devra constituer le forward test.", "",
              "## Décision", ""]
    if verdict == "REJECTED":
        lines.append("Le lundi n'est pas suffisamment distinct ou l'avantage ne survit pas aux coûts. Arrêt sans retuning.")
    elif verdict == "WEAK_EDGE":
        lines.append("Une anomalie est visible, mais elle n'est pas assez distincte, stable ou robuste aux coûts pour justifier une Phase 2. Arrêt sans retuning.")
    else:
        lines.append("Le filtre lundi satisfait les critères Phase 1. Arrêt ici; seule une Phase 2 séparée et préenregistrée est proposée.")
    lines += ["", "`orthogonality.csv` n'est pas produit : le test est conditionnel à un effet lundi positif net, condition non satisfaite."]
    lines += ["", "## Sources contractuelles", "", "- CME MBT : https://www.cmegroup.com/content/dam/cmegroup/education/files/getting-started-with-micro-bitcoin-futures.pdf", "- Topstep horaires : https://help.topstep.com/en/articles/8284206-when-and-what-products-can-i-trade", "- Topstep coûts : https://help.topstep.com/en/articles/8284213-topstepx-commissions-and-fees"]
    lines += ["", "## Livrables", "", "- `weekday_metrics.csv`", "- `weekday_by_year.csv`", "- `intraday_window_diagnostics.csv`", "- `weekend_gap_diagnostics.csv`", "- `btc_vs_mbt.csv`", "- `execution_cost_sensitivity.csv`", "- `run_metadata.json`", ""]
    (OUT / "PHASE1_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prepared = {asset: prepare(asset, path) for asset, path in DATA.items()}
    trades = {asset: run_asset(item) for asset, item in prepared.items()}
    daily = {asset: build_daily(prepared[asset], trades[asset]) for asset in DATA}
    metrics = pd.concat([weekday_metrics(asset, trades[asset], daily[asset]) for asset in DATA], ignore_index=True)
    stability_frame = pd.concat([stability(asset, trades[asset], daily[asset]) for asset in DATA], ignore_index=True)
    windows = pd.concat([intraday_windows(prepared[asset], trades[asset]) for asset in DATA], ignore_index=True)
    gaps = pd.concat([weekend_gaps(prepared[asset], trades[asset]) for asset in DATA], ignore_index=True)
    comparison = btc_vs_mbt(trades["BTC"], trades["MBT"])
    costs = cost_sensitivity(trades["MBT"])
    verdict, evidence = decide(metrics, stability_frame, costs, comparison)
    metrics.to_csv(OUT / "weekday_metrics.csv", index=False)
    stability_frame.to_csv(OUT / "weekday_by_year.csv", index=False)
    windows.to_csv(OUT / "intraday_window_diagnostics.csv", index=False)
    gaps.to_csv(OUT / "weekend_gap_diagnostics.csv", index=False)
    comparison.to_csv(OUT / "btc_vs_mbt.csv", index=False)
    costs.to_csv(OUT / "execution_cost_sensitivity.csv", index=False)
    pd.concat(trades.values(), ignore_index=True).to_parquet(OUT / "trade_ledger.parquet")
    metadata = {
        "verdict": verdict, "evidence": evidence, "optimization_performed": False, "grid_search_performed": False,
        "prior_hypotheses_reopened": False, "frozen_sleeves_loaded_or_retuned": False,
        "parameters": {"donchian": "20/10 on 1h", "supertrend": "ATR(10)x3 on 15m", "volatility_breakout": "0.6x prior Wilder ATR(14)"},
        "data": {asset: {"path": str(DATA[asset].relative_to(ROOT)), "sha256": sha256(DATA[asset]), "rows": len(prepared[asset].minute)} for asset in DATA},
        "costs": {"commission_round_turn_usd": COMMISSION_RT_USD, "tick_points": TICK_POINTS, "tick_value_usd": TICK_POINTS * MBT_POINT_VALUE},
    }
    (OUT / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n",
        encoding="utf-8",
    )
    report(verdict, evidence, metrics, stability_frame, comparison, costs, windows, gaps, prepared)
    print(json.dumps(
        {"verdict": verdict, "evidence": evidence, "trades": {k: len(v) for k, v in trades.items()}},
        indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    ))


if __name__ == "__main__":
    main()
