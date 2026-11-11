"""Frozen MNQ baseline tournament engine.

The module deliberately separates mechanism-specific signal construction from
the shared one-minute execution simulator.  It reads parameters from the
frozen JSON documents and never searches, tunes, or mutates them.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dateutil.easter import easter


NY = ZoneInfo("America/New_York")
INITIAL_CAPITAL_USD = 50_000.0
ANNUALIZATION_DAYS = 252


@dataclass(frozen=True)
class EntryInstruction:
    activation_time: pd.Timestamp
    session_date: date
    order_type: str
    signal_time: pd.Timestamp
    direction: int = 0
    entry_level: float | None = None
    long_level: float | None = None
    short_level: float | None = None
    stop_price: float | None = None
    stop_offset_points: float | None = None
    target_price: float | None = None
    long_stop: float | None = None
    short_stop: float | None = None
    long_target: float | None = None
    short_target: float | None = None
    target_offset_points: float | None = None
    require_between: bool = True
    reason: str = "signal"
    setup_id: str = ""


@dataclass(frozen=True)
class ExitInstruction:
    timestamp: pd.Timestamp
    session_date: date
    direction: int
    reason: str


@dataclass(frozen=True)
class StopUpdate:
    timestamp: pd.Timestamp
    session_date: date
    direction: int
    mode: str
    value: float | None = None
    bar_high: float | None = None
    bar_low: float | None = None
    atr: float | None = None


@dataclass
class SignalPlan:
    entries: list[EntryInstruction] = field(default_factory=list)
    exits: list[ExitInstruction] = field(default_factory=list)
    stop_updates: list[StopUpdate] = field(default_factory=list)
    time_exits: dict[date, tuple[time, str]] = field(default_factory=dict)


@dataclass
class PreparedData:
    minute: pd.DataFrame
    sessions: pd.DataFrame
    bars: dict[int, pd.DataFrame]
    daily: pd.DataFrame
    benchmark_daily: pd.DataFrame
    source_path: Path
    source_sha256: str
    data_warnings: list[str]


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    daily: pd.DataFrame
    equity: pd.DataFrame
    positions: pd.DataFrame
    checks: dict[str, Any]


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first_weekday, _ = calendar.monthrange(year, month)
    day = 1 + ((weekday - first_weekday) % 7) + 7 * (n - 1)
    return date(year, month, day)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    _, days = calendar.monthrange(year, month)
    result = date(year, month, days)
    while result.weekday() != weekday:
        result -= timedelta(days=1)
    return result


def _observed_fixed_holiday(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def build_cme_equity_early_close_calendar(start: date, end: date) -> pd.DataFrame:
    """Return deterministic CME equity-index early closes in New York time.

    The rules encode the CME equity-index holiday pattern used by the frozen
    specifications: 13:00 ET on U.S. holiday sessions, 13:15 ET on the day
    before Independence Day, Black Friday and Christmas Eve, and the published
    pre-RTH closes for Good Friday and the 2025 national day of mourning.
    """
    entries: dict[date, tuple[time, str]] = {}

    def add(day: date, close: time, reason: str) -> None:
        if not (start <= day <= end) or day.weekday() >= 5:
            return
        current = entries.get(day)
        if current is None or close < current[0]:
            entries[day] = (close, reason)

    for year in range(start.year, end.year + 1):
        add(_nth_weekday(year, 1, calendar.MONDAY, 3), time(13, 0), "Martin Luther King Jr. Day")
        add(_nth_weekday(year, 2, calendar.MONDAY, 3), time(13, 0), "Presidents Day")
        add(_last_weekday(year, 5, calendar.MONDAY), time(13, 0), "Memorial Day")
        add(_nth_weekday(year, 9, calendar.MONDAY, 1), time(13, 0), "Labor Day")
        thanksgiving = _nth_weekday(year, 11, calendar.THURSDAY, 4)
        add(thanksgiving, time(13, 0), "Thanksgiving Day")
        add(thanksgiving + timedelta(days=1), time(13, 15), "Day after Thanksgiving")

        if year >= 2022:
            add(_observed_fixed_holiday(date(year, 6, 19)), time(13, 0), "Juneteenth observed")

        independence_observed = _observed_fixed_holiday(date(year, 7, 4))
        add(independence_observed, time(13, 0), "Independence Day observed")
        july_third = date(year, 7, 3)
        if july_third.weekday() < 5 and july_third != independence_observed:
            add(july_third, time(13, 15), "Independence Day eve")

        christmas_eve = date(year, 12, 24)
        if christmas_eve.weekday() < 5:
            add(christmas_eve, time(13, 15), "Christmas Eve")

        add(easter(year) - timedelta(days=2), time(9, 15), "Good Friday")

    # CME Group announced an 08:30 CT close for U.S. equities on this day.
    add(date(2025, 1, 9), time(9, 30), "U.S. National Day of Mourning")

    rows = [
        {
            "session_date": day,
            "early_close_time": close.isoformat(),
            "reason": reason,
            "calendar": "official_CME_equity_index_futures_calendar",
        }
        for day, (close, reason) in sorted(entries.items())
    ]
    return pd.DataFrame(rows)


def _timestamp(day: date, local_time: time) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day, local_time, tzinfo=NY))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_specs(frozen_dir: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if len(manifest) != 17:
        raise ValueError(f"Expected 17 frozen manifest rows, found {len(manifest)}")

    specs: list[dict[str, Any]] = []
    benchmark: dict[str, Any] | None = None
    for row in manifest.itertuples(index=False):
        path = frozen_dir.parent / row.specification_path
        actual_hash = _sha256(path)
        if actual_hash != row.specification_sha256:
            raise ValueError(f"Frozen specification hash mismatch: {row.mechanism_id}")
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec["_frozen_spec_sha256"] = actual_hash
        spec["_frozen_spec_path"] = str(path)
        if spec["specification_type"] == "market_benchmark":
            benchmark = spec
        else:
            specs.append(spec)

    if len(specs) != 16 or benchmark is None:
        raise ValueError("Frozen registry must contain 16 strategies and MNQ_BUY_AND_HOLD")
    for spec in specs:
        execution = spec["canonical_parameters"]["execution"]
        required = {
            "session_timezone": "America/New_York",
            "commission_per_side_usd": 1.25,
            "slippage_ticks_per_side": 1,
            "tick_size": 0.25,
            "point_value_usd": 2.0,
            "position_policy": "one_position_no_pyramiding",
            "bar_conflict_policy": "stop_before_target",
            "forced_flatten_time": "16:08:00",
        }
        for key, expected in required.items():
            if execution.get(key) != expected:
                raise ValueError(f"Unsupported frozen execution value {spec['mechanism_id']}::{key}")
    return sorted(specs, key=lambda item: item["mechanism_id"]), benchmark, manifest


def _validate_minute_data(data: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(data.columns):
        raise ValueError(f"Minute data missing fields: {sorted(required - set(data.columns))}")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Minute data must use a DatetimeIndex")
    if data.index.tz is None:
        raise ValueError("Minute data timestamps must be timezone-aware")
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("Minute timestamps must be unique and increasing")
    if data[list(required)].isna().any().any():
        raise ValueError("Minute OHLCV contains missing values")
    bad = (
        (data["high"] < data[["open", "close"]].max(axis=1))
        | (data["low"] > data[["open", "close"]].min(axis=1))
        | (data["high"] < data["low"])
    )
    if bad.any():
        raise ValueError(f"Minute OHLC invariants fail on {int(bad.sum())} rows")


def _resample_rth(minute: pd.DataFrame, sessions: pd.DataFrame, minutes: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_date = {day: frame for day, frame in minute.groupby("session_date", sort=True)}
    for session in sessions.itertuples(index=False):
        frame = by_date.get(session.session_date)
        if frame is None:
            continue
        cash_close = pd.Timestamp(session.cash_close)
        rth = frame[(frame.index >= pd.Timestamp(session.rth_start)) & (frame.index < cash_close)]
        if rth.empty:
            continue
        start = pd.Timestamp(session.rth_start)
        offsets = ((rth.index - start).total_seconds() // 60).astype(int)
        buckets = offsets // minutes
        for bucket, group in rth.groupby(buckets, sort=True):
            bar_start = start + pd.Timedelta(minutes=int(bucket) * minutes)
            bar_end = bar_start + pd.Timedelta(minutes=minutes)
            if bar_end > cash_close:
                continue
            rows.append(
                {
                    "session_date": session.session_date,
                    "start_time": bar_start,
                    "end_time": bar_end,
                    "open": float(group["open"].iloc[0]),
                    "high": float(group["high"].max()),
                    "low": float(group["low"].min()),
                    "close": float(group["close"].iloc[-1]),
                    "volume": float(group["volume"].sum()),
                    "source_minute_count": int(len(group)),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("end_time").reset_index(drop=True)
    return result


def prepare_data(source_path: Path, start_date: date | None = None, end_date: date | None = None) -> PreparedData:
    filters: list[tuple[str, str, pd.Timestamp]] = []
    if start_date is not None:
        filters.append(("timestamp", ">=", pd.Timestamp(start_date, tz=NY)))
    if end_date is not None:
        filters.append(("timestamp", "<", pd.Timestamp(end_date + timedelta(days=1), tz=NY)))
    raw = pd.read_parquet(
        source_path,
        columns=["open", "high", "low", "close", "volume"],
        filters=filters or None,
    )
    _validate_minute_data(raw)
    raw = raw.copy()
    raw.index = raw.index.tz_convert(NY)
    raw.index.name = "timestamp"
    if start_date is not None:
        raw = raw[raw.index.date >= start_date]
    if end_date is not None:
        raw = raw[raw.index.date <= end_date]
    if raw.empty:
        raise ValueError("No minute data inside the requested date window")
    raw["session_date"] = raw.index.date

    start_date = raw.index.min().date()
    end_date = raw.index.max().date()
    early = build_cme_equity_early_close_calendar(start_date, end_date)
    early_map = {
        row.session_date: (time.fromisoformat(row.early_close_time), row.reason)
        for row in early.itertuples(index=False)
    }

    benchmark_rows: list[dict[str, Any]] = []
    for day, group in raw.groupby("session_date", sort=True):
        if day.weekday() >= 5:
            continue
        close_time, reason = early_map.get(day, (time(16, 0), "regular"))
        cash_close = _timestamp(day, close_time)
        expected_close_bar = cash_close - pd.Timedelta(minutes=1)
        if expected_close_bar not in group.index:
            continue
        benchmark_rows.append(
            {
                "session_date": day,
                "close": float(group.loc[expected_close_bar, "close"]),
                "close_timestamp": expected_close_bar,
                "cash_close": cash_close,
                "early_close": close_time != time(16, 0),
                "early_close_reason": reason,
            }
        )
    benchmark_daily = pd.DataFrame(benchmark_rows).sort_values("session_date").reset_index(drop=True)

    session_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for day, group in raw.groupby("session_date", sort=True):
        if day.weekday() >= 5:
            continue
        start_ts = _timestamp(day, time(9, 30))
        if start_ts not in group.index:
            continue
        close_time, reason = early_map.get(day, (time(16, 0), "regular"))
        if close_time <= time(9, 30):
            continue
        cash_close = _timestamp(day, close_time)
        last_entry = cash_close - pd.Timedelta(minutes=5) if close_time != time(16, 0) else _timestamp(day, time(15, 55))
        hard_flatten = cash_close - pd.Timedelta(minutes=1) if close_time != time(16, 0) else _timestamp(day, time(16, 8))
        session_rows.append(
            {
                "session_date": day,
                "rth_start": start_ts,
                "cash_close": cash_close,
                "calendar_last_entry": last_entry,
                "hard_flatten": hard_flatten,
                "early_close": close_time != time(16, 0),
                "early_close_reason": reason,
            }
        )
    sessions = pd.DataFrame(session_rows).sort_values("session_date").reset_index(drop=True)
    valid_dates = set(sessions["session_date"])
    minute = raw[raw["session_date"].isin(valid_dates)].copy()
    session_end = sessions.set_index("session_date")["hard_flatten"].to_dict()
    session_start = sessions.set_index("session_date")["rth_start"].to_dict()
    mask = [session_start[d] <= ts <= session_end[d] for ts, d in zip(minute.index, minute["session_date"])]
    minute = minute.loc[mask, ["open", "high", "low", "close", "volume", "session_date"]]

    daily_rows: list[dict[str, Any]] = []
    minute_by_date = {day: frame for day, frame in minute.groupby("session_date", sort=False)}
    for session in sessions.itertuples(index=False):
        group = minute_by_date.get(session.session_date)
        if group is None:
            warnings.append(f"missing_session:{session.session_date}")
            continue
        rth = group[(group.index >= session.rth_start) & (group.index < session.cash_close)]
        if rth.empty:
            warnings.append(f"empty_rth:{session.session_date}")
            continue
        daily_rows.append(
            {
                "session_date": session.session_date,
                "open": float(rth["open"].iloc[0]),
                "high": float(rth["high"].max()),
                "low": float(rth["low"].min()),
                "close": float(rth["close"].iloc[-1]),
                "close_timestamp": pd.Timestamp(rth.index[-1]),
                "volume": float(rth["volume"].sum()),
            }
        )
    daily = pd.DataFrame(daily_rows).sort_values("session_date").reset_index(drop=True)
    daily["true_range"] = _true_range(daily)
    daily["atr14"] = _wilder_rma(daily["true_range"], 14)
    daily["prior_close"] = daily["close"].shift(1)
    daily["prior_atr14"] = daily["atr14"].shift(1)

    bars = {interval: _resample_rth(minute, sessions, interval) for interval in (5, 15, 60)}
    return PreparedData(
        minute=minute,
        sessions=sessions,
        bars=bars,
        daily=daily,
        benchmark_daily=benchmark_daily,
        source_path=source_path,
        source_sha256=_sha256(source_path),
        data_warnings=warnings,
    )


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    valid_run: list[float] = []
    previous = np.nan
    for index, value in enumerate(values):
        if np.isnan(value):
            valid_run = []
            previous = np.nan
            continue
        if np.isnan(previous):
            valid_run.append(value)
            if len(valid_run) == period:
                previous = float(np.mean(valid_run))
                result[index] = previous
            continue
        previous = ((period - 1) * previous + value) / period
        result[index] = previous
    return pd.Series(result, index=series.index, dtype=float)


def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    values = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return values.max(axis=1)


def _atr(frame: pd.DataFrame, period: int) -> pd.Series:
    return _wilder_rma(_true_range(frame), period)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = _wilder_rma(gains, period)
    average_loss = _wilder_rma(losses, period)
    ratio = average_gain / average_loss.replace(0.0, np.nan)
    result = 100.0 - 100.0 / (1.0 + ratio)
    result = result.mask((average_loss == 0.0) & (average_gain > 0.0), 100.0)
    result = result.mask((average_gain == 0.0) & (average_loss > 0.0), 0.0)
    result = result.mask((average_gain == 0.0) & (average_loss == 0.0), 50.0)
    return result


def _dmi_adx(frame: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = frame["high"].diff()
    down_move = -frame["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=frame.index)
    atr = _atr(frame, period)
    plus_di = 100.0 * _wilder_rma(plus_dm, period) / atr.replace(0.0, np.nan)
    minus_di = 100.0 * _wilder_rma(minus_dm, period) / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = _wilder_rma(dx, period)
    return plus_di, minus_di, adx


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _market_entry(
    row: pd.Series,
    direction: int,
    *,
    stop: float | None = None,
    target: float | None = None,
    target_offset: float | None = None,
    stop_offset: float | None = None,
    reason: str,
    require_between: bool = True,
) -> EntryInstruction:
    return EntryInstruction(
        activation_time=pd.Timestamp(row["end_time"]),
        session_date=row["session_date"],
        order_type="market_open",
        signal_time=pd.Timestamp(row["end_time"]),
        direction=direction,
        stop_price=None if stop is None or pd.isna(stop) else float(stop),
        stop_offset_points=stop_offset,
        target_price=None if target is None or pd.isna(target) else float(target),
        target_offset_points=target_offset,
        require_between=require_between,
        reason=reason,
    )


def _exit(row: pd.Series, direction: int, reason: str) -> ExitInstruction:
    return ExitInstruction(
        timestamp=pd.Timestamp(row["end_time"]),
        session_date=row["session_date"],
        direction=direction,
        reason=reason,
    )


def _build_trend_pullback(spec: dict[str, Any], data: PreparedData) -> SignalPlan:
    bars = data.bars[15].copy()
    periods = [int(value) for value in spec["canonical_parameters"]["structural"]["ema_periods"]]
    fast, pullback, exit_reference, _, slow = periods
    for period in periods:
        bars[f"ema{period}"] = _ema(bars["close"], int(period))
    plan = SignalPlan()
    previous = bars.shift(1)
    for index, row in bars.iterrows():
        values = [row[f"ema{period}"] for period in periods]
        if any(pd.isna(value) for value in values):
            continue
        long_trend = all(values[i] > values[i + 1] for i in range(len(values) - 1))
        short_trend = all(values[i] < values[i + 1] for i in range(len(values) - 1))
        overlaps = row["low"] <= max(row[f"ema{fast}"], row[f"ema{pullback}"]) and row["high"] >= min(row[f"ema{fast}"], row[f"ema{pullback}"])
        if long_trend and overlaps and row["close"] >= row[f"ema{pullback}"]:
            plan.entries.append(_market_entry(row, 1, stop=row[f"ema{slow}"], reason="ema_ribbon_pullback"))
        elif short_trend and overlaps and row["close"] <= row[f"ema{pullback}"]:
            plan.entries.append(_market_entry(row, -1, stop=row[f"ema{slow}"], reason="ema_ribbon_pullback"))

        if index > 0 and pd.notna(previous.loc[index, f"ema{fast}"]) and pd.notna(previous.loc[index, f"ema{exit_reference}"]):
            if previous.loc[index, f"ema{fast}"] >= previous.loc[index, f"ema{exit_reference}"] and row[f"ema{fast}"] < row[f"ema{exit_reference}"]:
                plan.exits.append(_exit(row, 1, "ema8_cross_below_ema21"))
            if previous.loc[index, f"ema{fast}"] <= previous.loc[index, f"ema{exit_reference}"] and row[f"ema{fast}"] > row[f"ema{exit_reference}"]:
                plan.exits.append(_exit(row, -1, "ema8_cross_above_ema21"))
        plan.stop_updates.extend(
            [
                StopUpdate(pd.Timestamp(row["end_time"]), row["session_date"], 1, "monotonic", value=float(row[f"ema{slow}"])),
                StopUpdate(pd.Timestamp(row["end_time"]), row["session_date"], -1, "monotonic", value=float(row[f"ema{slow}"])),
            ]
        )
    return plan


def _build_vwap_reversion(spec: dict[str, Any], data: PreparedData) -> SignalPlan:
    params = spec["canonical_parameters"]["structural"]
    entry_sigma = float(params["entry_band_sigma"])
    minimum_bars = int(params["minimum_completed_bars"])
    bars = data.bars[5].copy()
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    bars["pv"] = typical * bars["volume"]
    grouped = bars.groupby("session_date", sort=False)
    bars["cum_volume"] = grouped["volume"].cumsum()
    bars["cum_pv"] = grouped["pv"].cumsum()
    bars["vwap"] = bars["cum_pv"] / bars["cum_volume"]
    bars["variance_part"] = bars["volume"] * typical * typical
    bars["cum_p2v"] = bars.groupby("session_date", sort=False)["variance_part"].cumsum()
    bars["sigma"] = np.sqrt((bars["cum_p2v"] / bars["cum_volume"] - bars["vwap"] ** 2).clip(lower=0.0))
    bars["session_bar"] = bars.groupby("session_date").cumcount() + 1
    plan = SignalPlan()
    for _, row in bars.iterrows():
        if row["session_bar"] < minimum_bars or not np.isfinite(row["sigma"]) or row["sigma"] <= 0:
            continue
        lower2 = row["vwap"] - entry_sigma * row["sigma"]
        upper2 = row["vwap"] + entry_sigma * row["sigma"]
        if row["low"] <= lower2 and row["close"] > row["open"]:
            plan.entries.append(_market_entry(row, 1, stop=row["vwap"] - 3.0 * row["sigma"], target=row["vwap"], reason="vwap_minus_2sigma"))
        elif row["high"] >= upper2 and row["close"] < row["open"]:
            plan.entries.append(_market_entry(row, -1, stop=row["vwap"] + 3.0 * row["sigma"], target=row["vwap"], reason="vwap_plus_2sigma"))
    return plan


def _build_donchian(spec: dict[str, Any], data: PreparedData) -> SignalPlan:
    params = spec["canonical_parameters"]["structural"]
    entry_length = int(params["entry_channel_length"])
    exit_length = int(params["exit_channel_length"])
    bars = data.bars[60].copy()
    bars["entry_high"] = bars["high"].rolling(entry_length, min_periods=entry_length).max().shift(1)
    bars["entry_low"] = bars["low"].rolling(entry_length, min_periods=entry_length).min().shift(1)
    bars["exit_high"] = bars["high"].rolling(exit_length, min_periods=exit_length).max().shift(1)
    bars["exit_low"] = bars["low"].rolling(exit_length, min_periods=exit_length).min().shift(1)
    plan = SignalPlan()
    for _, row in bars.iterrows():
        if pd.notna(row["exit_low"]) and row["close"] < row["exit_low"]:
            plan.exits.append(_exit(row, 1, "donchian_10_bar_exit"))
        if pd.notna(row["exit_high"]) and row["close"] > row["exit_high"]:
            plan.exits.append(_exit(row, -1, "donchian_10_bar_exit"))
        if pd.notna(row["entry_high"]) and row["close"] > row["entry_high"]:
            plan.entries.append(_market_entry(row, 1, reason="donchian_20_bar_breakout", require_between=False))
        elif pd.notna(row["entry_low"]) and row["close"] < row["entry_low"]:
            plan.entries.append(_market_entry(row, -1, reason="donchian_20_bar_breakout", require_between=False))
    return plan


def _supertrend(frame: pd.DataFrame, period: int, multiplier: float) -> pd.DataFrame:
    atr = _atr(frame, period)
    midpoint = (frame["high"] + frame["low"]) / 2.0
    basic_upper = midpoint + multiplier * atr
    basic_lower = midpoint - multiplier * atr
    final_upper = np.full(len(frame), np.nan)
    final_lower = np.full(len(frame), np.nan)
    state = np.zeros(len(frame), dtype=int)
    first_valid = next((i for i, value in enumerate(atr) if pd.notna(value)), None)
    if first_valid is None:
        return pd.DataFrame({"atr": atr, "final_upper": final_upper, "final_lower": final_lower, "state": state})
    for i in range(first_valid, len(frame)):
        if i == first_valid:
            final_upper[i] = basic_upper.iloc[i]
            final_lower[i] = basic_lower.iloc[i]
            state[i] = -1
            continue
        previous_close = frame["close"].iloc[i - 1]
        final_upper[i] = basic_upper.iloc[i] if basic_upper.iloc[i] < final_upper[i - 1] or previous_close > final_upper[i - 1] else final_upper[i - 1]
        final_lower[i] = basic_lower.iloc[i] if basic_lower.iloc[i] > final_lower[i - 1] or previous_close < final_lower[i - 1] else final_lower[i - 1]
        if state[i - 1] == -1:
            state[i] = 1 if frame["close"].iloc[i] > final_upper[i] else -1
        else:
            state[i] = -1 if frame["close"].iloc[i] < final_lower[i] else 1
    return pd.DataFrame({"atr": atr, "final_upper": final_upper, "final_lower": final_lower, "state": state}, index=frame.index)


def _build_supertrend(spec: dict[str, Any], data: PreparedData) -> SignalPlan:
    bars = data.bars[15].copy()
    params = spec["canonical_parameters"]["structural"]
    values = _supertrend(bars, int(params["atr_period"]), float(params["atr_multiplier"]))
    bars = pd.concat([bars, values], axis=1)
    plan = SignalPlan()
    for index in range(1, len(bars)):
        row = bars.iloc[index]
        previous = bars.iloc[index - 1]
        if row["state"] == 0 or previous["state"] == 0 or row["state"] == previous["state"]:
            continue
        new_direction = int(row["state"])
        plan.exits.append(_exit(row, -new_direction, "opposite_supertrend_flip"))
        plan.entries.append(_market_entry(row, new_direction, reason="supertrend_flip", require_between=False))
    return plan


def _build_opening_range(spec: dict[str, Any], data: PreparedData) -> SignalPlan:
    params = spec["canonical_parameters"]["structural"]
    opening_minutes = int(params["opening_range_minutes"])
    buffer_ticks = int(params["breakout_buffer_ticks"])
    opening_start = time.fromisoformat(
        str(params.get("opening_range_start", spec["canonical_parameters"]["execution"]["rth_start"]))
    )
    bars = data.bars[5].copy()
    plan = SignalPlan()
    tick = float(spec["canonical_parameters"]["execution"]["tick_size"])
    for day, group in bars.groupby("session_date", sort=True):
        group = group.sort_values("start_time")
        opening_start_ts = _timestamp(day, opening_start)
        opening_end = opening_start_ts + pd.Timedelta(minutes=opening_minutes)
        opening = group[(group["start_time"] >= opening_start_ts) & (group["start_time"] < opening_end)]
        if len(opening) != opening_minutes // 5:
            continue
        high = float(opening["high"].max())
        low = float(opening["low"].min())
        width = high - low
        if width <= 0:
            continue
        midpoint = (high + low) / 2.0
        candidates = group[group["start_time"] >= opening_end]
        previous_volume: float | None = float(opening.iloc[-1]["volume"])
        for _, row in candidates.iterrows():
            expanding = previous_volume is not None and row["volume"] > previous_volume
            previous_volume = float(row["volume"])
            if not expanding:
                continue
            if row["close"] >= high + buffer_ticks * tick:
                plan.entries.append(_market_entry(row, 1, stop=midpoint, target=high + width, reason="opening_range_breakout"))
                break
            if row["close"] <= low - buffer_ticks * tick:
                plan.entries.append(_market_entry(row, -1, stop=midpoint, target=low - width, reason="opening_range_breakout"))
                break
    return plan


def _build_compression(spec: dict[str, Any], data: PreparedData) -> SignalPlan:
    params = spec["canonical_parameters"]["structural"]
    risk = spec["canonical_parameters"]["risk_management"]
    lookback = int(params["nr_lookback"])
    buffer_ticks = int(params["entry_buffer_ticks"])
    target_multiple = float(risk["target_setup_range_multiple"])
    bars = data.bars[60].copy()
    ranges = bars["high"] - bars["low"]
    tick = float(spec["canonical_parameters"]["execution"]["tick_size"])
    plan = SignalPlan()
    for index in range(lookback - 1, len(bars)):
        row = bars.iloc[index]
        previous = bars.iloc[index - 1]
        window = ranges.iloc[index - lookback + 1 : index + 1]
        is_earliest_minimum = ranges.iloc[index] < window.iloc[:-1].min()
        inside = row["high"] <= previous["high"] and row["low"] >= previous["low"]
        if not (is_earliest_minimum and inside):
            continue
        width = float(ranges.iloc[index])
        if width <= 0:
            continue
        long_level = float(row["high"] + buffer_ticks * tick)
        short_level = float(row["low"] - buffer_ticks * tick)
        plan.entries.append(
            EntryInstruction(
                activation_time=pd.Timestamp(row["end_time"]),
                session_date=row["session_date"],
                order_type="stop_oco",
                signal_time=pd.Timestamp(row["end_time"]),
                long_level=long_level,
                short_level=short_level,
                long_stop=float(row["low"]),
                short_stop=float(row["high"]),
                long_target=long_level + target_multiple * width,
                short_target=short_level - target_multiple * width,
                reason="inside_nr7_breakout",
                setup_id=f"nr7:{row['end_time']}",
            )
        )
    return plan


def _build_zscore(spec: dict[str, Any], data: PreparedData) -> SignalPlan:
    params = spec["canonical_parameters"]["structural"]
    lookback = int(params["mean_lookback"])
    entry_z = float(params["entry_z_abs"])
    bars = data.bars[60].copy()
    mean = bars["close"].rolling(lookback, min_periods=lookback).mean()
    sigma = bars["close"].rolling(lookback, min_periods=lookback).std(ddof=0)
    bars["z"] = (bars["close"] - mean) / sigma.replace(0.0, np.nan)
    plan = SignalPlan()
    for index in range(1, len(bars)):
        row = bars.iloc[index]
        prior = bars.iloc[index - 1]
        if pd.isna(row["z"]) or pd.isna(prior["z"]):
            continue
        if row["z"] <= -3.0:
            plan.exits.append(_exit(row, 1, "zscore_stop"))
        elif row["z"] >= -0.5:
            plan.exits.append(_exit(row, 1, "zscore_target"))
        if row["z"] >= 3.0:
            plan.exits.append(_exit(row, -1, "zscore_stop"))
        elif row["z"] <= 0.5:
            plan.exits.append(_exit(row, -1, "zscore_target"))
        if prior["z"] >= -entry_z and row["z"] < -entry_z:
            plan.entries.append(_market_entry(row, 1, reason="zscore_minus_2_cross", require_between=False))
        elif prior["z"] <= entry_z and row["z"] > entry_z:
            plan.entries.append(_market_entry(row, -1, reason="zscore_plus_2_cross", require_between=False))
    return plan


def _rolling_regression(close: pd.Series, lookback: int) -> pd.DataFrame:
    slope = np.full(len(close), np.nan)
    endpoint = np.full(len(close), np.nan)
    sigma = np.full(len(close), np.nan)
    x = np.arange(lookback, dtype=float)
    for index in range(lookback - 1, len(close)):
        y = close.iloc[index - lookback + 1 : index + 1].to_numpy(dtype=float)
        coefficient, intercept = np.polyfit(x, y, 1)
        fitted = intercept + coefficient * x
        slope[index] = coefficient
        endpoint[index] = fitted[-1]
        sigma[index] = float(np.std(y - fitted, ddof=0))
    return pd.DataFrame({"reg_slope": slope, "reg_endpoint": endpoint, "reg_sigma": sigma}, index=close.index)


def _build_regression(spec: dict[str, Any], data: PreparedData) -> SignalPlan:
    params = spec["canonical_parameters"]["structural"]
    lookback = int(params["regression_lookback"])
    entry_sigma = float(params["entry_residual_sigma"])
    bars = data.bars[60].copy()
    bars = pd.concat([bars, _rolling_regression(bars["close"], lookback)], axis=1)
    plan = SignalPlan()
    for index in range(1, len(bars)):
        row = bars.iloc[index]
        prior = bars.iloc[index - 1]
        if pd.isna(row["reg_sigma"]) or row["reg_sigma"] <= 0 or pd.isna(prior["reg_sigma"]):
            continue
        lower2 = row["reg_endpoint"] - entry_sigma * row["reg_sigma"]
        upper2 = row["reg_endpoint"] + entry_sigma * row["reg_sigma"]
        prior_lower2 = prior["reg_endpoint"] - entry_sigma * prior["reg_sigma"]
        prior_upper2 = prior["reg_endpoint"] + entry_sigma * prior["reg_sigma"]
        if row["close"] <= row["reg_endpoint"] - 3.0 * row["reg_sigma"]:
            plan.exits.append(_exit(row, 1, "regression_3sigma_stop"))
        elif row["close"] >= row["reg_endpoint"]:
            plan.exits.append(_exit(row, 1, "regression_endpoint_target"))
        if row["close"] >= row["reg_endpoint"] + 3.0 * row["reg_sigma"]:
            plan.exits.append(_exit(row, -1, "regression_3sigma_stop"))
        elif row["close"] <= row["reg_endpoint"]:
            plan.exits.append(_exit(row, -1, "regression_endpoint_target"))
        if prior["close"] >= prior_lower2 and row["close"] < lower2 and row["reg_slope"] > 0:
            plan.entries.append(_market_entry(row, 1, reason="regression_minus_2sigma_cross", require_between=False))
        elif prior["close"] <= prior_upper2 and row["close"] > upper2 and row["reg_slope"] < 0:
            plan.entries.append(_market_entry(row, -1, reason="regression_plus_2sigma_cross", require_between=False))
    return plan


def _build_volatility_breakout(spec: dict[str, Any], data: PreparedData) -> SignalPlan:
    params = spec["canonical_parameters"]["structural"]
    risk = spec["canonical_parameters"]["risk_management"]
    execution = spec["canonical_parameters"]["execution"]
    atr_period = int(params["daily_atr_period"])
    entry_multiple = float(params["entry_atr_multiple"])
    daily = data.daily.copy()
    daily["study_prior_atr"] = _wilder_rma(daily["true_range"], atr_period).shift(1)
    plan = SignalPlan()
    for row in daily.itertuples(index=False):
        if pd.isna(row.study_prior_atr):
            continue
        session = data.sessions.loc[data.sessions["session_date"] == row.session_date].iloc[0]
        open_price = float(row.open)
        distance = entry_multiple * float(row.study_prior_atr)
        plan.entries.append(
            EntryInstruction(
                activation_time=pd.Timestamp(session["rth_start"]),
                session_date=row.session_date,
                order_type="stop_oco",
                signal_time=pd.Timestamp(session["rth_start"]),
                long_level=open_price + distance,
                short_level=open_price - distance,
                long_stop=open_price,
                short_stop=open_price,
                reason="session_open_prior_atr_breakout",
                setup_id=f"volatility:{row.session_date}",
            )
        )
        plan.time_exits[row.session_date] = (time.fromisoformat(str(pd.Timestamp(session["cash_close"]).time())), "cash_close")
    return plan


def _build_rsi2(spec: dict[str, Any], data: PreparedData) -> SignalPlan:
    params = spec["canonical_parameters"]["structural"]
    rsi_period = int(params["rsi_period"])
    long_entry = float(params["long_entry_rsi"])
    short_entry = float(params["short_entry_rsi"])
    trend_period = int(params["trend_sma_period"])
    bars = data.bars[15].copy()
    bars["rsi"] = _rsi(bars["close"], rsi_period)
    bars["sma200"] = bars["close"].rolling(trend_period, min_periods=trend_period).mean()
    plan = SignalPlan()
    for index in range(1, len(bars)):
        row = bars.iloc[index]
        prior = bars.iloc[index - 1]
        if pd.isna(row["rsi"]) or pd.isna(prior["rsi"]):
            continue
        if prior["rsi"] <= 65.0 and row["rsi"] > 65.0:
            plan.exits.append(_exit(row, 1, "rsi2_target"))
        if prior["rsi"] >= 35.0 and row["rsi"] < 35.0:
            plan.exits.append(_exit(row, -1, "rsi2_target"))
        if pd.isna(row["sma200"]):
            continue
        if prior["rsi"] >= long_entry and row["rsi"] < long_entry and row["close"] > row["sma200"]:
            plan.entries.append(_market_entry(row, 1, reason="rsi2_pullback", require_between=False))
        elif prior["rsi"] <= short_entry and row["rsi"] > short_entry and row["close"] < row["sma200"]:
            plan.entries.append(_market_entry(row, -1, reason="rsi2_pullback", require_between=False))
    return plan


def _build_momentum_breakout(spec: dict[str, Any], data: PreparedData) -> SignalPlan:
    params = spec["canonical_parameters"]["structural"]
    lookback = int(params["breakout_lookback"])
    adx_threshold = float(params["adx_threshold"])
    bars = data.bars[60].copy()
    bars["prior_high20"] = bars["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    bars["prior_low20"] = bars["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    bars["rsi"] = _rsi(bars["close"], 14)
    bars["atr"] = _atr(bars, 14)
    _, _, bars["adx"] = _dmi_adx(bars, 14)
    plan = SignalPlan()
    for _, row in bars.iterrows():
        if pd.notna(row["atr"]):
            plan.stop_updates.extend(
                [
                    StopUpdate(pd.Timestamp(row["end_time"]), row["session_date"], 1, "momentum_atr_trail", bar_high=float(row["high"]), bar_low=float(row["low"]), atr=float(row["atr"])),
                    StopUpdate(pd.Timestamp(row["end_time"]), row["session_date"], -1, "momentum_atr_trail", bar_high=float(row["high"]), bar_low=float(row["low"]), atr=float(row["atr"])),
                ]
            )
        if any(pd.isna(row[name]) for name in ("prior_high20", "prior_low20", "rsi", "atr", "adx")):
            continue
        if row["close"] > row["prior_high20"] and row["rsi"] > 60.0 and row["adx"] > adx_threshold:
            plan.entries.append(_market_entry(row, 1, stop_offset=2.0 * float(row["atr"]), reason="momentum_confirmed_breakout"))
        elif row["close"] < row["prior_low20"] and row["rsi"] < 40.0 and row["adx"] > adx_threshold:
            plan.entries.append(_market_entry(row, -1, stop_offset=2.0 * float(row["atr"]), reason="momentum_confirmed_breakout"))
    return plan


def _daily_lookup(data: PreparedData) -> dict[date, pd.Series]:
    return {row["session_date"]: row for _, row in data.daily.iterrows()}


def _build_gap_fade(spec: dict[str, Any], data: PreparedData) -> SignalPlan:
    params = spec["canonical_parameters"]["structural"]
    risk = spec["canonical_parameters"]["risk_management"]
    execution = spec["canonical_parameters"]["execution"]
    atr_period = int(params["daily_atr_period"])
    max_gap_multiple = float(params["maximum_gap_atr_multiple"])
    minimum_gap_ticks = int(params["minimum_gap_ticks"])
    daily = data.daily.copy()
    daily["study_prior_atr"] = _wilder_rma(daily["true_range"], atr_period).shift(1)
    plan = SignalPlan()
    tick = float(spec["canonical_parameters"]["execution"]["tick_size"])
    for row in daily.itertuples(index=False):
        if pd.isna(row.prior_close) or pd.isna(row.study_prior_atr):
            continue
        gap = float(row.open - row.prior_close)
        if abs(gap) < minimum_gap_ticks * tick or abs(gap) >= max_gap_multiple * float(row.study_prior_atr):
            continue
        direction = -1 if gap > 0 else 1
        stop = float(row.open + np.sign(gap) * 0.5 * abs(gap))
        signal_row = pd.Series(
            {
                "end_time": _timestamp(row.session_date, time.fromisoformat(execution["last_entry_time"])),
                "session_date": row.session_date,
            }
        )
        plan.entries.append(_market_entry(signal_row, direction, stop=stop, target=float(row.prior_close), reason="opening_gap_fade"))
        plan.time_exits[row.session_date] = (time.fromisoformat(risk["midday_time_exit"]), "midday_exit")
    return plan


def _build_gap_continuation(spec: dict[str, Any], data: PreparedData) -> SignalPlan:
    params = spec["canonical_parameters"]["structural"]
    execution = spec["canonical_parameters"]["execution"]
    gap_multiple = float(params["minimum_gap_atr_multiple"])
    volume_lookback = int(params["volume_average_lookback"])
    volume_multiple = float(params["volume_threshold_multiple"])
    bars = data.bars[15].copy()
    bars["prior_volume_mean20"] = bars["volume"].rolling(volume_lookback, min_periods=volume_lookback).mean().shift(1)
    first_bars = bars[bars["start_time"].dt.time == time.fromisoformat(execution["rth_start"])].copy()
    daily = _daily_lookup(data)
    plan = SignalPlan()
    for _, row in first_bars.iterrows():
        day = row["session_date"]
        daily_row = daily.get(day)
        if daily_row is None or pd.isna(daily_row["prior_close"]) or pd.isna(daily_row["prior_atr14"]) or pd.isna(row["prior_volume_mean20"]):
            continue
        gap = float(daily_row["open"] - daily_row["prior_close"])
        if abs(gap) <= gap_multiple * float(daily_row["prior_atr14"]) or row["volume"] <= volume_multiple * row["prior_volume_mean20"]:
            continue
        direction = 1 if gap > 0 else -1
        agrees = row["close"] > row["open"] if direction == 1 else row["close"] < row["open"]
        if not agrees:
            continue
        stop = (float(daily_row["open"]) + float(daily_row["prior_close"])) / 2.0
        plan.entries.append(_market_entry(row, direction, stop=stop, target_offset=2.0 * abs(gap), reason="gap_and_go_confirmation"))
    return plan


def _build_range_reversion(spec: dict[str, Any], data: PreparedData) -> SignalPlan:
    params = spec["canonical_parameters"]["structural"]
    lookback = int(params["range_lookback"])
    quartile = float(params["entry_quartile_fraction"])
    bars = data.bars[15].copy()
    bars["range_high"] = bars["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    bars["range_low"] = bars["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    width = bars["range_high"] - bars["range_low"]
    bars["lower_quartile"] = bars["range_low"] + quartile * width
    bars["upper_quartile"] = bars["range_low"] + (1.0 - quartile) * width
    bars["midpoint"] = bars["range_low"] + 0.5 * width
    bars["rsi"] = _rsi(bars["close"], 14)
    tick = float(spec["canonical_parameters"]["execution"]["tick_size"])
    plan = SignalPlan()
    for index in range(1, len(bars)):
        row = bars.iloc[index]
        prior = bars.iloc[index - 1]
        if any(pd.isna(row[name]) for name in ("range_high", "range_low", "lower_quartile", "upper_quartile", "midpoint", "rsi")):
            continue
        if row["range_low"] <= row["close"] <= row["lower_quartile"] and prior["close"] > prior["lower_quartile"] and row["rsi"] <= 30.0:
            plan.entries.append(_market_entry(row, 1, stop=row["range_low"] - tick, target=row["midpoint"], reason="lower_quartile_reversion"))
        elif row["upper_quartile"] <= row["close"] <= row["range_high"] and prior["close"] < prior["upper_quartile"] and row["rsi"] >= 70.0:
            plan.entries.append(_market_entry(row, -1, stop=row["range_high"] + tick, target=row["midpoint"], reason="upper_quartile_reversion"))
    return plan


def _build_ma_crossover(spec: dict[str, Any], data: PreparedData) -> SignalPlan:
    params = spec["canonical_parameters"]["structural"]
    fast_period = int(params["fast_ema_period"])
    slow_period = int(params["slow_ema_period"])
    adx_period = int(params["adx_period"])
    adx_threshold = float(params["adx_entry_threshold"])
    bars = data.bars[15].copy()
    bars["fast"] = _ema(bars["close"], fast_period)
    bars["slow"] = _ema(bars["close"], slow_period)
    _, _, bars["adx"] = _dmi_adx(bars, adx_period)
    plan = SignalPlan()
    for index in range(1, len(bars)):
        row = bars.iloc[index]
        prior = bars.iloc[index - 1]
        if any(pd.isna(row[name]) for name in ("fast", "slow")) or pd.isna(prior["fast"]) or pd.isna(prior["slow"]):
            continue
        cross_up = prior["fast"] <= prior["slow"] and row["fast"] > row["slow"]
        cross_down = prior["fast"] >= prior["slow"] and row["fast"] < row["slow"]
        if cross_up:
            plan.exits.append(_exit(row, -1, "opposing_ema_crossover"))
            if pd.notna(row["adx"]) and row["adx"] > adx_threshold:
                plan.entries.append(_market_entry(row, 1, reason="ema_cross_adx", require_between=False))
        elif cross_down:
            plan.exits.append(_exit(row, 1, "opposing_ema_crossover"))
            if pd.notna(row["adx"]) and row["adx"] > adx_threshold:
                plan.entries.append(_market_entry(row, -1, reason="ema_cross_adx", require_between=False))
    return plan


def _build_keltner(spec: dict[str, Any], data: PreparedData) -> SignalPlan:
    params = spec["canonical_parameters"]["structural"]
    ema_period = int(params["ema_period"])
    atr_period = int(params["atr_period"])
    multiplier = float(params["keltner_atr_multiplier"])
    rsi_period = int(params["rsi_period"])
    long_threshold = float(params["long_rsi_threshold"])
    short_threshold = float(params["short_rsi_threshold"])
    bars = data.bars[15].copy()
    bars["ema"] = _ema(bars["close"], ema_period)
    bars["atr"] = _atr(bars, atr_period)
    bars["rsi"] = _rsi(bars["close"], rsi_period)
    plan = SignalPlan()
    for _, row in bars.iterrows():
        if any(pd.isna(row[name]) for name in ("ema", "atr", "rsi")):
            continue
        lower = row["ema"] - multiplier * row["atr"]
        upper = row["ema"] + multiplier * row["atr"]
        if row["low"] <= lower and row["rsi"] < long_threshold:
            plan.entries.append(_market_entry(row, 1, stop=lower - row["atr"], target=row["ema"], reason="keltner_lower_extreme"))
        elif row["high"] >= upper and row["rsi"] > short_threshold:
            plan.entries.append(_market_entry(row, -1, stop=upper + row["atr"], target=row["ema"], reason="keltner_upper_extreme"))
    return plan


SIGNAL_BUILDERS: dict[str, Callable[[dict[str, Any], PreparedData], SignalPlan]] = {
    "trend_pullback_ma_structure": _build_trend_pullback,
    "vwap_equilibrium_reversion": _build_vwap_reversion,
    "rolling_range_breakout": _build_donchian,
    "atr_trend_flip": _build_supertrend,
    "session_range_breakout": _build_opening_range,
    "compression_breakout": _build_compression,
    "statistical_zscore_reversion": _build_zscore,
    "regression_equilibrium_reversion": _build_regression,
    "volatility_breakout": _build_volatility_breakout,
    "short_horizon_pullback_reversion": _build_rsi2,
    "momentum_confirmed_breakout": _build_momentum_breakout,
    "gap_fade": _build_gap_fade,
    "gap_continuation": _build_gap_continuation,
    "range_structure_reversion": _build_range_reversion,
    "ma_crossover_trend": _build_ma_crossover,
    "channel_extreme_reversion": _build_keltner,
}


def _round_tick(price: float, tick: float, direction: int, is_entry: bool) -> float:
    scaled = price / tick
    if (is_entry and direction == 1) or (not is_entry and direction == -1):
        return math.ceil(scaled - 1e-10) * tick
    return math.floor(scaled + 1e-10) * tick


def _apply_slippage(raw_price: float, tick: float, ticks: int, direction: int, is_entry: bool) -> float:
    adverse = direction if is_entry else -direction
    return raw_price + adverse * ticks * tick


def _effective_cutoff(session: pd.Series, strategy_cutoff: time) -> pd.Timestamp:
    requested = _timestamp(session["session_date"], strategy_cutoff)
    return min(requested, pd.Timestamp(session["calendar_last_entry"]))


def _instruction_maps(plan: SignalPlan) -> tuple[dict[pd.Timestamp, list[EntryInstruction]], dict[pd.Timestamp, list[ExitInstruction]], dict[pd.Timestamp, list[StopUpdate]]]:
    entries: dict[pd.Timestamp, list[EntryInstruction]] = {}
    exits: dict[pd.Timestamp, list[ExitInstruction]] = {}
    updates: dict[pd.Timestamp, list[StopUpdate]] = {}
    for item in plan.entries:
        entries.setdefault(item.activation_time, []).append(item)
    for item in plan.exits:
        exits.setdefault(item.timestamp, []).append(item)
    for item in plan.stop_updates:
        updates.setdefault(item.timestamp, []).append(item)
    return entries, exits, updates


def _open_position(
    instruction: EntryInstruction,
    direction: int,
    raw_entry: float,
    timestamp: pd.Timestamp,
    tick: float,
    slippage_ticks: int,
    point_value: float,
    stop: float | None,
    target: float | None,
    checks: dict[str, Any],
) -> dict[str, Any] | None:
    raw_entry = _round_tick(float(raw_entry), tick, direction, True)
    entry_price = _apply_slippage(raw_entry, tick, slippage_ticks, direction, True)
    rounded_stop = None if stop is None else _round_tick(float(stop), tick, direction, False)
    if instruction.stop_offset_points is not None:
        rounded_stop = _round_tick(entry_price - direction * float(instruction.stop_offset_points), tick, direction, False)
    rounded_target = None if target is None else _round_tick(float(target), tick, direction, False)
    if instruction.target_offset_points is not None:
        rounded_target = _round_tick(entry_price + direction * float(instruction.target_offset_points), tick, direction, False)
    valid = True
    if rounded_stop is not None:
        valid &= rounded_stop < entry_price if direction == 1 else rounded_stop > entry_price
    if rounded_target is not None:
        valid &= rounded_target > entry_price if direction == 1 else rounded_target < entry_price
    if instruction.require_between and not valid:
        checks["invalid_entry_geometry_skips"] += 1
        return None
    return {
        "session_date": instruction.session_date,
        "direction_int": direction,
        "entry_time": timestamp,
        "signal_time": instruction.signal_time,
        "raw_entry_price": raw_entry,
        "entry_price": entry_price,
        "stop_price": rounded_stop,
        "target_price": rounded_target,
        "entry_reason": instruction.reason,
        "setup_id": instruction.setup_id,
        "max_high": entry_price,
        "min_low": entry_price,
        "highest_completed_high": -np.inf,
        "lowest_completed_low": np.inf,
        "entry_slippage_usd": abs(entry_price - raw_entry) * point_value,
    }


def _close_position(
    position: dict[str, Any],
    raw_exit: float,
    timestamp: pd.Timestamp,
    reason: str,
    tick: float,
    slippage_ticks: int,
    point_value: float,
    commission_per_side: float,
    trade_id: int,
) -> dict[str, Any]:
    direction = int(position["direction_int"])
    raw_exit = _round_tick(float(raw_exit), tick, direction, False)
    exit_price = _apply_slippage(raw_exit, tick, slippage_ticks, direction, False)
    gross = (raw_exit - float(position["raw_entry_price"])) * direction * point_value
    slippage = float(position["entry_slippage_usd"]) + abs(exit_price - raw_exit) * point_value
    commissions = 2.0 * commission_per_side
    net = gross - slippage - commissions
    holding_minutes = (timestamp - pd.Timestamp(position["entry_time"])).total_seconds() / 60.0
    if direction == 1:
        mae = max(0.0, float(position["entry_price"]) - float(position["min_low"])) * point_value
        mfe = max(0.0, float(position["max_high"]) - float(position["entry_price"])) * point_value
    else:
        mae = max(0.0, float(position["max_high"]) - float(position["entry_price"])) * point_value
        mfe = max(0.0, float(position["entry_price"]) - float(position["min_low"])) * point_value
    return {
        "trade_id": trade_id,
        "session_date": position["session_date"],
        "direction": "long" if direction == 1 else "short",
        "position": direction,
        "quantity": 1,
        "signal_time": position["signal_time"],
        "entry_time": position["entry_time"],
        "exit_time": timestamp,
        "entry_reason": position["entry_reason"],
        "exit_reason": reason,
        "setup_id": position["setup_id"],
        "raw_entry_price": position["raw_entry_price"],
        "entry_price": position["entry_price"],
        "initial_stop_price": position.get("initial_stop_price", position["stop_price"]),
        "final_stop_price": position["stop_price"],
        "target_price": position["target_price"],
        "raw_exit_price": raw_exit,
        "exit_price": exit_price,
        "gross_pnl_usd": gross,
        "commissions_usd": commissions,
        "slippage_usd": slippage,
        "net_pnl_usd": net,
        "holding_minutes": holding_minutes,
        "mae_usd": mae,
        "mfe_usd": mfe,
    }


def _intrabar_exit(position: dict[str, Any], bar_open: float, bar_high: float, bar_low: float) -> tuple[float, str] | None:
    direction = int(position["direction_int"])
    stop = position.get("stop_price")
    target = position.get("target_price")
    if direction == 1:
        if stop is not None and bar_low <= stop:
            return min(float(stop), bar_open), "stop"
        if target is not None and bar_high >= target:
            return float(target), "target"
    else:
        if stop is not None and bar_high >= stop:
            return max(float(stop), bar_open), "stop"
        if target is not None and bar_low <= target:
            return float(target), "target"
    return None


def _resting_trigger(order: EntryInstruction, bar_open: float, bar_high: float, bar_low: float, tick: float) -> tuple[int, float, float | None, float | None] | str | None:
    long_level = None if order.long_level is None else _round_tick(order.long_level, tick, 1, True)
    short_level = None if order.short_level is None else _round_tick(order.short_level, tick, -1, True)
    long_hit = long_level is not None and bar_high >= long_level
    short_hit = short_level is not None and bar_low <= short_level
    if long_hit and short_hit:
        return "both"
    if long_hit:
        return 1, max(bar_open, float(long_level)), order.long_stop, order.long_target
    if short_hit:
        return -1, min(bar_open, float(short_level)), order.short_stop, order.short_target
    return None


def _apply_stop_updates(position: dict[str, Any], updates: list[StopUpdate], tick: float) -> None:
    direction = int(position["direction_int"])
    for update in updates:
        if update.direction != direction:
            continue
        candidate: float | None = None
        if update.mode == "monotonic" and update.value is not None:
            candidate = update.value
        elif update.mode == "momentum_atr_trail" and update.atr is not None:
            position["highest_completed_high"] = max(float(position["highest_completed_high"]), float(update.bar_high))
            position["lowest_completed_low"] = min(float(position["lowest_completed_low"]), float(update.bar_low))
            candidate = (
                float(position["highest_completed_high"]) - 2.0 * float(update.atr)
                if direction == 1
                else float(position["lowest_completed_low"]) + 2.0 * float(update.atr)
            )
        if candidate is None:
            continue
        candidate = _round_tick(candidate, tick, direction, False)
        current = position.get("stop_price")
        if current is None:
            position["stop_price"] = candidate
        elif direction == 1:
            position["stop_price"] = max(float(current), candidate)
        else:
            position["stop_price"] = min(float(current), candidate)


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "trade_id", "session_date", "direction", "position", "quantity", "signal_time", "entry_time", "exit_time",
            "entry_reason", "exit_reason", "setup_id", "raw_entry_price", "entry_price", "initial_stop_price",
            "final_stop_price", "target_price", "raw_exit_price", "exit_price", "gross_pnl_usd", "commissions_usd",
            "slippage_usd", "net_pnl_usd", "holding_minutes", "mae_usd", "mfe_usd",
        ]
    )


def slice_prepared_data(data: PreparedData, start_date: date, end_date: date) -> PreparedData:
    """Restrict accounting and execution data while retaining prebuilt indicator bars."""
    in_window = lambda frame: frame[frame["session_date"].between(start_date, end_date)].copy()
    return PreparedData(
        minute=in_window(data.minute),
        sessions=in_window(data.sessions),
        bars=data.bars,
        daily=in_window(data.daily),
        benchmark_daily=in_window(data.benchmark_daily),
        source_path=data.source_path,
        source_sha256=data.source_sha256,
        data_warnings=list(data.data_warnings),
    )


def run_baseline_backtest(
    spec: dict[str, Any],
    data: PreparedData,
    evaluation_start: date | None = None,
    evaluation_end: date | None = None,
) -> BacktestResult:
    """Run one frozen mechanism through the common minute execution engine."""
    mechanism_id = spec["mechanism_id"]
    if mechanism_id not in SIGNAL_BUILDERS:
        raise ValueError(f"No signal builder for {mechanism_id}")
    plan = SIGNAL_BUILDERS[mechanism_id](spec, data)
    if evaluation_start is not None or evaluation_end is not None:
        if evaluation_start is None or evaluation_end is None:
            raise ValueError("evaluation_start and evaluation_end must be provided together")
        if evaluation_start > evaluation_end:
            raise ValueError("evaluation_start must not be after evaluation_end")
        inside = lambda day: evaluation_start <= day <= evaluation_end
        plan = SignalPlan(
            entries=[item for item in plan.entries if inside(item.session_date)],
            exits=[item for item in plan.exits if inside(item.session_date)],
            stop_updates=[item for item in plan.stop_updates if inside(item.session_date)],
            time_exits={day: value for day, value in plan.time_exits.items() if inside(day)},
        )
        data = slice_prepared_data(data, evaluation_start, evaluation_end)
    entry_map, exit_map, update_map = _instruction_maps(plan)
    execution = spec["canonical_parameters"]["execution"]
    tick = float(execution["tick_size"])
    point_value = float(execution["point_value_usd"])
    slippage_ticks = int(execution["slippage_ticks_per_side"])
    commission = float(execution["commission_per_side_usd"])
    strategy_cutoff = time.fromisoformat(execution["last_entry_time"])

    checks: dict[str, Any] = {
        "spec_hash_match": True,
        "lookahead_violations": 0,
        "future_high_low_violations": 0,
        "same_bar_policy_violations": 0,
        "double_entry_violations": 0,
        "overnight_position_violations": 0,
        "entry_after_last_entry_violations": 0,
        "post_hard_flatten_violations": 0,
        "missing_execution_bar_skips": 0,
        "invalid_entry_geometry_skips": 0,
        "same_signal_indicator_stop_entry_skips": 0,
        "dual_oco_cancellations": 0,
        "multi_order_same_direction_priority_resolutions": 0,
        "ambiguous_multi_order_cancellations": 0,
        "hard_flatten_fallbacks": 0,
    }
    trades: list[dict[str, Any]] = []
    trade_id = 1
    # Landscape studies reuse the same immutable PreparedData for hundreds of
    # cells. Building this large dictionary for every cell dominated runtime.
    by_date = getattr(data, "_minute_by_date_cache", None)
    if by_date is None:
        by_date = {day: frame for day, frame in data.minute.groupby("session_date", sort=True)}
        setattr(data, "_minute_by_date_cache", by_date)
    active_session_dates = {item.session_date for item in plan.entries}
    entries_by_date: dict[date, list[EntryInstruction]] = {}
    for item in plan.entries:
        entries_by_date.setdefault(item.session_date, []).append(item)
    cutoff_by_date = {
        row["session_date"]: _effective_cutoff(row, strategy_cutoff)
        for _, row in data.sessions.iterrows()
    }
    hard_flatten_by_date = {
        row["session_date"]: pd.Timestamp(row["hard_flatten"])
        for _, row in data.sessions.iterrows()
    }

    for _, session in data.sessions.iterrows():
        day = session["session_date"]
        # Positions never carry overnight.  A session without any possible
        # entry instruction therefore cannot change trades or equity.
        if day not in active_session_dates:
            continue
        session_frame = by_date.get(day)
        if session_frame is None or session_frame.empty:
            continue
        session_frame = session_frame.sort_index()
        cutoff = cutoff_by_date[day]
        hard_flatten = hard_flatten_by_date[day]
        position: dict[str, Any] | None = None
        pending: list[EntryInstruction] = []
        exact_timestamps = set(session_frame.index)

        expected_entries = [item for item in entries_by_date.get(day, []) if item.order_type == "market_open"]
        checks["missing_execution_bar_skips"] += sum(item.activation_time not in exact_timestamps for item in expected_entries)

        for minute_row in session_frame.itertuples(index=True, name="Minute"):
            timestamp = pd.Timestamp(minute_row.Index)
            if timestamp > hard_flatten:
                break
            bar_open = float(minute_row.open)
            bar_high = float(minute_row.high)
            bar_low = float(minute_row.low)
            exited_at_open = False

            if position is not None and timestamp == hard_flatten:
                record = _close_position(position, bar_open, timestamp, "hard_flatten", tick, slippage_ticks, point_value, commission, trade_id)
                trades.append(record)
                trade_id += 1
                position = None
                exited_at_open = True
                pending.clear()

            if position is not None:
                time_exit = plan.time_exits.get(day)
                if time_exit is not None and timestamp.time() == time_exit[0]:
                    record = _close_position(position, bar_open, timestamp, time_exit[1], tick, slippage_ticks, point_value, commission, trade_id)
                    trades.append(record)
                    trade_id += 1
                    position = None
                    exited_at_open = True
                    pending.clear()

            if position is not None:
                matching_exits = [item for item in exit_map.get(timestamp, []) if item.direction == position["direction_int"]]
                if matching_exits:
                    priority = sorted(matching_exits, key=lambda item: ("stop" not in item.reason, item.reason))[0]
                    record = _close_position(position, bar_open, timestamp, priority.reason, tick, slippage_ticks, point_value, commission, trade_id)
                    trades.append(record)
                    trade_id += 1
                    position = None
                    exited_at_open = True
                    pending.clear()

            if position is not None:
                _apply_stop_updates(position, update_map.get(timestamp, []), tick)

            activations = entry_map.get(timestamp, [])
            if position is None:
                pending.extend([item for item in activations if item.order_type == "stop_oco"])
            pending = [item for item in pending if timestamp <= cutoff]

            if position is None and timestamp <= cutoff:
                market_entries = [item for item in activations if item.order_type == "market_open"]
                stop_directions = {
                    item.direction
                    for item in exit_map.get(timestamp, [])
                    if "stop" in item.reason
                }
                blocked_entries = [item for item in market_entries if item.direction in stop_directions]
                if blocked_entries:
                    checks["same_signal_indicator_stop_entry_skips"] += len(blocked_entries)
                    market_entries = [item for item in market_entries if item.direction not in stop_directions]
                if len(market_entries) > 1:
                    directions = {item.direction for item in market_entries}
                    if len(directions) > 1:
                        checks["ambiguous_multi_order_cancellations"] += 1
                        market_entries = []
                    else:
                        market_entries = market_entries[:1]
                if market_entries:
                    instruction = market_entries[0]
                    if instruction.signal_time > timestamp:
                        checks["lookahead_violations"] += 1
                    position = _open_position(
                        instruction, instruction.direction, bar_open, timestamp, tick, slippage_ticks,
                        point_value, instruction.stop_price, instruction.target_price, checks,
                    )
                    if position is not None:
                        position["initial_stop_price"] = position["stop_price"]
                        pending.clear()

            opened_this_bar = position is not None and position["entry_time"] == timestamp
            if position is not None:
                position["max_high"] = max(float(position["max_high"]), bar_high)
                position["min_low"] = min(float(position["min_low"]), bar_low)
                hit = _intrabar_exit(position, bar_open, bar_high, bar_low)
                if hit is not None:
                    raw_exit, reason = hit
                    record = _close_position(position, raw_exit, timestamp, reason, tick, slippage_ticks, point_value, commission, trade_id)
                    trades.append(record)
                    trade_id += 1
                    position = None
                    pending.clear()

            if position is None and not opened_this_bar and not exited_at_open and timestamp <= cutoff and pending:
                triggers: list[tuple[EntryInstruction, tuple[int, float, float | None, float | None]]] = []
                remaining: list[EntryInstruction] = []
                for order in pending:
                    outcome = _resting_trigger(order, bar_open, bar_high, bar_low, tick)
                    if outcome == "both":
                        checks["dual_oco_cancellations"] += 1
                        continue
                    if outcome is None:
                        remaining.append(order)
                    else:
                        triggers.append((order, outcome))
                pending = remaining
                if len(triggers) > 1:
                    trigger_directions = {trigger[1][0] for trigger in triggers}
                    if len(trigger_directions) == 1:
                        # Pending orders are kept in activation order.  With a
                        # one-position policy the oldest eligible order fills
                        # first and all younger orders are cancelled.
                        triggers = triggers[:1]
                        checks["multi_order_same_direction_priority_resolutions"] += 1
                    else:
                        checks["ambiguous_multi_order_cancellations"] += 1
                        triggers = []
                        pending.clear()
                if len(triggers) == 1:
                    instruction, (direction, raw_entry, stop, target) = triggers[0]
                    if instruction.signal_time > timestamp:
                        checks["lookahead_violations"] += 1
                    position = _open_position(instruction, direction, raw_entry, timestamp, tick, slippage_ticks, point_value, stop, target, checks)
                    pending.clear()
                    if position is not None:
                        position["initial_stop_price"] = position["stop_price"]
                        position["max_high"] = max(float(position["max_high"]), bar_high)
                        position["min_low"] = min(float(position["min_low"]), bar_low)
                        hit = _intrabar_exit(position, bar_open, bar_high, bar_low)
                        if hit is not None:
                            raw_exit, reason = hit
                            record = _close_position(position, raw_exit, timestamp, reason, tick, slippage_ticks, point_value, commission, trade_id)
                            trades.append(record)
                            trade_id += 1
                            position = None

        if position is not None:
            pre_flatten = session_frame[session_frame.index < hard_flatten]
            if pre_flatten.empty:
                checks["post_hard_flatten_violations"] += 1
            else:
                last_timestamp = pd.Timestamp(pre_flatten.index[-1])
                record = _close_position(position, float(pre_flatten.iloc[-1]["close"]), last_timestamp, "hard_flatten_missing_bar_fallback", tick, slippage_ticks, point_value, commission, trade_id)
                trades.append(record)
                trade_id += 1
                checks["hard_flatten_fallbacks"] += 1
                position = None

    trade_frame = pd.DataFrame(trades) if trades else _empty_trades()
    if not trade_frame.empty:
        trade_frame.insert(0, "mechanism_id", mechanism_id)
        trade_frame.insert(1, "strategy_id", spec["strategy_id"])
        for column in ("signal_time", "entry_time", "exit_time"):
            trade_frame[column] = pd.to_datetime(trade_frame[column])
        checks["entry_after_last_entry_violations"] = int(sum(
            pd.Timestamp(row.entry_time) > cutoff_by_date[row.session_date]
            for row in trade_frame.itertuples(index=False)
        ))
        checks["overnight_position_violations"] = int(sum(pd.Timestamp(row.entry_time).date() != pd.Timestamp(row.exit_time).date() for row in trade_frame.itertuples(index=False)))
        checks["post_hard_flatten_violations"] += int(sum(
            pd.Timestamp(row.exit_time) > hard_flatten_by_date[row.session_date]
            for row in trade_frame.itertuples(index=False)
        ))
        ordered = trade_frame.sort_values("entry_time")
        checks["double_entry_violations"] = int(sum(
            pd.Timestamp(ordered.iloc[i]["entry_time"]) < pd.Timestamp(ordered.iloc[i - 1]["exit_time"])
            for i in range(1, len(ordered))
        ))
    else:
        trade_frame.insert(0, "mechanism_id", pd.Series(dtype=str))
        trade_frame.insert(1, "strategy_id", pd.Series(dtype=str))

    # Preserve PnL from active sessions whose exact cash-close bar is absent.
    # The frozen strategy specs permit a last-pre-cutoff flatten fallback, while
    # the benchmark spec rejects a session when its required close is missing.
    session_dates = sorted(
        set(data.benchmark_daily["session_date"].tolist())
        | set(data.sessions["session_date"].tolist())
    )
    pnl_by_day = trade_frame.groupby("session_date")["net_pnl_usd"].sum().to_dict() if not trade_frame.empty else {}
    daily = pd.DataFrame({"session_date": session_dates})
    daily["mechanism_id"] = mechanism_id
    daily["daily_pnl_usd"] = daily["session_date"].map(pnl_by_day).fillna(0.0)
    daily["equity_usd"] = INITIAL_CAPITAL_USD + daily["daily_pnl_usd"].cumsum()
    prior_equity = daily["equity_usd"].shift(1).fillna(INITIAL_CAPITAL_USD)
    daily["daily_return"] = daily["daily_pnl_usd"] / prior_equity
    daily["trade_count"] = daily["session_date"].map(trade_frame.groupby("session_date").size().to_dict() if not trade_frame.empty else {}).fillna(0).astype(int)
    equity = daily[["session_date", "mechanism_id", "daily_pnl_usd", "daily_return", "equity_usd"]].copy()
    positions = trade_frame[["mechanism_id", "trade_id", "session_date", "position", "quantity", "entry_time", "exit_time"]].copy() if not trade_frame.empty else pd.DataFrame(columns=["mechanism_id", "trade_id", "session_date", "position", "quantity", "entry_time", "exit_time"])
    return BacktestResult(trades=trade_frame, daily=daily, equity=equity, positions=positions, checks=checks)


def build_buy_and_hold(benchmark_spec: dict[str, Any], data: PreparedData) -> BacktestResult:
    daily = data.benchmark_daily[["session_date", "close", "close_timestamp"]].copy()
    first_close = float(daily["close"].iloc[0])
    daily["mechanism_id"] = "MNQ_BUY_AND_HOLD"
    daily["equity_usd"] = INITIAL_CAPITAL_USD * daily["close"] / first_close
    daily["daily_pnl_usd"] = daily["equity_usd"].diff().fillna(0.0)
    daily["daily_return"] = daily["close"].pct_change().fillna(0.0)
    daily["trade_count"] = 0
    equity = daily[["session_date", "mechanism_id", "daily_pnl_usd", "daily_return", "equity_usd"]].copy()
    return BacktestResult(
        trades=_empty_trades(),
        daily=daily,
        equity=equity,
        positions=pd.DataFrame(columns=["mechanism_id", "trade_id", "session_date", "position", "quantity", "entry_time", "exit_time"]),
        checks={
            "spec_hash_match": True,
            "daily_flatten_applied": False,
            "topstep_rules_applied": False,
            "continuous_close_to_close": True,
        },
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator not in (0.0, -0.0) and np.isfinite(denominator) else np.nan


def compute_metrics(spec: dict[str, Any], result: BacktestResult, benchmark_daily: pd.DataFrame) -> dict[str, Any]:
    trades = result.trades
    daily = result.daily
    returns = daily["daily_return"].astype(float)
    pnl = daily["daily_pnl_usd"].astype(float)
    equity = daily["equity_usd"].astype(float)
    peak = pd.concat([pd.Series([INITIAL_CAPITAL_USD]), equity], ignore_index=True).cummax().iloc[1:].reset_index(drop=True)
    drawdown = equity.reset_index(drop=True) - peak
    drawdown_pct = drawdown / peak.replace(0.0, np.nan)
    return_std = float(returns.std(ddof=1))
    negative = returns[returns < 0]
    downside_std = float(negative.std(ddof=1)) if len(negative) > 1 else np.nan
    sharpe = _safe_ratio(float(returns.mean()) * math.sqrt(ANNUALIZATION_DAYS), return_std)
    sortino = _safe_ratio(float(returns.mean()) * math.sqrt(ANNUALIZATION_DAYS), downside_std)
    benchmark_aligned = benchmark_daily.set_index("session_date")["daily_return"].reindex(daily["session_date"]).fillna(0.0)
    correlation = returns.corr(benchmark_aligned.reset_index(drop=True))

    is_benchmark = spec["specification_type"] == "market_benchmark"
    if is_benchmark:
        trade_count = 0
        wins = pd.Series(dtype=float)
        losses = pd.Series(dtype=float)
    else:
        trade_count = int(len(trades))
        trade_pnl = trades["net_pnl_usd"].astype(float) if trade_count else pd.Series(dtype=float)
        wins = trade_pnl[trade_pnl > 0]
        losses = trade_pnl[trade_pnl < 0]

    positive_days = pnl[pnl > 0].sort_values(ascending=False)
    concentration = _safe_ratio(float(positive_days.head(5).sum()), float(positive_days.sum())) * 100.0 if len(positive_days) else np.nan
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    warning_parts: list[str] = []
    if not is_benchmark:
        if result.checks.get("missing_execution_bar_skips", 0):
            warning_parts.append(f"{result.checks['missing_execution_bar_skips']} signal(s) skipped: exact execution minute absent")
        if result.checks.get("invalid_entry_geometry_skips", 0):
            warning_parts.append(f"{result.checks['invalid_entry_geometry_skips']} entry/entries skipped: stop-target geometry invalid at fill")
        if result.checks.get("same_signal_indicator_stop_entry_skips", 0):
            warning_parts.append(
                f"{result.checks['same_signal_indicator_stop_entry_skips']} entry/entries skipped because the completed signal bar already met the indicator stop"
            )
        if result.checks.get("dual_oco_cancellations", 0):
            warning_parts.append(f"{result.checks['dual_oco_cancellations']} OCO minute(s) cancelled: both sides touched")
        if result.checks.get("multi_order_same_direction_priority_resolutions", 0):
            warning_parts.append(
                f"{result.checks['multi_order_same_direction_priority_resolutions']} minute(s) with concurrent same-direction orders resolved by oldest-order priority"
            )
        if result.checks.get("ambiguous_multi_order_cancellations", 0):
            warning_parts.append(f"{result.checks['ambiguous_multi_order_cancellations']} minute(s) cancelled: multiple orders ambiguous")
        if result.checks.get("hard_flatten_fallbacks", 0):
            warning_parts.append(f"{result.checks['hard_flatten_fallbacks']} hard flatten(s) used last pre-cutoff close")
    violation_keys = [
        "lookahead_violations", "future_high_low_violations", "same_bar_policy_violations", "double_entry_violations",
        "overnight_position_violations", "entry_after_last_entry_violations", "post_hard_flatten_violations",
        "ambiguous_multi_order_cancellations",
    ]
    valid = all(int(result.checks.get(key, 0)) == 0 for key in violation_keys) and bool(result.checks.get("spec_hash_match", False))
    if not valid:
        warning_parts.append("implementation consistency check failed")

    benchmark_net = float(benchmark_daily["equity_usd"].iloc[-1] - INITIAL_CAPITAL_USD)
    net_pnl = float(equity.iloc[-1] - INITIAL_CAPITAL_USD)
    return {
        "mechanism_id": spec["mechanism_id"],
        "strategy_id": spec["strategy_id"],
        "specification_type": spec["specification_type"],
        "research_family": spec["research_family"],
        "complexity_grade": spec["complexity_grade"],
        "frozen_spec_sha256": spec["_frozen_spec_sha256"],
        "start_date": daily["session_date"].iloc[0],
        "end_date": daily["session_date"].iloc[-1],
        "session_count": int(len(daily)),
        "trade_count": trade_count,
        "net_pnl_usd": net_pnl,
        "expectancy_per_trade_usd": float(trades["net_pnl_usd"].mean()) if trade_count else np.nan,
        "win_rate_pct": float((trades["net_pnl_usd"] > 0).mean() * 100.0) if trade_count else np.nan,
        "average_win_usd": float(wins.mean()) if len(wins) else np.nan,
        "average_loss_usd": float(losses.mean()) if len(losses) else np.nan,
        "payoff_ratio": _safe_ratio(float(wins.mean()) if len(wins) else np.nan, abs(float(losses.mean())) if len(losses) else np.nan),
        "profit_factor": _safe_ratio(gross_profit, gross_loss),
        "sharpe_annualized": sharpe,
        "sortino_annualized": sortino,
        "max_drawdown_usd": float(drawdown.min()),
        "max_drawdown_pct": float(drawdown_pct.min() * 100.0),
        "worst_trade_usd": float(trades["net_pnl_usd"].min()) if trade_count else np.nan,
        "worst_day_usd": float(pnl.min()),
        "best_day_usd": float(pnl.max()),
        "pnl_concentration_top_5_positive_days_pct": concentration,
        "average_holding_minutes": float(trades["holding_minutes"].mean()) if trade_count else np.nan,
        "average_mae_usd": float(trades["mae_usd"].mean()) if trade_count else np.nan,
        "average_mfe_usd": float(trades["mfe_usd"].mean()) if trade_count else np.nan,
        "commissions_usd": float(trades["commissions_usd"].sum()) if trade_count else 0.0,
        "estimated_slippage_usd": float(trades["slippage_usd"].sum()) if trade_count else 0.0,
        "benchmark_net_pnl_usd": benchmark_net,
        "net_pnl_minus_benchmark_usd": net_pnl - benchmark_net,
        "daily_return_correlation_to_benchmark": float(correlation) if pd.notna(correlation) else np.nan,
        "implementation_warning": "; ".join(warning_parts),
        "backtest_valid": valid,
    }


def serialize_checks(mechanism_id: str, checks: dict[str, Any]) -> dict[str, Any]:
    return {"mechanism_id": mechanism_id, **checks}
