"""Intraday Momentum Pullback Continuation (IMPC) signal construction."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.config.settings import DEFAULT_INITIAL_CAPITAL_USD
from src.data.session import add_session_date
from src.features.intraday import add_session_vwap
from src.features.volatility import add_atr


DEFAULT_SESSION_START = "09:30:00"
DEFAULT_SESSION_END = "16:00:00"
DEFAULT_ENTRY_START = "09:45:00"
DEFAULT_ENTRY_END = "15:30:00"
DEFAULT_BAR_MINUTES = 5
DEFAULT_ATR_WINDOW = 48
DEFAULT_PB_MAX_ATR = 1.20
DEFAULT_STOP_BUFFER_ATR = 0.10
DEFAULT_STRUCTURE_TOLERANCE_ATR = 0.10
DEFAULT_TIME_STOP_BARS = 12
DEFAULT_EMA_PAIRS = ((8, 21), (12, 34))
DEFAULT_SLOPE_PULLBACK_PAIRS = ((3, 3), (5, 5))
DEFAULT_PB_MIN_ATR_VALUES = (0.3, 0.5)
DEFAULT_TARGET_R_VALUES = (1.5, 2.0, 2.5)


@dataclass(frozen=True)
class IMPCVariantConfig:
    """Disciplined IMPC V1 variant definition."""

    name: str
    ema_fast: int
    ema_slow: int
    slope_lookback: int
    pullback_lookback: int
    pb_min_atr: float
    target_r: float
    atr_window: int = DEFAULT_ATR_WINDOW
    pb_max_atr: float = DEFAULT_PB_MAX_ATR
    stop_buffer_atr: float = DEFAULT_STOP_BUFFER_ATR
    structure_tolerance_atr: float = DEFAULT_STRUCTURE_TOLERANCE_ATR
    time_stop_bars: int = DEFAULT_TIME_STOP_BARS
    fixed_quantity: int = 1
    initial_capital_usd: float = DEFAULT_INITIAL_CAPITAL_USD
    session_start: str = DEFAULT_SESSION_START
    session_end: str = DEFAULT_SESSION_END
    entry_start: str = DEFAULT_ENTRY_START
    entry_end: str = DEFAULT_ENTRY_END


def build_default_impc_variants() -> list[IMPCVariantConfig]:
    """Return the disciplined 24-variant IMPC V1 grid."""
    variants: list[IMPCVariantConfig] = []
    for ema_fast, ema_slow in DEFAULT_EMA_PAIRS:
        for slope_lookback, pullback_lookback in DEFAULT_SLOPE_PULLBACK_PAIRS:
            for pb_min_atr in DEFAULT_PB_MIN_ATR_VALUES:
                for target_r in DEFAULT_TARGET_R_VALUES:
                    target_tag = str(float(target_r)).replace(".", "p")
                    pb_min_tag = str(float(pb_min_atr)).replace(".", "p")
                    variants.append(
                        IMPCVariantConfig(
                            name=(
                                f"impc_ef{int(ema_fast)}"
                                f"_es{int(ema_slow)}"
                                f"_sl{int(slope_lookback)}"
                                f"_pb{int(pullback_lookback)}"
                                f"_pm{pb_min_tag}"
                                f"_tr{target_tag}"
                            ),
                            ema_fast=int(ema_fast),
                            ema_slow=int(ema_slow),
                            slope_lookback=int(slope_lookback),
                            pullback_lookback=int(pullback_lookback),
                            pb_min_atr=float(pb_min_atr),
                            target_r=float(target_r),
                        )
                    )
    return variants


def _left_closed_time_mask(timestamps: pd.Series, start_time: str, end_time: str) -> pd.Series:
    start = dt.time.fromisoformat(start_time)
    end = dt.time.fromisoformat(end_time)
    times = pd.to_datetime(timestamps, errors="coerce").dt.time
    if start <= end:
        return (times >= start) & (times < end)
    return (times >= start) | (times < end)


def _inclusive_time_mask(timestamps: pd.Series, start_time: str, end_time: str) -> pd.Series:
    start = dt.time.fromisoformat(start_time)
    end = dt.time.fromisoformat(end_time)
    times = pd.to_datetime(timestamps, errors="coerce").dt.time
    if start <= end:
        return (times >= start) & (times <= end)
    return (times >= start) | (times <= end)


def filter_rth_bar_starts(
    df: pd.DataFrame,
    *,
    session_start: str = DEFAULT_SESSION_START,
    session_end: str = DEFAULT_SESSION_END,
) -> pd.DataFrame:
    """Filter a start-aligned frame to RTH bars in [start, end)."""
    mask = _left_closed_time_mask(df["timestamp"], session_start, session_end)
    return df.loc[mask].copy().reset_index(drop=True)


def _normalize_ints(values: Iterable[int]) -> list[int]:
    clean = sorted({int(value) for value in values if int(value) > 0})
    if not clean:
        raise ValueError("Expected at least one positive integer parameter.")
    return clean


def _session_shift_rolling(
    values: pd.Series,
    session_dates: pd.Series,
    window: int,
    reducer: str,
) -> pd.Series:
    if reducer not in {"max", "min"}:
        raise ValueError("reducer must be 'max' or 'min'.")
    grouped = values.groupby(session_dates, sort=True)
    return grouped.transform(
        lambda series: getattr(series.shift(1).rolling(int(window), min_periods=int(window)), reducer)()
    )


def _session_shift_rolling_position(
    values: pd.Series,
    session_dates: pd.Series,
    window: int,
    reducer: str,
) -> pd.Series:
    if reducer not in {"argmax", "argmin"}:
        raise ValueError("reducer must be 'argmax' or 'argmin'.")
    func = np.argmax if reducer == "argmax" else np.argmin
    return values.groupby(session_dates, sort=True).transform(
        lambda series: series.shift(1).rolling(int(window), min_periods=int(window)).apply(func, raw=True)
    )


def prepare_impc_feature_frame(
    df: pd.DataFrame,
    *,
    session_start: str = DEFAULT_SESSION_START,
    session_end: str = DEFAULT_SESSION_END,
    atr_window: int = DEFAULT_ATR_WINDOW,
    ema_fast_windows: Iterable[int] = (8, 12),
    ema_slow_windows: Iterable[int] = (21, 34),
    slope_lookbacks: Iterable[int] = (3, 5),
    pullback_lookbacks: Iterable[int] = (3, 5),
    vwap_price_mode: str = "typical",
) -> pd.DataFrame:
    """Build the leak-free 5-minute IMPC feature frame."""
    if int(atr_window) <= 0:
        raise ValueError("atr_window must be strictly positive.")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if out["timestamp"].isna().any():
        raise ValueError("Found unparsable timestamps while preparing the IMPC feature frame.")

    out = out.sort_values("timestamp").reset_index(drop=True)
    out = filter_rth_bar_starts(out, session_start=session_start, session_end=session_end)
    out = add_session_date(out)
    out = add_session_vwap(out, price_mode=vwap_price_mode)
    out["vwap_session"] = out["session_vwap"]
    out = add_atr(out, window=int(atr_window))

    atr_column = f"atr_{int(atr_window)}"
    if int(atr_window) == DEFAULT_ATR_WINDOW:
        out = out.rename(columns={atr_column: "atr_48"})
        atr_column = "atr_48"

    close = pd.to_numeric(out["close"], errors="coerce")
    out["bar_range"] = pd.to_numeric(out["high"], errors="coerce") - pd.to_numeric(out["low"], errors="coerce")
    out["bar_body"] = (close - pd.to_numeric(out["open"], errors="coerce")).abs()
    out["close_to_vwap"] = close - pd.to_numeric(out["session_vwap"], errors="coerce")

    session_dates = out["session_date"]
    group = out.groupby("session_date", sort=True)
    out["prev_high"] = group["high"].shift(1)
    out["prev_low"] = group["low"].shift(1)
    out["is_first_bar_of_session"] = group.cumcount().eq(0)
    out["is_last_bar_of_session"] = group.cumcount(ascending=False).eq(0)
    out["bar_index_in_session"] = group.cumcount()

    ema_windows = sorted({*set(_normalize_ints(ema_fast_windows)), *set(_normalize_ints(ema_slow_windows))})
    for window in ema_windows:
        out[f"ema_{int(window)}"] = close.ewm(span=int(window), adjust=False).mean()

    for fast_window in _normalize_ints(ema_fast_windows):
        fast_col = f"ema_{int(fast_window)}"
        for lookback in _normalize_ints(slope_lookbacks):
            out[f"ema_slope_{int(fast_window)}_{int(lookback)}"] = out[fast_col] - out[fast_col].shift(int(lookback))
        for slow_window in _normalize_ints(ema_slow_windows):
            out[f"ema_spread_{int(fast_window)}_{int(slow_window)}"] = out[fast_col] - out[f"ema_{int(slow_window)}"]

    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    for lookback in _normalize_ints(pullback_lookbacks):
        out[f"pullback_high_{int(lookback)}"] = _session_shift_rolling(high, session_dates, int(lookback), "max")
        out[f"pullback_low_{int(lookback)}"] = _session_shift_rolling(low, session_dates, int(lookback), "min")
        out[f"pullback_high_pos_{int(lookback)}"] = _session_shift_rolling_position(high, session_dates, int(lookback), "argmax")
        out[f"pullback_low_pos_{int(lookback)}"] = _session_shift_rolling_position(low, session_dates, int(lookback), "argmin")
        out[f"pullback_depth_{int(lookback)}"] = (
            pd.to_numeric(out[f"pullback_high_{int(lookback)}"], errors="coerce")
            - pd.to_numeric(out[f"pullback_low_{int(lookback)}"], errors="coerce")
        )

    return out.reset_index(drop=True)


def build_impc_signal_frame(
    feature_df: pd.DataFrame,
    variant: IMPCVariantConfig,
) -> pd.DataFrame:
    """Generate leak-free next-open IMPC entry signals and shifted stop metadata."""
    required = {
        "timestamp",
        "session_date",
        "open",
        "high",
        "low",
        "close",
        "session_vwap",
        "prev_high",
        "prev_low",
        "is_last_bar_of_session",
    }
    missing = sorted(required - set(feature_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for IMPC signal generation: {missing}")

    out = feature_df.copy().sort_values("timestamp").reset_index(drop=True)
    atr_col = "atr_48" if "atr_48" in out.columns else f"atr_{int(variant.atr_window)}"
    fast_col = f"ema_{int(variant.ema_fast)}"
    slow_col = f"ema_{int(variant.ema_slow)}"
    slope_col = f"ema_slope_{int(variant.ema_fast)}_{int(variant.slope_lookback)}"
    spread_col = f"ema_spread_{int(variant.ema_fast)}_{int(variant.ema_slow)}"
    pb_suffix = int(variant.pullback_lookback)
    pb_high_col = f"pullback_high_{pb_suffix}"
    pb_low_col = f"pullback_low_{pb_suffix}"
    pb_depth_col = f"pullback_depth_{pb_suffix}"
    pb_high_pos_col = f"pullback_high_pos_{pb_suffix}"
    pb_low_pos_col = f"pullback_low_pos_{pb_suffix}"
    required_variant = [
        atr_col,
        fast_col,
        slow_col,
        slope_col,
        spread_col,
        pb_high_col,
        pb_low_col,
        pb_depth_col,
        pb_high_pos_col,
        pb_low_pos_col,
    ]
    missing_variant = [column for column in required_variant if column not in out.columns]
    if missing_variant:
        raise ValueError(f"Missing IMPC feature columns for variant '{variant.name}': {missing_variant}")

    atr = pd.to_numeric(out[atr_col], errors="coerce")
    ema_fast = pd.to_numeric(out[fast_col], errors="coerce")
    ema_slow = pd.to_numeric(out[slow_col], errors="coerce")
    slope = pd.to_numeric(out[slope_col], errors="coerce")
    ema_spread = pd.to_numeric(out[spread_col], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    open_price = pd.to_numeric(out["open"], errors="coerce")
    prev_high = pd.to_numeric(out["prev_high"], errors="coerce")
    prev_low = pd.to_numeric(out["prev_low"], errors="coerce")
    session_vwap = pd.to_numeric(out["session_vwap"], errors="coerce")
    pb_high = pd.to_numeric(out[pb_high_col], errors="coerce")
    pb_low = pd.to_numeric(out[pb_low_col], errors="coerce")
    pb_depth = pd.to_numeric(out[pb_depth_col], errors="coerce")
    pb_high_pos = pd.to_numeric(out[pb_high_pos_col], errors="coerce")
    pb_low_pos = pd.to_numeric(out[pb_low_pos_col], errors="coerce")

    trend_core_long = (ema_fast > ema_slow) & (slope > 0.0)
    trend_core_short = (ema_fast < ema_slow) & (slope < 0.0)
    bias_long = trend_core_long & (close > session_vwap)
    bias_short = trend_core_short & (close < session_vwap)

    session_dates = out["session_date"]
    trend_core_long_recent = trend_core_long.astype(int).groupby(session_dates, sort=True).transform(
        lambda series: series.shift(1).rolling(pb_suffix, min_periods=1).max()
    )
    trend_core_short_recent = trend_core_short.astype(int).groupby(session_dates, sort=True).transform(
        lambda series: series.shift(1).rolling(pb_suffix, min_periods=1).max()
    )

    pb_min_points = float(variant.pb_min_atr) * atr
    pb_max_points = float(variant.pb_max_atr) * atr
    structure_tolerance = float(variant.structure_tolerance_atr) * atr
    stop_buffer_points = float(variant.stop_buffer_atr) * atr

    long_pullback_valid = (
        bias_long
        & trend_core_long_recent.fillna(0).gt(0)
        & pb_high_pos.lt(pb_low_pos)
        & pb_depth.ge(pb_min_points)
        & pb_depth.le(pb_max_points)
        & pb_low.ge(ema_slow - structure_tolerance)
    )
    short_pullback_valid = (
        bias_short
        & trend_core_short_recent.fillna(0).gt(0)
        & pb_low_pos.lt(pb_high_pos)
        & pb_depth.ge(pb_min_points)
        & pb_depth.le(pb_max_points)
        & pb_high.le(ema_slow + structure_tolerance)
    )

    raw_long = long_pullback_valid & close.gt(prev_high) & close.gt(open_price)
    raw_short = short_pullback_valid & close.lt(prev_low) & close.lt(open_price)
    raw_signal = np.where(raw_long, 1, np.where(raw_short, -1, 0))

    out["bias_long"] = bias_long
    out["bias_short"] = bias_short
    out["pullback_long_valid"] = long_pullback_valid
    out["pullback_short_valid"] = short_pullback_valid
    out["raw_signal"] = pd.Series(raw_signal, index=out.index, dtype=int)
    out["signal_atr"] = atr
    out["signal_pullback_depth"] = pb_depth
    out["signal_ema_spread"] = ema_spread
    out["stop_reference_long_signal"] = pb_low - stop_buffer_points
    out["stop_reference_short_signal"] = pb_high + stop_buffer_points
    out["target_r_signal"] = float(variant.target_r)
    out["trade_allowed"] = _inclusive_time_mask(out["timestamp"], variant.entry_start, variant.entry_end)

    group = out.groupby("session_date", sort=True)
    out["entry_signal_time"] = group["timestamp"].shift(1)
    out["entry_stop_reference_long"] = group["stop_reference_long_signal"].shift(1)
    out["entry_stop_reference_short"] = group["stop_reference_short_signal"].shift(1)
    out["entry_target_r"] = group["target_r_signal"].shift(1)
    out["entry_signal_atr"] = group["signal_atr"].shift(1)
    out["entry_signal_pullback_depth"] = group["signal_pullback_depth"].shift(1)
    out["entry_signal_ema_spread"] = group["signal_ema_spread"].shift(1)

    out["entry_long"] = (
        pd.Series(raw_long, index=out.index, dtype="boolean")
        .groupby(session_dates, sort=True)
        .shift(1)
        .fillna(False)
        .astype(bool)
        & out["trade_allowed"].fillna(False)
    )
    out["entry_short"] = (
        pd.Series(raw_short, index=out.index, dtype="boolean")
        .groupby(session_dates, sort=True)
        .shift(1)
        .fillna(False)
        .astype(bool)
        & out["trade_allowed"].fillna(False)
    )
    out["signal"] = np.where(out["entry_long"], 1, np.where(out["entry_short"], -1, 0))
    out["variant_name"] = variant.name
    return out.reset_index(drop=True)
