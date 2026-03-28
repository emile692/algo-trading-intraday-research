"""VWAP signal construction for paper and prop-style intraday variants."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.config.vwap_campaign import TimeWindow, VWAPVariantConfig
from src.data.session import add_session_date
from src.features.intraday import add_intraday_features, add_session_vwap
from src.features.volatility import add_atr


def _left_closed_time_mask(timestamps: pd.Series, start_time: str, end_time: str) -> pd.Series:
    start = dt.time.fromisoformat(start_time)
    end = dt.time.fromisoformat(end_time)
    times = timestamps.dt.time
    if start <= end:
        return (times >= start) & (times < end)
    return (times >= start) | (times < end)


def _window_mask(timestamps: pd.Series, windows: tuple[TimeWindow, ...]) -> pd.Series:
    if not windows:
        return pd.Series(True, index=timestamps.index, dtype=bool)

    mask = pd.Series(False, index=timestamps.index, dtype=bool)
    for window in windows:
        mask |= _left_closed_time_mask(timestamps, window.start, window.end)
    return mask


def _session_rolling_max(series: pd.Series, session_dates: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(session_dates)
        .rolling(window=window, min_periods=window)
        .max()
        .reset_index(level=0, drop=True)
    )


def _session_rolling_min(series: pd.Series, session_dates: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(session_dates)
        .rolling(window=window, min_periods=window)
        .min()
        .reset_index(level=0, drop=True)
    )


def _session_rolling_sum(series: pd.Series, session_dates: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(session_dates)
        .rolling(window=window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )


def filter_rth_bar_starts(
    df: pd.DataFrame,
    session_start: str = "09:30:00",
    session_end: str = "16:00:00",
) -> pd.DataFrame:
    """Filter a start-aligned intraday dataset to RTH bars in [start, end)."""
    mask = _left_closed_time_mask(df["timestamp"], session_start, session_end)
    return df.loc[mask].copy().reset_index(drop=True)


def prepare_vwap_feature_frame(
    df: pd.DataFrame,
    session_start: str = "09:30:00",
    session_end: str = "16:00:00",
    atr_windows: Iterable[int] = (14,),
    vwap_price_mode: str = "typical",
    vwap_price_volume_col: str | None = None,
) -> pd.DataFrame:
    """Build the normalized intraday feature frame used by VWAP variants."""
    out = filter_rth_bar_starts(df, session_start=session_start, session_end=session_end)
    out = add_session_date(out)
    out = add_intraday_features(out)
    out = add_session_vwap(out, price_mode=vwap_price_mode, price_volume_col=vwap_price_volume_col)
    out = add_atr(out, window=tuple(sorted({int(window) for window in atr_windows if int(window) > 0})))

    group = out.groupby("session_date", sort=True)
    out["prev_close"] = group["close"].shift(1)
    out["prev_session_vwap"] = group["session_vwap"].shift(1)
    out["prev_high"] = group["high"].shift(1)
    out["prev_low"] = group["low"].shift(1)
    out["is_first_bar_of_session"] = group.cumcount().eq(0)
    out["is_last_bar_of_session"] = group.cumcount(ascending=False).eq(0)
    out["trade_allowed"] = True
    return out


def _ensure_variant_features(df: pd.DataFrame, variant: VWAPVariantConfig) -> pd.DataFrame:
    out = df.copy()
    session_dates = out["session_date"]
    atr_col = f"atr_{variant.atr_period}"
    if atr_col not in out.columns:
        out = add_atr(out, window=variant.atr_period)

    vwap_slope_col = f"vwap_slope_{variant.slope_lookback}"
    if vwap_slope_col not in out.columns:
        out[vwap_slope_col] = out.groupby(session_dates, sort=True)["session_vwap"].diff(variant.slope_lookback)

    compression_high_col = f"compression_high_{variant.compression_length}"
    compression_low_col = f"compression_low_{variant.compression_length}"
    if compression_high_col not in out.columns:
        out[compression_high_col] = _session_rolling_max(out["high"], session_dates, variant.compression_length)
        out[compression_low_col] = _session_rolling_min(out["low"], session_dates, variant.compression_length)
        out[f"compression_width_{variant.compression_length}"] = (
            out[compression_high_col] - out[compression_low_col]
        )

    pullback_low_col = f"pullback_low_{variant.pullback_lookback}"
    pullback_high_col = f"pullback_high_{variant.pullback_lookback}"
    if pullback_low_col not in out.columns:
        out[pullback_low_col] = _session_rolling_min(out["low"], session_dates, variant.pullback_lookback)
        out[pullback_high_col] = _session_rolling_max(out["high"], session_dates, variant.pullback_lookback)

    close_delta = out.groupby(session_dates, sort=True)["close"].diff()
    down_bars = close_delta.lt(0).astype(int)
    up_bars = close_delta.gt(0).astype(int)
    out[f"down_close_count_{variant.pullback_lookback}"] = _session_rolling_sum(
        down_bars,
        session_dates,
        variant.pullback_lookback,
    )
    out[f"up_close_count_{variant.pullback_lookback}"] = _session_rolling_sum(
        up_bars,
        session_dates,
        variant.pullback_lookback,
    )
    return out


def generate_paper_baseline_signals(df: pd.DataFrame, variant: VWAPVariantConfig) -> pd.DataFrame:
    """Return the exact paper-style target position based on the previous close."""
    out = df.copy()
    session_dates = out["session_date"]
    out["trade_allowed"] = _window_mask(out["timestamp"], variant.time_windows)
    long_signal = out["prev_close"].notna() & out["prev_session_vwap"].notna() & (out["prev_close"] > out["prev_session_vwap"])
    short_signal = out["prev_close"].notna() & out["prev_session_vwap"].notna()

    long_filter = pd.Series(True, index=out.index, dtype=bool)
    short_filter = pd.Series(True, index=out.index, dtype=bool)

    if variant.require_vwap_slope_alignment:
        prev_vwap_lag = out.groupby(session_dates, sort=True)["prev_session_vwap"].shift(variant.slope_lookback)
        prev_vwap_slope = out["prev_session_vwap"] - prev_vwap_lag
        out["prev_vwap_slope_signal"] = prev_vwap_slope
        long_filter &= prev_vwap_slope > variant.slope_threshold
        short_filter &= prev_vwap_slope < -variant.slope_threshold

    if variant.max_vwap_distance_atr is not None:
        atr_col = f"atr_{variant.atr_period}"
        if atr_col not in out.columns:
            out = add_atr(out, window=variant.atr_period)
        prev_atr = out.groupby(session_dates, sort=True)[atr_col].shift(1)
        prev_distance = (out["prev_close"] - out["prev_session_vwap"]).abs()
        out["prev_atr_signal"] = prev_atr
        out["prev_vwap_distance_abs"] = prev_distance
        within_distance = prev_atr.notna() & (prev_distance <= prev_atr * float(variant.max_vwap_distance_atr))
        long_filter &= within_distance
        short_filter &= within_distance

    out["raw_target_position"] = np.select(
        [
            long_signal,
            short_signal,
        ],
        [1, -1],
        default=0,
    ).astype(int)
    out["target_position"] = np.select(
        [long_signal & long_filter, short_signal & short_filter],
        [1, -1],
        default=0,
    ).astype(int)
    out["raw_signal"] = out["raw_target_position"].astype(int)
    out["signal"] = out["target_position"].astype(int)
    return out


def _regime_masks(out: pd.DataFrame, variant: VWAPVariantConfig) -> tuple[pd.Series, pd.Series, str, str, str]:
    atr_col = f"atr_{variant.atr_period}"
    slope_col = f"vwap_slope_{variant.slope_lookback}"
    compression_high_col = f"compression_high_{variant.compression_length}"
    compression_low_col = f"compression_low_{variant.compression_length}"
    compression_width_col = f"compression_width_{variant.compression_length}"

    atr = out[atr_col]
    slope = out[slope_col]
    regime_long = out["close"].gt(out["session_vwap"]) & slope.gt(variant.slope_threshold)
    regime_short = out["close"].lt(out["session_vwap"]) & slope.lt(-variant.slope_threshold)

    out["regime_long"] = regime_long
    out["regime_short"] = regime_short
    out["compression_width"] = out[compression_width_col]
    out["compression_high_prev"] = out[compression_high_col].shift(1)
    out["compression_low_prev"] = out[compression_low_col].shift(1)
    out["atr_active"] = atr
    return regime_long, regime_short, atr_col, compression_width_col, slope_col


def _shift_close_based_entries_to_next_open(
    out: pd.DataFrame,
    session_dates: pd.Series,
    raw_entry_long: pd.Series,
    raw_entry_short: pd.Series,
    raw_stop_long: pd.Series,
    raw_stop_short: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    entry_long = raw_entry_long.astype("boolean").groupby(session_dates, sort=True).shift(1).fillna(False).astype(bool)
    entry_short = raw_entry_short.astype("boolean").groupby(session_dates, sort=True).shift(1).fillna(False).astype(bool)
    stop_long = raw_stop_long.groupby(session_dates, sort=True).shift(1)
    stop_short = raw_stop_short.groupby(session_dates, sort=True).shift(1)
    stop_long = stop_long.where(entry_long)
    stop_short = stop_short.where(entry_short)
    return entry_long, entry_short, stop_long, stop_short


def generate_reclaim_signals(df: pd.DataFrame, variant: VWAPVariantConfig) -> pd.DataFrame:
    """Return a reclaim-style discrete signal frame."""
    out = _ensure_variant_features(df, variant)
    regime_long, regime_short, atr_col, _, _ = _regime_masks(out, variant)
    atr = out[atr_col]

    out["trade_allowed"] = _window_mask(out["timestamp"], variant.time_windows)
    compression_ok = out["compression_width"] <= atr * (1.0 + variant.atr_buffer)
    near_vwap_long = out["compression_low_prev"] <= out["session_vwap"] + atr * variant.atr_buffer
    near_vwap_short = out["compression_high_prev"] >= out["session_vwap"] - atr * variant.atr_buffer
    stop_buffer = float(variant.stop_buffer) if variant.stop_buffer is not None else float(variant.atr_buffer)

    long_breakout = out["close"] > out["compression_high_prev"] + atr * variant.atr_buffer
    short_breakout = out["close"] < out["compression_low_prev"] - atr * variant.atr_buffer

    out["entry_long_raw"] = (
        regime_long
        & compression_ok
        & near_vwap_long
        & long_breakout
        & out["compression_high_prev"].notna()
        & out["compression_low_prev"].notna()
    )
    out["entry_short_raw"] = (
        regime_short
        & compression_ok
        & near_vwap_short
        & short_breakout
        & out["compression_high_prev"].notna()
        & out["compression_low_prev"].notna()
    )
    out["exit_long"] = out["prev_close"].lt(out["prev_session_vwap"]) if variant.exit_on_vwap_recross else False
    out["exit_short"] = out["prev_close"].gt(out["prev_session_vwap"]) if variant.exit_on_vwap_recross else False
    out["stop_reference_long_raw"] = out["compression_low_prev"] - atr * stop_buffer
    out["stop_reference_short_raw"] = out["compression_high_prev"] + atr * stop_buffer
    out["raw_signal"] = np.select([out["entry_long_raw"], out["entry_short_raw"]], [1, -1], default=0).astype(int)
    out["entry_long"], out["entry_short"], out["stop_reference_long"], out["stop_reference_short"] = (
        _shift_close_based_entries_to_next_open(
            out=out,
            session_dates=out["session_date"],
            raw_entry_long=out["entry_long_raw"],
            raw_entry_short=out["entry_short_raw"],
            raw_stop_long=out["stop_reference_long_raw"],
            raw_stop_short=out["stop_reference_short_raw"],
        )
    )
    out["signal"] = np.select([out["entry_long"], out["entry_short"]], [1, -1], default=0).astype(int)
    return out


def generate_pullback_continuation_signals(df: pd.DataFrame, variant: VWAPVariantConfig) -> pd.DataFrame:
    """Return a pullback-continuation discrete signal frame."""
    out = _ensure_variant_features(df, variant)
    regime_long, regime_short, atr_col, _, _ = _regime_masks(out, variant)
    atr = out[atr_col]

    pullback_low_col = f"pullback_low_{variant.pullback_lookback}"
    pullback_high_col = f"pullback_high_{variant.pullback_lookback}"
    down_count_col = f"down_close_count_{variant.pullback_lookback}"
    up_count_col = f"up_close_count_{variant.pullback_lookback}"

    out["trade_allowed"] = _window_mask(out["timestamp"], variant.time_windows)
    stop_buffer = float(variant.stop_buffer) if variant.stop_buffer is not None else float(variant.atr_buffer)

    recent_pullback_low = out[pullback_low_col].shift(1)
    recent_pullback_high = out[pullback_high_col].shift(1)
    down_count = out[down_count_col].shift(1).fillna(0.0)
    up_count = out[up_count_col].shift(1).fillna(0.0)

    structure_valid_long = recent_pullback_low >= out["session_vwap"] - atr * variant.atr_buffer
    structure_valid_short = recent_pullback_high <= out["session_vwap"] + atr * variant.atr_buffer
    pullback_seen_long = down_count >= 1
    pullback_seen_short = up_count >= 1
    continuation_long = out["close"] > out["prev_high"] + atr * variant.confirmation_threshold
    continuation_short = out["close"] < out["prev_low"] - atr * variant.confirmation_threshold

    out["entry_long_raw"] = (
        regime_long
        & structure_valid_long
        & pullback_seen_long
        & continuation_long
        & recent_pullback_low.notna()
    )
    out["entry_short_raw"] = (
        regime_short
        & structure_valid_short
        & pullback_seen_short
        & continuation_short
        & recent_pullback_high.notna()
    )
    out["exit_long"] = out["prev_close"].lt(out["prev_session_vwap"]) if variant.exit_on_vwap_recross else False
    out["exit_short"] = out["prev_close"].gt(out["prev_session_vwap"]) if variant.exit_on_vwap_recross else False
    out["stop_reference_long_raw"] = recent_pullback_low - atr * stop_buffer
    out["stop_reference_short_raw"] = recent_pullback_high + atr * stop_buffer
    out["raw_signal"] = np.select([out["entry_long_raw"], out["entry_short_raw"]], [1, -1], default=0).astype(int)
    out["entry_long"], out["entry_short"], out["stop_reference_long"], out["stop_reference_short"] = (
        _shift_close_based_entries_to_next_open(
            out=out,
            session_dates=out["session_date"],
            raw_entry_long=out["entry_long_raw"],
            raw_entry_short=out["entry_short_raw"],
            raw_stop_long=out["stop_reference_long_raw"],
            raw_stop_short=out["stop_reference_short_raw"],
        )
    )
    out["signal"] = np.select([out["entry_long"], out["entry_short"]], [1, -1], default=0).astype(int)
    return out


def build_vwap_signal_frame(df: pd.DataFrame, variant: VWAPVariantConfig) -> pd.DataFrame:
    """Dispatch to the requested VWAP signal builder."""
    if variant.mode == "target_position":
        return generate_paper_baseline_signals(df, variant)
    if variant.name in {"vwap_reclaim", "vwap_reclaim_with_prop_overlay"}:
        return generate_reclaim_signals(df, variant)
    if variant.name == "vwap_pullback_continuation":
        return generate_pullback_continuation_signals(df, variant)
    raise ValueError(f"Unsupported VWAP variant '{variant.name}'.")
