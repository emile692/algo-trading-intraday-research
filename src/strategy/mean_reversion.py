"""Signal construction for the intraday mean reversion campaign."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.config.mean_reversion_campaign import MeanReversionVariantConfig
from src.config.vwap_campaign import TimeWindow
from src.data.session import add_session_date
from src.features.intraday import add_intraday_features, add_session_vwap
from src.features.opening_range import compute_opening_range
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
        return pd.Series(False, index=timestamps.index, dtype=bool)
    mask = pd.Series(False, index=timestamps.index, dtype=bool)
    for window in windows:
        mask |= _left_closed_time_mask(timestamps, window.start, window.end)
    return mask


def filter_rth_bar_starts(
    df: pd.DataFrame,
    session_start: str = "09:30:00",
    session_end: str = "16:00:00",
) -> pd.DataFrame:
    """Filter a start-aligned dataset to RTH bars in [start, end)."""
    mask = _left_closed_time_mask(pd.to_datetime(df["timestamp"]), session_start, session_end)
    return df.loc[mask].copy().reset_index(drop=True)


def _normalize_ints(values: Iterable[int]) -> list[int]:
    clean = sorted({int(value) for value in values if int(value) > 0})
    return clean


def _normalize_stochastic_defs(
    values: Iterable[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    clean: set[tuple[int, int, int]] = set()
    for fast_k, smooth_k, smooth_d in values:
        fast_k_i = int(fast_k)
        smooth_k_i = int(smooth_k)
        smooth_d_i = int(smooth_d)
        if min(fast_k_i, smooth_k_i, smooth_d_i) > 0:
            clean.add((fast_k_i, smooth_k_i, smooth_d_i))
    return sorted(clean)


def _std_replace_zero(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.replace(0.0, np.nan)


def _add_ema_columns(df: pd.DataFrame, windows: Iterable[int], price_col: str = "close") -> pd.DataFrame:
    out = df.copy()
    for window in _normalize_ints(windows):
        col = f"ema_{window}"
        if col not in out.columns:
            out[col] = out[price_col].ewm(span=window, adjust=False, min_periods=window).mean()
    return out


def _add_rsi_columns(df: pd.DataFrame, windows: Iterable[int], price_col: str = "close") -> pd.DataFrame:
    out = df.copy()
    delta = out[price_col].diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    for window in _normalize_ints(windows):
        avg_gain = gains.ewm(alpha=1.0 / float(window), adjust=False, min_periods=window).mean()
        avg_loss = losses.ewm(alpha=1.0 / float(window), adjust=False, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        out[f"rsi_{window}"] = 100.0 - (100.0 / (1.0 + rs))
    return out


def _add_stochastic_columns(
    df: pd.DataFrame,
    definitions: Iterable[tuple[int, int, int]],
) -> pd.DataFrame:
    out = df.copy()
    for fast_k, smooth_k, smooth_d in _normalize_stochastic_defs(definitions):
        lowest_low = out["low"].rolling(fast_k, min_periods=fast_k).min()
        highest_high = out["high"].rolling(fast_k, min_periods=fast_k).max()
        raw_k = 100.0 * (out["close"] - lowest_low) / (highest_high - lowest_low).replace(0.0, np.nan)
        slow_k = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
        slow_d = slow_k.rolling(smooth_d, min_periods=smooth_d).mean()
        out[f"stoch_k_{fast_k}_{smooth_k}_{smooth_d}"] = slow_k
        out[f"stoch_d_{fast_k}_{smooth_k}_{smooth_d}"] = slow_d
    return out


def _add_adx_columns(df: pd.DataFrame, periods: Iterable[int]) -> pd.DataFrame:
    out = df.copy()
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=out.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=out.index,
        dtype=float,
    )

    for period in _normalize_ints(periods):
        atr = tr.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()
        plus_smoothed = plus_dm.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()
        minus_smoothed = minus_dm.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()
        plus_di = 100.0 * plus_smoothed / atr.replace(0.0, np.nan)
        minus_di = 100.0 * minus_smoothed / atr.replace(0.0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
        out[f"adx_{period}"] = dx.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()
    return out


def _add_opening_range_windows(
    df: pd.DataFrame,
    windows: Iterable[int],
    opening_time: str,
) -> pd.DataFrame:
    out = df.copy()
    for window in _normalize_ints(windows):
        or_frame = compute_opening_range(out, or_minutes=window, opening_time=opening_time)
        out[f"or_high_{window}"] = or_frame["or_high"]
        out[f"or_low_{window}"] = or_frame["or_low"]
        out[f"or_width_{window}"] = or_frame["or_width"]
        out[f"or_midpoint_{window}"] = or_frame["or_midpoint"]
    return out


def _add_streak_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    up = np.zeros(len(out), dtype=int)
    down = np.zeros(len(out), dtype=int)

    for _, session_df in out.groupby("session_date", sort=True):
        positions = session_df.index.to_list()
        closes = pd.to_numeric(session_df["close"], errors="coerce").to_numpy(dtype=float)
        up_count = 0
        down_count = 0
        for offset, idx in enumerate(positions):
            if offset == 0:
                up[idx] = 0
                down[idx] = 0
                continue
            if closes[offset] > closes[offset - 1]:
                up_count += 1
                down_count = 0
            elif closes[offset] < closes[offset - 1]:
                down_count += 1
                up_count = 0
            else:
                up_count = 0
                down_count = 0
            up[idx] = up_count
            down[idx] = down_count

    out["up_streak"] = up
    out["down_streak"] = down
    return out


def prepare_mean_reversion_feature_frame(
    df: pd.DataFrame,
    session_start: str = "09:30:00",
    session_end: str = "16:00:00",
    atr_windows: Iterable[int] = (14,),
    ema_windows: Iterable[int] = (20, 30),
    zscore_windows: Iterable[int] = (20, 30, 50),
    bollinger_windows: Iterable[int] = (20, 30),
    rsi_windows: Iterable[int] = (2, 3, 5),
    stochastic_defs: Iterable[tuple[int, int, int]] = ((5, 3, 3), (8, 3, 3)),
    adx_periods: Iterable[int] = (14,),
    opening_windows: Iterable[int] = (30, 45, 60),
    persistent_lookbacks: Iterable[int] = (4,),
    ema_slope_specs: Iterable[tuple[int, int]] = ((20, 3), (30, 3)),
    vwap_slope_lookbacks: Iterable[int] = (3,),
    vwap_price_mode: str = "typical",
    vwap_price_volume_col: str | None = None,
) -> pd.DataFrame:
    """Build the RTH feature frame used by all mean reversion families."""
    out = filter_rth_bar_starts(df, session_start=session_start, session_end=session_end)
    out = add_session_date(out)
    out = add_intraday_features(out)
    out = add_session_vwap(out, price_mode=vwap_price_mode, price_volume_col=vwap_price_volume_col)
    out = add_atr(out, window=_normalize_ints(atr_windows))
    out = _add_ema_columns(out, ema_windows)
    out = _add_rsi_columns(out, rsi_windows)
    out = _add_stochastic_columns(out, stochastic_defs)
    out = _add_adx_columns(out, adx_periods)
    out = _add_opening_range_windows(out, opening_windows, opening_time=session_start)
    out = _add_streak_columns(out)
    out = out.sort_values("timestamp").reset_index(drop=True)

    group = out.groupby("session_date", sort=True)
    out["prev_close"] = group["close"].shift(1)
    out["prev_high"] = group["high"].shift(1)
    out["prev_low"] = group["low"].shift(1)
    out["prev_session_vwap"] = group["session_vwap"].shift(1)
    out["is_first_bar_of_session"] = group.cumcount().eq(0)
    out["is_last_bar_of_session"] = group.cumcount(ascending=False).eq(0)
    out["session_open"] = group["open"].transform("first")
    out["session_high_so_far"] = group["high"].cummax()
    out["session_low_so_far"] = group["low"].cummin()
    out["session_range_so_far"] = out["session_high_so_far"] - out["session_low_so_far"]
    out["session_last_minute"] = group["minute_of_day"].transform("max")
    out["session_first_minute"] = group["minute_of_day"].transform("min")
    out["minutes_from_open"] = out["minute_of_day"] - out["session_first_minute"]
    out["minutes_to_close"] = out["session_last_minute"] - out["minute_of_day"]
    out["volume_ma_20"] = out["volume"].rolling(20, min_periods=10).mean()
    out["volume_std_20"] = _std_replace_zero(out["volume"].rolling(20, min_periods=10).std(ddof=0))
    out["volume_zscore_20"] = (out["volume"] - out["volume_ma_20"]) / out["volume_std_20"]
    out["bar_range"] = out["high"] - out["low"]
    out["bar_body"] = (out["close"] - out["open"]).abs()
    out["close_location_in_bar"] = (out["close"] - out["low"]) / out["bar_range"].replace(0.0, np.nan)
    out["open_location_in_bar"] = (out["open"] - out["low"]) / out["bar_range"].replace(0.0, np.nan)
    out["bullish_reversal_bar"] = (
        (out["close"] > out["open"])
        & (out["close_location_in_bar"] >= 0.65)
        & (out["open_location_in_bar"] <= 0.50)
    )
    out["bearish_reversal_bar"] = (
        (out["close"] < out["open"])
        & ((1.0 - out["close_location_in_bar"]) >= 0.65)
        & ((1.0 - out["open_location_in_bar"]) <= 0.50)
    )

    for window in sorted({*set(_normalize_ints(zscore_windows)), *set(_normalize_ints(bollinger_windows))}):
        rolling_mean = out["close"].rolling(window, min_periods=window).mean()
        rolling_std = _std_replace_zero(out["close"].rolling(window, min_periods=window).std(ddof=0))
        out[f"rolling_mean_{window}"] = rolling_mean
        out[f"rolling_std_{window}"] = rolling_std
        out[f"close_zscore_{window}"] = (out["close"] - rolling_mean) / rolling_std

    distance = out["close"] - out["session_vwap"]
    for window in _normalize_ints(zscore_windows):
        dist_mean = distance.rolling(window, min_periods=window).mean()
        dist_std = _std_replace_zero(distance.rolling(window, min_periods=window).std(ddof=0))
        out[f"vwap_distance_z_{window}"] = (distance - dist_mean) / dist_std

    for atr_window in _normalize_ints(atr_windows):
        atr_col = f"atr_{atr_window}"
        atr = out[atr_col].replace(0.0, np.nan)
        out[f"vwap_distance_atr_{atr_window}"] = (out["close"] - out["session_vwap"]) / atr
        out[f"session_open_extension_atr_{atr_window}"] = (out["close"] - out["session_open"]) / atr
        out[f"session_range_so_far_atr_{atr_window}"] = out["session_range_so_far"] / atr
        for lookback in _normalize_ints(persistent_lookbacks):
            out[f"persistent_vwap_distance_{lookback}_{atr_window}"] = (
                out[f"vwap_distance_atr_{atr_window}"]
                .abs()
                .groupby(out["session_date"], sort=True)
                .rolling(lookback, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    for ema_window, lookback in sorted({(int(window), int(lb)) for window, lb in ema_slope_specs if int(window) > 0 and int(lb) > 0}):
        ema_col = f"ema_{ema_window}"
        ref_atr = f"atr_{_normalize_ints(atr_windows)[0]}"
        out[f"ema_slope_atr_{ema_window}_{lookback}"] = (
            out[ema_col] - out[ema_col].shift(lookback)
        ) / out[ref_atr].replace(0.0, np.nan)

    base_atr_col = f"atr_{_normalize_ints(atr_windows)[0]}"
    for lookback in _normalize_ints(vwap_slope_lookbacks):
        out[f"vwap_slope_atr_{lookback}"] = (
            out.groupby("session_date", sort=True)["session_vwap"].diff(lookback)
            / out[base_atr_col].replace(0.0, np.nan)
        )

    out["trend_day_score"] = np.maximum(
        out[f"session_open_extension_atr_{_normalize_ints(atr_windows)[0]}"].abs(),
        (
            out[f"vwap_distance_atr_{_normalize_ints(atr_windows)[0]}"].abs()
            + 0.35 * out[f"session_range_so_far_atr_{_normalize_ints(atr_windows)[0]}"].clip(lower=0.0)
        ),
    )
    return out


def _trade_allowed_mask(out: pd.DataFrame, variant: MeanReversionVariantConfig) -> pd.Series:
    allowed = _left_closed_time_mask(out["timestamp"], variant.entry_start, variant.entry_end)
    allowed &= out["minutes_from_open"] >= int(variant.skip_first_minutes)
    allowed &= out["minutes_to_close"] >= int(variant.skip_last_minutes)
    if variant.excluded_windows:
        allowed &= ~_window_mask(out["timestamp"], variant.excluded_windows)
    return allowed


def _resolve_target_series(out: pd.DataFrame, variant: MeanReversionVariantConfig) -> pd.Series:
    source = variant.target_source
    if source == "session_vwap":
        return out["session_vwap"]
    if source == "session_open":
        return out["session_open"]
    if source == "or_midpoint":
        window = int(variant.opening_window_minutes or 30)
        return out[f"or_midpoint_{window}"]
    if source.startswith("or_midpoint_"):
        return out[source]
    if source.startswith("ema_"):
        return out[source]
    if source.startswith("rolling_mean_"):
        return out[source]
    raise ValueError(f"Unsupported target_source '{source}'.")


def _atr_for_variant(out: pd.DataFrame, variant: MeanReversionVariantConfig) -> pd.Series:
    return out[f"atr_{variant.atr_period}"]


def _common_entry_filters(out: pd.DataFrame, variant: MeanReversionVariantConfig) -> pd.Series:
    atr = _atr_for_variant(out, variant).replace(0.0, np.nan)
    filt = pd.Series(True, index=out.index, dtype=bool)
    if variant.adx_max is not None:
        filt &= pd.to_numeric(out[f"adx_{variant.adx_period}"], errors="coerce") <= float(variant.adx_max)
    if variant.ema_slope_max_atr is not None:
        slope_col = f"ema_slope_atr_{variant.ema_filter_window}_{variant.ema_slope_lookback}"
        filt &= pd.to_numeric(out[slope_col], errors="coerce").abs() <= float(variant.ema_slope_max_atr)
    if variant.vwap_slope_max_atr is not None:
        slope_col = f"vwap_slope_atr_{variant.vwap_slope_lookback}"
        filt &= pd.to_numeric(out[slope_col], errors="coerce").abs() <= float(variant.vwap_slope_max_atr)
    if variant.anti_trend_day_max is not None:
        filt &= pd.to_numeric(out["trend_day_score"], errors="coerce") <= float(variant.anti_trend_day_max) * 3.0
    if variant.session_range_max_atr is not None:
        filt &= pd.to_numeric(out[f"session_range_so_far_atr_{variant.atr_period}"], errors="coerce") <= float(
            variant.session_range_max_atr
        ) * 4.0
    if variant.persistent_vwap_distance_max is not None:
        distance_col = f"persistent_vwap_distance_{variant.persistent_lookback}_{variant.atr_period}"
        filt &= pd.to_numeric(out[distance_col], errors="coerce") <= float(variant.persistent_vwap_distance_max) * 2.0
    if variant.opening_impulse_max_atr is not None:
        impulse_col = f"session_open_extension_atr_{variant.atr_period}"
        filt &= pd.to_numeric(out[impulse_col], errors="coerce").abs() <= float(variant.opening_impulse_max_atr) * 1.5
    if variant.anchor_distance_max_atr is not None:
        anchor = _resolve_target_series(out, variant).replace(0.0, np.nan)
        anchor_distance = (out["close"] - anchor).abs() / atr
        filt &= anchor_distance <= float(variant.anchor_distance_max_atr)
    return filt & _trade_allowed_mask(out, variant) & atr.notna()


def _shift_entries_to_next_open(
    out: pd.DataFrame,
    raw_entry_long: pd.Series,
    raw_entry_short: pd.Series,
    raw_stop_long: pd.Series,
    raw_stop_short: pd.Series,
    raw_target_long: pd.Series,
    raw_target_short: pd.Series,
) -> pd.DataFrame:
    result = out.copy()
    group = result.groupby("session_date", sort=True)
    entry_long = raw_entry_long.astype("boolean").groupby(result["session_date"], sort=True).shift(1).fillna(False).astype(bool)
    entry_short = raw_entry_short.astype("boolean").groupby(result["session_date"], sort=True).shift(1).fillna(False).astype(bool)
    stop_reference_long = raw_stop_long.groupby(result["session_date"], sort=True).shift(1)
    stop_reference_short = raw_stop_short.groupby(result["session_date"], sort=True).shift(1)
    target_reference_long = raw_target_long.groupby(result["session_date"], sort=True).shift(1)
    target_reference_short = raw_target_short.groupby(result["session_date"], sort=True).shift(1)

    entry_long &= result["trade_allowed"]
    entry_short &= result["trade_allowed"]
    stop_reference_long = stop_reference_long.where(entry_long)
    stop_reference_short = stop_reference_short.where(entry_short)
    target_reference_long = target_reference_long.where(entry_long)
    target_reference_short = target_reference_short.where(entry_short)

    result["entry_long"] = entry_long
    result["entry_short"] = entry_short
    result["stop_reference_long"] = stop_reference_long
    result["stop_reference_short"] = stop_reference_short
    result["target_reference_long"] = target_reference_long
    result["target_reference_short"] = target_reference_short
    result["raw_signal"] = np.select([raw_entry_long, raw_entry_short], [1, -1], default=0).astype(int)
    result["signal"] = np.select([entry_long, entry_short], [1, -1], default=0).astype(int)
    return result


def _base_stop_series(out: pd.DataFrame, variant: MeanReversionVariantConfig) -> tuple[pd.Series, pd.Series]:
    atr = _atr_for_variant(out, variant)
    raw_stop_long = out["low"] - float(variant.stop_atr_multiple) * atr
    raw_stop_short = out["high"] + float(variant.stop_atr_multiple) * atr
    return raw_stop_long, raw_stop_short


def generate_vwap_extension_reversion_signals(
    df: pd.DataFrame,
    variant: MeanReversionVariantConfig,
) -> pd.DataFrame:
    out = df.copy()
    out["trade_allowed"] = _trade_allowed_mask(out, variant)
    filters = _common_entry_filters(out, variant)

    if variant.extension_mode == "atr":
        distance_col = f"vwap_distance_atr_{variant.atr_period}"
        if distance_col in out.columns:
            extension = pd.to_numeric(out[distance_col], errors="coerce")
        else:
            atr = _atr_for_variant(out, variant).replace(0.0, np.nan)
            extension = (out["close"] - out["session_vwap"]) / atr
    elif variant.extension_mode == "zscore":
        extension = pd.to_numeric(out[f"vwap_distance_z_{int(variant.zscore_window)}"], errors="coerce")
    else:
        raise ValueError(f"Unsupported extension_mode '{variant.extension_mode}'.")

    raw_entry_long = filters & (extension <= -float(variant.extension_threshold))
    raw_entry_short = filters & (extension >= float(variant.extension_threshold))
    raw_stop_long, raw_stop_short = _base_stop_series(out, variant)
    raw_target = _resolve_target_series(out, variant)
    return _shift_entries_to_next_open(
        out,
        raw_entry_long=raw_entry_long,
        raw_entry_short=raw_entry_short,
        raw_stop_long=raw_stop_long,
        raw_stop_short=raw_stop_short,
        raw_target_long=raw_target,
        raw_target_short=raw_target,
    )


def generate_bollinger_zscore_reversion_signals(
    df: pd.DataFrame,
    variant: MeanReversionVariantConfig,
) -> pd.DataFrame:
    out = df.copy()
    out["trade_allowed"] = _trade_allowed_mask(out, variant)
    filters = _common_entry_filters(out, variant)

    if variant.bollinger_window is not None and variant.bollinger_std is not None:
        mean_col = f"rolling_mean_{variant.bollinger_window}"
        std_col = f"rolling_std_{variant.bollinger_window}"
        mean = out[mean_col]
        std = out[std_col]
        lower = mean - float(variant.bollinger_std) * std
        upper = mean + float(variant.bollinger_std) * std
        if variant.bollinger_confirmation == "reentry":
            raw_entry_long = filters & (out["prev_close"] < lower.shift(1)) & (out["close"] >= lower)
            raw_entry_short = filters & (out["prev_close"] > upper.shift(1)) & (out["close"] <= upper)
        else:
            raw_entry_long = filters & (out["close"] <= lower)
            raw_entry_short = filters & (out["close"] >= upper)
        raw_target = out[mean_col]
    elif variant.zscore_window is not None and variant.extension_threshold is not None:
        zscore = pd.to_numeric(out[f"close_zscore_{variant.zscore_window}"], errors="coerce")
        if variant.bollinger_confirmation == "reentry":
            raw_entry_long = filters & (pd.to_numeric(out[f"close_zscore_{variant.zscore_window}"], errors="coerce").shift(1) < -float(variant.extension_threshold)) & (zscore >= -float(variant.extension_threshold))
            raw_entry_short = filters & (pd.to_numeric(out[f"close_zscore_{variant.zscore_window}"], errors="coerce").shift(1) > float(variant.extension_threshold)) & (zscore <= float(variant.extension_threshold))
        else:
            raw_entry_long = filters & (zscore <= -float(variant.extension_threshold))
            raw_entry_short = filters & (zscore >= float(variant.extension_threshold))
        raw_target = out[f"rolling_mean_{variant.zscore_window}"]
    else:
        raise ValueError("Bollinger/z-score variant requires either Bollinger parameters or a z-score window.")

    raw_stop_long, raw_stop_short = _base_stop_series(out, variant)
    return _shift_entries_to_next_open(
        out,
        raw_entry_long=raw_entry_long,
        raw_entry_short=raw_entry_short,
        raw_stop_long=raw_stop_long,
        raw_stop_short=raw_stop_short,
        raw_target_long=raw_target,
        raw_target_short=raw_target,
    )


def generate_rsi_stochastic_contrarian_signals(
    df: pd.DataFrame,
    variant: MeanReversionVariantConfig,
) -> pd.DataFrame:
    out = df.copy()
    out["trade_allowed"] = _trade_allowed_mask(out, variant)
    filters = _common_entry_filters(out, variant)

    if variant.oscillator_kind == "rsi":
        osc = pd.to_numeric(out[f"rsi_{int(variant.oscillator_period)}"], errors="coerce")
    elif variant.oscillator_kind == "stochastic":
        k_col = f"stoch_k_{int(variant.oscillator_period_fast)}_{int(variant.oscillator_period_slow)}_{int(variant.oscillator_smoothing)}"
        osc = pd.to_numeric(out[k_col], errors="coerce")
    else:
        raise ValueError(f"Unsupported oscillator_kind '{variant.oscillator_kind}'.")

    prev_osc = osc.shift(1)
    oversold = float(variant.oversold_level)
    overbought = float(variant.overbought_level)
    bullish_reversal = pd.Series(out["bullish_reversal_bar"], dtype=bool)
    bearish_reversal = pd.Series(out["bearish_reversal_bar"], dtype=bool)

    if variant.oscillator_trigger == "exit_extreme":
        raw_entry_long = filters & (prev_osc <= oversold) & (osc > oversold)
        raw_entry_short = filters & (prev_osc >= overbought) & (osc < overbought)
    elif variant.oscillator_trigger == "extreme_reversal":
        raw_entry_long = filters & (osc <= oversold) & bullish_reversal
        raw_entry_short = filters & (osc >= overbought) & bearish_reversal
    else:
        raw_entry_long = filters & (osc <= oversold)
        raw_entry_short = filters & (osc >= overbought)

    raw_stop_long, raw_stop_short = _base_stop_series(out, variant)
    raw_target = _resolve_target_series(out, variant)
    return _shift_entries_to_next_open(
        out,
        raw_entry_long=raw_entry_long,
        raw_entry_short=raw_entry_short,
        raw_stop_long=raw_stop_long,
        raw_stop_short=raw_stop_short,
        raw_target_long=raw_target,
        raw_target_short=raw_target,
    )


def generate_opening_stretch_fade_signals(
    df: pd.DataFrame,
    variant: MeanReversionVariantConfig,
) -> pd.DataFrame:
    out = df.copy()
    out["trade_allowed"] = _trade_allowed_mask(out, variant)
    filters = _common_entry_filters(out, variant)

    window = int(variant.opening_window_minutes or 30)
    after_window = out["minutes_from_open"] >= window
    if variant.stretch_reference == "session_open":
        reference = out["session_open"]
    elif variant.stretch_reference == "or_midpoint":
        reference = out[f"or_midpoint_{window}"]
    else:
        raise ValueError(f"Unsupported stretch_reference '{variant.stretch_reference}'.")

    atr = _atr_for_variant(out, variant).replace(0.0, np.nan)
    stretch = (out["close"] - reference) / atr
    raw_entry_long = filters & after_window & (stretch <= -float(variant.stretch_threshold_atr))
    raw_entry_short = filters & after_window & (stretch >= float(variant.stretch_threshold_atr))

    if variant.stretch_threshold_or_multiple is not None:
        or_width = pd.to_numeric(out[f"or_width_{window}"], errors="coerce")
        distance = (out["close"] - reference).abs()
        threshold = or_width * float(variant.stretch_threshold_or_multiple)
        raw_entry_long &= distance >= threshold
        raw_entry_short &= distance >= threshold

    if variant.stretch_volume_z_max is not None:
        volume_z = pd.to_numeric(out["volume_zscore_20"], errors="coerce").abs()
        raw_entry_long &= volume_z <= float(variant.stretch_volume_z_max)
        raw_entry_short &= volume_z <= float(variant.stretch_volume_z_max)

    raw_stop_long, raw_stop_short = _base_stop_series(out, variant)
    raw_target = _resolve_target_series(out, variant)
    return _shift_entries_to_next_open(
        out,
        raw_entry_long=raw_entry_long,
        raw_entry_short=raw_entry_short,
        raw_stop_long=raw_stop_long,
        raw_stop_short=raw_stop_short,
        raw_target_long=raw_target,
        raw_target_short=raw_target,
    )


def generate_keltner_snapback_signals(
    df: pd.DataFrame,
    variant: MeanReversionVariantConfig,
) -> pd.DataFrame:
    out = df.copy()
    out["trade_allowed"] = _trade_allowed_mask(out, variant)
    filters = _common_entry_filters(out, variant)

    ema_col = f"ema_{int(variant.ema_window)}"
    atr = _atr_for_variant(out, variant).replace(0.0, np.nan)
    lower = out[ema_col] - float(variant.band_width_atr) * atr
    upper = out[ema_col] + float(variant.band_width_atr) * atr
    outside_lower = out["close"] <= lower
    outside_upper = out["close"] >= upper

    if int(variant.require_closes_outside) > 1:
        lower_count = outside_lower.rolling(int(variant.require_closes_outside), min_periods=int(variant.require_closes_outside)).sum()
        upper_count = outside_upper.rolling(int(variant.require_closes_outside), min_periods=int(variant.require_closes_outside)).sum()
        raw_entry_long = filters & (lower_count >= int(variant.require_closes_outside))
        raw_entry_short = filters & (upper_count >= int(variant.require_closes_outside))
    else:
        raw_entry_long = filters & outside_lower
        raw_entry_short = filters & outside_upper

    raw_stop_long, raw_stop_short = _base_stop_series(out, variant)
    raw_target = _resolve_target_series(out, variant)
    return _shift_entries_to_next_open(
        out,
        raw_entry_long=raw_entry_long,
        raw_entry_short=raw_entry_short,
        raw_stop_long=raw_stop_long,
        raw_stop_short=raw_stop_short,
        raw_target_long=raw_target,
        raw_target_short=raw_target,
    )


def generate_streak_exhaustion_signals(
    df: pd.DataFrame,
    variant: MeanReversionVariantConfig,
) -> pd.DataFrame:
    out = df.copy()
    out["trade_allowed"] = _trade_allowed_mask(out, variant)
    filters = _common_entry_filters(out, variant)

    atr = _atr_for_variant(out, variant).replace(0.0, np.nan)
    shift_n = int(variant.streak_length)
    session_shift = out.groupby("session_date", sort=True)["close"].shift(shift_n)
    move_atr = (out["close"] - session_shift) / atr

    raw_entry_long = filters & (out["down_streak"] >= shift_n) & (move_atr <= -float(variant.streak_extension_atr))
    raw_entry_short = filters & (out["up_streak"] >= shift_n) & (move_atr >= float(variant.streak_extension_atr))

    if variant.require_exhaustion_bar:
        raw_entry_long &= pd.Series(out["bullish_reversal_bar"], dtype=bool)
        raw_entry_short &= pd.Series(out["bearish_reversal_bar"], dtype=bool)

    raw_stop_long, raw_stop_short = _base_stop_series(out, variant)
    raw_target = _resolve_target_series(out, variant)
    return _shift_entries_to_next_open(
        out,
        raw_entry_long=raw_entry_long,
        raw_entry_short=raw_entry_short,
        raw_stop_long=raw_stop_long,
        raw_stop_short=raw_stop_short,
        raw_target_long=raw_target,
        raw_target_short=raw_target,
    )


def build_mean_reversion_signal_frame(
    df: pd.DataFrame,
    variant: MeanReversionVariantConfig,
) -> pd.DataFrame:
    """Dispatch to the requested mean reversion family."""
    if variant.family == "vwap_extension_reversion":
        out = generate_vwap_extension_reversion_signals(df, variant)
    elif variant.family == "bollinger_zscore_reversion":
        out = generate_bollinger_zscore_reversion_signals(df, variant)
    elif variant.family == "rsi_stochastic_contrarian":
        out = generate_rsi_stochastic_contrarian_signals(df, variant)
    elif variant.family == "opening_stretch_fade":
        out = generate_opening_stretch_fade_signals(df, variant)
    elif variant.family == "keltner_band_snapback":
        out = generate_keltner_snapback_signals(df, variant)
    elif variant.family == "streak_exhaustion_reversion":
        out = generate_streak_exhaustion_signals(df, variant)
    else:
        raise ValueError(f"Unsupported family '{variant.family}'.")

    out["variant_name"] = variant.name
    out["family"] = variant.family
    out["symbol"] = variant.symbol
    out["timeframe"] = variant.timeframe
    return out
