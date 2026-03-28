"""Volume-context features built with strict past-only references."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _min_periods(window: int) -> int:
    return max(5, int(math.ceil(window / 4.0)))


def _safe_divide(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num.where(den.ne(0)).divide(den.where(den.ne(0)))


def add_bar_volume_context(
    df: pd.DataFrame,
    session_col: str = "session_date",
) -> pd.DataFrame:
    """Add bar-range, true-range, body efficiency, and signed-volume helpers."""
    out = df.copy()
    if session_col not in out.columns:
        out[session_col] = pd.to_datetime(out["timestamp"]).dt.date

    close = pd.to_numeric(out["close"], errors="coerce")
    open_ = pd.to_numeric(out["open"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)

    prev_close = close.groupby(out[session_col]).shift(1)
    bar_range = (high - low).clip(lower=0.0)
    true_range = pd.concat(
        [
            bar_range.rename("bar_range"),
            (high - prev_close).abs().rename("high_prev_close"),
            (low - prev_close).abs().rename("low_prev_close"),
        ],
        axis=1,
    ).max(axis=1)
    body = (close - open_).abs()

    candle_sign = pd.Series(0.0, index=out.index, dtype=float)
    candle_sign = candle_sign.mask(close > open_, 1.0)
    candle_sign = candle_sign.mask(close < open_, -1.0)
    candle_sign = candle_sign.mask(candle_sign.eq(0.0) & close.gt(prev_close), 1.0)
    candle_sign = candle_sign.mask(candle_sign.eq(0.0) & close.lt(prev_close), -1.0)

    out["bar_range"] = bar_range
    out["true_range_1m"] = true_range
    out["body_size"] = body
    out["body_efficiency"] = _safe_divide(body, bar_range.replace(0.0, np.nan)).clip(lower=0.0, upper=1.0)
    out["candle_sign"] = candle_sign.astype(float)
    out["signed_volume"] = volume * out["candle_sign"]
    out["up_volume"] = volume.where(out["candle_sign"] > 0.0, 0.0)
    out["down_volume"] = volume.where(out["candle_sign"] < 0.0, 0.0)
    return out


def add_rth_volume_history_features(
    df: pd.DataFrame,
    opening_time: str,
    time_exit: str,
    rolling_windows: tuple[int, ...] = (10, 20, 40),
    history_windows: tuple[int, ...] = (20, 60),
    session_col: str = "session_date",
) -> pd.DataFrame:
    """Add leak-free RTH volume history features.

    All historical references use either:
    - prior bars from the same session (`shift(1)` before rolling), or
    - prior sessions at the same minute-of-day (`shift(1)` before rolling).
    """

    out = add_bar_volume_context(df, session_col=session_col)
    if session_col not in out.columns:
        out[session_col] = pd.to_datetime(out["timestamp"]).dt.date

    timestamp = pd.to_datetime(out["timestamp"], errors="coerce")
    out["minute_of_day"] = timestamp.dt.hour * 60 + timestamp.dt.minute

    open_minutes = pd.Timestamp(opening_time).hour * 60 + pd.Timestamp(opening_time).minute
    close_minutes = pd.Timestamp(time_exit).hour * 60 + pd.Timestamp(time_exit).minute
    is_rth = out["minute_of_day"].between(open_minutes, close_minutes, inclusive="both")
    out["is_rth"] = is_rth

    rth = out.loc[is_rth].copy()
    if rth.empty:
        return out

    rth = rth.sort_values("timestamp").copy()
    volume = pd.to_numeric(rth["volume"], errors="coerce").fillna(0.0)
    rth["rth_cum_volume"] = volume.groupby(rth[session_col]).cumsum()

    session_totals = (
        rth.groupby(session_col, sort=True)["volume"]
        .sum()
        .rename("rth_final_volume")
        .reset_index()
        .sort_values(session_col)
        .reset_index(drop=True)
    )
    for window in history_windows:
        session_totals[f"rth_final_volume_mean_hist_{window}"] = (
            pd.to_numeric(session_totals["rth_final_volume"], errors="coerce")
            .shift(1)
            .rolling(window, min_periods=_min_periods(window))
            .mean()
        )
        session_totals[f"rth_final_volume_std_hist_{window}"] = (
            pd.to_numeric(session_totals["rth_final_volume"], errors="coerce")
            .shift(1)
            .rolling(window, min_periods=_min_periods(window))
            .std(ddof=0)
        )
    rth = rth.merge(session_totals, on=session_col, how="left")

    grouped_session = rth.groupby(session_col, sort=False)
    for window in rolling_windows:
        min_periods = _min_periods(window)
        rth[f"vol_mean_prev_{window}"] = grouped_session["volume"].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(window, min_periods=min_periods).mean()
        )
        rth[f"vol_std_prev_{window}"] = grouped_session["volume"].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(window, min_periods=min_periods).std(ddof=0)
        )
        rth[f"bar_range_mean_prev_{window}"] = grouped_session["bar_range"].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(window, min_periods=min_periods).mean()
        )
        rth[f"bar_range_std_prev_{window}"] = grouped_session["bar_range"].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(window, min_periods=min_periods).std(ddof=0)
        )
        rth[f"signed_volume_sum_prev_{window}"] = grouped_session["signed_volume"].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(window, min_periods=min_periods).sum()
        )
        rth[f"up_volume_sum_prev_{window}"] = grouped_session["up_volume"].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(window, min_periods=min_periods).sum()
        )
        rth[f"down_volume_sum_prev_{window}"] = grouped_session["down_volume"].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(window, min_periods=min_periods).sum()
        )
        rth[f"total_volume_sum_prev_{window}"] = grouped_session["volume"].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(window, min_periods=min_periods).sum()
        )

    grouped_minute = rth.groupby("minute_of_day", sort=False)
    for window in history_windows:
        min_periods = _min_periods(window)
        rth[f"same_minute_volume_mean_hist_{window}"] = grouped_minute["volume"].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(window, min_periods=min_periods).mean()
        )
        rth[f"same_minute_volume_std_hist_{window}"] = grouped_minute["volume"].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(window, min_periods=min_periods).std(ddof=0)
        )
        rth[f"same_minute_cum_volume_mean_hist_{window}"] = grouped_minute["rth_cum_volume"].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(window, min_periods=min_periods).mean()
        )
        rth[f"same_minute_cum_volume_std_hist_{window}"] = grouped_minute["rth_cum_volume"].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(window, min_periods=min_periods).std(ddof=0)
        )

    for column in rth.columns:
        if column not in out.columns:
            out[column] = np.nan
    out.loc[rth.index, rth.columns] = rth
    return out
