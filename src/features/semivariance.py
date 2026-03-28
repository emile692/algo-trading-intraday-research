"""Intraday realized semivariance utilities."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_EPS = 1e-12


def infer_bar_minutes(timestamp: pd.Series) -> int:
    """Infer the typical bar size in minutes from a timestamp series."""
    clean = pd.to_datetime(timestamp, errors="coerce").dropna().drop_duplicates().sort_values()
    if len(clean) < 2:
        return 1
    diffs = clean.diff().dropna().dt.total_seconds()
    positive = diffs[diffs > 0]
    if positive.empty:
        return 1
    return max(1, int(round(float(positive.median()) / 60.0)))


def _rolling_group_sum(values: pd.Series, groups: pd.Series, window_bars: int) -> pd.Series:
    rolled = (
        pd.Series(values, index=values.index, dtype=float)
        .groupby(groups, sort=False)
        .rolling(window=window_bars, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return pd.Series(rolled, index=values.index, dtype=float)


def _append_semivariance_family(
    out: pd.DataFrame,
    *,
    label: str,
    rs_plus: pd.Series,
    rs_minus: pd.Series,
    eps: float,
) -> None:
    rv = pd.to_numeric(rs_plus, errors="coerce").fillna(0.0) + pd.to_numeric(rs_minus, errors="coerce").fillna(0.0)
    denom = rv.clip(lower=float(eps))
    out[f"rs_plus_{label}"] = pd.to_numeric(rs_plus, errors="coerce").fillna(0.0)
    out[f"rs_minus_{label}"] = pd.to_numeric(rs_minus, errors="coerce").fillna(0.0)
    out[f"rv_{label}"] = rv
    out[f"rs_plus_share_{label}"] = out[f"rs_plus_{label}"] / denom
    out[f"rs_minus_share_{label}"] = out[f"rs_minus_{label}"] / denom
    out[f"rs_imbalance_{label}"] = (out[f"rs_plus_{label}"] - out[f"rs_minus_{label}"]) / denom
    out[f"abs_rs_imbalance_{label}"] = out[f"rs_imbalance_{label}"].abs()


def add_realized_semivariance_features(
    df: pd.DataFrame,
    *,
    price_col: str = "close",
    timestamp_col: str = "timestamp",
    continuous_session_col: str = "continuous_session_date",
    session_col: str = "session_date",
    session_open_time: str = "09:30:00",
    rth_end_time: str = "16:00:00",
    window_minutes: Iterable[int] = (30, 60, 90),
    eps: float = DEFAULT_EPS,
) -> pd.DataFrame:
    """Add ex-ante realized semivariance features on top of an intraday frame."""
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    if out[timestamp_col].isna().any():
        raise ValueError(f"Column '{timestamp_col}' contains unparsable timestamps.")
    out = out.sort_values(timestamp_col).copy()

    if continuous_session_col not in out.columns:
        out[continuous_session_col] = pd.to_datetime(out[session_col], errors="coerce").dt.date
    if session_col not in out.columns:
        out[session_col] = out[timestamp_col].dt.date

    out[session_col] = pd.to_datetime(out[session_col], errors="coerce").dt.date
    out[continuous_session_col] = pd.to_datetime(out[continuous_session_col], errors="coerce").dt.date

    returns = (
        pd.to_numeric(out[price_col], errors="coerce")
        .groupby(out[continuous_session_col], sort=False)
        .pct_change()
        .fillna(0.0)
    )
    rs_plus_bar = returns.clip(lower=0.0).pow(2)
    rs_minus_bar = returns.clip(upper=0.0).pow(2).abs()

    bar_minutes = infer_bar_minutes(out[timestamp_col])
    out["semivariance_bar_minutes"] = int(bar_minutes)
    out["ret_simple"] = pd.Series(returns, index=out.index, dtype=float)

    for minutes in window_minutes:
        label = f"{int(minutes)}m"
        window_bars = max(1, int(round(float(minutes) / float(bar_minutes))))
        rs_plus = _rolling_group_sum(rs_plus_bar, out[continuous_session_col], window_bars=window_bars)
        rs_minus = _rolling_group_sum(rs_minus_bar, out[continuous_session_col], window_bars=window_bars)
        _append_semivariance_family(out, label=label, rs_plus=rs_plus, rs_minus=rs_minus, eps=eps)

    minute_of_day = out[timestamp_col].dt.hour * 60 + out[timestamp_col].dt.minute
    session_open = pd.Timestamp(session_open_time)
    rth_close = pd.Timestamp(rth_end_time)
    open_minutes = int(session_open.hour * 60 + session_open.minute)
    close_minutes = int(rth_close.hour * 60 + rth_close.minute)
    is_rth = minute_of_day.between(open_minutes, close_minutes, inclusive="both")
    rs_plus_session = rs_plus_bar.where(is_rth, 0.0).groupby(out[session_col], sort=False).cumsum()
    rs_minus_session = rs_minus_bar.where(is_rth, 0.0).groupby(out[session_col], sort=False).cumsum()
    _append_semivariance_family(out, label="session", rs_plus=rs_plus_session, rs_minus=rs_minus_session, eps=eps)

    return out


def add_directional_semivariance_context(
    df: pd.DataFrame,
    *,
    horizons: Iterable[str],
    side_col: str = "breakout_side",
) -> pd.DataFrame:
    """Map semivariance components to direction-specific adverse features."""
    out = df.copy()
    side = pd.Series(out[side_col], index=out.index, dtype="string").str.lower()
    long_mask = side.eq("long")
    short_mask = side.eq("short")

    for horizon in horizons:
        plus_col = f"rs_plus_{horizon}"
        minus_col = f"rs_minus_{horizon}"
        plus_share_col = f"rs_plus_share_{horizon}"
        minus_share_col = f"rs_minus_share_{horizon}"
        if plus_col not in out.columns or minus_col not in out.columns:
            raise ValueError(f"Missing semivariance columns for horizon '{horizon}'.")

        out[f"adverse_semivariance_{horizon}"] = np.where(
            long_mask,
            pd.to_numeric(out[minus_col], errors="coerce"),
            np.where(short_mask, pd.to_numeric(out[plus_col], errors="coerce"), np.nan),
        )
        out[f"supportive_semivariance_{horizon}"] = np.where(
            long_mask,
            pd.to_numeric(out[plus_col], errors="coerce"),
            np.where(short_mask, pd.to_numeric(out[minus_col], errors="coerce"), np.nan),
        )
        if plus_share_col in out.columns and minus_share_col in out.columns:
            out[f"adverse_share_{horizon}"] = np.where(
                long_mask,
                pd.to_numeric(out[minus_share_col], errors="coerce"),
                np.where(short_mask, pd.to_numeric(out[plus_share_col], errors="coerce"), np.nan),
            )
            out[f"supportive_share_{horizon}"] = np.where(
                long_mask,
                pd.to_numeric(out[plus_share_col], errors="coerce"),
                np.where(short_mask, pd.to_numeric(out[minus_share_col], errors="coerce"), np.nan),
            )

        plus_pct_col = f"rs_plus_pct_{horizon}"
        minus_pct_col = f"rs_minus_pct_{horizon}"
        if plus_pct_col in out.columns and minus_pct_col in out.columns:
            out[f"adverse_pct_{horizon}"] = np.where(
                long_mask,
                pd.to_numeric(out[minus_pct_col], errors="coerce"),
                np.where(short_mask, pd.to_numeric(out[plus_pct_col], errors="coerce"), np.nan),
            )
            out[f"supportive_pct_{horizon}"] = np.where(
                long_mask,
                pd.to_numeric(out[plus_pct_col], errors="coerce"),
                np.where(short_mask, pd.to_numeric(out[minus_pct_col], errors="coerce"), np.nan),
            )
    return out


def rolling_percentile_rank(
    values: pd.Series,
    *,
    lookback: int,
    min_history: int = 1,
) -> pd.Series:
    """Compute a strict no-lookahead percentile rank against prior observations only."""
    if lookback <= 0:
        raise ValueError("lookback must be strictly positive.")
    if min_history < 1:
        raise ValueError("min_history must be >= 1.")

    clean = pd.to_numeric(values, errors="coerce")
    array = clean.to_numpy(dtype=float)
    out = np.full(len(array), np.nan, dtype=float)

    for idx, value in enumerate(array):
        if not np.isfinite(value):
            continue
        start = max(0, idx - int(lookback))
        history = array[start:idx]
        history = history[np.isfinite(history)]
        if len(history) < int(min_history):
            continue
        lt = float(np.sum(history < value))
        eq = float(np.sum(history == value))
        out[idx] = (lt + 0.5 * eq) / float(len(history))

    return pd.Series(out, index=values.index, dtype=float)


def add_rolling_percentile_ranks(
    df: pd.DataFrame,
    *,
    columns: Iterable[str],
    lookback: int,
    min_history: int = 1,
    suffix: str = "_pct",
) -> pd.DataFrame:
    """Append rolling percentile ranks for a set of numeric columns."""
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            raise ValueError(f"Missing column '{column}' for rolling percentile ranking.")
        out[f"{column}{suffix}"] = rolling_percentile_rank(
            out[column],
            lookback=lookback,
            min_history=min_history,
        )
    return out
