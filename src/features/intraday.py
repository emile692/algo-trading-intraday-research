"""Intraday calendar and bar-position features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add minute-of-day, weekday, and bar rank in session."""
    out = df.copy()
    out["session_date"] = out["timestamp"].dt.date
    out["minute_of_day"] = out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute
    out["weekday"] = out["timestamp"].dt.dayofweek
    out["bar_in_session"] = out.groupby("session_date").cumcount() + 1
    return out


def add_session_vwap(
    df: pd.DataFrame,
    price_mode: str = "typical",
    price_volume_col: str | None = None,
) -> pd.DataFrame:
    """Add an intraday VWAP column that resets each session."""
    out = df.copy()
    if "session_date" not in out.columns:
        out["session_date"] = out["timestamp"].dt.date

    volume = out["volume"].fillna(0.0)

    if price_volume_col is not None:
        if price_volume_col not in out.columns:
            raise ValueError(f"Missing precomputed price-volume column '{price_volume_col}'.")
        pv = pd.to_numeric(out[price_volume_col], errors="coerce").fillna(0.0)
    elif price_mode == "close":
        price = out["close"]
        pv = price * volume
    elif price_mode == "typical":
        price = (out["high"] + out["low"] + out["close"]) / 3.0
        pv = price * volume
    else:
        raise ValueError("price_mode must be 'close' or 'typical'.")

    cumulative_pv = pv.groupby(out["session_date"]).cumsum()
    cumulative_volume = volume.groupby(out["session_date"]).cumsum()
    out["session_vwap"] = np.where(cumulative_volume > 0, cumulative_pv / cumulative_volume, np.nan)
    return out


def add_ema(df: pd.DataFrame, window: int, price_col: str = "close") -> pd.DataFrame:
    """Add an exponential moving average column."""
    out = df.copy()
    out[f"ema_{window}"] = out[price_col].ewm(span=window, adjust=False).mean()
    return out

def add_continuous_session_vwap(
    df: pd.DataFrame,
    price_mode: str = "typical",
    session_start_hour: int = 18,
    tz: str = "America/New_York",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Add a futures-style continuous session VWAP.

    The VWAP resets at `session_start_hour` in local timezone, so it includes:
    - overnight
    - premarket
    - regular US session

    Example for CME equity index futures:
    session_start_hour = 18 means the session runs approximately
    from 18:00 ET to 17:59:59 ET next day.
    """
    out = df.copy()

    ts = out[timestamp_col]
    if ts.dt.tz is None:
        ts_local = ts.dt.tz_localize(tz)
    else:
        ts_local = ts.dt.tz_convert(tz)
    out["_ts_local"] = ts_local

    # Build continuous session date by shifting all bars from session_start_hour or later to next calendar day.
    session_date = out["_ts_local"].dt.date
    mask_next_session = out["_ts_local"].dt.hour >= session_start_hour
    session_date = session_date.where(~mask_next_session, (out["_ts_local"] + pd.Timedelta(days=1)).dt.date)
    out["continuous_session_date"] = session_date

    if price_mode == "close":
        price = out["close"]
    elif price_mode == "typical":
        price = (out["high"] + out["low"] + out["close"]) / 3.0
    else:
        raise ValueError("price_mode must be 'close' or 'typical'.")

    volume = out["volume"].fillna(0.0)
    pv = price * volume

    cumulative_pv = pv.groupby(out["continuous_session_date"]).cumsum()
    cumulative_volume = volume.groupby(out["continuous_session_date"]).cumsum()

    out["continuous_session_vwap"] = np.where(
        cumulative_volume > 0,
        cumulative_pv / cumulative_volume,
        np.nan,
    )

    out = out.drop(columns=["_ts_local"])
    return out
