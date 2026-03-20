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


def add_session_vwap(df: pd.DataFrame, price_mode: str = "typical") -> pd.DataFrame:
    """Add an intraday VWAP column that resets each session."""
    out = df.copy()
    if "session_date" not in out.columns:
        out["session_date"] = out["timestamp"].dt.date

    if price_mode == "close":
        price = out["close"]
    elif price_mode == "typical":
        price = (out["high"] + out["low"] + out["close"]) / 3.0
    else:
        raise ValueError("price_mode must be 'close' or 'typical'.")

    pv = price * out["volume"].fillna(0.0)
    cumulative_pv = pv.groupby(out["session_date"]).cumsum()
    cumulative_volume = out["volume"].fillna(0.0).groupby(out["session_date"]).cumsum()
    out["session_vwap"] = np.where(cumulative_volume > 0, cumulative_pv / cumulative_volume, np.nan)
    return out


def add_ema(df: pd.DataFrame, window: int, price_col: str = "close") -> pd.DataFrame:
    """Add an exponential moving average column."""
    out = df.copy()
    out[f"ema_{window}"] = out[price_col].ewm(span=window, adjust=False).mean()
    return out
