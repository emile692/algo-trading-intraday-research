"""Session filtering and grouping utilities."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from src.config.settings import RTH_END, RTH_START


def _session_mask(timestamps: pd.Series, start_time: str, end_time: str) -> pd.Series:
    """Build boolean mask for a time range, handling overnight windows."""
    times = timestamps.dt.time
    start = dt.time.fromisoformat(start_time)
    end = dt.time.fromisoformat(end_time)
    if start <= end:
        return (times >= start) & (times <= end)
    return (times >= start) | (times <= end)


def filter_session(df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    """Filter bars between start_time and end_time (inclusive), local dataframe timezone."""
    mask = _session_mask(df["timestamp"], start_time, end_time)
    return df.loc[mask].reset_index(drop=True)


def extract_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Extract regular trading hours."""
    return filter_session(df, RTH_START, RTH_END)


def extract_eth(df: pd.DataFrame) -> pd.DataFrame:
    """Extract extended trading hours as complement of RTH."""
    mask = _session_mask(df["timestamp"], RTH_START, RTH_END)
    return df.loc[~mask].reset_index(drop=True)


def add_session_date(df: pd.DataFrame) -> pd.DataFrame:
    """Add session_date from local timestamp date for robust session grouping."""
    out = df.copy()
    out["session_date"] = out["timestamp"].dt.date
    return out
