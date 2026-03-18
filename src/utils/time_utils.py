"""Time helpers."""

from __future__ import annotations

import datetime as dt

import pandas as pd


def ensure_timezone(series: pd.Series, timezone: str) -> pd.Series:
    """Ensure datetime series is timezone-aware in the target timezone."""
    dt_series = pd.to_datetime(series)
    if dt_series.dt.tz is None:
        return dt_series.dt.tz_localize(timezone)
    return dt_series.dt.tz_convert(timezone)


def build_session_time(session_timestamp: pd.Timestamp, clock_time: str) -> pd.Timestamp:
    """Return the session date at the requested clock time, preserving timezone."""
    time_value = dt.time.fromisoformat(clock_time)
    timestamp = pd.Timestamp(session_timestamp)
    return timestamp.replace(
        hour=time_value.hour,
        minute=time_value.minute,
        second=time_value.second,
        microsecond=0,
        nanosecond=0,
    )
