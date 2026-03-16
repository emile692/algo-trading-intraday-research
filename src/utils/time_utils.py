"""Time helpers."""

from __future__ import annotations

import pandas as pd


def ensure_timezone(series: pd.Series, timezone: str) -> pd.Series:
    """Ensure datetime series is timezone-aware in the target timezone."""
    dt_series = pd.to_datetime(series)
    if dt_series.dt.tz is None:
        return dt_series.dt.tz_localize(timezone)
    return dt_series.dt.tz_convert(timezone)
