"""Intraday calendar and bar-position features."""

from __future__ import annotations

import pandas as pd


def add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add minute-of-day, weekday, and bar rank in session."""
    out = df.copy()
    out["session_date"] = out["timestamp"].dt.date
    out["minute_of_day"] = out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute
    out["weekday"] = out["timestamp"].dt.dayofweek
    out["bar_in_session"] = out.groupby("session_date").cumcount() + 1
    return out
