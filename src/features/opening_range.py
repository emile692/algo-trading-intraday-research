"""Opening range feature engineering."""

from __future__ import annotations

import pandas as pd


def compute_opening_range(df: pd.DataFrame, or_minutes: int = 30) -> pd.DataFrame:
    """Compute opening range high/low/width/midpoint per session for first N minutes."""
    out = df.copy()
    out["session_date"] = out["timestamp"].dt.date

    range_frames = []
    for session_date, group in out.groupby("session_date", sort=True):
        start = group["timestamp"].min()
        end = start + pd.Timedelta(minutes=or_minutes)
        opening_window = group[(group["timestamp"] >= start) & (group["timestamp"] < end)]
        if opening_window.empty:
            continue

        high = opening_window["high"].max()
        low = opening_window["low"].min()
        width = high - low
        midpoint = (high + low) / 2.0
        range_frames.append(
            {
                "session_date": session_date,
                "or_high": high,
                "or_low": low,
                "or_width": width,
                "or_midpoint": midpoint,
            }
        )

    or_df = pd.DataFrame(range_frames)
    return out.merge(or_df, on="session_date", how="left")
