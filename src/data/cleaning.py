"""Cleaning and normalization helpers."""

from __future__ import annotations

import pandas as pd


def clean_ohlcv(df: pd.DataFrame, drop_duplicate_timestamps: bool = True) -> pd.DataFrame:
    """Sort, type-cast numeric columns, and optionally remove duplicate timestamps."""
    out = df.copy()
    out = out.sort_values("timestamp").reset_index(drop=True)

    if drop_duplicate_timestamps:
        out = out.drop_duplicates(subset=["timestamp"], keep="last")

    numeric_columns = [c for c in ["open", "high", "low", "close", "volume", "open interest"] if c in out.columns]
    for col in numeric_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    return out.reset_index(drop=True)
