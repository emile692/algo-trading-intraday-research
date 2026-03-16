"""Resampling helpers for OHLCV data."""

from __future__ import annotations

import pandas as pd


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV bars with proper aggregation mapping."""
    out = df.set_index("timestamp").sort_index()
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "open interest" in out.columns:
        agg["open interest"] = "last"

    result = out.resample(rule, label="left", closed="left").agg(agg)
    return result.dropna(subset=["open", "high", "low", "close"]).reset_index()
