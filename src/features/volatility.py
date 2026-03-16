"""Volatility-related features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_rolling_std(df: pd.DataFrame, window: int = 20, price_col: str = "close") -> pd.DataFrame:
    """Add rolling standard deviation of returns."""
    out = df.copy()
    returns = out[price_col].pct_change()
    out[f"vol_std_{window}"] = returns.rolling(window).std()
    return out


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Add simplified ATR feature."""
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = np.maximum(
        out["high"] - out["low"],
        np.maximum((out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()),
    )
    out[f"atr_{window}"] = tr.rolling(window).mean()
    return out
