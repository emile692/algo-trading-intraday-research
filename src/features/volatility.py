"""Volatility-related features."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def add_rolling_std(df: pd.DataFrame, window: int = 20, price_col: str = "close") -> pd.DataFrame:
    """Add rolling standard deviation of returns."""
    out = df.copy()
    returns = out[price_col].pct_change()
    out[f"vol_std_{window}"] = returns.rolling(window).std()
    return out


def _normalize_windows(window: int | Iterable[int]) -> list[int]:
    if isinstance(window, (int, np.integer)):
        windows = [int(window)]
    else:
        windows = [int(value) for value in window]

    clean = sorted({value for value in windows if value > 0})
    if not clean:
        raise ValueError("ATR window must contain at least one positive integer.")
    return clean


def add_atr(df: pd.DataFrame, window: int | Iterable[int] = 14) -> pd.DataFrame:
    """Add simplified ATR feature(s) for one or many rolling windows."""
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = np.maximum(
        out["high"] - out["low"],
        np.maximum((out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()),
    )
    for w in _normalize_windows(window):
        out[f"atr_{w}"] = tr.rolling(w).mean()
    return out
