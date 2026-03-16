"""Return-based features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_simple_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Add simple percentage returns."""
    out = df.copy()
    out["ret_simple"] = out[price_col].pct_change()
    return out


def add_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Add log returns."""
    out = df.copy()
    out["ret_log"] = np.log(out[price_col] / out[price_col].shift(1))
    return out
