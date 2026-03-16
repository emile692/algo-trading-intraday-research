"""Common feature utilities."""

from __future__ import annotations

import pandas as pd


def with_feature(df: pd.DataFrame, name: str, values: pd.Series) -> pd.DataFrame:
    """Return dataframe copy with a new feature column."""
    out = df.copy()
    out[name] = values
    return out
