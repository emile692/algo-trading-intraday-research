"""Small pandas helper functions."""

from __future__ import annotations

import pandas as pd


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError if columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
