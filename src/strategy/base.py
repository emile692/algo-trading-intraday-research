"""Strategy interface definitions."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    """Minimal strategy protocol."""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return dataframe with a 'signal' column (-1, 0, +1)."""
        ...
