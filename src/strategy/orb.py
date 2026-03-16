"""Opening Range Breakout strategy."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.settings import NQ_TICK_SIZE


@dataclass
class ORBStrategy:
    """Simple ORB signal generator."""

    or_minutes: int = 30
    direction: str = "both"
    one_trade_per_day: bool = True
    entry_buffer_ticks: int = 0
    stop_multiple: float = 1.0
    target_multiple: float = 1.5
    time_exit: str = "15:55"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ORB entry signals based on OR high/low breaks."""
        out = df.copy()
        out["signal"] = 0
        buffer = self.entry_buffer_ticks * NQ_TICK_SIZE

        for session_date, group in out.groupby("session_date", sort=True):
            has_trade = False
            for idx in group.index:
                row = out.loc[idx]
                if pd.isna(row.get("or_high")) or pd.isna(row.get("or_low")):
                    continue

                long_break = row["high"] > row["or_high"] + buffer
                short_break = row["low"] < row["or_low"] - buffer

                signal = 0
                if self.direction in ("both", "long") and long_break:
                    signal = 1
                if self.direction in ("both", "short") and short_break:
                    signal = -1 if signal == 0 else signal

                if signal != 0:
                    out.at[idx, "signal"] = signal
                    has_trade = True
                    if self.one_trade_per_day and has_trade:
                        break

        return out
