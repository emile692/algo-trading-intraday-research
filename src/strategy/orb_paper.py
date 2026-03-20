"""Paper-style ORB strategy variant."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.time_utils import build_session_time


@dataclass
class ORBPaperExactStrategy:
    """Replicate the paper's first-candle bias and second-candle entry timing."""

    opening_time: str = "09:30:00"
    or_minutes: int = 5
    one_trade_per_day: bool = True
    time_exit: str = "16:00:00"
    target_multiple: float = 10.0
    account_size_usd: float | None = None
    risk_per_trade_pct: float | None = None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Signal on the first 5-minute candle so the backtester enters next open."""
        out = df.copy()
        out["signal"] = 0
        out["raw_signal"] = 0
        out["atr_filter_pass"] = True
        out["direction_filter_pass"] = True
        out["filter_pass"] = True
        out["filtered_out"] = False

        for _, session_df in out.groupby("session_date", sort=True):
            if session_df.empty:
                continue

            session_df = session_df.sort_values("timestamp")
            session_start = build_session_time(session_df["timestamp"].iloc[0], self.opening_time)
            session_end = session_start + pd.Timedelta(minutes=self.or_minutes)
            first_window = session_df[
                (session_df["timestamp"] >= session_start) & (session_df["timestamp"] < session_end)
            ]
            if first_window.empty:
                continue

            first_bar = first_window.iloc[0]
            signal = 0
            if first_bar["close"] > first_bar["open"]:
                signal = 1
            elif first_bar["close"] < first_bar["open"]:
                signal = -1

            if signal == 0:
                continue

            idx = first_bar.name
            out.at[idx, "raw_signal"] = signal
            out.at[idx, "signal"] = signal

            if self.one_trade_per_day:
                continue

        return out
