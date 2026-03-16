"""Simple ORB backtesting engine."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from src.config.settings import DEFAULT_TICK_VALUE_USD, NQ_TICK_SIZE
from src.engine.execution_model import ExecutionModel
from src.engine.trade_log import empty_trade_log, trade_to_record


def run_backtest(
    df: pd.DataFrame,
    execution_model: ExecutionModel,
    tick_value_usd: float = DEFAULT_TICK_VALUE_USD,
    time_exit: str = "15:55",
    stop_multiple: float = 1.0,
    target_multiple: float = 1.5,
) -> pd.DataFrame:
    """Run a straightforward signal-driven single-position backtest."""
    trades = []
    trade_id = 1
    force_exit_time = dt.time.fromisoformat(time_exit)

    for session_date, session_df in df.groupby("session_date", sort=True):
        position = 0
        entry_price = stop_price = target_price = None
        entry_time = None

        for _, row in session_df.iterrows():
            current_time = row["timestamp"].time()

            if position == 0 and row.get("signal", 0) != 0 and row.get("or_width", 0) > 0:
                position = int(row["signal"])
                raw_entry = row["close"]
                entry_price = execution_model.apply_slippage(raw_entry, position, is_entry=True)
                entry_time = row["timestamp"]
                width = float(row["or_width"])
                if position == 1:
                    stop_price = entry_price - stop_multiple * width
                    target_price = entry_price + target_multiple * width
                else:
                    stop_price = entry_price + stop_multiple * width
                    target_price = entry_price - target_multiple * width
                continue

            if position != 0:
                exit_reason = None
                raw_exit_price = None

                if position == 1:
                    if row["low"] <= stop_price:
                        exit_reason = "stop"
                        raw_exit_price = stop_price
                    elif row["high"] >= target_price:
                        exit_reason = "target"
                        raw_exit_price = target_price
                else:
                    if row["high"] >= stop_price:
                        exit_reason = "stop"
                        raw_exit_price = stop_price
                    elif row["low"] <= target_price:
                        exit_reason = "target"
                        raw_exit_price = target_price

                if exit_reason is None and current_time >= force_exit_time:
                    exit_reason = "time_exit"
                    raw_exit_price = row["close"]

                if exit_reason is not None and raw_exit_price is not None:
                    exit_price = execution_model.apply_slippage(raw_exit_price, position, is_entry=False)
                    pnl_points = (exit_price - entry_price) * position
                    pnl_ticks = pnl_points / NQ_TICK_SIZE
                    pnl_usd = pnl_ticks * tick_value_usd
                    fees = execution_model.round_trip_fees()
                    net_pnl = pnl_usd - fees

                    trades.append(
                        trade_to_record(
                            trade_id,
                            {
                                "session_date": session_date,
                                "direction": "long" if position == 1 else "short",
                                "entry_time": entry_time,
                                "entry_price": entry_price,
                                "stop_price": stop_price,
                                "target_price": target_price,
                                "exit_time": row["timestamp"],
                                "exit_price": exit_price,
                                "exit_reason": exit_reason,
                                "pnl_points": pnl_points,
                                "pnl_ticks": pnl_ticks,
                                "pnl_usd": pnl_usd,
                                "fees": fees,
                                "net_pnl_usd": net_pnl,
                            },
                        )
                    )
                    trade_id += 1
                    position = 0

    if not trades:
        return empty_trade_log()

    return pd.DataFrame(trades)
