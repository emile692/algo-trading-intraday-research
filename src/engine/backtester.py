"""Simple ORB backtesting engine."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from src.config.settings import DEFAULT_TICK_VALUE_USD
from src.engine.execution_model import ExecutionModel
from src.engine.trade_log import empty_trade_log, trade_to_record


def _validate_risk_inputs(account_size_usd: float | None, risk_per_trade_pct: float | None) -> None:
    """Validate optional risk-based sizing inputs."""
    if account_size_usd is None and risk_per_trade_pct is None:
        return
    if account_size_usd is None or risk_per_trade_pct is None:
        raise ValueError("account_size_usd and risk_per_trade_pct must be provided together.")
    if account_size_usd <= 0:
        raise ValueError("account_size_usd must be strictly positive.")
    if risk_per_trade_pct <= 0 or risk_per_trade_pct > 100:
        raise ValueError("risk_per_trade_pct must be in the (0, 100] interval.")


def _validate_backtest_inputs(
    stop_buffer_ticks: int,
    tick_size: float,
    max_leverage: float | None,
    account_size_usd: float | None,
) -> None:
    """Validate non-risk backtest inputs."""
    if stop_buffer_ticks < 0:
        raise ValueError("stop_buffer_ticks must be non-negative.")
    if tick_size <= 0:
        raise ValueError("tick_size must be strictly positive.")
    if max_leverage is not None and max_leverage <= 0:
        raise ValueError("max_leverage must be strictly positive when provided.")
    if max_leverage is not None and account_size_usd is None:
        raise ValueError("account_size_usd must be provided when max_leverage is enabled.")


def _compute_risk_per_contract_usd(
    direction: int,
    entry_price: float,
    stop_price: float,
    execution_model: ExecutionModel,
    tick_value_usd: float,
) -> float:
    """Return per-contract risk in USD, including round-trip fees."""
    stop_exit_price = execution_model.apply_slippage(stop_price, direction, is_entry=False)
    stop_pnl_points = (stop_exit_price - entry_price) * direction
    stop_pnl_ticks = stop_pnl_points / execution_model.tick_size
    gross_loss_usd = abs(stop_pnl_ticks * tick_value_usd)
    fees_per_contract = execution_model.round_trip_fees(quantity=1)
    return gross_loss_usd + fees_per_contract


def _apply_leverage_cap(
    quantity: int,
    entry_price: float,
    account_size_usd: float | None,
    point_value_usd: float,
    max_leverage: float | None,
) -> int:
    """Cap quantity by a notional leverage constraint if configured."""
    if max_leverage is None or account_size_usd is None or point_value_usd <= 0:
        return quantity

    max_notional_usd = account_size_usd * max_leverage
    notional_per_contract_usd = entry_price * point_value_usd
    if notional_per_contract_usd <= 0:
        return 0

    leverage_capped_quantity = int(max_notional_usd / notional_per_contract_usd)
    return min(quantity, leverage_capped_quantity)


def run_backtest(
    df: pd.DataFrame,
    execution_model: ExecutionModel,
    tick_value_usd: float = DEFAULT_TICK_VALUE_USD,
    point_value_usd: float | None = None,
    time_exit: str = "15:55",
    stop_buffer_ticks: int = 0,
    target_multiple: float = 1.5,
    account_size_usd: float | None = None,
    risk_per_trade_pct: float | None = None,
    entry_on_next_open: bool = True,
    max_leverage: float | None = None,
) -> pd.DataFrame:
    """Run a straightforward signal-driven single-position backtest."""
    _validate_risk_inputs(account_size_usd, risk_per_trade_pct)
    _validate_backtest_inputs(
        stop_buffer_ticks=stop_buffer_ticks,
        tick_size=execution_model.tick_size,
        max_leverage=max_leverage,
        account_size_usd=account_size_usd,
    )

    trades = []
    trade_id = 1
    force_exit_time = dt.time.fromisoformat(time_exit)
    use_risk_sizing = account_size_usd is not None and risk_per_trade_pct is not None
    stop_buffer_points = stop_buffer_ticks * execution_model.tick_size
    point_value = point_value_usd if point_value_usd is not None else tick_value_usd / execution_model.tick_size

    for session_date, session_df in df.groupby("session_date", sort=True):
        session_df = session_df.sort_values("timestamp").reset_index(drop=True)
        signal_indices = session_df.index[session_df["signal"].fillna(0).ne(0)].tolist()
        if not signal_indices:
            continue

        next_search_index = 0

        for signal_idx in signal_indices:
            if signal_idx < next_search_index:
                continue

            row = session_df.loc[signal_idx]
            direction = int(row["signal"])
            or_high = row.get("or_high")
            or_low = row.get("or_low")
            if pd.isna(or_high) or pd.isna(or_low):
                continue

            entry_idx = signal_idx + 1 if entry_on_next_open else signal_idx
            if entry_idx >= len(session_df):
                continue

            entry_row = session_df.loc[entry_idx]
            raw_entry = entry_row["open"] if entry_on_next_open else row["close"]
            entry_price = execution_model.apply_slippage(raw_entry, direction, is_entry=True)
            entry_time = entry_row["timestamp"] if entry_on_next_open else row["timestamp"]

            if direction == 1:
                stop_price = float(or_low) - stop_buffer_points
            else:
                stop_price = float(or_high) + stop_buffer_points

            risk_points = (entry_price - stop_price) * direction
            if risk_points <= 0:
                continue

            target_price = entry_price + direction * target_multiple * risk_points
            risk_per_contract_usd = _compute_risk_per_contract_usd(
                direction=direction,
                entry_price=entry_price,
                stop_price=stop_price,
                execution_model=execution_model,
                tick_value_usd=tick_value_usd,
            )
            if risk_per_contract_usd <= 0:
                continue

            quantity = 1
            risk_budget_usd = None
            if use_risk_sizing:
                risk_budget_usd = account_size_usd * (risk_per_trade_pct / 100.0)
                quantity = int(risk_budget_usd / risk_per_contract_usd)
                if quantity < 1:
                    continue

            quantity = _apply_leverage_cap(
                quantity=quantity,
                entry_price=entry_price,
                account_size_usd=account_size_usd,
                point_value_usd=point_value,
                max_leverage=max_leverage,
            )
            if quantity < 1:
                continue

            trade_risk_usd = quantity * risk_per_contract_usd
            actual_risk_usd = trade_risk_usd if use_risk_sizing else None
            notional_usd = quantity * entry_price * point_value
            leverage_used = notional_usd / account_size_usd if account_size_usd is not None else None

            exit_scan_start = entry_idx if entry_on_next_open else entry_idx + 1
            if exit_scan_start >= len(session_df):
                continue

            exit_slice = session_df.loc[exit_scan_start:].copy()
            exit_reason = None
            raw_exit_price = None
            exit_idx = None

            if direction == 1:
                hit_mask = (exit_slice["low"] <= stop_price) | (exit_slice["high"] >= target_price)
            else:
                hit_mask = (exit_slice["high"] >= stop_price) | (exit_slice["low"] <= target_price)

            if hit_mask.any():
                exit_idx = hit_mask[hit_mask].index[0]
                exit_row = session_df.loc[exit_idx]
                if direction == 1:
                    if exit_row["low"] <= stop_price:
                        exit_reason = "stop"
                        raw_exit_price = stop_price
                    else:
                        exit_reason = "target"
                        raw_exit_price = target_price
                else:
                    if exit_row["high"] >= stop_price:
                        exit_reason = "stop"
                        raw_exit_price = stop_price
                    else:
                        exit_reason = "target"
                        raw_exit_price = target_price
            else:
                time_exit_mask = exit_slice["timestamp"].dt.time >= force_exit_time
                if time_exit_mask.any():
                    exit_idx = time_exit_mask[time_exit_mask].index[0]
                    exit_reason = "time_exit"
                else:
                    exit_idx = exit_slice.index[-1]
                    exit_reason = "eod_exit"
                raw_exit_price = float(session_df.loc[exit_idx, "close"])

            exit_row = session_df.loc[exit_idx]
            exit_price = execution_model.apply_slippage(raw_exit_price, direction, is_entry=False)
            pnl_points = (exit_price - entry_price) * direction
            pnl_ticks = pnl_points / execution_model.tick_size
            pnl_usd = pnl_ticks * tick_value_usd * quantity
            fees = execution_model.round_trip_fees(quantity=quantity)
            net_pnl = pnl_usd - fees

            trades.append(
                trade_to_record(
                    trade_id,
                    {
                        "session_date": session_date,
                        "direction": "long" if direction == 1 else "short",
                        "quantity": quantity,
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "stop_price": stop_price,
                        "target_price": target_price,
                        "exit_time": exit_row["timestamp"],
                        "exit_price": exit_price,
                        "exit_reason": exit_reason,
                        "account_size_usd": account_size_usd if account_size_usd is not None else None,
                        "risk_per_trade_pct": risk_per_trade_pct if use_risk_sizing else None,
                        "risk_budget_usd": risk_budget_usd,
                        "risk_per_contract_usd": risk_per_contract_usd,
                        "actual_risk_usd": actual_risk_usd,
                        "trade_risk_usd": trade_risk_usd,
                        "notional_usd": notional_usd,
                        "leverage_used": leverage_used,
                        "pnl_points": pnl_points,
                        "pnl_ticks": pnl_ticks,
                        "pnl_usd": pnl_usd,
                        "fees": fees,
                        "net_pnl_usd": net_pnl,
                    },
                )
            )
            trade_id += 1
            next_search_index = int(exit_idx) + 1

    if not trades:
        return empty_trade_log()

    return pd.DataFrame(trades)
