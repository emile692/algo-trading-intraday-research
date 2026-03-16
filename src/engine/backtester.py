"""Simple ORB backtesting engine."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from src.config.settings import DEFAULT_TICK_VALUE_USD, NQ_TICK_SIZE
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


def _compute_risk_sized_quantity(
    direction: int,
    entry_price: float,
    stop_price: float,
    execution_model: ExecutionModel,
    tick_value_usd: float,
    risk_budget_usd: float,
) -> tuple[int, float]:
    """Return integer quantity and risk per contract based on the stop distance."""
    stop_exit_price = execution_model.apply_slippage(stop_price, direction, is_entry=False)
    stop_pnl_points = (stop_exit_price - entry_price) * direction
    stop_pnl_ticks = stop_pnl_points / NQ_TICK_SIZE
    gross_loss_usd = abs(stop_pnl_ticks * tick_value_usd)
    fees_per_contract = execution_model.round_trip_fees(quantity=1)
    risk_per_contract_usd = gross_loss_usd + fees_per_contract

    if risk_per_contract_usd <= 0:
        return 0, 0.0

    quantity = int(risk_budget_usd / risk_per_contract_usd)
    return quantity, risk_per_contract_usd


def run_backtest(
    df: pd.DataFrame,
    execution_model: ExecutionModel,
    tick_value_usd: float = DEFAULT_TICK_VALUE_USD,
    time_exit: str = "15:55",
    stop_multiple: float = 1.0,
    target_multiple: float = 1.5,
    account_size_usd: float | None = None,
    risk_per_trade_pct: float | None = None,
) -> pd.DataFrame:
    """Run a straightforward signal-driven single-position backtest."""
    _validate_risk_inputs(account_size_usd, risk_per_trade_pct)

    trades = []
    trade_id = 1
    force_exit_time = dt.time.fromisoformat(time_exit)
    use_risk_sizing = account_size_usd is not None and risk_per_trade_pct is not None

    for session_date, session_df in df.groupby("session_date", sort=True):
        position = 0
        quantity = 0
        entry_price = stop_price = target_price = None
        entry_time = None
        risk_budget_usd = risk_per_contract_usd = actual_risk_usd = None

        for _, row in session_df.iterrows():
            current_time = row["timestamp"].time()

            if position == 0 and row.get("signal", 0) != 0 and row.get("or_width", 0) > 0:
                direction = int(row["signal"])
                raw_entry = row["close"]
                candidate_entry_price = execution_model.apply_slippage(raw_entry, direction, is_entry=True)
                candidate_entry_time = row["timestamp"]
                width = float(row["or_width"])
                if direction == 1:
                    candidate_stop_price = candidate_entry_price - stop_multiple * width
                    candidate_target_price = candidate_entry_price + target_multiple * width
                else:
                    candidate_stop_price = candidate_entry_price + stop_multiple * width
                    candidate_target_price = candidate_entry_price - target_multiple * width

                candidate_quantity = 1
                candidate_risk_budget_usd = None
                candidate_risk_per_contract_usd = None
                candidate_actual_risk_usd = None

                if use_risk_sizing:
                    candidate_risk_budget_usd = account_size_usd * (risk_per_trade_pct / 100.0)
                    candidate_quantity, candidate_risk_per_contract_usd = _compute_risk_sized_quantity(
                        direction=direction,
                        entry_price=candidate_entry_price,
                        stop_price=candidate_stop_price,
                        execution_model=execution_model,
                        tick_value_usd=tick_value_usd,
                        risk_budget_usd=candidate_risk_budget_usd,
                    )
                    if candidate_quantity < 1:
                        continue
                    candidate_actual_risk_usd = candidate_quantity * candidate_risk_per_contract_usd

                position = direction
                quantity = candidate_quantity
                entry_price = candidate_entry_price
                stop_price = candidate_stop_price
                target_price = candidate_target_price
                entry_time = candidate_entry_time
                risk_budget_usd = candidate_risk_budget_usd
                risk_per_contract_usd = candidate_risk_per_contract_usd
                actual_risk_usd = candidate_actual_risk_usd
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
                    pnl_usd = pnl_ticks * tick_value_usd * quantity
                    fees = execution_model.round_trip_fees(quantity=quantity)
                    net_pnl = pnl_usd - fees

                    trades.append(
                        trade_to_record(
                            trade_id,
                            {
                                "session_date": session_date,
                                "direction": "long" if position == 1 else "short",
                                "quantity": quantity,
                                "entry_time": entry_time,
                                "entry_price": entry_price,
                                "stop_price": stop_price,
                                "target_price": target_price,
                                "exit_time": row["timestamp"],
                                "exit_price": exit_price,
                                "exit_reason": exit_reason,
                                "account_size_usd": account_size_usd if use_risk_sizing else None,
                                "risk_per_trade_pct": risk_per_trade_pct if use_risk_sizing else None,
                                "risk_budget_usd": risk_budget_usd,
                                "risk_per_contract_usd": risk_per_contract_usd,
                                "actual_risk_usd": actual_risk_usd,
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
                    quantity = 0
                    entry_price = stop_price = target_price = None
                    entry_time = None
                    risk_budget_usd = risk_per_contract_usd = actual_risk_usd = None

    if not trades:
        return empty_trade_log()

    return pd.DataFrame(trades)
