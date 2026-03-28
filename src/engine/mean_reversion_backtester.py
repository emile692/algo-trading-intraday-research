"""Discrete backtester for intraday mean reversion variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config.mean_reversion_campaign import MeanReversionVariantConfig
from src.engine.execution_model import ExecutionModel
from src.engine.trade_log import empty_trade_log, trade_to_record
from src.engine.vwap_backtester import InstrumentDetails


@dataclass(frozen=True)
class MeanReversionBacktestResult:
    """Backtest outputs reused by screening, validation, and portfolio phases."""

    trades: pd.DataFrame
    bar_results: pd.DataFrame
    daily_results: pd.DataFrame


def _trade_log_frame(trades: list[dict[str, Any]]) -> pd.DataFrame:
    if not trades:
        frame = empty_trade_log()
        extra_columns = {
            "variant_name": object,
            "family": object,
            "symbol": object,
            "timeframe": object,
            "entry_signal_time": object,
            "bars_held": float,
            "holding_minutes": float,
            "gross_pnl_usd": float,
            "partial_exit_taken": object,
            "partial_exit_time": object,
            "partial_exit_price": float,
            "partial_exit_quantity": float,
        }
        for column, dtype in extra_columns.items():
            frame[column] = pd.Series(dtype=dtype)
        return frame
    return pd.DataFrame(trades)


def _trade_risk_values(
    execution_model: ExecutionModel,
    instrument: InstrumentDetails,
    entry_price: float,
    stop_price: float | None,
    quantity: int,
) -> tuple[float | None, float | None]:
    if stop_price is None or quantity <= 0:
        return None, None
    distance = abs(float(entry_price) - float(stop_price))
    if distance <= 0:
        return None, None
    risk_per_contract = distance * float(instrument.point_value_usd) + execution_model.round_trip_fees(quantity=1)
    if risk_per_contract <= 0:
        return None, None
    return risk_per_contract, risk_per_contract * int(quantity)


def _init_daily_row(session_date: object) -> dict[str, Any]:
    return {
        "session_date": session_date,
        "daily_pnl_usd": 0.0,
        "daily_gross_pnl_usd": 0.0,
        "daily_fees_usd": 0.0,
        "daily_trade_count": 0,
        "daily_loss_count": 0,
        "daily_stop_breached": False,
        "trading_halted": False,
    }


def _finalize_daily_results(daily_rows: list[dict[str, Any]], initial_capital: float) -> pd.DataFrame:
    if not daily_rows:
        return pd.DataFrame(
            columns=[
                "session_date",
                "daily_pnl_usd",
                "daily_gross_pnl_usd",
                "daily_fees_usd",
                "daily_trade_count",
                "daily_loss_count",
                "daily_stop_breached",
                "trading_halted",
                "equity",
                "peak_equity",
                "drawdown_usd",
                "green_day",
                "weekday",
            ]
        )
    daily = pd.DataFrame(daily_rows).sort_values("session_date").reset_index(drop=True)
    daily["equity"] = float(initial_capital) + pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").cumsum()
    daily["peak_equity"] = daily["equity"].cummax()
    daily["drawdown_usd"] = daily["equity"] - daily["peak_equity"]
    daily["green_day"] = daily["daily_pnl_usd"] > 0
    daily["weekday"] = pd.to_datetime(daily["session_date"]).dt.day_name()
    return daily


def _close_trade_record(
    trade_id: int,
    variant: MeanReversionVariantConfig,
    instrument: InstrumentDetails,
    execution_model: ExecutionModel,
    open_trade: dict[str, Any],
    raw_exit_price: float,
    exit_time: pd.Timestamp,
    exit_reason: str,
    account_size_usd: float,
) -> dict[str, Any]:
    direction = int(open_trade["direction_int"])
    quantity_initial = int(open_trade["initial_quantity"])
    quantity_remaining = int(open_trade["quantity_remaining"])
    exit_price = execution_model.apply_slippage(raw_exit_price, direction, is_entry=False)
    final_gross_pnl_usd = (
        (exit_price - float(open_trade["entry_price"]))
        * direction
        * float(instrument.point_value_usd)
        * quantity_remaining
    )
    gross_pnl_usd = float(open_trade["partial_gross_pnl_usd"]) + final_gross_pnl_usd
    fees = execution_model.round_trip_fees(quantity=quantity_initial)
    net_pnl_usd = gross_pnl_usd - fees
    avg_points = gross_pnl_usd / max(float(instrument.point_value_usd) * max(quantity_initial, 1), 1.0)
    risk_per_contract_usd, trade_risk_usd = _trade_risk_values(
        execution_model=execution_model,
        instrument=instrument,
        entry_price=float(open_trade["entry_price"]),
        stop_price=open_trade.get("stop_price"),
        quantity=quantity_initial,
    )
    record = trade_to_record(
        trade_id,
        {
            "session_date": open_trade["session_date"],
            "direction": "long" if direction == 1 else "short",
            "quantity": quantity_initial,
            "entry_time": open_trade["entry_time"],
            "entry_price": float(open_trade["entry_price"]),
            "stop_price": open_trade.get("stop_price"),
            "target_price": open_trade.get("target_price"),
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "account_size_usd": float(account_size_usd),
            "risk_per_trade_pct": None,
            "risk_budget_usd": None,
            "risk_per_contract_usd": risk_per_contract_usd,
            "actual_risk_usd": trade_risk_usd,
            "trade_risk_usd": trade_risk_usd,
            "notional_usd": quantity_initial * float(open_trade["entry_price"]) * float(instrument.point_value_usd),
            "leverage_used": (
                quantity_initial * float(open_trade["entry_price"]) * float(instrument.point_value_usd) / float(account_size_usd)
                if account_size_usd > 0
                else None
            ),
            "pnl_points": avg_points,
            "pnl_ticks": avg_points / float(instrument.tick_size) if instrument.tick_size > 0 else np.nan,
            "pnl_usd": gross_pnl_usd,
            "fees": fees,
            "net_pnl_usd": net_pnl_usd,
        },
    )
    record.update(
        {
            "variant_name": variant.name,
            "family": variant.family,
            "symbol": variant.symbol,
            "timeframe": variant.timeframe,
            "entry_signal_time": open_trade["entry_signal_time"],
            "bars_held": open_trade["bars_held"],
            "holding_minutes": float((exit_time - open_trade["entry_time"]).total_seconds() / 60.0),
            "gross_pnl_usd": gross_pnl_usd,
            "partial_exit_taken": bool(open_trade["partial_exit_taken"]),
            "partial_exit_time": open_trade["partial_exit_time"],
            "partial_exit_price": open_trade["partial_exit_price"],
            "partial_exit_quantity": open_trade["partial_exit_quantity"],
        }
    )
    return record


def run_mean_reversion_backtest(
    signal_df: pd.DataFrame,
    variant: MeanReversionVariantConfig,
    execution_model: ExecutionModel,
    instrument: InstrumentDetails,
    account_size_usd: float,
) -> MeanReversionBacktestResult:
    """Backtest a single-position, next-open mean reversion variant."""
    trades: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    trade_id = 1
    current_trade: dict[str, Any] | None = None

    for session_date, session_df in signal_df.groupby("session_date", sort=True):
        session_df = session_df.sort_values("timestamp").reset_index(drop=True)
        daily_state = _init_daily_row(session_date)
        trades_in_session = 0

        for row in session_df.itertuples(index=False, name="Bar"):
            timestamp = pd.Timestamp(getattr(row, "timestamp"))
            row_open = float(getattr(row, "open"))
            row_high = float(getattr(row, "high"))
            row_low = float(getattr(row, "low"))
            row_close = float(getattr(row, "close"))

            if current_trade is None and trades_in_session < int(variant.max_trades_per_day):
                signal_direction = 1 if bool(getattr(row, "entry_long", False)) else -1 if bool(getattr(row, "entry_short", False)) else 0
                if signal_direction != 0:
                    entry_price = execution_model.apply_slippage(row_open, signal_direction, is_entry=True)
                    target_price_raw = (
                        getattr(row, "target_reference_long", np.nan)
                        if signal_direction == 1
                        else getattr(row, "target_reference_short", np.nan)
                    )
                    stop_price_raw = (
                        getattr(row, "stop_reference_long", np.nan)
                        if signal_direction == 1
                        else getattr(row, "stop_reference_short", np.nan)
                    )

                    target_price = float(target_price_raw) if pd.notna(target_price_raw) else np.nan
                    stop_price = float(stop_price_raw) if pd.notna(stop_price_raw) else np.nan

                    if signal_direction == 1 and (pd.isna(target_price) or pd.isna(stop_price) or target_price <= entry_price or stop_price >= entry_price):
                        signal_direction = 0
                    if signal_direction == -1 and (pd.isna(target_price) or pd.isna(stop_price) or target_price >= entry_price or stop_price <= entry_price):
                        signal_direction = 0

                    if signal_direction != 0:
                        partial_quantity = 0
                        partial_exit_price = np.nan
                        if bool(variant.use_partial_exit) and int(variant.fixed_quantity) >= 2:
                            partial_quantity = max(1, int(np.floor(int(variant.fixed_quantity) / 2)))
                            partial_exit_price = entry_price + signal_direction * float(variant.partial_target_fraction) * (target_price - entry_price) * signal_direction

                        current_trade = {
                            "session_date": session_date,
                            "direction_int": signal_direction,
                            "entry_time": timestamp,
                            "entry_signal_time": timestamp,
                            "entry_price": entry_price,
                            "stop_price": stop_price,
                            "target_price": target_price,
                            "initial_quantity": int(variant.fixed_quantity),
                            "quantity_remaining": int(variant.fixed_quantity),
                            "bars_held": 0,
                            "partial_exit_taken": False,
                            "partial_exit_time": pd.NaT,
                            "partial_exit_price": np.nan,
                            "partial_exit_quantity": 0,
                            "partial_candidate_price": partial_exit_price,
                            "partial_candidate_quantity": partial_quantity,
                            "partial_gross_pnl_usd": 0.0,
                        }

            if current_trade is not None:
                current_trade["bars_held"] = int(current_trade["bars_held"]) + 1
                direction = int(current_trade["direction_int"])
                target_hit = bool(
                    row_high >= float(current_trade["target_price"])
                    if direction == 1
                    else row_low <= float(current_trade["target_price"])
                )
                stop_hit = bool(
                    row_low <= float(current_trade["stop_price"])
                    if direction == 1
                    else row_high >= float(current_trade["stop_price"])
                )

                if not current_trade["partial_exit_taken"] and int(current_trade["partial_candidate_quantity"]) > 0:
                    partial_price = float(current_trade["partial_candidate_price"])
                    partial_hit = bool(row_high >= partial_price if direction == 1 else row_low <= partial_price)
                    if partial_hit and not target_hit and not stop_hit:
                        partial_exit_price = execution_model.apply_slippage(partial_price, direction, is_entry=False)
                        partial_quantity = int(current_trade["partial_candidate_quantity"])
                        partial_gross_pnl = (
                            (partial_exit_price - float(current_trade["entry_price"]))
                            * direction
                            * float(instrument.point_value_usd)
                            * partial_quantity
                        )
                        current_trade["partial_exit_taken"] = True
                        current_trade["partial_exit_time"] = timestamp
                        current_trade["partial_exit_price"] = partial_exit_price
                        current_trade["partial_exit_quantity"] = partial_quantity
                        current_trade["partial_gross_pnl_usd"] = partial_gross_pnl
                        current_trade["quantity_remaining"] = int(current_trade["quantity_remaining"]) - partial_quantity

                exit_reason = None
                raw_exit_price = None
                if stop_hit:
                    exit_reason = "stop"
                    raw_exit_price = float(current_trade["stop_price"])
                elif target_hit:
                    exit_reason = "target_after_partial" if current_trade["partial_exit_taken"] else "target"
                    raw_exit_price = float(current_trade["target_price"])
                elif int(current_trade["bars_held"]) >= int(variant.timeout_bars):
                    exit_reason = "timeout"
                    raw_exit_price = row_close
                elif bool(getattr(row, "is_last_bar_of_session", False)):
                    exit_reason = "session_close"
                    raw_exit_price = row_close

                if exit_reason is not None and raw_exit_price is not None:
                    record = _close_trade_record(
                        trade_id=trade_id,
                        variant=variant,
                        instrument=instrument,
                        execution_model=execution_model,
                        open_trade=current_trade,
                        raw_exit_price=float(raw_exit_price),
                        exit_time=timestamp if exit_reason == "stop" or exit_reason.startswith("target") else timestamp + pd.Timedelta(minutes=1),
                        exit_reason=exit_reason,
                        account_size_usd=account_size_usd,
                    )
                    trades.append(record)
                    trade_id += 1
                    trades_in_session += 1
                    daily_state["daily_pnl_usd"] += float(record["net_pnl_usd"])
                    daily_state["daily_gross_pnl_usd"] += float(record["pnl_usd"])
                    daily_state["daily_fees_usd"] += float(record["fees"])
                    daily_state["daily_trade_count"] += 1
                    if float(record["net_pnl_usd"]) < 0:
                        daily_state["daily_loss_count"] += 1
                    current_trade = None

        if current_trade is not None:
            last_row = session_df.iloc[-1]
            record = _close_trade_record(
                trade_id=trade_id,
                variant=variant,
                instrument=instrument,
                execution_model=execution_model,
                open_trade=current_trade,
                raw_exit_price=float(last_row["close"]),
                exit_time=pd.Timestamp(last_row["timestamp"]) + pd.Timedelta(minutes=1),
                exit_reason="session_close",
                account_size_usd=account_size_usd,
            )
            trades.append(record)
            trade_id += 1
            trades_in_session += 1
            daily_state["daily_pnl_usd"] += float(record["net_pnl_usd"])
            daily_state["daily_gross_pnl_usd"] += float(record["pnl_usd"])
            daily_state["daily_fees_usd"] += float(record["fees"])
            daily_state["daily_trade_count"] += 1
            if float(record["net_pnl_usd"]) < 0:
                daily_state["daily_loss_count"] += 1
            current_trade = None

        daily_rows.append(daily_state)

    return MeanReversionBacktestResult(
        trades=_trade_log_frame(trades),
        bar_results=pd.DataFrame(),
        daily_results=_finalize_daily_results(daily_rows, initial_capital=account_size_usd),
    )
