"""Dedicated backtester for the volatility compression -> expansion breakout strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.engine.execution_model import ExecutionModel
from src.engine.trade_log import empty_trade_log, trade_to_record
from src.engine.vwap_backtester import InstrumentDetails
from src.strategy.volatility_compression_expansion import DEFAULT_BAR_MINUTES, VCEBVariantConfig


@dataclass(frozen=True)
class VCEBBacktestResult:
    """Structured VCEB backtest outputs reused by the campaign."""

    trades: pd.DataFrame
    bar_results: pd.DataFrame
    daily_results: pd.DataFrame


def _trade_log_frame(trades: list[dict[str, Any]]) -> pd.DataFrame:
    if not trades:
        frame = empty_trade_log()
        extra_columns = {
            "variant_name": object,
            "entry_signal_time": object,
            "bars_held": float,
            "holding_minutes": float,
            "gross_pnl_usd": float,
            "entry_hour": float,
            "signal_atr": float,
            "signal_box_width": float,
            "mfe_points": float,
            "mae_points": float,
            "mfe_usd": float,
            "mae_usd": float,
            "mfe_r": float,
            "mae_r": float,
        }
        for column, dtype in extra_columns.items():
            frame[column] = pd.Series(dtype=dtype)
        return frame
    return pd.DataFrame(trades)


def _rebuild_daily_results_from_trades(
    trades: pd.DataFrame,
    sessions: list,
    initial_capital: float,
) -> pd.DataFrame:
    session_index = pd.Index(pd.to_datetime(pd.Index(sessions)).date)
    daily = pd.DataFrame({"session_date": session_index})
    if trades.empty:
        daily["daily_pnl_usd"] = 0.0
        daily["daily_gross_pnl_usd"] = 0.0
        daily["daily_fees_usd"] = 0.0
        daily["daily_trade_count"] = 0
        daily["daily_loss_count"] = 0
    else:
        view = trades.copy()
        view["session_date"] = pd.to_datetime(view["session_date"]).dt.date
        view["is_loss"] = pd.to_numeric(view["net_pnl_usd"], errors="coerce").lt(0)
        grouped = (
            view.groupby("session_date", as_index=False)
            .agg(
                daily_pnl_usd=("net_pnl_usd", "sum"),
                daily_gross_pnl_usd=("pnl_usd", "sum"),
                daily_fees_usd=("fees", "sum"),
                daily_trade_count=("trade_id", "count"),
                daily_loss_count=("is_loss", "sum"),
            )
        )
        daily = daily.merge(grouped, on="session_date", how="left").fillna(
            {
                "daily_pnl_usd": 0.0,
                "daily_gross_pnl_usd": 0.0,
                "daily_fees_usd": 0.0,
                "daily_trade_count": 0,
                "daily_loss_count": 0,
            }
        )
    daily["daily_stop_breached"] = False
    daily["trading_halted"] = False
    daily = daily.sort_values("session_date").reset_index(drop=True)
    daily["equity"] = float(initial_capital) + pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0).cumsum()
    daily["peak_equity"] = daily["equity"].cummax()
    daily["drawdown_usd"] = daily["equity"] - daily["peak_equity"]
    daily["green_day"] = daily["daily_pnl_usd"] > 0
    daily["weekday"] = pd.to_datetime(daily["session_date"]).dt.day_name()
    return daily


def _trade_risk_usd(
    *,
    execution_model: ExecutionModel,
    instrument: InstrumentDetails,
    direction: int,
    entry_price: float,
    stop_price: float,
    quantity: int,
) -> tuple[float, float]:
    stop_exit_price = execution_model.apply_slippage(stop_price, direction, is_entry=False)
    stop_pnl_points = (stop_exit_price - float(entry_price)) * int(direction)
    risk_per_contract = abs(float(stop_pnl_points) * float(instrument.point_value_usd)) + execution_model.round_trip_fees(quantity=1)
    return risk_per_contract, risk_per_contract * int(quantity)


def _mfe_mae_points(
    *,
    direction: int,
    entry_price: float,
    max_high: float,
    min_low: float,
) -> tuple[float, float]:
    if int(direction) == 1:
        mfe_points = max(float(max_high) - float(entry_price), 0.0)
        mae_points = min(float(min_low) - float(entry_price), 0.0)
    else:
        mfe_points = max(float(entry_price) - float(min_low), 0.0)
        mae_points = min(float(entry_price) - float(max_high), 0.0)
    return float(mfe_points), float(mae_points)


def _close_trade_record(
    *,
    trade_id: int,
    variant: VCEBVariantConfig,
    instrument: InstrumentDetails,
    execution_model: ExecutionModel,
    open_trade: dict[str, Any],
    raw_exit_price: float,
    exit_time: pd.Timestamp,
    exit_reason: str,
) -> dict[str, Any]:
    direction = int(open_trade["direction_int"])
    quantity = int(open_trade["quantity"])
    entry_price = float(open_trade["entry_price"])
    stop_price = float(open_trade["stop_price"])
    exit_price = execution_model.apply_slippage(raw_exit_price, direction, is_entry=False)
    pnl_points = (exit_price - entry_price) * direction
    gross_pnl_usd = pnl_points * float(instrument.point_value_usd) * quantity
    fees = execution_model.round_trip_fees(quantity=quantity)
    net_pnl_usd = gross_pnl_usd - fees

    risk_per_contract_usd, trade_risk_usd = _trade_risk_usd(
        execution_model=execution_model,
        instrument=instrument,
        direction=direction,
        entry_price=entry_price,
        stop_price=stop_price,
        quantity=quantity,
    )
    mfe_points, mae_points = _mfe_mae_points(
        direction=direction,
        entry_price=entry_price,
        max_high=float(open_trade["max_high_seen"]),
        min_low=float(open_trade["min_low_seen"]),
    )
    mfe_usd = mfe_points * float(instrument.point_value_usd) * quantity
    mae_usd = mae_points * float(instrument.point_value_usd) * quantity
    mfe_r = mfe_usd / trade_risk_usd if trade_risk_usd > 0 else np.nan
    mae_r = mae_usd / trade_risk_usd if trade_risk_usd > 0 else np.nan

    record = trade_to_record(
        trade_id,
        {
            "session_date": open_trade["session_date"],
            "direction": "long" if direction == 1 else "short",
            "quantity": quantity,
            "entry_time": open_trade["entry_time"],
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": float(open_trade["target_price"]),
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "account_size_usd": float(variant.initial_capital_usd),
            "risk_per_trade_pct": None,
            "risk_budget_usd": None,
            "risk_per_contract_usd": risk_per_contract_usd,
            "actual_risk_usd": trade_risk_usd,
            "trade_risk_usd": trade_risk_usd,
            "notional_usd": quantity * entry_price * float(instrument.point_value_usd),
            "leverage_used": (
                quantity * entry_price * float(instrument.point_value_usd) / float(variant.initial_capital_usd)
                if float(variant.initial_capital_usd) > 0
                else None
            ),
            "pnl_points": pnl_points,
            "pnl_ticks": pnl_points / float(instrument.tick_size) if float(instrument.tick_size) > 0 else np.nan,
            "pnl_usd": gross_pnl_usd,
            "fees": fees,
            "net_pnl_usd": net_pnl_usd,
        },
    )
    record.update(
        {
            "variant_name": variant.name,
            "entry_signal_time": open_trade["entry_signal_time"],
            "bars_held": int(open_trade["bars_held"]),
            "holding_minutes": float((exit_time - open_trade["entry_time"]).total_seconds() / 60.0),
            "gross_pnl_usd": gross_pnl_usd,
            "entry_hour": float(pd.Timestamp(open_trade["entry_time"]).hour),
            "signal_atr": float(open_trade["signal_atr"]),
            "signal_box_width": float(open_trade["signal_box_width"]),
            "mfe_points": mfe_points,
            "mae_points": mae_points,
            "mfe_usd": mfe_usd,
            "mae_usd": mae_usd,
            "mfe_r": mfe_r,
            "mae_r": mae_r,
        }
    )
    return record


def run_volatility_compression_expansion_backtest(
    signal_df: pd.DataFrame,
    variant: VCEBVariantConfig,
    execution_model: ExecutionModel,
    instrument: InstrumentDetails,
    *,
    bar_minutes: int = DEFAULT_BAR_MINUTES,
) -> VCEBBacktestResult:
    """Backtest the VCEB strategy with strict next-open execution."""
    trades: list[dict[str, Any]] = []
    current_trade: dict[str, Any] | None = None
    trade_id = 1

    for session_date, session_df in signal_df.groupby("session_date", sort=True):
        session_df = session_df.sort_values("timestamp").reset_index(drop=True)

        for row in session_df.itertuples(index=False, name="Bar"):
            timestamp = pd.Timestamp(getattr(row, "timestamp"))
            bar_end_time = timestamp + pd.Timedelta(minutes=int(bar_minutes))
            row_open = float(getattr(row, "open"))
            row_high = float(getattr(row, "high"))
            row_low = float(getattr(row, "low"))
            row_close = float(getattr(row, "close"))

            if current_trade is None:
                signal_direction = 1 if bool(getattr(row, "entry_long", False)) else -1 if bool(getattr(row, "entry_short", False)) else 0
                stop_distance = pd.to_numeric(pd.Series([getattr(row, "entry_stop_distance", np.nan)]), errors="coerce").iloc[0]
                target_r = pd.to_numeric(pd.Series([getattr(row, "entry_target_r", np.nan)]), errors="coerce").iloc[0]
                if signal_direction != 0 and pd.notna(stop_distance) and pd.notna(target_r):
                    stop_distance_float = float(stop_distance)
                    target_r_float = float(target_r)
                    if stop_distance_float > 0.0 and target_r_float > 0.0:
                        entry_price = execution_model.apply_slippage(row_open, signal_direction, is_entry=True)
                        stop_price = entry_price - signal_direction * stop_distance_float
                        target_price = entry_price + signal_direction * target_r_float * stop_distance_float

                        if signal_direction == 1 and target_price > entry_price and stop_price < entry_price:
                            current_trade = {
                                "session_date": session_date,
                                "direction_int": signal_direction,
                                "entry_time": timestamp,
                                "entry_signal_time": getattr(row, "entry_signal_time", pd.NaT),
                                "entry_price": entry_price,
                                "stop_price": stop_price,
                                "target_price": target_price,
                                "quantity": int(variant.fixed_quantity),
                                "bars_held": 0,
                                "signal_atr": float(pd.to_numeric(pd.Series([getattr(row, "entry_signal_atr", np.nan)]), errors="coerce").iloc[0]),
                                "signal_box_width": float(pd.to_numeric(pd.Series([getattr(row, "entry_signal_box_width", np.nan)]), errors="coerce").iloc[0]),
                                "max_high_seen": row_open,
                                "min_low_seen": row_open,
                            }
                        elif signal_direction == -1 and target_price < entry_price and stop_price > entry_price:
                            current_trade = {
                                "session_date": session_date,
                                "direction_int": signal_direction,
                                "entry_time": timestamp,
                                "entry_signal_time": getattr(row, "entry_signal_time", pd.NaT),
                                "entry_price": entry_price,
                                "stop_price": stop_price,
                                "target_price": target_price,
                                "quantity": int(variant.fixed_quantity),
                                "bars_held": 0,
                                "signal_atr": float(pd.to_numeric(pd.Series([getattr(row, "entry_signal_atr", np.nan)]), errors="coerce").iloc[0]),
                                "signal_box_width": float(pd.to_numeric(pd.Series([getattr(row, "entry_signal_box_width", np.nan)]), errors="coerce").iloc[0]),
                                "max_high_seen": row_open,
                                "min_low_seen": row_open,
                            }

            if current_trade is None:
                continue

            current_trade["bars_held"] = int(current_trade["bars_held"]) + 1
            current_trade["max_high_seen"] = max(float(current_trade["max_high_seen"]), row_high)
            current_trade["min_low_seen"] = min(float(current_trade["min_low_seen"]), row_low)

            direction = int(current_trade["direction_int"])
            stop_price = float(current_trade["stop_price"])
            target_price = float(current_trade["target_price"])
            stop_hit = bool(row_low <= stop_price) if direction == 1 else bool(row_high >= stop_price)
            target_hit = bool(row_high >= target_price) if direction == 1 else bool(row_low <= target_price)

            exit_reason: str | None = None
            raw_exit_price: float | None = None
            if stop_hit:
                exit_reason = "stop"
                raw_exit_price = stop_price
            elif target_hit:
                exit_reason = "target"
                raw_exit_price = target_price
            elif int(current_trade["bars_held"]) >= int(variant.time_stop_bars):
                exit_reason = "time_stop"
                raw_exit_price = row_close
            elif bool(getattr(row, "is_last_bar_of_session", False)):
                exit_reason = "session_close"
                raw_exit_price = row_close

            if exit_reason is None or raw_exit_price is None:
                continue

            trades.append(
                _close_trade_record(
                    trade_id=trade_id,
                    variant=variant,
                    instrument=instrument,
                    execution_model=execution_model,
                    open_trade=current_trade,
                    raw_exit_price=float(raw_exit_price),
                    exit_time=bar_end_time,
                    exit_reason=exit_reason,
                )
            )
            trade_id += 1
            current_trade = None

    trades_df = _trade_log_frame(trades)
    all_sessions = sorted(pd.to_datetime(signal_df["session_date"]).dt.date.unique()) if not signal_df.empty else []
    daily_results = _rebuild_daily_results_from_trades(
        trades_df,
        sessions=all_sessions,
        initial_capital=float(variant.initial_capital_usd),
    )
    return VCEBBacktestResult(
        trades=trades_df,
        bar_results=pd.DataFrame(),
        daily_results=daily_results,
    )
