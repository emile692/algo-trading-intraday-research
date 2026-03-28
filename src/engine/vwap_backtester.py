"""Dedicated intraday backtester for VWAP research variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import INSTRUMENT_SPECS
from src.config.vwap_campaign import VWAPVariantConfig
from src.engine.execution_model import ExecutionModel
from src.engine.trade_log import TRADE_LOG_COLUMNS, empty_trade_log, trade_to_record


EQUITY_TICK_SIZE = 0.01
EQUITY_POINT_VALUE_USD = 1.0
EQUITY_PAPER_COMMISSION_PER_SHARE = 0.0005


@dataclass(frozen=True)
class InstrumentDetails:
    """Execution metadata inferred from the selected dataset symbol."""

    symbol: str
    asset_class: str
    tick_size: float
    tick_value_usd: float
    point_value_usd: float
    commission_per_side_usd: float
    slippage_ticks: int


@dataclass(frozen=True)
class VWAPBacktestResult:
    """Structured backtest outputs used by the campaign and notebook."""

    trades: pd.DataFrame
    bar_results: pd.DataFrame
    daily_results: pd.DataFrame


def resolve_instrument_details(symbol: str) -> InstrumentDetails:
    """Resolve instrument specifications from repo defaults or equity-like fallback."""
    upper_symbol = symbol.upper()
    if upper_symbol in INSTRUMENT_SPECS:
        spec = INSTRUMENT_SPECS[upper_symbol]
        return InstrumentDetails(
            symbol=upper_symbol,
            asset_class="futures",
            tick_size=float(spec["tick_size"]),
            tick_value_usd=float(spec["tick_value_usd"]),
            point_value_usd=float(spec["point_value_usd"]),
            commission_per_side_usd=float(spec["commission_per_side_usd"]),
            slippage_ticks=int(spec["slippage_ticks"]),
        )

    return InstrumentDetails(
        symbol=upper_symbol,
        asset_class="equity",
        tick_size=EQUITY_TICK_SIZE,
        tick_value_usd=EQUITY_TICK_SIZE * EQUITY_POINT_VALUE_USD,
        point_value_usd=EQUITY_POINT_VALUE_USD,
        commission_per_side_usd=EQUITY_PAPER_COMMISSION_PER_SHARE,
        slippage_ticks=0,
    )


def build_execution_model_for_profile(
    symbol: str,
    profile_name: str,
    cost_multiplier: float = 1.0,
) -> tuple[ExecutionModel, InstrumentDetails]:
    """Build the execution model used by a VWAP run."""
    details = resolve_instrument_details(symbol)
    if cost_multiplier <= 0:
        raise ValueError("cost_multiplier must be strictly positive.")

    if profile_name == "repo_realistic":
        commission = details.commission_per_side_usd * cost_multiplier
        slippage_ticks = int(round(details.slippage_ticks * cost_multiplier))
        return (
            ExecutionModel(
                commission_per_side_usd=commission,
                slippage_ticks=slippage_ticks,
                tick_size=details.tick_size,
            ),
            details,
        )

    if profile_name == "paper_reference":
        if details.asset_class == "futures":
            commission = details.commission_per_side_usd * cost_multiplier
            slippage_ticks = int(round(details.slippage_ticks * cost_multiplier))
            return (
                ExecutionModel(
                    commission_per_side_usd=commission,
                    slippage_ticks=slippage_ticks,
                    tick_size=details.tick_size,
                ),
                details,
            )
        return (
            ExecutionModel(
                commission_per_side_usd=EQUITY_PAPER_COMMISSION_PER_SHARE * cost_multiplier,
                slippage_ticks=0,
                tick_size=details.tick_size,
            ),
            details,
        )

    raise ValueError(f"Unsupported execution profile '{profile_name}'.")


def _trade_log_frame(trades: list[dict[str, Any]]) -> pd.DataFrame:
    if not trades:
        frame = empty_trade_log()
        for column in (
            "variant_name",
            "entry_reason",
            "bars_held",
            "holding_minutes",
            "gross_pnl_usd",
            "r_multiple",
            "entry_signal_time",
            "exit_signal_time",
        ):
            frame[column] = pd.Series(dtype=float if column in {"holding_minutes", "gross_pnl_usd", "r_multiple"} else object)
        return frame
    return pd.DataFrame(trades)


def _open_position_cost(
    execution_model: ExecutionModel,
    raw_entry_price: float,
    direction: int,
    quantity: int,
    point_value_usd: float,
) -> tuple[float, float]:
    entry_price = execution_model.apply_slippage(raw_entry_price, direction, is_entry=True)
    slippage_cost_usd = abs(entry_price - raw_entry_price) * quantity * point_value_usd
    return entry_price, slippage_cost_usd


def _close_position_record(
    trade_id: int,
    variant: VWAPVariantConfig,
    instrument: InstrumentDetails,
    execution_model: ExecutionModel,
    open_trade: dict[str, Any],
    raw_exit_price: float,
    exit_time: pd.Timestamp,
    exit_reason: str,
    exit_signal_time: pd.Timestamp | None = None,
) -> dict[str, Any]:
    direction = int(open_trade["direction_int"])
    quantity = int(open_trade["quantity"])
    exit_price = execution_model.apply_slippage(raw_exit_price, direction, is_entry=False)
    pnl_points = (exit_price - float(open_trade["entry_price"])) * direction
    gross_pnl_usd = pnl_points * instrument.point_value_usd * quantity
    fees = execution_model.round_trip_fees(quantity=quantity)
    net_pnl_usd = gross_pnl_usd - fees

    trade_risk_raw = open_trade.get("trade_risk_usd", np.nan)
    trade_risk_usd = pd.to_numeric(pd.Series([trade_risk_raw]), errors="coerce").iloc[0]
    if pd.notna(trade_risk_usd) and float(trade_risk_usd) > 0:
        r_multiple = net_pnl_usd / trade_risk_usd
    else:
        r_multiple = np.nan

    holding_minutes = float((exit_time - open_trade["entry_time"]).total_seconds() / 60.0)
    record = trade_to_record(
        trade_id,
        {
            "session_date": open_trade["session_date"],
            "direction": "long" if direction == 1 else "short",
            "quantity": quantity,
            "entry_time": open_trade["entry_time"],
            "entry_price": float(open_trade["entry_price"]),
            "stop_price": open_trade.get("stop_price"),
            "target_price": open_trade.get("target_price"),
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "account_size_usd": float(open_trade["equity_at_entry"]),
            "risk_per_trade_pct": variant.risk_per_trade_pct,
            "risk_budget_usd": open_trade.get("risk_budget_usd"),
            "risk_per_contract_usd": open_trade.get("risk_per_contract_usd"),
            "actual_risk_usd": open_trade.get("actual_risk_usd"),
            "trade_risk_usd": open_trade.get("trade_risk_usd"),
            "notional_usd": open_trade.get("notional_usd"),
            "leverage_used": open_trade.get("leverage_used"),
            "pnl_points": pnl_points,
            "pnl_ticks": pnl_points / instrument.tick_size if instrument.tick_size > 0 else np.nan,
            "pnl_usd": gross_pnl_usd,
            "fees": fees,
            "net_pnl_usd": net_pnl_usd,
        },
    )
    record.update(
        {
            "variant_name": variant.name,
            "entry_reason": open_trade.get("entry_reason"),
            "bars_held": open_trade.get("bars_held", np.nan),
            "holding_minutes": holding_minutes,
            "gross_pnl_usd": gross_pnl_usd,
            "r_multiple": r_multiple,
            "entry_signal_time": open_trade.get("entry_signal_time"),
            "exit_signal_time": exit_signal_time,
        }
    )
    return record


def _paper_full_notional_quantity(equity: float, entry_price: float, point_value_usd: float) -> int:
    if equity <= 0 or entry_price <= 0 or point_value_usd <= 0:
        return 0
    return int(equity // (entry_price * point_value_usd))


def _trade_risk_values(
    execution_model: ExecutionModel,
    instrument: InstrumentDetails,
    entry_price: float,
    stop_price: float | None,
    quantity: int,
) -> tuple[float | None, float | None]:
    if stop_price is None:
        return None, None

    risk_distance_points = abs(entry_price - stop_price)
    if risk_distance_points <= 0:
        return None, None

    risk_per_contract_usd = (
        risk_distance_points * instrument.point_value_usd
        + execution_model.round_trip_fees(quantity=1)
    )
    if risk_per_contract_usd <= 0:
        return None, None
    actual_risk_usd = int(quantity) * risk_per_contract_usd if int(quantity) > 0 else None
    return risk_per_contract_usd, actual_risk_usd


def _risk_based_quantity(
    equity: float,
    variant: VWAPVariantConfig,
    execution_model: ExecutionModel,
    instrument: InstrumentDetails,
    entry_price: float,
    stop_price: float | None,
) -> tuple[int, float | None, float | None, float | None]:
    risk_budget_usd = None
    if variant.risk_per_trade_pct is not None:
        risk_budget_usd = equity * (variant.risk_per_trade_pct / 100.0)

    risk_per_contract_usd, _ = _trade_risk_values(
        execution_model=execution_model,
        instrument=instrument,
        entry_price=entry_price,
        stop_price=stop_price,
        quantity=1,
    )
    if variant.risk_per_trade_pct is None or risk_per_contract_usd is None or risk_budget_usd is None:
        return 0, risk_budget_usd, risk_per_contract_usd, None
    if risk_budget_usd <= 0:
        return 0, risk_budget_usd, risk_per_contract_usd, None

    quantity = int(risk_budget_usd // risk_per_contract_usd)
    actual_risk_usd = quantity * risk_per_contract_usd if quantity > 0 else None
    return quantity, risk_budget_usd, risk_per_contract_usd, actual_risk_usd


def _scaled_quantity(quantity: int, variant: VWAPVariantConfig, losing_streak: int) -> int:
    scaled = int(quantity)
    if (
        variant.consecutive_losses_threshold is not None
        and losing_streak >= int(variant.consecutive_losses_threshold)
    ):
        scaled = int(np.floor(quantity * float(variant.deleverage_after_losing_streak)))
    return max(scaled, 0)


def _resolve_entry_quantity(
    variant: VWAPVariantConfig,
    instrument: InstrumentDetails,
    execution_model: ExecutionModel,
    entry_price: float,
    stop_price: float | None,
    equity: float,
    losing_streak: int,
) -> tuple[int, dict[str, float | None]]:
    risk_quantity, risk_budget_usd, risk_per_contract_usd, actual_risk_usd = _risk_based_quantity(
        equity=equity,
        variant=variant,
        execution_model=execution_model,
        instrument=instrument,
        entry_price=entry_price,
        stop_price=stop_price,
    )

    if variant.risk_per_trade_pct is not None and risk_quantity > 0:
        base_quantity = risk_quantity
    elif variant.quantity_mode == "paper_full_notional":
        base_quantity = _paper_full_notional_quantity(
            equity=equity,
            entry_price=entry_price,
            point_value_usd=instrument.point_value_usd,
        )
        if instrument.asset_class == "futures" and base_quantity == 0 and equity > 0:
            # Futures do not allow fractional contracts; keep the minimum tradable unit
            # so the paper logic remains testable on micro contracts across the sample.
            base_quantity = 1
    else:
        base_quantity = int(variant.fixed_quantity)

    quantity = _scaled_quantity(base_quantity, variant, losing_streak=losing_streak)
    risk_per_contract_usd, actual_risk_usd = _trade_risk_values(
        execution_model=execution_model,
        instrument=instrument,
        entry_price=entry_price,
        stop_price=stop_price,
        quantity=quantity,
    )
    notional_usd = quantity * entry_price * instrument.point_value_usd if quantity > 0 else None
    leverage_used = notional_usd / equity if quantity > 0 and equity > 0 else None
    return quantity, {
        "risk_budget_usd": risk_budget_usd,
        "risk_per_contract_usd": risk_per_contract_usd,
        "actual_risk_usd": actual_risk_usd if quantity > 0 else None,
        "trade_risk_usd": actual_risk_usd if quantity > 0 else None,
        "notional_usd": notional_usd,
        "leverage_used": leverage_used,
    }


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
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
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
    daily = daily.sort_values("session_date").reset_index(drop=True)
    daily["equity"] = float(initial_capital) + daily["daily_pnl_usd"].cumsum()
    daily["peak_equity"] = daily["equity"].cummax()
    daily["drawdown_usd"] = daily["equity"] - daily["peak_equity"]
    daily["green_day"] = daily["daily_pnl_usd"] > 0
    daily["weekday"] = pd.to_datetime(daily["session_date"]).dt.day_name()
    return daily


def _maybe_halt_trading(
    variant: VWAPVariantConfig,
    daily_state: dict[str, Any],
) -> bool:
    if variant.max_losses_per_day is not None and int(daily_state["daily_loss_count"]) >= int(variant.max_losses_per_day):
        return True
    if variant.daily_stop_threshold_usd is not None and float(daily_state["daily_pnl_usd"]) <= -float(variant.daily_stop_threshold_usd):
        daily_state["daily_stop_breached"] = True
        return True
    if variant.max_trades_per_day is not None and int(daily_state["daily_trade_count"]) >= int(variant.max_trades_per_day):
        return True
    return False


def run_target_position_backtest(
    signal_df: pd.DataFrame,
    variant: VWAPVariantConfig,
    execution_model: ExecutionModel,
    instrument: InstrumentDetails,
) -> VWAPBacktestResult:
    """Backtest always-in-market target-position logic."""
    trades: list[dict[str, Any]] = []
    bar_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []

    equity = float(variant.initial_capital_usd)
    current_trade: dict[str, Any] | None = None
    losing_streak = 0
    trade_id = 1

    for session_date, session_df in signal_df.groupby("session_date", sort=True):
        session_df = session_df.sort_values("timestamp").reset_index(drop=True)
        daily_state = _init_daily_row(session_date)

        for _, row in session_df.iterrows():
            trade_allowed = bool(row.get("trade_allowed", True))
            desired_position = int(row.get("target_position", 0))
            if not trade_allowed:
                desired_position = int(current_trade["direction_int"]) if current_trade is not None else 0
            if daily_state["trading_halted"]:
                desired_position = 0

            event_costs_usd = 0.0
            event_label = ""

            if current_trade is not None and desired_position != int(current_trade["direction_int"]):
                record = _close_position_record(
                    trade_id=trade_id,
                    variant=variant,
                    instrument=instrument,
                    execution_model=execution_model,
                    open_trade=current_trade,
                    raw_exit_price=float(row["open"]),
                    exit_time=pd.Timestamp(row["timestamp"]),
                    exit_reason="signal_flip" if desired_position != 0 else "forced_flat",
                    exit_signal_time=pd.Timestamp(row["timestamp"]),
                )
                trades.append(record)
                trade_id += 1
                equity += float(record["net_pnl_usd"])
                daily_state["daily_pnl_usd"] += float(record["net_pnl_usd"])
                daily_state["daily_gross_pnl_usd"] += float(record["pnl_usd"])
                daily_state["daily_fees_usd"] += float(record["fees"])
                daily_state["daily_trade_count"] += 1
                if float(record["net_pnl_usd"]) < 0:
                    daily_state["daily_loss_count"] += 1
                    losing_streak += 1
                else:
                    losing_streak = 0
                event_label = record["exit_reason"]
                current_trade = None
                daily_state["trading_halted"] = _maybe_halt_trading(variant, daily_state)

            if current_trade is None and desired_position != 0 and not daily_state["trading_halted"]:
                entry_price, slippage_cost_usd = _open_position_cost(
                    execution_model=execution_model,
                    raw_entry_price=float(row["open"]),
                    direction=desired_position,
                    quantity=1,
                    point_value_usd=instrument.point_value_usd,
                )
                quantity, sizing = _resolve_entry_quantity(
                    variant=variant,
                    instrument=instrument,
                    execution_model=execution_model,
                    entry_price=entry_price,
                    stop_price=None,
                    equity=equity,
                    losing_streak=losing_streak,
                )
                if quantity > 0:
                    event_costs_usd += slippage_cost_usd * quantity
                    current_trade = {
                        "session_date": session_date,
                        "direction_int": desired_position,
                        "quantity": quantity,
                        "entry_time": pd.Timestamp(row["timestamp"]),
                        "entry_price": entry_price,
                        "stop_price": np.nan,
                        "target_price": np.nan,
                        "entry_reason": "paper_target_position",
                        "entry_signal_time": pd.Timestamp(row["timestamp"]),
                        "bars_held": 0,
                        "equity_at_entry": equity,
                        **sizing,
                    }
                    event_label = f"{event_label}|entry" if event_label else "entry"

            if current_trade is not None:
                current_trade["bars_held"] = int(current_trade.get("bars_held", 0)) + 1

            direction = int(current_trade["direction_int"]) if current_trade is not None else 0
            quantity = int(current_trade["quantity"]) if current_trade is not None else 0
            gross_bar_pnl_usd = direction * quantity * (float(row["close"]) - float(row["open"])) * instrument.point_value_usd

            bar_rows.append(
                {
                    "timestamp": row["timestamp"],
                    "session_date": session_date,
                    "variant_name": variant.name,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row.get("volume"),
                    "session_vwap": row.get("session_vwap"),
                    "trade_allowed": trade_allowed,
                    "raw_signal": row.get("raw_signal", 0),
                    "target_position": row.get("target_position", 0),
                    "position": direction,
                    "quantity": quantity,
                    "event_label": event_label,
                    "gross_bar_pnl_usd": gross_bar_pnl_usd,
                    "event_costs_usd": event_costs_usd,
                    "net_bar_pnl_usd": gross_bar_pnl_usd - event_costs_usd,
                    "equity_reference_usd": equity,
                }
            )

        if current_trade is not None:
            last_row = session_df.iloc[-1]
            exit_time = pd.Timestamp(last_row["timestamp"]) + pd.Timedelta(minutes=1)
            record = _close_position_record(
                trade_id=trade_id,
                variant=variant,
                instrument=instrument,
                execution_model=execution_model,
                open_trade=current_trade,
                raw_exit_price=float(last_row["close"]),
                exit_time=exit_time,
                exit_reason="session_close",
            )
            trades.append(record)
            trade_id += 1
            equity += float(record["net_pnl_usd"])
            daily_state["daily_pnl_usd"] += float(record["net_pnl_usd"])
            daily_state["daily_gross_pnl_usd"] += float(record["pnl_usd"])
            daily_state["daily_fees_usd"] += float(record["fees"])
            daily_state["daily_trade_count"] += 1
            if float(record["net_pnl_usd"]) < 0:
                daily_state["daily_loss_count"] += 1
                losing_streak += 1
            else:
                losing_streak = 0
            current_trade = None
            daily_state["trading_halted"] = _maybe_halt_trading(variant, daily_state)

        daily_rows.append(daily_state)

    return VWAPBacktestResult(
        trades=_trade_log_frame(trades),
        bar_results=pd.DataFrame(bar_rows),
        daily_results=_finalize_daily_results(daily_rows, initial_capital=variant.initial_capital_usd),
    )


def _entry_signal_for_row(row: pd.Series) -> int:
    if bool(row.get("entry_long", False)):
        return 1
    if bool(row.get("entry_short", False)):
        return -1
    return 0


def _stop_price_from_row(row: pd.Series, direction: int) -> float | None:
    if direction == 1:
        value = row.get("stop_reference_long")
    else:
        value = row.get("stop_reference_short")
    if value is None or pd.isna(value):
        return None
    return float(value)


def run_discrete_signal_backtest(
    signal_df: pd.DataFrame,
    variant: VWAPVariantConfig,
    execution_model: ExecutionModel,
    instrument: InstrumentDetails,
) -> VWAPBacktestResult:
    """Backtest entry and exit events for discrete prop-style variants."""
    trades: list[dict[str, Any]] = []
    bar_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []

    equity = float(variant.initial_capital_usd)
    current_trade: dict[str, Any] | None = None
    losing_streak = 0
    trade_id = 1

    for session_date, session_df in signal_df.groupby("session_date", sort=True):
        session_df = session_df.sort_values("timestamp").reset_index(drop=True)
        daily_state = _init_daily_row(session_date)

        for row in session_df.itertuples(index=False, name="Bar"):
            trade_allowed = bool(getattr(row, "trade_allowed", True))
            event_label = ""
            realized_on_bar = 0.0
            event_costs_usd = 0.0
            timestamp = pd.Timestamp(getattr(row, "timestamp"))
            row_open = float(getattr(row, "open"))
            row_high = float(getattr(row, "high"))
            row_low = float(getattr(row, "low"))
            row_close = float(getattr(row, "close"))

            if current_trade is not None:
                exit_signal = bool(getattr(row, "exit_long", False)) if int(current_trade["direction_int"]) == 1 else bool(getattr(row, "exit_short", False))
                raw_exit_price = None
                exit_reason = None

                if exit_signal:
                    raw_exit_price = row_open
                    exit_reason = "vwap_recross"
                elif int(current_trade["direction_int"]) == 1 and pd.notna(current_trade.get("stop_price")) and row_low <= float(current_trade["stop_price"]):
                    raw_exit_price = float(current_trade["stop_price"])
                    exit_reason = "stop"
                elif int(current_trade["direction_int"]) == -1 and pd.notna(current_trade.get("stop_price")) and row_high >= float(current_trade["stop_price"]):
                    raw_exit_price = float(current_trade["stop_price"])
                    exit_reason = "stop"
                elif bool(getattr(row, "is_last_bar_of_session", False)):
                    raw_exit_price = row_close
                    exit_reason = "session_close"

                if raw_exit_price is not None and exit_reason is not None:
                    exit_time = timestamp if exit_reason != "session_close" else timestamp + pd.Timedelta(minutes=1)
                    record = _close_position_record(
                        trade_id=trade_id,
                        variant=variant,
                        instrument=instrument,
                        execution_model=execution_model,
                        open_trade=current_trade,
                        raw_exit_price=raw_exit_price,
                        exit_time=exit_time,
                        exit_reason=exit_reason,
                        exit_signal_time=timestamp,
                    )
                    trades.append(record)
                    trade_id += 1
                    realized_on_bar += float(record["net_pnl_usd"])
                    equity += float(record["net_pnl_usd"])
                    daily_state["daily_pnl_usd"] += float(record["net_pnl_usd"])
                    daily_state["daily_gross_pnl_usd"] += float(record["pnl_usd"])
                    daily_state["daily_fees_usd"] += float(record["fees"])
                    daily_state["daily_trade_count"] += 1
                    if float(record["net_pnl_usd"]) < 0:
                        daily_state["daily_loss_count"] += 1
                        losing_streak += 1
                    else:
                        losing_streak = 0
                    event_label = exit_reason
                    current_trade = None
                    daily_state["trading_halted"] = _maybe_halt_trading(variant, daily_state)

            if current_trade is None and not daily_state["trading_halted"] and trade_allowed:
                signal_direction = 1 if bool(getattr(row, "entry_long", False)) else -1 if bool(getattr(row, "entry_short", False)) else 0
                if signal_direction != 0:
                    entry_price, slippage_cost_usd = _open_position_cost(
                        execution_model=execution_model,
                        raw_entry_price=row_open,
                        direction=signal_direction,
                        quantity=1,
                        point_value_usd=instrument.point_value_usd,
                    )
                    stop_price = getattr(row, "stop_reference_long", None) if signal_direction == 1 else getattr(row, "stop_reference_short", None)
                    if stop_price is not None:
                        stop_price = float(stop_price)
                        if signal_direction == 1 and stop_price >= entry_price:
                            stop_price = entry_price - instrument.tick_size
                        if signal_direction == -1 and stop_price <= entry_price:
                            stop_price = entry_price + instrument.tick_size

                    quantity, sizing = _resolve_entry_quantity(
                        variant=variant,
                        instrument=instrument,
                        execution_model=execution_model,
                        entry_price=entry_price,
                        stop_price=stop_price,
                        equity=equity,
                        losing_streak=losing_streak,
                    )
                    if quantity > 0:
                        current_trade = {
                            "session_date": session_date,
                            "direction_int": signal_direction,
                            "quantity": quantity,
                            "entry_time": timestamp,
                            "entry_price": entry_price,
                            "stop_price": stop_price,
                            "target_price": np.nan,
                            "entry_reason": variant.name,
                            "entry_signal_time": timestamp,
                            "bars_held": 0,
                            "equity_at_entry": equity,
                            **sizing,
                        }
                        event_costs_usd += slippage_cost_usd * quantity
                        event_label = f"{event_label}|entry" if event_label else "entry"

            if current_trade is not None:
                current_trade["bars_held"] = int(current_trade.get("bars_held", 0)) + 1

            direction = int(current_trade["direction_int"]) if current_trade is not None else 0
            quantity = int(current_trade["quantity"]) if current_trade is not None else 0
            gross_bar_pnl_usd = direction * quantity * (row_close - row_open) * instrument.point_value_usd
            bar_rows.append(
                {
                    "timestamp": getattr(row, "timestamp"),
                    "session_date": session_date,
                    "variant_name": variant.name,
                    "open": row_open,
                    "high": row_high,
                    "low": row_low,
                    "close": row_close,
                    "volume": getattr(row, "volume", None),
                    "session_vwap": getattr(row, "session_vwap", None),
                    "trade_allowed": trade_allowed,
                    "raw_signal": getattr(row, "raw_signal", 0),
                    "entry_long": getattr(row, "entry_long", False),
                    "entry_short": getattr(row, "entry_short", False),
                    "exit_long": getattr(row, "exit_long", False),
                    "exit_short": getattr(row, "exit_short", False),
                    "position": direction,
                    "quantity": quantity,
                    "event_label": event_label,
                    "gross_bar_pnl_usd": gross_bar_pnl_usd,
                    "event_costs_usd": event_costs_usd,
                    "net_bar_pnl_usd": gross_bar_pnl_usd - event_costs_usd + realized_on_bar,
                    "equity_reference_usd": equity,
                }
            )

        if current_trade is not None:
            last_row = session_df.iloc[-1]
            record = _close_position_record(
                trade_id=trade_id,
                variant=variant,
                instrument=instrument,
                execution_model=execution_model,
                open_trade=current_trade,
                raw_exit_price=float(last_row["close"]),
                exit_time=pd.Timestamp(last_row["timestamp"]) + pd.Timedelta(minutes=1),
                exit_reason="session_close",
            )
            trades.append(record)
            trade_id += 1
            equity += float(record["net_pnl_usd"])
            daily_state["daily_pnl_usd"] += float(record["net_pnl_usd"])
            daily_state["daily_gross_pnl_usd"] += float(record["pnl_usd"])
            daily_state["daily_fees_usd"] += float(record["fees"])
            daily_state["daily_trade_count"] += 1
            if float(record["net_pnl_usd"]) < 0:
                daily_state["daily_loss_count"] += 1
                losing_streak += 1
            else:
                losing_streak = 0
            current_trade = None
            daily_state["trading_halted"] = _maybe_halt_trading(variant, daily_state)

        daily_rows.append(daily_state)

    return VWAPBacktestResult(
        trades=_trade_log_frame(trades),
        bar_results=pd.DataFrame(bar_rows),
        daily_results=_finalize_daily_results(daily_rows, initial_capital=variant.initial_capital_usd),
    )


def run_vwap_backtest(
    signal_df: pd.DataFrame,
    variant: VWAPVariantConfig,
    execution_model: ExecutionModel,
    instrument: InstrumentDetails,
) -> VWAPBacktestResult:
    """Dispatch to the appropriate VWAP backtest mode."""
    if variant.mode == "target_position":
        return run_target_position_backtest(
            signal_df=signal_df,
            variant=variant,
            execution_model=execution_model,
            instrument=instrument,
        )
    if variant.mode == "discrete":
        return run_discrete_signal_backtest(
            signal_df=signal_df,
            variant=variant,
            execution_model=execution_model,
            instrument=instrument,
        )
    raise ValueError(f"Unsupported VWAP backtest mode '{variant.mode}'.")
