"""Dedicated leak-free backtester for volume climax pullback V2 variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.engine.execution_model import ExecutionModel
from src.engine.trade_log import empty_trade_log, trade_to_record
from src.engine.vwap_backtester import InstrumentDetails
from src.strategy.volume_climax_pullback_v2 import VolumeClimaxPullbackV2Variant


@dataclass(frozen=True)
class VolumeClimaxPullbackV2BacktestResult:
    trades: pd.DataFrame


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if not pd.isna(value) else float("nan")
    try:
        parsed = pd.to_numeric(value, errors="coerce")
    except (TypeError, ValueError):
        return float("nan")
    return float(parsed) if pd.notna(parsed) else float("nan")


def _bool_or_false(value: Any) -> bool:
    return bool(value) if value is not None and not pd.isna(value) else False


def _entry_market_fill(execution_model: ExecutionModel, raw_price: float, direction: int) -> float:
    return float(execution_model.apply_slippage(raw_price, direction, is_entry=True))


def _entry_limit_fill(raw_open: float, limit_price: float, direction: int) -> float:
    if direction == 1:
        return float(raw_open) if raw_open <= limit_price else float(limit_price)
    return float(raw_open) if raw_open >= limit_price else float(limit_price)


def _resolve_target_price(
    variant: VolumeClimaxPullbackV2Variant,
    direction: int,
    entry_price: float,
    risk_points: float,
    reference_atr: float,
    reference_vwap: float,
) -> tuple[float | None, float | None, float | None]:
    if variant.exit_mode == "fixed_rr":
        target = entry_price + direction * float(variant.rr_target) * risk_points
        return float(target), None, None

    if variant.exit_mode == "target_vwap":
        if not np.isfinite(reference_vwap):
            return None, None, None
        favorable = (reference_vwap - entry_price) * direction
        if favorable <= 0:
            return None, None, None
        return float(reference_vwap), None, None

    if variant.exit_mode == "atr_fraction":
        if not np.isfinite(reference_atr) or reference_atr <= 0 or variant.atr_target_multiple is None:
            return None, None, None
        target = entry_price + direction * float(variant.atr_target_multiple) * reference_atr
        return float(target), None, None

    if variant.exit_mode == "mixed":
        if not np.isfinite(reference_atr) or reference_atr <= 0:
            return None, None, None
        partial_target = entry_price + direction * 0.5 * reference_atr
        trailing_offset = float(variant.trailing_atr_multiple) * reference_atr
        return None, float(partial_target), float(trailing_offset)

    raise ValueError(f"Unsupported exit_mode '{variant.exit_mode}'.")


def _trade_record(
    trade_id: int,
    session_date: Any,
    direction: int,
    quantity: float,
    entry_time: pd.Timestamp,
    entry_signal_time: pd.Timestamp | None,
    entry_price: float,
    stop_price: float,
    target_price: float | None,
    exit_time: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
    risk_usd: float,
    instrument: InstrumentDetails,
    execution_model: ExecutionModel,
    variant: VolumeClimaxPullbackV2Variant,
    bars_held: int,
) -> dict[str, Any]:
    pnl_points = (float(exit_price) - float(entry_price)) * int(direction) * float(quantity)
    gross = pnl_points * float(instrument.point_value_usd)
    fees = execution_model.round_trip_fees(quantity=1)
    net = gross - fees
    record = trade_to_record(
        trade_id,
        {
            "session_date": session_date,
            "direction": "long" if direction == 1 else "short",
            "quantity": float(quantity),
            "entry_time": entry_time,
            "entry_price": float(entry_price),
            "stop_price": float(stop_price),
            "target_price": float(target_price) if target_price is not None and np.isfinite(target_price) else np.nan,
            "exit_time": exit_time,
            "exit_price": float(exit_price),
            "exit_reason": exit_reason,
            "account_size_usd": np.nan,
            "risk_per_trade_pct": np.nan,
            "risk_budget_usd": np.nan,
            "risk_per_contract_usd": float(risk_usd),
            "actual_risk_usd": float(risk_usd),
            "trade_risk_usd": float(risk_usd),
            "notional_usd": float(entry_price) * float(instrument.point_value_usd),
            "leverage_used": np.nan,
            "pnl_points": float(pnl_points),
            "pnl_ticks": float(pnl_points / instrument.tick_size) if instrument.tick_size > 0 else np.nan,
            "pnl_usd": float(gross),
            "fees": float(fees),
            "net_pnl_usd": float(net),
        },
    )
    record["variant_name"] = variant.name
    record["entry_signal_time"] = entry_signal_time
    record["bars_held"] = int(bars_held)
    return record


def _build_open_trade(
    *,
    row: pd.Series,
    direction: int,
    entry_price: float,
    variant: VolumeClimaxPullbackV2Variant,
    execution_model: ExecutionModel,
    instrument: InstrumentDetails,
) -> dict[str, Any] | None:
    stop_col = "setup_stop_reference_long" if direction == 1 else "setup_stop_reference_short"
    stop_price = _float_or_nan(row.get(stop_col))
    reference_atr = _float_or_nan(row.get("setup_reference_atr"))
    reference_vwap = _float_or_nan(row.get("setup_reference_vwap"))
    entry_time = pd.Timestamp(row["timestamp"])
    signal_time_raw = row.get("setup_signal_time")
    entry_signal_time = pd.Timestamp(signal_time_raw) if pd.notna(signal_time_raw) else None

    if not np.isfinite(stop_price):
        return None

    risk_points = (float(entry_price) - float(stop_price)) * int(direction)
    if risk_points <= 0:
        return None

    target_price, partial_target_price, trailing_offset = _resolve_target_price(
        variant=variant,
        direction=direction,
        entry_price=float(entry_price),
        risk_points=float(risk_points),
        reference_atr=float(reference_atr),
        reference_vwap=float(reference_vwap),
    )
    if variant.exit_mode != "mixed" and target_price is None:
        return None

    risk_usd = float(risk_points * instrument.point_value_usd + execution_model.round_trip_fees(quantity=1))
    return {
        "entry_time": entry_time,
        "entry_signal_time": entry_signal_time,
        "entry_price": float(entry_price),
        "direction": int(direction),
        "stop_price": float(stop_price),
        "target_price": float(target_price) if target_price is not None else np.nan,
        "partial_target_price": float(partial_target_price) if partial_target_price is not None else np.nan,
        "trailing_offset": float(trailing_offset) if trailing_offset is not None else np.nan,
        "time_stop_bars": int(variant.time_stop_bars),
        "bars_held": 0,
        "risk_usd": risk_usd,
        "reference_atr": float(reference_atr) if np.isfinite(reference_atr) else np.nan,
        "partial_filled": False,
        "remaining_fraction": 1.0,
        "realized_weighted_pnl_points": 0.0,
        "last_close": _float_or_nan(row.get("close")),
        "session_date": row.get("session_date"),
    }


def _confirmation_triggered(
    row: pd.Series,
    direction: int,
    reference_close: float,
    reference_range: float,
    pullback_fraction: float,
) -> bool:
    if not np.isfinite(reference_close) or not np.isfinite(reference_range) or reference_range <= 0:
        return False

    high = _float_or_nan(row.get("high"))
    low = _float_or_nan(row.get("low"))
    open_ = _float_or_nan(row.get("open"))
    close = _float_or_nan(row.get("close"))
    pullback_distance = float(pullback_fraction) * float(reference_range)

    if direction == 1:
        retrace_level = float(reference_close) - pullback_distance
        return (low <= retrace_level) and (close > open_)

    retrace_level = float(reference_close) + pullback_distance
    return (high >= retrace_level) and (close < open_)


def _maybe_fill_pullback_limit(
    row: pd.Series,
    direction: int,
    reference_close: float,
    reference_range: float,
    pullback_fraction: float,
) -> float | None:
    if not np.isfinite(reference_close) or not np.isfinite(reference_range) or reference_range <= 0:
        return None

    open_ = _float_or_nan(row.get("open"))
    high = _float_or_nan(row.get("high"))
    low = _float_or_nan(row.get("low"))
    if direction == 1:
        limit_price = float(reference_close) - float(pullback_fraction) * float(reference_range)
        if low <= limit_price:
            return _entry_limit_fill(open_, limit_price, direction)
        return None

    limit_price = float(reference_close) + float(pullback_fraction) * float(reference_range)
    if high >= limit_price:
        return _entry_limit_fill(open_, limit_price, direction)
    return None


def _trail_stop_from_last_close(direction: int, stop_price: float, trailing_offset: float, last_close: float) -> float:
    if not np.isfinite(trailing_offset) or trailing_offset <= 0 or not np.isfinite(last_close):
        return float(stop_price)
    if direction == 1:
        return max(float(stop_price), float(last_close) - float(trailing_offset))
    return min(float(stop_price), float(last_close) + float(trailing_offset))


def _entry_row_from_pending_setup(current_row: pd.Series, pending_setup: dict[str, Any]) -> pd.Series:
    entry_row = current_row.copy()
    for key in (
        "setup_stop_reference_long",
        "setup_stop_reference_short",
        "setup_reference_atr",
        "setup_reference_vwap",
        "setup_signal_time",
    ):
        if key in pending_setup:
            entry_row[key] = pending_setup[key]
    return entry_row


def _finalize_trade(
    trades: list[dict[str, Any]],
    trade_id: int,
    session_date: Any,
    direction: int,
    entry_time: pd.Timestamp,
    entry_signal_time: pd.Timestamp | None,
    entry_price: float,
    stop_price: float,
    target_price: float | None,
    exit_time: pd.Timestamp,
    raw_exit_price: float,
    exit_reason: str,
    risk_usd: float,
    bars_held: int,
    remaining_fraction: float,
    realized_weighted_pnl_points: float,
    execution_model: ExecutionModel,
    instrument: InstrumentDetails,
    variant: VolumeClimaxPullbackV2Variant,
) -> int:
    fill_exit = float(execution_model.apply_slippage(raw_exit_price, direction, is_entry=False))
    weighted_final_points = float(remaining_fraction) * (fill_exit - float(entry_price)) * int(direction)
    total_weighted_points = float(realized_weighted_pnl_points) + weighted_final_points
    effective_exit_price = float(entry_price) + (float(total_weighted_points) * int(direction))
    trades.append(
        _trade_record(
            trade_id=trade_id,
            session_date=session_date,
            direction=direction,
            quantity=1.0,
            entry_time=entry_time,
            entry_signal_time=entry_signal_time,
            entry_price=float(entry_price),
            stop_price=float(stop_price),
            target_price=target_price,
            exit_time=exit_time,
            exit_price=effective_exit_price,
            exit_reason=exit_reason,
            risk_usd=float(risk_usd),
            instrument=instrument,
            execution_model=execution_model,
            variant=variant,
            bars_held=bars_held,
        )
    )
    return trade_id + 1


def run_volume_climax_pullback_v2_backtest(
    signal_df: pd.DataFrame,
    variant: VolumeClimaxPullbackV2Variant,
    execution_model: ExecutionModel,
    instrument: InstrumentDetails,
) -> VolumeClimaxPullbackV2BacktestResult:
    trades: list[dict[str, Any]] = []
    open_trade: dict[str, Any] | None = None
    trade_id = 1

    for session_date, sdf in signal_df.groupby("session_date", sort=True):
        sdf = sdf.sort_values("timestamp").reset_index(drop=True)
        pending_setup: dict[str, Any] | None = None
        last_idx = len(sdf) - 1

        for idx, row in sdf.iterrows():
            timestamp = pd.Timestamp(row["timestamp"])
            open_ = _float_or_nan(row.get("open"))
            high = _float_or_nan(row.get("high"))
            low = _float_or_nan(row.get("low"))
            close = _float_or_nan(row.get("close"))

            trade_was_open_at_bar_start = open_trade is not None

            if open_trade is not None:
                direction = int(open_trade["direction"])
                if variant.exit_mode == "mixed" and _bool_or_false(open_trade.get("partial_filled")):
                    open_trade["stop_price"] = _trail_stop_from_last_close(
                        direction=direction,
                        stop_price=float(open_trade["stop_price"]),
                        trailing_offset=_float_or_nan(open_trade.get("trailing_offset")),
                        last_close=_float_or_nan(open_trade.get("last_close")),
                    )

                open_trade["bars_held"] = int(open_trade["bars_held"]) + 1

                if variant.exit_mode == "mixed":
                    stop_hit = (low <= float(open_trade["stop_price"])) if direction == 1 else (high >= float(open_trade["stop_price"]))

                    if not _bool_or_false(open_trade.get("partial_filled")):
                        partial_target_price = _float_or_nan(open_trade.get("partial_target_price"))
                        partial_hit = (high >= partial_target_price) if direction == 1 else (low <= partial_target_price)

                        if stop_hit and partial_hit:
                            trade_id = _finalize_trade(
                                trades,
                                trade_id,
                                session_date,
                                direction,
                                open_trade["entry_time"],
                                open_trade.get("entry_signal_time"),
                                float(open_trade["entry_price"]),
                                float(open_trade["stop_price"]),
                                partial_target_price if np.isfinite(partial_target_price) else None,
                                timestamp,
                                float(open_trade["stop_price"]),
                                "stop_ambiguous_first",
                                float(open_trade["risk_usd"]),
                                int(open_trade["bars_held"]),
                                float(open_trade["remaining_fraction"]),
                                float(open_trade["realized_weighted_pnl_points"]),
                                execution_model,
                                instrument,
                                variant,
                            )
                            open_trade = None
                        elif stop_hit:
                            trade_id = _finalize_trade(
                                trades,
                                trade_id,
                                session_date,
                                direction,
                                open_trade["entry_time"],
                                open_trade.get("entry_signal_time"),
                                float(open_trade["entry_price"]),
                                float(open_trade["stop_price"]),
                                partial_target_price if np.isfinite(partial_target_price) else None,
                                timestamp,
                                float(open_trade["stop_price"]),
                                "stop",
                                float(open_trade["risk_usd"]),
                                int(open_trade["bars_held"]),
                                float(open_trade["remaining_fraction"]),
                                float(open_trade["realized_weighted_pnl_points"]),
                                execution_model,
                                instrument,
                                variant,
                            )
                            open_trade = None
                        elif partial_hit:
                            open_trade["realized_weighted_pnl_points"] = float(
                                open_trade["realized_weighted_pnl_points"]
                            ) + 0.5 * (float(partial_target_price) - float(open_trade["entry_price"])) * direction
                            open_trade["remaining_fraction"] = 0.5
                            open_trade["partial_filled"] = True
                            if direction == 1:
                                open_trade["stop_price"] = max(float(open_trade["stop_price"]), float(open_trade["entry_price"]))
                            else:
                                open_trade["stop_price"] = min(float(open_trade["stop_price"]), float(open_trade["entry_price"]))

                            if int(open_trade["bars_held"]) >= int(open_trade["time_stop_bars"]):
                                trade_id = _finalize_trade(
                                    trades,
                                    trade_id,
                                    session_date,
                                    direction,
                                    open_trade["entry_time"],
                                    open_trade.get("entry_signal_time"),
                                    float(open_trade["entry_price"]),
                                    float(open_trade["stop_price"]),
                                    float(partial_target_price),
                                    timestamp,
                                    close,
                                    "mixed_partial_time_stop",
                                    float(open_trade["risk_usd"]),
                                    int(open_trade["bars_held"]),
                                    float(open_trade["remaining_fraction"]),
                                    float(open_trade["realized_weighted_pnl_points"]),
                                    execution_model,
                                    instrument,
                                    variant,
                                )
                                open_trade = None
                            elif idx == last_idx:
                                trade_id = _finalize_trade(
                                    trades,
                                    trade_id,
                                    session_date,
                                    direction,
                                    open_trade["entry_time"],
                                    open_trade.get("entry_signal_time"),
                                    float(open_trade["entry_price"]),
                                    float(open_trade["stop_price"]),
                                    float(partial_target_price),
                                    timestamp,
                                    close,
                                    "mixed_partial_eod_flat",
                                    float(open_trade["risk_usd"]),
                                    int(open_trade["bars_held"]),
                                    float(open_trade["remaining_fraction"]),
                                    float(open_trade["realized_weighted_pnl_points"]),
                                    execution_model,
                                    instrument,
                                    variant,
                                )
                                open_trade = None
                    else:
                        if stop_hit:
                            trade_id = _finalize_trade(
                                trades,
                                trade_id,
                                session_date,
                                direction,
                                open_trade["entry_time"],
                                open_trade.get("entry_signal_time"),
                                float(open_trade["entry_price"]),
                                float(open_trade["stop_price"]),
                                _float_or_nan(open_trade.get("partial_target_price")),
                                timestamp,
                                float(open_trade["stop_price"]),
                                "mixed_trailing_stop",
                                float(open_trade["risk_usd"]),
                                int(open_trade["bars_held"]),
                                float(open_trade["remaining_fraction"]),
                                float(open_trade["realized_weighted_pnl_points"]),
                                execution_model,
                                instrument,
                                variant,
                            )
                            open_trade = None
                        elif int(open_trade["bars_held"]) >= int(open_trade["time_stop_bars"]):
                            trade_id = _finalize_trade(
                                trades,
                                trade_id,
                                session_date,
                                direction,
                                open_trade["entry_time"],
                                open_trade.get("entry_signal_time"),
                                float(open_trade["entry_price"]),
                                float(open_trade["stop_price"]),
                                _float_or_nan(open_trade.get("partial_target_price")),
                                timestamp,
                                close,
                                "mixed_partial_time_stop",
                                float(open_trade["risk_usd"]),
                                int(open_trade["bars_held"]),
                                float(open_trade["remaining_fraction"]),
                                float(open_trade["realized_weighted_pnl_points"]),
                                execution_model,
                                instrument,
                                variant,
                            )
                            open_trade = None
                        elif idx == last_idx:
                            trade_id = _finalize_trade(
                                trades,
                                trade_id,
                                session_date,
                                direction,
                                open_trade["entry_time"],
                                open_trade.get("entry_signal_time"),
                                float(open_trade["entry_price"]),
                                float(open_trade["stop_price"]),
                                _float_or_nan(open_trade.get("partial_target_price")),
                                timestamp,
                                close,
                                "mixed_partial_eod_flat",
                                float(open_trade["risk_usd"]),
                                int(open_trade["bars_held"]),
                                float(open_trade["remaining_fraction"]),
                                float(open_trade["realized_weighted_pnl_points"]),
                                execution_model,
                                instrument,
                                variant,
                            )
                            open_trade = None
                else:
                    stop_hit = (low <= float(open_trade["stop_price"])) if direction == 1 else (high >= float(open_trade["stop_price"]))
                    target_hit = (high >= float(open_trade["target_price"])) if direction == 1 else (low <= float(open_trade["target_price"]))
                    exit_price = None
                    exit_reason = None

                    if stop_hit and target_hit:
                        exit_price = float(open_trade["stop_price"])
                        exit_reason = "stop_ambiguous_first"
                    elif stop_hit:
                        exit_price = float(open_trade["stop_price"])
                        exit_reason = "stop"
                    elif target_hit:
                        exit_price = float(open_trade["target_price"])
                        exit_reason = "target"
                    elif int(open_trade["bars_held"]) >= int(open_trade["time_stop_bars"]):
                        exit_price = close
                        exit_reason = "time_stop"
                    elif idx == last_idx:
                        exit_price = close
                        exit_reason = "eod_flat"

                    if exit_price is not None:
                        trade_id = _finalize_trade(
                            trades,
                            trade_id,
                            session_date,
                            direction,
                            open_trade["entry_time"],
                            open_trade.get("entry_signal_time"),
                            float(open_trade["entry_price"]),
                            float(open_trade["stop_price"]),
                            _float_or_nan(open_trade.get("target_price")),
                            timestamp,
                            float(exit_price),
                            str(exit_reason),
                            float(open_trade["risk_usd"]),
                            int(open_trade["bars_held"]),
                            float(open_trade["remaining_fraction"]),
                            float(open_trade["realized_weighted_pnl_points"]),
                            execution_model,
                            instrument,
                            variant,
                        )
                        open_trade = None

                if open_trade is not None:
                    open_trade["last_close"] = close

            if trade_was_open_at_bar_start:
                continue

            if open_trade is None and pending_setup is not None and _bool_or_false(pending_setup.get("confirmed_for_next_open")):
                open_trade = _build_open_trade(
                    row=_entry_row_from_pending_setup(row, pending_setup),
                    direction=int(pending_setup["direction"]),
                    entry_price=_entry_market_fill(execution_model, open_, int(pending_setup["direction"])),
                    variant=variant,
                    execution_model=execution_model,
                    instrument=instrument,
                )
                pending_setup = None
                if open_trade is not None:
                    continue

            if open_trade is None and pending_setup is not None:
                pending_setup["bars_evaluated"] = int(pending_setup["bars_evaluated"]) + 1
                if _confirmation_triggered(
                    row=row,
                    direction=int(pending_setup["direction"]),
                    reference_close=float(pending_setup["reference_close"]),
                    reference_range=float(pending_setup["reference_range"]),
                    pullback_fraction=float(pending_setup["pullback_fraction"]),
                ):
                    pending_setup["confirmed_for_next_open"] = idx < last_idx
                    if idx == last_idx:
                        pending_setup = None
                elif int(pending_setup["bars_evaluated"]) >= int(pending_setup["confirmation_window"]):
                    pending_setup = None

            if open_trade is not None or pending_setup is not None:
                continue

            signal = int(_float_or_nan(row.get("signal")))
            if signal == 0:
                continue

            direction = int(signal)
            if variant.entry_mode == "next_open":
                open_trade = _build_open_trade(
                    row=row,
                    direction=direction,
                    entry_price=_entry_market_fill(execution_model, open_, direction),
                    variant=variant,
                    execution_model=execution_model,
                    instrument=instrument,
                )
                continue

            reference_close = _float_or_nan(row.get("setup_reference_close"))
            reference_range = _float_or_nan(row.get("setup_reference_range"))

            if variant.entry_mode == "pullback_limit":
                if variant.pullback_fraction is None:
                    continue
                limit_fill = _maybe_fill_pullback_limit(
                    row=row,
                    direction=direction,
                    reference_close=reference_close,
                    reference_range=reference_range,
                    pullback_fraction=float(variant.pullback_fraction),
                )
                if limit_fill is None:
                    continue
                open_trade = _build_open_trade(
                    row=row,
                    direction=direction,
                    entry_price=float(limit_fill),
                    variant=variant,
                    execution_model=execution_model,
                    instrument=instrument,
                )
                continue

            if variant.entry_mode == "confirmation":
                if variant.pullback_fraction is None or variant.confirmation_window is None:
                    continue
                pending_setup = {
                    "direction": direction,
                    "reference_close": reference_close,
                    "reference_range": reference_range,
                    "pullback_fraction": float(variant.pullback_fraction),
                    "confirmation_window": int(variant.confirmation_window),
                    "bars_evaluated": 1,
                    "confirmed_for_next_open": False,
                    "setup_stop_reference_long": row.get("setup_stop_reference_long"),
                    "setup_stop_reference_short": row.get("setup_stop_reference_short"),
                    "setup_reference_atr": row.get("setup_reference_atr"),
                    "setup_reference_vwap": row.get("setup_reference_vwap"),
                    "setup_signal_time": row.get("setup_signal_time"),
                }
                if _confirmation_triggered(
                    row=row,
                    direction=direction,
                    reference_close=reference_close,
                    reference_range=reference_range,
                    pullback_fraction=float(variant.pullback_fraction),
                ):
                    pending_setup["confirmed_for_next_open"] = idx < last_idx
                    if idx == last_idx:
                        pending_setup = None
                elif int(pending_setup["bars_evaluated"]) >= int(pending_setup["confirmation_window"]):
                    pending_setup = None
                continue

            raise ValueError(f"Unsupported entry_mode '{variant.entry_mode}'.")

    trades_df = pd.DataFrame(trades) if trades else empty_trade_log()
    if "variant_name" not in trades_df.columns:
        trades_df["variant_name"] = pd.Series(dtype=object)
    if "entry_signal_time" not in trades_df.columns:
        trades_df["entry_signal_time"] = pd.Series(dtype="datetime64[ns]")
    if "bars_held" not in trades_df.columns:
        trades_df["bars_held"] = pd.Series(dtype=float)
    return VolumeClimaxPullbackV2BacktestResult(trades=trades_df)
