"""Dedicated leak-free backtester for volume climax pullback V2 variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.engine.execution_model import ExecutionModel
from src.engine.trade_log import empty_trade_log, trade_to_record
from src.engine.vwap_backtester import InstrumentDetails
from src.risk.position_sizing import (
    PositionSizingConfig,
    PositionSizingDecision,
    compounds_realized_pnl,
    initial_capital_from_sizing,
    resolve_position_size,
    validate_position_sizing,
)
from src.strategy.volume_climax_pullback_v2 import VolumeClimaxPullbackV2Variant


@dataclass(frozen=True)
class VolumeClimaxPullbackV2BacktestResult:
    trades: pd.DataFrame
    sizing_decisions: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_volume_climax_pullback_v2_backtest_hybrid_1m(
    signal_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    variant: VolumeClimaxPullbackV2Variant,
    execution_model: ExecutionModel,
    instrument: InstrumentDetails,
    entry_timing: str = "next_execution_bar_open",
    protective_orders_active_from: str = "after_entry_fill",
) -> VolumeClimaxPullbackV2BacktestResult:
    """Hybrid backtest: 1H signal path with 1-minute execution path.

    This implementation preserves the existing alpha/signal construction and only changes
    fill/exit simulation to use minute bars.
    """
    if variant.entry_mode != "next_open":
        raise ValueError("Hybrid v1 currently supports variant.entry_mode='next_open' only.")
    if entry_timing not in {"next_execution_bar_open", "same_timestamp_execution_open"}:
        raise ValueError(f"Unsupported entry_timing '{entry_timing}'.")
    if protective_orders_active_from not in {"after_entry_fill", "next_execution_bar"}:
        raise ValueError(f"Unsupported protective_orders_active_from '{protective_orders_active_from}'.")

    minute = minute_df.copy()
    minute["timestamp"] = pd.to_datetime(minute["timestamp"], errors="coerce")
    minute = minute.sort_values("timestamp").reset_index(drop=True)
    minute_idx = minute.set_index("timestamp", drop=False)

    trades: list[dict[str, Any]] = []
    for row in signal_df.loc[pd.to_numeric(signal_df.get("signal"), errors="coerce").fillna(0).ne(0)].itertuples(index=False):
        direction = int(getattr(row, "signal"))
        setup_start = pd.Timestamp(getattr(row, "setup_signal_time"))
        signal_actionable_time = setup_start + pd.Timedelta(hours=1)

        if entry_timing == "same_timestamp_execution_open":
            candidates = minute.loc[minute["timestamp"] >= signal_actionable_time]
        else:
            candidates = minute.loc[minute["timestamp"] > signal_actionable_time]
        if candidates.empty:
            continue
        entry_bar = candidates.iloc[0]
        entry_time = pd.Timestamp(entry_bar["timestamp"])
        entry_raw = float(entry_bar["open"])
        entry_price = float(execution_model.apply_slippage(entry_raw, direction, is_entry=True))

        tmp_row = {
            "session_date": getattr(row, "session_date", pd.Timestamp(entry_time).date()),
            "entry_signal_time": getattr(row, "timestamp", pd.NaT),
            "setup_signal_time": setup_start,
            "setup_stop_reference_long": getattr(row, "setup_stop_reference_long", np.nan),
            "setup_stop_reference_short": getattr(row, "setup_stop_reference_short", np.nan),
            "setup_reference_atr": getattr(row, "setup_reference_atr", np.nan),
            "setup_reference_vwap": getattr(row, "setup_reference_vwap", np.nan),
            "setup_reference_range": getattr(row, "setup_reference_range", np.nan),
            "timestamp": entry_time,
        }
        open_trade = _build_open_trade(
            row=tmp_row,
            direction=direction,
            entry_price=entry_price,
            variant=variant,
            instrument=instrument,
            position_sizing=None,
            capital_before_trade_usd=None,
            sizing_decisions=[],
        )
        if open_trade is None:
            continue
        open_trade["entry_time"] = entry_time

        protect_from = entry_time if protective_orders_active_from == "after_entry_fill" else entry_time + pd.Timedelta(minutes=1)
        max_hold = pd.Timedelta(hours=int(variant.time_stop_bars))
        time_stop_at = entry_time + max_hold
        post = minute_idx.loc[minute_idx.index >= entry_time]
        if post.empty:
            continue
        exit_time = None
        exit_price = None
        exit_reason = None
        for _, m in post.iterrows():
            ts = pd.Timestamp(m["timestamp"])
            if ts >= protect_from:
                low = float(m["low"])
                high = float(m["high"])
                stop_hit = low <= float(open_trade["stop_price"]) if direction == 1 else high >= float(open_trade["stop_price"])
                target_hit = high >= float(open_trade["target_price"]) if direction == 1 else low <= float(open_trade["target_price"])
                if stop_hit and target_hit:
                    exit_time, exit_price, exit_reason = ts, float(open_trade["stop_price"]), "stop_ambiguous_first_1m"
                    break
                if stop_hit:
                    exit_time, exit_price, exit_reason = ts, float(open_trade["stop_price"]), "stop_1m"
                    break
                if target_hit:
                    exit_time, exit_price, exit_reason = ts, float(open_trade["target_price"]), "target_1m"
                    break
            if ts >= time_stop_at:
                exit_time, exit_price, exit_reason = ts, float(m["close"]), "time_stop_1m"
                break
        if exit_time is None:
            last = post.iloc[-1]
            exit_time, exit_price, exit_reason = pd.Timestamp(last["timestamp"]), float(last["close"]), "eod_flat_1m"

        rec = _trade_record(
            trade_id=len(trades) + 1,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=exit_reason,
            instrument=instrument,
            execution_model=execution_model,
            variant=variant,
            open_trade=open_trade,
        )
        rec.update(
            {
                "symbol": instrument.symbol,
                "setup_bar_label_time": setup_start,
                "setup_bar_start_time": setup_start,
                "setup_bar_close_time": signal_actionable_time,
                "signal_actionable_time": signal_actionable_time,
                "entry_time": entry_time,
                "entry_timing": entry_timing,
                "protective_orders_active_from": protective_orders_active_from,
                "protective_orders_active_at": protect_from,
                "time_stop_at": time_stop_at,
                "source_execution_timeframe": "1min",
                "bars_held_1m": int(max(0, ((exit_time - entry_time).total_seconds() // 60) + 1)),
                "minutes_held": float(max(0.0, (exit_time - entry_time).total_seconds() / 60.0)),
            }
        )
        trades.append(rec)

    return VolumeClimaxPullbackV2BacktestResult(trades=pd.DataFrame(trades) if trades else empty_trade_log(), sizing_decisions=_empty_sizing_decision_log())


SIZING_DECISION_COLUMNS = [
    "variant_name",
    "session_date",
    "entry_time",
    "entry_signal_time",
    "direction",
    "entry_price",
    "initial_stop_price",
    "position_sizing_mode",
    "capital_before_trade_usd",
    "risk_pct",
    "risk_budget_usd",
    "stop_distance_points",
    "risk_per_contract_usd",
    "contracts_raw",
    "contracts",
    "actual_risk_usd",
    "notional_usd",
    "leverage_used",
    "max_contracts",
    "skip_trade_if_too_small",
    "skipped",
    "skip_reason",
]


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


def _empty_sizing_decision_log() -> pd.DataFrame:
    return pd.DataFrame(columns=SIZING_DECISION_COLUMNS)


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


def _append_sizing_decision(
    sizing_decisions: list[dict[str, Any]],
    *,
    variant: VolumeClimaxPullbackV2Variant,
    session_date: Any,
    entry_time: pd.Timestamp,
    entry_signal_time: pd.Timestamp | None,
    direction: int,
    entry_price: float,
    initial_stop_price: float | None,
    decision: PositionSizingDecision | None = None,
    skip_reason: str | None = None,
    position_sizing_mode: str | None = None,
) -> None:
    record = {column: np.nan for column in SIZING_DECISION_COLUMNS}
    record.update(
        {
            "variant_name": variant.name,
            "session_date": session_date,
            "entry_time": entry_time,
            "entry_signal_time": entry_signal_time,
            "direction": "long" if int(direction) == 1 else "short",
            "entry_price": float(entry_price),
            "initial_stop_price": float(initial_stop_price) if initial_stop_price is not None and np.isfinite(initial_stop_price) else np.nan,
            "position_sizing_mode": position_sizing_mode,
            "skipped": True,
            "skip_reason": skip_reason,
        }
    )
    if initial_stop_price is not None and np.isfinite(initial_stop_price):
        record["stop_distance_points"] = abs(float(entry_price) - float(initial_stop_price))
    if decision is not None:
        record.update(
            {
                "position_sizing_mode": decision.position_sizing_mode,
                "capital_before_trade_usd": decision.capital_before_trade_usd,
                "risk_pct": decision.risk_pct,
                "risk_budget_usd": decision.risk_budget_usd,
                "stop_distance_points": decision.stop_distance_points,
                "risk_per_contract_usd": decision.risk_per_contract_usd,
                "contracts_raw": decision.contracts_raw,
                "contracts": int(decision.contracts),
                "actual_risk_usd": decision.actual_risk_usd,
                "notional_usd": decision.notional_usd,
                "leverage_used": decision.leverage_used,
                "max_contracts": decision.max_contracts,
                "skip_trade_if_too_small": decision.skip_trade_if_too_small,
                "skipped": bool(decision.skipped),
                "skip_reason": decision.skip_reason,
            }
        )
    sizing_decisions.append(record)


def _trade_record(
    trade_id: int,
    exit_time: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
    instrument: InstrumentDetails,
    execution_model: ExecutionModel,
    variant: VolumeClimaxPullbackV2Variant,
    open_trade: dict[str, Any],
) -> dict[str, Any]:
    direction = int(open_trade["direction"])
    quantity = int(open_trade["quantity"])
    entry_price = float(open_trade["entry_price"])
    pnl_points = (float(exit_price) - float(entry_price)) * direction
    gross = pnl_points * float(instrument.point_value_usd) * float(quantity)
    fees = execution_model.round_trip_fees(quantity=quantity)
    net = gross - fees
    capital_before_trade = _float_or_nan(open_trade.get("account_size_usd"))
    capital_after_trade = capital_before_trade + float(net) if np.isfinite(capital_before_trade) else np.nan
    record = trade_to_record(
        trade_id,
        {
            "session_date": open_trade["session_date"],
            "direction": "long" if direction == 1 else "short",
            "quantity": int(quantity),
            "entry_time": open_trade["entry_time"],
            "entry_price": float(entry_price),
            "stop_price": float(open_trade["stop_price"]),
            "target_price": float(open_trade["target_price"])
            if np.isfinite(_float_or_nan(open_trade.get("target_price")))
            else np.nan,
            "exit_time": exit_time,
            "exit_price": float(exit_price),
            "exit_reason": exit_reason,
            "account_size_usd": capital_before_trade,
            "risk_per_trade_pct": _float_or_nan(open_trade.get("risk_per_trade_pct")),
            "risk_budget_usd": _float_or_nan(open_trade.get("risk_budget_usd")),
            "risk_per_contract_usd": _float_or_nan(open_trade.get("risk_per_contract_usd")),
            "actual_risk_usd": _float_or_nan(open_trade.get("actual_risk_usd")),
            "trade_risk_usd": _float_or_nan(open_trade.get("trade_risk_usd")),
            "notional_usd": _float_or_nan(open_trade.get("notional_usd")),
            "leverage_used": _float_or_nan(open_trade.get("leverage_used")),
            "pnl_points": float(pnl_points),
            "pnl_ticks": float(pnl_points / instrument.tick_size) if instrument.tick_size > 0 else np.nan,
            "pnl_usd": float(gross),
            "fees": float(fees),
            "net_pnl_usd": float(net),
        },
    )
    record["variant_name"] = variant.name
    record["entry_signal_time"] = open_trade.get("entry_signal_time")
    record["bars_held"] = int(open_trade["bars_held"])
    record["position_sizing_mode"] = open_trade.get("position_sizing_mode")
    record["risk_pct_decimal"] = _float_or_nan(open_trade.get("risk_pct"))
    record["stop_distance_points"] = _float_or_nan(open_trade.get("stop_distance_points"))
    record["contracts_raw"] = _float_or_nan(open_trade.get("contracts_raw"))
    record["max_contracts"] = _float_or_nan(open_trade.get("max_contracts"))
    record["skip_trade_if_too_small"] = open_trade.get("skip_trade_if_too_small")
    record["capital_after_trade_usd"] = capital_after_trade
    record["initial_stop_price"] = _float_or_nan(open_trade.get("initial_stop_price"))
    return record


def _build_open_trade(
    *,
    row: pd.Series,
    direction: int,
    entry_price: float,
    variant: VolumeClimaxPullbackV2Variant,
    instrument: InstrumentDetails,
    position_sizing: PositionSizingConfig | None,
    capital_before_trade_usd: float | None,
    sizing_decisions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    stop_col = "setup_stop_reference_long" if direction == 1 else "setup_stop_reference_short"
    stop_price = _float_or_nan(row.get(stop_col))
    reference_atr = _float_or_nan(row.get("setup_reference_atr"))
    reference_vwap = _float_or_nan(row.get("setup_reference_vwap"))
    entry_time = pd.Timestamp(row["timestamp"])
    signal_time_raw = row.get("setup_signal_time")
    entry_signal_time = pd.Timestamp(signal_time_raw) if pd.notna(signal_time_raw) else None
    session_date = row.get("session_date")

    if not np.isfinite(stop_price):
        _append_sizing_decision(
            sizing_decisions,
            variant=variant,
            session_date=session_date,
            entry_time=entry_time,
            entry_signal_time=entry_signal_time,
            direction=direction,
            entry_price=float(entry_price),
            initial_stop_price=None,
            skip_reason="invalid_initial_stop",
            position_sizing_mode="unresolved",
        )
        return None

    risk_points = (float(entry_price) - float(stop_price)) * int(direction)
    if risk_points <= 0:
        _append_sizing_decision(
            sizing_decisions,
            variant=variant,
            session_date=session_date,
            entry_time=entry_time,
            entry_signal_time=entry_signal_time,
            direction=direction,
            entry_price=float(entry_price),
            initial_stop_price=float(stop_price),
            skip_reason="non_positive_stop_distance",
            position_sizing_mode="unresolved",
        )
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
        _append_sizing_decision(
            sizing_decisions,
            variant=variant,
            session_date=session_date,
            entry_time=entry_time,
            entry_signal_time=entry_signal_time,
            direction=direction,
            entry_price=float(entry_price),
            initial_stop_price=float(stop_price),
            skip_reason="invalid_target",
            position_sizing_mode="unresolved",
        )
        return None

    sizing = resolve_position_size(
        config=position_sizing,
        capital_before_trade_usd=capital_before_trade_usd,
        entry_price=float(entry_price),
        initial_stop_price=float(stop_price),
        point_value_usd=float(instrument.point_value_usd),
    )
    _append_sizing_decision(
        sizing_decisions,
        variant=variant,
        session_date=session_date,
        entry_time=entry_time,
        entry_signal_time=entry_signal_time,
        direction=direction,
        entry_price=float(entry_price),
        initial_stop_price=float(stop_price),
        decision=sizing,
    )
    if sizing.skipped or int(sizing.contracts) < 1:
        return None

    return {
        "entry_time": entry_time,
        "entry_signal_time": entry_signal_time,
        "entry_price": float(entry_price),
        "direction": int(direction),
        "quantity": int(sizing.contracts),
        "stop_price": float(stop_price),
        "initial_stop_price": float(stop_price),
        "target_price": float(target_price) if target_price is not None else np.nan,
        "partial_target_price": float(partial_target_price) if partial_target_price is not None else np.nan,
        "trailing_offset": float(trailing_offset) if trailing_offset is not None else np.nan,
        "time_stop_bars": int(variant.time_stop_bars),
        "bars_held": 0,
        "account_size_usd": float(sizing.capital_before_trade_usd) if sizing.capital_before_trade_usd is not None else np.nan,
        "risk_per_trade_pct": float(sizing.risk_pct * 100.0) if sizing.risk_pct is not None else np.nan,
        "risk_pct": float(sizing.risk_pct) if sizing.risk_pct is not None else np.nan,
        "risk_budget_usd": float(sizing.risk_budget_usd) if sizing.risk_budget_usd is not None else np.nan,
        "risk_per_contract_usd": float(sizing.risk_per_contract_usd) if sizing.risk_per_contract_usd is not None else np.nan,
        "actual_risk_usd": float(sizing.actual_risk_usd) if sizing.actual_risk_usd is not None else np.nan,
        "trade_risk_usd": float(sizing.actual_risk_usd) if sizing.actual_risk_usd is not None else np.nan,
        "notional_usd": float(sizing.notional_usd) if sizing.notional_usd is not None else np.nan,
        "leverage_used": float(sizing.leverage_used) if sizing.leverage_used is not None else np.nan,
        "position_sizing_mode": sizing.position_sizing_mode,
        "contracts_raw": float(sizing.contracts_raw) if sizing.contracts_raw is not None else np.nan,
        "stop_distance_points": float(sizing.stop_distance_points) if sizing.stop_distance_points is not None else np.nan,
        "max_contracts": float(sizing.max_contracts) if sizing.max_contracts is not None else np.nan,
        "skip_trade_if_too_small": sizing.skip_trade_if_too_small,
        "reference_atr": float(reference_atr) if np.isfinite(reference_atr) else np.nan,
        "partial_filled": False,
        "remaining_fraction": 1.0,
        "realized_weighted_pnl_points": 0.0,
        "last_close": _float_or_nan(row.get("close")),
        "session_date": session_date,
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
    exit_time: pd.Timestamp,
    raw_exit_price: float,
    exit_reason: str,
    execution_model: ExecutionModel,
    instrument: InstrumentDetails,
    variant: VolumeClimaxPullbackV2Variant,
    open_trade: dict[str, Any],
) -> int:
    direction = int(open_trade["direction"])
    entry_price = float(open_trade["entry_price"])
    fill_exit = float(execution_model.apply_slippage(raw_exit_price, direction, is_entry=False))
    weighted_final_points = float(open_trade["remaining_fraction"]) * (fill_exit - float(entry_price)) * int(direction)
    total_weighted_points = float(open_trade["realized_weighted_pnl_points"]) + weighted_final_points
    effective_exit_price = float(entry_price) + (float(total_weighted_points) * int(direction))
    trades.append(
        _trade_record(
            trade_id=trade_id,
            exit_time=exit_time,
            exit_price=effective_exit_price,
            exit_reason=exit_reason,
            instrument=instrument,
            execution_model=execution_model,
            variant=variant,
            open_trade=open_trade,
        )
    )
    return trade_id + 1


def run_volume_climax_pullback_v2_backtest(
    signal_df: pd.DataFrame,
    variant: VolumeClimaxPullbackV2Variant,
    execution_model: ExecutionModel,
    instrument: InstrumentDetails,
    position_sizing: PositionSizingConfig | None = None,
) -> VolumeClimaxPullbackV2BacktestResult:
    validate_position_sizing(position_sizing)
    trades: list[dict[str, Any]] = []
    sizing_decisions: list[dict[str, Any]] = []
    open_trade: dict[str, Any] | None = None
    trade_id = 1
    capital_before_trade_usd = initial_capital_from_sizing(position_sizing)
    compound_realized_pnl = compounds_realized_pnl(position_sizing)

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
                                trades=trades,
                                trade_id=trade_id,
                                exit_time=timestamp,
                                raw_exit_price=float(open_trade["stop_price"]),
                                exit_reason="stop_ambiguous_first",
                                execution_model=execution_model,
                                instrument=instrument,
                                variant=variant,
                                open_trade=open_trade,
                            )
                            if compound_realized_pnl and capital_before_trade_usd is not None:
                                capital_before_trade_usd += float(trades[-1]["net_pnl_usd"])
                            open_trade = None
                        elif stop_hit:
                            trade_id = _finalize_trade(
                                trades=trades,
                                trade_id=trade_id,
                                exit_time=timestamp,
                                raw_exit_price=float(open_trade["stop_price"]),
                                exit_reason="stop",
                                execution_model=execution_model,
                                instrument=instrument,
                                variant=variant,
                                open_trade=open_trade,
                            )
                            if compound_realized_pnl and capital_before_trade_usd is not None:
                                capital_before_trade_usd += float(trades[-1]["net_pnl_usd"])
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
                                    trades=trades,
                                    trade_id=trade_id,
                                    exit_time=timestamp,
                                    raw_exit_price=close,
                                    exit_reason="mixed_partial_time_stop",
                                    execution_model=execution_model,
                                    instrument=instrument,
                                    variant=variant,
                                    open_trade=open_trade,
                                )
                                if compound_realized_pnl and capital_before_trade_usd is not None:
                                    capital_before_trade_usd += float(trades[-1]["net_pnl_usd"])
                                open_trade = None
                            elif idx == last_idx:
                                trade_id = _finalize_trade(
                                    trades=trades,
                                    trade_id=trade_id,
                                    exit_time=timestamp,
                                    raw_exit_price=close,
                                    exit_reason="mixed_partial_eod_flat",
                                    execution_model=execution_model,
                                    instrument=instrument,
                                    variant=variant,
                                    open_trade=open_trade,
                                )
                                if compound_realized_pnl and capital_before_trade_usd is not None:
                                    capital_before_trade_usd += float(trades[-1]["net_pnl_usd"])
                                open_trade = None
                    else:
                        if stop_hit:
                            trade_id = _finalize_trade(
                                trades=trades,
                                trade_id=trade_id,
                                exit_time=timestamp,
                                raw_exit_price=float(open_trade["stop_price"]),
                                exit_reason="mixed_trailing_stop",
                                execution_model=execution_model,
                                instrument=instrument,
                                variant=variant,
                                open_trade=open_trade,
                            )
                            if compound_realized_pnl and capital_before_trade_usd is not None:
                                capital_before_trade_usd += float(trades[-1]["net_pnl_usd"])
                            open_trade = None
                        elif int(open_trade["bars_held"]) >= int(open_trade["time_stop_bars"]):
                            trade_id = _finalize_trade(
                                trades=trades,
                                trade_id=trade_id,
                                exit_time=timestamp,
                                raw_exit_price=close,
                                exit_reason="mixed_partial_time_stop",
                                execution_model=execution_model,
                                instrument=instrument,
                                variant=variant,
                                open_trade=open_trade,
                            )
                            if compound_realized_pnl and capital_before_trade_usd is not None:
                                capital_before_trade_usd += float(trades[-1]["net_pnl_usd"])
                            open_trade = None
                        elif idx == last_idx:
                            trade_id = _finalize_trade(
                                trades=trades,
                                trade_id=trade_id,
                                exit_time=timestamp,
                                raw_exit_price=close,
                                exit_reason="mixed_partial_eod_flat",
                                execution_model=execution_model,
                                instrument=instrument,
                                variant=variant,
                                open_trade=open_trade,
                            )
                            if compound_realized_pnl and capital_before_trade_usd is not None:
                                capital_before_trade_usd += float(trades[-1]["net_pnl_usd"])
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
                            trades=trades,
                            trade_id=trade_id,
                            exit_time=timestamp,
                            raw_exit_price=float(exit_price),
                            exit_reason=str(exit_reason),
                            execution_model=execution_model,
                            instrument=instrument,
                            variant=variant,
                            open_trade=open_trade,
                        )
                        if compound_realized_pnl and capital_before_trade_usd is not None:
                            capital_before_trade_usd += float(trades[-1]["net_pnl_usd"])
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
                    instrument=instrument,
                    position_sizing=position_sizing,
                    capital_before_trade_usd=capital_before_trade_usd,
                    sizing_decisions=sizing_decisions,
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
                    instrument=instrument,
                    position_sizing=position_sizing,
                    capital_before_trade_usd=capital_before_trade_usd,
                    sizing_decisions=sizing_decisions,
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
                    instrument=instrument,
                    position_sizing=position_sizing,
                    capital_before_trade_usd=capital_before_trade_usd,
                    sizing_decisions=sizing_decisions,
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
    sizing_df = pd.DataFrame(sizing_decisions) if sizing_decisions else _empty_sizing_decision_log()
    if "variant_name" not in trades_df.columns:
        trades_df["variant_name"] = pd.Series(dtype=object)
    if "entry_signal_time" not in trades_df.columns:
        trades_df["entry_signal_time"] = pd.Series(dtype="datetime64[ns]")
    if "bars_held" not in trades_df.columns:
        trades_df["bars_held"] = pd.Series(dtype=float)
    for column in (
        "position_sizing_mode",
        "risk_pct_decimal",
        "stop_distance_points",
        "contracts_raw",
        "max_contracts",
        "skip_trade_if_too_small",
        "capital_after_trade_usd",
        "initial_stop_price",
    ):
        if column not in trades_df.columns:
            trades_df[column] = pd.Series(dtype=float if column not in {"position_sizing_mode", "skip_trade_if_too_small"} else object)
    return VolumeClimaxPullbackV2BacktestResult(trades=trades_df, sizing_decisions=sizing_df)
