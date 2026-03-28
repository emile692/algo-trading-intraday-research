"""Leak-free intraday PnL overlay engine for fixed-nominal ORB research."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.orb_multi_asset_campaign import BaselineSpec
from src.engine.execution_model import ExecutionModel
from src.engine.trade_log import empty_trade_log, trade_to_record


HALT_STATES = {"halted", "locked_profit"}


@dataclass(frozen=True)
class IntradayPnlOverlaySpec:
    """Readable overlay definition applied on top of a frozen baseline."""

    name: str
    family: str
    description: str
    threshold_unit: str = "R"  # R | usd
    hard_loss_cap: float | None = None
    hard_profit_lock: float | None = None
    giveback_activation: float | None = None
    giveback_threshold: float | None = None
    giveback_locked_profit_only: bool = True
    max_trades_per_day: int | None = None
    continue_only_if_first_trade_wins: bool = False
    defensive_after_total_losses: int | None = None
    defensive_after_consecutive_losses: int | None = None
    defensive_multiplier: float = 0.5
    halt_after_total_losses: int | None = None
    halt_after_consecutive_losses: int | None = None


@dataclass
class OverlayBacktestResult:
    """Simulation bundle exported by the intraday overlay engine."""

    trades: pd.DataFrame
    controls: pd.DataFrame
    daily_results: pd.DataFrame
    state_transitions: pd.DataFrame


@dataclass
class _DayState:
    state_name: str = "neutral"
    realized_pnl_usd: float = 0.0
    realized_pnl_r: float = 0.0
    peak_day_pnl_usd: float = 0.0
    peak_day_pnl_r: float = 0.0
    max_giveback_usd: float = 0.0
    max_giveback_r: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    first_trade_result: str | None = None
    day_cut_by_rule: bool = False
    hard_loss_cap_triggered: bool = False
    hard_profit_lock_triggered: bool = False
    giveback_triggered: bool = False
    max_trades_cap_triggered: bool = False
    first_trade_gate_triggered: bool = False
    policy_block_count: int = 0
    final_block_reason: str = ""


def _normalize_unit(unit: str) -> str:
    clean = str(unit).strip().lower()
    if clean in {"r", "risk"}:
        return "R"
    if clean in {"usd", "$", "dollar", "dollars"}:
        return "usd"
    raise ValueError("threshold_unit must be one of R, risk, usd, $, dollar, dollars.")


def _mark_to_market_net_pnl_usd(
    direction: int,
    entry_price: float,
    reference_price: float,
    quantity: int,
    execution_model: ExecutionModel,
    tick_value_usd: float,
) -> float:
    exit_price = execution_model.apply_slippage(reference_price, direction, is_entry=False)
    pnl_points = (exit_price - entry_price) * direction
    pnl_ticks = pnl_points / execution_model.tick_size
    gross = pnl_ticks * tick_value_usd * quantity
    fees = execution_model.round_trip_fees(quantity=quantity)
    return float(gross - fees)


def _best_intrabar_reference_price(direction: int, high: float, low: float) -> float:
    return float(high) if direction == 1 else float(low)


def _quantity_from_multiplier(fixed_contracts: int, multiplier: float) -> int:
    if fixed_contracts <= 0 or multiplier <= 0:
        return 0
    return int(math.floor(float(fixed_contracts) * float(multiplier) + 1e-12))


def _day_metric_value(unit: str, usd_value: float, r_value: float) -> float:
    return float(r_value) if unit == "R" else float(usd_value)


def _update_peak_and_giveback(
    state: _DayState,
    current_day_pnl_usd: float,
    current_day_pnl_r: float,
    best_day_pnl_usd: float,
    best_day_pnl_r: float,
) -> None:
    state.peak_day_pnl_usd = max(float(state.peak_day_pnl_usd), float(best_day_pnl_usd))
    state.peak_day_pnl_r = max(float(state.peak_day_pnl_r), float(best_day_pnl_r))
    state.max_giveback_usd = max(float(state.max_giveback_usd), float(state.peak_day_pnl_usd - current_day_pnl_usd))
    state.max_giveback_r = max(float(state.max_giveback_r), float(state.peak_day_pnl_r - current_day_pnl_r))


def _append_transition(
    transitions: list[dict[str, Any]],
    variant_name: str,
    session_date: Any,
    timestamp: Any,
    from_state: str,
    to_state: str,
    trigger: str,
    current_metric_usd: float,
    current_metric_r: float,
    peak_metric_usd: float,
    peak_metric_r: float,
) -> None:
    if from_state == to_state:
        return
    transitions.append(
        {
            "variant_name": variant_name,
            "session_date": pd.to_datetime(session_date).date(),
            "timestamp": pd.to_datetime(timestamp),
            "from_state": from_state,
            "to_state": to_state,
            "trigger": trigger,
            "current_day_pnl_usd": float(current_metric_usd),
            "current_day_pnl_r": float(current_metric_r),
            "peak_day_pnl_usd": float(peak_metric_usd),
            "peak_day_pnl_r": float(peak_metric_r),
        }
    )


def _set_state(
    state: _DayState,
    new_state: str,
    *,
    transitions: list[dict[str, Any]],
    variant_name: str,
    session_date: Any,
    timestamp: Any,
    trigger: str,
    current_metric_usd: float,
    current_metric_r: float,
) -> None:
    previous = state.state_name
    state.state_name = str(new_state)
    _append_transition(
        transitions=transitions,
        variant_name=variant_name,
        session_date=session_date,
        timestamp=timestamp,
        from_state=previous,
        to_state=state.state_name,
        trigger=trigger,
        current_metric_usd=current_metric_usd,
        current_metric_r=current_metric_r,
        peak_metric_usd=state.peak_day_pnl_usd,
        peak_metric_r=state.peak_day_pnl_r,
    )


def _pre_trade_policy(
    state: _DayState,
    overlay: IntradayPnlOverlaySpec,
) -> tuple[bool, float, str]:
    if state.state_name in HALT_STATES:
        return False, 0.0, f"state_{state.state_name}"
    if overlay.max_trades_per_day is not None and state.trade_count >= int(overlay.max_trades_per_day):
        state.max_trades_cap_triggered = True
        state.final_block_reason = "max_trades_per_day"
        return False, 0.0, "max_trades_per_day"
    if overlay.continue_only_if_first_trade_wins and state.first_trade_result == "loss":
        state.first_trade_gate_triggered = True
        state.final_block_reason = "continue_only_if_first_trade_wins"
        return False, 0.0, "continue_only_if_first_trade_wins"
    multiplier = float(overlay.defensive_multiplier) if state.state_name == "defensive" else 1.0
    return True, multiplier, "allowed"


def _apply_post_trade_state_rules(
    state: _DayState,
    overlay: IntradayPnlOverlaySpec,
    *,
    variant_name: str,
    session_date: Any,
    timestamp: Any,
    transitions: list[dict[str, Any]],
) -> None:
    current_usd = float(state.realized_pnl_usd)
    current_r = float(state.realized_pnl_r)

    if overlay.halt_after_consecutive_losses is not None and state.consecutive_losses >= int(
        overlay.halt_after_consecutive_losses
    ):
        state.final_block_reason = "halt_after_consecutive_losses"
        _set_state(
            state,
            "halted",
            transitions=transitions,
            variant_name=variant_name,
            session_date=session_date,
            timestamp=timestamp,
            trigger="halt_after_consecutive_losses",
            current_metric_usd=current_usd,
            current_metric_r=current_r,
        )
        return

    if overlay.halt_after_total_losses is not None and state.loss_count >= int(overlay.halt_after_total_losses):
        state.final_block_reason = "halt_after_total_losses"
        _set_state(
            state,
            "halted",
            transitions=transitions,
            variant_name=variant_name,
            session_date=session_date,
            timestamp=timestamp,
            trigger="halt_after_total_losses",
            current_metric_usd=current_usd,
            current_metric_r=current_r,
        )
        return

    if overlay.continue_only_if_first_trade_wins and state.first_trade_result == "loss":
        state.first_trade_gate_triggered = True
        state.final_block_reason = "continue_only_if_first_trade_wins"
        _set_state(
            state,
            "halted",
            transitions=transitions,
            variant_name=variant_name,
            session_date=session_date,
            timestamp=timestamp,
            trigger="continue_only_if_first_trade_wins",
            current_metric_usd=current_usd,
            current_metric_r=current_r,
        )
        return

    if overlay.max_trades_per_day is not None and state.trade_count >= int(overlay.max_trades_per_day):
        state.max_trades_cap_triggered = True
        state.final_block_reason = "max_trades_per_day"
        _set_state(
            state,
            "halted",
            transitions=transitions,
            variant_name=variant_name,
            session_date=session_date,
            timestamp=timestamp,
            trigger="max_trades_per_day",
            current_metric_usd=current_usd,
            current_metric_r=current_r,
        )
        return

    if state.state_name not in HALT_STATES:
        if overlay.defensive_after_consecutive_losses is not None and state.consecutive_losses >= int(
            overlay.defensive_after_consecutive_losses
        ):
            _set_state(
                state,
                "defensive",
                transitions=transitions,
                variant_name=variant_name,
                session_date=session_date,
                timestamp=timestamp,
                trigger="defensive_after_consecutive_losses",
                current_metric_usd=current_usd,
                current_metric_r=current_r,
            )
            return

        if overlay.defensive_after_total_losses is not None and state.loss_count >= int(
            overlay.defensive_after_total_losses
        ):
            _set_state(
                state,
                "defensive",
                transitions=transitions,
                variant_name=variant_name,
                session_date=session_date,
                timestamp=timestamp,
                trigger="defensive_after_total_losses",
                current_metric_usd=current_usd,
                current_metric_r=current_r,
            )


def _overlay_exit_signal(
    state: _DayState,
    overlay: IntradayPnlOverlaySpec,
    *,
    current_day_pnl_usd: float,
    current_day_pnl_r: float,
) -> tuple[str | None, str | None]:
    unit = _normalize_unit(overlay.threshold_unit)
    current_metric = _day_metric_value(unit, current_day_pnl_usd, current_day_pnl_r)
    peak_metric = _day_metric_value(unit, state.peak_day_pnl_usd, state.peak_day_pnl_r)

    if overlay.hard_loss_cap is not None and current_metric <= float(overlay.hard_loss_cap):
        state.day_cut_by_rule = True
        state.hard_loss_cap_triggered = True
        return "overlay_hard_loss_cap", "halted"

    if overlay.hard_profit_lock is not None and current_metric >= float(overlay.hard_profit_lock):
        state.day_cut_by_rule = True
        state.hard_profit_lock_triggered = True
        return "overlay_hard_profit_lock", "locked_profit"

    if overlay.giveback_activation is not None and overlay.giveback_threshold is not None:
        if peak_metric >= float(overlay.giveback_activation):
            giveback = float(peak_metric - current_metric)
            if giveback >= float(overlay.giveback_threshold):
                state.day_cut_by_rule = True
                state.giveback_triggered = True
                if overlay.giveback_locked_profit_only and current_metric > 0:
                    return "overlay_peak_giveback_lock", "locked_profit"
                return "overlay_peak_giveback_halt", "halted"

    return None, None


def _empty_controls() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "variant_name",
            "session_date",
            "signal_time",
            "signal_index",
            "state_pre",
            "state_post",
            "policy_multiplier",
            "allowed_to_trade",
            "blocked_reason",
            "trade_executed",
            "quantity",
            "trade_number_in_day",
            "exit_reason",
            "overlay_exit_reason",
            "net_pnl_usd",
            "net_pnl_r",
        ]
    )


def run_intraday_pnl_overlay_backtest(
    signal_df: pd.DataFrame,
    execution_model: ExecutionModel,
    baseline: BaselineSpec,
    overlay: IntradayPnlOverlaySpec,
    *,
    fixed_contracts: int = 1,
    tick_value_usd: float,
) -> OverlayBacktestResult:
    """Run the official entry logic with a fixed-nominal intraday overlay.

    Overlay decisions are evaluated at bar close using mark-to-market net PnL,
    while baseline stops/targets remain intrabar with the same slippage/cost model.
    """

    unit = _normalize_unit(overlay.threshold_unit)
    force_exit_time = dt.time.fromisoformat(str(baseline.time_exit))
    stop_buffer_points = float(baseline.stop_buffer_ticks) * float(execution_model.tick_size)

    trades: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    trade_id = 1

    for session_date, session_df in signal_df.groupby("session_date", sort=True):
        frame = session_df.sort_values("timestamp").reset_index(drop=True)
        signal_indices = frame.index[frame["signal"].fillna(0).ne(0)].tolist()
        state = _DayState()
        next_search_index = 0
        trade_number_in_day = 0
        transition_count_before = len(transitions)

        for signal_idx in signal_indices:
            if int(signal_idx) < int(next_search_index):
                continue

            signal_row = frame.loc[signal_idx]
            direction = int(signal_row["signal"])
            or_high = pd.to_numeric(signal_row.get("or_high"), errors="coerce")
            or_low = pd.to_numeric(signal_row.get("or_low"), errors="coerce")
            if pd.isna(or_high) or pd.isna(or_low):
                continue

            allowed, policy_multiplier, block_reason = _pre_trade_policy(state, overlay)
            quantity = _quantity_from_multiplier(int(fixed_contracts), policy_multiplier) if allowed else 0
            if allowed and quantity < 1:
                allowed = False
                block_reason = "policy_multiplier_rounds_to_zero"
                state.policy_block_count += 1
                state.final_block_reason = block_reason

            if not allowed:
                control_rows.append(
                    {
                        "variant_name": overlay.name,
                        "session_date": pd.to_datetime(session_date).date(),
                        "signal_time": pd.to_datetime(signal_row["timestamp"]),
                        "signal_index": int(signal_idx),
                        "state_pre": state.state_name,
                        "state_post": state.state_name,
                        "policy_multiplier": float(policy_multiplier),
                        "allowed_to_trade": False,
                        "blocked_reason": block_reason,
                        "trade_executed": False,
                        "quantity": int(quantity),
                        "trade_number_in_day": int(trade_number_in_day),
                        "exit_reason": None,
                        "overlay_exit_reason": None,
                        "net_pnl_usd": np.nan,
                        "net_pnl_r": np.nan,
                    }
                )
                continue

            base_entry_idx = int(signal_idx) + 1 if bool(baseline.entry_on_next_open) else int(signal_idx)
            if base_entry_idx >= len(frame):
                continue

            entry_idx = base_entry_idx
            entry_row = frame.loc[entry_idx]
            use_open_entry = bool(baseline.entry_on_next_open)
            raw_entry = float(entry_row["open"] if use_open_entry else signal_row["close"])
            entry_price = execution_model.apply_slippage(raw_entry, direction, is_entry=True)
            entry_time = entry_row["timestamp"] if use_open_entry else signal_row["timestamp"]

            if direction == 1:
                stop_price = float(or_low) - stop_buffer_points
            else:
                stop_price = float(or_high) + stop_buffer_points

            risk_points = (entry_price - stop_price) * direction
            if risk_points <= 0:
                continue

            target_price = entry_price + direction * float(baseline.target_multiple) * risk_points
            risk_per_contract_usd = abs(
                _mark_to_market_net_pnl_usd(
                    direction=direction,
                    entry_price=entry_price,
                    reference_price=stop_price,
                    quantity=1,
                    execution_model=execution_model,
                    tick_value_usd=tick_value_usd,
                )
            )
            if risk_per_contract_usd <= 0:
                continue

            trade_risk_usd = float(quantity) * float(risk_per_contract_usd)
            exit_scan_start = entry_idx if use_open_entry else entry_idx + 1
            if exit_scan_start >= len(frame):
                continue

            exit_reason = "eod_exit"
            overlay_exit_reason = None
            pending_state_after_trade = None
            raw_exit_price = float(frame.iloc[-1]["close"])
            exit_idx = int(frame.index[-1])
            exit_time = frame.iloc[-1]["timestamp"]

            for bar_idx in range(exit_scan_start, len(frame)):
                bar = frame.loc[bar_idx]
                high = float(bar["high"])
                low = float(bar["low"])
                close = float(bar["close"])
                timestamp = bar["timestamp"]

                if direction == 1:
                    stop_hit = low <= stop_price
                    target_hit = high >= target_price
                else:
                    stop_hit = high >= stop_price
                    target_hit = low <= target_price

                if stop_hit:
                    raw_exit_price = float(stop_price)
                    exit_reason = "stop"
                    exit_idx = int(bar_idx)
                    exit_time = timestamp
                    break

                if target_hit:
                    raw_exit_price = float(target_price)
                    exit_reason = "target"
                    exit_idx = int(bar_idx)
                    exit_time = timestamp
                    break

                current_open_net_pnl_usd = _mark_to_market_net_pnl_usd(
                    direction=direction,
                    entry_price=entry_price,
                    reference_price=close,
                    quantity=quantity,
                    execution_model=execution_model,
                    tick_value_usd=tick_value_usd,
                )
                current_open_net_pnl_r = current_open_net_pnl_usd / trade_risk_usd if trade_risk_usd > 0 else 0.0
                best_reference = _best_intrabar_reference_price(direction, high=high, low=low)
                best_open_net_pnl_usd = _mark_to_market_net_pnl_usd(
                    direction=direction,
                    entry_price=entry_price,
                    reference_price=best_reference,
                    quantity=quantity,
                    execution_model=execution_model,
                    tick_value_usd=tick_value_usd,
                )
                best_open_net_pnl_r = best_open_net_pnl_usd / trade_risk_usd if trade_risk_usd > 0 else 0.0

                current_day_pnl_usd = float(state.realized_pnl_usd + current_open_net_pnl_usd)
                current_day_pnl_r = float(state.realized_pnl_r + current_open_net_pnl_r)
                best_day_pnl_usd = float(state.realized_pnl_usd + best_open_net_pnl_usd)
                best_day_pnl_r = float(state.realized_pnl_r + best_open_net_pnl_r)
                _update_peak_and_giveback(
                    state,
                    current_day_pnl_usd=current_day_pnl_usd,
                    current_day_pnl_r=current_day_pnl_r,
                    best_day_pnl_usd=best_day_pnl_usd,
                    best_day_pnl_r=best_day_pnl_r,
                )

                overlay_reason, next_state = _overlay_exit_signal(
                    state,
                    overlay,
                    current_day_pnl_usd=current_day_pnl_usd,
                    current_day_pnl_r=current_day_pnl_r,
                )
                if overlay_reason is not None:
                    raw_exit_price = float(close)
                    exit_reason = overlay_reason
                    overlay_exit_reason = overlay_reason
                    pending_state_after_trade = next_state
                    exit_idx = int(bar_idx)
                    exit_time = timestamp
                    break

                if pd.to_datetime(timestamp).time() >= force_exit_time:
                    raw_exit_price = float(close)
                    exit_reason = "time_exit"
                    exit_idx = int(bar_idx)
                    exit_time = timestamp
                    break

            exit_price = execution_model.apply_slippage(raw_exit_price, direction, is_entry=False)
            pnl_points = (exit_price - entry_price) * direction
            pnl_ticks = pnl_points / execution_model.tick_size
            pnl_usd = pnl_ticks * tick_value_usd * quantity
            fees = execution_model.round_trip_fees(quantity=quantity)
            net_pnl = pnl_usd - fees
            net_pnl_r = net_pnl / trade_risk_usd if trade_risk_usd > 0 else 0.0

            trade_number_in_day += 1
            trades.append(
                trade_to_record(
                    trade_id,
                    {
                        "session_date": pd.to_datetime(session_date).date(),
                        "direction": "long" if direction == 1 else "short",
                        "quantity": int(quantity),
                        "entry_time": pd.to_datetime(entry_time),
                        "entry_price": float(entry_price),
                        "stop_price": float(stop_price),
                        "target_price": float(target_price),
                        "exit_time": pd.to_datetime(exit_time),
                        "exit_price": float(exit_price),
                        "exit_reason": str(exit_reason),
                        "account_size_usd": None,
                        "risk_per_trade_pct": None,
                        "risk_budget_usd": None,
                        "risk_per_contract_usd": float(risk_per_contract_usd),
                        "actual_risk_usd": float(trade_risk_usd),
                        "trade_risk_usd": float(trade_risk_usd),
                        "notional_usd": float(quantity) * float(entry_price),
                        "leverage_used": np.nan,
                        "pnl_points": float(pnl_points),
                        "pnl_ticks": float(pnl_ticks),
                        "pnl_usd": float(pnl_usd),
                        "fees": float(fees),
                        "net_pnl_usd": float(net_pnl),
                        "overlay_variant": overlay.name,
                        "overlay_family": overlay.family,
                        "overlay_exit_reason": overlay_exit_reason,
                        "overlay_state_pre": state.state_name,
                        "day_trade_number": int(trade_number_in_day),
                        "net_pnl_r": float(net_pnl_r),
                    },
                )
            )
            trade_id += 1

            state.trade_count += 1
            state.realized_pnl_usd += float(net_pnl)
            state.realized_pnl_r += float(net_pnl_r)
            if net_pnl > 0:
                state.win_count += 1
                state.consecutive_wins += 1
                state.consecutive_losses = 0
                if state.first_trade_result is None:
                    state.first_trade_result = "win"
            elif net_pnl < 0:
                state.loss_count += 1
                state.consecutive_losses += 1
                state.consecutive_wins = 0
                if state.first_trade_result is None:
                    state.first_trade_result = "loss"
            else:
                state.consecutive_wins = 0
                state.consecutive_losses = 0
                if state.first_trade_result is None:
                    state.first_trade_result = "flat"

            state.peak_day_pnl_usd = max(float(state.peak_day_pnl_usd), float(state.realized_pnl_usd))
            state.peak_day_pnl_r = max(float(state.peak_day_pnl_r), float(state.realized_pnl_r))

            if pending_state_after_trade is not None:
                state.final_block_reason = str(overlay_exit_reason)
                _set_state(
                    state,
                    pending_state_after_trade,
                    transitions=transitions,
                    variant_name=overlay.name,
                    session_date=session_date,
                    timestamp=exit_time,
                    trigger=str(overlay_exit_reason),
                    current_metric_usd=float(state.realized_pnl_usd),
                    current_metric_r=float(state.realized_pnl_r),
                )

            _apply_post_trade_state_rules(
                state,
                overlay,
                variant_name=overlay.name,
                session_date=session_date,
                timestamp=exit_time,
                transitions=transitions,
            )

            control_rows.append(
                {
                    "variant_name": overlay.name,
                    "session_date": pd.to_datetime(session_date).date(),
                    "signal_time": pd.to_datetime(signal_row["timestamp"]),
                    "signal_index": int(signal_idx),
                    "state_pre": trades[-1].get("overlay_state_pre", "neutral"),
                    "state_post": state.state_name,
                    "policy_multiplier": float(policy_multiplier),
                    "allowed_to_trade": True,
                    "blocked_reason": "",
                    "trade_executed": True,
                    "quantity": int(quantity),
                    "trade_number_in_day": int(trade_number_in_day),
                    "exit_reason": str(exit_reason),
                    "overlay_exit_reason": overlay_exit_reason,
                    "net_pnl_usd": float(net_pnl),
                    "net_pnl_r": float(net_pnl_r),
                }
            )
            next_search_index = int(exit_idx) + 1

        daily_rows.append(
            {
                "variant_name": overlay.name,
                "family": overlay.family,
                "session_date": pd.to_datetime(session_date).date(),
                "daily_pnl_usd": float(state.realized_pnl_usd),
                "daily_pnl_r": float(state.realized_pnl_r),
                "daily_trade_count": int(state.trade_count),
                "daily_win_count": int(state.win_count),
                "daily_loss_count": int(state.loss_count),
                "state_final": state.state_name,
                "first_trade_result": state.first_trade_result,
                "peak_day_pnl_usd": float(state.peak_day_pnl_usd),
                "peak_day_pnl_r": float(state.peak_day_pnl_r),
                "max_giveback_usd": float(state.max_giveback_usd),
                "max_giveback_r": float(state.max_giveback_r),
                "day_cut_by_rule": bool(state.day_cut_by_rule),
                "hard_loss_cap_triggered": bool(state.hard_loss_cap_triggered),
                "hard_profit_lock_triggered": bool(state.hard_profit_lock_triggered),
                "giveback_triggered": bool(state.giveback_triggered),
                "max_trades_cap_triggered": bool(state.max_trades_cap_triggered),
                "first_trade_gate_triggered": bool(state.first_trade_gate_triggered),
                "policy_block_count": int(state.policy_block_count),
                "final_block_reason": state.final_block_reason,
                "transition_count": int(len(transitions) - transition_count_before),
                "threshold_unit": unit,
            }
        )

    if trades:
        trades_df = pd.DataFrame(trades).sort_values(["exit_time", "trade_id"]).reset_index(drop=True)
    else:
        trades_df = empty_trade_log()
        trades_df["overlay_variant"] = pd.Series(dtype="string")
        trades_df["overlay_family"] = pd.Series(dtype="string")
        trades_df["overlay_exit_reason"] = pd.Series(dtype="string")
        trades_df["overlay_state_pre"] = pd.Series(dtype="string")
        trades_df["day_trade_number"] = pd.Series(dtype=int)
        trades_df["net_pnl_r"] = pd.Series(dtype=float)

    controls_df = pd.DataFrame(control_rows) if control_rows else _empty_controls()
    daily_df = pd.DataFrame(daily_rows)
    transitions_df = pd.DataFrame(
        transitions,
        columns=[
            "variant_name",
            "session_date",
            "timestamp",
            "from_state",
            "to_state",
            "trigger",
            "current_day_pnl_usd",
            "current_day_pnl_r",
            "peak_day_pnl_usd",
            "peak_day_pnl_r",
        ],
    )
    return OverlayBacktestResult(
        trades=trades_df,
        controls=controls_df,
        daily_results=daily_df,
        state_transitions=transitions_df,
    )
