"""Execution-only exit overlays over frozen entry instructions."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, time
from typing import Any

import numpy as np
import pandas as pd

from research.baseline_backtests.baseline_engine import (
    INITIAL_CAPITAL_USD,
    BacktestResult,
    EntryInstruction,
    PreparedData,
    SIGNAL_BUILDERS,
    _apply_slippage,
    _close_position,
    _effective_cutoff,
    _empty_trades,
    _intrabar_exit,
    _open_position,
    _resting_trigger,
    _round_tick,
    _atr,
)


def _clean_entry(item: EntryInstruction) -> EntryInstruction:
    """Preserve timing and levels while removing every frozen exit field."""
    return replace(
        item,
        stop_price=None,
        stop_offset_points=None,
        target_price=None,
        long_stop=None,
        short_stop=None,
        long_target=None,
        short_target=None,
        target_offset_points=None,
        require_between=False,
    )


def _entry_map(entries: list[EntryInstruction]) -> dict[pd.Timestamp, list[EntryInstruction]]:
    output: dict[pd.Timestamp, list[EntryInstruction]] = {}
    for item in entries:
        output.setdefault(item.activation_time, []).append(item)
    return output


def _atr_context(data: PreparedData, timeframe: int) -> tuple[pd.Series, dict[pd.Timestamp, tuple[pd.Timestamp, float, float, float]]]:
    bars = data.bars[timeframe].copy()
    bars["mm_atr"] = _atr(bars, 14)
    valid = bars.dropna(subset=["mm_atr"]).copy()
    lookup = pd.Series(valid["mm_atr"].to_numpy(dtype=float), index=pd.DatetimeIndex(valid["end_time"]))
    lookup = lookup[~lookup.index.duplicated(keep="last")].sort_index()
    minute_atr = lookup.reindex(data.minute.index, method="ffill")
    updates = {
        pd.Timestamp(row.end_time): (pd.Timestamp(row.start_time), float(row.high), float(row.low), float(row.mm_atr))
        for row in valid.itertuples(index=False)
    }
    return minute_atr, updates


def build_overlay_context(
    spec: dict[str, Any], data: PreparedData, signal_timeframe: int,
) -> tuple[list[EntryInstruction], pd.Series, dict[pd.Timestamp, tuple[pd.Timestamp, float, float, float]]]:
    plan = SIGNAL_BUILDERS[spec["mechanism_id"]](spec, data)
    minute_atr, updates = _atr_context(data, signal_timeframe)
    return [_clean_entry(item) for item in plan.entries], minute_atr, updates


def _configure_position(position: dict[str, Any], policy: dict[str, Any], atr_value: float, tick: float) -> None:
    direction = int(position["direction_int"])
    entry = float(position["entry_price"])
    family = policy["family"]
    stop_distance: float | None = None
    target_distance: float | None = None
    if family == "atr_stop":
        stop_distance = float(policy["stop_atr"]) * atr_value
    elif family == "rr_target":
        stop_distance = float(policy["stop_atr"]) * atr_value
        if policy["target_R"] is not None:
            target_distance = float(policy["target_R"]) * stop_distance
    elif family == "atr_trailing":
        stop_distance = float(policy["trail_atr"]) * atr_value
        position["trail_atr"] = float(policy["trail_atr"])
    elif family == "breakeven":
        stop_distance = float(policy["initial_stop_atr"]) * atr_value
        position["breakeven_trigger_price"] = entry + direction * float(policy["move_to_BE_after_R"]) * stop_distance
        position["breakeven_pending"] = False
        position["breakeven_applied"] = False
    elif family == "time_stop":
        position["time_stop_at"] = pd.Timestamp(position["entry_time"]) + pd.Timedelta(minutes=int(policy["max_holding_minutes"]))
    else:
        raise ValueError(f"Unsupported exit family: {family}")
    if stop_distance is not None:
        position["stop_price"] = _round_tick(entry - direction * stop_distance, tick, direction, False)
    else:
        position["stop_price"] = None
    if target_distance is not None:
        position["target_price"] = _round_tick(entry + direction * target_distance, tick, direction, False)
    else:
        position["target_price"] = None
    position["initial_stop_price"] = position["stop_price"]


def _apply_dynamic_updates(
    position: dict[str, Any], policy: dict[str, Any], timestamp: pd.Timestamp,
    trailing_update: tuple[pd.Timestamp, float, float, float] | None, tick: float,
) -> None:
    direction = int(position["direction_int"])
    if policy["family"] == "breakeven" and position.get("breakeven_pending"):
        candidate = _round_tick(float(position["entry_price"]), tick, direction, False)
        current = position.get("stop_price")
        position["stop_price"] = candidate if current is None else (max(float(current), candidate) if direction == 1 else min(float(current), candidate))
        position["breakeven_pending"] = False
        position["breakeven_applied"] = True
    if policy["family"] != "atr_trailing" or trailing_update is None:
        return
    bar_start, bar_high, bar_low, atr_value = trailing_update
    if bar_start < pd.Timestamp(position["entry_time"]):
        return
    position["highest_completed_high"] = max(float(position["highest_completed_high"]), bar_high)
    position["lowest_completed_low"] = min(float(position["lowest_completed_low"]), bar_low)
    multiple = float(position["trail_atr"])
    candidate = position["highest_completed_high"] - multiple * atr_value if direction == 1 else position["lowest_completed_low"] + multiple * atr_value
    candidate = _round_tick(float(candidate), tick, direction, False)
    current = position.get("stop_price")
    position["stop_price"] = candidate if current is None else (max(float(current), candidate) if direction == 1 else min(float(current), candidate))


def _schedule_breakeven(position: dict[str, Any], bar_high: float, bar_low: float) -> None:
    if "breakeven_trigger_price" not in position or position.get("breakeven_applied"):
        return
    reached = bar_high >= position["breakeven_trigger_price"] if position["direction_int"] == 1 else bar_low <= position["breakeven_trigger_price"]
    if reached:
        position["breakeven_pending"] = True


def run_exit_overlay_backtest(
    spec: dict[str, Any], data: PreparedData, policy: dict[str, Any], signal_timeframe: int,
    context: tuple[list[EntryInstruction], pd.Series, dict[pd.Timestamp, tuple[pd.Timestamp, float, float, float]]] | None = None,
) -> BacktestResult:
    """Run frozen entry instructions with one alternative exit family."""
    mechanism_id = spec["mechanism_id"]
    entries, minute_atr, trailing_updates = context or build_overlay_context(spec, data, signal_timeframe)
    entry_map = _entry_map(entries)
    execution = spec["canonical_parameters"]["execution"]
    tick = float(execution["tick_size"])
    point_value = float(execution["point_value_usd"])
    slippage_ticks = int(execution["slippage_ticks_per_side"])
    commission = float(execution["commission_per_side_usd"])
    strategy_cutoff = time.fromisoformat(execution["last_entry_time"])
    checks: dict[str, Any] = {
        "spec_hash_match": True,
        "lookahead_violations": 0,
        "future_high_low_violations": 0,
        "same_bar_policy_violations": 0,
        "double_entry_violations": 0,
        "overnight_position_violations": 0,
        "entry_after_last_entry_violations": 0,
        "post_hard_flatten_violations": 0,
        "missing_execution_bar_skips": 0,
        "invalid_entry_geometry_skips": 0,
        "same_signal_indicator_stop_entry_skips": 0,
        "dual_oco_cancellations": 0,
        "multi_order_same_direction_priority_resolutions": 0,
        "ambiguous_multi_order_cancellations": 0,
        "hard_flatten_fallbacks": 0,
        "missing_atr_entry_skips": 0,
    }
    trades: list[dict[str, Any]] = []
    trade_id = 1
    by_date = {day: frame for day, frame in data.minute.groupby("session_date", sort=True)}
    active_dates = {item.session_date for item in entries}
    for _, session in data.sessions.iterrows():
        day = session["session_date"]
        if day not in active_dates:
            continue
        frame = by_date.get(day)
        if frame is None or frame.empty:
            continue
        frame = frame.sort_index()
        cutoff = _effective_cutoff(session, strategy_cutoff)
        hard_flatten = pd.Timestamp(session["hard_flatten"])
        position: dict[str, Any] | None = None
        pending: list[EntryInstruction] = []
        exact = set(frame.index)
        expected = [item for item in entries if item.session_date == day and item.order_type == "market_open"]
        checks["missing_execution_bar_skips"] += sum(item.activation_time not in exact for item in expected)
        for minute in frame.itertuples(index=True, name="Minute"):
            timestamp = pd.Timestamp(minute.Index)
            if timestamp > hard_flatten:
                break
            bar_open, bar_high, bar_low = float(minute.open), float(minute.high), float(minute.low)
            exited_at_open = False
            if position is not None and timestamp == hard_flatten:
                trades.append(_close_position(position, bar_open, timestamp, "hard_flatten", tick, slippage_ticks, point_value, commission, trade_id))
                trade_id += 1; position = None; pending.clear(); exited_at_open = True
            if position is not None and policy["family"] == "time_stop" and timestamp >= position["time_stop_at"]:
                trades.append(_close_position(position, bar_open, timestamp, "time_stop", tick, slippage_ticks, point_value, commission, trade_id))
                trade_id += 1; position = None; pending.clear(); exited_at_open = True
            if position is not None:
                _apply_dynamic_updates(position, policy, timestamp, trailing_updates.get(timestamp), tick)
                position["max_high"] = max(float(position["max_high"]), bar_high)
                position["min_low"] = min(float(position["min_low"]), bar_low)
                hit = _intrabar_exit(position, bar_open, bar_high, bar_low)
                if hit is not None:
                    trades.append(_close_position(position, hit[0], timestamp, hit[1], tick, slippage_ticks, point_value, commission, trade_id))
                    trade_id += 1; position = None; pending.clear()
                else:
                    _schedule_breakeven(position, bar_high, bar_low)
            activations = entry_map.get(timestamp, [])
            if position is None:
                pending.extend(item for item in activations if item.order_type == "stop_oco")
            pending = [item for item in pending if timestamp <= cutoff]
            if position is None and not exited_at_open and timestamp <= cutoff:
                market_entries = [item for item in activations if item.order_type == "market_open"]
                if len(market_entries) > 1:
                    directions = {item.direction for item in market_entries}
                    if len(directions) > 1:
                        checks["ambiguous_multi_order_cancellations"] += 1; market_entries = []
                    else:
                        checks["multi_order_same_direction_priority_resolutions"] += 1; market_entries = market_entries[:1]
                if market_entries:
                    instruction = market_entries[0]
                    position = _open_position(instruction, instruction.direction, bar_open, timestamp, tick, slippage_ticks, point_value, None, None, checks)
                    if position is not None:
                        atr_value = minute_atr.get(timestamp, np.nan)
                        if pd.isna(atr_value):
                            position = None; checks["missing_atr_entry_skips"] += 1
                        else:
                            _configure_position(position, policy, float(atr_value), tick); pending.clear()
            opened = position is not None and position["entry_time"] == timestamp
            if position is not None and opened:
                position["max_high"] = max(float(position["max_high"]), bar_high)
                position["min_low"] = min(float(position["min_low"]), bar_low)
                hit = _intrabar_exit(position, bar_open, bar_high, bar_low)
                if hit is not None:
                    trades.append(_close_position(position, hit[0], timestamp, hit[1], tick, slippage_ticks, point_value, commission, trade_id))
                    trade_id += 1; position = None; pending.clear()
                else:
                    _schedule_breakeven(position, bar_high, bar_low)
            if position is None and not opened and not exited_at_open and timestamp <= cutoff and pending:
                triggers: list[tuple[EntryInstruction, tuple[int, float, float | None, float | None]]] = []
                remaining: list[EntryInstruction] = []
                for order in pending:
                    outcome = _resting_trigger(order, bar_open, bar_high, bar_low, tick)
                    if outcome == "both":
                        checks["dual_oco_cancellations"] += 1
                    elif outcome is None:
                        remaining.append(order)
                    else:
                        triggers.append((order, outcome))
                pending = remaining
                if len(triggers) > 1:
                    directions = {item[1][0] for item in triggers}
                    if len(directions) == 1:
                        checks["multi_order_same_direction_priority_resolutions"] += 1; triggers = triggers[:1]
                    else:
                        checks["ambiguous_multi_order_cancellations"] += 1; triggers = []; pending.clear()
                if len(triggers) == 1:
                    instruction, (direction, raw_entry, _, _) = triggers[0]
                    position = _open_position(instruction, direction, raw_entry, timestamp, tick, slippage_ticks, point_value, None, None, checks)
                    pending.clear()
                    if position is not None:
                        atr_value = minute_atr.get(timestamp, np.nan)
                        if pd.isna(atr_value):
                            position = None; checks["missing_atr_entry_skips"] += 1
                        else:
                            _configure_position(position, policy, float(atr_value), tick)
                            position["max_high"] = max(float(position["max_high"]), bar_high)
                            position["min_low"] = min(float(position["min_low"]), bar_low)
                            hit = _intrabar_exit(position, bar_open, bar_high, bar_low)
                            if hit is not None:
                                trades.append(_close_position(position, hit[0], timestamp, hit[1], tick, slippage_ticks, point_value, commission, trade_id))
                                trade_id += 1; position = None
                            else:
                                _schedule_breakeven(position, bar_high, bar_low)
        if position is not None:
            pre = frame[frame.index < hard_flatten]
            if pre.empty:
                checks["post_hard_flatten_violations"] += 1
            else:
                timestamp = pd.Timestamp(pre.index[-1])
                trades.append(_close_position(position, float(pre.iloc[-1]["close"]), timestamp, "hard_flatten_missing_bar_fallback", tick, slippage_ticks, point_value, commission, trade_id))
                trade_id += 1; checks["hard_flatten_fallbacks"] += 1; position = None

    trade_frame = pd.DataFrame(trades) if trades else _empty_trades()
    if not trade_frame.empty:
        trade_frame.insert(0, "mechanism_id", mechanism_id)
        trade_frame.insert(1, "strategy_id", spec["strategy_id"])
        for column in ("signal_time", "entry_time", "exit_time"):
            trade_frame[column] = pd.to_datetime(trade_frame[column])
        ordered = trade_frame.sort_values("entry_time")
        checks["double_entry_violations"] = int(sum(pd.Timestamp(ordered.iloc[i]["entry_time"]) < pd.Timestamp(ordered.iloc[i - 1]["exit_time"]) for i in range(1, len(ordered))))
        checks["overnight_position_violations"] = int(sum(pd.Timestamp(row.entry_time).date() != pd.Timestamp(row.exit_time).date() for row in trade_frame.itertuples(index=False)))
        checks["entry_after_last_entry_violations"] = int(sum(pd.Timestamp(row.entry_time) > _effective_cutoff(data.sessions.loc[data.sessions["session_date"] == row.session_date].iloc[0], strategy_cutoff) for row in trade_frame.itertuples(index=False)))
        checks["post_hard_flatten_violations"] += int(sum(pd.Timestamp(row.exit_time) > pd.Timestamp(data.sessions.loc[data.sessions["session_date"] == row.session_date, "hard_flatten"].iloc[0]) for row in trade_frame.itertuples(index=False)))
    else:
        trade_frame.insert(0, "mechanism_id", pd.Series(dtype=str)); trade_frame.insert(1, "strategy_id", pd.Series(dtype=str))
    session_dates = sorted(set(data.benchmark_daily["session_date"]) | set(data.sessions["session_date"]))
    pnl_by_day = trade_frame.groupby("session_date")["net_pnl_usd"].sum().to_dict() if len(trade_frame) else {}
    daily = pd.DataFrame({"session_date": session_dates})
    daily["mechanism_id"] = mechanism_id
    daily["daily_pnl_usd"] = daily["session_date"].map(pnl_by_day).fillna(0.0)
    daily["equity_usd"] = INITIAL_CAPITAL_USD + daily["daily_pnl_usd"].cumsum()
    daily["daily_return"] = daily["daily_pnl_usd"] / daily["equity_usd"].shift(1).fillna(INITIAL_CAPITAL_USD)
    daily["trade_count"] = daily["session_date"].map(trade_frame.groupby("session_date").size().to_dict() if len(trade_frame) else {}).fillna(0).astype(int)
    equity = daily[["session_date", "mechanism_id", "daily_pnl_usd", "daily_return", "equity_usd"]].copy()
    positions = trade_frame[["mechanism_id", "trade_id", "session_date", "position", "quantity", "entry_time", "exit_time"]].copy() if len(trade_frame) else pd.DataFrame(columns=["mechanism_id", "trade_id", "session_date", "position", "quantity", "entry_time", "exit_time"])
    return BacktestResult(trades=trade_frame, daily=daily, equity=equity, positions=positions, checks=checks)


def slice_result(run: BacktestResult, start: date, end: date) -> BacktestResult:
    daily = run.daily[run.daily["session_date"].between(start, end)].copy().reset_index(drop=True)
    daily["equity_usd"] = INITIAL_CAPITAL_USD + daily["daily_pnl_usd"].cumsum()
    daily["daily_return"] = daily["daily_pnl_usd"] / daily["equity_usd"].shift(1).fillna(INITIAL_CAPITAL_USD)
    trades = run.trades[run.trades["session_date"].between(start, end)].copy().reset_index(drop=True)
    positions = run.positions[run.positions["session_date"].between(start, end)].copy().reset_index(drop=True)
    equity = daily[["session_date", "mechanism_id", "daily_pnl_usd", "daily_return", "equity_usd"]].copy()
    return BacktestResult(trades=trades, daily=daily, equity=equity, positions=positions, checks=dict(run.checks))
