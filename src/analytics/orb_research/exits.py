"""Exit variant backtesting for ORB research overlays."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from src.config.settings import DEFAULT_TICK_VALUE_USD
from src.engine.backtester import run_backtest
from src.engine.execution_model import ExecutionModel
from src.engine.trade_log import empty_trade_log, trade_to_record

from .types import BaselineEntryConfig, ExitConfig


def _compute_risk_per_contract_usd(
    direction: int,
    entry_price: float,
    stop_price: float,
    execution_model: ExecutionModel,
    tick_value_usd: float,
) -> float:
    stop_exit = execution_model.apply_slippage(stop_price, direction, is_entry=False)
    pnl_points = (stop_exit - entry_price) * direction
    pnl_ticks = pnl_points / execution_model.tick_size
    gross_loss = abs(pnl_ticks * tick_value_usd)
    return gross_loss + execution_model.round_trip_fees(quantity=1)


def _apply_leverage_cap(
    quantity: int,
    entry_price: float,
    account_size_usd: float,
    point_value_usd: float,
    max_leverage: float | None,
) -> int:
    if max_leverage is None:
        return quantity
    max_notional = account_size_usd * max_leverage
    per_contract_notional = entry_price * point_value_usd
    if per_contract_notional <= 0:
        return 0
    cap_qty = int(max_notional / per_contract_notional)
    return min(quantity, cap_qty)


def _resolve_force_exit_time(exit_cfg: ExitConfig, baseline: BaselineEntryConfig) -> dt.time:
    force_time = exit_cfg.force_exit_time or baseline.time_exit
    return dt.time.fromisoformat(force_time)


def _stagnation_hit(
    bars_since_entry: int,
    stagnation_bars: int | None,
    max_favorable_r: float,
    min_r_required: float,
) -> bool:
    if stagnation_bars is None:
        return False
    return bars_since_entry >= int(stagnation_bars) and max_favorable_r < float(min_r_required)


def run_exit_variant_backtest(
    signal_df: pd.DataFrame,
    execution_model: ExecutionModel,
    baseline: BaselineEntryConfig,
    exit_cfg: ExitConfig,
    tick_value_usd: float = DEFAULT_TICK_VALUE_USD,
    point_value_usd: float | None = None,
    max_leverage: float | None = None,
) -> pd.DataFrame:
    """Run baseline or overlay exit logic while keeping entry logic unchanged."""
    mode = exit_cfg.mode
    if mode == "baseline":
        return run_backtest(
            signal_df,
            execution_model=execution_model,
            tick_value_usd=tick_value_usd,
            point_value_usd=point_value_usd,
            time_exit=baseline.time_exit,
            stop_buffer_ticks=baseline.stop_buffer_ticks,
            target_multiple=baseline.target_multiple,
            account_size_usd=baseline.account_size_usd,
            risk_per_trade_pct=baseline.risk_per_trade_pct,
            entry_on_next_open=baseline.entry_on_next_open,
            max_leverage=max_leverage,
        )

    if signal_df.empty:
        return empty_trade_log()

    point_value = point_value_usd if point_value_usd is not None else tick_value_usd / execution_model.tick_size
    stop_buffer_points = baseline.stop_buffer_ticks * execution_model.tick_size
    force_exit_time = _resolve_force_exit_time(exit_cfg, baseline)

    trades: list[dict[str, object]] = []
    trade_id = 1

    required_vwap_modes = {
        "fail_fast_vwap",
        "trailing_vwap",
        "trailing_struct_plus_vwap",
        "partial_1R_then_vwap",
        "time_stop_plus_vwap",
    }

    for session_date, session_df in signal_df.groupby("session_date", sort=True):
        frame = session_df.sort_values("timestamp").reset_index(drop=True)
        signal_indices = frame.index[frame["signal"].fillna(0).ne(0)].tolist()
        if not signal_indices:
            continue

        next_search_index = 0
        for signal_idx in signal_indices:
            if signal_idx < next_search_index:
                continue

            signal_row = frame.loc[signal_idx]
            direction = int(signal_row["signal"])
            if direction != 1:
                continue

            or_high = pd.to_numeric(signal_row.get("or_high"), errors="coerce")
            or_low = pd.to_numeric(signal_row.get("or_low"), errors="coerce")
            if pd.isna(or_high) or pd.isna(or_low):
                continue

            entry_idx = signal_idx + 1 if baseline.entry_on_next_open else signal_idx
            if entry_idx >= len(frame):
                continue

            entry_row = frame.loc[entry_idx]
            raw_entry = float(entry_row["open"] if baseline.entry_on_next_open else signal_row["close"])
            entry_price = execution_model.apply_slippage(raw_entry, direction, is_entry=True)
            entry_time = entry_row["timestamp"] if baseline.entry_on_next_open else signal_row["timestamp"]

            initial_stop = float(or_low) - stop_buffer_points
            risk_points = (entry_price - initial_stop) * direction
            if risk_points <= 0:
                continue

            full_target = entry_price + direction * baseline.target_multiple * risk_points
            partial_target = entry_price + direction * risk_points

            risk_per_contract = _compute_risk_per_contract_usd(
                direction=direction,
                entry_price=entry_price,
                stop_price=initial_stop,
                execution_model=execution_model,
                tick_value_usd=tick_value_usd,
            )
            if risk_per_contract <= 0:
                continue

            risk_budget = baseline.account_size_usd * (baseline.risk_per_trade_pct / 100.0)
            quantity = int(risk_budget / risk_per_contract)
            if quantity < 1:
                continue

            quantity = _apply_leverage_cap(
                quantity=quantity,
                entry_price=entry_price,
                account_size_usd=baseline.account_size_usd,
                point_value_usd=point_value,
                max_leverage=max_leverage,
            )
            if quantity < 1:
                continue

            risk_usd = quantity * risk_per_contract
            notional_usd = quantity * entry_price * point_value
            leverage_used = notional_usd / baseline.account_size_usd if baseline.account_size_usd > 0 else np.nan

            exit_scan_start = entry_idx if baseline.entry_on_next_open else entry_idx + 1
            if exit_scan_start >= len(frame):
                continue

            current_stop = float(initial_stop)
            structural_stop = float(initial_stop)

            remaining_qty = int(quantity)
            partial_qty = int(math.floor(quantity * float(exit_cfg.partial_fraction)))
            partial_qty = max(0, min(partial_qty, remaining_qty - 1))
            runner_qty = remaining_qty

            legs: list[dict[str, object]] = []
            max_favorable_r = 0.0
            bars_since_entry = 0
            exit_idx = None
            final_reason = None

            for bar_idx in range(exit_scan_start, len(frame)):
                bar = frame.loc[bar_idx]
                bars_since_entry += 1

                high = float(bar["high"])
                low = float(bar["low"])
                close = float(bar["close"])
                ts = bar["timestamp"]
                ts_time = ts.time()
                vwap = pd.to_numeric(bar.get("continuous_session_vwap", np.nan), errors="coerce")
                if pd.isna(vwap):
                    vwap = pd.to_numeric(bar.get("session_vwap", np.nan), errors="coerce")

                favorable_r = max(0.0, (high - entry_price) / risk_points)
                max_favorable_r = max(max_favorable_r, favorable_r)

                stop_hit = low <= current_stop
                target_hit = high >= full_target

                if mode in {"trailing_vwap", "trailing_struct_plus_vwap", "partial_1R_then_vwap", "time_stop_plus_vwap"}:
                    # trailing-based modes keep target only for staged leg opening, not for full one-shot profit taking.
                    target_hit = False

                if stop_hit:
                    legs.append(
                        {
                            "qty": runner_qty,
                            "raw_exit_price": current_stop,
                            "exit_time": ts,
                            "reason": "stop",
                        }
                    )
                    final_reason = "stop"
                    exit_idx = bar_idx
                    runner_qty = 0
                    break

                if mode == "partial_1R_then_vwap" and partial_qty > 0 and runner_qty == quantity and high >= partial_target:
                    legs.append(
                        {
                            "qty": partial_qty,
                            "raw_exit_price": partial_target,
                            "exit_time": ts,
                            "reason": "partial_1R",
                        }
                    )
                    runner_qty -= partial_qty

                if target_hit and runner_qty > 0:
                    legs.append(
                        {
                            "qty": runner_qty,
                            "raw_exit_price": full_target,
                            "exit_time": ts,
                            "reason": "target",
                        }
                    )
                    final_reason = "target"
                    exit_idx = bar_idx
                    runner_qty = 0
                    break

                if mode == "fail_fast_vwap" and pd.notna(vwap) and close < float(vwap) and runner_qty > 0:
                    legs.append(
                        {
                            "qty": runner_qty,
                            "raw_exit_price": close,
                            "exit_time": ts,
                            "reason": "fail_fast_vwap",
                        }
                    )
                    final_reason = "fail_fast_vwap"
                    exit_idx = bar_idx
                    runner_qty = 0
                    break

                if mode == "time_stop_plus_vwap" and _stagnation_hit(
                    bars_since_entry=bars_since_entry,
                    stagnation_bars=exit_cfg.stagnation_bars,
                    max_favorable_r=max_favorable_r,
                    min_r_required=exit_cfg.stagnation_min_r_multiple,
                ) and runner_qty > 0:
                    legs.append(
                        {
                            "qty": runner_qty,
                            "raw_exit_price": close,
                            "exit_time": ts,
                            "reason": "stagnation_time_stop",
                        }
                    )
                    final_reason = "stagnation_time_stop"
                    exit_idx = bar_idx
                    runner_qty = 0
                    break

                if ts_time >= force_exit_time and runner_qty > 0:
                    legs.append(
                        {
                            "qty": runner_qty,
                            "raw_exit_price": close,
                            "exit_time": ts,
                            "reason": "time_exit",
                        }
                    )
                    final_reason = "time_exit"
                    exit_idx = bar_idx
                    runner_qty = 0
                    break

                if mode in required_vwap_modes and pd.notna(vwap):
                    vwap_val = float(vwap)
                    if mode in {"trailing_struct_plus_vwap", "partial_1R_then_vwap", "time_stop_plus_vwap"}:
                        structural_stop = max(structural_stop, low - stop_buffer_points)
                        current_stop = max(current_stop, structural_stop, vwap_val)
                    else:
                        current_stop = max(current_stop, vwap_val)

            if runner_qty > 0:
                last_row = frame.iloc[-1]
                legs.append(
                    {
                        "qty": runner_qty,
                        "raw_exit_price": float(last_row["close"]),
                        "exit_time": last_row["timestamp"],
                        "reason": "eod_exit",
                    }
                )
                final_reason = "eod_exit"
                exit_idx = len(frame) - 1

            gross_pnl_usd = 0.0
            final_exit_price = np.nan
            final_exit_time = None
            for leg in legs:
                qty = int(leg["qty"])
                if qty <= 0:
                    continue
                exit_price = execution_model.apply_slippage(float(leg["raw_exit_price"]), direction, is_entry=False)
                pnl_points = (exit_price - entry_price) * direction
                pnl_ticks = pnl_points / execution_model.tick_size
                gross_pnl_usd += pnl_ticks * tick_value_usd * qty
                final_exit_price = exit_price
                final_exit_time = leg["exit_time"]

            fees = execution_model.round_trip_fees(quantity=quantity)
            net_pnl = gross_pnl_usd - fees
            pnl_points_total = net_pnl / (tick_value_usd * quantity / execution_model.tick_size) if quantity > 0 else 0.0
            pnl_ticks_total = pnl_points_total / execution_model.tick_size

            trades.append(
                trade_to_record(
                    trade_id,
                    {
                        "session_date": session_date,
                        "direction": "long",
                        "quantity": quantity,
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "stop_price": initial_stop,
                        "target_price": full_target,
                        "exit_time": final_exit_time,
                        "exit_price": final_exit_price,
                        "exit_reason": final_reason,
                        "account_size_usd": baseline.account_size_usd,
                        "risk_per_trade_pct": baseline.risk_per_trade_pct,
                        "risk_budget_usd": risk_budget,
                        "risk_per_contract_usd": risk_per_contract,
                        "actual_risk_usd": risk_usd,
                        "trade_risk_usd": risk_usd,
                        "notional_usd": notional_usd,
                        "leverage_used": leverage_used,
                        "pnl_points": pnl_points_total,
                        "pnl_ticks": pnl_ticks_total,
                        "pnl_usd": gross_pnl_usd,
                        "fees": fees,
                        "net_pnl_usd": net_pnl,
                    },
                )
            )
            trade_id += 1
            next_search_index = int(exit_idx) + 1 if exit_idx is not None else next_search_index

    if not trades:
        return empty_trade_log()
    return pd.DataFrame(trades)
