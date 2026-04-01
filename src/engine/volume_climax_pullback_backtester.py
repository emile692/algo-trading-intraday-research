"""Dedicated leak-free backtester for volume climax pullback variants."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.engine.execution_model import ExecutionModel
from src.engine.trade_log import empty_trade_log, trade_to_record
from src.engine.vwap_backtester import InstrumentDetails
from src.strategy.volume_climax_pullback import VolumeClimaxPullbackVariant


@dataclass(frozen=True)
class VolumeClimaxBacktestResult:
    trades: pd.DataFrame


def run_volume_climax_pullback_backtest(
    signal_df: pd.DataFrame,
    variant: VolumeClimaxPullbackVariant,
    execution_model: ExecutionModel,
    instrument: InstrumentDetails,
) -> VolumeClimaxBacktestResult:
    trades: list[dict] = []
    open_trade: dict | None = None
    trade_id = 1

    for session_date, sdf in signal_df.groupby("session_date", sort=True):
        sdf = sdf.sort_values("timestamp").reset_index(drop=True)
        last_idx = len(sdf) - 1

        for i, row in sdf.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            o, h, l, c = (float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]))

            if open_trade is not None:
                open_trade["bars_held"] += 1
                open_trade["max_high"] = max(open_trade["max_high"], h)
                open_trade["min_low"] = min(open_trade["min_low"], l)
                direction = int(open_trade["direction"])
                exit_price = None
                exit_reason = None

                stop_hit = (l <= open_trade["stop_price"]) if direction == 1 else (h >= open_trade["stop_price"])
                target_hit = (h >= open_trade["target_price"]) if direction == 1 else (l <= open_trade["target_price"])
                if stop_hit and target_hit:
                    exit_price, exit_reason = open_trade["stop_price"], "stop_ambiguous_first"
                elif stop_hit:
                    exit_price, exit_reason = open_trade["stop_price"], "stop"
                elif target_hit:
                    exit_price, exit_reason = open_trade["target_price"], "target"
                elif open_trade["bars_held"] >= open_trade["time_stop_bars"]:
                    exit_price, exit_reason = c, "time_stop"
                elif i == last_idx:
                    exit_price, exit_reason = c, "eod_flat"

                if exit_price is not None:
                    fill_exit = execution_model.apply_slippage(exit_price, direction, is_entry=False)
                    pnl_points = (fill_exit - open_trade["entry_price"]) * direction
                    gross = pnl_points * instrument.point_value_usd
                    fees = execution_model.round_trip_fees(quantity=1)
                    net = gross - fees
                    trades.append(
                        trade_to_record(
                            trade_id,
                            {
                                "session_date": session_date,
                                "direction": "long" if direction == 1 else "short",
                                "quantity": 1,
                                "entry_time": open_trade["entry_time"],
                                "entry_price": open_trade["entry_price"],
                                "stop_price": open_trade["stop_price"],
                                "target_price": open_trade["target_price"],
                                "exit_time": ts,
                                "exit_price": fill_exit,
                                "exit_reason": exit_reason,
                                "account_size_usd": np.nan,
                                "risk_per_trade_pct": np.nan,
                                "risk_budget_usd": np.nan,
                                "risk_per_contract_usd": open_trade["risk_usd"],
                                "actual_risk_usd": open_trade["risk_usd"],
                                "trade_risk_usd": open_trade["risk_usd"],
                                "notional_usd": open_trade["entry_price"] * instrument.point_value_usd,
                                "leverage_used": np.nan,
                                "pnl_points": pnl_points,
                                "pnl_ticks": pnl_points / instrument.tick_size if instrument.tick_size > 0 else np.nan,
                                "pnl_usd": gross,
                                "fees": fees,
                                "net_pnl_usd": net,
                            },
                        )
                    )
                    trades[-1]["variant_name"] = variant.name
                    trades[-1]["bars_held"] = open_trade["bars_held"]
                    trade_id += 1
                    open_trade = None

            if open_trade is None:
                short_sig = bool(row.get("entry_short", False))
                long_sig = bool(row.get("entry_long", False))
                if short_sig == long_sig:
                    continue
                direction = -1 if short_sig else 1
                stop_col = "entry_stop_reference_short" if direction == -1 else "entry_stop_reference_long"
                stop_ref = pd.to_numeric(pd.Series([row.get(stop_col, np.nan)]), errors="coerce").iloc[0]
                if pd.isna(stop_ref):
                    continue
                if variant.stop_buffer_mode == "1_tick":
                    stop_ref = stop_ref + instrument.tick_size if direction == -1 else stop_ref - instrument.tick_size
                entry = execution_model.apply_slippage(o, direction, is_entry=True)
                risk_pts = (entry - float(stop_ref)) * direction
                if risk_pts <= 0:
                    continue
                target = entry + direction * float(row.get("entry_target_r", variant.rr_target)) * risk_pts
                open_trade = {
                    "entry_time": ts,
                    "entry_price": entry,
                    "direction": direction,
                    "stop_price": float(stop_ref),
                    "target_price": float(target),
                    "time_stop_bars": int(row.get("entry_time_stop_bars", variant.time_stop_bars)),
                    "bars_held": 0,
                    "max_high": h,
                    "min_low": l,
                    "risk_usd": risk_pts * instrument.point_value_usd + execution_model.round_trip_fees(quantity=1),
                }

    trades_df = pd.DataFrame(trades) if trades else empty_trade_log()
    if "variant_name" not in trades_df.columns:
        trades_df["variant_name"] = pd.Series(dtype=object)
    if "bars_held" not in trades_df.columns:
        trades_df["bars_held"] = pd.Series(dtype=float)
    return VolumeClimaxBacktestResult(trades=trades_df)
