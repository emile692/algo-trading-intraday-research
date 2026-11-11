"""Execute the frozen 42-cell MNQ volatility-breakout vote ensemble."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import defaultdict
from datetime import date
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.baseline_backtests.baseline_engine import (
    BacktestResult,
    EntryInstruction,
    INITIAL_CAPITAL_USD,
    _close_position,
    _empty_trades,
    _open_position,
    build_buy_and_hold,
    compute_metrics,
    load_frozen_specs,
    prepare_data,
    slice_prepared_data,
)
from research.money_management_lab.mm_engine import build_overlay_context, run_exit_overlay_backtest, slice_result


OUTPUT_DIR = ROOT / "research" / "signal_sheet_02_cluster"
MANIFEST_PATH = ROOT / "strategy_registry" / "psf_mnq" / "frozen_ensembles" / "volatility_breakout_42.json"
FROZEN_DIR = ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
DATA_PATH = ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"
IS_START, IS_END = date(2019, 5, 6), date(2023, 12, 31)
OOS_START, OOS_END = date(2024, 1, 1), date(2026, 3, 19)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_id(atr_period: int, multiple: float, duration: int) -> str:
    return f"atr{atr_period:02d}_m{int(round(multiple * 100)):02d}_d{duration:03d}"


def build_cell_positions(
    spec: dict[str, Any], prepared: Any, manifest: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    region = manifest["parameter_region"]
    rows: list[pd.DataFrame] = []
    checks: dict[str, Any] = {}
    for atr_period, multiple in product(region["daily_atr_period"], region["entry_atr_multiple"]):
        candidate = copy.deepcopy(spec)
        structural = candidate["canonical_parameters"]["structural"]
        structural["daily_atr_period"] = int(atr_period)
        structural["entry_atr_multiple"] = float(multiple)
        context = build_overlay_context(candidate, prepared, 5)
        for duration in region["max_holding_minutes"]:
            identifier = cell_id(int(atr_period), float(multiple), int(duration))
            run = run_exit_overlay_backtest(
                candidate,
                prepared,
                {"family": "time_stop", "max_holding_minutes": int(duration)},
                5,
                context=context,
            )
            watched = (
                "lookahead_violations",
                "future_high_low_violations",
                "same_bar_policy_violations",
                "double_entry_violations",
                "overnight_position_violations",
                "entry_after_last_entry_violations",
                "post_hard_flatten_violations",
                "ambiguous_multi_order_cancellations",
                "missing_atr_entry_skips",
            )
            violations = {key: run.checks.get(key, 0) for key in watched if run.checks.get(key, 0)}
            if violations:
                raise RuntimeError(f"Invalid cell {identifier}: {violations}")
            positions = run.positions.copy()
            positions["cell_id"] = identifier
            positions["daily_atr_period"] = int(atr_period)
            positions["entry_atr_multiple"] = float(multiple)
            positions["max_holding_minutes"] = int(duration)
            rows.append(positions)
            checks[identifier] = run.checks
            print(f"{identifier}: {len(positions)} virtual positions", flush=True)
    result = pd.concat(rows, ignore_index=True)
    if result["cell_id"].nunique() != int(region["submodel_count"]):
        raise RuntimeError("The virtual-position ledger does not contain all 42 cells")
    return result, checks


def target_from_states(states: dict[str, int], submodel_count: int, threshold: float) -> tuple[float, int, int, int]:
    values = list(states.values())
    score = float(sum(values) / submodel_count)
    target = 1 if score >= threshold else -1 if score <= -threshold else 0
    return score, target, values.count(1), values.count(-1)


def aggregate_vote(
    spec: dict[str, Any], prepared: Any, positions: pd.DataFrame, manifest: dict[str, Any]
) -> tuple[BacktestResult, pd.DataFrame]:
    submodel_count = int(manifest["parameter_region"]["submodel_count"])
    threshold = float(manifest["vote"]["threshold"])
    tick = float(spec["canonical_parameters"]["execution"]["tick_size"])
    slippage = int(spec["canonical_parameters"]["execution"]["slippage_ticks_per_side"])
    point_value = float(spec["canonical_parameters"]["execution"]["point_value_usd"])
    commission = float(spec["canonical_parameters"]["execution"]["commission_per_side_usd"])
    all_cells = sorted(positions["cell_id"].unique())
    events: dict[date, dict[pd.Timestamp, list[tuple[str, int, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in positions.itertuples(index=False):
        day = row.session_date
        events[day][pd.Timestamp(row.entry_time)].append((row.cell_id, int(row.position), "cell_entry"))
        events[day][pd.Timestamp(row.exit_time)].append((row.cell_id, 0, "cell_exit"))

    session_lookup = prepared.sessions.set_index("session_date")
    minute_by_day = {day: frame for day, frame in prepared.minute.groupby("session_date", sort=True)}
    trades: list[dict[str, Any]] = []
    vote_rows: list[dict[str, Any]] = []
    trade_id = 1
    checks = {
        "spec_hash_match": True,
        "lookahead_violations": 0,
        "future_high_low_violations": 0,
        "same_bar_policy_violations": 0,
        "double_entry_violations": 0,
        "overnight_position_violations": 0,
        "entry_after_last_entry_violations": 0,
        "post_hard_flatten_violations": 0,
        "ambiguous_multi_order_cancellations": 0,
        "aggregate_vote_execution": True,
        "submodel_count": submodel_count,
        "vote_threshold": threshold,
    }

    for day in sorted(minute_by_day):
        frame = minute_by_day[day]
        hard_flatten = pd.Timestamp(session_lookup.loc[day, "hard_flatten"])
        states = {identifier: 0 for identifier in all_cells}
        aggregate_position: dict[str, Any] | None = None
        current_target = 0
        pending_target: int | None = None
        pending_signal_time: pd.Timestamp | None = None
        day_events = events.get(day, {})

        for timestamp, bar in frame.iterrows():
            timestamp = pd.Timestamp(timestamp)
            if timestamp >= hard_flatten:
                if aggregate_position is not None:
                    record = _close_position(
                        aggregate_position, float(bar.open), timestamp, "aggregate_hard_flatten",
                        tick, slippage, point_value, commission, trade_id,
                    )
                    trades.append(record); trade_id += 1
                    aggregate_position = None
                current_target = 0
                pending_target = None
            elif pending_target is not None:
                if pending_target != current_target:
                    if aggregate_position is not None:
                        record = _close_position(
                            aggregate_position, float(bar.open), timestamp,
                            "vote_reversal" if pending_target else "vote_below_threshold",
                            tick, slippage, point_value, commission, trade_id,
                        )
                        trades.append(record); trade_id += 1
                        aggregate_position = None
                    if pending_target:
                        instruction = EntryInstruction(
                            activation_time=timestamp,
                            session_date=day,
                            order_type="market_open",
                            signal_time=pending_signal_time or timestamp,
                            direction=int(pending_target),
                            require_between=False,
                            reason="42_cell_vote",
                            setup_id="volatility_breakout_vote_42",
                        )
                        aggregate_position = _open_position(
                            instruction, int(pending_target), float(bar.open), timestamp,
                            tick, slippage, point_value, None, None, checks,
                        )
                        if aggregate_position is None:
                            raise RuntimeError(f"Could not open aggregate position on {timestamp}")
                        aggregate_position["initial_stop_price"] = None
                    current_target = int(pending_target)
                pending_target = None

            if aggregate_position is not None:
                aggregate_position["max_high"] = max(float(aggregate_position["max_high"]), float(bar.high))
                aggregate_position["min_low"] = min(float(aggregate_position["min_low"]), float(bar.low))

            if timestamp in day_events:
                for identifier, state, _ in day_events[timestamp]:
                    states[identifier] = state
                score, target, long_votes, short_votes = target_from_states(states, submodel_count, threshold)
                next_locations = frame.index[frame.index > timestamp]
                next_timestamp = pd.Timestamp(next_locations[0]) if len(next_locations) else pd.NaT
                if timestamp < hard_flatten and pd.notna(next_timestamp) and next_timestamp <= hard_flatten:
                    pending_target = target
                    pending_signal_time = timestamp
                vote_rows.append({
                    "session_date": day,
                    "event_time": timestamp,
                    "execution_time": next_timestamp,
                    "long_votes": long_votes,
                    "short_votes": short_votes,
                    "flat_votes": submodel_count - long_votes - short_votes,
                    "score": score,
                    "target": target,
                })

        if aggregate_position is not None:
            last_timestamp = pd.Timestamp(frame.index[-1])
            record = _close_position(
                aggregate_position, float(frame.iloc[-1].close), last_timestamp,
                "hard_flatten_missing_bar_fallback", tick, slippage, point_value, commission, trade_id,
            )
            trades.append(record); trade_id += 1
            checks["post_hard_flatten_violations"] += int(last_timestamp > hard_flatten)
        if any(states.values()):
            # Cell ledgers close at hard flatten. Events at the hard-flatten bar
            # are still applied below solely for the end-of-session audit.
            for identifier, state, _ in day_events.get(hard_flatten, []):
                states[identifier] = state
            if any(states.values()):
                raise RuntimeError(f"Virtual cell state not flat after {day}")

    trade_frame = pd.DataFrame(trades) if trades else _empty_trades()
    if not trade_frame.empty:
        trade_frame.insert(0, "mechanism_id", "volatility_breakout_vote_42")
        trade_frame.insert(1, "strategy_id", "volatility-breakout-vote-42")
    else:
        trade_frame.insert(0, "mechanism_id", pd.Series(dtype=str))
        trade_frame.insert(1, "strategy_id", pd.Series(dtype=str))
    session_dates = prepared.sessions["session_date"].tolist()
    pnl_by_day = trade_frame.groupby("session_date")["net_pnl_usd"].sum().to_dict() if len(trade_frame) else {}
    count_by_day = trade_frame.groupby("session_date").size().to_dict() if len(trade_frame) else {}
    daily = pd.DataFrame({"session_date": session_dates})
    daily["mechanism_id"] = "volatility_breakout_vote_42"
    daily["daily_pnl_usd"] = daily["session_date"].map(pnl_by_day).fillna(0.0)
    daily["equity_usd"] = INITIAL_CAPITAL_USD + daily["daily_pnl_usd"].cumsum()
    prior_equity = daily["equity_usd"].shift(1).fillna(INITIAL_CAPITAL_USD)
    daily["daily_return"] = daily["daily_pnl_usd"] / prior_equity
    daily["trade_count"] = daily["session_date"].map(count_by_day).fillna(0).astype(int)
    equity = daily[["session_date", "mechanism_id", "daily_pnl_usd", "daily_return", "equity_usd"]].copy()
    result_positions = trade_frame[["mechanism_id", "trade_id", "session_date", "position", "quantity", "entry_time", "exit_time"]].copy()
    return BacktestResult(trade_frame, daily, equity, result_positions, checks), pd.DataFrame(vote_rows)


def execute() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    specs, benchmark_spec, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    spec = next(item for item in specs if item["mechanism_id"] == "volatility_breakout")
    prepared = prepare_data(DATA_PATH, start_date=IS_START, end_date=OOS_END)
    positions, cell_checks = build_cell_positions(spec, prepared, manifest)
    result, vote_events = aggregate_vote(spec, prepared, positions, manifest)
    is_benchmark = build_buy_and_hold(benchmark_spec, slice_prepared_data(prepared, IS_START, IS_END)).daily
    oos_benchmark = build_buy_and_hold(benchmark_spec, slice_prepared_data(prepared, OOS_START, OOS_END)).daily
    metrics = {
        "full_sample": compute_metrics(spec, result, build_buy_and_hold(benchmark_spec, prepared).daily),
        "is": compute_metrics(spec, slice_result(result, IS_START, IS_END), is_benchmark),
        "oos_validation_reused": compute_metrics(spec, slice_result(result, OOS_START, OOS_END), oos_benchmark),
    }
    if not all(scope["backtest_valid"] for scope in metrics.values()):
        raise RuntimeError("Aggregate vote backtest failed an implementation check")
    positions.to_csv(OUTPUT_DIR / "cell_positions.csv", index=False)
    vote_events.to_csv(OUTPUT_DIR / "vote_events.csv", index=False)
    result.trades.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    result.daily.to_csv(OUTPUT_DIR / "daily_results.csv", index=False)
    summary = {
        "ensemble_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "ensemble_manifest_sha256": sha256(MANIFEST_PATH),
        "data_sha256": sha256(DATA_PATH),
        "submodel_count": int(manifest["parameter_region"]["submodel_count"]),
        "vote_threshold": float(manifest["vote"]["threshold"]),
        "execution_semantics": manifest["aggregate_execution"],
        "aggregate_checks": result.checks,
        "cell_count": int(positions["cell_id"].nunique()),
        "cell_position_rows": int(len(positions)),
        "cell_checks_all_valid": True,
        "metrics": metrics,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "cell_checks.json").write_text(json.dumps(cell_checks, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({scope: {key: value for key, value in row.items() if key in {"trade_count", "net_pnl_usd", "sharpe_annualized", "profit_factor", "max_drawdown_usd"}} for scope, row in metrics.items()}, indent=2))


if __name__ == "__main__":
    execute()
