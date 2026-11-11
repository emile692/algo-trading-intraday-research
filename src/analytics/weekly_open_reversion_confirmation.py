"""Frozen confirmatory phase for the Tuesday-to-weekly-open hypothesis.

This module deliberately exposes a finite, predeclared grid.  It does not
select a stop, horizon, direction, or distance bucket after observing returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Iterable

import numpy as np
import pandas as pd

from src.analytics.weekly_open_reversion import DISTANCE_BUCKETS, StudyConfig, distance_bucket
from src.config.settings import get_instrument_spec


STOP_MULTIPLES = (0.5, 1.0, 1.5, 2.0, 3.0)
HORIZONS = ("tue_1130", "tue_1600", "wed_1600")
ANCHORS = ("weekly_open", "opposite_placebo", "monday_close")
SUBPERIODS = (
    ("2010-2014", 2010, 2014),
    ("2015-2019", 2015, 2019),
    ("2020-2026", 2020, 2026),
)


@dataclass(frozen=True)
class ConfirmationConfig:
    timezone: str = "America/New_York"
    rth_open: time = time(9, 30)
    rth_close: time = time(16, 0)
    first_checkpoint: time = time(11, 30)
    include_overnight: bool = True
    require_same_contract: bool = True
    bootstrap_samples: int = 5000
    bootstrap_block_weeks: int = 8
    random_seed: int = 20261001
    minimum_positive_year_ratio: float = 0.60


def _at(day: pd.Timestamp, at: time, timezone: str) -> pd.Timestamp:
    return pd.Timestamp.combine(day.date(), at).tz_localize(timezone)


def _first_true(mask: np.ndarray) -> int | None:
    positions = np.flatnonzero(mask)
    return int(positions[0]) if len(positions) else None


def _simulate_path(
    window: pd.DataFrame,
    *,
    raw_entry: float,
    target: float,
    side: int,
    distance: float,
    stop_multiple: float,
    symbol: str,
) -> dict[str, object]:
    """Simulate one trade. A same-minute stop beats the target."""
    spec = get_instrument_spec(symbol)
    stop = raw_entry - side * stop_multiple * distance
    highs = window["high"].to_numpy(float)
    lows = window["low"].to_numpy(float)
    if side == 1:
        target_pos = _first_true(highs >= target)
        stop_pos = _first_true(lows <= stop)
    else:
        target_pos = _first_true(lows <= target)
        stop_pos = _first_true(highs >= stop)

    if stop_pos is not None and (target_pos is None or stop_pos <= target_pos):
        exit_pos, raw_exit, reason = stop_pos, stop, "stop"
    elif target_pos is not None:
        exit_pos, raw_exit, reason = target_pos, target, "target"
    else:
        exit_pos, raw_exit, reason = len(window) - 1, float(window["close"].iloc[-1]), "time"

    path = window.iloc[: exit_pos + 1]
    if side == 1:
        adverse_points = max(0.0, raw_entry - float(path["low"].min()))
    else:
        adverse_points = max(0.0, float(path["high"].max()) - raw_entry)

    slippage_points = float(spec["tick_size"]) * float(spec["slippage_ticks"])
    executed_entry = raw_entry + side * slippage_points
    executed_exit = raw_exit - side * slippage_points
    gross_points = side * (raw_exit - raw_entry)
    net_points = side * (executed_exit - executed_entry)
    commission = 2.0 * float(spec["commission_per_side_usd"])
    gross_usd = gross_points * float(spec["point_value_usd"])
    net_usd = net_points * float(spec["point_value_usd"]) - commission
    round_trip_cost_usd = gross_usd - net_usd
    return {
        "stop_price": stop,
        "exit_time": window.index[exit_pos],
        "raw_exit_price": raw_exit,
        "executed_entry_price": executed_entry,
        "executed_exit_price": executed_exit,
        "exit_reason": reason,
        "target_before_stop": reason == "target",
        "same_bar_target_stop": bool(
            target_pos is not None and stop_pos is not None and target_pos == stop_pos
        ),
        "gross_points": gross_points,
        "net_points": net_points,
        "gross_pnl_usd": gross_usd,
        "net_pnl_usd": net_usd,
        "round_trip_cost_usd": round_trip_cost_usd,
        "net_pnl_d": net_points / distance - commission / (float(spec["point_value_usd"]) * distance),
        "mae_points": adverse_points,
        "mae_d": adverse_points / distance,
    }


def build_confirmation_trades(
    bars: pd.DataFrame,
    *,
    symbol: str,
    config: ConfirmationConfig = ConfirmationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the complete anchor x stop x horizon ledger for one instrument."""
    required_columns = {"open", "high", "low", "close"}
    if not required_columns.issubset(bars.columns):
        raise ValueError(f"Missing OHLC columns: {sorted(required_columns - set(bars.columns))}")
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("bars must have a timezone-aware DatetimeIndex")
    local = bars.sort_index().tz_convert(config.timezone)
    normalized = pd.Series(local.index.normalize(), index=local.index)
    mondays = pd.DatetimeIndex(normalized[local.index.weekday == 0].unique()).sort_values()
    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []

    for monday in mondays:
        tuesday, wednesday = monday + pd.Timedelta(days=1), monday + pd.Timedelta(days=2)
        monday_open_ts = _at(monday, config.rth_open, config.timezone)
        monday_close_bar = _at(monday, config.rth_close, config.timezone) - pd.Timedelta(minutes=1)
        entry_ts = _at(tuesday, config.rth_open, config.timezone)
        end_times = {
            "tue_1130": _at(tuesday, config.first_checkpoint, config.timezone),
            "tue_1600": _at(tuesday, config.rth_close, config.timezone),
            "wed_1600": _at(wednesday, config.rth_close, config.timezone),
        }
        mandatory = {
            "monday_open": monday_open_ts,
            "tuesday_open": entry_ts,
            **{f"{key}_last_bar": value - pd.Timedelta(minutes=1) for key, value in end_times.items()},
        }
        missing = [key for key, stamp in mandatory.items() if stamp not in local.index]
        if missing:
            audit.append({"symbol": symbol, "week_monday": monday.date(), "status": "excluded", "reason": ";".join(missing)})
            continue

        weekly_open = float(local.at[monday_open_ts, "open"])
        raw_entry = float(local.at[entry_ts, "open"])
        weekly_distance = abs(raw_entry - weekly_open)
        if not np.isfinite(weekly_distance) or weekly_distance <= 0 or weekly_open <= 0:
            audit.append({"symbol": symbol, "week_monday": monday.date(), "status": "excluded", "reason": "zero_or_invalid_weekly_distance"})
            continue

        full_window = local.loc[(local.index >= entry_ts) & (local.index < end_times["wed_1600"])].copy()
        if config.require_same_contract and "instrument_id" in local.columns:
            anchor_contract = local.at[monday_open_ts, "instrument_id"]
            changed = bool(
                local.at[entry_ts, "instrument_id"] != anchor_contract
                or full_window["instrument_id"].nunique(dropna=False) != 1
                or full_window["instrument_id"].iloc[0] != anchor_contract
            )
            if changed:
                audit.append({"symbol": symbol, "week_monday": monday.date(), "status": "excluded", "reason": "contract_change"})
                continue

        weekly_side = 1 if weekly_open > raw_entry else -1
        anchor_specs = {
            "weekly_open": (weekly_open, weekly_side, weekly_distance),
            "opposite_placebo": (raw_entry - weekly_side * weekly_distance, -weekly_side, weekly_distance),
        }
        if monday_close_bar in local.index:
            monday_close = float(local.at[monday_close_bar, "close"])
            monday_distance = abs(raw_entry - monday_close)
            if np.isfinite(monday_distance) and monday_distance > 0 and monday_close > 0:
                anchor_specs["monday_close"] = (
                    monday_close,
                    1 if monday_close > raw_entry else -1,
                    monday_distance,
                )

        for anchor, (target, side, distance) in anchor_specs.items():
            bucket = distance_bucket(100.0 * distance / target) if target > 0 else None
            for horizon, end_time in end_times.items():
                window = full_window.loc[full_window.index < end_time]
                for stop_multiple in STOP_MULTIPLES:
                    result = _simulate_path(
                        window,
                        raw_entry=raw_entry,
                        target=float(target),
                        side=int(side),
                        distance=float(distance),
                        stop_multiple=stop_multiple,
                        symbol=symbol,
                    )
                    rows.append(
                        {
                            "symbol": symbol,
                            "week_monday": monday.date(),
                            "anchor": anchor,
                            "horizon": horizon,
                            "stop_multiple": stop_multiple,
                            "entry_time": entry_ts,
                            "raw_entry_price": raw_entry,
                            "target_price": target,
                            "direction": "long" if side == 1 else "short",
                            "distance_points": distance,
                            "distance_pct": 100.0 * distance / target,
                            "distance_bucket": bucket,
                            **result,
                        }
                    )
        audit.append({"symbol": symbol, "week_monday": monday.date(), "status": "included", "reason": ""})

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["distance_bucket"] = pd.Categorical(frame["distance_bucket"], categories=DISTANCE_BUCKETS, ordered=True)
    return frame, pd.DataFrame(audit)


def combine_ledgers(items: Iterable[tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = list(items)
    return pd.concat([x[0] for x in items], ignore_index=True), pd.concat([x[1] for x in items], ignore_index=True)


def deduplicate_underlying_weeks(trades: pd.DataFrame) -> pd.DataFrame:
    """Keep NQ on common weeks; use MNQ only when no NQ observation exists."""
    out = trades.copy()
    out["symbol_priority"] = out["symbol"].map({"NQ": 0, "MNQ": 1}).fillna(2)
    keys = ["week_monday", "anchor", "horizon", "stop_multiple"]
    out = out.sort_values(keys + ["symbol_priority"]).drop_duplicates(keys, keep="first")
    return out.drop(columns="symbol_priority").reset_index(drop=True)


def _profit_factor(values: pd.Series) -> float:
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    return gains / losses if losses > 0 else np.inf


def _max_drawdown(values: pd.Series) -> float:
    equity = values.cumsum()
    return float((equity.cummax() - equity).max()) if len(equity) else np.nan


def summarize_group(group: pd.DataFrame) -> dict[str, object]:
    values = pd.to_numeric(group["net_pnl_usd"], errors="coerce")
    mean_d = float(group["distance_points"].mean())
    point_value = float(get_instrument_spec(str(group["symbol"].iloc[0]))["point_value_usd"])
    mean_cost = float(group["round_trip_cost_usd"].mean())
    k = float(group["stop_multiple"].iloc[0])
    avg_reward = mean_d * point_value - mean_cost
    avg_risk = k * mean_d * point_value + mean_cost
    theoretical_be = avg_risk / (avg_reward + avg_risk) if avg_reward > 0 else 1.0
    return {
        "n": len(group),
        "target_before_stop_rate": float(group["target_before_stop"].mean()),
        "theoretical_break_even_rate": theoretical_be,
        "realized_win_rate": float((values > 0).mean()),
        "net_expectancy_usd": float(values.mean()),
        "net_expectancy_d": float(group["net_pnl_d"].mean()),
        "profit_factor": _profit_factor(values),
        "max_drawdown_usd": _max_drawdown(values),
        "mae_median_d": float(group["mae_d"].median()),
        "mae_mean_d": float(group["mae_d"].mean()),
        "target_count": int((group["exit_reason"] == "target").sum()),
        "stop_count": int((group["exit_reason"] == "stop").sum()),
        "time_count": int((group["exit_reason"] == "time").sum()),
        "same_bar_target_stop_count": int(group["same_bar_target_stop"].sum()),
    }


def summarize_grid(trades: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in trades.sort_values("week_monday").groupby(group_columns, observed=True, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(group_columns, keys)), **summarize_group(group)})
    return pd.DataFrame(rows)


def add_stability(trades: pd.DataFrame, grid: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = trades.copy()
    years = pd.to_datetime(work["week_monday"]).dt.year
    work["year"] = years
    work["subperiod"] = "outside"
    for label, start, end in SUBPERIODS:
        work.loc[years.between(start, end), "subperiod"] = label
    annual = summarize_grid(work, keys + ["year"])
    subperiod = summarize_grid(work.loc[work["subperiod"] != "outside"], keys + ["subperiod"])
    annual_stability = (
        annual.assign(positive_year=lambda x: x["net_expectancy_usd"] > 0)
        .groupby(keys, observed=True)
        .agg(year_count=("year", "size"), positive_year_ratio=("positive_year", "mean"))
        .reset_index()
    )
    sub_stability = (
        subperiod.assign(positive_subperiod=lambda x: x["net_expectancy_usd"] > 0)
        .groupby(keys, observed=True)
        .agg(subperiod_count=("subperiod", "size"), positive_subperiod_count=("positive_subperiod", "sum"))
        .reset_index()
    )
    return grid.merge(annual_stability, on=keys).merge(sub_stability, on=keys), annual, subperiod


def _moving_block_means(values: np.ndarray, samples: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.array([])
    block = min(block, n)
    starts = np.arange(n - block + 1)
    blocks_needed = int(np.ceil(n / block))
    output = np.empty(samples)
    for i in range(samples):
        picked = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[start : start + block] for start in picked])[:n]
        output[i] = sample.mean()
    return output


def bootstrap_inference(
    deduped: pd.DataFrame,
    *,
    config: ConfirmationConfig = ConfirmationConfig(),
) -> pd.DataFrame:
    """Block-bootstrap expectancy and paired anchor differences by grid cell."""
    rows = []
    rng = np.random.default_rng(config.random_seed)
    for (horizon, stop_multiple), weekly in deduped.loc[deduped["anchor"] == "weekly_open"].groupby(["horizon", "stop_multiple"]):
        weekly = weekly.sort_values("week_monday")
        draws = _moving_block_means(weekly["net_pnl_usd"].to_numpy(float), config.bootstrap_samples, config.bootstrap_block_weeks, rng)
        row: dict[str, object] = {
            "horizon": horizon,
            "stop_multiple": stop_multiple,
            "expectancy_ci95_low": float(np.quantile(draws, 0.025)),
            "expectancy_ci95_high": float(np.quantile(draws, 0.975)),
            "probability_expectancy_positive": float((draws > 0).mean()),
        }
        for comparator in ("opposite_placebo", "monday_close"):
            other = deduped.loc[
                (deduped["anchor"] == comparator)
                & (deduped["horizon"] == horizon)
                & (deduped["stop_multiple"] == stop_multiple),
                ["week_monday", "net_pnl_usd"],
            ]
            paired = weekly[["week_monday", "net_pnl_usd"]].merge(other, on="week_monday", suffixes=("_weekly", "_other"))
            differences = (paired["net_pnl_usd_weekly"] - paired["net_pnl_usd_other"]).to_numpy(float)
            diff_draws = _moving_block_means(differences, config.bootstrap_samples, config.bootstrap_block_weeks, rng)
            row[f"delta_vs_{comparator}_usd"] = float(differences.mean())
            row[f"delta_vs_{comparator}_ci95_low"] = float(np.quantile(diff_draws, 0.025))
            row[f"delta_vs_{comparator}_ci95_high"] = float(np.quantile(diff_draws, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def classify_grid(
    weekly_grid: pd.DataFrame,
    inference: pd.DataFrame,
    *,
    config: ConfirmationConfig = ConfirmationConfig(),
) -> tuple[pd.DataFrame, str, bool]:
    classified = weekly_grid.merge(inference, on=["horizon", "stop_multiple"])
    classified["positive_after_costs"] = (classified["net_expectancy_usd"] > 0) & (classified["profit_factor"] > 1)
    classified["temporally_stable"] = (
        (classified["positive_year_ratio"] >= config.minimum_positive_year_ratio)
        & (classified["positive_subperiod_count"] == classified["subperiod_count"])
        & (classified["subperiod_count"] == 3)
    )
    classified["statistically_positive"] = classified["expectancy_ci95_low"] > 0
    classified["beats_both_controls"] = (
        (classified["delta_vs_opposite_placebo_ci95_low"] > 0)
        & (classified["delta_vs_monday_close_ci95_low"] > 0)
    )
    classified["robust_cell"] = classified[["positive_after_costs", "temporally_stable", "statistically_positive", "beats_both_controls"]].all(axis=1)
    classified["positive_stable_cell"] = classified[["positive_after_costs", "temporally_stable"]].all(axis=1)

    horizon_order = {name: index for index, name in enumerate(HORIZONS)}
    stop_order = {value: index for index, value in enumerate(STOP_MULTIPLES)}
    def has_rectangle(column: str) -> bool:
        cells = {(horizon_order[r.horizon], stop_order[float(r.stop_multiple)]) for r in classified.itertuples() if bool(getattr(r, column))}
        return any(
            (h, s) in cells and (h + 1, s) in cells and (h, s + 1) in cells and (h + 1, s + 1) in cells
            for h in range(len(HORIZONS) - 1) for s in range(len(STOP_MULTIPLES) - 1)
        )

    robust_zone = has_rectangle("robust_cell")
    positive_stable_zone = has_rectangle("positive_stable_cell")
    if robust_zone:
        verdict = "ROBUST_ACTIONABLE_EDGE"
    elif positive_stable_zone:
        verdict = "WEAK_NON_ACTIONABLE"
    else:
        verdict = "REJECTED"
    return classified, verdict, robust_zone
