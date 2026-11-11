"""Tuesday-to-weekly-open reversion study for minute futures bars.

The study is deliberately descriptive: it creates one observation per week and
does not tune a trading rule.  All session labels are evaluated in New York
time, so daylight-saving changes are handled by the timezone conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Iterable

import numpy as np
import pandas as pd


NEW_YORK = "America/New_York"
DISTANCE_BUCKETS = ("0-0.25%", "0.25-0.50%", "0.50-1.00%", ">1.00%")


@dataclass(frozen=True)
class StudyConfig:
    timezone: str = NEW_YORK
    rth_open: time = time(9, 30)
    rth_close: time = time(16, 0)
    first_checkpoint: time = time(11, 30)
    include_overnight: bool = True
    require_same_contract: bool = True


def _validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bars must have a DatetimeIndex")
    if bars.index.tz is None:
        raise ValueError("bars index must be timezone-aware")
    if not bars.index.is_unique:
        raise ValueError("bars index must be unique")
    out = bars.sort_index().copy()
    if (out["high"] < out["low"]).any():
        raise ValueError("Found bars with high below low")
    return out


def distance_bucket(distance_pct: float) -> str:
    """Return a mutually exclusive bucket; 1.00% belongs to 0.50-1.00%."""
    if not np.isfinite(distance_pct) or distance_pct < 0:
        raise ValueError("distance_pct must be finite and non-negative")
    if distance_pct < 0.25:
        return DISTANCE_BUCKETS[0]
    if distance_pct < 0.50:
        return DISTANCE_BUCKETS[1]
    if distance_pct <= 1.00:
        return DISTANCE_BUCKETS[2]
    return DISTANCE_BUCKETS[3]


def _at(local: pd.DataFrame, day: pd.Timestamp, at: time) -> pd.Timestamp:
    return pd.Timestamp.combine(day.date(), at).tz_localize(local.index.tz)


def _first_hit(
    window: pd.DataFrame, *, side: int, target: float, adverse_distance: float
) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT]:
    if side == 1:
        target_mask = window["high"] >= target
        adverse_1_mask = window["low"] <= window.attrs["entry"] - adverse_distance
        adverse_2_mask = window["low"] <= window.attrs["entry"] - 2.0 * adverse_distance
    else:
        target_mask = window["low"] <= target
        adverse_1_mask = window["high"] >= window.attrs["entry"] + adverse_distance
        adverse_2_mask = window["high"] >= window.attrs["entry"] + 2.0 * adverse_distance

    def first(mask: pd.Series) -> pd.Timestamp | pd.NaT:
        hits = window.index[mask.to_numpy()]
        return hits[0] if len(hits) else pd.NaT

    return first(target_mask), first(adverse_1_mask), first(adverse_2_mask)


def _strictly_before(left: pd.Timestamp | pd.NaT, right: pd.Timestamp | pd.NaT) -> bool:
    """Conservative one-minute ordering: a same-bar hit is not 'before'."""
    return bool(pd.notna(left) and (pd.isna(right) or left < right))


def _checkpoint_pnl(
    window: pd.DataFrame,
    checkpoint: pd.Timestamp,
    *,
    entry: float,
    side: int,
    distance: float,
    target_time: pd.Timestamp | pd.NaT,
) -> tuple[float, float]:
    if pd.notna(target_time) and target_time <= checkpoint:
        return distance, 1.0
    eligible = window.loc[window.index < checkpoint, "close"]
    if eligible.empty:
        return np.nan, np.nan
    pnl = side * (float(eligible.iloc[-1]) - entry)
    return pnl, pnl / distance


def build_weekly_observations(
    bars: pd.DataFrame,
    *,
    symbol: str,
    config: StudyConfig = StudyConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build weekly observations and an exclusion audit.

    The input is expected to contain one-minute bars.  Timestamps label the
    beginning of each bar, hence the 16:00 mark uses the 15:59 close.
    """
    bars = _validate_bars(bars)
    local = bars.tz_convert(config.timezone)
    local_dates = pd.Series(local.index.normalize(), index=local.index)
    mondays = pd.DatetimeIndex(local_dates[local.index.weekday == 0].unique()).sort_values()
    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []

    for monday in mondays:
        tuesday = monday + pd.Timedelta(days=1)
        wednesday = monday + pd.Timedelta(days=2)
        monday_open_ts = _at(local, monday, config.rth_open)
        tuesday_open_ts = _at(local, tuesday, config.rth_open)
        tuesday_1130 = _at(local, tuesday, config.first_checkpoint)
        tuesday_close = _at(local, tuesday, config.rth_close)
        wednesday_close = _at(local, wednesday, config.rth_close)
        required = {
            "monday_open": monday_open_ts,
            "tuesday_open": tuesday_open_ts,
            "tuesday_11h30_last_bar": tuesday_1130 - pd.Timedelta(minutes=1),
            "tuesday_close_last_bar": tuesday_close - pd.Timedelta(minutes=1),
            "wednesday_close_last_bar": wednesday_close - pd.Timedelta(minutes=1),
        }
        missing = [name for name, stamp in required.items() if stamp not in local.index]
        if missing:
            audit.append(
                {"symbol": symbol, "week_monday": monday.date(), "status": "excluded", "reason": ";".join(missing)}
            )
            continue

        weekly_open = float(local.at[monday_open_ts, "open"])
        entry = float(local.at[tuesday_open_ts, "open"])
        distance = abs(entry - weekly_open)
        if not np.isfinite(distance) or distance <= 0 or weekly_open <= 0:
            audit.append(
                {"symbol": symbol, "week_monday": monday.date(), "status": "excluded", "reason": "zero_or_invalid_distance"}
            )
            continue

        side = 1 if weekly_open > entry else -1
        if config.include_overnight:
            window = local.loc[(local.index >= tuesday_open_ts) & (local.index < wednesday_close)].copy()
        else:
            is_tue_or_wed = local.index.normalize().isin([tuesday, wednesday])
            in_rth = (local.index.time >= config.rth_open) & (local.index.time < config.rth_close)
            window = local.loc[is_tue_or_wed & in_rth].copy()
        if window.empty:
            audit.append(
                {"symbol": symbol, "week_monday": monday.date(), "status": "excluded", "reason": "empty_window"}
            )
            continue

        contract_changed = False
        if "instrument_id" in window.columns:
            anchor_contract = local.at[monday_open_ts, "instrument_id"]
            contract_changed = bool(
                local.at[tuesday_open_ts, "instrument_id"] != anchor_contract
                or window["instrument_id"].nunique(dropna=False) != 1
                or window["instrument_id"].iloc[0] != anchor_contract
            )
        if config.require_same_contract and contract_changed:
            audit.append(
                {"symbol": symbol, "week_monday": monday.date(), "status": "excluded", "reason": "contract_change"}
            )
            continue

        window.attrs["entry"] = entry
        target_time, adverse_1_time, adverse_2_time = _first_hit(
            window, side=side, target=weekly_open, adverse_distance=distance
        )
        target_hit = pd.notna(target_time)
        pre_target = window.loc[:target_time] if target_hit else window
        favorable = side * (pre_target["high"] - entry) if side == 1 else side * (pre_target["low"] - entry)
        adverse = side * (pre_target["low"] - entry) if side == 1 else side * (pre_target["high"] - entry)
        full_favorable = side * (window["high"] - entry) if side == 1 else side * (window["low"] - entry)
        mae = max(0.0, -float(adverse.min()))
        mfe_before = max(0.0, float(favorable.max()))
        mfe_full = max(0.0, float(full_favorable.max()))

        pnl_1130, pnl_1130_d = _checkpoint_pnl(
            window, tuesday_1130, entry=entry, side=side, distance=distance, target_time=target_time
        )
        pnl_tue_close, pnl_tue_close_d = _checkpoint_pnl(
            window, tuesday_close, entry=entry, side=side, distance=distance, target_time=target_time
        )
        pnl_wed_close, pnl_wed_close_d = _checkpoint_pnl(
            window, wednesday_close, entry=entry, side=side, distance=distance, target_time=target_time
        )
        rows.append(
            {
                "symbol": symbol,
                "week_monday": monday.date(),
                "weekly_open": weekly_open,
                "entry_time": tuesday_open_ts,
                "entry_price": entry,
                "direction": "long" if side == 1 else "short",
                "distance_points": distance,
                "distance_pct": 100.0 * distance / weekly_open,
                "distance_bucket": distance_bucket(100.0 * distance / weekly_open),
                "target_hit": target_hit,
                "target_time": target_time,
                "target_before_1d_adverse": _strictly_before(target_time, adverse_1_time),
                "target_before_2d_adverse": _strictly_before(target_time, adverse_2_time),
                "adverse_1d_time": adverse_1_time,
                "adverse_2d_time": adverse_2_time,
                "same_bar_target_and_1d": bool(target_hit and pd.notna(adverse_1_time) and target_time == adverse_1_time),
                "same_bar_target_and_2d": bool(target_hit and pd.notna(adverse_2_time) and target_time == adverse_2_time),
                "mae_before_target_points": mae,
                "mae_before_target_d": mae / distance,
                "mfe_before_target_points": mfe_before,
                "mfe_before_target_d": mfe_before / distance,
                "mfe_to_wed_close_points": mfe_full,
                "mfe_to_wed_close_d": mfe_full / distance,
                "minutes_to_target": (target_time - tuesday_open_ts).total_seconds() / 60.0 if target_hit else np.nan,
                "pnl_tue_1130_points": pnl_1130,
                "pnl_tue_1130_d": pnl_1130_d,
                "pnl_tue_1600_points": pnl_tue_close,
                "pnl_tue_1600_d": pnl_tue_close_d,
                "pnl_wed_1600_points": pnl_wed_close,
                "pnl_wed_1600_d": pnl_wed_close_d,
                "contract_changed": contract_changed,
            }
        )
        audit.append({"symbol": symbol, "week_monday": monday.date(), "status": "included", "reason": ""})

    observations = pd.DataFrame(rows)
    if not observations.empty:
        observations["distance_bucket"] = pd.Categorical(
            observations["distance_bucket"], categories=DISTANCE_BUCKETS, ordered=True
        )
    return observations, pd.DataFrame(audit)


def summarize_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Summarize by instrument, direction and distance bucket."""
    if observations.empty:
        return pd.DataFrame()
    keys = ["symbol", "direction", "distance_bucket"]
    grouped = observations.groupby(keys, observed=False, dropna=False)
    summary = grouped.agg(
        n=("target_hit", "size"),
        target_rate=("target_hit", "mean"),
        target_before_1d_rate=("target_before_1d_adverse", "mean"),
        target_before_2d_rate=("target_before_2d_adverse", "mean"),
        same_bar_target_1d_rate=("same_bar_target_and_1d", "mean"),
        mae_median_d=("mae_before_target_d", "median"),
        mae_mean_d=("mae_before_target_d", "mean"),
        mfe_before_target_median_d=("mfe_before_target_d", "median"),
        mfe_before_target_mean_d=("mfe_before_target_d", "mean"),
        mfe_to_wed_close_median_d=("mfe_to_wed_close_d", "median"),
        mfe_to_wed_close_mean_d=("mfe_to_wed_close_d", "mean"),
        median_minutes_to_target=("minutes_to_target", "median"),
        pnl_tue_1130_mean_d=("pnl_tue_1130_d", "mean"),
        pnl_tue_1600_mean_d=("pnl_tue_1600_d", "mean"),
        pnl_wed_1600_mean_d=("pnl_wed_1600_d", "mean"),
    ).reset_index()
    summary = summary.loc[summary["n"] > 0].reset_index(drop=True)
    # Wilson intervals remain well behaved for the small tail buckets and for
    # observed rates equal to zero or one.
    z = 1.959963984540054
    for rate_column in ("target_rate", "target_before_1d_rate", "target_before_2d_rate"):
        p = summary[rate_column]
        n = summary["n"]
        denominator = 1.0 + z * z / n
        center = (p + z * z / (2.0 * n)) / denominator
        radius = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
        summary[f"{rate_column}_ci95_low"] = center - radius
        summary[f"{rate_column}_ci95_high"] = center + radius
    return summary


def combine_studies(studies: Iterable[tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    studies = list(studies)
    if not studies:
        return pd.DataFrame(), pd.DataFrame()
    return (
        pd.concat([item[0] for item in studies], ignore_index=True),
        pd.concat([item[1] for item in studies], ignore_index=True),
    )
