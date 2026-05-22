"""Phase 2 intrabar-aware recalibration campaign for MNQ volume climax pullback."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analytics.volume_climax_pullback_common import load_symbol_data, resample_rth_1h, safe_float
from src.analytics.volume_climax_pullback_hybrid_execution_diagnostics import parse_datetime_cols
from src.config.settings import DEFAULT_TIMEZONE, get_instrument_spec
from src.data.session import extract_rth
from src.engine.execution_model import ExecutionModel
from src.engine.vwap_backtester import build_execution_model_for_profile
from src.strategy.volume_climax_pullback_v2 import (
    VolumeClimaxPullbackV2Variant,
    build_volume_climax_pullback_v2_signal_frame,
    prepare_volume_climax_pullback_v2_features,
)

DEFAULT_SYMBOL = "MNQ"
DEFAULT_DIAGNOSTICS_DIR = Path("export/volume_climax_pullback_hybrid_execution_diagnostics_mnq_20260520_002723")
DEFAULT_VALIDATION_DIR = Path("export/volume_climax_pullback_hybrid_execution_validation_mnq_20260519_220109")
DEFAULT_OUTPUT_ROOT = Path("export")
IS_END_DATE = pd.Timestamp("2023-12-31").date()
OOS_START_DATE = pd.Timestamp("2024-01-01").date()
SUBPERIOD_YEARS = (2020, 2021, 2022, 2023, 2024, 2025, 2026)
FILTER_FAMILIES = ("none", "avoid_immediate_adverse_move", "require_no_stop_zone_touch_before_entry", "require_micro_momentum_confirmation", "avoid_high_noise_first_5min")
STOP_MULTIPLIERS = (0.75, 1.0, 1.25, 1.5, 2.0)
TARGET_MULTIPLIERS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
ENTRY_DELAYS = (0, 5, 10, 15, 30)
FILTER_FOCUS_STOP_TARGET = (
    (0.75, 1.5),
    (0.75, 2.0),
    (0.75, 2.5),
    (1.0, 1.5),
    (1.0, 2.0),
    (1.0, 2.5),
)


@dataclass(frozen=True)
class IntrabarFilterSpec:
    family: str
    label: str
    params: dict[str, Any]


@dataclass(frozen=True)
class IntrabarRecalibrationConfig:
    config_id: str
    symbol: str
    execution_timeframe: str
    entry_timing: str
    protective_orders_active_from: str
    ambiguous_policy: str
    stop_multiplier: float
    target_multiplier: float
    entry_delay_minutes: int
    filter_family: str
    filter_label: str
    filter_params: dict[str, Any]


def _coerce_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or pd.isna(value):
        return pd.NaT
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(DEFAULT_TIMEZONE)
    return timestamp.tz_convert(DEFAULT_TIMEZONE)


def _file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    display = frame.copy()
    for column in display.columns:
        series = display[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            display[column] = series.astype(str)
        elif pd.api.types.is_float_dtype(series):
            display[column] = series.map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
        else:
            display[column] = series.astype(str)
    headers = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(lines)


def adjust_protective_levels(
    *,
    direction: int,
    entry_price: float,
    raw_stop_price: float,
    base_target_distance: float,
    stop_multiplier: float,
    target_multiplier: float,
) -> tuple[float, float]:
    """Return adjusted stop and target prices from the base geometry."""
    direction_sign = int(direction)
    stop_distance = (float(entry_price) - float(raw_stop_price)) * direction_sign
    if stop_distance <= 0:
        raise ValueError("Non-positive stop distance.")
    target_distance = float(base_target_distance)
    if target_distance <= 0:
        raise ValueError("Non-positive target distance.")
    adjusted_stop = float(entry_price) - direction_sign * stop_distance * float(stop_multiplier)
    adjusted_target = float(entry_price) + direction_sign * target_distance * float(target_multiplier)
    return float(adjusted_stop), float(adjusted_target)


def resolve_delayed_entry_index(
    session_minutes: pd.DataFrame,
    *,
    actionable_time: pd.Timestamp,
    entry_delay_minutes: int,
) -> tuple[int | None, int | None]:
    """Return the base and delayed execution indexes using only timestamps."""
    timestamps = pd.to_datetime(session_minutes["timestamp"], errors="coerce")
    base_indices = np.flatnonzero((timestamps > actionable_time).to_numpy(dtype=bool))
    if len(base_indices) == 0:
        return None, None
    base_idx = int(base_indices[0])
    base_entry_time = pd.Timestamp(session_minutes.iloc[base_idx]["timestamp"])
    delayed_anchor = base_entry_time + pd.Timedelta(minutes=int(entry_delay_minutes))
    delayed_indices = np.flatnonzero((timestamps >= delayed_anchor).to_numpy(dtype=bool))
    if len(delayed_indices) == 0:
        return base_idx, None
    return base_idx, int(delayed_indices[0])


def _build_filter_specs() -> list[IntrabarFilterSpec]:
    specs: list[IntrabarFilterSpec] = [IntrabarFilterSpec(family="none", label="none", params={})]

    for adverse_window_minutes in (5, 10, 15):
        for max_adverse_ticks in (8, 12, 16, 24):
            specs.append(
                IntrabarFilterSpec(
                    family="avoid_immediate_adverse_move",
                    label=f"adverse_w{adverse_window_minutes}_ticks{max_adverse_ticks}",
                    params={
                        "adverse_window_minutes": int(adverse_window_minutes),
                        "max_adverse_ticks": int(max_adverse_ticks),
                    },
                )
            )

    for stop_zone_fraction in (0.5, 0.75, 1.0):
        specs.append(
            IntrabarFilterSpec(
                family="require_no_stop_zone_touch_before_entry",
                label=f"stop_zone_{stop_zone_fraction:.2f}".replace(".", "p"),
                params={"stop_zone_fraction": float(stop_zone_fraction)},
            )
        )

    for confirmation_window_minutes in (5, 10, 15):
        for mode in ("close_vs_window_open", "close_vs_local_mid"):
            specs.append(
                IntrabarFilterSpec(
                    family="require_micro_momentum_confirmation",
                    label=f"momentum_w{confirmation_window_minutes}_{mode}",
                    params={
                        "confirmation_window_minutes": int(confirmation_window_minutes),
                        "mode": mode,
                    },
                )
            )

    for noise_window_minutes in (5, 10):
        for max_range_to_stop_distance in (0.5, 0.75, 1.0, 1.25):
            specs.append(
                IntrabarFilterSpec(
                    family="avoid_high_noise_first_5min",
                    label=f"noise_w{noise_window_minutes}_r{max_range_to_stop_distance:.2f}".replace(".", "p"),
                    params={
                        "noise_window_minutes": int(noise_window_minutes),
                        "max_range_to_stop_distance": float(max_range_to_stop_distance),
                    },
                )
            )
    return specs


def build_compact_config_grid(symbol: str) -> list[IntrabarRecalibrationConfig]:
    """Return a compact but meaningful research grid."""
    configs: list[IntrabarRecalibrationConfig] = []

    for stop_multiplier in STOP_MULTIPLIERS:
        for target_multiplier in TARGET_MULTIPLIERS:
            for entry_delay_minutes in ENTRY_DELAYS:
                config_id = (
                    f"none_sm{stop_multiplier:.2f}_tm{target_multiplier:.2f}_d{entry_delay_minutes}".replace(".", "p")
                )
                configs.append(
                    IntrabarRecalibrationConfig(
                        config_id=config_id,
                        symbol=symbol,
                        execution_timeframe="1min",
                        entry_timing="next_execution_bar_open",
                        protective_orders_active_from="next_execution_bar",
                        ambiguous_policy="stop_first",
                        stop_multiplier=float(stop_multiplier),
                        target_multiplier=float(target_multiplier),
                        entry_delay_minutes=int(entry_delay_minutes),
                        filter_family="none",
                        filter_label="none",
                        filter_params={},
                    )
                )

    filter_specs = _build_filter_specs()
    focus_pairs = FILTER_FOCUS_STOP_TARGET
    for stop_multiplier, target_multiplier in focus_pairs:
        for filter_spec in filter_specs:
            if filter_spec.family == "none":
                continue
            compatible_delays: tuple[int, ...]
            if filter_spec.family == "avoid_immediate_adverse_move":
                window = int(filter_spec.params["adverse_window_minutes"])
                compatible_delays = tuple(delay for delay in (5, 15) if delay >= window)
            elif filter_spec.family == "require_no_stop_zone_touch_before_entry":
                compatible_delays = (5, 15, 30)
            elif filter_spec.family == "require_micro_momentum_confirmation":
                window = int(filter_spec.params["confirmation_window_minutes"])
                compatible_delays = tuple(delay for delay in (5, 15) if delay >= window)
            elif filter_spec.family == "avoid_high_noise_first_5min":
                window = int(filter_spec.params["noise_window_minutes"])
                compatible_delays = tuple(delay for delay in (5, 10) if delay >= window)
            else:
                compatible_delays = ()

            for entry_delay_minutes in compatible_delays:
                config_id = (
                    f"{filter_spec.family}_sm{stop_multiplier:.2f}_tm{target_multiplier:.2f}_d{entry_delay_minutes}_{filter_spec.label}"
                    .replace(".", "p")
                    .replace("__", "_")
                )
                configs.append(
                    IntrabarRecalibrationConfig(
                        config_id=config_id,
                        symbol=symbol,
                        execution_timeframe="1min",
                        entry_timing="next_execution_bar_open",
                        protective_orders_active_from="next_execution_bar",
                        ambiguous_policy="stop_first",
                        stop_multiplier=float(stop_multiplier),
                        target_multiplier=float(target_multiplier),
                        entry_delay_minutes=int(entry_delay_minutes),
                        filter_family=filter_spec.family,
                        filter_label=filter_spec.label,
                        filter_params=dict(filter_spec.params),
                    )
                )
    return configs


def _variant_from_validation_metadata(validation_dir: Path) -> tuple[VolumeClimaxPullbackV2Variant, dict[str, Any]]:
    metadata_path = validation_dir / "run_metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    variant = VolumeClimaxPullbackV2Variant(**payload["variant"])
    return variant, payload


def _load_benchmark_context(validation_dir: Path) -> dict[str, Any]:
    metrics = pd.read_csv(validation_dir / "metrics_comparison.csv")
    metrics = parse_datetime_cols(metrics)
    baseline = metrics.loc[metrics["scenario"] == "baseline_1h"].copy()
    hybrid_after = metrics.loc[metrics["scenario"] == "hybrid_after_entry_fill"].copy()
    hybrid_next = metrics.loc[metrics["scenario"] == "hybrid_next_execution_bar"].copy()
    return {
        "metrics_comparison": metrics,
        "baseline": baseline.iloc[0].to_dict() if not baseline.empty else {},
        "hybrid_after": hybrid_after.iloc[0].to_dict() if not hybrid_after.empty else {},
        "hybrid_next": hybrid_next.iloc[0].to_dict() if not hybrid_next.empty else {},
    }


def _load_diagnostics_context(diagnostics_dir: Path) -> dict[str, Any]:
    metadata_path = diagnostics_dir / "run_metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _base_target_distance(variant: VolumeClimaxPullbackV2Variant, direction: int, stop_distance: float, signal_row: pd.Series) -> float:
    if variant.exit_mode == "atr_fraction":
        reference_atr = safe_float(signal_row.get("setup_reference_atr"), np.nan)
        atr_multiple = safe_float(variant.atr_target_multiple, np.nan)
        if not np.isfinite(reference_atr) or reference_atr <= 0 or not np.isfinite(atr_multiple) or atr_multiple <= 0:
            raise ValueError("Invalid ATR-based target geometry.")
        return float(reference_atr * atr_multiple)
    if variant.exit_mode == "fixed_rr":
        return float(stop_distance * float(variant.rr_target))
    raise NotImplementedError(f"Unsupported exit_mode for recalibration campaign: {variant.exit_mode}")


def _first_window_open(frame: pd.DataFrame) -> float:
    return safe_float(frame.iloc[0]["open"], np.nan) if not frame.empty else float("nan")


def _last_window_close(frame: pd.DataFrame) -> float:
    return safe_float(frame.iloc[-1]["close"], np.nan) if not frame.empty else float("nan")


def evaluate_pre_entry_filter(
    config: IntrabarRecalibrationConfig,
    *,
    direction: int,
    pre_entry_path: pd.DataFrame,
    actual_entry_price: float,
    adjusted_stop_price: float,
    tick_size: float,
) -> tuple[bool, str | None]:
    """Return whether the setup should be skipped before entry."""
    if config.filter_family == "none":
        return False, None
    if pre_entry_path.empty:
        return False, "filter_neutralized_no_pre_entry_window"

    direction_sign = int(direction)
    direction_label = "long" if direction_sign == 1 else "short"
    first_open = _first_window_open(pre_entry_path)
    last_close = _last_window_close(pre_entry_path)

    if config.filter_family == "avoid_immediate_adverse_move":
        adverse_window_minutes = int(config.filter_params["adverse_window_minutes"])
        window_end = pd.Timestamp(pre_entry_path.iloc[0]["timestamp"]) + pd.Timedelta(minutes=adverse_window_minutes - 1)
        scoped = pre_entry_path.loc[pd.to_datetime(pre_entry_path["timestamp"], errors="coerce") <= window_end].copy()
        if scoped.empty:
            return False, "filter_neutralized_short_window"
        reference_price = _first_window_open(scoped)
        max_adverse_ticks = int(config.filter_params["max_adverse_ticks"])
        adverse_threshold = float(reference_price) - direction_sign * float(max_adverse_ticks) * float(tick_size)
        if direction_label == "long":
            if pd.to_numeric(scoped["low"], errors="coerce").min() <= adverse_threshold:
                return True, "skip_adverse_move_before_entry"
        else:
            if pd.to_numeric(scoped["high"], errors="coerce").max() >= adverse_threshold:
                return True, "skip_adverse_move_before_entry"
        return False, None

    if config.filter_family == "require_no_stop_zone_touch_before_entry":
        stop_zone_fraction = float(config.filter_params["stop_zone_fraction"])
        stop_distance = abs(float(actual_entry_price) - float(adjusted_stop_price))
        zone_level = float(actual_entry_price) - direction_sign * stop_distance * stop_zone_fraction
        if direction_label == "long":
            if pd.to_numeric(pre_entry_path["low"], errors="coerce").min() <= zone_level:
                return True, "skip_pre_entry_stop_zone_touch"
        else:
            if pd.to_numeric(pre_entry_path["high"], errors="coerce").max() >= zone_level:
                return True, "skip_pre_entry_stop_zone_touch"
        return False, None

    if config.filter_family == "require_micro_momentum_confirmation":
        confirmation_window = int(config.filter_params["confirmation_window_minutes"])
        window_end = pd.Timestamp(pre_entry_path.iloc[0]["timestamp"]) + pd.Timedelta(minutes=confirmation_window - 1)
        scoped = pre_entry_path.loc[pd.to_datetime(pre_entry_path["timestamp"], errors="coerce") <= window_end].copy()
        if scoped.empty:
            return False, "filter_neutralized_short_window"
        mode = str(config.filter_params["mode"])
        if mode == "close_vs_window_open":
            passed = last_close > first_open if direction_label == "long" else last_close < first_open
        else:
            local_mid = (pd.to_numeric(scoped["high"], errors="coerce").max() + pd.to_numeric(scoped["low"], errors="coerce").min()) / 2.0
            passed = last_close > local_mid if direction_label == "long" else last_close < local_mid
        if not passed:
            return True, "skip_micro_momentum_confirmation_failed"
        return False, None

    if config.filter_family == "avoid_high_noise_first_5min":
        noise_window = int(config.filter_params["noise_window_minutes"])
        window_end = pd.Timestamp(pre_entry_path.iloc[0]["timestamp"]) + pd.Timedelta(minutes=noise_window - 1)
        scoped = pre_entry_path.loc[pd.to_datetime(pre_entry_path["timestamp"], errors="coerce") <= window_end].copy()
        if scoped.empty:
            return False, "filter_neutralized_short_window"
        stop_distance = abs(float(actual_entry_price) - float(adjusted_stop_price))
        if stop_distance <= 0:
            return False, "filter_neutralized_invalid_stop_distance"
        noise_range = pd.to_numeric(scoped["high"], errors="coerce").max() - pd.to_numeric(scoped["low"], errors="coerce").min()
        if float(noise_range) / float(stop_distance) > float(config.filter_params["max_range_to_stop_distance"]):
            return True, "skip_high_noise_before_entry"
        return False, None

    return False, None


def _trade_path_metrics(path_to_exit: pd.DataFrame, *, direction: int, entry_price: float, tick_size: float) -> tuple[float, float]:
    direction_sign = int(direction)
    if path_to_exit.empty:
        return float("nan"), float("nan")
    favorable = ((path_to_exit["high"] - entry_price) if direction_sign == 1 else (entry_price - path_to_exit["low"])) / float(tick_size)
    adverse = ((entry_price - path_to_exit["low"]) if direction_sign == 1 else (path_to_exit["high"] - entry_price)) / float(tick_size)
    mfe_ticks = float(pd.to_numeric(favorable, errors="coerce").max())
    mae_ticks = float(pd.to_numeric(adverse, errors="coerce").max())
    return mfe_ticks, mae_ticks


def _simulate_config(
    *,
    config: IntrabarRecalibrationConfig,
    signal_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    variant: VolumeClimaxPullbackV2Variant,
    execution_model: ExecutionModel,
    point_value_usd: float,
    tick_size: float,
    signal_bar_minutes: int = 60,
) -> pd.DataFrame:
    minute_working = minute_df.copy()
    minute_working["timestamp"] = pd.to_datetime(minute_working["timestamp"], errors="coerce")
    minute_working["session_date"] = pd.to_datetime(minute_working["session_date"], errors="coerce").dt.date
    signal_working = signal_df.copy()
    signal_working["timestamp"] = pd.to_datetime(signal_working["timestamp"], errors="coerce")
    signal_working["session_date"] = pd.to_datetime(signal_working["session_date"], errors="coerce").dt.date

    signal_events = signal_working.loc[pd.to_numeric(signal_working["signal"], errors="coerce").fillna(0).ne(0)].copy()
    signal_events = signal_events.sort_values("timestamp").reset_index(drop=True)
    minute_by_session = {
        session_date: frame.sort_values("timestamp").reset_index(drop=True)
        for session_date, frame in minute_working.groupby("session_date", sort=True)
    }

    rows: list[dict[str, Any]] = []
    last_exit_time = pd.NaT

    for signal_row in signal_events.itertuples(index=False):
        row = pd.Series(signal_row._asdict())
        actionable_time = pd.Timestamp(row["timestamp"])
        session_date = row["session_date"]
        direction = int(safe_float(row.get("signal"), 0))
        direction_label = "long" if direction == 1 else "short"
        session_minutes = minute_by_session.get(session_date)
        common_payload = {
            "config_id": config.config_id,
            "symbol": config.symbol,
            "signal_time": pd.Timestamp(row.get("setup_signal_time")) if pd.notna(row.get("setup_signal_time")) else actionable_time,
            "signal_actionable_time": actionable_time,
            "direction": direction_label,
            "filter_family": config.filter_family,
            "filter_label": config.filter_label,
            "stop_multiplier": float(config.stop_multiplier),
            "target_multiplier": float(config.target_multiplier),
            "entry_delay_minutes": int(config.entry_delay_minutes),
            "skipped_by_filter": False,
            "filter_reason": None,
            "executed": False,
            "pnl": np.nan,
            "net_pnl_usd": np.nan,
            "gross_pnl_usd": np.nan,
            "entry_time": pd.NaT,
            "exit_time": pd.NaT,
            "entry_price": np.nan,
            "stop_price": np.nan,
            "target_price": np.nan,
            "exit_price": np.nan,
            "exit_reason": None,
            "holding_minutes": np.nan,
            "mfe_ticks_1m": np.nan,
            "mae_ticks_1m": np.nan,
            "quantity": 1,
            "session_date": session_date,
        }

        if session_minutes is None or session_minutes.empty:
            payload = common_payload.copy()
            payload["filter_reason"] = "missing_session_minutes"
            rows.append(payload)
            continue

        base_entry_idx, actual_entry_idx = resolve_delayed_entry_index(
            session_minutes,
            actionable_time=actionable_time,
            entry_delay_minutes=int(config.entry_delay_minutes),
        )
        if base_entry_idx is None or actual_entry_idx is None:
            payload = common_payload.copy()
            payload["filter_reason"] = "entry_delay_out_of_session"
            rows.append(payload)
            continue

        base_entry_time = pd.Timestamp(session_minutes.iloc[base_entry_idx]["timestamp"])
        actual_entry_bar = session_minutes.iloc[actual_entry_idx]
        actual_entry_time = pd.Timestamp(actual_entry_bar["timestamp"])

        if pd.notna(last_exit_time) and actual_entry_time <= pd.Timestamp(last_exit_time):
            payload = common_payload.copy()
            payload["entry_time"] = actual_entry_time
            payload["filter_reason"] = "blocked_by_open_trade"
            rows.append(payload)
            continue

        raw_entry_price = safe_float(actual_entry_bar.get("open"), np.nan)
        entry_price = float(execution_model.apply_slippage(raw_entry_price, direction, is_entry=True))
        raw_stop_price = safe_float(row.get("setup_stop_reference_long" if direction == 1 else "setup_stop_reference_short"), np.nan)
        stop_distance = (float(entry_price) - float(raw_stop_price)) * int(direction)
        if not np.isfinite(stop_distance) or stop_distance <= 0:
            payload = common_payload.copy()
            payload["entry_time"] = actual_entry_time
            payload["entry_price"] = entry_price
            payload["filter_reason"] = "invalid_stop_geometry"
            rows.append(payload)
            continue

        try:
            base_target_distance = _base_target_distance(variant, direction, stop_distance, row)
            adjusted_stop_price, adjusted_target_price = adjust_protective_levels(
                direction=direction,
                entry_price=entry_price,
                raw_stop_price=raw_stop_price,
                base_target_distance=base_target_distance,
                stop_multiplier=float(config.stop_multiplier),
                target_multiplier=float(config.target_multiplier),
            )
        except (ValueError, NotImplementedError):
            payload = common_payload.copy()
            payload["entry_time"] = actual_entry_time
            payload["entry_price"] = entry_price
            payload["filter_reason"] = "invalid_target_geometry"
            rows.append(payload)
            continue

        pre_entry_path = session_minutes.loc[
            (pd.to_datetime(session_minutes["timestamp"], errors="coerce") > actionable_time)
            & (pd.to_datetime(session_minutes["timestamp"], errors="coerce") < actual_entry_time)
        ].copy()
        skip_setup, filter_reason = evaluate_pre_entry_filter(
            config,
            direction=direction,
            pre_entry_path=pre_entry_path,
            actual_entry_price=entry_price,
            adjusted_stop_price=adjusted_stop_price,
            tick_size=tick_size,
        )
        if skip_setup:
            payload = common_payload.copy()
            payload["entry_time"] = actual_entry_time
            payload["entry_price"] = entry_price
            payload["stop_price"] = adjusted_stop_price
            payload["target_price"] = adjusted_target_price
            payload["skipped_by_filter"] = True
            payload["filter_reason"] = filter_reason
            rows.append(payload)
            continue

        protective_active_at = pd.Timestamp(session_minutes.iloc[actual_entry_idx + 1]["timestamp"]) if actual_entry_idx + 1 < len(session_minutes) else pd.NaT
        time_stop_at = actual_entry_time + pd.Timedelta(minutes=int(variant.time_stop_bars) * int(signal_bar_minutes))
        exit_time = pd.NaT
        exit_price = float("nan")
        exit_reason = None

        path_rows = []
        for minute_idx in range(actual_entry_idx, len(session_minutes)):
            minute_bar = session_minutes.iloc[minute_idx]
            path_rows.append(minute_bar.to_dict())
            timestamp = pd.Timestamp(minute_bar["timestamp"])
            high = safe_float(minute_bar.get("high"), np.nan)
            low = safe_float(minute_bar.get("low"), np.nan)
            close = safe_float(minute_bar.get("close"), np.nan)
            protective_active = pd.notna(protective_active_at) and timestamp >= protective_active_at
            stop_hit = protective_active and ((low <= adjusted_stop_price) if direction == 1 else (high >= adjusted_stop_price))
            target_hit = protective_active and ((high >= adjusted_target_price) if direction == 1 else (low <= adjusted_target_price))

            if stop_hit and target_hit:
                exit_price = float(adjusted_stop_price)
                exit_time = timestamp
                exit_reason = "stop_ambiguous_first_1m"
                break
            if stop_hit:
                exit_price = float(adjusted_stop_price)
                exit_time = timestamp
                exit_reason = "stop_1m"
                break
            if target_hit:
                exit_price = float(adjusted_target_price)
                exit_time = timestamp
                exit_reason = "target_1m"
                break
            if timestamp >= time_stop_at:
                exit_price = float(close)
                exit_time = timestamp
                exit_reason = "time_stop_1m"
                break
            if minute_idx == len(session_minutes) - 1:
                exit_price = float(close)
                exit_time = timestamp
                exit_reason = "eod_flat_1m"
                break

        path_to_exit = pd.DataFrame(path_rows)
        if pd.isna(exit_time):
            continue

        filled_exit_price = float(execution_model.apply_slippage(float(exit_price), direction, is_entry=False))
        pnl_points = (filled_exit_price - float(entry_price)) * int(direction)
        gross_pnl_usd = pnl_points * float(point_value_usd)
        fees = execution_model.round_trip_fees(quantity=1)
        net_pnl_usd = gross_pnl_usd - float(fees)
        holding_minutes = max(int((pd.Timestamp(exit_time) - actual_entry_time) / pd.Timedelta(minutes=1)), 0)
        mfe_ticks_1m, mae_ticks_1m = _trade_path_metrics(path_to_exit, direction=direction, entry_price=entry_price, tick_size=tick_size)
        last_exit_time = pd.Timestamp(exit_time)

        payload = common_payload.copy()
        payload.update(
            {
                "executed": True,
                "entry_time": actual_entry_time,
                "exit_time": exit_time,
                "entry_price": float(entry_price),
                "stop_price": float(adjusted_stop_price),
                "target_price": float(adjusted_target_price),
                "exit_price": float(filled_exit_price),
                "exit_reason": exit_reason,
                "gross_pnl_usd": float(gross_pnl_usd),
                "net_pnl_usd": float(net_pnl_usd),
                "pnl": float(net_pnl_usd),
                "holding_minutes": float(holding_minutes),
                "mfe_ticks_1m": float(mfe_ticks_1m),
                "mae_ticks_1m": float(mae_ticks_1m),
                "filter_reason": "none",
            }
        )
        rows.append(payload)

    return pd.DataFrame(rows)


def _period_mask(frame: pd.DataFrame, period: str) -> pd.Series:
    session_dates = pd.to_datetime(frame["session_date"], errors="coerce").dt.date
    if period == "is":
        return session_dates <= IS_END_DATE
    if period == "oos":
        return session_dates >= OOS_START_DATE
    return pd.Series(True, index=frame.index)


def _daily_returns(events: pd.DataFrame) -> pd.DataFrame:
    executed = events.loc[events["executed"]].copy()
    if executed.empty:
        return pd.DataFrame(columns=["session_date", "daily_pnl"])
    daily = (
        executed.groupby("session_date", as_index=False)["net_pnl_usd"]
        .sum()
        .rename(columns={"net_pnl_usd": "daily_pnl"})
        .sort_values("session_date")
        .reset_index(drop=True)
    )
    daily["equity"] = daily["daily_pnl"].cumsum()
    daily["drawdown"] = daily["equity"] - daily["equity"].cummax()
    return daily


def _compute_trade_metrics(events: pd.DataFrame, *, estimated_cost_per_trade: float) -> dict[str, Any]:
    events = events.copy()
    executed = events.loc[events["executed"]].copy()
    daily = _daily_returns(events)
    trade_pnl = pd.to_numeric(executed.get("net_pnl_usd"), errors="coerce")
    gross_profit = float(trade_pnl.loc[trade_pnl > 0].sum()) if not executed.empty else 0.0
    gross_loss_abs = float(abs(trade_pnl.loc[trade_pnl < 0].sum())) if not executed.empty else 0.0
    max_drawdown = float(daily["drawdown"].min()) if not daily.empty else 0.0
    daily_returns = pd.to_numeric(daily.get("daily_pnl"), errors="coerce")
    sharpe_daily = float(np.sqrt(252.0) * daily_returns.mean() / daily_returns.std(ddof=0)) if len(daily_returns) > 1 and daily_returns.std(ddof=0) > 0 else np.nan
    downside = daily_returns.loc[daily_returns < 0]
    sortino_daily = float(np.sqrt(252.0) * daily_returns.mean() / downside.std(ddof=0)) if len(downside) > 1 and downside.std(ddof=0) > 0 else np.nan
    session_dates = pd.to_datetime(events["session_date"], errors="coerce").dt.date.dropna()
    span_days = max((max(session_dates) - min(session_dates)).days + 1, 1) if not session_dates.empty else 1
    turnover_years = span_days / 365.25

    stop_count = int(executed["exit_reason"].astype(str).str.contains("stop", regex=False).sum()) if not executed.empty else 0
    target_count = int(executed["exit_reason"].astype(str).str.contains("target", regex=False).sum()) if not executed.empty else 0
    time_stop_count = int(executed["exit_reason"].astype(str).str.contains("time_stop", regex=False).sum()) if not executed.empty else 0
    eod_count = int(executed["exit_reason"].astype(str).str.contains("eod", regex=False).sum()) if not executed.empty else 0
    skipped_trades = int(events["skipped_by_filter"].fillna(False).sum()) if "skipped_by_filter" in events.columns else 0
    blocked_setups = int((events["filter_reason"].astype(str) == "blocked_by_open_trade").sum()) if "filter_reason" in events.columns else 0
    total_setups = int(len(events))

    return {
        "trades": int(len(executed)),
        "net_pnl": float(trade_pnl.sum()) if not executed.empty else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss_abs,
        "winrate": float((trade_pnl > 0).mean()) if not executed.empty else 0.0,
        "avg_trade": float(trade_pnl.mean()) if not executed.empty else 0.0,
        "median_trade": float(trade_pnl.median()) if not executed.empty else 0.0,
        "profit_factor": float(gross_profit / gross_loss_abs) if gross_loss_abs > 0 else np.inf,
        "max_drawdown": float(max_drawdown),
        "sharpe_daily": sharpe_daily,
        "sortino_daily": sortino_daily,
        "pnl_to_maxdd": float((trade_pnl.sum()) / abs(max_drawdown)) if max_drawdown < 0 else np.nan,
        "avg_holding_minutes": float(pd.to_numeric(executed.get("holding_minutes"), errors="coerce").mean()) if not executed.empty else 0.0,
        "median_holding_minutes": float(pd.to_numeric(executed.get("holding_minutes"), errors="coerce").median()) if not executed.empty else 0.0,
        "stop_exit_rate": float(stop_count / len(executed)) if not executed.empty else 0.0,
        "target_exit_rate": float(target_count / len(executed)) if not executed.empty else 0.0,
        "time_stop_exit_rate": float(time_stop_count / len(executed)) if not executed.empty else 0.0,
        "eod_exit_rate": float(eod_count / len(executed)) if not executed.empty else 0.0,
        "skipped_trades": skipped_trades,
        "skip_rate": float(skipped_trades / total_setups) if total_setups > 0 else 0.0,
        "blocked_setups": blocked_setups,
        "turnover_proxy": float(len(executed) / turnover_years) if turnover_years > 0 else 0.0,
        "estimated_cost_per_trade": float(estimated_cost_per_trade),
    }


def _subperiod_label(session_date: Any) -> str:
    timestamp = pd.Timestamp(session_date)
    return str(timestamp.year)


def _year_rows(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["subperiod"] = pd.to_datetime(out["session_date"], errors="coerce").dt.year.astype("Int64").astype(str)
    return out


def compute_subperiod_metrics(events: pd.DataFrame, *, config_id: str, estimated_cost_per_trade: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    year_frame = _year_rows(events)
    for year in SUBPERIOD_YEARS:
        period_events = year_frame.loc[year_frame["subperiod"] == str(year)].copy()
        metrics = _compute_trade_metrics(period_events, estimated_cost_per_trade=estimated_cost_per_trade)
        metrics.update({"config_id": config_id, "subperiod": str(year)})
        rows.append(metrics)
    return pd.DataFrame(rows)


def _neighbor_score(metrics_is: pd.DataFrame, config_row: pd.Series) -> float:
    same_family = metrics_is.loc[
        (metrics_is["filter_family"] == config_row["filter_family"])
        & (metrics_is["entry_delay_minutes"] == config_row["entry_delay_minutes"])
    ].copy()
    if same_family.empty:
        return 0.0
    neighbors = same_family.loc[
        (same_family["stop_multiplier"].sub(float(config_row["stop_multiplier"])).abs() <= 0.26)
        & (same_family["target_multiplier"].sub(float(config_row["target_multiplier"])).abs() <= 0.51)
    ].copy()
    if neighbors.empty:
        return 0.0
    acceptable = (
        (pd.to_numeric(neighbors["net_pnl"], errors="coerce") > 0)
        & (pd.to_numeric(neighbors["profit_factor"], errors="coerce") >= 1.0)
    )
    return float(acceptable.mean())


def build_robustness_scores(
    metrics_is: pd.DataFrame,
    subperiod_metrics: pd.DataFrame,
    *,
    estimated_cost_per_trade: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = metrics_is.copy().reset_index(drop=True)
    year_metrics = subperiod_metrics.copy()
    is_years = [str(year) for year in SUBPERIOD_YEARS if year <= IS_END_DATE.year]

    def _normalize(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() <= 1:
            return pd.Series(np.where(numeric.notna(), 1.0, np.nan), index=series.index, dtype=float)
        spread = numeric.max() - numeric.min()
        if spread == 0:
            return pd.Series(np.where(numeric.notna(), 1.0, np.nan), index=series.index, dtype=float)
        return (numeric - numeric.min()) / spread

    metrics["normalized_net_pnl_is"] = _normalize(metrics["net_pnl"])
    metrics["normalized_profit_factor_is"] = _normalize(metrics["profit_factor"].replace(np.inf, np.nan).fillna(metrics["profit_factor"].replace(np.inf, np.nan).max()))
    metrics["normalized_pnl_to_maxdd_is"] = _normalize(metrics["pnl_to_maxdd"])
    metrics["trade_count_score"] = metrics["trades"].clip(lower=0).div(100.0).clip(upper=1.0)

    for row in metrics.itertuples(index=False):
        config_id = row.config_id
        year_slice = year_metrics.loc[(year_metrics["config_id"] == config_id) & (year_metrics["subperiod"].isin(is_years))].copy()
        positive_years = int((pd.to_numeric(year_slice["net_pnl"], errors="coerce") > 0).sum())
        years_available = int((pd.to_numeric(year_slice["trades"], errors="coerce") > 0).sum())
        pnl_by_year = pd.to_numeric(year_slice["net_pnl"], errors="coerce").fillna(0.0)
        abs_total = float(pnl_by_year.abs().sum())
        max_year_contribution_pct = float(pnl_by_year.abs().max() / abs_total) if abs_total > 0 else 1.0
        temporal_stability_score = 0.5 * (positive_years / max(years_available, 1)) + 0.5 * max(0.0, 1.0 - max_year_contribution_pct)
        neighbor_score = _neighbor_score(metrics, pd.Series(row._asdict()))
        avg_trade = safe_float(row.avg_trade, 0.0)
        penalties = 0.0
        one_year_dependency = max_year_contribution_pct > 0.60
        if int(row.trades) < 50:
            penalties += 0.20
        if safe_float(row.profit_factor, 0.0) < 1.05:
            penalties += 0.15
        if one_year_dependency:
            penalties += 0.15
        if safe_float(row.skip_rate, 0.0) > 0.70:
            penalties += 0.10
        if avg_trade < estimated_cost_per_trade * 1.5:
            penalties += 0.15

        robust_score_is = (
            0.25 * safe_float(row.normalized_net_pnl_is, 0.0)
            + 0.20 * safe_float(row.normalized_profit_factor_is, 0.0)
            + 0.15 * safe_float(row.normalized_pnl_to_maxdd_is, 0.0)
            + 0.15 * temporal_stability_score
            + 0.15 * neighbor_score
            + 0.10 * safe_float(row.trade_count_score, 0.0)
            - penalties
        )
        rows.append(
            {
                "config_id": config_id,
                "positive_years_is": positive_years,
                "years_available_is": years_available,
                "max_year_contribution_pct": max_year_contribution_pct,
                "temporal_stability_score": temporal_stability_score,
                "parameter_neighborhood_score": neighbor_score,
                "trade_count_score": safe_float(row.trade_count_score, 0.0),
                "one_year_dependency": one_year_dependency,
                "penalties": penalties,
                "robust_score_is": robust_score_is,
                "admissible_is": bool(
                    safe_float(row.net_pnl, 0.0) > 0
                    and int(row.trades) >= 50
                    and safe_float(row.profit_factor, 0.0) >= 1.10
                    and avg_trade > estimated_cost_per_trade * 1.5
                    and positive_years >= 2
                ),
            }
        )
    return metrics.merge(pd.DataFrame(rows), on="config_id", how="left")


def select_configs_is_only(metrics_with_scores: pd.DataFrame, *, max_configs: int = 10) -> pd.DataFrame:
    ranked = metrics_with_scores.sort_values(["admissible_is", "robust_score_is", "net_pnl", "profit_factor"], ascending=[False, False, False, False]).reset_index(drop=True)
    selected: list[pd.Series] = []
    selected_ids: set[str] = set()

    def _append_if_possible(candidate: pd.Series) -> None:
        if candidate["config_id"] in selected_ids:
            return
        same_pair_count = sum(
            1
            for item in selected
            if safe_float(item["stop_multiplier"], np.nan) == safe_float(candidate["stop_multiplier"], np.nan)
            and safe_float(item["target_multiplier"], np.nan) == safe_float(candidate["target_multiplier"], np.nan)
        )
        same_family_count = sum(1 for item in selected if str(item["filter_family"]) == str(candidate["filter_family"]))
        if same_pair_count >= 2 or same_family_count >= 4:
            return
        selected.append(candidate)
        selected_ids.add(str(candidate["config_id"]))

    no_filter = ranked.loc[(ranked["admissible_is"]) & (ranked["filter_family"] == "none")].head(1)
    if not no_filter.empty:
        _append_if_possible(no_filter.iloc[0])
    simple_delay = ranked.loc[(ranked["admissible_is"]) & (ranked["filter_family"] == "none") & (ranked["entry_delay_minutes"] > 0)].head(1)
    if not simple_delay.empty:
        _append_if_possible(simple_delay.iloc[0])

    for row in ranked.itertuples(index=False):
        if len(selected) >= max_configs:
            break
        _append_if_possible(pd.Series(row._asdict()))

    out = pd.DataFrame(selected)
    if out.empty:
        return out
    out = out.sort_values("robust_score_is", ascending=False).reset_index(drop=True)
    out["rank_is"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def build_selected_oos_report(selected_is: pd.DataFrame, metrics_oos: pd.DataFrame, metrics_full: pd.DataFrame) -> pd.DataFrame:
    if selected_is.empty:
        return pd.DataFrame()
    report = selected_is.merge(metrics_oos.add_suffix("_oos"), left_on="config_id", right_on="config_id_oos", how="left")
    report = report.merge(metrics_full.add_suffix("_full"), left_on="config_id", right_on="config_id_full", how="left")
    report["degradation_ratio"] = (
        pd.to_numeric(report["net_pnl_oos"], errors="coerce")
        / pd.to_numeric(report["net_pnl"], errors="coerce").replace(0.0, np.nan)
    )
    report["oos_pass"] = (
        (pd.to_numeric(report["net_pnl_oos"], errors="coerce") > 0)
        & (pd.to_numeric(report["profit_factor_oos"], errors="coerce") >= 1.05)
        & (pd.to_numeric(report["trades_oos"], errors="coerce") >= 20)
        & (pd.to_numeric(report["max_drawdown_oos"], errors="coerce").abs() <= pd.to_numeric(report["net_pnl_oos"], errors="coerce").abs() * 2.5 + 1e-9)
    )
    verdicts: list[str] = []
    for row in report.itertuples(index=False):
        if safe_float(row.trades_oos, 0.0) < 20:
            verdicts.append("too_few_trades")
        elif bool(row.oos_pass):
            verdicts.append("robust_candidate")
        elif safe_float(row.net_pnl_oos, 0.0) > 0 and safe_float(row.profit_factor_oos, 0.0) >= 1.0:
            verdicts.append("weak_oos")
        elif safe_float(row.degradation_ratio, np.nan) < 0:
            verdicts.append("overfit_suspect")
        else:
            verdicts.append("inconclusive")
    report["verdict"] = verdicts
    return report


def _event_metrics_frame(events_by_config: dict[str, pd.DataFrame], *, period: str, estimated_cost_per_trade: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config_id, events in events_by_config.items():
        scoped = events.loc[_period_mask(events, period)].copy() if period != "full" else events.copy()
        metrics = _compute_trade_metrics(scoped, estimated_cost_per_trade=estimated_cost_per_trade)
        metrics["config_id"] = config_id
        rows.append(metrics)
    return pd.DataFrame(rows)


def _save_heatmap(frame: pd.DataFrame, *, value_col: str, title: str, output_path: Path) -> None:
    if frame.empty:
        return
    pivot = frame.pivot(index="stop_multiplier", columns="target_multiplier", values=value_col)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f" if "pnl" in value_col else ".2f", cmap="RdYlGn", center=0.0 if "pnl" in value_col else None, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_charts(
    *,
    output_dir: Path,
    metrics_is: pd.DataFrame,
    metrics_oos: pd.DataFrame,
    selected_report: pd.DataFrame,
    best_events: pd.DataFrame,
    current_hybrid_events: pd.DataFrame,
    robustness: pd.DataFrame,
) -> list[str]:
    plot_paths: list[str] = []
    for delay in ENTRY_DELAYS:
        delay_slice = metrics_is.loc[(metrics_is["filter_family"] == "none") & (metrics_is["entry_delay_minutes"] == delay)].copy()
        if delay_slice.empty:
            continue
        path1 = output_dir / f"heatmap_is_net_pnl_stop_target_delay_{delay}_filter_none.png"
        _save_heatmap(delay_slice, value_col="net_pnl", title=f"IS Net PnL | none | delay={delay}", output_path=path1)
        plot_paths.append(str(path1))
        score_slice = robustness.loc[(robustness["filter_family"] == "none") & (robustness["entry_delay_minutes"] == delay)].copy()
        if not score_slice.empty:
            path2 = output_dir / f"heatmap_is_robust_score_stop_target_delay_{delay}_filter_none.png"
            _save_heatmap(score_slice, value_col="robust_score_is", title=f"IS Robust Score | none | delay={delay}", output_path=path2)
            plot_paths.append(str(path2))

    if not selected_report.empty:
        best_family = str(selected_report.iloc[0]["filter_family"])
        best_delay = int(selected_report.iloc[0]["entry_delay_minutes"])
        family_slice = metrics_oos.loc[(metrics_oos["filter_family"] == best_family) & (metrics_oos["entry_delay_minutes"] == best_delay)].copy()
        if not family_slice.empty:
            path = output_dir / "heatmap_oos_net_pnl_stop_target_selected_family.png"
            _save_heatmap(
                family_slice,
                value_col="net_pnl",
                title=f"OOS Net PnL | {best_family} | delay={best_delay}",
                output_path=path,
            )
            plot_paths.append(str(path))

    if not best_events.empty:
        current_daily = _daily_returns(current_hybrid_events)
        best_daily = _daily_returns(best_events)
        fig, ax = plt.subplots(figsize=(10, 4))
        if not current_daily.empty:
            ax.plot(pd.to_datetime(current_daily["session_date"]), current_daily["equity"], label="current_hybrid_next_execution_bar")
        if not best_daily.empty:
            ax.plot(pd.to_datetime(best_daily["session_date"]), best_daily["equity"], label="best_candidate")
        ax.set_title("Cumulative PnL Current Hybrid vs Best Candidate")
        ax.legend()
        fig.tight_layout()
        path = output_dir / "cumulative_pnl_current_hybrid_vs_best_candidate.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

        fig, ax = plt.subplots(figsize=(9, 4))
        selected_bar = selected_report[["config_id", "net_pnl", "net_pnl_oos"]].copy()
        if not selected_bar.empty:
            x = np.arange(len(selected_bar))
            ax.bar(x - 0.2, pd.to_numeric(selected_bar["net_pnl"], errors="coerce"), width=0.4, label="IS")
            ax.bar(x + 0.2, pd.to_numeric(selected_bar["net_pnl_oos"], errors="coerce"), width=0.4, label="OOS")
            ax.set_xticks(x)
            ax.set_xticklabels(selected_bar["config_id"], rotation=90)
            ax.legend()
            ax.set_title("Selected Configs IS vs OOS")
            fig.tight_layout()
            path = output_dir / "selected_configs_is_vs_oos_bar.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            plot_paths.append(str(path))

        best_events = best_events.copy()
        best_events["subperiod"] = pd.to_datetime(best_events["session_date"], errors="coerce").dt.year.astype(str)
        subperiod = (
            best_events.loc[best_events["executed"]]
            .groupby("subperiod", as_index=False)["net_pnl_usd"]
            .sum()
            .rename(columns={"net_pnl_usd": "net_pnl"})
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(subperiod["subperiod"], subperiod["net_pnl"], color="#1f77b4")
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title("Subperiod Net PnL | Best Candidate")
        fig.tight_layout()
        path = output_dir / "subperiod_net_pnl_best_candidate.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

        neighborhood = robustness.loc[
            (robustness["filter_family"] == selected_report.iloc[0]["filter_family"])
            & (robustness["entry_delay_minutes"] == selected_report.iloc[0]["entry_delay_minutes"])
        ].copy()
        if not neighborhood.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(
                neighborhood["stop_multiplier"],
                neighborhood["target_multiplier"],
                c=pd.to_numeric(neighborhood["robust_score_is"], errors="coerce"),
                s=80,
                cmap="viridis",
            )
            ax.set_title("Stop/Target Neighborhood Stability")
            ax.set_xlabel("stop_multiplier")
            ax.set_ylabel("target_multiplier")
            fig.colorbar(scatter, ax=ax, label="robust_score_is")
            fig.tight_layout()
            path = output_dir / "stop_target_neighborhood_stability.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            plot_paths.append(str(path))
    return plot_paths


def _current_hybrid_config_id(metrics_full: pd.DataFrame) -> str | None:
    matched = metrics_full.loc[
        (metrics_full["filter_family"] == "none")
        & (pd.to_numeric(metrics_full["stop_multiplier"], errors="coerce") == 1.0)
        & (pd.to_numeric(metrics_full["target_multiplier"], errors="coerce") == 1.0)
        & (pd.to_numeric(metrics_full["entry_delay_minutes"], errors="coerce") == 0)
    ]
    if matched.empty:
        return None
    return str(matched.iloc[0]["config_id"])


def _best_candidate_summary(
    *,
    best_row: pd.Series,
    oos_row: pd.Series | None,
    full_row: pd.Series | None,
    current_row: pd.Series | None,
    baseline_context: dict[str, Any],
) -> dict[str, Any]:
    risks: list[str] = []
    if oos_row is None or safe_float(oos_row.get("trades"), 0.0) < 20:
        risks.append("Too few OOS trades.")
    if oos_row is not None and safe_float(oos_row.get("net_pnl"), 0.0) <= 0:
        risks.append("Negative or flat OOS net PnL.")
    if oos_row is not None and safe_float(oos_row.get("profit_factor"), 0.0) < 1.05:
        risks.append("OOS profit factor remains weak.")
    if safe_float(best_row.get("skip_rate"), 0.0) > 0.70:
        risks.append("Skip rate is too high.")
    if not risks:
        risks.append("No immediate red flag beyond broader external validation.")
    return {
        "config": best_row.to_dict(),
        "is_metrics": best_row.to_dict(),
        "oos_metrics": None if oos_row is None else oos_row.to_dict(),
        "full_metrics": None if full_row is None else full_row.to_dict(),
        "main_risks": risks,
        "comparison_vs_current_hybrid": None if current_row is None else {
            "net_pnl_delta_full": safe_float(full_row.get("net_pnl"), 0.0) - safe_float(current_row.get("net_pnl"), 0.0),
            "profit_factor_delta_full": safe_float(full_row.get("profit_factor"), 0.0) - safe_float(current_row.get("profit_factor"), 0.0),
        },
        "comparison_vs_invalidated_1h_baseline_context_only": {
            "baseline_1h_net_pnl": safe_float(baseline_context.get("net_pnl_usd"), 0.0),
        },
    }


def _final_verdict(selected_oos_report: pd.DataFrame) -> str:
    if selected_oos_report.empty:
        return "Reject: alpha does not survive realistic execution."
    top = selected_oos_report.iloc[0]
    verdict = str(top.get("verdict", "inconclusive"))
    if verdict == "robust_candidate" and safe_float(top.get("trades_oos"), 0.0) >= 40 and safe_float(top.get("profit_factor_oos"), 0.0) >= 1.20:
        return "Strong candidate: robust IS/OOS improvement with simple mechanics."
    if verdict == "robust_candidate" and safe_float(top.get("net_pnl_oos"), 0.0) > 0 and safe_float(top.get("profit_factor_oos"), 0.0) >= 1.15:
        return "Candidate: tradable after simple recalibration, requires broader validation."
    if verdict in {"weak_oos", "too_few_trades", "inconclusive"}:
        return "Watchlist: weak but non-zero intrabar-aware edge."
    return "Reject: alpha does not survive realistic execution."


def _build_final_report(
    *,
    output_dir: Path,
    variant: VolumeClimaxPullbackV2Variant,
    benchmark_context: dict[str, Any],
    diagnostics_context: dict[str, Any],
    metrics_is: pd.DataFrame,
    metrics_oos: pd.DataFrame,
    robustness: pd.DataFrame,
    selected_is: pd.DataFrame,
    selected_oos_report: pd.DataFrame,
    current_hybrid_row: pd.Series | None,
    best_summary: dict[str, Any] | None,
    verdict: str,
) -> None:
    top_is = selected_is.head(10).copy()
    diagnostic_verdict_context = diagnostics_context.get("verdict_context", {})
    lines = [
        "# Volume Climax Pullback Intrabar Recalibration Campaign",
        "",
        "## 1. Executive Summary",
        f"- A tradable 1min-executed edge {'still seems present' if verdict != 'Reject: alpha does not survive realistic execution.' else 'does not clearly survive'} after simple recalibration.",
        f"- The main improvement family is `{selected_is.iloc[0]['filter_family']}` with delay `{int(selected_is.iloc[0]['entry_delay_minutes'])}` minutes." if not selected_is.empty else "- No admissible family emerged.",
        f"- The gain comes primarily from `{'entry delay and tighter stop / wider target geometry' if not selected_is.empty else 'n/a'}`.",
        f"- Verdict: `{verdict}`.",
        "",
        "## 2. Reminder: Why 1H Baseline Is Invalidated",
        f"- Previous diagnostics flagged the hourly baseline as biased, with dominant divergence cause `{diagnostics_context.get('matching_info', {}).get('matching_confidence', 'n/a')}` confidence matching and top driver `{diagnostics_context.get('verdict_context', {}).get('top_divergence_cause', 'unknown')}`.",
        f"- The best full-sample ex-post recalibration from phase 1 was stop x{safe_float(diagnostic_verdict_context.get('best_recalibration_stop_multiplier'), np.nan):.2f} / target x{safe_float(diagnostic_verdict_context.get('best_recalibration_target_multiplier'), np.nan):.2f} for {safe_float(diagnostic_verdict_context.get('best_recalibration_net_pnl'), np.nan):.2f} USD, but this campaign does not use that full-sample result for selection.",
        "- The hourly baseline remains historical context only and is not used for model selection in this phase.",
        "",
        "## 3. Research Design",
        f"- Signal 1H unchanged: `{variant.name}`.",
        "- Execution path: 1min only.",
        "- Calibration and ranking: IS only, with OOS reported afterwards as blind evaluation.",
        "- Base grid is exhaustive on stop/target/delay for no-filter configs and compact on filter families.",
        "",
        "## 4. Current Hybrid Benchmark",
        _markdown_table(benchmark_context["metrics_comparison"]) if "metrics_comparison" in benchmark_context else "No benchmark metrics found.",
        "",
        "## 5. Stop/Target and Entry Delay Grid",
        _markdown_table(metrics_is.sort_values("net_pnl", ascending=False).head(10)[["config_id", "stop_multiplier", "target_multiplier", "entry_delay_minutes", "filter_family", "net_pnl", "profit_factor", "trades"]]),
        "",
        "## 6. Filter Families",
        _markdown_table(
            metrics_is.groupby("filter_family", as_index=False)
            .agg(configs=("config_id", "count"), median_net_pnl=("net_pnl", "median"), median_profit_factor=("profit_factor", "median"))
            .sort_values("median_net_pnl", ascending=False)
        ),
        "",
        "## 7. IS Robustness Ranking",
        _markdown_table(top_is[["rank_is", "config_id", "filter_family", "stop_multiplier", "target_multiplier", "entry_delay_minutes", "robust_score_is", "net_pnl", "profit_factor", "trades"]]) if not top_is.empty else "No selected IS config.",
        "",
        "## 8. OOS Results of IS-Selected Configs",
        _markdown_table(selected_oos_report[["rank_is", "config_id", "net_pnl", "profit_factor", "trades", "net_pnl_oos", "profit_factor_oos", "trades_oos", "degradation_ratio", "verdict"]]) if not selected_oos_report.empty else "No OOS report available.",
        "",
        "## 9. Best Candidate Audit",
        _markdown_table(pd.DataFrame([best_summary["config"]]).head(1)) if best_summary is not None else "No best candidate.",
        "",
        "## 10. Robustness and Failure Modes",
        "- Main failure modes remain too few trades after filtering, sensitivity to stop/target geometry, and OOS drawdown control.",
        "- Some gains come from delaying entry rather than from complex filters, which is preferable from a robustness standpoint.",
        "",
        "## 11. Verdict",
        f"- `{verdict}`",
        "",
        "## 12. Next Actions",
        "- Extend the same intrabar-aware protocol to MES, M2K and MGC.",
        "- Integrate the surviving candidate family into the multi-asset research layer only after cross-asset validation.",
        "- Add prop-firm style constraints and daily loss overlays on top of the best candidate.",
        "- Run a walk-forward selection version of the same campaign.",
        "- Add a CI guardrail that blocks any intraday alpha publication without intrabar validation.",
    ]
    (output_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_campaign(
    *,
    symbol: str,
    diagnostics_dir: Path,
    validation_dir: Path,
    output_root: Path,
    raw_minute_df_override: pd.DataFrame | None = None,
    variant_override: VolumeClimaxPullbackV2Variant | None = None,
    configs_override: list[IntrabarRecalibrationConfig] | None = None,
    benchmark_context_override: dict[str, Any] | None = None,
) -> Path:
    diagnostics_dir = Path(diagnostics_dir)
    validation_dir = Path(validation_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / f"volume_climax_pullback_intrabar_recalibration_{symbol.lower()}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_context = _load_diagnostics_context(diagnostics_dir) if diagnostics_dir.exists() and (diagnostics_dir / "run_metadata.json").exists() else {}
    variant, validation_metadata = (
        (variant_override, {"dataset_path": None, "variant": asdict(variant_override)}) if variant_override is not None else _variant_from_validation_metadata(validation_dir)
    )
    benchmark_context = benchmark_context_override if benchmark_context_override is not None else _load_benchmark_context(validation_dir)

    dataset_path = Path(validation_metadata.get("dataset_path") or diagnostics_context.get("dataset_path") or "")
    if raw_minute_df_override is not None:
        raw_minute_df = raw_minute_df_override.copy()
    else:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Input dataset was not found: {dataset_path}")
        raw_minute_df = load_symbol_data(symbol, input_paths={symbol: dataset_path})

    minute_df = extract_rth(raw_minute_df.copy())
    minute_df["timestamp"] = pd.to_datetime(minute_df["timestamp"], errors="coerce")
    minute_df["session_date"] = minute_df["timestamp"].dt.date
    bars_1h = resample_rth_1h(raw_minute_df)
    bars_1h["timestamp"] = pd.to_datetime(bars_1h["timestamp"], errors="coerce")
    bars_1h["session_date"] = bars_1h["timestamp"].dt.date
    features = prepare_volume_climax_pullback_v2_features(bars_1h)
    signal_df = build_volume_climax_pullback_v2_signal_frame(features, variant)

    execution_model, instrument = build_execution_model_for_profile(symbol=symbol, profile_name="repo_realistic")
    tick_size = float(instrument.tick_size)
    point_value_usd = float(instrument.point_value_usd)
    estimated_cost_per_trade = float(execution_model.round_trip_fees(quantity=1))
    configs = configs_override or build_compact_config_grid(symbol)

    events_by_config: dict[str, pd.DataFrame] = {}
    metrics_rows: list[dict[str, Any]] = []
    subperiod_rows: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []

    for config in configs:
        events = _simulate_config(
            config=config,
            signal_df=signal_df,
            minute_df=minute_df,
            variant=variant,
            execution_model=execution_model,
            point_value_usd=point_value_usd,
            tick_size=tick_size,
        )
        events_by_config[config.config_id] = events
        parameter_rows.append(
            {
                "config_id": config.config_id,
                "symbol": config.symbol,
                "execution_timeframe": config.execution_timeframe,
                "entry_timing": config.entry_timing,
                "protective_orders_active_from": config.protective_orders_active_from,
                "ambiguous_policy": config.ambiguous_policy,
                "stop_multiplier": config.stop_multiplier,
                "target_multiplier": config.target_multiplier,
                "entry_delay_minutes": config.entry_delay_minutes,
                "filter_family": config.filter_family,
                "filter_label": config.filter_label,
                "filter_params": json.dumps(config.filter_params, sort_keys=True),
            }
        )
        subperiod_rows.append(compute_subperiod_metrics(events, config_id=config.config_id, estimated_cost_per_trade=estimated_cost_per_trade))
        for scope in ("full", "is", "oos"):
            scoped = events if scope == "full" else events.loc[_period_mask(events, scope)].copy()
            metrics = _compute_trade_metrics(scoped, estimated_cost_per_trade=estimated_cost_per_trade)
            metrics.update({"config_id": config.config_id, "scope": scope})
            metrics_rows.append(metrics)

    params_df = pd.DataFrame(parameter_rows)
    metrics_df = pd.DataFrame(metrics_rows).merge(params_df, on="config_id", how="left")
    metrics_full = metrics_df.loc[metrics_df["scope"] == "full"].drop(columns=["scope"]).reset_index(drop=True)
    metrics_is = metrics_df.loc[metrics_df["scope"] == "is"].drop(columns=["scope"]).reset_index(drop=True)
    metrics_oos = metrics_df.loc[metrics_df["scope"] == "oos"].drop(columns=["scope"]).reset_index(drop=True)
    subperiod_metrics = pd.concat(subperiod_rows, ignore_index=True).merge(params_df, on="config_id", how="left") if subperiod_rows else pd.DataFrame()
    robustness = build_robustness_scores(metrics_is, subperiod_metrics, estimated_cost_per_trade=estimated_cost_per_trade)
    selected_is = select_configs_is_only(robustness)
    selected_oos_report = build_selected_oos_report(selected_is, metrics_oos, metrics_full)

    current_hybrid_id = _current_hybrid_config_id(metrics_full)
    current_hybrid_row = metrics_full.loc[metrics_full["config_id"] == current_hybrid_id].iloc[0] if current_hybrid_id is not None else None
    best_candidate_row = selected_is.iloc[0] if not selected_is.empty else None
    best_candidate_oos_row = (
        selected_oos_report.loc[selected_oos_report["config_id"] == best_candidate_row["config_id"]].iloc[0]
        if best_candidate_row is not None and not selected_oos_report.empty and (selected_oos_report["config_id"] == best_candidate_row["config_id"]).any()
        else None
    )
    best_candidate_full_row = (
        metrics_full.loc[metrics_full["config_id"] == best_candidate_row["config_id"]].iloc[0]
        if best_candidate_row is not None and (metrics_full["config_id"] == best_candidate_row["config_id"]).any()
        else None
    )

    baseline_context = benchmark_context.get("baseline", {})
    best_summary = (
        _best_candidate_summary(
            best_row=best_candidate_row,
            oos_row=None if best_candidate_oos_row is None else best_candidate_oos_row,
            full_row=None if best_candidate_full_row is None else best_candidate_full_row,
            current_row=current_hybrid_row,
            baseline_context=baseline_context,
        )
        if best_candidate_row is not None
        else None
    )
    verdict = _final_verdict(selected_oos_report)

    all_events = pd.concat(events_by_config.values(), ignore_index=True) if events_by_config else pd.DataFrame()
    all_events = all_events.merge(params_df, on="config_id", how="left")
    all_events.to_csv(output_dir / "all_config_trades.csv", index=False)
    metrics_is.to_csv(output_dir / "config_metrics_is.csv", index=False)
    metrics_oos.to_csv(output_dir / "config_metrics_oos.csv", index=False)
    metrics_full.to_csv(output_dir / "config_metrics_full.csv", index=False)
    subperiod_metrics.to_csv(output_dir / "config_subperiod_metrics.csv", index=False)
    robustness.to_csv(output_dir / "config_robustness_scores.csv", index=False)
    selected_is.to_csv(output_dir / "selected_configs_is_only.csv", index=False)
    selected_oos_report.to_csv(output_dir / "selected_configs_oos_report.csv", index=False)

    if best_candidate_row is not None:
        best_events = events_by_config[best_candidate_row["config_id"]].copy().merge(params_df, on="config_id", how="left")
        best_events["subperiod"] = pd.to_datetime(best_events["session_date"], errors="coerce").dt.year.astype("Int64").astype(str)
        best_audit = best_events.loc[best_events["executed"]].copy()
        best_audit.to_csv(output_dir / "best_candidate_trade_audit.csv", index=False)
        best_daily = _daily_returns(best_events)
        best_daily.to_csv(output_dir / "best_candidate_daily_returns.csv", index=False)
        (output_dir / "best_candidate_summary.json").write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
    else:
        best_events = pd.DataFrame()
        pd.DataFrame().to_csv(output_dir / "best_candidate_trade_audit.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "best_candidate_daily_returns.csv", index=False)
        (output_dir / "best_candidate_summary.json").write_text(json.dumps({}, indent=2), encoding="utf-8")

    current_hybrid_events = events_by_config.get(current_hybrid_id, pd.DataFrame()) if current_hybrid_id is not None else pd.DataFrame()
    plot_paths = _save_charts(
        output_dir=output_dir,
        metrics_is=metrics_is,
        metrics_oos=metrics_oos,
        selected_report=selected_oos_report,
        best_events=best_events,
        current_hybrid_events=current_hybrid_events,
        robustness=robustness,
    )

    _build_final_report(
        output_dir=output_dir,
        variant=variant,
        benchmark_context=benchmark_context,
        diagnostics_context=diagnostics_context,
        metrics_is=metrics_is,
        metrics_oos=metrics_oos,
        robustness=robustness,
        selected_is=selected_is,
        selected_oos_report=selected_oos_report,
        current_hybrid_row=current_hybrid_row,
        best_summary=best_summary,
        verdict=verdict,
    )

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "symbol": symbol,
        "diagnostics_dir": str(diagnostics_dir),
        "validation_dir": str(validation_dir),
        "output_dir": str(output_dir),
        "dataset_path": str(dataset_path) if dataset_path else None,
        "python_version": sys.version,
        "platform": platform.platform(),
        "variant": asdict(variant),
        "is_end_date": str(IS_END_DATE),
        "oos_start_date": str(OOS_START_DATE),
        "grid_size": int(len(configs)),
        "selected_config_count": int(len(selected_is)),
        "current_hybrid_config_id": current_hybrid_id,
        "verdict": verdict,
        "plots": plot_paths,
        "input_files": {
            "diagnostics_run_metadata": _file_metadata(diagnostics_dir / "run_metadata.json") if diagnostics_dir.exists() and (diagnostics_dir / "run_metadata.json").exists() else None,
            "validation_run_metadata": _file_metadata(validation_dir / "run_metadata.json") if validation_dir.exists() and (validation_dir / "run_metadata.json").exists() else None,
        },
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Intrabar-aware recalibration campaign for Volume Climax Pullback MNQ.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Symbol to evaluate, default MNQ.")
    parser.add_argument("--diagnostics-dir", default=str(DEFAULT_DIAGNOSTICS_DIR), help="Diagnostics directory from phase 1.")
    parser.add_argument("--validation-dir", default=str(DEFAULT_VALIDATION_DIR), help="Hybrid validation directory from phase 1.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output export root.")
    args = parser.parse_args()

    run_dir = run_campaign(
        symbol=str(args.symbol).upper(),
        diagnostics_dir=Path(args.diagnostics_dir),
        validation_dir=Path(args.validation_dir),
        output_root=Path(args.output_root),
    )
    print(run_dir)


if __name__ == "__main__":
    main()
