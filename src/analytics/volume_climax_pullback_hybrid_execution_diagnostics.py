"""Diagnostics for baseline 1H vs hybrid 1min execution divergences."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
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
from src.config.settings import DEFAULT_TIMEZONE, get_instrument_spec
from src.data.session import extract_rth
from src.engine.volume_climax_pullback_v2_backtester import run_volume_climax_pullback_v2_backtest
from src.engine.vwap_backtester import build_execution_model_for_profile
from src.strategy.volume_climax_pullback_v2 import (
    VolumeClimaxPullbackV2Variant,
    build_volume_climax_pullback_v2_signal_frame,
    prepare_volume_climax_pullback_v2_features,
)

DEFAULT_COMPARISON_DIR = Path("export/volume_climax_pullback_hybrid_execution_validation_mnq_20260519_220109")
DEFAULT_OUTPUT_ROOT = Path("export")
CSV_INPUTS = (
    "trades_baseline_1h.csv",
    "trades_hybrid_after_entry_fill.csv",
    "trades_hybrid_next_execution_bar.csv",
    "trade_comparison.csv",
    "metrics_comparison.csv",
)
AMBIGUOUS_POLICY_EXECUTED = "stop_first"
TICK_BUCKETS = [-np.inf, -40, -20, -10, -5, 5, 10, 20, 40, np.inf]
TICK_BUCKET_LABELS = ["<=-40", "-40:-20", "-20:-10", "-10:-5", "-5:5", "5:10", "10:20", "20:40", ">=40"]


def parse_datetime_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Parse likely datetime columns without touching numeric columns."""
    out = df.copy()
    for column in out.columns:
        lower = str(column).strip().lower()
        if not any(token in lower for token in ("time", "date", "timestamp", "_at")):
            continue
        if lower == "session_date":
            parsed_date = pd.to_datetime(out[column], errors="coerce")
            if parsed_date.notna().sum() > 0:
                out[column] = parsed_date.dt.date
            continue

        parsed = out[column].apply(_coerce_timestamp_like)
        if parsed.notna().sum() > 0:
            out[column] = parsed
    return out


def _coerce_timestamp_like(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return pd.NaT
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if timestamp.tzinfo is None:
        try:
            return timestamp.tz_localize(DEFAULT_TIMEZONE)
        except (TypeError, ValueError):
            return timestamp
    try:
        return timestamp.tz_convert(DEFAULT_TIMEZONE)
    except TypeError:
        return timestamp


def normalize_trade_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical identifiers used by the diagnostic matcher."""
    out = parse_datetime_cols(df)
    if "trade_id" not in out.columns:
        out["trade_id"] = np.arange(1, len(out) + 1, dtype=int)
    else:
        fallback_ids = pd.Series(np.arange(1, len(out) + 1, dtype=int), index=out.index)
        parsed_ids = pd.to_numeric(out["trade_id"], errors="coerce")
        out["trade_id"] = parsed_ids.where(parsed_ids.notna(), fallback_ids).astype(int)

    if "direction" in out.columns:
        out["direction"] = out["direction"].astype(str).str.lower()
    else:
        out["direction"] = "unknown"

    out["signal_time"] = pd.NaT
    for column in ("setup_bar_label_time", "entry_signal_time", "signal_actionable_time", "entry_time"):
        if column in out.columns:
            mask = out["signal_time"].isna() & out[column].notna()
            out.loc[mask, "signal_time"] = out.loc[mask, column]

    out["actionable_time"] = pd.NaT
    for column in ("signal_actionable_time", "setup_bar_close_time", "entry_time"):
        if column in out.columns:
            mask = out["actionable_time"].isna() & out[column].notna()
            out.loc[mask, "actionable_time"] = out.loc[mask, column]

    out["setup_id"] = pd.Series(index=out.index, dtype=object)
    signal_mask = out["signal_time"].notna()
    out.loc[signal_mask, "setup_id"] = (
        out.loc[signal_mask, "direction"].astype(str)
        + "|"
        + out.loc[signal_mask, "signal_time"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    )
    actionable_mask = out["setup_id"].isna() & out["actionable_time"].notna()
    out.loc[actionable_mask, "setup_id"] = (
        out.loc[actionable_mask, "direction"].astype(str)
        + "|actionable|"
        + out.loc[actionable_mask, "actionable_time"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    )
    index_mask = out["setup_id"].isna()
    out.loc[index_mask, "setup_id"] = (
        out.loc[index_mask, "direction"].astype(str)
        + "|index|"
        + out.loc[index_mask].index.astype(str)
    )
    return out


def infer_trade_key(df: pd.DataFrame) -> dict[str, Any]:
    """Infer the best deterministic key available for matching trades."""
    working = normalize_trade_id_columns(df)
    total = max(len(working), 1)
    if "setup_id" in working.columns and working["setup_id"].notna().all() and working["setup_id"].nunique() == len(working):
        source = "signal_time" if working["signal_time"].notna().sum() == len(working) else "actionable_time_or_index"
        confidence = "high" if source == "signal_time" else "medium"
        return {"key_columns": ["setup_id"], "method": source, "confidence": confidence}
    if working["signal_time"].notna().sum() / total >= 0.8:
        return {"key_columns": ["direction", "signal_time"], "method": "direction_plus_signal_time", "confidence": "medium"}
    if working["actionable_time"].notna().sum() / total >= 0.8:
        return {"key_columns": ["direction", "actionable_time"], "method": "direction_plus_actionable_time", "confidence": "medium"}
    if "entry_time" in working.columns and working["entry_time"].notna().sum() / total >= 0.8:
        return {"key_columns": ["direction", "entry_time"], "method": "direction_plus_entry_time", "confidence": "low"}
    return {"key_columns": ["trade_id"], "method": "ordered_index_fallback", "confidence": "low"}


def compute_first_touch(
    path_df: pd.DataFrame,
    *,
    direction: str,
    stop_price: float | None,
    target_price: float | None,
) -> dict[str, Any]:
    """Detect the first protective-order touch on a 1min path."""
    if path_df.empty or not np.isfinite(safe_float(stop_price, np.nan)) or not np.isfinite(safe_float(target_price, np.nan)):
        return {
            "first_touch": "none",
            "first_touch_time": pd.NaT,
            "first_touch_index": None,
            "stop_touch_time": pd.NaT,
            "target_touch_time": pd.NaT,
            "ambiguous_policy_applied": None,
        }

    direction_label = str(direction).lower()
    stop_value = float(stop_price)
    target_value = float(target_price)
    stop_touch_time = pd.NaT
    target_touch_time = pd.NaT

    for idx, row in path_df.reset_index(drop=True).iterrows():
        high = safe_float(row.get("high"), np.nan)
        low = safe_float(row.get("low"), np.nan)
        timestamp = pd.Timestamp(row["timestamp"])
        if direction_label == "long":
            stop_hit = low <= stop_value
            target_hit = high >= target_value
        else:
            stop_hit = high >= stop_value
            target_hit = low <= target_value

        if stop_hit and pd.isna(stop_touch_time):
            stop_touch_time = timestamp
        if target_hit and pd.isna(target_touch_time):
            target_touch_time = timestamp

        if stop_hit and target_hit:
            return {
                "first_touch": "both_same_minute",
                "first_touch_time": timestamp,
                "first_touch_index": int(idx),
                "stop_touch_time": stop_touch_time,
                "target_touch_time": target_touch_time,
                "ambiguous_policy_applied": AMBIGUOUS_POLICY_EXECUTED,
            }
        if stop_hit:
            return {
                "first_touch": "stop",
                "first_touch_time": timestamp,
                "first_touch_index": int(idx),
                "stop_touch_time": stop_touch_time,
                "target_touch_time": target_touch_time,
                "ambiguous_policy_applied": None,
            }
        if target_hit:
            return {
                "first_touch": "target",
                "first_touch_time": timestamp,
                "first_touch_index": int(idx),
                "stop_touch_time": stop_touch_time,
                "target_touch_time": target_touch_time,
                "ambiguous_policy_applied": None,
            }

    return {
        "first_touch": "none",
        "first_touch_time": pd.NaT,
        "first_touch_index": None,
        "stop_touch_time": stop_touch_time,
        "target_touch_time": target_touch_time,
        "ambiguous_policy_applied": None,
    }


def classify_divergence(row: pd.Series) -> dict[str, Any]:
    """Assign a primary divergence taxonomy for baseline vs hybrid-after."""
    baseline_pnl = safe_float(row.get("baseline_pnl"), np.nan)
    hybrid_pnl = safe_float(row.get("hybrid_after_entry_fill_pnl"), np.nan)
    matched_status = str(row.get("matched_status", "matched"))
    baseline_winner = np.isfinite(baseline_pnl) and baseline_pnl > 0
    hybrid_winner = np.isfinite(hybrid_pnl) and hybrid_pnl > 0
    first_touch = str(row.get("first_touch", "none"))
    baseline_exit_reason = str(row.get("baseline_exit_reason", ""))
    hybrid_exit_reason = str(row.get("hybrid_after_exit_reason", ""))
    entry_delta_ticks = safe_float(row.get("entry_price_delta_ticks"), 0.0)
    exit_reason_changed = baseline_exit_reason != hybrid_exit_reason
    winner_flipped = baseline_winner != hybrid_winner and np.isfinite(baseline_pnl) and np.isfinite(hybrid_pnl)

    divergence_type = "same_outcome_minor_delta"
    divergence_subtype = "unknown"

    if matched_status == "baseline_only":
        divergence_type = "baseline_trade_missing_in_hybrid"
        divergence_subtype = "missing_data_or_alignment"
    elif matched_status == "hybrid_only":
        divergence_type = "hybrid_trade_missing_in_baseline"
        divergence_subtype = "missing_data_or_alignment"
    elif matched_status == "unmatched_uncertain":
        divergence_type = "unmatched_uncertain"
        divergence_subtype = "missing_data_or_alignment"
    elif winner_flipped and baseline_winner and not hybrid_winner:
        divergence_type = "winner_to_loser"
        if first_touch == "both_same_minute":
            divergence_subtype = "stop_before_target_inside_baseline_hour"
        elif first_touch == "stop":
            divergence_subtype = "stop_reached_on_1min_path_only"
        elif entry_delta_ticks < 0:
            divergence_subtype = "worse_entry_price"
        else:
            divergence_subtype = "target_not_reached_on_1min_path"
    elif winner_flipped and (not baseline_winner) and hybrid_winner:
        divergence_type = "loser_to_winner"
        if first_touch == "target":
            divergence_subtype = "target_not_reached_on_1min_path"
        elif entry_delta_ticks > 0:
            divergence_subtype = "better_entry_price"
    elif first_touch == "both_same_minute":
        divergence_type = "ambiguous_same_minute_stop_first"
        divergence_subtype = "stop_before_target_inside_baseline_hour"
    elif first_touch == "stop" and baseline_exit_reason.startswith("target"):
        divergence_type = "intrabar_stop_before_target"
        divergence_subtype = "stop_before_target_inside_baseline_hour"
    elif first_touch == "target" and baseline_exit_reason.startswith("stop"):
        divergence_type = "intrabar_target_before_stop"
        divergence_subtype = "target_not_reached_on_1min_path"
    elif hybrid_exit_reason.startswith("time_stop") and not baseline_exit_reason.startswith("time_stop"):
        divergence_type = "time_stop_changed"
        divergence_subtype = "time_stop_conversion_effect"
    elif hybrid_exit_reason.startswith("eod_flat") and not baseline_exit_reason.startswith("eod_flat"):
        divergence_type = "eod_flat_changed"
        divergence_subtype = "session_boundary_effect"
    elif exit_reason_changed:
        divergence_type = "exit_reason_changed"
        if row.get("baseline_exit_time") == row.get("hybrid_after_exit_time"):
            divergence_subtype = "same_exit_different_entry"
        else:
            divergence_subtype = "same_entry_different_exit"
    elif entry_delta_ticks < -2:
        divergence_type = "entry_timing_degradation"
        divergence_subtype = "worse_entry_price"
    elif entry_delta_ticks > 2:
        divergence_type = "same_outcome_minor_delta"
        divergence_subtype = "better_entry_price"

    return {
        "divergence_type": divergence_type,
        "divergence_subtype": divergence_subtype,
        "is_baseline_winner": baseline_winner,
        "is_hybrid_winner": hybrid_winner,
        "winner_flipped": winner_flipped,
        "exit_reason_changed": exit_reason_changed,
    }


def build_pnl_bridge(diagnostic_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate baseline-to-hybrid delta into a simple causal bridge."""
    baseline_net = float(pd.to_numeric(diagnostic_df.get("baseline_pnl"), errors="coerce").fillna(0.0).sum())
    hybrid_net = float(pd.to_numeric(diagnostic_df.get("hybrid_after_entry_fill_pnl"), errors="coerce").fillna(0.0).sum())
    deltas = pd.to_numeric(diagnostic_df.get("delta_pnl_after_entry_fill"), errors="coerce").fillna(0.0)
    categories = diagnostic_df.get("primary_pnl_driver")
    bridge_rows = [
        {"bridge_component": "baseline_net_pnl", "amount": baseline_net},
    ]
    for driver in (
        "removed_or_missing_trades_effect",
        "entry_price_effect",
        "stop_before_target_effect",
        "target_not_reached_effect",
        "time_stop_effect",
        "eod_effect",
        "ambiguous_stop_first_effect",
        "residual_unexplained",
    ):
        amount = float(deltas.loc[categories.astype(str) == driver].sum()) if categories is not None else 0.0
        bridge_rows.append({"bridge_component": driver, "amount": amount})
    running_total = baseline_net + sum(row["amount"] for row in bridge_rows[1:])
    residual_fix = hybrid_net - running_total
    for row in bridge_rows:
        if row["bridge_component"] == "residual_unexplained":
            row["amount"] += residual_fix
            break
    bridge_rows.append({"bridge_component": "hybrid_net_pnl", "amount": hybrid_net})
    return pd.DataFrame(bridge_rows)


def run_recalibration_grid(
    trades_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    *,
    tick_size: float,
    point_value_usd: float,
    stop_multipliers: list[float] | tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0),
    target_multipliers: list[float] | tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0),
) -> pd.DataFrame:
    """Replay hybrid trades ex-post with a stop/target multiplier grid."""
    rows: list[dict[str, Any]] = []
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "stop_multiplier",
                "target_multiplier",
                "trades",
                "net_pnl",
                "winrate",
                "avg_trade",
                "profit_factor",
                "max_drawdown_if_reconstructable",
                "median_holding_minutes",
                "comment",
            ]
        )

    minute_lookup = minute_df.copy()
    minute_lookup["timestamp"] = pd.to_datetime(minute_lookup["timestamp"], errors="coerce")
    minute_lookup["session_date"] = pd.to_datetime(minute_lookup["session_date"], errors="coerce").dt.date
    grouped_minutes = {
        session_date: frame.sort_values("timestamp").reset_index(drop=True)
        for session_date, frame in minute_lookup.groupby("session_date", sort=True)
    }

    for stop_mult in stop_multipliers:
        for target_mult in target_multipliers:
            trade_results: list[dict[str, Any]] = []
            for trade in trades_df.itertuples(index=False):
                session_date = pd.Timestamp(trade.session_date).date() if not pd.isna(trade.session_date) else pd.Timestamp(trade.entry_time).date()
                session_minutes = grouped_minutes.get(session_date)
                if session_minutes is None or session_minutes.empty:
                    continue
                entry_time = pd.Timestamp(trade.entry_time)
                path = session_minutes.loc[session_minutes["timestamp"] >= entry_time].copy()
                if path.empty:
                    continue

                direction = str(trade.direction).lower()
                direction_sign = 1 if direction == "long" else -1
                entry_price = float(trade.entry_price)
                stop_distance = abs(entry_price - float(trade.initial_stop_price if pd.notna(trade.initial_stop_price) else trade.stop_price))
                target_distance = abs(float(trade.target_price) - entry_price)
                stop_price = entry_price - direction_sign * stop_distance * float(stop_mult)
                target_price = entry_price + direction_sign * target_distance * float(target_mult)
                time_stop_at = pd.Timestamp(trade.time_stop_at) if pd.notna(trade.time_stop_at) else pd.Timestamp(trade.exit_time)
                exit_price = safe_float(path.iloc[-1]["close"], entry_price)
                exit_time = pd.Timestamp(path.iloc[-1]["timestamp"])
                exit_reason = "eod_flat_1m"
                holding_minutes = max(int((exit_time - entry_time) / pd.Timedelta(minutes=1)), 0)

                for _, bar in path.iterrows():
                    timestamp = pd.Timestamp(bar["timestamp"])
                    high = safe_float(bar.get("high"), np.nan)
                    low = safe_float(bar.get("low"), np.nan)
                    close = safe_float(bar.get("close"), np.nan)
                    if direction_sign == 1:
                        stop_hit = low <= stop_price
                        target_hit = high >= target_price
                    else:
                        stop_hit = high >= stop_price
                        target_hit = low <= target_price

                    if stop_hit and target_hit:
                        exit_price = stop_price
                        exit_time = timestamp
                        exit_reason = "stop_ambiguous_first_1m"
                        holding_minutes = max(int((exit_time - entry_time) / pd.Timedelta(minutes=1)), 0)
                        break
                    if stop_hit:
                        exit_price = stop_price
                        exit_time = timestamp
                        exit_reason = "stop_1m"
                        holding_minutes = max(int((exit_time - entry_time) / pd.Timedelta(minutes=1)), 0)
                        break
                    if target_hit:
                        exit_price = target_price
                        exit_time = timestamp
                        exit_reason = "target_1m"
                        holding_minutes = max(int((exit_time - entry_time) / pd.Timedelta(minutes=1)), 0)
                        break
                    if timestamp >= time_stop_at:
                        exit_price = close
                        exit_time = timestamp
                        exit_reason = "time_stop_1m"
                        holding_minutes = max(int((exit_time - entry_time) / pd.Timedelta(minutes=1)), 0)
                        break

                pnl_points = (float(exit_price) - entry_price) * direction_sign
                gross = pnl_points * float(point_value_usd) * float(trade.quantity)
                fees = safe_float(getattr(trade, "fees", np.nan), 0.0)
                net = gross - fees
                trade_results.append(
                    {
                        "net_pnl_usd": net,
                        "holding_minutes": holding_minutes,
                        "won": net > 0,
                        "exit_reason": exit_reason,
                    }
                )

            if not trade_results:
                continue

            result_df = pd.DataFrame(trade_results)
            wins = result_df.loc[result_df["net_pnl_usd"] > 0, "net_pnl_usd"].sum()
            losses = result_df.loc[result_df["net_pnl_usd"] < 0, "net_pnl_usd"].sum()
            cumulative = result_df["net_pnl_usd"].cumsum()
            drawdown = cumulative - cumulative.cummax()
            rows.append(
                {
                    "stop_multiplier": float(stop_mult),
                    "target_multiplier": float(target_mult),
                    "trades": int(len(result_df)),
                    "net_pnl": float(result_df["net_pnl_usd"].sum()),
                    "winrate": float((result_df["net_pnl_usd"] > 0).mean()),
                    "avg_trade": float(result_df["net_pnl_usd"].mean()),
                    "profit_factor": float(wins / abs(losses)) if losses < 0 else np.inf,
                    "max_drawdown_if_reconstructable": float(drawdown.min()) if not drawdown.empty else 0.0,
                    "median_holding_minutes": float(result_df["holding_minutes"].median()),
                    "comment": "Ex-post 1min replay with fixed alpha and stop-first ambiguity convention.",
                }
            )
    return pd.DataFrame(rows)


def _file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "sha256": digest,
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


def _to_ticks(price_delta: float | None, tick_size: float) -> float:
    if not np.isfinite(safe_float(price_delta, np.nan)) or tick_size <= 0:
        return float("nan")
    return float(price_delta / tick_size)


def _generic_level_touch_time(path_df: pd.DataFrame, level: float, start_time: pd.Timestamp | None = None) -> pd.Timestamp | pd.NaT:
    if path_df.empty or not np.isfinite(level):
        return pd.NaT
    scoped = path_df if start_time is None else path_df.loc[path_df["timestamp"] >= pd.Timestamp(start_time)].copy()
    for _, row in scoped.iterrows():
        low = safe_float(row.get("low"), np.nan)
        high = safe_float(row.get("high"), np.nan)
        if low <= level <= high:
            return pd.Timestamp(row["timestamp"])
    return pd.NaT


def _path_window_for_trade(
    trade_row: pd.Series,
    minute_by_session: dict[Any, pd.DataFrame],
    *,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    entry_time_value = trade_row.get("entry_time")
    if pd.isna(entry_time_value):
        entry_time_value = trade_row.get("hybrid_entry_time")
    if pd.isna(entry_time_value):
        return pd.DataFrame()
    entry_time = pd.Timestamp(entry_time_value)
    session_date = pd.Timestamp(trade_row["session_date"]).date() if pd.notna(trade_row.get("session_date")) else entry_time.date()
    session_df = minute_by_session.get(session_date)
    if session_df is None or session_df.empty:
        return pd.DataFrame()
    return session_df.loc[(session_df["timestamp"] >= entry_time) & (session_df["timestamp"] <= end_time)].copy().reset_index(drop=True)


def _primary_pnl_driver(row: pd.Series) -> str:
    divergence_type = str(row.get("divergence_type", ""))
    divergence_subtype = str(row.get("divergence_subtype", ""))
    if divergence_type in {"baseline_trade_missing_in_hybrid", "hybrid_trade_missing_in_baseline", "unmatched_uncertain"}:
        return "removed_or_missing_trades_effect"
    if divergence_type == "entry_timing_degradation" or divergence_subtype in {"worse_entry_price", "better_entry_price"}:
        return "entry_price_effect"
    if divergence_type == "intrabar_stop_before_target" or divergence_subtype == "stop_reached_on_1min_path_only":
        return "stop_before_target_effect"
    if divergence_type == "intrabar_target_before_stop" or divergence_subtype == "target_not_reached_on_1min_path":
        return "target_not_reached_effect"
    if divergence_type == "time_stop_changed" or divergence_subtype == "time_stop_conversion_effect":
        return "time_stop_effect"
    if divergence_type == "eod_flat_changed" or divergence_subtype == "session_boundary_effect":
        return "eod_effect"
    if divergence_type == "ambiguous_same_minute_stop_first":
        return "ambiguous_stop_first_effect"
    return "residual_unexplained"


def _pnl_delta_bucket(delta: float) -> str:
    if not np.isfinite(delta):
        return "nan"
    return str(pd.cut(pd.Series([delta]), bins=TICK_BUCKETS, labels=TICK_BUCKET_LABELS, include_lowest=True).iloc[0])


def _build_diagnostic_trade_level(
    baseline_trades: pd.DataFrame,
    hybrid_after_trades: pd.DataFrame,
    hybrid_next_trades: pd.DataFrame,
    *,
    tick_size: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline = normalize_trade_id_columns(baseline_trades)
    hybrid_after = normalize_trade_id_columns(hybrid_after_trades)
    hybrid_next = normalize_trade_id_columns(hybrid_next_trades)

    baseline_key = infer_trade_key(baseline)
    after_key = infer_trade_key(hybrid_after)
    next_key = infer_trade_key(hybrid_next)
    matching_confidence = "high" if baseline_key["confidence"] == after_key["confidence"] == next_key["confidence"] == "high" else "mixed"

    baseline_cols = {
        "setup_id": "setup_id",
        "direction": "direction",
        "signal_time": "signal_time",
        "entry_time": "baseline_entry_time",
        "entry_price": "baseline_entry_price",
        "exit_time": "baseline_exit_time",
        "exit_price": "baseline_exit_price",
        "exit_reason": "baseline_exit_reason",
        "net_pnl_usd": "baseline_pnl",
        "bars_held": "bars_to_exit_baseline_1h",
        "stop_price": "baseline_stop_price",
        "target_price": "baseline_target_price",
    }
    after_cols = {
        "setup_id": "setup_id",
        "direction": "direction",
        "signal_time": "signal_time_after",
        "entry_time": "hybrid_entry_time",
        "entry_price": "hybrid_entry_price",
        "exit_time": "hybrid_after_exit_time",
        "exit_price": "hybrid_after_exit_price",
        "exit_reason": "hybrid_after_exit_reason",
        "net_pnl_usd": "hybrid_after_entry_fill_pnl",
        "minutes_held": "minutes_to_exit_hybrid",
        "stop_price": "hybrid_stop_price",
        "target_price": "hybrid_target_price",
        "time_stop_at": "hybrid_time_stop_at",
        "session_date": "session_date",
        "quantity": "quantity",
        "fees": "fees",
        "initial_stop_price": "initial_stop_price",
    }
    next_cols = {
        "setup_id": "setup_id",
        "direction": "direction",
        "entry_time": "hybrid_next_entry_time",
        "entry_price": "hybrid_next_entry_price",
        "exit_time": "hybrid_next_exit_time",
        "exit_price": "hybrid_next_exit_price",
        "exit_reason": "hybrid_next_exit_reason",
        "net_pnl_usd": "hybrid_next_execution_bar_pnl",
    }

    baseline_view = baseline[[column for column in baseline_cols if column in baseline.columns]].rename(columns=baseline_cols)
    after_view = hybrid_after[[column for column in after_cols if column in hybrid_after.columns]].rename(columns=after_cols)
    next_view = hybrid_next[[column for column in next_cols if column in hybrid_next.columns]].rename(columns=next_cols)

    diagnostic = baseline_view.merge(after_view, on=["setup_id", "direction"], how="outer")
    diagnostic = diagnostic.merge(next_view, on=["setup_id", "direction"], how="outer")
    diagnostic["signal_time"] = diagnostic["signal_time"].combine_first(diagnostic.get("signal_time_after"))
    if "signal_time_after" in diagnostic.columns:
        diagnostic = diagnostic.drop(columns=["signal_time_after"])

    diagnostic["matched_status"] = "matched"
    baseline_missing = diagnostic["baseline_entry_time"].isna()
    after_missing = diagnostic["hybrid_entry_time"].isna()
    diagnostic.loc[baseline_missing & (~after_missing), "matched_status"] = "hybrid_only"
    diagnostic.loc[(~baseline_missing) & after_missing, "matched_status"] = "baseline_only"
    uncertain_mask = diagnostic["setup_id"].astype(str).str.contains("|index|", regex=False)
    diagnostic.loc[uncertain_mask, "matched_status"] = "unmatched_uncertain"

    diagnostic["delta_pnl_after_entry_fill"] = pd.to_numeric(diagnostic.get("hybrid_after_entry_fill_pnl"), errors="coerce").fillna(0.0) - pd.to_numeric(
        diagnostic.get("baseline_pnl"), errors="coerce"
    ).fillna(0.0)
    diagnostic["delta_pnl_next_execution_bar"] = pd.to_numeric(
        diagnostic.get("hybrid_next_execution_bar_pnl"), errors="coerce"
    ).fillna(0.0) - pd.to_numeric(diagnostic.get("baseline_pnl"), errors="coerce").fillna(0.0)

    direction_sign = diagnostic["direction"].map({"long": 1.0, "short": -1.0}).fillna(1.0)
    diagnostic["entry_price_delta_ticks"] = (
        (pd.to_numeric(diagnostic.get("baseline_entry_price"), errors="coerce") - pd.to_numeric(diagnostic.get("hybrid_entry_price"), errors="coerce"))
        * direction_sign
        / float(tick_size)
    )
    diagnostic["exit_price_delta_ticks"] = (
        (pd.to_numeric(diagnostic.get("hybrid_after_exit_price"), errors="coerce") - pd.to_numeric(diagnostic.get("baseline_exit_price"), errors="coerce"))
        * direction_sign
        / float(tick_size)
    )
    diagnostic["pnl_delta_bucket"] = diagnostic["delta_pnl_after_entry_fill"].apply(_pnl_delta_bucket)

    classification = diagnostic.apply(classify_divergence, axis=1, result_type="expand")
    diagnostic = pd.concat([diagnostic, classification], axis=1)
    diagnostic["primary_pnl_driver"] = diagnostic.apply(_primary_pnl_driver, axis=1)
    diagnostic["is_baseline_winner"] = diagnostic["is_baseline_winner"].fillna(False)
    diagnostic["is_hybrid_winner"] = diagnostic["is_hybrid_winner"].fillna(False)
    diagnostic = diagnostic.sort_values(["signal_time", "direction", "setup_id"]).reset_index(drop=True)

    matching_info = {
        "baseline_key": baseline_key,
        "hybrid_after_key": after_key,
        "hybrid_next_key": next_key,
        "matching_confidence": matching_confidence,
        "uncertain_matches": int((diagnostic["matched_status"] == "unmatched_uncertain").sum()),
        "baseline_only": int((diagnostic["matched_status"] == "baseline_only").sum()),
        "hybrid_only": int((diagnostic["matched_status"] == "hybrid_only").sum()),
    }
    return diagnostic, matching_info


def _build_path_reconstruction(
    diagnostic_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    *,
    tick_size: float,
    point_value_usd: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    minute_working = parse_datetime_cols(minute_df)
    minute_working["session_date"] = pd.to_datetime(minute_working["session_date"], errors="coerce").dt.date
    minute_working = minute_working.sort_values("timestamp").reset_index(drop=True)
    minute_by_session = {
        session_date: frame.sort_values("timestamp").reset_index(drop=True)
        for session_date, frame in minute_working.groupby("session_date", sort=True)
    }
    path_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    output_columns = [
        "setup_id",
        "signal_time",
        "direction",
        "entry_time",
        "entry_price",
        "stop_price",
        "target_price",
        "evaluation_end_time",
        "mfe_ticks_1m",
        "mae_ticks_1m",
        "mfe_usd_1m",
        "mae_usd_1m",
        "first_touch",
        "first_touch_time",
        "first_touch_minutes_after_entry",
        "baseline_exit_level_touched_before_hybrid_exit",
        "hybrid_stop_touched_before_baseline_target",
        "hybrid_target_touched_before_baseline_stop",
        "min_low_before_target",
        "max_high_before_stop",
        "adverse_excursion_before_favorable_excursion",
        "favorable_excursion_before_adverse_excursion",
        "time_to_mfe_minutes",
        "time_to_mae_minutes",
        "stop_touch_time",
        "target_touch_time",
        "ambiguous_policy_applied",
    ]

    for row in diagnostic_df.itertuples(index=False):
        row_dict = row._asdict()
        entry_time = row_dict.get("hybrid_entry_time")
        if pd.isna(entry_time):
            continue

        time_candidates = [
            value
            for value in (
                row_dict.get("baseline_exit_time"),
                row_dict.get("hybrid_after_exit_time"),
                row_dict.get("hybrid_next_exit_time"),
                row_dict.get("hybrid_time_stop_at"),
            )
            if pd.notna(value)
        ]
        if not time_candidates:
            continue
        evaluation_end_time = max(pd.Timestamp(value) for value in time_candidates)
        path = _path_window_for_trade(pd.Series(row_dict), minute_by_session, end_time=evaluation_end_time)
        if path.empty:
            continue
        path_cache[str(row_dict["setup_id"])] = path.copy()

        direction = str(row_dict.get("direction", ""))
        direction_sign = 1 if direction == "long" else -1
        entry_price = safe_float(row_dict.get("hybrid_entry_price"), np.nan)
        stop_price = safe_float(row_dict.get("hybrid_stop_price"), np.nan)
        target_price = safe_float(row_dict.get("hybrid_target_price"), np.nan)

        favorable_series = ((path["high"] - entry_price) if direction_sign == 1 else (entry_price - path["low"])) / float(tick_size)
        adverse_series = ((entry_price - path["low"]) if direction_sign == 1 else (path["high"] - entry_price)) / float(tick_size)
        favorable_series = pd.to_numeric(favorable_series, errors="coerce")
        adverse_series = pd.to_numeric(adverse_series, errors="coerce")
        mfe_ticks = float(favorable_series.max()) if not favorable_series.empty else np.nan
        mae_ticks = float(adverse_series.max()) if not adverse_series.empty else np.nan
        mfe_idx = int(favorable_series.idxmax()) if favorable_series.notna().any() else None
        mae_idx = int(adverse_series.idxmax()) if adverse_series.notna().any() else None
        mfe_time = pd.Timestamp(path.loc[mfe_idx, "timestamp"]) if mfe_idx is not None else pd.NaT
        mae_time = pd.Timestamp(path.loc[mae_idx, "timestamp"]) if mae_idx is not None else pd.NaT

        first_touch = compute_first_touch(path, direction=direction, stop_price=stop_price, target_price=target_price)
        first_touch_time = first_touch["first_touch_time"]
        first_touch_minutes = (
            int((pd.Timestamp(first_touch_time) - pd.Timestamp(entry_time)) / pd.Timedelta(minutes=1))
            if pd.notna(first_touch_time)
            else np.nan
        )

        baseline_exit_level = safe_float(row_dict.get("baseline_exit_price"), np.nan)
        hybrid_exit_time = pd.Timestamp(row_dict["hybrid_after_exit_time"]) if pd.notna(row_dict.get("hybrid_after_exit_time")) else pd.NaT
        path_before_hybrid_exit = path.loc[path["timestamp"] <= hybrid_exit_time].copy() if pd.notna(hybrid_exit_time) else path.copy()
        baseline_exit_level_touch_time = _generic_level_touch_time(path_before_hybrid_exit, baseline_exit_level)
        baseline_target_touch_time = _generic_level_touch_time(path, safe_float(row_dict.get("baseline_target_price"), np.nan))
        baseline_stop_touch_time = _generic_level_touch_time(path, safe_float(row_dict.get("baseline_stop_price"), np.nan))
        hybrid_stop_touch_time = _generic_level_touch_time(path, stop_price)
        hybrid_target_touch_time = _generic_level_touch_time(path, target_price)

        target_touch_index = first_touch["first_touch_index"] if first_touch["first_touch"] in {"target", "both_same_minute"} else None
        stop_touch_index = first_touch["first_touch_index"] if first_touch["first_touch"] in {"stop", "both_same_minute"} else None
        min_low_before_target = float(path.loc[:target_touch_index, "low"].min()) if target_touch_index is not None else float(path["low"].min())
        max_high_before_stop = float(path.loc[:stop_touch_index, "high"].max()) if stop_touch_index is not None else float(path["high"].max())

        rows.append(
            {
                "setup_id": row_dict["setup_id"],
                "signal_time": row_dict.get("signal_time"),
                "direction": direction,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "evaluation_end_time": evaluation_end_time,
                "mfe_ticks_1m": mfe_ticks,
                "mae_ticks_1m": mae_ticks,
                "mfe_usd_1m": mfe_ticks * float(tick_size) * float(point_value_usd) if np.isfinite(mfe_ticks) else np.nan,
                "mae_usd_1m": mae_ticks * float(tick_size) * float(point_value_usd) if np.isfinite(mae_ticks) else np.nan,
                "first_touch": first_touch["first_touch"],
                "first_touch_time": first_touch_time,
                "first_touch_minutes_after_entry": first_touch_minutes,
                "baseline_exit_level_touched_before_hybrid_exit": pd.notna(baseline_exit_level_touch_time),
                "hybrid_stop_touched_before_baseline_target": pd.notna(hybrid_stop_touch_time)
                and pd.notna(baseline_target_touch_time)
                and pd.Timestamp(hybrid_stop_touch_time) <= pd.Timestamp(baseline_target_touch_time),
                "hybrid_target_touched_before_baseline_stop": pd.notna(hybrid_target_touch_time)
                and pd.notna(baseline_stop_touch_time)
                and pd.Timestamp(hybrid_target_touch_time) <= pd.Timestamp(baseline_stop_touch_time),
                "min_low_before_target": min_low_before_target,
                "max_high_before_stop": max_high_before_stop,
                "adverse_excursion_before_favorable_excursion": pd.notna(mae_time)
                and pd.notna(mfe_time)
                and pd.Timestamp(mae_time) < pd.Timestamp(mfe_time),
                "favorable_excursion_before_adverse_excursion": pd.notna(mae_time)
                and pd.notna(mfe_time)
                and pd.Timestamp(mfe_time) < pd.Timestamp(mae_time),
                "time_to_mfe_minutes": int((pd.Timestamp(mfe_time) - pd.Timestamp(entry_time)) / pd.Timedelta(minutes=1))
                if pd.notna(mfe_time)
                else np.nan,
                "time_to_mae_minutes": int((pd.Timestamp(mae_time) - pd.Timestamp(entry_time)) / pd.Timedelta(minutes=1))
                if pd.notna(mae_time)
                else np.nan,
                "stop_touch_time": first_touch["stop_touch_time"],
                "target_touch_time": first_touch["target_touch_time"],
                "ambiguous_policy_applied": first_touch["ambiguous_policy_applied"],
            }
        )

    return pd.DataFrame(rows, columns=output_columns), path_cache


def _build_divergence_summary(diagnostic_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        diagnostic_df.groupby("divergence_type", dropna=False)
        .agg(
            count=("setup_id", "count"),
            baseline_pnl_sum=("baseline_pnl", "sum"),
            hybrid_after_pnl_sum=("hybrid_after_entry_fill_pnl", "sum"),
            hybrid_next_pnl_sum=("hybrid_next_execution_bar_pnl", "sum"),
            delta_pnl_after_sum=("delta_pnl_after_entry_fill", "sum"),
            delta_pnl_next_sum=("delta_pnl_next_execution_bar", "sum"),
            avg_delta_pnl_after=("delta_pnl_after_entry_fill", "mean"),
            median_delta_pnl_after=("delta_pnl_after_entry_fill", "median"),
            winrate_baseline=("is_baseline_winner", "mean"),
            winrate_hybrid_after=("is_hybrid_winner", "mean"),
            winrate_hybrid_next=("hybrid_next_execution_bar_pnl", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
        )
        .reset_index()
        .sort_values("delta_pnl_after_sum")
    )
    grouped["pct_of_setups"] = grouped["count"] / max(len(diagnostic_df), 1)
    cols = [
        "divergence_type",
        "count",
        "pct_of_setups",
        "baseline_pnl_sum",
        "hybrid_after_pnl_sum",
        "hybrid_next_pnl_sum",
        "delta_pnl_after_sum",
        "delta_pnl_next_sum",
        "avg_delta_pnl_after",
        "median_delta_pnl_after",
        "winrate_baseline",
        "winrate_hybrid_after",
        "winrate_hybrid_next",
    ]
    return grouped[cols]


def _build_exit_reason_transition_matrix(diagnostic_df: pd.DataFrame) -> pd.DataFrame:
    pivot_count = pd.pivot_table(
        diagnostic_df,
        index="baseline_exit_reason",
        columns="hybrid_after_exit_reason",
        values="setup_id",
        aggfunc="count",
        fill_value=0,
    )
    pivot_delta = pd.pivot_table(
        diagnostic_df,
        index="baseline_exit_reason",
        columns="hybrid_after_exit_reason",
        values="delta_pnl_after_entry_fill",
        aggfunc="sum",
        fill_value=0.0,
    )
    rows: list[dict[str, Any]] = []
    for baseline_reason in pivot_count.index:
        for hybrid_reason in pivot_count.columns:
            rows.append(
                {
                    "baseline_exit_reason": baseline_reason,
                    "hybrid_exit_reason": hybrid_reason,
                    "count": int(pivot_count.loc[baseline_reason, hybrid_reason]),
                    "delta_pnl_sum": float(pivot_delta.loc[baseline_reason, hybrid_reason]),
                }
            )
    return pd.DataFrame(rows).sort_values(["baseline_exit_reason", "hybrid_exit_reason"]).reset_index(drop=True)


def _comment_auto(row: pd.Series) -> str:
    first_touch = str(row.get("first_touch", "none"))
    baseline_exit_reason = str(row.get("baseline_exit_reason", ""))
    hybrid_exit_reason = str(row.get("hybrid_exit_reason", ""))
    entry_delta_ticks = safe_float(row.get("entry_price_delta_ticks"), 0.0)
    if first_touch == "both_same_minute":
        return "Baseline winner became loser because 1min path touched stop before target inside the same 1H holding window."
    if first_touch == "stop" and baseline_exit_reason.startswith("target"):
        return "Baseline target was not observed on 1min path before stop/time stop."
    if entry_delta_ticks < -2:
        return "PnL degradation mostly explained by worse 1min entry price."
    if hybrid_exit_reason.startswith("time_stop"):
        return "Trade decayed intraday and exited on the 1min time-stop path instead of the baseline hourly path."
    return "Unclear from available columns."


def _build_execution_convention_sensitivity(
    *,
    symbol: str,
    dataset_path: Path,
    variant: VolumeClimaxPullbackV2Variant,
    baseline_net: float,
    hybrid_after_net: float,
) -> tuple[pd.DataFrame, list[str]]:
    raw = load_symbol_data(symbol, input_paths={symbol: dataset_path})
    minute_df = extract_rth(raw.copy())
    minute_df["timestamp"] = pd.to_datetime(minute_df["timestamp"], errors="coerce")
    minute_df["session_date"] = minute_df["timestamp"].dt.date
    bars_1h = resample_rth_1h(raw)
    bars_1h["timestamp"] = pd.to_datetime(bars_1h["timestamp"], errors="coerce")
    bars_1h["session_date"] = bars_1h["timestamp"].dt.date
    features = prepare_volume_climax_pullback_v2_features(bars_1h)
    signal_df = build_volume_climax_pullback_v2_signal_frame(features, variant)
    execution_model, instrument = build_execution_model_for_profile(symbol=symbol, profile_name="repo_realistic")

    configs = [
        {
            "config_name": "A_baseline_1h",
            "execution_timeframe": "1h",
            "entry_timing": "n/a",
            "protective_orders_active_from": "n/a",
            "ambiguous_policy": "n/a",
            "entry_delay_minutes": 0,
            "signal_df": signal_df,
            "kwargs": {"execution_timeframe": "1h"},
        },
        {
            "config_name": "B_hybrid_after_entry_fill",
            "execution_timeframe": "1min",
            "entry_timing": "next_execution_bar_open",
            "protective_orders_active_from": "after_entry_fill",
            "ambiguous_policy": "stop_first",
            "entry_delay_minutes": 0,
            "signal_df": signal_df,
            "kwargs": {
                "execution_timeframe": "1min",
                "minute_df": minute_df,
                "entry_timing": "next_execution_bar_open",
                "protective_orders_active_from": "after_entry_fill",
            },
        },
        {
            "config_name": "C_hybrid_next_execution_bar",
            "execution_timeframe": "1min",
            "entry_timing": "next_execution_bar_open",
            "protective_orders_active_from": "next_execution_bar",
            "ambiguous_policy": "stop_first",
            "entry_delay_minutes": 0,
            "signal_df": signal_df,
            "kwargs": {
                "execution_timeframe": "1min",
                "minute_df": minute_df,
                "entry_timing": "next_execution_bar_open",
                "protective_orders_active_from": "next_execution_bar",
            },
        },
        {
            "config_name": "D_hybrid_same_timestamp_after_entry_fill",
            "execution_timeframe": "1min",
            "entry_timing": "same_timestamp_execution_open",
            "protective_orders_active_from": "after_entry_fill",
            "ambiguous_policy": "stop_first",
            "entry_delay_minutes": 0,
            "signal_df": signal_df,
            "kwargs": {
                "execution_timeframe": "1min",
                "minute_df": minute_df,
                "entry_timing": "same_timestamp_execution_open",
                "protective_orders_active_from": "after_entry_fill",
            },
        },
        {
            "config_name": "E_hybrid_same_timestamp_next_execution_bar",
            "execution_timeframe": "1min",
            "entry_timing": "same_timestamp_execution_open",
            "protective_orders_active_from": "next_execution_bar",
            "ambiguous_policy": "stop_first",
            "entry_delay_minutes": 0,
            "signal_df": signal_df,
            "kwargs": {
                "execution_timeframe": "1min",
                "minute_df": minute_df,
                "entry_timing": "same_timestamp_execution_open",
                "protective_orders_active_from": "next_execution_bar",
            },
        },
    ]
    for delay_minutes in (5, 15):
        delayed = signal_df.copy()
        delayed["timestamp"] = pd.to_datetime(delayed["timestamp"], errors="coerce") + pd.Timedelta(minutes=delay_minutes)
        configs.append(
            {
                "config_name": f"{'H' if delay_minutes == 5 else 'I'}_hybrid_delayed_entry_{delay_minutes}min",
                "execution_timeframe": "1min",
                "entry_timing": "next_execution_bar_open",
                "protective_orders_active_from": "after_entry_fill",
                "ambiguous_policy": "stop_first",
                "entry_delay_minutes": delay_minutes,
                "signal_df": delayed,
                "kwargs": {
                    "execution_timeframe": "1min",
                    "minute_df": minute_df,
                    "entry_timing": "next_execution_bar_open",
                    "protective_orders_active_from": "after_entry_fill",
                },
            }
        )

    rows: list[dict[str, Any]] = []
    skipped_configs = [
        "F_hybrid_next_execution_bar_open_but_neutral_ambiguous_skip",
        "G_hybrid_next_execution_bar_open_but_target_first_ambiguous",
    ]
    for config in configs:
        result = run_volume_climax_pullback_v2_backtest(
            signal_df=config["signal_df"],
            variant=variant,
            execution_model=execution_model,
            instrument=instrument,
            **config["kwargs"],
        )
        trades = result.trades.copy()
        net_pnl = float(pd.to_numeric(trades.get("net_pnl_usd"), errors="coerce").fillna(0.0).sum())
        avg_trade = float(pd.to_numeric(trades.get("net_pnl_usd"), errors="coerce").mean()) if not trades.empty else 0.0
        wins = pd.to_numeric(trades.get("net_pnl_usd"), errors="coerce")
        profit_factor = float(wins.loc[wins > 0].sum() / abs(wins.loc[wins < 0].sum())) if (wins < 0).any() else np.inf
        cumulative = wins.fillna(0.0).cumsum() if not wins.empty else pd.Series(dtype=float)
        drawdown = cumulative - cumulative.cummax() if not cumulative.empty else pd.Series(dtype=float)
        rows.append(
            {
                "config_name": config["config_name"],
                "execution_timeframe": config["execution_timeframe"],
                "entry_timing": config["entry_timing"],
                "protective_orders_active_from": config["protective_orders_active_from"],
                "ambiguous_policy": config["ambiguous_policy"],
                "entry_delay_minutes": int(config["entry_delay_minutes"]),
                "trades": int(len(trades)),
                "net_pnl": net_pnl,
                "winrate": float((wins > 0).mean()) if not trades.empty else 0.0,
                "avg_trade": avg_trade,
                "profit_factor": profit_factor,
                "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
                "sharpe_if_available": np.nan,
                "pnl_vs_baseline": net_pnl - float(baseline_net),
                "pnl_vs_hybrid_after": net_pnl - float(hybrid_after_net),
            }
        )
    return pd.DataFrame(rows), skipped_configs


def _build_mfe_mae_summary(path_reconstruction: pd.DataFrame, diagnostic_df: pd.DataFrame) -> pd.DataFrame:
    if {"mfe_ticks_1m", "mae_ticks_1m", "time_to_mfe_minutes", "time_to_mae_minutes"}.issubset(diagnostic_df.columns):
        merged = diagnostic_df.copy()
    else:
        merged = diagnostic_df.merge(path_reconstruction, on=["setup_id", "signal_time", "direction"], how="left")
    buckets = {
        "all trades": merged,
        "baseline winners": merged.loc[merged["is_baseline_winner"]],
        "baseline losers": merged.loc[~merged["is_baseline_winner"]],
        "winner_to_loser": merged.loc[merged["divergence_type"] == "winner_to_loser"],
        "same_outcome_minor_delta": merged.loc[merged["divergence_type"] == "same_outcome_minor_delta"],
        "intrabar_stop_before_target": merged.loc[merged["divergence_type"] == "intrabar_stop_before_target"],
        "target_not_reached_on_1min_path": merged.loc[merged["divergence_subtype"] == "target_not_reached_on_1min_path"],
    }
    rows: list[dict[str, Any]] = []
    for label, frame in buckets.items():
        mfe = pd.to_numeric(frame.get("mfe_ticks_1m"), errors="coerce")
        mae = pd.to_numeric(frame.get("mae_ticks_1m"), errors="coerce")
        rows.append(
            {
                "bucket": label,
                "count": int(len(frame)),
                "median_mfe_ticks": float(mfe.median()) if not frame.empty else np.nan,
                "median_mae_ticks": float(mae.median()) if not frame.empty else np.nan,
                "avg_mfe_ticks": float(mfe.mean()) if not frame.empty else np.nan,
                "avg_mae_ticks": float(mae.mean()) if not frame.empty else np.nan,
                "p25_mfe_ticks": float(mfe.quantile(0.25)) if not frame.empty else np.nan,
                "p75_mfe_ticks": float(mfe.quantile(0.75)) if not frame.empty else np.nan,
                "p25_mae_ticks": float(mae.quantile(0.25)) if not frame.empty else np.nan,
                "p75_mae_ticks": float(mae.quantile(0.75)) if not frame.empty else np.nan,
                "median_time_to_mfe_minutes": float(pd.to_numeric(frame.get("time_to_mfe_minutes"), errors="coerce").median())
                if not frame.empty
                else np.nan,
                "median_time_to_mae_minutes": float(pd.to_numeric(frame.get("time_to_mae_minutes"), errors="coerce").median())
                if not frame.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _save_plots(
    *,
    output_dir: Path,
    pnl_bridge: pd.DataFrame,
    divergence_summary: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    diagnostic_df: pd.DataFrame,
    path_reconstruction: pd.DataFrame,
    recalibration_grid: pd.DataFrame,
) -> list[str]:
    plot_paths: list[str] = []

    fig, ax = plt.subplots(figsize=(10, 4))
    bridge_plot = pnl_bridge.loc[~pnl_bridge["bridge_component"].isin(["baseline_net_pnl", "hybrid_net_pnl"])].copy()
    ax.bar(bridge_plot["bridge_component"], bridge_plot["amount"], color=["#b22222" if value < 0 else "#228b22" for value in bridge_plot["amount"]])
    ax.set_title("PnL Bridge")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(0.0, color="black", linewidth=1)
    fig.tight_layout()
    path = output_dir / "pnl_bridge.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plot_paths.append(str(path))

    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = divergence_summary.sort_values("delta_pnl_after_sum")
    sns.barplot(data=ordered, x="delta_pnl_after_sum", y="divergence_type", ax=ax, color="#1f77b4")
    ax.set_title("Divergence Type Impact on Hybrid PnL")
    fig.tight_layout()
    path = output_dir / "divergence_type_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plot_paths.append(str(path))

    heatmap_count = transition_matrix.pivot(index="baseline_exit_reason", columns="hybrid_exit_reason", values="count").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(heatmap_count, annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_title("Exit Reason Transition Count")
    fig.tight_layout()
    path = output_dir / "exit_reason_transition_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plot_paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4))
    pd.to_numeric(diagnostic_df["delta_pnl_after_entry_fill"], errors="coerce").hist(ax=ax, bins=40, color="#444444")
    ax.set_title("Delta PnL Distribution")
    fig.tight_layout()
    path = output_dir / "delta_pnl_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plot_paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        pd.to_numeric(path_reconstruction.get("mfe_ticks_1m"), errors="coerce"),
        pd.to_numeric(path_reconstruction.get("mae_ticks_1m"), errors="coerce"),
        alpha=0.5,
        s=12,
    )
    ax.set_title("MFE vs MAE Scatter")
    ax.set_xlabel("MFE ticks")
    ax.set_ylabel("MAE ticks")
    fig.tight_layout()
    path = output_dir / "mfe_vs_mae_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plot_paths.append(str(path))

    if not recalibration_grid.empty:
        heat = recalibration_grid.pivot(index="stop_multiplier", columns="target_multiplier", values="net_pnl")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(heat, annot=True, fmt=".0f", cmap="RdYlGn", center=0.0, ax=ax)
        ax.set_title("Stop/Target Recalibration Heatmap")
        fig.tight_layout()
        path = output_dir / "stop_target_recalibration_heatmap.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, column in (
        ("baseline_1h", "baseline_pnl"),
        ("hybrid_after_entry_fill", "hybrid_after_entry_fill_pnl"),
        ("hybrid_next_execution_bar", "hybrid_next_execution_bar_pnl"),
    ):
        frame = diagnostic_df[["signal_time", column]].copy()
        frame = frame.sort_values("signal_time").reset_index(drop=True)
        frame["cum"] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).cumsum()
        ax.plot(frame["signal_time"], frame["cum"], label=label)
    ax.legend()
    ax.set_title("Cumulative PnL Baseline vs Hybrid")
    fig.tight_layout()
    path = output_dir / "cumulative_pnl_baseline_vs_hybrid.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plot_paths.append(str(path))

    return plot_paths


def _report_verdict(
    *,
    baseline_net: float,
    hybrid_net: float,
    divergence_summary: pd.DataFrame,
    recalibration_grid: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    top_row = divergence_summary.sort_values("delta_pnl_after_sum").iloc[0] if not divergence_summary.empty else None
    top_cause = None if top_row is None else str(top_row["divergence_type"])
    best_recalibration = recalibration_grid.sort_values("net_pnl", ascending=False).iloc[0] if not recalibration_grid.empty else None
    recovered = best_recalibration is not None and safe_float(best_recalibration["net_pnl"]) > 0.0

    baseline_biased = baseline_net > 0 and hybrid_net < 0
    if baseline_biased and recovered:
        verdict = "Alpha partially survives but requires intrabar-aware risk geometry."
    elif baseline_biased:
        verdict = "Likely execution artifact: baseline 1H/1H should not be trusted."
    elif abs(hybrid_net - baseline_net) <= max(250.0, abs(baseline_net) * 0.1):
        verdict = "Alpha appears robust; discrepancy mainly implementation-related."
    else:
        verdict = "Inconclusive due to missing columns/data."

    summary = {
        "baseline_biased": baseline_biased,
        "top_divergence_cause": top_cause,
        "best_recalibration_net_pnl": None if best_recalibration is None else safe_float(best_recalibration["net_pnl"]),
        "best_recalibration_stop_multiplier": None if best_recalibration is None else safe_float(best_recalibration["stop_multiplier"]),
        "best_recalibration_target_multiplier": None if best_recalibration is None else safe_float(best_recalibration["target_multiplier"]),
        "recovered_with_recalibration": recovered,
    }
    return verdict, summary


def _build_final_report(
    *,
    output_dir: Path,
    comparison_dir: Path,
    dataset_path: Path,
    run_metadata: dict[str, Any],
    metrics_comparison: pd.DataFrame,
    diagnostic_df: pd.DataFrame,
    divergence_summary: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    pnl_bridge: pd.DataFrame,
    path_reconstruction: pd.DataFrame,
    execution_sensitivity: pd.DataFrame,
    skipped_sensitivity: list[str],
    recalibration_grid: pd.DataFrame,
    matching_info: dict[str, Any],
    verdict: str,
) -> None:
    baseline_net = safe_float(metrics_comparison.loc[metrics_comparison["scenario"] == "baseline_1h", "net_pnl_usd"].squeeze())
    hybrid_net = safe_float(metrics_comparison.loc[metrics_comparison["scenario"] == "hybrid_after_entry_fill", "net_pnl_usd"].squeeze())
    next_net = safe_float(metrics_comparison.loc[metrics_comparison["scenario"] == "hybrid_next_execution_bar", "net_pnl_usd"].squeeze())
    top_divergences = divergence_summary.sort_values("delta_pnl_after_sum").head(5)
    destroyed = diagnostic_df.loc[
        (pd.to_numeric(diagnostic_df["baseline_pnl"], errors="coerce") > 0)
        & (pd.to_numeric(diagnostic_df["hybrid_after_entry_fill_pnl"], errors="coerce") <= 0)
    ]
    best_recalibration = recalibration_grid.sort_values("net_pnl", ascending=False).head(1)
    sensitivity_lines = execution_sensitivity.sort_values("net_pnl", ascending=False).head(7)

    lines = [
        "# Volume Climax Pullback Hybrid Execution Diagnostics",
        "",
        "## 1. Executive Summary",
        f"- Baseline 1H/1H net PnL: `{baseline_net:.2f}` USD.",
        f"- Hybrid after-entry-fill net PnL: `{hybrid_net:.2f}` USD.",
        f"- Hybrid next-execution-bar net PnL: `{next_net:.2f}` USD.",
        f"- Matching confidence: `{matching_info['matching_confidence']}`. Baseline-only `{matching_info['baseline_only']}`, hybrid-only `{matching_info['hybrid_only']}`, uncertain `{matching_info['uncertain_matches']}`.",
        f"- Baseline 1H/1H appears {'biased favorably' if baseline_net > 0 and hybrid_net < 0 else 'not obviously biased favorably'} relative to the intrabar-aware replay.",
        f"- Main driver of alpha loss: `{top_divergences.iloc[0]['divergence_type']}`." if not top_divergences.empty else "- Main driver of alpha loss: `n/a`.",
        f"- Recalibration {'finds a positive ex-post zone' if not best_recalibration.empty and safe_float(best_recalibration.iloc[0]['net_pnl']) > 0 else 'does not recover positive ex-post PnL convincingly'} without changing the alpha.",
        "",
        "## 2. Input Runs",
        f"- Comparison dir: `{comparison_dir}`.",
        f"- Minute dataset: `{dataset_path}`.",
        f"- Symbol: `{run_metadata.get('symbol')}`.",
        f"- Variant: `{run_metadata.get('variant_name')}`.",
        "",
        "## 3. Baseline vs Hybrid Metrics",
        _markdown_table(metrics_comparison),
        "",
        "## 4. PnL Bridge",
        _markdown_table(pnl_bridge),
        "",
        "## 5. Divergence Taxonomy",
        _markdown_table(top_divergences) if not top_divergences.empty else "No divergences available.",
        "",
        "## 6. Exit Reason Transition",
        _markdown_table(transition_matrix.sort_values(['count', 'delta_pnl_sum'], ascending=[False, True]).head(12))
        if not transition_matrix.empty
        else "No transition matrix available.",
        "",
        "## 7. Baseline Winners Destroyed",
        f"- Count: `{len(destroyed)}`.",
        f"- Aggregate baseline PnL of destroyed winners: `{safe_float(destroyed['baseline_pnl'].sum() if not destroyed.empty else 0.0):.2f}` USD.",
        f"- Aggregate hybrid PnL on the same setups: `{safe_float(destroyed['hybrid_after_entry_fill_pnl'].sum() if not destroyed.empty else 0.0):.2f}` USD.",
        "",
        "## 8. 1min Path Reconstruction",
        f"- Reconstructed paths: `{len(path_reconstruction)}`.",
        f"- First touch = stop: `{int((path_reconstruction.get('first_touch') == 'stop').sum()) if not path_reconstruction.empty else 0}`.",
        f"- First touch = target: `{int((path_reconstruction.get('first_touch') == 'target').sum()) if not path_reconstruction.empty else 0}`.",
        f"- First touch = both same minute: `{int((path_reconstruction.get('first_touch') == 'both_same_minute').sum()) if not path_reconstruction.empty else 0}`.",
        "",
        "## 9. Execution Convention Sensitivity",
        _markdown_table(sensitivity_lines) if not sensitivity_lines.empty else "No sensitivity results available.",
        f"- Not executed: `{', '.join(skipped_sensitivity)}`." if skipped_sensitivity else "- All requested sensitivity variants were executed.",
        "",
        "## 10. Stop/Target Recalibration Diagnostic",
        _markdown_table(best_recalibration) if not best_recalibration.empty else "No recalibration results available.",
        "",
        "## 11. Verdict",
        f"- `{verdict}`",
        "",
        "## 12. Next Actions",
        "- Abandonner le baseline 1H comme reference de performance brute pour toute strategie intraday path-dependent.",
        "- Exiger un diagnostic intrabar 1min avant de publier une strategie research avec stops/targets/time-stop intraday.",
        "- Recalibrer la geometrie stop/target a partir des distributions MAE/MFE 1min avant tout nouveau jugement sur l'alpha.",
        "- Tester des conventions d'entree decalees ou confirmees post-signal pour verifier si l'immediatete degrade structurellement l'execution.",
        "- Ajouter un guardrail CI qui bloque les campagnes intraday si aucun audit baseline-vs-path n'est genere.",
    ]
    (output_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostics(
    *,
    comparison_dir: Path,
    output_root: Path,
    minute_data_path: Path | None = None,
) -> Path:
    comparison_dir = Path(comparison_dir)
    run_metadata_path = comparison_dir / "run_metadata.json"
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8")) if run_metadata_path.exists() else {}

    loaded_inputs: dict[str, pd.DataFrame] = {}
    input_files: dict[str, Path] = {}
    for name in CSV_INPUTS:
        path = comparison_dir / name
        if path.exists():
            loaded_inputs[name] = pd.read_csv(path)
            input_files[name] = path

    if "trades_baseline_1h.csv" not in loaded_inputs or "trades_hybrid_after_entry_fill.csv" not in loaded_inputs or "trades_hybrid_next_execution_bar.csv" not in loaded_inputs:
        raise FileNotFoundError("The comparison dir must contain the three trade CSV exports.")

    baseline_trades = parse_datetime_cols(loaded_inputs["trades_baseline_1h.csv"])
    hybrid_after_trades = parse_datetime_cols(loaded_inputs["trades_hybrid_after_entry_fill.csv"])
    hybrid_next_trades = parse_datetime_cols(loaded_inputs["trades_hybrid_next_execution_bar.csv"])
    metrics_comparison = loaded_inputs.get("metrics_comparison.csv", pd.DataFrame())

    symbol = str(run_metadata.get("symbol", baseline_trades.get("symbol", pd.Series(["MNQ"])).iloc[0] if not baseline_trades.empty else "MNQ")).upper()
    dataset_path = Path(minute_data_path) if minute_data_path is not None else Path(run_metadata.get("dataset_path", ""))
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Minute dataset path does not exist: {dataset_path}. Provide --minute-data-path explicitly."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / f"volume_climax_pullback_hybrid_execution_diagnostics_{symbol.lower()}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    instrument_spec = get_instrument_spec(symbol)
    tick_size = float(instrument_spec["tick_size"])
    point_value_usd = float(instrument_spec["point_value_usd"])

    diagnostic_df, matching_info = _build_diagnostic_trade_level(
        baseline_trades=baseline_trades,
        hybrid_after_trades=hybrid_after_trades,
        hybrid_next_trades=hybrid_next_trades,
        tick_size=tick_size,
    )

    raw_minute_df = load_symbol_data(symbol, input_paths={symbol: dataset_path})
    minute_df = extract_rth(raw_minute_df.copy())
    minute_df["timestamp"] = pd.to_datetime(minute_df["timestamp"], errors="coerce")
    minute_df["session_date"] = minute_df["timestamp"].dt.date

    path_reconstruction, _ = _build_path_reconstruction(
        diagnostic_df=diagnostic_df,
        minute_df=minute_df,
        tick_size=tick_size,
        point_value_usd=point_value_usd,
    )
    diagnostic_df = diagnostic_df.merge(path_reconstruction, on=["setup_id", "signal_time", "direction"], how="left")
    refreshed_classification = diagnostic_df.apply(classify_divergence, axis=1, result_type="expand")
    for column in refreshed_classification.columns:
        diagnostic_df[column] = refreshed_classification[column]
    diagnostic_df["primary_pnl_driver"] = diagnostic_df.apply(_primary_pnl_driver, axis=1)

    divergence_summary = _build_divergence_summary(diagnostic_df)
    transition_matrix = _build_exit_reason_transition_matrix(diagnostic_df)
    pnl_bridge = build_pnl_bridge(diagnostic_df)

    baseline_winners_destroyed = diagnostic_df.loc[
        (pd.to_numeric(diagnostic_df["baseline_pnl"], errors="coerce") > 0)
        & (pd.to_numeric(diagnostic_df["hybrid_after_entry_fill_pnl"], errors="coerce") <= 0)
    ].copy()
    baseline_winners_destroyed["delta_pnl"] = pd.to_numeric(
        baseline_winners_destroyed["delta_pnl_after_entry_fill"], errors="coerce"
    )
    baseline_winners_destroyed["hybrid_pnl"] = baseline_winners_destroyed["hybrid_after_entry_fill_pnl"]
    baseline_winners_destroyed["hybrid_exit_reason"] = baseline_winners_destroyed["hybrid_after_exit_reason"]
    baseline_winners_destroyed["comment_auto"] = baseline_winners_destroyed.apply(_comment_auto, axis=1)
    baseline_winners_destroyed = baseline_winners_destroyed[
        [
            "signal_time",
            "direction",
            "baseline_pnl",
            "hybrid_pnl",
            "delta_pnl",
            "baseline_exit_reason",
            "hybrid_exit_reason",
            "first_touch",
            "first_touch_time",
            "first_touch_minutes_after_entry",
            "mfe_ticks_1m",
            "mae_ticks_1m",
            "hybrid_entry_price",
            "hybrid_stop_price",
            "hybrid_target_price",
            "comment_auto",
        ]
    ].rename(
        columns={
            "hybrid_entry_price": "entry_price",
            "hybrid_stop_price": "stop_price",
            "hybrid_target_price": "target_price",
        }
    )
    top_50_pnl_destructions = diagnostic_df.sort_values("delta_pnl_after_entry_fill").head(50).copy()

    mfe_mae_summary = _build_mfe_mae_summary(path_reconstruction, diagnostic_df)
    recalibration_input = diagnostic_df.loc[diagnostic_df["hybrid_entry_time"].notna()].copy()
    recalibration_grid = run_recalibration_grid(
        recalibration_input.rename(
            columns={
                "hybrid_entry_time": "entry_time",
                "hybrid_time_stop_at": "time_stop_at",
                "hybrid_stop_price": "stop_price",
                "hybrid_target_price": "target_price",
            }
        ),
        minute_df,
        tick_size=tick_size,
        point_value_usd=point_value_usd,
    )

    variant_payload = run_metadata.get("variant")
    if variant_payload is None:
        raise ValueError("run_metadata.json must contain the variant payload for sensitivity replays.")
    variant = VolumeClimaxPullbackV2Variant(**variant_payload)
    baseline_net = safe_float(metrics_comparison.loc[metrics_comparison["scenario"] == "baseline_1h", "net_pnl_usd"].squeeze(), 0.0)
    hybrid_after_net = safe_float(
        metrics_comparison.loc[metrics_comparison["scenario"] == "hybrid_after_entry_fill", "net_pnl_usd"].squeeze(),
        0.0,
    )
    execution_sensitivity, skipped_sensitivity = _build_execution_convention_sensitivity(
        symbol=symbol,
        dataset_path=dataset_path,
        variant=variant,
        baseline_net=baseline_net,
        hybrid_after_net=hybrid_after_net,
    )

    diagnostic_df.to_csv(output_dir / "diagnostic_trade_level.csv", index=False)
    path_reconstruction.to_csv(output_dir / "path_reconstruction_1m.csv", index=False)
    divergence_summary.to_csv(output_dir / "divergence_summary.csv", index=False)
    transition_matrix.to_csv(output_dir / "exit_reason_transition_matrix.csv", index=False)
    pnl_bridge.to_csv(output_dir / "pnl_bridge.csv", index=False)
    baseline_winners_destroyed.to_csv(output_dir / "baseline_winners_destroyed.csv", index=False)
    top_50_pnl_destructions.to_csv(output_dir / "top_50_pnl_destructions.csv", index=False)
    execution_sensitivity.to_csv(output_dir / "execution_convention_sensitivity.csv", index=False)
    mfe_mae_summary.to_csv(output_dir / "mfe_mae_summary.csv", index=False)
    recalibration_grid.to_csv(output_dir / "stop_target_recalibration_grid.csv", index=False)

    plot_paths = _save_plots(
        output_dir=output_dir,
        pnl_bridge=pnl_bridge,
        divergence_summary=divergence_summary,
        transition_matrix=transition_matrix,
        diagnostic_df=diagnostic_df,
        path_reconstruction=path_reconstruction,
        recalibration_grid=recalibration_grid,
    )

    verdict, verdict_context = _report_verdict(
        baseline_net=baseline_net,
        hybrid_net=hybrid_after_net,
        divergence_summary=divergence_summary,
        recalibration_grid=recalibration_grid,
    )
    _build_final_report(
        output_dir=output_dir,
        comparison_dir=comparison_dir,
        dataset_path=dataset_path,
        run_metadata=run_metadata,
        metrics_comparison=metrics_comparison,
        diagnostic_df=diagnostic_df,
        divergence_summary=divergence_summary,
        transition_matrix=transition_matrix,
        pnl_bridge=pnl_bridge,
        path_reconstruction=path_reconstruction,
        execution_sensitivity=execution_sensitivity,
        skipped_sensitivity=skipped_sensitivity,
        recalibration_grid=recalibration_grid,
        matching_info=matching_info,
        verdict=verdict,
    )

    output_metadata = {
        "generated_at": datetime.now().isoformat(),
        "comparison_dir": str(comparison_dir),
        "output_dir": str(output_dir),
        "symbol": symbol,
        "dataset_path": str(dataset_path),
        "python_version": sys.version,
        "platform": platform.platform(),
        "matching_info": matching_info,
        "variant_name": variant.name,
        "verdict": verdict,
        "verdict_context": verdict_context,
        "plots": plot_paths,
        "skipped_sensitivity_configs": skipped_sensitivity,
        "input_files": {name: _file_metadata(path) for name, path in input_files.items()},
        "minute_data_file": _file_metadata(dataset_path),
    }
    if run_metadata_path.exists():
        output_metadata["comparison_run_metadata_file"] = _file_metadata(run_metadata_path)
    (output_dir / "run_metadata.json").write_text(json.dumps(output_metadata, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain baseline vs hybrid execution divergences for volume climax pullback.")
    parser.add_argument("--comparison-dir", default=str(DEFAULT_COMPARISON_DIR), help="Directory containing the existing hybrid validation exports.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Export root for the diagnostics run.")
    parser.add_argument("--minute-data-path", default=None, help="Optional explicit 1min parquet/csv dataset path.")
    args = parser.parse_args()

    run_dir = run_diagnostics(
        comparison_dir=Path(args.comparison_dir),
        output_root=Path(args.output_root),
        minute_data_path=Path(args.minute_data_path) if args.minute_data_path else None,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
