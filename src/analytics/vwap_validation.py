"""Rigorous validation campaign for the VWAP pullback continuation strategy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analytics.metrics import compute_metrics
from src.analytics.vwap_metrics import (
    build_long_short_stats,
    build_pnl_by_hour_table,
    build_trade_hour_table,
    build_weekday_pnl_table,
)
from src.config.paths import EXPORTS_DIR, NOTEBOOKS_DIR, ensure_directories
from src.config.vwap_campaign import (
    DEFAULT_PAPER_TIME_EXIT,
    DEFAULT_RTH_SESSION_END,
    DEFAULT_RTH_SESSION_START,
    PropFirmConstraintConfig,
    TimeWindow,
    VWAPVariantConfig,
    build_default_prop_constraints,
    build_default_vwap_variants,
    infer_symbol_from_dataset_path,
    resolve_default_vwap_dataset,
)
from src.data.cleaning import clean_ohlcv
from src.data.loader import load_ohlcv_file
from src.engine.execution_model import ExecutionModel
from src.engine.portfolio import build_equity_curve
from src.engine.vwap_backtester import InstrumentDetails, VWAPBacktestResult, build_execution_model_for_profile, run_vwap_backtest
from src.strategy.vwap import build_vwap_signal_frame, prepare_vwap_feature_frame


REFERENCE_VARIANT_NAME = "vwap_pullback_continuation"
DEFAULT_SPLIT_FRACTIONS = (0.60, 0.65, 0.70, 0.75)
VALIDATION_PHASES = (
    "nominal",
    "stress",
    "local",
    "splits",
    "concentration",
    "challenge",
    "cross",
    "representative",
    "notebook",
)
SUMMARY_METRIC_COLUMNS = [
    "total_trades",
    "net_pnl",
    "gross_pnl",
    "sharpe_ratio",
    "sortino_ratio",
    "profit_factor",
    "hit_rate",
    "expectancy_per_trade",
    "expectancy_per_day",
    "avg_win",
    "avg_loss",
    "avg_gain_loss_ratio",
    "max_drawdown",
    "return_over_max_drawdown",
    "calmar_ratio",
    "ulcer_index",
    "annualized_vol",
    "mean_trades_per_day",
    "median_trades_per_day",
    "max_trades_per_day",
    "avg_time_in_position_min",
    "avg_exposure_pct",
    "long_trade_share",
    "short_trade_share",
    "worst_losing_trades_streak",
    "worst_losing_days_streak",
    "median_recovery_days",
    "max_recovery_days",
    "top_1_day_contribution_pct",
    "top_3_day_contribution_pct",
    "top_5_day_contribution_pct",
    "top_10_day_contribution_pct",
    "pnl_excluding_top_1_day",
    "pnl_excluding_top_3_days",
    "pnl_excluding_top_5_days",
    "pnl_excluding_best_month",
    "worst_daily_loss_usd",
    "days_below_neg_0p25r_freq",
    "days_below_neg_0p5r_freq",
    "days_below_neg_1r_freq",
    "red_streak_ge_2_freq",
    "red_streak_ge_3_freq",
    "red_streak_ge_4_freq",
    "red_streak_ge_5_freq",
    "trade_loss_streak_ge_3_freq",
    "trade_loss_streak_ge_4_freq",
    "trade_loss_streak_ge_5_freq",
    "trade_loss_streak_ge_6_freq",
    "profit_to_drawdown_ratio",
    "daily_loss_limit_breach_freq",
    "trailing_drawdown_breach_freq",
]


@dataclass(frozen=True)
class ValidationSpec:
    """Top-level VWAP validation campaign settings."""

    dataset_path: Path
    reference_variant_name: str = REFERENCE_VARIANT_NAME
    source_run_metadata_path: Path | None = EXPORTS_DIR / "vwap_full_smoke_v2" / "summary" / "run_metadata.json"
    is_fraction: float = 0.70
    session_start: str = DEFAULT_RTH_SESSION_START
    session_end: str = DEFAULT_RTH_SESSION_END
    paper_time_exit: str = DEFAULT_PAPER_TIME_EXIT
    rolling_window_days: int = 20
    bootstrap_paths: int = 750
    random_seed: int = 42
    prop_constraints: PropFirmConstraintConfig = build_default_prop_constraints()
    cross_instruments: tuple[str, ...] = ("MNQ", "NQ", "MES")


@dataclass(frozen=True)
class StressScenario:
    """Execution / microstructure stress applied on top of the nominal run."""

    name: str
    commission_multiplier: float = 1.0
    slippage_multiplier: float = 1.0
    entry_penalty_ticks: float = 0.0
    open_penalty_ticks: float = 0.0
    open_penalty_minutes: int = 15
    notes: str = ""


@dataclass(frozen=True)
class ChallengeScenario:
    """Prop-firm style trade-path simulation scenario."""

    name: str
    label: str
    risk_per_trade_pct: float
    max_contracts: int
    stop_after_losses_in_day: int
    daily_loss_limit_usd: float
    trailing_drawdown_limit_usd: float
    profit_target_usd: float
    horizon_days: int
    deleverage_after_red_days: int
    deleverage_factor: float


@dataclass
class NominalEvaluation:
    """Full nominal evaluation payload reused across the campaign."""

    variant: VWAPVariantConfig
    signal_df: pd.DataFrame
    result: VWAPBacktestResult
    trades: pd.DataFrame
    daily_results: pd.DataFrame
    bar_results: pd.DataFrame
    instrument: InstrumentDetails
    execution_model: ExecutionModel
    all_sessions: list
    is_sessions: list
    oos_sessions: list
    summary_by_scope: pd.DataFrame
    tables: dict[str, pd.DataFrame]


def _serialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    clean = {key: _serialize_value(value) for key, value in payload.items()}
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")


def _split_sessions(all_sessions: list, is_fraction: float) -> tuple[list, list]:
    if len(all_sessions) < 2:
        raise ValueError("Not enough sessions for an IS/OOS split.")
    split_idx = int(len(all_sessions) * float(is_fraction))
    split_idx = max(1, min(len(all_sessions) - 1, split_idx))
    return all_sessions[:split_idx], all_sessions[split_idx:]


def _subset_frame_by_sessions(frame: pd.DataFrame, sessions: list) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    session_set = set(pd.to_datetime(pd.Index(sessions)).date)
    out = frame.copy()
    out_dates = pd.to_datetime(out["session_date"]).dt.date
    return out.loc[out_dates.isin(session_set)].copy().reset_index(drop=True)


def _variant_dict(variant: VWAPVariantConfig) -> dict[str, Any]:
    payload = asdict(variant)
    payload["time_windows"] = [asdict(window) for window in variant.time_windows]
    return payload


def _variant_cache_key(variant: VWAPVariantConfig) -> str:
    return json.dumps(_variant_dict(variant), sort_keys=True, default=_serialize_value)


def _time_window_label(windows: tuple[TimeWindow, ...]) -> str:
    if not windows:
        return "full_rth"
    return "|".join(f"{window.start}->{window.end}" for window in windows)


def _parse_time_label(label: str) -> tuple[int, int]:
    if label == "off":
        return (-1, -1)
    parts = label.split(":")
    return (int(parts[0]), int(parts[1]))


def _sortino_ratio(daily_pnl: pd.Series, capital: float) -> float:
    if capital <= 0:
        return 0.0
    daily_returns = pd.Series(daily_pnl, dtype=float) / capital
    downside = daily_returns[daily_returns < 0]
    if len(daily_returns) < 2 or downside.empty:
        return 0.0
    downside_std = float(np.sqrt(np.mean(np.square(downside))))
    if downside_std <= 0:
        return 0.0
    return float((daily_returns.mean() / downside_std) * math.sqrt(252.0))


def _annualized_vol(daily_pnl: pd.Series, capital: float) -> float:
    if capital <= 0 or len(daily_pnl) < 2:
        return 0.0
    daily_returns = pd.Series(daily_pnl, dtype=float) / capital
    return float(daily_returns.std(ddof=0) * math.sqrt(252.0))


def _ulcer_index(equity: pd.Series, peak: pd.Series) -> float:
    if equity.empty or peak.empty:
        return 0.0
    drawdown_pct = np.where(peak > 0, (equity - peak) / peak * 100.0, 0.0)
    return float(np.sqrt(np.mean(np.square(drawdown_pct))))


def _run_lengths(mask: pd.Series) -> list[int]:
    lengths: list[int] = []
    current = 0
    for flag in pd.Series(mask).fillna(False).astype(bool).tolist():
        if flag:
            current += 1
        elif current > 0:
            lengths.append(current)
            current = 0
    if current > 0:
        lengths.append(current)
    return lengths


def _streak_distribution(lengths: list[int], label: str) -> pd.DataFrame:
    if not lengths:
        return pd.DataFrame(columns=["label", "streak_length", "count", "frequency"])
    table = pd.Series(lengths, dtype=int).value_counts().sort_index().rename_axis("streak_length").reset_index(name="count")
    table["frequency"] = table["count"] / table["count"].sum()
    table.insert(0, "label", label)
    return table


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return default
    value = numerator / denominator
    return float(value) if math.isfinite(value) else default


def _drawdown_episode_table(daily_results: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    columns = [
        "episode_id",
        "start_date",
        "trough_date",
        "recovery_date",
        "peak_equity_before",
        "trough_drawdown_usd",
        "duration_sessions",
        "recovered",
    ]
    if daily_results.empty:
        return pd.DataFrame(columns=columns)

    daily = daily_results[["session_date", "daily_pnl_usd"]].copy().sort_values("session_date").reset_index(drop=True)
    daily["equity"] = initial_capital + pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0).cumsum()
    peak_equity = float(initial_capital)
    in_drawdown = False
    episode_id = 1
    start_idx = 0
    start_date = None
    peak_before = peak_equity
    trough_drawdown = 0.0
    trough_date = None
    episodes: list[dict[str, Any]] = []

    for idx, row in daily.iterrows():
        equity = float(row["equity"])
        session_date = row["session_date"]
        drawdown = equity - peak_equity
        if not in_drawdown and drawdown < 0:
            in_drawdown = True
            start_idx = idx
            start_date = session_date
            peak_before = peak_equity
            trough_drawdown = drawdown
            trough_date = session_date
        if in_drawdown:
            if drawdown < trough_drawdown:
                trough_drawdown = drawdown
                trough_date = session_date
            if equity >= peak_before:
                episodes.append(
                    {
                        "episode_id": episode_id,
                        "start_date": start_date,
                        "trough_date": trough_date,
                        "recovery_date": session_date,
                        "peak_equity_before": peak_before,
                        "trough_drawdown_usd": trough_drawdown,
                        "duration_sessions": idx - start_idx + 1,
                        "recovered": True,
                    }
                )
                episode_id += 1
                in_drawdown = False
        peak_equity = max(peak_equity, equity)

    if in_drawdown:
        last_idx = len(daily) - 1
        episodes.append(
            {
                "episode_id": episode_id,
                "start_date": start_date,
                "trough_date": trough_date,
                "recovery_date": pd.NaT,
                "peak_equity_before": peak_before,
                "trough_drawdown_usd": trough_drawdown,
                "duration_sessions": last_idx - start_idx + 1,
                "recovered": False,
            }
        )
    return pd.DataFrame(episodes, columns=columns)


def _rolling_validation_table(
    daily_results: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float,
    window_days: int,
) -> pd.DataFrame:
    if daily_results.empty:
        return pd.DataFrame(
            columns=[
                "session_date",
                "rolling_sharpe_20d",
                "rolling_expectancy_20d",
                "rolling_hit_rate_20d",
                "rolling_trade_count_20d",
            ]
        )

    daily = daily_results[["session_date", "daily_pnl_usd"]].copy().sort_values("session_date").reset_index(drop=True)
    daily["session_date"] = pd.to_datetime(daily["session_date"])
    daily_returns = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0) / max(float(initial_capital), 1.0)
    rolling_mean = daily_returns.rolling(window_days).mean()
    rolling_std = daily_returns.rolling(window_days).std(ddof=0)
    daily["rolling_sharpe_20d"] = np.where(
        rolling_std > 0,
        (rolling_mean / rolling_std) * math.sqrt(252.0),
        np.nan,
    )
    daily["rolling_expectancy_20d"] = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").rolling(window_days).mean()

    if trades.empty:
        daily["rolling_hit_rate_20d"] = np.nan
        daily["rolling_trade_count_20d"] = 0
        return daily

    trade_view = trades.copy()
    trade_view["session_date"] = pd.to_datetime(trade_view["session_date"])
    trade_view["is_win"] = pd.to_numeric(trade_view["net_pnl_usd"], errors="coerce").gt(0)
    daily_trade = (
        trade_view.groupby("session_date", as_index=False)
        .agg(trades=("trade_id", "count"), wins=("is_win", "sum"))
        .sort_values("session_date")
        .reset_index(drop=True)
    )
    merged = daily.merge(daily_trade, on="session_date", how="left").fillna({"trades": 0, "wins": 0})
    merged["rolling_wins"] = merged["wins"].rolling(window_days).sum()
    merged["rolling_trades_20d"] = merged["trades"].rolling(window_days).sum()
    merged["rolling_hit_rate_20d"] = np.where(
        merged["rolling_trades_20d"] > 0,
        merged["rolling_wins"] / merged["rolling_trades_20d"],
        np.nan,
    )
    merged["rolling_trade_count_20d"] = merged["rolling_trades_20d"]
    return merged[
        [
            "session_date",
            "rolling_sharpe_20d",
            "rolling_expectancy_20d",
            "rolling_hit_rate_20d",
            "rolling_trade_count_20d",
        ]
    ]


def _monthly_pnl_table(daily_results: pd.DataFrame) -> pd.DataFrame:
    if daily_results.empty:
        return pd.DataFrame(columns=["period", "net_pnl_usd", "days", "avg_day_pnl_usd"])
    out = daily_results.copy()
    out["period"] = pd.to_datetime(out["session_date"]).dt.to_period("M").astype(str)
    return (
        out.groupby("period", as_index=False)
        .agg(net_pnl_usd=("daily_pnl_usd", "sum"), days=("session_date", "count"), avg_day_pnl_usd=("daily_pnl_usd", "mean"))
        .sort_values("period")
        .reset_index(drop=True)
    )


def _quarterly_pnl_table(daily_results: pd.DataFrame) -> pd.DataFrame:
    if daily_results.empty:
        return pd.DataFrame(columns=["period", "net_pnl_usd", "days", "avg_day_pnl_usd"])
    out = daily_results.copy()
    out["period"] = pd.to_datetime(out["session_date"]).dt.to_period("Q").astype(str)
    return (
        out.groupby("period", as_index=False)
        .agg(net_pnl_usd=("daily_pnl_usd", "sum"), days=("session_date", "count"), avg_day_pnl_usd=("daily_pnl_usd", "mean"))
        .sort_values("period")
        .reset_index(drop=True)
    )


def _ensure_trade_risk(
    trades: pd.DataFrame,
    instrument: InstrumentDetails,
    execution_model: ExecutionModel,
) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    out = trades.copy()
    missing_mask = pd.to_numeric(out.get("trade_risk_usd"), errors="coerce").isna()
    if not missing_mask.any():
        return out

    entry_price = pd.to_numeric(out["entry_price"], errors="coerce")
    stop_price = pd.to_numeric(out["stop_price"], errors="coerce")
    quantity = pd.to_numeric(out["quantity"], errors="coerce").fillna(0.0)
    distance = (entry_price - stop_price).abs()
    per_contract = distance * float(instrument.point_value_usd) + execution_model.round_trip_fees(quantity=1)
    effective_risk = per_contract * quantity
    out.loc[missing_mask, "risk_per_contract_usd"] = per_contract.loc[missing_mask]
    out.loc[missing_mask, "actual_risk_usd"] = effective_risk.loc[missing_mask]
    out.loc[missing_mask, "trade_risk_usd"] = effective_risk.loc[missing_mask]
    out["r_multiple"] = np.where(
        pd.to_numeric(out["trade_risk_usd"], errors="coerce") > 0,
        pd.to_numeric(out["net_pnl_usd"], errors="coerce") / pd.to_numeric(out["trade_risk_usd"], errors="coerce"),
        np.nan,
    )
    return out


def _rebuild_daily_results_from_trades(
    trades: pd.DataFrame,
    all_sessions: list,
    initial_capital: float,
) -> pd.DataFrame:
    session_index = pd.Index(pd.to_datetime(pd.Index(all_sessions)).date)
    daily = pd.DataFrame({"session_date": session_index})
    if trades.empty:
        daily["daily_pnl_usd"] = 0.0
        daily["daily_gross_pnl_usd"] = 0.0
        daily["daily_fees_usd"] = 0.0
        daily["daily_trade_count"] = 0
        daily["daily_loss_count"] = 0
    else:
        view = trades.copy()
        view["session_date"] = pd.to_datetime(view["session_date"]).dt.date
        view["loss_trade"] = pd.to_numeric(view["net_pnl_usd"], errors="coerce").lt(0)
        grouped = (
            view.groupby("session_date", as_index=False)
            .agg(
                daily_pnl_usd=("net_pnl_usd", "sum"),
                daily_gross_pnl_usd=("pnl_usd", "sum"),
                daily_fees_usd=("fees", "sum"),
                daily_trade_count=("trade_id", "count"),
                daily_loss_count=("loss_trade", "sum"),
            )
        )
        daily = daily.merge(grouped, on="session_date", how="left").fillna(
            {"daily_pnl_usd": 0.0, "daily_gross_pnl_usd": 0.0, "daily_fees_usd": 0.0, "daily_trade_count": 0, "daily_loss_count": 0}
        )

    daily["daily_stop_breached"] = False
    daily["trading_halted"] = False
    daily = daily.sort_values("session_date").reset_index(drop=True)
    daily["equity"] = float(initial_capital) + pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").cumsum()
    daily["peak_equity"] = daily["equity"].cummax()
    daily["drawdown_usd"] = daily["equity"] - daily["peak_equity"]
    daily["green_day"] = daily["daily_pnl_usd"] > 0
    daily["weekday"] = pd.to_datetime(daily["session_date"]).dt.day_name()
    return daily


def _hourly_validation_table(bar_results: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    pnl_by_hour = build_pnl_by_hour_table(bar_results)
    trade_by_hour = build_trade_hour_table(trades)
    if pnl_by_hour.empty and trade_by_hour.empty:
        return pd.DataFrame(columns=["hour", "net_bar_pnl_usd", "bars", "trades", "expectancy_per_trade", "win_rate"])
    return (
        pnl_by_hour.merge(trade_by_hour, on="hour", how="outer")
        .rename(columns={"avg_trade_pnl_usd": "expectancy_per_trade"})
        .sort_values("hour")
        .reset_index(drop=True)
    )


def _weekday_expectancy_table(daily_results: pd.DataFrame) -> pd.DataFrame:
    table = build_weekday_pnl_table(daily_results)
    if table.empty:
        return table
    return table.rename(columns={"avg_day_pnl_usd": "expectancy_per_day"})


def _concentration_tables(daily_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if daily_results.empty:
        summary = {
            "top_1_day_contribution_pct": 0.0,
            "top_3_day_contribution_pct": 0.0,
            "top_5_day_contribution_pct": 0.0,
            "top_10_day_contribution_pct": 0.0,
            "bottom_1_day_contribution_pct": 0.0,
            "bottom_3_day_contribution_pct": 0.0,
            "bottom_5_day_contribution_pct": 0.0,
            "bottom_10_day_contribution_pct": 0.0,
            "pnl_excluding_top_1_day": 0.0,
            "pnl_excluding_top_3_days": 0.0,
            "pnl_excluding_top_5_days": 0.0,
            "pnl_excluding_best_month": 0.0,
            "top_day_contribution_ratio": 0.0,
        }
        return pd.DataFrame(), pd.DataFrame(), summary

    ranked_days = daily_results[["session_date", "daily_pnl_usd", "daily_trade_count"]].copy()
    ranked_days = ranked_days.sort_values("daily_pnl_usd", ascending=False).reset_index(drop=True)
    total_pnl = float(pd.to_numeric(ranked_days["daily_pnl_usd"], errors="coerce").sum())
    ranked_days["rank"] = np.arange(1, len(ranked_days) + 1)
    ranked_days["cumulative_pnl"] = pd.to_numeric(ranked_days["daily_pnl_usd"], errors="coerce").cumsum()
    ranked_days["cumulative_contribution_pct"] = np.where(
        total_pnl != 0.0,
        ranked_days["cumulative_pnl"] / total_pnl,
        np.nan,
    )

    weekly = daily_results.copy()
    weekly["week"] = pd.to_datetime(weekly["session_date"]).dt.to_period("W-FRI").astype(str)
    ranked_weeks = (
        weekly.groupby("week", as_index=False)
        .agg(net_pnl_usd=("daily_pnl_usd", "sum"), days=("session_date", "count"))
        .sort_values("net_pnl_usd", ascending=False)
        .reset_index(drop=True)
    )
    ranked_weeks["rank"] = np.arange(1, len(ranked_weeks) + 1)

    monthly = _monthly_pnl_table(daily_results)
    best_month_pnl = float(monthly["net_pnl_usd"].max()) if not monthly.empty else 0.0

    def contrib(n: int, ascending: bool = False) -> float:
        subset = ranked_days.sort_values("daily_pnl_usd", ascending=ascending).head(n)
        numerator = float(subset["daily_pnl_usd"].sum())
        return float(numerator / total_pnl) if total_pnl != 0 else 0.0

    summary = {
        "top_1_day_contribution_pct": contrib(1, ascending=False),
        "top_3_day_contribution_pct": contrib(3, ascending=False),
        "top_5_day_contribution_pct": contrib(5, ascending=False),
        "top_10_day_contribution_pct": contrib(10, ascending=False),
        "bottom_1_day_contribution_pct": contrib(1, ascending=True),
        "bottom_3_day_contribution_pct": contrib(3, ascending=True),
        "bottom_5_day_contribution_pct": contrib(5, ascending=True),
        "bottom_10_day_contribution_pct": contrib(10, ascending=True),
        "pnl_excluding_top_1_day": total_pnl - float(ranked_days.head(1)["daily_pnl_usd"].sum()),
        "pnl_excluding_top_3_days": total_pnl - float(ranked_days.head(3)["daily_pnl_usd"].sum()),
        "pnl_excluding_top_5_days": total_pnl - float(ranked_days.head(5)["daily_pnl_usd"].sum()),
        "pnl_excluding_best_month": total_pnl - best_month_pnl,
        "top_day_contribution_ratio": float(abs(contrib(10, ascending=False))),
    }
    return ranked_days, ranked_weeks, summary


def _prop_frequency_summary(
    trades: pd.DataFrame,
    daily_results: pd.DataFrame,
    initial_capital: float,
    constraints: PropFirmConstraintConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    daily_pnl = pd.to_numeric(daily_results.get("daily_pnl_usd"), errors="coerce").fillna(0.0)
    trade_pnl = pd.to_numeric(trades.get("net_pnl_usd"), errors="coerce").fillna(0.0)
    reference_r = pd.to_numeric(trades.get("trade_risk_usd"), errors="coerce").dropna().median()
    if not math.isfinite(float(reference_r)) or float(reference_r) <= 0:
        reference_r = np.nan
    daily_r = daily_pnl / float(reference_r) if pd.notna(reference_r) else pd.Series(np.nan, index=daily_pnl.index)
    red_lengths = _run_lengths(daily_pnl < 0)
    trade_loss_lengths = _run_lengths(trade_pnl < 0)
    equity = initial_capital + daily_pnl.cumsum()
    peak = equity.cummax()
    trailing_drawdown = peak - equity

    summary = {
        "reference_r_usd": float(reference_r) if pd.notna(reference_r) else np.nan,
        "worst_daily_loss_usd": float(daily_pnl.min()) if not daily_pnl.empty else 0.0,
        "days_below_neg_0p25r_freq": float((daily_r <= -0.25).mean()) if daily_r.notna().any() else np.nan,
        "days_below_neg_0p5r_freq": float((daily_r <= -0.5).mean()) if daily_r.notna().any() else np.nan,
        "days_below_neg_1r_freq": float((daily_r <= -1.0).mean()) if daily_r.notna().any() else np.nan,
        "red_streak_ge_2_freq": float(sum(length >= 2 for length in red_lengths) / len(red_lengths)) if red_lengths else 0.0,
        "red_streak_ge_3_freq": float(sum(length >= 3 for length in red_lengths) / len(red_lengths)) if red_lengths else 0.0,
        "red_streak_ge_4_freq": float(sum(length >= 4 for length in red_lengths) / len(red_lengths)) if red_lengths else 0.0,
        "red_streak_ge_5_freq": float(sum(length >= 5 for length in red_lengths) / len(red_lengths)) if red_lengths else 0.0,
        "trade_loss_streak_ge_3_freq": float(sum(length >= 3 for length in trade_loss_lengths) / len(trade_loss_lengths)) if trade_loss_lengths else 0.0,
        "trade_loss_streak_ge_4_freq": float(sum(length >= 4 for length in trade_loss_lengths) / len(trade_loss_lengths)) if trade_loss_lengths else 0.0,
        "trade_loss_streak_ge_5_freq": float(sum(length >= 5 for length in trade_loss_lengths) / len(trade_loss_lengths)) if trade_loss_lengths else 0.0,
        "trade_loss_streak_ge_6_freq": float(sum(length >= 6 for length in trade_loss_lengths) / len(trade_loss_lengths)) if trade_loss_lengths else 0.0,
        "daily_loss_limit_breach_freq": float((daily_pnl <= -constraints.daily_loss_limit_usd).mean()) if not daily_pnl.empty else 0.0,
        "trailing_drawdown_breach_freq": float((trailing_drawdown >= constraints.trailing_drawdown_limit_usd).mean()) if len(trailing_drawdown) > 0 else 0.0,
        "profit_to_drawdown_ratio": float(daily_pnl.sum() / max(abs(float((equity - peak).min())), 1.0)),
    }
    thresholds = pd.DataFrame(
        [
            {"metric": "days_below_neg_0p25r_freq", "value": summary["days_below_neg_0p25r_freq"]},
            {"metric": "days_below_neg_0p5r_freq", "value": summary["days_below_neg_0p5r_freq"]},
            {"metric": "days_below_neg_1r_freq", "value": summary["days_below_neg_1r_freq"]},
            {"metric": "red_streak_ge_2_freq", "value": summary["red_streak_ge_2_freq"]},
            {"metric": "red_streak_ge_3_freq", "value": summary["red_streak_ge_3_freq"]},
            {"metric": "red_streak_ge_4_freq", "value": summary["red_streak_ge_4_freq"]},
            {"metric": "red_streak_ge_5_freq", "value": summary["red_streak_ge_5_freq"]},
            {"metric": "trade_loss_streak_ge_3_freq", "value": summary["trade_loss_streak_ge_3_freq"]},
            {"metric": "trade_loss_streak_ge_4_freq", "value": summary["trade_loss_streak_ge_4_freq"]},
            {"metric": "trade_loss_streak_ge_5_freq", "value": summary["trade_loss_streak_ge_5_freq"]},
            {"metric": "trade_loss_streak_ge_6_freq", "value": summary["trade_loss_streak_ge_6_freq"]},
            {"metric": "daily_loss_limit_breach_freq", "value": summary["daily_loss_limit_breach_freq"]},
            {"metric": "trailing_drawdown_breach_freq", "value": summary["trailing_drawdown_breach_freq"]},
        ]
    )
    return summary, thresholds


def _compute_scope_summary(
    trades: pd.DataFrame,
    daily_results: pd.DataFrame,
    bar_results: pd.DataFrame | None,
    signal_df: pd.DataFrame | None,
    sessions: list,
    initial_capital: float,
    constraints: PropFirmConstraintConfig,
    instrument: InstrumentDetails,
    execution_model: ExecutionModel,
    rolling_window_days: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    daily = daily_results.copy() if not daily_results.empty else _rebuild_daily_results_from_trades(trades, sessions, initial_capital)
    daily = daily.sort_values("session_date").reset_index(drop=True)
    trades = _ensure_trade_risk(trades, instrument=instrument, execution_model=execution_model)
    base = compute_metrics(trades, signal_df=signal_df, session_dates=sessions, initial_capital=initial_capital)

    net_pnl = pd.to_numeric(trades.get("net_pnl_usd"), errors="coerce").fillna(0.0)
    gross_pnl = pd.to_numeric(trades.get("pnl_usd"), errors="coerce").fillna(0.0)
    daily_pnl = pd.to_numeric(daily.get("daily_pnl_usd"), errors="coerce").fillna(0.0)
    daily_trade_count = pd.to_numeric(daily.get("daily_trade_count"), errors="coerce").fillna(0.0)
    equity = initial_capital + daily_pnl.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0
    annual_return = float((daily_pnl / max(initial_capital, 1.0)).mean() * 252.0) if len(daily_pnl) > 0 else 0.0
    calmar = _safe_ratio(annual_return, abs(max_drawdown / initial_capital) if max_drawdown < 0 and initial_capital > 0 else 0.0, default=np.inf)
    return_over_max_drawdown = _safe_ratio(float(net_pnl.sum()), abs(max_drawdown), default=np.inf)
    drawdown_episodes = _drawdown_episode_table(daily, initial_capital=initial_capital)
    recovery_durations = pd.to_numeric(drawdown_episodes.get("duration_sessions"), errors="coerce").dropna()
    rolling_table = _rolling_validation_table(daily, trades, initial_capital=initial_capital, window_days=rolling_window_days)
    hourly_table = _hourly_validation_table(bar_results if bar_results is not None else pd.DataFrame(), trades)
    weekday_table = _weekday_expectancy_table(daily)
    long_short_table = build_long_short_stats(trades)
    trade_loss_lengths = _run_lengths(net_pnl < 0)
    day_loss_lengths = _run_lengths(daily_pnl < 0)
    concentration_days, concentration_weeks, concentration_summary = _concentration_tables(daily)
    prop_summary, prop_thresholds = _prop_frequency_summary(trades=trades, daily_results=daily, initial_capital=initial_capital, constraints=constraints)

    if bar_results is not None and not bar_results.empty:
        bar_view = bar_results.copy()
        notional = (
            pd.to_numeric(bar_view.get("quantity"), errors="coerce").fillna(0.0)
            * pd.to_numeric(bar_view.get("open"), errors="coerce").fillna(0.0)
            * float(instrument.point_value_usd)
        )
        avg_exposure_pct = float((notional.abs() / max(initial_capital, 1.0)).mean())
    else:
        avg_exposure_pct = float(pd.to_numeric(trades.get("holding_minutes"), errors="coerce").fillna(0.0).sum() / max(len(sessions) * 390.0, 1.0))

    long_count = int((trades.get("direction") == "long").sum()) if not trades.empty else 0
    short_count = int((trades.get("direction") == "short").sum()) if not trades.empty else 0
    total_count = max(int(len(trades)), 1)

    summary = {
        "total_trades": int(len(trades)),
        "n_trades": int(len(trades)),
        "net_pnl": float(net_pnl.sum()),
        "gross_pnl": float(gross_pnl.sum()),
        "gross_profit": float(gross_pnl[gross_pnl > 0].sum()),
        "gross_loss_abs": abs(float(gross_pnl[gross_pnl < 0].sum())),
        "sharpe_ratio": float(base.get("sharpe_ratio", 0.0)),
        "sortino_ratio": _sortino_ratio(daily_pnl, initial_capital),
        "profit_factor": float(base.get("profit_factor", 0.0)),
        "hit_rate": float(base.get("win_rate", 0.0)),
        "expectancy_per_trade": float(base.get("expectancy", 0.0)),
        "expectancy_per_day": float(daily_pnl.mean()) if len(daily_pnl) > 0 else 0.0,
        "avg_win": float(base.get("avg_win", 0.0)),
        "avg_loss": float(base.get("avg_loss", 0.0)),
        "avg_gain_loss_ratio": _safe_ratio(float(base.get("avg_win", 0.0)), abs(float(base.get("avg_loss", 0.0))), default=np.inf),
        "max_drawdown": max_drawdown,
        "return_over_max_drawdown": return_over_max_drawdown,
        "calmar_ratio": calmar,
        "ulcer_index": _ulcer_index(equity, peak),
        "annualized_vol": _annualized_vol(daily_pnl, initial_capital),
        "mean_trades_per_day": float(daily_trade_count.mean()) if len(daily_trade_count) > 0 else 0.0,
        "median_trades_per_day": float(daily_trade_count.median()) if len(daily_trade_count) > 0 else 0.0,
        "max_trades_per_day": int(daily_trade_count.max()) if len(daily_trade_count) > 0 else 0,
        "avg_time_in_position_min": float(pd.to_numeric(trades.get("holding_minutes"), errors="coerce").mean()) if not trades.empty else 0.0,
        "avg_exposure_pct": avg_exposure_pct,
        "long_trade_share": float(long_count / total_count) if total_count > 0 else 0.0,
        "short_trade_share": float(short_count / total_count) if total_count > 0 else 0.0,
        "worst_losing_trades_streak": int(max(trade_loss_lengths, default=0)),
        "worst_losing_days_streak": int(max(day_loss_lengths, default=0)),
        "median_recovery_days": float(recovery_durations.median()) if not recovery_durations.empty else np.nan,
        "max_recovery_days": float(recovery_durations.max()) if not recovery_durations.empty else np.nan,
        "rolling_sharpe_20d_median": float(pd.to_numeric(rolling_table.get("rolling_sharpe_20d"), errors="coerce").median()) if not rolling_table.empty else np.nan,
        "rolling_expectancy_20d_median": float(pd.to_numeric(rolling_table.get("rolling_expectancy_20d"), errors="coerce").median()) if not rolling_table.empty else np.nan,
        "rolling_hit_rate_20d_median": float(pd.to_numeric(rolling_table.get("rolling_hit_rate_20d"), errors="coerce").median()) if not rolling_table.empty else np.nan,
        **concentration_summary,
        **prop_summary,
    }
    tables = {
        "monthly_pnl": _monthly_pnl_table(daily),
        "quarterly_pnl": _quarterly_pnl_table(daily),
        "rolling_20d_metrics": rolling_table,
        "losing_streaks_trades": _streak_distribution(trade_loss_lengths, "trade_losses"),
        "losing_streaks_days": _streak_distribution(day_loss_lengths, "day_losses"),
        "drawdown_episodes": drawdown_episodes,
        "intraday_by_hour": hourly_table,
        "weekday_expectancy": weekday_table,
        "long_short_breakdown": long_short_table,
        "concentration_days": concentration_days,
        "concentration_weeks": concentration_weeks,
        "prop_frequency_table": prop_thresholds,
    }
    return summary, tables


def _metrics_row(prefix: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": summary.get(key, np.nan) for key in SUMMARY_METRIC_COLUMNS}


def _build_scope_summary_table(
    trades: pd.DataFrame,
    daily_results: pd.DataFrame,
    bar_results: pd.DataFrame,
    signal_df: pd.DataFrame,
    sessions_all: list,
    is_sessions: list,
    oos_sessions: list,
    initial_capital: float,
    constraints: PropFirmConstraintConfig,
    instrument: InstrumentDetails,
    execution_model: ExecutionModel,
    rolling_window_days: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, dict[str, pd.DataFrame]]]:
    summaries: dict[str, dict[str, Any]] = {}
    tables_by_scope: dict[str, dict[str, pd.DataFrame]] = {}
    for label, sessions in (("overall", sessions_all), ("is", is_sessions), ("oos", oos_sessions)):
        sub_trades = _subset_frame_by_sessions(trades, sessions)
        sub_daily = _subset_frame_by_sessions(daily_results, sessions)
        sub_bar = _subset_frame_by_sessions(bar_results, sessions)
        sub_signal = _subset_frame_by_sessions(signal_df, sessions)
        summary, tables = _compute_scope_summary(
            trades=sub_trades,
            daily_results=sub_daily,
            bar_results=sub_bar,
            signal_df=sub_signal,
            sessions=sessions,
            initial_capital=initial_capital,
            constraints=constraints,
            instrument=instrument,
            execution_model=execution_model,
            rolling_window_days=rolling_window_days,
        )
        summaries[label] = summary
        tables_by_scope[label] = tables
    summary_table = pd.DataFrame([{"scope": scope, **summary} for scope, summary in summaries.items()])
    return summary_table, summaries, tables_by_scope


def _load_source_run_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_reference_variant(reference_variant_name: str) -> VWAPVariantConfig:
    for variant in build_default_vwap_variants():
        if variant.name == reference_variant_name:
            return variant
    raise ValueError(f"Unknown reference variant '{reference_variant_name}'.")


def build_default_validation_spec(dataset_path: Path | None = None) -> ValidationSpec:
    resolved = dataset_path or resolve_default_vwap_dataset("MNQ")
    return ValidationSpec(dataset_path=resolved, prop_constraints=build_default_prop_constraints())


def _prepare_feature_frame(spec: ValidationSpec, atr_windows: list[int]) -> pd.DataFrame:
    raw = load_ohlcv_file(spec.dataset_path)
    clean = clean_ohlcv(raw)
    return prepare_vwap_feature_frame(
        clean,
        session_start=spec.session_start,
        session_end=spec.session_end,
        atr_windows=atr_windows,
    )


def _run_full_nominal(feature_df: pd.DataFrame, spec: ValidationSpec) -> NominalEvaluation:
    symbol = infer_symbol_from_dataset_path(spec.dataset_path)
    reference_variant = _resolve_reference_variant(spec.reference_variant_name)
    signal_df = build_vwap_signal_frame(feature_df, reference_variant)
    execution_model, instrument = build_execution_model_for_profile(symbol=symbol, profile_name=reference_variant.execution_profile)
    result = run_vwap_backtest(signal_df, reference_variant, execution_model, instrument)
    trades = _ensure_trade_risk(result.trades, instrument=instrument, execution_model=execution_model)
    all_sessions = sorted(pd.to_datetime(feature_df["session_date"]).dt.date.unique())
    is_sessions, oos_sessions = _split_sessions(all_sessions, spec.is_fraction)
    summary_by_scope, _, tables_by_scope = _build_scope_summary_table(
        trades=trades,
        daily_results=result.daily_results,
        bar_results=result.bar_results,
        signal_df=signal_df,
        sessions_all=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        initial_capital=reference_variant.initial_capital_usd,
        constraints=spec.prop_constraints,
        instrument=instrument,
        execution_model=execution_model,
        rolling_window_days=spec.rolling_window_days,
    )
    overall_tables = tables_by_scope["overall"]
    overall_tables["metrics_by_scope"] = summary_by_scope
    return NominalEvaluation(
        variant=reference_variant,
        signal_df=signal_df,
        result=result,
        trades=trades,
        daily_results=result.daily_results.copy(),
        bar_results=result.bar_results.copy(),
        instrument=instrument,
        execution_model=execution_model,
        all_sessions=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        summary_by_scope=summary_by_scope,
        tables=overall_tables,
    )


def _evaluate_variant_metrics(
    feature_df: pd.DataFrame,
    variant: VWAPVariantConfig,
    symbol: str,
    all_sessions: list,
    is_sessions: list,
    oos_sessions: list,
    spec: ValidationSpec,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cache_key = _variant_cache_key(variant)
    if cache_key in cache:
        return cache[cache_key]

    signal_df = build_vwap_signal_frame(feature_df, variant)
    execution_model, instrument = build_execution_model_for_profile(symbol=symbol, profile_name=variant.execution_profile)
    result = run_vwap_backtest(signal_df, variant, execution_model, instrument)
    trades = _ensure_trade_risk(result.trades, instrument=instrument, execution_model=execution_model)
    summary_table, summaries, _ = _build_scope_summary_table(
        trades=trades,
        daily_results=result.daily_results,
        bar_results=result.bar_results,
        signal_df=signal_df,
        sessions_all=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        initial_capital=variant.initial_capital_usd,
        constraints=spec.prop_constraints,
        instrument=instrument,
        execution_model=execution_model,
        rolling_window_days=spec.rolling_window_days,
    )
    row = {
        "variant_name": variant.name,
        "time_windows": _time_window_label(variant.time_windows),
        "slope_lookback": variant.slope_lookback,
        "slope_threshold": variant.slope_threshold,
        "atr_buffer": variant.atr_buffer,
        "stop_buffer": variant.stop_buffer if variant.stop_buffer is not None else variant.atr_buffer,
        "pullback_lookback": variant.pullback_lookback,
        "compression_length": variant.compression_length,
        "confirmation_threshold": variant.confirmation_threshold,
        "max_trades_per_day": variant.max_trades_per_day,
        "max_losses_per_day": variant.max_losses_per_day,
        "daily_stop_threshold_usd": variant.daily_stop_threshold_usd,
        "exit_on_vwap_recross": variant.exit_on_vwap_recross,
        **_metrics_row("overall", summaries["overall"]),
        **_metrics_row("is", summaries["is"]),
        **_metrics_row("oos", summaries["oos"]),
    }
    out = {"row": row, "summary_by_scope": summary_table}
    cache[cache_key] = out
    return out


def _reference_spec_markdown(spec: ValidationSpec, nominal: NominalEvaluation, source_metadata: dict[str, Any]) -> str:
    variant = nominal.variant
    live_param_lines = [
        f"- Strategy name: `{variant.name}`",
        f"- Time windows: `{_time_window_label(variant.time_windows)}`",
        f"- Slope lookback / threshold: `{variant.slope_lookback}` / `{variant.slope_threshold}`",
        f"- ATR period / buffer / stop buffer: `{variant.atr_period}` / `{variant.atr_buffer}` / `{variant.stop_buffer}`",
        f"- Pullback lookback: `{variant.pullback_lookback}`",
        f"- Confirmation threshold: `{variant.confirmation_threshold}` ATR above/below `prev_high` / `prev_low`.",
        "- Pullback definition: at least one counter-trend close inside the last pullback window, while the pullback extreme stays within the VWAP regime buffer.",
        "- Confirmation definition: close-based continuation through `prev_high` / `prev_low`, executed on the next bar open.",
        "- Stop logic: pullback extreme +/- `stop_buffer * ATR`.",
        f"- Exit logic: VWAP recross = `{variant.exit_on_vwap_recross}`, plus structural stop, plus forced session close.",
        f"- Max trades per day: `{variant.max_trades_per_day}`",
        f"- Daily kill switches: `max_losses_per_day={variant.max_losses_per_day}`, `daily_stop_threshold_usd={variant.daily_stop_threshold_usd}`",
        f"- Sizing: `{variant.quantity_mode}`, fixed quantity `{variant.fixed_quantity}`, risk per trade `{variant.risk_per_trade_pct}`",
        f"- Costs: commission `{nominal.execution_model.commission_per_side_usd}` USD / side, slippage `{nominal.execution_model.slippage_ticks}` tick(s) / side.",
        f"- Session assumptions: RTH `[09:30, 16:00)`, flat overnight, dataset `{spec.dataset_path.name}`.",
    ]
    inactive_lines = [
        "- `compression_length` is present in the config object but is not consumed by `generate_pullback_continuation_signals`; it is documented as inactive for this variant.",
        "- `use_partial_exit`, `partial_exit_r_multiple`, and `keep_runner_until_close` are currently inactive in the VWAP discrete backtester and are excluded from robustness conclusions.",
        "- The source discovery run selected `vwap_pullback_continuation` as best variant, but that legacy run used same-bar execution for close-based discrete entries. Validation reruns therefore use the corrected next-open semantics and should be treated as the only defendable evidence.",
    ]
    provenance_lines = []
    if source_metadata:
        provenance_lines.extend(
            [
                f"- Source run metadata: `{spec.source_run_metadata_path}`",
                f"- Source best variant: `{source_metadata.get('best_variant_name')}`",
                f"- Source run timestamp: `{source_metadata.get('run_timestamp')}`",
            ]
        )
    return "\n".join(
        [
            "# Reference Spec",
            "",
            "## Provenance",
            "",
            *provenance_lines,
            "",
            "## Audited Strategy Definition",
            "",
            *live_param_lines,
            "",
            "## Audit Warnings",
            "",
            *inactive_lines,
            "",
        ]
    ).rstrip() + "\n"


def _reference_config_payload(spec: ValidationSpec, nominal: NominalEvaluation, source_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_path": str(spec.dataset_path),
        "reference_variant": _variant_dict(nominal.variant),
        "symbol": nominal.instrument.symbol,
        "instrument": asdict(nominal.instrument),
        "execution_model": {
            "commission_per_side_usd": nominal.execution_model.commission_per_side_usd,
            "slippage_ticks": nominal.execution_model.slippage_ticks,
            "tick_size": nominal.execution_model.tick_size,
        },
        "session_assumptions": {
            "session_start": spec.session_start,
            "session_end": spec.session_end,
            "flat_overnight": True,
            "discrete_entries_execute_next_open": True,
        },
        "audit_notes": {
            "source_best_run_metadata": str(spec.source_run_metadata_path) if spec.source_run_metadata_path is not None else None,
            "source_best_variant_name": source_metadata.get("best_variant_name"),
            "legacy_discovery_run_had_same_bar_execution": True,
            "compression_length_inactive_for_pullback_continuation": True,
            "partial_exit_logic_active": False,
        },
    }


def _export_nominal(output_dir: Path, spec: ValidationSpec, nominal: NominalEvaluation, source_metadata: dict[str, Any]) -> dict[str, Path]:
    nominal_dir = output_dir / "nominal"
    tables_dir = nominal_dir / "tables"
    nominal_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    reference_config_path = nominal_dir / "reference_config.json"
    reference_spec_path = nominal_dir / "reference_spec.md"
    metrics_by_scope_path = nominal_dir / "metrics_summary_by_scope.csv"
    trades_path = nominal_dir / "trades.csv"
    daily_path = nominal_dir / "daily_results.csv"
    equity_curve_path = nominal_dir / "equity_curve.csv"

    reference_spec_path.write_text(_reference_spec_markdown(spec, nominal, source_metadata), encoding="utf-8")
    _json_dump(reference_config_path, _reference_config_payload(spec, nominal, source_metadata))
    nominal.summary_by_scope.to_csv(metrics_by_scope_path, index=False)
    nominal.trades.to_csv(trades_path, index=False)
    nominal.daily_results.to_csv(daily_path, index=False)
    build_equity_curve(nominal.trades, initial_capital=nominal.variant.initial_capital_usd).to_csv(equity_curve_path, index=False)
    for name, table in nominal.tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)

    return {
        "reference_config_json": reference_config_path,
        "reference_spec_md": reference_spec_path,
        "nominal_metrics_csv": metrics_by_scope_path,
        "nominal_trades_csv": trades_path,
        "nominal_daily_csv": daily_path,
        "nominal_equity_curve_csv": equity_curve_path,
    }


def _apply_trade_controls_overlay(
    trades: pd.DataFrame,
    max_trades_per_day: int | None = None,
    max_losses_per_day: int | None = None,
    daily_stop_threshold_usd: float | None = None,
) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()

    kept_rows: list[pd.DataFrame] = []
    for _, session_df in trades.sort_values(["session_date", "entry_time"]).groupby("session_date", sort=True):
        day_pnl = 0.0
        day_losses = 0
        day_kept = 0
        selected_idx: list[int] = []
        for idx, trade in session_df.iterrows():
            if max_trades_per_day is not None and day_kept >= int(max_trades_per_day):
                break
            if max_losses_per_day is not None and day_losses >= int(max_losses_per_day):
                break
            if daily_stop_threshold_usd is not None and day_pnl <= -float(daily_stop_threshold_usd):
                break
            selected_idx.append(idx)
            pnl = float(trade["net_pnl_usd"])
            day_pnl += pnl
            day_kept += 1
            if pnl < 0:
                day_losses += 1
        if selected_idx:
            kept_rows.append(session_df.loc[selected_idx].copy())

    if not kept_rows:
        return trades.iloc[0:0].copy()
    out = pd.concat(kept_rows, ignore_index=True)
    out["trade_id"] = np.arange(1, len(out) + 1)
    return out


def _apply_cost_stress_overlay(
    trades: pd.DataFrame,
    scenario: StressScenario,
    instrument: InstrumentDetails,
    execution_model: ExecutionModel,
    session_start: str,
) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()

    out = trades.copy()
    qty = pd.to_numeric(out["quantity"], errors="coerce").fillna(0.0)
    entry_time = pd.to_datetime(out["entry_time"], utc=False, errors="coerce")

    delta_commission = 2.0 * execution_model.commission_per_side_usd * max(scenario.commission_multiplier - 1.0, 0.0) * qty
    delta_slippage_ticks = execution_model.slippage_ticks * max(scenario.slippage_multiplier - 1.0, 0.0)
    delta_slippage = 2.0 * delta_slippage_ticks * instrument.tick_size * instrument.point_value_usd * qty
    delta_entry_penalty = scenario.entry_penalty_ticks * instrument.tick_size * instrument.point_value_usd * qty

    start_hour, start_minute = _parse_time_label(session_start)
    open_cutoff = entry_time.dt.normalize() + pd.to_timedelta(start_hour, unit="h") + pd.to_timedelta(start_minute + scenario.open_penalty_minutes, unit="m")
    in_open_window = entry_time <= open_cutoff
    delta_open_penalty = np.where(in_open_window.fillna(False), scenario.open_penalty_ticks * instrument.tick_size * instrument.point_value_usd * qty, 0.0)
    total_extra_slippage = delta_slippage + delta_entry_penalty + delta_open_penalty
    total_extra_cost = total_extra_slippage + delta_commission

    out["pnl_usd"] = pd.to_numeric(out["pnl_usd"], errors="coerce").fillna(0.0) - total_extra_slippage
    out["gross_pnl_usd"] = pd.to_numeric(out.get("gross_pnl_usd"), errors="coerce").fillna(0.0) - total_extra_slippage
    out["fees"] = pd.to_numeric(out["fees"], errors="coerce").fillna(0.0) + delta_commission
    out["stress_cost_adjustment_usd"] = total_extra_cost
    out["net_pnl_usd"] = pd.to_numeric(out["net_pnl_usd"], errors="coerce").fillna(0.0) - total_extra_cost
    out["r_multiple"] = np.where(
        pd.to_numeric(out["trade_risk_usd"], errors="coerce") > 0,
        pd.to_numeric(out["net_pnl_usd"], errors="coerce") / pd.to_numeric(out["trade_risk_usd"], errors="coerce"),
        np.nan,
    )
    return out


def _stress_scenarios() -> list[StressScenario]:
    return [
        StressScenario(name="nominal", notes="Base run."),
        StressScenario(name="slippage_x2", slippage_multiplier=2.0, notes="Slippage doubled."),
        StressScenario(name="slippage_x3", slippage_multiplier=3.0, notes="Slippage tripled."),
        StressScenario(name="commission_plus_25pct", commission_multiplier=1.25, notes="Commission +25%."),
        StressScenario(name="commission_plus_50pct", commission_multiplier=1.50, notes="Commission +50%."),
        StressScenario(name="combined_x2_plus25", slippage_multiplier=2.0, commission_multiplier=1.25, notes="Slippage x2 + commission +25%."),
        StressScenario(name="combined_x3_plus50", slippage_multiplier=3.0, commission_multiplier=1.50, notes="Slippage x3 + commission +50%."),
        StressScenario(name="open_penalty_early15m", open_penalty_ticks=1.0, open_penalty_minutes=15, notes="Extra one-tick entry penalty during the first 15 minutes."),
        StressScenario(name="entry_penalty_1tick", entry_penalty_ticks=1.0, notes="Extra one-tick entry penalty on every trade."),
    ]


def _stress_verdict(stress_df: pd.DataFrame) -> str:
    stressed = stress_df.loc[stress_df["scenario"] != "nominal"].copy()
    if stressed.empty:
        return "undetermined"
    oos_positive = (stressed["oos_net_pnl"] > 0).mean()
    oos_pf_ok = (stressed["oos_profit_factor"] > 1.0).mean()
    if oos_positive >= 0.85 and oos_pf_ok >= 0.85:
        return "strategie robuste aux couts"
    if oos_positive >= 0.50 and oos_pf_ok >= 0.50:
        return "strategie moderement fragile"
    return "strategie tres fragile a la microstructure"


def _run_stress_suite(output_dir: Path, spec: ValidationSpec, nominal: NominalEvaluation) -> dict[str, Path]:
    stress_dir = output_dir / "stress"
    stress_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    nominal_oos = nominal.summary_by_scope.loc[nominal.summary_by_scope["scope"] == "oos"].iloc[0]
    for scenario in _stress_scenarios():
        stressed_trades = _apply_cost_stress_overlay(
            nominal.trades,
            scenario=scenario,
            instrument=nominal.instrument,
            execution_model=nominal.execution_model,
            session_start=spec.session_start,
        )
        stressed_daily = _rebuild_daily_results_from_trades(stressed_trades, nominal.all_sessions, nominal.variant.initial_capital_usd)
        summary_table, summaries, _ = _build_scope_summary_table(
            trades=stressed_trades,
            daily_results=stressed_daily,
            bar_results=pd.DataFrame(),
            signal_df=nominal.signal_df,
            sessions_all=nominal.all_sessions,
            is_sessions=nominal.is_sessions,
            oos_sessions=nominal.oos_sessions,
            initial_capital=nominal.variant.initial_capital_usd,
            constraints=spec.prop_constraints,
            instrument=nominal.instrument,
            execution_model=nominal.execution_model,
            rolling_window_days=spec.rolling_window_days,
        )
        oos = summaries["oos"]
        rows.append(
            {
                "scenario": scenario.name,
                "notes": scenario.notes,
                "overall_net_pnl": summaries["overall"]["net_pnl"],
                "overall_profit_factor": summaries["overall"]["profit_factor"],
                "overall_sharpe_ratio": summaries["overall"]["sharpe_ratio"],
                "overall_max_drawdown": summaries["overall"]["max_drawdown"],
                "oos_net_pnl": oos["net_pnl"],
                "oos_profit_factor": oos["profit_factor"],
                "oos_sharpe_ratio": oos["sharpe_ratio"],
                "oos_max_drawdown": oos["max_drawdown"],
                "delta_oos_net_pnl_vs_nominal": oos["net_pnl"] - float(nominal_oos["net_pnl"]),
                "delta_oos_profit_factor_vs_nominal": oos["profit_factor"] - float(nominal_oos["profit_factor"]),
                "delta_oos_sharpe_vs_nominal": oos["sharpe_ratio"] - float(nominal_oos["sharpe_ratio"]),
            }
        )
        summary_table.to_csv(stress_dir / f"{scenario.name}_metrics_by_scope.csv", index=False)

    stress_df = pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)
    stress_csv = stress_dir / "stress_test_summary.csv"
    stress_md = stress_dir / "stress_test_report.md"
    stress_df.to_csv(stress_csv, index=False)
    stress_md.write_text(
        "\n".join(
            [
                "# Stress Test Report",
                "",
                f"- Verdict: `{_stress_verdict(stress_df)}`",
                "",
                "```text",
                stress_df.to_string(index=False),
                "```",
                "",
                "- Cost stresses are applied as path-preserving overlays on the corrected nominal trade log.",
                "- This is exact for fixed cost changes and intentionally conservative for extra entry penalties.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"stress_summary_csv": stress_csv, "stress_report_md": stress_md}


def _with_open_window_end(variant: VWAPVariantConfig, end_time: str, session_start: str, session_end: str) -> VWAPVariantConfig:
    if end_time == session_end:
        return replace(variant, time_windows=())
    return replace(variant, time_windows=(TimeWindow(session_start, end_time),))


def _pair_metric_panel(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    ref_x: Any,
    ref_y: Any,
    output_path: Path,
) -> None:
    metrics = [
        ("is_sharpe_ratio", "IS Sharpe", "RdYlGn", 0.0, False),
        ("is_profit_factor", "IS PF", "RdYlGn", 1.0, False),
        ("is_net_pnl", "IS Net PnL", "RdYlGn", 0.0, False),
        ("is_max_drawdown", "IS Max DD", "RdYlGn_r", None, True),
        ("oos_sharpe_ratio", "OOS Sharpe", "RdYlGn", 0.0, False),
        ("oos_profit_factor", "OOS PF", "RdYlGn", 1.0, False),
        ("oos_net_pnl", "OOS Net PnL", "RdYlGn", 0.0, False),
        ("oos_max_drawdown", "OOS Max DD", "RdYlGn_r", None, True),
    ]

    x_levels = list(df[[x_col, f"{x_col}_sort"]].drop_duplicates().sort_values(f"{x_col}_sort")[x_col])
    y_levels = list(df[[y_col, f"{y_col}_sort"]].drop_duplicates().sort_values(f"{y_col}_sort")[y_col])
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for ax, (metric_col, title, cmap, center, reverse) in zip(axes.flatten(), metrics):
        pivot = df.pivot_table(index=y_col, columns=x_col, values=metric_col, aggfunc="mean").reindex(index=y_levels, columns=x_levels)
        trades_col = metric_col.replace("_sharpe_ratio", "_total_trades").replace("_profit_factor", "_total_trades").replace("_net_pnl", "_total_trades").replace("_max_drawdown", "_total_trades")
        trades_pivot = df.pivot_table(index=y_col, columns=x_col, values=trades_col, aggfunc="mean").reindex(index=y_levels, columns=x_levels)
        values = pivot.to_numpy(dtype=float)
        cmap_obj = plt.get_cmap(cmap)
        if reverse:
            cmap_obj = cmap_obj.reversed()
        if center is not None and np.isfinite(values).any():
            vmax = float(np.nanmax(np.abs(values - center)))
            vmax = vmax if vmax > 0 else 1.0
            im = ax.imshow(values, cmap=cmap_obj, aspect="auto", vmin=center - vmax, vmax=center + vmax)
        else:
            im = ax.imshow(values, cmap=cmap_obj, aspect="auto")
        ax.set_xticks(range(len(x_levels)))
        ax.set_xticklabels(x_levels, rotation=35, ha="right")
        ax.set_yticks(range(len(y_levels)))
        ax.set_yticklabels(y_levels)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        for i, y_value in enumerate(y_levels):
            for j, x_value in enumerate(x_levels):
                value = values[i, j]
                trades = trades_pivot.iloc[i, j] if not trades_pivot.empty else np.nan
                if metric_col.endswith("profit_factor"):
                    text = "nan" if not math.isfinite(value) else f"{value:.2f}\nN={int(round(trades)) if pd.notna(trades) else 0}"
                elif metric_col.endswith("net_pnl") or metric_col.endswith("max_drawdown"):
                    text = "nan" if not math.isfinite(value) else f"{value:.0f}\nN={int(round(trades)) if pd.notna(trades) else 0}"
                else:
                    text = "nan" if not math.isfinite(value) else f"{value:.2f}\nN={int(round(trades)) if pd.notna(trades) else 0}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")
                if x_value == ref_x and y_value == ref_y:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.4))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _heatmap_topology_readout(df: pd.DataFrame, x_col: str, y_col: str, ref_x: Any, ref_y: Any) -> dict[str, Any]:
    reference = df.loc[(df[x_col] == ref_x) & (df[y_col] == ref_y)].copy()
    if reference.empty:
        return {"verdict": "undetermined", "comment": "Reference cell missing."}
    ref = reference.iloc[0]
    x_map = {value: idx for idx, value in enumerate(df[[x_col, f"{x_col}_sort"]].drop_duplicates().sort_values(f"{x_col}_sort")[x_col])}
    y_map = {value: idx for idx, value in enumerate(df[[y_col, f"{y_col}_sort"]].drop_duplicates().sort_values(f"{y_col}_sort")[y_col])}
    neighbors = df.loc[
        df[x_col].map(x_map).between(x_map[ref_x] - 1, x_map[ref_x] + 1)
        & df[y_col].map(y_map).between(y_map[ref_y] - 1, y_map[ref_y] + 1)
    ].copy()
    if neighbors.empty:
        return {"verdict": "undetermined", "comment": "Reference neighborhood empty."}
    stable_share = float(((neighbors["oos_profit_factor"] > 1.0) & (neighbors["oos_net_pnl"] > 0) & (neighbors["oos_sharpe_ratio"] > 0)).mean())
    ref_rank = float(df["oos_sharpe_ratio"].rank(method="min", ascending=False).loc[reference.index].iloc[0] / max(len(df), 1))
    if stable_share >= 0.60:
        verdict = "stable localement"
        comment = "La case de reference reste entouree d'un plateau OOS globalement positif."
    elif stable_share >= 0.30:
        verdict = "moderetement stable"
        comment = "Le voisinage n'est pas vide, mais la robustesse locale reste inegale."
    else:
        verdict = "instable / pic etroit"
        comment = "La reference n'est soutenue que par peu de voisins OOS credibles."
    if ref_rank <= 0.15 and stable_share < 0.30:
        comment += " L'optimum apparait isole."
    elif ref_rank <= 0.30 and stable_share >= 0.60:
        comment += " L'optimum n'est pas strictement isole."
    return {"verdict": verdict, "stable_neighbor_share": stable_share, "reference_sharpe_rank_pct": ref_rank, "comment": comment}


def _run_local_robustness_suite(output_dir: Path, spec: ValidationSpec, nominal: NominalEvaluation, feature_df: pd.DataFrame) -> dict[str, Path]:
    local_dir = output_dir / "local_robustness"
    heatmaps_dir = local_dir / "heatmaps"
    local_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    symbol = infer_symbol_from_dataset_path(spec.dataset_path)
    cache: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []

    def add_variant_row(tag: str, variant: VWAPVariantConfig, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        evaluated = _evaluate_variant_metrics(
            feature_df=feature_df,
            variant=variant,
            symbol=symbol,
            all_sessions=nominal.all_sessions,
            is_sessions=nominal.is_sessions,
            oos_sessions=nominal.oos_sessions,
            spec=spec,
            cache=cache,
        )["row"].copy()
        evaluated["analysis_tag"] = tag
        if extra:
            evaluated.update(extra)
        all_rows.append(evaluated)
        return evaluated

    reference_stop_buffer = nominal.variant.stop_buffer if nominal.variant.stop_buffer is not None else nominal.variant.atr_buffer
    one_d_variants = [
        ("slope_lookback", replace(nominal.variant, slope_lookback=value), {"param_name": "slope_lookback", "param_value": value})
        for value in (3, nominal.variant.slope_lookback, 8)
    ] + [
        ("stop_buffer", replace(nominal.variant, stop_buffer=value), {"param_name": "stop_buffer", "param_value": value})
        for value in (0.20, reference_stop_buffer, 0.40)
    ] + [
        ("exit_on_vwap_recross", replace(nominal.variant, exit_on_vwap_recross=value), {"param_name": "exit_on_vwap_recross", "param_value": value})
        for value in (True, False)
    ]
    for tag, variant, extra in one_d_variants:
        add_variant_row(tag, variant, extra)

    heatmap_1_rows: list[dict[str, Any]] = []
    for slope in (0.00, 0.01, 0.02):
        for atr_buffer in (0.20, nominal.variant.atr_buffer, 0.40):
            heatmap_1_rows.append(
                add_variant_row(
                    "heatmap_slope_atr",
                    replace(nominal.variant, slope_threshold=slope, atr_buffer=atr_buffer),
                    {
                        "x_value": f"{slope:.2f}",
                        "y_value": f"{atr_buffer:.2f}",
                        "x_value_sort": slope,
                        "y_value_sort": atr_buffer,
                        "pair_name": "slope_threshold_x_atr_buffer",
                    },
                )
            )

    heatmap_2_rows: list[dict[str, Any]] = []
    for pullback in (6, nominal.variant.pullback_lookback, 10):
        for threshold in (0.00, nominal.variant.confirmation_threshold, 0.10):
            heatmap_2_rows.append(
                add_variant_row(
                    "heatmap_pullback_confirmation",
                    replace(nominal.variant, pullback_lookback=pullback, confirmation_threshold=threshold),
                    {
                        "x_value": f"{threshold:.2f}",
                        "y_value": str(pullback),
                        "x_value_sort": threshold,
                        "y_value_sort": pullback,
                        "pair_name": "pullback_length_x_confirmation_threshold",
                    },
                )
            )

    heatmap_3_rows: list[dict[str, Any]] = []
    time_rows: dict[str, pd.DataFrame] = {}
    for end_time in ("11:30", "15:00", "16:00"):
        time_variant = _with_open_window_end(nominal.variant, f"{end_time}:00", spec.session_start, spec.session_end)
        signal_df = build_vwap_signal_frame(feature_df, time_variant)
        execution_model, instrument = build_execution_model_for_profile(symbol=symbol, profile_name=time_variant.execution_profile)
        result = run_vwap_backtest(signal_df, time_variant, execution_model, instrument)
        time_rows[end_time] = _ensure_trade_risk(result.trades, instrument, execution_model)
        add_variant_row("heatmap_open_window_base", time_variant, {"base_window_end": end_time})

    for end_time, trades in time_rows.items():
        for max_trades in (1, 2, nominal.variant.max_trades_per_day or 3):
            overlay_trades = _apply_trade_controls_overlay(trades, max_trades_per_day=max_trades)
            overlay_daily = _rebuild_daily_results_from_trades(overlay_trades, nominal.all_sessions, nominal.variant.initial_capital_usd)
            _, summaries, _ = _build_scope_summary_table(
                trades=overlay_trades,
                daily_results=overlay_daily,
                bar_results=pd.DataFrame(),
                signal_df=nominal.signal_df,
                sessions_all=nominal.all_sessions,
                is_sessions=nominal.is_sessions,
                oos_sessions=nominal.oos_sessions,
                initial_capital=nominal.variant.initial_capital_usd,
                constraints=spec.prop_constraints,
                instrument=nominal.instrument,
                execution_model=nominal.execution_model,
                rolling_window_days=spec.rolling_window_days,
            )
            row = {
                "analysis_tag": "heatmap_open_window",
                "pair_name": "open_window_end_x_max_trades",
                "x_value": end_time,
                "y_value": str(max_trades),
                "x_value_sort": _parse_time_label(end_time),
                "y_value_sort": max_trades,
                **_metrics_row("overall", summaries["overall"]),
                **_metrics_row("is", summaries["is"]),
                **_metrics_row("oos", summaries["oos"]),
            }
            heatmap_3_rows.append(row)
            all_rows.append(row)

    heatmap_4_rows: list[dict[str, Any]] = []
    for daily_stop in (None, 500.0, 1000.0):
        for max_losses in (None, 2, 3):
            overlay_trades = _apply_trade_controls_overlay(nominal.trades, max_losses_per_day=max_losses, daily_stop_threshold_usd=daily_stop)
            overlay_daily = _rebuild_daily_results_from_trades(overlay_trades, nominal.all_sessions, nominal.variant.initial_capital_usd)
            _, summaries, _ = _build_scope_summary_table(
                trades=overlay_trades,
                daily_results=overlay_daily,
                bar_results=pd.DataFrame(),
                signal_df=nominal.signal_df,
                sessions_all=nominal.all_sessions,
                is_sessions=nominal.is_sessions,
                oos_sessions=nominal.oos_sessions,
                initial_capital=nominal.variant.initial_capital_usd,
                constraints=spec.prop_constraints,
                instrument=nominal.instrument,
                execution_model=nominal.execution_model,
                rolling_window_days=spec.rolling_window_days,
            )
            row = {
                "analysis_tag": "heatmap_daily_controls",
                "pair_name": "daily_stop_threshold_x_consecutive_losses_proxy",
                "x_value": "off" if daily_stop is None else f"{int(daily_stop)}",
                "y_value": "off" if max_losses is None else str(int(max_losses)),
                "x_value_sort": -1 if daily_stop is None else int(daily_stop),
                "y_value_sort": -1 if max_losses is None else int(max_losses),
                **_metrics_row("overall", summaries["overall"]),
                **_metrics_row("is", summaries["is"]),
                **_metrics_row("oos", summaries["oos"]),
            }
            heatmap_4_rows.append(row)
            all_rows.append(row)

    pair_definitions = [
        (pd.DataFrame(heatmap_1_rows), "slope_threshold_x_atr_buffer", "x_value", "y_value", f"{nominal.variant.slope_threshold:.2f}", f"{nominal.variant.atr_buffer:.2f}", "Slope Threshold", "ATR Buffer"),
        (pd.DataFrame(heatmap_2_rows), "pullback_length_x_confirmation_threshold", "x_value", "y_value", f"{nominal.variant.confirmation_threshold:.2f}", str(nominal.variant.pullback_lookback), "Confirmation Threshold", "Pullback Length"),
        (pd.DataFrame(heatmap_3_rows), "open_window_end_x_max_trades", "x_value", "y_value", "16:00", str(nominal.variant.max_trades_per_day or 3), "Open Window End", "Max Trades/Day"),
        (pd.DataFrame(heatmap_4_rows), "daily_stop_threshold_x_consecutive_losses_proxy", "x_value", "y_value", "off", "off", "Daily Stop Threshold", "Loss Count Threshold"),
    ]

    heatmap_readouts: list[dict[str, Any]] = []
    for df, pair_name, x_col, y_col, ref_x, ref_y, x_label, y_label in pair_definitions:
        if df.empty:
            continue
        heatmap_path = heatmaps_dir / f"{pair_name}.png"
        _pair_metric_panel(df=df, x_col=x_col, y_col=y_col, x_label=x_label, y_label=y_label, ref_x=ref_x, ref_y=ref_y, output_path=heatmap_path)
        readout = _heatmap_topology_readout(df, x_col=x_col, y_col=y_col, ref_x=ref_x, ref_y=ref_y)
        readout["pair_name"] = pair_name
        heatmap_readouts.append(readout)

    sensitivity_df = pd.DataFrame(all_rows)
    sensitivity_csv = local_dir / "sensitivity_results.csv"
    readout_csv = local_dir / "heatmap_readouts.csv"
    summary_md = local_dir / "local_robustness_summary.md"
    sensitivity_df.to_csv(sensitivity_csv, index=False)
    pd.DataFrame(heatmap_readouts).to_csv(readout_csv, index=False)

    stable_count = int(sum(item["verdict"] == "stable localement" for item in heatmap_readouts))
    moderate_count = int(sum(item["verdict"] == "moderetement stable" for item in heatmap_readouts))
    overall_verdict = "stable localement" if stable_count >= 3 else "moderetement stable" if stable_count + moderate_count >= 2 else "instable / pic etroit"
    summary_md.write_text(
        "\n".join(
            [
                "# Local Robustness Summary",
                "",
                f"- Overall local-read verdict: `{overall_verdict}`",
                "- Daily-stop vs consecutive-loss heatmap uses `max_losses_per_day` as the closest live proxy for the requested daily consecutive-loss kill-switch.",
                "- Daily-control overlays are conservative path-preserving filters applied on the corrected trade path.",
                "",
                "```text",
                pd.DataFrame(heatmap_readouts).to_string(index=False),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"local_sensitivity_csv": sensitivity_csv, "local_heatmap_readouts_csv": readout_csv, "local_summary_md": summary_md}


def _build_split_rows(nominal: NominalEvaluation, spec: ValidationSpec) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    for fraction in DEFAULT_SPLIT_FRACTIONS:
        is_sessions, oos_sessions = _split_sessions(nominal.all_sessions, fraction)
        _, summaries, _ = _build_scope_summary_table(
            trades=nominal.trades,
            daily_results=nominal.daily_results,
            bar_results=nominal.bar_results,
            signal_df=nominal.signal_df,
            sessions_all=nominal.all_sessions,
            is_sessions=is_sessions,
            oos_sessions=oos_sessions,
            initial_capital=nominal.variant.initial_capital_usd,
            constraints=spec.prop_constraints,
            instrument=nominal.instrument,
            execution_model=nominal.execution_model,
            rolling_window_days=spec.rolling_window_days,
        )
        rows.append(
            {
                "split_name": f"is_{int(fraction * 100)}_oos_{100 - int(fraction * 100)}",
                "is_fraction": fraction,
                "oos_start_date": str(oos_sessions[0]),
                "is_sharpe_ratio": summaries["is"]["sharpe_ratio"],
                "is_profit_factor": summaries["is"]["profit_factor"],
                "is_net_pnl": summaries["is"]["net_pnl"],
                "is_max_drawdown": summaries["is"]["max_drawdown"],
                "is_total_trades": summaries["is"]["total_trades"],
                "oos_sharpe_ratio": summaries["oos"]["sharpe_ratio"],
                "oos_profit_factor": summaries["oos"]["profit_factor"],
                "oos_net_pnl": summaries["oos"]["net_pnl"],
                "oos_max_drawdown": summaries["oos"]["max_drawdown"],
                "oos_total_trades": summaries["oos"]["total_trades"],
            }
        )
    splits = pd.DataFrame(rows).sort_values("is_fraction").reset_index(drop=True)
    positive_rate = float(((splits["oos_net_pnl"] > 0) & (splits["oos_profit_factor"] > 1.0)).mean()) if not splits.empty else 0.0
    verdict = "stable entre splits" if positive_rate >= 0.75 else "acceptable mais variable" if positive_rate >= 0.50 else "trop dependant du split"
    return splits, verdict


def _run_multi_split_suite(output_dir: Path, spec: ValidationSpec, nominal: NominalEvaluation) -> dict[str, Path]:
    split_dir = output_dir / "multi_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    splits, verdict = _build_split_rows(nominal, spec)
    split_csv = split_dir / "split_summary.csv"
    split_md = split_dir / "split_summary.md"
    splits.to_csv(split_csv, index=False)
    split_md.write_text(
        "\n".join(["# Multi-Split Summary", "", f"- Verdict: `{verdict}`", "", "```text", splits.to_string(index=False), "```", ""]),
        encoding="utf-8",
    )
    return {"split_summary_csv": split_csv, "split_summary_md": split_md}


def _outlier_verdict(concentration_summary: pd.Series) -> str:
    top5 = float(concentration_summary.get("top_5_day_contribution_pct", 0.0))
    pnl_without_top5 = float(concentration_summary.get("pnl_excluding_top_5_days", concentration_summary.get("net_pnl", 0.0)))
    if top5 <= 0.50 and pnl_without_top5 > 0:
        return "distribution saine"
    if top5 <= 0.80 and pnl_without_top5 >= 0:
        return "dependance moderee aux outliers"
    return "forte dependance aux meilleurs jours"


def _plot_daily_histogram(daily_results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(pd.to_numeric(daily_results["daily_pnl_usd"], errors="coerce").fillna(0.0), bins=40, color="#6b7280", edgecolor="white")
    ax.set_title("Daily PnL Distribution")
    ax.set_xlabel("Daily PnL (USD)")
    ax.set_ylabel("Days")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_cumulative_contribution(concentration_days: pd.DataFrame, output_path: Path) -> None:
    if concentration_days.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(concentration_days["rank"], concentration_days["cumulative_contribution_pct"], linewidth=1.8)
    ax.axhline(1.0, color="#111827", linewidth=1.0, linestyle="--")
    ax.set_title("Cumulative Contribution of Ranked Days")
    ax.set_xlabel("Day rank (best to worst)")
    ax.set_ylabel("Cumulative contribution / total PnL")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _run_concentration_suite(output_dir: Path, nominal: NominalEvaluation) -> dict[str, Path]:
    concentration_dir = output_dir / "concentration"
    charts_dir = concentration_dir / "charts"
    concentration_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    concentration_days = nominal.tables["concentration_days"]
    concentration_weeks = nominal.tables["concentration_weeks"]
    overall_summary = nominal.summary_by_scope.loc[nominal.summary_by_scope["scope"] == "overall"].iloc[0]
    verdict = _outlier_verdict(overall_summary)
    summary = pd.DataFrame(
        [
            {
                "scope": "overall",
                "top_1_day_contribution_pct": overall_summary["top_1_day_contribution_pct"],
                "top_3_day_contribution_pct": overall_summary["top_3_day_contribution_pct"],
                "top_5_day_contribution_pct": overall_summary["top_5_day_contribution_pct"],
                "top_10_day_contribution_pct": overall_summary["top_10_day_contribution_pct"],
                "pnl_excluding_top_1_day": overall_summary["pnl_excluding_top_1_day"],
                "pnl_excluding_top_3_days": overall_summary["pnl_excluding_top_3_days"],
                "pnl_excluding_top_5_days": overall_summary["pnl_excluding_top_5_days"],
                "pnl_excluding_best_month": overall_summary["pnl_excluding_best_month"],
                "verdict": verdict,
            }
        ]
    )
    summary_csv = concentration_dir / "concentration_summary.csv"
    days_csv = concentration_dir / "concentration_days.csv"
    weeks_csv = concentration_dir / "concentration_weeks.csv"
    summary_md = concentration_dir / "concentration_summary.md"
    summary.to_csv(summary_csv, index=False)
    concentration_days.to_csv(days_csv, index=False)
    concentration_weeks.to_csv(weeks_csv, index=False)
    hist_path = charts_dir / "daily_pnl_histogram.png"
    curve_path = charts_dir / "daily_contribution_curve.png"
    _plot_daily_histogram(nominal.daily_results, hist_path)
    _plot_cumulative_contribution(concentration_days, curve_path)
    summary_md.write_text(
        "\n".join(["# Concentration Summary", "", f"- Verdict: `{verdict}`", "", "```text", summary.to_string(index=False), "```", ""]),
        encoding="utf-8",
    )
    return {
        "concentration_summary_csv": summary_csv,
        "concentration_summary_md": summary_md,
        "concentration_histogram_png": hist_path,
        "concentration_curve_png": curve_path,
    }


def _challenge_scenarios() -> list[ChallengeScenario]:
    return [
        ChallengeScenario("scenario_a_conservative", "A_conservative", 0.10, 2, 2, 750.0, 1500.0, 3000.0, 60, 2, 0.5),
        ChallengeScenario("scenario_b_standard", "B_standard", 0.20, 4, 3, 1000.0, 2000.0, 3000.0, 45, 2, 0.75),
        ChallengeScenario("scenario_c_aggressive_controlled", "C_aggressive_controlled", 0.30, 6, 3, 1250.0, 2250.0, 3000.0, 30, 2, 0.75),
    ]


def _simulate_challenge_path(
    trades: pd.DataFrame,
    scenario: ChallengeScenario,
    account_size_usd: float,
    horizon_days: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ordered = trades.copy()
    ordered["session_date"] = pd.to_datetime(ordered["session_date"]).dt.date
    ordered["entry_time"] = pd.to_datetime(ordered["entry_time"])
    ordered = ordered.sort_values(["session_date", "entry_time"]).reset_index(drop=True)
    equity = float(account_size_usd)
    peak_equity = float(account_size_usd)
    red_day_streak = 0
    busted = False
    trade_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    session_counter = 0

    for session_date, session_df in ordered.groupby("session_date", sort=True):
        session_counter += 1
        if horizon_days is not None and session_counter > int(horizon_days):
            break
        if busted:
            break
        red_scale = scenario.deleverage_factor if red_day_streak >= scenario.deleverage_after_red_days else 1.0
        day_pnl = 0.0
        day_losses = 0
        trades_taken = 0
        breached_daily_limit = False

        for _, trade in session_df.iterrows():
            if day_losses >= int(scenario.stop_after_losses_in_day):
                break
            if day_pnl <= -float(scenario.daily_loss_limit_usd):
                breached_daily_limit = True
                break
            trade_risk = float(pd.to_numeric(pd.Series([trade.get("trade_risk_usd")]), errors="coerce").iloc[0])
            if not math.isfinite(trade_risk) or trade_risk <= 0:
                continue
            risk_budget = equity * (scenario.risk_per_trade_pct / 100.0)
            quantity = int(max(0, math.floor(risk_budget / trade_risk)))
            if quantity <= 0:
                continue
            quantity = min(quantity, int(scenario.max_contracts))
            if red_scale < 1.0:
                quantity = max(1, int(math.floor(quantity * red_scale)))
            trade_net = float(trade["net_pnl_usd"]) * quantity
            trade_row = trade.to_dict()
            trade_row["scenario_quantity"] = quantity
            trade_row["scenario_net_pnl_usd"] = trade_net
            trade_rows.append(trade_row)
            day_pnl += trade_net
            trades_taken += 1
            if trade_net < 0:
                day_losses += 1
            if day_pnl <= -float(scenario.daily_loss_limit_usd):
                breached_daily_limit = True
                break

        equity += day_pnl
        peak_equity = max(peak_equity, equity)
        trailing_drawdown = peak_equity - equity
        if trailing_drawdown >= float(scenario.trailing_drawdown_limit_usd):
            busted = True

        daily_rows.append(
            {
                "session_date": session_date,
                "daily_pnl_usd": day_pnl,
                "daily_trade_count": trades_taken,
                "daily_loss_limit_breached": breached_daily_limit,
                "equity": equity,
                "peak_equity": peak_equity,
                "trailing_drawdown_usd": trailing_drawdown,
                "busted": busted,
            }
        )

        red_day_streak = red_day_streak + 1 if day_pnl < 0 else 0
        if equity - float(account_size_usd) >= float(scenario.profit_target_usd):
            break

    trade_frame = pd.DataFrame(trade_rows)
    daily_frame = pd.DataFrame(daily_rows)
    if daily_frame.empty:
        daily_frame = pd.DataFrame(columns=["session_date", "daily_pnl_usd", "daily_trade_count", "daily_loss_limit_breached", "equity", "peak_equity", "trailing_drawdown_usd", "busted"])
    final_pnl = float(daily_frame["daily_pnl_usd"].sum()) if not daily_frame.empty else 0.0
    success = final_pnl >= float(scenario.profit_target_usd)
    days_to_target = np.nan
    if not daily_frame.empty:
        cumulative = daily_frame["daily_pnl_usd"].cumsum()
        hits = np.flatnonzero(cumulative >= float(scenario.profit_target_usd))
        if len(hits) > 0:
            success = True
            days_to_target = float(hits[0] + 1)
    summary = {
        "success": bool(success),
        "days_to_target": days_to_target,
        "busted": bool(daily_frame["busted"].iloc[-1]) if not daily_frame.empty else False,
        "max_drawdown_usd": float((daily_frame["equity"] - daily_frame["peak_equity"]).min()) if not daily_frame.empty else 0.0,
        "daily_loss_limit_breaches": int(daily_frame["daily_loss_limit_breached"].sum()) if not daily_frame.empty else 0,
        "trailing_drawdown_breaches": int(daily_frame["busted"].sum()) if not daily_frame.empty else 0,
        "final_pnl_usd": final_pnl,
    }
    return trade_frame, daily_frame, summary


def _challenge_empirical_summary(trades: pd.DataFrame, scenario: ChallengeScenario, account_size_usd: float) -> tuple[dict[str, Any], pd.DataFrame]:
    ordered = trades.copy()
    ordered["session_date"] = pd.to_datetime(ordered["session_date"]).dt.date
    unique_sessions = sorted(pd.Index(ordered["session_date"]).unique())
    path_rows: list[dict[str, Any]] = []
    success_days: list[float] = []
    success_count = 0
    bust_count = 0
    for start_session in unique_sessions:
        subset = ordered.loc[ordered["session_date"] >= start_session].copy()
        _, _, summary = _simulate_challenge_path(subset, scenario=scenario, account_size_usd=account_size_usd, horizon_days=scenario.horizon_days)
        path_rows.append({"start_session": start_session, "success": summary["success"], "days_to_target": summary["days_to_target"], "busted": summary["busted"], "final_pnl_usd": summary["final_pnl_usd"], "max_drawdown_usd": summary["max_drawdown_usd"], "daily_loss_limit_breaches": summary["daily_loss_limit_breaches"]})
        if summary["success"]:
            success_count += 1
            if pd.notna(summary["days_to_target"]):
                success_days.append(float(summary["days_to_target"]))
        if summary["busted"]:
            bust_count += 1
    paths = pd.DataFrame(path_rows)
    total_paths = max(len(paths), 1)
    summary = {
        "scenario": scenario.name,
        "label": scenario.label,
        "success_rate_empirical": float(success_count / total_paths),
        "median_days_to_target": float(np.median(success_days)) if success_days else np.nan,
        "bust_rate_empirical": float(bust_count / total_paths),
        "max_drawdown_usd_worst_path": float(paths["max_drawdown_usd"].min()) if not paths.empty else 0.0,
        "average_daily_loss_limit_breaches": float(paths["daily_loss_limit_breaches"].mean()) if not paths.empty else 0.0,
    }
    return summary, paths


def _challenge_verdict(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "trop fragile pour challenge en l'etat"
    best = summary_df.sort_values("success_rate_empirical", ascending=False).iloc[0]
    if float(best["success_rate_empirical"]) >= 0.60 and float(best["bust_rate_empirical"]) <= 0.20:
        return "challenge-compatible"
    if float(best["success_rate_empirical"]) >= 0.35 and float(best["bust_rate_empirical"]) <= 0.35:
        return "compatible sous contraintes prudentes"
    return "trop fragile pour challenge en l'etat"


def _run_challenge_suite(output_dir: Path, nominal: NominalEvaluation) -> dict[str, Path]:
    challenge_dir = output_dir / "challenge_mode"
    challenge_dir.mkdir(parents=True, exist_ok=True)
    oos_trades = _subset_frame_by_sessions(nominal.trades, nominal.oos_sessions)
    summary_rows: list[dict[str, Any]] = []
    for scenario in _challenge_scenarios():
        _, daily_path, path_summary = _simulate_challenge_path(oos_trades, scenario=scenario, account_size_usd=nominal.variant.initial_capital_usd)
        empirical_summary, rolling_paths = _challenge_empirical_summary(oos_trades, scenario=scenario, account_size_usd=nominal.variant.initial_capital_usd)
        row = {
            **empirical_summary,
            "target_usd": scenario.profit_target_usd,
            "horizon_days": scenario.horizon_days,
            "daily_loss_limit_usd": scenario.daily_loss_limit_usd,
            "trailing_drawdown_limit_usd": scenario.trailing_drawdown_limit_usd,
            "risk_per_trade_pct": scenario.risk_per_trade_pct,
            "max_contracts": scenario.max_contracts,
            "historical_path_success": path_summary["success"],
            "historical_days_to_target": path_summary["days_to_target"],
            "historical_path_busted": path_summary["busted"],
            "historical_path_final_pnl_usd": path_summary["final_pnl_usd"],
        }
        summary_rows.append(row)
        daily_path.to_csv(challenge_dir / f"{scenario.name}_historical_daily_path.csv", index=False)
        rolling_paths.to_csv(challenge_dir / f"{scenario.name}_rolling_paths.csv", index=False)
    summary_df = pd.DataFrame(summary_rows).sort_values("success_rate_empirical", ascending=False).reset_index(drop=True)
    summary_csv = challenge_dir / "challenge_mode_summary.csv"
    summary_md = challenge_dir / "challenge_mode_summary.md"
    summary_df.to_csv(summary_csv, index=False)
    summary_md.write_text("\n".join(["# Challenge Mode Summary", "", f"- Verdict: `{_challenge_verdict(summary_df)}`", "", "```text", summary_df.to_string(index=False), "```", ""]), encoding="utf-8")
    return {"challenge_summary_csv": summary_csv, "challenge_summary_md": summary_md}


def _run_cross_instrument_suite(output_dir: Path, spec: ValidationSpec, nominal: NominalEvaluation) -> dict[str, Path]:
    cross_dir = output_dir / "cross_instrument"
    cross_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for symbol in spec.cross_instruments:
        dataset = resolve_default_vwap_dataset(symbol)
        if not dataset.exists():
            continue
        local_spec = replace(spec, dataset_path=dataset)
        feature_df = _prepare_feature_frame(local_spec, atr_windows=[nominal.variant.atr_period])
        local_nominal = _run_full_nominal(feature_df, local_spec)
        overall = local_nominal.summary_by_scope.loc[local_nominal.summary_by_scope["scope"] == "overall"].iloc[0]
        oos = local_nominal.summary_by_scope.loc[local_nominal.summary_by_scope["scope"] == "oos"].iloc[0]
        stressed_trades = _apply_cost_stress_overlay(local_nominal.trades, scenario=StressScenario(name="cross_stress", slippage_multiplier=2.0, commission_multiplier=1.25), instrument=local_nominal.instrument, execution_model=local_nominal.execution_model, session_start=local_spec.session_start)
        stressed_daily = _rebuild_daily_results_from_trades(stressed_trades, all_sessions=local_nominal.all_sessions, initial_capital=local_nominal.variant.initial_capital_usd)
        _, stressed_summaries, _ = _build_scope_summary_table(
            trades=stressed_trades,
            daily_results=stressed_daily,
            bar_results=pd.DataFrame(),
            signal_df=local_nominal.signal_df,
            sessions_all=local_nominal.all_sessions,
            is_sessions=local_nominal.is_sessions,
            oos_sessions=local_nominal.oos_sessions,
            initial_capital=local_nominal.variant.initial_capital_usd,
            constraints=local_spec.prop_constraints,
            instrument=local_nominal.instrument,
            execution_model=local_nominal.execution_model,
            rolling_window_days=local_spec.rolling_window_days,
        )
        rows.append({
            "symbol": symbol,
            "dataset_path": str(dataset),
            "overall_net_pnl": overall["net_pnl"],
            "overall_profit_factor": overall["profit_factor"],
            "overall_sharpe_ratio": overall["sharpe_ratio"],
            "overall_max_drawdown": overall["max_drawdown"],
            "oos_net_pnl": oos["net_pnl"],
            "oos_profit_factor": oos["profit_factor"],
            "oos_sharpe_ratio": oos["sharpe_ratio"],
            "oos_max_drawdown": oos["max_drawdown"],
            "oos_net_pnl_stress_x2_plus25": stressed_summaries["oos"]["net_pnl"],
            "oos_profit_factor_stress_x2_plus25": stressed_summaries["oos"]["profit_factor"],
        })
    cross_df = pd.DataFrame(rows)
    summary_csv = cross_dir / "cross_instrument_summary.csv"
    summary_md = cross_dir / "cross_instrument_summary.md"
    cross_df.to_csv(summary_csv, index=False)
    positive_rate = float(((cross_df["oos_net_pnl"] > 0) & (cross_df["oos_profit_factor"] > 1.0)).mean()) if not cross_df.empty else 0.0
    verdict = "logique structurelle plausible" if positive_rate >= 0.66 else "logique tres specifique a l'instrument" if positive_rate > 0.0 else "pas assez d'evidence cross-instrument"
    summary_md.write_text("\n".join(["# Cross-Instrument Summary", "", f"- Verdict: `{verdict}`", "", "```text", cross_df.to_string(index=False) if not cross_df.empty else "No cross-instrument dataset available.", "```", ""]), encoding="utf-8")
    return {"cross_instrument_summary_csv": summary_csv, "cross_instrument_summary_md": summary_md}


def _representative_day_selection(daily_results: pd.DataFrame) -> pd.DataFrame:
    if daily_results.empty:
        return pd.DataFrame(columns=["label", "session_date"])
    daily = daily_results.copy()
    daily["abs_pnl"] = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").abs()
    daily["trade_count"] = pd.to_numeric(daily["daily_trade_count"], errors="coerce").fillna(0.0)
    selected: list[dict[str, Any]] = []
    used_dates: set = set()

    def add_rows(frame: pd.DataFrame, label: str, n: int) -> None:
        for _, row in frame.iterrows():
            session_date = row["session_date"]
            if session_date in used_dates:
                continue
            selected.append({"label": label, "session_date": session_date})
            used_dates.add(session_date)
            if sum(item["label"] == label for item in selected) >= n:
                break

    add_rows(daily.sort_values("daily_pnl_usd", ascending=False), "excellent", 2)
    add_rows(daily.sort_values("daily_pnl_usd", ascending=True), "bad", 2)
    choppy = daily.loc[daily["abs_pnl"] <= daily["abs_pnl"].median()].copy()
    choppy["chop_score"] = choppy["trade_count"] - choppy["abs_pnl"] / max(float(choppy["abs_pnl"].median()), 1.0)
    add_rows(choppy.sort_values("chop_score", ascending=False), "choppy", 2)
    average = daily.copy()
    average["distance_to_median"] = (pd.to_numeric(average["daily_pnl_usd"], errors="coerce") - float(average["daily_pnl_usd"].median())).abs()
    add_rows(average.sort_values(["distance_to_median", "trade_count"], ascending=[True, True]), "clean_average", 2)
    return pd.DataFrame(selected)


def _representative_day_comment(day_trades: pd.DataFrame, day_daily: pd.Series) -> str:
    pnl = float(day_daily["daily_pnl_usd"])
    trade_count = int(day_daily["daily_trade_count"])
    if day_trades.empty:
        return f"Flat day, {trade_count} trade, daily PnL {pnl:.1f} USD."
    dominant_direction = day_trades["direction"].mode().iloc[0] if "direction" in day_trades.columns and not day_trades["direction"].mode().empty else "mixed"
    first_entry = pd.to_datetime(day_trades["entry_time"]).min().strftime("%H:%M")
    last_exit = pd.to_datetime(day_trades["exit_time"]).max().strftime("%H:%M")
    dominant_exit = day_trades["exit_reason"].mode().iloc[0] if "exit_reason" in day_trades.columns and not day_trades["exit_reason"].mode().empty else "mixed"
    return f"{trade_count} trades, net {pnl:.1f} USD, dominant direction {dominant_direction}, active from {first_entry} to {last_exit}, most common exit {dominant_exit}."


def _plot_representative_day(feature_df: pd.DataFrame, trades: pd.DataFrame, bar_results: pd.DataFrame, session_date: Any, label: str, output_path: Path) -> str:
    day_features = feature_df.loc[pd.to_datetime(feature_df["session_date"]).dt.date == pd.to_datetime(session_date).date()].copy()
    day_trades = trades.loc[pd.to_datetime(trades["session_date"]).dt.date == pd.to_datetime(session_date).date()].copy()
    day_bars = bar_results.loc[pd.to_datetime(bar_results["session_date"]).dt.date == pd.to_datetime(session_date).date()].copy()
    day_daily = {"daily_pnl_usd": float(day_trades["net_pnl_usd"].sum()) if not day_trades.empty else 0.0, "daily_trade_count": int(len(day_trades))}
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(pd.to_datetime(day_features["timestamp"]), day_features["close"], label="Close", linewidth=1.3)
    axes[0].plot(pd.to_datetime(day_features["timestamp"]), day_features["session_vwap"], label="VWAP", linewidth=1.2)
    for _, trade in day_trades.iterrows():
        color = "#15803d" if trade["direction"] == "long" else "#b91c1c"
        axes[0].scatter(pd.to_datetime(trade["entry_time"]), trade["entry_price"], color=color, marker="^", s=55)
        axes[0].scatter(pd.to_datetime(trade["exit_time"]), trade["exit_price"], color=color, marker="v", s=55)
    axes[0].set_title(f"{label} | {session_date}")
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="best")
    if not day_bars.empty:
        intraday = day_bars.copy()
        intraday["cum_pnl"] = pd.to_numeric(intraday["net_bar_pnl_usd"], errors="coerce").fillna(0.0).cumsum()
        axes[1].plot(pd.to_datetime(intraday["timestamp"]), intraday["cum_pnl"], linewidth=1.2, color="#111827")
    axes[1].axhline(0.0, color="#9ca3af", linewidth=0.8)
    axes[1].set_ylabel("Cum PnL")
    axes[1].set_xlabel("Time")
    comment = _representative_day_comment(day_trades, pd.Series(day_daily))
    fig.suptitle(comment, y=0.98, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return comment


def _run_representative_day_suite(output_dir: Path, feature_df: pd.DataFrame, nominal: NominalEvaluation) -> dict[str, Path]:
    rep_dir = output_dir / "representative_days"
    charts_dir = rep_dir / "charts"
    rep_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    selected = _representative_day_selection(nominal.daily_results)
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        session_date = row["session_date"]
        label = row["label"]
        safe_date = str(session_date).replace(":", "")
        output_path = charts_dir / f"{label}_{safe_date}.png"
        comment = _plot_representative_day(feature_df, nominal.trades, nominal.bar_results, session_date, label, output_path)
        rows.append({"label": label, "session_date": session_date, "chart_path": str(output_path), "comment": comment})
    table = pd.DataFrame(rows)
    summary_csv = rep_dir / "representative_days.csv"
    summary_md = rep_dir / "representative_days.md"
    table.to_csv(summary_csv, index=False)
    summary_md.write_text("\n".join(["# Representative Days", "", "```text", table.to_string(index=False) if not table.empty else "No representative day selected.", "```", ""]), encoding="utf-8")
    return {"representative_days_csv": summary_csv, "representative_days_md": summary_md}


def _final_verdict_payload(
    nominal: NominalEvaluation,
    stress_paths: dict[str, Path] | None,
    local_paths: dict[str, Path] | None,
    split_paths: dict[str, Path] | None,
    concentration_paths: dict[str, Path] | None,
    challenge_paths: dict[str, Path] | None,
    cross_paths: dict[str, Path] | None,
) -> dict[str, Any]:
    overall = nominal.summary_by_scope.loc[nominal.summary_by_scope["scope"] == "overall"].iloc[0]
    oos = nominal.summary_by_scope.loc[nominal.summary_by_scope["scope"] == "oos"].iloc[0]

    stress_verdict = _stress_verdict(pd.read_csv(stress_paths["stress_summary_csv"])) if stress_paths is not None and Path(stress_paths["stress_summary_csv"]).exists() else None
    local_verdict = None
    if local_paths is not None and Path(local_paths["local_heatmap_readouts_csv"]).exists():
        local_readouts = pd.read_csv(local_paths["local_heatmap_readouts_csv"])
        if not local_readouts.empty:
            counts = local_readouts["verdict"].value_counts()
            local_verdict = "stable localement" if counts.get("stable localement", 0) >= 3 else "moderetement stable" if counts.get("stable localement", 0) + counts.get("moderetement stable", 0) >= 2 else "instable / pic etroit"
    split_verdict = None
    if split_paths is not None and Path(split_paths["split_summary_csv"]).exists():
        split_df = pd.read_csv(split_paths["split_summary_csv"])
        positive_rate = float(((split_df["oos_net_pnl"] > 0) & (split_df["oos_profit_factor"] > 1.0)).mean()) if not split_df.empty else 0.0
        split_verdict = "stable entre splits" if positive_rate >= 0.75 else "acceptable mais variable" if positive_rate >= 0.50 else "trop dependant du split"
    concentration_verdict = None
    if concentration_paths is not None and Path(concentration_paths["concentration_summary_csv"]).exists():
        concentration_df = pd.read_csv(concentration_paths["concentration_summary_csv"])
        if not concentration_df.empty:
            concentration_verdict = str(concentration_df.iloc[0]["verdict"])
    challenge_verdict = _challenge_verdict(pd.read_csv(challenge_paths["challenge_summary_csv"])) if challenge_paths is not None and Path(challenge_paths["challenge_summary_csv"]).exists() else None
    cross_verdict = None
    if cross_paths is not None and Path(cross_paths["cross_instrument_summary_md"]).exists():
        cross_text = Path(cross_paths["cross_instrument_summary_md"]).read_text(encoding="utf-8")
        cross_verdict = "logique structurelle plausible" if "logique structurelle plausible" in cross_text else "logique tres specifique a l'instrument" if "logique tres specifique a l'instrument" in cross_text else "pas assez d'evidence cross-instrument"

    statistical_block = (
        "Le run de decouverte historique n'est pas defendable tel quel a cause d'une fuite temporelle discrete entree-close/same-bar; le rerun corrige ne montre plus qu'un edge tres faible."
        if float(oos["profit_factor"]) <= 1.05 or float(oos["sharpe_ratio"]) <= 0.50
        else "Le rerun corrige conserve un edge OOS positif, mais il doit encore passer les autres tests de robustesse."
    )
    execution_block = "Les stress de couts gardent un profil exploitable." if stress_verdict == "strategie robuste aux couts" else "Le profil se degrade vite des qu'on stresse l'execution."
    param_block = "La topologie locale ressemble a une mesa acceptable." if local_verdict == "stable localement" else "La topologie locale reste etroite ou fragile."
    temporal_block = "Le signal tient de facon relativement stable a travers plusieurs splits." if split_verdict == "stable entre splits" else "Le resultat reste tres dependant du split choisi."
    prop_block = "La strategie est compatible avec un challenge prudent." if challenge_verdict in {"challenge-compatible", "compatible sous contraintes prudentes"} else "La strategie reste trop fragile pour un challenge prop en l'etat."

    negative_flags = sum(
        [
            float(oos["profit_factor"]) <= 1.0,
            float(oos["sharpe_ratio"]) <= 0.0,
            stress_verdict == "strategie tres fragile a la microstructure",
            local_verdict == "instable / pic etroit",
            split_verdict == "trop dependant du split",
            challenge_verdict == "trop fragile pour challenge en l'etat",
        ]
    )
    final_category = "non defendable en l'etat" if negative_flags >= 3 else "candidat interessant mais trop instable" if negative_flags == 2 else "candidat prometteur mais fragile" if negative_flags == 1 else "candidat solide"

    return {
        "statistical_robustness": statistical_block,
        "execution_robustness": execution_block,
        "parametric_robustness": param_block,
        "temporal_robustness": temporal_block,
        "prop_firm_viability": prop_block,
        "concentration_verdict": concentration_verdict,
        "cross_instrument_verdict": cross_verdict,
        "overall_category": final_category,
        "nominal_oos_net_pnl": float(oos["net_pnl"]),
        "nominal_oos_profit_factor": float(oos["profit_factor"]),
        "nominal_oos_sharpe_ratio": float(oos["sharpe_ratio"]),
        "nominal_overall_net_pnl": float(overall["net_pnl"]),
    }


def _write_validation_report(
    output_dir: Path,
    nominal: NominalEvaluation,
    verdict: dict[str, Any],
    stress_paths: dict[str, Path] | None,
    local_paths: dict[str, Path] | None,
    split_paths: dict[str, Path] | None,
    concentration_paths: dict[str, Path] | None,
    challenge_paths: dict[str, Path] | None,
    cross_paths: dict[str, Path] | None,
) -> dict[str, Path]:
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    verdict_json = summary_dir / "final_verdict.json"
    report_md = summary_dir / "validation_report.md"
    _json_dump(verdict_json, verdict)
    scope_view = nominal.summary_by_scope[["scope", "net_pnl", "profit_factor", "sharpe_ratio", "max_drawdown", "total_trades", "expectancy_per_trade"]].copy()
    report_md.write_text(
        "\n".join(
            [
                "# VWAP Pullback Continuation Validation",
                "",
                "## Executive Summary",
                "",
                f"- Final category: `{verdict['overall_category']}`",
                "- Validation is based on the corrected next-open discrete execution semantics, not on the legacy discovery run.",
                "",
                "## Nominal Corrected Run",
                "",
                "```text",
                scope_view.to_string(index=False),
                "```",
                "",
                "## Final Verdict Blocks",
                "",
                f"- Robustesse statistique: {verdict['statistical_robustness']}",
                f"- Robustesse execution: {verdict['execution_robustness']}",
                f"- Robustesse parametrique: {verdict['parametric_robustness']}",
                f"- Robustesse temporelle: {verdict['temporal_robustness']}",
                f"- Viabilite prop firm: {verdict['prop_firm_viability']}",
                "",
                "## Artifact Pointers",
                "",
                f"- Stress: `{stress_paths['stress_summary_csv']}`" if stress_paths is not None else "- Stress: not generated",
                f"- Local robustness: `{local_paths['local_sensitivity_csv']}`" if local_paths is not None else "- Local robustness: not generated",
                f"- Multi-split: `{split_paths['split_summary_csv']}`" if split_paths is not None else "- Multi-split: not generated",
                f"- Concentration: `{concentration_paths['concentration_summary_csv']}`" if concentration_paths is not None else "- Concentration: not generated",
                f"- Challenge mode: `{challenge_paths['challenge_summary_csv']}`" if challenge_paths is not None else "- Challenge mode: not generated",
                f"- Cross instrument: `{cross_paths['cross_instrument_summary_csv']}`" if cross_paths is not None else "- Cross instrument: not generated",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"final_verdict_json": verdict_json, "validation_report_md": report_md}


def _notebook_cell(cell_type: str, source: str) -> dict[str, Any]:
    if not source.endswith("\n"):
        source = source + "\n"
    cell = {
        "cell_type": cell_type,
        "id": hashlib.sha1(f"{cell_type}:{source}".encode("utf-8")).hexdigest()[:8],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def generate_validation_notebook(notebook_path: Path, output_dir: Path) -> Path:
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_dir = output_dir.resolve()
    setup_code = """from pathlib import Path
import json
import sys
from IPython.display import Image, Markdown, display
import pandas as pd

root = Path.cwd().resolve()
while root != root.parent and not (root / "pyproject.toml").exists():
    root = root.parent
if not (root / "pyproject.toml").exists():
    raise RuntimeError("Could not locate repository root.")
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 120)
"""
    config_code = f"OUTPUT_DIR = Path(r\"{str(resolved_output_dir)}\")\nprint(OUTPUT_DIR)\n"
    summary_code = """report = (OUTPUT_DIR / "summary" / "validation_report.md").read_text(encoding="utf-8")
display(Markdown(report))
verdict = json.loads((OUTPUT_DIR / "summary" / "final_verdict.json").read_text(encoding="utf-8"))
verdict
"""
    reference_code = """display(Markdown((OUTPUT_DIR / "nominal" / "reference_spec.md").read_text(encoding="utf-8")))
reference_config = json.loads((OUTPUT_DIR / "nominal" / "reference_config.json").read_text(encoding="utf-8"))
reference_config
"""
    nominal_code = """metrics = pd.read_csv(OUTPUT_DIR / "nominal" / "metrics_summary_by_scope.csv")
display(metrics)
for name in ["monthly_pnl", "quarterly_pnl", "rolling_20d_metrics", "intraday_by_hour", "weekday_expectancy"]:
    path = OUTPUT_DIR / "nominal" / "tables" / f"{name}.csv"
    if path.exists():
        display(Markdown(f"### {name}"))
        display(pd.read_csv(path).head(20))
"""
    stress_code = """path = OUTPUT_DIR / "stress" / "stress_test_summary.csv"
if path.exists():
    display(pd.read_csv(path))
"""
    local_code = """path = OUTPUT_DIR / "local_robustness" / "heatmap_readouts.csv"
if path.exists():
    display(pd.read_csv(path))
for png in sorted((OUTPUT_DIR / "local_robustness" / "heatmaps").glob("*.png")):
    display(Markdown(f"### {png.stem}"))
    display(Image(filename=str(png)))
"""
    split_code = """path = OUTPUT_DIR / "multi_split" / "split_summary.csv"
if path.exists():
    display(pd.read_csv(path))
"""
    concentration_code = """path = OUTPUT_DIR / "concentration" / "concentration_summary.csv"
if path.exists():
    display(pd.read_csv(path))
for png in sorted((OUTPUT_DIR / "concentration" / "charts").glob("*.png")):
    display(Image(filename=str(png)))
"""
    challenge_code = """path = OUTPUT_DIR / "challenge_mode" / "challenge_mode_summary.csv"
if path.exists():
    display(pd.read_csv(path))
"""
    cross_code = """path = OUTPUT_DIR / "cross_instrument" / "cross_instrument_summary.csv"
if path.exists():
    display(pd.read_csv(path))
"""
    representative_code = """path = OUTPUT_DIR / "representative_days" / "representative_days.csv"
if path.exists():
    display(pd.read_csv(path))
for png in sorted((OUTPUT_DIR / "representative_days" / "charts").glob("*.png")):
    display(Image(filename=str(png)))
"""
    notebook = {
        "cells": [
            _notebook_cell("markdown", "# VWAP Pullback Continuation Final Validation Notebook"),
            _notebook_cell("code", setup_code),
            _notebook_cell("code", config_code),
            _notebook_cell("markdown", "## 1) Executive Summary"),
            _notebook_cell("code", summary_code),
            _notebook_cell("markdown", "## 2) Reference Config"),
            _notebook_cell("code", reference_code),
            _notebook_cell("markdown", "## 3) Nominal Metrics"),
            _notebook_cell("code", nominal_code),
            _notebook_cell("markdown", "## 4) Stress Tests"),
            _notebook_cell("code", stress_code),
            _notebook_cell("markdown", "## 5) Parametric Robustness"),
            _notebook_cell("code", local_code),
            _notebook_cell("markdown", "## 6) Multi-Split"),
            _notebook_cell("code", split_code),
            _notebook_cell("markdown", "## 7) Concentration"),
            _notebook_cell("code", concentration_code),
            _notebook_cell("markdown", "## 8) Challenge Mode"),
            _notebook_cell("code", challenge_code),
            _notebook_cell("markdown", "## 9) Cross Instrument"),
            _notebook_cell("code", cross_code),
            _notebook_cell("markdown", "## 10) Representative Days"),
            _notebook_cell("code", representative_code),
            _notebook_cell("markdown", "## 11) Final Verdict\n\nThe final report above is the decision-ready source of truth."),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return notebook_path


def run_vwap_validation_campaign(
    spec: ValidationSpec,
    output_dir: Path,
    phases: tuple[str, ...] = VALIDATION_PHASES,
    notebook_path: Path | None = None,
) -> dict[str, Path]:
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    active_phases = set(phases)
    if "all" in active_phases:
        active_phases = set(VALIDATION_PHASES)
    feature_df = _prepare_feature_frame(spec, atr_windows=[_resolve_reference_variant(spec.reference_variant_name).atr_period])
    nominal = _run_full_nominal(feature_df, spec)
    source_metadata = _load_source_run_metadata(spec.source_run_metadata_path)
    artifacts: dict[str, Path] = {"output_dir": output_dir}
    artifacts.update(_export_nominal(output_dir, spec, nominal, source_metadata))

    stress_paths = _run_stress_suite(output_dir, spec, nominal) if "stress" in active_phases else None
    local_paths = _run_local_robustness_suite(output_dir, spec, nominal, feature_df) if "local" in active_phases else None
    split_paths = _run_multi_split_suite(output_dir, spec, nominal) if "splits" in active_phases else None
    concentration_paths = _run_concentration_suite(output_dir, nominal) if "concentration" in active_phases else None
    challenge_paths = _run_challenge_suite(output_dir, nominal) if "challenge" in active_phases else None
    cross_paths = _run_cross_instrument_suite(output_dir, spec, nominal) if "cross" in active_phases else None
    representative_paths = _run_representative_day_suite(output_dir, feature_df, nominal) if "representative" in active_phases else None

    for path_map in (stress_paths, local_paths, split_paths, concentration_paths, challenge_paths, cross_paths, representative_paths):
        if path_map is not None:
            artifacts.update(path_map)

    verdict = _final_verdict_payload(nominal, stress_paths, local_paths, split_paths, concentration_paths, challenge_paths, cross_paths)
    artifacts.update(_write_validation_report(output_dir, nominal, verdict, stress_paths, local_paths, split_paths, concentration_paths, challenge_paths, cross_paths))

    generated_notebook = generate_validation_notebook(notebook_path=notebook_path, output_dir=output_dir) if notebook_path is not None and "notebook" in active_phases else None
    if generated_notebook is not None:
        artifacts["validation_notebook"] = generated_notebook

    _json_dump(
        output_dir / "summary" / "run_metadata.json",
        {"run_timestamp": datetime.now().isoformat(), "dataset_path": spec.dataset_path, "reference_variant_name": spec.reference_variant_name, "active_phases": sorted(active_phases), "output_dir": output_dir},
    )
    return artifacts


def _parse_phases(value: str) -> tuple[str, ...]:
    phases = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not phases:
        return VALIDATION_PHASES
    if "all" in phases:
        return VALIDATION_PHASES
    invalid = sorted(set(phases) - set(VALIDATION_PHASES))
    if invalid:
        raise ValueError(f"Unknown phases: {', '.join(invalid)}")
    return phases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VWAP pullback continuation validation campaign.")
    parser.add_argument("--dataset", type=Path, default=None, help="Optional dataset path. Defaults to the latest MNQ 1-minute file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional export directory.")
    parser.add_argument("--notebook-path", type=Path, default=None, help="Optional notebook output path.")
    parser.add_argument("--phases", type=str, default="nominal,stress,local,splits,concentration,challenge,cross,representative,notebook", help="Comma-separated phases among nominal,stress,local,splits,concentration,challenge,cross,representative,notebook.")
    args = parser.parse_args()

    spec = build_default_validation_spec(dataset_path=args.dataset)
    phases = _parse_phases(args.phases)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (EXPORTS_DIR / f"vwap_pullback_validation_{timestamp}")
    notebook_path = args.notebook_path or (NOTEBOOKS_DIR / "vwap_pullback_continuation_validation.ipynb")
    artifacts = run_vwap_validation_campaign(spec=spec, output_dir=output_dir, phases=phases, notebook_path=notebook_path)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
