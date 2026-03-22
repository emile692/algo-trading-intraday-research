"""Multi-asset transfer campaign for the existing Ensemble ORB baseline.

The goal of this module is deliberately narrow:
- reuse the current long-only ORB + VWAP baseline logic,
- evaluate the ATR cross-product sub-signals on already processed data,
- compare simple ensemble vote rules across symbols,
- export compact research artifacts for client delivery.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.metrics import compute_metrics
from src.analytics.orb_notebook_utils import normalize_curve
from src.config.paths import PROCESSED_DATA_DIR
from src.config.settings import DEFAULT_INITIAL_CAPITAL_USD, get_instrument_spec
from src.data.cleaning import clean_ohlcv
from src.data.loader import load_ohlcv_file
from src.engine.backtester import run_backtest
from src.engine.execution_model import ExecutionModel
from src.engine.portfolio import build_equity_curve
from src.features.intraday import (
    add_continuous_session_vwap,
    add_intraday_features,
    add_session_vwap,
)
from src.features.opening_range import compute_opening_range
from src.features.volatility import add_atr
from src.strategy.orb import ORBStrategy

REPO_ROOT = ROOT
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "export" / "orb_multi_asset_campaign"
DEFAULT_CHARTS_DIR = DEFAULT_OUTPUT_ROOT / "charts"

AGGREGATION_THRESHOLDS = {
    "majority_50": 0.50,
    "consensus_75": 0.75,
    "unanimity_100": 1.00,
}


@dataclass(frozen=True)
class BaselineSpec:
    """Frozen Ensemble ORB baseline transferred across tickers."""

    or_minutes: int = 15
    opening_time: str = "09:30:00"
    direction: str = "long"
    one_trade_per_day: bool = True
    entry_buffer_ticks: int = 2
    stop_buffer_ticks: int = 2
    target_multiple: float = 2.0
    vwap_confirmation: bool = True
    vwap_column: str = "continuous_session_vwap"
    time_exit: str = "16:00:00"
    account_size_usd: float = DEFAULT_INITIAL_CAPITAL_USD
    risk_per_trade_pct: float = 1.5
    entry_on_next_open: bool = True


@dataclass(frozen=True)
class SearchGrid:
    """Parametric grid kept intentionally close to the MNQ reference notebook."""

    atr_periods: tuple[int, ...] = (25, 26, 27, 28, 29, 30)
    q_lows_pct: tuple[int, ...] = (25, 26, 27, 28, 29, 30)
    q_highs_pct: tuple[int, ...] = (90, 91, 92, 93, 94, 95)
    aggregation_rules: tuple[str, ...] = ("majority_50", "consensus_75", "unanimity_100")


@dataclass(frozen=True)
class CampaignConfig:
    """Top-level multi-asset campaign configuration."""

    symbols: tuple[str, ...] = ("MES", "M2K")
    reference_symbol: str = "MNQ"
    is_fraction: float = 0.70
    baseline: BaselineSpec = BaselineSpec()
    grid: SearchGrid = SearchGrid()
    data_timeframe: str | None = None
    output_root: Path = DEFAULT_OUTPUT_ROOT


@dataclass
class SymbolAnalysis:
    """Full in-memory result bundle for one symbol."""

    symbol: str
    dataset_path: Path
    instrument_spec: dict[str, Any]
    baseline: BaselineSpec
    grid: SearchGrid
    feature_df: pd.DataFrame
    signal_df: pd.DataFrame
    baseline_trades: pd.DataFrame
    candidate_df: pd.DataFrame
    session_sanity: pd.DataFrame
    all_sessions: list
    is_sessions: list
    oos_sessions: list
    baseline_metrics_overall: dict[str, Any]
    baseline_metrics_is: dict[str, Any]
    baseline_metrics_oos: dict[str, Any]
    point_results: pd.DataFrame
    ensemble_results: pd.DataFrame
    robust_clusters: pd.DataFrame
    best_cell: dict[str, Any]
    robust_cell: dict[str, Any]
    baseline_transfer: dict[str, Any]
    best_ensemble: dict[str, Any]
    robust_ensemble: dict[str, Any]
    baseline_like_ensemble: dict[str, Any]
    export_paths: dict[str, Path] = field(default_factory=dict)


def _normalize_timeframe_tag(timeframe: str | None) -> str | None:
    if timeframe is None:
        return None
    clean = str(timeframe).strip().lower().replace(" ", "")
    if clean.endswith("min"):
        clean = f"{clean[:-3]}m"
    return clean or None


def _dataset_timeframe_tag(path: Path) -> str | None:
    parts = path.stem.split("_")
    if len(parts) >= 6:
        return _normalize_timeframe_tag(parts[3])
    return None


def _infer_bar_minutes(timestamp: pd.Series) -> int:
    clean = pd.to_datetime(timestamp, errors="coerce").dropna().drop_duplicates().sort_values()
    if len(clean) < 2:
        return 1
    diffs = clean.diff().dropna().dt.total_seconds()
    positive = diffs[diffs > 0]
    if positive.empty:
        return 1
    return max(1, int(round(float(positive.median()) / 60.0)))


def resolve_processed_dataset(
    symbol: str,
    processed_dir: Path | None = None,
    timeframe: str | None = None,
) -> Path:
    """Return the latest processed parquet for a symbol, optionally filtered by timeframe."""
    root = processed_dir or (PROCESSED_DATA_DIR / "parquet")
    matches = sorted(root.glob(f"{symbol.upper()}_*.parquet"))
    wanted_timeframe = _normalize_timeframe_tag(timeframe)
    if wanted_timeframe is not None:
        matches = [path for path in matches if _dataset_timeframe_tag(path) == wanted_timeframe]
    if not matches:
        if wanted_timeframe is None:
            raise FileNotFoundError(f"No processed parquet found for {symbol} in {root}")
        raise FileNotFoundError(
            f"No processed parquet found for {symbol} with timeframe {wanted_timeframe} in {root}"
        )
    return matches[-1]


def resolve_aggregation_threshold(rule: str) -> float:
    """Translate a named aggregation rule to its vote threshold."""
    if rule not in AGGREGATION_THRESHOLDS:
        available = ", ".join(sorted(AGGREGATION_THRESHOLDS))
        raise ValueError(f"Unsupported aggregation rule '{rule}'. Available: {available}")
    return float(AGGREGATION_THRESHOLDS[rule])


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return float(default)
    value = numerator / denominator
    return float(value) if math.isfinite(value) else float(default)


def _safe_rel(candidate: float, baseline: float, eps: float = 1e-9) -> float:
    if not math.isfinite(candidate) or not math.isfinite(baseline):
        return 0.0
    return float((candidate - baseline) / max(abs(baseline), eps))


def _quantile(series: pd.Series, q: float) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    value = float(clean.quantile(q))
    return value if math.isfinite(value) else None


def _daily_pnl(trades: pd.DataFrame, sessions: list) -> pd.Series:
    idx = pd.Index(pd.to_datetime(pd.Index(sessions)).date)
    if idx.empty:
        return pd.Series(dtype=float)
    if trades.empty:
        return pd.Series(0.0, index=idx, dtype=float)
    grouped = trades.groupby(pd.to_datetime(trades["session_date"]).dt.date)["net_pnl_usd"].sum()
    return grouped.reindex(idx, fill_value=0.0).astype(float)


def _trade_losing_streak(pnl: pd.Series) -> int:
    max_streak = 0
    current = 0
    for value in pd.to_numeric(pnl, errors="coerce").fillna(0.0):
        if value < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return int(max_streak)


def _build_strategy(baseline: BaselineSpec, tick_size: float) -> ORBStrategy:
    return ORBStrategy(
        or_minutes=baseline.or_minutes,
        direction=baseline.direction,
        one_trade_per_day=baseline.one_trade_per_day,
        entry_buffer_ticks=baseline.entry_buffer_ticks,
        stop_buffer_ticks=baseline.stop_buffer_ticks,
        target_multiple=baseline.target_multiple,
        opening_time=baseline.opening_time,
        time_exit=baseline.time_exit,
        account_size_usd=baseline.account_size_usd,
        risk_per_trade_pct=baseline.risk_per_trade_pct,
        tick_size=tick_size,
        vwap_confirmation=baseline.vwap_confirmation,
        vwap_column=baseline.vwap_column,
    )


def _prepare_feature_dataset(
    dataset_path: Path,
    baseline: BaselineSpec,
    grid: SearchGrid,
) -> pd.DataFrame:
    raw = load_ohlcv_file(dataset_path)
    raw = clean_ohlcv(raw)
    feat = add_intraday_features(raw)
    feat = add_session_vwap(feat)
    feat = add_continuous_session_vwap(feat, session_start_hour=18)
    feat = compute_opening_range(
        feat,
        or_minutes=baseline.or_minutes,
        opening_time=baseline.opening_time,
    )
    feat = add_atr(feat, window=grid.atr_periods)
    return feat


def _opening_window_mask(df: pd.DataFrame, opening_time: str, or_minutes: int) -> pd.Series:
    minutes = pd.to_datetime(df["timestamp"]).dt.hour * 60 + pd.to_datetime(df["timestamp"]).dt.minute
    open_minutes = pd.to_datetime(opening_time).hour * 60 + pd.to_datetime(opening_time).minute
    return (minutes >= open_minutes) & (minutes < open_minutes + int(or_minutes))


def _build_session_sanity(
    feat: pd.DataFrame,
    baseline: BaselineSpec,
) -> pd.DataFrame:
    working = feat.copy()
    timestamp = pd.to_datetime(working["timestamp"])
    bar_minutes = _infer_bar_minutes(timestamp)
    minute_of_day = timestamp.dt.hour * 60 + timestamp.dt.minute
    rth_start = pd.to_datetime(baseline.opening_time).hour * 60 + pd.to_datetime(baseline.opening_time).minute
    rth_end = pd.to_datetime(baseline.time_exit).hour * 60 + pd.to_datetime(baseline.time_exit).minute
    expected_opening_window_bars = max(1, int(math.ceil(float(baseline.or_minutes) / float(bar_minutes))))
    rth_expected_bars = max(1, int(((rth_end - rth_start) / float(bar_minutes)) + 1))
    working["is_rth"] = (minute_of_day >= rth_start) & (minute_of_day <= rth_end)
    working["is_opening_window"] = _opening_window_mask(working, baseline.opening_time, baseline.or_minutes)
    agg = (
        working.groupby("session_date", sort=True)
        .agg(
            bars_total=("timestamp", "size"),
            rth_bars=("is_rth", "sum"),
            opening_window_bars=("is_opening_window", "sum"),
            has_opening_range=("or_high", lambda s: bool(s.notna().any())),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            min_open=("open", "min"),
            min_low=("low", "min"),
            max_high=("high", "max"),
            max_close=("close", "max"),
            total_volume=("volume", "sum"),
        )
        .reset_index()
    )
    agg["bar_minutes"] = bar_minutes
    agg["expected_opening_window_bars"] = expected_opening_window_bars
    agg["opening_window_complete"] = agg["opening_window_bars"] >= expected_opening_window_bars
    agg["rth_expected_bars"] = rth_expected_bars
    agg["rth_missing_bars"] = (agg["rth_expected_bars"] - agg["rth_bars"]).clip(lower=0)
    return agg


def _split_sessions(all_sessions: list, is_fraction: float) -> tuple[list, list]:
    if len(all_sessions) < 2:
        raise ValueError("Not enough sessions to run an IS/OOS split.")
    split_idx = int(len(all_sessions) * float(is_fraction))
    split_idx = max(1, min(len(all_sessions) - 1, split_idx))
    return all_sessions[:split_idx], all_sessions[split_idx:]


def _run_baseline_backtest(
    signal_df: pd.DataFrame,
    baseline: BaselineSpec,
    instrument_spec: dict[str, Any],
) -> pd.DataFrame:
    execution_model = ExecutionModel(
        commission_per_side_usd=float(instrument_spec["commission_per_side_usd"]),
        slippage_ticks=int(instrument_spec["slippage_ticks"]),
        tick_size=float(instrument_spec["tick_size"]),
    )
    return run_backtest(
        signal_df,
        execution_model=execution_model,
        tick_value_usd=float(instrument_spec["tick_value_usd"]),
        point_value_usd=float(instrument_spec["point_value_usd"]),
        time_exit=baseline.time_exit,
        stop_buffer_ticks=baseline.stop_buffer_ticks,
        target_multiple=baseline.target_multiple,
        account_size_usd=baseline.account_size_usd,
        risk_per_trade_pct=baseline.risk_per_trade_pct,
        entry_on_next_open=baseline.entry_on_next_open,
    )


def compute_campaign_metrics(
    trades: pd.DataFrame,
    sessions: list,
    initial_capital: float,
) -> dict[str, Any]:
    """Return research metrics with a simple prop-oriented ranking score."""
    base = compute_metrics(
        trades,
        session_dates=sessions,
        initial_capital=initial_capital,
    )
    daily = _daily_pnl(trades, sessions)
    net_pnl = float(base.get("cumulative_pnl", 0.0))
    max_dd = float(base.get("max_drawdown", 0.0))
    max_dd_abs = abs(max_dd)
    worst_5d = float(daily.rolling(5).sum().min()) if len(daily) >= 5 else float(daily.sum()) if not daily.empty else 0.0
    ret_dd = _safe_div(net_pnl, max(max_dd_abs, 1.0), default=0.0)
    pf = float(base.get("profit_factor", 0.0))
    pf_capped = min(pf, 4.0) if math.isfinite(pf) else 4.0
    expectancy = float(base.get("expectancy", 0.0))
    participation = float(base.get("percent_of_days_traded", 0.0))
    n_trades = int(base.get("n_trades", 0))
    losing_streak = int(base.get("longest_loss_streak", 0))
    worst_day = float(daily.min()) if not daily.empty else 0.0
    score = (
        2.40 * np.tanh(ret_dd / 3.0)
        + 1.60 * np.tanh((pf_capped - 1.0) / 0.50)
        + 1.20 * np.tanh(expectancy / 40.0)
        + 0.45 * np.tanh(participation / 0.18)
        + 0.35 * np.tanh(n_trades / 120.0)
        - 1.35 * np.tanh(max_dd_abs / 2500.0)
        - 1.10 * np.tanh(abs(worst_5d) / 1600.0)
        - 0.95 * np.tanh(max(losing_streak - 2.0, 0.0) / 5.0)
    )
    return {
        **base,
        "net_pnl": net_pnl,
        "sharpe": float(base.get("sharpe_ratio", 0.0)),
        "profit_factor": pf,
        "expectancy": expectancy,
        "max_drawdown_abs": max_dd_abs,
        "return_over_drawdown": ret_dd,
        "nb_trades": n_trades,
        "pct_days_traded": participation,
        "avg_winner": float(base.get("avg_win", 0.0)),
        "avg_loser": float(base.get("avg_loss", 0.0)),
        "hit_ratio": float(base.get("win_rate", 0.0)),
        "worst_day": worst_day,
        "worst_5day_drawdown": worst_5d,
        "max_losing_streak": losing_streak,
        "trade_losing_streak": _trade_losing_streak(
            trades["net_pnl_usd"] if not trades.empty else pd.Series(dtype=float)
        ),
        "composite_score": float(score),
    }


def _prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    keys = [
        "net_pnl",
        "sharpe",
        "profit_factor",
        "expectancy",
        "max_drawdown",
        "max_drawdown_abs",
        "return_over_drawdown",
        "nb_trades",
        "pct_days_traded",
        "avg_winner",
        "avg_loser",
        "hit_ratio",
        "worst_day",
        "worst_5day_drawdown",
        "max_losing_streak",
        "trade_losing_streak",
        "composite_score",
    ]
    return {f"{prefix}_{key}": metrics.get(key, np.nan) for key in keys}


def _selected_candidate_rows(signal_df: pd.DataFrame, atr_periods: tuple[int, ...]) -> pd.DataFrame:
    atr_cols = [f"atr_{period}" for period in atr_periods if f"atr_{period}" in signal_df.columns]
    keep_cols = ["session_date", "timestamp", "signal"] + atr_cols
    selected = signal_df.loc[signal_df["signal"].ne(0), keep_cols].copy()
    selected = selected.sort_values("timestamp").drop_duplicates(subset=["session_date"], keep="first")
    selected["signal_index"] = selected.index
    return selected.reset_index(drop=True)


def _local_neighbor_average(df: pd.DataFrame, candidate: tuple[int, int, int], value_col: str) -> float:
    period, q_low, q_high = candidate
    mask = (
        df["atr_period"].between(period - 1, period + 1)
        & df["q_low_pct"].between(q_low - 1, q_low + 1)
        & df["q_high_pct"].between(q_high - 1, q_high + 1)
    )
    neighborhood = pd.to_numeric(df.loc[mask, value_col], errors="coerce").dropna()
    if neighborhood.empty:
        return 0.0
    return float(neighborhood.mean())


def _score_point_robustness(point_results: pd.DataFrame) -> pd.DataFrame:
    if point_results.empty:
        return point_results.copy()

    out = point_results.copy()
    out["delta_sharpe_oos_minus_is"] = out["oos_sharpe"] - out["is_sharpe"]
    out["delta_pf_oos_minus_is"] = out["oos_profit_factor"] - out["is_profit_factor"]
    out["delta_expectancy_oos_minus_is"] = out["oos_expectancy"] - out["is_expectancy"]

    out["neighbor_score_mean"] = out.apply(
        lambda row: _local_neighbor_average(
            out,
            (int(row["atr_period"]), int(row["q_low_pct"]), int(row["q_high_pct"])),
            "oos_composite_score",
        ),
        axis=1,
    )

    drift = (
        out["delta_sharpe_oos_minus_is"].abs() / out["is_sharpe"].abs().clip(lower=0.25)
        + out["delta_pf_oos_minus_is"].abs() / out["is_profit_factor"].abs().clip(lower=0.25)
        + out["delta_expectancy_oos_minus_is"].abs() / out["is_expectancy"].abs().clip(lower=1.0)
    ) / 3.0
    out["drift_penalty"] = drift.clip(upper=3.0)
    out["local_robustness_score"] = (
        0.65 * out["oos_composite_score"]
        + 0.25 * out["neighbor_score_mean"]
        - 0.30 * out["drift_penalty"]
    )

    trade_floor = max(30.0, float(out["oos_nb_trades"].median()))
    score_cutoff = float(out["local_robustness_score"].quantile(0.75))
    out["robust_flag"] = (
        (out["oos_nb_trades"] >= trade_floor)
        & (out["oos_composite_score"] > 0.0)
        & (out["local_robustness_score"] >= score_cutoff)
    )
    return out


def _robust_clusters(
    point_results: pd.DataFrame,
    grid: SearchGrid,
) -> pd.DataFrame:
    robust = point_results.loc[point_results["robust_flag"]].copy()
    if robust.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "n_cells",
                "period_min",
                "period_max",
                "q_low_min",
                "q_low_max",
                "q_high_min",
                "q_high_max",
                "avg_oos_score",
                "avg_oos_sharpe",
                "avg_oos_profit_factor",
                "avg_oos_expectancy",
                "cells",
            ]
        )

    periods = list(grid.atr_periods)
    lows = list(grid.q_lows_pct)
    highs = list(grid.q_highs_pct)
    period_pos = {value: idx for idx, value in enumerate(periods)}
    low_pos = {value: idx for idx, value in enumerate(lows)}
    high_pos = {value: idx for idx, value in enumerate(highs)}

    keys = {
        (int(row["atr_period"]), int(row["q_low_pct"]), int(row["q_high_pct"]))
        for _, row in robust.iterrows()
    }
    visited: set[tuple[int, int, int]] = set()
    rows: list[dict[str, Any]] = []
    cluster_id = 0

    for key in sorted(keys):
        if key in visited:
            continue
        cluster_id += 1
        queue = [key]
        component: list[tuple[int, int, int]] = []
        visited.add(key)

        while queue:
            current = queue.pop(0)
            component.append(current)
            cp, cl, ch = current
            for dp, dl, dh in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                ip = period_pos[cp] + dp
                il = low_pos[cl] + dl
                ih = high_pos[ch] + dh
                if ip < 0 or ip >= len(periods) or il < 0 or il >= len(lows) or ih < 0 or ih >= len(highs):
                    continue
                neighbor = (periods[ip], lows[il], highs[ih])
                if neighbor not in keys or neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

        component_set = set(component)
        comp_df = robust.loc[
            robust.apply(
                lambda row: (int(row["atr_period"]), int(row["q_low_pct"]), int(row["q_high_pct"])) in component_set,
                axis=1,
            )
        ].copy()
        rows.append(
            {
                "cluster_id": cluster_id,
                "n_cells": int(len(comp_df)),
                "period_min": int(comp_df["atr_period"].min()),
                "period_max": int(comp_df["atr_period"].max()),
                "q_low_min": int(comp_df["q_low_pct"].min()),
                "q_low_max": int(comp_df["q_low_pct"].max()),
                "q_high_min": int(comp_df["q_high_pct"].min()),
                "q_high_max": int(comp_df["q_high_pct"].max()),
                "avg_oos_score": float(comp_df["oos_composite_score"].mean()),
                "avg_oos_sharpe": float(comp_df["oos_sharpe"].mean()),
                "avg_oos_profit_factor": float(comp_df["oos_profit_factor"].mean()),
                "avg_oos_expectancy": float(comp_df["oos_expectancy"].mean()),
                "cells": ";".join([f"({p},{ql},{qh})" for p, ql, qh in sorted(component)]),
            }
        )

    cluster_df = pd.DataFrame(rows)
    if cluster_df.empty:
        return cluster_df
    return cluster_df.sort_values(["n_cells", "avg_oos_score"], ascending=[False, False]).reset_index(drop=True)


def _evaluate_point_grid(
    candidate_df: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    baseline: BaselineSpec,
    grid: SearchGrid,
    all_sessions: list,
    is_sessions: list,
    oos_sessions: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    point_pass_frame = candidate_df[["session_date", "signal_index"]].copy()
    pass_columns: dict[str, pd.Series] = {}

    is_mask = candidate_df["session_date"].isin(set(is_sessions))
    for period in grid.atr_periods:
        atr_col = f"atr_{period}"
        if atr_col not in candidate_df.columns:
            continue
        atr_all = pd.to_numeric(candidate_df[atr_col], errors="coerce")
        atr_is_values = atr_all.loc[is_mask].dropna()
        if atr_is_values.empty:
            continue

        for q_low in grid.q_lows_pct:
            for q_high in grid.q_highs_pct:
                if q_low >= q_high:
                    continue

                low_thr = _quantile(atr_is_values, float(q_low) / 100.0)
                high_thr = _quantile(atr_is_values, float(q_high) / 100.0)
                if low_thr is None or high_thr is None or low_thr >= high_thr:
                    continue

                model_id = f"atr{period}_q{int(q_low)}_q{int(q_high)}"
                pass_col = f"pass__{model_id}"
                pass_mask = atr_all.between(low_thr, high_thr, inclusive="both").fillna(False)
                pass_columns[pass_col] = pass_mask.astype(bool)

                selected_sessions = sorted(set(candidate_df.loc[pass_mask, "session_date"]))
                trades = baseline_trades.loc[baseline_trades["session_date"].isin(set(selected_sessions))].copy()

                overall = compute_campaign_metrics(trades, all_sessions, baseline.account_size_usd)
                is_metrics = compute_campaign_metrics(
                    trades.loc[trades["session_date"].isin(set(is_sessions))].copy(),
                    is_sessions,
                    baseline.account_size_usd,
                )
                oos_metrics = compute_campaign_metrics(
                    trades.loc[trades["session_date"].isin(set(oos_sessions))].copy(),
                    oos_sessions,
                    baseline.account_size_usd,
                )

                rows.append(
                    {
                        "config_type": "point_cell",
                        "config_id": model_id,
                        "aggregation_rule": "point",
                        "threshold": np.nan,
                        "atr_period": int(period),
                        "q_low_pct": int(q_low),
                        "q_high_pct": int(q_high),
                        "pair": f"q{int(q_low)}/q{int(q_high)}",
                        "low_threshold_is": float(low_thr),
                        "high_threshold_is": float(high_thr),
                        "n_subsignals": 1,
                        "selected_days_total": int(len(selected_sessions)),
                        "selected_days_is": int(sum(day in set(is_sessions) for day in selected_sessions)),
                        "selected_days_oos": int(sum(day in set(oos_sessions) for day in selected_sessions)),
                        "selection_rate_total": _safe_div(len(selected_sessions), len(all_sessions), default=0.0),
                        **_prefix_metrics(overall, "overall"),
                        **_prefix_metrics(is_metrics, "is"),
                        **_prefix_metrics(oos_metrics, "oos"),
                    }
                )

    point_results = pd.DataFrame(rows)
    if point_results.empty:
        raise RuntimeError("No valid point-cell ATR models were produced.")

    if pass_columns:
        point_pass_frame = pd.concat([point_pass_frame, pd.DataFrame(pass_columns)], axis=1)

    point_results = _score_point_robustness(point_results)
    point_results = point_results.sort_values(
        ["oos_composite_score", "oos_return_over_drawdown", "oos_profit_factor", "oos_sharpe"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return point_results, point_pass_frame


def _score_ensemble_rows(ensemble_results: pd.DataFrame) -> pd.DataFrame:
    if ensemble_results.empty:
        return ensemble_results.copy()

    out = ensemble_results.copy()
    out["stability_gap"] = (
        (out["oos_sharpe"] - out["is_sharpe"]).abs() / out["is_sharpe"].abs().clip(lower=0.25)
        + (out["oos_profit_factor"] - out["is_profit_factor"]).abs() / out["is_profit_factor"].abs().clip(lower=0.25)
        + (out["oos_expectancy"] - out["is_expectancy"]).abs() / out["is_expectancy"].abs().clip(lower=1.0)
    ) / 3.0
    out["ensemble_robustness_score"] = out["oos_composite_score"] - 0.35 * out["stability_gap"]
    return out.sort_values(
        ["ensemble_robustness_score", "oos_composite_score", "oos_profit_factor", "oos_sharpe"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _evaluate_ensemble_rules(
    candidate_df: pd.DataFrame,
    point_pass_frame: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    baseline: BaselineSpec,
    grid: SearchGrid,
    all_sessions: list,
    is_sessions: list,
    oos_sessions: list,
) -> pd.DataFrame:
    pass_cols = [col for col in point_pass_frame.columns if col.startswith("pass__")]
    if not pass_cols:
        raise RuntimeError("The ensemble evaluation requires at least one point sub-signal.")

    ensemble_rows: list[dict[str, Any]] = []
    scored_candidates = point_pass_frame.copy()
    scored_candidates["pass_count"] = scored_candidates[pass_cols].sum(axis=1)
    scored_candidates["consensus_score"] = scored_candidates["pass_count"] / float(len(pass_cols))

    for rule in grid.aggregation_rules:
        threshold = resolve_aggregation_threshold(rule)
        selected = scored_candidates.loc[scored_candidates["consensus_score"] >= threshold].copy()
        selected_sessions = sorted(set(selected["session_date"]))
        trades = baseline_trades.loc[baseline_trades["session_date"].isin(set(selected_sessions))].copy()

        overall = compute_campaign_metrics(trades, all_sessions, baseline.account_size_usd)
        is_metrics = compute_campaign_metrics(
            trades.loc[trades["session_date"].isin(set(is_sessions))].copy(),
            is_sessions,
            baseline.account_size_usd,
        )
        oos_metrics = compute_campaign_metrics(
            trades.loc[trades["session_date"].isin(set(oos_sessions))].copy(),
            oos_sessions,
            baseline.account_size_usd,
        )

        ensemble_rows.append(
            {
                "config_type": "ensemble_rule",
                "config_id": f"ensemble__{rule}",
                "aggregation_rule": rule,
                "threshold": float(threshold),
                "atr_period": np.nan,
                "q_low_pct": np.nan,
                "q_high_pct": np.nan,
                "pair": "",
                "low_threshold_is": np.nan,
                "high_threshold_is": np.nan,
                "n_subsignals": int(len(pass_cols)),
                "selected_days_total": int(len(selected_sessions)),
                "selected_days_is": int(sum(day in set(is_sessions) for day in selected_sessions)),
                "selected_days_oos": int(sum(day in set(oos_sessions) for day in selected_sessions)),
                "selection_rate_total": _safe_div(len(selected_sessions), len(all_sessions), default=0.0),
                **_prefix_metrics(overall, "overall"),
                **_prefix_metrics(is_metrics, "is"),
                **_prefix_metrics(oos_metrics, "oos"),
            }
        )

    ensemble_results = pd.DataFrame(ensemble_rows)
    if ensemble_results.empty:
        raise RuntimeError("No ensemble aggregation rule was evaluated.")
    return _score_ensemble_rows(ensemble_results)


def _select_best_rows(
    point_results: pd.DataFrame,
    ensemble_results: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    best_cell = point_results.iloc[0].to_dict()

    robust_cells = point_results.loc[point_results["robust_flag"]].copy()
    if robust_cells.empty:
        robust_cells = point_results.copy()
    robust_cell = robust_cells.sort_values(
        ["local_robustness_score", "oos_composite_score", "oos_profit_factor"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()

    baseline_transfer_row = ensemble_results.loc[ensemble_results["aggregation_rule"] == "majority_50"].copy()
    if baseline_transfer_row.empty:
        baseline_transfer = ensemble_results.iloc[0].to_dict()
    else:
        baseline_transfer = baseline_transfer_row.iloc[0].to_dict()

    best_ensemble = ensemble_results.iloc[0].to_dict()
    robust_ensemble = ensemble_results.sort_values(
        ["ensemble_robustness_score", "oos_composite_score", "oos_profit_factor"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()

    baseline_like_candidates = ensemble_results.copy()
    best_score = float(best_ensemble["oos_composite_score"])
    baseline_like_candidates["threshold_distance_from_baseline"] = (
        baseline_like_candidates["threshold"] - resolve_aggregation_threshold("majority_50")
    ).abs()
    solid = baseline_like_candidates.loc[
        (baseline_like_candidates["oos_composite_score"] >= 0.90 * best_score)
        & (baseline_like_candidates["oos_profit_factor"] >= 0.95 * float(best_ensemble["oos_profit_factor"]))
    ].copy()
    if solid.empty:
        solid = baseline_like_candidates.copy()
    baseline_like = solid.sort_values(
        ["threshold_distance_from_baseline", "ensemble_robustness_score", "oos_composite_score"],
        ascending=[True, False, False],
    ).iloc[0].to_dict()

    return best_cell, robust_cell, baseline_transfer, best_ensemble, robust_ensemble, baseline_like


def analyze_symbol(
    symbol: str,
    baseline: BaselineSpec | None = None,
    grid: SearchGrid | None = None,
    is_fraction: float = 0.70,
    dataset_path: Path | None = None,
    data_timeframe: str | None = None,
) -> SymbolAnalysis:
    """Run the full Ensemble ORB transfer analysis for one symbol."""
    baseline_spec = baseline or BaselineSpec()
    grid_spec = grid or SearchGrid()
    path = dataset_path or resolve_processed_dataset(symbol, timeframe=data_timeframe)
    instrument_spec = get_instrument_spec(symbol)

    feat = _prepare_feature_dataset(path, baseline_spec, grid_spec)
    session_sanity = _build_session_sanity(feat, baseline_spec)
    all_sessions = sorted(
        pd.to_datetime(session_sanity.loc[session_sanity["has_opening_range"], "session_date"]).dt.date.unique()
    )
    is_sessions, oos_sessions = _split_sessions(all_sessions, is_fraction)

    strategy = _build_strategy(baseline_spec, tick_size=float(instrument_spec["tick_size"]))
    signal_df = strategy.generate_signals(feat)
    baseline_trades = _run_baseline_backtest(signal_df, baseline_spec, instrument_spec)
    candidate_df = _selected_candidate_rows(signal_df, grid_spec.atr_periods)

    baseline_metrics_overall = compute_campaign_metrics(baseline_trades, all_sessions, baseline_spec.account_size_usd)
    baseline_metrics_is = compute_campaign_metrics(
        baseline_trades.loc[baseline_trades["session_date"].isin(set(is_sessions))].copy(),
        is_sessions,
        baseline_spec.account_size_usd,
    )
    baseline_metrics_oos = compute_campaign_metrics(
        baseline_trades.loc[baseline_trades["session_date"].isin(set(oos_sessions))].copy(),
        oos_sessions,
        baseline_spec.account_size_usd,
    )

    point_results, point_pass_frame = _evaluate_point_grid(
        candidate_df=candidate_df,
        baseline_trades=baseline_trades,
        baseline=baseline_spec,
        grid=grid_spec,
        all_sessions=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
    )
    ensemble_results = _evaluate_ensemble_rules(
        candidate_df=candidate_df,
        point_pass_frame=point_pass_frame,
        baseline_trades=baseline_trades,
        baseline=baseline_spec,
        grid=grid_spec,
        all_sessions=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
    )
    robust_clusters = _robust_clusters(point_results, grid_spec)
    best_cell, robust_cell, baseline_transfer, best_ensemble, robust_ensemble, baseline_like = _select_best_rows(
        point_results=point_results,
        ensemble_results=ensemble_results,
    )

    return SymbolAnalysis(
        symbol=symbol.upper(),
        dataset_path=path,
        instrument_spec=instrument_spec,
        baseline=baseline_spec,
        grid=grid_spec,
        feature_df=feat,
        signal_df=signal_df,
        baseline_trades=baseline_trades,
        candidate_df=candidate_df,
        session_sanity=session_sanity,
        all_sessions=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        baseline_metrics_overall=baseline_metrics_overall,
        baseline_metrics_is=baseline_metrics_is,
        baseline_metrics_oos=baseline_metrics_oos,
        point_results=point_results,
        ensemble_results=ensemble_results,
        robust_clusters=robust_clusters,
        best_cell=best_cell,
        robust_cell=robust_cell,
        baseline_transfer=baseline_transfer,
        best_ensemble=best_ensemble,
        robust_ensemble=robust_ensemble,
        baseline_like_ensemble=baseline_like,
    )


def build_benchmark_curve(feature_df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """Build a simple buy-and-hold benchmark on daily closes."""
    bench_src = feature_df[["timestamp", "close", "session_date"]].copy()
    bench_src["timestamp"] = pd.to_datetime(bench_src["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    bench_src["close"] = pd.to_numeric(bench_src["close"], errors="coerce")
    bench_src = bench_src.dropna(subset=["timestamp", "close"])
    daily_close = bench_src.groupby("session_date")["close"].last().dropna()
    if daily_close.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "drawdown", "drawdown_pct"])

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(daily_close.index),
            "equity": float(initial_capital) * (daily_close / float(daily_close.iloc[0])),
        }
    ).sort_values("timestamp")
    out["peak"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] - out["peak"]
    out["drawdown_pct"] = ((out["equity"] / out["peak"]) - 1.0) * 100.0
    return out.drop(columns=["peak"]).reset_index(drop=True)


def analyze_symbol_cache_pass_matrix(analysis: SymbolAnalysis) -> pd.DataFrame:
    """Rebuild the point pass matrix from cached thresholds for notebook reruns."""
    point_pass = analysis.candidate_df[["session_date", "signal_index"]].copy()
    pass_columns: dict[str, pd.Series] = {}
    for _, row in analysis.point_results.iterrows():
        atr_col = f"atr_{int(row['atr_period'])}"
        if atr_col not in analysis.candidate_df.columns:
            continue
        atr_all = pd.to_numeric(analysis.candidate_df[atr_col], errors="coerce")
        low_thr = float(row["low_threshold_is"])
        high_thr = float(row["high_threshold_is"])
        pass_columns[f"pass__{row['config_id']}"] = atr_all.between(low_thr, high_thr, inclusive="both").fillna(False)
    if pass_columns:
        point_pass = pd.concat([point_pass, pd.DataFrame(pass_columns)], axis=1)
    return point_pass


def _point_trade_subset(analysis: SymbolAnalysis, config_id: str) -> pd.DataFrame:
    pass_matrix = analyze_symbol_cache_pass_matrix(analysis)
    pass_col = f"pass__{config_id}"
    if pass_col not in pass_matrix.columns:
        return pd.DataFrame()
    selected_sessions = set(pass_matrix.loc[pass_matrix[pass_col].fillna(False), "session_date"])
    return analysis.baseline_trades.loc[analysis.baseline_trades["session_date"].isin(selected_sessions)].copy()


def _ensemble_trade_subset(analysis: SymbolAnalysis, aggregation_rule: str) -> pd.DataFrame:
    point_pass = analyze_symbol_cache_pass_matrix(analysis)
    pass_cols = [col for col in point_pass.columns if col.startswith("pass__")]
    scored = point_pass.copy()
    scored["consensus_score"] = scored[pass_cols].sum(axis=1) / float(len(pass_cols))
    threshold = resolve_aggregation_threshold(aggregation_rule)
    selected_sessions = set(scored.loc[scored["consensus_score"] >= threshold, "session_date"])
    return analysis.baseline_trades.loc[analysis.baseline_trades["session_date"].isin(selected_sessions)].copy()


def build_notebook_bundle(
    symbol: str,
    baseline: BaselineSpec | None = None,
    grid: SearchGrid | None = None,
    aggregation_rule: str = "majority_50",
    is_fraction: float = 0.70,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    """Convenience wrapper used by the client notebooks."""
    analysis = analyze_symbol(
        symbol=symbol,
        baseline=baseline,
        grid=grid,
        is_fraction=is_fraction,
        dataset_path=dataset_path,
    )
    ensemble_trades = _ensemble_trade_subset(analysis, aggregation_rule)
    is_session_set = set(analysis.is_sessions)
    oos_session_set = set(analysis.oos_sessions)
    ensemble_trades_is = ensemble_trades.loc[ensemble_trades["session_date"].isin(is_session_set)].copy()
    ensemble_trades_oos = ensemble_trades.loc[ensemble_trades["session_date"].isin(oos_session_set)].copy()

    ensemble_curve = normalize_curve(
        build_equity_curve(ensemble_trades, initial_capital=analysis.baseline.account_size_usd)
    )
    ensemble_curve_is = normalize_curve(
        build_equity_curve(ensemble_trades_is, initial_capital=analysis.baseline.account_size_usd)
    )
    ensemble_curve_oos = normalize_curve(
        build_equity_curve(ensemble_trades_oos, initial_capital=analysis.baseline.account_size_usd)
    )

    benchmark_curve = build_benchmark_curve(analysis.feature_df, analysis.baseline.account_size_usd)
    selected_row = analysis.ensemble_results.loc[
        analysis.ensemble_results["aggregation_rule"] == aggregation_rule
    ]
    if selected_row.empty:
        selected_ensemble = analysis.best_ensemble
    else:
        selected_ensemble = selected_row.iloc[0].to_dict()

    return {
        "analysis": analysis,
        "selected_ensemble": selected_ensemble,
        "ensemble_trades": ensemble_trades,
        "ensemble_trades_is": ensemble_trades_is,
        "ensemble_trades_oos": ensemble_trades_oos,
        "ensemble_curve": ensemble_curve,
        "ensemble_curve_is": ensemble_curve_is,
        "ensemble_curve_oos": ensemble_curve_oos,
        "benchmark_curve": benchmark_curve,
    }


def _plot_heatmap(
    point_results: pd.DataFrame,
    value_col: str,
    title: str,
    output_path: Path,
    cmap: str = "viridis",
    reverse: bool = False,
) -> None:
    heat = (
        point_results.pivot_table(index="atr_period", columns="pair", values=value_col, aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(16.0, 4.8))
    if heat.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
    else:
        values = heat.to_numpy(dtype=float)
        color_map = plt.get_cmap(cmap)
        if reverse:
            color_map = color_map.reversed()
        im = ax.imshow(values, aspect="auto", cmap=color_map)
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels(list(heat.columns), rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels([str(int(v)) for v in heat.index], fontsize=9)
        ax.set_xlabel("Quantile pair")
        ax.set_ylabel("ATR window")
        ax.set_title(title)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                text = "nan" if not math.isfinite(value) else f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color="black")
        fig.colorbar(im, ax=ax, shrink=0.92)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_equity_drawdown_compare(
    trades_map: dict[str, pd.DataFrame],
    initial_capital: float,
    equity_path: Path,
    drawdown_path: Path,
) -> None:
    curves = {
        name: build_equity_curve(frame, initial_capital=initial_capital)
        for name, frame in trades_map.items()
    }

    fig1, ax1 = plt.subplots(figsize=(11.5, 5.0))
    for name, curve in curves.items():
        if curve.empty:
            continue
        ax1.plot(pd.to_datetime(curve["timestamp"]), curve["equity"], label=name, linewidth=1.5)
    ax1.set_title("Equity Comparison")
    ax1.set_ylabel("Equity (USD)")
    ax1.legend(loc="best", fontsize=9)
    fig1.tight_layout()
    fig1.savefig(equity_path, dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(11.5, 5.0))
    for name, curve in curves.items():
        if curve.empty:
            continue
        ax2.plot(pd.to_datetime(curve["timestamp"]), curve["drawdown"], label=name, linewidth=1.5)
    ax2.set_title("Drawdown Comparison")
    ax2.set_ylabel("Drawdown (USD)")
    ax2.legend(loc="best", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(drawdown_path, dpi=150)
    plt.close(fig2)


def _cross_asset_metrics_plot(compare_df: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("sharpe", "Sharpe"),
        ("profit_factor", "Profit Factor"),
        ("expectancy", "Expectancy"),
        ("max_drawdown_abs", "|Max DD|"),
        ("return_over_drawdown", "Return/DD"),
        ("pct_days_traded", "% Days Traded"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0))
    axes = axes.flatten()
    labels = compare_df["label"].tolist()

    for ax, (column, title) in zip(axes, metrics):
        ax.bar(labels, compare_df[column].astype(float).tolist())
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _recommendation_table(analysis: SymbolAnalysis) -> pd.DataFrame:
    rows = [
        {
            "recommendation": "best_oos_ensemble",
            **analysis.best_ensemble,
        },
        {
            "recommendation": "most_robust_ensemble",
            **analysis.robust_ensemble,
        },
        {
            "recommendation": "baseline_like_ensemble",
            **analysis.baseline_like_ensemble,
        },
        {
            "recommendation": "best_oos_cell",
            **analysis.best_cell,
        },
        {
            "recommendation": "most_robust_cell",
            **analysis.robust_cell,
        },
    ]
    return pd.DataFrame(rows)


def _export_symbol_analysis(analysis: SymbolAnalysis, output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    charts_dir = output_root / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    symbol_lower = analysis.symbol.lower()
    combined = pd.concat([analysis.point_results, analysis.ensemble_results], ignore_index=True, sort=False)
    results_path = output_root / f"{symbol_lower}_results_full.csv"
    tops_path = output_root / f"{symbol_lower}_top_configs.csv"
    summary_path = output_root / f"{symbol_lower}_summary.md"
    aggregation_path = output_root / f"{symbol_lower}_aggregation_summary.csv"

    combined.to_csv(results_path, index=False)
    recommendation_df = _recommendation_table(analysis)
    recommendation_df.to_csv(tops_path, index=False)
    analysis.ensemble_results.to_csv(aggregation_path, index=False)

    _plot_heatmap(
        analysis.point_results,
        value_col="oos_sharpe",
        title=f"{analysis.symbol} OOS Sharpe",
        output_path=charts_dir / f"{symbol_lower}_heatmap_sharpe.png",
        cmap="RdYlGn",
    )
    _plot_heatmap(
        analysis.point_results,
        value_col="oos_profit_factor",
        title=f"{analysis.symbol} OOS Profit Factor",
        output_path=charts_dir / f"{symbol_lower}_heatmap_pf.png",
        cmap="RdYlGn",
    )
    _plot_heatmap(
        analysis.point_results,
        value_col="oos_net_pnl",
        title=f"{analysis.symbol} OOS Net PnL",
        output_path=charts_dir / f"{symbol_lower}_heatmap_netpnl.png",
        cmap="RdYlGn",
    )
    _plot_heatmap(
        analysis.point_results,
        value_col="oos_max_drawdown_abs",
        title=f"{analysis.symbol} OOS |Max Drawdown|",
        output_path=charts_dir / f"{symbol_lower}_heatmap_maxdd.png",
        cmap="RdYlGn",
        reverse=True,
    )
    _plot_heatmap(
        analysis.point_results,
        value_col="oos_composite_score",
        title=f"{analysis.symbol} OOS Composite Score",
        output_path=charts_dir / f"{symbol_lower}_heatmap_score.png",
        cmap="RdYlGn",
    )

    best_cell_trades = _point_trade_subset(analysis, str(analysis.best_cell["config_id"]))
    best_ensemble_trades = _ensemble_trade_subset(analysis, str(analysis.best_ensemble["aggregation_rule"]))
    baseline_transfer_trades = _ensemble_trade_subset(analysis, str(analysis.baseline_transfer["aggregation_rule"]))
    _plot_equity_drawdown_compare(
        {
            "baseline_transfer": baseline_transfer_trades,
            "best_ensemble": best_ensemble_trades,
            "best_cell": best_cell_trades,
        },
        initial_capital=analysis.baseline.account_size_usd,
        equity_path=charts_dir / f"{symbol_lower}_equity_top.png",
        drawdown_path=charts_dir / f"{symbol_lower}_drawdown_top.png",
    )

    cluster_lines = ["- No robust cluster detected."] if analysis.robust_clusters.empty else [
        f"- Cluster {int(row['cluster_id'])}: {int(row['n_cells'])} cells, "
        f"ATR {int(row['period_min'])}-{int(row['period_max'])}, "
        f"Qlow {int(row['q_low_min'])}-{int(row['q_low_max'])}, "
        f"Qhigh {int(row['q_high_min'])}-{int(row['q_high_max'])}, "
        f"avg score={float(row['avg_oos_score']):.3f}."
        for _, row in analysis.robust_clusters.head(3).iterrows()
    ]

    transfer_ok = (
        float(analysis.best_ensemble["oos_profit_factor"]) > 1.0
        and float(analysis.best_ensemble["oos_return_over_drawdown"]) > 0.25
        and float(analysis.best_ensemble["oos_composite_score"]) > 0.0
    )
    baseline_transfer_gap = float(analysis.best_ensemble["oos_composite_score"]) - float(
        analysis.baseline_transfer["oos_composite_score"]
    )
    summary_lines = [
        f"# {analysis.symbol} Ensemble ORB Summary",
        "",
        f"- Dataset: `{analysis.dataset_path.name}`",
        f"- Tradable sessions analysed: {len(analysis.all_sessions)}",
        f"- IS/OOS split: {len(analysis.is_sessions)} / {len(analysis.oos_sessions)} sessions",
        f"- Instrument specs loaded from config: tick_size={analysis.instrument_spec['tick_size']}, "
        f"tick_value={analysis.instrument_spec['tick_value_usd']}, point_value={analysis.instrument_spec['point_value_usd']}, "
        f"commission_per_side={analysis.instrument_spec['commission_per_side_usd']}, slippage_ticks={analysis.instrument_spec['slippage_ticks']}",
        "",
        "## Baseline Transfer",
        "",
        f"- Baseline transfer rule: `{analysis.baseline_transfer['aggregation_rule']}`",
        f"- OOS score={float(analysis.baseline_transfer['oos_composite_score']):.3f}, "
        f"PF={float(analysis.baseline_transfer['oos_profit_factor']):.3f}, "
        f"Sharpe={float(analysis.baseline_transfer['oos_sharpe']):.3f}, "
        f"Return/DD={float(analysis.baseline_transfer['oos_return_over_drawdown']):.3f}",
        "",
        "## Final Recommendations",
        "",
        f"- Best OOS ensemble: `{analysis.best_ensemble['aggregation_rule']}` "
        f"(score={float(analysis.best_ensemble['oos_composite_score']):.3f}, PF={float(analysis.best_ensemble['oos_profit_factor']):.3f}, "
        f"Sharpe={float(analysis.best_ensemble['oos_sharpe']):.3f}, MaxDD={float(analysis.best_ensemble['oos_max_drawdown']):.2f}).",
        f"- Most robust ensemble: `{analysis.robust_ensemble['aggregation_rule']}` "
        f"(robustness={float(analysis.robust_ensemble['ensemble_robustness_score']):.3f}).",
        f"- Baseline-like ensemble retained if solid: `{analysis.baseline_like_ensemble['aggregation_rule']}`.",
        f"- Best point cell: `ATR {int(analysis.best_cell['atr_period'])} / q{int(analysis.best_cell['q_low_pct'])}/q{int(analysis.best_cell['q_high_pct'])}` "
        f"(score={float(analysis.best_cell['oos_composite_score']):.3f}).",
        f"- Most robust point cell: `ATR {int(analysis.robust_cell['atr_period'])} / q{int(analysis.robust_cell['q_low_pct'])}/q{int(analysis.robust_cell['q_high_pct'])}` "
        f"(local robustness={float(analysis.robust_cell['local_robustness_score']):.3f}).",
        "",
        "## Robust Clusters",
        "",
        *cluster_lines,
        "",
        "## Campaign Readout",
        "",
        f"- Transferability verdict: {'transfer acceptable' if transfer_ok else 'transfer weak / requires caution'}.",
        f"- Baseline MNQ-style majority rule gap vs best ensemble: {baseline_transfer_gap:.3f} score points.",
        f"- Sessions with incomplete opening window: {int((~analysis.session_sanity['opening_window_complete']).sum())}.",
        f"- Sessions missing part of RTH bars: {int((analysis.session_sanity['rth_missing_bars'] > 0).sum())}.",
        "",
        "## Exports",
        "",
        f"- Full results: `{results_path.name}`",
        f"- Top configs: `{tops_path.name}`",
        f"- Aggregation summary: `{aggregation_path.name}`",
        f"- Charts folder: `charts/`",
    ]
    summary_path.write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")

    return {
        "results": results_path,
        "top_configs": tops_path,
        "aggregation_summary": aggregation_path,
        "summary": summary_path,
        "heatmap_sharpe": charts_dir / f"{symbol_lower}_heatmap_sharpe.png",
        "heatmap_pf": charts_dir / f"{symbol_lower}_heatmap_pf.png",
        "heatmap_netpnl": charts_dir / f"{symbol_lower}_heatmap_netpnl.png",
        "heatmap_maxdd": charts_dir / f"{symbol_lower}_heatmap_maxdd.png",
        "heatmap_score": charts_dir / f"{symbol_lower}_heatmap_score.png",
        "equity_top": charts_dir / f"{symbol_lower}_equity_top.png",
        "drawdown_top": charts_dir / f"{symbol_lower}_drawdown_top.png",
    }


def _data_sanity_markdown(
    symbol_analyses: list[SymbolAnalysis],
    output_path: Path,
) -> None:
    lines = [
        "# Data Sanity Check",
        "",
        "The campaign uses already processed datasets from `data/processed/parquet`.",
        "",
    ]
    for analysis in symbol_analyses:
        sanity = analysis.session_sanity
        first_ts = pd.to_datetime(analysis.feature_df["timestamp"]).min()
        last_ts = pd.to_datetime(analysis.feature_df["timestamp"]).max()
        duplicate_timestamps = int(analysis.feature_df["timestamp"].duplicated().sum())
        bad_ohlc = int(
            (
                (analysis.feature_df["high"] < analysis.feature_df["low"])
                | (analysis.feature_df["open"] < analysis.feature_df["low"])
                | (analysis.feature_df["open"] > analysis.feature_df["high"])
                | (analysis.feature_df["close"] < analysis.feature_df["low"])
                | (analysis.feature_df["close"] > analysis.feature_df["high"])
            ).sum()
        )
        lines.extend(
            [
                f"## {analysis.symbol}",
                "",
                f"- File: `{analysis.dataset_path.name}`",
                f"- Rows: {len(analysis.feature_df):,}",
                f"- Date range: {first_ts} -> {last_ts}",
                f"- Timezone: `{getattr(first_ts.tz, 'zone', first_ts.tz)}`",
                f"- Duplicate timestamps after cleaning: {duplicate_timestamps}",
                f"- OHLC incoherent rows: {bad_ohlc}",
                f"- Sessions with OR available: {int(sanity['has_opening_range'].sum())}",
                f"- Sessions with incomplete {int(analysis.baseline.or_minutes)}-minute opening window: {int((~sanity['opening_window_complete']).sum())}",
                f"- Sessions missing at least one RTH bar: {int((sanity['rth_missing_bars'] > 0).sum())}",
                f"- Median RTH missing bars per session: {float(sanity['rth_missing_bars'].median()):.1f}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _cross_asset_summary(
    reference_analysis: SymbolAnalysis,
    target_analyses: list[SymbolAnalysis],
    output_root: Path,
) -> dict[str, Path]:
    charts_dir = output_root / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    compare_rows: list[dict[str, Any]] = []
    trades_map: dict[str, pd.DataFrame] = {
        f"{reference_analysis.symbol}_baseline": _ensemble_trade_subset(reference_analysis, "majority_50")
    }
    compare_rows.append(
        {
            "label": f"{reference_analysis.symbol}_baseline",
            "symbol": reference_analysis.symbol,
            "selection": "baseline_majority_50",
            "sharpe": float(reference_analysis.baseline_transfer["oos_sharpe"]),
            "profit_factor": float(reference_analysis.baseline_transfer["oos_profit_factor"]),
            "expectancy": float(reference_analysis.baseline_transfer["oos_expectancy"]),
            "max_drawdown_abs": float(reference_analysis.baseline_transfer["oos_max_drawdown_abs"]),
            "return_over_drawdown": float(reference_analysis.baseline_transfer["oos_return_over_drawdown"]),
            "nb_trades": int(reference_analysis.baseline_transfer["oos_nb_trades"]),
            "pct_days_traded": float(reference_analysis.baseline_transfer["oos_pct_days_traded"]),
            "composite_score": float(reference_analysis.baseline_transfer["oos_composite_score"]),
        }
    )

    for analysis in target_analyses:
        best_rule = str(analysis.best_ensemble["aggregation_rule"])
        label = f"{analysis.symbol}_best"
        compare_rows.append(
            {
                "label": label,
                "symbol": analysis.symbol,
                "selection": best_rule,
                "sharpe": float(analysis.best_ensemble["oos_sharpe"]),
                "profit_factor": float(analysis.best_ensemble["oos_profit_factor"]),
                "expectancy": float(analysis.best_ensemble["oos_expectancy"]),
                "max_drawdown_abs": float(analysis.best_ensemble["oos_max_drawdown_abs"]),
                "return_over_drawdown": float(analysis.best_ensemble["oos_return_over_drawdown"]),
                "nb_trades": int(analysis.best_ensemble["oos_nb_trades"]),
                "pct_days_traded": float(analysis.best_ensemble["oos_pct_days_traded"]),
                "composite_score": float(analysis.best_ensemble["oos_composite_score"]),
            }
        )
        trades_map[label] = _ensemble_trade_subset(analysis, best_rule)

    compare_df = pd.DataFrame(compare_rows)
    compare_path = output_root / "cross_asset_comparison.csv"
    compare_df.to_csv(compare_path, index=False)

    _plot_equity_drawdown_compare(
        trades_map=trades_map,
        initial_capital=reference_analysis.baseline.account_size_usd,
        equity_path=charts_dir / "cross_asset_equity_compare.png",
        drawdown_path=charts_dir / "cross_asset_drawdown_compare.png",
    )
    _cross_asset_metrics_plot(compare_df, charts_dir / "cross_asset_metrics_compare.png")

    mes = next((a for a in target_analyses if a.symbol == "MES"), None)
    m2k = next((a for a in target_analyses if a.symbol == "M2K"), None)
    best_target = max(target_analyses, key=lambda item: float(item.best_ensemble["oos_composite_score"]))
    mes_transfer = "yes" if mes and float(mes.best_ensemble["oos_composite_score"]) > 0 else "no"
    m2k_transfer = "yes" if m2k and float(m2k.best_ensemble["oos_composite_score"]) > 0 else "no"

    def _cluster_answer(analysis: SymbolAnalysis | None) -> str:
        if analysis is None or analysis.robust_clusters.empty:
            return "No robust cluster identified."
        top = analysis.robust_clusters.iloc[0]
        return (
            f"Yes, around ATR {int(top['period_min'])}-{int(top['period_max'])}, "
            f"Qlow {int(top['q_low_min'])}-{int(top['q_low_max'])}, "
            f"Qhigh {int(top['q_high_min'])}-{int(top['q_high_max'])}."
        )

    summary_lines = [
        "# Cross-Asset Summary",
        "",
        "## Comparison Table",
        "",
        compare_df.to_string(index=False),
        "",
        "## Direct Answers",
        "",
        f"1. MES transferability: {mes_transfer}.",
        f"2. M2K transferability: {m2k_transfer}.",
        (
            "3. MNQ parameters transfer as-is: mostly yes."
            if all(float(a.baseline_transfer["oos_composite_score"]) >= 0.90 * float(a.best_ensemble["oos_composite_score"]) for a in target_analyses)
            else "3. MNQ parameters transfer partially, but ticker-specific recalibration improves results."
        ),
        f"4. Robust MES cluster: {_cluster_answer(mes)}",
        f"5. Robust M2K cluster: {_cluster_answer(m2k)}",
        f"6. Most promising complement to MNQ: {best_target.symbol}.",
        (
            "7. M2K looks sufficiently differentiated from MNQ."
            if m2k is not None and abs(float(m2k.best_ensemble['oos_pct_days_traded']) - float(reference_analysis.baseline_transfer['oos_pct_days_traded'])) > 0.02
            else "7. M2K looks close to MNQ in trading rhythm."
        ),
        (
            "8. MES adds robustness more than novelty."
            if mes is not None and not mes.robust_clusters.empty and float(mes.best_ensemble["oos_sharpe"]) <= 1.10 * float(reference_analysis.baseline_transfer["oos_sharpe"])
            else "8. MES is not just a smoother clone; it changes the payoff profile materially."
        ),
        "",
        "## Charts",
        "",
        "- `charts/cross_asset_equity_compare.png`",
        "- `charts/cross_asset_metrics_compare.png`",
    ]
    summary_path = output_root / "cross_asset_summary.md"
    summary_path.write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")

    return {
        "comparison": compare_path,
        "summary": summary_path,
        "equity": charts_dir / "cross_asset_equity_compare.png",
        "metrics": charts_dir / "cross_asset_metrics_compare.png",
    }


def run_multi_asset_campaign(config: CampaignConfig) -> dict[str, Any]:
    """Run the complete campaign and export everything under export/."""
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "charts").mkdir(parents=True, exist_ok=True)

    reference_analysis = analyze_symbol(
        symbol=config.reference_symbol,
        baseline=config.baseline,
        grid=config.grid,
        is_fraction=config.is_fraction,
        data_timeframe=config.data_timeframe,
    )
    reference_analysis.export_paths = _export_symbol_analysis(reference_analysis, output_root)

    target_analyses: list[SymbolAnalysis] = []
    for symbol in config.symbols:
        analysis = analyze_symbol(
            symbol=symbol,
            baseline=config.baseline,
            grid=config.grid,
            is_fraction=config.is_fraction,
            data_timeframe=config.data_timeframe,
        )
        analysis.export_paths = _export_symbol_analysis(analysis, output_root)
        target_analyses.append(analysis)

    sanity_path = output_root / "data_sanity_check.md"
    _data_sanity_markdown([reference_analysis, *target_analyses], sanity_path)
    cross_paths = _cross_asset_summary(reference_analysis, target_analyses, output_root)

    metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "config": {
            **asdict(config),
            "output_root": str(output_root),
        },
        "datasets": {
            reference_analysis.symbol: str(reference_analysis.dataset_path),
            **{analysis.symbol: str(analysis.dataset_path) for analysis in target_analyses},
        },
    }
    metadata_path = output_root / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return {
        "reference": reference_analysis,
        "targets": target_analyses,
        "data_sanity_check": sanity_path,
        "cross_asset": cross_paths,
        "run_metadata": metadata_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-asset Ensemble ORB transfer campaign.")
    parser.add_argument("--symbols", nargs="*", default=["MES", "M2K"])
    parser.add_argument("--reference-symbol", default="MNQ")
    parser.add_argument("--is-fraction", type=float, default=0.70)
    parser.add_argument("--data-timeframe", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    config = CampaignConfig(
        symbols=tuple(str(symbol).upper() for symbol in args.symbols),
        reference_symbol=str(args.reference_symbol).upper(),
        is_fraction=float(args.is_fraction),
        data_timeframe=str(args.data_timeframe).lower() if args.data_timeframe else None,
        output_root=Path(args.output_root),
    )
    artifacts = run_multi_asset_campaign(config)
    print(f"data_sanity_check: {artifacts['data_sanity_check']}")
    print(f"cross_asset_summary: {artifacts['cross_asset']['summary']}")
    for analysis in artifacts["targets"]:
        print(f"{analysis.symbol}_summary: {analysis.export_paths['summary']}")


if __name__ == "__main__":
    main()
