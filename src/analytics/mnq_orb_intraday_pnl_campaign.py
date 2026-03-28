"""Leak-free intraday PnL overlay campaign for the official MNQ ORB baseline."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.intraday_pnl_overlay import (
    IntradayPnlOverlaySpec,
    OverlayBacktestResult,
    run_intraday_pnl_overlay_backtest,
)
from src.analytics.metrics import compute_metrics
from src.analytics.orb_multi_asset_campaign import (
    BaselineSpec,
    SearchGrid,
    SymbolAnalysis,
    analyze_symbol,
    analyze_symbol_cache_pass_matrix,
    resolve_aggregation_threshold,
    resolve_processed_dataset,
)
from src.config.orb_campaign import PropConstraintConfig, build_prop_constraints
from src.config.paths import EXPORTS_DIR, ensure_directories
from src.config.settings import get_instrument_spec
from src.engine.execution_model import ExecutionModel


SUMMARY_COLUMNS = [
    "variant_name",
    "family",
    "description",
    "calibration_scope",
    "parameters_json",
    "note",
    "verdict",
    "screening_score",
    "validation_score",
    "structurally_identical_to_baseline",
    "overall_net_pnl",
    "overall_sharpe",
    "overall_sortino",
    "overall_profit_factor",
    "overall_expectancy",
    "overall_max_drawdown",
    "overall_n_trades",
    "overall_hit_rate",
    "overall_avg_win",
    "overall_avg_loss",
    "overall_n_days_traded",
    "overall_pct_days_traded",
    "overall_avg_trades_per_day",
    "overall_mean_daily_pnl",
    "overall_median_daily_pnl",
    "overall_worst_day",
    "overall_pct_days_gt_plus_1r",
    "overall_pct_days_lt_minus_1r",
    "overall_longest_losing_streak_daily",
    "overall_longest_losing_streak_trade",
    "overall_cut_day_freq",
    "overall_hard_loss_cap_freq",
    "overall_profit_lock_freq",
    "overall_giveback_freq",
    "overall_daily_loss_limit_breach_freq",
    "overall_expected_profit_per_trading_day",
    "is_net_pnl",
    "is_sharpe",
    "is_sortino",
    "is_profit_factor",
    "is_expectancy",
    "is_max_drawdown",
    "is_n_trades",
    "is_hit_rate",
    "is_avg_win",
    "is_avg_loss",
    "is_n_days_traded",
    "is_pct_days_traded",
    "is_avg_trades_per_day",
    "is_mean_daily_pnl",
    "is_median_daily_pnl",
    "is_worst_day",
    "is_pct_days_gt_plus_1r",
    "is_pct_days_lt_minus_1r",
    "is_longest_losing_streak_daily",
    "is_longest_losing_streak_trade",
    "is_cut_day_freq",
    "is_hard_loss_cap_freq",
    "is_profit_lock_freq",
    "is_giveback_freq",
    "is_daily_loss_limit_breach_freq",
    "is_expected_profit_per_trading_day",
    "oos_net_pnl",
    "oos_sharpe",
    "oos_sortino",
    "oos_profit_factor",
    "oos_expectancy",
    "oos_max_drawdown",
    "oos_n_trades",
    "oos_hit_rate",
    "oos_avg_win",
    "oos_avg_loss",
    "oos_n_days_traded",
    "oos_pct_days_traded",
    "oos_avg_trades_per_day",
    "oos_mean_daily_pnl",
    "oos_median_daily_pnl",
    "oos_worst_day",
    "oos_pct_days_gt_plus_1r",
    "oos_pct_days_lt_minus_1r",
    "oos_longest_losing_streak_daily",
    "oos_longest_losing_streak_trade",
    "oos_cut_day_freq",
    "oos_hard_loss_cap_freq",
    "oos_profit_lock_freq",
    "oos_giveback_freq",
    "oos_daily_loss_limit_breach_freq",
    "oos_expected_profit_per_trading_day",
    "is_net_pnl_retention_vs_baseline",
    "is_sharpe_delta_vs_baseline",
    "is_profit_factor_delta_vs_baseline",
    "is_expectancy_delta_vs_baseline",
    "is_max_drawdown_improvement_vs_baseline",
    "is_worst_day_improvement_vs_baseline",
    "is_daily_loss_limit_breach_reduction_vs_baseline",
    "is_cut_day_freq_delta_vs_baseline",
    "oos_net_pnl_retention_vs_baseline",
    "oos_sharpe_delta_vs_baseline",
    "oos_profit_factor_delta_vs_baseline",
    "oos_expectancy_delta_vs_baseline",
    "oos_max_drawdown_improvement_vs_baseline",
    "oos_worst_day_improvement_vs_baseline",
    "oos_daily_loss_limit_breach_reduction_vs_baseline",
    "oos_cut_day_freq_delta_vs_baseline",
]


@dataclass
class VariantRun:
    name: str
    family: str
    description: str
    calibration_scope: str
    parameters: dict[str, Any]
    trades: pd.DataFrame
    daily_results: pd.DataFrame
    controls: pd.DataFrame
    state_transitions: pd.DataFrame
    summary_by_scope: pd.DataFrame
    note: str = ""


@dataclass(frozen=True)
class MnqIntradayPnlCampaignSpec:
    symbol: str = "MNQ"
    dataset_path: Path | None = None
    is_fraction: float = 0.70
    aggregation_rule: str = "majority_50"
    baseline: BaselineSpec = BaselineSpec(
        or_minutes=30,
        opening_time="09:30:00",
        direction="both",
        one_trade_per_day=True,
        entry_buffer_ticks=2,
        stop_buffer_ticks=2,
        target_multiple=2.0,
        vwap_confirmation=True,
        vwap_column="continuous_session_vwap",
        time_exit="16:00:00",
        account_size_usd=50_000.0,
        risk_per_trade_pct=1.5,
        entry_on_next_open=True,
    )
    grid: SearchGrid = SearchGrid(
        atr_periods=(25, 26, 27, 28, 29, 30),
        q_lows_pct=(25, 26, 27, 28, 29, 30),
        q_highs_pct=(90, 91, 92, 93, 94, 95),
        aggregation_rules=("majority_50", "consensus_75", "unanimity_100"),
    )
    fixed_contracts: int = 1
    initial_capital_usd: float = 50_000.0
    commission_per_side_usd: float | None = 0.62
    slippage_ticks: float | None = 1.0
    prop_constraints: PropConstraintConfig = field(default_factory=build_prop_constraints)
    output_root: Path | None = None


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
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    clean = {key: _serialize_value(value) for key, value in payload.items()}
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return float(default)
    out = numerator / denominator
    return float(out) if math.isfinite(out) else float(default)


def _selected_ensemble_sessions(analysis: SymbolAnalysis, aggregation_rule: str) -> set:
    point_pass = analyze_symbol_cache_pass_matrix(analysis)
    pass_cols = [column for column in point_pass.columns if column.startswith("pass__")]
    if not pass_cols:
        return set()
    scored = point_pass.copy()
    scored["consensus_score"] = scored[pass_cols].sum(axis=1) / float(len(pass_cols))
    threshold = resolve_aggregation_threshold(aggregation_rule)
    return set(pd.to_datetime(scored.loc[scored["consensus_score"] >= threshold, "session_date"]).dt.date)


def _subset_frame_by_sessions(frame: pd.DataFrame, sessions: list) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    session_set = set(pd.to_datetime(pd.Index(sessions)).date)
    view = frame.copy()
    view_dates = pd.to_datetime(view["session_date"]).dt.date
    return view.loc[view_dates.isin(session_set)].copy().reset_index(drop=True)


def _negative_streak_lengths(values: pd.Series) -> list[int]:
    streaks: list[int] = []
    current = 0
    for value in pd.Series(values, dtype=float).fillna(0.0):
        if value < 0:
            current += 1
        elif current > 0:
            streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return streaks


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


def _ensure_daily_scope_frame(daily_results: pd.DataFrame, sessions: list) -> pd.DataFrame:
    session_index = pd.Index(pd.to_datetime(pd.Index(sessions)).date)
    daily = pd.DataFrame({"session_date": session_index})
    if daily_results.empty:
        daily["daily_pnl_usd"] = 0.0
        daily["daily_pnl_r"] = 0.0
        daily["daily_trade_count"] = 0
        daily["daily_loss_count"] = 0
        daily["peak_day_pnl_usd"] = 0.0
        daily["peak_day_pnl_r"] = 0.0
        daily["max_giveback_usd"] = 0.0
        daily["max_giveback_r"] = 0.0
        daily["day_cut_by_rule"] = False
        daily["hard_loss_cap_triggered"] = False
        daily["hard_profit_lock_triggered"] = False
        daily["giveback_triggered"] = False
        return daily
    view = daily_results.copy()
    view["session_date"] = pd.to_datetime(view["session_date"]).dt.date
    out = daily.merge(view, on="session_date", how="left")
    fill_map = {
        "daily_pnl_usd": 0.0,
        "daily_pnl_r": 0.0,
        "daily_trade_count": 0,
        "daily_loss_count": 0,
        "peak_day_pnl_usd": 0.0,
        "peak_day_pnl_r": 0.0,
        "max_giveback_usd": 0.0,
        "max_giveback_r": 0.0,
        "day_cut_by_rule": False,
        "hard_loss_cap_triggered": False,
        "hard_profit_lock_triggered": False,
        "giveback_triggered": False,
    }
    return out.fillna(fill_map)


def _phase_column(session_dates: pd.Series, is_sessions: list, oos_sessions: list) -> pd.Series:
    is_set = set(pd.to_datetime(pd.Index(is_sessions)).date)
    oos_set = set(pd.to_datetime(pd.Index(oos_sessions)).date)
    dates = pd.to_datetime(session_dates).dt.date
    return pd.Series(
        np.where(
            dates.isin(is_set),
            "is",
            np.where(dates.isin(oos_set), "oos", "outside"),
        ),
        index=session_dates.index,
    )


def _scope_summary(
    trades: pd.DataFrame,
    daily_results: pd.DataFrame,
    sessions: list,
    initial_capital: float,
    constraints: PropConstraintConfig,
) -> dict[str, Any]:
    daily = _ensure_daily_scope_frame(daily_results, sessions)
    daily = daily.sort_values("session_date").reset_index(drop=True)
    base = compute_metrics(
        trades,
        session_dates=sessions,
        initial_capital=initial_capital,
        prop_constraints=constraints,
    )

    daily_pnl = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0)
    daily_pnl_r = pd.to_numeric(daily["daily_pnl_r"], errors="coerce").fillna(0.0)
    daily_trade_count = pd.to_numeric(daily["daily_trade_count"], errors="coerce").fillna(0.0)
    day_loss_lengths = _negative_streak_lengths(daily_pnl)

    return {
        "net_pnl": float(base.get("cumulative_pnl", 0.0)),
        "sharpe": float(base.get("sharpe_ratio", 0.0)),
        "sortino": _sortino_ratio(daily_pnl, initial_capital),
        "profit_factor": float(base.get("profit_factor", 0.0)),
        "expectancy": float(base.get("expectancy", 0.0)),
        "max_drawdown": float(base.get("max_drawdown", 0.0)),
        "n_trades": int(base.get("n_trades", 0)),
        "hit_rate": float(base.get("win_rate", 0.0)),
        "avg_win": float(base.get("avg_win", 0.0)),
        "avg_loss": float(base.get("avg_loss", 0.0)),
        "n_days_traded": int((daily_trade_count > 0).sum()),
        "pct_days_traded": float(base.get("percent_of_days_traded", 0.0)),
        "avg_trades_per_day": _safe_div(float(base.get("n_trades", 0)), max(len(daily), 1), default=0.0),
        "mean_daily_pnl": float(daily_pnl.mean()) if not daily.empty else 0.0,
        "median_daily_pnl": float(daily_pnl.median()) if not daily.empty else 0.0,
        "worst_day": float(daily_pnl.min()) if not daily.empty else 0.0,
        "pct_days_gt_plus_1r": float((daily_pnl_r >= 1.0).mean()) if not daily.empty else 0.0,
        "pct_days_lt_minus_1r": float((daily_pnl_r <= -1.0).mean()) if not daily.empty else 0.0,
        "longest_losing_streak_daily": int(max(day_loss_lengths, default=0)),
        "longest_losing_streak_trade": int(base.get("longest_loss_streak", 0)),
        "cut_day_freq": float(pd.Series(daily["day_cut_by_rule"]).fillna(False).astype(bool).mean()) if not daily.empty else 0.0,
        "hard_loss_cap_freq": float(pd.Series(daily["hard_loss_cap_triggered"]).fillna(False).astype(bool).mean()) if not daily.empty else 0.0,
        "profit_lock_freq": float(pd.Series(daily["hard_profit_lock_triggered"]).fillna(False).astype(bool).mean()) if not daily.empty else 0.0,
        "giveback_freq": float(pd.Series(daily["giveback_triggered"]).fillna(False).astype(bool).mean()) if not daily.empty else 0.0,
        "daily_loss_limit_breach_freq": _safe_div(
            float(base.get("number_of_daily_loss_limit_breaches", 0)),
            max(len(daily), 1),
            default=0.0,
        ),
        "expected_profit_per_trading_day": _safe_div(
            float(base.get("cumulative_pnl", 0.0)),
            max(int((daily_trade_count > 0).sum()), 1),
            default=0.0,
        ),
    }


def _build_summary_by_scope(
    trades: pd.DataFrame,
    daily_results: pd.DataFrame,
    all_sessions: list,
    is_sessions: list,
    oos_sessions: list,
    initial_capital: float,
    constraints: PropConstraintConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, sessions in (("overall", all_sessions), ("is", is_sessions), ("oos", oos_sessions)):
        rows.append(
            {
                "scope": scope,
                **_scope_summary(
                    trades=_subset_frame_by_sessions(trades, sessions),
                    daily_results=_subset_frame_by_sessions(daily_results, sessions),
                    sessions=sessions,
                    initial_capital=initial_capital,
                    constraints=constraints,
                ),
            }
        )
    return pd.DataFrame(rows)


def _scope_value(summary_by_scope: pd.DataFrame, scope: str, column: str) -> Any:
    row = summary_by_scope.loc[summary_by_scope["scope"] == scope]
    if row.empty:
        return np.nan
    return row.iloc[0].get(column, np.nan)


def _delta_rows(variant: VariantRun, baseline_variant: VariantRun, scope: str) -> dict[str, Any]:
    variant_net = float(_scope_value(variant.summary_by_scope, scope, "net_pnl"))
    baseline_net = float(_scope_value(baseline_variant.summary_by_scope, scope, "net_pnl"))
    variant_sharpe = float(_scope_value(variant.summary_by_scope, scope, "sharpe"))
    baseline_sharpe = float(_scope_value(baseline_variant.summary_by_scope, scope, "sharpe"))
    variant_pf = float(_scope_value(variant.summary_by_scope, scope, "profit_factor"))
    baseline_pf = float(_scope_value(baseline_variant.summary_by_scope, scope, "profit_factor"))
    variant_exp = float(_scope_value(variant.summary_by_scope, scope, "expectancy"))
    baseline_exp = float(_scope_value(baseline_variant.summary_by_scope, scope, "expectancy"))
    variant_dd = float(_scope_value(variant.summary_by_scope, scope, "max_drawdown"))
    baseline_dd = float(_scope_value(baseline_variant.summary_by_scope, scope, "max_drawdown"))
    variant_worst = float(_scope_value(variant.summary_by_scope, scope, "worst_day"))
    baseline_worst = float(_scope_value(baseline_variant.summary_by_scope, scope, "worst_day"))
    variant_breach = float(_scope_value(variant.summary_by_scope, scope, "daily_loss_limit_breach_freq"))
    baseline_breach = float(_scope_value(baseline_variant.summary_by_scope, scope, "daily_loss_limit_breach_freq"))
    variant_cut = float(_scope_value(variant.summary_by_scope, scope, "cut_day_freq"))
    baseline_cut = float(_scope_value(baseline_variant.summary_by_scope, scope, "cut_day_freq"))

    dd_improvement = _safe_div(abs(baseline_dd) - abs(variant_dd), max(abs(baseline_dd), 1.0), default=0.0)
    worst_day_improvement = _safe_div(abs(baseline_worst) - abs(variant_worst), max(abs(baseline_worst), 1.0), default=0.0)
    return {
        f"{scope}_net_pnl_retention_vs_baseline": _safe_div(variant_net, baseline_net, default=0.0)
        if abs(baseline_net) > 1e-9
        else 0.0,
        f"{scope}_sharpe_delta_vs_baseline": float(variant_sharpe - baseline_sharpe),
        f"{scope}_profit_factor_delta_vs_baseline": float(variant_pf - baseline_pf),
        f"{scope}_expectancy_delta_vs_baseline": float(variant_exp - baseline_exp),
        f"{scope}_max_drawdown_improvement_vs_baseline": float(dd_improvement),
        f"{scope}_worst_day_improvement_vs_baseline": float(worst_day_improvement),
        f"{scope}_daily_loss_limit_breach_reduction_vs_baseline": float(baseline_breach - variant_breach),
        f"{scope}_cut_day_freq_delta_vs_baseline": float(variant_cut - baseline_cut),
    }


def _score_scope(prefix: str, row: dict[str, Any]) -> float:
    pnl_retention = float(row.get(f"{prefix}_net_pnl_retention_vs_baseline", 0.0))
    sharpe_delta = float(row.get(f"{prefix}_sharpe_delta_vs_baseline", 0.0))
    pf_delta = float(row.get(f"{prefix}_profit_factor_delta_vs_baseline", 0.0))
    exp_delta = float(row.get(f"{prefix}_expectancy_delta_vs_baseline", 0.0))
    dd_improvement = float(row.get(f"{prefix}_max_drawdown_improvement_vs_baseline", 0.0))
    worst_day_improvement = float(row.get(f"{prefix}_worst_day_improvement_vs_baseline", 0.0))
    breach_reduction = float(row.get(f"{prefix}_daily_loss_limit_breach_reduction_vs_baseline", 0.0))
    cut_delta = float(row.get(f"{prefix}_cut_day_freq_delta_vs_baseline", 0.0))

    retention_component = 0.0
    if pnl_retention > 0:
        retention_component = max(min((pnl_retention - 0.80) / 0.20, 1.5), -2.0)

    expectancy_component = math.tanh(exp_delta / 20.0)
    return float(
        2.0 * dd_improvement
        + 1.6 * worst_day_improvement
        + 1.2 * breach_reduction
        + 0.8 * sharpe_delta
        + 0.6 * pf_delta
        + 0.5 * retention_component
        + 0.4 * expectancy_component
        - 0.2 * max(cut_delta, 0.0)
    )


def _frames_equivalent(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if left.empty and right.empty:
        return True
    key_cols = [column for column in ["session_date", "entry_time", "exit_time", "quantity", "exit_reason", "net_pnl_usd"] if column in left.columns and column in right.columns]
    if not key_cols:
        return False
    left_view = left[key_cols].copy().reset_index(drop=True)
    right_view = right[key_cols].copy().reset_index(drop=True)
    if len(left_view) != len(right_view):
        return False
    return left_view.equals(right_view)


def _variant_verdict(row: dict[str, Any]) -> str:
    if bool(row.get("structurally_identical_to_baseline")):
        return "inactive_under_current_baseline"

    oos_retention = float(row.get("oos_net_pnl_retention_vs_baseline", 0.0))
    oos_pf_delta = float(row.get("oos_profit_factor_delta_vs_baseline", 0.0))
    oos_exp_delta = float(row.get("oos_expectancy_delta_vs_baseline", 0.0))
    oos_dd_improvement = float(row.get("oos_max_drawdown_improvement_vs_baseline", 0.0))
    oos_worst_improvement = float(row.get("oos_worst_day_improvement_vs_baseline", 0.0))
    oos_breach_reduction = float(row.get("oos_daily_loss_limit_breach_reduction_vs_baseline", 0.0))
    is_score = float(row.get("screening_score", 0.0))
    oos_score = float(row.get("validation_score", 0.0))

    if (
        oos_retention >= 0.95
        and oos_pf_delta >= 0.0
        and oos_exp_delta >= 0.0
        and oos_dd_improvement >= 0.0
        and oos_score > 0.30
    ):
        return "potential_alpha_and_defensive"

    if (
        oos_retention >= 0.85
        and oos_dd_improvement >= 0.10
        and oos_worst_improvement >= 0.05
        and oos_breach_reduction >= 0.0
        and oos_score > 0.20
    ):
        return "robust_defensive_candidate"

    if (
        oos_retention >= 0.70
        and (oos_dd_improvement >= 0.15 or oos_breach_reduction > 0.0 or oos_worst_improvement >= 0.10)
        and oos_score > 0.0
    ):
        return "defensive_but_costly"

    if is_score > 0.0 and oos_score <= 0.0:
        return "is_only"

    return "negative"


def _baseline_note(spec: MnqIntradayPnlCampaignSpec) -> str:
    return (
        "Official nominal baseline rerun with fixed nominal contracts. "
        "Entry logic, ATR ensemble selection, time exit, stop/target and costs are otherwise preserved."
    )


def _build_execution_model(
    symbol: str,
    commission_per_side_usd: float | None,
    slippage_ticks: float | None,
) -> tuple[ExecutionModel, dict[str, Any]]:
    instrument = get_instrument_spec(symbol)
    return (
        ExecutionModel(
            commission_per_side_usd=float(
                instrument["commission_per_side_usd"] if commission_per_side_usd is None else commission_per_side_usd
            ),
            slippage_ticks=float(instrument["slippage_ticks"] if slippage_ticks is None else slippage_ticks),
            tick_size=float(instrument["tick_size"]),
        ),
        instrument,
    )


def _build_variant_run(
    signal_df: pd.DataFrame,
    overlay: IntradayPnlOverlaySpec,
    *,
    baseline: BaselineSpec,
    all_sessions: list,
    is_sessions: list,
    oos_sessions: list,
    fixed_contracts: int,
    initial_capital: float,
    constraints: PropConstraintConfig,
    execution_model: ExecutionModel,
    tick_value_usd: float,
    note: str = "",
) -> VariantRun:
    result: OverlayBacktestResult = run_intraday_pnl_overlay_backtest(
        signal_df,
        execution_model=execution_model,
        baseline=baseline,
        overlay=overlay,
        fixed_contracts=fixed_contracts,
        tick_value_usd=tick_value_usd,
    )
    summary_by_scope = _build_summary_by_scope(
        trades=result.trades,
        daily_results=result.daily_results,
        all_sessions=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        initial_capital=initial_capital,
        constraints=constraints,
    )
    return VariantRun(
        name=overlay.name,
        family=overlay.family,
        description=overlay.description,
        calibration_scope="is_only_grid_screening",
        parameters={key: value for key, value in asdict(overlay).items() if key not in {"name", "family", "description"}},
        trades=result.trades,
        daily_results=result.daily_results,
        controls=result.controls,
        state_transitions=result.state_transitions,
        summary_by_scope=summary_by_scope,
        note=note,
    )


def _default_overlay_variants(spec: MnqIntradayPnlCampaignSpec) -> list[IntradayPnlOverlaySpec]:
    max_profit_lock = min(float(spec.baseline.target_multiple), 1.50)
    profit_candidates = [value for value in (0.75, 1.00, 1.25, 1.50) if value <= max_profit_lock + 1e-12]
    variants: list[IntradayPnlOverlaySpec] = []

    for threshold in (0.50, 0.75):
        variants.append(
            IntradayPnlOverlaySpec(
                name=f"hard_loss_cap_r_{str(threshold).replace('.', 'p')}",
                family="block_1_hard_cap",
                description=f"Exit and halt the day once mark-to-market day PnL reaches -{threshold:.2f}R.",
                hard_loss_cap=-threshold,
            )
        )

    for threshold in profit_candidates:
        variants.append(
            IntradayPnlOverlaySpec(
                name=f"hard_profit_lock_r_{str(threshold).replace('.', 'p')}",
                family="block_1_hard_cap",
                description=f"Lock the day once mark-to-market day PnL reaches +{threshold:.2f}R.",
                hard_profit_lock=threshold,
            )
        )

    for loss_cap, profit_lock in ((0.50, 1.00), (0.50, 1.50), (0.75, 1.00), (0.75, 1.50)):
        if profit_lock > max_profit_lock + 1e-12:
            continue
        variants.append(
            IntradayPnlOverlaySpec(
                name=f"combo_loss_{str(loss_cap).replace('.', 'p')}__profit_{str(profit_lock).replace('.', 'p')}",
                family="block_1_hard_cap",
                description=f"Combine hard loss cap -{loss_cap:.2f}R and profit lock +{profit_lock:.2f}R.",
                hard_loss_cap=-loss_cap,
                hard_profit_lock=profit_lock,
            )
        )

    for activation, giveback in ((1.00, 0.50), (1.25, 0.50), (1.50, 0.75)):
        if activation > max(float(spec.baseline.target_multiple), 1.0) + 1e-12:
            continue
        variants.append(
            IntradayPnlOverlaySpec(
                name=f"giveback_after_{str(activation).replace('.', 'p')}r__{str(giveback).replace('.', 'p')}r",
                family="block_2_giveback",
                description=(
                    f"Once intraday day PnL reaches +{activation:.2f}R, exit on a giveback of {giveback:.2f}R."
                ),
                giveback_activation=activation,
                giveback_threshold=giveback,
            )
        )

    variants.extend(
        [
            IntradayPnlOverlaySpec(
                name="max_trades_1",
                family="block_3_sequence_logic",
                description="Cap the day at one trade.",
                max_trades_per_day=1,
            ),
            IntradayPnlOverlaySpec(
                name="stop_after_2_consecutive_losses",
                family="block_3_sequence_logic",
                description="Stop the day after two consecutive losing trades.",
                halt_after_consecutive_losses=2,
            ),
            IntradayPnlOverlaySpec(
                name="continue_only_if_first_trade_wins",
                family="block_3_sequence_logic",
                description="After a losing first trade, halt the rest of the day.",
                continue_only_if_first_trade_wins=True,
            ),
        ]
    )

    for name, loss_cap, profit_lock, giveback_activation, giveback, defensive_losses, halt_losses in (
        ("state_machine_a", 0.50, 1.00, 1.00, 0.50, 1, 2),
        ("state_machine_b", 0.75, 1.25, 1.25, 0.50, 1, 2),
        ("state_machine_c", 0.75, 1.50, 1.50, 0.75, 1, 2),
    ):
        if profit_lock > max_profit_lock + 1e-12:
            continue
        variants.append(
            IntradayPnlOverlaySpec(
                name=name,
                family="block_4_state_machine",
                description=(
                    "Simple daily state machine: neutral -> defensive after a loss, "
                    "locked_profit on profit lock / giveback, halted on hard daily stop."
                ),
                hard_loss_cap=-loss_cap,
                hard_profit_lock=profit_lock,
                giveback_activation=giveback_activation,
                giveback_threshold=giveback,
                defensive_after_total_losses=defensive_losses,
                halt_after_total_losses=halt_losses,
            )
        )
    return variants


def _variant_row(
    variant: VariantRun,
    baseline_variant: VariantRun,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant_name": variant.name,
        "family": variant.family,
        "description": variant.description,
        "calibration_scope": variant.calibration_scope,
        "parameters_json": json.dumps({key: _serialize_value(value) for key, value in variant.parameters.items()}, sort_keys=True),
        "note": variant.note,
        "structurally_identical_to_baseline": _frames_equivalent(variant.trades, baseline_variant.trades),
    }
    for scope in ("overall", "is", "oos"):
        for metric in [
            "net_pnl",
            "sharpe",
            "sortino",
            "profit_factor",
            "expectancy",
            "max_drawdown",
            "n_trades",
            "hit_rate",
            "avg_win",
            "avg_loss",
            "n_days_traded",
            "pct_days_traded",
            "avg_trades_per_day",
            "mean_daily_pnl",
            "median_daily_pnl",
            "worst_day",
            "pct_days_gt_plus_1r",
            "pct_days_lt_minus_1r",
            "longest_losing_streak_daily",
            "longest_losing_streak_trade",
            "cut_day_freq",
            "hard_loss_cap_freq",
            "profit_lock_freq",
            "giveback_freq",
            "daily_loss_limit_breach_freq",
            "expected_profit_per_trading_day",
        ]:
            row[f"{scope}_{metric}"] = _scope_value(variant.summary_by_scope, scope, metric)
    row.update(_delta_rows(variant, baseline_variant, "is"))
    row.update(_delta_rows(variant, baseline_variant, "oos"))
    row["screening_score"] = _score_scope("is", row)
    row["validation_score"] = _score_scope("oos", row)
    row["verdict"] = _variant_verdict(row)
    return row


def _export_variant_artifacts(root: Path, variant: VariantRun) -> None:
    variant_dir = root / "variants" / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    variant.controls.to_csv(variant_dir / "controls.csv", index=False)
    variant.trades.to_csv(variant_dir / "trades.csv", index=False)
    variant.daily_results.to_csv(variant_dir / "daily_results.csv", index=False)
    variant.state_transitions.to_csv(variant_dir / "state_transitions.csv", index=False)
    variant.summary_by_scope.to_csv(variant_dir / "metrics_by_scope.csv", index=False)


def _state_transition_summary(
    variants: list[VariantRun],
    is_sessions: list,
    oos_sessions: list,
) -> pd.DataFrame:
    parts = []
    for variant in variants:
        if variant.state_transitions.empty:
            continue
        view = variant.state_transitions.copy()
        view["scope"] = _phase_column(view["session_date"], is_sessions=is_sessions, oos_sessions=oos_sessions)
        parts.append(view)
    if not parts:
        return pd.DataFrame(columns=["variant_name", "scope", "from_state", "to_state", "trigger", "transition_count"])
    joined = pd.concat(parts, ignore_index=True)
    return (
        joined.groupby(["variant_name", "scope", "from_state", "to_state", "trigger"], as_index=False)
        .size()
        .rename(columns={"size": "transition_count"})
        .sort_values(["variant_name", "scope", "transition_count"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def _daily_path_summary(
    variants: list[VariantRun],
    is_sessions: list,
    oos_sessions: list,
) -> pd.DataFrame:
    parts = []
    for variant in variants:
        view = variant.daily_results.copy()
        if view.empty:
            continue
        view["scope"] = _phase_column(view["session_date"], is_sessions=is_sessions, oos_sessions=oos_sessions)
        parts.append(view)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _synthesise_verdict(
    results_df: pd.DataFrame,
    spec: MnqIntradayPnlCampaignSpec,
) -> dict[str, Any]:
    non_baseline = results_df.loc[results_df["family"].ne("baseline")].copy()
    robust = non_baseline.loc[non_baseline["verdict"].isin(["potential_alpha_and_defensive", "robust_defensive_candidate"])].copy()
    defensive = non_baseline.loc[non_baseline["verdict"] == "defensive_but_costly"].copy()

    if not robust.empty:
        best = robust.sort_values(["validation_score", "oos_net_pnl_retention_vs_baseline"], ascending=[False, False]).iloc[0]
    elif not defensive.empty:
        best = defensive.sort_values(["validation_score", "oos_max_drawdown_improvement_vs_baseline"], ascending=[False, False]).iloc[0]
    elif not non_baseline.empty:
        best = non_baseline.sort_values(["validation_score", "screening_score"], ascending=[False, False]).iloc[0]
    else:
        best = pd.Series(dtype=object)

    best_dict = best.to_dict() if not best.empty else {}
    best_verdict = str(best_dict.get("verdict", "none"))
    best_pf_delta = float(best_dict.get("oos_profit_factor_delta_vs_baseline", 0.0) or 0.0)
    best_exp_delta = float(best_dict.get("oos_expectancy_delta_vs_baseline", 0.0) or 0.0)
    best_retention = float(best_dict.get("oos_net_pnl_retention_vs_baseline", 0.0) or 0.0)
    best_dd = float(best_dict.get("oos_max_drawdown_improvement_vs_baseline", 0.0) or 0.0)
    best_worst = float(best_dict.get("oos_worst_day_improvement_vs_baseline", 0.0) or 0.0)

    overlay_adds_robust_value = best_verdict in {"potential_alpha_and_defensive", "robust_defensive_candidate"}
    if overlay_adds_robust_value and best_pf_delta > 0 and best_exp_delta > 0 and best_retention >= 0.95:
        value_character = "potential_alpha_plus_defensive"
    elif best_dd > 0 or best_worst > 0:
        value_character = "prop_defensive"
    else:
        value_character = "none"

    phase2 = bool(
        overlay_adds_robust_value
        and best_retention >= 0.85
        and best_dd >= 0.10
        and not bool(best_dict.get("structurally_identical_to_baseline"))
    )

    return {
        "run_type": "mnq_orb_intraday_pnl_overlay_campaign",
        "baseline_is_one_trade_per_day": bool(spec.baseline.one_trade_per_day),
        "baseline_fixed_contracts": int(spec.fixed_contracts),
        "overlay_bar_close_mark_to_market": True,
        "best_variant_name": best_dict.get("variant_name"),
        "best_variant_family": best_dict.get("family"),
        "best_variant_verdict": best_verdict,
        "best_variant_oos_net_pnl_retention_vs_baseline": best_dict.get("oos_net_pnl_retention_vs_baseline"),
        "best_variant_oos_profit_factor_delta_vs_baseline": best_dict.get("oos_profit_factor_delta_vs_baseline"),
        "best_variant_oos_expectancy_delta_vs_baseline": best_dict.get("oos_expectancy_delta_vs_baseline"),
        "best_variant_oos_max_drawdown_improvement_vs_baseline": best_dict.get("oos_max_drawdown_improvement_vs_baseline"),
        "best_variant_oos_worst_day_improvement_vs_baseline": best_dict.get("oos_worst_day_improvement_vs_baseline"),
        "best_variant_oos_cut_day_freq": best_dict.get("oos_cut_day_freq"),
        "best_variant_oos_profit_lock_freq": best_dict.get("oos_profit_lock_freq"),
        "best_variant_oos_giveback_freq": best_dict.get("oos_giveback_freq"),
        "overlay_adds_robust_value": overlay_adds_robust_value,
        "overlay_value_character": value_character,
        "phase2_with_3state_recommended": phase2,
        "structural_note": (
            "The official nominal baseline trades at most once per day, so trade-count and post-first-trade sequence rules are mostly structurally inactive."
            if spec.baseline.one_trade_per_day
            else "The baseline can reuse the full trade-count / sequence logic."
        ),
        "assumptions": [
            f"baseline direction={spec.baseline.direction}",
            f"baseline OR window={spec.baseline.or_minutes}m",
            f"baseline target_multiple={spec.baseline.target_multiple}",
            f"aggregation_rule={spec.aggregation_rule}",
            f"fixed_contracts={spec.fixed_contracts}",
            f"commission_per_side_usd={spec.commission_per_side_usd}",
            f"slippage_ticks={spec.slippage_ticks}",
        ],
    }


def _write_report(
    output_path: Path,
    spec: MnqIntradayPnlCampaignSpec,
    analysis: SymbolAnalysis,
    results_df: pd.DataFrame,
    verdict: dict[str, Any],
) -> None:
    baseline = results_df.loc[results_df["variant_name"] == "baseline_nominal_fixed_intraday"].iloc[0]
    top = results_df.loc[results_df["variant_name"] == verdict.get("best_variant_name")].copy()

    top_line = "- Aucun overlay PnL intraday ne sort du lot de facon utile."
    driver_line = "- Pas de driver OOS defendable detecte."
    candidate_line = "- Aucun candidat simple a pousser vers la phase 2."
    if not top.empty:
        row = top.iloc[0]
        top_line = (
            f"- Meilleur overlay: `{row['variant_name']}` | verdict `{row['verdict']}` | "
            f"OOS pnl retention `{100.0 * float(row['oos_net_pnl_retention_vs_baseline']):.1f}%` | "
            f"PF delta `{float(row['oos_profit_factor_delta_vs_baseline']):+.3f}` | "
            f"expectancy delta `{float(row['oos_expectancy_delta_vs_baseline']):+.2f}` | "
            f"maxDD improvement `{100.0 * float(row['oos_max_drawdown_improvement_vs_baseline']):.1f}%` | "
            f"worst-day improvement `{100.0 * float(row['oos_worst_day_improvement_vs_baseline']):.1f}%`."
        )
        driver_line = (
            f"- Driver principal observe: cut-day freq `{100.0 * float(row['oos_cut_day_freq']):.1f}%`, "
            f"profit-lock freq `{100.0 * float(row['oos_profit_lock_freq']):.1f}%`, "
            f"giveback freq `{100.0 * float(row['oos_giveback_freq']):.1f}%`, "
            f"daily-loss-breach freq `{100.0 * float(row['oos_daily_loss_limit_breach_freq']):.1f}%`."
        )
        candidate_line = (
            f"- Candidat phase 2 le plus simple: `{row['variant_name']}` "
            f"({verdict.get('overlay_value_character', 'none')})."
        )

    lines = [
        "# MNQ ORB Intraday PnL Overlay Report",
        "",
        "## Baseline",
        "",
        f"- Baseline officielle conservee: OR{int(spec.baseline.or_minutes)} / direction `{spec.baseline.direction}` / RR `{float(spec.baseline.target_multiple):.2f}` / one_trade_per_day `{bool(spec.baseline.one_trade_per_day)}`.",
        f"- Filtre ATR / selection ensemble conservee via `{spec.aggregation_rule}` et calibree strictement sur IS.",
        f"- Sizing de phase 1 force en `fixed_contracts={int(spec.fixed_contracts)}` pour isoler l'effet overlay PnL intraday.",
        f"- Hypotheses d'execution appliquees dans cette campagne: commission side `{float(spec.commission_per_side_usd):.2f}` USD, slippage `{float(spec.slippage_ticks):.2f}` tick(s).",
        f"- Regle overlay importante: evaluation **bar-close mark-to-market** pour les hard caps / locks / giveback, alors que le stop/target baseline reste intrabar. Cette convention est leak-free mais doit etre lue comme un overlay de gestion, pas comme un changement du signal.",
        f"- Dataset: `{analysis.dataset_path.name}` | sessions IS/OOS `{len(analysis.is_sessions)}` / `{len(analysis.oos_sessions)}` | sessions selectionnees par l'ensemble `{len(_selected_ensemble_sessions(analysis, spec.aggregation_rule))}`.",
        "",
        "## Baseline Readout",
        "",
        f"- Baseline OOS: net pnl `{float(baseline['oos_net_pnl']):.2f}` | Sharpe `{float(baseline['oos_sharpe']):.3f}` | Sortino `{float(baseline['oos_sortino']):.3f}` | PF `{float(baseline['oos_profit_factor']):.3f}` | expectancy `{float(baseline['oos_expectancy']):.2f}` | maxDD `{float(baseline['oos_max_drawdown']):.2f}`.",
        f"- Baseline OOS days/trades: `{int(baseline['oos_n_days_traded'])}` jours trades | `{int(baseline['oos_n_trades'])}` trades | avg trades/day `{float(baseline['oos_avg_trades_per_day']):.3f}`.",
        f"- Baseline OOS prop daily-loss breach freq `{100.0 * float(baseline['oos_daily_loss_limit_breach_freq']):.1f}%` | cut-day freq `{100.0 * float(baseline['oos_cut_day_freq']):.1f}%`.",
        "",
        "## Main Findings",
        "",
        top_line,
        driver_line,
        candidate_line,
        f"- Variants robustes: `{int(results_df['verdict'].isin(['potential_alpha_and_defensive', 'robust_defensive_candidate']).sum())}` | defensifs mais couteux: `{int((results_df['verdict'] == 'defensive_but_costly').sum())}` | inactifs structurellement: `{int((results_df['verdict'] == 'inactive_under_current_baseline').sum())}`.",
        "",
        "## Reponses Directes",
        "",
        (
            f"- Est-ce qu'un overlay PnL intraday ameliore reellement la baseline ? "
            f"{'Oui, avec un signal OOS defendable.' if verdict.get('overlay_adds_robust_value') else 'Pas de facon assez robuste pour revendiquer un vrai upgrade de baseline.'}"
        ),
        (
            f"- Les gains viennent-ils surtout du drawdown / giveback / baisse du nombre de trades ou d'une vraie amelioration du PF ? "
            f"{'Lecture principale: surtout defensif / prop-compatible.' if verdict.get('overlay_value_character') == 'prop_defensive' else 'Lecture principale: potentiel mix alpha + defense.' if verdict.get('overlay_value_character') == 'potential_alpha_plus_defensive' else 'Pas de moteur stable identifie.'}"
        ),
        f"- Quelles regles sont les plus robustes OOS ? {top_line[2:]}",
        (
            "- Y a-t-il un bon candidat simple et interpretable pour une phase 2 avec sizing 3-state ? "
            + ("Oui." if verdict.get("phase2_with_3state_recommended") else "Pas encore de facon assez nette.")
        ),
        (
            "- L'interet est-il surtout prop defensif ou y a-t-il aussi un gain alpha net ? "
            + ("Plutot prop defensif." if verdict.get("overlay_value_character") == "prop_defensive" else "Il y a peut-etre une composante alpha additionnelle, mais elle reste a confirmer." if verdict.get("overlay_value_character") == "potential_alpha_plus_defensive" else "Aucun gain alpha net robustement etabli.")
        ),
        "",
        "## Structural Notes",
        "",
        (
            "- La baseline officielle est `one_trade_per_day=True`. "
            "Cela rend les blocs `trade cap`, `stop apres 2 pertes`, `continuer seulement si le premier trade gagne` en grande partie structurellement inactifs dans ce setup precis."
        ),
        "- Les overlays vraiment informatifs dans ce run sont donc surtout ceux qui coupent plus tot le trade en cours: hard loss cap, hard profit lock, giveback, state machine simple.",
        "- Si vous souhaitez tester la pleine richesse des blocs sequence / trade-count sur une baseline multi-trades, il faudra une campagne separee ou une baseline differente.",
        "",
        "## Exports",
        "",
        "- `screening_summary.csv`",
        "- `validation_summary.csv`",
        "- `full_variant_results.csv`",
        "- `daily_path_summary.csv`",
        "- `intraday_state_transition_summary.csv`",
        "- `final_report.md`",
        "- `final_verdict.json`",
        "- `variants/<variant>/trades.csv`",
        "- `variants/<variant>/daily_results.csv`",
        "- `variants/<variant>/controls.csv`",
        "- `variants/<variant>/state_transitions.csv`",
    ]
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_mnq_orb_intraday_pnl_campaign(spec: MnqIntradayPnlCampaignSpec) -> dict[str, Any]:
    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(spec.output_root) if spec.output_root is not None else EXPORTS_DIR / f"mnq_orb_intraday_pnl_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_path = spec.dataset_path or resolve_processed_dataset(spec.symbol, timeframe="1m")
    analysis = analyze_symbol(
        symbol=spec.symbol,
        baseline=spec.baseline,
        grid=spec.grid,
        is_fraction=spec.is_fraction,
        dataset_path=dataset_path,
    )
    selected_sessions = _selected_ensemble_sessions(analysis, spec.aggregation_rule)

    tradable_signal_df = analysis.signal_df.copy()
    tradable_signal_df["session_date"] = pd.to_datetime(tradable_signal_df["session_date"]).dt.date
    tradable_signal_df["selected_by_ensemble"] = tradable_signal_df["session_date"].isin(selected_sessions)
    tradable_signal_df.loc[~tradable_signal_df["selected_by_ensemble"], "signal"] = 0

    execution_model, instrument = _build_execution_model(
        spec.symbol,
        commission_per_side_usd=spec.commission_per_side_usd,
        slippage_ticks=spec.slippage_ticks,
    )

    baseline_overlay = IntradayPnlOverlaySpec(
        name="baseline_nominal_fixed_intraday",
        family="baseline",
        description="Official nominal baseline rerun with fixed nominal size and no intraday overlay.",
    )
    baseline_variant = _build_variant_run(
        tradable_signal_df,
        baseline_overlay,
        baseline=spec.baseline,
        all_sessions=analysis.all_sessions,
        is_sessions=analysis.is_sessions,
        oos_sessions=analysis.oos_sessions,
        fixed_contracts=spec.fixed_contracts,
        initial_capital=spec.initial_capital_usd,
        constraints=spec.prop_constraints,
        execution_model=execution_model,
        tick_value_usd=float(instrument["tick_value_usd"]),
        note=_baseline_note(spec),
    )

    variants: list[VariantRun] = [baseline_variant]
    for overlay in _default_overlay_variants(spec):
        note = ""
        if spec.baseline.one_trade_per_day and overlay.family == "block_3_sequence_logic":
            note = "Mostly structurally inactive because the official baseline already trades at most once per day."
        variants.append(
            _build_variant_run(
                tradable_signal_df,
                overlay,
                baseline=spec.baseline,
                all_sessions=analysis.all_sessions,
                is_sessions=analysis.is_sessions,
                oos_sessions=analysis.oos_sessions,
                fixed_contracts=spec.fixed_contracts,
                initial_capital=spec.initial_capital_usd,
                constraints=spec.prop_constraints,
                execution_model=execution_model,
                tick_value_usd=float(instrument["tick_value_usd"]),
                note=note,
            )
        )

    results_df = pd.DataFrame([_variant_row(variant, baseline_variant) for variant in variants])
    results_df = results_df[[column for column in SUMMARY_COLUMNS if column in results_df.columns]]

    full_results_path = output_root / "full_variant_results.csv"
    results_df.to_csv(full_results_path, index=False)

    screening_summary = results_df.sort_values(["screening_score", "is_max_drawdown_improvement_vs_baseline"], ascending=[False, False]).reset_index(drop=True)
    screening_path = output_root / "screening_summary.csv"
    screening_summary.to_csv(screening_path, index=False)

    validation_summary = results_df.sort_values(["validation_score", "oos_max_drawdown_improvement_vs_baseline"], ascending=[False, False]).reset_index(drop=True)
    validation_path = output_root / "validation_summary.csv"
    validation_summary.to_csv(validation_path, index=False)

    daily_summary = _daily_path_summary(variants, is_sessions=analysis.is_sessions, oos_sessions=analysis.oos_sessions)
    daily_summary_path = output_root / "daily_path_summary.csv"
    daily_summary.to_csv(daily_summary_path, index=False)

    transition_summary = _state_transition_summary(variants, is_sessions=analysis.is_sessions, oos_sessions=analysis.oos_sessions)
    transition_path = output_root / "intraday_state_transition_summary.csv"
    transition_summary.to_csv(transition_path, index=False)

    for variant in variants:
        _export_variant_artifacts(output_root, variant)

    verdict = _synthesise_verdict(validation_summary, spec)
    verdict_path = output_root / "final_verdict.json"
    _json_dump(verdict_path, verdict)

    report_path = output_root / "final_report.md"
    _write_report(report_path, spec, analysis, validation_summary, verdict)

    metadata_path = output_root / "run_metadata.json"
    _json_dump(
        metadata_path,
        {
            "run_timestamp": datetime.now().isoformat(),
            "dataset_path": dataset_path,
            "selected_symbol": spec.symbol,
            "selected_aggregation_rule": spec.aggregation_rule,
            "selected_session_count": int(len(selected_sessions)),
            "analysis_best_ensemble": analysis.best_ensemble,
            "analysis_baseline_transfer": analysis.baseline_transfer,
            "execution_model": {
                "commission_per_side_usd": execution_model.commission_per_side_usd,
                "slippage_ticks": execution_model.slippage_ticks,
                "tick_size": execution_model.tick_size,
            },
            "spec": asdict(spec),
        },
    )

    return {
        "output_root": output_root,
        "screening_summary": screening_path,
        "validation_summary": validation_path,
        "full_variant_results": full_results_path,
        "daily_path_summary": daily_summary_path,
        "intraday_state_transition_summary": transition_path,
        "final_report": report_path,
        "final_verdict": verdict_path,
        "run_metadata": metadata_path,
    }


def build_default_spec(output_root: Path | None = None) -> MnqIntradayPnlCampaignSpec:
    return MnqIntradayPnlCampaignSpec(output_root=output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MNQ ORB intraday PnL overlay campaign.")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--is-fraction", type=float, default=0.70)
    parser.add_argument("--fixed-contracts", type=int, default=1)
    parser.add_argument("--commission-per-side-usd", type=float, default=0.62)
    parser.add_argument("--slippage-ticks", type=float, default=1.0)
    args = parser.parse_args()

    spec = MnqIntradayPnlCampaignSpec(
        dataset_path=Path(args.dataset_path) if args.dataset_path is not None else None,
        output_root=Path(args.output_root) if args.output_root is not None else None,
        is_fraction=float(args.is_fraction),
        fixed_contracts=int(args.fixed_contracts),
        commission_per_side_usd=float(args.commission_per_side_usd) if args.commission_per_side_usd is not None else None,
        slippage_ticks=float(args.slippage_ticks) if args.slippage_ticks is not None else None,
    )
    artifacts = run_mnq_orb_intraday_pnl_campaign(spec)
    print(f"validation_summary: {artifacts['validation_summary']}")
    print(f"final_report: {artifacts['final_report']}")


if __name__ == "__main__":
    main()
