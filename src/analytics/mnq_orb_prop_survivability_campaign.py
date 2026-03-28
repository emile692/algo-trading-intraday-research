"""Prop-survivability campaign for the validated MNQ Ensemble ORB baseline."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

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
from src.engine.backtester import run_backtest
from src.engine.execution_model import ExecutionModel
from src.engine.trade_log import empty_trade_log


DEFAULT_VARIANT_COLUMNS = [
    "variant_name",
    "family",
    "description",
    "calibration_scope",
    "parameters_json",
    "note",
    "verdict",
    "selected_session_count",
    "policy_skip_days",
    "overall_net_pnl",
    "overall_sharpe",
    "overall_sortino",
    "overall_profit_factor",
    "overall_expectancy",
    "overall_max_drawdown",
    "overall_n_trades",
    "overall_n_days_traded",
    "overall_pct_days_traded",
    "overall_worst_day",
    "overall_longest_losing_streak_daily",
    "overall_median_recovery_days",
    "overall_max_recovery_days",
    "overall_daily_loss_limit_breach_freq",
    "overall_days_to_profit_target",
    "overall_profit_target_reached_before_max_loss",
    "overall_max_loss_limit_buffer_usd",
    "is_net_pnl",
    "is_sharpe",
    "is_sortino",
    "is_profit_factor",
    "is_expectancy",
    "is_max_drawdown",
    "is_n_trades",
    "is_n_days_traded",
    "is_pct_days_traded",
    "is_worst_day",
    "is_longest_losing_streak_daily",
    "is_median_recovery_days",
    "is_max_recovery_days",
    "oos_net_pnl",
    "oos_sharpe",
    "oos_sortino",
    "oos_profit_factor",
    "oos_expectancy",
    "oos_max_drawdown",
    "oos_n_trades",
    "oos_n_days_traded",
    "oos_pct_days_traded",
    "oos_worst_day",
    "oos_longest_losing_streak_daily",
    "oos_median_recovery_days",
    "oos_max_recovery_days",
    "oos_daily_loss_limit_breach_freq",
    "oos_days_to_profit_target",
    "oos_profit_target_reached_before_max_loss",
    "oos_max_loss_limit_buffer_usd",
    "oos_net_pnl_retention_vs_nominal",
    "oos_trade_retention_vs_nominal",
    "oos_day_retention_vs_nominal",
    "oos_max_drawdown_improvement_vs_nominal",
    "year_positive_ratio",
    "semester_positive_ratio",
    "rolling_63_positive_ratio",
    "rolling_63_worst_net_pnl",
    "rolling_63_median_sharpe",
]


@dataclass(frozen=True)
class StressScenario:
    name: str
    family: str
    description: str
    slippage_multiplier: float = 1.0
    commission_multiplier: float = 1.0
    entry_delay_bars: int = 0
    note: str = ""


@dataclass(frozen=True)
class DailyLossLimiterScenario:
    name: str
    threshold_usd: float


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
    summary_by_scope: pd.DataFrame
    note: str = ""
    verdict: str = ""


@dataclass(frozen=True)
class MnqPropSurvivabilitySpec:
    symbol: str = "MNQ"
    dataset_path: Path | None = None
    is_fraction: float = 0.70
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
    aggregation_rule: str = "majority_50"
    rolling_window_sessions: int = 63
    min_period_trades: int = 20
    prop_constraints: PropConstraintConfig = field(default_factory=build_prop_constraints)
    stress_scenarios: tuple[StressScenario, ...] = (
        StressScenario(
            name="nominal",
            family="stress_execution",
            description="Baseline valide sous hypotheses d'execution nominales du repo.",
            note="Reference for all comparisons.",
        ),
        StressScenario(
            name="slippage_x1p5",
            family="stress_execution",
            description="Stress de slippage a 1.5x le nominal.",
            slippage_multiplier=1.5,
        ),
        StressScenario(
            name="slippage_x2",
            family="stress_execution",
            description="Stress de slippage a 2.0x le nominal.",
            slippage_multiplier=2.0,
        ),
        StressScenario(
            name="slippage_x3",
            family="stress_execution",
            description="Stress de slippage a 3.0x le nominal.",
            slippage_multiplier=3.0,
        ),
        StressScenario(
            name="commission_plus_25pct",
            family="stress_execution",
            description="Stress de commission +25%.",
            commission_multiplier=1.25,
        ),
        StressScenario(
            name="commission_plus_50pct",
            family="stress_execution",
            description="Stress de commission +50%.",
            commission_multiplier=1.50,
        ),
        StressScenario(
            name="entry_delay_1bar",
            family="stress_execution",
            description="Entree degradee d'une barre supplementaire.",
            entry_delay_bars=1,
            note="The signal selection is unchanged; only execution timing is worsened.",
        ),
    )
    daily_loss_limiter_scenarios: tuple[DailyLossLimiterScenario, ...] = (
        DailyLossLimiterScenario(name="daily_loss_limit_500", threshold_usd=500.0),
        DailyLossLimiterScenario(name="daily_loss_limit_750", threshold_usd=750.0),
        DailyLossLimiterScenario(name="daily_loss_limit_1000", threshold_usd=1000.0),
    )
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
        if not math.isfinite(float(value)):
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


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return float(default)
    value = numerator / denominator
    return float(value) if math.isfinite(value) else float(default)


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
    daily["equity"] = float(initial_capital) + pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0).cumsum()
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
        episodes.append(
            {
                "episode_id": episode_id,
                "start_date": start_date,
                "trough_date": trough_date,
                "recovery_date": pd.NaT,
                "peak_equity_before": peak_before,
                "trough_drawdown_usd": trough_drawdown,
                "duration_sessions": len(daily) - start_idx,
                "recovered": False,
            }
        )
    return pd.DataFrame(episodes, columns=columns)


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
            {
                "daily_pnl_usd": 0.0,
                "daily_gross_pnl_usd": 0.0,
                "daily_fees_usd": 0.0,
                "daily_trade_count": 0,
                "daily_loss_count": 0,
            }
        )

    daily = daily.sort_values("session_date").reset_index(drop=True)
    daily["equity"] = float(initial_capital) + pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").cumsum()
    daily["peak_equity"] = daily["equity"].cummax()
    daily["drawdown_usd"] = daily["equity"] - daily["peak_equity"]
    daily["drawdown_pct"] = np.where(
        daily["peak_equity"] > 0,
        (daily["peak_equity"] - daily["equity"]) / daily["peak_equity"],
        0.0,
    )
    daily["green_day"] = daily["daily_pnl_usd"] > 0
    return daily


def _subset_frame_by_sessions(frame: pd.DataFrame, sessions: list) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    session_set = set(pd.to_datetime(pd.Index(sessions)).date)
    out = frame.copy()
    out_dates = pd.to_datetime(out["session_date"]).dt.date
    return out.loc[out_dates.isin(session_set)].copy().reset_index(drop=True)


def _selected_ensemble_sessions(analysis: SymbolAnalysis, aggregation_rule: str) -> set:
    point_pass = analyze_symbol_cache_pass_matrix(analysis)
    pass_cols = [column for column in point_pass.columns if column.startswith("pass__")]
    if not pass_cols:
        return set()
    scored = point_pass.copy()
    scored["consensus_score"] = scored[pass_cols].sum(axis=1) / float(len(pass_cols))
    threshold = resolve_aggregation_threshold(aggregation_rule)
    return set(scored.loc[scored["consensus_score"] >= threshold, "session_date"])


def _run_backtest_with_execution(
    signal_df: pd.DataFrame,
    baseline: BaselineSpec,
    execution_model: ExecutionModel,
    instrument_spec: dict[str, Any],
    entry_delay_bars: int = 0,
    risk_multiplier: float = 1.0,
) -> pd.DataFrame:
    scaled_risk_pct = float(baseline.risk_per_trade_pct) * float(risk_multiplier)
    if scaled_risk_pct <= 0:
        return empty_trade_log()
    return run_backtest(
        signal_df,
        execution_model=execution_model,
        tick_value_usd=float(instrument_spec["tick_value_usd"]),
        point_value_usd=float(instrument_spec["point_value_usd"]),
        time_exit=baseline.time_exit,
        stop_buffer_ticks=baseline.stop_buffer_ticks,
        target_multiple=baseline.target_multiple,
        account_size_usd=baseline.account_size_usd,
        risk_per_trade_pct=scaled_risk_pct,
        entry_on_next_open=baseline.entry_on_next_open,
        entry_delay_bars=entry_delay_bars,
    )


def _build_execution_model(
    instrument_spec: dict[str, Any],
    slippage_multiplier: float = 1.0,
    commission_multiplier: float = 1.0,
) -> ExecutionModel:
    return ExecutionModel(
        commission_per_side_usd=float(instrument_spec["commission_per_side_usd"]) * float(commission_multiplier),
        slippage_ticks=float(instrument_spec["slippage_ticks"]) * float(slippage_multiplier),
        tick_size=float(instrument_spec["tick_size"]),
    )


def _compute_scope_summary(
    trades: pd.DataFrame,
    daily_results: pd.DataFrame,
    sessions: list,
    initial_capital: float,
    constraints: PropConstraintConfig,
) -> dict[str, Any]:
    daily = daily_results.copy() if not daily_results.empty else _rebuild_daily_results_from_trades(trades, sessions, initial_capital)
    daily = daily.sort_values("session_date").reset_index(drop=True)
    base = compute_metrics(
        trades,
        session_dates=sessions,
        initial_capital=initial_capital,
        prop_constraints=constraints,
    )

    daily_pnl = pd.to_numeric(daily.get("daily_pnl_usd"), errors="coerce").fillna(0.0)
    recovery = _drawdown_episode_table(daily, initial_capital=initial_capital)
    recovery_durations = pd.to_numeric(recovery.get("duration_sessions"), errors="coerce").dropna()
    day_loss_lengths = _negative_streak_lengths(daily_pnl)

    return {
        "net_pnl": float(base.get("cumulative_pnl", 0.0)),
        "sharpe": float(base.get("sharpe_ratio", 0.0)),
        "sortino": _sortino_ratio(daily_pnl, initial_capital),
        "profit_factor": float(base.get("profit_factor", 0.0)),
        "expectancy": float(base.get("expectancy", 0.0)),
        "max_drawdown": float(base.get("max_drawdown", 0.0)),
        "n_trades": int(base.get("n_trades", 0)),
        "n_days_traded": int((pd.to_numeric(daily.get("daily_trade_count"), errors="coerce").fillna(0.0) > 0).sum()),
        "pct_days_traded": float(base.get("percent_of_days_traded", 0.0)),
        "worst_day": float(daily_pnl.min()) if not daily.empty else 0.0,
        "longest_losing_streak_daily": int(max(day_loss_lengths, default=0)),
        "median_recovery_days": float(recovery_durations.median()) if not recovery_durations.empty else np.nan,
        "max_recovery_days": float(recovery_durations.max()) if not recovery_durations.empty else np.nan,
        "daily_loss_limit_breach_freq": _safe_div(
            float(base.get("number_of_daily_loss_limit_breaches", 0)),
            max(len(daily), 1),
            default=0.0,
        ),
        "days_to_profit_target": float(base.get("days_to_profit_target", np.nan)),
        "profit_target_reached_before_max_loss": bool(base.get("profit_target_reached_before_max_loss", False)),
        "max_loss_limit_buffer_usd": float(base.get("max_loss_limit_buffer_usd", 0.0)),
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
        sub_trades = _subset_frame_by_sessions(trades, sessions)
        sub_daily = _subset_frame_by_sessions(daily_results, sessions)
        rows.append(
            {
                "scope": scope,
                **_compute_scope_summary(
                    trades=sub_trades,
                    daily_results=sub_daily,
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


def _phase_for_sessions(sessions: list, is_set: set, oos_set: set) -> str:
    session_set = set(pd.to_datetime(pd.Index(sessions)).date)
    if session_set and session_set.issubset(is_set):
        return "is"
    if session_set and session_set.issubset(oos_set):
        return "oos"
    return "mixed"


def _period_summary_rows(
    variant: VariantRun,
    all_sessions: list,
    initial_capital: float,
    constraints: PropConstraintConfig,
    is_sessions: list,
    oos_sessions: list,
    period: str,
    min_period_trades: int,
) -> pd.DataFrame:
    daily = variant.daily_results.copy()
    if daily.empty:
        return pd.DataFrame(
            columns=[
                "variant_name",
                "family",
                "period_type",
                "period_label",
                "phase",
                "n_sessions",
                "sufficient_observations",
                "net_pnl",
                "sharpe",
                "sortino",
                "profit_factor",
                "expectancy",
                "max_drawdown",
                "n_trades",
                "n_days_traded",
                "pct_days_traded",
                "worst_day",
                "longest_losing_streak_daily",
                "median_recovery_days",
                "max_recovery_days",
            ]
        )

    out = daily.copy()
    out["session_date"] = pd.to_datetime(out["session_date"])
    if period == "year":
        out["period_label"] = out["session_date"].dt.year.astype(str)
    elif period == "semester":
        semester = np.where(out["session_date"].dt.month <= 6, "H1", "H2")
        out["period_label"] = out["session_date"].dt.year.astype(str) + semester
    else:
        raise ValueError(f"Unsupported period '{period}'.")

    is_set = set(pd.to_datetime(pd.Index(is_sessions)).date)
    oos_set = set(pd.to_datetime(pd.Index(oos_sessions)).date)
    rows: list[dict[str, Any]] = []

    for label, period_daily in out.groupby("period_label", sort=True):
        period_sessions = pd.Index(period_daily["session_date"].dt.date.unique()).tolist()
        period_trades = _subset_frame_by_sessions(variant.trades, period_sessions)
        summary = _compute_scope_summary(
            trades=period_trades,
            daily_results=period_daily.assign(session_date=period_daily["session_date"].dt.date),
            sessions=period_sessions,
            initial_capital=initial_capital,
            constraints=constraints,
        )
        rows.append(
            {
                "variant_name": variant.name,
                "family": variant.family,
                "period_type": period,
                "period_label": label,
                "phase": _phase_for_sessions(period_sessions, is_set=is_set, oos_set=oos_set),
                "n_sessions": int(len(period_sessions)),
                "sufficient_observations": bool(int(summary["n_trades"]) >= int(min_period_trades)),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def _rolling_summary_rows(
    variant: VariantRun,
    all_sessions: list,
    initial_capital: float,
    constraints: PropConstraintConfig,
    is_sessions: list,
    oos_sessions: list,
    window_sessions: int,
) -> pd.DataFrame:
    if len(all_sessions) < window_sessions:
        return pd.DataFrame(
            columns=[
                "variant_name",
                "family",
                "period_type",
                "window_start",
                "window_end",
                "phase",
                "n_sessions",
                "net_pnl",
                "sharpe",
                "sortino",
                "profit_factor",
                "expectancy",
                "max_drawdown",
                "n_trades",
                "n_days_traded",
                "pct_days_traded",
                "worst_day",
                "longest_losing_streak_daily",
            ]
        )

    is_set = set(pd.to_datetime(pd.Index(is_sessions)).date)
    oos_set = set(pd.to_datetime(pd.Index(oos_sessions)).date)
    rows: list[dict[str, Any]] = []

    for end_idx in range(window_sessions - 1, len(all_sessions)):
        window = all_sessions[end_idx - window_sessions + 1 : end_idx + 1]
        period_trades = _subset_frame_by_sessions(variant.trades, window)
        period_daily = _subset_frame_by_sessions(variant.daily_results, window)
        summary = _compute_scope_summary(
            trades=period_trades,
            daily_results=period_daily,
            sessions=window,
            initial_capital=initial_capital,
            constraints=constraints,
        )
        rows.append(
            {
                "variant_name": variant.name,
                "family": variant.family,
                "period_type": f"rolling_{window_sessions}d",
                "window_start": pd.to_datetime(window[0]).date(),
                "window_end": pd.to_datetime(window[-1]).date(),
                "phase": _phase_for_sessions(window, is_set=is_set, oos_set=oos_set),
                "n_sessions": int(len(window)),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def compute_drawdown_deleveraging_multipliers(
    daily_pnl_sequence: list[float] | pd.Series,
    initial_capital: float,
    soft_drawdown_pct: float = 0.03,
    hard_drawdown_pct: float = 0.05,
    soft_multiplier: float = 0.75,
    hard_multiplier: float = 0.50,
) -> list[float]:
    equity = float(initial_capital)
    peak_equity = float(initial_capital)
    multipliers: list[float] = []
    for pnl in pd.Series(daily_pnl_sequence, dtype=float).fillna(0.0):
        drawdown_pct = _safe_div(peak_equity - equity, peak_equity, default=0.0)
        if drawdown_pct > hard_drawdown_pct:
            multiplier = float(hard_multiplier)
        elif drawdown_pct > soft_drawdown_pct:
            multiplier = float(soft_multiplier)
        else:
            multiplier = 1.0
        multipliers.append(multiplier)
        equity += float(pnl)
        peak_equity = max(peak_equity, equity)
    return multipliers


def compute_skip_after_large_loss_multipliers(
    daily_pnl_sequence: list[float] | pd.Series,
    threshold_usd: float,
    cooldown_days: int = 1,
) -> list[float]:
    remaining = 0
    multipliers: list[float] = []
    for pnl in pd.Series(daily_pnl_sequence, dtype=float).fillna(0.0):
        if remaining > 0:
            multipliers.append(0.0)
            remaining -= 1
            continue
        multipliers.append(1.0)
        if float(pnl) <= -float(threshold_usd):
            remaining = int(max(cooldown_days, 0))
    return multipliers


def compute_half_after_two_red_days_multipliers(
    daily_pnl_sequence: list[float] | pd.Series,
    traded_day_mask: list[bool] | pd.Series,
) -> list[float]:
    streak = 0
    multipliers: list[float] = []
    pnls = pd.Series(daily_pnl_sequence, dtype=float).fillna(0.0)
    traded = pd.Series(traded_day_mask).fillna(False).astype(bool)
    for pnl, traded_day in zip(pnls.tolist(), traded.tolist()):
        multipliers.append(0.5 if streak >= 2 else 1.0)
        if not traded_day:
            continue
        if pnl < 0:
            streak += 1
        else:
            streak = 0
    return multipliers


def compute_skip_after_three_red_days_multipliers(
    daily_pnl_sequence: list[float] | pd.Series,
    traded_day_mask: list[bool] | pd.Series,
) -> list[float]:
    streak = 0
    skip_next = False
    multipliers: list[float] = []
    pnls = pd.Series(daily_pnl_sequence, dtype=float).fillna(0.0)
    traded = pd.Series(traded_day_mask).fillna(False).astype(bool)
    for pnl, traded_day in zip(pnls.tolist(), traded.tolist()):
        if skip_next:
            multipliers.append(0.0)
            skip_next = False
            streak = 0
            continue
        multipliers.append(1.0)
        if not traded_day:
            continue
        if pnl < 0:
            streak += 1
        else:
            streak = 0
        if streak >= 3:
            skip_next = True
    return multipliers


def _empty_controls(all_sessions: list, selected_sessions: set, variant_name: str, family: str) -> pd.DataFrame:
    rows = [
        {
            "session_date": pd.to_datetime(session).date(),
            "variant_name": variant_name,
            "family": family,
            "selected_by_ensemble": bool(session in selected_sessions),
            "policy_allows_trading": bool(session in selected_sessions),
            "risk_multiplier": 1.0 if session in selected_sessions else 0.0,
            "policy_reason": "ensemble_selected" if session in selected_sessions else "ensemble_filtered_out",
            "trade_executed": False,
            "state_pre_json": "{}",
            "state_post_json": "{}",
        }
        for session in all_sessions
    ]
    return pd.DataFrame(rows)


def _finalize_variant_run(
    name: str,
    family: str,
    description: str,
    calibration_scope: str,
    parameters: dict[str, Any],
    trades: pd.DataFrame,
    all_sessions: list,
    is_sessions: list,
    oos_sessions: list,
    initial_capital: float,
    constraints: PropConstraintConfig,
    controls: pd.DataFrame,
    note: str = "",
) -> VariantRun:
    if trades.empty:
        trade_frame = empty_trade_log()
    else:
        trade_frame = trades.copy().sort_values("exit_time").reset_index(drop=True)
        trade_frame["trade_id"] = np.arange(1, len(trade_frame) + 1)

    daily_results = _rebuild_daily_results_from_trades(
        trade_frame,
        all_sessions=all_sessions,
        initial_capital=initial_capital,
    )
    if not controls.empty:
        controls = controls.copy()
        controls["session_date"] = pd.to_datetime(controls["session_date"]).dt.date
        daily_results = daily_results.merge(controls, on="session_date", how="left")
    summary_by_scope = _build_summary_by_scope(
        trades=trade_frame,
        daily_results=daily_results,
        all_sessions=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        initial_capital=initial_capital,
        constraints=constraints,
    )
    return VariantRun(
        name=name,
        family=family,
        description=description,
        calibration_scope=calibration_scope,
        parameters=parameters,
        trades=trade_frame,
        daily_results=daily_results,
        controls=controls,
        summary_by_scope=summary_by_scope,
        note=note,
    )


def _run_stress_variant(
    analysis: SymbolAnalysis,
    selected_sessions: set,
    scenario: StressScenario,
    constraints: PropConstraintConfig,
) -> VariantRun:
    execution_model = _build_execution_model(
        analysis.instrument_spec,
        slippage_multiplier=scenario.slippage_multiplier,
        commission_multiplier=scenario.commission_multiplier,
    )
    if (
        scenario.slippage_multiplier == 1.0
        and scenario.commission_multiplier == 1.0
        and scenario.entry_delay_bars == 0
    ):
        trades = analysis.baseline_trades.loc[analysis.baseline_trades["session_date"].isin(selected_sessions)].copy()
    else:
        trades_all = _run_backtest_with_execution(
            signal_df=analysis.signal_df,
            baseline=analysis.baseline,
            execution_model=execution_model,
            instrument_spec=analysis.instrument_spec,
            entry_delay_bars=scenario.entry_delay_bars,
            risk_multiplier=1.0,
        )
        trades = trades_all.loc[trades_all["session_date"].isin(selected_sessions)].copy()

    controls = _empty_controls(analysis.all_sessions, selected_sessions, variant_name=scenario.name, family=scenario.family)
    traded_sessions = set(pd.to_datetime(trades["session_date"]).dt.date) if not trades.empty else set()
    controls["trade_executed"] = pd.to_datetime(controls["session_date"]).dt.date.isin(traded_sessions)
    return _finalize_variant_run(
        name=scenario.name,
        family=scenario.family,
        description=scenario.description,
        calibration_scope="frozen_is_thresholds",
        parameters={
            "slippage_multiplier": scenario.slippage_multiplier,
            "commission_multiplier": scenario.commission_multiplier,
            "entry_delay_bars": scenario.entry_delay_bars,
        },
        trades=trades,
        all_sessions=analysis.all_sessions,
        is_sessions=analysis.is_sessions,
        oos_sessions=analysis.oos_sessions,
        initial_capital=analysis.baseline.account_size_usd,
        constraints=constraints,
        controls=controls,
        note=scenario.note,
    )


def _run_sequential_overlay(
    analysis: SymbolAnalysis,
    selected_sessions: set,
    name: str,
    family: str,
    description: str,
    constraints: PropConstraintConfig,
    initial_state: dict[str, Any],
    pre_session_control: Callable[[dict[str, Any]], tuple[float, str]],
    post_session_update: Callable[[dict[str, Any], bool, float], None],
) -> VariantRun:
    session_frames = {
        pd.to_datetime(session_date).date(): frame.sort_values("timestamp").copy()
        for session_date, frame in analysis.signal_df.groupby("session_date", sort=True)
    }
    execution_model = _build_execution_model(analysis.instrument_spec)

    trade_parts: list[pd.DataFrame] = []
    control_rows: list[dict[str, Any]] = []
    state = dict(initial_state)

    for session in analysis.all_sessions:
        session_date = pd.to_datetime(session).date()
        selected = bool(session_date in selected_sessions)
        multiplier, reason = pre_session_control(state)
        daily_pnl = 0.0
        traded = False

        session_df = session_frames.get(session_date)
        if selected and multiplier > 0 and session_df is not None and not session_df.empty:
            day_trades = _run_backtest_with_execution(
                signal_df=session_df,
                baseline=analysis.baseline,
                execution_model=execution_model,
                instrument_spec=analysis.instrument_spec,
                risk_multiplier=multiplier,
            )
            if not day_trades.empty:
                traded = True
                day_trades = day_trades.copy()
                day_trades["overlay_variant"] = name
                day_trades["risk_multiplier"] = multiplier
                trade_parts.append(day_trades)
                daily_pnl = float(day_trades["net_pnl_usd"].sum())

        state_pre_json = json.dumps({key: _serialize_value(value) for key, value in state.items()}, sort_keys=True)
        post_session_update(state, traded, daily_pnl)
        state_post_json = json.dumps({key: _serialize_value(value) for key, value in state.items()}, sort_keys=True)

        control_rows.append(
            {
                "session_date": session_date,
                "variant_name": name,
                "family": family,
                "selected_by_ensemble": selected,
                "policy_allows_trading": bool(selected and multiplier > 0),
                "risk_multiplier": float(multiplier) if selected else 0.0,
                "policy_reason": reason if selected else "ensemble_filtered_out",
                "trade_executed": traded,
                "state_pre_json": state_pre_json,
                "state_post_json": state_post_json,
            }
        )

    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else empty_trade_log()
    controls = pd.DataFrame(control_rows)
    return _finalize_variant_run(
        name=name,
        family=family,
        description=description,
        calibration_scope="frozen_is_thresholds",
        parameters={},
        trades=trades,
        all_sessions=analysis.all_sessions,
        is_sessions=analysis.is_sessions,
        oos_sessions=analysis.oos_sessions,
        initial_capital=analysis.baseline.account_size_usd,
        constraints=constraints,
        controls=controls,
    )


def _run_drawdown_deleveraging_variant(
    analysis: SymbolAnalysis,
    selected_sessions: set,
    constraints: PropConstraintConfig,
) -> VariantRun:
    initial_capital = float(analysis.baseline.account_size_usd)

    def pre_session_control(state: dict[str, Any]) -> tuple[float, str]:
        drawdown_pct = _safe_div(state["peak_equity"] - state["equity"], state["peak_equity"], default=0.0)
        if drawdown_pct > 0.05:
            return 0.50, "drawdown_gt_5pct"
        if drawdown_pct > 0.03:
            return 0.75, "drawdown_gt_3pct"
        return 1.0, "normal_risk"

    def post_session_update(state: dict[str, Any], traded: bool, daily_pnl: float) -> None:
        state["equity"] = float(state["equity"]) + float(daily_pnl)
        state["peak_equity"] = max(float(state["peak_equity"]), float(state["equity"]))
        state["traded"] = bool(traded)
        state["last_daily_pnl"] = float(daily_pnl)

    return _run_sequential_overlay(
        analysis=analysis,
        selected_sessions=selected_sessions,
        name="deleveraging_drawdown_3pct_5pct",
        family="deleveraging",
        description="Taille x0.75 au-dela de 3% de drawdown courant, puis x0.50 au-dela de 5%, reset au nouveau high-watermark.",
        constraints=constraints,
        initial_state={"equity": initial_capital, "peak_equity": initial_capital},
        pre_session_control=pre_session_control,
        post_session_update=post_session_update,
    )


def _run_skip_after_large_loss_variant(
    analysis: SymbolAnalysis,
    selected_sessions: set,
    constraints: PropConstraintConfig,
    threshold_usd: float,
) -> VariantRun:
    def pre_session_control(state: dict[str, Any]) -> tuple[float, str]:
        if int(state.get("cooldown_remaining", 0)) > 0:
            return 0.0, "cooldown_after_large_loss"
        return 1.0, "normal_risk"

    def post_session_update(state: dict[str, Any], traded: bool, daily_pnl: float) -> None:
        remaining = int(state.get("cooldown_remaining", 0))
        if remaining > 0:
            state["cooldown_remaining"] = remaining - 1
            state["last_daily_pnl"] = 0.0
            return
        if traded and float(daily_pnl) <= -float(threshold_usd):
            state["cooldown_remaining"] = 1
        else:
            state["cooldown_remaining"] = 0
        state["last_daily_pnl"] = float(daily_pnl)

    return _run_sequential_overlay(
        analysis=analysis,
        selected_sessions=selected_sessions,
        name="cooldown_after_large_loss_1r",
        family="cooldown",
        description="Pause d'un jour calendaire apres une perte journaliere inferieure ou egale a -1R nominal (-750 USD).",
        constraints=constraints,
        initial_state={"cooldown_remaining": 0},
        pre_session_control=pre_session_control,
        post_session_update=post_session_update,
    )


def _run_half_after_two_red_days_variant(
    analysis: SymbolAnalysis,
    selected_sessions: set,
    constraints: PropConstraintConfig,
) -> VariantRun:
    def pre_session_control(state: dict[str, Any]) -> tuple[float, str]:
        streak = int(state.get("loss_streak", 0))
        if streak >= 2:
            return 0.5, "half_size_after_2_red_days"
        return 1.0, "normal_risk"

    def post_session_update(state: dict[str, Any], traded: bool, daily_pnl: float) -> None:
        if not traded:
            return
        if float(daily_pnl) < 0:
            state["loss_streak"] = int(state.get("loss_streak", 0)) + 1
        else:
            state["loss_streak"] = 0
        state["last_daily_pnl"] = float(daily_pnl)

    return _run_sequential_overlay(
        analysis=analysis,
        selected_sessions=selected_sessions,
        name="half_after_2_red_days",
        family="cooldown",
        description="Taille reduite de moitie apres 2 jours rouges consecutifs sur jours effectivement trades.",
        constraints=constraints,
        initial_state={"loss_streak": 0},
        pre_session_control=pre_session_control,
        post_session_update=post_session_update,
    )


def _run_skip_after_three_red_days_variant(
    analysis: SymbolAnalysis,
    selected_sessions: set,
    constraints: PropConstraintConfig,
) -> VariantRun:
    def pre_session_control(state: dict[str, Any]) -> tuple[float, str]:
        if bool(state.get("skip_next_session", False)):
            return 0.0, "skip_after_3_red_days"
        return 1.0, "normal_risk"

    def post_session_update(state: dict[str, Any], traded: bool, daily_pnl: float) -> None:
        if bool(state.get("skip_next_session", False)):
            state["skip_next_session"] = False
            state["loss_streak"] = 0
            state["last_daily_pnl"] = 0.0
            return
        if not traded:
            return
        if float(daily_pnl) < 0:
            state["loss_streak"] = int(state.get("loss_streak", 0)) + 1
        else:
            state["loss_streak"] = 0
        if int(state.get("loss_streak", 0)) >= 3:
            state["skip_next_session"] = True
        state["last_daily_pnl"] = float(daily_pnl)

    return _run_sequential_overlay(
        analysis=analysis,
        selected_sessions=selected_sessions,
        name="pause_after_3_red_days",
        family="cooldown",
        description="Pause d'un jour calendaire apres 3 jours rouges consecutifs sur jours effectivement trades.",
        constraints=constraints,
        initial_state={"loss_streak": 0, "skip_next_session": False},
        pre_session_control=pre_session_control,
        post_session_update=post_session_update,
    )


def _daily_loss_limiter_diagnostic_rows(
    variant: VariantRun,
    scenarios: tuple[DailyLossLimiterScenario, ...],
) -> pd.DataFrame:
    daily = variant.daily_results.copy()
    pnl = pd.to_numeric(daily.get("daily_pnl_usd"), errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        breaches = int((pnl <= -float(scenario.threshold_usd)).sum()) if len(pnl) > 0 else 0
        rows.append(
            {
                "diagnostic_name": scenario.name,
                "threshold_usd": float(scenario.threshold_usd),
                "breach_days": breaches,
                "breach_freq": _safe_div(breaches, len(daily), default=0.0),
                "effective_behavioral_change": False,
                "explanation": "No intraday effect with one trade per day and no trailing daily-stop engine.",
                "reference_variant": variant.name,
            }
        )
    return pd.DataFrame(rows)


def _period_stability_meta(
    period_rows: pd.DataFrame,
    rolling_rows: pd.DataFrame,
) -> dict[str, Any]:
    year_rows = period_rows.loc[period_rows["period_type"] == "year"].copy()
    semester_rows = period_rows.loc[period_rows["period_type"] == "semester"].copy()
    year_positive_ratio = float((pd.to_numeric(year_rows.get("net_pnl"), errors="coerce") > 0).mean()) if not year_rows.empty else np.nan
    semester_positive_ratio = float((pd.to_numeric(semester_rows.get("net_pnl"), errors="coerce") > 0).mean()) if not semester_rows.empty else np.nan
    rolling_positive_ratio = float((pd.to_numeric(rolling_rows.get("net_pnl"), errors="coerce") > 0).mean()) if not rolling_rows.empty else np.nan
    worst_rolling = float(pd.to_numeric(rolling_rows.get("net_pnl"), errors="coerce").min()) if not rolling_rows.empty else np.nan
    median_rolling_sharpe = float(pd.to_numeric(rolling_rows.get("sharpe"), errors="coerce").median()) if not rolling_rows.empty else np.nan
    return {
        "year_positive_ratio": year_positive_ratio,
        "semester_positive_ratio": semester_positive_ratio,
        "rolling_63_positive_ratio": rolling_positive_ratio,
        "rolling_63_worst_net_pnl": worst_rolling,
        "rolling_63_median_sharpe": median_rolling_sharpe,
    }


def _variant_verdict(row: dict[str, Any], baseline_row: dict[str, Any]) -> str:
    family = str(row.get("family", ""))
    if family == "daily_loss_limiter":
        return "trivial_no_effect"
    if row.get("variant_name") == baseline_row.get("variant_name"):
        return "baseline_reference"

    dd_improvement = float(row.get("oos_max_drawdown_improvement_vs_nominal", 0.0))
    pnl_retention = float(row.get("oos_net_pnl_retention_vs_nominal", 0.0))
    trade_retention = float(row.get("oos_trade_retention_vs_nominal", 0.0))

    if dd_improvement >= 0.10 and pnl_retention >= 0.85 and trade_retention >= 0.80:
        return "useful_prop_overlay"
    if dd_improvement >= 0.10 and (pnl_retention < 0.85 or trade_retention < 0.80):
        return "protective_but_costly"
    if abs(dd_improvement) < 0.03 and 0.95 <= pnl_retention <= 1.05:
        return "mostly_cosmetic"
    if dd_improvement < 0 and pnl_retention < 0.95:
        return "worse_than_baseline"
    return "mixed"


def _variant_row(
    variant: VariantRun,
    nominal_variant: VariantRun,
    period_rows: pd.DataFrame,
    rolling_rows: pd.DataFrame,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant_name": variant.name,
        "family": variant.family,
        "description": variant.description,
        "calibration_scope": variant.calibration_scope,
        "parameters_json": json.dumps({key: _serialize_value(value) for key, value in variant.parameters.items()}, sort_keys=True),
        "note": variant.note,
        "selected_session_count": int(variant.controls["selected_by_ensemble"].fillna(False).sum()) if not variant.controls.empty else 0,
        "policy_skip_days": int(
            (
                variant.controls["selected_by_ensemble"].fillna(False)
                & pd.to_numeric(variant.controls.get("risk_multiplier"), errors="coerce").fillna(0.0).eq(0.0)
            ).sum()
        )
        if not variant.controls.empty
        else 0,
    }

    for scope in ("overall", "is", "oos"):
        for column in [
            "net_pnl",
            "sharpe",
            "sortino",
            "profit_factor",
            "expectancy",
            "max_drawdown",
            "n_trades",
            "n_days_traded",
            "pct_days_traded",
            "worst_day",
            "longest_losing_streak_daily",
            "median_recovery_days",
            "max_recovery_days",
            "daily_loss_limit_breach_freq",
            "days_to_profit_target",
            "profit_target_reached_before_max_loss",
            "max_loss_limit_buffer_usd",
        ]:
            row[f"{scope}_{column}"] = _scope_value(variant.summary_by_scope, scope, column)

    baseline_oos_dd = abs(float(_scope_value(nominal_variant.summary_by_scope, "oos", "max_drawdown")))
    variant_oos_dd = abs(float(_scope_value(variant.summary_by_scope, "oos", "max_drawdown")))
    row["oos_net_pnl_retention_vs_nominal"] = _safe_div(
        float(_scope_value(variant.summary_by_scope, "oos", "net_pnl")),
        float(_scope_value(nominal_variant.summary_by_scope, "oos", "net_pnl")),
        default=0.0,
    )
    row["oos_trade_retention_vs_nominal"] = _safe_div(
        float(_scope_value(variant.summary_by_scope, "oos", "n_trades")),
        float(_scope_value(nominal_variant.summary_by_scope, "oos", "n_trades")),
        default=0.0,
    )
    row["oos_day_retention_vs_nominal"] = _safe_div(
        float(_scope_value(variant.summary_by_scope, "oos", "n_days_traded")),
        float(_scope_value(nominal_variant.summary_by_scope, "oos", "n_days_traded")),
        default=0.0,
    )
    row["oos_max_drawdown_improvement_vs_nominal"] = _safe_div(
        baseline_oos_dd - variant_oos_dd,
        max(baseline_oos_dd, 1.0),
        default=0.0,
    )
    row.update(_period_stability_meta(period_rows=period_rows, rolling_rows=rolling_rows))
    return row


def _export_variant_artifacts(root: Path, variant: VariantRun) -> None:
    variant_dir = root / "variants" / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    variant.trades.to_csv(variant_dir / "trades.csv", index=False)
    variant.daily_results.to_csv(variant_dir / "daily_results.csv", index=False)
    variant.controls.to_csv(variant_dir / "controls.csv", index=False)
    variant.summary_by_scope.to_csv(variant_dir / "metrics_by_scope.csv", index=False)


def _write_summary_markdown(
    output_path: Path,
    analysis: SymbolAnalysis,
    summary_df: pd.DataFrame,
    daily_loss_diag: pd.DataFrame,
) -> None:
    nominal = summary_df.loc[summary_df["variant_name"] == "nominal"].iloc[0]
    non_trivial = summary_df.loc[
        ~summary_df["family"].isin(["stress_execution", "daily_loss_limiter"])
        & summary_df["variant_name"].ne("nominal")
    ].copy()
    useful = non_trivial.loc[non_trivial["verdict"] == "useful_prop_overlay"].copy()
    protective = non_trivial.loc[non_trivial["verdict"] == "protective_but_costly"].copy()
    mixed = non_trivial.loc[non_trivial["verdict"].isin(["mixed", "worse_than_baseline", "mostly_cosmetic"])].copy()

    useful_line = (
        f"- Overlay simple le plus defendable: `{useful.iloc[0]['variant_name']}` "
        f"(retention OOS {float(useful.iloc[0]['oos_net_pnl_retention_vs_nominal']):.2f}, "
        f"amelioration maxDD OOS {100.0 * float(useful.iloc[0]['oos_max_drawdown_improvement_vs_nominal']):.1f}%)."
        if not useful.empty
        else "- Aucun overlay simple ne passe le filtre utile = drawdown mieux controle sans couper trop d'exposition."
    )
    protective_line = (
        f"- Overlay protecteur mais couteux: `{protective.iloc[0]['variant_name']}`."
        if not protective.empty
        else "- Aucun overlay n'apporte un compromis protection / retention vraiment convaincant."
    )
    mixed_line = (
        f"- Overlay a eviter ou cosmetique: `{mixed.iloc[0]['variant_name']}`."
        if not mixed.empty
        else "- Aucun overlay simple n'apparait franchement negatif."
    )

    if not useful.empty:
        final_verdict = (
            f"L'edge MNQ reste exploitable apres stress, et `{useful.iloc[0]['variant_name']}` est le seul overlay simple qui ameliore la respirabilite sans destruction evidente du profil OOS."
        )
    else:
        final_verdict = (
            "L'edge MNQ valide tient encore sous stress raisonnable, mais aucun overlay simple teste ici ne justifie clairement une adoption systematique en plus de la baseline. "
            "Dans ce setup 1 trade/jour, la compatibilite prop vient surtout du sizing de depart et de l'acceptation du path risk, pas d'un kill switch magique."
        )

    lines = [
        "# MNQ ORB Prop Survivability Campaign",
        "",
        "## Baseline",
        "",
        f"- Baseline rehydratee: `{analysis.symbol}` / OR{int(analysis.baseline.or_minutes)} / direction `{analysis.baseline.direction}` / RR `{float(analysis.baseline.target_multiple):.1f}` / ensemble `majority_50`.",
        f"- Dataset: `{analysis.dataset_path.name}`",
        f"- IS/OOS: `{len(analysis.is_sessions)}` / `{len(analysis.oos_sessions)}` sessions",
        f"- Grid gele apres calibration IS: ATR `{list(analysis.grid.atr_periods)}` | qlow `{list(analysis.grid.q_lows_pct)}` | qhigh `{list(analysis.grid.q_highs_pct)}` | rule `majority_50`.",
        "",
        "## Readout",
        "",
        f"- Nominal OOS: net pnl `{float(nominal['oos_net_pnl']):.2f}` | Sharpe `{float(nominal['oos_sharpe']):.3f}` | Sortino `{float(nominal['oos_sortino']):.3f}` | PF `{float(nominal['oos_profit_factor']):.3f}` | maxDD `{float(nominal['oos_max_drawdown']):.2f}`.",
        f"- Stabilite temporelle nominale: annees positives `{100.0 * float(nominal['year_positive_ratio']):.1f}%` | semestres positifs `{100.0 * float(nominal['semester_positive_ratio']):.1f}%` | rolling 63j positives `{100.0 * float(nominal['rolling_63_positive_ratio']):.1f}%`.",
        useful_line,
        protective_line,
        mixed_line,
        "",
        "## Daily Loss Limiter",
        "",
        "- Diagnostic honnete: avec deja `1 trade par jour`, un daily loss limiter realise ex-post ne change pas le comportement du systeme dans cette architecture.",
        f"- Seuils verifies: `{', '.join(str(int(x)) for x in daily_loss_diag['threshold_usd'].tolist())}` USD.",
        f"- Breach frequencies observees: `{', '.join(f'{int(row.threshold_usd)} -> {100.0 * float(row.breach_freq):.1f}%' for row in daily_loss_diag.itertuples())}`.",
        "",
        "## Verdict",
        "",
        final_verdict,
        "",
        "## Exports",
        "",
        "- `summary_variants.csv`",
        "- `daily_loss_limiter_diagnostic.csv`",
        "- `temporal_stability_yearly.csv`",
        "- `temporal_stability_semester.csv`",
        "- `temporal_stability_rolling_63d.csv`",
        "- `variants/<variant>/...`",
    ]
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_mnq_prop_survivability_campaign(spec: MnqPropSurvivabilitySpec) -> dict[str, Any]:
    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(spec.output_root) if spec.output_root is not None else EXPORTS_DIR / f"mnq_orb_prop_survivability_{timestamp}"
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

    variants: list[VariantRun] = [
        _run_stress_variant(analysis, selected_sessions, scenario, spec.prop_constraints)
        for scenario in spec.stress_scenarios
    ]
    variants.append(_run_drawdown_deleveraging_variant(analysis, selected_sessions, spec.prop_constraints))

    nominal_risk_budget_usd = float(analysis.baseline.account_size_usd) * float(analysis.baseline.risk_per_trade_pct) / 100.0
    variants.append(
        _run_skip_after_large_loss_variant(
            analysis,
            selected_sessions,
            spec.prop_constraints,
            threshold_usd=nominal_risk_budget_usd,
        )
    )
    variants.append(_run_half_after_two_red_days_variant(analysis, selected_sessions, spec.prop_constraints))
    variants.append(_run_skip_after_three_red_days_variant(analysis, selected_sessions, spec.prop_constraints))

    nominal_variant = next(variant for variant in variants if variant.name == "nominal")
    daily_loss_diag = _daily_loss_limiter_diagnostic_rows(nominal_variant, spec.daily_loss_limiter_scenarios)
    daily_loss_diag_path = output_root / "daily_loss_limiter_diagnostic.csv"
    daily_loss_diag.to_csv(daily_loss_diag_path, index=False)

    period_frames: list[pd.DataFrame] = []
    rolling_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for variant in variants:
        _export_variant_artifacts(output_root, variant)
        period_rows = pd.concat(
            [
                _period_summary_rows(
                    variant,
                    all_sessions=analysis.all_sessions,
                    initial_capital=analysis.baseline.account_size_usd,
                    constraints=spec.prop_constraints,
                    is_sessions=analysis.is_sessions,
                    oos_sessions=analysis.oos_sessions,
                    period="year",
                    min_period_trades=spec.min_period_trades,
                ),
                _period_summary_rows(
                    variant,
                    all_sessions=analysis.all_sessions,
                    initial_capital=analysis.baseline.account_size_usd,
                    constraints=spec.prop_constraints,
                    is_sessions=analysis.is_sessions,
                    oos_sessions=analysis.oos_sessions,
                    period="semester",
                    min_period_trades=spec.min_period_trades,
                ),
            ],
            ignore_index=True,
        )
        rolling_rows = _rolling_summary_rows(
            variant,
            all_sessions=analysis.all_sessions,
            initial_capital=analysis.baseline.account_size_usd,
            constraints=spec.prop_constraints,
            is_sessions=analysis.is_sessions,
            oos_sessions=analysis.oos_sessions,
            window_sessions=spec.rolling_window_sessions,
        )
        period_frames.append(period_rows)
        rolling_frames.append(rolling_rows)
        summary_rows.append(_variant_row(variant, nominal_variant=nominal_variant, period_rows=period_rows, rolling_rows=rolling_rows))

    summary_df = pd.DataFrame(summary_rows)
    baseline_row = summary_df.loc[summary_df["variant_name"] == "nominal"].iloc[0].to_dict()
    summary_df["verdict"] = summary_df.apply(lambda row: _variant_verdict(row.to_dict(), baseline_row), axis=1)
    summary_df = summary_df[[column for column in DEFAULT_VARIANT_COLUMNS if column in summary_df.columns]]
    summary_path = output_root / "summary_variants.csv"
    summary_df.to_csv(summary_path, index=False)

    yearly_rows = pd.concat([frame.loc[frame["period_type"] == "year"] for frame in period_frames], ignore_index=True)
    semester_rows = pd.concat([frame.loc[frame["period_type"] == "semester"] for frame in period_frames], ignore_index=True)
    rolling_rows = pd.concat(rolling_frames, ignore_index=True)

    yearly_path = output_root / "temporal_stability_yearly.csv"
    semester_path = output_root / "temporal_stability_semester.csv"
    rolling_path = output_root / f"temporal_stability_rolling_{spec.rolling_window_sessions}d.csv"
    yearly_rows.to_csv(yearly_path, index=False)
    semester_rows.to_csv(semester_path, index=False)
    rolling_rows.to_csv(rolling_path, index=False)

    markdown_path = output_root / "campaign_summary.md"
    _write_summary_markdown(markdown_path, analysis=analysis, summary_df=summary_df, daily_loss_diag=daily_loss_diag)

    metadata_path = output_root / "run_metadata.json"
    _json_dump(
        metadata_path,
        {
            "run_timestamp": datetime.now().isoformat(),
            "dataset_path": dataset_path,
            "selected_symbol": spec.symbol,
            "selected_aggregation_rule": spec.aggregation_rule,
            "selected_session_count": len(selected_sessions),
            "spec": asdict(spec),
            "analysis_best_ensemble": analysis.best_ensemble,
            "analysis_baseline_transfer": analysis.baseline_transfer,
            "nominal_risk_budget_usd": nominal_risk_budget_usd,
        },
    )

    return {
        "output_root": output_root,
        "summary": summary_path,
        "daily_loss_limiter_diagnostic": daily_loss_diag_path,
        "yearly": yearly_path,
        "semester": semester_path,
        "rolling": rolling_path,
        "markdown": markdown_path,
        "metadata": metadata_path,
    }


def build_default_spec(output_root: Path | None = None) -> MnqPropSurvivabilitySpec:
    return MnqPropSurvivabilitySpec(output_root=output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the validated MNQ ORB prop-survivability campaign.")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--is-fraction", type=float, default=0.70)
    parser.add_argument("--rolling-window-sessions", type=int, default=63)
    args = parser.parse_args()

    spec = MnqPropSurvivabilitySpec(
        dataset_path=Path(args.dataset_path) if args.dataset_path is not None else None,
        output_root=Path(args.output_root) if args.output_root is not None else None,
        is_fraction=float(args.is_fraction),
        rolling_window_sessions=int(args.rolling_window_sessions),
    )
    artifacts = run_mnq_prop_survivability_campaign(spec)
    print(f"summary: {artifacts['summary']}")
    print(f"markdown: {artifacts['markdown']}")


if __name__ == "__main__":
    main()
