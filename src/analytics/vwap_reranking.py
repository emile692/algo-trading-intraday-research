"""Leak-free reranking campaign for simple VWAP variants."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analytics.vwap_validation import (
    StressScenario,
    _apply_cost_stress_overlay,
    _build_scope_summary_table,
    _challenge_empirical_summary,
    _challenge_scenarios,
    _ensure_trade_risk,
    _heatmap_topology_readout,
    _json_dump,
    _notebook_cell,
    _outlier_verdict,
    _rebuild_daily_results_from_trades,
    _split_sessions,
    _subset_frame_by_sessions,
    _time_window_label,
    _variant_cache_key,
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
    build_default_vwap_reranking_variants,
    infer_symbol_from_dataset_path,
    resolve_default_vwap_dataset,
    resolve_vwap_variant,
)
from src.data.cleaning import clean_ohlcv
from src.data.loader import load_ohlcv_file
from src.engine.execution_model import ExecutionModel
from src.engine.vwap_backtester import (
    InstrumentDetails,
    VWAPBacktestResult,
    build_execution_model_for_profile,
    run_vwap_backtest,
)
from src.strategy.vwap import build_vwap_signal_frame, prepare_vwap_feature_frame


DEFAULT_SPLIT_FRACTIONS = (0.60, 0.65, 0.70, 0.75)
DEFAULT_STRESS_SCENARIO = StressScenario(
    name="slippage_x2",
    slippage_multiplier=2.0,
    notes="Primary mandatory stress: slippage doubled.",
)
PAPER_BASELINE_NAME = "paper_vwap_baseline"
REALISTIC_BASELINE_NAME = "baseline_futures_adapted"
RERANKING_MODES = ("baseline", "stress", "splits", "full", "notebook")


@dataclass(frozen=True)
class RerankingSpec:
    """Top-level settings for the VWAP reranking campaign."""

    dataset_path: Path
    variant_names: tuple[str, ...]
    is_fraction: float = 0.70
    split_fractions: tuple[float, ...] = DEFAULT_SPLIT_FRACTIONS
    session_start: str = DEFAULT_RTH_SESSION_START
    session_end: str = DEFAULT_RTH_SESSION_END
    paper_time_exit: str = DEFAULT_PAPER_TIME_EXIT
    prop_constraints: PropFirmConstraintConfig = build_default_prop_constraints()
    max_heatmap_survivors: int = 3


@dataclass
class VariantEvaluation:
    """Full evaluation payload for one variant under the common leak-free protocol."""

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
    tables_by_scope: dict[str, dict[str, pd.DataFrame]]


@dataclass(frozen=True)
class HeatmapSpec:
    """Small local robustness grid evaluated only for reranking survivors."""

    name: str
    x_label: str
    y_label: str
    x_values: tuple[Any, ...]
    y_values: tuple[Any, ...]
    ref_x: Any
    ref_y: Any
    mutator: Callable[[VWAPVariantConfig, Any, Any], VWAPVariantConfig]


def build_default_reranking_spec(dataset_path: Path | None = None) -> RerankingSpec:
    """Return the default compact reranking universe."""
    resolved = dataset_path or resolve_default_vwap_dataset("MNQ")
    variants = build_default_vwap_reranking_variants()
    return RerankingSpec(
        dataset_path=resolved,
        variant_names=tuple(variant.name for variant in variants),
        prop_constraints=build_default_prop_constraints(),
    )


def _role_for_variant(variant_name: str) -> str:
    if variant_name == PAPER_BASELINE_NAME:
        return "paper_baseline_reference"
    if variant_name == REALISTIC_BASELINE_NAME:
        return "realistic_baseline_reference"
    return "candidate"


def _role_sort_key(role: str) -> int:
    if role == "paper_baseline_reference":
        return 0
    if role == "realistic_baseline_reference":
        return 1
    return 2


def _what_changes_vs_baseline(variant: VWAPVariantConfig) -> str:
    mapping = {
        PAPER_BASELINE_NAME: "Reference officielle: close[t-1] vs VWAP[t-1], next open, always-in-market RTH, flat overnight.",
        REALISTIC_BASELINE_NAME: "Meme signal que la baseline paper, mais sous sizing futures fixe et couts repo_realistic.",
        "vwap_time_filtered_baseline": "Baseline paper gatee par une fenetre horaire simple pour n'autoriser que certaines entrees et flips.",
        "vwap_baseline_trade_capped": "Baseline paper avec plafond dur sur le nombre de flips/trades par jour.",
        "vwap_baseline_regime_filtered": "Baseline paper gatee par un filtre de pente VWAP et un filtre simple de distance au VWAP.",
        "vwap_baseline_with_killswitch": "Baseline paper avec kill switches journaliers simples: pertes max, stop daily, plafond de trades.",
        "vwap_reclaim": "Variante discrete reclaim simple avec pente VWAP, buffer ATR et sortie sur recross/stop/session close.",
        "vwap_reclaim_with_prop_overlay": "Variante reclaim simple avec overlay prop: fenetres horaires, cap de trades, daily stop et deleveraging.",
    }
    return mapping.get(variant.name, variant.notes or "Variant simple derivee de la baseline VWAP leak-free.")


def _safe_float(value: Any, default: float = 0.0) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float(default)
    return float(numeric)


def _format_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "off"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.2f}"
    return str(value)


def _prepare_feature_frame(spec: RerankingSpec) -> pd.DataFrame:
    raw = load_ohlcv_file(spec.dataset_path)
    clean = clean_ohlcv(raw)
    variants = [resolve_vwap_variant(name) for name in spec.variant_names]
    atr_windows = sorted({int(variant.atr_period) for variant in variants if int(variant.atr_period) > 0})
    return prepare_vwap_feature_frame(
        clean,
        session_start=spec.session_start,
        session_end=spec.session_end,
        atr_windows=atr_windows,
    )


def _evaluate_variant(
    feature_df: pd.DataFrame,
    spec: RerankingSpec,
    variant: VWAPVariantConfig,
    cache: dict[str, VariantEvaluation],
) -> VariantEvaluation:
    cache_key = _variant_cache_key(variant)
    if cache_key in cache:
        return cache[cache_key]

    symbol = infer_symbol_from_dataset_path(spec.dataset_path)
    signal_df = build_vwap_signal_frame(feature_df, variant)
    execution_model, instrument = build_execution_model_for_profile(symbol=symbol, profile_name=variant.execution_profile)
    result = run_vwap_backtest(signal_df, variant, execution_model, instrument)
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
        initial_capital=variant.initial_capital_usd,
        constraints=spec.prop_constraints,
        instrument=instrument,
        execution_model=execution_model,
        rolling_window_days=20,
    )
    evaluation = VariantEvaluation(
        variant=variant,
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
        tables_by_scope=tables_by_scope,
    )
    cache[cache_key] = evaluation
    return evaluation


def _scope_row(evaluation: VariantEvaluation, scope: str) -> pd.Series:
    return evaluation.summary_by_scope.loc[evaluation.summary_by_scope["scope"] == scope].iloc[0]


def _variant_catalog_rows(spec: RerankingSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, variant_name in enumerate(spec.variant_names, start=1):
        variant = resolve_vwap_variant(variant_name)
        rows.append(
            {
                "display_order": order,
                "strategy_id": variant.name,
                "role": _role_for_variant(variant.name),
                "family": variant.family,
                "mode": variant.mode,
                "execution_profile": variant.execution_profile,
                "what_changes_vs_baseline": _what_changes_vs_baseline(variant),
                "time_windows": _time_window_label(variant.time_windows),
                "slope_lookback": variant.slope_lookback,
                "slope_threshold": variant.slope_threshold,
                "require_vwap_slope_alignment": variant.require_vwap_slope_alignment,
                "max_vwap_distance_atr": variant.max_vwap_distance_atr,
                "atr_buffer": variant.atr_buffer,
                "stop_buffer": variant.stop_buffer if variant.stop_buffer is not None else variant.atr_buffer,
                "max_trades_per_day": variant.max_trades_per_day,
                "max_losses_per_day": variant.max_losses_per_day,
                "daily_stop_threshold_usd": variant.daily_stop_threshold_usd,
                "consecutive_losses_threshold": variant.consecutive_losses_threshold,
                "exit_on_vwap_recross": variant.exit_on_vwap_recross,
                "notes": variant.notes,
            }
        )
    return rows


def _export_variant_catalog(output_dir: Path, spec: RerankingSpec) -> dict[str, Path]:
    catalog_dir = output_dir / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_df = pd.DataFrame(_variant_catalog_rows(spec))
    catalog_csv = catalog_dir / "variant_catalog.csv"
    catalog_md = catalog_dir / "variant_catalog.md"
    catalog_df.to_csv(catalog_csv, index=False)
    catalog_md.write_text(
        "\n".join(
            [
                "# Variant Catalog",
                "",
                "```text",
                catalog_df[
                    [
                        "display_order",
                        "strategy_id",
                        "role",
                        "mode",
                        "execution_profile",
                        "time_windows",
                        "max_trades_per_day",
                        "daily_stop_threshold_usd",
                        "what_changes_vs_baseline",
                    ]
                ].to_string(index=False),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"variant_catalog_csv": catalog_csv, "variant_catalog_md": catalog_md}


def _stress_oos_summary(evaluation: VariantEvaluation, spec: RerankingSpec) -> pd.Series:
    stressed_trades = _apply_cost_stress_overlay(
        evaluation.trades,
        scenario=DEFAULT_STRESS_SCENARIO,
        instrument=evaluation.instrument,
        execution_model=evaluation.execution_model,
        session_start=spec.session_start,
    )
    stressed_daily = _rebuild_daily_results_from_trades(
        stressed_trades,
        all_sessions=evaluation.all_sessions,
        initial_capital=evaluation.variant.initial_capital_usd,
    )
    stressed_summary, _, _ = _build_scope_summary_table(
        trades=stressed_trades,
        daily_results=stressed_daily,
        bar_results=pd.DataFrame(),
        signal_df=evaluation.signal_df,
        sessions_all=evaluation.all_sessions,
        is_sessions=evaluation.is_sessions,
        oos_sessions=evaluation.oos_sessions,
        initial_capital=evaluation.variant.initial_capital_usd,
        constraints=spec.prop_constraints,
        instrument=evaluation.instrument,
        execution_model=evaluation.execution_model,
        rolling_window_days=20,
    )
    return stressed_summary.loc[stressed_summary["scope"] == "oos"].iloc[0]


def _export_baseline_reference(
    output_dir: Path,
    spec: RerankingSpec,
    baseline_eval: VariantEvaluation,
    baseline_stress_oos: pd.Series,
) -> dict[str, Path]:
    baseline_dir = output_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    overall = _scope_row(baseline_eval, "overall")
    oos = _scope_row(baseline_eval, "oos")
    reference_df = pd.DataFrame(
        [
            {
                "strategy_id": baseline_eval.variant.name,
                "signal_rule": "close[t-1] vs session VWAP[t-1]",
                "execution_rule": "next open leak-free",
                "session": f"{spec.session_start} -> {spec.session_end}",
                "flat_overnight": True,
                "quantity_mode": baseline_eval.variant.quantity_mode,
                "initial_capital_usd": baseline_eval.variant.initial_capital_usd,
                "execution_profile": baseline_eval.variant.execution_profile,
                "overall_total_trades": overall["total_trades"],
                "overall_net_pnl": overall["net_pnl"],
                "overall_profit_factor": overall["profit_factor"],
                "overall_sharpe_ratio": overall["sharpe_ratio"],
                "overall_max_drawdown": overall["max_drawdown"],
                "overall_worst_daily_loss_usd": overall["worst_daily_loss_usd"],
                "overall_daily_loss_limit_breach_freq": overall["daily_loss_limit_breach_freq"],
                "overall_trailing_drawdown_breach_freq": overall["trailing_drawdown_breach_freq"],
                "oos_total_trades": oos["total_trades"],
                "oos_net_pnl": oos["net_pnl"],
                "oos_profit_factor": oos["profit_factor"],
                "oos_sharpe_ratio": oos["sharpe_ratio"],
                "oos_max_drawdown": oos["max_drawdown"],
                "oos_worst_daily_loss_usd": oos["worst_daily_loss_usd"],
                "oos_daily_loss_limit_breach_freq": oos["daily_loss_limit_breach_freq"],
                "oos_trailing_drawdown_breach_freq": oos["trailing_drawdown_breach_freq"],
                "oos_net_pnl_slippage_x2": baseline_stress_oos["net_pnl"],
                "oos_profit_factor_slippage_x2": baseline_stress_oos["profit_factor"],
                "oos_sharpe_ratio_slippage_x2": baseline_stress_oos["sharpe_ratio"],
            }
        ]
    )
    reference_csv = baseline_dir / "paper_baseline_reference.csv"
    reference_md = baseline_dir / "paper_baseline_reference.md"
    reference_df.to_csv(reference_csv, index=False)
    reference_md.write_text(
        "\n".join(
            [
                "# Paper Baseline Reference",
                "",
                "- Official reference baseline is rerun under the current leak-free semantics.",
                "- Signal: previous close vs previous session VWAP.",
                "- Execution: next open, always-in-market during RTH, flat overnight.",
                f"- Session: `{spec.session_start}` -> `{spec.session_end}`.",
                f"- Costs/profile: `{baseline_eval.variant.execution_profile}`.",
                "",
                "```text",
                reference_df.to_string(index=False),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"paper_baseline_reference_csv": reference_csv, "paper_baseline_reference_md": reference_md}


def _split_oos_rows(evaluation: VariantEvaluation, spec: RerankingSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for split_fraction in spec.split_fractions:
        is_sessions, oos_sessions = _split_sessions(evaluation.all_sessions, split_fraction)
        split_summary, _, _ = _build_scope_summary_table(
            trades=evaluation.trades,
            daily_results=evaluation.daily_results,
            bar_results=evaluation.bar_results,
            signal_df=evaluation.signal_df,
            sessions_all=evaluation.all_sessions,
            is_sessions=is_sessions,
            oos_sessions=oos_sessions,
            initial_capital=evaluation.variant.initial_capital_usd,
            constraints=spec.prop_constraints,
            instrument=evaluation.instrument,
            execution_model=evaluation.execution_model,
            rolling_window_days=20,
        )
        oos_row = split_summary.loc[split_summary["scope"] == "oos"].iloc[0]
        rows.append(
            {
                "strategy_id": evaluation.variant.name,
                "split_name": f"is_{int(split_fraction * 100)}_oos_{int((1.0 - split_fraction) * 100)}",
                "is_fraction": split_fraction,
                "oos_start_date": str(oos_sessions[0]) if oos_sessions else None,
                "oos_net_pnl": oos_row["net_pnl"],
                "oos_profit_factor": oos_row["profit_factor"],
                "oos_sharpe_ratio": oos_row["sharpe_ratio"],
                "oos_max_drawdown": oos_row["max_drawdown"],
                "oos_expectancy_per_trade": oos_row["expectancy_per_trade"],
                "oos_total_trades": oos_row["total_trades"],
            }
        )
    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        aggregate = pd.DataFrame(
            columns=[
                "strategy_id",
                "positive_oos_splits",
                "total_splits",
                "mean_oos_net_pnl",
                "mean_oos_profit_factor",
                "mean_oos_sharpe_ratio",
                "worst_oos_split_net_pnl",
                "best_oos_split_net_pnl",
                "pass_fail_splits",
            ]
        )
        return detail_df, aggregate

    aggregate = (
        detail_df.groupby("strategy_id", as_index=False)
        .agg(
            positive_oos_splits=("oos_net_pnl", lambda values: int((pd.Series(values) > 0).sum())),
            total_splits=("split_name", "count"),
            mean_oos_net_pnl=("oos_net_pnl", "mean"),
            mean_oos_profit_factor=("oos_profit_factor", "mean"),
            mean_oos_sharpe_ratio=("oos_sharpe_ratio", "mean"),
            worst_oos_split_net_pnl=("oos_net_pnl", "min"),
            best_oos_split_net_pnl=("oos_net_pnl", "max"),
        )
        .reset_index(drop=True)
    )
    majority_threshold = np.ceil(aggregate["total_splits"] / 2.0)
    aggregate["pass_fail_splits"] = aggregate["positive_oos_splits"] >= majority_threshold
    return detail_df, aggregate


def _standard_challenge_row(evaluation: VariantEvaluation) -> dict[str, Any]:
    scenario = next(s for s in _challenge_scenarios() if s.name == "scenario_b_standard")
    oos_trades = _subset_frame_by_sessions(evaluation.trades, evaluation.oos_sessions)
    empirical_summary, _ = _challenge_empirical_summary(
        oos_trades,
        scenario=scenario,
        account_size_usd=evaluation.variant.initial_capital_usd,
    )
    return empirical_summary


def _prop_verdict(
    oos_row: pd.Series,
    challenge_success_rate: float,
    constraints: PropFirmConstraintConfig,
) -> str:
    oos_net_pnl = _safe_float(oos_row.get("net_pnl"))
    oos_pf = _safe_float(oos_row.get("profit_factor"))
    worst_day = _safe_float(oos_row.get("worst_daily_loss_usd"))
    daily_breach = _safe_float(oos_row.get("daily_loss_limit_breach_freq"))
    trailing_breach = _safe_float(oos_row.get("trailing_drawdown_breach_freq"))
    max_dd = abs(_safe_float(oos_row.get("max_drawdown")))

    if oos_net_pnl <= 0 or oos_pf <= 1.0:
        return "non defendable"
    if (
        worst_day <= -float(constraints.daily_loss_limit_usd)
        or daily_breach > 0.03
        or trailing_breach > 0.10
        or max_dd > float(constraints.trailing_drawdown_limit_usd) * 2.0
    ):
        return "trop fragile"
    if challenge_success_rate >= 0.20 and daily_breach <= 0.01 and trailing_breach <= 0.05:
        return "prop-compatible"
    return "potentiellement compatible sous contraintes prudentes"


def _build_stress_summary(
    evaluations: dict[str, VariantEvaluation],
    spec: RerankingSpec,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    rows: list[dict[str, Any]] = []
    stress_rows: dict[str, pd.Series] = {}
    for strategy_id in spec.variant_names:
        evaluation = evaluations[strategy_id]
        nominal_oos = _scope_row(evaluation, "oos")
        stress_oos = _stress_oos_summary(evaluation, spec)
        stress_rows[strategy_id] = stress_oos
        rows.append(
            {
                "strategy_id": strategy_id,
                "role": _role_for_variant(strategy_id),
                "pnl_nominal": nominal_oos["net_pnl"],
                "pnl_slip_x2": stress_oos["net_pnl"],
                "pf_nominal": nominal_oos["profit_factor"],
                "pf_slip_x2": stress_oos["profit_factor"],
                "sharpe_nominal": nominal_oos["sharpe_ratio"],
                "sharpe_slip_x2": stress_oos["sharpe_ratio"],
                "dd_nominal": nominal_oos["max_drawdown"],
                "dd_slip_x2": stress_oos["max_drawdown"],
                "delta_pnl_nominal_vs_slip_x2": stress_oos["net_pnl"] - nominal_oos["net_pnl"],
                "pass_fail_cost_stress": bool(stress_oos["net_pnl"] > 0 and stress_oos["profit_factor"] > 1.0),
            }
        )
    stress_df = pd.DataFrame(rows)
    stress_df = stress_df.sort_values(["role", "strategy_id"], key=lambda series: series.map(_role_sort_key) if series.name == "role" else series).reset_index(drop=True)
    return stress_df, stress_rows


def _build_prop_summary(
    evaluations: dict[str, VariantEvaluation],
    spec: RerankingSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy_id in spec.variant_names:
        evaluation = evaluations[strategy_id]
        oos = _scope_row(evaluation, "oos")
        challenge_row = _standard_challenge_row(evaluation)
        challenge_success = _safe_float(challenge_row.get("success_rate_empirical"))
        oos_daily = _subset_frame_by_sessions(evaluation.daily_results, evaluation.oos_sessions)
        red_days_freq = 1.0 - _safe_float(oos_daily.get("green_day", pd.Series(dtype=float)).mean(), default=0.0)
        rows.append(
            {
                "strategy_id": strategy_id,
                "role": _role_for_variant(strategy_id),
                "worst_daily_loss_usd": oos["worst_daily_loss_usd"],
                "red_days_freq": red_days_freq,
                "days_below_neg_0p25r_freq": oos["days_below_neg_0p25r_freq"],
                "days_below_neg_0p5r_freq": oos["days_below_neg_0p5r_freq"],
                "days_below_neg_1r_freq": oos["days_below_neg_1r_freq"],
                "worst_losing_trades_streak": oos["worst_losing_trades_streak"],
                "worst_losing_days_streak": oos["worst_losing_days_streak"],
                "max_drawdown": oos["max_drawdown"],
                "daily_loss_limit_breach_freq": oos["daily_loss_limit_breach_freq"],
                "trailing_drawdown_breach_freq": oos["trailing_drawdown_breach_freq"],
                "avg_trades_per_day": oos["mean_trades_per_day"],
                "max_trades_per_day": oos["max_trades_per_day"],
                "challenge_success_rate_standard": challenge_success,
                "challenge_bust_rate_standard": _safe_float(challenge_row.get("bust_rate_empirical")),
                "prop_verdict": _prop_verdict(oos, challenge_success_rate=challenge_success, constraints=spec.prop_constraints),
            }
        )
    prop_df = pd.DataFrame(rows)
    return prop_df.sort_values(["role", "strategy_id"], key=lambda series: series.map(_role_sort_key) if series.name == "role" else series).reset_index(drop=True)


def _build_concentration_summary(evaluations: dict[str, VariantEvaluation], spec: RerankingSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy_id in spec.variant_names:
        evaluation = evaluations[strategy_id]
        oos = _scope_row(evaluation, "oos")
        rows.append(
            {
                "strategy_id": strategy_id,
                "role": _role_for_variant(strategy_id),
                "top_3_day_contribution_pct": oos["top_3_day_contribution_pct"],
                "top_5_day_contribution_pct": oos["top_5_day_contribution_pct"],
                "pnl_excluding_top_3_days": oos["pnl_excluding_top_3_days"],
                "pnl_excluding_top_5_days": oos["pnl_excluding_top_5_days"],
                "pnl_excluding_best_month": oos["pnl_excluding_best_month"],
                "concentration_verdict": _outlier_verdict(oos),
            }
        )
    concentration_df = pd.DataFrame(rows)
    return concentration_df.sort_values(["role", "strategy_id"], key=lambda series: series.map(_role_sort_key) if series.name == "role" else series).reset_index(drop=True)


def _merge_reranking_tables(
    spec: RerankingSpec,
    evaluations: dict[str, VariantEvaluation],
    stress_df: pd.DataFrame,
    split_aggregate_df: pd.DataFrame,
    prop_df: pd.DataFrame,
    concentration_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_map = split_aggregate_df.set_index("strategy_id").to_dict(orient="index") if not split_aggregate_df.empty else {}
    stress_map = stress_df.set_index("strategy_id").to_dict(orient="index") if not stress_df.empty else {}
    prop_map = prop_df.set_index("strategy_id").to_dict(orient="index") if not prop_df.empty else {}
    concentration_map = concentration_df.set_index("strategy_id").to_dict(orient="index") if not concentration_df.empty else {}

    for strategy_id in spec.variant_names:
        evaluation = evaluations[strategy_id]
        variant = evaluation.variant
        role = _role_for_variant(strategy_id)
        oos = _scope_row(evaluation, "oos")
        overall = _scope_row(evaluation, "overall")
        split_row = split_map.get(strategy_id, {})
        stress_row = stress_map.get(strategy_id, {})
        prop_row = prop_map.get(strategy_id, {})
        concentration_row = concentration_map.get(strategy_id, {})
        pass_oos_nominal = bool(oos["net_pnl"] > 0 and oos["profit_factor"] > 1.0)
        pass_cost_stress = bool(stress_row.get("pass_fail_cost_stress", False))
        total_splits = int(split_row.get("total_splits", 0))
        positive_splits = int(split_row.get("positive_oos_splits", 0))
        pass_splits = bool(total_splits > 0 and positive_splits >= math.ceil(total_splits / 2.0))
        prop_verdict = str(prop_row.get("prop_verdict", "non defendable"))
        pass_prop = prop_verdict in {"prop-compatible", "potentiellement compatible sous contraintes prudentes"}
        concentration_verdict = str(concentration_row.get("concentration_verdict", "forte dependance aux meilleurs jours"))
        survives_primary_filter = role == "candidate" and pass_oos_nominal and pass_cost_stress and pass_splits and pass_prop

        if role == "paper_baseline_reference":
            final_bucket = "reference_officielle"
            elimination_reason = "Ancre officielle de comparaison, non candidate au reranking."
        elif role == "realistic_baseline_reference":
            final_bucket = "baseline_realiste_de_reference"
            elimination_reason = "Baseline economique de reference pour juger si une variante apporte vraiment mieux."
        elif not pass_oos_nominal or not pass_cost_stress:
            final_bucket = "eliminee immediatement"
            elimination_reason = "Ne survit pas au filtre nominal OOS ou au stress slippage x2."
        elif survives_primary_filter:
            final_bucket = "survivante"
            elimination_reason = "Passe les filtres nominaux, couts, splits et prop."
        else:
            final_bucket = "interessante mais trop fragile"
            elimination_reason = "Garde un signal partiel mais echoue sur les splits ou les contraintes prop."

        rows.append(
            {
                "strategy_id": strategy_id,
                "role": role,
                "family": variant.family,
                "mode": variant.mode,
                "execution_profile": variant.execution_profile,
                "what_changes_vs_baseline": _what_changes_vs_baseline(variant),
                "overall_net_pnl": overall["net_pnl"],
                "oos_net_pnl": oos["net_pnl"],
                "oos_profit_factor": oos["profit_factor"],
                "oos_sharpe_ratio": oos["sharpe_ratio"],
                "oos_max_drawdown": oos["max_drawdown"],
                "oos_expectancy_per_trade": oos["expectancy_per_trade"],
                "oos_total_trades": oos["total_trades"],
                "pnl_nominal": stress_row.get("pnl_nominal", oos["net_pnl"]),
                "pnl_slip_x2": stress_row.get("pnl_slip_x2", np.nan),
                "pf_nominal": stress_row.get("pf_nominal", oos["profit_factor"]),
                "pf_slip_x2": stress_row.get("pf_slip_x2", np.nan),
                "sharpe_nominal": stress_row.get("sharpe_nominal", oos["sharpe_ratio"]),
                "sharpe_slip_x2": stress_row.get("sharpe_slip_x2", np.nan),
                "dd_nominal": stress_row.get("dd_nominal", oos["max_drawdown"]),
                "dd_slip_x2": stress_row.get("dd_slip_x2", np.nan),
                "delta_pnl_nominal_vs_slip_x2": stress_row.get("delta_pnl_nominal_vs_slip_x2", np.nan),
                "pass_fail_cost_stress": pass_cost_stress,
                "positive_oos_splits": positive_splits,
                "total_splits": total_splits,
                "mean_oos_net_pnl_splits": split_row.get("mean_oos_net_pnl", np.nan),
                "mean_oos_profit_factor_splits": split_row.get("mean_oos_profit_factor", np.nan),
                "mean_oos_sharpe_ratio_splits": split_row.get("mean_oos_sharpe_ratio", np.nan),
                "worst_oos_split_net_pnl": split_row.get("worst_oos_split_net_pnl", np.nan),
                "best_oos_split_net_pnl": split_row.get("best_oos_split_net_pnl", np.nan),
                "pass_fail_splits": pass_splits,
                "worst_daily_loss_usd": prop_row.get("worst_daily_loss_usd", oos["worst_daily_loss_usd"]),
                "daily_loss_limit_breach_freq": prop_row.get("daily_loss_limit_breach_freq", oos["daily_loss_limit_breach_freq"]),
                "trailing_drawdown_breach_freq": prop_row.get("trailing_drawdown_breach_freq", oos["trailing_drawdown_breach_freq"]),
                "avg_trades_per_day": prop_row.get("avg_trades_per_day", oos["mean_trades_per_day"]),
                "max_trades_per_day": prop_row.get("max_trades_per_day", oos["max_trades_per_day"]),
                "challenge_success_rate_standard": prop_row.get("challenge_success_rate_standard", np.nan),
                "prop_verdict": prop_verdict,
                "top_3_day_contribution_pct": concentration_row.get("top_3_day_contribution_pct", oos["top_3_day_contribution_pct"]),
                "top_5_day_contribution_pct": concentration_row.get("top_5_day_contribution_pct", oos["top_5_day_contribution_pct"]),
                "pnl_excluding_top_3_days": concentration_row.get("pnl_excluding_top_3_days", oos["pnl_excluding_top_3_days"]),
                "pnl_excluding_top_5_days": concentration_row.get("pnl_excluding_top_5_days", oos["pnl_excluding_top_5_days"]),
                "concentration_verdict": concentration_verdict,
                "pass_oos_nominal": pass_oos_nominal,
                "pass_cost_stress": pass_cost_stress,
                "pass_splits": pass_splits,
                "pass_prop": pass_prop,
                "survives_primary_filter": survives_primary_filter,
                "reranking_score": int(pass_oos_nominal) + int(pass_cost_stress) + int(pass_splits) + int(pass_prop),
                "final_bucket": final_bucket,
                "elimination_reason": elimination_reason,
            }
        )

    summary_df = pd.DataFrame(rows)
    survivor_mask = summary_df["survives_primary_filter"].fillna(False)
    survivors = summary_df.loc[survivor_mask].copy()
    if not survivors.empty:
        survivors = survivors.sort_values(
            [
                "oos_profit_factor",
                "oos_sharpe_ratio",
                "mean_oos_sharpe_ratio_splits",
                "pnl_excluding_top_3_days",
                "oos_max_drawdown",
            ],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
        survivors["rank_within_survivors"] = np.arange(1, len(survivors) + 1)
        top_strategy = survivors.iloc[0]["strategy_id"]
        summary_df = summary_df.merge(survivors[["strategy_id", "rank_within_survivors"]], on="strategy_id", how="left")
        summary_df.loc[summary_df["strategy_id"] == top_strategy, "final_bucket"] = "candidat prioritaire pour validation complementaire"
        summary_df.loc[
            summary_df["survives_primary_filter"] & (summary_df["strategy_id"] != top_strategy),
            "final_bucket",
        ] = "candidat secondaire a surveiller"
    else:
        summary_df["rank_within_survivors"] = np.nan

    summary_df = summary_df.sort_values(
        ["role", "rank_within_survivors", "strategy_id"],
        key=lambda series: series.map(_role_sort_key) if series.name == "role" else series,
        na_position="last",
    ).reset_index(drop=True)
    return summary_df


def _time_window_with_morning_end(variant: VWAPVariantConfig, morning_end: str) -> tuple[TimeWindow, ...]:
    if not variant.time_windows:
        return (TimeWindow("09:35:00", morning_end),)
    windows = list(variant.time_windows)
    first = windows[0]
    windows[0] = TimeWindow(first.start, morning_end)
    return tuple(windows)


def _daily_stop_value(value: Any) -> float | None:
    if value in {None, "off"}:
        return None
    return float(value)


def _heatmap_specs_for_variant(variant: VWAPVariantConfig) -> list[HeatmapSpec]:
    if variant.name == "vwap_baseline_regime_filtered":
        return [
            HeatmapSpec(
                name=f"{variant.name}__slope_x_distance",
                x_label="Slope Threshold",
                y_label="Max Distance (ATR)",
                x_values=("0.00", "0.01", "0.02"),
                y_values=("0.75", "1.00", "1.25"),
                ref_x=f"{variant.slope_threshold:.2f}",
                ref_y=f"{float(variant.max_vwap_distance_atr or 1.0):.2f}",
                mutator=lambda base, x, y: replace(
                    base,
                    require_vwap_slope_alignment=True,
                    slope_threshold=float(x),
                    max_vwap_distance_atr=float(y),
                ),
            )
        ]
    if variant.name == "vwap_time_filtered_baseline":
        return [
            HeatmapSpec(
                name=f"{variant.name}__window_x_max_trades",
                x_label="Morning Window End",
                y_label="Max Trades/Day",
                x_values=("11:00:00", "11:30:00", "12:00:00"),
                y_values=("off", "4", "6"),
                ref_x=variant.time_windows[0].end if variant.time_windows else "11:30:00",
                ref_y="off",
                mutator=lambda base, x, y: replace(
                    base,
                    time_windows=_time_window_with_morning_end(base, str(x)),
                    max_trades_per_day=None if y == "off" else int(y),
                ),
            )
        ]
    if variant.name == "vwap_baseline_trade_capped":
        return [
            HeatmapSpec(
                name=f"{variant.name}__trades_x_daily_stop",
                x_label="Max Trades/Day",
                y_label="Daily Stop (USD)",
                x_values=("4", "6", "8"),
                y_values=("off", "500", "750"),
                ref_x=str(int(variant.max_trades_per_day or 6)),
                ref_y="off",
                mutator=lambda base, x, y: replace(
                    base,
                    max_trades_per_day=int(x),
                    daily_stop_threshold_usd=_daily_stop_value(y),
                ),
            )
        ]
    if variant.name == "vwap_baseline_with_killswitch":
        return [
            HeatmapSpec(
                name=f"{variant.name}__trades_x_daily_stop",
                x_label="Max Trades/Day",
                y_label="Daily Stop (USD)",
                x_values=("8", "12", "16"),
                y_values=("500", "750", "1000"),
                ref_x=str(int(variant.max_trades_per_day or 12)),
                ref_y=_format_value(variant.daily_stop_threshold_usd),
                mutator=lambda base, x, y: replace(
                    base,
                    max_trades_per_day=int(x),
                    daily_stop_threshold_usd=float(y),
                ),
            )
        ]
    if variant.name in {"vwap_reclaim", "vwap_reclaim_with_prop_overlay"}:
        return [
            HeatmapSpec(
                name=f"{variant.name}__slope_x_atr_buffer",
                x_label="Slope Threshold",
                y_label="ATR Buffer",
                x_values=("0.00", "0.01", "0.02"),
                y_values=("0.20", "0.25", "0.30"),
                ref_x=f"{variant.slope_threshold:.2f}",
                ref_y=f"{variant.atr_buffer:.2f}",
                mutator=lambda base, x, y: replace(
                    base,
                    slope_threshold=float(x),
                    atr_buffer=float(y),
                    stop_buffer=float(y),
                ),
            )
        ]
    return []


def _plot_single_heatmap(
    ax: plt.Axes,
    grid_df: pd.DataFrame,
    value_col: str,
    x_order: list[str],
    y_order: list[str],
    x_label: str,
    y_label: str,
    title: str,
    ref_x: str,
    ref_y: str,
) -> None:
    pivot = (
        grid_df.pivot(index="y_value", columns="x_value", values=value_col)
        .reindex(index=y_order, columns=x_order)
        .astype(float)
    )
    values = pivot.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if not finite_values.size:
        values = np.zeros_like(values, dtype=float)
        finite_values = np.array([0.0])

    if value_col in {"oos_net_pnl", "oos_sharpe_ratio"}:
        vmax = float(np.nanmax(np.abs(finite_values)))
        vmax = max(vmax, 1e-9)
        norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        cmap = "coolwarm"
    elif value_col == "oos_max_drawdown":
        vmax = float(np.nanmax(np.abs(finite_values)))
        vmax = max(vmax, 1e-9)
        norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        cmap = "coolwarm_r"
    else:
        norm = None
        cmap = "cividis"

    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(x_order)))
    ax.set_xticklabels(x_order, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(y_order)))
    ax.set_yticklabels(y_order)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    for row_idx in range(len(y_order)):
        for col_idx in range(len(x_order)):
            value = values[row_idx, col_idx]
            label = "na" if not math.isfinite(float(value)) else f"{value:.2f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8, color="black")

    if ref_x in x_order and ref_y in y_order:
        ref_col = x_order.index(ref_x)
        ref_row = y_order.index(ref_y)
        rect = plt.Rectangle((ref_col - 0.5, ref_row - 0.5), 1.0, 1.0, fill=False, edgecolor="#111827", linewidth=2.5)
        ax.add_patch(rect)

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _render_heatmap_figure(
    heatmap_df: pd.DataFrame,
    spec: HeatmapSpec,
    output_path: Path,
    strategy_id: str,
) -> None:
    x_order = [_format_value(value) for value in spec.x_values]
    y_order = [_format_value(value) for value in spec.y_values]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metric_specs = (
        ("oos_sharpe_ratio", "OOS Sharpe"),
        ("oos_profit_factor", "OOS PF"),
        ("oos_net_pnl", "OOS Net PnL"),
        ("oos_max_drawdown", "OOS Max DD"),
    )
    for ax, (column, title) in zip(axes.flatten(), metric_specs):
        _plot_single_heatmap(
            ax=ax,
            grid_df=heatmap_df,
            value_col=column,
            x_order=x_order,
            y_order=y_order,
            x_label=spec.x_label,
            y_label=spec.y_label,
            title=title,
            ref_x=_format_value(spec.ref_x),
            ref_y=_format_value(spec.ref_y),
        )
    fig.suptitle(f"{strategy_id} - {spec.name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _run_survivor_heatmaps(
    output_dir: Path,
    spec: RerankingSpec,
    feature_df: pd.DataFrame,
    evaluations: dict[str, VariantEvaluation],
    reranking_df: pd.DataFrame,
    cache: dict[str, VariantEvaluation],
) -> dict[str, Path]:
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    survivors = reranking_df.loc[
        (reranking_df["role"] == "candidate") & (reranking_df["survives_primary_filter"] == True)  # noqa: E712
    ].copy()
    survivors = survivors.sort_values("rank_within_survivors").head(spec.max_heatmap_survivors)

    rows: list[dict[str, Any]] = []
    readouts: list[dict[str, Any]] = []

    for _, survivor in survivors.iterrows():
        strategy_id = str(survivor["strategy_id"])
        variant = evaluations[strategy_id].variant
        for heatmap_spec in _heatmap_specs_for_variant(variant):
            local_rows: list[dict[str, Any]] = []
            for x_value in heatmap_spec.x_values:
                for y_value in heatmap_spec.y_values:
                    local_variant = heatmap_spec.mutator(variant, x_value, y_value)
                    local_eval = _evaluate_variant(feature_df, spec, local_variant, cache)
                    oos = _scope_row(local_eval, "oos")
                    local_rows.append(
                        {
                            "strategy_id": strategy_id,
                            "heatmap_name": heatmap_spec.name,
                            "x_value": _format_value(x_value),
                            "y_value": _format_value(y_value),
                            "x_value_sort": str(x_value),
                            "y_value_sort": str(y_value),
                            "oos_net_pnl": oos["net_pnl"],
                            "oos_profit_factor": oos["profit_factor"],
                            "oos_sharpe_ratio": oos["sharpe_ratio"],
                            "oos_max_drawdown": oos["max_drawdown"],
                            "oos_total_trades": oos["total_trades"],
                            "is_reference_cell": _format_value(x_value) == _format_value(heatmap_spec.ref_x)
                            and _format_value(y_value) == _format_value(heatmap_spec.ref_y),
                        }
                    )
            local_df = pd.DataFrame(local_rows)
            rows.extend(local_rows)
            output_path = heatmap_dir / f"{heatmap_spec.name}.png"
            _render_heatmap_figure(local_df, heatmap_spec, output_path, strategy_id=strategy_id)
            readout = _heatmap_topology_readout(
                local_df,
                x_col="x_value",
                y_col="y_value",
                ref_x=_format_value(heatmap_spec.ref_x),
                ref_y=_format_value(heatmap_spec.ref_y),
            )
            readout["strategy_id"] = strategy_id
            readout["heatmap_name"] = heatmap_spec.name
            readout["chart_path"] = output_path
            readouts.append(readout)

    heatmap_csv = output_dir / "survivors_heatmaps.csv"
    readout_csv = output_dir / "survivors_heatmap_readouts.csv"
    pd.DataFrame(
        rows,
        columns=[
            "strategy_id",
            "heatmap_name",
            "x_value",
            "y_value",
            "x_value_sort",
            "y_value_sort",
            "oos_net_pnl",
            "oos_profit_factor",
            "oos_sharpe_ratio",
            "oos_max_drawdown",
            "oos_total_trades",
            "is_reference_cell",
        ],
    ).to_csv(heatmap_csv, index=False)
    pd.DataFrame(
        readouts,
        columns=[
            "verdict",
            "stable_neighbor_share",
            "reference_sharpe_rank_pct",
            "comment",
            "strategy_id",
            "heatmap_name",
            "chart_path",
        ],
    ).to_csv(readout_csv, index=False)
    return {"survivors_heatmaps_csv": heatmap_csv, "survivors_heatmap_readouts_csv": readout_csv}


def _final_verdict_payload(reranking_df: pd.DataFrame) -> dict[str, Any]:
    paper_row = reranking_df.loc[reranking_df["strategy_id"] == PAPER_BASELINE_NAME].iloc[0]
    realistic_row = reranking_df.loc[reranking_df["strategy_id"] == REALISTIC_BASELINE_NAME].iloc[0]
    candidates = reranking_df.loc[reranking_df["role"] == "candidate"].copy()
    survivors = candidates.loc[candidates["survives_primary_filter"] == True].copy()  # noqa: E712
    top_candidate = survivors.sort_values("rank_within_survivors").iloc[0] if not survivors.empty else None

    realistic_baseline_positive = bool(realistic_row["oos_net_pnl"] > 0 and realistic_row["pf_slip_x2"] > 1.0)
    beats_realistic_baseline = False
    if top_candidate is not None:
        beats_realistic_baseline = bool(
            top_candidate["oos_profit_factor"] > realistic_row["oos_profit_factor"]
            and top_candidate["oos_sharpe_ratio"] > realistic_row["oos_sharpe_ratio"]
            and top_candidate["pnl_slip_x2"] > realistic_row["pnl_slip_x2"]
        )

    if top_candidate is None:
        global_verdict = "Aucune variante n'est assez robuste pour meriter une validation approfondie supplementaire."
        recommended_action = "abandonner cette famille de variantes pour le moment"
    else:
        global_verdict = f"La meilleure survivante est {top_candidate['strategy_id']}, mais elle reste a confirmer par une validation plus profonde."
        recommended_action = f"poursuivre avec {top_candidate['strategy_id']} comme candidat prioritaire"

    return {
        "paper_baseline_reference_strategy_id": PAPER_BASELINE_NAME,
        "realistic_baseline_strategy_id": REALISTIC_BASELINE_NAME,
        "paper_baseline_oos_net_pnl": _safe_float(paper_row["oos_net_pnl"]),
        "paper_baseline_oos_profit_factor": _safe_float(paper_row["oos_profit_factor"]),
        "realistic_baseline_oos_net_pnl": _safe_float(realistic_row["oos_net_pnl"]),
        "realistic_baseline_oos_profit_factor": _safe_float(realistic_row["oos_profit_factor"]),
        "survivor_count": int(len(survivors)),
        "top_candidate": None if top_candidate is None else str(top_candidate["strategy_id"]),
        "global_verdict": global_verdict,
        "variant_verdicts": reranking_df[
            ["strategy_id", "role", "final_bucket", "elimination_reason", "prop_verdict", "concentration_verdict"]
        ].to_dict(orient="records"),
        "answers": {
            "paper_baseline_confirms_exploitable_under_costs": realistic_baseline_positive,
            "any_variant_robustly_beats_realistic_baseline": beats_realistic_baseline,
            "at_least_one_candidate_deserves_deeper_validation": bool(top_candidate is not None),
            "recommended_next_action": recommended_action,
        },
    }


def _write_summary_outputs(
    output_dir: Path,
    reranking_df: pd.DataFrame,
    verdict: dict[str, Any],
) -> dict[str, Path]:
    summary_csv = output_dir / "reranking_summary.csv"
    summary_md = output_dir / "reranking_summary.md"
    verdict_json = output_dir / "final_verdict.json"
    reranking_df.to_csv(summary_csv, index=False)
    _json_dump(verdict_json, verdict)

    survivors = reranking_df.loc[reranking_df["survives_primary_filter"] == True, ["strategy_id", "rank_within_survivors"]]  # noqa: E712
    eliminated = reranking_df.loc[reranking_df["final_bucket"] == "eliminee immediatement", "strategy_id"].tolist()
    summary_md.write_text(
        "\n".join(
            [
                "# VWAP Reranking Summary",
                "",
                f"- Global verdict: `{verdict['global_verdict']}`",
                f"- Survivors after primary filter: `{', '.join(survivors.sort_values('rank_within_survivors')['strategy_id'].tolist()) if not survivors.empty else 'none'}`",
                f"- Eliminated immediately: `{', '.join(eliminated) if eliminated else 'none'}`",
                "",
                "```text",
                reranking_df[
                    [
                        "strategy_id",
                        "role",
                        "oos_net_pnl",
                        "oos_profit_factor",
                        "oos_sharpe_ratio",
                        "pnl_slip_x2",
                        "positive_oos_splits",
                        "total_splits",
                        "prop_verdict",
                        "concentration_verdict",
                        "final_bucket",
                    ]
                ].to_string(index=False),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "reranking_summary_csv": summary_csv,
        "reranking_summary_md": summary_md,
        "final_verdict_json": verdict_json,
    }


def generate_reranking_notebook(notebook_path: Path, output_dir: Path) -> Path:
    """Render the decision notebook for the reranking campaign."""
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
    baseline_code = """display(Markdown((OUTPUT_DIR / "baseline" / "paper_baseline_reference.md").read_text(encoding="utf-8")))
"""
    all_variants_code = """display(pd.read_csv(OUTPUT_DIR / "catalog" / "variant_catalog.csv"))
display(pd.read_csv(OUTPUT_DIR / "reranking_summary.csv"))
"""
    stress_code = """display(pd.read_csv(OUTPUT_DIR / "stress_test_summary.csv"))
"""
    split_code = """display(pd.read_csv(OUTPUT_DIR / "split_summary.csv"))
"""
    prop_code = """display(pd.read_csv(OUTPUT_DIR / "prop_summary.csv"))
"""
    concentration_code = """display(pd.read_csv(OUTPUT_DIR / "concentration_summary.csv"))
"""
    heatmap_code = """path = OUTPUT_DIR / "survivors_heatmap_readouts.csv"
if path.exists():
    display(pd.read_csv(path))
for png in sorted((OUTPUT_DIR / "heatmaps").glob("*.png")):
    display(Image(filename=str(png)))
"""
    verdict_code = """verdict = json.loads((OUTPUT_DIR / "final_verdict.json").read_text(encoding="utf-8"))
verdict
display(Markdown((OUTPUT_DIR / "reranking_summary.md").read_text(encoding="utf-8")))
"""
    notebook = {
        "cells": [
            _notebook_cell("markdown", "# VWAP Leak-Free Reranking Notebook"),
            _notebook_cell("code", setup_code),
            _notebook_cell("code", config_code),
            _notebook_cell("markdown", "## 1) Paper Baseline Reference"),
            _notebook_cell("code", baseline_code),
            _notebook_cell("markdown", "## 2) Full Variant Comparison"),
            _notebook_cell("code", all_variants_code),
            _notebook_cell("markdown", "## 3) Stress x2"),
            _notebook_cell("code", stress_code),
            _notebook_cell("markdown", "## 4) Multi-Split"),
            _notebook_cell("code", split_code),
            _notebook_cell("markdown", "## 5) Prop Metrics"),
            _notebook_cell("code", prop_code),
            _notebook_cell("markdown", "## 6) Concentration"),
            _notebook_cell("code", concentration_code),
            _notebook_cell("markdown", "## 7) Survivors Heatmaps"),
            _notebook_cell("code", heatmap_code),
            _notebook_cell("markdown", "## 8) Final Verdict"),
            _notebook_cell("code", verdict_code),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return notebook_path


def run_vwap_reranking_campaign(
    spec: RerankingSpec,
    output_dir: Path,
    mode: str = "full",
    notebook_path: Path | None = None,
) -> dict[str, Path]:
    """Execute the compact reranking campaign."""
    if mode not in RERANKING_MODES:
        raise ValueError(f"Unsupported reranking mode '{mode}'.")

    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {"output_dir": output_dir}

    artifacts.update(_export_variant_catalog(output_dir, spec))
    feature_df = _prepare_feature_frame(spec)
    cache: dict[str, VariantEvaluation] = {}

    if mode == "baseline":
        baseline_eval = _evaluate_variant(feature_df, spec, resolve_vwap_variant(PAPER_BASELINE_NAME), cache)
        baseline_stress_oos = _stress_oos_summary(baseline_eval, spec)
        artifacts.update(_export_baseline_reference(output_dir, spec, baseline_eval, baseline_stress_oos))
        _json_dump(
            output_dir / "run_metadata.json",
            {"run_timestamp": datetime.now().isoformat(), "mode": mode, "dataset_path": spec.dataset_path},
        )
        return artifacts

    evaluations = {
        strategy_id: _evaluate_variant(feature_df, spec, resolve_vwap_variant(strategy_id), cache)
        for strategy_id in spec.variant_names
    }

    baseline_eval = evaluations[PAPER_BASELINE_NAME]
    stress_df, stress_map = _build_stress_summary(evaluations, spec)
    artifacts.update(_export_baseline_reference(output_dir, spec, baseline_eval, stress_map[PAPER_BASELINE_NAME]))

    if mode in {"stress", "full", "notebook"}:
        stress_df.to_csv(output_dir / "stress_test_summary.csv", index=False)
        artifacts["stress_test_summary_csv"] = output_dir / "stress_test_summary.csv"

    split_detail_rows: list[pd.DataFrame] = []
    split_aggregate_rows: list[pd.DataFrame] = []
    if mode in {"splits", "full", "notebook"}:
        for evaluation in evaluations.values():
            detail_df, aggregate_df = _split_oos_rows(evaluation, spec)
            split_detail_rows.append(detail_df)
            split_aggregate_rows.append(aggregate_df)
        split_details = pd.concat(split_detail_rows, ignore_index=True) if split_detail_rows else pd.DataFrame()
        split_summary = pd.concat(split_aggregate_rows, ignore_index=True) if split_aggregate_rows else pd.DataFrame()
        split_details.to_csv(output_dir / "split_details.csv", index=False)
        split_summary.to_csv(output_dir / "split_summary.csv", index=False)
        artifacts["split_details_csv"] = output_dir / "split_details.csv"
        artifacts["split_summary_csv"] = output_dir / "split_summary.csv"
    else:
        split_summary = pd.DataFrame(columns=["strategy_id"])

    if mode in {"full", "notebook"}:
        prop_df = _build_prop_summary(evaluations, spec)
        concentration_df = _build_concentration_summary(evaluations, spec)
        prop_df.to_csv(output_dir / "prop_summary.csv", index=False)
        concentration_df.to_csv(output_dir / "concentration_summary.csv", index=False)
        artifacts["prop_summary_csv"] = output_dir / "prop_summary.csv"
        artifacts["concentration_summary_csv"] = output_dir / "concentration_summary.csv"

        reranking_df = _merge_reranking_tables(
            spec=spec,
            evaluations=evaluations,
            stress_df=stress_df,
            split_aggregate_df=split_summary,
            prop_df=prop_df,
            concentration_df=concentration_df,
        )
        verdict = _final_verdict_payload(reranking_df)
        artifacts.update(_write_summary_outputs(output_dir, reranking_df, verdict))
        artifacts.update(_run_survivor_heatmaps(output_dir, spec, feature_df, evaluations, reranking_df, cache))

    if mode == "notebook" and notebook_path is not None:
        artifacts["validation_notebook"] = generate_reranking_notebook(notebook_path=notebook_path, output_dir=output_dir)

    _json_dump(
        output_dir / "run_metadata.json",
        {
            "run_timestamp": datetime.now().isoformat(),
            "mode": mode,
            "dataset_path": spec.dataset_path,
            "variant_names": list(spec.variant_names),
            "output_dir": output_dir,
        },
    )
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the leak-free VWAP reranking campaign.")
    parser.add_argument("--dataset", type=Path, default=None, help="Optional dataset path. Defaults to the latest MNQ 1-minute file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional export directory.")
    parser.add_argument("--mode", type=str, default="full", choices=RERANKING_MODES, help="Campaign mode: baseline, stress, splits, full, or notebook.")
    parser.add_argument("--notebook-path", type=Path, default=None, help="Optional notebook output path.")
    args = parser.parse_args()

    spec = build_default_reranking_spec(dataset_path=args.dataset)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (EXPORTS_DIR / f"vwap_reranking_{timestamp}")
    notebook_path = args.notebook_path or (NOTEBOOKS_DIR / "vwap_reranking_validation.ipynb")
    artifacts = run_vwap_reranking_campaign(spec=spec, output_dir=output_dir, mode=args.mode, notebook_path=notebook_path)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
