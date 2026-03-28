"""Macro-event overlay research on the audited MNQ ORB Topstep baseline."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.mnq_orb_prop_challenge_simulation import (
    DEFAULT_PRIMARY_SCOPE,
    PropChallengeRuleset,
    VariantInput,
    _find_latest_source_run,
    _json_dump,
    _load_variant_input,
    _normalize_daily_results,
    _read_run_metadata,
    _scope_daily_results,
    _source_is_fraction,
    _summary_row_map,
    aggregate_simulation_runs,
    run_rolling_start_simulations,
)
from src.config.paths import EXPORTS_DIR, PROCESSED_DATA_DIR, ensure_directories


DEFAULT_SOURCE_VARIANT_NAME = "nominal"
DEFAULT_CALENDAR_DAILY_FEATURES_PATH = (
    PROCESSED_DATA_DIR / "economic_calendar" / "economic_calendar_daily_features.csv"
)
DEFAULT_RESET_COST_USD = 100.0
DEFAULT_PAYOUT_VALUE_USD = 3_000.0
BOOL_FEATURE_COLUMNS = (
    "is_fomc_day",
    "is_fomc_minutes_day",
    "is_powell_day",
    "is_cpi_day",
    "is_core_cpi_day",
    "is_nfp_day",
    "is_core_pce_day",
    "is_ppi_day",
    "is_ism_manufacturing_day",
    "is_ism_services_day",
    "is_retail_sales_day",
    "is_gdp_day",
    "is_high_impact_macro_day",
    "has_pre_930_event",
    "has_rth_event",
    "has_post_1600_event",
)
RAW_COHORT_ORDER = (
    "normal_day",
    "fomc_day",
    "cpi_or_nfp_day",
    "other_high_impact_macro_day",
    "any_high_impact_macro_day",
)
PRIORITY_BUCKET_ORDER = (
    "cpi_or_nfp_day",
    "fomc_day",
    "other_high_impact_macro_day",
    "normal_day",
)


@dataclass(frozen=True)
class MacroVariantDefinition:
    name: str
    family: str
    description: str
    trigger_column: str | None
    event_scale: float = 1.0


@dataclass(frozen=True)
class MacroEventCampaignSpec:
    source_run_root: Path | None = None
    source_variant_name: str = DEFAULT_SOURCE_VARIANT_NAME
    primary_scope: str = DEFAULT_PRIMARY_SCOPE
    calendar_daily_features_path: Path = DEFAULT_CALENDAR_DAILY_FEATURES_PATH
    include_quarterx_variants: bool = True
    reset_cost_usd: float = DEFAULT_RESET_COST_USD
    payout_value_usd: float = DEFAULT_PAYOUT_VALUE_USD
    ruleset: PropChallengeRuleset = PropChallengeRuleset(
        name="topstep_50k_macro_overlay",
        family="topstep_trailing_drawdown",
        resembles="Topstep-style trailing DD challenge",
        description="Trailing-DD challenge on the audited overlap window, without a hard traded-day expiry.",
        account_size_usd=50_000.0,
        profit_target_usd=3_000.0,
        max_traded_days=None,
        daily_loss_limit_usd=1_000.0,
        trailing_drawdown_usd=2_000.0,
        notes=(
            "No hard traded-day expiry is used here because the production macro calendar coverage only overlaps "
            "a recent 2026 slice of the audited strategy history."
        ),
    )
    output_root: Path | None = None


@dataclass
class MacroAnalysisInput:
    source_root: Path
    metadata: dict[str, Any]
    is_fraction: float
    variant_input: VariantInput
    analysis_daily: pd.DataFrame
    analysis_trades: pd.DataFrame
    calendar_daily: pd.DataFrame
    coverage_summary: dict[str, Any]


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(float(denominator)):
        return float(default)
    value = float(numerator) / float(denominator)
    return float(value) if math.isfinite(value) else float(default)


def _nan_mean(values: pd.Series | np.ndarray | list[float]) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.mean()) if not series.empty else float("nan")


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.map(
        lambda value: ""
        if pd.isna(value)
        else str(value).strip().lower()
    )
    return normalized.isin({"true", "1", "yes", "y", "t"})


def _negative_streak_lengths(values: pd.Series | list[float] | np.ndarray) -> list[int]:
    streaks: list[int] = []
    current = 0
    for value in pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0):
        if value < 0:
            current += 1
            continue
        if current > 0:
            streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return streaks


def _sharpe_ratio(daily_pnl: pd.Series, capital: float) -> float:
    if capital <= 0:
        return 0.0
    daily_returns = pd.to_numeric(daily_pnl, errors="coerce").fillna(0.0) / float(capital)
    if len(daily_returns) < 2:
        return 0.0
    std = float(daily_returns.std(ddof=0))
    if std <= 0:
        return 0.0
    return float((daily_returns.mean() / std) * math.sqrt(252.0))


def _sortino_ratio(daily_pnl: pd.Series, capital: float) -> float:
    if capital <= 0:
        return 0.0
    daily_returns = pd.to_numeric(daily_pnl, errors="coerce").fillna(0.0) / float(capital)
    downside = daily_returns[daily_returns < 0]
    if len(daily_returns) < 2 or downside.empty:
        return 0.0
    downside_std = float(np.sqrt(np.mean(np.square(downside))))
    if downside_std <= 0:
        return 0.0
    return float((daily_returns.mean() / downside_std) * math.sqrt(252.0))


def _profit_factor(pnl: pd.Series) -> float:
    values = pd.to_numeric(pnl, errors="coerce").fillna(0.0)
    gross_profit = float(values[values > 0].sum())
    gross_loss_abs = float(abs(values[values < 0].sum()))
    if gross_loss_abs == 0.0:
        return float("inf") if gross_profit > 0.0 else 0.0
    return float(gross_profit / gross_loss_abs)


def _remaining_trade_start_dates(daily_results: pd.DataFrame) -> list:
    ordered = _normalize_daily_results(daily_results)
    traded_mask = pd.to_numeric(ordered["daily_trade_count"], errors="coerce").fillna(0.0).gt(0.0).astype(int)
    remaining_traded = traded_mask.iloc[::-1].cumsum().iloc[::-1]
    return ordered.loc[remaining_traded >= 1, "session_date"].tolist()


def _prefixed(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def default_macro_variant_definitions(include_quarterx_variants: bool = True) -> tuple[MacroVariantDefinition, ...]:
    variants = [
        MacroVariantDefinition(
            name="baseline",
            family="baseline",
            description="Trade every eligible day at nominal risk.",
            trigger_column=None,
            event_scale=1.0,
        ),
        MacroVariantDefinition(
            name="skip_fomc",
            family="hard_filter",
            description="Skip FOMC rate-decision days.",
            trigger_column="fomc_day",
            event_scale=0.0,
        ),
        MacroVariantDefinition(
            name="skip_cpi_nfp",
            family="hard_filter",
            description="Skip CPI / core CPI / NFP days.",
            trigger_column="cpi_or_nfp_day",
            event_scale=0.0,
        ),
        MacroVariantDefinition(
            name="skip_all_high_impact",
            family="hard_filter",
            description="Skip every high-impact macro day.",
            trigger_column="any_high_impact_macro_day",
            event_scale=0.0,
        ),
        MacroVariantDefinition(
            name="deleverage_fomc_0.5x",
            family="deleverage",
            description="Trade FOMC days at 0.5x nominal risk.",
            trigger_column="fomc_day",
            event_scale=0.5,
        ),
        MacroVariantDefinition(
            name="deleverage_cpi_nfp_0.5x",
            family="deleverage",
            description="Trade CPI / core CPI / NFP days at 0.5x nominal risk.",
            trigger_column="cpi_or_nfp_day",
            event_scale=0.5,
        ),
        MacroVariantDefinition(
            name="deleverage_all_high_impact_0.5x",
            family="deleverage",
            description="Trade every high-impact macro day at 0.5x nominal risk.",
            trigger_column="any_high_impact_macro_day",
            event_scale=0.5,
        ),
    ]
    if include_quarterx_variants:
        variants.extend(
            [
                MacroVariantDefinition(
                    name="deleverage_fomc_0.25x",
                    family="deleverage",
                    description="Trade FOMC days at 0.25x nominal risk.",
                    trigger_column="fomc_day",
                    event_scale=0.25,
                ),
                MacroVariantDefinition(
                    name="deleverage_cpi_nfp_0.25x",
                    family="deleverage",
                    description="Trade CPI / core CPI / NFP days at 0.25x nominal risk.",
                    trigger_column="cpi_or_nfp_day",
                    event_scale=0.25,
                ),
                MacroVariantDefinition(
                    name="deleverage_all_high_impact_0.25x",
                    family="deleverage",
                    description="Trade every high-impact macro day at 0.25x nominal risk.",
                    trigger_column="any_high_impact_macro_day",
                    event_scale=0.25,
                ),
            ]
        )
    return tuple(variants)


def load_calendar_daily_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Macro daily-features file not found: {path}")
    daily = pd.read_csv(path)
    if "trade_date" not in daily.columns:
        raise ValueError("Calendar daily-features file must include a 'trade_date' column.")
    daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.date
    for column in BOOL_FEATURE_COLUMNS:
        if column in daily.columns:
            daily[column] = _coerce_bool(daily[column])
    if "nb_high_impact_events" in daily.columns:
        daily["nb_high_impact_events"] = pd.to_numeric(
            daily["nb_high_impact_events"],
            errors="coerce",
        ).fillna(0).astype(int)
    return daily.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)


def assign_macro_day_cohorts(daily_frame: pd.DataFrame) -> pd.DataFrame:
    out = daily_frame.copy()
    for column in BOOL_FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = False
        out[column] = _coerce_bool(out[column])

    out["fomc_day"] = out["is_fomc_day"]
    out["cpi_or_nfp_day"] = out["is_cpi_day"] | out["is_core_cpi_day"] | out["is_nfp_day"]
    out["any_high_impact_macro_day"] = out["is_high_impact_macro_day"]
    out["other_high_impact_macro_day"] = out["any_high_impact_macro_day"] & ~(
        out["fomc_day"] | out["cpi_or_nfp_day"]
    )
    out["normal_day"] = ~out["any_high_impact_macro_day"]
    out["priority_bucket"] = np.select(
        [
            out["cpi_or_nfp_day"],
            out["fomc_day"],
            out["other_high_impact_macro_day"],
        ],
        [
            "cpi_or_nfp_day",
            "fomc_day",
            "other_high_impact_macro_day",
        ],
        default="normal_day",
    )
    return out


def merge_strategy_with_calendar(
    daily_results: pd.DataFrame,
    calendar_daily_features: pd.DataFrame,
) -> pd.DataFrame:
    ordered = _normalize_daily_results(daily_results)
    ordered["session_date"] = pd.to_datetime(ordered["session_date"]).dt.date
    calendar = calendar_daily_features.copy()
    calendar["trade_date"] = pd.to_datetime(calendar["trade_date"]).dt.date
    merged = ordered.merge(
        calendar,
        left_on="session_date",
        right_on="trade_date",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(
            "No overlap was found between the audited strategy daily results and the macro calendar coverage."
        )
    merged = merged.sort_values("session_date").reset_index(drop=True)
    return assign_macro_day_cohorts(merged)


def _filter_trades_to_sessions(trades: pd.DataFrame, session_dates: pd.Series | list) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    allowed_dates = set(pd.to_datetime(pd.Index(session_dates)).date)
    view = trades.copy()
    view["session_date"] = pd.to_datetime(view["session_date"]).dt.date
    return view.loc[view["session_date"].isin(allowed_dates)].copy().reset_index(drop=True)


def _recompute_equity_columns(daily_results: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    out = daily_results.copy()
    pnl = pd.to_numeric(out["daily_pnl_usd"], errors="coerce").fillna(0.0)
    out["daily_pnl_usd"] = pnl
    out["daily_gross_pnl_usd"] = pd.to_numeric(
        out.get("daily_gross_pnl_usd", 0.0),
        errors="coerce",
    ).fillna(0.0)
    out["daily_fees_usd"] = pd.to_numeric(out.get("daily_fees_usd", 0.0), errors="coerce").fillna(0.0)
    out["daily_trade_count"] = pd.to_numeric(out.get("daily_trade_count", 0.0), errors="coerce").fillna(0.0)
    out["daily_loss_count"] = pd.to_numeric(out.get("daily_loss_count", 0.0), errors="coerce").fillna(0.0)
    out["equity"] = float(initial_capital) + pnl.cumsum()
    out["peak_equity"] = out["equity"].cummax()
    out["drawdown_usd"] = out["equity"] - out["peak_equity"]
    out["drawdown_pct"] = np.where(
        out["peak_equity"] > 0.0,
        (out["peak_equity"] - out["equity"]) / out["peak_equity"],
        0.0,
    )
    out["green_day"] = pnl > 0.0
    return out


def apply_macro_overlay(
    analysis_daily: pd.DataFrame,
    variant: MacroVariantDefinition,
    initial_capital: float,
) -> pd.DataFrame:
    out = analysis_daily.copy()
    if variant.trigger_column is None:
        overlay_factor = pd.Series(1.0, index=out.index, dtype=float)
    else:
        trigger_mask = _coerce_bool(out[variant.trigger_column])
        overlay_factor = pd.Series(
            np.where(trigger_mask, float(variant.event_scale), 1.0),
            index=out.index,
            dtype=float,
        )
    out["overlay_factor"] = overlay_factor
    out["overlay_impacted_day"] = overlay_factor.ne(1.0)

    for column in ("daily_pnl_usd", "daily_gross_pnl_usd", "daily_fees_usd"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0) * overlay_factor

    original_trade_count = pd.to_numeric(out["daily_trade_count"], errors="coerce").fillna(0.0)
    out["daily_trade_count"] = np.where(overlay_factor <= 0.0, 0.0, original_trade_count)
    out["daily_loss_count"] = np.where(
        out["daily_trade_count"] > 0.0,
        (pd.to_numeric(out["daily_pnl_usd"], errors="coerce").fillna(0.0) < 0.0).astype(float),
        0.0,
    )
    out = _recompute_equity_columns(out, initial_capital=initial_capital)
    return out


def summarize_daily_performance(daily_results: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    ordered = _normalize_daily_results(daily_results)
    pnl = pd.to_numeric(ordered["daily_pnl_usd"], errors="coerce").fillna(0.0)
    trade_count = pd.to_numeric(ordered["daily_trade_count"], errors="coerce").fillna(0.0)
    traded_mask = trade_count > 0.0
    traded_pnl = pnl[traded_mask]
    win_days = traded_pnl[traded_pnl > 0.0]
    loss_days = traded_pnl[traded_pnl < 0.0]
    streaks = _negative_streak_lengths(pnl)

    if ordered.empty:
        start_date = pd.NA
        end_date = pd.NA
    else:
        start_date = ordered["session_date"].iloc[0]
        end_date = ordered["session_date"].iloc[-1]

    return {
        "start_date": start_date,
        "end_date": end_date,
        "calendar_day_count": int(len(ordered)),
        "traded_day_count": int(traded_mask.sum()),
        "trade_count": int(trade_count.sum()),
        "total_pnl_usd": float(pnl.sum()),
        "avg_pnl_per_calendar_day": float(pnl.mean()) if len(pnl) > 0 else 0.0,
        "avg_pnl_per_traded_day": float(traded_pnl.mean()) if not traded_pnl.empty else 0.0,
        "expectancy_per_trade": _safe_div(float(pnl.sum()), float(trade_count.sum()), default=0.0),
        "win_rate_by_day": float((traded_pnl > 0.0).mean()) if not traded_pnl.empty else 0.0,
        "avg_win_day": float(win_days.mean()) if not win_days.empty else 0.0,
        "avg_loss_day": float(loss_days.mean()) if not loss_days.empty else 0.0,
        "profit_factor": _profit_factor(traded_pnl),
        "sharpe": _sharpe_ratio(pnl, capital=initial_capital),
        "sortino": _sortino_ratio(pnl, capital=initial_capital),
        "max_drawdown": float(
            pd.to_numeric(ordered.get("drawdown_usd", pd.Series([0.0])), errors="coerce").min()
        )
        if not ordered.empty
        else 0.0,
        "worst_day": float(pnl.min()) if not pnl.empty else 0.0,
        "longest_losing_streak": int(max(streaks, default=0)),
    }


def summarize_challenge_runs(
    runs: pd.DataFrame,
    payout_value_usd: float,
    reset_cost_usd: float,
) -> dict[str, Any]:
    metrics = aggregate_simulation_runs(runs)
    if runs.empty:
        metrics.update(
            {
                "expected_cycle_days": float("nan"),
                "expected_net_profit_per_cycle": 0.0,
                "expected_net_profit_per_day": 0.0,
                "probability_reaching_target_before_breach": 0.0,
                "probability_breaching_daily_loss_limit": 0.0,
                "probability_breaching_max_loss_limit": 0.0,
                "expected_days_to_pass": float("nan"),
                "insufficient_history_fail_rate": 0.0,
                "daily_loss_failure_rate": 0.0,
                "trailing_drawdown_failure_rate": 0.0,
            }
        )
        return metrics

    cycle_days = pd.to_numeric(runs["days_traded"], errors="coerce").dropna()
    failure_reason = runs["failure_reason"].fillna("").astype(str)
    expected_cycle_days = _nan_mean(cycle_days)
    expected_net_profit_per_cycle = (
        float(metrics["pass_rate"]) * float(payout_value_usd)
        - float(metrics["fail_rate"]) * float(reset_cost_usd)
    )

    metrics.update(
        {
            "expected_cycle_days": expected_cycle_days,
            "expected_net_profit_per_cycle": float(expected_net_profit_per_cycle),
            "expected_net_profit_per_day": _safe_div(
                expected_net_profit_per_cycle,
                expected_cycle_days,
                default=0.0,
            ),
            "probability_reaching_target_before_breach": float(metrics["pass_rate"]),
            "probability_breaching_daily_loss_limit": float(metrics["daily_loss_violation_rate"]),
            "probability_breaching_max_loss_limit": float(metrics["global_max_loss_violation_rate"]),
            "expected_days_to_pass": float(metrics["mean_days_to_pass"]),
            "insufficient_history_fail_rate": float(failure_reason.eq("insufficient_history").mean()),
            "daily_loss_failure_rate": float(failure_reason.eq("daily_loss_limit").mean()),
            "trailing_drawdown_failure_rate": float(failure_reason.eq("trailing_drawdown").mean()),
        }
    )
    return metrics


def _make_overlay_variant_input(
    base_variant: VariantInput,
    variant: MacroVariantDefinition,
    daily_results: pd.DataFrame,
) -> VariantInput:
    return VariantInput(
        variant_name=variant.name,
        label=variant.name,
        source_root=base_variant.source_root,
        trades=base_variant.trades.copy(),
        daily_results=daily_results.copy(),
        controls=pd.DataFrame(),
        reference_account_size_usd=base_variant.reference_account_size_usd,
        source_summary_row=base_variant.source_summary_row,
    )


def load_macro_analysis_input(spec: MacroEventCampaignSpec) -> MacroAnalysisInput:
    source_root = spec.source_run_root or _find_latest_source_run()
    metadata = _read_run_metadata(source_root)
    is_fraction = _source_is_fraction(metadata)
    summary_rows = _summary_row_map(source_root)

    baseline_variant = _load_variant_input(
        source_root=source_root,
        variant_name=spec.source_variant_name,
        summary_rows=summary_rows,
    )
    scoped_daily = _scope_daily_results(
        baseline_variant.daily_results,
        is_fraction=is_fraction,
        scope=spec.primary_scope,
    )
    calendar_daily = load_calendar_daily_features(spec.calendar_daily_features_path)
    merged_daily = merge_strategy_with_calendar(scoped_daily, calendar_daily)
    analysis_trades = _filter_trades_to_sessions(baseline_variant.trades, merged_daily["session_date"])
    analysis_variant = VariantInput(
        variant_name=baseline_variant.variant_name,
        label=baseline_variant.label,
        source_root=baseline_variant.source_root,
        trades=analysis_trades,
        daily_results=merged_daily.copy(),
        controls=baseline_variant.controls.copy(),
        reference_account_size_usd=baseline_variant.reference_account_size_usd,
        source_summary_row=baseline_variant.source_summary_row,
    )

    scoped_daily_dates = pd.to_datetime(pd.Index(scoped_daily["session_date"])).date
    calendar_dates = pd.to_datetime(pd.Index(calendar_daily["trade_date"])).date
    merged_dates = pd.to_datetime(pd.Index(merged_daily["session_date"])).date
    coverage_summary = {
        "source_scope_start_date": min(scoped_daily_dates).isoformat() if len(scoped_daily_dates) else None,
        "source_scope_end_date": max(scoped_daily_dates).isoformat() if len(scoped_daily_dates) else None,
        "calendar_start_date": min(calendar_dates).isoformat() if len(calendar_dates) else None,
        "calendar_end_date": max(calendar_dates).isoformat() if len(calendar_dates) else None,
        "overlap_start_date": min(merged_dates).isoformat() if len(merged_dates) else None,
        "overlap_end_date": max(merged_dates).isoformat() if len(merged_dates) else None,
        "source_scope_session_count": int(len(scoped_daily)),
        "calendar_session_count": int(len(calendar_daily)),
        "overlap_session_count": int(len(merged_daily)),
        "overlap_traded_day_count": int(
            pd.to_numeric(merged_daily["daily_trade_count"], errors="coerce").fillna(0.0).gt(0.0).sum()
        ),
        "overlap_trade_count": int(pd.to_numeric(analysis_trades.get("trade_id"), errors="coerce").notna().sum())
        if not analysis_trades.empty and "trade_id" in analysis_trades.columns
        else int(pd.to_numeric(merged_daily["daily_trade_count"], errors="coerce").fillna(0.0).sum()),
    }
    return MacroAnalysisInput(
        source_root=source_root,
        metadata=metadata,
        is_fraction=is_fraction,
        variant_input=analysis_variant,
        analysis_daily=merged_daily,
        analysis_trades=analysis_trades,
        calendar_daily=calendar_daily,
        coverage_summary=coverage_summary,
    )


def build_macro_cohort_summary(
    analysis: MacroAnalysisInput,
    spec: MacroEventCampaignSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_daily = analysis.analysis_daily.copy()
    eligible_start_set = set(pd.to_datetime(pd.Index(_remaining_trade_start_dates(baseline_daily))).date)

    for cohort_group, cohorts in (
        ("raw_flag", RAW_COHORT_ORDER),
        ("priority_bucket", PRIORITY_BUCKET_ORDER),
    ):
        for cohort_name in cohorts:
            if cohort_group == "raw_flag":
                mask = _coerce_bool(baseline_daily[cohort_name])
            else:
                mask = baseline_daily["priority_bucket"].eq(cohort_name)

            cohort_daily = baseline_daily.loc[mask].copy().reset_index(drop=True)
            performance = summarize_daily_performance(
                cohort_daily,
                initial_capital=spec.ruleset.account_size_usd,
            )
            start_dates = [
                session_date
                for session_date in pd.to_datetime(pd.Index(cohort_daily["session_date"])).date
                if session_date in eligible_start_set
            ]
            rolling_runs = run_rolling_start_simulations(
                daily_results=baseline_daily,
                variant=analysis.variant_input,
                ruleset=spec.ruleset,
                start_dates=start_dates,
            )
            rolling_summary = summarize_challenge_runs(
                rolling_runs,
                payout_value_usd=spec.payout_value_usd,
                reset_cost_usd=spec.reset_cost_usd,
            )

            rows.append(
                {
                    "cohort_group": cohort_group,
                    "cohort_name": cohort_name,
                    "rare_sample_flag": bool(
                        performance["calendar_day_count"] < 5 or performance["traded_day_count"] < 5
                    ),
                    **performance,
                    **_prefixed("rolling", rolling_summary),
                }
            )

    return pd.DataFrame(rows)


def build_macro_variant_results(
    analysis: MacroAnalysisInput,
    spec: MacroEventCampaignSpec,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list]:
    definitions = default_macro_variant_definitions(include_quarterx_variants=spec.include_quarterx_variants)
    initial_capital = float(spec.ruleset.account_size_usd)

    overlay_daily_map: dict[str, pd.DataFrame] = {}
    for definition in definitions:
        overlay_daily_map[definition.name] = apply_macro_overlay(
            analysis_daily=analysis.analysis_daily,
            variant=definition,
            initial_capital=initial_capital,
        )

    common_start_dates: set | None = None
    for daily_results in overlay_daily_map.values():
        starts = set(pd.to_datetime(pd.Index(_remaining_trade_start_dates(daily_results))).date)
        common_start_dates = starts if common_start_dates is None else (common_start_dates & starts)
    ordered_common_start_dates = sorted(common_start_dates) if common_start_dates else []

    rows: list[dict[str, Any]] = []
    rolling_runs_by_variant: dict[str, pd.DataFrame] = {}
    for definition in definitions:
        overlay_daily = overlay_daily_map[definition.name]
        overlay_variant = _make_overlay_variant_input(
            base_variant=analysis.variant_input,
            variant=definition,
            daily_results=overlay_daily,
        )
        rolling_runs = run_rolling_start_simulations(
            daily_results=overlay_daily,
            variant=overlay_variant,
            ruleset=spec.ruleset,
            start_dates=ordered_common_start_dates,
        )
        rolling_runs_by_variant[definition.name] = rolling_runs

        sample_metrics = summarize_daily_performance(
            overlay_daily,
            initial_capital=initial_capital,
        )
        rolling_summary = summarize_challenge_runs(
            rolling_runs,
            payout_value_usd=spec.payout_value_usd,
            reset_cost_usd=spec.reset_cost_usd,
        )
        impacted_day_mask = _coerce_bool(overlay_daily["overlay_impacted_day"])

        rows.append(
            {
                "variant_name": definition.name,
                "variant_family": definition.family,
                "description": definition.description,
                "trigger_column": definition.trigger_column or "none",
                "event_scale": float(definition.event_scale),
                "common_start_count": int(len(ordered_common_start_dates)),
                "impacted_calendar_days": int(impacted_day_mask.sum()),
                "impacted_traded_days": int(
                    (
                        impacted_day_mask
                        & pd.to_numeric(overlay_daily["daily_trade_count"], errors="coerce").fillna(0.0).gt(0.0)
                    ).sum()
                ),
                **_prefixed("sample", sample_metrics),
                **_prefixed("rolling", rolling_summary),
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        return results, rolling_runs_by_variant, ordered_common_start_dates

    baseline = results.loc[results["variant_name"].eq("baseline")].iloc[0]
    results["sample_total_pnl_retention_vs_baseline"] = results["sample_total_pnl_usd"].apply(
        lambda value: _safe_div(value, float(baseline["sample_total_pnl_usd"]), default=0.0)
    )
    results["sample_max_drawdown_improvement_vs_baseline"] = (
        abs(float(baseline["sample_max_drawdown"]))
        - abs(pd.to_numeric(results["sample_max_drawdown"], errors="coerce"))
    )
    results["rolling_pass_rate_delta_vs_baseline"] = (
        pd.to_numeric(results["rolling_pass_rate"], errors="coerce") - float(baseline["rolling_pass_rate"])
    )
    results["rolling_expected_net_profit_per_day_delta_vs_baseline"] = (
        pd.to_numeric(results["rolling_expected_net_profit_per_day"], errors="coerce")
        - float(baseline["rolling_expected_net_profit_per_day"])
    )
    results["rolling_global_max_loss_improvement_vs_baseline"] = (
        float(baseline["rolling_global_max_loss_violation_rate"])
        - pd.to_numeric(results["rolling_global_max_loss_violation_rate"], errors="coerce")
    )
    results["rolling_days_to_pass_delta_vs_baseline"] = (
        pd.to_numeric(results["rolling_mean_days_to_pass"], errors="coerce")
        - float(baseline["rolling_mean_days_to_pass"])
    )
    return results, rolling_runs_by_variant, ordered_common_start_dates


def build_macro_variant_ranking(variant_results: pd.DataFrame) -> pd.DataFrame:
    if variant_results.empty:
        return variant_results.copy()

    ranked = variant_results.copy()
    ranked["baseline_preference"] = ranked["variant_name"].eq("baseline").astype(int)
    ranked = ranked.sort_values(
        [
            "rolling_pass_rate",
            "rolling_expected_net_profit_per_day",
            "rolling_global_max_loss_violation_rate",
            "rolling_daily_loss_violation_rate",
            "rolling_mean_days_to_pass",
            "sample_avg_pnl_per_calendar_day",
            "baseline_preference",
            "variant_name",
        ],
        ascending=[False, False, True, True, True, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked.drop(columns=["baseline_preference"])


def _recommendation_block(ranking: pd.DataFrame) -> dict[str, pd.Series | None]:
    best_overall = ranking.iloc[0] if not ranking.empty else None
    best_skip = (
        ranking.loc[ranking["variant_family"].eq("hard_filter")].iloc[0]
        if not ranking.loc[ranking["variant_family"].eq("hard_filter")].empty
        else None
    )
    best_deleverage = (
        ranking.loc[ranking["variant_family"].eq("deleverage")].iloc[0]
        if not ranking.loc[ranking["variant_family"].eq("deleverage")].empty
        else None
    )
    baseline = (
        ranking.loc[ranking["variant_name"].eq("baseline")].iloc[0]
        if not ranking.loc[ranking["variant_name"].eq("baseline")].empty
        else None
    )
    return {
        "best_overall": best_overall,
        "best_skip": best_skip,
        "best_deleverage": best_deleverage,
        "baseline": baseline,
    }


def _format_candidate_line(row: pd.Series | None) -> str:
    if row is None:
        return "n/a"
    return (
        f"`{row['variant_name']}` | pass `{float(row['rolling_pass_rate']):.1%}` | "
        f"max-loss breach `{float(row['rolling_global_max_loss_violation_rate']):.1%}` | "
        f"days to pass `{float(row['rolling_mean_days_to_pass']):.1f}` | "
        f"expected net/day `{float(row['rolling_expected_net_profit_per_day']):.2f}`"
    )


def build_macro_business_report(
    output_path: Path,
    analysis: MacroAnalysisInput,
    spec: MacroEventCampaignSpec,
    cohort_summary: pd.DataFrame,
    variant_results: pd.DataFrame,
    ranking: pd.DataFrame,
    common_start_dates: list,
) -> None:
    if variant_results.empty:
        output_path.write_text("# Macro Business Report\n\nNo variant rows were produced.\n", encoding="utf-8")
        return

    recommendation = _recommendation_block(ranking)
    baseline = recommendation["baseline"]
    best_skip = recommendation["best_skip"]
    best_deleverage = recommendation["best_deleverage"]
    best_overall = recommendation["best_overall"]

    hard_filter_verdict = "inconclusive"
    speed_verdict = "inconclusive"
    deleverage_verdict = "inconclusive"
    recommended_policy = "No recommendation."
    hard_filter_impact_count = int(
        pd.to_numeric(
            ranking.loc[ranking["variant_family"].eq("hard_filter"), "impacted_traded_days"],
            errors="coerce",
        ).max()
        if not ranking.loc[ranking["variant_family"].eq("hard_filter")].empty
        else 0
    )
    deleverage_impact_count = int(
        pd.to_numeric(
            ranking.loc[ranking["variant_family"].eq("deleverage"), "impacted_traded_days"],
            errors="coerce",
        ).max()
        if not ranking.loc[ranking["variant_family"].eq("deleverage")].empty
        else 0
    )

    if hard_filter_impact_count <= 0:
        hard_filter_verdict = "not measurable in sample"
        speed_verdict = "not measurable in sample"
    elif baseline is not None and best_skip is not None:
        filter_better_survival = (
            float(best_skip["rolling_pass_rate"]) > float(baseline["rolling_pass_rate"])
            or float(best_skip["rolling_global_max_loss_violation_rate"])
            < float(baseline["rolling_global_max_loss_violation_rate"])
        )
        filter_clearly_worse = (
            float(best_skip["rolling_pass_rate"]) <= float(baseline["rolling_pass_rate"])
            and float(best_skip["rolling_global_max_loss_violation_rate"])
            >= float(baseline["rolling_global_max_loss_violation_rate"])
        )
        if filter_better_survival and not filter_clearly_worse:
            hard_filter_verdict = "yes"
        elif filter_clearly_worse:
            hard_filter_verdict = "no"

        speed_penalty_material = (
            pd.notna(best_skip["rolling_mean_days_to_pass"])
            and pd.notna(baseline["rolling_mean_days_to_pass"])
            and float(best_skip["rolling_mean_days_to_pass"]) > float(baseline["rolling_mean_days_to_pass"]) * 1.25
        ) or (
            float(best_skip["rolling_expected_net_profit_per_day"])
            < float(baseline["rolling_expected_net_profit_per_day"]) * 0.90
        )
        speed_verdict = "yes" if speed_penalty_material else "no"

    if hard_filter_impact_count <= 0 and deleverage_impact_count <= 0:
        deleverage_verdict = "not measurable in sample"
    elif best_skip is not None and best_deleverage is not None:
        deleverage_verdict = "yes" if int(best_deleverage["rank"]) < int(best_skip["rank"]) else "no"

    if best_overall is not None:
        recommended_policy = (
            f"Recommend `{best_overall['variant_name']}` "
            f"({best_overall['variant_family']}) on this overlap window."
        )
    if hard_filter_impact_count <= 0 and deleverage_impact_count <= 0:
        recommended_policy = (
            "Keep `baseline`; the overlap sample contains zero traded high-impact macro days, "
            "so macro overlays are not identified yet."
        )

    priority_view = cohort_summary.loc[cohort_summary["cohort_group"].eq("priority_bucket")].copy()
    rare_priority = priority_view.loc[priority_view["rare_sample_flag"]]
    ranking_display = ranking[
        [
            "rank",
            "variant_name",
            "variant_family",
            "impacted_traded_days",
            "rolling_pass_rate",
            "rolling_global_max_loss_violation_rate",
            "rolling_mean_days_to_pass",
            "rolling_expected_net_profit_per_day",
            "sample_total_pnl_usd",
        ]
    ]

    rare_text = "None."
    if not rare_priority.empty:
        rare_text = ", ".join(
            f"{row['cohort_name']} ({int(row['calendar_day_count'])} calendar / {int(row['traded_day_count'])} traded)"
            for _, row in rare_priority.iterrows()
        )

    lines = [
        "# Macro Business Report",
        "",
        "## Scope",
        f"- Source export: `{analysis.source_root}`",
        f"- Source variant: `{spec.source_variant_name}`",
        f"- Primary scope: `{spec.primary_scope}`",
        f"- Calendar features: `{spec.calendar_daily_features_path}`",
        f"- Overlap window: `{analysis.coverage_summary['overlap_start_date']}` to `{analysis.coverage_summary['overlap_end_date']}`",
        f"- Overlap sample: `{analysis.coverage_summary['overlap_session_count']}` sessions | `{analysis.coverage_summary['overlap_traded_day_count']}` traded days | `{analysis.coverage_summary['overlap_trade_count']}` trades",
        f"- Rolling common starts across variants: `{len(common_start_dates)}`",
        f"- Challenge rules: target `{spec.ruleset.profit_target_usd:,.0f}` | trailing DD `{float(spec.ruleset.trailing_drawdown_usd or 0.0):,.0f}` | daily loss `{float(spec.ruleset.daily_loss_limit_usd or 0.0):,.0f}`",
        f"- Economic objective reused from Topstep optimization: `pass_rate * {spec.payout_value_usd:,.0f} - fail_rate * {spec.reset_cost_usd:,.0f}`",
        "",
        "## Sample Limits",
        "- The production macro calendar currently overlaps only a recent 2026 slice of the audited strategy history, so this study is deliberately an overlap-window decision aid rather than a full-history claim.",
        f"- Rare priority cohorts: {rare_text}",
        f"- Traded event-day coverage inside the overlap: hard-filter variants `{hard_filter_impact_count}` impacted traded days | deleverage variants `{deleverage_impact_count}` impacted traded days.",
        "",
        "## Cohort Snapshot",
        "",
        "```text",
        priority_view[
            [
                "cohort_name",
                "calendar_day_count",
                "traded_day_count",
                "total_pnl_usd",
                "avg_pnl_per_calendar_day",
                "profit_factor",
                "rolling_pass_rate",
                "rolling_probability_breaching_max_loss_limit",
                "rolling_expected_days_to_pass",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Variant Ranking",
        "",
        "```text",
        ranking_display.to_string(index=False),
        "```",
        "",
        "## Verdict",
        f"- Hard filtering improves survivability: **{hard_filter_verdict}**. Best hard filter: {_format_candidate_line(best_skip)}.",
        f"- Hard filtering hurts passing speed too much: **{speed_verdict}**.",
        f"- Deleveraging is superior to hard filtering: **{deleverage_verdict}**. Best deleverage: {_format_candidate_line(best_deleverage)}.",
        f"- Recommended policy for Topstep 50K: **{recommended_policy}** Best overall row: {_format_candidate_line(best_overall)}.",
        "",
        "## Baseline",
        f"- {_format_candidate_line(baseline)}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_campaign(spec: MacroEventCampaignSpec) -> dict[str, Path]:
    ensure_directories()
    analysis = load_macro_analysis_input(spec)
    cohort_summary = build_macro_cohort_summary(analysis=analysis, spec=spec)
    variant_results, rolling_runs_by_variant, common_start_dates = build_macro_variant_results(
        analysis=analysis,
        spec=spec,
    )
    ranking = build_macro_variant_ranking(variant_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = spec.output_root or (EXPORTS_DIR / f"mnq_orb_macro_event_campaign_{timestamp}")
    output_root.mkdir(parents=True, exist_ok=True)

    cohort_summary_path = output_root / "macro_cohort_summary.csv"
    variant_results_path = output_root / "macro_variant_campaign_results.csv"
    ranking_path = output_root / "macro_variant_ranking.csv"
    report_path = output_root / "macro_business_report.md"
    metadata_path = output_root / "run_metadata.json"

    cohort_summary.to_csv(cohort_summary_path, index=False)
    variant_results.to_csv(variant_results_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    build_macro_business_report(
        output_path=report_path,
        analysis=analysis,
        spec=spec,
        cohort_summary=cohort_summary,
        variant_results=variant_results,
        ranking=ranking,
        common_start_dates=common_start_dates,
    )
    _json_dump(
        metadata_path,
        {
            "run_timestamp": datetime.now().isoformat(),
            "source_run_root": analysis.source_root,
            "source_variant_name": spec.source_variant_name,
            "primary_scope": spec.primary_scope,
            "source_is_fraction": analysis.is_fraction,
            "calendar_daily_features_path": spec.calendar_daily_features_path,
            "coverage_summary": analysis.coverage_summary,
            "ruleset": asdict(spec.ruleset),
            "payout_value_usd": spec.payout_value_usd,
            "reset_cost_usd": spec.reset_cost_usd,
            "include_quarterx_variants": spec.include_quarterx_variants,
            "variant_definitions": [
                asdict(definition) for definition in default_macro_variant_definitions(spec.include_quarterx_variants)
            ],
            "common_start_dates": [pd.Timestamp(value).date().isoformat() for value in common_start_dates],
            "rolling_run_counts": {name: int(len(frame)) for name, frame in rolling_runs_by_variant.items()},
        },
    )
    return {
        "macro_cohort_summary_csv": cohort_summary_path,
        "macro_variant_campaign_results_csv": variant_results_path,
        "macro_variant_ranking_csv": ranking_path,
        "macro_business_report_md": report_path,
        "run_metadata_json": metadata_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, default=None)
    parser.add_argument("--source-variant-name", type=str, default=DEFAULT_SOURCE_VARIANT_NAME)
    parser.add_argument("--primary-scope", type=str, default=DEFAULT_PRIMARY_SCOPE, choices=("overall", "oos"))
    parser.add_argument("--calendar-daily-features-path", type=Path, default=DEFAULT_CALENDAR_DAILY_FEATURES_PATH)
    parser.add_argument("--reset-cost-usd", type=float, default=DEFAULT_RESET_COST_USD)
    parser.add_argument("--payout-value-usd", type=float, default=DEFAULT_PAYOUT_VALUE_USD)
    parser.add_argument("--no-quarterx-variants", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = run_campaign(
        MacroEventCampaignSpec(
            source_run_root=args.source_run_root,
            source_variant_name=args.source_variant_name,
            primary_scope=args.primary_scope,
            calendar_daily_features_path=args.calendar_daily_features_path,
            include_quarterx_variants=not bool(args.no_quarterx_variants),
            reset_cost_usd=args.reset_cost_usd,
            payout_value_usd=args.payout_value_usd,
            output_root=args.output_root,
        )
    )
    print(f"Macro event campaign export written to {artifacts['macro_business_report_md'].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
