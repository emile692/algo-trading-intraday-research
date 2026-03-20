"""Structured ORB research campaign runner."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.analytics.metrics import compute_metrics
from src.config.orb_campaign import (
    DEFAULT_CAMPAIGN_DATASET,
    DEFAULT_OPENING_TIME,
    FocusedRankingConfig,
    ORBExperiment,
    PropConstraintConfig,
    RankingConfig,
    build_atr_regimes,
    build_execution_profiles,
    build_focused_atr_regimes,
    build_focused_orb_experiments,
    build_focused_ranking_config,
    build_orb_experiments,
    build_prop_constraints,
    build_ranking_config,
)
from src.config.paths import EXPORTS_DIR, NOTEBOOKS_DIR, ensure_directories
from src.data.cleaning import clean_ohlcv
from src.data.loader import load_ohlcv_file
from src.data.resampling import resample_ohlcv
from src.data.session import add_session_date, extract_rth
from src.engine.backtester import run_backtest
from src.engine.execution_model import ExecutionModel
from src.engine.portfolio import build_equity_curve
from src.features.intraday import add_ema, add_intraday_features, add_session_vwap
from src.features.opening_range import compute_opening_range
from src.features.volatility import add_atr
from src.strategy.orb import ORBStrategy
from src.strategy.orb_paper import ORBPaperExactStrategy
from src.visualization.equity import plot_drawdown_curve, plot_equity_curve


def prepare_base_rth_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load, clean, and filter the source dataset to RTH."""
    df = load_ohlcv_file(dataset_path)
    df = clean_ohlcv(df)
    df = extract_rth(df)
    df = add_session_date(df)
    df = add_intraday_features(df)
    return df


def prepare_current_logic_dataset(
    base_rth_df: pd.DataFrame,
    ema_lengths: tuple[int, ...] | list[int] | None = None,
    include_vwap: bool = True,
) -> pd.DataFrame:
    """Build the feature set used by the current ORB logic and filter campaign."""
    out = base_rth_df.copy()
    out = add_atr(out, window=14)
    if include_vwap:
        out = add_session_vwap(out)

    resolved_lengths = sorted({int(length) for length in (ema_lengths or (20, 50)) if length is not None})
    for ema_length in resolved_lengths:
        out = add_ema(out, window=ema_length)

    out = compute_opening_range(out, or_minutes=15, opening_time=DEFAULT_OPENING_TIME)
    return out


def prepare_paper_dataset(base_rth_df: pd.DataFrame) -> pd.DataFrame:
    """Resample to 5-minute bars and compute the first-candle OR for the paper variant."""
    paper_df = resample_ohlcv(base_rth_df[["timestamp", "open", "high", "low", "close", "volume"]], rule="5min")
    paper_df = add_session_date(paper_df)
    paper_df = add_intraday_features(paper_df)
    paper_df = compute_opening_range(paper_df, or_minutes=5, opening_time=DEFAULT_OPENING_TIME)
    return paper_df


def resolve_atr_bounds(
    df: pd.DataFrame,
    atr_regimes=None,
) -> dict[str, tuple[float | None, float | None]]:
    """Resolve named ATR regimes to explicit numeric thresholds."""
    atr_regimes = atr_regimes or build_atr_regimes()
    atr_series = df["atr_14"].dropna()
    bounds: dict[str, tuple[float | None, float | None]] = {"none": (None, None)}

    for name, regime in atr_regimes.items():
        if name == "none":
            continue
        lower = float(atr_series.quantile(regime.lower_quantile)) if regime.lower_quantile is not None else None
        upper = float(atr_series.quantile(regime.upper_quantile)) if regime.upper_quantile is not None else None
        bounds[name] = (lower, upper)
    return bounds


def apply_resolved_atr_bounds(
    experiments: list[ORBExperiment],
    atr_bounds: dict[str, tuple[float | None, float | None]],
) -> list[ORBExperiment]:
    """Return experiments with ATR regime names resolved to numeric bands."""
    resolved: list[ORBExperiment] = []
    for experiment in experiments:
        if experiment.atr_regime == "none":
            resolved.append(experiment)
            continue
        atr_min, atr_max = atr_bounds[experiment.atr_regime]
        resolved.append(replace(experiment, atr_min=atr_min, atr_max=atr_max))
    return resolved


def build_strategy(experiment: ORBExperiment, tick_size: float):
    """Instantiate the configured strategy for the experiment."""
    if experiment.strategy_variant == "paper_exact":
        return ORBPaperExactStrategy(
            opening_time=experiment.opening_time,
            or_minutes=experiment.or_minutes,
            one_trade_per_day=experiment.one_trade_per_day,
            time_exit=experiment.time_exit,
            target_multiple=experiment.target_multiple,
            account_size_usd=experiment.initial_capital_usd,
            risk_per_trade_pct=experiment.risk_per_trade_pct,
        )

    if experiment.strategy_variant == "current_orb":
        return ORBStrategy(
            or_minutes=experiment.or_minutes,
            direction=experiment.side_mode,
            one_trade_per_day=experiment.one_trade_per_day,
            entry_buffer_ticks=experiment.entry_buffer_ticks,
            stop_buffer_ticks=experiment.stop_buffer_ticks,
            target_multiple=experiment.target_multiple,
            opening_time=experiment.opening_time,
            time_exit=experiment.time_exit,
            account_size_usd=experiment.initial_capital_usd,
            risk_per_trade_pct=experiment.risk_per_trade_pct,
            tick_size=tick_size,
            atr_period=experiment.atr_period,
            atr_min=experiment.atr_min,
            atr_max=experiment.atr_max,
            atr_regime=experiment.atr_regime,
            direction_filter_mode=experiment.direction_filter_mode,
            ema_length=experiment.ema_length,
        )

    raise ValueError(f"Unsupported strategy_variant: {experiment.strategy_variant}")


def _strategy_group_key(experiment: ORBExperiment, tick_size: float) -> tuple:
    """Group experiments that can reuse the same generated signal dataframe."""
    return (
        experiment.strategy_variant,
        experiment.dataset_key,
        experiment.or_minutes,
        experiment.side_mode,
        experiment.entry_buffer_ticks,
        experiment.stop_buffer_ticks,
        experiment.opening_time,
        experiment.one_trade_per_day,
        experiment.atr_period,
        experiment.atr_regime,
        experiment.atr_min,
        experiment.atr_max,
        experiment.direction_filter_mode,
        experiment.ema_length,
        tick_size,
    )


def _build_execution_model(profile_name: str) -> ExecutionModel:
    profiles = build_execution_profiles()
    profile = profiles[profile_name]
    return ExecutionModel(
        commission_per_side_usd=profile.commission_per_side_usd,
        slippage_ticks=profile.slippage_ticks,
        tick_size=profile.tick_size,
    )


def _signal_session_dates(df: pd.DataFrame) -> pd.Index:
    return pd.Index(pd.to_datetime(df["session_date"]).dt.date.unique())


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def score_experiments(results: pd.DataFrame, ranking: RankingConfig) -> pd.DataFrame:
    """Apply the robustness-oriented leaderboard score."""

    def _score_row(row: pd.Series) -> pd.Series:
        avg_r_component = ranking.avg_r_weight * _clip(float(row["avg_R"]), -1.0, 2.0)

        profit_factor = float(row["profit_factor"])
        if not math.isfinite(profit_factor):
            profit_factor = 3.0
        profit_factor_component = ranking.profit_factor_weight * _clip((profit_factor - 1.0) / 1.5, -1.0, 2.0)

        scale = max(abs(float(row["avg_loss"])) if row["avg_loss"] != 0 else 0.0, 1.0)
        expectancy_norm = float(row["expectancy"]) / scale
        expectancy_component = ranking.expectancy_weight * _clip(expectancy_norm, -1.0, 2.0)

        drawdown_penalty = ranking.drawdown_weight * _clip(float(row["max_drawdown_pct"]) / 0.20, 0.0, 2.0)
        loss_streak_penalty = ranking.loss_streak_weight * float(row["longest_loss_streak"])
        participation_component = ranking.participation_weight * _clip(
            float(row["percent_of_days_traded"]) / ranking.target_days_traded,
            0.0,
            1.25,
        )
        trade_penalty = ranking.insufficient_trades_penalty * _clip(
            (ranking.min_trades - float(row["n_trades"])) / ranking.min_trades,
            0.0,
            1.0,
        )

        score = (
            avg_r_component
            + profit_factor_component
            + expectancy_component
            + participation_component
            - drawdown_penalty
            - loss_streak_penalty
            - trade_penalty
        )

        return pd.Series(
            {
                "passes_min_trades": bool(row["n_trades"] >= ranking.min_trades),
                "avg_r_component": avg_r_component,
                "profit_factor_component": profit_factor_component,
                "expectancy_component": expectancy_component,
                "participation_component": participation_component,
                "drawdown_penalty": drawdown_penalty,
                "loss_streak_penalty": loss_streak_penalty,
                "trade_penalty": trade_penalty,
                "robustness_score": score,
            }
        )

    scored = results.copy()
    components = scored.apply(_score_row, axis=1)
    scored = pd.concat([scored, components], axis=1)
    scored = scored.sort_values(
        by=["passes_min_trades", "robustness_score", "profit_factor", "expectancy"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return scored


def score_focused_experiments(
    results: pd.DataFrame,
    ranking: FocusedRankingConfig,
    prop_constraints: PropConstraintConfig,
) -> pd.DataFrame:
    """Apply the focused prop-style leaderboard score."""

    def _score_row(row: pd.Series) -> pd.Series:
        profit_factor = float(row["profit_factor"])
        if not math.isfinite(profit_factor):
            profit_factor = 3.0
        profit_factor_component = ranking.profit_factor_weight * _clip((profit_factor - 1.0) / 0.30, -1.0, 2.0)

        scale = max(abs(float(row["avg_loss"])) if row["avg_loss"] != 0 else 0.0, 1.0)
        expectancy_component = ranking.expectancy_weight * _clip(float(row["expectancy"]) / scale, -1.0, 2.0)
        trade_count_component = ranking.trade_count_weight * _clip(
            float(row["n_trades"]) / ranking.min_trades,
            0.0,
            1.25,
        )
        participation_component = ranking.participation_weight * _clip(
            float(row["percent_of_days_traded"]) / ranking.target_days_traded,
            0.0,
            1.25,
        )
        target_reached_component = ranking.target_reached_weight if bool(row["profit_target_reached_before_max_loss"]) else 0.0

        months_to_target = pd.to_numeric(pd.Series([row["estimated_months_to_profit_target"]]), errors="coerce").iloc[0]
        if pd.notna(months_to_target) and float(months_to_target) > 0:
            target_speed_component = ranking.target_speed_weight * _clip(
                ranking.target_months_to_goal / float(months_to_target),
                0.0,
                2.0,
            )
        else:
            target_speed_component = 0.0

        drawdown_penalty = ranking.drawdown_penalty_weight * _clip(
            abs(float(row["max_drawdown"])) / prop_constraints.max_loss_limit_usd,
            0.0,
            2.0,
        )
        loss_streak_penalty = ranking.loss_streak_penalty_weight * _clip(
            float(row["longest_loss_streak"]) / ranking.acceptable_loss_streak,
            0.0,
            2.0,
        )
        max_loss_breach_penalty = ranking.max_loss_breach_penalty_weight * int(bool(row["breaches_max_loss_limit"]))
        daily_loss_breach_penalty = ranking.daily_loss_breach_penalty_weight * _clip(
            float(row["number_of_daily_loss_limit_breaches"]),
            0.0,
            2.0,
        )
        subscription_drag_penalty = ranking.subscription_drag_penalty_weight * _clip(
            float(row["subscription_drag_estimate"]) / prop_constraints.profit_target_usd,
            0.0,
            2.0,
        )
        insufficient_trades_penalty = ranking.insufficient_trades_penalty_weight * _clip(
            (ranking.min_trades - float(row["n_trades"])) / ranking.min_trades,
            0.0,
            1.0,
        )

        score = (
            profit_factor_component
            + expectancy_component
            + trade_count_component
            + participation_component
            + target_reached_component
            + target_speed_component
            - drawdown_penalty
            - loss_streak_penalty
            - max_loss_breach_penalty
            - daily_loss_breach_penalty
            - subscription_drag_penalty
            - insufficient_trades_penalty
        )

        return pd.Series(
            {
                "passes_min_trades": bool(row["n_trades"] >= ranking.min_trades),
                "profit_factor_component": profit_factor_component,
                "expectancy_component": expectancy_component,
                "trade_count_component": trade_count_component,
                "participation_component": participation_component,
                "target_reached_component": target_reached_component,
                "target_speed_component": target_speed_component,
                "drawdown_penalty": drawdown_penalty,
                "loss_streak_penalty": loss_streak_penalty,
                "max_loss_breach_penalty": max_loss_breach_penalty,
                "daily_loss_breach_penalty": daily_loss_breach_penalty,
                "subscription_drag_penalty": subscription_drag_penalty,
                "insufficient_trades_penalty": insufficient_trades_penalty,
                "robustness_score": score,
            }
        )

    scored = results.copy()
    components = scored.apply(_score_row, axis=1)
    scored = pd.concat([scored, components], axis=1)
    scored = scored.sort_values(
        by=[
            "profit_target_reached_before_max_loss",
            "passes_min_trades",
            "robustness_score",
            "profit_factor",
            "expectancy",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    return scored


def _results_columns_for_report() -> list[str]:
    return [
        "name",
        "robustness_score",
        "n_trades",
        "profit_factor",
        "expectancy",
        "avg_R",
        "cumulative_pnl",
        "max_drawdown",
        "longest_loss_streak",
        "percent_of_days_traded",
    ]


def _focused_results_columns_for_report() -> list[str]:
    return [
        "name",
        "robustness_score",
        "n_trades",
        "profit_factor",
        "expectancy",
        "cumulative_pnl",
        "max_drawdown",
        "longest_loss_streak",
        "days_to_profit_target",
        "estimated_months_to_profit_target",
        "subscription_drag_estimate",
        "profit_target_reached_before_max_loss",
    ]


def _frame_as_code_block(df: pd.DataFrame) -> str:
    if df.empty:
        return "No experiments matched.\n"
    return "```text\n" + df.to_string(index=False) + "\n```\n"


def _format_metric(value: object, precision: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.{precision}f}"
    return str(value)


def _median_numeric(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else math.nan


def _mean_bool(series: pd.Series) -> float:
    return float(pd.Series(series).astype(bool).mean()) if len(series) > 0 else 0.0


def _dimension_summary(results: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Aggregate a compact robustness summary by one campaign dimension."""
    return (
        results.groupby(dimension, dropna=False)
        .agg(
            configs=("name", "count"),
            median_score=("robustness_score", "median"),
            best_score=("robustness_score", "max"),
            median_trades=("n_trades", "median"),
            median_profit_factor=("profit_factor", "median"),
            median_expectancy=("expectancy", "median"),
            median_drawdown=("max_drawdown", "median"),
            median_longest_loss_streak=("longest_loss_streak", "median"),
            target_reached_rate=("profit_target_reached", _mean_bool),
            target_before_loss_rate=("profit_target_reached_before_max_loss", _mean_bool),
            median_days_to_target=("days_to_profit_target", _median_numeric),
            median_subscription_drag=("subscription_drag_estimate", "median"),
        )
        .reset_index()
        .sort_values(["median_score", "best_score"], ascending=[False, False])
        .reset_index(drop=True)
    )


def generate_report(
    results: pd.DataFrame,
    leaderboard: pd.DataFrame,
    atr_bounds: dict[str, tuple[float | None, float | None]],
    ranking: RankingConfig,
    dataset_path: Path,
    output_dir: Path,
) -> str:
    """Build the markdown report summarizing the campaign."""
    report_columns = _results_columns_for_report()
    top_axis_a = leaderboard.loc[leaderboard["axis"] == "axis_a_paper_replication", report_columns].head(5)
    top_axis_b = leaderboard.loc[
        leaderboard["axis"].isin(["axis_b_current_logic_baselines", "axis_b_prop_viable"]),
        report_columns,
    ].head(8)
    top_axis_c = leaderboard.loc[leaderboard["axis"] == "axis_c_filter_campaign", report_columns].head(8)

    prop_candidates = leaderboard[
        leaderboard["axis"].isin(["axis_b_prop_viable", "axis_c_filter_campaign"]) & leaderboard["passes_min_trades"]
    ]
    recommendation_row = prop_candidates.head(1)

    atr_rows = [
        {
            "atr_regime": regime_name,
            "atr_min": lower,
            "atr_max": upper,
        }
        for regime_name, (lower, upper) in atr_bounds.items()
    ]
    atr_table = _frame_as_code_block(pd.DataFrame(atr_rows))

    recommendation_text = "No prop-style candidate passed the minimum-trade check."
    if not recommendation_row.empty:
        row = recommendation_row.iloc[0]
        recommendation_text = (
            f"{row['name']} ranked highest on the robustness score with {int(row['n_trades'])} trades, "
            f"profit factor {row['profit_factor']:.2f}, expectancy {row['expectancy']:.2f}, "
            f"max drawdown {row['max_drawdown']:.2f}, and longest loss streak {int(row['longest_loss_streak'])}."
        )

    formula = (
        "robustness_score = "
        f"{ranking.avg_r_weight} * clip(avg_R, -1, 2) + "
        f"{ranking.profit_factor_weight} * clip((profit_factor - 1) / 1.5, -1, 2) + "
        f"{ranking.expectancy_weight} * clip(expectancy / max(abs(avg_loss), 1), -1, 2) + "
        f"{ranking.participation_weight} * clip(percent_of_days_traded / {ranking.target_days_traded:.2f}, 0, 1.25) - "
        f"{ranking.drawdown_weight} * clip(max_drawdown_pct / 0.20, 0, 2) - "
        f"{ranking.loss_streak_weight} * longest_loss_streak - "
        f"{ranking.insufficient_trades_penalty} * clip((min_trades - n_trades) / min_trades, 0, 1), "
        f"with min_trades = {ranking.min_trades}."
    )

    return f"""# ORB Research Campaign

## Repo Inspection Findings

- Global market assumptions and default capital/costs live in `src/config/settings.py`.
- Session filtering is handled in `src/data/session.py` and the existing notebook uses `extract_rth(...)`.
- The current OR range is built in `src/features/opening_range.py`.
- The current breakout-after-OR signal logic lives in `src/strategy/orb.py`.
- Stop, target, entry timing, sizing, and exit handling live in `src/engine/backtester.py`.
- Execution costs are modeled in `src/engine/execution_model.py`.
- Metrics are computed in `src/analytics/metrics.py`.
- The repo already had a sweep helper in `src/analytics/heatmaps.py`, and outputs conventionally belong under `data/exports`.
- The central notebook loads the main research dataset from `{dataset_path.name}` and converts it to `America/New_York`.

## Current ORB vs Paper-Exact

- Current repo logic: 15-minute opening range on intraday futures bars, then enter only after a later breakout through OR high/low.
- Paper-exact variant here: 5-minute resampled bars, first candle defines bias, and the backtester enters at the next bar open, which is the second 5-minute candle open.
- Current repo stop logic: OR boundary stop plus optional buffer.
- Paper-exact stop logic here: first 5-minute candle high/low, which matches the 5-minute OR boundaries on the resampled dataset.
- Current repo default execution assumptions: futures-style slippage and per-side commissions.
- Paper-exact reference profile here: zero slippage and `0.0005` per unit commission in the engine. On the default MNQ futures dataset this is only an approximation of the paper's ETF/share assumptions.

## ATR Regimes

The filter campaign resolves ATR(14) regimes from the current RTH dataset using tertiles:

{atr_table}

## Ranking Framework

{formula}

The leaderboard applies the `passes_min_trades` guardrail so a low-activity configuration does not rank first just because it avoids drawdown.

## Axis A Top Results

{_frame_as_code_block(top_axis_a)}

## Axis B Top Results

{_frame_as_code_block(top_axis_b)}

## Axis C Top Results

{_frame_as_code_block(top_axis_c)}

## Recommendation

{recommendation_text}

## Caveats

- The default campaign dataset is `{dataset_path.name}`. If you want a closer paper replication on QQQ/TQQQ, you still need matching instrument data.
- The paper reference cost profile is only exact when quantity represents units/shares for a compatible instrument model.
- On futures-style contract data, lower risk-per-trade settings can reduce trade count because integer position sizing may skip signals that would require fractional contracts.
- Daily Sharpe is computed from daily PnL divided by a static initial capital base, then annualized with `sqrt(252)`.
- End-of-day exits use the configured `time_exit` or the last available bar in the session when no stop/target hit occurs first.
- The current filter campaign applies VWAP and EMA checks on the signal bar close to avoid lookahead.

## Outputs

- Full results CSV: `{(output_dir / 'orb_campaign_results.csv').as_posix()}`
- Leaderboard CSV: `{(output_dir / 'orb_campaign_leaderboard.csv').as_posix()}`
- Plot directory: `{(output_dir / 'plots').as_posix()}`
"""


def generate_focused_report(
    results: pd.DataFrame,
    leaderboard: pd.DataFrame,
    atr_bounds: dict[str, tuple[float | None, float | None]],
    ranking: FocusedRankingConfig,
    dataset_path: Path,
    output_dir: Path,
    prop_constraints: PropConstraintConfig,
    notebook_path: Path | None,
) -> str:
    """Build the markdown report for the focused prop-style campaign."""
    report_columns = _focused_results_columns_for_report()
    top_rows = leaderboard.loc[:, report_columns].head(12)

    atr_rows = [
        {"atr_regime": regime_name, "atr_min": lower, "atr_max": upper}
        for regime_name, (lower, upper) in atr_bounds.items()
    ]
    atr_table = _frame_as_code_block(pd.DataFrame(atr_rows))

    ema_summary = _dimension_summary(leaderboard, "ema_length")
    atr_summary = _dimension_summary(leaderboard, "atr_regime")
    rr_summary = _dimension_summary(leaderboard, "target_multiple")
    risk_summary = _dimension_summary(leaderboard, "risk_per_trade_pct")

    best_ema = ema_summary.iloc[0]
    best_atr = atr_summary.iloc[0]
    best_rr = rr_summary.iloc[0]
    best_risk = risk_summary.iloc[0]
    recommendation = leaderboard.iloc[0]
    target_hits = leaderboard[leaderboard["profit_target_reached_before_max_loss"]].copy()
    viable_target_hits = target_hits[target_hits["subscription_drag_estimate"] <= prop_constraints.profit_target_usd].copy()
    fastest_target_row = target_hits.sort_values(["days_to_profit_target", "max_drawdown"]).head(1)

    none_atr = atr_summary.loc[atr_summary["atr_regime"] == "none"].head(1)
    restrictive_atr = atr_summary.loc[atr_summary["atr_regime"] == "restrictive_band"].head(1)
    atr_frequency_text = ""
    if not none_atr.empty and not restrictive_atr.empty:
        atr_frequency_text = (
            f" Compared with no ATR filter, the restrictive regime changed median trades from "
            f"{_format_metric(none_atr.iloc[0]['median_trades'], 0)} to "
            f"{_format_metric(restrictive_atr.iloc[0]['median_trades'], 0)} and median drawdown from "
            f"{_format_metric(none_atr.iloc[0]['median_drawdown'])} to "
            f"{_format_metric(restrictive_atr.iloc[0]['median_drawdown'])}."
        )

    if target_hits.empty:
        target_speed_answer = (
            f"No configuration reached the `{prop_constraints.profit_target_usd:,.0f}` target before the "
            f"`{prop_constraints.max_loss_limit_usd:,.0f}` max-loss reference."
        )
    elif viable_target_hits.empty:
        fastest = fastest_target_row.iloc[0]
        target_speed_answer = (
            f"No configuration reached the `{prop_constraints.profit_target_usd:,.0f}` target before estimated "
            f"subscription drag exceeded the target itself. The fastest target hit still took "
            f"`{_format_metric(fastest['days_to_profit_target'], 0)}` trading days, or about "
            f"`{_format_metric(fastest['estimated_months_to_profit_target'])}` trading months."
        )
    else:
        fastest = viable_target_hits.sort_values(["days_to_profit_target", "max_drawdown"]).iloc[0]
        target_speed_answer = (
            f"The fastest subscription-viable target hit took `{_format_metric(fastest['days_to_profit_target'], 0)}` "
            f"trading days, or about `{_format_metric(fastest['estimated_months_to_profit_target'])}` trading months."
        )

    recommendation_sentence = (
        f"`{recommendation['name']}` is the strongest robust candidate for visual validation in this branch, "
        "but it is still too slow to look subscription-efficient for a typical prop evaluation."
        if viable_target_hits.empty
        else f"`{recommendation['name']}`."
    )

    formula = (
        "robustness_score = "
        f"{ranking.profit_factor_weight} * clip((profit_factor - 1) / 0.30, -1, 2) + "
        f"{ranking.expectancy_weight} * clip(expectancy / max(abs(avg_loss), 1), -1, 2) + "
        f"{ranking.trade_count_weight} * clip(n_trades / {ranking.min_trades}, 0, 1.25) + "
        f"{ranking.participation_weight} * clip(percent_of_days_traded / {ranking.target_days_traded:.2f}, 0, 1.25) + "
        f"{ranking.target_reached_weight} * int(target_reached_before_max_loss) + "
        f"{ranking.target_speed_weight} * clip({ranking.target_months_to_goal:.1f} / months_to_target, 0, 2) - "
        f"{ranking.drawdown_penalty_weight} * clip(abs(max_drawdown) / {prop_constraints.max_loss_limit_usd:.0f}, 0, 2) - "
        f"{ranking.loss_streak_penalty_weight} * clip(longest_loss_streak / {ranking.acceptable_loss_streak:.1f}, 0, 2) - "
        f"{ranking.max_loss_breach_penalty_weight} * int(breaches_max_loss_limit) - "
        f"{ranking.daily_loss_breach_penalty_weight} * clip(number_of_daily_loss_limit_breaches, 0, 2) - "
        f"{ranking.subscription_drag_penalty_weight} * clip(subscription_drag_estimate / {prop_constraints.profit_target_usd:.0f}, 0, 2) - "
        f"{ranking.insufficient_trades_penalty_weight} * clip((min_trades - n_trades) / min_trades, 0, 1)."
    )

    recommendation_lines = [
        f"- Name: `{recommendation['name']}`",
        f"- RR: `{_format_metric(recommendation['target_multiple'], 0)}`",
        f"- EMA length: `EMA{_format_metric(recommendation['ema_length'], 0)}`",
        f"- ATR mode: `{recommendation['atr_regime']}`",
        f"- Risk per trade: `{_format_metric(recommendation['risk_per_trade_pct'])}%`",
        f"- Trades: `{_format_metric(recommendation['n_trades'], 0)}`",
        f"- Profit factor: `{_format_metric(recommendation['profit_factor'])}`",
        f"- Expectancy: `${_format_metric(recommendation['expectancy'])}`",
        f"- Max drawdown: `${_format_metric(recommendation['max_drawdown'])}`",
        f"- Longest loss streak: `{_format_metric(recommendation['longest_loss_streak'], 0)}`",
        f"- Days to target: `{_format_metric(recommendation['days_to_profit_target'], 0)}`",
        f"- Estimated months to target: `{_format_metric(recommendation['estimated_months_to_profit_target'])}`",
        f"- Target reached before max loss: `{_format_metric(bool(recommendation['profit_target_reached_before_max_loss']))}`",
    ]

    notebook_line = (
        f"- Validation notebook: `{notebook_path.as_posix()}`"
        if notebook_path is not None
        else "- Validation notebook: not generated"
    )

    return f"""# Focused ORB Prop Campaign

## Objective

This campaign deepens the strongest practical branch from the previous sweep:

- current repo ORB logic
- long-only only
- EMA directional filter only
- optional ATR regime filter
- prop-style constraints centered on a 50K evaluation reference

## Research Constraints

- Account size reference: `${prop_constraints.account_size_usd:,.0f}`
- Max loss limit reference: `${prop_constraints.max_loss_limit_usd:,.0f}`
- Daily loss limit reference: `{_format_metric(prop_constraints.daily_loss_limit_usd) if prop_constraints.daily_loss_limit_usd is not None else 'disabled'}`
- Profit target reference: `${prop_constraints.profit_target_usd:,.0f}`
- Monthly subscription cost reference: `${prop_constraints.monthly_subscription_cost_usd:,.0f}`
- Daily loss limit basis: `{prop_constraints.daily_loss_limit_basis}`

These are research settings, not engine hard-codes.

## Focused Grid

- Reward ratio: `3`, `4`, `5`
- EMA length: `30`, `50`, `70`, `100`
- ATR filter mode: `none`, `moderate_band`, `restrictive_band`
- Risk per trade: `0.10%`, `0.15%`, `0.20%`, `0.25%`
- Dataset: `{dataset_path.name}`

## Resolved ATR Bands

{atr_table}

## Ranking Formula

{formula}

The score is intentionally practical: it rewards configurations that reach the target without violating the max loss reference, keeps trade count meaningful, and penalizes slow or sparse variants even if their raw PnL looks attractive.

## Top Ranked Configurations

{_frame_as_code_block(top_rows)}

## Direct Answers

- Which EMA length is most robust? `EMA{_format_metric(best_ema['ema_length'], 0)}` led on median robustness score at `{_format_metric(best_ema['median_score'])}` with median drawdown `{_format_metric(best_ema['median_drawdown'])}` and target-before-loss hit rate `{_format_metric(100 * best_ema['target_before_loss_rate'])}%`.
- Does ATR filtering genuinely help, or only reduce frequency? `{best_atr['atr_regime']}` ranked best on median score. The ATR filters can improve control in some pockets, but they materially change trade frequency rather than delivering a free improvement.{atr_frequency_text}
- Which RR is the best practical compromise? `RR {_format_metric(best_rr['target_multiple'], 0)}` had the strongest median score, with median trades `{_format_metric(best_rr['median_trades'], 0)}` and median PF `{_format_metric(best_rr['median_profit_factor'])}`.
- Which risk-per-trade level is most compatible with Topstep-style constraints? `{_format_metric(best_risk['risk_per_trade_pct'])}%` ranked best on the focused score, balancing target reach, drawdown, and trade count.
- Which final config is recommended? {recommendation_sentence}
- How quickly does it typically reach the `{prop_constraints.profit_target_usd:,.0f}` target, if at all? {target_speed_answer}
- What are the main caveats? Daily loss is approximated from realized daily PnL, max-loss feasibility is path-based rather than probabilistic, and the backtester keeps static account-size sizing instead of compounding.

## Recommendation

{chr(10).join(recommendation_lines)}

This candidate gave the best overall balance between prop-style survivability and robustness inside the tested branch. The main practical limitation is speed: the entire branch remains too slow to reach the evaluation target efficiently once subscription drag is considered.

## Caveats

- The daily loss rule is modeled as an optional research constraint on realized daily PnL. It does not attempt intraday trailing enforcement.
- Max loss feasibility is evaluated on the realized cumulative PnL path, not on a Monte Carlo distribution.
- Position sizing still uses the repo's fixed account-size reference per trade. This keeps the extension localized and comparable to prior runs.
- Restrictive ATR filters can improve some risk statistics while sharply reducing participation.

## Outputs

- Full results CSV: `{(output_dir / 'orb_campaign_results.csv').as_posix()}`
- Leaderboard CSV: `{(output_dir / 'orb_campaign_leaderboard.csv').as_posix()}`
- Markdown report: `{(output_dir / 'orb_campaign_report.md').as_posix()}`
- Plot directory: `{(output_dir / 'plots').as_posix()}`
{notebook_line}
"""


def save_top_plots(
    leaderboard: pd.DataFrame,
    experiments: dict[str, ORBExperiment],
    trades_by_name: dict[str, pd.DataFrame],
    output_dir: Path,
    max_plots: int = 5,
) -> None:
    """Generate equity and drawdown plots for top-ranked distinct variants."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    selected_names: list[str] = []
    for name in leaderboard["name"]:
        if name not in selected_names:
            selected_names.append(name)
        if len(selected_names) >= max_plots:
            break

    for axis_name in leaderboard["axis"].dropna().unique():
        axis_rows = leaderboard.loc[leaderboard["axis"] == axis_name, "name"]
        if not axis_rows.empty and axis_rows.iloc[0] not in selected_names:
            selected_names.append(axis_rows.iloc[0])

    for name in selected_names:
        trades = trades_by_name.get(name)
        experiment = experiments[name]
        if trades is None or trades.empty:
            continue

        equity = build_equity_curve(trades, initial_capital=experiment.initial_capital_usd)
        plot_equity_curve(equity)
        plt.savefig(plot_dir / f"{name}_equity.png", dpi=150, bbox_inches="tight")
        plt.close()

        plot_drawdown_curve(equity)
        plt.savefig(plot_dir / f"{name}_drawdown.png", dpi=150, bbox_inches="tight")
        plt.close()


def _prepare_datasets_for_experiments(
    dataset_path: Path,
    experiments: list[ORBExperiment],
    atr_regimes=None,
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[float | None, float | None]]]:
    """Load only the datasets and features required by the requested experiment list."""
    base_rth_df = prepare_base_rth_dataset(dataset_path)
    prepared_datasets: dict[str, pd.DataFrame] = {}

    current_experiments = [experiment for experiment in experiments if experiment.dataset_key == "current_1m_rth"]
    if current_experiments:
        ema_lengths = sorted({experiment.ema_length for experiment in current_experiments if experiment.ema_length})
        include_vwap = any(
            experiment.direction_filter_mode in ("vwap_only", "vwap_and_ema") for experiment in current_experiments
        )
        prepared_datasets["current_1m_rth"] = prepare_current_logic_dataset(
            base_rth_df,
            ema_lengths=ema_lengths,
            include_vwap=include_vwap,
        )

    if any(experiment.dataset_key == "paper_5m_rth" for experiment in experiments):
        prepared_datasets["paper_5m_rth"] = prepare_paper_dataset(base_rth_df)

    atr_bounds = {"none": (None, None)}
    if "current_1m_rth" in prepared_datasets:
        atr_bounds = resolve_atr_bounds(prepared_datasets["current_1m_rth"], atr_regimes)

    return prepared_datasets, atr_bounds


def _run_experiment_batch(
    dataset_path: Path,
    experiments: list[ORBExperiment],
    atr_regimes=None,
    prop_constraints: PropConstraintConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, ORBExperiment], dict[str, tuple[float | None, float | None]]]:
    """Execute a reusable batch of ORB experiments."""
    prepared_datasets, atr_bounds = _prepare_datasets_for_experiments(dataset_path, experiments, atr_regimes)
    session_dates = {name: _signal_session_dates(df) for name, df in prepared_datasets.items()}

    resolved_experiments = apply_resolved_atr_bounds(experiments, atr_bounds)
    experiments_by_name = {experiment.name: experiment for experiment in resolved_experiments}

    grouped: dict[tuple, list[ORBExperiment]] = defaultdict(list)
    profiles = build_execution_profiles()
    for experiment in resolved_experiments:
        tick_size = profiles[experiment.execution_profile].tick_size
        grouped[_strategy_group_key(experiment, tick_size)].append(experiment)

    result_rows: list[dict[str, object]] = []
    trades_by_name: dict[str, pd.DataFrame] = {}

    for grouped_experiments in grouped.values():
        anchor = grouped_experiments[0]
        tick_size = profiles[anchor.execution_profile].tick_size
        strategy = build_strategy(anchor, tick_size=tick_size)
        signal_df = strategy.generate_signals(prepared_datasets[anchor.dataset_key])

        for experiment in grouped_experiments:
            execution_model = _build_execution_model(experiment.execution_profile)
            account_size = (
                experiment.initial_capital_usd
                if experiment.risk_per_trade_pct is not None or experiment.max_leverage is not None
                else None
            )
            trades = run_backtest(
                signal_df,
                execution_model=execution_model,
                tick_value_usd=experiment.tick_value_usd,
                point_value_usd=experiment.point_value_usd,
                time_exit=experiment.time_exit,
                stop_buffer_ticks=experiment.stop_buffer_ticks,
                target_multiple=experiment.target_multiple,
                account_size_usd=account_size,
                risk_per_trade_pct=experiment.risk_per_trade_pct,
                entry_on_next_open=experiment.entry_on_next_open,
                max_leverage=experiment.max_leverage,
            )
            trades_by_name[experiment.name] = trades

            metrics = compute_metrics(
                trades,
                signal_df=signal_df,
                session_dates=session_dates[experiment.dataset_key],
                initial_capital=experiment.initial_capital_usd,
                prop_constraints=prop_constraints,
            )
            profile = profiles[experiment.execution_profile]
            row = {
                **asdict(experiment),
                "dataset_path": dataset_path.as_posix(),
                "commission_per_side_usd": profile.commission_per_side_usd,
                "slippage_ticks": profile.slippage_ticks,
                "execution_tick_size": profile.tick_size,
                **metrics,
            }
            if prop_constraints is not None:
                row.update(
                    {
                        "prop_config_name": prop_constraints.name,
                        "prop_config_account_size_usd": prop_constraints.account_size_usd,
                        "prop_config_max_loss_limit_usd": prop_constraints.max_loss_limit_usd,
                        "prop_config_daily_loss_limit_usd": prop_constraints.daily_loss_limit_usd,
                        "prop_config_profit_target_usd": prop_constraints.profit_target_usd,
                        "prop_config_monthly_subscription_cost_usd": prop_constraints.monthly_subscription_cost_usd,
                        "prop_config_trading_days_per_month": prop_constraints.trading_days_per_month,
                        "prop_config_daily_loss_limit_basis": prop_constraints.daily_loss_limit_basis,
                    }
                )
            result_rows.append(row)

    return pd.DataFrame(result_rows), trades_by_name, experiments_by_name, atr_bounds


def _build_notebook_cell(cell_type: str, source: str) -> dict[str, object]:
    if not source.endswith("\n"):
        source = source + "\n"
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def generate_validation_notebook(
    selected_experiment: ORBExperiment,
    leaderboard_row: pd.Series,
    dataset_path: Path,
    notebook_path: Path,
    prop_constraints: PropConstraintConfig,
) -> Path:
    """Create a runnable notebook for the final selected configuration only."""
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    experiment_literal = repr(asdict(selected_experiment))
    prop_literal = repr(asdict(prop_constraints))

    intro_md = f"""# ORB Final Validation Notebook

This notebook validates the final focused-campaign recommendation:

- Config: `{selected_experiment.name}`
- RR: `{_format_metric(selected_experiment.target_multiple, 0)}`
- EMA: `EMA{_format_metric(selected_experiment.ema_length, 0)}`
- ATR mode: `{selected_experiment.atr_regime}`
- Risk per trade: `{_format_metric(selected_experiment.risk_per_trade_pct)}%`
- Dataset: `{dataset_path.name}`

It is the strongest relative robustness candidate from the focused branch, but the companion report should still be consulted for the subscription-drag caveat.
"""

    conclusion_md = f"""## Conclusion

This configuration was selected because it led the focused robustness leaderboard, kept max drawdown at `{_format_metric(leaderboard_row.get('max_drawdown'))}`, and preserved a practical trade count of `{_format_metric(leaderboard_row.get('n_trades'), 0)}`. The main caveat from the campaign is still time-to-target: visual validation here is about confirming the best relative setup in the tested branch, not claiming that the branch is already subscription-efficient for a prop evaluation.
"""

    setup_code = """from pathlib import Path
import sys

root = Path.cwd().resolve()
while root != root.parent and not (root / "pyproject.toml").exists():
    root = root.parent

if not (root / "pyproject.toml").exists():
    raise RuntimeError("Could not locate the repository root from the current working directory.")

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

print(f"Project root: {root}")
"""

    imports_code = """from dataclasses import asdict, replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analytics.diagnostics import performance_by_month, performance_by_year
from src.analytics.metrics import compute_metrics
from src.analytics.orb_campaign import build_strategy, prepare_base_rth_dataset, prepare_current_logic_dataset, resolve_atr_bounds
from src.config.orb_campaign import ORBExperiment, PropConstraintConfig, build_execution_profiles, build_focused_atr_regimes
from src.engine.backtester import run_backtest
from src.engine.execution_model import ExecutionModel
from src.engine.portfolio import build_equity_curve
from src.visualization.equity import plot_drawdown_curve, plot_equity_curve
from src.visualization.plots import plot_trade_histogram

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 80)
"""

    config_code = f"""DATASET_PATH = root / "data" / "raw" / "{dataset_path.name}"
EXPERIMENT = ORBExperiment(**{experiment_literal})
PROP_CONSTRAINTS = PropConstraintConfig(**{prop_literal})

pd.DataFrame({{"field": list(asdict(EXPERIMENT).keys()), "value": list(asdict(EXPERIMENT).values())}})
"""

    run_code = """base_rth_df = prepare_base_rth_dataset(DATASET_PATH)
current_df = prepare_current_logic_dataset(
    base_rth_df,
    ema_lengths=[EXPERIMENT.ema_length] if EXPERIMENT.ema_length is not None else [],
    include_vwap=False,
)
atr_bounds = resolve_atr_bounds(current_df, build_focused_atr_regimes())
atr_min, atr_max = atr_bounds[EXPERIMENT.atr_regime]
experiment = replace(EXPERIMENT, atr_min=atr_min, atr_max=atr_max)

profiles = build_execution_profiles()
profile = profiles[experiment.execution_profile]
execution_model = ExecutionModel(
    commission_per_side_usd=profile.commission_per_side_usd,
    slippage_ticks=profile.slippage_ticks,
    tick_size=profile.tick_size,
)

strategy = build_strategy(experiment, tick_size=profile.tick_size)
signal_df = strategy.generate_signals(current_df)
session_dates = pd.Index(pd.to_datetime(current_df["session_date"]).dt.date.unique())

trades = run_backtest(
    signal_df,
    execution_model=execution_model,
    tick_value_usd=experiment.tick_value_usd,
    point_value_usd=experiment.point_value_usd,
    time_exit=experiment.time_exit,
    stop_buffer_ticks=experiment.stop_buffer_ticks,
    target_multiple=experiment.target_multiple,
    account_size_usd=experiment.initial_capital_usd,
    risk_per_trade_pct=experiment.risk_per_trade_pct,
    entry_on_next_open=experiment.entry_on_next_open,
    max_leverage=experiment.max_leverage,
)

metrics = compute_metrics(
    trades,
    signal_df=signal_df,
    session_dates=session_dates,
    initial_capital=experiment.initial_capital_usd,
    prop_constraints=PROP_CONSTRAINTS,
)
equity = build_equity_curve(trades, initial_capital=experiment.initial_capital_usd)
"""

    metrics_code = """key_metric_order = [
    "n_trades",
    "win_rate",
    "avg_win",
    "avg_loss",
    "expectancy",
    "profit_factor",
    "cumulative_pnl",
    "max_drawdown",
    "sharpe_ratio",
    "longest_loss_streak",
    "average_loss_streak_length",
    "count_loss_streaks_2_plus",
    "worst_trade",
    "worst_day",
    "stop_hit_rate",
    "target_hit_rate",
    "eod_exit_rate",
    "percent_of_days_traded",
    "days_to_profit_target",
    "estimated_months_to_profit_target",
    "profit_target_reached_before_max_loss",
    "any_daily_loss_limit_breach",
    "number_of_daily_loss_limit_breaches",
    "subscription_drag_estimate",
    "estimated_pnl_after_subscription",
]

metrics_table = pd.DataFrame([{"metric": metric, "value": metrics.get(metric)} for metric in key_metric_order])
metrics_table
"""

    plots_code = """if trades.empty:
    print("No trades were generated for the selected configuration.")
else:
    plot_equity_curve(equity)
    plot_drawdown_curve(equity)
    plt.show()

    plot_trade_histogram(trades)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(trades["net_pnl_usd"], bins=30)
    plt.title("Trade PnL Distribution")
    plt.xlabel("Net PnL (USD)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
"""

    streaks_code = """def loss_streak_lengths(pnl: pd.Series) -> list[int]:
    streaks = []
    current = 0
    for value in pnl:
        if value < 0:
            current += 1
        elif current > 0:
            streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return streaks

streaks = loss_streak_lengths(trades["net_pnl_usd"]) if not trades.empty else []
if not streaks:
    print("No losing streaks to plot.")
else:
    bins = range(1, max(streaks) + 2)
    plt.figure(figsize=(8, 4))
    plt.hist(streaks, bins=bins, align="left", rwidth=0.85)
    plt.title("Losing Streak Distribution")
    plt.xlabel("Consecutive losses")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
"""

    breakdown_code = """if trades.empty:
    print("No monthly or yearly breakdown available.")
else:
    monthly = performance_by_month(trades)
    yearly = performance_by_year(trades)
    display(monthly.tail(24))
    display(yearly)
"""

    sample_trade_code = """def plot_sample_trade_windows(price_df: pd.DataFrame, trades_df: pd.DataFrame, sample_count: int = 3) -> None:
    if trades_df.empty:
        print("No sample trades to visualize.")
        return

    ordered = trades_df.sort_values("entry_time").reset_index(drop=True)
    positions = sorted(set(np.linspace(0, len(ordered) - 1, num=min(sample_count, len(ordered)), dtype=int)))

    for position in positions:
        trade = ordered.iloc[position]
        window = price_df.loc[
            (price_df["timestamp"] >= trade["entry_time"] - pd.Timedelta(minutes=20))
            & (price_df["timestamp"] <= trade["exit_time"] + pd.Timedelta(minutes=40))
        ].copy()
        if window.empty:
            continue

        plt.figure(figsize=(12, 4))
        plt.plot(window["timestamp"], window["close"], label="Close", linewidth=1.25, color="#111827")
        plt.axvline(trade["entry_time"], color="#0f766e", linestyle="--", label="Entry")
        plt.axvline(trade["exit_time"], color="#b91c1c", linestyle="--", label="Exit")
        plt.axhline(trade["entry_price"], color="#0f766e", linewidth=1.0, alpha=0.8, label="Entry price")
        plt.axhline(trade["stop_price"], color="#dc2626", linewidth=1.0, alpha=0.8, label="Stop")
        plt.axhline(trade["target_price"], color="#2563eb", linewidth=1.0, alpha=0.8, label="Target")

        nearest_idx = (window["timestamp"] - trade["entry_time"]).abs().idxmin()
        entry_bar = window.loc[nearest_idx]
        if pd.notna(entry_bar.get("or_high")):
            plt.axhline(entry_bar["or_high"], color="#7c3aed", linewidth=0.9, alpha=0.6, label="OR high")
        if pd.notna(entry_bar.get("or_low")):
            plt.axhline(entry_bar["or_low"], color="#f59e0b", linewidth=0.9, alpha=0.6, label="OR low")

        plt.title(
            f"Trade {int(trade['trade_id'])} | {trade['session_date']} | {trade['direction']} | {trade['exit_reason']}"
        )
        plt.ylabel("Price")
        plt.xticks(rotation=45)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

plot_sample_trade_windows(current_df, trades, sample_count=3)
"""

    notebook = {
        "cells": [
            _build_notebook_cell("markdown", intro_md),
            _build_notebook_cell("code", setup_code),
            _build_notebook_cell("code", imports_code),
            _build_notebook_cell("code", config_code),
            _build_notebook_cell("code", run_code),
            _build_notebook_cell("code", metrics_code),
            _build_notebook_cell("code", plots_code),
            _build_notebook_cell("code", streaks_code),
            _build_notebook_cell("code", breakdown_code),
            _build_notebook_cell("code", sample_trade_code),
            _build_notebook_cell("markdown", conclusion_md),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return notebook_path


def run_campaign(dataset_path: Path, output_dir: Path) -> dict[str, Path]:
    """Execute the full structured ORB campaign and export the artifacts."""
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    results, trades_by_name, experiments_by_name, atr_bounds = _run_experiment_batch(
        dataset_path=dataset_path,
        experiments=build_orb_experiments(dataset_path=dataset_path),
        atr_regimes=build_atr_regimes(),
    )
    leaderboard = score_experiments(results, build_ranking_config())

    results_path = output_dir / "orb_campaign_results.csv"
    leaderboard_path = output_dir / "orb_campaign_leaderboard.csv"
    report_path = output_dir / "orb_campaign_report.md"

    results.to_csv(results_path, index=False)
    leaderboard.to_csv(leaderboard_path, index=False)
    report_path.write_text(
        generate_report(
            results=results,
            leaderboard=leaderboard,
            atr_bounds=atr_bounds,
            ranking=build_ranking_config(),
            dataset_path=dataset_path,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )
    save_top_plots(leaderboard, experiments_by_name, trades_by_name, output_dir)

    return {
        "results_csv": results_path,
        "leaderboard_csv": leaderboard_path,
        "report_md": report_path,
        "plot_dir": output_dir / "plots",
    }


def run_focused_campaign(
    dataset_path: Path,
    output_dir: Path,
    notebook_path: Path | None = None,
) -> dict[str, Path]:
    """Execute the focused long-only EMA campaign and export the artifacts."""
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)

    prop_constraints = build_prop_constraints()
    ranking = build_focused_ranking_config()
    experiments = build_focused_orb_experiments(dataset_path=dataset_path)

    results, trades_by_name, experiments_by_name, atr_bounds = _run_experiment_batch(
        dataset_path=dataset_path,
        experiments=experiments,
        atr_regimes=build_focused_atr_regimes(),
        prop_constraints=prop_constraints,
    )
    leaderboard = score_focused_experiments(results, ranking, prop_constraints)

    results_path = output_dir / "orb_campaign_results.csv"
    leaderboard_path = output_dir / "orb_campaign_leaderboard.csv"
    report_path = output_dir / "orb_campaign_report.md"

    results.to_csv(results_path, index=False)
    leaderboard.to_csv(leaderboard_path, index=False)

    generated_notebook_path = None
    if notebook_path is not None and not leaderboard.empty:
        selected_name = str(leaderboard.iloc[0]["name"])
        generated_notebook_path = generate_validation_notebook(
            selected_experiment=experiments_by_name[selected_name],
            leaderboard_row=leaderboard.iloc[0],
            dataset_path=dataset_path,
            notebook_path=notebook_path,
            prop_constraints=prop_constraints,
        )

    report_path.write_text(
        generate_focused_report(
            results=results,
            leaderboard=leaderboard,
            atr_bounds=atr_bounds,
            ranking=ranking,
            dataset_path=dataset_path,
            output_dir=output_dir,
            prop_constraints=prop_constraints,
            notebook_path=generated_notebook_path,
        ),
        encoding="utf-8",
    )
    save_top_plots(leaderboard, experiments_by_name, trades_by_name, output_dir)

    artifacts = {
        "results_csv": results_path,
        "leaderboard_csv": leaderboard_path,
        "report_md": report_path,
        "plot_dir": output_dir / "plots",
    }
    if generated_notebook_path is not None:
        artifacts["validation_notebook"] = generated_notebook_path
    return artifacts


def main() -> None:
    """CLI entry point for the structured ORB campaign."""
    parser = argparse.ArgumentParser(description="Run ORB research campaigns.")
    parser.add_argument(
        "--campaign",
        choices=["focused_prop", "legacy_structured"],
        default="focused_prop",
        help="Campaign profile to run.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_CAMPAIGN_DATASET,
        help="Path to the source OHLCV dataset (.csv or .parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults depend on the chosen campaign.",
    )
    parser.add_argument(
        "--notebook-path",
        type=Path,
        default=None,
        help="Optional notebook path for the focused validation notebook.",
    )
    args = parser.parse_args()

    if args.campaign == "focused_prop":
        output_dir = args.output_dir or (EXPORTS_DIR / "orb_campaign_focused_prop")
        notebook_path = args.notebook_path or (NOTEBOOKS_DIR / "orb_topstep_validation.ipynb")
        artifacts = run_focused_campaign(
            dataset_path=args.dataset,
            output_dir=output_dir,
            notebook_path=notebook_path,
        )
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir or (EXPORTS_DIR / f"orb_campaign_{timestamp}")
        artifacts = run_campaign(dataset_path=args.dataset, output_dir=output_dir)

    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
