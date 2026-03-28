"""Full intraday mean reversion research campaign."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.vwap_validation import (
    ChallengeScenario,
    StressScenario,
    _apply_cost_stress_overlay,
    _build_scope_summary_table,
    _challenge_empirical_summary,
    _json_dump,
    _notebook_cell,
    _rebuild_daily_results_from_trades,
    _split_sessions,
)
from src.config.mean_reversion_campaign import (
    MeanReversionCampaignSpec,
    MeanReversionVariantConfig,
    TimeframeDefinition,
    build_default_mean_reversion_campaign_spec,
    build_default_mean_reversion_variants,
)
from src.config.paths import NOTEBOOKS_DIR, ensure_directories
from src.data.cleaning import clean_ohlcv
from src.data.loader import load_ohlcv_file
from src.data.resampling import build_resampled_output_path, resample_ohlcv
from src.engine.backtester import run_backtest
from src.engine.mean_reversion_backtester import MeanReversionBacktestResult, run_mean_reversion_backtest
from src.engine.vwap_backtester import InstrumentDetails, build_execution_model_for_profile
from src.features.opening_range import compute_opening_range
from src.strategy.mean_reversion import build_mean_reversion_signal_frame, prepare_mean_reversion_feature_frame
from src.strategy.orb import ORBStrategy


VWAP_PV_TYPICAL_COL = "vwap_pv_typical"
CAMPAIGN_PHASES = ("screening", "validation", "portfolio", "notebook", "full")
DEFAULT_STRESS_SCENARIO = StressScenario(
    name="slippage_x2",
    slippage_multiplier=2.0,
    notes="Primary mandatory stress: slippage doubled.",
)
STANDARD_CHALLENGE = ChallengeScenario(
    name="scenario_standard",
    label="standard",
    risk_per_trade_pct=1.0,
    max_contracts=2,
    stop_after_losses_in_day=2,
    daily_loss_limit_usd=1_000.0,
    trailing_drawdown_limit_usd=2_000.0,
    profit_target_usd=3_000.0,
    horizon_days=20,
    deleverage_after_red_days=2,
    deleverage_factor=0.5,
)


@dataclass
class PreparedTimeframeData:
    """Prepared feature dataset for one symbol and timeframe."""

    symbol: str
    timeframe: TimeframeDefinition
    dataset_path: Path
    feature_df: pd.DataFrame


@dataclass
class MeanReversionEvaluation:
    """In-memory evaluation bundle for one variant."""

    variant: MeanReversionVariantConfig
    signal_df: pd.DataFrame
    result: MeanReversionBacktestResult
    instrument: InstrumentDetails
    execution_model: Any
    all_sessions: list
    is_sessions: list
    oos_sessions: list
    summary_by_scope: pd.DataFrame
    tables_by_scope: dict[str, dict[str, pd.DataFrame]]


@dataclass
class ORBEvaluation:
    """Reference ORB bundle used in portfolio comparison."""

    label: str
    symbol: str
    timeframe: str
    trades: pd.DataFrame
    daily_results: pd.DataFrame
    summary_by_scope: pd.DataFrame
    instrument: InstrumentDetails
    execution_model: Any
    all_sessions: list
    is_sessions: list
    oos_sessions: list


def _safe_float(value: Any, default: float = 0.0) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float(default)
    return float(numeric)


def _scope_row(summary_by_scope: pd.DataFrame, scope: str) -> pd.Series:
    return summary_by_scope.loc[summary_by_scope["scope"] == scope].iloc[0]


def _positive_month_ratio(daily_results: pd.DataFrame) -> float:
    if daily_results.empty:
        return 0.0
    monthly = daily_results.copy()
    monthly["period"] = pd.to_datetime(monthly["session_date"]).dt.to_period("M").astype(str)
    month_pnl = monthly.groupby("period")["daily_pnl_usd"].sum()
    if month_pnl.empty:
        return 0.0
    return float((month_pnl > 0).mean())


def _screening_score(oos_row: pd.Series, positive_month_ratio: float) -> float:
    pnl = _safe_float(oos_row.get("net_pnl"))
    pf = _safe_float(oos_row.get("profit_factor"), default=1.0)
    sharpe = _safe_float(oos_row.get("sharpe_ratio"))
    expectancy = _safe_float(oos_row.get("expectancy_per_trade"))
    max_dd = abs(_safe_float(oos_row.get("max_drawdown")))
    concentration = _safe_float(oos_row.get("top_5_day_contribution_pct"))
    trades = _safe_float(oos_row.get("total_trades"))
    return float(
        2.0 * np.tanh((pf - 1.0) / 0.15)
        + 1.5 * np.tanh(sharpe / 1.0)
        + 0.9 * np.tanh(expectancy / 25.0)
        + 0.6 * np.tanh(pnl / 2_500.0)
        + 0.5 * np.tanh(trades / 80.0)
        + 0.8 * positive_month_ratio
        - 1.1 * np.tanh(max_dd / 3_000.0)
        - 1.2 * concentration
    )


def _validation_score(
    oos_row: pd.Series,
    stress_oos_row: pd.Series,
    split_positive_rate: float,
    challenge_success_rate: float,
    positive_month_ratio: float,
) -> float:
    return float(
        _screening_score(oos_row, positive_month_ratio)
        + 0.9 * np.tanh((_safe_float(stress_oos_row.get("net_pnl")) - 0.0) / 2_500.0)
        + 0.8 * np.tanh((_safe_float(stress_oos_row.get("profit_factor")) - 1.0) / 0.15)
        + 1.0 * split_positive_rate
        + 0.8 * challenge_success_rate
    )


def _pass_screening(row: pd.Series) -> bool:
    return bool(
        (_safe_float(row.get("oos_net_pnl")) > 0.0)
        and (_safe_float(row.get("oos_profit_factor")) > 1.02)
        and (_safe_float(row.get("oos_sharpe_ratio")) > 0.10)
        and (_safe_float(row.get("oos_total_trades")) >= _safe_float(row.get("min_oos_trades")))
        and (_safe_float(row.get("oos_top_5_day_contribution_pct"), default=1.0) < 0.80)
        and (_safe_float(row.get("oos_positive_month_ratio")) >= 0.35)
    )


def _screening_verdict(pass_count: int, total_count: int) -> str:
    if total_count <= 0 or pass_count == 0:
        return "famille morte"
    if pass_count / max(total_count, 1) >= 0.40:
        return "famille vivante"
    return "famille fragile"


def _validation_verdict(
    oos_row: pd.Series,
    stress_oos_row: pd.Series,
    split_positive_rate: float,
    challenge_success_rate: float,
    positive_month_ratio: float,
) -> str:
    if _safe_float(oos_row.get("net_pnl")) <= 0.0 or _safe_float(oos_row.get("profit_factor")) <= 1.0:
        return "dead"
    if _safe_float(stress_oos_row.get("net_pnl")) <= 0.0 or _safe_float(stress_oos_row.get("profit_factor")) <= 1.0:
        return "fragile"
    if split_positive_rate < 0.50:
        return "fragile"
    if challenge_success_rate >= 0.20 and positive_month_ratio >= 0.45:
        return "survivor"
    return "portfolio_candidate"


def _enrich_with_price_volume(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    out[VWAP_PV_TYPICAL_COL] = typical * out["volume"].fillna(0.0)
    return out


def _variants_for_key(
    variants: list[MeanReversionVariantConfig],
    symbol: str,
    timeframe: str,
) -> list[MeanReversionVariantConfig]:
    return [variant for variant in variants if variant.symbol == symbol and variant.timeframe == timeframe]


def _parse_window_from_source(source: str, prefix: str) -> int | None:
    if source.startswith(prefix):
        try:
            return int(source.split("_")[-1])
        except ValueError:
            return None
    return None


def _prepare_timeframe_data(
    symbol: str,
    timeframe: TimeframeDefinition,
    dataset_path: Path,
    variants: list[MeanReversionVariantConfig],
    spec: MeanReversionCampaignSpec,
    output_dir: Path,
) -> PreparedTimeframeData:
    raw = load_ohlcv_file(dataset_path)
    clean = clean_ohlcv(raw)
    enriched = _enrich_with_price_volume(clean)
    resampled = resample_ohlcv(
        enriched,
        rule=timeframe.resample_rule,
        aggregation_overrides={VWAP_PV_TYPICAL_COL: "sum"},
    )
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = build_resampled_output_path(dataset_path, rule=timeframe.resample_rule, output_dir=data_dir)
    parquet_df = resampled.set_index("timestamp")
    parquet_df.index.name = "timestamp"
    parquet_df.to_parquet(output_path)

    atr_windows = sorted({int(variant.atr_period) for variant in variants})
    ema_windows = sorted(
        {
            int(value)
            for variant in variants
            for value in (
                variant.ema_window,
                variant.ema_filter_window,
                _parse_window_from_source(variant.target_source, "ema_"),
            )
            if value is not None and int(value) > 0
        }
    )
    zscore_windows = sorted(
        {
            int(value)
            for variant in variants
            for value in (
                variant.zscore_window,
                variant.bollinger_window,
                _parse_window_from_source(variant.target_source, "rolling_mean_"),
            )
            if value is not None and int(value) > 0
        }
    )
    bollinger_windows = sorted({int(variant.bollinger_window) for variant in variants if variant.bollinger_window is not None})
    rsi_windows = sorted({int(variant.oscillator_period) for variant in variants if variant.oscillator_kind == "rsi" and variant.oscillator_period is not None})
    stochastic_defs = sorted(
        {
            (
                int(variant.oscillator_period_fast),
                int(variant.oscillator_period_slow),
                int(variant.oscillator_smoothing),
            )
            for variant in variants
            if variant.oscillator_kind == "stochastic"
            and variant.oscillator_period_fast is not None
            and variant.oscillator_period_slow is not None
            and variant.oscillator_smoothing is not None
        }
    )
    adx_periods = sorted({int(variant.adx_period) for variant in variants if variant.adx_period is not None})
    opening_windows = sorted({int(variant.opening_window_minutes) for variant in variants if variant.opening_window_minutes is not None})
    persistent_lookbacks = sorted({int(variant.persistent_lookback) for variant in variants})
    ema_slope_specs = sorted({(int(variant.ema_filter_window), int(variant.ema_slope_lookback)) for variant in variants})
    vwap_slope_lookbacks = sorted({int(variant.vwap_slope_lookback) for variant in variants})

    feature_df = prepare_mean_reversion_feature_frame(
        resampled,
        session_start=spec.session_start,
        session_end=spec.session_end,
        atr_windows=atr_windows,
        ema_windows=ema_windows,
        zscore_windows=zscore_windows,
        bollinger_windows=bollinger_windows,
        rsi_windows=rsi_windows,
        stochastic_defs=stochastic_defs,
        adx_periods=adx_periods,
        opening_windows=opening_windows,
        persistent_lookbacks=persistent_lookbacks,
        ema_slope_specs=ema_slope_specs,
        vwap_slope_lookbacks=vwap_slope_lookbacks,
        vwap_price_volume_col=VWAP_PV_TYPICAL_COL,
    )
    return PreparedTimeframeData(
        symbol=symbol,
        timeframe=timeframe,
        dataset_path=output_path,
        feature_df=feature_df,
    )


def _evaluate_variant(
    prepared: PreparedTimeframeData,
    variant: MeanReversionVariantConfig,
    spec: MeanReversionCampaignSpec,
) -> MeanReversionEvaluation:
    signal_df = build_mean_reversion_signal_frame(prepared.feature_df, variant)
    execution_model, instrument = build_execution_model_for_profile(variant.symbol, "repo_realistic")
    result = run_mean_reversion_backtest(
        signal_df=signal_df,
        variant=variant,
        execution_model=execution_model,
        instrument=instrument,
        account_size_usd=float(spec.prop_constraints.account_size_usd),
    )
    all_sessions = sorted(pd.to_datetime(prepared.feature_df["session_date"]).dt.date.unique())
    is_sessions, oos_sessions = _split_sessions(all_sessions, spec.is_fraction)
    summary_by_scope, _, tables_by_scope = _build_scope_summary_table(
        trades=result.trades,
        daily_results=result.daily_results,
        bar_results=result.bar_results,
        signal_df=signal_df,
        sessions_all=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        initial_capital=float(spec.prop_constraints.account_size_usd),
        constraints=spec.prop_constraints,
        instrument=instrument,
        execution_model=execution_model,
        rolling_window_days=20,
    )
    return MeanReversionEvaluation(
        variant=variant,
        signal_df=signal_df,
        result=result,
        instrument=instrument,
        execution_model=execution_model,
        all_sessions=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        summary_by_scope=summary_by_scope,
        tables_by_scope=tables_by_scope,
    )


def _screening_row(evaluation: MeanReversionEvaluation, prepared: PreparedTimeframeData) -> dict[str, Any]:
    overall = _scope_row(evaluation.summary_by_scope, "overall")
    is_row = _scope_row(evaluation.summary_by_scope, "is")
    oos = _scope_row(evaluation.summary_by_scope, "oos")
    overall_positive_month_ratio = _positive_month_ratio(evaluation.result.daily_results)
    oos_daily = evaluation.result.daily_results.loc[
        pd.to_datetime(evaluation.result.daily_results["session_date"]).dt.date.isin(set(evaluation.oos_sessions))
    ].copy()
    oos_positive_month_ratio = _positive_month_ratio(oos_daily)
    row = {
        "name": evaluation.variant.name,
        "family": evaluation.variant.family,
        "symbol": evaluation.variant.symbol,
        "timeframe": evaluation.variant.timeframe,
        "bar_minutes": prepared.timeframe.bar_minutes,
        "dataset_path": str(prepared.dataset_path),
        "min_oos_trades": evaluation.variant.min_oos_trades,
        "overall_net_pnl": overall["net_pnl"],
        "overall_profit_factor": overall["profit_factor"],
        "overall_sharpe_ratio": overall["sharpe_ratio"],
        "overall_sortino_ratio": overall["sortino_ratio"],
        "overall_max_drawdown": overall["max_drawdown"],
        "overall_total_trades": overall["total_trades"],
        "overall_expectancy_per_trade": overall["expectancy_per_trade"],
        "overall_avg_holding_time_min": overall["avg_time_in_position_min"],
        "overall_time_in_market_pct": overall["avg_exposure_pct"],
        "overall_top_5_day_contribution_pct": overall["top_5_day_contribution_pct"],
        "overall_positive_month_ratio": overall_positive_month_ratio,
        "is_net_pnl": is_row["net_pnl"],
        "is_profit_factor": is_row["profit_factor"],
        "is_sharpe_ratio": is_row["sharpe_ratio"],
        "oos_net_pnl": oos["net_pnl"],
        "oos_profit_factor": oos["profit_factor"],
        "oos_sharpe_ratio": oos["sharpe_ratio"],
        "oos_sortino_ratio": oos["sortino_ratio"],
        "oos_max_drawdown": oos["max_drawdown"],
        "oos_total_trades": oos["total_trades"],
        "oos_expectancy_per_trade": oos["expectancy_per_trade"],
        "oos_avg_holding_time_min": oos["avg_time_in_position_min"],
        "oos_time_in_market_pct": oos["avg_exposure_pct"],
        "oos_top_5_day_contribution_pct": oos["top_5_day_contribution_pct"],
        "oos_positive_month_ratio": oos_positive_month_ratio,
        "notes": evaluation.variant.notes,
    }
    row["screening_score"] = _screening_score(oos, oos_positive_month_ratio)
    row["pass_screening"] = _pass_screening(pd.Series(row))
    return row


def _variant_catalog_rows(variants: list[MeanReversionVariantConfig]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, variant in enumerate(variants, start=1):
        rows.append(
            {
                "display_order": order,
                "name": variant.name,
                "family": variant.family,
                "symbol": variant.symbol,
                "timeframe": variant.timeframe,
                "fixed_quantity": variant.fixed_quantity,
                "max_trades_per_day": variant.max_trades_per_day,
                "min_oos_trades": variant.min_oos_trades,
                "target_source": variant.target_source,
                "stop_atr_multiple": variant.stop_atr_multiple,
                "timeout_bars": variant.timeout_bars,
                "notes": variant.notes,
            }
        )
    return rows


def _select_survivors(screening_df: pd.DataFrame, spec: MeanReversionCampaignSpec) -> pd.DataFrame:
    if screening_df.empty:
        return screening_df.copy()
    passing = screening_df.loc[screening_df["pass_screening"]].copy()
    if passing.empty:
        return passing
    passing = passing.sort_values(
        ["screening_score", "oos_profit_factor", "oos_sharpe_ratio", "oos_net_pnl"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    family_winners = (
        passing.groupby(["symbol", "timeframe", "family"], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    if len(family_winners) >= spec.max_validation_survivors:
        return family_winners.head(spec.max_validation_survivors).reset_index(drop=True)
    remaining = passing.loc[~passing["name"].isin(set(family_winners["name"]))].copy()
    slots = max(int(spec.max_validation_survivors) - len(family_winners), 0)
    filler = remaining.head(slots)
    return pd.concat([family_winners, filler], ignore_index=True).reset_index(drop=True)


def run_screening(
    spec: MeanReversionCampaignSpec,
    output_dir: Path,
    universe: list[MeanReversionVariantConfig] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], PreparedTimeframeData]]:
    """Run the full screening phase and export the research catalog."""
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    screening_dir = output_dir / "screening"
    screening_dir.mkdir(parents=True, exist_ok=True)

    variants = universe or build_default_mean_reversion_variants()
    catalog_df = pd.DataFrame(_variant_catalog_rows(variants))
    catalog_df.to_csv(screening_dir / "variant_catalog.csv", index=False)

    prepared_cache: dict[tuple[str, str], PreparedTimeframeData] = {}
    rows: list[dict[str, Any]] = []

    for symbol, dataset_path in spec.datasets_by_symbol.items():
        for timeframe in spec.timeframes:
            local_variants = _variants_for_key(variants, symbol, timeframe.label)
            if not local_variants:
                continue
            prepared = _prepare_timeframe_data(
                symbol=symbol,
                timeframe=timeframe,
                dataset_path=dataset_path,
                variants=local_variants,
                spec=spec,
                output_dir=output_dir,
            )
            prepared_cache[(symbol, timeframe.label)] = prepared
            for variant in local_variants:
                evaluation = _evaluate_variant(prepared, variant, spec)
                rows.append(_screening_row(evaluation, prepared))

    screening_df = pd.DataFrame(rows)
    if not screening_df.empty:
        screening_df = screening_df.sort_values(
            ["screening_score", "oos_profit_factor", "oos_sharpe_ratio"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    survivors_df = _select_survivors(screening_df, spec)

    family_summary = (
        screening_df.groupby(["family"], as_index=False)
        .agg(
            total_variants=("name", "count"),
            pass_screening_count=("pass_screening", "sum"),
            best_name=("name", "first"),
            best_oos_profit_factor=("oos_profit_factor", "max"),
            best_oos_sharpe_ratio=("oos_sharpe_ratio", "max"),
        )
        if not screening_df.empty
        else pd.DataFrame(columns=["family", "total_variants", "pass_screening_count"])
    )
    if not family_summary.empty:
        family_summary["screening_verdict"] = family_summary.apply(
            lambda row: _screening_verdict(int(row["pass_screening_count"]), int(row["total_variants"])),
            axis=1,
        )

    screening_df.to_csv(screening_dir / "screening_results.csv", index=False)
    survivors_df.to_csv(screening_dir / "screening_survivors.csv", index=False)
    family_summary.to_csv(screening_dir / "family_summary.csv", index=False)
    _json_dump(
        screening_dir / "run_metadata.json",
        {
            "run_timestamp": datetime.now().isoformat(),
            "datasets_by_symbol": spec.datasets_by_symbol,
            "timeframes": [asdict(timeframe) for timeframe in spec.timeframes],
            "variant_count": len(variants),
            "survivor_count": len(survivors_df),
        },
    )

    summary_lines = [
        "# Mean Reversion Screening",
        "",
        f"- Variants screened: {len(screening_df)}",
        f"- Survivors retained for validation: {len(survivors_df)}",
        "",
        "## Family Verdicts",
        "",
        "```text",
        family_summary.to_string(index=False) if not family_summary.empty else "No family summary available.",
        "```",
        "",
        "## Top Screening Rows",
        "",
        "```text",
        screening_df[
            [
                "name",
                "family",
                "symbol",
                "timeframe",
                "oos_net_pnl",
                "oos_profit_factor",
                "oos_sharpe_ratio",
                "oos_total_trades",
                "oos_top_5_day_contribution_pct",
                "oos_positive_month_ratio",
                "screening_score",
                "pass_screening",
            ]
        ].head(20).to_string(index=False)
        if not screening_df.empty
        else "No screening row available.",
        "```",
        "",
    ]
    (screening_dir / "screening_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    return screening_df, survivors_df, prepared_cache


def _subset_oos_trades(evaluation: MeanReversionEvaluation) -> pd.DataFrame:
    if evaluation.result.trades.empty:
        return evaluation.result.trades.copy()
    oos_set = set(evaluation.oos_sessions)
    view = evaluation.result.trades.copy()
    view["session_date"] = pd.to_datetime(view["session_date"]).dt.date
    return view.loc[view["session_date"].isin(oos_set)].copy().reset_index(drop=True)


def _split_stability_table(evaluation: MeanReversionEvaluation, spec: MeanReversionCampaignSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_fraction in spec.split_fractions:
        is_sessions, oos_sessions = _split_sessions(evaluation.all_sessions, split_fraction)
        summary_by_scope, _, _ = _build_scope_summary_table(
            trades=evaluation.result.trades,
            daily_results=evaluation.result.daily_results,
            bar_results=evaluation.result.bar_results,
            signal_df=evaluation.signal_df,
            sessions_all=evaluation.all_sessions,
            is_sessions=is_sessions,
            oos_sessions=oos_sessions,
            initial_capital=float(spec.prop_constraints.account_size_usd),
            constraints=spec.prop_constraints,
            instrument=evaluation.instrument,
            execution_model=evaluation.execution_model,
            rolling_window_days=20,
        )
        oos = _scope_row(summary_by_scope, "oos")
        rows.append(
            {
                "split_fraction": split_fraction,
                "oos_net_pnl": oos["net_pnl"],
                "oos_profit_factor": oos["profit_factor"],
                "oos_sharpe_ratio": oos["sharpe_ratio"],
                "oos_total_trades": oos["total_trades"],
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["positive_split"] = (table["oos_net_pnl"] > 0.0) & (table["oos_profit_factor"] > 1.0)
    return table


def _period_stability_table(
    evaluation: MeanReversionEvaluation,
    account_size_usd: float,
) -> pd.DataFrame:
    daily = evaluation.result.daily_results.copy()
    if daily.empty:
        return pd.DataFrame(
            columns=["period", "net_pnl", "trades", "sharpe_like", "positive_days_ratio", "positive_month_ratio"]
        )
    daily["session_date"] = pd.to_datetime(daily["session_date"])
    oos_set = set(evaluation.oos_sessions)
    daily = daily.loc[daily["session_date"].dt.date.isin(oos_set)].copy()
    if daily.empty:
        return pd.DataFrame()

    trades = evaluation.result.trades.copy()
    trades["session_date"] = pd.to_datetime(trades["session_date"])
    trades = trades.loc[trades["session_date"].dt.date.isin(oos_set)].copy()
    daily["period"] = daily["session_date"].dt.to_period("Y").astype(str)

    rows: list[dict[str, Any]] = []
    for period, period_daily in daily.groupby("period", sort=True):
        returns = pd.to_numeric(period_daily["daily_pnl_usd"], errors="coerce").fillna(0.0) / float(max(account_size_usd, 1.0))
        std = float(returns.std(ddof=0))
        sharpe_like = float((returns.mean() / std) * math.sqrt(252.0)) if std > 0 else 0.0
        trade_count = int(trades.loc[trades["session_date"].dt.to_period("Y").astype(str) == period].shape[0])
        rows.append(
            {
                "period": period,
                "net_pnl": float(pd.to_numeric(period_daily["daily_pnl_usd"], errors="coerce").sum()),
                "trades": trade_count,
                "sharpe_like": sharpe_like,
                "positive_days_ratio": float((pd.to_numeric(period_daily["daily_pnl_usd"], errors="coerce") > 0).mean()),
                "positive_month_ratio": _positive_month_ratio(period_daily),
            }
        )
    return pd.DataFrame(rows)


def run_validation(
    spec: MeanReversionCampaignSpec,
    output_dir: Path,
    survivors_df: pd.DataFrame,
    prepared_cache: dict[tuple[str, str], PreparedTimeframeData],
) -> tuple[pd.DataFrame, dict[str, MeanReversionEvaluation]]:
    """Validate only the screening survivors with stricter diagnostics."""
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    variants_by_name = {variant.name: variant for variant in build_default_mean_reversion_variants()}
    evaluation_map: dict[str, MeanReversionEvaluation] = {}
    rows: list[dict[str, Any]] = []

    for _, survivor in survivors_df.iterrows():
        variant = variants_by_name[str(survivor["name"])]
        prepared = prepared_cache[(variant.symbol, variant.timeframe)]
        evaluation = _evaluate_variant(prepared, variant, spec)
        evaluation_map[variant.name] = evaluation
        variant_dir = validation_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)

        stressed_trades = _apply_cost_stress_overlay(
            evaluation.result.trades,
            scenario=DEFAULT_STRESS_SCENARIO,
            instrument=evaluation.instrument,
            execution_model=evaluation.execution_model,
            session_start=spec.session_start,
        )
        stressed_daily = _rebuild_daily_results_from_trades(
            stressed_trades,
            all_sessions=evaluation.all_sessions,
            initial_capital=float(spec.prop_constraints.account_size_usd),
        )
        stressed_summary_by_scope, _, _ = _build_scope_summary_table(
            trades=stressed_trades,
            daily_results=stressed_daily,
            bar_results=pd.DataFrame(),
            signal_df=evaluation.signal_df,
            sessions_all=evaluation.all_sessions,
            is_sessions=evaluation.is_sessions,
            oos_sessions=evaluation.oos_sessions,
            initial_capital=float(spec.prop_constraints.account_size_usd),
            constraints=spec.prop_constraints,
            instrument=evaluation.instrument,
            execution_model=evaluation.execution_model,
            rolling_window_days=20,
        )
        stress_oos = _scope_row(stressed_summary_by_scope, "oos")
        split_table = _split_stability_table(evaluation, spec)
        split_positive_rate = float(split_table["positive_split"].mean()) if not split_table.empty else 0.0
        challenge_summary, _ = _challenge_empirical_summary(
            _subset_oos_trades(evaluation),
            scenario=STANDARD_CHALLENGE,
            account_size_usd=float(spec.prop_constraints.account_size_usd),
        )
        challenge_success_rate = float(challenge_summary.get("success_rate_empirical", 0.0))
        period_table = _period_stability_table(evaluation, account_size_usd=float(spec.prop_constraints.account_size_usd))
        oos_row = _scope_row(evaluation.summary_by_scope, "oos")
        oos_positive_month_ratio = _positive_month_ratio(
            evaluation.result.daily_results.loc[
                pd.to_datetime(evaluation.result.daily_results["session_date"]).dt.date.isin(set(evaluation.oos_sessions))
            ]
        )
        verdict = _validation_verdict(
            oos_row=oos_row,
            stress_oos_row=stress_oos,
            split_positive_rate=split_positive_rate,
            challenge_success_rate=challenge_success_rate,
            positive_month_ratio=oos_positive_month_ratio,
        )
        validation_score = _validation_score(
            oos_row=oos_row,
            stress_oos_row=stress_oos,
            split_positive_rate=split_positive_rate,
            challenge_success_rate=challenge_success_rate,
            positive_month_ratio=oos_positive_month_ratio,
        )

        evaluation.summary_by_scope.to_csv(variant_dir / "metrics_summary_by_scope.csv", index=False)
        evaluation.result.trades.to_csv(variant_dir / "trades.csv", index=False)
        evaluation.result.daily_results.to_csv(variant_dir / "daily_results.csv", index=False)
        stressed_summary_by_scope.to_csv(variant_dir / "stress_slippage_x2_metrics_by_scope.csv", index=False)
        split_table.to_csv(variant_dir / "split_stability.csv", index=False)
        period_table.to_csv(variant_dir / "period_stability.csv", index=False)
        pd.DataFrame([challenge_summary]).to_csv(variant_dir / "challenge_summary.csv", index=False)

        row = {
            "name": variant.name,
            "family": variant.family,
            "symbol": variant.symbol,
            "timeframe": variant.timeframe,
            "validation_verdict": verdict,
            "validation_score": validation_score,
            "oos_net_pnl": oos_row["net_pnl"],
            "oos_profit_factor": oos_row["profit_factor"],
            "oos_sharpe_ratio": oos_row["sharpe_ratio"],
            "oos_sortino_ratio": oos_row["sortino_ratio"],
            "oos_max_drawdown": oos_row["max_drawdown"],
            "oos_total_trades": oos_row["total_trades"],
            "oos_expectancy_per_trade": oos_row["expectancy_per_trade"],
            "oos_avg_holding_time_min": oos_row["avg_time_in_position_min"],
            "oos_time_in_market_pct": oos_row["avg_exposure_pct"],
            "oos_top_5_day_contribution_pct": oos_row["top_5_day_contribution_pct"],
            "oos_positive_month_ratio": oos_positive_month_ratio,
            "stress_x2_oos_net_pnl": stress_oos["net_pnl"],
            "stress_x2_oos_profit_factor": stress_oos["profit_factor"],
            "stress_x2_oos_sharpe_ratio": stress_oos["sharpe_ratio"],
            "split_positive_rate": split_positive_rate,
            "challenge_success_rate": challenge_success_rate,
            "challenge_bust_rate": challenge_summary.get("bust_rate_empirical"),
            "challenge_median_days_to_target": challenge_summary.get("median_days_to_target"),
        }
        rows.append(row)
        _json_dump(
            variant_dir / "validation_summary.json",
            {
                **row,
                "split_table_path": variant_dir / "split_stability.csv",
                "period_table_path": variant_dir / "period_stability.csv",
            },
        )

    validation_df = (
        pd.DataFrame(rows).sort_values(
            ["validation_score", "oos_profit_factor", "oos_sharpe_ratio"],
            ascending=[False, False, False],
        )
        if rows
        else pd.DataFrame()
    )
    validation_df.to_csv(validation_dir / "survivor_validation_summary.csv", index=False)
    _json_dump(
        validation_dir / "run_metadata.json",
        {
            "run_timestamp": datetime.now().isoformat(),
            "survivor_count": len(survivors_df),
            "validated_count": len(validation_df),
        },
    )
    report_lines = [
        "# Mean Reversion Validation",
        "",
        f"- Survivors validated: {len(validation_df)}",
        "",
        "```text",
        validation_df.to_string(index=False) if not validation_df.empty else "No validated survivor.",
        "```",
        "",
    ]
    (validation_dir / "validation_summary.md").write_text("\n".join(report_lines), encoding="utf-8")
    return validation_df, evaluation_map


def _evaluate_orb_reference(
    prepared: PreparedTimeframeData,
    spec: MeanReversionCampaignSpec,
) -> ORBEvaluation:
    execution_model, instrument = build_execution_model_for_profile(prepared.symbol, "repo_realistic")
    feat = compute_opening_range(
        prepared.feature_df.copy(),
        or_minutes=spec.orb_reference.or_minutes,
        opening_time=spec.orb_reference.opening_time,
    )
    strategy = ORBStrategy(
        or_minutes=spec.orb_reference.or_minutes,
        direction=spec.orb_reference.direction,
        one_trade_per_day=spec.orb_reference.one_trade_per_day,
        entry_buffer_ticks=spec.orb_reference.entry_buffer_ticks,
        stop_buffer_ticks=spec.orb_reference.stop_buffer_ticks,
        target_multiple=spec.orb_reference.target_multiple,
        opening_time=spec.orb_reference.opening_time,
        time_exit=spec.orb_reference.time_exit,
        tick_size=instrument.tick_size,
        atr_period=spec.orb_reference.atr_period,
        vwap_confirmation=spec.orb_reference.vwap_confirmation,
        vwap_column=spec.orb_reference.vwap_column,
    )
    signal_df = strategy.generate_signals(feat)
    trades = run_backtest(
        signal_df,
        execution_model=execution_model,
        tick_value_usd=instrument.tick_value_usd,
        point_value_usd=instrument.point_value_usd,
        time_exit=spec.orb_reference.time_exit,
        stop_buffer_ticks=spec.orb_reference.stop_buffer_ticks,
        target_multiple=spec.orb_reference.target_multiple,
        account_size_usd=None,
        risk_per_trade_pct=None,
        entry_on_next_open=True,
    )
    if "holding_minutes" not in trades.columns:
        if {"entry_time", "exit_time"}.issubset(trades.columns):
            entry_time = pd.to_datetime(trades["entry_time"], errors="coerce")
            exit_time = pd.to_datetime(trades["exit_time"], errors="coerce")
            holding_minutes = (exit_time - entry_time).dt.total_seconds().div(60.0)
            trades["holding_minutes"] = holding_minutes.fillna(0.0)
        else:
            trades["holding_minutes"] = 0.0
    all_sessions = sorted(pd.to_datetime(prepared.feature_df["session_date"]).dt.date.unique())
    is_sessions, oos_sessions = _split_sessions(all_sessions, spec.is_fraction)
    daily_results = _rebuild_daily_results_from_trades(
        trades=trades,
        all_sessions=all_sessions,
        initial_capital=float(spec.prop_constraints.account_size_usd),
    )
    summary_by_scope, _, _ = _build_scope_summary_table(
        trades=trades,
        daily_results=daily_results,
        bar_results=pd.DataFrame(),
        signal_df=signal_df,
        sessions_all=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        initial_capital=float(spec.prop_constraints.account_size_usd),
        constraints=spec.prop_constraints,
        instrument=instrument,
        execution_model=execution_model,
        rolling_window_days=20,
    )
    return ORBEvaluation(
        label=f"orb_{prepared.symbol}_{prepared.timeframe.label}",
        symbol=prepared.symbol,
        timeframe=prepared.timeframe.label,
        trades=trades,
        daily_results=daily_results,
        summary_by_scope=summary_by_scope,
        instrument=instrument,
        execution_model=execution_model,
        all_sessions=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
    )


def _component_daily_returns(
    name: str,
    daily_results: pd.DataFrame,
    sessions: list,
    capital: float,
) -> pd.Series:
    idx = pd.Index(pd.to_datetime(pd.Index(sessions)).date)
    if daily_results.empty:
        return pd.Series(0.0, index=idx, name=name, dtype=float)
    grouped = (
        daily_results.assign(session_date=pd.to_datetime(daily_results["session_date"]).dt.date)
        .groupby("session_date")["daily_pnl_usd"]
        .sum()
    )
    series = grouped.reindex(idx, fill_value=0.0).astype(float) / float(max(capital, 1.0))
    series.name = name
    return series


def _portfolio_metrics(daily_returns: pd.Series, capital: float) -> dict[str, Any]:
    if daily_returns.empty:
        return {"net_pnl": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0}
    pnl = daily_returns * float(capital)
    equity = float(capital) + pnl.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    std = float(daily_returns.std(ddof=0))
    downside = daily_returns[daily_returns < 0]
    downside_std = float(np.sqrt(np.mean(np.square(downside)))) if not downside.empty else 0.0
    sharpe = float((daily_returns.mean() / std) * math.sqrt(252.0)) if std > 0 else 0.0
    sortino = float((daily_returns.mean() / downside_std) * math.sqrt(252.0)) if downside_std > 0 else 0.0
    return {
        "net_pnl": float(pnl.sum()),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(drawdown.min()),
    }


def run_portfolio(
    spec: MeanReversionCampaignSpec,
    output_dir: Path,
    validation_df: pd.DataFrame,
    evaluation_map: dict[str, MeanReversionEvaluation],
    prepared_cache: dict[tuple[str, str], PreparedTimeframeData],
) -> dict[str, Path]:
    """Build the phase-3 portfolio with validated survivors and optional ORB references."""
    portfolio_dir = output_dir / "portfolio"
    portfolio_dir.mkdir(parents=True, exist_ok=True)

    if validation_df.empty or "validation_verdict" not in validation_df.columns:
        candidate_df = pd.DataFrame()
    else:
        candidate_df = validation_df.loc[
            validation_df["validation_verdict"].isin(["survivor", "portfolio_candidate"])
        ].copy()
        candidate_df = candidate_df.sort_values(["validation_score"], ascending=[False]).head(spec.max_portfolio_candidates)

    component_series: dict[str, pd.Series] = {}
    selection_rows: list[dict[str, Any]] = []
    capital = float(spec.prop_constraints.account_size_usd)

    for _, row in candidate_df.iterrows():
        evaluation = evaluation_map[str(row["name"])]
        series = _component_daily_returns(
            name=str(row["name"]),
            daily_results=evaluation.result.daily_results,
            sessions=evaluation.oos_sessions,
            capital=capital,
        )
        component_series[str(row["name"])] = series
        selection_rows.append(
            {
                "name": row["name"],
                "family": row["family"],
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "source": "mean_reversion",
                "validation_verdict": row["validation_verdict"],
                "validation_score": row["validation_score"],
            }
        )

    orb_rows: list[dict[str, Any]] = []
    if spec.include_orb_baseline:
        for key, prepared in prepared_cache.items():
            symbol, timeframe = key
            if timeframe != "5m":
                continue
            orb_eval = _evaluate_orb_reference(prepared, spec)
            oos_row = _scope_row(orb_eval.summary_by_scope, "oos")
            orb_rows.append(
                {
                    "name": orb_eval.label,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "oos_net_pnl": oos_row["net_pnl"],
                    "oos_profit_factor": oos_row["profit_factor"],
                    "oos_sharpe_ratio": oos_row["sharpe_ratio"],
                }
            )
            if _safe_float(oos_row.get("net_pnl")) > 0.0 and _safe_float(oos_row.get("profit_factor")) > 1.0:
                component_series[orb_eval.label] = _component_daily_returns(
                    name=orb_eval.label,
                    daily_results=orb_eval.daily_results,
                    sessions=orb_eval.oos_sessions,
                    capital=capital,
                )
                selection_rows.append(
                    {
                        "name": orb_eval.label,
                        "family": "orb_reference",
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "source": "orb_reference",
                        "validation_verdict": "reference",
                        "validation_score": _screening_score(oos_row, _positive_month_ratio(orb_eval.daily_results)),
                    }
                )

    selection_df = pd.DataFrame(selection_rows)
    selection_df.to_csv(portfolio_dir / "portfolio_selection.csv", index=False)
    pd.DataFrame(orb_rows).to_csv(portfolio_dir / "orb_reference_summary.csv", index=False)

    if not component_series:
        summary_csv = portfolio_dir / "portfolio_summary.csv"
        pd.DataFrame(
            [
                {
                    "portfolio": "none",
                    "components_retained": 0,
                    "diversification_benefit": 0.0,
                    "status": "no_eligible_component",
                }
            ]
        ).to_csv(summary_csv, index=False)
        summary_path = portfolio_dir / "portfolio_summary.md"
        summary_path.write_text("# Portfolio\n\nNo eligible component survived for portfolio construction.\n", encoding="utf-8")
        return {
            "portfolio_selection_csv": portfolio_dir / "portfolio_selection.csv",
            "orb_reference_summary_csv": portfolio_dir / "orb_reference_summary.csv",
            "portfolio_summary_csv": summary_csv,
            "portfolio_summary_md": summary_path,
        }

    return_matrix = pd.concat(component_series.values(), axis=1).fillna(0.0).sort_index()
    downside_matrix = return_matrix.clip(upper=0.0)
    corr = return_matrix.corr()
    downside_corr = downside_matrix.corr()

    equal_weight_returns = return_matrix.mean(axis=1)
    component_vol = return_matrix.std(ddof=0).replace(0.0, np.nan)
    inv_vol = (1.0 / component_vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol_weights = inv_vol / max(float(inv_vol.sum()), 1.0)
    vol_targeted_returns = return_matrix.mul(vol_weights, axis=1).sum(axis=1)

    equal_metrics = _portfolio_metrics(equal_weight_returns, capital=capital)
    vol_target_metrics = _portfolio_metrics(vol_targeted_returns, capital=capital)

    weighted_component_vol = float((component_vol.fillna(0.0) * vol_weights).sum())
    portfolio_vol = float(vol_targeted_returns.std(ddof=0))
    diversification_benefit = float(1.0 - (portfolio_vol / max(weighted_component_vol, 1e-9))) if weighted_component_vol > 0 else 0.0

    contribution = pd.DataFrame(
        {
            "name": return_matrix.columns,
            "equal_weight_contribution_pct": (
                (return_matrix.mean(axis=0) / max(equal_weight_returns.mean(), 1e-9))
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            ).values,
            "vol_target_weight": vol_weights.reindex(return_matrix.columns).fillna(0.0).values,
            "vol_target_contribution_pct": (
                (return_matrix.mul(vol_weights, axis=1).mean(axis=0) / max(vol_targeted_returns.mean(), 1e-9))
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            ).values,
        }
    )

    equal_curve = pd.DataFrame(
        {
            "session_date": pd.Index(equal_weight_returns.index).astype(str),
            "daily_return": equal_weight_returns.values,
            "daily_pnl_usd": (equal_weight_returns * capital).values,
        }
    )
    vol_curve = pd.DataFrame(
        {
            "session_date": pd.Index(vol_targeted_returns.index).astype(str),
            "daily_return": vol_targeted_returns.values,
            "daily_pnl_usd": (vol_targeted_returns * capital).values,
        }
    )

    corr.to_csv(portfolio_dir / "daily_return_correlation.csv")
    downside_corr.to_csv(portfolio_dir / "downside_correlation.csv")
    equal_curve.to_csv(portfolio_dir / "equal_weight_portfolio_daily.csv", index=False)
    vol_curve.to_csv(portfolio_dir / "vol_target_portfolio_daily.csv", index=False)
    contribution.to_csv(portfolio_dir / "portfolio_contribution.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {"portfolio": "equal_weight", **equal_metrics},
            {"portfolio": "vol_targeted", **vol_target_metrics},
        ]
    )
    summary_df["diversification_benefit"] = diversification_benefit
    summary_df.to_csv(portfolio_dir / "portfolio_summary.csv", index=False)

    summary_lines = [
        "# Mean Reversion Portfolio",
        "",
        f"- Components retained: {len(selection_df)}",
        f"- Diversification benefit (inverse-vol view): {diversification_benefit:.2%}",
        "",
        "## Selection",
        "",
        "```text",
        selection_df.to_string(index=False),
        "```",
        "",
        "## Portfolio Summary",
        "",
        "```text",
        summary_df.to_string(index=False),
        "```",
        "",
    ]
    summary_path = portfolio_dir / "portfolio_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return {
        "portfolio_selection_csv": portfolio_dir / "portfolio_selection.csv",
        "daily_return_corr_csv": portfolio_dir / "daily_return_correlation.csv",
        "downside_corr_csv": portfolio_dir / "downside_correlation.csv",
        "equal_weight_daily_csv": portfolio_dir / "equal_weight_portfolio_daily.csv",
        "vol_target_daily_csv": portfolio_dir / "vol_target_portfolio_daily.csv",
        "portfolio_contribution_csv": portfolio_dir / "portfolio_contribution.csv",
        "portfolio_summary_csv": portfolio_dir / "portfolio_summary.csv",
        "portfolio_summary_md": summary_path,
    }


def generate_notebook(notebook_path: Path, output_dir: Path) -> Path:
    """Create the final executable notebook for the full campaign."""
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_dir = output_dir.resolve()
    setup_code = """from pathlib import Path
import json
import sys
import pandas as pd
from IPython.display import Markdown, display

root = Path.cwd().resolve()
while root != root.parent and not (root / "pyproject.toml").exists():
    root = root.parent
if not (root / "pyproject.toml").exists():
    raise RuntimeError("Could not locate repository root.")
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 160)
"""
    config_code = f"OUTPUT_DIR = Path(r\"{str(resolved_output_dir)}\")\nprint(OUTPUT_DIR)\n"
    screening_code = """summary_md = (OUTPUT_DIR / "screening" / "screening_summary.md").read_text(encoding="utf-8")
display(Markdown(summary_md))
display(pd.read_csv(OUTPUT_DIR / "screening" / "screening_survivors.csv"))
"""
    validation_code = """path = OUTPUT_DIR / "validation" / "survivor_validation_summary.csv"
if path.exists():
    display(pd.read_csv(path))
"""
    portfolio_code = """path = OUTPUT_DIR / "portfolio" / "portfolio_summary.csv"
if path.exists():
    display(pd.read_csv(path))
for optional in [
    OUTPUT_DIR / "portfolio" / "portfolio_selection.csv",
    OUTPUT_DIR / "portfolio" / "daily_return_correlation.csv",
    OUTPUT_DIR / "portfolio" / "downside_correlation.csv",
]:
    if optional.exists():
        display(Markdown(f"### {optional.name}"))
        display(pd.read_csv(optional))
"""
    notebook = {
        "cells": [
            _notebook_cell("markdown", "# Intraday Mean Reversion Campaign"),
            _notebook_cell("code", setup_code),
            _notebook_cell("code", config_code),
            _notebook_cell("markdown", "## 1) Screening"),
            _notebook_cell("code", screening_code),
            _notebook_cell("markdown", "## 2) Validation"),
            _notebook_cell("code", validation_code),
            _notebook_cell("markdown", "## 3) Portfolio"),
            _notebook_cell("code", portfolio_code),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return notebook_path


def run_mean_reversion_campaign(
    spec: MeanReversionCampaignSpec,
    output_dir: Path,
    phase: str = "full",
    notebook_path: Path | None = None,
) -> dict[str, Path]:
    """Run the requested campaign phase, automatically resolving dependencies."""
    if phase not in CAMPAIGN_PHASES:
        raise ValueError(f"Unsupported phase '{phase}'.")
    output_dir.mkdir(parents=True, exist_ok=True)

    _, survivors_df, prepared_cache = run_screening(spec=spec, output_dir=output_dir)
    artifacts: dict[str, Path] = {
        "screening_results_csv": output_dir / "screening" / "screening_results.csv",
        "screening_survivors_csv": output_dir / "screening" / "screening_survivors.csv",
        "screening_summary_md": output_dir / "screening" / "screening_summary.md",
    }
    if phase == "screening":
        return artifacts

    validation_df, evaluation_map = run_validation(
        spec=spec,
        output_dir=output_dir,
        survivors_df=survivors_df,
        prepared_cache=prepared_cache,
    )
    artifacts.update(
        {
            "validation_summary_csv": output_dir / "validation" / "survivor_validation_summary.csv",
            "validation_summary_md": output_dir / "validation" / "validation_summary.md",
        }
    )
    if phase == "validation":
        return artifacts

    portfolio_artifacts = run_portfolio(
        spec=spec,
        output_dir=output_dir,
        validation_df=validation_df,
        evaluation_map=evaluation_map,
        prepared_cache=prepared_cache,
    )
    artifacts.update(portfolio_artifacts)
    if phase == "portfolio":
        return artifacts

    generated_notebook = generate_notebook(
        notebook_path=notebook_path or (NOTEBOOKS_DIR / "mean_reversion_intraday_campaign.ipynb"),
        output_dir=output_dir,
    )
    artifacts["notebook"] = generated_notebook
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the intraday mean reversion research campaign.")
    parser.add_argument("--phase", type=str, default="full", choices=CAMPAIGN_PHASES)
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory.")
    parser.add_argument("--notebook-path", type=Path, default=None, help="Optional notebook path.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    spec = build_default_mean_reversion_campaign_spec()
    output_dir = args.output_dir or (spec.output_root.parent / f"{spec.output_root.name}_{timestamp}")
    artifacts = run_mean_reversion_campaign(
        spec=spec,
        output_dir=Path(output_dir),
        phase=str(args.phase),
        notebook_path=args.notebook_path,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
