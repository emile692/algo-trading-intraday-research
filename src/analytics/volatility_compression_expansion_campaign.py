"""Leak-free validation campaign for the VCEB intraday strategy."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.metrics import compute_metrics
from src.analytics.vwap_validation import (
    StressScenario,
    _apply_cost_stress_overlay,
    _build_scope_summary_table,
    _split_sessions,
)
from src.config.paths import EXPORTS_DIR, ensure_directories
from src.config.vwap_campaign import PropFirmConstraintConfig, build_default_prop_constraints, resolve_default_vwap_dataset
from src.data.cleaning import clean_ohlcv
from src.data.loader import load_ohlcv_file
from src.data.resampling import build_resampled_output_path, resample_ohlcv
from src.engine.execution_model import ExecutionModel
from src.engine.volatility_compression_expansion_backtester import (
    VCEBBacktestResult,
    run_volatility_compression_expansion_backtest,
)
from src.engine.vwap_backtester import InstrumentDetails, build_execution_model_for_profile
from src.strategy.volatility_compression_expansion import (
    DEFAULT_BAR_MINUTES,
    DEFAULT_BARS_PER_SESSION,
    DEFAULT_ENTRY_END,
    DEFAULT_ENTRY_START,
    DEFAULT_SESSION_END,
    DEFAULT_SESSION_START,
    VCEBVariantConfig,
    build_vceb_signal_frame,
    prepare_vceb_feature_frame,
)


DEFAULT_SYMBOLS = ("MNQ", "MES", "M2K", "MGC")
DEFAULT_STRESS_SCENARIOS = (
    StressScenario(name="nominal", notes="Base realistic-cost run."),
    StressScenario(
        name="combined_x2_plus25",
        slippage_multiplier=2.0,
        commission_multiplier=1.25,
        notes="Primary harder stress: slippage x2 plus commission +25%.",
    ),
    StressScenario(
        name="entry_penalty_1tick",
        entry_penalty_ticks=1.0,
        notes="Uniform extra one-tick entry penalty on every trade.",
    ),
)
OOS_YEARLY_COLUMNS = [
    "symbol",
    "variant_name",
    "box_lookback",
    "compression_threshold",
    "target_r",
    "year",
    "oos_sessions",
    "n_trades",
    "net_pnl",
    "profit_factor",
    "sharpe_ratio",
    "win_rate",
    "average_trade",
    "max_drawdown",
]
STRESS_COLUMNS = [
    "symbol",
    "variant_name",
    "box_lookback",
    "compression_threshold",
    "target_r",
    "scenario",
    "notes",
    "oos_total_trades",
    "oos_net_pnl",
    "oos_profit_factor",
    "oos_sharpe_ratio",
    "oos_max_drawdown",
    "oos_expectancy",
]
SURVIVOR_COLUMNS = [
    "variant_name",
    "box_lookback",
    "compression_threshold",
    "target_r",
    "screening_score",
    "asset_count",
    "oos_positive_assets",
    "oos_total_trades",
    "oos_total_net_pnl",
    "oos_median_profit_factor",
    "oos_median_sharpe",
    "oos_median_top_5_day_contribution_pct",
    "stress_positive_rows",
    "stress_total_rows",
    "worst_stress_oos_net_pnl",
    "worst_stress_oos_profit_factor",
    "median_mfe_r",
    "median_mae_r",
    "pct_trades_reaching_1r_mfe",
]


@dataclass(frozen=True)
class PreparedInstrumentData:
    """Prepared 5-minute VCEB input for one symbol."""

    symbol: str
    source_dataset_path: Path
    resampled_dataset_path: Path
    feature_df: pd.DataFrame


@dataclass
class VCEBEvaluation:
    """In-memory evaluation bundle for one instrument / variant pair."""

    symbol: str
    variant: VCEBVariantConfig
    signal_df: pd.DataFrame
    result: VCEBBacktestResult
    instrument: InstrumentDetails
    execution_model: ExecutionModel
    all_sessions: list
    is_sessions: list
    oos_sessions: list
    summary_by_scope: pd.DataFrame


@dataclass(frozen=True)
class VCEBCampaignSpec:
    """Top-level VCEB campaign settings."""

    output_root: Path
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    dataset_paths: dict[str, Path] | None = None
    is_fraction: float = 0.70
    session_start: str = DEFAULT_SESSION_START
    session_end: str = DEFAULT_SESSION_END
    entry_start: str = DEFAULT_ENTRY_START
    entry_end: str = DEFAULT_ENTRY_END
    resample_rule: str = "5min"
    atr_window: int = 48
    box_lengths: tuple[int, ...] = (8, 12, 16)
    compression_thresholds: tuple[float, ...] = (15.0, 20.0)
    target_rs: tuple[float, ...] = (1.8, 2.2)
    expansion_threshold: float = 1.20
    breakout_buffer_atr: float = 0.10
    time_stop_bars: int = 12
    fixed_quantity: int = 1
    percentile_history_sessions: int = 20
    min_percentile_history_sessions: int = 5
    rolling_window_days: int = 20
    prop_constraints: PropFirmConstraintConfig = build_default_prop_constraints()
    max_validation_survivors: int = 3
    start_date: str | None = None
    end_date: str | None = None
    stress_scenarios: tuple[StressScenario, ...] = DEFAULT_STRESS_SCENARIOS


def build_default_campaign_spec(output_root: Path | None = None) -> VCEBCampaignSpec:
    root = output_root or (EXPORTS_DIR / f"volatility_compression_expansion_{datetime.now():%Y%m%d}_run")
    return VCEBCampaignSpec(output_root=root)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    def _serialize(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): _serialize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_serialize(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            numeric = float(value)
            if math.isnan(numeric) or math.isinf(numeric):
                return None
            return numeric
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if pd.isna(value):
            return None
        return value

    clean = {str(key): _serialize(value) for key, value in payload.items()}
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float(default)
    return float(numeric)


def _scope_row(summary_by_scope: pd.DataFrame, scope: str) -> pd.Series:
    return summary_by_scope.loc[summary_by_scope["scope"] == scope].iloc[0]


def _subset_frame_by_sessions(frame: pd.DataFrame, sessions: list) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    session_set = set(pd.to_datetime(pd.Index(sessions)).date)
    out = frame.copy()
    out_dates = pd.to_datetime(out["session_date"]).dt.date
    return out.loc[out_dates.isin(session_set)].copy().reset_index(drop=True)


def _subset_frame_by_dates(
    frame: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    out = frame.copy()
    values = pd.to_datetime(out[timestamp_col], errors="coerce")
    dates = values.dt.date
    mask = pd.Series(True, index=out.index, dtype=bool)

    if start_date:
        start = pd.Timestamp(start_date).date()
        mask &= dates >= start
    if end_date:
        end = pd.Timestamp(end_date).date()
        mask &= dates <= end
    return out.loc[mask].copy().reset_index(drop=True)


def _warmup_filtered_raw(raw: pd.DataFrame, spec: VCEBCampaignSpec) -> pd.DataFrame:
    if raw.empty or (spec.start_date is None and spec.end_date is None):
        return raw.copy()

    out = raw.copy()
    dates = pd.to_datetime(out["timestamp"], errors="coerce").dt.date
    mask = pd.Series(True, index=out.index, dtype=bool)
    if spec.start_date is not None:
        warmup_start = (pd.Timestamp(spec.start_date) - pd.Timedelta(days=60)).date()
        mask &= dates >= warmup_start
    if spec.end_date is not None:
        end_date = pd.Timestamp(spec.end_date).date()
        mask &= dates <= end_date
    return out.loc[mask].copy().reset_index(drop=True)


def _median_holding_minutes(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return _safe_float(pd.to_numeric(trades.get("holding_minutes"), errors="coerce").median())


def _oos_trade_summary(trades: pd.DataFrame) -> tuple[float, float, float]:
    if trades.empty:
        return 0.0, 0.0, 0.0
    mfe_r = pd.to_numeric(trades.get("mfe_r"), errors="coerce")
    mae_r = pd.to_numeric(trades.get("mae_r"), errors="coerce")
    reach_1r = (mfe_r >= 1.0).mean() if mfe_r.notna().any() else 0.0
    return _safe_float(mfe_r.median(), default=np.nan), _safe_float(mae_r.median(), default=np.nan), float(reach_1r)


def _build_variants(spec: VCEBCampaignSpec) -> list[VCEBVariantConfig]:
    configured: list[VCEBVariantConfig] = []
    for box_lookback in spec.box_lengths:
        for compression_threshold in spec.compression_thresholds:
            for target_r in spec.target_rs:
                configured.append(
                    VCEBVariantConfig(
                        name=(
                            f"vceb_n{int(box_lookback)}"
                            f"_ct{int(compression_threshold)}"
                            f"_tr{str(float(target_r)).replace('.', 'p')}"
                        ),
                        box_lookback=int(box_lookback),
                        compression_threshold=float(compression_threshold),
                        target_r=float(target_r),
                        atr_window=int(spec.atr_window),
                        expansion_threshold=float(spec.expansion_threshold),
                        breakout_buffer_atr=float(spec.breakout_buffer_atr),
                        time_stop_bars=int(spec.time_stop_bars),
                        fixed_quantity=int(spec.fixed_quantity),
                        initial_capital_usd=float(spec.prop_constraints.account_size_usd),
                        session_start=spec.session_start,
                        session_end=spec.session_end,
                        entry_start=spec.entry_start,
                        entry_end=spec.entry_end,
                    )
                )
    return configured


def _resolve_dataset_path(symbol: str, spec: VCEBCampaignSpec) -> Path:
    if spec.dataset_paths and symbol in spec.dataset_paths:
        return Path(spec.dataset_paths[symbol])
    return resolve_default_vwap_dataset(symbol)


def _prepare_instrument_data(symbol: str, spec: VCEBCampaignSpec, output_root: Path) -> PreparedInstrumentData:
    dataset_path = _resolve_dataset_path(symbol, spec)
    raw = load_ohlcv_file(dataset_path)
    clean = clean_ohlcv(raw)
    trimmed = _warmup_filtered_raw(clean, spec)
    resampled = resample_ohlcv(trimmed, rule=spec.resample_rule)

    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    resampled_path = build_resampled_output_path(dataset_path, rule=spec.resample_rule, output_dir=data_dir)
    parquet_df = resampled.set_index("timestamp")
    parquet_df.index.name = "timestamp"
    parquet_df.to_parquet(resampled_path)

    feature_df = prepare_vceb_feature_frame(
        resampled,
        session_start=spec.session_start,
        session_end=spec.session_end,
        box_lengths=spec.box_lengths,
        atr_window=spec.atr_window,
        compression_percentile_lookback_bars=int(spec.percentile_history_sessions) * DEFAULT_BARS_PER_SESSION,
        compression_percentile_min_history_bars=int(spec.min_percentile_history_sessions) * DEFAULT_BARS_PER_SESSION,
    )
    feature_df = _subset_frame_by_dates(
        feature_df,
        start_date=spec.start_date,
        end_date=spec.end_date,
        timestamp_col="timestamp",
    )
    return PreparedInstrumentData(
        symbol=symbol,
        source_dataset_path=dataset_path,
        resampled_dataset_path=resampled_path,
        feature_df=feature_df,
    )


def _evaluate_variant(
    prepared: PreparedInstrumentData,
    variant: VCEBVariantConfig,
    spec: VCEBCampaignSpec,
) -> VCEBEvaluation:
    signal_df = build_vceb_signal_frame(prepared.feature_df, variant)
    execution_model, instrument = build_execution_model_for_profile(prepared.symbol, "repo_realistic")
    result = run_volatility_compression_expansion_backtest(
        signal_df=signal_df,
        variant=variant,
        execution_model=execution_model,
        instrument=instrument,
        bar_minutes=DEFAULT_BAR_MINUTES,
    )
    all_sessions = sorted(pd.to_datetime(signal_df["session_date"]).dt.date.unique())
    is_sessions, oos_sessions = _split_sessions(all_sessions, spec.is_fraction)
    summary_by_scope, _, _ = _build_scope_summary_table(
        trades=result.trades,
        daily_results=result.daily_results,
        bar_results=result.bar_results,
        signal_df=signal_df,
        sessions_all=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        initial_capital=float(variant.initial_capital_usd),
        constraints=spec.prop_constraints,
        instrument=instrument,
        execution_model=execution_model,
        rolling_window_days=int(spec.rolling_window_days),
    )
    return VCEBEvaluation(
        symbol=prepared.symbol,
        variant=variant,
        signal_df=signal_df,
        result=result,
        instrument=instrument,
        execution_model=execution_model,
        all_sessions=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        summary_by_scope=summary_by_scope,
    )


def _instrument_variant_row(evaluation: VCEBEvaluation, prepared: PreparedInstrumentData) -> dict[str, Any]:
    overall = _scope_row(evaluation.summary_by_scope, "overall")
    is_row = _scope_row(evaluation.summary_by_scope, "is")
    oos = _scope_row(evaluation.summary_by_scope, "oos")
    oos_trades = _subset_frame_by_sessions(evaluation.result.trades, evaluation.oos_sessions)
    overall_trades = evaluation.result.trades.copy()
    return {
        "symbol": evaluation.symbol,
        "dataset_path": str(prepared.source_dataset_path),
        "resampled_dataset_path": str(prepared.resampled_dataset_path),
        "variant_name": evaluation.variant.name,
        "box_lookback": int(evaluation.variant.box_lookback),
        "compression_threshold": float(evaluation.variant.compression_threshold),
        "target_r": float(evaluation.variant.target_r),
        "is_sessions": int(len(evaluation.is_sessions)),
        "oos_sessions": int(len(evaluation.oos_sessions)),
        "overall_total_trades": int(overall["total_trades"]),
        "overall_net_pnl": float(overall["net_pnl"]),
        "overall_profit_factor": float(overall["profit_factor"]),
        "overall_sharpe_ratio": float(overall["sharpe_ratio"]),
        "overall_max_drawdown": float(overall["max_drawdown"]),
        "overall_win_rate": float(overall["hit_rate"]),
        "overall_average_trade": float(overall["expectancy_per_trade"]),
        "overall_expectancy": float(overall["expectancy_per_trade"]),
        "overall_median_holding_minutes": _median_holding_minutes(overall_trades),
        "overall_time_in_market_pct": float(overall["avg_exposure_pct"]),
        "is_total_trades": int(is_row["total_trades"]),
        "is_net_pnl": float(is_row["net_pnl"]),
        "is_profit_factor": float(is_row["profit_factor"]),
        "is_sharpe_ratio": float(is_row["sharpe_ratio"]),
        "oos_total_trades": int(oos["total_trades"]),
        "oos_net_pnl": float(oos["net_pnl"]),
        "oos_profit_factor": float(oos["profit_factor"]),
        "oos_sharpe_ratio": float(oos["sharpe_ratio"]),
        "oos_max_drawdown": float(oos["max_drawdown"]),
        "oos_win_rate": float(oos["hit_rate"]),
        "oos_average_trade": float(oos["expectancy_per_trade"]),
        "oos_expectancy": float(oos["expectancy_per_trade"]),
        "oos_median_holding_minutes": _median_holding_minutes(oos_trades),
        "oos_time_in_market_pct": float(oos["avg_exposure_pct"]),
        "oos_top_5_day_contribution_pct": float(oos["top_5_day_contribution_pct"]),
    }


def _screening_score(row: pd.Series) -> float:
    return float(
        1.8 * np.tanh(_safe_float(row.get("oos_median_profit_factor"), default=1.0) - 1.0)
        + 1.4 * np.tanh(_safe_float(row.get("oos_median_sharpe"), default=0.0))
        + 1.0 * np.tanh(_safe_float(row.get("oos_total_net_pnl")) / 4_000.0)
        + 0.8 * np.tanh(_safe_float(row.get("oos_total_trades")) / 120.0)
        + 0.8 * (_safe_float(row.get("oos_positive_assets")) / max(_safe_float(row.get("asset_count"), default=1.0), 1.0))
        - 1.1 * _safe_float(row.get("oos_median_top_5_day_contribution_pct"))
        - 0.8 * np.tanh(abs(_safe_float(row.get("oos_worst_max_drawdown"))) / 4_000.0)
    )


def _pass_screening(row: pd.Series) -> bool:
    return bool(
        (_safe_float(row.get("oos_total_net_pnl")) > 0.0)
        and (_safe_float(row.get("oos_median_profit_factor"), default=1.0) > 1.0)
        and (_safe_float(row.get("oos_median_sharpe")) > 0.0)
        and (_safe_float(row.get("oos_total_trades")) >= 60.0)
        and (_safe_float(row.get("oos_positive_assets")) >= 2.0)
        and (_safe_float(row.get("oos_median_top_5_day_contribution_pct"), default=1.0) < 0.75)
    )


def _aggregate_screening(instrument_variant_df: pd.DataFrame) -> pd.DataFrame:
    if instrument_variant_df.empty:
        return pd.DataFrame()

    working = instrument_variant_df.copy()
    working["oos_top_5_day_contribution_abs"] = pd.to_numeric(
        working["oos_top_5_day_contribution_pct"],
        errors="coerce",
    ).abs()
    summary = (
        working.groupby(
            ["variant_name", "box_lookback", "compression_threshold", "target_r"],
            as_index=False,
            sort=False,
        )
        .agg(
            asset_count=("symbol", "nunique"),
            oos_positive_assets=("oos_net_pnl", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            oos_profit_factor_gt_1_assets=("oos_profit_factor", lambda s: int((pd.to_numeric(s, errors="coerce") > 1.0).sum())),
            oos_total_trades=("oos_total_trades", "sum"),
            oos_total_net_pnl=("oos_net_pnl", "sum"),
            oos_median_profit_factor=("oos_profit_factor", "median"),
            oos_median_sharpe=("oos_sharpe_ratio", "median"),
            oos_mean_expectancy=("oos_expectancy", "mean"),
            oos_median_win_rate=("oos_win_rate", "median"),
            oos_worst_max_drawdown=("oos_max_drawdown", "min"),
            oos_median_holding_minutes=("oos_median_holding_minutes", "median"),
            oos_median_time_in_market_pct=("oos_time_in_market_pct", "median"),
            oos_median_top_5_day_contribution_pct=("oos_top_5_day_contribution_abs", "median"),
        )
    )
    summary["screening_score"] = summary.apply(_screening_score, axis=1)
    summary["pass_screening"] = summary.apply(_pass_screening, axis=1)
    return summary.sort_values(
        ["pass_screening", "screening_score", "oos_median_profit_factor", "oos_median_sharpe"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _select_survivors(screening_summary: pd.DataFrame, spec: VCEBCampaignSpec) -> pd.DataFrame:
    if screening_summary.empty:
        return screening_summary.copy()
    passing = screening_summary.loc[screening_summary["pass_screening"]].copy()
    if not passing.empty:
        return passing.head(int(spec.max_validation_survivors)).reset_index(drop=True)
    return screening_summary.head(min(int(spec.max_validation_survivors), 2)).reset_index(drop=True)


def _variant_params_row(variant: VCEBVariantConfig) -> dict[str, Any]:
    return {
        "variant_name": variant.name,
        "box_lookback": int(variant.box_lookback),
        "compression_threshold": float(variant.compression_threshold),
        "target_r": float(variant.target_r),
    }


def _oos_yearly_rows(evaluation: VCEBEvaluation) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not evaluation.oos_sessions:
        return rows

    oos_years = sorted(pd.Index(pd.to_datetime(pd.Index(evaluation.oos_sessions)).year).unique())
    for year in oos_years:
        year_sessions = [session for session in evaluation.oos_sessions if pd.Timestamp(session).year == int(year)]
        if not year_sessions:
            continue
        year_trades = _subset_frame_by_sessions(evaluation.result.trades, year_sessions)
        metrics = compute_metrics(
            year_trades,
            session_dates=year_sessions,
            initial_capital=float(evaluation.variant.initial_capital_usd),
        )
        rows.append(
            {
                "symbol": evaluation.symbol,
                **_variant_params_row(evaluation.variant),
                "year": int(year),
                "oos_sessions": int(len(year_sessions)),
                "n_trades": int(metrics.get("n_trades", 0)),
                "net_pnl": float(metrics.get("cumulative_pnl", 0.0)),
                "profit_factor": float(metrics.get("profit_factor", 0.0)),
                "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
                "win_rate": float(metrics.get("win_rate", 0.0)),
                "average_trade": float(metrics.get("expectancy", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
            }
        )
    return rows


def _stress_rows(evaluation: VCEBEvaluation, spec: VCEBCampaignSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in spec.stress_scenarios:
        stressed_trades = _apply_cost_stress_overlay(
            evaluation.result.trades,
            scenario=scenario,
            instrument=evaluation.instrument,
            execution_model=evaluation.execution_model,
            session_start=spec.session_start,
        )
        daily_results = evaluation.result.daily_results.copy()
        if scenario.name != "nominal":
            session_index = pd.Index(pd.to_datetime(pd.Index(evaluation.all_sessions)).date)
            daily_results = pd.DataFrame({"session_date": session_index})
            if stressed_trades.empty:
                daily_results["daily_pnl_usd"] = 0.0
                daily_results["daily_gross_pnl_usd"] = 0.0
                daily_results["daily_fees_usd"] = 0.0
                daily_results["daily_trade_count"] = 0
            else:
                grouped = (
                    stressed_trades.assign(session_date=pd.to_datetime(stressed_trades["session_date"]).dt.date)
                    .groupby("session_date", as_index=False)
                    .agg(
                        daily_pnl_usd=("net_pnl_usd", "sum"),
                        daily_gross_pnl_usd=("pnl_usd", "sum"),
                        daily_fees_usd=("fees", "sum"),
                        daily_trade_count=("trade_id", "count"),
                    )
                )
                daily_results = daily_results.merge(grouped, on="session_date", how="left").fillna(
                    {
                        "daily_pnl_usd": 0.0,
                        "daily_gross_pnl_usd": 0.0,
                        "daily_fees_usd": 0.0,
                        "daily_trade_count": 0,
                    }
                )
            daily_results["daily_loss_count"] = 0
            daily_results["daily_stop_breached"] = False
            daily_results["trading_halted"] = False
            daily_results = daily_results.sort_values("session_date").reset_index(drop=True)
            daily_results["equity"] = float(evaluation.variant.initial_capital_usd) + pd.to_numeric(
                daily_results["daily_pnl_usd"], errors="coerce"
            ).cumsum()
            daily_results["peak_equity"] = daily_results["equity"].cummax()
            daily_results["drawdown_usd"] = daily_results["equity"] - daily_results["peak_equity"]
            daily_results["green_day"] = daily_results["daily_pnl_usd"] > 0
            daily_results["weekday"] = pd.to_datetime(daily_results["session_date"]).dt.day_name()

        summary_by_scope, _, _ = _build_scope_summary_table(
            trades=stressed_trades,
            daily_results=daily_results,
            bar_results=pd.DataFrame(),
            signal_df=evaluation.signal_df,
            sessions_all=evaluation.all_sessions,
            is_sessions=evaluation.is_sessions,
            oos_sessions=evaluation.oos_sessions,
            initial_capital=float(evaluation.variant.initial_capital_usd),
            constraints=spec.prop_constraints,
            instrument=evaluation.instrument,
            execution_model=evaluation.execution_model,
            rolling_window_days=int(spec.rolling_window_days),
        )
        oos = _scope_row(summary_by_scope, "oos")
        rows.append(
            {
                "symbol": evaluation.symbol,
                **_variant_params_row(evaluation.variant),
                "scenario": scenario.name,
                "notes": scenario.notes,
                "oos_total_trades": int(oos["total_trades"]),
                "oos_net_pnl": float(oos["net_pnl"]),
                "oos_profit_factor": float(oos["profit_factor"]),
                "oos_sharpe_ratio": float(oos["sharpe_ratio"]),
                "oos_max_drawdown": float(oos["max_drawdown"]),
                "oos_expectancy": float(oos["expectancy_per_trade"]),
            }
        )
    return rows


def _entry_hour_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["entry_hour", "trade_count", "net_pnl_usd", "expectancy_per_trade", "win_rate"])
    out = trades.copy()
    entry_hour = (
        pd.to_numeric(out["entry_hour"], errors="coerce")
        if "entry_hour" in out.columns
        else pd.Series(np.nan, index=out.index, dtype=float)
    )
    out["entry_hour"] = entry_hour.fillna(pd.to_datetime(out["entry_time"], errors="coerce").dt.hour)
    out["is_win"] = pd.to_numeric(out["net_pnl_usd"], errors="coerce") > 0
    return (
        out.groupby("entry_hour", as_index=False)
        .agg(
            trade_count=("trade_id", "count"),
            net_pnl_usd=("net_pnl_usd", "sum"),
            expectancy_per_trade=("net_pnl_usd", "mean"),
            win_rate=("is_win", "mean"),
        )
        .sort_values("entry_hour")
        .reset_index(drop=True)
    )


def _survivor_summary_rows(
    survivors: pd.DataFrame,
    evaluation_map: dict[tuple[str, str], VCEBEvaluation],
    stress_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for survivor in survivors.to_dict(orient="records"):
        variant_name = str(survivor["variant_name"])
        asset_evaluations = [evaluation for (symbol, name), evaluation in evaluation_map.items() if name == variant_name]
        oos_frames = [_subset_frame_by_sessions(evaluation.result.trades, evaluation.oos_sessions) for evaluation in asset_evaluations]
        oos_trades = pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
        stress_subset = stress_df.loc[(stress_df["variant_name"] == variant_name) & (stress_df["scenario"] != "nominal")].copy()
        median_mfe_r, median_mae_r, reach_1r = _oos_trade_summary(oos_trades)
        rows.append(
            {
                "variant_name": variant_name,
                "box_lookback": int(survivor["box_lookback"]),
                "compression_threshold": float(survivor["compression_threshold"]),
                "target_r": float(survivor["target_r"]),
                "screening_score": float(survivor["screening_score"]),
                "asset_count": int(survivor["asset_count"]),
                "oos_positive_assets": int(survivor["oos_positive_assets"]),
                "oos_total_trades": int(survivor["oos_total_trades"]),
                "oos_total_net_pnl": float(survivor["oos_total_net_pnl"]),
                "oos_median_profit_factor": float(survivor["oos_median_profit_factor"]),
                "oos_median_sharpe": float(survivor["oos_median_sharpe"]),
                "oos_median_top_5_day_contribution_pct": float(survivor["oos_median_top_5_day_contribution_pct"]),
                "stress_positive_rows": int((pd.to_numeric(stress_subset["oos_net_pnl"], errors="coerce") > 0).sum()) if not stress_subset.empty else 0,
                "stress_total_rows": int(len(stress_subset)),
                "worst_stress_oos_net_pnl": float(pd.to_numeric(stress_subset["oos_net_pnl"], errors="coerce").min()) if not stress_subset.empty else 0.0,
                "worst_stress_oos_profit_factor": float(pd.to_numeric(stress_subset["oos_profit_factor"], errors="coerce").min()) if not stress_subset.empty else 0.0,
                "median_mfe_r": median_mfe_r,
                "median_mae_r": median_mae_r,
                "pct_trades_reaching_1r_mfe": reach_1r,
            }
        )
    if not rows:
        return pd.DataFrame(columns=SURVIVOR_COLUMNS)
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["oos_positive_assets", "oos_total_net_pnl", "oos_median_profit_factor", "oos_median_sharpe"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )


def _cross_asset_character(positive_assets: int) -> str:
    if positive_assets >= 3:
        return "cross_asset"
    if positive_assets == 2:
        return "mixed"
    if positive_assets == 1:
        return "mono_asset"
    return "dead"


def _build_final_verdict(
    screening_summary: pd.DataFrame,
    survivor_validation_summary: pd.DataFrame,
) -> dict[str, Any]:
    if not survivor_validation_summary.empty:
        best = survivor_validation_summary.iloc[0]
        stress_positive_ratio = _safe_float(best.get("stress_positive_rows")) / max(
            _safe_float(best.get("stress_total_rows"), default=1.0),
            1.0,
        )
        candidate_v2 = bool(
            (_safe_float(best.get("oos_total_net_pnl")) > 0.0)
            and (_safe_float(best.get("oos_median_profit_factor"), default=1.0) > 1.02)
            and (_safe_float(best.get("oos_median_sharpe")) > 0.10)
            and (_safe_float(best.get("oos_total_trades")) >= 80.0)
            and (_safe_float(best.get("oos_positive_assets")) >= 2.0)
            and (stress_positive_ratio >= 0.50)
            and (_safe_float(best.get("oos_median_top_5_day_contribution_pct"), default=1.0) < 0.75)
        )
        verdict = "candidate_a_v2" if candidate_v2 else "non_defendable"
        reason = (
            "Cross-asset OOS evidence remains positive after costs with enough trades to justify a V2."
            if candidate_v2
            else "The best observed configuration stays too fragile cross-asset, too sparse, or too cost-sensitive for a defendable V2."
        )
        return {
            "research_verdict": verdict,
            "best_variant_name": str(best["variant_name"]),
            "best_variant_parameters": {
                "box_lookback": int(best["box_lookback"]),
                "compression_threshold": float(best["compression_threshold"]),
                "target_r": float(best["target_r"]),
            },
            "cross_asset_character": _cross_asset_character(int(best["oos_positive_assets"])),
            "oos_positive_assets": int(best["oos_positive_assets"]),
            "oos_total_trades": int(best["oos_total_trades"]),
            "oos_total_net_pnl": float(best["oos_total_net_pnl"]),
            "worst_stress_oos_net_pnl": float(best["worst_stress_oos_net_pnl"]),
            "median_mfe_r": _safe_float(best.get("median_mfe_r"), default=np.nan),
            "median_mae_r": _safe_float(best.get("median_mae_r"), default=np.nan),
            "reason": reason,
        }

    if not screening_summary.empty:
        best = screening_summary.iloc[0]
        return {
            "research_verdict": "non_defendable",
            "best_variant_name": str(best["variant_name"]),
            "best_variant_parameters": {
                "box_lookback": int(best["box_lookback"]),
                "compression_threshold": float(best["compression_threshold"]),
                "target_r": float(best["target_r"]),
            },
            "cross_asset_character": _cross_asset_character(int(best["oos_positive_assets"])),
            "oos_positive_assets": int(best["oos_positive_assets"]),
            "oos_total_trades": int(best["oos_total_trades"]),
            "oos_total_net_pnl": float(best["oos_total_net_pnl"]),
            "reason": "No screened configuration cleared the minimum cross-asset OOS and robustness hurdles.",
        }

    return {
        "research_verdict": "non_defendable",
        "best_variant_name": None,
        "best_variant_parameters": {},
        "cross_asset_character": "dead",
        "oos_positive_assets": 0,
        "oos_total_trades": 0,
        "oos_total_net_pnl": 0.0,
        "reason": "No evaluation rows were produced.",
    }


def _write_final_report(
    output_path: Path,
    *,
    spec: VCEBCampaignSpec,
    prepared_data: dict[str, PreparedInstrumentData],
    screening_summary: pd.DataFrame,
    survivor_validation_summary: pd.DataFrame,
    verdict: dict[str, Any],
) -> None:
    lines = [
        "# Volatility Compression -> Expansion Breakout Validation",
        "",
        "## Methodology",
        "",
        "- Independent VCEB strategy built from scratch on top of the repo execution assumptions, without changing ORB / VWAP research modules.",
        f"- Universe: `{', '.join(spec.symbols)}`.",
        "- Source data: 1-minute intraday futures data, resampled deterministically to 5-minute OHLCV bars.",
        f"- Session filter: RTH only `{spec.session_start}` -> `{spec.session_end}`.",
        f"- Allowed entry window: `{spec.entry_start}` -> `{spec.entry_end}`.",
        f"- V1 fixed parameters: ATR `{spec.atr_window}`, expansion threshold `{spec.expansion_threshold:.2f}`, breakout buffer `{spec.breakout_buffer_atr:.2f} * ATR`, time stop `{spec.time_stop_bars}` bars.",
        f"- Compression percentile calibration: `{spec.percentile_history_sessions}` prior RTH sessions ({spec.percentile_history_sessions * DEFAULT_BARS_PER_SESSION} bars) with minimum history `{spec.min_percentile_history_sessions}` sessions ({spec.min_percentile_history_sessions * DEFAULT_BARS_PER_SESSION} bars), strict ex ante only.",
        f"- IS/OOS split: chronological `{spec.is_fraction:.0%}` / `{1.0 - spec.is_fraction:.0%}` per instrument.",
        "",
        "## Data Coverage",
        "",
    ]
    for symbol, prepared in prepared_data.items():
        session_count = int(pd.to_datetime(prepared.feature_df["session_date"]).dt.date.nunique()) if not prepared.feature_df.empty else 0
        lines.append(
            f"- `{symbol}`: source `{prepared.source_dataset_path.name}`, resampled `{prepared.resampled_dataset_path.name}`, analysed sessions `{session_count}`."
        )

    lines.extend(
        [
            "",
            "## Cross-Asset Screening",
            "",
            "```text",
            screening_summary.head(12).to_string(index=False) if not screening_summary.empty else "No screening rows.",
            "```",
            "",
            "## Survivor Validation",
            "",
            "```text",
            survivor_validation_summary.to_string(index=False) if not survivor_validation_summary.empty else "No survivor validation rows.",
            "```",
            "",
            "## Research Verdict",
            "",
            f"- Verdict: `{verdict['research_verdict']}`.",
            f"- Best variant: `{verdict.get('best_variant_name')}`.",
            f"- Cross-asset read: `{verdict.get('cross_asset_character')}`.",
            f"- OOS positive assets: `{verdict.get('oos_positive_assets')}`.",
            f"- OOS total trades: `{verdict.get('oos_total_trades')}`.",
            f"- OOS total net PnL: `{_safe_float(verdict.get('oos_total_net_pnl')):.2f}` USD.",
            f"- Conclusion: {verdict.get('reason')}",
            "",
            "## Export Inventory",
            "",
            "- `screening_summary.csv`",
            "- `instrument_variant_summary.csv`",
            "- `oos_yearly_summary.csv`",
            "- `stress_test_summary.csv`",
            "- `survivor_validation_summary.csv`",
            "- `final_report.md`",
            "- `final_verdict.json`",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_campaign(spec: VCEBCampaignSpec) -> dict[str, Path]:
    """Run the full VCEB screening + validation campaign."""
    ensure_directories()
    output_root = Path(spec.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    variants = _build_variants(spec)
    prepared_data: dict[str, PreparedInstrumentData] = {}
    evaluation_map: dict[tuple[str, str], VCEBEvaluation] = {}
    instrument_rows: list[dict[str, Any]] = []

    for symbol in spec.symbols:
        prepared = _prepare_instrument_data(symbol, spec, output_root)
        prepared_data[symbol] = prepared
        for variant in variants:
            evaluation = _evaluate_variant(prepared, variant, spec)
            evaluation_map[(symbol, variant.name)] = evaluation
            instrument_rows.append(_instrument_variant_row(evaluation, prepared))

    instrument_variant_summary = (
        pd.DataFrame(instrument_rows).sort_values(["symbol", "variant_name"], ascending=[True, True]).reset_index(drop=True)
        if instrument_rows
        else pd.DataFrame()
    )
    screening_summary = _aggregate_screening(instrument_variant_summary)
    survivors = _select_survivors(screening_summary, spec)

    oos_yearly_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    survivors_dir = output_root / "survivors"
    survivors_dir.mkdir(parents=True, exist_ok=True)

    for survivor in survivors.to_dict(orient="records"):
        variant_name = str(survivor["variant_name"])
        variant_dir = survivors_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        for symbol in spec.symbols:
            evaluation = evaluation_map[(symbol, variant_name)]
            oos_yearly_rows.extend(_oos_yearly_rows(evaluation))
            stress_rows.extend(_stress_rows(evaluation, spec))

            oos_trades = _subset_frame_by_sessions(evaluation.result.trades, evaluation.oos_sessions)
            _entry_hour_table(oos_trades).to_csv(variant_dir / f"{symbol.lower()}_oos_entry_hour_summary.csv", index=False)
            evaluation.summary_by_scope.to_csv(variant_dir / f"{symbol.lower()}_metrics_by_scope.csv", index=False)
            evaluation.result.trades.to_csv(variant_dir / f"{symbol.lower()}_trades.csv", index=False)
            evaluation.result.daily_results.to_csv(variant_dir / f"{symbol.lower()}_daily_results.csv", index=False)

    oos_yearly_summary = (
        pd.DataFrame(oos_yearly_rows).sort_values(["variant_name", "symbol", "year"], ascending=[True, True, True]).reset_index(drop=True)
        if oos_yearly_rows
        else pd.DataFrame(columns=OOS_YEARLY_COLUMNS)
    )
    stress_test_summary = (
        pd.DataFrame(stress_rows).sort_values(["variant_name", "symbol", "scenario"], ascending=[True, True, True]).reset_index(drop=True)
        if stress_rows
        else pd.DataFrame(columns=STRESS_COLUMNS)
    )
    survivor_validation_summary = _survivor_summary_rows(
        survivors=survivors,
        evaluation_map=evaluation_map,
        stress_df=stress_test_summary,
    )

    screening_path = output_root / "screening_summary.csv"
    instrument_path = output_root / "instrument_variant_summary.csv"
    yearly_path = output_root / "oos_yearly_summary.csv"
    stress_path = output_root / "stress_test_summary.csv"
    survivor_path = output_root / "survivor_validation_summary.csv"
    report_path = output_root / "final_report.md"
    verdict_path = output_root / "final_verdict.json"
    metadata_path = output_root / "run_metadata.json"

    screening_summary.to_csv(screening_path, index=False)
    instrument_variant_summary.to_csv(instrument_path, index=False)
    oos_yearly_summary.to_csv(yearly_path, index=False)
    stress_test_summary.to_csv(stress_path, index=False)
    survivor_validation_summary.to_csv(survivor_path, index=False)

    verdict = _build_final_verdict(screening_summary, survivor_validation_summary)
    _write_final_report(
        report_path,
        spec=spec,
        prepared_data=prepared_data,
        screening_summary=screening_summary,
        survivor_validation_summary=survivor_validation_summary,
        verdict=verdict,
    )
    _json_dump(verdict_path, verdict)
    _json_dump(
        metadata_path,
        {
            "run_timestamp": datetime.now().isoformat(),
            "symbols": list(spec.symbols),
            "variant_count": len(variants),
            "survivor_count": int(len(survivors)),
            "start_date": spec.start_date,
            "end_date": spec.end_date,
            "output_root": output_root,
            "source_datasets": {symbol: prepared.source_dataset_path for symbol, prepared in prepared_data.items()},
        },
    )

    return {
        "output_root": output_root,
        "screening_summary": screening_path,
        "instrument_variant_summary": instrument_path,
        "oos_yearly_summary": yearly_path,
        "stress_test_summary": stress_path,
        "survivor_validation_summary": survivor_path,
        "final_report": report_path,
        "final_verdict": verdict_path,
        "run_metadata": metadata_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VCEB intraday validation campaign.")
    parser.add_argument("--output-root", type=str, default=None, help="Export directory for the VCEB run.")
    parser.add_argument("--symbols", nargs="*", default=None, help="Optional symbol override.")
    parser.add_argument("--start-date", type=str, default=None, help="Optional analysis start date YYYY-MM-DD.")
    parser.add_argument("--end-date", type=str, default=None, help="Optional analysis end date YYYY-MM-DD.")
    parser.add_argument("--max-validation-survivors", type=int, default=None, help="Override the number of survivors validated.")
    args = parser.parse_args()

    spec = build_default_campaign_spec(output_root=Path(args.output_root) if args.output_root else None)
    if args.symbols:
        spec = replace(spec, symbols=tuple(str(symbol).upper() for symbol in args.symbols))
    if args.start_date or args.end_date:
        spec = replace(spec, start_date=args.start_date, end_date=args.end_date)
    if args.max_validation_survivors is not None:
        spec = replace(spec, max_validation_survivors=int(args.max_validation_survivors))

    artifacts = run_campaign(spec)
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
