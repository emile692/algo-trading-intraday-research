"""Leak-free VIX/VVIX validation campaign for the validated MNQ ORB baseline."""

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
from src.config.paths import EXPORTS_DIR, ensure_directories
from src.config.settings import get_instrument_spec
from src.engine.backtester import run_backtest
from src.engine.execution_model import ExecutionModel
from src.features.implied_volatility import (
    DEFAULT_VIX_DAILY_PATH,
    DEFAULT_VVIX_DAILY_PATH,
    load_vix_daily,
    load_vix_vvix_daily_features,
    load_vvix_daily,
)


SUMMARY_COLUMNS = [
    "variant_name",
    "block",
    "family",
    "feature_name",
    "description",
    "calibration_scope",
    "selection_rule",
    "kept_buckets",
    "parameters_json",
    "screening_score",
    "validation_score",
    "screening_status",
    "verdict",
    "overall_net_pnl",
    "overall_sharpe",
    "overall_sortino",
    "overall_profit_factor",
    "overall_expectancy",
    "overall_max_drawdown",
    "overall_n_trades",
    "overall_n_days_traded",
    "overall_pct_days_traded",
    "overall_hit_rate",
    "overall_avg_win",
    "overall_avg_loss",
    "overall_stop_hit_rate",
    "overall_target_hit_rate",
    "overall_exposure_time_pct",
    "is_net_pnl",
    "is_sharpe",
    "is_sortino",
    "is_profit_factor",
    "is_expectancy",
    "is_max_drawdown",
    "is_n_trades",
    "is_n_days_traded",
    "is_pct_days_traded",
    "is_hit_rate",
    "is_avg_win",
    "is_avg_loss",
    "is_stop_hit_rate",
    "is_target_hit_rate",
    "is_exposure_time_pct",
    "oos_net_pnl",
    "oos_sharpe",
    "oos_sortino",
    "oos_profit_factor",
    "oos_expectancy",
    "oos_max_drawdown",
    "oos_n_trades",
    "oos_n_days_traded",
    "oos_pct_days_traded",
    "oos_hit_rate",
    "oos_avg_win",
    "oos_avg_loss",
    "oos_stop_hit_rate",
    "oos_target_hit_rate",
    "oos_exposure_time_pct",
    "is_trade_coverage_vs_baseline",
    "is_net_pnl_retention_vs_baseline",
    "is_sharpe_delta_vs_baseline",
    "is_sortino_delta_vs_baseline",
    "is_profit_factor_delta_vs_baseline",
    "is_expectancy_delta_vs_baseline",
    "is_hit_rate_delta_vs_baseline",
    "is_stop_hit_rate_delta_vs_baseline",
    "is_avg_win_delta_vs_baseline",
    "is_avg_loss_delta_vs_baseline",
    "is_exposure_delta_vs_baseline",
    "is_max_drawdown_improvement_vs_baseline",
    "oos_trade_coverage_vs_baseline",
    "oos_day_coverage_vs_baseline",
    "oos_net_pnl_retention_vs_baseline",
    "oos_sharpe_delta_vs_baseline",
    "oos_sortino_delta_vs_baseline",
    "oos_profit_factor_delta_vs_baseline",
    "oos_expectancy_delta_vs_baseline",
    "oos_hit_rate_delta_vs_baseline",
    "oos_stop_hit_rate_delta_vs_baseline",
    "oos_avg_win_delta_vs_baseline",
    "oos_avg_loss_delta_vs_baseline",
    "oos_exposure_delta_vs_baseline",
    "oos_max_drawdown_improvement_vs_baseline",
]


@dataclass(frozen=True)
class BucketCalibration:
    feature_name: str
    bucket_kind: str
    labels: tuple[str, ...]
    bins: tuple[float, ...] = ()


@dataclass(frozen=True)
class ImpliedVolFeatureSpec:
    name: str
    family: str
    description: str


@dataclass(frozen=True)
class ImpliedVolVariantSpec:
    name: str
    block: str
    family: str
    feature_name: str
    description: str
    selection_rule: str
    kept_buckets: tuple[str, ...]
    calibration_scope: str = "is_only"


@dataclass
class VariantRun:
    name: str
    block: str
    family: str
    feature_name: str
    description: str
    selection_rule: str
    kept_buckets: tuple[str, ...]
    calibration_scope: str
    parameters: dict[str, Any]
    controls: pd.DataFrame
    trades: pd.DataFrame
    daily_results: pd.DataFrame
    summary_by_scope: pd.DataFrame
    note: str = ""


@dataclass(frozen=True)
class MnqOrbVixVvixValidationSpec:
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
    initial_capital_usd: float = 50_000.0
    fixed_contracts: int = 1
    commission_per_side_usd: float | None = 0.62
    slippage_ticks: float | None = 1.0
    vix_daily_path: Path = DEFAULT_VIX_DAILY_PATH
    vvix_daily_path: Path = DEFAULT_VVIX_DAILY_PATH
    bucket_count: int = 3
    min_bucket_obs_is: int = 50
    min_oos_trades_for_positive: int = 20
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


def _bucket_labels(bucket_count: int) -> tuple[str, ...]:
    if bucket_count == 3:
        return ("low", "mid", "high")
    if bucket_count == 4:
        return ("very_low", "low_mid", "high_mid", "high")
    return tuple(f"bucket_{idx + 1}" for idx in range(bucket_count))


def calibrate_quantile_buckets(feature_name: str, is_values: pd.Series, bucket_count: int) -> BucketCalibration:
    clean = pd.to_numeric(is_values, errors="coerce").dropna()
    if len(clean) < bucket_count:
        raise ValueError(f"Not enough IS observations to calibrate {feature_name}.")
    _, bins = pd.qcut(clean, q=bucket_count, labels=False, retbins=True, duplicates="drop")
    bins_tuple = tuple(float(value) for value in bins)
    actual_bucket_count = len(bins_tuple) - 1
    if actual_bucket_count < 2:
        raise ValueError(f"Quantile calibration collapsed for {feature_name}.")
    return BucketCalibration(
        feature_name=feature_name,
        bucket_kind="quantile",
        labels=_bucket_labels(actual_bucket_count),
        bins=bins_tuple,
    )


def calibrate_categorical_buckets(feature_name: str, labels: pd.Series) -> BucketCalibration:
    ordered = tuple(str(value) for value in labels.dropna().astype(str).drop_duplicates().tolist())
    if not ordered:
        raise ValueError(f"No categorical labels available for {feature_name}.")
    return BucketCalibration(feature_name=feature_name, bucket_kind="categorical", labels=ordered)


def apply_bucket_calibration(values: pd.Series, calibration: BucketCalibration) -> pd.Series:
    if calibration.bucket_kind == "categorical":
        clean = pd.Series(values).astype("object")
        return clean.where(clean.astype(str).isin(set(calibration.labels))).astype("string")
    if not calibration.bins:
        return pd.Series(pd.NA, index=values.index, dtype="string")

    bins = list(calibration.bins)
    bins[0] = -np.inf
    bins[-1] = np.inf
    bucketed = pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=bins,
        labels=list(calibration.labels),
        include_lowest=True,
    )
    return pd.Series(bucketed, index=values.index, dtype="string")


def _bucket_bounds(calibration: BucketCalibration, label: str) -> tuple[float | None, float | None]:
    if calibration.bucket_kind != "quantile" or not calibration.bins:
        return None, None
    try:
        position = list(calibration.labels).index(str(label))
    except ValueError:
        return None, None
    lower = calibration.bins[position] if position < len(calibration.bins) else None
    upper = calibration.bins[position + 1] if (position + 1) < len(calibration.bins) else None
    return lower, upper


def _selected_ensemble_sessions(analysis: SymbolAnalysis, aggregation_rule: str) -> set:
    point_pass = analyze_symbol_cache_pass_matrix(analysis)
    pass_cols = [column for column in point_pass.columns if column.startswith("pass__")]
    if not pass_cols:
        return set()
    scored = point_pass.copy()
    scored["consensus_score"] = scored[pass_cols].sum(axis=1) / float(len(pass_cols))
    threshold = resolve_aggregation_threshold(aggregation_rule)
    return set(pd.to_datetime(scored.loc[scored["consensus_score"] >= threshold, "session_date"]).dt.date)


def _build_execution_model(
    symbol: str,
    commission_per_side_usd: float | None,
    slippage_ticks: float | None,
) -> tuple[ExecutionModel, dict[str, Any]]:
    instrument_spec = get_instrument_spec(symbol)
    execution_model = ExecutionModel(
        commission_per_side_usd=float(
            instrument_spec["commission_per_side_usd"] if commission_per_side_usd is None else commission_per_side_usd
        ),
        slippage_ticks=float(instrument_spec["slippage_ticks"] if slippage_ticks is None else slippage_ticks),
        tick_size=float(instrument_spec["tick_size"]),
    )
    return execution_model, instrument_spec


def _scale_trade_log_to_fixed_contracts(
    trades: pd.DataFrame,
    quantity: int,
    initial_capital: float,
    point_value_usd: float,
) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    out = trades.copy()
    if quantity <= 1:
        out["quantity"] = int(quantity)
        return out
    scale = float(quantity)
    for column in [
        "risk_budget_usd",
        "actual_risk_usd",
        "trade_risk_usd",
        "notional_usd",
        "pnl_usd",
        "fees",
        "net_pnl_usd",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce") * scale
    out["quantity"] = int(quantity)
    if "entry_price" in out.columns:
        entry_price = pd.to_numeric(out["entry_price"], errors="coerce")
        out["notional_usd"] = entry_price * float(point_value_usd) * float(quantity)
        out["leverage_used"] = out["notional_usd"] / float(max(initial_capital, 1.0))
    return out


def _build_fixed_nominal_baseline(
    analysis: SymbolAnalysis,
    selected_sessions: set,
    spec: MnqOrbVixVvixValidationSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_df = analysis.signal_df.copy()
    signal_df["session_date"] = pd.to_datetime(signal_df["session_date"]).dt.date
    selected_mask = signal_df["session_date"].isin(selected_sessions)
    signal_df["selected_by_atr_baseline"] = selected_mask
    signal_df.loc[~selected_mask, "signal"] = 0

    execution_model, instrument_spec = _build_execution_model(
        analysis.symbol,
        commission_per_side_usd=spec.commission_per_side_usd,
        slippage_ticks=spec.slippage_ticks,
    )
    trades = run_backtest(
        signal_df,
        execution_model=execution_model,
        tick_value_usd=float(instrument_spec["tick_value_usd"]),
        point_value_usd=float(instrument_spec["point_value_usd"]),
        time_exit=analysis.baseline.time_exit,
        stop_buffer_ticks=analysis.baseline.stop_buffer_ticks,
        target_multiple=analysis.baseline.target_multiple,
        account_size_usd=None,
        risk_per_trade_pct=None,
        entry_on_next_open=analysis.baseline.entry_on_next_open,
    )
    trades = _scale_trade_log_to_fixed_contracts(
        trades=trades,
        quantity=int(spec.fixed_contracts),
        initial_capital=float(spec.initial_capital_usd),
        point_value_usd=float(instrument_spec["point_value_usd"]),
    )
    return signal_df, trades


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


def _daily_results_from_trades(trades: pd.DataFrame, sessions: list, initial_capital: float) -> pd.DataFrame:
    session_index = pd.Index(pd.to_datetime(pd.Index(sessions)).date)
    daily = pd.DataFrame({"session_date": session_index})
    if trades.empty:
        daily["daily_pnl_usd"] = 0.0
        daily["daily_trade_count"] = 0
    else:
        view = trades.copy()
        view["session_date"] = pd.to_datetime(view["session_date"]).dt.date
        grouped = (
            view.groupby("session_date", as_index=False)
            .agg(daily_pnl_usd=("net_pnl_usd", "sum"), daily_trade_count=("trade_id", "count"))
        )
        daily = daily.merge(grouped, on="session_date", how="left").fillna({"daily_pnl_usd": 0.0, "daily_trade_count": 0})
    daily = daily.sort_values("session_date").reset_index(drop=True)
    daily["equity"] = float(initial_capital) + pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").cumsum()
    daily["peak_equity"] = daily["equity"].cummax()
    daily["drawdown_usd"] = daily["equity"] - daily["peak_equity"]
    daily["drawdown_pct"] = np.where(
        daily["peak_equity"] > 0,
        (daily["equity"] - daily["peak_equity"]) / daily["peak_equity"],
        0.0,
    )
    return daily


def _trade_holding_minutes(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    entry_time = pd.to_datetime(trades["entry_time"], errors="coerce")
    exit_time = pd.to_datetime(trades["exit_time"], errors="coerce")
    return ((exit_time - entry_time).dt.total_seconds() / 60.0).clip(lower=0.0)


def _scope_summary(
    trades: pd.DataFrame,
    sessions: list,
    initial_capital: float,
    session_minutes: float,
) -> dict[str, Any]:
    base = compute_metrics(trades, session_dates=sessions, initial_capital=initial_capital)
    daily = _daily_results_from_trades(trades, sessions, initial_capital)
    holding_minutes = _trade_holding_minutes(trades)
    n_sessions = max(len(sessions), 1)
    return {
        "net_pnl": float(base.get("cumulative_pnl", 0.0)),
        "sharpe": float(base.get("sharpe_ratio", 0.0)),
        "sortino": _sortino_ratio(pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0), initial_capital),
        "profit_factor": float(base.get("profit_factor", 0.0)),
        "expectancy": float(base.get("expectancy", 0.0)),
        "max_drawdown": float(base.get("max_drawdown", 0.0)),
        "n_trades": int(base.get("n_trades", 0)),
        "n_days_traded": int((pd.to_numeric(daily["daily_trade_count"], errors="coerce").fillna(0.0) > 0).sum()),
        "pct_days_traded": float(base.get("percent_of_days_traded", 0.0)),
        "hit_rate": float(base.get("win_rate", 0.0)),
        "avg_win": float(base.get("avg_win", 0.0)),
        "avg_loss": float(base.get("avg_loss", 0.0)),
        "stop_hit_rate": float(base.get("stop_hit_rate", 0.0)),
        "target_hit_rate": float(base.get("target_hit_rate", 0.0)),
        "exposure_time_pct": float(holding_minutes.sum() / max(n_sessions * session_minutes, 1.0)),
    }


def _build_summary_by_scope(
    trades: pd.DataFrame,
    all_sessions: list,
    is_sessions: list,
    oos_sessions: list,
    initial_capital: float,
    session_minutes: float,
) -> pd.DataFrame:
    rows = []
    for scope, sessions in (("overall", all_sessions), ("is", is_sessions), ("oos", oos_sessions)):
        sub = trades.copy()
        if not sub.empty:
            sub["session_date"] = pd.to_datetime(sub["session_date"]).dt.date
            sub = sub.loc[sub["session_date"].isin(set(pd.to_datetime(pd.Index(sessions)).date))].copy()
        rows.append(
            {
                "scope": scope,
                **_scope_summary(
                    trades=sub,
                    sessions=sessions,
                    initial_capital=initial_capital,
                    session_minutes=session_minutes,
                ),
            }
        )
    return pd.DataFrame(rows)


def _scope_value(summary_by_scope: pd.DataFrame, scope: str, column: str) -> Any:
    row = summary_by_scope.loc[summary_by_scope["scope"] == scope]
    if row.empty:
        return np.nan
    return row.iloc[0].get(column, np.nan)


def _trade_subset(trades: pd.DataFrame, sessions: list) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    session_set = set(pd.to_datetime(pd.Index(sessions)).date)
    out = trades.copy()
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.date
    return out.loc[out["session_date"].isin(session_set)].copy()


def _implied_vol_feature_specs() -> tuple[ImpliedVolFeatureSpec, ...]:
    return (
        ImpliedVolFeatureSpec("vix_level_t1", "vix_level", "Lagged VIX daily close known before the trade session."),
        ImpliedVolFeatureSpec("vvix_level_t1", "vvix_level", "Lagged VVIX daily close known before the trade session."),
        ImpliedVolFeatureSpec("vvix_over_vix_t1", "ratio", "Lagged VVIX/VIX ratio as a simple vol-of-vol stress proxy."),
        ImpliedVolFeatureSpec("vix_pct_63_t1", "vix_percentile", "63-day percentile of lagged VIX level."),
        ImpliedVolFeatureSpec("vix_pct_126_t1", "vix_percentile", "126-day percentile of lagged VIX level."),
        ImpliedVolFeatureSpec("vix_pct_252_t1", "vix_percentile", "252-day percentile of lagged VIX level."),
        ImpliedVolFeatureSpec("vvix_pct_63_t1", "vvix_percentile", "63-day percentile of lagged VVIX level."),
        ImpliedVolFeatureSpec("vvix_pct_126_t1", "vvix_percentile", "126-day percentile of lagged VVIX level."),
        ImpliedVolFeatureSpec("vvix_pct_252_t1", "vvix_percentile", "252-day percentile of lagged VVIX level."),
        ImpliedVolFeatureSpec("vix_change_1d", "vix_change", "1-day percent change of the lagged VIX level."),
        ImpliedVolFeatureSpec("vix_change_5d", "vix_change", "5-day percent change of the lagged VIX level."),
        ImpliedVolFeatureSpec("vvix_change_1d", "vvix_change", "1-day percent change of the lagged VVIX level."),
        ImpliedVolFeatureSpec("vvix_change_5d", "vvix_change", "5-day percent change of the lagged VVIX level."),
    )


def build_vix_vvix_feature_frame(
    analysis: SymbolAnalysis,
    baseline_trades: pd.DataFrame,
    daily_features: pd.DataFrame,
) -> pd.DataFrame:
    selected_sessions = set(pd.to_datetime(baseline_trades["session_date"]).dt.date) if not baseline_trades.empty else set()
    candidates = analysis.candidate_df.copy()
    candidates["session_date"] = pd.to_datetime(candidates["session_date"]).dt.date
    candidates = candidates.loc[candidates["session_date"].isin(selected_sessions), ["session_date", "signal_index"]].copy()
    if candidates.empty:
        return pd.DataFrame()

    signal_rows = analysis.signal_df.loc[candidates["signal_index"].tolist()].copy()
    signal_rows = signal_rows.reset_index().rename(columns={"index": "signal_index"})
    signal_rows["session_date"] = pd.to_datetime(signal_rows["session_date"]).dt.date
    signal_rows["timestamp"] = pd.to_datetime(signal_rows["timestamp"], errors="coerce")

    baseline_view = baseline_trades.copy()
    baseline_view["session_date"] = pd.to_datetime(baseline_view["session_date"]).dt.date

    daily_view = daily_features.copy()
    daily_view["session_date"] = pd.to_datetime(daily_view["session_date"]).dt.date

    feature_frame = candidates.merge(
        signal_rows[
            [
                "session_date",
                "signal_index",
                "timestamp",
                "close",
                "or_high",
                "or_low",
                "or_width",
                "signal",
            ]
        ],
        on=["session_date", "signal_index"],
        how="left",
    )
    feature_frame = feature_frame.merge(
        baseline_view[
            [
                "session_date",
                "trade_id",
                "entry_time",
                "exit_time",
                "direction",
                "quantity",
                "net_pnl_usd",
                "pnl_usd",
                "fees",
                "exit_reason",
            ]
        ],
        on="session_date",
        how="inner",
    )
    feature_frame = feature_frame.merge(daily_view, on="session_date", how="inner", validate="one_to_one")
    feature_frame = feature_frame.sort_values("session_date").reset_index(drop=True)

    is_set = set(pd.to_datetime(pd.Index(analysis.is_sessions)).date)
    feature_frame["phase"] = np.where(feature_frame["session_date"].isin(is_set), "is", "oos")
    feature_frame["breakout_side"] = feature_frame["direction"].astype("string")

    open_minutes = pd.Timestamp(analysis.baseline.opening_time).hour * 60 + pd.Timestamp(analysis.baseline.opening_time).minute
    or_end_minutes = open_minutes + int(analysis.baseline.or_minutes)
    feature_frame["minute_of_day"] = pd.to_datetime(feature_frame["timestamp"], errors="coerce").dt.hour * 60 + pd.to_datetime(
        feature_frame["timestamp"],
        errors="coerce",
    ).dt.minute
    feature_frame["breakout_offset_minutes"] = pd.to_numeric(feature_frame["minute_of_day"], errors="coerce") - or_end_minutes
    feature_frame["breakout_timing_bucket"] = np.select(
        [
            pd.to_numeric(feature_frame["breakout_offset_minutes"], errors="coerce") <= 15,
            pd.to_numeric(feature_frame["breakout_offset_minutes"], errors="coerce") <= 60,
        ],
        ["early", "mid"],
        default="late",
    )
    feature_frame["selected_by_baseline_atr"] = True
    return feature_frame


def build_regime_summary(
    feature_frame: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    all_sessions: list,
    is_sessions: list,
    oos_sessions: list,
    initial_capital: float,
    session_minutes: float,
    feature_specs: tuple[ImpliedVolFeatureSpec, ...],
    bucket_count: int,
    min_bucket_obs_is: int,
) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, BucketCalibration]]:
    rows: list[dict[str, Any]] = []
    assignments: dict[str, pd.Series] = {}
    calibrations: dict[str, BucketCalibration] = {}

    for feature in feature_specs:
        try:
            calibration = calibrate_quantile_buckets(
                feature_name=feature.name,
                is_values=feature_frame.loc[feature_frame["phase"] == "is", feature.name],
                bucket_count=bucket_count,
            )
        except ValueError:
            continue

        labels = apply_bucket_calibration(feature_frame[feature.name], calibration)
        assignments[feature.name] = labels
        calibrations[feature.name] = calibration

        for label in calibration.labels:
            label_sessions = feature_frame.loc[labels.eq(label), "session_date"].tolist()
            label_trades = _trade_subset(baseline_trades, label_sessions)
            summary = _build_summary_by_scope(
                trades=label_trades,
                all_sessions=all_sessions,
                is_sessions=is_sessions,
                oos_sessions=oos_sessions,
                initial_capital=initial_capital,
                session_minutes=session_minutes,
            )
            is_n_obs = int((labels.eq(label) & feature_frame["phase"].eq("is")).sum())
            oos_n_obs = int((labels.eq(label) & feature_frame["phase"].eq("oos")).sum())
            lower_bound, upper_bound = _bucket_bounds(calibration, label)
            rows.append(
                {
                    "feature_name": feature.name,
                    "family": feature.family,
                    "description": feature.description,
                    "bucket_label": label,
                    "bucket_kind": calibration.bucket_kind,
                    "bucket_position": list(calibration.labels).index(label) + 1,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "is_n_obs": is_n_obs,
                    "oos_n_obs": oos_n_obs,
                    "valid_for_variants": bool(is_n_obs >= min_bucket_obs_is),
                    "is_net_pnl": _scope_value(summary, "is", "net_pnl"),
                    "is_sharpe": _scope_value(summary, "is", "sharpe"),
                    "is_expectancy": _scope_value(summary, "is", "expectancy"),
                    "is_profit_factor": _scope_value(summary, "is", "profit_factor"),
                    "is_hit_rate": _scope_value(summary, "is", "hit_rate"),
                    "oos_net_pnl": _scope_value(summary, "oos", "net_pnl"),
                    "oos_sharpe": _scope_value(summary, "oos", "sharpe"),
                    "oos_expectancy": _scope_value(summary, "oos", "expectancy"),
                    "oos_profit_factor": _scope_value(summary, "oos", "profit_factor"),
                    "oos_hit_rate": _scope_value(summary, "oos", "hit_rate"),
                }
            )

    if "vix_level_t1" in assignments and "vvix_level_t1" in assignments:
        vix_labels = assignments["vix_level_t1"].astype("string")
        vvix_labels = assignments["vvix_level_t1"].astype("string")
        valid_mask = vix_labels.notna() & vvix_labels.notna()
        cross_labels = pd.Series(pd.NA, index=feature_frame.index, dtype="string")
        cross_labels.loc[valid_mask] = (
            "vix_"
            + vix_labels.loc[valid_mask].astype(str)
            + "__vvix_"
            + vvix_labels.loc[valid_mask].astype(str)
        )
        ordered_cross_labels = tuple(
            label
            for label in (
                f"vix_{vix_bucket}__vvix_{vvix_bucket}"
                for vix_bucket in ("low", "mid", "high")
                for vvix_bucket in ("low", "mid", "high")
            )
            if label in set(cross_labels.dropna().astype(str))
        )
        try:
            cross_calibration = calibrate_categorical_buckets(
                feature_name="vix_x_vvix_regime",
                labels=pd.Series(list(ordered_cross_labels), dtype="string"),
            )
        except ValueError:
            cross_calibration = None

        if cross_calibration is not None:
            assignments["vix_x_vvix_regime"] = cross_labels
            calibrations["vix_x_vvix_regime"] = cross_calibration
            for label in cross_calibration.labels:
                label_sessions = feature_frame.loc[cross_labels.eq(label), "session_date"].tolist()
                label_trades = _trade_subset(baseline_trades, label_sessions)
                summary = _build_summary_by_scope(
                    trades=label_trades,
                    all_sessions=all_sessions,
                    is_sessions=is_sessions,
                    oos_sessions=oos_sessions,
                    initial_capital=initial_capital,
                    session_minutes=session_minutes,
                )
                rows.append(
                    {
                        "feature_name": "vix_x_vvix_regime",
                        "family": "cross_regime",
                        "description": "Joint low/mid/high regime from IS-calibrated VIX and VVIX buckets.",
                        "bucket_label": label,
                        "bucket_kind": "categorical",
                        "bucket_position": list(cross_calibration.labels).index(label) + 1,
                        "lower_bound": np.nan,
                        "upper_bound": np.nan,
                        "is_n_obs": int((cross_labels.eq(label) & feature_frame["phase"].eq("is")).sum()),
                        "oos_n_obs": int((cross_labels.eq(label) & feature_frame["phase"].eq("oos")).sum()),
                        "valid_for_variants": bool(
                            int((cross_labels.eq(label) & feature_frame["phase"].eq("is")).sum()) >= min_bucket_obs_is
                        ),
                        "is_net_pnl": _scope_value(summary, "is", "net_pnl"),
                        "is_sharpe": _scope_value(summary, "is", "sharpe"),
                        "is_expectancy": _scope_value(summary, "is", "expectancy"),
                        "is_profit_factor": _scope_value(summary, "is", "profit_factor"),
                        "is_hit_rate": _scope_value(summary, "is", "hit_rate"),
                        "oos_net_pnl": _scope_value(summary, "oos", "net_pnl"),
                        "oos_sharpe": _scope_value(summary, "oos", "sharpe"),
                        "oos_expectancy": _scope_value(summary, "oos", "expectancy"),
                        "oos_profit_factor": _scope_value(summary, "oos", "profit_factor"),
                        "oos_hit_rate": _scope_value(summary, "oos", "hit_rate"),
                    }
                )

    return pd.DataFrame(rows), assignments, calibrations


def _standard_variant_specs(feature: ImpliedVolFeatureSpec, calibration: BucketCalibration) -> list[ImpliedVolVariantSpec]:
    labels = list(calibration.labels)
    if len(labels) < 2:
        return []
    low_label = labels[0]
    high_label = labels[-1]
    mid_label = labels[len(labels) // 2]

    return [
        ImpliedVolVariantSpec(
            name=f"regime_low__{feature.name}",
            block="regime_selection",
            family=feature.family,
            feature_name=feature.name,
            description=f"Trade only low bucket for {feature.name}.",
            selection_rule="bucket_equals_low",
            kept_buckets=(low_label,),
        ),
        ImpliedVolVariantSpec(
            name=f"regime_mid__{feature.name}",
            block="regime_selection",
            family=feature.family,
            feature_name=feature.name,
            description=f"Trade only mid bucket for {feature.name}.",
            selection_rule="bucket_equals_mid",
            kept_buckets=(mid_label,),
        ),
        ImpliedVolVariantSpec(
            name=f"regime_high__{feature.name}",
            block="regime_selection",
            family=feature.family,
            feature_name=feature.name,
            description=f"Trade only high bucket for {feature.name}.",
            selection_rule="bucket_equals_high",
            kept_buckets=(high_label,),
        ),
        ImpliedVolVariantSpec(
            name=f"filter_drop_low__{feature.name}",
            block="filter_as_veto",
            family=feature.family,
            feature_name=feature.name,
            description=f"Skip low bucket for {feature.name} and keep mid/high.",
            selection_rule="drop_low_bucket",
            kept_buckets=tuple(labels[1:]),
        ),
        ImpliedVolVariantSpec(
            name=f"filter_drop_high__{feature.name}",
            block="filter_as_veto",
            family=feature.family,
            feature_name=feature.name,
            description=f"Skip high bucket for {feature.name} and keep low/mid.",
            selection_rule="drop_high_bucket",
            kept_buckets=tuple(labels[:-1]),
        ),
    ]


def _cross_regime_variant_specs(calibration: BucketCalibration) -> list[ImpliedVolVariantSpec]:
    labels = set(calibration.labels)

    def _label(vix_bucket: str, vvix_bucket: str) -> str:
        return f"vix_{vix_bucket}__vvix_{vvix_bucket}"

    def _present(pairs: list[tuple[str, str]]) -> tuple[str, ...]:
        return tuple(_label(vix_bucket, vvix_bucket) for vix_bucket, vvix_bucket in pairs if _label(vix_bucket, vvix_bucket) in labels)

    stressed = _present([("high", "high")])
    elevated_contained = _present([("high", "low"), ("high", "mid")])
    calm_joint = _present([("low", "low"), ("low", "mid"), ("mid", "low"), ("mid", "mid")])
    orderly_mid = _present([("mid", "mid")])
    keep_not_stressed = tuple(label for label in calibration.labels if label not in set(stressed))
    keep_not_low_or_high = tuple(
        label
        for label in calibration.labels
        if not label.startswith("vix_low__") and "__vvix_high" not in label
    )

    variants: list[ImpliedVolVariantSpec] = []
    if stressed:
        variants.append(
            ImpliedVolVariantSpec(
                name="regime_elevated_stressed__vix_x_vvix",
                block="cross_regime",
                family="cross_regime",
                feature_name="vix_x_vvix_regime",
                description="Trade only regimes with elevated VIX and stressed VVIX.",
                selection_rule="keep_elevated_stressed",
                kept_buckets=stressed,
            )
        )
        variants.append(
            ImpliedVolVariantSpec(
                name="filter_drop_elevated_stressed__vix_x_vvix",
                block="filter_as_veto",
                family="cross_regime",
                feature_name="vix_x_vvix_regime",
                description="Skip elevated-and-stressed VIX/VVIX regimes.",
                selection_rule="drop_elevated_stressed",
                kept_buckets=keep_not_stressed,
            )
        )
    if elevated_contained:
        variants.append(
            ImpliedVolVariantSpec(
                name="regime_elevated_contained__vix_x_vvix",
                block="cross_regime",
                family="cross_regime",
                feature_name="vix_x_vvix_regime",
                description="Trade only elevated VIX regimes where VVIX stays contained.",
                selection_rule="keep_elevated_contained",
                kept_buckets=elevated_contained,
            )
        )
    if calm_joint:
        variants.append(
            ImpliedVolVariantSpec(
                name="regime_joint_calm__vix_x_vvix",
                block="cross_regime",
                family="cross_regime",
                feature_name="vix_x_vvix_regime",
                description="Trade only joint calm / orderly VIX-VVIX regimes.",
                selection_rule="keep_joint_calm",
                kept_buckets=calm_joint,
            )
        )
    if orderly_mid:
        variants.append(
            ImpliedVolVariantSpec(
                name="regime_joint_mid__vix_x_vvix",
                block="cross_regime",
                family="cross_regime",
                feature_name="vix_x_vvix_regime",
                description="Trade only balanced mid/mid VIX-VVIX regimes.",
                selection_rule="keep_joint_mid",
                kept_buckets=orderly_mid,
            )
        )
    if keep_not_low_or_high:
        variants.append(
            ImpliedVolVariantSpec(
                name="filter_skip_low_vix_or_high_vvix__vix_x_vvix",
                block="filter_as_veto",
                family="cross_regime",
                feature_name="vix_x_vvix_regime",
                description="Skip low-VIX regimes and also skip VVIX-high stressed regimes.",
                selection_rule="drop_vix_low_or_vvix_high",
                kept_buckets=keep_not_low_or_high,
            )
        )
    return variants


def _build_variant_run(
    variant_spec: ImpliedVolVariantSpec,
    feature_frame: pd.DataFrame,
    bucket_labels: pd.Series,
    baseline_trades: pd.DataFrame,
    all_sessions: list,
    is_sessions: list,
    oos_sessions: list,
    initial_capital: float,
    session_minutes: float,
) -> VariantRun:
    controls = feature_frame[["session_date", "phase", "breakout_side", "breakout_timing_bucket"]].copy()
    if variant_spec.feature_name in feature_frame.columns:
        controls["feature_value"] = feature_frame[variant_spec.feature_name]
    else:
        controls["feature_value"] = pd.NA
    controls["feature_name"] = variant_spec.feature_name
    controls["bucket_label"] = bucket_labels.astype("string")
    controls["selected"] = controls["bucket_label"].isin(set(variant_spec.kept_buckets))
    controls["selected_by_baseline_atr"] = True
    controls["skip_trade"] = ~controls["selected"]

    selected_sessions = controls.loc[controls["selected"], "session_date"].tolist()
    trades = _trade_subset(baseline_trades, selected_sessions)
    daily_results = _daily_results_from_trades(trades, all_sessions, initial_capital)
    summary_by_scope = _build_summary_by_scope(
        trades=trades,
        all_sessions=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        initial_capital=initial_capital,
        session_minutes=session_minutes,
    )
    return VariantRun(
        name=variant_spec.name,
        block=variant_spec.block,
        family=variant_spec.family,
        feature_name=variant_spec.feature_name,
        description=variant_spec.description,
        selection_rule=variant_spec.selection_rule,
        kept_buckets=variant_spec.kept_buckets,
        calibration_scope=variant_spec.calibration_scope,
        parameters={"kept_buckets": variant_spec.kept_buckets},
        controls=controls,
        trades=trades,
        daily_results=daily_results,
        summary_by_scope=summary_by_scope,
    )


def _baseline_variant(
    baseline_trades: pd.DataFrame,
    feature_frame: pd.DataFrame,
    all_sessions: list,
    is_sessions: list,
    oos_sessions: list,
    initial_capital: float,
    session_minutes: float,
) -> VariantRun:
    controls = feature_frame[["session_date", "phase", "breakout_side", "breakout_timing_bucket"]].copy()
    controls["feature_name"] = "baseline"
    controls["feature_value"] = np.nan
    controls["bucket_label"] = "baseline"
    controls["selected"] = True
    controls["selected_by_baseline_atr"] = True
    controls["skip_trade"] = False
    return VariantRun(
        name="baseline_fixed_nominal_atr",
        block="baseline",
        family="baseline",
        feature_name="baseline",
        description="Official ORB + ATR baseline, rerun with fixed nominal size on the VIX/VVIX coverage window.",
        selection_rule="baseline_selected_sessions",
        kept_buckets=("baseline",),
        calibration_scope="none",
        parameters={},
        controls=controls,
        trades=baseline_trades.copy(),
        daily_results=_daily_results_from_trades(baseline_trades, all_sessions, initial_capital),
        summary_by_scope=_build_summary_by_scope(
            trades=baseline_trades,
            all_sessions=all_sessions,
            is_sessions=is_sessions,
            oos_sessions=oos_sessions,
            initial_capital=initial_capital,
            session_minutes=session_minutes,
        ),
    )


def _score_variant_row(row: dict[str, Any], prefix: str) -> float:
    trade_cov = float(row.get(f"{prefix}_trade_coverage_vs_baseline", 0.0))
    sharpe_delta = float(row.get(f"{prefix}_sharpe_delta_vs_baseline", 0.0))
    expectancy_delta = float(row.get(f"{prefix}_expectancy_delta_vs_baseline", 0.0))
    dd_improvement = float(row.get(f"{prefix}_max_drawdown_improvement_vs_baseline", 0.0))
    hit_delta = float(row.get(f"{prefix}_hit_rate_delta_vs_baseline", 0.0))
    stop_delta = float(row.get(f"{prefix}_stop_hit_rate_delta_vs_baseline", 0.0))
    pnl_retention = float(row.get(f"{prefix}_net_pnl_retention_vs_baseline", 0.0))
    avg_loss_delta = float(row.get(f"{prefix}_avg_loss_delta_vs_baseline", 0.0))
    return float(
        1.80 * np.tanh(sharpe_delta)
        + 1.20 * np.tanh(expectancy_delta / 8.0)
        + 1.00 * np.tanh(3.0 * dd_improvement)
        + 0.80 * np.tanh(6.0 * hit_delta)
        + 0.80 * np.tanh(6.0 * stop_delta)
        + 0.50 * np.tanh(avg_loss_delta / 8.0)
        + 0.40 * np.tanh((pnl_retention - 1.0) / 0.10)
        - 1.10 * np.tanh(max(0.45 - trade_cov, 0.0) / 0.20)
    )


def _screening_status(row: dict[str, Any]) -> str:
    if row["variant_name"] == "baseline_fixed_nominal_atr":
        return "baseline_reference"
    if float(row.get("is_trade_coverage_vs_baseline", 0.0)) < 0.20:
        return "too_sparse"
    if float(row.get("screening_score", 0.0)) > 0.35:
        return "selected_for_validation"
    if float(row.get("screening_score", 0.0)) > 0.0:
        return "watchlist"
    return "screen_fail"


def _validation_verdict(row: dict[str, Any], min_oos_trades_for_positive: int) -> str:
    if row["variant_name"] == "baseline_fixed_nominal_atr":
        return "baseline_reference"
    oos_trades = int(row.get("oos_n_trades", 0))
    coverage = float(row.get("oos_trade_coverage_vs_baseline", 0.0))
    pnl_retention = float(row.get("oos_net_pnl_retention_vs_baseline", 0.0))
    validation_score = float(row.get("validation_score", 0.0))
    dd_improvement = float(row.get("oos_max_drawdown_improvement_vs_baseline", 0.0))
    expectancy_delta = float(row.get("oos_expectancy_delta_vs_baseline", 0.0))

    if oos_trades < min_oos_trades_for_positive:
        return "insufficient_oos"
    if validation_score > 0.40 and coverage >= 0.55 and pnl_retention >= 0.85:
        return "robust_positive"
    if dd_improvement > 0.10 and coverage >= 0.35:
        return "protective_filter"
    if validation_score > 0.0 and expectancy_delta > 0.0:
        return "mixed_positive"
    if float(row.get("screening_score", 0.0)) > 0.20 and validation_score <= 0.0:
        return "is_only"
    if coverage < 0.20:
        return "cuts_too_much_exposure"
    return "no_value"


def _variant_row(variant: VariantRun, baseline_variant: VariantRun, spec: MnqOrbVixVvixValidationSpec) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant_name": variant.name,
        "block": variant.block,
        "family": variant.family,
        "feature_name": variant.feature_name,
        "description": variant.description,
        "calibration_scope": variant.calibration_scope,
        "selection_rule": variant.selection_rule,
        "kept_buckets": ",".join(str(value) for value in variant.kept_buckets),
        "parameters_json": json.dumps({key: _serialize_value(value) for key, value in variant.parameters.items()}, sort_keys=True),
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
            "n_days_traded",
            "pct_days_traded",
            "hit_rate",
            "avg_win",
            "avg_loss",
            "stop_hit_rate",
            "target_hit_rate",
            "exposure_time_pct",
        ]:
            row[f"{scope}_{metric}"] = _scope_value(variant.summary_by_scope, scope, metric)

    baseline_oos_dd = abs(float(_scope_value(baseline_variant.summary_by_scope, "oos", "max_drawdown")))
    variant_oos_dd = abs(float(_scope_value(variant.summary_by_scope, "oos", "max_drawdown")))
    baseline_is_dd = abs(float(_scope_value(baseline_variant.summary_by_scope, "is", "max_drawdown")))
    variant_is_dd = abs(float(_scope_value(variant.summary_by_scope, "is", "max_drawdown")))
    row["is_trade_coverage_vs_baseline"] = _safe_div(
        float(_scope_value(variant.summary_by_scope, "is", "n_trades")),
        float(_scope_value(baseline_variant.summary_by_scope, "is", "n_trades")),
        default=0.0,
    )
    row["oos_trade_coverage_vs_baseline"] = _safe_div(
        float(_scope_value(variant.summary_by_scope, "oos", "n_trades")),
        float(_scope_value(baseline_variant.summary_by_scope, "oos", "n_trades")),
        default=0.0,
    )
    row["oos_day_coverage_vs_baseline"] = _safe_div(
        float(_scope_value(variant.summary_by_scope, "oos", "n_days_traded")),
        float(_scope_value(baseline_variant.summary_by_scope, "oos", "n_days_traded")),
        default=0.0,
    )
    row["oos_net_pnl_retention_vs_baseline"] = _safe_div(
        float(_scope_value(variant.summary_by_scope, "oos", "net_pnl")),
        float(_scope_value(baseline_variant.summary_by_scope, "oos", "net_pnl")),
        default=0.0,
    )
    row["is_net_pnl_retention_vs_baseline"] = _safe_div(
        float(_scope_value(variant.summary_by_scope, "is", "net_pnl")),
        float(_scope_value(baseline_variant.summary_by_scope, "is", "net_pnl")),
        default=0.0,
    )
    for scope in ("is", "oos"):
        for metric in ("sharpe", "sortino", "profit_factor", "expectancy", "hit_rate", "avg_win", "avg_loss", "exposure_time_pct"):
            row[f"{scope}_{metric}_delta_vs_baseline"] = float(_scope_value(variant.summary_by_scope, scope, metric)) - float(
                _scope_value(baseline_variant.summary_by_scope, scope, metric)
            )
        row[f"{scope}_stop_hit_rate_delta_vs_baseline"] = float(_scope_value(baseline_variant.summary_by_scope, scope, "stop_hit_rate")) - float(
            _scope_value(variant.summary_by_scope, scope, "stop_hit_rate")
        )
    row["is_max_drawdown_improvement_vs_baseline"] = _safe_div(baseline_is_dd - variant_is_dd, max(baseline_is_dd, 1.0), default=0.0)
    row["oos_max_drawdown_improvement_vs_baseline"] = _safe_div(
        baseline_oos_dd - variant_oos_dd,
        max(baseline_oos_dd, 1.0),
        default=0.0,
    )
    row["screening_score"] = _score_variant_row(row, prefix="is")
    row["validation_score"] = _score_variant_row(row, prefix="oos")
    row["screening_status"] = _screening_status(row)
    row["verdict"] = _validation_verdict(row, min_oos_trades_for_positive=spec.min_oos_trades_for_positive)
    return row


def _feature_importance_like_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df.copy()
    rows: list[dict[str, Any]] = []
    filtered = results_df.loc[results_df["variant_name"].ne("baseline_fixed_nominal_atr")].copy()
    for (family, feature_name), group in filtered.groupby(["family", "feature_name"], dropna=False):
        ordered = group.sort_values(["validation_score", "oos_sharpe_delta_vs_baseline"], ascending=[False, False]).reset_index(drop=True)
        best = ordered.iloc[0]
        rows.append(
            {
                "family": family,
                "feature_name": feature_name,
                "n_variants": int(len(group)),
                "n_screen_selected": int((group["screening_status"] == "selected_for_validation").sum()),
                "n_robust_positive": int((group["verdict"] == "robust_positive").sum()),
                "best_variant_name": best["variant_name"],
                "best_variant_verdict": best["verdict"],
                "best_validation_score": float(best["validation_score"]),
                "best_oos_sharpe_delta_vs_baseline": float(best["oos_sharpe_delta_vs_baseline"]),
                "best_oos_expectancy_delta_vs_baseline": float(best["oos_expectancy_delta_vs_baseline"]),
                "best_oos_dd_improvement_vs_baseline": float(best["oos_max_drawdown_improvement_vs_baseline"]),
                "median_oos_trade_coverage_vs_baseline": float(pd.to_numeric(group["oos_trade_coverage_vs_baseline"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_robust_positive", "best_validation_score"], ascending=[False, False]).reset_index(drop=True)


def _interaction_summary(
    feature_frame: pd.DataFrame,
    baseline_variant: VariantRun,
    variants: list[VariantRun],
    results_df: pd.DataFrame,
    initial_capital: float,
    session_minutes: float,
) -> pd.DataFrame:
    robust_names = results_df.loc[results_df["verdict"].isin(["robust_positive", "protective_filter"]), "variant_name"].head(5).tolist()
    if not robust_names:
        robust_names = results_df.loc[results_df["variant_name"].ne("baseline_fixed_nominal_atr"), "variant_name"].head(3).tolist()
    selected_variants = [variant for variant in variants if variant.name in set(robust_names)]
    if not selected_variants:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for dimension in ("breakout_side", "breakout_timing_bucket"):
        for value in sorted(feature_frame[dimension].dropna().astype(str).unique().tolist()):
            subgroup_sessions = feature_frame.loc[feature_frame[dimension].astype(str).eq(value), "session_date"].tolist()
            baseline_sub = _trade_subset(baseline_variant.trades, subgroup_sessions)
            baseline_summary = _scope_summary(
                trades=baseline_sub,
                sessions=subgroup_sessions,
                initial_capital=initial_capital,
                session_minutes=session_minutes,
            )
            for variant in selected_variants:
                variant_sub = _trade_subset(variant.trades, subgroup_sessions)
                variant_summary = _scope_summary(
                    trades=variant_sub,
                    sessions=subgroup_sessions,
                    initial_capital=initial_capital,
                    session_minutes=session_minutes,
                )
                rows.append(
                    {
                        "variant_name": variant.name,
                        "dimension": dimension,
                        "bucket": value,
                        "baseline_n_trades": int(baseline_summary["n_trades"]),
                        "variant_n_trades": int(variant_summary["n_trades"]),
                        "trade_coverage_vs_baseline": _safe_div(
                            float(variant_summary["n_trades"]),
                            float(baseline_summary["n_trades"]),
                            default=0.0,
                        ),
                        "baseline_expectancy": float(baseline_summary["expectancy"]),
                        "variant_expectancy": float(variant_summary["expectancy"]),
                        "expectancy_delta": float(variant_summary["expectancy"]) - float(baseline_summary["expectancy"]),
                        "baseline_hit_rate": float(baseline_summary["hit_rate"]),
                        "variant_hit_rate": float(variant_summary["hit_rate"]),
                        "hit_rate_delta": float(variant_summary["hit_rate"]) - float(baseline_summary["hit_rate"]),
                        "baseline_stop_hit_rate": float(baseline_summary["stop_hit_rate"]),
                        "variant_stop_hit_rate": float(variant_summary["stop_hit_rate"]),
                        "stop_hit_rate_delta": float(baseline_summary["stop_hit_rate"]) - float(variant_summary["stop_hit_rate"]),
                    }
                )
    return pd.DataFrame(rows)


def _export_variant_artifacts(root: Path, variant: VariantRun) -> None:
    variant_dir = root / "variants" / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    variant.controls.to_csv(variant_dir / "controls.csv", index=False)
    variant.trades.to_csv(variant_dir / "trades.csv", index=False)
    variant.daily_results.to_csv(variant_dir / "daily_results.csv", index=False)
    variant.summary_by_scope.to_csv(variant_dir / "metrics_by_scope.csv", index=False)


def _top_variant_by_filter(results_df: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    view = results_df.loc[mask].copy()
    if view.empty:
        return {}
    ordered = view.sort_values(["validation_score", "oos_sharpe_delta_vs_baseline"], ascending=[False, False]).reset_index(drop=True)
    return ordered.iloc[0].to_dict()


def _positive_variant(row: dict[str, Any]) -> bool:
    return str(row.get("verdict")) in {"robust_positive", "protective_filter", "mixed_positive"}


def _edge_character(best_row: dict[str, Any]) -> str:
    if not best_row:
        return "none"
    coverage = float(best_row.get("oos_trade_coverage_vs_baseline", 0.0))
    pnl_retention = float(best_row.get("oos_net_pnl_retention_vs_baseline", 0.0))
    sharpe_delta = float(best_row.get("oos_sharpe_delta_vs_baseline", 0.0))
    pf_delta = float(best_row.get("oos_profit_factor_delta_vs_baseline", 0.0))
    dd_improvement = float(best_row.get("oos_max_drawdown_improvement_vs_baseline", 0.0))
    if coverage >= 0.75 and pnl_retention >= 0.95 and (sharpe_delta > 0.05 or pf_delta > 0.05):
        return "alpha"
    if dd_improvement >= 0.10 and coverage <= 0.90:
        return "defensive_filter"
    if coverage < 0.75:
        return "regime_control"
    return "mixed"


def _primary_mechanism(best_row: dict[str, Any]) -> str:
    if not best_row:
        return "none"
    coverage = float(best_row.get("oos_trade_coverage_vs_baseline", 0.0))
    pnl_retention = float(best_row.get("oos_net_pnl_retention_vs_baseline", 0.0))
    stop_delta = float(best_row.get("oos_stop_hit_rate_delta_vs_baseline", 0.0))
    dd_improvement = float(best_row.get("oos_max_drawdown_improvement_vs_baseline", 0.0))
    pf_delta = float(best_row.get("oos_profit_factor_delta_vs_baseline", 0.0))
    sharpe_delta = float(best_row.get("oos_sharpe_delta_vs_baseline", 0.0))
    if stop_delta > 0.05:
        return "fewer_false_breakouts"
    if coverage < 0.85 and pnl_retention >= 0.90:
        return "better_day_selection"
    if dd_improvement > 0.10 and pf_delta <= 0.0:
        return "drawdown_reduction"
    if pf_delta > 0.05 or sharpe_delta > 0.10:
        return "distribution_improvement"
    return "none"


def _synthesise_verdict(
    results_df: pd.DataFrame,
    baseline_variant: VariantRun,
    spec: MnqOrbVixVvixValidationSpec,
    coverage_summary: dict[str, Any],
) -> dict[str, Any]:
    vix_mask = results_df["feature_name"].str.startswith("vix_", na=False) & results_df["feature_name"].ne("vix_x_vvix_regime")
    vvix_only_mask = results_df["feature_name"].str.startswith("vvix_", na=False) & results_df["feature_name"].ne("vvix_over_vix_t1")
    ratio_mask = results_df["feature_name"].eq("vvix_over_vix_t1")
    cross_mask = results_df["feature_name"].eq("vix_x_vvix_regime")

    active = results_df.loc[results_df["variant_name"].ne("baseline_fixed_nominal_atr")].copy()
    best_vix = _top_variant_by_filter(results_df, vix_mask & results_df["variant_name"].ne("baseline_fixed_nominal_atr"))
    best_vvix = _top_variant_by_filter(results_df, vvix_only_mask & results_df["variant_name"].ne("baseline_fixed_nominal_atr"))
    best_ratio = _top_variant_by_filter(results_df, ratio_mask & results_df["variant_name"].ne("baseline_fixed_nominal_atr"))
    best_cross = _top_variant_by_filter(results_df, cross_mask & results_df["variant_name"].ne("baseline_fixed_nominal_atr"))
    robust = active.loc[active["verdict"] == "robust_positive"].copy()
    mixed = active.loc[active["verdict"] == "mixed_positive"].copy()
    protective = active.loc[active["verdict"] == "protective_filter"].copy()
    if not robust.empty:
        best_overall = robust.sort_values(["validation_score", "oos_sharpe_delta_vs_baseline"], ascending=[False, False]).iloc[0].to_dict()
    elif not mixed.empty:
        best_overall = mixed.sort_values(["validation_score", "oos_sharpe_delta_vs_baseline"], ascending=[False, False]).iloc[0].to_dict()
    elif not protective.empty:
        best_overall = protective.sort_values(["validation_score", "oos_sharpe_delta_vs_baseline"], ascending=[False, False]).iloc[0].to_dict()
    else:
        best_overall = _top_variant_by_filter(results_df, results_df["variant_name"].ne("baseline_fixed_nominal_atr"))

    vvix_complement_candidates = [candidate for candidate in (best_vvix, best_ratio, best_cross) if candidate]
    best_vvix_complement = (
        max(vvix_complement_candidates, key=lambda row: (float(row.get("validation_score", 0.0)), float(row.get("oos_sharpe_delta_vs_baseline", 0.0))))
        if vvix_complement_candidates
        else {}
    )

    vix_edge = _positive_variant(best_vix)
    vvix_edge = _positive_variant(best_vvix_complement) and (
        not vix_edge
        or float(best_vvix_complement.get("validation_score", 0.0)) >= float(best_vix.get("validation_score", 0.0)) - 0.05
        or float(best_vvix_complement.get("oos_max_drawdown_improvement_vs_baseline", 0.0))
        > float(best_vix.get("oos_max_drawdown_improvement_vs_baseline", 0.0)) + 0.05
    )

    phase2_candidates = robust.loc[
        (pd.to_numeric(robust["oos_trade_coverage_vs_baseline"], errors="coerce") >= 0.55)
        & (pd.to_numeric(robust["oos_net_pnl_retention_vs_baseline"], errors="coerce") >= 0.85)
    ].copy()
    phase2_recommended = bool(not phase2_candidates.empty)

    return {
        "run_type": "mnq_orb_vix_vvix_validation",
        "baseline_variant_name": baseline_variant.name,
        "vix_standalone_edge": bool(vix_edge),
        "vvix_adds_complementary_value": bool(vvix_edge),
        "best_variant_name": best_overall.get("variant_name"),
        "best_variant_verdict": best_overall.get("verdict"),
        "best_variant_block": best_overall.get("block"),
        "best_variant_feature_name": best_overall.get("feature_name"),
        "best_variant_oos_trade_coverage_vs_baseline": best_overall.get("oos_trade_coverage_vs_baseline"),
        "best_variant_oos_net_pnl_retention_vs_baseline": best_overall.get("oos_net_pnl_retention_vs_baseline"),
        "best_variant_oos_sharpe_delta_vs_baseline": best_overall.get("oos_sharpe_delta_vs_baseline"),
        "best_variant_oos_profit_factor_delta_vs_baseline": best_overall.get("oos_profit_factor_delta_vs_baseline"),
        "best_variant_oos_expectancy_delta_vs_baseline": best_overall.get("oos_expectancy_delta_vs_baseline"),
        "best_variant_oos_hit_rate_delta_vs_baseline": best_overall.get("oos_hit_rate_delta_vs_baseline"),
        "best_variant_oos_stop_hit_rate_delta_vs_baseline": best_overall.get("oos_stop_hit_rate_delta_vs_baseline"),
        "best_variant_oos_max_drawdown_improvement_vs_baseline": best_overall.get("oos_max_drawdown_improvement_vs_baseline"),
        "best_vix_variant_name": best_vix.get("variant_name"),
        "best_vvix_variant_name": best_vvix.get("variant_name"),
        "best_ratio_variant_name": best_ratio.get("variant_name"),
        "best_cross_variant_name": best_cross.get("variant_name"),
        "best_vix_variant_verdict": best_vix.get("verdict"),
        "best_vvix_variant_verdict": best_vvix.get("verdict"),
        "best_ratio_variant_verdict": best_ratio.get("verdict"),
        "best_cross_variant_verdict": best_cross.get("verdict"),
        "edge_character": _edge_character(best_overall),
        "primary_mechanism": _primary_mechanism(best_overall),
        "phase2_vix_vvix_plus_3state_recommended": phase2_recommended,
        "coverage_summary": coverage_summary,
        "assumptions": [
            f"baseline direction={spec.baseline.direction}",
            f"baseline OR window={spec.baseline.or_minutes}m",
            f"aggregation_rule={spec.aggregation_rule}",
            f"fixed_contracts={spec.fixed_contracts}",
            "daily inputs are strictly t-1",
        ],
    }


def _write_report(
    output_path: Path,
    spec: MnqOrbVixVvixValidationSpec,
    analysis: SymbolAnalysis,
    feature_frame: pd.DataFrame,
    results_df: pd.DataFrame,
    feature_importance: pd.DataFrame,
    interaction_summary: pd.DataFrame,
    verdict: dict[str, Any],
    coverage_summary: dict[str, Any],
) -> None:
    baseline = results_df.loc[results_df["variant_name"] == "baseline_fixed_nominal_atr"].iloc[0]
    top = results_df.loc[results_df["variant_name"] == verdict.get("best_variant_name")].copy()
    best_vix = results_df.loc[results_df["variant_name"] == verdict.get("best_vix_variant_name")].copy()
    best_vvix = results_df.loc[results_df["variant_name"] == verdict.get("best_vvix_variant_name")].copy()
    best_cross = results_df.loc[results_df["variant_name"] == verdict.get("best_cross_variant_name")].copy()

    if top.empty:
        top_line = "- Aucun variant VIX/VVIX ne ressort comme candidat convaincant."
        driver_line = "- Pas de mecanisme OOS defensible detecte."
    else:
        row = top.iloc[0]
        top_line = (
            f"- Meilleur candidat: `{row['variant_name']}` | verdict `{row['verdict']}` | "
            f"OOS Sharpe delta `{float(row['oos_sharpe_delta_vs_baseline']):+.3f}` | "
            f"OOS PF delta `{float(row['oos_profit_factor_delta_vs_baseline']):+.3f}` | "
            f"OOS expectancy delta `{float(row['oos_expectancy_delta_vs_baseline']):+.2f}` | "
            f"retention pnl `{100.0 * float(row['oos_net_pnl_retention_vs_baseline']):.1f}%` | "
            f"trade coverage `{100.0 * float(row['oos_trade_coverage_vs_baseline']):.1f}%` | "
            f"maxDD improvement `{100.0 * float(row['oos_max_drawdown_improvement_vs_baseline']):.1f}%`."
        )
        driver_line = (
            f"- Driver principal: `{verdict.get('primary_mechanism')}` | hit delta `{100.0 * float(row['oos_hit_rate_delta_vs_baseline']):+.1f} pts`, "
            f"stop-hit delta `{100.0 * float(row['oos_stop_hit_rate_delta_vs_baseline']):+.1f} pts`, "
            f"avg win delta `{float(row['oos_avg_win_delta_vs_baseline']):+.2f}`, "
            f"avg loss delta `{float(row['oos_avg_loss_delta_vs_baseline']):+.2f}`."
        )

    best_family_line = "- Aucun family VIX/VVIX ne produit d'avantage OOS robuste."
    if not feature_importance.empty:
        best_family = feature_importance.iloc[0]
        best_family_line = (
            f"- Family / feature la plus prometteuse: `{best_family['family']}` / `{best_family['feature_name']}` "
            f"via `{best_family['best_variant_name']}` (`{best_family['best_variant_verdict']}`)."
        )

    vix_line = "- Le bloc VIX seul ne montre pas d'edge clair."
    if not best_vix.empty:
        row = best_vix.iloc[0]
        vix_line = (
            f"- Meilleur VIX standalone: `{row['variant_name']}` (`{row['verdict']}`), "
            f"validation score `{float(row['validation_score']):.3f}`, "
            f"OOS Sharpe delta `{float(row['oos_sharpe_delta_vs_baseline']):+.3f}`."
        )

    vvix_line = "- Le bloc VVIX / ratio n'ajoute pas d'information claire."
    if not best_vvix.empty:
        row = best_vvix.iloc[0]
        vvix_line = (
            f"- Meilleur VVIX standalone: `{row['variant_name']}` (`{row['verdict']}`), "
            f"validation score `{float(row['validation_score']):.3f}`, "
            f"OOS Sharpe delta `{float(row['oos_sharpe_delta_vs_baseline']):+.3f}`."
        )

    cross_line = "- Les regimes croises VIX x VVIX ne dominent pas le run."
    if not best_cross.empty:
        row = best_cross.iloc[0]
        cross_line = (
            f"- Meilleur regime croise: `{row['variant_name']}` (`{row['verdict']}`), "
            f"validation score `{float(row['validation_score']):.3f}`, "
            f"maxDD improvement `{100.0 * float(row['oos_max_drawdown_improvement_vs_baseline']):.1f}%`."
        )

    direction_interaction = interaction_summary.loc[interaction_summary["dimension"] == "breakout_side"].copy()
    direction_line = "- Interaction haussier/baissier non disponible ou non significative."
    if not direction_interaction.empty:
        best_dir = direction_interaction.sort_values(["expectancy_delta", "hit_rate_delta"], ascending=[False, False]).iloc[0]
        direction_line = (
            f"- Sous-groupe directionnel le plus sensible: `{best_dir['bucket']}` pour `{best_dir['variant_name']}` "
            f"(expectancy delta `{float(best_dir['expectancy_delta']):+.2f}`, hit delta `{100.0 * float(best_dir['hit_rate_delta']):+.1f} pts`)."
        )

    timing_interaction = interaction_summary.loc[interaction_summary["dimension"] == "breakout_timing_bucket"].copy()
    timing_line = "- Interaction timing early/mid/late non disponible."
    if not timing_interaction.empty:
        best_timing = timing_interaction.sort_values(["expectancy_delta", "hit_rate_delta"], ascending=[False, False]).iloc[0]
        timing_line = (
            f"- Sous-groupe timing le plus sensible: `{best_timing['bucket']}` pour `{best_timing['variant_name']}` "
            f"(expectancy delta `{float(best_timing['expectancy_delta']):+.2f}`, hit delta `{100.0 * float(best_timing['hit_rate_delta']):+.1f} pts`)."
        )

    lines = [
        "# MNQ ORB VIX/VVIX Validation Report",
        "",
        "## Baseline",
        "",
        f"- Baseline ORB conservee: OR{int(spec.baseline.or_minutes)} / direction `{spec.baseline.direction}` / RR `{float(spec.baseline.target_multiple):.2f}` / VWAP confirmation `{bool(spec.baseline.vwap_confirmation)}`.",
        f"- Filtre ATR conserve tel quel via l'ensemble `{spec.aggregation_rule}` calibre uniquement sur IS.",
        f"- Sizing retire pour cette campagne: backtest rerun en `fixed_contracts={int(spec.fixed_contracts)}` pour isoler le bloc VIX/VVIX.",
        f"- Dataset: `{analysis.dataset_path.name}` | sessions IS/OOS d'origine: `{len(analysis.is_sessions)}` / `{len(analysis.oos_sessions)}`.",
        "",
        "## Data Coverage",
        "",
        f"- Sessions baseline ATR selectionnees: `{int(coverage_summary.get('selected_sessions_before_coverage', 0))}`.",
        f"- Sessions avec contexte VIX/VVIX exploitable: `{int(coverage_summary.get('selected_sessions_with_implied_vol_context', 0))}`.",
        f"- Sessions exclues faute de contexte daily t-1: `{int(coverage_summary.get('excluded_selected_sessions_no_implied_vol_data', 0))}`.",
        f"- Fin de couverture source VIX: `{coverage_summary.get('vix_source_max_date')}` | fin de couverture source VVIX: `{coverage_summary.get('vvix_source_max_date')}`.",
        "",
        "## Baseline Readout",
        "",
        f"- Baseline OOS sur fenetre couverte: net pnl `{float(baseline['oos_net_pnl']):.2f}` | Sharpe `{float(baseline['oos_sharpe']):.3f}` | Sortino `{float(baseline['oos_sortino']):.3f}` | PF `{float(baseline['oos_profit_factor']):.3f}` | expectancy `{float(baseline['oos_expectancy']):.2f}` | maxDD `{float(baseline['oos_max_drawdown']):.2f}`.",
        f"- Baseline OOS trades/days: `{int(baseline['oos_n_trades'])}` trades | `{int(baseline['oos_n_days_traded'])}` jours trades | exposition temps `{100.0 * float(baseline['oos_exposure_time_pct']):.2f}%`.",
        f"- Univers effectivement teste: `{len(feature_frame)}` trades / sessions baseline avec contexte daily exploitable.",
        "",
        "## Main Findings",
        "",
        top_line,
        best_family_line,
        vix_line,
        vvix_line,
        cross_line,
        driver_line,
        direction_line,
        timing_line,
        f"- Variants robustes OOS: `{int((results_df['verdict'] == 'robust_positive').sum())}` | protecteurs mais couteux: `{int((results_df['verdict'] == 'protective_filter').sum())}` | faux positifs IS-only: `{int((results_df['verdict'] == 'is_only').sum())}`.",
        "",
        "## Reponses Directes",
        "",
        f"- Le VIX apporte-t-il un edge standalone comme filtre de regime ? {'Oui, avec confirmation OOS suffisante.' if verdict.get('vix_standalone_edge') else 'Non, pas de facon assez robuste en OOS.'}",
        f"- Le VVIX ajoute-t-il une information complementaire utile au VIX ? {'Oui, au moins sous forme de regime croise / ratio ou de filtre complementaire.' if verdict.get('vvix_adds_complementary_value') else 'Non, pas de gain complementaire assez clair face au VIX seul.'}",
        f"- Le gain eventuel vient-il d'une meilleure selection des jours, d'une baisse du drawdown, d'une reduction des faux breakouts ou d'une vraie hausse du PF / Sharpe ? `{verdict.get('primary_mechanism')}`.",
        f"- Les resultats justifient-ils une phase 2 avec meilleur filtre VIX/VVIX + sizing 3-state ? {'Oui.' if verdict.get('phase2_vix_vvix_plus_3state_recommended') else 'Non, mieux vaut ne pas melanger avec le 3-state tant que le gain VIX/VVIX seul reste ambigu.'}",
        f"- L'interet est-il plutot alpha, filtrage defensif ou simple controle de regime ? `{verdict.get('edge_character')}`.",
        "",
        "## Notes Methodologiques",
        "",
        "- Les regles d'entree/sortie ORB, les couts, le slippage, la commission et le filtre ATR baseline restent inchanges.",
        "- Les features VIX/VVIX sont construites strictement en daily t-1; aucune cloture du jour courant n'est utilisee pour decider du trade du meme jour.",
        "- Tous les seuils et buckets sont calibres sur IS seulement puis figes sur OOS.",
        "- Cette campagne n'inclut aucun sizing dynamique 3-state.",
        "- OR15 vs OR30 n'est pas re-optimise ici: la baseline est gardee figee sur sa reference tradable actuelle.",
        "",
        "## Exports",
        "",
        "- `selected_trade_vix_vvix_features.csv`",
        "- `regime_summary.csv`",
        "- `screening_summary.csv`",
        "- `validation_summary.csv`",
        "- `full_variant_results.csv`",
        "- `feature_importance_like_summary.csv`",
        "- `interaction_summary.csv`",
        "- `final_verdict.json`",
        "- `variants/<variant>/...`",
    ]
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_mnq_orb_vix_vvix_validation_campaign(spec: MnqOrbVixVvixValidationSpec) -> dict[str, Any]:
    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(spec.output_root) if spec.output_root is not None else EXPORTS_DIR / f"mnq_orb_vix_vvix_validation_{timestamp}"
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
    _, baseline_trades_full = _build_fixed_nominal_baseline(analysis, selected_sessions, spec)

    daily_features = load_vix_vvix_daily_features(
        vix_path=spec.vix_daily_path,
        vvix_path=spec.vvix_daily_path,
    )
    feature_frame = build_vix_vvix_feature_frame(analysis, baseline_trades_full, daily_features)
    if feature_frame.empty:
        raise RuntimeError("No overlap was found between the baseline selected sessions and VIX/VVIX daily features.")

    campaign_sessions = sorted(pd.to_datetime(feature_frame["session_date"]).dt.date.unique().tolist())
    campaign_session_set = set(campaign_sessions)
    campaign_is_sessions = [session for session in pd.to_datetime(pd.Index(analysis.is_sessions)).date if session in campaign_session_set]
    campaign_oos_sessions = [session for session in pd.to_datetime(pd.Index(analysis.oos_sessions)).date if session in campaign_session_set]
    baseline_trades = _trade_subset(baseline_trades_full, campaign_sessions)

    session_minutes = float((pd.Timestamp(spec.baseline.time_exit) - pd.Timestamp(spec.baseline.opening_time)).total_seconds() / 60.0)
    baseline_variant = _baseline_variant(
        baseline_trades=baseline_trades,
        feature_frame=feature_frame,
        all_sessions=campaign_sessions,
        is_sessions=campaign_is_sessions,
        oos_sessions=campaign_oos_sessions,
        initial_capital=float(spec.initial_capital_usd),
        session_minutes=session_minutes,
    )

    regime_summary, assignments, calibrations = build_regime_summary(
        feature_frame=feature_frame,
        baseline_trades=baseline_trades,
        all_sessions=campaign_sessions,
        is_sessions=campaign_is_sessions,
        oos_sessions=campaign_oos_sessions,
        initial_capital=float(spec.initial_capital_usd),
        session_minutes=session_minutes,
        feature_specs=_implied_vol_feature_specs(),
        bucket_count=spec.bucket_count,
        min_bucket_obs_is=spec.min_bucket_obs_is,
    )

    feature_frame_export = feature_frame.copy()
    for feature_name, labels in assignments.items():
        feature_frame_export[f"{feature_name}_bucket"] = pd.Series(labels, index=feature_frame_export.index, dtype="string")

    feature_path = output_root / "selected_trade_vix_vvix_features.csv"
    feature_frame_export.to_csv(feature_path, index=False)

    regime_summary_path = output_root / "regime_summary.csv"
    regime_summary.to_csv(regime_summary_path, index=False)

    variants: list[VariantRun] = [baseline_variant]
    for feature_spec in _implied_vol_feature_specs():
        if feature_spec.name not in calibrations:
            continue
        for variant_spec in _standard_variant_specs(feature_spec, calibrations[feature_spec.name]):
            variants.append(
                _build_variant_run(
                    variant_spec=variant_spec,
                    feature_frame=feature_frame,
                    bucket_labels=assignments[feature_spec.name],
                    baseline_trades=baseline_trades,
                    all_sessions=campaign_sessions,
                    is_sessions=campaign_is_sessions,
                    oos_sessions=campaign_oos_sessions,
                    initial_capital=float(spec.initial_capital_usd),
                    session_minutes=session_minutes,
                )
            )

    if "vix_x_vvix_regime" in calibrations:
        for variant_spec in _cross_regime_variant_specs(calibrations["vix_x_vvix_regime"]):
            variants.append(
                _build_variant_run(
                    variant_spec=variant_spec,
                    feature_frame=feature_frame,
                    bucket_labels=assignments["vix_x_vvix_regime"],
                    baseline_trades=baseline_trades,
                    all_sessions=campaign_sessions,
                    is_sessions=campaign_is_sessions,
                    oos_sessions=campaign_oos_sessions,
                    initial_capital=float(spec.initial_capital_usd),
                    session_minutes=session_minutes,
                )
            )

    results_df = pd.DataFrame([_variant_row(variant, baseline_variant, spec) for variant in variants])
    results_df = results_df[[column for column in SUMMARY_COLUMNS if column in results_df.columns]]
    full_results_path = output_root / "full_variant_results.csv"
    results_df.to_csv(full_results_path, index=False)

    screening_summary = results_df.sort_values(["screening_score", "is_sharpe_delta_vs_baseline"], ascending=[False, False]).reset_index(drop=True)
    screening_path = output_root / "screening_summary.csv"
    screening_summary.to_csv(screening_path, index=False)

    validation_summary = results_df.sort_values(["validation_score", "oos_sharpe_delta_vs_baseline"], ascending=[False, False]).reset_index(drop=True)
    validation_path = output_root / "validation_summary.csv"
    validation_summary.to_csv(validation_path, index=False)

    feature_importance = _feature_importance_like_summary(results_df)
    feature_importance_path = output_root / "feature_importance_like_summary.csv"
    feature_importance.to_csv(feature_importance_path, index=False)

    interaction_summary = _interaction_summary(
        feature_frame=feature_frame,
        baseline_variant=baseline_variant,
        variants=variants,
        results_df=validation_summary,
        initial_capital=float(spec.initial_capital_usd),
        session_minutes=session_minutes,
    )
    interaction_path = output_root / "interaction_summary.csv"
    interaction_summary.to_csv(interaction_path, index=False)

    for variant in variants:
        _export_variant_artifacts(output_root, variant)

    vix_source = load_vix_daily(spec.vix_daily_path)
    vvix_source = load_vvix_daily(spec.vvix_daily_path)
    coverage_summary = {
        "selected_sessions_before_coverage": int(len(selected_sessions)),
        "selected_sessions_with_implied_vol_context": int(len(campaign_sessions)),
        "excluded_selected_sessions_no_implied_vol_data": int(len(selected_sessions) - len(campaign_sessions)),
        "campaign_is_sessions": int(len(campaign_is_sessions)),
        "campaign_oos_sessions": int(len(campaign_oos_sessions)),
        "vix_source_max_date": str(pd.to_datetime(vix_source["source_date"]).max().date()) if not vix_source.empty else None,
        "vvix_source_max_date": str(pd.to_datetime(vvix_source["source_date"]).max().date()) if not vvix_source.empty else None,
    }

    verdict = _synthesise_verdict(results_df, baseline_variant=baseline_variant, spec=spec, coverage_summary=coverage_summary)
    verdict_path = output_root / "final_verdict.json"
    _json_dump(verdict_path, verdict)

    report_path = output_root / "final_report.md"
    _write_report(
        output_path=report_path,
        spec=spec,
        analysis=analysis,
        feature_frame=feature_frame_export,
        results_df=validation_summary,
        feature_importance=feature_importance,
        interaction_summary=interaction_summary,
        verdict=verdict,
        coverage_summary=coverage_summary,
    )

    metadata_path = output_root / "run_metadata.json"
    _json_dump(
        metadata_path,
        {
            "run_timestamp": datetime.now().isoformat(),
            "dataset_path": dataset_path,
            "selected_symbol": spec.symbol,
            "selected_aggregation_rule": spec.aggregation_rule,
            "analysis_baseline_transfer": analysis.baseline_transfer,
            "analysis_best_ensemble": analysis.best_ensemble,
            "coverage_summary": coverage_summary,
            "spec": asdict(spec),
        },
    )

    return {
        "output_root": output_root,
        "feature_frame": feature_path,
        "regime_summary": regime_summary_path,
        "screening_summary": screening_path,
        "validation_summary": validation_path,
        "full_variant_results": full_results_path,
        "feature_importance": feature_importance_path,
        "interaction_summary": interaction_path,
        "final_report": report_path,
        "final_verdict": verdict_path,
        "run_metadata": metadata_path,
    }


def build_default_spec(output_root: Path | None = None) -> MnqOrbVixVvixValidationSpec:
    return MnqOrbVixVvixValidationSpec(output_root=output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MNQ ORB VIX/VVIX validation campaign.")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--is-fraction", type=float, default=0.70)
    parser.add_argument("--aggregation-rule", type=str, default="majority_50")
    parser.add_argument("--vix-daily-path", type=Path, default=DEFAULT_VIX_DAILY_PATH)
    parser.add_argument("--vvix-daily-path", type=Path, default=DEFAULT_VVIX_DAILY_PATH)
    parser.add_argument("--fixed-contracts", type=int, default=1)
    args = parser.parse_args()

    spec = MnqOrbVixVvixValidationSpec(
        dataset_path=args.dataset_path,
        output_root=args.output_root,
        is_fraction=float(args.is_fraction),
        aggregation_rule=str(args.aggregation_rule),
        vix_daily_path=args.vix_daily_path,
        vvix_daily_path=args.vvix_daily_path,
        fixed_contracts=int(args.fixed_contracts),
    )
    artifacts = run_mnq_orb_vix_vvix_validation_campaign(spec)
    print(f"output_root: {artifacts['output_root']}")
    print(f"validation_summary: {artifacts['validation_summary']}")
    print(f"final_report: {artifacts['final_report']}")


if __name__ == "__main__":
    main()
