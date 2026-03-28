"""Regime-filter and dynamic-sizing campaign for the validated MNQ ensemble ORB."""

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

from src.analytics.mnq_orb_prop_survivability_campaign import (
    _build_summary_by_scope,
    _rebuild_daily_results_from_trades,
    _scope_value,
)
from src.analytics.orb_multi_asset_campaign import (
    BaselineSpec,
    SearchGrid,
    SymbolAnalysis,
    analyze_symbol,
    analyze_symbol_cache_pass_matrix,
    compute_campaign_metrics,
    resolve_aggregation_threshold,
    resolve_processed_dataset,
)
from src.config.orb_campaign import PropConstraintConfig, build_prop_constraints
from src.config.paths import EXPORTS_DIR, ensure_directories
from src.features.volatility import add_atr, add_rolling_std


WEEKDAY_LABELS = {
    0: "monday",
    1: "tuesday",
    2: "wednesday",
    3: "thursday",
    4: "friday",
    5: "saturday",
    6: "sunday",
}

SUMMARY_COLUMNS = [
    "variant_name",
    "family",
    "feature_name",
    "bucketing",
    "description",
    "calibration_scope",
    "parameters_json",
    "note",
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
    "overall_worst_day",
    "overall_longest_losing_streak_daily",
    "overall_median_recovery_days",
    "overall_max_recovery_days",
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
    "overall_trade_coverage_vs_nominal",
    "overall_day_coverage_vs_nominal",
    "is_trade_coverage_vs_nominal",
    "is_day_coverage_vs_nominal",
    "oos_trade_coverage_vs_nominal",
    "oos_day_coverage_vs_nominal",
    "oos_net_pnl_retention_vs_nominal",
    "oos_sharpe_delta_vs_nominal",
    "oos_max_drawdown_improvement_vs_nominal",
]


@dataclass(frozen=True)
class BucketCalibration:
    feature_name: str
    bucket_kind: str
    labels: tuple[str, ...]
    bins: tuple[float, ...] = ()


@dataclass(frozen=True)
class RegimeFeatureSpec:
    name: str
    family: str
    description: str
    value_column: str
    bucket_kind: str = "quantile"
    bucket_count: int = 3


@dataclass
class RegimeVariantRun:
    name: str
    family: str
    feature_name: str
    bucketing: str
    description: str
    calibration_scope: str
    parameters: dict[str, Any]
    trades: pd.DataFrame
    daily_results: pd.DataFrame
    controls: pd.DataFrame
    summary_by_scope: pd.DataFrame
    note: str = ""


@dataclass(frozen=True)
class MnqRegimeFilterSizingSpec:
    symbol: str = "MNQ"
    dataset_path: Path | None = None
    is_fraction: float = 0.70
    aggregation_rule: str = "majority_50"
    min_bucket_obs_is: int = 50
    min_bucket_obs_oos: int = 20
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


def _safe_series_div(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> pd.Series:
    num = pd.to_numeric(pd.Series(numerator), errors="coerce")
    den = pd.to_numeric(pd.Series(denominator), errors="coerce")
    return num.where(den.ne(0)).divide(den.where(den.ne(0)))


def _bucket_labels(bucket_count: int) -> tuple[str, ...]:
    if bucket_count == 3:
        return ("low", "mid", "high")
    if bucket_count == 4:
        return ("very_low", "low_mid", "high_mid", "high")
    return tuple(f"bucket_{idx + 1}" for idx in range(bucket_count))


def _selected_ensemble_sessions(analysis: SymbolAnalysis, aggregation_rule: str) -> set:
    point_pass = analyze_symbol_cache_pass_matrix(analysis)
    pass_cols = [column for column in point_pass.columns if column.startswith("pass__")]
    if not pass_cols:
        return set()
    scored = point_pass.copy()
    scored["consensus_score"] = scored[pass_cols].sum(axis=1) / float(len(pass_cols))
    threshold = resolve_aggregation_threshold(aggregation_rule)
    return set(pd.to_datetime(scored.loc[scored["consensus_score"] >= threshold, "session_date"]).dt.date)


def calibrate_quantile_buckets(
    feature_name: str,
    is_values: pd.Series,
    bucket_count: int,
) -> BucketCalibration:
    clean = pd.to_numeric(is_values, errors="coerce").dropna()
    if len(clean) < bucket_count:
        raise ValueError(f"Not enough IS observations to calibrate {bucket_count} buckets for {feature_name}.")
    _, raw_bins = pd.qcut(clean, q=bucket_count, labels=False, retbins=True, duplicates="drop")
    raw_bins = tuple(float(value) for value in raw_bins)
    actual_bucket_count = len(raw_bins) - 1
    if actual_bucket_count < 2:
        raise ValueError(f"Quantile calibration collapsed for feature {feature_name}.")
    return BucketCalibration(
        feature_name=feature_name,
        bucket_kind="quantile",
        labels=_bucket_labels(actual_bucket_count),
        bins=raw_bins,
    )


def calibrate_categorical_buckets(
    feature_name: str,
    is_values: pd.Series,
) -> BucketCalibration:
    labels = tuple(str(value) for value in pd.Series(is_values).dropna().astype(str).drop_duplicates().tolist())
    if not labels:
        raise ValueError(f"No categorical labels available for feature {feature_name}.")
    return BucketCalibration(feature_name=feature_name, bucket_kind="categorical", labels=labels)


def apply_bucket_calibration(values: pd.Series, calibration: BucketCalibration) -> pd.Series:
    if calibration.bucket_kind == "categorical":
        clean = pd.Series(values).astype("object")
        return clean.where(clean.astype(str).isin(set(calibration.labels))).astype("string")
    if not calibration.bins:
        return pd.Series(pd.NA, index=values.index, dtype="string")

    extended_bins = list(calibration.bins)
    extended_bins[0] = -np.inf
    extended_bins[-1] = np.inf
    bucketed = pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=extended_bins,
        labels=list(calibration.labels),
        include_lowest=True,
    )
    return pd.Series(bucketed, index=values.index, dtype="string")


def build_session_reference_features(
    frame: pd.DataFrame,
    opening_time: str,
    time_exit: str,
) -> pd.DataFrame:
    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working = working.dropna(subset=["timestamp"]).copy()
    working["session_date"] = pd.to_datetime(working["session_date"]).dt.date
    if "continuous_session_date" in working.columns:
        working["continuous_session_key"] = pd.to_datetime(working["continuous_session_date"]).dt.date
    else:
        working["continuous_session_key"] = working["session_date"]

    open_ts = pd.Timestamp(opening_time)
    close_ts = pd.Timestamp(time_exit)
    open_minutes = int(open_ts.hour * 60 + open_ts.minute)
    close_minutes = int(close_ts.hour * 60 + close_ts.minute)
    working["minute_of_day"] = working["timestamp"].dt.hour * 60 + working["timestamp"].dt.minute
    working["bar_date"] = working["timestamp"].dt.date

    rth = working.loc[working["minute_of_day"].between(open_minutes, close_minutes, inclusive="both")].copy()
    if rth.empty:
        return pd.DataFrame(
            columns=["session_date", "rth_open", "rth_close", "prev_rth_close", "atr_20_open", "overnight_range_pts"]
        )

    rth = rth.sort_values("timestamp")
    rth_open = (
        rth.groupby("session_date", sort=True)
        .first()[["open", "atr_20"]]
        .rename(columns={"open": "rth_open", "atr_20": "atr_20_open"})
    )
    rth_close = (
        rth.groupby("session_date", sort=True)
        .last()[["close"]]
        .rename(columns={"close": "rth_close"})
    )
    references = rth_open.join(rth_close, how="outer").reset_index()
    references["prev_rth_close"] = pd.to_numeric(references["rth_close"], errors="coerce").shift(1)

    overnight_mask = (working["bar_date"] < working["continuous_session_key"]) | (working["minute_of_day"] < open_minutes)
    overnight = (
        working.loc[overnight_mask]
        .groupby("continuous_session_key", sort=True)
        .agg(overnight_high=("high", "max"), overnight_low=("low", "min"))
        .reset_index()
        .rename(columns={"continuous_session_key": "session_date"})
    )
    overnight["overnight_range_pts"] = pd.to_numeric(overnight["overnight_high"], errors="coerce") - pd.to_numeric(
        overnight["overnight_low"], errors="coerce"
    )
    return references.merge(overnight[["session_date", "overnight_range_pts"]], on="session_date", how="left")


def _feature_specs() -> tuple[RegimeFeatureSpec, ...]:
    return (
        RegimeFeatureSpec(
            name="atr_ratio_10_30",
            family="volatility",
            description="Short-vs-long ATR ratio at the selected signal bar.",
            value_column="atr_ratio_10_30",
        ),
        RegimeFeatureSpec(
            name="overnight_range_pts",
            family="volatility",
            description="Overnight range from prior 18:00 session roll to the RTH open.",
            value_column="overnight_range_pts",
        ),
        RegimeFeatureSpec(
            name="opening_range_width_pts",
            family="volatility",
            description="Opening-range width in points after the 30-minute OR window is complete.",
            value_column="opening_range_width_pts",
        ),
        RegimeFeatureSpec(
            name="realized_vol_ratio_15_60",
            family="volatility",
            description="Realized volatility ratio using rolling close-return stdev 15 vs 60 bars.",
            value_column="realized_vol_ratio_15_60",
        ),
        RegimeFeatureSpec(
            name="gap_abs_atr20",
            family="extension",
            description="Absolute RTH opening gap normalized by ATR20 at the open.",
            value_column="gap_abs_atr20",
        ),
        RegimeFeatureSpec(
            name="signal_vwap_distance_atr20",
            family="extension",
            description="Distance from the selected signal bar close to continuous VWAP, normalized by ATR20.",
            value_column="signal_vwap_distance_atr20",
        ),
        RegimeFeatureSpec(
            name="signal_extension_over_or",
            family="extension",
            description="Extension beyond the OR boundary at the selected signal bar, scaled by OR width.",
            value_column="signal_extension_over_or",
        ),
        RegimeFeatureSpec(
            name="weekday_name",
            family="structural",
            description="Simple weekday context.",
            value_column="weekday_name",
            bucket_kind="categorical",
            bucket_count=5,
        ),
    )


def build_regime_dataset(analysis: SymbolAnalysis, selected_sessions: set) -> pd.DataFrame:
    signal_enriched = add_atr(analysis.signal_df.copy(), window=(10, 20))
    signal_enriched = add_rolling_std(signal_enriched, window=15)
    signal_enriched = add_rolling_std(signal_enriched, window=60)
    signal_enriched["session_date"] = pd.to_datetime(signal_enriched["session_date"]).dt.date

    references = build_session_reference_features(
        signal_enriched,
        opening_time=analysis.baseline.opening_time,
        time_exit=analysis.baseline.time_exit,
    )

    selected_index = analysis.candidate_df.copy()
    selected_index["session_date"] = pd.to_datetime(selected_index["session_date"]).dt.date
    selected_index = selected_index.loc[
        selected_index["session_date"].isin(selected_sessions),
        ["session_date", "signal_index"],
    ].copy()

    signal_rows = signal_enriched.loc[selected_index["signal_index"].tolist()].copy()
    signal_rows = signal_rows.reset_index().rename(columns={"index": "signal_index"})
    signal_rows["session_date"] = pd.to_datetime(signal_rows["session_date"]).dt.date

    nominal_trades = analysis.baseline_trades.copy()
    nominal_trades["session_date"] = pd.to_datetime(nominal_trades["session_date"]).dt.date
    nominal_trades = nominal_trades.loc[nominal_trades["session_date"].isin(selected_sessions)].copy()

    regime = selected_index.merge(signal_rows, on=["session_date", "signal_index"], how="left")
    regime = regime.merge(references, on="session_date", how="left")
    regime = regime.merge(
        nominal_trades[
            [
                "session_date",
                "trade_id",
                "entry_time",
                "exit_time",
                "direction",
                "quantity",
                "net_pnl_usd",
                "trade_risk_usd",
                "fees",
                "exit_reason",
            ]
        ],
        on="session_date",
        how="inner",
    )
    regime = regime.sort_values("session_date").reset_index(drop=True)

    is_set = set(pd.to_datetime(pd.Index(analysis.is_sessions)).date)
    regime["phase"] = np.where(regime["session_date"].isin(is_set), "is", "oos")
    regime["weekday_name"] = pd.to_numeric(regime["weekday"], errors="coerce").map(WEEKDAY_LABELS)

    atr_30 = pd.to_numeric(regime["atr_30"], errors="coerce")
    atr_20 = pd.to_numeric(regime["atr_20"], errors="coerce")
    atr_20_open = pd.to_numeric(regime["atr_20_open"], errors="coerce")
    vol_15 = pd.to_numeric(regime["vol_std_15"], errors="coerce")
    vol_60 = pd.to_numeric(regime["vol_std_60"], errors="coerce")
    or_width = pd.to_numeric(regime["or_width"], errors="coerce")
    signal_close = pd.to_numeric(regime["close"], errors="coerce")
    or_high = pd.to_numeric(regime["or_high"], errors="coerce")
    or_low = pd.to_numeric(regime["or_low"], errors="coerce")
    signal_side = pd.to_numeric(regime["signal"], errors="coerce")

    regime["atr_ratio_10_30"] = _safe_series_div(pd.to_numeric(regime["atr_10"], errors="coerce"), atr_30)
    regime["opening_range_width_pts"] = or_width
    regime["realized_vol_ratio_15_60"] = _safe_series_div(vol_15, vol_60)
    regime["gap_abs_atr20"] = _safe_series_div(
        (pd.to_numeric(regime["rth_open"], errors="coerce") - pd.to_numeric(regime["prev_rth_close"], errors="coerce")).abs(),
        atr_20_open,
    )
    regime["signal_vwap_distance_atr20"] = _safe_series_div(
        (signal_close - pd.to_numeric(regime["continuous_session_vwap"], errors="coerce")).abs(),
        atr_20,
    )
    regime["signal_extension_over_or"] = _safe_series_div(
        np.where(signal_side.eq(1), signal_close - or_high, or_low - signal_close),
        or_width,
    )
    regime["nominal_selected"] = True
    return regime


def _scope_sessions(regime_df: pd.DataFrame, scope: str) -> list:
    if scope == "overall":
        return list(regime_df["session_date"])
    return list(regime_df.loc[regime_df["phase"] == scope, "session_date"])


def _scope_trade_subset(trades: pd.DataFrame, sessions: list) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    session_set = set(pd.to_datetime(pd.Index(sessions)).date)
    out = trades.copy()
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.date
    return out.loc[out["session_date"].isin(session_set)].copy()


def _scope_metric_row(trades: pd.DataFrame, sessions: list, initial_capital: float) -> dict[str, Any]:
    metrics = compute_campaign_metrics(trades, sessions=sessions, initial_capital=initial_capital)
    return {
        "n_obs": int(len(sessions)),
        "net_pnl": float(metrics.get("net_pnl", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "expectancy": float(metrics.get("expectancy", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "nb_trades": int(metrics.get("nb_trades", 0)),
        "worst_day": float(metrics.get("worst_day", 0.0)),
        "composite_score": float(metrics.get("composite_score", 0.0)),
    }


def build_conditional_bucket_analysis(
    regime_df: pd.DataFrame,
    nominal_trades: pd.DataFrame,
    initial_capital: float,
    feature_specs: tuple[RegimeFeatureSpec, ...],
    min_bucket_obs_is: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series], dict[str, BucketCalibration]]:
    rows: list[dict[str, Any]] = []
    feature_scores: list[dict[str, Any]] = []
    assignments: dict[str, pd.Series] = {}
    calibrations: dict[str, BucketCalibration] = {}

    phase_session_map = {
        "is": set(regime_df.loc[regime_df["phase"] == "is", "session_date"]),
        "oos": set(regime_df.loc[regime_df["phase"] == "oos", "session_date"]),
    }
    nominal_scope_pnl = {
        scope: float(
            _scope_trade_subset(nominal_trades, _scope_sessions(regime_df, scope))["net_pnl_usd"].sum()
            if not nominal_trades.empty
            else 0.0
        )
        for scope in ("overall", "is", "oos")
    }
    nominal_scope_obs = {scope: len(_scope_sessions(regime_df, scope)) for scope in ("overall", "is", "oos")}

    for feature in feature_specs:
        is_values = regime_df.loc[regime_df["phase"] == "is", feature.value_column]
        try:
            calibration = (
                calibrate_quantile_buckets(feature.name, is_values, feature.bucket_count)
                if feature.bucket_kind == "quantile"
                else calibrate_categorical_buckets(feature.name, is_values)
            )
        except ValueError:
            continue

        bucket_labels = apply_bucket_calibration(regime_df[feature.value_column], calibration)
        assignments[feature.name] = bucket_labels
        calibrations[feature.name] = calibration

        feature_frame = regime_df.copy()
        feature_frame["bucket_label"] = bucket_labels

        feature_rows: list[dict[str, Any]] = []
        for idx, label in enumerate(calibration.labels):
            bucket_sessions = list(feature_frame.loc[feature_frame["bucket_label"] == label, "session_date"])
            bucket_trades = _scope_trade_subset(nominal_trades, bucket_sessions)

            row: dict[str, Any] = {
                "feature_name": feature.name,
                "family": feature.family,
                "description": feature.description,
                "bucket_kind": calibration.bucket_kind,
                "bucket_label": label,
                "bucket_position": idx + 1,
                "lower_bound": float(calibration.bins[idx]) if calibration.bucket_kind == "quantile" and calibration.bins else np.nan,
                "upper_bound": float(calibration.bins[idx + 1]) if calibration.bucket_kind == "quantile" and calibration.bins else np.nan,
            }

            for scope in ("overall", "is", "oos"):
                scope_sessions = bucket_sessions if scope == "overall" else [session for session in bucket_sessions if session in phase_session_map[scope]]
                scope_trades = _scope_trade_subset(bucket_trades, scope_sessions)
                metrics = _scope_metric_row(scope_trades, scope_sessions, initial_capital=initial_capital)
                row.update({f"{scope}_{key}": value for key, value in metrics.items()})
                row[f"{scope}_coverage_vs_nominal"] = _safe_div(len(scope_sessions), max(nominal_scope_obs[scope], 1), default=0.0)
                row[f"{scope}_pnl_contribution"] = _safe_div(metrics["net_pnl"], nominal_scope_pnl[scope], default=0.0)

            feature_rows.append(row)
            rows.append(row)

        is_frame = pd.DataFrame(feature_rows)
        if is_frame.empty:
            continue
        ranked_is = is_frame.sort_values(
            ["is_composite_score", "is_expectancy", "is_profit_factor"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        best_bucket = str(ranked_is.iloc[0]["bucket_label"])
        worst_bucket = str(ranked_is.iloc[-1]["bucket_label"])
        is_counts = pd.to_numeric(is_frame["is_n_obs"], errors="coerce")
        min_obs = int(is_counts.min()) if not is_counts.empty else 0
        max_obs = int(is_counts.max()) if not is_counts.empty else 0
        balance = _safe_div(min_obs, max(max_obs, 1), default=0.0)
        spread = float(ranked_is["is_composite_score"].max() - ranked_is["is_composite_score"].min())
        selection_score = spread * (0.50 + 0.50 * balance)
        worst_is_obs = int(is_frame.loc[is_frame["bucket_label"] == worst_bucket, "is_n_obs"].iloc[0])

        feature_scores.append(
            {
                "feature_name": feature.name,
                "family": feature.family,
                "bucket_kind": calibration.bucket_kind,
                "bucket_count": int(len(calibration.labels)),
                "min_bucket_obs_is": min_obs,
                "balance_is": balance,
                "is_score_spread": spread,
                "feature_selection_score": selection_score,
                "best_bucket_is": best_bucket,
                "worst_bucket_is": worst_bucket,
                "skip_coverage_is": 1.0 - _safe_div(worst_is_obs, max(len(regime_df.loc[regime_df["phase"] == "is"]), 1), default=0.0),
                "valid_for_overlay": bool(min_obs >= min_bucket_obs_is and len(calibration.labels) >= 3),
            }
        )

    conditional_df = pd.DataFrame(rows)
    feature_score_df = pd.DataFrame(feature_scores)
    if not feature_score_df.empty:
        feature_score_df = feature_score_df.sort_values(
            ["valid_for_overlay", "feature_selection_score", "min_bucket_obs_is"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return conditional_df, feature_score_df, assignments, calibrations


def build_state_mapping_from_is_scores(
    feature_rows: pd.DataFrame,
    multipliers_by_rank: tuple[float, ...],
) -> dict[str, float]:
    ranked = feature_rows.sort_values(
        ["is_composite_score", "is_expectancy", "is_profit_factor"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    if len(ranked) != len(multipliers_by_rank):
        raise ValueError("Multiplier count must match the number of ranked buckets.")
    return {str(bucket): float(multipliers_by_rank[idx]) for idx, bucket in enumerate(ranked["bucket_label"].tolist())}


def build_static_regime_controls(
    regime_df: pd.DataFrame,
    feature_name: str,
    bucket_labels: pd.Series,
    bucket_multipliers: dict[str, float],
) -> pd.DataFrame:
    controls = regime_df[["session_date", "phase"]].copy()
    controls["selected_by_ensemble"] = True
    controls["feature_name"] = feature_name
    controls["bucket_label"] = pd.Series(bucket_labels, index=regime_df.index, dtype="string")
    controls["risk_multiplier"] = controls["bucket_label"].map(bucket_multipliers).fillna(0.0).astype(float)
    controls["skip_trade"] = controls["risk_multiplier"].eq(0.0)
    return controls.sort_values("session_date").reset_index(drop=True)


def _nominal_controls(regime_df: pd.DataFrame) -> pd.DataFrame:
    controls = regime_df[["session_date", "phase"]].copy()
    controls["selected_by_ensemble"] = True
    controls["feature_name"] = "nominal"
    controls["bucket_label"] = "all"
    controls["risk_multiplier"] = 1.0
    controls["skip_trade"] = False
    return controls.sort_values("session_date").reset_index(drop=True)


def _scale_nominal_trades_by_multiplier(
    nominal_trades: pd.DataFrame,
    controls: pd.DataFrame,
    account_size_usd: float,
    base_risk_pct: float,
    tick_value_usd: float,
    point_value_usd: float,
    commission_per_side_usd: float,
) -> pd.DataFrame:
    if nominal_trades.empty or controls.empty:
        return pd.DataFrame()

    trades = nominal_trades.copy()
    trades["session_date"] = pd.to_datetime(trades["session_date"]).dt.date
    multiplier_map = dict(zip(pd.to_datetime(controls["session_date"]).dt.date, controls["risk_multiplier"]))
    trades["risk_multiplier"] = trades["session_date"].map(multiplier_map).fillna(0.0).astype(float)
    trades = trades.loc[trades["risk_multiplier"] > 0].copy()
    if trades.empty:
        return trades

    risk_budget_base = float(account_size_usd) * float(base_risk_pct) / 100.0
    per_contract_risk = pd.to_numeric(trades["risk_per_contract_usd"], errors="coerce")
    scaled_budget = risk_budget_base * pd.to_numeric(trades["risk_multiplier"], errors="coerce")
    quantity = np.floor(scaled_budget / per_contract_risk).astype(int)
    trades["quantity"] = quantity
    trades = trades.loc[trades["quantity"] >= 1].copy()
    if trades.empty:
        return trades

    fees_per_contract = 2.0 * float(commission_per_side_usd)
    trades["risk_per_trade_pct"] = float(base_risk_pct) * pd.to_numeric(trades["risk_multiplier"], errors="coerce")
    trades["risk_budget_usd"] = risk_budget_base * pd.to_numeric(trades["risk_multiplier"], errors="coerce")
    trades["actual_risk_usd"] = pd.to_numeric(trades["quantity"], errors="coerce") * per_contract_risk.loc[trades.index]
    trades["trade_risk_usd"] = trades["actual_risk_usd"]
    trades["pnl_usd"] = pd.to_numeric(trades["pnl_ticks"], errors="coerce") * float(tick_value_usd) * pd.to_numeric(
        trades["quantity"], errors="coerce"
    )
    trades["fees"] = fees_per_contract * pd.to_numeric(trades["quantity"], errors="coerce")
    trades["net_pnl_usd"] = pd.to_numeric(trades["pnl_usd"], errors="coerce") - pd.to_numeric(trades["fees"], errors="coerce")
    trades["notional_usd"] = (
        pd.to_numeric(trades["entry_price"], errors="coerce")
        * float(point_value_usd)
        * pd.to_numeric(trades["quantity"], errors="coerce")
    )
    trades["leverage_used"] = _safe_series_div(pd.to_numeric(trades["notional_usd"], errors="coerce"), pd.Series(account_size_usd, index=trades.index))
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    trades["trade_id"] = np.arange(1, len(trades) + 1)
    return trades


def _build_variant(
    analysis: SymbolAnalysis,
    controls: pd.DataFrame,
    name: str,
    family: str,
    feature_name: str,
    bucketing: str,
    description: str,
    calibration_scope: str,
    parameters: dict[str, Any],
    constraints: PropConstraintConfig,
    note: str = "",
    rerun_with_sizing: bool = False,
) -> RegimeVariantRun:
    selected_map = dict(zip(pd.to_datetime(controls["session_date"]).dt.date, controls["risk_multiplier"]))
    keep_sessions = {session for session, multiplier in selected_map.items() if float(multiplier) > 0}
    nominal_trades = analysis.baseline_trades.copy()
    if not nominal_trades.empty:
        nominal_trades["session_date"] = pd.to_datetime(nominal_trades["session_date"]).dt.date
        nominal_trades = nominal_trades.loc[nominal_trades["session_date"].isin(set(selected_map))].copy().reset_index(drop=True)

    if rerun_with_sizing:
        trades = _scale_nominal_trades_by_multiplier(
            nominal_trades=nominal_trades,
            controls=controls,
            account_size_usd=analysis.baseline.account_size_usd,
            base_risk_pct=analysis.baseline.risk_per_trade_pct,
            tick_value_usd=float(analysis.instrument_spec["tick_value_usd"]),
            point_value_usd=float(analysis.instrument_spec["point_value_usd"]),
            commission_per_side_usd=float(analysis.instrument_spec["commission_per_side_usd"]),
        )
    else:
        trades = nominal_trades.loc[nominal_trades["session_date"].isin(keep_sessions)].copy().reset_index(drop=True)

    daily_results = _rebuild_daily_results_from_trades(
        trades,
        all_sessions=analysis.all_sessions,
        initial_capital=analysis.baseline.account_size_usd,
    )
    summary_by_scope = _build_summary_by_scope(
        trades=trades,
        daily_results=daily_results,
        all_sessions=analysis.all_sessions,
        is_sessions=analysis.is_sessions,
        oos_sessions=analysis.oos_sessions,
        initial_capital=analysis.baseline.account_size_usd,
        constraints=constraints,
    )
    return RegimeVariantRun(
        name=name,
        family=family,
        feature_name=feature_name,
        bucketing=bucketing,
        description=description,
        calibration_scope=calibration_scope,
        parameters=parameters,
        trades=trades,
        daily_results=daily_results,
        controls=controls,
        summary_by_scope=summary_by_scope,
        note=note,
    )


def _scope_trade_coverage(variant: RegimeVariantRun, nominal: RegimeVariantRun, scope: str) -> tuple[float, float]:
    trade_cov = _safe_div(
        float(_scope_value(variant.summary_by_scope, scope, "n_trades")),
        float(_scope_value(nominal.summary_by_scope, scope, "n_trades")),
        default=0.0,
    )
    day_cov = _safe_div(
        float(_scope_value(variant.summary_by_scope, scope, "n_days_traded")),
        float(_scope_value(nominal.summary_by_scope, scope, "n_days_traded")),
        default=0.0,
    )
    return trade_cov, day_cov


def _variant_verdict(row: dict[str, Any]) -> str:
    if row["variant_name"] == "nominal":
        return "baseline_reference"

    coverage = float(row.get("oos_trade_coverage_vs_nominal", 0.0))
    pnl_retention = float(row.get("oos_net_pnl_retention_vs_nominal", 0.0))
    sharpe_delta = float(row.get("oos_sharpe_delta_vs_nominal", 0.0))
    dd_improvement = float(row.get("oos_max_drawdown_improvement_vs_nominal", 0.0))

    if coverage < 0.55 and pnl_retention < 0.85:
        return "cuts_too_much_exposure"
    if pnl_retention < 0.90 and dd_improvement <= 0.0:
        return "worse_than_baseline"
    if (sharpe_delta >= 0.10 or dd_improvement >= 0.10) and coverage >= 0.60 and pnl_retention >= 0.85:
        return "useful"
    if abs(sharpe_delta) < 0.05 and abs(pnl_retention - 1.0) < 0.05 and abs(dd_improvement) < 0.05:
        return "cosmetic"
    return "mixed"


def _variant_row(variant: RegimeVariantRun, nominal: RegimeVariantRun) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant_name": variant.name,
        "family": variant.family,
        "feature_name": variant.feature_name,
        "bucketing": variant.bucketing,
        "description": variant.description,
        "calibration_scope": variant.calibration_scope,
        "parameters_json": json.dumps({key: _serialize_value(value) for key, value in variant.parameters.items()}, sort_keys=True),
        "note": variant.note,
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
        ]:
            row[f"{scope}_{column}"] = _scope_value(variant.summary_by_scope, scope, column)
        trade_cov, day_cov = _scope_trade_coverage(variant, nominal, scope)
        row[f"{scope}_trade_coverage_vs_nominal"] = trade_cov
        row[f"{scope}_day_coverage_vs_nominal"] = day_cov

    nominal_oos_pnl = float(_scope_value(nominal.summary_by_scope, "oos", "net_pnl"))
    nominal_oos_sharpe = float(_scope_value(nominal.summary_by_scope, "oos", "sharpe"))
    nominal_oos_dd = abs(float(_scope_value(nominal.summary_by_scope, "oos", "max_drawdown")))
    variant_oos_dd = abs(float(_scope_value(variant.summary_by_scope, "oos", "max_drawdown")))
    row["oos_net_pnl_retention_vs_nominal"] = _safe_div(
        float(_scope_value(variant.summary_by_scope, "oos", "net_pnl")),
        nominal_oos_pnl,
        default=0.0,
    )
    row["oos_sharpe_delta_vs_nominal"] = float(_scope_value(variant.summary_by_scope, "oos", "sharpe")) - nominal_oos_sharpe
    row["oos_max_drawdown_improvement_vs_nominal"] = _safe_div(
        nominal_oos_dd - variant_oos_dd,
        max(nominal_oos_dd, 1.0),
        default=0.0,
    )
    row["verdict"] = _variant_verdict(row)
    return row


def _export_variant_artifacts(root: Path, variant: RegimeVariantRun) -> None:
    variant_dir = root / "variants" / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    variant.trades.to_csv(variant_dir / "trades.csv", index=False)
    variant.daily_results.to_csv(variant_dir / "daily_results.csv", index=False)
    variant.controls.to_csv(variant_dir / "controls.csv", index=False)
    variant.summary_by_scope.to_csv(variant_dir / "metrics_by_scope.csv", index=False)


def _best_feature_per_family(feature_scores: pd.DataFrame) -> pd.DataFrame:
    if feature_scores.empty:
        return feature_scores.copy()
    valid = feature_scores.loc[feature_scores["valid_for_overlay"]].copy()
    if valid.empty:
        return valid
    return (
        valid.sort_values(["family", "feature_selection_score", "min_bucket_obs_is"], ascending=[True, False, False])
        .groupby("family", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _conditional_rows_for_feature(conditional_df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    out = conditional_df.loc[conditional_df["feature_name"] == feature_name].copy()
    return out.sort_values("bucket_position").reset_index(drop=True)


def _build_mapping_rows(
    variant_name: str,
    feature_name: str,
    feature_rows: pd.DataFrame,
    multipliers: dict[str, float],
) -> pd.DataFrame:
    rows = feature_rows.copy()
    rows["variant_name"] = variant_name
    rows["feature_name"] = feature_name
    rows["risk_multiplier"] = rows["bucket_label"].map(multipliers).astype(float)
    return rows


def _write_summary_markdown(
    output_path: Path,
    analysis: SymbolAnalysis,
    summary_df: pd.DataFrame,
    feature_scores: pd.DataFrame,
) -> None:
    nominal = summary_df.loc[summary_df["variant_name"] == "nominal"].iloc[0]
    useful = summary_df.loc[summary_df["verdict"] == "useful"].copy()
    if not useful.empty:
        useful = useful.sort_values(["oos_sharpe_delta_vs_nominal", "oos_max_drawdown_improvement_vs_nominal"], ascending=[False, False])

    volatility = feature_scores.loc[feature_scores["family"] == "volatility"].head(1)
    extension = feature_scores.loc[feature_scores["family"] == "extension"].head(1)
    structural = feature_scores.loc[feature_scores["family"] == "structural"].head(1)

    lines = [
        "# MNQ ORB Regime Filter And Dynamic Sizing Campaign",
        "",
        "## Baseline",
        "",
        f"- Baseline reused as official reference: `{analysis.symbol}` / OR{int(analysis.baseline.or_minutes)} / direction `{analysis.baseline.direction}` / RR `{float(analysis.baseline.target_multiple):.1f}` / ensemble `majority_50`.",
        f"- Dataset: `{analysis.dataset_path.name}`",
        f"- IS/OOS sessions: `{len(analysis.is_sessions)}` / `{len(analysis.oos_sessions)}`",
        "",
        "## Regime Readout",
        "",
        f"- Nominal OOS: net pnl `{float(nominal['oos_net_pnl']):.2f}` | Sharpe `{float(nominal['oos_sharpe']):.3f}` | Sortino `{float(nominal['oos_sortino']):.3f}` | PF `{float(nominal['oos_profit_factor']):.3f}` | maxDD `{float(nominal['oos_max_drawdown']):.2f}`.",
        (
            f"- Best volatility splitter in IS: `{volatility.iloc[0]['feature_name']}` (score spread `{float(volatility.iloc[0]['is_score_spread']):.3f}`, min bucket obs `{int(volatility.iloc[0]['min_bucket_obs_is'])}`)."
            if not volatility.empty
            else "- No volatility feature produced a defensible IS split."
        ),
        (
            f"- Best extension splitter in IS: `{extension.iloc[0]['feature_name']}` (score spread `{float(extension.iloc[0]['is_score_spread']):.3f}`, min bucket obs `{int(extension.iloc[0]['min_bucket_obs_is'])}`)."
            if not extension.empty
            else "- No extension feature produced a defensible IS split."
        ),
        (
            f"- Structural context readout: `{structural.iloc[0]['feature_name']}` selected, but it still needs OOS confirmation."
            if not structural.empty
            else "- Structural context did not produce a clean enough split for overlays."
        ),
        "",
        "## Overlay Verdict",
        "",
    ]

    if useful.empty:
        lines.append(
            "- No filter or sizing overlay tested here clearly beats the nominal ensemble once OOS retention and coverage are penalized honestly."
        )
    else:
        best = useful.iloc[0]
        lines.append(
            f"- Most defendable overlay: `{best['variant_name']}` with OOS pnl retention `{float(best['oos_net_pnl_retention_vs_nominal']):.2f}`, OOS trade coverage `{float(best['oos_trade_coverage_vs_nominal']):.2f}`, Sharpe delta `{float(best['oos_sharpe_delta_vs_nominal']):+.3f}`, maxDD improvement `{100.0 * float(best['oos_max_drawdown_improvement_vs_nominal']):.1f}%`."
        )

    cuts_too_much = summary_df.loc[summary_df["verdict"] == "cuts_too_much_exposure"].copy()
    if not cuts_too_much.empty:
        worst = cuts_too_much.iloc[0]
        lines.append(
            f"- Over-selective overlay to avoid: `{worst['variant_name']}` because it keeps only `{100.0 * float(worst['oos_trade_coverage_vs_nominal']):.1f}%` of nominal OOS trades for pnl retention `{100.0 * float(worst['oos_net_pnl_retention_vs_nominal']):.1f}%`."
        )

    worse = summary_df.loc[summary_df["verdict"] == "worse_than_baseline"].copy()
    if not worse.empty:
        lines.append(f"- Overlay clearly worse than baseline: `{worse.iloc[0]['variant_name']}`.")

    lines.extend(
        [
            "",
            "## Exports",
            "",
            "- `summary_variants.csv`",
            "- `conditional_bucket_analysis.csv`",
            "- `feature_ranking.csv`",
            "- `regime_state_mappings.csv`",
            "- `selected_session_regimes.csv`",
            "- `variants/<variant>/...`",
        ]
    )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_mnq_orb_regime_filter_sizing_campaign(spec: MnqRegimeFilterSizingSpec) -> dict[str, Any]:
    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(spec.output_root) if spec.output_root is not None else EXPORTS_DIR / f"mnq_orb_regime_filter_sizing_{timestamp}"
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
    regime_df = build_regime_dataset(analysis, selected_sessions)
    regime_path = output_root / "selected_session_regimes.csv"
    regime_df.to_csv(regime_path, index=False)

    nominal_trades = analysis.baseline_trades.copy()
    if not nominal_trades.empty:
        nominal_trades["session_date"] = pd.to_datetime(nominal_trades["session_date"]).dt.date
        nominal_trades = nominal_trades.loc[nominal_trades["session_date"].isin(set(regime_df["session_date"]))].copy()

    conditional_df, feature_score_df, assignments, calibrations = build_conditional_bucket_analysis(
        regime_df=regime_df,
        nominal_trades=nominal_trades,
        initial_capital=analysis.baseline.account_size_usd,
        feature_specs=_feature_specs(),
        min_bucket_obs_is=spec.min_bucket_obs_is,
    )
    conditional_path = output_root / "conditional_bucket_analysis.csv"
    conditional_df.to_csv(conditional_path, index=False)

    feature_ranking_path = output_root / "feature_ranking.csv"
    feature_score_df.to_csv(feature_ranking_path, index=False)

    variants: list[RegimeVariantRun] = []
    mapping_frames: list[pd.DataFrame] = []

    nominal_variant = _build_variant(
        analysis=analysis,
        controls=_nominal_controls(regime_df),
        name="nominal",
        family="baseline",
        feature_name="nominal",
        bucketing="none",
        description="Validated MNQ ORB ensemble baseline without extra regime overlay.",
        calibration_scope="none",
        parameters={},
        constraints=spec.prop_constraints,
        note="Reference official baseline for all retention and coverage comparisons.",
        rerun_with_sizing=False,
    )
    variants.append(nominal_variant)

    top_by_family = _best_feature_per_family(feature_score_df)
    for row in top_by_family.itertuples():
        feature_name = str(row.feature_name)
        feature_rows = _conditional_rows_for_feature(conditional_df, feature_name)
        if feature_rows.empty:
            continue
        worst_bucket = str(row.worst_bucket_is)
        keep_multipliers = {label: (0.0 if str(label) == worst_bucket else 1.0) for label in calibrations[feature_name].labels}
        controls = build_static_regime_controls(regime_df, feature_name, assignments[feature_name], keep_multipliers)
        variants.append(
            _build_variant(
                analysis=analysis,
                controls=controls,
                name=f"filter_skip_worst_{feature_name}",
                family="regime_filter",
                feature_name=feature_name,
                bucketing=f"{calibrations[feature_name].bucket_kind}_{len(calibrations[feature_name].labels)}",
                description=f"Skip the weakest IS bucket for feature {feature_name}.",
                calibration_scope="is_only",
                parameters={"bucket_multipliers": keep_multipliers},
                constraints=spec.prop_constraints,
                note=f"Bucket {worst_bucket} removed using IS-only ranking.",
                rerun_with_sizing=False,
            )
        )
        mapping_frames.append(_build_mapping_rows(f"filter_skip_worst_{feature_name}", feature_name, feature_rows, keep_multipliers))

    continuous_candidates = feature_score_df.loc[
        feature_score_df["valid_for_overlay"] & feature_score_df["bucket_kind"].eq("quantile")
    ].copy()
    if not continuous_candidates.empty:
        best_cont = continuous_candidates.iloc[0]
        best_feature = str(best_cont["feature_name"])
        feature_rows = _conditional_rows_for_feature(conditional_df, best_feature)
        sizing_3state = build_state_mapping_from_is_scores(feature_rows, multipliers_by_rank=(0.50, 0.75, 1.00))
        controls_3state = build_static_regime_controls(regime_df, best_feature, assignments[best_feature], sizing_3state)
        variants.append(
            _build_variant(
                analysis=analysis,
                controls=controls_3state,
                name=f"sizing_3state_{best_feature}",
                family="dynamic_sizing",
                feature_name=best_feature,
                bucketing=f"{calibrations[best_feature].bucket_kind}_{len(calibrations[best_feature].labels)}",
                description=f"Three-state discrete sizing on feature {best_feature}.",
                calibration_scope="is_only",
                parameters={"bucket_multipliers": sizing_3state},
                constraints=spec.prop_constraints,
                note="Worst bucket keeps reduced participation instead of a full skip.",
                rerun_with_sizing=True,
            )
        )
        mapping_frames.append(_build_mapping_rows(f"sizing_3state_{best_feature}", best_feature, feature_rows, sizing_3state))

        quartile_feature = next((feature for feature in _feature_specs() if feature.name == best_feature), None)
        if quartile_feature is not None:
            quartile_spec = RegimeFeatureSpec(
                name=quartile_feature.name,
                family=quartile_feature.family,
                description=quartile_feature.description,
                value_column=quartile_feature.value_column,
                bucket_kind="quantile",
                bucket_count=4,
            )
            quart_conditional, _, quart_assignments, quart_calibrations = build_conditional_bucket_analysis(
                regime_df=regime_df,
                nominal_trades=nominal_trades,
                initial_capital=analysis.baseline.account_size_usd,
                feature_specs=(quartile_spec,),
                min_bucket_obs_is=spec.min_bucket_obs_is,
            )
            quart_rows = _conditional_rows_for_feature(quart_conditional, best_feature)
            if len(quart_rows) == 4 and int(pd.to_numeric(quart_rows["is_n_obs"], errors="coerce").min()) >= spec.min_bucket_obs_is:
                sizing_4state = build_state_mapping_from_is_scores(quart_rows, multipliers_by_rank=(0.0, 0.50, 0.75, 1.00))
                controls_4state = build_static_regime_controls(regime_df, best_feature, quart_assignments[best_feature], sizing_4state)
                variants.append(
                    _build_variant(
                        analysis=analysis,
                        controls=controls_4state,
                        name=f"sizing_4state_{best_feature}",
                        family="dynamic_sizing",
                        feature_name=best_feature,
                        bucketing=f"{quart_calibrations[best_feature].bucket_kind}_{len(quart_calibrations[best_feature].labels)}",
                        description=f"Four-state discrete sizing on feature {best_feature}.",
                        calibration_scope="is_only",
                        parameters={"bucket_multipliers": sizing_4state},
                        constraints=spec.prop_constraints,
                        note="Bottom quartile is skipped; higher quartiles keep increasing participation.",
                        rerun_with_sizing=True,
                    )
                )
                mapping_frames.append(_build_mapping_rows(f"sizing_4state_{best_feature}", best_feature, quart_rows, sizing_4state))

    summary_rows = [_variant_row(variant, nominal_variant) for variant in variants]
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[[column for column in SUMMARY_COLUMNS if column in summary_df.columns]]
    summary_path = output_root / "summary_variants.csv"
    summary_df.to_csv(summary_path, index=False)

    mapping_df = pd.concat(mapping_frames, ignore_index=True) if mapping_frames else pd.DataFrame()
    mapping_path = output_root / "regime_state_mappings.csv"
    mapping_df.to_csv(mapping_path, index=False)

    for variant in variants:
        _export_variant_artifacts(output_root, variant)

    markdown_path = output_root / "campaign_summary.md"
    _write_summary_markdown(markdown_path, analysis=analysis, summary_df=summary_df, feature_scores=feature_score_df)

    metadata_path = output_root / "run_metadata.json"
    _json_dump(
        metadata_path,
        {
            "run_timestamp": datetime.now().isoformat(),
            "dataset_path": dataset_path,
            "selected_symbol": spec.symbol,
            "selected_aggregation_rule": spec.aggregation_rule,
            "selected_session_count": int(len(regime_df)),
            "spec": asdict(spec),
            "analysis_baseline_transfer": analysis.baseline_transfer,
            "analysis_best_ensemble": analysis.best_ensemble,
        },
    )

    return {
        "output_root": output_root,
        "summary": summary_path,
        "conditional": conditional_path,
        "feature_ranking": feature_ranking_path,
        "mappings": mapping_path,
        "regime_dataset": regime_path,
        "markdown": markdown_path,
        "metadata": metadata_path,
    }


def build_default_spec(output_root: Path | None = None) -> MnqRegimeFilterSizingSpec:
    return MnqRegimeFilterSizingSpec(output_root=output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the validated MNQ regime filter and dynamic sizing campaign.")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--is-fraction", type=float, default=0.70)
    args = parser.parse_args()

    spec = MnqRegimeFilterSizingSpec(
        dataset_path=Path(args.dataset_path) if args.dataset_path is not None else None,
        output_root=Path(args.output_root) if args.output_root is not None else None,
        is_fraction=float(args.is_fraction),
    )
    artifacts = run_mnq_orb_regime_filter_sizing_campaign(spec)
    print(f"summary: {artifacts['summary']}")
    print(f"markdown: {artifacts['markdown']}")


if __name__ == "__main__":
    main()
