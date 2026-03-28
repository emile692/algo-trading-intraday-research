"""Leak-free VVIX sizing-modulation campaign for the validated MNQ ORB baseline."""

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

from src.analytics.mnq_orb_regime_filter_sizing_campaign import _scale_nominal_trades_by_multiplier
from src.analytics.mnq_orb_vix_vvix_validation_campaign import (
    _bucket_bounds,
    _build_fixed_nominal_baseline,
    _build_summary_by_scope,
    _daily_results_from_trades,
    _safe_div,
    _scope_value,
    _selected_ensemble_sessions,
    _trade_subset,
    apply_bucket_calibration,
    build_vix_vvix_feature_frame,
    calibrate_quantile_buckets,
)
from src.analytics.mnq_orb_vvix_3state_phase2_campaign import (
    DEFAULT_SIZING_EXPORT_PREFIX,
    DEFAULT_SIZING_VARIANT,
    DEFAULT_VVIX_EXPORT_PREFIX,
    DEFAULT_VVIX_SURVIVOR_VARIANT,
    build_reference_3state_controls,
    compose_phase2_controls,
    find_latest_export,
)
from src.analytics.orb_multi_asset_campaign import (
    BaselineSpec,
    SearchGrid,
    SymbolAnalysis,
    analyze_symbol,
    resolve_processed_dataset,
)
from src.analytics.orb_vvix_overlay import build_vvix_filter_controls
from src.config.paths import EXPORTS_DIR, ensure_directories
from src.features.implied_volatility import (
    DEFAULT_VIX_DAILY_PATH,
    DEFAULT_VVIX_DAILY_PATH,
    load_vix_vvix_daily_features,
)


METRIC_COLUMNS = [
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
]

DELTA_METRICS = [
    "sharpe",
    "sortino",
    "profit_factor",
    "expectancy",
    "hit_rate",
    "avg_win",
    "avg_loss",
    "exposure_time_pct",
]

SUMMARY_COLUMNS = [
    "variant_name",
    "category",
    "source_variant_name",
    "primary_reference_variant_name",
    "is_core_configuration",
    "family",
    "feature_name",
    "combination_mode",
    "description",
    "uses_dynamic_sizing",
    "uses_3state_sizing",
    "uses_hard_filter_reference",
    "parameters_json",
    "note",
    "primary_screening_score",
    "primary_validation_score",
    "screening_score_vs_nominal",
    "validation_score_vs_nominal",
    "screening_score_vs_3state",
    "validation_score_vs_3state",
    "screening_status",
    "verdict",
]
for scope in ("overall", "is", "oos"):
    for metric in METRIC_COLUMNS:
        SUMMARY_COLUMNS.append(f"{scope}_{metric}")
for reference_name in ("baseline_nominal", "baseline_3state"):
    for scope in ("is", "oos"):
        SUMMARY_COLUMNS.extend(
            [
                f"{scope}_trade_coverage_vs_{reference_name}",
                f"{scope}_day_coverage_vs_{reference_name}",
                f"{scope}_net_pnl_retention_vs_{reference_name}",
            ]
        )
        for metric in DELTA_METRICS:
            SUMMARY_COLUMNS.append(f"{scope}_{metric}_delta_vs_{reference_name}")
        SUMMARY_COLUMNS.extend(
            [
                f"{scope}_stop_hit_rate_delta_vs_{reference_name}",
                f"{scope}_max_drawdown_improvement_vs_{reference_name}",
            ]
        )


@dataclass(frozen=True)
class VvixModulatorSpec:
    name: str
    family: str
    feature_name: str
    description: str
    bucket_multipliers: dict[str, float]
    calibration_scope: str = "is_only"


@dataclass
class VvixSizingVariantRun:
    name: str
    category: str
    source_variant_name: str | None
    primary_reference_variant_name: str
    is_core_configuration: bool
    family: str
    feature_name: str | None
    combination_mode: str | None
    description: str
    uses_dynamic_sizing: bool
    uses_3state_sizing: bool
    uses_hard_filter_reference: bool
    parameters: dict[str, Any]
    controls: pd.DataFrame
    trades: pd.DataFrame
    daily_results: pd.DataFrame
    summary_by_scope: pd.DataFrame
    note: str = ""


@dataclass(frozen=True)
class MnqOrbVvixSizingModulationSpec:
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
    commission_per_side_usd: float | None = None
    slippage_ticks: float | None = None
    vix_daily_path: Path = DEFAULT_VIX_DAILY_PATH
    vvix_daily_path: Path = DEFAULT_VVIX_DAILY_PATH
    vvix_export_root: Path | None = None
    hard_filter_variant_name: str = DEFAULT_VVIX_SURVIVOR_VARIANT
    sizing_export_root: Path | None = None
    sizing_variant_name: str = DEFAULT_SIZING_VARIANT
    primary_feature_name: str = "vvix_pct_63_t1"
    sensitivity_feature_names: tuple[str, ...] = ("vvix_pct_126_t1",)
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


def _score_relative_fields(row: dict[str, Any], scope: str, reference_name: str) -> float:
    trade_cov = float(row.get(f"{scope}_trade_coverage_vs_{reference_name}", 0.0))
    sharpe_delta = float(row.get(f"{scope}_sharpe_delta_vs_{reference_name}", 0.0))
    expectancy_delta = float(row.get(f"{scope}_expectancy_delta_vs_{reference_name}", 0.0))
    dd_improvement = float(row.get(f"{scope}_max_drawdown_improvement_vs_{reference_name}", 0.0))
    hit_delta = float(row.get(f"{scope}_hit_rate_delta_vs_{reference_name}", 0.0))
    stop_delta = float(row.get(f"{scope}_stop_hit_rate_delta_vs_{reference_name}", 0.0))
    pnl_retention = float(row.get(f"{scope}_net_pnl_retention_vs_{reference_name}", 0.0))
    avg_loss_delta = float(row.get(f"{scope}_avg_loss_delta_vs_{reference_name}", 0.0))
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
    if row["variant_name"] in {"baseline_nominal", "baseline_3state"}:
        return "baseline_reference"
    reference_name = str(row.get("primary_reference_variant_name") or "baseline_nominal")
    trade_cov = float(row.get(f"is_trade_coverage_vs_{reference_name}", 0.0))
    screening_score = float(row.get("primary_screening_score", 0.0))
    if trade_cov < 0.20:
        return "too_sparse"
    if screening_score > 0.35:
        return "selected_for_validation"
    if screening_score > 0.0:
        return "watchlist"
    return "screen_fail"


def _validation_verdict(row: dict[str, Any], min_oos_trades_for_positive: int) -> str:
    if row["variant_name"] in {"baseline_nominal", "baseline_3state"}:
        return "baseline_reference"
    oos_trades = int(row.get("oos_n_trades", 0))
    reference_name = str(row.get("primary_reference_variant_name") or "baseline_nominal")
    coverage = float(row.get(f"oos_trade_coverage_vs_{reference_name}", 0.0))
    pnl_retention = float(row.get(f"oos_net_pnl_retention_vs_{reference_name}", 0.0))
    validation_score = float(row.get("primary_validation_score", 0.0))
    dd_improvement = float(row.get(f"oos_max_drawdown_improvement_vs_{reference_name}", 0.0))
    expectancy_delta = float(row.get(f"oos_expectancy_delta_vs_{reference_name}", 0.0))
    sharpe_delta = float(row.get(f"oos_sharpe_delta_vs_{reference_name}", 0.0))
    if oos_trades < min_oos_trades_for_positive:
        return "insufficient_oos"
    if validation_score > 0.40 and coverage >= 0.55 and pnl_retention >= 0.85:
        return "robust_positive"
    if dd_improvement > 0.10 and coverage >= 0.35:
        return "defensive_positive"
    if validation_score > 0.0 and (expectancy_delta > 0.0 or sharpe_delta > 0.0):
        return "mixed_positive"
    if float(row.get("primary_screening_score", 0.0)) > 0.20 and validation_score <= 0.0:
        return "is_only"
    if coverage < 0.20:
        return "cuts_too_much_exposure"
    return "no_value"


def _build_nominal_controls(session_context: pd.DataFrame) -> pd.DataFrame:
    controls = session_context[["session_date", "phase"]].copy()
    controls["feature_name"] = "nominal"
    controls["feature_value"] = np.nan
    controls["bucket_label"] = "all"
    controls["risk_multiplier"] = 1.0
    controls["selected"] = True
    controls["skip_trade"] = False
    return controls.sort_values("session_date").reset_index(drop=True)


def _candidate_modulator_specs(feature_name: str) -> tuple[VvixModulatorSpec, ...]:
    return (
        VvixModulatorSpec(
            name=f"candidate_low_penalty_025__{feature_name}",
            family="low_vvix_penalty",
            feature_name=feature_name,
            description=f"`{feature_name}` low bucket -> 0.25x, else 1.0x.",
            bucket_multipliers={"low": 0.25, "mid": 1.0, "high": 1.0},
        ),
        VvixModulatorSpec(
            name=f"candidate_low_penalty_050__{feature_name}",
            family="low_vvix_penalty",
            feature_name=feature_name,
            description=f"`{feature_name}` low bucket -> 0.50x, else 1.0x.",
            bucket_multipliers={"low": 0.50, "mid": 1.0, "high": 1.0},
        ),
        VvixModulatorSpec(
            name=f"candidate_low_penalty_075__{feature_name}",
            family="low_vvix_penalty",
            feature_name=feature_name,
            description=f"`{feature_name}` low bucket -> 0.75x, else 1.0x.",
            bucket_multipliers={"low": 0.75, "mid": 1.0, "high": 1.0},
        ),
        VvixModulatorSpec(
            name=f"candidate_bucket_defensive_balanced__{feature_name}",
            family="bucket_modulation",
            feature_name=feature_name,
            description=f"`{feature_name}` low -> 0.50x, mid -> 1.0x, high -> 0.75x.",
            bucket_multipliers={"low": 0.50, "mid": 1.0, "high": 0.75},
        ),
    )


def build_vvix_modulation_controls(
    session_context: pd.DataFrame,
    feature_name: str,
    bucket_labels: pd.Series,
    bucket_multipliers: dict[str, float],
) -> pd.DataFrame:
    controls = session_context[["session_date", "phase"]].copy()
    controls["feature_name"] = str(feature_name)
    controls["feature_value"] = pd.to_numeric(session_context[feature_name], errors="coerce")
    controls["bucket_label"] = pd.Series(bucket_labels, index=session_context.index, dtype="string")
    controls["vvix_multiplier"] = controls["bucket_label"].map(bucket_multipliers).fillna(0.0).astype(float)
    controls["risk_multiplier"] = controls["vvix_multiplier"]
    controls["selected"] = controls["risk_multiplier"] > 0.0
    controls["skip_trade"] = ~controls["selected"]
    return controls.sort_values("session_date").reset_index(drop=True)


def combine_risk_multipliers(
    sizing_multipliers: pd.Series | np.ndarray,
    vvix_multipliers: pd.Series | np.ndarray,
    mode: str,
) -> pd.Series:
    sizing = pd.to_numeric(pd.Series(sizing_multipliers), errors="coerce").fillna(0.0).astype(float)
    vvix = pd.to_numeric(pd.Series(vvix_multipliers), errors="coerce").fillna(0.0).astype(float)
    if mode == "multiplicative":
        out = sizing * vvix
    elif mode == "cap":
        out = np.minimum(sizing, vvix)
    else:
        raise ValueError(f"Unsupported combination mode: {mode!r}")
    return pd.Series(out, index=sizing.index, dtype=float).clip(lower=0.0)


def build_combined_vvix_3state_controls(
    session_context: pd.DataFrame,
    vvix_controls: pd.DataFrame,
    sizing_controls: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    controls = session_context.copy()
    vvix_view = vvix_controls.copy()
    vvix_view["session_date"] = pd.to_datetime(vvix_view["session_date"]).dt.date
    vvix_view = vvix_view[
        ["session_date", "feature_name", "feature_value", "bucket_label", "vvix_multiplier"]
    ].rename(
        columns={
            "feature_name": "vvix_feature_name",
            "feature_value": "vvix_feature_value",
            "bucket_label": "vvix_bucket_label",
        }
    )
    controls = controls.merge(vvix_view, on="session_date", how="left", validate="one_to_one")
    sizing_view = sizing_controls.copy()
    sizing_view["session_date"] = pd.to_datetime(sizing_view["session_date"]).dt.date
    sizing_view = sizing_view[
        ["session_date", "feature_name", "feature_value", "bucket_label", "risk_multiplier"]
    ].rename(
        columns={
            "feature_name": "sizing_feature_name",
            "feature_value": "sizing_feature_value",
            "bucket_label": "sizing_bucket_label",
            "risk_multiplier": "sizing_risk_multiplier",
        }
    )
    controls = controls.merge(sizing_view, on="session_date", how="left", validate="one_to_one")
    controls["vvix_multiplier"] = pd.to_numeric(controls["vvix_multiplier"], errors="coerce").fillna(0.0)
    controls["sizing_risk_multiplier"] = pd.to_numeric(controls["sizing_risk_multiplier"], errors="coerce").fillna(1.0)
    controls["risk_multiplier"] = combine_risk_multipliers(
        controls["sizing_risk_multiplier"],
        controls["vvix_multiplier"],
        mode=mode,
    )
    controls["selected"] = controls["risk_multiplier"] > 0.0
    controls["skip_trade"] = ~controls["selected"]
    return controls.sort_values("session_date").reset_index(drop=True)


def _build_variant_run(
    analysis: SymbolAnalysis,
    spec: MnqOrbVvixSizingModulationSpec,
    name: str,
    category: str,
    source_variant_name: str | None,
    primary_reference_variant_name: str,
    is_core_configuration: bool,
    family: str,
    feature_name: str | None,
    combination_mode: str | None,
    description: str,
    controls: pd.DataFrame,
    base_nominal_trades: pd.DataFrame,
    all_sessions: list,
    is_sessions: list,
    oos_sessions: list,
    uses_dynamic_sizing: bool,
    uses_3state_sizing: bool,
    uses_hard_filter_reference: bool,
    parameters: dict[str, Any] | None = None,
    note: str = "",
) -> VvixSizingVariantRun:
    resolved_commission_per_side = float(
        analysis.instrument_spec["commission_per_side_usd"]
        if spec.commission_per_side_usd is None
        else spec.commission_per_side_usd
    )
    if uses_dynamic_sizing:
        trades = _scale_nominal_trades_by_multiplier(
            nominal_trades=base_nominal_trades,
            controls=controls,
            account_size_usd=float(spec.initial_capital_usd),
            base_risk_pct=float(analysis.baseline.risk_per_trade_pct),
            tick_value_usd=float(analysis.instrument_spec["tick_value_usd"]),
            point_value_usd=float(analysis.instrument_spec["point_value_usd"]),
            commission_per_side_usd=resolved_commission_per_side,
        )
    else:
        keep_sessions = set(pd.to_datetime(controls.loc[controls["risk_multiplier"] > 0.0, "session_date"]).dt.date)
        trades = _trade_subset(base_nominal_trades, list(keep_sessions))

    session_minutes = float(
        (pd.Timestamp(spec.baseline.time_exit) - pd.Timestamp(spec.baseline.opening_time)).total_seconds() / 60.0
    )
    return VvixSizingVariantRun(
        name=name,
        category=category,
        source_variant_name=source_variant_name,
        primary_reference_variant_name=primary_reference_variant_name,
        is_core_configuration=bool(is_core_configuration),
        family=family,
        feature_name=feature_name,
        combination_mode=combination_mode,
        description=description,
        uses_dynamic_sizing=bool(uses_dynamic_sizing),
        uses_3state_sizing=bool(uses_3state_sizing),
        uses_hard_filter_reference=bool(uses_hard_filter_reference),
        parameters=parameters or {},
        controls=controls.copy(),
        trades=trades.copy(),
        daily_results=_daily_results_from_trades(trades, all_sessions, float(spec.initial_capital_usd)),
        summary_by_scope=_build_summary_by_scope(
            trades=trades,
            all_sessions=all_sessions,
            is_sessions=is_sessions,
            oos_sessions=oos_sessions,
            initial_capital=float(spec.initial_capital_usd),
            session_minutes=session_minutes,
        ),
        note=note,
    )


def _clone_variant_run(
    variant: VvixSizingVariantRun,
    name: str,
    category: str,
    primary_reference_variant_name: str,
    description: str,
    note: str,
) -> VvixSizingVariantRun:
    return VvixSizingVariantRun(
        name=name,
        category=category,
        source_variant_name=variant.name if variant.source_variant_name is None else variant.source_variant_name,
        primary_reference_variant_name=primary_reference_variant_name,
        is_core_configuration=True,
        family=variant.family,
        feature_name=variant.feature_name,
        combination_mode=variant.combination_mode,
        description=description,
        uses_dynamic_sizing=variant.uses_dynamic_sizing,
        uses_3state_sizing=variant.uses_3state_sizing,
        uses_hard_filter_reference=variant.uses_hard_filter_reference,
        parameters=dict(variant.parameters),
        controls=variant.controls.copy(),
        trades=variant.trades.copy(),
        daily_results=variant.daily_results.copy(),
        summary_by_scope=variant.summary_by_scope.copy(),
        note=note,
    )


def _attach_relative_fields(
    row: dict[str, Any],
    variant: VvixSizingVariantRun,
    reference_variant: VvixSizingVariantRun,
    reference_name: str,
) -> None:
    for scope in ("is", "oos"):
        row[f"{scope}_trade_coverage_vs_{reference_name}"] = _safe_div(
            float(_scope_value(variant.summary_by_scope, scope, "n_trades")),
            float(_scope_value(reference_variant.summary_by_scope, scope, "n_trades")),
            default=0.0,
        )
        row[f"{scope}_day_coverage_vs_{reference_name}"] = _safe_div(
            float(_scope_value(variant.summary_by_scope, scope, "n_days_traded")),
            float(_scope_value(reference_variant.summary_by_scope, scope, "n_days_traded")),
            default=0.0,
        )
        row[f"{scope}_net_pnl_retention_vs_{reference_name}"] = _safe_div(
            float(_scope_value(variant.summary_by_scope, scope, "net_pnl")),
            float(_scope_value(reference_variant.summary_by_scope, scope, "net_pnl")),
            default=0.0,
        )
        for metric in DELTA_METRICS:
            row[f"{scope}_{metric}_delta_vs_{reference_name}"] = float(
                _scope_value(variant.summary_by_scope, scope, metric)
            ) - float(_scope_value(reference_variant.summary_by_scope, scope, metric))
        row[f"{scope}_stop_hit_rate_delta_vs_{reference_name}"] = float(
            _scope_value(reference_variant.summary_by_scope, scope, "stop_hit_rate")
        ) - float(_scope_value(variant.summary_by_scope, scope, "stop_hit_rate"))
        reference_dd = abs(float(_scope_value(reference_variant.summary_by_scope, scope, "max_drawdown")))
        variant_dd = abs(float(_scope_value(variant.summary_by_scope, scope, "max_drawdown")))
        row[f"{scope}_max_drawdown_improvement_vs_{reference_name}"] = _safe_div(
            reference_dd - variant_dd,
            max(reference_dd, 1.0),
            default=0.0,
        )


def _variant_row(
    variant: VvixSizingVariantRun,
    baseline_nominal: VvixSizingVariantRun,
    baseline_3state: VvixSizingVariantRun,
    spec: MnqOrbVvixSizingModulationSpec,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant_name": variant.name,
        "category": variant.category,
        "source_variant_name": variant.source_variant_name,
        "primary_reference_variant_name": variant.primary_reference_variant_name,
        "is_core_configuration": variant.is_core_configuration,
        "family": variant.family,
        "feature_name": variant.feature_name,
        "combination_mode": variant.combination_mode,
        "description": variant.description,
        "uses_dynamic_sizing": variant.uses_dynamic_sizing,
        "uses_3state_sizing": variant.uses_3state_sizing,
        "uses_hard_filter_reference": variant.uses_hard_filter_reference,
        "parameters_json": json.dumps(
            {key: _serialize_value(value) for key, value in variant.parameters.items()},
            sort_keys=True,
        ),
        "note": variant.note,
    }
    for scope in ("overall", "is", "oos"):
        for metric in METRIC_COLUMNS:
            row[f"{scope}_{metric}"] = _scope_value(variant.summary_by_scope, scope, metric)

    _attach_relative_fields(row, variant, baseline_nominal, "baseline_nominal")
    _attach_relative_fields(row, variant, baseline_3state, "baseline_3state")

    row["screening_score_vs_nominal"] = _score_relative_fields(row, scope="is", reference_name="baseline_nominal")
    row["validation_score_vs_nominal"] = _score_relative_fields(row, scope="oos", reference_name="baseline_nominal")
    row["screening_score_vs_3state"] = _score_relative_fields(row, scope="is", reference_name="baseline_3state")
    row["validation_score_vs_3state"] = _score_relative_fields(row, scope="oos", reference_name="baseline_3state")

    if variant.primary_reference_variant_name == "baseline_3state":
        row["primary_screening_score"] = row["screening_score_vs_3state"]
        row["primary_validation_score"] = row["validation_score_vs_3state"]
    else:
        row["primary_screening_score"] = row["screening_score_vs_nominal"]
        row["primary_validation_score"] = row["validation_score_vs_nominal"]

    row["screening_status"] = _screening_status(row)
    row["verdict"] = _validation_verdict(row, min_oos_trades_for_positive=spec.min_oos_trades_for_positive)
    return row


def _comparison_row(
    comparison_name: str,
    scope: str,
    left: VvixSizingVariantRun,
    right: VvixSizingVariantRun,
) -> dict[str, Any]:
    left_dd = abs(float(_scope_value(left.summary_by_scope, scope, "max_drawdown")))
    right_dd = abs(float(_scope_value(right.summary_by_scope, scope, "max_drawdown")))
    return {
        "comparison_type": "pairwise",
        "comparison_name": comparison_name,
        "scope": scope,
        "variant_name": left.name,
        "reference_variant_name": right.name,
        "trade_coverage_vs_reference": _safe_div(
            float(_scope_value(left.summary_by_scope, scope, "n_trades")),
            float(_scope_value(right.summary_by_scope, scope, "n_trades")),
            default=0.0,
        ),
        "day_coverage_vs_reference": _safe_div(
            float(_scope_value(left.summary_by_scope, scope, "n_days_traded")),
            float(_scope_value(right.summary_by_scope, scope, "n_days_traded")),
            default=0.0,
        ),
        "net_pnl_ratio_vs_reference": _safe_div(
            float(_scope_value(left.summary_by_scope, scope, "net_pnl")),
            float(_scope_value(right.summary_by_scope, scope, "net_pnl")),
            default=0.0,
        ),
        "net_pnl_delta": float(_scope_value(left.summary_by_scope, scope, "net_pnl"))
        - float(_scope_value(right.summary_by_scope, scope, "net_pnl")),
        "sharpe_delta": float(_scope_value(left.summary_by_scope, scope, "sharpe"))
        - float(_scope_value(right.summary_by_scope, scope, "sharpe")),
        "sortino_delta": float(_scope_value(left.summary_by_scope, scope, "sortino"))
        - float(_scope_value(right.summary_by_scope, scope, "sortino")),
        "profit_factor_delta": float(_scope_value(left.summary_by_scope, scope, "profit_factor"))
        - float(_scope_value(right.summary_by_scope, scope, "profit_factor")),
        "expectancy_delta": float(_scope_value(left.summary_by_scope, scope, "expectancy"))
        - float(_scope_value(right.summary_by_scope, scope, "expectancy")),
        "max_drawdown_improvement_vs_reference": _safe_div(
            right_dd - left_dd,
            max(right_dd, 1.0),
            default=0.0,
        ),
        "hit_rate_delta": float(_scope_value(left.summary_by_scope, scope, "hit_rate"))
        - float(_scope_value(right.summary_by_scope, scope, "hit_rate")),
        "avg_win_delta": float(_scope_value(left.summary_by_scope, scope, "avg_win"))
        - float(_scope_value(right.summary_by_scope, scope, "avg_win")),
        "avg_loss_delta": float(_scope_value(left.summary_by_scope, scope, "avg_loss"))
        - float(_scope_value(right.summary_by_scope, scope, "avg_loss")),
        "stop_hit_rate_delta": float(_scope_value(right.summary_by_scope, scope, "stop_hit_rate"))
        - float(_scope_value(left.summary_by_scope, scope, "stop_hit_rate")),
    }


def build_sizing_component_comparison_summary(
    variants: dict[str, VvixSizingVariantRun],
) -> pd.DataFrame:
    baseline_nominal = variants["baseline_nominal"]
    baseline_3state = variants["baseline_3state"]
    baseline_vvix_modulator = variants["baseline_vvix_modulator"]
    baseline_3state_vvix_modulator = variants["baseline_3state_vvix_modulator"]

    rows: list[dict[str, Any]] = []
    pairings = [
        ("marginal_3state_vs_baseline_nominal", baseline_3state, baseline_nominal),
        ("marginal_vvix_modulator_vs_baseline_nominal", baseline_vvix_modulator, baseline_nominal),
        ("vvix_modulator_vs_3state", baseline_vvix_modulator, baseline_3state),
        ("combined_vs_baseline_nominal", baseline_3state_vvix_modulator, baseline_nominal),
        ("incremental_vvix_on_top_of_3state", baseline_3state_vvix_modulator, baseline_3state),
        ("incremental_3state_on_top_of_vvix_modulator", baseline_3state_vvix_modulator, baseline_vvix_modulator),
    ]
    if "reference_vvix_hard_filter_nominal" in variants:
        pairings.extend(
            [
                (
                    "vvix_modulator_vs_hard_filter_nominal",
                    baseline_vvix_modulator,
                    variants["reference_vvix_hard_filter_nominal"],
                ),
                (
                    "hard_filter_nominal_vs_baseline_nominal",
                    variants["reference_vvix_hard_filter_nominal"],
                    baseline_nominal,
                ),
            ]
        )
    if "reference_vvix_hard_filter_3state" in variants:
        pairings.extend(
            [
                (
                    "combined_vs_hard_filter_3state",
                    baseline_3state_vvix_modulator,
                    variants["reference_vvix_hard_filter_3state"],
                ),
                (
                    "hard_filter_3state_vs_baseline_3state",
                    variants["reference_vvix_hard_filter_3state"],
                    baseline_3state,
                ),
            ]
        )

    for scope in ("overall", "is", "oos"):
        for comparison_name, left, right in pairings:
            rows.append(_comparison_row(comparison_name, scope, left, right))

        combined_vs_nominal = _comparison_row(
            "combined_vs_baseline_nominal",
            scope,
            baseline_3state_vvix_modulator,
            baseline_nominal,
        )
        sizing_vs_nominal = _comparison_row(
            "marginal_3state_vs_baseline_nominal",
            scope,
            baseline_3state,
            baseline_nominal,
        )
        vvix_vs_nominal = _comparison_row(
            "marginal_vvix_modulator_vs_baseline_nominal",
            scope,
            baseline_vvix_modulator,
            baseline_nominal,
        )
        rows.append(
            {
                "comparison_type": "interaction",
                "comparison_name": "interaction_excess_vs_additive",
                "scope": scope,
                "variant_name": baseline_3state_vvix_modulator.name,
                "reference_variant_name": baseline_nominal.name,
                "combined_net_pnl_delta": combined_vs_nominal["net_pnl_delta"],
                "additive_net_pnl_delta": sizing_vs_nominal["net_pnl_delta"] + vvix_vs_nominal["net_pnl_delta"],
                "interaction_excess_net_pnl_delta": combined_vs_nominal["net_pnl_delta"]
                - (sizing_vs_nominal["net_pnl_delta"] + vvix_vs_nominal["net_pnl_delta"]),
                "combined_sharpe_delta": combined_vs_nominal["sharpe_delta"],
                "additive_sharpe_delta": sizing_vs_nominal["sharpe_delta"] + vvix_vs_nominal["sharpe_delta"],
                "interaction_excess_sharpe_delta": combined_vs_nominal["sharpe_delta"]
                - (sizing_vs_nominal["sharpe_delta"] + vvix_vs_nominal["sharpe_delta"]),
                "combined_profit_factor_delta": combined_vs_nominal["profit_factor_delta"],
                "additive_profit_factor_delta": sizing_vs_nominal["profit_factor_delta"]
                + vvix_vs_nominal["profit_factor_delta"],
                "interaction_excess_profit_factor_delta": combined_vs_nominal["profit_factor_delta"]
                - (sizing_vs_nominal["profit_factor_delta"] + vvix_vs_nominal["profit_factor_delta"]),
                "combined_expectancy_delta": combined_vs_nominal["expectancy_delta"],
                "additive_expectancy_delta": sizing_vs_nominal["expectancy_delta"] + vvix_vs_nominal["expectancy_delta"],
                "interaction_excess_expectancy_delta": combined_vs_nominal["expectancy_delta"]
                - (sizing_vs_nominal["expectancy_delta"] + vvix_vs_nominal["expectancy_delta"]),
                "combined_max_drawdown_improvement": combined_vs_nominal["max_drawdown_improvement_vs_reference"],
                "additive_max_drawdown_improvement": sizing_vs_nominal["max_drawdown_improvement_vs_reference"]
                + vvix_vs_nominal["max_drawdown_improvement_vs_reference"],
                "interaction_excess_max_drawdown_improvement": combined_vs_nominal["max_drawdown_improvement_vs_reference"]
                - (
                    sizing_vs_nominal["max_drawdown_improvement_vs_reference"]
                    + vvix_vs_nominal["max_drawdown_improvement_vs_reference"]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_vvix_modulation_regime_summary(
    session_context: pd.DataFrame,
    calibration: Any,
    vvix_controls: pd.DataFrame,
    combined_controls: pd.DataFrame,
    variant_map: dict[str, VvixSizingVariantRun],
) -> pd.DataFrame:
    regime = session_context[["session_date", "phase"]].copy()
    regime = regime.merge(
        vvix_controls[["session_date", "bucket_label", "feature_value", "vvix_multiplier"]].rename(
            columns={
                "bucket_label": "vvix_bucket_label",
                "feature_value": "vvix_feature_value",
            }
        ),
        on="session_date",
        how="left",
        validate="one_to_one",
    )
    regime = regime.merge(
        combined_controls[
            [
                "session_date",
                "sizing_bucket_label",
                "sizing_risk_multiplier",
                "risk_multiplier",
            ]
        ].rename(columns={"risk_multiplier": "combined_risk_multiplier"}),
        on="session_date",
        how="left",
        validate="one_to_one",
    )
    regime["lower_bound"] = regime["vvix_bucket_label"].map(
        lambda label: _bucket_bounds(calibration, str(label))[0] if pd.notna(label) else np.nan
    )
    regime["upper_bound"] = regime["vvix_bucket_label"].map(
        lambda label: _bucket_bounds(calibration, str(label))[1] if pd.notna(label) else np.nan
    )

    for variant_name, variant in variant_map.items():
        grouped = (
            variant.trades.assign(session_date=pd.to_datetime(variant.trades["session_date"]).dt.date)
            .groupby("session_date", as_index=False)
            .agg(
                net_pnl_usd=("net_pnl_usd", "sum"),
                n_trades=("trade_id", "count"),
                avg_quantity=("quantity", "mean"),
            )
            if not variant.trades.empty
            else pd.DataFrame(columns=["session_date", "net_pnl_usd", "n_trades", "avg_quantity"])
        )
        regime = regime.merge(
            grouped.rename(
                columns={
                    "net_pnl_usd": f"{variant_name}_net_pnl_usd",
                    "n_trades": f"{variant_name}_n_trades",
                    "avg_quantity": f"{variant_name}_avg_quantity",
                }
            ),
            on="session_date",
            how="left",
        )

    fill_values = {
        column: 0.0
        for column in regime.columns
        if column.endswith("_net_pnl_usd") or column.endswith("_avg_quantity")
    }
    fill_values.update({column: 0 for column in regime.columns if column.endswith("_n_trades")})
    regime = regime.fillna(fill_values)

    return (
        regime.groupby(["phase", "vvix_bucket_label", "lower_bound", "upper_bound", "sizing_bucket_label"], dropna=False)
        .agg(
            session_count=("session_date", "count"),
            avg_vvix_feature_value=("vvix_feature_value", "mean"),
            avg_vvix_multiplier=("vvix_multiplier", "mean"),
            avg_sizing_multiplier=("sizing_risk_multiplier", "mean"),
            avg_combined_multiplier=("combined_risk_multiplier", "mean"),
            baseline_nominal_net_pnl_usd=("baseline_nominal_net_pnl_usd", "sum"),
            baseline_3state_net_pnl_usd=("baseline_3state_net_pnl_usd", "sum"),
            baseline_vvix_modulator_net_pnl_usd=("baseline_vvix_modulator_net_pnl_usd", "sum"),
            baseline_3state_vvix_modulator_net_pnl_usd=("baseline_3state_vvix_modulator_net_pnl_usd", "sum"),
            baseline_nominal_n_trades=("baseline_nominal_n_trades", "sum"),
            baseline_3state_n_trades=("baseline_3state_n_trades", "sum"),
            baseline_vvix_modulator_n_trades=("baseline_vvix_modulator_n_trades", "sum"),
            baseline_3state_vvix_modulator_n_trades=("baseline_3state_vvix_modulator_n_trades", "sum"),
        )
        .reset_index()
        .sort_values(["phase", "vvix_bucket_label", "sizing_bucket_label"])
        .reset_index(drop=True)
    )


def _export_variant_artifacts(root: Path, variant: VvixSizingVariantRun) -> None:
    variant_dir = root / "variants" / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    variant.controls.to_csv(variant_dir / "controls.csv", index=False)
    variant.trades.to_csv(variant_dir / "trades.csv", index=False)
    variant.daily_results.to_csv(variant_dir / "daily_results.csv", index=False)
    variant.summary_by_scope.to_csv(variant_dir / "metrics_by_scope.csv", index=False)


def _top_row(results_df: pd.DataFrame, variant_name: str) -> pd.Series:
    row = results_df.loc[results_df["variant_name"].astype(str).eq(str(variant_name))]
    if row.empty:
        raise KeyError(f"Variant {variant_name!r} was not found in results.")
    return row.iloc[0]


def _select_primary_standalone(
    results_df: pd.DataFrame,
    feature_name: str,
) -> pd.Series:
    candidates = results_df.loc[
        (results_df["category"].astype(str) == "vvix_candidate")
        & (results_df["feature_name"].astype(str) == str(feature_name))
    ].copy()
    if candidates.empty:
        raise RuntimeError(f"No standalone VVIX candidates were available for {feature_name}.")
    return candidates.sort_values(
        ["primary_screening_score", "is_sharpe_delta_vs_baseline_nominal", "is_profit_factor_delta_vs_baseline_nominal"],
        ascending=[False, False, False],
    ).iloc[0]


def _select_primary_combination(
    results_df: pd.DataFrame,
    source_variant_name: str,
) -> pd.Series:
    candidates = results_df.loc[
        (results_df["category"].astype(str) == "combo_candidate")
        & (results_df["source_variant_name"].astype(str) == str(source_variant_name))
    ].copy()
    if candidates.empty:
        raise RuntimeError(f"No combined VVIX candidates were available for source {source_variant_name!r}.")
    return candidates.sort_values(
        ["primary_screening_score", "is_sharpe_delta_vs_baseline_3state", "is_profit_factor_delta_vs_baseline_3state"],
        ascending=[False, False, False],
    ).iloc[0]


def _primary_mechanism(component_df: pd.DataFrame) -> str:
    oos = component_df.loc[
        (component_df["comparison_type"] == "pairwise") & (component_df["scope"] == "oos")
    ].copy()
    if oos.empty:
        return "none"

    mod_vs_nominal = oos.loc[oos["comparison_name"] == "marginal_vvix_modulator_vs_baseline_nominal"]
    combined_vs_3state = oos.loc[oos["comparison_name"] == "incremental_vvix_on_top_of_3state"]
    if not combined_vs_3state.empty:
        row = combined_vs_3state.iloc[0]
        if float(row["max_drawdown_improvement_vs_reference"]) > 0.10 and float(row["sharpe_delta"]) <= 0.05:
            return "drawdown_reduction"
        if float(row["sharpe_delta"]) > 0.10 or float(row["profit_factor_delta"]) > 0.05:
            return "risk_allocation_control"
    if not mod_vs_nominal.empty:
        row = mod_vs_nominal.iloc[0]
        if float(row["trade_coverage_vs_reference"]) > 0.90 and float(row["sharpe_delta"]) > 0.05:
            return "better_day_intensity_control"
    return "mixed_distribution_improvement"


def _synthesise_verdict(
    spec: MnqOrbVvixSizingModulationSpec,
    results_df: pd.DataFrame,
    component_df: pd.DataFrame,
    coverage_summary: dict[str, Any],
    primary_calibration: Any,
) -> dict[str, Any]:
    baseline_nominal = _top_row(results_df, "baseline_nominal")
    baseline_3state = _top_row(results_df, "baseline_3state")
    baseline_vvix_modulator = _top_row(results_df, "baseline_vvix_modulator")
    baseline_3state_vvix_modulator = _top_row(results_df, "baseline_3state_vvix_modulator")
    hard_filter_nominal = (
        _top_row(results_df, "reference_vvix_hard_filter_nominal")
        if "reference_vvix_hard_filter_nominal" in set(results_df["variant_name"].astype(str))
        else pd.Series(dtype="object")
    )
    hard_filter_3state = (
        _top_row(results_df, "reference_vvix_hard_filter_3state")
        if "reference_vvix_hard_filter_3state" in set(results_df["variant_name"].astype(str))
        else pd.Series(dtype="object")
    )

    combined_vs_3state = component_df.loc[
        (component_df["comparison_type"] == "pairwise")
        & (component_df["comparison_name"] == "incremental_vvix_on_top_of_3state")
        & (component_df["scope"] == "oos")
    ]
    modulator_vs_3state = component_df.loc[
        (component_df["comparison_type"] == "pairwise")
        & (component_df["comparison_name"] == "vvix_modulator_vs_3state")
        & (component_df["scope"] == "oos")
    ]
    modulator_vs_hard_filter = component_df.loc[
        (component_df["comparison_type"] == "pairwise")
        & (component_df["comparison_name"] == "vvix_modulator_vs_hard_filter_nominal")
        & (component_df["scope"] == "oos")
    ]
    combined_vs_hard_filter_3state = component_df.loc[
        (component_df["comparison_type"] == "pairwise")
        & (component_df["comparison_name"] == "combined_vs_hard_filter_3state")
        & (component_df["scope"] == "oos")
    ]
    interaction_oos = component_df.loc[
        (component_df["comparison_type"] == "interaction")
        & (component_df["comparison_name"] == "interaction_excess_vs_additive")
        & (component_df["scope"] == "oos")
    ]

    combined_vs_3state_row = combined_vs_3state.iloc[0] if not combined_vs_3state.empty else pd.Series(dtype="object")
    modulator_vs_3state_row = modulator_vs_3state.iloc[0] if not modulator_vs_3state.empty else pd.Series(dtype="object")
    modulator_vs_hard_filter_row = modulator_vs_hard_filter.iloc[0] if not modulator_vs_hard_filter.empty else pd.Series(dtype="object")
    combined_vs_hard_filter_row = (
        combined_vs_hard_filter_3state.iloc[0] if not combined_vs_hard_filter_3state.empty else pd.Series(dtype="object")
    )
    interaction_row = interaction_oos.iloc[0] if not interaction_oos.empty else pd.Series(dtype="object")

    modulator_beats_nominal = bool(
        float(baseline_vvix_modulator["oos_sharpe_delta_vs_baseline_nominal"]) > 0.05
        or (
            float(baseline_vvix_modulator["oos_max_drawdown_improvement_vs_baseline_nominal"]) > 0.10
            and float(baseline_vvix_modulator["oos_net_pnl_retention_vs_baseline_nominal"]) >= 0.90
        )
    )
    modulator_beats_3state = bool(
        not modulator_vs_3state.empty
        and float(modulator_vs_3state_row["sharpe_delta"]) > 0.05
        and float(modulator_vs_3state_row["profit_factor_delta"]) > 0.02
    )
    combined_beats_3state = bool(
        not combined_vs_3state.empty
        and float(combined_vs_3state_row["sharpe_delta"]) > -0.05
        and (
            float(combined_vs_3state_row["profit_factor_delta"]) > 0.05
            or float(combined_vs_3state_row["max_drawdown_improvement_vs_reference"]) > 0.12
        )
    )
    modulator_beats_hard_filter = bool(
        modulator_vs_hard_filter.empty
        or (
            float(modulator_vs_hard_filter_row["sharpe_delta"]) > 0.02
            or (
                float(modulator_vs_hard_filter_row["net_pnl_ratio_vs_reference"]) > 1.05
                and float(modulator_vs_hard_filter_row["max_drawdown_improvement_vs_reference"]) > -0.05
            )
        )
    )
    combined_beats_hard_filter_3state = bool(
        combined_vs_hard_filter_3state.empty
        or (
            float(combined_vs_hard_filter_row["sharpe_delta"]) > 0.02
            or float(combined_vs_hard_filter_row["profit_factor_delta"]) > 0.02
        )
    )

    if combined_beats_3state and float(interaction_row.get("interaction_excess_sharpe_delta", 0.0)) >= -0.05:
        recommendation = "promote_as_complementary_with_3state"
    elif modulator_beats_nominal and not modulator_beats_3state:
        recommendation = "keep_as_defensive_modulator_only"
    elif modulator_beats_3state:
        recommendation = "promote_as_primary_sizing_block"
    else:
        recommendation = "reject_for_primary_sizing_role"

    return {
        "run_type": "mnq_orb_vvix_sizing_modulation_validation",
        "primary_feature_name": spec.primary_feature_name,
        "hard_filter_reference_variant_name": spec.hard_filter_variant_name,
        "sizing_variant_name": spec.sizing_variant_name,
        "primary_vvix_modulator_variant_name": str(baseline_vvix_modulator["source_variant_name"]),
        "primary_vvix_modulator_family": str(baseline_vvix_modulator["family"]),
        "primary_vvix_modulator_feature_name": str(baseline_vvix_modulator["feature_name"]),
        "primary_combined_variant_name": str(baseline_3state_vvix_modulator["source_variant_name"]),
        "primary_combination_mode": str(baseline_3state_vvix_modulator["combination_mode"]),
        "vvix_modulator_better_than_nominal": modulator_beats_nominal,
        "vvix_modulator_better_than_3state": modulator_beats_3state,
        "vvix_modulator_better_than_hard_filter": modulator_beats_hard_filter,
        "vvix_better_as_modulator_than_hard_filter": modulator_beats_hard_filter,
        "combined_vvix_modulator_and_3state_better_than_3state": combined_beats_3state,
        "combined_vvix_modulator_and_3state_better_than_hard_filter_3state": combined_beats_hard_filter_3state,
        "oos_baseline_nominal_sharpe": float(baseline_nominal["oos_sharpe"]),
        "oos_baseline_3state_sharpe": float(baseline_3state["oos_sharpe"]),
        "oos_baseline_vvix_modulator_sharpe": float(baseline_vvix_modulator["oos_sharpe"]),
        "oos_baseline_3state_vvix_modulator_sharpe": float(baseline_3state_vvix_modulator["oos_sharpe"]),
        "oos_vvix_modulator_sharpe_delta_vs_nominal": float(
            baseline_vvix_modulator["oos_sharpe_delta_vs_baseline_nominal"]
        ),
        "oos_vvix_modulator_profit_factor_delta_vs_nominal": float(
            baseline_vvix_modulator["oos_profit_factor_delta_vs_baseline_nominal"]
        ),
        "oos_vvix_modulator_max_drawdown_improvement_vs_nominal": float(
            baseline_vvix_modulator["oos_max_drawdown_improvement_vs_baseline_nominal"]
        ),
        "oos_vvix_modulator_vs_3state_sharpe_delta": float(modulator_vs_3state_row.get("sharpe_delta", 0.0)),
        "oos_vvix_modulator_vs_3state_profit_factor_delta": float(
            modulator_vs_3state_row.get("profit_factor_delta", 0.0)
        ),
        "oos_vvix_modulator_vs_3state_max_drawdown_improvement": float(
            modulator_vs_3state_row.get("max_drawdown_improvement_vs_reference", 0.0)
        ),
        "oos_combined_vs_3state_sharpe_delta": float(combined_vs_3state_row.get("sharpe_delta", 0.0)),
        "oos_combined_vs_3state_profit_factor_delta": float(
            combined_vs_3state_row.get("profit_factor_delta", 0.0)
        ),
        "oos_combined_vs_3state_max_drawdown_improvement": float(
            combined_vs_3state_row.get("max_drawdown_improvement_vs_reference", 0.0)
        ),
        "oos_modulator_vs_hard_filter_sharpe_delta": float(modulator_vs_hard_filter_row.get("sharpe_delta", 0.0)),
        "oos_combined_vs_hard_filter_3state_sharpe_delta": float(
            combined_vs_hard_filter_row.get("sharpe_delta", 0.0)
        ),
        "oos_interaction_excess_sharpe_delta": float(interaction_row.get("interaction_excess_sharpe_delta", 0.0)),
        "oos_interaction_excess_profit_factor_delta": float(
            interaction_row.get("interaction_excess_profit_factor_delta", 0.0)
        ),
        "oos_interaction_excess_expectancy_delta": float(
            interaction_row.get("interaction_excess_expectancy_delta", 0.0)
        ),
        "oos_interaction_excess_max_drawdown_improvement": float(
            interaction_row.get("interaction_excess_max_drawdown_improvement", 0.0)
        ),
        "oos_hard_filter_nominal_sharpe": float(hard_filter_nominal.get("oos_sharpe", 0.0)),
        "oos_hard_filter_3state_sharpe": float(hard_filter_3state.get("oos_sharpe", 0.0)),
        "primary_mechanism": _primary_mechanism(component_df),
        "recommendation": recommendation,
        "primary_bucket_bounds": {
            str(label): {
                "lower_bound": _bucket_bounds(primary_calibration, str(label))[0],
                "upper_bound": _bucket_bounds(primary_calibration, str(label))[1],
            }
            for label in getattr(primary_calibration, "labels", ())
        },
        "coverage_summary": coverage_summary,
        "assumptions": [
            f"baseline direction={spec.baseline.direction}",
            f"baseline OR window={spec.baseline.or_minutes}m",
            f"aggregation_rule={spec.aggregation_rule}",
            f"fixed_contracts={spec.fixed_contracts}",
            f"primary_feature={spec.primary_feature_name}",
            f"hard_filter_reference={spec.hard_filter_variant_name}",
            f"sizing_variant={spec.sizing_variant_name}",
            "VVIX daily inputs remain strictly t-1.",
            "VVIX modulation buckets are calibrated on IS only and then frozen.",
            "3-state sizing overlay is reused from the validated export without recalibration on OOS.",
        ],
    }


def _write_report(
    output_path: Path,
    spec: MnqOrbVvixSizingModulationSpec,
    analysis: SymbolAnalysis,
    results_df: pd.DataFrame,
    component_df: pd.DataFrame,
    regime_summary_df: pd.DataFrame,
    coverage_summary: dict[str, Any],
    verdict: dict[str, Any],
) -> None:
    baseline_nominal = _top_row(results_df, "baseline_nominal")
    baseline_3state = _top_row(results_df, "baseline_3state")
    baseline_vvix_modulator = _top_row(results_df, "baseline_vvix_modulator")
    baseline_3state_vvix_modulator = _top_row(results_df, "baseline_3state_vvix_modulator")
    hard_filter_nominal = (
        _top_row(results_df, "reference_vvix_hard_filter_nominal")
        if "reference_vvix_hard_filter_nominal" in set(results_df["variant_name"].astype(str))
        else pd.Series(dtype="object")
    )

    modulator_vs_3state = component_df.loc[
        (component_df["comparison_type"] == "pairwise")
        & (component_df["comparison_name"] == "vvix_modulator_vs_3state")
        & (component_df["scope"] == "oos")
    ]
    combined_vs_3state = component_df.loc[
        (component_df["comparison_type"] == "pairwise")
        & (component_df["comparison_name"] == "incremental_vvix_on_top_of_3state")
        & (component_df["scope"] == "oos")
    ]
    modulator_vs_hard_filter = component_df.loc[
        (component_df["comparison_type"] == "pairwise")
        & (component_df["comparison_name"] == "vvix_modulator_vs_hard_filter_nominal")
        & (component_df["scope"] == "oos")
    ]
    interaction_oos = component_df.loc[
        (component_df["comparison_type"] == "interaction")
        & (component_df["comparison_name"] == "interaction_excess_vs_additive")
        & (component_df["scope"] == "oos")
    ]

    modulator_vs_3state_line = "- Le VVIX modulator seul reste en dessous du 3-state."
    if not modulator_vs_3state.empty:
        row = modulator_vs_3state.iloc[0]
        modulator_vs_3state_line = (
            f"- VVIX modulator seul vs 3-state: Sharpe delta `{float(row['sharpe_delta']):+.3f}`, "
            f"PF delta `{float(row['profit_factor_delta']):+.3f}`, "
            f"maxDD improvement `{100.0 * float(row['max_drawdown_improvement_vs_reference']):+.1f}%`."
        )

    combined_vs_3state_line = "- La combinaison n'améliore pas le 3-state de facon nette."
    if not combined_vs_3state.empty:
        row = combined_vs_3state.iloc[0]
        combined_vs_3state_line = (
            f"- VVIX + 3-state vs 3-state seul: Sharpe delta `{float(row['sharpe_delta']):+.3f}`, "
            f"PF delta `{float(row['profit_factor_delta']):+.3f}`, "
            f"maxDD improvement `{100.0 * float(row['max_drawdown_improvement_vs_reference']):+.1f}%`, "
            f"trade coverage `{100.0 * float(row['trade_coverage_vs_reference']):.1f}%`."
        )

    hard_filter_line = "- Pas de reference hard-filter disponible."
    if not modulator_vs_hard_filter.empty:
        row = modulator_vs_hard_filter.iloc[0]
        hard_filter_line = (
            f"- Modulator vs hard filter nominal: Sharpe delta `{float(row['sharpe_delta']):+.3f}`, "
            f"PF delta `{float(row['profit_factor_delta']):+.3f}`, "
            f"net pnl ratio `{float(row['net_pnl_ratio_vs_reference']):.3f}`."
        )

    interaction_line = "- Pas d'interaction mesurable."
    if not interaction_oos.empty:
        row = interaction_oos.iloc[0]
        interaction_line = (
            f"- Interaction combinee vs somme additive: Sharpe excess `{float(row['interaction_excess_sharpe_delta']):+.3f}`, "
            f"PF excess `{float(row['interaction_excess_profit_factor_delta']):+.3f}`, "
            f"expectancy excess `{float(row['interaction_excess_expectancy_delta']):+.2f}`, "
            f"maxDD excess `{100.0 * float(row['interaction_excess_max_drawdown_improvement']):+.1f}%`."
        )

    bucket_line = "- Lecture par regime indisponible."
    if not regime_summary_df.empty:
        low_bucket = regime_summary_df.loc[
            regime_summary_df["vvix_bucket_label"].astype(str).eq("low")
            & regime_summary_df["phase"].astype(str).eq("oos")
        ].copy()
        if not low_bucket.empty:
            row = low_bucket.sort_values("session_count", ascending=False).iloc[0]
            bucket_line = (
                f"- En OOS, bucket VVIX `low`: `{int(row['session_count'])}` sessions | "
                f"avg multiplier VVIX `{float(row['avg_vvix_multiplier']):.2f}` | "
                f"avg final multiplier combine `{float(row['avg_combined_multiplier']):.2f}`."
            )

    lines = [
        "# MNQ ORB VVIX Sizing Modulation Validation",
        "",
        "## Baseline And Scope",
        "",
        f"- Baseline ORB conservee: OR{int(spec.baseline.or_minutes)} / direction `{spec.baseline.direction}` / RR `{float(spec.baseline.target_multiple):.2f}` / VWAP confirmation `{bool(spec.baseline.vwap_confirmation)}`.",
        f"- Filtre ATR structurel conserve via l'ensemble `{spec.aggregation_rule}`; aucun changement du signal ORB ni des hypotheses d'execution du repo.",
        f"- Dataset: `{analysis.dataset_path.name}` | sessions IS/OOS d'origine: `{len(analysis.is_sessions)}` / `{len(analysis.oos_sessions)}`.",
        f"- Univers commun teste pour toutes les comparaisons coeur: `{int(coverage_summary['common_campaign_sessions'])}` sessions.",
        "",
        "## Calibration",
        "",
        f"- Modulateur VVIX primaire choisi sur IS: `{verdict['primary_vvix_modulator_variant_name']}` via `{verdict['primary_vvix_modulator_feature_name']}`.",
        f"- Mode de combinaison retenu sur IS: `{verdict['primary_combined_variant_name']}` (mode `{verdict['primary_combination_mode']}`).",
        f"- Survivor hard-filter de reference: `{spec.hard_filter_variant_name}`.",
        f"- 3-state de reference: `{spec.sizing_variant_name}`.",
        "- Buckets VVIX calibres sur IS seulement puis figes sur OOS.",
        "",
        "## Four Core Configurations",
        "",
        f"- `baseline_nominal`: OOS pnl `{float(baseline_nominal['oos_net_pnl']):.2f}` | Sharpe `{float(baseline_nominal['oos_sharpe']):.3f}` | PF `{float(baseline_nominal['oos_profit_factor']):.3f}` | maxDD `{float(baseline_nominal['oos_max_drawdown']):.2f}`.",
        f"- `baseline_3state`: OOS pnl `{float(baseline_3state['oos_net_pnl']):.2f}` | Sharpe `{float(baseline_3state['oos_sharpe']):.3f}` | PF `{float(baseline_3state['oos_profit_factor']):.3f}` | maxDD `{float(baseline_3state['oos_max_drawdown']):.2f}`.",
        f"- `baseline_vvix_modulator`: OOS pnl `{float(baseline_vvix_modulator['oos_net_pnl']):.2f}` | Sharpe `{float(baseline_vvix_modulator['oos_sharpe']):.3f}` | PF `{float(baseline_vvix_modulator['oos_profit_factor']):.3f}` | maxDD `{float(baseline_vvix_modulator['oos_max_drawdown']):.2f}`.",
        f"- `baseline_3state_vvix_modulator`: OOS pnl `{float(baseline_3state_vvix_modulator['oos_net_pnl']):.2f}` | Sharpe `{float(baseline_3state_vvix_modulator['oos_sharpe']):.3f}` | PF `{float(baseline_3state_vvix_modulator['oos_profit_factor']):.3f}` | maxDD `{float(baseline_3state_vvix_modulator['oos_max_drawdown']):.2f}`.",
        (
            f"- `reference_vvix_hard_filter_nominal`: OOS pnl `{float(hard_filter_nominal.get('oos_net_pnl', 0.0)):.2f}` | Sharpe `{float(hard_filter_nominal.get('oos_sharpe', 0.0)):.3f}` | PF `{float(hard_filter_nominal.get('oos_profit_factor', 0.0)):.3f}`."
            if not hard_filter_nominal.empty
            else "- Reference hard-filter nominale non disponible."
        ),
        "",
        "## Attribution",
        "",
        hard_filter_line,
        modulator_vs_3state_line,
        combined_vs_3state_line,
        interaction_line,
        bucket_line,
        "",
        "## Direct Answers",
        "",
        f"- Le VVIX fonctionne-t-il mieux comme sizing modulator que comme hard filter ? {'Oui, dans le cadre de cette campagne compacte.' if verdict['vvix_better_as_modulator_than_hard_filter'] else 'Non, le hard filter reste au moins aussi convaincant sur cet echantillon.'}",
        f"- Le VVIX modulator seul fait-il mieux que le nominal ? {'Oui.' if verdict['vvix_modulator_better_than_nominal'] else 'Non, pas de gain suffisant face au nominal.'}",
        f"- Le VVIX modulator seul fait-il mieux que le 3-state ? {'Oui.' if verdict['vvix_modulator_better_than_3state'] else 'Non, le 3-state garde l avantage principal.'}",
        f"- La combinaison VVIX modulator + 3-state ameliore-t-elle le 3-state ? {'Oui, au moins sur un axe cle du profil risque/performance.' if verdict['combined_vvix_modulator_and_3state_better_than_3state'] else 'Non, ou seulement de facon trop marginale pour declasser le 3-state.'}",
        f"- Le gain eventuel vient-il surtout du drawdown, de la regularite, ou d'un vrai moteur de perf ? `{verdict['primary_mechanism']}`.",
        f"- Faut-il promouvoir le VVIX comme bloc de sizing ? `{verdict['recommendation']}`.",
        "",
        "## Methodology Notes",
        "",
        "- La comparaison coeur se fait sur le meme univers commun de sessions VVIX + 3-state pour garder l'attribution propre.",
        "- Le signal ORB, les couts, le slippage, les commissions et le filtre ATR baseline restent inchanges.",
        "- Les features VVIX sont strictement laggees t-1.",
        "- Aucune reouverture d'une large campagne VIX/VVIX: seul un set compact de modulateurs interpretable autour de `vvix_pct_63_t1` est teste, avec une sensibilite locale sur `vvix_pct_126_t1`.",
        "",
        "## Exports",
        "",
        "- `screening_summary.csv`",
        "- `validation_summary.csv`",
        "- `full_variant_results.csv`",
        "- `sizing_component_comparison_summary.csv`",
        "- `vvix_modulation_regime_summary.csv`",
        "- `final_report.md`",
        "- `final_verdict.json`",
        "- `variants/<variant>/...`",
    ]
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_mnq_orb_vvix_sizing_modulation_campaign(
    spec: MnqOrbVvixSizingModulationSpec,
) -> dict[str, Any]:
    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(spec.output_root)
        if spec.output_root is not None
        else EXPORTS_DIR / f"mnq_orb_vvix_sizing_modulation_{timestamp}"
    )
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

    _, fixed_nominal_trades_full = _build_fixed_nominal_baseline(analysis, selected_sessions, spec)
    daily_features = load_vix_vvix_daily_features(
        vix_path=spec.vix_daily_path,
        vvix_path=spec.vvix_daily_path,
    )
    feature_frame_full = build_vix_vvix_feature_frame(analysis, fixed_nominal_trades_full, daily_features)
    if feature_frame_full.empty:
        raise RuntimeError("No overlap was found between the baseline selected sessions and VVIX daily features.")

    sizing_spec, sizing_controls_full = build_reference_3state_controls(
        analysis=analysis,
        selected_sessions=selected_sessions,
        export_root=spec.sizing_export_root,
        variant_name=spec.sizing_variant_name,
    )

    vvix_session_set = set(pd.to_datetime(feature_frame_full["session_date"]).dt.date)
    sizing_session_set = set(pd.to_datetime(sizing_controls_full["session_date"]).dt.date)
    common_sessions = sorted(vvix_session_set & sizing_session_set)
    if not common_sessions:
        raise RuntimeError("No common sessions were found between VVIX coverage and the 3-state sizing controls.")

    common_session_set = set(common_sessions)
    feature_frame = feature_frame_full.loc[
        pd.to_datetime(feature_frame_full["session_date"]).dt.date.isin(common_session_set)
    ].copy()
    fixed_nominal_trades = _trade_subset(fixed_nominal_trades_full, common_sessions)
    sizing_controls = sizing_controls_full.loc[
        pd.to_datetime(sizing_controls_full["session_date"]).dt.date.isin(common_session_set)
    ].copy()
    sizing_controls["session_date"] = pd.to_datetime(sizing_controls["session_date"]).dt.date

    session_columns = ["session_date", "phase", "breakout_side", "breakout_timing_bucket", spec.primary_feature_name]
    for feature_name in spec.sensitivity_feature_names:
        if feature_name in feature_frame.columns:
            session_columns.append(feature_name)
    session_context = (
        feature_frame[session_columns]
        .drop_duplicates(subset=["session_date"])
        .sort_values("session_date")
        .reset_index(drop=True)
    )

    campaign_is_sessions = [
        session for session in pd.to_datetime(pd.Index(analysis.is_sessions)).date if session in common_session_set
    ]
    campaign_oos_sessions = [
        session for session in pd.to_datetime(pd.Index(analysis.oos_sessions)).date if session in common_session_set
    ]

    base_controls = _build_nominal_controls(session_context)

    hard_filter_spec, hard_filter_controls = build_vvix_filter_controls(
        session_dates=common_sessions,
        export_root=spec.vvix_export_root,
        variant_name=spec.hard_filter_variant_name,
        vix_path=spec.vix_daily_path,
        vvix_path=spec.vvix_daily_path,
    )
    hard_filter_controls["session_date"] = pd.to_datetime(hard_filter_controls["session_date"]).dt.date
    hard_filter_controls = hard_filter_controls.loc[
        hard_filter_controls["session_date"].isin(common_session_set)
    ].copy()

    variants: list[VvixSizingVariantRun] = [
        _build_variant_run(
            analysis=analysis,
            spec=spec,
            name="baseline_nominal",
            category="baseline",
            source_variant_name=None,
            primary_reference_variant_name="baseline_nominal",
            is_core_configuration=True,
            family="baseline",
            feature_name=None,
            combination_mode=None,
            description="ORB baseline + ATR structurel + nominal fixe sur l'univers commun.",
            controls=base_controls,
            base_nominal_trades=fixed_nominal_trades,
            all_sessions=common_sessions,
            is_sessions=campaign_is_sessions,
            oos_sessions=campaign_oos_sessions,
            uses_dynamic_sizing=False,
            uses_3state_sizing=False,
            uses_hard_filter_reference=False,
            parameters={},
        ),
        _build_variant_run(
            analysis=analysis,
            spec=spec,
            name="baseline_3state",
            category="baseline",
            source_variant_name=sizing_spec.variant_name,
            primary_reference_variant_name="baseline_3state",
            is_core_configuration=True,
            family="3state_reference",
            feature_name=sizing_spec.feature_name,
            combination_mode=None,
            description="ORB baseline + ATR structurel + sizing 3-state fige.",
            controls=sizing_controls,
            base_nominal_trades=fixed_nominal_trades,
            all_sessions=common_sessions,
            is_sessions=campaign_is_sessions,
            oos_sessions=campaign_oos_sessions,
            uses_dynamic_sizing=True,
            uses_3state_sizing=True,
            uses_hard_filter_reference=False,
            parameters={"bucket_multipliers": sizing_spec.bucket_multipliers},
        ),
        _build_variant_run(
            analysis=analysis,
            spec=spec,
            name="reference_vvix_hard_filter_nominal",
            category="reference_hard_filter",
            source_variant_name=hard_filter_spec.variant_name,
            primary_reference_variant_name="baseline_nominal",
            is_core_configuration=False,
            family="vvix_hard_filter_reference",
            feature_name=hard_filter_spec.feature_name,
            combination_mode="hard_filter",
            description="Reference historique: survivor VVIX applique comme hard filter sur sizing nominal.",
            controls=compose_phase2_controls(session_context=session_context, vvix_controls=hard_filter_controls),
            base_nominal_trades=fixed_nominal_trades,
            all_sessions=common_sessions,
            is_sessions=campaign_is_sessions,
            oos_sessions=campaign_oos_sessions,
            uses_dynamic_sizing=False,
            uses_3state_sizing=False,
            uses_hard_filter_reference=True,
            parameters={"kept_buckets": hard_filter_spec.kept_buckets},
        ),
        _build_variant_run(
            analysis=analysis,
            spec=spec,
            name="reference_vvix_hard_filter_3state",
            category="reference_hard_filter",
            source_variant_name=hard_filter_spec.variant_name,
            primary_reference_variant_name="baseline_3state",
            is_core_configuration=False,
            family="vvix_hard_filter_reference",
            feature_name=hard_filter_spec.feature_name,
            combination_mode="hard_filter",
            description="Reference historique: survivor VVIX applique comme hard filter au-dessus du 3-state.",
            controls=compose_phase2_controls(
                session_context=session_context,
                vvix_controls=hard_filter_controls,
                sizing_controls=sizing_controls,
            ),
            base_nominal_trades=fixed_nominal_trades,
            all_sessions=common_sessions,
            is_sessions=campaign_is_sessions,
            oos_sessions=campaign_oos_sessions,
            uses_dynamic_sizing=True,
            uses_3state_sizing=True,
            uses_hard_filter_reference=True,
            parameters={
                "kept_buckets": hard_filter_spec.kept_buckets,
                "bucket_multipliers": sizing_spec.bucket_multipliers,
            },
        ),
    ]

    primary_calibration = calibrate_quantile_buckets(
        feature_name=spec.primary_feature_name,
        is_values=session_context.loc[session_context["phase"].astype(str) == "is", spec.primary_feature_name],
        bucket_count=spec.bucket_count,
    )
    primary_bucket_labels = apply_bucket_calibration(session_context[spec.primary_feature_name], primary_calibration)
    if len(getattr(primary_calibration, "labels", ())) != 3:
        raise RuntimeError(f"Primary VVIX calibration for {spec.primary_feature_name} did not produce 3 buckets.")

    standalone_specs = _candidate_modulator_specs(spec.primary_feature_name)
    for candidate_spec in standalone_specs:
        variants.append(
            _build_variant_run(
                analysis=analysis,
                spec=spec,
                name=candidate_spec.name,
                category="vvix_candidate",
                source_variant_name=None,
                primary_reference_variant_name="baseline_nominal",
                is_core_configuration=False,
                family=candidate_spec.family,
                feature_name=candidate_spec.feature_name,
                combination_mode=None,
                description=candidate_spec.description,
                controls=build_vvix_modulation_controls(
                    session_context=session_context,
                    feature_name=candidate_spec.feature_name,
                    bucket_labels=primary_bucket_labels,
                    bucket_multipliers=candidate_spec.bucket_multipliers,
                ),
                base_nominal_trades=fixed_nominal_trades,
                all_sessions=common_sessions,
                is_sessions=campaign_is_sessions,
                oos_sessions=campaign_oos_sessions,
                uses_dynamic_sizing=True,
                uses_3state_sizing=False,
                uses_hard_filter_reference=False,
                parameters={
                    "bucket_multipliers": candidate_spec.bucket_multipliers,
                    "calibration_scope": candidate_spec.calibration_scope,
                    "bucket_labels": getattr(primary_calibration, "labels", ()),
                },
            )
        )

    variant_map = {variant.name: variant for variant in variants}
    initial_results = pd.DataFrame(
        [_variant_row(variant, variant_map["baseline_nominal"], variant_map["baseline_3state"], spec) for variant in variants]
    )
    primary_standalone_row = _select_primary_standalone(initial_results, spec.primary_feature_name)
    primary_standalone_variant = variant_map[str(primary_standalone_row["variant_name"])]

    variants.append(
        _clone_variant_run(
            variant=primary_standalone_variant,
            name="baseline_vvix_modulator",
            category="core",
            primary_reference_variant_name="baseline_nominal",
            description="Configuration coeur: baseline + modulateur VVIX primaire selectionne sur IS.",
            note=f"Alias coeur du candidat {primary_standalone_variant.name}.",
        )
    )

    for mode, description in (
        ("multiplicative", "3-state puis modulation multiplicative par le VVIX."),
        ("cap", "3-state puis cap/downscale par le multiplicateur VVIX."),
    ):
        variants.append(
            _build_variant_run(
                analysis=analysis,
                spec=spec,
                name=f"candidate_combo_{mode}__{primary_standalone_variant.name}",
                category="combo_candidate",
                source_variant_name=primary_standalone_variant.name,
                primary_reference_variant_name="baseline_3state",
                is_core_configuration=False,
                family=primary_standalone_variant.family,
                feature_name=primary_standalone_variant.feature_name,
                combination_mode=mode,
                description=description,
                controls=build_combined_vvix_3state_controls(
                    session_context=session_context,
                    vvix_controls=primary_standalone_variant.controls,
                    sizing_controls=sizing_controls,
                    mode=mode,
                ),
                base_nominal_trades=fixed_nominal_trades,
                all_sessions=common_sessions,
                is_sessions=campaign_is_sessions,
                oos_sessions=campaign_oos_sessions,
                uses_dynamic_sizing=True,
                uses_3state_sizing=True,
                uses_hard_filter_reference=False,
                parameters={
                    "vvix_bucket_multipliers": primary_standalone_variant.parameters.get("bucket_multipliers", {}),
                    "sizing_bucket_multipliers": sizing_spec.bucket_multipliers,
                    "combination_mode": mode,
                },
            )
        )

    for feature_name in spec.sensitivity_feature_names:
        if feature_name not in session_context.columns:
            continue
        sensitivity_calibration = calibrate_quantile_buckets(
            feature_name=feature_name,
            is_values=session_context.loc[session_context["phase"].astype(str) == "is", feature_name],
            bucket_count=spec.bucket_count,
        )
        sensitivity_bucket_labels = apply_bucket_calibration(session_context[feature_name], sensitivity_calibration)
        variants.append(
            _build_variant_run(
                analysis=analysis,
                spec=spec,
                name=f"sensitivity_primary_mapping__{feature_name}",
                category="local_sensitivity",
                source_variant_name=primary_standalone_variant.name,
                primary_reference_variant_name="baseline_nominal",
                is_core_configuration=False,
                family="local_sensitivity",
                feature_name=feature_name,
                combination_mode=None,
                description=f"Sensibilite locale: meme mapping primaire applique a `{feature_name}`.",
                controls=build_vvix_modulation_controls(
                    session_context=session_context,
                    feature_name=feature_name,
                    bucket_labels=sensitivity_bucket_labels,
                    bucket_multipliers=primary_standalone_variant.parameters.get("bucket_multipliers", {}),
                ),
                base_nominal_trades=fixed_nominal_trades,
                all_sessions=common_sessions,
                is_sessions=campaign_is_sessions,
                oos_sessions=campaign_oos_sessions,
                uses_dynamic_sizing=True,
                uses_3state_sizing=False,
                uses_hard_filter_reference=False,
                parameters={
                    "source_variant_name": primary_standalone_variant.name,
                    "bucket_multipliers": primary_standalone_variant.parameters.get("bucket_multipliers", {}),
                    "calibration_scope": "is_only",
                },
            )
        )

    variant_map = {variant.name: variant for variant in variants}
    mid_results = pd.DataFrame(
        [_variant_row(variant, variant_map["baseline_nominal"], variant_map["baseline_3state"], spec) for variant in variants]
    )
    primary_combo_row = _select_primary_combination(mid_results, primary_standalone_variant.name)
    primary_combo_variant = variant_map[str(primary_combo_row["variant_name"])]

    variants.append(
        _clone_variant_run(
            variant=primary_combo_variant,
            name="baseline_3state_vvix_modulator",
            category="core",
            primary_reference_variant_name="baseline_3state",
            description="Configuration coeur: baseline 3-state + modulateur VVIX primaire via le meilleur mode simple choisi sur IS.",
            note=f"Alias coeur du candidat {primary_combo_variant.name}.",
        )
    )

    variant_map = {variant.name: variant for variant in variants}
    results_df = pd.DataFrame(
        [_variant_row(variant, variant_map["baseline_nominal"], variant_map["baseline_3state"], spec) for variant in variants]
    )
    results_df = results_df[[column for column in SUMMARY_COLUMNS if column in results_df.columns]]

    screening_summary = results_df.sort_values(
        ["primary_screening_score", "primary_validation_score", "variant_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    validation_summary = results_df.sort_values(
        ["primary_validation_score", "primary_screening_score", "variant_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    component_summary = build_sizing_component_comparison_summary(variant_map)
    regime_summary = build_vvix_modulation_regime_summary(
        session_context=session_context,
        calibration=primary_calibration,
        vvix_controls=variant_map["baseline_vvix_modulator"].controls,
        combined_controls=variant_map["baseline_3state_vvix_modulator"].controls,
        variant_map={
            "baseline_nominal": variant_map["baseline_nominal"],
            "baseline_3state": variant_map["baseline_3state"],
            "baseline_vvix_modulator": variant_map["baseline_vvix_modulator"],
            "baseline_3state_vvix_modulator": variant_map["baseline_3state_vvix_modulator"],
        },
    )

    full_results_path = output_root / "full_variant_results.csv"
    screening_path = output_root / "screening_summary.csv"
    validation_path = output_root / "validation_summary.csv"
    component_path = output_root / "sizing_component_comparison_summary.csv"
    regime_path = output_root / "vvix_modulation_regime_summary.csv"
    results_df.to_csv(full_results_path, index=False)
    screening_summary.to_csv(screening_path, index=False)
    validation_summary.to_csv(validation_path, index=False)
    component_summary.to_csv(component_path, index=False)
    regime_summary.to_csv(regime_path, index=False)

    for variant in variants:
        _export_variant_artifacts(output_root, variant)

    coverage_summary = {
        "selected_sessions_before_overlay_coverage": int(len(selected_sessions)),
        "sessions_with_vvix_context": int(len(vvix_session_set)),
        "sessions_with_sizing_context": int(len(sizing_session_set)),
        "common_campaign_sessions": int(len(common_sessions)),
        "excluded_selected_sessions_no_vvix_context": int(len(selected_sessions - vvix_session_set)),
        "excluded_selected_sessions_no_sizing_context": int(len(selected_sessions - sizing_session_set)),
        "campaign_is_sessions": int(len(campaign_is_sessions)),
        "campaign_oos_sessions": int(len(campaign_oos_sessions)),
    }

    verdict = _synthesise_verdict(
        spec=spec,
        results_df=validation_summary,
        component_df=component_summary,
        coverage_summary=coverage_summary,
        primary_calibration=primary_calibration,
    )
    verdict_path = output_root / "final_verdict.json"
    _json_dump(verdict_path, verdict)

    report_path = output_root / "final_report.md"
    _write_report(
        output_path=report_path,
        spec=spec,
        analysis=analysis,
        results_df=validation_summary,
        component_df=component_summary,
        regime_summary_df=regime_summary,
        coverage_summary=coverage_summary,
        verdict=verdict,
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
            "vvix_export_root": spec.vvix_export_root or find_latest_export(DEFAULT_VVIX_EXPORT_PREFIX),
            "sizing_export_root": spec.sizing_export_root or find_latest_export(DEFAULT_SIZING_EXPORT_PREFIX),
            "coverage_summary": coverage_summary,
            "spec": asdict(spec),
        },
    )

    return {
        "output_root": output_root,
        "screening_summary": screening_path,
        "validation_summary": validation_path,
        "full_variant_results": full_results_path,
        "sizing_component_comparison_summary": component_path,
        "vvix_modulation_regime_summary": regime_path,
        "final_report": report_path,
        "final_verdict": verdict_path,
        "run_metadata": metadata_path,
    }


def build_default_spec(output_root: Path | None = None) -> MnqOrbVvixSizingModulationSpec:
    return MnqOrbVvixSizingModulationSpec(output_root=output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MNQ ORB VVIX sizing modulation validation campaign.")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--is-fraction", type=float, default=0.70)
    parser.add_argument("--aggregation-rule", type=str, default="majority_50")
    parser.add_argument("--vvix-export-root", type=Path, default=None)
    parser.add_argument("--sizing-export-root", type=Path, default=None)
    parser.add_argument("--commission-per-side-usd", type=float, default=None)
    parser.add_argument("--slippage-ticks", type=float, default=None)
    args = parser.parse_args()

    spec = MnqOrbVvixSizingModulationSpec(
        dataset_path=args.dataset_path,
        output_root=args.output_root,
        is_fraction=float(args.is_fraction),
        aggregation_rule=str(args.aggregation_rule),
        vvix_export_root=args.vvix_export_root,
        sizing_export_root=args.sizing_export_root,
        commission_per_side_usd=args.commission_per_side_usd,
        slippage_ticks=args.slippage_ticks,
    )
    artifacts = run_mnq_orb_vvix_sizing_modulation_campaign(spec)
    print(f"output_root: {artifacts['output_root']}")
    print(f"validation_summary: {artifacts['validation_summary']}")
    print(f"final_report: {artifacts['final_report']}")


if __name__ == "__main__":
    main()
