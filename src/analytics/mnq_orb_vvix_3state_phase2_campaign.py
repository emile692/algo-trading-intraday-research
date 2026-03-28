"""Leak-free phase-2 campaign combining the audited VVIX filter and validated 3-state sizing."""

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

from src.analytics.mnq_orb_regime_filter_sizing_campaign import (
    _scale_nominal_trades_by_multiplier,
    build_regime_dataset,
)
from src.analytics.mnq_orb_vix_vvix_validation_campaign import (
    _build_fixed_nominal_baseline,
    _build_summary_by_scope,
    _daily_results_from_trades,
    _safe_div,
    _scope_value,
    _selected_ensemble_sessions,
    _trade_subset,
    build_vix_vvix_feature_frame,
)
from src.analytics.orb_multi_asset_campaign import (
    BaselineSpec,
    SearchGrid,
    SymbolAnalysis,
    analyze_symbol,
    resolve_processed_dataset,
)
from src.analytics.orb_vvix_overlay import (
    assign_bucket_labels_from_export,
    build_vvix_filter_controls,
)
from src.config.paths import EXPORTS_DIR, ensure_directories
from src.features.implied_volatility import (
    DEFAULT_VIX_DAILY_PATH,
    DEFAULT_VVIX_DAILY_PATH,
    load_vix_vvix_daily_features,
)


DEFAULT_VVIX_EXPORT_PREFIX = "mnq_orb_vix_vvix_validation"
DEFAULT_SIZING_EXPORT_PREFIX = "mnq_orb_regime_filter_sizing"
DEFAULT_VVIX_SURVIVOR_VARIANT = "filter_drop_low__vvix_pct_63_t1"
DEFAULT_LOCAL_VVIX_VARIANTS = ("filter_drop_low__vvix_pct_126_t1",)
DEFAULT_SIZING_VARIANT = "sizing_3state_realized_vol_ratio_15_60"

SUMMARY_COLUMNS = [
    "variant_name",
    "category",
    "description",
    "uses_vvix_filter",
    "vvix_variant_name",
    "vvix_feature_name",
    "vvix_kept_buckets",
    "uses_3state_sizing",
    "sizing_variant_name",
    "sizing_feature_name",
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
    "is_trade_coverage_vs_baseline_nominal",
    "is_day_coverage_vs_baseline_nominal",
    "is_net_pnl_retention_vs_baseline_nominal",
    "is_sharpe_delta_vs_baseline_nominal",
    "is_sortino_delta_vs_baseline_nominal",
    "is_profit_factor_delta_vs_baseline_nominal",
    "is_expectancy_delta_vs_baseline_nominal",
    "is_hit_rate_delta_vs_baseline_nominal",
    "is_stop_hit_rate_delta_vs_baseline_nominal",
    "is_avg_win_delta_vs_baseline_nominal",
    "is_avg_loss_delta_vs_baseline_nominal",
    "is_exposure_delta_vs_baseline_nominal",
    "is_max_drawdown_improvement_vs_baseline_nominal",
    "oos_trade_coverage_vs_baseline_nominal",
    "oos_day_coverage_vs_baseline_nominal",
    "oos_net_pnl_retention_vs_baseline_nominal",
    "oos_sharpe_delta_vs_baseline_nominal",
    "oos_sortino_delta_vs_baseline_nominal",
    "oos_profit_factor_delta_vs_baseline_nominal",
    "oos_expectancy_delta_vs_baseline_nominal",
    "oos_hit_rate_delta_vs_baseline_nominal",
    "oos_stop_hit_rate_delta_vs_baseline_nominal",
    "oos_avg_win_delta_vs_baseline_nominal",
    "oos_avg_loss_delta_vs_baseline_nominal",
    "oos_exposure_delta_vs_baseline_nominal",
    "oos_max_drawdown_improvement_vs_baseline_nominal",
]


@dataclass(frozen=True)
class SizingOverlaySpec:
    export_root: Path
    variant_name: str
    feature_name: str
    bucket_rows: pd.DataFrame
    bucket_multipliers: dict[str, float]


@dataclass
class Phase2VariantRun:
    name: str
    category: str
    description: str
    uses_vvix_filter: bool
    vvix_variant_name: str | None
    vvix_feature_name: str | None
    vvix_kept_buckets: tuple[str, ...]
    uses_3state_sizing: bool
    sizing_variant_name: str | None
    sizing_feature_name: str | None
    parameters: dict[str, Any]
    controls: pd.DataFrame
    trades: pd.DataFrame
    daily_results: pd.DataFrame
    summary_by_scope: pd.DataFrame
    note: str = ""


@dataclass(frozen=True)
class MnqOrbVvix3StatePhase2Spec:
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
    vvix_variant_names: tuple[str, ...] = (DEFAULT_VVIX_SURVIVOR_VARIANT, *DEFAULT_LOCAL_VVIX_VARIANTS)
    sizing_export_root: Path | None = None
    sizing_variant_name: str = DEFAULT_SIZING_VARIANT
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


def find_latest_export(prefix: str, exports_root: Path = EXPORTS_DIR) -> Path:
    candidates = [path for path in exports_root.glob(f"{prefix}_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No export folder found for prefix {prefix!r} under {exports_root}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_3state_overlay_spec(
    export_root: Path | None = None,
    variant_name: str = DEFAULT_SIZING_VARIANT,
) -> SizingOverlaySpec:
    resolved_root = Path(export_root) if export_root is not None else find_latest_export(DEFAULT_SIZING_EXPORT_PREFIX)
    if not resolved_root.exists():
        raise FileNotFoundError(f"Sizing export root not found: {resolved_root}")

    summary = pd.read_csv(resolved_root / "summary_variants.csv")
    mappings = pd.read_csv(resolved_root / "regime_state_mappings.csv")

    variant_rows = summary.loc[summary["variant_name"].astype(str).eq(str(variant_name))].copy()
    if variant_rows.empty:
        raise ValueError(f"Variant {variant_name!r} not found in {resolved_root / 'summary_variants.csv'}.")
    feature_name = str(variant_rows.iloc[0].get("feature_name") or "")
    if not feature_name:
        raise ValueError(f"Could not resolve feature name for sizing variant {variant_name!r}.")

    bucket_rows = (
        mappings.loc[mappings["variant_name"].astype(str).eq(str(variant_name))]
        .sort_values("bucket_position")
        .reset_index(drop=True)
    )
    if bucket_rows.empty:
        raise ValueError(f"Variant {variant_name!r} not found in {resolved_root / 'regime_state_mappings.csv'}.")

    multipliers = {
        str(label): float(multiplier)
        for label, multiplier in zip(
            bucket_rows["bucket_label"].astype(str).tolist(),
            pd.to_numeric(bucket_rows["risk_multiplier"], errors="coerce").tolist(),
        )
    }
    return SizingOverlaySpec(
        export_root=resolved_root,
        variant_name=str(variant_name),
        feature_name=feature_name,
        bucket_rows=bucket_rows,
        bucket_multipliers=multipliers,
    )


def build_reference_3state_controls(
    analysis: SymbolAnalysis,
    selected_sessions: set,
    export_root: Path | None = None,
    variant_name: str = DEFAULT_SIZING_VARIANT,
) -> tuple[SizingOverlaySpec, pd.DataFrame]:
    spec = resolve_3state_overlay_spec(export_root=export_root, variant_name=variant_name)
    regime_df = build_regime_dataset(analysis, selected_sessions)
    if regime_df.empty:
        raise RuntimeError("No regime dataset rows were available for the selected ensemble sessions.")

    controls = (
        regime_df[["session_date", "phase", spec.feature_name]]
        .drop_duplicates(subset=["session_date"])
        .sort_values("session_date")
        .reset_index(drop=True)
    )
    controls["feature_value"] = pd.to_numeric(controls[spec.feature_name], errors="coerce")
    controls["bucket_label"] = assign_bucket_labels_from_export(controls["feature_value"], spec.bucket_rows)
    controls["risk_multiplier"] = controls["bucket_label"].map(spec.bucket_multipliers).fillna(0.0).astype(float)
    controls["selected"] = controls["risk_multiplier"] > 0.0
    controls["skip_trade"] = ~controls["selected"]
    controls["feature_name"] = spec.feature_name
    return spec, controls.sort_values("session_date").reset_index(drop=True)


def compose_phase2_controls(
    session_context: pd.DataFrame,
    vvix_controls: pd.DataFrame | None = None,
    sizing_controls: pd.DataFrame | None = None,
) -> pd.DataFrame:
    controls = session_context.copy()
    controls["selected_by_baseline_atr"] = True

    if vvix_controls is not None:
        vvix_view = vvix_controls.copy()
        vvix_view["session_date"] = pd.to_datetime(vvix_view["session_date"]).dt.date
        candidate_feature_columns = [
            column
            for column in vvix_view.columns
            if column not in {"session_date", "bucket_label", "selected", "skip_trade", "feature_name", "kept_buckets"}
        ]
        feature_column = next((column for column in candidate_feature_columns if column == str(vvix_view["feature_name"].iloc[0])), None)
        if feature_column is not None:
            vvix_view["vvix_feature_value"] = pd.to_numeric(vvix_view[feature_column], errors="coerce")
        else:
            vvix_view["vvix_feature_value"] = np.nan
        vvix_view = vvix_view[
            ["session_date", "feature_name", "vvix_feature_value", "bucket_label", "selected", "kept_buckets"]
        ].rename(
            columns={
                "feature_name": "vvix_feature_name",
                "bucket_label": "vvix_bucket_label",
                "selected": "vvix_selected",
                "kept_buckets": "vvix_kept_buckets",
            }
        )
        controls = controls.merge(vvix_view, on="session_date", how="left", validate="one_to_one")
    else:
        controls["vvix_feature_name"] = pd.NA
        controls["vvix_feature_value"] = np.nan
        controls["vvix_bucket_label"] = pd.Series(pd.NA, index=controls.index, dtype="string")
        controls["vvix_selected"] = True
        controls["vvix_kept_buckets"] = ""

    if sizing_controls is not None:
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
    else:
        controls["sizing_feature_name"] = pd.NA
        controls["sizing_feature_value"] = np.nan
        controls["sizing_bucket_label"] = pd.Series(pd.NA, index=controls.index, dtype="string")
        controls["sizing_risk_multiplier"] = 1.0

    controls["vvix_selected"] = controls["vvix_selected"].fillna(True).astype(bool)
    controls["sizing_risk_multiplier"] = pd.to_numeric(controls["sizing_risk_multiplier"], errors="coerce").fillna(1.0)
    controls["risk_multiplier"] = np.where(
        controls["vvix_selected"],
        controls["sizing_risk_multiplier"],
        0.0,
    )
    controls["selected"] = controls["risk_multiplier"] > 0.0
    controls["skip_trade"] = ~controls["selected"]
    return controls.sort_values("session_date").reset_index(drop=True)


def _score_variant_row(row: dict[str, Any], prefix: str) -> float:
    trade_cov = float(row.get(f"{prefix}_trade_coverage_vs_baseline_nominal", 0.0))
    sharpe_delta = float(row.get(f"{prefix}_sharpe_delta_vs_baseline_nominal", 0.0))
    expectancy_delta = float(row.get(f"{prefix}_expectancy_delta_vs_baseline_nominal", 0.0))
    dd_improvement = float(row.get(f"{prefix}_max_drawdown_improvement_vs_baseline_nominal", 0.0))
    hit_delta = float(row.get(f"{prefix}_hit_rate_delta_vs_baseline_nominal", 0.0))
    stop_delta = float(row.get(f"{prefix}_stop_hit_rate_delta_vs_baseline_nominal", 0.0))
    pnl_retention = float(row.get(f"{prefix}_net_pnl_retention_vs_baseline_nominal", 0.0))
    avg_loss_delta = float(row.get(f"{prefix}_avg_loss_delta_vs_baseline_nominal", 0.0))
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
    if row["variant_name"] == "baseline_nominal":
        return "baseline_reference"
    if float(row.get("is_trade_coverage_vs_baseline_nominal", 0.0)) < 0.20:
        return "too_sparse"
    if float(row.get("screening_score", 0.0)) > 0.35:
        return "selected_for_validation"
    if float(row.get("screening_score", 0.0)) > 0.0:
        return "watchlist"
    return "screen_fail"


def _validation_verdict(row: dict[str, Any], min_oos_trades_for_positive: int) -> str:
    if row["variant_name"] == "baseline_nominal":
        return "baseline_reference"
    oos_trades = int(row.get("oos_n_trades", 0))
    coverage = float(row.get("oos_trade_coverage_vs_baseline_nominal", 0.0))
    pnl_retention = float(row.get("oos_net_pnl_retention_vs_baseline_nominal", 0.0))
    validation_score = float(row.get("validation_score", 0.0))
    dd_improvement = float(row.get("oos_max_drawdown_improvement_vs_baseline_nominal", 0.0))
    expectancy_delta = float(row.get("oos_expectancy_delta_vs_baseline_nominal", 0.0))

    if oos_trades < min_oos_trades_for_positive:
        return "insufficient_oos"
    if validation_score > 0.40 and coverage >= 0.55 and (pnl_retention >= 0.85 or float(row.get("uses_3state_sizing", False))):
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


def _build_variant_run(
    analysis: SymbolAnalysis,
    spec: MnqOrbVvix3StatePhase2Spec,
    name: str,
    category: str,
    description: str,
    controls: pd.DataFrame,
    base_nominal_trades: pd.DataFrame,
    all_sessions: list,
    is_sessions: list,
    oos_sessions: list,
    use_dynamic_sizing: bool,
    vvix_variant_name: str | None = None,
    vvix_feature_name: str | None = None,
    vvix_kept_buckets: tuple[str, ...] = (),
    sizing_variant_name: str | None = None,
    sizing_feature_name: str | None = None,
    parameters: dict[str, Any] | None = None,
    note: str = "",
) -> Phase2VariantRun:
    resolved_commission_per_side = float(
        analysis.instrument_spec["commission_per_side_usd"]
        if spec.commission_per_side_usd is None
        else spec.commission_per_side_usd
    )
    if use_dynamic_sizing:
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
        keep_sessions = set(
            pd.to_datetime(controls.loc[controls["risk_multiplier"] > 0.0, "session_date"], errors="coerce").dt.date
        )
        trades = _trade_subset(base_nominal_trades, list(keep_sessions))

    session_minutes = float(
        (pd.Timestamp(spec.baseline.time_exit) - pd.Timestamp(spec.baseline.opening_time)).total_seconds() / 60.0
    )
    return Phase2VariantRun(
        name=name,
        category=category,
        description=description,
        uses_vvix_filter=bool(vvix_variant_name),
        vvix_variant_name=vvix_variant_name,
        vvix_feature_name=vvix_feature_name,
        vvix_kept_buckets=tuple(vvix_kept_buckets),
        uses_3state_sizing=bool(use_dynamic_sizing),
        sizing_variant_name=sizing_variant_name,
        sizing_feature_name=sizing_feature_name,
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


def _variant_row(variant: Phase2VariantRun, baseline_variant: Phase2VariantRun, spec: MnqOrbVvix3StatePhase2Spec) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant_name": variant.name,
        "category": variant.category,
        "description": variant.description,
        "uses_vvix_filter": variant.uses_vvix_filter,
        "vvix_variant_name": variant.vvix_variant_name,
        "vvix_feature_name": variant.vvix_feature_name,
        "vvix_kept_buckets": ",".join(variant.vvix_kept_buckets),
        "uses_3state_sizing": variant.uses_3state_sizing,
        "sizing_variant_name": variant.sizing_variant_name,
        "sizing_feature_name": variant.sizing_feature_name,
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

    baseline_is_dd = abs(float(_scope_value(baseline_variant.summary_by_scope, "is", "max_drawdown")))
    baseline_oos_dd = abs(float(_scope_value(baseline_variant.summary_by_scope, "oos", "max_drawdown")))
    variant_is_dd = abs(float(_scope_value(variant.summary_by_scope, "is", "max_drawdown")))
    variant_oos_dd = abs(float(_scope_value(variant.summary_by_scope, "oos", "max_drawdown")))

    for scope in ("is", "oos"):
        row[f"{scope}_trade_coverage_vs_baseline_nominal"] = _safe_div(
            float(_scope_value(variant.summary_by_scope, scope, "n_trades")),
            float(_scope_value(baseline_variant.summary_by_scope, scope, "n_trades")),
            default=0.0,
        )
        row[f"{scope}_day_coverage_vs_baseline_nominal"] = _safe_div(
            float(_scope_value(variant.summary_by_scope, scope, "n_days_traded")),
            float(_scope_value(baseline_variant.summary_by_scope, scope, "n_days_traded")),
            default=0.0,
        )
        row[f"{scope}_net_pnl_retention_vs_baseline_nominal"] = _safe_div(
            float(_scope_value(variant.summary_by_scope, scope, "net_pnl")),
            float(_scope_value(baseline_variant.summary_by_scope, scope, "net_pnl")),
            default=0.0,
        )
        for metric in ("sharpe", "sortino", "profit_factor", "expectancy", "hit_rate", "avg_win", "avg_loss", "exposure_time_pct"):
            row[f"{scope}_{metric}_delta_vs_baseline_nominal"] = float(_scope_value(variant.summary_by_scope, scope, metric)) - float(
                _scope_value(baseline_variant.summary_by_scope, scope, metric)
            )
        row[f"{scope}_stop_hit_rate_delta_vs_baseline_nominal"] = float(_scope_value(baseline_variant.summary_by_scope, scope, "stop_hit_rate")) - float(
            _scope_value(variant.summary_by_scope, scope, "stop_hit_rate")
        )

    row["is_max_drawdown_improvement_vs_baseline_nominal"] = _safe_div(
        baseline_is_dd - variant_is_dd,
        max(baseline_is_dd, 1.0),
        default=0.0,
    )
    row["oos_max_drawdown_improvement_vs_baseline_nominal"] = _safe_div(
        baseline_oos_dd - variant_oos_dd,
        max(baseline_oos_dd, 1.0),
        default=0.0,
    )
    row["screening_score"] = _score_variant_row(row, prefix="is")
    row["validation_score"] = _score_variant_row(row, prefix="oos")
    row["screening_status"] = _screening_status(row)
    row["verdict"] = _validation_verdict(row, min_oos_trades_for_positive=spec.min_oos_trades_for_positive)
    return row


def _comparison_row(
    comparison_name: str,
    scope: str,
    left: Phase2VariantRun,
    right: Phase2VariantRun,
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
        "net_pnl_delta": float(_scope_value(left.summary_by_scope, scope, "net_pnl")) - float(
            _scope_value(right.summary_by_scope, scope, "net_pnl")
        ),
        "sharpe_delta": float(_scope_value(left.summary_by_scope, scope, "sharpe")) - float(
            _scope_value(right.summary_by_scope, scope, "sharpe")
        ),
        "sortino_delta": float(_scope_value(left.summary_by_scope, scope, "sortino")) - float(
            _scope_value(right.summary_by_scope, scope, "sortino")
        ),
        "profit_factor_delta": float(_scope_value(left.summary_by_scope, scope, "profit_factor")) - float(
            _scope_value(right.summary_by_scope, scope, "profit_factor")
        ),
        "expectancy_delta": float(_scope_value(left.summary_by_scope, scope, "expectancy")) - float(
            _scope_value(right.summary_by_scope, scope, "expectancy")
        ),
        "max_drawdown_improvement_vs_reference": _safe_div(
            right_dd - left_dd,
            max(right_dd, 1.0),
            default=0.0,
        ),
        "hit_rate_delta": float(_scope_value(left.summary_by_scope, scope, "hit_rate")) - float(
            _scope_value(right.summary_by_scope, scope, "hit_rate")
        ),
        "avg_win_delta": float(_scope_value(left.summary_by_scope, scope, "avg_win")) - float(
            _scope_value(right.summary_by_scope, scope, "avg_win")
        ),
        "avg_loss_delta": float(_scope_value(left.summary_by_scope, scope, "avg_loss")) - float(
            _scope_value(right.summary_by_scope, scope, "avg_loss")
        ),
        "stop_hit_rate_delta": float(_scope_value(right.summary_by_scope, scope, "stop_hit_rate")) - float(
            _scope_value(left.summary_by_scope, scope, "stop_hit_rate")
        ),
    }


def build_component_comparison_summary(variants: dict[str, Phase2VariantRun]) -> pd.DataFrame:
    baseline_nominal = variants["baseline_nominal"]
    baseline_3state = variants["baseline_3state"]
    baseline_vvix_nominal = variants["baseline_vvix_nominal"]
    baseline_vvix_3state = variants["baseline_vvix_3state"]

    rows: list[dict[str, Any]] = []
    pairings = [
        ("marginal_sizing_vs_baseline_nominal", baseline_3state, baseline_nominal),
        ("marginal_vvix_vs_baseline_nominal", baseline_vvix_nominal, baseline_nominal),
        ("combined_vs_baseline_nominal", baseline_vvix_3state, baseline_nominal),
        ("incremental_vvix_on_top_of_3state", baseline_vvix_3state, baseline_3state),
        ("incremental_3state_on_top_of_vvix", baseline_vvix_3state, baseline_vvix_nominal),
        ("vvix_only_vs_3state_only", baseline_vvix_nominal, baseline_3state),
    ]
    for scope in ("overall", "is", "oos"):
        for comparison_name, left, right in pairings:
            rows.append(_comparison_row(comparison_name, scope, left, right))

        combined_vs_baseline = _comparison_row("combined_vs_baseline_nominal", scope, baseline_vvix_3state, baseline_nominal)
        sizing_vs_baseline = _comparison_row("marginal_sizing_vs_baseline_nominal", scope, baseline_3state, baseline_nominal)
        vvix_vs_baseline = _comparison_row("marginal_vvix_vs_baseline_nominal", scope, baseline_vvix_nominal, baseline_nominal)
        rows.append(
            {
                "comparison_type": "interaction",
                "comparison_name": "interaction_excess_vs_additive",
                "scope": scope,
                "variant_name": baseline_vvix_3state.name,
                "reference_variant_name": baseline_nominal.name,
                "combined_net_pnl_delta": combined_vs_baseline["net_pnl_delta"],
                "additive_net_pnl_delta": sizing_vs_baseline["net_pnl_delta"] + vvix_vs_baseline["net_pnl_delta"],
                "interaction_excess_net_pnl_delta": combined_vs_baseline["net_pnl_delta"]
                - (sizing_vs_baseline["net_pnl_delta"] + vvix_vs_baseline["net_pnl_delta"]),
                "combined_sharpe_delta": combined_vs_baseline["sharpe_delta"],
                "additive_sharpe_delta": sizing_vs_baseline["sharpe_delta"] + vvix_vs_baseline["sharpe_delta"],
                "interaction_excess_sharpe_delta": combined_vs_baseline["sharpe_delta"]
                - (sizing_vs_baseline["sharpe_delta"] + vvix_vs_baseline["sharpe_delta"]),
                "combined_profit_factor_delta": combined_vs_baseline["profit_factor_delta"],
                "additive_profit_factor_delta": sizing_vs_baseline["profit_factor_delta"] + vvix_vs_baseline["profit_factor_delta"],
                "interaction_excess_profit_factor_delta": combined_vs_baseline["profit_factor_delta"]
                - (sizing_vs_baseline["profit_factor_delta"] + vvix_vs_baseline["profit_factor_delta"]),
                "combined_expectancy_delta": combined_vs_baseline["expectancy_delta"],
                "additive_expectancy_delta": sizing_vs_baseline["expectancy_delta"] + vvix_vs_baseline["expectancy_delta"],
                "interaction_excess_expectancy_delta": combined_vs_baseline["expectancy_delta"]
                - (sizing_vs_baseline["expectancy_delta"] + vvix_vs_baseline["expectancy_delta"]),
                "combined_max_drawdown_improvement": combined_vs_baseline["max_drawdown_improvement_vs_reference"],
                "additive_max_drawdown_improvement": sizing_vs_baseline["max_drawdown_improvement_vs_reference"]
                + vvix_vs_baseline["max_drawdown_improvement_vs_reference"],
                "interaction_excess_max_drawdown_improvement": combined_vs_baseline["max_drawdown_improvement_vs_reference"]
                - (
                    sizing_vs_baseline["max_drawdown_improvement_vs_reference"]
                    + vvix_vs_baseline["max_drawdown_improvement_vs_reference"]
                ),
            }
        )

    return pd.DataFrame(rows)


def build_regime_impact_summary(
    session_context: pd.DataFrame,
    combined_controls: pd.DataFrame,
    variant_map: dict[str, Phase2VariantRun],
) -> pd.DataFrame:
    regime = session_context[["session_date", "phase"]].copy()
    regime = regime.merge(
        combined_controls[
            [
                "session_date",
                "vvix_bucket_label",
                "vvix_selected",
                "sizing_bucket_label",
                "sizing_risk_multiplier",
                "risk_multiplier",
            ]
        ],
        on="session_date",
        how="left",
        validate="one_to_one",
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
        grouped = grouped.rename(
            columns={
                "net_pnl_usd": f"{variant_name}_net_pnl_usd",
                "n_trades": f"{variant_name}_n_trades",
                "avg_quantity": f"{variant_name}_avg_quantity",
            }
        )
        regime = regime.merge(grouped, on="session_date", how="left")

    fill_values = {
        column: 0.0
        for column in regime.columns
        if column.endswith("_net_pnl_usd") or column.endswith("_avg_quantity")
    }
    fill_values.update({column: 0 for column in regime.columns if column.endswith("_n_trades")})
    regime = regime.fillna(fill_values)

    return (
        regime.groupby(["phase", "vvix_bucket_label", "vvix_selected", "sizing_bucket_label"], dropna=False)
        .agg(
            session_count=("session_date", "count"),
            avg_sizing_multiplier=("sizing_risk_multiplier", "mean"),
            avg_combined_multiplier=("risk_multiplier", "mean"),
            baseline_nominal_net_pnl_usd=("baseline_nominal_net_pnl_usd", "sum"),
            baseline_3state_net_pnl_usd=("baseline_3state_net_pnl_usd", "sum"),
            baseline_vvix_nominal_net_pnl_usd=("baseline_vvix_nominal_net_pnl_usd", "sum"),
            baseline_vvix_3state_net_pnl_usd=("baseline_vvix_3state_net_pnl_usd", "sum"),
            baseline_nominal_n_trades=("baseline_nominal_n_trades", "sum"),
            baseline_3state_n_trades=("baseline_3state_n_trades", "sum"),
            baseline_vvix_nominal_n_trades=("baseline_vvix_nominal_n_trades", "sum"),
            baseline_vvix_3state_n_trades=("baseline_vvix_3state_n_trades", "sum"),
        )
        .reset_index()
        .sort_values(["phase", "vvix_bucket_label", "sizing_bucket_label"])
        .reset_index(drop=True)
    )


def _export_variant_artifacts(root: Path, variant: Phase2VariantRun) -> None:
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


def _primary_mechanism(component_df: pd.DataFrame) -> str:
    oos = component_df.loc[
        (component_df["comparison_type"] == "pairwise") & (component_df["scope"] == "oos")
    ].copy()
    if oos.empty:
        return "none"

    incremental_vvix = oos.loc[oos["comparison_name"] == "incremental_vvix_on_top_of_3state"]
    incremental_sizing = oos.loc[oos["comparison_name"] == "incremental_3state_on_top_of_vvix"]
    if not incremental_vvix.empty:
        row = incremental_vvix.iloc[0]
        if float(row["trade_coverage_vs_reference"]) < 0.90 and float(row["net_pnl_ratio_vs_reference"]) >= 0.95:
            return "better_day_selection"
        if float(row["max_drawdown_improvement_vs_reference"]) > 0.10:
            return "drawdown_reduction"
    if not incremental_sizing.empty:
        row = incremental_sizing.iloc[0]
        if float(row["sharpe_delta"]) > 0.10 or float(row["profit_factor_delta"]) > 0.05:
            return "risk_allocation_control"
    return "mixed_distribution_improvement"


def _synthesise_verdict(
    spec: MnqOrbVvix3StatePhase2Spec,
    results_df: pd.DataFrame,
    component_df: pd.DataFrame,
    coverage_summary: dict[str, Any],
) -> dict[str, Any]:
    baseline_nominal = _top_row(results_df, "baseline_nominal")
    baseline_3state = _top_row(results_df, "baseline_3state")
    baseline_vvix_nominal = _top_row(results_df, "baseline_vvix_nominal")
    baseline_vvix_3state = _top_row(results_df, "baseline_vvix_3state")
    core_results = results_df.loc[results_df["category"].astype(str).eq("core")].copy()
    best_defensive_variant = core_results.sort_values(
        ["validation_score", "oos_max_drawdown_improvement_vs_baseline_nominal"],
        ascending=[False, False],
    ).iloc[0]
    best_tradable_variant = core_results.sort_values(
        ["oos_sharpe", "oos_profit_factor", "oos_net_pnl"],
        ascending=[False, False, False],
    ).iloc[0]

    interaction_oos = component_df.loc[
        (component_df["comparison_type"] == "interaction") & (component_df["scope"] == "oos")
    ]
    interaction_row = interaction_oos.iloc[0] if not interaction_oos.empty else pd.Series(dtype="object")

    combined_vs_3state = component_df.loc[
        (component_df["comparison_type"] == "pairwise")
        & (component_df["comparison_name"] == "incremental_vvix_on_top_of_3state")
        & (component_df["scope"] == "oos")
    ]
    combined_vs_vvix = component_df.loc[
        (component_df["comparison_type"] == "pairwise")
        & (component_df["comparison_name"] == "incremental_3state_on_top_of_vvix")
        & (component_df["scope"] == "oos")
    ]
    combined_vs_3state_row = combined_vs_3state.iloc[0] if not combined_vs_3state.empty else pd.Series(dtype="object")
    combined_vs_vvix_row = combined_vs_vvix.iloc[0] if not combined_vs_vvix.empty else pd.Series(dtype="object")
    combined_beats_3state = bool(
        not combined_vs_3state.empty
        and float(combined_vs_3state_row["sharpe_delta"]) > -0.05
        and (
            float(combined_vs_3state_row["profit_factor_delta"]) > 0.05
            or float(combined_vs_3state_row["max_drawdown_improvement_vs_reference"]) > 0.15
        )
    )
    combined_beats_vvix = bool(
        not combined_vs_vvix.empty
        and float(combined_vs_vvix_row["sharpe_delta"]) > 0.05
        and float(combined_vs_vvix_row["profit_factor_delta"]) > 0.05
    )
    vvix_survivor_holds = bool(
        float(baseline_vvix_nominal["oos_sharpe_delta_vs_baseline_nominal"]) > 0.0
        or float(baseline_vvix_nominal["oos_max_drawdown_improvement_vs_baseline_nominal"]) > 0.10
        or float(combined_vs_3state_row.get("max_drawdown_improvement_vs_reference", 0.0)) > 0.10
    )
    complementary = bool(
        combined_beats_3state
        and interaction_row.get("interaction_excess_sharpe_delta", 0.0) > -0.10
        and interaction_row.get("interaction_excess_profit_factor_delta", 0.0) > -0.05
    )
    promote_combined = bool(
        str(baseline_vvix_3state["verdict"]) == "robust_positive"
        and combined_beats_3state
        and combined_beats_vvix
        and complementary
    )

    return {
        "run_type": "mnq_orb_vvix_3state_phase2_validation",
        "baseline_nominal_variant_name": "baseline_nominal",
        "baseline_3state_variant_name": "baseline_3state",
        "baseline_vvix_nominal_variant_name": "baseline_vvix_nominal",
        "baseline_vvix_3state_variant_name": "baseline_vvix_3state",
        "primary_vvix_variant_name": DEFAULT_VVIX_SURVIVOR_VARIANT,
        "sizing_variant_name": spec.sizing_variant_name,
        "best_overall_variant_name": str(best_tradable_variant["variant_name"]),
        "best_overall_verdict": str(best_tradable_variant["verdict"]),
        "best_defensive_variant_name": str(best_defensive_variant["variant_name"]),
        "best_defensive_verdict": str(best_defensive_variant["verdict"]),
        "vvix_survivor_remains_robust_with_3state": vvix_survivor_holds,
        "combined_beats_3state_alone": combined_beats_3state,
        "combined_beats_vvix_alone": combined_beats_vvix,
        "components_appear_complementary": complementary,
        "promote_combined_as_new_baseline": promote_combined,
        "oos_baseline_nominal_sharpe": float(baseline_nominal["oos_sharpe"]),
        "oos_baseline_3state_sharpe": float(baseline_3state["oos_sharpe"]),
        "oos_baseline_vvix_nominal_sharpe": float(baseline_vvix_nominal["oos_sharpe"]),
        "oos_baseline_vvix_3state_sharpe": float(baseline_vvix_3state["oos_sharpe"]),
        "oos_baseline_vvix_3state_sharpe_delta_vs_nominal": float(
            baseline_vvix_3state["oos_sharpe_delta_vs_baseline_nominal"]
        ),
        "oos_baseline_vvix_3state_profit_factor_delta_vs_nominal": float(
            baseline_vvix_3state["oos_profit_factor_delta_vs_baseline_nominal"]
        ),
        "oos_baseline_vvix_3state_max_drawdown_improvement_vs_nominal": float(
            baseline_vvix_3state["oos_max_drawdown_improvement_vs_baseline_nominal"]
        ),
        "oos_combined_vs_3state_sharpe_delta": float(combined_vs_3state.iloc[0]["sharpe_delta"]) if not combined_vs_3state.empty else 0.0,
        "oos_combined_vs_3state_profit_factor_delta": float(combined_vs_3state.iloc[0]["profit_factor_delta"]) if not combined_vs_3state.empty else 0.0,
        "oos_combined_vs_3state_max_drawdown_improvement": float(combined_vs_3state.iloc[0]["max_drawdown_improvement_vs_reference"])
        if not combined_vs_3state.empty
        else 0.0,
        "oos_combined_vs_vvix_sharpe_delta": float(combined_vs_vvix.iloc[0]["sharpe_delta"]) if not combined_vs_vvix.empty else 0.0,
        "oos_combined_vs_vvix_profit_factor_delta": float(combined_vs_vvix.iloc[0]["profit_factor_delta"]) if not combined_vs_vvix.empty else 0.0,
        "oos_combined_vs_vvix_max_drawdown_improvement": float(combined_vs_vvix.iloc[0]["max_drawdown_improvement_vs_reference"])
        if not combined_vs_vvix.empty
        else 0.0,
        "oos_interaction_excess_sharpe_delta": float(interaction_row.get("interaction_excess_sharpe_delta", 0.0)),
        "oos_interaction_excess_profit_factor_delta": float(interaction_row.get("interaction_excess_profit_factor_delta", 0.0)),
        "oos_interaction_excess_expectancy_delta": float(interaction_row.get("interaction_excess_expectancy_delta", 0.0)),
        "oos_interaction_excess_max_drawdown_improvement": float(
            interaction_row.get("interaction_excess_max_drawdown_improvement", 0.0)
        ),
        "primary_mechanism": _primary_mechanism(component_df),
        "coverage_summary": coverage_summary,
        "assumptions": [
            f"baseline direction={spec.baseline.direction}",
            f"baseline OR window={spec.baseline.or_minutes}m",
            f"aggregation_rule={spec.aggregation_rule}",
            f"fixed_contracts={spec.fixed_contracts}",
            f"primary_vvix_variant={DEFAULT_VVIX_SURVIVOR_VARIANT}",
            f"sizing_variant={spec.sizing_variant_name}",
            "VVIX and sizing overlays are frozen from prior IS-only validation exports.",
            "VVIX daily inputs remain strictly t-1.",
        ],
    }


def _write_report(
    output_path: Path,
    spec: MnqOrbVvix3StatePhase2Spec,
    analysis: SymbolAnalysis,
    coverage_summary: dict[str, Any],
    results_df: pd.DataFrame,
    component_df: pd.DataFrame,
    verdict: dict[str, Any],
) -> None:
    baseline_nominal = _top_row(results_df, "baseline_nominal")
    baseline_3state = _top_row(results_df, "baseline_3state")
    baseline_vvix_nominal = _top_row(results_df, "baseline_vvix_nominal")
    baseline_vvix_3state = _top_row(results_df, "baseline_vvix_3state")

    local_rows = results_df.loc[results_df["variant_name"].astype(str).str.contains("sensitivity", na=False)].copy()
    local_line = "- Aucune variante voisine n'a ete incluse."
    if not local_rows.empty:
        best_local = local_rows.sort_values(["validation_score", "oos_sharpe_delta_vs_baseline_nominal"], ascending=[False, False]).iloc[0]
        local_line = (
            f"- Robustesse locale: `{best_local['variant_name']}` | OOS Sharpe delta vs baseline nominal "
            f"`{float(best_local['oos_sharpe_delta_vs_baseline_nominal']):+.3f}` | "
            f"maxDD improvement `{100.0 * float(best_local['oos_max_drawdown_improvement_vs_baseline_nominal']):.1f}%`."
        )

    combined_vs_3state = component_df.loc[
        (component_df["comparison_type"] == "pairwise")
        & (component_df["comparison_name"] == "incremental_vvix_on_top_of_3state")
        & (component_df["scope"] == "oos")
    ]
    combined_vs_vvix = component_df.loc[
        (component_df["comparison_type"] == "pairwise")
        & (component_df["comparison_name"] == "incremental_3state_on_top_of_vvix")
        & (component_df["scope"] == "oos")
    ]
    interaction_oos = component_df.loc[
        (component_df["comparison_type"] == "interaction")
        & (component_df["comparison_name"] == "interaction_excess_vs_additive")
        & (component_df["scope"] == "oos")
    ]

    combined_vs_3state_line = "- Le gain incremental du VVIX au-dessus du 3-state seul est neutre ou negatif."
    if not combined_vs_3state.empty:
        row = combined_vs_3state.iloc[0]
        combined_vs_3state_line = (
            f"- Incremental VVIX au-dessus du 3-state seul: Sharpe delta `{float(row['sharpe_delta']):+.3f}`, "
            f"PF delta `{float(row['profit_factor_delta']):+.3f}`, "
            f"maxDD improvement `{100.0 * float(row['max_drawdown_improvement_vs_reference']):+.1f}%`, "
            f"trade coverage `{100.0 * float(row['trade_coverage_vs_reference']):.1f}%`."
        )

    combined_vs_vvix_line = "- Le gain incremental du 3-state au-dessus du VVIX seul est neutre ou negatif."
    if not combined_vs_vvix.empty:
        row = combined_vs_vvix.iloc[0]
        combined_vs_vvix_line = (
            f"- Incremental 3-state au-dessus du VVIX seul: Sharpe delta `{float(row['sharpe_delta']):+.3f}`, "
            f"PF delta `{float(row['profit_factor_delta']):+.3f}`, "
            f"maxDD improvement `{100.0 * float(row['max_drawdown_improvement_vs_reference']):+.1f}%`."
        )

    interaction_line = "- Pas d'effet d'interaction mesurable."
    if not interaction_oos.empty:
        row = interaction_oos.iloc[0]
        interaction_line = (
            f"- Interaction combinee vs somme additive: Sharpe excess `{float(row['interaction_excess_sharpe_delta']):+.3f}`, "
            f"PF excess `{float(row['interaction_excess_profit_factor_delta']):+.3f}`, "
            f"expectancy excess `{float(row['interaction_excess_expectancy_delta']):+.2f}`, "
            f"maxDD excess `{100.0 * float(row['interaction_excess_max_drawdown_improvement']):+.1f}%`."
        )

    strongest_core_line = (
        f"- Candidat tradable le plus fort sur cette phase 2: `{verdict['best_overall_variant_name']}` "
        f"(OOS Sharpe `{float(_top_row(results_df, verdict['best_overall_variant_name'])['oos_sharpe']):.3f}`)."
    )
    defensive_core_line = (
        f"- Variante defensive la plus convaincante: `{verdict['best_defensive_variant_name']}` "
        f"(OOS maxDD `{float(_top_row(results_df, verdict['best_defensive_variant_name'])['oos_max_drawdown']):.2f}`)."
    )

    lines = [
        "# MNQ ORB VVIX + 3-State Phase 2 Validation",
        "",
        "## Baseline And Integration",
        "",
        f"- Baseline ORB conservee: OR{int(spec.baseline.or_minutes)} / direction `{spec.baseline.direction}` / RR `{float(spec.baseline.target_multiple):.2f}` / VWAP confirmation `{bool(spec.baseline.vwap_confirmation)}`.",
        f"- Filtre ATR structurel conserve via l'ensemble `{spec.aggregation_rule}`; aucune modification du signal ORB.",
        f"- Survivor VVIX principal fige: `{DEFAULT_VVIX_SURVIVOR_VARIANT}`.",
        f"- Overlay sizing 3-state fige: `{spec.sizing_variant_name}`.",
        f"- Dataset: `{analysis.dataset_path.name}` | sessions IS/OOS d'origine: `{len(analysis.is_sessions)}` / `{len(analysis.oos_sessions)}`.",
        "",
        "## Coverage",
        "",
        f"- Sessions selectionnees par la baseline ATR: `{int(coverage_summary['selected_sessions_before_overlay_coverage'])}`.",
        f"- Sessions avec contexte VVIX t-1 exploitable: `{int(coverage_summary['sessions_with_vvix_context'])}`.",
        f"- Sessions avec bucket 3-state exploitable: `{int(coverage_summary['sessions_with_sizing_context'])}`.",
        f"- Univers commun teste pour les 4 variantes coeur: `{int(coverage_summary['common_campaign_sessions'])}` sessions.",
        f"- IS/OOS dans l'univers commun: `{int(coverage_summary['campaign_is_sessions'])}` / `{int(coverage_summary['campaign_oos_sessions'])}`.",
        "",
        "## Four Core Configurations",
        "",
        f"- `baseline_nominal`: OOS pnl `{float(baseline_nominal['oos_net_pnl']):.2f}` | Sharpe `{float(baseline_nominal['oos_sharpe']):.3f}` | PF `{float(baseline_nominal['oos_profit_factor']):.3f}` | maxDD `{float(baseline_nominal['oos_max_drawdown']):.2f}`.",
        f"- `baseline_3state`: OOS pnl `{float(baseline_3state['oos_net_pnl']):.2f}` | Sharpe `{float(baseline_3state['oos_sharpe']):.3f}` | PF `{float(baseline_3state['oos_profit_factor']):.3f}` | maxDD `{float(baseline_3state['oos_max_drawdown']):.2f}`.",
        f"- `baseline_vvix_nominal`: OOS pnl `{float(baseline_vvix_nominal['oos_net_pnl']):.2f}` | Sharpe `{float(baseline_vvix_nominal['oos_sharpe']):.3f}` | PF `{float(baseline_vvix_nominal['oos_profit_factor']):.3f}` | maxDD `{float(baseline_vvix_nominal['oos_max_drawdown']):.2f}`.",
        f"- `baseline_vvix_3state`: OOS pnl `{float(baseline_vvix_3state['oos_net_pnl']):.2f}` | Sharpe `{float(baseline_vvix_3state['oos_sharpe']):.3f}` | PF `{float(baseline_vvix_3state['oos_profit_factor']):.3f}` | maxDD `{float(baseline_vvix_3state['oos_max_drawdown']):.2f}`.",
        "",
        "## Attribution",
        "",
        strongest_core_line,
        defensive_core_line,
        combined_vs_3state_line,
        combined_vs_vvix_line,
        interaction_line,
        local_line,
        "",
        "## Direct Answers",
        "",
        f"- Le filtre VVIX survivor reste-t-il robuste une fois combine au 3-state ? {'Oui.' if verdict['vvix_survivor_remains_robust_with_3state'] else 'Non, son avantage marginal devient ambigu.'}",
        f"- La combinaison VVIX + 3-state est-elle meilleure que baseline + 3-state seul ? {'Oui.' if verdict['combined_beats_3state_alone'] else 'Non: le 3-state seul garde le meilleur moteur de perf, la combinaison n apportant surtout qu un profil plus defensif.'}",
        f"- La combinaison VVIX + 3-state est-elle meilleure que baseline + VVIX seul ? {'Oui, surtout via le bloc d allocation.' if verdict['combined_beats_vvix_alone'] else 'Non, pas de facon assez nette.'}",
        f"- Les deux blocs semblent-ils complementaires ou redondants ? {'Plutot complementaires.' if verdict['components_appear_complementary'] else 'Plutot redondants / additifs sans vraie synergie.'}",
        f"- Le gain eventuel vient-il surtout de la selection des jours, du controle du risque, ou d'une vraie amelioration du moteur ? `{verdict['primary_mechanism']}`.",
        f"- Y a-t-il un candidat suffisamment propre pour devenir la nouvelle baseline tradable ? {'Oui, la combinaison merite promotion.' if verdict['promote_combined_as_new_baseline'] else 'Pas encore: le 3-state seul reste la reference tradable la plus solide, la combinaison restant plutot une variante defensive comparee.'}",
        "",
        "## Methodology Notes",
        "",
        "- Le signal ORB, le filtre ATR structurel, les regles d'execution et les couts du repo restent inchanges dans cette campagne.",
        "- Le survivor VVIX et le 3-state sont appliques comme overlays geles issus de campagnes precedentes; aucun redemarrage d'une grille massive VVIX ou sizing.",
        "- Les inputs VVIX restent strictement lagges t-1; aucune information future n'est utilisee.",
        "- La comparaison des 4 variantes coeur se fait sur le meme univers commun de sessions pour garder l'attribution propre.",
        "",
        "## Exports",
        "",
        "- `screening_summary.csv`",
        "- `validation_summary.csv`",
        "- `full_variant_results.csv`",
        "- `component_comparison_summary.csv`",
        "- `regime_impact_summary.csv`",
        "- `final_report.md`",
        "- `final_verdict.json`",
        "- `variants/<variant>/...`",
    ]
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_mnq_orb_vvix_3state_phase2_campaign(spec: MnqOrbVvix3StatePhase2Spec) -> dict[str, Any]:
    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(spec.output_root)
        if spec.output_root is not None
        else EXPORTS_DIR / f"mnq_orb_vvix_3state_phase2_{timestamp}"
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

    session_context = (
        feature_frame[["session_date", "phase", "breakout_side", "breakout_timing_bucket"]]
        .drop_duplicates(subset=["session_date"])
        .sort_values("session_date")
        .reset_index(drop=True)
    )
    campaign_is_sessions = [
        session
        for session in pd.to_datetime(pd.Index(analysis.is_sessions)).date
        if session in common_session_set
    ]
    campaign_oos_sessions = [
        session
        for session in pd.to_datetime(pd.Index(analysis.oos_sessions)).date
        if session in common_session_set
    ]

    vvix_control_specs: dict[str, tuple[Any, pd.DataFrame]] = {}
    for variant_name in spec.vvix_variant_names:
        vvix_spec, vvix_controls = build_vvix_filter_controls(
            session_dates=common_sessions,
            export_root=spec.vvix_export_root,
            variant_name=variant_name,
            vix_path=spec.vix_daily_path,
            vvix_path=spec.vvix_daily_path,
        )
        vvix_controls["session_date"] = pd.to_datetime(vvix_controls["session_date"]).dt.date
        vvix_control_specs[variant_name] = (
            vvix_spec,
            vvix_controls.loc[vvix_controls["session_date"].isin(common_session_set)].copy().reset_index(drop=True),
        )

    base_controls = compose_phase2_controls(session_context=session_context)
    sizing_only_controls = compose_phase2_controls(session_context=session_context, sizing_controls=sizing_controls)

    primary_vvix_spec, primary_vvix_controls = vvix_control_specs[DEFAULT_VVIX_SURVIVOR_VARIANT]
    primary_vvix_nominal_controls = compose_phase2_controls(
        session_context=session_context,
        vvix_controls=primary_vvix_controls,
    )
    primary_combined_controls = compose_phase2_controls(
        session_context=session_context,
        vvix_controls=primary_vvix_controls,
        sizing_controls=sizing_controls,
    )

    variants: list[Phase2VariantRun] = [
        _build_variant_run(
            analysis=analysis,
            spec=spec,
            name="baseline_nominal",
            category="core",
            description="ORB baseline + ATR structurel + nominal fixe sur l'univers commun VVIX/3-state.",
            controls=base_controls,
            base_nominal_trades=fixed_nominal_trades,
            all_sessions=common_sessions,
            is_sessions=campaign_is_sessions,
            oos_sessions=campaign_oos_sessions,
            use_dynamic_sizing=False,
            parameters={},
        ),
        _build_variant_run(
            analysis=analysis,
            spec=spec,
            name="baseline_3state",
            category="core",
            description="ORB baseline + ATR structurel + overlay sizing 3-state fige.",
            controls=sizing_only_controls,
            base_nominal_trades=fixed_nominal_trades,
            all_sessions=common_sessions,
            is_sessions=campaign_is_sessions,
            oos_sessions=campaign_oos_sessions,
            use_dynamic_sizing=True,
            sizing_variant_name=sizing_spec.variant_name,
            sizing_feature_name=sizing_spec.feature_name,
            parameters={"bucket_multipliers": sizing_spec.bucket_multipliers},
        ),
        _build_variant_run(
            analysis=analysis,
            spec=spec,
            name="baseline_vvix_nominal",
            category="core",
            description="ORB baseline + ATR structurel + survivor VVIX principal + nominal fixe.",
            controls=primary_vvix_nominal_controls,
            base_nominal_trades=fixed_nominal_trades,
            all_sessions=common_sessions,
            is_sessions=campaign_is_sessions,
            oos_sessions=campaign_oos_sessions,
            use_dynamic_sizing=False,
            vvix_variant_name=primary_vvix_spec.variant_name,
            vvix_feature_name=primary_vvix_spec.feature_name,
            vvix_kept_buckets=primary_vvix_spec.kept_buckets,
            parameters={"kept_buckets": primary_vvix_spec.kept_buckets},
        ),
        _build_variant_run(
            analysis=analysis,
            spec=spec,
            name="baseline_vvix_3state",
            category="core",
            description="ORB baseline + ATR structurel + survivor VVIX principal + sizing 3-state.",
            controls=primary_combined_controls,
            base_nominal_trades=fixed_nominal_trades,
            all_sessions=common_sessions,
            is_sessions=campaign_is_sessions,
            oos_sessions=campaign_oos_sessions,
            use_dynamic_sizing=True,
            vvix_variant_name=primary_vvix_spec.variant_name,
            vvix_feature_name=primary_vvix_spec.feature_name,
            vvix_kept_buckets=primary_vvix_spec.kept_buckets,
            sizing_variant_name=sizing_spec.variant_name,
            sizing_feature_name=sizing_spec.feature_name,
            parameters={
                "kept_buckets": primary_vvix_spec.kept_buckets,
                "bucket_multipliers": sizing_spec.bucket_multipliers,
            },
        ),
    ]

    for variant_name in spec.vvix_variant_names:
        if variant_name == DEFAULT_VVIX_SURVIVOR_VARIANT:
            continue
        vvix_spec, vvix_controls = vvix_control_specs[variant_name]
        sensitivity_nominal_controls = compose_phase2_controls(
            session_context=session_context,
            vvix_controls=vvix_controls,
        )
        sensitivity_combined_controls = compose_phase2_controls(
            session_context=session_context,
            vvix_controls=vvix_controls,
            sizing_controls=sizing_controls,
        )
        short_name = vvix_spec.feature_name.replace("_t1", "")
        variants.append(
            _build_variant_run(
                analysis=analysis,
                spec=spec,
                name=f"sensitivity_{short_name}_nominal",
                category="local_sensitivity",
                description=f"Variante voisine du survivor VVIX: `{variant_name}` en nominal fixe.",
                controls=sensitivity_nominal_controls,
                base_nominal_trades=fixed_nominal_trades,
                all_sessions=common_sessions,
                is_sessions=campaign_is_sessions,
                oos_sessions=campaign_oos_sessions,
                use_dynamic_sizing=False,
                vvix_variant_name=vvix_spec.variant_name,
                vvix_feature_name=vvix_spec.feature_name,
                vvix_kept_buckets=vvix_spec.kept_buckets,
                parameters={"kept_buckets": vvix_spec.kept_buckets},
            )
        )
        variants.append(
            _build_variant_run(
                analysis=analysis,
                spec=spec,
                name=f"sensitivity_{short_name}_3state",
                category="local_sensitivity",
                description=f"Variante voisine du survivor VVIX: `{variant_name}` combinee au 3-state.",
                controls=sensitivity_combined_controls,
                base_nominal_trades=fixed_nominal_trades,
                all_sessions=common_sessions,
                is_sessions=campaign_is_sessions,
                oos_sessions=campaign_oos_sessions,
                use_dynamic_sizing=True,
                vvix_variant_name=vvix_spec.variant_name,
                vvix_feature_name=vvix_spec.feature_name,
                vvix_kept_buckets=vvix_spec.kept_buckets,
                sizing_variant_name=sizing_spec.variant_name,
                sizing_feature_name=sizing_spec.feature_name,
                parameters={
                    "kept_buckets": vvix_spec.kept_buckets,
                    "bucket_multipliers": sizing_spec.bucket_multipliers,
                },
            )
        )

    variant_map = {variant.name: variant for variant in variants}
    results_df = pd.DataFrame([_variant_row(variant, variant_map["baseline_nominal"], spec) for variant in variants])
    results_df = results_df[[column for column in SUMMARY_COLUMNS if column in results_df.columns]]

    screening_summary = results_df.sort_values(["screening_score", "is_sharpe_delta_vs_baseline_nominal"], ascending=[False, False]).reset_index(drop=True)
    validation_summary = results_df.sort_values(["validation_score", "oos_sharpe_delta_vs_baseline_nominal"], ascending=[False, False]).reset_index(drop=True)
    component_summary = build_component_comparison_summary(variant_map)
    regime_impact_summary = build_regime_impact_summary(
        session_context,
        primary_combined_controls,
        {
            "baseline_nominal": variant_map["baseline_nominal"],
            "baseline_3state": variant_map["baseline_3state"],
            "baseline_vvix_nominal": variant_map["baseline_vvix_nominal"],
            "baseline_vvix_3state": variant_map["baseline_vvix_3state"],
        },
    )

    full_results_path = output_root / "full_variant_results.csv"
    screening_path = output_root / "screening_summary.csv"
    validation_path = output_root / "validation_summary.csv"
    component_path = output_root / "component_comparison_summary.csv"
    regime_impact_path = output_root / "regime_impact_summary.csv"
    results_df.to_csv(full_results_path, index=False)
    screening_summary.to_csv(screening_path, index=False)
    validation_summary.to_csv(validation_path, index=False)
    component_summary.to_csv(component_path, index=False)
    regime_impact_summary.to_csv(regime_impact_path, index=False)

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
        "vvix_primary_trade_coverage_oos_vs_baseline_nominal": float(_top_row(results_df, "baseline_vvix_nominal")["oos_trade_coverage_vs_baseline_nominal"]),
        "vvix_primary_day_coverage_oos_vs_baseline_nominal": float(_top_row(results_df, "baseline_vvix_nominal")["oos_day_coverage_vs_baseline_nominal"]),
    }

    verdict = _synthesise_verdict(spec=spec, results_df=validation_summary, component_df=component_summary, coverage_summary=coverage_summary)
    verdict_path = output_root / "final_verdict.json"
    _json_dump(verdict_path, verdict)

    report_path = output_root / "final_report.md"
    _write_report(
        output_path=report_path,
        spec=spec,
        analysis=analysis,
        coverage_summary=coverage_summary,
        results_df=validation_summary,
        component_df=component_summary,
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
        "full_variant_results": full_results_path,
        "screening_summary": screening_path,
        "validation_summary": validation_path,
        "component_comparison_summary": component_path,
        "regime_impact_summary": regime_impact_path,
        "final_report": report_path,
        "final_verdict": verdict_path,
        "run_metadata": metadata_path,
    }


def build_default_spec(output_root: Path | None = None) -> MnqOrbVvix3StatePhase2Spec:
    return MnqOrbVvix3StatePhase2Spec(output_root=output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MNQ ORB VVIX + 3-state phase-2 validation campaign.")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--is-fraction", type=float, default=0.70)
    parser.add_argument("--aggregation-rule", type=str, default="majority_50")
    parser.add_argument("--vvix-export-root", type=Path, default=None)
    parser.add_argument("--sizing-export-root", type=Path, default=None)
    parser.add_argument("--commission-per-side-usd", type=float, default=None)
    parser.add_argument("--slippage-ticks", type=float, default=None)
    args = parser.parse_args()

    spec = MnqOrbVvix3StatePhase2Spec(
        dataset_path=args.dataset_path,
        output_root=args.output_root,
        is_fraction=float(args.is_fraction),
        aggregation_rule=str(args.aggregation_rule),
        vvix_export_root=args.vvix_export_root,
        sizing_export_root=args.sizing_export_root,
        commission_per_side_usd=args.commission_per_side_usd,
        slippage_ticks=args.slippage_ticks,
    )
    artifacts = run_mnq_orb_vvix_3state_phase2_campaign(spec)
    print(f"output_root: {artifacts['output_root']}")
    print(f"validation_summary: {artifacts['validation_summary']}")
    print(f"final_report: {artifacts['final_report']}")


if __name__ == "__main__":
    main()
