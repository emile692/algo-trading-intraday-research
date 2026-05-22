"""Strict train-only regime-gated validation for the M2K/MGC 1H pullback portfolio."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.analytics.volume_climax_pullback_common import latest_path_for_symbol, load_symbol_data, safe_float
from src.analytics.volume_climax_pullback_intrabar_recalibration_campaign import (
    IntrabarRecalibrationConfig,
    _compute_trade_metrics,
    _daily_returns,
    _file_metadata,
    _markdown_table,
    _simulate_config,
)
from src.analytics.volume_climax_pullback_multiasset_multitimeframe_campaign import (
    _clone_variant_for_timeframe,
    _period_mask,
    _resolve_seed_variant,
    resample_rth_timeframe,
    timeframe_to_minutes,
)
from src.analytics.volume_climax_pullback_survivor_audit import (
    SurvivorAuditConfig,
    build_argument_parser as _unused_survivor_parser,
    compute_extended_metrics,
    derive_survivor_verdict,
)
from src.config.settings import DEFAULT_TIMEZONE
from src.data.session import extract_rth
from src.engine.vwap_backtester import build_execution_model_for_profile
from src.strategy.volume_climax_pullback_v2 import (
    VolumeClimaxPullbackV2Variant,
    build_volume_climax_pullback_v2_signal_frame,
    prepare_volume_climax_pullback_v2_features,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ("M2K", "MGC", "MNQ")
DEFAULT_SIGNAL_TIMEFRAME = "1H"
DEFAULT_EXECUTION_TIMEFRAME = "1min"
DEFAULT_OUTPUT_ROOT = Path("export")


@dataclass(frozen=True)
class RegimeRuleSpec:
    rule_id: str
    family: str
    params: dict[str, Any]
    allocation_scheme: str


def latest_survivor_audit_dir(output_root: Path) -> Path:
    candidates = sorted(
        [path for path in Path(output_root).glob("volume_climax_pullback_survivor_audit_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No survivor audit export found. Run volume_climax_pullback_survivor_audit first or provide --survivor-audit-dir.")
    return candidates[0]


def parse_survivor_config_row(row: pd.Series) -> SurvivorAuditConfig:
    payload = json.loads(str(row.get("filter_params_json", "{}"))) if pd.notna(row.get("filter_params_json")) else {}
    return SurvivorAuditConfig(
        config_id=str(row["config_id"]),
        symbol=str(row["symbol"]),
        signal_timeframe=str(row["signal_timeframe"]),
        execution_timeframe=str(row["execution_timeframe"]),
        base_signal_variant=str(row["base_signal_variant"]),
        family=str(row["family"]),
        cluster_id=str(row["cluster_id"]),
        stop_multiplier=float(row["stop_multiplier"]),
        target_multiplier=float(row["target_multiplier"]),
        entry_delay_minutes=int(safe_float(row["entry_delay_minutes"], 0.0)),
        time_stop_bars=int(safe_float(row["variant_time_stop_bars"], 0.0)),
        filter_name=str(row["filter_name"]),
        filter_params=dict(payload),
    )


def load_survivor_selected_configs(
    *,
    survivor_audit_dir: Path,
    symbols: Sequence[str],
) -> tuple[dict[str, dict[str, SurvivorAuditConfig]], pd.DataFrame, pd.DataFrame]:
    selection_catalog = pd.read_csv(survivor_audit_dir / "config_selection_by_fold.csv")
    config_catalog = pd.read_csv(survivor_audit_dir / "local_parameter_stability.csv")
    strict_fold_breakdown = pd.read_csv(survivor_audit_dir / "strict_wfa_fold_breakdown.csv")
    strict_wfa_summary = pd.read_csv(survivor_audit_dir / "strict_wfa_summary.csv")

    selection_catalog = selection_catalog.loc[selection_catalog["symbol"].isin(list(symbols))].copy()
    config_catalog = config_catalog.loc[config_catalog["symbol"].isin(list(symbols))].copy()
    strict_fold_breakdown = strict_fold_breakdown.loc[strict_fold_breakdown["symbol"].isin(list(symbols))].copy()
    strict_wfa_summary = strict_wfa_summary.loc[strict_wfa_summary["symbol"].isin(list(symbols))].copy()
    if config_catalog.empty or strict_fold_breakdown.empty:
        raise ValueError("Survivor audit exports do not contain the requested symbols.")

    unique_rows = (
        config_catalog.sort_values(["symbol", "config_id"])
        .drop_duplicates(subset=["config_id"], keep="first")
        .reset_index(drop=True)
    )
    config_map = {str(row["config_id"]): parse_survivor_config_row(row) for _, row in unique_rows.iterrows()}
    selected_by_fold: dict[str, dict[str, SurvivorAuditConfig]] = {}
    for _, row in strict_fold_breakdown.iterrows():
        symbol = str(row["symbol"])
        selected_by_fold.setdefault(symbol, {})
        config_id = str(row["selected_config_id"])
        selected_by_fold[symbol][str(row["fold_id"])] = config_map[config_id]
    return selected_by_fold, strict_fold_breakdown, strict_wfa_summary


def clone_variant_with_time_stop(base_variant: VolumeClimaxPullbackV2Variant, timeframe: str, time_stop_bars: int) -> VolumeClimaxPullbackV2Variant:
    variant = _clone_variant_for_timeframe(base_variant, timeframe)
    payload = asdict(variant)
    payload["time_stop_bars"] = int(time_stop_bars)
    return VolumeClimaxPullbackV2Variant(**payload)


def prepare_symbol_context(
    *,
    symbol: str,
    signal_timeframe: str,
    execution_timeframe: str,
    selected_configs: Sequence[SurvivorAuditConfig],
    raw_minute_df_override: pd.DataFrame | None = None,
) -> dict[str, Any]:
    raw_path = latest_path_for_symbol(symbol)
    raw_minute_df = raw_minute_df_override.copy() if raw_minute_df_override is not None else load_symbol_data(symbol, input_paths={symbol: raw_path})
    raw_minute_df["timestamp"] = pd.to_datetime(raw_minute_df["timestamp"], errors="coerce")
    minute_df = extract_rth(raw_minute_df.copy())
    minute_df["timestamp"] = pd.to_datetime(minute_df["timestamp"], errors="coerce")
    minute_df["session_date"] = minute_df["timestamp"].dt.date
    bars = resample_rth_timeframe(raw_minute_df, signal_timeframe)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")
    bars["session_date"] = bars["timestamp"].dt.date
    base_variant, _, _ = _resolve_seed_variant(symbol)
    signal_variant = _clone_variant_for_timeframe(base_variant, signal_timeframe)
    features = prepare_volume_climax_pullback_v2_features(bars)
    signal_df = build_volume_climax_pullback_v2_signal_frame(features, signal_variant)
    feature_cols = [
        "timestamp",
        "session_date",
        "close",
        "ema20",
        "ema50",
        "ema20_slope_3_atr",
        "ema50_slope_3_atr",
        "atr_20",
        "atr_percentile_100",
        "atr_ratio_5_20",
        "session_vwap",
    ]
    feature_frame = features.loc[:, [column for column in feature_cols if column in features.columns]].copy()
    feature_frame = feature_frame.rename(columns={"timestamp": "signal_time", "session_date": "signal_session_date"})
    execution_model, instrument = build_execution_model_for_profile(symbol=symbol, profile_name="repo_realistic")
    signal_bar_minutes = timeframe_to_minutes(signal_timeframe)

    events_by_config: dict[str, pd.DataFrame] = {}
    for config in selected_configs:
        variant = clone_variant_with_time_stop(base_variant, signal_timeframe, config.time_stop_bars)
        events = _simulate_config(
            config=config.to_intrabar_config(),
            signal_df=signal_df,
            minute_df=minute_df,
            variant=variant,
            execution_model=execution_model,
            point_value_usd=float(instrument.point_value_usd),
            tick_size=float(instrument.tick_size),
            signal_bar_minutes=signal_bar_minutes,
        )
        events["symbol"] = symbol
        events["signal_timeframe"] = signal_timeframe
        events["config_id"] = config.config_id
        events = events.merge(feature_frame, on="signal_time", how="left")
        if {"close", "ema50", "atr_20"}.issubset(events.columns):
            events["distance_from_ema50_atr"] = (
                (pd.to_numeric(events["close"], errors="coerce") - pd.to_numeric(events["ema50"], errors="coerce"))
                / pd.to_numeric(events["atr_20"], errors="coerce").replace(0.0, np.nan)
            )
        if {"close", "session_vwap", "atr_20"}.issubset(events.columns):
            events["distance_from_vwap_atr"] = (
                (pd.to_numeric(events["close"], errors="coerce") - pd.to_numeric(events["session_vwap"], errors="coerce"))
                / pd.to_numeric(events["atr_20"], errors="coerce").replace(0.0, np.nan)
            )
        events_by_config[config.config_id] = events

    data_audit = {
        "symbol": symbol,
        "signal_timeframe": signal_timeframe,
        "execution_timeframe": execution_timeframe,
        "source_path": str(raw_path),
        "first_timestamp": str(raw_minute_df["timestamp"].min()) if not raw_minute_df.empty else None,
        "last_timestamp": str(raw_minute_df["timestamp"].max()) if not raw_minute_df.empty else None,
        "rows_1m": int(len(raw_minute_df)),
        "rows_rth": int(len(minute_df)),
        "rows_signal": int(len(signal_df)),
        "timezone": DEFAULT_TIMEZONE,
        "estimated_cost_per_trade": float(execution_model.round_trip_fees(quantity=1)),
        "variant_name": signal_variant.name,
    }
    return {
        "symbol": symbol,
        "minute_df": minute_df,
        "features": features,
        "signal_df": signal_df,
        "events_by_config": events_by_config,
        "estimated_cost_per_trade": float(execution_model.round_trip_fees(quantity=1)),
        "data_audit": data_audit,
    }


def build_regime_rule_universe(max_regime_rules: int) -> list[RegimeRuleSpec]:
    rules: list[RegimeRuleSpec] = []

    def _add(rule_id: str, family: str, params: dict[str, Any], allocation_scheme: str) -> None:
        rules.append(RegimeRuleSpec(rule_id=rule_id, family=family, params=params, allocation_scheme=allocation_scheme))

    allocation_schemes = ("conditional_equal_weight", "conditional_inverse_vol", "conditional_capped_mgc_40")
    for scheme in allocation_schemes:
        _add(f"always_on__{scheme}", "always_on", {}, scheme)
        for quantile in (0.60, 0.70):
            _add(f"atr_pct_above_q{int(quantile*100)}__{scheme}", "atr_pct_above", {"quantile": quantile}, scheme)
        for quantile in (0.30, 0.40):
            _add(f"atr_pct_below_q{int(quantile*100)}__{scheme}", "atr_pct_below", {"quantile": quantile}, scheme)
        _add("atr_pct_mid_q20_q80__" + scheme, "atr_pct_between", {"low_quantile": 0.20, "high_quantile": 0.80}, scheme)
        _add("atr_pct_mid_q30_q70__" + scheme, "atr_pct_between", {"low_quantile": 0.30, "high_quantile": 0.70}, scheme)
        _add("atr_ratio_above_q60__" + scheme, "atr_ratio_above", {"quantile": 0.60}, scheme)
        _add("atr_ratio_mid_q20_q80__" + scheme, "atr_ratio_between", {"low_quantile": 0.20, "high_quantile": 0.80}, scheme)
        _add("trend_align_ema50__" + scheme, "trend_align_ema50", {}, scheme)
        _add("trend_align_ema20__" + scheme, "trend_align_ema20", {}, scheme)
        _add("slope_dir_q60__" + scheme, "slope_dir_above", {"feature": "ema50_slope_3_atr", "quantile": 0.60}, scheme)
        _add("slope_dir_q75__" + scheme, "slope_dir_above", {"feature": "ema50_slope_3_atr", "quantile": 0.75}, scheme)
        _add("vwap_momentum_q55__" + scheme, "vwap_momentum", {"feature": "distance_from_vwap_atr", "quantile": 0.55}, scheme)
    deduped = {rule.rule_id: rule for rule in rules}
    ordered = sorted(deduped.values(), key=lambda rule: (rule.family, rule.allocation_scheme, json.dumps(rule.params, sort_keys=True), rule.rule_id))
    return ordered[: max(1, int(max_regime_rules))]


def fit_rule_on_train(rule: RegimeRuleSpec, mgc_train_events: pd.DataFrame) -> dict[str, Any]:
    params = dict(rule.params)
    frame = mgc_train_events.copy()
    if rule.family in {"always_on", "trend_align_ema50", "trend_align_ema20"}:
        return {"rule_id": rule.rule_id, "family": rule.family, "allocation_scheme": rule.allocation_scheme, "params": params}
    if rule.family in {"atr_pct_above", "atr_pct_below"}:
        series = pd.to_numeric(frame["atr_percentile_100"], errors="coerce").dropna()
        params["threshold"] = float(series.quantile(float(params["quantile"]))) if not series.empty else np.nan
    elif rule.family == "atr_pct_between":
        series = pd.to_numeric(frame["atr_percentile_100"], errors="coerce").dropna()
        params["low_threshold"] = float(series.quantile(float(params["low_quantile"]))) if not series.empty else np.nan
        params["high_threshold"] = float(series.quantile(float(params["high_quantile"]))) if not series.empty else np.nan
    elif rule.family == "atr_ratio_above":
        series = pd.to_numeric(frame["atr_ratio_5_20"], errors="coerce").dropna()
        params["threshold"] = float(series.quantile(float(params["quantile"]))) if not series.empty else np.nan
    elif rule.family == "atr_ratio_between":
        series = pd.to_numeric(frame["atr_ratio_5_20"], errors="coerce").dropna()
        params["low_threshold"] = float(series.quantile(float(params["low_quantile"]))) if not series.empty else np.nan
        params["high_threshold"] = float(series.quantile(float(params["high_quantile"]))) if not series.empty else np.nan
    elif rule.family == "slope_dir_above":
        feature = str(params["feature"])
        series = pd.to_numeric(frame[feature], errors="coerce").abs().dropna()
        params["threshold"] = float(series.quantile(float(params["quantile"]))) if not series.empty else np.nan
    elif rule.family == "vwap_momentum":
        feature = str(params["feature"])
        momentum = (pd.to_numeric(frame["direction"], errors="coerce").fillna(0.0) * pd.to_numeric(frame[feature], errors="coerce")).dropna()
        params["threshold"] = float(momentum.quantile(float(params["quantile"]))) if not momentum.empty else np.nan
    return {"rule_id": rule.rule_id, "family": rule.family, "allocation_scheme": rule.allocation_scheme, "params": params}


def apply_fitted_rule(events: pd.DataFrame, fitted_rule: dict[str, Any]) -> pd.DataFrame:
    out = events.copy()
    direction_num = out["direction"].map({"long": 1, "short": -1}).fillna(pd.to_numeric(out["direction"], errors="coerce")).fillna(0).astype(int)
    out["regime_rule_id"] = str(fitted_rule["rule_id"])
    out["regime_family"] = str(fitted_rule["family"])
    out["allocation_scheme"] = str(fitted_rule["allocation_scheme"])
    out["regime_active"] = True
    out["regime_reason"] = "active"
    family = str(fitted_rule["family"])
    params = dict(fitted_rule["params"])
    if family == "always_on":
        return out
    if family == "atr_pct_above":
        threshold = safe_float(params.get("threshold"), np.nan)
        mask = pd.to_numeric(out["atr_percentile_100"], errors="coerce") >= threshold
    elif family == "atr_pct_below":
        threshold = safe_float(params.get("threshold"), np.nan)
        mask = pd.to_numeric(out["atr_percentile_100"], errors="coerce") <= threshold
    elif family == "atr_pct_between":
        low = safe_float(params.get("low_threshold"), np.nan)
        high = safe_float(params.get("high_threshold"), np.nan)
        values = pd.to_numeric(out["atr_percentile_100"], errors="coerce")
        mask = values.between(low, high)
    elif family == "atr_ratio_above":
        threshold = safe_float(params.get("threshold"), np.nan)
        mask = pd.to_numeric(out["atr_ratio_5_20"], errors="coerce") >= threshold
    elif family == "atr_ratio_between":
        low = safe_float(params.get("low_threshold"), np.nan)
        high = safe_float(params.get("high_threshold"), np.nan)
        values = pd.to_numeric(out["atr_ratio_5_20"], errors="coerce")
        mask = values.between(low, high)
    elif family == "trend_align_ema50":
        close = pd.to_numeric(out["close"], errors="coerce")
        ema = pd.to_numeric(out["ema50"], errors="coerce")
        slope = pd.to_numeric(out["ema50_slope_3_atr"], errors="coerce")
        mask = ((direction_num == 1) & (close >= ema) & (slope >= 0)) | ((direction_num == -1) & (close <= ema) & (slope <= 0))
    elif family == "trend_align_ema20":
        close = pd.to_numeric(out["close"], errors="coerce")
        ema = pd.to_numeric(out["ema20"], errors="coerce")
        slope = pd.to_numeric(out["ema20_slope_3_atr"], errors="coerce")
        mask = ((direction_num == 1) & (close >= ema) & (slope >= 0)) | ((direction_num == -1) & (close <= ema) & (slope <= 0))
    elif family == "slope_dir_above":
        feature = str(params["feature"])
        threshold = safe_float(params.get("threshold"), np.nan)
        aligned = direction_num * pd.to_numeric(out[feature], errors="coerce")
        mask = aligned >= threshold
    elif family == "vwap_momentum":
        feature = str(params["feature"])
        threshold = safe_float(params.get("threshold"), np.nan)
        aligned = direction_num * pd.to_numeric(out[feature], errors="coerce")
        mask = aligned >= threshold
    else:
        raise ValueError(f"Unsupported rule family {family!r}")
    out["regime_active"] = mask.fillna(False)
    out.loc[~out["regime_active"], "regime_reason"] = f"inactive_{family}"
    return out


def _trade_concentration(events: pd.DataFrame, *, pnl_column: str) -> dict[str, Any]:
    executed = events.loc[events.get("executed", False)].copy()
    pnl = pd.to_numeric(executed.get(pnl_column), errors="coerce").dropna().sort_values(ascending=False)
    total = float(pnl.sum()) if not pnl.empty else 0.0
    worst = pnl.sort_values(ascending=True)
    def _share(series: pd.Series, n: int) -> float:
        if total == 0 or series.empty:
            return np.nan
        return float(series.head(n).sum() / total)
    return {
        "trade_count": int(len(pnl)),
        "total_pnl": total,
        "top1_contribution_pct": _share(pnl, 1),
        "top3_contribution_pct": _share(pnl, 3),
        "top5_contribution_pct": _share(pnl, 5),
        "worst1_contribution_pct": _share(worst, 1),
        "worst3_contribution_pct": _share(worst, 3),
        "worst5_contribution_pct": _share(worst, 5),
    }


def build_conditional_portfolio_events(
    *,
    m2k_events: pd.DataFrame,
    mgc_events: pd.DataFrame,
    allocation_scheme: str,
    train_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    base_cols = list(set(m2k_events.columns).union(mgc_events.columns))
    parts: list[pd.DataFrame] = []
    m2k = m2k_events.copy()
    mgc = mgc_events.copy()
    m2k["asset"] = "M2K"
    mgc["asset"] = "MGC"
    for frame in (m2k, mgc):
        frame["executed"] = frame["executed"].fillna(False).astype(bool)
        frame["session_date"] = pd.to_datetime(frame["session_date"], errors="coerce").dt.date
    all_dates = sorted(set(m2k["session_date"].dropna()).union(set(mgc["session_date"].dropna())))
    train_weights = dict(train_weights or {})
    static_inverse = {
        "M2K": float(train_weights.get("M2K", 0.5)),
        "MGC": float(train_weights.get("MGC", 0.5)),
    }
    for session_date in all_dates:
        m2k_day = m2k.loc[m2k["session_date"] == session_date].copy()
        mgc_day = mgc.loc[mgc["session_date"] == session_date].copy()
        mgc_active = bool(mgc_day.loc[mgc_day["executed"], :].shape[0] > 0)
        if allocation_scheme == "static_equal_weight":
            weights = {"M2K": 0.5, "MGC": 0.5}
        elif allocation_scheme == "conditional_equal_weight":
            weights = {"M2K": 0.5, "MGC": 0.5} if mgc_active else {"M2K": 1.0, "MGC": 0.0}
        elif allocation_scheme == "conditional_inverse_vol":
            weights = static_inverse if mgc_active else {"M2K": 1.0, "MGC": 0.0}
        elif allocation_scheme == "conditional_capped_mgc_40":
            weights = {"M2K": 0.6, "MGC": 0.4} if mgc_active else {"M2K": 1.0, "MGC": 0.0}
        elif allocation_scheme == "m2k_only":
            weights = {"M2K": 1.0, "MGC": 0.0}
        elif allocation_scheme == "mgc_only":
            weights = {"M2K": 0.0, "MGC": 1.0}
        else:
            raise ValueError(f"Unsupported allocation scheme {allocation_scheme!r}")
        for asset_frame, asset in ((m2k_day, "M2K"), (mgc_day, "MGC")):
            if asset_frame.empty:
                continue
            asset_frame["portfolio_weight"] = float(weights.get(asset, 0.0))
            for column in ("net_pnl_usd", "gross_pnl_usd", "pnl"):
                if column in asset_frame.columns:
                    asset_frame[column] = pd.to_numeric(asset_frame[column], errors="coerce") * float(weights.get(asset, 0.0))
            asset_frame["portfolio_name"] = allocation_scheme
            asset_frame["mgc_active_day"] = mgc_active
            parts.append(asset_frame)
    if not parts:
        return pd.DataFrame(columns=base_cols + ["asset", "portfolio_weight", "portfolio_name", "mgc_active_day"])
    return pd.concat(parts, ignore_index=True)


def build_train_inverse_vol_weights(m2k_train: pd.DataFrame, mgc_train: pd.DataFrame) -> dict[str, float]:
    vols: dict[str, float] = {}
    for symbol, frame in (("M2K", m2k_train), ("MGC", mgc_train)):
        daily = _daily_returns(frame)
        series = pd.to_numeric(daily.get("daily_pnl"), errors="coerce")
        vol = float(series.std(ddof=0)) if len(series) > 1 and series.std(ddof=0) > 0 else np.nan
        vols[symbol] = 1.0 / vol if np.isfinite(vol) and vol > 0 else 0.0
    total = sum(vols.values())
    if total <= 0:
        return {"M2K": 0.5, "MGC": 0.5}
    return {symbol: value / total for symbol, value in vols.items()}


def compute_portfolio_metrics(portfolio_events: pd.DataFrame, *, estimated_cost_per_trade: float) -> dict[str, Any]:
    metrics = compute_extended_metrics(portfolio_events, estimated_cost_per_trade=estimated_cost_per_trade)
    daily = _daily_returns(portfolio_events)
    metrics["monthly_hit_rate"] = float((daily.assign(month=pd.to_datetime(daily["session_date"], errors="coerce").dt.to_period("M")).groupby("month")["daily_pnl"].sum() > 0).mean()) if not daily.empty else 0.0
    metrics["active_months"] = int(daily.assign(month=pd.to_datetime(daily["session_date"], errors="coerce").dt.to_period("M")).groupby("month")["daily_pnl"].sum().shape[0]) if not daily.empty else 0
    metrics.update(_trade_concentration(portfolio_events, pnl_column="net_pnl_usd"))
    return metrics


def score_train_rule_result(metrics: dict[str, Any], *, mgc_retention_rate: float, mgc_active_months: int) -> float:
    score = 0.0
    score += max(0.0, safe_float(metrics["net_pnl"], 0.0)) / 1000.0
    score += min(safe_float(metrics["profit_factor"], 0.0), 3.0) * 0.35
    score += max(0.0, safe_float(metrics.get("pnl_to_maxdd"), 0.0)) * 0.20 if pd.notna(metrics.get("pnl_to_maxdd")) else 0.0
    score += safe_float(metrics.get("monthly_hit_rate"), 0.0) * 0.25
    score += min(mgc_retention_rate, 0.60) * 0.20
    if safe_float(metrics.get("top1_contribution_pct"), 0.0) > 0.50:
        score -= 0.25
    if safe_float(metrics.get("top3_contribution_pct"), 0.0) > 0.90:
        score -= 0.15
    if safe_float(metrics["trades"], 0.0) < 12:
        score -= 0.35
    if mgc_retention_rate < 0.10:
        score -= 0.25
    if mgc_active_months < 4:
        score -= 0.15
    if safe_float(metrics["profit_factor"], 0.0) < 1.05:
        score -= 0.20
    if safe_float(metrics["avg_trade"], 0.0) <= safe_float(metrics["estimated_cost_per_trade"], 0.0) * 1.5:
        score -= 0.20
    return float(score)


def select_best_train_rule(train_ranking: pd.DataFrame) -> pd.Series:
    ranked = train_ranking.sort_values(
        ["train_score", "train_net_pnl", "train_profit_factor", "mgc_retention_rate_train"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def build_regime_rule_train_ranking(
    *,
    fold_id: str,
    m2k_train_events: pd.DataFrame,
    mgc_train_events: pd.DataFrame,
    rules: Sequence[RegimeRuleSpec],
    estimated_cost_per_trade: float,
) -> pd.DataFrame:
    mgc_base_executed = int(mgc_train_events.get("executed", False).fillna(False).astype(bool).sum())
    rows: list[dict[str, Any]] = []
    for rule in rules:
        fitted = fit_rule_on_train(rule, mgc_train_events)
        gated_mgc = apply_fitted_rule(mgc_train_events, fitted)
        gated_mgc = gated_mgc.loc[gated_mgc["regime_active"].fillna(False)].copy()
        train_weights = build_train_inverse_vol_weights(m2k_train_events, gated_mgc)
        portfolio_events = build_conditional_portfolio_events(
            m2k_events=m2k_train_events,
            mgc_events=gated_mgc,
            allocation_scheme=rule.allocation_scheme,
            train_weights=train_weights,
        )
        metrics = compute_portfolio_metrics(portfolio_events, estimated_cost_per_trade=estimated_cost_per_trade)
        mgc_retained = int(gated_mgc.get("executed", False).fillna(False).astype(bool).sum())
        mgc_retention_rate = float(mgc_retained / mgc_base_executed) if mgc_base_executed > 0 else 0.0
        mgc_daily = _daily_returns(gated_mgc)
        mgc_active_months = int(mgc_daily.assign(month=pd.to_datetime(mgc_daily["session_date"], errors="coerce").dt.to_period("M")).groupby("month")["daily_pnl"].sum().shape[0]) if not mgc_daily.empty else 0
        train_score = score_train_rule_result(metrics, mgc_retention_rate=mgc_retention_rate, mgc_active_months=mgc_active_months)
        rows.append(
            {
                "fold_id": fold_id,
                "rule_id": rule.rule_id,
                "family": rule.family,
                "allocation_scheme": rule.allocation_scheme,
                "fitted_params_json": json.dumps(fitted["params"], sort_keys=True),
                "train_score": train_score,
                "train_net_pnl": float(metrics["net_pnl"]),
                "train_profit_factor": float(metrics["profit_factor"]),
                "train_trades": int(metrics["trades"]),
                "train_max_drawdown": float(metrics["max_drawdown"]),
                "train_monthly_hit_rate": float(metrics["monthly_hit_rate"]),
                "train_top1_contribution_pct": float(metrics.get("top1_contribution_pct")) if pd.notna(metrics.get("top1_contribution_pct")) else np.nan,
                "train_top3_contribution_pct": float(metrics.get("top3_contribution_pct")) if pd.notna(metrics.get("top3_contribution_pct")) else np.nan,
                "mgc_retained_trades_train": mgc_retained,
                "mgc_retention_rate_train": mgc_retention_rate,
                "mgc_active_months_train": mgc_active_months,
                "selected_in_fold": False,
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["train_score", "train_net_pnl", "train_profit_factor", "mgc_retention_rate_train"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    if not ranking.empty:
        ranking.loc[0, "selected_in_fold"] = True
    return ranking


def compute_fold_baseline_portfolio(
    *,
    name: str,
    m2k_events: pd.DataFrame,
    mgc_events: pd.DataFrame,
    estimated_cost_per_trade: float,
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    portfolio_events = build_conditional_portfolio_events(
        m2k_events=m2k_events,
        mgc_events=mgc_events,
        allocation_scheme=name,
        train_weights=weights,
    )
    metrics = compute_portfolio_metrics(portfolio_events, estimated_cost_per_trade=estimated_cost_per_trade)
    return portfolio_events, metrics


def build_portfolio_daily_frame(entity_name: str, events: pd.DataFrame) -> pd.DataFrame:
    daily = _daily_returns(events)
    if daily.empty:
        return pd.DataFrame(columns=["entity_name", "session_date", "daily_pnl", "equity", "drawdown"])
    out = daily.copy()
    out["entity_name"] = entity_name
    return out[["entity_name", "session_date", "daily_pnl", "equity", "drawdown"]]


def build_monthly_yearly_frames(entity_name: str, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = _daily_returns(events)
    if daily.empty:
        return (
            pd.DataFrame(columns=["entity_name", "month", "pnl"]),
            pd.DataFrame(columns=["entity_name", "year", "pnl"]),
        )
    month = (
        daily.assign(month=pd.to_datetime(daily["session_date"], errors="coerce").dt.to_period("M").astype(str))
        .groupby("month", as_index=False)["daily_pnl"]
        .sum()
        .rename(columns={"daily_pnl": "pnl"})
    )
    month["entity_name"] = entity_name
    year = (
        daily.assign(year=pd.to_datetime(daily["session_date"], errors="coerce").dt.year)
        .groupby("year", as_index=False)["daily_pnl"]
        .sum()
        .rename(columns={"daily_pnl": "pnl"})
    )
    year["entity_name"] = entity_name
    return month[["entity_name", "month", "pnl"]], year[["entity_name", "year", "pnl"]]


def build_diagnostic_posthoc_rows(strict_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if strict_summary.empty:
        return pd.DataFrame(columns=["result_type", "name", "deployable", "reason", "net_pnl", "profit_factor", "positive_folds"])
    positive = strict_summary.loc[
        (pd.to_numeric(strict_summary["net_pnl"], errors="coerce") > 0)
        & (strict_summary["entity_name"] != "strict_best_regime_gated")
    ].copy()
    for _, row in positive.iterrows():
        rows.append(
            {
                "result_type": "posthoc_positive_baseline",
                "name": str(row["entity_name"]),
                "deployable": False,
                "reason": "baseline_or_posthoc_reference_only",
                "net_pnl": float(row["net_pnl"]),
                "profit_factor": float(row["profit_factor"]),
                "positive_folds": int(row["positive_folds"]),
            }
        )
    return pd.DataFrame(rows)


def strict_regime_portfolio_verdict(
    *,
    portfolio_metrics: dict[str, Any],
    positive_folds: int,
    m2k_only_metrics: dict[str, Any],
    raw_m2k_mgc_metrics: dict[str, Any],
    top1_contribution_pct: float,
    top3_contribution_pct: float,
) -> str:
    if safe_float(portfolio_metrics["net_pnl"], 0.0) <= 0 or safe_float(portfolio_metrics["profit_factor"], 0.0) < 1.10 or positive_folds < 3:
        return "reject"
    if (
        safe_float(portfolio_metrics["net_pnl"], 0.0) > 0
        and safe_float(portfolio_metrics["profit_factor"], 0.0) > 1.15
        and positive_folds >= 3
    ):
        verdict = "weak_watchlist"
    else:
        verdict = "reject"
    if (
        safe_float(portfolio_metrics["net_pnl"], 0.0) > safe_float(m2k_only_metrics["net_pnl"], 0.0)
        and safe_float(portfolio_metrics["net_pnl"], 0.0) > safe_float(raw_m2k_mgc_metrics["net_pnl"], 0.0)
        and safe_float(portfolio_metrics["profit_factor"], 0.0) > 1.20
        and positive_folds >= 4
        and safe_float(portfolio_metrics.get("monthly_hit_rate"), 0.0) >= 0.50
        and safe_float(portfolio_metrics["max_drawdown"], 0.0) >= -400.0
    ):
        verdict = "watchlist"
    if (
        verdict == "watchlist"
        and safe_float(portfolio_metrics["profit_factor"], 0.0) > 1.30
        and positive_folds >= 4
        and safe_float(portfolio_metrics.get("monthly_hit_rate"), 0.0) > safe_float(raw_m2k_mgc_metrics.get("monthly_hit_rate"), 0.0)
        and safe_float(top1_contribution_pct, 1.0) < 0.40
        and safe_float(top3_contribution_pct, 1.0) < 0.75
        and safe_float(portfolio_metrics["max_drawdown"], 0.0) >= -250.0
    ):
        verdict = "candidate"
    return verdict


def evaluate_regime_gated_portfolio(
    *,
    symbols: Sequence[str],
    signal_timeframe: str,
    execution_timeframe: str,
    survivor_audit_dir: Path,
    max_regime_rules: int,
    include_negative_control: bool,
    dataset_overrides: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    requested_symbols = [symbol for symbol in symbols if include_negative_control or symbol != "MNQ"]
    selected_by_fold, survivor_fold_breakdown, survivor_summary = load_survivor_selected_configs(
        survivor_audit_dir=survivor_audit_dir,
        symbols=requested_symbols,
    )
    contexts: dict[str, dict[str, Any]] = {}
    for symbol in requested_symbols:
        selected_configs = list({config.config_id: config for config in selected_by_fold.get(symbol, {}).values()}.values())
        override = None if dataset_overrides is None else dataset_overrides.get(symbol)
        contexts[symbol] = prepare_symbol_context(
            symbol=symbol,
            signal_timeframe=signal_timeframe,
            execution_timeframe=execution_timeframe,
            selected_configs=selected_configs,
            raw_minute_df_override=override,
        )

    rules = build_regime_rule_universe(max_regime_rules=max_regime_rules)
    folds = sorted(set(survivor_fold_breakdown["fold_id"].astype(str)))
    estimated_cost = float(np.mean([contexts[symbol]["estimated_cost_per_trade"] for symbol in contexts]))

    ranking_rows: list[pd.DataFrame] = []
    selected_rule_rows: list[dict[str, Any]] = []
    strict_fold_rows: list[dict[str, Any]] = []
    strict_parts: list[pd.DataFrame] = []
    baseline_m2k_parts: list[pd.DataFrame] = []
    baseline_mgc_parts: list[pd.DataFrame] = []
    baseline_raw_parts: list[pd.DataFrame] = []
    retention_rows: list[dict[str, Any]] = []

    for fold_id in folds:
        m2k_config = selected_by_fold["M2K"][fold_id]
        mgc_config = selected_by_fold["MGC"][fold_id]
        m2k_events_all = contexts["M2K"]["events_by_config"][m2k_config.config_id].copy()
        mgc_events_all = contexts["MGC"]["events_by_config"][mgc_config.config_id].copy()

        fold_row = survivor_fold_breakdown.loc[(survivor_fold_breakdown["symbol"] == "M2K") & (survivor_fold_breakdown["fold_id"] == fold_id)].iloc[0]
        train_start = None
        train_end = None
        test_start = None
        test_end = None
        # reconstruct fold boundaries from stitched test windows using survivor audit preferred folds
        # derive from known fold ids
        preferred = {
            "fold_1": (date(2020, 1, 1), date(2021, 12, 31), date(2022, 1, 1), date(2022, 12, 31)),
            "fold_2": (date(2020, 1, 1), date(2022, 12, 31), date(2023, 1, 1), date(2023, 12, 31)),
            "fold_3": (date(2020, 1, 1), date(2023, 12, 31), date(2024, 1, 1), date(2024, 12, 31)),
            "fold_4": (date(2020, 1, 1), date(2024, 12, 31), date(2025, 1, 1), date(2025, 12, 31)),
            "fold_5": (date(2020, 1, 1), date(2025, 12, 31), date(2026, 1, 1), date(2026, 12, 31)),
        }
        if fold_id not in preferred:
            raise ValueError(f"Unsupported fold id {fold_id!r} in regime-gated portfolio.")
        train_start, train_end, test_start, test_end = preferred[fold_id]

        m2k_train = m2k_events_all.loc[_period_mask(m2k_events_all, train_start, train_end)].copy()
        m2k_test = m2k_events_all.loc[_period_mask(m2k_events_all, test_start, test_end)].copy()
        mgc_train = mgc_events_all.loc[_period_mask(mgc_events_all, train_start, train_end)].copy()
        mgc_test = mgc_events_all.loc[_period_mask(mgc_events_all, test_start, test_end)].copy()

        ranking = build_regime_rule_train_ranking(
            fold_id=fold_id,
            m2k_train_events=m2k_train,
            mgc_train_events=mgc_train,
            rules=rules,
            estimated_cost_per_trade=estimated_cost,
        )
        ranking_rows.append(ranking)
        winner = select_best_train_rule(ranking)
        selected_rule_rows.append(winner.to_dict())

        selected_rule = next(rule for rule in rules if rule.rule_id == str(winner["rule_id"]))
        fitted_train = fit_rule_on_train(selected_rule, mgc_train)
        gated_mgc_train = apply_fitted_rule(mgc_train, fitted_train)
        gated_mgc_train = gated_mgc_train.loc[gated_mgc_train["regime_active"].fillna(False)].copy()
        gated_mgc_test = apply_fitted_rule(mgc_test, fitted_train)
        gated_mgc_test = gated_mgc_test.loc[gated_mgc_test["regime_active"].fillna(False)].copy()
        train_weights = build_train_inverse_vol_weights(m2k_train, gated_mgc_train)

        strict_test = build_conditional_portfolio_events(
            m2k_events=m2k_test,
            mgc_events=gated_mgc_test,
            allocation_scheme=str(selected_rule.allocation_scheme),
            train_weights=train_weights,
        )
        strict_test["fold_id"] = fold_id
        strict_test["selected_rule_id"] = str(selected_rule.rule_id)
        strict_parts.append(strict_test)

        baseline_m2k, m2k_metrics = compute_fold_baseline_portfolio(
            name="m2k_only",
            m2k_events=m2k_test,
            mgc_events=mgc_test.iloc[0:0].copy(),
            estimated_cost_per_trade=estimated_cost,
        )
        baseline_m2k["fold_id"] = fold_id
        baseline_m2k_parts.append(baseline_m2k)

        baseline_mgc, mgc_metrics = compute_fold_baseline_portfolio(
            name="mgc_only",
            m2k_events=m2k_test.iloc[0:0].copy(),
            mgc_events=mgc_test,
            estimated_cost_per_trade=estimated_cost,
        )
        baseline_mgc["fold_id"] = fold_id
        baseline_mgc_parts.append(baseline_mgc)

        baseline_raw, raw_metrics = compute_fold_baseline_portfolio(
            name="static_equal_weight",
            m2k_events=m2k_test,
            mgc_events=mgc_test,
            estimated_cost_per_trade=estimated_cost,
        )
        baseline_raw["fold_id"] = fold_id
        baseline_raw_parts.append(baseline_raw)

        strict_metrics = compute_portfolio_metrics(strict_test, estimated_cost_per_trade=estimated_cost)
        mgc_base_trades = int(mgc_test.get("executed", False).fillna(False).astype(bool).sum())
        mgc_retained_trades = int(gated_mgc_test.get("executed", False).fillna(False).astype(bool).sum())
        retention_rows.append(
            {
                "fold_id": fold_id,
                "selected_rule_id": str(selected_rule.rule_id),
                "mgc_train_trades_base": int(mgc_train.get("executed", False).fillna(False).astype(bool).sum()),
                "mgc_train_trades_retained": int(gated_mgc_train.get("executed", False).fillna(False).astype(bool).sum()),
                "mgc_train_retention_rate": float(
                    int(gated_mgc_train.get("executed", False).fillna(False).astype(bool).sum())
                    / max(int(mgc_train.get("executed", False).fillna(False).astype(bool).sum()), 1)
                ),
                "mgc_test_trades_base": mgc_base_trades,
                "mgc_test_trades_retained": mgc_retained_trades,
                "mgc_test_retention_rate": float(mgc_retained_trades / max(mgc_base_trades, 1)),
                "mgc_test_pnl_base": float(pd.to_numeric(mgc_test.loc[mgc_test.get("executed", False).fillna(False).astype(bool), "net_pnl_usd"], errors="coerce").sum()),
                "mgc_test_pnl_retained": float(pd.to_numeric(gated_mgc_test.loc[gated_mgc_test.get("executed", False).fillna(False).astype(bool), "net_pnl_usd"], errors="coerce").sum()),
                "allocation_scheme": str(selected_rule.allocation_scheme),
            }
        )
        strict_fold_rows.append(
            {
                "fold_id": fold_id,
                "selected_rule_id": str(selected_rule.rule_id),
                "regime_family": str(selected_rule.family),
                "allocation_scheme": str(selected_rule.allocation_scheme),
                "fitted_params_json": json.dumps(fitted_train["params"], sort_keys=True),
                "train_score": float(winner["train_score"]),
                "test_net_pnl": float(strict_metrics["net_pnl"]),
                "test_profit_factor": float(strict_metrics["profit_factor"]),
                "test_trades": int(strict_metrics["trades"]),
                "test_win_rate": float(strict_metrics["winrate"]),
                "test_avg_trade": float(strict_metrics["avg_trade"]),
                "test_median_trade": float(strict_metrics["median_trade"]),
                "test_max_drawdown": float(strict_metrics["max_drawdown"]),
                "test_max_daily_drawdown": float(strict_metrics["max_daily_drawdown"]),
                "test_monthly_hit_rate": float(strict_metrics["monthly_hit_rate"]),
                "test_top1_contribution_pct": float(strict_metrics["top1_contribution_pct"]) if pd.notna(strict_metrics["top1_contribution_pct"]) else np.nan,
                "test_top3_contribution_pct": float(strict_metrics["top3_contribution_pct"]) if pd.notna(strict_metrics["top3_contribution_pct"]) else np.nan,
                "m2k_only_test_net_pnl": float(m2k_metrics["net_pnl"]),
                "mgc_only_test_net_pnl": float(mgc_metrics["net_pnl"]),
                "raw_m2k_mgc_test_net_pnl": float(raw_metrics["net_pnl"]),
                "mgc_test_retention_rate": float(mgc_retained_trades / max(mgc_base_trades, 1)),
                "strict_train_only": True,
            }
        )

    regime_rule_train_ranking = pd.concat(ranking_rows, ignore_index=True) if ranking_rows else pd.DataFrame()
    selected_regime_rule_by_fold = pd.DataFrame(selected_rule_rows)
    strict_regime_wfa_fold_breakdown = pd.DataFrame(strict_fold_rows)
    strict_events = pd.concat(strict_parts, ignore_index=True) if strict_parts else pd.DataFrame()
    m2k_only_events = pd.concat(baseline_m2k_parts, ignore_index=True) if baseline_m2k_parts else pd.DataFrame()
    mgc_only_events = pd.concat(baseline_mgc_parts, ignore_index=True) if baseline_mgc_parts else pd.DataFrame()
    raw_equal_events = pd.concat(baseline_raw_parts, ignore_index=True) if baseline_raw_parts else pd.DataFrame()
    retention_summary = pd.DataFrame(retention_rows)

    strict_metrics = compute_portfolio_metrics(strict_events, estimated_cost_per_trade=estimated_cost)
    m2k_only_metrics = compute_portfolio_metrics(m2k_only_events, estimated_cost_per_trade=estimated_cost)
    mgc_only_metrics = compute_portfolio_metrics(mgc_only_events, estimated_cost_per_trade=estimated_cost)
    raw_equal_metrics = compute_portfolio_metrics(raw_equal_events, estimated_cost_per_trade=estimated_cost)

    strict_positive_folds = int((pd.to_numeric(strict_regime_wfa_fold_breakdown["test_net_pnl"], errors="coerce") > 0).sum()) if not strict_regime_wfa_fold_breakdown.empty else 0
    strict_verdict = strict_regime_portfolio_verdict(
        portfolio_metrics=strict_metrics,
        positive_folds=strict_positive_folds,
        m2k_only_metrics=m2k_only_metrics,
        raw_m2k_mgc_metrics=raw_equal_metrics,
        top1_contribution_pct=safe_float(strict_metrics.get("top1_contribution_pct"), np.nan),
        top3_contribution_pct=safe_float(strict_metrics.get("top3_contribution_pct"), np.nan),
    )

    summary_rows = [
        {
            "entity_name": "m2k_only_baseline",
            "selection_basis": "strict_train_only",
            "deployable": False,
            "net_pnl": float(m2k_only_metrics["net_pnl"]),
            "profit_factor": float(m2k_only_metrics["profit_factor"]),
            "trades": int(m2k_only_metrics["trades"]),
            "positive_folds": int((pd.to_numeric(strict_regime_wfa_fold_breakdown["m2k_only_test_net_pnl"], errors="coerce") > 0).sum()) if not strict_regime_wfa_fold_breakdown.empty else 0,
            "max_drawdown": float(m2k_only_metrics["max_drawdown"]),
            "max_daily_drawdown": float(m2k_only_metrics["max_daily_drawdown"]),
            "win_rate": float(m2k_only_metrics["winrate"]),
            "avg_trade": float(m2k_only_metrics["avg_trade"]),
            "median_trade": float(m2k_only_metrics["median_trade"]),
            "monthly_hit_rate": float(m2k_only_metrics["monthly_hit_rate"]),
            "active_months": int(m2k_only_metrics["active_months"]),
            "mgc_trade_retention_rate": 0.0,
            "mgc_contribution_pnl": 0.0,
            "top1_contribution_pct": float(m2k_only_metrics["top1_contribution_pct"]) if pd.notna(m2k_only_metrics["top1_contribution_pct"]) else np.nan,
            "top3_contribution_pct": float(m2k_only_metrics["top3_contribution_pct"]) if pd.notna(m2k_only_metrics["top3_contribution_pct"]) else np.nan,
            "top5_contribution_pct": float(m2k_only_metrics["top5_contribution_pct"]) if pd.notna(m2k_only_metrics["top5_contribution_pct"]) else np.nan,
            "worst1_contribution_pct": float(m2k_only_metrics["worst1_contribution_pct"]) if pd.notna(m2k_only_metrics["worst1_contribution_pct"]) else np.nan,
            "worst3_contribution_pct": float(m2k_only_metrics["worst3_contribution_pct"]) if pd.notna(m2k_only_metrics["worst3_contribution_pct"]) else np.nan,
            "worst5_contribution_pct": float(m2k_only_metrics["worst5_contribution_pct"]) if pd.notna(m2k_only_metrics["worst5_contribution_pct"]) else np.nan,
            "verdict": "baseline",
        },
        {
            "entity_name": "mgc_only_baseline",
            "selection_basis": "strict_train_only",
            "deployable": False,
            "net_pnl": float(mgc_only_metrics["net_pnl"]),
            "profit_factor": float(mgc_only_metrics["profit_factor"]),
            "trades": int(mgc_only_metrics["trades"]),
            "positive_folds": int((pd.to_numeric(strict_regime_wfa_fold_breakdown["mgc_only_test_net_pnl"], errors="coerce") > 0).sum()) if not strict_regime_wfa_fold_breakdown.empty else 0,
            "max_drawdown": float(mgc_only_metrics["max_drawdown"]),
            "max_daily_drawdown": float(mgc_only_metrics["max_daily_drawdown"]),
            "win_rate": float(mgc_only_metrics["winrate"]),
            "avg_trade": float(mgc_only_metrics["avg_trade"]),
            "median_trade": float(mgc_only_metrics["median_trade"]),
            "monthly_hit_rate": float(mgc_only_metrics["monthly_hit_rate"]),
            "active_months": int(mgc_only_metrics["active_months"]),
            "mgc_trade_retention_rate": 1.0,
            "mgc_contribution_pnl": float(mgc_only_metrics["net_pnl"]),
            "top1_contribution_pct": float(mgc_only_metrics["top1_contribution_pct"]) if pd.notna(mgc_only_metrics["top1_contribution_pct"]) else np.nan,
            "top3_contribution_pct": float(mgc_only_metrics["top3_contribution_pct"]) if pd.notna(mgc_only_metrics["top3_contribution_pct"]) else np.nan,
            "top5_contribution_pct": float(mgc_only_metrics["top5_contribution_pct"]) if pd.notna(mgc_only_metrics["top5_contribution_pct"]) else np.nan,
            "worst1_contribution_pct": float(mgc_only_metrics["worst1_contribution_pct"]) if pd.notna(mgc_only_metrics["worst1_contribution_pct"]) else np.nan,
            "worst3_contribution_pct": float(mgc_only_metrics["worst3_contribution_pct"]) if pd.notna(mgc_only_metrics["worst3_contribution_pct"]) else np.nan,
            "worst5_contribution_pct": float(mgc_only_metrics["worst5_contribution_pct"]) if pd.notna(mgc_only_metrics["worst5_contribution_pct"]) else np.nan,
            "verdict": "baseline",
        },
        {
            "entity_name": "raw_m2k_mgc_equal_weight",
            "selection_basis": "strict_train_only",
            "deployable": False,
            "net_pnl": float(raw_equal_metrics["net_pnl"]),
            "profit_factor": float(raw_equal_metrics["profit_factor"]),
            "trades": int(raw_equal_metrics["trades"]),
            "positive_folds": int((pd.to_numeric(strict_regime_wfa_fold_breakdown["raw_m2k_mgc_test_net_pnl"], errors="coerce") > 0).sum()) if not strict_regime_wfa_fold_breakdown.empty else 0,
            "max_drawdown": float(raw_equal_metrics["max_drawdown"]),
            "max_daily_drawdown": float(raw_equal_metrics["max_daily_drawdown"]),
            "win_rate": float(raw_equal_metrics["winrate"]),
            "avg_trade": float(raw_equal_metrics["avg_trade"]),
            "median_trade": float(raw_equal_metrics["median_trade"]),
            "monthly_hit_rate": float(raw_equal_metrics["monthly_hit_rate"]),
            "active_months": int(raw_equal_metrics["active_months"]),
            "mgc_trade_retention_rate": 1.0,
            "mgc_contribution_pnl": float(pd.to_numeric(raw_equal_events.loc[raw_equal_events["asset"] == "MGC", "net_pnl_usd"], errors="coerce").sum()),
            "top1_contribution_pct": float(raw_equal_metrics["top1_contribution_pct"]) if pd.notna(raw_equal_metrics["top1_contribution_pct"]) else np.nan,
            "top3_contribution_pct": float(raw_equal_metrics["top3_contribution_pct"]) if pd.notna(raw_equal_metrics["top3_contribution_pct"]) else np.nan,
            "top5_contribution_pct": float(raw_equal_metrics["top5_contribution_pct"]) if pd.notna(raw_equal_metrics["top5_contribution_pct"]) else np.nan,
            "worst1_contribution_pct": float(raw_equal_metrics["worst1_contribution_pct"]) if pd.notna(raw_equal_metrics["worst1_contribution_pct"]) else np.nan,
            "worst3_contribution_pct": float(raw_equal_metrics["worst3_contribution_pct"]) if pd.notna(raw_equal_metrics["worst3_contribution_pct"]) else np.nan,
            "worst5_contribution_pct": float(raw_equal_metrics["worst5_contribution_pct"]) if pd.notna(raw_equal_metrics["worst5_contribution_pct"]) else np.nan,
            "verdict": "baseline",
        },
        {
            "entity_name": "strict_best_regime_gated",
            "selection_basis": "strict_train_only",
            "deployable": strict_verdict == "candidate",
            "net_pnl": float(strict_metrics["net_pnl"]),
            "profit_factor": float(strict_metrics["profit_factor"]),
            "trades": int(strict_metrics["trades"]),
            "positive_folds": strict_positive_folds,
            "max_drawdown": float(strict_metrics["max_drawdown"]),
            "max_daily_drawdown": float(strict_metrics["max_daily_drawdown"]),
            "win_rate": float(strict_metrics["winrate"]),
            "avg_trade": float(strict_metrics["avg_trade"]),
            "median_trade": float(strict_metrics["median_trade"]),
            "monthly_hit_rate": float(strict_metrics["monthly_hit_rate"]),
            "active_months": int(strict_metrics["active_months"]),
            "mgc_trade_retention_rate": float(pd.to_numeric(retention_summary["mgc_test_retention_rate"], errors="coerce").mean()) if not retention_summary.empty else 0.0,
            "mgc_contribution_pnl": float(pd.to_numeric(strict_events.loc[strict_events["asset"] == "MGC", "net_pnl_usd"], errors="coerce").sum()),
            "top1_contribution_pct": float(strict_metrics["top1_contribution_pct"]) if pd.notna(strict_metrics["top1_contribution_pct"]) else np.nan,
            "top3_contribution_pct": float(strict_metrics["top3_contribution_pct"]) if pd.notna(strict_metrics["top3_contribution_pct"]) else np.nan,
            "top5_contribution_pct": float(strict_metrics["top5_contribution_pct"]) if pd.notna(strict_metrics["top5_contribution_pct"]) else np.nan,
            "worst1_contribution_pct": float(strict_metrics["worst1_contribution_pct"]) if pd.notna(strict_metrics["worst1_contribution_pct"]) else np.nan,
            "worst3_contribution_pct": float(strict_metrics["worst3_contribution_pct"]) if pd.notna(strict_metrics["worst3_contribution_pct"]) else np.nan,
            "worst5_contribution_pct": float(strict_metrics["worst5_contribution_pct"]) if pd.notna(strict_metrics["worst5_contribution_pct"]) else np.nan,
            "verdict": strict_verdict,
        },
    ]
    strict_regime_wfa_summary = pd.DataFrame(summary_rows)
    baseline_comparison = strict_regime_wfa_summary.loc[strict_regime_wfa_summary["entity_name"] != "strict_best_regime_gated"].copy()
    diagnostic_rows = build_diagnostic_posthoc_rows(strict_regime_wfa_summary)
    return {
        "strict_regime_wfa_summary": strict_regime_wfa_summary,
        "strict_regime_wfa_fold_breakdown": strict_regime_wfa_fold_breakdown,
        "selected_regime_rule_by_fold": selected_regime_rule_by_fold,
        "regime_rule_train_ranking": regime_rule_train_ranking,
        "portfolio_daily_returns": pd.concat(
            [
                build_portfolio_daily_frame("m2k_only_baseline", m2k_only_events),
                build_portfolio_daily_frame("mgc_only_baseline", mgc_only_events),
                build_portfolio_daily_frame("raw_m2k_mgc_equal_weight", raw_equal_events),
                build_portfolio_daily_frame("strict_best_regime_gated", strict_events),
            ],
            ignore_index=True,
        ),
        "portfolio_monthly_pnl": pd.concat(
            [
                build_monthly_yearly_frames("m2k_only_baseline", m2k_only_events)[0],
                build_monthly_yearly_frames("mgc_only_baseline", mgc_only_events)[0],
                build_monthly_yearly_frames("raw_m2k_mgc_equal_weight", raw_equal_events)[0],
                build_monthly_yearly_frames("strict_best_regime_gated", strict_events)[0],
            ],
            ignore_index=True,
        ),
        "portfolio_yearly_pnl": pd.concat(
            [
                build_monthly_yearly_frames("m2k_only_baseline", m2k_only_events)[1],
                build_monthly_yearly_frames("mgc_only_baseline", mgc_only_events)[1],
                build_monthly_yearly_frames("raw_m2k_mgc_equal_weight", raw_equal_events)[1],
                build_monthly_yearly_frames("strict_best_regime_gated", strict_events)[1],
            ],
            ignore_index=True,
        ),
        "mgc_regime_retention_summary": retention_summary,
        "trade_concentration": pd.DataFrame(
            [
                {"entity_name": "m2k_only_baseline", **_trade_concentration(m2k_only_events, pnl_column="net_pnl_usd")},
                {"entity_name": "mgc_only_baseline", **_trade_concentration(mgc_only_events, pnl_column="net_pnl_usd")},
                {"entity_name": "raw_m2k_mgc_equal_weight", **_trade_concentration(raw_equal_events, pnl_column="net_pnl_usd")},
                {"entity_name": "strict_best_regime_gated", **_trade_concentration(strict_events, pnl_column="net_pnl_usd")},
            ]
        ),
        "baseline_comparison": baseline_comparison,
        "rejected_or_diagnostic_results": diagnostic_rows,
        "strict_events": strict_events,
        "data_audits": {symbol: contexts[symbol]["data_audit"] for symbol in contexts},
        "survivor_summary": survivor_summary,
        "survivor_audit_dir": str(survivor_audit_dir),
    }


def build_final_report(
    *,
    output_dir: Path,
    strict_summary: pd.DataFrame,
    fold_breakdown: pd.DataFrame,
    selected_rules: pd.DataFrame,
    retention_summary: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
) -> None:
    def _lookup(entity_name: str) -> pd.Series | None:
        part = strict_summary.loc[strict_summary["entity_name"] == entity_name].copy()
        return None if part.empty else part.iloc[0]

    best = _lookup("strict_best_regime_gated")
    m2k = _lookup("m2k_only_baseline")
    raw = _lookup("raw_m2k_mgc_equal_weight")
    rule_labels = (
        selected_rules[["fold_id", "rule_id", "allocation_scheme"]].astype(str).to_dict("records")
        if not selected_rules.empty
        else []
    )
    lines = [
        "# Volume Climax Pullback Regime-Gated Portfolio",
        "",
        "## 1. Executive Summary",
        f"- M2K-only baseline: `{safe_float(m2k['net_pnl'], 0.0):.2f} USD`, `PF {safe_float(m2k['profit_factor'], 0.0):.2f}`." if m2k is not None else "- M2K-only baseline missing.",
        f"- Raw M2K+MGC baseline: `{safe_float(raw['net_pnl'], 0.0):.2f} USD`, `PF {safe_float(raw['profit_factor'], 0.0):.2f}`." if raw is not None else "- Raw M2K+MGC baseline missing.",
        f"- Best strict regime-gated portfolio: `{safe_float(best['net_pnl'], 0.0):.2f} USD`, `PF {safe_float(best['profit_factor'], 0.0):.2f}`, verdict `{best['verdict']}`." if best is not None else "- No strict regime-gated portfolio row.",
        f"- Average MGC retention rate: `{safe_float(retention_summary['mgc_test_retention_rate'].mean() if not retention_summary.empty else 0.0, 0.0):.2%}`.",
        "",
        "## 2. Selected Rules By Fold",
        _markdown_table(selected_rules[["fold_id", "rule_id", "allocation_scheme", "fitted_params_json"]]) if not selected_rules.empty else "No selected rules.",
        "",
        "## 3. Strict Fold Breakdown",
        _markdown_table(fold_breakdown) if not fold_breakdown.empty else "No fold breakdown.",
        "",
        "## 4. Baseline Comparison",
        _markdown_table(baseline_comparison) if not baseline_comparison.empty else "No baselines.",
        "",
        "## 5. MGC Retention",
        _markdown_table(retention_summary) if not retention_summary.empty else "No retention rows.",
        "",
        "## 6. Verdict",
        f"- Final strict verdict: `{best['verdict']}`" if best is not None else "- Final strict verdict unavailable.",
        f"- Fold rule path: `{json.dumps(rule_labels, ensure_ascii=True)}`",
        "- Posthoc diagnostics remain non-deployable by construction.",
    ]
    (output_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_campaign(
    *,
    symbols: Sequence[str],
    signal_timeframe: str,
    execution_timeframe: str,
    output_root: Path,
    smoke: bool = False,
    skip_negative_control: bool = False,
    max_regime_rules: int = 36,
    log_level: str = "INFO",
    survivor_audit_dir: Path | None = None,
    dataset_overrides: dict[str, pd.DataFrame] | None = None,
) -> Path:
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / f"volume_climax_pullback_regime_gated_portfolio_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    survivor_dir = Path(survivor_audit_dir) if survivor_audit_dir is not None else latest_survivor_audit_dir(Path(output_root))

    results = evaluate_regime_gated_portfolio(
        symbols=list(symbols),
        signal_timeframe=signal_timeframe,
        execution_timeframe=execution_timeframe,
        survivor_audit_dir=survivor_dir,
        max_regime_rules=8 if smoke else max_regime_rules,
        include_negative_control=not skip_negative_control,
        dataset_overrides=dataset_overrides,
    )

    run_metadata = {
        "timestamp": timestamp,
        "symbols": list(symbols),
        "signal_timeframe": signal_timeframe,
        "execution_timeframe": execution_timeframe,
        "smoke": bool(smoke),
        "skip_negative_control": bool(skip_negative_control),
        "max_regime_rules": int(8 if smoke else max_regime_rules),
        "survivor_audit_dir": str(survivor_dir),
        "python_version": sys.version,
        "platform": platform.platform(),
        "data_audits": results["data_audits"],
        "input_files": {
            symbol: _file_metadata(Path(audit["source_path"]))
            for symbol, audit in results["data_audits"].items()
            if audit.get("source_path")
        },
    }

    results["strict_regime_wfa_summary"].to_csv(output_dir / "strict_regime_wfa_summary.csv", index=False)
    results["strict_regime_wfa_fold_breakdown"].to_csv(output_dir / "strict_regime_wfa_fold_breakdown.csv", index=False)
    results["selected_regime_rule_by_fold"].to_csv(output_dir / "selected_regime_rule_by_fold.csv", index=False)
    results["regime_rule_train_ranking"].to_csv(output_dir / "regime_rule_train_ranking.csv", index=False)
    results["portfolio_daily_returns"].to_csv(output_dir / "portfolio_daily_returns.csv", index=False)
    results["portfolio_monthly_pnl"].to_csv(output_dir / "portfolio_monthly_pnl.csv", index=False)
    results["portfolio_yearly_pnl"].to_csv(output_dir / "portfolio_yearly_pnl.csv", index=False)
    results["mgc_regime_retention_summary"].to_csv(output_dir / "mgc_regime_retention_summary.csv", index=False)
    results["trade_concentration"].to_csv(output_dir / "trade_concentration.csv", index=False)
    results["baseline_comparison"].to_csv(output_dir / "baseline_comparison.csv", index=False)
    results["rejected_or_diagnostic_results"].to_csv(output_dir / "rejected_or_diagnostic_results.csv", index=False)
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    build_final_report(
        output_dir=output_dir,
        strict_summary=results["strict_regime_wfa_summary"],
        fold_breakdown=results["strict_regime_wfa_fold_breakdown"],
        selected_rules=results["selected_regime_rule_by_fold"],
        retention_summary=results["mgc_regime_retention_summary"],
        baseline_comparison=results["baseline_comparison"],
    )
    return output_dir


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--signal-timeframe", default=DEFAULT_SIGNAL_TIMEFRAME)
    parser.add_argument("--execution-timeframe", default=DEFAULT_EXECUTION_TIMEFRAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-negative-control", action="store_true")
    parser.add_argument("--max-regime-rules", type=int, default=36)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--survivor-audit-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    output_dir = run_campaign(
        symbols=list(args.symbols),
        signal_timeframe=str(args.signal_timeframe),
        execution_timeframe=str(args.execution_timeframe),
        output_root=Path(args.output_root),
        smoke=bool(args.smoke),
        skip_negative_control=bool(args.skip_negative_control),
        max_regime_rules=int(args.max_regime_rules),
        log_level=str(args.log_level),
        survivor_audit_dir=args.survivor_audit_dir,
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
