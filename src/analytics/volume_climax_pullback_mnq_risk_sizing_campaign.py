"""Dedicated symbol-level risk-sizing campaign for Volume Climax Pullback."""

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
from src.analytics.volume_climax_pullback_common import (
    load_latest_reference_run,
    load_symbol_data,
    resample_rth_1h,
    split_sessions,
)
from src.config.orb_campaign import PropConstraintConfig
from src.config.paths import EXPORTS_DIR, ensure_directories
from src.engine.volume_climax_pullback_v2_backtester import run_volume_climax_pullback_v2_backtest
from src.engine.vwap_backtester import build_execution_model_for_profile
from src.risk.position_sizing import (
    FixedContractPositionSizing,
    PositionSizingConfig,
    RiskPercentPositionSizing,
)
from src.strategy.volume_climax_pullback_v2 import (
    VolumeClimaxPullbackV2Variant,
    build_volume_climax_pullback_v2_signal_frame,
    build_volume_climax_pullback_v3_variants,
    prepare_volume_climax_pullback_v2_features,
)


DEFAULT_SYMBOL = "MNQ"
DEFAULT_INITIAL_CAPITAL_USD = 50_000.0
DEFAULT_RISK_PCTS = (0.0025, 0.0050, 0.0075, 0.0100)
DEFAULT_MAX_CONTRACTS = (3, 5, 10, 15)
DEFAULT_SKIP_FLAGS = (True, False)
DEFAULT_OUTPUT_PREFIX = "volume_climax_pullback_mnq_risk_sizing_"
DEFAULT_REFERENCE_ROOT = EXPORTS_DIR / "volume_climax_pullback_v3_run"
DEFAULT_REFERENCE_PREFIX = "volume_climax_pullback_v3_"

TOPSTEP_50K_PROP = PropConstraintConfig(
    name="topstep_50k_reference",
    account_size_usd=DEFAULT_INITIAL_CAPITAL_USD,
    max_loss_limit_usd=2_000.0,
    daily_loss_limit_usd=1_000.0,
    profit_target_usd=3_000.0,
)


@dataclass(frozen=True)
class CampaignVariantSpec:
    campaign_variant_name: str
    sizing_mode: str
    fixed_contracts: int | None
    risk_pct: float | None
    max_contracts: int | None
    skip_trade_if_too_small: bool | None
    initial_capital_usd: float
    position_sizing: PositionSizingConfig


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
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    clean = {key: _serialize_value(value) for key, value in payload.items()}
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")


def _optional_int(series: pd.Series, name: str) -> int | None:
    value = series.get(name)
    if pd.isna(value):
        return None
    return int(value)


def _optional_float(series: pd.Series, name: str) -> float | None:
    value = series.get(name)
    if pd.isna(value):
        return None
    return float(value)


def _variant_from_summary_row(row: pd.Series) -> VolumeClimaxPullbackV2Variant:
    return VolumeClimaxPullbackV2Variant(
        name=str(row.get("variant_name") or row.get("name")),
        family=str(row["family"]),
        timeframe=str(row.get("timeframe", "1h")),
        volume_quantile=float(row["volume_quantile"]),
        volume_lookback=int(row.get("volume_lookback", 50)),
        min_body_fraction=float(row["min_body_fraction"]),
        min_range_atr=float(row["min_range_atr"]),
        trend_ema_window=_optional_int(row, "trend_ema_window"),
        ema_slope_threshold=_optional_float(row, "ema_slope_threshold"),
        atr_percentile_low=_optional_float(row, "atr_percentile_low"),
        atr_percentile_high=_optional_float(row, "atr_percentile_high"),
        compression_ratio_max=_optional_float(row, "compression_ratio_max"),
        entry_mode=str(row.get("entry_mode", "next_open")),
        pullback_fraction=_optional_float(row, "pullback_fraction"),
        confirmation_window=_optional_int(row, "confirmation_window"),
        exit_mode=str(row["exit_mode"]),
        rr_target=float(row.get("rr_target", 1.0)),
        atr_target_multiple=_optional_float(row, "atr_target_multiple"),
        time_stop_bars=int(row["time_stop_bars"]),
        trailing_atr_multiple=float(row.get("trailing_atr_multiple", 0.5)),
        session_overlay=str(row.get("session_overlay", "all_rth")),
    )


def _status_rank(frame: pd.DataFrame) -> pd.Series:
    if "status_rank" in frame.columns:
        return pd.to_numeric(frame["status_rank"], errors="coerce").fillna(99.0)
    if "variant_status" not in frame.columns:
        return pd.Series(99.0, index=frame.index, dtype=float)
    mapping = {"survivor": 0.0, "fragile": 1.0, "dead": 2.0}
    return frame["variant_status"].astype(str).map(mapping).fillna(99.0)


def _resolve_base_alpha_variant(
    *,
    base_variant_name: str | None,
    reference_v3_dir: Path | None,
) -> tuple[VolumeClimaxPullbackV2Variant, Path | None, str | None]:
    catalog = {variant.name: variant for variant in build_volume_climax_pullback_v3_variants(DEFAULT_SYMBOL)}

    resolved_reference_dir: Path | None = None
    reference_summary: pd.DataFrame | None = None
    if reference_v3_dir is not None:
        resolved_reference_dir = Path(reference_v3_dir)
    else:
        try:
            resolved_reference_dir = load_latest_reference_run(DEFAULT_REFERENCE_ROOT, DEFAULT_REFERENCE_PREFIX)
        except FileNotFoundError:
            resolved_reference_dir = None

    if resolved_reference_dir is not None:
        summary_path = resolved_reference_dir / "summary_variants.csv"
        if summary_path.exists():
            reference_summary = pd.read_csv(summary_path)

    if base_variant_name is not None:
        if base_variant_name in catalog:
            return catalog[base_variant_name], resolved_reference_dir, base_variant_name
        if reference_summary is not None:
            matched = reference_summary.loc[
                (reference_summary["symbol"].astype(str) == DEFAULT_SYMBOL)
                & (reference_summary["variant_name"].astype(str) == base_variant_name)
            ].copy()
            if not matched.empty:
                return _variant_from_summary_row(matched.iloc[0]), resolved_reference_dir, base_variant_name
        raise ValueError(f"Base alpha variant {base_variant_name!r} was not found.")

    if reference_summary is None:
        raise FileNotFoundError(
            "No V3 reference summary is available. Pass --base-variant-name explicitly or provide --reference-v3-dir."
        )

    symbol_summary = reference_summary.loc[reference_summary["symbol"].astype(str) == DEFAULT_SYMBOL].copy()
    if symbol_summary.empty:
        raise ValueError(f"No {DEFAULT_SYMBOL} rows were found in the V3 reference summary.")

    symbol_summary["status_rank_local"] = _status_rank(symbol_summary)
    ordered = symbol_summary.sort_values(
        ["status_rank_local", "selection_score", "oos_sharpe", "oos_net_pnl"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    selected = ordered.iloc[0]
    variant_name = str(selected["variant_name"])
    if variant_name in catalog:
        return catalog[variant_name], resolved_reference_dir, variant_name
    return _variant_from_summary_row(selected), resolved_reference_dir, variant_name


def _risk_label(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def _campaign_variants(
    *,
    initial_capital_usd: float,
    risk_pcts: tuple[float, ...],
    max_contracts_grid: tuple[int, ...],
    skip_flags: tuple[bool, ...],
) -> list[CampaignVariantSpec]:
    variants = [
        CampaignVariantSpec(
            campaign_variant_name="fixed_1_contract",
            sizing_mode="fixed_contracts",
            fixed_contracts=1,
            risk_pct=None,
            max_contracts=None,
            skip_trade_if_too_small=None,
            initial_capital_usd=float(initial_capital_usd),
            position_sizing=FixedContractPositionSizing(fixed_contracts=1),
        )
    ]
    for risk_pct in risk_pcts:
        for max_contracts in max_contracts_grid:
            for skip_flag in skip_flags:
                variants.append(
                    CampaignVariantSpec(
                        campaign_variant_name=(
                            f"risk_pct_{_risk_label(risk_pct)}"
                            f"__max_contracts_{int(max_contracts)}"
                            f"__skip_trade_if_too_small_{str(bool(skip_flag)).lower()}"
                        ),
                        sizing_mode="risk_percent",
                        fixed_contracts=None,
                        risk_pct=float(risk_pct),
                        max_contracts=int(max_contracts),
                        skip_trade_if_too_small=bool(skip_flag),
                        initial_capital_usd=float(initial_capital_usd),
                        position_sizing=RiskPercentPositionSizing(
                            initial_capital_usd=float(initial_capital_usd),
                            risk_pct=float(risk_pct),
                            max_contracts=int(max_contracts),
                            skip_trade_if_too_small=bool(skip_flag),
                        ),
                    )
                )
    return variants


def _subset_by_sessions(frame: pd.DataFrame, sessions: list) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    session_set = set(pd.to_datetime(pd.Index(sessions)).date)
    out = frame.copy()
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.date
    return out.loc[out["session_date"].isin(session_set)].copy().reset_index(drop=True)


def _daily_results_from_trades(trades: pd.DataFrame, sessions: list, initial_capital: float) -> pd.DataFrame:
    session_index = pd.Index(pd.to_datetime(pd.Index(sessions)).date)
    daily = pd.DataFrame({"session_date": session_index})
    if trades.empty:
        daily["daily_pnl_usd"] = 0.0
        daily["daily_gross_pnl_usd"] = 0.0
        daily["daily_fees_usd"] = 0.0
        daily["daily_trade_count"] = 0
        daily["daily_loss_count"] = 0
    else:
        view = trades.copy()
        view["session_date"] = pd.to_datetime(view["session_date"]).dt.date
        view["loss_trade"] = pd.to_numeric(view["net_pnl_usd"], errors="coerce").lt(0)
        grouped = (
            view.groupby("session_date", as_index=False)
            .agg(
                daily_pnl_usd=("net_pnl_usd", "sum"),
                daily_gross_pnl_usd=("pnl_usd", "sum"),
                daily_fees_usd=("fees", "sum"),
                daily_trade_count=("trade_id", "count"),
                daily_loss_count=("loss_trade", "sum"),
            )
        )
        daily = daily.merge(grouped, on="session_date", how="left").fillna(
            {
                "daily_pnl_usd": 0.0,
                "daily_gross_pnl_usd": 0.0,
                "daily_fees_usd": 0.0,
                "daily_trade_count": 0,
                "daily_loss_count": 0,
            }
        )
    daily = daily.sort_values("session_date").reset_index(drop=True)
    daily["equity"] = float(initial_capital) + pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0).cumsum()
    daily["peak_equity"] = daily["equity"].cummax()
    daily["drawdown_usd"] = daily["equity"] - daily["peak_equity"]
    daily["drawdown_pct"] = np.where(
        daily["peak_equity"] > 0,
        (daily["equity"] / daily["peak_equity"] - 1.0) * 100.0,
        0.0,
    )
    daily["cumulative_pnl_usd"] = daily["equity"] - float(initial_capital)
    return daily


def _sortino_ratio(daily_pnl: pd.Series, capital: float) -> float:
    if capital <= 0:
        return 0.0
    returns = pd.Series(daily_pnl, dtype=float) / float(capital)
    downside = returns[returns < 0]
    if len(returns) < 2 or downside.empty:
        return 0.0
    downside_std = float(np.sqrt(np.mean(np.square(downside))))
    if downside_std <= 0:
        return 0.0
    return float((returns.mean() / downside_std) * math.sqrt(252.0))


def _annualized_vol_pct(daily_pnl: pd.Series, capital: float) -> float:
    if capital <= 0:
        return 0.0
    returns = pd.Series(daily_pnl, dtype=float) / float(capital)
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=0) * math.sqrt(252.0) * 100.0)


def _cagr_pct(daily_results: pd.DataFrame, initial_capital: float) -> float:
    if daily_results.empty or initial_capital <= 0:
        return float("nan")
    ordered = daily_results.sort_values("session_date").reset_index(drop=True)
    start = pd.Timestamp(ordered.iloc[0]["session_date"])
    end = pd.Timestamp(ordered.iloc[-1]["session_date"])
    span_days = max((end - start).days + 1, 1)
    years = float(span_days / 365.25)
    if years <= 0:
        return float("nan")
    final_equity = float(ordered.iloc[-1]["equity"])
    if final_equity <= 0:
        return float("nan")
    return float(((final_equity / float(initial_capital)) ** (1.0 / years) - 1.0) * 100.0)


def _exposure_time_in_market_pct(trades: pd.DataFrame, signal_df: pd.DataFrame) -> float:
    total_bars = max(int(len(signal_df)), 1)
    held_bars = pd.to_numeric(trades.get("bars_held"), errors="coerce").fillna(0.0).sum() if not trades.empty else 0.0
    return float(min(max(held_bars / float(total_bars), 0.0), 1.0) * 100.0)


def _sizing_activity_metrics(sizing_decisions: pd.DataFrame) -> dict[str, float | int]:
    if sizing_decisions.empty:
        return {
            "signal_attempt_count": 0,
            "entered_trade_count": 0,
            "skip_too_small_count": 0,
            "forced_one_contract_count": 0,
            "avg_contracts_entered": 0.0,
            "max_contracts_used": 0,
            "avg_contracts_raw": 0.0,
            "cap_hit_count": 0,
            "cap_hit_rate": 0.0,
        }

    contracts = pd.to_numeric(sizing_decisions["contracts"], errors="coerce").fillna(0.0)
    contracts_raw = pd.to_numeric(sizing_decisions["contracts_raw"], errors="coerce")
    max_contracts = pd.to_numeric(sizing_decisions["max_contracts"], errors="coerce")
    skipped = sizing_decisions["skipped"].fillna(True).astype(bool)
    entered = ~skipped & contracts.gt(0)
    cap_hit = entered & max_contracts.notna() & contracts_raw.gt(max_contracts)
    forced_one = entered & contracts.eq(1.0) & contracts_raw.lt(1.0)
    return {
        "signal_attempt_count": int(len(sizing_decisions)),
        "entered_trade_count": int(entered.sum()),
        "skip_too_small_count": int(sizing_decisions["skip_reason"].astype(str).eq("contracts_below_one").sum()),
        "forced_one_contract_count": int(forced_one.sum()),
        "avg_contracts_entered": float(contracts.loc[entered].mean()) if entered.any() else 0.0,
        "max_contracts_used": int(contracts.loc[entered].max()) if entered.any() else 0,
        "avg_contracts_raw": float(contracts_raw.loc[entered].mean()) if entered.any() else 0.0,
        "cap_hit_count": int(cap_hit.sum()),
        "cap_hit_rate": float(cap_hit.mean()) if len(sizing_decisions) > 0 else 0.0,
    }


def _prop_path_metrics(daily_results: pd.DataFrame, initial_capital: float) -> dict[str, float | int | bool]:
    if daily_results.empty:
        return {
            "worst_day_pnl_usd": 0.0,
            "max_daily_drawdown_usd": 0.0,
            "days_below_minus_500_usd": 0,
            "days_below_minus_1000_usd": 0,
            "pass_target_3000_usd_without_breaching_2000_dd": False,
            "days_to_target_3000_usd": np.nan,
            "max_trailing_drawdown_observed_usd": 0.0,
            "max_static_drawdown_observed_usd": 0.0,
        }

    ordered = daily_results.sort_values("session_date").reset_index(drop=True)
    daily_pnl = pd.to_numeric(ordered["daily_pnl_usd"], errors="coerce").fillna(0.0)
    cumulative = pd.to_numeric(ordered["cumulative_pnl_usd"], errors="coerce").fillna(0.0)
    trailing_dd = pd.to_numeric(ordered["peak_equity"], errors="coerce").fillna(float(initial_capital)) - pd.to_numeric(
        ordered["equity"], errors="coerce"
    ).fillna(float(initial_capital))
    static_dd = float(initial_capital) - pd.to_numeric(ordered["equity"], errors="coerce").fillna(float(initial_capital))

    target_hits = np.flatnonzero(cumulative >= 3_000.0)
    trailing_breaches = np.flatnonzero(trailing_dd > 2_000.0)
    static_breaches = np.flatnonzero(static_dd > 2_000.0)
    target_idx = int(target_hits[0]) if len(target_hits) > 0 else None
    trailing_idx = int(trailing_breaches[0]) if len(trailing_breaches) > 0 else None
    static_idx = int(static_breaches[0]) if len(static_breaches) > 0 else None

    pass_without_breach = bool(
        target_idx is not None
        and (trailing_idx is None or target_idx <= trailing_idx)
        and (static_idx is None or target_idx <= static_idx)
    )

    return {
        "worst_day_pnl_usd": float(daily_pnl.min()) if not daily_pnl.empty else 0.0,
        "max_daily_drawdown_usd": float(trailing_dd.max()) if not trailing_dd.empty else 0.0,
        "days_below_minus_500_usd": int((daily_pnl <= -500.0).sum()),
        "days_below_minus_1000_usd": int((daily_pnl <= -1_000.0).sum()),
        "pass_target_3000_usd_without_breaching_2000_dd": pass_without_breach,
        "days_to_target_3000_usd": float(target_idx + 1) if target_idx is not None else np.nan,
        "max_trailing_drawdown_observed_usd": float(trailing_dd.max()) if not trailing_dd.empty else 0.0,
        "max_static_drawdown_observed_usd": float(max(static_dd.max(), 0.0)) if not static_dd.empty else 0.0,
    }


def _summarize_scope(
    *,
    trades: pd.DataFrame,
    signal_df: pd.DataFrame,
    sessions: list,
    daily_results: pd.DataFrame,
    sizing_decisions: pd.DataFrame,
    initial_capital: float,
) -> dict[str, Any]:
    metrics = compute_metrics(
        trades,
        signal_df=signal_df,
        session_dates=sessions,
        initial_capital=initial_capital,
        prop_constraints=TOPSTEP_50K_PROP,
    )

    pnl = pd.to_numeric(trades.get("net_pnl_usd"), errors="coerce").fillna(0.0) if not trades.empty else pd.Series(dtype=float)
    daily_pnl = pd.to_numeric(daily_results["daily_pnl_usd"], errors="coerce").fillna(0.0) if not daily_results.empty else pd.Series(dtype=float)
    prop_metrics = _prop_path_metrics(daily_results, initial_capital=initial_capital)
    sizing_metrics = _sizing_activity_metrics(sizing_decisions)

    return {
        "net_pnl_usd": float(metrics.get("cumulative_pnl", 0.0)),
        "return_pct": float(metrics.get("cumulative_pnl", 0.0) / float(initial_capital) * 100.0) if initial_capital > 0 else 0.0,
        "cagr_pct": _cagr_pct(daily_results, initial_capital=initial_capital),
        "sharpe": float(metrics.get("sharpe_ratio", 0.0)),
        "sortino": _sortino_ratio(daily_pnl, capital=initial_capital),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "expectancy_usd": float(metrics.get("expectancy", 0.0)),
        "nb_trades": int(metrics.get("n_trades", 0)),
        "win_rate": float(metrics.get("win_rate", 0.0)),
        "avg_win_usd": float(metrics.get("avg_win", 0.0)),
        "avg_loss_usd": float(metrics.get("avg_loss", 0.0)),
        "max_drawdown_pct": float(abs(metrics.get("max_drawdown_pct", 0.0)) * 100.0),
        "max_drawdown_usd": float(abs(metrics.get("max_drawdown", 0.0))),
        "annualized_vol_pct": _annualized_vol_pct(daily_pnl, capital=initial_capital),
        "exposure_time_in_market_pct": _exposure_time_in_market_pct(trades, signal_df),
        "worst_trade_loss_usd": float(abs(min(pnl.min(), 0.0))) if not pnl.empty else 0.0,
        "trades_loss_gt_250_usd": int((pnl <= -250.0).sum()) if not pnl.empty else 0,
        "trades_loss_gt_500_usd": int((pnl <= -500.0).sum()) if not pnl.empty else 0,
        **prop_metrics,
        **sizing_metrics,
    }


def _phase_map(is_sessions: list, oos_sessions: list) -> dict[Any, str]:
    mapping = {}
    for session in is_sessions:
        mapping[pd.Timestamp(session).date()] = "is"
    for session in oos_sessions:
        mapping[pd.Timestamp(session).date()] = "oos"
    return mapping


def _format_variant_brief(row: pd.Series) -> str:
    if row["sizing_mode"] == "fixed_contracts":
        return "fixed_1_contract"
    return (
        f"risk={float(row['risk_pct']):.4f}, cap={int(row['max_contracts'])}, "
        f"skip_small={bool(row['skip_trade_if_too_small'])}"
    )


def _top_table(summary: pd.DataFrame, *, scope_prefix: str, top_n: int = 8) -> str:
    cols = [
        "campaign_variant_name",
        f"{scope_prefix}_net_pnl_usd",
        f"{scope_prefix}_cagr_pct",
        f"{scope_prefix}_sharpe",
        f"{scope_prefix}_max_drawdown_usd",
        f"{scope_prefix}_pass_target_3000_usd_without_breaching_2000_dd",
        f"{scope_prefix}_avg_contracts_entered",
    ]
    table = summary[cols].copy()
    table = table.sort_values(
        [f"{scope_prefix}_net_pnl_usd", f"{scope_prefix}_sharpe", f"{scope_prefix}_max_drawdown_usd"],
        ascending=[False, False, True],
    ).head(top_n)
    return table.to_string(index=False)


def _paired_skip_impact(summary: pd.DataFrame) -> pd.DataFrame:
    risk_rows = summary.loc[summary["sizing_mode"] == "risk_percent"].copy()
    if risk_rows.empty:
        return pd.DataFrame()
    true_rows = risk_rows.loc[risk_rows["skip_trade_if_too_small"].eq(True)].copy()
    false_rows = risk_rows.loc[risk_rows["skip_trade_if_too_small"].eq(False)].copy()
    merged = true_rows.merge(
        false_rows,
        on=["risk_pct", "max_contracts"],
        suffixes=("_skip_true", "_skip_false"),
    )
    if merged.empty:
        return merged
    merged["oos_net_pnl_delta_force_minus_skip"] = (
        pd.to_numeric(merged["oos_net_pnl_usd_skip_false"], errors="coerce")
        - pd.to_numeric(merged["oos_net_pnl_usd_skip_true"], errors="coerce")
    )
    merged["oos_max_dd_delta_force_minus_skip"] = (
        pd.to_numeric(merged["oos_max_drawdown_usd_skip_false"], errors="coerce")
        - pd.to_numeric(merged["oos_max_drawdown_usd_skip_true"], errors="coerce")
    )
    merged["oos_trade_delta_force_minus_skip"] = (
        pd.to_numeric(merged["oos_nb_trades_skip_false"], errors="coerce")
        - pd.to_numeric(merged["oos_nb_trades_skip_true"], errors="coerce")
    )
    merged["oos_forced_one_contract_count"] = pd.to_numeric(
        merged["oos_forced_one_contract_count_skip_false"], errors="coerce"
    ).fillna(0.0)
    return merged


def _recommendation_rows(summary: pd.DataFrame) -> dict[str, pd.Series | None]:
    baseline = summary.loc[summary["campaign_variant_name"] == "fixed_1_contract"].iloc[0]
    risk_rows = summary.loc[summary["sizing_mode"] == "risk_percent"].copy()
    if risk_rows.empty:
        return {"baseline": baseline, "recommended": None, "aggressive": None, "prop_safe": None}

    improving = risk_rows.loc[
        (pd.to_numeric(risk_rows["oos_net_pnl_usd"], errors="coerce") > float(baseline["oos_net_pnl_usd"]))
        & (pd.to_numeric(risk_rows["oos_cagr_pct"], errors="coerce") >= float(baseline["oos_cagr_pct"]))
        & (pd.to_numeric(risk_rows["oos_max_drawdown_usd"], errors="coerce") <= float(baseline["oos_max_drawdown_usd"]) * 1.25)
    ].copy()
    if improving.empty:
        improving = risk_rows.loc[
            (pd.to_numeric(risk_rows["oos_sharpe"], errors="coerce") >= float(baseline["oos_sharpe"]))
            & (pd.to_numeric(risk_rows["oos_max_drawdown_usd"], errors="coerce") <= float(baseline["oos_max_drawdown_usd"]) * 1.10)
        ].copy()

    recommended = (
        improving.sort_values(["oos_sharpe", "oos_net_pnl_usd", "oos_max_drawdown_usd"], ascending=[False, False, True]).iloc[0]
        if not improving.empty
        else None
    )
    aggressive = risk_rows.sort_values(
        ["oos_net_pnl_usd", "oos_cagr_pct", "oos_sharpe"],
        ascending=[False, False, False],
    ).iloc[0]
    prop_pool = risk_rows.loc[
        pd.Series(risk_rows["oos_pass_target_3000_usd_without_breaching_2000_dd"], dtype="boolean").fillna(False)
    ].copy()
    if prop_pool.empty:
        prop_pool = risk_rows.loc[
            pd.to_numeric(risk_rows["oos_max_trailing_drawdown_observed_usd"], errors="coerce") <= 2_000.0
        ].copy()
    prop_safe = (
        prop_pool.sort_values(
            ["oos_max_trailing_drawdown_observed_usd", "oos_days_below_minus_1000_usd", "oos_net_pnl_usd", "oos_sharpe"],
            ascending=[True, True, False, False],
        ).iloc[0]
        if not prop_pool.empty
        else None
    )
    return {
        "baseline": baseline,
        "recommended": recommended,
        "aggressive": aggressive,
        "prop_safe": prop_safe,
    }


def _concat_non_empty_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame.dropna(axis=1, how="all") for frame in frames if not frame.empty]
    if not usable:
        return pd.DataFrame()
    return pd.concat(usable, ignore_index=True, sort=False)


def _build_final_report(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    base_alpha_variant: VolumeClimaxPullbackV2Variant,
    reference_variant_name: str | None,
    reference_v3_dir: Path | None,
    dataset_path: Path | None,
    is_sessions: list,
    oos_sessions: list,
) -> dict[str, Any]:
    picks = _recommendation_rows(summary)
    baseline = picks["baseline"]
    recommended = picks["recommended"]
    aggressive = picks["aggressive"]
    prop_safe = picks["prop_safe"]
    paired_skip = _paired_skip_impact(summary)

    risk_rows = summary.loc[summary["sizing_mode"] == "risk_percent"].copy()
    improved_rows = risk_rows.loc[
        (pd.to_numeric(risk_rows["oos_net_pnl_usd"], errors="coerce") > float(baseline["oos_net_pnl_usd"]))
        & (pd.to_numeric(risk_rows["oos_max_drawdown_usd"], errors="coerce") <= float(baseline["oos_max_drawdown_usd"]) * 1.25)
    ].copy()
    cap_best = (
        risk_rows.sort_values(["oos_net_pnl_usd", "oos_sharpe"], ascending=[False, False])
        .groupby("max_contracts", dropna=True, as_index=False)
        .head(1)[["campaign_variant_name", "max_contracts", "oos_net_pnl_usd", "oos_sharpe", "oos_cap_hit_rate"]]
        .sort_values("max_contracts")
    )

    final_verdict = "ne_pas_retenir" if recommended is None else "retenir_avec_pref"

    lines = [
        f"# Volume Climax Pullback {DEFAULT_SYMBOL} Risk Sizing - Final Report",
        "",
        "## Scope",
        f"- Symbol: `{DEFAULT_SYMBOL}` only.",
        f"- Base alpha reused unchanged: `{reference_variant_name or base_alpha_variant.name}`.",
        f"- Resolved alpha variant object: `{base_alpha_variant.name}`.",
        f"- Reference V3 run: `{reference_v3_dir}`." if reference_v3_dir is not None else "- Reference V3 run: `not used`.",
        f"- Dataset: `{dataset_path}`." if dataset_path is not None else f"- Dataset: `repo latest {DEFAULT_SYMBOL} source`.",
        f"- Sessions: full `{len(is_sessions) + len(oos_sessions)}` | IS `{len(is_sessions)}` | OOS `{len(oos_sessions)}`.",
        "- OOS-only runs restart from the same 50k capital while keeping signals precomputed on the full leak-free history.",
        "",
        "## Baseline Vs Sizing",
        f"- Baseline OOS: net `{float(baseline['oos_net_pnl_usd']):.2f}` | CAGR `{float(baseline['oos_cagr_pct']):.2f}%` | Sharpe `{float(baseline['oos_sharpe']):.3f}` | maxDD `{float(baseline['oos_max_drawdown_usd']):.2f}`.",
        f"- Best risk-sized OOS row by net PnL: `{aggressive['campaign_variant_name']}` | net `{float(aggressive['oos_net_pnl_usd']):.2f}` | CAGR `{float(aggressive['oos_cagr_pct']):.2f}%` | Sharpe `{float(aggressive['oos_sharpe']):.3f}` | maxDD `{float(aggressive['oos_max_drawdown_usd']):.2f}`."
        if aggressive is not None
        else "- No risk-sized row was available.",
        "",
        "```text",
        _top_table(summary, scope_prefix="oos"),
        "```",
        "",
        "## Readout",
        (
            f"1. Baseline 1 contrat vs risk sizing: `{int(len(improved_rows))}` risk-sized rows improved OOS net PnL while keeping OOS maxDD within +25% of baseline."
            if not improved_rows.empty
            else "1. Baseline 1 contrat vs risk sizing: no risk-sized row improved OOS net PnL without a material maxDD trade-off."
        ),
        (
            "2. Variantes qui ameliorent CAGR / net PnL sans degrader excessivement le maxDD: "
            + ", ".join(f"`{name}`" for name in improved_rows["campaign_variant_name"].head(5).tolist())
            + "."
            if not improved_rows.empty
            else "2. Variantes qui ameliorent CAGR / net PnL sans degrader excessivement le maxDD: aucune claire."
        ),
        (
            f"3. Cadre prop 50k: meilleure lecture defensive = `{prop_safe['campaign_variant_name']}` | pass flag `{bool(prop_safe['oos_pass_target_3000_usd_without_breaching_2000_dd'])}` | trailing DD `{float(prop_safe['oos_max_trailing_drawdown_observed_usd']):.2f}` | jours <= -1000 USD `{int(prop_safe['oos_days_below_minus_1000_usd'])}`."
            if prop_safe is not None
            else "3. Cadre prop 50k: aucune variante ne ressort proprement."
        ),
    ]

    if paired_skip.empty:
        lines.append("4. Impact de `skip_trade_if_too_small`: effet nul ou non observable sur cet echantillon.")
    else:
        lines.append(
            "4. Impact de `skip_trade_if_too_small`: "
            f"forcer 1 contrat change en moyenne l'OOS net PnL de `{float(paired_skip['oos_net_pnl_delta_force_minus_skip'].mean()):+.2f}` USD, "
            f"le maxDD OOS de `{float(paired_skip['oos_max_dd_delta_force_minus_skip'].mean()):+.2f}` USD, "
            f"et ajoute `{float(paired_skip['oos_trade_delta_force_minus_skip'].mean()):+.2f}` trade(s) OOS."
        )

    if cap_best.empty:
        lines.append("5. Impact du cap `max_contracts`: non observable.")
    else:
        cap_fragments = [
            f"`cap={int(row.max_contracts)}` -> `{row.campaign_variant_name}` net `{float(row.oos_net_pnl_usd):.2f}` / Sharpe `{float(row.oos_sharpe):.3f}` / cap-hit `{100.0 * float(row.oos_cap_hit_rate):.1f}%`"
            for row in cap_best.itertuples()
        ]
        lines.append("5. Impact du cap `max_contracts`: " + "; ".join(cap_fragments) + ".")

    if recommended is None:
        lines.append("6. Stabilite de la courbe d'equity: le sizing augmente surtout la dispersion sans produire de vainqueur OOS assez propre.")
    else:
        vol_delta = float(recommended["oos_annualized_vol_pct"]) - float(baseline["oos_annualized_vol_pct"])
        dd_delta = float(recommended["oos_max_drawdown_usd"]) - float(baseline["oos_max_drawdown_usd"])
        lines.append(
            "6. Stabilite de la courbe d'equity: "
            f"`{recommended['campaign_variant_name']}` vs baseline = Sharpe `{float(recommended['oos_sharpe']) - float(baseline['oos_sharpe']):+.3f}`, "
            f"vol annualisee `{vol_delta:+.2f}` pts, maxDD `{dd_delta:+.2f}` USD."
        )

    lines.extend(["", "## Recommendation"])
    if final_verdict == "ne_pas_retenir":
        lines.append("- Verdict final: `ne pas retenir`.")
        lines.append("- Raison: aucune variante risk-based ne bat la baseline OOS avec un compromis rendement / drawdown suffisamment propre.")
    else:
        lines.append(f"- Verdict final: `retenir {recommended['campaign_variant_name']}`.")
        lines.append(
            f"- Variante recommandee: `{recommended['campaign_variant_name']}` | {_format_variant_brief(recommended)}."
        )
        lines.append(
            f"- Variante agressive: `{aggressive['campaign_variant_name']}` | {_format_variant_brief(aggressive)}."
            if aggressive is not None
            else "- Variante agressive: `n/a`."
        )
        lines.append(
            f"- Variante prop-safe: `{prop_safe['campaign_variant_name']}` | {_format_variant_brief(prop_safe)}."
            if prop_safe is not None
            else "- Variante prop-safe: `n/a`."
        )

    report_path = output_dir / "final_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    verdict = {
        "final_verdict": final_verdict,
        "recommended_variant": None if recommended is None else str(recommended["campaign_variant_name"]),
        "aggressive_variant": None if aggressive is None else str(aggressive["campaign_variant_name"]),
        "prop_safe_variant": None if prop_safe is None else str(prop_safe["campaign_variant_name"]),
    }
    _json_dump(output_dir / "final_verdict.json", verdict)
    return verdict


def run_campaign(
    *,
    output_root: Path | None = None,
    input_path: Path | None = None,
    initial_capital_usd: float = DEFAULT_INITIAL_CAPITAL_USD,
    risk_pcts: tuple[float, ...] = DEFAULT_RISK_PCTS,
    max_contracts_grid: tuple[int, ...] = DEFAULT_MAX_CONTRACTS,
    skip_flags: tuple[bool, ...] = DEFAULT_SKIP_FLAGS,
    base_variant_name: str | None = None,
    reference_v3_dir: Path | None = None,
) -> Path:
    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) if output_root is not None else EXPORTS_DIR / f"{DEFAULT_OUTPUT_PREFIX}{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_alpha_variant, resolved_reference_dir, resolved_reference_name = _resolve_base_alpha_variant(
        base_variant_name=base_variant_name,
        reference_v3_dir=reference_v3_dir,
    )
    dataset_path = Path(input_path) if input_path is not None else None
    raw = load_symbol_data(DEFAULT_SYMBOL, input_paths=None if dataset_path is None else {DEFAULT_SYMBOL: dataset_path})
    bars = resample_rth_1h(raw)
    if bars.empty:
        raise ValueError(f"No RTH 1h bars are available for the {DEFAULT_SYMBOL} campaign.")

    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")
    bars["session_date"] = bars["timestamp"].dt.date
    is_sessions, oos_sessions = split_sessions(bars[["session_date"]].copy())
    phase_lookup = _phase_map(is_sessions, oos_sessions)

    features = prepare_volume_climax_pullback_v2_features(bars)
    signal_df = build_volume_climax_pullback_v2_signal_frame(features, base_alpha_variant)
    oos_signal_df = _subset_by_sessions(signal_df, oos_sessions)

    execution_model, instrument = build_execution_model_for_profile(symbol=DEFAULT_SYMBOL, profile_name="repo_realistic")
    variants = _campaign_variants(
        initial_capital_usd=float(initial_capital_usd),
        risk_pcts=tuple(risk_pcts),
        max_contracts_grid=tuple(max_contracts_grid),
        skip_flags=tuple(skip_flags),
    )

    summary_rows: list[dict[str, Any]] = []
    trade_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []
    prop_rows: list[dict[str, Any]] = []
    sizing_rows: list[pd.DataFrame] = []

    for order_index, campaign_variant in enumerate(variants):
        full_result = run_volume_climax_pullback_v2_backtest(
            signal_df=signal_df,
            variant=base_alpha_variant,
            execution_model=execution_model,
            instrument=instrument,
            position_sizing=campaign_variant.position_sizing,
        )
        oos_result = run_volume_climax_pullback_v2_backtest(
            signal_df=oos_signal_df,
            variant=base_alpha_variant,
            execution_model=execution_model,
            instrument=instrument,
            position_sizing=campaign_variant.position_sizing,
        )

        evaluations: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]] = {}
        for scope_name, result, scoped_signal_df, scoped_sessions in (
            ("full", full_result, signal_df, list(pd.to_datetime(signal_df["session_date"]).dt.date.unique())),
            ("oos_only", oos_result, oos_signal_df, oos_sessions),
        ):
            scoped_trades = result.trades.copy()
            scoped_trades["campaign_variant_name"] = campaign_variant.campaign_variant_name
            scoped_trades["scope"] = scope_name
            scoped_trades["alpha_variant_name"] = base_alpha_variant.name
            if not scoped_trades.empty:
                scoped_trades["phase"] = pd.to_datetime(scoped_trades["session_date"]).dt.date.map(phase_lookup).fillna(
                    "oos" if scope_name == "oos_only" else "unknown"
                )

            scoped_decisions = result.sizing_decisions.copy()
            scoped_decisions["campaign_variant_name"] = campaign_variant.campaign_variant_name
            scoped_decisions["scope"] = scope_name
            scoped_decisions["alpha_variant_name"] = base_alpha_variant.name
            if not scoped_decisions.empty:
                scoped_decisions["phase"] = pd.to_datetime(scoped_decisions["session_date"]).dt.date.map(phase_lookup).fillna(
                    "oos" if scope_name == "oos_only" else "unknown"
                )

            daily_results = _daily_results_from_trades(
                trades=scoped_trades,
                sessions=scoped_sessions,
                initial_capital=float(campaign_variant.initial_capital_usd),
            )
            daily_results["campaign_variant_name"] = campaign_variant.campaign_variant_name
            daily_results["scope"] = scope_name
            daily_results["alpha_variant_name"] = base_alpha_variant.name
            daily_results["phase"] = pd.to_datetime(daily_results["session_date"]).dt.date.map(phase_lookup).fillna(
                "oos" if scope_name == "oos_only" else "unknown"
            )

            evaluations[scope_name] = (scoped_trades, scoped_decisions, daily_results, scoped_sessions)
            trade_rows.append(scoped_trades)
            daily_rows.append(daily_results)
            sizing_rows.append(scoped_decisions)

        full_summary = _summarize_scope(
            trades=evaluations["full"][0],
            signal_df=signal_df,
            sessions=evaluations["full"][3],
            daily_results=evaluations["full"][2],
            sizing_decisions=evaluations["full"][1],
            initial_capital=float(campaign_variant.initial_capital_usd),
        )
        oos_summary = _summarize_scope(
            trades=evaluations["oos_only"][0],
            signal_df=oos_signal_df,
            sessions=evaluations["oos_only"][3],
            daily_results=evaluations["oos_only"][2],
            sizing_decisions=evaluations["oos_only"][1],
            initial_capital=float(campaign_variant.initial_capital_usd),
        )

        for scope_name, scope_summary in (("full", full_summary), ("oos", oos_summary)):
            prop_rows.append(
                {
                    "campaign_variant_name": campaign_variant.campaign_variant_name,
                    "scope": scope_name,
                    "sizing_mode": campaign_variant.sizing_mode,
                    "risk_pct": campaign_variant.risk_pct,
                    "max_contracts": campaign_variant.max_contracts,
                    "skip_trade_if_too_small": campaign_variant.skip_trade_if_too_small,
                    "fixed_contracts": campaign_variant.fixed_contracts,
                    "initial_capital_usd": campaign_variant.initial_capital_usd,
                    "worst_trade_loss_usd": scope_summary["worst_trade_loss_usd"],
                    "worst_day_pnl_usd": scope_summary["worst_day_pnl_usd"],
                    "max_daily_drawdown_usd": scope_summary["max_daily_drawdown_usd"],
                    "days_below_minus_500_usd": scope_summary["days_below_minus_500_usd"],
                    "days_below_minus_1000_usd": scope_summary["days_below_minus_1000_usd"],
                    "trades_loss_gt_250_usd": scope_summary["trades_loss_gt_250_usd"],
                    "trades_loss_gt_500_usd": scope_summary["trades_loss_gt_500_usd"],
                    "pass_target_3000_usd_without_breaching_2000_dd": scope_summary["pass_target_3000_usd_without_breaching_2000_dd"],
                    "days_to_target_3000_usd": scope_summary["days_to_target_3000_usd"],
                    "max_trailing_drawdown_observed_usd": scope_summary["max_trailing_drawdown_observed_usd"],
                    "max_static_drawdown_observed_usd": scope_summary["max_static_drawdown_observed_usd"],
                }
            )

        summary_rows.append(
            {
                "order_index": order_index,
                "campaign_variant_name": campaign_variant.campaign_variant_name,
                "alpha_variant_name": base_alpha_variant.name,
                "sizing_mode": campaign_variant.sizing_mode,
                "fixed_contracts": campaign_variant.fixed_contracts,
                "risk_pct": campaign_variant.risk_pct,
                "max_contracts": campaign_variant.max_contracts,
                "skip_trade_if_too_small": campaign_variant.skip_trade_if_too_small,
                "initial_capital_usd": campaign_variant.initial_capital_usd,
                **{f"full_{key}": value for key, value in full_summary.items()},
                **{f"oos_{key}": value for key, value in oos_summary.items()},
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["order_index", "campaign_variant_name"]).reset_index(drop=True)
    baseline = summary.loc[summary["campaign_variant_name"] == "fixed_1_contract"].iloc[0]
    for prefix in ("full", "oos"):
        summary[f"{prefix}_net_pnl_delta_vs_baseline_usd"] = pd.to_numeric(summary[f"{prefix}_net_pnl_usd"], errors="coerce") - float(
            baseline[f"{prefix}_net_pnl_usd"]
        )
        summary[f"{prefix}_cagr_delta_vs_baseline_pct"] = pd.to_numeric(summary[f"{prefix}_cagr_pct"], errors="coerce") - float(
            baseline[f"{prefix}_cagr_pct"]
        )
        summary[f"{prefix}_sharpe_delta_vs_baseline"] = pd.to_numeric(summary[f"{prefix}_sharpe"], errors="coerce") - float(
            baseline[f"{prefix}_sharpe"]
        )
        summary[f"{prefix}_max_drawdown_delta_vs_baseline_usd"] = pd.to_numeric(
            summary[f"{prefix}_max_drawdown_usd"], errors="coerce"
        ) - float(baseline[f"{prefix}_max_drawdown_usd"])

    summary.to_csv(output_dir / "summary_by_variant.csv", index=False)
    trade_export = _concat_non_empty_frames(trade_rows)
    daily_export = _concat_non_empty_frames(daily_rows)
    sizing_export = _concat_non_empty_frames(sizing_rows)
    trade_export.to_csv(output_dir / "trades_by_variant.csv", index=False)
    daily_export.to_csv(output_dir / "daily_equity_by_variant.csv", index=False)
    pd.DataFrame(prop_rows).to_csv(output_dir / "prop_constraints_summary.csv", index=False)
    sizing_export.to_csv(output_dir / "sizing_decisions_by_variant.csv", index=False)

    verdict = _build_final_report(
        output_dir=output_dir,
        summary=summary,
        base_alpha_variant=base_alpha_variant,
        reference_variant_name=resolved_reference_name,
        reference_v3_dir=resolved_reference_dir,
        dataset_path=dataset_path,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
    )

    _json_dump(
        output_dir / "run_metadata.json",
        {
            "generated_at": datetime.now().isoformat(),
            "symbol": DEFAULT_SYMBOL,
            "dataset_path": dataset_path,
            "reference_v3_dir": resolved_reference_dir,
            "resolved_base_alpha_variant_name": resolved_reference_name or base_alpha_variant.name,
            "resolved_base_alpha_variant": asdict(base_alpha_variant),
            "initial_capital_usd": initial_capital_usd,
            "risk_pcts": list(risk_pcts),
            "max_contracts_grid": list(max_contracts_grid),
            "skip_flags": list(skip_flags),
            "session_count_full": int(len(is_sessions) + len(oos_sessions)),
            "session_count_is": int(len(is_sessions)),
            "session_count_oos": int(len(oos_sessions)),
            "variant_count": int(len(summary)),
            "verdict": verdict,
        },
    )
    return output_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--base-variant-name", type=str, default=None)
    parser.add_argument("--reference-v3-dir", type=Path, default=None)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    output_dir = run_campaign(
        output_root=args.output_root,
        input_path=args.input_path,
        base_variant_name=args.base_variant_name,
        reference_v3_dir=args.reference_v3_dir,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
