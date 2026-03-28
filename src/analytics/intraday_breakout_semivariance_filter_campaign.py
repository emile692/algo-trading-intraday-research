"""Intraday realized semivariance overlay campaign for audited breakout baselines."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from src.features.semivariance import (
    add_directional_semivariance_context,
    add_realized_semivariance_features,
    add_rolling_percentile_ranks,
)


UNIVERSE = ("MNQ", "MES", "MGC", "M2K")
HORIZONS = ("30m", "60m", "90m", "session")


@dataclass(frozen=True)
class AssetBaselineConfig:
    symbol: str
    source_reference: str
    source_note: str
    baseline: BaselineSpec
    grid: SearchGrid
    aggregation_rule: str
    timeframe: str = "1m"
    dataset_path: Path | None = None


@dataclass(frozen=True)
class SemivarianceCampaignSpec:
    symbols: tuple[str, ...] = UNIVERSE
    is_fraction: float = 0.70
    semivariance_horizons: tuple[str, ...] = HORIZONS
    percentile_thresholds: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95)
    downsizing_multipliers: tuple[float, ...] = (0.75, 0.50, 0.25, 0.00)
    percentile_history: int = 252
    min_percentile_history: int = 60
    output_root: Path | None = None
    asset_baselines: dict[str, AssetBaselineConfig] = field(default_factory=dict)


@dataclass
class VariantRun:
    asset: str
    name: str
    family: str
    horizon: str
    description: str
    parameters: dict[str, Any]
    controls: pd.DataFrame
    trades: pd.DataFrame
    daily_results: pd.DataFrame
    summary_by_scope: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass
class AssetCampaignRun:
    asset: str
    baseline_config: AssetBaselineConfig
    analysis: SymbolAnalysis
    selected_sessions: set
    selected_trades: pd.DataFrame
    trade_features: pd.DataFrame
    baseline_run: VariantRun
    variant_runs: dict[str, VariantRun]
    summary_df: pd.DataFrame


@dataclass
class CampaignArtifacts:
    output_dir: Path
    spec: SemivarianceCampaignSpec
    asset_runs: dict[str, AssetCampaignRun]
    asset_summary: pd.DataFrame
    portfolio_summary: pd.DataFrame
    portfolio_sessions: list
    portfolio_is_sessions: list
    portfolio_oos_sessions: list
    best_portfolio_variant: dict[str, Any]
    final_verdict: dict[str, Any]


def _default_asset_baselines() -> dict[str, AssetBaselineConfig]:
    repo_root = ROOT
    return {
        "MNQ": AssetBaselineConfig(
            symbol="MNQ",
            source_reference=str(repo_root / "data" / "exports" / "mnq_orb_vix_vvix_validation_20260327_run" / "run_metadata.json"),
            source_note="Most recent audited leak-free official MNQ OR30 both-direction baseline reused by the VIX/VVIX validation campaign.",
            baseline=BaselineSpec(
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
            ),
            grid=SearchGrid(
                atr_periods=(25, 26, 27, 28, 29, 30),
                q_lows_pct=(25, 26, 27, 28, 29, 30),
                q_highs_pct=(90, 91, 92, 93, 94, 95),
                aggregation_rules=("majority_50", "consensus_75", "unanimity_100"),
            ),
            aggregation_rule="majority_50",
        ),
        "MES": AssetBaselineConfig(
            symbol="MES",
            source_reference=str(repo_root / "notebooks" / "orb_MES_final_ensemble_validation.ipynb"),
            source_note="Final MES ensemble validation notebook baseline.",
            baseline=BaselineSpec(
                or_minutes=15,
                opening_time="09:30:00",
                direction="long",
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
            ),
            grid=SearchGrid(
                atr_periods=(25, 26, 27, 28, 29, 30),
                q_lows_pct=(25, 26, 27, 28, 29, 30),
                q_highs_pct=(90, 91, 92, 93, 94, 95),
                aggregation_rules=("majority_50", "consensus_75", "unanimity_100"),
            ),
            aggregation_rule="majority_50",
        ),
        "MGC": AssetBaselineConfig(
            symbol="MGC",
            source_reference=str(repo_root / "notebooks" / "orb_MGC_final_ensemble_validation.ipynb"),
            source_note="Final MGC ensemble validation notebook baseline.",
            baseline=BaselineSpec(
                or_minutes=30,
                opening_time="09:30:00",
                direction="long",
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
            ),
            grid=SearchGrid(
                atr_periods=(20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30),
                q_lows_pct=(25, 26, 27, 28, 29, 30),
                q_highs_pct=(95, 96, 97, 98, 99, 100),
                aggregation_rules=("majority_50", "consensus_75", "unanimity_100"),
            ),
            aggregation_rule="majority_50",
        ),
        "M2K": AssetBaselineConfig(
            symbol="M2K",
            source_reference=str(repo_root / "notebooks" / "orb_M2K_final_ensemble_validation.ipynb"),
            source_note="Final M2K ensemble validation notebook baseline.",
            baseline=BaselineSpec(
                or_minutes=15,
                opening_time="09:30:00",
                direction="long",
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
            ),
            grid=SearchGrid(
                atr_periods=(25, 26, 27, 28, 29, 30),
                q_lows_pct=(25, 26, 27, 28, 29, 30),
                q_highs_pct=(90, 91, 92, 93, 94, 95),
                aggregation_rules=("majority_50", "consensus_75", "unanimity_100"),
            ),
            aggregation_rule="majority_50",
        ),
    }


def default_campaign_spec(*, output_root: Path | None = None) -> SemivarianceCampaignSpec:
    return SemivarianceCampaignSpec(
        output_root=output_root,
        asset_baselines=_default_asset_baselines(),
    )


def smoke_campaign_spec(*, output_root: Path | None = None) -> SemivarianceCampaignSpec:
    spec = default_campaign_spec(output_root=output_root)
    return SemivarianceCampaignSpec(
        symbols=spec.symbols,
        is_fraction=spec.is_fraction,
        semivariance_horizons=("30m", "session"),
        percentile_thresholds=(0.85, 0.90),
        downsizing_multipliers=(0.50, 0.25, 0.00),
        percentile_history=63,
        min_percentile_history=20,
        output_root=spec.output_root,
        asset_baselines=spec.asset_baselines,
    )


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
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps({key: _serialize_value(value) for key, value in payload.items()}, indent=2), encoding="utf-8")


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return float(default)
    value = numerator / denominator
    return float(value) if math.isfinite(value) else float(default)


def _subset_by_sessions(frame: pd.DataFrame, sessions: list | set) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    session_set = set(pd.to_datetime(pd.Index(list(sessions))).date)
    out = frame.copy()
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.date
    return out.loc[out["session_date"].isin(session_set)].copy().reset_index(drop=True)


def _split_sessions(all_sessions: list, is_fraction: float) -> tuple[list, list]:
    if len(all_sessions) < 2:
        return list(all_sessions), []
    split_idx = int(len(all_sessions) * float(is_fraction))
    split_idx = max(1, min(len(all_sessions) - 1, split_idx))
    return list(all_sessions[:split_idx]), list(all_sessions[split_idx:])


def _selected_ensemble_sessions(analysis: SymbolAnalysis, aggregation_rule: str) -> set:
    point_pass = analyze_symbol_cache_pass_matrix(analysis)
    pass_cols = [column for column in point_pass.columns if column.startswith("pass__")]
    if not pass_cols:
        return set()
    scored = point_pass.copy()
    scored["consensus_score"] = scored[pass_cols].sum(axis=1) / float(len(pass_cols))
    threshold = resolve_aggregation_threshold(aggregation_rule)
    return set(pd.to_datetime(scored.loc[scored["consensus_score"] >= threshold, "session_date"]).dt.date)


def _daily_results_from_trades(trades: pd.DataFrame, sessions: list, initial_capital: float) -> pd.DataFrame:
    session_index = pd.Index(pd.to_datetime(pd.Index(sessions)).date)
    daily = pd.DataFrame({"session_date": session_index})
    if trades.empty:
        daily["daily_pnl_usd"] = 0.0
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
                daily_trade_count=("trade_id", "count"),
                daily_loss_count=("loss_trade", "sum"),
            )
        )
        daily = daily.merge(grouped, on="session_date", how="left").fillna(
            {"daily_pnl_usd": 0.0, "daily_trade_count": 0, "daily_loss_count": 0}
        )

    daily = daily.sort_values("session_date").reset_index(drop=True)
    daily["equity"] = float(initial_capital) + pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").cumsum()
    daily["peak_equity"] = daily["equity"].cummax()
    daily["drawdown_usd"] = daily["equity"] - daily["peak_equity"]
    return daily


def _negative_streak_lengths(values: pd.Series) -> list[int]:
    streaks: list[int] = []
    current = 0
    for value in pd.Series(values, dtype=float).fillna(0.0):
        if value < 0:
            current += 1
        elif current > 0:
            streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return streaks


def _sortino_ratio(daily_pnl: pd.Series, capital: float) -> float:
    if capital <= 0:
        return 0.0
    returns = pd.Series(daily_pnl, dtype=float) / capital
    downside = returns[returns < 0]
    if len(returns) < 2 or downside.empty:
        return 0.0
    downside_std = float(np.sqrt(np.mean(np.square(downside))))
    if downside_std <= 0:
        return 0.0
    return float((returns.mean() / downside_std) * math.sqrt(252.0))


def _trades_per_month(n_trades: int, n_sessions: int, trading_days_per_month: float = 21.0) -> float:
    months = float(n_sessions) / float(trading_days_per_month) if n_sessions > 0 else 0.0
    if months <= 0:
        return 0.0
    return float(n_trades) / months


def _summarize_scope(
    trades: pd.DataFrame,
    sessions: list,
    *,
    initial_capital: float,
) -> dict[str, Any]:
    metrics = compute_metrics(trades, session_dates=sessions, initial_capital=initial_capital)
    daily = _daily_results_from_trades(trades, sessions, initial_capital=initial_capital)
    daily_pnl = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0)
    n_days_traded = int((pd.to_numeric(daily["daily_trade_count"], errors="coerce").fillna(0.0) > 0).sum())
    trade_count = int(metrics.get("n_trades", 0))
    max_dd = float(metrics.get("max_drawdown", 0.0))
    return {
        "net_pnl": float(metrics.get("cumulative_pnl", 0.0)),
        "sharpe": float(metrics.get("sharpe_ratio", 0.0)),
        "sortino": _sortino_ratio(daily_pnl, initial_capital),
        "max_drawdown": max_dd,
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "win_rate": float(metrics.get("win_rate", 0.0)),
        "average_trade": float(metrics.get("expectancy", 0.0)),
        "trade_count": trade_count,
        "n_days_traded": n_days_traded,
        "participation_rate": float(metrics.get("percent_of_days_traded", 0.0)),
        "trades_per_month": _trades_per_month(trade_count, len(sessions)),
        "longest_losing_streak_trade": int(metrics.get("longest_loss_streak", 0)),
        "longest_losing_streak_daily": int(max(_negative_streak_lengths(daily_pnl), default=0)),
    }


def _build_summary_by_scope(
    trades: pd.DataFrame,
    *,
    all_sessions: list,
    is_sessions: list,
    oos_sessions: list,
    initial_capital: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, sessions in (("overall", all_sessions), ("is", is_sessions), ("oos", oos_sessions)):
        rows.append(
            {
                "scope": scope,
                **_summarize_scope(
                    _subset_by_sessions(trades, sessions),
                    sessions,
                    initial_capital=initial_capital,
                ),
            }
        )
    return pd.DataFrame(rows)


def _scope_value(summary_by_scope: pd.DataFrame, scope: str, column: str) -> Any:
    row = summary_by_scope.loc[summary_by_scope["scope"] == scope]
    if row.empty:
        return np.nan
    return row.iloc[0].get(column, np.nan)


def _threshold_tag(value: float) -> str:
    return f"t{int(round(float(value) * 100.0)):02d}"


def _multiplier_tag(value: float) -> str:
    return f"m{int(round(float(value) * 100.0)):02d}"


def _variant_name(family: str, horizon: str, threshold: float, multiplier: float | None = None) -> str:
    base = f"{family}__{horizon}__{_threshold_tag(threshold)}"
    if multiplier is not None:
        base = f"{base}__{_multiplier_tag(multiplier)}"
    return base


def _apply_trade_multipliers(
    nominal_trades: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    account_size_usd: float,
    base_risk_pct: float,
    tick_value_usd: float,
    point_value_usd: float,
    commission_per_side_usd: float,
) -> pd.DataFrame:
    if nominal_trades.empty or controls.empty:
        return pd.DataFrame(columns=nominal_trades.columns)

    trades = nominal_trades.copy()
    trades["session_date"] = pd.to_datetime(trades["session_date"]).dt.date
    control_map = dict(zip(pd.to_datetime(controls["session_date"]).dt.date, controls["risk_multiplier"]))
    trades["risk_multiplier"] = trades["session_date"].map(control_map).fillna(0.0).astype(float)
    trades = trades.loc[trades["risk_multiplier"] > 0.0].copy()
    if trades.empty:
        return trades

    risk_budget_base = float(account_size_usd) * float(base_risk_pct) / 100.0
    per_contract_risk = pd.to_numeric(trades["risk_per_contract_usd"], errors="coerce")
    scaled_budget = risk_budget_base * pd.to_numeric(trades["risk_multiplier"], errors="coerce")
    trades["quantity"] = np.floor(scaled_budget / per_contract_risk).astype(int)
    trades = trades.loc[trades["quantity"] >= 1].copy()
    if trades.empty:
        return trades

    fees_per_contract = 2.0 * float(commission_per_side_usd)
    quantity = pd.to_numeric(trades["quantity"], errors="coerce")
    trades["risk_per_trade_pct"] = float(base_risk_pct) * pd.to_numeric(trades["risk_multiplier"], errors="coerce")
    trades["risk_budget_usd"] = risk_budget_base * pd.to_numeric(trades["risk_multiplier"], errors="coerce")
    trades["actual_risk_usd"] = quantity * per_contract_risk.loc[trades.index]
    trades["trade_risk_usd"] = trades["actual_risk_usd"]
    trades["pnl_usd"] = pd.to_numeric(trades["pnl_ticks"], errors="coerce") * float(tick_value_usd) * quantity
    trades["fees"] = fees_per_contract * quantity
    trades["net_pnl_usd"] = pd.to_numeric(trades["pnl_usd"], errors="coerce") - pd.to_numeric(trades["fees"], errors="coerce")
    trades["notional_usd"] = pd.to_numeric(trades["entry_price"], errors="coerce") * float(point_value_usd) * quantity
    trades["leverage_used"] = trades["notional_usd"] / float(account_size_usd)
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    trades["trade_id"] = np.arange(1, len(trades) + 1)
    return trades


def _build_variant_controls(
    trade_features: pd.DataFrame,
    *,
    family: str,
    horizon: str,
    threshold: float,
    multiplier: float | None,
) -> pd.DataFrame:
    controls = trade_features[
        [
            "asset",
            "session_date",
            "phase",
            "trade_id",
            "breakout_side",
            f"adverse_pct_{horizon}",
            f"rv_pct_{horizon}",
            f"rs_plus_pct_{horizon}",
            f"rs_minus_pct_{horizon}",
            f"adverse_share_{horizon}",
        ]
    ].copy()
    controls = controls.rename(
        columns={
            f"adverse_pct_{horizon}": "adverse_pct",
            f"rv_pct_{horizon}": "rv_pct",
            f"rs_plus_pct_{horizon}": "rs_plus_pct",
            f"rs_minus_pct_{horizon}": "rs_minus_pct",
            f"adverse_share_{horizon}": "adverse_share",
        }
    )
    controls["family"] = family
    controls["horizon"] = horizon
    controls["threshold"] = float(threshold)
    controls["size_multiplier_setting"] = np.nan if multiplier is None else float(multiplier)

    adverse_flag = pd.to_numeric(controls["adverse_pct"], errors="coerce") >= float(threshold)
    chop_flag = (
        (pd.to_numeric(controls["rv_pct"], errors="coerce") >= float(threshold))
        & (pd.to_numeric(controls["rs_plus_pct"], errors="coerce") >= float(threshold))
        & (pd.to_numeric(controls["rs_minus_pct"], errors="coerce") >= float(threshold))
    )
    missing_context = (
        pd.to_numeric(controls["adverse_pct"], errors="coerce").isna()
        | pd.to_numeric(controls["rv_pct"], errors="coerce").isna()
    )

    controls["adverse_flag"] = adverse_flag.fillna(False)
    controls["chop_flag"] = chop_flag.fillna(False)
    controls["missing_context"] = missing_context.fillna(True)

    favorable_cut = max(0.50, float(threshold) - 0.25)
    favorable_flag = (
        (pd.to_numeric(controls["adverse_pct"], errors="coerce") < favorable_cut)
        & (pd.to_numeric(controls["rv_pct"], errors="coerce") < float(threshold))
        & ~controls["chop_flag"]
    )

    if family == "adverse_hard_skip":
        risk_multiplier = np.where(controls["adverse_flag"], 0.0, 1.0)
        state = np.where(controls["adverse_flag"], "hostile", "favorable")
    elif family == "chop_hard_skip":
        risk_multiplier = np.where(controls["chop_flag"], 0.0, 1.0)
        state = np.where(controls["chop_flag"], "hostile", "favorable")
    elif family == "adverse_downsize":
        if multiplier is None:
            raise ValueError("adverse_downsize requires a multiplier.")
        risk_multiplier = np.where(controls["adverse_flag"], float(multiplier), 1.0)
        state = np.where(controls["adverse_flag"], "hostile", "favorable")
    elif family == "three_state_modulation":
        if multiplier is None:
            raise ValueError("three_state_modulation requires a multiplier.")
        hostile = controls["adverse_flag"] | controls["chop_flag"]
        neutral = ~hostile & ~favorable_flag.fillna(False)
        risk_multiplier = np.where(hostile, 0.0, np.where(neutral, float(multiplier), 1.0))
        state = np.where(hostile, "hostile", np.where(neutral, "neutral", "favorable"))
    elif family == "combined_overlay":
        if multiplier is None:
            raise ValueError("combined_overlay requires a multiplier.")
        risk_multiplier = np.where(controls["chop_flag"], 0.0, np.where(controls["adverse_flag"], float(multiplier), 1.0))
        state = np.where(controls["chop_flag"], "hostile_chop", np.where(controls["adverse_flag"], "adverse", "favorable"))
    else:
        raise ValueError(f"Unsupported family '{family}'.")

    controls["risk_multiplier"] = np.where(controls["missing_context"], 1.0, risk_multiplier).astype(float)
    controls["state"] = np.where(controls["missing_context"], "insufficient_history", state)
    controls["skip_trade"] = controls["risk_multiplier"].eq(0.0)
    controls["downscaled_trade"] = controls["risk_multiplier"].between(0.0, 1.0, inclusive="neither")
    return controls.sort_values("session_date").reset_index(drop=True)


def _variant_diagnostics(
    nominal_trades: pd.DataFrame,
    variant_trades: pd.DataFrame,
    controls: pd.DataFrame,
) -> dict[str, Any]:
    nominal = nominal_trades.copy()
    nominal["session_date"] = pd.to_datetime(nominal["session_date"]).dt.date
    control_view = controls.copy()
    control_view["session_date"] = pd.to_datetime(control_view["session_date"]).dt.date
    merged = nominal.merge(
        control_view[["session_date", "risk_multiplier", "skip_trade", "downscaled_trade", "state"]],
        on="session_date",
        how="left",
    )

    skipped = merged.loc[merged["skip_trade"].fillna(False)].copy()
    downscaled = merged.loc[merged["downscaled_trade"].fillna(False)].copy()
    realized = variant_trades.copy()
    realized["session_date"] = pd.to_datetime(realized["session_date"]).dt.date

    baseline_by_session = nominal.set_index("session_date")["net_pnl_usd"]
    realized_by_session = realized.groupby("session_date")["net_pnl_usd"].sum() if not realized.empty else pd.Series(dtype=float)
    pnl_delta = (baseline_by_session - realized_by_session.reindex(baseline_by_session.index, fill_value=0.0)).astype(float)
    downscaled_sessions = pd.Index(downscaled["session_date"]).unique().tolist()
    downscaled_delta = pnl_delta.loc[pnl_delta.index.isin(downscaled_sessions)]

    return {
        "skipped_trades": int(len(skipped)),
        "downscaled_trades": int(len(downscaled)),
        "losers_avoided": int((pd.to_numeric(skipped["net_pnl_usd"], errors="coerce") < 0).sum()),
        "winners_sacrificed": int((pd.to_numeric(skipped["net_pnl_usd"], errors="coerce") > 0).sum()),
        "losers_avoided_usd": float(-pd.to_numeric(skipped.loc[pd.to_numeric(skipped["net_pnl_usd"], errors="coerce") < 0, "net_pnl_usd"], errors="coerce").sum()),
        "winners_sacrificed_usd": float(pd.to_numeric(skipped.loc[pd.to_numeric(skipped["net_pnl_usd"], errors="coerce") > 0, "net_pnl_usd"], errors="coerce").sum()),
        "downscaled_losers": int((pd.to_numeric(downscaled["net_pnl_usd"], errors="coerce") < 0).sum()),
        "downscaled_winners": int((pd.to_numeric(downscaled["net_pnl_usd"], errors="coerce") > 0).sum()),
        "downsizing_loss_saved_usd": float(downscaled_delta[downscaled_delta < 0].abs().sum()),
        "downsizing_profit_given_up_usd": float(downscaled_delta[downscaled_delta > 0].sum()),
    }


def _build_variant_run(
    *,
    asset: str,
    nominal_trades: pd.DataFrame,
    controls: pd.DataFrame,
    family: str,
    horizon: str,
    threshold: float,
    multiplier: float | None,
    description: str,
    all_sessions: list,
    is_sessions: list,
    oos_sessions: list,
    baseline: BaselineSpec,
    instrument_spec: dict[str, Any],
) -> VariantRun:
    if family in {"adverse_downsize", "three_state_modulation", "combined_overlay"}:
        trades = _apply_trade_multipliers(
            nominal_trades=nominal_trades,
            controls=controls,
            account_size_usd=baseline.account_size_usd,
            base_risk_pct=baseline.risk_per_trade_pct,
            tick_value_usd=float(instrument_spec["tick_value_usd"]),
            point_value_usd=float(instrument_spec["point_value_usd"]),
            commission_per_side_usd=float(instrument_spec["commission_per_side_usd"]),
        )
    else:
        keep_sessions = set(pd.to_datetime(controls.loc[controls["risk_multiplier"] > 0.0, "session_date"]).dt.date)
        trades = _subset_by_sessions(nominal_trades, keep_sessions)

    daily_results = _daily_results_from_trades(trades, all_sessions, initial_capital=baseline.account_size_usd)
    summary_by_scope = _build_summary_by_scope(
        trades,
        all_sessions=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        initial_capital=baseline.account_size_usd,
    )
    diagnostics = _variant_diagnostics(nominal_trades=nominal_trades, variant_trades=trades, controls=controls)
    name = _variant_name(family, horizon, threshold, multiplier)
    return VariantRun(
        asset=asset,
        name=name,
        family=family,
        horizon=horizon,
        description=description,
        parameters={
            "threshold": float(threshold),
            "multiplier": None if multiplier is None else float(multiplier),
            "horizon": horizon,
        },
        controls=controls,
        trades=trades,
        daily_results=daily_results,
        summary_by_scope=summary_by_scope,
        diagnostics=diagnostics,
    )


def _variant_row(run: VariantRun, baseline_run: VariantRun) -> dict[str, Any]:
    row: dict[str, Any] = {
        "asset": run.asset,
        "variant_name": run.name,
        "family": run.family,
        "horizon": run.horizon,
        "description": run.description,
        "parameters_json": json.dumps({key: _serialize_value(value) for key, value in run.parameters.items()}, sort_keys=True),
        **run.diagnostics,
    }
    for scope in ("overall", "is", "oos"):
        for column in (
            "net_pnl",
            "sharpe",
            "sortino",
            "max_drawdown",
            "profit_factor",
            "win_rate",
            "average_trade",
            "trade_count",
            "n_days_traded",
            "participation_rate",
            "trades_per_month",
            "longest_losing_streak_trade",
            "longest_losing_streak_daily",
        ):
            row[f"{scope}_{column}"] = _scope_value(run.summary_by_scope, scope, column)

    baseline_oos_sharpe = float(_scope_value(baseline_run.summary_by_scope, "oos", "sharpe"))
    baseline_oos_dd = abs(float(_scope_value(baseline_run.summary_by_scope, "oos", "max_drawdown")))
    baseline_oos_trades = float(_scope_value(baseline_run.summary_by_scope, "oos", "trade_count"))
    baseline_oos_pnl = float(_scope_value(baseline_run.summary_by_scope, "oos", "net_pnl"))
    run_oos_dd = abs(float(_scope_value(run.summary_by_scope, "oos", "max_drawdown")))

    row["oos_sharpe_delta_vs_baseline"] = float(_scope_value(run.summary_by_scope, "oos", "sharpe")) - baseline_oos_sharpe
    row["oos_max_drawdown_improvement_vs_baseline"] = _safe_div(baseline_oos_dd - run_oos_dd, max(baseline_oos_dd, 1.0), default=0.0)
    row["oos_trade_retention_vs_baseline"] = _safe_div(
        float(_scope_value(run.summary_by_scope, "oos", "trade_count")),
        max(baseline_oos_trades, 1.0),
        default=0.0,
    )
    row["oos_net_pnl_retention_vs_baseline"] = _safe_div(
        float(_scope_value(run.summary_by_scope, "oos", "net_pnl")),
        baseline_oos_pnl,
        default=0.0,
    )
    return row


def _build_trade_feature_frame(
    analysis: SymbolAnalysis,
    *,
    selected_sessions: set,
    percentile_history: int,
    min_percentile_history: int,
    horizons: tuple[str, ...],
) -> pd.DataFrame:
    semivar_signal = add_realized_semivariance_features(
        analysis.signal_df,
        session_open_time=analysis.baseline.opening_time,
        rth_end_time=analysis.baseline.time_exit,
        window_minutes=(30, 60, 90),
    )
    selected_candidates = analysis.candidate_df.copy()
    selected_candidates["session_date"] = pd.to_datetime(selected_candidates["session_date"]).dt.date
    selected_candidates = selected_candidates.loc[selected_candidates["session_date"].isin(selected_sessions)].copy()

    signal_rows = semivar_signal.loc[selected_candidates["signal_index"].tolist()].copy()
    signal_rows = signal_rows.reset_index().rename(columns={"index": "signal_index"})
    signal_rows["session_date"] = pd.to_datetime(signal_rows["session_date"]).dt.date

    selected_trades = _subset_by_sessions(analysis.baseline_trades, selected_sessions)
    trade_features = selected_candidates.merge(
        signal_rows,
        on=["session_date", "signal_index"],
        how="inner",
        suffixes=("", "_signal"),
    )
    trade_features = trade_features.merge(
        selected_trades[
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
                "risk_per_contract_usd",
                "pnl_ticks",
                "entry_price",
                "trade_risk_usd",
            ]
        ],
        on="session_date",
        how="inner",
    )
    trade_features = trade_features.sort_values(["session_date", "timestamp"]).reset_index(drop=True)
    trade_features["asset"] = analysis.symbol
    trade_features["breakout_side"] = pd.Series(trade_features["direction"], dtype="string").str.lower()
    is_set = set(pd.to_datetime(pd.Index(analysis.is_sessions)).date)
    trade_features["phase"] = np.where(trade_features["session_date"].isin(is_set), "is", "oos")

    base_pct_columns: list[str] = []
    for horizon in horizons:
        base_pct_columns.extend(
            [
                f"rs_plus_{horizon}",
                f"rs_minus_{horizon}",
                f"rv_{horizon}",
                f"abs_rs_imbalance_{horizon}",
            ]
        )
    trade_features = add_rolling_percentile_ranks(
        trade_features,
        columns=tuple(base_pct_columns),
        lookback=percentile_history,
        min_history=min_percentile_history,
    )
    rename_map = {}
    for horizon in horizons:
        rename_map[f"rs_plus_{horizon}_pct"] = f"rs_plus_pct_{horizon}"
        rename_map[f"rs_minus_{horizon}_pct"] = f"rs_minus_pct_{horizon}"
        rename_map[f"rv_{horizon}_pct"] = f"rv_pct_{horizon}"
        rename_map[f"abs_rs_imbalance_{horizon}_pct"] = f"abs_rs_imbalance_pct_{horizon}"
    trade_features = trade_features.rename(columns=rename_map)
    trade_features = add_directional_semivariance_context(
        add_directional_semivariance_context(trade_features, horizons=horizons, side_col="breakout_side"),
        horizons=horizons,
        side_col="breakout_side",
    )
    return trade_features


def _build_baseline_run(
    analysis: SymbolAnalysis,
    *,
    selected_sessions: set,
) -> VariantRun:
    selected_trades = _subset_by_sessions(analysis.baseline_trades, selected_sessions)
    controls = pd.DataFrame(
        {
            "asset": analysis.symbol,
            "session_date": sorted(selected_sessions),
            "phase": np.where(
                pd.Index(sorted(selected_sessions)).isin(set(pd.to_datetime(pd.Index(analysis.is_sessions)).date)),
                "is",
                "oos",
            ),
            "risk_multiplier": 1.0,
            "skip_trade": False,
            "downscaled_trade": False,
            "state": "baseline",
        }
    )
    daily_results = _daily_results_from_trades(selected_trades, analysis.all_sessions, initial_capital=analysis.baseline.account_size_usd)
    summary_by_scope = _build_summary_by_scope(
        selected_trades,
        all_sessions=analysis.all_sessions,
        is_sessions=analysis.is_sessions,
        oos_sessions=analysis.oos_sessions,
        initial_capital=analysis.baseline.account_size_usd,
    )
    return VariantRun(
        asset=analysis.symbol,
        name="baseline",
        family="baseline",
        horizon="baseline",
        description="Unchanged audited baseline after the existing ensemble selection.",
        parameters={},
        controls=controls,
        trades=selected_trades,
        daily_results=daily_results,
        summary_by_scope=summary_by_scope,
        diagnostics={
            "skipped_trades": 0,
            "downscaled_trades": 0,
            "losers_avoided": 0,
            "winners_sacrificed": 0,
            "losers_avoided_usd": 0.0,
            "winners_sacrificed_usd": 0.0,
            "downscaled_losers": 0,
            "downscaled_winners": 0,
            "downsizing_loss_saved_usd": 0.0,
            "downsizing_profit_given_up_usd": 0.0,
        },
    )


def _run_asset_campaign(
    baseline_config: AssetBaselineConfig,
    spec: SemivarianceCampaignSpec,
) -> AssetCampaignRun:
    dataset_path = baseline_config.dataset_path or resolve_processed_dataset(baseline_config.symbol, timeframe=baseline_config.timeframe)
    analysis = analyze_symbol(
        baseline_config.symbol,
        baseline=baseline_config.baseline,
        grid=baseline_config.grid,
        is_fraction=spec.is_fraction,
        dataset_path=dataset_path,
        data_timeframe=baseline_config.timeframe,
    )
    selected_sessions = _selected_ensemble_sessions(analysis, baseline_config.aggregation_rule)
    baseline_run = _build_baseline_run(analysis, selected_sessions=selected_sessions)
    trade_features = _build_trade_feature_frame(
        analysis,
        selected_sessions=selected_sessions,
        percentile_history=spec.percentile_history,
        min_percentile_history=spec.min_percentile_history,
        horizons=spec.semivariance_horizons,
    )
    nominal_trades = baseline_run.trades.copy()

    variant_runs: dict[str, VariantRun] = {"baseline": baseline_run}
    for horizon in spec.semivariance_horizons:
        for threshold in spec.percentile_thresholds:
            hard_skip_controls = _build_variant_controls(
                trade_features,
                family="adverse_hard_skip",
                horizon=horizon,
                threshold=threshold,
                multiplier=None,
            )
            run = _build_variant_run(
                asset=analysis.symbol,
                nominal_trades=nominal_trades,
                controls=hard_skip_controls,
                family="adverse_hard_skip",
                horizon=horizon,
                threshold=threshold,
                multiplier=None,
                description=f"Skip trades when the directional adverse semivariance percentile is >= {threshold:.0%}.",
                all_sessions=analysis.all_sessions,
                is_sessions=analysis.is_sessions,
                oos_sessions=analysis.oos_sessions,
                baseline=analysis.baseline,
                instrument_spec=analysis.instrument_spec,
            )
            variant_runs[run.name] = run

            chop_controls = _build_variant_controls(
                trade_features,
                family="chop_hard_skip",
                horizon=horizon,
                threshold=threshold,
                multiplier=None,
            )
            run = _build_variant_run(
                asset=analysis.symbol,
                nominal_trades=nominal_trades,
                controls=chop_controls,
                family="chop_hard_skip",
                horizon=horizon,
                threshold=threshold,
                multiplier=None,
                description=f"Skip trades when total realized variance and both semivariances are jointly >= {threshold:.0%}.",
                all_sessions=analysis.all_sessions,
                is_sessions=analysis.is_sessions,
                oos_sessions=analysis.oos_sessions,
                baseline=analysis.baseline,
                instrument_spec=analysis.instrument_spec,
            )
            variant_runs[run.name] = run

            for multiplier in spec.downsizing_multipliers:
                if multiplier == 0.0:
                    continue
                adverse_controls = _build_variant_controls(
                    trade_features,
                    family="adverse_downsize",
                    horizon=horizon,
                    threshold=threshold,
                    multiplier=multiplier,
                )
                run = _build_variant_run(
                    asset=analysis.symbol,
                    nominal_trades=nominal_trades,
                    controls=adverse_controls,
                    family="adverse_downsize",
                    horizon=horizon,
                    threshold=threshold,
                    multiplier=multiplier,
                    description=f"Downsize to {multiplier:.0%} of nominal risk when the directional adverse semivariance percentile is >= {threshold:.0%}.",
                    all_sessions=analysis.all_sessions,
                    is_sessions=analysis.is_sessions,
                    oos_sessions=analysis.oos_sessions,
                    baseline=analysis.baseline,
                    instrument_spec=analysis.instrument_spec,
                )
                variant_runs[run.name] = run

                three_state_controls = _build_variant_controls(
                    trade_features,
                    family="three_state_modulation",
                    horizon=horizon,
                    threshold=threshold,
                    multiplier=multiplier,
                )
                run = _build_variant_run(
                    asset=analysis.symbol,
                    nominal_trades=nominal_trades,
                    controls=three_state_controls,
                    family="three_state_modulation",
                    horizon=horizon,
                    threshold=threshold,
                    multiplier=multiplier,
                    description=f"Three-state sizing: 1.0 in favorable semivariance states, {multiplier:.0%} in neutral states, 0 in hostile states.",
                    all_sessions=analysis.all_sessions,
                    is_sessions=analysis.is_sessions,
                    oos_sessions=analysis.oos_sessions,
                    baseline=analysis.baseline,
                    instrument_spec=analysis.instrument_spec,
                )
                variant_runs[run.name] = run

                if threshold < 0.85:
                    continue
                combined_controls = _build_variant_controls(
                    trade_features,
                    family="combined_overlay",
                    horizon=horizon,
                    threshold=threshold,
                    multiplier=multiplier,
                )
                run = _build_variant_run(
                    asset=analysis.symbol,
                    nominal_trades=nominal_trades,
                    controls=combined_controls,
                    family="combined_overlay",
                    horizon=horizon,
                    threshold=threshold,
                    multiplier=multiplier,
                    description=f"Skip choppy regimes and downsize adverse regimes to {multiplier:.0%} of nominal risk at threshold {threshold:.0%}.",
                    all_sessions=analysis.all_sessions,
                    is_sessions=analysis.is_sessions,
                    oos_sessions=analysis.oos_sessions,
                    baseline=analysis.baseline,
                    instrument_spec=analysis.instrument_spec,
                )
                variant_runs[run.name] = run

    summary_rows = [_variant_row(run, baseline_run) for run in variant_runs.values()]
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["asset", "oos_sharpe_delta_vs_baseline", "oos_max_drawdown_improvement_vs_baseline"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return AssetCampaignRun(
        asset=analysis.symbol,
        baseline_config=baseline_config,
        analysis=analysis,
        selected_sessions=selected_sessions,
        selected_trades=baseline_run.trades.copy(),
        trade_features=trade_features,
        baseline_run=baseline_run,
        variant_runs=variant_runs,
        summary_df=summary_df,
    )


def _portfolio_variant_row(
    name: str,
    runs: dict[str, AssetCampaignRun],
    common_sessions: list,
    is_sessions: list,
    oos_sessions: list,
) -> dict[str, Any]:
    combined_trades: list[pd.DataFrame] = []
    asset_rows: list[pd.Series] = []
    for asset, run in runs.items():
        variant = run.variant_runs[name]
        trades = _subset_by_sessions(variant.trades, common_sessions).copy()
        if not trades.empty:
            trades["asset"] = asset
            combined_trades.append(trades)
        asset_rows.append(run.summary_df.loc[run.summary_df["variant_name"] == name].iloc[0])

    portfolio_trades = pd.concat(combined_trades, ignore_index=True) if combined_trades else pd.DataFrame()
    initial_capital = sum(float(run.analysis.baseline.account_size_usd) for run in runs.values())
    summary = _build_summary_by_scope(
        portfolio_trades,
        all_sessions=common_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        initial_capital=initial_capital,
    )

    baseline_trades: list[pd.DataFrame] = []
    for asset, run in runs.items():
        base_trades = _subset_by_sessions(run.baseline_run.trades, common_sessions).copy()
        if not base_trades.empty:
            base_trades["asset"] = asset
            baseline_trades.append(base_trades)
    baseline_portfolio = pd.concat(baseline_trades, ignore_index=True) if baseline_trades else pd.DataFrame()
    baseline_summary = _build_summary_by_scope(
        baseline_portfolio,
        all_sessions=common_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        initial_capital=initial_capital,
    )

    row: dict[str, Any] = {
        "variant_name": name,
        "family": asset_rows[0]["family"],
        "horizon": asset_rows[0]["horizon"],
        "asset_count": len(runs),
    }
    for scope in ("overall", "is", "oos"):
        for column in (
            "net_pnl",
            "sharpe",
            "sortino",
            "max_drawdown",
            "profit_factor",
            "win_rate",
            "average_trade",
            "trade_count",
            "n_days_traded",
            "participation_rate",
            "trades_per_month",
            "longest_losing_streak_trade",
            "longest_losing_streak_daily",
        ):
            row[f"{scope}_{column}"] = _scope_value(summary, scope, column)

    row["oos_sharpe_delta_vs_baseline"] = float(_scope_value(summary, "oos", "sharpe")) - float(_scope_value(baseline_summary, "oos", "sharpe"))
    baseline_oos_dd = abs(float(_scope_value(baseline_summary, "oos", "max_drawdown")))
    row["oos_max_drawdown_improvement_vs_baseline"] = _safe_div(
        baseline_oos_dd - abs(float(_scope_value(summary, "oos", "max_drawdown"))),
        max(baseline_oos_dd, 1.0),
        default=0.0,
    )
    row["oos_trade_retention_vs_baseline"] = _safe_div(
        float(_scope_value(summary, "oos", "trade_count")),
        max(float(_scope_value(baseline_summary, "oos", "trade_count")), 1.0),
        default=0.0,
    )
    row["oos_net_pnl_retention_vs_baseline"] = _safe_div(
        float(_scope_value(summary, "oos", "net_pnl")),
        float(_scope_value(baseline_summary, "oos", "net_pnl")),
        default=0.0,
    )
    row["oos_assets_sharpe_improved"] = int(sum(float(asset_row["oos_sharpe_delta_vs_baseline"]) > 0.0 for asset_row in asset_rows))
    row["oos_assets_dd_improved"] = int(sum(float(asset_row["oos_max_drawdown_improvement_vs_baseline"]) > 0.0 for asset_row in asset_rows))
    row["oos_assets_with_value"] = int(
        sum(
            (float(asset_row["oos_sharpe_delta_vs_baseline"]) > 0.0)
            and (float(asset_row["oos_max_drawdown_improvement_vs_baseline"]) > 0.0)
            and (float(asset_row["oos_trade_retention_vs_baseline"]) >= 0.60)
            for asset_row in asset_rows
        )
    )
    return row


def _portfolio_verdict(row: pd.Series) -> str:
    if row["variant_name"] == "baseline":
        return "baseline_reference"
    if (
        float(row["oos_sharpe_delta_vs_baseline"]) > 0.0
        and float(row["oos_max_drawdown_improvement_vs_baseline"]) > 0.0
        and float(row["oos_trade_retention_vs_baseline"]) >= 0.60
        and int(row["oos_assets_with_value"]) >= 3
    ):
        return "interesting"
    if float(row["oos_max_drawdown_improvement_vs_baseline"]) > 0.0 and float(row["oos_trade_retention_vs_baseline"]) >= 0.60:
        return "protective_but_mixed"
    if float(row["oos_trade_retention_vs_baseline"]) < 0.45:
        return "cuts_too_much"
    return "no_value"


def _best_portfolio_variant(portfolio_summary: pd.DataFrame) -> dict[str, Any]:
    interesting = portfolio_summary.loc[portfolio_summary["verdict"] == "interesting"].copy()
    if not interesting.empty:
        return interesting.sort_values(
            ["oos_sharpe_delta_vs_baseline", "oos_max_drawdown_improvement_vs_baseline", "oos_trade_retention_vs_baseline"],
            ascending=[False, False, False],
        ).iloc[0].to_dict()
    non_baseline = portfolio_summary.loc[portfolio_summary["variant_name"] != "baseline"].copy()
    if non_baseline.empty:
        return portfolio_summary.iloc[0].to_dict()
    return non_baseline.sort_values(
        ["oos_sharpe_delta_vs_baseline", "oos_max_drawdown_improvement_vs_baseline", "oos_trade_retention_vs_baseline"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()


def _build_final_verdict(
    *,
    best_portfolio_variant: dict[str, Any],
    asset_summary: pd.DataFrame,
) -> dict[str, Any]:
    best_name = str(best_portfolio_variant["variant_name"])
    best_asset_rows = asset_summary.loc[asset_summary["variant_name"] == best_name].copy()
    m2k_row = best_asset_rows.loc[best_asset_rows["asset"] == "M2K"]
    m2k_helped_more = bool(
        not m2k_row.empty and float(m2k_row.iloc[0]["oos_sharpe_delta_vs_baseline"]) == float(best_asset_rows["oos_sharpe_delta_vs_baseline"].max())
    )
    credible = bool(best_portfolio_variant.get("verdict") == "interesting")

    return {
        "credible_overlay": credible,
        "best_variant_name": best_name,
        "best_variant_family": str(best_portfolio_variant["family"]),
        "best_variant_horizon": str(best_portfolio_variant["horizon"]),
        "best_variant_oos_sharpe_delta_vs_baseline": float(best_portfolio_variant["oos_sharpe_delta_vs_baseline"]),
        "best_variant_oos_max_drawdown_improvement_vs_baseline": float(best_portfolio_variant["oos_max_drawdown_improvement_vs_baseline"]),
        "best_variant_oos_trade_retention_vs_baseline": float(best_portfolio_variant["oos_trade_retention_vs_baseline"]),
        "assets_with_value_on_best_variant": int(best_portfolio_variant["oos_assets_with_value"]),
        "m2k_helped_more_than_others": m2k_helped_more,
    }


def _write_report(
    output_path: Path,
    *,
    spec: SemivarianceCampaignSpec,
    asset_runs: dict[str, AssetCampaignRun],
    asset_summary: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    best_portfolio_variant: dict[str, Any],
    final_verdict: dict[str, Any],
    portfolio_sessions: list,
    portfolio_is_sessions: list,
    portfolio_oos_sessions: list,
) -> None:
    baseline_row = portfolio_summary.loc[portfolio_summary["variant_name"] == "baseline"].iloc[0]
    best_name = str(best_portfolio_variant["variant_name"])
    best_asset_rows = asset_summary.loc[asset_summary["variant_name"] == best_name].copy()

    lines = [
        "# Intraday Breakout Semivariance Filter Campaign",
        "",
        "## Methodology",
        "",
        "- Objective: test ex-ante realized semivariance as a meta-signal overlay on top of the existing audited breakout baselines, without changing alpha logic or execution assumptions.",
        "- Universe: `MNQ`, `MES`, `MGC`, `M2K`.",
        "- Data: latest processed `1m` parquet per asset from the repo.",
        "- Fixed horizons: trailing `30m`, `60m`, `90m` semivariance inside the current continuous futures session.",
        "- Session horizon: RTH `09:30` to signal timestamp only.",
        f"- Percentile calibration: rolling prior-trade history only, lookback `{spec.percentile_history}` baseline-qualified trades, minimum history `{spec.min_percentile_history}` trades.",
        "- Percentile inputs: `rs_plus`, `rs_minus`, `rv`, and `abs(rs_imbalance)`.",
        "- Directional adverse mapping: long -> `rs_minus`, short -> `rs_plus`.",
        f"- Equal-weight portfolio: four standalone `50k` sleeves aggregated only on the common four-asset overlap window (`{len(portfolio_sessions)}` sessions, `{len(portfolio_is_sessions)}` IS / `{len(portfolio_oos_sessions)}` OOS).",
        "",
        "## Audited Baselines Used",
        "",
    ]
    for asset in spec.symbols:
        baseline = asset_runs[asset].baseline_config
        lines.append(
            f"- `{asset}`: source `{baseline.source_reference}`, rule `{baseline.aggregation_rule}`, OR `{baseline.baseline.or_minutes}m`, direction `{baseline.baseline.direction}`, grid `{list(baseline.grid.atr_periods)} x {list(baseline.grid.q_lows_pct)} x {list(baseline.grid.q_highs_pct)}`."
        )

    lines.extend(
        [
            "",
            "## Variants Tested",
            "",
            "- `adverse_hard_skip`: skip when directional adverse semivariance percentile is above threshold.",
            "- `chop_hard_skip`: skip when `rv`, `rs_plus`, and `rs_minus` are all jointly elevated.",
            "- `adverse_downsize`: keep the trade but reduce size on adverse regimes.",
            "- `three_state_modulation`: favorable `1.0x`, neutral reduced, hostile `0.0x` using semivariance only.",
            "- `combined_overlay`: chop skip plus adverse downsizing.",
            "",
            "## Portfolio OOS Snapshot",
            "",
            f"- Baseline OOS Sharpe: `{float(baseline_row['oos_sharpe']):.3f}` | MaxDD: `{float(baseline_row['oos_max_drawdown']):.2f}` | Trades: `{int(baseline_row['oos_trade_count'])}`.",
            f"- Best tested variant: `{best_name}` | family `{best_portfolio_variant['family']}` | horizon `{best_portfolio_variant['horizon']}`.",
            f"- Best variant OOS Sharpe: `{float(best_portfolio_variant['oos_sharpe']):.3f}` | Sharpe delta vs baseline: `{float(best_portfolio_variant['oos_sharpe_delta_vs_baseline']):.3f}`.",
            f"- Best variant OOS MaxDD: `{float(best_portfolio_variant['oos_max_drawdown']):.2f}` | DD improvement vs baseline: `{float(best_portfolio_variant['oos_max_drawdown_improvement_vs_baseline']):.2%}`.",
            f"- Best variant OOS trade retention vs baseline: `{float(best_portfolio_variant['oos_trade_retention_vs_baseline']):.2%}`.",
            f"- Assets improved on best variant: `{int(best_portfolio_variant['oos_assets_with_value'])}` / 4.",
            "",
            "## Per-Asset OOS Results For Best Portfolio Variant",
            "",
        ]
    )
    for _, row in best_asset_rows.sort_values("asset").iterrows():
        lines.append(
            f"- `{row['asset']}`: OOS Sharpe `{float(row['oos_sharpe']):.3f}` (delta `{float(row['oos_sharpe_delta_vs_baseline']):+.3f}`), MaxDD `{float(row['oos_max_drawdown']):.2f}` (improvement `{float(row['oos_max_drawdown_improvement_vs_baseline']):+.2%}`), trade retention `{float(row['oos_trade_retention_vs_baseline']):.2%}`."
        )

    lines.extend(
        [
            "",
            "## Final Verdict",
            "",
            f"- Credible overlay for the breakout portfolio: `{'yes' if final_verdict['credible_overlay'] else 'no'}`.",
            f"- Best family: `{final_verdict['best_variant_family']}` on horizon `{final_verdict['best_variant_horizon']}`.",
            f"- M2K helped more than the other assets: `{'yes' if final_verdict['m2k_helped_more_than_others'] else 'no'}`.",
            "",
            "## Notes",
            "",
            "- `adverse_downsize` does not retest the fully redundant `0.0x` setting because that is already covered by the hard-skip family.",
            "- All overlays operate only on sessions already selected by the audited baseline ensemble for each asset.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_campaign(spec: SemivarianceCampaignSpec | None = None) -> CampaignArtifacts:
    ensure_directories()
    active_spec = spec or default_campaign_spec()
    if not active_spec.asset_baselines:
        active_spec = SemivarianceCampaignSpec(
            symbols=active_spec.symbols,
            is_fraction=active_spec.is_fraction,
            semivariance_horizons=active_spec.semivariance_horizons,
            percentile_thresholds=active_spec.percentile_thresholds,
            downsizing_multipliers=active_spec.downsizing_multipliers,
            percentile_history=active_spec.percentile_history,
            min_percentile_history=active_spec.min_percentile_history,
            output_root=active_spec.output_root,
            asset_baselines=_default_asset_baselines(),
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = active_spec.output_root or (EXPORTS_DIR / f"intraday_breakout_semivariance_filter_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    asset_runs = {
        symbol: _run_asset_campaign(active_spec.asset_baselines[symbol], active_spec)
        for symbol in active_spec.symbols
    }
    asset_summary = pd.concat([run.summary_df for run in asset_runs.values()], ignore_index=True)

    portfolio_session_sets = [set(run.analysis.all_sessions) for run in asset_runs.values()]
    common_sessions = sorted(set.intersection(*portfolio_session_sets)) if portfolio_session_sets else []
    portfolio_is_sessions, portfolio_oos_sessions = _split_sessions(common_sessions, active_spec.is_fraction)

    variant_names = sorted(set.intersection(*(set(run.variant_runs.keys()) for run in asset_runs.values())))
    portfolio_rows = [
        _portfolio_variant_row(name, asset_runs, common_sessions, portfolio_is_sessions, portfolio_oos_sessions)
        for name in variant_names
    ]
    portfolio_summary = pd.DataFrame(portfolio_rows).sort_values(
        ["oos_sharpe_delta_vs_baseline", "oos_max_drawdown_improvement_vs_baseline", "oos_trade_retention_vs_baseline"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    portfolio_summary["verdict"] = portfolio_summary.apply(_portfolio_verdict, axis=1)

    best_portfolio_variant = _best_portfolio_variant(portfolio_summary)
    final_verdict = _build_final_verdict(
        best_portfolio_variant=best_portfolio_variant,
        asset_summary=asset_summary,
    )

    asset_summary.to_csv(output_dir / "asset_variant_results.csv", index=False)
    portfolio_summary.to_csv(output_dir / "portfolio_variant_results.csv", index=False)
    portfolio_summary.sort_values(["oos_sharpe", "oos_sharpe_delta_vs_baseline"], ascending=[False, False]).to_csv(
        output_dir / "portfolio_ranking_by_oos_sharpe.csv", index=False
    )
    portfolio_summary.sort_values(
        ["oos_max_drawdown_improvement_vs_baseline", "oos_sharpe_delta_vs_baseline"], ascending=[False, False]
    ).to_csv(output_dir / "portfolio_ranking_by_drawdown_improvement.csv", index=False)

    heatmap_ready = portfolio_summary.loc[
        portfolio_summary["family"].isin({"adverse_downsize", "three_state_modulation", "combined_overlay"})
    ].copy()
    heatmap_ready["threshold"] = heatmap_ready["variant_name"].str.extract(r"__t(\d+)")
    heatmap_ready["multiplier"] = heatmap_ready["variant_name"].str.extract(r"__m(\d+)")
    heatmap_ready.to_csv(output_dir / "portfolio_heatmap_ready.csv", index=False)

    baseline_sources = pd.DataFrame(
        [
            {
                "asset": run.asset,
                "source_reference": run.baseline_config.source_reference,
                "source_note": run.baseline_config.source_note,
                "aggregation_rule": run.baseline_config.aggregation_rule,
                "dataset_path": str(run.analysis.dataset_path),
                "baseline_json": json.dumps(asdict(run.baseline_config.baseline), sort_keys=True),
                "grid_json": json.dumps(asdict(run.baseline_config.grid), sort_keys=True),
            }
            for run in asset_runs.values()
        ]
    )
    baseline_sources.to_csv(output_dir / "baseline_registry.csv", index=False)

    for asset, run in asset_runs.items():
        run.trade_features.to_csv(output_dir / f"{asset.lower()}_baseline_trade_features.csv", index=False)

    run_metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "spec": asdict(active_spec),
        "portfolio_common_sessions": len(common_sessions),
        "portfolio_is_sessions": len(portfolio_is_sessions),
        "portfolio_oos_sessions": len(portfolio_oos_sessions),
    }
    _json_dump(output_dir / "run_metadata.json", run_metadata)
    _json_dump(output_dir / "final_verdict.json", final_verdict)
    _write_report(
        output_dir / "final_report.md",
        spec=active_spec,
        asset_runs=asset_runs,
        asset_summary=asset_summary,
        portfolio_summary=portfolio_summary,
        best_portfolio_variant=best_portfolio_variant,
        final_verdict=final_verdict,
        portfolio_sessions=common_sessions,
        portfolio_is_sessions=portfolio_is_sessions,
        portfolio_oos_sessions=portfolio_oos_sessions,
    )

    return CampaignArtifacts(
        output_dir=output_dir,
        spec=active_spec,
        asset_runs=asset_runs,
        asset_summary=asset_summary,
        portfolio_summary=portfolio_summary,
        portfolio_sessions=common_sessions,
        portfolio_is_sessions=portfolio_is_sessions,
        portfolio_oos_sessions=portfolio_oos_sessions,
        best_portfolio_variant=best_portfolio_variant,
        final_verdict=final_verdict,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the intraday breakout semivariance overlay campaign.")
    parser.add_argument("--smoke", action="store_true", help="Run a smaller smoke configuration first.")
    parser.add_argument("--output-root", type=Path, default=None, help="Explicit export directory.")
    parser.add_argument("--percentile-history", type=int, default=None, help="Override the rolling percentile lookback.")
    parser.add_argument("--min-percentile-history", type=int, default=None, help="Override the minimum percentile history.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    spec = smoke_campaign_spec(output_root=args.output_root) if args.smoke else default_campaign_spec(output_root=args.output_root)
    if args.percentile_history is not None or args.min_percentile_history is not None:
        spec = SemivarianceCampaignSpec(
            symbols=spec.symbols,
            is_fraction=spec.is_fraction,
            semivariance_horizons=spec.semivariance_horizons,
            percentile_thresholds=spec.percentile_thresholds,
            downsizing_multipliers=spec.downsizing_multipliers,
            percentile_history=args.percentile_history or spec.percentile_history,
            min_percentile_history=args.min_percentile_history or spec.min_percentile_history,
            output_root=spec.output_root,
            asset_baselines=spec.asset_baselines,
        )
    artifacts = run_campaign(spec)
    print(f"Exports written to: {artifacts.output_dir}")


if __name__ == "__main__":
    main()
