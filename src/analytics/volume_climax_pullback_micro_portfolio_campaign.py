"""Portfolio campaign for MNQ, M2K, and MES Volume Climax Pullback motors."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analytics.metrics import compute_metrics
from src.analytics.volume_climax_pullback_common import load_latest_reference_run
from src.analytics.volume_climax_pullback_mnq_risk_sizing_campaign import (
    TOPSTEP_50K_PROP,
    _annualized_vol_pct,
    _cagr_pct,
    _daily_results_from_trades,
    _json_dump,
    _prop_path_metrics,
    _sortino_ratio,
)
from src.config.paths import EXPORTS_DIR, ensure_directories

DEFAULT_INITIAL_CAPITAL_USD = 50_000.0
DEFAULT_OUTPUT_PREFIX = "volume_climax_pullback_micro_portfolio_"
DEFAULT_PORTFOLIO_RISK_CAP_USD = 250.0

MNQ_REFINEMENT_PREFIX = "volume_climax_pullback_mnq_risk_sizing_refinement_"
M2K_REFINEMENT_PREFIX = "volume_climax_pullback_m2k_risk_sizing_refinement_"
MES_REFINEMENT_PREFIX = "volume_climax_pullback_mes_risk_sizing_refinement_"


@dataclass(frozen=True)
class MotorConfigSpec:
    motor_key: str
    symbol: str
    config_label: str
    campaign_variant_name: str
    risk_pct_decimal: float
    max_contracts: int
    export_root: Path
    notes: str


@dataclass(frozen=True)
class PortfolioVariantSpec:
    portfolio_variant_name: str
    portfolio_name: str
    portfolio_family: str
    allocation_scheme: str
    config_bundle: str
    member_motor_keys: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class MotorVariantData:
    spec: MotorConfigSpec
    summary_row: pd.Series
    daily_full: pd.DataFrame
    trades_full: pd.DataFrame


def compute_portfolio_score(row: pd.Series | dict[str, Any], *, prefix: str = "oos") -> float:
    series = row if isinstance(row, pd.Series) else pd.Series(row)
    base = f"{prefix}_" if prefix and not str(prefix).endswith("_") else str(prefix)

    def _float(name: str) -> float:
        value = pd.to_numeric(series.get(f"{base}{name}"), errors="coerce")
        return float(value) if pd.notna(value) else 0.0

    pass_flag = bool(series.get(f"{base}pass_target_3000_usd_without_breaching_2000_dd", False))
    sharpe_bonus = min(max(_float("sharpe"), 0.0), 3.0) * 4.0
    net_bonus = min(max(_float("net_pnl_usd"), 0.0) / 3_000.0, 3.0) * 3.0
    dd_penalty = min(max(_float("max_drawdown_usd"), 0.0) / 2_000.0, 3.0) * 3.0
    daily_dd_penalty = min(max(_float("max_daily_drawdown_usd"), 0.0) / 1_000.0, 3.0) * 4.0
    pass_penalty = 6.0 if not pass_flag else 0.0
    return float(sharpe_bonus + net_bonus - dd_penalty - daily_dd_penalty - pass_penalty)


def build_default_motor_configs() -> dict[str, MotorConfigSpec]:
    mnq_root = load_latest_reference_run(EXPORTS_DIR, MNQ_REFINEMENT_PREFIX)
    m2k_root = load_latest_reference_run(EXPORTS_DIR, M2K_REFINEMENT_PREFIX)
    mes_root = load_latest_reference_run(EXPORTS_DIR, MES_REFINEMENT_PREFIX)
    return {
        "MNQ_default": MotorConfigSpec(
            motor_key="MNQ_default",
            symbol="MNQ",
            config_label="default",
            campaign_variant_name="risk_pct_0p0015__max_contracts_4__skip_trade_if_too_small_true",
            risk_pct_decimal=0.0015,
            max_contracts=4,
            export_root=mnq_root,
            notes="MNQ default retained from the local refinement campaign.",
        ),
        "MNQ_alt_perf": MotorConfigSpec(
            motor_key="MNQ_alt_perf",
            symbol="MNQ",
            config_label="alt_perf",
            campaign_variant_name="risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true",
            risk_pct_decimal=0.0025,
            max_contracts=3,
            export_root=mnq_root,
            notes="MNQ alternate performance-biased variant requested by the user.",
        ),
        "M2K_default": MotorConfigSpec(
            motor_key="M2K_default",
            symbol="M2K",
            config_label="default",
            campaign_variant_name="risk_pct_0p0030__max_contracts_6__skip_trade_if_too_small_true",
            risk_pct_decimal=0.0030,
            max_contracts=6,
            export_root=m2k_root,
            notes="M2K default retained from the local refinement campaign.",
        ),
        "M2K_alt_conservative": MotorConfigSpec(
            motor_key="M2K_alt_conservative",
            symbol="M2K",
            config_label="alt_conservative",
            campaign_variant_name="risk_pct_0p0030__max_contracts_5__skip_trade_if_too_small_true",
            risk_pct_decimal=0.0030,
            max_contracts=5,
            export_root=m2k_root,
            notes="M2K conservative alternate inside the same refined zone.",
        ),
        "MES_default": MotorConfigSpec(
            motor_key="MES_default",
            symbol="MES",
            config_label="default",
            campaign_variant_name="risk_pct_0p0020__max_contracts_6__skip_trade_if_too_small_true",
            risk_pct_decimal=0.0020,
            max_contracts=6,
            export_root=mes_root,
            notes="MES default chosen from the robust zone as the strongest prop-score point.",
        ),
        "MES_alt_conservative": MotorConfigSpec(
            motor_key="MES_alt_conservative",
            symbol="MES",
            config_label="alt_conservative",
            campaign_variant_name="risk_pct_0p0015__max_contracts_5__skip_trade_if_too_small_true",
            risk_pct_decimal=0.0015,
            max_contracts=5,
            export_root=mes_root,
            notes="MES conservative alternate inside the robust zone with lower drawdown.",
        ),
    }


def build_default_portfolio_variants() -> list[PortfolioVariantSpec]:
    variants: list[PortfolioVariantSpec] = []
    singles = [
        ("MNQ_only", ("MNQ_default",), "single"),
        ("M2K_only", ("M2K_default",), "single"),
        ("MES_only", ("MES_default",), "single"),
    ]
    for portfolio_name, members, family in singles:
        variants.append(
            PortfolioVariantSpec(
                portfolio_variant_name=f"{portfolio_name}__standalone__core_default",
                portfolio_name=portfolio_name,
                portfolio_family=family,
                allocation_scheme="standalone",
                config_bundle="core_default",
                member_motor_keys=members,
                description=f"Standalone 50k sleeve for {portfolio_name}.",
            )
        )

    pairs = [
        ("MNQ_M2K", ("MNQ_default", "M2K_default")),
        ("MNQ_MES", ("MNQ_default", "MES_default")),
        ("M2K_MES", ("M2K_default", "MES_default")),
    ]
    for portfolio_name, members in pairs:
        for scheme in ("equal_weight_notional", "equal_weight_risk_budget", "capped_overlay"):
            variants.append(
                PortfolioVariantSpec(
                    portfolio_variant_name=f"{portfolio_name}__{scheme}__core_default",
                    portfolio_name=portfolio_name,
                    portfolio_family="pair",
                    allocation_scheme=scheme,
                    config_bundle="core_default",
                    member_motor_keys=members,
                    description=f"{portfolio_name} with {scheme} on the default motor configs.",
                )
            )

    for scheme in ("equal_weight_notional", "equal_weight_risk_budget", "capped_overlay"):
        variants.append(
            PortfolioVariantSpec(
                portfolio_variant_name=f"MNQ_M2K_MES__{scheme}__core_default",
                portfolio_name="MNQ_M2K_MES",
                portfolio_family="three_way",
                allocation_scheme=scheme,
                config_bundle="core_default",
                member_motor_keys=("MNQ_default", "M2K_default", "MES_default"),
                description=f"Three-way portfolio with {scheme} on the default configs.",
            )
        )

    variants.append(
        PortfolioVariantSpec(
            portfolio_variant_name="MNQ_M2K_MES__equal_weight_risk_budget__mnq_alt_perf",
            portfolio_name="MNQ_M2K_MES",
            portfolio_family="three_way",
            allocation_scheme="equal_weight_risk_budget",
            config_bundle="mnq_alt_perf",
            member_motor_keys=("MNQ_alt_perf", "M2K_default", "MES_default"),
            description="Three-way portfolio with the MNQ performance alternate under inverse-risk weighting.",
        )
    )
    variants.append(
        PortfolioVariantSpec(
            portfolio_variant_name="MNQ_M2K_MES__equal_weight_risk_budget__conservative_mix",
            portfolio_name="MNQ_M2K_MES",
            portfolio_family="three_way",
            allocation_scheme="equal_weight_risk_budget",
            config_bundle="conservative_mix",
            member_motor_keys=("MNQ_default", "M2K_alt_conservative", "MES_alt_conservative"),
            description="Three-way portfolio with the conservative M2K / MES alternates under inverse-risk weighting.",
        )
    )
    return variants


def _load_motor_variant(spec: MotorConfigSpec) -> MotorVariantData:
    summary = pd.read_csv(spec.export_root / "summary_by_variant.csv")
    matched_summary = summary.loc[summary["campaign_variant_name"].astype(str) == spec.campaign_variant_name].copy()
    if matched_summary.empty:
        raise ValueError(
            f"Variant {spec.campaign_variant_name!r} was not found in {spec.export_root / 'summary_by_variant.csv'}."
        )

    daily = pd.read_csv(spec.export_root / "daily_equity_by_variant.csv", parse_dates=["session_date"])
    daily = daily.loc[
        (daily["campaign_variant_name"].astype(str) == spec.campaign_variant_name)
        & (daily["scope"].astype(str) == "full")
    ].copy()
    if daily.empty:
        raise ValueError(f"No full-scope daily results were found for {spec.motor_key}.")
    daily["session_date"] = pd.to_datetime(daily["session_date"]).dt.normalize()

    trades = pd.read_csv(spec.export_root / "trades_by_variant.csv", parse_dates=["entry_time", "exit_time", "session_date"])
    trades = trades.loc[
        (trades["campaign_variant_name"].astype(str) == spec.campaign_variant_name)
        & (trades["scope"].astype(str) == "full")
    ].copy()
    trades["session_date"] = pd.to_datetime(trades["session_date"]).dt.normalize()

    return MotorVariantData(
        spec=spec,
        summary_row=matched_summary.iloc[0],
        daily_full=daily.reset_index(drop=True),
        trades_full=trades.reset_index(drop=True),
    )


def build_master_calendar(loaded_motors: dict[str, MotorVariantData]) -> pd.DataFrame:
    phase_rows: list[pd.DataFrame] = []
    for data in loaded_motors.values():
        phase_rows.append(data.daily_full[["session_date", "phase"]].drop_duplicates().copy())
    calendar = pd.concat(phase_rows, ignore_index=True).drop_duplicates()
    calendar["session_date"] = pd.to_datetime(calendar["session_date"]).dt.normalize()
    conflicts = (
        calendar.groupby("session_date")["phase"].nunique().loc[lambda value: value > 1]
        if not calendar.empty
        else pd.Series(dtype=int)
    )
    if not conflicts.empty:
        raise ValueError(f"Phase conflicts were found across motor calendars: {conflicts.index.tolist()[:5]}")
    calendar = calendar.groupby("session_date", as_index=False)["phase"].first().sort_values("session_date").reset_index(drop=True)
    return calendar


def _member_scale_map(
    portfolio_variant: PortfolioVariantSpec,
    loaded_motors: dict[str, MotorVariantData],
) -> dict[str, float]:
    member_keys = list(portfolio_variant.member_motor_keys)
    if portfolio_variant.allocation_scheme in {"standalone", "capped_overlay"}:
        return {member_key: 1.0 for member_key in member_keys}
    if portfolio_variant.allocation_scheme == "equal_weight_notional":
        weight = 1.0 / float(len(member_keys))
        return {member_key: weight for member_key in member_keys}
    if portfolio_variant.allocation_scheme == "equal_weight_risk_budget":
        inverse_risk = {member_key: 1.0 / float(loaded_motors[member_key].spec.risk_pct_decimal) for member_key in member_keys}
        total = sum(inverse_risk.values())
        return {member_key: value / total for member_key, value in inverse_risk.items()}
    raise ValueError(f"Unknown allocation scheme: {portfolio_variant.allocation_scheme}")


def scale_trades_constant(
    trades: pd.DataFrame,
    *,
    scale_factor: float,
    motor_key: str,
    symbol: str,
    config_label: str,
) -> pd.DataFrame:
    if trades.empty or scale_factor <= 0:
        return pd.DataFrame()
    scaled = trades.copy()
    scaled["motor_key"] = motor_key
    scaled["symbol"] = symbol
    scaled["config_label"] = config_label
    scaled["portfolio_scale_factor"] = float(scale_factor)
    numeric_columns = [
        "quantity",
        "risk_per_contract_usd",
        "actual_risk_usd",
        "trade_risk_usd",
        "notional_usd",
        "pnl_points",
        "pnl_ticks",
        "pnl_usd",
        "fees",
        "net_pnl_usd",
        "risk_budget_usd",
    ]
    for column in numeric_columns:
        if column in scaled.columns:
            scaled[column] = pd.to_numeric(scaled[column], errors="coerce").fillna(0.0) * float(scale_factor)
    scaled["trade_id"] = np.arange(1, len(scaled) + 1)
    return scaled.reset_index(drop=True)


def scale_trades_with_capped_overlay(
    trades_by_motor: dict[str, pd.DataFrame],
    *,
    member_specs: dict[str, MotorConfigSpec],
    risk_cap_usd: float,
    base_scale_map: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    all_entries: list[pd.DataFrame] = []
    for motor_key, trades in trades_by_motor.items():
        if trades.empty:
            continue
        frame = trades.copy()
        frame["motor_key"] = motor_key
        frame["symbol"] = member_specs[motor_key].symbol
        frame["config_label"] = member_specs[motor_key].config_label
        frame["base_scale_factor"] = float((base_scale_map or {}).get(motor_key, 1.0))
        frame["requested_risk_usd"] = (
            pd.to_numeric(frame.get("trade_risk_usd"), errors="coerce")
            .fillna(pd.to_numeric(frame.get("actual_risk_usd"), errors="coerce"))
            .fillna(0.0)
            * frame["base_scale_factor"]
        )
        all_entries.append(frame)

    if not all_entries:
        return pd.DataFrame(), {
            "cap_hit_count": 0,
            "partial_scale_trade_count": 0,
            "rejected_trade_count": 0,
            "max_concurrent_risk_usd": 0.0,
        }

    combined = pd.concat(all_entries, ignore_index=True)
    combined["portfolio_trade_id"] = np.arange(1, len(combined) + 1)
    combined["entry_time"] = pd.to_datetime(combined["entry_time"], errors="coerce")
    combined["exit_time"] = pd.to_datetime(combined["exit_time"], errors="coerce")

    entries_by_time = {
        key: group.copy()
        for key, group in combined.groupby("entry_time", sort=True)
    }
    exits_by_time: dict[pd.Timestamp, list[int]] = {}
    for row in combined.itertuples():
        exits_by_time.setdefault(pd.Timestamp(row.exit_time), []).append(int(row.portfolio_trade_id))

    event_times = sorted(set(entries_by_time).union(exits_by_time))
    accepted_risk_by_trade: dict[int, float] = {}
    accepted_scale_by_trade: dict[int, float] = {}
    cap_hit_count = 0
    max_concurrent_risk = 0.0

    for event_time in event_times:
        for trade_id in exits_by_time.get(event_time, []):
            accepted_risk_by_trade.pop(int(trade_id), None)

        entry_group = entries_by_time.get(event_time)
        if entry_group is None or entry_group.empty:
            continue

        active_risk = float(sum(accepted_risk_by_trade.values()))
        requested = pd.to_numeric(entry_group["requested_risk_usd"], errors="coerce").fillna(0.0)
        requested_total = float(requested.sum())
        available_risk = max(float(risk_cap_usd) - active_risk, 0.0)
        common_scale = min(1.0, available_risk / requested_total) if requested_total > 0 else 0.0
        if requested_total > 0 and common_scale < 0.999999:
            cap_hit_count += 1

        for row in entry_group.itertuples():
            accepted_scale = float(row.base_scale_factor) * common_scale
            accepted_scale_by_trade[int(row.portfolio_trade_id)] = accepted_scale
            accepted_risk = float(row.requested_risk_usd) * common_scale
            if accepted_risk > 0:
                accepted_risk_by_trade[int(row.portfolio_trade_id)] = accepted_risk
        max_concurrent_risk = max(max_concurrent_risk, float(sum(accepted_risk_by_trade.values())))

    scaled = combined.copy()
    scaled["portfolio_scale_factor"] = scaled["portfolio_trade_id"].map(accepted_scale_by_trade).fillna(0.0).astype(float)
    partial_scale_trade_count = int(((scaled["portfolio_scale_factor"] > 0) & (scaled["portfolio_scale_factor"] < scaled["base_scale_factor"])).sum())
    rejected_trade_count = int((scaled["portfolio_scale_factor"] <= 0).sum())
    scaled = scaled.loc[scaled["portfolio_scale_factor"] > 0].copy()

    numeric_columns = [
        "quantity",
        "risk_per_contract_usd",
        "actual_risk_usd",
        "trade_risk_usd",
        "notional_usd",
        "pnl_points",
        "pnl_ticks",
        "pnl_usd",
        "fees",
        "net_pnl_usd",
        "risk_budget_usd",
    ]
    for column in numeric_columns:
        if column in scaled.columns:
            scaled[column] = pd.to_numeric(scaled[column], errors="coerce").fillna(0.0) * scaled["portfolio_scale_factor"]
    scaled["trade_id"] = np.arange(1, len(scaled) + 1)
    return scaled.reset_index(drop=True), {
        "cap_hit_count": int(cap_hit_count),
        "partial_scale_trade_count": partial_scale_trade_count,
        "rejected_trade_count": rejected_trade_count,
        "max_concurrent_risk_usd": float(max_concurrent_risk),
    }


def build_daily_motor_contributions(
    trades: pd.DataFrame,
    *,
    calendar: pd.DataFrame,
    member_specs: dict[str, MotorConfigSpec],
) -> pd.DataFrame:
    sessions = pd.to_datetime(calendar["session_date"]).dt.normalize()
    rows: list[pd.DataFrame] = []
    for motor_key, spec in member_specs.items():
        base = pd.DataFrame({"session_date": sessions})
        motor_trades = trades.loc[trades["motor_key"].astype(str) == motor_key].copy() if not trades.empty else pd.DataFrame()
        if motor_trades.empty:
            base["daily_pnl_usd"] = 0.0
            base["daily_trade_count"] = 0
        else:
            motor_trades["session_date"] = pd.to_datetime(motor_trades["session_date"]).dt.normalize()
            grouped = (
                motor_trades.groupby("session_date", as_index=False)
                .agg(
                    daily_pnl_usd=("net_pnl_usd", "sum"),
                    daily_trade_count=("trade_id", "count"),
                )
            )
            grouped["session_date"] = pd.to_datetime(grouped["session_date"]).dt.normalize()
            base = base.merge(grouped, on="session_date", how="left")
            base["daily_pnl_usd"] = pd.to_numeric(base["daily_pnl_usd"], errors="coerce").fillna(0.0)
            base["daily_trade_count"] = pd.to_numeric(base["daily_trade_count"], errors="coerce").fillna(0).astype(int)
        base["motor_key"] = motor_key
        base["symbol"] = spec.symbol
        base["config_label"] = spec.config_label
        base["phase"] = calendar["phase"].tolist()
        rows.append(base)
    return pd.concat(rows, ignore_index=True)


def _drawdown_interval(daily_results: pd.DataFrame) -> pd.Index | None:
    if daily_results.empty:
        return None
    ordered = daily_results.sort_values("session_date").reset_index(drop=True)
    drawdown = pd.to_numeric(ordered["drawdown_usd"], errors="coerce").fillna(0.0)
    trough_idx = int(drawdown.idxmin())
    if drawdown.iloc[trough_idx] >= 0:
        return None
    equity = pd.to_numeric(ordered["equity"], errors="coerce").ffill().fillna(0.0)
    peak_value = float(pd.to_numeric(ordered["peak_equity"], errors="coerce").iloc[trough_idx])
    peak_candidates = np.flatnonzero(equity.iloc[: trough_idx + 1].to_numpy() == peak_value)
    peak_idx = int(peak_candidates[-1]) if len(peak_candidates) > 0 else 0
    return pd.Index(ordered.loc[peak_idx : trough_idx, "session_date"])


def _time_under_water_days(daily_results: pd.DataFrame) -> tuple[int, float]:
    if daily_results.empty:
        return 0, 0.0
    underwater = pd.to_numeric(daily_results["equity"], errors="coerce").lt(
        pd.to_numeric(daily_results["peak_equity"], errors="coerce")
    )
    streaks: list[int] = []
    current = 0
    for flag in underwater:
        if bool(flag):
            current += 1
        elif current > 0:
            streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return max(streaks, default=0), float(np.mean(streaks)) if streaks else 0.0


def summarize_diversification(
    daily_motor: pd.DataFrame,
    *,
    daily_portfolio: pd.DataFrame,
    member_specs: dict[str, MotorConfigSpec],
) -> dict[str, Any]:
    if daily_motor.empty:
        return {
            "mean_pairwise_daily_corr": 0.0,
            "mean_pairwise_weekly_corr": 0.0,
            "pct_days_multiple_motors_lose_together": 0.0,
            "pct_days_offsetting_pnl": 0.0,
            "pct_days_positive_offsetting": 0.0,
            "overlap_rate_pct": 0.0,
            "pnl_contribution_pct_MNQ": np.nan,
            "pnl_contribution_pct_M2K": np.nan,
            "pnl_contribution_pct_MES": np.nan,
            "drawdown_contribution_pct_MNQ": np.nan,
            "drawdown_contribution_pct_M2K": np.nan,
            "drawdown_contribution_pct_MES": np.nan,
        }

    pivot = (
        daily_motor.pivot_table(index="session_date", columns="symbol", values="daily_pnl_usd", aggfunc="sum")
        .sort_index()
        .fillna(0.0)
    )
    symbols = [spec.symbol for spec in member_specs.values()]
    pivot = pivot.reindex(columns=sorted(set(symbols)), fill_value=0.0)
    corr_daily = pivot.corr() if len(pivot) > 1 else pd.DataFrame(np.eye(len(pivot.columns)), index=pivot.columns, columns=pivot.columns)

    weekly = pivot.copy()
    weekly.index = pd.to_datetime(weekly.index)
    weekly = weekly.resample("W-FRI").sum()
    corr_weekly = weekly.corr() if len(weekly) > 1 else corr_daily.copy()

    pairwise_daily = []
    pairwise_weekly = []
    columns = list(pivot.columns)
    for index, left in enumerate(columns):
        for right in columns[index + 1 :]:
            pairwise_daily.append(float(corr_daily.loc[left, right]))
            pairwise_weekly.append(float(corr_weekly.loc[left, right]))

    loss_counts = pivot.lt(0).sum(axis=1)
    positive_counts = pivot.gt(0).sum(axis=1)
    trade_counts = (
        daily_motor.pivot_table(index="session_date", columns="symbol", values="daily_trade_count", aggfunc="sum")
        .sort_index()
        .fillna(0.0)
        .reindex(columns=columns, fill_value=0.0)
    )
    days_any_trade = trade_counts.gt(0).any(axis=1)
    overlap_rate = float(trade_counts.gt(0).sum(axis=1).ge(2).sum() / days_any_trade.sum() * 100.0) if days_any_trade.any() else 0.0

    total_sessions = max(int(len(pivot)), 1)
    pnl_total = float(pd.to_numeric(daily_portfolio["daily_pnl_usd"], errors="coerce").sum())
    contribution_interval = _drawdown_interval(daily_portfolio)
    drawdown_total = abs(float(pd.to_numeric(daily_portfolio["drawdown_usd"], errors="coerce").min())) if not daily_portfolio.empty else 0.0
    drawdown_slice = pivot.loc[contribution_interval] if contribution_interval is not None else pd.DataFrame(columns=columns)

    result: dict[str, Any] = {
        "mean_pairwise_daily_corr": float(np.mean(pairwise_daily)) if pairwise_daily else 0.0,
        "mean_pairwise_weekly_corr": float(np.mean(pairwise_weekly)) if pairwise_weekly else 0.0,
        "pct_days_multiple_motors_lose_together": float(loss_counts.ge(2).sum() / total_sessions * 100.0),
        "pct_days_offsetting_pnl": float(((loss_counts >= 1) & (positive_counts >= 1)).sum() / total_sessions * 100.0),
        "pct_days_positive_offsetting": float(((loss_counts >= 1) & (positive_counts >= 1) & pivot.sum(axis=1).gt(0)).sum() / total_sessions * 100.0),
        "overlap_rate_pct": overlap_rate,
    }

    pnl_by_symbol = pivot.sum(axis=0)
    drawdown_by_symbol = drawdown_slice.sum(axis=0) if not drawdown_slice.empty else pd.Series(0.0, index=columns)
    for symbol in ("MNQ", "M2K", "MES"):
        pnl_value = float(pnl_by_symbol.get(symbol, np.nan))
        dd_value = float(drawdown_by_symbol.get(symbol, np.nan))
        result[f"pnl_contribution_pct_{symbol}"] = float(pnl_value / pnl_total * 100.0) if math.isfinite(pnl_value) and pnl_total != 0 else np.nan
        result[f"drawdown_contribution_pct_{symbol}"] = float(-dd_value / drawdown_total * 100.0) if math.isfinite(dd_value) and drawdown_total > 0 else np.nan
    return result


def summarize_portfolio_scope(
    trades: pd.DataFrame,
    *,
    daily_results: pd.DataFrame,
    daily_motor: pd.DataFrame,
    calendar: pd.DataFrame,
    member_specs: dict[str, MotorConfigSpec],
    overlay_metrics: dict[str, float | int],
    initial_capital_usd: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = compute_metrics(
        trades,
        session_dates=pd.to_datetime(calendar["session_date"]).dt.normalize().tolist(),
        initial_capital=float(initial_capital_usd),
        prop_constraints=TOPSTEP_50K_PROP,
    )
    prop_metrics = _prop_path_metrics(daily_results, initial_capital=float(initial_capital_usd))
    daily_pnl = pd.to_numeric(daily_results["daily_pnl_usd"], errors="coerce").fillna(0.0)
    max_tuw_days, avg_tuw_days = _time_under_water_days(daily_results)
    diversification = summarize_diversification(daily_motor, daily_portfolio=daily_results, member_specs=member_specs)

    summary = {
        "net_pnl_usd": float(pd.to_numeric(daily_results["daily_pnl_usd"], errors="coerce").fillna(0.0).sum()),
        "return_pct": float(pd.to_numeric(daily_results["cumulative_pnl_usd"], errors="coerce").iloc[-1] / float(initial_capital_usd) * 100.0)
        if not daily_results.empty
        else 0.0,
        "cagr_pct": _cagr_pct(daily_results, initial_capital=float(initial_capital_usd)),
        "sharpe": float(metrics.get("sharpe_ratio", 0.0)),
        "sortino": _sortino_ratio(daily_pnl, capital=float(initial_capital_usd)),
        "annualized_vol_pct": _annualized_vol_pct(daily_pnl, capital=float(initial_capital_usd)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "expectancy_usd": float(metrics.get("expectancy", 0.0)),
        "nb_trades": int(metrics.get("n_trades", 0)),
        "max_drawdown_usd": float(abs(pd.to_numeric(daily_results["drawdown_usd"], errors="coerce").min())) if not daily_results.empty else 0.0,
        "max_drawdown_pct": float(abs(pd.to_numeric(daily_results["drawdown_pct"], errors="coerce").min())) if not daily_results.empty else 0.0,
        "max_daily_drawdown_usd": float(prop_metrics["max_daily_drawdown_usd"]),
        "worst_day_pnl_usd": float(prop_metrics["worst_day_pnl_usd"]),
        "worst_trade_pnl_usd": float(metrics.get("worst_trade", 0.0)),
        "max_time_under_water_days": int(max_tuw_days),
        "avg_time_under_water_days": float(avg_tuw_days),
        "percent_days_traded": float((pd.to_numeric(daily_results["daily_trade_count"], errors="coerce") > 0).mean() * 100.0) if not daily_results.empty else 0.0,
        "nb_days_below_minus_250": int((daily_pnl <= -250.0).sum()) if not daily_results.empty else 0,
        "nb_days_below_minus_500": int((daily_pnl <= -500.0).sum()) if not daily_results.empty else 0,
        "nb_days_below_minus_1000": int((daily_pnl <= -1_000.0).sum()) if not daily_results.empty else 0,
        "pass_target_3000_usd_without_breaching_2000_dd": bool(prop_metrics["pass_target_3000_usd_without_breaching_2000_dd"]),
        "days_to_3000_usd_if_reached": float(prop_metrics["days_to_target_3000_usd"]),
        "max_trailing_drawdown_observed_usd": float(prop_metrics["max_trailing_drawdown_observed_usd"]),
        "max_static_drawdown_observed_usd": float(prop_metrics["max_static_drawdown_observed_usd"]),
        "overlay_cap_hit_count": int(overlay_metrics.get("cap_hit_count", 0)),
        "overlay_partial_scale_trade_count": int(overlay_metrics.get("partial_scale_trade_count", 0)),
        "overlay_rejected_trade_count": int(overlay_metrics.get("rejected_trade_count", 0)),
        "overlay_max_concurrent_risk_usd": float(overlay_metrics.get("max_concurrent_risk_usd", 0.0)),
        **diversification,
    }
    return summary, diversification


def build_portfolio_scope(
    portfolio_variant: PortfolioVariantSpec,
    *,
    loaded_motors: dict[str, MotorVariantData],
    calendar: pd.DataFrame,
    initial_capital_usd: float,
    portfolio_risk_cap_usd: float,
    scope_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    scoped_calendar = calendar.copy()
    if scope_name == "oos":
        scoped_calendar = scoped_calendar.loc[scoped_calendar["phase"].astype(str) == "oos"].copy().reset_index(drop=True)

    member_specs = {member_key: loaded_motors[member_key].spec for member_key in portfolio_variant.member_motor_keys}
    member_trades: dict[str, pd.DataFrame] = {}
    for member_key in portfolio_variant.member_motor_keys:
        trades = loaded_motors[member_key].trades_full.copy()
        if scope_name == "oos":
            trades = trades.loc[trades["phase"].astype(str) == "oos"].copy()
        member_trades[member_key] = trades.reset_index(drop=True)

    overlay_metrics: dict[str, float | int] = {
        "cap_hit_count": 0,
        "partial_scale_trade_count": 0,
        "rejected_trade_count": 0,
        "max_concurrent_risk_usd": 0.0,
    }
    if portfolio_variant.allocation_scheme == "capped_overlay":
        scaled_trades, overlay_metrics = scale_trades_with_capped_overlay(
            member_trades,
            member_specs=member_specs,
            risk_cap_usd=float(portfolio_risk_cap_usd),
        )
    else:
        weight_map = _member_scale_map(portfolio_variant, loaded_motors)
        scaled_frames = [
            scale_trades_constant(
                trades,
                scale_factor=float(weight_map[member_key]),
                motor_key=member_key,
                symbol=member_specs[member_key].symbol,
                config_label=member_specs[member_key].config_label,
            )
            for member_key, trades in member_trades.items()
        ]
        scaled_trades = pd.concat([frame for frame in scaled_frames if not frame.empty], ignore_index=True) if scaled_frames else pd.DataFrame()

    if not scaled_trades.empty:
        scaled_trades["session_date"] = pd.to_datetime(scaled_trades["session_date"]).dt.normalize()
        scaled_trades["portfolio_variant_name"] = portfolio_variant.portfolio_variant_name
        scaled_trades["portfolio_name"] = portfolio_variant.portfolio_name
        scaled_trades["portfolio_family"] = portfolio_variant.portfolio_family
        scaled_trades["allocation_scheme"] = portfolio_variant.allocation_scheme
        scaled_trades["config_bundle"] = portfolio_variant.config_bundle
        scaled_trades["scope"] = scope_name
        scaled_trades["account_size_usd"] = float(initial_capital_usd)

    sessions = pd.to_datetime(scoped_calendar["session_date"]).dt.normalize().tolist()
    daily_results = _daily_results_from_trades(
        trades=scaled_trades,
        sessions=sessions,
        initial_capital=float(initial_capital_usd),
    )
    daily_results["session_date"] = pd.to_datetime(daily_results["session_date"]).dt.normalize()
    daily_results = daily_results.merge(scoped_calendar[["session_date", "phase"]], on="session_date", how="left")
    daily_results["portfolio_variant_name"] = portfolio_variant.portfolio_variant_name
    daily_results["portfolio_name"] = portfolio_variant.portfolio_name
    daily_results["portfolio_family"] = portfolio_variant.portfolio_family
    daily_results["allocation_scheme"] = portfolio_variant.allocation_scheme
    daily_results["config_bundle"] = portfolio_variant.config_bundle
    daily_results["scope"] = scope_name

    daily_motor = build_daily_motor_contributions(
        scaled_trades,
        calendar=scoped_calendar,
        member_specs=member_specs,
    )
    daily_motor["portfolio_variant_name"] = portfolio_variant.portfolio_variant_name
    daily_motor["portfolio_name"] = portfolio_variant.portfolio_name
    daily_motor["portfolio_family"] = portfolio_variant.portfolio_family
    daily_motor["allocation_scheme"] = portfolio_variant.allocation_scheme
    daily_motor["config_bundle"] = portfolio_variant.config_bundle
    daily_motor["scope"] = scope_name
    return scaled_trades, daily_results, overlay_metrics, daily_motor


def summarize_portfolio_variant(
    portfolio_variant: PortfolioVariantSpec,
    *,
    loaded_motors: dict[str, MotorVariantData],
    calendar: pd.DataFrame,
    initial_capital_usd: float,
    portfolio_risk_cap_usd: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    scope_summaries: dict[str, dict[str, Any]] = {}
    trade_exports: list[pd.DataFrame] = []
    daily_exports: list[pd.DataFrame] = []
    motor_daily_exports: list[pd.DataFrame] = []
    diversification_rows: list[dict[str, Any]] = []

    for scope_name in ("full", "oos"):
        trades, daily_results, overlay_metrics, daily_motor = build_portfolio_scope(
            portfolio_variant,
            loaded_motors=loaded_motors,
            calendar=calendar,
            initial_capital_usd=float(initial_capital_usd),
            portfolio_risk_cap_usd=float(portfolio_risk_cap_usd),
            scope_name=scope_name,
        )
        trade_exports.append(trades)
        daily_exports.append(daily_results)
        motor_daily_exports.append(daily_motor)

        member_specs = {member_key: loaded_motors[member_key].spec for member_key in portfolio_variant.member_motor_keys}
        summary, diversification = summarize_portfolio_scope(
            trades,
            daily_results=daily_results,
            daily_motor=daily_motor,
            calendar=daily_results[["session_date", "phase"]].copy(),
            member_specs=member_specs,
            overlay_metrics=overlay_metrics,
            initial_capital_usd=float(initial_capital_usd),
        )
        diversification_rows.append(
            {
                "portfolio_variant_name": portfolio_variant.portfolio_variant_name,
                "portfolio_name": portfolio_variant.portfolio_name,
                "portfolio_family": portfolio_variant.portfolio_family,
                "allocation_scheme": portfolio_variant.allocation_scheme,
                "config_bundle": portfolio_variant.config_bundle,
                "scope": scope_name,
                **diversification,
            }
        )
        scope_summaries[scope_name] = summary

    summary_row = {
        "portfolio_variant_name": portfolio_variant.portfolio_variant_name,
        "portfolio_name": portfolio_variant.portfolio_name,
        "portfolio_family": portfolio_variant.portfolio_family,
        "allocation_scheme": portfolio_variant.allocation_scheme,
        "config_bundle": portfolio_variant.config_bundle,
        "member_motor_keys": "|".join(portfolio_variant.member_motor_keys),
        "member_symbols": "|".join(loaded_motors[member_key].spec.symbol for member_key in portfolio_variant.member_motor_keys),
        "member_config_labels": "|".join(
            f"{loaded_motors[member_key].spec.symbol}:{loaded_motors[member_key].spec.config_label}"
            for member_key in portfolio_variant.member_motor_keys
        ),
        "motor_count": int(len(portfolio_variant.member_motor_keys)),
        "description": portfolio_variant.description,
        **{f"full_{key}": value for key, value in scope_summaries["full"].items()},
        **{f"oos_{key}": value for key, value in scope_summaries["oos"].items()},
    }
    return (
        summary_row,
        pd.concat([frame for frame in trade_exports if not frame.empty], ignore_index=True) if trade_exports else pd.DataFrame(),
        pd.concat(daily_exports, ignore_index=True) if daily_exports else pd.DataFrame(),
        pd.concat(motor_daily_exports, ignore_index=True) if motor_daily_exports else pd.DataFrame(),
        diversification_rows,
    )


def _concat_non_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(usable, ignore_index=True) if usable else pd.DataFrame()


def _correlation_matrix_from_default_motors(
    loaded_motors: dict[str, MotorVariantData],
    *,
    scope_name: str = "oos",
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for motor_key in ("MNQ_default", "M2K_default", "MES_default"):
        if motor_key not in loaded_motors:
            continue
        data = loaded_motors[motor_key]
        daily = data.daily_full.copy()
        if scope_name == "oos":
            daily = daily.loc[daily["phase"].astype(str) == "oos"].copy()
        rows.append(
            daily[["session_date", "daily_pnl_usd"]]
            .rename(columns={"daily_pnl_usd": data.spec.symbol})
            .sort_values("session_date")
            .reset_index(drop=True)
        )
    merged = None
    for frame in rows:
        merged = frame if merged is None else merged.merge(frame, on="session_date", how="outer")
    if merged is None or merged.empty:
        return pd.DataFrame()
    merged = merged.sort_values("session_date").fillna(0.0)
    return merged.drop(columns=["session_date"]).corr()


def _plot_equity_curves(
    summary: pd.DataFrame,
    daily_equity: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    selected_names = []
    for family in ("single", "pair", "three_way"):
        family_rows = summary.loc[summary["portfolio_family"] == family].copy()
        if family_rows.empty:
            continue
        selected_names.append(
            str(
                family_rows.sort_values(
                    ["oos_portfolio_score", "oos_sharpe", "oos_net_pnl_usd"],
                    ascending=[False, False, False],
                ).iloc[0]["portfolio_variant_name"]
            )
        )
    baseline = "MNQ_only__standalone__core_default"
    if baseline in summary["portfolio_variant_name"].tolist():
        selected_names.append(baseline)
    selected_names = list(dict.fromkeys(selected_names))

    fig, ax = plt.subplots(figsize=(11, 6))
    scoped = daily_equity.loc[
        (daily_equity["scope"].astype(str) == "oos")
        & (daily_equity["portfolio_variant_name"].astype(str).isin(selected_names))
    ].copy()
    if scoped.empty:
        plt.close(fig)
        return
    for name in selected_names:
        frame = scoped.loc[scoped["portfolio_variant_name"].astype(str) == name].copy()
        if frame.empty:
            continue
        ax.plot(pd.to_datetime(frame["session_date"]), pd.to_numeric(frame["equity"], errors="coerce"), label=name)
    ax.set_title("OOS Equity Curves")
    ax.set_xlabel("Session Date")
    ax.set_ylabel("Equity (USD)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_drawdown_curves(
    summary: pd.DataFrame,
    daily_equity: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    selected_names = []
    for family in ("single", "pair", "three_way"):
        family_rows = summary.loc[summary["portfolio_family"] == family].copy()
        if family_rows.empty:
            continue
        selected_names.append(
            str(
                family_rows.sort_values(
                    ["oos_portfolio_score", "oos_sharpe", "oos_net_pnl_usd"],
                    ascending=[False, False, False],
                ).iloc[0]["portfolio_variant_name"]
            )
        )
    baseline = "MNQ_only__standalone__core_default"
    if baseline in summary["portfolio_variant_name"].tolist():
        selected_names.append(baseline)
    selected_names = list(dict.fromkeys(selected_names))

    fig, ax = plt.subplots(figsize=(11, 6))
    scoped = daily_equity.loc[
        (daily_equity["scope"].astype(str) == "oos")
        & (daily_equity["portfolio_variant_name"].astype(str).isin(selected_names))
    ].copy()
    if scoped.empty:
        plt.close(fig)
        return
    for name in selected_names:
        frame = scoped.loc[scoped["portfolio_variant_name"].astype(str) == name].copy()
        if frame.empty:
            continue
        ax.plot(pd.to_datetime(frame["session_date"]), pd.to_numeric(frame["drawdown_usd"], errors="coerce"), label=name)
    ax.set_title("OOS Drawdown Curves")
    ax.set_xlabel("Session Date")
    ax.set_ylabel("Drawdown (USD)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_correlation_heatmap(correlation_matrix: pd.DataFrame, *, output_path: Path) -> None:
    if correlation_matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(correlation_matrix.to_numpy(dtype=float), cmap="RdYlGn_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns.tolist())
    ax.set_yticks(np.arange(len(correlation_matrix.index)))
    ax.set_yticklabels(correlation_matrix.index.tolist())
    ax.set_title("Daily PnL Correlation Heatmap (OOS, default motors)")
    for row_index in range(correlation_matrix.shape[0]):
        for col_index in range(correlation_matrix.shape[1]):
            ax.text(col_index, row_index, f"{float(correlation_matrix.iat[row_index, col_index]):.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_diversification_contribution(
    summary: pd.DataFrame,
    diversification: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    recommended = summary.sort_values(
        ["oos_portfolio_score", "oos_sharpe", "oos_net_pnl_usd"],
        ascending=[False, False, False],
    ).iloc[0]
    scoped = diversification.loc[
        (diversification["portfolio_variant_name"].astype(str) == str(recommended["portfolio_variant_name"]))
        & (diversification["scope"].astype(str) == "oos")
    ].copy()
    if scoped.empty:
        return
    row = scoped.iloc[0]
    symbols = ["MNQ", "M2K", "MES"]
    pnl_values = [pd.to_numeric(row.get(f"pnl_contribution_pct_{symbol}"), errors="coerce") for symbol in symbols]
    dd_values = [pd.to_numeric(row.get(f"drawdown_contribution_pct_{symbol}"), errors="coerce") for symbol in symbols]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(symbols, pnl_values, color="#2563eb")
    axes[0].set_title("PnL Contribution %")
    axes[0].set_ylabel("Percent")
    axes[0].axhline(0.0, color="black", linewidth=0.8)

    axes[1].bar(symbols, dd_values, color="#ef4444")
    axes[1].set_title("Drawdown Contribution %")
    axes[1].axhline(0.0, color="black", linewidth=0.8)

    fig.suptitle(f"Diversification Contribution - {recommended['portfolio_variant_name']}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_portfolio_score(summary: pd.DataFrame, *, output_path: Path) -> None:
    ordered = summary.sort_values(["oos_portfolio_score", "oos_sharpe"], ascending=[False, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(ordered)), pd.to_numeric(ordered["oos_portfolio_score"], errors="coerce"), color="#16a34a")
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels(ordered["portfolio_variant_name"], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("OOS portfolio_score")
    ax.set_title("Portfolio Score by Portfolio Variant")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _format_brief(row: pd.Series) -> str:
    return (
        f"net `{float(row['oos_net_pnl_usd']):.2f}` | Sharpe `{float(row['oos_sharpe']):.3f}` | "
        f"maxDD `{float(row['oos_max_drawdown_usd']):.2f}` | pass `{bool(row['oos_pass_target_3000_usd_without_breaching_2000_dd'])}`"
    )


def _top_table(summary: pd.DataFrame, *, top_n: int = 10) -> str:
    cols = [
        "portfolio_variant_name",
        "portfolio_family",
        "allocation_scheme",
        "config_bundle",
        "oos_net_pnl_usd",
        "oos_sharpe",
        "oos_max_drawdown_usd",
        "oos_max_daily_drawdown_usd",
        "oos_portfolio_score",
        "oos_pass_target_3000_usd_without_breaching_2000_dd",
    ]
    view = summary[cols].copy().sort_values(
        ["oos_portfolio_score", "oos_sharpe", "oos_net_pnl_usd"],
        ascending=[False, False, False],
    ).head(top_n)
    return view.to_string(index=False)


def _final_verdict_label(recommended: pd.Series | None) -> str:
    if recommended is None:
        return "aucune_recommandation"
    name = str(recommended["portfolio_name"])
    if name == "MNQ_only":
        return "retenir_MNQ_seul"
    if name == "MNQ_M2K":
        return "retenir_MNQ_plus_M2K"
    if name == "MNQ_M2K_MES":
        return "retenir_MNQ_plus_M2K_plus_MES"
    return f"retenir_{name}"


def _build_final_report(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    diversification: pd.DataFrame,
    correlation_matrix_daily: pd.DataFrame,
) -> dict[str, Any]:
    best_single = summary.loc[summary["portfolio_family"] == "single"].sort_values(
        ["oos_portfolio_score", "oos_sharpe", "oos_net_pnl_usd"],
        ascending=[False, False, False],
    ).iloc[0]
    best_pair = summary.loc[summary["portfolio_family"] == "pair"].sort_values(
        ["oos_portfolio_score", "oos_sharpe", "oos_net_pnl_usd"],
        ascending=[False, False, False],
    ).iloc[0]
    best_three_way = summary.loc[summary["portfolio_family"] == "three_way"].sort_values(
        ["oos_portfolio_score", "oos_sharpe", "oos_net_pnl_usd"],
        ascending=[False, False, False],
    ).iloc[0]

    passing = summary.loc[
        pd.Series(summary["oos_pass_target_3000_usd_without_breaching_2000_dd"], dtype="boolean").fillna(False)
    ].copy()
    recommended = (
        passing.sort_values(["oos_portfolio_score", "oos_sharpe", "oos_net_pnl_usd"], ascending=[False, False, False]).iloc[0]
        if not passing.empty
        else summary.sort_values(["oos_portfolio_score", "oos_sharpe", "oos_net_pnl_usd"], ascending=[False, False, False]).iloc[0]
    )
    prop_safe = (
        passing.sort_values(["oos_max_daily_drawdown_usd", "oos_max_drawdown_usd", "oos_sharpe"], ascending=[True, True, False]).iloc[0]
        if not passing.empty
        else recommended
    )
    aggressive = (
        passing.sort_values(["oos_net_pnl_usd", "oos_portfolio_score", "oos_sharpe"], ascending=[False, False, False]).iloc[0]
        if not passing.empty
        else summary.sort_values(["oos_net_pnl_usd", "oos_sharpe"], ascending=[False, False]).iloc[0]
    )

    mnq_single = summary.loc[summary["portfolio_variant_name"].astype(str) == "MNQ_only__standalone__core_default"].iloc[0]
    mnq_m2k_pair = summary.loc[
        summary["portfolio_variant_name"].astype(str) == "MNQ_M2K__equal_weight_risk_budget__core_default"
    ].iloc[0]
    mnq_m2k_mes_three = summary.loc[
        summary["portfolio_variant_name"].astype(str) == "MNQ_M2K_MES__equal_weight_risk_budget__core_default"
    ].iloc[0]

    mes_incremental = {
        "net_pnl_delta_usd": float(mnq_m2k_mes_three["oos_net_pnl_usd"]) - float(mnq_m2k_pair["oos_net_pnl_usd"]),
        "sharpe_delta": float(mnq_m2k_mes_three["oos_sharpe"]) - float(mnq_m2k_pair["oos_sharpe"]),
        "maxdd_delta_usd": float(mnq_m2k_mes_three["oos_max_drawdown_usd"]) - float(mnq_m2k_pair["oos_max_drawdown_usd"]),
        "score_delta": float(mnq_m2k_mes_three["oos_portfolio_score"]) - float(mnq_m2k_pair["oos_portfolio_score"]),
    }

    corr_pairs = []
    if not correlation_matrix_daily.empty:
        for left in correlation_matrix_daily.index:
            for right in correlation_matrix_daily.columns:
                if left >= right:
                    continue
                corr_pairs.append(f"`{left}/{right}` `{float(correlation_matrix_daily.loc[left, right]):.3f}`")

    lines = [
        "# Volume Climax Pullback Micro Portfolio - Final Report",
        "",
        "## Scope",
        "- Motors combined without changing their alpha or execution assumptions.",
        f"- Portfolio capital reference: `{DEFAULT_INITIAL_CAPITAL_USD:,.0f} USD`.",
        f"- Capped overlay rule: open-risk cap `{DEFAULT_PORTFOLIO_RISK_CAP_USD:.0f} USD` on the sum of active initial trade risk; exits free capacity first; simultaneous entrants are prorated pro-rata.",
        "- MES default chosen explicitly from its robust zone: `risk_pct_0p0020__max_contracts_6__skip_trade_if_too_small_true`.",
        "- MES alternate tested inside the same zone: `risk_pct_0p0015__max_contracts_5__skip_trade_if_too_small_true`.",
        "",
        "## OOS Leaders",
        f"- Best single: `{best_single['portfolio_variant_name']}` | {_format_brief(best_single)}.",
        f"- Best pair: `{best_pair['portfolio_variant_name']}` | {_format_brief(best_pair)}.",
        f"- Best 3-way: `{best_three_way['portfolio_variant_name']}` | {_format_brief(best_three_way)}.",
        "",
        "```text",
        _top_table(summary),
        "```",
        "",
        "## Decision Readout",
        f"1. Meilleur single OOS: `{best_single['portfolio_variant_name']}`.",
        f"2. Meilleure pair OOS: `{best_pair['portfolio_variant_name']}`.",
        f"3. Meilleur 3-way OOS: `{best_three_way['portfolio_variant_name']}`.",
        (
            f"4. Meilleur portefeuille contre MNQ seul: recommended `{recommended['portfolio_variant_name']}` vs `MNQ_only__standalone__core_default` = "
            f"net `{float(recommended['oos_net_pnl_usd']) - float(mnq_single['oos_net_pnl_usd']):+.2f}` | "
            f"Sharpe `{float(recommended['oos_sharpe']) - float(mnq_single['oos_sharpe']):+.3f}` | "
            f"maxDD `{float(recommended['oos_max_drawdown_usd']) - float(mnq_single['oos_max_drawdown_usd']):+.2f}`."
        ),
        f"5. Correlations journalières OOS entre moteurs default: {', '.join(corr_pairs) if corr_pairs else 'n/a'}.",
        (
            "6. Valeur marginale de MES dans `MNQ + M2K` sous le schema risk-budget default: "
            f"net `{mes_incremental['net_pnl_delta_usd']:+.2f}` | Sharpe `{mes_incremental['sharpe_delta']:+.3f}` | "
            f"maxDD `{mes_incremental['maxdd_delta_usd']:+.2f}` | score `{mes_incremental['score_delta']:+.3f}`."
        ),
        f"7. Portefeuille recherche: `{recommended['portfolio_variant_name']}` | prop-safe: `{prop_safe['portfolio_variant_name']}` | agressif mais defendable: `{aggressive['portfolio_variant_name']}`.",
        f"8. Verdict net: `{_final_verdict_label(recommended)}`.",
        "",
        "## Interpretation",
        (
            f"- `MNQ + M2K` bat `MNQ` seul sur le score OOS de `{float(best_pair['oos_portfolio_score']) - float(mnq_single['oos_portfolio_score']):+.3f}`."
            if str(best_pair["portfolio_name"]) == "MNQ_M2K"
            else f"- La meilleure pair n'est pas `MNQ + M2K` mais `{best_pair['portfolio_variant_name']}`."
        ),
        (
            "- L'ajout de MES a une vraie valeur marginale."
            if mes_incremental["score_delta"] > 0 and mes_incremental["sharpe_delta"] >= 0
            else "- L'ajout de MES n'apporte pas assez de diversification marginale au portefeuille coeur `MNQ + M2K`."
        ),
        (
            "- Le portefeuille recommande un 3-way."
            if str(recommended["portfolio_name"]) == "MNQ_M2K_MES"
            else "- Le portefeuille recommande de rester sur une structure plus simple qu'un 3-way."
        ),
    ]

    report_path = output_dir / "final_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    verdict = {
        "recommended_portfolio_variant": str(recommended["portfolio_variant_name"]),
        "prop_safe_portfolio_variant": str(prop_safe["portfolio_variant_name"]),
        "aggressive_portfolio_variant": str(aggressive["portfolio_variant_name"]),
        "best_single_portfolio_variant": str(best_single["portfolio_variant_name"]),
        "best_pair_portfolio_variant": str(best_pair["portfolio_variant_name"]),
        "best_three_way_portfolio_variant": str(best_three_way["portfolio_variant_name"]),
        "final_verdict": _final_verdict_label(recommended),
    }
    _json_dump(output_dir / "final_verdict.json", verdict)
    return verdict


def run_campaign(
    *,
    output_root: Path | None = None,
    initial_capital_usd: float = DEFAULT_INITIAL_CAPITAL_USD,
    portfolio_risk_cap_usd: float = DEFAULT_PORTFOLIO_RISK_CAP_USD,
    motor_configs: dict[str, MotorConfigSpec] | None = None,
    portfolio_variants: list[PortfolioVariantSpec] | None = None,
) -> Path:
    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) if output_root is not None else EXPORTS_DIR / f"{DEFAULT_OUTPUT_PREFIX}{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_motor_configs = motor_configs or build_default_motor_configs()
    resolved_portfolio_variants = portfolio_variants or build_default_portfolio_variants()
    loaded_motors = {key: _load_motor_variant(spec) for key, spec in resolved_motor_configs.items()}
    calendar = build_master_calendar(loaded_motors)

    summary_rows: list[dict[str, Any]] = []
    trade_exports: list[pd.DataFrame] = []
    daily_exports: list[pd.DataFrame] = []
    motor_daily_exports: list[pd.DataFrame] = []
    diversification_rows: list[dict[str, Any]] = []

    for portfolio_variant in resolved_portfolio_variants:
        summary_row, trades, daily, motor_daily, diversification = summarize_portfolio_variant(
            portfolio_variant,
            loaded_motors=loaded_motors,
            calendar=calendar,
            initial_capital_usd=float(initial_capital_usd),
            portfolio_risk_cap_usd=float(portfolio_risk_cap_usd),
        )
        summary_rows.append(summary_row)
        trade_exports.append(trades)
        daily_exports.append(daily)
        motor_daily_exports.append(motor_daily)
        diversification_rows.extend(diversification)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["portfolio_family", "portfolio_name", "allocation_scheme", "config_bundle", "portfolio_variant_name"]
    ).reset_index(drop=True)
    for prefix in ("full", "oos"):
        summary[f"{prefix}_portfolio_score"] = summary.apply(lambda row: compute_portfolio_score(row, prefix=prefix), axis=1)

    summary.to_csv(output_dir / "summary_by_portfolio.csv", index=False)
    oos_only = summary[
        [
            "portfolio_variant_name",
            "portfolio_name",
            "portfolio_family",
            "allocation_scheme",
            "config_bundle",
            "member_motor_keys",
            "member_symbols",
            "member_config_labels",
            "motor_count",
            *[column for column in summary.columns if column.startswith("oos_")],
        ]
    ].copy()
    oos_only = oos_only.rename(columns={column: column[4:] for column in oos_only.columns if column.startswith("oos_")})
    oos_only.to_csv(output_dir / "summary_oos_only.csv", index=False)

    trade_export = _concat_non_empty(trade_exports)
    daily_export = _concat_non_empty(daily_exports)
    motor_daily_export = _concat_non_empty(motor_daily_exports)
    diversification_export = pd.DataFrame(diversification_rows)

    trade_export.to_csv(output_dir / "trades_by_portfolio.csv", index=False)
    daily_export.to_csv(output_dir / "daily_equity_by_portfolio.csv", index=False)
    motor_daily_export.to_csv(output_dir / "daily_pnl_by_motor.csv", index=False)
    diversification_export.to_csv(output_dir / "diversification_summary.csv", index=False)

    prop_columns = [
        "portfolio_variant_name",
        "portfolio_name",
        "portfolio_family",
        "allocation_scheme",
        "config_bundle",
        "full_nb_days_below_minus_250",
        "full_nb_days_below_minus_500",
        "full_nb_days_below_minus_1000",
        "full_pass_target_3000_usd_without_breaching_2000_dd",
        "full_days_to_3000_usd_if_reached",
        "full_max_trailing_drawdown_observed_usd",
        "full_max_static_drawdown_observed_usd",
        "oos_nb_days_below_minus_250",
        "oos_nb_days_below_minus_500",
        "oos_nb_days_below_minus_1000",
        "oos_pass_target_3000_usd_without_breaching_2000_dd",
        "oos_days_to_3000_usd_if_reached",
        "oos_max_trailing_drawdown_observed_usd",
        "oos_max_static_drawdown_observed_usd",
    ]
    summary[prop_columns].to_csv(output_dir / "prop_constraints_summary.csv", index=False)

    correlation_matrix_daily = _correlation_matrix_from_default_motors(loaded_motors, scope_name="oos")
    correlation_matrix_daily.to_csv(output_dir / "correlation_matrix_daily.csv")

    _plot_equity_curves(summary, daily_export, output_path=output_dir / "equity_curves_oos.png")
    _plot_drawdown_curves(summary, daily_export, output_path=output_dir / "drawdown_curves_oos.png")
    _plot_correlation_heatmap(correlation_matrix_daily, output_path=output_dir / "daily_pnl_correlation_heatmap.png")
    _plot_diversification_contribution(summary, diversification_export, output_path=output_dir / "diversification_contribution_chart.png")
    _plot_portfolio_score(summary, output_path=output_dir / "prop_score_by_portfolio.png")

    verdict = _build_final_report(
        output_dir=output_dir,
        summary=summary,
        diversification=diversification_export,
        correlation_matrix_daily=correlation_matrix_daily,
    )
    _json_dump(
        output_dir / "run_metadata.json",
        {
            "generated_at": datetime.now().isoformat(),
            "initial_capital_usd": float(initial_capital_usd),
            "portfolio_risk_cap_usd": float(portfolio_risk_cap_usd),
            "motor_configs": {
                key: {
                    "symbol": spec.symbol,
                    "config_label": spec.config_label,
                    "campaign_variant_name": spec.campaign_variant_name,
                    "risk_pct_decimal": spec.risk_pct_decimal,
                    "max_contracts": spec.max_contracts,
                    "export_root": spec.export_root,
                    "notes": spec.notes,
                }
                for key, spec in resolved_motor_configs.items()
            },
            "portfolio_variant_count": int(len(summary)),
            "portfolio_variants": [variant.portfolio_variant_name for variant in resolved_portfolio_variants],
            "verdict": verdict,
        },
    )
    return output_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--initial-capital-usd", type=float, default=DEFAULT_INITIAL_CAPITAL_USD)
    parser.add_argument("--portfolio-risk-cap-usd", type=float, default=DEFAULT_PORTFOLIO_RISK_CAP_USD)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    output_dir = run_campaign(
        output_root=args.output_root,
        initial_capital_usd=args.initial_capital_usd,
        portfolio_risk_cap_usd=args.portfolio_risk_cap_usd,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
