"""Simulation engine for Topstep-style account optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.mnq_orb_prop_challenge_simulation import _normalize_daily_results


@dataclass(frozen=True)
class TopstepRules:
    starting_balance_usd: float = 50_000.0
    profit_target_usd: float = 3_000.0
    trailing_drawdown_usd: float = 2_000.0
    daily_loss_limit_usd: float = 1_000.0
    max_trading_days: int = 60


@dataclass(frozen=True)
class ScaledVariantSeries:
    variant: str
    base_variant: str
    source_variant_name: str
    leverage_factor: float
    daily_results: pd.DataFrame
    reference_account_size_usd: float = 50_000.0


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(float(denominator)):
        return float(default)
    value = float(numerator) / float(denominator)
    return float(value) if math.isfinite(value) else float(default)


def build_variant_name(base_variant: str, leverage_factor: float) -> str:
    if abs(float(leverage_factor) - 1.0) < 1e-12:
        return str(base_variant)
    return f"{base_variant}_x{float(leverage_factor):.1f}"


def scale_daily_results(daily_results: pd.DataFrame, leverage_factor: float) -> pd.DataFrame:
    scaled = _normalize_daily_results(daily_results)
    factor = float(leverage_factor)
    if abs(factor - 1.0) < 1e-12:
        return scaled
    out = scaled.copy()
    for column in ("daily_pnl_usd", "daily_gross_pnl_usd", "daily_fees_usd"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0) * factor
    return out


def eligible_start_dates(daily_results: pd.DataFrame, max_trading_days: int) -> list:
    ordered = _normalize_daily_results(daily_results)
    traded_mask = pd.to_numeric(ordered["daily_trade_count"], errors="coerce").fillna(0.0).gt(0.0).astype(int)
    remaining_traded_days = traded_mask.iloc[::-1].cumsum().iloc[::-1]
    return ordered.loc[remaining_traded_days >= int(max_trading_days), "session_date"].tolist()


def sample_block_bootstrap_daily(
    daily_results: pd.DataFrame,
    rules: TopstepRules,
    block_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    ordered = _normalize_daily_results(daily_results)
    if ordered.empty:
        return ordered.copy()

    effective_block = max(1, min(int(block_size), len(ordered)))
    max_start = max(len(ordered) - effective_block, 0)
    sampled_rows: list[dict[str, Any]] = []
    traded_days = 0
    synthetic_idx = 0
    max_rows = max(len(ordered) * 4, rules.max_trading_days * 5)

    while len(sampled_rows) < max_rows:
        start_idx = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        block = ordered.iloc[start_idx : start_idx + effective_block]
        for _, row in block.iterrows():
            synthetic_idx += 1
            row_dict = row.to_dict()
            row_dict["source_session_date"] = row_dict["session_date"]
            row_dict["session_date"] = synthetic_idx
            sampled_rows.append(row_dict)
            if float(row.get("daily_trade_count", 0.0)) > 0.0:
                traded_days += 1
            if traded_days >= int(rules.max_trading_days):
                return _normalize_daily_results(pd.DataFrame(sampled_rows))

    return _normalize_daily_results(pd.DataFrame(sampled_rows))


def simulate_account_path(
    daily_results: pd.DataFrame,
    rules: TopstepRules,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered = _normalize_daily_results(daily_results)

    equity = float(rules.starting_balance_usd)
    high_watermark = float(rules.starting_balance_usd)
    traded_days = 0
    calendar_days = 0
    history_rows: list[dict[str, Any]] = []

    status = "open"
    failure_reason = ""
    days_to_pass = float("nan")
    days_to_fail = float("nan")
    daily_loss_limit_breached = False
    trailing_drawdown_breached = False

    for _, row in ordered.iterrows():
        calendar_days += 1
        daily_pnl = float(pd.to_numeric(pd.Series([row.get("daily_pnl_usd", 0.0)]), errors="coerce").iloc[0])
        traded = bool(float(pd.to_numeric(pd.Series([row.get("daily_trade_count", 0.0)]), errors="coerce").iloc[0]) > 0.0)
        if traded:
            traded_days += 1

        equity += daily_pnl
        high_watermark = max(high_watermark, equity)
        trailing_floor = float(high_watermark - float(rules.trailing_drawdown_usd))
        cumulative_profit = float(equity - float(rules.starting_balance_usd))

        daily_loss_limit_breached = bool(daily_pnl < -float(rules.daily_loss_limit_usd))
        trailing_drawdown_breached = bool(equity <= trailing_floor)
        profit_target_reached = bool(cumulative_profit >= float(rules.profit_target_usd))

        history_rows.append(
            {
                "session_date": row["session_date"],
                "daily_pnl_usd": daily_pnl,
                "equity": equity,
                "high_watermark": high_watermark,
                "trailing_floor_usd": trailing_floor,
                "cumulative_profit_usd": cumulative_profit,
                "daily_loss_limit_breached": daily_loss_limit_breached,
                "trailing_drawdown_breached": trailing_drawdown_breached,
                "profit_target_reached": profit_target_reached,
                "traded_days_elapsed": traded_days,
                "calendar_days_elapsed": calendar_days,
            }
        )

        if daily_loss_limit_breached:
            status = "fail"
            failure_reason = "daily_loss_limit"
            days_to_fail = float(traded_days)
            break

        if trailing_drawdown_breached:
            status = "fail"
            failure_reason = "trailing_drawdown"
            days_to_fail = float(traded_days)
            break

        if profit_target_reached:
            status = "pass"
            days_to_pass = float(traded_days)
            break

        if traded_days >= int(rules.max_trading_days):
            status = "expire"
            break

    if status == "open":
        status = "expire"

    cycle_trading_days = (
        float(days_to_pass)
        if status == "pass"
        else float(days_to_fail)
        if status == "fail"
        else float(min(traded_days, int(rules.max_trading_days)))
    )

    history = pd.DataFrame(history_rows)
    return history, {
        "status": status,
        "pass": bool(status == "pass"),
        "fail": bool(status == "fail"),
        "expire": bool(status == "expire"),
        "failure_reason": failure_reason,
        "days_to_pass": days_to_pass,
        "days_to_fail": days_to_fail,
        "cycle_trading_days": cycle_trading_days,
        "trading_days_elapsed": int(min(traded_days, int(rules.max_trading_days))),
        "calendar_days_elapsed": int(calendar_days),
        "daily_loss_limit_breached": bool(daily_loss_limit_breached),
        "trailing_drawdown_breached": bool(trailing_drawdown_breached),
        "final_profit_usd": float(equity - float(rules.starting_balance_usd)),
        "final_equity_usd": float(equity),
    }


def run_historical_rolling_simulations(
    series: ScaledVariantSeries,
    rules: TopstepRules,
    common_start_dates: list,
) -> pd.DataFrame:
    ordered = _normalize_daily_results(series.daily_results)
    allowed_dates = set(pd.to_datetime(pd.Index(common_start_dates)).date)
    rows: list[dict[str, Any]] = []

    for idx, session_date in enumerate(ordered["session_date"]):
        if session_date not in allowed_dates:
            continue
        subset = ordered.iloc[idx:].copy().reset_index(drop=True)
        _, result = simulate_account_path(subset, rules=rules)
        rows.append(
            {
                "simulation_mode": "historical_rolling",
                "variant": series.variant,
                "base_variant": series.base_variant,
                "source_variant_name": series.source_variant_name,
                "leverage_factor": float(series.leverage_factor),
                "run_id": f"rolling_{idx}",
                "start_session_date": session_date,
                **result,
            }
        )

    return pd.DataFrame(rows)


def run_block_bootstrap_simulations(
    series: ScaledVariantSeries,
    rules: TopstepRules,
    bootstrap_paths: int,
    block_size: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(random_seed))

    for path_idx in range(int(bootstrap_paths)):
        sampled = sample_block_bootstrap_daily(series.daily_results, rules=rules, block_size=block_size, rng=rng)
        _, result = simulate_account_path(sampled, rules=rules)
        rows.append(
            {
                "simulation_mode": "block_bootstrap",
                "variant": series.variant,
                "base_variant": series.base_variant,
                "source_variant_name": series.source_variant_name,
                "leverage_factor": float(series.leverage_factor),
                "run_id": f"bootstrap_{path_idx}",
                "start_session_date": pd.NaT,
                **result,
            }
        )

    return pd.DataFrame(rows)
