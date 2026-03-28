"""Business-level Topstep simulation with challenge resets and funded payouts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.mnq_orb_prop_challenge_simulation import _normalize_daily_results


@dataclass(frozen=True)
class TopstepPlan:
    name: str
    subscription_monthly_usd: float
    reset_cost_usd: float
    activation_fee_usd: float


@dataclass(frozen=True)
class BusinessRules:
    starting_balance_usd: float = 50_000.0
    challenge_profit_target_usd: float = 3_000.0
    trailing_drawdown_usd: float = 2_000.0
    daily_loss_limit_usd: float = 1_000.0
    funded_trading_days: int = 60
    payout_threshold_usd: float = 1_000.0
    challenge_safety_max_days: int = 360
    max_challenge_attempts: int = 50
    subscription_days_per_month: float = 30.0


@dataclass(frozen=True)
class StrategySeries:
    strategy_name: str
    source_variant_name: str
    leverage_factor: float
    daily_pnl: np.ndarray
    daily_trade_count: np.ndarray


def _effective_block_size(n_obs: int, block_size: int) -> int:
    if int(n_obs) <= 0:
        return 0
    return max(1, min(int(block_size), int(n_obs)))


def build_strategy_name(base_name: str, leverage_factor: float) -> str:
    if abs(float(leverage_factor) - 1.0) < 1e-12:
        return str(base_name)
    return f"{base_name}_x{float(leverage_factor):.1f}"


def prepare_strategy_series(
    daily_results: pd.DataFrame,
    strategy_name: str,
    source_variant_name: str,
    leverage_factor: float = 1.0,
) -> StrategySeries:
    ordered = _normalize_daily_results(daily_results)
    pnl = pd.to_numeric(ordered["daily_pnl_usd"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    trade_count = pd.to_numeric(ordered["daily_trade_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    factor = float(leverage_factor)
    if abs(factor - 1.0) > 1e-12:
        pnl = pnl * factor
    return StrategySeries(
        strategy_name=build_strategy_name(strategy_name, leverage_factor),
        source_variant_name=source_variant_name,
        leverage_factor=float(leverage_factor),
        daily_pnl=pnl,
        daily_trade_count=trade_count,
    )


def _draw_block_start(n_obs: int, block_size: int, rng: np.random.Generator) -> int:
    effective_block = _effective_block_size(n_obs, block_size)
    if effective_block <= 0:
        return 0
    max_start = max(int(n_obs) - effective_block, 0)
    return int(rng.integers(0, max_start + 1)) if max_start > 0 else 0


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(float(denominator)):
        return float(default)
    value = float(numerator) / float(denominator)
    return float(value) if math.isfinite(value) else float(default)


def simulate_challenge_attempt(
    strategy: StrategySeries,
    rules: BusinessRules,
    block_size: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    effective_block = _effective_block_size(len(strategy.daily_pnl), block_size)
    if effective_block <= 0:
        return {
            "status": "timeout",
            "failure_reason": "empty_series",
            "days": 0,
            "final_equity_usd": float(rules.starting_balance_usd),
        }

    equity = float(rules.starting_balance_usd)
    high_watermark = float(rules.starting_balance_usd)
    days = 0
    failure_reason = ""

    while days < int(rules.challenge_safety_max_days):
        start_idx = _draw_block_start(len(strategy.daily_pnl), block_size=effective_block, rng=rng)
        end_idx = min(start_idx + effective_block, len(strategy.daily_pnl))
        for idx in range(start_idx, end_idx):
            days += 1
            daily_pnl = float(strategy.daily_pnl[idx])
            equity += daily_pnl

            if daily_pnl < -float(rules.daily_loss_limit_usd):
                failure_reason = "daily_loss_limit"
                return {
                    "status": "fail",
                    "failure_reason": failure_reason,
                    "days": int(days),
                    "final_equity_usd": float(equity),
                }

            high_watermark = max(high_watermark, equity)
            if equity <= high_watermark - float(rules.trailing_drawdown_usd):
                failure_reason = "trailing_drawdown"
                return {
                    "status": "fail",
                    "failure_reason": failure_reason,
                    "days": int(days),
                    "final_equity_usd": float(equity),
                }

            if equity >= float(rules.starting_balance_usd + rules.challenge_profit_target_usd):
                return {
                    "status": "pass",
                    "failure_reason": "",
                    "days": int(days),
                    "final_equity_usd": float(equity),
                }

    return {
        "status": "timeout",
        "failure_reason": "challenge_timeout",
        "days": int(rules.challenge_safety_max_days),
        "final_equity_usd": float(equity),
    }


def simulate_funded_phase(
    strategy: StrategySeries,
    rules: BusinessRules,
    block_size: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    effective_block = _effective_block_size(len(strategy.daily_pnl), block_size)
    if effective_block <= 0:
        return {
            "status": "survived",
            "failure_reason": "empty_series",
            "days": 0,
            "payouts": 0,
            "realized_payout_value_usd": 0.0,
            "funded_profit_usd": 0.0,
            "at_least_one_payout": False,
        }

    equity = float(rules.starting_balance_usd)
    high_watermark = float(rules.starting_balance_usd)
    days = 0
    payouts = 0
    next_payout_threshold = float(rules.payout_threshold_usd)
    failure_reason = ""

    while days < int(rules.funded_trading_days):
        start_idx = _draw_block_start(len(strategy.daily_pnl), block_size=effective_block, rng=rng)
        end_idx = min(start_idx + effective_block, len(strategy.daily_pnl))
        for idx in range(start_idx, end_idx):
            if days >= int(rules.funded_trading_days):
                break
            days += 1
            daily_pnl = float(strategy.daily_pnl[idx])
            equity += daily_pnl

            if daily_pnl < -float(rules.daily_loss_limit_usd):
                failure_reason = "daily_loss_limit"
                final_profit = float(equity - float(rules.starting_balance_usd))
                return {
                    "status": "breach",
                    "failure_reason": failure_reason,
                    "days": int(days),
                    "payouts": int(payouts),
                    "realized_payout_value_usd": float(payouts * rules.payout_threshold_usd),
                    "funded_profit_usd": final_profit,
                    "at_least_one_payout": bool(payouts > 0),
                }

            high_watermark = max(high_watermark, equity)
            if equity <= high_watermark - float(rules.trailing_drawdown_usd):
                failure_reason = "trailing_drawdown"
                final_profit = float(equity - float(rules.starting_balance_usd))
                return {
                    "status": "breach",
                    "failure_reason": failure_reason,
                    "days": int(days),
                    "payouts": int(payouts),
                    "realized_payout_value_usd": float(payouts * rules.payout_threshold_usd),
                    "funded_profit_usd": final_profit,
                    "at_least_one_payout": bool(payouts > 0),
                }

            cumulative_profit = float(equity - float(rules.starting_balance_usd))
            while cumulative_profit >= next_payout_threshold:
                payouts += 1
                next_payout_threshold += float(rules.payout_threshold_usd)

    final_profit = float(equity - float(rules.starting_balance_usd))
    return {
        "status": "survived",
        "failure_reason": failure_reason,
        "days": int(days),
        "payouts": int(payouts),
        "realized_payout_value_usd": float(payouts * rules.payout_threshold_usd),
        "funded_profit_usd": final_profit,
        "at_least_one_payout": bool(payouts > 0),
    }


def simulate_business_cycle(
    plan: TopstepPlan,
    challenge_strategy: StrategySeries,
    funded_strategy: StrategySeries,
    rules: BusinessRules,
    block_size: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    attempts = 0
    resets = 0
    first_attempt_pass = False
    eventual_pass = False
    total_challenge_days = 0
    total_subscription_cost = 0.0
    total_reset_cost = 0.0
    challenge_timeout_count = 0
    last_challenge_status = ""
    last_challenge_failure_reason = ""

    while attempts < int(rules.max_challenge_attempts):
        attempts += 1
        challenge_result = simulate_challenge_attempt(
            strategy=challenge_strategy,
            rules=rules,
            block_size=block_size,
            rng=rng,
        )
        last_challenge_status = str(challenge_result["status"])
        last_challenge_failure_reason = str(challenge_result["failure_reason"])
        attempt_days = int(challenge_result["days"])
        total_challenge_days += attempt_days
        total_subscription_cost += _safe_div(plan.subscription_monthly_usd, rules.subscription_days_per_month) * attempt_days

        if challenge_result["status"] == "pass":
            eventual_pass = True
            first_attempt_pass = attempts == 1
            break

        if challenge_result["status"] == "timeout":
            challenge_timeout_count += 1
        resets += 1
        total_reset_cost += float(plan.reset_cost_usd)

    activation_fee = float(plan.activation_fee_usd) if eventual_pass else 0.0
    funded_result = {
        "status": "not_started",
        "failure_reason": "",
        "days": 0,
        "payouts": 0,
        "realized_payout_value_usd": 0.0,
        "funded_profit_usd": 0.0,
        "at_least_one_payout": False,
    }
    if eventual_pass:
        funded_result = simulate_funded_phase(
            strategy=funded_strategy,
            rules=rules,
            block_size=block_size,
            rng=rng,
        )

    total_days = int(total_challenge_days + int(funded_result["days"]))
    total_cost = float(total_subscription_cost + total_reset_cost + activation_fee)
    realized_payout_value = float(funded_result["realized_payout_value_usd"])
    net_profit = float(realized_payout_value - total_cost)

    return {
        "plan": plan.name,
        "challenge_strategy": challenge_strategy.strategy_name,
        "challenge_source_variant_name": challenge_strategy.source_variant_name,
        "challenge_leverage_factor": float(challenge_strategy.leverage_factor),
        "funded_strategy": funded_strategy.strategy_name,
        "funded_source_variant_name": funded_strategy.source_variant_name,
        "funded_leverage_factor": float(funded_strategy.leverage_factor),
        "attempts": int(attempts),
        "first_attempt_pass": bool(first_attempt_pass),
        "eventual_pass": bool(eventual_pass),
        "avg_attempt_pass_proxy": _safe_div(1.0, attempts, default=0.0) if eventual_pass else 0.0,
        "resets": int(resets),
        "challenge_days_to_pass": float(total_challenge_days) if eventual_pass else float("nan"),
        "challenge_days_total": int(total_challenge_days),
        "challenge_timeout_count": int(challenge_timeout_count),
        "last_challenge_status": last_challenge_status,
        "last_challenge_failure_reason": last_challenge_failure_reason,
        "funded_status": str(funded_result["status"]),
        "funded_failure_reason": str(funded_result["failure_reason"]),
        "funded_days": int(funded_result["days"]),
        "payouts": int(funded_result["payouts"]),
        "at_least_one_payout": bool(funded_result["at_least_one_payout"]),
        "realized_payout_value_usd": realized_payout_value,
        "funded_profit_usd": float(funded_result["funded_profit_usd"]),
        "subscription_cost_usd": float(total_subscription_cost),
        "reset_cost_usd": float(total_reset_cost),
        "activation_fee_usd": activation_fee,
        "total_cost_usd": total_cost,
        "net_profit_usd": net_profit,
        "total_days": int(total_days),
        "net_profit_per_day": _safe_div(net_profit, total_days, default=0.0),
    }
