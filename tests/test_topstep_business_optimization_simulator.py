from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.analytics.topstep_business_optimization.simulator import (
    BusinessRules,
    TopstepPlan,
    prepare_strategy_series,
    simulate_business_cycle,
    simulate_challenge_attempt,
    simulate_funded_phase,
)


def _daily_frame(pnls: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_date": pd.date_range("2024-01-01", periods=len(pnls), freq="D"),
            "daily_pnl_usd": pnls,
            "daily_gross_pnl_usd": pnls,
            "daily_fees_usd": [0.0] * len(pnls),
            "daily_trade_count": [1] * len(pnls),
            "daily_loss_count": [1 if pnl < 0 else 0 for pnl in pnls],
        }
    )


def test_simulate_challenge_attempt_passes_on_profit_target() -> None:
    strategy = prepare_strategy_series(_daily_frame([900.0, 1100.0, 1200.0]), "nominal", "nominal")
    rules = BusinessRules(challenge_safety_max_days=10)

    result = simulate_challenge_attempt(strategy=strategy, rules=rules, block_size=3, rng=np.random.default_rng(42))

    assert result["status"] == "pass"
    assert result["days"] == 3


def test_simulate_challenge_attempt_fails_on_daily_loss_limit() -> None:
    strategy = prepare_strategy_series(_daily_frame([250.0, -1001.0, 3000.0]), "nominal", "nominal")
    rules = BusinessRules(challenge_safety_max_days=10)

    result = simulate_challenge_attempt(strategy=strategy, rules=rules, block_size=3, rng=np.random.default_rng(7))

    assert result["status"] == "fail"
    assert result["failure_reason"] == "daily_loss_limit"
    assert result["days"] == 2


def test_simulate_funded_phase_counts_multiple_payouts_before_breach() -> None:
    strategy = prepare_strategy_series(_daily_frame([600.0, 600.0, 900.0, -1001.0]), "sizing_3state", "sizing")
    rules = BusinessRules(funded_trading_days=4, payout_threshold_usd=1000.0)

    result = simulate_funded_phase(strategy=strategy, rules=rules, block_size=4, rng=np.random.default_rng(11))

    assert result["status"] == "breach"
    assert result["payouts"] == 2
    assert result["realized_payout_value_usd"] == 2000.0
    assert result["at_least_one_payout"] is True


def test_simulate_business_cycle_includes_activation_costs_and_realized_payouts() -> None:
    challenge = prepare_strategy_series(_daily_frame([3500.0]), "nominal", "nominal")
    funded = prepare_strategy_series(_daily_frame([600.0, 600.0, 100.0]), "sizing_3state", "sizing")
    plan = TopstepPlan(name="standard", subscription_monthly_usd=49.0, reset_cost_usd=49.0, activation_fee_usd=149.0)
    rules = BusinessRules(funded_trading_days=3, payout_threshold_usd=1000.0)

    result = simulate_business_cycle(
        plan=plan,
        challenge_strategy=challenge,
        funded_strategy=funded,
        rules=rules,
        block_size=3,
        rng=np.random.default_rng(5),
    )

    assert result["eventual_pass"] is True
    assert result["first_attempt_pass"] is True
    assert result["resets"] == 0
    assert result["payouts"] == 1
    assert result["realized_payout_value_usd"] == 1000.0
    assert math.isclose(result["subscription_cost_usd"], 49.0 / 30.0, rel_tol=1e-9)
    assert math.isclose(result["total_cost_usd"], (49.0 / 30.0) + 149.0, rel_tol=1e-9)
    assert math.isclose(result["net_profit_usd"], 1000.0 - ((49.0 / 30.0) + 149.0), rel_tol=1e-9)


def test_simulate_business_cycle_counts_resets_when_never_passing() -> None:
    challenge = prepare_strategy_series(_daily_frame([-1001.0]), "nominal", "nominal")
    funded = prepare_strategy_series(_daily_frame([500.0]), "nominal", "nominal")
    plan = TopstepPlan(name="standard", subscription_monthly_usd=49.0, reset_cost_usd=49.0, activation_fee_usd=149.0)
    rules = BusinessRules(max_challenge_attempts=3, funded_trading_days=1)

    result = simulate_business_cycle(
        plan=plan,
        challenge_strategy=challenge,
        funded_strategy=funded,
        rules=rules,
        block_size=1,
        rng=np.random.default_rng(17),
    )

    assert result["eventual_pass"] is False
    assert result["resets"] == 3
    assert result["funded_status"] == "not_started"
    assert result["activation_fee_usd"] == 0.0
    assert math.isclose(result["reset_cost_usd"], 3 * 49.0, rel_tol=1e-9)
