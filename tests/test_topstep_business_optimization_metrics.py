from __future__ import annotations

import math

import pandas as pd

from src.analytics.topstep_business_optimization.metrics import build_ranking_table, summarize_business_runs


def test_summarize_business_runs_computes_cycle_economics() -> None:
    runs = pd.DataFrame(
        {
            "plan": ["standard", "standard"],
            "challenge_strategy": ["nominal_x1.5", "nominal_x1.5"],
            "challenge_source_variant_name": ["nominal", "nominal"],
            "challenge_leverage_factor": [1.5, 1.5],
            "funded_strategy": ["sizing_3state", "sizing_3state"],
            "funded_source_variant_name": ["sizing_3state_realized_vol_ratio_15_60"] * 2,
            "funded_leverage_factor": [1.0, 1.0],
            "eventual_pass": [True, False],
            "first_attempt_pass": [True, False],
            "challenge_days_to_pass": [12.0, math.nan],
            "resets": [0, 3],
            "payouts": [2, 0],
            "at_least_one_payout": [True, False],
            "funded_profit_usd": [1800.0, 0.0],
            "subscription_cost_usd": [20.0, 50.0],
            "reset_cost_usd": [0.0, 147.0],
            "activation_fee_usd": [149.0, 0.0],
            "total_cost_usd": [169.0, 197.0],
            "net_profit_usd": [1831.0, -197.0],
            "total_days": [32, 25],
        }
    )

    summary = summarize_business_runs(runs)
    row = summary.iloc[0]

    assert row["pass_rate"] == 0.5
    assert row["first_attempt_pass_rate"] == 0.5
    assert row["avg_days_to_pass"] == 12.0
    assert row["avg_resets"] == 1.5
    assert row["avg_payouts"] == 1.0
    assert row["avg_profit_post_pass"] == 1800.0
    assert row["total_cost"] == 183.0
    assert row["net_profit"] == 817.0
    assert row["expected_net_profit_per_cycle"] == 817.0
    assert row["expected_days_per_cycle"] == 28.5
    assert math.isclose(row["expected_net_profit_per_day"], 817.0 / 28.5, rel_tol=1e-9)


def test_build_ranking_table_sorts_by_profit_per_day_desc() -> None:
    summary = pd.DataFrame(
        {
            "plan": ["standard", "no_activation_fee"],
            "challenge_strategy": ["nominal_x1.2", "nominal_x1.8"],
            "challenge_source_variant_name": ["nominal", "nominal"],
            "challenge_leverage_factor": [1.2, 1.8],
            "funded_strategy": ["nominal", "sizing_3state"],
            "funded_source_variant_name": ["nominal", "sizing_3state_realized_vol_ratio_15_60"],
            "funded_leverage_factor": [1.0, 1.0],
            "expected_net_profit_per_day": [20.0, 35.0],
            "expected_net_profit_per_cycle": [500.0, 650.0],
            "pass_rate": [0.4, 0.5],
            "avg_payouts": [0.5, 0.8],
        }
    )

    ranking = build_ranking_table(summary)

    assert list(ranking["challenge_strategy"]) == ["nominal_x1.8", "nominal_x1.2"]
    assert list(ranking["overall_rank"]) == [1, 2]
