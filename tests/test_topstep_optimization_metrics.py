from __future__ import annotations

import math

import pandas as pd

from src.analytics.topstep_optimization.metrics import build_ranking_table, summarize_simulation_results


def test_summarize_simulation_results_computes_economic_fields() -> None:
    runs = pd.DataFrame(
        {
            "simulation_mode": ["historical_rolling"] * 4,
            "variant": ["nominal"] * 4,
            "base_variant": ["nominal"] * 4,
            "source_variant_name": ["nominal"] * 4,
            "leverage_factor": [1.0] * 4,
            "pass": [True, True, False, False],
            "fail": [False, False, True, False],
            "expire": [False, False, False, True],
            "days_to_pass": [10.0, 20.0, math.nan, math.nan],
            "days_to_fail": [math.nan, math.nan, 15.0, math.nan],
            "cycle_trading_days": [10.0, 20.0, 15.0, 60.0],
        }
    )

    summary = summarize_simulation_results(runs, payout_value=3000.0, reset_cost=100.0)
    row = summary.iloc[0]

    assert row["pass_rate"] == 0.5
    assert row["fail_rate"] == 0.25
    assert row["expire_rate"] == 0.25
    assert row["avg_time_to_pass"] == 15.0
    assert row["avg_time_to_fail"] == 15.0
    assert row["expected_profit_per_cycle"] == 1475.0
    assert row["expected_days_per_cycle"] == 26.25
    assert round(float(row["expected_profit_per_day"]), 6) == round(1475.0 / 26.25, 6)
    assert row["probability_pass_within_20_days"] == 0.5
    assert row["probability_pass_within_30_days"] == 0.5


def test_build_ranking_table_sorts_by_expected_profit_per_day_desc() -> None:
    summary = pd.DataFrame(
        {
            "simulation_mode": ["historical_rolling", "block_bootstrap"],
            "variant": ["nominal", "sizing_3state_x1.2"],
            "base_variant": ["nominal", "sizing_3state"],
            "source_variant_name": ["nominal", "sizing_3state_realized_vol_ratio_15_60"],
            "leverage_factor": [1.0, 1.2],
            "expected_profit_per_day": [10.0, 15.0],
            "expected_profit_per_cycle": [1000.0, 1200.0],
            "pass_rate": [0.5, 0.55],
            "avg_time_to_pass": [15.0, 14.0],
        }
    )

    ranking = build_ranking_table(summary)

    assert list(ranking["variant"]) == ["sizing_3state_x1.2", "nominal"]
    assert list(ranking["rank"]) == [1, 2]
