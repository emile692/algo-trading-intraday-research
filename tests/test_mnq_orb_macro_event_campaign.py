from __future__ import annotations

import math

import pandas as pd

from src.analytics.mnq_orb_macro_event_campaign import (
    MacroVariantDefinition,
    apply_macro_overlay,
    assign_macro_day_cohorts,
    merge_strategy_with_calendar,
    summarize_challenge_runs,
)


def _daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2026-01-07", "2026-01-08", "2026-01-09"]),
            "daily_pnl_usd": [100.0, -200.0, 300.0],
            "daily_gross_pnl_usd": [110.0, -190.0, 315.0],
            "daily_fees_usd": [10.0, 10.0, 15.0],
            "daily_trade_count": [1.0, 1.0, 1.0],
            "daily_loss_count": [0.0, 1.0, 0.0],
            "equity": [50_100.0, 49_900.0, 50_200.0],
            "peak_equity": [50_100.0, 50_100.0, 50_200.0],
            "drawdown_usd": [0.0, -200.0, 0.0],
            "drawdown_pct": [0.0, 0.003992015968063872, 0.0],
            "green_day": [True, False, True],
        }
    )


def _calendar_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-08", "2026-01-09", "2026-01-10"]),
            "is_fomc_day": [False, True, False],
            "is_cpi_day": [True, False, False],
            "is_core_cpi_day": [False, False, False],
            "is_nfp_day": [False, False, False],
            "is_high_impact_macro_day": [True, True, False],
        }
    )


def test_merge_strategy_with_calendar_keeps_overlap_only() -> None:
    merged = merge_strategy_with_calendar(_daily_frame(), _calendar_frame())

    assert list(merged["session_date"]) == list(pd.to_datetime(["2026-01-08", "2026-01-09"]).date)
    assert list(merged["trade_date"]) == list(pd.to_datetime(["2026-01-08", "2026-01-09"]).date)


def test_assign_macro_day_cohorts_preserves_raw_flags_and_priority() -> None:
    frame = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2026-01-08", "2026-01-09", "2026-01-10", "2026-01-11"]).date,
            "is_fomc_day": [True, True, False, False],
            "is_cpi_day": [True, False, False, False],
            "is_core_cpi_day": [False, False, False, False],
            "is_nfp_day": [False, False, False, False],
            "is_high_impact_macro_day": [True, True, True, False],
        }
    )

    cohorts = assign_macro_day_cohorts(frame)

    assert cohorts.loc[0, "fomc_day"] == True
    assert cohorts.loc[0, "cpi_or_nfp_day"] == True
    assert cohorts.loc[0, "priority_bucket"] == "cpi_or_nfp_day"
    assert cohorts.loc[1, "priority_bucket"] == "fomc_day"
    assert cohorts.loc[2, "priority_bucket"] == "other_high_impact_macro_day"
    assert cohorts.loc[3, "priority_bucket"] == "normal_day"


def test_apply_macro_overlay_scales_targeted_days_and_recomputes_equity() -> None:
    merged = merge_strategy_with_calendar(_daily_frame(), _calendar_frame())
    variant = MacroVariantDefinition(
        name="skip_cpi_nfp",
        family="hard_filter",
        description="skip CPI/NFP",
        trigger_column="cpi_or_nfp_day",
        event_scale=0.0,
    )

    overlay = apply_macro_overlay(merged, variant=variant, initial_capital=50_000.0)

    assert overlay.loc[0, "daily_pnl_usd"] == 0.0
    assert overlay.loc[0, "daily_trade_count"] == 0.0
    assert overlay.loc[1, "daily_pnl_usd"] == 300.0
    assert overlay.loc[1, "equity"] == 50_300.0
    assert overlay.loc[1, "drawdown_usd"] == 0.0


def test_summarize_challenge_runs_adds_expected_net_profit_metrics() -> None:
    runs = pd.DataFrame(
        {
            "pass": [True, False, False],
            "fail": [False, True, True],
            "expire": [False, False, False],
            "days_to_pass": [4.0, math.nan, math.nan],
            "days_to_fail": [math.nan, 2.0, 5.0],
            "days_traded": [4, 2, 5],
            "final_pnl_usd": [3_100.0, -1_001.0, -2_100.0],
            "daily_loss_limit_breached": [False, True, False],
            "global_max_loss_breached": [False, False, True],
            "static_max_loss_breached": [False, False, False],
            "trailing_drawdown_breached": [False, False, True],
            "half_target_reached": [True, False, True],
            "daily_limit_before_half_target": [False, True, False],
            "global_limit_before_half_target": [False, False, True],
            "near_fail": [False, False, False],
            "time_near_limit_share": [0.0, 0.1, 0.2],
            "max_favorable_excursion_usd": [3_100.0, 150.0, 250.0],
            "max_adverse_excursion_usd": [-50.0, -1_001.0, -2_100.0],
            "max_drawdown_usd": [-50.0, -1_001.0, -2_100.0],
            "failure_reason": ["", "daily_loss_limit", "trailing_drawdown"],
        }
    )

    summary = summarize_challenge_runs(runs, payout_value_usd=3_000.0, reset_cost_usd=100.0)

    assert math.isclose(summary["pass_rate"], 1.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(summary["probability_breaching_daily_loss_limit"], 1.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(summary["probability_breaching_max_loss_limit"], 1.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(summary["expected_net_profit_per_cycle"], (1.0 / 3.0) * 3_000.0 - (2.0 / 3.0) * 100.0, rel_tol=1e-9)
    assert math.isclose(summary["expected_cycle_days"], (4.0 + 2.0 + 5.0) / 3.0, rel_tol=1e-9)
    assert summary["expected_days_to_pass"] == 4.0
