from __future__ import annotations

import pandas as pd

from src.analytics.mnq_orb_topstep_50k_simulation import (
    TopstepRuleset,
    aggregate_topstep_runs,
    build_comparison_table,
    simulate_topstep_path,
)


def _daily(rows: list[tuple[str, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["session_date", "daily_pnl_usd", "daily_trade_count"])


def test_trailing_mll_tracks_high_watermark_without_lock() -> None:
    ruleset = TopstepRuleset(
        name="demo",
        description="demo",
        starting_balance_usd=50_000.0,
        profit_target_usd=10_000.0,
        trailing_mll_usd=2_000.0,
        max_traded_days=5,
        lock_at_starting_balance=False,
    )
    daily = _daily(
        [
            ("2024-01-02", 500.0, 1),
            ("2024-01-03", 700.0, 1),
            ("2024-01-04", -2_100.0, 1),
        ]
    )

    history, result = simulate_topstep_path(daily_results=daily, ruleset=ruleset)

    assert float(history.iloc[0]["trailing_floor_usd"]) == 48_500.0
    assert float(history.iloc[1]["trailing_floor_usd"]) == 49_200.0
    assert bool(result["fail"]) is True
    assert result["failure_reason"] == "trailing_mll"


def test_consistency_can_delay_pass_after_economic_target() -> None:
    ruleset = TopstepRuleset(
        name="demo",
        description="demo",
        starting_balance_usd=50_000.0,
        profit_target_usd=3_000.0,
        trailing_mll_usd=2_000.0,
        max_traded_days=5,
    )
    daily = _daily(
        [
            ("2024-01-02", 1_800.0, 1),
            ("2024-01-03", 1_500.0, 1),
            ("2024-01-04", 500.0, 1),
        ]
    )

    _, result = simulate_topstep_path(daily_results=daily, ruleset=ruleset)

    assert bool(result["economic_target_hit"]) is True
    assert bool(result["economic_target_immediate_validation"]) is False
    assert bool(result["delayed_pass_after_inconsistency"]) is True
    assert float(result["extra_traded_days_to_consistency"]) == 1.0
    assert bool(result["pass"]) is True


def test_run_can_fail_after_hitting_economic_target_if_consistency_not_met() -> None:
    ruleset = TopstepRuleset(
        name="demo",
        description="demo",
        starting_balance_usd=50_000.0,
        profit_target_usd=3_000.0,
        trailing_mll_usd=2_000.0,
        max_traded_days=5,
    )
    daily = _daily(
        [
            ("2024-01-02", 1_800.0, 1),
            ("2024-01-03", 1_500.0, 1),
            ("2024-01-04", -2_600.0, 1),
        ]
    )

    _, result = simulate_topstep_path(daily_results=daily, ruleset=ruleset)

    assert bool(result["economic_target_hit"]) is True
    assert bool(result["economic_target_immediate_validation"]) is False
    assert bool(result["failed_after_economic_target"]) is True
    assert bool(result["fail"]) is True


def test_expiry_is_respected_when_consistency_never_catches_up() -> None:
    ruleset = TopstepRuleset(
        name="demo",
        description="demo",
        starting_balance_usd=50_000.0,
        profit_target_usd=3_000.0,
        trailing_mll_usd=2_000.0,
        max_traded_days=2,
    )
    daily = _daily(
        [
            ("2024-01-02", 1_800.0, 1),
            ("2024-01-03", 1_500.0, 1),
            ("2024-01-04", 100.0, 1),
        ]
    )

    _, result = simulate_topstep_path(daily_results=daily, ruleset=ruleset)

    assert result["status"] == "expire"
    assert bool(result["economic_target_without_immediate_validation"]) is True


def test_comparison_table_can_flag_survival_speed_tradeoff() -> None:
    nominal_runs = pd.DataFrame(
        [
            {"pass": True, "fail": False, "expire": False, "days_to_pass": 10.0, "days_to_fail": float("nan"), "trailing_mll_breached": False, "economic_target_hit": True, "economic_target_immediate_validation": True, "economic_target_without_immediate_validation": False, "delayed_pass_after_inconsistency": False, "failed_after_economic_target": False, "near_validation_then_fail": False, "extra_traded_days_to_consistency": float("nan"), "final_pnl_usd": 3_500.0, "max_drawdown_usd": -700.0},
            {"pass": False, "fail": True, "expire": False, "days_to_pass": float("nan"), "days_to_fail": 8.0, "trailing_mll_breached": True, "economic_target_hit": False, "economic_target_immediate_validation": False, "economic_target_without_immediate_validation": False, "delayed_pass_after_inconsistency": False, "failed_after_economic_target": False, "near_validation_then_fail": False, "extra_traded_days_to_consistency": float("nan"), "final_pnl_usd": -2_000.0, "max_drawdown_usd": -2_000.0},
        ]
    )
    sizing_runs = pd.DataFrame(
        [
            {"pass": True, "fail": False, "expire": False, "days_to_pass": 14.0, "days_to_fail": float("nan"), "trailing_mll_breached": False, "economic_target_hit": True, "economic_target_immediate_validation": False, "economic_target_without_immediate_validation": True, "delayed_pass_after_inconsistency": True, "failed_after_economic_target": False, "near_validation_then_fail": False, "extra_traded_days_to_consistency": 2.0, "final_pnl_usd": 3_500.0, "max_drawdown_usd": -500.0},
            {"pass": True, "fail": False, "expire": False, "days_to_pass": 16.0, "days_to_fail": float("nan"), "trailing_mll_breached": False, "economic_target_hit": True, "economic_target_immediate_validation": False, "economic_target_without_immediate_validation": True, "delayed_pass_after_inconsistency": True, "failed_after_economic_target": False, "near_validation_then_fail": False, "extra_traded_days_to_consistency": 3.0, "final_pnl_usd": 3_800.0, "max_drawdown_usd": -400.0},
        ]
    )

    summary = pd.DataFrame(
        [
            {"ruleset_name": "demo", "variant_name": "nominal", **{f"rolling_{k}": v for k, v in aggregate_topstep_runs(nominal_runs).items()}, **{f"bootstrap_{k}": v for k, v in aggregate_topstep_runs(nominal_runs).items()}},
            {"ruleset_name": "demo", "variant_name": "sizing_3state_realized_vol_ratio_15_60", **{f"rolling_{k}": v for k, v in aggregate_topstep_runs(sizing_runs).items()}, **{f"bootstrap_{k}": v for k, v in aggregate_topstep_runs(sizing_runs).items()}},
        ]
    )

    comparison = build_comparison_table(summary)

    assert comparison.iloc[0]["verdict"] == "TopstepX 50K favorise sizing_3state"
