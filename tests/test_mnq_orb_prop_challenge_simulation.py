from __future__ import annotations

import pandas as pd

from src.analytics.mnq_orb_prop_challenge_simulation import (
    PropChallengeRuleset,
    aggregate_simulation_runs,
    compare_ruleset_pair,
    simulate_challenge_path,
)


def _daily_frame(rows: list[tuple[str, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["session_date", "daily_pnl_usd", "daily_trade_count"])


def test_simulate_challenge_path_passes_before_expiry() -> None:
    ruleset = PropChallengeRuleset(
        name="test",
        family="classic",
        resembles="demo",
        description="demo",
        account_size_usd=50_000.0,
        profit_target_usd=1_000.0,
        max_traded_days=5,
        daily_loss_limit_usd=600.0,
        static_max_loss_usd=2_000.0,
    )
    daily = _daily_frame(
        [
            ("2024-01-02", 300.0, 1),
            ("2024-01-03", 350.0, 1),
            ("2024-01-04", 400.0, 1),
        ]
    )

    _, result = simulate_challenge_path(daily_results=daily, ruleset=ruleset)

    assert bool(result["pass"]) is True
    assert result["status"] == "pass"
    assert float(result["days_to_pass"]) == 3.0


def test_simulate_challenge_path_fails_on_daily_loss_limit() -> None:
    ruleset = PropChallengeRuleset(
        name="test",
        family="classic",
        resembles="demo",
        description="demo",
        account_size_usd=50_000.0,
        profit_target_usd=1_000.0,
        max_traded_days=5,
        daily_loss_limit_usd=300.0,
        static_max_loss_usd=2_000.0,
    )
    daily = _daily_frame(
        [
            ("2024-01-02", -350.0, 1),
            ("2024-01-03", 900.0, 1),
        ]
    )

    _, result = simulate_challenge_path(daily_results=daily, ruleset=ruleset)

    assert bool(result["fail"]) is True
    assert result["failure_reason"] == "daily_loss_limit"
    assert float(result["days_to_fail"]) == 1.0


def test_simulate_challenge_path_fails_on_trailing_drawdown() -> None:
    ruleset = PropChallengeRuleset(
        name="test",
        family="trailing",
        resembles="demo",
        description="demo",
        account_size_usd=1_000.0,
        profit_target_usd=2_000.0,
        max_traded_days=5,
        trailing_drawdown_usd=500.0,
    )
    daily = _daily_frame(
        [
            ("2024-01-02", 700.0, 1),
            ("2024-01-03", -600.0, 1),
        ]
    )

    _, result = simulate_challenge_path(daily_results=daily, ruleset=ruleset, reference_account_size_usd=1_000.0)

    assert bool(result["fail"]) is True
    assert result["failure_reason"] == "trailing_drawdown"
    assert bool(result["trailing_drawdown_breached"]) is True


def test_expiry_counts_only_traded_days() -> None:
    ruleset = PropChallengeRuleset(
        name="test",
        family="classic",
        resembles="demo",
        description="demo",
        account_size_usd=50_000.0,
        profit_target_usd=1_000.0,
        max_traded_days=2,
        static_max_loss_usd=2_000.0,
    )
    daily = _daily_frame(
        [
            ("2024-01-02", 0.0, 0),
            ("2024-01-03", 100.0, 1),
            ("2024-01-04", 0.0, 0),
            ("2024-01-05", 100.0, 1),
            ("2024-01-08", 400.0, 1),
        ]
    )

    _, result = simulate_challenge_path(daily_results=daily, ruleset=ruleset)

    assert result["status"] == "expire"
    assert int(result["days_traded"]) == 2
    assert int(result["calendar_days"]) == 4


def test_aggregate_and_compare_favor_sizing_when_pass_is_higher_and_fail_not_worse() -> None:
    nominal_runs = pd.DataFrame(
        [
            {"pass": True, "fail": False, "expire": False, "days_to_pass": 10.0, "days_to_fail": float("nan"), "final_pnl_usd": 3000.0, "daily_loss_limit_breached": False, "global_max_loss_breached": False, "static_max_loss_breached": False, "trailing_drawdown_breached": False, "daily_limit_before_half_target": False, "global_limit_before_half_target": False, "half_target_reached": True, "near_fail": False, "time_near_limit_share": 0.1, "max_favorable_excursion_usd": 3200.0, "max_adverse_excursion_usd": -400.0, "max_drawdown_usd": -400.0, "days_traded": 10},
            {"pass": False, "fail": True, "expire": False, "days_to_pass": float("nan"), "days_to_fail": 8.0, "final_pnl_usd": -2000.0, "daily_loss_limit_breached": True, "global_max_loss_breached": True, "static_max_loss_breached": True, "trailing_drawdown_breached": False, "daily_limit_before_half_target": True, "global_limit_before_half_target": True, "half_target_reached": False, "near_fail": False, "time_near_limit_share": 0.5, "max_favorable_excursion_usd": 200.0, "max_adverse_excursion_usd": -2100.0, "max_drawdown_usd": -2100.0, "days_traded": 8},
        ]
    )
    sizing_runs = pd.DataFrame(
        [
            {"pass": True, "fail": False, "expire": False, "days_to_pass": 9.0, "days_to_fail": float("nan"), "final_pnl_usd": 3000.0, "daily_loss_limit_breached": False, "global_max_loss_breached": False, "static_max_loss_breached": False, "trailing_drawdown_breached": False, "daily_limit_before_half_target": False, "global_limit_before_half_target": False, "half_target_reached": True, "near_fail": False, "time_near_limit_share": 0.1, "max_favorable_excursion_usd": 3200.0, "max_adverse_excursion_usd": -300.0, "max_drawdown_usd": -300.0, "days_traded": 9},
            {"pass": True, "fail": False, "expire": False, "days_to_pass": 11.0, "days_to_fail": float("nan"), "final_pnl_usd": 3000.0, "daily_loss_limit_breached": False, "global_max_loss_breached": False, "static_max_loss_breached": False, "trailing_drawdown_breached": False, "daily_limit_before_half_target": False, "global_limit_before_half_target": False, "half_target_reached": True, "near_fail": True, "time_near_limit_share": 0.2, "max_favorable_excursion_usd": 3300.0, "max_adverse_excursion_usd": -500.0, "max_drawdown_usd": -500.0, "days_traded": 11},
        ]
    )

    summary = pd.DataFrame(
        [
            {
                "ruleset_name": "demo",
                "variant_name": "nominal",
                **{f"rolling_start_{k}": v for k, v in aggregate_simulation_runs(nominal_runs).items()},
                **{f"bootstrap_{k}": v for k, v in aggregate_simulation_runs(nominal_runs).items()},
            },
            {
                "ruleset_name": "demo",
                "variant_name": "sizing_3state_realized_vol_ratio_15_60",
                **{f"rolling_start_{k}": v for k, v in aggregate_simulation_runs(sizing_runs).items()},
                **{f"bootstrap_{k}": v for k, v in aggregate_simulation_runs(sizing_runs).items()},
            },
        ]
    )

    comparison = compare_ruleset_pair(summary)

    assert comparison.iloc[0]["verdict"] == "sizing_3state meilleur candidat prop"
