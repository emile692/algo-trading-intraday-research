import pytest

from src.analytics.meta_risk_campaign import (
    ChallengeSimulationConfig,
    VARIANT_HALF_AFTER_2_LOSSES,
    VARIANT_LOCAL_DD_SCALING,
    VARIANT_SKIP_AFTER_3_LOSSES,
    compute_policy_multipliers,
    simulate_prop_challenge,
)


def test_policy_multipliers_half_after_two_losses() -> None:
    pnl_path = [-100.0, -50.0, 200.0, -10.0]
    multipliers = compute_policy_multipliers(
        VARIANT_HALF_AFTER_2_LOSSES,
        pnl_path,
        initial_capital=50_000.0,
    )
    assert multipliers == [1.0, 1.0, 0.5, 1.0]


def test_policy_multipliers_skip_after_three_losses() -> None:
    pnl_path = [-100.0, -50.0, -25.0, 300.0, -10.0]
    multipliers = compute_policy_multipliers(
        VARIANT_SKIP_AFTER_3_LOSSES,
        pnl_path,
        initial_capital=50_000.0,
    )
    assert multipliers == [1.0, 1.0, 1.0, 0.0, 1.0]


def test_policy_multipliers_local_drawdown_scaling() -> None:
    pnl_path = [-0.5, -0.6, -1.2, 5.0]
    multipliers = compute_policy_multipliers(
        VARIANT_LOCAL_DD_SCALING,
        pnl_path,
        initial_capital=100.0,
    )
    assert multipliers == [1.0, 1.0, 0.5, 0.0]


def test_simulate_prop_challenge_extreme_cases() -> None:
    success = simulate_prop_challenge(
        daily_pnl=[3000.0],
        account_size_usd=50_000.0,
        config=ChallengeSimulationConfig(
            target_return_pct=0.06,
            max_drawdown_pct=0.04,
            n_bootstrap_paths=200,
            random_seed=123,
        ),
    )
    assert success["challenge_pass_rate"] == pytest.approx(1.0)
    assert success["challenge_median_days_to_target"] == pytest.approx(1.0)
    assert bool(success["historical_path_pass"]) is True

    failure = simulate_prop_challenge(
        daily_pnl=[-3000.0],
        account_size_usd=50_000.0,
        config=ChallengeSimulationConfig(
            target_return_pct=0.06,
            max_drawdown_pct=0.04,
            n_bootstrap_paths=200,
            random_seed=123,
        ),
    )
    assert failure["challenge_pass_rate"] == pytest.approx(0.0)
    assert bool(failure["historical_path_pass"]) is False
