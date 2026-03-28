import pytest

from src.analytics.mnq_orb_prop_survivability_campaign import (
    compute_drawdown_deleveraging_multipliers,
    compute_half_after_two_red_days_multipliers,
    compute_skip_after_large_loss_multipliers,
    compute_skip_after_three_red_days_multipliers,
)


def test_drawdown_deleveraging_multipliers_step_down_with_path() -> None:
    multipliers = compute_drawdown_deleveraging_multipliers(
        daily_pnl_sequence=[-2.0, -2.0, -2.0, 6.0],
        initial_capital=100.0,
    )

    assert multipliers == [1.0, 1.0, 0.75, 0.5]


def test_skip_after_large_loss_multipliers_use_calendar_cooldown() -> None:
    multipliers = compute_skip_after_large_loss_multipliers(
        daily_pnl_sequence=[-800.0, 0.0, 50.0],
        threshold_usd=750.0,
    )

    assert multipliers == [1.0, 0.0, 1.0]


def test_half_after_two_red_days_only_counts_traded_days() -> None:
    multipliers = compute_half_after_two_red_days_multipliers(
        daily_pnl_sequence=[-10.0, -5.0, 0.0, 12.0],
        traded_day_mask=[True, True, False, True],
    )

    assert multipliers == [1.0, 1.0, 0.5, 0.5]


def test_skip_after_three_red_days_only_activates_after_traded_streak() -> None:
    multipliers = compute_skip_after_three_red_days_multipliers(
        daily_pnl_sequence=[-1.0, -1.0, -1.0, 0.0, 2.0],
        traded_day_mask=[True, True, True, False, True],
    )

    assert multipliers == [1.0, 1.0, 1.0, 0.0, 1.0]
