from __future__ import annotations

import pandas as pd

from src.analytics.topstep_optimization.topstep_simulator import (
    ScaledVariantSeries,
    TopstepRules,
    build_variant_name,
    run_block_bootstrap_simulations,
    scale_daily_results,
    simulate_account_path,
)


def _daily_frame(pnls: list[float], trade_counts: list[int] | None = None) -> pd.DataFrame:
    counts = trade_counts or [1] * len(pnls)
    return pd.DataFrame(
        {
            "session_date": pd.date_range("2024-01-01", periods=len(pnls), freq="D"),
            "daily_pnl_usd": pnls,
            "daily_gross_pnl_usd": pnls,
            "daily_fees_usd": [0.0] * len(pnls),
            "daily_trade_count": counts,
            "daily_loss_count": [1 if pnl < 0 else 0 for pnl in pnls],
        }
    )


def test_build_variant_name_keeps_base_name_for_unit_leverage() -> None:
    assert build_variant_name("nominal", 1.0) == "nominal"
    assert build_variant_name("nominal", 1.2) == "nominal_x1.2"


def test_scale_daily_results_multiplies_daily_pnl_columns() -> None:
    frame = _daily_frame([100.0, -50.0])
    scaled = scale_daily_results(frame, 1.5)

    assert list(scaled["daily_pnl_usd"]) == [150.0, -75.0]
    assert list(scaled["daily_gross_pnl_usd"]) == [150.0, -75.0]


def test_simulate_account_path_passes_on_profit_target() -> None:
    frame = _daily_frame([800.0, 900.0, 700.0, 700.0])
    _, result = simulate_account_path(frame, rules=TopstepRules())

    assert result["pass"] is True
    assert result["fail"] is False
    assert result["days_to_pass"] == 4.0


def test_simulate_account_path_fails_on_daily_loss_limit_before_trailing() -> None:
    frame = _daily_frame([500.0, -1001.0, 2500.0])
    _, result = simulate_account_path(frame, rules=TopstepRules())

    assert result["fail"] is True
    assert result["failure_reason"] == "daily_loss_limit"
    assert result["days_to_fail"] == 2.0


def test_simulate_account_path_fails_on_trailing_drawdown() -> None:
    frame = _daily_frame([1500.0, -900.0, -900.0, -400.0])
    _, result = simulate_account_path(frame, rules=TopstepRules())

    assert result["fail"] is True
    assert result["failure_reason"] == "trailing_drawdown"


def test_block_bootstrap_is_reproducible_with_fixed_seed() -> None:
    series = ScaledVariantSeries(
        variant="nominal",
        base_variant="nominal",
        source_variant_name="nominal",
        leverage_factor=1.0,
        daily_results=_daily_frame([300.0, -200.0, 250.0, 100.0, -150.0, 400.0, 350.0]),
    )
    rules = TopstepRules(max_trading_days=5)

    first = run_block_bootstrap_simulations(series, rules=rules, bootstrap_paths=5, block_size=3, random_seed=42)
    second = run_block_bootstrap_simulations(series, rules=rules, bootstrap_paths=5, block_size=3, random_seed=42)

    pd.testing.assert_frame_equal(first, second)
