from __future__ import annotations

import numpy as np
import pandas as pd

from research.topstep_portfolio_scaling.run_topstep_portfolio_scaling import (
    integer_apportion,
    round_half_up,
    simulate_path,
    wilson,
)
from src.analytics.mnq_orb_topstep_50k_simulation import TopstepRuleset, simulate_topstep_path


def test_integer_apportion_is_deterministic_long_only_and_exact_total() -> None:
    weights = np.array([0.15, 0.08, 0.36, 0.08, 0.18, 0.15])
    result = integer_apportion(weights, total=12, minimum=1)
    assert result.tolist() == integer_apportion(weights, total=12, minimum=1).tolist()
    assert int(result.sum()) == 12
    assert np.all(result >= 1)


def test_round_half_up_contract_budget() -> None:
    assert round_half_up(1.5) == 2
    assert round_half_up(2.5) == 3


def test_vector_simulator_matches_existing_topstep_terminal_logic() -> None:
    ruleset = TopstepRuleset(
        name="test", description="test", starting_balance_usd=50_000.0,
        profit_target_usd=3_000.0, trailing_mll_usd=2_000.0,
        consistency_share_limit=0.50, max_traded_days=5, lock_at_starting_balance=False,
    )
    pnl = np.array([600.0, 700.0, 800.0, 900.0, 1_000.0])
    traded = np.ones(len(pnl), dtype=bool)
    ours = simulate_path(pnl, traded, ruleset.__dict__)
    frame = pd.DataFrame(
        {"session_date": pd.date_range("2025-01-01", periods=len(pnl)),
         "daily_pnl_usd": pnl, "daily_trade_count": 1}
    )
    _, existing = simulate_topstep_path(frame, ruleset)
    expected = "PASS" if existing["pass"] else ("MLL_FAIL" if existing["fail"] else "EXPIRY")
    assert ours["outcome"] == expected
    assert ours["terminal_pnl_usd"] == existing["final_pnl_usd"]


def test_wilson_interval_contains_observed_probability() -> None:
    low, high = wilson(40, 100)
    assert low < 0.4 < high
