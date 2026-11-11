from __future__ import annotations

import json

import numpy as np

from research.topstep_account_size_analysis.run_topstep_account_size_analysis import (
    CONFIG_PATH,
    mapping_grid,
    simulate_path_diagnostics,
)


def _rules(target: float = 3000.0, mll: float = 2000.0, horizon: int = 5) -> dict:
    return {
        "starting_balance_usd": 50000.0,
        "profit_target_usd": target,
        "trailing_mll_usd": mll,
        "consistency_share_limit": 0.5,
        "lock_at_starting_balance": False,
        "max_traded_days": horizon,
    }


def test_diagnostics_records_pass_mll_and_distance() -> None:
    passed = simulate_path_diagnostics(np.array([600, 700, 800, 900, 1000], float), np.ones(5, bool), _rules())
    assert passed["outcome"] == "PASS"
    assert passed["days_to_pass"] == 4
    failed = simulate_path_diagnostics(np.array([-2100, 0], float), np.ones(2, bool), _rules())
    assert failed["outcome"] == "MLL_FAIL"
    assert failed["days_to_mll"] == 1
    assert failed["minimum_distance_to_mll_usd"] <= 0


def test_local_integer_grid_is_predeclared_22_mapping_neighborhood() -> None:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    grid = mapping_grid(cfg)
    assert len(grid) == 22
    assert "q_111111" in set(grid["mapping_id"])
    assert grid["total_sleeve_copies"].min() == 6
    assert grid["total_sleeve_copies"].max() == 8
    copy_columns = [column for column in grid if column.startswith("copies__")]
    assert set(np.unique(grid[copy_columns].to_numpy())) == {1, 2}
