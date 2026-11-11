from research.mbt_strategy_heatmaps_20261003.run_shortlist_wide_heatmaps import wide_grids


def test_shortlist_and_requested_edges_are_covered() -> None:
    grids = wide_grids()
    assert set(grids) == {
        "rolling_range_breakout",
        "session_range_breakout",
        "trend_pullback_ma_structure",
        "volatility_breakout",
        "momentum_confirmed_breakout",
    }
    assert 40 in grids["rolling_range_breakout"].x_values
    assert {45, 60, 75, 90}.issubset(grids["session_range_breakout"].x_values)
    assert min(grids["trend_pullback_ma_structure"].x_values) < 0.75
    assert {15.0, 17.5}.issubset(grids["momentum_confirmed_breakout"].y_values)
    for grid in grids.values():
        assert grid.nominal_x in grid.x_values
        assert grid.nominal_y in grid.y_values
