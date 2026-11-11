from __future__ import annotations

from research.mbt_strategy_heatmaps_20261003.run_mbt_heatmaps import grid_definitions, mbt_config


def test_all_16_grids_are_three_by_three_and_include_nominal_cell() -> None:
    grids = grid_definitions()
    assert len(grids) == 16
    for grid in grids.values():
        assert len(grid.x_values) == 3
        assert len(grid.y_values) == 3
        assert grid.nominal_x in grid.x_values
        assert grid.nominal_y in grid.y_values


def test_mbt_contract_economics() -> None:
    instrument = mbt_config()["instrument"]
    assert instrument["symbol"] == "MBT"
    assert instrument["tick_size"] == 5.0
    assert instrument["tick_value_usd"] == 0.5
    assert instrument["point_value_usd"] == 0.1
    assert 2 * instrument["commission_per_side_usd"] == 2.82
