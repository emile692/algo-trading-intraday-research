from __future__ import annotations

import pandas as pd

from src.analytics.weekly_open_reversion_confirmation import _simulate_path, deduplicate_underlying_weeks


def _window() -> pd.DataFrame:
    index = pd.date_range("2024-01-09 09:30", periods=3, freq="1min", tz="America/New_York")
    return pd.DataFrame({"open": [99.0] * 3, "high": [99.5, 100.0, 100.0], "low": [98.5, 98.0, 99.0], "close": [99.0, 99.0, 100.0]}, index=index)


def test_same_bar_stop_precedes_target_and_costs_are_applied() -> None:
    result = _simulate_path(_window(), raw_entry=99.0, target=100.0, side=1, distance=1.0, stop_multiple=1.0, symbol="MNQ")
    assert result["exit_reason"] == "stop"
    assert result["same_bar_target_stop"]
    assert result["gross_pnl_usd"] == -2.0
    assert result["net_pnl_usd"] == -5.5


def test_target_exit_includes_round_trip_slippage_and_commission() -> None:
    window = _window().copy()
    window.loc[window.index[1], "low"] = 98.5
    result = _simulate_path(window, raw_entry=99.0, target=100.0, side=1, distance=1.0, stop_multiple=1.0, symbol="MNQ")
    assert result["exit_reason"] == "target"
    assert result["gross_pnl_usd"] == 2.0
    assert result["net_pnl_usd"] == -1.5


def test_underlying_deduplication_prefers_nq() -> None:
    rows = []
    for symbol in ("MNQ", "NQ"):
        rows.append({"symbol": symbol, "week_monday": "2024-01-08", "anchor": "weekly_open", "horizon": "tue_1130", "stop_multiple": 1.0})
    result = deduplicate_underlying_weeks(pd.DataFrame(rows))
    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "NQ"
