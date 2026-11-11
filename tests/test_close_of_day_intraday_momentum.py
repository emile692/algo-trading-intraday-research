from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.close_of_day_intraday_momentum import build_canonical_events, executable_results, newey_west_regression


def _two_sessions() -> pd.DataFrame:
    parts = []
    for date in ("2024-03-08", "2024-03-11"):
        index = pd.date_range(f"{date} 09:30", f"{date} 15:59", freq="min", tz="America/New_York")
        parts.append(pd.DataFrame({"open": 100.0, "close": 100.0, "instrument_id": 7}, index=index))
    return pd.concat(parts).tz_convert("UTC")


def test_signal_is_observable_before_entry_and_dst_safe() -> None:
    bars = _two_sessions()
    signal = pd.Timestamp("2024-03-11 15:29", tz="America/New_York").tz_convert("UTC")
    entry = pd.Timestamp("2024-03-11 15:30", tz="America/New_York").tz_convert("UTC")
    exit_ = pd.Timestamp("2024-03-11 15:59", tz="America/New_York").tz_convert("UTC")
    bars.loc[signal, "close"] = 102.0
    bars.loc[entry, "open"] = 103.0
    bars.loc[exit_, "close"] = 104.0
    events, diagnostics = build_canonical_events(bars, instrument="MNQ", timezone="America/New_York", session_start="09:30", close_time="16:00")
    assert len(events) == 1
    assert diagnostics["signal_bar_label"] == "15:29"
    assert np.isclose(events.iloc[0]["r_signal"], 0.02)
    assert np.isclose(events.iloc[0]["r_close_raw"], 104 / 102 - 1)
    assert np.isclose(events.iloc[0]["r_close_executable"], 104 / 103 - 1)


def test_roll_transition_is_excluded() -> None:
    bars = _two_sessions()
    second = bars.index.tz_convert("America/New_York").date == pd.Timestamp("2024-03-11").date()
    bars.loc[second, "instrument_id"] = 8
    events, diagnostics = build_canonical_events(bars, instrument="MNQ", timezone="America/New_York", session_start="09:30", close_time="16:00")
    assert events.empty
    assert diagnostics["roll_exclusions"] == 1


def test_hac_regression_recovers_positive_slope() -> None:
    x = pd.Series(np.linspace(-0.02, 0.02, 200))
    y = 0.001 + 0.4 * x
    result = newey_west_regression(y, x)
    assert np.isclose(result["beta"], 0.4)
    assert result["hac_t_stat"] > 10


def test_missing_standard_close_does_not_bridge_previous_close() -> None:
    bars = _two_sessions()
    third_index = pd.date_range("2024-03-12 09:30", "2024-03-12 15:59", freq="min", tz="America/New_York")
    third = pd.DataFrame({"open": 100.0, "close": 100.0, "instrument_id": 7}, index=third_index).tz_convert("UTC")
    bars = pd.concat([bars, third])
    missing_close = pd.Timestamp("2024-03-11 15:59", tz="America/New_York").tz_convert("UTC")
    bars = bars.drop(index=missing_close)
    events, diagnostics = build_canonical_events(bars, instrument="MNQ", timezone="America/New_York", session_start="09:30", close_time="16:00")
    assert events.empty
    assert diagnostics["missing_target_sessions"] == 1


def test_execution_cost_is_round_trip_commission_plus_slippage() -> None:
    events = pd.DataFrame(
        {
            "instrument": ["MNQ"], "session_date": [pd.Timestamp("2024-01-02")], "position": [1.0],
            "entry_price_open_Tminus30": [100.0], "entry_price_open_Tminus29": [100.25], "exit_price_close_Tminus1": [101.0],
            "sign_edge_executable": [0.01], "sign_edge_one_bar_delay": [101 / 100.25 - 1],
        }
    )
    config = {"instruments": [{"instrument": "MNQ", "point_value": 2.0, "tick_size": 0.25, "commission_per_side": 1.25, "slippage_ticks_per_side": 1.0}]}
    detail, _ = executable_results(events, config)
    assert detail.loc[0, "gross_pnl"] == 2.0
    assert detail.loc[0, "normal_round_trip_cost"] == 3.5
    assert detail.loc[0, "normal_net_pnl"] == -1.5
