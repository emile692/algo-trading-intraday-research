from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.fast_alpha_execution_overlay import construct_session_bars, newey_west_mean


SPEC = {"timezone": "America/New_York", "session_start": "09:30", "session_end": "10:00"}


def test_five_minute_bar_is_session_anchored_and_observable_at_end() -> None:
    index = pd.date_range("2024-03-11 09:30", "2024-03-11 09:39", freq="min", tz="America/New_York").tz_convert("UTC")
    source = pd.DataFrame({"open": np.arange(10) + 100.0, "high": np.arange(10) + 101.0, "low": np.arange(10) + 99.0, "close": np.arange(10) + 100.5, "volume": 1, "instrument_id": 7}, index=index)
    bars = construct_session_bars(source, SPEC, 5)
    assert len(bars) == 2
    assert bars.loc[0, "open"] == 100.0
    assert bars.loc[0, "close"] == 104.5
    assert bars.loc[0, "bar_end_local"] == pd.Timestamp("2024-03-11 09:35", tz="America/New_York")


def test_runs_reset_on_missing_interval_and_session_boundary() -> None:
    index = pd.DatetimeIndex([
        pd.Timestamp("2024-01-02 09:30", tz="America/New_York"),
        pd.Timestamp("2024-01-02 09:35", tz="America/New_York"),
        pd.Timestamp("2024-01-02 09:45", tz="America/New_York"),
        pd.Timestamp("2024-01-03 09:30", tz="America/New_York"),
    ]).tz_convert("UTC")
    source = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 101.0, "volume": 1, "instrument_id": 7}, index=index)
    bars = construct_session_bars(source, SPEC, 5)
    assert bars.run_length.tolist() == [1, 2, 1, 1]


def test_next_return_never_crosses_session_boundary() -> None:
    index = pd.DatetimeIndex([pd.Timestamp("2024-01-02 09:55", tz="America/New_York"), pd.Timestamp("2024-01-03 09:30", tz="America/New_York")]).tz_convert("UTC")
    source = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 101.0, "volume": 1, "instrument_id": 7}, index=index)
    bars = construct_session_bars(source, SPEC, 5)
    assert bars.next_return.isna().all()


def test_newey_west_mean_detects_positive_constant_edge() -> None:
    values = pd.Series(0.001 + np.sin(np.arange(500)) * 0.0001)
    standard_error, t_stat, lags = newey_west_mean(values)
    assert standard_error > 0
    assert t_stat > 10
    assert lags > 0


def test_partial_bar_at_session_end_is_excluded() -> None:
    spec = {"timezone": "America/New_York", "session_start": "09:30", "session_end": "09:40"}
    index = pd.date_range("2024-01-02 09:30", "2024-01-02 09:39", freq="min", tz="America/New_York").tz_convert("UTC")
    source = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 101.0, "volume": 1, "instrument_id": 7}, index=index)
    bars = construct_session_bars(source, spec, 15)
    assert bars.empty
