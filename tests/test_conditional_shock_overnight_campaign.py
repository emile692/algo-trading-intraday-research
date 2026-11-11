from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics.conditional_shock_overnight_campaign import Instrument, next_open, session_frames, simulate_trade


def _bars(start: str, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1min", tz="America/New_York")
    price = pd.Series(range(periods), index=idx, dtype=float) + 100
    return pd.DataFrame({"open": price, "high": price + 1, "low": price - 1, "close": price + .5, "volume": 1, "instrument_id": 7}, index=idx)


def test_completed_bar_signal_enters_at_next_open() -> None:
    frame = _bars("2024-01-02 09:30", 30)
    signal_bar_start = pd.Timestamp("2024-01-02 09:30", tz="America/New_York")
    entry_ts, price = next_open(frame, signal_bar_start + pd.Timedelta(minutes=5))
    assert entry_ts == pd.Timestamp("2024-01-02 09:35", tz="America/New_York")
    assert price == frame.at[entry_ts, "open"]


def test_session_assignment_is_dst_aware_and_rth_bounded() -> None:
    frame = _bars("2024-03-11 09:25", 405)
    inst = Instrument("MNQ", Path("unused.parquet"), "09:30", "16:00", .25, 2.0)
    sessions, daily, five = session_frames(frame, inst)
    assert len(sessions) == 1
    assert daily.iloc[0].open_ts.hour == 9
    assert daily.iloc[0].open_ts.utcoffset().total_seconds() == -4 * 3600
    assert five.index.min().time() == pd.Timestamp("09:30").time()
    assert five.index.max().time() < pd.Timestamp("16:00").time()


def test_stop_precedes_target_when_both_touch_same_bar() -> None:
    frame = _bars("2024-01-02 09:30", 4)
    frame.loc[frame.index[1], ["high", "low"]] = [110.0, 90.0]
    inst = Instrument("MNQ", Path("unused.parquet"), "09:30", "16:00", .25, 2.0)
    trade = simulate_trade(frame, frame.index[0], 1, 95.0, 105.0, frame.index[-1], inst)
    assert trade is not None
    assert trade["exit_reason"] == "stop"


def test_delayed_execution_moves_one_full_minute_bar() -> None:
    frame = _bars("2024-01-02 09:30", 5)
    inst = Instrument("MNQ", Path("unused.parquet"), "09:30", "16:00", .25, 2.0)
    trade = simulate_trade(frame, frame.index[0], 1, None, None, frame.index[-1], inst, delay_bars=1)
    assert trade is not None
    assert trade["entry_ts"] == frame.index[1]
