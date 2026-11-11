from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.tsi_mean_reversion_campaign import add_indicators, detect_events, prepare_rth_bars, simulate_trades


def _bars(n: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2024-01-03 09:30", periods=n, freq="5min", tz="America/New_York")
    close = np.linspace(100, 102, n)
    return pd.DataFrame({"open": close, "high": close + .5, "low": close - .5, "close": close}, index=idx)


def test_prepare_rth_excludes_1600_start_bar() -> None:
    frame = _bars()
    result = prepare_rth_bars(frame)
    assert result.index[0].time().isoformat() == "09:30:00"
    assert result.index[-1].time().isoformat() == "15:55:00"
    assert len(result) == 78


def test_event_requires_extreme_then_cross_and_enters_next_open() -> None:
    frame = prepare_rth_bars(_bars(10))
    frame["tsi"] = [-10, -30, -28, -20, -10, -5, 0, 1, 2, 3]
    frame["tsi_signal"] = [-9, -20, -22, -21, -15, -10, -5, 0, 1, 2]
    frame["atr14"] = 1.0
    events = detect_events(frame, threshold=25, side=1)
    assert len(events) == 1
    assert events.iloc[0].signal_time == frame.index[3]
    assert events.iloc[0].entry_time == frame.index[4]


def test_stop_has_priority_and_costs_are_applied() -> None:
    frame = prepare_rth_bars(_bars(10))
    frame = add_indicators(frame, long_span=2, short_span=2, signal_span=2, atr_period=2)
    frame["tsi"] = 10.0
    frame.loc[frame.index[4], "low"] = 97.0
    event = pd.DataFrame([{"session_date": frame.iloc[3].session_date, "signal_time": frame.index[3],
                           "entry_time": frame.index[4], "signal_position": 3, "entry_position": 4,
                           "side": 1, "direction": "long", "threshold": 25.0, "tsi": -20.0,
                           "tsi_signal": -21.0, "atr14": 1.0}])
    trades = simulate_trades(frame, event, symbol="MNQ")
    assert trades.iloc[0].exit_reason == "stop"
    assert trades.iloc[0].raw_exit_price <= trades.iloc[0].stop_price
    assert trades.iloc[0].net_pnl_usd < trades.iloc[0].gross_pnl_usd
