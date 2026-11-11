from __future__ import annotations

import numpy as np
import pandas as pd

from research.mean_reversion_cross_asset_20261006.run_campaign import (
    EXIT_SPECS,
    INSTRUMENTS,
    _prepare_features,
    _signals,
    backtest,
)


def _bars() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 09:30", periods=40, freq="5min", tz="America/New_York")
    close = np.r_[np.full(18, 100.0), 96.0, 98.0, np.full(20, 100.0)]
    x = pd.DataFrame({"open": close, "high": close + .25, "low": close - .25, "close": close, "volume": 100.0}, index=idx)
    x["session_date"] = x.index.date
    return _prepare_features(x)


def test_signal_is_filled_no_earlier_than_next_bar_open() -> None:
    x = _bars()
    signal, fair = _signals(x, "vwap_extension", {"lookback": 0, "threshold": .75})
    trades = backtest(x, signal, fair, INSTRUMENTS["MNQ"], EXIT_SPECS[2])
    assert not trades.empty
    assert pd.to_datetime(trades.iloc[0].entry_time) > pd.to_datetime(trades.iloc[0].signal_time)


def test_opposite_direction_placebo_flips_trade_direction() -> None:
    x = _bars()
    signal, fair = _signals(x, "vwap_extension", {"lookback": 0, "threshold": .75})
    normal = backtest(x, signal, fair, INSTRUMENTS["MNQ"], EXIT_SPECS[2])
    opposite = backtest(x, signal, fair, INSTRUMENTS["MNQ"], EXIT_SPECS[2], flip=True)
    assert int(normal.iloc[0].direction) == -int(opposite.iloc[0].direction)
