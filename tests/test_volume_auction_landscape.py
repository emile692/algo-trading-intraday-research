from __future__ import annotations

import numpy as np
import pandas as pd

from research.volume_auction_landscape_20261007.run_campaign import (
    INSTRUMENTS,
    add_causal_volume_baselines,
    add_price_features,
    backtest,
)


def _bars(days: int = 25, bars_per_day: int = 3) -> pd.DataFrame:
    rows = []
    for d, date in enumerate(pd.bdate_range("2025-01-02", periods=days)):
        for b in range(bars_per_day):
            ts = (date + pd.Timedelta(hours=9, minutes=30 + 5 * b)).tz_localize("America/New_York")
            price = 100 + d + b / 10
            rows.append((ts, price, price + 1, price - 1, price + .25, 10 * (d + 1) + b, date, 570 + 5 * b, b))
    x = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "session_date", "slot", "bar_no"]).set_index("timestamp")
    x["last_bar"] = x.bar_no.eq(bars_per_day - 1)
    return x


def test_same_time_volume_baseline_is_past_only() -> None:
    x = _bars()
    out = add_causal_volume_baselines(x, windows=(10,))
    target = out[(out.session_date == out.session_date.drop_duplicates().iloc[10]) & (out.slot == 570)].iloc[0]
    expected = np.median([10 * (d + 1) for d in range(10)])
    assert target.expected_median_10 == expected

    changed = x.copy()
    changed.loc[changed.session_date > target.session_date, "volume"] = 10_000_000
    changed_out = add_causal_volume_baselines(changed, windows=(10,))
    changed_target = changed_out.loc[target.name]
    assert changed_target.expected_median_10 == target.expected_median_10


def test_signal_close_fills_next_bar_open_without_crossing_session() -> None:
    x = add_price_features(add_causal_volume_baselines(_bars(), windows=(10, 20, 40)))
    sig = pd.Series(0, index=x.index, dtype=int)
    eligible = x.index[(x.bar_no == 0) & x.rvol_20.notna()]
    sig.loc[eligible] = 1
    # A signal on a session's last bar must not fill the following session.
    last = x.index[(x.last_bar) & x.rvol_20.notna()][0]
    sig.loc[last] = 1
    trades = backtest(x, sig, INSTRUMENTS["MNQ"], hold=1)
    assert (trades.entry_time > trades.signal_time).all()
    assert (pd.to_datetime(trades.entry_time).dt.date == pd.to_datetime(trades.signal_time).dt.date).all()
    assert not (trades.signal_time == last).any()
    for row in trades.itertuples(index=False):
        assert row.entry == x.loc[row.entry_time, "open"]
