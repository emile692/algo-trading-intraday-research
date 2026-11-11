from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.cross_asset_lead_lag_campaign import _event_pnl, causal_features, synchronize_pair


def _sync(rows: int = 140) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 09:30", periods=rows, freq="5min", tz="America/New_York")
    leader = 100 * np.exp(np.cumsum(np.linspace(-.001, .001, rows)))
    target = 200 * np.exp(np.cumsum(np.linspace(-.0005, .0007, rows)))
    return pd.DataFrame({"leader_open": leader, "leader_close": leader, "target_open": target, "target_close": target, "session_date": pd.Timestamp("2024-01-02")}, index=idx)


def test_synchronization_is_exact_inner_join() -> None:
    idx = pd.date_range("2024-01-02 09:30", periods=4, freq="5min", tz="America/New_York")
    leader = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "session_date": pd.Timestamp("2024-01-02")}, index=idx.delete(1))
    target = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "session_date": pd.Timestamp("2024-01-02")}, index=idx.delete(2))
    result = synchronize_pair(leader, target)
    assert result.index.equals(idx[[0, 3]])


def test_beta_and_lagged_target_exclude_current_bar() -> None:
    base = _sync()
    changed = base.copy()
    changed.iloc[-1, changed.columns.get_loc("target_close")] *= 1.5
    a = causal_features(base, beta_window=60, impulse_bars=3).iloc[-1]
    b = causal_features(changed, beta_window=60, impulse_bars=3).iloc[-1]
    assert a.beta == b.beta
    assert a.target_lagged_response == b.target_lagged_response
    assert a.residual == b.residual


def test_current_leader_changes_impulse_but_not_beta() -> None:
    base = _sync(); changed = base.copy(); changed.iloc[-1, changed.columns.get_loc("leader_close")] *= 1.1
    a = causal_features(base, 60, 3).iloc[-1]; b = causal_features(changed, 60, 3).iloc[-1]
    assert a.beta == b.beta
    assert a.leader_impulse != b.leader_impulse


def test_event_execution_is_next_bar_and_extra_delay_is_one_more_bar() -> None:
    frame = _sync(20)
    nominal = _event_pnl(frame, 5, 1, 15, .25, 2.0)
    delayed = _event_pnl(frame, 5, 1, 15, .25, 2.0, delay_bars=1)
    assert nominal is not None and delayed is not None
    assert nominal["entry_ts"] == frame.index[6]
    assert delayed["entry_ts"] == frame.index[7]


def test_event_cannot_cross_session_boundary() -> None:
    frame = _sync(20)
    frame.loc[frame.index[8]:, "session_date"] = pd.Timestamp("2024-01-03")
    assert _event_pnl(frame, 5, 1, 15, .25, 2.0) is None


def test_impulses_and_returns_never_cross_session_boundary() -> None:
    frame = _sync(140)
    boundary = 100
    frame.loc[frame.index[boundary]:, "session_date"] = pd.Timestamp("2024-01-03")
    result = causal_features(frame, beta_window=60, impulse_bars=3)
    assert pd.isna(result.leader_impulse.iloc[boundary])
    assert pd.isna(result.target_lagged_response.iloc[boundary + 2])
    assert pd.notna(result.beta.iloc[boundary])
