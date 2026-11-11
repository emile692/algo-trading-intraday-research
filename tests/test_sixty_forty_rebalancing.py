from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.sixty_forty_rebalancing import StudyVariant, build_events, compute_drift


def test_monthly_drift_uses_prior_period_close_as_anchor() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-29"])
    frame = pd.DataFrame({
        "session_date": dates,
        "equity_index": [1.0, 1.01, 1.10],
        "bond_index": [1.0, 1.00, 1.00],
        "equity_signal_index": [1.0, 1.01, 1.10],
        "bond_signal_index": [1.0, 1.00, 1.00],
    })
    result = compute_drift(frame, "monthly")
    expected_weight = 0.60 * 1.10 / (0.60 * 1.10 + 0.40)
    assert np.isclose(result.loc[2, "equity_weight"], expected_weight)
    assert np.isclose(result.loc[2, "drift"], expected_weight - 0.60)


def _daily(symbol: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-29"])
    return pd.DataFrame({
        "session_date": dates,
        "rth_open": closes,
        "rth_open_instrument_id": 1,
        "signal_price": closes,
        "signal_price_instrument_id": 1,
        "entry_price": closes,
        "entry_price_instrument_id": 1,
        "close_price": closes,
        "close_price_instrument_id": 1,
        "symbol": symbol,
    })


def test_signal_and_future_return_are_separated_at_entry() -> None:
    equity = _daily("MES", [100.0, 101.0, 110.0])
    equity.loc[2, "entry_price"] = 111.0
    equity.loc[2, "close_price"] = 112.0
    bond = _daily("ZN", [100.0, 100.0, 100.0])
    events = build_events(equity, bond, StudyVariant("test", "MES", "ZN"))
    assert np.isclose(events.loc[2, "future_final_hour_return"], 112.0 / 111.0 - 1.0)
    # Drift uses the 14:59 signal price (110), not the entry or exit price.
    expected_weight = 0.60 * 1.10 / (0.60 * 1.10 + 0.40)
    assert np.isclose(events.loc[2, "equity_weight"], expected_weight)


def test_roll_transition_is_not_used_as_a_return() -> None:
    frame = pd.DataFrame({
        "session_date": pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"]),
        "equity_close_price": [100.0, 200.0, 202.0],
        "equity_signal_price": [100.0, 200.0, 202.0],
        "equity_rth_open": [100.0, 198.0, 201.0],
        "equity_close_price_instrument_id": [1, 2, 2],
        "equity_signal_price_instrument_id": [1, 2, 2],
        "equity_rth_open_instrument_id": [1, 2, 2],
    })
    from src.analytics.sixty_forty_rebalancing import add_roll_neutral_index

    result = add_roll_neutral_index(frame, "equity")
    assert result.loc[1, "equity_roll_transition"]
    assert np.isclose(result.loc[1, "equity_index"], 200.0 / 198.0)
    assert np.isclose(result.loc[2, "equity_index"], (200.0 / 198.0) * (202.0 / 200.0))
