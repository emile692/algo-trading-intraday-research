from __future__ import annotations

import pandas as pd

from research.topstep_portfolio_risk_governor.run_topstep_portfolio_risk_governor import (
    SLEEVES,
    apply_governor,
)


def _row(sleeve: str, trade_id: int, entry: str, exit_: str, signed_quantity: int) -> dict:
    return {
        "sleeve": sleeve,
        "instrument": "MNQ" if sleeve.startswith("mnq_") else "MGC",
        "trade_id": trade_id,
        "session_date": pd.Timestamp("2025-01-02"),
        "entry_ts_utc": pd.Timestamp(entry, tz="UTC"),
        "exit_ts_utc": pd.Timestamp(exit_, tz="UTC"),
        "quantity": abs(signed_quantity),
        "signed_quantity": signed_quantity,
        "net_pnl_usd": 10.0,
        "explicit_cost_usd": 2.0,
        "signal_id": f"{sleeve}:{trade_id}",
        "sleeve_order": SLEEVES.index(sleeve),
    }


def _risk() -> dict[str, float]:
    return {sleeve: float(idx + 1) for idx, sleeve in enumerate(SLEEVES)}


def test_governor_counts_actual_contracts_and_rejects_whole_trade() -> None:
    tape = pd.DataFrame([
        _row(SLEEVES[0], 1, "2025-01-02 14:30", "2025-01-02 15:30", 2),
        _row(SLEEVES[1], 2, "2025-01-02 14:31", "2025-01-02 15:00", 1),
    ])
    decisions, exposure = apply_governor(
        tape,
        {"config_id": "x", "handler": "first_signal_first_served", "global_cap": 2, "mnq_cap": 2, "mgc_cap": 1},
        _risk(),
    )
    assert decisions["accepted"].tolist() == [True, False]
    assert exposure["max_simultaneous_total_contracts"] == 2


def test_exits_are_processed_before_same_timestamp_entries() -> None:
    tape = pd.DataFrame([
        _row(SLEEVES[0], 1, "2025-01-02 14:30", "2025-01-02 15:00", 1),
        _row(SLEEVES[1], 2, "2025-01-02 15:00", "2025-01-02 15:30", 1),
    ])
    decisions, _ = apply_governor(
        tape,
        {"config_id": "x", "handler": "first_signal_first_served", "global_cap": 1, "mnq_cap": 1, "mgc_cap": 1},
        _risk(),
    )
    assert decisions["accepted"].tolist() == [True, True]


def test_lowest_frozen_risk_breaks_same_timestamp_collision_without_pnl() -> None:
    tape = pd.DataFrame([
        _row(SLEEVES[4], 1, "2025-01-02 14:30", "2025-01-02 15:00", 1),
        _row(SLEEVES[5], 2, "2025-01-02 14:30", "2025-01-02 15:00", 1),
    ])
    risk = _risk()
    risk[SLEEVES[4]], risk[SLEEVES[5]] = 5.0, 1.0
    decisions, _ = apply_governor(
        tape,
        {"config_id": "x", "handler": "lowest_frozen_standalone_risk", "global_cap": 1, "mnq_cap": 1, "mgc_cap": 1},
        risk,
    )
    accepted = decisions.loc[decisions["accepted"], "sleeve"].tolist()
    assert accepted == [SLEEVES[5]]
