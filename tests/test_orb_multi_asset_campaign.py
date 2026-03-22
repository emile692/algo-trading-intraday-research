from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analytics.orb_multi_asset_campaign import (
    BaselineSpec,
    _build_session_sanity,
    compute_campaign_metrics,
    resolve_aggregation_threshold,
    resolve_processed_dataset,
)


def test_resolve_aggregation_thresholds() -> None:
    assert resolve_aggregation_threshold("majority_50") == pytest.approx(0.50)
    assert resolve_aggregation_threshold("consensus_75") == pytest.approx(0.75)
    assert resolve_aggregation_threshold("unanimity_100") == pytest.approx(1.00)


def test_resolve_processed_dataset_returns_latest_match(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    older = root / "MES_c_0_1m_20260320_120000.parquet"
    newer = root / "MES_c_0_1m_20260322_120000.parquet"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")

    assert resolve_processed_dataset("MES", processed_dir=root) == newer


def test_resolve_processed_dataset_can_filter_by_timeframe(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    one_min = root / "MNQ_c_0_1m_20260322_120000.parquet"
    five_min = root / "MNQ_c_0_5m_20260322_120000.parquet"
    one_min.write_bytes(b"one")
    five_min.write_bytes(b"five")

    assert resolve_processed_dataset("MNQ", processed_dir=root, timeframe="5min") == five_min


def test_build_session_sanity_respects_bar_granularity() -> None:
    timestamp = pd.date_range("2024-01-02 09:30:00", "2024-01-02 16:00:00", freq="5min", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "session_date": timestamp.date,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "or_high": 101.0,
        }
    )

    sanity = _build_session_sanity(frame, BaselineSpec(or_minutes=30, opening_time="09:30:00", time_exit="16:00:00"))
    row = sanity.iloc[0]

    assert int(row["bar_minutes"]) == 5
    assert int(row["opening_window_bars"]) == 6
    assert bool(row["opening_window_complete"]) is True
    assert int(row["expected_opening_window_bars"]) == 6
    assert int(row["rth_expected_bars"]) == 79
    assert int(row["rth_missing_bars"]) == 0


def test_compute_campaign_metrics_uses_full_session_universe() -> None:
    trades = pd.DataFrame(
        [
            {
                "session_date": pd.Timestamp("2024-01-02").date(),
                "entry_time": pd.Timestamp("2024-01-02 09:45:00"),
                "exit_time": pd.Timestamp("2024-01-02 10:10:00"),
                "net_pnl_usd": 120.0,
                "exit_reason": "target",
            }
        ]
    )
    sessions = [pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-03").date()]

    metrics = compute_campaign_metrics(trades, sessions=sessions, initial_capital=50_000.0)

    assert metrics["nb_trades"] == 1
    assert metrics["net_pnl"] == pytest.approx(120.0)
    assert metrics["pct_days_traded"] == pytest.approx(0.5)
    assert "worst_5day_drawdown" in metrics
    assert "return_over_drawdown" in metrics
    assert "composite_score" in metrics
