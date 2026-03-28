from __future__ import annotations

import pandas as pd
import pytest

from src.config.mean_reversion_campaign import MeanReversionVariantConfig
from src.engine.mean_reversion_backtester import run_mean_reversion_backtest
from src.engine.vwap_backtester import build_execution_model_for_profile


def test_mean_reversion_backtester_hits_target() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:35:00",
            "2024-01-02 09:40:00",
            "2024-01-02 09:45:00",
        ]
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "open": [100.0, 100.2, 100.8],
            "high": [100.4, 101.7, 101.0],
            "low": [99.5, 100.0, 100.6],
            "close": [100.2, 101.0, 100.9],
            "entry_long": [True, False, False],
            "entry_short": [False, False, False],
            "stop_reference_long": [99.0, pd.NA, pd.NA],
            "stop_reference_short": [pd.NA, pd.NA, pd.NA],
            "target_reference_long": [101.5, pd.NA, pd.NA],
            "target_reference_short": [pd.NA, pd.NA, pd.NA],
            "is_last_bar_of_session": [False, False, True],
        }
    )
    variant = MeanReversionVariantConfig(
        name="mes_5m_target_test",
        family="vwap_extension_reversion",
        symbol="MES",
        timeframe="5m",
        fixed_quantity=1,
        max_trades_per_day=1,
        timeout_bars=5,
    )
    execution_model, instrument = build_execution_model_for_profile("MES", "repo_realistic")

    result = run_mean_reversion_backtest(
        signal_df=frame,
        variant=variant,
        execution_model=execution_model,
        instrument=instrument,
        account_size_usd=50_000.0,
    )

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "target"
    assert float(result.trades.iloc[0]["net_pnl_usd"]) > 0.0


def test_mean_reversion_backtester_keeps_partial_exit_metadata() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:35:00",
            "2024-01-02 09:40:00",
            "2024-01-02 09:45:00",
        ]
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "open": [100.0, 100.4, 101.0],
            "high": [100.4, 101.2, 102.2],
            "low": [99.6, 100.2, 100.8],
            "close": [100.2, 100.8, 102.0],
            "entry_long": [True, False, False],
            "entry_short": [False, False, False],
            "stop_reference_long": [99.0, pd.NA, pd.NA],
            "stop_reference_short": [pd.NA, pd.NA, pd.NA],
            "target_reference_long": [102.0, pd.NA, pd.NA],
            "target_reference_short": [pd.NA, pd.NA, pd.NA],
            "is_last_bar_of_session": [False, False, True],
        }
    )
    variant = MeanReversionVariantConfig(
        name="mes_5m_partial_test",
        family="vwap_extension_reversion",
        symbol="MES",
        timeframe="5m",
        fixed_quantity=2,
        max_trades_per_day=1,
        timeout_bars=5,
        use_partial_exit=True,
        partial_target_fraction=0.5,
    )
    execution_model, instrument = build_execution_model_for_profile("MES", "repo_realistic")

    result = run_mean_reversion_backtest(
        signal_df=frame,
        variant=variant,
        execution_model=execution_model,
        instrument=instrument,
        account_size_usd=50_000.0,
    )

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "target_after_partial"
    assert bool(trade["partial_exit_taken"]) is True
    assert int(trade["partial_exit_quantity"]) == 1
    assert float(trade["net_pnl_usd"]) == pytest.approx(float(trade["pnl_usd"]) - float(trade["fees"]))
