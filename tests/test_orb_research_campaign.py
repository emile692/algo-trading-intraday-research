import pandas as pd
import pytest

from src.engine.backtester import run_backtest
from src.engine.execution_model import ExecutionModel

from src.analytics.orb_research.exits import run_exit_variant_backtest
from src.analytics.orb_research.features import compute_noise_sigma
from src.analytics.orb_research.types import BaselineEntryConfig, ExitConfig


def _build_trade_df() -> pd.DataFrame:
    timestamp = pd.to_datetime(["2024-01-02 09:32:00", "2024-01-02 09:33:00"]).tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "session_date": timestamp.normalize().date,
            "signal": [1, 0],
            "or_high": [101.0, 101.0],
            "or_low": [99.0, 99.0],
            "open": [100.0, 100.0],
            "close": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 98.75],
            "volume": [100, 100],
            "session_vwap": [99.5, 99.6],
            "continuous_session_vwap": [99.5, 99.6],
        }
    )


def test_exit_variant_baseline_matches_engine_baseline() -> None:
    df = _build_trade_df()
    baseline = BaselineEntryConfig(
        stop_buffer_ticks=1,
        target_multiple=2.0,
        account_size_usd=50_000.0,
        risk_per_trade_pct=1.0,
        entry_on_next_open=False,
    )

    expected = run_backtest(
        df,
        execution_model=ExecutionModel(),
        stop_buffer_ticks=baseline.stop_buffer_ticks,
        target_multiple=baseline.target_multiple,
        account_size_usd=baseline.account_size_usd,
        risk_per_trade_pct=baseline.risk_per_trade_pct,
        entry_on_next_open=baseline.entry_on_next_open,
    )
    got = run_exit_variant_backtest(
        signal_df=df,
        execution_model=ExecutionModel(),
        baseline=baseline,
        exit_cfg=ExitConfig(mode="baseline"),
    )

    assert len(expected) == len(got)
    if not expected.empty:
        assert got.iloc[0]["net_pnl_usd"] == pytest.approx(expected.iloc[0]["net_pnl_usd"])
        assert got.iloc[0]["exit_reason"] == expected.iloc[0]["exit_reason"]


def test_noise_sigma_uses_only_past_sessions() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-01 09:30:00",
            "2024-01-02 09:30:00",
            "2024-01-03 09:30:00",
        ]
    ).tz_localize("America/New_York")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "minute_of_day": [570, 570, 570],
            "open_rth": [100.0, 100.0, 100.0],
            "close": [101.0, 103.0, 100.0],
        }
    )

    sigma = compute_noise_sigma(df, lookback=2)
    # Day-3 sigma should be mean(abs(1%), abs(3%)) = 2%
    assert sigma.iloc[2] == pytest.approx(0.02)
