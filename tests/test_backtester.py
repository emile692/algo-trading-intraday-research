import pandas as pd
import pytest

from src.config.paths import RAW_DATA_DIR
from src.data.loader import load_ohlcv_csv
from src.features.intraday import add_intraday_features
from src.features.opening_range import compute_opening_range
from src.engine.backtester import run_backtest
from src.engine.execution_model import ExecutionModel


def _build_trade_df() -> pd.DataFrame:
    timestamp = pd.to_datetime(["2024-01-02 09:32:00", "2024-01-02 09:33:00"])
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "session_date": timestamp.normalize().date,
            "signal": [1, 0],
            "or_high": [101.0, 101.0],
            "or_low": [99.0, 99.0],
            "or_width": [2.0, 2.0],
            "open": [100.0, 100.0],
            "close": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 98.75],
        }
    )


def test_backtester_runs() -> None:
    df = load_ohlcv_csv(RAW_DATA_DIR / "NQ_1min_sample.csv")
    df = add_intraday_features(df)
    df = compute_opening_range(df, or_minutes=2)
    df["signal"] = 0
    df.loc[df.index[2], "signal"] = 1

    trades = run_backtest(df, execution_model=ExecutionModel(), time_exit="09:35")
    assert isinstance(trades, type(df))
    if not trades.empty:
        assert "net_pnl_usd" in trades.columns


def test_backtester_sizes_position_from_account_risk() -> None:
    trades = run_backtest(
        _build_trade_df(),
        execution_model=ExecutionModel(),
        tick_value_usd=5.0,
        account_size_usd=50_000,
        risk_per_trade_pct=1.0,
        stop_buffer_ticks=1,
        time_exit="09:35",
    )

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["stop_price"] == pytest.approx(98.75)
    assert trade["target_price"] == pytest.approx(102.5)
    assert trade["quantity"] == 13
    assert trade["risk_budget_usd"] == pytest.approx(500.0)
    assert trade["risk_per_contract_usd"] == pytest.approx(37.5)
    assert trade["actual_risk_usd"] == pytest.approx(487.5)
    assert trade["net_pnl_usd"] == pytest.approx(-487.5)


def test_backtester_skips_trade_when_risk_budget_is_too_small() -> None:
    trades = run_backtest(
        _build_trade_df(),
        execution_model=ExecutionModel(),
        tick_value_usd=5.0,
        account_size_usd=100.0,
        risk_per_trade_pct=1.0,
        stop_buffer_ticks=1,
        time_exit="09:35",
    )

    assert trades.empty


def test_backtester_targets_a_multiple_of_entry_to_stop_risk() -> None:
    df = _build_trade_df()
    df.loc[df.index[1], "high"] = 103.25
    df.loc[df.index[1], "low"] = 100.0

    trades = run_backtest(
        df,
        execution_model=ExecutionModel(),
        tick_value_usd=5.0,
        stop_buffer_ticks=1,
        target_multiple=2.0,
        time_exit="09:35",
    )

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["stop_price"] == pytest.approx(98.75)
    assert trade["target_price"] == pytest.approx(103.25)
    assert trade["exit_reason"] == "target"
    assert trade["net_pnl_usd"] == pytest.approx(52.5)


def test_backtester_requires_complete_risk_parameters() -> None:
    with pytest.raises(ValueError, match="provided together"):
        run_backtest(
            _build_trade_df(),
            execution_model=ExecutionModel(),
            account_size_usd=50_000,
            time_exit="09:35",
        )


def test_backtester_applies_leverage_cap_after_risk_sizing() -> None:
    trades = run_backtest(
        _build_trade_df(),
        execution_model=ExecutionModel(),
        tick_value_usd=5.0,
        point_value_usd=20.0,
        account_size_usd=10_000,
        risk_per_trade_pct=50.0,
        max_leverage=1.0,
        stop_buffer_ticks=1,
        time_exit="09:35",
    )

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["quantity"] == 4
    assert trade["notional_usd"] == pytest.approx(4 * 100.25 * 20.0)
    assert trade["leverage_used"] == pytest.approx((4 * 100.25 * 20.0) / 10_000.0)
