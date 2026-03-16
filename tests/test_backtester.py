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
            "or_width": [1.0, 1.0],
            "close": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 99.0],
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
        account_size_usd=50_000,
        risk_per_trade_pct=1.0,
        stop_multiple=1.0,
        time_exit="09:35",
    )

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["quantity"] == 18
    assert trade["risk_budget_usd"] == pytest.approx(500.0)
    assert trade["risk_per_contract_usd"] == pytest.approx(27.5)
    assert trade["actual_risk_usd"] == pytest.approx(495.0)
    assert trade["net_pnl_usd"] == pytest.approx(-495.0)


def test_backtester_skips_trade_when_risk_budget_is_too_small() -> None:
    trades = run_backtest(
        _build_trade_df(),
        execution_model=ExecutionModel(),
        account_size_usd=100.0,
        risk_per_trade_pct=1.0,
        stop_multiple=1.0,
        time_exit="09:35",
    )

    assert trades.empty


def test_backtester_requires_complete_risk_parameters() -> None:
    with pytest.raises(ValueError, match="provided together"):
        run_backtest(
            _build_trade_df(),
            execution_model=ExecutionModel(),
            account_size_usd=50_000,
            time_exit="09:35",
        )
