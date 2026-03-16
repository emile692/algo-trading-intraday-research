from src.config.paths import RAW_DATA_DIR
from src.data.loader import load_ohlcv_csv
from src.features.intraday import add_intraday_features
from src.features.opening_range import compute_opening_range
from src.engine.backtester import run_backtest
from src.engine.execution_model import ExecutionModel


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
