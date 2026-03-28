import pandas as pd
import pytest

from src.config.vwap_campaign import TimeWindow, VWAPVariantConfig
from src.engine.vwap_backtester import build_execution_model_for_profile, run_vwap_backtest


def _base_target_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-03 09:30:00",
            "2024-01-03 09:31:00",
        ]
    )
    session_dates = timestamps.normalize().date
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": session_dates,
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 101.0, 100.0, 101.0],
            "low": [100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 101.0, 100.0, 101.0],
            "volume": [10, 10, 10, 10],
            "session_vwap": [100.0, 100.5, 100.0, 100.5],
            "trade_allowed": [True, True, True, True],
            "target_position": [0, 1, 0, 1],
            "raw_signal": [0, 1, 0, 1],
            "is_last_bar_of_session": [False, True, False, True],
        }
    )


def test_target_backtest_flats_overnight() -> None:
    variant = VWAPVariantConfig(
        name="paper_vwap_baseline",
        family="baseline",
        mode="target_position",
        execution_profile="repo_realistic",
        initial_capital_usd=50_000.0,
        quantity_mode="fixed_quantity",
        fixed_quantity=1,
    )
    execution_model, instrument = build_execution_model_for_profile("MNQ", "repo_realistic")

    result = run_vwap_backtest(_base_target_frame(), variant, execution_model, instrument)

    assert len(result.trades) == 2
    assert set(result.trades["exit_reason"]) == {"session_close"}
    assert result.trades["session_date"].nunique() == 2
    assert (pd.to_datetime(result.trades["entry_time"]).dt.date == pd.to_datetime(result.trades["exit_time"]).dt.date).all()


def test_time_filter_defers_mid_session_flip_until_window_reopens() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
            "2024-01-02 09:33:00",
            "2024-01-02 09:34:00",
        ]
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 101.0, 100.0, 100.0, 100.0],
            "low": [100.0, 99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 101.0, 99.0, 99.0, 99.0],
            "volume": [10, 10, 10, 10, 10],
            "session_vwap": [100.0, 100.0, 100.0, 100.0, 100.0],
            "trade_allowed": [False, True, False, False, True],
            "target_position": [0, 1, -1, -1, -1],
            "raw_signal": [0, 1, -1, -1, -1],
            "is_last_bar_of_session": [False, False, False, False, True],
        }
    )
    variant = VWAPVariantConfig(
        name="vwap_time_filtered_baseline",
        family="prop_variant",
        mode="target_position",
        execution_profile="repo_realistic",
        initial_capital_usd=50_000.0,
        quantity_mode="fixed_quantity",
        fixed_quantity=1,
        time_windows=(TimeWindow("09:31:00", "09:32:00"), TimeWindow("09:34:00", "09:35:00")),
    )
    execution_model, instrument = build_execution_model_for_profile("MNQ", "repo_realistic")

    result = run_vwap_backtest(frame, variant, execution_model, instrument)

    assert len(result.trades) == 2
    assert result.trades.iloc[0]["direction"] == "long"
    assert pd.to_datetime(result.trades.iloc[0]["exit_time"]) == pd.Timestamp("2024-01-02 09:34:00")
    assert result.trades.iloc[1]["direction"] == "short"


def test_target_position_window_blocks_entry_from_flat() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
        ]
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "open": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 100.0],
            "close": [100.0, 100.0],
            "volume": [10, 10],
            "session_vwap": [100.0, 100.0],
            "trade_allowed": [False, False],
            "target_position": [1, 1],
            "raw_signal": [1, 1],
            "is_last_bar_of_session": [False, True],
        }
    )
    variant = VWAPVariantConfig(
        name="window_test_target",
        family="prop_variant",
        mode="target_position",
        execution_profile="repo_realistic",
        initial_capital_usd=50_000.0,
        quantity_mode="fixed_quantity",
        fixed_quantity=1,
    )
    execution_model, instrument = build_execution_model_for_profile("MNQ", "repo_realistic")

    result = run_vwap_backtest(frame, variant, execution_model, instrument)

    assert result.trades.empty


def test_daily_killswitch_stops_new_trades_after_first_loss() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:35:00",
            "2024-01-02 09:36:00",
            "2024-01-02 09:37:00",
            "2024-01-02 09:38:00",
        ]
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "open": [100.0, 100.0, 98.0, 98.0],
            "high": [100.0, 100.0, 98.0, 98.0],
            "low": [100.0, 98.0, 98.0, 98.0],
            "close": [100.0, 98.0, 98.0, 98.0],
            "volume": [10, 10, 10, 10],
            "session_vwap": [100.0, 99.0, 98.0, 98.0],
            "trade_allowed": [True, True, True, True],
            "entry_long": [True, False, True, False],
            "entry_short": [False, False, False, False],
            "exit_long": [False, False, False, False],
            "exit_short": [False, False, False, False],
            "stop_reference_long": [99.0, 99.0, 97.0, 97.0],
            "stop_reference_short": [pd.NA, pd.NA, pd.NA, pd.NA],
            "raw_signal": [1, 0, 1, 0],
            "is_last_bar_of_session": [False, False, False, True],
        }
    )
    variant = VWAPVariantConfig(
        name="kill_test",
        family="prop_variant",
        mode="discrete",
        execution_profile="repo_realistic",
        initial_capital_usd=50_000.0,
        quantity_mode="fixed_quantity",
        fixed_quantity=1,
        max_losses_per_day=1,
        daily_stop_threshold_usd=1.0,
    )
    execution_model, instrument = build_execution_model_for_profile("MNQ", "repo_realistic")

    result = run_vwap_backtest(frame, variant, execution_model, instrument)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "stop"
    assert bool(result.daily_results.iloc[0]["trading_halted"]) is True
    assert float(result.trades.iloc[0]["trade_risk_usd"]) > 0.0


def test_no_trade_after_daily_cutoff_window() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:35:00",
            "2024-01-02 09:37:00",
        ]
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "open": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 100.0],
            "close": [100.0, 100.0],
            "volume": [10, 10],
            "session_vwap": [100.0, 100.0],
            "trade_allowed": [False, False],
            "entry_long": [False, True],
            "entry_short": [False, False],
            "exit_long": [False, False],
            "exit_short": [False, False],
            "stop_reference_long": [99.0, 99.0],
            "stop_reference_short": [pd.NA, pd.NA],
            "raw_signal": [0, 1],
            "is_last_bar_of_session": [False, True],
        }
    )
    variant = VWAPVariantConfig(
        name="window_test",
        family="prop_variant",
        mode="discrete",
        execution_profile="repo_realistic",
        initial_capital_usd=50_000.0,
        quantity_mode="fixed_quantity",
        fixed_quantity=1,
        time_windows=(TimeWindow("09:35:00", "09:36:00"),),
    )
    execution_model, instrument = build_execution_model_for_profile("MNQ", "repo_realistic")

    result = run_vwap_backtest(frame, variant, execution_model, instrument)

    assert result.trades.empty


def test_long_short_symmetry_on_mirrored_target_positions() -> None:
    long_frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02 09:30:00", "2024-01-02 09:31:00"]),
            "session_date": pd.to_datetime(["2024-01-02", "2024-01-02"]).date,
            "open": [100.0, 100.0],
            "high": [100.0, 101.0],
            "low": [100.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [10, 10],
            "session_vwap": [100.0, 100.5],
            "trade_allowed": [True, True],
            "target_position": [0, 1],
            "raw_signal": [0, 1],
            "is_last_bar_of_session": [False, True],
        }
    )
    short_frame = long_frame.copy()
    short_frame["high"] = [100.0, 100.0]
    short_frame["low"] = [100.0, 99.0]
    short_frame["close"] = [100.0, 99.0]
    short_frame["target_position"] = [0, -1]
    short_frame["raw_signal"] = [0, -1]

    variant = VWAPVariantConfig(
        name="symmetry_test",
        family="baseline",
        mode="target_position",
        execution_profile="repo_realistic",
        initial_capital_usd=50_000.0,
        quantity_mode="fixed_quantity",
        fixed_quantity=1,
    )
    execution_model, instrument = build_execution_model_for_profile("MNQ", "repo_realistic")

    long_result = run_vwap_backtest(long_frame, variant, execution_model, instrument)
    short_result = run_vwap_backtest(short_frame, variant, execution_model, instrument)

    assert long_result.trades.iloc[0]["net_pnl_usd"] == pytest.approx(short_result.trades.iloc[0]["net_pnl_usd"])
