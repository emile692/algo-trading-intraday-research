import pandas as pd

from src.engine.execution_model import ExecutionModel
from src.engine.volume_climax_pullback_v2_backtester import run_volume_climax_pullback_v2_backtest_hybrid_1m
from src.engine.vwap_backtester import InstrumentDetails, resolve_instrument_details
from src.strategy.volume_climax_pullback_v2 import VolumeClimaxPullbackV2Variant, build_volume_climax_pullback_v3_variants


def _variant() -> VolumeClimaxPullbackV2Variant:
    v = build_volume_climax_pullback_v3_variants("MNQ")[0]
    return VolumeClimaxPullbackV2Variant(**{**v.__dict__, "name": "hybrid_test", "entry_mode": "next_open", "exit_mode": "fixed_rr", "rr_target": 1.0, "time_stop_bars": 2})


def _instrument() -> InstrumentDetails:
    return resolve_instrument_details("MNQ")


def _signal_row(ts: str = "2024-01-02 14:30:00") -> pd.DataFrame:
    return pd.DataFrame([
        {
            "timestamp": pd.Timestamp(ts),
            "session_date": pd.Timestamp("2024-01-02").date(),
            "signal": 1,
            "setup_signal_time": pd.Timestamp("2024-01-02 13:30:00"),
            "setup_stop_reference_long": 99.0,
            "setup_stop_reference_short": 102.0,
            "setup_reference_atr": 1.0,
            "setup_reference_vwap": 110.0,
            "setup_reference_range": 2.0,
        }
    ])


def test_entry_timing_next_bar_and_same_timestamp() -> None:
    minute = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02 14:30:00", "2024-01-02 14:31:00", "2024-01-02 14:32:00"]),
            "open": [100.0, 101.0, 101.0],
            "high": [100.2, 101.2, 101.2],
            "low": [99.8, 100.8, 100.8],
            "close": [100.0, 101.0, 101.0],
        }
    )
    sig = _signal_row()
    model = ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0.0, tick_size=0.25)

    nxt = run_volume_climax_pullback_v2_backtest_hybrid_1m(sig, minute, _variant(), model, _instrument())
    same = run_volume_climax_pullback_v2_backtest_hybrid_1m(
        sig, minute, _variant(), model, _instrument(), entry_timing="same_timestamp_execution_open"
    )

    assert pd.Timestamp(nxt.trades.iloc[0]["entry_time"]) == pd.Timestamp("2024-01-02 14:31:00")
    assert pd.Timestamp(same.trades.iloc[0]["entry_time"]) == pd.Timestamp("2024-01-02 14:30:00")


def test_stop_target_and_ambiguous_and_time_stop() -> None:
    sig = _signal_row()
    model = ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0.0, tick_size=0.25)
    base = pd.date_range("2024-01-02 14:31:00", periods=130, freq="1min")
    minute = pd.DataFrame({"timestamp": base, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0})
    # stop at 14:35
    minute.loc[4, "low"] = 98.8
    out = run_volume_climax_pullback_v2_backtest_hybrid_1m(sig, minute, _variant(), model, _instrument())
    assert out.trades.iloc[0]["exit_reason"] == "stop_1m"
    assert pd.Timestamp(out.trades.iloc[0]["exit_time"]) == pd.Timestamp("2024-01-02 14:35:00")

    # target at 14:40
    minute2 = minute.copy()
    minute2.loc[4, "low"] = 99.5
    minute2.loc[9, "high"] = 102.2
    out2 = run_volume_climax_pullback_v2_backtest_hybrid_1m(sig, minute2, _variant(), model, _instrument())
    assert out2.trades.iloc[0]["exit_reason"] == "target_1m"
    assert pd.Timestamp(out2.trades.iloc[0]["exit_time"]) == pd.Timestamp("2024-01-02 14:40:00")

    # ambiguous
    minute3 = minute.copy()
    minute3.loc[4, "low"] = 98.8
    minute3.loc[4, "high"] = 102.2
    out3 = run_volume_climax_pullback_v2_backtest_hybrid_1m(sig, minute3, _variant(), model, _instrument())
    assert out3.trades.iloc[0]["exit_reason"] == "stop_ambiguous_first_1m"

    # time stop = 2h from 14:31 => 16:31
    minute4 = pd.DataFrame({"timestamp": base, "open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0})
    out4 = run_volume_climax_pullback_v2_backtest_hybrid_1m(sig, minute4, _variant(), model, _instrument())
    assert out4.trades.iloc[0]["exit_reason"] == "time_stop_1m"
    assert pd.Timestamp(out4.trades.iloc[0]["exit_time"]) == pd.Timestamp("2024-01-02 16:31:00")
