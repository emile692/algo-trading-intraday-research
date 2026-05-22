from __future__ import annotations

import pandas as pd

from src.engine.execution_model import ExecutionModel
from src.engine.volume_climax_pullback_v2_backtester import run_volume_climax_pullback_v2_backtest
from src.engine.vwap_backtester import InstrumentDetails
from src.strategy.volume_climax_pullback_v2 import VolumeClimaxPullbackV2Variant


TZ = "America/New_York"


def _variant(**overrides: object) -> VolumeClimaxPullbackV2Variant:
    payload: dict[str, object] = {
        "name": "hybrid_test",
        "family": "dynamic_exit",
        "timeframe": "1h",
        "volume_quantile": 0.95,
        "volume_lookback": 50,
        "min_body_fraction": 0.5,
        "min_range_atr": 1.2,
        "trend_ema_window": None,
        "ema_slope_threshold": None,
        "atr_percentile_low": None,
        "atr_percentile_high": None,
        "compression_ratio_max": None,
        "entry_mode": "next_open",
        "pullback_fraction": None,
        "confirmation_window": None,
        "exit_mode": "fixed_rr",
        "rr_target": 1.0,
        "atr_target_multiple": None,
        "time_stop_bars": 2,
        "trailing_atr_multiple": 0.5,
        "session_overlay": "all_rth",
    }
    payload.update(overrides)
    return VolumeClimaxPullbackV2Variant(**payload)


def _instrument() -> InstrumentDetails:
    return InstrumentDetails(
        symbol="MNQ",
        asset_class="futures",
        tick_size=0.25,
        tick_value_usd=0.5,
        point_value_usd=1.0,
        commission_per_side_usd=0.0,
        slippage_ticks=0,
    )


def _execution_model() -> ExecutionModel:
    return ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0, tick_size=0.25)


def _minute_df(
    *,
    start: str,
    end: str,
    default_open: float = 100.0,
) -> pd.DataFrame:
    timestamps = pd.date_range(start, end, freq="1min", tz=TZ)
    rows: list[dict[str, object]] = []
    for timestamp in timestamps:
        rows.append(
            {
                "timestamp": timestamp,
                "open": default_open,
                "high": default_open + 0.20,
                "low": default_open - 0.20,
                "close": default_open,
                "volume": 100.0,
                "session_date": timestamp.date(),
            }
        )
    return pd.DataFrame(rows)


def _signal_df(
    *,
    actionable_time: str,
    setup_time: str,
    direction: int = 1,
    stop_price: float = 99.0,
) -> pd.DataFrame:
    actionable_ts = pd.Timestamp(actionable_time, tz=TZ)
    setup_ts = pd.Timestamp(setup_time, tz=TZ)
    return pd.DataFrame(
        [
            {
                "timestamp": actionable_ts,
                "session_date": actionable_ts.date(),
                "open": 100.0,
                "high": 100.2,
                "low": 99.8,
                "close": 100.0,
                "signal": direction,
                "setup_signal_time": setup_ts,
                "setup_reference_close": 100.0,
                "setup_reference_range": 2.0,
                "setup_reference_atr": 2.0,
                "setup_reference_vwap": 101.0,
                "setup_stop_reference_long": stop_price if direction == 1 else pd.NA,
                "setup_stop_reference_short": stop_price if direction == -1 else pd.NA,
            }
        ]
    )


def test_hybrid_entry_timing_uses_1m_path() -> None:
    minute_df = _minute_df(start="2024-01-02 13:30:00", end="2024-01-02 15:30:00")
    signal_df = _signal_df(actionable_time="2024-01-02 14:30:00", setup_time="2024-01-02 13:30:00")
    variant = _variant(time_stop_bars=10)

    next_bar = run_volume_climax_pullback_v2_backtest(
        signal_df=signal_df,
        variant=variant,
        execution_model=_execution_model(),
        instrument=_instrument(),
        execution_timeframe="1min",
        minute_df=minute_df,
        entry_timing="next_execution_bar_open",
        protective_orders_active_from="after_entry_fill",
    ).trades
    same_ts = run_volume_climax_pullback_v2_backtest(
        signal_df=signal_df,
        variant=variant,
        execution_model=_execution_model(),
        instrument=_instrument(),
        execution_timeframe="1min",
        minute_df=minute_df,
        entry_timing="same_timestamp_execution_open",
        protective_orders_active_from="after_entry_fill",
    ).trades

    assert pd.Timestamp(next_bar.iloc[0]["entry_time"]) == pd.Timestamp("2024-01-02 14:31:00", tz=TZ)
    assert pd.Timestamp(same_ts.iloc[0]["entry_time"]) == pd.Timestamp("2024-01-02 14:30:00", tz=TZ)


def test_hybrid_stop_is_active_after_entry_fill() -> None:
    minute_df = _minute_df(start="2024-01-02 13:30:00", end="2024-01-02 15:30:00")
    minute_df.loc[minute_df["timestamp"] == pd.Timestamp("2024-01-02 14:35:00", tz=TZ), "low"] = 98.5
    signal_df = _signal_df(actionable_time="2024-01-02 14:30:00", setup_time="2024-01-02 13:30:00")

    trades = run_volume_climax_pullback_v2_backtest(
        signal_df=signal_df,
        variant=_variant(time_stop_bars=10),
        execution_model=_execution_model(),
        instrument=_instrument(),
        execution_timeframe="1min",
        minute_df=minute_df,
        entry_timing="next_execution_bar_open",
        protective_orders_active_from="after_entry_fill",
    ).trades

    trade = trades.iloc[0]
    assert pd.Timestamp(trade["entry_time"]) == pd.Timestamp("2024-01-02 14:31:00", tz=TZ)
    assert pd.Timestamp(trade["exit_time"]) == pd.Timestamp("2024-01-02 14:35:00", tz=TZ)
    assert str(trade["exit_reason"]) == "stop_1m"


def test_hybrid_target_is_active_after_entry_fill() -> None:
    minute_df = _minute_df(start="2024-01-02 13:30:00", end="2024-01-02 15:30:00")
    minute_df.loc[minute_df["timestamp"] == pd.Timestamp("2024-01-02 14:40:00", tz=TZ), "high"] = 101.2
    signal_df = _signal_df(actionable_time="2024-01-02 14:30:00", setup_time="2024-01-02 13:30:00")

    trades = run_volume_climax_pullback_v2_backtest(
        signal_df=signal_df,
        variant=_variant(time_stop_bars=10),
        execution_model=_execution_model(),
        instrument=_instrument(),
        execution_timeframe="1min",
        minute_df=minute_df,
        entry_timing="next_execution_bar_open",
        protective_orders_active_from="after_entry_fill",
    ).trades

    trade = trades.iloc[0]
    assert pd.Timestamp(trade["exit_time"]) == pd.Timestamp("2024-01-02 14:40:00", tz=TZ)
    assert str(trade["exit_reason"]) == "target_1m"


def test_hybrid_stop_wins_when_stop_and_target_touch_same_minute() -> None:
    minute_df = _minute_df(start="2024-01-02 13:30:00", end="2024-01-02 15:30:00")
    minute_mask = minute_df["timestamp"] == pd.Timestamp("2024-01-02 14:35:00", tz=TZ)
    minute_df.loc[minute_mask, "low"] = 98.8
    minute_df.loc[minute_mask, "high"] = 101.2
    signal_df = _signal_df(actionable_time="2024-01-02 14:30:00", setup_time="2024-01-02 13:30:00")

    trades = run_volume_climax_pullback_v2_backtest(
        signal_df=signal_df,
        variant=_variant(time_stop_bars=10),
        execution_model=_execution_model(),
        instrument=_instrument(),
        execution_timeframe="1min",
        minute_df=minute_df,
        entry_timing="next_execution_bar_open",
        protective_orders_active_from="after_entry_fill",
    ).trades

    trade = trades.iloc[0]
    assert str(trade["exit_reason"]) == "stop_ambiguous_first_1m"
    assert float(trade["exit_price"]) == 99.0


def test_hybrid_time_stop_keeps_1h_units() -> None:
    minute_df = _minute_df(start="2024-01-02 13:30:00", end="2024-01-02 16:35:00")
    signal_df = _signal_df(actionable_time="2024-01-02 14:30:00", setup_time="2024-01-02 13:30:00")

    trades = run_volume_climax_pullback_v2_backtest(
        signal_df=signal_df,
        variant=_variant(time_stop_bars=2, rr_target=10.0),
        execution_model=_execution_model(),
        instrument=_instrument(),
        execution_timeframe="1min",
        minute_df=minute_df,
        entry_timing="next_execution_bar_open",
        protective_orders_active_from="after_entry_fill",
    ).trades

    trade = trades.iloc[0]
    assert pd.Timestamp(trade["entry_time"]) == pd.Timestamp("2024-01-02 14:31:00", tz=TZ)
    assert pd.Timestamp(trade["time_stop_at"]) == pd.Timestamp("2024-01-02 16:31:00", tz=TZ)
    assert pd.Timestamp(trade["exit_time"]) == pd.Timestamp("2024-01-02 16:31:00", tz=TZ)
    assert str(trade["exit_reason"]) == "time_stop_1m"


def test_hybrid_eod_flat_uses_last_available_minute() -> None:
    minute_df = _minute_df(start="2024-01-02 13:30:00", end="2024-01-02 16:00:00")
    signal_df = _signal_df(actionable_time="2024-01-02 14:30:00", setup_time="2024-01-02 13:30:00")

    trades = run_volume_climax_pullback_v2_backtest(
        signal_df=signal_df,
        variant=_variant(time_stop_bars=10, rr_target=10.0),
        execution_model=_execution_model(),
        instrument=_instrument(),
        execution_timeframe="1min",
        minute_df=minute_df,
        entry_timing="next_execution_bar_open",
        protective_orders_active_from="next_execution_bar",
    ).trades

    trade = trades.iloc[0]
    assert pd.Timestamp(trade["protective_orders_active_at"]) == pd.Timestamp("2024-01-02 14:32:00", tz=TZ)
    assert pd.Timestamp(trade["exit_time"]) == pd.Timestamp("2024-01-02 16:00:00", tz=TZ)
    assert str(trade["exit_reason"]) == "eod_flat_1m"
