import pandas as pd
import pytest
import numpy as np

from src.config.vwap_campaign import VWAPVariantConfig
from src.data.resampling import resample_ohlcv
from src.strategy.vwap import (
    build_vwap_signal_frame,
    generate_pullback_continuation_signals,
    prepare_vwap_feature_frame,
)


def _strategy_input_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
            "2024-01-03 09:30:00",
            "2024-01-03 09:31:00",
        ]
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 101.0, 100.0, 200.0, 201.0],
            "high": [101.0, 101.0, 100.0, 205.0, 202.0],
            "low": [98.0, 99.0, 98.0, 199.0, 200.0],
            "close": [101.0, 99.0, 98.0, 204.0, 201.0],
            "volume": [100, 100, 100, 50, 50],
        }
    )


def test_prepare_vwap_feature_frame_resets_vwap_each_session() -> None:
    out = prepare_vwap_feature_frame(_strategy_input_frame())

    session_two_first = out.loc[out["timestamp"] == pd.Timestamp("2024-01-03 09:30:00")].iloc[0]
    typical_price = (205.0 + 199.0 + 204.0) / 3.0

    assert session_two_first["session_vwap"] == pytest.approx(typical_price)


def test_paper_baseline_signal_is_above_below_vwap_on_previous_close() -> None:
    feat = prepare_vwap_feature_frame(_strategy_input_frame())
    variant = VWAPVariantConfig(
        name="paper_vwap_baseline",
        family="baseline",
        mode="target_position",
        execution_profile="paper_reference",
        initial_capital_usd=25_000.0,
        quantity_mode="paper_full_notional",
    )

    out = build_vwap_signal_frame(feat, variant)

    assert int(out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:30:00"), "target_position"].iloc[0]) == 0
    assert int(out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:31:00"), "target_position"].iloc[0]) == 1
    assert int(out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:32:00"), "target_position"].iloc[0]) == -1


def test_prepare_vwap_feature_frame_accepts_precomputed_price_volume_for_resampled_vwap() -> None:
    timestamps = pd.date_range("2024-01-02 09:30:00", periods=10, freq="1min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 110.0, 111.0, 112.0, 113.0, 114.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0, 111.0, 112.0, 113.0, 114.0, 115.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 109.0, 110.0, 111.0, 112.0, 113.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 110.5, 111.5, 112.5, 113.5, 114.5],
            "volume": [10, 20, 30, 40, 50, 10, 20, 30, 40, 50],
        }
    )
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    frame["vwap_pv_typical"] = typical * frame["volume"]
    resampled = resample_ohlcv(frame, rule="5min", aggregation_overrides={"vwap_pv_typical": "sum"})

    out = prepare_vwap_feature_frame(resampled, vwap_price_volume_col="vwap_pv_typical")

    expected_second_bar_vwap = frame["vwap_pv_typical"].sum() / frame["volume"].sum()
    second_bar = out.iloc[1]

    assert len(out) == 2
    assert second_bar["session_vwap"] == pytest.approx(expected_second_bar_vwap)


def test_paper_baseline_regime_filter_uses_prev_bar_slope_and_distance() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
            "2024-01-02 09:33:00",
        ]
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "prev_close": [np.nan, 101.0, 101.2, 102.0],
            "prev_session_vwap": [np.nan, 100.0, 100.4, 100.9],
            "atr_2": [np.nan, 0.8, 0.8, 0.8],
        }
    )
    variant = VWAPVariantConfig(
        name="vwap_baseline_regime_filtered",
        family="prop_variant",
        mode="target_position",
        execution_profile="repo_realistic",
        initial_capital_usd=50_000.0,
        quantity_mode="fixed_quantity",
        fixed_quantity=1,
        slope_lookback=1,
        slope_threshold=0.0,
        require_vwap_slope_alignment=True,
        max_vwap_distance_atr=1.0,
        atr_period=2,
    )

    out = build_vwap_signal_frame(frame, variant)

    assert int(out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:31:00"), "raw_target_position"].iloc[0]) == 1
    assert int(out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:31:00"), "target_position"].iloc[0]) == 0
    assert int(out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:32:00"), "target_position"].iloc[0]) == 1
    assert int(out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:33:00"), "target_position"].iloc[0]) == 0


def test_pullback_continuation_entries_shift_to_next_bar_open() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
            "2024-01-02 09:33:00",
        ]
    )
    session_date = timestamps.normalize().date
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": session_date,
            "open": [100.0, 101.0, 100.0, 101.5],
            "high": [101.0, 101.2, 102.0, 103.0],
            "low": [99.0, 100.0, 100.0, 101.0],
            "close": [101.0, 100.2, 101.8, 102.5],
            "volume": [100, 100, 100, 100],
            "session_vwap": [100.0, 100.1, 100.4, 100.8],
            "prev_close": [np.nan, 101.0, 100.2, 101.8],
            "prev_session_vwap": [np.nan, 100.0, 100.1, 100.4],
            "prev_high": [np.nan, 101.0, 101.2, 102.0],
            "prev_low": [np.nan, 99.0, 100.0, 100.0],
            "is_first_bar_of_session": [True, False, False, False],
            "is_last_bar_of_session": [False, False, False, True],
            "trade_allowed": [True, True, True, True],
            "atr_2": [0.8, 0.8, 0.8, 0.8],
        }
    )
    variant = VWAPVariantConfig(
        name="vwap_pullback_continuation",
        family="prop_variant",
        mode="discrete",
        execution_profile="repo_realistic",
        initial_capital_usd=50_000.0,
        quantity_mode="fixed_quantity",
        fixed_quantity=1,
        slope_lookback=1,
        atr_period=2,
        atr_buffer=5.0,
        stop_buffer=5.0,
        pullback_lookback=2,
        confirmation_threshold=0.0,
    )

    out = generate_pullback_continuation_signals(frame, variant)

    assert int(out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:32:00"), "raw_signal"].iloc[0]) == 1
    assert bool(out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:32:00"), "entry_long"].iloc[0]) is False
    assert bool(out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:33:00"), "entry_long"].iloc[0]) is True
    assert pd.notna(out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:33:00"), "stop_reference_long"].iloc[0])
