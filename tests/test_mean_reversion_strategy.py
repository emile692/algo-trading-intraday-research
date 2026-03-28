from __future__ import annotations

import pandas as pd

from src.config.mean_reversion_campaign import MeanReversionVariantConfig
from src.strategy.mean_reversion import build_mean_reversion_signal_frame


def test_vwap_extension_entries_shift_to_next_open() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:35:00",
            "2024-01-02 09:40:00",
            "2024-01-02 09:45:00",
        ]
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "open": [100.0, 100.0, 99.0, 98.0],
            "high": [100.2, 100.1, 99.2, 99.5],
            "low": [99.8, 98.8, 96.8, 97.8],
            "close": [100.0, 99.0, 97.0, 98.5],
            "session_vwap": [100.0, 100.0, 100.0, 99.8],
            "atr_14": [1.0, 1.0, 1.0, 1.0],
            "adx_14": [10.0, 10.0, 10.0, 10.0],
            "ema_slope_atr_20_3": [0.0, 0.0, 0.0, 0.0],
            "vwap_slope_atr_3": [0.0, 0.0, 0.0, 0.0],
            "trend_day_score": [1.0, 1.0, 1.0, 1.0],
            "session_range_so_far_atr_14": [0.2, 1.2, 3.2, 3.0],
            "persistent_vwap_distance_4_14": [0.0, 1.0, 2.5, 2.0],
            "minutes_from_open": [0, 5, 10, 15],
            "minutes_to_close": [30, 25, 20, 15],
            "is_last_bar_of_session": [False, False, False, True],
        }
    )
    variant = MeanReversionVariantConfig(
        name="mnq_5m_vwap_ext_test",
        family="vwap_extension_reversion",
        symbol="MNQ",
        timeframe="5m",
        entry_start="09:30:00",
        entry_end="15:30:00",
        skip_first_minutes=0,
        skip_last_minutes=0,
        atr_period=14,
        extension_mode="atr",
        extension_threshold=2.0,
        adx_max=20.0,
        ema_slope_max_atr=0.2,
        vwap_slope_max_atr=0.2,
        anti_trend_day_max=3.0,
        session_range_max_atr=4.0,
        persistent_vwap_distance_max=3.0,
    )

    out = build_mean_reversion_signal_frame(frame, variant)

    assert int(out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:40:00"), "raw_signal"].iloc[0]) == 1
    assert bool(out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:40:00"), "entry_long"].iloc[0]) is False
    assert bool(out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:45:00"), "entry_long"].iloc[0]) is True
    assert float(out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:45:00"), "target_reference_long"].iloc[0]) == 100.0
