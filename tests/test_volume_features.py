from __future__ import annotations

import pandas as pd
import pytest

from src.features.volume import add_rth_volume_history_features


def test_same_minute_volume_history_uses_only_prior_sessions() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02 09:30:00",
                    "2024-01-02 09:31:00",
                    "2024-01-03 09:30:00",
                    "2024-01-03 09:31:00",
                    "2024-01-04 09:30:00",
                    "2024-01-04 09:31:00",
                    "2024-01-05 09:30:00",
                    "2024-01-05 09:31:00",
                    "2024-01-08 09:30:00",
                    "2024-01-08 09:31:00",
                    "2024-01-09 09:30:00",
                    "2024-01-09 09:31:00",
                ]
            ),
            "open": [100.0] * 12,
            "high": [101.0] * 12,
            "low": [99.0] * 12,
            "close": [100.5] * 12,
            "volume": [10, 20, 11, 21, 12, 22, 13, 23, 14, 24, 100, 25],
        }
    )
    frame["session_date"] = frame["timestamp"].dt.date

    out = add_rth_volume_history_features(
        frame,
        opening_time="09:30:00",
        time_exit="09:31:00",
        rolling_windows=(10,),
        history_windows=(20,),
    )

    row = out.loc[out["timestamp"] == pd.Timestamp("2024-01-09 09:30:00")].iloc[0]
    assert float(row["same_minute_volume_mean_hist_20"]) == pytest.approx((10 + 11 + 12 + 13 + 14) / 5.0)


def test_intra_session_volume_window_uses_only_prior_bars() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02 09:30:00",
                    "2024-01-02 09:31:00",
                    "2024-01-02 09:32:00",
                    "2024-01-02 09:33:00",
                    "2024-01-02 09:34:00",
                    "2024-01-02 09:35:00",
                ]
            ),
            "open": [100.0] * 6,
            "high": [101.0] * 6,
            "low": [99.0] * 6,
            "close": [100.5] * 6,
            "volume": [1, 2, 3, 4, 5, 6],
        }
    )
    frame["session_date"] = frame["timestamp"].dt.date

    out = add_rth_volume_history_features(
        frame,
        opening_time="09:30:00",
        time_exit="09:35:00",
        rolling_windows=(5,),
        history_windows=(20,),
    )

    row = out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:35:00")].iloc[0]
    assert float(row["vol_mean_prev_5"]) == pytest.approx(3.0)

