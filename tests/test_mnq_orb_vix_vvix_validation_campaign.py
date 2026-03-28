from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.analytics.mnq_orb_vix_vvix_validation_campaign import (
    apply_bucket_calibration,
    build_vix_vvix_feature_frame,
    calibrate_quantile_buckets,
)


def test_vix_vvix_bucket_calibration_is_only() -> None:
    calibration = calibrate_quantile_buckets(
        feature_name="demo",
        is_values=pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        bucket_count=3,
    )

    bucketed = apply_bucket_calibration(pd.Series([1.0, 2.0, 6.0, 100.0]), calibration)

    assert len(calibration.labels) == 3
    assert calibration.bins[-1] < 100.0
    assert str(bucketed.iloc[0]) == "low"
    assert str(bucketed.iloc[2]) == "high"
    assert str(bucketed.iloc[3]) == "high"


def test_build_vix_vvix_feature_frame_merges_daily_context_and_phases() -> None:
    analysis = SimpleNamespace(
        candidate_df=pd.DataFrame(
            {
                "session_date": [pd.Timestamp("2024-01-03").date(), pd.Timestamp("2024-01-04").date()],
                "signal_index": [0, 1],
            }
        ),
        signal_df=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-03 10:02:00", "2024-01-04 10:18:00"]),
                "session_date": [pd.Timestamp("2024-01-03").date(), pd.Timestamp("2024-01-04").date()],
                "close": [100.5, 101.5],
                "or_high": [100.0, 101.0],
                "or_low": [99.0, 100.0],
                "or_width": [1.0, 1.0],
                "signal": [1, -1],
            }
        ),
        baseline=SimpleNamespace(opening_time="09:30:00", or_minutes=30),
        is_sessions=[pd.Timestamp("2024-01-03").date()],
        oos_sessions=[pd.Timestamp("2024-01-04").date()],
    )
    baseline_trades = pd.DataFrame(
        {
            "session_date": [pd.Timestamp("2024-01-03").date(), pd.Timestamp("2024-01-04").date()],
            "trade_id": [1, 2],
            "entry_time": pd.to_datetime(["2024-01-03 10:03:00", "2024-01-04 10:19:00"]),
            "exit_time": pd.to_datetime(["2024-01-03 11:00:00", "2024-01-04 11:00:00"]),
            "direction": ["long", "short"],
            "quantity": [1, 1],
            "net_pnl_usd": [100.0, -50.0],
            "pnl_usd": [101.0, -49.0],
            "fees": [1.0, 1.0],
            "exit_reason": ["target", "stop"],
        }
    )
    daily_features = pd.DataFrame(
        {
            "session_date": [pd.Timestamp("2024-01-03").date(), pd.Timestamp("2024-01-04").date()],
            "vix_reference_date_t1": [pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-03").date()],
            "vvix_reference_date_t1": [pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-03").date()],
            "vix_level_t1": [15.0, 16.0],
            "vvix_level_t1": [90.0, 92.0],
            "vvix_over_vix_t1": [6.0, 5.75],
            "vix_pct_63_t1": [0.40, 0.45],
            "vix_pct_126_t1": [0.42, 0.46],
            "vix_pct_252_t1": [0.43, 0.47],
            "vvix_pct_63_t1": [0.50, 0.55],
            "vvix_pct_126_t1": [0.52, 0.56],
            "vvix_pct_252_t1": [0.53, 0.57],
            "vix_change_1d": [0.01, -0.02],
            "vix_change_5d": [0.05, 0.03],
            "vvix_change_1d": [0.02, -0.01],
            "vvix_change_5d": [0.06, 0.04],
        }
    )

    feature_frame = build_vix_vvix_feature_frame(analysis, baseline_trades, daily_features)

    assert feature_frame["phase"].tolist() == ["is", "oos"]
    assert feature_frame["breakout_side"].tolist() == ["long", "short"]
    assert feature_frame["breakout_timing_bucket"].tolist() == ["early", "mid"]
    assert feature_frame["vix_reference_date_t1"].tolist() == [
        pd.Timestamp("2024-01-02").date(),
        pd.Timestamp("2024-01-03").date(),
    ]
