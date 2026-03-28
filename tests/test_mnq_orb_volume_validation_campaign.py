from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.analytics.mnq_orb_volume_validation_campaign import (
    MnqOrbVolumeValidationSpec,
    _build_fixed_nominal_baseline,
    apply_bucket_calibration,
    calibrate_quantile_buckets,
)


def test_volume_bucket_calibration_is_only() -> None:
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


def test_build_fixed_nominal_baseline_forces_quantity_one() -> None:
    signal_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02 09:30:00",
                    "2024-01-02 10:00:00",
                    "2024-01-02 10:01:00",
                ]
            ),
            "session_date": [pd.Timestamp("2024-01-02").date()] * 3,
            "signal": [0, 1, 0],
            "or_high": [100.0, 100.0, 100.0],
            "or_low": [99.0, 99.0, 99.0],
            "open": [99.5, 100.0, 100.5],
            "high": [100.0, 100.5, 103.0],
            "low": [99.0, 99.5, 100.5],
            "close": [99.8, 100.4, 102.5],
        }
    )
    analysis = SimpleNamespace(
        symbol="MNQ",
        signal_df=signal_df,
        baseline=SimpleNamespace(
            time_exit="16:00:00",
            stop_buffer_ticks=0,
            target_multiple=1.5,
            entry_on_next_open=True,
        ),
    )
    spec = MnqOrbVolumeValidationSpec(fixed_contracts=1, commission_per_side_usd=0.0, slippage_ticks=0.0)

    _, trades = _build_fixed_nominal_baseline(
        analysis=analysis,
        selected_sessions={pd.Timestamp("2024-01-02").date()},
        spec=spec,
    )

    assert len(trades) == 1
    assert int(trades.iloc[0]["quantity"]) == 1
    assert float(trades.iloc[0]["net_pnl_usd"]) > 0.0

