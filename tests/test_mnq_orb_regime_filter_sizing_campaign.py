from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.mnq_orb_regime_filter_sizing_campaign import (
    _scale_nominal_trades_by_multiplier,
    apply_bucket_calibration,
    build_session_reference_features,
    build_state_mapping_from_is_scores,
    build_static_regime_controls,
    calibrate_quantile_buckets,
)


def test_build_session_reference_features_uses_previous_evening_and_preopen() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-02 16:00:00"),
                "session_date": pd.Timestamp("2024-01-02").date(),
                "continuous_session_date": pd.Timestamp("2024-01-02").date(),
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "atr_20": 9.0,
            },
            {
                "timestamp": pd.Timestamp("2024-01-02 18:00:00"),
                "session_date": pd.Timestamp("2024-01-02").date(),
                "continuous_session_date": pd.Timestamp("2024-01-03").date(),
                "open": 101.0,
                "high": 104.0,
                "low": 99.0,
                "close": 102.0,
                "atr_20": 9.0,
            },
            {
                "timestamp": pd.Timestamp("2024-01-03 08:00:00"),
                "session_date": pd.Timestamp("2024-01-03").date(),
                "continuous_session_date": pd.Timestamp("2024-01-03").date(),
                "open": 102.0,
                "high": 105.0,
                "low": 98.0,
                "close": 103.0,
                "atr_20": 9.5,
            },
            {
                "timestamp": pd.Timestamp("2024-01-03 09:30:00"),
                "session_date": pd.Timestamp("2024-01-03").date(),
                "continuous_session_date": pd.Timestamp("2024-01-03").date(),
                "open": 103.0,
                "high": 103.5,
                "low": 102.5,
                "close": 103.0,
                "atr_20": 10.0,
            },
            {
                "timestamp": pd.Timestamp("2024-01-03 16:00:00"),
                "session_date": pd.Timestamp("2024-01-03").date(),
                "continuous_session_date": pd.Timestamp("2024-01-03").date(),
                "open": 104.0,
                "high": 104.5,
                "low": 103.5,
                "close": 104.0,
                "atr_20": 10.0,
            },
        ]
    )

    refs = build_session_reference_features(frame, opening_time="09:30:00", time_exit="16:00:00")
    row = refs.loc[refs["session_date"] == pd.Timestamp("2024-01-03").date()].iloc[0]

    assert float(row["rth_open"]) == pytest.approx(103.0)
    assert float(row["prev_rth_close"]) == pytest.approx(100.0)
    assert float(row["atr_20_open"]) == pytest.approx(10.0)
    assert float(row["overnight_range_pts"]) == pytest.approx(7.0)


def test_quantile_buckets_are_calibrated_on_is_only() -> None:
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


def test_build_static_regime_controls_applies_trade_filter() -> None:
    regime_df = pd.DataFrame(
        {
            "session_date": [
                pd.Timestamp("2024-01-02").date(),
                pd.Timestamp("2024-01-03").date(),
                pd.Timestamp("2024-01-04").date(),
            ],
            "phase": ["is", "is", "oos"],
        }
    )
    bucket_labels = pd.Series(["low", "mid", "high"])

    controls = build_static_regime_controls(
        regime_df=regime_df,
        feature_name="atr_ratio_10_30",
        bucket_labels=bucket_labels,
        bucket_multipliers={"low": 0.0, "mid": 1.0, "high": 1.0},
    )

    assert controls["risk_multiplier"].tolist() == [0.0, 1.0, 1.0]
    assert controls["skip_trade"].tolist() == [True, False, False]


def test_build_state_mapping_and_controls_apply_discrete_sizing() -> None:
    feature_rows = pd.DataFrame(
        {
            "bucket_label": ["low", "mid", "high"],
            "is_composite_score": [-1.0, 0.5, 2.0],
            "is_expectancy": [-10.0, 20.0, 50.0],
            "is_profit_factor": [0.8, 1.1, 1.4],
        }
    )
    mapping = build_state_mapping_from_is_scores(feature_rows, multipliers_by_rank=(0.5, 0.75, 1.0))

    regime_df = pd.DataFrame(
        {
            "session_date": [
                pd.Timestamp("2024-01-02").date(),
                pd.Timestamp("2024-01-03").date(),
                pd.Timestamp("2024-01-04").date(),
            ],
            "phase": ["is", "is", "oos"],
        }
    )
    controls = build_static_regime_controls(
        regime_df=regime_df,
        feature_name="signal_extension_over_or",
        bucket_labels=pd.Series(["low", "mid", "high"]),
        bucket_multipliers=mapping,
    )

    assert mapping == {"low": 0.5, "mid": 0.75, "high": 1.0}
    assert controls["risk_multiplier"].tolist() == [0.5, 0.75, 1.0]


def test_scale_nominal_trades_by_multiplier_resizes_quantity_and_pnl() -> None:
    nominal_trades = pd.DataFrame(
        [
            {
                "trade_id": 1,
                "session_date": pd.Timestamp("2024-01-02").date(),
                "entry_time": pd.Timestamp("2024-01-02 10:00:00"),
                "entry_price": 100.0,
                "quantity": 4,
                "risk_per_trade_pct": 1.5,
                "risk_budget_usd": 750.0,
                "risk_per_contract_usd": 150.0,
                "actual_risk_usd": 600.0,
                "trade_risk_usd": 600.0,
                "pnl_ticks": 10.0,
                "pnl_usd": 200.0,
                "fees": 10.0,
                "net_pnl_usd": 190.0,
                "notional_usd": 20000.0,
                "leverage_used": 0.4,
            }
        ]
    )
    controls = pd.DataFrame(
        {
            "session_date": [pd.Timestamp("2024-01-02").date()],
            "risk_multiplier": [0.5],
        }
    )

    scaled = _scale_nominal_trades_by_multiplier(
        nominal_trades=nominal_trades,
        controls=controls,
        account_size_usd=50_000.0,
        base_risk_pct=1.5,
        tick_value_usd=5.0,
        point_value_usd=20.0,
        commission_per_side_usd=1.25,
    )

    row = scaled.iloc[0]
    assert int(row["quantity"]) == 2
    assert float(row["risk_budget_usd"]) == pytest.approx(375.0)
    assert float(row["pnl_usd"]) == pytest.approx(100.0)
    assert float(row["fees"]) == pytest.approx(5.0)
    assert float(row["net_pnl_usd"]) == pytest.approx(95.0)
