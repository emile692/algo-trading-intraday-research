from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.mnq_orb_3state_vol_sizing_variant_audit import (
    build_bucket_distribution,
    build_monthly_returns,
    build_worst_periods,
    prepare_daily_frame,
)


def _sample_daily() -> pd.DataFrame:
    session_dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-02-01",
            "2024-02-02",
        ]
    )
    rows: list[dict[str, object]] = []
    for variant_name, pnl_values in (
        ("single_15_60", [100.0, -50.0, 25.0, -75.0, 40.0, -10.0]),
        ("median_plateau_compact", [110.0, -20.0, 30.0, -40.0, 60.0, 5.0]),
        ("median_fast15_slow_60_70_80", [95.0, -30.0, 15.0, -60.0, 45.0, 0.0]),
    ):
        for session_date, pnl in zip(session_dates, pnl_values):
            rows.append(
                {
                    "session_date": session_date,
                    "daily_pnl_usd": pnl,
                    "daily_gross_pnl_usd": pnl + 2.5,
                    "daily_fees_usd": 2.5 if pnl != 0 else 0.0,
                    "daily_trade_count": 1.0 if pnl != 0 else 0.0,
                    "daily_loss_count": 1.0 if pnl < 0 else 0.0,
                    "equity": 50000.0,
                    "peak_equity": 50000.0,
                    "drawdown_usd": min(pnl, 0.0),
                    "drawdown_pct": abs(min(pnl, 0.0)) / 50000.0,
                    "green_day": pnl > 0,
                    "variant_name": variant_name,
                    "phase": "oos",
                    "daily_return": pnl / 50000.0,
                }
            )
    return pd.DataFrame(rows)


def _sample_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": list(range(1, 19)),
            "session_date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-02-01",
                    "2024-02-02",
                ]
                * 3
            ),
            "entry_time": pd.to_datetime(["2024-01-02 10:00:00"] * 18),
            "exit_time": pd.to_datetime(["2024-01-02 15:00:00"] * 18),
            "variant_name": ["single_15_60"] * 6 + ["median_plateau_compact"] * 6 + ["median_fast15_slow_60_70_80"] * 6,
            "bucket_label": [
                "low",
                "low",
                "mid",
                "high",
                "mid",
                "mid",
                "low",
                "mid",
                "mid",
                "mid",
                "mid",
                "high",
                "low",
                "low",
                "mid",
                "mid",
                "high",
                "high",
            ],
            "risk_multiplier": [
                0.5,
                0.5,
                1.0,
                0.25,
                1.0,
                1.0,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
                0.25,
                0.5,
                0.5,
                1.0,
                1.0,
                0.25,
                0.25,
            ],
            "net_pnl_usd": [
                100.0,
                -50.0,
                25.0,
                -75.0,
                40.0,
                -10.0,
                110.0,
                -20.0,
                30.0,
                -40.0,
                60.0,
                5.0,
                95.0,
                -30.0,
                15.0,
                -60.0,
                45.0,
                0.0,
            ],
        }
    )


def test_monthly_returns_aggregate_expected_values() -> None:
    prepared = prepare_daily_frame(_sample_daily(), _sample_trades())
    monthly = build_monthly_returns(prepared)

    baseline_jan = monthly.loc[
        (monthly["variant_name"] == "single_15_60") & (monthly["month"] == "2024-01")
    ].iloc[0]
    assert float(baseline_jan["monthly_pnl_usd"]) == pytest.approx(0.0)
    assert float(baseline_jan["monthly_return"]) == pytest.approx(0.0)
    assert not bool(baseline_jan["positive_month"])

    plateau_feb = monthly.loc[
        (monthly["variant_name"] == "median_plateau_compact") & (monthly["month"] == "2024-02")
    ].iloc[0]
    assert float(plateau_feb["monthly_pnl_usd"]) == pytest.approx(65.0)
    assert bool(plateau_feb["positive_month"])


def test_worst_periods_extract_day_week_month_and_rolling_dd() -> None:
    prepared = prepare_daily_frame(_sample_daily(), _sample_trades())
    worst = build_worst_periods(prepared)

    worst_day = worst.loc[
        (worst["variant_name"] == "single_15_60") & (worst["period_type"] == "day")
    ].iloc[0]
    assert float(worst_day["period_value"]) == pytest.approx(-75.0)

    worst_month = worst.loc[
        (worst["variant_name"] == "single_15_60") & (worst["period_type"] == "month")
    ].iloc[0]
    assert float(worst_month["period_value"]) == pytest.approx(0.0)

    rolling_20 = worst.loc[
        (worst["variant_name"] == "single_15_60") & (worst["period_type"] == "rolling_20d_maxdd")
    ].iloc[0]
    assert float(rolling_20["period_value"]) <= 0.0


def test_bucket_distribution_reports_day_and_trade_mix() -> None:
    prepared = prepare_daily_frame(_sample_daily(), _sample_trades())
    distribution = build_bucket_distribution(prepared, _sample_trades())

    plateau_mid = distribution.loc[
        (distribution["variant_name"] == "median_plateau_compact") & (distribution["bucket_label"] == "mid")
    ].iloc[0]
    assert int(plateau_mid["day_count"]) == 4
    assert float(plateau_mid["pct_days"]) == pytest.approx(4.0 / 6.0)
    assert int(plateau_mid["trade_count"]) == 4
    assert float(plateau_mid["pct_trades"]) == pytest.approx(4.0 / 6.0)

