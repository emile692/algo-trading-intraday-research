from __future__ import annotations

import pandas as pd

from src.analytics.weekly_open_reversion import (
    StudyConfig,
    build_weekly_observations,
    distance_bucket,
    summarize_observations,
)


def _synthetic_week(*, target_and_adverse_same_bar: bool = False) -> pd.DataFrame:
    index = pd.date_range("2024-01-08 09:30", "2024-01-10 15:59", freq="1min", tz="America/New_York")
    bars = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "instrument_id": 7},
        index=index,
    )
    tue = pd.Timestamp("2024-01-09 09:30", tz="America/New_York")
    bars.loc[tue:, ["open", "high", "low", "close"]] = [99.0, 99.2, 98.8, 99.0]
    hit = pd.Timestamp("2024-01-09 10:00", tz="America/New_York")
    bars.loc[hit, "high"] = 100.0
    bars.loc[hit:, "close"] = 100.0
    if target_and_adverse_same_bar:
        bars.loc[hit, "low"] = 98.0
    return bars.tz_convert("UTC")


def test_long_target_and_checkpoint_pnl_are_measured() -> None:
    observations, audit = build_weekly_observations(_synthetic_week(), symbol="NQ")
    row = observations.iloc[0]
    assert row["direction"] == "long"
    assert row["distance_points"] == 1.0
    assert row["target_before_1d_adverse"]
    assert row["minutes_to_target"] == 30.0
    assert row["pnl_tue_1130_d"] == 1.0
    assert (audit["status"] == "included").sum() == 1


def test_same_minute_target_and_adverse_is_conservative() -> None:
    observations, _ = build_weekly_observations(
        _synthetic_week(target_and_adverse_same_bar=True), symbol="NQ"
    )
    row = observations.iloc[0]
    assert row["same_bar_target_and_1d"]
    assert not row["target_before_1d_adverse"]
    assert row["mae_before_target_d"] == 1.0


def test_contract_change_is_excluded() -> None:
    bars = _synthetic_week()
    bars.loc[pd.Timestamp("2024-01-10 09:30", tz="UTC") :, "instrument_id"] = 8
    observations, audit = build_weekly_observations(bars, symbol="NQ")
    assert observations.empty
    assert "contract_change" in set(audit["reason"])


def test_bucket_boundaries() -> None:
    assert distance_bucket(0.2499) == "0-0.25%"
    assert distance_bucket(0.25) == "0.25-0.50%"
    assert distance_bucket(0.50) == "0.50-1.00%"
    assert distance_bucket(1.00) == "0.50-1.00%"
    assert distance_bucket(1.0001) == ">1.00%"


def test_rth_only_option_still_builds_observation() -> None:
    observations, _ = build_weekly_observations(
        _synthetic_week(), symbol="NQ", config=StudyConfig(include_overnight=False)
    )
    assert len(observations) == 1


def test_summary_adds_bounded_wilson_intervals() -> None:
    observations, _ = build_weekly_observations(_synthetic_week(), symbol="NQ")
    summary = summarize_observations(observations)
    assert 0.0 <= summary.loc[0, "target_rate_ci95_low"] <= 1.0
    assert 0.0 <= summary.loc[0, "target_rate_ci95_high"] <= 1.0
