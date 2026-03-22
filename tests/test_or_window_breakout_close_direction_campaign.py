from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.or_window_breakout_close_direction_campaign import (
    evaluate_session_window,
    summarize_breakout_results,
)


def _build_session(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    data = pd.DataFrame(rows, columns=["time", "high", "low", "close"])
    data["timestamp"] = pd.to_datetime("2024-01-02 " + data["time"]).dt.tz_localize("America/New_York")
    data["minute_of_day"] = data["timestamp"].dt.hour * 60 + data["timestamp"].dt.minute
    data["session_date"] = data["timestamp"].dt.date
    return data[["timestamp", "high", "low", "close", "minute_of_day", "session_date"]]


def test_evaluate_session_window_up_breakout_success() -> None:
    session = _build_session(
        [
            ("09:30", 100.0, 99.0, 99.5),
            ("09:31", 100.5, 99.2, 100.0),
            ("09:32", 100.8, 99.4, 100.2),
            ("09:33", 101.0, 99.6, 100.4),
            ("09:34", 100.9, 99.5, 100.1),
            ("09:35", 101.2, 100.0, 101.1),
            ("10:00", 100.8, 98.8, 99.0),
            ("15:59", 102.3, 101.7, 102.0),
            ("16:00", 102.5, 101.9, 102.2),
        ]
    )

    result = evaluate_session_window(session, symbol="MES", or_window_minutes=5)

    assert result["eligible"] is True
    assert result["breakout_direction"] == "up"
    assert result["same_direction"] is True
    assert result["failed_direction"] is False
    assert result["close_reference_exact_1600"] is True
    assert result["close_extension"] == pytest.approx(1.2)


def test_evaluate_session_window_down_breakout_failure() -> None:
    session = _build_session(
        [
            ("09:30", 100.0, 99.0, 99.6),
            ("09:31", 100.3, 99.2, 99.9),
            ("09:32", 100.4, 99.4, 99.8),
            ("09:33", 100.2, 99.3, 99.7),
            ("09:34", 100.1, 99.1, 99.5),
            ("09:35", 99.0, 98.4, 98.5),
            ("10:15", 101.2, 100.8, 101.0),
            ("15:59", 100.0, 99.7, 99.8),
            ("16:00", 100.0, 99.7, 99.8),
        ]
    )

    result = evaluate_session_window(session, symbol="MES", or_window_minutes=5)

    assert result["eligible"] is True
    assert result["breakout_direction"] == "down"
    assert result["same_direction"] is False
    assert result["failed_direction"] is True
    assert result["close_extension"] == pytest.approx(-0.8)


def test_evaluate_session_window_uses_close_proxy_without_leakage() -> None:
    session = _build_session(
        [
            ("09:30", 100.0, 99.0, 99.5),
            ("09:31", 100.2, 99.1, 99.7),
            ("09:32", 100.4, 99.2, 99.8),
            ("09:33", 100.3, 99.3, 99.9),
            ("09:34", 100.1, 99.4, 99.8),
            ("15:59", 101.5, 100.8, 101.2),
        ]
    )

    result = evaluate_session_window(session, symbol="MES", or_window_minutes=5)

    assert result["eligible"] is True
    assert result["close_reference_exact_1600"] is False
    assert result["breakout_direction"] == "no_breakout"
    assert pd.isna(result["same_direction"])


def test_summarize_breakout_results_counts_and_rates() -> None:
    day_level = pd.DataFrame(
        [
            {
                "asset": "MES",
                "session_date": pd.Timestamp("2024-01-02"),
                "or_window_minutes": 5,
                "eligible": True,
                "breakout_direction": "up",
                "same_direction": True,
                "failed_direction": False,
                "close_extension": 1.0,
            },
            {
                "asset": "MES",
                "session_date": pd.Timestamp("2024-01-03"),
                "or_window_minutes": 5,
                "eligible": True,
                "breakout_direction": "down",
                "same_direction": False,
                "failed_direction": True,
                "close_extension": -0.5,
            },
            {
                "asset": "MES",
                "session_date": pd.Timestamp("2024-01-04"),
                "or_window_minutes": 5,
                "eligible": True,
                "breakout_direction": "no_breakout",
                "same_direction": pd.NA,
                "failed_direction": pd.NA,
                "close_extension": pd.NA,
            },
        ]
    )

    summary = summarize_breakout_results(day_level)
    row = summary.iloc[0]

    assert row["n_days"] == 3
    assert row["n_no_breakout"] == 1
    assert row["n_breakout_up"] == 1
    assert row["n_breakout_down"] == 1
    assert row["n_valid_breakouts"] == 2
    assert row["n_same_direction"] == 1
    assert row["n_failed_direction"] == 1
    assert row["pct_no_breakout"] == pytest.approx(1 / 3)
    assert row["hit_rate"] == pytest.approx(0.5)
    assert row["hit_rate_up"] == pytest.approx(1.0)
    assert row["hit_rate_down"] == pytest.approx(0.0)
    assert row["avg_close_extension"] == pytest.approx(0.25)
    assert row["median_close_extension"] == pytest.approx(0.25)
