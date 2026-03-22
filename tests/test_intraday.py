import pandas as pd

from src.features.intraday import add_session_vwap, add_continuous_session_vwap


def test_add_session_vwap_regular_date() -> None:
    timestamps = pd.to_datetime([
        "2026-03-18 09:30:00",
        "2026-03-18 09:31:00",
        "2026-03-18 09:32:00",
    ]).tz_localize("America/New_York")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [100, 200, 100],
        }
    )
    out = add_session_vwap(df)
    assert "session_vwap" in out.columns
    assert out.loc[2, "session_vwap"] > 0


def test_add_continuous_session_vwap_evening_rolls_to_next_day() -> None:
    timestamps = pd.to_datetime([
        "2026-03-18 17:59:00",
        "2026-03-18 18:00:00",
        "2026-03-18 18:01:00",
        "2026-03-19 09:30:00",
    ]).tz_localize("America/New_York")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100, 101, 102, 103],
            "high": [100, 101, 102, 103],
            "low": [99, 100, 101, 102],
            "close": [100, 101, 102, 103],
            "volume": [10, 10, 10, 10],
        }
    )
    out = add_continuous_session_vwap(df, session_start_hour=18)
    assert out.loc[0, "continuous_session_date"] == pd.Timestamp("2026-03-18").date()
    assert out.loc[1, "continuous_session_date"] == pd.Timestamp("2026-03-19").date()
    assert out.loc[2, "continuous_session_date"] == pd.Timestamp("2026-03-19").date()
    assert out.loc[3, "continuous_session_date"] == pd.Timestamp("2026-03-19").date()
    assert out.loc[1, "continuous_session_vwap"] == out.loc[2, "continuous_session_vwap"]


def test_add_continuous_session_vwap_returns_different_from_daily_when_overnight_included() -> None:
    timestamps = pd.to_datetime([
        "2026-03-18 17:59:00",
        "2026-03-18 18:00:00",
        "2026-03-19 09:30:00",
    ]).tz_localize("America/New_York")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100, 120, 130],
            "high": [100, 120, 130],
            "low": [99, 119, 129],
            "close": [100, 120, 130],
            "volume": [10, 10, 10],
        }
    )
    out_daily = add_session_vwap(df)
    out_cont = add_continuous_session_vwap(df, session_start_hour=18)
    assert out_daily.loc[1, "session_vwap"] != out_cont.loc[1, "continuous_session_vwap"]
