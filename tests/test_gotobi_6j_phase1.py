import datetime as dt

import pandas as pd

from src.analytics.gotobi_6j_phase1 import (
    CHICAGO,
    JST,
    build_japan_business_calendar,
    fixed_event_windows,
    fixing_timestamps,
    japan_bank_holidays,
    topstep_window_is_structurally_tradable,
    usd_jpy_return_from_6j,
)


def test_6j_quote_is_inverse_of_usd_jpy() -> None:
    assert usd_jpy_return_from_6j(0.0067, 0.0066) > 0  # 6J down -> USD/JPY up
    assert usd_jpy_return_from_6j(0.0066, 0.0067) < 0  # 6J up -> USD/JPY down


def test_tokyo_has_no_dst_but_chicago_does() -> None:
    winter = fixing_timestamps(dt.date(2024, 1, 10))
    summer = fixing_timestamps(dt.date(2024, 7, 10))
    assert winter["fix_jst"].utcoffset() == dt.timedelta(hours=9)
    assert summer["fix_jst"].utcoffset() == dt.timedelta(hours=9)
    assert winter["fix_chicago"] == pd.Timestamp("2024-01-09 18:55", tz=CHICAGO)
    assert summer["fix_chicago"] == pd.Timestamp("2024-07-09 19:55", tz=CHICAGO)


def test_japanese_date_maps_to_cme_session_containing_fix() -> None:
    monday = fixing_timestamps(dt.date(2024, 7, 8))
    assert monday["fix_chicago"] == pd.Timestamp("2024-07-07 19:55", tz=CHICAGO)
    assert monday["cme_session_date"] == dt.date(2024, 7, 8)


def test_all_requested_windows_are_structurally_topstep_tradable() -> None:
    for day in (dt.date(2024, 1, 10), dt.date(2024, 7, 10)):
        for start, end in fixed_event_windows(day).values():
            assert topstep_window_is_structurally_tradable(start, end, day)
            assert start.tz_convert(JST).date() == day


def test_gotobi_rolls_to_prior_japanese_bank_day_and_february_has_no_fake_30() -> None:
    official = {
        dt.date(2024, 5, 3),
        dt.date(2024, 5, 6),
        dt.date(2024, 2, 23),
    }
    holidays = japan_bank_holidays(official, 2024, 2024)
    frame = build_japan_business_calendar(dt.date(2024, 2, 1), dt.date(2024, 5, 31), holidays)
    may2 = frame.loc[frame.japan_date == dt.date(2024, 5, 2)].iloc[0]
    assert may2.is_gotobi
    assert "5" in may2.gotobi_anchor_days
    feb29 = frame.loc[frame.japan_date == dt.date(2024, 2, 29)].iloc[0]
    assert feb29.is_eom
    assert "30" not in feb29.gotobi_anchor_days


def test_boj_year_end_closure_is_used() -> None:
    holidays = japan_bank_holidays(set(), 2024, 2025)
    frame = build_japan_business_calendar(dt.date(2024, 12, 20), dt.date(2025, 1, 10), holidays)
    assert dt.date(2024, 12, 30) in set(frame.japan_date)
    assert dt.date(2024, 12, 31) not in set(frame.japan_date)
    assert dt.date(2025, 1, 2) not in set(frame.japan_date)
