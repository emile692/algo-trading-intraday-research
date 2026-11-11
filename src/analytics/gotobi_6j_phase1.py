"""Frozen calendar/time conventions for the 6J Gotobi Phase-1 study.

This module deliberately contains no strategy selection.  It centralizes the
quote inversion, Japanese banking calendar, timezone mapping, CME session date,
and Topstep tradability rules that must pass tests before market data are used.
"""

from __future__ import annotations

import calendar
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


JST = ZoneInfo("Asia/Tokyo")
CHICAGO = ZoneInfo("America/Chicago")
UTC = ZoneInfo("UTC")
GOTOBI_ANCHORS = (5, 10, 15, 20, 25, 30)
FIX_TIME_JST = dt.time(9, 55)


def usd_jpy_return_from_6j(start_6j: float, end_6j: float) -> float:
    """Return of USD/JPY implied by 6J, which is quoted USD per JPY."""
    if start_6j <= 0 or end_6j <= 0:
        raise ValueError("6J prices must be positive")
    return start_6j / end_6j - 1.0


def load_official_japan_holidays(path: str | Path) -> set[dt.date]:
    """Load the Cabinet Office CSV (CP932, first column is the holiday date)."""
    frame = pd.read_csv(path, encoding="cp932")
    dates = pd.to_datetime(frame.iloc[:, 0], errors="raise").dt.date
    return set(dates)


def japan_bank_holidays(official_holidays: set[dt.date], start_year: int, end_year: int) -> set[dt.date]:
    """Add Bank of Japan's recurring Dec-31 through Jan-3 closures."""
    holidays = set(official_holidays)
    for year in range(start_year - 1, end_year + 2):
        holidays.add(dt.date(year, 12, 31))
        for day in (1, 2, 3):
            holidays.add(dt.date(year, 1, day))
    return holidays


def is_japan_business_day(day: dt.date, holidays: set[dt.date]) -> bool:
    return day.weekday() < 5 and day not in holidays


def previous_japan_business_day(day: dt.date, holidays: set[dt.date]) -> dt.date:
    result = day
    while not is_japan_business_day(result, holidays):
        result -= dt.timedelta(days=1)
    return result


def last_japan_business_day(year: int, month: int, holidays: set[dt.date]) -> dt.date:
    final_day = calendar.monthrange(year, month)[1]
    return previous_japan_business_day(dt.date(year, month, final_day), holidays)


def jst_timestamp(day: dt.date, clock: dt.time) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(day, clock, tzinfo=JST))


def fixing_timestamps(japan_date: dt.date) -> dict[str, pd.Timestamp | dt.date]:
    jst = jst_timestamp(japan_date, FIX_TIME_JST)
    chicago = jst.tz_convert(CHICAGO)
    return {
        "japan_date": japan_date,
        "fix_jst": jst,
        "fix_utc": jst.tz_convert(UTC),
        "fix_chicago": chicago,
        # The Globex session opened at 17:00 CT on the prior calendar day and
        # carries the next business/session date.
        "cme_session_date": japan_date,
    }


def topstep_session_bounds(cme_session_date: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    open_ct = pd.Timestamp(
        dt.datetime.combine(cme_session_date - dt.timedelta(days=1), dt.time(17, 0), tzinfo=CHICAGO)
    )
    close_ct = pd.Timestamp(
        dt.datetime.combine(cme_session_date, dt.time(15, 10), tzinfo=CHICAGO)
    )
    return open_ct, close_ct


def topstep_window_is_structurally_tradable(
    start: pd.Timestamp, end: pd.Timestamp, cme_session_date: dt.date
) -> bool:
    open_ct, close_ct = topstep_session_bounds(cme_session_date)
    start_ct = start.tz_convert(CHICAGO)
    end_ct = end.tz_convert(CHICAGO)
    return open_ct <= start_ct < end_ct <= close_ct


def fixed_event_windows(japan_date: dt.date) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    fix = jst_timestamp(japan_date, FIX_TIME_JST)
    session_open_ct, _ = topstep_session_bounds(japan_date)
    points = {
        "0930": jst_timestamp(japan_date, dt.time(9, 30)),
        "0950": jst_timestamp(japan_date, dt.time(9, 50)),
        "0955": fix,
        "1000": jst_timestamp(japan_date, dt.time(10, 0)),
        "1015": jst_timestamp(japan_date, dt.time(10, 15)),
        "1030": jst_timestamp(japan_date, dt.time(10, 30)),
        "1100": jst_timestamp(japan_date, dt.time(11, 0)),
    }
    return {
        "reopen_to_0930": (session_open_ct.tz_convert(JST), points["0930"]),
        "0930_0950": (points["0930"], points["0950"]),
        "0950_0955": (points["0950"], points["0955"]),
        "0955_1000": (points["0955"], points["1000"]),
        "1000_1015": (points["1000"], points["1015"]),
        "1000_1030": (points["1000"], points["1030"]),
        "1000_1100": (points["1000"], points["1100"]),
    }


def build_japan_business_calendar(
    start: dt.date, end: dt.date, holidays: set[dt.date]
) -> pd.DataFrame:
    """One row per Japanese banking day with Gotobi/EOM flags."""
    anchors_by_date: dict[dt.date, list[int]] = {}
    cursor = dt.date(start.year, start.month, 1)
    last_month = dt.date(end.year, end.month, 1)
    while cursor <= last_month:
        year, month = cursor.year, cursor.month
        max_day = calendar.monthrange(year, month)[1]
        for anchor in GOTOBI_ANCHORS:
            # February has no day 30: EOM is studied separately and no Gotobi
            # observation is invented.
            if anchor > max_day:
                continue
            base = dt.date(year, month, anchor)
            adjusted = previous_japan_business_day(base, holidays)
            anchors_by_date.setdefault(adjusted, []).append(anchor)
        cursor = dt.date(year + (month == 12), 1 if month == 12 else month + 1, 1)

    rows = []
    for ts in pd.date_range(start, end, freq="D"):
        day = ts.date()
        if not is_japan_business_day(day, holidays):
            continue
        anchors = tuple(sorted(anchors_by_date.get(day, [])))
        eom = day == last_japan_business_day(day.year, day.month, holidays)
        times = fixing_timestamps(day)
        windows = fixed_event_windows(day)
        rows.append(
            {
                "japan_date": day,
                "weekday": day.strftime("%A"),
                "is_gotobi": bool(anchors),
                "gotobi_anchor_days": ";".join(map(str, anchors)),
                "is_eom": eom,
                "is_gotobi_ex_eom": bool(anchors) and not eom,
                "fix_jst": times["fix_jst"],
                "fix_utc": times["fix_utc"],
                "fix_chicago": times["fix_chicago"],
                "cme_session_date": times["cme_session_date"],
                "all_requested_windows_structurally_topstep_tradable": all(
                    topstep_window_is_structurally_tradable(a, b, day)
                    for a, b in windows.values()
                ),
            }
        )
    return pd.DataFrame(rows)
