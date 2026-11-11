from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.event_flow_studies import _event_matrix, _london_fix_events, _mnq_daily_events
from src.data.treasury_auction_calendar import normalize_auction_records


def test_treasury_calendar_keeps_exact_deadline_and_does_not_invent_result_time() -> None:
    records = [
        {
            "auction_date": "2024-06-12",
            "original_security_term": "10-Year",
            "security_term": "9-Year 11-Month",
            "security_type": "Note",
            "inflation_index_security": "No",
            "closing_time_comp": "01:00 PM",
            "closing_time_noncomp": "12:00 PM",
            "announcemt_date": "2024-06-06",
            "issue_date": "2024-06-17",
            "cusip": "TEST",
            "reopening": "Yes",
            "pdf_filenm_announcemt": "a.pdf",
            "pdf_filenm_comp_results": "r.pdf",
        },
        {
            "auction_date": "2024-06-20",
            "original_security_term": "5-Year",
            "security_term": "4-Year 10-Month",
            "security_type": "TIPS Note",
            "inflation_index_security": "Yes",
            "closing_time_comp": "01:00 PM",
        },
    ]
    result = normalize_auction_records(records)
    assert len(result) == 1
    assert result.loc[0, "tenor"] == "10-Year"
    assert result.loc[0, "bid_deadline_ts_utc"] == pd.Timestamp("2024-06-12 17:00:00+00:00")
    assert pd.isna(result.loc[0, "result_ts_utc"])
    assert result.loc[0, "result_time_status"] == "not_available_from_fiscaldata_api"


def test_mnq_signal_uses_1529_close_and_executes_at_1530_open() -> None:
    index = pd.date_range("2024-01-02 09:30", "2024-01-03 15:59", freq="min", tz="America/New_York")
    index = index[(index.time >= pd.Timestamp("09:30").time()) & (index.time <= pd.Timestamp("15:59").time())]
    frame = pd.DataFrame({"open": 100.0, "close": 100.0, "volume": 1}, index=index)
    frame.loc[pd.Timestamp("2024-01-02 15:59", tz="America/New_York"), "close"] = 100.0
    frame.loc[pd.Timestamp("2024-01-03 15:29", tz="America/New_York"), "close"] = 102.0
    frame.loc[pd.Timestamp("2024-01-03 15:30", tz="America/New_York"), "open"] = 103.0
    frame.loc[pd.Timestamp("2024-01-03 15:59", tz="America/New_York"), "close"] = 104.0
    events = _mnq_daily_events(frame)
    assert len(events) == 1
    assert np.isclose(events.loc[0, "rod_return"], np.log(102.0 / 100.0))
    assert np.isclose(events.loc[0, "last30_return"], np.log(104.0 / 103.0))


def test_london_fix_uses_local_timezone_through_dst_mismatch() -> None:
    index = pd.date_range("2024-03-25", "2024-04-05", freq="h", tz="UTC")
    frame = pd.DataFrame({"close": 1.0}, index=index)
    events = _london_fix_events(frame)
    before = events.loc[pd.to_datetime(events["fix_date_local"]).eq(pd.Timestamp("2024-03-29"))].iloc[0]
    after = events.loc[pd.to_datetime(events["fix_date_local"]).eq(pd.Timestamp("2024-04-01"))].iloc[0]
    assert before["fix_ts_utc"] == pd.Timestamp("2024-03-29 16:00:00+00:00")
    assert after["fix_ts_utc"] == pd.Timestamp("2024-04-01 15:00:00+00:00")
    assert before["london_fix_time_new_york"] == "12:00"
    assert after["london_fix_time_new_york"] == "11:00"


def test_event_matrix_preserves_missing_bars_as_nan() -> None:
    index = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02 12:59", tz="UTC"), pd.Timestamp("2024-01-02 13:01", tz="UTC")]
    )
    frame = pd.DataFrame({"close": [100.0, 101.0]}, index=index)
    events = pd.DataFrame({"event_ts": [pd.Timestamp("2024-01-02 13:00", tz="UTC")]})
    matrix, details = _event_matrix(frame, events, "event_ts", -1, 1)
    assert np.isnan(matrix.loc[0, 0])
    assert details.loc[0, "available_path_bars"] == 2
