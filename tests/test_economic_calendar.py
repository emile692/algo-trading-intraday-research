import datetime as dt

import pandas as pd

from src.data.economic_calendar import (
    build_calendar_datasets,
    build_daily_calendar_features,
    load_calendar_daily_features,
    load_calendar_events,
    merge_calendar_features_on_daily_pnl,
    normalize_event_name,
    tag_intraday_bars_with_calendar_flags,
)
from src.data.economic_calendar.clean_calendar import clean_calendar_dataframe


def _build_raw_calendar_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_name": "FOMC Meeting Minutes",
                "country": "United States",
                "date": "2026-01-07",
                "time": "14:00",
                "impact": "Medium",
                "actual": None,
                "forecast": None,
                "previous": None,
            },
            {
                "event_name": "Consumer Price Index YoY",
                "country": "United States",
                "date": "2026-01-14",
                "time": "08:30",
                "impact": "High",
                "actual": "2.9%",
                "forecast": "2.8%",
                "previous": "2.7%",
            },
            {
                "event_name": "Core Consumer Price Index MoM",
                "country": "US",
                "date": "2026-01-14",
                "time": "08:30",
                "impact": "High",
                "actual": "0.3%",
                "forecast": "0.3%",
                "previous": "0.2%",
            },
            {
                "event_name": "Non Farm Payrolls",
                "country": "USD",
                "date": "2026-02-06",
                "time": "08:30",
                "impact": "High",
                "actual": "195K",
                "forecast": "175K",
                "previous": "165K",
            },
            {
                "event_name": "FOMC Rate Decision",
                "country": "United States",
                "date": "2026-03-18",
                "time": "14:00",
                "impact": "High",
                "actual": None,
                "forecast": None,
                "previous": None,
            },
            {
                "event_name": "Fed Chair Powell Speaks",
                "country": "United States",
                "date": "2026-03-20",
                "time": "13:00",
                "impact": "High",
                "actual": None,
                "forecast": None,
                "previous": None,
            },
            {
                "event_name": "GDP QoQ",
                "country": "United States",
                "date": "2026-03-26",
                "time": "",
                "impact": "Medium",
                "actual": "2.0%",
                "forecast": "2.1%",
                "previous": "2.2%",
            },
            {
                "event_name": "Consumer Price Index YoY",
                "country": "Eurozone",
                "date": "2026-01-14",
                "time": "05:00",
                "impact": "High",
                "actual": "2.0%",
                "forecast": "2.1%",
                "previous": "2.2%",
            },
        ]
    )


def _get_row(df: pd.DataFrame, date_value: str) -> pd.Series:
    target_date = pd.Timestamp(date_value).date()
    return df.loc[df["trade_date"] == target_date].iloc[0]


def test_normalize_event_name_aliases() -> None:
    assert normalize_event_name("Consumer Price Index YoY") == "CPI"
    assert normalize_event_name("Core Consumer Price Index MoM") == "CORE_CPI"
    assert normalize_event_name("Non Farm Payrolls") == "NFP"
    assert normalize_event_name("FOMC Meeting Minutes") == "FOMC_MINUTES"
    assert normalize_event_name("Fed Chair Powell Speaks") == "POWELL_SPEECH"


def test_clean_calendar_dataframe_handles_timezone_and_missing_times() -> None:
    cleaned = clean_calendar_dataframe(_build_raw_calendar_frame())

    assert "Eurozone" not in cleaned["country"].tolist()
    assert cleaned["event_type"].tolist().count("CPI") == 1

    fomc_row = cleaned.loc[cleaned["event_type"] == "FOMC"].iloc[0]
    assert fomc_row["event_ts_local"] == pd.Timestamp("2026-03-18 14:00:00", tz="America/New_York")
    assert fomc_row["event_ts_utc"] == pd.Timestamp("2026-03-18 18:00:00+00:00")
    assert fomc_row["event_date_local"] == dt.date(2026, 3, 18)
    assert fomc_row["event_time_local"] == "14:00:00"

    gdp_row = cleaned.loc[cleaned["event_type"] == "GDP"].iloc[0]
    assert not gdp_row["has_precise_time"]
    assert pd.isna(gdp_row["event_ts_local"])
    assert gdp_row["event_date_local"] == dt.date(2026, 3, 26)
    assert pd.isna(gdp_row["event_time_local"])


def test_build_daily_calendar_features_sets_expected_flags() -> None:
    cleaned = clean_calendar_dataframe(_build_raw_calendar_frame())
    daily = build_daily_calendar_features(cleaned)

    cpi_day = _get_row(daily, "2026-01-14")
    assert cpi_day["is_cpi_day"]
    assert cpi_day["is_core_cpi_day"]
    assert cpi_day["is_high_impact_macro_day"]
    assert cpi_day["nb_high_impact_events"] == 2
    assert cpi_day["first_event_time_local"] == "08:30:00"
    assert cpi_day["has_pre_930_event"]
    assert not cpi_day["has_rth_event"]

    nfp_day = _get_row(daily, "2026-02-06")
    assert nfp_day["is_nfp_day"]

    fomc_day = _get_row(daily, "2026-03-18")
    assert fomc_day["is_fomc_day"]
    assert fomc_day["has_rth_event"]
    assert not fomc_day["has_pre_930_event"]

    gdp_day = _get_row(daily, "2026-03-26")
    assert gdp_day["is_gdp_day"]
    assert pd.isna(gdp_day["first_event_time_local"])
    assert not gdp_day["has_pre_930_event"]
    assert not gdp_day["has_rth_event"]
    assert not gdp_day["has_post_1600_event"]


def test_merge_calendar_features_on_daily_pnl_left_joins_defaults() -> None:
    cleaned = clean_calendar_dataframe(_build_raw_calendar_frame())
    daily = build_daily_calendar_features(cleaned)
    pnl = pd.DataFrame(
        {
            "date": ["2026-01-14", "2026-01-16", "2026-03-18"],
            "net_pnl": [150.0, -20.0, 75.0],
        }
    )

    merged = merge_calendar_features_on_daily_pnl(pnl, daily, date_col="date")

    no_event_row = merged.loc[merged["date"] == "2026-01-16"].iloc[0]
    assert not no_event_row["is_fomc_day"]
    assert not no_event_row["is_cpi_day"]
    assert no_event_row["nb_high_impact_events"] == 0

    fomc_row = merged.loc[merged["date"] == "2026-03-18"].iloc[0]
    assert fomc_row["is_fomc_day"]
    assert fomc_row["is_high_impact_macro_day"]


def test_tag_intraday_bars_with_calendar_flags() -> None:
    cleaned = clean_calendar_dataframe(_build_raw_calendar_frame())
    intraday = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-18 13:30:00",
                    "2026-03-18 14:01:00",
                    "2026-03-19 09:30:00",
                ]
            ).tz_localize("America/New_York"),
            "close": [100.0, 101.0, 102.0],
        }
    )

    tagged = tag_intraday_bars_with_calendar_flags(intraday, cleaned)

    before_event = tagged.iloc[0]
    assert before_event["is_event_day"]
    assert before_event["has_high_impact_event_today"]
    assert before_event["next_event_type"] == "FOMC"
    assert before_event["minutes_to_next_high_impact_event"] == 30.0

    after_event = tagged.iloc[1]
    assert after_event["is_event_day"]
    assert after_event["minutes_since_last_high_impact_event"] == 1.0
    assert after_event["next_event_type"] == "POWELL_SPEECH"

    normal_day = tagged.iloc[2]
    assert not normal_day["is_event_day"]
    assert not normal_day["has_high_impact_event_today"]
    assert normal_day["next_event_type"] == "POWELL_SPEECH"


def test_build_calendar_datasets_writes_csv_outputs(tmp_path) -> None:
    raw_path = tmp_path / "economic_calendar_raw.csv"
    _build_raw_calendar_frame().to_csv(raw_path, index=False)

    events_df, daily_df = build_calendar_datasets(raw_path=raw_path, output_dir=tmp_path)
    loaded_events = load_calendar_events(tmp_path / "economic_calendar_events.csv")
    loaded_daily = load_calendar_daily_features(tmp_path / "economic_calendar_daily_features.csv")

    assert not events_df.empty
    assert not daily_df.empty
    assert len(loaded_events) == len(events_df)
    assert len(loaded_daily) == len(daily_df)
    assert "event_ts_local" in loaded_events.columns
    assert loaded_events.columns.tolist().count("source_timezone") == 1
    assert "is_fomc_day" in loaded_daily.columns
