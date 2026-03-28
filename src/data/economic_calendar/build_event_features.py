"""Feature builders and IO helpers for economic calendar research datasets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.paths import PROCESSED_DATA_DIR
from src.config.settings import DEFAULT_TIMEZONE
from src.data.economic_calendar.clean_calendar import clean_calendar_dataframe
from src.data.economic_calendar.fetch_calendar import (
    DEFAULT_RAW_CALENDAR_PATH,
    get_calendar_source_dataframe,
)

DEFAULT_CALENDAR_OUTPUT_DIR = PROCESSED_DATA_DIR / "economic_calendar"
EVENTS_FILENAME = "economic_calendar_events.csv"
DAILY_FEATURES_FILENAME = "economic_calendar_daily_features.csv"

EVENT_FLAG_COLUMNS = {
    "FOMC": "is_fomc_day",
    "FOMC_MINUTES": "is_fomc_minutes_day",
    "POWELL_SPEECH": "is_powell_day",
    "CPI": "is_cpi_day",
    "CORE_CPI": "is_core_cpi_day",
    "NFP": "is_nfp_day",
    "CORE_PCE": "is_core_pce_day",
    "PPI": "is_ppi_day",
    "ISM_MANUFACTURING": "is_ism_manufacturing_day",
    "ISM_SERVICES": "is_ism_services_day",
    "RETAIL_SALES": "is_retail_sales_day",
    "GDP": "is_gdp_day",
}

DAILY_BOOLEAN_COLUMNS = list(EVENT_FLAG_COLUMNS.values()) + [
    "is_high_impact_macro_day",
    "has_pre_930_event",
    "has_rth_event",
    "has_post_1600_event",
]


def _build_empty_daily_features_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "trade_date",
            *EVENT_FLAG_COLUMNS.values(),
            "is_high_impact_macro_day",
            "nb_high_impact_events",
            "first_event_time_local",
            "has_pre_930_event",
            "has_rth_event",
            "has_post_1600_event",
        ]
    )


def _has_timezone_suffix(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    text = non_null.astype(str)
    return bool(text.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True).any())


def _coerce_local_timestamp_series(series: pd.Series, timezone: str = DEFAULT_TIMEZONE) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert(timezone)
    if pd.api.types.is_datetime64_ns_dtype(series):
        return series.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")

    if _has_timezone_suffix(series):
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        return parsed.dt.tz_convert(timezone)

    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")


def _coerce_utc_timestamp_series(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert("UTC")
    if pd.api.types.is_datetime64_ns_dtype(series):
        return series.dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert("UTC")


def _coerce_date_series(values: pd.Series | pd.Index | list[Any]) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert(DEFAULT_TIMEZONE).dt.date
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.date


def _coerce_bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    if series.empty:
        return series.astype(bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)

    def _to_bool(value: Any) -> bool:
        if value is None or pd.isna(value):
            return default
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        return text in {"true", "1", "yes", "y"}

    return series.map(_to_bool).astype(bool)


def _timedelta_to_clock_string(value: Any) -> str | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    total_seconds = int(pd.Timedelta(value).total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _ensure_event_frame_types(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return events_df.copy()

    out = events_df.copy()
    if "event_ts_original" in out.columns:
        out["event_ts_original"] = pd.to_datetime(out["event_ts_original"], errors="coerce")
    if "event_ts_local" in out.columns:
        out["event_ts_local"] = _coerce_local_timestamp_series(out["event_ts_local"], DEFAULT_TIMEZONE)
    if "event_ts_utc" in out.columns:
        out["event_ts_utc"] = _coerce_utc_timestamp_series(out["event_ts_utc"])

    if "event_date_local" in out.columns:
        out["event_date_local"] = _coerce_date_series(out["event_date_local"])
    elif "event_ts_local" in out.columns:
        out["event_date_local"] = out["event_ts_local"].dt.date
    else:
        raise ValueError("events_df must include 'event_date_local' or 'event_ts_local'.")

    if "event_time_local" not in out.columns:
        out["event_time_local"] = pd.NA
    if "event_ts_local" in out.columns:
        derived_times = out["event_ts_local"].dt.strftime("%H:%M:%S")
        out["event_time_local"] = out["event_time_local"].fillna(derived_times)

    if "has_precise_time" not in out.columns:
        out["has_precise_time"] = out["event_ts_local"].notna()
    out["has_precise_time"] = _coerce_bool_series(out["has_precise_time"])

    if "is_high_impact" not in out.columns:
        out["is_high_impact"] = False
    out["is_high_impact"] = _coerce_bool_series(out["is_high_impact"])

    return out


def build_daily_calendar_features(
    events_df: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate event-level data into one row per local date."""
    events = _ensure_event_frame_types(events_df)
    if events.empty and start_date is None and end_date is None:
        return _build_empty_daily_features_frame()

    if start_date is None:
        if events.empty:
            raise ValueError("start_date is required when building daily features from an empty events frame.")
        start = min(events["event_date_local"])
    else:
        start = _coerce_date_series(pd.Series([start_date])).iloc[0]

    if end_date is None:
        if events.empty:
            raise ValueError("end_date is required when building daily features from an empty events frame.")
        end = max(events["event_date_local"])
    else:
        end = _coerce_date_series(pd.Series([end_date])).iloc[0]

    if start is None or end is None:
        raise ValueError("Could not infer daily calendar date range.")
    if end < start:
        raise ValueError("end_date must be on or after start_date.")

    daily = pd.DataFrame({"trade_date": pd.date_range(start=start, end=end, freq="D").date})
    if events.empty:
        for column in DAILY_BOOLEAN_COLUMNS:
            daily[column] = False
        daily["nb_high_impact_events"] = 0
        daily["first_event_time_local"] = pd.NA
        return daily

    event_flags = (
        pd.crosstab(events["event_date_local"], events["event_type"])
        .reindex(columns=list(EVENT_FLAG_COLUMNS), fill_value=0)
        .gt(0)
        .rename(columns=EVENT_FLAG_COLUMNS)
        .reset_index()
        .rename(columns={"event_date_local": "trade_date"})
    )

    impact_agg = (
        events.groupby("event_date_local")
        .agg(
            is_high_impact_macro_day=("is_high_impact", "any"),
            nb_high_impact_events=("is_high_impact", "sum"),
        )
        .reset_index()
        .rename(columns={"event_date_local": "trade_date"})
    )

    precise_events = events[events["has_precise_time"] & events["event_time_local"].notna()].copy()
    if precise_events.empty:
        time_agg = pd.DataFrame(
            columns=[
                "trade_date",
                "first_event_time_local",
                "has_pre_930_event",
                "has_rth_event",
                "has_post_1600_event",
            ]
        )
    else:
        precise_events["_event_time_td"] = pd.to_timedelta(precise_events["event_time_local"], errors="coerce")
        pre_930_cutoff = pd.Timedelta(hours=9, minutes=30)
        rth_end = pd.Timedelta(hours=16)
        time_agg = (
            precise_events.groupby("event_date_local")
            .agg(
                first_event_td=("_event_time_td", "min"),
                has_pre_930_event=("_event_time_td", lambda series: bool((series < pre_930_cutoff).any())),
                has_rth_event=(
                    "_event_time_td",
                    lambda series: bool(((series >= pre_930_cutoff) & (series <= rth_end)).any()),
                ),
                has_post_1600_event=("_event_time_td", lambda series: bool((series > rth_end).any())),
            )
            .reset_index()
            .rename(columns={"event_date_local": "trade_date"})
        )
        time_agg["first_event_time_local"] = time_agg["first_event_td"].map(_timedelta_to_clock_string)
        time_agg = time_agg.drop(columns=["first_event_td"])

    daily = daily.merge(event_flags, on="trade_date", how="left")
    daily = daily.merge(impact_agg, on="trade_date", how="left")
    daily = daily.merge(time_agg, on="trade_date", how="left")

    for column in EVENT_FLAG_COLUMNS.values():
        if column not in daily.columns:
            daily[column] = False
    for column in DAILY_BOOLEAN_COLUMNS:
        daily[column] = _coerce_bool_series(daily[column], default=False)
    daily["nb_high_impact_events"] = daily["nb_high_impact_events"].fillna(0).astype(int)
    if "first_event_time_local" not in daily.columns:
        daily["first_event_time_local"] = pd.NA

    ordered = [
        "trade_date",
        *EVENT_FLAG_COLUMNS.values(),
        "is_high_impact_macro_day",
        "nb_high_impact_events",
        "first_event_time_local",
        "has_pre_930_event",
        "has_rth_event",
        "has_post_1600_event",
    ]
    return daily[ordered].copy()


def save_calendar_outputs(
    events_df: pd.DataFrame,
    daily_features_df: pd.DataFrame,
    output_dir: Path | str = DEFAULT_CALENDAR_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Persist the event-level and daily calendar datasets to deterministic CSV files."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    events_path = target_dir / EVENTS_FILENAME
    daily_path = target_dir / DAILY_FEATURES_FILENAME

    events_df.to_csv(events_path, index=False)
    daily_features_df.to_csv(daily_path, index=False)
    return events_path, daily_path


def build_calendar_datasets(
    raw_path: Path | str = DEFAULT_RAW_CALENDAR_PATH,
    output_dir: Path | str = DEFAULT_CALENDAR_OUTPUT_DIR,
    api_url: str | None = None,
    api_key: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    source_timezone: str = DEFAULT_TIMEZONE,
    local_timezone: str = DEFAULT_TIMEZONE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full ingestion pipeline and write both research datasets."""
    raw_df = get_calendar_source_dataframe(
        raw_csv_path=raw_path,
        api_url=api_url,
        api_key=api_key,
        start_date=start_date,
        end_date=end_date,
    )
    events_df = clean_calendar_dataframe(
        raw_df=raw_df,
        source_timezone=source_timezone,
        local_timezone=local_timezone,
    )
    daily_features_df = build_daily_calendar_features(events_df, start_date=start_date, end_date=end_date)
    save_calendar_outputs(events_df, daily_features_df, output_dir=output_dir)
    return events_df, daily_features_df


def load_calendar_events(path: Path | str = DEFAULT_CALENDAR_OUTPUT_DIR / EVENTS_FILENAME) -> pd.DataFrame:
    """Load the event-level economic calendar dataset from disk."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    return _ensure_event_frame_types(df)


def load_calendar_daily_features(
    path: Path | str = DEFAULT_CALENDAR_OUTPUT_DIR / DAILY_FEATURES_FILENAME,
) -> pd.DataFrame:
    """Load the daily calendar feature dataset from disk."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    df["trade_date"] = _coerce_date_series(df["trade_date"])
    for column in DAILY_BOOLEAN_COLUMNS:
        if column in df.columns:
            df[column] = _coerce_bool_series(df[column], default=False)
    if "nb_high_impact_events" in df.columns:
        df["nb_high_impact_events"] = df["nb_high_impact_events"].fillna(0).astype(int)
    return df


def merge_calendar_features_on_daily_pnl(
    daily_df: pd.DataFrame,
    calendar_daily_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Left-join calendar daily features onto a daily PnL dataframe."""
    if date_col not in daily_df.columns:
        raise ValueError(f"Missing date column '{date_col}' in daily_df.")
    if "trade_date" not in calendar_daily_df.columns:
        raise ValueError("calendar_daily_df must include a 'trade_date' column.")

    out = daily_df.copy()
    calendar = calendar_daily_df.copy()
    out["_calendar_merge_date"] = _coerce_date_series(out[date_col])
    calendar["_calendar_merge_date"] = _coerce_date_series(calendar["trade_date"])
    calendar = calendar.drop(columns=["trade_date"])

    merged = out.merge(calendar, on="_calendar_merge_date", how="left")
    merged = merged.drop(columns=["_calendar_merge_date"])

    for column in DAILY_BOOLEAN_COLUMNS:
        if column in merged.columns:
            merged[column] = _coerce_bool_series(merged[column], default=False)
    if "nb_high_impact_events" in merged.columns:
        merged["nb_high_impact_events"] = merged["nb_high_impact_events"].fillna(0).astype(int)
    return merged


def tag_intraday_bars_with_calendar_flags(
    intraday_df: pd.DataFrame,
    calendar_events_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Attach event-day flags and next/last high-impact event timing to intraday bars."""
    if timestamp_col not in intraday_df.columns:
        raise ValueError(f"Missing timestamp column '{timestamp_col}' in intraday_df.")

    out = intraday_df.copy()
    out[timestamp_col] = _coerce_local_timestamp_series(out[timestamp_col], DEFAULT_TIMEZONE)
    out["trade_date"] = out[timestamp_col].dt.date

    events = _ensure_event_frame_types(calendar_events_df)
    event_dates = set(events["event_date_local"]) if not events.empty else set()
    start_date = out["trade_date"].min() if not out.empty else None
    end_date = out["trade_date"].max() if not out.empty else None

    if start_date is None or end_date is None:
        out["is_event_day"] = False
        out["has_high_impact_event_today"] = False
        out["minutes_to_next_high_impact_event"] = np.nan
        out["minutes_since_last_high_impact_event"] = np.nan
        out["next_event_type"] = pd.NA
        return out

    calendar_daily = build_daily_calendar_features(events, start_date=start_date, end_date=end_date)
    calendar_daily["has_high_impact_event_today"] = calendar_daily["is_high_impact_macro_day"]
    calendar_daily["is_event_day"] = calendar_daily["trade_date"].isin(event_dates)
    daily_lookup = calendar_daily[["trade_date", "is_event_day", "has_high_impact_event_today"]]

    out = out.merge(daily_lookup, on="trade_date", how="left")
    out["is_event_day"] = _coerce_bool_series(out["is_event_day"], default=False)
    out["has_high_impact_event_today"] = _coerce_bool_series(
        out["has_high_impact_event_today"],
        default=False,
    )

    precise_high_impact_events = events[
        events["is_high_impact"] & events["has_precise_time"] & events["event_ts_local"].notna()
    ].copy()
    if precise_high_impact_events.empty or out.empty:
        out["minutes_to_next_high_impact_event"] = np.nan
        out["minutes_since_last_high_impact_event"] = np.nan
        out["next_event_type"] = pd.NA
        return out

    precise_high_impact_events = precise_high_impact_events.sort_values("event_ts_local").reset_index(drop=True)
    event_ns = (
        precise_high_impact_events["event_ts_local"].dt.tz_convert("UTC").astype("int64").to_numpy()
    )
    bar_ns = out[timestamp_col].dt.tz_convert("UTC").astype("int64").to_numpy()
    event_types = precise_high_impact_events["event_type"].to_numpy(dtype=object)

    next_indices = np.searchsorted(event_ns, bar_ns, side="left")
    prev_indices = np.searchsorted(event_ns, bar_ns, side="right") - 1

    minutes_to = np.full(len(out), np.nan, dtype=float)
    minutes_since = np.full(len(out), np.nan, dtype=float)
    next_event_type = np.full(len(out), pd.NA, dtype=object)

    valid_next = next_indices < len(event_ns)
    valid_prev = prev_indices >= 0

    minutes_to[valid_next] = (event_ns[next_indices[valid_next]] - bar_ns[valid_next]) / 60_000_000_000.0
    minutes_since[valid_prev] = (bar_ns[valid_prev] - event_ns[prev_indices[valid_prev]]) / 60_000_000_000.0
    next_event_type[valid_next] = event_types[next_indices[valid_next]]

    out["minutes_to_next_high_impact_event"] = minutes_to
    out["minutes_since_last_high_impact_event"] = minutes_since
    out["next_event_type"] = next_event_type
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build economic calendar research datasets.")
    parser.add_argument("--raw-path", default=str(DEFAULT_RAW_CALENDAR_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_CALENDAR_OUTPUT_DIR))
    parser.add_argument("--api-url", default=os.getenv("ECON_CAL_API_URL"))
    parser.add_argument("--api-key-env-var", default="ECONOMIC_CALENDAR_API_KEY")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--source-timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--local-timezone", default=DEFAULT_TIMEZONE)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    api_key = os.getenv(args.api_key_env_var) if args.api_url else None

    events_df, daily_features_df = build_calendar_datasets(
        raw_path=args.raw_path,
        output_dir=args.output_dir,
        api_url=args.api_url,
        api_key=api_key,
        start_date=args.start_date,
        end_date=args.end_date,
        source_timezone=args.source_timezone,
        local_timezone=args.local_timezone,
    )
    output_dir = Path(args.output_dir)
    print(output_dir / EVENTS_FILENAME)
    print(output_dir / DAILY_FEATURES_FILENAME)
    print(len(events_df))
    print(len(daily_features_df))


if __name__ == "__main__":
    main()
