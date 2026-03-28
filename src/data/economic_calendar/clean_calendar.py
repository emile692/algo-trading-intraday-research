"""Cleaning and normalization for research-grade economic calendar data."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import DEFAULT_TIMEZONE
from src.data.economic_calendar.fetch_calendar import (
    DEFAULT_RAW_CALENDAR_PATH,
    coerce_raw_calendar_schema,
    load_raw_calendar_csv,
)

HIGH_IMPACT_THRESHOLD = 3


@dataclass(frozen=True)
class EventSpec:
    """Canonical metadata for a supported macro event."""

    event_group: str
    default_importance_score: int
    aliases: tuple[str, ...]


CANONICAL_EVENT_SPECS: dict[str, EventSpec] = {
    "FOMC_MINUTES": EventSpec(
        event_group="fed",
        default_importance_score=2,
        aliases=(
            r"\bfomc\b.*\bminutes\b",
            r"\bfederal reserve\b.*\bminutes\b",
            r"\bmeeting minutes\b",
        ),
    ),
    "FOMC": EventSpec(
        event_group="fed",
        default_importance_score=3,
        aliases=(
            r"\bfomc\b.*\brate\b.*\bdecision\b",
            r"\bfed\b.*\brate\b.*\bdecision\b",
            r"\binterest rate decision\b",
            r"\bfederal funds rate\b",
        ),
    ),
    "POWELL_SPEECH": EventSpec(
        event_group="fed",
        default_importance_score=3,
        aliases=(
            r"(powell|chair powell|fed chair|federal reserve chair).*(speech|speaks|remarks|testifies|press conference)",
            r"(speech|speaks|remarks|testifies|press conference).*(powell|chair powell|fed chair|federal reserve chair)",
        ),
    ),
    "CORE_CPI": EventSpec(
        event_group="inflation",
        default_importance_score=3,
        aliases=(
            r"\bcore\b.*\bcpi\b",
            r"\bcore consumer price index\b",
        ),
    ),
    "CPI": EventSpec(
        event_group="inflation",
        default_importance_score=3,
        aliases=(
            r"\bcpi\b",
            r"\bconsumer price index\b",
        ),
    ),
    "NFP": EventSpec(
        event_group="labor",
        default_importance_score=3,
        aliases=(
            r"\bnon[\s-]?farm payrolls\b",
            r"\bnonfarm payrolls\b",
            r"\bnonfarm employment change\b",
        ),
    ),
    "CORE_PCE": EventSpec(
        event_group="inflation",
        default_importance_score=3,
        aliases=(
            r"\bcore\b.*\bpce\b",
            r"\bcore personal consumption expenditures\b",
        ),
    ),
    "PPI": EventSpec(
        event_group="inflation",
        default_importance_score=2,
        aliases=(
            r"\bppi\b",
            r"\bproducer price index\b",
        ),
    ),
    "ISM_MANUFACTURING": EventSpec(
        event_group="activity",
        default_importance_score=2,
        aliases=(
            r"\bism\b.*\bmanufacturing\b",
            r"\bmanufacturing pmi\b",
        ),
    ),
    "ISM_SERVICES": EventSpec(
        event_group="activity",
        default_importance_score=2,
        aliases=(
            r"\bism\b.*\bservices\b",
            r"\bservices pmi\b",
            r"\bnon-manufacturing pmi\b",
        ),
    ),
    "RETAIL_SALES": EventSpec(
        event_group="activity",
        default_importance_score=2,
        aliases=(r"\bretail sales\b",),
    ),
    "GDP": EventSpec(
        event_group="growth",
        default_importance_score=2,
        aliases=(
            r"\bgdp\b",
            r"\bgross domestic product\b",
        ),
    ),
    "ADP_NONFARM_EMPLOYMENT": EventSpec(
        event_group="labor",
        default_importance_score=2,
        aliases=(
            r"\badp\b.*\bemployment\b",
            r"\badp\b.*\bnonfarm\b",
        ),
    ),
    "INITIAL_JOBLESS_CLAIMS": EventSpec(
        event_group="labor",
        default_importance_score=1,
        aliases=(
            r"\binitial jobless claims\b",
            r"\bjobless claims\b",
        ),
    ),
    "UMICH_SENTIMENT": EventSpec(
        event_group="sentiment",
        default_importance_score=1,
        aliases=(
            r"\buniversity of michigan\b.*\bsentiment\b",
            r"\bumich\b.*\bsentiment\b",
            r"\bconsumer sentiment\b",
        ),
    ),
}

SUPPORTED_EVENT_TYPES = tuple(CANONICAL_EVENT_SPECS.keys())
US_COUNTRY_ALIASES = {
    "united states",
    "united states of america",
    "us",
    "u.s.",
    "usa",
    "u.s.a.",
    "usd",
}
MISSING_TIME_TOKENS = {
    "",
    "nan",
    "nat",
    "na",
    "n/a",
    "none",
    "tbd",
    "all day",
    "tentative",
}


def normalize_event_name(event_name: Any) -> str | None:
    """Map a raw event name into the supported canonical event universe."""
    if event_name is None or pd.isna(event_name):
        return None

    normalized = str(event_name).strip().lower()
    normalized = re.sub(r"[^0-9a-zA-Z]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None

    for event_type, spec in CANONICAL_EVENT_SPECS.items():
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in spec.aliases):
            return event_type
    return None


def normalize_country(country: Any, currency: Any = None) -> str | None:
    """Return 'US' for US events, else None."""
    candidates = []
    for value in (country, currency):
        if value is None or pd.isna(value):
            continue
        normalized = str(value).strip().lower()
        normalized = re.sub(r"[^a-z.]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if normalized:
            candidates.append(normalized)

    if any(candidate in US_COUNTRY_ALIASES for candidate in candidates):
        return "US"
    return None


def normalize_impact_label(impact: Any) -> str:
    """Map source impact labels into a compact normalized representation."""
    if impact is None or pd.isna(impact):
        return "Unknown"

    text = str(impact).strip()
    if not text:
        return "Unknown"

    lowered = text.lower()
    if set(lowered) <= {"!", "*"}:
        if len(lowered) >= 3:
            return "High"
        if len(lowered) == 2:
            return "Medium"
        return "Low"
    if "high" in lowered:
        return "High"
    if "medium" in lowered or "med" in lowered:
        return "Medium"
    if "low" in lowered:
        return "Low"
    if lowered in {"3", "3.0"}:
        return "High"
    if lowered in {"2", "2.0"}:
        return "Medium"
    if lowered in {"1", "1.0"}:
        return "Low"
    return text.strip().title()


def impact_to_score(impact: Any) -> int | None:
    """Convert normalized impact labels into a 1-3 score when possible."""
    label = normalize_impact_label(impact)
    mapping = {"Low": 1, "Medium": 2, "High": 3}
    return mapping.get(label)


def _clean_observation_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _parse_date_value(value: Any) -> dt.date | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _parse_timestamp_value(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.Timestamp(pd.to_datetime(value, errors="coerce"))
    if pd.isna(parsed):
        return None
    return parsed


def _parse_time_value(value: Any) -> dt.time | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    lowered = re.sub(r"\s+", " ", text.lower()).strip()
    if lowered in MISSING_TIME_TOKENS or any(token in lowered for token in ("tentative", "all day")):
        return None

    cleaned = text.upper().strip()
    cleaned = cleaned.replace(".", ":")
    cleaned = re.sub(r"\b(ET|EST|EDT|UTC|GMT)\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if re.fullmatch(r"\d{4}", cleaned):
        cleaned = f"{cleaned[:2]}:{cleaned[2:]}"
    elif re.fullmatch(r"\d{1,2}", cleaned):
        cleaned = f"{int(cleaned):02d}:00"

    formats = (
        "%H:%M",
        "%H:%M:%S",
        "%I:%M %p",
        "%I:%M:%S %p",
        "%I %p",
        "%I%p",
        "%I:%M%p",
        "%H%M",
    )
    for time_format in formats:
        try:
            return dt.datetime.strptime(cleaned, time_format).time().replace(microsecond=0)
        except ValueError:
            continue

    parsed = pd.to_datetime(cleaned, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.time().replace(microsecond=0)


def _localize_naive_timestamp(naive_timestamp: dt.datetime, timezone: str) -> pd.Timestamp | pd.NaT:
    localized = pd.DatetimeIndex([pd.Timestamp(naive_timestamp)]).tz_localize(
        timezone,
        ambiguous="NaT",
        nonexistent="NaT",
    )[0]
    return localized


def _build_timestamp_fields(
    row: pd.Series,
    default_source_timezone: str,
    local_timezone: str,
) -> dict[str, Any]:
    row_source_timezone = _clean_observation_value(row.get("source_timezone")) or default_source_timezone

    parsed_date = _parse_date_value(row.get("date"))
    parsed_time = _parse_time_value(row.get("time"))
    raw_timestamp = _parse_timestamp_value(row.get("event_timestamp")) or _parse_timestamp_value(row.get("timestamp"))

    if raw_timestamp is not None and parsed_date is None:
        if raw_timestamp.tzinfo is None:
            timestamp_in_source_tz = _localize_naive_timestamp(raw_timestamp.to_pydatetime(), row_source_timezone)
        else:
            timestamp_in_source_tz = raw_timestamp.tz_convert(row_source_timezone)
        if not pd.isna(timestamp_in_source_tz):
            parsed_date = timestamp_in_source_tz.date()
            if parsed_time is None:
                parsed_time = timestamp_in_source_tz.timetz().replace(tzinfo=None)
    elif raw_timestamp is not None and parsed_time is None:
        timestamp_in_source_tz = (
            _localize_naive_timestamp(raw_timestamp.to_pydatetime(), row_source_timezone)
            if raw_timestamp.tzinfo is None
            else raw_timestamp.tz_convert(row_source_timezone)
        )
        if not pd.isna(timestamp_in_source_tz):
            parsed_time = timestamp_in_source_tz.timetz().replace(tzinfo=None)

    if parsed_date is None:
        return {
            "source_timezone": row_source_timezone,
            "event_ts_original": pd.NaT,
            "event_ts_local": pd.NaT,
            "event_ts_utc": pd.NaT,
            "event_date_local": pd.NaT,
            "event_time_local": pd.NA,
            "event_weekday_local": pd.NA,
            "has_precise_time": False,
        }

    if parsed_time is None:
        weekday = pd.Timestamp(parsed_date).day_name()
        return {
            "source_timezone": row_source_timezone,
            "event_ts_original": pd.NaT,
            "event_ts_local": pd.NaT,
            "event_ts_utc": pd.NaT,
            "event_date_local": parsed_date,
            "event_time_local": pd.NA,
            "event_weekday_local": weekday,
            "has_precise_time": False,
        }

    naive_original = dt.datetime.combine(parsed_date, parsed_time)
    source_timestamp = _localize_naive_timestamp(naive_original, row_source_timezone)
    if pd.isna(source_timestamp):
        weekday = pd.Timestamp(parsed_date).day_name()
        return {
            "source_timezone": row_source_timezone,
            "event_ts_original": pd.NaT,
            "event_ts_local": pd.NaT,
            "event_ts_utc": pd.NaT,
            "event_date_local": parsed_date,
            "event_time_local": pd.NA,
            "event_weekday_local": weekday,
            "has_precise_time": False,
        }

    local_timestamp = source_timestamp.tz_convert(local_timezone)
    return {
        "source_timezone": row_source_timezone,
        "event_ts_original": pd.Timestamp(naive_original),
        "event_ts_local": local_timestamp,
        "event_ts_utc": local_timestamp.tz_convert("UTC"),
        "event_date_local": local_timestamp.date(),
        "event_time_local": local_timestamp.strftime("%H:%M:%S"),
        "event_weekday_local": local_timestamp.day_name(),
        "has_precise_time": True,
    }


def _build_empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_name_raw",
            "event_type",
            "event_group",
            "country",
            "impact",
            "importance_score",
            "is_high_impact",
            "actual",
            "forecast",
            "previous",
            "source_timezone",
            "event_ts_original",
            "event_ts_local",
            "event_ts_utc",
            "event_date_local",
            "event_time_local",
            "event_weekday_local",
            "has_precise_time",
            "date_raw",
            "time_raw",
        ]
    )


def clean_calendar_dataframe(
    raw_df: pd.DataFrame,
    source_timezone: str = DEFAULT_TIMEZONE,
    local_timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Normalize raw calendar records into a research-ready event table."""
    out = coerce_raw_calendar_schema(raw_df)
    out["event_name_raw"] = out["event_name"].map(_clean_observation_value)
    out["country"] = [
        normalize_country(country, currency)
        for country, currency in zip(out["country"], out["currency"], strict=False)
    ]
    out["event_type"] = out["event_name_raw"].map(normalize_event_name)
    out = out[out["country"].eq("US") & out["event_type"].notna() & out["event_name_raw"].notna()].copy()
    if out.empty:
        return _build_empty_events_frame()

    out["impact"] = out["impact"].map(normalize_impact_label)
    out["actual"] = out["actual"].map(_clean_observation_value)
    out["forecast"] = out["forecast"].map(_clean_observation_value)
    out["previous"] = out["previous"].map(_clean_observation_value)
    out["event_group"] = out["event_type"].map(lambda event_type: CANONICAL_EVENT_SPECS[event_type].event_group)
    out["importance_score"] = [
        max(
            CANONICAL_EVENT_SPECS[event_type].default_importance_score,
            impact_to_score(impact) or CANONICAL_EVENT_SPECS[event_type].default_importance_score,
        )
        for event_type, impact in zip(out["event_type"], out["impact"], strict=False)
    ]
    out["is_high_impact"] = out["importance_score"] >= HIGH_IMPACT_THRESHOLD

    timestamp_fields = out.apply(
        lambda row: _build_timestamp_fields(
            row=row,
            default_source_timezone=source_timezone,
            local_timezone=local_timezone,
        ),
        axis=1,
        result_type="expand",
    )
    out = out.drop(columns=["source_timezone"])
    out = pd.concat([out.reset_index(drop=True), timestamp_fields.reset_index(drop=True)], axis=1)

    out["date_raw"] = out["date"].map(_clean_observation_value)
    out["time_raw"] = out["time"].map(_clean_observation_value)

    out = out[
        [
            "event_name_raw",
            "event_type",
            "event_group",
            "country",
            "impact",
            "importance_score",
            "is_high_impact",
            "actual",
            "forecast",
            "previous",
            "source_timezone",
            "event_ts_original",
            "event_ts_local",
            "event_ts_utc",
            "event_date_local",
            "event_time_local",
            "event_weekday_local",
            "has_precise_time",
            "date_raw",
            "time_raw",
        ]
    ].copy()

    out = out.drop_duplicates(
        subset=["event_name_raw", "event_type", "event_ts_original", "event_date_local"],
        keep="first",
    )

    sort_key = pd.to_datetime(out["event_date_local"], errors="coerce")
    time_key = out["event_time_local"].fillna("99:99:99")
    out = out.assign(_sort_date=sort_key, _sort_time=time_key)
    out = out.sort_values(["_sort_date", "_sort_time", "event_type", "event_name_raw"]).drop(
        columns=["_sort_date", "_sort_time"]
    )
    return out.reset_index(drop=True)


def clean_calendar_csv(
    raw_path: Path | str = DEFAULT_RAW_CALENDAR_PATH,
    source_timezone: str = DEFAULT_TIMEZONE,
    local_timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Load and clean a raw calendar CSV in one step."""
    return clean_calendar_dataframe(
        raw_df=load_raw_calendar_csv(raw_path),
        source_timezone=source_timezone,
        local_timezone=local_timezone,
    )
