"""Economic calendar ingestion and feature-building utilities."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "CANONICAL_EVENT_SPECS",
    "DEFAULT_RAW_CALENDAR_PATH",
    "build_calendar_datasets",
    "build_daily_calendar_features",
    "clean_calendar_csv",
    "clean_calendar_dataframe",
    "fetch_calendar_from_api",
    "get_calendar_source_dataframe",
    "load_calendar_daily_features",
    "load_calendar_events",
    "load_raw_calendar_csv",
    "merge_calendar_features_on_daily_pnl",
    "normalize_event_name",
    "tag_intraday_bars_with_calendar_flags",
]

_LAZY_EXPORTS = {
    "CANONICAL_EVENT_SPECS": ("src.data.economic_calendar.clean_calendar", "CANONICAL_EVENT_SPECS"),
    "DEFAULT_RAW_CALENDAR_PATH": ("src.data.economic_calendar.fetch_calendar", "DEFAULT_RAW_CALENDAR_PATH"),
    "build_calendar_datasets": (
        "src.data.economic_calendar.build_event_features",
        "build_calendar_datasets",
    ),
    "build_daily_calendar_features": (
        "src.data.economic_calendar.build_event_features",
        "build_daily_calendar_features",
    ),
    "clean_calendar_csv": ("src.data.economic_calendar.clean_calendar", "clean_calendar_csv"),
    "clean_calendar_dataframe": ("src.data.economic_calendar.clean_calendar", "clean_calendar_dataframe"),
    "fetch_calendar_from_api": ("src.data.economic_calendar.fetch_calendar", "fetch_calendar_from_api"),
    "get_calendar_source_dataframe": (
        "src.data.economic_calendar.fetch_calendar",
        "get_calendar_source_dataframe",
    ),
    "load_calendar_daily_features": (
        "src.data.economic_calendar.build_event_features",
        "load_calendar_daily_features",
    ),
    "load_calendar_events": ("src.data.economic_calendar.build_event_features", "load_calendar_events"),
    "load_raw_calendar_csv": ("src.data.economic_calendar.fetch_calendar", "load_raw_calendar_csv"),
    "merge_calendar_features_on_daily_pnl": (
        "src.data.economic_calendar.build_event_features",
        "merge_calendar_features_on_daily_pnl",
    ),
    "normalize_event_name": ("src.data.economic_calendar.clean_calendar", "normalize_event_name"),
    "tag_intraday_bars_with_calendar_flags": (
        "src.data.economic_calendar.build_event_features",
        "tag_intraday_bars_with_calendar_flags",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
