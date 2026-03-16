"""Schema definitions and helpers for OHLCV data."""

from __future__ import annotations

INTRADAY_REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
DAILY_REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open interest",
]


def is_daily_schema(columns: list[str]) -> bool:
    """Return True if columns satisfy the expected daily schema."""
    normalized = {c.strip().lower() for c in columns}
    return set(DAILY_REQUIRED_COLUMNS).issubset(normalized)


def is_intraday_schema(columns: list[str]) -> bool:
    """Return True if columns satisfy the expected intraday schema."""
    normalized = {c.strip().lower() for c in columns}
    return set(INTRADAY_REQUIRED_COLUMNS).issubset(normalized)


def required_columns_for_dataset(columns: list[str]) -> list[str]:
    """Return required columns based on detected dataset type."""
    if is_daily_schema(columns):
        return DAILY_REQUIRED_COLUMNS
    return INTRADAY_REQUIRED_COLUMNS
