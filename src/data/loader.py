"""CSV data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.settings import DEFAULT_TIMEZONE


def load_ohlcv_csv(path: Path | str, timezone: str = DEFAULT_TIMEZONE) -> pd.DataFrame:
    """Load OHLCV csv with timestamp parsing and normalized lowercase columns.

    Assumptions:
    - timestamps in source csv are US Eastern local times
    - timestamp marks beginning of bar
    - intraday bars may have missing minutes due to zero-volume bars not present
    """
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    df.columns = [col.strip().lower() for col in df.columns]

    if "timestamp" not in df.columns:
        raise ValueError("Missing required 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Found unparsable timestamp values")

    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(timezone)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(timezone)

    return df
