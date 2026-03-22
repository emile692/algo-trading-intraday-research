"""OHLCV data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.settings import DEFAULT_TIMEZONE


def _normalize_ohlcv_frame(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Normalize columns and timestamps on a dataframe already loaded from disk."""
    out = df.copy()
    out.columns = [col.strip().lower() for col in out.columns]

    if "timestamp" not in out.columns and "ts_event" not in out.columns:
        raise ValueError("Missing required 'timestamp' or 'ts_event' column")

    timestamp_col = "timestamp" if "timestamp" in out.columns else "ts_event"
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    if out[timestamp_col].isna().any():
        raise ValueError(f"Found unparsable {timestamp_col} values")

    if timestamp_col != "timestamp":
        out = out.rename(columns={timestamp_col: "timestamp"})

    if out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(timezone)
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert(timezone)

    return out


def load_ohlcv_csv(path: Path | str, timezone: str = DEFAULT_TIMEZONE) -> pd.DataFrame:
    """Load OHLCV csv with timestamp parsing and normalized lowercase columns.

    Assumptions:
    - timestamps in source csv are US Eastern local times
    - timestamp marks beginning of bar
    - intraday bars may have missing minutes due to zero-volume bars not present
    """
    csv_path = Path(path)
    return _normalize_ohlcv_frame(pd.read_csv(csv_path), timezone=timezone)


def load_ohlcv_file(path: Path | str, timezone: str = DEFAULT_TIMEZONE) -> pd.DataFrame:
    """Load a supported OHLCV file and normalize timestamps/column names."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return load_ohlcv_csv(file_path, timezone=timezone)
    if suffix == ".parquet":
        df = pd.read_parquet(file_path)
        if df.index.name == 'timestamp' or 'timestamp' not in df.columns:
            df = df.reset_index()
        return _normalize_ohlcv_frame(df, timezone=timezone)

    raise ValueError(f"Unsupported OHLCV file format: {file_path.suffix}")
