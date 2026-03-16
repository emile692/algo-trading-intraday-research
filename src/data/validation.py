"""Data quality validation checks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.schema import required_columns_for_dataset


@dataclass
class QualityReport:
    """Simple quality report container."""

    rows: int
    missing_required_columns: list[str]
    is_chronological: bool
    duplicate_timestamps: int
    invalid_ohlc_rows: int
    negative_volume_rows: int

    @property
    def is_valid(self) -> bool:
        return (
            not self.missing_required_columns
            and self.is_chronological
            and self.duplicate_timestamps == 0
            and self.invalid_ohlc_rows == 0
            and self.negative_volume_rows == 0
        )


def validate_ohlcv(df: pd.DataFrame) -> QualityReport:
    """Run structural and content checks for OHLCV datasets."""
    required = required_columns_for_dataset(df.columns.tolist())
    missing = [c for c in required if c not in df.columns]

    if missing:
        return QualityReport(
            rows=len(df),
            missing_required_columns=missing,
            is_chronological=False,
            duplicate_timestamps=0,
            invalid_ohlc_rows=0,
            negative_volume_rows=0,
        )

    chronological = df["timestamp"].is_monotonic_increasing
    duplicate_ts = int(df["timestamp"].duplicated().sum())

    invalid_ohlc = (
        (df["high"] < df["low"])
        | (df["open"] > df["high"])
        | (df["open"] < df["low"])
        | (df["close"] > df["high"])
        | (df["close"] < df["low"])
    )

    negative_volume = df["volume"] < 0

    return QualityReport(
        rows=len(df),
        missing_required_columns=missing,
        is_chronological=bool(chronological),
        duplicate_timestamps=duplicate_ts,
        invalid_ohlc_rows=int(invalid_ohlc.sum()),
        negative_volume_rows=int(negative_volume.sum()),
    )
