from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.resampling import build_resampled_output_path, resample_ohlcv


def test_resample_ohlcv_preserves_processed_metadata_and_aggregates_bars() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02 09:30:00",
                    "2024-01-02 09:31:00",
                    "2024-01-02 09:32:00",
                    "2024-01-02 09:33:00",
                    "2024-01-02 09:34:00",
                ],
                utc=True,
            ),
            "rtype": [1, 1, 1, 1, 1],
            "publisher_id": [2, 2, 2, 2, 2],
            "instrument_id": [3, 3, 3, 3, 3],
            "symbol": ["MNQ", "MNQ", "MNQ", "MNQ", "MNQ"],
            "open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "high": [11.0, 12.5, 13.0, 13.5, 15.5],
            "low": [9.5, 10.5, 11.5, 12.5, 13.5],
            "close": [10.5, 12.0, 12.5, 13.0, 15.0],
            "volume": [100, 200, 300, 400, 500],
        }
    )

    out = resample_ohlcv(df, rule="5min")

    assert len(out) == 1
    row = out.iloc[0]
    assert float(row["open"]) == 10.0
    assert float(row["high"]) == 15.5
    assert float(row["low"]) == 9.5
    assert float(row["close"]) == 15.0
    assert float(row["volume"]) == 1500.0
    assert row["symbol"] == "MNQ"
    assert int(row["instrument_id"]) == 3


def test_build_resampled_output_path_rewrites_timeframe_segment() -> None:
    path = Path("data/processed/parquet/MNQ_c_0_1m_20260321_094501.parquet")

    out = build_resampled_output_path(path, rule="5min")

    assert out.name == "MNQ_c_0_5m_20260321_094501.parquet"


def test_resample_ohlcv_supports_aggregation_overrides_for_auxiliary_columns() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02 09:30:00",
                    "2024-01-02 09:31:00",
                    "2024-01-02 09:32:00",
                    "2024-01-02 09:33:00",
                    "2024-01-02 09:34:00",
                ],
                utc=True,
            ),
            "open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "high": [11.0, 12.5, 13.0, 13.5, 15.5],
            "low": [9.5, 10.5, 11.5, 12.5, 13.5],
            "close": [10.5, 12.0, 12.5, 13.0, 15.0],
            "volume": [100, 200, 300, 400, 500],
            "vwap_pv_typical": [1000.0, 2200.0, 3600.0, 5200.0, 7000.0],
        }
    )

    out = resample_ohlcv(df, rule="5min", aggregation_overrides={"vwap_pv_typical": "sum"})

    assert len(out) == 1
    assert float(out.iloc[0]["vwap_pv_typical"]) == 19000.0
