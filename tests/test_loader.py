import pandas as pd

from src.config.paths import RAW_DATA_DIR
from src.data.loader import load_ohlcv_csv, load_ohlcv_file


def test_load_ohlcv_csv_basic() -> None:
    df = load_ohlcv_csv(RAW_DATA_DIR / "NQ_1min_sample.csv")
    assert "timestamp" in df.columns
    assert df["timestamp"].dt.tz is not None
    assert all(col == col.lower() for col in df.columns)


def test_load_ohlcv_file_normalizes_parquet_ts_event(tmp_path) -> None:
    parquet_path = tmp_path / "sample.parquet"
    pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2024-01-02 14:30:00+00:00", "2024-01-02 14:31:00+00:00"]),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [10, 11],
        }
    ).to_parquet(parquet_path)

    df = load_ohlcv_file(parquet_path)

    assert "timestamp" in df.columns
    assert "ts_event" not in df.columns
    assert df["timestamp"].dt.tz is not None
