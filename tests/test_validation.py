import pandas as pd

from src.data.validation import validate_ohlcv


def test_validate_ohlcv_flags_invalid_ohlc() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 09:30", "2024-01-01 09:31"]),
            "open": [100, 100],
            "high": [101, 99],
            "low": [99, 98],
            "close": [100.5, 98.5],
            "volume": [10, 11],
        }
    )
    report = validate_ohlcv(df)
    assert report.invalid_ohlc_rows == 1
