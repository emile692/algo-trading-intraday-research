from src.data.loader import load_ohlcv_csv
from src.config.paths import RAW_DATA_DIR


def test_load_ohlcv_csv_basic() -> None:
    df = load_ohlcv_csv(RAW_DATA_DIR / "NQ_1min_sample.csv")
    assert "timestamp" in df.columns
    assert df["timestamp"].dt.tz is not None
    assert all(col == col.lower() for col in df.columns)
