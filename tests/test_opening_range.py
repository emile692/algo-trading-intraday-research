import pandas as pd

from src.data.loader import load_ohlcv_csv
from src.config.paths import DOWNLOADED_DATA_DIR
from src.features.opening_range import compute_opening_range


def test_compute_opening_range_columns() -> None:
    df = load_ohlcv_csv(DOWNLOADED_DATA_DIR / "NQ_1min_sample.csv")
    out = compute_opening_range(df, or_minutes=3)
    for col in ["or_high", "or_low", "or_width", "or_midpoint"]:
        assert col in out.columns
    assert out["or_width"].dropna().iloc[0] > 0


def test_compute_opening_range_respects_configured_opening_time() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 08:58:00",
            "2024-01-02 08:59:00",
            "2024-01-02 09:00:00",
            "2024-01-02 09:01:00",
            "2024-01-02 09:02:00",
        ]
    )
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "high": [110.0, 111.0, 101.0, 103.0, 120.0],
            "low": [100.0, 99.0, 95.0, 96.0, 94.0],
        }
    )

    out = compute_opening_range(df, or_minutes=2, opening_time="09:00:00")
    first_row = out.iloc[0]

    assert first_row["or_high"] == 103.0
    assert first_row["or_low"] == 95.0
    assert first_row["or_width"] == 8.0
