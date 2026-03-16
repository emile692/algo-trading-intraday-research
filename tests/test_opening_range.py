from src.data.loader import load_ohlcv_csv
from src.features.opening_range import compute_opening_range
from src.config.paths import RAW_DATA_DIR


def test_compute_opening_range_columns() -> None:
    df = load_ohlcv_csv(RAW_DATA_DIR / "NQ_1min_sample.csv")
    out = compute_opening_range(df, or_minutes=3)
    for col in ["or_high", "or_low", "or_width", "or_midpoint"]:
        assert col in out.columns
    assert out["or_width"].dropna().iloc[0] > 0
