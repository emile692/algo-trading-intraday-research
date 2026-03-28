from __future__ import annotations

from datetime import date

import pandas as pd

from src.features.implied_volatility import build_vix_vvix_daily_features, load_daily_index_series


def test_load_daily_index_series_supports_standard_and_yfinance_layouts(tmp_path) -> None:
    standard_path = tmp_path / "vix.csv"
    standard_path.write_text(
        "\n".join(
            [
                "DATE,OPEN,HIGH,LOW,CLOSE",
                "01/02/2024,10,11,9,10.5",
                "01/03/2024,11,12,10,11.5",
            ]
        ),
        encoding="utf-8",
    )

    yfinance_path = tmp_path / "vvix.csv"
    yfinance_path.write_text(
        "\n".join(
            [
                "Price,Close,High,Low,Open,Volume",
                "Ticker,^VVIX,^VVIX,^VVIX,^VVIX,^VVIX",
                "Date,,,,,",
                "2024-01-02,100,101,99,100,0",
                "2024-01-03,110,111,109,110,0",
            ]
        ),
        encoding="utf-8",
    )

    standard = load_daily_index_series(standard_path)
    yfinance = load_daily_index_series(yfinance_path)

    assert standard["source_date"].dt.date.tolist() == [date(2024, 1, 2), date(2024, 1, 3)]
    assert standard["close"].tolist() == [10.5, 11.5]
    assert yfinance["source_date"].dt.date.tolist() == [date(2024, 1, 2), date(2024, 1, 3)]
    assert yfinance["close"].tolist() == [100.0, 110.0]


def test_build_vix_vvix_daily_features_is_strictly_lagged() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    vix = pd.DataFrame({"source_date": dates, "close": [10.0, 30.0, 20.0, 40.0, 50.0]})
    vvix = pd.DataFrame({"source_date": dates, "close": [100.0, 120.0, 110.0, 130.0, 140.0]})

    features = build_vix_vvix_daily_features(vix, vvix, percentile_windows=(2,))

    row = features.loc[features["session_date"] == date(2024, 1, 4)].iloc[0]
    assert row["vix_reference_date_t1"] == date(2024, 1, 3)
    assert float(row["vix_level_t1"]) == 30.0
    assert float(row["vvix_level_t1"]) == 120.0
    assert float(row["vvix_over_vix_t1"]) == 4.0

    later_row = features.loc[features["session_date"] == date(2024, 1, 8)].iloc[0]
    assert later_row["vix_reference_date_t1"] == date(2024, 1, 5)
    assert float(later_row["vix_level_t1"]) == 40.0
    assert float(later_row["vix_pct_2_t1"]) == 1.0
