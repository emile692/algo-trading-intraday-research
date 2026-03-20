import pandas as pd

from src.strategy.orb_paper import ORBPaperExactStrategy


def test_paper_strategy_signals_on_first_bullish_candle() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:35:00",
            "2024-01-02 09:40:00",
        ]
    )
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "open": [100.0, 101.0, 101.5],
            "high": [101.5, 102.0, 102.5],
            "low": [99.5, 100.5, 101.0],
            "close": [101.0, 101.5, 102.0],
        }
    )

    out = ORBPaperExactStrategy(opening_time="09:30:00").generate_signals(df)

    assert out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:30:00"), "signal"].iat[0] == 1
    assert out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:35:00"), "signal"].iat[0] == 0


def test_paper_strategy_skips_doji_first_candle() -> None:
    timestamps = pd.to_datetime(["2024-01-02 09:30:00", "2024-01-02 09:35:00"])
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
        }
    )

    out = ORBPaperExactStrategy(opening_time="09:30:00").generate_signals(df)

    assert out["signal"].sum() == 0
