import pandas as pd

from src.strategy.orb import ORBStrategy


def test_orb_strategy_uses_opening_time_for_or_expiry() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 08:58:00",
            "2024-01-02 08:59:00",
            "2024-01-02 09:00:00",
            "2024-01-02 09:01:00",
            "2024-01-02 09:02:00",
            "2024-01-02 09:03:00",
            "2024-01-02 09:04:00",
            "2024-01-02 09:05:00",
        ]
    )
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "or_high": [100.0] * len(timestamps),
            "or_low": [95.0] * len(timestamps),
            "high": [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 101.0, 101.0],
            "low": [96.0] * len(timestamps),
        }
    )

    strategy = ORBStrategy(or_minutes=5, opening_time="09:00:00")
    out = strategy.generate_signals(df)

    assert out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:04:00"), "signal"].iat[0] == 0
    assert out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:05:00"), "signal"].iat[0] == 1
