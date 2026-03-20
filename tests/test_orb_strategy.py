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


def test_orb_strategy_can_filter_and_wait_for_later_breakout() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
            "2024-01-02 09:33:00",
        ]
    )
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "or_high": [100.0] * len(timestamps),
            "or_low": [95.0] * len(timestamps),
            "high": [99.0, 101.0, 101.5, 100.0],
            "low": [96.0, 96.0, 96.0, 96.0],
            "close": [99.0, 99.0, 101.2, 99.5],
            "volume": [100, 100, 100, 100],
            "atr_14": [1.0, 0.5, 1.2, 1.1],
            "session_vwap": [98.0, 100.5, 100.0, 99.8],
            "ema_20": [98.5, 100.5, 100.2, 99.9],
        }
    )

    strategy = ORBStrategy(
        or_minutes=1,
        opening_time="09:30:00",
        atr_period=14,
        atr_regime="band_2",
        atr_min=1.0,
        atr_max=2.0,
        direction_filter_mode="vwap_and_ema",
        ema_length=20,
    )
    out = strategy.generate_signals(df)

    first_break = out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:31:00")].iloc[0]
    second_break = out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 09:32:00")].iloc[0]

    assert first_break["raw_signal"] == 1
    assert first_break["signal"] == 0
    assert bool(first_break["filtered_out"]) is True
    assert second_break["raw_signal"] == 1
    assert second_break["signal"] == 1
