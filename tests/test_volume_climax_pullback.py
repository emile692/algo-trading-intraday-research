from __future__ import annotations

import pandas as pd

from src.engine.execution_model import ExecutionModel
from src.engine.volume_climax_pullback_backtester import run_volume_climax_pullback_backtest
from src.engine.vwap_backtester import InstrumentDetails
from src.strategy.volume_climax_pullback import VolumeClimaxPullbackVariant, build_signal_frame, prepare_volume_climax_features


def _sample_df() -> pd.DataFrame:
    rows = []
    ts = pd.Timestamp("2024-01-02 09:30:00", tz="America/New_York")
    for i in range(40):
        o = 100 + i * 0.1
        c = o + (0.4 if i % 2 == 0 else -0.2)
        h = max(o, c) + 0.2
        l = min(o, c) - 0.2
        v = 100 + i
        if i == 25:
            c = o + 1.2
            h = c + 0.3
            l = o - 0.1
            v = 5_000
        rows.append({"timestamp": ts + pd.Timedelta(minutes=5 * i), "open": o, "high": h, "low": l, "close": c, "volume": v})
    return pd.DataFrame(rows)


def test_signal_is_next_bar_leak_free_directional():
    df = _sample_df()
    feat = prepare_volume_climax_features(df)
    variant = VolumeClimaxPullbackVariant(
        name="t", family="pure_climax", timeframe="5m", volume_quantile=0.95, volume_lookback=20,
        min_body_fraction=None, min_range_atr=None, stretch_ref=None, min_stretch_atr=None, wick_fraction=None,
        stop_buffer_mode="0_tick", rr_target=1.0, time_stop_bars=2, session_overlay="all_rth",
    )
    sig = build_signal_frame(feat, variant)
    trigger_ts = df.iloc[26]["timestamp"]
    signal_row = sig.loc[sig["timestamp"] == trigger_ts].iloc[0]
    assert bool(signal_row["entry_short"])


def test_backtester_next_open_single_position_flat_eod():
    df = _sample_df()
    feat = prepare_volume_climax_features(df)
    variant = VolumeClimaxPullbackVariant(
        name="t", family="pure_climax", timeframe="5m", volume_quantile=0.95, volume_lookback=20,
        min_body_fraction=None, min_range_atr=None, stretch_ref=None, min_stretch_atr=None, wick_fraction=None,
        stop_buffer_mode="0_tick", rr_target=1.0, time_stop_bars=2, session_overlay="all_rth",
    )
    sig = build_signal_frame(feat, variant)
    instrument = InstrumentDetails(
        symbol="MNQ", asset_class="futures", tick_size=0.25, tick_value_usd=0.5, point_value_usd=2.0,
        commission_per_side_usd=1.25, slippage_ticks=1,
    )
    model = ExecutionModel(commission_per_side_usd=1.25, slippage_ticks=0, tick_size=0.25)
    trades = run_volume_climax_pullback_backtest(sig, variant, model, instrument).trades
    assert len(trades) >= 1
    assert (pd.to_datetime(trades["entry_time"]) <= pd.to_datetime(trades["exit_time"])).all()
