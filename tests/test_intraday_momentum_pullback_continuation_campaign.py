from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics.intraday_momentum_pullback_continuation_campaign import IMPCCampaignSpec, run_campaign
from src.data.resampling import resample_ohlcv
from src.engine.execution_model import ExecutionModel
from src.engine.intraday_momentum_pullback_continuation_backtester import (
    run_intraday_momentum_pullback_continuation_backtest,
)
from src.engine.vwap_backtester import resolve_instrument_details
from src.strategy.intraday_momentum_pullback_continuation import (
    IMPCVariantConfig,
    build_impc_signal_frame,
    prepare_impc_feature_frame,
)


def _five_minute_frame(periods: int = 10) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:30:00", periods=periods, freq="5min", tz="America/New_York")
    closes = [100.0, 100.3, 100.7, 100.4, 100.9, 101.2, 100.8, 101.3, 101.1, 101.5][:periods]
    rows: list[dict[str, float | pd.Timestamp]] = []
    previous_close = 99.9
    for idx, timestamp in enumerate(timestamps):
        close = closes[idx]
        open_price = previous_close
        high = max(open_price, close) + 0.12
        low = min(open_price, close) - 0.12
        rows.append(
            {
                "timestamp": timestamp,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": 100.0 + idx,
            }
        )
        previous_close = close
    return pd.DataFrame(rows)


def _manual_feature_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:45:00", periods=3, freq="5min", tz="America/New_York")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.date,
            "open": [100.0, 100.4, 101.2],
            "high": [100.6, 101.3, 101.6],
            "low": [99.8, 100.2, 100.9],
            "close": [100.3, 101.1, 101.4],
            "session_vwap": [100.1, 100.5, 100.8],
            "prev_high": [pd.NA, 100.6, 101.3],
            "prev_low": [pd.NA, 99.8, 100.2],
            "atr_48": [1.0, 1.0, 1.0],
            "ema_8": [100.2, 100.9, 101.0],
            "ema_21": [100.0, 100.3, 100.5],
            "ema_slope_8_3": [0.2, 0.3, 0.3],
            "ema_spread_8_21": [0.2, 0.6, 0.5],
            "pullback_high_3": [pd.NA, 101.0, 101.1],
            "pullback_low_3": [pd.NA, 100.4, 100.5],
            "pullback_depth_3": [pd.NA, 0.6, 0.6],
            "pullback_high_pos_3": [pd.NA, 0.0, 0.0],
            "pullback_low_pos_3": [pd.NA, 2.0, 2.0],
            "is_last_bar_of_session": [False, False, True],
        }
    )


def _write_synthetic_asset_dataset(path: Path, *, sessions: int = 30, asset_bias: float = 0.0) -> None:
    rows: list[dict[str, object]] = []
    session_dates = pd.bdate_range("2024-01-02", periods=sessions)

    for day_idx, session_date in enumerate(session_dates):
        session_open = pd.Timestamp(session_date.date()).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        base = 100.0 + day_idx * 0.06 + asset_bias
        previous_close = base

        for minute_idx in range(390):
            timestamp = session_open + pd.Timedelta(minutes=minute_idx)
            phase = minute_idx % 90
            if phase < 25:
                close = base + phase * 0.010
            elif phase < 45:
                close = base + 0.25 - (phase - 25) * 0.008
            elif phase < 70:
                close = base + 0.09 + (phase - 45) * 0.012
            else:
                close = previous_close + 0.002

            open_price = previous_close
            high = max(open_price, close) + 0.02
            low = min(open_price, close) - 0.02
            rows.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 100 + day_idx,
                }
            )
            previous_close = close

    pd.DataFrame(rows).to_parquet(path, index=False)


def test_resample_1m_to_5m_is_correct_for_impc_inputs() -> None:
    one_minute = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30:00", periods=5, freq="1min", tz="America/New_York"),
            "open": [100.0, 100.1, 100.2, 100.3, 100.4],
            "high": [100.2, 100.3, 100.5, 100.4, 100.8],
            "low": [99.9, 100.0, 100.1, 100.2, 100.3],
            "close": [100.1, 100.2, 100.3, 100.4, 100.7],
            "volume": [10, 11, 12, 13, 14],
        }
    )

    out = resample_ohlcv(one_minute, rule="5min")

    assert len(out) == 1
    row = out.iloc[0]
    assert float(row["open"]) == 100.0
    assert float(row["high"]) == 100.8
    assert float(row["low"]) == 99.9
    assert float(row["close"]) == 100.7
    assert float(row["volume"]) == 60.0


def test_feature_pipeline_is_strictly_no_lookahead() -> None:
    frame = _five_minute_frame(periods=10)
    variant = IMPCVariantConfig(
        name="impc_test_small",
        ema_fast=8,
        ema_slow=21,
        slope_lookback=3,
        pullback_lookback=3,
        pb_min_atr=0.3,
        target_r=2.0,
    )

    short = prepare_impc_feature_frame(frame.iloc[:8].copy(), slope_lookbacks=(3,), pullback_lookbacks=(3,))
    full = prepare_impc_feature_frame(frame.copy(), slope_lookbacks=(3,), pullback_lookbacks=(3,))

    short_signal = build_impc_signal_frame(short, variant)
    full_signal = build_impc_signal_frame(full, variant).iloc[: len(short_signal)].reset_index(drop=True)

    pd.testing.assert_series_equal(short_signal["pullback_depth_3"], full_signal["pullback_depth_3"], check_names=False)
    pd.testing.assert_series_equal(short_signal["raw_signal"], full_signal["raw_signal"], check_names=False)
    pd.testing.assert_series_equal(short_signal["signal"], full_signal["signal"], check_names=False)


def test_signal_requires_bias_and_enters_only_next_open() -> None:
    feature_df = _manual_feature_frame()
    variant = IMPCVariantConfig(
        name="impc_gate",
        ema_fast=8,
        ema_slow=21,
        slope_lookback=3,
        pullback_lookback=3,
        pb_min_atr=0.3,
        target_r=2.0,
    )

    valid = build_impc_signal_frame(feature_df, variant)
    invalid_bias = build_impc_signal_frame(feature_df.assign(session_vwap=[100.1, 101.2, 100.8]), variant)

    assert int(valid.loc[1, "raw_signal"]) == 1
    assert bool(valid.loc[1, "entry_long"]) is False
    assert bool(valid.loc[2, "entry_long"]) is True
    assert int(valid.loc[2, "signal"]) == 1
    assert int(invalid_bias.loc[1, "raw_signal"]) == 0
    assert int(invalid_bias.loc[2, "signal"]) == 0


def test_signal_requires_valid_pullback_structure() -> None:
    feature_df = _manual_feature_frame()
    variant = IMPCVariantConfig(
        name="impc_pullback_gate",
        ema_fast=8,
        ema_slow=21,
        slope_lookback=3,
        pullback_lookback=3,
        pb_min_atr=0.3,
        target_r=2.0,
    )

    invalid_pullback = build_impc_signal_frame(feature_df.assign(pullback_low_3=[pd.NA, 100.0, 100.0]), variant)

    assert invalid_pullback["raw_signal"].tolist() == [0, 0, 0]
    assert invalid_pullback["signal"].tolist() == [0, 0, 0]


def test_backtester_applies_structural_stop_and_target_levels() -> None:
    timestamps = pd.date_range("2024-01-02 09:55:00", periods=3, freq="5min", tz="America/New_York")
    signal_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.date,
            "open": [100.0, 100.4, 100.6],
            "high": [102.4, 100.8, 100.9],
            "low": [99.8, 100.1, 100.3],
            "close": [101.9, 100.5, 100.7],
            "entry_long": [True, False, False],
            "entry_short": [False, False, False],
            "entry_stop_reference_long": [99.0, float("nan"), float("nan")],
            "entry_stop_reference_short": [float("nan"), float("nan"), float("nan")],
            "entry_target_r": [2.0, float("nan"), float("nan")],
            "entry_signal_time": [timestamps[0] - pd.Timedelta(minutes=5), pd.NaT, pd.NaT],
            "entry_signal_atr": [1.0, float("nan"), float("nan")],
            "entry_signal_pullback_depth": [0.6, float("nan"), float("nan")],
            "entry_signal_ema_spread": [0.5, float("nan"), float("nan")],
            "is_last_bar_of_session": [False, False, True],
        }
    )
    variant = IMPCVariantConfig(
        name="impc_backtest_target",
        ema_fast=8,
        ema_slow=21,
        slope_lookback=3,
        pullback_lookback=3,
        pb_min_atr=0.3,
        target_r=2.0,
    )
    execution_model = ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0.0, tick_size=0.25)
    instrument = resolve_instrument_details("MNQ")

    result = run_intraday_momentum_pullback_continuation_backtest(signal_df, variant, execution_model, instrument)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "target"
    assert float(trade["entry_price"]) == 100.0
    assert float(trade["stop_price"]) == 99.0
    assert float(trade["target_price"]) == 102.0
    assert float(trade["holding_minutes"]) == 5.0


def test_backtester_forces_time_stop_after_twelve_bars() -> None:
    timestamps = pd.date_range("2024-01-02 09:55:00", periods=12, freq="5min", tz="America/New_York")
    signal_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.date,
            "open": [100.0] + [100.1] * 11,
            "high": [100.6] * 12,
            "low": [99.6] * 12,
            "close": [100.1] * 12,
            "entry_long": [True] + [False] * 11,
            "entry_short": [False] * 12,
            "entry_stop_reference_long": [98.0] + [float("nan")] * 11,
            "entry_stop_reference_short": [float("nan")] * 12,
            "entry_target_r": [2.0] + [float("nan")] * 11,
            "entry_signal_time": [timestamps[0] - pd.Timedelta(minutes=5)] + [pd.NaT] * 11,
            "entry_signal_atr": [2.0] + [float("nan")] * 11,
            "entry_signal_pullback_depth": [0.8] + [float("nan")] * 11,
            "entry_signal_ema_spread": [0.4] + [float("nan")] * 11,
            "is_last_bar_of_session": [False] * 11 + [True],
        }
    )
    variant = IMPCVariantConfig(
        name="impc_backtest_time",
        ema_fast=8,
        ema_slow=21,
        slope_lookback=3,
        pullback_lookback=3,
        pb_min_atr=0.3,
        target_r=2.0,
    )
    execution_model = ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0.0, tick_size=0.25)
    instrument = resolve_instrument_details("MNQ")

    result = run_intraday_momentum_pullback_continuation_backtest(signal_df, variant, execution_model, instrument)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "time_stop"
    assert float(trade["holding_minutes"]) == 60.0
    assert int(trade["bars_held"]) == 12


def test_backtester_flats_at_session_end() -> None:
    timestamps = pd.date_range("2024-01-02 15:45:00", periods=3, freq="5min", tz="America/New_York")
    signal_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.date,
            "open": [100.0, 100.1, 100.2],
            "high": [100.4, 100.5, 100.6],
            "low": [99.8, 99.9, 100.0],
            "close": [100.1, 100.2, 100.3],
            "entry_long": [True, False, False],
            "entry_short": [False, False, False],
            "entry_stop_reference_long": [98.5, float("nan"), float("nan")],
            "entry_stop_reference_short": [float("nan"), float("nan"), float("nan")],
            "entry_target_r": [3.0, float("nan"), float("nan")],
            "entry_signal_time": [timestamps[0] - pd.Timedelta(minutes=5), pd.NaT, pd.NaT],
            "entry_signal_atr": [1.0, float("nan"), float("nan")],
            "entry_signal_pullback_depth": [0.5, float("nan"), float("nan")],
            "entry_signal_ema_spread": [0.4, float("nan"), float("nan")],
            "is_last_bar_of_session": [False, False, True],
        }
    )
    variant = IMPCVariantConfig(
        name="impc_backtest_session",
        ema_fast=8,
        ema_slow=21,
        slope_lookback=3,
        pullback_lookback=3,
        pb_min_atr=0.3,
        target_r=3.0,
    )
    execution_model = ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0.0, tick_size=0.25)
    instrument = resolve_instrument_details("MNQ")

    result = run_intraday_momentum_pullback_continuation_backtest(signal_df, variant, execution_model, instrument)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "session_close"


def test_campaign_smoke_run_creates_expected_exports(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "exports" / "impc_smoke"

    dataset_paths: dict[str, Path] = {}
    for idx, symbol in enumerate(("MNQ", "MES", "M2K", "MGC")):
        dataset_path = data_dir / f"{symbol}_c_0_1m_smoke.parquet"
        _write_synthetic_asset_dataset(dataset_path, sessions=30, asset_bias=idx * 0.4)
        dataset_paths[symbol] = dataset_path

    spec = IMPCCampaignSpec(
        output_root=output_dir,
        symbols=("MNQ", "MES", "M2K", "MGC"),
        dataset_paths=dataset_paths,
        max_validation_survivors=2,
    )

    artifacts = run_campaign(spec)

    assert artifacts["output_root"] == output_dir
    assert (output_dir / "screening_summary.csv").exists()
    assert (output_dir / "instrument_variant_summary.csv").exists()
    assert (output_dir / "oos_yearly_summary.csv").exists()
    assert (output_dir / "stress_test_summary.csv").exists()
    assert (output_dir / "survivor_validation_summary.csv").exists()
    assert (output_dir / "mono_asset_candidates_summary.csv").exists()
    assert (output_dir / "final_report.md").exists()
    assert (output_dir / "final_verdict.json").exists()

    instrument_summary = pd.read_csv(output_dir / "instrument_variant_summary.csv")
    screening_summary = pd.read_csv(output_dir / "screening_summary.csv")

    assert len(instrument_summary) == 96
    assert set(instrument_summary["symbol"].unique()) == {"MNQ", "MES", "M2K", "MGC"}
    assert len(screening_summary) == 24
