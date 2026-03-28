from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics.volatility_compression_expansion_campaign import VCEBCampaignSpec, run_campaign
from src.data.resampling import resample_ohlcv
from src.engine.execution_model import ExecutionModel
from src.engine.volatility_compression_expansion_backtester import run_volatility_compression_expansion_backtest
from src.engine.vwap_backtester import resolve_instrument_details
from src.strategy.volatility_compression_expansion import (
    VCEBVariantConfig,
    build_vceb_signal_frame,
    prepare_vceb_feature_frame,
)


def _five_minute_frame(periods: int = 8) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:30:00", periods=periods, freq="5min", tz="America/New_York")
    closes = [100.0, 100.2, 100.1, 100.3, 100.25, 101.4, 101.7, 101.9][:periods]
    rows: list[dict[str, float | pd.Timestamp]] = []
    previous_close = 100.0
    for idx, timestamp in enumerate(timestamps):
        close = closes[idx]
        open_price = previous_close
        high = max(open_price, close) + 0.15
        low = min(open_price, close) - 0.15
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


def test_feature_pipeline_is_strictly_no_lookahead() -> None:
    frame = _five_minute_frame(periods=8)

    short = prepare_vceb_feature_frame(
        frame.iloc[:6].copy(),
        box_lengths=(2,),
        atr_window=3,
        compression_percentile_lookback_bars=4,
        compression_percentile_min_history_bars=1,
    )
    full = prepare_vceb_feature_frame(
        frame.copy(),
        box_lengths=(2,),
        atr_window=3,
        compression_percentile_lookback_bars=4,
        compression_percentile_min_history_bars=1,
    )
    variant = VCEBVariantConfig(
        name="vceb_test_small",
        box_lookback=2,
        compression_threshold=100.0,
        target_r=1.8,
        atr_window=3,
    )

    short_signal = build_vceb_signal_frame(short, variant)
    full_signal = build_vceb_signal_frame(full, variant).iloc[: len(short_signal)].reset_index(drop=True)

    assert float(full_signal.loc[3, "box_high_2"]) == max(float(frame.loc[1, "high"]), float(frame.loc[2, "high"]))
    assert float(full_signal.loc[3, "box_low_2"]) == min(float(frame.loc[1, "low"]), float(frame.loc[2, "low"]))
    pd.testing.assert_series_equal(short_signal["compression_pct_2"], full_signal["compression_pct_2"], check_names=False)
    pd.testing.assert_series_equal(short_signal["raw_signal"], full_signal["raw_signal"], check_names=False)


def test_resample_1m_to_5m_is_correct_for_vceb_inputs() -> None:
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


def test_signal_requires_valid_compression_and_shifts_to_next_open() -> None:
    timestamps = pd.date_range("2024-01-02 09:45:00", periods=3, freq="5min", tz="America/New_York")
    base = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.date,
            "open": [100.0, 100.2, 101.8],
            "high": [100.3, 102.3, 102.0],
            "low": [99.8, 100.0, 101.4],
            "close": [100.2, 102.0, 101.9],
            "atr_48": [1.0, 1.0, 1.0],
            "true_range": [0.5, 2.3, 0.6],
            "expansion_ratio": [0.5, 2.3, 0.6],
            "box_high_8": [100.5, 101.0, 101.0],
            "box_low_8": [99.5, 99.7, 99.7],
            "box_width_8": [1.0, 1.3, 1.3],
            "compression_ratio_8": [1.0, 1.3, 1.3],
            "compression_pct_8": [0.50, 0.10, 0.10],
            "is_last_bar_of_session": [False, False, True],
        }
    )
    variant = VCEBVariantConfig(name="vceb_gate", box_lookback=8, compression_threshold=15.0, target_r=1.8)

    valid = build_vceb_signal_frame(base, variant)
    invalid = build_vceb_signal_frame(base.assign(compression_pct_8=[0.50, 0.30, 0.30]), variant)

    assert int(valid.loc[1, "raw_signal"]) == 1
    assert bool(valid.loc[1, "entry_long"]) is False
    assert bool(valid.loc[2, "entry_long"]) is True
    assert int(valid.loc[2, "signal"]) == 1
    assert invalid["raw_signal"].tolist() == [0, 0, 0]
    assert invalid["signal"].tolist() == [0, 0, 0]


def test_backtester_applies_stop_and_target_levels_from_entry_bar() -> None:
    timestamps = pd.date_range("2024-01-02 09:55:00", periods=3, freq="5min", tz="America/New_York")
    signal_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.date,
            "open": [100.0, 100.5, 100.2],
            "high": [102.4, 100.8, 100.5],
            "low": [99.8, 100.1, 99.9],
            "close": [101.8, 100.3, 100.1],
            "entry_long": [True, False, False],
            "entry_short": [False, False, False],
            "entry_stop_distance": [1.0, float("nan"), float("nan")],
            "entry_target_r": [2.0, float("nan"), float("nan")],
            "entry_signal_time": [timestamps[0] - pd.Timedelta(minutes=5), pd.NaT, pd.NaT],
            "entry_signal_atr": [1.0, float("nan"), float("nan")],
            "entry_signal_box_width": [1.2, float("nan"), float("nan")],
            "is_last_bar_of_session": [False, False, True],
        }
    )
    variant = VCEBVariantConfig(name="vceb_backtest_target", box_lookback=8, compression_threshold=15.0, target_r=2.0)
    execution_model = ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0.0, tick_size=0.25)
    instrument = resolve_instrument_details("MNQ")

    result = run_volatility_compression_expansion_backtest(signal_df, variant, execution_model, instrument)

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
            "entry_stop_distance": [2.0] + [float("nan")] * 11,
            "entry_target_r": [2.0] + [float("nan")] * 11,
            "entry_signal_time": [timestamps[0] - pd.Timedelta(minutes=5)] + [pd.NaT] * 11,
            "entry_signal_atr": [2.0] + [float("nan")] * 11,
            "entry_signal_box_width": [1.5] + [float("nan")] * 11,
            "is_last_bar_of_session": [False] * 11 + [True],
        }
    )
    variant = VCEBVariantConfig(name="vceb_backtest_time", box_lookback=8, compression_threshold=15.0, target_r=2.0)
    execution_model = ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0.0, tick_size=0.25)
    instrument = resolve_instrument_details("MNQ")

    result = run_volatility_compression_expansion_backtest(signal_df, variant, execution_model, instrument)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "time_stop"
    assert float(trade["holding_minutes"]) == 60.0
    assert int(trade["bars_held"]) == 12


def _write_synthetic_asset_dataset(path: Path, *, sessions: int = 30, asset_bias: float = 0.0) -> None:
    rows: list[dict[str, object]] = []
    session_dates = pd.bdate_range("2024-01-02", periods=sessions)

    for day_idx, session_date in enumerate(session_dates):
        session_open = pd.Timestamp(session_date.date()).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        base = 100.0 + day_idx * 0.08 + asset_bias
        direction = 1.0 if day_idx % 2 == 0 else -1.0
        previous_close = base

        for minute_idx in range(390):
            timestamp = session_open + pd.Timedelta(minutes=minute_idx)
            if minute_idx < 90:
                close = base + (((minute_idx % 6) - 3) * 0.015)
            elif minute_idx < 150:
                drift = (minute_idx - 90) * 0.012 * direction
                close = base + direction * 0.18 + drift
            else:
                close = previous_close + direction * 0.003

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


def test_campaign_smoke_run_creates_expected_exports(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "exports" / "vceb_smoke"

    dataset_paths: dict[str, Path] = {}
    for idx, symbol in enumerate(("MNQ", "MES", "M2K", "MGC")):
        dataset_path = data_dir / f"{symbol}_c_0_1m_smoke.parquet"
        _write_synthetic_asset_dataset(dataset_path, sessions=30, asset_bias=idx * 0.5)
        dataset_paths[symbol] = dataset_path

    spec = VCEBCampaignSpec(
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
    assert (output_dir / "final_report.md").exists()
    assert (output_dir / "final_verdict.json").exists()

    instrument_summary = pd.read_csv(output_dir / "instrument_variant_summary.csv")
    screening_summary = pd.read_csv(output_dir / "screening_summary.csv")

    assert len(instrument_summary) == 48
    assert set(instrument_summary["symbol"].unique()) == {"MNQ", "MES", "M2K", "MGC"}
    assert len(screening_summary) == 12
