from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from src.analytics.volume_climax_pullback_v2_campaign import run_campaign as run_v2_campaign
from src.analytics.volume_climax_pullback_v3_campaign import run_campaign as run_v3_campaign
from src.engine.execution_model import ExecutionModel
from src.engine.volume_climax_pullback_v2_backtester import run_volume_climax_pullback_v2_backtest
from src.engine.vwap_backtester import InstrumentDetails
from src.strategy.volume_climax_pullback_v2 import (
    VolumeClimaxPullbackV2Variant,
    build_volume_climax_pullback_v2_signal_frame,
    build_volume_climax_pullback_v3_variants,
    prepare_volume_climax_pullback_v2_features,
)


def _base_variant(**overrides: object) -> VolumeClimaxPullbackV2Variant:
    payload: dict[str, object] = {
        "name": "v2_test",
        "family": "signal_core",
        "timeframe": "1h",
        "volume_quantile": 0.95,
        "volume_lookback": 50,
        "min_body_fraction": 0.5,
        "min_range_atr": 1.2,
        "trend_ema_window": None,
        "ema_slope_threshold": None,
        "atr_percentile_low": None,
        "atr_percentile_high": None,
        "compression_ratio_max": None,
        "entry_mode": "next_open",
        "pullback_fraction": None,
        "confirmation_window": None,
        "exit_mode": "fixed_rr",
        "rr_target": 1.0,
        "atr_target_multiple": None,
        "time_stop_bars": 2,
        "trailing_atr_multiple": 0.5,
        "session_overlay": "all_rth",
    }
    payload.update(overrides)
    return VolumeClimaxPullbackV2Variant(**payload)


def _instrument() -> InstrumentDetails:
    return InstrumentDetails(
        symbol="MNQ",
        asset_class="futures",
        tick_size=0.25,
        tick_value_usd=0.5,
        point_value_usd=1.0,
        commission_per_side_usd=0.0,
        slippage_ticks=0,
    )


def _hourly_feature_frame(periods: int = 72) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:30:00", periods=periods, freq="1h", tz="America/New_York")
    rows: list[dict[str, object]] = []
    for idx, timestamp in enumerate(timestamps):
        base = 100.0 + idx * 0.05
        open_price = base
        close = base - 0.20 if idx % 3 == 0 else base + 0.10
        high = max(open_price, close) + 0.15
        low = min(open_price, close) - 0.15
        volume = 100.0 + idx
        if idx == 55:
            close = open_price + 1.8
            high = close + 0.20
            low = open_price - 0.20
            volume = 10_000.0
        rows.append(
            {
                "timestamp": timestamp,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
    return pd.DataFrame(rows)


def _write_synthetic_minute_dataset(path: Path, *, asset_bias: float = 0.0, sessions: int = 12) -> None:
    rows: list[dict[str, object]] = []
    for day_index, session_date in enumerate(pd.bdate_range("2024-01-02", periods=sessions)):
        session_open = pd.Timestamp(session_date.date()).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        previous_close = 100.0 + asset_bias + day_index * 0.02
        for minute_index in range(390):
            timestamp = session_open + pd.Timedelta(minutes=minute_index)
            phase = minute_index % 90
            drift = 0.006 * phase if phase < 45 else 0.27 - 0.008 * (phase - 45)
            close = 100.0 + asset_bias + day_index * 0.02 + drift
            if minute_index in {59, 149, 239, 329}:
                close -= 0.30
            open_price = previous_close
            high = max(open_price, close) + 0.03
            low = min(open_price, close) - 0.03
            volume = 120.0 + (minute_index % 25)
            if minute_index in {59, 149, 239, 329}:
                volume = 6_000.0 + day_index * 10.0
            rows.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
            previous_close = close
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_synthetic_v2_reference_summary(path: Path) -> Path:
    reference_dir = path / "volume_climax_pullback_v2_20260401_214553"
    reference_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "symbol": "MNQ",
                "variant_name": "dynamic_exit_atr_target_1p0_ts3_vq0p95_bf0p5_ra1p2",
                "family": "dynamic_exit",
                "oos_net_pnl": 250.0,
                "oos_profit_factor": 1.30,
                "oos_sharpe": 0.80,
                "oos_expectancy": 12.0,
                "oos_nb_trades": 8,
                "stability_is_oos_sharpe_ratio": 0.90,
                "selection_score": 1.20,
            },
            {
                "symbol": "MES",
                "variant_name": "dynamic_exit_mixed_ts4_vq0p95_bf0p6_ra1p2",
                "family": "dynamic_exit",
                "oos_net_pnl": 210.0,
                "oos_profit_factor": 1.25,
                "oos_sharpe": 0.75,
                "oos_expectancy": 10.0,
                "oos_nb_trades": 8,
                "stability_is_oos_sharpe_ratio": 0.85,
                "selection_score": 1.10,
            },
            {
                "symbol": "M2K",
                "variant_name": "dynamic_exit_atr_target_1p0_ts3_vq0p95_bf0p5_ra1p2",
                "family": "dynamic_exit",
                "oos_net_pnl": 180.0,
                "oos_profit_factor": 1.22,
                "oos_sharpe": 0.70,
                "oos_expectancy": 9.0,
                "oos_nb_trades": 7,
                "stability_is_oos_sharpe_ratio": 0.82,
                "selection_score": 1.00,
            },
            {
                "symbol": "MGC",
                "variant_name": "regime_filtered_trend_ema50_medium_vq0p975_bf0p5_ra1p2",
                "family": "regime_filtered",
                "oos_net_pnl": 190.0,
                "oos_profit_factor": 1.35,
                "oos_sharpe": 0.78,
                "oos_expectancy": 11.0,
                "oos_nb_trades": 7,
                "stability_is_oos_sharpe_ratio": 0.88,
                "selection_score": 1.15,
            },
            {
                "symbol": "MGC",
                "variant_name": "dynamic_exit_mixed_ts4_vq0p95_bf0p5_ra1p2",
                "family": "dynamic_exit",
                "oos_net_pnl": 175.0,
                "oos_profit_factor": 1.28,
                "oos_sharpe": 0.74,
                "oos_expectancy": 10.5,
                "oos_nb_trades": 7,
                "stability_is_oos_sharpe_ratio": 0.84,
                "selection_score": 1.05,
            },
        ]
    ).to_csv(reference_dir / "summary_variants.csv", index=False)
    return reference_dir


def test_v2_signal_prefix_is_strictly_historical() -> None:
    frame = _hourly_feature_frame()
    variant = _base_variant()

    short = build_volume_climax_pullback_v2_signal_frame(
        prepare_volume_climax_pullback_v2_features(frame.iloc[:60].copy()),
        variant,
    )
    full = build_volume_climax_pullback_v2_signal_frame(
        prepare_volume_climax_pullback_v2_features(frame.copy()),
        variant,
    ).iloc[: len(short)].reset_index(drop=True)

    pdt.assert_series_equal(short["raw_signal"], full["raw_signal"], check_names=False)
    pdt.assert_series_equal(short["signal"], full["signal"], check_names=False)
    pdt.assert_series_equal(short["setup_reference_close"], full["setup_reference_close"], check_names=False)


def test_backtester_handles_pullback_limit_entry() -> None:
    timestamps = pd.date_range("2024-01-02 09:30:00", periods=3, freq="1h", tz="America/New_York")
    signal_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": [timestamps[0].date()] * 3,
            "open": [100.5, 101.8, 101.4],
            "high": [102.4, 102.0, 101.6],
            "low": [99.8, 100.8, 100.9],
            "close": [100.0, 101.1, 101.0],
            "signal": [-1, 0, 0],
            "setup_reference_close": [100.0, pd.NA, pd.NA],
            "setup_reference_range": [4.0, pd.NA, pd.NA],
            "setup_stop_reference_long": [pd.NA, pd.NA, pd.NA],
            "setup_stop_reference_short": [103.0, pd.NA, pd.NA],
            "setup_reference_atr": [2.0, pd.NA, pd.NA],
            "setup_reference_vwap": [99.0, pd.NA, pd.NA],
            "setup_signal_time": [timestamps[0] - pd.Timedelta(hours=1), pd.NaT, pd.NaT],
        }
    )
    variant = _base_variant(name="pullback_limit_test", entry_mode="pullback_limit", pullback_fraction=0.5)
    trades = run_volume_climax_pullback_v2_backtest(
        signal_df,
        variant,
        ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0, tick_size=0.25),
        _instrument(),
    ).trades

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert float(trade["entry_price"]) == 102.0
    assert str(trade["exit_reason"]) == "target"


def test_backtester_confirmation_enters_next_bar_open_only() -> None:
    timestamps = pd.date_range("2024-01-02 09:30:00", periods=4, freq="1h", tz="America/New_York")
    signal_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": [timestamps[0].date()] * 4,
            "open": [100.0, 100.5, 101.0, 100.9],
            "high": [100.8, 101.4, 103.2, 101.2],
            "low": [98.8, 100.1, 100.7, 100.6],
            "close": [100.4, 101.1, 102.9, 100.8],
            "signal": [1, 0, 0, 0],
            "setup_reference_close": [100.0, pd.NA, pd.NA, pd.NA],
            "setup_reference_range": [4.0, pd.NA, pd.NA, pd.NA],
            "setup_stop_reference_long": [98.0, pd.NA, pd.NA, pd.NA],
            "setup_stop_reference_short": [pd.NA, pd.NA, pd.NA, pd.NA],
            "setup_reference_atr": [2.0, pd.NA, pd.NA, pd.NA],
            "setup_reference_vwap": [101.5, pd.NA, pd.NA, pd.NA],
            "setup_signal_time": [timestamps[0] - pd.Timedelta(hours=1), pd.NaT, pd.NaT, pd.NaT],
        }
    )
    variant = _base_variant(
        name="confirmation_test",
        entry_mode="confirmation",
        pullback_fraction=0.25,
        confirmation_window=1,
    )
    trades = run_volume_climax_pullback_v2_backtest(
        signal_df,
        variant,
        ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0, tick_size=0.25),
        _instrument(),
    ).trades

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert pd.Timestamp(trade["entry_time"]) == timestamps[1]
    assert pd.Timestamp(trade["entry_time"]) != timestamps[0]


def test_backtester_supports_mixed_partial_then_trailing_exit() -> None:
    timestamps = pd.date_range("2024-01-02 09:30:00", periods=4, freq="1h", tz="America/New_York")
    signal_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": [timestamps[0].date()] * 4,
            "open": [100.0, 100.8, 101.2, 100.7],
            "high": [100.4, 101.2, 101.5, 100.9],
            "low": [99.7, 100.7, 100.4, 100.5],
            "close": [100.1, 101.5, 100.8, 100.6],
            "signal": [1, 0, 0, 0],
            "setup_reference_close": [100.0, pd.NA, pd.NA, pd.NA],
            "setup_reference_range": [2.0, pd.NA, pd.NA, pd.NA],
            "setup_stop_reference_long": [99.0, pd.NA, pd.NA, pd.NA],
            "setup_stop_reference_short": [pd.NA, pd.NA, pd.NA, pd.NA],
            "setup_reference_atr": [2.0, pd.NA, pd.NA, pd.NA],
            "setup_reference_vwap": [101.0, pd.NA, pd.NA, pd.NA],
            "setup_signal_time": [timestamps[0] - pd.Timedelta(hours=1), pd.NaT, pd.NaT, pd.NaT],
        }
    )
    variant = _base_variant(name="mixed_test", exit_mode="mixed", time_stop_bars=4)
    trades = run_volume_climax_pullback_v2_backtest(
        signal_df,
        variant,
        ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0, tick_size=0.25),
        _instrument(),
    ).trades

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert str(trade["exit_reason"]) == "mixed_trailing_stop"
    assert round(float(trade["pnl_points"]), 4) == 0.75


def test_v2_campaign_smoke_outputs_requested_files(tmp_path: Path) -> None:
    input_paths: dict[str, Path] = {}
    for symbol, bias in {"MNQ": 0.0, "MES": 0.2, "M2K": -0.1, "MGC": 0.35}.items():
        path = tmp_path / f"{symbol}_c_0_1m_test.parquet"
        _write_synthetic_minute_dataset(path, asset_bias=bias)
        input_paths[symbol] = path

    variant = _base_variant(name="smoke_variant")
    output_dir = run_v2_campaign(
        tmp_path / "exports",
        symbols=("MNQ", "MES", "M2K", "MGC"),
        variants=[variant],
        input_paths=input_paths,
    )

    assert (output_dir / "summary_variants.csv").exists()
    assert (output_dir / "ranking_oos.csv").exists()
    assert (output_dir / "ranking_oos_by_asset.csv").exists()
    assert (output_dir / "comparison_vs_v1.csv").exists()
    assert (output_dir / "breakdown_by_asset.csv").exists()
    assert (output_dir / "family_research_summary.csv").exists()
    assert (output_dir / "final_report.md").exists()

    summary = pd.read_csv(output_dir / "summary_variants.csv")
    assert set(summary["generation"]) == {"v1_ref", "v2"}


def test_v3_variant_grid_is_asset_aware_and_compact() -> None:
    mnq_variants = build_volume_climax_pullback_v3_variants("MNQ")
    mgc_variants = build_volume_climax_pullback_v3_variants("MGC")

    assert len(mnq_variants) == 48
    assert len({variant.name for variant in mnq_variants}) == len(mnq_variants)
    assert {variant.family for variant in mnq_variants} == {"dynamic_exit"}

    assert len(mgc_variants) == 114
    assert len({variant.name for variant in mgc_variants}) == len(mgc_variants)
    assert {variant.family for variant in mgc_variants} == {"dynamic_exit", "regime_filtered"}
    assert sum(variant.family == "dynamic_exit" for variant in mgc_variants) == 48
    assert sum(variant.family == "regime_filtered" for variant in mgc_variants) == 66


def test_v3_campaign_smoke_outputs_requested_files(tmp_path: Path) -> None:
    input_paths: dict[str, Path] = {}
    for symbol, bias in {"MNQ": 0.0, "MES": 0.2, "M2K": -0.1, "MGC": 0.35}.items():
        path = tmp_path / f"{symbol}_c_0_1m_test.parquet"
        _write_synthetic_minute_dataset(path, asset_bias=bias)
        input_paths[symbol] = path

    reference_dir = _write_synthetic_v2_reference_summary(tmp_path / "v2_refs")
    variants_by_symbol = {
        "MNQ": [build_volume_climax_pullback_v3_variants("MNQ")[0]],
        "MES": [build_volume_climax_pullback_v3_variants("MES")[0]],
        "M2K": [build_volume_climax_pullback_v3_variants("M2K")[0]],
        "MGC": [
            build_volume_climax_pullback_v3_variants("MGC")[0],
            next(variant for variant in build_volume_climax_pullback_v3_variants("MGC") if variant.family == "regime_filtered"),
        ],
    }

    output_dir = run_v3_campaign(
        tmp_path / "exports_v3",
        symbols=("MNQ", "MES", "M2K", "MGC"),
        variants_by_symbol=variants_by_symbol,
        input_paths=input_paths,
        v2_reference_dir=reference_dir,
        max_workers=1,
    )

    assert (output_dir / "summary_variants.csv").exists()
    assert (output_dir / "ranking_oos.csv").exists()
    assert (output_dir / "ranking_oos_by_asset.csv").exists()
    assert (output_dir / "comparison_vs_v2.csv").exists()
    assert (output_dir / "final_report.md").exists()

    summary = pd.read_csv(output_dir / "summary_variants.csv")
    assert set(summary["generation"]) == {"v3"}
    assert "v2_reference_variant_name" in summary.columns
    assert "anomaly_profit_factor_inf" in summary.columns
    assert "variant_status" in summary.columns
    assert set(summary["symbol"]) == {"MNQ", "MES", "M2K", "MGC"}

    report = (output_dir / "final_report.md").read_text(encoding="utf-8")
    assert "Verdicts By Asset" in report
    assert "Global Conclusion" in report
