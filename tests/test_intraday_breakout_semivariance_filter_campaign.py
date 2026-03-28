from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analytics.intraday_breakout_semivariance_filter_campaign import (
    AssetBaselineConfig,
    SemivarianceCampaignSpec,
    run_campaign,
)
from src.analytics.orb_multi_asset_campaign import BaselineSpec, SearchGrid
from src.features.semivariance import (
    add_directional_semivariance_context,
    add_realized_semivariance_features,
    rolling_percentile_rank,
)


def test_realized_semivariance_features_compute_expected_values() -> None:
    timestamp = pd.date_range("2024-01-02 09:30:00", periods=4, freq="1min", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "session_date": timestamp.date,
            "continuous_session_date": timestamp.date,
            "open": [100.0, 100.0, 110.0, 104.5],
            "high": [100.0, 110.0, 110.0, 125.4],
            "low": [100.0, 100.0, 104.5, 104.5],
            "close": [100.0, 110.0, 104.5, 125.4],
            "volume": [10.0, 10.0, 10.0, 10.0],
        }
    )

    featured = add_realized_semivariance_features(frame, window_minutes=(2,))
    last = featured.iloc[-1]

    assert float(last["rs_plus_2m"]) == pytest.approx(0.04)
    assert float(last["rs_minus_2m"]) == pytest.approx(0.0025)
    assert float(last["rv_2m"]) == pytest.approx(0.0425)
    assert float(last["rs_plus_session"]) == pytest.approx(0.05)
    assert float(last["rs_minus_session"]) == pytest.approx(0.0025)
    assert float(last["rs_imbalance_2m"]) == pytest.approx((0.04 - 0.0025) / 0.0425)


def test_rolling_percentile_rank_is_strictly_no_lookahead() -> None:
    values = pd.Series([10.0, 20.0, 15.0, 30.0])
    ranks = rolling_percentile_rank(values, lookback=3, min_history=1)

    assert pd.isna(ranks.iloc[0])
    assert float(ranks.iloc[1]) == pytest.approx(1.0)
    assert float(ranks.iloc[2]) == pytest.approx(0.5)
    assert float(ranks.iloc[3]) == pytest.approx(1.0)


def test_directional_adverse_mapping_uses_long_vs_short_correctly() -> None:
    frame = pd.DataFrame(
        {
            "breakout_side": ["long", "short"],
            "rs_plus_30m": [1.0, 2.0],
            "rs_minus_30m": [3.0, 4.0],
            "rs_plus_share_30m": [0.25, 0.40],
            "rs_minus_share_30m": [0.75, 0.60],
            "rs_plus_pct_30m": [0.10, 0.20],
            "rs_minus_pct_30m": [0.90, 0.80],
        }
    )

    mapped = add_directional_semivariance_context(frame, horizons=("30m",), side_col="breakout_side")

    assert mapped["adverse_semivariance_30m"].tolist() == [3.0, 2.0]
    assert mapped["supportive_semivariance_30m"].tolist() == [1.0, 4.0]
    assert mapped["adverse_share_30m"].tolist() == [0.75, 0.40]
    assert mapped["adverse_pct_30m"].tolist() == [0.90, 0.20]


def _write_synthetic_asset_dataset(
    path: Path,
    *,
    sessions: int = 16,
    direction: str = "long",
) -> None:
    rows: list[dict[str, object]] = []
    session_dates = pd.bdate_range("2024-01-02", periods=sessions)

    for day_idx, session_date in enumerate(session_dates):
        base = 100.0 + day_idx * 0.15
        vol = 0.03 + day_idx * 0.003
        previous_close = base

        for minute_idx in range(120):
            timestamp = pd.Timestamp(session_date.date()).tz_localize("America/New_York") + pd.Timedelta(
                hours=9, minutes=30 + minute_idx
            )
            if minute_idx < 15:
                close = base + ((minute_idx % 5) - 2) * vol
            else:
                drift = (minute_idx - 14) * vol * 0.45
                if direction == "short":
                    close = base - (3.0 * vol) - drift
                else:
                    close = base + (3.0 * vol) + drift

            open_price = previous_close
            high = max(open_price, close) + vol * 0.35
            low = min(open_price, close) - vol * 0.35
            rows.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 100.0 + day_idx,
                }
            )
            previous_close = close

    pd.DataFrame(rows).to_parquet(path, index=False)


def test_campaign_smoke_exports_expected_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "exports" / "semivar_smoke"

    datasets = {}
    for symbol in ("MNQ", "MES", "MGC", "M2K"):
        dataset_path = data_dir / f"{symbol}_c_0_1m_synth.parquet"
        _write_synthetic_asset_dataset(dataset_path, direction="long")
        datasets[symbol] = dataset_path

    baseline = BaselineSpec(
        or_minutes=15,
        opening_time="09:30:00",
        direction="long",
        one_trade_per_day=True,
        entry_buffer_ticks=1,
        stop_buffer_ticks=1,
        target_multiple=1.5,
        vwap_confirmation=True,
        vwap_column="continuous_session_vwap",
        time_exit="16:00:00",
        account_size_usd=50_000.0,
        risk_per_trade_pct=1.5,
        entry_on_next_open=True,
    )
    grid = SearchGrid(
        atr_periods=(20,),
        q_lows_pct=(10,),
        q_highs_pct=(90,),
        aggregation_rules=("majority_50",),
    )
    registry = {
        symbol: AssetBaselineConfig(
            symbol=symbol,
            source_reference=str(datasets[symbol]),
            source_note="synthetic smoke baseline",
            baseline=baseline,
            grid=grid,
            aggregation_rule="majority_50",
            dataset_path=datasets[symbol],
        )
        for symbol in ("MNQ", "MES", "MGC", "M2K")
    }
    spec = SemivarianceCampaignSpec(
        symbols=("MNQ", "MES", "MGC", "M2K"),
        semivariance_horizons=("30m", "session"),
        percentile_thresholds=(0.80,),
        downsizing_multipliers=(0.50, 0.00),
        percentile_history=5,
        min_percentile_history=3,
        output_root=output_dir,
        asset_baselines=registry,
    )

    artifacts = run_campaign(spec)

    assert artifacts.output_dir == output_dir
    assert (output_dir / "asset_variant_results.csv").exists()
    assert (output_dir / "portfolio_variant_results.csv").exists()
    assert (output_dir / "final_report.md").exists()
    assert (output_dir / "run_metadata.json").exists()
    assert (output_dir / "final_verdict.json").exists()

    asset_results = pd.read_csv(output_dir / "asset_variant_results.csv")
    portfolio_results = pd.read_csv(output_dir / "portfolio_variant_results.csv")

    assert set(asset_results["asset"].unique()) == {"MNQ", "MES", "MGC", "M2K"}
    assert "baseline" in set(asset_results["variant_name"])
    assert "baseline" in set(portfolio_results["variant_name"])
