from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics.m2k_intraday_semivariance_sizing_campaign import (
    M2KBaselineConfig,
    M2KSemivarianceSizingSpec,
    _build_variant_controls,
    _select_promotion_candidates,
    run_campaign,
)
from src.analytics.orb_multi_asset_campaign import BaselineSpec, SearchGrid


def test_build_variant_controls_respects_three_state_and_context_confirmation() -> None:
    trade_features = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]).date,
            "phase": ["is", "is", "oos"],
            "trade_id": [1, 2, 3],
            "weekday_name": ["Tuesday", "Wednesday", "Thursday"],
            "rs_ratio_pct_30m": [0.10, 0.55, 0.92],
            "opening_gap_pct": [0.95, 0.30, 0.40],
            "or_width_pct": [0.20, 0.25, 0.91],
            "atr_pct": [0.15, 0.20, 0.88],
        }
    )

    three_state = _build_variant_controls(
        trade_features,
        family="downside_three_state",
        feature_key="rs_ratio_pct",
        horizon="30m",
        threshold=0.90,
        low_threshold=0.25,
        up_multiplier=1.10,
        down_multiplier=0.75,
    )
    assert three_state["risk_multiplier"].tolist() == [1.10, 1.0, 0.75]
    assert three_state["state"].tolist() == ["favorable", "neutral", "hostile"]

    conditional = _build_variant_controls(
        trade_features,
        family="conditional_downsize_with_context",
        feature_key="rs_ratio_pct",
        horizon="30m",
        threshold=0.90,
        down_multiplier=0.50,
        context_key="wide_or",
    )
    assert conditional["risk_multiplier"].tolist() == [1.0, 1.0, 0.50]
    assert conditional["state"].tolist() == ["nominal", "nominal", "hostile_wide_or"]


def test_select_promotion_candidates_uses_is_only_screen() -> None:
    results = pd.DataFrame(
        [
            {
                "variant_name": "baseline",
                "family": "baseline",
                "feature_key": "baseline",
                "context_key": "",
                "is_reference_variant": False,
                "is_sharpe_delta_vs_baseline": 0.0,
                "is_max_drawdown_improvement_vs_baseline": 0.0,
                "is_profit_factor_delta_vs_baseline": 0.0,
                "is_trade_retention_vs_baseline": 1.0,
                "is_trade_count": 100,
                "oos_sharpe_delta_vs_baseline": 0.0,
                "oos_max_drawdown_improvement_vs_baseline": 0.0,
                "oos_trade_retention_vs_baseline": 1.0,
            },
            {
                "variant_name": "oos_only_winner",
                "family": "downside_downsize",
                "feature_key": "rs_minus_pct",
                "context_key": "",
                "is_reference_variant": False,
                "is_sharpe_delta_vs_baseline": -0.20,
                "is_max_drawdown_improvement_vs_baseline": -0.10,
                "is_profit_factor_delta_vs_baseline": -0.05,
                "is_trade_retention_vs_baseline": 0.90,
                "is_trade_count": 95,
                "oos_sharpe_delta_vs_baseline": 0.40,
                "oos_max_drawdown_improvement_vs_baseline": 0.25,
                "oos_trade_retention_vs_baseline": 0.90,
            },
            {
                "variant_name": "is_screened_candidate",
                "family": "downside_downsize",
                "feature_key": "rs_minus_pct",
                "context_key": "",
                "is_reference_variant": False,
                "is_sharpe_delta_vs_baseline": 0.30,
                "is_max_drawdown_improvement_vs_baseline": 0.20,
                "is_profit_factor_delta_vs_baseline": 0.10,
                "is_trade_retention_vs_baseline": 0.85,
                "is_trade_count": 92,
                "oos_sharpe_delta_vs_baseline": 0.05,
                "oos_max_drawdown_improvement_vs_baseline": 0.02,
                "oos_trade_retention_vs_baseline": 0.82,
            },
        ]
    )
    spec = M2KSemivarianceSizingSpec(min_trade_retention=0.70, min_is_trade_count=60)

    scored, candidates = _select_promotion_candidates(results, spec)

    candidate_names = set(candidates["variant_name"].tolist())
    assert "is_screened_candidate" in candidate_names
    assert "oos_only_winner" not in candidate_names

    promoted = scored.set_index("variant_name")
    assert bool(promoted.loc["is_screened_candidate", "promotion_candidate"])
    assert not bool(promoted.loc["oos_only_winner", "promotion_candidate"])


def _write_synthetic_m2k_dataset(path: Path, *, sessions: int = 18) -> None:
    rows: list[dict[str, object]] = []
    session_dates = pd.bdate_range("2024-01-02", periods=sessions)

    for day_idx, session_date in enumerate(session_dates):
        base = 100.0 + day_idx * 0.10
        vol = 0.025 + day_idx * 0.002
        previous_close = base

        for minute_idx in range(120):
            timestamp = pd.Timestamp(session_date.date()).tz_localize("America/New_York") + pd.Timedelta(
                hours=9, minutes=30 + minute_idx
            )
            if minute_idx < 15:
                close = base + ((minute_idx % 5) - 2) * vol
            else:
                drift = (minute_idx - 14) * vol * 0.35
                close = base + (2.5 * vol) + drift

            open_price = previous_close
            high = max(open_price, close) + vol * 0.30
            low = min(open_price, close) - vol * 0.30
            rows.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 200.0 + day_idx,
                }
            )
            previous_close = close

    pd.DataFrame(rows).to_parquet(path, index=False)


def test_campaign_smoke_exports_expected_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dataset_path = data_dir / "M2K_c_0_1m_synth.parquet"
    _write_synthetic_m2k_dataset(dataset_path)

    baseline_config = M2KBaselineConfig(
        symbol="M2K",
        source_reference=str(dataset_path),
        source_note="synthetic smoke baseline",
        baseline=BaselineSpec(
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
        ),
        grid=SearchGrid(
            atr_periods=(20,),
            q_lows_pct=(10,),
            q_highs_pct=(90,),
            aggregation_rules=("majority_50",),
        ),
        aggregation_rule="majority_50",
        dataset_path=dataset_path,
    )
    output_dir = tmp_path / "exports" / "m2k_semivar_smoke"
    spec = M2KSemivarianceSizingSpec(
        semivariance_horizons=("30m", "session"),
        downside_feature_keys=("rs_minus_share_pct",),
        downside_thresholds=(0.85,),
        downside_multipliers=(0.75,),
        three_state_pairs=((1.10, 0.75),),
        conditional_feature_keys=("rs_minus_share_pct",),
        context_keys=("wide_or",),
        reference_skip_feature_keys=("rs_minus_pct",),
        reference_skip_horizons=("session",),
        reference_skip_thresholds=(0.90,),
        percentile_history=5,
        min_percentile_history=3,
        min_trade_retention=0.50,
        min_is_trade_count=3,
        output_root=output_dir,
        baseline_config=baseline_config,
    )

    artifacts = run_campaign(spec)

    assert artifacts.output_dir == output_dir
    assert (output_dir / "variant_results.csv").exists()
    assert (output_dir / "promotion_candidates.csv").exists()
    assert (output_dir / "heatmap_ready.csv").exists()
    assert (output_dir / "m2k_trade_features.csv").exists()
    assert (output_dir / "weekday_downside_bucket_summary.csv").exists()
    assert (output_dir / "final_report.md").exists()
    assert (output_dir / "final_verdict.json").exists()
    assert (output_dir / "run_metadata.json").exists()

    features = pd.read_csv(output_dir / "m2k_trade_features.csv")
    results = pd.read_csv(output_dir / "variant_results.csv")

    assert {"rs_ratio_30m", "rs_ratio_pct_30m", "opening_gap_pct", "or_width_pct", "atr_pct", "weekday_name"}.issubset(
        set(features.columns)
    )
    assert "baseline" in set(results["variant_name"])
