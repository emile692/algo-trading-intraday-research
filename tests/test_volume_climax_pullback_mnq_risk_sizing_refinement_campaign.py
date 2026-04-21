from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analytics.volume_climax_pullback_mnq_risk_sizing_refinement_campaign import (
    BEST_PREVIOUS_WINNER_ALIAS,
    build_refinement_variants,
    compute_prop_score,
    identify_connected_clusters,
    run_campaign,
)
from src.strategy.volume_climax_pullback_v2 import build_volume_climax_pullback_v3_variants


def _write_synthetic_minute_dataset(path: Path, sessions: int = 12) -> None:
    rows: list[dict[str, object]] = []
    for day_index, session_date in enumerate(pd.bdate_range("2024-01-02", periods=sessions)):
        session_open = pd.Timestamp(session_date.date()).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        previous_close = 100.0 + day_index * 0.02
        for minute_index in range(390):
            timestamp = session_open + pd.Timedelta(minutes=minute_index)
            phase = minute_index % 90
            drift = 0.006 * phase if phase < 45 else 0.27 - 0.008 * (phase - 45)
            close = 100.0 + day_index * 0.02 + drift
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


def test_build_refinement_variants_grid_and_alias() -> None:
    variants = build_refinement_variants(
        initial_capital_usd=50_000.0,
        risk_pcts=(0.0025, 0.0030),
        max_contracts_grid=(2, 3),
        include_best_previous_winner_alias=True,
    )

    assert len(variants) == 6
    assert variants[0].campaign_variant_name == "fixed_1_contract"
    assert variants[0].variant_role == "baseline"
    assert sum(variant.variant_role == "grid" for variant in variants) == 4
    assert any(variant.campaign_variant_name == BEST_PREVIOUS_WINNER_ALIAS for variant in variants)
    assert all(
        variant.skip_trade_if_too_small in (None, True)
        for variant in variants
    )


def test_compute_prop_score_rewards_cleaner_prop_profile() -> None:
    good = pd.Series(
        {
            "oos_net_pnl_usd": 7_000.0,
            "oos_sharpe": 1.8,
            "oos_max_drawdown_usd": 650.0,
            "oos_max_daily_drawdown_usd": 620.0,
            "oos_worst_trade_loss_usd": 140.0,
            "oos_nb_days_below_minus_500": 0,
            "oos_pass_target_3000_usd_without_breaching_2000_dd": True,
        }
    )
    bad = pd.Series(
        {
            "oos_net_pnl_usd": 9_000.0,
            "oos_sharpe": 1.1,
            "oos_max_drawdown_usd": 3_000.0,
            "oos_max_daily_drawdown_usd": 1_250.0,
            "oos_worst_trade_loss_usd": 320.0,
            "oos_nb_days_below_minus_500": 2,
            "oos_pass_target_3000_usd_without_breaching_2000_dd": False,
        }
    )

    assert compute_prop_score(good, prefix="oos") > compute_prop_score(bad, prefix="oos")


def test_identify_connected_clusters_on_synthetic_grid() -> None:
    frame = pd.DataFrame(
        [
            {"campaign_variant_name": "a", "risk_pct": 0.0015, "max_contracts": 2, "oos_net_pnl_usd": 1, "oos_sharpe": 1.0, "oos_max_drawdown_usd": 100, "oos_prop_score": 5.0, "is_top_quartile_prop_score": True},
            {"campaign_variant_name": "b", "risk_pct": 0.0015, "max_contracts": 3, "oos_net_pnl_usd": 1, "oos_sharpe": 1.0, "oos_max_drawdown_usd": 100, "oos_prop_score": 4.0, "is_top_quartile_prop_score": True},
            {"campaign_variant_name": "c", "risk_pct": 0.0020, "max_contracts": 3, "oos_net_pnl_usd": 1, "oos_sharpe": 1.0, "oos_max_drawdown_usd": 100, "oos_prop_score": 6.0, "is_top_quartile_prop_score": True},
            {"campaign_variant_name": "d", "risk_pct": 0.0040, "max_contracts": 6, "oos_net_pnl_usd": 1, "oos_sharpe": 1.0, "oos_max_drawdown_usd": 100, "oos_prop_score": 7.0, "is_top_quartile_prop_score": True},
            {"campaign_variant_name": "e", "risk_pct": 0.0030, "max_contracts": 5, "oos_net_pnl_usd": 1, "oos_sharpe": 1.0, "oos_max_drawdown_usd": 100, "oos_prop_score": 1.0, "is_top_quartile_prop_score": False},
        ]
    )

    clustered, summary = identify_connected_clusters(frame, eligible_column="is_top_quartile_prop_score", connectivity=8)

    assert summary["cluster_size"].tolist() == [3, 1]
    assert int(clustered.loc[clustered["campaign_variant_name"] == "a", "cluster_id"].iloc[0]) == 1
    assert int(clustered.loc[clustered["campaign_variant_name"] == "c", "cluster_id"].iloc[0]) == 1
    assert int(clustered.loc[clustered["campaign_variant_name"] == "d", "cluster_id"].iloc[0]) == 2


def test_refinement_campaign_smoke_outputs_requested_files(tmp_path: Path) -> None:
    dataset_path = tmp_path / "MNQ_c_0_1m_smoke.parquet"
    _write_synthetic_minute_dataset(dataset_path)
    base_variant_name = build_volume_climax_pullback_v3_variants("MNQ")[0].name

    output_dir = run_campaign(
        output_root=tmp_path / "exports",
        input_path=dataset_path,
        initial_capital_usd=50_000.0,
        risk_pcts=(0.0025, 0.0030),
        max_contracts_grid=(2, 3),
        base_variant_name=base_variant_name,
    )

    expected_files = {
        "summary_by_variant.csv",
        "summary_oos_only.csv",
        "trades_by_variant.csv",
        "daily_equity_by_variant.csv",
        "prop_constraints_summary.csv",
        "heatmap_metrics.csv",
        "robustness_zone_summary.csv",
        "final_report.md",
        "heatmap_oos_net_pnl.png",
        "heatmap_oos_sharpe.png",
        "heatmap_oos_maxdd_usd.png",
        "heatmap_oos_prop_score.png",
        "robustness_cluster_map.png",
    }
    assert expected_files.issubset({path.name for path in output_dir.iterdir()})

    summary = pd.read_csv(output_dir / "summary_by_variant.csv")
    assert set(summary["variant_role"]) == {"baseline", "grid", "best_previous_winner"}
    assert len(summary) == 6

    heatmap = pd.read_csv(output_dir / "heatmap_metrics.csv")
    assert len(heatmap) == 4
    assert set(heatmap["max_contracts"]) == {2, 3}
    assert set(heatmap["risk_pct"].round(4)) == {0.0025, 0.0030}
