from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.mnq_orb_3state_vol_sizing_variant_smoke import (
    MnqOrb3StateVolSizingVariantSmokeSpec,
    add_variant_ratio_columns,
    build_variant_specs,
    run_smoke_campaign,
)
from src.analytics.orb_multi_asset_campaign import BaselineSpec, SearchGrid
from src.config.orb_campaign import PropConstraintConfig


def test_variant_ratio_construction_matches_baseline_and_rowwise_median() -> None:
    frame = pd.DataFrame(
        {
            "vol_std_14": [1.4, 2.8, 4.2],
            "vol_std_15": [1.5, 3.0, 4.5],
            "vol_std_16": [1.6, 3.2, 4.8],
            "vol_std_60": [6.0, 6.0, 6.0],
            "vol_std_70": [7.0, 7.0, 7.0],
            "vol_std_75": [7.5, 7.5, 7.5],
            "vol_std_80": [8.0, 8.0, 8.0],
        }
    )

    enriched = add_variant_ratio_columns(frame, build_variant_specs())

    expected_single = pd.Series([0.25, 0.50, 0.75], name="single_15_60")
    pd.testing.assert_series_equal(enriched["single_15_60"], expected_single, check_names=False)

    manual_median = pd.DataFrame(
        {
            "a": frame["vol_std_15"] / frame["vol_std_60"],
            "b": frame["vol_std_15"] / frame["vol_std_70"],
            "c": frame["vol_std_15"] / frame["vol_std_80"],
        }
    ).median(axis=1)
    pd.testing.assert_series_equal(
        enriched["median_fast15_slow_60_70_80"],
        manual_median,
        check_names=False,
    )


def test_median_variants_do_not_depend_on_future_rows() -> None:
    base = pd.DataFrame(
        {
            "vol_std_14": [1.4, 2.8, 4.2],
            "vol_std_15": [1.5, 3.0, 4.5],
            "vol_std_16": [1.6, 3.2, 4.8],
            "vol_std_60": [6.0, 6.0, 6.0],
            "vol_std_70": [7.0, 7.0, 7.0],
            "vol_std_75": [7.5, 7.5, 7.5],
            "vol_std_80": [8.0, 8.0, 8.0],
        }
    )
    mutated = base.copy()
    mutated.loc[2, ["vol_std_14", "vol_std_15", "vol_std_16", "vol_std_60", "vol_std_70", "vol_std_75", "vol_std_80"]] = [
        999.0,
        999.0,
        999.0,
        999.0,
        999.0,
        999.0,
        999.0,
    ]

    base_out = add_variant_ratio_columns(base, build_variant_specs())
    mutated_out = add_variant_ratio_columns(mutated, build_variant_specs())

    pd.testing.assert_series_equal(
        base_out.loc[:1, "median_fast15_slow_60_70_80"],
        mutated_out.loc[:1, "median_fast15_slow_60_70_80"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        base_out.loc[:1, "median_plateau_compact"],
        mutated_out.loc[:1, "median_plateau_compact"],
        check_names=False,
    )


def _write_synthetic_mnq_dataset(path: Path, *, sessions: int = 22) -> None:
    rows: list[dict[str, object]] = []
    session_dates = pd.bdate_range("2024-01-02", periods=sessions)

    for day_idx, session_date in enumerate(session_dates):
        base = 16000.0 + day_idx * 6.0
        day_vol = 1.5 + (day_idx % 5) * 0.35
        previous_close = base

        for minute_idx in range(120):
            timestamp = pd.Timestamp(session_date.date()).tz_localize("America/New_York") + pd.Timedelta(
                hours=9, minutes=30 + minute_idx
            )
            if minute_idx < 15:
                close = base + ((minute_idx % 5) - 2) * day_vol
            else:
                drift = (minute_idx - 14) * day_vol * (0.18 + 0.03 * (day_idx % 3))
                wobble = ((minute_idx % 7) - 3) * day_vol * 0.22
                if day_idx % 4 == 0:
                    close = base - 2.2 * day_vol - drift + wobble
                else:
                    close = base + 2.2 * day_vol + drift + wobble

            open_price = previous_close
            high = max(open_price, close) + day_vol * 0.30
            low = min(open_price, close) - day_vol * 0.30
            rows.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 1000.0 + 5.0 * day_idx + minute_idx,
                }
            )
            previous_close = close

    pd.DataFrame(rows).to_parquet(path, index=False)


def test_smoke_campaign_exports_expected_files(tmp_path: Path) -> None:
    dataset_path = tmp_path / "MNQ_c_0_1m_synth.parquet"
    _write_synthetic_mnq_dataset(dataset_path)

    reference_dir = tmp_path / "reference" / "mnq_orb_regime_filter_sizing_20260325_150405"
    reference_dir.mkdir(parents=True)
    reference_metadata = {
        "dataset_path": str(dataset_path),
        "selected_aggregation_rule": "majority_50",
        "spec": {
            "is_fraction": 0.70,
            "aggregation_rule": "majority_50",
            "baseline": {
                "or_minutes": 15,
                "opening_time": "09:30:00",
                "direction": "both",
                "one_trade_per_day": True,
                "entry_buffer_ticks": 1,
                "stop_buffer_ticks": 1,
                "target_multiple": 1.5,
                "vwap_confirmation": True,
                "vwap_column": "continuous_session_vwap",
                "time_exit": "16:00:00",
                "account_size_usd": 50_000.0,
                "risk_per_trade_pct": 1.5,
                "entry_on_next_open": True,
            },
            "grid": {
                "atr_periods": [20],
                "q_lows_pct": [10],
                "q_highs_pct": [90],
                "aggregation_rules": ["majority_50"],
            },
            "prop_constraints": {
                "name": "topstep_50k_reference",
                "account_size_usd": 50_000.0,
                "max_loss_limit_usd": 2_000.0,
                "daily_loss_limit_usd": 1_000.0,
                "profit_target_usd": 3_000.0,
                "monthly_subscription_cost_usd": 150.0,
                "trading_days_per_month": 21.0,
                "daily_loss_limit_basis": "realized_daily_pnl",
            },
        },
    }
    (reference_dir / "run_metadata.json").write_text(json.dumps(reference_metadata, indent=2), encoding="utf-8")

    artifacts = run_smoke_campaign(
        MnqOrb3StateVolSizingVariantSmokeSpec(
            symbol="MNQ",
            dataset_path=dataset_path,
            reference_export_root=reference_dir,
            output_root=tmp_path / "exports",
            is_fraction=0.70,
            baseline=BaselineSpec(**reference_metadata["spec"]["baseline"]),
            grid=SearchGrid(
                atr_periods=(20,),
                q_lows_pct=(10,),
                q_highs_pct=(90,),
                aggregation_rules=("majority_50",),
            ),
            prop_constraints=PropConstraintConfig(**reference_metadata["spec"]["prop_constraints"]),
            min_bucket_obs_is=3,
        )
    )

    output_dir = artifacts["output_dir"]
    assert output_dir.exists()
    assert output_dir.name.startswith("mnq_orb_3state_vol_sizing_variant_smoke_")
    assert (output_dir / "variant_summary.csv").exists()
    assert (output_dir / "variant_daily_returns.csv").exists()
    assert (output_dir / "variant_trade_summary.csv").exists()
    assert (output_dir / "variant_bucket_contribution.csv").exists()
    assert (output_dir / "final_report.md").exists()
    assert (output_dir / "run_metadata.json").exists()

    summary = pd.read_csv(output_dir / "variant_summary.csv")
    assert "single_15_60" in set(summary["variant_name"])
    assert "median_fast15_slow_60_70_80" in set(summary["variant_name"])
    assert "median_plateau_compact" in set(summary["variant_name"])
    assert {
        "variant_name",
        "net_pnl",
        "sharpe",
        "sortino",
        "max_drawdown",
        "max_daily_loss",
        "profit_factor",
        "win_rate",
        "num_trades",
        "avg_trade_pnl",
        "delta_sharpe_vs_single_15_60",
        "delta_net_pnl_vs_single_15_60",
        "delta_maxdd_vs_single_15_60",
    }.issubset(set(summary.columns))

    report_text = (output_dir / "final_report.md").read_text(encoding="utf-8")
    assert "Final verdict" in report_text
    assert "single_15_60" in report_text
    assert "median_fast15_slow_60_70_80" in report_text

    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert str(metadata["selected_symbol"]) == "MNQ"

