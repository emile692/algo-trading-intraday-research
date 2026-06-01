from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analytics.build_mnq_orb_strategy_memo import MemoBuildConfig, build_strategy_memo


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_mock_exports(tmp_path: Path) -> tuple[Path, Path, Path]:
    variant_export = tmp_path / "variant_export"
    audit_export = tmp_path / "audit_export"
    regime_export = tmp_path / "regime_export"
    for path in (variant_export, audit_export, regime_export):
        path.mkdir(parents=True, exist_ok=True)

    variant_metadata = {
        "reference_export_root": str(regime_export),
        "dataset_path": str(tmp_path / "unused_dataset.parquet"),
        "spec": {
            "baseline": {
                "or_minutes": 30,
                "opening_time": "09:30:00",
                "direction": "both",
                "one_trade_per_day": True,
                "entry_buffer_ticks": 2,
                "stop_buffer_ticks": 2,
                "target_multiple": 2.0,
                "vwap_confirmation": True,
                "vwap_column": "continuous_session_vwap",
                "time_exit": "16:00:00",
                "account_size_usd": 50000.0,
                "risk_per_trade_pct": 1.5,
                "entry_on_next_open": True,
            }
        },
    }
    (variant_export / "run_metadata.json").write_text(json.dumps(variant_metadata), encoding="utf-8")
    (audit_export / "run_metadata.json").write_text(
        json.dumps({"final_recommendation": "Stay with single_15_60."}),
        encoding="utf-8",
    )
    (regime_export / "run_metadata.json").write_text(
        json.dumps({"spec": {"min_bucket_obs_is": 50}, "dataset_path": str(tmp_path / "unused_dataset.parquet")}),
        encoding="utf-8",
    )

    variant_rows = []
    for idx, name in enumerate(
        [
            "single_15_60",
            "single_14_60",
            "single_16_60",
            "single_15_70",
            "single_15_80",
            "single_16_75",
            "median_fast15_slow_60_70_80",
            "median_plateau_compact",
        ]
    ):
        variant_rows.append(
            {
                "variant_name": name,
                "net_pnl": 10000.0 + idx * 250.0,
                "sharpe": 1.5 + idx * 0.05,
                "sortino": 1.4 + idx * 0.05,
                "max_drawdown": -2000.0 - idx * 25.0,
                "max_daily_loss": -500.0,
                "profit_factor": 1.2,
                "win_rate": 0.56,
                "num_trades": 120,
                "avg_trade_pnl": 83.3,
                "prop_pass": True,
                "pass_prop_constraints": True,
                "daily_loss_limit_breach_freq": 0.0,
                "profit_target_reached_before_max_loss": True,
                "max_loss_limit_buffer_usd": 300.0,
                "is_net_pnl": 4000.0,
                "is_sharpe": 0.8,
                "is_sortino": 0.75,
                "delta_sharpe_vs_single_15_60": 0.0 if name == "single_15_60" else idx * 0.05,
                "delta_net_pnl_vs_single_15_60": 0.0 if name == "single_15_60" else idx * 250.0,
                "delta_maxdd_vs_single_15_60": 0.0 if name == "single_15_60" else -idx * 25.0,
                "abs_max_drawdown": abs(-2000.0 - idx * 25.0),
            }
        )
    _write_csv(variant_export / "variant_summary.csv", variant_rows)

    _write_csv(
        variant_export / "variant_daily_returns.csv",
        [
            {
                "session_date": "2025-01-02",
                "daily_pnl_usd": 200.0,
                "daily_gross_pnl_usd": 205.0,
                "daily_fees_usd": 5.0,
                "daily_trade_count": 1,
                "daily_loss_count": 0,
                "equity": 50200.0,
                "peak_equity": 50200.0,
                "drawdown_usd": 0.0,
                "drawdown_pct": 0.0,
                "green_day": True,
                "variant_name": "single_15_60",
                "phase": "oos",
                "daily_return": 0.004,
            },
            {
                "session_date": "2025-01-03",
                "daily_pnl_usd": -100.0,
                "daily_gross_pnl_usd": -95.0,
                "daily_fees_usd": 5.0,
                "daily_trade_count": 1,
                "daily_loss_count": 1,
                "equity": 50100.0,
                "peak_equity": 50200.0,
                "drawdown_usd": -100.0,
                "drawdown_pct": -0.001992,
                "green_day": False,
                "variant_name": "single_15_60",
                "phase": "oos",
                "daily_return": -0.002,
            },
            {
                "session_date": "2025-02-03",
                "daily_pnl_usd": 150.0,
                "daily_gross_pnl_usd": 155.0,
                "daily_fees_usd": 5.0,
                "daily_trade_count": 1,
                "daily_loss_count": 0,
                "equity": 50250.0,
                "peak_equity": 50250.0,
                "drawdown_usd": 0.0,
                "drawdown_pct": 0.0,
                "green_day": True,
                "variant_name": "single_15_60",
                "phase": "oos",
                "daily_return": 0.003,
            },
        ],
    )

    _write_csv(
        variant_export / "variant_trade_summary.csv",
        [
            {
                "trade_id": 1,
                "session_date": "2025-01-02",
                "direction": "long",
                "quantity": 2,
                "entry_time": "2025-01-02 10:01:00-05:00",
                "entry_price": 20000.0,
                "stop_price": 19900.0,
                "target_price": 20200.0,
                "exit_time": "2025-01-02 12:00:00-05:00",
                "exit_price": 20100.0,
                "exit_reason": "time_exit",
                "account_size_usd": 50000.0,
                "risk_per_trade_pct": 1.5,
                "risk_budget_usd": 750.0,
                "risk_per_contract_usd": 125.0,
                "actual_risk_usd": 250.0,
                "trade_risk_usd": 250.0,
                "notional_usd": 80000.0,
                "leverage_used": 1.6,
                "pnl_points": 100.0,
                "pnl_ticks": 400.0,
                "pnl_usd": 200.0,
                "fees": 5.0,
                "net_pnl_usd": 195.0,
                "risk_multiplier_trade": 1.0,
                "variant_name": "single_15_60",
                "phase": "oos",
                "bucket_label": "mid",
                "risk_multiplier_control": 1.0,
                "risk_multiplier": 1.0,
            },
            {
                "trade_id": 2,
                "session_date": "2025-01-03",
                "direction": "short",
                "quantity": 1,
                "entry_time": "2025-01-03 10:02:00-05:00",
                "entry_price": 20100.0,
                "stop_price": 20220.0,
                "target_price": 19860.0,
                "exit_time": "2025-01-03 11:30:00-05:00",
                "exit_price": 20220.0,
                "exit_reason": "stop",
                "account_size_usd": 50000.0,
                "risk_per_trade_pct": 0.5,
                "risk_budget_usd": 250.0,
                "risk_per_contract_usd": 125.0,
                "actual_risk_usd": 125.0,
                "trade_risk_usd": 125.0,
                "notional_usd": 40200.0,
                "leverage_used": 0.804,
                "pnl_points": -120.0,
                "pnl_ticks": -480.0,
                "pnl_usd": -120.0,
                "fees": 2.5,
                "net_pnl_usd": -122.5,
                "risk_multiplier_trade": 0.5,
                "variant_name": "single_15_60",
                "phase": "oos",
                "bucket_label": "low",
                "risk_multiplier_control": 0.5,
                "risk_multiplier": 0.5,
            },
            {
                "trade_id": 3,
                "session_date": "2025-02-03",
                "direction": "long",
                "quantity": 1,
                "entry_time": "2025-02-03 10:03:00-05:00",
                "entry_price": 20250.0,
                "stop_price": 20130.0,
                "target_price": 20490.0,
                "exit_time": "2025-02-03 14:00:00-05:00",
                "exit_price": 20325.0,
                "exit_reason": "time_exit",
                "account_size_usd": 50000.0,
                "risk_per_trade_pct": 0.25,
                "risk_budget_usd": 125.0,
                "risk_per_contract_usd": 125.0,
                "actual_risk_usd": 125.0,
                "trade_risk_usd": 125.0,
                "notional_usd": 40500.0,
                "leverage_used": 0.81,
                "pnl_points": 75.0,
                "pnl_ticks": 300.0,
                "pnl_usd": 150.0,
                "fees": 2.5,
                "net_pnl_usd": 147.5,
                "risk_multiplier_trade": 0.25,
                "variant_name": "single_15_60",
                "phase": "oos",
                "bucket_label": "high",
                "risk_multiplier_control": 0.25,
                "risk_multiplier": 0.25,
            },
        ],
    )

    _write_csv(
        variant_export / "variant_bucket_contribution.csv",
        [
            {"variant_name": "single_15_60", "phase": "oos", "bucket_label": "low", "risk_multiplier": 0.5, "num_trades": 1, "net_pnl": -122.5, "avg_trade_pnl": -122.5, "win_rate": 0.0},
            {"variant_name": "single_15_60", "phase": "oos", "bucket_label": "mid", "risk_multiplier": 1.0, "num_trades": 1, "net_pnl": 195.0, "avg_trade_pnl": 195.0, "win_rate": 1.0},
            {"variant_name": "single_15_60", "phase": "oos", "bucket_label": "high", "risk_multiplier": 0.25, "num_trades": 1, "net_pnl": 147.5, "avg_trade_pnl": 147.5, "win_rate": 1.0},
        ],
    )

    _write_csv(
        audit_export / "variant_audit_summary.csv",
        [
            {
                "variant_name": "single_15_60",
                "net_pnl": 10000.0,
                "sharpe": 2.2,
                "sortino": 2.0,
                "max_drawdown": -2000.0,
                "positive_diff_day_share": 0.0,
                "positive_diff_month_share": 0.0,
                "excess_pnl_after_top_5": 0.0,
            },
            {
                "variant_name": "median_fast15_slow_60_70_80",
                "net_pnl": 10250.0,
                "sharpe": 2.3,
                "sortino": 2.1,
                "max_drawdown": -2050.0,
                "positive_diff_day_share": 0.52,
                "positive_diff_month_share": 0.30,
                "excess_pnl_after_top_5": -200.0,
            },
            {
                "variant_name": "median_plateau_compact",
                "net_pnl": 10500.0,
                "sharpe": 2.5,
                "sortino": 2.3,
                "max_drawdown": -2200.0,
                "positive_diff_day_share": 0.531,
                "positive_diff_month_share": 0.241,
                "excess_pnl_after_top_5": -4229.5,
            },
        ],
    )

    _write_csv(
        audit_export / "monthly_returns_by_variant.csv",
        [
            {"variant_name": "single_15_60", "month": "2025-01", "month_start": "2025-01-01", "month_end": "2025-01-31", "monthly_pnl_usd": 100.0, "monthly_return": 0.002, "positive_days": 1, "negative_days": 1, "trading_days": 2, "traded_days": 2, "positive_month": True},
            {"variant_name": "single_15_60", "month": "2025-02", "month_start": "2025-02-01", "month_end": "2025-02-28", "monthly_pnl_usd": 150.0, "monthly_return": 0.003, "positive_days": 1, "negative_days": 0, "trading_days": 1, "traded_days": 1, "positive_month": True},
        ],
    )

    _write_csv(
        audit_export / "baseline_vs_ensemble_daily_diff.csv",
        [
            {"session_date": "2025-01-02", "variant_name": "median_plateau_compact", "daily_pnl_diff": 600.0},
            {"session_date": "2025-01-03", "variant_name": "median_plateau_compact", "daily_pnl_diff": -300.0},
            {"session_date": "2025-02-03", "variant_name": "median_plateau_compact", "daily_pnl_diff": 200.0},
            {"session_date": "2025-02-04", "variant_name": "median_plateau_compact", "daily_pnl_diff": -50.0},
            {"session_date": "2025-02-05", "variant_name": "median_plateau_compact", "daily_pnl_diff": 100.0},
            {"session_date": "2025-02-06", "variant_name": "median_plateau_compact", "daily_pnl_diff": -500.0},
        ],
    )

    _write_csv(
        audit_export / "bucket_distribution_by_variant.csv",
        [
            {"variant_name": "single_15_60", "bucket_label": "low", "day_count": 1, "pct_days": 0.33, "trade_count": 1, "pct_trades": 0.33},
            {"variant_name": "single_15_60", "bucket_label": "mid", "day_count": 1, "pct_days": 0.33, "trade_count": 1, "pct_trades": 0.33},
            {"variant_name": "single_15_60", "bucket_label": "high", "day_count": 1, "pct_days": 0.33, "trade_count": 1, "pct_trades": 0.33},
        ],
    )
    _write_csv(
        audit_export / "worst_periods_by_variant.csv",
        [{"variant_name": "single_15_60", "period_type": "day", "period_label": "2025-01-03", "period_start": "2025-01-03", "period_end": "2025-01-03", "period_value": -122.5}],
    )
    _write_csv(
        audit_export / "multiplier_transition_by_variant.csv",
        [{"variant_name": "single_15_60", "transition_from": "start", "transition_to": "mid", "transition_count": 1, "transition_pct": 1.0, "bucket_switches": 2, "bucket_switch_rate": 1.0, "avg_multiplier": 0.583, "median_multiplier": 0.5, "multiplier_corr_vs_single_15_60": 1.0}],
    )

    _write_csv(
        regime_export / "feature_ranking.csv",
        [
            {
                "feature_name": "realized_vol_ratio_15_60",
                "family": "volatility",
                "bucket_kind": "quantile",
                "bucket_count": 3,
                "min_bucket_obs_is": 50,
                "balance_is": 1.0,
                "is_score_spread": 4.935254,
                "feature_selection_score": 4.935254,
                "best_bucket_is": "mid",
                "worst_bucket_is": "low",
                "skip_coverage_is": 0.666667,
                "valid_for_overlay": True,
            }
        ],
    )
    _write_csv(
        regime_export / "conditional_bucket_analysis.csv",
        [
            {"feature_name": "realized_vol_ratio_15_60", "bucket_label": "low", "bucket_position": 1, "lower_bound": 0.33, "upper_bound": 0.9429454367121718, "is_n_obs": 60, "is_net_pnl": -1000.0, "is_sharpe": -0.4, "oos_n_obs": 30, "oos_net_pnl": 500.0, "oos_sharpe": 1.4},
            {"feature_name": "realized_vol_ratio_15_60", "bucket_label": "mid", "bucket_position": 2, "lower_bound": 0.9429454367121718, "upper_bound": 1.1407799880685539, "is_n_obs": 60, "is_net_pnl": 3000.0, "is_sharpe": 1.7, "oos_n_obs": 30, "oos_net_pnl": 2500.0, "oos_sharpe": 3.2},
            {"feature_name": "realized_vol_ratio_15_60", "bucket_label": "high", "bucket_position": 3, "lower_bound": 1.1407799880685539, "upper_bound": 1.82, "is_n_obs": 60, "is_net_pnl": -750.0, "is_sharpe": -0.3, "oos_n_obs": 30, "oos_net_pnl": 100.0, "oos_sharpe": 0.2},
        ],
    )
    _write_csv(
        regime_export / "summary_variants.csv",
        [
            {
                "variant_name": "sizing_3state_realized_vol_ratio_15_60",
                "oos_sharpe": 2.1,
                "oos_max_drawdown": -1500.0,
            }
        ],
    )

    return variant_export, audit_export, regime_export


def test_build_strategy_memo_creates_core_outputs_and_sections(tmp_path: Path) -> None:
    variant_export, audit_export, regime_export = _build_mock_exports(tmp_path)
    output_dir = tmp_path / "docs"

    artifacts = build_strategy_memo(
        MemoBuildConfig(
            output_dir=output_dir,
            variant_export=variant_export,
            audit_export=audit_export,
            regime_export=regime_export,
            include_stability_heatmaps=False,
        )
    )

    markdown_text = artifacts.markdown_path.read_text(encoding="utf-8")
    html_text = artifacts.html_path.read_text(encoding="utf-8")

    assert artifacts.markdown_path.exists()
    assert artifacts.html_path.exists()
    assert "# MNQ ORB Strategy - Institutional Research Memo" in markdown_text
    assert "## 1. Executive Summary" in markdown_text
    assert "## 4. Where VWAP and ATR Live" in markdown_text
    assert "## 6. Separate 3-State Sizing Branch" in markdown_text
    assert "## 11. Production Recommendation" in markdown_text
    assert "single_15_60" in markdown_text
    assert "15/60" in markdown_text
    assert "low=0.50x" in markdown_text
    assert "mid=1.00x" in markdown_text
    assert "high=0.25x" in markdown_text
    assert "OR window: `15` minutes" in markdown_text
    assert "- Direction: `long`" in markdown_text
    assert "The implemented retained sleeve is **not** the 30-minute both-direction baseline." in markdown_text
    assert "This is where the previous memo drifted" in markdown_text
    assert "retained final" in markdown_text
    assert "VWAP" in markdown_text
    assert "ATR" in markdown_text
    assert "<html" in html_text.lower()
    assert "MNQ ORB Strategy - Institutional Research Memo" in html_text

    produced = {path.name for path in artifacts.figure_paths}
    assert "orb_mechanics_diagram.png" in produced
    assert "variant_sharpe_comparison.png" in produced
    assert "variant_net_pnl_comparison.png" in produced
    assert "variant_maxdd_comparison.png" in produced
    assert "monthly_returns_single_15_60.png" in produced
    assert "baseline_vs_ensemble_excess_contribution.png" in produced
