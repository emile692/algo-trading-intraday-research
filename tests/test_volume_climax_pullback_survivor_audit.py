from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics import volume_climax_pullback_survivor_audit as audit


TZ = "America/New_York"


def _events(symbol: str, pnls: list[float]) -> pd.DataFrame:
    rows = []
    for index, pnl in enumerate(pnls, start=1):
        ts = pd.Timestamp(f"2024-01-{index:02d} 10:31:00", tz=TZ)
        rows.append(
            {
                "symbol": symbol,
                "signal_timeframe": "1H",
                "config_id": f"{symbol.lower()}_cfg",
                "signal_time": ts - pd.Timedelta(hours=1),
                "entry_time": ts,
                "exit_time": ts + pd.Timedelta(minutes=15),
                "direction": "long" if index % 2 else "short",
                "entry_price": 100.0,
                "stop_price": 99.0,
                "target_price": 102.0,
                "exit_price": 100.0 + pnl,
                "net_pnl_usd": float(pnl),
                "gross_pnl_usd": float(pnl),
                "holding_minutes": 15.0,
                "session_date": ts.date(),
                "executed": True,
                "skipped_by_filter": False,
                "filter_reason": "none",
                "exit_reason": "target_1m" if pnl > 0 else "stop_1m",
                "quantity": 1,
            }
        )
    return pd.DataFrame(rows)


def test_strict_train_only_selection_does_not_use_oos_data() -> None:
    ranking = pd.DataFrame(
        [
            {
                "config_id": "train_good_test_bad",
                "train_robust_score": 0.9,
                "train_net_pnl": 100.0,
                "train_profit_factor": 1.3,
                "train_avg_trade": 5.0,
            },
            {
                "config_id": "train_bad_test_good",
                "train_robust_score": 0.2,
                "train_net_pnl": -10.0,
                "train_profit_factor": 0.8,
                "train_avg_trade": -1.0,
            },
        ]
    )

    winner = audit.select_fold_winner(ranking)

    assert str(winner["config_id"]) == "train_good_test_bad"


def test_posthoc_diagnostic_portfolios_are_labeled_non_deployable() -> None:
    summary = pd.DataFrame(
        [
            {"symbol": "M2K", "signal_timeframe": "1H", "total_test_net_pnl": 100.0, "test_profit_factor": 1.2, "positive_folds": 3},
            {"symbol": "MGC", "signal_timeframe": "1H", "total_test_net_pnl": -50.0, "test_profit_factor": 0.9, "positive_folds": 1},
        ]
    )

    rows = audit.build_diagnostic_posthoc_rows(summary)

    assert not rows.empty
    assert bool(rows["deployable"].eq(False).all())


def test_portfolio_weights_are_computed_from_train_data_only() -> None:
    train_daily = {
        "M2K": pd.DataFrame({"session_date": ["2024-01-02", "2024-01-03"], "daily_pnl": [1.0, 1.5]}),
        "MGC": pd.DataFrame({"session_date": ["2024-01-02", "2024-01-03"], "daily_pnl": [10.0, -10.0]}),
    }

    weights = audit.build_portfolio_weights_from_train(train_daily, scheme="inverse_vol")

    assert set(weights) == {"M2K", "MGC"}
    assert float(weights["M2K"]) > float(weights["MGC"])


def test_local_parameter_neighborhood_generation_is_deterministic() -> None:
    first = audit.build_candidate_universe(
        symbols=["M2K", "MGC", "MNQ"],
        signal_timeframe="1H",
        execution_timeframe="1min",
        include_negative_control=True,
        max_configs_per_family=8,
    )
    second = audit.build_candidate_universe(
        symbols=["M2K", "MGC", "MNQ"],
        signal_timeframe="1H",
        execution_timeframe="1min",
        include_negative_control=True,
        max_configs_per_family=8,
    )

    assert [config.config_id for config in first] == [config.config_id for config in second]
    assert "m2k_1h_adverse_core" in {config.cluster_id for config in first}
    assert "mgc_1h_stop_zone_core" in {config.cluster_id for config in first}


def test_verdict_logic_works_for_all_levels() -> None:
    assert audit.derive_survivor_verdict(
        net_pnl=-1.0,
        profit_factor=0.9,
        positive_folds=1,
        trades=10,
        cluster_positive_ratio=0.1,
        max_drawdown=-100.0,
        monthly_positive_ratio=0.1,
    ) == "reject"
    assert audit.derive_survivor_verdict(
        net_pnl=100.0,
        profit_factor=1.16,
        positive_folds=3,
        trades=12,
        cluster_positive_ratio=0.3,
        max_drawdown=-200.0,
        monthly_positive_ratio=0.4,
    ) == "weak_watchlist"
    assert audit.derive_survivor_verdict(
        net_pnl=300.0,
        profit_factor=1.25,
        positive_folds=4,
        trades=25,
        cluster_positive_ratio=0.6,
        max_drawdown=-300.0,
        monthly_positive_ratio=0.6,
    ) == "watchlist"
    assert audit.derive_survivor_verdict(
        net_pnl=500.0,
        profit_factor=1.30,
        positive_folds=4,
        trades=40,
        cluster_positive_ratio=0.6,
        max_drawdown=-250.0,
        monthly_positive_ratio=0.6,
        improvement_vs_m2k_only=50.0,
        is_portfolio=True,
    ) == "candidate"


def test_exports_are_created_with_expected_columns(tmp_path: Path, monkeypatch) -> None:
    source_file = tmp_path / "source.parquet"
    source_file.write_text("x", encoding="utf-8")

    def _fake_evaluate_symbol(*, symbol: str, signal_timeframe: str, execution_timeframe: str, configs, output_dir: Path, raw_minute_df_override=None):
        trades = _events(symbol, [50.0, -10.0, 20.0])
        daily = pd.DataFrame({"session_date": ["2024-01-02", "2024-01-03"], "daily_pnl": [40.0, 20.0]})
        selection = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "fold_id": "fold_1",
                    "config_id": f"{symbol.lower()}_cfg",
                    "family": "delay_adverse_filter",
                    "cluster_id": f"{symbol.lower()}_cluster",
                    "filter_name": "avoid_immediate_adverse_move",
                    "stop_multiplier": 0.75,
                    "target_multiplier": 2.0,
                    "entry_delay_minutes": 15,
                    "variant_time_stop_bars": 4,
                    "stop_zone_fraction": pd.NA,
                    "adverse_window_minutes": 5,
                    "max_adverse_ticks": 8,
                    "train_trades": 40,
                    "train_net_pnl": 100.0,
                    "train_profit_factor": 1.3,
                    "train_avg_trade": 2.5,
                    "train_robust_score": 0.8,
                    "rank_train": 1,
                    "selected_in_fold": True,
                }
            ]
        )
        fold_breakdown = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "signal_timeframe": signal_timeframe,
                    "fold_id": "fold_1",
                    "selected_config_id": f"{symbol.lower()}_cfg",
                    "selected_family": "delay_adverse_filter",
                    "selected_cluster_id": f"{symbol.lower()}_cluster",
                    "train_robust_score": 0.8,
                    "train_net_pnl": 100.0,
                    "train_profit_factor": 1.3,
                    "train_trades": 40,
                    "test_net_pnl": 60.0,
                    "test_profit_factor": 1.4,
                    "test_trades": 3,
                    "test_win_rate": 0.66,
                    "test_avg_trade": 20.0,
                    "test_median_trade": 20.0,
                    "test_max_drawdown": -10.0,
                    "test_max_daily_drawdown": -10.0,
                    "test_fold_sharpe": 1.0,
                    "test_active_days": 2,
                    "test_exposure_ratio": 1.0,
                    "test_avg_holding_minutes": 15.0,
                    "raw_hybrid_test_net_pnl": 20.0,
                    "strict_train_only": True,
                }
            ]
        )
        strict_summary = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "signal_timeframe": signal_timeframe,
                    "total_test_trades": 3,
                    "total_test_net_pnl": 60.0 if symbol != "MGC" else 30.0,
                    "gross_profit": 70.0,
                    "gross_loss": 10.0,
                    "test_profit_factor": 1.4 if symbol != "MGC" else 1.15,
                    "test_win_rate": 0.66,
                    "avg_trade": 20.0,
                    "median_trade": 20.0,
                    "max_drawdown": -10.0,
                    "max_daily_drawdown": -10.0,
                    "positive_folds": 3 if symbol != "MNQ" else 1,
                    "fold_count": 5,
                    "fold_sharpe": 1.0,
                    "active_days": 2,
                    "exposure_ratio": 1.0,
                    "avg_holding_minutes": 15.0,
                    "long_trades": 2,
                    "short_trades": 1,
                    "long_pnl": 70.0,
                    "short_pnl": -10.0,
                    "monthly_positive_ratio": 1.0,
                    "cluster_positive_ratio": 0.6 if symbol != "MNQ" else 0.1,
                    "selected_family_counts": "{\"delay_adverse_filter\": 1}",
                    "selected_cluster_counts": "{\"cluster\": 1}",
                    "train_score_test_corr": 0.5,
                    "benchmark_raw_hybrid_net_pnl_same_windows": 20.0,
                    "improvement_vs_raw_hybrid": 40.0 if symbol != "MGC" else 10.0,
                    "verdict": "watchlist" if symbol != "MNQ" else "reject",
                }
            ]
        )
        local = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "config_id": f"{symbol.lower()}_cfg",
                    "family": "delay_adverse_filter",
                    "cluster_id": f"{symbol.lower()}_cluster",
                    "net_pnl": 100.0,
                    "profit_factor": 1.3,
                    "rank_is": 1,
                    "net_pnl_oos": 60.0,
                    "profit_factor_oos": 1.4,
                    "rank_oos": 1,
                    "fixed_wfa_net_pnl": 60.0,
                    "fixed_positive_fold_ratio": 1.0,
                    "strict_selected_count": 1,
                    "neighbor_median_oos_pnl": 55.0,
                    "neighbor_positive_fold_ratio": 1.0,
                }
            ]
        )
        cluster = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "cluster_id": f"{symbol.lower()}_cluster",
                    "family": "delay_adverse_filter",
                    "configs": 1,
                    "median_is_net_pnl": 100.0,
                    "median_oos_net_pnl": 60.0,
                    "median_fixed_wfa_net_pnl": 60.0,
                    "pct_configs_positive_oos": 1.0,
                    "pct_configs_positive_fixed_wfa": 1.0,
                    "median_neighbor_oos_pnl": 55.0,
                    "selected_in_any_fold": 1,
                }
            ]
        )
        fixed = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "signal_timeframe": signal_timeframe,
                    "fold_id": "fold_1",
                    "config_id": f"{symbol.lower()}_cfg",
                    "family": "delay_adverse_filter",
                    "cluster_id": f"{symbol.lower()}_cluster",
                    "test_net_pnl": 60.0,
                    "test_profit_factor": 1.4,
                    "test_trades": 3,
                    "positive_test_fold": True,
                }
            ]
        )
        return {
            "symbol": symbol,
            "config_frame": pd.DataFrame([{"config_id": f"{symbol.lower()}_cfg"}]),
            "metrics_is": pd.DataFrame(),
            "metrics_oos": pd.DataFrame(),
            "metrics_full": pd.DataFrame(),
            "robustness": pd.DataFrame(),
            "local_stability": local,
            "cluster_stability_summary": cluster,
            "config_selection_by_fold": selection,
            "strict_wfa_fold_breakdown": fold_breakdown,
            "strict_wfa_summary": strict_summary,
            "strict_wfa_stitched_trades": trades,
            "strict_wfa_stitched_daily": daily,
            "raw_benchmark_daily": daily,
            "train_daily_by_fold_symbol": {("fold_1", symbol): daily},
            "test_daily_by_fold_symbol": {("fold_1", symbol): daily, ("fold_1_raw", symbol): daily},
            "fixed_fold_frame": fixed,
            "data_audit": {
                "symbol": symbol,
                "signal_timeframe": signal_timeframe,
                "execution_timeframe": execution_timeframe,
                "source_path": str(source_file),
            },
        }

    monkeypatch.setattr(audit, "evaluate_symbol", _fake_evaluate_symbol)

    output_dir = audit.run_campaign(
        symbols=["M2K", "MGC", "MNQ"],
        signal_timeframe="1H",
        execution_timeframe="1min",
        output_root=tmp_path,
        smoke=True,
        dataset_overrides={},
    )

    summary = pd.read_csv(output_dir / "strict_wfa_summary.csv")
    portfolio = pd.read_csv(output_dir / "strict_portfolio_summary.csv")
    diagnostics = pd.read_csv(output_dir / "rejected_or_diagnostic_results.csv")

    assert {"symbol", "total_test_net_pnl", "test_profit_factor", "verdict"}.issubset(summary.columns)
    assert {"portfolio_name", "selection_basis", "deployable", "verdict"}.issubset(portfolio.columns)
    assert {"result_type", "deployable", "reason"}.issubset(diagnostics.columns)
    assert (output_dir / "final_report.md").exists()


def test_campaign_runs_in_smoke_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(audit, "evaluate_symbol", lambda **kwargs: {
        "symbol": kwargs["symbol"],
        "config_frame": pd.DataFrame([{"config_id": f"{kwargs['symbol'].lower()}_cfg"}]),
        "metrics_is": pd.DataFrame(),
        "metrics_oos": pd.DataFrame(),
        "metrics_full": pd.DataFrame(),
        "robustness": pd.DataFrame(),
        "local_stability": pd.DataFrame(columns=["symbol", "config_id", "family", "cluster_id", "net_pnl", "net_pnl_oos", "fixed_positive_fold_ratio"]),
        "cluster_stability_summary": pd.DataFrame(columns=["symbol", "cluster_id", "family"]),
        "config_selection_by_fold": pd.DataFrame(columns=["fold_id", "config_id", "train_robust_score", "rank_train", "selected_in_fold"]),
        "strict_wfa_fold_breakdown": pd.DataFrame(columns=["fold_id", "selected_config_id", "test_net_pnl"]),
        "strict_wfa_summary": pd.DataFrame([{
            "symbol": kwargs["symbol"],
            "signal_timeframe": "1H",
            "total_test_trades": 0,
            "total_test_net_pnl": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "test_profit_factor": 0.0,
            "test_win_rate": 0.0,
            "avg_trade": 0.0,
            "median_trade": 0.0,
            "max_drawdown": 0.0,
            "max_daily_drawdown": 0.0,
            "positive_folds": 0,
            "fold_count": 0,
            "fold_sharpe": pd.NA,
            "active_days": 0,
            "exposure_ratio": 0.0,
            "avg_holding_minutes": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "long_pnl": 0.0,
            "short_pnl": 0.0,
            "monthly_positive_ratio": 0.0,
            "cluster_positive_ratio": 0.0,
            "selected_family_counts": "{}",
            "selected_cluster_counts": "{}",
            "train_score_test_corr": pd.NA,
            "benchmark_raw_hybrid_net_pnl_same_windows": 0.0,
            "improvement_vs_raw_hybrid": 0.0,
            "verdict": "reject",
        }]),
        "strict_wfa_stitched_trades": pd.DataFrame(columns=["symbol", "executed", "entry_time", "net_pnl_usd", "session_date", "direction", "holding_minutes"]),
        "strict_wfa_stitched_daily": pd.DataFrame(columns=["session_date", "daily_pnl"]),
        "raw_benchmark_daily": pd.DataFrame(columns=["session_date", "daily_pnl"]),
        "train_daily_by_fold_symbol": {},
        "test_daily_by_fold_symbol": {},
        "fixed_fold_frame": pd.DataFrame(columns=["config_id"]),
        "data_audit": {"symbol": kwargs["symbol"], "signal_timeframe": "1H", "execution_timeframe": "1min", "source_path": str(tmp_path / "source.parquet")},
    })
    (tmp_path / "source.parquet").write_text("x", encoding="utf-8")

    output_dir = audit.run_campaign(
        symbols=["M2K", "MGC"],
        signal_timeframe="1H",
        execution_timeframe="1min",
        output_root=tmp_path,
        smoke=True,
        dataset_overrides={},
    )

    assert output_dir.exists()
    assert (output_dir / "strict_wfa_summary.csv").exists()
    assert (output_dir / "run_metadata.json").exists()
