from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics import volume_climax_pullback_multiasset_multitimeframe_campaign as campaign


TZ = "America/New_York"


def _minute_df() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:30:00", periods=31, freq="1min", tz=TZ)
    return pd.DataFrame(
        [
            {
                "timestamp": ts,
                "open": float(idx),
                "high": float(idx) + 0.5,
                "low": float(idx) - 0.5,
                "close": float(idx) + 0.25,
                "volume": 100.0,
                "session_date": ts.date(),
            }
            for idx, ts in enumerate(timestamps, start=1)
        ]
    )


def _events(dates: list[str], pnls: list[float], *, symbol: str = "MNQ", timeframe: str = "15min", config_id: str = "cfg") -> pd.DataFrame:
    rows = []
    for raw_date, pnl in zip(dates, pnls, strict=True):
        ts = pd.Timestamp(f"{raw_date} 10:31:00", tz=TZ)
        rows.append(
            {
                "symbol": symbol,
                "signal_timeframe": timeframe,
                "config_id": config_id,
                "signal_time": ts - pd.Timedelta(minutes=15),
                "entry_time": ts,
                "exit_time": ts + pd.Timedelta(minutes=5),
                "direction": "long",
                "entry_price": 100.0,
                "stop_price": 99.0,
                "target_price": 102.0,
                "exit_price": 100.0 + pnl,
                "pnl": float(pnl),
                "net_pnl_usd": float(pnl),
                "gross_pnl_usd": float(pnl),
                "exit_reason": "target_1m" if pnl > 0 else "stop_1m",
                "holding_minutes": 5.0,
                "mfe_ticks_1m": 4.0,
                "mae_ticks_1m": 2.0,
                "skipped_by_filter": False,
                "filter_reason": "none",
                "executed": True,
                "quantity": 1,
                "session_date": ts.date(),
            }
        )
    return pd.DataFrame(rows)


def test_resample_rth_timeframe_no_lookahead() -> None:
    df = _minute_df()
    bars = campaign.resample_rth_timeframe(df, "15min")

    assert len(bars) == 3
    first = bars.iloc[0]
    second = bars.iloc[1]
    assert pd.Timestamp(first["timestamp"]) == pd.Timestamp("2024-01-02 09:30:00", tz=TZ)
    assert float(first["open"]) == 1.0
    assert float(first["close"]) == 15.25
    assert pd.Timestamp(second["timestamp"]) == pd.Timestamp("2024-01-02 09:45:00", tz=TZ)
    assert float(second["open"]) == 16.0


def test_config_universe_contains_all_assets_timeframes() -> None:
    universe = campaign.build_config_universe(["MNQ", "MES", "M2K", "MGC"], ["15min", "30min", "1H"], "1min")
    frame = campaign._config_universe_frame(universe)

    combos = set(zip(frame["symbol"], frame["signal_timeframe"]))
    for symbol in ["MNQ", "MES", "M2K", "MGC"]:
        for timeframe in ["15min", "30min", "1H"]:
            assert (symbol, timeframe) in combos


def test_is_only_selection_ignores_oos() -> None:
    robustness = pd.DataFrame(
        [
            {
                "symbol": "MNQ",
                "signal_timeframe": "15min",
                "config_id": "bad_is_good_oos",
                "family": "delay_only",
                "admissible_is": False,
                "robust_score_is": 0.1,
                "net_pnl": -10.0,
                "profit_factor": 0.9,
                "skip_rate": 0.0,
                "trades": 120,
            },
            {
                "symbol": "MNQ",
                "signal_timeframe": "15min",
                "config_id": "good_is_bad_oos",
                "family": "delay_stop_target",
                "admissible_is": True,
                "robust_score_is": 0.8,
                "net_pnl": 100.0,
                "profit_factor": 1.2,
                "skip_rate": 0.0,
                "trades": 120,
            },
        ]
    )

    selected = campaign.select_top_configs_is_only(robustness, max_configs=1)

    assert list(selected["config_id"]) == ["good_is_bad_oos"]


def test_walkforward_selection_train_only_multiasset() -> None:
    fold = campaign.WalkforwardFold("fold_1", pd.Timestamp("2020-01-01").date(), pd.Timestamp("2021-12-31").date(), pd.Timestamp("2022-01-01").date(), pd.Timestamp("2022-12-31").date())
    config_frame = pd.DataFrame(
        [
            {"config_id": "a", "family": "delay_only", "filter_name": "none", "stop_multiplier": 1.0, "target_multiplier": 2.0, "entry_delay_minutes": 5, "stop_zone_fraction": pd.NA},
            {"config_id": "b", "family": "delay_stop_target", "filter_name": "none", "stop_multiplier": 1.25, "target_multiplier": 2.0, "entry_delay_minutes": 15, "stop_zone_fraction": pd.NA},
        ]
    )
    train_dates = [f"2020-01-{day:02d}" for day in range(1, 21)] + [f"2021-01-{day:02d}" for day in range(1, 21)]
    test_dates = [f"2022-01-{day:02d}" for day in range(1, 12)]
    events_by_config = {
        "a": pd.concat([_events(train_dates, [-1.0] * len(train_dates), config_id="a"), _events(test_dates, [5.0] * len(test_dates), config_id="a")], ignore_index=True),
        "b": pd.concat([_events(train_dates, [2.0] * len(train_dates), config_id="b"), _events(test_dates, [-2.0] * len(test_dates), config_id="b")], ignore_index=True),
    }

    ranking = campaign.compute_fold_train_ranking(
        symbol="MNQ",
        signal_timeframe="15min",
        fold=fold,
        config_frame=config_frame,
        events_by_config=events_by_config,
        estimated_cost_per_trade=0.25,
    )

    assert str(ranking.iloc[0]["config_id"]) == "b"


def test_portfolio_uses_preselected_configs_only() -> None:
    selected_oos_report = pd.DataFrame(
        [
            {"symbol": "MNQ", "signal_timeframe": "15min", "config_id": "sel1", "verdict": "candidate"},
        ]
    )
    selected_oos_daily = pd.DataFrame(
        [
            {"session_date": "2024-01-02", "daily_pnl": 10.0, "equity": 10.0, "sleeve_id": "MNQ_15min_sel1", "symbol": "MNQ", "signal_timeframe": "15min", "config_id": "sel1"},
            {"session_date": "2024-01-02", "daily_pnl": 999.0, "equity": 999.0, "sleeve_id": "MNQ_15min_not_selected", "symbol": "MNQ", "signal_timeframe": "15min", "config_id": "not_selected"},
        ]
    )
    selected_is_daily = selected_oos_daily.copy()
    walkforward_daily = pd.DataFrame()
    walkforward_summary = pd.DataFrame(columns=["symbol", "signal_timeframe", "verdict"])

    portfolio_oos_summary, _, _, _, _ = campaign._build_portfolio_outputs(
        selected_oos_daily=selected_oos_daily,
        selected_is_daily=selected_is_daily,
        walkforward_daily=walkforward_daily,
        selected_oos_report=selected_oos_report,
        walkforward_summary=walkforward_summary,
    )

    row = portfolio_oos_summary.loc[portfolio_oos_summary["portfolio_name"] == "equal_weight_all_candidates"].iloc[0]
    assert float(row["net_pnl"]) == 10.0


def test_timeframe_summary_outputs_required_columns() -> None:
    frame = campaign.build_timeframe_summary(
        pd.DataFrame([{"signal_timeframe": "15min", "profit_factor": 1.1, "net_pnl": 10.0}]),
        pd.DataFrame([{"signal_timeframe": "15min", "profit_factor": 1.0, "net_pnl": 5.0}]),
        pd.DataFrame([{"signal_timeframe": "15min", "oos_pass": True}]),
        pd.DataFrame([{"signal_timeframe": "15min", "symbol": "MNQ", "verdict": "candidate", "total_test_net_pnl": 20.0}]),
    )
    required = {
        "signal_timeframe",
        "total_configs",
        "median_is_pf",
        "median_oos_pf",
        "pct_configs_is_positive",
        "pct_configs_oos_positive",
        "pct_selected_configs_oos_pass",
        "wfa_stitched_median_pnl",
        "best_symbol",
        "verdict",
    }
    assert required.issubset(set(frame.columns))


def test_asset_summary_outputs_required_columns() -> None:
    frame = campaign.build_asset_summary(
        pd.DataFrame([{"symbol": "MNQ", "signal_timeframe": "15min", "robust_score_is": 0.8, "net_pnl": 10.0, "oos_pass": True}]),
        pd.DataFrame([{"symbol": "MNQ", "signal_timeframe": "15min", "verdict": "candidate", "total_test_net_pnl": 20.0, "test_profit_factor": 1.2, "positive_folds": 3, "benchmark_raw_hybrid_net_pnl_same_windows": -5.0, "improvement_vs_raw_hybrid": 25.0, "pass_rate": 0.8}]),
    )
    required = {
        "symbol",
        "best_timeframe_is_only",
        "best_timeframe_wfa",
        "selected_configs_count",
        "oos_pass_count",
        "wfa_verdict",
        "net_pnl_stitched",
        "pf_stitched",
        "positive_folds",
        "main_failure_mode",
    }
    assert required.issubset(set(frame.columns))


def test_campaign_smoke_outputs_required_files(tmp_path: Path, monkeypatch) -> None:
    def _fake_eval(*, symbol: str, signal_timeframe: str, execution_timeframe: str, output_dir: Path, raw_minute_df_override=None):
        config_frame = pd.DataFrame(
            [
                {
                    "config_id": f"{symbol}_{signal_timeframe}_raw",
                    "symbol": symbol,
                    "signal_timeframe": signal_timeframe,
                    "execution_timeframe": execution_timeframe,
                    "base_signal_variant": "seed",
                    "stop_multiplier": 1.0,
                    "target_multiplier": 1.0,
                    "entry_delay_minutes": 0,
                    "filter_name": "none",
                    "filter_params_json": "{}",
                    "family": "raw_hybrid",
                    "stop_zone_fraction": pd.NA,
                    "adverse_window_minutes": pd.NA,
                    "max_adverse_ticks": pd.NA,
                }
            ]
        )
        metrics = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "signal_timeframe": signal_timeframe,
                    "config_id": f"{symbol}_{signal_timeframe}_raw",
                    "family": "raw_hybrid",
                    "trades": 20,
                    "net_pnl": 10.0,
                    "gross_profit": 20.0,
                    "gross_loss": 10.0,
                    "winrate": 0.5,
                    "avg_trade": 0.5,
                    "median_trade": 0.5,
                    "profit_factor": 2.0,
                    "max_drawdown": -5.0,
                    "pnl_to_maxdd": 2.0,
                    "avg_holding_minutes": 10.0,
                    "median_holding_minutes": 10.0,
                    "stop_exit_rate": 0.5,
                    "target_exit_rate": 0.5,
                    "time_stop_exit_rate": 0.0,
                    "eod_exit_rate": 0.0,
                    "skip_rate": 0.0,
                    "trades_per_year": 20.0,
                    "estimated_cost_per_trade": 0.1,
                    "avg_trade_to_cost": 5.0,
                    "filter_name": "none",
                    "filter_params_json": "{}",
                    "stop_multiplier": 1.0,
                    "target_multiplier": 1.0,
                    "entry_delay_minutes": 0,
                }
            ]
        )
        selected = pd.DataFrame(
            [
                {
                    **metrics.iloc[0].to_dict(),
                    "robust_score_is": 0.9,
                    "admissible_is": True,
                    "rank_is": 1,
                    "oos_pass": True,
                    "net_pnl_oos": 5.0,
                    "profit_factor_oos": 1.2,
                    "trades_oos": 12,
                    "avg_trade_oos": 0.4,
                    "estimated_cost_per_trade_oos": 0.1,
                    "verdict": "candidate",
                    "params": "{}",
                }
            ]
        )
        folds = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "signal_timeframe": signal_timeframe,
                    "fold_id": "fold_1",
                    "train_start": "2020-01-01",
                    "train_end": "2021-12-31",
                    "test_start": "2022-01-01",
                    "test_end": "2022-12-31",
                    "train_days": 731,
                    "test_days": 365,
                }
            ]
        )
        fold_rank = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "signal_timeframe": signal_timeframe,
                    "fold_id": "fold_1",
                    "config_id": f"{symbol}_{signal_timeframe}_raw",
                    "family": "raw_hybrid",
                    "train_trades": 20,
                    "train_net_pnl": 10.0,
                    "train_profit_factor": 2.0,
                    "train_winrate": 0.5,
                    "train_avg_trade": 0.5,
                    "train_max_drawdown": -5.0,
                    "train_pnl_to_maxdd": 2.0,
                    "train_skip_rate": 0.0,
                    "train_robust_score": 0.9,
                    "selected_in_fold": True,
                }
            ]
        )
        fold_selected = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "signal_timeframe": signal_timeframe,
                    "fold_id": "fold_1",
                    "selected_config_id": f"{symbol}_{signal_timeframe}_raw",
                    "selected_family": "raw_hybrid",
                    "train_robust_score": 0.9,
                    "train_net_pnl": 10.0,
                    "train_profit_factor": 2.0,
                    "test_trades": 12,
                    "test_net_pnl": 5.0,
                    "test_profit_factor": 1.2,
                    "test_winrate": 0.5,
                    "test_avg_trade": 0.4,
                    "test_max_drawdown": -4.0,
                    "test_pnl_to_maxdd": 1.25,
                    "test_pass": True,
                }
            ]
        )
        stitched = _events(["2022-01-03"], [5.0], symbol=symbol, timeframe=signal_timeframe, config_id="walkforward_stitched")
        daily = pd.DataFrame(
            [
                {
                    "session_date": "2022-01-03",
                    "daily_pnl": 5.0,
                    "equity": 5.0,
                    "sleeve_id": f"{symbol}_{signal_timeframe}_wfa",
                    "symbol": symbol,
                    "signal_timeframe": signal_timeframe,
                    "config_id": "walkforward_stitched",
                    "selection_basis": "walkforward_train_only",
                }
            ]
        )
        audit = {
            "symbol": symbol,
            "signal_timeframe": signal_timeframe,
            "execution_timeframe": execution_timeframe,
            "source_path": "fake",
            "first_timestamp": "2024-01-02 09:30:00-05:00",
            "last_timestamp": "2024-01-02 16:00:00-05:00",
            "number_of_1min_rows": 100,
            "number_of_signal_rows": 10,
            "rth_rows": 100,
            "timezone": "America/New_York",
            "session_convention": "RTH",
            "missing_days": [],
            "split_mode": "fixed_calendar",
            "variant_name": "seed",
        }
        (output_dir / f"data_audit_{symbol}_{signal_timeframe}.json").write_text("{}", encoding="utf-8")
        return {
            "data_audit": audit,
            "config_frame": config_frame,
            "metrics_is": metrics,
            "metrics_oos": metrics.assign(net_pnl=5.0, profit_factor=1.2),
            "metrics_full": metrics.assign(net_pnl=15.0, profit_factor=1.5),
            "robustness": metrics.assign(robust_score_is=0.9, admissible_is=True),
            "selected_is": selected,
            "selected_oos_report": selected,
            "walkforward_folds": folds,
            "fold_train_ranking": fold_rank,
            "fold_selected_test_results": fold_selected,
            "walkforward_stitched_trades": stitched,
            "walkforward_stitched_daily_returns": daily,
            "walkforward_summary": pd.DataFrame(
                [
                    {
                        "symbol": symbol,
                        "signal_timeframe": signal_timeframe,
                        "total_test_trades": 12,
                        "total_test_net_pnl": 5.0,
                        "test_profit_factor": 1.2,
                        "test_winrate": 0.5,
                        "avg_trade": 0.4,
                        "max_drawdown": -4.0,
                        "pnl_to_maxdd": 1.25,
                        "number_of_folds": 1,
                        "positive_folds": 1,
                        "pass_rate": 1.0,
                        "selected_family_counts": "{\"raw_hybrid\": 1}",
                        "train_score_test_corr": 0.0,
                        "benchmark_raw_hybrid_net_pnl_same_windows": 1.0,
                        "improvement_vs_raw_hybrid": 4.0,
                        "verdict": "candidate",
                    }
                ]
            ),
            "selected_oos_daily": pd.DataFrame(
                [
                    {
                        "session_date": "2024-01-03",
                        "daily_pnl": 5.0,
                        "equity": 5.0,
                        "sleeve_id": f"{symbol}_{signal_timeframe}_{symbol}_{signal_timeframe}_raw",
                        "symbol": symbol,
                        "signal_timeframe": signal_timeframe,
                        "config_id": f"{symbol}_{signal_timeframe}_raw",
                        "selection_basis": "is_only",
                    }
                ]
            ),
            "selected_is_daily": pd.DataFrame(
                [
                    {
                        "session_date": "2023-01-03",
                        "daily_pnl": 5.0,
                        "equity": 5.0,
                        "sleeve_id": f"{symbol}_{signal_timeframe}_{symbol}_{signal_timeframe}_raw",
                        "symbol": symbol,
                        "signal_timeframe": signal_timeframe,
                        "config_id": f"{symbol}_{signal_timeframe}_raw",
                    }
                ]
            ),
        }

    monkeypatch.setattr(campaign, "_evaluate_symbol_timeframe", _fake_eval)

    run_dir = campaign.run_campaign(
        symbols=["MNQ"],
        signal_timeframes=["15min"],
        execution_timeframe="1min",
        output_root=tmp_path,
        max_workers=1,
        dataset_overrides={"MNQ": _minute_df()},
    )

    required = [
        "config_universe.csv",
        "config_metrics_is.csv",
        "config_metrics_oos.csv",
        "config_metrics_full.csv",
        "config_robustness_scores.csv",
        "selected_configs_is_only.csv",
        "selected_configs_oos_report.csv",
        "walkforward_folds.csv",
        "fold_train_ranking.csv",
        "fold_selected_test_results.csv",
        "walkforward_stitched_trades.csv",
        "walkforward_summary.csv",
        "portfolio_oos_summary.csv",
        "portfolio_walkforward_summary.csv",
        "portfolio_daily_pnl.csv",
        "asset_correlation_matrix.csv",
        "portfolio_asset_contribution.csv",
        "timeframe_summary.csv",
        "asset_summary.csv",
        "final_report.md",
        "run_metadata.json",
    ]
    for filename in required:
        assert (run_dir / filename).exists(), filename

