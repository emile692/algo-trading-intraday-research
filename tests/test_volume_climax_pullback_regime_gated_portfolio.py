from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics import volume_climax_pullback_regime_gated_portfolio as campaign


TZ = "America/New_York"


def _mgc_events() -> pd.DataFrame:
    rows = []
    values = [0.1, 0.2, 0.3, 0.4, 0.9]
    for idx, atr_pct in enumerate(values, start=1):
        ts = pd.Timestamp(f"2024-01-{idx:02d} 10:31:00", tz=TZ)
        rows.append(
            {
                "signal_time": ts - pd.Timedelta(hours=1),
                "entry_time": ts,
                "exit_time": ts + pd.Timedelta(minutes=15),
                "session_date": ts.date(),
                "direction": "long",
                "executed": True,
                "net_pnl_usd": float(idx * 10),
                "gross_pnl_usd": float(idx * 10),
                "pnl": float(idx * 10),
                "holding_minutes": 15.0,
                "exit_reason": "target_1m",
                "atr_percentile_100": atr_pct,
                "atr_ratio_5_20": atr_pct,
                "close": 100.0 + idx,
                "ema20": 99.0,
                "ema50": 98.0,
                "ema20_slope_3_atr": 0.1,
                "ema50_slope_3_atr": 0.1,
                "distance_from_vwap_atr": 0.5,
            }
        )
    return pd.DataFrame(rows)


def _m2k_events() -> pd.DataFrame:
    rows = []
    for idx in range(1, 6):
        ts = pd.Timestamp(f"2024-01-{idx:02d} 09:45:00", tz=TZ)
        rows.append(
            {
                "signal_time": ts - pd.Timedelta(hours=1),
                "entry_time": ts,
                "exit_time": ts + pd.Timedelta(minutes=10),
                "session_date": ts.date(),
                "direction": "long",
                "executed": True,
                "net_pnl_usd": 5.0,
                "gross_pnl_usd": 5.0,
                "pnl": 5.0,
                "holding_minutes": 10.0,
                "exit_reason": "target_1m",
            }
        )
    return pd.DataFrame(rows)


def test_thresholds_fitted_on_train_data_only() -> None:
    rule = campaign.RegimeRuleSpec("atr_pct_above_q60__conditional_equal_weight", "atr_pct_above", {"quantile": 0.60}, "conditional_equal_weight")
    train = _mgc_events().iloc[:4].copy()
    fitted = campaign.fit_rule_on_train(rule, train)

    assert abs(float(fitted["params"]["threshold"]) - 0.28) < 1e-9


def test_oos_data_not_used_in_rule_selection() -> None:
    m2k_train = _m2k_events()
    mgc_train = _mgc_events().iloc[:4].copy()
    rules = [
        campaign.RegimeRuleSpec("always_on__conditional_equal_weight", "always_on", {}, "conditional_equal_weight"),
        campaign.RegimeRuleSpec("atr_pct_above_q70__conditional_equal_weight", "atr_pct_above", {"quantile": 0.70}, "conditional_equal_weight"),
    ]

    ranking = campaign.build_regime_rule_train_ranking(
        fold_id="fold_1",
        m2k_train_events=m2k_train,
        mgc_train_events=mgc_train,
        rules=rules,
        estimated_cost_per_trade=1.0,
    )
    winner = campaign.select_best_train_rule(ranking)

    assert str(winner["rule_id"]) in set(ranking["rule_id"])


def test_posthoc_diagnostics_labeled_non_deployable() -> None:
    summary = pd.DataFrame(
        [
            {"entity_name": "m2k_only_baseline", "net_pnl": 10.0, "profit_factor": 1.1, "positive_folds": 3},
            {"entity_name": "strict_best_regime_gated", "net_pnl": 20.0, "profit_factor": 1.3, "positive_folds": 4},
        ]
    )

    diagnostics = campaign.build_diagnostic_posthoc_rows(summary)

    assert bool(diagnostics["deployable"].eq(False).all())


def test_mgc_gating_does_not_alter_m2k_baseline_trades() -> None:
    m2k = _m2k_events()
    mgc = _mgc_events().iloc[:0].copy()

    portfolio = campaign.build_conditional_portfolio_events(
        m2k_events=m2k,
        mgc_events=mgc,
        allocation_scheme="m2k_only",
        train_weights=None,
    )

    assert int(portfolio["executed"].sum()) == int(m2k["executed"].sum())
    assert float(pd.to_numeric(portfolio["net_pnl_usd"], errors="coerce").sum()) == float(pd.to_numeric(m2k["net_pnl_usd"], errors="coerce").sum())


def test_low_trade_count_rules_are_penalized() -> None:
    metrics = {
        "net_pnl": 100.0,
        "profit_factor": 1.4,
        "trades": 5,
        "avg_trade": 5.0,
        "estimated_cost_per_trade": 1.0,
        "monthly_hit_rate": 0.5,
        "top1_contribution_pct": 0.2,
        "top3_contribution_pct": 0.5,
        "pnl_to_maxdd": 1.0,
    }

    low = campaign.score_train_rule_result(metrics, mgc_retention_rate=0.05, mgc_active_months=2)
    high = campaign.score_train_rule_result(metrics, mgc_retention_rate=0.40, mgc_active_months=8)

    assert high > low


def test_verdict_logic_for_all_levels() -> None:
    m2k_only = {"net_pnl": 100.0, "monthly_hit_rate": 0.4}
    raw = {"net_pnl": 150.0, "monthly_hit_rate": 0.45}
    assert campaign.strict_regime_portfolio_verdict(
        portfolio_metrics={"net_pnl": -1.0, "profit_factor": 0.9, "max_drawdown": -100.0, "monthly_hit_rate": 0.2},
        positive_folds=1,
        m2k_only_metrics=m2k_only,
        raw_m2k_mgc_metrics=raw,
        top1_contribution_pct=0.2,
        top3_contribution_pct=0.5,
    ) == "reject"
    assert campaign.strict_regime_portfolio_verdict(
        portfolio_metrics={"net_pnl": 120.0, "profit_factor": 1.16, "max_drawdown": -300.0, "monthly_hit_rate": 0.45},
        positive_folds=3,
        m2k_only_metrics=m2k_only,
        raw_m2k_mgc_metrics=raw,
        top1_contribution_pct=0.2,
        top3_contribution_pct=0.5,
    ) == "weak_watchlist"
    assert campaign.strict_regime_portfolio_verdict(
        portfolio_metrics={"net_pnl": 300.0, "profit_factor": 1.22, "max_drawdown": -200.0, "monthly_hit_rate": 0.55},
        positive_folds=4,
        m2k_only_metrics=m2k_only,
        raw_m2k_mgc_metrics=raw,
        top1_contribution_pct=0.3,
        top3_contribution_pct=0.6,
    ) == "watchlist"
    assert campaign.strict_regime_portfolio_verdict(
        portfolio_metrics={"net_pnl": 400.0, "profit_factor": 1.35, "max_drawdown": -150.0, "monthly_hit_rate": 0.60},
        positive_folds=4,
        m2k_only_metrics=m2k_only,
        raw_m2k_mgc_metrics=raw,
        top1_contribution_pct=0.3,
        top3_contribution_pct=0.7,
    ) == "candidate"


def test_required_exports_exist_with_expected_columns(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.parquet"
    source.write_text("x", encoding="utf-8")

    def _fake_eval(**kwargs):
        return {
            "strict_regime_wfa_summary": pd.DataFrame(
                [
                    {"entity_name": "m2k_only_baseline", "selection_basis": "strict_train_only", "deployable": False, "net_pnl": 10.0, "profit_factor": 1.2, "trades": 5, "positive_folds": 3, "max_drawdown": -5.0, "max_daily_drawdown": -5.0, "win_rate": 0.5, "avg_trade": 2.0, "median_trade": 2.0, "monthly_hit_rate": 0.5, "active_months": 3, "mgc_trade_retention_rate": 0.0, "mgc_contribution_pnl": 0.0, "top1_contribution_pct": 0.2, "top3_contribution_pct": 0.4, "top5_contribution_pct": 0.5, "worst1_contribution_pct": -0.1, "worst3_contribution_pct": -0.2, "worst5_contribution_pct": -0.3, "verdict": "baseline"},
                    {"entity_name": "raw_m2k_mgc_equal_weight", "selection_basis": "strict_train_only", "deployable": False, "net_pnl": 20.0, "profit_factor": 1.3, "trades": 8, "positive_folds": 3, "max_drawdown": -8.0, "max_daily_drawdown": -8.0, "win_rate": 0.55, "avg_trade": 2.5, "median_trade": 2.0, "monthly_hit_rate": 0.5, "active_months": 3, "mgc_trade_retention_rate": 1.0, "mgc_contribution_pnl": 10.0, "top1_contribution_pct": 0.3, "top3_contribution_pct": 0.5, "top5_contribution_pct": 0.6, "worst1_contribution_pct": -0.1, "worst3_contribution_pct": -0.2, "worst5_contribution_pct": -0.3, "verdict": "baseline"},
                    {"entity_name": "strict_best_regime_gated", "selection_basis": "strict_train_only", "deployable": False, "net_pnl": 30.0, "profit_factor": 1.4, "trades": 6, "positive_folds": 4, "max_drawdown": -6.0, "max_daily_drawdown": -6.0, "win_rate": 0.6, "avg_trade": 5.0, "median_trade": 3.0, "monthly_hit_rate": 0.6, "active_months": 4, "mgc_trade_retention_rate": 0.5, "mgc_contribution_pnl": 12.0, "top1_contribution_pct": 0.2, "top3_contribution_pct": 0.4, "top5_contribution_pct": 0.5, "worst1_contribution_pct": -0.1, "worst3_contribution_pct": -0.2, "worst5_contribution_pct": -0.3, "verdict": "watchlist"},
                ]
            ),
            "strict_regime_wfa_fold_breakdown": pd.DataFrame([{"fold_id": "fold_1", "selected_rule_id": "always_on__conditional_equal_weight", "test_net_pnl": 30.0}]),
            "selected_regime_rule_by_fold": pd.DataFrame([{"fold_id": "fold_1", "rule_id": "always_on__conditional_equal_weight", "allocation_scheme": "conditional_equal_weight", "fitted_params_json": "{}"}]),
            "regime_rule_train_ranking": pd.DataFrame([{"fold_id": "fold_1", "rule_id": "always_on__conditional_equal_weight", "train_score": 1.0}]),
            "portfolio_daily_returns": pd.DataFrame([{"entity_name": "strict_best_regime_gated", "session_date": "2024-01-02", "daily_pnl": 10.0, "equity": 10.0, "drawdown": 0.0}]),
            "portfolio_monthly_pnl": pd.DataFrame([{"entity_name": "strict_best_regime_gated", "month": "2024-01", "pnl": 10.0}]),
            "portfolio_yearly_pnl": pd.DataFrame([{"entity_name": "strict_best_regime_gated", "year": 2024, "pnl": 10.0}]),
            "mgc_regime_retention_summary": pd.DataFrame([{"fold_id": "fold_1", "selected_rule_id": "always_on__conditional_equal_weight", "mgc_test_retention_rate": 0.5}]),
            "trade_concentration": pd.DataFrame([{"entity_name": "strict_best_regime_gated", "trade_count": 5, "total_pnl": 30.0}]),
            "baseline_comparison": pd.DataFrame([{"entity_name": "m2k_only_baseline", "net_pnl": 10.0}]),
            "rejected_or_diagnostic_results": pd.DataFrame([{"result_type": "posthoc_positive_baseline", "deployable": False, "reason": "baseline_or_posthoc_reference_only"}]),
            "strict_events": pd.DataFrame(),
            "data_audits": {"M2K": {"source_path": str(source)}},
            "survivor_summary": pd.DataFrame(),
            "survivor_audit_dir": str(tmp_path / "survivor"),
        }

    monkeypatch.setattr(campaign, "latest_survivor_audit_dir", lambda output_root: tmp_path / "survivor")
    monkeypatch.setattr(campaign, "evaluate_regime_gated_portfolio", _fake_eval)

    output_dir = campaign.run_campaign(
        symbols=["M2K", "MGC"],
        signal_timeframe="1H",
        execution_timeframe="1min",
        output_root=tmp_path,
        smoke=True,
        dataset_overrides={},
    )

    summary = pd.read_csv(output_dir / "strict_regime_wfa_summary.csv")
    baselines = pd.read_csv(output_dir / "baseline_comparison.csv")
    diagnostics = pd.read_csv(output_dir / "rejected_or_diagnostic_results.csv")

    assert {"entity_name", "net_pnl", "profit_factor", "verdict"}.issubset(summary.columns)
    assert {"entity_name", "net_pnl"}.issubset(baselines.columns)
    assert {"result_type", "deployable", "reason"}.issubset(diagnostics.columns)
    assert (output_dir / "final_report.md").exists()


def test_smoke_campaign_runs(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.parquet"
    source.write_text("x", encoding="utf-8")
    monkeypatch.setattr(campaign, "latest_survivor_audit_dir", lambda output_root: tmp_path / "survivor")
    monkeypatch.setattr(campaign, "evaluate_regime_gated_portfolio", lambda **kwargs: {
        "strict_regime_wfa_summary": pd.DataFrame([{"entity_name": "strict_best_regime_gated", "selection_basis": "strict_train_only", "deployable": False, "net_pnl": 0.0, "profit_factor": 0.0, "trades": 0, "positive_folds": 0, "max_drawdown": 0.0, "max_daily_drawdown": 0.0, "win_rate": 0.0, "avg_trade": 0.0, "median_trade": 0.0, "monthly_hit_rate": 0.0, "active_months": 0, "mgc_trade_retention_rate": 0.0, "mgc_contribution_pnl": 0.0, "top1_contribution_pct": pd.NA, "top3_contribution_pct": pd.NA, "top5_contribution_pct": pd.NA, "worst1_contribution_pct": pd.NA, "worst3_contribution_pct": pd.NA, "worst5_contribution_pct": pd.NA, "verdict": "reject"}]),
        "strict_regime_wfa_fold_breakdown": pd.DataFrame(columns=["fold_id", "selected_rule_id", "test_net_pnl"]),
        "selected_regime_rule_by_fold": pd.DataFrame(columns=["fold_id", "rule_id", "allocation_scheme", "fitted_params_json"]),
        "regime_rule_train_ranking": pd.DataFrame(columns=["fold_id", "rule_id", "train_score"]),
        "portfolio_daily_returns": pd.DataFrame(columns=["entity_name", "session_date", "daily_pnl", "equity", "drawdown"]),
        "portfolio_monthly_pnl": pd.DataFrame(columns=["entity_name", "month", "pnl"]),
        "portfolio_yearly_pnl": pd.DataFrame(columns=["entity_name", "year", "pnl"]),
        "mgc_regime_retention_summary": pd.DataFrame(columns=["fold_id", "selected_rule_id", "mgc_test_retention_rate"]),
        "trade_concentration": pd.DataFrame(columns=["entity_name", "trade_count", "total_pnl"]),
        "baseline_comparison": pd.DataFrame(columns=["entity_name", "net_pnl"]),
        "rejected_or_diagnostic_results": pd.DataFrame(columns=["result_type", "deployable", "reason"]),
        "strict_events": pd.DataFrame(),
        "data_audits": {"M2K": {"source_path": str(source)}},
        "survivor_summary": pd.DataFrame(),
        "survivor_audit_dir": str(tmp_path / "survivor"),
    })

    output_dir = campaign.run_campaign(
        symbols=["M2K", "MGC"],
        signal_timeframe="1H",
        execution_timeframe="1min",
        output_root=tmp_path,
        smoke=True,
        dataset_overrides={},
    )

    assert output_dir.exists()
    assert (output_dir / "strict_regime_wfa_summary.csv").exists()
    assert (output_dir / "run_metadata.json").exists()
