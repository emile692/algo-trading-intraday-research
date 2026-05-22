from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analytics.volume_climax_pullback_portfolio_integration import (
    DEFAULT_PULLBACK_M2K_ONLY,
    DEFAULT_PULLBACK_PORTFOLIO,
    align_daily_pnl,
    bootstrap_portfolios,
    build_combined_portfolio_series,
    classify_verdict,
    expand_pullback_series,
    generate_portfolio_specs,
    load_baseline_daily_results,
    load_pullback_portfolios,
    run_campaign,
)


def _write_baseline(path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "session_date": pd.to_datetime(
                [
                    "2023-12-27",
                    "2023-12-28",
                    "2023-12-29",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-08",
                ]
            ),
            "daily_pnl_usd": [100.0, -50.0, 0.0, 120.0, -80.0, 30.0, 0.0, 60.0],
            "daily_trade_count": [1, 1, 0, 1, 1, 1, 0, 1],
        }
    )
    frame.to_csv(path, index=False)
    return path


def _write_pullback_export(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {
                "portfolio_name": DEFAULT_PULLBACK_PORTFOLIO,
                "selection_basis": "strict_train_only",
                "deployable": True,
                "net_pnl": 50.0,
            },
            {
                "portfolio_name": DEFAULT_PULLBACK_M2K_ONLY,
                "selection_basis": "strict_train_only",
                "deployable": True,
                "net_pnl": 40.0,
            },
            {
                "portfolio_name": "posthoc_superior",
                "selection_basis": "diagnostic_posthoc",
                "deployable": False,
                "net_pnl": 999.0,
            },
        ]
    )
    summary.to_csv(root / "strict_portfolio_summary.csv", index=False)
    daily = pd.DataFrame(
        [
            {
                "portfolio_name": DEFAULT_PULLBACK_PORTFOLIO,
                "fold_id": "fold_1",
                "session_date": "2023-12-28",
                "daily_pnl": 20.0,
                "selection_basis": "strict_train_only",
                "deployable": True,
            },
            {
                "portfolio_name": DEFAULT_PULLBACK_PORTFOLIO,
                "fold_id": "fold_2",
                "session_date": "2024-01-02",
                "daily_pnl": 15.0,
                "selection_basis": "strict_train_only",
                "deployable": True,
            },
            {
                "portfolio_name": DEFAULT_PULLBACK_PORTFOLIO,
                "fold_id": "fold_2",
                "session_date": "2024-01-04",
                "daily_pnl": -5.0,
                "selection_basis": "strict_train_only",
                "deployable": True,
            },
            {
                "portfolio_name": DEFAULT_PULLBACK_PORTFOLIO,
                "fold_id": "fold_2",
                "session_date": "2024-01-08",
                "daily_pnl": 8.0,
                "selection_basis": "strict_train_only",
                "deployable": True,
            },
            {
                "portfolio_name": DEFAULT_PULLBACK_M2K_ONLY,
                "fold_id": "fold_1",
                "session_date": "2023-12-28",
                "daily_pnl": 10.0,
                "selection_basis": "strict_train_only",
                "deployable": True,
            },
            {
                "portfolio_name": DEFAULT_PULLBACK_M2K_ONLY,
                "fold_id": "fold_2",
                "session_date": "2024-01-02",
                "daily_pnl": 12.0,
                "selection_basis": "strict_train_only",
                "deployable": True,
            },
            {
                "portfolio_name": DEFAULT_PULLBACK_M2K_ONLY,
                "fold_id": "fold_2",
                "session_date": "2024-01-04",
                "daily_pnl": -2.0,
                "selection_basis": "strict_train_only",
                "deployable": True,
            },
            {
                "portfolio_name": "posthoc_superior",
                "fold_id": "fold_2",
                "session_date": "2024-01-03",
                "daily_pnl": 300.0,
                "selection_basis": "diagnostic_posthoc",
                "deployable": False,
            },
        ]
    )
    daily.to_csv(root / "strict_portfolio_daily_returns.csv", index=False)
    return root


def _aligned_frame(tmp_path: Path) -> pd.DataFrame:
    baseline_path = _write_baseline(tmp_path / "baseline.csv")
    pullback_root = _write_pullback_export(tmp_path / "pullback")
    baseline = load_baseline_daily_results(baseline_path)
    pullbacks = load_pullback_portfolios(pullback_root)
    pullback_equal = expand_pullback_series(pullbacks[DEFAULT_PULLBACK_PORTFOLIO], baseline["session_date"])
    m2k_only = expand_pullback_series(pullbacks[DEFAULT_PULLBACK_M2K_ONLY], baseline["session_date"])
    return align_daily_pnl(baseline, pullback_equal, m2k_only)


def test_pullback_sleeve_is_fixed_and_not_reoptimized(tmp_path: Path) -> None:
    pullbacks = load_pullback_portfolios(_write_pullback_export(tmp_path / "pullback"))
    assert set(pullbacks) == {DEFAULT_PULLBACK_PORTFOLIO, DEFAULT_PULLBACK_M2K_ONLY}


def test_no_posthoc_asset_filtering_in_specs() -> None:
    specs = generate_portfolio_specs()
    names = {spec.portfolio_name for spec in specs}
    assert "posthoc_superior" not in names
    assert "baseline_plus_pullback_equal_notional" in names


def test_daily_pnl_alignment_handles_missing_dates_correctly(tmp_path: Path) -> None:
    aligned = _aligned_frame(tmp_path)
    row = aligned.loc[aligned["session_date"].eq(pd.Timestamp("2024-01-03"))].iloc[0]
    assert bool(row["pullback_defined"])
    assert row["pullback_daily_pnl_usd"] == 0.0


def test_portfolio_combination_math(tmp_path: Path) -> None:
    aligned = _aligned_frame(tmp_path)
    spec = next(spec for spec in generate_portfolio_specs() if spec.portfolio_name == "baseline_plus_pullback_equal_notional")
    combined = build_combined_portfolio_series(aligned, spec)
    row = combined.loc[combined["session_date"].eq(pd.Timestamp("2024-01-02"))].iloc[0]
    assert row["daily_pnl_usd"] == 135.0


def test_bootstrap_is_deterministic(tmp_path: Path) -> None:
    aligned = _aligned_frame(tmp_path)
    spec = next(spec for spec in generate_portfolio_specs() if spec.portfolio_name == "baseline_plus_pullback_equal_notional")
    portfolio = {"baseline_plus_pullback_equal_notional": build_combined_portfolio_series(aligned, spec)}
    left = bootstrap_portfolios(portfolio, bootstrap_paths=16, block_size=2, seed=11)
    right = bootstrap_portfolios(portfolio, bootstrap_paths=16, block_size=2, seed=11)
    pd.testing.assert_frame_equal(left, right)


def test_verdict_logic() -> None:
    assert (
        classify_verdict(
            baseline_net_pnl=100.0,
            net_pnl=-1.0,
            profit_factor=1.2,
            max_drawdown_delta=0.0,
            sleeve_correlation_to_baseline=0.1,
            sharpe_delta=0.1,
            sortino_delta=0.1,
            p05_delta=0.0,
            daily_loss_breach_delta=0,
        )
        == "reject"
    )
    assert (
        classify_verdict(
            baseline_net_pnl=100.0,
            net_pnl=180.0,
            profit_factor=1.18,
            max_drawdown_delta=10.0,
            sleeve_correlation_to_baseline=0.2,
            sharpe_delta=0.2,
            sortino_delta=0.2,
            p05_delta=0.0,
            daily_loss_breach_delta=0,
        )
        == "diversifier_watchlist"
    )
    assert (
        classify_verdict(
            baseline_net_pnl=100.0,
            net_pnl=420.0,
            profit_factor=1.35,
            max_drawdown_delta=20.0,
            sleeve_correlation_to_baseline=0.1,
            sharpe_delta=0.3,
            sortino_delta=0.3,
            p05_delta=10.0,
            daily_loss_breach_delta=0,
        )
        == "portfolio_candidate"
    )


def test_required_exports_exist(tmp_path: Path) -> None:
    baseline_path = _write_baseline(tmp_path / "baseline.csv")
    pullback_export = _write_pullback_export(tmp_path / "pullback")
    output_dir = run_campaign(
        pullback_export=pullback_export,
        baseline_daily_pnl_path=str(baseline_path),
        output_root=tmp_path / "out",
        smoke=True,
    )
    expected = {
        "final_report.md",
        "run_metadata.json",
        "daily_pnl_aligned.csv",
        "portfolio_summary.csv",
        "portfolio_correlation.csv",
        "incremental_metrics.csv",
        "monthly_pnl.csv",
        "yearly_pnl.csv",
        "worst_days.csv",
        "drawdown_comparison.csv",
        "bootstrap_summary.csv",
        "prop_constraint_summary.csv",
        "rejected_or_diagnostic_results.csv",
    }
    actual = {path.name for path in output_dir.iterdir()}
    assert expected.issubset(actual)


def test_campaign_runs_in_smoke_mode(tmp_path: Path) -> None:
    baseline_path = _write_baseline(tmp_path / "baseline.csv")
    pullback_export = _write_pullback_export(tmp_path / "pullback")
    output_dir = run_campaign(
        pullback_export=pullback_export,
        baseline_daily_pnl_path=str(baseline_path),
        output_root=tmp_path / "smoke",
        smoke=True,
    )
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["fixed_pullback_portfolio"] == DEFAULT_PULLBACK_PORTFOLIO
    assert metadata["smoke"] is True
