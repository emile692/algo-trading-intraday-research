from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics.volume_climax_pullback_micro_portfolio_campaign import (
    MotorConfigSpec,
    PortfolioVariantSpec,
    build_default_portfolio_variants,
    build_master_calendar,
    build_portfolio_scope,
    run_campaign,
    summarize_diversification,
    _load_motor_variant,
)


def _write_synthetic_motor_export(
    root: Path,
    *,
    variant_name: str,
    daily_pnls: list[float],
    trade_pnls: list[float],
    trade_risks: list[float],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    phases = ["is", "is", "oos", "oos"]

    daily = pd.DataFrame(
        {
            "session_date": dates,
            "daily_pnl_usd": daily_pnls,
            "daily_gross_pnl_usd": daily_pnls,
            "daily_fees_usd": [0.0] * 4,
            "daily_trade_count": [1 if value != 0 else 0 for value in daily_pnls],
            "daily_loss_count": [1 if value < 0 else 0 for value in daily_pnls],
            "equity": 50_000.0 + pd.Series(daily_pnls).cumsum(),
            "peak_equity": (50_000.0 + pd.Series(daily_pnls).cumsum()).cummax(),
            "drawdown_usd": (50_000.0 + pd.Series(daily_pnls).cumsum()) - (50_000.0 + pd.Series(daily_pnls).cumsum()).cummax(),
            "drawdown_pct": 0.0,
            "cumulative_pnl_usd": pd.Series(daily_pnls).cumsum(),
            "campaign_variant_name": [variant_name] * 4,
            "variant_role": ["grid"] * 4,
            "scope": ["full"] * 4,
            "alpha_variant_name": ["synthetic_alpha"] * 4,
            "phase": phases,
        }
    )
    daily.to_csv(root / "daily_equity_by_variant.csv", index=False)

    trade_dates = pd.to_datetime(["2024-01-02 10:30", "2024-01-03 10:30", "2024-01-04 10:30", "2024-01-05 10:30"])
    trades = pd.DataFrame(
        {
            "trade_id": [1, 2, 3, 4],
            "session_date": trade_dates.normalize(),
            "entry_time": trade_dates,
            "exit_time": trade_dates + pd.Timedelta(hours=1),
            "quantity": [1.0, 1.0, 1.0, 1.0],
            "pnl_usd": trade_pnls,
            "fees": [0.0, 0.0, 0.0, 0.0],
            "net_pnl_usd": trade_pnls,
            "trade_risk_usd": trade_risks,
            "actual_risk_usd": trade_risks,
            "risk_per_contract_usd": trade_risks,
            "notional_usd": [10_000.0] * 4,
            "exit_reason": ["stop", "target", "stop", "target"],
            "bars_held": [1, 1, 1, 1],
            "campaign_variant_name": [variant_name] * 4,
            "variant_role": ["grid"] * 4,
            "scope": ["full"] * 4,
            "alpha_variant_name": ["synthetic_alpha"] * 4,
            "phase": phases,
            "account_size_usd": [50_000.0] * 4,
        }
    )
    trades.to_csv(root / "trades_by_variant.csv", index=False)

    pd.DataFrame({"campaign_variant_name": [variant_name]}).to_csv(root / "summary_by_variant.csv", index=False)


def test_build_default_portfolio_variants_has_disciplined_size_and_families() -> None:
    variants = build_default_portfolio_variants()

    assert len(variants) == 17
    assert {variant.portfolio_family for variant in variants} == {"single", "pair", "three_way"}
    assert sum(variant.portfolio_family == "single" for variant in variants) == 3
    assert sum(variant.portfolio_family == "pair" for variant in variants) == 9
    assert sum(variant.portfolio_family == "three_way" for variant in variants) == 5


def test_build_portfolio_scope_aggregates_daily_pnl_correctly(tmp_path: Path) -> None:
    mnq_root = tmp_path / "mnq"
    m2k_root = tmp_path / "m2k"
    _write_synthetic_motor_export(mnq_root, variant_name="mnq_variant", daily_pnls=[100.0, -50.0, 40.0, 10.0], trade_pnls=[100.0, -50.0, 40.0, 10.0], trade_risks=[50.0, 50.0, 50.0, 50.0])
    _write_synthetic_motor_export(m2k_root, variant_name="m2k_variant", daily_pnls=[20.0, 30.0, -10.0, 0.0], trade_pnls=[20.0, 30.0, -10.0, 0.0], trade_risks=[25.0, 25.0, 25.0, 25.0])

    specs = {
        "MNQ_default": MotorConfigSpec("MNQ_default", "MNQ", "default", "mnq_variant", 0.0015, 4, mnq_root, "synthetic"),
        "M2K_default": MotorConfigSpec("M2K_default", "M2K", "default", "m2k_variant", 0.0030, 6, m2k_root, "synthetic"),
    }
    loaded = {key: _load_motor_variant(spec) for key, spec in specs.items()}
    calendar = build_master_calendar(loaded)
    variant = PortfolioVariantSpec(
        portfolio_variant_name="MNQ_M2K__equal_weight_notional__core_default",
        portfolio_name="MNQ_M2K",
        portfolio_family="pair",
        allocation_scheme="equal_weight_notional",
        config_bundle="core_default",
        member_motor_keys=("MNQ_default", "M2K_default"),
        description="synthetic pair",
    )

    _, daily_results, _, _ = build_portfolio_scope(
        variant,
        loaded_motors=loaded,
        calendar=calendar,
        initial_capital_usd=50_000.0,
        portfolio_risk_cap_usd=250.0,
        scope_name="full",
    )

    assert daily_results["daily_pnl_usd"].round(6).tolist() == [60.0, -10.0, 15.0, 5.0]


def test_summarize_diversification_on_small_example() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    daily_motor = pd.DataFrame(
        {
            "session_date": list(dates) * 2,
            "symbol": ["MNQ"] * 4 + ["M2K"] * 4,
            "daily_pnl_usd": [100.0, -50.0, 20.0, -10.0, -20.0, 30.0, -10.0, 5.0],
            "daily_trade_count": [1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    daily_portfolio = pd.DataFrame(
        {
            "session_date": dates,
            "daily_pnl_usd": [80.0, -20.0, 10.0, -5.0],
            "equity": [50_080.0, 50_060.0, 50_070.0, 50_065.0],
            "peak_equity": [50_080.0, 50_080.0, 50_080.0, 50_080.0],
            "drawdown_usd": [0.0, -20.0, -10.0, -15.0],
        }
    )
    member_specs = {
        "MNQ_default": MotorConfigSpec("MNQ_default", "MNQ", "default", "mnq_variant", 0.0015, 4, Path("."), "synthetic"),
        "M2K_default": MotorConfigSpec("M2K_default", "M2K", "default", "m2k_variant", 0.0030, 6, Path("."), "synthetic"),
    }

    summary = summarize_diversification(daily_motor, daily_portfolio=daily_portfolio, member_specs=member_specs)

    assert summary["pct_days_multiple_motors_lose_together"] == 0.0
    assert summary["pct_days_offsetting_pnl"] == 100.0
    assert summary["overlap_rate_pct"] == 100.0


def test_micro_portfolio_campaign_smoke_outputs_requested_files(tmp_path: Path) -> None:
    mnq_root = tmp_path / "mnq_export"
    m2k_root = tmp_path / "m2k_export"
    mes_root = tmp_path / "mes_export"
    _write_synthetic_motor_export(mnq_root, variant_name="mnq_variant", daily_pnls=[100.0, -50.0, 40.0, 10.0], trade_pnls=[100.0, -50.0, 40.0, 10.0], trade_risks=[50.0, 50.0, 50.0, 50.0])
    _write_synthetic_motor_export(m2k_root, variant_name="m2k_variant", daily_pnls=[20.0, 30.0, -10.0, 0.0], trade_pnls=[20.0, 30.0, -10.0, 0.0], trade_risks=[25.0, 25.0, 25.0, 25.0])
    _write_synthetic_motor_export(mes_root, variant_name="mes_variant", daily_pnls=[0.0, 10.0, 25.0, -5.0], trade_pnls=[0.0, 10.0, 25.0, -5.0], trade_risks=[20.0, 20.0, 20.0, 20.0])

    motor_configs = {
        "MNQ_default": MotorConfigSpec("MNQ_default", "MNQ", "default", "mnq_variant", 0.0015, 4, mnq_root, "synthetic"),
        "M2K_default": MotorConfigSpec("M2K_default", "M2K", "default", "m2k_variant", 0.0030, 6, m2k_root, "synthetic"),
        "MES_default": MotorConfigSpec("MES_default", "MES", "default", "mes_variant", 0.0020, 6, mes_root, "synthetic"),
    }
    portfolio_variants = [
        PortfolioVariantSpec("MNQ_only__standalone__core_default", "MNQ_only", "single", "standalone", "core_default", ("MNQ_default",), "single synthetic"),
        PortfolioVariantSpec("MNQ_M2K__equal_weight_risk_budget__core_default", "MNQ_M2K", "pair", "equal_weight_risk_budget", "core_default", ("MNQ_default", "M2K_default"), "pair synthetic"),
        PortfolioVariantSpec("MNQ_M2K_MES__equal_weight_risk_budget__core_default", "MNQ_M2K_MES", "three_way", "equal_weight_risk_budget", "core_default", ("MNQ_default", "M2K_default", "MES_default"), "three synthetic"),
    ]

    output_dir = run_campaign(
        output_root=tmp_path / "portfolio_exports",
        initial_capital_usd=50_000.0,
        portfolio_risk_cap_usd=250.0,
        motor_configs=motor_configs,
        portfolio_variants=portfolio_variants,
    )

    expected_files = {
        "summary_by_portfolio.csv",
        "summary_oos_only.csv",
        "daily_equity_by_portfolio.csv",
        "daily_pnl_by_motor.csv",
        "correlation_matrix_daily.csv",
        "diversification_summary.csv",
        "prop_constraints_summary.csv",
        "final_report.md",
        "equity_curves_oos.png",
        "drawdown_curves_oos.png",
        "daily_pnl_correlation_heatmap.png",
        "diversification_contribution_chart.png",
        "prop_score_by_portfolio.png",
    }
    assert expected_files.issubset({path.name for path in output_dir.iterdir()})

    summary = pd.read_csv(output_dir / "summary_by_portfolio.csv")
    assert len(summary) == 3
    assert set(summary["portfolio_family"]) == {"single", "pair", "three_way"}
