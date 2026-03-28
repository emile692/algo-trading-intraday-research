from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.mnq_orb_vvix_sizing_modulation_campaign import (
    VvixSizingVariantRun,
    build_sizing_component_comparison_summary,
    build_vvix_modulation_controls,
    combine_risk_multipliers,
)


def _summary_row(scope: str, **metrics: float | int) -> dict[str, float | int | str]:
    base = {
        "scope": scope,
        "net_pnl": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "profit_factor": 0.0,
        "expectancy": 0.0,
        "max_drawdown": 0.0,
        "n_trades": 0,
        "n_days_traded": 0,
        "pct_days_traded": 0.0,
        "hit_rate": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "stop_hit_rate": 0.0,
        "target_hit_rate": 0.0,
        "exposure_time_pct": 0.0,
    }
    base.update(metrics)
    return base


def _variant(
    name: str,
    oos_metrics: dict[str, float | int],
    overall_metrics: dict[str, float | int] | None = None,
) -> VvixSizingVariantRun:
    overall = overall_metrics or oos_metrics
    summary = pd.DataFrame(
        [
            _summary_row("overall", **overall),
            _summary_row("is", **overall),
            _summary_row("oos", **oos_metrics),
        ]
    )
    empty = pd.DataFrame()
    return VvixSizingVariantRun(
        name=name,
        category="core",
        source_variant_name=None,
        primary_reference_variant_name="baseline_nominal",
        is_core_configuration=True,
        family="demo",
        feature_name="vvix_pct_63_t1",
        combination_mode=None,
        description=name,
        uses_dynamic_sizing=True,
        uses_3state_sizing=False,
        uses_hard_filter_reference=False,
        parameters={},
        controls=empty,
        trades=empty,
        daily_results=empty,
        summary_by_scope=summary,
    )


def test_build_vvix_modulation_controls_maps_bucket_multipliers() -> None:
    session_context = pd.DataFrame(
        {
            "session_date": [
                pd.Timestamp("2024-01-02").date(),
                pd.Timestamp("2024-01-03").date(),
                pd.Timestamp("2024-01-04").date(),
            ],
            "phase": ["is", "is", "oos"],
            "vvix_pct_63_t1": [0.15, 0.45, 0.80],
        }
    )

    controls = build_vvix_modulation_controls(
        session_context=session_context,
        feature_name="vvix_pct_63_t1",
        bucket_labels=pd.Series(["low", "mid", "high"]),
        bucket_multipliers={"low": 0.5, "mid": 1.0, "high": 0.75},
    )

    assert controls["vvix_multiplier"].tolist() == [0.5, 1.0, 0.75]
    assert controls["risk_multiplier"].tolist() == [0.5, 1.0, 0.75]
    assert controls["skip_trade"].tolist() == [False, False, False]


def test_combine_risk_multipliers_supports_multiplicative_and_cap() -> None:
    sizing = pd.Series([0.5, 1.0, 0.75])
    vvix = pd.Series([0.5, 0.5, 1.0])

    multiplicative = combine_risk_multipliers(sizing, vvix, mode="multiplicative")
    cap = combine_risk_multipliers(sizing, vvix, mode="cap")

    assert multiplicative.tolist() == pytest.approx([0.25, 0.5, 0.75])
    assert cap.tolist() == pytest.approx([0.5, 0.5, 0.75])


def test_build_sizing_component_comparison_summary_reports_interaction_excess() -> None:
    variants = {
        "baseline_nominal": _variant(
            "baseline_nominal",
            {
                "net_pnl": 100.0,
                "sharpe": 1.0,
                "sortino": 1.0,
                "profit_factor": 1.10,
                "expectancy": 10.0,
                "max_drawdown": -1000.0,
                "n_trades": 100,
                "n_days_traded": 100,
                "hit_rate": 0.50,
                "avg_win": 100.0,
                "avg_loss": -80.0,
                "stop_hit_rate": 0.30,
            },
        ),
        "baseline_3state": _variant(
            "baseline_3state",
            {
                "net_pnl": 130.0,
                "sharpe": 1.2,
                "sortino": 1.2,
                "profit_factor": 1.20,
                "expectancy": 13.0,
                "max_drawdown": -900.0,
                "n_trades": 100,
                "n_days_traded": 100,
                "hit_rate": 0.52,
                "avg_win": 102.0,
                "avg_loss": -79.0,
                "stop_hit_rate": 0.28,
            },
        ),
        "baseline_vvix_modulator": _variant(
            "baseline_vvix_modulator",
            {
                "net_pnl": 120.0,
                "sharpe": 1.1,
                "sortino": 1.1,
                "profit_factor": 1.15,
                "expectancy": 12.0,
                "max_drawdown": -800.0,
                "n_trades": 80,
                "n_days_traded": 80,
                "hit_rate": 0.51,
                "avg_win": 101.0,
                "avg_loss": -78.0,
                "stop_hit_rate": 0.27,
            },
        ),
        "baseline_3state_vvix_modulator": _variant(
            "baseline_3state_vvix_modulator",
            {
                "net_pnl": 170.0,
                "sharpe": 1.5,
                "sortino": 1.5,
                "profit_factor": 1.30,
                "expectancy": 17.0,
                "max_drawdown": -700.0,
                "n_trades": 80,
                "n_days_traded": 80,
                "hit_rate": 0.54,
                "avg_win": 104.0,
                "avg_loss": -77.0,
                "stop_hit_rate": 0.24,
            },
        ),
    }

    summary = build_sizing_component_comparison_summary(variants)
    interaction = summary.loc[
        (summary["comparison_type"] == "interaction")
        & (summary["comparison_name"] == "interaction_excess_vs_additive")
        & (summary["scope"] == "oos")
    ].iloc[0]

    assert float(interaction["combined_sharpe_delta"]) == pytest.approx(0.5)
    assert float(interaction["additive_sharpe_delta"]) == pytest.approx(0.3)
    assert float(interaction["interaction_excess_sharpe_delta"]) == pytest.approx(0.2)
