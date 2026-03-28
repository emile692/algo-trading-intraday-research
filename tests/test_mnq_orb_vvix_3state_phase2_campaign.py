from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.mnq_orb_vvix_3state_phase2_campaign import (
    Phase2VariantRun,
    build_component_comparison_summary,
    compose_phase2_controls,
    resolve_3state_overlay_spec,
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


def _variant(name: str, oos_metrics: dict[str, float | int], overall_metrics: dict[str, float | int] | None = None) -> Phase2VariantRun:
    overall = overall_metrics or oos_metrics
    summary = pd.DataFrame(
        [
            _summary_row("overall", **overall),
            _summary_row("is", **overall),
            _summary_row("oos", **oos_metrics),
        ]
    )
    empty = pd.DataFrame()
    return Phase2VariantRun(
        name=name,
        category="core",
        description=name,
        uses_vvix_filter=False,
        vvix_variant_name=None,
        vvix_feature_name=None,
        vvix_kept_buckets=(),
        uses_3state_sizing=False,
        sizing_variant_name=None,
        sizing_feature_name=None,
        parameters={},
        controls=empty,
        trades=empty,
        daily_results=empty,
        summary_by_scope=summary,
    )


def test_resolve_3state_overlay_spec_reads_mapping_and_feature_name(tmp_path) -> None:
    export_root = tmp_path / "mnq_orb_regime_filter_sizing_20260325_150405"
    export_root.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "variant_name": "sizing_3state_realized_vol_ratio_15_60",
                "feature_name": "realized_vol_ratio_15_60",
            }
        ]
    ).to_csv(export_root / "summary_variants.csv", index=False)

    pd.DataFrame(
        [
            {"variant_name": "sizing_3state_realized_vol_ratio_15_60", "bucket_label": "low", "bucket_position": 1, "risk_multiplier": 0.5},
            {"variant_name": "sizing_3state_realized_vol_ratio_15_60", "bucket_label": "mid", "bucket_position": 2, "risk_multiplier": 1.0},
            {"variant_name": "sizing_3state_realized_vol_ratio_15_60", "bucket_label": "high", "bucket_position": 3, "risk_multiplier": 0.75},
        ]
    ).to_csv(export_root / "regime_state_mappings.csv", index=False)

    spec = resolve_3state_overlay_spec(export_root=export_root)

    assert spec.variant_name == "sizing_3state_realized_vol_ratio_15_60"
    assert spec.feature_name == "realized_vol_ratio_15_60"
    assert spec.bucket_multipliers == {"low": 0.5, "mid": 1.0, "high": 0.75}


def test_compose_phase2_controls_applies_vvix_veto_on_top_of_sizing() -> None:
    session_context = pd.DataFrame(
        {
            "session_date": [pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-03").date()],
            "phase": ["is", "oos"],
            "breakout_side": ["long", "short"],
            "breakout_timing_bucket": ["early", "mid"],
        }
    )
    vvix_controls = pd.DataFrame(
        {
            "session_date": [pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-03").date()],
            "feature_name": ["vvix_pct_63_t1", "vvix_pct_63_t1"],
            "vvix_pct_63_t1": [0.10, 0.60],
            "bucket_label": ["low", "high"],
            "selected": [False, True],
            "skip_trade": [True, False],
            "kept_buckets": ["mid,high", "mid,high"],
        }
    )
    sizing_controls = pd.DataFrame(
        {
            "session_date": [pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-03").date()],
            "feature_name": ["realized_vol_ratio_15_60", "realized_vol_ratio_15_60"],
            "feature_value": [0.8, 1.1],
            "bucket_label": ["low", "mid"],
            "risk_multiplier": [0.5, 1.0],
        }
    )

    controls = compose_phase2_controls(
        session_context=session_context,
        vvix_controls=vvix_controls,
        sizing_controls=sizing_controls,
    )

    assert controls["vvix_selected"].tolist() == [False, True]
    assert controls["sizing_risk_multiplier"].tolist() == [0.5, 1.0]
    assert controls["risk_multiplier"].tolist() == [0.0, 1.0]
    assert controls["skip_trade"].tolist() == [True, False]


def test_build_component_comparison_summary_reports_interaction_excess() -> None:
    variants = {
        "baseline_nominal": _variant(
            "baseline_nominal",
            {"net_pnl": 100.0, "sharpe": 1.0, "sortino": 1.0, "profit_factor": 1.10, "expectancy": 10.0, "max_drawdown": -1000.0, "n_trades": 100, "n_days_traded": 100, "hit_rate": 0.50, "avg_win": 100.0, "avg_loss": -80.0, "stop_hit_rate": 0.30},
        ),
        "baseline_3state": _variant(
            "baseline_3state",
            {"net_pnl": 130.0, "sharpe": 1.2, "sortino": 1.2, "profit_factor": 1.20, "expectancy": 13.0, "max_drawdown": -900.0, "n_trades": 100, "n_days_traded": 100, "hit_rate": 0.52, "avg_win": 102.0, "avg_loss": -79.0, "stop_hit_rate": 0.28},
        ),
        "baseline_vvix_nominal": _variant(
            "baseline_vvix_nominal",
            {"net_pnl": 120.0, "sharpe": 1.1, "sortino": 1.1, "profit_factor": 1.15, "expectancy": 12.0, "max_drawdown": -800.0, "n_trades": 80, "n_days_traded": 80, "hit_rate": 0.51, "avg_win": 101.0, "avg_loss": -78.0, "stop_hit_rate": 0.27},
        ),
        "baseline_vvix_3state": _variant(
            "baseline_vvix_3state",
            {"net_pnl": 170.0, "sharpe": 1.5, "sortino": 1.5, "profit_factor": 1.30, "expectancy": 17.0, "max_drawdown": -700.0, "n_trades": 80, "n_days_traded": 80, "hit_rate": 0.54, "avg_win": 104.0, "avg_loss": -77.0, "stop_hit_rate": 0.24},
        ),
    }

    summary = build_component_comparison_summary(variants)
    interaction = summary.loc[
        (summary["comparison_type"] == "interaction")
        & (summary["comparison_name"] == "interaction_excess_vs_additive")
        & (summary["scope"] == "oos")
    ].iloc[0]

    assert float(interaction["combined_sharpe_delta"]) == pytest.approx(0.5)
    assert float(interaction["additive_sharpe_delta"]) == pytest.approx(0.3)
    assert float(interaction["interaction_excess_sharpe_delta"]) == pytest.approx(0.2)
