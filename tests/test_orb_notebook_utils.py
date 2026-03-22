from __future__ import annotations

import pandas as pd

from src.analytics.orb_notebook_utils import (
    build_scope_readout_markdown,
    build_selected_ensemble_kpi_frame,
    curve_max_drawdown_pct,
    curve_total_return_pct,
    normalize_curve,
)


def test_curve_total_return_uses_initial_capital_not_first_curve_point() -> None:
    curve = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "equity": [49_500.0, 35_000.0],
        }
    )

    normalized = normalize_curve(curve)

    assert round(curve_total_return_pct(normalized, 50_000.0), 2) == -30.0
    assert round(curve_max_drawdown_pct(normalized), 2) == -29.29


def test_build_selected_ensemble_kpi_frame_keeps_overall_and_oos_scopes() -> None:
    frame = build_selected_ensemble_kpi_frame(
        {
            "aggregation_rule": "majority_50",
            "overall_sharpe": -0.36,
            "overall_net_pnl": -25_332.5,
            "overall_nb_trades": 1029,
            "oos_sharpe": 0.53,
            "oos_net_pnl": 10_958.5,
            "oos_nb_trades": 299,
        }
    )

    assert list(frame.columns[:3]) == ["model", "aggregation_rule", "overall_score"]
    assert float(frame.loc[0, "overall_sharpe"]) == -0.36
    assert float(frame.loc[0, "oos_sharpe"]) == 0.53
    assert int(frame.loc[0, "overall_trades"]) == 1029
    assert int(frame.loc[0, "oos_trades"]) == 299


def test_scope_readout_calls_out_full_vs_oos_difference() -> None:
    full_curve = normalize_curve(
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-03"], utc=True),
                "equity": [49_500.0, 35_000.0],
            }
        )
    )
    oos_curve = normalize_curve(
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-02-01", "2024-02-03"], utc=True),
                "equity": [50_250.0, 60_000.0],
            }
        )
    )

    text = build_scope_readout_markdown(full_curve, oos_curve, initial_capital=50_000.0)

    assert "Full sample and OOS point in opposite directions" in text
    assert "Read `overall_*` metrics against the full-sample curve" in text
