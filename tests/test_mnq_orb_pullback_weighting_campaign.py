from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytics.mnq_orb_pullback_weighting_campaign import (
    DEFAULT_WEIGHT_PAIRS,
    WeightingCampaignConfig,
    _validate_weight_pairs,
    run_campaign,
)


def _synthetic_daily_source(path: Path, n_days: int = 96) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    idx = np.arange(n_days, dtype=float)
    frame = pd.DataFrame(
        {
            "session_date": dates,
            "orb_return": 0.00025 + 0.0030 * np.sin(idx / 5.0),
            "pullback_return": 0.00032 + 0.0016 * np.cos(idx / 7.0),
            "benchmark_return": 0.00020 + 0.0060 * np.sin(idx / 11.0),
            "orb_daily_trade_count": (idx.astype(int) % 5 == 0).astype(int),
            "pullback_daily_trade_count": (idx.astype(int) % 3 == 0).astype(int),
        }
    )
    frame.to_csv(path, index=False)
    return frame


def test_default_weights_are_valid() -> None:
    _validate_weight_pairs(DEFAULT_WEIGHT_PAIRS)
    for w_orb, w_pullback in DEFAULT_WEIGHT_PAIRS:
        assert w_orb >= 0.0
        assert w_pullback >= 0.0
        assert w_orb + w_pullback == pytest.approx(1.0)


def test_invalid_weights_are_rejected() -> None:
    with pytest.raises(ValueError):
        _validate_weight_pairs(((0.6, 0.5),))
    with pytest.raises(ValueError):
        _validate_weight_pairs(((-0.1, 1.1),))


def test_weighting_campaign_smoke_exports_and_no_is_oos_leakage(tmp_path: Path) -> None:
    source_path = tmp_path / "synthetic_daily_source.csv"
    source = _synthetic_daily_source(source_path)
    output_dir = tmp_path / "weighting_run"

    artifacts = run_campaign(
        WeightingCampaignConfig(
            output_dir=output_dir,
            daily_source_path=source_path,
            weight_pairs=((0.20, 0.80), (0.50, 0.50), (0.80, 0.20)),
        )
    )

    expected_exports = [
        "ranking_weights_oos",
        "ranking_weights_is",
        "weighting_subperiods",
        "weighting_pairwise_correlation",
        "best_config_summary",
        "final_report",
    ]
    for key in expected_exports:
        assert artifacts[key].exists(), key

    ranking_oos = pd.read_csv(artifacts["ranking_weights_oos"])
    required_metric_cols = {
        "net_profit_usd",
        "cagr_pct",
        "sharpe",
        "sortino",
        "profit_factor_daily",
        "max_drawdown_usd",
        "max_daily_drawdown_usd",
        "calmar",
        "days_traded",
        "orb_pullback_daily_corr",
        "return_over_drawdown",
        "composite_score",
    }
    assert required_metric_cols.issubset(ranking_oos.columns)
    assert pd.to_numeric(ranking_oos["net_profit_usd"], errors="coerce").abs().max() > 0.0
    assert pd.to_numeric(ranking_oos["sharpe"], errors="coerce").abs().max() > 0.0
    assert {"static", "risk_scaled", "inverse_risk"}.issubset(set(ranking_oos["method"]))
    assert "baseline__orb_standalone" in set(ranking_oos["variant_name"])
    assert "baseline__pullback_standalone" in set(ranking_oos["variant_name"])
    assert "static__orb50_pull50" in set(ranking_oos["variant_name"])

    tradable = ranking_oos.loc[ranking_oos["method"].ne("baseline")].copy()
    assert (tradable["nominal_orb_weight"] >= 0.0).all()
    assert (tradable["nominal_pullback_weight"] >= 0.0).all()
    assert np.allclose(tradable["nominal_orb_weight"] + tradable["nominal_pullback_weight"], 1.0)
    assert (tradable["effective_orb_weight"] >= 0.0).all()
    assert (tradable["effective_pullback_weight"] >= 0.0).all()
    assert np.allclose(tradable["effective_orb_weight"] + tradable["effective_pullback_weight"], 1.0)

    summary = json.loads(artifacts["best_config_summary"].read_text(encoding="utf-8"))
    calibration = summary["calibration"]
    split_idx = int(len(source) * 0.70)
    is_source = source.iloc[:split_idx]
    assert calibration["calibration_scope"] == "is"
    assert calibration["risk_measure"] == "daily_return_volatility"
    assert calibration["orb_is_vol"] == pytest.approx(float(is_source["orb_return"].std(ddof=0)))
    assert calibration["pullback_is_vol"] == pytest.approx(float(is_source["pullback_return"].std(ddof=0)))

    subperiods = pd.read_csv(artifacts["weighting_subperiods"])
    assert {"oos_full", "oos_first_half", "oos_second_half"}.issubset(set(subperiods["scope"]))

    corr = pd.read_csv(artifacts["weighting_pairwise_correlation"])
    assert corr.loc[corr["scope"].eq("is"), "calibration_used"].iloc[0] in {True, "True"}
    assert corr.loc[corr["scope"].eq("oos"), "calibration_used"].iloc[0] in {False, "False"}
