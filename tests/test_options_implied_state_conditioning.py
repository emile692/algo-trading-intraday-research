from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.analytics.options_implied_state_conditioning import build_causal_states, load_vix, nw_ols, phase2_overlay


def test_states_use_previous_row_only() -> None:
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    vix = pd.DataFrame({"vix_raw": np.arange(300, dtype=float) + 10, "vix3m_raw": np.arange(300, dtype=float) + 12}, index=dates)
    mes = pd.DataFrame({"mes_rv10_raw_pct": np.arange(300, dtype=float) + 5, "mes_atr14_raw_pct": 2.0}, index=dates)
    states = build_causal_states(vix, mes)
    assert states.loc[dates[10], "vix_level"] == vix.loc[dates[9], "vix_raw"]
    assert states.loc[dates[10], "evrp"] == vix.loc[dates[9], "vix_raw"] - mes.loc[dates[9], "mes_rv10_raw_pct"]


def test_expanding_vix_partition_has_252_day_warmup_and_no_future_dependency() -> None:
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    vix = pd.DataFrame({"vix_raw": np.linspace(10, 30, 300), "vix3m_raw": 20.0}, index=dates)
    mes = pd.DataFrame({"mes_rv10_raw_pct": 15.0, "mes_atr14_raw_pct": 2.0}, index=dates)
    first = build_causal_states(vix, mes)
    modified = vix.copy(); modified.loc[dates[-1], "vix_raw"] = 1000
    second = build_causal_states(modified, mes)
    assert pd.isna(first.loc[dates[251], "vix_state"])
    assert first.loc[dates[280], "vix_state"] == second.loc[dates[280], "vix_state"]


def test_controlled_hac_regression_recovers_external_beta() -> None:
    rng = np.random.default_rng(7); n = 1000
    x = rng.normal(size=n); rv = rng.normal(size=n); atr = rng.normal(size=n)
    frame = pd.DataFrame({"y": 3 * x + 2 * rv - atr + rng.normal(scale=.2, size=n), "x": x, "rv": rv, "atr": atr})
    result = nw_ols(frame, "y", ["x", "rv", "atr"])
    assert abs(result["beta"] - 3) < .05
    assert result["hac_t"] > 20


def test_bloomberg_multiheader_vix_parser_preserves_values() -> None:
    path = Path("data/raw/vix_indexes.csv")
    parsed = load_vix(path)
    assert parsed.loc[pd.Timestamp("2010-01-04"), "vix_raw"] == 20.04
    assert parsed.loc[pd.Timestamp("2010-01-04"), "vix3m_raw"] == 22.77


def test_phase2_overlay_is_one_common_backwardation_rule() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    sleeves = ["a", "b", "c", "d"]
    panel = pd.DataFrame({name: np.arange(1, 11, dtype=float) for name in sleeves}, index=dates)
    panel["term_spread"] = [-1, 1] * 5
    config = {
        "inference": {"bootstrap_samples": 20, "bootstrap_block_days": 2, "random_seed": 3},
        "phase_2": {"rule": "common test rule"},
    }
    summary, _, _, _, decision = phase2_overlay(panel, sleeves, config)
    expected = sum(range(1, 11, 2))
    filtered = summary.loc[summary.variant.eq("common_backwardation_filter")]
    assert (filtered.total_pnl == expected).all()
    assert decision["filtered_sessions"] == 5
