from __future__ import annotations

import numpy as np
import pandas as pd

from research.portfolio_integration_phase1.run_portfolio_integration_phase1 import (
    SLEEVES,
    allocation_weights,
    connected_clusters,
    normalize_trades,
)


def test_normalize_trades_uses_signed_quantity_and_utc() -> None:
    source = pd.DataFrame(
        {
            "session_date": ["2025-01-02", "2025-01-02"],
            "direction": ["long", "short"],
            "quantity": [2, 1],
            "entry_time": ["2025-01-02 09:30:00-05:00", "2025-01-02 10:00:00-05:00"],
            "exit_time": ["2025-01-02 09:45:00-05:00", "2025-01-02 10:30:00-05:00"],
            "net_pnl_usd": [10.0, -5.0],
        }
    )
    result = normalize_trades(source, "mnq_orb_vwap_atr")
    assert result["signed_quantity"].tolist() == [2.0, -1.0]
    assert str(result["entry_ts_utc"].dt.tz) == "UTC"


def test_connected_clusters_uses_threshold_graph_components() -> None:
    correlation = pd.DataFrame(np.eye(len(SLEEVES)), index=SLEEVES, columns=SLEEVES)
    correlation.loc[SLEEVES[0], SLEEVES[1]] = 0.55
    correlation.loc[SLEEVES[1], SLEEVES[0]] = 0.55
    correlation.loc[SLEEVES[1], SLEEVES[2]] = 0.45
    correlation.loc[SLEEVES[2], SLEEVES[1]] = 0.45
    clusters = connected_clusters(correlation, 0.40)
    assert sorted(clusters[0]) == sorted(SLEEVES[:3])
    assert len(clusters) == 4


def test_naive_allocation_weights_are_long_only_and_sum_to_one() -> None:
    rng = np.random.default_rng(7)
    pnl = pd.DataFrame(rng.normal(size=(300, len(SLEEVES))), columns=SLEEVES)
    weights, diagnostics = allocation_weights(pnl)
    for allocation in ("equal_sleeve_weight", "inverse_volatility", "equal_risk_contribution"):
        values = weights[allocation].to_numpy(float)
        assert np.isclose(values.sum(), 1.0)
        assert np.all(values > 0)
    assert diagnostics["erc_max_budget_error"] <= 1e-10
