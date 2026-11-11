from __future__ import annotations

import pandas as pd

from research.equity_index_gap_reversion_audit_20261007.run_audit import THRESHOLDS, daily_metrics


def test_frozen_thresholds_are_unchanged() -> None:
    assert THRESHOLDS == (0.25, 0.50, 0.75, 1.00)


def test_daily_metrics_has_causal_order_invariant_total() -> None:
    s = pd.Series([10.0, -5.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert daily_metrics(s)["net_pnl"] == 5.0
