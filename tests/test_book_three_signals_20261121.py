"""BOOK.md section 6: the active eight-sleeve book on its full sample, as three instrument signals (2026-11-21)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "research" / "book_three_signals_20261121"
LOCK = ROOT / "research" / "locked_eight_sleeve_portfolio_20261121"


def _daily() -> pd.DataFrame:
    return pd.read_csv(HERE / "daily_signals.csv", parse_dates=["session_date"]).set_index("session_date")


def test_book_reproduces_the_lock_window() -> None:
    d = _daily()
    lock = pd.read_csv(LOCK / "daily_pnl.csv", parse_dates=["session_date"]).set_index("session_date")
    w = d.loc["2024-01-02":"2026-03-19"]
    assert w.index.equals(lock.index)
    assert np.allclose(w["Book 8"], lock["portfolio_pnl_usd"], atol=1e-6)
    assert abs(float(w["Book 8"].sum()) - 31639.46) < 1e-6


def test_signals_sum_to_book_and_mbt_starts_2021() -> None:
    d = _daily()
    assert d.index.min() == pd.Timestamp("2019-05-06") and d.index.max() == pd.Timestamp("2026-03-19")
    assert d.loc[d["MBT"].notna()].index.min() == pd.Timestamp("2021-05-03")
    assert np.allclose(d[["MNQ", "MGC", "MBT"]].sum(axis=1, min_count=1), d["Book 8"], atol=1e-6)


def test_segments_follow_the_sleeve_is_ends() -> None:
    bounds = json.loads((HERE / "results.json").read_text(encoding="utf-8"))["segment_bounds"]
    assert bounds["Book 8"] == {"IS": ["2019-05-06", "2023-12-29"], "mixte": ["2024-01-02", "2025-05-28"],
                                "OOS": ["2025-05-29", "2026-03-19"]}
    assert bounds["MNQ"]["mixte"] == ["2024-01-02", "2024-02-27"] and bounds["MNQ"]["OOS"][0] == "2024-02-28"
    assert bounds["MGC"] == {"IS": ["2019-05-06", "2023-12-29"], "OOS": ["2024-01-02", "2026-03-19"]}
    assert bounds["MBT"] == {"IS": ["2021-05-03", "2025-05-28"], "OOS": ["2025-05-29", "2026-03-19"]}


def test_published_metrics_and_correlations() -> None:
    m = pd.read_csv(HERE / "signal_metrics.csv").set_index(["series", "segment"])
    assert abs(m.loc[("Book 8", "FS"), "net_pnl_usd"] - 66583.10) < 1e-6
    assert round(m.loc[("Book 8", "FS"), "sharpe"], 2) == 1.68
    assert [round(m.loc[(s, "FS"), "sharpe"], 2) for s in ("MNQ", "MGC", "MBT")] == [1.43, 0.61, 1.13]
    c = pd.read_csv(HERE / "correlation_fs_common.csv", index_col=0)
    assert round(c.loc["MNQ", "MBT"], 2) == 0.16 and abs(c.loc["MNQ", "MGC"]) < 0.05 and abs(c.loc["MGC", "MBT"]) < 0.05
