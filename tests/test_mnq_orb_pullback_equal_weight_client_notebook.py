from __future__ import annotations

import sys
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytics.build_mnq_orb_pullback_equal_weight_client_notebook import (
    DEFAULT_ORB_EXPORT_ROOT,
    DEFAULT_PULLBACK_EXPORT_ROOT,
    build_notebook,
)


def test_build_notebook_contains_client_controls_and_benchmark() -> None:
    notebook = build_notebook(DEFAULT_ORB_EXPORT_ROOT, DEFAULT_PULLBACK_EXPORT_ROOT)

    nbformat.validate(notebook)

    sources = "\n".join(str(cell.source) for cell in notebook.cells)
    assert "Full MNQ Client Notebook" in sources
    assert "Parametrage client" in sources
    assert "ORB_WEIGHT" in sources
    assert "PULLBACK_WEIGHT" in sources
    assert "ORB_LEVERAGE" in sources
    assert "PULLBACK_LEVERAGE" in sources
    assert "BLEND_LEVERAGE" in sources
    assert "BENCHMARK_LEVERAGE" in sources
    assert "LEVERAGE_GRID" in sources
    assert "Leverage Sensitivity - Blend" in sources
    assert "BENCHMARK_LABEL" in sources
    assert "benchmark_oos" in sources
    assert "Client Scorecard - OOS" in sources
    assert len(notebook.cells) >= 8
