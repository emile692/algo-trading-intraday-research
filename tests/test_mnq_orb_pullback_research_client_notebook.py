from __future__ import annotations

import sys
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytics.build_mnq_orb_pullback_research_client_notebook import build_notebook


def test_research_notebook_recomputes_signals_and_documents_parameters() -> None:
    notebook = build_notebook()

    nbformat.validate(notebook)

    sources = "\n".join(str(cell.source) for cell in notebook.cells)
    assert "Notebook de recherche complet" in sources
    assert "load_symbol_data" in sources
    assert "prepare_minute_dataset" in sources
    assert "build_candidate_universe" in sources
    assert "_evaluate_experiment(" in sources
    assert "full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate" in sources
    assert "noise_area_gate" in sources
    assert "run_volume_climax_pullback_v2_backtest(" in sources
    assert "ORB heatmaps" in sources
    assert "Pullback heatmaps" in sources
    assert "parameter_table" in sources
    assert "ORB_RISK_PER_TRADE_PCT" in sources
    assert "PULLBACK_COMPOUND_REALIZED_PNL" in sources
    assert "ORB_FIXED_CONTRACTS" not in sources
    assert "summary_by_variant.csv" not in sources
    assert "daily_equity_by_variant.csv" not in sources
    assert len(notebook.cells) >= 12
