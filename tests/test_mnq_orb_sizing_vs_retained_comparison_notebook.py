from __future__ import annotations

import nbformat

from src.analytics.build_mnq_orb_sizing_vs_retained_comparison_notebook import build_notebook, find_latest_export


def test_build_notebook_contains_core_comparison_sections() -> None:
    regime_export_root = find_latest_export("mnq_orb_regime_filter_sizing")
    prop_export_root = find_latest_export("mnq_orb_prop_challenge_readiness")

    notebook = build_notebook(regime_export_root, prop_export_root)

    nbformat.validate(notebook)

    sources = "\n".join(str(cell.source) for cell in notebook.cells)
    assert "sizing_3state_realized_vol_ratio_15_60" in sources
    assert "full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate" in sources
    assert "MNQ ORB comparison - sizing_3state vs retained final" in sources
    assert len(notebook.cells) >= 16
