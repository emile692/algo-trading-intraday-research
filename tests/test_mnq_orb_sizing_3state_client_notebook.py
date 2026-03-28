from __future__ import annotations

import nbformat

from src.analytics.build_mnq_orb_sizing_3state_client_notebook import build_notebook, find_latest_export


def test_build_notebook_contains_variant_and_topstep_sections() -> None:
    regime_export_root = find_latest_export("mnq_orb_regime_filter_sizing")
    topstep_export_root = find_latest_export("mnq_orb_topstep_50k_simulation")

    notebook = build_notebook(regime_export_root, topstep_export_root)

    nbformat.validate(notebook)

    sources = "\n".join(str(cell.source) for cell in notebook.cells)
    assert "sizing_3state_realized_vol_ratio_15_60" in sources
    assert "TopstepX 50K" in sources
    assert len(notebook.cells) >= 20
