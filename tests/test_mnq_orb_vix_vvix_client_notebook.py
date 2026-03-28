from __future__ import annotations

import nbformat

from src.analytics.build_mnq_orb_vix_vvix_client_notebook import build_notebook, find_latest_export


def test_build_notebook_contains_variant_and_visual_sections() -> None:
    export_root = find_latest_export("mnq_orb_vix_vvix_validation")

    notebook = build_notebook(export_root)

    nbformat.validate(notebook)

    sources = "\n".join(str(cell.source) for cell in notebook.cells)
    assert "filter_drop_low__vvix_pct_63_t1" in sources
    assert "vvix_pct_63_t1" in sources
    assert "Drawdown - OOS only" in sources
    assert len(notebook.cells) >= 16
