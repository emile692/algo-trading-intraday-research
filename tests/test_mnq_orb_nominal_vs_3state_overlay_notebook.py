from __future__ import annotations

import nbformat

from src.analytics.build_mnq_orb_nominal_vs_3state_overlay_notebook import build_notebook, find_latest_export


def test_build_notebook_contains_overlay_comparison_sections() -> None:
    regime_export_root = find_latest_export("mnq_orb_regime_filter_sizing")
    notebook = build_notebook(regime_export_root)

    nbformat.validate(notebook)

    sources = "\n".join(str(cell.source) for cell in notebook.cells)
    assert "sizing_3state_realized_vol_ratio_15_60" in sources
    assert '"nominal"' in sources or "nominal" in sources
    assert "same trade set" in sources.lower() or "même trade set" in sources.lower()
    assert "high = 0.25x" in sources or "HIGH_BUCKET_MULTIPLIER = 0.25" in sources
    assert len(notebook.cells) >= 18
