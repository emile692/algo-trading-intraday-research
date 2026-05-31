from __future__ import annotations

import nbformat

from src.analytics.build_mnq_orb_sizing_3state_client_notebook import (
    REPORT_EXPORTS_ROOT,
    build_notebook,
    find_latest_export,
)


def test_build_notebook_contains_retained_overlay_and_heatmaps() -> None:
    regime_export_root = find_latest_export("mnq_orb_regime_filter_sizing")
    stress_export_root = find_latest_export("mnq_orb_3state_high_bucket_stress", exports_root=REPORT_EXPORTS_ROOT)

    notebook = build_notebook(regime_export_root, stress_export_root)

    nbformat.validate(notebook)

    sources = "\n".join(str(cell.source) for cell in notebook.cells)
    assert "realized_vol_ratio_15_60" in sources
    assert "high = 0.25x" in sources
    assert "Heatmap IS" in sources
    assert len(notebook.cells) >= 20
