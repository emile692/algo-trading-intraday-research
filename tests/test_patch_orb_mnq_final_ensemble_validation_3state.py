from __future__ import annotations

from pathlib import Path

import nbformat

from src.analytics.patch_orb_mnq_final_ensemble_validation_3state import NOTEBOOK_PATH, patch_notebook


def test_patch_mnq_notebook_adds_3state_controls(tmp_path: Path) -> None:
    target = tmp_path / "orb_MNQ_final_ensemble_validation.ipynb"
    target.write_text(NOTEBOOK_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    patch_notebook(target)

    notebook = nbformat.read(target, as_version=4)
    content = "\n".join(str(cell.source) for cell in notebook.cells)

    assert "ENABLE_3STATE_COMPARISON" in content
    assert "SIZING_FEATURE_NAME" in content
    assert "ensemble_3state" in content
    assert "build_regime_dataset" in content
    assert "ENABLE_VVIX_FILTER_COMPARISON" in content
    assert "VVIX_FILTER_VARIANT_NAME" in content
    assert "ensemble_vvix_filter" in content
    assert "build_vvix_filter_controls" in content
