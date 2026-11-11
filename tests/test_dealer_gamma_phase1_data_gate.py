import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "dealer_gamma_phase1_20261003"


def test_causality_gate_stops_all_diagnostics() -> None:
    metadata = json.loads((OUT / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "STOPPED_CAUSALITY_GATE_FAILED"
    assert metadata["alpha_verdict"] is None
    assert metadata["gex_named_files_found"] == 0
    assert metadata["gex_schema_hits_found"] == 0
    assert metadata["diagnostics_run"] is False
    assert metadata["prototypes_run"] is False
    assert metadata["topstep_run"] is False


def test_requested_outputs_are_explicitly_not_run() -> None:
    names = [
        "gamma_regime_summary.csv",
        "autocorrelation_by_regime.csv",
        "shock_response_by_regime.csv",
        "session_response_by_regime.csv",
        "vwap_reversion_by_regime.csv",
        "temporal_stability.csv",
        "frozen_sleeves_by_gamma.csv",
    ]
    for name in names:
        result = pd.read_csv(OUT / name)
        assert result.status.tolist() == ["NOT_RUN"]
        assert "CAUSALITY_GATE_FAILED" in result.reason.iloc[0]
    assert not (OUT / "prototype_results.csv").exists()


def test_no_frozen_assets_were_changed_by_campaign() -> None:
    metadata = json.loads((OUT / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["frozen_sleeves_modified"] is False
    assert metadata["gotobi_modified"] is False
