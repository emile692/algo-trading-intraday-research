import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "intraday_periodicity_phase1_20261003"


def test_frozen_design_and_assets() -> None:
    meta = json.loads((OUT / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["lookbacks"] == [10, 20, 40]
    assert list(meta["windows_ct"]) == ["A_0830_0900_CT", "B_0900_0930_CT", "C_1430_1500_CT"]
    assert meta["selected_instruments"] == ["NQ", "ZN", "6E", "MGC", "MCL"]
    assert meta["frozen_sleeves_modified"] is False
    assert meta["gotobi_modified"] is False
    assert meta["topstep_simulation_run"] is False


def test_outputs_cover_design_without_parameter_search() -> None:
    ts = pd.read_csv(OUT / "timeseries_results.csv")
    cross = pd.read_csv(OUT / "cross_section_results.csv")
    assert set(ts.lookback) == {10, 20, 40}
    assert set(ts.window) == {"A_0830_0900_CT", "B_0900_0930_CT", "C_1430_1500_CT"}
    assert set(cross.lookback) == {10, 20, 40}
    assert set(cross.window) == {"A_0830_0900_CT", "B_0900_0930_CT", "C_1430_1500_CT", "ALL_WINDOWS"}
    assert set(cross.cost_scenario) == {"gross_pnl", "commission_only", "commission_plus_1tick_side", "commission_plus_2ticks_side"}


def test_strictly_causal_sample_and_placebos() -> None:
    meta = json.loads((OUT / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["common_start"] >= "2021-07-12"
    placebo = pd.read_csv(OUT / "placebo_results.csv")
    assert {"same_slot_reference", "extra_one_day_lag", "previous_slot_predicts_other_slot"}.issubset(set(placebo.placebo))
    assert not placebo.empty


def test_required_artifacts_and_conditional_orthogonality() -> None:
    required = [
        "PHASE1_REPORT.md", "universe_audit.csv", "timeseries_results.csv", "cross_section_results.csv",
        "cost_analysis.csv", "temporal_stability.csv", "placebo_results.csv", "concentration_analysis.csv",
    ]
    assert all((OUT / name).exists() for name in required)
    verdict = json.loads((OUT / "run_metadata.json").read_text(encoding="utf-8"))["verdict"]
    if verdict != "PROMISING_FOR_VALIDATION":
        assert not (OUT / "orthogonality.csv").exists()
