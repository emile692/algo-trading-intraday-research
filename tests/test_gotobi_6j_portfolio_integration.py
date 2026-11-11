import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "gotobi_6j_portfolio_integration_20261003"
SCALING = ROOT / "research" / "topstep_portfolio_scaling"


def test_reference_gates_and_freeze_flags():
    meta = json.loads((OUT / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["six_sleeve_reference_gate_passed"]
    assert meta["topstep_historical_and_bootstrap_gate_passed"]
    assert not meta["sleeve_sources_modified"]
    assert not meta["parameter_retuning"]
    assert meta["gotobi_events_common_period"] == 24


def test_nominal_and_risk_matched_tables_are_separate():
    nominal = pd.read_csv(OUT / "book_6_vs_7_metrics.csv")
    risk_matched = pd.read_csv(OUT / "book_6_vs_7_risk_matched.csv")
    assert set(nominal.portfolio) == {"book_6", "book_7_1c", "book_7_2c"}
    assert set(risk_matched.portfolio) == {"book_7_1c_risk_matched", "book_7_2c_risk_matched"}
    assert (risk_matched.implementation == "continuous_global_rescale_diagnostic").all()


def test_book_seven_is_exact_read_only_addition():
    table = pd.read_csv(OUT / "book_6_vs_7_metrics.csv").set_index("portfolio")
    gotobi = pd.read_csv(
        ROOT / "research" / "gotobi_6j_phase2_20261003" / "explicit_contract_trades.csv",
        parse_dates=["japan_date"],
    )
    common = pd.read_csv(
        ROOT / "research" / "portfolio_integration_phase1" / "daily_sleeve_pnl.csv",
        parse_dates=["session_date"],
    )
    incremental = gotobi[gotobi.japan_date.isin(common.session_date)].quote_fill_net_pnl_usd.sum()
    np.testing.assert_allclose(
        table.loc["book_7_1c", "net_pnl_usd"] - table.loc["book_6", "net_pnl_usd"],
        incremental,
        atol=1e-8,
    )


def test_topstep_baseline_reproduces_saved_pooled_results():
    ours = pd.read_csv(OUT / "topstep_6_vs_7.csv")
    ours = ours[
        ours.method.eq("moving_block_bootstrap")
        & ours.scenario.eq("book_6")
        & ours.block_length_sessions.eq("pooled")
    ].set_index("horizon_traded_days")
    reference = pd.read_csv(SCALING / "bootstrap_summary_pooled.csv")
    reference = reference[
        reference.allocation.eq("equal_sleeve_weight") & reference.scaling.eq(6)
    ].set_index("horizon_traded_days")
    for horizon in (35, 60):
        for column in ("pass_probability", "mll_probability", "expiry_probability"):
            np.testing.assert_allclose(ours.loc[horizon, column], reference.loc[horizon, column], atol=1e-12)


def test_contract_collision_audit_is_mechanically_compatible():
    audit = json.loads((OUT / "operational_collision_audit.json").read_text(encoding="utf-8"))
    assert audit["simultaneous_position_minutes"] == 0
    assert audit["with_gotobi_2c_max_simultaneous_contracts"] <= audit["topstep_50k_current_max_contracts"]
