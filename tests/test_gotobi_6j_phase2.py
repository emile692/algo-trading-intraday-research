from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "gotobi_6j_phase2_20261003"


def test_frozen_explicit_ledger_is_complete_and_uses_no_mid_fills():
    trades = pd.read_csv(OUT / "explicit_contract_trades.csv")
    assert len(trades) == 136
    assert trades.japan_date.nunique() == 136
    assert (trades.entry_ask >= trades.entry_bid).all()
    assert (trades.exit_ask >= trades.exit_bid).all()
    expected = (trades.entry_bid - trades.exit_ask) * 12_500_000 - 4.22
    np.testing.assert_allclose(trades.quote_fill_net_pnl_usd, expected, atol=1e-8)


def test_explicit_contract_opens_replicate_phase1_within_one_tick():
    trades = pd.read_csv(OUT / "explicit_contract_trades.csv")
    assert trades.explicit_minus_phase1_gross_usd.abs().max() <= 6.25 + 1e-6
    assert trades.explicit_open_gross_pnl_usd.corr(trades.phase1_gross_pnl_usd) > 0.999


def test_cost_sensitivity_is_monotone_and_breaks_between_two_and_two_half_ticks():
    table = pd.read_csv(OUT / "execution_cost_sensitivity.csv")
    fixed = table[table.scenario.str.contains("tick_per_side")].sort_values("slippage_ticks_per_side")
    assert fixed.expectancy_usd.is_monotonic_decreasing
    assert fixed.loc[fixed.slippage_ticks_per_side.eq(2), "expectancy_usd"].iloc[0] > 0
    assert fixed.loc[fixed.slippage_ticks_per_side.eq(2.5), "expectancy_usd"].iloc[0] < 0


def test_chronological_blocks_are_frozen_and_exhaustive():
    table = pd.read_csv(OUT / "chronological_validation.csv")
    assert table.n.sum() == 136
    assert table.block.tolist() == [
        "development_replication_2015_2020",
        "intermediate_validation_2021_2023",
        "final_frozen_retrospective_2024_2026",
    ]


def test_all_downloaded_contract_ids_match_phase1_mapping():
    trades = pd.read_csv(OUT / "explicit_contract_trades.csv")
    assert trades.instrument_id.notna().all()
    assert trades.raw_symbol.str.match(r"^6J[HMUZ]\d$").all()
    assert len(list((OUT / "mbp1_events").glob("*.parquet"))) == 136
    assert len(list((OUT / "definition_events").glob("*.parquet"))) == 136
