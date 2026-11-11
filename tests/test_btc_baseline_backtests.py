from __future__ import annotations

import json
from pathlib import Path

from research.baseline_backtests.baseline_engine import load_frozen_specs
from research.btc_baseline_backtests.run_btc_baselines import adapt_benchmark_to_btc, adapt_spec_to_btc


ROOT = Path(__file__).resolve().parents[1]


def test_btc_adaptation_only_changes_execution_metadata_and_session_relative_clocks() -> None:
    frozen = ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
    specs, benchmark, _ = load_frozen_specs(frozen, ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv")
    config = json.loads((ROOT / "research" / "btc_baseline_backtests" / "btc_baseline_config.json").read_text())
    adapted = [adapt_spec_to_btc(spec, config) for spec in specs]
    assert len(adapted) == 16
    assert all(spec["canonical_parameters"]["execution"]["tick_size"] == 5.0 for spec in adapted)
    assert all(spec["canonical_parameters"]["execution"]["point_value_usd"] == 5.0 for spec in adapted)
    assert all(spec["_source_frozen_spec_sha256"] for spec in adapted)
    assert next(spec for spec in adapted if spec["mechanism_id"] == "gap_fade")["canonical_parameters"]["execution"]["last_entry_time"] == "09:31:00"
    assert adapt_benchmark_to_btc(benchmark, config)["mechanism_id"] == "BTC_BUY_AND_HOLD"
