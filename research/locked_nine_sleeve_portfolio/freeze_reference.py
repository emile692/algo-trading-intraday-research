"""Materialize the user-approved nine-sleeve equal-contract reference lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "research" / "mbt_nine_sleeve_topstep_20261004"
ENSEMBLE = REPO / "research" / "mbt_three_sleeve_freeze_20261004" / "vote_ensemble"
SIX = REPO / "research" / "portfolio_integration_phase1"
SLEEVES = [
    "mnq_orb_vwap_atr",
    "mnq_volatility_breakout",
    "mnq_compression_breakout",
    "mnq_atr_trend_flip",
    "mgc_volatility_breakout",
    "mgc_gap_continuation",
    "mbt_rolling_range_breakout",
    "mbt_trend_pullback_ma_structure",
    "mbt_volatility_breakout",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(REPO)).replace("\\", "/")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    mappings = pd.read_csv(STUDY / "integer_mapping_table.csv")
    selected = mappings[(mappings.allocation_family == "equal_contract") & (mappings.book == "nine")]
    observed = selected.set_index("sleeve").integer_copies.reindex(SLEEVES)
    if observed.isna().any() or not observed.eq(1).all():
        raise RuntimeError(f"Expected one integer copy of every sleeve, got {observed.to_dict()}")

    summary = pd.read_csv(STUDY / "topstep_summary.csv")
    reference = summary[
        summary.method.eq("moving_block_bootstrap")
        & summary.scenario.eq("equal_contract__nominal__nine")
        & summary.block.astype(str).eq("pooled")
    ].sort_values("horizon")
    if reference.horizon.tolist() != [35, 60]:
        raise RuntimeError("Missing canonical 35d/60d Topstep rows")
    row60 = reference[reference.horizon.eq(60)].iloc[0]
    if abs(float(row60.pass_probability) - 0.452375) > 1e-12:
        raise RuntimeError("The approved 45.2375% PASS reference was not reproduced")

    source_daily = pd.read_csv(STUDY / "daily_nine_sleeves_nominal.csv", parse_dates=["session_date"])
    missing = set(SLEEVES) - set(source_daily.columns)
    if missing:
        raise RuntimeError(f"Missing sleeves from canonical daily source: {sorted(missing)}")
    daily = source_daily[["session_date", *SLEEVES]].copy()
    daily["portfolio_pnl_usd"] = daily[SLEEVES].sum(axis=1)
    daily.to_csv(HERE / "daily_pnl.csv", index=False, float_format="%.8f")

    sleeve_rows = []
    for sleeve in SLEEVES:
        instrument = sleeve.split("_")[0].upper()
        source_type = "legacy_frozen_sleeve" if sleeve in SLEEVES[:6] else "mbt_vote_ensemble"
        sleeve_rows.append({
            "sleeve": sleeve,
            "instrument": instrument,
            "integer_copies": 1,
            "source_type": source_type,
            "status": "LOCKED_ACTIVE_REFERENCE",
        })
    pd.DataFrame(sleeve_rows).to_csv(HERE / "sleeve_manifest.csv", index=False)
    reference.to_csv(HERE / "topstep_reference_metrics.csv", index=False)

    input_paths = {
        "six_sleeve_daily_pnl": SIX / "daily_sleeve_pnl.csv",
        "six_sleeve_timestamp_portfolio": SIX / "common_timestamp_portfolio.csv",
        "mbt_ensemble_manifest": ENSEMBLE / "ensemble_manifest.json",
        "mbt_ensemble_daily": ENSEMBLE / "daily_ensemble_results.csv",
        "nine_sleeve_study_metadata": STUDY / "run_metadata.json",
        "nine_sleeve_mapping_table": STUDY / "integer_mapping_table.csv",
        "nine_sleeve_topstep_summary": STUDY / "topstep_summary.csv",
        "nine_sleeve_nominal_daily_source": STUDY / "daily_nine_sleeves_nominal.csv",
    }
    config = {
        "schema_version": "1.0",
        "portfolio_id": "nine_sleeve_equal_contract_reference_20261004",
        "status": "LOCKED_ACTIVE_REFERENCE",
        "decision_date": "2026-10-04",
        "decision_authority": "explicit_user_promotion",
        "supersedes_active_reference": "six_sleeve_reference_20260926",
        "superseded_reference_retained_for_audit": relative(SIX),
        "allocation_family": "equal_contract",
        "mapping": {sleeve: 1 for sleeve in SLEEVES},
        "portfolio_rule": "sum the executable PnL of one whole-contract copy of each frozen sleeve; no fractional scaling",
        "mbt_aggregation_manifest": relative(ENSEMBLE / "ensemble_manifest.json"),
        "topstep_reference": {
            "account": "Trading Combine 50K",
            "rules": {"profit_target_usd": 3000, "mll_usd": 2000, "consistency": 0.55},
            "method": "moving_block_bootstrap_pooled_blocks_3_5_10_20",
            "paths": 8000,
            "seed": 20261004,
            "horizon_35d": reference[reference.horizon.eq(35)].iloc[0][
                ["pass_probability", "mll_probability", "expiry_probability"]
            ].to_dict(),
            "horizon_60d": reference[reference.horizon.eq(60)].iloc[0][
                ["pass_probability", "mll_probability", "expiry_probability"]
            ].to_dict(),
        },
        "governance": {
            "no_retuning": True,
            "whole_contracts_only": True,
            "pass_rate_optimization_performed": False,
            "prospective_start_after": "2026-10-02",
            "caveat": "45.2375% is a retrospective bootstrap estimate, not a guaranteed pass probability or pristine OOS result.",
        },
        "locked_inputs": {
            key: {"path": relative(path), "sha256": sha256(path)} for key, path in input_paths.items()
        },
    }
    (HERE / "locked_nine_sleeve_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    outputs = [
        HERE / "locked_nine_sleeve_config.json",
        HERE / "daily_pnl.csv",
        HERE / "sleeve_manifest.csv",
        HERE / "topstep_reference_metrics.csv",
    ]
    lock = {
        "portfolio_id": config["portfolio_id"],
        "status": config["status"],
        "files": {relative(path): sha256(path) for path in outputs},
    }
    (HERE / "lock_manifest.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2))
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
