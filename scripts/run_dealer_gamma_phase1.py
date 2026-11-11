"""Audit the causal data gate for the dealer-gamma Phase 1 study.

The requested return diagnostics must not run unless a point-in-time GEX series
exists. This script records the repository evidence and emits explicit NOT_RUN
artifacts when that prerequisite is absent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "dealer_gamma_phase1_20261003"
NQ = ROOT / "data" / "processed" / "parquet" / "NQ_c_0_1m_20260321_095036.parquet"
MNQ = ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"
VIX = ROOT / "data" / "raw" / "vix_indexes.csv"
CAUSAL_VIX = ROOT / "research" / "options_implied_state_conditioning" / "causal_state_dataset.csv"
FORBIDDEN_PROXY_COLUMNS = {"gex", "gamma", "dealer_gamma", "gamma_exposure"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_summary(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    frame = pd.read_parquet(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "rows": parquet.metadata.num_rows,
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "columns": "|".join(parquet.schema.names),
        "sha256": sha256(path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nq = parquet_summary(NQ)
    mnq = parquet_summary(MNQ)

    data_files = [path for path in (ROOT / "data").rglob("*") if path.is_file()]
    named_candidates = [
        path for path in data_files
        if any(term in path.name.lower() for term in ("gex", "gamma", "dealer_gamma"))
    ]
    schema_hits: list[dict[str, str]] = []
    for path in data_files:
        try:
            if path.suffix.lower() == ".parquet":
                columns = pq.ParquetFile(path).schema.names
            elif path.suffix.lower() == ".csv":
                columns = list(pd.read_csv(path, nrows=0).columns)
            else:
                continue
        except Exception:
            continue
        hits = sorted({str(column).lower() for column in columns} & FORBIDDEN_PROXY_COLUMNS)
        if hits:
            schema_hits.append({"path": str(path.relative_to(ROOT)), "columns": "|".join(hits)})

    gate_passed = bool(named_candidates or schema_hits)
    if gate_passed:
        raise RuntimeError(
            "A possible GEX source appeared after this audit was frozen. "
            "Inspect its provenance and availability timestamps before analysis."
        )

    audit = pd.DataFrame(
        [
            {
                "source": "NQ 1-minute continuous futures",
                "path": nq["path"],
                "rows": nq["rows"],
                "coverage_start": nq["start"],
                "coverage_end": nq["end"],
                "gex_value": False,
                "historical_availability_timestamp": False,
                "usable_for_gamma_regime": False,
                "reason": "OHLCV futures bars only; price-derived GEX proxy is prohibited.",
                "sha256": nq["sha256"],
            },
            {
                "source": "MNQ 1-minute continuous futures",
                "path": mnq["path"],
                "rows": mnq["rows"],
                "coverage_start": mnq["start"],
                "coverage_end": mnq["end"],
                "gex_value": False,
                "historical_availability_timestamp": False,
                "usable_for_gamma_regime": False,
                "reason": "OHLCV futures bars only; price-derived GEX proxy is prohibited.",
                "sha256": mnq["sha256"],
            },
            {
                "source": "VIX/VIX3M daily closes",
                "path": str(VIX.relative_to(ROOT)),
                "rows": len(pd.read_csv(VIX)),
                "coverage_start": "",
                "coverage_end": "",
                "gex_value": False,
                "historical_availability_timestamp": False,
                "usable_for_gamma_regime": False,
                "reason": "Options-implied volatility state, not dealer gamma exposure.",
                "sha256": sha256(VIX),
            },
            {
                "source": "Existing lagged VIX state dataset",
                "path": str(CAUSAL_VIX.relative_to(ROOT)),
                "rows": len(pd.read_csv(CAUSAL_VIX)),
                "coverage_start": "",
                "coverage_end": "",
                "gex_value": False,
                "historical_availability_timestamp": True,
                "usable_for_gamma_regime": False,
                "reason": "Causally lagged, but contains no GEX and cannot replace it.",
                "sha256": sha256(CAUSAL_VIX),
            },
        ]
    )
    audit.to_csv(OUT / "gex_data_audit.csv", index=False)

    not_run_reason = (
        "NOT_RUN_CAUSALITY_GATE_FAILED: no point-in-time GEX value with source "
        "and historical availability timestamp exists in the repository."
    )
    requested = {
        "gamma_regime_summary.csv": ["status", "reason"],
        "autocorrelation_by_regime.csv": ["status", "reason"],
        "shock_response_by_regime.csv": ["status", "reason"],
        "session_response_by_regime.csv": ["status", "reason"],
        "vwap_reversion_by_regime.csv": ["status", "reason"],
        "temporal_stability.csv": ["status", "reason"],
        "frozen_sleeves_by_gamma.csv": ["status", "reason"],
    }
    for filename, columns in requested.items():
        pd.DataFrame([["NOT_RUN", not_run_reason]], columns=columns).to_csv(OUT / filename, index=False)

    report = f"""# Dealer gamma NQ/MNQ — Phase 1 data gate

## Status: STOPPED — CAUSALITY GATE FAILED

No alpha verdict (`REJECTED`, `WEAK_MECHANISM`, or `PROMISING_FOR_VALIDATION`) is issued because the hypothesis was not statistically tested. Calling it `REJECTED` would confuse missing causal data with evidence against the mechanism.

## Repository audit

- No file under `data/` has GEX, gamma, or dealer-gamma in its filename.
- No CSV/Parquet schema under `data/` contains a `gex`, `gamma`, `dealer_gamma`, or `gamma_exposure` field.
- NQ 1-minute is available from {nq['start']} through {nq['end']} ({nq['rows']:,} bars).
- MNQ 1-minute is available from {mnq['start']} through {mnq['end']} ({mnq['rows']:,} bars).
- Those futures files contain OHLCV bars and identifiers only. The mandate explicitly prohibits reconstructing a gamma proxy from NQ prices.
- The repository contains VIX/VIX3M closes and a causally lagged volatility-state dataset. These are volatility-state variables, not dealer gamma exposure, and cannot answer the requested mechanism.

The detailed source inventory and hashes are in `gex_data_audit.csv`.

## Why the requested tests cannot be run

A causal session label needs, at minimum, a GEX value, the market/reference date it describes, its source and methodology, and the actual timestamp at which that historical observation became usable. A date-only series or a file created later from revised option-chain data would not establish that the label was known before the session.

Without that information, even the sign split can leak future information. Rolling terciles do not repair the problem: they make the threshold causal only after the underlying observations themselves are point-in-time causal.

Consequently, autocorrelation, shock response, open-to-later-session, VWAP reversion, volatility controls, temporal stability, day/hour controls, frozen-sleeve conditioning, prototypes, costs, and orthogonality were not run. Their requested CSVs contain an explicit `NOT_RUN` record rather than fabricated empty results. `prototype_results.csv` is correctly absent because prototypes are conditional on a successful mechanism diagnostic.

## Minimum acceptable input for a restart

Provide a point-in-time table with at least:

- `reference_date` or `reference_session`;
- numeric `gex_value` and units/notional convention;
- source/vendor and methodology/version;
- timezone-aware `available_at` timestamp reflecting historical publication, not download time;
- underlying definition (NDX, NQ options, or an explicitly documented aggregate);
- whether open interest, positions, greeks and spot inputs are preliminary or revised;
- a stable identifier for corrections/restatements.

The restart gate should verify `available_at <= session decision time` row by row before defining either raw-sign or expanding-tercile regimes. No frozen sleeve, Gotobi rule, Topstep simulation, proxy, or trading prototype was modified or run.
"""
    (OUT / "PHASE1_REPORT.md").write_text(report, encoding="utf-8")

    metadata = {
        "study": "dealer_gamma_nq_mnq_phase1",
        "as_of": "2026-10-03",
        "status": "STOPPED_CAUSALITY_GATE_FAILED",
        "alpha_verdict": None,
        "gex_named_files_found": len(named_candidates),
        "gex_schema_hits_found": len(schema_hits),
        "diagnostics_run": False,
        "prototypes_run": False,
        "topstep_run": False,
        "frozen_sleeves_modified": False,
        "gotobi_modified": False,
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
