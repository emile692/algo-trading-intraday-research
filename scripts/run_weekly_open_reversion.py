"""Run the pre-specified MNQ/NQ Tuesday-to-weekly-open study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.weekly_open_reversion import (
    StudyConfig,
    build_weekly_observations,
    combine_studies,
    summarize_observations,
)


DEFAULT_INPUTS = {
    "MNQ": ROOT / "data" / "downloaded" / "MNQ_c_0_1m_20260321_094501.parquet",
    "NQ": ROOT / "data" / "downloaded" / "NQ_c_0_1m_20260321_095036.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnq", type=Path, default=DEFAULT_INPUTS["MNQ"])
    parser.add_argument("--nq", type=Path, default=DEFAULT_INPUTS["NQ"])
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "research" / "weekly_open_reversion_20261001"
    )
    parser.add_argument("--rth-only", action="store_true", help="Exclude the Tue-Wed overnight session")
    parser.add_argument("--allow-contract-change", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StudyConfig(
        include_overnight=not args.rth_only,
        require_same_contract=not args.allow_contract_change,
    )
    inputs = {"MNQ": args.mnq, "NQ": args.nq}
    studies = []
    for symbol, path in inputs.items():
        bars = pd.read_parquet(path)
        studies.append(build_weekly_observations(bars, symbol=symbol, config=config))
    observations, audit = combine_studies(studies)
    summary = summarize_observations(observations)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(args.output_dir / "weekly_observations.csv", index=False)
    summary.to_csv(args.output_dir / "summary_by_symbol_direction_bucket.csv", index=False)
    audit.to_csv(args.output_dir / "exclusion_audit.csv", index=False)
    metadata = {
        "hypothesis": "Tuesday RTH open reverts toward Monday 09:30 ET Weekly Open",
        "entry": "Tuesday 09:30 ET bar open",
        "target": "Monday 09:30 ET bar open",
        "D": "absolute distance between entry and target",
        "horizon": "Wednesday 16:00 ET",
        "include_overnight": config.include_overnight,
        "same_bar_ordering": "conservative: target is not before an adverse barrier hit in the same minute",
        "checkpoint_pnl": "directional points, capped at +1D after target; no adverse stop",
        "inputs": {symbol: str(path.resolve()) for symbol, path in inputs.items()},
        "included_observations": int(len(observations)),
        "excluded_weeks": int((audit["status"] == "excluded").sum()),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(observations)} observations to {args.output_dir}")


if __name__ == "__main__":
    main()
