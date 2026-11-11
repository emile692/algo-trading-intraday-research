"""Run the causal Phase-1 60/40 rebalancing study."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analytics.sixty_forty_rebalancing import DEFAULT_OUTPUT, REPO_ROOT, run_study


DEFAULT_PATHS = {
    "MES": REPO_ROOT / "data" / "processed" / "parquet" / "MES_c_0_1m_20260322_135702.parquet",
    "MNQ": REPO_ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet",
    "NQ": REPO_ROOT / "data" / "processed" / "parquet" / "NQ_c_0_1m_20260321_095036.parquet",
    "ZN": REPO_ROOT / "data" / "processed" / "parquet" / "ZN_v_0_1m_20130517_20260920.parquet",
    "ZF": REPO_ROOT / "data" / "processed" / "parquet" / "ZF_v_0_1m_20130517_20260920.parquet",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    missing = [str(path) for path in DEFAULT_PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    artifacts = run_study(DEFAULT_PATHS, args.output.resolve())
    central = artifacts["backtest_summary"].iloc[0]
    print(f"Wrote Phase-1 study to {args.output.resolve()}")
    print(f"Central MES: {int(central['trades'])} trades, net ${central['net_pnl_usd']:.2f}, t={central['simple_t_net']:.2f}")


if __name__ == "__main__":
    main()

