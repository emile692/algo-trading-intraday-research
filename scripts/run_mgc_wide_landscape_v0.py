"""Run the locked 16-mechanism MGC landscape into a separate v.0 output tree."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from research.mgc_wide_parameter_landscape import run_mgc_wide_landscape as campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be >= 1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / campaign.GRID_PATH.name
    shutil.copy2(campaign.GRID_PATH, grid_path)

    campaign.OUTPUT_DIR = output_dir
    campaign.HEATMAP_DIR = output_dir / "heatmaps"
    campaign.MPL_CACHE = output_dir / ".matplotlib"
    campaign.GRID_PATH = grid_path
    campaign.GRID_HASH_PATH = output_dir / "mgc_wide_parameter_grids.sha256"
    campaign.RESULT_PATH = output_dir / "mgc_wide_landscape_results.csv"
    campaign.REPORT_PATH = output_dir / "MGC_WIDE_LANDSCAPE_REPORT.md"
    campaign.PARTIAL_PATH = output_dir / "mgc_wide_landscape_results.partial.csv"

    campaign.execute(args.data.resolve(), args.workers)


if __name__ == "__main__":
    main()
