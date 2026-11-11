"""Persist the daily net PnL of every refinement grid cell.

Re-runs each cell of a refinement campaign with the exact runner functions and
stores one column per cell.  Every recomputed IS/OOS Sharpe is checked against
the campaign's grid_metrics.csv, so the series are provably those already mapped.
Nothing outside the campaign directory is written.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from run_signal_sheet_parameter_refinement import (
    CAMPAIGNS,
    OUTPUT_ROOT,
    apply_parameters,
    prepare_campaign,
    run_candidate,
    scoped_metrics,
)


def export(key: str) -> None:
    campaign = CAMPAIGNS[key]
    directory = OUTPUT_ROOT / key
    output = directory / "cell_daily_pnl.parquet"
    grid = pd.read_csv(directory / "grid_metrics.csv")
    params = list(campaign["grids"])
    cells = grid.drop_duplicates("cell_id")[["cell_id", *params]].sort_values("cell_id")
    if output.is_file() and set(pd.read_parquet(output).columns) == set(cells["cell_id"]):
        print(f"{key}: déjà exporté ({len(cells)} cellules)", flush=True)
        return

    prepared, base_spec, _ = prepare_campaign(key, campaign)
    series: dict[str, pd.Series] = {}
    for position, row in enumerate(cells.itertuples(index=False), 1):
        values = {name: getattr(row, name) for name in params}
        candidate = apply_parameters(base_spec, campaign, values)
        run = run_candidate(key, candidate, prepared, campaign)
        for metrics in scoped_metrics(run, campaign):
            expected = grid[(grid["cell_id"] == row.cell_id) & (grid["scope"] == metrics["scope"])]["sharpe_annualized"].iloc[0]
            same = (math.isnan(expected) and math.isnan(metrics["sharpe_annualized"])) or np.isclose(expected, metrics["sharpe_annualized"], atol=1e-9)
            if not same:
                raise RuntimeError(f"{key} {row.cell_id} {metrics['scope']}: Sharpe {metrics['sharpe_annualized']} != {expected}")
        daily = run.daily[["session_date", "daily_pnl_usd"]].copy()
        daily["session_date"] = pd.to_datetime(daily["session_date"])
        daily = daily[daily["session_date"].between(pd.Timestamp(campaign["start"]), pd.Timestamp(campaign["end"]))]
        series[row.cell_id] = daily.set_index("session_date")["daily_pnl_usd"].astype(float)
        print(f"{key} {position}/{len(cells)} {row.cell_id} vérifié", flush=True)

    frame = pd.DataFrame(series).sort_index()
    frame.index.name = "session_date"
    frame.to_parquet(output)
    print(f"{key}: {output} ({frame.shape[0]} séances × {frame.shape[1]} cellules)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleeve", required=True, choices=sorted(CAMPAIGNS))
    args = parser.parse_args()
    export(args.sleeve)


if __name__ == "__main__":
    main()
