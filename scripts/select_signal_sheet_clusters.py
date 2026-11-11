"""Select one contiguous parameter cluster per signal sheet S03-S09.

Selection uses the IS period only: among every axis-aligned box of consecutive
grid values, keep the one with the highest median IS Sharpe.  The OOS period is
a pass/fail filter (every cell must keep a positive Sharpe) and never ranks
candidates.  The OOS windows were already consulted when the current locks were
frozen, so passing this filter is a reused validation, not an independent test.

This script only writes a proposal.  It never edits frozen specifications or
portfolio configurations.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "research" / "signal_sheet_parameter_refinement"
OUTPUT_DIR = RESULT_ROOT / "cluster_selection"
SLEEVES = ("s03", "s04", "s05", "s06", "s07", "s08", "s09")

MIN_VALUES_PER_AXIS = 2
MIN_CELLS = 6
# IS Sharpe gaps below this are noise for ~4.5 years of daily data (standard error ~0.5).
PLATEAU_TOLERANCE = 0.05

# Currently frozen point or region, expressed in grid coordinates.
CURRENT_LOCK = {
    "s03": {"nr_lookback": [10], "entry_buffer_ticks": [1], "target_setup_range_multiple": [2.0]},
    "s04": {"atr_period": [10], "atr_multiplier": [1.75]},
    "s05": {"daily_atr_period": [30], "entry_atr_multiple": [1.1]},
    "s06": {"minimum_gap_atr_multiple": [0.5], "volume_threshold_multiple": [0.6]},
    "s07": {"entry_channel_length": [20, 24], "exit_channel_length": [7, 10, 12, 15]},
    "s08": {"fast_pair_scale": [0.5, 0.6, 0.75], "slow_tail_scale": [0.75, 1.0, 1.25]},
    "s09": {"daily_atr_period": [10, 14, 20, 30], "entry_atr_multiple": [0.3, 0.4, 0.6]},
}


def load(sleeve: str) -> tuple[pd.DataFrame, list[str], dict[str, list[float]]]:
    directory = RESULT_ROOT / sleeve
    frame = pd.read_csv(directory / "grid_metrics.csv")
    params = list(json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["grids"])
    return frame, params, duplicate_values(frame, params)


def duplicate_values(frame: pd.DataFrame, params: list[str]) -> dict[str, list[float]]:
    """Return axis values whose every cell repeats the previous value exactly."""
    columns = ["net_pnl_usd", "sharpe_annualized", "trade_count"]
    duplicates: dict[str, list[float]] = {}
    for axis in params:
        others = [name for name in params if name != axis]
        values = sorted(frame[axis].unique())
        for left, right in zip(values, values[1:]):
            a = frame[frame[axis].eq(left)].sort_values([*others, "scope"])[columns].to_numpy(float)
            b = frame[frame[axis].eq(right)].sort_values([*others, "scope"])[columns].to_numpy(float)
            if np.allclose(a, b):
                duplicates.setdefault(axis, []).append(float(right))
    return duplicates


def wide_sharpe(frame: pd.DataFrame, params: list[str], duplicates: dict[str, list[float]]) -> pd.DataFrame:
    wide = frame.pivot(index=["cell_id", *params], columns="scope", values="sharpe_annualized").reset_index()
    for axis, values in duplicates.items():
        wide = wide[~wide[axis].isin(values)]
    return wide


def candidate_boxes(wide: pd.DataFrame, params: list[str]) -> list[dict[str, list[float]]]:
    ranges_per_axis = []
    for axis in params:
        values = sorted(wide[axis].unique())
        ranges_per_axis.append([
            values[start:stop]
            for start in range(len(values))
            for stop in range(start + MIN_VALUES_PER_AXIS, len(values) + 1)
        ])
    return [dict(zip(params, combo)) for combo in itertools.product(*ranges_per_axis)]


def in_box(frame: pd.DataFrame, box: dict[str, list[float]]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for axis, values in box.items():
        mask &= frame[axis].round(9).isin([round(float(v), 9) for v in values])
    return mask


def score(wide: pd.DataFrame, box: dict[str, list[float]]) -> dict[str, Any] | None:
    cells = wide[in_box(wide, box)]
    if len(cells) < MIN_CELLS:
        return None
    return {
        "box": {axis: [float(v) for v in values] for axis, values in box.items()},
        "cell_count": int(len(cells)),
        "is_median_sharpe": float(cells["is"].median()),
        "is_min_sharpe": float(cells["is"].min()),
        "oos_median_sharpe": float(cells["oos"].median()),
        "oos_min_sharpe": float(cells["oos"].min()),
        "passes_oos_filter": bool((cells["oos"] > 0).all()),
    }


def cluster_metrics(frame: pd.DataFrame, box: dict[str, list[float]], duplicates: dict[str, list[float]]) -> dict[str, Any]:
    selected = frame[in_box(frame, box)]
    for axis, values in duplicates.items():
        selected = selected[~selected[axis].isin(values)]
    result: dict[str, Any] = {}
    for scope in ("is", "oos"):
        scoped = selected[selected["scope"].eq(scope)]
        result[scope] = {
            "median_sharpe": float(scoped["sharpe_annualized"].median()),
            "min_sharpe": float(scoped["sharpe_annualized"].min()),
            "max_sharpe": float(scoped["sharpe_annualized"].max()),
            "median_profit_factor": float(scoped["profit_factor"].median()),
            "median_trade_count": float(scoped["trade_count"].median()),
            "worst_cell_max_drawdown_usd": float(scoped["max_drawdown_usd"].min()),
            "worst_day_usd": float(scoped["worst_day_usd"].min()),
        }
    return result


def touches_grid_edge(wide: pd.DataFrame, box: dict[str, list[float]]) -> list[str]:
    edges = []
    for axis, values in box.items():
        grid = sorted(wide[axis].unique())
        if values[0] == grid[0] or values[-1] == grid[-1]:
            edges.append(axis)
    return edges


def select(sleeve: str) -> dict[str, Any]:
    frame, params, duplicates = load(sleeve)
    wide = wide_sharpe(frame, params, duplicates)
    scored = [item for box in candidate_boxes(wide, params) if (item := score(wide, box))]
    passing = [item for item in scored if item["passes_oos_filter"]]
    ranked = sorted(passing, key=lambda item: (-item["is_median_sharpe"], -item["cell_count"], -item["is_min_sharpe"]))
    if not ranked:
        raise RuntimeError(f"{sleeve}: no candidate cluster keeps every OOS cell positive")
    best = ranked[0]
    near_best = [item for item in passing if item["is_median_sharpe"] >= best["is_median_sharpe"] - PLATEAU_TOLERANCE]
    plateau = sorted(near_best, key=lambda item: (-item["cell_count"], -item["is_median_sharpe"]))[0]
    lock = CURRENT_LOCK[sleeve]
    lock_cells = wide[in_box(wide, lock)]
    return {
        "sleeve": sleeve,
        "parameters": params,
        "collapsed_duplicate_values": duplicates,
        "candidates_scored": len(scored),
        "candidates_passing_oos_filter": len(passing),
        "selected": {**best, "grid_edge_axes": touches_grid_edge(wide, best["box"]),
                     "metrics": cluster_metrics(frame, best["box"], duplicates)},
        "runners_up": ranked[1:5],
        "plateau_variant": {**plateau, "grid_edge_axes": touches_grid_edge(wide, plateau["box"]),
                            "candidates_within_tolerance": len(near_best)},
        "current_lock": {
            "box": lock, "cell_count": int(len(lock_cells)),
            "is_median_sharpe": float(lock_cells["is"].median()),
            "oos_median_sharpe": float(lock_cells["oos"].median()),
            "oos_min_sharpe": float(lock_cells["oos"].min()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleeve", choices=SLEEVES, action="append")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [select(sleeve) for sleeve in (args.sleeve or SLEEVES)]
    payload = {
        "schema_version": "1.0",
        "purpose": "cluster_selection_proposal_not_applied",
        "rule": {
            "candidates": f"axis-aligned boxes of consecutive grid values, >= {MIN_VALUES_PER_AXIS} values per axis, >= {MIN_CELLS} cells",
            "objective": "maximise median IS Sharpe; ties broken by cell count, then minimum IS Sharpe",
            "plateau_variant": f"largest passing cluster whose median IS Sharpe is within {PLATEAU_TOLERANCE} of the best",
            "oos_filter": "every cell of the cluster keeps a positive OOS Sharpe; OOS never ranks candidates",
            "independence_caveat": "OOS windows were consulted when the current locks were frozen; this is reused validation",
            "aggregation_caveat": "cell statistics only; the executable vote ensemble must be re-run before any adoption",
        },
        "frozen_files_modified": False,
        "results": results,
    }
    output = OUTPUT_DIR / "cluster_selection.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
