"""Robust cluster selection for signal sheets S03-S09, validated inside IS.

Candidate selection rules only ever see IS data.  The rules themselves are
validated by combinatorially symmetric cross-validation (CSCV, Bailey & López
de Prado): IS is cut into contiguous blocks, half of the blocks select a
cluster and the other half measure it, over every half/half combination.  One
rule, the one that generalises best on average across all sleeves, is then
applied to the full IS.  OOS is read once, as a final verdict, and never
selects or re-ranks anything.

A cluster is measured through its equal-weight proxy: the mean daily PnL of its
cells.  It is a selection device, not the executable vote ensemble, which must
be re-run before any adoption.  Frozen specifications are never touched.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from select_signal_sheet_clusters import (
    CURRENT_LOCK,
    MIN_CELLS,
    MIN_VALUES_PER_AXIS,
    RESULT_ROOT,
    SLEEVES,
    candidate_boxes,
    duplicate_values,
    in_box,
)


OUTPUT_DIR = RESULT_ROOT / "cluster_selection"
BLOCKS = 10
PLATEAU_TOLERANCE = 0.05
ANNUALISATION = math.sqrt(252.0)

# Performance-agnostic alternative: one grid step around the already declared
# lock, or the frozen region itself for the MBT vote ensembles (S07 without its
# duplicate exit length 15, identical to 12).  No new degree of freedom is fitted.
LOCK_NEIGHBOURHOOD = {
    "s03": {"nr_lookback": [8, 10, 12], "entry_buffer_ticks": [0, 1, 2], "target_setup_range_multiple": [1.5, 2.0, 2.5]},
    "s04": {"atr_period": [8, 10, 12], "atr_multiplier": [1.5, 1.75, 2.0]},
    "s05": {"daily_atr_period": [20, 30, 40], "entry_atr_multiple": [0.9, 1.1, 1.3]},
    "s06": {"minimum_gap_atr_multiple": [0.4, 0.5, 0.6], "volume_threshold_multiple": [0.5, 0.6, 0.7]},
    "s07": {"entry_channel_length": [20, 24], "exit_channel_length": [7, 10, 12]},
    "s08": {"fast_pair_scale": [0.5, 0.6, 0.75], "slow_tail_scale": [0.75, 1.0, 1.25]},
    "s09": {"daily_atr_period": [10, 14, 20, 30], "entry_atr_multiple": [0.3, 0.4, 0.6]},
}


@dataclass
class Sleeve:
    key: str
    params: list[str]
    cells: pd.DataFrame
    boxes: list[dict[str, list[float]]]
    is_pnl: np.ndarray
    oos_pnl: np.ndarray
    weights: np.ndarray
    membership: np.ndarray
    dilated: np.ndarray
    blocks: list[np.ndarray]
    raw_daily: pd.DataFrame
    is_index: pd.Index
    oos_index: pd.Index
    grid: pd.DataFrame


def sharpe(matrix: np.ndarray) -> np.ndarray:
    std = matrix.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = ANNUALISATION * matrix.mean(axis=0) / std
    return np.where(std > 0, values, -np.inf)


def dilate(box: dict[str, list[float]], cells: pd.DataFrame) -> dict[str, list[float]]:
    widened = {}
    for axis, values in box.items():
        grid = sorted(cells[axis].unique())
        low = max(grid.index(values[0]) - 1, 0)
        high = min(grid.index(values[-1]) + 1, len(grid) - 1)
        widened[axis] = grid[low:high + 1]
    return widened


def load(key: str) -> Sleeve:
    directory = RESULT_ROOT / key
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    params = list(manifest["grids"])
    grid = pd.read_csv(directory / "grid_metrics.csv")
    duplicates = duplicate_values(grid, params)
    cells = grid.drop_duplicates("cell_id")[["cell_id", *params]]
    for axis, values in duplicates.items():
        cells = cells[~cells[axis].isin(values)]
    cells = cells.sort_values(params).reset_index(drop=True)

    raw_daily = pd.read_parquet(directory / "cell_daily_pnl.parquet").fillna(0.0).sort_index()
    periods = manifest["periods"]
    is_mask = (raw_daily.index >= pd.Timestamp(periods["start"])) & (raw_daily.index <= pd.Timestamp(periods["is_end"]))
    oos_mask = (raw_daily.index >= pd.Timestamp(periods["oos_start"])) & (raw_daily.index <= pd.Timestamp(periods["end"]))
    daily = raw_daily[cells["cell_id"]]

    boxes = [box for box in candidate_boxes(cells, params) if in_box(cells, box).sum() >= MIN_CELLS]
    membership = np.array([in_box(cells, box).to_numpy() for box in boxes])
    dilated = np.array([in_box(cells, dilate(box, cells)).to_numpy() for box in boxes])
    weights = (membership / membership.sum(axis=1, keepdims=True)).T
    is_pnl = daily[is_mask].to_numpy(float)
    return Sleeve(
        key=key, params=params, cells=cells, boxes=boxes,
        is_pnl=is_pnl, oos_pnl=daily[oos_mask].to_numpy(float),
        weights=weights, membership=membership, dilated=dilated,
        blocks=np.array_split(np.arange(len(is_pnl)), BLOCKS),
        raw_daily=raw_daily, is_index=raw_daily.index[is_mask], oos_index=raw_daily.index[oos_mask], grid=grid,
    )


@dataclass
class Evidence:
    cell_sharpe: np.ndarray
    proxy_sharpe: np.ndarray
    worst_block_sharpe: np.ndarray


def evidence(sleeve: Sleeve, block_ids: tuple[int, ...]) -> Evidence:
    rows = np.concatenate([sleeve.blocks[i] for i in block_ids])
    pnl = sleeve.is_pnl[rows]
    per_block = np.vstack([sharpe(sleeve.is_pnl[sleeve.blocks[i]] @ sleeve.weights) for i in block_ids])
    return Evidence(sharpe(pnl), sharpe(pnl @ sleeve.weights), per_block.min(axis=0))


def masked(values: np.ndarray, mask: np.ndarray, reducer: Callable) -> np.ndarray:
    return reducer(np.where(mask, values[None, :], np.nan), axis=1)


def argbest(*keys: np.ndarray) -> int:
    """Index of the lexicographic maximum, first key most important."""
    cleaned = [np.nan_to_num(key, nan=-np.inf) for key in keys]
    return int(np.lexsort(tuple(reversed(cleaned)))[-1])


def rule_max_median(s: Sleeve, e: Evidence) -> int:
    median = masked(e.cell_sharpe, s.membership, np.nanmedian)
    return argbest(median, s.membership.sum(axis=1))


def rule_plateau(s: Sleeve, e: Evidence) -> int:
    median = masked(e.cell_sharpe, s.membership, np.nanmedian)
    eligible = median >= np.nanmax(median) - PLATEAU_TOLERANCE
    return argbest(eligible.astype(float), np.where(eligible, s.membership.sum(axis=1), 0), median)


def rule_maximin(s: Sleeve, e: Evidence) -> int:
    median = masked(e.cell_sharpe, s.membership, np.nanmedian)
    return argbest(masked(e.cell_sharpe, s.membership, np.nanmin), s.membership.sum(axis=1), median)


def rule_maximin_margin(s: Sleeve, e: Evidence) -> int:
    median = masked(e.cell_sharpe, s.membership, np.nanmedian)
    return argbest(masked(e.cell_sharpe, s.dilated, np.nanmin), s.membership.sum(axis=1), median)


def rule_proxy(s: Sleeve, e: Evidence) -> int:
    return argbest(e.proxy_sharpe, s.membership.sum(axis=1))


def rule_worst_block(s: Sleeve, e: Evidence) -> int:
    return argbest(e.worst_block_sharpe, s.membership.sum(axis=1))


RULES: dict[str, tuple[str, Callable[[Sleeve, Evidence], int]]] = {
    "max_median": ("Sharpe médian des cellules maximal", rule_max_median),
    "plateau": (f"plus grand cluster à ≤{PLATEAU_TOLERANCE} du meilleur Sharpe médian", rule_plateau),
    "maximin": ("Sharpe de la cellule la plus faible maximal", rule_maximin),
    "maximin_margin": ("Sharpe minimal maximal sur le cluster élargi d'un cran (voisins inclus)", rule_maximin_margin),
    "proxy": ("Sharpe du cluster agrégé maximal", rule_proxy),
    "worst_block": ("Sharpe du cluster agrégé sur sa pire sous-période maximal", rule_worst_block),
}


def cscv(sleeve: Sleeve) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, list[float]]] = {name: {"omega": [], "test": [], "decay": []} for name in RULES}
    candidate_count = len(sleeve.boxes)
    for train in itertools.combinations(range(BLOCKS), BLOCKS // 2):
        test = tuple(i for i in range(BLOCKS) if i not in train)
        train_evidence = evidence(sleeve, train)
        test_rows = np.concatenate([sleeve.blocks[i] for i in test])
        test_sharpe = sharpe(sleeve.is_pnl[test_rows] @ sleeve.weights)
        ranks = pd.Series(test_sharpe).rank(method="average").to_numpy()
        for name, (_, rule) in RULES.items():
            chosen = rule(sleeve, train_evidence)
            stats[name]["omega"].append(ranks[chosen] / (candidate_count + 1))
            stats[name]["test"].append(test_sharpe[chosen])
            stats[name]["decay"].append(train_evidence.proxy_sharpe[chosen] - test_sharpe[chosen])
    return {
        name: {
            "mean_test_rank": float(np.mean(values["omega"])),
            "pbo": float(np.mean(np.array(values["omega"]) < 0.5)),
            "mean_test_sharpe": float(np.mean(values["test"])),
            "mean_sharpe_decay": float(np.mean(values["decay"])),
        }
        for name, values in stats.items()
    }


def fixed_cluster_rank(sleeve: Sleeve, mask: np.ndarray) -> float:
    """Mean out-of-fold rank of a cluster fixed in advance, against every candidate."""
    weights = mask / mask.sum()
    ranks = []
    for train in itertools.combinations(range(BLOCKS), BLOCKS // 2):
        rows = np.concatenate([sleeve.blocks[i] for i in range(BLOCKS) if i not in train])
        candidates = sharpe(sleeve.is_pnl[rows] @ sleeve.weights)
        mine = sharpe((sleeve.is_pnl[rows] @ weights)[:, None])[0]
        ranks.append(float((candidates < mine).mean() + 0.5 * (candidates == mine).mean()))
    return float(np.mean(ranks))


def edges(sleeve: Sleeve, box: dict[str, list[float]]) -> list[str]:
    return [axis for axis, values in box.items()
            if min(values) == min(sleeve.cells[axis]) or max(values) == max(sleeve.cells[axis])]


def verdict(sleeve: Sleeve, box: dict[str, list[float]], mask: np.ndarray) -> dict[str, Any]:
    box = {axis: [float(v) for v in values] for axis, values in box.items()}
    cell_ids = sleeve.cells.loc[mask, "cell_id"].tolist()
    weights = mask / mask.sum()
    is_proxy = sleeve.is_pnl @ weights
    oos_proxy = sleeve.oos_pnl @ weights
    is_cells, oos_cells = sharpe(sleeve.is_pnl[:, mask]), sharpe(sleeve.oos_pnl[:, mask])
    holds = bool(sharpe(oos_proxy[:, None])[0] > 0 and (oos_cells > 0).all())
    partial = bool(sharpe(oos_proxy[:, None])[0] > 0)
    selected = sleeve.grid[sleeve.grid["cell_id"].isin(cell_ids)]
    return {
        "box": box, "cell_count": int(mask.sum()), "cell_ids": cell_ids, "grid_edge_axes": edges(sleeve, box),
        "is": {"proxy_sharpe": float(sharpe(is_proxy[:, None])[0]), "cell_sharpe_min": float(is_cells.min()),
               "cell_sharpe_median": float(np.median(is_cells))},
        "oos": {"proxy_sharpe": float(sharpe(oos_proxy[:, None])[0]), "cell_sharpe_min": float(oos_cells.min()),
                "cell_sharpe_median": float(np.median(oos_cells)), "positive_cells": int((oos_cells > 0).sum())},
        "worst_cell_day_usd": {scope: float(selected[selected["scope"].eq(scope)]["worst_day_usd"].min()) for scope in ("is", "oos")},
        "worst_cell_max_drawdown_usd": {scope: float(selected[selected["scope"].eq(scope)]["max_drawdown_usd"].min()) for scope in ("is", "oos")},
        "verdict": "tient" if holds else ("tient partiellement" if partial else "ne tient pas"),
    }


def lock_reference(sleeve: Sleeve) -> dict[str, Any]:
    """The frozen point or region as actually defined, duplicate cells included."""
    lock = CURRENT_LOCK[sleeve.key]
    cells = sleeve.grid.drop_duplicates("cell_id")
    cell_ids = cells.loc[in_box(cells, lock), "cell_id"].tolist()
    daily = sleeve.raw_daily[cell_ids]
    is_proxy = daily.loc[sleeve.is_index].mean(axis=1).to_numpy()[:, None]
    oos_proxy = daily.loc[sleeve.oos_index].mean(axis=1).to_numpy()[:, None]
    return {"box": lock, "cell_count": len(cell_ids),
            "is_proxy_sharpe": float(sharpe(is_proxy)[0]), "oos_proxy_sharpe": float(sharpe(oos_proxy)[0])}


def main() -> None:
    sleeves = [load(key) for key in SLEEVES]
    validation = {sleeve.key: cscv(sleeve) for sleeve in sleeves}
    summary = {
        name: {metric: float(np.mean([validation[s.key][name][metric] for s in sleeves]))
               for metric in ("mean_test_rank", "pbo", "mean_test_sharpe", "mean_sharpe_decay")}
        for name in RULES
    }
    best_rule = max(summary, key=lambda name: (summary[name]["mean_test_rank"], -summary[name]["pbo"]))
    optimisation_generalises = summary[best_rule]["mean_test_rank"] > 0.5
    full_is = tuple(range(BLOCKS))
    results = []
    for sleeve in sleeves:
        index = RULES[best_rule][1](sleeve, evidence(sleeve, full_is))
        neighbourhood_mask = in_box(sleeve.cells, LOCK_NEIGHBOURHOOD[sleeve.key]).to_numpy()
        results.append({
            "sleeve": sleeve.key, "candidates": len(sleeve.boxes),
            "lock_neighbourhood": {**verdict(sleeve, LOCK_NEIGHBOURHOOD[sleeve.key], neighbourhood_mask),
                                   "cscv_mean_test_rank": fixed_cluster_rank(sleeve, neighbourhood_mask)},
            "best_optimising_rule": verdict(sleeve, sleeve.boxes[index], sleeve.membership[index]),
            "current_lock": lock_reference(sleeve),
        })

    payload = {
        "schema_version": "1.0",
        "purpose": "robust_cluster_selection_proposal_not_applied",
        "method": {
            "candidates": f"axis-aligned boxes of consecutive grid values, >= {MIN_VALUES_PER_AXIS} values per axis, >= {MIN_CELLS} cells, duplicate parameter values collapsed",
            "cluster_measure": "equal-weight mean of cell daily net PnL (selection proxy, not the executable vote)",
            "validation": f"CSCV inside IS: {BLOCKS} contiguous blocks, every {BLOCKS // 2}/{BLOCKS // 2} split ({math.comb(BLOCKS, BLOCKS // 2)} splits)",
            "rule_choice": "single rule for every sleeve: highest mean out-of-fold rank, then lowest PBO",
            "oos_use": "read once after selection as a verdict; never selects or re-ranks",
            "independence_caveat": "OOS windows were consulted when the current locks were frozen; this is reused validation",
        },
        "rules": {name: description for name, (description, _) in RULES.items()},
        "rule_validation_by_sleeve": validation,
        "rule_validation_mean": summary,
        "best_optimising_rule": best_rule,
        "optimisation_generalises": optimisation_generalises,
        "recommendation": best_rule if optimisation_generalises else "lock_neighbourhood",
        "recommendation_caveat": (
            "lock_neighbourhood ranks are optimistic: the lock centres were chosen with the full IS visible; "
            "the failure of every optimising rule is the unbiased finding"
        ),
        "results": results,
        "frozen_files_modified": False,
    }
    output = OUTPUT_DIR / "cluster_selection_cscv.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
