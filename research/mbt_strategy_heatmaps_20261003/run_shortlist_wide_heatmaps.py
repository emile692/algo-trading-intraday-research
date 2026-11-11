"""Expand the five shortlisted MBT robustness landscapes.

Cells already present in mbt_heatmap_metrics.csv are reused verbatim.  Only new
coordinates are backtested, with the same MBT economics, session preparation,
and chronological IS/OOS split as the original campaign.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from research.baseline_backtests.baseline_engine import load_frozen_specs, run_baseline_backtest
from research.btc_baseline_backtests.run_btc_baselines import adapt_spec_to_btc, prepare_btc_data
from research.mbt_strategy_heatmaps_20261003.run_mbt_heatmaps import _ribbon, _split_metrics, mbt_config


ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSV = ROOT / "mbt_heatmap_metrics.csv"
SOURCE_METADATA = ROOT / "run_metadata.json"
OUTPUT_CSV = ROOT / "mbt_shortlist_wide_heatmap_metrics.csv"
OUTPUT_METADATA = ROOT / "mbt_shortlist_wide_run_metadata.json"
OUTPUT_DIR = ROOT / "charts_shortlist_wide"
GALLERY_PATH = ROOT / "MBT_SHORTLIST_5_WIDE_HEATMAPS.png"
FROZEN_DIR = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
DEFAULT_DATA = REPO_ROOT / "data" / "downloaded" / "MBT_v_0_1m_20210425_20261003.parquet"


@dataclass(frozen=True)
class WideGrid:
    x_name: str
    x_values: tuple[float, ...]
    y_name: str
    y_values: tuple[float, ...]
    nominal_x: float
    nominal_y: float
    apply: Callable[[dict[str, Any], float, float], None]


def _set(structural: dict[str, Any], **updates: Any) -> None:
    structural.update(updates)


def wide_grids() -> dict[str, WideGrid]:
    return {
        "rolling_range_breakout": WideGrid(
            "entry_channel_length",
            (10, 14, 18, 20, 24, 30, 40, 50),
            "exit_channel_length",
            (5, 7, 10, 12, 15, 20),
            20,
            10,
            lambda s, x, y: _set(s, entry_channel_length=int(x), exit_channel_length=int(y)),
        ),
        "session_range_breakout": WideGrid(
            "opening_range_minutes",
            (15, 30, 45, 60, 75, 90, 120),
            "breakout_buffer_ticks",
            (0, 1, 2, 3, 4, 6),
            30,
            1,
            lambda s, x, y: _set(s, opening_range_minutes=int(x), breakout_buffer_ticks=int(y)),
        ),
        "trend_pullback_ma_structure": WideGrid(
            "fast_pair_scale",
            (0.40, 0.50, 0.60, 0.75, 1.00, 1.25, 1.50),
            "slow_tail_scale",
            (0.50, 0.75, 1.00, 1.25, 1.50),
            1.00,
            1.00,
            _ribbon,
        ),
        "volatility_breakout": WideGrid(
            "daily_atr_period",
            (7, 10, 14, 20, 30, 40),
            "entry_atr_multiple",
            (0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.20),
            14,
            0.60,
            lambda s, x, y: _set(s, daily_atr_period=int(x), entry_atr_multiple=float(y)),
        ),
        "momentum_confirmed_breakout": WideGrid(
            "breakout_lookback",
            (10, 14, 20, 30, 40, 60),
            "adx_threshold",
            (10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0, 35.0),
            20,
            25.0,
            lambda s, x, y: _set(s, breakout_lookback=int(x), adx_threshold=float(y)),
        ),
    }


def _existing_cell(existing: pd.DataFrame, mechanism_id: str, x: float, y: float) -> pd.Series | None:
    candidates = existing.loc[existing["mechanism_id"].eq(mechanism_id)]
    if candidates.empty:
        return None
    mask = np.isclose(candidates["x_value"].astype(float), float(x)) & np.isclose(candidates["y_value"].astype(float), float(y))
    matches = candidates.loc[mask]
    return None if matches.empty else matches.iloc[0]


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{float(value):g}"


def _plot(frame: pd.DataFrame, grid: WideGrid, periods: dict[str, list[str]], path: Path) -> None:
    nx, ny = len(grid.x_values), len(grid.y_values)
    matrices = {
        scope: frame.pivot(index="y_index", columns="x_index", values=f"{scope}_daily_pnl_sharpe")
        .reindex(index=range(ny), columns=range(nx))
        .to_numpy(dtype=float)
        for scope in ("is", "oos")
    }
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices.values()])
    bound = max(0.5, float(np.max(np.abs(finite)))) if finite.size else 1.0
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    width = max(21.0, 2.15 * nx + 5.0)
    height = max(7.5, 1.05 * ny + 2.8)
    fig, axes = plt.subplots(1, 2, figsize=(width, height), constrained_layout=True)
    mechanism = str(frame["mechanism_id"].iloc[0])
    strategy = str(frame["strategy_id"].iloc[0])
    image = None
    for axis, scope in zip(axes, ("is", "oos")):
        matrix = matrices[scope]
        image = axis.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto", origin="lower")
        label = scope.upper()
        start, end = periods[f"{scope}_range"]
        axis.set_title(f"{label} · {start} → {end}\nSharpe annualisé du PnL quotidien", fontsize=17, weight="bold", pad=12)
        axis.set_xlabel(grid.x_name, fontsize=15, labelpad=10)
        axis.set_ylabel(grid.y_name, fontsize=15, labelpad=10)
        axis.set_xticks(range(nx), [_fmt(value) for value in grid.x_values])
        axis.set_yticks(range(ny), [_fmt(value) for value in grid.y_values])
        axis.tick_params(axis="both", labelsize=13, width=1.2, length=5)
        for y_index in range(ny):
            for x_index in range(nx):
                cell = frame.loc[(frame["x_index"] == x_index) & (frame["y_index"] == y_index)].iloc[0]
                sharpe = float(cell[f"{scope}_daily_pnl_sharpe"])
                pnl = float(cell[f"{scope}_net_pnl_usd"])
                trades = int(cell[f"{scope}_trades"])
                color = "white" if abs(sharpe) >= 0.64 * bound else "#111111"
                axis.text(
                    x_index,
                    y_index,
                    f"S {sharpe:.2f}\n${pnl:,.0f}\n{trades} tr",
                    ha="center",
                    va="center",
                    fontsize=9.2,
                    weight="semibold",
                    color=color,
                    linespacing=1.25,
                )
        nominal_x = list(grid.x_values).index(grid.nominal_x)
        nominal_y = list(grid.y_values).index(grid.nominal_y)
        axis.add_patch(Rectangle((nominal_x - 0.485, nominal_y - 0.485), 0.97, 0.97, fill=False, edgecolor="#1259e8", linewidth=4.0))
    colorbar = fig.colorbar(image, ax=axes, shrink=0.88, pad=0.02)
    colorbar.set_label("Sharpe annualisé du PnL quotidien", fontsize=14)
    colorbar.ax.tick_params(labelsize=12)
    fig.suptitle(f"MBT · {strategy}\n{mechanism} · paysage élargi {nx}×{ny} · nominal encadré en bleu", fontsize=19, weight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _gallery(paths: list[Path]) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(24, 30), constrained_layout=True)
    for axis, path in zip(axes, paths):
        axis.imshow(plt.imread(path))
        axis.axis("off")
    fig.suptitle("MBT — shortlist : paysages de robustesse élargis", fontsize=26, weight="bold")
    fig.savefig(GALLERY_PATH, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def execute(data_path: Path) -> None:
    existing = pd.read_csv(SOURCE_CSV)
    source_metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    start_date = date.fromisoformat(source_metadata["date_range"][0])
    end_date = date.fromisoformat(source_metadata["date_range"][1])
    is_start, is_end = (date.fromisoformat(value) for value in source_metadata["is_range"])
    oos_start, oos_end = (date.fromisoformat(value) for value in source_metadata["oos_range"])
    prepared, roll_sessions = prepare_btc_data(data_path, start_date, end_date)
    specs, _, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    specs_by_mechanism = {spec["mechanism_id"]: spec for spec in specs}
    grids = wide_grids()
    config = mbt_config()
    rows: list[dict[str, Any]] = []
    reused = 0
    computed = 0

    for strategy_number, (mechanism_id, grid) in enumerate(grids.items(), start=1):
        print(f"[{strategy_number}/5] {mechanism_id}", flush=True)
        source_spec = specs_by_mechanism[mechanism_id]
        for y_index, y_value in enumerate(grid.y_values):
            for x_index, x_value in enumerate(grid.x_values):
                cached = _existing_cell(existing, mechanism_id, x_value, y_value)
                if cached is not None:
                    row = cached.to_dict()
                    reused += 1
                    result_source = "reused_original_grid"
                else:
                    spec = copy.deepcopy(adapt_spec_to_btc(source_spec, config))
                    grid.apply(spec["canonical_parameters"]["structural"], x_value, y_value)
                    run = run_baseline_backtest(spec, prepared)
                    is_metrics = _split_metrics(run, is_start, is_end)
                    oos_metrics = _split_metrics(run, oos_start, oos_end)
                    row = {"mechanism_id": mechanism_id, "strategy_id": source_spec["strategy_id"]}
                    row.update({f"is_{key}": value for key, value in is_metrics.items()})
                    row.update({f"oos_{key}": value for key, value in oos_metrics.items()})
                    computed += 1
                    result_source = "new_backtest"
                row.update(
                    {
                        "x_parameter": grid.x_name,
                        "x_value": x_value,
                        "x_index": x_index,
                        "y_parameter": grid.y_name,
                        "y_value": y_value,
                        "y_index": y_index,
                        "is_nominal_cell": bool(x_value == grid.nominal_x and y_value == grid.nominal_y),
                        "result_source": result_source,
                    }
                )
                rows.append(row)
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    result = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chart_paths: list[Path] = []
    for number, (mechanism_id, grid) in enumerate(grids.items(), start=1):
        frame = result.loc[result["mechanism_id"].eq(mechanism_id)].copy()
        path = OUTPUT_DIR / f"{number:02d}_{mechanism_id}_wide_is_oos.png"
        _plot(frame, grid, source_metadata, path)
        chart_paths.append(path)
    _gallery(chart_paths)
    metadata = {
        "run_type": "mbt_shortlist_wide_parameter_landscapes",
        "source_metrics": str(SOURCE_CSV.relative_to(REPO_ROOT)),
        "data_path": str(data_path.relative_to(REPO_ROOT)),
        "data_sha256": prepared.source_sha256,
        "date_range": source_metadata["date_range"],
        "is_range": source_metadata["is_range"],
        "oos_range": source_metadata["oos_range"],
        "instrument": source_metadata["instrument"],
        "strategy_count": 5,
        "total_cells": int(len(result)),
        "reused_original_cells": reused,
        "new_backtests": computed,
        "intraday_roll_sessions_excluded": [str(day) for day in roll_sessions],
        "grids": {
            mechanism: {
                "x_parameter": grid.x_name,
                "x_values": list(grid.x_values),
                "y_parameter": grid.y_name,
                "y_values": list(grid.y_values),
                "nominal": [grid.nominal_x, grid.nominal_y],
            }
            for mechanism, grid in grids.items()
        },
    }
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"cells={len(result)} reused={reused} new_backtests={computed}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    execute(args.data.resolve())


if __name__ == "__main__":
    main()
