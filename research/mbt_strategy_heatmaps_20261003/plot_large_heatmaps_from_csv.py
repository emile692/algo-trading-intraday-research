"""Render large MBT IS/OOS heatmaps from saved metrics only.

This module deliberately has no dependency on, or import from, any backtest
engine.  It treats mbt_heatmap_metrics.csv as the immutable source of results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "mbt_heatmap_metrics.csv"
METADATA_JSON = ROOT / "run_metadata.json"
OUTPUT_DIR = ROOT / "charts_large"
GALLERY_PATH = ROOT / "MBT_16_HEATMAPS_IS_OOS_LARGE_GALLERY.png"


def _value_label(value: float) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _ordered_values(frame: pd.DataFrame, value_column: str, index_column: str) -> list[float]:
    return (
        frame[[index_column, value_column]]
        .drop_duplicates()
        .sort_values(index_column)[value_column]
        .astype(float)
        .tolist()
    )


def _matrix(frame: pd.DataFrame, metric: str) -> np.ndarray:
    return (
        frame.pivot(index="y_index", columns="x_index", values=metric)
        .reindex(index=range(3), columns=range(3))
        .to_numpy(dtype=float)
    )


def _plot_strategy(
    frame: pd.DataFrame,
    *,
    is_period: tuple[str, str],
    oos_period: tuple[str, str],
    output_path: Path,
) -> None:
    mechanism_id = str(frame["mechanism_id"].iloc[0])
    strategy_id = str(frame["strategy_id"].iloc[0])
    x_parameter = str(frame["x_parameter"].iloc[0])
    y_parameter = str(frame["y_parameter"].iloc[0])
    x_values = _ordered_values(frame, "x_value", "x_index")
    y_values = _ordered_values(frame, "y_value", "y_index")
    is_matrix = _matrix(frame, "is_daily_pnl_sharpe")
    oos_matrix = _matrix(frame, "oos_daily_pnl_sharpe")

    finite = np.concatenate((is_matrix[np.isfinite(is_matrix)], oos_matrix[np.isfinite(oos_matrix)]))
    bound = max(0.5, float(np.max(np.abs(finite)))) if finite.size else 1.0
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.5), constrained_layout=True)
    images = []
    panel_specs = (
        (axes[0], "IS", is_period, is_matrix),
        (axes[1], "OOS", oos_period, oos_matrix),
    )
    for axis, scope, period, matrix in panel_specs:
        image = axis.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto", origin="lower")
        images.append(image)
        axis.set_title(
            f"{scope} · {period[0]} → {period[1]}\nSharpe annualisé du PnL quotidien",
            fontsize=16,
            weight="bold",
            pad=12,
        )
        axis.set_xlabel(x_parameter, fontsize=14, labelpad=9)
        axis.set_ylabel(y_parameter, fontsize=14, labelpad=9)
        axis.set_xticks(range(3), [_value_label(value) for value in x_values])
        axis.set_yticks(range(3), [_value_label(value) for value in y_values])
        axis.tick_params(axis="both", labelsize=13, width=1.3, length=5)

        for y_index in range(3):
            for x_index in range(3):
                cell = frame.loc[(frame["x_index"] == x_index) & (frame["y_index"] == y_index)].iloc[0]
                sharpe = float(cell[f"{scope.lower()}_daily_pnl_sharpe"])
                pnl = float(cell[f"{scope.lower()}_net_pnl_usd"])
                trades = int(cell[f"{scope.lower()}_trades"])
                sharpe_text = f"{sharpe:.2f}" if np.isfinite(sharpe) else "n/a"
                text_color = "white" if np.isfinite(sharpe) and abs(sharpe) >= 0.62 * bound else "#111111"
                axis.text(
                    x_index,
                    y_index,
                    f"Sharpe {sharpe_text}\nPnL ${pnl:,.0f}\n{trades} trades",
                    ha="center",
                    va="center",
                    fontsize=11,
                    weight="semibold",
                    color=text_color,
                    linespacing=1.35,
                )

        nominal = frame.loc[frame["is_nominal_cell"].astype(str).str.lower().eq("true")].iloc[0]
        nominal_x = int(nominal["x_index"])
        nominal_y = int(nominal["y_index"])
        axis.add_patch(
            Rectangle(
                (nominal_x - 0.485, nominal_y - 0.485),
                0.97,
                0.97,
                fill=False,
                edgecolor="#1259e8",
                linewidth=4.0,
                zorder=5,
            )
        )
        for spine in axis.spines.values():
            spine.set_linewidth(1.2)

    colorbar = fig.colorbar(images[0], ax=axes, shrink=0.88, pad=0.025)
    colorbar.set_label("Sharpe annualisé du PnL quotidien", fontsize=13)
    colorbar.ax.tick_params(labelsize=12)
    fig.suptitle(
        f"MBT · {strategy_id}\n{mechanism_id} · grille locale 3×3 · cellule nominale encadrée en bleu",
        fontsize=18,
        weight="bold",
    )
    fig.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _build_gallery(chart_paths: list[Path]) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(28, 18), constrained_layout=True)
    for axis, chart_path in zip(axes.flat, chart_paths):
        axis.imshow(plt.imread(chart_path))
        axis.set_title(chart_path.stem.split("_", 1)[1], fontsize=10)
        axis.axis("off")
    fig.suptitle("MBT — 16 heatmaps détaillées IS/OOS", fontsize=25, weight="bold")
    fig.savefig(GALLERY_PATH, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    metrics = pd.read_csv(SOURCE_CSV)
    metadata = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chart_paths: list[Path] = []
    for number, (mechanism_id, frame) in enumerate(metrics.groupby("mechanism_id", sort=False), start=1):
        output_path = OUTPUT_DIR / f"{number:02d}_{mechanism_id}_is_oos_large.png"
        _plot_strategy(
            frame,
            is_period=tuple(metadata["is_range"]),
            oos_period=tuple(metadata["oos_range"]),
            output_path=output_path,
        )
        chart_paths.append(output_path)
        print(output_path.name, flush=True)

    if len(chart_paths) != 16:
        raise RuntimeError(f"Expected 16 strategy charts, generated {len(chart_paths)}")
    _build_gallery(chart_paths)


if __name__ == "__main__":
    main()
