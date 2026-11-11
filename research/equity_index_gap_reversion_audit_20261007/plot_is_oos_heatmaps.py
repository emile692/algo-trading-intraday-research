from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SOURCE = RESULTS / "frozen_positive_region_all_cells.csv"
ASSETS = ["MNQ", "MES", "M2K", "NQ"]
THRESHOLDS = [0.25, 0.50, 0.75, 1.00]


def annotated_heatmap(ax, values, title, fmt, cmap, center=None):
    finite = values[np.isfinite(values)]
    if center is not None and finite.size:
        span = max(abs(finite.min() - center), abs(finite.max() - center))
        image = ax.imshow(values, cmap=cmap, vmin=center - span, vmax=center + span)
    else:
        image = ax.imshow(values, cmap=cmap)

    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xticks(range(len(THRESHOLDS)), [f"{x:.2f}" for x in THRESHOLDS])
    ax.set_yticks(range(len(ASSETS)), ASSETS)
    ax.set_xlabel("Seuil de gap normalisé (ex ante)")

    midpoint = (image.norm.vmin + image.norm.vmax) / 2
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                color = "white" if image.norm(value) < 0.25 or image.norm(value) > 0.75 else "black"
                ax.text(col, row, fmt.format(value), ha="center", va="center", color=color, fontsize=9)
    return image


def matrix(frame, column):
    return (
        frame.pivot(index="asset", columns="threshold", values=column)
        .reindex(index=ASSETS, columns=THRESHOLDS)
        .to_numpy(dtype=float)
    )


def main():
    data = pd.read_csv(SOURCE)
    data["threshold"] = data["threshold"].astype(float)

    specs = [
        ("net_pnl", "PnL net ($)", "{:.0f}", "RdYlGn", 0.0),
        ("sharpe", "Sharpe", "{:.2f}", "RdYlGn", 0.0),
        ("profit_factor", "Profit factor", "{:.2f}", "RdYlGn", 1.0),
        ("trades", "Nombre de trades", "{:.0f}", "Blues", None),
    ]

    fig, axes = plt.subplots(len(specs), 2, figsize=(13, 16), constrained_layout=True)
    for row, (suffix, label, fmt, cmap, center) in enumerate(specs):
        for col, sample in enumerate(("is", "oos")):
            values = matrix(data, f"{sample}_{suffix}")
            image = annotated_heatmap(
                axes[row, col], values, f"{sample.upper()} — {label}", fmt, cmap, center
            )
            fig.colorbar(image, ax=axes[row, col], shrink=0.82)

    fig.suptitle(
        "Région gelée de réversion du gap — diagnostics IS / OOS\n"
        "Lookback = 3 barres, sortie = clôture complète du gap, coûts nominaux inclus",
        fontsize=16,
        weight="bold",
    )
    output = RESULTS / "frozen_region_is_oos_heatmaps.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
