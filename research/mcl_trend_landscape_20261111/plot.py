"""Sharpe heatmaps from landscape.csv.  python -m research.mcl_trend_landscape_20261111.plot"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
CMAP = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
CMAP.set_bad("#e6e5e0")
LIM = 1.5
PANELS = [("PRE_CL", "CL 2015-01 → 2021-07 (contrôle)"), ("IS", "MCL 2021-07 → 2023-12 (IS)"), ("OOS", "MCL 2024-01 → 2026-03 (OOS)")]
CHARTS = [
    ("s7_core", "S7 intraday — signe EMA rapide/lente (15 min), tenu 09:00–14:30 New York", "EMA lente (barres 15 min)", "EMA rapide", (50, 200)),
    ("s7_day", "S7 journée Topstep — signe EMA rapide/lente (15 min), tenu 18:00 → 16:00 New York", "EMA lente (barres 15 min)", "EMA rapide", (50, 200)),
    ("s8", "S8 — cassure Donchian intraday (barres 15 min), première cassure du jour", "Heure de début des entrées (New York)", "Longueur du canal (barres)", (20, "09:00")),
]


def main() -> None:
    res = pd.read_csv(HERE / "landscape.csv", dtype={"p2": str})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    for key, title, xlabel, ylabel, canon in CHARTS:
        sub = res[res.strategy == key]
        ys = sorted(sub.p1.unique())
        xs = sorted(sub.p2.unique(), key=lambda v: (len(v), v)) if key == "s8" else [str(v) for v in sorted(sub.p2.astype(int).unique())]
        fig, axes = plt.subplots(1, 3, figsize=(15, 0.5 * len(ys) + 2.6), facecolor=SURFACE, sharey=True)
        for ax, (window, label) in zip(axes, PANELS):
            grid = sub[sub.window == window].pivot(index="p1", columns="p2", values="sharpe").reindex(index=ys, columns=xs)
            data = np.ma.masked_invalid(grid.to_numpy(dtype=float))
            im = ax.pcolormesh(data, cmap=CMAP, norm=TwoSlopeNorm(0, -LIM, LIM), edgecolors=SURFACE, linewidth=2)
            for i, y in enumerate(ys):
                for j, x in enumerate(xs):
                    v = grid.iloc[i, j]
                    if pd.notna(v):
                        ax.text(j + 0.5, i + 0.5, f"{v:+.2f}".replace(".", ","), ha="center", va="center", fontsize=8.5,
                                color="#ffffff" if abs(v) > 0.95 else INK)
                    if (y, x) == (canon[0], str(canon[1])):
                        ax.add_patch(plt.Rectangle((j + 0.06, i + 0.06), 0.88, 0.88, fill=False, ec=INK, lw=1.6))
            ax.set_xticks(np.arange(len(xs)) + 0.5, xs)
            ax.set_yticks(np.arange(len(ys)) + 0.5, ys)
            ax.set_title(label, fontsize=10, color=INK, loc="left", pad=8)
            ax.set_xlabel(xlabel)
            ax.set_facecolor(SURFACE)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            pos = int((grid > 0).sum().sum())
            ax.text(1.0, 1.02, f"{pos}/{int(grid.notna().sum().sum())} cellules > 0", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK2)
        axes[0].set_ylabel(ylabel)
        fig.suptitle(title, x=0.012, ha="left", fontsize=12, color=INK, fontweight="bold")
        fig.text(0.012, 0.015, "Sharpe annualisé du PnL journalier net (1 contrat, 4,50 USD par aller-retour). Cadre noir : cellule testée dans la campagne initiale. "
                 "Gris : combinaison non valide.", fontsize=8.5, color=INK2)
        fig.tight_layout(rect=(0, 0.04, 0.93, 0.94))
        cax = fig.add_axes((0.945, 0.2, 0.012, 0.6))
        cb = fig.colorbar(im, cax=cax, ticks=[-LIM, -0.75, 0, 0.75, LIM])
        cb.outline.set_visible(False)
        cb.ax.set_yticklabels([f"≤ −{LIM}".replace(".", ","), "−0,75", "0", "+0,75", f"≥ +{LIM}".replace(".", ",")])
        cb.ax.tick_params(length=0)
        fig.savefig(HERE / f"heatmap_{key}.png", dpi=130, facecolor=SURFACE)
        plt.close(fig)


if __name__ == "__main__":
    main()
