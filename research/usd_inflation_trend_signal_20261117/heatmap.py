"""Sharpe heatmaps IS / OOS of the MA-crossover signal B alone, fast x slow MA, minimal lag.

Asked by the user on 2026-11-17 after the CLOSED_IS verdict: reading the OOS here unseals it for the whole grid.

Lag conventions (same P&L formula as the workbook: pnl(t) = pos(t-1) * r(t) - |dpos| * cost):
  L=1 : workbook minimum (Main!D14 = 1). Return of day t uses the nowcast dated t-2 (trading days).
  L=0 : sensitivity. Return of day t uses the nowcast dated t-1: only valid if that value is
        published before the DX close of t-1.

    uv run --no-project --with pandas --with openpyxl --with matplotlib python heatmap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

FAST = (1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120)
SLOW = (20, 30, 45, 60, 90, 120, 150, 180, 250, 365)
IS = (pd.Timestamp("2017-03-01"), pd.Timestamp("2021-12-31"))  # common window: slow 365 valid
OOS = (pd.Timestamp("2022-01-01"), None)

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
CMAP = LinearSegmentedColormap.from_list(
    "div", ["#8f1f1f", "#e34948", "#f0efec", "#3987e5", "#104281"])


def grid(yoy, ret, lag):
    tdates = ret.index
    out = {}
    for name, (a, b) in {"IS": IS, "OOS": OOS}.items():
        out[name] = pd.DataFrame(np.nan, index=FAST, columns=SLOW)
    for f in FAST:
        for s in SLOW:
            if f >= s:
                continue
            sig = run.signal_trend(yoy, tdates, (f,), (s,), lags=(lag,))
            p = run.pnl(sig, ret)
            for name, (a, b) in {"IS": IS, "OOS": OOS}.items():
                out[name].loc[f, s] = run.sharpe(run.window(p, a, b))
    sigA = run.signal_accel(yoy, tdates, lags=(lag,))
    pA = run.pnl(sigA, ret)
    refA = {name: run.sharpe(run.window(pA, a, b)) for name, (a, b) in {"IS": IS, "OOS": OOS}.items()}
    return out, refA


def plot(out, refA, lag, path):
    vmax = np.nanmax(np.abs(np.concatenate([out["IS"].to_numpy().ravel(), out["OOS"].to_numpy().ravel()])))
    vmax = float(np.ceil(vmax * 10) / 10)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), facecolor=SURFACE, constrained_layout=True)
    labels = {"IS": f"IS {IS[0]:%Y-%m} → {IS[1]:%Y-%m}", "OOS": "OOS 2022-01 → 2026-09"}
    for ax, name in zip(axes, ("IS", "OOS")):
        m = out[name]
        ax.set_facecolor(SURFACE)
        im = ax.imshow(m.to_numpy(), cmap=CMAP, norm=norm, aspect="auto")
        for i, f in enumerate(m.index):
            for j, s in enumerate(m.columns):
                v = m.loc[f, s]
                if np.isnan(v):
                    continue
                rgba = CMAP(norm(v))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if lum < 0.5 else INK)
        ax.set_xticks(range(len(m.columns)), [str(c) for c in m.columns], color=INK2, fontsize=9)
        ax.set_yticks(range(len(m.index)), [str(r) for r in m.index], color=INK2, fontsize=9)
        ax.set_xlabel("MA lente (jours calendaires)", color=INK2, fontsize=10)
        ax.set_ylabel("MA rapide (jours calendaires ; 1 = YoY brut)", color=INK2, fontsize=10)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        pos = m.stack()
        ax.set_title(f"{labels[name]}\nSignal A, même lag : Sharpe {refA[name]:.2f}   ·   "
                     f"paires > 0 : {(pos > 0).sum()}/{len(pos)}   ·   médiane {pos.median():.2f}",
                     color=INK, fontsize=10.5, loc="left")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("Sharpe net (2 bp one-way)", color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, length=0)
    lag_txt = ("lag 1 : rendement du jour t tradé avec le nowcast daté t−2"
               if lag == 1 else "lag 0 : rendement du jour t tradé avec le nowcast daté t−1 (suppose publication avant la clôture DX)")
    fig.suptitle(f"USD Inflation — croisement de MA seul (sign(MA rapide − MA lente) du YoY), DX1 · {lag_txt}",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    yoy, ret = run.load()
    for lag in (1, 0):
        out, refA = grid(yoy, ret, lag)
        for name, m in out.items():
            m.to_csv(run.OUT / f"heatmap_sharpe_{name.lower()}_lag{lag}.csv")
        plot(out, refA, lag, run.OUT / f"heatmap_sharpe_lag{lag}.png")
        is_, oos = out["IS"].stack(), out["OOS"].stack()
        print(f"lag {lag}: A IS {refA['IS']:.3f} OOS {refA['OOS']:.3f} | "
              f"B IS median {is_.median():.3f} pos {(is_ > 0).mean():.0%} max {is_.max():.2f} {is_.idxmax()} | "
              f"OOS median {oos.median():.3f} pos {(oos > 0).mean():.0%} max {oos.max():.2f} {oos.idxmax()} | "
              f"rank corr IS-OOS {is_.rank().corr(oos.rank()):.2f} | "
              f"OOS of IS-best {oos[is_.idxmax()]:.2f} | cells beating A: IS {(is_ > refA['IS']).sum()}, OOS {(oos > refA['OOS']).sum()} / {len(is_)}")


if __name__ == "__main__":
    main()
