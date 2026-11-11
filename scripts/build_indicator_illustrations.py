"""Build deterministic teaching diagrams for the nine client signal sheets.

These figures explain indicator mechanics. They are deliberately schematic and
must not be interpreted as market observations or backtest results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "signal_sheets" / "assets"
BLUE = "#1769aa"
NAVY = "#102b4e"
GREEN = "#25855a"
RED = "#c45a4a"
GOLD = "#d9a21b"
GREY = "#7b8794"


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.text(0.99, 0.012, "Schéma pédagogique — valeurs illustratives", ha="right", fontsize=8, color=GREY)
    fig.savefig(path, dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _candles(ax: plt.Axes, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> None:
    for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
        color = GREEN if c >= o else RED
        ax.vlines(i, l, h, color=color, linewidth=1.3)
        ax.add_patch(plt.Rectangle((i - 0.31, min(o, c)), 0.62, max(abs(c - o), 0.08), color=color, alpha=0.85))


def build_orb_vwap() -> None:
    closes = np.array([100.2, 100.7, 100.1, 101.0, 100.6, 101.1, 101.3, 101.7, 102.1, 102.8, 103.2, 103.8])
    opens = np.r_[100.0, closes[:-1]]
    highs = np.maximum(opens, closes) + np.array([.4,.3,.25,.35,.2,.25,.2,.3,.25,.3,.25,.3])
    lows = np.minimum(opens, closes) - np.array([.2,.25,.3,.2,.25,.2,.15,.2,.2,.15,.2,.2])
    typical = (highs + lows + closes) / 3
    volumes = np.array([12, 15, 11, 17, 10, 9, 10, 12, 16, 20, 18, 17])
    vwap = np.cumsum(typical * volumes) / np.cumsum(volumes)
    or_high, or_low = highs[:5].max(), lows[:5].min()
    fig, ax = plt.subplots(figsize=(10.8, 4.5))
    _candles(ax, opens, highs, lows, closes)
    ax.fill_between(np.arange(5), or_low, or_high, color="#dbe8f4", alpha=.8, label="Opening range 09:30–09:45")
    ax.hlines([or_high, or_low], 0, 11, colors=GREY, linestyles="--", linewidth=1)
    ax.plot(vwap, color=GOLD, linewidth=2.2, label="VWAP cumulée")
    ax.scatter([9], [closes[9]], s=80, color=BLUE, zorder=5)
    ax.annotate("Cassure clôturée au-dessus de l’OR\n+ prix au-dessus de la VWAP\n+ filtre ATR favorable", (9, closes[9]), (6.0, 103.55),
                arrowprops={"arrowstyle": "->", "color": NAVY}, color=NAVY)
    ax.set_title("ORB + VWAP + filtre de régime ATR", weight="bold")
    ax.set_xlabel("Minutes successives (schéma)")
    ax.set_ylabel("Prix")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=.2)
    _finish(fig, ASSET_DIR / "01_indicator_orb_vwap.png")


def build_atr() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    ax = axes[0]
    ax.hlines(100, .1, 1.0, color=GREY, linestyle="--", label="Clôture J−1")
    ax.vlines(1.5, 103, 108, color=GREEN, linewidth=5)
    ax.scatter([1.5, 1.5], [103, 108], color=NAVY, s=35, zorder=4)
    ax.annotate("High−Low", (1.5, 105.5), (2.05, 105.5), arrowprops={"arrowstyle": "<->"}, va="center")
    ax.annotate("Gap pris en compte", (.8, 101.5), (1.55, 102.5), arrowprops={"arrowstyle": "->"})
    ax.set_xlim(0, 3.4); ax.set_ylim(98.5, 109.5)
    ax.set_xticks([.55, 1.5], ["J−1", "J"])
    ax.set_title("True Range d’une séance")
    ax.set_ylabel("Points de cours")
    ax.legend(loc="lower right")

    tr = np.array([72, 84, 76, 91, 110, 98, 121, 105, 95, 112, 126, 118, 109, 102])
    atr = np.empty_like(tr, dtype=float); atr[0] = tr[0]
    for i in range(1, len(tr)): atr[i] = (13 * atr[i - 1] + tr[i]) / 14
    axes[1].bar(np.arange(len(tr)), tr, color="#b8cee2", label="True Range quotidien")
    axes[1].plot(atr, color=BLUE, linewidth=2.4, label="ATR14 Wilder")
    axes[1].set_title("Lissage ATR : amplitude, pas direction")
    axes[1].set_xlabel("Séances terminées")
    axes[1].set_ylabel("Points de cours")
    axes[1].legend(loc="upper left")
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("Comment lire l’ATR", weight="bold")
    fig.tight_layout(rect=(0, .04, 1, .92))
    _finish(fig, ASSET_DIR / "indicator_atr.png")


def build_volatility_breakout(filename: str, instrument: str, open_price: float, atr: float, multiple: float, point_value: float) -> None:
    upper, lower = open_price + multiple * atr, open_price - multiple * atr
    path = np.array([open_price, open_price+.08*atr, open_price-.04*atr, open_price+.18*atr,
                     open_price+.34*atr, open_price+.58*atr, open_price+.72*atr, open_price+.55*atr])
    fig, ax = plt.subplots(figsize=(10.8, 4.3))
    ax.plot(path, color=NAVY, marker="o", linewidth=2)
    ax.axhline(upper, color=GREEN, linestyle="--", linewidth=2, label=f"U = O + {multiple:.2f} × ATR")
    ax.axhline(lower, color=RED, linestyle="--", linewidth=2, label=f"L = O − {multiple:.2f} × ATR")
    ax.axhline(open_price, color=GREY, linewidth=1.5, label="Open O = stop d’invalidation")
    hit = int(np.argmax(path >= upper))
    ax.scatter([hit], [path[hit]], s=100, color=GREEN, zorder=5)
    ax.annotate("Premier seuil touché → long", (hit, path[hit]), (hit-2.6, upper+.05*atr),
                arrowprops={"arrowstyle": "->", "color": GREEN}, color=GREEN)
    risk_usd = multiple * atr * point_value
    ax.text(.02, .045, f"Exemple : ATR = {atr:g} points\nDistance entrée–stop ≈ {risk_usd:,.0f} USD/contrat",
            transform=ax.transAxes, color=NAVY, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88, "pad": 2})
    ax.set_title(f"{instrument} — breakout normalisé par l’ATR", weight="bold")
    ax.set_xlabel("Temps intrajournalier (schéma)")
    ax.set_ylabel("Prix")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=.2)
    _finish(fig, ASSET_DIR / filename)


def build_compression() -> None:
    opens = np.array([100, 102, 101, 103, 102, 102.5])
    closes = np.array([102, 101, 103, 102, 102.4, 104.5])
    highs = np.array([103, 103.2, 104, 104.2, 103.0, 105])
    lows = np.array([99, 100, 100.5, 100, 101.4, 102.2])
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    _candles(ax, opens, highs, lows, closes)
    ax.add_patch(plt.Rectangle((3.55, lows[4]), .9, highs[4]-lows[4], fill=False, edgecolor=BLUE, linewidth=3))
    ax.annotate("Setup bar : inside + plus petite range (NR10)", (4, highs[4]), (1.4, 104.65),
                arrowprops={"arrowstyle": "->", "color": BLUE}, color=BLUE)
    ax.hlines([highs[4]+.25, lows[4]-.25], 4.4, 5.5, colors=[GREEN, RED], linestyles="--", linewidth=2)
    ax.annotate("Buy stop + 1 tick", (5.15, highs[4]+.25), (5.5, 104.0), arrowprops={"arrowstyle": "->"})
    ax.annotate("Sell stop − 1 tick", (4.7, lows[4]-.25), (2.4, 99.5), arrowprops={"arrowstyle": "->"})
    ax.set_title("Compression : mother bar, inside bar et range de setup", weight="bold")
    ax.set_xticks(range(6), ["barre −4", "−3", "−2", "Mother bar", "Setup", "Breakout"])
    ax.set_ylabel("Prix")
    ax.grid(axis="y", alpha=.2)
    _finish(fig, ASSET_DIR / "03_indicator_compression.png")


def build_trend_flip() -> None:
    x = np.arange(18)
    price = np.array([100,100.5,101,101.4,101.1,100.7,100.2,99.6,99.2,99.5,100,100.6,101.3,102,102.6,103,103.5,104])
    center = np.convolve(np.pad(price, (2, 0), mode="edge"), np.ones(3)/3, mode="valid")[:18]
    upper, lower = center + 1.15, center - 1.15
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    ax.plot(x, price, color=NAVY, linewidth=2.2, label="Clôture 15 min")
    ax.plot(x, upper, color=GREEN, linestyle="--", label="Bande haute ajustée ATR")
    ax.plot(x, lower, color=RED, linestyle="--", label="Bande basse ajustée ATR")
    flip = 12
    ax.scatter([flip], [price[flip]], color=BLUE, s=90, zorder=5)
    ax.annotate("Flip haussier confirmé à la clôture\n→ entrée minute suivante", (flip, price[flip]), (8.2, 104.3),
                arrowprops={"arrowstyle": "->", "color": BLUE}, color=BLUE)
    ax.annotate("Stop fixe = fill − ATR14", (14, price[14]-1.15), (13, 99.3), arrowprops={"arrowstyle": "->"})
    ax.set_title("ATR trend flip — bandes adaptatives et changement d’état", weight="bold")
    ax.set_xlabel("Barres 15 min terminées")
    ax.set_ylabel("Prix")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=.2)
    _finish(fig, ASSET_DIR / "04_indicator_trend_flip.png")


def build_gap() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    prev_close, open_today, entry = 2480, 2500, 2504
    midpoint, target = 2490, 2544
    ax.hlines(prev_close, 0, 2.2, color=GREY, linewidth=2, label="Clôture RTH J−1")
    ax.scatter([2.8], [open_today], s=90, color=BLUE, label="Open 08:20")
    ax.annotate("Gap = 20 points", (2.5, 2490), (.6, 2490), arrowprops={"arrowstyle": "<->"}, va="center")
    ax.add_patch(plt.Rectangle((3.2, 2498), .8, 8, color=GREEN, alpha=.8))
    ax.vlines(3.6, 2496, 2508, color=GREEN)
    ax.annotate("Bougie 08:20–08:35\nconfirme le gap", (3.6, 2507), (4.2, 2518), arrowprops={"arrowstyle": "->"})
    ax.hlines(midpoint, 2.3, 6.8, color=RED, linestyle="--", label="Stop : milieu du gap")
    ax.hlines(target, 4.0, 6.8, color=GREEN, linestyle="--", label="Cible : entrée + 2 × gap")
    ax.scatter([4.2], [entry], color=NAVY, s=70, label="Entrée 08:35")
    ax.set_xlim(0, 7); ax.set_ylim(2474, 2550)
    ax.set_xticks([1, 2.8, 3.6, 4.2], ["J−1", "08:20", "08:35 close", "Entrée"])
    ax.set_title("Gap continuation — confirmation, invalidation et cible", weight="bold")
    ax.set_ylabel("Prix MGC")
    ax.legend(loc="upper left", ncol=2, fontsize=8.5)
    ax.grid(axis="y", alpha=.2)
    _finish(fig, ASSET_DIR / "06_indicator_gap.png")


def build_donchian() -> None:
    x = np.arange(22)
    price = np.array([100,101,100.4,102,101.5,103,102.2,103.4,104,103.5,104.2,105,106,106.5,107,108,107.2,106.4,105.5,104.8,103.8,103])
    upper = np.array([np.nan if i < 5 else price[max(0, i-5):i].max() for i in x])
    lower = np.array([np.nan if i < 3 else price[max(0, i-3):i].min() for i in x])
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    ax.plot(x, price, color=NAVY, linewidth=2.2, label="Clôture 1 h")
    ax.step(x, upper, where="post", color=GREEN, linewidth=2, label="Canal haut d’entrée N")
    ax.step(x, lower, where="post", color=RED, linewidth=2, label="Canal bas de sortie M")
    ax.scatter([11], [price[11]], color=GREEN, s=90, zorder=5)
    ax.annotate("Clôture > canal précédent → long", (11, price[11]), (6.5, 107.55), arrowprops={"arrowstyle": "->"})
    ax.scatter([20], [price[20]], color=RED, s=90, zorder=5)
    ax.annotate("Clôture < canal de sortie → flat", (20, price[20]), (14.2, 101), arrowprops={"arrowstyle": "->"})
    ax.set_title("Canal de Donchian — cassure d’entrée et canal opposé de sortie", weight="bold")
    ax.set_xlabel("Barres 1 h terminées")
    ax.set_ylabel("Prix")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=.2)
    _finish(fig, ASSET_DIR / "07_indicator_donchian.png")


def build_ema_pullback() -> None:
    x = np.arange(28)
    price = np.array([100,100.5,101,101.8,102.2,103,103.6,104.2,105,105.8,106.3,106.8,107.4,107,106.5,106.1,106.6,107.3,108.1,109,109.6,110.2,110.8,111.2,111.8,112.5,113,113.4])
    def ema(values: np.ndarray, span: int) -> np.ndarray:
        alpha = 2 / (span + 1); out = np.empty_like(values, dtype=float); out[0] = values[0]
        for i in range(1, len(values)): out[i] = alpha * values[i] + (1-alpha) * out[i-1]
        return out
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    ax.plot(x, price, color=NAVY, linewidth=2.3, label="Prix")
    colors = ["#d9a21b", "#e07a3f", BLUE, "#7367a8", GREY]
    for span, color in zip([3, 5, 8, 13, 18], colors):
        ax.plot(x, ema(price, span), color=color, linewidth=1.5, label=f"EMA {span}")
    ax.axvspan(13, 16.5, color="#dbe8f4", alpha=.65)
    ax.annotate("Pullback vers les EMA rapides\ndans une structure ordonnée", (15, price[15]), (9.5, 111.5),
                arrowprops={"arrowstyle": "->", "color": BLUE}, color=BLUE)
    ax.annotate("Reprise → cellule longue", (17, price[17]), (18.5, 104.2), arrowprops={"arrowstyle": "->"})
    ax.set_title("Ruban d’EMA — tendance, pullback et reprise", weight="bold")
    ax.set_xlabel("Barres 15 min terminées")
    ax.set_ylabel("Prix")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=.2)
    _finish(fig, ASSET_DIR / "08_indicator_ema_pullback.png")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    build_orb_vwap()
    build_atr()
    build_volatility_breakout("02_indicator_atr_breakout.png", "MNQ", 20_000, 200, .60, 2.0)
    build_compression()
    build_trend_flip()
    build_volatility_breakout("05_indicator_atr_breakout.png", "MGC", 2_500, 20, 1.10, 10.0)
    build_gap()
    build_donchian()
    build_ema_pullback()
    build_volatility_breakout("09_indicator_atr_breakout.png", "MBT", 100_000, 4_000, .40, .10)
    for path in sorted(ASSET_DIR.glob("*indicator*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
