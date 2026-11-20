"""IS equity curves (cumulative net R, default parameters) for REPORT.md.  Reads outputs_is/trades_is_default.csv."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
COLORS = {"MNQ": "#2a78d6", "MES": "#eb6834"}  # reference palette slots 1-2 (validated pair)
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TITLES = {"A": "A · Inside Day False-Break", "B": "B · Intraday 3-Push"}


def main() -> None:
    t = pd.read_csv(HERE / "outputs_is" / "trades_is_default.csv", parse_dates=["signal_time"])
    t = t[t["status"] == "filled"].sort_values("signal_time")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, facecolor=SURFACE)
    for ax, setup in zip(axes, ("A", "B")):
        ax.set_facecolor(SURFACE)
        for sym, col in COLORS.items():
            s = t[(t["setup_code"] == setup) & (t["instrument"] == sym)]
            if s.empty:
                continue
            x = pd.to_datetime(s["signal_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
            y = s["r_net"].cumsum()
            x = pd.concat([pd.Series([pd.Timestamp("2021-01-01")]), x])
            y = pd.concat([pd.Series([0.0]), y])
            ax.step(x, y, where="post", color=col, lw=2, label=f"{sym} ({len(s)} trades)")
            ax.annotate(f"{sym} {y.iloc[-1]:+.1f} R", (x.iloc[-1], y.iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=9, color=TEXT)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(TITLES[setup], loc="left", fontsize=11, color=TEXT)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2024-04-30"))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper left")
    axes[0].set_ylabel("R nets cumulés (1 contrat)", color=MUTED, fontsize=9)
    fig.suptitle("Dans l'échantillon 2021–2023, paramètres par défaut (5 min, fin 11:00, N = 2)",
                 x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(HERE / "equity_is.png", dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
