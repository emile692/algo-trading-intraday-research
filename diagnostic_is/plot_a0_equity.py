"""Plot the preregistered A0 NQ equity curves with IS and OOS kept separate."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRADES = ROOT / "diagnostic_is" / "phase2_outputs" / "A_phase2_trades.csv"
OUTPUT = ROOT / "diagnostic_is" / "phase2_outputs" / "A0_NQ_equity_IS_OOS.png"

PERIODS = {
    "IS · 2010-06-07 → 2024-02-22": ("2010-06-07", "2024-02-22"),
    "OOS · 2024-02-23 → 2026-03-19": ("2024-02-23", "2026-03-19"),
}


def daily_curve(trades: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    selected = trades[trades["session_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    daily = selected.groupby("session_date", as_index=True).agg(net_R=("net_R", "sum"), net_usd=("net_pnl", "sum"))
    calendar = pd.bdate_range(start, end)
    daily = daily.reindex(calendar, fill_value=0.0)
    daily["equity_R"] = daily["net_R"].cumsum()
    daily["equity_usd"] = daily["net_usd"].cumsum()
    return daily


def main() -> None:
    trades = pd.read_csv(TRADES, parse_dates=["session_date"])
    trades = trades[(trades["asset"] == "NQ") & (trades["variant"] == "A0")].copy()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    colors = ["#8c3b3b", "#167c5a"]

    for ax, (label, bounds), color in zip(axes, PERIODS.items(), colors):
        daily = daily_curve(trades, *bounds)
        equity = daily["equity_R"]
        drawdown = equity - pd.concat([pd.Series([0.0]), equity.reset_index(drop=True)], ignore_index=True).cummax().iloc[1:].to_numpy()
        max_drawdown = -float(drawdown.min())
        terminal_r = float(equity.iloc[-1])
        terminal_usd = float(daily["equity_usd"].iloc[-1])

        ax.plot(equity.index, equity, color=color, linewidth=1.7)
        ax.axhline(0, color="#303030", linewidth=0.8)
        ax.fill_between(equity.index, equity, 0, where=equity.ge(0), color=color, alpha=0.10)
        ax.fill_between(equity.index, equity, 0, where=equity.lt(0), color="#b23a48", alpha=0.10)
        ax.set_title(label, fontsize=13, weight="bold")
        ax.set_ylabel("PnL net cumulé (R)")
        ax.set_xlabel("Date")
        ax.text(
            0.03,
            0.95,
            f"Fin : {terminal_r:+.2f} R / {terminal_usd:+,.0f} $\nMax DD : {max_drawdown:.2f} R",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": color, "alpha": 0.92},
        )

    fig.suptitle("Signal A0 — NQ — equity nette en unités de risque", fontsize=16, weight="bold")
    fig.text(0.5, -0.01, "IS et OOS sont remis à zéro séparément ; aucun full sample agrégé.", ha="center", fontsize=10)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
