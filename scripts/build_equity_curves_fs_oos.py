"""Build Full Sample and OOS/validation equity curves for every signal sheet."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from research.locked_four_sleeve_portfolio.run_locked_four_sleeve_portfolio import run_alpha_sleeves


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "signal_sheets" / "assets"
MNQ_CONFIG = ROOT / "research" / "locked_four_sleeve_portfolio" / "locked_four_sleeve_config.json"
# Executed vote ensembles frozen on 2026-11-01 (see strategy_registry/psf_mnq/frozen_ensembles).
CLUSTER_DAILY = {
    sleeve: ROOT / "research" / f"signal_sheet_{number:02d}_cluster" / "daily_results.csv"
    for number, sleeve in (
        (5, "mgc_volatility_breakout"),
        (6, "mgc_gap_continuation"),
        (7, "mbt_rolling_range_breakout"),
        (8, "mbt_trend_pullback_ma_structure"),
        (9, "mbt_volatility_breakout"),
    )
}

LABELS = {
    "mnq_orb_vwap_atr": "01 · MNQ ORB + VWAP + ATR",
    "mnq_volatility_breakout": "02 · MNQ volatility breakout",
    "mnq_compression_breakout": "03 · MNQ compression breakout",
    "mnq_atr_trend_flip": "04 · MNQ ATR trend flip",
    "mgc_volatility_breakout": "05 · MGC volatility breakout",
    "mgc_gap_continuation": "06 · MGC gap continuation",
    "mbt_rolling_range_breakout": "07 · MBT rolling range breakout",
    "mbt_trend_pullback_ma_structure": "08 · MBT trend pullback MA",
    "mbt_volatility_breakout": "09 · MBT volatility breakout",
}

OOS_START = {
    "mnq_orb_vwap_atr": "2024-02-28",
    "mnq_volatility_breakout": "2024-01-01",
    "mnq_compression_breakout": "2024-01-01",
    "mnq_atr_trend_flip": "2024-01-01",
    "mgc_volatility_breakout": "2024-01-01",
    "mgc_gap_continuation": "2024-01-01",
    "mbt_rolling_range_breakout": "2025-05-29",
    "mbt_trend_pullback_ma_structure": "2025-05-29",
    "mbt_volatility_breakout": "2025-05-29",
}

OOS_STATUS = {
    "mnq_orb_vwap_atr": "OOS chronologique",
    "mnq_volatility_breakout": "OOS / validation réutilisée",
    "mnq_compression_breakout": "OOS / validation réutilisée",
    "mnq_atr_trend_flip": "OOS / validation réutilisée",
    "mgc_volatility_breakout": "OOS / validation réutilisée",
    "mgc_gap_continuation": "OOS / validation réutilisée",
    "mbt_rolling_range_breakout": "Legacy OOS réutilisé",
    "mbt_trend_pullback_ma_structure": "Legacy OOS réutilisé",
    "mbt_volatility_breakout": "Legacy OOS réutilisé",
}


def _daily(frame: pd.DataFrame, pnl_column: str = "daily_pnl_usd") -> pd.DataFrame:
    out = frame[["session_date", pnl_column]].copy()
    out["session_date"] = pd.to_datetime(out["session_date"])
    out[pnl_column] = pd.to_numeric(out[pnl_column], errors="raise").fillna(0.0)
    return out.sort_values("session_date").rename(columns={pnl_column: "daily_pnl_usd"})


def load_full_sample_daily() -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}

    mnq_config = json.loads(MNQ_CONFIG.read_text(encoding="utf-8"))
    mnq_runs, _, _ = run_alpha_sleeves(mnq_config)
    for source in ("volatility_breakout", "compression_breakout", "atr_trend_flip"):
        result[f"mnq_{source}"] = _daily(mnq_runs[source].daily)

    orb = pd.read_csv(ROOT / mnq_config["sleeves"]["orb_vwap_atr"]["source"])
    orb["session_date"] = pd.to_datetime(orb["session_date"])
    result["mnq_orb_vwap_atr"] = _daily(
        orb.groupby("session_date", as_index=False)["net_pnl_usd"].sum(), "net_pnl_usd"
    )

    for sleeve, path in CLUSTER_DAILY.items():
        result[sleeve] = _daily(pd.read_csv(path))

    if set(result) != set(LABELS):
        raise RuntimeError(f"Unexpected sleeve set: {sorted(result)}")
    return result


def plot_equity(sleeve: str, daily: pd.DataFrame, number: int) -> None:
    split = pd.Timestamp(OOS_START[sleeve])
    oos = daily[daily["session_date"].ge(split)].copy()
    if oos.empty:
        raise RuntimeError(f"No OOS observations for {sleeve}")

    fs_equity = daily["daily_pnl_usd"].cumsum()
    oos_equity = oos["daily_pnl_usd"].cumsum()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    axes[0].plot(daily["session_date"], fs_equity, color="#1769aa", linewidth=1.8, drawstyle="steps-post")
    axes[0].axvline(split, color="#c45a4a", linestyle="--", linewidth=1.4, label=f"Début : {split.date():%d/%m/%Y}")
    axes[0].axvspan(split, daily["session_date"].max(), color="#dbe8f4", alpha=.45)
    axes[0].set_title("FS — Full Sample", weight="bold")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(oos["session_date"], oos_equity, color="#25855a", linewidth=1.8, drawstyle="steps-post")
    axes[1].set_title(OOS_STATUS[sleeve], weight="bold")
    axes[1].text(.02, .04, f"PnL de période : {oos['daily_pnl_usd'].sum():,.2f} USD".replace(",", " "),
                 transform=axes[1].transAxes, color="#102b4e",
                 bbox={"facecolor": "white", "edgecolor": "#c5d0dc", "alpha": .9, "pad": 3})

    for ax in axes:
        ax.axhline(0, color="#7b8794", linewidth=.8)
        ax.set_ylabel("PnL cumulé (USD)")
        ax.set_xlabel("Date")
        ax.grid(axis="y", alpha=.22)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.suptitle(f"Equity curves nette — {LABELS[sleeve]}", weight="bold")
    fig.text(.99, .012, "FS inclut la période de droite ; chaque courbe démarre à 0 USD", ha="right", fontsize=8, color="#7b8794")
    fig.tight_layout(rect=(0, .04, 1, .92))
    fig.savefig(ASSET_DIR / f"{number:02d}_equity_curve.png", dpi=190, facecolor="white")
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    all_daily = load_full_sample_daily()
    for number, sleeve in enumerate(LABELS, 1):
        plot_equity(sleeve, all_daily[sleeve], number)
        frame = all_daily[sleeve]
        print(number, sleeve, frame["session_date"].min().date(), frame["session_date"].max().date(), len(frame))


if __name__ == "__main__":
    main()
