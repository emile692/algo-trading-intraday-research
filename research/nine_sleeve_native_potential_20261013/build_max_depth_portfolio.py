"""Build the longest available stitched equity view for the active sleeve family.

The locked nine-sleeve daily ledger is authoritative on its common window.
Before that window, only the six MNQ/MGC histories are available.  After it,
only the three MBT vote ensembles are available.  The output is therefore a
diagnostic availability view, not a constant-composition backtest.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIGURES = OUT / "figures"

COMMON_START = pd.Timestamp("2024-01-02")
ALL_NINE_START = pd.Timestamp("2024-01-03")
COMMON_END = pd.Timestamp("2026-03-19")


def main() -> None:
    locked = pd.read_csv(
        ROOT / "research/locked_nine_sleeve_portfolio/daily_pnl.csv",
        parse_dates=["session_date"],
    ).set_index("session_date")

    resized = pd.read_csv(
        OUT / "resized_trade_ledger.csv", parse_dates=["session_date"]
    )
    fixed = resized[
        resized["variant"].eq("fixed_1")
        & resized["sleeve"].isin(
            [
                "mnq_volatility_breakout",
                "mnq_compression_breakout",
                "mnq_atr_trend_flip",
                "mgc_volatility_breakout",
                "mgc_gap_continuation",
            ]
        )
    ]
    pre_daily = fixed.groupby("session_date")["net_pnl_nominal_usd"].sum()

    # S01 is a copy of its frozen executable sleeve: USD 250 risk budget with
    # integer quantity, not the q1 normalization used by the sizing study.
    orb = pd.read_csv(
        ROOT / "export/mnq_orb_vwap_atr_retained/trades.csv",
        parse_dates=["session_date"],
    )
    pre_daily = pre_daily.add(
        orb.groupby("session_date")["net_pnl_usd"].sum(), fill_value=0.0
    )
    pre_daily = pre_daily[pre_daily.index < COMMON_START]

    mbt = pd.read_csv(
        ROOT
        / "research/mbt_three_sleeve_freeze_20261004/vote_ensemble/daily_ensemble_results.csv",
        parse_dates=["session_date"],
    )
    active_mbt = {
        "rolling_range_breakout",
        "trend_pullback_ma_structure",
        "volatility_breakout",
    }
    post_daily = (
        mbt[
            mbt["cost_scenario"].eq("nominal")
            & mbt["sleeve"].isin(active_mbt)
        ]
        .groupby("session_date")["daily_pnl_usd"]
        .sum()
    )
    post_daily = post_daily[post_daily.index > COMMON_END]

    start = min(pre_daily.index.min(), locked.index.min())
    end = max(post_daily.index.max(), locked.index.max())
    index = pd.bdate_range(start, end)
    daily = pd.DataFrame(index=index)
    daily.index.name = "session_date"
    daily["portfolio_pnl_usd"] = 0.0
    daily.loc[pre_daily.index, "portfolio_pnl_usd"] = pre_daily
    daily.loc[locked.index, "portfolio_pnl_usd"] = locked["portfolio_pnl_usd"]
    daily.loc[post_daily.index, "portfolio_pnl_usd"] = post_daily
    daily["composition"] = "six_sleeves_only"
    daily.loc[ALL_NINE_START:COMMON_END, "composition"] = "all_nine_sleeves"
    daily.loc[COMMON_END + pd.Timedelta(days=1):, "composition"] = "three_mbt_only"
    daily["cumulative_pnl_usd"] = daily["portfolio_pnl_usd"].cumsum()
    daily["drawdown_usd"] = (
        daily["cumulative_pnl_usd"] - daily["cumulative_pnl_usd"].cummax().clip(lower=0)
    )
    daily.reset_index().to_csv(OUT / "max_depth_portfolio_daily.csv", index=False)

    plt.rcParams.update({"font.size": 10})
    fig, (ax, dd_ax) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08},
    )
    ax.axvspan(start, ALL_NINE_START, color="#e5e7eb", alpha=0.65)
    ax.axvspan(ALL_NINE_START, COMMON_END, color="#dbeafe", alpha=0.65)
    ax.axvspan(COMMON_END, end, color="#fef3c7", alpha=0.70)
    ax.plot(daily.index, daily["cumulative_pnl_usd"], color="#0f2f57", lw=2.0)
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_ylabel("PnL net cumule (USD)")
    ax.set_title("Equity en profondeur maximale — composition variable selon disponibilite")
    ax.grid(alpha=0.20)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))

    labels = [
        (pd.Timestamp("2021-08-01"), "6 sleeves MNQ/MGC\n(historique disponible)"),
        (pd.Timestamp("2025-02-01"), "9 sleeves simultanees\n(reference verrouillee)"),
        (pd.Timestamp("2026-06-25"), "3 sleeves MBT seulement\n(extension disponible)"),
    ]
    top = daily["cumulative_pnl_usd"].max()
    for x, label in labels:
        ax.text(x, top * 0.96, label, ha="center", va="top", fontsize=9)

    dd_ax.fill_between(
        daily.index, daily["drawdown_usd"].to_numpy(), 0,
        color="#b91c1c", alpha=0.28,
    )
    dd_ax.plot(daily.index, daily["drawdown_usd"], color="#991b1b", lw=0.9)
    dd_ax.set_ylabel("Drawdown")
    dd_ax.set_xlabel(
        "Mai 2019 – octobre 2026 | zones grise/jaune = book incomplet; bleu = 9 sleeves"
    )
    dd_ax.grid(alpha=0.20)
    dd_ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "portfolio_equity_max_depth.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(
        daily.groupby("composition", sort=False)["portfolio_pnl_usd"]
        .agg(["count", "sum"])
        .to_string()
    )
    print(f"start={start.date()} end={end.date()}")
    print(f"stitched_pnl={daily['portfolio_pnl_usd'].sum():.2f}")
    print(f"stitched_max_dd={daily['drawdown_usd'].min():.2f}")


if __name__ == "__main__":
    main()
