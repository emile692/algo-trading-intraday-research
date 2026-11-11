"""Build the frozen Step-1 documentary tables from audited outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "research" / "signal_sheets_audit_20261014"
OUT = ROOT / "research" / "signal_sheets_step1_20261029"
CONFIG = OUT / "step1_config.json"
ORB_DIR = ROOT / "research" / "signal_sheet_01_atr18"
ORB_TRADES = ORB_DIR / "trades.csv"
ORB_DAILY = ORB_DIR / "daily_results.csv"


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    metrics = pd.read_csv(AUDIT / "sleeve_period_reproduction.csv")
    # S01 was subsequently frozen as the 18-submodel ATR vote ensemble.  Replace
    # the superseded Step-0 ORB rows from its own reproducible daily/trade files.
    metrics = metrics.loc[~metrics["sleeve"].isin(["S01", "mnq_orb_vwap_atr"])].copy()
    orb_daily = pd.read_csv(ORB_DAILY, parse_dates=["session_date"])
    orb_trades = pd.read_csv(ORB_TRADES, parse_dates=["session_date"])
    orb_rows = []
    for period, start, end in (
        ("IS", "2019-05-05", "2024-02-27"),
        ("validation_reused", "2024-02-28", "2026-03-20"),
        ("total", "2019-05-05", "2026-03-20"),
    ):
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        daily = orb_daily.loc[orb_daily["session_date"].between(lo, hi), "daily_pnl_usd"].astype(float)
        trades_period = orb_trades.loc[orb_trades["session_date"].between(lo, hi)]
        sharpe = daily.mean() / daily.std(ddof=1) * math.sqrt(252) if daily.std(ddof=1) else float("nan")
        orb_rows.append({
            "sleeve": "S01", "period": period, "start": start, "end": end,
            "sessions": len(daily), "trades": len(trades_period),
            "net_pnl_usd": daily.sum(), "sharpe_annualized": sharpe,
            "max_drawdown_usd": (daily.cumsum() - daily.cumsum().cummax()).min(),
            "source": "research/signal_sheet_01_atr18/daily_results.csv + trades.csv",
        })
    metrics = pd.concat([pd.DataFrame(orb_rows), metrics], ignore_index=True)
    metrics["years_252"] = metrics["sessions"] / float(config["canonical_sharpe"]["annualization_sessions"])
    metrics["t_stat_approx"] = metrics["sharpe_annualized"] * metrics["years_252"].pow(0.5)
    metrics.to_csv(OUT / "significance_by_sleeve_period.csv", index=False)

    trades = orb_trades
    rows = []
    for label, bounds in config["s01_quantity_distribution"]["periods"].items():
        start, end = map(pd.Timestamp, bounds)
        x = trades.loc[trades["session_date"].between(start, end, inclusive="both"), "quantity"].astype(float)
        if x.empty:
            raise RuntimeError(f"No S01 trades for {label}")
        rows.append(
            {
                "period": label,
                "start": bounds[0],
                "end": bounds[1],
                "trades": int(len(x)),
                "quantity_min": int(x.min()),
                "quantity_median": float(x.median()),
                "quantity_mean": float(x.mean()),
                "quantity_max": int(x.max()),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "s01_quantity_distribution.csv", index=False)

    trace = []
    for row in metrics.itertuples(index=False):
        trace.append(
            {
                "figure": row.t_stat_approx,
                "section": f"{row.sleeve} / {row.period}",
                "metric": "t_stat_approx",
                "source_file": "research/signal_sheets_audit_20261014/sleeve_period_reproduction.csv",
                "function": "build_signal_sheets_step1_tables.main: sharpe * sqrt(sessions/252)",
            }
        )
    for row in rows:
        for metric in ("quantity_min", "quantity_median", "quantity_mean", "quantity_max"):
            trace.append(
                {
                    "figure": row[metric],
                    "section": f"S01 sizing / {row['period']}",
                    "metric": metric,
                    "source_file": "research/signal_sheet_01_atr18/trades.csv",
                    "function": "build_signal_sheets_step1_tables.main",
                }
            )
    pd.DataFrame(trace).to_csv(OUT / "step1_number_traceability.csv", index=False)
    print(metrics[["sleeve", "period", "trades", "net_pnl_usd", "sharpe_annualized", "t_stat_approx"]].to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
