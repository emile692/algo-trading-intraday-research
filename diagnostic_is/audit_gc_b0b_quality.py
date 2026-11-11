"""Sensitivity of B0b to Databento daily dataset-quality flags."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostic_is" / "gc_b0b"
CONDITIONS = OUT / "databento_dataset_conditions.json"


def summarize(name: str, frame: pd.DataFrame, degraded: set[str]) -> dict[str, object]:
    dates = pd.to_datetime(frame.session_date).dt.strftime("%Y-%m-%d")
    affected = frame[dates.isin(degraded)]
    clean = frame[~dates.isin(degraded)]
    return {
        "version": name,
        "all_trades": len(frame),
        "trades_on_degraded_dates": len(affected),
        "all_net_usd": frame.net_pnl.sum(),
        "degraded_dates_net_usd": affected.net_pnl.sum(),
        "excluding_degraded_net_usd": clean.net_pnl.sum(),
        "all_net_R": frame.net_R.sum(),
        "degraded_dates_net_R": affected.net_R.sum(),
        "excluding_degraded_net_R": clean.net_R.sum(),
    }


def main() -> None:
    conditions = json.loads(CONDITIONS.read_text(encoding="utf-8"))
    degraded = {
        row["date"] for row in conditions if str(row.get("condition", "")).lower() == "degraded"
    }
    original = pd.read_csv(ROOT / "diagnostic_is" / "risk_tables" / "signal_b_macro_risk_trades.csv")
    gc = pd.read_csv(OUT / "B0b_gc_volume_mgc_trades.csv")
    result = pd.DataFrame([
        summarize("MGC_volume_canonical", original, degraded),
        summarize("GC_v0_v9_volume", gc, degraded),
    ])
    result.to_csv(OUT / "B0b_dataset_quality_sensitivity.csv", index=False)
    print(result.to_string(index=False))
    print(f"Degraded GLBX.MDP3 dates in locked window: {len(degraded)}")


if __name__ == "__main__":
    main()
