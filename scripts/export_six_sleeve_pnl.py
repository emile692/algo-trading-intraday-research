"""Export the canonical daily PnL series for the six active sleeves."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from research.baseline_backtests.baseline_engine import (
    load_frozen_specs,
    run_baseline_backtest,
)
from research.mgc_baseline_backtests.run_mgc_baselines import (
    CONFIG_PATH as MGC_CONFIG_PATH,
    FROZEN_DIR,
    FROZEN_MANIFEST,
    adapt_spec_to_mgc,
    prepare_mgc_data,
)
from research.mgc_wide_parameter_landscape.run_mgc_wide_landscape import (
    _apply_parameters,
)


ROOT = Path(__file__).resolve().parents[1]
MNQ_DAILY = ROOT / "research" / "locked_four_sleeve_portfolio" / "daily_pnl.csv"
MGC_DATA = ROOT / "data" / "processed" / "parquet" / "MGC_v_0_1m_20150101_20260923.parquet"
DEFAULT_OUTPUT = ROOT / "research" / "six_sleeve_daily_pnl.csv"
START_DATE = date(2019, 5, 6)
END_DATE = date(2026, 3, 19)

MGC_SLEEVES = {
    "mgc_volatility_breakout_pnl_usd": {
        "mechanism_id": "volatility_breakout",
        "parameter_names": {
            "parameter_1": "daily_atr_period",
            "parameter_2": "entry_atr_multiple",
        },
        "values": (30, 1.10),
    },
    "mgc_gap_continuation_pnl_usd": {
        "mechanism_id": "gap_continuation",
        "parameter_names": {
            "parameter_1": "minimum_gap_atr_multiple",
            "parameter_2": "volume_threshold_multiple",
        },
        "values": (0.50, 0.60),
    },
}


def build_export() -> pd.DataFrame:
    mnq = pd.read_csv(MNQ_DAILY, parse_dates=["session_date"])
    mnq = mnq[
        [
            "session_date",
            "orb_vwap_atr",
            "volatility_breakout",
            "compression_breakout",
            "atr_trend_flip",
        ]
    ].rename(
        columns={
            "orb_vwap_atr": "mnq_orb_vwap_atr_pnl_usd",
            "volatility_breakout": "mnq_volatility_breakout_pnl_usd",
            "compression_breakout": "mnq_compression_breakout_pnl_usd",
            "atr_trend_flip": "mnq_atr_trend_flip_pnl_usd",
        }
    )

    source_specs, _, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    source_specs_by_id = {spec["mechanism_id"]: spec for spec in source_specs}
    mgc_config = json.loads(MGC_CONFIG_PATH.read_text(encoding="utf-8"))
    prepared, _ = prepare_mgc_data(MGC_DATA, START_DATE, END_DATE)

    result = mnq
    for column, sleeve in MGC_SLEEVES.items():
        mechanism_id = sleeve["mechanism_id"]
        spec = adapt_spec_to_mgc(source_specs_by_id[mechanism_id], mgc_config)
        spec = _apply_parameters(
            spec,
            sleeve["parameter_names"],
            sleeve["values"][0],
            sleeve["values"][1],
        )
        run = run_baseline_backtest(spec, prepared)
        violations = {
            key: value
            for key, value in run.checks.items()
            if key.endswith("violations") and value
        }
        if violations:
            raise RuntimeError(f"{mechanism_id}: {violations}")
        daily = run.daily[["session_date", "daily_pnl_usd"]].copy()
        daily["session_date"] = pd.to_datetime(daily["session_date"])
        daily = daily.rename(columns={"daily_pnl_usd": column})
        result = result.merge(daily, on="session_date", how="outer", validate="one_to_one")

    pnl_columns = [column for column in result.columns if column != "session_date"]
    result[pnl_columns] = result[pnl_columns].fillna(0.0)
    result = result.sort_values("session_date").reset_index(drop=True)
    result["session_date"] = result["session_date"].dt.strftime("%Y-%m-%d")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = build_export()
    frame.to_csv(output, index=False, float_format="%.2f")
    print(f"Wrote {len(frame)} rows and 6 PnL series to {output}")


if __name__ == "__main__":
    main()
