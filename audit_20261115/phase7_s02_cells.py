"""Daily net PnL of the 42 cells of the S02 frozen cluster (phase 7, reference (b)), cached.

Same engine calls as scripts/rebuild_signal_sheet_02_cluster.py::build_cell_positions, keeping run.daily.
Data window: 2019-05-06 -> 2026-03-19 (campaign window).  Output: audit_20261115/cache/s02_cell_daily_pnl.parquet.
"""

from __future__ import annotations

import copy
import json
from itertools import product

import pandas as pd

from .common import CACHE

OUT_FILE = CACHE / "s02_cell_daily_pnl.parquet"


def main() -> pd.DataFrame:
    if OUT_FILE.exists():
        return pd.read_parquet(OUT_FILE)
    import rebuild_signal_sheet_02_cluster as r
    manifest = json.loads(r.MANIFEST_PATH.read_text(encoding="utf-8"))
    specs, _, _ = r.load_frozen_specs(r.FROZEN_DIR, r.FROZEN_MANIFEST)
    spec = next(s for s in specs if s["mechanism_id"] == "volatility_breakout")
    prepared = r.prepare_data(r.DATA_PATH, start_date=r.IS_START, end_date=r.OOS_END)
    region = manifest["parameter_region"]
    cols = {}
    for atr_period, multiple in product(region["daily_atr_period"], region["entry_atr_multiple"]):
        cand = copy.deepcopy(spec)
        cand["canonical_parameters"]["structural"]["daily_atr_period"] = int(atr_period)
        cand["canonical_parameters"]["structural"]["entry_atr_multiple"] = float(multiple)
        ctx = r.build_overlay_context(cand, prepared, 5)
        for duration in region["max_holding_minutes"]:
            cid = r.cell_id(int(atr_period), float(multiple), int(duration))
            run = r.run_exit_overlay_backtest(cand, prepared, {"family": "time_stop", "max_holding_minutes": int(duration)},
                                              5, context=ctx)
            d = run.daily.copy()
            d["session_date"] = pd.to_datetime(d["session_date"])
            cols[cid] = d.set_index("session_date")["daily_pnl_usd"]
            print(cid, flush=True)
    out = pd.DataFrame(cols).fillna(0.0)
    out.to_parquet(OUT_FILE)
    return out


if __name__ == "__main__":
    print(main().shape)
