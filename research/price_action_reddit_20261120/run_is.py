"""In-sample (2021-01-01 -> 2023-12-31) runs of the two price-action setups on MNQ.v.0 and MES.v.0.

The data is loaded with end = 2024-01-01 (exclusive): the out-of-sample period cannot be read by this script.
Outputs in outputs_is/: grid, robustness (+/-20 %), tables by year x side for the default parameters, Monte Carlo
of the trade order, trade lists.  Command: python -m research.price_action_reddit_20261120.run_is
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.price_action.data import SPECS, load_minutes, rth_daily, signal_bars  # noqa: E402
from src.price_action.execution import CostModel, ExecConfig, minutes_by_date, simulate  # noqa: E402
from src.price_action.metrics import mc_drawdown, summarize, table  # noqa: E402
from src.price_action.signals import inside_day_false_break, three_push  # noqa: E402

DATA = {"MNQ": ROOT / "data" / "downloaded" / "MNQ_v_0_1m_20190501_20261120.parquet",
        "MES": ROOT / "data" / "downloaded" / "MES_v_0_1m_20190501_20261120.parquet"}
LOAD_START, IS_START, IS_END_EXCL = "2020-11-01", "2021-01-01", "2024-01-01"
OUT = HERE / "outputs_is"
OUT.mkdir(exist_ok=True)

DEFAULT = {"A": {"tf": 5, "end": "11:00"}, "B": {"tf": 5, "end": "11:00", "n": 2}}
# Exploratory grid (descriptive only: the defaults stay frozen unless the user decides otherwise before the OOS).
# The 1- and 3-minute bars were added after the first MNQ smoke run showed 36 (A) and 9 (B) IS trades at 5 min,
# far from the >= 100 OOS trades the validation criterion needs.  Logged in research_log.md.
GRID = {"A": [{"tf": tf, "end": e} for tf in (1, 3, 5, 15) for e in ("10:30", "11:00", "11:30")],
        "B": [{"tf": tf, "end": e, "n": n} for tf in (1, 3, 5) for n in (1, 2, 3) for e in ("10:30", "11:00", "11:30")]}


def robustness(setup: str) -> list[tuple[str, dict, ExecConfig]]:
    base, cfg = DEFAULT[setup], ExecConfig()
    rows = [("default", base, cfg)]
    rows += [(f"tf={tf}", {**base, "tf": tf}, cfg) for tf in (4, 6)]
    rows += [(f"end={e}", {**base, "end": e}, cfg) for e in ("10:42", "11:18")]
    if setup == "B":
        rows += [(f"n={n}", {**base, "n": n}, cfg) for n in (1, 3)]
    rows += [(f"min_stop={m}", base, replace(cfg, min_stop_ticks=m)) for m in (6, 10)]
    rows += [(f"target={t}R", base, replace(cfg, target_r=t)) for t in (1.6, 2.4)]
    return rows


class Instrument:
    def __init__(self, sym: str):
        self.sym, self.spec = sym, SPECS[sym]
        self.df = load_minutes(DATA[sym], LOAD_START, IS_END_EXCL)
        assert self.df.index.max() < pd.Timestamp(IS_END_EXCL, tz="America/New_York")
        self.daily = rth_daily(self.df)
        self.mins = minutes_by_date(self.df)
        self._bars: dict[int, pd.DataFrame] = {}

    def bars(self, tf: int) -> pd.DataFrame:
        if tf not in self._bars:
            self._bars[tf] = signal_bars(self.df, tf)
        return self._bars[tf]

    def run(self, setup: str, p: dict, cfg: ExecConfig = ExecConfig(), costs: CostModel = CostModel()) -> pd.DataFrame:
        tick = self.spec.tick_size
        if setup == "A":
            sig = inside_day_false_break(self.bars(p["tf"]), self.daily, tick, p["end"])
        else:
            sig = three_push(self.bars(p["tf"]), tick, n=p["n"], window_end=p["end"])
        sig = sig[(sig["date"] >= IS_START) & (sig["date"] < IS_END_EXCL)]
        t = simulate(sig, self.mins, self.spec, cfg, costs)
        if t.empty:
            return t
        t["instrument"], t["year"] = self.sym, pd.to_datetime(t["date"]).dt.year
        t["side_label"] = t["side"].map({1: "long", -1: "short"})
        return t


def main() -> None:
    inst = {s: Instrument(s) for s in ("MNQ", "MES")}
    grid_rows, rob_rows, mc, base_trades = [], [], {}, []
    for setup in ("A", "B"):
        for sym, ins in inst.items():
            for p in GRID[setup]:
                grid_rows.append({"setup": setup, "instrument": sym, **p, **summarize(ins.run(setup, p))})
            for label, p, cfg in robustness(setup):
                t = ins.run(setup, p, cfg)
                rob_rows.append({"setup": setup, "instrument": sym, "variant": label, **summarize(t)})
                if label == "default":
                    base_trades.append(t.assign(setup_code=setup))
                    filled = t[t["status"] == "filled"].sort_values("signal_time")
                    mc[f"{setup}_{sym}"] = {"usd_1_contract": mc_drawdown(filled["pnl_net_usd"].to_numpy()),
                                            "r": mc_drawdown(filled["r_net"].to_numpy())}
            print(setup, sym, "done", flush=True)
    trades = pd.concat(base_trades, ignore_index=True)
    trades.to_csv(OUT / "trades_is_default.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(OUT / "grid_is.csv", index=False)
    pd.DataFrame(rob_rows).to_csv(OUT / "robustness_is.csv", index=False)
    table(trades, ["setup_code", "instrument"]).to_csv(OUT / "summary_is.csv", index=False)
    table(trades, ["setup_code", "instrument", "year", "side_label"]).to_csv(OUT / "by_year_side_is.csv", index=False)
    table(trades, ["setup_code", "instrument", "year"]).to_csv(OUT / "by_year_is.csv", index=False)
    table(trades, ["setup_code", "instrument", "side_label"]).to_csv(OUT / "by_side_is.csv", index=False)
    (OUT / "mc_is.json").write_text(json.dumps(mc, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
