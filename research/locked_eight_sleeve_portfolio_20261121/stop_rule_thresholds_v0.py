"""Stop-rule thresholds (R0, R2 curve, R3) of the MNQ sleeves and of Book 8, recomputed on the MNQ v.0 histories.

User decision of 2026-11-21: recompute the stop-rule thresholds on v.0.  Same method and seed as
audit_20261115/phase5_stop_rules.py (stationary bootstrap, mean block 20, 10 000 reference paths, seed 20261115 + 5,
random draws consumed in the same order, including the scenario simulations).  Check first: run on the c.0
histories, the loop must reproduce the published audit thresholds exactly.  MGC/MBT thresholds keep their
registered values (their histories do not change); the values a v.0 run would give them are reported, not used.
R0 = the 126-session historical minimum of research/locked_eight_sleeve_portfolio_20261121/sleeve_stop_rule_thresholds.csv.
Command: python -m research.locked_eight_sleeve_portfolio_20261121.stop_rule_thresholds_v0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_20261115 import phase5_stop_rules as P5  # noqa: E402
from audit_20261115.common import ACTIVE8, CODE_OF, CODES, INSTRUMENT, OUT as AUDIT_OUT, SEED, sleeve_dir  # noqa: E402
from audit_20261115.stats import stationary_indices  # noqa: E402

V0_DIRS = {"S01": "signal_sheet_01_atr18_v0", "S02": "signal_sheet_02_cluster_v0", "S03": "signal_sheet_03_cluster_v0",
           "S04": "signal_sheet_04_cluster_v0"}
MNQ_CODES = ["S01", "S02", "S03", "S04"]


def daily(code: str, v0: bool) -> pd.Series:
    d = ROOT / "research" / V0_DIRS[code] if v0 and code in V0_DIRS else sleeve_dir(code)
    f = pd.read_csv(d / "daily_results.csv", parse_dates=["session_date"])
    return f.groupby("session_date")["daily_pnl_usd"].sum().astype(float)


def histories(v0: bool) -> dict[str, pd.Series]:
    cals: dict[str, set] = {"MNQ": set(), "MGC": set(), "MBT": set()}
    for c in CODES[1:]:
        cals[INSTRUMENT[c]] |= set(daily(c, v0).index)
    cals = {k: pd.DatetimeIndex(sorted(v)) for k, v in cals.items()}
    union = pd.DatetimeIndex(sorted(set().union(*[set(v) for v in cals.values()])))
    out = {}
    for c in CODES:
        inst = INSTRUMENT[c]
        s = daily(c, v0).reindex(cals[inst]).fillna(0.0)
        out[c] = s[(s.index >= P5.HIST_START[inst]) & (s.index <= P5.HIST_END[inst])]
    u = union[(union >= "2021-05-03") & (union <= "2026-03-19")]
    out["Book 8"] = sum(daily(c, v0).reindex(u).fillna(0.0) for c in ACTIVE8)
    return out


def thresholds(hist: dict[str, pd.Series]) -> tuple[dict, dict]:
    """The phase-5 loop, keeping only what the thresholds need but consuming the random draws identically."""
    rng = np.random.default_rng(SEED + 5)
    detail, curves = {}, {}
    for name, s in hist.items():
        x = s.to_numpy(float)
        sd = x.std(ddof=1)
        ref = x[stationary_indices(len(x), P5.N_REF, P5.BLOCK, rng, length=P5.LEN)]
        dd = P5.running_maxdd(ref)
        cum = np.cumsum(ref, axis=1)
        p01 = np.percentile(dd, 1, axis=0)
        detail[name] = {"history": [str(s.index.min().date()), str(s.index.max().date()), len(s)],
                        "R1_dd_p05_250": float(np.percentile(dd, 5, axis=0)[249]), "R2_dd_p01_250": float(p01[249]),
                        "R3_pnl120_p05": float(np.percentile(cum[:, 119], 5))}
        curves[name] = np.round(p01, 2)
        for base in (x, x - x.mean(), x - x.mean() - 0.5 * sd / np.sqrt(252.0)):
            stationary_indices(len(base), P5.N_SIM, P5.BLOCK, rng, length=P5.LEN)  # same consumption as phase 5
    return detail, curves


def main() -> None:
    published = json.loads((AUDIT_OUT / "05_thresholds.json").read_text(encoding="utf-8"))
    published_curve = pd.read_csv(AUDIT_OUT / "05_r2_threshold_curve.csv").set_index("h")
    c0_hist = histories(v0=False)
    ref_hist = P5.histories()
    for k in ref_hist:
        if not np.allclose(c0_hist[k].to_numpy(), ref_hist[k].to_numpy(), atol=1e-9) or not c0_hist[k].index.equals(ref_hist[k].index):
            raise RuntimeError(f"c.0 history of {k} not reproduced")
    c0_detail, _ = thresholds(c0_hist)
    for k, v in c0_detail.items():
        for key in ("R2_dd_p01_250", "R3_pnl120_p05"):
            if abs(v[key] - published[k][key]) > 1e-9:
                raise RuntimeError(f"published {key} of {k} not reproduced: {v[key]} vs {published[k][key]}")

    v0_detail, v0_curves = thresholds(histories(v0=True))
    r0 = pd.read_csv(HERE / "sleeve_stop_rule_thresholds.csv").set_index("sleeve")["threshold_usd"]
    rows, curve = [], {"h": np.arange(1, P5.LEN + 1)}
    for name in [*CODES, "Book 8"]:
        use_v0 = name in MNQ_CODES or name == "Book 8"
        src = v0_detail[name] if use_v0 else published[name]
        curve[name] = v0_curves[name] if use_v0 else published_curve[name].to_numpy()
        rows.append({"series": name, "sleeve": {c: s for s, c in CODE_OF.items()}.get(name, "book_8"),
                     "source": "v.0 recomputed" if use_v0 else "registered (audit_20261115, unchanged data)",
                     "R0_126_usd": float(r0[{c: s for s, c in CODE_OF.items()}[name]]) if name in CODES else None,
                     "R2_dd_p01_250_usd": round(src["R2_dd_p01_250"], 2), "R3_pnl120_p05_usd": round(src["R3_pnl120_p05"], 2),
                     "c0_R2_dd_p01_250_usd": round(published[name]["R2_dd_p01_250"], 2),
                     "c0_R3_pnl120_p05_usd": round(published[name]["R3_pnl120_p05"], 2),
                     "v0_run_R2_dd_p01_250_usd": round(v0_detail[name]["R2_dd_p01_250"], 2),
                     "v0_run_R3_pnl120_p05_usd": round(v0_detail[name]["R3_pnl120_p05"], 2),
                     "history": v0_detail[name]["history"] if use_v0 else published[name]["history"]})
    pd.DataFrame(rows).to_csv(HERE / "stop_rule_thresholds_v0.csv", index=False)
    pd.DataFrame(curve).to_csv(HERE / "r2_threshold_curve_v0.csv", index=False)
    print(pd.DataFrame(rows).drop(columns=["history"]).to_string(index=False))


if __name__ == "__main__":
    main()
