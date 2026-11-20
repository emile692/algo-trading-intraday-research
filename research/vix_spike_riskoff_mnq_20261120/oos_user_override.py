"""One-shot OOS read of the IS-selected cell (x 50 %, y 10, z 10), on the user's explicit request of 2026-11-20.

This is a documented override: the campaign closed CLOSED_IS (PBO 0.90, bootstrap 0.80) and its closure note said the
VAL should not be read for this cell.  The user asked: 'adopte la meilleure config et teste-la en OOS maintenant'.
The pre-registered VAL criteria V1/V2 of FROZEN_SPECIFICATION.json are applied unchanged; EXT and tail metrics are
information.  Written once to results/oos_user_override.json; a second run is refused.

    python -m research.vix_spike_riskoff_mnq_20261120.oos_user_override
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import run_campaign as r

OUT = r.RES / "oos_user_override.json"
CELL = (50, 10, 10)


def stats(x: np.ndarray, cut: np.ndarray | None = None) -> dict:
    out = {"sessions": int(len(x)), "net_usd": float(x.sum()), "sharpe": r.sharpe(x), "max_dd": r.max_dd(x),
           "worst_day": float(x.min()), "days_below_500": int((x < -500).sum()), "days_below_1000": int((x < -1000).sum())}
    roll = pd.Series(x).rolling(15).sum()
    out["worst_15_sessions"] = float(roll.min()) if roll.notna().any() else None
    return out


def main() -> dict:
    if OUT.exists():
        raise SystemExit("OOS override already read once")
    is_res = json.loads((r.RES / "is.json").read_text(encoding="utf-8"))
    sel = is_res["selected"]
    if (int(sel["x_pct"]), int(sel["y"]), int(sel["z"])) != CELL:
        raise SystemExit("CELL is not the IS-selected cell")
    v, s_all = r.vix(), r.mnq_daily()
    res = {"label": "override after CLOSED_IS, on explicit user request of 2026-11-20; not a pre-registered verdict path",
           "cell": {"x_pct": CELL[0], "y": CELL[1], "z": CELL[2]}, "read_at": r.now()}
    for w in ("VAL", "EXT"):
        s = r.window(s_all, w)
        cut = r.cut_mask(s.index, v, *CELL)
        base, over = s.to_numpy(), np.where(cut, 0.0, s.to_numpy())
        res[w] = {"base": stats(base), "overlay": stats(over), "cut_sessions": int(cut.sum()), "cut_share": float(cut.mean()),
                  "pnl_on_cut_days": float(base[cut].sum()),
                  "cut_months": pd.Series(s.index[cut]).dt.to_period("M").astype(str).value_counts().sort_index().to_dict()}
    b, o = res["VAL"]["base"], res["VAL"]["overlay"]
    res["preregistered_criteria"] = {"V1_sharpe_not_lower": o["sharpe"] - b["sharpe"] >= 0, "V2_pnl_at_least_85pct": o["net_usd"] >= 0.85 * b["net_usd"]}
    res["outcome"] = "PASS_V1_V2" if all(res["preregistered_criteria"].values()) else "FAIL_V1_V2"
    r.write_json(OUT, res)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
