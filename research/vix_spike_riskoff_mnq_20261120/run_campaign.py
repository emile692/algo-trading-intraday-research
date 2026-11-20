"""VIX spike risk-off overlay on the MNQ sleeves (spec: FROZEN_SPECIFICATION.json, hash in SPEC_LOCK.json).

    python -m research.vix_spike_riskoff_mnq_20261120.run_campaign selftest   # T1-T3 (no overlay PnL)
    python -m research.vix_spike_riskoff_mnq_20261120.run_campaign run        # IS grid, gates, then VAL once

Daily-level computation (the sleeves are flat every evening), deterministic (bootstrap seed 20261120).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CAMPAIGN = Path(__file__).resolve().parent
ROOT = CAMPAIGN.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_20261115.stats import stationary_indices  # noqa: E402

SPEC_PATH, LOCK_PATH, RES = CAMPAIGN / "FROZEN_SPECIFICATION.json", CAMPAIGN / "SPEC_LOCK.json", CAMPAIGN / "results"
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
W = {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in SPEC["windows"].items()}
G = SPEC["grid"]
SLEEVES = ["S01", "S02", "S03"]
SEED = 20261120


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n",
                    encoding="utf-8")


def sharpe(x) -> float:
    x = np.asarray(x, float)
    s = x.std(ddof=1)
    return float(math.sqrt(252.0) * x.mean() / s) if s > 0 else float("nan")


def max_dd(x) -> float:
    e = np.r_[0.0, np.cumsum(np.asarray(x, float))]
    return float((e - np.maximum.accumulate(e)).min())


# ----------------------------------------------------------------------------------------------- data
def vix() -> pd.Series:
    v = pd.read_csv(ROOT / "data" / "raw" / "vix-cboe.csv")
    v.index = pd.to_datetime(v["DATE"], format="%m/%d/%Y")
    return v["CLOSE"].astype(float).sort_index()


def mnq_daily() -> pd.Series:
    p = pd.read_csv(ROOT / "audit_20261115" / "outputs" / "daily_pnl_union.csv", parse_dates=["session_date"]).set_index("session_date")
    m = p[SLEEVES].dropna()
    hist = m.sum(axis=1)
    rep = ROOT / "research" / "frozen_vote_forward_replay_20261101"
    s02 = pd.read_csv(rep / "s02_daily.csv", parse_dates=["session_date"]).set_index("session_date")["daily_pnl_usd"]
    s03 = pd.read_csv(rep / "s03_daily.csv", parse_dates=["session_date"]).set_index("session_date")["daily_pnl_usd"]
    cal = s02.index[(s02.index >= W["EXT"][0]) & (s02.index <= W["EXT"][1])]
    t01 = pd.read_csv(ROOT / "audit_20261115" / "outputs" / "04b_s01_window_trades.csv", parse_dates=["session_date"])
    s01 = t01.groupby("session_date")["net_pnl_usd"].sum().reindex(cal).fillna(0.0)
    ext = s01 + s02.reindex(cal).fillna(0.0) + s03.reindex(cal).fillna(0.0)
    return pd.concat([hist, ext]).sort_index()


def cut_mask(sessions: pd.DatetimeIndex, v: pd.Series, x: float, y: int, z: int) -> np.ndarray:
    rise = v / v.shift(y) - 1.0
    trig_dates = rise.index[rise >= x / 100.0]
    pos = np.searchsorted(sessions.values, trig_dates.values, side="right")  # first session strictly after the VIX date
    cut = np.zeros(len(sessions), bool)
    for p in pos:
        cut[p:p + z] = True
    return cut


def window(s: pd.Series, name: str) -> pd.Series:
    a, b = W[name]
    return s[(s.index >= a) & (s.index <= b)]


# ----------------------------------------------------------------------------------------------- stats
def pbo_cscv(M: np.ndarray, blocks: int = 10) -> dict:
    T, N = M.shape
    edges = np.linspace(0, T, blocks + 1).astype(int)
    idx = [np.arange(edges[b], edges[b + 1]) for b in range(blocks)]

    def metric(rows):
        x = M[rows]
        sd = x.std(axis=0, ddof=1)
        return np.where(sd > 0, x.mean(axis=0) / np.where(sd > 0, sd, 1), -np.inf)

    logits = []
    for combo in itertools.combinations(range(blocks), blocks // 2):
        is_rows = np.concatenate([idx[b] for b in combo])
        oos_rows = np.concatenate([idx[b] for b in range(blocks) if b not in combo])
        best = int(np.argmax(metric(is_rows)))
        rank = pd.Series(metric(oos_rows)).rank(method="average").to_numpy()[best]
        omega = rank / (N + 1)
        logits.append(math.log(omega / (1 - omega)))
    logits = np.array(logits)
    return {"pbo": float(np.mean(logits <= 0)), "splits": len(logits), "configs": N, "median_logit": float(np.median(logits))}


def boot_diff(base: np.ndarray, over: np.ndarray, n: int = 5000) -> dict:
    rng = np.random.default_rng(SEED)
    idx = stationary_indices(len(base), n, 20, rng)
    b, o = base[idx], over[idx]
    sb = b.mean(axis=1) / b.std(axis=1, ddof=1)
    so = o.mean(axis=1) / o.std(axis=1, ddof=1)
    d = (so - sb) * math.sqrt(252.0)
    return {"p_diff_gt_0": float((d > 0).mean()), "diff_p05": float(np.percentile(d, 5)), "diff_p50": float(np.median(d)),
            "diff_p95": float(np.percentile(d, 95)), "draws": n}


# ----------------------------------------------------------------------------------------------- commands
def selftest() -> dict:
    v = vix()
    d = pd.read_csv(ROOT / "data" / "raw" / "vix-daily.csv")
    d.index = pd.to_datetime(d.iloc[:, 0], format="%m/%d/%Y")
    common = v.index.intersection(d.index)
    t1 = {"overlap_days": int(len(common)), "max_abs_gap": float((v.loc[common] - d.loc[common].iloc[:, 4].astype(float)).abs().max())}
    t1["passed"] = t1["max_abs_gap"] <= 0.01
    p = pd.read_csv(ROOT / "audit_20261115" / "outputs" / "daily_pnl_union.csv", parse_dates=["session_date"]).set_index("session_date")
    lock = pd.read_csv(ROOT / "research" / "locked_eight_sleeve_portfolio_20261115" / "daily_pnl.csv", parse_dates=["session_date"]).set_index("session_date")
    lsum = lock[["mnq_orb_vwap_atr", "mnq_volatility_breakout", "mnq_compression_breakout"]].sum(axis=1)
    psum = p[SLEEVES].sum(axis=1, min_count=1).reindex(lsum.index).fillna(0.0)
    t2 = {"common_sessions": int(len(lsum)), "max_abs_gap": float((psum - lsum).abs().max())}
    t2["passed"] = t2["max_abs_gap"] <= 1e-6
    s = mnq_daily()
    prev = np.searchsorted(v.index.values, s.index.values, side="left") - 1
    lag = (s.index - v.index[prev]).days
    t3 = {"mnq_sessions": int(len(s)), "max_calendar_days_to_previous_vix_close": int(lag.max()), "first": str(s.index.min().date()),
          "last": str(s.index.max().date())}
    t3["passed"] = bool((prev >= 0).all() and lag.max() <= 5)
    out = {"T1": t1, "T2": t2, "T3": t3, "all_passed": t1["passed"] and t2["passed"] and t3["passed"], "run_at": now(),
           "note": "no overlay PnL computed"}
    write_json(RES / "selftest.json", out)
    print(json.dumps(out, indent=2))
    return out


def run() -> dict:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest() != lock["spec_sha256"]:
        raise SystemExit("spec changed after the freeze")
    if (RES / "is.json").exists():
        raise SystemExit("IS already run once")
    if not json.loads((RES / "selftest.json").read_text(encoding="utf-8"))["all_passed"]:
        raise SystemExit("technical gates failed")
    v, s_all = vix(), mnq_daily()
    cells = list(itertools.product(G["x_pct"], G["y_vix_days"], G["z_mnq_sessions"]))
    s = window(s_all, "IS")
    base = s.to_numpy()
    rows, mats = [], [base]
    for (x, y, z) in cells:
        cut = cut_mask(s.index, v, x, y, z)
        o = np.where(cut, 0.0, base)
        mats.append(o)
        rows.append({"x_pct": x, "y": y, "z": z, "cut_sessions": int(cut.sum()), "cut_share": float(cut.mean()),
                     "net_usd": float(o.sum()), "sharpe": sharpe(o), "max_dd": max_dd(o), "worst_day": float(o.min()),
                     "days_below_500": int((o < -500).sum()), "pnl_on_cut_days": float(base[cut].sum())})
    tab = pd.DataFrame(rows)
    b = {"net_usd": float(base.sum()), "sharpe": sharpe(base), "max_dd": max_dd(base), "worst_day": float(base.min()),
         "days_below_500": int((base < -500).sum()), "sessions": int(len(base))}
    # plateau score
    ix = {k: i for i, k in enumerate(cells)}
    axes = [G["x_pct"], G["y_vix_days"], G["z_mnq_sessions"]]
    plate = []
    for (x, y, z) in cells:
        pos = [axes[0].index(x), axes[1].index(y), axes[2].index(z)]
        neigh = []
        for d in itertools.product((-1, 0, 1), repeat=3):
            q = [pos[k] + d[k] for k in range(3)]
            if all(0 <= q[k] < len(axes[k]) for k in range(3)):
                neigh.append(tab.loc[ix[(axes[0][q[0]], axes[1][q[1]], axes[2][q[2]])], "sharpe"])
        plate.append(float(np.median(neigh)))
    tab["plateau"] = plate
    tab.to_csv(RES / "is_grid.csv", index=False)
    order = tab.sort_values(["plateau", "cut_sessions", "z", "x_pct"], ascending=[False, True, True, False])
    sel = order.iloc[0]
    key = (int(sel.x_pct), int(sel.y), int(sel.z))
    o = mats[1 + ix[key]]
    pbo = pbo_cscv(np.column_stack(mats), 10)
    boot = boot_diff(base, o)
    gates = {"G1": bool(sel.sharpe - b["sharpe"] >= 0.10 and sel.plateau > b["sharpe"]), "G2": pbo["pbo"] <= 0.40,
             "G3": boot["p_diff_gt_0"] >= 0.90, "G4": bool(sel.cut_share <= 0.20 and sel.net_usd >= 0.85 * b["net_usd"])}
    cut_dates = [str(d.date()) for d, c in zip(s.index, cut_mask(s.index, v, *key)) if c]
    res = {"baseline": b, "selected": {k: (v_.item() if hasattr(v_, "item") else v_) for k, v_ in sel.to_dict().items()},
           "pbo": pbo, "bootstrap": boot, "gates": gates, "passed": all(gates.values()), "cut_dates": cut_dates,
           "cells_beating_base": int((tab["sharpe"] > b["sharpe"]).sum()),
           "cells_beating_base_by_0_10": int((tab["sharpe"] - b["sharpe"] >= 0.10).sum()), "run_at": now()}
    write_json(RES / "is.json", res)
    print("baseline IS:", json.dumps(b))
    print("selected:", json.dumps(res["selected"]))
    print("pbo:", json.dumps(pbo), "boot:", json.dumps(boot), "gates:", json.dumps(gates))
    if not res["passed"]:
        write_json(RES / "validation.json", {"verdict": "CLOSED_IS", "read": False, "written_at": now()})
        return res
    if (RES / "validation.json").exists():
        raise SystemExit("validation already read once")
    val = {}
    for w in ("VAL", "EXT"):
        sv = window(s_all, w)
        cv = cut_mask(sv.index, v, *key)
        ov = np.where(cv, 0.0, sv.to_numpy())
        bv = sv.to_numpy()
        val[w] = {"base": {"net_usd": float(bv.sum()), "sharpe": sharpe(bv), "max_dd": max_dd(bv), "worst_day": float(bv.min()),
                           "days_below_500": int((bv < -500).sum()), "days_below_1000": int((bv < -1000).sum())},
                  "overlay": {"net_usd": float(ov.sum()), "sharpe": sharpe(ov), "max_dd": max_dd(ov), "worst_day": float(ov.min()),
                              "days_below_500": int((ov < -500).sum()), "days_below_1000": int((ov < -1000).sum()),
                              "cut_sessions": int(cv.sum()), "pnl_on_cut_days": float(bv[cv].sum())}}
    vb, vo = val["VAL"]["base"], val["VAL"]["overlay"]
    val["gates"] = {"V1": vo["sharpe"] - vb["sharpe"] >= 0, "V2": vo["net_usd"] >= 0.85 * vb["net_usd"]}
    val["verdict"] = "CANDIDATE_SHADOW" if all(val["gates"].values()) else "REJECTED_VAL"
    val["read_at"] = now()
    write_json(RES / "validation.json", val)
    print("validation:", json.dumps(val["gates"]), val["verdict"])
    return val


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
