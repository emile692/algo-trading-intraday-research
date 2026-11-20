"""S01 scale-invariant 2x2 campaign (spec: FROZEN_SPECIFICATION.json, hash in SPEC_LOCK.json).

    python -m research.s01_scale_invariant_20261118.run_campaign selftest   # T1-T2 on A and B (already read)
    python -m research.s01_scale_invariant_20261118.run_campaign run        # T3, stage 0, stage 1, stage 2 (once)

Reuses the engine functions of research/s01_atr_price_normalized_20261115/run_campaign.py unchanged.
Deterministic: no random draw.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CAMPAIGN = Path(__file__).resolve().parent
ROOT = CAMPAIGN.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_20261115.stats import dsr, lo_sharpe, psr  # noqa: E402
from research.s01_atr_price_normalized_20261115 import run_campaign as old  # noqa: E402

SPEC_PATH, LOCK_PATH, RES = CAMPAIGN / "FROZEN_SPECIFICATION.json", CAMPAIGN / "SPEC_LOCK.json", CAMPAIGN / "results"
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
OLD_RES = ROOT / "research" / "s01_atr_price_normalized_20261115" / "results"
TH_N = OLD_RES / "thresholds_is_atr_over_price.csv"
IS_END = old.W["IS"][1]
# UNBOUNDED = 100 % of the 50 000 USD account, the largest budget the engine accepts; no MNQ contract risks that much,
# so no signal is skipped for sizing.  (Attempt 1 used 1e9 and stopped in the calibration before any metric: run_log_attempt1.txt.)
BUDGET, CAP, UNBOUNDED = 250.0, 500.0, 50_000.0
IS_Q = [f"{y}Q{q}" for y in range(2019, 2024) for q in (1, 2, 3, 4) if f"{y}Q{q}" >= "2019Q3"]
LATE_Q = ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2", "2026Q3"]
YEARS_IS = ["2019", "2020", "2021", "2022", "2023"]
YEARS_PRE = [str(y) for y in range(2010, 2019)]


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def check_lock() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if old.sha256(SPEC_PATH) != lock["spec_sha256"]:
        raise SystemExit("FROZEN_SPECIFICATION.json changed after the freeze")


def _date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.date


def is_only(final: pd.DataFrame) -> pd.DataFrame:
    return final[_date(final["session_date"]) <= IS_END]


def pct(t: pd.DataFrame) -> pd.Series:
    return (t["entry_price"] - t["stop_price"]) / t["entry_price"]


def calibrate_x(data, final: pd.DataFrame) -> dict:
    """PnL-free: unbounded budget, only entry/stop/risk kept, IS signals only."""
    t = old.backtest(data, is_only(final), UNBOUNDED)[["session_date", "entry_price", "stop_price", "risk_per_contract_usd"]]
    s = float((t["risk_per_contract_usd"] > BUDGET).mean())
    x = float(pct(t).quantile(1.0 - s))
    return {"X": x, "skip_share_usd_rule": s, "is_signals_with_risk": int(len(t)),
            "skipped_usd_rule": int((t["risk_per_contract_usd"] > BUDGET).sum()), "skipped_pct_rule": int((pct(t) > x).sum()),
            "cap_binding_is": int((t["risk_per_contract_usd"] > CAP).sum())}


def pct_arm(data, final: pd.DataFrame, x: float) -> pd.DataFrame:
    base = old.backtest(data, final, BUDGET)
    wide = old.backtest(data, final, CAP)
    add = wide[~wide["session_date"].isin(set(base["session_date"]))].copy()
    if len(add) and not ((add["quantity"] == 1).all() and add["risk_per_contract_usd"].between(BUDGET, CAP, inclusive="right").all()):
        raise RuntimeError("floor: an added trade does not have quantity 1 with risk in (250, 500] USD")
    u = pd.concat([base, add], ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    return u[pct(u) <= x].reset_index(drop=True)


def is_metrics(t: pd.DataFrame, cal: list) -> dict:
    ti = old.in_window(t, "IS")
    d = old.daily(ti, cal)
    return old.metrics(ti, d) | {"by_year": old.by_year(d)}, d


def selftest() -> dict:
    data = old.Data(old.MNQ)
    th_n = pd.read_csv(TH_N)
    recal = old.calibrate(data, "atrn").merge(th_n, on="model", suffixes=("_r", "_f"))
    gap = float(np.max(np.abs(np.r_[recal.low_threshold_is_r - recal.low_threshold_is_f,
                                     recal.high_threshold_is_r - recal.high_threshold_is_f])))
    t2 = {"models": len(recal), "max_abs_gap": gap, "passed": len(recal) == 18 and gap <= 1e-12}
    prev = json.loads((OLD_RES / "stage1.json").read_text(encoding="utf-8"))
    cal = data.calendar("IS")
    t1 = {}
    for arm in ("A", "B"):
        m, _ = is_metrics(old.arm_trades(data, old.kept(data, th_n, "atrn"), arm), cal)
        p = prev[arm]
        t1[arm] = {"trades": [m["trades"], p["trades"]], "net_usd": [m["net_usd"], p["net_usd"]], "sharpe": [m["sharpe"], p["sharpe"]],
                   "passed": m["trades"] == p["trades"] and abs(m["net_usd"] - p["net_usd"]) < 1e-6 and abs(m["sharpe"] - p["sharpe"]) < 1e-9}
    out = {"T1": t1, "T2": t2, "all_passed": t2["passed"] and all(v["passed"] for v in t1.values()), "run_at": now(),
           "note": "arms A and B only (already read on 2026-11-15); no C1/C2 computation"}
    old.write_json(RES / "selftest.json", out)
    print(json.dumps(out, indent=2, default=old._j))
    return out


def run() -> dict:
    check_lock()
    for f in ("stage1.json", "stage2.json"):
        if (RES / f).exists():
            raise SystemExit(f"{f} already written; one run per campaign (no_rescue_rule)")
    if not json.loads((RES / "selftest.json").read_text(encoding="utf-8"))["all_passed"]:
        raise SystemExit("technical gates T1-T2 did not pass")

    data = old.Data(old.MNQ)
    th = {"points": pd.read_csv(old.TH_REF), "atrn": pd.read_csv(TH_N)}
    final = {"points": old.kept(data, th["points"], "atr"), "atrn": old.kept(data, th["atrn"], "atrn")}

    # T3 + calibration (PnL-free).
    calib = {"C1": calibrate_x(data, final["points"]), "C2": calibrate_x(data, final["atrn"])}
    trades = {"REF": old.arm_trades(data, final["points"], "A"), "A": old.arm_trades(data, final["atrn"], "A"),
              "C1": pct_arm(data, final["points"], calib["C1"]["X"]), "C2": pct_arm(data, final["atrn"], calib["C2"]["X"])}
    t3 = {arm: {**c, "passed": abs(c["skipped_pct_rule"] - c["skipped_usd_rule"]) <= 1} for arm, c in calib.items()}
    old.write_json(RES / "calibration_T3.json", t3)
    if not all(v["passed"] for v in t3.values()):
        raise SystemExit("T3 failed: skip shares differ")

    # Stage 0: frequency only.
    counts = pd.DataFrame({a: old.quarterly_counts(t[["session_date", "quantity"]]) for a, t in trades.items()}).fillna(0).astype(int)
    counts.index = counts.index.astype(str)
    counts.to_csv(RES / "stage0_trades_per_quarter.csv")
    g0 = {}
    for a in trades:
        c = counts[a].reindex(IS_Q + LATE_Q).fillna(0)
        r = float(c[LATE_Q].median() / c[IS_Q].median())
        g0[a] = {"is_median": float(c[IS_Q].median()), "late_median": float(c[LATE_Q].median()), "ratio": r, "passed": r >= 0.60,
                 "2026Q1_Q3": [int(c[q]) for q in ("2026Q1", "2026Q2", "2026Q3")]}
    old.write_json(RES / "stage0.json", g0)

    # Stage 1: IS.
    cal = data.calendar("IS")
    s1, dd = {}, {}
    for a, t in trades.items():
        s1[a], dd[a] = is_metrics(t, cal)
    ref, c2 = s1["REF"], s1["C2"]
    yrs = [c2["by_year"].get(y, 0.0) for y in YEARS_IS]
    c2["gates"] = {"G0": g0["C2"]["passed"], "G1": c2["trades"] >= 0.75 * ref["trades"], "G2": c2["net_usd"] > 0 and c2["t_hac"] >= 2.0,
                   "G3": c2["sharpe"] >= 0.75 * ref["sharpe"], "G4": sum(v > 0 for v in yrs) >= 3}
    c2["passed_all"] = all(c2["gates"].values())
    prev = json.loads((OLD_RES / "stage1.json").read_text(encoding="utf-8"))
    srs = [s1["A"]["sharpe"], prev["B"]["sharpe"], s1["C1"]["sharpe"], s1["C2"]["sharpe"]]
    x = dd["C2"].to_numpy()
    c2["psr_0"] = float(psr(x, 0.0))
    c2["dsr_4_trials"], c2["dsr_sr0"] = (float(v) for v in dsr(x, 4, float(np.var(srs, ddof=1))))
    attribution = {k: {"M1_bands": (s1["A"][k] - s1["REF"][k] + s1["C2"][k] - s1["C1"][k]) / 2,
                       "M2_skip": (s1["C1"][k] - s1["REF"][k] + s1["C2"][k] - s1["A"][k]) / 2} for k in ("net_usd", "trades", "sharpe")}
    s1["attribution_2x2"] = attribution
    s1["run_at"] = now()
    old.write_json(RES / "stage1.json", s1)
    pd.DataFrame(dd).to_csv(RES / "stage1_is_daily.csv", index_label="session_date")
    print("T3:", json.dumps({a: {k: v[k] for k in ("X", "skip_share_usd_rule", "passed")} for a, v in t3.items()}))
    print("stage 0:", json.dumps({a: (round(v["ratio"], 3), v["2026Q1_Q3"]) for a, v in g0.items()}))
    print("stage 1 C2 gates:", json.dumps(c2["gates"]))
    if not c2["passed_all"]:
        old.write_json(RES / "stage2.json", {"verdict": "CLOSED_IS", "read": False, "written_at": now()})
        return {"verdict": "CLOSED_IS"}

    # Stage 2 (one shot).
    p_ref = float(old.in_window(trades["C2"], "IS")["entry_price"].median())
    s2 = {"p_ref": p_ref, "read_at": now()}
    for win in ("VAL", "EXT"):
        c = data.calendar(win)
        s2[win] = {}
        for a in ("C2", "REF"):
            t = old.in_window(trades[a], win)
            s2[win][a] = old.metrics(t, old.daily(t, c)) | {"by_year": old.by_year(old.daily(t, c))}
            t.to_csv(RES / f"stage2_{win.lower()}_trades_{a}.csv", index=False)
        s2[win]["diff_net_usd"] = s2[win]["C2"]["net_usd"] - s2[win]["REF"]["net_usd"]
    nofilter_x = calibrate_x(data, old.kept(data, None, None))["X"]
    del data
    nq = old.Data(old.truncated(old.NQ, *old.SPEC["data"]["NQ_PRE"]["used_range"], name="NQ_pre"))
    cal = nq.calendar("PRE")
    fin = {"points": old.kept(nq, th["points"], "atr"), "atrn": old.kept(nq, th["atrn"], "atrn"), "none": old.kept(nq, None, None)}
    pre_trades = {"C2": pct_arm(nq, fin["atrn"], calib["C2"]["X"]), "REF": old.arm_trades(nq, fin["points"], "A"),
                  "A": old.arm_trades(nq, fin["atrn"], "A"), "C1": pct_arm(nq, fin["points"], calib["C1"]["X"]),
                  "NOFILTER_pct": pct_arm(nq, fin["none"], nofilter_x)}
    raw = old.backtest(nq, fin["atrn"], UNBOUNDED)[["entry_price", "stop_price", "risk_per_contract_usd"]]
    s2["PRE_cuts_C2"] = {"signals": int(len(raw)), "cut_by_pct_rule": int((pct(raw) > calib["C2"]["X"]).sum()),
                         "cut_by_usd_cap": int((raw["risk_per_contract_usd"] > CAP).sum()),
                         "would_be_cut_by_250_usd_rule": int((raw["risk_per_contract_usd"] > BUDGET).sum())}
    s2["PRE"] = {}
    for a, t in pre_trades.items():
        t = old.r_units(old.in_window(t, "PRE"), p_ref)
        t.to_csv(RES / f"stage2_pre_trades_{a}.csv", index=False)
        dg, dn, du = old.daily(t, cal, "gross_R"), old.daily(t, cal, "net_scaled_R"), old.daily(t, cal, "net_pnl_usd")
        lg, ln = lo_sharpe(dg.to_numpy()), lo_sharpe(dn.to_numpy())
        s2["PRE"][a] = {"trades": int(len(t)), "gross_R_total": float(dg.sum()), "gross_t_hac": float(lg["t_hac"]),
                        "net_scaled_R_total": float(dn.sum()), "net_scaled_t_hac": float(ln["t_hac"]), "net_scaled_sharpe": float(ln["sharpe"]),
                        "net_scaled_R_by_year": old.by_year(dn), "raw_net_usd_fixed_mnq_costs": float(du.sum()),
                        "trades_by_year": {str(k): int(v) for k, v in pd.to_datetime(pd.Series(t["session_date"])).dt.year.value_counts().sort_index().items()}}
    p = s2["PRE"]["C2"]
    yrs = [p["net_scaled_R_by_year"].get(y, 0.0) for y in YEARS_PRE]
    g = {"P1": p["gross_R_total"] > 0 and p["gross_t_hac"] >= 2.0, "P2": p["net_scaled_R_total"] > 0 and p["net_scaled_t_hac"] >= 1.65,
         "P3": sum(v > 0 for v in yrs) >= 5, "V1": s2["VAL"]["C2"]["net_usd"] > 0}
    s2["gates"] = g
    s2["verdict"] = ("CANDIDATE_SHADOW" if all(g.values()) else
                     "REJECTED_PRE" if not (g["P1"] and g["P2"] and g["P3"]) else "REJECTED_VAL")
    old.write_json(RES / "stage2.json", s2)
    print("stage 2:", json.dumps(g), s2["verdict"])
    return s2


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
