"""S01 with ATR / price bands: campaign runner (spec: FROZEN_SPECIFICATION.json, hash in SPEC_LOCK.json).

Commands (from the repository root):
    python -m research.s01_atr_price_normalized_20261115.run_campaign selftest   # T1-T3, reference only (pre-freeze)
    python -m research.s01_atr_price_normalized_20261115.run_campaign run        # stage 0, 1, selection, stage 2 (once)

Signal logic is the one of scripts/rebuild_signal_sheet_01_atr18.py (imported, unchanged).  Only the column fed to
the 18-model vote changes (atr_p -> atr_p / close) and, for arm B, the sizing floor.  Deterministic: no random draw.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CAMPAIGN = Path(__file__).resolve().parent
ROOT = CAMPAIGN.parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audit_20261115.phase0_engine import _norm  # noqa: E402
from audit_20261115.stats import dsr, lo_sharpe, psr  # noqa: E402

SPEC_PATH = CAMPAIGN / "FROZEN_SPECIFICATION.json"
LOCK_PATH = CAMPAIGN / "SPEC_LOCK.json"
RES = CAMPAIGN / "results"
CACHE = CAMPAIGN / "cache"
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
W = {k: (pd.Timestamp(a).date(), pd.Timestamp(b).date()) for k, (a, b) in SPEC["windows"].items()}
MNQ = ROOT / SPEC["data"]["MNQ"]["path"]
NQ = ROOT / SPEC["data"]["NQ_PRE"]["path"]
TH_REF = ROOT / SPEC["data"]["reference_thresholds"]["path"]
TR_REF = ROOT / SPEC["data"]["reference_trades"]["path"]
BUDGET = 250.0
CAP = 500.0
COST_PTS = 1.75
SLIP_BACK_PTS = 0.5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_j) + "\n", encoding="utf-8")


def _j(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp, dt.date)):
        return v.isoformat()
    return str(v)


def check_lock() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if sha256(SPEC_PATH) != lock["spec_sha256"]:
        raise SystemExit("FROZEN_SPECIFICATION.json changed after the freeze")
    for key in ("MNQ", "NQ_PRE", "reference_thresholds", "reference_trades"):
        if sha256(ROOT / SPEC["data"][key]["path"]) != SPEC["data"][key]["sha256"]:
            raise SystemExit(f"data file {key} differs from the frozen sha256")


# ----------------------------------------------------------------------------------------------------- engine
_M = None


def engine():
    global _M
    if _M is None:
        spec = importlib.util.spec_from_file_location("s01norm_rebuild", ROOT / "scripts" / "rebuild_signal_sheet_01_atr18.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        _M = m
    return _M


def truncated(path: Path, start: str, end: str, name: str) -> Path:
    """Deterministic parquet restricted to [start, end + 1 day) New York time (cached)."""
    CACHE.mkdir(exist_ok=True)
    out = CACHE / f"{name}_{start}_{end}.parquet"
    if not out.exists():
        a = pd.Timestamp(start, tz="America/New_York")
        b = pd.Timestamp(end, tz="America/New_York") + pd.Timedelta(days=1)
        df = pd.read_parquet(path, filters=[("timestamp", ">=", a), ("timestamp", "<", b)])
        df.to_parquet(out)
    return out


class Data:
    """Minute frame, first-pass candidate rows and weekday session calendar of one file."""

    def __init__(self, path: Path):
        m = engine()
        self.entry = m.BaselineEntryConfig()
        minute = m.prepare_minute_dataset(path, self.entry, m.ATR_PERIODS)
        self.minute = m.attach_daily_reference(minute, m.build_daily_reference(minute))
        cand = m.build_candidate_universe(self.minute, baseline_entry=self.entry)
        pre = m.first_pass_signal_rows(cand, cand["candidate_base_pass"].fillna(False).astype(bool)).copy()
        for p in m.ATR_PERIODS:
            pre[f"atrn_{p}"] = pd.to_numeric(pre[f"atr_{p}"], errors="coerce") / pd.to_numeric(pre["close"], errors="coerce")
        self.pre = pre
        sd = pd.to_datetime(pd.Series(self.minute["session_date"].unique()))
        self.sessions = sorted(d.date() for d in sd if d.dayofweek < 5)

    def calendar(self, window: str) -> list:
        a, b = W[window]
        return [d for d in self.sessions if a <= d <= b]


def calibrate(data: Data, prefix: str) -> pd.DataFrame:
    m = engine()
    is_sessions = [d for d in sorted(pd.to_datetime(data.minute["session_date"]).dt.date.unique()) if d <= W["IS"][1]]
    rows = []
    for p in m.ATR_PERIODS:
        th = m.calibrate_ensemble_thresholds(data.pre, is_sessions, f"{prefix}_{p}", m.Q_LOWS, m.Q_HIGHS)
        if len(th) != 6:
            raise RuntimeError(f"expected 6 thresholds for {prefix}_{p}")
        for (ql, qh), (lo, hi) in zip(((a, b) for a in m.Q_LOWS for b in m.Q_HIGHS), th):
            rows.append({"model": f"atr{p}_q{ql}_q{qh}", "atr_period": p, "q_low_pct": ql, "q_high_pct": qh,
                         "low_threshold_is": lo, "high_threshold_is": hi})
    return pd.DataFrame(rows)


def kept(data: Data, th: pd.DataFrame, prefix: str | None) -> pd.DataFrame:
    """Signal rows kept by the 18-model vote (prefix None = no ATR filter)."""
    m = engine()
    if prefix is None:
        return data.pre.copy()
    votes = {r.model: pd.to_numeric(data.pre[f"{prefix}_{int(r.atr_period)}"], errors="coerce").between(
        float(r.low_threshold_is), float(r.high_threshold_is), inclusive="both").fillna(False) for r in th.itertuples(index=False)}
    count = pd.DataFrame(votes, index=data.pre.index).sum(axis=1)
    return data.pre.loc[count.ge(m.MINIMUM_VOTES)].copy()


def backtest(data: Data, final: pd.DataFrame, budget: float = BUDGET) -> pd.DataFrame:
    m = engine()
    entry = dataclasses.replace(data.entry, risk_per_trade_pct=budget / data.entry.account_size_usd * 100.0)
    exp = m._build_incremental_experiments(data.entry, m.BaselineEnsembleConfig())[0]
    sig = m.build_signal_frame_for_backtest(data.minute, final)
    t = m.run_exit_variant_backtest(signal_df=sig, execution_model=m.ExecutionModel(), baseline=entry, exit_cfg=exp.exit)
    t["session_date"] = pd.to_datetime(t["session_date"]).dt.date
    return t


def arm_trades(data: Data, final: pd.DataFrame, arm: str) -> pd.DataFrame:
    base = backtest(data, final, BUDGET)
    if arm != "B":
        return base
    wide = backtest(data, final, CAP)
    add = wide[~wide["session_date"].isin(set(base["session_date"]))].copy()
    if len(add) and not ((add["quantity"] == 1).all() and add["risk_per_contract_usd"].between(BUDGET, CAP, inclusive="right").all()):
        raise RuntimeError("arm B: an added trade does not have quantity 1 with risk in (250, 500] USD")
    out = pd.concat([base, add], ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    return out


def in_window(t: pd.DataFrame, window: str) -> pd.DataFrame:
    a, b = W[window]
    return t[(t["session_date"] >= a) & (t["session_date"] <= b)].copy()


def daily(t: pd.DataFrame, calendar: list, col: str = "net_pnl_usd") -> pd.Series:
    s = t.groupby("session_date")[col].sum() if len(t) else pd.Series(dtype=float)
    off = set(s.index) - set(calendar)
    if off:
        raise RuntimeError(f"trades outside the calendar: {sorted(off)[:5]}")
    return s.reindex(calendar).fillna(0.0)


def metrics(t: pd.DataFrame, d: pd.Series) -> dict:
    ls = lo_sharpe(d.to_numpy()) if d.std() > 0 else {"sharpe": float("nan"), "t_hac": float("nan")}
    eq = d.cumsum()
    wins, losses = t.loc[t["net_pnl_usd"] > 0, "net_pnl_usd"].sum(), -t.loc[t["net_pnl_usd"] < 0, "net_pnl_usd"].sum()
    return {"sessions": len(d), "trades": int(len(t)), "net_usd": float(d.sum()), "sharpe": float(ls["sharpe"]),
            "t_hac": float(ls["t_hac"]), "max_dd_usd": float((eq - eq.cummax()).min()), "worst_day_usd": float(d.min()),
            "profit_factor": float(wins / losses) if losses > 0 else float("nan"),
            "win_rate": float((t["net_pnl_usd"] > 0).mean()) if len(t) else float("nan")}


def by_year(d: pd.Series) -> dict:
    s = pd.Series(d.to_numpy(), index=pd.to_datetime(pd.Index(d.index)))
    return {str(k): float(v) for k, v in s.groupby(s.index.year).sum().items()}


def quarterly_counts(t: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(pd.Series(t["session_date"])).dt.to_period("Q").value_counts().sort_index()


def r_units(t: pd.DataFrame, p_ref: float) -> pd.DataFrame:
    t = t.copy()
    risk = t["entry_price"] - t["stop_price"]
    gross = t["pnl_points"] + SLIP_BACK_PTS
    t["gross_R"] = gross / risk
    t["net_scaled_R"] = (gross - COST_PTS * t["entry_price"] / p_ref) / risk
    return t


# ----------------------------------------------------------------------------------------------------- selftest
def selftest() -> dict:
    data = Data(MNQ)
    th_ref = pd.read_csv(TH_REF)
    recal = calibrate(data, "atr")
    m = recal.merge(th_ref, on="model", suffixes=("_r", "_f"))
    gap = float(np.max(np.abs(np.r_[m.low_threshold_is_r - m.low_threshold_is_f, m.high_threshold_is_r - m.high_threshold_is_f])))
    t2 = {"models": len(m), "max_abs_gap": gap, "passed": len(m) == 18 and gap <= 1e-9}
    final = kept(data, th_ref, "atr")
    ref = backtest(data, final)
    before = ref[ref["session_date"] <= pd.Timestamp("2026-03-19").date()]
    a, b = _norm(pd.read_csv(TR_REF)), _norm(before.assign(session_date=pd.to_datetime(before["session_date"])))
    cols = ["session_date", "entry_time", "exit_time", "position", "quantity"]
    ka, kb = set(map(tuple, a[cols].astype(str).to_numpy())), set(map(tuple, b[cols].astype(str).to_numpy()))
    mg = a.merge(b, on=cols, suffixes=("_f", "_r"))
    pnl_gap = float((mg["net_pnl_usd_f"] - mg["net_pnl_usd_r"]).abs().max())
    t1 = {"frozen": len(a), "replayed_to_20260319": len(b), "only_frozen": len(ka - kb), "only_replayed": len(kb - ka),
          "max_abs_pnl_gap": pnl_gap}
    t1["passed"] = t1["only_frozen"] == 0 and t1["only_replayed"] == 0 and pnl_gap <= 0.01
    bb = arm_trades(data, final, "B")
    base_keys = set(map(tuple, _norm(ref.assign(session_date=pd.to_datetime(ref["session_date"])))[cols].astype(str).to_numpy()))
    bkeys = set(map(tuple, _norm(bb.assign(session_date=pd.to_datetime(bb["session_date"])))[cols].astype(str).to_numpy()))
    t3 = {"ref_trades": len(ref), "b_trades": len(bb), "ref_subset_of_b": base_keys <= bkeys, "added": len(bb) - len(ref)}
    t3["passed"] = t3["ref_subset_of_b"]
    out = {"T1": t1, "T2": t2, "T3": t3, "all_passed": t1["passed"] and t2["passed"] and t3["passed"],
           "run_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "note": "reference arm only; PnL of REF is already known; no variant computed"}
    write_json(RES / "selftest.json", out)
    print(json.dumps(out, indent=2, default=_j))
    return out


# ----------------------------------------------------------------------------------------------------- run
def run() -> dict:
    check_lock()
    if (RES / "stage2.json").exists():
        raise SystemExit("stage 2 already read once; refusing a second run (no_rescue_rule)")
    if (RES / "stage1.json").exists():
        raise SystemExit("stage 1 already written; delete nothing, inspect results/ (one run per campaign)")
    st = json.loads((RES / "selftest.json").read_text(encoding="utf-8"))
    if not st["all_passed"]:
        raise SystemExit("technical gates T1-T3 did not pass")

    data = Data(MNQ)
    th = {"REF": pd.read_csv(TH_REF), "A": calibrate(data, "atrn")}
    th["B"] = th["A"]
    th["A"].to_csv(RES / "thresholds_is_atr_over_price.csv", index=False)
    prefix = {"REF": "atr", "A": "atrn", "B": "atrn"}
    trades = {arm: arm_trades(data, kept(data, th[arm], prefix[arm]), arm) for arm in ("REF", "A", "B")}

    # Stage 0: frequency only (session_date and quantity kept, nothing else written).
    counts = pd.DataFrame({arm: quarterly_counts(t[["session_date", "quantity"]]) for arm, t in trades.items()}).fillna(0).astype(int)
    counts.index = counts.index.astype(str)
    counts.to_csv(RES / "stage0_trades_per_quarter.csv")
    is_q = [f"{y}Q{q}" for y in range(2019, 2024) for q in (1, 2, 3, 4) if f"{y}Q{q}" >= "2019Q3"]
    late_q = ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2", "2026Q3"]
    g0 = {}
    for arm in ("REF", "A", "B"):
        c = counts[arm].reindex(is_q + late_q).fillna(0)
        ratio = float(c[late_q].median() / c[is_q].median())
        g0[arm] = {"is_median": float(c[is_q].median()), "late_median": float(c[late_q].median()), "ratio": ratio,
                   "passed": ratio >= 0.60}
    write_json(RES / "stage0.json", g0)

    # Stage 1: IS only.
    cal_is = data.calendar("IS")
    s1 = {}
    is_daily = {}
    for arm in ("REF", "A", "B"):
        t = in_window(trades[arm], "IS")
        d = daily(t, cal_is)
        is_daily[arm] = d
        s1[arm] = metrics(t, d) | {"by_year": by_year(d)}
    ref = s1["REF"]
    for arm in ("A", "B"):
        r = s1[arm]
        yrs = [r["by_year"].get(y, 0.0) for y in SPEC["conventions"]["years_G4"]]
        r["gates"] = {"G0": g0[arm]["passed"], "G1": r["trades"] >= 0.75 * ref["trades"],
                      "G2": r["net_usd"] > 0 and r["t_hac"] >= 2.0, "G3": r["sharpe"] >= 0.75 * ref["sharpe"],
                      "G4": sum(v > 0 for v in yrs) >= 3}
        r["passed_all"] = all(r["gates"].values())
    var_sr = float(np.var([s1["A"]["sharpe"], s1["B"]["sharpe"]], ddof=1))
    for arm in ("A", "B"):
        x = is_daily[arm].to_numpy()
        s1[arm]["psr_0"] = float(psr(x, 0.0))
        s1[arm]["dsr_2_trials"], s1[arm]["dsr_sr0"] = (float(v) for v in dsr(x, 2, var_sr))
    pd.DataFrame(is_daily).to_csv(RES / "stage1_is_daily.csv", index_label="session_date")
    selected = "A" if s1["A"]["passed_all"] else ("B" if s1["B"]["passed_all"] else None)
    s1["selected_arm"] = selected
    s1["run_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(RES / "stage1.json", s1)
    print("stage 0:", json.dumps(g0, default=_j))
    print("stage 1:", json.dumps({a: s1[a].get("gates") for a in ("A", "B")}, default=_j), "selected:", selected)
    if selected is None:
        write_json(RES / "stage2.json", {"verdict": "CLOSED_IS", "read": False})
        return {"verdict": "CLOSED_IS"}

    # Stage 2 (one shot): VAL, EXT on MNQ; PRE on NQ.
    p_ref = float(in_window(trades[selected], "IS")["entry_price"].median())
    s2 = {"selected_arm": selected, "p_ref": p_ref, "read_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    for win in ("VAL", "EXT"):
        cal = data.calendar(win)
        out = {}
        for arm in (selected, "REF"):
            t = in_window(trades[arm], win)
            out[arm] = metrics(t, daily(t, cal)) | {"by_year": by_year(daily(t, cal))}
            t.to_csv(RES / f"stage2_{win.lower()}_trades_{arm}.csv", index=False)
        out["diff_net_usd"] = out[selected]["net_usd"] - out["REF"]["net_usd"]
        s2[win] = out
    s2["V1"] = s2["VAL"][selected]["net_usd"] > 0
    del data

    pre_path = truncated(NQ, *SPEC["data"]["NQ_PRE"]["used_range"], name="NQ_pre")
    nq = Data(pre_path)
    cal = nq.calendar("PRE")
    pre_out = {}
    for arm, (thr, pfx) in {selected: (th[selected], "atrn"), "REF": (th["REF"], "atr"), "NOFILTER": (None, None)}.items():
        t = in_window(arm_trades(nq, kept(nq, thr, pfx), "B" if arm == "B" else "A"), "PRE")
        t = r_units(t, p_ref)
        t.to_csv(RES / f"stage2_pre_trades_{arm}.csv", index=False)
        dg, dn, du = daily(t, cal, "gross_R"), daily(t, cal, "net_scaled_R"), daily(t, cal, "net_pnl_usd")
        lg, ln = lo_sharpe(dg.to_numpy()), lo_sharpe(dn.to_numpy())
        pre_out[arm] = {"sessions": len(cal), "trades": int(len(t)), "trades_by_year": {str(k): int(v) for k, v in
                        pd.to_datetime(pd.Series(t["session_date"])).dt.year.value_counts().sort_index().items()},
                        "gross_R_total": float(dg.sum()), "gross_R_per_trade": float(t["gross_R"].mean()), "gross_t_hac": float(lg["t_hac"]),
                        "net_scaled_R_total": float(dn.sum()), "net_scaled_R_per_trade": float(t["net_scaled_R"].mean()),
                        "net_scaled_t_hac": float(ln["t_hac"]), "net_scaled_sharpe": float(ln["sharpe"]),
                        "net_scaled_R_by_year": by_year(dn), "raw_net_usd_fixed_mnq_costs": float(du.sum()),
                        "median_entry_price": float(t["entry_price"].median()) if len(t) else float("nan")}
    sel = pre_out[selected]
    yrs = [sel["net_scaled_R_by_year"].get(y, 0.0) for y in SPEC["conventions"]["years_P3"]]
    s2["PRE"] = pre_out
    s2["gates"] = {"P1": sel["gross_R_total"] > 0 and sel["gross_t_hac"] >= 2.0,
                   "P2": sel["net_scaled_R_total"] > 0 and sel["net_scaled_t_hac"] >= 1.65,
                   "P3": sum(v > 0 for v in yrs) >= 5, "V1": bool(s2["V1"])}
    g = s2["gates"]
    s2["verdict"] = ("CANDIDATE_SHADOW" if all(g.values()) else
                     "REJECTED_PRE" if not (g["P1"] and g["P2"] and g["P3"]) else "REJECTED_VAL")
    write_json(RES / "stage2.json", s2)
    print("stage 2:", json.dumps(s2["gates"]), s2["verdict"])
    return s2


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"selftest": selftest, "run": run}[cmd]()
