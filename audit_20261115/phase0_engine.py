"""Phase 0, level B — re-execute every sleeve from raw minute bars, in memory, with the repository engines.

Each sleeve's trade ledger is regenerated and compared trade by trade with the frozen ledger.  Nothing is
written outside audit_20261115/ (regenerated ledgers are cached in audit_20261115/cache/engine_<code>.csv).
The data windows are those of the frozen campaigns (MNQ/MGC <= 2026-03-19 or the MNQ file ending 2026-03-21,
MBT <= 2026-10-02): no unseen data is touched.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

from .common import CACHE, CODES, ROOT, load_trades, sleeve_dir

KEY_COLS = ["session_date", "entry_time", "exit_time", "position", "quantity"]


def _norm(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    if "position" not in t:
        t["position"] = np.where(t["direction"].str.lower() == "long", 1, -1)
    t["session_date"] = pd.to_datetime(t["session_date"]).dt.strftime("%Y-%m-%d")
    for c in ("entry_time", "exit_time"):
        t[c] = pd.to_datetime(t[c], utc=True).dt.strftime("%Y-%m-%dT%H:%MZ")
    t["quantity"] = t["quantity"].astype(float)
    return t[KEY_COLS + ["net_pnl_usd"]].sort_values(["entry_time"]).reset_index(drop=True)


def run_s01_atr18() -> pd.DataFrame:
    import importlib.util
    spec = importlib.util.spec_from_file_location("audit_rebuild_s01", ROOT / "scripts" / "rebuild_signal_sheet_01_atr18.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    entry = m.BaselineEntryConfig()
    minute = m.prepare_minute_dataset(m.DATASET, entry, m.ATR_PERIODS)
    minute = m.attach_daily_reference(minute, m.build_daily_reference(minute))
    candidates = m.build_candidate_universe(minute, baseline_entry=entry)
    sessions = sorted(pd.to_datetime(minute["session_date"]).dt.date.unique())
    is_sessions, _ = m._split_sessions(sessions, 0.70)
    pre = m.first_pass_signal_rows(candidates, candidates["candidate_base_pass"].fillna(False).astype(bool))
    votes = {}
    for period in m.ATR_PERIODS:
        col = f"atr_{period}"
        th = m.calibrate_ensemble_thresholds(pre, is_sessions, col, m.Q_LOWS, m.Q_HIGHS)
        for (ql, qh), (lo, hi) in zip(((a, b) for a in m.Q_LOWS for b in m.Q_HIGHS), th):
            votes[f"atr{period}_q{ql}_q{qh}"] = pd.to_numeric(pre[col], errors="coerce").between(lo, hi, inclusive="both").fillna(False)
    v = pd.DataFrame(votes, index=pre.index)
    pre = pre.copy()
    pre["atr_vote_count"] = v.sum(axis=1).astype(int)
    final = pre.loc[pre["atr_vote_count"].ge(m.MINIMUM_VOTES)].copy()
    exp = m._build_incremental_experiments(entry, m.BaselineEnsembleConfig())[0]
    sig = m.build_signal_frame_for_backtest(minute, final)
    return m.run_exit_variant_backtest(signal_df=sig, execution_model=m.ExecutionModel(), baseline=entry, exit_cfg=exp.exit)


def run_s01_baseline_fixed() -> pd.DataFrame:
    import importlib.util
    spec = importlib.util.spec_from_file_location("audit_report_s01", ROOT / "scripts" / "report_mnq_orb_vwap_atr.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    entry, ens = m.BaselineEntryConfig(), m.BaselineEnsembleConfig()
    exp = m._build_incremental_experiments(entry, ens)[0]
    minute = m.prepare_minute_dataset(m.DATASET, entry, (10, 14, 20, 30))
    minute = m.attach_daily_reference(minute, m.build_daily_reference(minute))
    cand = m.build_candidate_universe(minute, baseline_entry=entry)
    sessions = sorted(pd.to_datetime(minute["session_date"]).dt.date.unique())
    is_s, oos_s = m._split_sessions(sessions, 0.70)
    ctx = m.CampaignContext(all_sessions=sessions, is_sessions=is_s, oos_sessions=oos_s, minute_df=minute,
                            candidate_base_df=cand, daily_patterns=None)
    _, details = m._evaluate_experiment(exp, ctx, bootstrap_paths=300, random_seed=42, keep_details=True)
    return details["trades"]


def run_s02() -> pd.DataFrame:
    import rebuild_signal_sheet_02_cluster as r
    manifest = json.loads(r.MANIFEST_PATH.read_text(encoding="utf-8"))
    specs, _, _ = r.load_frozen_specs(r.FROZEN_DIR, r.FROZEN_MANIFEST)
    spec = next(s for s in specs if s["mechanism_id"] == "volatility_breakout")
    prepared = r.prepare_data(r.DATA_PATH, start_date=r.IS_START, end_date=r.OOS_END)
    positions, _ = r.build_cell_positions(spec, prepared, manifest)
    result, _ = r.aggregate_vote(spec, prepared, positions, manifest)
    return result.trades


def run_cluster(key: str) -> pd.DataFrame:
    import freeze_signal_sheet_cluster_vote as f
    campaign = f.CAMPAIGNS[key]
    prepared, base_spec, source = f.prepare_campaign(key, campaign)
    manifest = f.build_manifest(key, base_spec, source)
    positions, _ = f.build_cell_positions(key, base_spec, prepared)
    result, _ = f.aggregate_vote(base_spec, prepared, positions, manifest)
    return f.relabel(result.trades, key)


RUNNERS = {
    "S01": run_s01_atr18, "S01_book": run_s01_baseline_fixed, "S02": run_s02,
    **{c: (lambda k=c.lower(): run_cluster(k)) for c in CODES[2:]},
}


def regenerate(code: str) -> pd.DataFrame:
    path = CACHE / f"engine_{code}.csv"
    if path.exists():
        return pd.read_csv(path)
    t0 = time.time()
    t = RUNNERS[code]()
    t.to_csv(path, index=False)
    print(f"{code}: {len(t)} trades regenerated in {time.time() - t0:.0f}s", flush=True)
    return t


def frozen(code: str) -> pd.DataFrame:
    if code == "S01_book":
        return pd.read_csv(ROOT / "export" / "mnq_orb_vwap_atr_retained" / "trades.csv")
    return load_trades(code)


def compare(code: str) -> dict:
    a, b = _norm(frozen(code)), _norm(regenerate(code))
    ka = set(map(tuple, a[KEY_COLS].astype(str).to_numpy()))
    kb = set(map(tuple, b[KEY_COLS].astype(str).to_numpy()))
    merged = a.merge(b, on=KEY_COLS, how="inner", suffixes=("_frozen", "_engine"))
    pnl_diff = float((merged["net_pnl_usd_frozen"] - merged["net_pnl_usd_engine"]).abs().max()) if len(merged) else 0.0
    return {"frozen_trades": len(a), "engine_trades": len(b), "only_frozen": len(ka - kb), "only_engine": len(kb - ka),
            "max_abs_pnl_diff_matched": pnl_diff, "frozen_total": float(a.net_pnl_usd.sum()),
            "engine_total": float(b.net_pnl_usd.sum())}


def compare_all() -> tuple[list[dict], dict]:
    rows, detail = [], {}
    for code in ["S01", "S01_book"] + CODES[1:]:
        c = compare(code)
        detail[code] = c
        ok = c["only_frozen"] == 0 and c["only_engine"] == 0 and c["max_abs_pnl_diff_matched"] < 0.01
        label = "S01 version book (baseline_fixed)" if code == "S01_book" else code
        rows.append({"élément": f"{label} ré-exécuté depuis les barres 1 min", "métrique": "trades identiques (clé + PnL)",
                     "attendu": c["frozen_trades"], "obtenu": c["engine_trades"],
                     "écart": c["only_frozen"] + c["only_engine"], "ok": ok})
        rows.append({"élément": f"{label} ré-exécuté depuis les barres 1 min", "métrique": "PnL total",
                     "attendu": c["frozen_total"], "obtenu": c["engine_total"],
                     "écart": c["engine_total"] - c["frozen_total"], "ok": abs(c["engine_total"] - c["frozen_total"]) < 1.0})
    return rows, detail


if __name__ == "__main__":
    codes = sys.argv[1:] or ["S01", "S01_book"] + CODES[1:]
    for code in codes:
        regenerate(code)
        print(code, json.dumps(compare(code)), flush=True)
