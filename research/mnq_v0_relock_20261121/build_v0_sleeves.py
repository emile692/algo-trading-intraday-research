"""Freeze MNQ.v.0 (back-adjusted by the roll spread) as the MNQ data of the book, and rebuild S01-S04 on it.

User decision of 2026-11-21: "freeze le mnq v.0 au lieu de c.0" with the back-adjusted variant, the same window as
the 2026-11-15 lock, stop-rule thresholds recomputed on the v.0 histories.

1. Data: MNQ.v.0 1m (data/downloaded/MNQ_v_0_1m_20190501_20261120.parquet) cut at the last timestamp of the frozen
   c.0 input (2026-03-20 09:29 ET), back-adjusted additively by the roll spread measured at each v.0 contract switch
   (median v.0 - c.0 close over the common minutes of the switch session).  Written once to
   data/processed/parquet/MNQ_v_0_adj_1m_20190505_20260320.parquet with the c.0 column layout.
2. Sleeves, engines unchanged, definitions unchanged, output in NEW directories (the c.0 ones are never modified):
   S01 -> research/signal_sheet_01_atr18_v0 : frozen IS thresholds of research/signal_sheet_01_atr18/thresholds_is.csv
         (no recalibration), same IS/OOS split dates as the c.0 sheet;
   S02 -> research/signal_sheet_02_cluster_v0 : scripts/rebuild_signal_sheet_02_cluster.py::execute, data path swapped;
   S03/S04 -> research/signal_sheet_0{3,4}_cluster_v0 : the core of scripts/freeze_signal_sheet_cluster_vote.py::freeze
         with the frozen manifests of strategy_registry/psf_mnq/frozen_ensembles read, never rewritten.
Command: python -m research.mnq_v0_relock_20261121.build_v0_sleeves
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research.mnq_v0_roll_replay_20261120 import replay  # noqa: E402

FROZEN_DATA = ROOT / "data" / "processed" / "parquet" / "MNQ_v_0_adj_1m_20190505_20260320.parquet"
S01_C0 = ROOT / "research" / "signal_sheet_01_atr18"
OUT = {"S01": ROOT / "research" / "signal_sheet_01_atr18_v0", "S02": ROOT / "research" / "signal_sheet_02_cluster_v0",
       "s03": ROOT / "research" / "signal_sheet_03_cluster_v0", "s04": ROOT / "research" / "signal_sheet_04_cluster_v0"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ----------------------------------------------------------------------------- 1. data
def freeze_data() -> dict:
    if FROZEN_DATA.exists():
        raise SystemExit(f"{rel(FROZEN_DATA)} already exists: the frozen MNQ v.0 file is written once")
    spreads = replay.build_variants()  # writes research/mnq_v0_roll_replay_20261120/cache/MNQ_v0_adj.parquet
    shutil.copyfile(replay.CACHE / "MNQ_v0_adj.parquet", FROZEN_DATA)
    df = pd.read_parquet(FROZEN_DATA)
    meta = {
        "file": rel(FROZEN_DATA), "sha256": sha256(FROZEN_DATA), "rows": int(len(df)),
        "first": str(df.index.min()), "last": str(df.index.max()),
        "source": {"file": rel(replay.V0), "sha256": sha256(replay.V0), "symbol": "MNQ.v.0",
                   "dataset": "GLBX.MDP3 ohlcv-1m (Databento), volume-ranked front month, unadjusted"},
        "cut": f"last timestamp of the frozen c.0 input {rel(replay.C0)} (sha256 {sha256(replay.C0)})",
        "adjustment": "additive back-adjustment: every bar before a v.0 contract switch += roll spread of that switch "
                      "(median v.0 - c.0 close over the common minutes of the switch session, c.0 still on the old "
                      "contract), rounded to 0.25; prices after the last switch are actual prices",
        "columns": "same layout and dtypes as the c.0 file; symbol column set to the c.0 value",
        "roll_spreads": spreads.to_dict(orient="records"),
        "decision": "user, 2026-11-21: freeze MNQ v.0 (back-adjusted) instead of c.0, same window",
    }
    FROZEN_DATA.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


# ----------------------------------------------------------------------------- 2. S01
def build_s01() -> dict:
    m = _load("rebuild_s01_v0", ROOT / "scripts" / "rebuild_signal_sheet_01_atr18.py")
    th = pd.read_csv(S01_C0 / "thresholds_is.csv")
    c0_summary = json.loads((S01_C0 / "summary.json").read_text(encoding="utf-8"))
    is_end = pd.Timestamp(c0_summary["split"]["is_end"]).date()
    entry = m.BaselineEntryConfig()
    minute = m.prepare_minute_dataset(FROZEN_DATA, entry, m.ATR_PERIODS)
    minute = m.attach_daily_reference(minute, m.build_daily_reference(minute))
    cand = m.build_candidate_universe(minute, baseline_entry=entry)
    sessions = sorted(pd.to_datetime(minute["session_date"]).dt.date.unique())
    is_sessions = [s for s in sessions if s <= is_end]
    oos_sessions = [s for s in sessions if s > is_end]
    pre = m.first_pass_signal_rows(cand, cand["candidate_base_pass"].fillna(False).astype(bool))
    votes = pd.DataFrame({r.model: pd.to_numeric(pre[f"atr_{int(r.atr_period)}"], errors="coerce").between(
        float(r.low_threshold_is), float(r.high_threshold_is), inclusive="both").fillna(False)
        for r in th.itertuples(index=False)}, index=pre.index)
    pre = pre.copy()
    pre["atr_vote_count"] = votes.sum(axis=1).astype(int)
    pre["atr_vote_score"] = pre["atr_vote_count"] / float(len(th))
    final = pre.loc[pre["atr_vote_count"].ge(m.MINIMUM_VOTES)].copy()
    exp = m._build_incremental_experiments(entry, m.BaselineEnsembleConfig())[0]
    trades = m.run_exit_variant_backtest(signal_df=m.build_signal_frame_for_backtest(minute, final),
                                         execution_model=m.ExecutionModel(), baseline=entry, exit_cfg=exp.exit)
    td = pd.to_datetime(trades["session_date"]).dt.date
    scoped = {"overall": (trades, sessions), "is": (trades[td.isin(set(is_sessions))].copy(), is_sessions),
              "oos": (trades[td.isin(set(oos_sessions))].copy(), oos_sessions)}
    metrics = {scope: m.compute_extended_metrics(trades=t, signal_df=None, sessions=s, initial_capital=entry.account_size_usd,
                                                 bootstrap_paths=300, random_seed=42 + i * 17)
               for i, (scope, (t, s)) in enumerate(scoped.items())}
    daily = m._daily_path(trades, sessions)
    out = OUT["S01"]
    out.mkdir(parents=True, exist_ok=False)
    trades.to_csv(out / "trades.csv", index=False)
    daily.to_csv(out / "daily_results.csv", index=False)
    shutil.copyfile(S01_C0 / "thresholds_is.csv", out / "thresholds_is.csv")
    pre[["session_date", "timestamp", "atr_vote_count", "atr_vote_score"]].to_csv(out / "candidate_votes.csv", index=False)
    payload = {"dataset": rel(FROZEN_DATA), "dataset_sha256": sha256(FROZEN_DATA),
               "thresholds": "frozen IS thresholds copied from research/signal_sheet_01_atr18/thresholds_is.csv (calibrated on c.0, never recalibrated)",
               "ensemble": c0_summary["ensemble"],
               "split": {"is_start": str(is_sessions[0]), "is_end": str(is_sessions[-1]), "oos_start": str(oos_sessions[0]),
                         "oos_end": str(oos_sessions[-1]), "is_sessions": len(is_sessions), "oos_sessions": len(oos_sessions),
                         "rule": "same IS end date as the c.0 sheet (2024-02-27)"},
               "metrics": metrics}
    (out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=m._json_value) + "\n", encoding="utf-8")
    return {"trades": int(len(trades)), "net_pnl_usd": float(trades["net_pnl_usd"].sum())}


# ----------------------------------------------------------------------------- 3. S02
def build_s02() -> dict:
    import rebuild_signal_sheet_02_cluster as r
    r.DATA_PATH = FROZEN_DATA
    r.OUTPUT_DIR = OUT["S02"]
    if OUT["S02"].exists():
        raise SystemExit(f"{rel(OUT['S02'])} exists")
    r.execute()
    t = pd.read_csv(OUT["S02"] / "trades.csv")
    return {"trades": int(len(t)), "net_pnl_usd": float(t["net_pnl_usd"].sum())}


# ----------------------------------------------------------------------------- 4. S03 / S04
def build_cluster(key: str) -> dict:
    import freeze_signal_sheet_cluster_vote as f
    import run_signal_sheet_parameter_refinement as rr
    rr.MNQ_DATA = FROZEN_DATA
    if hasattr(f, "MNQ_DATA"):
        f.MNQ_DATA = FROZEN_DATA
    campaign = f.CAMPAIGNS[key]
    ensemble_id, _, count = f.identifiers(key)
    frozen_manifest = f.ENSEMBLE_DIR / f"{ensemble_id}.json"
    frozen_manifest_sha = sha256(frozen_manifest)
    out = OUT[key]
    out.mkdir(parents=True, exist_ok=False)
    prepared, base_spec, source = f.prepare_campaign(key, campaign)
    manifest = f.build_manifest(key, base_spec, source)
    positions, cell_checks = f.build_cell_positions(key, base_spec, prepared)
    if positions["cell_id"].nunique() != count:
        raise RuntimeError(f"{key}: {positions['cell_id'].nunique()} cells, {count} expected")
    f.audit_event_alignment(positions, prepared)
    result, vote_events = f.aggregate_vote(base_spec, prepared, positions, manifest)
    violations = {n: result.checks.get(n) for n in f.WATCHED_CHECKS if result.checks.get(n, 0)}
    if violations:
        raise RuntimeError(f"{key} aggregate vote: {violations}")
    trades, daily = f.relabel(result.trades, key), f.relabel(result.daily, key)
    positions.to_csv(out / "cell_positions.csv", index=False)
    vote_events.to_csv(out / "vote_events.csv", index=False)
    trades.to_csv(out / "trades.csv", index=False)
    daily.to_csv(out / "daily_results.csv", index=False)
    summary = {"ensemble_manifest": rel(frozen_manifest), "ensemble_manifest_sha256": frozen_manifest_sha,
               "ensemble_manifest_rewritten": False, "data": rel(FROZEN_DATA), "data_sha256": source["data_sha256"],
               "submodel_count": count, "vote_threshold": 0.5, "execution_semantics": manifest["aggregate_execution"],
               "aggregate_checks": result.checks, "cell_count": int(positions["cell_id"].nunique()),
               "cell_position_rows": int(len(positions)), "cell_checks_all_valid": True,
               "sharpe_definition": "sqrt(252) x mean / std of daily net PnL, as in the refinement grid",
               "metrics": f.scoped(daily, trades, campaign)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "cell_checks.json").write_text(json.dumps(cell_checks, indent=2, default=str) + "\n", encoding="utf-8")
    if sha256(frozen_manifest) != frozen_manifest_sha:
        raise RuntimeError(f"{rel(frozen_manifest)} was modified")
    return {"trades": int(len(trades)), "net_pnl_usd": float(trades["net_pnl_usd"].sum())}


def main() -> None:
    meta = freeze_data()
    print("data", meta["file"], meta["sha256"], meta["rows"], flush=True)
    res = {"S01": build_s01()}
    print("S01", res["S01"], flush=True)
    res["S02"] = build_s02()
    print("S02", res["S02"], flush=True)
    for key in ("s03", "s04"):
        res[key.upper()] = build_cluster(key)
        print(key, res[key.upper()], flush=True)
    (HERE / "build_v0_sleeves_result.json").write_text(json.dumps({"data": {k: meta[k] for k in ("file", "sha256", "rows", "first", "last")},
                                                                   "sleeves": res}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
