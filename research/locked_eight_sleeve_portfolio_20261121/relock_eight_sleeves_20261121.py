"""Versioned eight-sleeve lock (2026-11-21): the MNQ sleeves run on MNQ.v.0 back-adjusted instead of MNQ.c.0.

User decision of 2026-11-21 ("freeze le mnq v.0 au lieu de c.0"): back-adjusted v.0, same window as the 2026-11-15
lock, stop-rule thresholds recomputed on the v.0 histories.  Only the MNQ DATA changes:

- MNQ.c.0 kept the expiring contract through roll week, had no RTH bar on quarterly expiry Fridays and showed an
  artificial gap (up to 276 points) on its contract switch, which entered the daily ATR.  Production trades the
  volume front month, i.e. what v.0 follows (research/mnq_v0_roll_replay_20261120/RESULTS.md).
- S01-S04 are rebuilt with unchanged engines and definitions by research/mnq_v0_relock_20261121/build_v0_sleeves.py
  (S01 keeps its frozen IS thresholds; S02-S04 their frozen vote manifests).  MGC and MBT sleeves keep their sources
  byte for byte.  S04 stays in shadow.  One whole contract per sleeve, S01 at its native 1-4 contracts.

Steps: union calendar of the instruments' sessions on 2024-01-02 -> 2026-03-19; minute paths marked on the v.0
bars and reconciled with the daily PnL; Topstep 50K moving-block bootstrap (same seed and harness as the previous
locks), also run on the 2026-11-15 book with the same paths for comparison; stop-rule thresholds (rolling 126-session
minimum) recomputed, MGC/MBT checked equal to the registered ones; netting engine reconciliation on the v.0 bars.
The 2026-11-15 lock directory is never modified.  Command:
    python -m research.locked_eight_sleeve_portfolio_20261121.relock_eight_sleeves_20261121
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PREV = REPO / "research" / "locked_eight_sleeve_portfolio_20261115"
NETTING = REPO / "research" / "book_aggregation_risk_r1_20261112"
BUILD = REPO / "research" / "mnq_v0_relock_20261121"
MNQ_V0 = REPO / "data" / "processed" / "parquet" / "MNQ_v_0_adj_1m_20190505_20260320.parquet"
PORTFOLIO_ID = "eight_sleeve_equal_contract_reference_20261121"
SUPERSEDED_ID = "eight_sleeve_equal_contract_reference_20261115"
DECISION_DATE = "2026-11-21"
WINDOW = ("2024-01-02", "2026-03-19")
ET = "America/New_York"
S01 = "mnq_orb_vwap_atr"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


L = _load("relock_eight_sleeves_20261115_for_v0", PREV / "relock_eight_sleeves_20261115.py")
E, R, T = L.E, L.R, L.T
SLEEVES, SHADOW = L.SLEEVES, L.SHADOW
NINE = E.NINE_SLEEVES
DIRS_C0 = {S01: "signal_sheet_01_atr18", **{s: d for s, (_, d) in R.VOTE_SLEEVES.items()}}
DIRS = dict(DIRS_C0)
DIRS.update({S01: "signal_sheet_01_atr18_v0", "mnq_volatility_breakout": "signal_sheet_02_cluster_v0",
             "mnq_compression_breakout": "signal_sheet_03_cluster_v0", "mnq_atr_trend_flip": "signal_sheet_04_cluster_v0"})
MNQ_SLEEVES = [s for s in NINE if s.startswith("mnq_")]
NETTING_IDS = {S01: "S01", "mnq_volatility_breakout": "S02", "mnq_compression_breakout": "S03"}


def ledger(sleeve: str, dirs: dict[str, str]) -> pd.DataFrame:
    if sleeve != S01:
        return R.vote_trades(dirs[sleeve])
    t = pd.read_csv(REPO / "research" / dirs[S01] / "trades.csv")
    t["entry_ts_utc"] = pd.to_datetime(t["entry_time"], utc=True)
    t["exit_ts_utc"] = pd.to_datetime(t["exit_time"], utc=True)
    t["signed_quantity"] = np.where(t["direction"].str.lower() == "long", 1, -1) * t["quantity"]
    return t


def daily_frame(dirs: dict[str, str], calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily, counts = pd.DataFrame(index=calendar), pd.DataFrame(index=calendar)
    for sleeve in SLEEVES:
        frame = pd.read_csv(REPO / "research" / dirs[sleeve] / "daily_results.csv", parse_dates=["session_date"]).set_index("session_date")
        pnl = frame["daily_pnl_usd"].astype(float)
        off = pnl[~pnl.index.isin(calendar) & (pnl.index >= calendar.min()) & (pnl.index <= calendar.max())]
        if (off != 0).any():
            raise RuntimeError(f"{sleeve}: PnL on a session outside the union calendar")
        daily[sleeve] = pnl.reindex(calendar).fillna(0.0)
        t = ledger(sleeve, dirs)
        counts[sleeve] = t.groupby(pd.to_datetime(t["session_date"])).size().reindex(calendar).fillna(0).astype(int)
        sd = pd.to_datetime(t["session_date"])
        total = float(t.loc[(sd >= WINDOW[0]) & (sd <= WINDOW[1]), "net_pnl_usd"].sum())
        if abs(total - float(daily[sleeve].sum())) > 1e-6:
            raise RuntimeError(f"{sleeve}: daily PnL differs from its trades on the window")
    return daily, counts


def union_calendar(dirs: dict[str, str]) -> pd.DatetimeIndex:
    dates: set[pd.Timestamp] = set()
    for sleeve in NINE:
        if sleeve == S01:
            continue
        d = pd.read_csv(REPO / "research" / dirs[sleeve] / "daily_results.csv", usecols=["session_date"], parse_dates=["session_date"])
        dates |= set(d["session_date"])
    cal = pd.DatetimeIndex(sorted(dates))
    return cal[(cal >= WINDOW[0]) & (cal <= WINDOW[1])]


def timeline_for(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    src = pd.read_csv(T.SIX_DIR / "common_timestamp_portfolio.csv", usecols=["timestamp_utc"], parse_dates=["timestamp_utc"])
    timeline = pd.DatetimeIndex(src["timestamp_utc"]).tz_convert("UTC")
    have = pd.DatetimeIndex(pd.to_datetime(sorted(set(timeline.tz_convert(ET).date))))
    tl = L.extend_timeline(timeline, calendar.difference(have))
    keep = pd.DatetimeIndex(pd.to_datetime(tl.tz_convert(ET).date)).isin(calendar)
    return tl[keep]


def stop_rule_thresholds() -> pd.DataFrame:
    """Same rule as the 2026-11-03 registration (rolling 126-session sum, threshold = historical minimum), on the
    v.0 histories for the MNQ sleeves (S01 aligned on the MNQ v.0 vote calendar)."""
    mnq_cal = pd.DatetimeIndex(pd.read_csv(REPO / "research" / DIRS["mnq_volatility_breakout"] / "daily_results.csv",
                                           parse_dates=["session_date"])["session_date"])
    rows = []
    for sleeve in NINE:
        frame = pd.read_csv(REPO / "research" / DIRS[sleeve] / "daily_results.csv", parse_dates=["session_date"])
        history = frame.set_index("session_date")["daily_pnl_usd"].astype(float)
        if sleeve == S01:
            history = history.reindex(mnq_cal).fillna(0.0)
        rolling = history.rolling(E.STOP_WINDOW).sum().dropna()
        rows.append({"sleeve": sleeve, "status": "SHADOW_FORWARD_ONLY" if sleeve == SHADOW else "LOCKED_ACTIVE_REFERENCE",
                     "history_start": str(history.index.min().date()), "history_end": str(history.index.max().date()),
                     "history_sessions": int(len(history)), "window_sessions": E.STOP_WINDOW, "rolling_windows": int(len(rolling)),
                     "threshold_usd": float(rolling.min()), "historical_p01_usd": float(rolling.quantile(0.01)),
                     "historical_median_usd": float(rolling.median())})
    return pd.DataFrame(rows)


def netting_check(union_daily: pd.DataFrame) -> dict[str, Any]:
    rs = _load("netting_run_study_20261121_lock", NETTING / "run_study.py")
    spec = rs.load_spec()
    for sleeve, sid in NETTING_IDS.items():
        spec["sleeves"][sid] = dict(spec["sleeves"][sid], trades=f"research/{DIRS[sleeve]}/trades.csv")
    spec["instruments"]["MNQ"] = dict(spec["instruments"]["MNQ"], bars_1m=MNQ_V0.relative_to(REPO).as_posix())
    trades = rs.load_trades(spec)
    calendars = rs.load_calendars(spec)[0]
    dates = rs.union_sessions(calendars, rs.d(WINDOW[0]), rs.d(WINDOW[1]))
    end = pd.Timestamp(WINDOW[1], tz=ET) + pd.Timedelta(days=1)
    import pyarrow.parquet as pq
    bars = {}
    for name in rs.INSTRUMENTS:
        path = REPO / spec["instruments"][name]["bars_1m"]
        col = next(c for c in ("timestamp", "ts_event") if c in pq.read_schema(path).names)
        tz = str(pq.read_schema(path).field(col).type.tz)
        b = pd.read_parquet(path, columns=["open", "high", "low", "close"], filters=[(col, "<", end.tz_convert(tz))])
        b.index = b.index.tz_convert("UTC")
        bars[name] = b[~b.index.duplicated(keep="last")].sort_index()
    cube = rs.BarCube(bars, dates)
    sizes = np.where(trades.sleeve == "S01", trades.qty_native, 1).astype(int)
    events = rs.build_events(trades, sizes)
    specs = rs.instrument_specs(spec)
    from src.portfolio.aggregator import RiskLimits, simulate_session
    res = [simulate_session(cube.session(x), events.get(x, rs.EMPTY_EVENTS), rs.SLEEVE_INSTRUMENT, specs, RiskLimits())
           for x in dates]
    in_window = trades[(trades.session_date >= rs.d(WINDOW[0])) & (trades.session_date <= rs.d(WINDOW[1]))]
    separate_gross, netted_gross = float(in_window["gross_native"].sum()), float(sum(r.gross for r in res))
    if len(dates) != len(union_daily) or abs(separate_gross - netted_gross) > 1e-6:
        raise RuntimeError(f"netting reconciliation failed: {len(dates)} sessions, {separate_gross} vs {netted_gross}")
    if abs(float(in_window["net_native"].sum()) - float(union_daily[SLEEVES].sum().sum())) > 1e-6:
        raise RuntimeError("netting trade ledger differs from the lock daily PnL")
    return {"window": list(WINDOW), "sessions": len(dates),
            "separate_net_usd": float(in_window["net_native"].sum()), "netted_net_usd": float(sum(r.final for r in res)),
            "gross_usd": netted_gross, "separate_costs_usd": float(in_window["cost_native"].sum()),
            "netted_costs_usd": float(sum(r.costs for r in res)), "peak_lots_uncapped": max(r.peak_units for r in res) / 10.0,
            "engine": "src/portfolio/aggregator.py (unchanged); MNQ bars and S01-S03 ledgers = v.0 back-adjusted"}


def main() -> None:
    union = union_calendar(DIRS)
    union_c0 = union_calendar(DIRS_C0)
    if not union.equals(union_c0):
        raise RuntimeError(f"union calendar changed: {len(union)} vs {len(union_c0)} sessions")
    daily, counts = daily_frame(DIRS, union)
    daily_c0, counts_c0 = daily_frame(DIRS_C0, union)
    published = pd.read_csv(PREV / "daily_pnl.csv", parse_dates=["session_date"]).set_index("session_date")
    if float((published[SLEEVES] - daily_c0[SLEEVES]).abs().max().max()) > 1e-6:
        raise RuntimeError("the 2026-11-15 book is not reproduced from its sources")

    timeline = timeline_for(union)
    tdates = pd.Series(pd.to_datetime(timeline.tz_convert(ET).date), index=timeline)
    path_c0, adverse_c0 = R.intraday_paths(timeline, {s: ledger(s, DIRS_C0) for s in SLEEVES})
    R.reconcile(path_c0, daily_c0[SLEEVES], tdates, "2026-11-15 book, c.0 bars")
    R.PRICE_SOURCES["MNQ"] = (MNQ_V0, "timestamp")
    path_v0, adverse_v0 = R.intraday_paths(timeline, {s: ledger(s, DIRS) for s in SLEEVES})
    R.reconcile(path_v0, daily[SLEEVES], tdates, "2026-11-21 book, v.0 bars")

    books = {"eight_20261121": E.book(daily, adverse_v0[SLEEVES], counts, tdates, SLEEVES),
             "eight_20261115_c0": E.book(daily_c0, adverse_c0[SLEEVES], counts_c0, tdates, SLEEVES)}
    prev_metrics = json.loads((PREV / "locked_eight_sleeve_config.json").read_text(encoding="utf-8"))["book_window_metrics"]
    for key in ("net_pnl_usd", "daily_sharpe", "worst_day_usd", "max_intraday_drawdown_usd"):
        if abs(float(books["eight_20261115_c0"]["risk"][key]) - float(prev_metrics[key])) > 1e-6:
            raise RuntimeError(f"2026-11-15 book risk not reproduced on this harness: {key}")
    sampler = counts[SLEEVES].gt(0).any(axis=1).to_numpy(bool)
    summary = R.bootstrap(books, sampler)
    pooled = summary[summary["block"].astype(str).eq("pooled")].set_index(["scenario", "horizon"])

    thresholds = stop_rule_thresholds()
    registered = pd.read_csv(PREV / "sleeve_stop_rule_thresholds.csv").set_index("sleeve")
    for row in thresholds.itertuples(index=False):
        if not row.sleeve.startswith("mnq_") and abs(row.threshold_usd - float(registered.loc[row.sleeve, "threshold_usd"])) > 1e-9:
            raise RuntimeError(f"{row.sleeve}: stop-rule threshold changed although its source did not")
    thresholds = thresholds.merge(registered["threshold_usd"].rename("threshold_usd_c0_20261103"), left_on="sleeve", right_index=True)

    netting = netting_check(daily)

    # --- write
    out = daily[SLEEVES].copy()
    out["portfolio_pnl_usd"] = out[SLEEVES].sum(axis=1)
    out.index.name = "session_date"
    out.reset_index().to_csv(HERE / "daily_pnl.csv", index=False, float_format="%.8f")
    summary.to_csv(HERE / "topstep_summary.csv", index=False)
    ref = pooled.loc["eight_20261121"].reset_index().assign(method="moving_block_bootstrap", block="pooled",
                                                            scenario="equal_contract__nominal__eight__mnq_v0_adj__union")
    ref = ref[["method", "scenario", "horizon", "block", "paths", "pass_probability", "mll_probability",
               "expiry_probability", "median_days_to_pass", "median_max_drawdown_usd", "p05_max_drawdown_usd"]]
    ref.to_csv(HERE / "topstep_reference_metrics.csv", index=False)
    pd.DataFrame([{"book": n, "calendar": f"union ({len(union)})", **v["risk"]} for n, v in books.items()]).to_csv(
        HERE / "book_risk_comparison.csv", index=False)
    thresholds.to_csv(HERE / "sleeve_stop_rule_thresholds.csv", index=False)
    (HERE / "netting_check.json").write_text(json.dumps(netting, indent=2) + "\n", encoding="utf-8")

    prev_config = json.loads((PREV / "locked_eight_sleeve_config.json").read_text(encoding="utf-8"))
    build = json.loads((BUILD / "build_v0_sleeves_result.json").read_text(encoding="utf-8"))
    data_meta = {"path": R.relative(MNQ_V0), "sha256": R.sha256(MNQ_V0), "metadata": R.relative(MNQ_V0.with_suffix(".metadata.json")),
                 "builder": "research/mnq_v0_relock_20261121/build_v0_sleeves.py"}
    if data_meta["sha256"] != build["data"]["sha256"]:
        raise RuntimeError("MNQ v.0 file differs from the one the sleeves were built on")
    sources: dict[str, Any] = {}
    for sleeve in SLEEVES:
        if not sleeve.startswith("mnq_"):
            src = prev_config["sleeve_sources"][sleeve]
            if R.sha256(REPO / src["daily_source"]) != src["daily_source_sha256"]:
                raise RuntimeError(f"frozen source changed: {src['daily_source']}")
            sources[sleeve] = src
            continue
        d = REPO / "research" / DIRS[sleeve]
        files = {name: R.relative(d / name) for name in ("daily_results.csv", "trades.csv", "summary.json")}
        src = {k: v for k, v in prev_config["sleeve_sources"][sleeve].items() if not k.startswith("daily_source")
               and k not in ("trades", "trades_sha256", "summary", "summary_sha256", "candidate_votes", "candidate_votes_sha256",
                             "reproduced_from_raw_bars", "rebuild_script", "rebuild_script_sha256")}
        src.update({"data": data_meta["path"], "data_sha256": data_meta["sha256"], "rebuilt_by": data_meta["builder"],
                    "daily_source": files["daily_results.csv"], "daily_source_sha256": R.sha256(d / "daily_results.csv"),
                    "trades": files["trades.csv"], "trades_sha256": R.sha256(d / "trades.csv"),
                    "summary": files["summary.json"], "summary_sha256": R.sha256(d / "summary.json"),
                    "c0_source_retained": prev_config["sleeve_sources"][sleeve]["daily_source"]})
        if "ensemble_manifest" in src and R.sha256(REPO / src["ensemble_manifest"]) != src["ensemble_manifest_sha256"]:
            raise RuntimeError(f"frozen manifest changed: {src['ensemble_manifest']}")
        if sleeve == S01:
            src["thresholds_is"] = "research/signal_sheet_01_atr18_v0/thresholds_is.csv"
            src["thresholds_is_sha256"] = R.sha256(REPO / src["thresholds_is"])
            src["execution_requirement"] = "use thresholds_is.csv as frozen (calibrated on c.0 IS, copied unchanged); never recalibrate"
        sources[sleeve] = src
    shadow = dict(prev_config["shadow_sleeves"])
    s4 = REPO / "research" / DIRS[SHADOW]
    shadow[SHADOW] = dict(shadow[SHADOW], daily_source=R.relative(s4 / "daily_results.csv"),
                          daily_source_sha256=R.sha256(s4 / "daily_results.csv"), data=data_meta["path"],
                          c0_source_retained=shadow[SHADOW]["daily_source"])
    manifest = pd.read_csv(PREV / "sleeve_manifest.csv")
    manifest.loc[manifest["sleeve"].eq(S01), "source"] = sources[S01]["trades"]
    manifest["data"] = np.where(manifest["instrument"].eq("MNQ"), data_meta["path"], "unchanged (2026-11-15 sources)")
    manifest.to_csv(HERE / "sleeve_manifest.csv", index=False)

    def hb(name: str, horizon: int) -> dict[str, float]:
        row = pooled.loc[(name, horizon)]
        return {k: float(row[k]) for k in ("pass_probability", "mll_probability", "expiry_probability")}

    risk = books["eight_20261121"]["risk"]
    eq = np.r_[0.0, out["portfolio_pnl_usd"].cumsum().to_numpy()]
    config = {
        "schema_version": "1.0", "portfolio_id": PORTFOLIO_ID, "status": "LOCKED_ACTIVE_REFERENCE",
        "decision_date": DECISION_DATE,
        "decision_authority": "explicit_user_decision (2026-11-21: 'freeze le mnq v.0 au lieu de c.0' ; choices: back-adjusted v.0, same window, stop-rule thresholds recomputed on v.0)",
        "supersedes_active_reference": SUPERSEDED_ID, "superseded_reference_retained_for_audit": R.relative(PREV),
        "changes_vs_superseded": [
            "MNQ data: MNQ.c.0 replaced by MNQ.v.0 back-adjusted by the roll spread (data/processed/parquet/MNQ_v_0_adj_1m_20190505_20260320.parquet); S01-S04 rebuilt with unchanged engines and definitions",
            "MNQ sleeves now trade the quarterly expiry Fridays and the roll weeks on the contract production trades; the artificial roll gap no longer enters the daily ATR",
            "stop-rule thresholds of the MNQ sleeves recomputed with the 2026-11-03 rule on the v.0 histories (MGC/MBT unchanged, checked equal)",
        ],
        "allocation_family": "equal_contract", "mapping": {name: 1 for name in SLEEVES},
        "portfolio_rule": prev_config["portfolio_rule"],
        "mnq_data": data_meta, "sleeve_sources": sources, "shadow_sleeves": shadow,
        "calendar": {"rule": prev_config["calendar"]["rule"], "window": list(WINDOW), "sessions": int(len(union)),
                     "note": "same 572 sessions as the 2026-11-15 lock; MNQ now has PnL on the 8 expiry Fridays"},
        "book_window_metrics": {"net_pnl_usd": risk["net_pnl_usd"], "daily_sharpe": risk["daily_sharpe"],
                                "worst_day_usd": risk["worst_day_usd"],
                                "max_close_drawdown_usd": float((eq - np.maximum.accumulate(eq)).min()),
                                "max_intraday_drawdown_usd": risk["max_intraday_drawdown_usd"]},
        "topstep_reference": {**{k: prev_config["topstep_reference"][k] for k in ("account", "rules", "method", "seed")},
                              "paths": int(pooled.loc[("eight_20261121", 60), "paths"]), "window": list(WINDOW),
                              "path_sampler": "sessions of the union calendar where at least one active sleeve of the v.0 book trades",
                              "horizon_35d": hb("eight_20261121", 35), "horizon_60d": hb("eight_20261121", 60)},
        "comparison_same_paths": {"eight_20261115_c0": {"horizon_35d": hb("eight_20261115_c0", 35), "horizon_60d": hb("eight_20261115_c0", 60),
                                                        "risk": books["eight_20261115_c0"]["risk"]},
                                  "file": R.relative(HERE / "topstep_summary.csv")},
        "sleeve_stop_rule": {**{k: v for k, v in prev_config["sleeve_stop_rule"].items() if k not in ("note", "thresholds")},
                             "thresholds": R.relative(HERE / "sleeve_stop_rule_thresholds.csv"),
                             "note": "MNQ thresholds recomputed on the v.0 histories (user decision 2026-11-21); the c.0 values are kept in column threshold_usd_c0_20261103"},
        "execution_rule": prev_config["execution_rule"], "netting_check": R.relative(HERE / "netting_check.json"),
        "governance": {**prev_config["governance"],
                       "sleeve_definition_change": "none: data change only (MNQ c.0 -> v.0 back-adjusted); engines, parameters, vote manifests and S01 thresholds unchanged",
                       "data_note": "v.0 back-adjusted prices before each roll are shifted by the cumulated roll spreads: price levels in the MNQ ledgers before 2026-03 are not traded prices; PnL in points is exact",
                       "data_discipline": "same window as the 2026-11-15 lock: nothing after 2026-03-20 09:29 ET was used; the windows of audit_20261115 options (O) and (P) stay unread on v.0"},
        "locked_inputs": {"superseded_lock_manifest": {"path": R.relative(PREV / "lock_manifest.json"), "sha256": R.sha256(PREV / "lock_manifest.json")},
                          "superseded_relock_script": {"path": R.relative(PREV / "relock_eight_sleeves_20261115.py"), "sha256": R.sha256(PREV / "relock_eight_sleeves_20261115.py")},
                          "v0_builder": {"path": data_meta["builder"], "sha256": R.sha256(REPO / data_meta["builder"])},
                          "netting_study_script": {"path": R.relative(NETTING / "run_study.py"), "sha256": R.sha256(NETTING / "run_study.py")},
                          "aggregator": {"path": "src/portfolio/aggregator.py", "sha256": R.sha256(REPO / "src" / "portfolio" / "aggregator.py")}},
    }
    (HERE / "locked_eight_sleeve_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    files = ("locked_eight_sleeve_config.json", "daily_pnl.csv", "sleeve_manifest.csv", "topstep_reference_metrics.csv",
             "sleeve_stop_rule_thresholds.csv", "netting_check.json")
    (HERE / "lock_manifest.json").write_text(json.dumps({"portfolio_id": PORTFOLIO_ID, "status": "LOCKED_ACTIVE_REFERENCE",
                                                         "files": {R.relative(HERE / n): R.sha256(HERE / n) for n in files}}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"v0": {"35d": hb("eight_20261121", 35), "60d": hb("eight_20261121", 60), "risk": risk},
                      "c0_same_paths": {"35d": hb("eight_20261115_c0", 35), "60d": hb("eight_20261115_c0", 60),
                                        "risk": books["eight_20261115_c0"]["risk"]},
                      "netting": netting}, indent=2, default=float))


if __name__ == "__main__":
    main()
