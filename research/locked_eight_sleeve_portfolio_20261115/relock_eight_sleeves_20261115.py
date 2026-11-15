"""Versioned eight-sleeve lock (2026-11-15): S01 is the 18-model ATR ensemble, daily PnL on the union calendar.

Two documented defects of ``eight_sleeve_equal_contract_reference_20261103`` are corrected, nothing else:

1. ``mnq_orb_vwap_atr`` (S01) was the legacy ``baseline_fixed`` ledger (ATR 14 x q_high {90, 95}, 823 trades),
   whereas the reference sleeve is the one of signal sheet 01: the 18-model ATR vote
   (ATR {14, 16, 18} x q_low {20, 25, 30} x q_high {80, 85}, 701 trades, IS-calibrated thresholds frozen in
   ``research/signal_sheet_01_atr18/thresholds_is.csv``).  User decision of 2026-11-15 (audit_20261115, phase 0).
2. The daily PnL was aligned on the MNQ calendar, which drops the quarterly-expiry Fridays traded by MGC and MBT
   (8 sessions, -509.26 USD on 2024-01-02 -> 2026-03-19).  The daily PnL is now on the union of the three
   instruments' calendars (572 sessions).

The other seven active sleeves keep their 2026-11-01 frozen vote sources byte for byte; S04 stays in shadow; one
whole contract per sleeve, S01 at its native 1-4 contracts.  Nothing is retuned.

Checks before anything is written:
- the 2026-10-04, 2026-11-01 and 2026-11-03 books are rebuilt on the MNQ calendar and must reproduce their
  published Topstep PASS rates on the same bootstrap paths;
- the S01 ledger must reconcile minute paths to daily PnL, and its sleeve stop-rule threshold must equal the one
  already registered on 2026-11-03 (which was already computed on the ATR 18 history);
- the netting engine must reproduce the separate gross PnL of the new book on 2024-01-02 -> 2026-03-19.

Data discipline: S01 ATR 18 is NOT computed on 2026-03-20 -> 2026-10-02.  That window is reserved by option (O)
of audit_20261115/preregistration.md, pending the user's validation.  Previous lock directories are never modified.
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
OLD8 = REPO / "research" / "locked_eight_sleeve_portfolio_20261103"
ATR18 = REPO / "research" / "signal_sheet_01_atr18"
NETTING = REPO / "research" / "book_aggregation_risk_r1_20261112"
PORTFOLIO_ID = "eight_sleeve_equal_contract_reference_20261115"
SUPERSEDED_ID = "eight_sleeve_equal_contract_reference_20261103"
DECISION_DATE = "2026-11-15"
S01 = "mnq_orb_vwap_atr"
WINDOW = ("2024-01-02", "2026-03-19")
ET = "America/New_York"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = _load("relock_eight_sleeves_20261103", OLD8 / "relock_eight_sleeves.py")
R, T = E.R, E.T
SLEEVES = E.SLEEVES
SHADOW = E.SHADOW


# ----------------------------------------------------------------------------- S01 reference ledger
def s01_ledger() -> pd.DataFrame:
    t = pd.read_csv(ATR18 / "trades.csv")
    t["entry_ts_utc"] = pd.to_datetime(t["entry_time"], utc=True)
    t["exit_ts_utc"] = pd.to_datetime(t["exit_time"], utc=True)
    direction = np.where(t["direction"].str.lower() == "long", 1, -1)
    t["signed_quantity"] = direction * t["quantity"]
    if len(t) != 701 or abs(float(t["net_pnl_usd"].sum()) - 19416.5) > 1e-6:
        raise RuntimeError("signal_sheet_01_atr18 ledger is not the 701-trade reference")
    return t


def s01_daily(calendar: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    daily = pd.read_csv(ATR18 / "daily_results.csv", parse_dates=["session_date"]).set_index("session_date")["daily_pnl_usd"]
    off = daily[~daily.index.isin(calendar) & (daily.index >= calendar.min()) & (daily.index <= calendar.max())]
    if (off != 0).any():
        raise RuntimeError("S01 has PnL on a session outside the calendar")
    t = s01_ledger()
    counts = t.groupby(pd.to_datetime(t["session_date"])).size()
    return daily.reindex(calendar).fillna(0.0).rename(S01), counts.reindex(calendar).fillna(0).astype(int).rename(S01)


# ----------------------------------------------------------------------------- calendars
def union_calendar() -> pd.DatetimeIndex:
    dates: set[pd.Timestamp] = set()
    for _, directory in R.VOTE_SLEEVES.values():
        d = pd.read_csv(REPO / "research" / directory / "daily_results.csv", usecols=["session_date"], parse_dates=["session_date"])
        dates |= set(d["session_date"])
    cal = pd.DatetimeIndex(sorted(dates))
    return cal[(cal >= WINDOW[0]) & (cal <= WINDOW[1])]


def extend_timeline(timeline: pd.DatetimeIndex, extra_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Same 08:20 -> 16:08 ET minute grid (469 minutes) as the 2026-10-04 timeline, for the added sessions."""
    local = timeline.tz_convert(ET)
    first = local[0]
    minutes = pd.Series(local).groupby(local.date).size()
    if set(minutes) != {469} or first.strftime("%H:%M") != "08:20":
        raise RuntimeError("unexpected timeline grid")
    added = []
    for day in extra_days:
        start = pd.Timestamp(f"{day.date()} 08:20", tz=ET)
        added.append(pd.date_range(start, periods=469, freq="min"))
    new = timeline.append(pd.DatetimeIndex(np.concatenate([a.tz_convert("UTC").values for a in added])).tz_localize("UTC"))
    return new.sort_values()


# ----------------------------------------------------------------------------- netting reconciliation
def netting_check(union_daily: pd.DataFrame) -> dict[str, Any]:
    rs = _load("netting_run_study_20261115_lock", NETTING / "run_study.py")
    spec = rs.load_spec()
    spec["sleeves"]["S01"] = dict(spec["sleeves"]["S01"], trades="research/signal_sheet_01_atr18/trades.csv")
    trades = rs.load_trades(spec)
    calendars = rs.load_calendars(spec)[0]
    dates = rs.union_sessions(calendars, rs.d(WINDOW[0]), rs.d(WINDOW[1]))
    end = pd.Timestamp(WINDOW[1], tz=ET) + pd.Timedelta(days=1)
    bars = {}
    import pyarrow.parquet as pq
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
    separate_gross = float(in_window["gross_native"].sum())
    netted_gross = float(sum(r.gross for r in res))
    if len(dates) != len(union_daily) or abs(separate_gross - netted_gross) > 1e-6:
        raise RuntimeError(f"netting reconciliation failed: {separate_gross} vs {netted_gross}")
    if abs(float(in_window["net_native"].sum()) - float(union_daily[SLEEVES].sum().sum())) > 1e-6:
        raise RuntimeError("netting trade ledger differs from the lock daily PnL")
    return {"window": list(WINDOW), "sessions": len(dates),
            "separate_net_usd": float(in_window["net_native"].sum()), "netted_net_usd": float(sum(r.final for r in res)),
            "gross_usd": netted_gross, "separate_costs_usd": float(in_window["cost_native"].sum()),
            "netted_costs_usd": float(sum(r.costs for r in res)),
            "peak_lots_uncapped": max(r.peak_units for r in res) / 10.0,
            "engine": "src/portfolio/aggregator.py (unchanged), S01 ledger = research/signal_sheet_01_atr18/trades.csv"}


# ----------------------------------------------------------------------------- main
def main() -> None:
    six_pnl, six_counts, six_trades = T.load_six()
    calendar = six_pnl.index
    mbt_nominal, _, mbt_counts, _ = T.load_mbt(calendar)
    timeline_source = pd.read_csv(T.SIX_DIR / "common_timestamp_portfolio.csv", usecols=["timestamp_utc"], parse_dates=["timestamp_utc"])
    timeline = pd.DatetimeIndex(timeline_source["timestamp_utc"]).tz_convert("UTC")
    timeline_dates = pd.Series(pd.to_datetime(timeline.tz_convert(ET).date), index=timeline)

    # --- 1. Rebuild the three previous books exactly as in the 2026-11-03 relock (MNQ calendar).
    old_daily = six_pnl[T.OLD].join(mbt_nominal[T.NEW], how="left").fillna(0.0)
    old_counts = six_counts[T.OLD].join(mbt_counts[T.NEW], how="left").fillna(0).astype(int)
    six_path, six_adverse, _ = T.six_intraday_paths(timeline, six_trades)
    _, _, mbt_adverse = T.load_mbt_minute(timeline, "nominal")
    R.reconcile(six_path, six_pnl, timeline_dates, "superseded six")
    old_adverse = six_adverse.join(mbt_adverse[T.NEW], how="left").fillna(0.0)

    nine_daily = pd.DataFrame(index=calendar)
    nine_counts = pd.DataFrame(index=calendar)
    nine_daily[S01] = six_pnl[S01]
    nine_counts[S01] = six_counts[S01]
    ledgers = {}
    for sleeve, (_, directory) in R.VOTE_SLEEVES.items():
        results = pd.read_csv(REPO / "research" / directory / "daily_results.csv", parse_dates=["session_date"]).set_index("session_date")
        nine_daily[sleeve] = results["daily_pnl_usd"].reindex(calendar).fillna(0.0)
        nine_counts[sleeve] = results["trade_count"].reindex(calendar).fillna(0).astype(int)
        ledgers[sleeve] = R.vote_trades(directory)
    nine_daily, nine_counts = nine_daily[E.NINE_SLEEVES], nine_counts[E.NINE_SLEEVES]
    vote_path, vote_adverse = R.intraday_paths(timeline, ledgers)
    R.reconcile(vote_path, nine_daily, timeline_dates, "vote ensembles")
    nine_adverse = six_adverse[[S01]].join(vote_adverse)[E.NINE_SLEEVES]
    published8 = pd.read_csv(OLD8 / "daily_pnl.csv", parse_dates=["session_date"]).set_index("session_date")
    if float((published8[SLEEVES] - nine_daily[SLEEVES]).abs().max().max()) > 1e-6:
        raise RuntimeError("Rebuilt ledger differs from the 2026-11-03 lock")

    # S01 reference on the MNQ calendar (isolates the S01 change on identical paths).
    s01_mnq, s01_mnq_counts = s01_daily(calendar)
    s01_path, s01_adverse = R.intraday_paths(timeline, {S01: s01_ledger()})
    R.reconcile(s01_path, s01_mnq.to_frame(), timeline_dates, "S01 ATR 18, MNQ calendar")
    ref_mnq_daily, ref_mnq_counts, ref_mnq_adverse = nine_daily.copy(), nine_counts.copy(), nine_adverse.copy()
    ref_mnq_daily[S01], ref_mnq_counts[S01], ref_mnq_adverse[S01] = s01_mnq, s01_mnq_counts, s01_adverse[S01]

    variants = {
        "superseded_20261004": E.book(old_daily, old_adverse, old_counts, timeline_dates, T.ALL),
        "nine_20261101": E.book(nine_daily, nine_adverse, nine_counts, timeline_dates, E.NINE_SLEEVES),
        "eight_20261103": E.book(nine_daily, nine_adverse, nine_counts, timeline_dates, SLEEVES),
        "eight_s01_atr18_mnq_calendar": E.book(ref_mnq_daily, ref_mnq_adverse, ref_mnq_counts, timeline_dates, SLEEVES),
    }
    sampler_traded = six_counts[T.OLD].gt(0).any(axis=1).to_numpy(bool)
    harness = R.bootstrap(variants, sampler_traded)
    hp = harness[harness["block"].astype(str).eq("pooled")].set_index(["scenario", "horizon"])
    old8_ref = pd.read_csv(OLD8 / "topstep_reference_metrics.csv").set_index("horizon")["pass_probability"]
    checks = {("superseded_20261004", 60): 0.452375, ("superseded_20261004", 35): 0.35025,
              ("nine_20261101", 60): 0.3895, ("nine_20261101", 35): 0.308,
              ("eight_20261103", 60): float(old8_ref.loc[60]), ("eight_20261103", 35): float(old8_ref.loc[35])}
    for key, expected in checks.items():
        if abs(float(hp.loc[key, "pass_probability"]) - expected) > 1e-12:
            raise RuntimeError(f"Harness check failed for {key}: {hp.loc[key, 'pass_probability']} instead of {expected}")

    # --- 2. New reference: S01 ATR 18, union calendar.
    union = union_calendar()
    extra = union.difference(calendar)
    if len(union) != 572 or len(extra) != 8 or len(calendar.difference(union)):
        raise RuntimeError(f"union calendar: {len(union)} sessions, {len(extra)} added")
    daily_u = pd.DataFrame(index=union)
    counts_u = pd.DataFrame(index=union)
    daily_u[S01], counts_u[S01] = s01_daily(union)
    for sleeve, (_, directory) in R.VOTE_SLEEVES.items():
        results = pd.read_csv(REPO / "research" / directory / "daily_results.csv", parse_dates=["session_date"]).set_index("session_date")
        daily_u[sleeve] = results["daily_pnl_usd"].reindex(union).fillna(0.0)
        counts_u[sleeve] = results["trade_count"].reindex(union).fillna(0).astype(int)
    for sleeve in SLEEVES:  # union daily == trade sum of the window, per sleeve
        ledger = s01_ledger() if sleeve == S01 else ledgers[sleeve]
        sd = pd.to_datetime(ledger["session_date"])
        total = float(ledger.loc[(sd >= WINDOW[0]) & (sd <= WINDOW[1]), "net_pnl_usd"].sum())
        if abs(total - float(daily_u[sleeve].sum())) > 1e-6:
            raise RuntimeError(f"{sleeve}: union daily PnL differs from its trades")
    timeline_u = extend_timeline(timeline, extra)
    timeline_dates_u = pd.Series(pd.to_datetime(timeline_u.tz_convert(ET).date), index=timeline_u)
    u_ledgers = {S01: s01_ledger(), **{s: ledgers[s] for s in SLEEVES if s != S01}}
    path_u, adverse_u = R.intraday_paths(timeline_u, u_ledgers)
    R.reconcile(path_u, daily_u[SLEEVES], timeline_dates_u, "reference book, union calendar")
    reference = {"eight_20261115": E.book(daily_u, adverse_u[SLEEVES], counts_u, timeline_dates_u, SLEEVES)}
    sampler_u = counts_u[SLEEVES].gt(0).any(axis=1).to_numpy(bool)
    summary = R.bootstrap(reference, sampler_u)
    pooled = summary[summary["block"].astype(str).eq("pooled")].set_index(["scenario", "horizon"])

    # --- 3. Stop rule: the 2026-11-03 thresholds were already computed on the ATR 18 history of S01.
    thresholds = E.stop_rule_thresholds()
    registered = pd.read_csv(OLD8 / "sleeve_stop_rule_thresholds.csv")
    if not np.allclose(thresholds["threshold_usd"], registered["threshold_usd"], atol=1e-9) or \
            list(thresholds["sleeve"]) != list(registered["sleeve"]):
        raise RuntimeError("stop-rule thresholds differ from the ones registered on 2026-11-03")

    # --- 4. Netting engine on the new book.
    netting = netting_check(daily_u)

    # --- write
    daily = daily_u[SLEEVES].copy()
    daily["portfolio_pnl_usd"] = daily[SLEEVES].sum(axis=1)
    daily.index.name = "session_date"
    daily.reset_index().to_csv(HERE / "daily_pnl.csv", index=False, float_format="%.8f")
    summary.to_csv(HERE / "topstep_summary.csv", index=False)
    harness.to_csv(HERE / "topstep_harness_mnq_calendar.csv", index=False)
    ref = pooled.loc["eight_20261115"].reset_index().assign(method="moving_block_bootstrap",
                                                            scenario="equal_contract__nominal__eight__s01_atr18__union", block="pooled")
    ref = ref[["method", "scenario", "horizon", "block", "paths", "pass_probability", "mll_probability",
               "expiry_probability", "median_days_to_pass", "median_max_drawdown_usd", "p05_max_drawdown_usd"]]
    ref.to_csv(HERE / "topstep_reference_metrics.csv", index=False)
    risk_rows = [{"book": n, "calendar": "MNQ (564)", **v["risk"]} for n, v in variants.items()]
    risk_rows.append({"book": "eight_20261115", "calendar": "union (572)", **reference["eight_20261115"]["risk"]})
    pd.DataFrame(risk_rows).to_csv(HERE / "book_risk_comparison.csv", index=False)
    thresholds.to_csv(HERE / "sleeve_stop_rule_thresholds.csv", index=False)
    (HERE / "netting_check.json").write_text(json.dumps(netting, indent=2) + "\n", encoding="utf-8")

    old_config = json.loads((OLD8 / "locked_eight_sleeve_config.json").read_text(encoding="utf-8"))
    sleeve_sources = {name: old_config["sleeve_sources"][name] for name in SLEEVES if name != S01}
    for source in sleeve_sources.values():
        if R.sha256(REPO / source["daily_source"]) != source["daily_source_sha256"]:
            raise RuntimeError(f"Frozen source changed since 2026-11-01: {source['daily_source']}")
        if R.sha256(REPO / source["ensemble_manifest"]) != source["ensemble_manifest_sha256"]:
            raise RuntimeError(f"Frozen manifest changed since 2026-11-01: {source['ensemble_manifest']}")
    s01_files = {key: f"research/signal_sheet_01_atr18/{name}" for key, name in
                 (("daily_source", "daily_results.csv"), ("trades", "trades.csv"), ("thresholds_is", "thresholds_is.csv"),
                  ("summary", "summary.json"), ("candidate_votes", "candidate_votes.csv"))}
    s01_files["rebuild_script"] = "scripts/rebuild_signal_sheet_01_atr18.py"
    s01_source = {"type": "frozen_atr_vote_ensemble",
                  "definition": "ORB15 long-only + continuous VWAP; ATR vote of 18 sub-models ATR {14,16,18} x q_low {20,25,30} x q_high {80,85}, thresholds calibrated on IS 2019-05-05 -> 2024-02-27 only, trade when >= 9 of 18 votes; stop OR_low - 2 ticks, target 2R, exit 16:00; 0.5 % of 50 000 USD risk, 1-4 contracts",
                  "signal_sheet": "docs/signal_sheets/01_mnq_orb_vwap_atr.md",
                  "execution_requirement": "use thresholds_is.csv as frozen; never recalibrate (the 70/30 split of the rebuild script would move on a longer file)",
                  "reproduced_from_raw_bars": "audit_20261115/phase0_engine.py on 2026-11-15: 701/701 trades identical, 0.00 USD",
                  **{k: v for k, v in s01_files.items()},
                  **{f"{k}_sha256": R.sha256(REPO / v) for k, v in s01_files.items()}}
    sleeve_sources = {S01: s01_source, **sleeve_sources}
    manifest = pd.read_csv(OLD8 / "sleeve_manifest.csv")
    manifest.loc[manifest["sleeve"].eq(S01), ["source_type", "source"]] = ["frozen_atr_vote_ensemble", s01_files["trades"]]
    manifest.to_csv(HERE / "sleeve_manifest.csv", index=False)

    def hb(frame: pd.DataFrame, name: str, horizon: int) -> dict[str, float]:
        row = frame.loc[(name, horizon)]
        return {key: float(row[key]) for key in ("pass_probability", "mll_probability", "expiry_probability")}

    risk_ref = reference["eight_20261115"]["risk"]
    config = {
        "schema_version": "1.0",
        "portfolio_id": PORTFOLIO_ID,
        "status": "LOCKED_ACTIVE_REFERENCE",
        "decision_date": DECISION_DATE,
        "decision_authority": "explicit_user_promotion (2026-11-15: 'le S01 de référence c'est bien celui de la fiche actuelle' ; 'fais le nouveau lock versionné')",
        "supersedes_active_reference": SUPERSEDED_ID,
        "superseded_reference_retained_for_audit": R.relative(OLD8),
        "changes_vs_superseded": [
            "mnq_orb_vwap_atr: legacy baseline_fixed ledger (ATR 14 x q_high {90,95}, 823 trades) replaced by the 18-model ATR vote of signal sheet 01 (701 trades)",
            "daily PnL on the union of the MNQ, MGC and MBT session calendars (572 sessions on 2024-01-02 -> 2026-03-19) instead of the MNQ calendar (564): adds 8 quarterly-expiry Fridays traded by MGC/MBT (-509.26 USD)",
        ],
        "allocation_family": "equal_contract",
        "mapping": {name: 1 for name in SLEEVES},
        "portfolio_rule": "sum the executable PnL of one whole-contract copy of each frozen sleeve (S01 at its native 1-4 contracts); no fractional scaling",
        "sleeve_sources": sleeve_sources,
        "shadow_sleeves": old_config["shadow_sleeves"],
        "calendar": {"rule": "union of the instruments' session calendars; a sleeve is 0 on a union session its instrument does not trade",
                     "window": list(WINDOW), "sessions": int(len(union)),
                     "added_vs_mnq_calendar": [str(x.date()) for x in extra],
                     "known_data_gap": "the MNQ c.0 minute file has no bars on quarterly expiry Fridays; MNQ sleeves have never been backtested on those days"},
        "book_window_metrics": {"net_pnl_usd": risk_ref["net_pnl_usd"], "daily_sharpe": risk_ref["daily_sharpe"],
                                "worst_day_usd": risk_ref["worst_day_usd"],
                                "max_close_drawdown_usd": float((lambda e: (e - np.maximum.accumulate(e)).min())(np.r_[0.0, daily["portfolio_pnl_usd"].cumsum().to_numpy()])),
                                "max_intraday_drawdown_usd": risk_ref["max_intraday_drawdown_usd"]},
        "topstep_reference": {
            "account": "Trading Combine 50K",
            "rules": {"profit_target_usd": T.PROFIT_TARGET, "mll_usd": T.MLL_DISTANCE, "consistency": T.CONSISTENCY},
            "method": "moving_block_bootstrap_pooled_blocks_3_5_10_20",
            "paths": int(pooled.loc[("eight_20261115", 60), "paths"]),
            "seed": T.SEED,
            "window": list(WINDOW),
            "path_sampler": "sessions of the union calendar where at least one active sleeve trades (own sampler, so paths differ from the previous locks')",
            "horizon_35d": hb(pooled, "eight_20261115", 35),
            "horizon_60d": hb(pooled, "eight_20261115", 60),
        },
        "comparison_on_previous_paths": {
            "calendar": "MNQ (564 sessions), six-sleeve traded-day sampler of the 2026-10-04 study, identical paths",
            "eight_20261103": {"horizon_35d": hb(hp, "eight_20261103", 35), "horizon_60d": hb(hp, "eight_20261103", 60)},
            "eight_s01_atr18_mnq_calendar": {"horizon_35d": hb(hp, "eight_s01_atr18_mnq_calendar", 35),
                                             "horizon_60d": hb(hp, "eight_s01_atr18_mnq_calendar", 60)},
            "nine_20261101": {"horizon_35d": hb(hp, "nine_20261101", 35), "horizon_60d": hb(hp, "nine_20261101", 60)},
            "superseded_20261004": {"horizon_35d": hb(hp, "superseded_20261004", 35), "horizon_60d": hb(hp, "superseded_20261004", 60)},
            "file": R.relative(HERE / "topstep_harness_mnq_calendar.csv"),
        },
        "sleeve_stop_rule": {**old_config["sleeve_stop_rule"], "thresholds": R.relative(HERE / "sleeve_stop_rule_thresholds.csv"),
                             "note": "unchanged: the 2026-11-03 thresholds were already computed on the ATR 18 history of S01 (checked equal)"},
        "execution_rule": "strategy_registry/execution/netting_rule_20261115_r2.json",
        "netting_check": R.relative(HERE / "netting_check.json"),
        "governance": {
            "no_retuning": True,
            "whole_contracts_only": True,
            "pass_rate_optimization_performed": False,
            "sleeve_definition_change": "S01 only, to the signal-sheet definition the user designates as reference; the seven other active sleeves keep their 2026-11-01 sources byte for byte",
            "prospective_start_after": "2026-11-03",
            "prospective_start_note": "unchanged from the 2026-11-03 lock: no session after 2026-10-02 had been loaded when this lock was made, and every sleeve definition predates 2026-11-03; consistent with audit_20261115/preregistration.md (P)",
            "data_discipline": "S01 ATR 18 is not evaluated on 2026-03-20 -> 2026-10-02: reserved by option (O) of audit_20261115/preregistration.md",
            "known_issue": old_config["governance"]["known_issue"],
            "caveat": "The 2024-01-02 -> 2026-03-19 window and the 2026-03-20 -> 2026-10-02 window were both inspected before this lock; Topstep figures are retrospective bootstrap estimates, not validations.",
        },
        "locked_inputs": {
            "superseded_lock_manifest": {"path": R.relative(OLD8 / "lock_manifest.json"), "sha256": R.sha256(OLD8 / "lock_manifest.json")},
            "relock_harness_script": {"path": R.relative(OLD8 / "relock_eight_sleeves.py"), "sha256": R.sha256(OLD8 / "relock_eight_sleeves.py")},
            "netting_study_script": {"path": R.relative(NETTING / "run_study.py"), "sha256": R.sha256(NETTING / "run_study.py")},
            "aggregator": {"path": "src/portfolio/aggregator.py", "sha256": R.sha256(REPO / "src" / "portfolio" / "aggregator.py")},
        },
    }
    (HERE / "locked_eight_sleeve_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    files = ("locked_eight_sleeve_config.json", "daily_pnl.csv", "sleeve_manifest.csv", "topstep_reference_metrics.csv",
             "sleeve_stop_rule_thresholds.csv", "netting_check.json")
    lock_manifest = {"portfolio_id": PORTFOLIO_ID, "status": "LOCKED_ACTIVE_REFERENCE",
                     "files": {R.relative(HERE / name): R.sha256(HERE / name) for name in files}}
    (HERE / "lock_manifest.json").write_text(json.dumps(lock_manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"reference": {"35d": hb(pooled, "eight_20261115", 35), "60d": hb(pooled, "eight_20261115", 60)},
                      "same_paths": {n: {"35d": hb(hp, n, 35), "60d": hb(hp, n, 60)} for n in variants},
                      "risk": {**{n: v["risk"] for n, v in variants.items()}, "eight_20261115": risk_ref},
                      "netting": netting}, indent=2, default=float))


if __name__ == "__main__":
    main()
