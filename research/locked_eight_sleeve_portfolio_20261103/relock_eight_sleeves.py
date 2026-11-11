"""Versioned eight-sleeve lock (2026-11-03): mnq_atr_trend_flip moves to shadow.

The decision comes from the frozen-vote forward replay
(``research/frozen_vote_forward_replay_20261101``): on 2026-03-20 -> 2026-10-02
S04 lost 6,363.5 USD, about twice its worst 141-session window since 2019.  The
other eight sleeves are kept exactly as locked on 2026-11-01: same frozen
sources, one whole contract each, nothing retuned.

The Topstep 50K reference reuses the 2026-11-01 relock harness unchanged.  The
2026-10-04 and 2026-11-01 books are simulated on the same bootstrap paths and
must reproduce their published PASS rates before anything is written.  The
script also writes the pre-registered sleeve stop rule thresholds; previous
lock directories are never modified.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NINE = REPO / "research" / "locked_nine_sleeve_portfolio_20261101"
REPLAY = REPO / "research" / "frozen_vote_forward_replay_20261101"
PAPER_EXTENSION = REPO / "research" / "nine_sleeve_paper_extension_20261013" / "daily_nine_sleeves_through_20261002.csv"
REPLAY_KEYS = {"mnq_volatility_breakout": "s02", "mnq_compression_breakout": "s03", "mnq_atr_trend_flip": "s04",
               "mgc_volatility_breakout": "s05", "mgc_gap_continuation": "s06"}
PORTFOLIO_ID = "eight_sleeve_equal_contract_reference_20261103"
SUPERSEDED_ID = "nine_sleeve_equal_contract_reference_20261101"
DECISION_DATE = "2026-11-03"
SHADOW = "mnq_atr_trend_flip"
STOP_WINDOW = 126
FORWARD_START = "2026-03-20"

_spec = importlib.util.spec_from_file_location("relock_nine_sleeves_20261101", NINE / "relock_nine_sleeves.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
T = R.T

NINE_SLEEVES = R.SLEEVES
SLEEVES = [name for name in NINE_SLEEVES if name != SHADOW]
HISTORY = {"mnq_orb_vwap_atr": "signal_sheet_01_atr18",
           **{sleeve: directory for sleeve, (_, directory) in R.VOTE_SLEEVES.items()}}
MNQ_CALENDAR_SOURCE = "signal_sheet_02_cluster"


def book(daily: pd.DataFrame, adverse: pd.DataFrame, counts: pd.DataFrame, timeline_dates: pd.Series,
         sleeves: list[str]) -> dict[str, Any]:
    mapping = {name: 1 for name in sleeves}
    portfolio = T.portfolio_daily(daily, mapping)
    adverse_path = T.portfolio_path(adverse, mapping)
    intraday_min = adverse_path.groupby(timeline_dates, sort=False).min().reindex(daily.index).fillna(0.0)
    return {
        "daily": portfolio.to_numpy(float),
        "intraday_min": intraday_min.to_numpy(float),
        "traded": counts[sleeves].gt(0).any(axis=1).to_numpy(bool),
        "risk": T.loss_cluster_metrics(portfolio, adverse_path, timeline_dates),
    }


def sleeve_history(sleeve: str) -> pd.Series:
    """Full frozen daily history of one sleeve on its instrument's session calendar."""
    frame = pd.read_csv(REPO / "research" / HISTORY[sleeve] / "daily_results.csv", parse_dates=["session_date"])
    series = frame.set_index("session_date")["daily_pnl_usd"].astype(float)
    if sleeve == "mnq_orb_vwap_atr":
        # The ORB ledger carries non-session calendar rows; align it on the MNQ vote calendar.
        calendar = pd.read_csv(REPO / "research" / MNQ_CALENDAR_SOURCE / "daily_results.csv", parse_dates=["session_date"])["session_date"]
        series = series.reindex(pd.DatetimeIndex(calendar)).fillna(0.0)
    return series


def stop_rule_thresholds() -> pd.DataFrame:
    rows = []
    for sleeve in NINE_SLEEVES:
        history = sleeve_history(sleeve)
        rolling = history.rolling(STOP_WINDOW).sum().dropna()
        rows.append({
            "sleeve": sleeve, "status": "SHADOW_FORWARD_ONLY" if sleeve == SHADOW else "LOCKED_ACTIVE_REFERENCE",
            "history_start": str(history.index.min().date()), "history_end": str(history.index.max().date()),
            "history_sessions": int(len(history)), "window_sessions": STOP_WINDOW,
            "rolling_windows": int(len(rolling)),
            # The historical minimum never fires in-sample; a 1 % quantile would fire on ~1 % of past windows.
            "threshold_usd": float(rolling.min()),
            "historical_p01_usd": float(rolling.quantile(0.01)), "historical_median_usd": float(rolling.median()),
        })
    return pd.DataFrame(rows)


def forward_context(thresholds: pd.DataFrame) -> dict[str, Any]:
    """Descriptive MNQ/MGC view of the replay window that motivated the decision (not a validation)."""
    window = pd.read_csv(REPLAY / "forward_daily_six_new.csv", parse_dates=["session_date"]).set_index("session_date")

    def stats(pnl: pd.Series) -> dict[str, float]:
        equity = np.r_[0.0, pnl.cumsum().to_numpy()]
        return {"net_pnl_usd": float(pnl.sum()), "sharpe": float(np.sqrt(252.0) * pnl.mean() / pnl.std(ddof=1)),
                "max_drawdown_usd": float((equity - np.maximum.accumulate(equity)).min())}

    limits = thresholds.set_index("sleeve")["threshold_usd"]
    paper = pd.read_csv(PAPER_EXTENSION, parse_dates=["session_date"]).set_index("session_date")
    stop_check = {}
    for sleeve in ["mnq_orb_vwap_atr", *REPLAY_KEYS]:
        if sleeve == "mnq_orb_vwap_atr":
            pnl = paper[sleeve]
        else:
            pnl = pd.read_csv(REPLAY / f"{REPLAY_KEYS[sleeve]}_daily.csv", parse_dates=["session_date"]).set_index("session_date")["daily_pnl_usd"]
        rolling = pnl.loc[: window.index.max()].rolling(STOP_WINDOW).sum().loc[FORWARD_START:]
        breach = rolling[rolling < float(limits[sleeve])]
        stop_check[sleeve] = {
            "threshold_usd": float(limits[sleeve]), "forward_min_usd": float(rolling.min()), "last_usd": float(rolling.iloc[-1]),
            "breach_sessions": int(len(breach)),
            "breach_span": [str(breach.index.min().date()), str(breach.index.max().date())] if len(breach) else None,
            "in_breach_at_window_end": bool(rolling.iloc[-1] < float(limits[sleeve])),
        }
    return {
        "window": [str(window.index.min().date()), str(window.index.max().date())],
        "sessions": int(len(window)),
        "scope": "MNQ/MGC sleeves only; no MBT data after 2026-10-02",
        "mnq_mgc_with_s04": stats(window.sum(axis=1)),
        "mnq_mgc_without_s04": stats(window.drop(columns=SHADOW).sum(axis=1)),
        "stop_rule_replay": stop_check,
        "stop_rule_replay_note": "Descriptive: the rule is registered on 2026-11-03 and applies prospectively only.",
        "caveat": "This window motivated the removal; it cannot also validate it.",
    }


def main() -> None:
    six_pnl, six_counts, six_trades = T.load_six()
    calendar = six_pnl.index
    mbt_nominal, _, mbt_counts, _ = T.load_mbt(calendar)
    timeline_source = pd.read_csv(T.SIX_DIR / "common_timestamp_portfolio.csv", usecols=["timestamp_utc"], parse_dates=["timestamp_utc"])
    timeline = pd.DatetimeIndex(timeline_source["timestamp_utc"]).tz_convert("UTC")
    timeline_dates = pd.Series(pd.to_datetime(timeline.tz_convert("America/New_York").date), index=timeline)

    old_daily = six_pnl[T.OLD].join(mbt_nominal[T.NEW], how="left").fillna(0.0)
    old_counts = six_counts[T.OLD].join(mbt_counts[T.NEW], how="left").fillna(0).astype(int)
    six_path, six_adverse, _ = T.six_intraday_paths(timeline, six_trades)
    _, _, mbt_adverse = T.load_mbt_minute(timeline, "nominal")
    R.reconcile(six_path, six_pnl, timeline_dates, "superseded six")
    old_adverse = six_adverse.join(mbt_adverse[T.NEW], how="left").fillna(0.0)

    new_daily = pd.DataFrame(index=calendar)
    new_counts = pd.DataFrame(index=calendar)
    new_daily["mnq_orb_vwap_atr"] = six_pnl["mnq_orb_vwap_atr"]
    new_counts["mnq_orb_vwap_atr"] = six_counts["mnq_orb_vwap_atr"]
    ledgers = {}
    for sleeve, (_, directory) in R.VOTE_SLEEVES.items():
        results = pd.read_csv(REPO / "research" / directory / "daily_results.csv", parse_dates=["session_date"]).set_index("session_date")
        new_daily[sleeve] = results["daily_pnl_usd"].reindex(calendar).fillna(0.0)
        new_counts[sleeve] = results["trade_count"].reindex(calendar).fillna(0).astype(int)
        ledgers[sleeve] = R.vote_trades(directory)
    new_daily, new_counts = new_daily[NINE_SLEEVES], new_counts[NINE_SLEEVES]
    vote_path, vote_adverse = R.intraday_paths(timeline, ledgers)
    R.reconcile(vote_path, new_daily, timeline_dates, "vote ensembles")
    new_adverse = six_adverse[["mnq_orb_vwap_atr"]].join(vote_adverse)[NINE_SLEEVES]

    published = pd.read_csv(NINE / "daily_pnl.csv", parse_dates=["session_date"]).set_index("session_date")
    if float((published[NINE_SLEEVES] - new_daily).abs().max().max()) > 1e-6:
        raise RuntimeError("Rebuilt nine-sleeve ledger differs from the 2026-11-01 lock")

    variants = {
        "superseded_20261004": book(old_daily, old_adverse, old_counts, timeline_dates, T.ALL),
        "nine_20261101": book(new_daily, new_adverse, new_counts, timeline_dates, NINE_SLEEVES),
        "eight_20261103": book(new_daily, new_adverse, new_counts, timeline_dates, SLEEVES),
    }
    sampler_traded = six_counts[T.OLD].gt(0).any(axis=1).to_numpy(bool)
    summary = R.bootstrap(variants, sampler_traded)
    pooled = summary[summary["block"].astype(str).eq("pooled")].set_index(["scenario", "horizon"])
    checks = {("superseded_20261004", 60): 0.452375, ("superseded_20261004", 35): 0.35025,
              ("nine_20261101", 60): 0.3895, ("nine_20261101", 35): 0.308}
    for key, expected in checks.items():
        if abs(float(pooled.loc[key, "pass_probability"]) - expected) > 1e-12:
            raise RuntimeError(f"Harness check failed for {key}: {pooled.loc[key, 'pass_probability']} instead of {expected}")

    thresholds = stop_rule_thresholds()
    context = forward_context(thresholds)

    daily = new_daily[SLEEVES].copy()
    daily["portfolio_pnl_usd"] = daily[SLEEVES].sum(axis=1)
    daily.reset_index().rename(columns={"index": "session_date"}).to_csv(HERE / "daily_pnl.csv", index=False, float_format="%.8f")
    summary.to_csv(HERE / "topstep_summary.csv", index=False)
    reference = pooled.loc["eight_20261103"].reset_index().assign(method="moving_block_bootstrap", scenario="equal_contract__nominal__eight", block="pooled")
    reference = reference[["method", "scenario", "horizon", "block", "paths", "pass_probability", "mll_probability",
                           "expiry_probability", "median_days_to_pass", "median_max_drawdown_usd", "p05_max_drawdown_usd"]]
    reference.to_csv(HERE / "topstep_reference_metrics.csv", index=False)
    pd.DataFrame([{"book": name, **values["risk"]} for name, values in variants.items()]).to_csv(HERE / "book_risk_comparison.csv", index=False)
    thresholds.to_csv(HERE / "sleeve_stop_rule_thresholds.csv", index=False)
    (HERE / "forward_context.json").write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")

    nine_config = json.loads((NINE / "locked_nine_sleeve_config.json").read_text(encoding="utf-8"))
    sleeve_sources = {name: nine_config["sleeve_sources"][name] for name in SLEEVES}
    for source in sleeve_sources.values():
        if R.sha256(REPO / source["daily_source"]) != source["daily_source_sha256"]:
            raise RuntimeError(f"Frozen source changed since 2026-11-01: {source['daily_source']}")
    nine_manifest = pd.read_csv(NINE / "sleeve_manifest.csv")
    shadow_row = nine_manifest["sleeve"].eq(SHADOW)
    manifest = nine_manifest.assign(status=np.where(shadow_row, "SHADOW_FORWARD_ONLY", "LOCKED_ACTIVE_REFERENCE"),
                                    integer_copies=np.where(shadow_row, 0, 1))
    manifest.to_csv(HERE / "sleeve_manifest.csv", index=False)

    def horizon_block(name: str, horizon: int) -> dict[str, float]:
        row = pooled.loc[(name, horizon)]
        return {key: float(row[key]) for key in ("pass_probability", "mll_probability", "expiry_probability")}

    config = {
        "schema_version": "1.0",
        "portfolio_id": PORTFOLIO_ID,
        "status": "LOCKED_ACTIVE_REFERENCE",
        "decision_date": DECISION_DATE,
        "decision_authority": "explicit_user_promotion",
        "supersedes_active_reference": SUPERSEDED_ID,
        "superseded_reference_retained_for_audit": R.relative(NINE),
        "allocation_family": "equal_contract",
        "mapping": {name: 1 for name in SLEEVES},
        "portfolio_rule": "sum the executable PnL of one whole-contract copy of each frozen sleeve; no fractional scaling",
        "sleeve_sources": sleeve_sources,
        "shadow_sleeves": {SHADOW: {
            "status": "SHADOW_FORWARD_ONLY", "allocation": 0, **nine_config["sleeve_sources"][SHADOW],
            "removal_reason": "forward replay loss of 6,363.5 USD on 2026-03-20 -> 2026-10-02 (141 sessions), about twice its worst 141-session window since 2019; largest risk contributor after S02 (30 %) and correlated 0.57 with mnq_volatility_breakout",
            "evidence": R.relative(REPLAY / "forward_sleeve_comparison.csv"),
        }},
        "topstep_reference": {
            "account": "Trading Combine 50K",
            "rules": {"profit_target_usd": T.PROFIT_TARGET, "mll_usd": T.MLL_DISTANCE, "consistency": T.CONSISTENCY},
            "method": "moving_block_bootstrap_pooled_blocks_3_5_10_20",
            "paths": int(pooled.loc[("eight_20261103", 60), "paths"]),
            "seed": T.SEED,
            "window": [str(calendar.min().date()), str(calendar.max().date())],
            "path_sampler": "identical to the 2026-10-04 study (six-sleeve traded-day sampler, same RNG order)",
            "horizon_35d": horizon_block("eight_20261103", 35),
            "horizon_60d": horizon_block("eight_20261103", 60),
        },
        "comparison_with_superseded": {
            "same_bootstrap_paths": True,
            "nine_20261101": {"horizon_35d": horizon_block("nine_20261101", 35), "horizon_60d": horizon_block("nine_20261101", 60)},
            "superseded_20261004": {"horizon_35d": horizon_block("superseded_20261004", 35), "horizon_60d": horizon_block("superseded_20261004", 60)},
        },
        "sleeve_stop_rule": {
            "metric": f"sum of net daily PnL over the last {STOP_WINDOW} sessions of the sleeve's own calendar",
            "threshold": "minimum of the same rolling sum over the sleeve's frozen history (worse than anything already seen)",
            "action": "the sleeve moves to SHADOW_FORWARD_ONLY in a new versioned lock; no retuning",
            "reinstatement": "only by a new versioned decision after at least 252 shadow sessions",
            "thresholds": R.relative(HERE / "sleeve_stop_rule_thresholds.csv"),
            "registered_on": DECISION_DATE,
        },
        "forward_context": R.relative(HERE / "forward_context.json"),
        "governance": {
            "no_retuning": True,
            "whole_contracts_only": True,
            "pass_rate_optimization_performed": False,
            "sleeve_definition_change": "none: the eight remaining sleeves keep their 2026-11-01 frozen sources byte for byte",
            "prospective_start_after": DECISION_DATE,
            "known_issue": "mbt_volatility_breakout vote is flat on its legacy OOS window (Sharpe -0.040) despite positive cells; flagged for review",
            "caveat": "The removal of mnq_atr_trend_flip was decided after seeing the 2026-03-20 -> 2026-10-02 window; that window is no longer out-of-sample for this book. Topstep figures are retrospective bootstrap estimates on an already-inspected window.",
        },
        "locked_inputs": {
            "superseded_lock_manifest": {"path": R.relative(NINE / "lock_manifest.json"), "sha256": R.sha256(NINE / "lock_manifest.json")},
            "relock_harness_script": {"path": R.relative(NINE / "relock_nine_sleeves.py"), "sha256": R.sha256(NINE / "relock_nine_sleeves.py")},
            "forward_replay_comparison": {"path": R.relative(REPLAY / "forward_sleeve_comparison.csv"), "sha256": R.sha256(REPLAY / "forward_sleeve_comparison.csv")},
        },
    }
    (HERE / "locked_eight_sleeve_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    files = ("locked_eight_sleeve_config.json", "daily_pnl.csv", "sleeve_manifest.csv", "topstep_reference_metrics.csv",
             "sleeve_stop_rule_thresholds.csv")
    lock_manifest = {"portfolio_id": PORTFOLIO_ID, "status": "LOCKED_ACTIVE_REFERENCE",
                     "files": {R.relative(HERE / name): R.sha256(HERE / name) for name in files}}
    (HERE / "lock_manifest.json").write_text(json.dumps(lock_manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "eight": {"35d": horizon_block("eight_20261103", 35), "60d": horizon_block("eight_20261103", 60)},
        "nine": {"35d": horizon_block("nine_20261101", 35), "60d": horizon_block("nine_20261101", 60)},
        "risk": {name: values["risk"] for name, values in variants.items()},
        "forward_context": context,
    }, indent=2, default=float))
    print(thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
