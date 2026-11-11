"""Versioned re-lock of the nine-sleeve equal-contract reference (2026-11-01).

Sleeves S02-S09 now execute their frozen vote ensembles
(``strategy_registry/psf_mnq/frozen_ensembles``); S01 ORB is unchanged.  The
mapping stays one whole contract per sleeve: nothing is retuned.

The Topstep 50K reference is recomputed with the exact functions, seed, blocks
and path sampler of ``research/mbt_nine_sleeve_topstep_20261004``.  Both the
superseded and the new book are simulated on the same bootstrap paths, and the
superseded book must reproduce its published 45.2375 % PASS before anything is
written.  The superseded lock directory is never modified.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STUDY = REPO / "research" / "mbt_nine_sleeve_topstep_20261004"
SUPERSEDED = REPO / "research" / "locked_nine_sleeve_portfolio"
ENSEMBLES = REPO / "strategy_registry" / "psf_mnq" / "frozen_ensembles"
MBT_DATA = REPO / "data" / "downloaded" / "MBT_v_0_1m_20210425_20261003.parquet"
PORTFOLIO_ID = "nine_sleeve_equal_contract_reference_20261101"
SUPERSEDED_ID = "nine_sleeve_equal_contract_reference_20261004"
DECISION_DATE = "2026-11-01"

_spec = importlib.util.spec_from_file_location("run_topstep_20261004", STUDY / "run_topstep.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

SLEEVES = T.ALL
VOTE_SLEEVES = {
    "mnq_volatility_breakout": ("volatility_breakout_42.json", "signal_sheet_02_cluster"),
    "mnq_compression_breakout": ("mnq_compression_breakout_vote_27.json", "signal_sheet_03_cluster"),
    "mnq_atr_trend_flip": ("mnq_atr_trend_flip_vote_9.json", "signal_sheet_04_cluster"),
    "mgc_volatility_breakout": ("mgc_volatility_breakout_vote_9.json", "signal_sheet_05_cluster"),
    "mgc_gap_continuation": ("mgc_gap_continuation_vote_9.json", "signal_sheet_06_cluster"),
    "mbt_rolling_range_breakout": ("mbt_rolling_range_breakout_vote_6.json", "signal_sheet_07_cluster"),
    "mbt_trend_pullback_ma_structure": ("mbt_trend_pullback_ma_structure_vote_9.json", "signal_sheet_08_cluster"),
    "mbt_volatility_breakout": ("mbt_volatility_breakout_vote_12.json", "signal_sheet_09_cluster"),
}
PRICE_SOURCES = {"MNQ": (T.MNQ_DATA, "timestamp"), "MGC": (T.MGC_DATA, "timestamp"), "MBT": (MBT_DATA, "ts_event")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def read_prices(path: Path, index_name: str) -> pd.DataFrame:
    column_tz = str(pq.read_schema(path).field(index_name).type.tz)
    start = pd.Timestamp(T.START, tz="America/New_York").tz_convert(column_tz)
    end = pd.Timestamp(T.END + pd.Timedelta(days=1), tz="America/New_York").tz_convert(column_tz)
    frame = pd.read_parquet(path, columns=["open", "high", "low", "close"],
                            filters=[(index_name, ">=", start), (index_name, "<", end)])
    frame.index = frame.index.tz_convert("UTC").floor("min")
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def vote_trades(directory: str) -> pd.DataFrame:
    trades = pd.read_csv(REPO / "research" / directory / "trades.csv")
    trades["entry_ts_utc"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_ts_utc"] = pd.to_datetime(trades["exit_time"], utc=True)
    trades["signed_quantity"] = trades["position"] * trades["quantity"]
    trades["fees"] = trades["commissions_usd"]
    return trades


def intraday_paths(timeline: pd.DatetimeIndex, trades: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Close-marked and adverse-OHLC-marked PnL from day start, as in the 2026-10-04 study."""
    prices = {asset: read_prices(*source) for asset, source in PRICE_SOURCES.items()}
    normal = pd.DataFrame(0.0, index=timeline, columns=list(trades))
    adverse = pd.DataFrame(0.0, index=timeline, columns=list(trades))
    timeline_dates = pd.Series(timeline.tz_convert("America/New_York").date, index=timeline)
    for sleeve, ledger in trades.items():
        instrument = T.INSTRUMENT[sleeve]
        px = prices[instrument].reindex(timeline).groupby(timeline_dates, sort=False).ffill()
        point_value = T.POINT_VALUE[instrument]
        for day, group in ledger.groupby(pd.to_datetime(ledger["session_date"]).dt.date, sort=True):
            day_idx = timeline[timeline_dates.eq(day).to_numpy()]
            if not len(day_idx):
                continue
            base = normal.loc[day_idx, sleeve].to_numpy(copy=True)
            bad = adverse.loc[day_idx, sleeve].to_numpy(copy=True)
            for row in group.itertuples(index=False):
                start_i = int(day_idx.searchsorted(pd.Timestamp(row.entry_ts_utc).floor("min"), side="left"))
                end_i = int(day_idx.searchsorted(pd.Timestamp(row.exit_ts_utc).floor("min"), side="left"))
                direction = float(row.signed_quantity) / float(row.quantity)
                quantity = float(row.quantity)
                entry_fee = float(row.fees) / 2.0
                if end_i > start_i:
                    marks = px.loc[day_idx[start_i:end_i], "close"].to_numpy(float)
                    adverse_marks = px.loc[day_idx[start_i:end_i], "low" if direction > 0 else "high"].to_numpy(float)
                    base[start_i:end_i] += (marks - float(row.entry_price)) * direction * quantity * point_value - entry_fee
                    bad[start_i:end_i] += (adverse_marks - float(row.entry_price)) * direction * quantity * point_value - entry_fee
                if end_i < len(day_idx):
                    base[end_i:] += float(row.net_pnl_usd)
                    bad[end_i:] += float(row.net_pnl_usd)
            normal.loc[day_idx, sleeve] = base
            adverse.loc[day_idx, sleeve] = bad
    return normal, adverse


def reconcile(paths: pd.DataFrame, daily: pd.DataFrame, timeline_dates: pd.Series, label: str) -> float:
    last = ~timeline_dates.duplicated(keep="last")
    eod = paths.loc[last.to_numpy()].set_axis(pd.DatetimeIndex(timeline_dates.loc[last].to_numpy()))
    error = float((eod.reindex(daily.index).fillna(0.0)[paths.columns] - daily[paths.columns]).abs().max().max())
    if error > 1e-6:
        raise RuntimeError(f"{label}: intraday/daily reconciliation failed ({error})")
    return error


def book(daily: pd.DataFrame, adverse: pd.DataFrame, counts: pd.DataFrame, timeline_dates: pd.Series) -> dict[str, Any]:
    mapping = {name: 1 for name in SLEEVES}
    portfolio = T.portfolio_daily(daily, mapping)
    adverse_path = T.portfolio_path(adverse, mapping)
    intraday_min = adverse_path.groupby(timeline_dates, sort=False).min().reindex(daily.index).fillna(0.0)
    return {
        "daily": portfolio.to_numpy(float),
        "intraday_min": intraday_min.to_numpy(float),
        "traded": counts[SLEEVES].gt(0).any(axis=1).to_numpy(bool),
        "risk": T.loss_cluster_metrics(portfolio, adverse_path, timeline_dates),
    }


def bootstrap(variants: dict[str, dict[str, Any]], sampler_traded: np.ndarray) -> pd.DataFrame:
    """Same RNG consumption order as the 2026-10-04 campaign, so paths are identical."""
    rows = []
    rng = np.random.default_rng(T.SEED)
    for horizon in T.HORIZONS:
        for block in T.BLOCKS:
            for path_id, indices in enumerate(T.bootstrap_paths(sampler_traded, horizon, block, rng), start=1):
                for name, values in variants.items():
                    result = T.simulate_path(values["daily"][indices], values["intraday_min"][indices], values["traded"][indices], horizon)
                    rows.append({"method": "moving_block_bootstrap", "block": block, "path_id": path_id,
                                 "scenario": name, "horizon": horizon, **result})
    runs = pd.DataFrame(rows)
    detail = T.summarize(runs, ["method", "scenario", "horizon", "block"])
    pooled = T.summarize(runs, ["method", "scenario", "horizon"]).assign(block="pooled")
    return pd.concat([detail, pooled], ignore_index=True)


def main() -> None:
    six_pnl, six_counts, six_trades = T.load_six()
    calendar = six_pnl.index
    mbt_nominal, _, mbt_counts, _ = T.load_mbt(calendar)
    timeline_source = pd.read_csv(T.SIX_DIR / "common_timestamp_portfolio.csv", usecols=["timestamp_utc"], parse_dates=["timestamp_utc"])
    timeline = pd.DatetimeIndex(timeline_source["timestamp_utc"]).tz_convert("UTC")
    timeline_dates = pd.Series(pd.to_datetime(timeline.tz_convert("America/New_York").date), index=timeline)

    # Superseded book, rebuilt exactly as in the 2026-10-04 study.
    old_daily = six_pnl[T.OLD].join(mbt_nominal[T.NEW], how="left").fillna(0.0)
    old_counts = six_counts[T.OLD].join(mbt_counts[T.NEW], how="left").fillna(0).astype(int)
    six_path, six_adverse, _ = T.six_intraday_paths(timeline, six_trades)
    _, mbt_path, mbt_adverse = T.load_mbt_minute(timeline, "nominal")
    reconcile(six_path, six_pnl, timeline_dates, "superseded six")
    old_adverse = six_adverse.join(mbt_adverse[T.NEW], how="left").fillna(0.0)

    # New book: ORB unchanged, S02-S09 from their executed frozen vote ensembles.
    new_daily = pd.DataFrame(index=calendar)
    new_counts = pd.DataFrame(index=calendar)
    new_daily["mnq_orb_vwap_atr"] = six_pnl["mnq_orb_vwap_atr"]
    new_counts["mnq_orb_vwap_atr"] = six_counts["mnq_orb_vwap_atr"]
    ledgers = {}
    for sleeve, (_, directory) in VOTE_SLEEVES.items():
        results = pd.read_csv(REPO / "research" / directory / "daily_results.csv", parse_dates=["session_date"]).set_index("session_date")
        new_daily[sleeve] = results["daily_pnl_usd"].reindex(calendar).fillna(0.0)
        new_counts[sleeve] = results["trade_count"].reindex(calendar).fillna(0).astype(int)
        ledgers[sleeve] = vote_trades(directory)
    new_daily, new_counts = new_daily[SLEEVES], new_counts[SLEEVES]
    vote_path, vote_adverse = intraday_paths(timeline, ledgers)
    reconcile(vote_path, new_daily, timeline_dates, "new vote ensembles")
    new_adverse = six_adverse[["mnq_orb_vwap_atr"]].join(vote_adverse)[SLEEVES]

    variants = {
        "superseded_20261004": book(old_daily, old_adverse, old_counts, timeline_dates),
        "relock_20261101": book(new_daily, new_adverse, new_counts, timeline_dates),
    }
    sampler_traded = six_counts[T.OLD].gt(0).any(axis=1).to_numpy(bool)
    summary = bootstrap({name: values for name, values in variants.items()}, sampler_traded)
    pooled = summary[summary["block"].astype(str).eq("pooled")].set_index(["scenario", "horizon"])
    reproduced = float(pooled.loc[("superseded_20261004", 60), "pass_probability"])
    if abs(reproduced - 0.452375) > 1e-12 or abs(float(pooled.loc[("superseded_20261004", 35), "pass_probability"]) - 0.35025) > 1e-12:
        raise RuntimeError(f"Harness check failed: superseded book PASS {reproduced} instead of 0.452375")

    HERE.mkdir(parents=True, exist_ok=True)
    daily = new_daily.copy()
    daily["portfolio_pnl_usd"] = daily[SLEEVES].sum(axis=1)
    daily.reset_index().rename(columns={"index": "session_date"}).to_csv(HERE / "daily_pnl.csv", index=False, float_format="%.8f")
    summary.to_csv(HERE / "topstep_summary.csv", index=False)
    reference = pooled.loc["relock_20261101"].reset_index().assign(method="moving_block_bootstrap", scenario="equal_contract__nominal__nine", block="pooled")
    reference = reference[["method", "scenario", "horizon", "block", "paths", "pass_probability", "mll_probability",
                           "expiry_probability", "median_days_to_pass", "median_max_drawdown_usd", "p05_max_drawdown_usd"]]
    reference.to_csv(HERE / "topstep_reference_metrics.csv", index=False)
    pd.DataFrame([{"book": name, **values["risk"]} for name, values in variants.items()]).to_csv(HERE / "book_risk_comparison.csv", index=False)

    sleeve_rows = [{"sleeve": "mnq_orb_vwap_atr", "instrument": "MNQ", "integer_copies": 1, "source_type": "legacy_frozen_sleeve",
                    "source": "research/portfolio_integration_phase1/sleeve_trades/mnq_orb_vwap_atr.csv", "status": "LOCKED_ACTIVE_REFERENCE"}]
    sleeve_sources: dict[str, Any] = {"mnq_orb_vwap_atr": {
        "type": "legacy_frozen_sleeve",
        "daily_source": "research/portfolio_integration_phase1/daily_sleeve_pnl.csv",
        "daily_source_sha256": sha256(T.SIX_DIR / "daily_sleeve_pnl.csv"),
    }}
    for sleeve, (manifest_name, directory) in VOTE_SLEEVES.items():
        manifest = ENSEMBLES / manifest_name
        sleeve_rows.append({"sleeve": sleeve, "instrument": T.INSTRUMENT[sleeve], "integer_copies": 1,
                            "source_type": "frozen_vote_ensemble", "source": relative(manifest), "status": "LOCKED_ACTIVE_REFERENCE"})
        sleeve_sources[sleeve] = {
            "type": "frozen_vote_ensemble",
            "ensemble_manifest": relative(manifest),
            "ensemble_manifest_sha256": sha256(manifest),
            "daily_source": f"research/{directory}/daily_results.csv",
            "daily_source_sha256": sha256(REPO / "research" / directory / "daily_results.csv"),
        }
    pd.DataFrame(sleeve_rows).to_csv(HERE / "sleeve_manifest.csv", index=False)

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
        "superseded_reference_retained_for_audit": relative(SUPERSEDED),
        "allocation_family": "equal_contract",
        "mapping": {name: 1 for name in SLEEVES},
        "portfolio_rule": "sum the executable PnL of one whole-contract copy of each frozen sleeve; no fractional scaling",
        "sleeve_sources": sleeve_sources,
        "topstep_reference": {
            "account": "Trading Combine 50K",
            "rules": {"profit_target_usd": T.PROFIT_TARGET, "mll_usd": T.MLL_DISTANCE, "consistency": T.CONSISTENCY},
            "method": "moving_block_bootstrap_pooled_blocks_3_5_10_20",
            "paths": int(pooled.loc[("relock_20261101", 60), "paths"]),
            "seed": T.SEED,
            "window": [str(calendar.min().date()), str(calendar.max().date())],
            "path_sampler": "identical to the 2026-10-04 study (six-sleeve traded-day sampler, same RNG order)",
            "horizon_35d": horizon_block("relock_20261101", 35),
            "horizon_60d": horizon_block("relock_20261101", 60),
        },
        "comparison_with_superseded": {
            "same_bootstrap_paths": True,
            "superseded_reproduced_pass_60d": reproduced,
            "horizon_35d": horizon_block("superseded_20261004", 35),
            "horizon_60d": horizon_block("superseded_20261004", 60),
        },
        "governance": {
            "no_retuning": True,
            "whole_contracts_only": True,
            "pass_rate_optimization_performed": False,
            "sleeve_definition_change": "S02-S09 execute frozen parameter-vote ensembles; S03-S06 clusters are lock neighbourhoods chosen without a performance criterion, S07 drops a duplicate cell, S08-S09 regions unchanged",
            "cluster_selection_evidence": "research/signal_sheet_parameter_refinement/cluster_selection/cluster_selection_cscv.json",
            "prospective_start_after": DECISION_DATE,
            "known_issue": "mbt_volatility_breakout vote is flat on its legacy OOS window (Sharpe -0.040) despite positive cells; flagged for review",
            "caveat": "Topstep figures are retrospective bootstrap estimates on an already-inspected window, not guaranteed pass probabilities or pristine OOS results.",
        },
        "locked_inputs": {
            "superseded_lock_manifest": {"path": relative(SUPERSEDED / "lock_manifest.json"), "sha256": sha256(SUPERSEDED / "lock_manifest.json")},
            "topstep_study_script": {"path": relative(STUDY / "run_topstep.py"), "sha256": sha256(STUDY / "run_topstep.py")},
            "six_timestamp_portfolio": {"path": relative(T.SIX_DIR / "common_timestamp_portfolio.csv"), "sha256": sha256(T.SIX_DIR / "common_timestamp_portfolio.csv")},
        },
    }
    (HERE / "locked_nine_sleeve_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    files = ("locked_nine_sleeve_config.json", "daily_pnl.csv", "sleeve_manifest.csv", "topstep_reference_metrics.csv")
    lock_manifest = {"portfolio_id": PORTFOLIO_ID, "status": "LOCKED_ACTIVE_REFERENCE",
                     "files": {relative(HERE / name): sha256(HERE / name) for name in files}}
    (HERE / "lock_manifest.json").write_text(json.dumps(lock_manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "harness_superseded_pass_60d": reproduced,
        "relock": {"35d": horizon_block("relock_20261101", 35), "60d": horizon_block("relock_20261101", 60)},
        "superseded": {"35d": horizon_block("superseded_20261004", 35), "60d": horizon_block("superseded_20261004", 60)},
        "risk": {name: values["risk"] for name, values in variants.items()},
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
