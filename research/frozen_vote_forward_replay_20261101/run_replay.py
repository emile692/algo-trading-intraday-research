"""Replay the frozen S02-S06 vote ensembles on MNQ/MGC data after their OOS window.

The forward window 2026-03-20 -> 2026-10-02 was never used to map parameters
or to select clusters (both stop on 2026-03-19).  Each ensemble is executed
exactly as frozen: cells from the frozen manifest, the refinement-grid cell
engine and the S02 aggregator, on the extended continuous files used by the
2026-10-13 paper extension.  The run starts in 2019 so indicators carry their
full history; the overlap up to 2026-03-19 is compared with the frozen result.
MBT has no data after its legacy OOS window and is out of scope.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_signal_sheet_parameter_refinement as refinement  # noqa: E402
from freeze_signal_sheet_cluster_vote import WATCHED_CHECKS, audit_event_alignment, relabel  # noqa: E402
from rebuild_signal_sheet_02_cluster import aggregate_vote, build_cell_positions as build_s02_cells  # noqa: E402
from research.baseline_backtests.baseline_engine import load_frozen_specs, prepare_data  # noqa: E402


EXTENDED_DATA = {
    "MNQ": ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20190505_20261003.parquet",
    "MGC": ROOT / "data" / "processed" / "parquet" / "MGC_v_0_1m_20150101_20261003.parquet",
}
ENSEMBLES = ROOT / "strategy_registry" / "psf_mnq" / "frozen_ensembles"
START, OOS_END, FORWARD_START, FORWARD_END = "2019-05-06", "2026-03-19", "2026-03-20", "2026-10-02"
SLEEVES = {
    "s02": ("mnq_volatility_breakout", "volatility_breakout_42.json"),
    "s03": ("mnq_compression_breakout", "mnq_compression_breakout_vote_27.json"),
    "s04": ("mnq_atr_trend_flip", "mnq_atr_trend_flip_vote_9.json"),
    "s05": ("mgc_volatility_breakout", "mgc_volatility_breakout_vote_9.json"),
    "s06": ("mgc_gap_continuation", "mgc_gap_continuation_vote_9.json"),
}
PAPER_EXTENSION = ROOT / "research" / "nine_sleeve_paper_extension_20261013" / "daily_nine_sleeves_through_20261002.csv"


def cell_positions(key: str, manifest: dict) -> tuple[pd.DataFrame, object, dict]:
    if key == "s02":
        specs, _, _ = load_frozen_specs(refinement.FROZEN_DIR, refinement.FROZEN_MANIFEST)
        spec = next(item for item in specs if item["mechanism_id"] == "volatility_breakout")
        prepared = prepare_data(EXTENDED_DATA["MNQ"], start_date=pd.Timestamp(START).date(), end_date=pd.Timestamp(FORWARD_END).date())
        positions, _ = build_s02_cells(spec, prepared, manifest)
        return positions, prepared, spec
    campaign = copy.deepcopy(refinement.CAMPAIGNS[key])
    campaign["end"] = FORWARD_END
    # The refinement runner reads its data paths from module constants.
    refinement.MNQ_DATA, refinement.MGC_DATA = EXTENDED_DATA["MNQ"], EXTENDED_DATA["MGC"]
    prepared, spec, _ = refinement.prepare_campaign(key, campaign)
    region = {axis: values for axis, values in manifest["parameter_region"].items() if isinstance(values, list)}
    names = list(region)
    rows = []
    for values in pd.MultiIndex.from_product([region[name] for name in names]):
        params = dict(zip(names, values))
        run = refinement.run_candidate(key, refinement.apply_parameters(spec, campaign, params), prepared, campaign)
        violations = {name: run.checks.get(name) for name in WATCHED_CHECKS if run.checks.get(name, 0)}
        if violations:
            raise RuntimeError(f"{key} {params}: {violations}")
        positions = run.positions.copy()
        positions["cell_id"] = "__".join(f"{name}={value:g}" for name, value in params.items())
        rows.append(positions)
        print(f"{key} {params}: {len(positions)} positions virtuelles", flush=True)
    return pd.concat(rows, ignore_index=True), prepared, spec


def replay(key: str) -> None:
    sleeve, manifest_name = SLEEVES[key]
    manifest = json.loads((ENSEMBLES / manifest_name).read_text(encoding="utf-8"))
    positions, prepared, spec = cell_positions(key, manifest)
    if positions["cell_id"].nunique() != int(manifest["parameter_region"]["submodel_count"]):
        raise RuntimeError(f"{key}: wrong number of executed cells")
    audit_event_alignment(positions, prepared)
    result, _ = aggregate_vote(spec, prepared, positions, manifest)
    violations = {name: result.checks.get(name) for name in WATCHED_CHECKS if result.checks.get(name, 0)}
    if violations:
        raise RuntimeError(f"{key} aggregate vote: {violations}")
    trades = relabel(result.trades, key) if key != "s02" else result.trades
    daily = result.daily[["session_date", "daily_pnl_usd", "trade_count"]].copy()
    daily.to_csv(HERE / f"{key}_daily.csv", index=False)
    trades.to_csv(HERE / f"{key}_trades.csv", index=False)
    print(f"{key} ({sleeve}) : replay écrit", flush=True)


def metrics(daily: pd.Series, trades: int | None = None) -> dict:
    pnl = daily.astype(float)
    equity = np.r_[0.0, pnl.cumsum().to_numpy()]
    std = pnl.std(ddof=1)
    return {
        "sessions": int(len(pnl)), "trades": trades, "net_pnl_usd": float(pnl.sum()),
        "sharpe": float(np.sqrt(252.0) * pnl.mean() / std) if std > 0 else float("nan"),
        "max_drawdown_usd": float((equity - np.maximum.accumulate(equity)).min()),
        "worst_day_usd": float(pnl.min()), "positive_days_share": float((pnl[pnl != 0] > 0).mean()) if (pnl != 0).any() else float("nan"),
    }


def summarize() -> None:
    old = pd.read_csv(PAPER_EXTENSION, parse_dates=["session_date"]).set_index("session_date")
    forward_old = old.loc[FORWARD_START:FORWARD_END]
    calendar = forward_old.index
    rows, new_book = [], pd.DataFrame(index=calendar)
    new_book["mnq_orb_vwap_atr"] = forward_old["mnq_orb_vwap_atr"]
    for key, (sleeve, _) in SLEEVES.items():
        replayed = pd.read_csv(HERE / f"{key}_daily.csv", parse_dates=["session_date"]).set_index("session_date")
        frozen_dir = ROOT / "research" / f"signal_sheet_{key[1:]}_cluster"
        frozen = pd.read_csv(frozen_dir / "daily_results.csv", parse_dates=["session_date"]).set_index("session_date")["daily_pnl_usd"]
        overlap = replayed["daily_pnl_usd"].loc[START:OOS_END].reindex(frozen.loc[START:OOS_END].index).fillna(0.0)
        difference = (overlap - frozen.loc[START:OOS_END]).abs()
        window = replayed.loc[FORWARD_START:FORWARD_END]
        new_book[sleeve] = window["daily_pnl_usd"].reindex(calendar).fillna(0.0)
        rows.append({
            "sleeve": sleeve,
            "overlap_identical_days_share": float((difference < 0.01).mean()),
            "overlap_pnl_frozen_usd": float(frozen.loc[START:OOS_END].sum()),
            "overlap_pnl_replay_usd": float(overlap.sum()),
            **{f"vote_{k}": v for k, v in metrics(new_book[sleeve], int(window["trade_count"].sum())).items()},
            **{f"old_point_{k}": v for k, v in metrics(forward_old[sleeve]).items() if k != "trades"},
        })
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "forward_sleeve_comparison.csv", index=False)
    six = list(new_book.columns)
    books = {
        "mnq_mgc_six_new": metrics(new_book[six].sum(axis=1)),
        "mnq_mgc_six_old": metrics(forward_old[six].sum(axis=1)),
    }
    (HERE / "forward_book_comparison.json").write_text(json.dumps(books, indent=2), encoding="utf-8")
    new_book.reset_index().to_csv(HERE / "forward_daily_six_new.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(json.dumps(books, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleeve", choices=sorted(SLEEVES))
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    else:
        replay(args.sleeve)


if __name__ == "__main__":
    main()
