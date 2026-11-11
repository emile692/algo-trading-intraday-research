"""Export the active eight-sleeve book as a versioned package for the prop-firm simulator.

The package is the only interface between this research repository and
``algo-trading-prop-firm-simulator``: daily net PnL per sleeve and for the book,
the book's worst intraday PnL from day start (adverse OHLC marks), traded-day
flags and the exact path-sampler flags of the Topstep harness, so the simulator
can reproduce the locked 45.5875 % PASS at 60 days.

Two periods are written separately:
- ``locked_window``: 2024-01-02 -> 2026-03-19, the Topstep reference window,
  already inspected (cluster selection, S04 removal);
- ``forward_partial``: 2026-03-20 -> 2026-10-02, MNQ/MGC sleeves only (no MBT
  data), daily PnL without intraday marks.  It motivated the S04 removal, so it
  is not out-of-sample for this book either.

Nothing in the lock directories is modified; the rebuilt book must match the
locked ledger before anything is written.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOCK = REPO / "research" / "locked_eight_sleeve_portfolio_20261103"
REPLAY = REPO / "research" / "frozen_vote_forward_replay_20261101"
PAPER_EXTENSION = REPO / "research" / "nine_sleeve_paper_extension_20261013" / "daily_nine_sleeves_through_20261002.csv"
OUT = HERE / "eight_sleeve_20261103"
PACKAGE_ID = "book_eight_sleeve_20261103_v1"

_spec = importlib.util.spec_from_file_location("relock_eight_sleeves_20261103", LOCK / "relock_eight_sleeves.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)
R, T = E.R, E.T
FORWARD_SLEEVES = ["mnq_orb_vwap_atr", "mnq_volatility_breakout", "mnq_compression_breakout",
                   "mgc_volatility_breakout", "mgc_gap_continuation"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locked_window() -> pd.DataFrame:
    """Rebuild the eight-sleeve book exactly as the 2026-11-03 lock does, with intraday marks."""
    six_pnl, six_counts, six_trades = T.load_six()
    calendar = six_pnl.index
    timeline_source = pd.read_csv(T.SIX_DIR / "common_timestamp_portfolio.csv", usecols=["timestamp_utc"], parse_dates=["timestamp_utc"])
    timeline = pd.DatetimeIndex(timeline_source["timestamp_utc"]).tz_convert("UTC")
    timeline_dates = pd.Series(pd.to_datetime(timeline.tz_convert("America/New_York").date), index=timeline)
    _, six_adverse, _ = T.six_intraday_paths(timeline, six_trades)

    daily = pd.DataFrame(index=calendar)
    counts = pd.DataFrame(index=calendar)
    daily["mnq_orb_vwap_atr"] = six_pnl["mnq_orb_vwap_atr"]
    counts["mnq_orb_vwap_atr"] = six_counts["mnq_orb_vwap_atr"]
    ledgers = {}
    for sleeve in E.SLEEVES[1:]:
        directory = R.VOTE_SLEEVES[sleeve][1]
        results = pd.read_csv(REPO / "research" / directory / "daily_results.csv", parse_dates=["session_date"]).set_index("session_date")
        daily[sleeve] = results["daily_pnl_usd"].reindex(calendar).fillna(0.0)
        counts[sleeve] = results["trade_count"].reindex(calendar).fillna(0).astype(int)
        ledgers[sleeve] = R.vote_trades(directory)
    vote_path, vote_adverse = R.intraday_paths(timeline, ledgers)
    R.reconcile(vote_path, daily[list(ledgers)], timeline_dates, "eight-sleeve vote ensembles")
    adverse = six_adverse[["mnq_orb_vwap_atr"]].join(vote_adverse)[E.SLEEVES]

    book = E.book(daily, adverse, counts, timeline_dates, E.SLEEVES)
    published = pd.read_csv(LOCK / "daily_pnl.csv", parse_dates=["session_date"]).set_index("session_date")
    if float((published["portfolio_pnl_usd"] - book["daily"]).abs().max()) > 1e-6:
        raise RuntimeError("Rebuilt book differs from the locked eight-sleeve ledger")

    frame = daily[E.SLEEVES].copy()
    frame["book_pnl_usd"] = book["daily"]
    frame["book_intraday_min_usd"] = book["intraday_min"]
    frame["book_traded"] = book["traded"]
    # The Topstep harness samples blocks on the six-sleeve traded calendar; keep it for exact reproduction.
    frame["sampler_traded"] = six_counts[T.OLD].gt(0).any(axis=1).to_numpy(bool)
    frame.insert(0, "period", "locked_window")
    return frame.rename_axis("session_date").reset_index()


def forward_partial() -> pd.DataFrame:
    paper = pd.read_csv(PAPER_EXTENSION, parse_dates=["session_date"]).set_index("session_date")
    window = paper.loc[E.FORWARD_START:"2026-10-02"]
    frame = pd.DataFrame(index=window.index)
    frame["mnq_orb_vwap_atr"] = window["mnq_orb_vwap_atr"]
    for sleeve in FORWARD_SLEEVES[1:]:
        replay = pd.read_csv(REPLAY / f"{E.REPLAY_KEYS[sleeve]}_daily.csv", parse_dates=["session_date"]).set_index("session_date")
        frame[sleeve] = replay["daily_pnl_usd"].reindex(frame.index).fillna(0.0)
    frame["book_pnl_usd"] = frame[FORWARD_SLEEVES].sum(axis=1)
    frame["book_traded"] = frame[FORWARD_SLEEVES].ne(0).any(axis=1)
    frame.insert(0, "period", "forward_partial")
    return frame.rename_axis("session_date").reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    locked, forward = locked_window(), forward_partial()
    locked.to_csv(OUT / "book_daily_locked_window.csv", index=False, float_format="%.8f")
    forward.to_csv(OUT / "book_daily_forward_partial.csv", index=False, float_format="%.8f")

    lock_config = json.loads((LOCK / "locked_eight_sleeve_config.json").read_text(encoding="utf-8"))
    reference = pd.read_csv(LOCK / "topstep_reference_metrics.csv").set_index("horizon")
    manifest = {
        "package_id": PACKAGE_ID,
        "source_repository": "algo-trading-intraday-research",
        "source_lock": {"portfolio_id": lock_config["portfolio_id"],
                        "path": "research/locked_eight_sleeve_portfolio_20261103/locked_eight_sleeve_config.json",
                        "sha256": sha256(LOCK / "locked_eight_sleeve_config.json")},
        "sleeves": E.SLEEVES,
        "units": "USD, one whole contract per sleeve, net of nominal costs",
        "periods": {
            "locked_window": {
                "file": "book_daily_locked_window.csv",
                "start": str(locked["session_date"].min().date()), "end": str(locked["session_date"].max().date()),
                "sessions": int(len(locked)),
                "intraday_marks": True,
                "status": "IN_SAMPLE_INSPECTED",
                "note": "Topstep reference window; already used for cluster selection and the S04 removal.",
            },
            "forward_partial": {
                "file": "book_daily_forward_partial.csv",
                "start": str(forward["session_date"].min().date()), "end": str(forward["session_date"].max().date()),
                "sessions": int(len(forward)),
                "intraday_marks": False,
                "sleeves_present": FORWARD_SLEEVES,
                "sleeves_missing": [name for name in E.SLEEVES if name not in FORWARD_SLEEVES],
                "status": "INSPECTED_PARTIAL",
                "note": "MNQ/MGC only (no MBT data); motivated the S04 removal, so not out-of-sample for this book.",
            },
        },
        "columns": {
            "<sleeve>": "net daily PnL of one contract of that sleeve",
            "book_pnl_usd": "sum of the sleeve columns",
            "book_intraday_min_usd": "worst book PnL from day start, adverse OHLC marks, closed trades included",
            "book_traded": "at least one sleeve traded that session",
            "sampler_traded": "six-sleeve traded flag used by the Topstep harness block sampler",
        },
        "reproduction_target": {
            "method": "moving_block_bootstrap, pooled blocks 3/5/10/20, 2000 paths per block, RNG order horizon -> block -> path",
            "seed": T.SEED, "blocks": list(T.BLOCKS), "paths_per_block": T.PATHS_PER_BLOCK, "horizons": list(T.HORIZONS),
            "rules": {"starting_balance_usd": T.STARTING_BALANCE, "profit_target_usd": T.PROFIT_TARGET,
                      "mll_usd": T.MLL_DISTANCE, "consistency": T.CONSISTENCY},
            "expected": {str(horizon): {key: float(reference.loc[horizon, key]) for key in
                                        ("pass_probability", "mll_probability", "expiry_probability")}
                         for horizon in T.HORIZONS},
        },
        "files": {name: sha256(OUT / name) for name in ("book_daily_locked_window.csv", "book_daily_forward_partial.csv")},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("package_id", "periods", "files")}, indent=2))


if __name__ == "__main__":
    main()
