"""Step 0 only: inventory and reproduce the frozen IS trades.

This module deliberately does not calculate any OOS statistic.  Market data is
predicate-filtered by PyArrow while reading the parquet files, before it is
passed to the repository's canonical feature and backtest functions.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.mean_reversion_cross_asset_20261006.run_campaign import (
    EXIT_SPECS,
    INSTRUMENTS as GAP_INSTRUMENTS,
    _load_bars,
    _prepare_features,
    _signals,
    backtest as gap_backtest,
)
from research.volume_auction_landscape_20261007.run_campaign import (
    INSTRUMENTS as CLIMAX_INSTRUMENTS,
    add_causal_volume_baselines,
    add_price_features,
    backtest as climax_backtest,
    load_bars,
    signals,
)


OUT = Path(__file__).resolve().parent

GAP_SOURCES = {
    "M2K": REPO / "data/processed/parquet/M2K_c_0_5m_20260322_134808.parquet",
    "MES": REPO / "data/processed/parquet/MES_c_0_5m_20260322_135702.parquet",
    "MNQ": REPO / "data/processed/parquet/MNQ_c_0_5m_20260321_094501.parquet",
    "NQ": REPO / "data/processed/parquet/NQ_c_0_5m_20260321_095036.parquet",
}
GAP_SPLIT_START = {
    "M2K": pd.Timestamp("2024-02-23"),
    "MES": pd.Timestamp("2024-02-23"),
    "MNQ": pd.Timestamp("2024-02-23"),
    "NQ": pd.Timestamp("2021-07-02"),
}
GAP_THRESHOLDS = (0.25, 0.50, 0.75, 1.00)
GAP_EXPECTED = {
    ("M2K", 0.25): (678, -673.0714285714372),
    ("M2K", 0.50): (334, 448.3214285714232),
    ("M2K", 0.75): (163, 88.8928571428512),
    ("M2K", 1.00): (80, -579.4642857142958),
    ("MES", 0.25): (746, -3180.625),
    ("MES", 0.50): (432, -2021.8749999999955),
    ("MES", 0.75): (250, -577.6785714285688),
    ("MES", 1.00): (137, -947.9464285714266),
    ("MNQ", 0.25): (711, -6870.250000000007),
    ("MNQ", 0.50): (392, -1906.2857142857138),
    ("MNQ", 0.75): (201, 1841.0),
    ("MNQ", 1.00): (113, -828.0),
    ("NQ", 0.25): (1586, -50677.85714285714),
    ("NQ", 0.50): (850, -33167.142857142884),
    ("NQ", 0.75): (450, 2773.571428571422),
    ("NQ", 1.00): (228, -4115.0),
}

CLIMAX_SOURCE = REPO / "data/processed/parquet/MGC_v_0_5m_20150101_20260923.parquet"
CLIMAX_SPLIT_START = pd.Timestamp("2023-03-17")
CLIMAX_PARAMS = {
    "mechanism": "B",
    "mode": "confirmed_fade",
    "lookback": 1,
    "move": 1.5,
    "rvol": 2.0,
    "profile": 0,
    "confirm": 1,
}
CLIMAX_EXPECTED = (1333, -6762.500000000022)
CLIMAX_REFERENCE = (
    REPO
    / "research/mgc_volume_climax_audit_20261007/results/canonical_representative_trades.csv"
)


def _write_is_only_parquet(source: Path, destination: Path, oos_start: pd.Timestamp) -> None:
    """Read only rows strictly before ``oos_start`` and persist a temporary view."""
    cutoff = oos_start.tz_localize("America/New_York")
    frame = pd.read_parquet(source, filters=[("timestamp", "<", cutoff)])
    if frame.empty:
        raise RuntimeError(f"No IS rows loaded from {source}")
    timestamps = pd.DatetimeIndex(frame.index)
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize("America/New_York")
    else:
        timestamps = timestamps.tz_convert("America/New_York")
    if not (timestamps < cutoff).all():
        raise AssertionError(f"Post-IS bar entered the load from {source}")
    frame.to_parquet(destination)


def _assert_is_only_bars(bars: pd.DataFrame, oos_start: pd.Timestamp) -> None:
    dates = pd.to_datetime(bars["session_date"])
    if dates.empty or not dates.lt(oos_start).all():
        raise AssertionError("A post-IS bar entered an analysis")


def _assert_is_only_trades(trades: pd.DataFrame, oos_start: pd.Timestamp) -> None:
    for column in ("session_date", "signal_time", "entry_time", "exit_time"):
        values = pd.to_datetime(trades[column], utc=True).dt.tz_convert("America/New_York")
        if not values.dt.tz_localize(None).dt.normalize().lt(oos_start).all():
            raise AssertionError(f"A post-IS trade timestamp entered via {column}")


def _yearly(trades: pd.DataFrame) -> pd.DataFrame:
    years = pd.to_datetime(trades["session_date"]).dt.year
    return (
        trades.assign(year=years)
        .groupby("year", as_index=False)
        .agg(trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"))
    )


def reproduce_gap(temp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fair_exit = next(spec for spec in EXIT_SPECS if spec[0] == "fair_value")
    summaries: list[dict[str, object]] = []
    annual: list[pd.DataFrame] = []
    ledgers: list[pd.DataFrame] = []
    for symbol, source in GAP_SOURCES.items():
        oos_start = GAP_SPLIT_START[symbol]
        temp_path = temp_dir / f"{symbol}_is.parquet"
        _write_is_only_parquet(source, temp_path, oos_start)
        bars = _prepare_features(_load_bars(temp_path, GAP_INSTRUMENTS[symbol]))
        _assert_is_only_bars(bars, oos_start)
        for threshold in GAP_THRESHOLDS:
            signal, fair = _signals(
                bars, "gap_reversion", {"lookback": 3, "threshold": threshold}
            )
            trades = gap_backtest(
                bars, signal, fair, GAP_INSTRUMENTS[symbol], fair_exit
            )
            _assert_is_only_trades(trades, oos_start)
            expected_count, expected_pnl = GAP_EXPECTED[(symbol, threshold)]
            actual_pnl = float(trades["net_pnl"].sum())
            if len(trades) != expected_count or not np.isclose(
                actual_pnl, expected_pnl, rtol=0.0, atol=1e-8
            ):
                raise AssertionError(
                    f"Gap replay mismatch for {symbol} N={threshold}: "
                    f"got ({len(trades)}, {actual_pnl}), expected "
                    f"({expected_count}, {expected_pnl})"
                )
            start = pd.Timestamp(bars["session_date"].min())
            end = pd.Timestamp(bars["session_date"].max())
            summaries.append(
                {
                    "signal": "A",
                    "asset": symbol,
                    "configuration": f"N>={threshold:g}",
                    "is_start": start.date().isoformat(),
                    "is_end": end.date().isoformat(),
                    "trades": len(trades),
                    "net_pnl": actual_pnl,
                    "reference_match": True,
                }
            )
            yearly = _yearly(trades)
            yearly.insert(0, "configuration", f"N>={threshold:g}")
            yearly.insert(0, "asset", symbol)
            yearly.insert(0, "signal", "A")
            annual.append(yearly)
            ledger = trades.copy()
            ledger.insert(0, "configuration", f"N>={threshold:g}")
            ledger.insert(0, "asset", symbol)
            ledger.insert(0, "signal", "A")
            ledgers.append(ledger)
    return pd.DataFrame(summaries), pd.concat(annual, ignore_index=True), pd.concat(ledgers, ignore_index=True)


def _load_climax_is_reference() -> pd.DataFrame:
    """Stream the frozen ledger and retain only IS rows during ingestion."""
    rows: list[dict[str, str]] = []
    with CLIMAX_REFERENCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if pd.Timestamp(row["session_date"]) < CLIMAX_SPLIT_START:
                rows.append(row)
    return pd.DataFrame(rows)


def reproduce_climax(temp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temp_path = temp_dir / "MGC_is.parquet"
    _write_is_only_parquet(CLIMAX_SOURCE, temp_path, CLIMAX_SPLIT_START)
    inst = CLIMAX_INSTRUMENTS["MGC"]
    bars = add_price_features(add_causal_volume_baselines(load_bars(temp_path, inst)))
    _assert_is_only_bars(bars, CLIMAX_SPLIT_START)
    trades = climax_backtest(bars, signals(bars, CLIMAX_PARAMS), inst)
    _assert_is_only_trades(trades, CLIMAX_SPLIT_START)
    expected_count, expected_pnl = CLIMAX_EXPECTED
    actual_pnl = float(trades["net_pnl"].sum())
    if len(trades) != expected_count or not np.isclose(
        actual_pnl, expected_pnl, rtol=0.0, atol=1e-8
    ):
        raise AssertionError(
            f"Climax replay mismatch: got ({len(trades)}, {actual_pnl}), "
            f"expected ({expected_count}, {expected_pnl})"
        )

    reference = _load_climax_is_reference()
    if len(reference) != len(trades):
        raise AssertionError("MGC frozen ledger count differs from the IS-only replay")
    for column in ("direction", "entry", "exit", "gross_pnl", "net_pnl"):
        if not np.allclose(
            trades[column].to_numpy(float),
            reference[column].astype(float).to_numpy(),
            rtol=0.0,
            atol=1e-8,
        ):
            raise AssertionError(f"MGC frozen ledger differs in {column}")
    for column in ("signal_time", "entry_time", "exit_time", "session_date"):
        actual = pd.to_datetime(trades[column], utc=True)
        expected = pd.to_datetime(reference[column], utc=True)
        if not actual.equals(expected):
            raise AssertionError(f"MGC frozen ledger dates differ in {column}")

    start = pd.Timestamp(bars["session_date"].min())
    end = pd.Timestamp(bars["session_date"].max())
    summary = pd.DataFrame(
        [
            {
                "signal": "B",
                "asset": "MGC",
                "configuration": "B__confirmed_fade__n1__m1.5__rv2.0__p0__c1",
                "is_start": start.date().isoformat(),
                "is_end": end.date().isoformat(),
                "trades": len(trades),
                "net_pnl": actual_pnl,
                "reference_match": True,
            }
        ]
    )
    annual = _yearly(trades)
    annual.insert(0, "configuration", summary.loc[0, "configuration"])
    annual.insert(0, "asset", "MGC")
    annual.insert(0, "signal", "B")
    ledger = trades.copy()
    ledger.insert(0, "configuration", summary.loc[0, "configuration"])
    ledger.insert(0, "asset", "MGC")
    ledger.insert(0, "signal", "B")
    return summary, annual, ledger


def main() -> None:
    OUT.mkdir(exist_ok=True)
    temp_dir = OUT / "_tmp"
    temp_dir.mkdir(exist_ok=True)
    try:
        gap_summary, gap_yearly, gap_trades = reproduce_gap(temp_dir)
        climax_summary, climax_yearly, climax_trades = reproduce_climax(temp_dir)
    finally:
        for path in temp_dir.glob("*_is.parquet"):
            path.unlink(missing_ok=True)
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    summary = pd.concat([gap_summary, climax_summary], ignore_index=True)
    yearly = pd.concat([gap_yearly, climax_yearly], ignore_index=True)
    summary.to_csv(OUT / "step0_is_summary.csv", index=False)
    yearly.to_csv(OUT / "step0_is_yearly.csv", index=False)
    gap_trades.to_csv(OUT / "step0_signal_a_trades_is.csv", index=False)
    climax_trades.to_csv(OUT / "step0_signal_b_trades_is.csv", index=False)
    metadata = {
        "scope": "IS_ONLY",
        "oos_statistics_computed": False,
        "signal_a_reference": "mean_reversion_cross_asset_20261006 frozen aggregate rows",
        "signal_b_reference": str(CLIMAX_REFERENCE.relative_to(REPO)),
        "signal_b_trade_by_trade_match": True,
        "anti_oos_assertions": ["bar session_date", "trade session_date", "signal_time", "entry_time", "exit_time"],
    }
    (OUT / "step0_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
