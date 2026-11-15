"""Thin read-only wrapper around the frozen netting engine of book_aggregation_risk_r1_20261112.

The study module is imported (its code and spec are hash-checked by its own loaders); bars are re-read
here with an explicit upper cutoff so that no minute after ``cutoff`` is ever loaded.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
from functools import lru_cache

import numpy as np
import pandas as pd

from .common import ROOT, CACHE

_STUDY = ROOT / "research" / "book_aggregation_risk_r1_20261112" / "run_study.py"
_spec = importlib.util.spec_from_file_location("audit_netting_run_study", _STUDY)
RS = importlib.util.module_from_spec(_spec)
import sys as _sys  # noqa: E402
_sys.modules[_spec.name] = RS
_spec.loader.exec_module(RS)

from src.portfolio.aggregator import RiskLimits, simulate_session  # noqa: E402

SLEEVES8 = RS.SLEEVES  # S01 S02 S03 S05 S06 S07 S08 S09


@lru_cache(maxsize=1)
def spec() -> dict:
    s = RS.load_spec()
    RS.verify_inputs(s)
    return s


@lru_cache(maxsize=1)
def trades() -> pd.DataFrame:
    return RS.load_trades(spec())


@lru_cache(maxsize=1)
def calendars() -> dict:
    return RS.load_calendars(spec())[0]


@lru_cache(maxsize=4)
def bars(cutoff: str = "2026-03-19") -> dict:
    """1-minute OHLC per instrument, strictly before cutoff + 1 day 00:00 ET (cached as parquet)."""
    out = {}
    end = pd.Timestamp(cutoff, tz=RS.ET) + pd.Timedelta(days=1)
    for name in RS.INSTRUMENTS:
        cache = CACHE / f"bars_{name}_to_{cutoff}.parquet"
        if cache.exists():
            out[name] = pd.read_parquet(cache)
            continue
        path = ROOT / spec()["instruments"][name]["bars_1m"]
        import pyarrow.parquet as pq
        sch = pq.read_schema(path)
        idx_name = [f for f in sch.names if f in ("timestamp", "ts_event")]
        if idx_name:
            col = idx_name[0]
            tz = str(sch.field(col).type.tz)
            b = pd.read_parquet(path, columns=["open", "high", "low", "close"],
                                filters=[(col, "<", end.tz_convert(tz))])
        else:  # pragma: no cover
            raise RuntimeError(f"no timestamp column in {path}")
        b.index = b.index.tz_convert("UTC")
        b = b[~b.index.duplicated(keep="last")].sort_index()
        if len(b) and b.index.max() >= end.tz_convert("UTC"):
            raise RuntimeError("cutoff violated")
        b.to_parquet(cache)
        out[name] = b
    return out


def union_dates(a: str, b: str) -> list[dt.date]:
    return RS.union_sessions(calendars(), RS.d(a), RS.d(b))


def b0_sizes(t: pd.DataFrame | None = None) -> np.ndarray:
    t = trades() if t is None else t
    return np.where(t.sleeve == "S01", t.qty_native, 1).astype(int)


def specs(mbt_lot_units: int = 10):
    return RS.instrument_specs(spec(), mbt_lot_units)


def run_book(a: str, b: str, lot_cap_units: int | None = None, cutoff: str = "2026-03-19",
             trades_df: pd.DataFrame | None = None, keep_paths: bool = False) -> dict:
    t = trades() if trades_df is None else trades_df
    dates = union_dates(a, b)
    cube = RS.BarCube(bars(cutoff), dates)
    events = RS.build_events(t, b0_sizes(t))
    lim = RiskLimits(lot_cap_units=lot_cap_units)
    res = [simulate_session(cube.session(x), events.get(x, RS.EMPTY_EVENTS), RS.SLEEVE_INSTRUMENT, specs(), lim,
                            keep_paths=keep_paths) for x in dates]
    final = np.array([r.final for r in res])
    return {"dates": [str(x) for x in dates], "sessions": len(dates), "total": float(final.sum()),
            "costs": float(sum(r.costs for r in res)), "gross": float(sum(r.gross for r in res)),
            "contracts": int(sum(r.contracts for r in res)), "final": final.tolist(),
            "instrument_final": np.array([r.instrument_final for r in res]).tolist(),
            "table": res}


def trade_sum_separate(a: str, b: str) -> dict:
    t = trades()
    m = (t.session_date >= RS.d(a)) & (t.session_date <= RS.d(b))
    by = t[m].groupby("sleeve")["net_native"].sum()
    return {"total": float(by.sum()), "by_sleeve": {k: float(v) for k, v in by.items()},
            "costs": float(t[m]["cost_native"].sum())}


@lru_cache(maxsize=1)
def trades_ref() -> pd.DataFrame:
    """Netting ledger of the reference book: S01 = 18-model ATR ensemble (lock eight_..._20261115)."""
    s = dict(spec())
    s["sleeves"] = dict(s["sleeves"])
    s["sleeves"]["S01"] = dict(s["sleeves"]["S01"], trades="research/signal_sheet_01_atr18/trades.csv")
    return RS.load_trades(s)


def run_ref(a: str, b: str, lot_cap_units: int | None = None, cutoff: str = "2026-03-19", keep_paths: bool = False) -> dict:
    return run_book(a, b, lot_cap_units=lot_cap_units, cutoff=cutoff, trades_df=trades_ref(), keep_paths=keep_paths)
