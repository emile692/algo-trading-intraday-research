"""R2 - Market Intraday Momentum (Gao, Han, Li, Zhou 2018): data audit.

Coverage and scale only. This script never computes a return signed by the
signal, and never computes the first-half-hour return jointly with the
last-half-hour return. It can be run before the specification is frozen.

    python audit_data.py

Outputs go to ./audit/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "audit"
TOL = 5  # minutes of tolerance around a node, reported next to the exact-bar count

# clock = (session open, end of first half hour, entry, exit) in New York wall-clock minutes
EQUITY_CLOCK = (9 * 60 + 30, 10 * 60, 15 * 60 + 30, 16 * 60)
INSTRUMENTS = {
    "MNQ": dict(path="data/processed/parquet/MNQ_c_0_1m_20190505_20261003.parquet", pv=2.0, tick=0.25, comm=1.25,
                clocks={"equity_0930_1600": EQUITY_CLOCK}),
    "MES": dict(path="data/processed/parquet/MES_c_0_1m_20260322_135702.parquet", pv=5.0, tick=0.25, comm=1.25,
                clocks={"equity_0930_1600": EQUITY_CLOCK}),
    "NQ": dict(path="data/processed/parquet/NQ_c_0_1m_20260321_095036.parquet", pv=20.0, tick=0.25, comm=1.25,
               clocks={"equity_0930_1600": EQUITY_CLOCK}),
    "MGC": dict(path="data/processed/parquet/MGC_v_0_1m_20150101_20261003.parquet", pv=10.0, tick=0.1, comm=1.25,
                clocks={"equity_0930_1600": EQUITY_CLOCK,
                        "native_0820_1330": (8 * 60 + 20, 8 * 60 + 50, 13 * 60, 13 * 60 + 30)}),
    "MCL": dict(path="data/downloaded/MCL_v_0_1m_20210712_20261111.parquet", pv=100.0, tick=0.01, comm=1.25,
                clocks={"equity_0930_1600": EQUITY_CLOCK,
                        "native_0900_1430": (9 * 60, 9 * 60 + 30, 14 * 60, 14 * 60 + 30)}),
    "CL": dict(path="data/downloaded/CL_v_0_1m_20150101_20210712.parquet", pv=100.0, tick=0.01, comm=1.25,
               clocks={"equity_0930_1600": EQUITY_CLOCK,
                       "native_0900_1430": (9 * 60, 9 * 60 + 30, 14 * 60, 14 * 60 + 30)}),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["open", "high", "low", "close", "volume", "instrument_id"])
    if df.index.tz is None:
        raise RuntimeError("Naive timestamps")
    df.index = df.index.tz_convert("America/New_York")
    df = df.sort_index()
    w = df.index.tz_localize(None)
    df["m"] = w.hour * 60 + w.minute
    df["date"] = w.normalize()
    return df


def wide(d: pd.DataFrame, col: str, lo: int, hi: int, dates: pd.DatetimeIndex) -> np.ndarray:
    sub = d[(d["m"] >= lo) & (d["m"] < hi)]
    return (sub.pivot(index="date", columns="m", values=col)
            .reindex(index=dates, columns=range(lo, hi)).to_numpy(float))


def first_valid(b: np.ndarray) -> np.ndarray:
    out = np.full(b.shape[0], np.nan)
    for j in range(b.shape[1] - 1, -1, -1):
        out = np.where(np.isnan(b[:, j]), out, b[:, j])
    return out


def last_valid(b: np.ndarray) -> np.ndarray:
    return first_valid(b[:, ::-1])


def audit_clock(name: str, cfg: dict, d: pd.DataFrame, clock_name: str, clock: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    o, h, e, x = clock
    d = d[d["date"].dt.dayofweek < 5]
    dates = pd.DatetimeIndex(sorted(d["date"].unique()))
    day = pd.DataFrame(index=dates)
    day["instrument"] = name
    day["clock"] = clock_name

    def n_bars(lo, hi):
        return np.isfinite(wide(d, "close", lo, hi, dates)).sum(axis=1)

    day["bars_first30"] = n_bars(o, h)
    day["bars_last30"] = n_bars(e, x)
    day["bars_session"] = n_bars(o, x)
    # exact bars
    for label, minute, col in [("open_bar", o, "open"), ("sig_bar_exact", h - 1, "close"), ("next_bar_after_sig", h, "open"),
                               ("entry_bar_exact", e, "open"), ("exit_bar_exact", x - 1, "close"), ("bar_at_exit_time", x, "open")]:
        day[label] = np.isfinite(wide(d, col, minute, minute + 1, dates)[:, 0])
    # tolerant nodes
    p_open = first_valid(wide(d, "open", o, o + TOL, dates))
    p_sig = last_valid(wide(d, "close", h - TOL, h, dates))
    p_entry = first_valid(wide(d, "open", e, e + TOL, dates))
    p_exit = last_valid(wide(d, "close", x - TOL, x, dates))
    iid_exit = last_valid(wide(d, "instrument_id", x - TOL, x, dates))
    iid_sig = last_valid(wide(d, "instrument_id", h - TOL, h, dates))
    iid_entry = first_valid(wide(d, "instrument_id", e, e + TOL, dates))
    day["has_open"], day["has_sig"] = np.isfinite(p_open), np.isfinite(p_sig)
    day["has_entry"], day["has_exit"] = np.isfinite(p_entry), np.isfinite(p_exit)
    full = day["has_open"] & day["has_sig"] & day["has_entry"] & day["has_exit"]
    day["full_session"] = full
    # session type: no entry/exit node but a morning => early close or exchange holiday with Globex trading
    # last bar of the day session, to tell exchange holidays (Globex halt 13:00 ET) from early closes (13:15 ET)
    rth = d[(d["m"] >= o) & (d["m"] < 17 * 60)]
    day["last_bar_min"] = rth.groupby("date")["m"].max().reindex(dates).to_numpy()
    early = (~full & (day["last_bar_min"] == 13 * 60 + 14)).to_numpy() if clock_name.startswith("equity") else np.zeros(len(dates), bool)
    day["early_close_1300"] = early
    day["type"] = np.where(full, "full", np.where(early, "early_close", np.where(
        day["last_bar_min"] == 12 * 60 + 59, "holiday_globex_halt", np.where(day["bars_first30"] > 0, "data_gap", "no_rth"))))
    # previous close = price at the close of the most recent cash session: exit node of the last full session,
    # or the 12:59 close of an early-close session, at most 5 calendar days back (weekend + one holiday)
    p_early = last_valid(wide(d, "close", 13 * 60 - TOL, 13 * 60, dates))
    iid_early = last_valid(wide(d, "instrument_id", 13 * 60 - TOL, 13 * 60, dates))
    mark = np.where(full, p_exit, np.where(early, p_early, np.nan))
    mark_iid = np.where(full, iid_exit, np.where(early, iid_early, np.nan))
    mark_date = pd.Series(np.where(np.isfinite(mark), dates.asi8, np.nan), index=dates).ffill().shift(1)
    prev_exit = pd.Series(mark, index=dates).ffill().shift(1)
    prev_iid = pd.Series(mark_iid, index=dates).ffill().shift(1)
    prev_gap = (pd.Series(dates.asi8, index=dates) - mark_date) / 86400e9
    day["prev_close_ok"] = prev_exit.notna().to_numpy()
    day["prev_gap_days"] = prev_gap.to_numpy()
    day["prev_close_is_early"] = pd.Series(early, index=dates).shift(1, fill_value=False).to_numpy()
    day["roll_vs_prev_close"] = (prev_iid.to_numpy() != iid_sig) & prev_exit.notna().to_numpy() & np.isfinite(iid_sig)
    day["roll_intraday"] = np.isfinite(iid_sig) & np.isfinite(iid_exit) & ((iid_sig != iid_exit) | (iid_entry != iid_exit))
    day["eligible"] = full & day["prev_close_ok"] & ~day["roll_vs_prev_close"] & ~day["roll_intraday"] & (day["prev_gap_days"] <= 5)
    # volume and unsigned scale (no signal involved)
    vol_last = np.nan_to_num(wide(d, "volume", e, x, dates))
    day["vol_entry_min"] = vol_last[:, 0]
    day["vol_exit_min"] = vol_last[:, -1]
    day["vol_last30"] = vol_last.sum(axis=1)
    day["vol_first30"] = np.nan_to_num(wide(d, "volume", o, h, dates)).sum(axis=1)
    day["price_entry"] = p_entry
    with np.errstate(invalid="ignore", divide="ignore"):
        day["abs_last30_bp"] = np.abs(p_exit / p_entry - 1) * 1e4
        hi = np.nanmax(wide(d, "high", e, e + 1, dates), axis=1)
        lo = np.nanmin(wide(d, "low", e, e + 1, dates), axis=1)
        day["entry_bar_range_ticks"] = (hi - lo) / cfg["tick"]
    rt = 2 * cfg["tick"] * cfg["pv"] + 2 * cfg["comm"]
    day["cost_bp"] = rt / (cfg["pv"] * p_entry) * 1e4
    day["year"] = dates.year

    el = day[day["eligible"]]
    g = day.groupby("year")
    yr = pd.DataFrame({
        "weekdays_in_file": g.size(),
        "full_sessions": g["full_session"].sum(),
        "early_close": g["type"].apply(lambda s: (s == "early_close").sum()),
        "holiday_globex_halt": g["type"].apply(lambda s: (s == "holiday_globex_halt").sum()),
        "data_gap": g["type"].apply(lambda s: (s == "data_gap").sum()),
        "no_rth": g["type"].apply(lambda s: (s == "no_rth").sum()),
        "roll_excluded": g.apply(lambda t: ((t["roll_vs_prev_close"] | t["roll_intraday"]) & t["full_session"]).sum()),
        "eligible": g["eligible"].sum(),
        "exact_sig_bar_pct": g.apply(lambda t: 100 * t.loc[t["full_session"], "sig_bar_exact"].mean()),
        "exact_entry_bar_pct": g.apply(lambda t: 100 * t.loc[t["full_session"], "entry_bar_exact"].mean()),
        "exact_exit_bar_pct": g.apply(lambda t: 100 * t.loc[t["full_session"], "exit_bar_exact"].mean()),
        "bar_at_exit_time_pct": g.apply(lambda t: 100 * t.loc[t["full_session"], "bar_at_exit_time"].mean()),
        "median_bars_last30": el.groupby("year")["bars_last30"].median(),
        "median_vol_entry_min": el.groupby("year")["vol_entry_min"].median(),
        "median_vol_exit_min": el.groupby("year")["vol_exit_min"].median(),
        "p05_vol_entry_min": el.groupby("year")["vol_entry_min"].quantile(0.05),
        "median_vol_last30": el.groupby("year")["vol_last30"].median(),
        "median_entry_bar_range_ticks": el.groupby("year")["entry_bar_range_ticks"].median(),
        "median_abs_last30_bp": el.groupby("year")["abs_last30_bp"].median(),
        "mean_abs_last30_bp": el.groupby("year")["abs_last30_bp"].mean(),
        "cost_round_trip_bp": el.groupby("year")["cost_bp"].median(),
    }).round(2)
    yr.insert(0, "clock", clock_name)
    yr.insert(0, "instrument", name)
    return day, yr.reset_index()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    meta, days, years = {}, [], []
    for name, cfg in INSTRUMENTS.items():
        path = ROOT / cfg["path"]
        d = load(path)
        dup = int(d.index.duplicated().sum())
        meta[name] = dict(path=cfg["path"], sha256=sha256(path), rows=int(len(d)), first=str(d.index[0]), last=str(d.index[-1]),
                          duplicated_timestamps=dup, n_instrument_ids=int(d["instrument_id"].nunique()),
                          nonpositive_prices=int((d[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
                          zero_volume_bars=int((d["volume"] <= 0).sum()),
                          round_trip_cost_usd=2 * cfg["tick"] * cfg["pv"] + 2 * cfg["comm"])
        for cname, clock in cfg["clocks"].items():
            day, yr = audit_clock(name, cfg, d, cname, clock)
            days.append(day.rename_axis("date").reset_index())
            years.append(yr)
        print(name, "done")
    days = pd.concat(days, ignore_index=True)
    years = pd.concat(years, ignore_index=True)
    days.to_csv(OUT / "session_coverage_daily.csv", index=False)
    years.to_csv(OUT / "session_coverage_yearly.csv", index=False)
    atyp = days[days["type"] != "full"]
    atyp[["date", "instrument", "clock", "type", "bars_first30", "bars_session", "bars_last30", "last_bar_min"]].to_csv(
        OUT / "atypical_sessions.csv", index=False)
    tot = days.groupby(["instrument", "clock"]).agg(
        first=("date", "min"), last=("date", "max"), weekdays=("date", "size"), full=("full_session", "sum"),
        eligible=("eligible", "sum"), rolls=("roll_vs_prev_close", "sum")).reset_index()
    el = days[days["eligible"]]
    sc = el.groupby(["instrument", "clock"]).agg(
        mean_abs_last30_bp=("abs_last30_bp", "mean"), median_abs_last30_bp=("abs_last30_bp", "median"),
        cost_bp=("cost_bp", "median"), median_vol_entry_min=("vol_entry_min", "median"),
        p05_vol_entry_min=("vol_entry_min", lambda v: v.quantile(0.05)), median_vol_last30=("vol_last30", "median")).reset_index()
    tot = tot.merge(sc, on=["instrument", "clock"])
    tot["cost_over_mean_abs_move"] = tot["cost_bp"] / tot["mean_abs_last30_bp"]
    tot.round(3).to_csv(OUT / "summary.csv", index=False)
    (OUT / "files.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(tot.round(3).to_string())
    print(days.groupby(["instrument", "clock"])["type"].value_counts().unstack(fill_value=0).to_string())
    eq = days[(days["clock"] == "equity_0930_1600") & days["instrument"].isin(["MNQ", "MES", "NQ"])]
    ec = eq[eq["type"] == "early_close"]
    print("early-close month-day:", sorted(set(ec["date"].dt.strftime("%m-%d"))))
    print("data gaps:", eq[eq["type"] == "data_gap"][["date", "instrument", "bars_session", "last_bar_min"]].to_string())
    nr = eq[(eq["type"] == "no_rth") & (eq["instrument"] == "NQ")]["date"]
    print("NQ no_rth by month-day:", nr.dt.strftime("%m").value_counts().sort_index().to_dict())
    print(eq[eq["eligible"]].groupby(["instrument", "year"]).size().unstack(0).to_string())


if __name__ == "__main__":
    main()
