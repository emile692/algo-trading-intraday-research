"""R1 - unsigned distribution of the candidate normalisations on IS (no return of any kind).

Used only to translate thresholds into event frequencies before the freeze.
    python audit_thresholds.py   -> ./audit/threshold_frequencies.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IS = ("2019-05-06", "2023-12-31")
INST = {
    "MNQ": ("data/processed/parquet/MNQ_c_0_1m_20190505_20261003.parquet", "09:30", "16:00"),
    "MGC": ("data/processed/parquet/MGC_v_0_1m_20150101_20261003.parquet", "08:20", "13:30"),
    "MCL": ("data/downloaded/MCL_v_0_1m_20210712_20261111.parquet", "09:00", "14:30"),
}


def bars5(path: str, s0: str, s1: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / path, columns=["open", "high", "low", "close", "volume"])
    df.index = df.index.tz_convert("America/New_York")
    loc = df.index.tz_localize(None)
    hhmm = loc.strftime("%H:%M")
    df = df[(hhmm >= s0) & (hhmm < s1) & (loc.dayofweek < 5) & (loc >= IS[0]) & (loc <= IS[1] + " 23:59")]
    df["date"] = df.index.tz_localize(None).normalize()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cv = df["volume"].groupby(df["date"]).cumsum()
    df["vwap"] = (tp * df["volume"]).groupby(df["date"]).cumsum() / cv
    df["sig_vw"] = np.sqrt(((df["volume"] * (tp - df["vwap"]) ** 2).groupby(df["date"]).cumsum()) / cv)
    a = df.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "vwap": "last", "sig_vw": "last", "date": "first"}
    ).dropna(subset=["close"])
    return a


def main() -> None:
    rows = []
    for name, (path, s0, s1) in INST.items():
        a = bars5(path, s0, s1)
        n_min = (pd.Timestamp(s1) - pd.Timestamp(s0)).seconds / 60
        t = a.index.tz_localize(None)
        el = (t - t.normalize() - pd.Timedelta(s0 + ":00")).total_seconds() / 60 + 5      # minutes elapsed at bar close
        a["tau"] = np.clip(el / n_min, 0, 1)
        day = a.groupby("date").agg(o=("open", "first"), c=("close", "last"))
        sd = np.log(day["c"] / day["o"]).rolling(20, min_periods=15).std().shift(1)        # previous 20 sessions only
        a["sd_d"] = a["date"].map(sd)
        a["n1"] = (a["close"] - a["vwap"]) / a["sig_vw"]
        a["n2"] = (a["close"] / a["vwap"] - 1) / (a["sd_d"] * np.sqrt(a["tau"]))
        so = a.groupby("date")["open"].transform("first")
        path_len = a["close"].diff().abs().groupby(a["date"]).cumsum()
        a["er"] = (a["close"] - so).abs() / path_len
        ok = (el >= 30) & (el <= n_min - 30)
        b = a[ok & np.isfinite(a["n1"]) & np.isfinite(a["n2"])]
        ns = b["date"].nunique()
        lunch = (t[ok & np.isfinite(a["n1"]) & np.isfinite(a["n2"])].strftime("%H:%M") >= "11:30") & \
                (t[ok & np.isfinite(a["n1"]) & np.isfinite(a["n2"])].strftime("%H:%M") < "13:30")
        for col, zs in [("n1", (1.5, 2.0, 2.5, 3.0)), ("n2", (0.5, 0.75, 1.0, 1.5))]:
            for z in zs:
                hit = b[col].abs() >= z
                first = hit & ~hit.groupby(b["date"]).shift(1, fill_value=False)       # crossings, not bars
                rows.append({"instrument": name, "norm": col, "z": z, "bars_pct": 100 * hit.mean(),
                             "crossings_per_session": first.sum() / ns,
                             "sessions_with_crossing_pct": 100 * first.groupby(b["date"]).any().mean(),
                             "lunch_crossings_per_session": (first & lunch).sum() / ns,
                             "er_median_at_crossing": b.loc[first, "er"].median()})
        rows.append({"instrument": name, "norm": "er_all_bars", "z": np.nan, "bars_pct": np.nan,
                     "crossings_per_session": np.nan, "sessions_with_crossing_pct": np.nan,
                     "lunch_crossings_per_session": np.nan, "er_median_at_crossing": b["er"].median()})
        print(name, "done", flush=True)
    r = pd.DataFrame(rows)
    r.round(3).to_csv(HERE / "audit" / "threshold_frequencies.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
