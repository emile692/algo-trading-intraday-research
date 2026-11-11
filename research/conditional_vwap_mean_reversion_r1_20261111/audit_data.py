"""R1 - Conditional VWAP mean reversion: data audit.

Coverage, activity profile and unsigned scales only. No strategy return, no signed
post-signal return is computed here. Safe to run before the specification is frozen.

    python audit_data.py      -> ./audit/
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "audit"
IS = ("2019-05-06", "2023-12-31")

INSTRUMENTS = {
    "MNQ": dict(path="data/processed/parquet/MNQ_c_0_1m_20190505_20261003.parquet", pv=2.0, tick=0.25, comm=1.25, session=("09:30", "16:00")),
    "MES": dict(path="data/processed/parquet/MES_c_0_1m_20260322_135702.parquet", pv=5.0, tick=0.25, comm=1.25, session=("09:30", "16:00")),
    "MGC": dict(path="data/processed/parquet/MGC_v_0_1m_20150101_20261003.parquet", pv=10.0, tick=0.1, comm=1.25, session=("08:20", "13:30")),
    "MCL": dict(path="data/downloaded/MCL_v_0_1m_20210712_20261111.parquet", pv=100.0, tick=0.01, comm=1.25, session=("09:00", "14:30")),
    "NQ": dict(path="data/processed/parquet/NQ_c_0_1m_20260321_095036.parquet", pv=20.0, tick=0.25, comm=1.25, session=("09:30", "16:00")),
    "CL": dict(path="data/downloaded/CL_v_0_1m_20150101_20210712.parquet", pv=100.0, tick=0.01, comm=1.25, session=("09:00", "14:30")),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def hm(s: str) -> int:
    a, b = map(int, s.split(":"))
    return a * 60 + b


def main() -> None:
    OUT.mkdir(exist_ok=True)
    summary, prof, years, meta = [], [], [], {}
    for name, cfg in INSTRUMENTS.items():
        path = ROOT / cfg["path"]
        meta[name] = {"path": cfg["path"], "sha256": sha256(path)}
        df = pd.read_parquet(path, columns=["open", "high", "low", "close", "volume", "instrument_id"])
        df.index = df.index.tz_convert("America/New_York")
        w = df.index.tz_localize(None)
        m = np.asarray(w.hour * 60 + w.minute)
        s0, s1 = map(hm, cfg["session"])
        keep = (m >= s0) & (m < s1) & (np.asarray(w.dayofweek) < 5)
        x = df[keep].copy()
        x["m"] = m[keep]
        x["date"] = w[keep].normalize()
        n_min = s1 - s0
        # coverage per session
        g = x.groupby("date")
        cov = g.size() / n_min
        full = cov[cov >= 0.9].index
        x = x[x["date"].isin(full)]
        yr = pd.DataFrame({"sessions_any": g.size().groupby(g.size().index.year).size(),
                           "sessions_90pct": pd.Series(1, index=full).groupby(full.year).size(),
                           "median_minute_coverage": cov.groupby(cov.index.year).median()})
        yr.insert(0, "instrument", name)
        years.append(yr.reset_index(names="year"))
        # half-hour activity profile (median share of session volume), IS dates or whole file
        xi = x[(x["date"] >= IS[0]) & (x["date"] <= IS[1])] if name not in ("NQ", "CL") else x
        tot = xi.groupby("date")["volume"].transform("sum")
        xi = xi.assign(share=xi["volume"] / tot, bucket=(xi["m"] // 30) * 30)
        pr = xi.groupby(["date", "bucket"])["share"].sum().groupby("bucket").median()
        rng = (xi["high"] - xi["low"]).groupby(xi["bucket"]).mean() / cfg["tick"]
        for b, v in pr.items():
            prof.append({"instrument": name, "bucket": f"{b // 60:02d}:{b % 60:02d}", "median_volume_share_pct": 100 * v,
                         "mean_1m_range_ticks": rng.get(b, np.nan),
                         "zero_or_missing_minutes_pct": 100 * (1 - xi[xi["bucket"] == b].groupby("date").size().median() / 30)})
        # 5-minute bars, session VWAP, causal ATR14, unsigned deviation (IS window for micros)
        b5 = xi.copy()
        b5["t5"] = b5.index.floor("5min")
        a = b5.groupby("t5").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                                 volume=("volume", "sum"), date=("date", "first"))
        tp = (a["high"] + a["low"] + a["close"]) / 3
        a["vwap"] = (tp * a["volume"]).groupby(a["date"]).cumsum() / a["volume"].groupby(a["date"]).cumsum()
        pc = a.groupby("date")["close"].shift(1)
        tr = pd.concat([a["high"] - a["low"], (a["high"] - pc).abs(), (a["low"] - pc).abs()], axis=1).max(axis=1)
        a["atr"] = tr.groupby(a["date"]).transform(lambda s: s.rolling(14, min_periods=8).mean())
        a = a.dropna(subset=["atr", "vwap"])
        a = a[a["atr"] > 0]
        dev_atr = ((a["close"] - a["vwap"]).abs() / a["atr"])
        dev_bp = ((a["close"] - a["vwap"]).abs() / a["close"] * 1e4)
        cost = (2 * cfg["tick"] * cfg["pv"] + 2 * cfg["comm"])
        cost_bp = (cost / (cfg["pv"] * a["close"]) * 1e4)
        atr_bp = a["atr"] / a["close"] * 1e4
        row = {"instrument": name, "window": "IS" if name not in ("NQ", "CL") else "full_file", "bars_5m": len(a),
               "sessions": int(a["date"].nunique()), "round_trip_cost_usd": cost, "cost_bp_median": cost_bp.median(),
               "atr5_bp_median": atr_bp.median(), "cost_in_atr5_median": (cost_bp / atr_bp).median()}
        for q in (0.5, 0.9, 0.95, 0.99):
            row[f"dev_atr_q{int(q * 100)}"] = dev_atr.quantile(q)
        for z in (1.0, 1.5, 2.0, 2.5, 3.0):
            sel = dev_atr >= z
            row[f"bars_per_session_dev_ge_{z}"] = sel.sum() / a["date"].nunique()
            row[f"dev_bp_median_given_ge_{z}"] = dev_bp[sel].median()
            row[f"cost_over_dev_given_ge_{z}"] = (cost_bp[sel] / dev_bp[sel]).median()
        summary.append(row)
        print(name, "done", flush=True)
    s = pd.DataFrame(summary)
    s.round(4).to_csv(OUT / "scale_summary.csv", index=False)
    pd.DataFrame(prof).round(3).to_csv(OUT / "activity_profile.csv", index=False)
    pd.concat(years).to_csv(OUT / "session_coverage_yearly.csv", index=False)
    (OUT / "files.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 250, "display.max_columns", 60)
    print(s.round(3).T.to_string())
    p = pd.DataFrame(prof)
    print(p.pivot(index="bucket", columns="instrument", values="median_volume_share_pct").round(2).to_string())
    print(p.pivot(index="bucket", columns="instrument", values="zero_or_missing_minutes_pct").round(1).to_string())
    print(pd.concat(years).pivot(index="year", columns="instrument", values="sessions_90pct").to_string())


if __name__ == "__main__":
    main()
