"""Feasibility check for a daily macro-quadrant regime built from free nowcasts.

No price, return or P&L data is read here. The script only answers:
  * are the free sources available point-in-time over PRE / IS?
  * how often do they update, and how persistent is the notebook-style
    quadrant regime (sign of the `window`-day change of a growth and an
    inflation nowcast) built from them?
  * how many independent regime episodes would an IS test actually have?

Sources (raw files in data/downloaded/macro_nowcasts_20261111/, not committed):
  * Growth: Atlanta Fed GDPNow, sheet TrackingArchives (real-time archive since
    2014-05; one target quarter per forecast date).
  * Inflation A: Cleveland Fed Inflation Nowcasting, headline CPI YoY
    (nowcast_year.json; one chart per target month, daily nowcasts).
  * Inflation B: 10y breakeven = Treasury par nominal 10y - real 10y
    (home.treasury.gov daily curves; FRED T10YIE was unreachable).

Regime logic is a literal port of the notebook's ComputeRegimes (lag, window,
turnoverAdj), with US GDP only (no China). It is causal because toAdj < lag:
the confirmation look-ahead stays within the lag.

Usage: python feasibility.py   (needs pandas + openpyxl)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "downloaded" / "macro_nowcasts_20261111"
OUT = Path(__file__).resolve().parent / "results"

PRE = ("2014-07-01", "2019-04-30")
IS = ("2019-05-01", "2023-12-31")
NOTEBOOK = dict(lag=4, window=5, toAdj=2)
WINDOWS = [5, 10, 20]  # persistence sensitivity only (no returns involved)
QUADRANTS = {0: "slow_growth", 1: "goldilocks", 2: "stagflation", 3: "heating_up"}


# --------------------------------------------------------------------------- loaders
def load_gdpnow() -> pd.DataFrame:
    ta = pd.read_excel(RAW / "GDPTrackingModelDataAndForecasts.xlsx", sheet_name="TrackingArchives")
    df = ta[["Forecast Date", "Quarter being forecasted", "GDP Nowcast"]].dropna()
    df.columns = ["date", "target", "value"]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["target"] = pd.to_datetime(df["target"]).dt.normalize()
    return df.drop_duplicates(["date", "target"], keep="last").sort_values(["date", "target"])


def _label_date(label: str, target: pd.Timestamp) -> pd.Timestamp:
    m, d = (int(x) for x in label.split("/"))
    y = target.year
    if m - target.month <= -6:  # e.g. target Dec, nowcast made in Jan
        y += 1
    elif m - target.month >= 6:  # e.g. target Jan, nowcast made in Dec
        y -= 1
    return pd.Timestamp(year=y, month=m, day=d)


def load_cleveland(kind: str = "year", series: str = "CPI Inflation") -> pd.DataFrame:
    charts = json.loads((RAW / f"nowcast_{kind}.json").read_text(encoding="utf-8"))
    rows = []
    for c in charts:
        target = pd.Timestamp(c["chart"]["subcaption"] + "-01")
        labels = [x["label"] for x in c["categories"][0]["category"] if not x.get("vline")]
        ds = next(s for s in c["dataset"] if s["seriesname"] == series)
        vals = [x.get("value", "") for x in ds["data"]]
        if len(vals) != len(labels):
            raise ValueError(f"label/value mismatch for {target:%Y-%m}: {len(labels)} vs {len(vals)}")
        for lab, v in zip(labels, vals):
            if v not in ("", None):
                rows.append((_label_date(lab, target), target, float(v)))
    df = pd.DataFrame(rows, columns=["date", "target", "value"])
    return df.drop_duplicates(["date", "target"], keep="last").sort_values(["date", "target"])


def load_breakeven() -> pd.DataFrame:
    def read(kind: str, col: str) -> pd.Series:
        parts = []
        for f in sorted((RAW / "treasury").glob(f"tr_{kind}_*.csv")):
            t = pd.read_csv(f)
            parts.append(pd.Series(t[col].values, index=pd.to_datetime(t["Date"], format="%m/%d/%Y")))
        return pd.concat(parts).sort_index().groupby(level=0).last()

    be = (read("daily_treasury_yield_curve", "10 Yr") - read("daily_treasury_real_yield_curve", "10 YR")).dropna()
    return pd.DataFrame({"date": be.index, "target": pd.NaT, "value": be.values})


# --------------------------------------------------------------------------- series
def chained_index(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Continuous index whose increments are within-target revisions only.

    For each publication date keep the latest live target; the increment is the
    change versus the previous publication of that same target, 0 on a target roll.
    Returns (index on publication dates, boolean 'new value published' flag).
    """
    if df["target"].isna().all():  # market series, no target
        s = df.set_index("date")["value"]
        return s - s.iloc[0], s.diff().fillna(0).ne(0)
    cur = df.sort_values(["date", "target"]).groupby("date").tail(1).set_index("date")
    prev_same = df.set_index(["target", "date"])["value"].groupby(level=0).shift(1)
    prev_same.index = prev_same.index.swaplevel()
    inc = []
    for d, row in cur.iterrows():
        p = prev_same.get((d, row["target"]), np.nan)
        inc.append(0.0 if np.isnan(p) else row["value"] - p)
    inc = pd.Series(inc, index=cur.index)
    return inc.cumsum(), inc.ne(0)


def to_bdays(s: pd.Series, start: str, end: str) -> pd.Series:
    """Value known at the close of each business day (publication date d -> usable from d)."""
    idx = pd.bdate_range(start, end)
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def compute_regimes(q3: pd.DataFrame, lag: int, window: int, toAdj: int) -> pd.Series:
    """Literal port of the notebook's ComputeRegimes, US GDP only."""
    assert toAdj < lag, "toAdj must be < lag to stay causal"
    idx = pd.date_range(q3.index.min(), q3.index.max() + pd.Timedelta(days=toAdj), freq="B")
    signal = q3.diff(window).apply(np.sign).clip(0, 1).reindex(idx).shift(lag)
    regime = (2 * signal["CPI"] + signal["GDP"]).dropna()
    adj = regime.copy()
    r, a = regime.to_numpy(), adj.to_numpy().copy()
    for i in range(1, len(a) - toAdj - 1):
        if r[i] != a[i - 1] and len(np.unique(r[i : i + toAdj + 1])) != 1:
            a[i] = a[i - 1]
    return pd.Series(a, index=regime.index).iloc[: len(a) - toAdj]


# --------------------------------------------------------------------------- stats
def episode_stats(reg: pd.Series, start: str, end: str) -> dict:
    r = reg.loc[start:end].dropna().astype(int)
    run_id = r.ne(r.shift()).cumsum()
    runs = r.groupby(run_id).agg(["first", "size"])
    lens = runs["size"]
    out = {
        "bdays": int(len(r)),
        "switches": int(r.ne(r.shift()).sum() - 1),
        "episodes": int(len(runs)),
        "episode_len_median": float(lens.median()),
        "episode_len_mean": round(float(lens.mean()), 1),
        "episode_len_p10": float(lens.quantile(0.1)),
        "episode_len_p90": float(lens.quantile(0.9)),
        "share_episodes_le_5d": round(float((lens <= 5).mean()), 3),
    }
    for k, name in QUADRANTS.items():
        out[f"days_share_{name}"] = round(float((r == k).mean()), 3)
        out[f"episodes_{name}"] = int((runs["first"] == k).sum())
    return out


def component_stats(sig_level: pd.Series, window: int, start: str, end: str) -> dict:
    d = sig_level.diff(window).loc[start:end]
    sign = np.sign(d).clip(0, 1)
    return {
        "zero_diff_share": round(float(d.eq(0).mean()), 3),
        "up_share": round(float(sign.mean()), 3),
        "sign_flips": int(sign.ne(sign.shift()).sum() - 1),
    }


def coverage(df: pd.DataFrame, flag: pd.Series, start: str, end: str) -> dict:
    pub = flag.loc[start:end]
    bd = pd.bdate_range(start, end)
    new = pub[pub].index
    gaps = pd.Series(new).diff().dt.days.dropna()
    return {
        "source_first_date": str(df["date"].min().date()),
        "source_last_date": str(df["date"].max().date()),
        "publication_days": int(len(pub)),
        "days_with_changed_value": int(pub.sum()),
        "share_bdays_with_changed_value": round(len(new) / len(bd), 3),
        "median_gap_days_between_changes": float(gaps.median()) if len(gaps) else None,
        "max_gap_days_between_changes": float(gaps.max()) if len(gaps) else None,
        "max_live_targets_per_date": int(df.groupby("date")["target"].nunique().max()) if df["target"].notna().any() else 1,
    }


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start, end = "2014-05-01", IS[1]
    raw = {"GDP_gdpnow": load_gdpnow(), "CPI_cleveland_yoy": load_cleveland("year", "CPI Inflation"),
           "CPI_breakeven10y": load_breakeven()}
    idx, flags, cov = {}, {}, {}
    for name, df in raw.items():
        i, f = chained_index(df)
        idx[name], flags[name] = to_bdays(i, start, end), f
        cov[name] = {w: coverage(df, f, *w_) for w, w_ in {"PRE": PRE, "IS": IS}.items()}

    rows, daily = [], {}
    for cpi_name in ["CPI_cleveland_yoy", "CPI_breakeven10y"]:
        q3 = pd.DataFrame({"GDP": idx["GDP_gdpnow"], "CPI": idx[cpi_name]}).dropna()
        for w in WINDOWS:
            reg = compute_regimes(q3, NOTEBOOK["lag"], w, NOTEBOOK["toAdj"])
            if w == NOTEBOOK["window"]:
                daily[f"regime_{cpi_name}"] = reg
            for per, (a, b) in {"PRE": PRE, "IS": IS}.items():
                row = {"inflation_source": cpi_name, "window": w, "period": per, **episode_stats(reg, a, b)}
                for comp, s in [("gdp", q3["GDP"]), ("cpi", q3["CPI"])]:
                    row.update({f"{comp}_{k}": v for k, v in component_stats(s, w, a, b).items()})
                rows.append(row)

    agree = {}
    for w in WINDOWS:
        sg = {k: np.sign(idx[k].diff(w)).clip(0, 1) for k in ["CPI_cleveland_yoy", "CPI_breakeven10y"]}
        for per, (a, b) in {"PRE": PRE, "IS": IS}.items():
            x, y = sg["CPI_cleveland_yoy"].loc[a:b], sg["CPI_breakeven10y"].loc[a:b]
            agree[f"w{w}_{per}_cpi_sign_agreement"] = round(float((x == y).mean()), 3)
    ra, rb = daily["regime_CPI_cleveland_yoy"], daily["regime_CPI_breakeven10y"]
    for per, (a, b) in {"PRE": PRE, "IS": IS}.items():
        agree[f"w{NOTEBOOK['window']}_{per}_regime_agreement"] = round(float((ra.loc[a:b] == rb.loc[a:b]).mean()), 3)

    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / "regime_persistence.csv", index=False)
    pd.DataFrame(daily).to_csv(OUT / "regime_daily_notebook_params.csv", index_label="date")
    meta = {
        "note": "No price/return/P&L data read. PRE/IS only; OOS not touched.",
        "notebook_params": NOTEBOOK, "windows": WINDOWS, "periods": {"PRE": PRE, "IS": IS},
        "coverage": cov,
        "inflation_source_agreement": agree,
        "raw_sha256": {str(p.relative_to(RAW)): sha256(p) for p in sorted(RAW.rglob("*")) if p.is_file()},
    }
    (OUT / "feasibility.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 60)
    print(json.dumps(agree, indent=1))
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
