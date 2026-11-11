"""Cross-asset rolling intraday momentum (close.diff(n) sign, pure EMA crossover), phase 1.

Runs exactly FROZEN_SPECIFICATION.json (hash-checked against SPEC_LOCK.json).

    python run_phase1.py selftest   synthetic data only, no real return computed
    python run_phase1.py is         IS selection, tests, PRE holdout, gate
    python run_phase1.py oos        verdict; refused unless the IS gate passed
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
NW_LAGS = 5
SIGMA_WINDOW = 20
SIGMA_MIN = 15
STALE = 5
MAX_GAP_DAYS = 7
SEED = 20261111
N_PERM = 5000
CSCV_BLOCKS = 12

TFS = [5, 15, 30, 60]
NS = [1, 2, 4, 8, 16]
PAIRS = [(5, 20), (10, 40), (20, 80)]
FAMILIES = [("diff", len(TFS), len(NS)), ("ema", len(TFS), len(PAIRS))]
N_CELLS = sum(a * b for _, a, b in FAMILIES)


def cell_label(c: int) -> dict:
    if c < len(TFS) * len(NS):
        return {"index": c, "family": "diff", "tf_minutes": TFS[c // len(NS)], "n_bars": NS[c % len(NS)]}
    j = c - len(TFS) * len(NS)
    f, s = PAIRS[j % len(PAIRS)]
    return {"index": c, "family": "ema", "tf_minutes": TFS[j // len(PAIRS)], "fast": f, "slow": s}


# ------------------------------------------------------------------ spec / io
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_spec() -> dict:
    spec_path = HERE / "FROZEN_SPECIFICATION.json"
    lock = json.loads((HERE / "SPEC_LOCK.json").read_text())
    if sha256(spec_path) != lock["spec_sha256"]:
        raise RuntimeError("Specification changed after freeze")
    spec = json.loads(spec_path.read_text())
    g = spec["grid"]["families"]
    if g["diff"]["tf_minutes"] != TFS or g["diff"]["n_bars"] != NS or [tuple(p) for p in g["ema"]["fast_slow_bars"]] != PAIRS:
        raise RuntimeError("Grid constants differ from the spec")
    for item in list(spec["universe"].values()) + [spec["pre_sample_holdout_universe"]["NQ"]]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Input checksum mismatch: {item['path']}")
    return spec


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["open", "close", "instrument_id"])
    if df.index.tz is None:
        raise RuntimeError("Naive timestamps")
    df.index = df.index.tz_convert("America/New_York")
    return df.sort_index()


# ------------------------------------------------------------ session features
def session_frame(bars: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    L = int(cfg["session_minutes"])
    hh, mm = map(int, cfg["session_start"].split(":"))
    idx = bars.index
    m = idx.hour * 60 + idx.minute - (hh * 60 + mm)
    keep = (m >= 0) & (m < L) & (idx.dayofweek < 5)
    d = bars.loc[keep, ["open", "close", "instrument_id"]].copy()
    d["m"] = np.asarray(m[keep])
    d["date"] = d.index.tz_localize(None).normalize()
    g = d.groupby("date")
    day = pd.DataFrame({
        "count": g.size(),
        "n_inst": g["instrument_id"].nunique(),
        "open_px": d[d["m"] < STALE].groupby("date")["open"].first(),
        "exit_px": d[d["m"] >= L - STALE].groupby("date")["close"].last(),
    })
    day = day[(day["count"] >= 0.75 * L) & day["open_px"].notna() & day["exit_px"].notna() & (day["n_inst"] == 1)].copy()
    day["sigma_day"] = np.sqrt(
        (np.log(day["exit_px"] / day["open_px"]) ** 2).rolling(SIGMA_WINDOW, min_periods=SIGMA_MIN).mean().shift(1)
    )
    d = d[d["date"].isin(day.index)]
    return d, day


def tf_bars(d: pd.DataFrame, tf: int) -> pd.DataFrame:
    b = (d.assign(b=d["m"] // tf)
          .groupby(["date", "b"], sort=True)
          .agg(O=("open", "first"), C=("close", "last"), iid=("instrument_id", "last"))
          .reset_index())
    return b


def diff_signal(b: pd.DataFrame, n: int) -> np.ndarray:
    lc = np.log(b["C"])
    s = np.sign(lc - lc.shift(n))
    ok = (b["iid"] == b["iid"].shift(n)) & ((b["date"] - b["date"].shift(n)).dt.days <= MAX_GAP_DAYS)
    return np.where(ok, s, 0.0)


def ema_signal(b: pd.DataFrame, fast: int, slow: int) -> np.ndarray:
    lc = np.log(b["C"])
    brk = (b["iid"] != b["iid"].shift()) | ((b["date"] - b["date"].shift()).dt.days > MAX_GAP_DAYS)
    seg = brk.cumsum()
    ef = lc.groupby(seg).transform(lambda x: x.ewm(span=fast, adjust=False).mean())
    es = lc.groupby(seg).transform(lambda x: x.ewm(span=slow, adjust=False).mean())
    pos_in_seg = lc.groupby(seg).cumcount() + 1
    return np.where(pos_in_seg >= slow, np.sign(ef - es), 0.0)


def daily_pnl(b: pd.DataFrame, s: np.ndarray, exit_px: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Gross points and sides per date for target positions s decided at each tf bar close."""
    date = b["date"]
    last = date.ne(date.shift(-1)).to_numpy()
    p = np.where(last, 0.0, np.nan_to_num(s))
    A = b["O"].groupby(date).shift(-1)
    B = b["O"].groupby(date).shift(-2)
    next_is_last = date.eq(date.shift(-1)).to_numpy() & np.append(last[1:], True)
    B = np.where(next_is_last, exit_px.reindex(date).to_numpy(), B)
    pts = np.where(p != 0, p * (B - A.to_numpy()), 0.0)
    prev = pd.Series(p).groupby(date.to_numpy()).shift(1).fillna(0.0).to_numpy()
    sides = np.abs(p - prev)
    return (pd.Series(pts, index=date).groupby(level=0).sum(),
            pd.Series(sides, index=date).groupby(level=0).sum())


def lag_within_day(b: pd.DataFrame, s: np.ndarray) -> np.ndarray:
    return pd.Series(s).groupby(b["date"].to_numpy()).shift(1).fillna(0.0).to_numpy()


def instrument_table(bars: pd.DataFrame, cfg: dict) -> dict:
    d, day = session_frame(bars, cfg)
    side_cost = cfg["tick_size"] * cfg["point_value"] + cfg["commission_per_side"]
    dates = day.index
    T = len(dates)
    gross_usd = np.zeros((T, N_CELLS))
    sides = np.zeros((T, N_CELLS))
    lag_usd = np.zeros((T, N_CELLS))
    lag_sides = np.zeros((T, N_CELLS))
    for i, tf in enumerate(TFS):
        b = tf_bars(d, tf)
        sigs = {}
        for j, n in enumerate(NS):
            sigs[i * len(NS) + j] = diff_signal(b, n)
        for j, (f, s_) in enumerate(PAIRS):
            sigs[len(TFS) * len(NS) + i * len(PAIRS) + j] = ema_signal(b, f, s_)
        for c, s in sigs.items():
            pts, sd = daily_pnl(b, s, day["exit_px"])
            gross_usd[:, c] = pts.reindex(dates).fillna(0).to_numpy() * cfg["point_value"]
            sides[:, c] = sd.reindex(dates).fillna(0).to_numpy()
            pts_l, sd_l = daily_pnl(b, lag_within_day(b, s), day["exit_px"])
            lag_usd[:, c] = pts_l.reindex(dates).fillna(0).to_numpy() * cfg["point_value"]
            lag_sides[:, c] = sd_l.reindex(dates).fillna(0).to_numpy()
    denom = (cfg["point_value"] * day["open_px"] * day["sigma_day"]).to_numpy()[:, None]
    warm = day["sigma_day"].notna().to_numpy() & (day["sigma_day"].to_numpy() > 0)
    cost_usd = sides * side_cost
    tab = {
        "dates": dates[warm],
        "gross": (gross_usd / denom)[warm], "cost": (cost_usd / denom)[warm],
        "gross_lag": (lag_usd / denom)[warm], "cost_lag": (lag_sides * side_cost / denom)[warm],
        "gross_usd": gross_usd[warm], "cost_usd": cost_usd[warm], "sides": sides[warm],
    }
    return tab


def restrict(tab: dict, start: str, end: str) -> dict:
    mask = (tab["dates"] >= pd.Timestamp(start)) & (tab["dates"] <= pd.Timestamp(end))
    return {k: v[mask] for k, v in tab.items()}


# -------------------------------------------------------------------- pooling
def pool(tabs: dict[str, dict], clusters: dict[str, str], key: str):
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(t["dates"]) for t in tabs.values()])))
    total = np.zeros((len(all_dates), N_CELLS))
    active = np.zeros(len(all_dates))
    for cl in sorted(set(clusters.values())):
        s = np.zeros((len(all_dates), N_CELLS))
        n = np.zeros(len(all_dates))
        for a, t in tabs.items():
            if clusters[a] != cl:
                continue
            pos = all_dates.get_indexer(t["dates"])
            s[pos] += t[key]
            n[pos] += 1
        has = n > 0
        total[has] += s[has] / n[has, None]
        active += has
    return all_dates, total / active[:, None]


# ----------------------------------------------------------------- statistics
def hac_t(x: np.ndarray, lags: int = NW_LAGS) -> tuple[float, float]:
    x = np.asarray(x, float)
    n = len(x)
    mu = x.mean()
    u = x - mu
    s = u @ u
    for lag in range(1, lags + 1):
        s += 2 * (1 - lag / (lags + 1.0)) * (u[lag:] @ u[:-lag])
    se = np.sqrt(s) / n
    return float(mu), float(mu / se) if se > 0 else float("nan")


def daily_sharpe(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0, ddof=1)
    return np.divide(X.mean(axis=0), sd, out=np.zeros(X.shape[1]), where=sd > 0)


def _smooth_block(g: np.ndarray) -> np.ndarray:
    nh, nk = g.shape
    out = np.zeros_like(g)
    for i in range(nh):
        for j in range(nk):
            vals = [g[i, j]]
            if i > 0: vals.append(g[i - 1, j])
            if i < nh - 1: vals.append(g[i + 1, j])
            if j > 0: vals.append(g[i, j - 1])
            if j < nk - 1: vals.append(g[i, j + 1])
            out[i, j] = np.mean(vals)
    return out.ravel()


def smooth(sh: np.ndarray) -> np.ndarray:
    parts, start = [], 0
    for _, a, b in FAMILIES:
        parts.append(_smooth_block(sh[start:start + a * b].reshape(a, b)))
        start += a * b
    return np.concatenate(parts)


def select_cell(X: np.ndarray) -> tuple[int, np.ndarray]:
    sm = smooth(daily_sharpe(X))
    return int(np.argmax(sm)), sm


def permutation_p(G: np.ndarray, K: np.ndarray, observed: float) -> float:
    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(N_PERM):
        flip = rng.choice([-1.0, 1.0], size=G.shape[0])[:, None]
        hits += smooth(daily_sharpe(flip * G - K)).max() >= observed
    return (hits + 1) / (N_PERM + 1)


def cscv_pbo(X: np.ndarray) -> dict:
    blocks = np.array_split(np.arange(X.shape[0]), CSCV_BLOCKS)
    logits, test_sh = [], []
    for combo in itertools.combinations(range(CSCV_BLOCKS), CSCV_BLOCKS // 2):
        tr = np.concatenate([blocks[b] for b in combo])
        te = np.concatenate([blocks[b] for b in range(CSCV_BLOCKS) if b not in combo])
        chosen, _ = select_cell(X[tr])
        sh_te = daily_sharpe(X[te])
        w = ((sh_te < sh_te[chosen]).sum() + 1) / (N_CELLS + 1)
        logits.append(np.log(w / (1 - w)))
        test_sh.append(sh_te[chosen])
    logits = np.array(logits)
    return {"splits": len(logits), "pbo": float((logits <= 0).mean()), "median_logit": float(np.median(logits)),
            "share_test_sharpe_positive": float((np.array(test_sh) > 0).mean())}


def series_stats(x: np.ndarray) -> dict:
    mu, t = hac_t(x)
    sd = x.std(ddof=1)
    return {"days": int(len(x)), "mean_R": mu, "hac_t": t,
            "sharpe_annual": float(mu / sd * np.sqrt(252)) if sd > 0 else float("nan")}


def asset_stats(tab: dict, c: int) -> dict:
    net = tab["gross"][:, c] - tab["cost"][:, c]
    out = series_stats(net)
    out.update({
        "gross_mean_R": float(tab["gross"][:, c].mean()),
        "sides_per_day": float(tab["sides"][:, c].mean()),
        "gross_usd_per_day": float(tab["gross_usd"][:, c].mean()),
        "cost_usd_per_day": float(tab["cost_usd"][:, c].mean()),
        "net_usd_per_day": float((tab["gross_usd"][:, c] - tab["cost_usd"][:, c]).mean()),
    })
    return out


def evaluate(tabs: dict[str, dict], clusters: dict[str, str], c: int) -> dict:
    for t in tabs.values():
        t["net"] = t["gross"] - t["cost"]
        t["net_x2"] = t["gross"] - 2 * t["cost"]
        t["net_lag"] = t["gross_lag"] - t["cost_lag"]
    dates, X = pool(tabs, clusters, "net")
    x = X[:, c]
    yearly = pd.Series(x, index=dates).groupby(dates.year).sum()
    by_cluster = {}
    for cl in sorted(set(clusters.values())):
        sub = {a: t for a, t in tabs.items() if clusters[a] == cl}
        by_cluster[cl] = series_stats(pool(sub, {a: cl for a in sub}, "net")[1][:, c])
    return {
        "pooled": series_stats(x),
        "pooled_gross_mean_R": float(pool(tabs, clusters, "gross")[1][:, c].mean()),
        "pooled_cost_x2": series_stats(pool(tabs, clusters, "net_x2")[1][:, c]),
        "pooled_signal_lag_1bar": series_stats(pool(tabs, clusters, "net_lag")[1][:, c]),
        "yearly_sum_R": {int(y): float(v) for y, v in yearly.items()},
        "by_cluster": by_cluster,
        "by_instrument": {a: asset_stats(t, c) for a, t in tabs.items()},
    }


# --------------------------------------------------------------------- stages
def build(universe: dict, period: list[str]) -> tuple[dict, dict]:
    tabs, clusters = {}, {}
    for name, cfg in universe.items():
        print(f"  building {name} ...", flush=True)
        tabs[name] = restrict(instrument_table(load_bars(ROOT / cfg["path"]), cfg), *period)
        clusters[name] = cfg["cluster"]
    return tabs, clusters


def stage_is(spec: dict) -> None:
    OUT.mkdir(exist_ok=True)
    tabs, clusters = build(spec["universe"], spec["periods"]["IS"])
    for t in tabs.values():
        t["net"] = t["gross"] - t["cost"]
    dates, X = pool(tabs, clusters, "net")
    _, G = pool(tabs, clusters, "gross")
    _, K = pool(tabs, clusters, "cost")
    chosen, sm = select_cell(X)
    sh = daily_sharpe(X)

    rows = []
    for c in range(N_CELLS):
        mu, t = hac_t(X[:, c])
        row = {**cell_label(c), "pooled_mean_R": mu, "pooled_hac_t": t, "pooled_gross_mean_R": G[:, c].mean(),
               "sharpe_annual": sh[c] * np.sqrt(252), "smoothed_sharpe_annual": sm[c] * np.sqrt(252)}
        for a, tab in tabs.items():
            row[f"{a}_net_R"] = tab["net"][:, c].mean()
            row[f"{a}_gross_R"] = tab["gross"][:, c].mean()
            row[f"{a}_sides_day"] = tab["sides"][:, c].mean()
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "is_cell_table.csv", index=False)

    ev = evaluate(tabs, clusters, chosen)
    print("  permutation ...", flush=True)
    p_perm = permutation_p(G, K, float(sm.max()))
    print("  cscv ...", flush=True)
    cscv = cscv_pbo(X)
    pre_universe = {"NQ": spec["pre_sample_holdout_universe"]["NQ"], "MGC": spec["universe"]["MGC"]}
    pre_tabs, pre_clusters = build(pre_universe, spec["periods"]["PRE"])
    pre = evaluate(pre_tabs, pre_clusters, chosen)

    cl_pos = sum(v["mean_R"] > 0 for v in ev["by_cluster"].values())
    as_pos = sum(v["mean_R"] > 0 for v in ev["by_instrument"].values())
    years = ev["yearly_sum_R"]
    gates = {
        "G1_hac": bool(ev["pooled"]["mean_R"] > 0 and ev["pooled"]["hac_t"] >= 2.0),
        "G2_permutation": bool(p_perm <= 0.05),
        "G3_pbo": bool(cscv["pbo"] <= 0.35),
        "G4_breadth": bool(cl_pos >= 2 and as_pos >= 3),
        "G5_years": bool(sum(v > 0 for v in years.values()) > len(years) / 2),
        "G6_pre_holdout": bool(pre["pooled"]["mean_R"] > 0 and pre["pooled"]["hac_t"] >= 1.0),
    }
    result = {"stage": "IS", "period": spec["periods"]["IS"], "pooled_days": int(len(dates)),
              "selected_cell": cell_label(chosen), "selected_cell_IS": ev,
              "max_stat_permutation_p": p_perm, "cscv": cscv, "selected_cell_PRE": pre,
              "clusters_positive": cl_pos, "instruments_positive": as_pos,
              "gates": gates, "gate_pass": bool(all(gates.values()))}
    (OUT / "is_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def stage_oos(spec: dict) -> None:
    is_res = json.loads((OUT / "is_results.json").read_text())
    if not is_res["gate_pass"]:
        raise SystemExit("IS gate failed: OOS is not computed.")
    c = is_res["selected_cell"]["index"]
    tabs, clusters = build(spec["universe"], spec["periods"]["OOS"])
    ev = evaluate(tabs, clusters, c)
    ext_tabs, ext_clusters = build({a: spec["universe"][a] for a in ("MNQ", "MGC", "MBT")}, spec["periods"]["EXT_informational"])
    ext = evaluate(ext_tabs, ext_clusters, c)
    cl_pos = sum(v["mean_R"] > 0 for v in ev["by_cluster"].values())
    verdict = bool(ev["pooled"]["mean_R"] > 0 and ev["pooled"]["hac_t"] >= 1.0 and cl_pos >= 2)
    result = {"stage": "OOS", "selected_cell": is_res["selected_cell"], "OOS": ev,
              "clusters_positive": cl_pos, "verdict_pass": verdict, "EXT_informational": ext}
    (OUT / "oos_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


# ------------------------------------------------------------------ self-test
def _synthetic(phi: float, seed: int, days: int = 400) -> pd.DataFrame:
    """15-minute block returns follow an AR(1) with coefficient phi (carried across days)."""
    rng = np.random.default_rng(seed)
    idx, opens, closes = [], [], []
    px, prev_block = 100.0, 0.0
    for d in pd.bdate_range("2020-01-01", periods=days):
        r = rng.normal(0, 0.0004, 390)
        for blk in range(26):
            sl = slice(blk * 15, (blk + 1) * 15)
            r[sl] += phi * prev_block / 15
            prev_block = r[sl].sum()
        path = px * np.exp(np.cumsum(r))
        o = np.concatenate([[px], path[:-1]])
        idx.append(pd.date_range(d + pd.Timedelta("9h30m"), periods=390, freq="min", tz="America/New_York"))
        opens.append(o); closes.append(path)
        px = path[-1]
    return pd.DataFrame({"open": np.concatenate(opens), "close": np.concatenate(closes), "instrument_id": 1},
                        index=idx[0].append(idx[1:]))


def selftest() -> None:
    cfg = {"session_start": "09:30", "session_minutes": 390, "point_value": 1.0, "tick_size": 0.0, "commission_per_side": 0.0}
    bars0 = _synthetic(0.0, 1)
    # manual loop reconstruction of one day, diff tf=15 n=1, and of the side count
    d, day = session_frame(bars0, cfg)
    b = tf_bars(d, 15)
    s = diff_signal(b, 1)
    pts, sides = daily_pnl(b, s, day["exit_px"])
    dt = day.index[30]
    rows = b.index[b["date"] == dt]
    tot, pos, n_sides = 0.0, 0.0, 0.0
    for k_pos, k in enumerate(rows):
        if k_pos == len(rows) - 1:
            n_sides += abs(pos)
            tot += pos * (day.loc[dt, "exit_px"] - b.loc[k, "O"])
            break
        target = s[k]
        fill = b.loc[rows[k_pos + 1], "O"]
        if k_pos > 0:
            tot += pos * (fill - b.loc[k, "O"])
        n_sides += abs(target - pos)
        pos = target
    # bar-by-bar accrual above starts at bar 1's open; daily_pnl does the same thing
    assert abs(pts[dt] - tot) < 1e-9, (pts[dt], tot)
    assert abs(sides[dt] - n_sides) < 1e-9, (sides[dt], n_sides)
    # signal causality: changing the future must not change past signals
    b2 = b.copy(); b2.loc[b2.index[-50:], "C"] *= 1.5
    assert np.array_equal(diff_signal(b, 4)[:-50], diff_signal(b2, 4)[:-50])
    assert np.array_equal(ema_signal(b, 10, 40)[:-50], ema_signal(b2, 10, 40)[:-50])
    # roll handling: an instrument change zeroes diff across it and restarts the EMA
    b3 = b.copy(); b3.loc[b3.index[500:], "iid"] = 2
    assert (diff_signal(b3, 4)[500:504] == 0).all() and (ema_signal(b3, 5, 20)[500:519] == 0).all()
    # power: null vs momentum world
    t0 = instrument_table(bars0, cfg)
    t1 = instrument_table(_synthetic(0.3, 2), cfg)
    c15 = 1 * len(NS) + 0       # diff, tf 15, n 1
    _, tn = hac_t(t0["gross"][:, c15])
    _, tm = hac_t(t1["gross"][:, c15])
    assert abs(tn) < 2.5 and tm > 4, (tn, tm)
    ch1, sm1 = select_cell(t1["gross"])
    p1 = permutation_p(t1["gross"], t1["cost"], float(sm1.max()))
    p0 = permutation_p(t0["gross"], t0["cost"], float(select_cell(t0["gross"])[1].max()))
    assert p1 < 0.01 and p0 > 0.05, (p1, p0)
    # pooling: two instruments in one cluster count once
    _, X = pool({"A": t0, "B": t0, "C": t1}, {"A": "x", "B": "x", "C": "y"}, "gross")
    assert np.allclose(X, (t0["gross"] + t1["gross"]) / 2)
    x = np.random.default_rng(0).normal(0.1, 1, 400)
    assert abs(hac_t(x, 0)[1] - x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))) < 1e-9
    print(f"selftest ok: null t={tn:.2f}, momentum t={tm:.2f}, p_null={p0:.3f}, p_mom={p1:.4f}, chosen={cell_label(ch1)}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "selftest":
        selftest()
    elif stage == "is":
        stage_is(load_spec())
    elif stage == "oos":
        stage_oos(load_spec())
    else:
        raise SystemExit(__doc__)
