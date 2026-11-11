"""Cross-asset intraday time-series momentum, phase 1.

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
SEED = 20261111
N_PERM = 5000
CSCV_BLOCKS = 12


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
    items = list(spec["universe"].values()) + [spec["pre_sample_holdout_universe"]["NQ"]]
    for item in items:
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
def _first_valid(block: np.ndarray) -> np.ndarray:
    out = np.full(block.shape[0], np.nan)
    for j in range(block.shape[1] - 1, -1, -1):
        col = block[:, j]
        out = np.where(np.isnan(col), out, col)
    return out


def _last_valid(block: np.ndarray) -> np.ndarray:
    return _first_valid(block[:, ::-1])


def session_table(bars: pd.DataFrame, cfg: dict, grid_h: list[int], grid_k: list[float]) -> dict:
    """Per-session arrays for one instrument. Nothing here depends on the period split."""
    L = int(cfg["session_minutes"])
    hh, mm = map(int, cfg["session_start"].split(":"))
    idx = bars.index
    m = idx.hour * 60 + idx.minute - (hh * 60 + mm)
    keep = (m >= 0) & (m < L) & (idx.dayofweek < 5)
    d = bars.loc[keep, ["open", "close", "instrument_id"]].copy()
    d["m"] = np.asarray(m[keep])
    d["date"] = d.index.tz_localize(None).normalize()
    O = d.pivot(index="date", columns="m", values="open").reindex(columns=range(L)).to_numpy(float)
    grp = d.groupby("date")
    Cdf = d.pivot(index="date", columns="m", values="close").reindex(columns=range(L))
    dates = Cdf.index
    C = Cdf.to_numpy(float)
    count = grp["close"].size().reindex(dates).to_numpy()
    n_inst = grp["instrument_id"].nunique().reindex(dates).to_numpy()

    open_px = _first_valid(O[:, :STALE])
    exit_px = _last_valid(C[:, L - STALE:])
    eligible = (count >= 0.75 * L) & ~np.isnan(open_px) & ~np.isnan(exit_px) & (n_inst == 1)

    dates, O, C = dates[eligible], O[eligible], C[eligible]
    open_px, exit_px = open_px[eligible], exit_px[eligible]
    day_ret = np.log(exit_px / open_px)
    sigma_day = np.sqrt(
        pd.Series(day_ret ** 2).rolling(SIGMA_WINDOW, min_periods=SIGMA_MIN).mean().shift(1).to_numpy()
    )
    cost_usd = 2 * cfg["tick_size"] * cfg["point_value"] + 2 * cfg["commission_per_side"]

    n_cells = len(grid_h) * len(grid_k)
    T = len(dates)
    direction = np.zeros((T, n_cells))
    gross = np.zeros((T, n_cells))       # R units
    cost = np.zeros((T, n_cells))        # R units
    gross_delay = np.zeros((T, n_cells))
    usd_gross = np.zeros((T, n_cells))
    for i, h in enumerate(grid_h):
        sig_px = _last_valid(C[:, h - STALE:h])
        entry = _first_valid(O[:, h:h + STALE])
        entry_delay = _first_valid(O[:, h + 1:h + 1 + STALE])
        s = np.log(sig_px / open_px)
        sigma_h = np.sqrt(
            pd.Series(s ** 2).rolling(SIGMA_WINDOW, min_periods=SIGMA_MIN).mean().shift(1).to_numpy()
        )
        for j, k in enumerate(grid_k):
            c = i * len(grid_k) + j
            ok = ~np.isnan(s) & ~np.isnan(sigma_h) & ~np.isnan(entry) & ~np.isnan(sigma_day) & (sigma_day > 0)
            with np.errstate(invalid="ignore"):
                ok &= (np.abs(s) >= k * sigma_h) & (s != 0)
            dirn = np.where(ok, np.sign(s), 0.0)
            e = np.where(ok, entry, 1.0)
            ed = np.where(ok & ~np.isnan(entry_delay), entry_delay, e)
            sd = np.where(ok, sigma_day, 1.0)
            direction[:, c] = dirn
            gross[:, c] = dirn * (exit_px - e) / e / sd
            gross_delay[:, c] = dirn * (exit_px - ed) / ed / sd
            cost[:, c] = np.where(ok, cost_usd / (cfg["point_value"] * e) / sd, 0.0)
            usd_gross[:, c] = dirn * (exit_px - e) * cfg["point_value"]
    warm = ~np.isnan(sigma_day)
    return {
        "dates": dates[warm], "direction": direction[warm], "gross": gross[warm], "cost": cost[warm],
        "gross_delay": gross_delay[warm], "usd_gross": usd_gross[warm], "cost_usd": cost_usd,
    }


def restrict(tab: dict, start: str, end: str) -> dict:
    mask = (tab["dates"] >= pd.Timestamp(start)) & (tab["dates"] <= pd.Timestamp(end))
    out = {k: (v[mask] if isinstance(v, (np.ndarray, pd.DatetimeIndex)) else v) for k, v in tab.items()}
    return out


# -------------------------------------------------------------------- pooling
def pool(tabs: dict[str, dict], clusters: dict[str, str], key: str, sign: dict[str, np.ndarray] | None = None):
    """Cluster-balanced daily series. Returns (dates, matrix T x cells)."""
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(t["dates"]) for t in tabs.values()])))
    n_cells = next(iter(tabs.values()))[key].shape[1]
    names = sorted(set(clusters.values()))
    total = np.zeros((len(all_dates), n_cells))
    active = np.zeros(len(all_dates))
    for cl in names:
        s = np.zeros((len(all_dates), n_cells))
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


def smooth(sh: np.ndarray, nh: int, nk: int) -> np.ndarray:
    g = sh.reshape(nh, nk)
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


def select_cell(X: np.ndarray, nh: int, nk: int) -> tuple[int, np.ndarray]:
    sm = smooth(daily_sharpe(X), nh, nk)
    return int(np.argmax(sm)), sm   # argmax returns the first maximum: lower h, then lower k


def permutation_p(G: np.ndarray, K: np.ndarray, nh: int, nk: int, observed: float) -> float:
    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(N_PERM):
        flip = rng.choice([-1.0, 1.0], size=G.shape[0])[:, None]
        stat = smooth(daily_sharpe(flip * G - K), nh, nk).max()
        hits += stat >= observed
    return (hits + 1) / (N_PERM + 1)


def cscv_pbo(X: np.ndarray, nh: int, nk: int) -> dict:
    blocks = np.array_split(np.arange(X.shape[0]), CSCV_BLOCKS)
    n = X.shape[1]
    logits, test_sh = [], []
    for combo in itertools.combinations(range(CSCV_BLOCKS), CSCV_BLOCKS // 2):
        tr = np.concatenate([blocks[b] for b in combo])
        te = np.concatenate([blocks[b] for b in range(CSCV_BLOCKS) if b not in combo])
        chosen, _ = select_cell(X[tr], nh, nk)
        sh_te = daily_sharpe(X[te])
        rank = (sh_te < sh_te[chosen]).sum() + 1
        w = rank / (n + 1)
        logits.append(np.log(w / (1 - w)))
        test_sh.append(sh_te[chosen])
    logits = np.array(logits)
    return {
        "splits": len(logits),
        "pbo": float((logits <= 0).mean()),
        "median_logit": float(np.median(logits)),
        "share_test_sharpe_positive": float((np.array(test_sh) > 0).mean()),
    }


def series_stats(x: np.ndarray) -> dict:
    mu, t = hac_t(x)
    sd = x.std(ddof=1)
    return {
        "days": int(len(x)), "mean_R": mu, "hac_t": t,
        "sharpe_annual": float(mu / sd * np.sqrt(252)) if sd > 0 else float("nan"),
    }


def asset_stats(tab: dict, c: int) -> dict:
    net = tab["gross"][:, c] - tab["cost"][:, c]
    traded = tab["direction"][:, c] != 0
    usd = tab["usd_gross"][:, c] - traded * tab["cost_usd"]
    out = series_stats(net) if len(net) > NW_LAGS + 2 else {"days": int(len(net))}
    out.update({
        "trades": int(traded.sum()),
        "mean_net_usd_per_trade": float(usd[traded].mean()) if traded.any() else float("nan"),
        "mean_gross_usd_per_trade": float(tab["usd_gross"][traded, c].mean()) if traded.any() else float("nan"),
        "win_rate": float((usd[traded] > 0).mean()) if traded.any() else float("nan"),
        "long_mean_R": float(net[tab["direction"][:, c] > 0].mean()) if (tab["direction"][:, c] > 0).any() else float("nan"),
        "short_mean_R": float(net[tab["direction"][:, c] < 0].mean()) if (tab["direction"][:, c] < 0).any() else float("nan"),
    })
    return out


def evaluate(tabs: dict[str, dict], clusters: dict[str, str], c: int) -> dict:
    """Stats of one fixed cell on one period."""
    for t in tabs.values():
        t["net"] = t["gross"] - t["cost"]
        t["net_x2"] = t["gross"] - 2 * t["cost"]
        t["net_delay"] = t["gross_delay"] - t["cost"]
    dates, X = pool(tabs, clusters, "net")
    x = X[:, c]
    yearly = pd.Series(x, index=dates).groupby(dates.year).sum()
    by_cluster = {}
    for cl in sorted(set(clusters.values())):
        sub = {a: t for a, t in tabs.items() if clusters[a] == cl}
        if sub:
            _, Xc = pool(sub, {a: cl for a in sub}, "net")
            by_cluster[cl] = series_stats(Xc[:, c])
    return {
        "pooled": series_stats(x),
        "pooled_cost_x2": series_stats(pool(tabs, clusters, "net_x2")[1][:, c]),
        "pooled_entry_delay": series_stats(pool(tabs, clusters, "net_delay")[1][:, c]),
        "yearly_sum_R": {int(y): float(v) for y, v in yearly.items()},
        "by_cluster": by_cluster,
        "by_instrument": {a: asset_stats(t, c) for a, t in tabs.items()},
    }


# --------------------------------------------------------------------- stages
def build(spec: dict, universe: dict, period: list[str]) -> tuple[dict, dict]:
    gh, gk = spec["grid"]["h_minutes"], spec["grid"]["k_sigma"]
    tabs, clusters = {}, {}
    for name, cfg in universe.items():
        tab = session_table(load_bars(ROOT / cfg["path"]), cfg, gh, gk)
        tabs[name] = restrict(tab, *period)
        clusters[name] = cfg["cluster"]
    return tabs, clusters


def cell_label(spec: dict, c: int) -> dict:
    gh, gk = spec["grid"]["h_minutes"], spec["grid"]["k_sigma"]
    return {"index": c, "h_minutes": gh[c // len(gk)], "k_sigma": gk[c % len(gk)]}


def stage_is(spec: dict) -> None:
    OUT.mkdir(exist_ok=True)
    gh, gk = spec["grid"]["h_minutes"], spec["grid"]["k_sigma"]
    nh, nk = len(gh), len(gk)
    tabs, clusters = build(spec, spec["universe"], spec["periods"]["IS"])
    for t in tabs.values():
        t["net"] = t["gross"] - t["cost"]
    dates, X = pool(tabs, clusters, "net")
    _, G = pool(tabs, clusters, "gross")
    _, K = pool(tabs, clusters, "cost")
    chosen, sm = select_cell(X, nh, nk)
    sh = daily_sharpe(X)

    rows = []
    for c in range(nh * nk):
        mu, t = hac_t(X[:, c])
        row = {**cell_label(spec, c), "pooled_mean_R": mu, "pooled_hac_t": t,
               "sharpe_annual": sh[c] * np.sqrt(252), "smoothed_sharpe_annual": sm[c] * np.sqrt(252),
               "pooled_gross_mean_R": G[:, c].mean()}
        for a, tab in tabs.items():
            row[f"{a}_mean_R"] = tab["net"][:, c].mean()
            row[f"{a}_trades"] = int((tab["direction"][:, c] != 0).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "is_cell_table.csv", index=False)

    ev = evaluate(tabs, clusters, chosen)
    p_perm = permutation_p(G, K, nh, nk, float(sm.max()))
    cscv = cscv_pbo(X, nh, nk)

    pre_universe = {"NQ": spec["pre_sample_holdout_universe"]["NQ"], "MGC": spec["universe"]["MGC"]}
    pre_tabs, pre_clusters = build(spec, pre_universe, spec["periods"]["PRE"])
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
    result = {
        "stage": "IS", "period": spec["periods"]["IS"], "pooled_days": int(len(dates)),
        "selected_cell": cell_label(spec, chosen),
        "selected_cell_IS": ev,
        "max_stat_permutation_p": p_perm,
        "cscv": cscv,
        "selected_cell_PRE": pre,
        "clusters_positive": cl_pos, "instruments_positive": as_pos,
        "gates": gates, "gate_pass": bool(all(gates.values())),
    }
    (OUT / "is_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def stage_oos(spec: dict) -> None:
    is_res = json.loads((OUT / "is_results.json").read_text())
    if not is_res["gate_pass"]:
        raise SystemExit("IS gate failed: OOS is not computed.")
    c = is_res["selected_cell"]["index"]
    tabs, clusters = build(spec, spec["universe"], spec["periods"]["OOS"])
    ev = evaluate(tabs, clusters, c)
    ext_universe = {a: spec["universe"][a] for a in ("MNQ", "MGC", "MBT")}
    ext_tabs, ext_clusters = build(spec, ext_universe, spec["periods"]["EXT_informational"])
    ext = evaluate(ext_tabs, ext_clusters, c)
    cl_pos = sum(v["mean_R"] > 0 for v in ev["by_cluster"].values())
    verdict = bool(ev["pooled"]["mean_R"] > 0 and ev["pooled"]["hac_t"] >= 1.0 and cl_pos >= 2)
    result = {"stage": "OOS", "selected_cell": is_res["selected_cell"], "OOS": ev,
              "clusters_positive": cl_pos, "verdict_pass": verdict, "EXT_informational": ext}
    (OUT / "oos_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


# ------------------------------------------------------------------ self-test
def _synthetic(beta: float, seed: int, days: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx, opens, closes = [], [], []
    px = 100.0
    for d in pd.bdate_range("2020-01-01", periods=days):
        r = rng.normal(0, 0.0005, 390)
        r[120:] += beta * r[:120].sum() / 270
        path = px * np.exp(np.cumsum(r))
        o = np.concatenate([[px], path[:-1]])
        ts = pd.date_range(d + pd.Timedelta("9h30m"), periods=390, freq="min", tz="America/New_York")
        idx.append(ts); opens.append(o); closes.append(path)
        px = path[-1]
    return pd.DataFrame({"open": np.concatenate(opens), "close": np.concatenate(closes), "instrument_id": 1},
                        index=idx[0].append(idx[1:]))


def selftest() -> None:
    cfg = {"session_start": "09:30", "session_minutes": 390, "point_value": 1000.0, "tick_size": 0.0, "commission_per_side": 0.0}
    gh, gk = [30, 60, 90, 120, 180], [0.0, 0.5, 1.0]
    c120 = 3 * 3
    bars = _synthetic(0.0, 1)
    tab = session_table(bars, cfg, gh, gk)
    # exact reconstruction of one trade
    day = tab["dates"][10]
    b = bars[bars.index.tz_localize(None).normalize() == day]
    s = np.log(b["close"].iloc[119] / b["open"].iloc[0])
    manual = np.sign(s) * (b["close"].iloc[-1] - b["open"].iloc[120]) / b["open"].iloc[120]
    hist = bars.groupby(bars.index.tz_localize(None).normalize()).agg(o=("open", "first"), c=("close", "last"))
    pos = hist.index.get_loc(day)
    sig_day = np.sqrt((np.log(hist["c"] / hist["o"]).iloc[pos - 20:pos] ** 2).mean())
    assert abs(tab["gross"][10, c120] - manual / sig_day) < 1e-12, "trade reconstruction"
    assert (tab["direction"][:, c120] != 0).all() and (tab["direction"][:, c120 + 2] != 0).mean() < 0.5
    _, t0 = hac_t(tab["gross"][:, c120])
    tab1 = session_table(_synthetic(1.0, 1), cfg, gh, gk)
    _, t1 = hac_t(tab1["gross"][:, c120])
    assert abs(t0) < 2.5 and t1 > 4, (t0, t1)
    # HAC with zero lags equals the plain t statistic
    x = np.random.default_rng(0).normal(0.1, 1, 400)
    assert abs(hac_t(x, 0)[1] - x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))) < 1e-9
    # pooling: two instruments in one cluster count once
    tabs = {"A": tab, "B": tab, "C": tab1}
    _, X = pool(tabs, {"A": "x", "B": "x", "C": "y"}, "gross")
    assert np.allclose(X, (tab["gross"] + tab1["gross"]) / 2)
    # selection, permutation and CSCV: momentum world vs null world
    G1, K1 = tab1["gross"], tab1["cost"]
    ch, sm = select_cell(G1, 5, 3)
    p1 = permutation_p(G1[:, :], K1, 5, 3, float(sm.max()))
    p0 = permutation_p(tab["gross"], tab["cost"], 5, 3, float(select_cell(tab["gross"], 5, 3)[1].max()))
    pbo1 = cscv_pbo(G1, 5, 3)
    assert p1 < 0.01 and p0 > 0.05, (p1, p0)
    assert pbo1["share_test_sharpe_positive"] > 0.95
    print(f"selftest ok: null t={t0:.2f}, momentum t={t1:.2f}, p_null={p0:.3f}, p_mom={p1:.4f}, "
          f"chosen h={gh[ch // 3]}, pbo={pbo1['pbo']:.2f}")


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
