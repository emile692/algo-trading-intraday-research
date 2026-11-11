"""Night-session (Globex) long premium, cross-asset, phase 1.

Runs exactly FROZEN_SPECIFICATION.json (hash-checked against SPEC_LOCK.json).

    python run_phase1.py selftest   synthetic data only, no real return computed
    python run_phase1.py coverage   session counts only, no return (used before the freeze)
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
SEED = 20261111
N_PERM = 5000
CSCV_BLOCKS = 12
SHIFT_MIN = 360                 # wall clock + 6 h: the 18:00 ET reopen becomes minute 0 of the trade date
RTH_OPEN, RTH_CLOSE = 930, 1320  # 09:30 and 16:00 ET on the shifted clock
MAX_PRIOR_GAP_DAYS = 4


# ------------------------------------------------------------------ spec / io
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_spec(check_lock: bool = True) -> dict:
    spec_path = HERE / "FROZEN_SPECIFICATION.json"
    if check_lock:
        lock = json.loads((HERE / "SPEC_LOCK.json").read_text())
        if sha256(spec_path) != lock["spec_sha256"]:
            raise RuntimeError("Specification changed after freeze")
    spec = json.loads(spec_path.read_text())
    for item in list(spec["universe"].values()) + list(spec["pre_sample_holdout_universe"].values()):
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Input checksum mismatch: {item['path']}")
    return spec


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["open", "close", "instrument_id"])
    if df.index.tz is None:
        raise RuntimeError("Naive timestamps")
    df.index = df.index.tz_convert("America/New_York")
    return df.sort_index()


def node_minutes(spec: dict) -> list[int]:
    out = []
    for hhmm in spec["session"]["nodes_et"]:
        hh, mm = map(int, hhmm.split(":"))
        out.append((hh * 60 + mm + SHIFT_MIN) % 1440)
    if out != sorted(out):
        raise RuntimeError("Nodes must be ordered inside the night session")
    return out


def cells(spec: dict) -> list[dict]:
    nodes = spec["session"]["nodes_et"]
    out = []
    for a, b in itertools.combinations(range(len(nodes)), 2):
        for cond in spec["grid"]["conditions"]:
            out.append({"index": len(out), "entry": nodes[a], "exit": nodes[b], "a": a, "b": b, "condition": cond})
    return out


# ------------------------------------------------------------ session features
def _first_valid(block: np.ndarray) -> np.ndarray:
    out = np.full(block.shape[0], np.nan)
    for j in range(block.shape[1] - 1, -1, -1):
        col = block[:, j]
        out = np.where(np.isnan(col), out, col)
    return out


def _last_valid(block: np.ndarray) -> np.ndarray:
    return _first_valid(block[:, ::-1])


def _wide(d: pd.DataFrame, col: str, lo: int, hi: int, dates: pd.DatetimeIndex) -> np.ndarray:
    sub = d[(d["m"] >= lo) & (d["m"] < hi)]
    return (sub.pivot(index="date", columns="m", values=col)
            .reindex(index=dates, columns=range(lo, hi)).to_numpy(float))


def session_table(bars: pd.DataFrame, cfg: dict, spec: dict, tol: int | None = None) -> dict:
    """Per-night arrays for one instrument. Nothing here depends on the period split."""
    tol = int(spec["session"]["node_tolerance_minutes"]) if tol is None else tol
    nodes = node_minutes(spec)
    grid = cells(spec)
    w = bars.index.tz_localize(None) + pd.Timedelta(minutes=SHIFT_MIN)
    d = pd.DataFrame({"open": bars["open"].to_numpy(), "close": bars["close"].to_numpy(),
                      "instrument_id": bars["instrument_id"].to_numpy(),
                      "m": w.hour * 60 + w.minute, "date": w.normalize()})
    d = d[d["date"].dt.dayofweek < 5]
    dates = pd.DatetimeIndex(sorted(d["date"].unique()))

    night = d[d["m"] < nodes[-1] + tol + 1]
    n_inst = night.groupby("date")["instrument_id"].nunique().reindex(dates).to_numpy()
    px = np.column_stack([_first_valid(_wide(d, "open", n, n + tol, dates)) for n in nodes])
    px_delay = np.column_stack([_first_valid(_wide(d, "open", n + 1, n + 1 + tol, dates)) for n in nodes])
    with np.errstate(invalid="ignore"):
        eligible = ~np.isnan(px).any(axis=1) & (px > 0).all(axis=1) & (n_inst == 1)

    rth_open = _first_valid(_wide(d, "open", RTH_OPEN, RTH_OPEN + tol, dates))
    rth_close = _last_valid(_wide(d, "close", RTH_CLOSE - tol, RTH_CLOSE, dates))
    with np.errstate(invalid="ignore", divide="ignore"):
        rth = np.where((rth_open > 0) & (rth_close > 0), rth_close / rth_open - 1.0, np.nan)
    # prior RTH return: last weekday with an RTH return strictly before the night's trade date
    has = ~np.isnan(rth)
    last = np.where(has, np.arange(len(dates)), -1)
    last = np.maximum.accumulate(last)
    prev = np.concatenate([[-1], last[:-1]])
    gap = (dates - dates[np.maximum(prev, 0)]).days.to_numpy()
    prior_rth = np.where((prev >= 0) & (gap <= MAX_PRIOR_GAP_DAYS), rth[np.maximum(prev, 0)], np.nan)

    dates, px, px_delay = dates[eligible], px[eligible], px_delay[eligible]
    rth, prior_rth = rth[eligible], prior_rth[eligible]
    full = px[:, -1] / px[:, 0] - 1.0
    sigma = np.sqrt(pd.Series(full ** 2).rolling(SIGMA_WINDOW, min_periods=SIGMA_MIN).mean().shift(1).to_numpy())
    cost_usd = 2 * cfg["tick_size"] * cfg["point_value"] + 2 * cfg["commission_per_side"]

    T, n_cells = len(dates), len(grid)
    direction = np.zeros((T, n_cells))
    gross = np.zeros((T, n_cells))
    gross_delay = np.zeros((T, n_cells))
    cost = np.zeros((T, n_cells))
    usd_gross = np.zeros((T, n_cells))
    with np.errstate(invalid="ignore"):
        sig_ok = ~np.isnan(sigma) & (sigma > 0)
        down = sig_ok & (prior_rth < 0)
    sd = np.where(sig_ok, sigma, 1.0)
    for cell in grid:
        c, a, b = cell["index"], cell["a"], cell["b"]
        ok = sig_ok if cell["condition"] == "all" else down
        e, x = px[:, a], px[:, b]
        ed = np.where(np.isnan(px_delay[:, a]), e, px_delay[:, a])
        direction[:, c] = ok
        gross[:, c] = np.where(ok, (x - e) / e / sd, 0.0)
        gross_delay[:, c] = np.where(ok, (x - ed) / ed / sd, 0.0)
        cost[:, c] = np.where(ok, cost_usd / (cfg["point_value"] * e) / sd, 0.0)
        usd_gross[:, c] = np.where(ok, (x - e) * cfg["point_value"], 0.0)
    rth_R = np.where(sig_ok & ~np.isnan(rth), rth / sd, 0.0)[:, None]
    warm = sig_ok
    return {
        "dates": dates[warm], "direction": direction[warm], "gross": gross[warm], "cost": cost[warm],
        "gross_delay": gross_delay[warm], "usd_gross": usd_gross[warm], "rth_long": rth_R[warm],
        "cost_usd": cost_usd,
    }


def restrict(tab: dict, start: str, end: str) -> dict:
    mask = (tab["dates"] >= pd.Timestamp(start)) & (tab["dates"] <= pd.Timestamp(end))
    return {k: (v[mask] if isinstance(v, (np.ndarray, pd.DatetimeIndex)) else v) for k, v in tab.items()}


# -------------------------------------------------------------------- pooling
def pool(tabs: dict[str, dict], clusters: dict[str, str], key: str):
    """Cluster-balanced daily series. Returns (dates, matrix T x cells)."""
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(t["dates"]) for t in tabs.values()])))
    n_cells = next(iter(tabs.values()))[key].shape[1]
    total = np.zeros((len(all_dates), n_cells))
    active = np.zeros(len(all_dates))
    for cl in sorted(set(clusters.values())):
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


def select_cell(X: np.ndarray) -> tuple[int, np.ndarray]:
    sh = daily_sharpe(X)
    return int(np.argmax(sh)), sh   # first maximum: grid order breaks ties


def permutation_p(G: np.ndarray, K: np.ndarray, observed: float) -> float:
    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(N_PERM):
        flip = rng.choice([-1.0, 1.0], size=G.shape[0])[:, None]
        hits += daily_sharpe(flip * G - K).max() >= observed
    return (hits + 1) / (N_PERM + 1)


def cscv_pbo(X: np.ndarray) -> dict:
    blocks = np.array_split(np.arange(X.shape[0]), CSCV_BLOCKS)
    n = X.shape[1]
    logits, test_sh = [], []
    for combo in itertools.combinations(range(CSCV_BLOCKS), CSCV_BLOCKS // 2):
        tr = np.concatenate([blocks[b] for b in combo])
        te = np.concatenate([blocks[b] for b in range(CSCV_BLOCKS) if b not in combo])
        chosen, _ = select_cell(X[tr])
        sh_te = daily_sharpe(X[te])
        w = ((sh_te < sh_te[chosen]).sum() + 1) / (n + 1)
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
    return {"days": int(len(x)), "mean_R": mu, "hac_t": t,
            "sharpe_annual": float(mu / sd * np.sqrt(252)) if sd > 0 else float("nan")}


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
        "worst_trade_usd": float(usd[traded].min()) if traded.any() else float("nan"),
        "rth_long_mean_R": float(tab["rth_long"][:, 0].mean()),
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
        _, Xc = pool(sub, {a: cl for a in sub}, "net")
        by_cluster[cl] = series_stats(Xc[:, c])
    return {
        "pooled": series_stats(x),
        "pooled_cost_x2": series_stats(pool(tabs, clusters, "net_x2")[1][:, c]),
        "pooled_entry_delay": series_stats(pool(tabs, clusters, "net_delay")[1][:, c]),
        "pooled_rth_long_benchmark": series_stats(pool(tabs, clusters, "rth_long")[1][:, 0]),
        "yearly_sum_R": {int(y): float(v) for y, v in yearly.items()},
        "by_cluster": by_cluster,
        "by_instrument": {a: asset_stats(t, c) for a, t in tabs.items()},
    }


# --------------------------------------------------------------------- stages
def build(spec: dict, universe: dict, period_key: str) -> tuple[dict, dict]:
    tabs, clusters = {}, {}
    for name, cfg in universe.items():
        period = cfg.get("periods", {}).get(period_key, spec["periods"][period_key])
        tabs[name] = restrict(session_table(load_bars(ROOT / cfg["path"]), cfg, spec), *period)
        clusters[name] = cfg["cluster"]
    return tabs, clusters


def label(cell: dict) -> dict:
    return {k: cell[k] for k in ("index", "entry", "exit", "condition")}


def stage_coverage(spec: dict) -> None:
    grid = cells(spec)
    c_all = next(c["index"] for c in grid if c["condition"] == "all")
    c_down = next(c["index"] for c in grid if c["condition"] == "after_down_rth")
    items = {**spec["universe"], **{f"PRE_{k}": v for k, v in spec["pre_sample_holdout_universe"].items()}}
    for name, cfg in items.items():
        bars = load_bars(ROOT / cfg["path"])
        for tol in (5, 15, 30):
            tab = session_table(bars, cfg, spec, tol)
            yr = pd.Series(tab["direction"][:, c_all], index=tab["dates"]).groupby(tab["dates"].year).sum().astype(int)
            dn = pd.Series(tab["direction"][:, c_down], index=tab["dates"]).groupby(tab["dates"].year).sum().astype(int)
            print(f"{name:8s} tol={tol:2d} nights/year {dict(yr)}")
            print(f"{'':8s}        after_down  {dict(dn)}")


def stage_is(spec: dict) -> None:
    OUT.mkdir(exist_ok=True)
    grid = cells(spec)
    tabs, clusters = build(spec, spec["universe"], "IS")
    for t in tabs.values():
        t["net"] = t["gross"] - t["cost"]
    dates, X = pool(tabs, clusters, "net")
    _, G = pool(tabs, clusters, "gross")
    _, K = pool(tabs, clusters, "cost")
    chosen, sh = select_cell(X)

    rows = []
    for cell in grid:
        c = cell["index"]
        mu, t = hac_t(X[:, c])
        row = {**label(cell), "pooled_mean_R": mu, "pooled_hac_t": t, "sharpe_annual": sh[c] * np.sqrt(252),
               "pooled_gross_mean_R": G[:, c].mean()}
        for a, tab in tabs.items():
            row[f"{a}_mean_R"] = tab["net"][:, c].mean()
            row[f"{a}_trades"] = int((tab["direction"][:, c] != 0).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "is_cell_table.csv", index=False)

    ev = evaluate(tabs, clusters, chosen)
    p_perm = permutation_p(G, K, float(sh.max()))
    cscv = cscv_pbo(X)
    pre_tabs, pre_clusters = build(spec, spec["pre_sample_holdout_universe"], "PRE")
    pre = evaluate(pre_tabs, pre_clusters, chosen)

    cl_pos = sum(v["mean_R"] > 0 for v in ev["by_cluster"].values())
    as_pos = sum(v["mean_R"] > 0 for v in ev["by_instrument"].values())
    years = ev["yearly_sum_R"]
    gates = {
        "G1_hac": bool(ev["pooled"]["mean_R"] > 0 and ev["pooled"]["hac_t"] >= 2.0),
        "G2_permutation": bool(p_perm <= 0.05),
        "G3_pbo": bool(cscv["pbo"] <= 0.35),
        "G4_breadth": bool(cl_pos >= 3 and as_pos >= 4),
        "G5_years": bool(sum(v > 0 for v in years.values()) > len(years) / 2),
        "G6_pre_holdout": bool(pre["pooled"]["mean_R"] > 0 and pre["pooled"]["hac_t"] >= 1.0),
        "G7_not_plain_beta": bool(ev["pooled"]["mean_R"] > ev["pooled_rth_long_benchmark"]["mean_R"]
                                  and pre["pooled"]["mean_R"] > pre["pooled_rth_long_benchmark"]["mean_R"]),
    }
    result = {
        "stage": "IS", "period": spec["periods"]["IS"], "pooled_days": int(len(dates)),
        "selected_cell": label(grid[chosen]), "selected_cell_IS": ev,
        "max_stat_permutation_p": p_perm, "cscv": cscv, "selected_cell_PRE": pre,
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
    tabs, clusters = build(spec, spec["universe"], "OOS")
    ev = evaluate(tabs, clusters, c)
    ext_universe = {a: spec["universe"][a] for a in spec["oos_verdict"]["EXT_instruments"]}
    ext_tabs, ext_clusters = build(spec, ext_universe, "EXT_informational")
    ext = evaluate(ext_tabs, ext_clusters, c)
    cl_pos = sum(v["mean_R"] > 0 for v in ev["by_cluster"].values())
    verdict = bool(ev["pooled"]["mean_R"] > 0 and ev["pooled"]["hac_t"] >= 1.0 and cl_pos >= 3)
    result = {"stage": "OOS", "selected_cell": is_res["selected_cell"], "OOS": ev,
              "clusters_positive": cl_pos, "verdict_pass": verdict, "EXT_informational": ext}
    (OUT / "oos_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


# ------------------------------------------------------------------ self-test
def _synthetic(drift: float, seed: int, days: int = 400) -> pd.DataFrame:
    """22 h of one-minute bars per weekday: 18:00 the evening before to 15:59. Drift sits in 02:00-03:00."""
    rng = np.random.default_rng(seed)
    idx, opens, closes = [], [], []
    px = 100.0
    for day in pd.bdate_range("2020-01-06", periods=days):
        r = rng.normal(0, 0.0002, 1320)
        r[480:540] += drift / 60
        path = px * np.exp(np.cumsum(r))
        o = np.concatenate([[px], path[:-1]])
        idx.append(pd.date_range(day - pd.Timedelta("6h"), periods=1320, freq="min", tz="America/New_York"))
        opens.append(o); closes.append(path)
        px = path[-1]
    return pd.DataFrame({"open": np.concatenate(opens), "close": np.concatenate(closes), "instrument_id": 1},
                        index=idx[0].append(idx[1:]))


def selftest() -> None:
    spec = {"session": {"nodes_et": ["18:00", "02:00", "03:00", "08:00"], "node_tolerance_minutes": 5},
            "grid": {"conditions": ["all", "after_down_rth"]}}
    cfg = {"point_value": 1000.0, "tick_size": 0.0, "commission_per_side": 0.0}
    grid = cells(spec)
    assert len(grid) == 12
    c_eu = next(c["index"] for c in grid if (c["entry"], c["exit"], c["condition"]) == ("02:00", "03:00", "all"))
    c_eu_dn = c_eu + 1
    bars = _synthetic(0.0, 1)
    tab = session_table(bars, cfg, spec)
    # exact reconstruction of one trade and of its risk unit
    day = tab["dates"][40]
    naive = bars.index.tz_localize(None)
    e = bars["open"][naive == day + pd.Timedelta("2h")].iloc[0]
    x = bars["open"][naive == day + pd.Timedelta("3h")].iloc[0]
    all_days = pd.bdate_range("2020-01-06", periods=400)
    pos = all_days.get_loc(day)
    fulls = []
    for dd in all_days[pos - 20:pos]:
        a = bars["open"][naive == dd - pd.Timedelta("6h")].iloc[0]
        b = bars["open"][naive == dd + pd.Timedelta("8h")].iloc[0]
        fulls.append(b / a - 1)
    sig = np.sqrt(np.mean(np.square(fulls)))
    assert abs(tab["gross"][40, c_eu] - (x - e) / e / sig) < 1e-12, "trade reconstruction"
    # the conditional cell trades only after a down RTH session, and that return is known at 18:00
    prev = all_days[pos - 1]
    ro = bars["open"][naive == prev + pd.Timedelta("9h30m")].iloc[0]
    rc = bars["close"][naive == prev + pd.Timedelta("15h59m")].iloc[0]
    assert bool(tab["direction"][40, c_eu_dn]) == (rc / ro - 1 < 0), "conditional cell"
    share = tab["direction"][:, c_eu_dn].mean()
    assert 0.35 < share < 0.65 and tab["direction"][:, c_eu].all()
    # null world vs drift world
    _, t0 = hac_t(tab["gross"][:, c_eu])
    tab1 = session_table(_synthetic(0.0015, 1), cfg, spec)
    _, t1 = hac_t(tab1["gross"][:, c_eu])
    assert abs(t0) < 2.5 and t1 > 4, (t0, t1)
    x0 = np.random.default_rng(0).normal(0.1, 1, 400)
    assert abs(hac_t(x0, 0)[1] - x0.mean() / (x0.std(ddof=0) / np.sqrt(len(x0)))) < 1e-9
    # pooling: two instruments in one cluster count once
    _, X = pool({"A": tab, "B": tab, "C": tab1}, {"A": "x", "B": "x", "C": "y"}, "gross")
    assert np.allclose(X, (tab["gross"] + tab1["gross"]) / 2)
    # selection, permutation, CSCV
    ch, sh = select_cell(tab1["gross"])
    p1 = permutation_p(tab1["gross"], tab1["cost"], float(sh.max()))
    p0 = permutation_p(tab["gross"], tab["cost"], float(select_cell(tab["gross"])[1].max()))
    pbo1 = cscv_pbo(tab1["gross"])
    assert p1 < 0.01 and p0 > 0.05, (p1, p0)
    assert grid[ch]["entry"] == "02:00" and grid[ch]["exit"] == "03:00", grid[ch]
    assert pbo1["share_test_sharpe_positive"] > 0.95
    # a sparse tape: dropping the node bar moves the fill to the next available bar inside the tolerance
    drop = (naive.hour == 2) & (naive.minute == 0)
    tab2 = session_table(bars[~drop], cfg, spec)
    e2 = bars["open"][naive == day + pd.Timedelta("2h1m")].iloc[0]
    assert abs(tab2["usd_gross"][40, c_eu] - (x - e2) * 1000.0) < 1e-9
    print(f"selftest ok: null t={t0:.2f}, drift t={t1:.2f}, p_null={p0:.3f}, p_drift={p1:.4f}, "
          f"chosen {grid[ch]['entry']}->{grid[ch]['exit']} {grid[ch]['condition']}, pbo={pbo1['pbo']:.2f}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "selftest":
        selftest()
    elif stage == "coverage":
        stage_coverage(read_spec(check_lock=False))
    elif stage == "is":
        stage_is(read_spec())
    elif stage == "oos":
        stage_oos(read_spec())
    else:
        raise SystemExit(__doc__)
