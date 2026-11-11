"""Intraday relative value between US equity index futures (MNQ, MES, M2K), phase 1.

Runs exactly FROZEN_SPECIFICATION.json (hash-checked against SPEC_LOCK.json).

    python run_phase1.py selftest    synthetic data only, no real return computed
    python run_phase1.py costcheck   session counts, spread volatility and cost ratio only (used before the freeze)
    python run_phase1.py is          IS selection, tests, gate
    python run_phase1.py oos         verdict; refused unless the IS gate passed
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
MODES = ["reversion", "continuation"]


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
    for item in spec["instruments"].values():
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


def leg_table(bars: pd.DataFrame, session: dict, grid_h: list[int]) -> pd.DataFrame:
    """Per-session prices of one instrument: open, exit, and for each h the signal, entry and delayed entry."""
    L = int(session["session_minutes"])
    hh, mm = map(int, session["session_start"].split(":"))
    idx = bars.index
    m = idx.hour * 60 + idx.minute - (hh * 60 + mm)
    keep = (m >= 0) & (m < L) & (idx.dayofweek < 5)
    d = bars.loc[keep, ["open", "close", "instrument_id"]].copy()
    d["m"] = np.asarray(m[keep])
    d["date"] = d.index.tz_localize(None).normalize()
    Cdf = d.pivot(index="date", columns="m", values="close").reindex(columns=range(L))
    dates = Cdf.index
    C = Cdf.to_numpy(float)
    O = d.pivot(index="date", columns="m", values="open").reindex(index=dates, columns=range(L)).to_numpy(float)
    grp = d.groupby("date")
    count = grp["close"].size().reindex(dates).to_numpy()
    n_inst = grp["instrument_id"].nunique().reindex(dates).to_numpy()
    out = {"open": _first_valid(O[:, :STALE]), "exit": _last_valid(C[:, L - STALE:])}
    for h in grid_h:
        out[f"sig_{h}"] = _last_valid(C[:, h - STALE:h])
        out[f"ent_{h}"] = _first_valid(O[:, h:h + STALE])
        out[f"entd_{h}"] = _first_valid(O[:, h + 1:h + 1 + STALE])
    t = pd.DataFrame(out, index=dates)
    eligible = (count >= 0.75 * L) & t["open"].notna().to_numpy() & t["exit"].notna().to_numpy() & (n_inst == 1)
    return t[eligible]


def rms_prev(x: np.ndarray) -> np.ndarray:
    return np.sqrt(pd.Series(x ** 2).rolling(SIGMA_WINDOW, min_periods=SIGMA_MIN).mean().shift(1).to_numpy())


def pair_table(la: pd.DataFrame, lb: pd.DataFrame, ca: dict, cb: dict, grid_h: list[int], grid_k: list[float]) -> dict:
    """Spread A minus B, equal notional on both legs. Cell index = mode * nh * nk + i_h * nk + j_k."""
    dates = la.index.intersection(lb.index)
    a, b = la.loc[dates], lb.loc[dates]
    day = np.log(a["exit"] / a["open"]).to_numpy() - np.log(b["exit"] / b["open"]).to_numpy()
    sigma_day = rms_prev(day)
    rt_a = 2 * ca["tick_size"] * ca["point_value"] + 2 * ca["commission_per_side"]
    rt_b = 2 * cb["tick_size"] * cb["point_value"] + 2 * cb["commission_per_side"]
    nh, nk = len(grid_h), len(grid_k)
    T, n_cells = len(dates), 2 * nh * nk
    direction = np.zeros((T, n_cells))
    gross = np.zeros((T, n_cells))
    gross_delay = np.zeros((T, n_cells))
    cost = np.zeros((T, n_cells))
    gross_bp = np.zeros((T, n_cells))
    cost_bp = np.zeros((T, n_cells))
    xa, xb = a["exit"].to_numpy(), b["exit"].to_numpy()
    for i, h in enumerate(grid_h):
        s = (np.log(a[f"sig_{h}"] / a["open"]) - np.log(b[f"sig_{h}"] / b["open"])).to_numpy()
        sigma_h = rms_prev(s)
        ea, eb = a[f"ent_{h}"].to_numpy(), b[f"ent_{h}"].to_numpy()
        da, db = a[f"entd_{h}"].to_numpy(), b[f"entd_{h}"].to_numpy()
        for j, k in enumerate(grid_k):
            ok = ~np.isnan(s) & ~np.isnan(sigma_h) & ~np.isnan(ea) & ~np.isnan(eb) & ~np.isnan(sigma_day) & (sigma_day > 0)
            with np.errstate(invalid="ignore"):
                ok &= (np.abs(s) >= k * sigma_h) & (s != 0)
            sd = np.where(ok, sigma_day, 1.0)
            pa, pb = np.where(ok, ea, 1.0), np.where(ok, eb, 1.0)
            qa = np.where(ok & ~np.isnan(da), da, pa)
            qb = np.where(ok & ~np.isnan(db), db, pb)
            spread = np.where(ok, (xa - pa) / pa - (xb - pb) / pb, 0.0)
            spread_d = np.where(ok, (xa - qa) / qa - (xb - qb) / qb, 0.0)
            c_ret = np.where(ok, rt_a / (ca["point_value"] * pa) + rt_b / (cb["point_value"] * pb), 0.0)
            for mi, sgn in enumerate((-1.0, 1.0)):       # reversion fades the spread move, continuation follows it
                c = mi * nh * nk + i * nk + j
                dirn = np.where(ok, sgn * np.sign(s), 0.0)
                direction[:, c] = dirn
                gross[:, c] = dirn * spread / sd
                gross_delay[:, c] = dirn * spread_d / sd
                cost[:, c] = c_ret / sd
                gross_bp[:, c] = dirn * spread * 1e4
                cost_bp[:, c] = c_ret * 1e4
    warm = ~np.isnan(sigma_day)
    return {"dates": dates[warm], "direction": direction[warm], "gross": gross[warm], "cost": cost[warm],
            "gross_delay": gross_delay[warm], "gross_bp": gross_bp[warm], "cost_bp": cost_bp[warm],
            "sigma_day": sigma_day[warm]}


def restrict(tab: dict, start: str, end: str) -> dict:
    mask = (tab["dates"] >= pd.Timestamp(start)) & (tab["dates"] <= pd.Timestamp(end))
    return {k: v[mask] for k, v in tab.items()}


def pool(tabs: dict[str, dict], key: str):
    """Equal-weight average of the pairs present on each date (the three pairs are one bet)."""
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(t["dates"]) for t in tabs.values()])))
    n_cells = next(iter(tabs.values()))[key].shape[1]
    s = np.zeros((len(all_dates), n_cells))
    n = np.zeros(len(all_dates))
    for t in tabs.values():
        pos = all_dates.get_indexer(t["dates"])
        s[pos] += t[key]
        n[pos] += 1
    return all_dates, s / n[:, None]


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
    """Average of a cell and its orthogonal (h, k) neighbours, inside its own mode."""
    g = sh.reshape(2, nh, nk)
    out = np.zeros_like(g)
    for mi in range(2):
        for i in range(nh):
            for j in range(nk):
                vals = [g[mi, i, j]]
                if i > 0: vals.append(g[mi, i - 1, j])
                if i < nh - 1: vals.append(g[mi, i + 1, j])
                if j > 0: vals.append(g[mi, i, j - 1])
                if j < nk - 1: vals.append(g[mi, i, j + 1])
                out[mi, i, j] = np.mean(vals)
    return out.ravel()


def select_cell(X: np.ndarray, nh: int, nk: int) -> tuple[int, np.ndarray]:
    sm = smooth(daily_sharpe(X), nh, nk)
    return int(np.argmax(sm)), sm


def permutation_p(G: np.ndarray, K: np.ndarray, nh: int, nk: int, observed: float) -> float:
    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(N_PERM):
        flip = rng.choice([-1.0, 1.0], size=G.shape[0])[:, None]
        hits += smooth(daily_sharpe(flip * G - K), nh, nk).max() >= observed
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
        w = ((sh_te < sh_te[chosen]).sum() + 1) / (n + 1)
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


def pair_stats(tab: dict, c: int) -> dict:
    net = tab["gross"][:, c] - tab["cost"][:, c]
    traded = tab["direction"][:, c] != 0
    out = series_stats(net)
    net_bp = tab["gross_bp"][traded, c] - tab["cost_bp"][traded, c]
    out.update({
        "trades": int(traded.sum()),
        "mean_gross_bp_per_trade": float(tab["gross_bp"][traded, c].mean()) if traded.any() else float("nan"),
        "mean_cost_bp_per_trade": float(tab["cost_bp"][traded, c].mean()) if traded.any() else float("nan"),
        "mean_net_R_per_trade": float(net[traded].mean()) if traded.any() else float("nan"),
        "win_rate": float((net_bp > 0).mean()) if traded.any() else float("nan"),
    })
    return out


def evaluate(tabs: dict[str, dict], c: int) -> dict:
    for t in tabs.values():
        t["net"] = t["gross"] - t["cost"]
        t["net_x2"] = t["gross"] - 2 * t["cost"]
        t["net_delay"] = t["gross_delay"] - t["cost"]
    dates, X = pool(tabs, "net")
    x = X[:, c]
    yearly = pd.Series(x, index=dates).groupby(dates.year).sum()
    half = len(x) // 2
    return {
        "pooled": series_stats(x),
        "pooled_first_half": series_stats(x[:half]),
        "pooled_second_half": series_stats(x[half:]),
        "pooled_gross": series_stats(pool(tabs, "gross")[1][:, c]),
        "pooled_cost_x2": series_stats(pool(tabs, "net_x2")[1][:, c]),
        "pooled_entry_delay": series_stats(pool(tabs, "net_delay")[1][:, c]),
        "yearly_sum_R": {int(y): float(v) for y, v in yearly.items()},
        "by_pair": {p: pair_stats(t, c) for p, t in tabs.items()},
    }


# --------------------------------------------------------------------- stages
def build(spec: dict, period: list[str] | None) -> dict:
    gh, gk = spec["grid"]["h_minutes"], spec["grid"]["k_sigma"]
    legs = {n: leg_table(load_bars(ROOT / cfg["path"]), spec["session"], gh) for n, cfg in spec["instruments"].items()}
    tabs = {}
    for a, b in spec["pairs"]:
        tab = pair_table(legs[a], legs[b], spec["instruments"][a], spec["instruments"][b], gh, gk)
        tabs[f"{a}-{b}"] = restrict(tab, *period) if period else tab
    return tabs


def cell_label(spec: dict, c: int) -> dict:
    gh, gk = spec["grid"]["h_minutes"], spec["grid"]["k_sigma"]
    per_mode = len(gh) * len(gk)
    r = c % per_mode
    return {"index": c, "mode": MODES[c // per_mode], "h_minutes": gh[r // len(gk)], "k_sigma": gk[r % len(gk)]}


def stage_costcheck(spec: dict) -> None:
    """No strategy return: sessions per year, daily spread volatility, round-trip cost as a share of it."""
    gh, gk = spec["grid"]["h_minutes"], spec["grid"]["k_sigma"]
    tabs = build(spec, None)
    for p, t in tabs.items():
        yrs = pd.Series(1, index=t["dates"]).groupby(t["dates"].year).sum()
        print(p, "sessions/year", dict(yrs.astype(int)))
        c0 = 0  # h = first, k = first
        traded = t["direction"][:, c0] != 0
        ratio = t["cost"][traded, c0]
        by_year = pd.Series(ratio, index=t["dates"][traded]).groupby(t["dates"][traded].year).median()
        print(f"   sigma_day median {np.median(t['sigma_day']) * 1e4:.1f} bp, round-trip cost median "
              f"{np.median(t['cost_bp'][traded, c0]):.2f} bp, cost / sigma_day median {np.median(ratio):.3f} R")
        print("   cost / sigma_day by year", {int(y): round(float(v), 3) for y, v in by_year.items()})
        for j, k in enumerate(gk):
            share = [(t["direction"][:, i * len(gk) + j] != 0).mean() for i in range(len(gh))]
            print(f"   k={k}: share of sessions traded by h {[round(float(s), 2) for s in share]}")


def stage_is(spec: dict) -> None:
    OUT.mkdir(exist_ok=True)
    gh, gk = spec["grid"]["h_minutes"], spec["grid"]["k_sigma"]
    nh, nk = len(gh), len(gk)
    tabs = build(spec, spec["periods"]["IS"])
    for t in tabs.values():
        t["net"] = t["gross"] - t["cost"]
    dates, X = pool(tabs, "net")
    _, G = pool(tabs, "gross")
    _, K = pool(tabs, "cost")
    chosen, sm = select_cell(X, nh, nk)
    sh = daily_sharpe(X)
    rows = []
    for c in range(2 * nh * nk):
        mu, t = hac_t(X[:, c])
        row = {**cell_label(spec, c), "pooled_mean_R": mu, "pooled_hac_t": t, "sharpe_annual": sh[c] * np.sqrt(252),
               "smoothed_sharpe_annual": sm[c] * np.sqrt(252), "pooled_gross_mean_R": G[:, c].mean(),
               "pooled_cost_mean_R": K[:, c].mean()}
        for p, tab in tabs.items():
            row[f"{p}_mean_R"] = tab["net"][:, c].mean()
            row[f"{p}_trades"] = int((tab["direction"][:, c] != 0).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "is_cell_table.csv", index=False)

    ev = evaluate(tabs, chosen)
    p_perm = permutation_p(G, K, nh, nk, float(sm.max()))
    cscv = cscv_pbo(X, nh, nk)
    pairs_pos = sum(v["mean_R"] > 0 for v in ev["by_pair"].values())
    years = ev["yearly_sum_R"]
    gates = {
        "G1_hac": bool(ev["pooled"]["mean_R"] > 0 and ev["pooled"]["hac_t"] >= 2.0),
        "G2_permutation": bool(p_perm <= 0.05),
        "G3_pbo": bool(cscv["pbo"] <= 0.35),
        "G4_breadth": bool(pairs_pos >= 2),
        "G5_years": bool(sum(v > 0 for v in years.values()) > len(years) / 2),
        "G6_both_halves": bool(ev["pooled_first_half"]["mean_R"] > 0 and ev["pooled_second_half"]["mean_R"] > 0),
        "G7_cost_x1_5": bool(ev["pooled"]["mean_R"] + ev["pooled_cost_x2"]["mean_R"] > 0),
    }
    result = {"stage": "IS", "period": spec["periods"]["IS"], "pooled_days": int(len(dates)),
              "selected_cell": cell_label(spec, chosen), "selected_cell_IS": ev,
              "max_stat_permutation_p": p_perm, "cscv": cscv, "pairs_positive": pairs_pos,
              "gates": gates, "gate_pass": bool(all(gates.values()))}
    (OUT / "is_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def stage_oos(spec: dict) -> None:
    is_res = json.loads((OUT / "is_results.json").read_text())
    if not is_res["gate_pass"]:
        raise SystemExit("IS gate failed: OOS is not computed.")
    c = is_res["selected_cell"]["index"]
    ev = evaluate(build(spec, spec["periods"]["OOS"]), c)
    pairs_pos = sum(v["mean_R"] > 0 for v in ev["by_pair"].values())
    verdict = bool(ev["pooled"]["mean_R"] > 0 and ev["pooled"]["hac_t"] >= 1.0 and pairs_pos >= 2)
    result = {"stage": "OOS", "selected_cell": is_res["selected_cell"], "OOS": ev,
              "pairs_positive": pairs_pos, "verdict_pass": verdict}
    (OUT / "oos_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


# ------------------------------------------------------------------ self-test
def _synthetic(beta: float, seed: int, days: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two instruments sharing a market factor. A's own move over the first 120 minutes reverts by beta afterwards."""
    rng = np.random.default_rng(seed)
    out = {"A": ([], [], []), "B": ([], [], [])}
    px = {"A": 100.0, "B": 50.0}
    for d in pd.bdate_range("2020-01-01", periods=days):
        mkt = rng.normal(0, 0.0005, 390)
        ts = pd.date_range(d + pd.Timedelta("9h30m"), periods=390, freq="min", tz="America/New_York")
        for name in ("A", "B"):
            idio = rng.normal(0, 0.0002, 390)
            if name == "A":
                idio[120:] -= beta * idio[:120].sum() / 270
            path = px[name] * np.exp(np.cumsum(mkt + idio))
            out[name][0].append(ts); out[name][1].append(np.concatenate([[px[name]], path[:-1]])); out[name][2].append(path)
            px[name] = path[-1]
    frames = []
    for name in ("A", "B"):
        idx = out[name][0][0].append(out[name][0][1:])
        frames.append(pd.DataFrame({"open": np.concatenate(out[name][1]), "close": np.concatenate(out[name][2]),
                                    "instrument_id": 1}, index=idx))
    return frames[0], frames[1]


def selftest() -> None:
    session = {"session_start": "09:30", "session_minutes": 390}
    cfg = {"point_value": 1000.0, "tick_size": 0.0, "commission_per_side": 0.0}
    gh, gk = [30, 60, 90, 120, 180], [0.5, 1.0, 1.5]
    nh, nk = 5, 3
    c_rev = 3 * nk            # reversion, h=120, k=0.5
    c_con = nh * nk + c_rev   # continuation, same (h, k)

    def tab_for(beta):
        A, B = _synthetic(beta, 1)
        return A, B, pair_table(leg_table(A, session, gh), leg_table(B, session, gh), cfg, cfg, gh, gk)

    A, B, tab = tab_for(0.0)
    assert np.allclose(tab["gross"][:, c_rev], -tab["gross"][:, c_con])
    # exact reconstruction of one trade and of its risk unit
    i = next(n for n in range(30, 200) if tab["direction"][n, c_rev] != 0)
    day = tab["dates"][i]
    a = A[A.index.tz_localize(None).normalize() == day]
    b = B[B.index.tz_localize(None).normalize() == day]
    s = np.log(a["close"].iloc[119] / a["open"].iloc[0]) - np.log(b["close"].iloc[119] / b["open"].iloc[0])
    ret = (a["close"].iloc[-1] / a["open"].iloc[120] - 1) - (b["close"].iloc[-1] / b["open"].iloc[120] - 1)
    ha = A.groupby(A.index.tz_localize(None).normalize()).agg(o=("open", "first"), c=("close", "last"))
    hb = B.groupby(B.index.tz_localize(None).normalize()).agg(o=("open", "first"), c=("close", "last"))
    pos = ha.index.get_loc(day)
    dsp = (np.log(ha["c"] / ha["o"]) - np.log(hb["c"] / hb["o"])).iloc[pos - 20:pos]
    assert abs(tab["gross"][i, c_rev] - (-np.sign(s) * ret / np.sqrt((dsp ** 2).mean()))) < 1e-12, "trade reconstruction"
    # two legs are both charged
    cfg_c = {"point_value": 1.0, "tick_size": 0.01, "commission_per_side": 0.0}
    tab_c = pair_table(leg_table(A, session, gh), leg_table(B, session, gh), cfg_c, cfg_c, gh, gk)
    ea, eb = a["open"].iloc[120], b["open"].iloc[120]
    assert abs(tab_c["cost_bp"][i, c_rev] - (0.02 / ea + 0.02 / eb) * 1e4) < 1e-9, "two-leg cost"
    # null world vs reversion world; the market factor cancels
    _, t0 = hac_t(tab["gross"][:, c_rev])
    _, _, tab1 = tab_for(1.0)
    _, t1 = hac_t(tab1["gross"][:, c_rev])
    assert abs(t0) < 2.5 and t1 > 4, (t0, t1)
    ch, sm = select_cell(tab1["gross"], nh, nk)
    assert ch < nh * nk, "selection must pick the reversion mode"
    p1 = permutation_p(tab1["gross"], tab1["cost"], nh, nk, float(sm.max()))
    p0 = permutation_p(tab["gross"], tab["cost"], nh, nk, float(select_cell(tab["gross"], nh, nk)[1].max()))
    assert p1 < 0.01 and p0 > 0.05, (p1, p0)
    assert cscv_pbo(tab1["gross"], nh, nk)["share_test_sharpe_positive"] > 0.95
    _, X = pool({"x": tab, "y": tab1}, "gross")
    assert np.allclose(X, (tab["gross"] + tab1["gross"]) / 2)
    print(f"selftest ok: null t={t0:.2f}, reversion t={t1:.2f}, p_null={p0:.3f}, p_rev={p1:.4f}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "selftest":
        selftest()
    elif stage == "costcheck":
        stage_costcheck(read_spec(check_lock=False))
    elif stage == "is":
        stage_is(read_spec())
    elif stage == "oos":
        stage_oos(read_spec())
    else:
        raise SystemExit(__doc__)
