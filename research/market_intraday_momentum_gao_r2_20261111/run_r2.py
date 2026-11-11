"""R2 - Market Intraday Momentum (Gao, Han, Li, Zhou 2018) on index, gold and crude futures.

Runs exactly FROZEN_SPECIFICATION.json (hash-checked against SPEC_LOCK.json).

    python run_r2.py selftest   synthetic data only (signal causality, PnL sign, placebo)
    python run_r2.py a          campaign A: strict replication, PRE / IS / OOS / EXT, gates S1-S7
    python run_r2.py b          campaign B: variants on PRE and IS (OOS only under the read rule)
    python run_r2.py c          campaign C: regime terciles on PRE and IS (OOS only for a conditional candidate)
    python run_r2.py div        diversification against the frozen eight-sleeve book (OOS)
    python run_r2.py all        a, b, c, div in order
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
CACHE = HERE / "cache"
SEED = 20261111
N_PERM = 5000
N_BOOT = 5000
BLOCK = 10
M0, M1 = 8 * 60, 17 * 60 + 10          # wall-clock minutes kept in the day matrix
MAIN = ["MNQ", "MES", "MGC", "MCL"]


def hm(s: str) -> int:
    h, m = map(int, s.split(":"))
    return h * 60 + m


# ------------------------------------------------------------------ spec / io
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_spec() -> dict:
    p = HERE / "FROZEN_SPECIFICATION.json"
    lock = json.loads((HERE / "SPEC_LOCK.json").read_text())
    if sha256(p) != lock["spec_sha256"]:
        raise RuntimeError("Specification changed after freeze")
    return json.loads(p.read_text())


def day_matrix(path: Path) -> tuple[pd.DatetimeIndex, dict]:
    df = pd.read_parquet(path, columns=["open", "close", "volume", "instrument_id"])
    if df.index.tz is None:
        raise RuntimeError("Naive timestamps")
    df.index = df.index.tz_convert("America/New_York")
    w = df.index.tz_localize(None)
    m = np.asarray(w.hour * 60 + w.minute)
    keep = (m >= M0) & (m < M1) & (np.asarray(w.dayofweek) < 5)
    df, w, m = df[keep], w[keep], m[keep]
    dates = w.normalize()
    udates = pd.DatetimeIndex(np.unique(dates))
    di = udates.get_indexer(dates)
    mi = m - M0
    mats = {}
    for c in ["open", "close", "volume", "instrument_id"]:
        a = np.full((len(udates), M1 - M0), np.nan)
        a[di, mi] = df[c].to_numpy(float)
        mats[c] = a
    return udates, mats


def node(mats: dict, minute: int, tol: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """kind 'open': open of first bar in [minute, minute+tol-1]; 'close': close of last bar in [minute-tol, minute-1]."""
    cols = list(range(minute, minute + tol)) if kind == "open" else list(range(minute - 1, minute - 1 - tol, -1))
    idx = [c - M0 for c in cols]
    if min(idx) < 0 or max(idx) >= M1 - M0:
        raise ValueError("node outside the day matrix")
    P = mats["open" if kind == "open" else "close"][:, idx]
    I = mats["instrument_id"][:, idx]
    valid = np.isfinite(P)
    j = valid.argmax(axis=1)
    has = valid.any(axis=1)
    r = np.arange(len(P))
    return np.where(has, P[r, j], np.nan), np.where(has, I[r, j], np.nan)


EXTRA_NODES_EQUITY = {   # name: (minute, kind)
    "o1531": ("15:31", "open"), "c1531": ("15:31", "close"), "o1600": ("16:00", "open"),
    "o1525": ("15:25", "open"), "c1555": ("15:55", "close"), "o1535": ("15:35", "open"), "c1605": ("16:05", "close"),
    "c0955": ("09:55", "close"), "c1005": ("10:05", "close"), "c0945": ("09:45", "close"), "c1015": ("10:15", "close"),
    "c1030": ("10:30", "close"), "o1430": ("14:30", "open"), "c1500": ("15:00", "close"), "o1500": ("15:00", "open"),
    "c1530": ("15:30", "close"), "o1545": ("15:45", "open"), "o1515": ("15:15", "open"),
}


def session_table(udates: pd.DatetimeIndex, mats: dict, clock: dict, tol: int, equity: bool) -> pd.DataFrame:
    o, h, e, x = (hm(clock[k]) for k in ("open", "signal", "entry", "exit"))
    t = pd.DataFrame(index=udates)
    for name, mnt, kind in [("open", o, "open"), ("sig", h, "close"), ("entry", e, "open"), ("exit", x, "close")]:
        t["p_" + name], t["i_" + name] = node(mats, mnt, tol, kind)
    full = t[["p_open", "p_sig", "p_entry", "p_exit"]].notna().all(axis=1).to_numpy()
    cl = mats["close"][:, o - M0:17 * 60 - M0]
    has = np.isfinite(cl).any(axis=1)
    last = np.where(has, o + (cl.shape[1] - 1 - np.isfinite(cl[:, ::-1]).argmax(axis=1)), np.nan)
    early = equity & ~full & (last == 13 * 60 + 14)
    p_early, i_early = node(mats, 13 * 60, tol, "close")
    mark = np.where(full, t["p_exit"], np.where(early, p_early, np.nan))
    mark_i = np.where(full, t["i_exit"], np.where(early, i_early, np.nan))
    ns = udates.asi8.astype(float)
    mark_date = pd.Series(np.where(np.isfinite(mark), ns, np.nan), index=udates).ffill().shift(1)
    t["prev_close"] = pd.Series(mark, index=udates).ffill().shift(1)
    t["i_prev"] = pd.Series(mark_i, index=udates).ffill().shift(1)
    t["prev_gap_days"] = (ns - mark_date.to_numpy()) / 86400e9
    t["full"], t["early_close"] = full, early
    same = ((t["i_prev"] == t["i_sig"]) & (t["i_open"] == t["i_sig"]) & (t["i_entry"] == t["i_sig"])
            & (t["i_exit"] == t["i_sig"]))
    t["eligible"] = full & t["prev_close"].notna() & (t["prev_gap_days"] <= 5) & same
    if equity:
        for name, (hhmm, kind) in EXTRA_NODES_EQUITY.items():
            p, i = node(mats, hm(hhmm), tol, kind)
            t["p_" + name] = np.where(i == t["i_sig"].to_numpy(), p, np.nan)
    vol = mats["volume"]
    t["vol_first30"] = np.nansum(vol[:, o - M0:h - M0], axis=1)
    # roll window: within 10 calendar days before the next instrument_id change
    iid = t["i_sig"].where(t["full"])
    nxt_change = pd.Series(np.nan, index=udates)
    cur, nxt_date = np.nan, np.nan
    for d, v in zip(udates[::-1], iid.to_numpy()[::-1]):
        if np.isfinite(v):
            if np.isfinite(cur) and v != cur:
                nxt_date = last_seen_date
            cur, last_seen_date = v, d.value
        nxt_change[d] = nxt_date
    t["roll_window"] = ((nxt_change.to_numpy() - ns) / 86400e9 <= 10)
    t["r_first"] = t["p_sig"] / t["prev_close"] - 1
    t["r_last"] = t["p_exit"] / t["p_entry"] - 1
    return t


def load_tables(spec: dict) -> dict:
    CACHE.mkdir(exist_ok=True)
    out = {}
    for name, cfg in spec["instruments"].items():
        path = ROOT / cfg["path"]
        digest = sha256(path)
        if digest != cfg["sha256"]:
            raise RuntimeError(f"Input checksum mismatch: {cfg['path']}")
        cp = CACHE / f"{name}_{digest[:12]}_{sha256(Path(__file__))[:12]}.pkl"
        if cp.exists():
            out[name] = pickle.loads(cp.read_bytes())
            continue
        udates, mats = day_matrix(path)
        tabs = {}
        for ck in cfg["clocks"]:
            tabs[ck] = session_table(udates, mats, spec["clocks_et"][ck], int(cfg["node_tolerance_minutes"]), ck == "equity")
        cp.write_bytes(pickle.dumps(tabs))
        out[name] = tabs
        print(f"  loaded {name}")
    return out


def period_mask(spec: dict, inst: str, period: str, idx: pd.DatetimeIndex) -> np.ndarray:
    if period not in spec["instruments"][inst]["periods"]:
        return np.zeros(len(idx), bool)
    p = spec["periods"][period]
    a, b = p[inst] if period == "PRE" else p
    return np.asarray((idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)))


# ------------------------------------------------------------------ statistics
def nw_lag(n: int) -> int:
    return int(math.floor(4 * (n / 100) ** (2 / 9)))


def nw_t(x) -> float:
    x = np.asarray(x, float)
    n = len(x)
    if n < 20:
        return float("nan")
    u = x - x.mean()
    L = nw_lag(n)
    s = u @ u / n
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (u[l:] @ u[:-l]) / n
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else float("nan")


def ols_nw(y, X) -> tuple[np.ndarray, np.ndarray, float]:
    y, X = np.asarray(y, float), np.asarray(X, float)
    n = len(y)
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    g = X * u[:, None]
    L = nw_lag(n)
    S = g.T @ g
    for l in range(1, L + 1):
        G = g[l:].T @ g[:-l]
        S += (1 - l / (L + 1)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1 - (u @ u) / ((y - y.mean()) @ (y - y.mean()))
    return b, b / se, float(r2)


def p_one_sided(t: float) -> float:
    return float(0.5 * math.erfc(t / math.sqrt(2))) if np.isfinite(t) else 1.0


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for k, (key, p) in enumerate(items):
        run = max(run, min(1.0, (m - k) * p))
        out[key] = run
    return out


def boot_ci(x, rng) -> tuple[float, float]:
    x = np.asarray(x, float)
    n = len(x)
    if n < 20:
        return float("nan"), float("nan")
    means = np.empty(N_BOOT)
    ar = np.arange(n)
    for b in range(N_BOOT):
        new = rng.random(n) < 1 / BLOCK
        new[0] = True
        starts = rng.integers(n, size=n)
        beg = np.maximum.accumulate(np.where(new, ar, 0))
        means[b] = x[(starts[beg] + ar - beg) % n].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def perm_p(side, r_bp, c_bp, groups, rng) -> tuple[float, float]:
    """Share of permuted-signal draws whose mean net bp >= observed. Costs follow the permuted position."""
    side, r_bp, c_bp = map(lambda a: np.asarray(a, float), (side, r_bp, c_bp))
    obs = np.mean(side * r_bp - np.abs(side) * c_bp)
    order0 = np.argsort(groups, kind="stable")
    sims = np.empty(N_PERM)
    for b in range(N_PERM):
        ordr = np.lexsort((rng.random(len(side)), groups))
        s = np.empty_like(side)
        s[order0] = side[ordr]
        sims[b] = np.mean(s * r_bp - np.abs(s) * c_bp)
    return float((np.sum(sims >= obs) + 1) / (N_PERM + 1)), float(sims.mean())


# ------------------------------------------------------------------ strategy
def evaluate(t: pd.DataFrame, mask, side, entry, exit_, cfg: dict, cost_mult: float = 1.0) -> pd.DataFrame:
    side, entry, exit_ = (np.asarray(a, float) for a in (side, entry, exit_))
    ok = np.asarray(mask, bool) & np.isfinite(entry) & np.isfinite(exit_) & np.isfinite(side)
    pv, rt = cfg["point_value"], cost_mult * (2 * cfg["tick_size"] * cfg["point_value"] + 2 * cfg["commission_per_side"])
    s, en, ex = side[ok], entry[ok], exit_[ok]
    d = pd.DataFrame(index=t.index[ok])
    d["side"] = s
    d["r_bp"] = (ex / en - 1) * 1e4                 # unsigned window return
    d["c_bp"] = rt / (pv * en) * 1e4                 # cost if traded
    d["gross_bp"] = s * d["r_bp"]
    d["net_bp"] = d["gross_bp"] - np.abs(s) * d["c_bp"]
    d["gross_usd"] = s * (ex - en) * pv
    d["net_usd"] = d["gross_usd"] - np.abs(s) * rt
    return d


def metrics(d: pd.DataFrame) -> dict:
    n = len(d)
    if n == 0:
        return {"sessions": 0}
    tr = d[d["side"] != 0]
    net = d["net_bp"].to_numpy()
    sd = net.std(ddof=1) if n > 1 else float("nan")
    cum = d["net_usd"].cumsum()
    wins, losses = tr["net_usd"][tr["net_usd"] > 0], tr["net_usd"][tr["net_usd"] < 0]
    return {
        "sessions": n, "trades": len(tr), "long": int((tr["side"] > 0).sum()), "short": int((tr["side"] < 0).sum()),
        "gross_mean_bp": d["gross_bp"].mean(), "cost_mean_bp": (np.abs(d["side"]) * d["c_bp"]).mean(),
        "net_mean_bp": net.mean(), "nw_t_gross": nw_t(d["gross_bp"]), "nw_t_net": nw_t(net),
        "gross_usd": d["gross_usd"].sum(), "net_usd": d["net_usd"].sum(), "net_usd_per_year": d["net_usd"].mean() * 252,
        "sharpe": net.mean() / sd * math.sqrt(252) if sd and sd > 0 else float("nan"),
        "ann_return_pct": net.mean() * 252 / 100,
        "max_dd_usd": float((cum - cum.cummax()).min()),
        "hit_rate": float((tr["net_usd"] > 0).mean()) if len(tr) else float("nan"),
        "gross_hit_rate": float((tr["gross_usd"] > 0).mean()) if len(tr) else float("nan"),
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("nan"),
        "avg_win_usd": wins.mean(), "avg_loss_usd": losses.mean(),
    }


def distribution(d: pd.DataFrame) -> dict:
    tr = d[d["side"] != 0]["net_bp"]
    q = tr.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    out = {f"q{int(k * 100):02d}_bp": v for k, v in q.items()}
    out.update(skew=float(tr.skew()), kurtosis=float(tr.kurt()), worst_usd=float(d["net_usd"].min()),
               best_usd=float(d["net_usd"].max()))
    return out


def vix_regime(dates: pd.DatetimeIndex) -> pd.Series:
    v = pd.read_csv(ROOT / "data/raw/vix-cboe.csv")
    v["DATE"] = pd.to_datetime(v["DATE"], format="%m/%d/%Y")
    v = v.set_index("DATE")["CLOSE"].astype(float).sort_index()
    q1, q2 = v.rolling(252).quantile(1 / 3), v.rolling(252).quantile(2 / 3)
    reg = pd.Series(np.where(v <= q1, "low", np.where(v <= q2, "normal", "high")), index=v.index).where(q1.notna())
    src = pd.DataFrame({"d": v.index, "reg": reg.to_numpy()})
    tgt = pd.DataFrame({"d": dates})
    m = pd.merge_asof(tgt, src, on="d", allow_exact_matches=False)   # last VIX close strictly before the session
    return pd.Series(m["reg"].to_numpy(), index=dates)


def signal_A(t: pd.DataFrame, sig_col: str = "p_sig") -> np.ndarray:
    return np.sign(t[sig_col] / t["prev_close"] - 1).to_numpy()


# ------------------------------------------------------------------ campaign A
def campaign_A(spec: dict, tabs: dict) -> dict:
    rng = np.random.default_rng(SEED)
    rows, years, longshort, vixrows, dists, sens, plac, regs, roll = [], [], [], [], [], [], [], [], []
    daily = {}
    for inst, cfg in spec["instruments"].items():
        t = tabs[inst]["equity"]
        el = t["eligible"].to_numpy()
        side = signal_A(t)
        vix = vix_regime(t.index)
        for period in cfg["periods"]:
            pm = period_mask(spec, inst, period, t.index) & el
            if not pm.any():
                continue
            d = evaluate(t, pm, side, t["p_entry"], t["p_exit"], cfg)
            daily[(inst, period)] = d
            key = {"instrument": inst, "period": period}
            m = metrics(d)
            lo, hi = boot_ci(d["net_bp"], rng)
            m.update(boot_ci_lo_bp=lo, boot_ci_hi_bp=hi)
            g_year = d.index.year.to_numpy()
            m["perm_p_within_year"], m["perm_mean_within_year"] = perm_p(d["side"], d["r_bp"], d["c_bp"], g_year, rng)
            m["perm_p_full"], m["comparable_exposure_mean_bp"] = perm_p(d["side"], d["r_bp"], d["c_bp"], np.zeros(len(d)), rng)
            for lbl, sd in [("always_long", 1.0), ("always_short", -1.0)]:
                dd = evaluate(t, pm, np.full(len(t), sd), t["p_entry"], t["p_exit"], cfg)
                m[f"{lbl}_net_mean_bp"], m[f"{lbl}_nw_t"] = dd["net_bp"].mean(), nw_t(dd["net_bp"])
            rows.append({**key, **m})
            # replication regression r_last on r_first
            ok = pm & np.isfinite(t["r_first"]) & np.isfinite(t["r_last"])
            b, tt, r2 = ols_nw(t["r_last"][ok] * 1e4, np.column_stack([np.ones(ok.sum()), t["r_first"][ok] * 1e4]))
            regs.append({**key, "n": int(ok.sum()), "alpha_bp": b[0], "alpha_t": tt[0], "beta": b[1], "beta_t": tt[1], "r2_pct": 100 * r2})
            # years
            for y, g in d.groupby(d.index.year):
                years.append({**key, "year": y, "sessions": len(g), "net_usd": g["net_usd"].sum(), "gross_usd": g["gross_usd"].sum(),
                              "net_mean_bp": g["net_bp"].mean(), "nw_t_net": nw_t(g["net_bp"]),
                              "hit_rate": (g.loc[g["side"] != 0, "net_usd"] > 0).mean()})
            for lbl, sg in [("long", 1), ("short", -1)]:
                g = d[d["side"] == sg]
                mm = metrics(g)
                longshort.append({**key, "side": lbl, **{k: mm.get(k) for k in ["trades", "gross_mean_bp", "net_mean_bp", "nw_t_net", "net_usd", "hit_rate", "profit_factor"]}})
            for reg in ["low", "normal", "high"]:
                g = d[(vix.reindex(d.index) == reg).to_numpy()]
                if len(g):
                    mm = metrics(g)
                    vixrows.append({**key, "vix_tercile": reg, **{k: mm.get(k) for k in ["sessions", "gross_mean_bp", "net_mean_bp", "nw_t_net", "net_usd", "hit_rate"]}})
            dists.append({**key, **distribution(d)})
            # placebos
            prev_side = pd.Series(side[el], index=t.index[el]).shift(1).reindex(t.index).to_numpy()
            for lbl, sd_, en, ex in [("lag_signal", prev_side, t["p_entry"], t["p_exit"]),
                                     ("window_1430_1500", side, t["p_o1430"], t["p_c1500"]) if "p_o1430" in t else ("", None, None, None),
                                     ("window_1500_1530", side, t["p_o1500"], t["p_c1530"]) if "p_o1500" in t else ("", None, None, None)]:
                if sd_ is None:
                    continue
                dd = evaluate(t, pm, sd_, en, ex, cfg)
                plac.append({**key, "placebo": lbl, "sessions": len(dd), "gross_mean_bp": dd["gross_bp"].mean(),
                             "nw_t_gross": nw_t(dd["gross_bp"]), "net_mean_bp": dd["net_bp"].mean(), "nw_t_net": nw_t(dd["net_bp"])})
            # execution sensitivities
            cases = [("base", side, "p_entry", "p_exit", 1.0), ("cost_x2", side, "p_entry", "p_exit", 2.0)]
            if "p_o1531" in t:
                cases += [("entry_open_1531", side, "p_o1531", "p_exit", 1.0), ("entry_close_1530_bar", side, "p_c1531", "p_exit", 1.0),
                          ("exit_open_1600_bar", side, "p_entry", "p_o1600", 1.0), ("window_minus5", side, "p_o1525", "p_c1555", 1.0),
                          ("window_plus5", side, "p_o1535", "p_c1605", 1.0), ("signal_0955", signal_A(t, "p_c0955"), "p_entry", "p_exit", 1.0),
                          ("signal_1005", signal_A(t, "p_c1005"), "p_entry", "p_exit", 1.0)]
            for lbl, sd_, en, ex, cm in cases:
                dd = evaluate(t, pm, sd_, t[en], t[ex], cfg, cm)
                sens.append({**key, "case": lbl, "sessions": len(dd), "gross_mean_bp": dd["gross_bp"].mean(), "net_mean_bp": dd["net_bp"].mean(),
                             "nw_t_net": nw_t(dd["net_bp"]), "net_usd": dd["net_usd"].sum()})
        # rolling 252-session stability over IS + OOS (+ EXT)
        parts = [daily[(inst, p)] for p in ["IS", "OOS", "EXT"] if (inst, p) in daily]
        if parts:
            s = pd.concat(parts)["net_bp"].rolling(252).mean().dropna()
            if len(s):
                roll.append({"instrument": inst, "windows": len(s), "share_positive": float((s > 0).mean()),
                             "min_bp": float(s.min()), "median_bp": float(s.median()), "max_bp": float(s.max())})
    res = pd.DataFrame(rows)
    regs = pd.DataFrame(regs)
    # gates
    is_ = res[res["period"] == "IS"].set_index("instrument")
    pvals = {i: p_one_sided(is_.loc[i, "nw_t_net"]) for i in MAIN}
    adj = holm(pvals)
    sens_df = pd.DataFrame(sens)
    yrs = pd.DataFrame(years)
    gates = []
    for i in MAIN:
        r = is_.loc[i]
        sv = sens_df[(sens_df["instrument"] == i) & (sens_df["period"] == "IS")].set_index("case")["net_mean_bp"]
        y = yrs[(yrs["instrument"] == i) & (yrs["period"] == "IS")]
        proxy = spec["instruments"][i].get("pre_proxy")
        pre = regs[(regs["instrument"] == proxy) & (regs["period"] == "PRE")] if proxy else pd.DataFrame()
        oos = res[(res["instrument"] == i) & (res["period"] == "OOS")].iloc[0]
        g = {
            "instrument": i,
            "S1": bool(r["net_mean_bp"] > 0 and r["nw_t_net"] >= 2.0 and adj[i] <= 0.05),
            "S2": bool(r["perm_p_within_year"] <= 0.05),
            "S3": bool(r["perm_p_full"] <= 0.05),
            "S4": bool(sv["cost_x2"] > 0 and sv.get("entry_open_1531", sv["base"]) > 0),
            "S5": bool((y["net_usd"] > 0).sum() > len(y) / 2),
            "S6": bool(len(pre) and pre["beta"].iloc[0] > 0 and pre["beta_t"].iloc[0] >= 2.0),
            "S7": bool(oos["net_mean_bp"] > 0 and oos["nw_t_net"] >= 1.0),
            "is_t_net": r["nw_t_net"], "holm_p": adj[i], "is_slope_t": regs[(regs["instrument"] == i) & (regs["period"] == "IS")]["beta_t"].iloc[0],
            "is_gross_mean_bp": r["gross_mean_bp"], "is_cost_mean_bp": r["cost_mean_bp"],
        }
        g["S1_S6"] = all(g[k] for k in ["S1", "S2", "S3", "S4", "S5", "S6"])
        g["all"] = g["S1_S6"] and g["S7"]
        gates.append(g)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "A_summary.csv", index=False)
    regs.to_csv(OUT / "A_replication_regression.csv", index=False)
    yrs.to_csv(OUT / "A_years.csv", index=False)
    pd.DataFrame(longshort).to_csv(OUT / "A_long_short.csv", index=False)
    pd.DataFrame(vixrows).to_csv(OUT / "A_vix_terciles.csv", index=False)
    pd.DataFrame(dists).to_csv(OUT / "A_distribution.csv", index=False)
    sens_df.to_csv(OUT / "A_execution_sensitivity.csv", index=False)
    pd.DataFrame(plac).to_csv(OUT / "A_placebos.csv", index=False)
    pd.DataFrame(roll).to_csv(OUT / "A_rolling_252.csv", index=False)
    pd.DataFrame(gates).to_csv(OUT / "A_gates.csv", index=False)
    for (inst, period), d in daily.items():
        d.assign(instrument=inst, period=period).to_csv(OUT / f"A_daily_{inst}_{period}.csv")
    a_pass = any(g["S1_S6"] for g in gates)
    (OUT / "A_status.json").write_text(json.dumps({"any_S1_S6": a_pass, "gates": gates}, indent=2, default=float))
    return {"gates": gates, "any_S1_S6": a_pass}


# ------------------------------------------------------------------ campaign B
def campaign_B(spec: dict, tabs: dict) -> None:
    status = json.loads((OUT / "A_status.json").read_text())
    periods = ["PRE", "IS"] + (["OOS"] if status["any_S1_S6"] else [])
    rows, sec = [], []
    insts = MAIN + ["NQ", "CL"]
    for inst in insts:
        cfg = spec["instruments"][inst]
        t = tabs[inst]["equity"]
        el = t["eligible"].to_numpy()
        rf = pd.Series(t["r_first"].to_numpy()[el], index=t.index[el])
        sd = rf.shift(1).rolling(20, min_periods=15).std()
        z = (rf / sd).reindex(t.index).to_numpy()
        variants = {
            "B0_reference": (signal_A(t), "p_entry", "p_exit", t["r_first"]),
            "B1_vol_normalised": (np.where(np.abs(z) >= 0.5, np.sign(z), np.where(np.isfinite(z), 0.0, np.nan)), "p_entry", "p_exit", pd.Series(z, index=t.index)),
            "B2_no_gap": (np.sign(t["p_sig"] / t["p_open"] - 1).to_numpy(), "p_entry", "p_exit", t["p_sig"] / t["p_open"] - 1),
            "B3_gap_only": (np.sign(t["p_open"] / t["prev_close"] - 1).to_numpy(), "p_entry", "p_exit", t["p_open"] / t["prev_close"] - 1),
            "B4_penultimate_half_hour": (np.sign(t["p_c1530"] / t["p_c1500"] - 1).to_numpy(), "p_entry", "p_exit", t["p_c1530"] / t["p_c1500"] - 1),
        }
        tables = {"equity": (t, variants)}
        native = [c for c in cfg["clocks"] if c.startswith("native")]
        if native:
            tn = tabs[inst][native[0]]
            tables["native"] = (tn, {"B5_native_clock": (signal_A(tn), "p_entry", "p_exit", tn["r_first"])})
        for _, (tt, vs) in tables.items():
            elt = tt["eligible"].to_numpy()
            for period in periods:
                pm = period_mask(spec, inst, period, tt.index) & elt
                if not pm.any():
                    continue
                for v, (side, en, ex, pred) in vs.items():
                    d = evaluate(tt, pm, side, tt[en], tt[ex], cfg)
                    if len(d) < 50:
                        continue
                    m = metrics(d)
                    ok = pm & np.isfinite(pred.to_numpy()) & np.isfinite(tt["r_last"].to_numpy())
                    b, bt, r2 = ols_nw(tt["r_last"][ok] * 1e4, np.column_stack([np.ones(ok.sum()), pred[ok] * (1 if v == "B1_vol_normalised" else 1e4)]))
                    rows.append({"instrument": inst, "period": period, "variant": v, "sessions": m["sessions"], "trades": m["trades"],
                                 "gross_mean_bp": m["gross_mean_bp"], "net_mean_bp": m["net_mean_bp"], "nw_t_net": m["nw_t_net"],
                                 "net_usd": m["net_usd"], "sharpe": m["sharpe"], "hit_rate": m["hit_rate"],
                                 "slope": b[1], "slope_t": bt[1], "r2_pct": 100 * r2})
            # secondary sensitivities (equity clock only, PRE and IS only, not counted)
            if "p_c0945" in tt:
                for period in ["PRE", "IS"]:
                    pm = period_mask(spec, inst, period, tt.index) & elt
                    if not pm.any():
                        continue
                    for lbl, side, en in [("first_to_0945", signal_A(tt, "p_c0945"), "p_entry"), ("first_to_1015", signal_A(tt, "p_c1015"), "p_entry"),
                                          ("first_to_1030", signal_A(tt, "p_c1030"), "p_entry"), ("last_from_1545", signal_A(tt), "p_o1545"),
                                          ("last_from_1515", signal_A(tt), "p_o1515"), ("last_from_1500", signal_A(tt), "p_o1500")]:
                        d = evaluate(tt, pm, side, tt[en], tt["p_exit"], cfg)
                        sec.append({"instrument": inst, "period": period, "case": lbl, "sessions": len(d), "gross_mean_bp": d["gross_bp"].mean(),
                                    "nw_t_gross": nw_t(d["gross_bp"]), "net_mean_bp": d["net_bp"].mean(), "nw_t_net": nw_t(d["net_bp"])})
    df = pd.DataFrame(rows)
    is_ = df[(df["period"] == "IS") & (df["variant"] != "B0_reference") & df["instrument"].isin(MAIN)]
    adj = holm({(r.instrument, r.variant): p_one_sided(r.nw_t_net) for r in is_.itertuples()})
    df["holm_p_IS"] = [adj.get((r.instrument, r.variant), np.nan) if r.period == "IS" else np.nan for r in df.itertuples()]
    df.to_csv(OUT / "B_variants.csv", index=False)
    pd.DataFrame(sec).to_csv(OUT / "B_secondary_sensitivities.csv", index=False)
    (OUT / "B_status.json").write_text(json.dumps({"periods_read": periods, "n_holm_tests": len(adj)}, indent=2))


# ------------------------------------------------------------------ campaign C
def regimes(t: pd.DataFrame) -> pd.DataFrame:
    el = t["eligible"].to_numpy()
    out = pd.DataFrame(index=t.index)
    out["vix"] = vix_regime(t.index)

    def tercile_past(s: pd.Series) -> pd.Series:
        past = s.shift(1)
        q1 = past.rolling(252, min_periods=120).quantile(1 / 3)
        q2 = past.rolling(252, min_periods=120).quantile(2 / 3)
        return pd.Series(np.where(s <= q1, "low", np.where(s <= q2, "normal", "high")), index=s.index).where(q1.notna() & s.notna())

    rv = pd.Series(np.abs(t["p_sig"] / t["p_open"] - 1).to_numpy()[el], index=t.index[el])
    out["realized_first30"] = tercile_past(rv).reindex(t.index)
    nr = el & ~t["roll_window"].to_numpy()
    v = pd.Series(t["vol_first30"].to_numpy()[nr], index=t.index[nr])
    ratio = v / v.shift(1).rolling(20, min_periods=15).median()
    out["volume_first30"] = tercile_past(ratio.dropna()).reindex(t.index)
    return out


def campaign_C(spec: dict, tabs: dict) -> None:
    rng = np.random.default_rng(SEED + 1)
    cells, diffs = [], []
    store = {}
    for inst in MAIN + ["NQ", "CL"]:
        cfg = spec["instruments"][inst]
        t = tabs[inst]["equity"]
        reg = regimes(t)
        side = signal_A(t)
        for period in ["PRE", "IS"]:
            pm = period_mask(spec, inst, period, t.index) & t["eligible"].to_numpy()
            if not pm.any():
                continue
            for cat in ["vix", "realized_first30", "volume_first30"]:
                lab = reg[cat].to_numpy()
                for terc in ["low", "normal", "high"]:
                    mk = pm & (lab == terc)
                    d = evaluate(t, mk, side, t["p_entry"], t["p_exit"], cfg)
                    if len(d) < 50:
                        continue
                    m = metrics(d)
                    ok = mk & np.isfinite(t["r_first"]).to_numpy() & np.isfinite(t["r_last"]).to_numpy()
                    b, bt, _ = ols_nw(t["r_last"][ok] * 1e4, np.column_stack([np.ones(ok.sum()), t["r_first"][ok] * 1e4]))
                    yr = d.groupby(d.index.year)["net_usd"].sum()
                    row = {"instrument": inst, "period": period, "category": cat, "tercile": terc, "sessions": m["sessions"],
                           "gross_mean_bp": m["gross_mean_bp"], "net_mean_bp": m["net_mean_bp"], "nw_t_net": m["nw_t_net"],
                           "net_usd": m["net_usd"], "slope": b[1], "slope_t": bt[1], "years_positive": int((yr > 0).sum()), "years": len(yr)}
                    if period == "IS" and inst in ("MNQ", "MES") and m["net_mean_bp"] > 0:
                        row["perm_p_within_year"] = perm_p(d["side"], d["r_bp"], d["c_bp"], d.index.year.to_numpy(), rng)[0]
                    cells.append(row)
                    store[(inst, period, cat, terc)] = d
                # high minus low
                both = pm & np.isin(lab, ["low", "high"])
                d = evaluate(t, both, side, t["p_entry"], t["p_exit"], cfg)
                if len(d) > 100:
                    D = (reg[cat].reindex(d.index) == "high").to_numpy(float)
                    b1, t1, _ = ols_nw(d["net_bp"], np.column_stack([np.ones(len(d)), D]))
                    rf = t["r_first"].reindex(d.index).to_numpy() * 1e4
                    rl = t["r_last"].reindex(d.index).to_numpy() * 1e4
                    b2, t2, _ = ols_nw(rl, np.column_stack([np.ones(len(d)), rf, D, D * rf]))
                    diffs.append({"instrument": inst, "period": period, "category": cat, "net_diff_high_low_bp": b1[1], "net_diff_t": t1[1],
                                  "slope_diff_high_low": b2[3], "slope_diff_t": t2[3]})
    df = pd.DataFrame(cells)
    is_ = df[(df["period"] == "IS") & df["instrument"].isin(MAIN)]
    adj = holm({(r.instrument, r.category, r.tercile): p_one_sided(r.nw_t_net) for r in is_.itertuples()})
    df["holm_p_IS"] = [adj.get((r.instrument, r.category, r.tercile), np.nan) if r.period == "IS" else np.nan for r in df.itertuples()]
    status = json.loads((OUT / "A_status.json").read_text())
    s1_fail = not any(g["S1"] for g in status["gates"] if g["instrument"] in ("MNQ", "MES"))
    cands = []
    for r in df[(df["period"] == "IS") & df["instrument"].isin(["MNQ", "MES"])].itertuples():
        pre = df[(df["instrument"] == "NQ") & (df["period"] == "PRE") & (df["category"] == r.category) & (df["tercile"] == r.tercile)]
        ok = (r.net_mean_bp > 0 and r.holm_p_IS <= 0.05 and getattr(r, "perm_p_within_year", 1.0) <= 0.05
              and r.years_positive > r.years / 2 and len(pre) and pre["slope"].iloc[0] > 0 and pre["slope_t"].iloc[0] >= 2.0)
        if ok:
            cands.append({"instrument": r.instrument, "category": r.category, "tercile": r.tercile})
    oos = []
    if cands and s1_fail:
        for c in cands:
            inst = c["instrument"]
            cfg = spec["instruments"][inst]
            t = tabs[inst]["equity"]
            reg = regimes(t)
            pm = period_mask(spec, inst, "OOS", t.index) & t["eligible"].to_numpy() & (reg[c["category"]].to_numpy() == c["tercile"])
            d = evaluate(t, pm, signal_A(t), t["p_entry"], t["p_exit"], cfg)
            m = metrics(d)
            oos.append({**c, "oos_sessions": m["sessions"], "oos_net_mean_bp": m["net_mean_bp"], "oos_nw_t": m["nw_t_net"],
                        "pass": bool(m["net_mean_bp"] > 0 and m["nw_t_net"] >= 1.0)})
    df.to_csv(OUT / "C_regime_cells.csv", index=False)
    pd.DataFrame(diffs).to_csv(OUT / "C_high_minus_low.csv", index=False)
    (OUT / "C_status.json").write_text(json.dumps({"n_holm_tests": len(adj), "s1_failed_on_MNQ_MES": s1_fail,
                                                  "conditional_candidates": cands, "oos_read": oos}, indent=2, default=float))


# ------------------------------------------------------------------ diversification
def sharpe(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * math.sqrt(252)) if x.std(ddof=1) > 0 else float("nan")


def max_dd(x) -> float:
    c = np.cumsum(x)
    return float((c - np.maximum.accumulate(c)).min())


def worst_drawdowns(pnl: pd.Series, k: int = 5) -> list[tuple]:
    c = pnl.cumsum()
    peak = c.cummax()
    out, used = [], np.zeros(len(c), bool)
    dd = (c - peak).to_numpy()
    for _ in range(k):
        cand = np.where(used, 0, dd)
        j = int(np.argmin(cand))
        if cand[j] >= 0:
            break
        i = int(np.argmax(c.to_numpy()[: j + 1] == peak.to_numpy()[j]))
        out.append((c.index[i], c.index[j], float(dd[j])))
        used[i: j + 1] = True
        # also block the recovery after the trough until a new peak
        rec = np.where(c.to_numpy()[j:] >= peak.to_numpy()[j])[0]
        used[j: j + (rec[0] if len(rec) else len(c) - j)] = True
    return out


def diversification(spec: dict) -> None:
    book = pd.read_csv(ROOT / spec["diversification"]["book"].split(" ")[0], parse_dates=["session_date"]).set_index("session_date")
    sleeves = [c for c in book.columns if c != "portfolio_pnl_usd"]
    B = book["portfolio_pnl_usd"]
    rows, loo, sleeves_corr, ddrows = [], [], [], []
    for inst in MAIN:
        p = OUT / f"A_daily_{inst}_OOS.csv"
        s = pd.read_csv(p, parse_dates=[0], index_col=0)["net_usd"].reindex(book.index).fillna(0.0)
        tot = B + s
        neg = B < 0
        cov = np.cov(s, tot)[0, 1]
        rows.append({
            "instrument": inst, "sessions_book": len(B), "strategy_active_days": int((s != 0).sum()),
            "corr_book": float(np.corrcoef(s, B)[0, 1]), "corr_book_negative_days": float(np.corrcoef(s[neg], B[neg])[0, 1]),
            "strategy_pnl_usd": float(s.sum()), "book_pnl_usd": float(B.sum()), "contribution_pct": float(100 * s.sum() / tot.sum()),
            "sharpe_book": sharpe(B), "sharpe_with": sharpe(tot), "sharpe_strategy": sharpe(s),
            "max_dd_book": max_dd(B), "max_dd_with": max_dd(tot), "max_dd_strategy": max_dd(s),
            "variance_share_pct": float(100 * cov / tot.var()), "strategy_sd_usd": float(s.std()), "book_sd_usd": float(B.std()),
            "strategy_pnl_on_book_negative_days": float(s[neg].sum()),
            "marginal_positive": bool(sharpe(tot) > sharpe(B) and max_dd(tot) >= 1.10 * max_dd(B)),
        })
        for sl in sleeves:
            sleeves_corr.append({"instrument": inst, "sleeve": sl, "corr": float(np.corrcoef(s, book[sl])[0, 1]) if book[sl].std() > 0 else np.nan})
        comps = book[sleeves].assign(R2=s)
        for c in comps.columns:
            x = comps.drop(columns=c).sum(axis=1)
            loo.append({"instrument": inst, "removed": c, "sharpe": sharpe(x), "max_dd_usd": max_dd(x), "pnl_usd": float(x.sum())})
        for a, b, depth in worst_drawdowns(B):
            ddrows.append({"instrument": inst, "dd_peak": a.date(), "dd_trough": b.date(), "book_dd_usd": depth,
                           "strategy_pnl_usd": float(s[a:b].sum())})
    pd.DataFrame(rows).to_csv(OUT / "D_diversification.csv", index=False)
    pd.DataFrame(loo).to_csv(OUT / "D_leave_one_out.csv", index=False)
    pd.DataFrame(sleeves_corr).to_csv(OUT / "D_sleeve_correlations.csv", index=False)
    pd.DataFrame(ddrows).to_csv(OUT / "D_book_drawdowns.csv", index=False)


# ------------------------------------------------------------------ self-test
def selftest() -> None:
    """Synthetic minutes with a causal continuation injected over 15:30-16:00; checks PnL sign, causality, lag placebo."""
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2020-01-01", periods=400)
    prev_close, chunks = None, []
    p = 10000.0
    for d in days:
        mins = pd.date_range(d + pd.Timedelta(hours=8), d + pd.Timedelta(hours=17, minutes=9), freq="1min")
        mm = np.asarray(mins.hour * 60 + mins.minute)
        path = p * np.exp(np.cumsum(rng.normal(0, 2e-4, len(mins))))
        if prev_close is not None:
            s = np.sign(path[mm == 599][0] / prev_close - 1)
            path = path * (1 + s * 5e-4 * np.clip((mm - 930) / 29, 0, 1))
        prev_close = path[mm == 959][0]
        p = path[-1]
        chunks.append(pd.DataFrame({"open": path, "close": path, "volume": 10.0, "instrument_id": 1.0},
                                   index=mins.tz_localize("America/New_York")))
    CACHE.mkdir(exist_ok=True)
    tmp = CACHE / "selftest.parquet"
    pd.concat(chunks).to_parquet(tmp)
    clock = {"open": "09:30", "signal": "10:00", "entry": "15:30", "exit": "16:00"}
    udates, mats = day_matrix(tmp)
    t = session_table(udates, mats, clock, 1, True)
    cfg = {"point_value": 2.0, "tick_size": 0.25, "commission_per_side": 0.0}
    el = t["eligible"].to_numpy()
    d = evaluate(t, el, signal_A(t), t["p_entry"], t["p_exit"], cfg)
    assert d["gross_bp"].mean() > 3, d["gross_bp"].mean()
    mats2 = {k: v.copy() for k, v in mats.items()}
    for k in ("open", "close"):
        mats2[k][:, 600 - M0:] *= rng.uniform(0.9, 1.1, size=mats2[k][:, 600 - M0:].shape)
    t2 = session_table(udates, mats2, clock, 1, True)
    assert np.allclose(t["p_sig"], t2["p_sig"], equal_nan=True) and np.allclose(t["p_open"], t2["p_open"], equal_nan=True)
    lag = pd.Series(signal_A(t)[el], index=t.index[el]).shift(1).reindex(t.index).to_numpy()
    dl = evaluate(t, el, lag, t["p_entry"], t["p_exit"], cfg)
    assert abs(nw_t(dl["gross_bp"])) < 3, nw_t(dl["gross_bp"])
    rng2 = np.random.default_rng(2)
    pp, _ = perm_p(d["side"], d["r_bp"], d["c_bp"], d.index.year.to_numpy(), rng2)
    assert pp < 0.01, pp
    print(f"selftest OK: n={len(d)}, injected gross {d['gross_bp'].mean():.2f} bp, lag placebo t {nw_t(dl['gross_bp']):.2f}, perm p {pp:.4f}")
    tmp.unlink()


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "selftest":
        selftest()
        return
    spec = read_spec()
    OUT.mkdir(exist_ok=True)
    tabs = load_tables(spec)
    if cmd in ("a", "all"):
        r = campaign_A(spec, tabs)
        print(json.dumps(r, indent=1, default=float))
    if cmd in ("b", "all"):
        campaign_B(spec, tabs)
    if cmd in ("c", "all"):
        campaign_C(spec, tabs)
    if cmd in ("div", "all"):
        diversification(spec)


if __name__ == "__main__":
    main()
