"""R1 - Conditional VWAP mean reversion (H1 VWAP deviation, H2 volatility exhaustion, H3 lunch hour).

Runs exactly FROZEN_SPECIFICATION.json (hash-checked against SPEC_LOCK.json).

    python run_r1.py selftest   synthetic data only
    python run_r1.py is         all cells on IS and PRE, central-cell diagnostics, IS gate (EXT read for MGC G7 only)
    python run_r1.py oos        refused unless a central cell passed the IS gate and OOS_APPROVED.json exists
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
CACHE = HERE / "cache"
SEED = 20261111
N_PLACEBO = 1000
MAIN = ["MNQ", "MGC", "MCL"]
TESTS = ["H1_N1", "H1_N2", "H2_D1", "H2_D2", "H3_VWAP", "H3_RANGE"]


def hm(s: str) -> int:
    a, b = map(int, s.split(":"))
    return a * 60 + b


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def read_spec() -> dict:
    p = HERE / "FROZEN_SPECIFICATION.json"
    lock = json.loads((HERE / "SPEC_LOCK.json").read_text())
    if sha256(p) != lock["spec_sha256"]:
        raise RuntimeError("Specification changed after freeze")
    return json.loads(p.read_text())


# ======================================================================= grid + features
@dataclass
class Grid:
    name: str
    dates: pd.DatetimeIndex
    M: int
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray       # forward-filled close
    V: np.ndarray       # volume, 0 where missing
    eligible: np.ndarray
    f: dict             # features
    cfg: dict


def build_grid(name: str, cfg: dict, df: pd.DataFrame | None = None) -> Grid:
    if df is None:
        df = pd.read_parquet(ROOT / cfg["path"], columns=["open", "high", "low", "close", "volume", "instrument_id"])
    if df.index.tz is None:
        raise RuntimeError("Naive timestamps")
    df.index = df.index.tz_convert("America/New_York")
    w = df.index.tz_localize(None)
    m = np.asarray(w.hour * 60 + w.minute)
    s0, s1 = (hm(t) for t in cfg["session"])
    keep = (m >= s0) & (m < s1) & (np.asarray(w.dayofweek) < 5)
    df, w, m = df[keep], w[keep], m[keep]
    d = w.normalize()
    dates = pd.DatetimeIndex(np.unique(d))
    di, mi = dates.get_indexer(d), m - s0
    M = s1 - s0
    arr = {}
    for c in ["open", "high", "low", "close", "volume", "instrument_id"]:
        a = np.full((len(dates), M), np.nan)
        a[di, mi] = df[c].to_numpy(float)
        arr[c] = a
    O, H, L, Craw, Vr = arr["open"], arr["high"], arr["low"], arr["close"], arr["volume"]
    coverage = np.isfinite(Craw).sum(1) / M
    complete = coverage >= 0.9
    iid = np.nanmax(arr["instrument_id"], axis=1)
    C = pd.DataFrame(Craw).ffill(axis=1).to_numpy()
    V = np.nan_to_num(Vr)
    # roll exclusion: k sessions before an instrument_id change
    k = int(cfg["pre_roll_excluded_sessions"])
    pre_roll = np.zeros(len(dates), bool)
    ch = np.flatnonzero(np.diff(iid) != 0) + 1
    for p in ch:
        pre_roll[max(0, p - k):p] = True
    f = features(O, H, L, C, V, complete, M)
    eligible = complete & ~pre_roll & np.isfinite(f["sigma_day"]) & np.isfinite(f["sigma5"])
    return Grid(name, dates, M, O, H, L, C, V, eligible, f, cfg)


def _roll_prev(x: np.ndarray, mask: np.ndarray, fn: str, win: int = 20, minp: int = 15) -> np.ndarray:
    """Statistic over the `win` previous sessions where mask is True (causal), mapped back to all sessions."""
    s = pd.Series(np.where(mask, x, np.nan))
    sub = s.dropna()
    r = getattr(sub.rolling(win, min_periods=minp), fn)().shift(1)
    return r.reindex(s.index).ffill().to_numpy() if fn != "x" else None


def features(O, H, L, C, V, complete, M) -> dict:
    n = O.shape[0]
    tp = np.where(np.isfinite(H), (H + L + C) / 3, 0.0)
    cv = np.cumsum(V, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.cumsum(V * tp, 1) / cv
        var = np.cumsum(V * tp * tp, 1) / cv - vwap ** 2
    vwap = pd.DataFrame(vwap).ffill(axis=1).to_numpy()
    sig_vw = pd.DataFrame(np.sqrt(np.clip(var, 0, None))).ffill(axis=1).to_numpy()
    vwap_prev = np.concatenate([np.full((n, 1), np.nan), vwap[:, :-1]], 1)
    nb = M // 5
    idx_close = np.arange(nb) * 5 + 4
    c5 = C[:, idx_close]
    h5 = np.stack([np.nanmax(H[:, 5 * b:5 * b + 5], 1) for b in range(nb)], 1)
    l5 = np.stack([np.nanmin(L[:, 5 * b:5 * b + 5], 1) for b in range(nb)], 1)
    v5 = np.stack([V[:, 5 * b:5 * b + 5].sum(1) for b in range(nb)], 1)
    vw5, sv5 = vwap[:, idx_close], sig_vw[:, idx_close]
    first_open = pd.DataFrame(O).bfill(axis=1).to_numpy()[:, 0]
    # efficiency ratio from the session open
    prev = np.concatenate([first_open[:, None], c5[:, :-1]], 1)
    path = np.cumsum(np.abs(c5 - prev), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        er = np.abs(c5 - first_open[:, None]) / path
    # daily scales (previous complete sessions only)
    with np.errstate(invalid="ignore", divide="ignore"):
        dret = np.log(C[:, -1] / first_open)
        r5 = np.diff(np.log(c5), axis=1)
    s = pd.Series(np.where(complete, dret, np.nan)).dropna()
    sigma_day = s.rolling(20, min_periods=15).std().shift(1).reindex(range(n)).ffill().to_numpy()
    ss = pd.Series(np.where(complete, np.nansum(r5 ** 2, 1), np.nan)).dropna()
    cnt = pd.Series(np.where(complete, np.isfinite(r5).sum(1), np.nan)).dropna()
    sigma5 = np.sqrt(ss.rolling(20, min_periods=15).sum().shift(1) / cnt.rolling(20, min_periods=15).sum().shift(1))
    sigma5 = sigma5.reindex(range(n)).ffill().to_numpy()
    # session running high / low on 5m bars
    hs, ls = np.fmax.accumulate(h5, axis=1), np.fmin.accumulate(l5, axis=1)
    # volume of the last 30 minutes (bars b-5..b)
    v30 = pd.DataFrame(v5).T.rolling(6, min_periods=6).sum().T.to_numpy()
    return dict(vwap=vwap, vwap_prev=vwap_prev, c5=c5, h5=h5, l5=l5, v5=v5, vw5=vw5, sv5=sv5, er=er,
                sigma_day=sigma_day, sigma5=sigma5, hs=hs, ls=ls, v30=v30, first_open=first_open, nb=nb)


# ======================================================================= signals
def fire(score: np.ndarray, th: float, hyst: float, cond: np.ndarray) -> np.ndarray:
    """Contrarian side per (session, bar): -1 short when score >= th, +1 long when score <= -th, with re-arm."""
    n, nb = score.shape
    out = np.zeros((n, nb), np.int8)
    arm_s = np.ones(n, bool)
    arm_l = np.ones(n, bool)
    for b in range(nb):
        z = score[:, b]
        ok = cond[:, b] & np.isfinite(z)
        fs = ok & arm_s & (z >= th)
        fl = ok & arm_l & (z <= -th)
        out[fs, b], out[fl, b] = -1, 1
        arm_s = np.where(fs, False, arm_s | (np.isfinite(z) & (z < th - hyst)))
        arm_l = np.where(fl, False, arm_l | (np.isfinite(z) & (z > -th + hyst)))
    return out


def window_mask(g: Grid, test: str) -> np.ndarray:
    cfg, M, nb = g.cfg, g.M, g.f["nb"]
    e = np.arange(nb) * 5 + 5
    s0 = hm(cfg["session"][0])
    if test.startswith("H3"):
        a, b = (hm(t) - s0 for t in cfg["lunch_entry"])
        ok = (e >= a) & (e < b)
    else:
        ok = (e >= 30) & (e <= M - 30) & (e < hm(cfg["forced_exit"]) - s0)
        if test.startswith("H2"):
            ok &= np.arange(nb) >= 6
    return np.broadcast_to(ok, (len(g.dates), nb)) & g.eligible[:, None]


def exit_minute(g: Grid, test: str, e: np.ndarray) -> np.ndarray:
    s0 = hm(g.cfg["session"][0])
    if test.startswith("H3"):
        return np.full(len(e), hm(g.cfg["lunch_exit"]) - s0)
    return np.minimum(e + 60, hm(g.cfg["forced_exit"]) - s0)


def candidates(g: Grid, test: str, p: dict) -> dict:
    """All candidate entries (before position gating) for one cell."""
    f, nb = g.f, g.f["nb"]
    win = window_mask(g, test)
    tau = (np.arange(nb) * 5 + 5) / g.M
    with np.errstate(invalid="ignore", divide="ignore"):
        if test in ("H1_N1", "H3_VWAP"):
            score = (f["c5"] - f["vw5"]) / np.where(f["sv5"] > 0, f["sv5"], np.nan)
        elif test == "H1_N2":
            score = (f["c5"] / f["vw5"] - 1) / (f["sigma_day"][:, None] * np.sqrt(tau)[None, :])
        elif test.startswith("H2"):
            lc = np.log(f["c5"])
            mv = np.full_like(lc, np.nan)
            mv[:, 6:] = lc[:, 6:] - lc[:, :-6]
            score = mv / (f["sigma5"][:, None] * math.sqrt(6))
        elif test == "H3_RANGE":
            rngw = f["hs"] - f["ls"]
            score = (f["c5"] - f["ls"]) / np.where(rngw > 0, rngw, np.nan) - 0.5
    cond = win.copy()
    if test.startswith("H1"):
        if p["er_max"] is not None:
            cond &= f["er"] <= p["er_max"]
        th, hyst = p["k"], 0.5
    elif test.startswith("H2"):
        r = np.full_like(f["c5"], np.nan)
        r[:, 1:] = np.diff(np.log(f["c5"]), axis=1)
        sgn = np.sign(score)
        if test == "H2_D1":
            rec = np.full_like(r, np.nan)
            pri = np.full_like(r, np.nan)
            rec[:, 6:] = np.sqrt((r[:, 6:] ** 2 + r[:, 5:-1] ** 2) / 2)
            pri[:, 6:] = np.sqrt((r[:, 4:-2] ** 2 + r[:, 3:-3] ** 2 + r[:, 2:-4] ** 2 + r[:, 1:-5] ** 2) / 4)
            dec = rec <= 0.5 * pri
        else:
            hprev = np.full_like(r, np.nan)
            lprev = np.full_like(r, np.nan)
            for b in range(6, nb):
                hprev[:, b] = np.nanmax(f["h5"][:, b - 5:b], 1)
                lprev[:, b] = np.nanmin(f["l5"][:, b - 5:b], 1)
            stall = (np.sign(r) != sgn) | (r == 0)
            noext = np.where(sgn > 0, f["h5"] < hprev, f["l5"] > lprev)
            dec = stall & noext
        cond &= dec
        if p["volume_fade"]:
            v = f["v5"]
            vr = np.full_like(v, np.nan)
            vp = np.full_like(v, np.nan)
            vr[:, 6:] = (v[:, 6:] + v[:, 5:-1]) / 2
            vp[:, 6:] = (v[:, 4:-2] + v[:, 3:-3] + v[:, 2:-4] + v[:, 1:-5]) / 4
            cond &= vr <= vp
        th, hyst = p["x"], 0.5
    else:
        if p["low_volume"]:
            v30 = pd.DataFrame(np.where(g.eligible[:, None], f["v30"], np.nan))
            sub = v30[g.eligible]
            med = sub.rolling(20, min_periods=15).median().shift(1).reindex(v30.index).to_numpy()
            cond &= f["v30"] <= med
        if test == "H3_VWAP":
            th, hyst = p["k"], 0.5
        else:
            th, hyst = p["p"] - 0.5, 0.05
    side = fire(score, th, hyst, cond)
    s_idx, b_idx = np.nonzero(side)
    d = side[s_idx, b_idx].astype(int)
    e = b_idx * 5 + 5
    out = {"s": s_idx, "b": b_idx, "d": d, "e": e, "xmax": exit_minute(g, test, e)}
    # signal-time levels
    if test in ("H1_N1", "H1_N2", "H3_VWAP"):
        out["kind"] = "vwap"
        out["anchor"] = f["vw5"][s_idx, b_idx]
        out["target"] = np.full(len(e), np.nan)
    elif test == "H3_RANGE":
        out["kind"] = "fixed"
        out["target"] = (f["hs"][s_idx, b_idx] + f["ls"][s_idx, b_idx]) / 2
        out["anchor"] = out["target"]
    else:
        out["kind"] = "h2"
        c_now, c_then = f["c5"][s_idx, b_idx], f["c5"][s_idx, b_idx - 6]
        out["target"] = c_now - 0.5 * (c_now - c_then)
        ext_hi = np.array([np.nanmax(f["h5"][s, b - 5:b + 1]) for s, b in zip(s_idx, b_idx)]) if len(s_idx) else np.array([])
        ext_lo = np.array([np.nanmin(f["l5"][s, b - 5:b + 1]) for s, b in zip(s_idx, b_idx)]) if len(s_idx) else np.array([])
        out["extreme"] = np.where(d < 0, ext_hi, ext_lo)
        out["min_dist"] = 0.5 * f["sigma5"][s_idx] * math.sqrt(6) * c_now
        out["anchor"] = out["target"]
    return out


# ======================================================================= simulation
def simulate(g: Grid, s, e, d, stop_of_entry, target_fixed, moving: bool, xmax, cost_mult: float = 1.0) -> pd.DataFrame:
    """Vectorised minute-level simulation. stop_of_entry(entry_fill) -> stop level."""
    tick, pv, comm = g.cfg["tick_size"], g.cfg["point_value"], g.cfg["commission_per_side"]
    s, e, d, xmax = (np.asarray(a, int) for a in (s, e, d, xmax))
    n = len(s)
    if n == 0:
        return pd.DataFrame()
    raw_entry = g.O[s, np.minimum(e, g.M - 1)]
    entry = raw_entry + d * tick
    stop = stop_of_entry(entry)
    W = int(max(1, (xmax - e).max()))
    J = e[:, None] + np.arange(W)[None, :]
    valid = J < xmax[:, None]
    Jc = np.minimum(J, g.M - 1)
    Hh, Ll, Oo, Cc = g.H[s[:, None], Jc], g.L[s[:, None], Jc], g.O[s[:, None], Jc], g.C[s[:, None], Jc]
    T = g.f["vwap_prev"][s[:, None], Jc] if moving else np.broadcast_to(np.asarray(target_fixed)[:, None], (n, W))
    lg = (d == 1)[:, None]
    with np.errstate(invalid="ignore"):
        sh = valid & np.where(lg, Ll <= stop[:, None], Hh >= stop[:, None])
        th = valid & np.where(lg, Hh >= T + tick, Ll <= T - tick)
    fs = np.where(sh.any(1), sh.argmax(1), W)
    ft = np.where(th.any(1), th.argmax(1), W)
    last = np.clip(xmax - e - 1, 0, W - 1)
    r = np.arange(n)
    reason = np.where((fs <= ft) & (fs < W), "stop", np.where(ft < W, "target", "time"))
    xi = np.where(reason == "stop", fs, np.where(reason == "target", ft, last))
    o_at = Oo[r, xi]
    stop_px = np.where(d == 1, np.fmin(np.where(np.isfinite(o_at), o_at, stop), stop) - tick,
                       np.fmax(np.where(np.isfinite(o_at), o_at, stop), stop) + tick)
    exit_px = np.where(reason == "stop", stop_px, np.where(reason == "target", T[r, xi], Cc[r, last] - d * tick))
    slip_ticks = 1 + (reason != "target").astype(int)
    net = d * (exit_px - entry) * pv - 2 * comm
    unit_cost = 2 * comm + slip_ticks * tick * pv
    gross = net + unit_cost
    net = gross - cost_mult * unit_cost
    risk = np.abs(entry - stop) * pv
    # MAE / MFE up to the exit minute
    upto = np.arange(W)[None, :] <= xi[:, None]
    lo = np.nanmin(np.where(upto, Ll, np.nan), 1)
    hi = np.nanmax(np.where(upto, Hh, np.nan), 1)
    mae = np.where(d == 1, lo - entry, entry - hi) * pv / risk
    mfe = np.where(d == 1, hi - entry, entry - lo) * pv / risk
    tgt0 = T[:, 0]
    ok = (np.isfinite(raw_entry) & np.isfinite(stop) & (risk > 0) & np.isfinite(tgt0)
          & (d * (tgt0 - entry) > 0) & (d * (entry - stop) > 0))
    out = pd.DataFrame({"s": s, "e": e, "d": d, "entry": entry, "stop": stop, "target0": tgt0, "exit": exit_px,
                        "exit_min": e + xi, "reason": reason, "hold_min": xi + 1, "gross_usd": gross, "net_usd": net,
                        "risk_usd": risk, "R": net / risk, "gross_R": gross / risk, "mae_R": mae, "mfe_R": mfe})
    return out[ok].reset_index(drop=True)


def stop_fn(g: Grid, c: dict, idx: np.ndarray):
    if c["kind"] == "h2":
        ext, md, d = c["extreme"][idx], c["min_dist"][idx], c["d"][idx]
        tick = g.cfg["tick_size"]
        base = ext - d * tick
        return lambda en: np.where(d == 1, np.fmin(base, en - md), np.fmax(base, en + md))
    anchor, d = c["anchor"][idx], c["d"][idx]
    return lambda en: en - d * np.abs(en - anchor)


def run_cell(g: Grid, test: str, p: dict, delay: int = 0, cost_mult: float = 1.0) -> pd.DataFrame:
    c = candidates(g, test, p)
    if len(c["s"]) == 0:
        return pd.DataFrame()
    idx = np.arange(len(c["s"]))
    e = c["e"] + delay
    t = simulate(g, c["s"], e, c["d"], stop_fn(g, c, idx), c["target"], c["kind"] == "vwap", c["xmax"], cost_mult)
    if t.empty:
        return t
    # position gating: one position at a time per session
    keep, last_s, last_x = [], -1, -1
    for i, (s_, e_, x_) in enumerate(zip(t["s"].to_numpy(), t["e"].to_numpy(), t["exit_min"].to_numpy())):
        if s_ != last_s:
            last_s, last_x = s_, -1
        if e_ > last_x:
            keep.append(i)
            last_x = x_
    t = t.iloc[keep].reset_index(drop=True)
    t["date"] = g.dates[t["s"].to_numpy()]
    s0 = hm(g.cfg["session"][0])
    t["entry_hhmm"] = [f"{(s0 + m) // 60:02d}:{((s0 + m) % 60) // 30 * 30:02d}" for m in t["e"]]
    return t


# ======================================================================= statistics
def nw_t(x) -> float:
    x = np.asarray(x, float)
    n = len(x)
    if n < 20 or np.allclose(x, 0):
        return float("nan")
    u = x - x.mean()
    L = int(math.floor(4 * (n / 100) ** (2 / 9)))
    s = u @ u / n
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (u[l:] @ u[:-l]) / n
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else float("nan")


def p_one(t: float) -> float:
    return float(0.5 * math.erfc(t / math.sqrt(2))) if np.isfinite(t) else 1.0


def holm(p: dict) -> dict:
    items = sorted(p.items(), key=lambda kv: kv[1])
    out, run, m = {}, 0.0, len(items)
    for k, (key, v) in enumerate(items):
        run = max(run, min(1.0, (m - k) * v))
        out[key] = run
    return out


def period_mask(spec: dict, inst: str, period: str, dates: pd.DatetimeIndex) -> np.ndarray:
    p = spec["periods"][period]
    a, b = p[inst] if period == "PRE" else p
    return np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))


def daily(g: Grid, t: pd.DataFrame, pm: np.ndarray, col: str = "net_usd") -> pd.Series:
    days = g.dates[pm & g.eligible]
    if t.empty:
        return pd.Series(0.0, index=days)
    return t.groupby("date")[col].sum().reindex(days).fillna(0.0)


def metrics(g: Grid, t: pd.DataFrame, pm: np.ndarray) -> dict:
    dd = daily(g, t, pm)
    tt = t[t["date"].isin(dd.index)] if not t.empty else t
    n_days = len(dd)
    out = {"sessions": n_days, "trades": len(tt)}
    if n_days == 0:
        return out
    cum = dd.cumsum()
    out.update(net_usd=dd.sum(), gross_usd=tt["gross_usd"].sum() if len(tt) else 0.0, mean_daily_usd=dd.mean(),
               nw_t=nw_t(dd), sharpe=dd.mean() / dd.std() * math.sqrt(252) if dd.std() > 0 else np.nan,
               max_dd_usd=float((cum - cum.cummax()).min()))
    if len(tt):
        w, l = tt["net_usd"][tt["net_usd"] > 0], tt["net_usd"][tt["net_usd"] <= 0]
        out.update(trades_per_session=len(tt) / n_days, mean_R=tt["R"].mean(), mean_gross_R=tt["gross_R"].mean(),
                   median_R=tt["R"].median(), hit_rate=(tt["net_usd"] > 0).mean(),
                   profit_factor=w.sum() / -l.sum() if l.sum() < 0 else np.nan,
                   exposure_pct=100 * tt["hold_min"].sum() / (n_days * g.M), mean_hold_min=tt["hold_min"].mean(),
                   pct_target=(tt["reason"] == "target").mean(), pct_stop=(tt["reason"] == "stop").mean(),
                   pct_time=(tt["reason"] == "time").mean(), median_mae_R=tt["mae_R"].median(),
                   median_mfe_R=tt["mfe_R"].median(), mean_risk_usd=tt["risk_usd"].mean(),
                   long_net_usd=tt.loc[tt["d"] == 1, "net_usd"].sum(), short_net_usd=tt.loc[tt["d"] == -1, "net_usd"].sum(),
                   long_gross_usd=tt.loc[tt["d"] == 1, "gross_usd"].sum(), short_gross_usd=tt.loc[tt["d"] == -1, "gross_usd"].sum(),
                   long_trades=int((tt["d"] == 1).sum()), short_trades=int((tt["d"] == -1).sum()))
    return out


def cell_list(spec: dict, test: str) -> list[dict]:
    grid = spec["tests"][test]["grid"]
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]


def cell_id(p: dict) -> str:
    return "|".join(f"{k}={v}" for k, v in p.items())


def neighbours(spec: dict, test: str) -> list[dict]:
    grid, cen = spec["tests"][test]["grid"], spec["tests"][test]["central"]
    out = []
    for k, vals in grid.items():
        i = vals.index(cen[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                q = dict(cen)
                q[k] = vals[j]
                out.append(q)
    return out


# ======================================================================= placebo
def placebo(g: Grid, test: str, t: pd.DataFrame, pm: np.ndarray, rng) -> dict:
    """Random session (same period), same half-hour bucket, same side, same fractional geometry."""
    if len(t) < 30:
        return {"placebo_p": np.nan}
    tick = g.cfg["tick_size"]
    frac_stop = np.abs(t["entry"] - t["stop"]).to_numpy() / t["entry"].to_numpy()
    frac_tgt = np.abs(t["target0"] - t["entry"]).to_numpy() / t["entry"].to_numpy()
    d = t["d"].to_numpy()
    nb = g.f["nb"]
    win_ok = np.asarray(window_mask(g, test)).any(0)

    def run(s, e, dd):
        xmax = exit_minute(g, test, e)
        en_raw = g.O[s, np.minimum(e, g.M - 1)]
        tgt = en_raw * (1 + dd * frac_tgt)
        sim = simulate(g, s, e, dd, lambda en: en * (1 - dd * frac_stop), tgt, False, xmax)
        return sim

    real = run(t["s"].to_numpy(), t["e"].to_numpy(), d)
    obs = real["R"].mean()
    opp = run(t["s"].to_numpy(), t["e"].to_numpy(), -d)
    pool = np.flatnonzero(pm & g.eligible)
    bucket = (t["e"].to_numpy() // 30) * 30
    allowed_e = np.arange(nb)[win_ok] * 5 + 5
    sims = np.empty(N_PLACEBO)
    for i in range(N_PLACEBO):
        s_new = rng.choice(pool, size=len(t))
        e_new = np.empty(len(t), int)
        for bk in np.unique(bucket):
            m = bucket == bk
            cand = allowed_e[(allowed_e // 30) * 30 == bk]
            if len(cand) == 0:
                cand = allowed_e
            e_new[m] = rng.choice(cand, size=m.sum())
        sim = run(s_new, e_new, d)
        sims[i] = sim["R"].mean() if len(sim) else np.nan
    sims = sims[np.isfinite(sims)]
    return {"placebo_p": float((np.sum(sims >= obs) + 1) / (len(sims) + 1)), "fixed_geometry_mean_R": float(obs),
            "placebo_mean_R": float(sims.mean()), "opposite_mean_R": float(opp["R"].mean())}


# ======================================================================= regimes
def regime_labels(g: Grid) -> pd.Series:
    if g.cfg["regime"] == "vix":
        v = pd.read_csv(ROOT / "data/raw/vix-cboe.csv")
        v["DATE"] = pd.to_datetime(v["DATE"], format="%m/%d/%Y")
        v = v.set_index("DATE")["CLOSE"].astype(float).sort_index()
        q1, q2 = v.rolling(252).quantile(1 / 3), v.rolling(252).quantile(2 / 3)
        reg = pd.Series(np.where(v <= q1, "low", np.where(v <= q2, "normal", "high")), index=v.index).where(q1.notna())
        m = pd.merge_asof(pd.DataFrame({"d": g.dates}), pd.DataFrame({"d": v.index, "r": reg.to_numpy()}), on="d",
                          allow_exact_matches=False)
        return pd.Series(m["r"].to_numpy(), index=g.dates)
    sd = pd.Series(g.f["sigma_day"], index=g.dates)
    past = sd.shift(1)
    q1, q2 = past.rolling(252, min_periods=120).quantile(1 / 3), past.rolling(252, min_periods=120).quantile(2 / 3)
    return pd.Series(np.where(sd <= q1, "low", np.where(sd <= q2, "normal", "high")), index=g.dates).where(q1.notna())


# ======================================================================= PBO + walk-forward
def pbo(mat: pd.DataFrame, blocks: int = 12) -> float:
    X = mat.to_numpy()
    edges = np.linspace(0, len(X), blocks + 1).astype(int)
    parts = [np.arange(edges[i], edges[i + 1]) for i in range(blocks)]
    logits = []
    for comb in itertools.combinations(range(blocks), blocks // 2):
        tr = np.concatenate([parts[i] for i in comb])
        te = np.concatenate([parts[i] for i in range(blocks) if i not in comb])
        sr_tr = X[tr].mean(0) / (X[tr].std(0) + 1e-12)
        sr_te = X[te].mean(0) / (X[te].std(0) + 1e-12)
        best = int(np.argmax(sr_tr))
        rank = (sr_te < sr_te[best]).sum() + 0.5 * ((sr_te == sr_te[best]).sum() - 1)
        w = (rank + 0.5) / (X.shape[1])
        logits.append(math.log(w / (1 - w)))
    return float(np.mean(np.array(logits) <= 0))


def walk_forward(spec: dict, test: str, mats: pd.DataFrame) -> list[dict]:
    """Expanding by calendar year: pick the cell with the best neighbourhood-smoothed Sharpe, apply next year."""
    cells = cell_list(spec, test)
    ids = [cell_id(p) for p in cells]
    nb_map = {}
    for p in cells:
        grid = spec["tests"][test]["grid"]
        nbs = [p]
        for k, vals in grid.items():
            i = vals.index(p[k])
            for j in (i - 1, i + 1):
                if 0 <= j < len(vals):
                    q = dict(p)
                    q[k] = vals[j]
                    nbs.append(q)
        nb_map[cell_id(p)] = [cell_id(q) for q in nbs]
    cen = cell_id(spec["tests"][test]["central"])
    years = sorted(set(mats.index.year))
    out = []
    for y in years[1:]:
        tr = mats[mats.index.year < y][ids]
        te = mats[mats.index.year == y][ids]
        if len(tr) < 120:
            continue
        sr = tr.mean() / tr.std().replace(0, np.nan)
        smooth = {c: sr[nb_map[c]].mean() for c in ids}
        pick = max(smooth, key=lambda c: (smooth[c] if np.isfinite(smooth[c]) else -1e9))
        out.append({"test": test, "year": y, "picked": pick, "picked_net_usd": te[pick].sum(), "central_net_usd": te[cen].sum()})
    return out


# ======================================================================= IS run
def load_grids(spec: dict, names: list[str]) -> dict:
    CACHE.mkdir(exist_ok=True)
    tag = sha256(Path(__file__))[:12]
    out = {}
    for n in names:
        cfg = spec["instruments"][n]
        path = ROOT / cfg["path"]
        if sha256(path) != cfg["sha256"]:
            raise RuntimeError(f"Input checksum mismatch: {cfg['path']}")
        cp = CACHE / f"{n}_{tag}.pkl"
        if cp.exists():
            out[n] = pickle.loads(cp.read_bytes())
        else:
            out[n] = build_grid(n, cfg)
            cp.write_bytes(pickle.dumps(out[n]))
        print(f"  grid {n}: {len(out[n].dates)} sessions, {int(out[n].eligible.sum())} eligible", flush=True)
    return out


def run_is(spec: dict) -> None:
    rng = np.random.default_rng(SEED)
    grids = load_grids(spec, MAIN + ["NQ_as_MNQ", "CL_as_MCL"])
    OUT.mkdir(exist_ok=True)
    (OUT / "heatmaps").mkdir(exist_ok=True)
    (OUT / "trades").mkdir(exist_ok=True)
    rows, central_trades, daily_mats = [], {}, {}
    plan = [(i, "IS") for i in MAIN] + [("NQ_as_MNQ", "PRE"), ("CL_as_MCL", "PRE")]
    for inst, period in plan:
        g = grids[inst]
        pm = period_mask(spec, inst, period, g.dates)
        mats = {}
        for test in TESTS:
            for p in cell_list(spec, test):
                t = run_cell(g, test, p)
                m = metrics(g, t, pm)
                cid = cell_id(p)
                central = p == spec["tests"][test]["central"]
                rows.append({"instrument": inst, "period": period, "test": test, "cell": cid, "central": central, **p, **m})
                mats[(test, cid)] = daily(g, t, pm)
                if central:
                    tt = t[t["date"].isin(g.dates[pm & g.eligible])] if not t.empty else t
                    central_trades[(inst, period, test)] = tt
                    tt.to_csv(OUT / "trades" / f"{inst}_{period}_{test}_central.csv", index=False)
            print(f"  {inst} {period} {test} done", flush=True)
        daily_mats[(inst, period)] = pd.DataFrame(mats)
    cells = pd.DataFrame(rows)
    cells.to_csv(OUT / "cells_all.csv", index=False)

    # ---------------- central diagnostics and gates (IS)
    gates, plac_rows, cd_rows, ls_rows, yr_rows, reg_rows, hr_rows, dist_rows, wf_rows, pbo_rows, roll_rows = ([] for _ in range(11))
    pvals = {}
    for inst in MAIN:
        for test in TESTS:
            r = cells[(cells["instrument"] == inst) & (cells["period"] == "IS") & (cells["test"] == test) & cells["central"]].iloc[0]
            pvals[(inst, test)] = p_one(r["nw_t"])
    adj = holm(pvals)
    ext_cache = {}
    for inst in MAIN:
        g = grids[inst]
        pm = period_mask(spec, inst, "IS", g.dates)
        reg = regime_labels(g)
        dm = daily_mats[(inst, "IS")]
        pbo_rows.append({"instrument": inst, "cells": dm.shape[1], "pbo": pbo(dm)})
        for test in TESTS:
            cen = spec["tests"][test]["central"]
            t = central_trades[(inst, "IS", test)]
            sub = cells[(cells["instrument"] == inst) & (cells["period"] == "IS") & (cells["test"] == test)]
            r = sub[sub["central"]].iloc[0]
            # G2 region
            nbr_ids = [cell_id(q) for q in neighbours(spec, test)]
            pos_share = (sub["net_usd"] > 0).mean()
            nbr_pos = bool((sub[sub["cell"].isin(nbr_ids)]["net_usd"] > 0).all())
            # G3 cost x2 and delay
            m2 = metrics(g, run_cell(g, test, cen, cost_mult=2.0), pm)
            md = metrics(g, run_cell(g, test, cen, delay=1), pm)
            cd_rows.append({"instrument": inst, "test": test, "base_net_usd": r["net_usd"], "cost_x2_net_usd": m2.get("net_usd"),
                            "delay_1min_net_usd": md.get("net_usd"), "base_t": r["nw_t"], "cost_x2_t": m2.get("nw_t"), "delay_t": md.get("nw_t")})
            # G4 years
            dd = daily(g, t, pm)
            yr = dd.groupby(dd.index.year).sum()
            for y, v in yr.items():
                yt = t[t["date"].dt.year == y] if len(t) else t
                yr_rows.append({"instrument": inst, "test": test, "year": y, "net_usd": v, "trades": len(yt),
                                "mean_R": yt["R"].mean() if len(yt) else np.nan})
            tot = yr.sum()
            g4 = bool((yr > 0).sum() > len(yr) / 2 and tot > 0 and yr.max() <= 0.5 * tot)
            # G5 long/short
            ln, sn = r.get("long_net_usd", np.nan), r.get("short_net_usd", np.nan)
            lg, sg = r.get("long_gross_usd", np.nan), r.get("short_gross_usd", np.nan)
            g5 = bool((ln > 0 and sn > 0) or (ln > 0 and sg >= 0) or (sn > 0 and lg >= 0))
            for side, sd in [("long", 1), ("short", -1)]:
                tt = t[t["d"] == sd] if len(t) else t
                ls_rows.append({"instrument": inst, "test": test, "side": side, "trades": len(tt),
                                "net_usd": tt["net_usd"].sum() if len(tt) else 0, "gross_usd": tt["gross_usd"].sum() if len(tt) else 0,
                                "mean_R": tt["R"].mean() if len(tt) else np.nan, "hit_rate": (tt["net_usd"] > 0).mean() if len(tt) else np.nan})
            # regimes, hours, distribution
            if len(t):
                lab = reg.reindex(t["date"]).to_numpy()
                for rg in ["low", "normal", "high"]:
                    tt = t[lab == rg]
                    reg_rows.append({"instrument": inst, "test": test, "regime": rg, "trades": len(tt), "net_usd": tt["net_usd"].sum(),
                                     "mean_R": tt["R"].mean() if len(tt) else np.nan})
                for h, tt in t.groupby("entry_hhmm"):
                    hr_rows.append({"instrument": inst, "test": test, "entry_half_hour": h, "trades": len(tt), "net_usd": tt["net_usd"].sum(),
                                    "mean_R": tt["R"].mean(), "hit_rate": (tt["net_usd"] > 0).mean()})
                q = t["R"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
                dist_rows.append({"instrument": inst, "test": test, **{f"R_q{int(k * 100):02d}": v for k, v in q.items()},
                                  "R_skew": t["R"].skew(), "worst_usd": t["net_usd"].min(), "best_usd": t["net_usd"].max(),
                                  "mae_R_q50": t["mae_R"].median(), "mae_R_q90": t["mae_R"].quantile(0.1),
                                  "mfe_R_q50": t["mfe_R"].median(), "mfe_R_q90": t["mfe_R"].quantile(0.9)})
            # rolling 252 stability
            rs = dd.rolling(252).mean().dropna()
            roll_rows.append({"instrument": inst, "test": test, "windows": len(rs), "share_positive": float((rs > 0).mean()) if len(rs) else np.nan})
            # G6 placebo
            pl = placebo(g, test, t, pm, rng)
            plac_rows.append({"instrument": inst, "test": test, **pl})
            g6 = bool(pl.get("placebo_p", 1) <= 0.05 and pl.get("opposite_mean_R", 1) < 0)
            # G7 PRE / EXT
            proxy = spec["instruments"][inst]["pre_proxy"]
            if proxy:
                pr = cells[(cells["instrument"] == proxy) & (cells["period"] == "PRE") & (cells["test"] == test) & cells["central"]].iloc[0]
                g7 = bool(pr["net_usd"] > 0 and pr["nw_t"] >= 1.0)
                g7_detail = f"{proxy} PRE net {pr['net_usd']:.0f} t {pr['nw_t']:.2f}"
            else:
                g7, g7_detail = None, "EXT read only if G1-G6 pass"
            wf_rows += walk_forward(spec, test, pd.DataFrame({c[1]: dm[c] for c in dm.columns if c[0] == test}))
            gate = {"instrument": inst, "test": test, "trades": r["trades"], "net_usd": r["net_usd"], "nw_t": r["nw_t"],
                    "mean_R": r.get("mean_R"), "holm_p": adj[(inst, test)],
                    "G1": bool(r["net_usd"] > 0 and r["nw_t"] >= 2.0 and adj[(inst, test)] <= 0.05),
                    "G2": bool(pos_share >= 2 / 3 and nbr_pos), "G2_positive_share": pos_share,
                    "G3": bool((m2.get("net_usd", -1) or -1) > 0 and (md.get("net_usd", -1) or -1) > 0),
                    "G4": g4, "G5": g5, "G6": g6, "G6_p": pl.get("placebo_p"), "G7": g7, "G7_detail": g7_detail}
            if inst == "MGC" and all(gate[k] for k in ["G1", "G2", "G3", "G4", "G5", "G6"]):
                pe = period_mask(spec, inst, "EXT", g.dates)
                me = metrics(g, run_cell(g, test, cen), pe)
                gate["G7"], gate["G7_detail"] = bool(me.get("net_usd", -1) > 0), f"EXT net {me.get('net_usd', 0):.0f}"
            gate["pass_IS_gate"] = all(bool(gate[k]) for k in ["G1", "G2", "G3", "G4", "G5", "G6", "G7"])
            gates.append(gate)
            print(f"  gate {inst} {test}: {'PASS' if gate['pass_IS_gate'] else 'fail'}", flush=True)
    pd.DataFrame(gates).to_csv(OUT / "gates_IS.csv", index=False)
    for name, rr in [("placebo_IS", plac_rows), ("cost_delay_IS", cd_rows), ("long_short_IS", ls_rows), ("years_IS", yr_rows),
                     ("regimes_IS", reg_rows), ("entry_hour_IS", hr_rows), ("distribution_mae_mfe_IS", dist_rows),
                     ("walk_forward_IS", wf_rows), ("pbo_IS", pbo_rows), ("rolling252_IS", roll_rows)]:
        pd.DataFrame(rr).to_csv(OUT / f"{name}.csv", index=False)
    heatmaps(spec, cells)
    status = {"any_pass": any(g_["pass_IS_gate"] for g_ in gates),
              "passing": [(g_["instrument"], g_["test"]) for g_ in gates if g_["pass_IS_gate"]],
              "mes_stage2": any(g_["pass_IS_gate"] for g_ in gates if g_["instrument"] in ("MNQ", "MGC"))}
    (OUT / "IS_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=1))


def heatmaps(spec: dict, cells: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for (inst, period), sub in cells.groupby(["instrument", "period"]):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, test in zip(axes.ravel(), TESTS):
            grid = spec["tests"][test]["grid"]
            k1, k2 = list(grid)
            s = sub[sub["test"] == test]
            Z = np.full((len(grid[k1]), len(grid[k2])), np.nan)
            for i, a in enumerate(grid[k1]):
                for j, b in enumerate(grid[k2]):
                    row = s[s["cell"] == cell_id({k1: a, k2: b})]
                    if len(row):
                        Z[i, j] = row["sharpe"].iloc[0]
            lim = np.nanmax(np.abs(Z)) if np.isfinite(Z).any() else 1
            ax.imshow(Z, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    row = s[s["cell"] == cell_id({k1: grid[k1][i], k2: grid[k2][j]})]
                    if len(row):
                        ax.text(j, i, f"{Z[i, j]:.2f}\n{int(row['trades'].iloc[0])} tr\n{row['net_usd'].iloc[0]:.0f}$", ha="center", va="center", fontsize=7)
            ax.set_xticks(range(len(grid[k2])), [str(v) for v in grid[k2]])
            ax.set_yticks(range(len(grid[k1])), [str(v) for v in grid[k1]])
            ax.set_xlabel(k2)
            ax.set_ylabel(k1)
            cen = spec["tests"][test]["central"]
            ax.add_patch(plt.Rectangle((grid[k2].index(cen[k2]) - 0.5, grid[k1].index(cen[k1]) - 0.5), 1, 1, fill=False, lw=2.5, ec="k"))
            ax.set_title(f"{test} — daily Sharpe (net)")
        fig.suptitle(f"R1 {inst} {period}: net Sharpe, trades, net USD per cell (black box = central cell)")
        fig.tight_layout()
        fig.savefig(OUT / "heatmaps" / f"{inst}_{period}.png", dpi=110)
        plt.close(fig)


# ======================================================================= self-test
def selftest() -> None:
    rng = np.random.default_rng(3)
    cfg = {"session": ["09:30", "16:00"], "forced_exit": "15:55", "lunch_entry": ["11:30", "13:30"], "lunch_exit": "14:00",
           "pre_roll_excluded_sessions": 4, "point_value": 2.0, "tick_size": 0.25, "commission_per_side": 0.0, "regime": "sigma_day"}

    def make(kappa: float) -> pd.DataFrame:
        chunks, p = [], 10000.0
        for d in pd.bdate_range("2021-01-04", periods=260):
            mins = pd.date_range(d + pd.Timedelta(hours=9, minutes=30), periods=390, freq="1min")
            x, mu, px = p, p, []
            for _ in range(390):
                x = x + kappa * (mu - x) + rng.normal(0, 3.0)
                px.append(x)
            px = np.round(np.array(px) / 0.25) * 0.25
            o = np.concatenate([[p], px[:-1]])
            chunks.append(pd.DataFrame({"open": o, "high": np.maximum(o, px) + 0.25, "low": np.minimum(o, px) - 0.25, "close": px,
                                        "volume": 100.0, "instrument_id": 1.0}, index=mins.tz_localize("America/New_York")))
            p = px[-1]
        return pd.concat(chunks)

    p1 = {"k": 2.0, "er_max": None}
    g_rw = build_grid("rw", cfg, make(0.0))
    g_ou = build_grid("ou", cfg, make(0.05))
    t_rw, t_ou = run_cell(g_rw, "H1_N1", p1), run_cell(g_ou, "H1_N1", p1)
    print(f"  random walk: {len(t_rw)} trades, mean gross R {t_rw['gross_R'].mean():.3f}")
    print(f"  mean-reverting: {len(t_ou)} trades, mean gross R {t_ou['gross_R'].mean():.3f}")
    assert len(t_ou) > 200 and t_ou["gross_R"].mean() > 0.1
    assert abs(t_rw["gross_R"].mean()) < 0.1
    # causality: perturb minutes >= m0 of one session S; signals of earlier sessions and of S before m0 must not change
    m0, S = 200, 150
    df = make(0.05)
    g1 = build_grid("a", cfg, df)
    df2 = df.copy()
    loc = df2.index.tz_localize(None)
    late = ((loc.hour * 60 + loc.minute - 570) >= m0) & (loc.normalize() == g1.dates[S])
    for c in ["open", "high", "low", "close"]:
        df2.loc[late, c] = df2.loc[late, c] * rng.uniform(0.98, 1.02, late.sum())
    df2.loc[late, "volume"] = rng.uniform(1, 500, late.sum())
    g2 = build_grid("b", cfg, df2)
    n_checked = 0
    for test in TESTS:
        p = spec_central(test)
        c1, c2 = candidates(g1, test, p), candidates(g2, test, p)
        k1 = {(s_, b, d) for s_, b, d, e in zip(c1["s"], c1["b"], c1["d"], c1["e"]) if s_ < S or (s_ == S and e <= m0)}
        k2 = {(s_, b, d) for s_, b, d, e in zip(c2["s"], c2["b"], c2["d"], c2["e"]) if s_ < S or (s_ == S and e <= m0)}
        assert k1 == k2, test
        n_checked += len(k1)
    assert n_checked > 100
    # stop before target in an ambiguous minute; fills
    g = g1
    s, e = np.array([30]), np.array([100])
    en = g.O[30, 100] + 0.25
    g.H[30, 100], g.L[30, 100] = en + 50, en - 50
    t = simulate(g, s, e, np.array([1]), lambda x: x - 10, np.array([en + 10]), False, np.array([160]))
    assert t["reason"].iloc[0] == "stop" and abs(t["exit"].iloc[0] - (min(g.O[30, 100], en - 10) - 0.25)) < 1e-9
    print("selftest OK: random walk gross R ~ 0, OU gross R > 0, signals causal on all six tests, stop-first and fills checked")


def spec_central(test: str) -> dict:
    return json.loads((HERE / "FROZEN_SPECIFICATION.json").read_text())["tests"][test]["central"]


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "is"
    if cmd == "selftest":
        selftest()
        return
    spec = read_spec()
    if cmd == "is":
        run_is(spec)
    elif cmd == "oos":
        st = json.loads((OUT / "IS_status.json").read_text())
        if not st["any_pass"] or not (HERE / "OOS_APPROVED.json").exists():
            raise SystemExit("OOS refused: no central cell passed the IS gate, or no user approval recorded")
        raise SystemExit("OOS runner to be added only after approval")


if __name__ == "__main__":
    main()
