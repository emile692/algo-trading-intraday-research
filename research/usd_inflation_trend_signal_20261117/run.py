"""USD Inflation strategy: test of a second 'Inflation Trend' signal (MA crossover on the QuantCube US CPI nowcast).

Protocol frozen in research_design.md. Run with an env that has pandas, numpy, openpyxl:
    uv run --no-project --with pandas --with openpyxl python -I run.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

XLSX = Path(r"C:\Data_local\excel_repo\FX_USD_Inflation.xlsx")
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

COST = 0.0002
ANN = 252
LAGS = (1, 2, 3)
A_WINDOWS = range(15, 61)
B_SHORT = (10, 20, 30)
B_LONG = (60, 90, 120, 180)
GRID_SHORT = (1, 5, 10, 20, 30, 45, 60)
GRID_LONG = (60, 90, 120, 180, 250, 365)
IS_START_EXCEL = pd.Timestamp("2016-02-24")
IS_START = pd.Timestamp("2016-09-01")
GRID_START = pd.Timestamp("2017-03-01")
IS_END = pd.Timestamp("2021-12-31")
OOS_START = pd.Timestamp("2022-01-01")
BLOCK = 20
N_BOOT = 2000
CSCV_BLOCKS = 10


# ---------------------------------------------------------------- data
def load():
    qc = pd.read_excel(XLSX, sheet_name="QuantCube Data", usecols="A:C")
    qc = qc[qc["iso2"] == "US"].dropna(subset=["date"])
    qc["date"] = pd.to_datetime(qc["date"])
    qc = qc.groupby("date")["value"].sum()  # SUMIFS in the workbook
    cal = pd.date_range(qc.index.min(), qc.index.max(), freq="D")
    yoy = qc.reindex(cal)
    mkt = pd.read_excel(XLSX, sheet_name="Market Data", usecols="A:B").dropna(subset=["date"])
    mkt["date"] = pd.to_datetime(mkt["date"])
    ret = mkt.set_index("date")["DX1 Curncy"].astype(float)
    return yoy, ret


def lagged_dates(tdates: pd.DatetimeIndex, lag: int) -> pd.Series:
    """Trading date `lag` rows earlier (NaT when unavailable)."""
    return pd.Series(tdates, index=tdates).shift(lag)


def at(series: pd.Series, dates: pd.Series) -> np.ndarray:
    return series.reindex(pd.DatetimeIndex(dates)).to_numpy(dtype=float)


def ensemble(sub_signals: list[np.ndarray], index) -> pd.Series:
    m = np.vstack(sub_signals)
    ok = ~np.isnan(m)
    n = ok.sum(axis=0)
    s = np.where(ok, np.sign(np.nan_to_num(m)), 0.0).sum(axis=0)
    out = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return pd.Series(out, index=index)


def signal_accel(yoy, tdates, windows=A_WINDOWS, lags=LAGS):
    subs = []
    for lag in lags:
        td = lagged_dates(tdates, lag)
        cur = at(yoy, td)
        for w in windows:
            prv = at(yoy, td - pd.Timedelta(days=w))
            subs.append(cur - prv)
    sig = ensemble(subs, tdates)
    sig.iloc[: max(lags)] = np.nan  # ROW()-5 <= max lag -> ""
    return sig


def signal_trend(yoy, tdates, shorts, longs, lags=LAGS):
    mas = {k: yoy.rolling(k, min_periods=k).mean() for k in set(shorts) | set(longs)}
    subs = []
    for lag in lags:
        td = lagged_dates(tdates, lag)
        for s, l in itertools.product(shorts, longs):
            if s >= l:
                continue
            subs.append(at(mas[s], td) - at(mas[l], td))
    sig = ensemble(subs, tdates)
    sig.iloc[: max(lags)] = np.nan
    return sig


# ---------------------------------------------------------------- P&L and stats
def pnl(sig: pd.Series, ret: pd.Series) -> pd.Series:
    """Workbook formula: E(t) = C(t)*D(t-1) - |D(t-1)-D(t-2)|*cost."""
    p1, p2 = sig.shift(1), sig.shift(2)
    out = ret * p1 - (p1 - p2).abs() * COST
    return out[p1.notna() & p2.notna()]


def window(x: pd.Series, start, end=None):
    x = x[x.index >= start]
    return x if end is None else x[x.index <= end]


def stats(x: pd.Series, sig: pd.Series | None = None) -> dict:
    x = x.dropna()
    eq = (1 + x).cumprod()
    dd = eq / eq.cummax() - 1
    ar, vol = x.mean() * ANN, x.std(ddof=1) * np.sqrt(ANN)
    d = {
        "start": str(x.index.min().date()), "end": str(x.index.max().date()), "n": int(len(x)),
        "ann_ret_pct": 100 * ar, "ann_vol_pct": 100 * vol, "sharpe": ar / vol if vol > 0 else np.nan,
        "max_dd_pct": 100 * dd.min(),
        "hit_pct": 100 * (x > 0).sum() / max((x != 0).sum(), 1),
    }
    if sig is not None:
        s = sig.reindex(x.index)
        d["turnover_pct"] = 100 * s.diff().abs().mean() * ANN
        d["avg_abs_pos"] = float(s.abs().mean())
    return d


def sharpe(x) -> float:
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return x.mean() / sd * np.sqrt(ANN) if sd > 0 else np.nan


def nw_alpha(y: pd.Series, x: pd.Series, lags=10):
    df = pd.concat([y, x], axis=1).dropna()
    Y, X = df.iloc[:, 0].to_numpy(), np.column_stack([np.ones(len(df)), df.iloc[:, 1].to_numpy()])
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    e = Y - X @ beta
    xe = X * e[:, None]
    S = xe.T @ xe
    for k in range(1, lags + 1):
        g = xe[k:].T @ xe[:-k]
        S += (1 - k / (lags + 1)) * (g + g.T)
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ S @ XtXi
    return {"alpha_ann_pct": 100 * beta[0] * ANN, "beta": beta[1], "t_alpha_nw": beta[0] / np.sqrt(V[0, 0])}


def block_boot_dsharpe(a: pd.Series, c: pd.Series, rng) -> dict:
    df = pd.concat([a, c], axis=1).dropna().to_numpy()
    n = len(df)
    nb = int(np.ceil(n / BLOCK))
    ds = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK, nb)
        idx = (starts[:, None] + np.arange(BLOCK)).ravel()[:n]
        s = df[idx]
        ds[i] = sharpe(s[:, 1]) - sharpe(s[:, 0])
    return {"p_dsharpe_gt0": float((ds > 0).mean()), "q05": float(np.quantile(ds, 0.05)), "q95": float(np.quantile(ds, 0.95))}


def cscv_pbo(M: pd.DataFrame, S=CSCV_BLOCKS) -> dict:
    """Bailey et al. CSCV: M = T x N matrix of daily returns, one column per candidate."""
    M = M.dropna()
    blocks = np.array_split(np.arange(len(M)), S)
    X = M.to_numpy()
    logits, best_names = [], []
    for comb in itertools.combinations(range(S), S // 2):
        tr = np.concatenate([blocks[i] for i in comb])
        te = np.concatenate([blocks[i] for i in range(S) if i not in comb])
        sr_tr = np.array([sharpe(X[tr, j]) for j in range(X.shape[1])])
        sr_te = np.array([sharpe(X[te, j]) for j in range(X.shape[1])])
        j = int(np.nanargmax(sr_tr))
        rank = (sr_te < sr_te[j]).sum() + 0.5 * ((sr_te == sr_te[j]).sum() - 1)
        w = (rank + 0.5) / len(sr_te)  # relative rank in (0,1)
        logits.append(np.log(w / (1 - w)))
        best_names.append(M.columns[j])
    logits = np.array(logits)
    vc = pd.Series(best_names).value_counts()
    return {"pbo": float((logits <= 0).mean()), "n_splits": len(logits), "n_candidates": M.shape[1],
            "most_selected": vc.head(5).to_dict()}


# ---------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(20261117)
    yoy, ret = load()
    tdates = ret.index
    report: dict = {"input_sha256": hashlib.sha256(XLSX.read_bytes()).hexdigest(),
                    "nowcast_range": [str(yoy.index.min().date()), str(yoy.index.max().date())],
                    "nowcast_missing_days": int(yoy.isna().sum()),
                    "dx1_range": [str(tdates.min().date()), str(tdates.max().date())]}

    # --- signals
    A = signal_accel(yoy, tdates)
    B = signal_trend(yoy, tdates, B_SHORT, B_LONG)
    C = 0.5 * A + 0.5 * B
    C = C.where(B.notna())  # combination defined only where B exists

    # --- replication check vs workbook cached values
    cached = pd.read_excel(XLSX, sheet_name="Main", header=None, skiprows=45, usecols="B:E",
                           names=["date", "ret", "signal", "strat"]).dropna(subset=["date"])
    cached["date"] = pd.to_datetime(cached["date"])
    cached = cached.set_index("date")
    sig_diff = (A.reindex(cached.index) - cached["signal"].astype(float)).abs()
    pa_full = pnl(A, ret)
    pnl_diff = (pa_full.reindex(cached.index) - cached["strat"].astype(float)).abs()
    report["replication"] = {"max_abs_signal_diff": float(sig_diff.max()),
                             "max_abs_pnl_diff": float(pnl_diff.max()),
                             "A_is_excel_window": stats(window(pa_full, IS_START_EXCEL, IS_END), A),
                             "A_overall_excel_window": stats(window(pa_full, IS_START_EXCEL), A)}

    pA, pB, pC = pnl(A, ret), pnl(B, ret), pnl(C, ret)
    isA, isB, isC = (window(p, IS_START, IS_END) for p in (pA, pB, pC))

    # --- IS diagnostics
    sA, sB = window(A, IS_START, IS_END), window(B, IS_START, IS_END)
    rep_is = {
        "A": stats(isA, A), "B_ens": stats(isB, B), "C_50_50": stats(isC, C),
        "corr_signal_AB": float(sA.corr(sB)),
        "same_sign_share": float((np.sign(sA) == np.sign(sB)).mean()),
        "corr_pnl_AB": float(isA.corr(isB)),
        "alpha_B_on_A": nw_alpha(isB, isA),
        "boot_dsharpe_C_minus_A": block_boot_dsharpe(isA, isC, rng),
        "mix_weight_on_A": {f"{w:.2f}": sharpe(window(pnl((w * A + (1 - w) * B).where(B.notna()), ret), IS_START, IS_END))
                            for w in (0.0, 0.25, 0.5, 0.75, 1.0)},
        "by_year_sharpe": {
            str(y): {"A": sharpe(isA[isA.index.year == y]), "B": sharpe(isB[isB.index.year == y]),
                     "C": sharpe(isC[isC.index.year == y])}
            for y in sorted(set(isA.index.year))},
    }
    rep_is["dsharpe_C_minus_A"] = rep_is["C_50_50"]["sharpe"] - rep_is["A"]["sharpe"]
    report["is"] = rep_is

    # --- plateau grid (common window 2017-03 -> 2021-12)
    gA = window(pA, GRID_START, IS_END)
    sharpe_A_grid = sharpe(gA)
    rows, cands = [], {"A_only": gA}
    for s, l in itertools.product(GRID_SHORT, GRID_LONG):
        if s >= l:
            continue
        b = signal_trend(yoy, tdates, (s,), (l,))
        c = (0.5 * A + 0.5 * b).where(b.notna())
        pb, pc = window(pnl(b, ret), GRID_START, IS_END), window(pnl(c, ret), GRID_START, IS_END)
        rows.append({"s": s, "l": l, "sharpe_B": sharpe(pb), "sharpe_C": sharpe(pc),
                     "dsharpe": sharpe(pc) - sharpe_A_grid, "corr_sig_AB": float(window(A, GRID_START, IS_END).corr(window(b, GRID_START, IS_END)))})
        cands[f"C_{s}_{l}"] = pc
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "is_grid.csv", index=False)
    share_pos = float((grid["dsharpe"] > 0).mean())
    pbo = cscv_pbo(pd.DataFrame(cands))
    report["grid"] = {"window": [str(GRID_START.date()), str(IS_END.date())], "sharpe_A": sharpe_A_grid,
                      "share_dsharpe_pos": share_pos, "median_dsharpe": float(grid["dsharpe"].median()),
                      "median_sharpe_B": float(grid["sharpe_B"].median()), "cscv": pbo}

    # --- gates
    gates = {
        "G1_B_sharpe_pos": rep_is["B_ens"]["sharpe"] > 0,
        "G2_dsharpe_pos": rep_is["dsharpe_C_minus_A"] > 0,
        "G3_plateau_60pct": share_pos >= 0.60,
        "G4_pbo_lt_half": pbo["pbo"] < 0.5,
    }
    report["gates"] = {k: bool(v) for k, v in gates.items()}
    passed = all(gates.values())
    report["is_verdict"] = "PASS_IS" if passed else "CLOSED_IS"

    # --- OOS: computed, sealed unless the IS gates pass
    oos = {"A": stats(window(pA, OOS_START), A), "B_ens": stats(window(pB, OOS_START), B),
           "C_50_50": stats(window(pC, OOS_START), C)}
    oos["dsharpe_C_minus_A"] = oos["C_50_50"]["sharpe"] - oos["A"]["sharpe"]
    if passed:
        report["oos"] = oos
        report["verdict"] = "ADOPT_COMPLEMENT" if (oos["dsharpe_C_minus_A"] >= 0 and oos["B_ens"]["sharpe"] > 0) else "CLOSED_OOS"
    else:
        (OUT / "oos_sealed.json").write_text(json.dumps(oos, indent=2, default=float))
        report["verdict"] = "CLOSED_IS"

    # --- series for charts
    pd.DataFrame({"A": A, "B_ens": B, "C": C, "ret": ret}).to_csv(OUT / "signals.csv")
    pd.DataFrame({"A": pA, "B_ens": pB, "C": pC}).to_csv(OUT / "pnl.csv")
    pd.DataFrame({"yoy": yoy, **{f"ma{k}": yoy.rolling(k, min_periods=k).mean() for k in (10, 20, 30, 60, 90, 120, 180)}}).to_csv(OUT / "nowcast_ma.csv")
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
