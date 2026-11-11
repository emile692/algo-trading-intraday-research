"""MES dealer-gamma regime, phase 1 mechanism test.

Runs exactly FROZEN_SPECIFICATION.json (hash-checked against SPEC_LOCK.json).
IS is evaluated first; OOS is computed only if every IS gate passes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
NW_LAGS = 5
POINT_VALUE = 5.0
COST_RT = 5.0


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
    for name, item in spec["data"].items():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Input checksum mismatch: {name}")
    return spec


# ---------------------------------------------------------------- statistics
def ols_hac(y: np.ndarray, X: np.ndarray, lags: int = NW_LAGS) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West (Bartlett) covariance."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = Xu[lag:].T @ Xu[:-lag]
        S += w * (G + G.T)
    return beta, XtX_inv @ S @ XtX_inv


def contrast(beta: np.ndarray, cov: np.ndarray, c: np.ndarray) -> tuple[float, float, float]:
    est = float(c @ beta)
    se = float(np.sqrt(c @ cov @ c))
    return est, se, est / se


def mean_hac(x: np.ndarray) -> tuple[float, float]:
    b, cov = ols_hac(x, np.ones((len(x), 1)))
    return float(b[0]), float(b[0] / np.sqrt(cov[0, 0]))


# ---------------------------------------------------------------- data
def session_table(spec: dict) -> pd.DataFrame:
    x = pd.read_parquet(ROOT / spec["data"]["mes"]["path"])
    x.index = pd.to_datetime(x.index).tz_convert("America/New_York")
    x = x[["open", "high", "low", "close"]].astype(float).sort_index()
    hm = x.index.hour * 60 + x.index.minute
    x = x[(hm >= 570) & (hm < 960) & (x.index.dayofweek < 5)]
    rows = []
    for day, g in x.groupby(x.index.date):
        m = g.index.hour * 60 + g.index.minute
        g = g.set_axis(m)
        if len(g) < 370 or not {570, 629, 959}.issubset(g.index):
            continue
        o930, c1029, c1559 = g.at[570, "open"], g.at[629, "close"], g.at[959, "close"]
        grid = g["close"].groupby((g.index - 570) // 5).last()
        path = np.log(np.concatenate([[o930], grid.to_numpy()]))
        r5 = np.diff(path)
        rows.append({
            "session_date": pd.Timestamp(day),
            "o930": o930, "c1029": c1029, "c1559": c1559,
            "r_am": np.log(c1029 / o930), "r_pm": np.log(c1559 / c1029),
            "rv": float(np.sum(r5 ** 2)),
            "efficiency": abs(np.log(c1559 / o930)) / float(np.sum(np.abs(r5))),
        })
    s = pd.DataFrame(rows).set_index("session_date").sort_index()
    s["sigma_scale"] = np.sqrt(s.rv.shift(1).rolling(20, min_periods=20).mean())
    s["z_am"] = s.r_am / s.sigma_scale
    s["z_pm"] = s.r_pm / s.sigma_scale
    s["ln_rv"] = np.log(s.rv)
    s["ln_rv_prev"] = s.ln_rv.shift(1)
    s["ln_rv_5"] = np.log(s.rv.shift(1).rolling(5, min_periods=5).mean())
    s["eff_20"] = s.efficiency.shift(1).rolling(20, min_periods=20).mean()
    return s


def attach_gex(s: pd.DataFrame, spec: dict) -> pd.DataFrame:
    g = pd.read_csv(ROOT / spec["data"]["gex"]["path"], parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    g["pct"] = g.gex.rolling(252, min_periods=252).apply(lambda w: (w <= w[-1]).sum() / 252.0, raw=True)
    pos = np.searchsorted(g.date.to_numpy(), s.index.to_numpy(), side="left") - 1  # latest row strictly before
    out = s.copy()
    for tag, shift in (("lag1", 0), ("lag2", 1)):
        p = pos - shift
        valid = p >= 0
        pct = np.full(len(out), np.nan)
        pct[valid] = g.pct.to_numpy()[p[valid]]
        out[f"pct_{tag}"] = pct
        out[f"regime_{tag}"] = np.where(np.isnan(pct), None,
                                        np.where(pct <= 1 / 3, "LOW", np.where(pct > 2 / 3, "HIGH", "MID")))
    v = pd.read_csv(ROOT / spec["data"]["vix"]["path"])
    v["date"] = pd.to_datetime(v.DATE, format="%m/%d/%Y")
    v = v.sort_values("date")
    vp = np.searchsorted(v.date.to_numpy(), out.index.to_numpy(), side="left") - 1
    vix = np.where(vp >= 0, v.CLOSE.to_numpy()[np.clip(vp, 0, None)], np.nan)
    out["ln_vix_prev"] = np.log(vix)
    return out


# ---------------------------------------------------------------- tests
REG = ["LOW", "MID", "HIGH"]


def h2(frame: pd.DataFrame, regime_col: str) -> dict:
    f = frame.dropna(subset=["z_am", "z_pm", regime_col])
    D = np.column_stack([(f[regime_col] == k).to_numpy(float) for k in REG])
    X = np.column_stack([D, D * f.z_am.to_numpy()[:, None]])
    beta, cov = ols_hac(f.z_pm.to_numpy(), X)
    est, se, t = contrast(beta, cov, np.array([0, 0, 0, 1, 0, -1.0]))
    slopes = {f"b_{k}": float(beta[3 + i]) for i, k in enumerate(REG)}
    slope_t = {f"t_b_{k}": float(beta[3 + i] / np.sqrt(cov[3 + i, 3 + i])) for i, k in enumerate(REG)}
    counts = {f"n_{k}": int((f[regime_col] == k).sum()) for k in REG}
    return {"n": len(f), **counts, **slopes, **slope_t, "b_LOW_minus_b_HIGH": est, "se": se, "t": t}


def regime_regression(frame: pd.DataFrame, y: str, controls: list[str], regime_col: str = "regime_lag1") -> dict:
    f = frame.dropna(subset=[y, regime_col, *controls])
    X = np.column_stack([np.ones(len(f)), (f[regime_col] == "LOW").to_numpy(float),
                         (f[regime_col] == "HIGH").to_numpy(float), f[controls].to_numpy()])
    beta, cov = ols_hac(f[y].to_numpy(), X)
    est, se, t = contrast(beta, cov, np.r_[0, 1, -1, np.zeros(len(controls))])
    return {"n": len(f), "c_LOW": float(beta[1]), "t_c_LOW": float(beta[1] / np.sqrt(cov[1, 1])),
            "c_HIGH": float(beta[2]), "t_c_HIGH": float(beta[2] / np.sqrt(cov[2, 2])),
            "c_LOW_minus_c_HIGH": est, "se": se, "t": t}


def canonical_rule(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    f = frame.dropna(subset=["regime_lag1"]).copy()
    sign_am = np.sign(f.c1029 - f.o930)
    direction = np.where(f.regime_lag1 == "LOW", sign_am, np.where(f.regime_lag1 == "HIGH", -sign_am, 0.0))
    f["direction"] = direction
    t = f[f.direction != 0].copy()
    t["gross_usd"] = t.direction * (t.c1559 - t.c1029) * POINT_VALUE
    t["net_usd"] = t.gross_usd - COST_RT
    rows = {}
    for label, sub in (("ALL", t), ("LOW_follow", t[t.regime_lag1 == "LOW"]), ("HIGH_fade", t[t.regime_lag1 == "HIGH"])):
        m, tt = mean_hac(sub.net_usd.to_numpy()) if len(sub) > NW_LAGS + 2 else (np.nan, np.nan)
        rows[label] = {"trades": len(sub), "mean_gross_usd": float(sub.gross_usd.mean()),
                       "mean_net_usd": m, "t_net": tt, "total_net_usd": float(sub.net_usd.sum()),
                       "win_rate": float((sub.net_usd > 0).mean())}
    yearly = t.groupby(t.index.year).agg(trades=("net_usd", "size"), net_usd=("net_usd", "sum"),
                                         mean_net_usd=("net_usd", "mean"))
    return t, rows, yearly


def evaluate(frame: pd.DataFrame) -> dict:
    res = {
        "H2_primary": h2(frame, "regime_lag1"),
        "H2_lag2": h2(frame, "regime_lag2"),
        "H2b_efficiency": regime_regression(frame, "efficiency", ["ln_vix_prev", "eff_20"]),
        "H1_volatility": regime_regression(frame, "ln_rv", ["ln_rv_prev", "ln_rv_5", "ln_vix_prev"]),
    }
    trades, rule, yearly = canonical_rule(frame)
    res["canonical_rule"] = rule
    gate = {
        "H2_primary": res["H2_primary"]["b_LOW_minus_b_HIGH"] > 0 and res["H2_primary"]["t"] >= 1.645,
        "H2_lag2": res["H2_lag2"]["b_LOW_minus_b_HIGH"] > 0 and res["H2_lag2"]["t"] >= 1.0,
        "H2b_efficiency": res["H2b_efficiency"]["c_LOW_minus_c_HIGH"] > 0,
        "canonical_rule": rule["ALL"]["mean_net_usd"] > 0,
    }
    res["gate"] = {**gate, "PASS": all(gate.values())}
    return res, trades, yearly


def self_test() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    e = np.zeros(n)
    eps = rng.normal(size=n)
    for i in range(1, n):
        e[i] = 0.5 * e[i - 1] + eps[i]
    y = 1.0 + 0.3 * x + e
    beta, cov = ols_hac(y, np.column_stack([np.ones(n), x]))
    assert abs(beta[1] - 0.3) < 0.05
    naive = np.sqrt(np.var(y - np.column_stack([np.ones(n), x]) @ beta) / n)
    assert np.sqrt(cov[0, 0]) > 1.4 * naive, "HAC SE should exceed iid SE under AR(1) errors"


def main() -> None:
    self_test()
    spec = load_spec()
    OUT.mkdir(exist_ok=True)
    s = attach_gex(session_table(spec), spec)
    s = s.dropna(subset=["sigma_scale", "ln_rv_5", "eff_20", "ln_vix_prev", "pct_lag1", "pct_lag2"])
    s.to_csv(OUT / "session_features.csv")
    is_lo, is_hi = spec["periods"]["IS"]
    res_is, trades_is, yearly_is = evaluate(s.loc[is_lo:is_hi])
    trades_is.to_csv(OUT / "canonical_trades_IS.csv")
    yearly_is.to_csv(OUT / "canonical_yearly_IS.csv")
    report = {"spec_sha256": json.loads((HERE / "SPEC_LOCK.json").read_text())["spec_sha256"],
              "sessions_IS": int(len(s.loc[is_lo:is_hi])), "IS": res_is}
    if res_is["gate"]["PASS"]:
        oo_lo, oo_hi = spec["periods"]["OOS"]
        res_oos, trades_oos, yearly_oos = evaluate(s.loc[oo_lo:oo_hi])
        trades_oos.to_csv(OUT / "canonical_trades_OOS.csv")
        yearly_oos.to_csv(OUT / "canonical_yearly_OOS.csv")
        report["OOS"] = res_oos
        report["decision"] = "IS_GATE_PASS_OOS_REPORTED"
    else:
        report["OOS"] = "NOT_COMPUTED"
        report["decision"] = "REJECT_MECHANISM"
    (OUT / "phase1_results.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))
    print(yearly_is)


if __name__ == "__main__":
    sys.exit(main())
