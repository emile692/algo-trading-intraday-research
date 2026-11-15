"""Statistical tools: Lo (2002) HAC Sharpe standard error, Politis-Romano stationary bootstrap, PSR, DSR."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def newey_west_lag(T: int) -> int:
    return int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def lo_sharpe(x, lags: int | None = None) -> dict:
    """Lo (2002) GMM standard error of the Sharpe ratio, Newey-West HAC (Bartlett kernel).

    Returns per-period and annualised (x sqrt(252)) Sharpe, its HAC SE, IID SE and the t-statistic.
    """
    x = np.asarray(x, float)
    T = len(x)
    mu, sd = x.mean(), x.std(ddof=1)
    sr = mu / sd
    var = x.var(ddof=0)
    u = np.column_stack([x - mu, (x - mu) ** 2 - var])
    L = newey_west_lag(T) if lags is None else lags
    S = u.T @ u / T
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        G = u[k:].T @ u[:-k] / T
        S += w * (G + G.T)
    g = np.array([1.0 / math.sqrt(var), -mu / (2.0 * var ** 1.5)])
    se = math.sqrt(max(g @ S @ g, 0.0) / T)
    se_iid = math.sqrt((1.0 + 0.5 * sr ** 2) / T)
    a = math.sqrt(252.0)
    return {"T": T, "sharpe": a * sr, "se_hac": a * se, "se_iid": a * se_iid, "t_hac": sr / se if se > 0 else float("nan"),
            "nw_lags": L}


def stationary_indices(T: int, n: int, mean_block: float, rng: np.random.Generator,
                       length: int | None = None) -> np.ndarray:
    """Politis-Romano (1994) stationary bootstrap indices into a sample of size T, shape (n, length or T),
    circular wrapping."""
    p = 1.0 / mean_block
    L = T if length is None else int(length)
    idx = np.empty((n, L), dtype=np.int32)
    idx[:, 0] = rng.integers(0, T, n)
    new = rng.random((n, L)) < p
    starts = rng.integers(0, T, (n, L))
    for t in range(1, L):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def boot_stats(x: np.ndarray, idx: np.ndarray, chunk: int = 2000) -> dict[str, np.ndarray]:
    """Sharpe (annualised), total PnL and max drawdown of each bootstrap path."""
    x = np.asarray(x, float)
    out = {"sharpe": [], "pnl": [], "max_dd": []}
    for s in range(0, len(idx), chunk):
        m = x[idx[s:s + chunk]]
        sd = m.std(axis=1, ddof=1)
        out["sharpe"].append(np.where(sd > 0, math.sqrt(252.0) * m.mean(axis=1) / np.where(sd > 0, sd, 1), 0.0))
        out["pnl"].append(m.sum(axis=1))
        eq = np.concatenate([np.zeros((len(m), 1)), np.cumsum(m, axis=1)], axis=1)
        out["max_dd"].append((eq - np.maximum.accumulate(eq, axis=1)).min(axis=1))
    return {k: np.concatenate(v) for k, v in out.items()}


def psr(x, sr_star_annual: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012), benchmark given annualised."""
    x = np.asarray(x, float)
    T = len(x)
    sr = x.mean() / x.std(ddof=1)
    sk = stats.skew(x, bias=False)
    ku = stats.kurtosis(x, fisher=False, bias=False)
    sr_star = sr_star_annual / math.sqrt(252.0)
    den = math.sqrt(max(1.0 - sk * sr + (ku - 1.0) / 4.0 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_star) * math.sqrt(T - 1.0) / den))


def expected_max_sharpe(n_trials: int, var_sr_annual: float) -> float:
    """SR_0 of the Deflated Sharpe Ratio (Bailey & López de Prado 2014), annualised units in and out."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sr_annual) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def dsr(x, n_trials: int, var_sr_annual: float) -> tuple[float, float]:
    sr0 = expected_max_sharpe(n_trials, var_sr_annual)
    return psr(x, sr0), sr0
