"""Metrics layer: per-group statistics in R and USD (1 contract), Monte Carlo of trade order."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pf(x: pd.Series) -> float:
    gains, losses = x[x > 0].sum(), -x[x < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def max_drawdown(pnl: np.ndarray) -> float:
    """Max peak-to-trough drawdown of the cumulative sum (positive number), starting from 0."""
    if len(pnl) == 0:
        return 0.0
    eq = np.concatenate([[0.0], np.cumsum(pnl)])
    return float(np.max(np.maximum.accumulate(eq) - eq))


def worst_losing_streak(pnl: np.ndarray) -> int:
    best = cur = 0
    for v in pnl:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best


def summarize(trades: pd.DataFrame) -> dict:
    """Statistics of FILLED trades (net = after slippage and commission)."""
    if trades.empty or "status" not in trades:
        return {"trades": 0}
    t = trades[trades["status"] == "filled"].sort_values("signal_time")
    n = len(t)
    if n == 0:
        return {"trades": 0}
    net, r = t["pnl_net_usd"], t["r_net"]
    wins, losses = t[net > 0], t[net <= 0]
    daily = net.groupby(t["date"]).sum()
    monthly = net.groupby(pd.to_datetime(t["date"]).dt.to_period("M")).sum()
    return {
        "trades": n,
        "rejected_min_stop": int((trades["status"] == "rejected_min_stop").sum()),
        "win_rate": float((net > 0).mean()),
        "avg_win_usd": float(wins["pnl_net_usd"].mean()) if len(wins) else np.nan,
        "avg_loss_usd": float(losses["pnl_net_usd"].mean()) if len(losses) else np.nan,
        "avg_win_r": float(wins["r_net"].mean()) if len(wins) else np.nan,
        "avg_loss_r": float(losses["r_net"].mean()) if len(losses) else np.nan,
        "expectancy_r": float(r.mean()),
        "expectancy_usd": float(net.mean()),
        "expectancy_r_gross": float(t["r_gross"].mean()),
        "pf_gross": _pf(t["pnl_gross_usd"]),
        "pf_net": _pf(net),
        "total_net_usd": float(net.sum()),
        "total_r": float(r.sum()),
        "max_dd_usd": max_drawdown(net.to_numpy()),
        "max_dd_r": max_drawdown(r.to_numpy()),
        "worst_day_usd": float(daily.min()),
        "worst_losing_streak": worst_losing_streak(net.to_numpy()),
        "pct_positive_months": float((monthly > 0).mean()),
        "median_risk_ticks": float(t["risk_ticks"].median()),
        "cost_per_trade_r": float((t["r_gross"] - t["r_net"]).mean()),
        "share_stop": float((t["exit_reason"] == "stop").mean()),
        "share_target": float((t["exit_reason"] == "target").mean()),
        "share_time": float((t["exit_reason"] == "time").mean()),
    }


def table(trades: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in trades.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **summarize(g)})
    return pd.DataFrame(rows)


def mc_drawdown(pnl: np.ndarray, n_perm: int = 10_000, seed: int = 20261120,
                quantiles=(0.5, 0.9, 0.95, 0.99)) -> dict:
    """Distribution of max drawdown over random permutations of the trade order."""
    rng = np.random.default_rng(seed)
    pnl = np.asarray(pnl, float)
    if len(pnl) == 0:
        return {f"dd_q{int(q * 100)}": 0.0 for q in quantiles}
    perms = np.argsort(rng.random((n_perm, len(pnl))), axis=1)
    eq = np.cumsum(pnl[perms], axis=1)
    eq = np.concatenate([np.zeros((n_perm, 1)), eq], axis=1)
    dd = (np.maximum.accumulate(eq, axis=1) - eq).max(axis=1)
    return {f"dd_q{int(q * 100)}": float(np.quantile(dd, q)) for q in quantiles}
