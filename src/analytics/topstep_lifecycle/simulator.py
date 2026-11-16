"""Vectorised Topstep lifecycle: Combine -> XFA -> payouts, on bootstrap paths of sessions.

Rules: research/book_aggregation_risk_r1_20261112/FROZEN_SPECIFICATION.json, key `topstep_50k_rules`.

Each session of a path is looked up in a `SessionTable` built for the lot cap in force (Combine cap,
or the XFA Scaling Plan tier read on the previous end-of-session balance). The optional Combine
profit lock truncates the session at the first minute-close equity high >= threshold and takes the
value of a flatten at the next open (records produced by `src.portfolio.aggregator`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CombineRules:
    max_lot_units: int = 50
    profit_target: float = 3000.0
    consistency: float = 0.55
    min_traded_days: int = 2
    mll: float = 2000.0


@dataclass(frozen=True)
class XfaRules:
    tier_bounds: tuple = (1500.0, 2000.0)  # < 1500 -> tier 0 ; 1500..2000 -> tier 1 ; > 2000 -> tier 2
    tier_units: tuple = (20, 30, 50)
    mll: float = 2000.0
    winning_day_usd: float = 150.0
    winning_days_for_payout: int = 5
    payout_fraction: float = 0.5
    payout_cap: float = 2000.0
    trader_share: float = 0.90

    def units_for_balance(self, balance: np.ndarray) -> np.ndarray:
        lo, hi = self.tier_bounds
        return np.where(balance < lo, self.tier_units[0], np.where(balance <= hi, self.tier_units[1], self.tier_units[2]))


@dataclass(frozen=True)
class Fees:
    subscription: float = 49.0
    reset: float = 49.0
    activation: float = 149.0
    sessions_per_subscription: int = 21


@dataclass(frozen=True)
class SessionTable:
    final: np.ndarray  # (S,)
    min_worst: np.ndarray  # (S,)
    traded: np.ndarray  # (S,) bool
    rec_val: np.ndarray  # (S, K), -inf padded (a padding slot never reaches a threshold)
    rec_liq: np.ndarray  # (S, K)
    rec_minw: np.ndarray  # (S, K)

    @staticmethod
    def from_results(results) -> "SessionTable":
        S = len(results)
        K = max(1, max(len(r.rec_val) for r in results))
        rec_val = np.full((S, K), -np.inf)
        rec_liq = np.zeros((S, K))
        rec_minw = np.zeros((S, K))
        for s, r in enumerate(results):
            n = len(r.rec_val)
            rec_val[s, :n] = r.rec_val
            rec_liq[s, :n] = r.rec_liq
            rec_minw[s, :n] = r.rec_minw
        return SessionTable(
            final=np.array([r.final for r in results]),
            min_worst=np.array([r.min_worst for r in results]),
            traded=np.array([r.traded for r in results], dtype=bool),
            rec_val=rec_val, rec_liq=rec_liq, rec_minw=rec_minw,
        )


def lookup(table: SessionTable, s: np.ndarray, threshold: np.ndarray):
    """Session PnL, intraday worst and traded flag, with the profit lock at `threshold` (inf = off)."""
    pnl = table.final[s].copy()
    minw = table.min_worst[s].copy()
    traded = table.traded[s].copy()
    lock = np.isfinite(threshold)
    if lock.any():
        idx = np.flatnonzero(lock)
        vals = table.rec_val[s[idx]]
        hit = vals >= threshold[idx, None]
        has = hit.any(axis=1)
        first = hit.argmax(axis=1)
        rows = s[idx]
        pnl[idx] = np.where(has, table.rec_liq[rows, first], pnl[idx])
        minw[idx] = np.where(has, table.rec_minw[rows, first], minw[idx])
    return pnl, minw, traded


def block_bootstrap_paths(n_sessions: int, n_paths: int, horizon: int, block_lengths=(3, 5, 10, 20), seed: int = 0) -> np.ndarray:
    """Moving-block bootstrap, block length pooled across paths (path p uses block_lengths[p % len])."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_paths, horizon), dtype=np.int64)
    for p in range(n_paths):
        L = min(block_lengths[p % len(block_lengths)], n_sessions)
        n_blocks = -(-horizon // L)
        starts = rng.integers(0, n_sessions - L + 1, size=n_blocks)
        out[p] = (starts[:, None] + np.arange(L)[None, :]).ravel()[:horizon]
    return out


def simulate_lifecycle(paths: np.ndarray, tables: dict, combine: CombineRules = CombineRules(), xfa: XfaRules = XfaRules(),
                       fees: Fees = Fees(), profit_lock: bool = False, lock_margin: float = 50.0,
                       combine_only: bool = False) -> dict:
    P, H = paths.shape
    phase = np.zeros(P, dtype=np.int8)  # 0 Combine, 1 XFA, 2 done (combine_only)
    sub_counter = np.zeros(P, dtype=np.int64)
    c_pnl = np.zeros(P); c_max = np.zeros(P); best = np.zeros(P); c_days = np.zeros(P, dtype=np.int64)
    x_bal = np.zeros(P); x_max = np.zeros(P); paid_once = np.zeros(P, dtype=bool); win = np.zeros(P, dtype=np.int64)
    fee_paid = np.zeros(P); payouts = np.zeros(P)
    first_payout = np.full(P, np.inf); first_pass = np.full(P, np.inf)
    n_combine_mll = np.zeros(P, dtype=np.int64); n_xfa_mll = np.zeros(P, dtype=np.int64); n_pass = np.zeros(P, dtype=np.int64)
    n_payouts = np.zeros(P, dtype=np.int64)

    for k in range(H):
        s = paths[:, k]
        in_c = phase == 0
        in_x = phase == 1
        due = in_c & (sub_counter % fees.sessions_per_subscription == 0)
        fee_paid += np.where(due, fees.subscription, 0.0)
        sub_counter += in_c
        units = np.where(in_c, combine.max_lot_units, xfa.units_for_balance(x_bal))
        target = np.maximum(combine.profit_target, best / combine.consistency)
        threshold = np.where(in_c & profit_lock, combine.consistency * target - lock_margin, np.inf)
        pnl = np.zeros(P); minw = np.zeros(P); traded = np.zeros(P, dtype=bool)
        live = in_c | in_x
        for u in np.unique(units[live]):
            m = live & (units == u)
            if int(u) not in tables:
                raise KeyError(f"no session table for lot cap {u}")
            pnl[m], minw[m], traded[m] = lookup(tables[int(u)], s[m], threshold[m])

        # Combine
        level_c = np.minimum(c_max - combine.mll, 0.0)
        breach_c = in_c & (c_pnl + minw <= level_c)
        ok_c = in_c & ~breach_c
        c_pnl = np.where(ok_c, c_pnl + pnl, c_pnl)
        c_max = np.where(ok_c, np.maximum(c_max, c_pnl), c_max)
        best = np.where(ok_c, np.maximum(best, pnl), best)
        c_days = np.where(ok_c & traded, c_days + 1, c_days)
        target_now = np.maximum(combine.profit_target, best / combine.consistency)
        passed = ok_c & (c_pnl >= target_now - 1e-9) & (c_days >= combine.min_traded_days)
        fee_paid += np.where(breach_c, fees.reset, 0.0)
        n_combine_mll += breach_c
        reset_c = breach_c | passed
        c_pnl = np.where(reset_c, 0.0, c_pnl); c_max = np.where(reset_c, 0.0, c_max)
        best = np.where(reset_c, 0.0, best); c_days = np.where(reset_c, 0, c_days)
        n_pass += passed
        first_pass = np.where(passed & np.isinf(first_pass), k + 1, first_pass)
        fee_paid += np.where(passed & ~np.bool_(combine_only), fees.activation, 0.0)

        # XFA
        level_x = np.where(paid_once, 0.0, np.minimum(x_max - xfa.mll, 0.0))
        breach_x = in_x & (x_bal + minw <= level_x)
        ok_x = in_x & ~breach_x
        x_bal = np.where(ok_x, x_bal + pnl, x_bal)
        x_max = np.where(ok_x, np.maximum(x_max, x_bal), x_max)
        win = np.where(ok_x & (pnl >= xfa.winning_day_usd), win + 1, win)
        eligible = ok_x & (win >= xfa.winning_days_for_payout) & (x_bal > 0)
        gross = np.where(eligible, np.minimum(xfa.payout_fraction * x_bal, xfa.payout_cap), 0.0)
        x_bal = x_bal - gross
        payouts += gross
        n_payouts += eligible
        win = np.where(eligible, 0, win)
        paid_once = paid_once | eligible
        first_payout = np.where(eligible & np.isinf(first_payout), k + 1, first_payout)
        n_xfa_mll += breach_x

        # phase transitions (after both blocks used the start-of-session phase)
        phase = np.where(passed, 2 if combine_only else 1, phase)
        start_x = passed & ~np.bool_(combine_only)
        x_bal = np.where(start_x, 0.0, x_bal); x_max = np.where(start_x, 0.0, x_max)
        paid_once = np.where(start_x, False, paid_once); win = np.where(start_x, 0, win)
        phase = np.where(breach_x, 0, phase)
        sub_counter = np.where(breach_x | passed, 0, sub_counter)

    return {
        "net_cash": xfa.trader_share * payouts - fee_paid,
        "gross_payouts": payouts,
        "fees": fee_paid,
        "first_payout_session": first_payout,
        "first_pass_session": first_pass,
        "n_payouts": n_payouts,
        "n_pass": n_pass,
        "n_combine_mll": n_combine_mll,
        "n_xfa_mll": n_xfa_mll,
    }


def summarize(out: dict) -> dict:
    first = out["first_payout_session"]
    first_pass = out["first_pass_session"]
    return {
        "expected_net_cash": float(np.mean(out["net_cash"])),
        "median_net_cash": float(np.median(out["net_cash"])),
        "median_sessions_to_first_payout": float(np.median(first)),
        "p_first_payout_le_60": float(np.mean(first <= 60)),
        "p_any_payout": float(np.mean(np.isfinite(first))),
        "p_combine_mll": float(np.mean(out["n_combine_mll"] > 0)),
        "p_xfa_lost": float(np.mean(out["n_xfa_mll"] > 0)),
        "p_pass": float(np.mean(np.isfinite(first_pass))),
        "median_sessions_to_pass": float(np.median(first_pass)),
        "mean_payouts": float(np.mean(out["n_payouts"])),
        "mean_fees": float(np.mean(out["fees"])),
    }
