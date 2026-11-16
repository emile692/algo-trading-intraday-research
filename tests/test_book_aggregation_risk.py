"""Synthetic self-tests of book_aggregation_risk_r1_20261112 (T1 linearity, T3 caps, T4 overlays, T5 no lookahead,
Topstep lifecycle rules)."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.analytics.topstep_lifecycle.simulator import (
    CombineRules, Fees, SessionTable, XfaRules, block_bootstrap_paths, lookup, simulate_lifecycle,
)
from src.portfolio.aggregator import (
    KIND_INTRABAR, KIND_OPEN, InstrumentSpec, RiskLimits, SessionBars, SessionEvents, allowed_abs, next_position,
    simulate_session,
)

SPECS = [
    InstrumentSpec("MNQ", 2.0, 0.25, 1.25, 1, 1),
    InstrumentSpec("MGC", 10.0, 0.1, 1.25, 1, 1),
    InstrumentSpec("MBT", 0.1, 5.0, 1.41, 1, 10),
]
SLEEVE_INST = np.array([0, 0, 0, 1, 1, 2, 2, 2])
T = 459


def make_bars(rng=None, T=T, base=(100.0, 50.0, 1000.0), drift=None):
    rng = rng or np.random.default_rng(0)
    I = 3
    steps = rng.normal(0, 0.2, size=(T, I)) * np.array([1.0, 0.2, 10.0])
    if drift is not None:
        steps = steps + np.asarray(drift)[None, :]
    close = np.array(base)[None, :] + np.cumsum(steps, axis=0)
    opn = np.vstack([np.array(base)[None, :], close[:-1]]) + rng.normal(0, 0.05, size=(T, I))
    high = np.maximum(opn, close) + 0.1
    low = np.minimum(opn, close) - 0.1
    next_open = np.vstack([opn[1:], close[-1:]])
    return SessionBars(close, low, high, next_open), opn


def events(rows):
    """rows: (minute, kind, inst, sleeve, contribution, price)."""
    a = np.array(rows, dtype=float).reshape(-1, 6)
    return SessionEvents(a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2].astype(np.int64),
                         a[:, 3].astype(np.int64), a[:, 4].astype(np.int64), a[:, 5])


def random_trades(rng, opn, n_sleeves=8, max_size=3):
    rows, gross = [], 0.0
    for s in range(n_sleeves):
        i = SLEEVE_INST[s]
        t = 5
        while True:
            t_in = t + int(rng.integers(1, 60))
            t_out = t_in + int(rng.integers(1, 90))
            if t_out >= T - 1:
                break
            direction = int(rng.choice([-1, 1]))
            n = int(rng.integers(1, max_size + 1))
            rows.append((t_in, KIND_OPEN, i, s, direction * n, opn[t_in, i]))
            rows.append((t_out, KIND_OPEN, i, s, 0, opn[t_out, i]))
            gross += direction * n * (opn[t_out, i] - opn[t_in, i]) * SPECS[i].point_value
            t = t_out
    rows.sort(key=lambda r: (r[0], r[4] != 0))  # exit before entry at the same minute
    return events(rows), gross


# ----------------------------------------------------------------------------- policy
def test_next_position_policy():
    assert next_position(0, 3, 2) == 2
    assert next_position(2, 3, 0) == 2  # never cut an open position because of a cap
    assert next_position(3, 1, 0) == 1  # reductions always follow
    assert next_position(2, -3, 1) == -1  # flip goes through flat then capped
    assert next_position(2, 0, 0) == 0
    assert next_position(-1, -4, 10) == -4


def test_allowed_abs_counts_mbt_as_one_lot():
    lim = RiskLimits(lot_cap_units=20, instrument_caps=(None, None, None))
    assert allowed_abs(2, [5, 0, 0], SPECS, lim) == 1  # 15 units left -> 1 MBT
    assert allowed_abs(0, [0, 0, 1], SPECS, lim) == 10  # 10 units left -> 10 micros
    lim2 = RiskLimits(lot_cap_units=None, instrument_caps=(None, None, 2))
    assert allowed_abs(2, [0, 0, 0], SPECS, lim2) == 2


# ----------------------------------------------------------------------------- T1 linearity
@pytest.mark.parametrize("seed", range(20))
def test_T1_netting_is_linear_without_caps(seed):
    rng = np.random.default_rng(seed)
    bars, opn = make_bars(rng)
    ev, gross = random_trades(rng, opn)
    r = simulate_session(bars, ev, SLEEVE_INST, SPECS, RiskLimits())
    assert r.gross == pytest.approx(gross, abs=1e-6)
    assert r.final == pytest.approx(r.gross - r.costs, abs=1e-6)
    assert np.all(r.final_contributions == 0)


def test_T1_opposite_sleeves_cost_less_than_separate():
    bars, opn = make_bars()
    ev = events([(10, KIND_OPEN, 0, 0, 1, opn[10, 0]), (10, KIND_OPEN, 0, 1, -1, opn[10, 0]),
                 (50, KIND_OPEN, 0, 0, 0, opn[50, 0]), (50, KIND_OPEN, 0, 1, 0, opn[50, 0])])
    r = simulate_session(bars, ev, SLEEVE_INST, SPECS, RiskLimits())
    assert r.contracts == 0 and r.costs == 0.0 and r.final == 0.0


# ----------------------------------------------------------------------------- T3 caps
@pytest.mark.parametrize("cap", [50, 30, 20])
def test_T3_all_eight_sleeves_enter_together(cap):
    bars, opn = make_bars()
    rows = [(10, KIND_OPEN, SLEEVE_INST[s], s, 3, opn[10, SLEEVE_INST[s]]) for s in range(8)]
    rows += [(200, KIND_OPEN, SLEEVE_INST[s], s, 0, opn[200, SLEEVE_INST[s]]) for s in range(8)]
    r = simulate_session(bars, events(rows), SLEEVE_INST, SPECS, RiskLimits(lot_cap_units=cap), keep_paths=True)
    pos = np.zeros(3, dtype=int)
    for f in r.fills:
        if f["minute"] == 10:
            pos[f["instrument"]] += f["delta"]
    assert pos[0] == 9 and pos[1] == 6  # micros first: 15 units
    assert pos[2] == (cap - 15) // 10  # MBT gets what is left, in whole lots
    assert r.peak_units <= cap


def test_T3_mbt_net_cap():
    bars, opn = make_bars()
    rows = [(10, KIND_OPEN, 2, s, 1, opn[10, 2]) for s in (5, 6, 7)]
    rows += [(100, KIND_OPEN, 2, s, 0, opn[100, 2]) for s in (5, 6, 7)]
    r = simulate_session(bars, events(rows), SLEEVE_INST, SPECS, RiskLimits(50, (None, None, 1)), keep_paths=True)
    assert [f["delta"] for f in r.fills] == [1, -1]


@pytest.mark.parametrize("seed", range(30))
def test_T3_fuzz_caps_never_exceeded(seed):
    rng = np.random.default_rng(100 + seed)
    bars, opn = make_bars(rng)
    ev, _ = random_trades(rng, opn, max_size=4)
    cap = int(rng.choice([20, 30, 50]))
    mbt_cap = rng.choice([None, 1, 2])
    lim = RiskLimits(lot_cap_units=cap, instrument_caps=(None, None, mbt_cap))
    r = simulate_session(bars, ev, SLEEVE_INST, SPECS, lim, keep_paths=True)
    pos = np.zeros(3, dtype=int)
    for f in r.fills:
        before = pos.copy()
        pos[f["instrument"]] += f["delta"]
        units = int(np.abs(pos) @ np.array([1, 1, 10]))
        if abs(pos[f["instrument"]]) > abs(before[f["instrument"]]):
            assert units <= cap
            if mbt_cap is not None and f["instrument"] == 2:
                assert abs(pos[2]) <= mbt_cap
    assert np.all(pos == 0)


# ----------------------------------------------------------------------------- T4 overlays
def falling_bars():
    T_ = T
    close = np.zeros((T_, 3))
    close[:, 0] = 100.0 - 0.2 * np.arange(T_)  # MNQ falls
    close[:, 1] = 50.0 + 0.01 * np.arange(T_)  # MGC rises
    close[:, 2] = 1000.0
    opn = np.vstack([close[:1], close[:-1]])
    nxt = np.vstack([opn[1:], close[-1:]])
    return SessionBars(close, close - 0.05, close + 0.05, nxt), opn


def test_T4_instrument_stop_flattens_and_locks_but_virtual_states_evolve():
    bars, opn = falling_bars()
    rows = [(5, KIND_OPEN, 0, 1, 1, opn[5, 0]), (5, KIND_OPEN, 1, 3, 1, opn[5, 1]),
            (200, KIND_OPEN, 0, 2, 1, opn[200, 0]), (300, KIND_OPEN, 0, 1, 0, opn[300, 0]),
            (350, KIND_OPEN, 0, 2, 0, opn[350, 0]), (400, KIND_OPEN, 1, 3, 0, opn[400, 1])]
    lim = RiskLimits(instrument_stop_usd=(50.0, None, None))
    r = simulate_session(bars, events(rows), SLEEVE_INST, SPECS, lim, keep_paths=True)
    mnq = [f for f in r.fills if f["instrument"] == 0]
    assert [f["tag"] for f in mnq] == ["open", "stop_flatten"]
    assert mnq[1]["minute"] < 200
    assert r.instrument_stop_hits[0] and not r.book_stop_hit
    assert [f["minute"] for f in r.fills if f["instrument"] == 1] == [5, 400]  # MGC unaffected
    assert np.all(r.final_contributions == 0)
    assert r.e_close[mnq[1]["minute"]:].min() >= -50.0 - 10.0  # loss bounded near the stop


def test_T4_book_stop_flattens_everything():
    bars, opn = falling_bars()
    rows = [(5, KIND_OPEN, 0, 1, 2, opn[5, 0]), (5, KIND_OPEN, 1, 3, 1, opn[5, 1]),
            (300, KIND_OPEN, 0, 1, 0, opn[300, 0]), (400, KIND_OPEN, 1, 3, 0, opn[400, 1]),
            (350, KIND_OPEN, 1, 4, 1, opn[350, 1]), (360, KIND_OPEN, 1, 4, 0, opn[360, 1])]
    rows.sort(key=lambda x: x[0])
    r = simulate_session(bars, events(rows), SLEEVE_INST, SPECS, RiskLimits(book_stop_usd=100.0), keep_paths=True)
    assert r.book_stop_hit
    stop_minute = max(f["minute"] for f in r.fills if f["tag"] == "stop_flatten")
    assert all(f["minute"] <= stop_minute for f in r.fills)


def test_T4_intrabar_stop_fills_at_its_price():
    bars, opn = make_bars()
    rows = [(10, KIND_OPEN, 0, 0, 2, opn[10, 0]), (40, KIND_INTRABAR, 0, 0, 0, 99.0)]
    r = simulate_session(bars, events(rows), SLEEVE_INST, SPECS, RiskLimits(), keep_paths=True)
    assert r.fills[1]["fill"] == pytest.approx(99.0 - 0.25)
    assert r.gross == pytest.approx(2 * (99.0 - opn[10, 0]) * 2.0)


def test_T4_profit_lock_lookup_truncates_at_first_high():
    res = [SimpleNamespace(final=500.0, min_worst=-300.0, traded=True, rec_val=np.array([100.0, 800.0, 1700.0]),
                           rec_liq=np.array([95.0, 790.0, 1690.0]), rec_minw=np.array([-10.0, -50.0, -60.0]))]
    tab = SessionTable.from_results(res)
    pnl, minw, _ = lookup(tab, np.array([0, 0, 0]), np.array([np.inf, 700.0, 5000.0]))
    assert pnl.tolist() == [500.0, 790.0, 500.0]
    assert minw.tolist() == [-300.0, -50.0, -300.0]


def test_T4_profit_lock_padding_never_triggers():
    # sessions with different numbers of equity highs: padding slots must not count as a hit
    res = [SimpleNamespace(final=300.0, min_worst=-20.0, traded=True, rec_val=np.array([100.0]),
                           rec_liq=np.array([99.0]), rec_minw=np.array([-5.0])),
           SimpleNamespace(final=1800.0, min_worst=-40.0, traded=True,
                           rec_val=np.array([100.0, 900.0, 1700.0, 1900.0]),
                           rec_liq=np.array([98.0, 880.0, 1680.0, 1880.0]),
                           rec_minw=np.array([-5.0, -10.0, -20.0, -30.0]))]
    tab = SessionTable.from_results(res)
    pnl, minw, _ = lookup(tab, np.array([0, 1]), np.array([1600.0, 1600.0]))
    assert pnl.tolist() == [300.0, 1680.0]
    assert minw.tolist() == [-20.0, -20.0]


def test_lifecycle_profit_lock_still_passes():
    rec = SimpleNamespace(final=1000.0, min_worst=0.0, traded=True, rec_val=np.array([500.0]),
                          rec_liq=np.array([495.0]), rec_minw=np.array([0.0]))
    other = SimpleNamespace(final=0.0, min_worst=0.0, traded=False, rec_val=np.array([10.0, 20.0, 30.0]),
                            rec_liq=np.array([10.0, 20.0, 30.0]), rec_minw=np.array([0.0, 0.0, 0.0]))
    tab = SessionTable.from_results([rec, other])
    out = simulate_lifecycle(np.zeros((1, 3), dtype=int), {20: tab, 30: tab, 50: tab}, profit_lock=True)
    assert out["first_pass_session"][0] == 3


# ----------------------------------------------------------------------------- T5 no lookahead
@pytest.mark.parametrize("seed", range(10))
def test_T5_decisions_ignore_later_minutes(seed):
    rng = np.random.default_rng(500 + seed)
    bars, opn = make_bars(rng)
    ev, _ = random_trades(rng, opn)
    lim = RiskLimits(lot_cap_units=30, instrument_caps=(None, None, 1), instrument_stop_usd=(40.0, 30.0, 20.0),
                     book_stop_usd=80.0)
    a = simulate_session(bars, ev, SLEEVE_INST, SPECS, lim, keep_paths=True)
    tau = 150
    pert = SessionBars(*(np.vstack([x[:tau + 1], x[tau + 1:] * 1.3]) for x in (bars.close, bars.low, bars.high, bars.next_open)))
    b = simulate_session(pert, ev, SLEEVE_INST, SPECS, lim, keep_paths=True)
    fa = [(f["minute"], f["instrument"], f["delta"], f["tag"]) for f in a.fills if f["minute"] <= tau]
    fb = [(f["minute"], f["instrument"], f["delta"], f["tag"]) for f in b.fills if f["minute"] <= tau]
    assert fa == fb


# ----------------------------------------------------------------------------- lifecycle
def const_tables(pnls, minws=None, traded=True):
    minws = minws if minws is not None else [min(p, 0.0) for p in pnls]
    res = [SimpleNamespace(final=p, min_worst=m, traded=traded, rec_val=np.array([]), rec_liq=np.array([]),
                           rec_minw=np.array([])) for p, m in zip(pnls, minws)]
    tab = SessionTable.from_results(res)
    return {20: tab, 30: tab, 50: tab}


def run(pnls, **kw):
    tables = const_tables(pnls, kw.pop("minws", None))
    paths = np.arange(len(pnls))[None, :]
    return simulate_lifecycle(paths, tables, **kw)


def test_lifecycle_pass_then_two_payouts():
    out = run([1000.0] * 13)
    assert out["first_pass_session"][0] == 3
    assert out["first_payout_session"][0] == 8
    assert out["gross_payouts"][0] == 4000.0
    assert out["fees"][0] == 49.0 + 149.0
    assert out["net_cash"][0] == pytest.approx(0.9 * 4000.0 - 198.0)


def test_lifecycle_consistency_raises_target():
    out = run([2200.0] + [200.0] * 12)
    assert out["first_pass_session"][0] == 10  # target 4000 = 2200 / 0.55


def test_lifecycle_combine_mll_reset_fee():
    out = run([-2000.0, 0.0])
    assert out["n_combine_mll"][0] == 1
    assert out["fees"][0] == 49.0 + 49.0


def test_lifecycle_subscription_every_21_sessions():
    out = run([0.0] * 42)
    assert out["fees"][0] == 98.0


def test_lifecycle_xfa_loss_restarts_combine_with_new_subscription():
    out = run([1500.0, 1500.0, -2100.0, 0.0])
    assert out["n_pass"][0] == 1 and out["n_xfa_mll"][0] == 1
    assert out["fees"][0] == 49.0 + 149.0 + 49.0


def test_xfa_tiers():
    x = XfaRules()
    assert x.units_for_balance(np.array([0.0, 1499.0, 1500.0, 2000.0, 2000.01])).tolist() == [20, 20, 30, 30, 50]


def test_lifecycle_uses_tier_tables():
    good = SessionTable.from_results([SimpleNamespace(final=1000.0, min_worst=0.0, traded=True, rec_val=np.array([]),
                                                      rec_liq=np.array([]), rec_minw=np.array([]))])
    small = SessionTable.from_results([SimpleNamespace(final=100.0, min_worst=0.0, traded=True, rec_val=np.array([]),
                                                       rec_liq=np.array([]), rec_minw=np.array([]))])
    out = simulate_lifecycle(np.zeros((1, 8), dtype=int), {50: good, 20: small, 30: good})
    # Combine (cap 50): 3 x 1000 -> pass ; XFA starts at 0 -> tier 20 -> 100/day
    assert out["first_pass_session"][0] == 3
    assert out["gross_payouts"][0] == 0.0


def test_bootstrap_paths_deterministic_and_in_range():
    a = block_bootstrap_paths(100, 8, 126, seed=7)
    b = block_bootstrap_paths(100, 8, 126, seed=7)
    assert np.array_equal(a, b) and a.shape == (8, 126) and a.min() >= 0 and a.max() < 100
    # path 0 uses blocks of 3 consecutive sessions
    assert np.all(np.diff(a[0][:3]) == 1)


def test_combine_only_mode_stops_at_pass():
    out = run([1000.0] * 10, combine_only=True)
    assert out["n_pass"][0] == 1 and out["fees"][0] == 49.0
