"""Execution variants of the instrument-netting engine (phase 9).  Signals and sizes never change.

Copy of src/portfolio/aggregator.py::simulate_session (no instrument/book stop overlays: they were rejected and are
off in the active rule) with a parameterised routing:
  "i"   current rule: first come, first served, catch-up on any later event of the instrument;
  "ii"  no catch-up once the time is past the last entry time of every under-filled sleeve of the instrument;
  "iii" priority by IS risk-adjusted contribution: an increase triggered by a sleeve may not use the room reserved for
        the flat sleeves of higher priority (reserve = their native size in lots);
  "iv"  MBT sleeves are ignored while the lot cap is below 3 lots (30 units).
Also records refusals (entries reduced by the cap) per sleeve.  Gate: variant "i" must reproduce
aggregator.simulate_session exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.portfolio.aggregator import KIND_INTRABAR, KIND_OPEN, SessionResult, allowed_abs, next_position, RiskLimits


@dataclass
class VariantSpec:
    name: str = "i"
    last_entry_minute: np.ndarray | None = None  # per sleeve, grid minute index
    priority: np.ndarray | None = None  # per sleeve, smaller = higher priority
    reserve_units: np.ndarray | None = None  # per sleeve, lots (tenths) to reserve when flat
    mbt_index: int = 2


@dataclass
class RefusalLog:
    sessions_reduced: bool = False
    refused_by_sleeve: dict = field(default_factory=dict)  # sleeve -> contracts refused at entry


def simulate_session_v(bars, events, sleeve_instrument, specs, limits: RiskLimits, v: VariantSpec,
                       keep_paths: bool = False):
    T, I = bars.close.shape
    pv = np.array([s.point_value for s in specs], dtype=float)
    comm = np.array([s.commission_per_side for s in specs], dtype=float)
    slip = np.array([s.slippage_price for s in specs], dtype=float)
    units = np.array([s.lot_units for s in specs], dtype=np.int64)
    sleeve_instrument = np.asarray(sleeve_instrument, dtype=np.int64)
    n_sl = len(sleeve_instrument)
    contrib = np.zeros(n_sl, dtype=np.int64)
    pos = np.zeros(I, dtype=np.int64)
    cash = np.zeros(I)
    gross_cash = np.zeros(I)
    pos_path = np.zeros((T, I), dtype=np.int64)
    cash_path = np.zeros((T, I))
    pre_states: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    fills: list = []
    acc = {"costs": 0.0, "contracts": 0, "peak": 0}
    log = RefusalLog()
    underfilled = np.zeros(n_sl, dtype=bool)
    ignore_mbt = v.name == "iv" and limits.lot_cap_units is not None and limits.lot_cap_units < 30

    def trade(i: int, new: int, price: float, t: int, tag: str) -> None:
        old = int(pos[i])
        delta = new - old
        if delta == 0:
            return
        fill = price + math.copysign(slip[i], delta)
        cash[i] -= delta * fill * pv[i] + abs(delta) * comm[i]
        gross_cash[i] -= delta * price * pv[i]
        acc["costs"] += abs(delta) * (comm[i] + slip[i] * pv[i])
        acc["contracts"] += abs(delta)
        pos[i] = new
        lots = int(np.abs(pos) @ units)
        if limits.lot_cap_units is not None and abs(new) > abs(old) and lots > limits.lot_cap_units:
            raise AssertionError(f"lot cap exceeded at minute {t}: {lots} > {limits.lot_cap_units}")
        acc["peak"] = max(acc["peak"], lots)
        if keep_paths:
            fills.append({"minute": t, "tag": tag, "instrument": i, "delta": delta, "fill": fill, "raw": price})

    def effective(s: int) -> int:
        if ignore_mbt and sleeve_instrument[s] == v.mbt_index:
            return 0
        return int(contrib[s])

    def desired_of(i: int) -> int:
        return int(sum(effective(s) for s in np.flatnonzero(sleeve_instrument == i)))

    def route(i: int, price: float, t: int, tag: str, trigger: int | None, d_before: int) -> None:
        desired = desired_of(i)
        allowed = allowed_abs(i, pos, specs, limits)
        if v.name == "iii" and trigger is not None and limits.lot_cap_units is not None:
            # room reserved for flat sleeves of higher priority than the triggering sleeve
            p0 = v.priority[trigger]
            reserve = 0
            for s in range(n_sl):
                if s != trigger and v.priority[s] < p0 and contrib[s] == 0:
                    reserve += int(v.reserve_units[s])
            used_other = sum(abs(int(pos[j])) * specs[j].lot_units for j in range(I) if j != i)
            room = max(0, int(limits.lot_cap_units) - used_other - reserve) // specs[i].lot_units
            allowed = min(allowed, float(room))
        cur = int(pos[i])
        new = next_position(cur, desired, allowed)
        if v.name == "ii" and limits.lot_cap_units is not None:
            # catch-up = part of the move not explained by the event's own change of the desired net
            d_change, move = desired - d_before, new - cur
            explained = int(np.sign(d_change)) * min(abs(d_change), abs(move)) if np.sign(move) == np.sign(d_change) else 0
            if move != explained and abs(new) > abs(cur):
                owners = [x for x in np.flatnonzero(sleeve_instrument == i) if underfilled[x]]
                latest = max((int(v.last_entry_minute[x]) for x in owners), default=-1)
                if t > latest:
                    new = cur + explained
        if ignore_mbt and trigger is not None and sleeve_instrument[trigger] == v.mbt_index and contrib[trigger] != 0:
            log.sessions_reduced = True
            log.refused_by_sleeve[int(trigger)] = log.refused_by_sleeve.get(int(trigger), 0) + abs(int(contrib[trigger]))
        # refusal bookkeeping: an entry whose increase of the desired net is not fully executed
        if trigger is not None and contrib[trigger] != 0 and abs(desired) > abs(cur) and abs(new) < abs(desired):
            short = min(abs(desired) - abs(new), abs(int(contrib[trigger])))
            log.sessions_reduced = True
            log.refused_by_sleeve[int(trigger)] = log.refused_by_sleeve.get(int(trigger), 0) + int(short)
            underfilled[trigger] = True
        trade(i, new, price, t, tag)
        for x in np.flatnonzero(sleeve_instrument == i):
            if desired == new or contrib[x] == 0:
                underfilled[x] = False

    order = (np.lexsort((np.arange(len(events)), events.kind, events.minute)) if len(events) else np.array([], dtype=int))
    minute = events.minute[order]
    kind = events.kind[order]
    inst = events.instrument[order]
    sleeve = events.sleeve[order]
    contribution = events.contribution[order]
    price = events.price[order]
    if len(events) and (minute.min() < 0 or minute.max() >= T):
        raise ValueError("event outside the session grid")

    boundaries = list(np.unique(minute)) + [T]
    t0 = 0
    for te in boundaries:
        te = int(te)
        pos_path[t0:te] = pos
        cash_path[t0:te] = cash
        t0 = te
        if te == T:
            break
        sel = np.flatnonzero(minute == te)
        open_idx = [k for k in sel if kind[k] == KIND_OPEN]
        intra_idx = [k for k in sel if kind[k] == KIND_INTRABAR]
        groups: dict[tuple[int, float], list[int]] = {}
        for k in open_idx:
            groups.setdefault((int(inst[k]), float(price[k])), []).append(k)

        def is_increase(key):
            i, _ = key
            trial = contrib.copy()
            for k in groups[key]:
                trial[sleeve[k]] = contribution[k]
            return abs(int(trial[sleeve_instrument == i].sum())) > abs(int(pos[i]))

        for key in sorted(groups, key=lambda g: (is_increase(g), g[0], g[1])):
            d_before = desired_of(key[0])
            trigger = None
            for k in groups[key]:
                if contribution[k] != 0:
                    trigger = int(sleeve[k])
                contrib[sleeve[k]] = contribution[k]
            route(key[0], key[1], te, "open", trigger, d_before)
        if intra_idx:
            pre_states[te] = (pos.copy(), cash.copy())
            for k in intra_idx:
                d_before = desired_of(int(inst[k]))
                contrib[sleeve[k]] = contribution[k]
                route(int(inst[k]), float(price[k]), te, "intrabar", int(sleeve[k]) if contribution[k] != 0 else None, d_before)

    if np.any(pos != 0):
        raise RuntimeError(f"positions still open at session end: {pos.tolist()}")

    mtm_close = np.where(pos_path != 0, pos_path * bars.close * pv, 0.0)
    e_close = (cash_path + mtm_close).sum(axis=1)
    adverse = np.where(pos_path > 0, bars.low, bars.high)
    e_worst = (cash_path + np.where(pos_path != 0, pos_path * adverse * pv, 0.0)).sum(axis=1)
    for t, (p_pre, c_pre) in pre_states.items():
        adv = np.where(p_pre > 0, bars.low[t], bars.high[t])
        e_worst[t] = min(e_worst[t], float((c_pre + np.where(p_pre != 0, p_pre * adv * pv, 0.0)).sum()))
    exit_px = bars.next_open - np.sign(pos_path) * slip
    liq = (cash_path + np.where(pos_path != 0, pos_path * exit_px * pv - np.abs(pos_path) * comm, 0.0)).sum(axis=1)
    prev_max = np.maximum.accumulate(np.concatenate([[0.0], e_close[:-1]]))
    rec = np.flatnonzero(e_close > prev_max + 1e-9)
    cummin = np.minimum.accumulate(np.minimum(e_worst, 0.0))
    final = float(cash.sum())
    res = SessionResult(
        final=final, gross=float(gross_cash.sum()), costs=float(acc["costs"]), min_worst=float(min(e_worst.min(), 0.0)),
        traded=acc["contracts"] > 0, contracts=int(acc["contracts"]), peak_units=int(acc["peak"]),
        instrument_final=cash.copy(), rec_val=e_close[rec].copy(), rec_liq=liq[rec].copy(),
        rec_minw=np.minimum(cummin[rec], liq[rec]), instrument_stop_hits=np.zeros(I, dtype=bool), book_stop_hit=False,
        fills=fills, e_close=e_close if keep_paths else None, e_worst=e_worst if keep_paths else None,
        final_contributions=contrib.copy(),
    )
    return res, log
