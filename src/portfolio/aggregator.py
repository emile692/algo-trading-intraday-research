"""Net position per instrument from frozen sleeve states.

Layer A and the B3 limits of research/book_aggregation_risk_r1_20261112/FROZEN_SPECIFICATION.json.

Each sleeve keeps a virtual signed contribution in contracts. The instrument target is the sum of
the contributions of its sleeves; `next_position` turns it into an executable position under the
lot cap and the per-instrument cap, and only the delta to the current position is traded. The same
`next_position` / `allowed_abs` policy is meant to drive live execution.

`simulate_session` replays one session minute by minute:

* events of kind ``KIND_OPEN`` execute at the open of their minute, grouped by instrument and price,
  reductions first; events of kind ``KIND_INTRABAR`` (S01 stop, target, end-of-day exit) execute
  afterwards inside the minute at their own price;
* every fill pays the commission and the slippage (ticks per side) of its instrument;
* instrument and book daily stops are checked on the marked PnL at each minute close; a trigger
  flattens at the next available open and locks the instrument (or the book) until session end,
  while virtual sleeve states keep evolving;
* the result carries the minute-close equity, a conservative intraminute worst equity (low for
  longs, high for shorts) and, for the profit lock applied later by the lifecycle simulator, the
  successive equity highs with the value of a flatten at the next open.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

KIND_OPEN = 0
KIND_INTRABAR = 1


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    point_value: float
    tick_size: float
    commission_per_side: float
    slippage_ticks_per_side: int
    lot_units: int  # Topstep lot in tenths: micro = 1, MBT = 10 (1 lot)

    @property
    def slippage_price(self) -> float:
        return self.slippage_ticks_per_side * self.tick_size

    @property
    def cost_per_contract_side(self) -> float:
        return self.commission_per_side + self.slippage_price * self.point_value


@dataclass(frozen=True)
class RiskLimits:
    lot_cap_units: int | None = None  # tenths of a lot, None = uncapped
    instrument_caps: tuple = ()  # |net| cap in contracts per instrument (None = uncapped)
    instrument_stop_usd: tuple = ()  # daily stop per instrument in USD (None = off)
    book_stop_usd: float | None = None


def allowed_abs(i: int, positions, specs, limits: RiskLimits) -> float:
    """Largest |position| instrument i may reach given the other instruments' positions."""
    cap = math.inf
    if limits.instrument_caps and limits.instrument_caps[i] is not None:
        cap = float(limits.instrument_caps[i])
    if limits.lot_cap_units is not None:
        used = 0
        for j, spec in enumerate(specs):
            if j != i:
                used += abs(int(positions[j])) * spec.lot_units
        room = max(0, int(limits.lot_cap_units) - used) // specs[i].lot_units
        cap = min(cap, float(room))
    return cap


def next_position(current: int, desired: int, allowed: float) -> int:
    """Executable position: follow the desired net, never exceed `allowed` when adding exposure,
    never cut an open position because of a cap."""
    if desired == current:
        return current
    if desired == 0:
        return 0
    sign = 1 if desired > 0 else -1
    if current != 0 and (current > 0) != (desired > 0):  # flip: through flat, then capped
        return int(sign * min(abs(desired), allowed))
    if abs(desired) <= abs(current):  # reduction
        return int(desired)
    return int(sign * min(abs(desired), max(abs(current), allowed)))


@dataclass(frozen=True)
class SessionBars:
    close: np.ndarray  # (T, I), forward-filled
    low: np.ndarray  # (T, I), close where no bar
    high: np.ndarray  # (T, I), close where no bar
    next_open: np.ndarray  # (T, I), fill price of an order sent at the close of minute t


@dataclass(frozen=True)
class SessionEvents:
    minute: np.ndarray  # int, index in the session grid
    kind: np.ndarray  # KIND_OPEN / KIND_INTRABAR
    instrument: np.ndarray  # int
    sleeve: np.ndarray  # int
    contribution: np.ndarray  # int, new signed contribution of the sleeve (0 on exit)
    price: np.ndarray  # raw price (before slippage)

    def __len__(self) -> int:
        return int(len(self.minute))


@dataclass
class SessionResult:
    final: float
    gross: float
    costs: float
    min_worst: float
    traded: bool
    contracts: int
    peak_units: int
    instrument_final: np.ndarray
    rec_val: np.ndarray
    rec_liq: np.ndarray
    rec_minw: np.ndarray
    instrument_stop_hits: np.ndarray
    book_stop_hit: bool
    fills: list = field(default_factory=list)
    e_close: np.ndarray | None = None
    e_worst: np.ndarray | None = None
    final_contributions: np.ndarray | None = None


def simulate_session(bars: SessionBars, events: SessionEvents, sleeve_instrument, specs,
                     limits: RiskLimits = RiskLimits(), keep_paths: bool = False) -> SessionResult:
    T, I = bars.close.shape
    pv = np.array([s.point_value for s in specs], dtype=float)
    comm = np.array([s.commission_per_side for s in specs], dtype=float)
    slip = np.array([s.slippage_price for s in specs], dtype=float)
    units = np.array([s.lot_units for s in specs], dtype=np.int64)
    sleeve_instrument = np.asarray(sleeve_instrument, dtype=np.int64)
    contrib = np.zeros(len(sleeve_instrument), dtype=np.int64)
    pos = np.zeros(I, dtype=np.int64)
    cash = np.zeros(I)
    gross_cash = np.zeros(I)
    locked = np.zeros(I, dtype=bool)
    book_locked = False
    stop_x = [limits.instrument_stop_usd[i] if limits.instrument_stop_usd else None for i in range(I)]
    book_y = limits.book_stop_usd
    inst_hits = np.zeros(I, dtype=bool)
    book_hit = False
    pos_path = np.zeros((T, I), dtype=np.int64)
    cash_path = np.zeros((T, I))
    pre_states: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    fills: list = []
    acc = {"costs": 0.0, "contracts": 0, "peak": 0}

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
        if (limits.instrument_caps and limits.instrument_caps[i] is not None and abs(new) > abs(old)
                and abs(new) > limits.instrument_caps[i]):
            raise AssertionError(f"instrument cap exceeded at minute {t}")
        acc["peak"] = max(acc["peak"], lots)
        if keep_paths:
            fills.append({"minute": t, "tag": tag, "instrument": i, "delta": delta, "fill": fill, "raw": price})

    def route(i: int, price: float, t: int, tag: str) -> None:
        if locked[i]:
            return
        desired = int(contrib[sleeve_instrument == i].sum())
        trade(i, next_position(int(pos[i]), desired, allowed_abs(i, pos, specs, limits)), price, t, tag)

    def first_trigger(t0: int, te: int):
        active = [i for i in range(I) if stop_x[i] is not None and not locked[i]]
        if not active and (book_y is None or book_locked):
            return None
        seg = bars.close[t0:te]
        inst_pnl = cash + np.where(pos != 0, pos * seg * pv, 0.0)
        cand = {}
        for i in active:
            hit = np.flatnonzero(inst_pnl[:, i] <= -stop_x[i])
            if hit.size:
                cand[i] = int(hit[0])
        book_first = None
        if book_y is not None and not book_locked:
            hit = np.flatnonzero(inst_pnl.sum(axis=1) <= -book_y)
            if hit.size:
                book_first = int(hit[0])
        times = list(cand.values()) + ([book_first] if book_first is not None else [])
        if not times:
            return None
        tau = min(times)
        return t0 + tau, [i for i, v in cand.items() if v == tau], book_first == tau

    # input order breaks ties: callers put a sleeve's exit before its entry at the same minute
    order = (np.lexsort((np.arange(len(events)), events.kind, events.minute))
             if len(events) else np.array([], dtype=int))
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
        while t0 < te:
            trig = first_trigger(t0, te)
            if trig is None:
                pos_path[t0:te] = pos
                cash_path[t0:te] = cash
                t0 = te
                break
            tau, insts, book = trig
            pos_path[t0:tau + 1] = pos
            cash_path[t0:tau + 1] = cash
            targets = list(range(I)) if book else insts
            for i in targets:
                if tau + 1 < T:
                    trade(i, 0, float(bars.next_open[tau, i]), tau + 1, "stop_flatten")
                locked[i] = True
            if book:
                book_locked = True
                book_hit = True
            for i in insts:
                inst_hits[i] = True
            t0 = tau + 1
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
            for k in groups[key]:
                contrib[sleeve[k]] = contribution[k]
            route(key[0], key[1], te, "open")
        if intra_idx:
            pre_states[te] = (pos.copy(), cash.copy())
            for k in intra_idx:
                contrib[sleeve[k]] = contribution[k]
                route(int(inst[k]), float(price[k]), te, "intrabar")
        t0 = te

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
    gross = float(gross_cash.sum())
    if abs(final - (gross - acc["costs"])) > 1e-6:
        raise AssertionError("cash accounting mismatch")
    return SessionResult(
        final=final,
        gross=gross,
        costs=float(acc["costs"]),
        min_worst=float(min(e_worst.min(), 0.0)),
        traded=acc["contracts"] > 0,
        contracts=int(acc["contracts"]),
        peak_units=int(acc["peak"]),
        instrument_final=cash.copy(),
        rec_val=e_close[rec].copy(),
        rec_liq=liq[rec].copy(),
        rec_minw=np.minimum(cummin[rec], liq[rec]),
        instrument_stop_hits=inst_hits,
        book_stop_hit=book_hit,
        fills=fills,
        e_close=e_close if keep_paths else None,
        e_worst=e_worst if keep_paths else None,
        final_contributions=contrib.copy(),
    )
