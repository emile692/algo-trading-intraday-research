"""book_aggregation_risk_r1_20261112 — runs exactly FROZEN_SPECIFICATION.json (hash-checked against SPEC_LOCK.json).

Subcommands
-----------
selftest   synthetic tests (tests/test_book_aggregation_risk.py) + real-data reproduction checks T1, T2, T3, T5
           on IS sessions only. Writes results/selftest_report.json with the code hashes.
estimate   IS estimates: ES per contract, sigma_ref, B1 sizing per k, instrument ES95, B2 diagnostic, S01 sizing change.
is         sequential selection on IS_common with the Topstep lifecycle bootstrap, CSCV/PBO, sensitivities.
oos        single OOS read (refuses to run twice), verdict versus B0.
report     results/summary.md from the JSON outputs.

Code lock: `is` requires a passing selftest under the current code hashes; `oos` requires the selftest and the
IS selection to have run under the current code hashes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.analytics.topstep_lifecycle.simulator import (  # noqa: E402
    CombineRules, Fees, SessionTable, XfaRules, block_bootstrap_paths, simulate_lifecycle, summarize,
)
from src.portfolio.aggregator import (  # noqa: E402
    KIND_INTRABAR, KIND_OPEN, InstrumentSpec, RiskLimits, SessionBars, SessionEvents, simulate_session,
)

RESULTS = HERE / "results"
TEST_FILE = ROOT / "tests" / "test_book_aggregation_risk.py"
LOCKED_CODE = [HERE / "run_study.py", ROOT / "src/portfolio/aggregator.py",
               ROOT / "src/analytics/topstep_lifecycle/simulator.py", TEST_FILE]
INSTRUMENTS = ["MNQ", "MGC", "MBT"]
SLEEVES = ["S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09"]
GRID_MINUTES = 459  # 08:30 .. 16:08 America/New_York
GRID_START_MIN = 8 * 60 + 30
CLOSE_MINUTE = 16 * 60  # daily close used for sigma: last price at or before 16:00 ET
ET = "America/New_York"


# ----------------------------------------------------------------------------- utilities
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def code_hashes() -> dict:
    return {rel(p): sha256(p) for p in LOCKED_CODE}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (dt.date, pd.Timestamp)):
        return str(o)
    raise TypeError(type(o))


def es95(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = max(1, math.ceil(0.05 * len(x)))
    return float(-x[:n].mean())


def d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


# ----------------------------------------------------------------------------- spec and inputs
def load_spec() -> dict:
    lock = json.loads((HERE / "SPEC_LOCK.json").read_text(encoding="utf-8"))
    p = HERE / lock["spec_file"]
    if sha256(p) != lock["spec_sha256"]:
        raise SystemExit("FROZEN_SPECIFICATION.json does not match SPEC_LOCK.json")
    return json.loads(p.read_text(encoding="utf-8"))


def verify_inputs(spec: dict) -> dict:
    checked = {}
    items = [(s["trades"], s["sha256"]) for s in spec["sleeves"].values()]
    items += [(v["bars_1m"], v["sha256"]) for v in spec["instruments"].values()]
    items += [(spec["frozen_book"][k]["path"], spec["frozen_book"][k]["sha256"]) for k in ("lock_config", "lock_daily_pnl")]
    for path, expected in items:
        got = sha256(ROOT / path)
        if got != expected:
            raise SystemExit(f"input hash mismatch: {path}")
        checked[path] = got
    return checked


def instrument_specs(spec: dict, mbt_lot_units: int = 10) -> list[InstrumentSpec]:
    out = []
    for name in INSTRUMENTS:
        v = spec["instruments"][name]
        units = int(round(v["topstep_lot_per_contract"] * 10))
        if name == "MBT":
            units = mbt_lot_units
        out.append(InstrumentSpec(name, v["point_value"], v["tick_size"], v["commission_per_side"],
                                  v["slippage_ticks_per_side"], units))
    return out


def load_trades(spec: dict) -> pd.DataFrame:
    frames = []
    for s_idx, sid in enumerate(SLEEVES):
        s = spec["sleeves"][sid]
        inst = s["instrument"]
        ins = spec["instruments"][inst]
        pv, tick = ins["point_value"], ins["tick_size"]
        t = pd.read_csv(ROOT / s["trades"])
        e = pd.to_datetime(t["entry_time"], utc=True)
        x = pd.to_datetime(t["exit_time"], utc=True)
        direction = np.where(t["direction"].str.lower() == "long", 1, -1)
        if sid == "S01":
            qty = t["quantity"].astype(int).to_numpy()
            raw_e = t["entry_price"].to_numpy() - direction * tick
            raw_x = t["exit_price"].to_numpy() + direction * tick
            gross = direction * (raw_x - raw_e) * pv * qty
            cost = t["fees"].to_numpy() + 2 * tick * pv * qty
            exit_kind = np.where(t["exit_reason"].isin(["stop", "target", "eod_exit"]), KIND_INTRABAR, KIND_OPEN)
        else:
            if not np.array_equal(direction, t["position"].to_numpy()):
                raise SystemExit(f"{sid}: direction and position disagree")
            qty = t["quantity"].astype(int).to_numpy()
            raw_e = t["raw_entry_price"].to_numpy()
            raw_x = t["raw_exit_price"].to_numpy()
            gross = t["gross_pnl_usd"].to_numpy()
            cost = (t["commissions_usd"] + t["slippage_usd"]).to_numpy()
            exit_kind = np.full(len(t), KIND_OPEN)
        et_e, et_x = e.dt.tz_convert(ET), x.dt.tz_convert(ET)
        session = pd.to_datetime(t["session_date"]).dt.date
        if not ((et_e.dt.date == session).all() and (et_x.dt.date == session).all()):
            raise SystemExit(f"{sid}: trade crosses its session date")
        frames.append(pd.DataFrame({
            "sleeve": sid, "s_idx": s_idx, "instrument": inst, "i_idx": INSTRUMENTS.index(inst),
            "session_date": session,
            "m_entry": (et_e.dt.hour * 60 + et_e.dt.minute - GRID_START_MIN).to_numpy(),
            "m_exit": (et_x.dt.hour * 60 + et_x.dt.minute - GRID_START_MIN).to_numpy(),
            "direction": direction, "qty_native": qty, "raw_entry": raw_e, "raw_exit": raw_x,
            "entry_kind": KIND_OPEN, "exit_kind": exit_kind,
            "gross_native": gross, "cost_native": cost, "net_native": t["net_pnl_usd"].to_numpy(),
            "net_per_contract": t["net_pnl_usd"].to_numpy() / qty,
            "native_entry_fill": t["entry_price"].to_numpy(), "native_exit_fill": t["exit_price"].to_numpy(),
        }))
    trades = pd.concat(frames, ignore_index=True)
    if (trades.m_entry < 0).any() or (trades.m_exit >= GRID_MINUTES).any() or (trades.m_exit < trades.m_entry).any():
        raise SystemExit("trade outside the 08:30-16:08 ET grid")
    if not np.allclose(trades.gross_native - trades.cost_native, trades.net_native, atol=1e-6):
        raise SystemExit("native gross - cost != net")
    return trades


def load_calendars(spec: dict) -> tuple[dict, dict]:
    cal = {i: set() for i in INSTRUMENTS}
    hashes = {}
    for sid in SLEEVES[1:]:
        s = spec["sleeves"][sid]
        p = ROOT / Path(s["trades"]).parent / "daily_results.csv"
        hashes[rel(p)] = sha256(p)
        cal[s["instrument"]] |= set(pd.to_datetime(pd.read_csv(p, usecols=["session_date"])["session_date"]).dt.date)
    return {i: sorted(v) for i, v in cal.items()}, hashes


def union_sessions(calendars: dict, start: dt.date, end: dt.date) -> list[dt.date]:
    return sorted({x for v in calendars.values() for x in v if start <= x <= end})


def load_bars(spec: dict) -> dict:
    out = {}
    for name in INSTRUMENTS:
        b = pd.read_parquet(ROOT / spec["instruments"][name]["bars_1m"], columns=["open", "high", "low", "close"])
        b.index = b.index.tz_convert("UTC")
        out[name] = b[~b.index.duplicated(keep="last")].sort_index()
    return out


def session_grid(dates) -> pd.DatetimeIndex:
    starts = pd.DatetimeIndex([pd.Timestamp(f"{x} 08:30", tz=ET) for x in dates]).tz_convert("UTC")
    offs = np.arange(GRID_MINUTES).astype("timedelta64[m]")
    flat = (starts.values.astype("datetime64[ns]")[:, None] + offs[None, :]).ravel()
    return pd.DatetimeIndex(flat).tz_localize("UTC")


class BarCube:
    def __init__(self, bars: dict, dates):
        self.dates = list(dates)
        self.pos = {x: k for k, x in enumerate(self.dates)}
        S, T, I = len(self.dates), GRID_MINUTES, len(INSTRUMENTS)
        grid = session_grid(self.dates)
        self.close = np.empty((S, T, I)); self.low = np.empty((S, T, I))
        self.high = np.empty((S, T, I)); self.next_open = np.empty((S, T, I))
        for j, name in enumerate(INSTRUMENTS):
            b = bars[name]
            op = b["open"].reindex(grid).to_numpy().reshape(S, T)
            hi = b["high"].reindex(grid).to_numpy().reshape(S, T)
            lo = b["low"].reindex(grid).to_numpy().reshape(S, T)
            cl = b["close"].reindex(grid, method="ffill").to_numpy().reshape(S, T)
            nxt = pd.DataFrame(op).shift(-1, axis=1).bfill(axis=1).to_numpy()
            self.close[:, :, j] = cl
            self.low[:, :, j] = np.where(np.isnan(lo), cl, lo)
            self.high[:, :, j] = np.where(np.isnan(hi), cl, hi)
            self.next_open[:, :, j] = np.where(np.isnan(nxt), cl, nxt)

    def session(self, date) -> SessionBars:
        k = self.pos[date]
        return SessionBars(self.close[k], self.low[k], self.high[k], self.next_open[k])


def instrument_sigma(bar_close: pd.Series, dates) -> pd.Series:
    """sigma_it: std of daily log returns over the 20 previous sessions (min 15), known before session t."""
    ts = pd.DatetimeIndex([pd.Timestamp(f"{x} 16:00", tz=ET) for x in dates]).tz_convert("UTC")
    c = bar_close.reindex(ts, method="ffill").to_numpy()
    r = pd.Series(np.log(c)).diff()
    sigma = r.shift(1).rolling(20, min_periods=15).std()
    return pd.Series(sigma.to_numpy(), index=list(dates))


# ----------------------------------------------------------------------------- events
def build_events(trades: pd.DataFrame, n: np.ndarray) -> dict:
    """Session date -> SessionEvents. Entry sets the sleeve contribution to direction * n, exit to 0."""
    n = np.asarray(n, dtype=np.int64)
    ent = pd.DataFrame({"session_date": trades.session_date, "minute": trades.m_entry, "kind": trades.entry_kind,
                        "inst": trades.i_idx, "sleeve": trades.s_idx, "contrib": trades.direction.to_numpy() * n,
                        "price": trades.raw_entry, "is_entry": 1})
    ext = pd.DataFrame({"session_date": trades.session_date, "minute": trades.m_exit, "kind": trades.exit_kind,
                        "inst": trades.i_idx, "sleeve": trades.s_idx, "contrib": 0,
                        "price": trades.raw_exit, "is_entry": 0})
    ev = pd.concat([ent, ext], ignore_index=True).sort_values(
        ["session_date", "minute", "kind", "is_entry", "sleeve"], kind="mergesort")
    out = {}
    for date, g in ev.groupby("session_date", sort=False):
        out[date] = SessionEvents(g.minute.to_numpy(np.int64), g.kind.to_numpy(np.int64), g.inst.to_numpy(np.int64),
                                  g.sleeve.to_numpy(np.int64), g.contrib.to_numpy(np.int64), g.price.to_numpy(float))
    return out


EMPTY_EVENTS = SessionEvents(*(np.array([], dtype=np.int64) for _ in range(5)), np.array([], dtype=float))
SLEEVE_INSTRUMENT = np.array([INSTRUMENTS.index(i) for i in ["MNQ", "MNQ", "MNQ", "MGC", "MGC", "MBT", "MBT", "MBT"]])


def simulate_sessions(cube: BarCube, dates, events: dict, specs, limits: RiskLimits, keep_paths=False):
    return [simulate_session(cube.session(x), events.get(x, EMPTY_EVENTS), SLEEVE_INSTRUMENT, specs, limits,
                             keep_paths=keep_paths) for x in dates]


# ----------------------------------------------------------------------------- estimates
def per_contract_daily(trades: pd.DataFrame, sleeve: str, dates) -> pd.Series:
    t = trades[trades.sleeve == sleeve]
    daily = t.groupby("session_date")["net_per_contract"].sum()
    return daily.reindex(list(dates), fill_value=0.0)


def peak_lots(trades: pd.DataFrame, n_by_sleeve: np.ndarray, dates, units: np.ndarray) -> float:
    """Peak simultaneous Topstep lots of the uncapped netted book (state after each event group)."""
    t = trades[trades.session_date.isin(set(dates))]
    ent = pd.DataFrame({"session_date": t.session_date, "minute": t.m_entry, "kind": t.entry_kind,
                        "sleeve": t.s_idx, "dir": t.direction, "is_entry": 1})
    ext = pd.DataFrame({"session_date": t.session_date, "minute": t.m_exit, "kind": t.exit_kind,
                        "sleeve": t.s_idx, "dir": 0, "is_entry": 0})
    ev = pd.concat([ent, ext]).sort_values(["session_date", "minute", "kind", "is_entry"], kind="mergesort")
    ev["key"] = ev.groupby(["session_date", "minute", "kind"], sort=False).ngroup()
    piv = ev.pivot_table(index="key", columns="sleeve", values="dir", aggfunc="last")
    piv = piv.reindex(columns=range(len(SLEEVES))).ffill().fillna(0.0).to_numpy()
    contrib = piv * n_by_sleeve[None, :]
    lots = np.zeros(len(piv))
    for j in range(len(INSTRUMENTS)):
        lots += np.abs(contrib[:, SLEEVE_INSTRUMENT == j].sum(axis=1)) * units[j]
    return float(lots.max() / 10.0) if len(lots) else 0.0


def estimate(spec, trades, calendars, bars, specs) -> dict:
    per = spec["periods"]["IS_by_sleeve"]
    win = {"MNQ": per["MNQ_MGC"], "MGC": per["MNQ_MGC"], "MBT": per["MBT"]}
    sleeve_inst = {sid: spec["sleeves"][sid]["instrument"] for sid in SLEEVES}

    def is_dates(inst):
        a, b = map(d, win[inst])
        return [x for x in calendars[inst] if a <= x <= b]

    es_s = {sid: es95(per_contract_daily(trades, sid, is_dates(sleeve_inst[sid]))) for sid in SLEEVES}
    a, b = map(d, spec["periods"]["IS_common"]["range"])
    common = union_sessions(calendars, a, b)
    units = np.array([s.lot_units for s in specs])
    pcd_common = np.column_stack([per_contract_daily(trades, sid, common).to_numpy() for sid in SLEEVES])
    es_arr = np.array([es_s[sid] for sid in SLEEVES])
    w = es_arr.max() / es_arr
    b1 = spec["layer_B_risk"]["B1_static_budget"]
    grid = np.round(np.arange(1.0, 10.0 + 1e-9, 0.05), 2)
    mll = spec["topstep_50k_rules"]["combine"]["mll"]["distance"]
    cap_lots = spec["topstep_50k_rules"]["combine"]["max_lots"]
    cache = {}
    sizing = {}
    for k in b1["k_candidates"]:
        feasible = []
        for lam in grid:
            n = np.maximum(1, np.floor(lam * w + 0.5)).astype(int)
            key = tuple(n)
            if key not in cache:
                cache[key] = (es95(pcd_common @ n), peak_lots(trades, n, common, units))
            book_es, peak = cache[key]
            if book_es <= k * mll + 1e-9 and peak <= cap_lots + 1e-9:
                feasible.append(lam)
        lam1_ok = 1.0 in feasible
        if not lam1_ok:
            n = np.ones(len(SLEEVES), dtype=int); lam_star = None; fallback = True
        else:
            lam_star = max(feasible)
            n = np.maximum(1, np.floor(lam_star * w + 0.5)).astype(int); fallback = False
        book_es, peak = cache.get(tuple(n)) or (es95(pcd_common @ n), peak_lots(trades, n, common, units))
        inst_es = {}
        for inst in INSTRUMENTS:
            dates_i = is_dates(inst)
            cols = [sid for sid in SLEEVES if sleeve_inst[sid] == inst]
            series = sum(n[SLEEVES.index(sid)] * per_contract_daily(trades, sid, dates_i).to_numpy() for sid in cols)
            inst_es[inst] = es95(series)
        sizing[str(k)] = {"lambda": lam_star, "fallback_all_ones": fallback, "lambda_1_feasible": lam1_ok,
                          "n": dict(zip(SLEEVES, n.tolist())), "book_es95_is_common": book_es,
                          "peak_lots_is_common": peak, "instrument_es95_is": inst_es,
                          "n_feasible_lambdas": len(feasible)}
    # sigma
    sig = {}
    sigma_ref = {}
    for inst in INSTRUMENTS:
        s = instrument_sigma(bars[inst]["close"], calendars[inst])
        sig[inst] = s
        a_i, b_i = map(d, win[inst])
        sigma_ref[inst] = float(np.nanmedian([v for x, v in s.items() if a_i <= x <= b_i]))
    # B2 diagnostic (report only)
    diag = []
    for sid in SLEEVES:
        inst = sleeve_inst[sid]
        dates_i = is_dates(inst)
        pcd = per_contract_daily(trades, sid, dates_i)
        sv = sig[inst].reindex(dates_i)
        q1, q2 = np.nanquantile(sv, [1 / 3, 2 / 3])
        terc = np.where(sv.isna(), "na", np.where(sv <= q1, "low", np.where(sv <= q2, "mid", "high")))
        active = pcd != 0
        for lab in ["low", "mid", "high"]:
            m = (terc == lab)
            diag.append({"sleeve": sid, "tercile": lab, "sessions": int(m.sum()), "active_sessions": int((m & active).sum()),
                         "pnl_per_contract_sum": float(pcd[m].sum()),
                         "pnl_per_contract_mean_active": float(pcd[m & active].mean()) if (m & active).any() else None})
    s01 = trades[(trades.sleeve == "S01") & (trades.session_date <= d(win["MNQ"][1])) & (trades.session_date >= d(win["MNQ"][0]))]
    s01_change = {"native_net_usd": float(s01.net_native.sum()), "one_unit_net_usd": float(s01.net_per_contract.sum()),
                  "native_mean_qty": float(s01.qty_native.mean()), "trades": int(len(s01))}
    return {"ES95_per_contract_sleeve_is": es_s, "weights": dict(zip(SLEEVES, w.tolist())), "sizing_by_k": sizing,
            "sigma_ref": sigma_ref, "b2_diagnostic": diag, "s01_sizing_change_is": s01_change,
            "is_common_sessions": len(common)}, sig


# ----------------------------------------------------------------------------- configurations
@dataclass(frozen=True)
class Config:
    sizing: str  # "B0" or "B1"
    k: float | None = None
    b2: bool = False
    mbt_cap: int | None = None
    q: float | None = None
    Y: float | None = None
    lock: bool = False
    mbt_lot_units: int = 10

    @property
    def name(self) -> str:
        if self.sizing == "B0":
            parts = ["B0"]
        else:
            parts = [f"B1k{self.k:.2f}"]
        if self.b2:
            parts.append("B2")
        if self.mbt_cap is not None:
            parts.append(f"mbtcap{self.mbt_cap}")
        if self.q is not None:
            parts.append(f"istop{self.q}")
        if self.Y is not None:
            parts.append(f"bstop{int(self.Y)}")
        if self.lock:
            parts.append("lock")
        if self.mbt_lot_units != 10:
            parts.append(f"mbtunits{self.mbt_lot_units}")
        return "+".join(parts)

    @property
    def table_key(self):
        return replace(self, lock=False)


def trade_sizes(cfg: Config, trades: pd.DataFrame, est: dict, sig: dict) -> np.ndarray:
    if cfg.sizing == "B0":
        return np.where(trades.sleeve == "S01", trades.qty_native, 1).astype(int)
    n_s = est["sizing_by_k"][str(cfg.k)]["n"]
    base = trades.sleeve.map(n_s).to_numpy().astype(int)
    if not cfg.b2:
        return base
    m = np.ones(len(trades))
    for inst in INSTRUMENTS:
        sel = (trades.instrument == inst).to_numpy()
        sv = sig[inst].reindex(trades.session_date[sel]).to_numpy()
        ratio = est["sigma_ref"][inst] / sv
        m[sel] = np.where(np.isnan(sv), 1.0, np.clip(ratio, 0.5, 1.5))
    n = np.floor(base * m + 0.5)
    return np.clip(n, 1, np.ceil(1.5 * base)).astype(int)


def limits_for(cfg: Config, cap_units: int | None, est: dict) -> RiskLimits:
    stops = ()
    if cfg.q is not None:
        ref_k = cfg.k if cfg.sizing == "B1" else 0.40
        ies = est["sizing_by_k"][str(ref_k)]["instrument_es95_is"]
        stops = tuple(cfg.q * ies[i] for i in INSTRUMENTS)
    return RiskLimits(lot_cap_units=cap_units, instrument_caps=(None, None, cfg.mbt_cap),
                      instrument_stop_usd=stops, book_stop_usd=cfg.Y)


class Evaluator:
    def __init__(self, spec, trades, cube, dates, est, sig, caps, paths):
        self.spec, self.trades, self.cube, self.dates = spec, trades, cube, dates
        self.est, self.sig, self.caps, self.paths = est, sig, caps, paths
        self.tables: dict = {}
        self.extra: dict = {}

    def tables_for(self, cfg: Config) -> dict:
        key = cfg.table_key
        if key not in self.tables:
            specs = instrument_specs(self.spec, cfg.mbt_lot_units)
            events = build_events(self.trades, trade_sizes(cfg, self.trades, self.est, self.sig))
            out, extra = {}, {}
            for u in self.caps:
                res = simulate_sessions(self.cube, self.dates, events, specs, limits_for(cfg, u, self.est))
                out[u] = SessionTable.from_results(res)
                extra[u] = {"peak_lots": max(r.peak_units for r in res) / 10.0,
                            "instrument_stop_days": [int(sum(r.instrument_stop_hits[i] for r in res)) for i in range(3)],
                            "book_stop_days": int(sum(r.book_stop_hit for r in res)),
                            "contracts": int(sum(r.contracts for r in res))}
            self.tables[key] = out
            self.extra[key] = extra
        return self.tables[key]

    def run(self, cfg: Config, combine=CombineRules(), xfa=XfaRules(), fees=Fees(), combine_only=False) -> dict:
        tables = self.tables_for(cfg)
        out = simulate_lifecycle(self.paths, tables, combine, xfa, fees, profit_lock=cfg.lock,
                                 lock_margin=50.0, combine_only=combine_only)
        summ = summarize(out)
        summ.update(history_metrics(tables[combine.max_lot_units]))
        summ.update({f"cap{u}_{k}": v for u, e in self.extra[cfg.table_key].items() for k, v in e.items()})
        return {"config": cfg, "summary": summ, "net_cash": out["net_cash"]}


def history_metrics(table: SessionTable) -> dict:
    pnl = table.final
    eq = np.cumsum(pnl)
    prev = np.concatenate([[0.0], eq[:-1]])
    peak_prev = np.maximum.accumulate(np.concatenate([[0.0], eq[:-1]]))
    dd = float(np.max(peak_prev - (prev + table.min_worst)))
    sd = pnl.std(ddof=1)
    return {"hist_mean_daily": float(pnl.mean()), "hist_sharpe": float(pnl.mean() / sd * math.sqrt(252)) if sd > 0 else None,
            "hist_intraday_max_dd": dd, "hist_es95_daily": es95(pnl), "hist_total": float(pnl.sum())}


def compare(cand: dict, inc: dict, rule: dict) -> dict:
    m_inc = inc["summary"]["expected_net_cash"]
    m_c = cand["summary"]["expected_net_cash"]
    delta = m_c - m_inc
    diff = cand["net_cash"] - inc["net_cash"]
    nz = np.abs(diff) > 1e-9
    win = float((diff[nz] > 0).mean()) if nz.any() else 0.0
    dx = cand["summary"]["p_xfa_lost"] - inc["summary"]["p_xfa_lost"]
    keep = bool(delta > 0 and delta >= 0.10 * abs(m_inc) and win >= 0.55 and dx <= 0.02 + 1e-12)
    return {"candidate": cand["config"].name, "incumbent": inc["config"].name, "delta_primary": delta,
            "relative_delta": delta / abs(m_inc) if m_inc != 0 else None, "paired_win_fraction": win,
            "paths_differing": int(nz.sum()), "delta_p_xfa_lost": dx, "keep": keep}


def pbo_cscv(M: np.ndarray, blocks: int = 10) -> dict:
    T, N = M.shape
    edges = np.linspace(0, T, blocks + 1).astype(int)
    idx = [np.arange(edges[b], edges[b + 1]) for b in range(blocks)]

    def metric(rows):
        x = M[rows]
        out = np.empty(N)
        for j in range(N):
            e = es95(x[:, j])
            out[j] = x[:, j].mean() / e if e > 0 else np.inf
        return out

    logits = []
    for combo in itertools.combinations(range(blocks), blocks // 2):
        is_rows = np.concatenate([idx[b] for b in combo])
        oos_rows = np.concatenate([idx[b] for b in range(blocks) if b not in combo])
        best = int(np.argmax(metric(is_rows)))
        oos = metric(oos_rows)
        rank = (pd.Series(oos).rank(method="average").to_numpy())[best]
        omega = rank / (N + 1)
        logits.append(math.log(omega / (1 - omega)))
    logits = np.array(logits)
    return {"pbo": float(np.mean(logits <= 0)), "splits": len(logits), "configs": N,
            "median_logit": float(np.median(logits))}


# ----------------------------------------------------------------------------- subcommands
def common_setup(spec):
    checked = verify_inputs(spec)
    trades = load_trades(spec)
    calendars, cal_hashes = load_calendars(spec)
    for inst in INSTRUMENTS:
        missing = set(trades.session_date[trades.instrument == inst]) - set(calendars[inst])
        if missing:
            raise SystemExit(f"{inst}: {len(missing)} trade dates outside the calendar")
    return checked, trades, calendars, cal_hashes


def require_selftest():
    p = RESULTS / "selftest_report.json"
    if not p.exists():
        raise SystemExit("run selftest first")
    r = json.loads(p.read_text(encoding="utf-8"))
    if not r.get("passed"):
        raise SystemExit("selftest did not pass")
    if r["code_sha256"] != code_hashes():
        raise SystemExit("code changed since the passing selftest: rerun selftest")
    return r


def cmd_selftest(spec):
    report = {"started": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "code_sha256": code_hashes()}
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", str(TEST_FILE)], cwd=ROOT, capture_output=True, text=True)
    report["pytest_returncode"] = proc.returncode
    report["pytest_tail"] = proc.stdout.strip().splitlines()[-3:]
    checked, trades, calendars, cal_hashes = common_setup(spec)
    report["inputs_sha256"] = checked
    report["calendar_sha256"] = cal_hashes
    a = d(spec["periods"]["IS_by_sleeve"]["MNQ_MGC"][0])
    b = d(spec["periods"]["IS_by_sleeve"]["MNQ_MGC"][1])
    dates = union_sessions(calendars, a, b)
    trades_is = trades[(trades.session_date >= a) & (trades.session_date <= b)].reset_index(drop=True)
    bars = load_bars(spec)
    cube = BarCube(bars, dates)
    specs = instrument_specs(spec)

    # T1 netting, B0 sizing, no limits
    n_b0 = np.where(trades_is.sleeve == "S01", trades_is.qty_native, 1)
    res = simulate_sessions(cube, dates, build_events(trades_is, n_b0), specs, RiskLimits())
    sleeve_gross = trades_is.groupby("session_date").gross_native.sum().reindex(dates, fill_value=0.0).to_numpy()
    sleeve_cost = trades_is.groupby("session_date").cost_native.sum().reindex(dates, fill_value=0.0).to_numpy()
    gross = np.array([r.gross for r in res]); costs = np.array([r.costs for r in res])
    t1 = {"sessions": len(dates), "max_abs_gross_diff": float(np.max(np.abs(gross - sleeve_gross))),
          "cost_violations": int(np.sum(costs > sleeve_cost + 1e-6)), "netted_costs": float(costs.sum()),
          "sleeve_costs": float(sleeve_cost.sum())}
    t1["passed"] = t1["max_abs_gross_diff"] <= 0.01 and t1["cost_violations"] == 0

    # T2 S01 alone at native size reproduces every native trade
    s01 = trades_is[trades_is.sleeve == "S01"].reset_index(drop=True)
    res2 = simulate_sessions(cube, sorted(set(s01.session_date)), build_events(s01, s01.qty_native.to_numpy()),
                             specs, RiskLimits(), keep_paths=True)
    fills = [f for r in res2 for f in r.fills]
    fills_by_session = []
    for r, date in zip(res2, sorted(set(s01.session_date))):
        for f in r.fills:
            fills_by_session.append((date, f["minute"], f["delta"], round(f["fill"], 6)))
    expected = []
    for _, t in s01.iterrows():
        expected.append((t.session_date, int(t.m_entry), int(t.qty_native * t.direction), round(t.native_entry_fill, 6)))
        expected.append((t.session_date, int(t.m_exit), int(-t.qty_native * t.direction), round(t.native_exit_fill, 6)))
    net_native = s01.groupby("session_date").net_native.sum()
    net_sim = pd.Series([r.final for r in res2], index=sorted(set(s01.session_date)))
    t2 = {"trades": len(s01), "fills": len(fills), "fills_match": sorted(fills_by_session) == sorted(expected),
          "max_abs_net_diff": float((net_sim - net_native).abs().max())}
    t2["passed"] = t2["fills_match"] and t2["max_abs_net_diff"] <= 0.01

    # T3 real data under the tightest XFA tier and an MBT cap (engine asserts on any breach)
    lim = RiskLimits(lot_cap_units=20, instrument_caps=(None, None, 1))
    res3 = simulate_sessions(cube, dates, build_events(trades_is, n_b0), specs, lim)
    t3 = {"peak_units": max(r.peak_units for r in res3), "cap_units": 20}
    t3["passed"] = t3["peak_units"] <= 20

    # T5 sigma uses only past sessions
    cut = d("2022-06-15")
    s_full = instrument_sigma(bars["MNQ"]["close"], calendars["MNQ"])
    pert = bars["MNQ"]["close"].copy()
    after = pert.index >= pd.Timestamp(f"{cut} 00:00", tz=ET).tz_convert("UTC")
    pert[after] = pert[after] * 1.5
    s_pert = instrument_sigma(pert, calendars["MNQ"])
    keys = [x for x in calendars["MNQ"] if x <= cut]
    t5 = {"sessions_checked": len(keys),
          "identical": bool(np.allclose(s_full[keys].to_numpy(), s_pert[keys].to_numpy(), equal_nan=True))}
    t5["passed"] = t5["identical"]

    report.update({"T1_netting": t1, "T2_S01": t2, "T3_caps_real": t3, "T5_sigma_no_lookahead": t5})
    report["passed"] = bool(proc.returncode == 0 and t1["passed"] and t2["passed"] and t3["passed"] and t5["passed"])
    write_json(RESULTS / "selftest_report.json", report)
    print(json.dumps({k: report[k] for k in ["pytest_returncode", "pytest_tail", "T1_netting", "T2_S01", "T3_caps_real",
                                              "T5_sigma_no_lookahead", "passed"]}, indent=2, default=_json_default))
    return report


def cmd_estimate(spec, quiet=False):
    require_selftest()
    _, trades, calendars, _ = common_setup(spec)
    bars = load_bars(spec)
    est, sig = estimate(spec, trades, calendars, bars, instrument_specs(spec))
    est["code_sha256"] = code_hashes()
    write_json(RESULTS / "is" / "estimates.json", est)
    if not quiet:
        print(json.dumps({k: est[k] for k in ["ES95_per_contract_sleeve_is", "sizing_by_k", "sigma_ref"]}, indent=2,
                         default=_json_default))
    return est, sig, trades, calendars, bars


def cmd_is(spec):
    est, sig, trades, calendars, bars = cmd_estimate(spec, quiet=True)
    a, b = map(d, spec["periods"]["IS_common"]["range"])
    dates = union_sessions(calendars, a, b)
    cube = BarCube(bars, dates)
    bs = spec["evaluation"]["bootstrap"]
    paths = block_bootstrap_paths(len(dates), bs["paths"], bs["horizon_traded_sessions"], (3, 5, 10, 20), bs["seed"])
    ev = Evaluator(spec, trades, cube, dates, est, sig, caps=(20, 30, 50), paths=paths)
    rule = spec["evaluation"]["sequential_selection"]
    rows, steps = [], []

    def record(r):
        rows.append({"config": r["config"].name, **asdict(r["config"]), **r["summary"]})
        return r

    b0 = record(ev.run(Config("B0")))
    inc = record(ev.run(Config("B1", k=0.40)))
    evaluated = {inc["config"].name: inc}
    plan = [
        ("B1_k", lambda c: [replace(c, k=0.30), replace(c, k=0.50)]),
        ("B2", lambda c: [replace(c, b2=True)]),
        ("mbt_net_cap", lambda c: [replace(c, mbt_cap=1), replace(c, mbt_cap=2)]),
        ("instrument_daily_stop", lambda c: [replace(c, q=1.0), replace(c, q=1.5)]),
        ("book_daily_stop", lambda c: [replace(c, Y=1000.0)]),
        ("consistency_profit_lock", lambda c: [replace(c, lock=True)]),
    ]
    for step, gen in plan:
        comps = []
        best = None
        for cfg in gen(inc["config"]):
            r = evaluated.get(cfg.name) or record(ev.run(cfg))
            evaluated[cfg.name] = r
            c = compare(r, inc, rule)
            comps.append(c)
            if c["keep"] and (best is None or r["summary"]["expected_net_cash"] > best["summary"]["expected_net_cash"]):
                best = r
        steps.append({"step": step, "incumbent_before": inc["config"].name, "comparisons": comps,
                      "kept": best["config"].name if best else None})
        if best is not None:
            inc = best
    # CSCV on the evaluated table keys (Combine cap, no lock)
    keys, cols = [], []
    for name, r in evaluated.items():
        k = r["config"].table_key
        if k not in keys:
            keys.append(k)
            cols.append(ev.tables_for(r["config"])[50].final)
    cscv = pbo_cscv(np.column_stack(cols), spec["evaluation"]["cscv"]["blocks"])
    cscv["config_names"] = [k.name for k in keys]
    retained = inc
    fallback_used = False
    if cscv["pbo"] > spec["evaluation"]["cscv"]["fallback_if_pbo_above"]:
        retained = evaluated[Config("B1", k=0.40).name]
        fallback_used = True
    # sensitivities (reported only)
    sens = {}
    for label, r in [("retained", retained), ("B0", b0)]:
        cfg = r["config"]
        sens[f"{label}_no_activation_fee"] = ev.run(cfg, fees=Fees(95.0, 95.0, 0.0, 21))["summary"]
        sens[f"{label}_mbt_0.1_lot"] = ev.run(replace(cfg, mbt_lot_units=1))["summary"]
    ev100 = Evaluator(spec, trades, cube, dates, est, sig, caps=(100,), paths=paths)
    c100 = CombineRules(max_lot_units=100, profit_target=6000.0, mll=3000.0)
    for label, r in [("retained", retained), ("B0", b0)]:
        sens[f"{label}_100k_combine_only"] = ev100.run(r["config"], combine=c100, combine_only=True)["summary"]
    selection = {"code_sha256": code_hashes(), "sessions": len(dates), "window": [str(a), str(b)],
                 "b0": b0["summary"], "steps": steps, "incumbent_after_steps": inc["config"].name,
                 "cscv": cscv, "fallback_used": fallback_used, "retained": asdict(retained["config"]),
                 "retained_name": retained["config"].name, "retained_summary": retained["summary"],
                 "sensitivities": sens,
                 "written": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    pd.DataFrame(rows).to_csv(RESULTS / "is" / "configs.csv", index=False)
    write_json(RESULTS / "is" / "selection.json", selection)
    print(json.dumps({"steps": steps, "retained": selection["retained_name"], "pbo": cscv["pbo"],
                      "fallback": fallback_used, "b0": b0["summary"], "retained_summary": retained["summary"]},
                     indent=2, default=_json_default))


def cmd_oos(spec):
    st = require_selftest()
    sel_p = RESULTS / "is" / "selection.json"
    out_p = RESULTS / "oos" / "oos_results.json"
    if not sel_p.exists():
        raise SystemExit("IS selection missing")
    if out_p.exists():
        raise SystemExit("OOS already read; refusing a second read")
    sel = json.loads(sel_p.read_text(encoding="utf-8"))
    if sel["code_sha256"] != code_hashes() or st["code_sha256"] != code_hashes():
        raise SystemExit("code changed since the IS selection")
    est = json.loads((RESULTS / "is" / "estimates.json").read_text(encoding="utf-8"))
    _, trades, calendars, _ = common_setup(spec)
    bars = load_bars(spec)
    sig = {inst: instrument_sigma(bars[inst]["close"], calendars[inst]) for inst in INSTRUMENTS}
    a, b = map(d, spec["periods"]["OOS"]["range"])
    dates = union_sessions(calendars, a, b)
    cube = BarCube(bars, dates)
    bs = spec["evaluation"]["bootstrap"]
    paths = block_bootstrap_paths(len(dates), bs["paths"], bs["horizon_traded_sessions"], (3, 5, 10, 20), bs["seed"])
    ev = Evaluator(spec, trades, cube, dates, est, sig, caps=(20, 30, 50), paths=paths)
    retained = Config(**sel["retained"])
    r_ret = ev.run(retained)
    r_b0 = ev.run(Config("B0"))
    c = compare(r_ret, r_b0, None)
    adopt = (r_ret["summary"]["expected_net_cash"] > r_b0["summary"]["expected_net_cash"]
             and c["delta_p_xfa_lost"] <= 0.02 + 1e-12)
    verdict = "ADOPT" if adopt else "ADOPT_AGGREGATION_ONLY"
    out = {"code_sha256": code_hashes(), "window": [str(a), str(b)], "sessions": len(dates),
           "contaminated": True, "retained": retained.name, "retained_summary": r_ret["summary"],
           "b0_summary": r_b0["summary"], "comparison_vs_b0": c, "verdict": verdict,
           "ext": "not computed: frozen MNQ/MGC trade inputs end on 2026-03-19",
           "read_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    write_json(out_p, out)
    print(json.dumps(out, indent=2, default=_json_default))


def cmd_report(spec):
    lines = ["# book_aggregation_risk_r1_20261112 — résultats", ""]
    for name in ["selftest_report.json", "is/estimates.json", "is/selection.json", "oos/oos_results.json"]:
        p = RESULTS / name
        lines.append(f"- `{name}` : {'présent' if p.exists() else 'absent'}")
    sel_p = RESULTS / "is" / "selection.json"
    if sel_p.exists():
        sel = json.loads(sel_p.read_text(encoding="utf-8"))
        lines += ["", f"Configuration retenue sur l'IS : `{sel['retained_name']}` (PBO {sel['cscv']['pbo']:.2f}).", ""]
        for s in sel["steps"]:
            lines.append(f"- {s['step']} : gardé = {s['kept']}")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["selftest", "estimate", "is", "oos", "report"])
    args = ap.parse_args()
    spec = load_spec()
    RESULTS.mkdir(exist_ok=True)
    {"selftest": cmd_selftest, "estimate": cmd_estimate, "is": cmd_is, "oos": cmd_oos, "report": cmd_report}[args.command](spec)


if __name__ == "__main__":
    main()
