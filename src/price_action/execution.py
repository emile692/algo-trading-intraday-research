"""Execution layer: bracket (stop + fixed R target) simulated on 1-minute bars after the signal-bar close.

- Entry at the signal-bar close (`entry_ref`); net fill = entry_ref + 1 tick against the trade.
- Stop and target levels are fixed at signal time from entry_ref: risk = |entry_ref - stop|, target = entry_ref +
  side * target_r * risk, rounded outward to the tick (never closer than target_r * risk).
- Path: 1m bars starting at or after the signal time.  A bar that OPENS beyond the stop fills the stop at its open
  (gap); a bar that opens beyond the target fills the target at the target (limit, no price improvement).  If a bar
  touches both levels, the STOP is counted (1m is the finest data available, so the order cannot be resolved).
- Flat before the Topstep close: last bar used is the one closing at flat_time_ct - buffer (CT = ET - 1 h); time exit
  at its close.  No position is ever carried overnight.
- Costs in ticks, not in R: entry 1 tick, stop 2 ticks, target 0 tick, time exit 1 tick, + round-trip commission.
- Trades whose planned stop is smaller than `min_stop_ticks` are rejected (status 'rejected_min_stop').
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import ContractSpec


@dataclass(frozen=True)
class CostModel:
    commission_rt_usd: float = 2.50
    slip_entry_ticks: float = 1.0
    slip_stop_ticks: float = 2.0
    slip_target_ticks: float = 0.0
    slip_time_ticks: float = 1.0


@dataclass(frozen=True)
class ExecConfig:
    target_r: float = 2.0
    min_stop_ticks: float = 8.0
    flat_time_ct: str = "15:10"
    flat_buffer_min: int = 1


def last_bar_close_minute_et(cfg: ExecConfig) -> int:
    h, m = map(int, cfg.flat_time_ct.split(":"))
    return (h + 1) * 60 + m - cfg.flat_buffer_min  # ET minute at which the position must be flat


def minutes_by_date(df: pd.DataFrame) -> dict:
    """date -> (start_ns int64, minute, open, high, low, close) numpy arrays, for fast path scans."""
    out = {}
    ns = df.index.asi8
    cols = {k: df[k].to_numpy(float) for k in ("open", "high", "low", "close")}
    minute = df["minute"].to_numpy()
    codes, uniques = pd.factorize(df["date"], sort=True)
    bounds = np.flatnonzero(np.diff(codes)) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(codes)]
    for k, (a, b) in enumerate(zip(starts, ends)):
        out[uniques[codes[a]]] = (ns[a:b], minute[a:b], cols["open"][a:b], cols["high"][a:b],
                                  cols["low"][a:b], cols["close"][a:b])
    return out


def simulate(signals: pd.DataFrame, mins: dict, spec: ContractSpec, cfg: ExecConfig = ExecConfig(),
             costs: CostModel = CostModel()) -> pd.DataFrame:
    tick, pv = spec.tick_size, spec.point_value
    flat_min = last_bar_close_minute_et(cfg)
    rows = []
    for s in signals.itertuples(index=False):
        side = int(s.side)
        risk_pts = abs(s.entry_ref - s.stop)
        risk_ticks = risk_pts / tick
        base = {"date": s.date, "setup": s.setup, "side": side, "signal_time": s.signal_time,
                "entry_ref": s.entry_ref, "stop": s.stop, "risk_ticks": risk_ticks, "risk_usd": risk_pts * pv}
        if risk_ticks + 1e-9 < cfg.min_stop_ticks:
            rows.append({**base, "status": "rejected_min_stop"})
            continue
        raw_target = s.entry_ref + side * cfg.target_r * risk_pts
        target = (math.ceil(raw_target / tick - 1e-9) if side == 1 else math.floor(raw_target / tick + 1e-9)) * tick
        t_ns, minute, o, h, l, c = mins[s.date]
        sel = (t_ns >= pd.Timestamp(s.signal_time).value) & (minute + 1 <= flat_min)
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            rows.append({**base, "status": "no_bar_after_signal"})
            continue
        reason, gross_exit, k_exit = None, None, idx[-1]
        for k in idx:
            if side == 1:
                if o[k] <= s.stop:
                    reason, gross_exit = "stop", o[k]
                elif o[k] >= target:
                    reason, gross_exit = "target", target
                elif l[k] <= s.stop:
                    reason, gross_exit = "stop", s.stop
                elif h[k] >= target:
                    reason, gross_exit = "target", target
            else:
                if o[k] >= s.stop:
                    reason, gross_exit = "stop", o[k]
                elif o[k] <= target:
                    reason, gross_exit = "target", target
                elif h[k] >= s.stop:
                    reason, gross_exit = "stop", s.stop
                elif l[k] <= target:
                    reason, gross_exit = "target", target
            if reason is not None:
                k_exit = k
                break
        if reason is None:
            reason, gross_exit = "time", c[k_exit]
        slip_exit = {"stop": costs.slip_stop_ticks, "target": costs.slip_target_ticks,
                     "time": costs.slip_time_ticks}[reason] * tick
        net_entry = s.entry_ref + side * costs.slip_entry_ticks * tick
        net_exit = gross_exit - side * slip_exit
        gross_usd = side * (gross_exit - s.entry_ref) * pv
        net_usd = side * (net_exit - net_entry) * pv - costs.commission_rt_usd
        rows.append({**base, "status": "filled", "target": target, "exit_reason": reason,
                     "exit_time": pd.Timestamp(int(t_ns[k_exit]), tz="UTC").tz_convert("America/New_York")
                     + pd.Timedelta(minutes=1),
                     "gross_exit": gross_exit, "pnl_gross_usd": gross_usd, "pnl_net_usd": net_usd,
                     "r_gross": gross_usd / (risk_pts * pv), "r_net": net_usd / (risk_pts * pv)})
    return pd.DataFrame(rows)
