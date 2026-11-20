"""Tests of src.price_action: no look-ahead (truncation / poisoning), pattern definitions, execution and costs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.price_action.data import ET, SPECS, add_clock, rth_daily, signal_bars, wilder_atr_prev
from src.price_action.execution import CostModel, ExecConfig, minutes_by_date, simulate
from src.price_action.metrics import max_drawdown, mc_drawdown, worst_losing_streak
from src.price_action.signals import inside_day_false_break, three_push

TICK = 0.25
MNQ = SPECS["MNQ"]


def _synthetic(n_days: int = 160, seed: int = 7) -> pd.DataFrame:
    """1m bars 08:00-16:30 ET on business days, random walk with day-specific volatility, one contract."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2022-01-03", periods=n_days)
    frames, price = [], 15000.0
    for d in days:
        idx = pd.date_range(pd.Timestamp(d).tz_localize(ET) + pd.Timedelta(hours=8), periods=510, freq="1min")
        vol = rng.choice([1.0, 2.0, 4.0])
        steps = np.round(rng.normal(0, vol, len(idx)) / TICK) * TICK
        close = price + np.cumsum(steps)
        open_ = np.r_[price, close[:-1]]
        wig_h = np.round(np.abs(rng.normal(0, vol, len(idx))) / TICK) * TICK
        wig_l = np.round(np.abs(rng.normal(0, vol, len(idx))) / TICK) * TICK
        high = np.maximum(open_, close) + wig_h
        low = np.minimum(open_, close) - wig_l
        frames.append(pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 10,
                                    "instrument_id": 1}, index=idx))
        price = close[-1]
    df = pd.concat(frames)
    df.index.name = "timestamp"
    return add_clock(df)


def _signals(df: pd.DataFrame, asof=None) -> dict[str, pd.DataFrame]:
    bars = signal_bars(df, 5, asof=asof)
    daily = rth_daily(df)
    return {"A": inside_day_false_break(bars, daily, TICK, "11:00"),
            "B": three_push(bars, TICK, n=1, window_end="11:00")}


def _poison_after(df: pd.DataFrame, t: pd.Timestamp, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    m = out.index >= t
    noise = np.round(rng.normal(0, 50, m.sum()) / TICK) * TICK
    for col in ("open", "high", "low", "close"):
        out.loc[m, col] = out.loc[m, col] + noise
    out.loc[m, "high"] = out.loc[m, ["open", "high", "low", "close"]].max(axis=1) + 5
    out.loc[m, "low"] = out.loc[m, ["open", "high", "low", "close"]].min(axis=1) - 5
    return out


@pytest.fixture(scope="module")
def synth():
    df = _synthetic()
    sig = _signals(df)
    assert len(sig["A"]) >= 3 and len(sig["B"]) >= 3, "synthetic data must produce signals for the test to bite"
    return df, sig


def test_no_lookahead_poisoning_and_truncation(synth):
    df, full = synth
    rng = np.random.default_rng(1)
    cut_times = []
    for key in ("A", "B"):
        cut_times += list(full[key]["signal_time"].iloc[:6])
    cut_times += list(df.index[rng.integers(0, len(df), 6)])
    for i, t in enumerate(cut_times):
        t = pd.Timestamp(t)
        for variant, data, asof in (("poison", _poison_after(df, t, i), None), ("truncate", df[df.index < t], t)):
            got = _signals(data, asof=asof)
            for key in ("A", "B"):
                exp = full[key][full[key]["signal_time"] <= t].reset_index(drop=True)
                obs = got[key][got[key]["signal_time"] <= t].reset_index(drop=True)
                pd.testing.assert_frame_equal(obs, exp, check_dtype=False, obj=f"{variant} {key} t={t}")


def test_atr_prev_uses_only_previous_sessions(synth):
    df, _ = synth
    daily = rth_daily(df)
    a1 = wilder_atr_prev(daily)
    d2 = daily.copy()
    d2.iloc[-1, d2.columns.get_loc("high")] += 500
    a2 = wilder_atr_prev(d2)
    pd.testing.assert_series_equal(a1, a2)


def _day_bars(day: str, rows: list[tuple[float, float, float, float]], tf: int = 5, iid: int = 1) -> pd.DataFrame:
    """Build 1m bars from a list of 5m (o, h, l, c) starting 09:30: each 5m bar is 5 identical-ish 1m bars."""
    recs = []
    t0 = pd.Timestamp(f"{day} 09:30", tz=ET)
    for k, (o, h, l, c) in enumerate(rows):
        for j in range(tf):
            ts = t0 + pd.Timedelta(minutes=k * tf + j)
            recs.append((ts, o if j == 0 else c, h, l, c))
    df = pd.DataFrame(recs, columns=["timestamp", "open", "high", "low", "close"]).set_index("timestamp")
    df["volume"], df["instrument_id"] = 1, iid
    return df


def _flat_day(day: str, hi: float, lo: float, n: int = 78) -> pd.DataFrame:
    mid = (hi + lo) / 2
    rows = [(mid, hi, lo, mid)] + [(mid, mid + 0.25, mid - 0.25, mid)] * (n - 1)
    return _day_bars(day, rows)


def test_inside_day_false_break_definition():
    j2 = _flat_day("2023-03-06", 110.0, 90.0)
    j1 = _flat_day("2023-03-07", 105.0, 95.0)  # inside day
    rows = [(100, 104, 99, 103), (103, 107, 102, 106.5), (106.5, 108.5, 104, 104.5)]  # break 105, close back 104.5
    rows += [(104.5, 104.75, 104, 104.5)] * 75
    j = _day_bars("2023-03-08", rows)
    df = add_clock(pd.concat([j2, j1, j]))
    sig = inside_day_false_break(signal_bars(df, 5), rth_daily(df), TICK)
    assert len(sig) == 1
    s = sig.iloc[0]
    assert s["side"] == -1 and s["entry_ref"] == 104.5 and s["stop"] == 108.75
    assert s["signal_time"] == pd.Timestamp("2023-03-08 09:45", tz=ET)


def test_inside_day_requires_strict_inside():
    j2 = _flat_day("2023-03-06", 110.0, 90.0)
    j1 = _flat_day("2023-03-07", 110.0, 95.0)  # equal high: not inside
    j = _day_bars("2023-03-08", [(100, 112, 99, 103)] * 78)
    df = add_clock(pd.concat([j2, j1, j]))
    assert inside_day_false_break(signal_bars(df, 5), rth_daily(df), TICK).empty


def test_three_push_definition_n1():
    # n=1: pushes = pivot highs that are session highs, separated by pivot lows.
    rows = [(100, 101, 99, 100.5),   # 0
            (100.5, 103, 100, 102),  # 1 push 1 (pivot high: 103 > 101, >= next)
            (102, 102.5, 100, 101),  # 2 pivot low 100? left 100 not < 100 -> use lower
            (101, 101.5, 99.5, 101), # 3 pivot low (99.5 < 100, <= next)
            (101, 105, 101, 104),    # 4 push 2
            (104, 104.5, 102, 103),  # 5
            (103, 103.5, 101.5, 103),# 6 pivot low
            (103, 107, 103, 106),    # 7 push 3 (low 103)
            (106, 106.5, 104, 104.5),# 8 confirms push 3; close 104.5 > 103: no trigger
            (104.5, 105, 102, 102.5)]  # 9 close 102.5 < low(push3)=103 -> short
    rows += [(102.5, 102.75, 102.25, 102.5)] * 68
    df = add_clock(_day_bars("2023-03-08", rows))
    sig = three_push(signal_bars(df, 5), TICK, n=1)
    assert len(sig) == 1
    s = sig.iloc[0]
    assert s["side"] == -1 and s["stop"] == 107.25 and s["entry_ref"] == 102.5
    assert s["signal_time"] == pd.Timestamp("2023-03-08 10:20", tz=ET)


def _one_signal(side, entry, stop, time="2023-03-08 10:00"):
    return pd.DataFrame([{"date": pd.Timestamp("2023-03-08"), "setup": "T", "side": side,
                          "signal_time": pd.Timestamp(time, tz=ET), "entry_ref": entry, "stop": stop}])


def _path(rows, start="2023-03-08 10:00"):
    idx = pd.date_range(pd.Timestamp(start, tz=ET), periods=len(rows), freq="1min")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"], df["instrument_id"] = 1, 1
    return add_clock(df)


def test_same_bar_stop_and_target_counts_stop():
    df = _path([(100, 111, 94, 100)] + [(100, 100.25, 99.75, 100)] * 10)
    t = simulate(_one_signal(1, 100.0, 95.0), minutes_by_date(df), MNQ)
    assert t.iloc[0]["exit_reason"] == "stop"


def test_target_costs_in_ticks():
    df = _path([(100, 105, 99, 104), (104, 110.25, 103, 110)])
    t = simulate(_one_signal(1, 100.0, 95.0), minutes_by_date(df), MNQ).iloc[0]
    assert t["exit_reason"] == "target" and t["target"] == 110.0
    assert t["pnl_gross_usd"] == pytest.approx(20.0)  # 10 pts * 2 USD
    assert t["pnl_net_usd"] == pytest.approx((110.0 - 100.25) * 2 - 2.50)  # entry +1 tick, target 0, commission
    assert t["r_net"] == pytest.approx(t["pnl_net_usd"] / 10.0)


def test_stop_gap_fills_at_open_plus_two_ticks():
    df = _path([(100, 101, 99, 100), (93, 94, 92, 93)])
    t = simulate(_one_signal(1, 100.0, 95.0), minutes_by_date(df), MNQ).iloc[0]
    assert t["exit_reason"] == "stop" and t["gross_exit"] == 93.0
    assert t["pnl_net_usd"] == pytest.approx((93.0 - 0.5 - 100.25) * 2 - 2.50)


def test_min_stop_rejection_and_time_exit():
    df = _path([(100, 100.5, 99.5, 100)] * 400, start="2023-03-08 10:00")
    sig = pd.concat([_one_signal(1, 100.0, 98.5), _one_signal(1, 100.0, 95.0)], ignore_index=True)
    t = simulate(sig, minutes_by_date(df), MNQ, ExecConfig(min_stop_ticks=8))
    assert t.iloc[0]["status"] == "rejected_min_stop"  # 6 ticks < 8
    assert t.iloc[1]["exit_reason"] == "time"
    assert t.iloc[1]["exit_time"] == pd.Timestamp("2023-03-08 16:09", tz=ET)  # 15:10 CT minus 1 min buffer


def test_metrics_helpers():
    pnl = np.array([1.0, -2.0, -1.0, 3.0, -1.0])
    assert max_drawdown(pnl) == pytest.approx(3.0)
    assert worst_losing_streak(pnl) == 2
    mc = mc_drawdown(pnl, n_perm=2000)
    assert mc["dd_q95"] >= mc["dd_q50"] >= 0
