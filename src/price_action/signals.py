"""Signal layer.  Every signal is emitted at the END of a closed signal bar and uses only bars closed at that time.

Both functions return at most ONE candidate per day (the first in time); the execution layer applies the
minimum-stop rule to that candidate only (a rejected first candidate means no trade that day: no re-search).
Output columns: date, setup, side (+1 long / -1 short), signal_time (tz ET), entry_ref (signal-bar close), stop,
iid, detail columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["date", "setup", "side", "signal_time", "entry_ref", "stop", "iid"]


def _window_end_min(window_end: str) -> int:
    h, m = map(int, window_end.split(":"))
    return h * 60 + m


def inside_day_false_break(bars: pd.DataFrame, daily: pd.DataFrame, tick: float,
                           window_end: str = "11:00") -> pd.DataFrame:
    """Setup A.  J-1 inside day vs J-2 (strict), all three sessions valid and on the same contract.
    From the 09:30 bar of J: a bar whose high > H(J-1) starts an upside break (a gap open above counts); the break
    extreme is the highest high from that bar on.  The first bar whose close is strictly inside (L1 < close < H1)
    after (or on) the break bar is the signal: short at its close, stop = break extreme + 1 tick.  Mirror for the
    downside.  A bar that would signal both sides is ambiguous: the day is skipped.  Signal bar must close <= end."""
    end_min = _window_end_min(window_end)
    d = daily
    prev1 = d.shift(1)
    prev2 = d.shift(2)
    # Only J-1 and J-2 (complete sessions) are used here: nothing about day J itself, which is not finished.
    ok = (prev1["valid"].eq(True) & prev2["valid"].eq(True)
          & prev1["iid"].eq(prev2["iid"]) & (prev1["high"] < prev2["high"]) & (prev1["low"] > prev2["low"]))
    setup_days = d.index[ok.fillna(False)]
    h1, l1, iid1 = prev1["high"], prev1["low"], prev1["iid"]
    out = []
    for date, b in bars[bars["date"].isin(setup_days)].groupby("date", sort=True):
        b = b[b["end_min"] <= end_min]
        if b.empty:
            continue
        H1, L1 = float(h1.loc[date]), float(l1.loc[date])
        up_ext = dn_ext = None
        for row in b.itertuples(index=False):
            if int(row.iid) != int(iid1.loc[date]):
                break  # day J on another contract than J-1/J-2: levels not comparable
            if row.high > H1:
                up_ext = row.high if up_ext is None else max(up_ext, row.high)
            if row.low < L1:
                dn_ext = row.low if dn_ext is None else min(dn_ext, row.low)
            inside = L1 < row.close < H1
            short = inside and up_ext is not None
            long = inside and dn_ext is not None
            if short and long:
                break  # ambiguous day
            if short or long:
                side = -1 if short else 1
                stop = up_ext + tick if short else dn_ext - tick
                out.append({"date": date, "setup": "A_inside_day_false_break", "side": side,
                            "signal_time": row.end, "entry_ref": float(row.close), "stop": float(stop),
                            "iid": int(row.iid), "prev_high": H1, "prev_low": L1})
                break
    return pd.DataFrame(out, columns=COLUMNS + ["prev_high", "prev_low"])


def _three_push_day(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int, end_ok: np.ndarray):
    """Scan one day.  Returns (bar index j, side, extreme bar index p3) of the first trigger, or None.

    Bearish push sequence (gives a SHORT): pivot highs (h[p] > the n highs before, >= the n highs after, confirmed
    at the close of bar p+n) that are new session highs at their time; consecutive pushes must be separated by a
    confirmed pivot low; a new session-high pivot with no pivot low since the last push replaces it.  Trigger at the
    close of bar j >= p3+n when: >= 3 pushes, the last push high is still the session high, and c[j] < l[p3].
    Mirror for a LONG.  The first trigger of either side ends the day."""
    m = len(h)
    cmax, cmin = np.maximum.accumulate(h), np.minimum.accumulate(l)  # prefix extremes: index k uses bars <= k only
    pushes_hi: list[int] = []
    pushes_lo: list[int] = []
    piv_lo: list[int] = []
    piv_hi: list[int] = []
    for j in range(m):
        p = j - n
        if p >= n:
            win_h_left, win_h_right = h[p - n:p], h[p + 1:j + 1]
            win_l_left, win_l_right = l[p - n:p], l[p + 1:j + 1]
            is_ph = h[p] > win_h_left.max() and h[p] >= win_h_right.max()
            is_pl = l[p] < win_l_left.min() and l[p] <= win_l_right.min()
            if is_pl:
                piv_lo.append(p)
            if is_ph:
                piv_hi.append(p)
            if is_ph and h[p] > cmax[p - 1]:
                if pushes_hi and any(pushes_hi[-1] < q < p for q in piv_lo):
                    pushes_hi.append(p)
                elif pushes_hi:
                    pushes_hi[-1] = p
                else:
                    pushes_hi.append(p)
            if is_pl and l[p] < cmin[p - 1]:
                if pushes_lo and any(pushes_lo[-1] < q < p for q in piv_hi):
                    pushes_lo.append(p)
                elif pushes_lo:
                    pushes_lo[-1] = p
                else:
                    pushes_lo.append(p)
        if not end_ok[j]:
            break  # bars are in time order: no later bar can close inside the entry window
        short = (len(pushes_hi) >= 3 and j >= pushes_hi[-1] + n and cmax[j] == h[pushes_hi[-1]]
                 and c[j] < l[pushes_hi[-1]])
        long = (len(pushes_lo) >= 3 and j >= pushes_lo[-1] + n and cmin[j] == l[pushes_lo[-1]]
                and c[j] > h[pushes_lo[-1]])
        if short and long:
            return None
        if short:
            return j, -1, pushes_hi[-1]
        if long:
            return j, 1, pushes_lo[-1]
    return None


def three_push(bars: pd.DataFrame, tick: float, n: int = 2, window_end: str = "11:00") -> pd.DataFrame:
    """Setup B on RTH signal bars from 09:30 (session extremes ignore the overnight session)."""
    end_min = _window_end_min(window_end)
    out = []
    for date, b in bars.groupby("date", sort=True):
        same = b["iid"].to_numpy() == b["iid"].iloc[0]
        if not same.all():  # stop at the first contract change (known bar by bar, no look-ahead)
            b = b.iloc[: int(np.argmin(same))]
        h, l, c = b["high"].to_numpy(float), b["low"].to_numpy(float), b["close"].to_numpy(float)
        end_ok = b["end_min"].to_numpy() <= end_min
        hit = _three_push_day(h, l, c, n, end_ok)
        if hit is None:
            continue
        j, side, p3 = hit
        stop = h[p3] + tick if side == -1 else l[p3] - tick
        out.append({"date": date, "setup": "B_three_push", "side": side, "signal_time": b["end"].iloc[j],
                    "entry_ref": float(c[j]), "stop": float(stop), "iid": int(b["iid"].iloc[j]),
                    "push3_time": b["end"].iloc[p3]})
    return pd.DataFrame(out, columns=COLUMNS + ["push3_time"])
