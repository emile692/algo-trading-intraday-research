"""Active eight-sleeve book on its full sample, split IS / mixed / OOS, and seen as three instrument signals.

Requested on 2026-11-21 for BOOK.md: "trace le book FS en distinguant IS/OOS, considère maintenant qu'on a 3 signaux
indépendants (MNQ, MBT, MGC), donne leur matrice de corr, leur sharpe et graph respectif".

Inputs: the daily sources of the eight sleeves of lock ``eight_sleeve_equal_contract_reference_20261121`` (MNQ on
v.0 back-adjusted), read through their sha256 recorded in the lock config.  Nothing is re-simulated.

- Calendar: union of the session dates of the vote sleeves (as in the 2026-11-15 / 2026-11-21 locks), from the first
  session 2019-05-06 to the end of the frozen research window 2026-03-19.  A sleeve is 0 on a union session its
  instrument does not trade; MBT does not exist before its first session (2021-05-03) and is NaN there.
- Signals: MNQ = S01 + S02 + S03, MGC = S05 + S06, MBT = S07 + S08 + S09 (equal contract, before netting; the netting
  of section 7 changes the window PnL by +28 USD).  Book 8 = sum of the three.
- Segments: a period is IS when every component sleeve is inside its own IS, OOS when every component is past it,
  "mixte" otherwise.  Sleeve IS ends: S01 2024-02-27; S02, S03, S05, S06 2023-12-31; S07-S09 2025-05-28.  All these
  OOS were consulted before the freeze: they are reused, not prospective.
- Sharpe: sqrt(252) * mean / std(ddof=1) of daily USD PnL; a signal on its own instrument calendar, the book on the
  union calendar.  Max drawdown close to close from a zero origin.

Check: on 2024-01-02 -> 2026-03-19 the per-sleeve union daily PnL must equal the lock daily_pnl.csv exactly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCK = ROOT / "research" / "locked_eight_sleeve_portfolio_20261121"
FS_END = pd.Timestamp("2026-03-19")
LOCK_WINDOW = (pd.Timestamp("2024-01-02"), pd.Timestamp("2026-03-19"))

SIGNALS = {
    "MNQ": ("mnq_orb_vwap_atr", "mnq_volatility_breakout", "mnq_compression_breakout"),
    "MGC": ("mgc_volatility_breakout", "mgc_gap_continuation"),
    "MBT": ("mbt_rolling_range_breakout", "mbt_trend_pullback_ma_structure", "mbt_volatility_breakout"),
}
CODES = {
    "mnq_orb_vwap_atr": "S01", "mnq_volatility_breakout": "S02", "mnq_compression_breakout": "S03",
    "mgc_volatility_breakout": "S05", "mgc_gap_continuation": "S06", "mbt_rolling_range_breakout": "S07",
    "mbt_trend_pullback_ma_structure": "S08", "mbt_volatility_breakout": "S09",
}
IS_END = {
    "mnq_orb_vwap_atr": "2024-02-27", "mnq_volatility_breakout": "2023-12-31", "mnq_compression_breakout": "2023-12-31",
    "mgc_volatility_breakout": "2023-12-31", "mgc_gap_continuation": "2023-12-31",
    "mbt_rolling_range_breakout": "2025-05-28", "mbt_trend_pullback_ma_structure": "2025-05-28",
    "mbt_volatility_breakout": "2025-05-28",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sleeves() -> tuple[dict[str, pd.Series], dict[str, pd.DatetimeIndex]]:
    config = json.loads((LOCK / "locked_eight_sleeve_config.json").read_text(encoding="utf-8"))
    daily, calendars = {}, {}
    for sleeve, source in config["sleeve_sources"].items():
        path = ROOT / source["daily_source"]
        if sha256(path) != source["daily_source_sha256"]:
            raise RuntimeError(f"{sleeve}: daily source differs from the lock")
        frame = pd.read_csv(path, parse_dates=["session_date"])
        s = frame.groupby("session_date")["daily_pnl_usd"].sum().astype(float)
        daily[sleeve] = s[s.index <= FS_END]
        calendars[sleeve] = pd.DatetimeIndex(sorted(daily[sleeve].index))
    if set(daily) != set(CODES):
        raise RuntimeError(f"unexpected sleeves: {sorted(daily)}")
    return daily, calendars


def union_calendar(calendars: dict[str, pd.DatetimeIndex]) -> pd.DatetimeIndex:
    vote = [s for s in calendars if s != "mnq_orb_vwap_atr"]  # S01's file also lists non-session calendar days
    dates = sorted(set().union(*[set(calendars[s]) for s in vote]))
    return pd.DatetimeIndex(dates)


def instrument_calendar(signal: str, calendars: dict[str, pd.DatetimeIndex]) -> pd.DatetimeIndex:
    members = [s for s in SIGNALS[signal] if s != "mnq_orb_vwap_atr"]
    return pd.DatetimeIndex(sorted(set().union(*[set(calendars[s]) for s in members])))


def metrics(x: pd.Series) -> dict:
    x = x.dropna().astype(float)
    eq = np.r_[0.0, np.cumsum(x.to_numpy())]
    sd = float(x.std(ddof=1))
    return {
        "start": str(x.index.min().date()), "end": str(x.index.max().date()), "sessions": int(len(x)),
        "net_pnl_usd": float(x.sum()), "daily_vol_usd": sd,
        "sharpe": float(np.sqrt(252.0) * x.mean() / sd) if sd > 0 else float("nan"),
        "max_dd_usd": float((eq - np.maximum.accumulate(eq)).min()), "worst_day_usd": float(x.min()),
        "pnl_per_year_usd": float(x.sum() / len(x) * 252.0),
    }


def segments(members: tuple[str, ...], index: pd.DatetimeIndex) -> pd.Series:
    """IS when every member is inside its IS, OOS when every member is past it, 'mixte' otherwise."""
    inside = np.column_stack([index <= pd.Timestamp(IS_END[m]) for m in members])
    return pd.Series(np.where(inside.all(axis=1), "IS", np.where(~inside.any(axis=1), "OOS", "mixte")), index=index)


def main() -> None:
    daily, calendars = load_sleeves()
    union = union_calendar(calendars)
    s01 = daily["mnq_orb_vwap_atr"]
    off = s01[~s01.index.isin(union)]
    if (off != 0).any():
        raise RuntimeError("S01 has PnL on a session outside the union calendar")

    sleeves = pd.DataFrame(index=union)
    for sleeve in CODES:
        sleeves[sleeve] = daily[sleeve].reindex(union).fillna(0.0)
    mbt_start = calendars["mbt_rolling_range_breakout"].min()
    for sleeve in SIGNALS["MBT"]:
        sleeves.loc[sleeves.index < mbt_start, sleeve] = np.nan

    # Reproduce the lock on its window, per sleeve, to the cent.
    lock = pd.read_csv(LOCK / "daily_pnl.csv", parse_dates=["session_date"]).set_index("session_date")
    window = sleeves.loc[LOCK_WINDOW[0]:LOCK_WINDOW[1]]
    if not window.index.equals(lock.index):
        raise RuntimeError("union calendar differs from the lock on its window")
    gap = float((window[list(CODES)] - lock[list(CODES)]).abs().to_numpy().max())
    if gap > 1e-6:
        raise RuntimeError(f"sleeve daily PnL differs from the lock by {gap}")

    signals = pd.DataFrame(index=union)
    for name, members in SIGNALS.items():
        signals[name] = sleeves[list(members)].sum(axis=1, min_count=len(members))
    signals["Book 8"] = signals[list(SIGNALS)].sum(axis=1, min_count=1)
    if abs(float(signals.loc[LOCK_WINDOW[0]:LOCK_WINDOW[1], "Book 8"].sum()) - float(lock["portfolio_pnl_usd"].sum())) > 1e-6:
        raise RuntimeError("book 8 differs from the lock on its window")

    # Segment labels: each signal with its own members, the book with all eight sleeves.
    labels = pd.DataFrame(index=union)
    for name, members in SIGNALS.items():
        labels[name] = segments(members, union)
    labels["Book 8"] = segments(tuple(CODES), union)

    out = signals.copy()
    for col in labels:
        out[f"segment_{col}"] = labels[col]
    for name in SIGNALS:  # instrument sessions (a signal is NaN on union sessions its instrument does not trade)
        cal = instrument_calendar(name, calendars)
        out[f"session_{name}"] = union.isin(cal)
    out.index.name = "session_date"
    out.to_csv(HERE / "daily_signals.csv", float_format="%.8f")

    rows = []
    for name in [*SIGNALS, "Book 8"]:
        if name in SIGNALS:
            series = signals[name].where(out[f"session_{name}"])
        else:
            series = signals[name]
        for seg in ("FS", "IS", "mixte", "OOS"):
            s = series if seg == "FS" else series[labels[name] == seg]
            s = s.dropna()
            if len(s) == 0:
                continue
            rows.append({"series": name, "segment": seg, **metrics(s)})
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "signal_metrics.csv", index=False, float_format="%.6f")

    # Correlations: union calendar, 0 when an instrument does not trade that day, on the period where all three exist.
    common = signals.loc[mbt_start:, list(SIGNALS)].fillna(0.0)
    periods = {
        "FS commun": (mbt_start, FS_END),
        "2021-05-03 -> 2023-12-31": (mbt_start, pd.Timestamp("2023-12-31")),
        "2024-01-02 -> 2026-03-19": LOCK_WINDOW,
    }
    corr = {}
    for label, (a, b) in periods.items():
        c = common.loc[a:b].corr()
        corr[label] = {"start": str(common.loc[a:b].index.min().date()), "end": str(common.loc[a:b].index.max().date()),
                       "sessions": int(len(common.loc[a:b])), "matrix": c.round(6).to_dict()}
    fs_corr = common.corr()
    fs_corr.to_csv(HERE / "correlation_fs_common.csv", float_format="%.6f")

    # What independence would give: book Sharpe if the three signals had zero correlation (same means and vols).
    mu, sd = common.mean(), common.std(ddof=1)
    book_common = common.sum(axis=1)
    sharpe_actual = float(np.sqrt(252) * book_common.mean() / book_common.std(ddof=1))
    sharpe_indep = float(np.sqrt(252) * mu.sum() / np.sqrt((sd ** 2).sum()))
    # Same check per sleeve: book Sharpe if the eight sleeves were uncorrelated.
    sl = sleeves.loc[mbt_start:, list(CODES)].fillna(0.0)
    sharpe_indep_sleeves = float(np.sqrt(252) * sl.mean().sum() / np.sqrt((sl.std(ddof=1) ** 2).sum()))

    result = {
        "purpose": "BOOK.md section 6: active eight-sleeve book on its full sample (IS / mixte / OOS) and as three instrument signals",
        "lock": "eight_sleeve_equal_contract_reference_20261121",
        "fs_window": [str(union.min().date()), str(FS_END.date())],
        "union_sessions": int(len(union)),
        "mbt_first_session": str(mbt_start.date()),
        "lock_window_reproduced": {"window": [str(LOCK_WINDOW[0].date()), str(LOCK_WINDOW[1].date())],
                                   "max_abs_sleeve_gap_usd": gap,
                                   "book8_pnl_usd": float(lock["portfolio_pnl_usd"].sum())},
        "segment_rule": "IS when every component sleeve is inside its IS, OOS when every component is past it, 'mixte' otherwise",
        "sleeve_is_end": {CODES[k]: v for k, v in IS_END.items()},
        "segment_bounds": {name: {seg: [str(labels.index[(labels[name] == seg) & signals[name].notna()].min().date()),
                                        str(labels.index[(labels[name] == seg) & signals[name].notna()].max().date())]
                                  for seg in ("IS", "mixte", "OOS") if ((labels[name] == seg) & signals[name].notna()).any()}
                           for name in labels},
        "correlations": corr,
        "independence_check_common_period": {
            "period": [str(mbt_start.date()), str(FS_END.date())],
            "book_sharpe_actual": sharpe_actual,
            "book_sharpe_if_three_signals_uncorrelated": sharpe_indep,
            "book_sharpe_if_eight_sleeves_uncorrelated": sharpe_indep_sleeves,
        },
        "metrics": rows,
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(fs_corr.round(3))
    print(json.dumps(result["independence_check_common_period"], indent=1))
    print(json.dumps(result["segment_bounds"], indent=1))
    for k, v in corr.items():
        print(k, v["sessions"], pd.DataFrame(v["matrix"]).round(3).to_dict())


if __name__ == "__main__":
    main()
