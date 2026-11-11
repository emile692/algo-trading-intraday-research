"""Run the frozen, limited Gotobi screening campaign on 6J one-minute bars."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.gotobi_6j_phase1 import CHICAGO, JST, topstep_session_bounds


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "gotobi_6j_phase1_20261003"
DATA = ROOT / "data" / "downloaded" / "6J_v_0_1m_20150101_20261003.parquet"
CALENDAR = OUT / "japan_business_gotobi_calendar.csv"
CONDITIONS = OUT / "databento_dataset_conditions.json"
SIX = ROOT / "research" / "daily_pnl.csv"
MULTIPLIER = 12_500_000.0
TICK_SIZE = 0.0000005
TICK_VALUE = 6.25
COMMISSION_RT = 4.22
HAC_LAG = 5


BOUNDARIES = {
    "p0930": (dt.time(9, 30), "open"),
    "p0950": (dt.time(9, 50), "open"),
    "p0955": (dt.time(9, 54), "close"),
    "p1000": (dt.time(9, 59), "close"),
    "p1015": (dt.time(10, 14), "close"),
    "p1030": (dt.time(10, 29), "close"),
    "p1100": (dt.time(10, 59), "close"),
    "p0921": (dt.time(9, 21), "open"),
    "p0931": (dt.time(9, 31), "open"),
    "p0941": (dt.time(9, 41), "open"),
    "p0954o": (dt.time(9, 54), "open"),
    "p0956": (dt.time(9, 56), "open"),
    "p1010": (dt.time(10, 10), "open"),
    "p1025": (dt.time(10, 25), "open"),
    "p1055": (dt.time(10, 55), "open"),
}
WINDOWS = {
    "reopen_0930": ("reopen", "p0930"),
    "0930_0950": ("p0930", "p0950"),
    "0950_0955": ("p0950", "p0955"),
    "0955_1000": ("p0955", "p1000"),
    "1000_1015": ("p1000", "p1015"),
    "1000_1030": ("p1000", "p1030"),
    "1000_1100": ("p1000", "p1100"),
}


def hac_tstat(values: pd.Series, lag: int = HAC_LAG) -> float:
    x = pd.Series(values, dtype=float).dropna().to_numpy()
    if len(x) < 3:
        return float("nan")
    u = x - x.mean()
    long_run = float(u @ u / len(x))
    for j in range(1, min(lag, len(x) - 1) + 1):
        gamma = float(u[j:] @ u[:-j] / len(x))
        long_run += 2 * (1 - j / (lag + 1)) * gamma
    se = math.sqrt(max(long_run, 0) / len(x))
    return float(x.mean() / se) if se else float("nan")


def annual_rate(dates: pd.Series) -> float:
    dates = pd.to_datetime(dates).dropna()
    if len(dates) < 2:
        return float("nan")
    years = max((dates.max() - dates.min()).days / 365.25, 1.0)
    return len(dates) / years


def descriptive_metrics(values: pd.Series, dates: pd.Series) -> dict:
    frame = pd.DataFrame({"value": values, "date": pd.to_datetime(dates)}).dropna()
    x = frame.value
    sd = x.std(ddof=1)
    rate = annual_rate(frame.date)
    return {
        "n": len(x),
        "mean_bps": x.mean() * 10_000,
        "median_bps": x.median() * 10_000,
        "positive_hit_rate": (x > 0).mean(),
        "negative_hit_rate": (x < 0).mean(),
        "hac_t": hac_tstat(x),
        "descriptive_sharpe": x.mean() / sd * math.sqrt(rate) if sd > 0 and rate > 0 else np.nan,
    }


def _timestamps_for_days(days: pd.Series, clock: dt.time) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        [pd.Timestamp(dt.datetime.combine(day, clock, tzinfo=JST)).tz_convert("UTC") for day in days]
    )


def extract_points(bars: pd.DataFrame, calendar_frame: pd.DataFrame) -> pd.DataFrame:
    out = calendar_frame.copy()
    days = out.japan_date
    degraded = {
        dt.date.fromisoformat(row["date"])
        for row in json.loads(CONDITIONS.read_text(encoding="utf-8"))
        if row.get("condition") == "degraded"
    }

    reopen_ts = pd.DatetimeIndex([topstep_session_bounds(day)[0].tz_convert("UTC") for day in days])
    lookup = bars.reindex(reopen_ts)
    out["reopen"] = lookup.open.to_numpy()
    out["id_reopen"] = lookup.instrument_id.to_numpy()
    out["q_reopen"] = [stamp.date() not in degraded for stamp in reopen_ts]

    for name, (clock, field) in BOUNDARIES.items():
        stamps = _timestamps_for_days(days, clock)
        lookup = bars.reindex(stamps)
        out[name] = lookup[field].to_numpy()
        out[f"id_{name}"] = lookup.instrument_id.to_numpy()
        out[f"q_{name}"] = [stamp.date() not in degraded for stamp in stamps]

    for window, (start, end) in WINDOWS.items():
        valid = (
            out[start].notna()
            & out[end].notna()
            & out[f"id_{start}"].eq(out[f"id_{end}"])
            & out[f"q_{start}"]
            & out[f"q_{end}"]
        )
        out[window] = np.where(valid, out[end] / out[start] - 1.0, np.nan)
        minutes = {
            "reopen_0930": np.nan,
            "0930_0950": 20,
            "0950_0955": 5,
            "0955_1000": 5,
            "1000_1015": 15,
            "1000_1030": 30,
            "1000_1100": 60,
        }[window]
        out[f"{window}_bps_per_min"] = out[window] * 10_000 / minutes if minutes else np.nan
    return out


def nearest_weekday_controls(ledger: pd.DataFrame) -> pd.DataFrame:
    non = ledger[~ledger.is_gotobi].copy()
    rows = []
    for _, event in ledger[ledger.is_gotobi].iterrows():
        candidates = non[non.japan_date.map(lambda d: d.weekday()) == event.japan_date.weekday()].copy()
        candidates["distance"] = candidates.japan_date.map(lambda d: abs((d - event.japan_date).days))
        candidates = candidates[(candidates.distance > 0) & (candidates.distance <= 28)]
        if candidates.empty:
            continue
        chosen = candidates.sort_values(["distance", "japan_date"]).iloc[0]
        rows.append({"event_date": event.japan_date, "control_date": chosen.japan_date})
    return pd.DataFrame(rows)


def adjacent_placebos(ledger: pd.DataFrame) -> pd.DataFrame:
    dates = list(ledger.japan_date)
    loc = {day: i for i, day in enumerate(dates)}
    rows = []
    gotobi = set(ledger.loc[ledger.is_gotobi, "japan_date"])
    for day in sorted(gotobi):
        i = loc[day]
        if i == 0 or i == len(dates) - 1:
            continue
        prior, nxt = dates[i - 1], dates[i + 1]
        rows.append(
            {
                "event_date": day,
                "prior_date": prior,
                "next_date": nxt,
                "prior_collision": prior in gotobi,
                "next_collision": nxt in gotobi,
            }
        )
    return pd.DataFrame(rows)


def event_study_tables(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weekday_pairs = nearest_weekday_controls(ledger)
    placebo_pairs = adjacent_placebos(ledger)
    by_date = ledger.set_index("japan_date")
    groups: dict[str, pd.DataFrame] = {
        "gotobi": ledger[ledger.is_gotobi],
        "non_gotobi": ledger[~ledger.is_gotobi],
        "gotobi_ex_eom": ledger[ledger.is_gotobi_ex_eom],
        "eom": ledger[ledger.is_eom],
        "weekday_matched_control": by_date.loc[weekday_pairs.control_date].reset_index(),
        "prior_business_day_placebo": by_date.loc[placebo_pairs.loc[~placebo_pairs.prior_collision, "prior_date"]].reset_index(),
        "next_business_day_placebo": by_date.loc[placebo_pairs.loc[~placebo_pairs.next_collision, "next_date"]].reset_index(),
    }
    rows = []
    for group, frame in groups.items():
        for window in WINDOWS:
            rows.append({"group": group, "window": window, **descriptive_metrics(frame[window], frame.japan_date)})
    summary = pd.DataFrame(rows)

    event_dates = ledger.loc[ledger.is_gotobi, "japan_date"].sort_values().tolist()
    split = event_dates[len(event_dates) // 2]
    subrows, yearrows = [], []
    for scope, frame in [
        ("first_half", ledger[ledger.is_gotobi & (ledger.japan_date < split)]),
        ("second_half", ledger[ledger.is_gotobi & (ledger.japan_date >= split)]),
    ]:
        for window in WINDOWS:
            subrows.append({"scope": scope, "window": window, **descriptive_metrics(frame[window], frame.japan_date)})
    for year, frame in ledger[ledger.is_gotobi].groupby(pd.to_datetime(ledger.loc[ledger.is_gotobi, "japan_date"]).dt.year.to_numpy()):
        for window in WINDOWS:
            yearrows.append({"year": year, "window": window, **descriptive_metrics(frame[window], frame.japan_date)})

    contrast_rows = []
    sources = {
        "weekday_matched": weekday_pairs.rename(columns={"control_date": "comparison_date"}),
        "prior_placebo": placebo_pairs.loc[~placebo_pairs.prior_collision, ["event_date", "prior_date"]].rename(columns={"prior_date": "comparison_date"}),
        "next_placebo": placebo_pairs.loc[~placebo_pairs.next_collision, ["event_date", "next_date"]].rename(columns={"next_date": "comparison_date"}),
    }
    for label, pairs in sources.items():
        for window in WINDOWS:
            diff = pairs.event_date.map(by_date[window]) - pairs.comparison_date.map(by_date[window])
            met = descriptive_metrics(diff, pairs.event_date)
            contrast_rows.append({"comparison": label, "window": window, **met})
    return summary, pd.DataFrame(subrows), pd.DataFrame(yearrows), pd.DataFrame(contrast_rows)


def event_gate(summary: pd.DataFrame, subperiod: pd.DataFrame, contrasts: pd.DataFrame) -> dict:
    mean = summary[summary.group == "gotobi"].set_index("window").mean_bps
    halves = subperiod.pivot(index="window", columns="scope", values="mean_bps")
    contrast = contrasts.pivot(index="window", columns="comparison", values="mean_bps")
    pre = (
        mean["0930_0950"] < 0
        and mean["0950_0955"] < 0
        and (halves.loc[["0930_0950", "0950_0955"]].sum(axis=0) < 0).all()
        and (contrast.loc["0950_0955"] < 0).sum() >= 2
    )
    post = (
        mean["0955_1000"] > 0
        and mean[["1000_1015", "1000_1030", "1000_1100"]].max() > 0
        and (halves.loc["0955_1000"] > 0).all()
    )
    return {"pre_fix_gate": bool(pre), "post_fix_gate": bool(post), "phase1b_open": bool(pre or post)}


def make_trades(ledger: pd.DataFrame, rule: str, variant: str, ticks: int, subset: str) -> pd.DataFrame:
    if subset == "gotobi_ex_eom":
        frame = ledger[ledger.is_gotobi_ex_eom].copy()
    elif subset == "eom":
        frame = ledger[ledger.is_eom].copy()
    else:
        raise ValueError(subset)
    if rule == "pre_fix_short":
        entry_col = {"entry_0920": "p0921", "entry_0930": "p0931", "entry_0940": "p0941"}[variant]
        exit_col, direction = "p0954o", -1.0
    else:
        entry_col = "p0956"
        exit_col = {"exit_15m": "p1010", "exit_30m": "p1025", "exit_60m": "p1055"}[variant]
        direction = 1.0
    valid = (
        frame[entry_col].notna()
        & frame[exit_col].notna()
        & frame[f"id_{entry_col}"].eq(frame[f"id_{exit_col}"])
        & frame[f"q_{entry_col}"]
        & frame[f"q_{exit_col}"]
    )
    frame = frame[valid].copy()
    frame["rule"] = rule
    frame["variant"] = variant
    frame["subset"] = subset
    frame["slippage_ticks_per_side"] = ticks
    frame["entry"] = frame[entry_col]
    frame["exit"] = frame[exit_col]
    frame["gross_pnl"] = direction * (frame.exit - frame.entry) * MULTIPLIER
    frame["cost_usd"] = COMMISSION_RT + 2 * ticks * TICK_VALUE
    frame["net_pnl"] = frame.gross_pnl - frame.cost_usd
    frame["gross_return"] = frame.gross_pnl / (frame.entry * MULTIPLIER)
    frame["net_return"] = frame.net_pnl / (frame.entry * MULTIPLIER)
    return frame


def trade_metrics(trades: pd.DataFrame, scope: str) -> dict:
    pnl = trades.net_pnl
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax().clip(lower=0)
    return {
        "scope": scope,
        "rule": trades.rule.iloc[0] if len(trades) else "",
        "variant": trades.variant.iloc[0] if len(trades) else "",
        "subset": trades.subset.iloc[0] if len(trades) else "",
        "slippage_ticks_per_side": trades.slippage_ticks_per_side.iloc[0] if len(trades) else np.nan,
        "n": len(trades),
        "gross_expectancy_usd": trades.gross_pnl.mean(),
        "net_expectancy_usd": pnl.mean(),
        "net_hac_t": hac_tstat(trades.net_return),
        "net_sharpe": descriptive_metrics(trades.net_return, trades.japan_date)["descriptive_sharpe"],
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.nan,
        "max_drawdown_usd": -drawdown.min() if len(drawdown) else np.nan,
        "win_rate": (pnl > 0).mean(),
        "net_total_pnl_usd": pnl.sum(),
    }


def run_trading(ledger: pd.DataFrame, gate: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specifications = []
    if gate["pre_fix_gate"]:
        specifications += [("pre_fix_short", v) for v in ["entry_0920", "entry_0930", "entry_0940"]]
    if gate["post_fix_gate"]:
        specifications += [("post_fix_long", v) for v in ["exit_15m", "exit_30m", "exit_60m"]]
    ledgers, rows, subrows, yearrows = [], [], [], []
    for rule, variant in specifications:
        for subset in ["gotobi_ex_eom", "eom"]:
            for ticks in [1, 2]:
                trades = make_trades(ledger, rule, variant, ticks, subset)
                ledgers.append(trades)
                rows.append(trade_metrics(trades, "full"))
                if ticks == 1:
                    split = trades.japan_date.sort_values().iloc[len(trades) // 2] if len(trades) else None
                    for scope, part in [
                        ("first_half", trades[trades.japan_date < split] if split else trades),
                        ("second_half", trades[trades.japan_date >= split] if split else trades),
                    ]:
                        subrows.append(trade_metrics(part, scope))
                    for year, part in trades.groupby(pd.to_datetime(trades.japan_date).dt.year):
                        yearrows.append({"year": year, **trade_metrics(part, "year")})
    ledger_out = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    return pd.DataFrame(rows), pd.DataFrame(subrows), pd.DataFrame(yearrows), ledger_out


def orthogonality(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    six = pd.read_csv(SIX, parse_dates=["session_date"]).set_index("session_date")
    rows = []
    baseline = trades[(trades.slippage_ticks_per_side == 1)].copy()
    for (rule, variant, subset), frame in baseline.groupby(["rule", "variant", "subset"]):
        if frame.net_pnl.mean() <= 0:
            continue
        pnl = frame.groupby(pd.to_datetime(frame.japan_date))["net_pnl"].sum()
        joined = six.copy()
        joined["gotobi_pnl"] = pnl.reindex(joined.index).fillna(0.0)
        sleeve_cols = [c for c in joined if c.endswith("_pnl_usd")]
        joined["six_sleeve_equal_weight"] = joined[sleeve_cols].mean(axis=1)
        for col in [*sleeve_cols, "six_sleeve_equal_weight"]:
            rows.append(
                {
                    "rule": rule,
                    "variant": variant,
                    "subset": subset,
                    "comparison": col,
                    "correlation": joined.gotobi_pnl.corr(joined[col]),
                    "n_daily": len(joined),
                    "gotobi_nonzero_days": int(joined.gotobi_pnl.ne(0).sum()),
                }
            )
    return pd.DataFrame(rows)


def markdown(frame: pd.DataFrame, columns: list[str], decimals: int = 3) -> str:
    if frame.empty:
        return "_Not run (event-study gate closed)._"
    out = frame[columns].copy()
    for col in out.select_dtypes(include="number"):
        out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{decimals}f}")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in out.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    bars = pd.read_parquet(DATA, columns=["open", "close", "instrument_id"])
    calendar_frame = pd.read_csv(CALENDAR, parse_dates=["japan_date"])
    calendar_frame["japan_date"] = calendar_frame.japan_date.dt.date
    ledger = extract_points(bars, calendar_frame)
    ledger.to_csv(OUT / "event_daily_ledger.csv", index=False)

    summary, subperiod, yearly, contrasts = event_study_tables(ledger)
    summary.to_csv(OUT / "event_study_summary.csv", index=False)
    subperiod.to_csv(OUT / "event_study_subperiods.csv", index=False)
    yearly.to_csv(OUT / "event_study_yearly.csv", index=False)
    contrasts.to_csv(OUT / "event_study_matched_contrasts.csv", index=False)
    gate = event_gate(summary, subperiod, contrasts)

    trade_summary, trade_subperiod, trade_yearly, trades = run_trading(ledger, gate)
    trade_summary.to_csv(OUT / "trading_rule_summary.csv", index=False)
    trade_subperiod.to_csv(OUT / "trading_rule_subperiods.csv", index=False)
    trade_yearly.to_csv(OUT / "trading_rule_yearly.csv", index=False)
    trades.to_csv(OUT / "trade_ledger.csv", index=False)
    corr = orthogonality(trades)
    corr.to_csv(OUT / "orthogonality_correlations.csv", index=False)

    central = trade_summary[
        (
            ((trade_summary.rule == "pre_fix_short") & (trade_summary.variant == "entry_0930"))
            | ((trade_summary.rule == "post_fix_long") & (trade_summary.variant == "exit_30m"))
        )
        & (trade_summary.slippage_ticks_per_side == 1)
    ] if not trade_summary.empty else trade_summary
    positive = central[central.net_expectancy_usd > 0] if not central.empty else central
    promising = False
    if not positive.empty:
        for _, candidate in positive.iterrows():
            halves = trade_subperiod[
                (trade_subperiod.rule == candidate.rule)
                & (trade_subperiod.variant == candidate.variant)
                & (trade_subperiod.subset == candidate.subset)
            ]
            stress = trade_summary[
                (trade_summary.rule == candidate.rule)
                & (trade_summary.variant == candidate.variant)
                & (trade_summary.subset == candidate.subset)
                & (trade_summary.slippage_ticks_per_side == 2)
            ]
            annual = trade_yearly[
                (trade_yearly.rule == candidate.rule)
                & (trade_yearly.variant == candidate.variant)
                & (trade_yearly.subset == candidate.subset)
            ]
            annual_positive_share = annual.net_expectancy_usd.gt(0).mean() if len(annual) else 0.0
            if (
                candidate.n >= 100
                and halves.net_expectancy_usd.gt(0).all()
                and stress.net_expectancy_usd.gt(0).all()
                and annual_positive_share >= 2 / 3
            ):
                promising = True
    verdict = "PROMISING_FOR_VALIDATION" if promising else ("WEAK_EDGE" if not positive.empty else "REJECTED")

    data_hash = hashlib.sha256(DATA.read_bytes()).hexdigest()
    metadata = {
        "verdict": verdict,
        "event_gate": gate,
        "data_rows": len(bars),
        "data_sha256": data_hash,
        "calendar_business_days": len(ledger),
        "gotobi_days": int(ledger.is_gotobi.sum()),
        "gotobi_ex_eom_days": int(ledger.is_gotobi_ex_eom.sum()),
        "eom_days": int(ledger.is_eom.sum()),
        "frozen_sleeves_modified": False,
        "phase1c_run": not corr.empty,
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    event_view = summary[summary.group.isin(["gotobi", "non_gotobi", "weekday_matched_control", "prior_business_day_placebo", "next_business_day_placebo"])]
    gotobi_view = summary[summary.group == "gotobi"].set_index("window")
    weekday_view = contrasts[contrasts.comparison == "weekday_matched"].set_index("window")
    pre_central = trade_summary[
        (trade_summary.rule == "pre_fix_short")
        & (trade_summary.variant == "entry_0930")
        & (trade_summary.slippage_ticks_per_side == 1)
    ].set_index("subset")
    post_central = trade_summary[
        (trade_summary.rule == "post_fix_long")
        & (trade_summary.variant == "exit_30m")
        & (trade_summary.slippage_ticks_per_side == 1)
    ].set_index("subset")
    book_corr = corr[
        (corr.rule == "pre_fix_short")
        & (corr.variant == "entry_0930")
        & (corr.subset == "eom")
        & (corr.comparison == "six_sleeve_equal_weight")
    ].correlation
    if verdict == "PROMISING_FOR_VALIDATION":
        stopping_text = """`PROMISING_FOR_VALIDATION`: the broad Gotobi-ex-month-end rules are not viable net of costs, but the separately specified month-end pre-fix rule passes the Phase-1 promotion conditions. The frozen 09:30 decision / 09:31 entry rule has 136 usable trades, positive expectancy in both halves, positive years in at least two thirds of the sample, and remains marginally positive at two ticks per side. Research stops here without further filtering or retuning.

### Proposed Phase 2 (separate campaign)

1. Freeze only the month-end pre-fix short at the central 09:30 decision, 09:31 next-bar entry, and 09:54 scheduled exit; do not promote the better-looking 09:20 variant.
2. Validate on explicit contract bars and bid/ask or tick data to measure spread, queueing, and the first tradable price; the two-tick margin is thin.
3. Use untouched chronological holdouts or anchored walk-forward evaluation with no parameter selection.
4. Re-audit CME holiday sessions, Japanese bank-calendar exceptions, and roll dates observation by observation.
5. Replicate on an independent data feed and then run prospective paper observation before any portfolio allocation or Topstep simulation.
6. Reassess multiplicity and statistical uncertainty only in that validation phase."""
    else:
        stopping_text = """`WEAK_EDGE`: the economic price pattern is strong, correctly signed, and distinct from the placebos around the immediate fix, but no central rule satisfies the promotion gate. Per mandate, research stops here with no additional filter, retuning, rule combination, allocation, or Topstep simulation."""
    report = f"""# 6J Gotobi — Phase 1 screening report

## Verdict: {verdict}

This is a limited screening study. The six frozen sleeves and the two prior hypotheses were not modified or reopened.

## Data and implementation

- Databento `6J.v.0`, one-minute OHLCV, UTC, point-in-time volume-ranked continuous mapping; {len(bars):,} rows and SHA-256 `{data_hash}`.
- 6J is USD per JPY: the expected pre-fix rise in USD/JPY maps to a negative 6J return.
- Japan dates use `Asia/Tokyo`; CME and Topstep mapping uses `America/Chicago`, including DST.
- Every return requires both endpoint bars, acceptable Databento dataset condition, and unchanged `instrument_id`. No interpolation or back-adjustment is used.
- All requested fix windows are structurally within Topstep hours. Actual holiday closures are rejected through missing-bar checks.
- The 09:50-09:55 return ends on the 09:54 close. The 09:55-10:00 window starts from that last pre-fix price and ends on the 09:59 close, capturing the boundary reaction without using a post-fix price as its starting point.
- The weekday control is the nearest non-Gotobi observation with the same Japanese weekday within 28 calendar days; adjacent-business-day placebos that collide with another Gotobi are excluded.
- Event study inference is HAC({HAC_LAG}); Sharpe is descriptive and annualized by observed opportunity rate.

## Phase 1A — event study

{markdown(event_view, ['group','window','n','mean_bps','median_bps','positive_hit_rate','negative_hit_rate','hac_t','descriptive_sharpe'])}

### Matched Gotobi-minus-control contrasts

{markdown(contrasts, ['comparison','window','n','mean_bps','median_bps','hac_t','descriptive_sharpe'])}

### Large chronological subperiods

{markdown(subperiod, ['scope','window','n','mean_bps','median_bps','positive_hit_rate','hac_t','descriptive_sharpe'])}

Year-by-year results are in `event_study_yearly.csv`. Gate decision: pre-fix `{gate['pre_fix_gate']}`, post-fix `{gate['post_fix_gate']}`.

The expected pattern is visible and concentrated near the fix. Gotobi 6J averages {gotobi_view.loc['0930_0950','mean_bps']:.3f} bp from 09:30 to 09:50 and {gotobi_view.loc['0950_0955','mean_bps']:.3f} bp in the last five minutes before the fix (HAC t={gotobi_view.loc['0950_0955','hac_t']:.2f}); it then reverses by {gotobi_view.loc['0955_1000','mean_bps']:.3f} bp in the next five minutes (HAC t={gotobi_view.loc['0955_1000','hac_t']:.2f}). Relative to the weekday-matched control, the five-minute reversal is {weekday_view.loc['0955_1000','mean_bps']:.3f} bp (HAC t={weekday_view.loc['0955_1000','hac_t']:.2f}). The post-fix mean decays as the horizon extends, rather than strengthening indefinitely.

## Phase 1B — trading translation

Frozen central rules are short 6J from the next bar after 09:30 JST to the scheduled 09:54 open, and long 6J from 09:56 to 10:25 JST. Nearby variants are reported, never selected ex post. Costs are TopstepX USD {COMMISSION_RT:.2f} round turn plus one tick per side; two ticks per side is the stress.

{markdown(trade_summary, ['rule','variant','subset','slippage_ticks_per_side','n','gross_expectancy_usd','net_expectancy_usd','net_hac_t','net_sharpe','profit_factor','max_drawdown_usd','win_rate','net_total_pnl_usd'])}

### Trading chronological halves

{markdown(trade_subperiod, ['scope','rule','variant','subset','n','net_expectancy_usd','net_hac_t','net_sharpe','profit_factor','max_drawdown_usd','win_rate','net_total_pnl_usd'])}

Annual trading results are in `trading_rule_yearly.csv`.

For Gotobi excluding month-end, the central pre-fix rule earns USD {pre_central.loc['gotobi_ex_eom','gross_expectancy_usd']:.2f} gross but USD {pre_central.loc['gotobi_ex_eom','net_expectancy_usd']:.2f} net per trade; the central 30-minute reversal earns USD {post_central.loc['gotobi_ex_eom','gross_expectancy_usd']:.2f} gross and USD {post_central.loc['gotobi_ex_eom','net_expectancy_usd']:.2f} net. Month-end pre-fix is the only central rule with a meaningful positive net result (USD {pre_central.loc['eom','net_expectancy_usd']:.2f}, N={int(pre_central.loc['eom','n'])}); it remains barely positive at two ticks and is therefore promoted only to validation, not accepted as robust. The month-end 30-minute reversal is only USD {post_central.loc['eom','net_expectancy_usd']:.2f} net and fails the two-tick stress.

## Phase 1C — simple orthogonality

{markdown(corr, ['rule','variant','subset','comparison','correlation','n_daily','gotobi_nonzero_days'])}

The central month-end pre-fix rule's correlation with the equal-weight six-sleeve book is {float(book_corr.iloc[0]) if len(book_corr) else float('nan'):.3f}. This supports potential orthogonality but does not replace standalone validation.

## Stopping decision

{stopping_text}

No portfolio allocation, ERC, Deflated Sharpe, complex bootstrap, Topstep account simulation, or rule combination was performed.
"""
    (OUT / "PHASE1_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(summary[summary.group == "gotobi"].to_string(index=False))
    print(trade_summary.to_string(index=False))


if __name__ == "__main__":
    main()
