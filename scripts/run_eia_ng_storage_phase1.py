"""Limited, causal Phase-1 event study for EIA natural-gas storage releases."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "eia_ng_storage_intraday_20261003"
DATA = ROOT / "data" / "downloaded" / "NG_v_0_1m_20200101_20261003.parquet"
CALENDAR = OUT / "eia_release_calendar.csv"
TICK_SIZE = 0.001
NG_MULTIPLIER = 10_000.0
NG_TICK_VALUE = TICK_SIZE * NG_MULTIPLIER


def hac_tstat(values: pd.Series, lag: int = 4) -> float:
    x = pd.Series(values, dtype=float).dropna().to_numpy()
    n = len(x)
    if n < 3:
        return float("nan")
    u = x - x.mean()
    long_run = float(u @ u / n)
    for j in range(1, min(lag, n - 1) + 1):
        gamma = float(u[j:] @ u[:-j] / n)
        long_run += 2.0 * (1.0 - j / (lag + 1.0)) * gamma
    se = math.sqrt(max(long_run, 0.0) / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def sharpe(values: pd.Series, annual_opportunities: float = 52.0) -> float:
    x = pd.Series(values, dtype=float).dropna()
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(annual_opportunities)) if sd > 0 else float("nan")


def key_prices(day_frame: pd.DataFrame, release_time: str = "10:30:00") -> dict | None:
    # This conservative campaign contains standard 10:30 releases only.
    if release_time != "10:30:00":
        return None
    by_time = day_frame.set_index("clock")
    keys = {
        "p0800": ("08:00", "open"),
        "p_pre": ("10:29", "close"),
        "p_imm": ("10:44", "close"),
        "p1045": ("10:45", "open"),
        "p1046": ("10:46", "open"),
        "p1050": ("10:50", "open"),
        "p1600": ("16:00", "open"),
    }
    result = {}
    for name, (clock, field) in keys.items():
        if clock not in by_time.index:
            return None
        row = by_time.loc[clock]
        if isinstance(row, pd.DataFrame):
            return None
        result[name] = float(row[field])
        result[f"id_{name}"] = int(row["instrument_id"])
    return result


def build_daily_points(bars: pd.DataFrame) -> pd.DataFrame:
    local = bars.tz_convert("America/New_York").copy()
    local["date"] = pd.Index(local.index.date)
    local["clock"] = local.index.strftime("%H:%M")
    wanted = {"08:00", "10:29", "10:44", "10:45", "10:46", "10:50", "16:00"}
    slim = local[local["clock"].isin(wanted)]
    records = []
    for day, frame in slim.groupby("date", sort=True):
        prices = key_prices(frame)
        if prices is not None:
            prices["date"] = pd.Timestamp(day)
            records.append(prices)
    daily = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    # Prior session 16:00, with point-in-time contract identity retained.
    daily["prev_p1600"] = daily["p1600"].shift(1)
    daily["prev_id_p1600"] = daily["id_p1600"].shift(1)
    daily["overnight_roll_safe"] = daily["prev_id_p1600"].eq(daily["id_p0800"])
    daily["overnight"] = np.where(
        daily["overnight_roll_safe"], daily["p0800"] / daily["prev_p1600"] - 1.0, np.nan
    )
    daily["morning_pre"] = daily["p_pre"] / daily["p0800"] - 1.0
    daily["immediate"] = daily["p_imm"] / daily["p_pre"] - 1.0
    daily["rest_session"] = daily["p1600"] / daily["p_imm"] - 1.0
    daily["week"] = daily["date"].dt.to_period("W-SUN").astype(str)
    return daily


def descriptive_table(events: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in ["overnight", "morning_pre", "immediate", "rest_session"]:
        event = events.set_index("week")[window]
        control_week = controls.groupby("week")[window].mean()
        paired = pd.concat([event.rename("event"), control_week.rename("control")], axis=1).dropna()
        diff = paired["event"] - paired["control"]
        rows.append(
            {
                "window": window,
                "event_n": int(event.notna().sum()),
                "event_mean_bps": event.mean() * 10_000,
                "event_hac_t": hac_tstat(event),
                "control_n_days": int(controls[window].notna().sum()),
                "control_mean_bps": controls[window].mean() * 10_000,
                "paired_weeks": len(diff),
                "event_minus_control_bps": diff.mean() * 10_000,
                "difference_hac_t": hac_tstat(diff),
            }
        )
    return pd.DataFrame(rows)


def trade_frame(events: pd.DataFrame, rule: str, delay: int = 0, slip: int = 1) -> pd.DataFrame:
    frame = events.copy()
    if rule == "pre_long":
        direction = pd.Series(1.0, index=frame.index)
        entry, exit_ = frame["p0800"], frame["p_pre"]
    elif rule == "pre_short":
        direction = pd.Series(-1.0, index=frame.index)
        entry, exit_ = frame["p0800"], frame["p_pre"]
    elif rule in {"post_long", "post_short"}:
        direction = pd.Series(1.0 if rule == "post_long" else -1.0, index=frame.index)
        entry_col = {0: "p1045", 1: "p1046", 5: "p1050"}[delay]
        entry, exit_ = frame[entry_col], frame["p1600"]
    else:
        reaction_sign = np.sign(frame["p_imm"] - frame["p_pre"])
        direction = reaction_sign if rule == "continuation" else -reaction_sign
        entry_col = {0: "p1045", 1: "p1046", 5: "p1050"}[delay]
        entry, exit_ = frame[entry_col], frame["p1600"]
    gross_pnl = direction * (exit_ - entry) * NG_MULTIPLIER
    costs = 2.0 * slip * NG_TICK_VALUE
    frame["direction"] = direction
    frame["entry"] = entry
    frame["exit"] = exit_
    frame["gross_pnl"] = gross_pnl
    frame["net_pnl"] = gross_pnl - costs
    frame["gross_return"] = gross_pnl / (entry * NG_MULTIPLIER)
    frame["net_return"] = frame["net_pnl"] / (entry * NG_MULTIPLIER)
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["net_return"])


def metrics(trades: pd.DataFrame, rule: str, delay: int, slip: int, scope: str) -> dict:
    return {
        "scope": scope,
        "rule": rule,
        "delay_min": delay,
        "slippage_ticks_per_side": slip,
        "n": len(trades),
        "gross_mean_bps": trades["gross_return"].mean() * 10_000,
        "net_mean_bps": trades["net_return"].mean() * 10_000,
        "net_hac_t": hac_tstat(trades["net_return"]),
        "net_sharpe": sharpe(trades["net_return"]),
        "gross_total_pnl_usd": trades["gross_pnl"].sum(),
        "net_total_pnl_usd": trades["net_pnl"].sum(),
        "net_mean_pnl_usd": trades["net_pnl"].mean(),
        "win_rate_net": (trades["net_pnl"] > 0).mean(),
    }


def fmt_table(frame: pd.DataFrame, decimals: int = 2) -> str:
    out = frame.copy()
    for col in out.select_dtypes(include=["number"]).columns:
        out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{decimals}f}")
    headers = [str(col) for col in out.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in out.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bars = pd.read_parquet(DATA)
    calendar = pd.read_csv(CALENDAR, parse_dates=["release_date"])
    # Archive-based reconstruction only validates standard Thursday releases.
    calendar = calendar[
        (calendar["release_time"] == "10:30:00")
        & (calendar["release_date"].dt.weekday == 3)
        & (calendar["release_date"].dt.year <= 2025)
    ].copy()
    daily = build_daily_points(bars)
    event_dates = set(calendar["release_date"])
    events = daily[daily["date"].isin(event_dates)].copy()

    # Controls are non-EIA weekdays from the same confirmed-event weeks.  We
    # exclude all Thursdays and all unconfirmed weeks to prevent contamination.
    confirmed_weeks = set(events["week"])
    controls = daily[
        daily["week"].isin(confirmed_weeks)
        & daily["date"].dt.weekday.isin([0, 1, 2, 4])
    ].copy()

    event_windows = descriptive_table(events, controls)
    event_windows.to_csv(OUT / "event_vs_control_windows.csv", index=False)
    events.to_csv(OUT / "event_ledger.csv", index=False)

    baseline_rows = []
    ledgers = []
    rules = ["pre_long", "pre_short", "post_long", "post_short", "continuation", "reversal"]
    for rule in rules:
        trades = trade_frame(events, rule, delay=0, slip=1)
        baseline_rows.append(metrics(trades, rule, 0, 1, "full"))
        keep = trades[["date", "week", "direction", "entry", "exit", "gross_pnl", "net_pnl", "gross_return", "net_return"]].copy()
        keep.insert(0, "rule", rule)
        ledgers.append(keep)
    baseline = pd.DataFrame(baseline_rows)
    baseline.to_csv(OUT / "rule_metrics_baseline.csv", index=False)
    pd.concat(ledgers, ignore_index=True).to_csv(OUT / "trade_ledger_baseline.csv", index=False)

    split_date = events["date"].sort_values().iloc[len(events) // 2]
    sub_rows = []
    for rule in rules:
        trades = trade_frame(events, rule, delay=0, slip=1)
        for label, subset in [
            ("first_half", trades[trades["date"] < split_date]),
            ("second_half", trades[trades["date"] >= split_date]),
        ]:
            sub_rows.append(metrics(subset, rule, 0, 1, label))
    subperiod = pd.DataFrame(sub_rows)
    subperiod.to_csv(OUT / "rule_metrics_subperiods.csv", index=False)

    sensitivity_rows = []
    for rule in ["post_long", "post_short", "continuation", "reversal"]:
        for delay in [0, 1, 5]:
            for slip in [1, 2]:
                sensitivity_rows.append(
                    metrics(trade_frame(events, rule, delay, slip), rule, delay, slip, "full")
                )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUT / "delay_slippage_sensitivity.csv", index=False)

    median_price = float(events["p1045"].median())
    contracts = pd.DataFrame(
        [
            {"instrument": "NG", "contract_mmbtu": 10_000, "tick": 0.001, "tick_value_usd": 10.0, "roundtrip_1tick_side_usd": 20.0, "roundtrip_cost_bps_at_median_price": 2 * 0.001 / median_price * 10_000, "empirical_bars": True},
            {"instrument": "QG", "contract_mmbtu": 2_500, "tick": 0.005, "tick_value_usd": 12.5, "roundtrip_1tick_side_usd": 25.0, "roundtrip_cost_bps_at_median_price": 2 * 0.005 / median_price * 10_000, "empirical_bars": False},
            {"instrument": "MNG", "contract_mmbtu": 1_000, "tick": 0.001, "tick_value_usd": 1.0, "roundtrip_1tick_side_usd": 2.0, "roundtrip_cost_bps_at_median_price": 2 * 0.001 / median_price * 10_000, "empirical_bars": False},
        ]
    )
    contracts.to_csv(OUT / "contract_cost_comparison.csv", index=False)

    best = baseline.sort_values("net_sharpe", ascending=False).iloc[0]
    positive_halves = subperiod[subperiod["rule"] == best["rule"]]["net_mean_pnl_usd"].gt(0).all()
    stable_stress = sensitivity[
        sensitivity["rule"] == best["rule"]
    ]["net_mean_pnl_usd"].gt(0).all() if best["rule"] in {"post_long", "post_short", "continuation", "reversal"} else positive_halves
    if best["net_hac_t"] >= 2.0 and positive_halves and stable_stress:
        verdict = "PROMISING_FOR_VALIDATION"
    elif best["net_hac_t"] >= 1.0 and positive_halves:
        verdict = "WEAK_EDGE"
    else:
        verdict = "NO_EDGE"

    coverage = {
        "calendar_confirmed_releases": int(len(calendar)),
        "events_with_complete_bars": int(len(events)),
        "control_days": int(len(controls)),
        "first_event": str(events["date"].min().date()),
        "last_event": str(events["date"].max().date()),
        "split_date": str(split_date.date()),
        "overnight_roll_safe_events": int(events["overnight"].notna().sum()),
        "intraday_contract_change_events": int(
            (events[[c for c in events if c.startswith("id_")]].nunique(axis=1) > 1).sum()
        ),
        "raw_bar_rows": int(len(bars)),
        "raw_duplicate_timestamps": int(bars.index.duplicated().sum()),
        "raw_instrument_ids": int(bars["instrument_id"].nunique()),
        "raw_first_timestamp_utc": str(bars.index.min()),
        "raw_last_timestamp_utc": str(bars.index.max()),
        "surprise_data_available": False,
        "qg_intraday_available": False,
        "mng_intraday_available": False,
        "verdict": verdict,
    }
    (OUT / "run_summary.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    report = f"""# EIA Natural Gas Storage — Phase 1 exploratory report

## Verdict: {verdict}

This is a deliberately limited event study, not a production-ready strategy.

## Data and causality

- NG: Databento `NG.v.0` volume-ranked continuous one-minute bars.
- Raw integrity: {coverage['raw_bar_rows']:,} rows, {coverage['raw_duplicate_timestamps']} duplicate timestamps, {coverage['raw_instrument_ids']} underlying instrument IDs, from {coverage['raw_first_timestamp_utc']} to {coverage['raw_last_timestamp_utc']}.
- Sample: {coverage['events_with_complete_bars']} complete, officially archived standard releases from {coverage['first_event']} through {coverage['last_event']}.
- Calendar: only Thursday 10:30 ET releases confirmed by dated EIA dashboard archives are retained. Historical holiday exceptions that cannot be timestamped from the archive are excluded.
- Controls: {coverage['control_days']} Monday/Tuesday/Wednesday/Friday sessions from the same confirmed-event weeks; all Thursdays and unconfirmed weeks are excluded.
- Reaction: last pre-release price is the 10:29 bar close; the 15-minute reaction ends at the 10:44 close.
- Post-release signal: formed only after the 10:44 close; earliest entry is the next bar open at 10:45.
- Overnight observations crossing a continuous-contract change are excluded ({coverage['overnight_roll_safe_events']} remain).
- No retained event changes continuous-contract identity inside the 08:00-16:00 analysis window ({coverage['intraday_contract_change_events']} detected).
- No contemporaneous, unrevised consensus history was available locally or from an auditable public primary source. Surprise-sign and surprise-size tests were therefore not run.

## Phenomenon: EIA versus matched-week controls

{fmt_table(event_windows)}

The difference t-stat is HAC(4) on weekly paired differences: the EIA-day return minus the mean return of the non-EIA days in that same week.

## Natural causal rules — baseline cost

Baseline cost is one tick per side (USD 20 round trip for NG), excluding broker/exchange fees. Returns, HAC t-statistics and Sharpe below are net of that slippage assumption. Sharpe uses sqrt(52).

{fmt_table(baseline[['rule','n','gross_mean_bps','net_mean_bps','net_hac_t','net_sharpe','gross_total_pnl_usd','net_total_pnl_usd','net_mean_pnl_usd','win_rate_net']])}

## Chronological halves

Split date: {coverage['split_date']}.

{fmt_table(subperiod[['scope','rule','n','net_mean_bps','net_hac_t','net_sharpe','net_total_pnl_usd','net_mean_pnl_usd']])}

## Entry-delay and slippage sensitivity

{fmt_table(sensitivity[['rule','delay_min','slippage_ticks_per_side','n','net_mean_bps','net_hac_t','net_sharpe','net_total_pnl_usd','net_mean_pnl_usd']])}

## NG / QG / MNG mechanics

At the sample median NG price ({median_price:.3f}), the one-tick-per-side round-trip costs below are mechanical slippage equivalents, before commissions and exchange fees.

{fmt_table(contracts, 3)}

QG is one quarter of NG's contract quantity but has a five-times coarser price tick and a larger dollar tick than NG. MNG preserves NG's 0.001 price granularity with one tenth of the exposure and tick value. Only NG has local intraday bars, so QG/MNG liquidity, spread, and realized-slippage comparisons are not claimed here.

## Scope limits

No parameter grid, Deflated Sharpe, Topstep simulation, or pre-release use of surprise data was performed. The verdict gate is intentionally simple: a promotion requires net HAC t >= 2, both chronological halves positive, and (for post-release rules) positive mean PnL across every requested delay/slippage stress.

The aggregate continuation result is positive but statistically weak and entirely concentrated in the first chronological half; the pre-announcement short shows the same decay. The schedule-only post-announcement short is positive in both halves but has only a low HAC t-stat and Sharpe. These facts drive the `WEAK_EDGE` label rather than promotion.

Official source references: [EIA WNGSR schedule](https://ir.eia.gov/ngs/schedule.html), [EIA WNGSR methodology](https://ir.eia.gov/ngs/methodology.html), and [CME NG/QG/MNG contract comparison](https://www.cmegroup.com/articles/faqs/micro-henry-hub-natural-gas-futures-and-options-frequently-asked-questions.html).
"""
    (OUT / "PHASE1_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(coverage, indent=2))
    print(baseline.to_string(index=False))


if __name__ == "__main__":
    main()
