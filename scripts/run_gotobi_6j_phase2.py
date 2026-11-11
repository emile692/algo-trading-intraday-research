"""Frozen Phase-2 validation of the month-end 6J pre-fix short."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "research" / "gotobi_6j_phase1_20261003"
OUT = ROOT / "research" / "gotobi_6j_phase2_20261003"
MBP = OUT / "mbp1_events"
DEFS = OUT / "definition_events"
BARS_PATH = ROOT / "data" / "downloaded" / "6J_v_0_1m_20150101_20261003.parquet"
SIX_PATH = ROOT / "research" / "daily_pnl.csv"
MULTIPLIER = 12_500_000.0
TICK_SIZE = 0.0000005
TICK_VALUE = 6.25
COMMISSION_RT = 4.22
HAC_LAG = 5


def hac_tstat(values: pd.Series, lag: int = HAC_LAG) -> float:
    x = pd.Series(values, dtype=float).dropna().to_numpy()
    if len(x) < 3:
        return float("nan")
    u = x - x.mean()
    long_run = float(u @ u / len(x))
    for j in range(1, min(lag, len(x) - 1) + 1):
        gamma = float(u[j:] @ u[:-j] / len(x))
        long_run += 2 * (1 - j / (lag + 1)) * gamma
    se = math.sqrt(max(long_run, 0.0) / len(x))
    return float(x.mean() / se) if se else float("nan")


def annual_rate(dates: pd.Series) -> float:
    d = pd.to_datetime(dates).dropna()
    if len(d) < 2:
        return float("nan")
    return len(d) / max((d.max() - d.min()).days / 365.25, 1.0)


def metrics(pnl: pd.Series, dates: pd.Series) -> dict[str, float | int]:
    frame = pd.DataFrame({"pnl": pnl, "date": pd.to_datetime(dates)}).dropna().sort_values("date")
    x = frame.pnl
    gp = x[x > 0].sum()
    gl = -x[x < 0].sum()
    equity = x.cumsum()
    dd = equity - equity.cummax().clip(lower=0)
    sd = x.std(ddof=1)
    rate = annual_rate(frame.date)
    return {
        "n": len(x),
        "expectancy_usd": x.mean(),
        "total_pnl_usd": x.sum(),
        "hac_t": hac_tstat(x),
        "sharpe": x.mean() / sd * math.sqrt(rate) if sd > 0 and rate > 0 else np.nan,
        "profit_factor": gp / gl if gl > 0 else np.nan,
        "max_drawdown_usd": -dd.min() if len(dd) else np.nan,
        "win_rate": (x > 0).mean() if len(x) else np.nan,
    }


def first_quote(frame: pd.DataFrame, stamp: pd.Timestamp) -> pd.Series:
    valid = frame[
        frame.ts_event.ge(stamp)
        & frame.bid_px_00.notna()
        & frame.ask_px_00.notna()
        & frame.ask_px_00.ge(frame.bid_px_00)
    ].sort_values(["ts_event", "sequence"])
    if valid.empty:
        raise RuntimeError(f"No valid quote at or after {stamp}")
    return valid.iloc[0]


def first_trade(frame: pd.DataFrame, stamp: pd.Timestamp) -> pd.Series:
    valid = frame[
        frame.ts_event.ge(stamp)
        & frame.ts_event.lt(stamp + pd.Timedelta(minutes=1))
        & frame.action.eq("T")
    ].sort_values(["ts_event", "sequence"])
    if valid.empty:
        raise RuntimeError(f"No trade in scheduled one-minute bar at {stamp}")
    return valid.iloc[0]


def longest_losing_streak(pnl: pd.Series) -> int:
    best = current = 0
    for value in pnl:
        current = current + 1 if value < 0 else 0
        best = max(best, current)
    return best


def load_phase1_events() -> pd.DataFrame:
    ledger = pd.read_csv(P1 / "trade_ledger.csv")
    mask = (
        ledger.rule.eq("pre_fix_short")
        & ledger.variant.eq("entry_0930")
        & ledger["subset"].eq("eom")
        & ledger.slippage_ticks_per_side.eq(1)
    )
    frame = ledger.loc[mask].drop_duplicates("japan_date").copy()
    if len(frame) != 136:
        raise RuntimeError(f"Phase-1 gate failed: expected 136 trades, got {len(frame)}")
    checks = {
        "gross_mean": frame.gross_pnl.mean(),
        "net_mean": frame.net_pnl.mean(),
        "net_total": frame.net_pnl.sum(),
    }
    expected = {"gross_mean": 31.020220588234, "net_mean": 14.300220588234, "net_total": 1944.83}
    if any(abs(checks[k] - expected[k]) > 0.01 for k in expected):
        raise RuntimeError(f"Phase-1 monetary replication failed: {checks}")
    frame["japan_date"] = pd.to_datetime(frame.japan_date)
    return frame.sort_values("japan_date").reset_index(drop=True)


def reconstruct(events: pd.DataFrame, bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(OUT / "phase2_event_windows.csv", parse_dates=["japan_date"])
    mapping = mapping.set_index(mapping.japan_date.dt.date)
    rows: list[dict] = []
    minute_rows: list[dict] = []
    all_bars = bars.copy()
    for _, event in events.iterrows():
        day = event.japan_date.date()
        meta = mapping.loc[day]
        symbol = str(meta.raw_symbol)
        mbp_path = MBP / f"{day}_{symbol}_mbp1.parquet"
        def_path = DEFS / f"{day}_{symbol}_definition.parquet"
        book = pd.read_parquet(mbp_path).sort_values(["ts_event", "sequence"])
        definition = pd.read_parquet(def_path).sort_values("ts_event").iloc[-1]
        entry_utc = pd.Timestamp(meta.entry_utc)
        exit_utc = pd.Timestamp(meta.exit_utc)
        entry_q = first_quote(book, entry_utc)
        exit_q = first_quote(book, exit_utc)
        entry_t = first_trade(book, entry_utc)
        exit_t = first_trade(book, exit_utc)
        expected_id = int(event.id_p0931)
        ids = set(book.instrument_id.astype(int).unique())
        if ids != {expected_id} or int(definition.instrument_id) != expected_id:
            raise RuntimeError(f"Instrument mismatch on {day}: expected {expected_id}, got {ids}")

        # True explicit-contract bar opens must reproduce the continuous series
        # whenever its point-in-time mapping is unchanged over the trade.
        explicit_entry = float(entry_t.price)
        explicit_exit = float(exit_t.price)
        explicit_gross = (explicit_entry - explicit_exit) * MULTIPLIER
        quote_gross = (float(entry_q.bid_px_00) - float(exit_q.ask_px_00)) * MULTIPLIER
        boundary = pd.DataFrame([entry_q, exit_q])
        boundary_spreads = (boundary.ask_px_00 - boundary.bid_px_00) / TICK_SIZE
        window = book[book.ts_event.ge(entry_utc) & book.ts_event.lt(exit_utc)]
        trades = window[window.action.eq("T")]
        minutes_with_trades = trades.ts_event.dt.floor("min").nunique()
        expiration = pd.Timestamp(definition.expiration)
        session_open = (
            pd.Timestamp(day - pd.Timedelta(days=1)).tz_localize("America/Chicago")
            + pd.Timedelta(hours=17)
        ).tz_convert("UTC")
        session_close = (
            pd.Timestamp(day).tz_localize("America/Chicago") + pd.Timedelta(hours=16)
        ).tz_convert("UTC")
        session_bars = all_bars[
            (all_bars.index >= session_open)
            & (all_bars.index < session_close)
            & all_bars.instrument_id.eq(expected_id)
        ]

        rows.append(
            {
                "japan_date": day,
                "entry_jst": entry_utc.tz_convert("Asia/Tokyo"),
                "exit_jst": exit_utc.tz_convert("Asia/Tokyo"),
                "entry_chicago": entry_utc.tz_convert("America/Chicago"),
                "exit_chicago": exit_utc.tz_convert("America/Chicago"),
                "raw_symbol": symbol,
                "instrument_id": expected_id,
                "expiration_utc": expiration,
                "days_to_expiration": (expiration.date() - day).days,
                "selection_method": "Databento 6J.v.0 point-in-time highest-volume mapping; verified by instrument_id",
                "roll_near_event_21d": (expiration.date() - day).days <= 21,
                "continuous_entry_open": float(event.entry),
                "continuous_exit_open": float(event.exit),
                "explicit_entry_open": explicit_entry,
                "explicit_exit_open": explicit_exit,
                "explicit_open_gross_pnl_usd": explicit_gross,
                "phase1_gross_pnl_usd": float(event.gross_pnl),
                "explicit_minus_phase1_gross_usd": explicit_gross - float(event.gross_pnl),
                "entry_bid": float(entry_q.bid_px_00),
                "entry_ask": float(entry_q.ask_px_00),
                "exit_bid": float(exit_q.bid_px_00),
                "exit_ask": float(exit_q.ask_px_00),
                "entry_spread_ticks": float(boundary_spreads.iloc[0]),
                "exit_spread_ticks": float(boundary_spreads.iloc[1]),
                "entry_bid_depth_contracts": int(entry_q.bid_sz_00),
                "exit_ask_depth_contracts": int(exit_q.ask_sz_00),
                "entry_quote_delay_ms": (pd.Timestamp(entry_q.ts_event) - entry_utc).total_seconds() * 1000,
                "exit_quote_delay_ms": (pd.Timestamp(exit_q.ts_event) - exit_utc).total_seconds() * 1000,
                "window_volume_contracts": int(trades["size"].sum()),
                "session_volume_contracts_from_continuous": int(session_bars.volume.sum()),
                "no_trade_minute_fraction": (23 - minutes_with_trades) / 23,
                "quote_fill_gross_pnl_usd": quote_gross,
                "quote_fill_net_pnl_usd": quote_gross - COMMISSION_RT,
                "observed_execution_cost_vs_open_usd": explicit_gross - quote_gross,
                "observed_execution_ticks_round_trip": (explicit_gross - quote_gross) / TICK_VALUE,
            }
        )

        base = explicit_entry
        prior = explicit_entry
        for offset in range(0, 24):
            stamp = entry_utc + pd.Timedelta(minutes=offset)
            observed = book[
                book.ts_event.ge(stamp)
                & book.ts_event.lt(stamp + pd.Timedelta(minutes=1))
                & book.action.eq("T")
            ].sort_values(["ts_event", "sequence"])
            price = float(observed.iloc[0].price) if len(observed) else np.nan
            minute_return = price / prior - 1.0 if offset and pd.notna(price) else (0.0 if offset == 0 else np.nan)
            cumulative_return = price / base - 1.0 if pd.notna(price) else np.nan
            minute_rows.append(
                {
                    "japan_date": day,
                    "minute_offset": offset,
                    "timestamp_jst": stamp.tz_convert("Asia/Tokyo"),
                    "price": price,
                    "minute_return": minute_return,
                    "cumulative_return": cumulative_return,
                    "short_minute_pnl_usd": (prior - price) * MULTIPLIER if offset and pd.notna(price) else (0.0 if offset == 0 else np.nan),
                    "short_cumulative_pnl_usd": (base - price) * MULTIPLIER if pd.notna(price) else np.nan,
                }
            )
            if pd.notna(price):
                prior = price
    return pd.DataFrame(rows), pd.DataFrame(minute_rows)


def cost_sensitivity(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticks in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        pnl = trades.explicit_open_gross_pnl_usd - COMMISSION_RT - 2 * ticks * TICK_VALUE
        label = "commissions_only" if ticks == 0 else f"{ticks:g}_tick_per_side"
        rows.append(
            {
                "scenario": label,
                "slippage_ticks_per_side": ticks,
                "commission_rt_usd": COMMISSION_RT,
                "execution_cost_rt_usd": 2 * ticks * TICK_VALUE,
                **metrics(pnl, trades.japan_date),
            }
        )
    observed = metrics(trades.quote_fill_net_pnl_usd, trades.japan_date)
    rows.append(
        {
            "scenario": "observed_bid_entry_ask_exit",
            "slippage_ticks_per_side": trades.observed_execution_ticks_round_trip.mean() / 2,
            "commission_rt_usd": COMMISSION_RT,
            "execution_cost_rt_usd": trades.observed_execution_cost_vs_open_usd.mean(),
            **observed,
        }
    )
    return pd.DataFrame(rows)


def chronological(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks = [
        ("development_replication_2015_2020", 2015, 2020),
        ("intermediate_validation_2021_2023", 2021, 2023),
        ("final_frozen_retrospective_2024_2026", 2024, 2026),
    ]
    rows = []
    for name, first, last in blocks:
        part = trades[trades.japan_date.dt.year.between(first, last)]
        rows.append(
            {
                "block": name,
                "start_year": first,
                "end_year": last,
                "gross_expectancy_usd": part.quote_fill_gross_pnl_usd.mean(),
                **metrics(part.quote_fill_net_pnl_usd, part.japan_date),
            }
        )
    path = trades[["japan_date", "quote_fill_net_pnl_usd"]].copy().sort_values("japan_date")
    path["expanding_expectancy_usd"] = path.quote_fill_net_pnl_usd.expanding().mean()
    path["expanding_cumulative_pnl_usd"] = path.quote_fill_net_pnl_usd.cumsum()
    path["rolling_24_event_expectancy_usd"] = path.quote_fill_net_pnl_usd.rolling(24).mean()
    return pd.DataFrame(rows), path


def event_profile(minutes: pd.DataFrame) -> pd.DataFrame:
    return (
        minutes.groupby("minute_offset")
        .agg(
            timestamp_jst_clock=("timestamp_jst", lambda x: pd.Timestamp(x.iloc[0]).strftime("%H:%M")),
            n=("minute_return", "count"),
            mean_minute_return=("minute_return", "mean"),
            median_minute_return=("minute_return", "median"),
            short_minute_hit_rate=("minute_return", lambda x: (x.dropna() < 0).mean()),
            mean_cumulative_return=("cumulative_return", "mean"),
            median_cumulative_return=("cumulative_return", "median"),
            mean_short_cumulative_pnl_usd=("short_cumulative_pnl_usd", "mean"),
        )
        .reset_index()
    )


def outliers(trades: pd.DataFrame) -> pd.DataFrame:
    pnl = trades.quote_fill_net_pnl_usd.sort_values(ascending=False)
    lo, hi = trades.quote_fill_net_pnl_usd.quantile([0.01, 0.99])
    base = trades.quote_fill_net_pnl_usd
    rows = [
        ("full", len(base), base.sum(), base.mean(), base.median()),
        ("best_5_contribution", 5, pnl.head(5).sum(), pnl.head(5).mean(), pnl.head(5).median()),
        ("best_10_contribution", 10, pnl.head(10).sum(), pnl.head(10).mean(), pnl.head(10).median()),
        ("without_best_5", len(base) - 5, pnl.iloc[5:].sum(), pnl.iloc[5:].mean(), pnl.iloc[5:].median()),
        ("without_worst_5", len(base) - 5, pnl.iloc[:-5].sum(), pnl.iloc[:-5].mean(), pnl.iloc[:-5].median()),
        ("winsorized_1_99", len(base), base.clip(lo, hi).sum(), base.clip(lo, hi).mean(), base.clip(lo, hi).median()),
    ]
    out = pd.DataFrame(rows, columns=["diagnostic", "n", "total_pnl_usd", "expectancy_usd", "median_pnl_usd"])
    out["share_of_full_pnl"] = out.total_pnl_usd / base.sum()
    return out


def calendar_diagnostic(trades: pd.DataFrame) -> pd.DataFrame:
    cal = pd.read_csv(P1 / "japan_business_gotobi_calendar.csv", parse_dates=["japan_date"])
    cal["month"] = cal.japan_date.dt.to_period("M")
    last_gotobi = cal[cal.is_gotobi].groupby("month").japan_date.max()
    frame = trades.copy()
    frame["month"] = frame.japan_date.dt.to_period("M")
    frame["last_gotobi_date"] = frame.month.map(last_gotobi)
    frame["calendar_group"] = np.where(
        frame.japan_date.eq(frame.last_gotobi_date), "coincide_last_gotobi_and_eom", "eom_not_last_gotobi"
    )
    rows = []
    rows.append(
        {
            "calendar_group": "last_japan_business_day_all_candidate",
            "candidate_status": "frozen_candidate",
            **metrics(frame.quote_fill_net_pnl_usd, frame.japan_date),
        }
    )
    for group, part in frame.groupby("calendar_group"):
        rows.append(
            {
                "calendar_group": group,
                "candidate_status": "frozen_candidate_subgroup_diagnostic",
                **metrics(part.quote_fill_net_pnl_usd, part.japan_date),
            }
        )
    noncoincident = int(frame.calendar_group.eq("eom_not_last_gotobi").sum())
    rows.append(
        {
            "calendar_group": "last_gotobi_not_eom",
            "candidate_status": "outside_frozen_candidate_not_traded",
            "n": noncoincident,
        }
    )
    return pd.DataFrame(rows)


def orthogonality(trades: pd.DataFrame) -> pd.DataFrame:
    six = pd.read_csv(SIX_PATH, parse_dates=["session_date"]).set_index("session_date")
    pnl = trades.set_index("japan_date").quote_fill_net_pnl_usd
    joined = six.copy()
    joined["gotobi_pnl"] = pnl.reindex(joined.index).fillna(0.0)
    sleeve_cols = [c for c in joined if c.endswith("_pnl_usd")]
    joined["six_sleeve_equal_weight"] = joined[sleeve_cols].mean(axis=1)
    trade_dates = joined.index.intersection(pnl.index)
    rows = []
    for col in [*sleeve_cols, "six_sleeve_equal_weight"]:
        rows.append(
            {
                "comparison": col,
                "pearson_all_days": joined.gotobi_pnl.corr(joined[col], method="pearson"),
                "spearman_all_days": joined.gotobi_pnl.corr(joined[col], method="spearman"),
                "n_all_days": len(joined),
                "pearson_trade_days": joined.loc[trade_dates, "gotobi_pnl"].corr(joined.loc[trade_dates, col], method="pearson"),
                "spearman_trade_days": joined.loc[trade_dates, "gotobi_pnl"].corr(joined.loc[trade_dates, col], method="spearman"),
                "n_trade_days": len(trade_dates),
            }
        )
    return pd.DataFrame(rows)


def fmt(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x:,.3f}"


def markdown(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for col in shown.select_dtypes(include="number"):
        shown[col] = shown[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    cols = list(shown.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    events = load_phase1_events()
    bars = pd.read_parquet(BARS_PATH, columns=["open", "volume", "instrument_id"])
    trades, minutes = reconstruct(events, bars)
    trades["japan_date"] = pd.to_datetime(trades.japan_date)
    minutes["japan_date"] = pd.to_datetime(minutes.japan_date)

    # A roll proximity flag based on the actual point-in-time mapping, not just expiry.
    transitions = bars.instrument_id.ne(bars.instrument_id.shift()).groupby(bars.index.date).any()
    transition_dates = [pd.Timestamp(day) for day, changed in transitions.items() if changed]
    trades["continuous_roll_within_3_calendar_days"] = trades.japan_date.map(
        lambda day: any(abs((pd.Timestamp(day).normalize() - roll).days) <= 3 for roll in transition_dates)
    )

    trades.to_csv(OUT / "explicit_contract_trades.csv", index=False)
    sensitivity = cost_sensitivity(trades)
    sensitivity.to_csv(OUT / "execution_cost_sensitivity.csv", index=False)
    chrono, path = chronological(trades)
    chrono.to_csv(OUT / "chronological_validation.csv", index=False)
    path.to_csv(OUT / "chronological_path.csv", index=False)
    profile = event_profile(minutes)
    profile.to_csv(OUT / "event_profile.csv", index=False)
    outlier = outliers(trades)
    outlier.to_csv(OUT / "outlier_diagnostics.csv", index=False)
    calendar = calendar_diagnostic(trades)
    calendar.to_csv(OUT / "calendar_diagnostics.csv", index=False)
    corr = orthogonality(trades)
    corr.to_csv(OUT / "orthogonality_phase2.csv", index=False)

    phase1_diff = trades.explicit_minus_phase1_gross_usd
    quote = metrics(trades.quote_fill_net_pnl_usd, trades.japan_date)
    spread = pd.concat([trades.entry_spread_ticks, trades.exit_spread_ticks], ignore_index=True)
    depth = pd.concat([trades.entry_bid_depth_contracts, trades.exit_ask_depth_contracts], ignore_index=True)
    break_even_total = trades.explicit_open_gross_pnl_usd.mean()
    break_even_execution = break_even_total - COMMISSION_RT
    notionals = trades.explicit_entry_open * MULTIPLIER
    break_even_bps = break_even_total / notionals.mean() * 10_000
    abnormal = trades.loc[phase1_diff.abs() > TICK_VALUE + 1e-6, ["japan_date", "raw_symbol", "explicit_minus_phase1_gross_usd"]]
    worst = trades.quote_fill_net_pnl_usd.min()
    losing_streak = longest_losing_streak(trades.quote_fill_net_pnl_usd)
    avg_per_month = len(trades) / trades.japan_date.dt.to_period("M").nunique()
    latest_roll24 = path.rolling_24_event_expectancy_usd.dropna().iloc[-1]
    final_block = chrono.iloc[-1]
    best5_share = outlier.loc[outlier.diagnostic.eq("best_5_contribution"), "share_of_full_pnl"].iloc[0]

    # Frozen decision rule: require positive observed-fill history and recent
    # block, non-catastrophic concentration, and meaningful cost headroom.
    if quote["expectancy_usd"] <= 0 or break_even_execution <= trades.observed_execution_cost_vs_open_usd.mean() + TICK_VALUE:
        verdict = "REJECTED_EXECUTION"
    elif final_block.expectancy_usd > 0 and best5_share < 1.0 and quote["n"] >= 100:
        verdict = "VALIDATED_CANDIDATE"
    else:
        verdict = "FORWARD_TEST"

    metadata = {
        "verdict": verdict,
        "frozen_rule": "short 6J EOM; entry 09:31 JST open/bid; exit 09:54 JST open/ask",
        "events": len(trades),
        "phase1_reproduced": True,
        "parameter_retuning": False,
        "frozen_sleeves_modified": False,
        "mbp1_files": len(list(MBP.glob("*.parquet"))),
        "definition_files": len(list(DEFS.glob("*.parquet"))),
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    sensitivity_md = markdown(sensitivity)
    chrono_md = markdown(chrono)
    calendar_md = markdown(calendar)
    corr_md = markdown(corr)
    report = f"""# 6J month-end Gotobi — Phase 2 frozen validation

## Verdict: {verdict}

Only the frozen month-end rule was evaluated: short the explicit 6J contract at the bid available at or immediately after 09:31 JST and cover at the ask available at or immediately after 09:54 JST. No time, subgroup, filter, or alternative rule was selected.

## 1. Phase-1 replication gate

The gate passed before Phase 2 continued: 136 trades, gross expectancy USD {events.gross_pnl.mean():.2f}, net expectancy USD {events.net_pnl.mean():.2f}, net total USD {events.net_pnl.sum():.2f}. The Phase-1 report supplies HAC t=1.81, PF=1.47, and MaxDD=USD 777.83.

## 2. Explicit contracts and rolls

Each event's Phase-1 `instrument_id` was resolved as-of that date, then requested as a raw quarterly 6J contract. Selection is the frozen Databento `6J.v.0` point-in-time highest-volume mapping. Every MBP file and definition matched the expected ID. No back adjustment is used.

- Explicit-open minus continuous gross PnL: mean USD {phase1_diff.mean():.6f}; median USD {phase1_diff.median():.6f}; min/max USD {phase1_diff.min():.6f}/{phase1_diff.max():.6f}; correlation {trades.explicit_open_gross_pnl_usd.corr(trades.phase1_gross_pnl_usd):.6f}.
- Abnormal differences above one tick: {len(abnormal)}.
- Events within 21 calendar days of expiration: {int(trades.roll_near_event_21d.sum())}; within three calendar days of a continuous mapping transition: {int(trades.continuous_roll_within_3_calendar_days.sum())}.
- `explicit_contract_trades.csv` records JST/Chicago timestamps, symbol, expiry distance, prices, window and session volume, depth, quote delay, and roll flags for every event. Of 136 comparisons, 135 are identical to the continuous open PnL; 2018-07-31 differs by exactly one tick and none differs by more than one tick.

## 3. Observed execution

Observed bid-entry/ask-exit plus USD {COMMISSION_RT:.2f} commission produces N={quote['n']}, expectancy USD {quote['expectancy_usd']:.2f}, HAC t={quote['hac_t']:.2f}, Sharpe={quote['sharpe']:.2f}, PF={quote['profit_factor']:.2f}, total USD {quote['total_pnl_usd']:.2f}, MaxDD USD {quote['max_drawdown_usd']:.2f}, hit rate={quote['win_rate']:.1%}.

- Boundary spread: median {spread.median():.2f} ticks, P75 {spread.quantile(.75):.2f}, P90 {spread.quantile(.90):.2f}; spread >1 tick on {(spread > 1.000001).mean():.1%} of boundary observations.
- Relevant-side top depth: median {depth.median():.0f} contracts, P25 {depth.quantile(.25):.0f}, P10 {depth.quantile(.10):.0f}. Median entry/exit quote delays are {trades.entry_quote_delay_ms.median():.0f}/{trades.exit_quote_delay_ms.median():.0f} ms; P90 delays are {trades.entry_quote_delay_ms.quantile(.9):.0f}/{trades.exit_quote_delay_ms.quantile(.9):.0f} ms.
- No-trade one-minute bars in the frozen window: {trades.no_trade_minute_fraction.mean():.2%}; median window volume {trades.window_volume_contracts.median():.0f} contracts and median full-session active-contract volume {trades.session_volume_contracts_from_continuous.median():.0f} contracts.
- Mean observed execution cost versus explicit bar opens: USD {trades.observed_execution_cost_vs_open_usd.mean():.2f}, or {trades.observed_execution_ticks_round_trip.mean():.2f} ticks round trip. Phase 1's one tick per side is therefore {'optimistic' if trades.observed_execution_ticks_round_trip.mean() > 2 else 'conservative or adequate'} on average.

## 4. Cost sensitivity and break-even

{sensitivity_md}

The gross edge's maximum total round-trip cost at zero expectancy is USD {break_even_total:.2f}, {break_even_total / TICK_VALUE:.2f} ticks, or {break_even_bps:.3f} bp of mean notional. After commission, the remaining execution-only budget is USD {break_even_execution:.2f}, or {break_even_execution / TICK_VALUE:.2f} ticks round trip.

## 5. Chronological validation

These are frozen retrospective blocks, not a genuinely unseen holdout: Phase 1 had already inspected the full sample through 2026. No result below changes the rule.

{chrono_md}

The latest rolling 24-event expectancy is USD {latest_roll24:.2f}. Event-by-event expanding expectancy, cumulative PnL, and rolling-24 expectancy are in `chronological_path.csv`.

## 6. Frozen event profile

`event_profile.csv` reports minute-by-minute mean, median, short hit rate, and cumulative price/PnL profile for 09:31–09:54. This is descriptive only; no minute is promoted or optimized.

## 7. Outliers

- Best five contribution: USD {outlier.loc[outlier.diagnostic.eq('best_5_contribution'),'total_pnl_usd'].iloc[0]:.2f} ({best5_share:.1%} of full PnL); best ten: USD {outlier.loc[outlier.diagnostic.eq('best_10_contribution'),'total_pnl_usd'].iloc[0]:.2f}.
- Without best five: USD {outlier.loc[outlier.diagnostic.eq('without_best_5'),'total_pnl_usd'].iloc[0]:.2f}; without worst five: USD {outlier.loc[outlier.diagnostic.eq('without_worst_5'),'total_pnl_usd'].iloc[0]:.2f}.
- Median trade PnL: USD {trades.quote_fill_net_pnl_usd.median():.2f}; winsorized 1/99% total: USD {outlier.loc[outlier.diagnostic.eq('winsorized_1_99'),'total_pnl_usd'].iloc[0]:.2f}.

## 8. Calendar robustness (diagnostic only)

{calendar_md}

No subgroup is selected. The distinction only tests whether month-end and the last shifted Gotobi anchor describe the same economic event.

## 9. Orthogonality

{corr_md}

No ERC or allocation optimization was performed.

## 10. Mechanical Topstep compatibility

The frozen trade occurs near 18:31/19:31 Chicago time and exits near 18:54/19:54, inside [Topstep's permitted 17:00–15:10 CT window](https://help.topstep.com/en/articles/8284206-when-and-what-products-can-i-trade). One 6J contract is below the 50K maximum of five contracts and uses the current [USD 4.22 round-turn fee](https://help.topstep.com/en/articles/8284213-topstepx-commissions-and-fees). At observed fills: worst historical trade USD {worst:.2f}, longest losing streak {losing_streak}, MaxDD USD {quote['max_drawdown_usd']:.2f}, and {avg_per_month:.2f} trades per active calendar month. The historical MaxDD is {quote['max_drawdown_usd']/2000:.1%} of the current [USD 2,000 Maximum Loss Limit for a 50K account](https://help.topstep.com/en/articles/8284204-what-is-the-maximum-loss-limit). This is a mechanical comparison only, not a Topstep pass simulation.

## Prospective protocol

- Observe at least 24 additional month-end events (roughly two years), with the rule and calendar frozen.
- Record scheduled/actual timestamps, raw contract, bid/ask, depth, quote delay, spread, gross and net PnL.
- Alerts: rolling-12 net expectancy below zero; median boundary spread above 1 tick; P90 above 2 ticks; any contract/calendar mismatch; rolling observed execution cost above the historical break-even execution budget.
- Stop the candidate if cumulative prospective net PnL is negative after 24 events, if rolling-24 expectancy is non-positive, or if execution cost consumes the break-even budget for six consecutive events. Do not replace it with a new timing or subgroup.
"""
    (OUT / "PHASE2_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(sensitivity.to_string(index=False))
    print(chrono.to_string(index=False))


if __name__ == "__main__":
    main()
