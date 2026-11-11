"""IS-only macro and risk-unit diagnostics for Signals A and B."""

from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostic_is.run_step1 import (  # noqa: E402
    A_OOS_START,
    B_OOS_START,
    GAP_SOURCES,
    TEMP,
    _assert_is_only_bars,
    _write_is_only_parquet,
)
from research.mean_reversion_cross_asset_20261006.run_campaign import (  # noqa: E402
    INSTRUMENTS as GAP_INSTRUMENTS,
    _load_bars,
)


OUT = ROOT / "diagnostic_is"
STATE = OUT / "state_tables"
TABLES = OUT / "risk_tables"
CALENDAR = OUT / "macro_calendar_official.csv"
SEED = 20261008
BOOTSTRAPS = 2000
A_START = pd.Timestamp("2010-06-07")
B_START = pd.Timestamp("2015-01-02")


def add_r(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["gross_R"] = x.gross_daily_atr14
    x["cost_R"] = x.cost_daily_atr14
    x["net_R"] = x.gross_R - x.cost_R
    return x


def macro_calendar() -> pd.DataFrame:
    x = pd.read_csv(CALENDAR)
    x["date"] = pd.to_datetime(x.date)
    if x.duplicated(["date", "event"]).any():
        raise AssertionError("Duplicate official macro event rows")
    return x


def add_macro_flags(frame: pd.DataFrame, calendar: pd.DataFrame, signal: str) -> pd.DataFrame:
    x = frame.copy()
    dates = pd.to_datetime(x.session_date)
    by_date = calendar.groupby("date").event.agg(lambda s: "+".join(sorted(set(s))))
    x["macro_events"] = dates.map(by_date).fillna("none")
    x[f"{signal}1_macro_day"] = np.where(x.macro_events.eq("none"), "no_macro", "macro")
    return x


def add_b1(frame: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    dates = pd.to_datetime(x.session_date)
    cpi_nfp = set(calendar.loc[calendar.event.isin(["CPI", "NFP"]), "date"])
    fomc = set(calendar.loc[calendar.event.eq("FOMC"), "date"])
    times = pd.DatetimeIndex(pd.to_datetime(x.climax_time, utc=True)).tz_convert("America/New_York")
    mins = times.hour * 60 + times.minute
    morning = dates.isin(cpi_nfp).to_numpy() & (mins >= 8 * 60 + 30) & (mins < 8 * 60 + 45)
    afternoon = dates.isin(fomc).to_numpy() & (mins >= 14 * 60) & (mins < 14 * 60 + 30)
    x["B1_announcement_bar"] = np.where(morning | afternoon, "announcement", "outside")
    x["B1_event"] = np.where(morning, x.macro_events, np.where(afternoon, "FOMC", "none"))
    return x


def clustered_difference(a: pd.DataFrame, b: pd.DataFrame, value: str) -> tuple[float, float, float]:
    combined = pd.concat([a.assign(_x=1.0), b.assign(_x=0.0)], ignore_index=True).dropna(subset=[value]).reset_index(drop=True)
    y = combined[value].to_numpy(float)
    x = np.column_stack([np.ones(len(combined)), combined._x])
    inv = np.linalg.pinv(x.T @ x)
    beta = inv @ x.T @ y
    resid = y - x @ beta
    meat = np.zeros((2, 2))
    for _, idx in combined.groupby("session_date").groups.items():
        score = x[list(idx)].T @ resid[list(idx)]
        meat += np.outer(score, score)
    cov = inv @ meat @ inv
    se = math.sqrt(max(cov[1, 1], 0))
    t = beta[1] / se if se else np.nan
    return float(beta[1]), float(t), float(2 * stats.norm.sf(abs(t))) if np.isfinite(t) else np.nan


def bootstrap_difference(a: pd.DataFrame, b: pd.DataFrame, value: str, rng: np.random.Generator) -> tuple[float, float]:
    dates = np.array(sorted(set(a.session_date) | set(b.session_date)))
    av = a.groupby("session_date")[value].mean().reindex(dates).to_numpy()
    bv = b.groupby("session_date")[value].mean().reindex(dates).to_numpy()
    n, block = len(dates), 5
    starts = np.arange(max(n - block + 1, 1))
    draws = np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        idx: list[int] = []
        while len(idx) < n:
            start = int(rng.choice(starts))
            idx.extend(range(start, min(start + block, n)))
        take = np.asarray(idx[:n])
        draws[i] = np.nanmean(av[take]) - np.nanmean(bv[take])
    return float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975))


def tests_for_variable(frame: pd.DataFrame, signal: str, variable: str, rng: np.random.Generator) -> pd.DataFrame:
    buckets = sorted(str(x) for x in frame[variable].dropna().unique())
    rows = []
    if variable == "B1_announcement_bar" and buckets == ["outside"]:
        return pd.DataFrame([{
            "signal": signal, "variable": variable, "bucket_a": "announcement", "bucket_b": "outside",
            "mean_net_R_difference_a_minus_b": np.nan, "cluster_t_R": np.nan, "p_value_R": np.nan,
            "bootstrap95_R_low": np.nan, "bootstrap95_R_high": np.nan,
            "mean_net_usd_a": np.nan, "mean_net_usd_b": frame.net_pnl.mean(),
            "mean_net_usd_difference_a_minus_b": np.nan, "estimable": False,
        }])
    for first, second in combinations(buckets, 2):
        a = frame[frame[variable].astype(str).eq(first)]
        b = frame[frame[variable].astype(str).eq(second)]
        diff, t, p = clustered_difference(a, b, "net_R")
        low, high = bootstrap_difference(a, b, "net_R", rng)
        rows.append({
            "signal": signal, "variable": variable, "bucket_a": first, "bucket_b": second,
            "mean_net_R_difference_a_minus_b": diff, "cluster_t_R": t, "p_value_R": p,
            "bootstrap95_R_low": low, "bootstrap95_R_high": high,
            "mean_net_usd_a": a.net_pnl.mean(), "mean_net_usd_b": b.net_pnl.mean(),
            "mean_net_usd_difference_a_minus_b": a.net_pnl.mean() - b.net_pnl.mean(),
            "estimable": True,
        })
    return pd.DataFrame(rows)


def holm(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    p = out.p_value_R.dropna().sort_values()
    # The multiplicity is the complete set of planned hypotheses. B1 remains
    # in the family even when its announcement bucket has zero observations.
    m = len(out)
    adjusted = pd.Series(index=p.index, dtype=float)
    running = 0.0
    for rank, (idx, value) in enumerate(p.items()):
        running = max(running, (m - rank) * value)
        adjusted.loc[idx] = min(running, 1.0)
    out["holm_p_R"] = adjusted
    out["holm_reject_5pct_R"] = out.holm_p_R <= 0.05
    return out


def metrics(frame: pd.DataFrame, signal: str, variable: str) -> pd.DataFrame:
    rows = []
    for bucket, group in frame.dropna(subset=[variable, "net_R"]).groupby(variable, observed=True):
        r = group.net_R
        usd = group.net_pnl
        rows.append({
            "signal": signal, "variable": variable, "bucket": str(bucket), "trades": len(group),
            "gross_R": group.gross_R.sum(), "cost_R": group.cost_R.sum(), "net_R": r.sum(),
            "mean_gross_R": group.gross_R.mean(), "mean_cost_R": group.cost_R.mean(), "mean_net_R": r.mean(),
            "gross_usd": group.gross_pnl.sum(), "cost_usd": group.cost.sum(), "net_usd": usd.sum(),
            "mean_net_usd": usd.mean(), "hit_rate_R": (r > 0).mean(),
            "profit_factor_R": r[r > 0].sum() / abs(r[r < 0].sum()) if (r < 0).any() else np.inf,
            "t_stat_R_unclustered": r.mean() / (r.std(ddof=1) / math.sqrt(len(r))) if len(r) > 1 and r.std(ddof=1) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def coverage(calendar: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in range(2010, 2025):
        for event in ("CPI", "NFP", "FOMC"):
            same = calendar.event.eq(event) & calendar.date.dt.year.eq(year)
            all_dates = calendar.loc[same, "date"]
            rows.append({
                "year": year, "event": event, "official_page_dates": len(all_dates),
                "signal_A_IS_dates": int(((all_dates >= A_START) & (all_dates < A_OOS_START)).sum()),
                "signal_B_IS_dates": int(((all_dates >= B_START) & (all_dates < B_OOS_START)).sum()),
                "first_official_date": all_dates.min().date().isoformat() if len(all_dates) else None,
                "last_official_date": all_dates.max().date().isoformat() if len(all_dates) else None,
            })
    return pd.DataFrame(rows)


def b0_tables(b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = b.dropna(subset=["net_R"]).copy()
    z["year"] = pd.to_datetime(z.session_date).dt.year
    yearly = z.groupby("year", as_index=False).agg(
        trades=("net_R", "size"), gross_usd=("gross_pnl", "sum"), cost_usd=("cost", "sum"), net_usd=("net_pnl", "sum"),
        gross_R=("gross_R", "sum"), cost_R=("cost_R", "sum"), net_R=("net_R", "sum"), mean_net_R=("net_R", "mean"),
    )
    liquidity = pd.read_csv(STATE / "B0a_mgc_liquidity_yearly.csv")
    yearly = liquidity.merge(yearly, on="year", how="left")
    roll = z.groupby("roll_window", as_index=False).agg(
        trades=("net_R", "size"), gross_usd=("gross_pnl", "sum"), cost_usd=("cost", "sum"), net_usd=("net_pnl", "sum"),
        gross_R=("gross_R", "sum"), cost_R=("cost_R", "sum"), net_R=("net_R", "sum"), mean_net_R=("net_R", "mean"),
    )
    return yearly, roll


def tail_paths(a: pd.DataFrame, calendar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nq = a[a.asset.eq("NQ")].dropna(subset=["net_R"]).copy()
    chosen = nq.sort_values("net_R").groupby("bucket", observed=True, group_keys=False).head(10).copy()
    temp_path = TEMP / "risk_tail_A_NQ_is.parquet"
    TEMP.mkdir(exist_ok=True)
    try:
        _write_is_only_parquet(GAP_SOURCES["NQ"], temp_path, A_OOS_START)
        bars = _load_bars(temp_path, GAP_INSTRUMENTS["NQ"])
        _assert_is_only_bars(bars, A_OOS_START)
    finally:
        temp_path.unlink(missing_ok=True)
        try:
            TEMP.rmdir()
        except OSError:
            pass
    idx = pd.DatetimeIndex(bars.index)
    point_value = GAP_INSTRUMENTS["NQ"].point_value
    rows = []
    for _, trade in chosen.iterrows():
        entry = pd.Timestamp(trade.entry_time)
        exit_ = pd.Timestamp(trade.exit_time)
        path = bars.loc[(idx >= entry) & (idx <= exit_), ["high", "low"]]
        if path.empty:
            raise AssertionError(f"No NQ path for {entry}")
        adverse_usd = (trade.entry_price - path.low) * point_value if trade.direction == 1 else (path.high - trade.entry_price) * point_value
        cumulative_R = adverse_usd.clip(lower=0).cummax() / (trade.prior_daily_atr14 * point_value)
        elapsed = (pd.DatetimeIndex(path.index) - entry).total_seconds() / 60 + 5
        def at(minutes: int) -> float:
            valid = cumulative_R[elapsed <= minutes]
            return float(valid.iloc[-1]) if len(valid) else np.nan
        peak_pos = int(np.argmax(cumulative_R.to_numpy()))
        recovery = float(trade.recovery_fraction_20m)
        move_type = "reprise" if recovery > 0 else "extension" if recovery < 0 else "inchangé"
        rows.append({
            "bucket": trade.bucket, "session_date": trade.session_date, "entry_time": trade.entry_time,
            "exit_time": trade.exit_time, "direction": "long" if trade.direction == 1 else "short",
            "exit_reason": trade.exit_reason, "gross_usd": trade.gross_pnl, "net_usd": trade.net_pnl,
            "gross_R": trade.gross_R, "cost_R": trade.cost_R, "loss_net_R": trade.net_R,
            "mae_stored_usd": trade.mae_usd, "mae_peak_R": float(cumulative_R.max()),
            "mae_peak_time": str(cumulative_R.index[peak_pos]), "mae_by_5m_R": at(5),
            "mae_by_10m_R": at(10), "mae_by_20m_R": at(20),
            "mae_path_cumulative_R": "; ".join(f"{int(m)}m:{v:.4f}" for m, v in zip(elapsed, cumulative_R)),
            "vix_t_minus_1": trade.vix, "vix3m_t_minus_1": trade.vix3m,
            "vix_over_vix3m_t_minus_1": trade.vix / trade.vix3m,
            "first20m_gap_path": move_type, "first20m_gap_recovery_fraction": recovery,
            "macro_publication_day": trade.macro_events != "none", "macro_events": trade.macro_events,
        })
    details = pd.DataFrame(rows).sort_values(["bucket", "loss_net_R"])
    concentration = []
    for bucket, group in nq.groupby("bucket", observed=True):
        worst = group.nsmallest(10, "net_R")
        total = group.net_R.sum()
        negative_loss = -group.loc[group.net_R < 0, "net_R"].sum()
        worst_loss = -worst.net_R.sum()
        concentration.append({
            "bucket": bucket, "trades": len(group), "bucket_net_R": total,
            "worst10_net_R": worst.net_R.sum(),
            "worst10_share_of_bucket_net_loss_R": worst_loss / (-total) if total < 0 else np.nan,
            "worst10_share_of_all_negative_trade_R": worst_loss / negative_loss if negative_loss > 0 else np.nan,
        })
    return details, pd.DataFrame(concentration)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    calendar = macro_calendar()
    a = add_r(pd.read_csv(STATE / "signal_a_state_trades.csv"))
    b = add_r(pd.read_csv(STATE / "signal_b_state_trades.csv"))
    a = add_macro_flags(a, calendar, "A")
    b = add_b1(add_macro_flags(b, calendar, "B"), calendar)
    if pd.to_datetime(a.session_date).max() >= A_OOS_START or pd.to_datetime(b.session_date).max() >= B_OOS_START:
        raise AssertionError("Post-IS trade entered risk diagnostics")
    a.to_csv(TABLES / "signal_a_macro_risk_trades.csv", index=False)
    b.to_csv(TABLES / "signal_b_macro_risk_trades.csv", index=False)
    cov = coverage(calendar)
    cov.to_csv(TABLES / "official_macro_coverage_by_year.csv", index=False)

    a_vars = ["A2_term_structure", "A2_vix_tercile", "A3_ret20", "A3_sma50", "A4_recovery_tercile", "A5_gap_bucket", "A5_gap_atr_tercile", "A6_direction", "A1_macro_day"]
    b_vars = ["roll_window", "session_bucket", "B3_ret50", "B3_sma100", "B4_gold_rv20_tercile", "B5_volume_tercile", "B5_move_tercile", "B1_announcement_bar"]
    frames = [(a[a.asset.eq("NQ")], "A", a_vars), (b, "B", b_vars)]
    rng = np.random.default_rng(SEED)
    all_metrics, all_tests = [], []
    for frame, signal, variables in frames:
        for variable in variables:
            all_metrics.append(metrics(frame, signal, variable))
            all_tests.append(tests_for_variable(frame, signal, variable, rng))
    metric_frame = pd.concat(all_metrics, ignore_index=True)
    tests = holm(pd.concat(all_tests, ignore_index=True))
    if len(tests) != 36:
        raise AssertionError(f"Expected 36 total contrasts (34 old + A1/B1), got {len(tests)}")
    metric_frame.to_csv(TABLES / "bucket_metrics_R_primary_usd_secondary.csv", index=False)
    tests.to_csv(TABLES / "all_36_tests_holm_R.csv", index=False)

    b0a, b0d = b0_tables(b)
    b0a.to_csv(TABLES / "B0a_liquidity_and_pnl_yearly_R.csv", index=False)
    b0d.to_csv(TABLES / "B0d_roll_vs_outside_R.csv", index=False)
    tails, concentration = tail_paths(a, calendar)
    tails.to_csv(TABLES / "A_NQ_worst10_each_bucket_R_details.csv", index=False)
    concentration.to_csv(TABLES / "A_NQ_worst10_each_bucket_R_concentration.csv", index=False)

    metadata = {
        "seed": SEED,
        "bootstrap_draws": BOOTSTRAPS,
        "holm_family_size": len(tests),
        "estimable_tests": int(tests.p_value_R.notna().sum()),
        "holm_family_definition": "34 previously specified pairwise state contrasts plus binary A1 and B1, all tested once on net R",
        "R_definition": "trade PnL USD / (prior-session Wilder ATR14 price units * contract point value)",
        "usd_role": "descriptive secondary metric; not an additional hypothesis-test family",
        "A_IS": [str(A_START.date()), str((A_OOS_START - pd.Timedelta(days=1)).date())],
        "B_IS": [str(B_START.date()), str((B_OOS_START - pd.Timedelta(days=1)).date())],
        "post_is_statistics_computed": False,
        "holm_rejections_5pct": int(tests.holm_reject_5pct_R.sum()),
    }
    (TABLES / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "tests": len(tests), "holm_rejections": int(tests.holm_reject_5pct_R.sum()),
        "A1_counts": a[a.asset.eq("NQ")].A1_macro_day.value_counts().to_dict(),
        "B1_counts": b.B1_announcement_bar.value_counts().to_dict(),
        "tail_rows": len(tails),
    }, indent=2))


if __name__ == "__main__":
    main()
