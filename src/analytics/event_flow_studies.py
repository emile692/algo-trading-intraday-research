"""Frozen phase-1/2 event studies for the three event/flow mechanisms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


OOS_START = pd.Timestamp("2024-01-01")
RNG_SEED = 20260927


@dataclass(frozen=True)
class Phase12Spec:
    mnq_path: Path = REPO_ROOT / "data/processed/parquet/MNQ_c_0_1m_20260321_094501.parquet"
    six_e_path: Path = REPO_ROOT / "data/processed/parquet/6E_v_0_1m_20150101_20260923.parquet"
    zn_path: Path = REPO_ROOT / "data/processed/parquet/ZN_v_0_1m_20130517_20260920.parquet"
    auction_calendar_path: Path = REPO_ROOT / "data/raw/treasury_auction_calendar.csv"
    oos_start: str = "2024-01-01"
    bootstrap_samples: int = 1_000
    random_seed: int = RNG_SEED
    event_window_minutes: int = 240
    fx_window_minutes: int = 180
    treasury_pre_minutes: int = 180
    treasury_post_minutes: int = 180
    output_root: Path = REPO_ROOT / "research"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.relative_to(REPO_ROOT)) if value.is_relative_to(REPO_ROOT) else str(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_ready, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bars(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "timestamp" not in frame.columns:
            raise ValueError(f"No timestamp index/column in {path}")
        frame = frame.set_index("timestamp")
    frame = frame.sort_index()
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    return frame


def audit_bars(path: Path, frame: pd.DataFrame, session_tz: str, start: str, end: str) -> dict[str, Any]:
    index = frame.index.tz_convert(session_tz)
    times = pd.Series(index.time, index=index)
    start_t = pd.Timestamp(start).time()
    end_t = pd.Timestamp(end).time()
    mask = (times >= start_t) & (times < end_t)
    counts = pd.Series(1, index=index[mask.to_numpy()]).groupby(index[mask.to_numpy()].date).sum()
    expected = int((pd.Timedelta(end + ":00") - pd.Timedelta(start + ":00")).total_seconds() / 60)
    transitions: pd.Series | None = None
    if "instrument_id" in frame.columns:
        transitions = frame["instrument_id"].ne(frame["instrument_id"].shift())
        transitions.iloc[0] = False
    transition_returns = (
        frame["close"].pct_change().loc[transitions]
        if transitions is not None
        else pd.Series(dtype=float)
    )
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": _sha256(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "source_timezone": str(frame.index.tz),
        "first_timestamp": frame.index.min(),
        "last_timestamp": frame.index.max(),
        "resolution": "1 minute bar labels; absent zero-trade minutes are not filled",
        "bar_timestamp_semantics": "bar start",
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "chronological": bool(frame.index.is_monotonic_increasing),
        "missing_ohlcv_values": int(frame[["open", "high", "low", "close", "volume"]].isna().sum().sum()),
        "nonpositive_prices": int((frame[["open", "high", "low", "close"]] <= 0).sum().sum()),
        "negative_volume_rows": int((frame["volume"] < 0).sum()),
        "zero_volume_rows": int((frame["volume"] == 0).sum()),
        "continuous_construction": "calendar front month (c.0)" if "_c_0_" in path.name else "volume-ranked front month (v.0)",
        "roll_handling": "Databento continuous mapping; no back-adjustment performed by this campaign",
        "instrument_transitions": int(transitions.sum()) if transitions is not None else None,
        "instrument_transition_audit_status": "available" if transitions is not None else "instrument_id_not_retained_in_processed_file",
        "largest_abs_close_return_at_transition": float(transition_returns.abs().max()) if len(transition_returns) else None,
        "qa_session_timezone": session_tz,
        "qa_session_start_inclusive": start,
        "qa_session_end_exclusive": end,
        "expected_bars_per_full_session": expected,
        "observed_sessions": int(len(counts)),
        "median_bars_per_session": float(counts.median()) if len(counts) else 0.0,
        "sessions_below_80pct": int(counts.lt(0.8 * expected).sum()),
    }


def _event_matrix(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    event_col: str,
    start_offset: int,
    end_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = frame["close"].astype(float)
    close.index = close.index.tz_convert("UTC")
    offsets = np.arange(start_offset, end_offset + 1, dtype=int)
    paths: list[np.ndarray] = []
    details: list[dict[str, Any]] = []
    for event_idx, row in events.reset_index(drop=True).iterrows():
        event_ts = pd.Timestamp(row[event_col])
        event_ts = event_ts.tz_localize("UTC") if event_ts.tzinfo is None else event_ts.tz_convert("UTC")
        wanted = pd.DatetimeIndex(event_ts + pd.to_timedelta(offsets, unit="min"))
        prices = close.reindex(wanted).to_numpy(dtype=float)
        base = prices[0]
        returns = np.log(prices / base) if np.isfinite(base) and base > 0 else np.full(len(offsets), np.nan)
        paths.append(returns * 10_000.0)
        detail = row.to_dict()
        detail.update(
            {
                "event_id": event_idx,
                "event_ts_utc": event_ts,
                "available_path_bars": int(np.isfinite(prices).sum()),
                "expected_path_bars": len(offsets),
                "path_coverage": float(np.isfinite(prices).mean()),
            }
        )
        details.append(detail)
    matrix = pd.DataFrame(paths, columns=offsets)
    return matrix, pd.DataFrame(details)


def _bootstrap_path_summary(matrix: pd.DataFrame, samples: int, seed: int) -> pd.DataFrame:
    values = matrix.to_numpy(dtype=float)
    if values.size == 0:
        return pd.DataFrame(columns=["event_minute", "mean_bps", "median_bps", "ci_low_bps", "ci_high_bps", "n"])
    rng = np.random.default_rng(seed)
    boot = np.empty((samples, values.shape[1]), dtype=float)
    for idx in range(samples):
        chosen = rng.integers(0, len(values), size=len(values))
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            boot[idx] = np.nanmean(values[chosen], axis=0)
    return pd.DataFrame(
        {
            "event_minute": matrix.columns.astype(int),
            "mean_bps": np.nanmean(values, axis=0),
            "median_bps": np.nanmedian(values, axis=0),
            "ci_low_bps": np.nanpercentile(boot, 2.5, axis=0),
            "ci_high_bps": np.nanpercentile(boot, 97.5, axis=0),
            "n": np.isfinite(values).sum(axis=0),
        }
    )


def _endpoint_return(matrix: pd.DataFrame, start: int, end: int) -> pd.Series:
    return matrix[end] - matrix[start]


def _plot_path(summary: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = summary["event_minute"].to_numpy(dtype=float)
    ax.plot(x, summary["mean_bps"], label="Mean", color="#145DA0")
    ax.plot(x, summary["median_bps"], label="Median", color="#F28E2B", alpha=0.85)
    ax.fill_between(x, summary["ci_low_bps"], summary["ci_high_bps"], color="#145DA0", alpha=0.18, label="95% bootstrap CI")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set(title=title, xlabel="Minutes from event", ylabel="Cumulative log return (bps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _sample_stats(values: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"n": 0, "mean_bps": None, "median_bps": None, "t_stat": None, "p_value": None}
    result = stats.ttest_1samp(clean, 0.0)
    return {
        "n": len(clean),
        "mean_bps": float(clean.mean()),
        "median_bps": float(clean.median()),
        "t_stat": float(result.statistic) if len(clean) > 1 else None,
        "p_value": float(result.pvalue) if len(clean) > 1 else None,
    }


def _bootstrap_mean_interval(values: pd.Series, samples: int, seed: int) -> tuple[float | None, float | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for idx in range(samples):
        means[idx] = rng.choice(clean, size=len(clean), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _treasury_control_events(calendar: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Match each auction to same-weekday controls one week before and after."""
    all_auction_dates = set(pd.to_datetime(calendar["auction_date"]).dt.date)
    controls: list[dict[str, Any]] = []
    for event_id, row in events.reset_index(drop=True).iterrows():
        auction_date = pd.Timestamp(row["auction_date"])
        deadline = pd.Timestamp(row["bid_deadline_ts_utc"]).tz_convert("America/New_York")
        for direction, delta in (("week_before", -7), ("week_after", 7)):
            candidate = auction_date + pd.Timedelta(days=delta)
            step = -1 if delta < 0 else 1
            while candidate.date() in all_auction_dates or candidate.weekday() >= 5:
                candidate += pd.Timedelta(days=step)
            local_ts = pd.Timestamp.combine(candidate.date(), deadline.time()).tz_localize("America/New_York")
            controls.append(
                {
                    "matched_event_id": event_id,
                    "auction_date": candidate,
                    "tenor": row["tenor"],
                    "instrument": "ZN",
                    "reference": direction,
                    "bid_deadline_ts_utc": local_ts.tz_convert("UTC"),
                }
            )
    return pd.DataFrame(controls)


def run_treasury_study(spec: Phase12Spec, zn: pd.DataFrame) -> dict[str, Any]:
    out = spec.output_root / "treasury_auction_pressure"
    out.mkdir(parents=True, exist_ok=True)
    calendar = pd.read_csv(spec.auction_calendar_path)
    calendar["auction_date"] = pd.to_datetime(calendar["auction_date"])
    calendar["bid_deadline_ts_utc"] = pd.to_datetime(calendar["bid_deadline_ts_utc"], utc=True)
    # ZF is absent. Phase 2 can only test the economically natural ZN mappings.
    events = calendar.loc[calendar["tenor"].isin(["7-Year", "10-Year"])].copy()
    events["instrument"] = "ZN"
    events["reference"] = "competitive_bid_deadline"
    matrix, details = _event_matrix(
        zn, events, "bid_deadline_ts_utc", -spec.event_window_minutes, spec.event_window_minutes
    )
    details["pre_return_bps"] = _endpoint_return(matrix, -180, -1).to_numpy()
    details["immediate_return_bps"] = _endpoint_return(matrix, -1, 5).to_numpy()
    details["post_return_bps"] = _endpoint_return(matrix, 5, 180).to_numpy()
    details["phase"] = np.where(details["auction_date"] >= OOS_START, "oos", "is")
    details.to_csv(out / "event_level.csv", index=False)

    controls = _treasury_control_events(calendar, events)
    control_matrix, control_details = _event_matrix(
        zn, controls, "bid_deadline_ts_utc", -spec.event_window_minutes, spec.event_window_minutes
    )
    control_details["pre_return_bps"] = _endpoint_return(control_matrix, -180, -1).to_numpy()
    control_details["immediate_return_bps"] = _endpoint_return(control_matrix, -1, 5).to_numpy()
    control_details["post_return_bps"] = _endpoint_return(control_matrix, 5, 180).to_numpy()
    control_details.to_csv(out / "matched_control_event_level.csv", index=False)

    summary_frames: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for tenor in ("7-Year", "10-Year"):
        mask = events["tenor"].eq(tenor).to_numpy()
        tenor_matrix = matrix.loc[mask].reset_index(drop=True)
        summary = _bootstrap_path_summary(tenor_matrix, spec.bootstrap_samples, spec.random_seed)
        summary.insert(0, "tenor", tenor)
        summary.insert(1, "sample", "auction")
        summary_frames.append(summary)
        control_mask = controls["tenor"].eq(tenor).to_numpy()
        tenor_control_matrix = control_matrix.loc[control_mask].reset_index(drop=True)
        control_summary = _bootstrap_path_summary(
            tenor_control_matrix, spec.bootstrap_samples, spec.random_seed + 10
        )
        control_summary.insert(0, "tenor", tenor)
        control_summary.insert(1, "sample", "matched_control")
        summary_frames.append(control_summary)
        _plot_path(summary, out / f"zn_{tenor.lower().replace('-', '_')}_event_path.png", f"ZN around {tenor} Treasury auctions")
        tenor_details = details.loc[details["tenor"].eq(tenor)]
        tenor_controls = control_details.loc[control_details["tenor"].eq(tenor)]
        metrics.append(
            {
                "tenor": tenor,
                "instrument": "ZN",
                "events": len(tenor_details),
                "median_path_coverage": float(tenor_details["path_coverage"].median()),
                "pre": _sample_stats(tenor_details["pre_return_bps"]),
                "immediate": _sample_stats(tenor_details["immediate_return_bps"]),
                "post": _sample_stats(tenor_details["post_return_bps"]),
                "matched_controls": {
                    "events": len(tenor_controls),
                    "pre": _sample_stats(tenor_controls["pre_return_bps"]),
                    "immediate": _sample_stats(tenor_controls["immediate_return_bps"]),
                    "post": _sample_stats(tenor_controls["post_return_bps"]),
                },
            }
        )
    pd.concat(summary_frames, ignore_index=True).to_csv(out / "event_path_summary.csv", index=False)
    (
        details.assign(year=pd.to_datetime(details["auction_date"]).dt.year)
        .groupby(["tenor", "year"], as_index=False)
        .agg(events=("event_id", "size"), mean_pre_bps=("pre_return_bps", "mean"), mean_post_bps=("post_return_bps", "mean"), median_coverage=("path_coverage", "median"))
        .to_csv(out / "yearly_metrics.csv", index=False)
    )
    _write_json(out / "aggregate_metrics.json", {"mappings": metrics})
    return {"events": len(details), "metrics": metrics, "output": out}


def _mnq_daily_events(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame[["open", "close", "volume"]].copy()
    local.index = local.index.tz_convert("America/New_York")
    local["session_date"] = pd.to_datetime(local.index.date)
    local["minute"] = local.index.strftime("%H:%M")
    records: list[dict[str, Any]] = []
    prior_close: tuple[pd.Timestamp, float] | None = None
    for session_date, day in local.groupby("session_date", sort=True):
        by_minute = day.loc[~day["minute"].duplicated(keep="last")].set_index("minute")
        rth_bars = int(by_minute.loc[(by_minute.index >= "09:30") & (by_minute.index <= "15:59")].shape[0])
        required = {"15:29", "15:30", "15:59"}
        if required.issubset(by_minute.index) and prior_close is not None and rth_bars >= 380:
            signal_price = float(by_minute.at["15:29", "close"])
            entry_price = float(by_minute.at["15:30", "open"])
            exit_price = float(by_minute.at["15:59", "close"])
            records.append(
                {
                    "session_date": session_date,
                    "previous_cash_close_date": prior_close[0],
                    "previous_cash_close": prior_close[1],
                    "signal_price_1529_close": signal_price,
                    "entry_price_1530_open": entry_price,
                    "exit_price_1559_close": exit_price,
                    "rod_return": math.log(signal_price / prior_close[1]),
                    "last30_return": math.log(exit_price / entry_price),
                    "rth_bars": rth_bars,
                }
            )
        if "15:59" in by_minute.index and rth_bars >= 380:
            prior_close = (session_date, float(by_minute.at["15:59", "close"]))
    return pd.DataFrame(records)


def _newey_west_regression(y: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(y) & np.isfinite(x)
    y = y[valid]
    x = x[valid]
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    n, k = design.shape
    lag = int(math.floor(4 * (n / 100) ** (2 / 9))) if n else 0
    meat = np.zeros((k, k))
    for idx in range(n):
        meat += residual[idx] ** 2 * np.outer(design[idx], design[idx])
    for step in range(1, lag + 1):
        weight = 1.0 - step / (lag + 1.0)
        gamma = np.zeros((k, k))
        for idx in range(step, n):
            gamma += residual[idx] * residual[idx - step] * np.outer(design[idx], design[idx - step])
        meat += weight * (gamma + gamma.T)
    inv = np.linalg.inv(design.T @ design)
    covariance_hac = inv @ meat @ inv
    sigma2 = float(residual @ residual / max(n - k, 1))
    covariance_ols = sigma2 * inv
    se_hac = np.sqrt(np.diag(covariance_hac))
    se_ols = np.sqrt(np.diag(covariance_ols))
    return {
        "n": n,
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "t_stat": float(beta[1] / se_ols[1]),
        "hac_lags": lag,
        "hac_t_stat": float(beta[1] / se_hac[1]),
        "r_squared": float(1.0 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2)),
    }


def run_mnq_study(spec: Phase12Spec, mnq: pd.DataFrame) -> dict[str, Any]:
    out = spec.output_root / "mnq_close_hedging_flow"
    out.mkdir(parents=True, exist_ok=True)
    events = _mnq_daily_events(mnq)
    events["phase"] = np.where(events["session_date"] >= OOS_START, "oos", "is")
    events["rod_sign"] = np.sign(events["rod_return"]).astype(int)
    events["rod_decile"] = pd.qcut(events["rod_return"], 10, labels=False, duplicates="drop") + 1
    events.to_csv(out / "event_level.csv", index=False)

    deciles = (
        events.groupby("rod_decile", as_index=False)
        .agg(n=("last30_return", "size"), mean_rod=("rod_return", "mean"), mean_last30=("last30_return", "mean"), median_last30=("last30_return", "median"))
    )
    deciles[["mean_rod", "mean_last30", "median_last30"]] *= 10_000.0
    decile_intervals = [
        _bootstrap_mean_interval(
            events.loc[events["rod_decile"].eq(decile), "last30_return"] * 10_000,
            spec.bootstrap_samples,
            spec.random_seed + int(decile),
        )
        for decile in deciles["rod_decile"]
    ]
    deciles["ci_low_bps"] = [value[0] for value in decile_intervals]
    deciles["ci_high_bps"] = [value[1] for value in decile_intervals]
    deciles.to_csv(out / "conditional_returns_by_decile.csv", index=False)
    signs = (
        events.groupby("rod_sign", as_index=False)
        .agg(n=("last30_return", "size"), mean_rod=("rod_return", "mean"), mean_last30=("last30_return", "mean"), median_last30=("last30_return", "median"))
    )
    signs[["mean_rod", "mean_last30", "median_last30"]] *= 10_000.0
    sign_intervals = [
        _bootstrap_mean_interval(
            events.loc[events["rod_sign"].eq(sign), "last30_return"] * 10_000,
            spec.bootstrap_samples,
            spec.random_seed + 100 + int(sign),
        )
        for sign in signs["rod_sign"]
    ]
    signs["ci_low_bps"] = [value[0] for value in sign_intervals]
    signs["ci_high_bps"] = [value[1] for value in sign_intervals]
    signs.to_csv(out / "conditional_returns_by_sign.csv", index=False)

    full_reg = _newey_west_regression(events["last30_return"].to_numpy(), events["rod_return"].to_numpy())
    is_events = events.loc[events["phase"].eq("is")]
    oos_events = events.loc[events["phase"].eq("oos")]
    is_reg = _newey_west_regression(is_events["last30_return"].to_numpy(), is_events["rod_return"].to_numpy())
    oos_reg = _newey_west_regression(oos_events["last30_return"].to_numpy(), oos_events["rod_return"].to_numpy())
    prediction = is_reg["intercept"] + is_reg["slope"] * oos_events["rod_return"].to_numpy()
    benchmark = float(is_events["last30_return"].mean())
    oos_r2 = 1.0 - float(np.sum((oos_events["last30_return"].to_numpy() - prediction) ** 2)) / float(
        np.sum((oos_events["last30_return"].to_numpy() - benchmark) ** 2)
    )
    monotonicity = float(stats.spearmanr(deciles["rod_decile"], deciles["mean_last30"]).statistic)
    metrics = {
        "events": len(events),
        "is_events": len(is_events),
        "oos_events": len(oos_events),
        "full_regression": full_reg,
        "is_regression": is_reg,
        "oos_regression": oos_reg,
        "oos_predictive_r_squared_from_is_fit": oos_r2,
        "decile_mean_spearman_monotonicity": monotonicity,
        "positive_rod_last30": _sample_stats(events.loc[events["rod_sign"].eq(1), "last30_return"] * 10_000),
        "negative_rod_last30": _sample_stats(events.loc[events["rod_sign"].eq(-1), "last30_return"] * 10_000),
    }
    _write_json(out / "aggregate_metrics.json", metrics)
    (
        events.assign(year=events["session_date"].dt.year)
        .groupby("year", as_index=False)
        .agg(events=("session_date", "size"), mean_rod_bps=("rod_return", lambda x: x.mean() * 10_000), mean_last30_bps=("last30_return", lambda x: x.mean() * 10_000), correlation=("last30_return", lambda x: x.corr(events.loc[x.index, "rod_return"])))
        .to_csv(out / "yearly_metrics.csv", index=False)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(deciles["rod_decile"], deciles["mean_last30"], marker="o")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set(title="MNQ final 30-minute return conditional on ROD decile", xlabel="ROD decile", ylabel="Mean final-30m log return (bps)")
    fig.tight_layout()
    fig.savefig(out / "conditional_last30_by_rod_decile.png", dpi=160)
    plt.close(fig)
    return {"events": len(events), "metrics": metrics, "output": out}


def _london_fix_events(frame: pd.DataFrame) -> pd.DataFrame:
    start = frame.index.min().tz_convert("Europe/London").date()
    end = frame.index.max().tz_convert("Europe/London").date()
    dates = pd.date_range(start, end, freq="B")
    local = pd.DatetimeIndex([pd.Timestamp(date.date()).replace(hour=16) for date in dates]).tz_localize(
        "Europe/London", ambiguous="raise", nonexistent="raise"
    )
    utc = local.tz_convert("UTC")
    new_york = utc.tz_convert("America/New_York")
    return pd.DataFrame(
        {
            "fix_date_local": dates.date,
            "benchmark": "WM/Reuters London 4pm",
            "fix_timezone": "Europe/London",
            "fix_ts_utc": utc,
            "london_utc_offset_hours": [stamp.utcoffset().total_seconds() / 3600 for stamp in local],
            "new_york_utc_offset_hours": [stamp.utcoffset().total_seconds() / 3600 for stamp in new_york],
            "london_fix_time_new_york": new_york.strftime("%H:%M"),
        }
    )


def run_fx_study(spec: Phase12Spec, six_e: pd.DataFrame) -> dict[str, Any]:
    out = spec.output_root / "six_e_fx_fix_reversal"
    out.mkdir(parents=True, exist_ok=True)
    events = _london_fix_events(six_e)
    matrix, details = _event_matrix(
        six_e, events, "fix_ts_utc", -spec.fx_window_minutes, spec.fx_window_minutes
    )
    details["pre_return_bps"] = _endpoint_return(matrix, -180, -1).to_numpy()
    details["immediate_return_bps"] = _endpoint_return(matrix, -1, 5).to_numpy()
    details["post_return_bps"] = _endpoint_return(matrix, 5, 180).to_numpy()
    details["phase"] = np.where(pd.to_datetime(details["fix_date_local"]) >= OOS_START, "oos", "is")
    details = details.loc[details["path_coverage"] >= 0.90].reset_index(drop=True)
    matrix = matrix.loc[details["event_id"].to_numpy()].reset_index(drop=True)
    details.to_csv(out / "event_level.csv", index=False)
    summary = _bootstrap_path_summary(matrix, spec.bootstrap_samples, spec.random_seed + 1)
    summary.to_csv(out / "event_path_summary.csv", index=False)
    _plot_path(summary, out / "six_e_london_fix_event_path.png", "6E around the London 4pm fix")
    yearly = (
        details.assign(year=pd.to_datetime(details["fix_date_local"]).dt.year)
        .groupby("year", as_index=False)
        .agg(events=("event_id", "size"), mean_pre_bps=("pre_return_bps", "mean"), mean_post_bps=("post_return_bps", "mean"), median_coverage=("path_coverage", "median"))
    )
    yearly.to_csv(out / "yearly_metrics.csv", index=False)
    metrics = {
        "events": len(details),
        "is_events": int(details["phase"].eq("is").sum()),
        "oos_events": int(details["phase"].eq("oos").sum()),
        "pre": _sample_stats(details["pre_return_bps"]),
        "immediate": _sample_stats(details["immediate_return_bps"]),
        "post": _sample_stats(details["post_return_bps"]),
        "expected_6e_sign": "negative before fix, positive after fix",
        "sign_mapping": "paper long USD = short EURUSD = short 6E; paper post-fix short USD = long EURUSD = long 6E",
    }
    _write_json(out / "aggregate_metrics.json", metrics)
    return {"events": len(details), "metrics": metrics, "output": out}


def _write_specs(spec: Phase12Spec, audits: dict[str, Any]) -> None:
    common = spec.output_root / "event_flow_mechanisms_comparison"
    common.mkdir(parents=True, exist_ok=True)
    payload = {key: _json_ready(value) for key, value in asdict(spec).items()}
    payload.update(
        {
            "freeze_timestamp_policy": "Frozen before inspecting phase-2 performance",
            "phase_scope": "Phase 1 data audit plus Phase 2 canonical event definitions/event studies only",
            "no_strategy_backtest_in_this_run": True,
        }
    )
    _write_json(common / "frozen_phase12_config.json", payload)
    _write_json(common / "data_audit.json", audits)


def run_phase12(spec: Phase12Spec) -> dict[str, Any]:
    mnq = load_bars(spec.mnq_path)
    six_e = load_bars(spec.six_e_path)
    zn = load_bars(spec.zn_path)
    audits = {
        "MNQ": audit_bars(spec.mnq_path, mnq, "America/New_York", "09:30", "16:00"),
        "6E": audit_bars(spec.six_e_path, six_e, "UTC", "00:00", "23:00"),
        "ZN": audit_bars(spec.zn_path, zn, "America/New_York", "08:20", "15:00"),
        "ZF": {"available": False, "consequence": "Primary 5Y->ZF and 7Y->ZF mappings cannot be tested."},
    }
    _write_specs(spec, audits)
    results = {
        "treasury": run_treasury_study(spec, zn),
        "mnq": run_mnq_study(spec, mnq),
        "fx": run_fx_study(spec, six_e),
    }
    _write_json(
        spec.output_root / "event_flow_mechanisms_comparison" / "phase12_run_metadata.json",
        {
            "config": {key: _json_ready(value) for key, value in asdict(spec).items()},
            "event_counts": {key: value["events"] for key, value in results.items()},
            "status": "phase_2_complete_phase_3_report_pending",
        },
    )
    return results


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "research")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_phase12(Phase12Spec(bootstrap_samples=args.bootstrap_samples, output_root=args.output_root))


if __name__ == "__main__":
    main()
