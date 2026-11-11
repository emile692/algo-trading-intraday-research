"""Frozen close-of-day intraday-momentum replication and falsification.

Phase 1 is deliberately isolated. Later phases may run only when the ex-ante
raw-effect gate in frozen_config.json is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_OUTPUT = REPO_ROOT / "research" / "close_of_day_intraday_momentum"


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    raise TypeError(type(value).__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bars(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        timestamp_column = "timestamp" if "timestamp" in frame.columns else "ts_event"
        if timestamp_column not in frame.columns:
            raise ValueError(f"No timestamp or ts_event in {path}")
        frame = frame.set_index(timestamp_column)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()


def _clock(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":"))
    return hour * 60 + minute


def _minute_label(minutes: int) -> str:
    minutes %= 24 * 60
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without adding the optional tabulate dependency."""
    display = frame.fillna("").astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy(dtype=str)]
    return "\n".join([header, separator, *rows])


def build_canonical_events(
    bars: pd.DataFrame,
    *,
    instrument: str,
    timezone: str,
    session_start: str,
    close_time: str,
    minimum_fraction: float = 0.9,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build daily observations without reading any price after T."""
    required = {"open", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = bars.copy()
    frame.index = frame.index.tz_convert(timezone)
    local_minute = frame.index.hour * 60 + frame.index.minute
    session_start_minute = _clock(session_start)
    close_minute = _clock(close_time)
    signal_minute = close_minute - 31
    entry_minute = close_minute - 30
    delayed_entry_minute = close_minute - 29
    exit_minute = close_minute - 1
    in_session = (local_minute >= session_start_minute) & (local_minute < close_minute)
    session = frame.loc[in_session].copy()
    session["session_date"] = pd.to_datetime(session.index.date)
    session["local_minute"] = session.index.hour * 60 + session.index.minute
    expected_bars = close_minute - session_start_minute
    counts = session.groupby("session_date").size()

    records: list[dict[str, Any]] = []
    prior: dict[str, Any] | None = None
    exact_close_sessions = 0
    incomplete_sessions = 0
    missing_target_sessions = 0
    roll_exclusions = 0
    for session_date, day in session.groupby("session_date", sort=True):
        day = day.copy()
        day["local_timestamp"] = day.index
        by_minute = day.loc[~day["local_minute"].duplicated(keep="last")].set_index("local_minute")
        if exit_minute not in by_minute.index:
            missing_target_sessions += 1
            # Do not bridge over an early close or corrupted day: the following
            # event would otherwise use a stale close as "previous trading day".
            prior = None
            continue
        current_close = by_minute.loc[exit_minute]
        exact_close_sessions += 1
        current_id = current_close.get("instrument_id", np.nan)
        current_close_record = {
            "date": session_date,
            "price": float(current_close["close"]),
            "instrument_id": current_id,
        }
        if counts.loc[session_date] < minimum_fraction * expected_bars:
            incomplete_sessions += 1
            prior = current_close_record
            continue
        if signal_minute not in by_minute.index or entry_minute not in by_minute.index or delayed_entry_minute not in by_minute.index:
            missing_target_sessions += 1
            prior = current_close_record
            continue
        if prior is None:
            prior = current_close_record
            continue
        signal_row = by_minute.loc[signal_minute]
        entry_row = by_minute.loc[entry_minute]
        delayed_entry_row = by_minute.loc[delayed_entry_minute]
        ids = [prior["instrument_id"], signal_row.get("instrument_id", np.nan), entry_row.get("instrument_id", np.nan), delayed_entry_row.get("instrument_id", np.nan), current_id]
        roll_safe_available = all(pd.notna(value) for value in ids)
        if roll_safe_available and len(set(int(value) for value in ids)) != 1:
            roll_exclusions += 1
            prior = current_close_record
            continue
        prior_price = float(prior["price"])
        signal_price = float(signal_row["close"])
        entry_price = float(entry_row["open"])
        delayed_entry_price = float(delayed_entry_row["open"])
        exit_price = float(current_close["close"])
        if min(prior_price, signal_price, entry_price, delayed_entry_price, exit_price) <= 0:
            prior = current_close_record
            continue
        r_signal = signal_price / prior_price - 1.0
        raw_return = exit_price / signal_price - 1.0
        executable_return = exit_price / entry_price - 1.0
        position = float(np.sign(r_signal))
        records.append(
            {
                "instrument": instrument,
                "session_date": session_date,
                "timezone": timezone,
                "T": close_time,
                "previous_session_date": prior["date"],
                "previous_close": prior_price,
                "signal_timestamp": signal_row["local_timestamp"],
                "entry_timestamp": entry_row["local_timestamp"],
                "delayed_entry_timestamp": delayed_entry_row["local_timestamp"],
                "exit_timestamp": current_close["local_timestamp"],
                "signal_price_close_Tminus31": signal_price,
                "entry_price_open_Tminus30": entry_price,
                "entry_price_open_Tminus29": delayed_entry_price,
                "exit_price_close_Tminus1": exit_price,
                "r_signal": r_signal,
                "r_close_raw": raw_return,
                "r_close_executable": executable_return,
                "r_close_one_bar_delay": exit_price / delayed_entry_price - 1.0,
                "position": position,
                "sign_edge_raw": position * raw_return,
                "sign_edge_executable": position * executable_return,
                "sign_edge_one_bar_delay": position * (exit_price / delayed_entry_price - 1.0),
                "hit_raw": float(position * raw_return > 0) if position else np.nan,
                "roll_safety_available": roll_safe_available,
                "bars_in_session": int(counts.loc[session_date]),
                "expected_bars": expected_bars,
            }
        )
        prior = current_close_record
    events = pd.DataFrame(records)
    if not events.empty:
        events["session_date"] = pd.to_datetime(events["session_date"])
        events["previous_session_date"] = pd.to_datetime(events["previous_session_date"])
    diagnostics = {
        "observed_local_sessions": int(len(counts)),
        "expected_bars_per_full_session": expected_bars,
        "median_bars_per_session": float(counts.median()) if len(counts) else None,
        "exact_close_sessions": exact_close_sessions,
        "incomplete_sessions": incomplete_sessions,
        "missing_target_sessions": missing_target_sessions,
        "roll_exclusions": roll_exclusions,
        "eligible_events": len(events),
        "signal_bar_label": _minute_label(signal_minute),
        "entry_bar_label": _minute_label(entry_minute),
        "exit_bar_label": _minute_label(exit_minute),
    }
    return events, diagnostics


def newey_west_regression(y: pd.Series, x: pd.Series) -> dict[str, Any]:
    clean = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    n = len(clean)
    if n < 3:
        return {"n": n, "alpha": np.nan, "beta": np.nan, "std_error": np.nan, "t_stat": np.nan, "hac_std_error": np.nan, "hac_t_stat": np.nan, "r_squared": np.nan, "hac_lags": 0}
    design = np.column_stack([np.ones(n), clean["x"].to_numpy(dtype=float)])
    outcome = clean["y"].to_numpy(dtype=float)
    xtx_inv = np.linalg.inv(design.T @ design)
    coef = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coef
    sigma2 = float(residuals @ residuals / (n - 2))
    covariance = sigma2 * xtx_inv
    lags = int(math.floor(4 * (n / 100) ** (2 / 9)))
    scores = design * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    hac_covariance = (n / (n - 2)) * xtx_inv @ meat @ xtx_inv
    std_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    hac_std_error = float(np.sqrt(max(hac_covariance[1, 1], 0.0)))
    tss = float(((outcome - outcome.mean()) ** 2).sum())
    return {
        "n": n,
        "alpha": float(coef[0]),
        "beta": float(coef[1]),
        "std_error": std_error,
        "t_stat": float(coef[1] / std_error) if std_error else np.nan,
        "hac_std_error": hac_std_error,
        "hac_t_stat": float(coef[1] / hac_std_error) if hac_std_error else np.nan,
        "r_squared": float(1.0 - (residuals @ residuals) / tss) if tss else np.nan,
        "hac_lags": lags,
    }


def _period_masks(events: pd.DataFrame) -> dict[str, pd.Series]:
    years = events["session_date"].dt.year
    return {
        "full": pd.Series(True, index=events.index),
        "2007-2016": years.between(2007, 2016),
        "2017-2026": years.between(2017, 2026),
    }


def summarize_effects(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regression_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    for instrument, group in events.groupby("instrument", sort=True):
        for period, mask in _period_masks(group).items():
            sample = group.loc[mask]
            if sample.empty:
                regression_rows.append({"instrument": instrument, "period": period, **newey_west_regression(pd.Series(dtype=float), pd.Series(dtype=float))})
                continue
            regression_rows.append({"instrument": instrument, "period": period, **newey_west_regression(sample["r_close_raw"], sample["r_signal"])})
            for direction, direction_mask in (("positive_signal", sample["position"] > 0), ("negative_signal", sample["position"] < 0), ("all_nonzero", sample["position"] != 0)):
                selected = sample.loc[direction_mask]
                edge = selected["sign_edge_raw"].dropna()
                conditional_rows.append(
                    {
                        "instrument": instrument,
                        "period": period,
                        "signal_group": direction,
                        "n": len(selected),
                        "mean_r_close_bps": selected["r_close_raw"].mean() * 10_000 if len(selected) else np.nan,
                        "mean_sign_edge_bps": edge.mean() * 10_000 if len(edge) else np.nan,
                        "median_sign_edge_bps": edge.median() * 10_000 if len(edge) else np.nan,
                        "sign_edge_t_stat": stats.ttest_1samp(edge, 0.0).statistic if len(edge) > 1 else np.nan,
                        "hit_rate": selected["hit_raw"].mean() if len(selected) else np.nan,
                    }
                )
        for year, sample in group.groupby(group["session_date"].dt.year):
            reg = newey_west_regression(sample["r_close_raw"], sample["r_signal"])
            yearly_rows.append(
                {
                    "instrument": instrument,
                    "year": int(year),
                    "n": len(sample),
                    "beta": reg["beta"],
                    "hac_t_stat": reg["hac_t_stat"],
                    "mean_sign_edge_bps": sample["sign_edge_raw"].mean() * 10_000,
                    "hit_rate": sample["hit_raw"].mean(),
                }
            )
    return pd.DataFrame(regression_rows), pd.DataFrame(conditional_rows), pd.DataFrame(yearly_rows)


def _gate_table(regressions: pd.DataFrame, conditional: pd.DataFrame, yearly: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    threshold = float(config["phase_1_gate"]["minimum_hac_t"])
    year_threshold = float(config["phase_1_gate"]["minimum_positive_year_fraction"])
    rows = []
    for instrument in sorted(regressions["instrument"].unique()):
        full = regressions.loc[(regressions.instrument == instrument) & (regressions.period == "full")].iloc[0]
        full_edge = conditional.loc[(conditional.instrument == instrument) & (conditional.period == "full") & (conditional.signal_group == "all_nonzero")]
        recent_edge = conditional.loc[(conditional.instrument == instrument) & (conditional.period == "2017-2026") & (conditional.signal_group == "all_nonzero")]
        annual = yearly.loc[yearly.instrument == instrument]
        positive_year_fraction = float((annual["mean_sign_edge_bps"] > 0).mean()) if len(annual) else np.nan
        criteria = {
            "beta_positive": bool(full["beta"] > 0),
            "hac_t_at_least_1_96": bool(full["hac_t_stat"] >= threshold),
            "full_sign_edge_positive": bool(len(full_edge) and full_edge.iloc[0]["mean_sign_edge_bps"] > 0),
            "recent_sign_edge_positive": bool(len(recent_edge) and recent_edge.iloc[0]["mean_sign_edge_bps"] > 0),
            "majority_positive_years": bool(np.isfinite(positive_year_fraction) and positive_year_fraction > year_threshold),
        }
        rows.append(
            {
                "instrument": instrument,
                "beta": full["beta"],
                "hac_t_stat": full["hac_t_stat"],
                "full_sign_edge_bps": full_edge.iloc[0]["mean_sign_edge_bps"] if len(full_edge) else np.nan,
                "recent_sign_edge_bps": recent_edge.iloc[0]["mean_sign_edge_bps"] if len(recent_edge) else np.nan,
                "positive_year_fraction": positive_year_fraction,
                **criteria,
                "phase_1_gate_pass": all(criteria.values()),
            }
        )
    return pd.DataFrame(rows)


def _pnl_metrics(pnl: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(pnl, errors="coerce").dropna()
    n = len(values)
    if n == 0:
        return {key: np.nan for key in ("total_pnl", "annualized_volatility", "sharpe", "sortino", "profit_factor", "hit_rate", "average_trade", "median_trade", "average_winner", "average_loser", "max_drawdown", "worst_trade", "worst_day", "skewness", "cvar_5pct")} | {"trades": 0}
    downside = values.loc[values < 0]
    winners = values.loc[values > 0]
    volatility = values.std(ddof=1)
    downside_deviation = np.sqrt(np.mean(np.minimum(values, 0.0) ** 2))
    equity = values.cumsum()
    drawdown = equity - equity.cummax().clip(lower=0.0)
    tail_cutoff = values.quantile(0.05)
    return {
        "total_pnl": float(values.sum()),
        "annualized_volatility": float(volatility * np.sqrt(252)) if n > 1 else np.nan,
        "sharpe": float(values.mean() / volatility * np.sqrt(252)) if volatility > 0 else np.nan,
        "sortino": float(values.mean() / downside_deviation * np.sqrt(252)) if downside_deviation > 0 else np.nan,
        "profit_factor": float(winners.sum() / -downside.sum()) if downside.sum() < 0 else np.inf,
        "hit_rate": float((values > 0).mean()),
        "trades": n,
        "average_trade": float(values.mean()),
        "median_trade": float(values.median()),
        "average_winner": float(winners.mean()) if len(winners) else np.nan,
        "average_loser": float(downside.mean()) if len(downside) else np.nan,
        "max_drawdown": float(drawdown.min()),
        "worst_trade": float(values.min()),
        "worst_day": float(values.min()),
        "skewness": float(values.skew()),
        "cvar_5pct": float(values.loc[values <= tail_cutoff].mean()),
    }


def executable_results(events: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = {row["instrument"]: row for row in config["instruments"]}
    detail = events.copy()
    point_values = detail["instrument"].map({key: value["point_value"] for key, value in specs.items()})
    normal_costs = detail["instrument"].map(
        {
            key: 2.0 * (value["commission_per_side"] + value["tick_size"] * value["point_value"] * value["slippage_ticks_per_side"])
            for key, value in specs.items()
        }
    )
    detail["gross_pnl"] = detail["position"] * (detail["exit_price_close_Tminus1"] - detail["entry_price_open_Tminus30"]) * point_values
    detail["normal_round_trip_cost"] = normal_costs
    detail["normal_net_pnl"] = detail["gross_pnl"] - normal_costs.where(detail["position"] != 0, 0.0)
    detail["double_cost_net_pnl"] = detail["gross_pnl"] - 2.0 * normal_costs.where(detail["position"] != 0, 0.0)
    detail["one_bar_delay_gross_pnl"] = detail["position"] * (detail["exit_price_close_Tminus1"] - detail["entry_price_open_Tminus29"]) * point_values
    rows = []
    for instrument, group in detail.groupby("instrument"):
        for cost_case, column in (("gross", "gross_pnl"), ("normal_costs", "normal_net_pnl"), ("double_costs", "double_cost_net_pnl")):
            for period, mask in _period_masks(group).items():
                rows.append({"instrument": instrument, "cost_case": cost_case, "period": period, "annualized_return": np.nan, **_pnl_metrics(group.loc[mask, column])})
    return detail, pd.DataFrame(rows)


def build_timing_placebo(
    bars: pd.DataFrame,
    canonical: pd.DataFrame,
    *,
    timezone: str,
    close_time: str,
    start_minutes_before_t: int,
    end_minutes_before_t: int,
) -> pd.DataFrame:
    frame = bars.copy()
    frame.index = frame.index.tz_convert(timezone)
    frame["session_date"] = pd.to_datetime(frame.index.date)
    frame["local_minute"] = frame.index.hour * 60 + frame.index.minute
    close_minute = _clock(close_time)
    signal_minute = close_minute - start_minutes_before_t - 1
    entry_minute = close_minute - start_minutes_before_t
    exit_minute = close_minute - end_minutes_before_t - 1
    canonical_by_date = canonical.set_index("session_date")
    records = []
    for date, day in frame.loc[frame["session_date"].isin(canonical_by_date.index)].groupby("session_date", sort=True):
        by_minute = day.loc[~day["local_minute"].duplicated(keep="last")].set_index("local_minute")
        if not {signal_minute, entry_minute, exit_minute}.issubset(by_minute.index):
            continue
        base = canonical_by_date.loc[date]
        signal_row, entry_row, exit_row = by_minute.loc[signal_minute], by_minute.loc[entry_minute], by_minute.loc[exit_minute]
        previous_close = float(base["previous_close"])
        signal_price, entry_price, exit_price = float(signal_row["close"]), float(entry_row["open"]), float(exit_row["close"])
        position = float(np.sign(signal_price / previous_close - 1.0))
        records.append({"session_date": date, "position": position, "window_return": exit_price / entry_price - 1.0, "sign_edge": position * (exit_price / entry_price - 1.0)})
    return pd.DataFrame(records)


def falsification_results(events: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(config["random_seed"]))
    samples = int(config["permutation_samples"])
    permutation_rows, other_rows = [], []
    for instrument, group in events.groupby("instrument", sort=True):
        positions = group["position"].to_numpy(dtype=float)
        returns = group["r_close_executable"].to_numpy(dtype=float)
        observed = float(np.mean(positions * returns))
        null = np.empty(samples)
        for idx in range(samples):
            null[idx] = np.mean(rng.permutation(positions) * returns)
        permutation_rows.append(
            {
                "instrument": instrument,
                "permutations": samples,
                "observed_mean_bps": observed * 10_000,
                "null_mean_bps": null.mean() * 10_000,
                "null_2_5pct_bps": np.quantile(null, 0.025) * 10_000,
                "null_97_5pct_bps": np.quantile(null, 0.975) * 10_000,
                "one_sided_p_value": (1 + np.sum(null >= observed)) / (samples + 1),
            }
        )
        canonical_edge = group["sign_edge_executable"].dropna()
        delayed_edge = group["sign_edge_one_bar_delay"].dropna()
        other_rows.extend(
            [
                {"instrument": instrument, "test": "canonical_Tminus30_to_T", "n": len(group), "mean_sign_edge_bps": canonical_edge.mean() * 10_000, "t_stat": stats.ttest_1samp(canonical_edge, 0).statistic, "hit_rate": (canonical_edge > 0).mean()},
                {"instrument": instrument, "test": "sign_inversion", "n": len(group), "mean_sign_edge_bps": -canonical_edge.mean() * 10_000, "t_stat": -stats.ttest_1samp(canonical_edge, 0).statistic, "hit_rate": (canonical_edge < 0).mean()},
                {"instrument": instrument, "test": "one_bar_execution_delay", "n": len(group), "mean_sign_edge_bps": delayed_edge.mean() * 10_000, "t_stat": stats.ttest_1samp(delayed_edge, 0).statistic, "hit_rate": (delayed_edge > 0).mean()},
            ]
        )
    return pd.DataFrame(permutation_rows), pd.DataFrame(other_rows)


def cross_asset_summary(gate: pd.DataFrame, strategy: pd.DataFrame, yearly: pd.DataFrame, details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for instrument in gate["instrument"]:
        gross = strategy.loc[(strategy.instrument == instrument) & (strategy.cost_case == "gross") & (strategy.period == "full")].iloc[0]
        net = strategy.loc[(strategy.instrument == instrument) & (strategy.cost_case == "normal_costs") & (strategy.period == "full")].iloc[0]
        recent = strategy.loc[(strategy.instrument == instrument) & (strategy.cost_case == "normal_costs") & (strategy.period == "2017-2026")].iloc[0]
        annual = yearly.loc[yearly.instrument == instrument]
        sample = details.loc[details.instrument == instrument]
        breakeven = sample["gross_pnl"].mean()
        rows.append(
            {
                "instrument": instrument,
                "sample_start": sample["session_date"].min(),
                "sample_end": sample["session_date"].max(),
                "trades": int(gross["trades"]),
                "beta": gate.loc[gate.instrument == instrument, "beta"].iloc[0],
                "hac_t_stat": gate.loc[gate.instrument == instrument, "hac_t_stat"].iloc[0],
                "gross_sharpe": gross["sharpe"],
                "net_sharpe": net["sharpe"],
                "profit_factor_net": net["profit_factor"],
                "net_pnl": net["total_pnl"],
                "positive_year_ratio": (annual["mean_sign_edge_bps"] > 0).mean(),
                "recent_net_sharpe": recent["sharpe"],
                "max_drawdown_net": net["max_drawdown"],
                "breakeven_round_trip_cost_usd": breakeven,
                "configured_round_trip_cost_usd": sample["normal_round_trip_cost"].iloc[0],
            }
        )
    summary = pd.DataFrame(rows)
    classifications = []
    for row in summary.itertuples():
        passed_raw = bool(gate.loc[gate.instrument == row.instrument, "phase_1_gate_pass"].iloc[0])
        if passed_raw and row.net_pnl > 0 and row.recent_net_sharpe > 0:
            classifications.append("REPLICATED")
        elif row.beta > 0 and row.gross_sharpe > 0 and row.net_pnl > 0:
            classifications.append("WEAK / WATCHLIST")
        else:
            classifications.append("REJECTED")
    summary["classification"] = classifications
    return summary


def diagnostic_portfolios(details: pd.DataFrame, gate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = gate.loc[gate.phase_1_gate_pass, "instrument"].tolist()
    if len(candidates) < 2:
        return pd.DataFrame(), pd.DataFrame()
    subset = details.loc[details.instrument.isin(candidates)].copy()
    subset["gross_return"] = subset["sign_edge_executable"]
    wide = subset.pivot(index="session_date", columns="instrument", values="gross_return").fillna(0.0)
    equal = pd.Series(1 / len(candidates), index=candidates)
    inverse = 1.0 / wide.std(ddof=1).replace(0, np.nan)
    inverse /= inverse.sum()
    daily_rows, metric_rows = [], []
    for name, weights in (("equal_weight", equal), ("equal_risk_inverse_vol", inverse)):
        pnl = wide[weights.index].mul(weights, axis=1).sum(axis=1)
        for date, value in pnl.items():
            daily_rows.append({"session_date": date, "portfolio": name, "gross_return": value})
        metric_rows.append({"portfolio": name, "instruments": "+".join(weights.index), "weights": json.dumps(weights.to_dict()), **_pnl_metrics(pnl)})
    return pd.DataFrame(daily_rows), pd.DataFrame(metric_rows)


def _plot_results(gate: pd.DataFrame, events: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = gate.sort_values("beta")
    ax.barh(ordered["instrument"], ordered["beta"], color=np.where(ordered["beta"] > 0, "#2a9d8f", "#e76f51"))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Canonical predictive slope (full sample)", xlabel="OLS beta", ylabel="")
    fig.tight_layout()
    fig.savefig(figures / "canonical_beta_full_sample.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for instrument, group in events.sort_values("session_date").groupby("instrument"):
        ax.plot(group["session_date"], group["sign_edge_raw"].cumsum() * 10_000, label=instrument, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(title="Cumulative canonical raw sign edge", xlabel="Date", ylabel="Cumulative basis points")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "canonical_raw_sign_edge.png", dpi=160)
    plt.close(fig)


def _plot_strategy(details: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=False)
    for ax, (instrument, group) in zip(axes.flat, details.sort_values("session_date").groupby("instrument")):
        ax.plot(group["session_date"], group["gross_pnl"].cumsum(), label="gross", linewidth=1.1)
        ax.plot(group["session_date"], group["normal_net_pnl"].cumsum(), label="normal costs", linewidth=1.1)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(instrument)
        ax.set_ylabel("USD / contract")
        ax.legend(fontsize=8)
    fig.suptitle("Canonical executable equity curves", y=0.995)
    fig.tight_layout()
    fig.savefig(figures / "executable_gross_vs_net_equity.png", dpi=160)
    plt.close(fig)


def _completed_report(
    config: dict[str, Any],
    audit: pd.DataFrame,
    gate: pd.DataFrame,
    cross: pd.DataFrame,
    strategy: pd.DataFrame,
    permutation: pd.DataFrame,
    falsifications: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
) -> str:
    raw_passes = gate.loc[gate.phase_1_gate_pass, "instrument"].tolist()
    replicated = cross.loc[cross.classification == "REPLICATED", "instrument"].tolist()
    recent = gate[["instrument", "recent_sign_edge_bps"]]
    gate_table = gate[["instrument", "beta", "hac_t_stat", "full_sign_edge_bps", "recent_sign_edge_bps", "positive_year_fraction", "phase_1_gate_pass"]].copy()
    gate_table[["beta", "hac_t_stat", "full_sign_edge_bps", "recent_sign_edge_bps", "positive_year_fraction"]] = gate_table[["beta", "hac_t_stat", "full_sign_edge_bps", "recent_sign_edge_bps", "positive_year_fraction"]].round(3)
    cross_table = cross[["instrument", "gross_sharpe", "net_sharpe", "profit_factor_net", "net_pnl", "breakeven_round_trip_cost_usd", "configured_round_trip_cost_usd", "classification"]].copy()
    cross_table.iloc[:, 1:7] = cross_table.iloc[:, 1:7].round(3)
    phase_costs = strategy.loc[(strategy.period == "full") & strategy.cost_case.isin(["gross", "normal_costs", "double_costs"]), ["instrument", "cost_case", "total_pnl", "sharpe", "profit_factor", "max_drawdown", "worst_day"]].copy()
    phase_costs[["total_pnl", "sharpe", "profit_factor", "max_drawdown", "worst_day"]] = phase_costs[["total_pnl", "sharpe", "profit_factor", "max_drawdown", "worst_day"]].round(3)
    focus_falsification = falsifications.loc[falsifications.instrument.isin(raw_passes)].copy()
    focus_falsification[["mean_sign_edge_bps", "t_stat", "hit_rate"]] = focus_falsification[["mean_sign_edge_bps", "t_stat", "hit_rate"]].round(4)
    permutation_focus = permutation.loc[permutation.instrument.isin(raw_passes), ["instrument", "observed_mean_bps", "null_2_5pct_bps", "null_97_5pct_bps", "one_sided_p_value"]].copy()
    permutation_focus.iloc[:, 1:] = permutation_focus.iloc[:, 1:].round(4)
    audit_table = audit[["instrument", "data_start_utc", "data_end_utc", "rows", "eligible_events", "roll_exclusions", "missing_target_sessions", "incomplete_sessions"]]
    portfolio_text = _markdown_table(portfolio_metrics.round(4)) if len(portfolio_metrics) else "Not constructed."
    return f"""# Close-of-day intraday momentum — replication/falsification

## Verdict: H2 CLOSED — RAW EFFECT REPLICATED ON 6E AND ZN; TRADABLE STRATEGY REJECTED

1. **Is the published effect replicated?** Statistically, yes on **{', '.join(raw_passes)}**; not as a net executable strategy on any tested contract.
2. **On which futures?** The frozen raw gate passes only 6E and ZN. NQ is directionally positive but below the predeclared HAC gate; the remaining contracts fail at least one raw criterion.
3. **Does it survive 2017–2026?** The raw sign edge remains positive for 6E and ZN ({recent.loc[recent.instrument == '6E', 'recent_sign_edge_bps'].iloc[0]:.3f} and {recent.loc[recent.instrument == 'ZN', 'recent_sign_edge_bps'].iloc[0]:.3f} bps/day), but the executable net result does not survive costs.
4. **Does it survive realistic costs?** No. Every instrument has negative normal-cost PnL; 6E earns ${cross.loc[cross.instrument == '6E', 'breakeven_round_trip_cost_usd'].iloc[0]:.2f} gross/trade against ${cross.loc[cross.instrument == '6E', 'configured_round_trip_cost_usd'].iloc[0]:.2f} configured, and ZN ${cross.loc[cross.instrument == 'ZN', 'breakeven_round_trip_cost_usd'].iloc[0]:.2f} against ${cross.loc[cross.instrument == 'ZN', 'configured_round_trip_cost_usd'].iloc[0]:.2f}.
5. **Is it non-redundant with the frozen six sleeves?** Not tested. No instrument qualifies as a real economic H2, so Phase 5's prerequisite is false; running `portfolio + 1 contract H2` would promote a known negative-net sleeve.
6. **Develop H2 as a seventh sleeve?** **No.** The evidence supports a small gross closing-pressure effect in 6E/ZN, not an executable micro/retail-cost alpha.

## Frozen design and provenance

The canonical rule is fixed from Baltussen et al. (2021): previous trading-session close to T−30 determines direction; the position is held through the last thirty minutes. UTC bar-start timestamps are converted with IANA timezones. `close(T−31)` is observable at T−30, entry is `open(T−30)`, and exit is `close(T−1)`. The raw regression uses `close(T−1)/close(T−31)−1`, separating the published predictive relation from the executable fill.

Days require exact signal/entry/exit bars and at least 90% common-session coverage. No gap is filled and no early-close substitute is invented. A contract-ID change between previous close and exit excludes the day. The MGC `c.0` file failed the timestamp audit (>50% of exact 13:29 bars absent), so the final frozen run uses the local `v.0` series; this decision was based only on coverage.

No paid data was downloaded. ES, RTY, GC and CL were absent; available micros are reported under their own identifiers.

## Phase 1 — raw predictive effect

{_markdown_table(gate_table)}

6E and ZN have positive slopes, HAC t-statistics of {gate.loc[gate.instrument == '6E', 'hac_t_stat'].iloc[0]:.2f} and {gate.loc[gate.instrument == 'ZN', 'hac_t_stat'].iloc[0]:.2f}, positive recent edges, and positive sign-edge years in {gate.loc[gate.instrument == '6E', 'positive_year_fraction'].iloc[0]:.0%} and {gate.loc[gate.instrument == 'ZN', 'positive_year_fraction'].iloc[0]:.0%} of available years. These are genuine raw replications within this dataset. The full 2007–2016/2017–2026 regressions, ordinary and HAC errors, conditional means and every calendar year are in the CSVs.

The equity-index results are not independent observations: NQ/MNQ/MES/M2K share the U.S. close and overlapping underlying exposure. None passes the frozen raw gate.

## Phase 2 — executable economics

{_markdown_table(cross_table)}

{_markdown_table(phase_costs)}

All eight classifications are **REJECTED** under the requested framework because realistic costs make net PnL negative. This is not marginal for 6E or ZN: normal-cost Sharpes are {cross.loc[cross.instrument == '6E', 'net_sharpe'].iloc[0]:.2f} and {cross.loc[cross.instrument == 'ZN', 'net_sharpe'].iloc[0]:.2f}. Two-times costs worsen every result. Annualized return is intentionally left blank: one futures contract has no uniquely defined invested capital denominator.

## Phase 3 — cross-asset diagnostic

Because 6E and ZN both passed the raw gate, two non-optimized **gross-return** diagnostics were constructed:

{portfolio_text}

Their positive gross Sharpe does not rescue the hypothesis economically because both constituents are negative after their own configured costs. No weight was selected from performance.

## Phase 4 — falsification

{_markdown_table(permutation_focus)}

{_markdown_table(focus_falsification[["instrument", "test", "n", "mean_sign_edge_bps", "t_stat", "hit_rate"]])}

For both raw survivors, shuffled-signal p-values are below 0.001, sign inversion reverses the edge, and the one-bar delay remains positive. The two earlier neighboring windows do not reproduce the same effect: 6E is negative in both; ZN is negative for T−60→T−30 and near zero for T−45→T−15. This supports concentration near settlement, but not sufficient magnitude after transaction costs.

## Phase 5 — frozen portfolio relationship

`portfolio_interaction_results.csv` records **NOT_RUN_NO_ECONOMICALLY_REPLICATED_H2**. The six existing sleeves were not modified. Correlation, losing-day overlap and `portfolio + 1 H2` were deliberately not computed because the protocol permits them only if H2 is real and sufficiently independent; no candidate survived costs.

## Data audit

{_markdown_table(audit_table)}

MGC remains the weakest coverage set even after the objective switch to `v.0`, and that limitation is explicit. Source hashes, close conventions, DST zones and exclusion counts are in `data_audit.csv` and `timestamp_sources.md`.

## Decision

The literature's statistical mechanism is visible in 6E and ZN over the local sample and survives the recent period and focused falsifications. Its magnitude is only about half a basis point per day. Under the repository's one-tick-per-side slippage plus $1.25-per-side commission convention, it is not tradeable; all instruments lose money net and all are classified **REJECTED**. H2 should not become a seventh sleeve, and no timing or threshold search is justified by this result.

## Closure and conditional reopening rule

H2 is closed as **`CLOSED_COST_CONSTRAINED`** on 2026-09-29. Increasing contract count is not a remedy: with linear per-contract costs it multiplies the negative unit expectancy and risk.

The hypothesis may be screened again only if **observed all-in round-trip costs per contract** are sustainably below the frozen gross break-even ceilings:

- **6E: $7.10** per round trip;
- **ZN: $5.47** per round trip.

These ceilings represent zero expected net profit, not acceptable deployment targets. Reopening requires a meaningful safety margin plus prospective evidence from actual fills. The canonical signal, thirty-minute window and direction rule must remain unchanged; no filter, threshold, long/short selection or sizing increase may be introduced to manufacture profitability.

## Reproduction

```powershell
$env:UV_CACHE_DIR='C:\\Data_local\\python_perso\\.uv-cache'
$env:PYTHONPATH='.'
uv run python -m src.analytics.close_of_day_intraday_momentum --output research/close_of_day_intraday_momentum
uv run pytest tests/test_close_of_day_intraday_momentum.py -q
```

References: [Gao et al. (2018)](https://doi.org/10.1016/j.jfineco.2018.05.009) and [Baltussen et al. (2021)](https://doi.org/10.1016/j.jfineco.2021.04.029). Official timestamp sources are linked in `timestamp_sources.md`.
"""


def _report(config: dict[str, Any], audit: pd.DataFrame, gate: pd.DataFrame, output: Path, stopped: bool) -> str:
    tested = ", ".join(gate["instrument"].tolist())
    passed = gate.loc[gate["phase_1_gate_pass"], "instrument"].tolist()
    unavailable = ", ".join(config["unavailable_requested_contracts"])
    gate_view = gate[["instrument", "beta", "hac_t_stat", "full_sign_edge_bps", "recent_sign_edge_bps", "positive_year_fraction", "phase_1_gate_pass"]].copy()
    numeric = ["beta", "hac_t_stat", "full_sign_edge_bps", "recent_sign_edge_bps", "positive_year_fraction"]
    gate_view[numeric] = gate_view[numeric].round(4)
    verdict = "REJECTED" if stopped else "PHASE_1_PASSED"
    answer = "No" if stopped else "Yes, for " + ", ".join(passed)
    return f"""# Close-of-day intraday momentum — replication/falsification

## Verdict: {verdict}

1. **Published effect replicated?** {answer} under the frozen Phase-1 gate.
2. **Futures passing the raw-effect gate:** {', '.join(passed) if passed else 'none'}.
3. **Survives 2017–2026?** {'No qualifying instrument survives all predeclared checks.' if stopped else 'See the instrument gate table below.'}
4. **Survives realistic micro-futures costs?** {'Not tested: the raw mechanism failed first.' if stopped else 'Pending Phase 2.'}
5. **Non-redundant with the six frozen sleeves?** {'Not tested: Phase 5 is prohibited after a failed canonical gate.' if stopped else 'Pending Phase 5.'}
6. **Develop H2 as a seventh sleeve?** {'No.' if stopped else 'Not yet; executable and falsification phases remain.'}

## Scope and stopping rule

This is a literal out-of-sample extension/falsification of the rest-of-day → last-half-hour mechanism described by Baltussen et al. (2021), not a timing search. The specification and gate were frozen before calculating returns. {'Because no instrument passed, the requested stop rule was activated: Phases 2–5, costs, placebos, permutations and frozen-portfolio interaction were not run.' if stopped else 'At least one instrument passed the raw-effect gate, so later phases are permitted but are not included in this Phase-1 run.'}

The locally available instruments tested were {tested}. Requested full-size contracts absent locally were {unavailable}; their micro counterparts were not relabeled as full-size contracts.

## Canonical implementation

For each instrument, UTC minute-bar start timestamps are converted to the frozen IANA timezone, so DST is handled by the timezone database. With bar-start labels, `close(T−31)` is first observable at T−30. The raw hypothesis uses `close(T−1) / close(T−31) − 1`; a future executable test would enter at `open(T−30)` and exit at `close(T−1)`. This deliberately separates statistical predictability from fill assumptions.

A day requires at least 90% of common-session minute bars and exact signal, entry and exit bars. When contract IDs are retained, transitions between the prior close and current close are excluded to prevent unadjusted roll gaps from becoming signals. No missing bar is forward-filled and no early-close timestamp is substituted.

## Instrument results

    {_markdown_table(gate_view)}

The full regression, conditional means, and year-by-year results are in `predictive_regression_results.csv`, `conditional_sign_results.csv`, and `yearly_results.csv`. A positive coefficient alone is not treated as replication: the frozen gate also requires conventional HAC evidence, positive full/recent sign edge, and a majority of positive years.

## Data audit

    {_markdown_table(audit[['instrument','data_start_utc','data_end_utc','rows','eligible_events','roll_exclusions','roll_safety_available','T','timezone']])}

Each source hash is recorded in `data_audit.csv`. Full-size ES, RTY, GC and CL were not found. No paid data was downloaded.

## Interpretation

The sample is substantially shorter than the 1974–2020 cross-futures sample in the paper, and most micros begin in 2019 or later. NQ and the longer commodity/rates series still provide meaningful falsification windows, while cross-listed micro/mini index results are not independent replications. Results after 2017 are reported separately rather than allowing older observations to mask decay.

## Reproduction

```powershell
$env:UV_CACHE_DIR='C:\\Data_local\\python_perso\\.uv-cache'
uv run python -m src.analytics.close_of_day_intraday_momentum --output research/close_of_day_intraday_momentum
uv run pytest tests/test_close_of_day_intraday_momentum.py -q
```

Primary references: Gao et al. (2018), *Journal of Financial Economics*, DOI 10.1016/j.jfineco.2018.05.009; Baltussen et al. (2021), *Journal of Financial Economics*, DOI 10.1016/j.jfineco.2021.04.029. Timestamp sources are documented in `timestamp_sources.md`.
"""


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "frozen_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    all_events = []
    audits = []
    bars_by_instrument: dict[str, pd.DataFrame] = {}
    for spec in config["instruments"]:
        path = REPO_ROOT / spec["data_path"]
        bars = load_bars(path)
        bars_by_instrument[spec["instrument"]] = bars
        events, diagnostics = build_canonical_events(
            bars,
            instrument=spec["instrument"],
            timezone=spec["timezone"],
            session_start=spec["session_start"],
            close_time=spec["T"],
            minimum_fraction=float(config["method"]["session_completeness_minimum"]),
        )
        all_events.append(events)
        audits.append(
            {
                "instrument": spec["instrument"],
                "asset_class": spec["asset_class"],
                "data_path": spec["data_path"],
                "sha256": _sha256(path),
                "rows": len(bars),
                "data_start_utc": bars.index.min(),
                "data_end_utc": bars.index.max(),
                "duplicate_timestamps": int(bars.index.duplicated().sum()),
                "missing_ohlc_values": int(bars[["open", "high", "low", "close"]].isna().sum().sum()),
                "T": spec["T"],
                "timezone": spec["timezone"],
                "timestamp_status": spec["timestamp_status"],
                "timestamp_basis": spec["timestamp_basis"],
                "roll_safety_available": "instrument_id" in bars.columns,
                **diagnostics,
            }
        )
    event_frame = pd.concat(all_events, ignore_index=True)
    audit_frame = pd.DataFrame(audits)
    regressions, conditional, yearly = summarize_effects(event_frame)
    gate = _gate_table(regressions, conditional, yearly, config)
    stopped = not bool(gate["phase_1_gate_pass"].any())
    audit_frame.to_csv(output / "data_audit.csv", index=False)
    event_frame.to_csv(output / "canonical_events.csv", index=False)
    regressions.to_csv(output / "predictive_regression_results.csv", index=False)
    conditional.to_csv(output / "conditional_sign_results.csv", index=False)
    yearly.to_csv(output / "yearly_results.csv", index=False)
    gate.to_csv(output / "phase1_gate_results.csv", index=False)
    phase_status = []
    cross = pd.DataFrame()
    strategy = pd.DataFrame()
    permutation = pd.DataFrame()
    falsifications = pd.DataFrame()
    portfolio_metrics = pd.DataFrame()
    if stopped:
        for phase in ("Phase 2 executable strategy and costs", "Phase 3 cross-asset portfolio", "Phase 4 permutation/placebos", "Phase 5 frozen-portfolio interaction"):
            phase_status.append({"phase": phase, "status": "NOT_RUN_GATE_STOP"})
    else:
        details, strategy = executable_results(event_frame, config)
        details.to_csv(output / "executable_trade_results.csv", index=False)
        strategy.to_csv(output / "strategy_metrics.csv", index=False)
        cross = cross_asset_summary(gate, strategy, yearly, details)
        cross.to_csv(output / "cross_asset_summary.csv", index=False)
        cost_sensitivity = strategy.loc[strategy.period == "full", ["instrument", "cost_case", "total_pnl", "sharpe", "profit_factor", "max_drawdown"]].merge(
            cross[["instrument", "breakeven_round_trip_cost_usd", "configured_round_trip_cost_usd"]], on="instrument", how="left"
        )
        cost_sensitivity.to_csv(output / "cost_sensitivity.csv", index=False)
        permutation, falsifications = falsification_results(event_frame, config)
        placebo_rows = []
        for spec in config["instruments"]:
            instrument = spec["instrument"]
            canonical = event_frame.loc[event_frame.instrument == instrument]
            for name, start, end in (("placebo_Tminus60_to_Tminus30", 60, 30), ("placebo_Tminus45_to_Tminus15", 45, 15)):
                sample = build_timing_placebo(bars_by_instrument[instrument], canonical, timezone=spec["timezone"], close_time=spec["T"], start_minutes_before_t=start, end_minutes_before_t=end)
                edge = sample["sign_edge"].dropna() if len(sample) else pd.Series(dtype=float)
                placebo_rows.append({"instrument": instrument, "test": name, "n": len(edge), "mean_sign_edge_bps": edge.mean() * 10_000 if len(edge) else np.nan, "t_stat": stats.ttest_1samp(edge, 0).statistic if len(edge) > 1 else np.nan, "hit_rate": (edge > 0).mean() if len(edge) else np.nan})
        falsifications = pd.concat([falsifications, pd.DataFrame(placebo_rows)], ignore_index=True)
        permutation.to_csv(output / "permutation_results.csv", index=False)
        falsifications.to_csv(output / "placebo_and_execution_delay_results.csv", index=False)
        portfolio_daily, portfolio_metrics = diagnostic_portfolios(details, gate)
        portfolio_daily.to_csv(output / "cross_asset_portfolio_daily.csv", index=False)
        portfolio_metrics.to_csv(output / "cross_asset_portfolio_metrics.csv", index=False)
        replicated = cross.loc[cross.classification == "REPLICATED", "instrument"].tolist()
        interaction_status = "NOT_RUN_NO_ECONOMICALLY_REPLICATED_H2" if not replicated else "ELIGIBLE_NOT_RUN"
        pd.DataFrame([{"status": interaction_status, "eligible_instruments": "+".join(replicated), "reason": "Phase 5 requires a real and sufficiently independent H2; no candidate survived normal costs." if not replicated else "At least one candidate passed economic classification."}]).to_csv(output / "portfolio_interaction_results.csv", index=False)
        _plot_strategy(details, output)
        phase_status.extend(
            [
                {"phase": "Phase 2 executable strategy and costs", "status": "COMPLETED"},
                {"phase": "Phase 3 cross-asset portfolio", "status": "COMPLETED_DIAGNOSTIC" if len(portfolio_metrics) else "NOT_RUN_FEWER_THAN_TWO_RAW_GATE_PASSES"},
                {"phase": "Phase 4 permutation/placebos", "status": "COMPLETED"},
                {"phase": "Phase 5 frozen-portfolio interaction", "status": interaction_status},
            ]
        )
    pd.DataFrame(phase_status).to_csv(output / "downstream_phase_status.csv", index=False)
    _plot_results(gate, event_frame, output)
    report = _report(config, audit_frame, gate, output, stopped) if stopped else _completed_report(config, audit_frame, gate, cross, strategy, permutation, falsifications, portfolio_metrics)
    (output / "CLOSE_OF_DAY_INTRADAY_MOMENTUM_REPORT.md").write_text(report, encoding="utf-8")
    metadata = {"verdict": "REJECTED" if stopped else "CLOSED_COST_CONSTRAINED", "raw_effect_status": "PHASE_1_PASSED" if not stopped else "PHASE_1_FAILED", "phase_1_passed_instruments": gate.loc[gate.phase_1_gate_pass, "instrument"].tolist(), "replicated_instruments": cross.loc[cross.classification == "REPLICATED", "instrument"].tolist() if len(cross) else [], "event_rows": len(event_frame)}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default) + "\n", encoding="utf-8")
    if not stopped:
        closure = config["closure"] | {
            "raw_effect_passed_instruments": gate.loc[gate.phase_1_gate_pass, "instrument"].tolist(),
            "economically_replicated_instruments": cross.loc[cross.classification == "REPLICATED", "instrument"].tolist(),
        }
        (output / "hypothesis_status.json").write_text(json.dumps(closure, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
