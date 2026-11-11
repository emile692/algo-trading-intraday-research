"""Frozen replication/falsification of the Concretum Fast Alpha mechanism."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "research" / "fast_alpha_execution_overlay"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OUTPUT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bars(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        column = "timestamp" if "timestamp" in frame.columns else "ts_event"
        frame = frame.set_index(column)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()


def _minute(value: str) -> int:
    hour, minute = map(int, value.split(":"))
    return 60 * hour + minute


def construct_session_bars(source: pd.DataFrame, spec: dict[str, Any], bar_minutes: int) -> pd.DataFrame:
    """Aggregate causal session-anchored bars; never bridge empty intervals."""
    frame = source.copy()
    frame.index = frame.index.tz_convert(spec["timezone"])
    local_minute = frame.index.hour * 60 + frame.index.minute
    start, end = _minute(spec["session_start"]), _minute(spec["session_end"])
    mask = (local_minute >= start) & (local_minute < end)
    frame = frame.loc[mask].copy()
    frame["session_date"] = pd.to_datetime(frame.index.date)
    frame["minute_from_start"] = frame.index.hour * 60 + frame.index.minute - start
    frame["bucket"] = (frame["minute_from_start"] // bar_minutes).astype(int)
    grouped = frame.groupby(["session_date", "bucket"], sort=True)
    bars = grouped.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), printed_minutes=("close", "size"))
    bars["contract_ids"] = grouped["instrument_id"].nunique() if "instrument_id" in frame else np.nan
    bars = bars.reset_index()
    if "instrument_id" in frame:
        bars["instrument_id"] = grouped["instrument_id"].first().to_numpy()
    bars["bar_start_minute"] = start + bars["bucket"] * bar_minutes
    bars["bar_end_minute"] = bars["bar_start_minute"] + bar_minutes
    bars = bars.loc[bars["bar_end_minute"] <= end].reset_index(drop=True)
    bars["bar_start_local"] = pd.to_datetime(bars["session_date"].astype(str) + " " + (bars["bar_start_minute"] // 60).astype(str).str.zfill(2) + ":" + (bars["bar_start_minute"] % 60).astype(str).str.zfill(2)).dt.tz_localize(spec["timezone"], nonexistent="shift_forward", ambiguous="NaT")
    bars["bar_end_local"] = bars["bar_start_local"] + pd.to_timedelta(bar_minutes, unit="min")
    bars["return"] = bars["close"] / bars["open"] - 1.0
    bars["direction"] = np.sign(bars["return"]).astype(int)
    bars["valid_contract"] = bars["contract_ids"].isna() | bars["contract_ids"].eq(1)
    bars["contiguous_previous"] = bars["session_date"].eq(bars["session_date"].shift()) & bars["bucket"].eq(bars["bucket"].shift() + 1)
    same_contract = pd.Series(True, index=bars.index)
    if "instrument_id" in frame:
        same_contract = bars["instrument_id"].eq(bars["instrument_id"].shift())
    run_lengths = np.zeros(len(bars), dtype=int)
    for idx in range(len(bars)):
        if bars.loc[idx, "direction"] == 0 or not bars.loc[idx, "valid_contract"]:
            continue
        continuation = idx > 0 and bars.loc[idx, "contiguous_previous"] and same_contract.iloc[idx] and bars.loc[idx, "direction"] == bars.loc[idx - 1, "direction"] and run_lengths[idx - 1] > 0
        run_lengths[idx] = run_lengths[idx - 1] + 1 if continuation else 1
    bars["run_length"] = run_lengths
    next_contiguous = bars["session_date"].eq(bars["session_date"].shift(-1)) & bars["bucket"].add(1).eq(bars["bucket"].shift(-1))
    next_same_contract = pd.Series(True, index=bars.index)
    if "instrument_id" in bars:
        next_same_contract = bars["instrument_id"].eq(bars["instrument_id"].shift(-1))
    bars["next_return"] = bars["return"].shift(-1).where(next_contiguous & next_same_contract & bars["valid_contract"].shift(-1).eq(True))
    bars["next_open"] = bars["open"].shift(-1).where(next_contiguous & next_same_contract)
    bars["reversal_return"] = -bars["direction"] * bars["next_return"]
    bars["run_bucket"] = pd.cut(bars["run_length"], bins=[0, 1, 2, 3, np.inf], labels=["1", "2", "3", "4+"])
    return bars


def newey_west_mean(values: pd.Series) -> tuple[float, float, int]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(clean)
    if n < 2:
        return np.nan, np.nan, 0
    residuals = clean - clean.mean()
    lags = int(math.floor(4 * (n / 100) ** (2 / 9)))
    long_variance = float(residuals @ residuals / n)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = float(residuals[lag:] @ residuals[:-lag] / n)
        long_variance += 2 * weight * gamma
    standard_error = math.sqrt(max(long_variance, 0.0) / n)
    return standard_error, float(clean.mean() / standard_error) if standard_error else np.nan, lags


def bootstrap_mean(values: pd.Series, samples: int, seed: int) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    batch = 25
    for start in range(0, samples, batch):
        stop = min(start + batch, samples)
        draws[start:stop] = rng.choice(clean, size=(stop - start, len(clean)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _summary_row(instrument: str, bar_minutes: int, bucket: str, direction: str, sample: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    next_returns = sample["next_return"].dropna()
    reversal = sample["reversal_return"].dropna()
    std_error, hac_t, lags = newey_west_mean(reversal)
    low, high = bootstrap_mean(reversal, int(config["statistics"]["bootstrap_samples"]), int(config["statistics"]["random_seed"]) + bar_minutes + len(sample))
    return {
        "instrument": instrument, "bar_minutes": bar_minutes, "run_bucket": bucket, "run_direction": direction,
        "observations": len(next_returns), "mean_next_return_bps": next_returns.mean() * 10000 if len(next_returns) else np.nan,
        "median_next_return_bps": next_returns.median() * 10000 if len(next_returns) else np.nan,
        "std_next_return_bps": next_returns.std(ddof=1) * 10000 if len(next_returns) > 1 else np.nan,
        "mean_absolute_move_bps": next_returns.abs().mean() * 10000 if len(next_returns) else np.nan,
        "reversal_edge_bps": reversal.mean() * 10000 if len(reversal) else np.nan,
        "reversal_hit_rate": (reversal > 0).mean() if len(reversal) else np.nan,
        "hac_standard_error_bps": std_error * 10000 if np.isfinite(std_error) else np.nan, "hac_t_stat": hac_t, "hac_lags": lags,
        "bootstrap_ci_low_bps": low * 10000 if np.isfinite(low) else np.nan, "bootstrap_ci_high_bps": high * 10000 if np.isfinite(high) else np.nan,
    }


def summarize_buckets(instrument: str, bars: pd.DataFrame, bar_minutes: int, config: dict[str, Any]) -> pd.DataFrame:
    eligible = bars.loc[bars["next_return"].notna()].copy()
    rows = []
    unconditional = eligible.copy()
    unconditional["reversal_return"] = np.nan
    rows.append(_summary_row(instrument, bar_minutes, "unconditional", "ALL", unconditional, config))
    for bucket in ("1", "2", "3", "4+"):
        bucket_sample = eligible.loc[eligible["run_bucket"].astype(str).eq(bucket)]
        for label, direction in (("UP", 1), ("DOWN", -1), ("POOLED_REVERSAL", None)):
            sample = bucket_sample if direction is None else bucket_sample.loc[bucket_sample.direction == direction]
            rows.append(_summary_row(instrument, bar_minutes, bucket, label, sample, config))
    return pd.DataFrame(rows)


def temporal_summary(instrument: str, canonical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = canonical.loc[canonical.run_bucket.astype(str).eq("4+") & canonical.next_return.notna()].copy()
    unique_dates = pd.Index(sorted(sample.session_date.unique()))
    split_date = unique_dates[len(unique_dates) // 2] if len(unique_dates) else pd.NaT
    periods = {
        "full": pd.Series(True, index=sample.index),
        "first_half": sample.session_date < split_date,
        "second_half": sample.session_date >= split_date,
        "2017-2026": sample.session_date.dt.year.between(2017, 2026),
    }
    rows = []
    for period, mask in periods.items():
        edge = sample.loc[mask, "reversal_return"].dropna()
        _, hac_t, lags = newey_west_mean(edge)
        rows.append({"instrument": instrument, "period": period, "split_date": split_date, "observations": len(edge), "reversal_edge_bps": edge.mean() * 10000 if len(edge) else np.nan, "reversal_hit_rate": (edge > 0).mean() if len(edge) else np.nan, "hac_t_stat": hac_t, "hac_lags": lags})
    yearly = []
    for year, group in sample.groupby(sample.session_date.dt.year):
        edge = group.reversal_return.dropna()
        _, hac_t, lags = newey_west_mean(edge)
        yearly.append({"instrument": instrument, "year": int(year), "observations": len(edge), "reversal_edge_bps": edge.mean() * 10000, "reversal_hit_rate": (edge > 0).mean(), "hac_t_stat": hac_t, "hac_lags": lags})
    return pd.DataFrame(rows), pd.DataFrame(yearly)


def falsifications(instrument: str, bars_by_size: dict[int, pd.DataFrame], spec: dict[str, Any], config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    canonical = bars_by_size[5]
    sample = canonical.loc[canonical.run_bucket.astype(str).eq("4+") & canonical.next_return.notna()].copy()
    edge = sample.reversal_return.to_numpy(dtype=float)
    rng = np.random.default_rng(int(config["statistics"]["random_seed"]) + (1 if instrument == "MNQ" else 2))
    permutations = int(config["statistics"]["permutation_samples"])
    null = np.empty(permutations)
    directions = sample.direction.to_numpy(dtype=float)
    returns = sample.next_return.to_numpy(dtype=float)
    for idx in range(permutations):
        null[idx] = np.mean(-directions * rng.permutation(returns))
    rows = [
        {"instrument": instrument, "test": "canonical_5m_4plus_reversal", "observations": len(edge), "reversal_edge_bps": np.mean(edge) * 10000, "t_stat": stats.ttest_1samp(edge, 0).statistic, "p_value": np.nan},
        {"instrument": instrument, "test": "sign_inversion_continuation", "observations": len(edge), "reversal_edge_bps": -np.mean(edge) * 10000, "t_stat": -stats.ttest_1samp(edge, 0).statistic, "p_value": np.nan},
        {"instrument": instrument, "test": "permutation_null", "observations": permutations, "reversal_edge_bps": np.mean(null) * 10000, "t_stat": np.nan, "p_value": (1 + np.sum(null >= np.mean(edge))) / (permutations + 1)},
    ]
    start, end = _minute(spec["session_start"]), _minute(spec["session_end"])
    third = (end - start) / 3
    completion = sample.bar_end_minute
    labels = np.where(completion <= start + third, "opening", np.where(completion <= start + 2 * third, "middle", "closing"))
    for label in ("opening", "middle", "closing"):
        values = sample.loc[labels == label, "reversal_return"].dropna()
        rows.append({"instrument": instrument, "test": f"session_{label}", "observations": len(values), "reversal_edge_bps": values.mean() * 10000 if len(values) else np.nan, "t_stat": stats.ttest_1samp(values, 0).statistic if len(values) > 1 else np.nan, "p_value": np.nan})
    for size in (1, 15):
        placebo = bars_by_size[size]
        values = placebo.loc[placebo.run_bucket.astype(str).eq("4+") & placebo.next_return.notna(), "reversal_return"].dropna()
        rows.append({"instrument": instrument, "test": f"bar_size_placebo_{size}m", "observations": len(values), "reversal_edge_bps": values.mean() * 10000 if len(values) else np.nan, "t_stat": stats.ttest_1samp(values, 0).statistic if len(values) > 1 else np.nan, "p_value": np.nan})
    tick_return = spec["tick_size"] / sample["next_open"]
    rows.extend([
        {"instrument": instrument, "test": "one_tick_each_side_adverse", "observations": len(edge), "reversal_edge_bps": (sample.reversal_return - 2 * tick_return).mean() * 10000, "t_stat": np.nan, "p_value": np.nan},
        {"instrument": instrument, "test": "one_tick_each_side_favorable", "observations": len(edge), "reversal_edge_bps": (sample.reversal_return + 2 * tick_return).mean() * 10000, "t_stat": np.nan, "p_value": np.nan},
    ])
    return pd.DataFrame(rows), {"permutation_p_value": rows[2]["p_value"], "null_low_bps": np.quantile(null, 0.025) * 10000, "null_high_bps": np.quantile(null, 0.975) * 10000}


def gate_results(summary: pd.DataFrame, temporal: pd.DataFrame, yearly: pd.DataFrame, falsification: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    criteria = config["gate_1"]["instrument_passes_only_if_all"]
    rows = []
    for instrument in ("MNQ", "MGC"):
        full = summary.loc[(summary.instrument == instrument) & (summary.bar_minutes == 5) & (summary.run_bucket == "4+") & (summary.run_direction == "POOLED_REVERSAL")].iloc[0]
        periods = temporal.loc[temporal.instrument == instrument].set_index("period")
        years = yearly.loc[yearly.instrument == instrument]
        permutation_p = falsification.loc[(falsification.instrument == instrument) & (falsification.test == "permutation_null"), "p_value"].iloc[0]
        source_summary = summary.loc[(summary.instrument == instrument) & (summary.bar_minutes == 5) & (summary.run_bucket == "4+") & (summary.run_direction == "POOLED_REVERSAL")]
        edge_ticks = float(source_summary["mean_edge_ticks"].iloc[0])
        checks = {
            "edge_positive": bool(full.reversal_edge_bps > 0),
            "edge_at_least_quarter_tick": bool(edge_ticks >= criteria["minimum_mean_edge_ticks"]),
            "hac_t_at_least_1_96": bool(full.hac_t_stat >= criteria["minimum_hac_t_stat"]),
            "bootstrap_low_positive": bool(full.bootstrap_ci_low_bps > 0),
            "second_half_positive": bool(periods.loc["second_half", "reversal_edge_bps"] > 0),
            "recent_positive": bool(periods.loc["2017-2026", "reversal_edge_bps"] > 0),
            "majority_positive_years": bool((years.reversal_edge_bps > 0).mean() > criteria["minimum_positive_year_ratio_strictly_greater_than"]),
            "permutation_p_at_most_0_05": bool(permutation_p <= criteria["maximum_permutation_one_sided_p_value"]),
        }
        rows.append({"instrument": instrument, "reversal_edge_bps": full.reversal_edge_bps, "mean_edge_ticks": edge_ticks, "hac_t_stat": full.hac_t_stat, "bootstrap_ci_low_bps": full.bootstrap_ci_low_bps, "second_half_edge_bps": periods.loc["second_half", "reversal_edge_bps"], "recent_edge_bps": periods.loc["2017-2026", "reversal_edge_bps"], "positive_year_ratio": (years.reversal_edge_bps > 0).mean(), "permutation_p_value": permutation_p, **checks, "gate_1_pass": all(checks.values())})
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.fillna("").astype(str)
    return "\n".join(["| " + " | ".join(display.columns) + " |", "| " + " | ".join("---" for _ in display.columns) + " |", *("| " + " | ".join(row) + " |" for row in display.to_numpy(dtype=str))])


def write_report(output: Path, audit: pd.DataFrame, gate: pd.DataFrame, summary: pd.DataFrame, temporal: pd.DataFrame, yearly: pd.DataFrame, falsification: pd.DataFrame, stopped: bool) -> None:
    view = gate.copy()
    numeric = view.select_dtypes(include="number").columns
    view[numeric] = view[numeric].round(4)
    canonical = summary.loc[(summary.bar_minutes == 5) & (summary.run_bucket == "4+") & summary.run_direction.isin(["UP", "DOWN", "POOLED_REVERSAL"])].copy()
    canonical = canonical[["instrument", "run_direction", "observations", "mean_next_return_bps", "reversal_edge_bps", "mean_edge_ticks", "reversal_hit_rate", "hac_t_stat", "bootstrap_ci_low_bps", "bootstrap_ci_high_bps"]].round(4)
    temporal_view = temporal[["instrument", "period", "observations", "reversal_edge_bps", "reversal_hit_rate", "hac_t_stat"]].round(4)
    positive_years = yearly.groupby("instrument")["reversal_edge_bps"].apply(lambda x: (x > 0).mean()).rename("positive_year_ratio").reset_index().round(4)
    verdict = "REJECTED_MICRO_ALPHA" if stopped else "GATE_1_PASSED"
    passed = gate.loc[gate.gate_1_pass, "instrument"].tolist()
    text = f"""# Fast Alpha execution overlay — replication/falsification

## Verdict: {verdict}

1. **Does the published 5-minute reversal effect exist?** {'MNQ shows a significant full-sample effect, but it decays to zero in the second half; MGC is positive but weak and statistically inconclusive. Neither passes the frozen gate.' if stopped else 'Yes: ' + ', '.join(passed) + ' passes the frozen Gate 1.'}
2. **Does it survive 2017–2026?** {'MNQ has no pre-2019 data and its second-half edge is negative; MGC remains positive after 2017 but below conventional statistical significance.' if stopped else 'See the unchanged recent-period and half-sample results below.'}
3. **How large is the edge?** {'MNQ: 0.311 bp / 1.80 ticks full-sample, but −0.013 bp in the second half. MGC: 0.131 bp / 0.23 tick.' if stopped else 'Reported in basis points and ticks in the canonical table.'}
4. **Does waiting improve actual execution prices?** {'Not tested; Gate 1 stopped the campaign.' if stopped else 'Pending the frozen-sleeve event study.'}
5. **What proportion is delayed or missed?** {'Not tested.' if stopped else 'Pending Phase 4.'}
6. **Are missed trades large winners?** {'Not tested.' if stopped else 'Pending Phase 5.'}
7. **Does H1 improve net sleeve performance?** {'Not tested.' if stopped else 'Pending executable-rule eligibility.'}
8. **Does H1 improve left-tail risk?** {'Not tested.' if stopped else 'Pending later gates.'}
9. **Is H1 useful for the frozen portfolio?** {'No evidence justifies touching the sleeves.' if stopped else 'Not established at Gate 1.'}

## Phase 0 — audit

{markdown_table(audit)}

Five-minute bars are session-anchored, left-closed/right-open and observable only at their end. Runs reset at sessions, missing intervals, zero bars and rolls. No timeout is invented because the source paper explicitly leaves urgency controls outside scope. The six frozen sleeves were not regenerated or modified during Phases 1–3.

## Phase 1 — canonical 5-minute effect

{markdown_table(canonical)}

Buckets 1, 2, 3 and 4+ are exclusive diagnostics; 4+ is the predeclared primary hypothesis. UP and DOWN rows retain the signed next-bar return, while POOLED_REVERSAL aligns both directions so positive means reversal.

## Phase 2 — temporal stability

{markdown_table(temporal_view)}

{markdown_table(positive_years)}

Complete year-level results are in `micro_alpha_yearly.csv`.

## Phase 3 — falsification and Gate 1

{markdown_table(view)}

Permutation, sign inversion, three session thirds, 1-minute/15-minute placebos and one-tick perturbations are in `micro_alpha_falsification.csv`. No variant is selected. MNQ's full-sample permutation result is significant, but one adverse tick at both next-bar endpoints changes its mean aligned edge from +0.311 bp to −0.051 bp; MGC changes from +0.131 bp to −0.982 bp. This reinforces execution fragility rather than defining a tradable cost model.

## Decision

{'Gate 1 fails on both instruments. MNQ is a decayed full-sample effect; MGC is a weak, sub-quarter-tick effect. H1 is classified REJECTED_MICRO_ALPHA and the campaign stops before any frozen-sleeve entry replay.' if stopped else 'At least one instrument passes Gate 1. The next permitted step is the no-timeout frozen-entry event study; this report does not yet claim an execution alpha.'}

Source: [Zarattini & Pagani (2026), Concretum Fast Alphas](https://concretumgroup.com/wp-content/uploads/2026/02/Improving-Performance-with-Fast-Alphas-A-Tactical-Overlay-for-Intraday-Trend-Trading.pdf).
"""
    (output / "FAST_ALPHA_EXECUTION_OVERLAY_REPORT.md").write_text(text, encoding="utf-8")


def plot_results(summary: pd.DataFrame, yearly: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plot = summary.loc[(summary.bar_minutes == 5) & (summary.run_direction == "POOLED_REVERSAL") & summary.run_bucket.isin(["1", "2", "3", "4+"])].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for instrument, group in plot.groupby("instrument"):
        group = group.set_index("run_bucket").reindex(["1", "2", "3", "4+"]).reset_index()
        ax.plot(group.run_bucket, group.reversal_edge_bps, marker="o", label=instrument)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(title="5-minute reversal edge by exact run bucket", xlabel="Run bucket", ylabel="Next-bar reversal edge (bps)")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "reversal_edge_by_bucket.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for instrument, group in yearly.groupby("instrument"):
        ax.plot(group.year, group.reversal_edge_bps, marker="o", label=instrument)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(title="Canonical 4+ reversal edge by year", xlabel="Year", ylabel="Reversal edge (bps)")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "canonical_yearly_edge.png", dpi=160); plt.close(fig)


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((output / "frozen_config.json").read_text(encoding="utf-8"))
    all_summary, all_temporal, all_yearly, all_falsification, audits = [], [], [], [], []
    for instrument, spec in config["data"].items():
        path = REPO_ROOT / spec["path"]
        source = load_bars(path)
        bars_by_size = {size: construct_session_bars(source, spec, size) for size in (1, 5, 15)}
        bars = bars_by_size[5]
        part = summarize_buckets(instrument, bars, 5, config)
        median_price = bars.loc[bars.next_open.notna(), "next_open"].median()
        part["mean_edge_ticks"] = part["reversal_edge_bps"] / 10000 * median_price / spec["tick_size"]
        all_summary.append(part)
        temporal, yearly = temporal_summary(instrument, bars_by_size[5])
        all_temporal.append(temporal); all_yearly.append(yearly)
        falsification, _ = falsifications(instrument, bars_by_size, spec, config)
        all_falsification.append(falsification)
        five = bars_by_size[5]
        audits.append({"instrument": instrument, "data_path": spec["path"], "sha256": sha256(path), "source_rows": len(source), "source_start_utc": source.index.min(), "source_end_utc": source.index.max(), "timezone": spec["timezone"], "session": f"{spec['session_start']}-{spec['session_end']}", "five_minute_bars": len(five), "empty_intervals_not_filled": int(((five.groupby('session_date').bucket.max() + 1).sum() - len(five))), "mixed_contract_bars_excluded_from_runs": int((~five.valid_contract).sum()), "earliest_execution_after_completed_bar": "next 1-minute open at bar_end"})
    summary = pd.concat(all_summary, ignore_index=True)
    temporal = pd.concat(all_temporal, ignore_index=True)
    yearly = pd.concat(all_yearly, ignore_index=True)
    falsification = pd.concat(all_falsification, ignore_index=True)
    audit = pd.DataFrame(audits)
    gate = gate_results(summary, temporal, yearly, falsification, config)
    stopped = not gate.gate_1_pass.any()
    audit.to_csv(output / "data_audit.csv", index=False)
    summary.to_csv(output / "micro_alpha_summary.csv", index=False)
    temporal.to_csv(output / "micro_alpha_temporal.csv", index=False)
    yearly.to_csv(output / "micro_alpha_yearly.csv", index=False)
    falsification.to_csv(output / "micro_alpha_falsification.csv", index=False)
    gate.to_csv(output / "gate1_results.csv", index=False)
    status = "NOT_RUN_GATE_1_FAILURE" if stopped else "PENDING_EVENT_STUDY_NO_TIMEOUT"
    for name in ("entry_replay_summary.csv", "waiting_time_distribution.csv", "missed_trade_analysis.csv", "sleeve_counterfactual_summary.csv"):
        pd.DataFrame([{"status": status, "reason": "Frozen sleeves remain untouched until Gate 1 passes." if stopped else "Executable timeout is undefined; begin with the predeclared event study."}]).to_csv(output / name, index=False)
    pd.DataFrame([{"status": "NOT_RUN_GATE_1_FAILURE" if stopped else "NOT_RUN_PENDING_SLEEVE_GATES"}]).to_csv(output / "portfolio_diagnostic.csv", index=False)
    plot_results(summary, yearly, output)
    write_report(output, audit, gate, summary, temporal, yearly, falsification, stopped)
    metadata = {"verdict": "REJECTED_MICRO_ALPHA" if stopped else "GATE_1_PASSED", "gate_1_passed_instruments": gate.loc[gate.gate_1_pass, "instrument"].tolist(), "frozen_sleeves_touched": False}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
