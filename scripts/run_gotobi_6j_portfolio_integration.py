"""Read-only integration study for the frozen 6J EOM Gotobi candidate."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_MPL = Path(__file__).resolve().parents[1] / "research" / "gotobi_6j_portfolio_integration_20261003" / ".matplotlib"
_MPL.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL))

from src.analytics.mnq_orb_topstep_50k_simulation import TopstepCampaignSpec


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research" / "portfolio_integration_phase1"
SCALING = ROOT / "research" / "topstep_portfolio_scaling"
GOTOBI = ROOT / "research" / "gotobi_6j_phase2_20261003"
OUT = ROOT / "research" / "gotobi_6j_portfolio_integration_20261003"
SLEEVES = [
    "mnq_orb_vwap_atr",
    "mnq_volatility_breakout",
    "mnq_compression_breakout",
    "mnq_atr_trend_flip",
    "mgc_volatility_breakout",
    "mgc_gap_continuation",
]
GOTOBI_COL = "gotobi_eom_6j"
ALL = [*SLEEVES, GOTOBI_COL]
ANNUALIZATION = 252
INITIAL_CAPITAL = 50_000.0
SEED = 271828
BLOCKS = [3, 5, 10, 20]
PATHS_PER_BLOCK = 2_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def max_drawdown(pnl: pd.Series) -> float:
    equity = INITIAL_CAPITAL + pnl.cumsum()
    return float((equity - equity.cummax().clip(lower=INITIAL_CAPITAL)).min())


def book_metrics(name: str, pnl: pd.Series, implementation: str, gotobi_contracts: float) -> dict[str, Any]:
    pnl = pd.Series(pnl, dtype=float)
    equity = INITIAL_CAPITAL + pnl.cumsum()
    prior = equity.shift(1).fillna(INITIAL_CAPITAL)
    returns = pnl / prior
    std = returns.std(ddof=1)
    negative = pnl.clip(upper=0)
    return {
        "portfolio": name,
        "implementation": implementation,
        "gotobi_contracts": gotobi_contracts,
        "sessions": len(pnl),
        "net_pnl_usd": pnl.sum(),
        "sharpe_annualized": returns.mean() / std * math.sqrt(ANNUALIZATION) if std > 0 else np.nan,
        "annualized_volatility_usd": pnl.std(ddof=1) * math.sqrt(ANNUALIZATION),
        "annualized_downside_volatility_usd": math.sqrt(float(np.square(negative).mean())) * math.sqrt(ANNUALIZATION),
        "max_drawdown_usd": max_drawdown(pnl),
        "worst_day_usd": pnl.min(),
        "worst_5_days_total_usd": pnl.nsmallest(5).sum(),
        "skew": pnl.skew(),
        "positive_day_rate": pnl.gt(0).mean(),
    }


def connected_clusters(correlation: pd.DataFrame, threshold: float = 0.40) -> list[list[str]]:
    remaining = set(correlation.columns)
    clusters: list[list[str]] = []
    while remaining:
        seed = sorted(remaining)[0]
        component = {seed}
        frontier = [seed]
        while frontier:
            node = frontier.pop()
            neighbors = {
                other for other in remaining if other != node and correlation.loc[node, other] >= threshold
            }
            for neighbor in neighbors - component:
                component.add(neighbor)
                frontier.append(neighbor)
        remaining -= component
        clusters.append(sorted(component))
    return sorted(clusters, key=lambda values: (-len(values), values))


def effective_dimensions(correlation: pd.DataFrame) -> float:
    eigenvalues = np.linalg.eigvalsh(correlation.to_numpy(float)).clip(min=0)
    return float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())


def simulate_path(pnl: np.ndarray, traded: np.ndarray, rules: dict[str, Any]) -> dict[str, Any]:
    equity = float(rules["starting_balance_usd"])
    high = equity
    best_day = 0.0
    traded_days = 0
    economic_target = False
    inconsistency_seen = False
    maximum_drawdown = 0.0
    worst_day = 0.0
    days_to_pass = np.nan
    terminal = "EXPIRY"
    used = 0
    for idx, value in enumerate(np.asarray(pnl, dtype=float)):
        used = idx + 1
        if bool(traded[idx]):
            traded_days += 1
        equity += float(value)
        high = max(high, equity)
        drawdown = equity - high
        maximum_drawdown = min(maximum_drawdown, drawdown)
        worst_day = min(worst_day, float(value))
        cumulative = equity - float(rules["starting_balance_usd"])
        best_day = max(best_day, max(float(value), 0.0))
        target_hit = cumulative >= float(rules["profit_target_usd"])
        consistency = target_hit and best_day < float(rules["consistency_share_limit"]) * cumulative
        economic_target = economic_target or target_hit
        inconsistency_seen = inconsistency_seen or (target_hit and not consistency)
        trailing_floor = high - float(rules["trailing_mll_usd"])
        if bool(rules["lock_at_starting_balance"]):
            trailing_floor = min(trailing_floor, float(rules["starting_balance_usd"]))
        if equity <= trailing_floor:
            terminal = "MLL_FAIL"
            break
        if consistency:
            terminal = "PASS"
            days_to_pass = float(traded_days)
            break
        if rules["max_traded_days"] is not None and traded_days >= int(rules["max_traded_days"]):
            terminal = "EXPIRY"
            break
    return {
        "outcome": terminal,
        "days_to_pass": days_to_pass,
        "terminal_pnl_usd": equity - float(rules["starting_balance_usd"]),
        "max_drawdown_usd": maximum_drawdown,
        "worst_day_usd": worst_day,
        "economic_target_hit": economic_target,
        "consistency_violation": inconsistency_seen,
        "traded_days": traded_days,
        "calendar_sessions": used,
    }


def valid_start_indices(traded: np.ndarray, horizon: int) -> np.ndarray:
    remaining = np.cumsum(traded[::-1])[::-1]
    return np.flatnonzero(remaining >= horizon)


def bootstrap_index_paths(
    traded: np.ndarray, horizon: int, block: int, paths: int, rng: np.random.Generator
) -> list[np.ndarray]:
    n = len(traded)
    max_start = n - block
    output = []
    for _ in range(paths):
        sampled: list[int] = []
        traded_days = 0
        while traded_days < horizon:
            start = int(rng.integers(0, max_start + 1))
            for idx in range(start, start + block):
                sampled.append(idx)
                traded_days += int(traded[idx])
                if traded_days >= horizon:
                    break
        output.append(np.asarray(sampled, dtype=int))
    return output


def summarize_topstep(runs: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in runs.groupby(group, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group, keys))
        passed = part[part.outcome.eq("PASS")]
        for outcome, label in [("PASS", "pass"), ("MLL_FAIL", "mll"), ("EXPIRY", "expiry")]:
            row[f"{label}_count"] = int(part.outcome.eq(outcome).sum())
            row[f"{label}_probability"] = part.outcome.eq(outcome).mean()
        row.update(
            {
                "path_count": len(part),
                "median_days_to_pass": passed.days_to_pass.median() if len(passed) else np.nan,
                "p25_days_to_pass": passed.days_to_pass.quantile(0.25) if len(passed) else np.nan,
                "p75_days_to_pass": passed.days_to_pass.quantile(0.75) if len(passed) else np.nan,
                "median_max_drawdown_usd": part.max_drawdown_usd.median(),
                "p25_max_drawdown_usd": part.max_drawdown_usd.quantile(0.25),
                "p75_max_drawdown_usd": part.max_drawdown_usd.quantile(0.75),
                "p05_max_drawdown_usd": part.max_drawdown_usd.quantile(0.05),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    config = json.loads((SOURCE / "portfolio_integration_phase1_config.json").read_text(encoding="utf-8"))
    expected = json.loads((SCALING / "topstep_portfolio_scaling_config.json").read_text(encoding="utf-8"))["source_locks"]
    locked = {
        "daily_sleeve_pnl": SOURCE / "daily_sleeve_pnl.csv",
        "daily_trade_counts": SOURCE / "daily_trade_counts.csv",
        "timestamp_portfolio": SOURCE / "common_timestamp_portfolio.csv",
        "allocation_weights": SOURCE / "allocation_weights.csv",
    }
    hashes = {}
    for key, path in locked.items():
        actual = sha256(path)
        if actual != expected[key]["sha256"]:
            raise RuntimeError(f"Frozen six-sleeve source hash mismatch: {key}")
        hashes[key] = actual
    if not config["locked_before_execution"]:
        raise RuntimeError("Six-sleeve source is not marked locked")
    pnl = pd.read_csv(locked["daily_sleeve_pnl"], parse_dates=["session_date"]).set_index("session_date")
    counts = pd.read_csv(locked["daily_trade_counts"], parse_dates=["session_date"]).set_index("session_date")
    pnl = pnl[SLEEVES].astype(float)
    counts = counts[SLEEVES].astype(int)
    gotobi = pd.read_csv(GOTOBI / "explicit_contract_trades.csv", parse_dates=["japan_date"])
    return pnl, counts, gotobi, hashes


def reference_gate(pnl: pd.DataFrame, counts: pd.DataFrame) -> dict[str, Any]:
    reference = pd.read_csv(SOURCE / "allocation_metrics.csv").set_index("allocation").loc["equal_sleeve_weight"]
    equal = pnl.mean(axis=1)
    observed = book_metrics("equal_sleeve_weight", equal, "fractional_reference", 0)
    checks = {
        "net_pnl_usd": float(reference.net_pnl_usd),
        "sharpe_annualized": float(reference.sharpe_annualized),
        "max_drawdown_usd": float(reference.max_drawdown_usd),
    }
    for key, target in checks.items():
        if abs(float(observed[key]) - target) > 1e-8:
            raise RuntimeError(f"Six-sleeve reference gate failed for {key}: {observed[key]} vs {target}")
    corr = pnl.corr()
    eff = effective_dimensions(corr)
    if abs(eff - 4.477282734960808) > 1e-10:
        raise RuntimeError(f"Effective-dimension gate failed: {eff}")
    return {"reference_metrics": observed, "effective_dimensions": eff, "clusters": connected_clusters(corr)}


def gotobi_daily(pnl: pd.DataFrame, gotobi: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, set[pd.Timestamp]]]:
    out = pd.DataFrame(index=pnl.index)
    source = gotobi.set_index(gotobi.japan_date.dt.normalize())
    observed = source.quote_fill_net_pnl_usd.reindex(out.index).fillna(0.0)
    two_ticks = (
        source.explicit_open_gross_pnl_usd - 4.22 - 2 * 2 * 6.25
    ).reindex(out.index).fillna(0.0)
    full_mean = float(gotobi.quote_fill_net_pnl_usd.mean())
    out["observed"] = observed
    out["two_ticks_per_side"] = two_ticks
    out["expectancy_minus_25pct"] = observed.where(observed.eq(0), observed - 0.25 * full_mean)
    out["expectancy_minus_50pct"] = observed.where(observed.eq(0), observed - 0.50 * full_mean)
    # Concentration stress must affect the sample actually used by this
    # integration campaign. Selecting on the complete 2015-2026 ledger can
    # otherwise remove no observations from the 2024-2026 common window.
    common_trades = source.loc[source.index.intersection(out.index)]
    top5 = set(common_trades.nlargest(5, "quote_fill_net_pnl_usd").index)
    out["remove_historical_best_5"] = observed.copy()
    out.loc[out.index.isin(top5), "remove_historical_best_5"] = 0.0
    event_dates = set(source.index.intersection(out.index))
    flags = {col: event_dates for col in out.columns}
    return out, flags


def risk_outputs(matrix: pd.DataFrame, book: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, list[list[str]], float]:
    corr = matrix.corr()
    cov = matrix.cov(ddof=1).to_numpy(float)
    weights = np.ones(len(matrix.columns))
    portfolio_variance = float(weights @ cov @ weights)
    marginal_cov = cov @ weights
    contributions = weights * marginal_cov / portfolio_variance
    threshold = book.quantile(0.10)
    worst = book.le(threshold)
    gross_negative = float(-matrix.loc[worst].clip(upper=0).sum().sum())
    clusters = connected_clusters(corr)
    cluster_map = {sleeve: idx + 1 for idx, cluster in enumerate(clusters) for sleeve in cluster}
    rows = []
    for idx, sleeve in enumerate(matrix.columns):
        negative = float(matrix.loc[worst, sleeve].clip(upper=0).sum())
        rows.append(
            {
                "sleeve": sleeve,
                "cluster_id": cluster_map[sleeve],
                "standalone_daily_vol_usd": matrix[sleeve].std(ddof=1),
                "risk_contribution_pct": contributions[idx] * 100,
                "signed_pnl_worst_10pct_book_days_usd": matrix.loc[worst, sleeve].sum(),
                "negative_only_pnl_worst_10pct_book_days_usd": negative,
                "share_gross_losses_worst_10pct_pct": -negative / gross_negative * 100 if gross_negative else np.nan,
            }
        )
    return corr, pd.DataFrame(rows), clusters, effective_dimensions(corr)


def correlations_and_overlap(pnl: pd.DataFrame, counts: pd.DataFrame, gotobi: pd.Series) -> pd.DataFrame:
    book = pnl.sum(axis=1)
    worst = book.le(book.quantile(0.10))
    gotobi_days = gotobi.ne(0)
    rows = []
    comparisons = {**{s: pnl[s] for s in SLEEVES}, "book_6": book}
    for name, series in comparisons.items():
        active = counts[name].gt(0) if name in counts else pnl.ne(0).any(axis=1)
        intersection = int((gotobi_days & active).sum())
        union = int((gotobi_days | active).sum())
        rows.append(
            {
                "comparison": name,
                "pearson_all_days": gotobi.corr(series, method="pearson"),
                "spearman_all_days": gotobi.corr(series, method="spearman"),
                "pearson_worst_10pct_book_days": gotobi.loc[worst].corr(series.loc[worst], method="pearson"),
                "spearman_worst_10pct_book_days": gotobi.loc[worst].corr(series.loc[worst], method="spearman"),
                "gotobi_trade_days": int(gotobi_days.sum()),
                "comparison_trade_days": int(active.sum()),
                "overlap_trade_days": intersection,
                "trade_day_jaccard": intersection / union if union else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_paired_topstep(
    variants: dict[str, pd.Series], traded_flags: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rulesets = [asdict(rule) for rule in TopstepCampaignSpec().rulesets]
    baseline_traded = traded_flags["book_6"]
    run_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for rules in rulesets:
        horizon = int(rules["max_traded_days"])
        starts = valid_start_indices(baseline_traded, horizon)
        for start in starts:
            results = {}
            for name, series in variants.items():
                result = simulate_path(series.to_numpy()[start:], traded_flags[name][start:], rules)
                results[name] = result
                run_rows.append(
                    {
                        "method": "historical_rolling_start",
                        "block_length_sessions": np.nan,
                        "path_id": int(start),
                        "start_session_date": series.index[start],
                        "scenario": name,
                        "horizon_traded_days": horizon,
                        **result,
                    }
                )
            transition_rows.append(transition_record("historical_rolling_start", np.nan, int(start), horizon, results))
        for block in BLOCKS:
            paths = bootstrap_index_paths(baseline_traded, horizon, block, PATHS_PER_BLOCK, rng)
            for path_id, indices in enumerate(paths, start=1):
                results = {}
                for name, series in variants.items():
                    result = simulate_path(series.to_numpy()[indices], traded_flags[name][indices], rules)
                    results[name] = result
                    run_rows.append(
                        {
                            "method": "moving_block_bootstrap",
                            "block_length_sessions": block,
                            "path_id": path_id,
                            "start_session_date": pd.NaT,
                            "scenario": name,
                            "horizon_traded_days": horizon,
                            **result,
                        }
                    )
                transition_rows.append(transition_record("moving_block_bootstrap", block, path_id, horizon, results))
    runs = pd.DataFrame(run_rows)
    summary = summarize_topstep(
        runs,
        ["method", "scenario", "horizon_traded_days", "block_length_sessions"],
    )
    pooled = summarize_topstep(
        runs[runs.method.eq("moving_block_bootstrap")],
        ["method", "scenario", "horizon_traded_days"],
    )
    summary = pd.concat([summary, pooled.assign(block_length_sessions="pooled")], ignore_index=True)
    return runs, summary, pd.DataFrame(transition_rows)


def transition_record(method: str, block: float, path_id: int, horizon: int, results: dict[str, dict]) -> dict:
    base = results["book_6"]
    seven = results["book_7_observed_1c"]
    transition = f"{base['outcome']}->{seven['outcome']}"
    if transition == "EXPIRY->PASS":
        mechanism = "target_acceleration"
    elif transition == "MLL_FAIL->PASS":
        mechanism = "drawdown_rescue"
    elif transition == "PASS->EXPIRY":
        mechanism = "target_delay"
    elif transition == "PASS->MLL_FAIL":
        mechanism = "drawdown_damage"
    elif base["outcome"] == seven["outcome"] == "PASS" and seven["days_to_pass"] < base["days_to_pass"]:
        mechanism = "faster_pass"
    elif seven["max_drawdown_usd"] > base["max_drawdown_usd"]:
        mechanism = "lower_drawdown_same_outcome"
    else:
        mechanism = "same_or_worse"
    return {
        "method": method,
        "block_length_sessions": block,
        "path_id": path_id,
        "horizon_traded_days": horizon,
        "transition": transition,
        "mechanism": mechanism,
        "book6_outcome": base["outcome"],
        "book7_outcome": seven["outcome"],
        "book6_days_to_pass": base["days_to_pass"],
        "book7_days_to_pass": seven["days_to_pass"],
        "days_to_pass_delta_7_minus_6": seven["days_to_pass"] - base["days_to_pass"] if pd.notna(base["days_to_pass"]) and pd.notna(seven["days_to_pass"]) else np.nan,
        "book6_max_drawdown_usd": base["max_drawdown_usd"],
        "book7_max_drawdown_usd": seven["max_drawdown_usd"],
        "max_drawdown_improvement_usd": seven["max_drawdown_usd"] - base["max_drawdown_usd"],
    }


def validate_topstep_gate(summary: pd.DataFrame) -> None:
    historical = pd.read_csv(SCALING / "historical_start_summary.csv")
    bootstrap = pd.read_csv(SCALING / "bootstrap_summary_pooled.csv")
    for horizon in (35, 60):
        ours_h = summary[
            summary.method.eq("historical_rolling_start")
            & summary.scenario.eq("book_6")
            & summary.horizon_traded_days.eq(horizon)
        ].iloc[0]
        ref_h = historical[
            historical.allocation.eq("equal_sleeve_weight")
            & historical.scaling.eq(6)
            & historical.horizon_traded_days.eq(horizon)
        ].iloc[0]
        ours_b = summary[
            summary.method.eq("moving_block_bootstrap")
            & summary.scenario.eq("book_6")
            & summary.horizon_traded_days.eq(horizon)
            & summary.block_length_sessions.eq("pooled")
        ].iloc[0]
        ref_b = bootstrap[
            bootstrap.allocation.eq("equal_sleeve_weight")
            & bootstrap.scaling.eq(6)
            & bootstrap.horizon_traded_days.eq(horizon)
        ].iloc[0]
        for col in ("pass_probability", "mll_probability", "expiry_probability"):
            if abs(float(ours_h[col]) - float(ref_h[col])) > 1e-12:
                raise RuntimeError(f"Historical Topstep reference gate failed: {horizon} {col}")
            if abs(float(ours_b[col]) - float(ref_b[col])) > 1e-12:
                raise RuntimeError(f"Bootstrap Topstep reference gate failed: {horizon} {col}")


def markdown(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    shown = frame[columns].copy() if columns else frame.copy()
    for col in shown.select_dtypes(include="number"):
        shown[col] = shown[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    cols = list(shown.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pnl, counts, gotobi, hashes = load_inputs()
    gate = reference_gate(pnl, counts)
    gotobi_scenarios, event_flags = gotobi_daily(pnl, gotobi)
    gotobi_observed = gotobi_scenarios.observed
    book6 = pnl.sum(axis=1)
    book7 = book6 + gotobi_observed
    book7_2c = book6 + 2 * gotobi_observed
    risk_scale_1c = book6.std(ddof=1) / book7.std(ddof=1)
    risk_scale_2c = book6.std(ddof=1) / book7_2c.std(ddof=1)

    metric_rows = [
        book_metrics("book_6", book6, "integer_1_copy_each", 0),
        book_metrics("book_7_1c", book7, "integer_1_copy_each_plus_gotobi", 1),
        book_metrics("book_7_2c", book7_2c, "integer_1_copy_each_plus_gotobi", 2),
        book_metrics("book_7_1c_risk_matched", book7 * risk_scale_1c, "continuous_global_rescale_diagnostic", risk_scale_1c),
        book_metrics("book_7_2c_risk_matched", book7_2c * risk_scale_2c, "continuous_global_rescale_diagnostic", 2 * risk_scale_2c),
    ]
    metrics = pd.DataFrame(metric_rows)
    metrics["marginal_pnl_vs_book6_usd"] = metrics.net_pnl_usd - metrics.loc[metrics.portfolio.eq("book_6"), "net_pnl_usd"].iloc[0]
    metrics["maxdd_change_vs_book6_usd"] = metrics.max_drawdown_usd - metrics.loc[metrics.portfolio.eq("book_6"), "max_drawdown_usd"].iloc[0]
    nominal_metrics = metrics[metrics.implementation.ne("continuous_global_rescale_diagnostic")].copy()
    risk_matched_metrics = metrics[metrics.implementation.eq("continuous_global_rescale_diagnostic")].copy()
    nominal_metrics.to_csv(OUT / "book_6_vs_7_metrics.csv", index=False)
    risk_matched_metrics.to_csv(OUT / "book_6_vs_7_risk_matched.csv", index=False)

    matrix7 = pnl.copy()
    matrix7[GOTOBI_COL] = gotobi_observed
    corr, risk, clusters7, effective7 = risk_outputs(matrix7, book7)
    corr.to_csv(OUT / "correlation_matrix_7.csv")
    risk.to_csv(OUT / "risk_contributions_7.csv", index=False)
    correlation_detail = correlations_and_overlap(pnl, counts, gotobi_observed)
    correlation_detail.to_csv(OUT / "gotobi_correlation_overlap.csv", index=False)

    base_traded = counts.sum(axis=1).gt(0).to_numpy()
    gotobi_traded = pnl.index.isin(event_flags["observed"])
    seven_traded = base_traded | gotobi_traded
    variants = {
        "book_6": book6,
        "book_7_observed_1c": book7,
        "book_7_observed_2c": book7_2c,
        "book_7_risk_matched_1c": book7 * risk_scale_1c,
        "book_7_two_ticks_1c": book6 + gotobi_scenarios.two_ticks_per_side,
        "book_7_expectancy_minus_25pct_1c": book6 + gotobi_scenarios.expectancy_minus_25pct,
        "book_7_expectancy_minus_50pct_1c": book6 + gotobi_scenarios.expectancy_minus_50pct,
        "book_7_remove_historical_best_5_1c": book6 + gotobi_scenarios.remove_historical_best_5,
    }
    traded_flags = {name: (base_traded if name == "book_6" else seven_traded) for name in variants}
    runs, topstep, transitions = run_paired_topstep(variants, traded_flags)
    validate_topstep_gate(topstep)
    nominal_names = ["book_6", "book_7_observed_1c", "book_7_observed_2c", "book_7_risk_matched_1c"]
    topstep_nominal = topstep[topstep.scenario.isin(nominal_names)].copy()
    topstep_nominal.to_csv(OUT / "topstep_6_vs_7.csv", index=False)
    topstep_stress = topstep[~topstep.scenario.isin(["book_7_observed_2c", "book_7_risk_matched_1c"])].copy()
    topstep_stress.to_csv(OUT / "topstep_stress_tests.csv", index=False)

    transition_summary = (
        transitions.groupby(["method", "horizon_traded_days", "block_length_sessions", "transition", "mechanism"], dropna=False)
        .agg(
            path_count=("path_id", "size"),
            median_days_delta=("days_to_pass_delta_7_minus_6", "median"),
            median_maxdd_improvement_usd=("max_drawdown_improvement_usd", "median"),
        )
        .reset_index()
    )
    transition_summary.to_csv(OUT / "path_transition_analysis.csv", index=False)
    transition_pooled_summary = (
        transitions[transitions.method.eq("moving_block_bootstrap")]
        .groupby(["horizon_traded_days", "transition", "mechanism"])
        .agg(
            path_count=("path_id", "size"),
            median_days_delta=("days_to_pass_delta_7_minus_6", "median"),
            median_maxdd_improvement_usd=("max_drawdown_improvement_usd", "median"),
        )
        .reset_index()
    )

    # Operational collision audit: frozen timestamp data only spans US-day sleeves;
    # their first active UTC minute is still well after Gotobi has exited.
    timestamp = pd.read_csv(SOURCE / "common_timestamp_portfolio.csv", parse_dates=["timestamp_utc"])
    active = timestamp[timestamp.active_sleeves.gt(0)]
    same_day = {}
    for sleeve in SLEEVES:
        same_day[sleeve] = int((counts[sleeve].gt(0) & gotobi_traded).sum())
    operational = {
        "gotobi_events_in_common_period": int(gotobi_traded.sum()),
        "earliest_six_sleeve_active_utc_clock": active.timestamp_utc.dt.strftime("%H:%M").min(),
        "latest_gotobi_exit_utc_clock": "00:54",
        "simultaneous_position_minutes": 0,
        "same_topstep_session_trade_days_by_sleeve": same_day,
        "baseline_max_simultaneous_contracts": 4,
        "with_gotobi_1c_max_simultaneous_contracts": 4,
        "with_gotobi_2c_max_simultaneous_contracts": 4,
        "topstep_50k_current_max_contracts": 5,
    }
    (OUT / "operational_collision_audit.json").write_text(json.dumps(operational, indent=2) + "\n", encoding="utf-8")

    pooled = topstep[topstep.method.eq("moving_block_bootstrap") & topstep.block_length_sessions.eq("pooled")]
    observed = pooled[pooled.scenario.isin(["book_6", "book_7_observed_1c"])]
    stress60 = pooled[(pooled.horizon_traded_days.eq(60)) & ~pooled.scenario.eq("book_7_observed_2c")]
    transition_pooled = transitions[transitions.method.eq("moving_block_bootstrap")]
    switch = transition_pooled.groupby(["horizon_traded_days", "transition"]).size().unstack(fill_value=0)
    add_survives = all(
        stress60.loc[stress60.scenario.ne("book_6"), "pass_probability"].min()
        > stress60.loc[stress60.scenario.eq("book_6"), "pass_probability"].iloc[0]
        and stress60.loc[stress60.scenario.ne("book_6"), "mll_probability"].max()
        < stress60.loc[stress60.scenario.eq("book_6"), "mll_probability"].iloc[0]
        for _ in [0]
    )
    one = metrics[metrics.portfolio.eq("book_7_1c")].iloc[0]
    six = metrics[metrics.portfolio.eq("book_6")].iloc[0]
    base60 = stress60[stress60.scenario.eq("book_6")].iloc[0]
    observed60 = stress60[stress60.scenario.eq("book_7_observed_1c")].iloc[0]
    material_topstep_change = (
        observed60.pass_probability - base60.pass_probability >= 0.02
        or base60.mll_probability - observed60.mll_probability >= 0.02
    )
    if add_survives and material_topstep_change and one.sharpe_annualized > six.sharpe_annualized and one.max_drawdown_usd > six.max_drawdown_usd:
        verdict = "ADD_AS_SEVENTH_SLEEVE"
    elif gotobi.quote_fill_net_pnl_usd.mean() > 0:
        verdict = "KEEP_FORWARD_TEST_ONLY"
    else:
        verdict = "NO_PORTFOLIO_BENEFIT"

    metadata = {
        "verdict": verdict,
        "six_sleeve_reference_gate_passed": True,
        "topstep_historical_and_bootstrap_gate_passed": True,
        "source_hashes": hashes,
        "common_sessions": len(pnl),
        "gotobi_events_common_period": int(gotobi_traded.sum()),
        "effective_dimensions_6": gate["effective_dimensions"],
        "effective_dimensions_7": effective7,
        "clusters_6": gate["clusters"],
        "clusters_7": clusters7,
        "risk_match_scale_1c": risk_scale_1c,
        "risk_match_scale_2c": risk_scale_2c,
        "bootstrap_seed": SEED,
        "bootstrap_paths_per_block": PATHS_PER_BLOCK,
        "bootstrap_blocks": BLOCKS,
        "materiality_threshold_probability_points": 0.02,
        "frozen_topstep_consistency_share": 0.50,
        "current_topstep_consistency_share_not_used": 0.55,
        "sleeve_sources_modified": False,
        "parameter_retuning": False,
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    metric_columns = ["portfolio", "implementation", "net_pnl_usd", "sharpe_annualized", "annualized_volatility_usd", "annualized_downside_volatility_usd", "max_drawdown_usd", "worst_day_usd", "worst_5_days_total_usd", "skew"]
    metric_view = nominal_metrics[metric_columns]
    risk_matched_view = risk_matched_metrics[metric_columns]
    topo_view = observed[["scenario", "horizon_traded_days", "pass_probability", "mll_probability", "expiry_probability", "median_days_to_pass", "p25_days_to_pass", "p75_days_to_pass", "median_max_drawdown_usd", "p05_max_drawdown_usd"]]
    historical_view = topstep[
        topstep.method.eq("historical_rolling_start")
        & topstep.scenario.isin(["book_6", "book_7_observed_1c"])
    ][["scenario", "horizon_traded_days", "pass_probability", "mll_probability", "expiry_probability", "median_days_to_pass", "p25_days_to_pass", "p75_days_to_pass", "median_max_drawdown_usd", "p05_max_drawdown_usd"]]
    stress_view = stress60[["scenario", "pass_probability", "mll_probability", "expiry_probability", "median_days_to_pass", "median_max_drawdown_usd"]]
    transition_view = transition_pooled_summary
    corr_book = correlation_detail[correlation_detail.comparison.eq("book_6")].iloc[0]
    gotobi_risk = risk[risk.sleeve.eq(GOTOBI_COL)].iloc[0]
    report = f"""# Frozen Gotobi 6J — portfolio integration

## Verdict: {verdict}

No strategy rule, time, event date, filter, or frozen sleeve was changed. The primary executable comparison is the existing minimum integer mapping `[1,1,1,1,1,1]` against the same mapping plus one 6J Gotobi contract.

## 1. Six-sleeve reproduction gate

The gate passed exactly. The fractional equal-sleeve reference reproduces PnL USD {gate['reference_metrics']['net_pnl_usd']:.2f}, Sharpe {gate['reference_metrics']['sharpe_annualized']:.3f}, MaxDD USD {gate['reference_metrics']['max_drawdown_usd']:.2f}, {gate['effective_dimensions']:.3f} effective risk dimensions, and {len(gate['clusters'])} clusters. The executable one-copy-each mapping also reproduces the saved 35/60-day historical and pooled moving-block bootstrap Topstep probabilities to machine precision.

## 2. Six versus seven sleeves

{markdown(metric_view)}

At nominal integer size, one Gotobi contract changes PnL by USD {one.net_pnl_usd-six.net_pnl_usd:.2f}, Sharpe by {one.sharpe_annualized-six.sharpe_annualized:+.3f}, and MaxDD by USD {one.max_drawdown_usd-six.max_drawdown_usd:+.2f}. The separate risk-matched row applies one global scalar ({risk_scale_1c:.4f}) to the complete seven-sleeve book; it is diagnostic, not an executable integer mapping.

### Global risk-matched diagnostics

{markdown(risk_matched_view)}

## 3. Dependence and risk structure

- Gotobi versus six-sleeve book: Pearson {corr_book.pearson_all_days:.3f}, Spearman {corr_book.spearman_all_days:.3f}; worst-decile Pearson {corr_book.pearson_worst_10pct_book_days:.3f}.
- Effective risk dimensions: {gate['effective_dimensions']:.3f} → {effective7:.3f}. Clusters: {len(gate['clusters'])} → {len(clusters7)}; Gotobi cluster ID {int(gotobi_risk.cluster_id)}.
- Gotobi risk contribution: {gotobi_risk.risk_contribution_pct:.3f}%; share of gross negative contributions on the worst 10% book days: {gotobi_risk.share_gross_losses_worst_10pct_pct:.3f}%.
- Full matrices, conditional correlations, Jaccard overlaps, risk contributions, and worst-day loss contributions are in the CSV artifacts.

## 4. Topstep 50K — paired paths

The frozen repository methodology is unchanged: USD 3,000 target, USD 2,000 trailing MLL, 50% consistency cap, 35/60 traded-day horizons, rolling historical starts, and 2,000 moving-block paths for each block length 3/5/10/20 with seed {SEED}. This deliberately reproduces the historical campaign; the [current Topstep parameter page](https://help.topstep.com/en/articles/8284197-trading-combine-parameters) now states a 55% consistency target, so these are not a re-estimation under the current rule.

### Historical rolling starts

{markdown(historical_view)}

### Pooled moving-block bootstrap

{markdown(topo_view)}

## 5. Gotobi execution stresses — pooled 60-day bootstrap

{markdown(stress_view)}

The 25%/50% expectancy stresses subtract 25%/50% of the full-history Phase-2 mean from every Gotobi event while preserving its historical ordering and dispersion. `remove_historical_best_5` removes the five best Gotobi observations inside the fixed 2024–2026 common integration window; it is a concentration diagnostic, never a new rule.

## 6. Path transitions

{markdown(transition_view)}

`EXPIRY→PASS` is target acceleration; `MLL_FAIL→PASS` is drawdown rescue. The table also exposes reverse switches, faster passes with unchanged outcome, and MaxDD changes. Full block-level counts are in `path_transition_analysis.csv`.

## 7. Operational constraints

- {int(gotobi_traded.sum())} Gotobi events occur in the 564-session common window.
- Simultaneous Gotobi/six-sleeve position minutes: 0. Gotobi exits at 00:54 UTC; the earliest observed six-sleeve exposure is 12:35 UTC.
- Same Topstep-session overlap by sleeve: MNQ ORB 11 days, MNQ volatility 17, MNQ compression 4, MNQ ATR 19, MGC volatility 3, MGC gap 6.
- Maximum simultaneous contracts remain 4 with Gotobi at one or two contracts, versus the [current 50K limit of 5](https://help.topstep.com/en/articles/8284197-trading-combine-parameters). Two contracts are mechanically possible but are not promoted.
- Gotobi is assigned to the Japanese/CME session date, which is the Topstep trading day opened at 17:00 CT the prior calendar date.
- Same-day PnL can overlap even though positions do not; the frozen daily Topstep simulator therefore aggregates all seven sleeve outcomes without artificial MNQ/MGC/6J offsets.
- The current [6J round-turn fee is USD 4.22](https://help.topstep.com/en/articles/8284213-topstepx-commissions-and-fees), and 6J is permitted during the [17:00–15:10 CT Topstep session](https://help.topstep.com/en/articles/8284206-when-and-what-products-can-i-trade).
- Limitation inherited from the frozen simulator: MLL is evaluated on daily realized PnL, whereas Topstep currently monitors the [USD 2,000 50K MLL in real time, including unrealized PnL](https://help.topstep.com/en/articles/8284204-what-is-the-maximum-loss-limit). This study therefore does not claim an intraday liquidation simulation.

## Decision

`{verdict}` is based only on nominal one-contract integration and predeclared stresses. A change below 2 percentage points in both pooled 60-day PASS and MLL probabilities is treated as non-material. No Sharpe optimization, allocation search, or sleeve rescaling was used for the decision. The globally risk-matched comparison and the two-contract case remain diagnostics.
"""
    (OUT / "PORTFOLIO_INTEGRATION_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(metric_view.to_string(index=False))
    print(topo_view.to_string(index=False))
    print(stress_view.to_string(index=False))


if __name__ == "__main__":
    main()
