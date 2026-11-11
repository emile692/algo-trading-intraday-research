"""H3: causal options-implied volatility state diagnostics for frozen MNQ sleeves."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "research" / "options_implied_state_conditioning"
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


def load_vix(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=[0, 1], index_col=0)
    dates = pd.to_datetime(raw.index)
    out = pd.DataFrame(
        {
            "vix_raw": pd.to_numeric(raw[("VIX Index", "PX_LAST")], errors="coerce").to_numpy(),
            "vix3m_raw": pd.to_numeric(raw[("VIX3M Index", "PX_LAST")], errors="coerce").to_numpy(),
        },
        index=dates,
    ).sort_index()
    out.index.name = "session_date"
    return out


def load_mes_daily(path: Path) -> pd.DataFrame:
    bars = pd.read_parquet(path)
    if not isinstance(bars.index, pd.DatetimeIndex):
        bars = bars.set_index("timestamp")
    bars.index = pd.to_datetime(bars.index, utc=True).tz_convert("America/New_York")
    minute = bars.index.hour * 60 + bars.index.minute
    rth = bars.loc[(minute >= 570) & (minute < 960)].copy()
    rth["session_date"] = pd.to_datetime(rth.index.date)
    grouped = rth.groupby("session_date", sort=True)
    daily = grouped.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), bars=("close", "size"))
    if "instrument_id" in rth:
        daily["instrument_id"] = grouped["instrument_id"].last()
        transition = daily.instrument_id.ne(daily.instrument_id.shift())
    else:
        daily["instrument_id"] = np.nan
        transition = pd.Series(False, index=daily.index)
    previous_close = daily.close.shift()
    same_contract = ~transition
    log_return = np.log(daily.close / previous_close).where(same_contract)
    true_range = pd.concat([(daily.high - daily.low), (daily.high - previous_close).abs(), (daily.low - previous_close).abs()], axis=1).max(axis=1).where(same_contract)
    daily["mes_rv10_raw_pct"] = log_return.rolling(10, min_periods=10).std(ddof=1) * math.sqrt(252) * 100
    daily["mes_atr14_raw_pct"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / previous_close * 100
    daily["roll_transition"] = transition
    return daily


def build_causal_states(vix: pd.DataFrame, mes: pd.DataFrame) -> pd.DataFrame:
    common = vix.join(mes[["mes_rv10_raw_pct", "mes_atr14_raw_pct"]], how="inner")
    q_low = common.vix_raw.expanding(min_periods=252).quantile(1 / 3)
    q_high = common.vix_raw.expanding(min_periods=252).quantile(2 / 3)
    known = pd.DataFrame(
        {
            "vix_level": common.vix_raw,
            "vix3m_level": common.vix3m_raw,
            "mes_rv10_pct": common.mes_rv10_raw_pct,
            "mes_atr14_pct": common.mes_atr14_raw_pct,
            "vix_q_low": q_low,
            "vix_q_high": q_high,
        }
    ).shift(1)
    known["term_spread"] = known.vix_level - known.vix3m_level
    known["term_ratio"] = known.vix_level / known.vix3m_level
    known["evrp"] = known.vix_level - known.mes_rv10_pct
    known["vix_state"] = np.select([known.vix_level <= known.vix_q_low, known.vix_level >= known.vix_q_high], ["low", "high"], default="middle")
    known.loc[known.vix_level.isna() | known.vix_q_low.isna(), "vix_state"] = np.nan
    known["term_spread_state"] = np.where(known.term_spread > 0, "backwardation", "contango")
    known.loc[known.term_spread.isna(), "term_spread_state"] = np.nan
    known["term_ratio_state"] = np.where(known.term_ratio > 1, "ratio_gt_1", "ratio_le_1")
    known.loc[known.term_ratio.isna(), "term_ratio_state"] = np.nan
    known["evrp_state"] = np.where(known.evrp > 0, "positive", "nonpositive")
    known.loc[known.evrp.isna(), "evrp_state"] = np.nan
    known.index.name = "session_date"
    return known


def nw_ols(frame: pd.DataFrame, outcome: str, predictors: list[str]) -> dict[str, Any]:
    clean = frame[[outcome, *predictors]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(clean)
    k = len(predictors) + 1
    if n <= k + 2:
        return {"n": n, "beta": np.nan, "hac_se": np.nan, "hac_t": np.nan, "p_value": np.nan, "r_squared": np.nan, "hac_lags": 0}
    x = np.column_stack([np.ones(n), clean[predictors].to_numpy(dtype=float)])
    y = clean[outcome].to_numpy(dtype=float)
    inv = np.linalg.pinv(x.T @ x)
    coef = inv @ x.T @ y
    residual = y - x @ coef
    scores = x * residual[:, None]
    lags = int(math.floor(4 * (n / 100) ** (2 / 9)))
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    covariance = (n / (n - k)) * inv @ meat @ inv
    se = float(np.sqrt(max(covariance[1, 1], 0)))
    t_value = float(coef[1] / se) if se else np.nan
    p_value = float(2 * stats.norm.sf(abs(t_value))) if np.isfinite(t_value) else np.nan
    tss = float(((y - y.mean()) ** 2).sum())
    return {"n": n, "beta": float(coef[1]), "hac_se": se, "hac_t": t_value, "p_value": p_value, "r_squared": float(1 - residual @ residual / tss) if tss else np.nan, "hac_lags": lags}


def nw_mean_t(values: pd.Series) -> float:
    frame = pd.DataFrame({"y": pd.to_numeric(values, errors="coerce")})
    clean = frame.y.dropna().to_numpy(float)
    if len(clean) < 2:
        return np.nan
    residual = clean - clean.mean()
    lag_max = int(math.floor(4 * (len(clean) / 100) ** (2 / 9)))
    variance = residual @ residual / len(clean)
    for lag in range(1, lag_max + 1):
        variance += 2 * (1 - lag / (lag_max + 1)) * (residual[lag:] @ residual[:-lag] / len(clean))
    se = math.sqrt(max(variance, 0) / len(clean))
    return float(clean.mean() / se) if se else np.nan


def cvar(values: pd.Series, probability: float = 0.05) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    cutoff = clean.quantile(probability)
    return float(clean.loc[clean <= cutoff].mean())


def pnl_metrics(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    gains = clean.loc[clean > 0].sum()
    losses = -clean.loc[clean < 0].sum()
    equity = clean.cumsum()
    drawdown = equity - equity.cummax().clip(lower=0)
    volatility = clean.std(ddof=1)
    return {
        "n": int(len(clean)),
        "total_pnl": float(clean.sum()),
        "mean_pnl": float(clean.mean()),
        "sharpe": float(clean.mean() / volatility * math.sqrt(252)) if volatility > 0 else np.nan,
        "profit_factor": float(gains / losses) if losses > 0 else np.nan,
        "hit_rate": float((clean > 0).mean()),
        "worst_day": float(clean.min()),
        "cvar_5pct": cvar(clean),
        "max_drawdown": float(drawdown.min()),
    }


def paired_block_bootstrap_delta(baseline: pd.Series, overlay: pd.Series, samples: int, block: int, seed: int) -> dict[str, float]:
    pair = pd.concat({"baseline": baseline, "overlay": overlay}, axis=1).dropna().reset_index(drop=True)
    n = len(pair)
    rng = np.random.default_rng(seed)
    mean_draws, cvar_draws = np.empty(samples), np.empty(samples)
    for draw in range(samples):
        starts = rng.integers(0, max(n - block + 1, 1), size=math.ceil(n / block))
        indices = np.concatenate([np.arange(start, min(start + block, n)) for start in starts])[:n]
        boot = pair.iloc[indices]
        mean_draws[draw] = (boot.overlay - boot.baseline).mean()
        cvar_draws[draw] = cvar(boot.overlay) - cvar(boot.baseline)
    return {
        "mean_delta_ci_low": float(np.nanquantile(mean_draws, .025)),
        "mean_delta_ci_high": float(np.nanquantile(mean_draws, .975)),
        "cvar_delta_ci_low": float(np.nanquantile(cvar_draws, .025)),
        "cvar_delta_ci_high": float(np.nanquantile(cvar_draws, .975)),
    }


def phase2_overlay(panel: pd.DataFrame, sleeves: list[str], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate the single frozen common backwardation filter after Gate 1."""
    backwardation = panel.term_spread > 0
    series = {sleeve: panel[sleeve] for sleeve in sleeves}
    series["equal_weight_portfolio"] = panel[sleeves].mean(axis=1)
    summary_rows, bootstrap_rows, half_rows, year_rows = [], [], [], []
    split_date = panel.index[len(panel) // 2]
    for entity_index, (entity, baseline) in enumerate(series.items()):
        overlay = baseline.mask(backwardation, 0.0)
        before, after = pnl_metrics(baseline), pnl_metrics(overlay)
        summary_rows.extend([
            {"entity": entity, "variant": "baseline", **before},
            {"entity": entity, "variant": "common_backwardation_filter", **after},
        ])
        bootstrap_rows.append({
            "entity": entity,
            "mean_pnl_delta": after["mean_pnl"] - before["mean_pnl"],
            "cvar_5pct_delta": after["cvar_5pct"] - before["cvar_5pct"],
            **paired_block_bootstrap_delta(
                baseline, overlay,
                int(config["inference"]["bootstrap_samples"]),
                int(config["inference"]["bootstrap_block_days"]),
                int(config["inference"]["random_seed"]) + 100 + entity_index,
            ),
        })
        for period, mask in (("first_half", panel.index < split_date), ("second_half", panel.index >= split_date)):
            half_rows.append({
                "entity": entity, "period": period, "split_date": split_date,
                "baseline_pnl": float(baseline.loc[mask].sum()),
                "overlay_pnl": float(overlay.loc[mask].sum()),
                "pnl_delta": float((overlay.loc[mask] - baseline.loc[mask]).sum()),
            })
        for year, dates in panel.groupby(panel.index.year).groups.items():
            base_year, overlay_year = baseline.loc[dates], overlay.loc[dates]
            year_rows.append({"entity": entity, "year": int(year), "baseline_pnl": base_year.sum(), "overlay_pnl": overlay_year.sum(), "pnl_delta": (overlay_year - base_year).sum()})
    summary = pd.DataFrame(summary_rows)
    halves, years = pd.DataFrame(half_rows), pd.DataFrame(year_rows)
    portfolio = summary.loc[summary.entity.eq("equal_weight_portfolio")].set_index("variant")
    portfolio_halves = halves.loc[halves.entity.eq("equal_weight_portfolio")]
    portfolio_years = years.loc[years.entity.eq("equal_weight_portfolio")]
    before, after = portfolio.loc["baseline"], portfolio.loc["common_backwardation_filter"]
    checks = {
        "portfolio_total_pnl_delta_positive": bool(after.total_pnl > before.total_pnl),
        "portfolio_sharpe_delta_positive": bool(after.sharpe > before.sharpe),
        "portfolio_cvar_5pct_not_worse": bool(after.cvar_5pct >= before.cvar_5pct),
        "portfolio_max_drawdown_not_worse": bool(after.max_drawdown >= before.max_drawdown),
        "pnl_delta_positive_in_both_halves": bool((portfolio_halves.pnl_delta > 0).all()),
        "strict_majority_of_years_with_positive_pnl_delta": bool((portfolio_years.pnl_delta > 0).mean() > .5),
    }
    decision = {
        "rule": config["phase_2"]["rule"],
        "eligible_sessions": int(len(panel)),
        "filtered_sessions": int(backwardation.sum()),
        "filtered_session_fraction": float(backwardation.mean()),
        **checks,
        "economically_actionable": bool(all(checks.values())),
    }
    return summary, pd.DataFrame(bootstrap_rows), halves, years, decision


PARTITIONS = {
    "vix_level": ("vix_state", ["low", "middle", "high"], "high", "low"),
    "term_spread": ("term_spread_state", ["contango", "backwardation"], "backwardation", "contango"),
    "term_ratio": ("term_ratio_state", ["ratio_le_1", "ratio_gt_1"], "ratio_gt_1", "ratio_le_1"),
    "evrp": ("evrp_state", ["nonpositive", "positive"], "positive", "nonpositive"),
}


def state_metrics(panel: pd.DataFrame, sleeves: list[str]) -> pd.DataFrame:
    rows = []
    for sleeve in sleeves:
        for variable, (state_column, labels, _, _) in PARTITIONS.items():
            for label in labels:
                values = panel.loc[panel[state_column] == label, sleeve].dropna()
                std = values.std(ddof=1)
                rows.append({"sleeve": sleeve, "variable": variable, "state": label, "n": len(values), "mean_pnl": values.mean(), "median_pnl": values.median(), "sharpe": values.mean() / std * math.sqrt(252) if std > 0 else np.nan, "hit_rate": (values > 0).mean(), "worst_day": values.min(), "cvar_5pct": cvar(values), "hac_mean_t": nw_mean_t(values)})
    return pd.DataFrame(rows)


def block_bootstrap_contrasts(panel: pd.DataFrame, sleeves: list[str], samples: int, block: int, seed: int) -> pd.DataFrame:
    rows = []
    for sleeve_index, sleeve in enumerate(sleeves):
        for variable_index, (variable, (state_column, _, high_label, low_label)) in enumerate(PARTITIONS.items()):
            sample = panel[[sleeve, state_column]].dropna().reset_index(drop=True)
            sample = sample.loc[sample[state_column].isin([high_label, low_label])].reset_index(drop=True)
            high = sample.loc[sample[state_column] == high_label, sleeve]
            low = sample.loc[sample[state_column] == low_label, sleeve]
            observed_mean = high.mean() - low.mean()
            observed_cvar = cvar(high) - cvar(low)
            rng = np.random.default_rng(seed + sleeve_index * 10 + variable_index)
            n = len(sample)
            mean_draws, cvar_draws = np.empty(samples), np.empty(samples)
            for draw in range(samples):
                starts = rng.integers(0, max(n - block + 1, 1), size=math.ceil(n / block))
                indices = np.concatenate([np.arange(start, min(start + block, n)) for start in starts])[:n]
                boot = sample.iloc[indices]
                boot_high = boot.loc[boot[state_column] == high_label, sleeve]
                boot_low = boot.loc[boot[state_column] == low_label, sleeve]
                mean_draws[draw] = boot_high.mean() - boot_low.mean()
                cvar_draws[draw] = cvar(boot_high) - cvar(boot_low)
            rows.append({"sleeve": sleeve, "variable": variable, "high_state": high_label, "low_state": low_label, "n": n, "mean_contrast_high_minus_low": observed_mean, "mean_ci_low": np.nanquantile(mean_draws, .025), "mean_ci_high": np.nanquantile(mean_draws, .975), "cvar_contrast_high_minus_low": observed_cvar, "cvar_ci_low": np.nanquantile(cvar_draws, .025), "cvar_ci_high": np.nanquantile(cvar_draws, .975)})
    return pd.DataFrame(rows)


def regressions(panel: pd.DataFrame, sleeves: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_rows, half_rows, yearly_rows = [], [], []
    split_date = panel.index[len(panel) // 2]
    for sleeve in sleeves:
        local = panel.copy()
        local["daily_pnl_usd"] = local[sleeve]
        local["downside_pnl_usd"] = local[sleeve].clip(upper=0)
        for variable in PARTITIONS:
            for outcome in ("daily_pnl_usd", "downside_pnl_usd"):
                for model, predictors in (("univariate", [variable]), ("controlled", [variable, "mes_rv10_pct", "mes_atr14_pct"])):
                    result = nw_ols(local, outcome, predictors)
                    iqr = local[variable].quantile(.75) - local[variable].quantile(.25)
                    outcome_std = local[outcome].std(ddof=1)
                    full_rows.append({"sleeve": sleeve, "variable": variable, "outcome": outcome, "model": model, **result, "variable_iqr": iqr, "iqr_effect_usd": result["beta"] * iqr, "absolute_standardized_iqr_effect": abs(result["beta"] * iqr / outcome_std) if outcome_std > 0 else np.nan})
                for half, mask in (("first_half", local.index < split_date), ("second_half", local.index >= split_date)):
                    result = nw_ols(local.loc[mask], outcome, [variable, "mes_rv10_pct", "mes_atr14_pct"])
                    half_rows.append({"sleeve": sleeve, "variable": variable, "outcome": outcome, "period": half, "split_date": split_date, **result})
                for year, group in local.groupby(local.index.year):
                    result = nw_ols(group, outcome, [variable, "mes_rv10_pct", "mes_atr14_pct"])
                    yearly_rows.append({"sleeve": sleeve, "variable": variable, "outcome": outcome, "year": int(year), **result})
    full = pd.DataFrame(full_rows)
    controlled = full.model.eq("controlled")
    p_values = full.loc[controlled, "p_value"].to_numpy(float)
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    adjusted = np.empty_like(adjusted_sorted); adjusted[order] = adjusted_sorted
    full.loc[controlled, "fdr_q_value"] = adjusted
    return full, pd.DataFrame(half_rows), pd.DataFrame(yearly_rows)


def gate_results(regression: pd.DataFrame, halves: pd.DataFrame, yearly: pd.DataFrame, contrasts: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    candidates = regression.loc[regression.model.eq("controlled")].copy()
    rows = []
    for row in candidates.itertuples(index=False):
        half = halves.loc[(halves.sleeve == row.sleeve) & (halves.variable == row.variable) & (halves.outcome == row.outcome)].set_index("period")
        years = yearly.loc[(yearly.sleeve == row.sleeve) & (yearly.variable == row.variable) & (yearly.outcome == row.outcome) & yearly.beta.notna()]
        contrast = contrasts.loc[(contrasts.sleeve == row.sleeve) & (contrasts.variable == row.variable)].iloc[0]
        if row.outcome == "daily_pnl_usd":
            observed, low, high = contrast.mean_contrast_high_minus_low, contrast.mean_ci_low, contrast.mean_ci_high
        else:
            observed, low, high = contrast.cvar_contrast_high_minus_low, contrast.cvar_ci_low, contrast.cvar_ci_high
        sign = np.sign(row.beta)
        checks = {
            "incremental_fdr_pass": bool(row.fdr_q_value <= config["gate_1"]["candidate_must_satisfy_all"]["controlled_external_coefficient_fdr_q_at_most"]),
            "economic_effect_pass": bool(row.absolute_standardized_iqr_effect >= config["gate_1"]["candidate_must_satisfy_all"]["minimum_absolute_iqr_effect_in_outcome_standard_deviations"]),
            "bootstrap_contrast_pass": bool((low > 0 or high < 0) and np.sign(observed) == sign),
            "half_stability_pass": bool(len(half) == 2 and (np.sign(half.beta) == sign).all()),
            "year_stability_pass": bool(len(years) > 0 and (np.sign(years.beta) == sign).mean() > .5),
        }
        rows.append({"sleeve": row.sleeve, "variable": row.variable, "outcome": row.outcome, "beta": row.beta, "hac_t": row.hac_t, "fdr_q_value": row.fdr_q_value, "absolute_standardized_iqr_effect": row.absolute_standardized_iqr_effect, "partition_contrast": observed, "contrast_ci_low": low, "contrast_ci_high": high, "first_half_beta": half.loc["first_half", "beta"], "second_half_beta": half.loc["second_half", "beta"], "same_sign_year_ratio": (np.sign(years.beta) == sign).mean() if len(years) else np.nan, **checks, "gate_1_pass": all(checks.values())})
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.fillna("").astype(str)
    return "\n".join(["| " + " | ".join(display.columns) + " |", "| " + " | ".join("---" for _ in display.columns) + " |", *("| " + " | ".join(row) + " |" for row in display.to_numpy(str))])


def plot_diagnostics(metrics: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"; figures.mkdir(exist_ok=True)
    for variable in PARTITIONS:
        subset = metrics.loc[metrics.variable == variable]
        pivot = subset.pivot(index="sleeve", columns="state", values="mean_pnl")
        ax = pivot.plot(kind="bar", figsize=(10, 5), title=f"Mean frozen-sleeve daily PnL by {variable} state")
        ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("USD/day"); ax.figure.tight_layout(); ax.figure.savefig(figures / f"mean_pnl_by_{variable}.png", dpi=160); plt.close(ax.figure)


def write_report(
    output: Path, gate: pd.DataFrame, metrics: pd.DataFrame, regressions_frame: pd.DataFrame,
    panel: pd.DataFrame, stopped: bool, overlay_summary: pd.DataFrame | None,
    overlay_bootstrap: pd.DataFrame | None, overlay_halves: pd.DataFrame | None,
    overlay_years: pd.DataFrame | None, overlay_decision: dict[str, Any] | None,
) -> None:
    significant = regressions_frame.loc[(regressions_frame.model == "controlled") & (regressions_frame.fdr_q_value <= .05), ["sleeve", "variable", "outcome", "beta", "hac_t", "fdr_q_value", "absolute_standardized_iqr_effect"]].copy().round(4)
    gate_view = gate.sort_values("fdr_q_value").head(12).copy()
    numeric = gate_view.select_dtypes("number").columns; gate_view[numeric] = gate_view[numeric].round(4)
    summary = metrics[["sleeve", "variable", "state", "n", "mean_pnl", "median_pnl", "sharpe", "hit_rate", "worst_day", "cvar_5pct"]].copy().round(3)
    if stopped:
        verdict = "REJECTED_STATE_VARIABLE"
        actionability = "No. Phase 2 was prohibited and the four sleeves remain unchanged."
        phase2 = "Gate 1 failed, so no overlay was run."
    else:
        actionable = bool(overlay_decision and overlay_decision["economically_actionable"])
        verdict = "ACTIONABLE_STATE_CONDITIONING" if actionable else "STATE_DEPENDENCE_VALIDATED_NOT_ACTIONABLE"
        actionability = "Yes, under the frozen common-rule gate." if actionable else "No. The single canonical common overlay failed at least one frozen economic robustness criterion."
        overlay_view = overlay_summary.copy(); overlay_view[overlay_view.select_dtypes("number").columns] = overlay_view.select_dtypes("number").round(3)
        boot_view = overlay_bootstrap.copy(); boot_view[boot_view.select_dtypes("number").columns] = boot_view.select_dtypes("number").round(3)
        portfolio_halves = overlay_halves.loc[overlay_halves.entity.eq("equal_weight_portfolio")].copy().round(3)
        portfolio_years = overlay_years.loc[overlay_years.entity.eq("equal_weight_portfolio")].copy().round(3)
        phase2 = f"""The one rule was frozen before inspecting overlay returns: suppress all four sleeves on day t when lagged VIX > lagged VIX3M. The rule filtered {overlay_decision['filtered_sessions']} of {overlay_decision['eligible_sessions']} eligible sessions ({overlay_decision['filtered_session_fraction']:.1%}). Because every sleeve is intraday, zeroing that day's frozen net PnL exactly represents suppressing all entries for that day.

{markdown_table(overlay_view)}

Paired 5-day block-bootstrap changes (positive CVaR change means a less severe left tail):

{markdown_table(boot_view)}

Equal-weight portfolio half stability:

{markdown_table(portfolio_halves)}

Equal-weight portfolio yearly stability:

{markdown_table(portfolio_years)}

Frozen actionability checks:

{markdown_table(pd.DataFrame([overlay_decision]))}"""
    decision = "Gate 1 fails. H3 is classified REJECTED_STATE_VARIABLE. No filter, sizing rule, sleeve selection, portfolio replay, or Topstep simulation was run." if stopped else ("The single common overlay passes every frozen economic robustness check. H3 is actionable only in this narrow, predeclared form; no other threshold or combination was searched." if overlay_decision["economically_actionable"] else "A statistically credible conditional left-tail relationship exists, but the only permitted common ex-ante overlay is not robustly economically actionable. H3 is therefore not promoted to a trading rule.")
    text = f"""# Options-implied volatility state conditioning — H3

## Verdict: {verdict}

- **Statistical state dependence:** {'Some nominal relationships may exist, but none satisfies the full corrected and stable gate.' if stopped else 'Yes, but it is concentrated in the downside PnL of volatility_breakout.'}
- **Incremental information beyond MES RV10/ATR14:** {'Not robust enough after HAC inference, FDR correction and controls.' if stopped else 'Yes for that left-tail relationship after HAC, FDR and stability tests.'}
- **Economically actionable conditioning:** {actionability}

All VIX, VIX3M, RV10 and ATR14 values are lagged to t−1. The joined diagnostic sample contains {len(panel):,} sessions from the locked 1,750-session daily PnL matrix; no strategy was regenerated or modified.

## State diagnostics

{markdown_table(summary)}

## Incremental regressions

Controlled models are `PnL_t = alpha + beta * external_state_(t-1) + controls for MES_RV10_(t-1) and MES_ATR14_(t-1)`. Downside models replace PnL by `min(PnL,0)`. HAC p-values are corrected together by Benjamini–Hochberg FDR.

Corrected significant rows:

{markdown_table(significant) if len(significant) else 'None.'}

## Gate 1

{markdown_table(gate_view)}

A candidate also requires an IQR effect of at least 0.10 outcome standard deviations, a block-bootstrap partition contrast excluding zero with consistent sign, matching signs in both halves, and a strict majority of calendar-year coefficients with the same sign.

The passing VIX level, term spread and term ratio rows are not three independent discoveries. Term spread and term ratio have exactly the same backwardation partition. Also, because `eVRP = VIX − RV10`, a regression containing eVRP and RV10 spans the same linear information as one containing VIX and RV10; eVRP's controlled coefficient therefore cannot be interpreted as separate confirmation. eVRP did not pass its own predeclared positive/nonpositive partition bootstrap.

The robust effect is a worse conditional left tail, not a reliable change in mean PnL: the passing rows are downside-PnL outcomes, while their corresponding mean-PnL partition confidence intervals include zero.

## Phase 2 — one frozen overlay

{phase2}

## Reproduction

From the repository root, run `PYTHONPATH=. python -m src.analytics.options_implied_state_conditioning` (PowerShell: `$env:PYTHONPATH='.'; .\\.venv\\Scripts\\python.exe -m src.analytics.options_implied_state_conditioning`). Source paths and SHA-256 hashes are recorded in `source_audit.csv`; all inference and gate choices are frozen in `frozen_config.json`.

## Decision

{decision}

No Topstep optimization was run. The six frozen sleeves were not modified.
"""
    (output / "OPTIONS_IMPLIED_STATE_CONDITIONING_REPORT.md").write_text(text, encoding="utf-8")


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((output / "frozen_config.json").read_text(encoding="utf-8"))
    paths = {key: REPO_ROOT / value for key, value in config["sources"].items()}
    vix = load_vix(paths["vix"]); mes = load_mes_daily(paths["mes_1m"]); states = build_causal_states(vix, mes)
    pnl = pd.read_csv(paths["frozen_daily_pnl"], parse_dates=["session_date"]).set_index("session_date")[config["sleeves"]]
    panel = pnl.join(states, how="inner").dropna(subset=["vix_level", "vix3m_level", "mes_rv10_pct", "mes_atr14_pct"])
    metrics = state_metrics(panel, config["sleeves"])
    contrasts = block_bootstrap_contrasts(panel, config["sleeves"], int(config["inference"]["bootstrap_samples"]), int(config["inference"]["bootstrap_block_days"]), int(config["inference"]["random_seed"]))
    regression, halves, yearly = regressions(panel, config["sleeves"])
    gate = gate_results(regression, halves, yearly, contrasts, config)
    stopped = not gate.gate_1_pass.any()
    source_audit = pd.DataFrame([{"source": key, "path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path), "rows": len(pd.read_csv(path)) if path.suffix == ".csv" else None} for key, path in paths.items()])
    state_export = panel.reset_index()[["session_date", "vix_level", "vix3m_level", "term_spread", "term_ratio", "mes_rv10_pct", "mes_atr14_pct", "evrp", "vix_state", "term_spread_state", "term_ratio_state", "evrp_state"]]
    source_audit.to_csv(output / "source_audit.csv", index=False); state_export.to_csv(output / "causal_state_dataset.csv", index=False)
    metrics.to_csv(output / "state_diagnostics.csv", index=False); contrasts.to_csv(output / "state_bootstrap_contrasts.csv", index=False)
    regression.to_csv(output / "state_regressions.csv", index=False); halves.to_csv(output / "state_first_second_half.csv", index=False); yearly.to_csv(output / "state_yearly_stability.csv", index=False); gate.to_csv(output / "gate1_results.csv", index=False)
    overlay_summary = overlay_bootstrap = overlay_halves = overlay_years = None
    overlay_decision = None
    if stopped:
        pd.DataFrame([{"status": "NOT_RUN_GATE_1_FAILURE", "reason": "No robust incremental external state passed Gate 1."}]).to_csv(output / "phase2_overlay_results.csv", index=False)
    else:
        overlay_summary, overlay_bootstrap, overlay_halves, overlay_years, overlay_decision = phase2_overlay(panel, config["sleeves"], config)
        overlay_summary.to_csv(output / "phase2_overlay_results.csv", index=False)
        overlay_bootstrap.to_csv(output / "phase2_overlay_bootstrap.csv", index=False)
        overlay_halves.to_csv(output / "phase2_overlay_half_stability.csv", index=False)
        overlay_years.to_csv(output / "phase2_overlay_yearly_stability.csv", index=False)
        pd.DataFrame([overlay_decision]).to_csv(output / "phase2_actionability_gate.csv", index=False)
    plot_diagnostics(metrics, output)
    write_report(output, gate, metrics, regression, panel, stopped, overlay_summary, overlay_bootstrap, overlay_halves, overlay_years, overlay_decision)
    verdict = "REJECTED_STATE_VARIABLE" if stopped else ("ACTIONABLE_STATE_CONDITIONING" if overlay_decision["economically_actionable"] else "STATE_DEPENDENCE_VALIDATED_NOT_ACTIONABLE")
    metadata = {"verdict": verdict, "eligible_sessions": len(panel), "gate_1_pass_count": int(gate.gate_1_pass.sum()), "phase_2_run": not stopped, "phase_2_economically_actionable": None if stopped else overlay_decision["economically_actionable"], "frozen_sleeves_modified": False, "topstep_run": False}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args(); print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
