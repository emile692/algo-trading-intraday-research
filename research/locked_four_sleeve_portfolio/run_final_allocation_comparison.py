"""Compare four allocation rules on the locked MNQ sleeves and freeze one rule."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from research.locked_four_sleeve_portfolio.run_erc_portfolio import (
    INITIAL_CAPITAL,
    SLEEVES,
    portfolio_metrics,
    risk_details,
    solve_erc,
)


OUTPUT_DIR = Path(__file__).resolve().parent
RETURNS_PATH = OUTPUT_DIR / "erc_daily_returns.csv"
PNL_PATH = OUTPUT_DIR / "daily_pnl.csv"
CONFIG_PATH = OUTPUT_DIR / "locked_four_sleeve_config.json"
CONFIG_HASH_PATH = OUTPUT_DIR / "locked_four_sleeve_config.sha256"
TRADES_DIR = OUTPUT_DIR / "sleeve_trades"
WINDOW = 63
LOWER_BOUND = 0.10
UPPER_BOUND = 0.40
ANNUALIZATION = 252
ALLOCATION_NAMES = ["equal_weight", "erc_rolling", "constrained_erc", "activity_adjusted_erc"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_bounded_simplex(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if len(values) * lower > 1 or len(values) * upper < 1:
        raise ValueError("Infeasible bounded simplex")
    low = float(np.min(values - upper))
    high = float(np.max(values - lower))
    for _ in range(100):
        midpoint = (low + high) / 2.0
        projected = np.clip(values - midpoint, lower, upper)
        if projected.sum() > 1.0:
            low = midpoint
        else:
            high = midpoint
    projected = np.clip(values - (low + high) / 2.0, lower, upper)
    projected /= projected.sum()
    return projected


def risk_budget_objective_gradient(weights: np.ndarray, covariance: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    n = len(weights)
    marginal_variance = covariance @ weights
    components = weights * marginal_variance
    variance = float(components.sum())
    shares = components / variance
    target = np.full(n, 1.0 / n)
    residual = shares - target
    dc = np.diag(marginal_variance) + np.diag(weights) @ covariance
    dv = 2.0 * marginal_variance
    jacobian = (dc * variance - components[:, None] * dv[None, :]) / (variance * variance)
    gradient = 2.0 * jacobian.T @ residual
    return float(residual @ residual), gradient, shares


def solve_constrained_erc(covariance: np.ndarray) -> tuple[np.ndarray, int, float]:
    unconstrained, _, _ = solve_erc(covariance)
    start = project_bounded_simplex(unconstrained, LOWER_BOUND, UPPER_BOUND)

    def objective(weights: np.ndarray) -> float:
        return risk_budget_objective_gradient(weights, covariance)[0]

    def gradient(weights: np.ndarray) -> np.ndarray:
        return risk_budget_objective_gradient(weights, covariance)[1]

    result = minimize(
        objective,
        start,
        jac=gradient,
        method="SLSQP",
        bounds=[(LOWER_BOUND, UPPER_BOUND)] * len(SLEEVES),
        constraints=[{"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0), "jac": lambda weights: np.ones_like(weights)}],
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Constrained ERC solver failed: {result.message}")
    weights = np.asarray(result.x, dtype=float)
    _, _, shares = risk_budget_objective_gradient(weights, covariance)
    return weights, int(result.nit), float(np.max(np.abs(shares - 0.25)))


def activity_adjusted_covariance(history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    standard_vol = history.std(ddof=1).to_numpy(dtype=float)
    activity = history.ne(0.0).mean().to_numpy(dtype=float)
    conditional_vol = np.empty(len(SLEEVES), dtype=float)
    for i, sleeve in enumerate(SLEEVES):
        active = history.loc[history[sleeve].ne(0.0), sleeve]
        if active.empty:
            raise RuntimeError(f"No active observation for {sleeve}")
        conditional_vol[i] = float(np.sqrt(np.mean(np.square(active.to_numpy(dtype=float)))))
    correlation = history.corr().to_numpy(dtype=float)
    adjusted = np.diag(conditional_vol) @ correlation @ np.diag(conditional_vol)
    return adjusted, standard_vol, conditional_vol, activity


def allocation_metrics(name: str, pnl: pd.Series) -> dict[str, Any]:
    row = portfolio_metrics(name, pnl)
    return row


def summarize_weights(weights: pd.DataFrame, turnover: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for allocation in ALLOCATION_NAMES:
        subset = weights.xs(allocation, level="allocation")
        for sleeve in SLEEVES:
            values = subset[sleeve]
            rows.append({
                "allocation": allocation,
                "sleeve": sleeve,
                "mean_weight": values.mean(),
                "min_weight": values.min(),
                "max_weight": values.max(),
                "p05_weight": values.quantile(0.05),
                "p95_weight": values.quantile(0.95),
                "days_above_50_pct": int(values.gt(0.50).sum()),
                "mean_daily_turnover": float(turnover[allocation].mean()),
                "median_daily_turnover": float(turnover[allocation].median()),
                "max_daily_turnover": float(turnover[allocation].max()),
            })
    return pd.DataFrame(rows)


def trade_diagnostics(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sleeve in SLEEVES:
        trades = pd.read_csv(TRADES_DIR / f"{sleeve}.csv")
        pnl = trades["net_pnl_usd"].astype(float)
        losses = pnl[pnl < 0]
        series = returns[sleeve]
        active = series[series.ne(0.0)]
        rows.append({
            "sleeve": sleeve,
            "standard_daily_volatility": float(series.std(ddof=1)),
            "conditional_active_day_volatility": float(np.sqrt(np.mean(np.square(active.to_numpy(dtype=float))))),
            "activity_rate": float(series.ne(0.0).mean()),
            "active_day_count": int(series.ne(0.0).sum()),
            "trade_count": len(trades),
            "mean_absolute_pnl_per_trade_usd": float(pnl.abs().mean()),
            "average_losing_trade_usd": float(losses.mean()),
        })
    return pd.DataFrame(rows)


def target_volatility_comparison(pnl_map: dict[str, pd.Series], metrics: pd.DataFrame) -> pd.DataFrame:
    target = float(metrics["annualized_volatility_pct"].min())
    rows = []
    for allocation in ALLOCATION_NAMES:
        realized = float(metrics.loc[metrics["portfolio"] == allocation, "annualized_volatility_pct"].iloc[0])
        if realized <= target + 1e-12:
            multiplier = 1.0
        else:
            low, high = 0.0, 1.0
            for _ in range(80):
                midpoint = (low + high) / 2.0
                midpoint_vol = allocation_metrics(allocation, pnl_map[allocation] * midpoint)["annualized_volatility_pct"]
                if midpoint_vol < target:
                    low = midpoint
                else:
                    high = midpoint
            multiplier = (low + high) / 2.0
        scaled = pnl_map[allocation] * multiplier
        row = allocation_metrics(allocation, scaled)
        row["target_annualized_volatility_pct"] = target
        row["capital_multiplier"] = multiplier
        row["cash_weight"] = 1.0 - multiplier
        rows.append(row)
    return pd.DataFrame(rows)


def choose_and_freeze(metrics: pd.DataFrame, summary: pd.DataFrame, target: pd.DataFrame, input_hash: str) -> dict[str, Any]:
    metric = metrics.set_index("portfolio")
    constrained_summary = summary[summary["allocation"] == "constrained_erc"]
    turnover_by_allocation = summary.groupby("allocation")["mean_daily_turnover"].first()
    qualifies = (
        constrained_summary["max_weight"].le(UPPER_BOUND + 1e-10).all()
        and metric.loc["constrained_erc", "max_drawdown_usd"] > metric.loc["equal_weight", "max_drawdown_usd"]
        and turnover_by_allocation["constrained_erc"] < turnover_by_allocation["erc_rolling"]
    )
    selected = "constrained_erc" if qualifies else "equal_weight"
    reason = (
        "Constrained ERC retenu par la règle de praticabilité préengagée : chaque sleeve reste entre 10 % et 40 %, le drawdown s'améliore face à l'equal weight et le turnover est inférieur à celui de l'ERC classique."
        if qualifies else
        "Equal weight retained because constrained ERC did not satisfy every precommitted practicality condition."
    )
    return {
        "schema_version": "1.0",
        "status": "FINAL_ALLOCATION_LOCK",
        "selected_allocation": selected,
        "selection_rule": {
            "concentration_control": "all constrained ERC weights must remain between 0.10 and 0.40",
            "drawdown": "constrained ERC max drawdown must improve versus equal weight",
            "operability": "constrained ERC turnover must be lower than classic rolling ERC turnover",
            "tie_break": "prefer the simpler constrained ERC over activity adjustment when conditions pass",
            "sharpe_maximized": False,
        },
        "reason": reason,
        "parameters": {
            "covariance_window_sessions": WINDOW,
            "covariance_lag_sessions": 1,
            "long_only": True,
            "fully_invested": True,
            "leverage": False,
            "minimum_weight": LOWER_BOUND if selected == "constrained_erc" else 0.25,
            "maximum_weight": UPPER_BOUND if selected == "constrained_erc" else 0.25,
            "objective": "equal component risk contributions under the stated bounds",
        },
        "source_erc_daily_returns_sha256": input_hash,
        "selected_metrics": metric.loc[selected].to_dict(),
        "selected_target_volatility_metrics": target.set_index("portfolio").loc[selected].to_dict(),
    }


def write_report(
    metrics: pd.DataFrame, summary: pd.DataFrame, diagnostics: pd.DataFrame,
    target: pd.DataFrame, risk: pd.DataFrame, lock: dict[str, Any], lock_hash: str,
) -> None:
    lines = [
        "# Comparatif final d'allocation — quatre sleeves MNQ", "", "## Protocole", "",
        "- Covariance rolling 63 séances, strictement arrêtée à `t-1`.",
        "- Poids longs, somme égale à 1, sans levier et sans optimisation du Sharpe.",
        "- Constrained ERC : poids individuels entre 10 % et 40 %.",
        "- Activity-adjusted ERC : risque RMS conditionnel aux jours actifs `sqrt(mean(r² | actif))`, combiné à la corrélation quotidienne standard.",
        "- Comparaison sur la fenêtre commune **2019-08-02 → 2026-03-19**.", "", "## Résultats", "",
        "| Allocation | PnL | Sharpe | PF journalier | Vol annualisée | Max DD | Turnover moyen |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        turnover = summary.loc[summary["allocation"] == row.portfolio, "mean_daily_turnover"].iloc[0]
        lines.append(f"| `{row.portfolio}` | {row.net_pnl_usd:,.2f} | {row.sharpe_annualized:.3f} | {row.daily_profit_factor:.3f} | {row.annualized_volatility_pct:.2f}% | {row.max_drawdown_usd:,.2f} | {turnover:.2%} |")
    lines += ["", "## Poids moyens, minimums et maximums", "",
              "| Allocation | Sleeve | Moyen | Min | Max |",
              "|---|---|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| `{row.allocation}` | `{row.sleeve}` | {row.mean_weight:.2%} | {row.min_weight:.2%} | {row.max_weight:.2%} |")
    average_risk = risk.groupby(["allocation", "sleeve"], as_index=False)["risk_contribution_pct"].mean()
    lines += ["", "## Contributions moyennes au risque", "",
              "Mesure commune utilisant la covariance quotidienne standard rolling 63 séances.", "",
              "| Allocation | Sleeve | Contribution moyenne |",
              "|---|---|---:|"]
    for row in average_risk.itertuples(index=False):
        lines.append(f"| `{row.allocation}` | `{row.sleeve}` | {row.risk_contribution_pct:.2f}% |")
    lines += ["", "## Diagnostic d'activité", "",
              "| Sleeve | Vol quotidienne | Vol jours actifs | Activité | Trades | |PnL| moyen/trade | Perte moyenne |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in diagnostics.itertuples(index=False):
        lines.append(f"| `{row.sleeve}` | {row.standard_daily_volatility:.5%} | {row.conditional_active_day_volatility:.5%} | {row.activity_rate:.1%} | {row.trade_count} | {row.mean_absolute_pnl_per_trade_usd:,.2f} | {row.average_losing_trade_usd:,.2f} |")
    lines += ["", "## Comparaison à volatilité cible identique", "",
              "La cible est la plus faible volatilité réalisée des quatre allocations. Les autres portefeuilles sont réduits avec une poche cash, sans levier.", "",
              "| Allocation | Multiplicateur | Cash | PnL ajusté | Sharpe | Max DD |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in target.itertuples(index=False):
        lines.append(f"| `{row.portfolio}` | {row.capital_multiplier:.3f} | {row.cash_weight:.1%} | {row.net_pnl_usd:,.2f} | {row.sharpe_annualized:.3f} | {row.max_drawdown_usd:,.2f} |")
    lines += ["", "## Décision finale", "", f"Allocation verrouillée : **`{lock['selected_allocation']}`**.", "", lock["reason"], "", f"SHA-256 de la règle finale : `{lock_hash}`.", ""]
    (OUTPUT_DIR / "FINAL_ALLOCATION_COMPARISON_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def plot_locked_constrained_erc(portfolio: pd.DataFrame, metrics: pd.DataFrame) -> None:
    equity = portfolio["constrained_erc__equity_usd"]
    peak = equity.cummax().clip(lower=INITIAL_CAPITAL)
    drawdown = equity - peak
    row = metrics.set_index("portfolio").loc["constrained_erc"]
    dates = pd.to_datetime(portfolio.index)
    fig, (ax_equity, ax_drawdown) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_equity.plot(dates, equity, color="#2457a6", linewidth=2.0, label="Constrained ERC 63d")
    ax_equity.axhline(INITIAL_CAPITAL, color="#777777", linewidth=0.8)
    ax_equity.set_title("MNQ — portefeuille verrouillé Constrained ERC 63d")
    ax_equity.set_ylabel("Equity nette (USD)")
    ax_equity.grid(alpha=0.25)
    ax_equity.legend(loc="upper left")
    annotation = (
        f"PnL: ${row.net_pnl_usd:,.0f}   Sharpe: {row.sharpe_annualized:.2f}   "
        f"PF journalier: {row.daily_profit_factor:.2f}\n"
        f"Vol annualisée: {row.annualized_volatility_pct:.2f}%   "
        f"Max DD: ${row.max_drawdown_usd:,.0f} ({row.max_drawdown_pct:.2f}%)"
    )
    ax_equity.text(
        0.015, 0.95, annotation, transform=ax_equity.transAxes, va="top", ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "#bbbbbb"},
    )
    ax_drawdown.fill_between(dates, drawdown.to_numpy(), 0, color="#b33a3a", alpha=0.65)
    ax_drawdown.set_ylabel("Drawdown (USD)")
    ax_drawdown.set_xlabel("Date")
    ax_drawdown.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "constrained_erc_63d_equity_curve.png", dpi=180)
    plt.close(fig)


def execute() -> None:
    input_hash = sha256(RETURNS_PATH)
    config_hash = sha256(CONFIG_PATH)
    if config_hash != CONFIG_HASH_PATH.read_text(encoding="ascii").split()[0]:
        raise RuntimeError("Locked sleeve configuration hash mismatch")
    returns = pd.read_csv(RETURNS_PATH, parse_dates=["session_date"]).set_index("session_date")[SLEEVES]
    pnl = pd.read_csv(PNL_PATH, parse_dates=["session_date"]).set_index("session_date")[SLEEVES]
    if not returns.index.equals(pnl.index) or not np.allclose(returns, pnl / INITIAL_CAPITAL):
        raise RuntimeError("Locked PnL and ERC return inputs disagree")

    weights_by_name = {name: [] for name in ALLOCATION_NAMES}
    turnover_by_name = {name: [] for name in ALLOCATION_NAMES}
    previous = {name: np.full(4, 0.25) for name in ALLOCATION_NAMES}
    risk_rows, activity_rows, dates, window_starts, window_ends = [], [], [], [], []
    for position in range(WINDOW, len(returns)):
        date = returns.index[position]
        history = returns.iloc[position - WINDOW:position]
        standard_cov = history.cov(ddof=1).to_numpy(dtype=float)
        adjusted_cov, standard_vol, active_vol, activity = activity_adjusted_covariance(history)
        classic, _, _ = solve_erc(standard_cov)
        constrained, constrained_iterations, constrained_error = solve_constrained_erc(standard_cov)
        adjusted, _, _ = solve_erc(adjusted_cov)
        allocations = {
            "equal_weight": np.full(4, 0.25),
            "erc_rolling": classic,
            "constrained_erc": constrained,
            "activity_adjusted_erc": adjusted,
        }
        dates.append(date)
        window_starts.append(history.index[0])
        window_ends.append(history.index[-1])
        for name, weights in allocations.items():
            weights_by_name[name].append(weights)
            turnover_by_name[name].append(0.5 * float(np.abs(weights - previous[name]).sum()))
            _, marginal, component, shares = risk_details(weights, standard_cov)
            for i, sleeve in enumerate(SLEEVES):
                risk_rows.append({
                    "session_date": date, "allocation": name, "sleeve": sleeve,
                    "weight": weights[i], "marginal_risk_contribution": marginal[i],
                    "component_risk_contribution": component[i], "risk_contribution_pct": shares[i] * 100.0,
                    "risk_evaluation_covariance": "standard_63d_t_minus_1",
                })
            previous[name] = weights
        for i, sleeve in enumerate(SLEEVES):
            activity_rows.append({
                "session_date": date, "sleeve": sleeve,
                "standard_daily_volatility_63d": standard_vol[i],
                "conditional_active_day_volatility_63d": active_vol[i],
                "activity_rate_63d": activity[i],
                "activity_adjustment_multiple": active_vol[i] / standard_vol[i],
                "constrained_solver_iterations": constrained_iterations,
                "constrained_max_risk_budget_error": constrained_error,
            })

    index = pd.Index(dates, name="session_date")
    weight_frames = {name: pd.DataFrame(values, index=index, columns=SLEEVES) for name, values in weights_by_name.items()}
    turnover = pd.DataFrame(turnover_by_name, index=index)
    common_pnl = pnl.reindex(index)
    pnl_map = {name: (common_pnl * frame).sum(axis=1) for name, frame in weight_frames.items()}
    metrics = pd.DataFrame([allocation_metrics(name, pnl_map[name]) for name in ALLOCATION_NAMES])
    stacked_weights = pd.concat(weight_frames, names=["allocation", "session_date"])
    summary = summarize_weights(stacked_weights, turnover)
    diagnostics = trade_diagnostics(returns)
    target = target_volatility_comparison(pnl_map, metrics)

    portfolio = pd.DataFrame(index=index)
    for name in ALLOCATION_NAMES:
        portfolio[f"{name}__daily_pnl_usd"] = pnl_map[name]
        portfolio[f"{name}__equity_usd"] = INITIAL_CAPITAL + pnl_map[name].cumsum()
        portfolio[f"{name}__turnover"] = turnover[name]
    portfolio.to_csv(OUTPUT_DIR / "allocation_comparison_portfolio.csv")
    weight_output = stacked_weights.reset_index()
    window_lookup = pd.DataFrame({"session_date": index, "covariance_window_start": window_starts, "covariance_window_end": window_ends})
    weight_output = weight_output.merge(window_lookup, on="session_date", how="left", validate="many_to_one")
    weight_output.to_csv(OUTPUT_DIR / "allocation_weights.csv", index=False)
    risk_frame = pd.DataFrame(risk_rows)
    risk_frame.to_csv(OUTPUT_DIR / "allocation_risk_contributions.csv", index=False)
    pd.DataFrame(activity_rows).to_csv(OUTPUT_DIR / "activity_adjustment_rolling_diagnostics.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / "activity_risk_diagnostics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "allocation_weight_summary.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "allocation_comparison_metrics.csv", index=False)
    target.to_csv(OUTPUT_DIR / "allocation_target_volatility_comparison.csv", index=False)

    lock = choose_and_freeze(metrics, summary, target, input_hash)
    lock_path = OUTPUT_DIR / "final_allocation_rule.json"
    lock_path.write_text(json.dumps(lock, indent=2, default=str) + "\n", encoding="utf-8")
    lock_hash = sha256(lock_path)
    (OUTPUT_DIR / "final_allocation_rule.sha256").write_text(f"{lock_hash}  {lock_path.name}\n", encoding="ascii")
    write_report(metrics, summary, diagnostics, target, risk_frame, lock, lock_hash)
    plot_locked_constrained_erc(portfolio, metrics)

    for name, frame in weight_frames.items():
        if not np.allclose(frame.sum(axis=1), 1.0, atol=1e-10) or frame.lt(-1e-12).any().any():
            raise RuntimeError(f"Invalid long-only weights for {name}")
    constrained = weight_frames["constrained_erc"]
    if constrained.lt(LOWER_BOUND - 1e-10).any().any() or constrained.gt(UPPER_BOUND + 1e-10).any().any():
        raise RuntimeError("Constrained ERC bounds violated")
    if not (weight_output["covariance_window_end"] < weight_output["session_date"]).all():
        raise RuntimeError("Look-ahead detected in allocation covariance windows")
    if float(target["annualized_volatility_pct"].max() - target["annualized_volatility_pct"].min()) > 1e-8:
        raise RuntimeError("Target-volatility comparison is not aligned")
    metadata = {
        "erc_daily_returns_sha256": input_hash,
        "locked_sleeve_config_sha256": config_hash,
        "window_sessions": WINDOW,
        "shift_sessions": 1,
        "first_allocation_date": str(index[0].date()),
        "last_allocation_date": str(index[-1].date()),
        "observation_count": len(index),
        "sharpe_optimized": False,
        "long_only": True,
        "fully_invested": True,
        "leverage": False,
        "activity_adjustment": "conditional active-day RMS risk sqrt(mean(return^2 | active)) with standard full-daily correlation",
        "final_allocation_rule_sha256": lock_hash,
        "selected_allocation": lock["selected_allocation"],
    }
    (OUTPUT_DIR / "allocation_comparison_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(lock, indent=2, default=str), flush=True)


if __name__ == "__main__":
    execute()
