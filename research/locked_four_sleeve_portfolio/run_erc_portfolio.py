"""Rolling 63-session equal-risk-contribution portfolio for four locked MNQ sleeves."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
RETURNS_PATH = OUTPUT_DIR / "erc_daily_returns.csv"
PNL_PATH = OUTPUT_DIR / "daily_pnl.csv"
CONFIG_PATH = OUTPUT_DIR / "locked_four_sleeve_config.json"
CONFIG_HASH_PATH = OUTPUT_DIR / "locked_four_sleeve_config.sha256"
WINDOW = 63
ANNUALIZATION = 252
INITIAL_CAPITAL = 50_000.0
SLEEVES = ["orb_vwap_atr", "volatility_breakout", "compression_breakout", "atr_trend_flip"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def solve_erc(covariance: np.ndarray, tolerance: float = 1e-10, max_iterations: int = 20_000) -> tuple[np.ndarray, int, float]:
    """Solve equal component variance contributions by cyclical coordinate descent."""
    n = covariance.shape[0]
    if covariance.shape != (n, n) or not np.allclose(covariance, covariance.T, atol=1e-14):
        raise ValueError("Covariance must be square and symmetric")
    if np.any(np.diag(covariance) <= 0):
        raise ValueError("ERC is undefined with a non-positive sleeve variance")
    scale = float(np.trace(covariance) / n)
    sigma = covariance / scale
    budgets = np.full(n, 1.0 / n)
    x = 1.0 / np.sqrt(np.diag(sigma))
    x /= x.sum()
    error = math.inf
    for iteration in range(1, max_iterations + 1):
        for i in range(n):
            cross = float(sigma[i] @ x - sigma[i, i] * x[i])
            discriminant = cross * cross + 4.0 * sigma[i, i] * budgets[i]
            x[i] = (-cross + math.sqrt(discriminant)) / (2.0 * sigma[i, i])
        weights = x / x.sum()
        component = weights * (sigma @ weights)
        total = float(component.sum())
        shares = component / total
        error = float(np.max(np.abs(shares - budgets)))
        if error <= tolerance:
            return weights, iteration, error
    raise RuntimeError(f"ERC solver did not converge: error={error:.3e}")


def risk_details(weights: np.ndarray, covariance: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    portfolio_variance = float(weights @ covariance @ weights)
    portfolio_volatility = math.sqrt(portfolio_variance)
    marginal_volatility = covariance @ weights / portfolio_volatility
    component_volatility = weights * marginal_volatility
    shares = component_volatility / portfolio_volatility
    return portfolio_volatility, marginal_volatility, component_volatility, shares


def portfolio_metrics(name: str, pnl: pd.Series) -> dict[str, Any]:
    equity = INITIAL_CAPITAL + pnl.cumsum()
    prior = equity.shift(1).fillna(INITIAL_CAPITAL)
    returns = pnl / prior
    peak = equity.cummax().clip(lower=INITIAL_CAPITAL)
    drawdown = equity - peak
    gains = float(pnl[pnl > 0].sum())
    losses = abs(float(pnl[pnl < 0].sum()))
    volatility = float(returns.std(ddof=1))
    return {
        "portfolio": name,
        "start_date": pnl.index[0],
        "end_date": pnl.index[-1],
        "session_count": len(pnl),
        "net_pnl_usd": float(pnl.sum()),
        "sharpe_annualized": float(returns.mean() / volatility * math.sqrt(ANNUALIZATION)),
        "daily_profit_factor": gains / losses,
        "annualized_volatility_pct": volatility * math.sqrt(ANNUALIZATION) * 100.0,
        "max_drawdown_usd": float(drawdown.min()),
        "max_drawdown_pct": float((drawdown / peak).min() * 100.0),
    }


def weight_summary(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sleeve in SLEEVES:
        values = weights[sleeve]
        rows.append({
            "sleeve": sleeve,
            "mean_weight": values.mean(),
            "min_weight": values.min(),
            "max_weight": values.max(),
            "p05_weight": values.quantile(0.05),
            "p95_weight": values.quantile(0.95),
            "days_above_50_pct": int(values.gt(0.50).sum()),
            "days_above_75_pct": int(values.gt(0.75).sum()),
            "share_days_above_50_pct": float(values.gt(0.50).mean()),
        })
    return pd.DataFrame(rows)


def write_report(
    comparison: pd.DataFrame, summary: pd.DataFrame, risk: pd.DataFrame,
    portfolio: pd.DataFrame, full_pair_correlation: float, input_hash: str, config_hash: str,
) -> None:
    comp = comparison.set_index("portfolio")
    pair = ["volatility_breakout", "atr_trend_flip"]
    erc_pair_rc = risk[(risk["portfolio"] == "erc_rolling") & risk["sleeve"].isin(pair)].groupby("session_date")["risk_contribution_pct"].sum()
    ew_pair_rc = risk[(risk["portfolio"] == "equal_weight") & risk["sleeve"].isin(pair)].groupby("session_date")["risk_contribution_pct"].sum()
    compression = summary.set_index("sleeve").loc["compression_breakout"]
    concentration_text = (
        f"`compression_breakout` reçoit en moyenne {compression.mean_weight:.1%}, avec un maximum de {compression.max_weight:.1%}. "
        f"Son poids dépasse 50 % pendant {int(compression.days_above_50_pct)} séances ({compression.share_days_above_50_pct:.1%})."
    )
    lines = [
        "# Portefeuille ERC rolling — quatre sleeves MNQ verrouillées", "", "## Méthode", "",
        "- Covariance empirique sur les **63 séances précédentes**, arrêtée à `t-1` pour les poids de `t`.",
        "- Poids longs, somme égale à 1, aucun levier.",
        "- Le solveur égalise les contributions composantes au risque `w_i × (Σw)_i`; aucune moyenne de rendement et aucun Sharpe ne participent au calcul.",
        "- La comparaison equal weight utilise la même fenêtre commune que l'ERC.",
        "- Le turnover est `0.5 × somme(|w_t - w_(t-1)|)`; le premier jour est mesuré depuis 25 % par sleeve.",
        "- Aucun coût de réallocation entre sleeves n'est retranché.",
        f"- SHA-256 input rendements : `{input_hash}`.",
        f"- SHA-256 configuration des sleeves : `{config_hash}`.", "", "## Comparaison", "",
        "| Portefeuille | PnL net | Sharpe | PF journalier | Vol annualisée | Max DD | Max DD % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| `{row.portfolio}` | {row.net_pnl_usd:,.2f} | {row.sharpe_annualized:.3f} | "
            f"{row.daily_profit_factor:.3f} | {row.annualized_volatility_pct:.2f}% | "
            f"{row.max_drawdown_usd:,.2f} | {row.max_drawdown_pct:.2f}% |"
        )
    lines += ["", "## Poids ERC", "",
              "| Sleeve | Moyen | Minimum | Maximum | P5 | P95 | Jours > 50 % |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.sleeve}` | {row.mean_weight:.2%} | {row.min_weight:.2%} | {row.max_weight:.2%} | "
            f"{row.p05_weight:.2%} | {row.p95_weight:.2%} | {int(row.days_above_50_pct)} |"
        )
    lines += ["", "## Diagnostic volatility breakout / ATR trend flip", "",
              f"Corrélation descriptive sur tout l'échantillon : **{full_pair_correlation:.3f}**. Elle n'est pas utilisée pour calculer les poids.",
              f"Corrélation rolling 63 séances moyenne : **{portfolio['rolling_corr_volatility_atr'].mean():.3f}** "
              f"(min {portfolio['rolling_corr_volatility_atr'].min():.3f}, max {portfolio['rolling_corr_volatility_atr'].max():.3f}).",
              f"Contribution combinée moyenne au risque en equal weight : **{ew_pair_rc.mean():.2f}%**.",
              f"Contribution combinée moyenne au risque en ERC : **{erc_pair_rc.mean():.2f}%**.",
              ("L'ERC réduit leur concentration de risque combinée." if erc_pair_rc.mean() < ew_pair_rc.mean() else "L'ERC ne réduit pas leur concentration de risque combinée sur cette mesure."),
              "", "## Diagnostic compression breakout", "", concentration_text,
              "Ce résultat est présenté sans cap : sa faible volatilité incorpore aussi de nombreux jours sans position, donc le poids ERC ne doit pas être interprété comme une preuve de capacité ou de liquidité.",
              "", "## Turnover", "",
              f"Turnover quotidien moyen : **{portfolio['erc_weight_turnover'].mean():.2%}** ; médian : **{portfolio['erc_weight_turnover'].median():.2%}** ; maximum : **{portfolio['erc_weight_turnover'].max():.2%}**.",
              "", "## Fichiers", "", "- `erc_portfolio.csv`", "- `erc_weights.csv`", "- `erc_risk_contributions.csv`", "- `erc_weight_summary.csv`", "- `run_erc_portfolio.py`", ""]
    (OUTPUT_DIR / "ERC_PORTFOLIO_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def execute() -> None:
    input_hash = sha256(RETURNS_PATH)
    config_hash = sha256(CONFIG_PATH)
    if config_hash != CONFIG_HASH_PATH.read_text(encoding="ascii").split()[0]:
        raise RuntimeError("Locked four-sleeve configuration hash mismatch")
    returns = pd.read_csv(RETURNS_PATH, parse_dates=["session_date"]).set_index("session_date")[SLEEVES]
    pnl_source = pd.read_csv(PNL_PATH, parse_dates=["session_date"]).set_index("session_date")[SLEEVES]
    if len(returns) != len(pnl_source) or not returns.index.equals(pnl_source.index):
        raise RuntimeError("Return and PnL inputs are not aligned")
    if not np.allclose(returns.to_numpy(), pnl_source.to_numpy() / INITIAL_CAPITAL):
        raise RuntimeError("ERC return input does not reproduce locked sleeve PnL")

    weight_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    previous = np.full(len(SLEEVES), 0.25)
    for position in range(WINDOW, len(returns)):
        date = returns.index[position]
        history = returns.iloc[position - WINDOW:position]
        covariance = history.cov(ddof=1).to_numpy(dtype=float)
        weights, iterations, error = solve_erc(covariance)
        ew = np.full(len(SLEEVES), 0.25)
        erc_vol, erc_marginal, erc_component, erc_shares = risk_details(weights, covariance)
        ew_vol, ew_marginal, ew_component, ew_shares = risk_details(ew, covariance)
        corr = float(history["volatility_breakout"].corr(history["atr_trend_flip"]))
        row: dict[str, Any] = {
            "session_date": date,
            "covariance_window_start": history.index[0],
            "covariance_window_end": history.index[-1],
            "solver_iterations": iterations,
            "erc_max_risk_budget_error": error,
            "weight_turnover": 0.5 * float(np.abs(weights - previous).sum()),
            "rolling_corr_volatility_atr": corr,
        }
        row.update({sleeve: weights[i] for i, sleeve in enumerate(SLEEVES)})
        weight_rows.append(row)
        for portfolio_name, portfolio_weights, vol, marginal, component, shares in (
            ("erc_rolling", weights, erc_vol, erc_marginal, erc_component, erc_shares),
            ("equal_weight", ew, ew_vol, ew_marginal, ew_component, ew_shares),
        ):
            for i, sleeve in enumerate(SLEEVES):
                risk_rows.append({
                    "session_date": date,
                    "portfolio": portfolio_name,
                    "sleeve": sleeve,
                    "weight": portfolio_weights[i],
                    "ex_ante_daily_portfolio_volatility": vol,
                    "marginal_risk_contribution": marginal[i],
                    "component_risk_contribution": component[i],
                    "risk_contribution_pct": shares[i] * 100.0,
                })
        previous = weights

    weights = pd.DataFrame(weight_rows).set_index("session_date")
    risk = pd.DataFrame(risk_rows)
    common_pnl = pnl_source.reindex(weights.index)
    erc_components = common_pnl.mul(weights[SLEEVES])
    ew_components = common_pnl * 0.25
    erc_pnl = erc_components.sum(axis=1)
    ew_pnl = ew_components.sum(axis=1)
    erc_equity = INITIAL_CAPITAL + erc_pnl.cumsum()
    ew_equity = INITIAL_CAPITAL + ew_pnl.cumsum()
    portfolio = pd.DataFrame(index=weights.index)
    for sleeve in SLEEVES:
        portfolio[f"erc_pnl_contribution__{sleeve}"] = erc_components[sleeve]
        portfolio[f"equal_weight_pnl_contribution__{sleeve}"] = ew_components[sleeve]
    portfolio["erc_daily_pnl_usd"] = erc_pnl
    portfolio["erc_equity_usd"] = erc_equity
    portfolio["equal_weight_daily_pnl_usd"] = ew_pnl
    portfolio["equal_weight_equity_usd"] = ew_equity
    portfolio["erc_weight_turnover"] = weights["weight_turnover"]
    portfolio["rolling_corr_volatility_atr"] = weights["rolling_corr_volatility_atr"]
    portfolio.index.name = "session_date"

    comparison = pd.DataFrame([
        portfolio_metrics("equal_weight", ew_pnl),
        portfolio_metrics("erc_rolling", erc_pnl),
    ])
    summary = weight_summary(weights)
    full_pair_correlation = float(returns["volatility_breakout"].corr(returns["atr_trend_flip"]))
    portfolio.to_csv(OUTPUT_DIR / "erc_portfolio.csv")
    weights.to_csv(OUTPUT_DIR / "erc_weights.csv")
    risk.to_csv(OUTPUT_DIR / "erc_risk_contributions.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "erc_weight_summary.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "erc_portfolio_metrics.csv", index=False)
    write_report(comparison, summary, risk, portfolio, full_pair_correlation, input_hash, config_hash)

    if not np.allclose(weights[SLEEVES].sum(axis=1), 1.0, atol=1e-10):
        raise RuntimeError("ERC weights do not sum to one")
    if weights[SLEEVES].lt(-1e-12).any().any() or weights[SLEEVES].gt(1 + 1e-12).any().any():
        raise RuntimeError("ERC weights violate long-only/no-leverage constraints")
    if not (weights["covariance_window_end"] < weights.index).all():
        raise RuntimeError("Look-ahead detected in covariance windows")
    erc_risk = risk[risk["portfolio"] == "erc_rolling"].pivot(index="session_date", columns="sleeve", values="risk_contribution_pct")
    max_budget_error = float((erc_risk - 25.0).abs().to_numpy().max())
    if max_budget_error > 1e-5:
        raise RuntimeError(f"ERC risk contributions are not equal: {max_budget_error}")
    metadata = {
        "erc_returns_sha256": input_hash,
        "locked_four_sleeve_config_sha256": config_hash,
        "window_sessions": WINDOW,
        "covariance_lag_sessions": 1,
        "first_weight_date": str(weights.index[0].date()),
        "last_weight_date": str(weights.index[-1].date()),
        "observation_count": len(weights),
        "long_only": True,
        "fully_invested": True,
        "leverage": False,
        "sharpe_used_in_optimization": False,
        "max_weight_sum_error": float((weights[SLEEVES].sum(axis=1) - 1.0).abs().max()),
        "max_erc_risk_contribution_error_percentage_points": max_budget_error,
        "max_solver_iterations": int(weights["solver_iterations"].max()),
        "full_sample_volatility_atr_correlation_diagnostic": full_pair_correlation,
    }
    (OUTPUT_DIR / "erc_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    execute()
