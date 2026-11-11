"""Analyze diversification and build a causal inverse-volatility portfolio."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
HEATMAP_DIR = OUTPUT_DIR / "diversification_heatmaps"
MPL_CACHE = OUTPUT_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from research.baseline_backtests.baseline_engine import (  # noqa: E402
    load_frozen_specs,
    prepare_data,
    run_baseline_backtest,
)


CONFIG_PATH = OUTPUT_DIR / "locked_portfolio_config.json"
FROZEN_DIR = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
DATA_PATH = REPO_ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"
ORB_TRADES_PATH = REPO_ROOT / "export" / "mnq_orb_vwap_atr_retained" / "trades.csv"
INITIAL_CAPITAL = 50_000.0
ANNUALIZATION = 252
VOL_WINDOW = 63
MIN_VOL_OBSERVATIONS = 20
STRATEGIES = ["orb_vwap_atr", "volatility_breakout", "compression_breakout", "atr_trend_flip"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes() -> dict[str, str]:
    paths = sorted(FROZEN_DIR.glob("*.json")) + [FROZEN_MANIFEST]
    return {str(path.relative_to(REPO_ROOT)): sha256(path) for path in paths}


def apply_parameters(spec: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(spec)
    structural = updated["canonical_parameters"]["structural"]
    for name, value in values.items():
        if name not in structural:
            raise KeyError(f"Unknown parameter {updated['mechanism_id']}::{name}")
        structural[name] = value
    return updated


def build_daily_and_trades(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    start = pd.Timestamp(config["start_date"]).date()
    end = pd.Timestamp(config["end_date"]).date()
    specs, _, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    spec_by_id = {spec["mechanism_id"]: spec for spec in specs}
    prepared = prepare_data(DATA_PATH, start_date=start, end_date=end)
    daily_map: dict[str, pd.Series] = {}
    trades: dict[str, pd.DataFrame] = {}
    checks: dict[str, Any] = {}
    calendar: pd.Series | None = None
    for mechanism_id, parameters in config["strategies"].items():
        result = run_baseline_backtest(
            apply_parameters(spec_by_id[mechanism_id], parameters), prepared,
            evaluation_start=start, evaluation_end=end,
        )
        violations = {key: value for key, value in result.checks.items() if key.endswith("violations") and value}
        if violations:
            raise RuntimeError(f"{mechanism_id} implementation failure: {violations}")
        if calendar is None:
            calendar = result.daily["session_date"].copy()
        daily_map[mechanism_id] = result.daily.set_index("session_date")["daily_pnl_usd"]
        trades[mechanism_id] = result.trades.copy()
        checks[mechanism_id] = result.checks
    if calendar is None:
        raise RuntimeError("Missing common trading calendar")

    orb = pd.read_csv(ORB_TRADES_PATH)
    orb["session_date"] = pd.to_datetime(orb["session_date"]).dt.date
    orb = orb[orb["session_date"].between(start, end)].copy()
    orb_daily = orb.groupby("session_date")["net_pnl_usd"].sum()
    daily_map["orb_vwap_atr"] = pd.Series(calendar.map(orb_daily).fillna(0.0).to_numpy(), index=calendar)
    trades["orb_vwap_atr"] = orb

    wide = pd.DataFrame(index=pd.Index(calendar, name="session_date"))
    for name in STRATEGIES:
        wide[name] = daily_map[name].reindex(wide.index).fillna(0.0).to_numpy()
    return wide, trades, checks


def intervals(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    output: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for row in frame[["entry_time", "exit_time"]].itertuples(index=False):
        start = pd.to_datetime(row.entry_time, utc=True)
        end = pd.to_datetime(row.exit_time, utc=True)
        if end > start:
            output.append((start, end))
    return sorted(output)


def intersection_minutes(left: list[tuple[pd.Timestamp, pd.Timestamp]], right: list[tuple[pd.Timestamp, pd.Timestamp]]) -> float:
    i = j = 0
    seconds = 0.0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            seconds += (end - start).total_seconds()
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return seconds / 60.0


def downside_correlation(left: pd.Series, right: pd.Series) -> tuple[float, int]:
    mask = (left < 0) | (right < 0)
    return float(left[mask].corr(right[mask])), int(mask.sum())


def underwater(pnl: pd.Series) -> tuple[pd.Series, pd.Series]:
    equity = INITIAL_CAPITAL + pnl.cumsum()
    peak = pd.concat([pd.Series([INITIAL_CAPITAL]), equity.reset_index(drop=True)], ignore_index=True).cummax().iloc[1:].to_numpy()
    drawdown = pd.Series(equity.to_numpy() - peak, index=pnl.index)
    return drawdown, drawdown < -1e-12


def diversification_table(daily: pd.DataFrame, trades: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    position_intervals = {name: intervals(trades[name]) for name in STRATEGIES}
    occupied = {
        name: sum((end - start).total_seconds() for start, end in position_intervals[name]) / 60.0
        for name in STRATEGIES
    }
    underwater_map = {name: underwater(daily[name])[1] for name in STRATEGIES}
    rows: list[dict[str, Any]] = []
    for left, right in itertools.product(STRATEGIES, repeat=2):
        downside_corr, downside_days = downside_correlation(daily[left], daily[right])
        simultaneous_losses = (daily[left].lt(0) & daily[right].lt(0))
        overlap = intersection_minutes(position_intervals[left], position_intervals[right])
        union_minutes = occupied[left] + occupied[right] - overlap
        both_dd = underwater_map[left] & underwater_map[right]
        either_dd = underwater_map[left] | underwater_map[right]
        rows.append({
            "strategy_a": left,
            "strategy_b": right,
            "pnl_correlation": daily[left].corr(daily[right]),
            "downside_conditional_correlation": downside_corr,
            "downside_condition_days": downside_days,
            "simultaneous_loss_days": int(simultaneous_losses.sum()),
            "simultaneous_loss_days_pct": float(simultaneous_losses.mean() * 100.0),
            "position_overlap_minutes": overlap,
            "position_overlap_jaccard_pct": 100.0 * overlap / union_minutes if union_minutes else np.nan,
            "position_overlap_share_of_shorter_pct": 100.0 * overlap / min(occupied[left], occupied[right]) if min(occupied[left], occupied[right]) else np.nan,
            "simultaneous_drawdown_days": int(both_dd.sum()),
            "simultaneous_drawdown_jaccard_pct": 100.0 * both_dd.sum() / either_dd.sum() if either_dd.sum() else np.nan,
        })
    matrix = pd.DataFrame(rows)

    orb_loss_mask = daily["orb_vwap_atr"] < 0
    orb_rows = []
    for name in STRATEGIES:
        values = daily.loc[orb_loss_mask, name]
        orb_rows.append({
            "sleeve": name,
            "orb_loss_day_count": int(orb_loss_mask.sum()),
            "total_pnl_on_orb_loss_days_usd": float(values.sum()),
            "mean_pnl_on_orb_loss_days_usd": float(values.mean()),
            "median_pnl_on_orb_loss_days_usd": float(values.median()),
            "positive_rate_on_orb_loss_days": float((values > 0).mean()),
            "negative_rate_on_orb_loss_days": float((values < 0).mean()),
        })
    orb_loss = pd.DataFrame(orb_rows)

    dd_rows = []
    all_underwater = pd.DataFrame(underwater_map)
    for count in range(1, 5):
        dd_rows.append({"number_of_sleeves_in_drawdown": count, "day_count": int((all_underwater.sum(axis=1) == count).sum())})
    dd_rows.append({"number_of_sleeves_in_drawdown": "all_4", "day_count": int(all_underwater.all(axis=1).sum())})
    drawdown_summary = pd.DataFrame(dd_rows)
    return matrix, orb_loss, drawdown_summary


def portfolio_metrics(name: str, pnl: pd.Series) -> dict[str, Any]:
    equity = INITIAL_CAPITAL + pnl.cumsum()
    prior = equity.shift(1).fillna(INITIAL_CAPITAL)
    returns = pnl / prior
    peak = pd.concat([pd.Series([INITIAL_CAPITAL]), equity.reset_index(drop=True)], ignore_index=True).cummax().iloc[1:].to_numpy()
    drawdown = equity.to_numpy() - peak
    gains = float(pnl[pnl > 0].sum())
    losses = abs(float(pnl[pnl < 0].sum()))
    return {
        "portfolio": name,
        "start_date": pnl.index[0],
        "end_date": pnl.index[-1],
        "session_count": len(pnl),
        "net_pnl_usd": float(pnl.sum()),
        "total_return_pct": float(pnl.sum() / INITIAL_CAPITAL * 100.0),
        "sharpe_annualized": float(returns.mean() / returns.std(ddof=1) * math.sqrt(ANNUALIZATION)),
        "daily_profit_factor": gains / losses,
        "max_drawdown_usd": float(drawdown.min()),
        "max_drawdown_pct": float((drawdown / peak).min() * 100.0),
        "annualized_volatility_pct": float(returns.std(ddof=1) * math.sqrt(ANNUALIZATION) * 100.0),
        "positive_day_rate": float((pnl > 0).mean()),
    }


def risk_contributions(component_returns: pd.DataFrame, portfolio_returns: pd.Series, portfolio_name: str) -> pd.DataFrame:
    variance = float(portfolio_returns.var(ddof=1))
    rows = []
    for name in STRATEGIES:
        covariance = float(component_returns[name].cov(portfolio_returns))
        percent = 100.0 * covariance / variance if variance > 0 else np.nan
        rows.append({
            "portfolio": portfolio_name,
            "sleeve": name,
            "realized_risk_contribution_pct": percent,
            "annualized_component_volatility_pct": float(component_returns[name].std(ddof=1) * math.sqrt(ANNUALIZATION) * 100.0),
        })
    return pd.DataFrame(rows)


def build_portfolios(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_returns = daily / INITIAL_CAPITAL
    vol = base_returns.rolling(VOL_WINDOW, min_periods=MIN_VOL_OBSERVATIONS).std(ddof=1).shift(1)
    inverse = 1.0 / vol.replace(0.0, np.nan)
    weights = inverse.div(inverse.sum(axis=1), axis=0)
    eligible = weights.notna().all(axis=1)
    weights = weights.loc[eligible, STRATEGIES]
    base = daily.loc[eligible, STRATEGIES]
    equal_weight_weights = pd.DataFrame(0.25, index=weights.index, columns=STRATEGIES)

    er_components_pnl = weights * base
    ew_components_pnl = equal_weight_weights * base
    er_pnl = er_components_pnl.sum(axis=1)
    ew_pnl = ew_components_pnl.sum(axis=1)
    er_equity = INITIAL_CAPITAL + er_pnl.cumsum()
    ew_equity = INITIAL_CAPITAL + ew_pnl.cumsum()
    er_peak = er_equity.cummax().clip(lower=INITIAL_CAPITAL)
    ew_peak = ew_equity.cummax().clip(lower=INITIAL_CAPITAL)

    output = pd.DataFrame(index=weights.index)
    for name in STRATEGIES:
        output[f"weight__{name}"] = weights[name]
        output[f"pnl_contribution__{name}"] = er_components_pnl[name]
        output[f"ex_ante_daily_vol__{name}"] = vol.loc[eligible, name]
        output[f"ex_ante_scaled_vol__{name}"] = weights[name] * vol.loc[eligible, name]
    output["equal_risk_daily_pnl_usd"] = er_pnl
    output["equal_risk_equity_usd"] = er_equity
    output["equal_risk_drawdown_usd"] = er_equity - er_peak
    output["equal_weight_daily_pnl_usd"] = ew_pnl
    output["equal_weight_equity_usd"] = ew_equity
    output["equal_weight_drawdown_usd"] = ew_equity - ew_peak
    output.index.name = "session_date"

    comparison = pd.DataFrame([
        portfolio_metrics("equal_weight_common_window", ew_pnl),
        portfolio_metrics("equal_risk_inverse_vol", er_pnl),
        portfolio_metrics("equal_weight_full_period_reference", daily.mean(axis=1)),
    ])
    er_prior = er_equity.shift(1).fillna(INITIAL_CAPITAL)
    ew_prior = ew_equity.shift(1).fillna(INITIAL_CAPITAL)
    er_component_returns = er_components_pnl.div(er_prior, axis=0)
    ew_component_returns = ew_components_pnl.div(ew_prior, axis=0)
    rc = pd.concat([
        risk_contributions(ew_component_returns, ew_pnl / ew_prior, "equal_weight_common_window"),
        risk_contributions(er_component_returns, er_pnl / er_prior, "equal_risk_inverse_vol"),
    ], ignore_index=True)
    mean_weights = weights.mean()
    rc["mean_portfolio_weight_pct"] = rc.apply(
        lambda row: 25.0 if row["portfolio"] == "equal_weight_common_window" else 100.0 * mean_weights[row["sleeve"]], axis=1
    )
    return output, comparison, rc


def heatmap(matrix: pd.DataFrame, column: str, title: str, filename: str, fmt: str = ".2f") -> None:
    pivot = matrix.pivot(index="strategy_a", columns="strategy_b", values=column).reindex(index=STRATEGIES, columns=STRATEGIES)
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap="RdYlGn", center=0, linewidths=0.5, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(HEATMAP_DIR / filename, dpi=170)
    plt.close(fig)


def make_plots(daily: pd.DataFrame, matrix: pd.DataFrame, portfolios: pd.DataFrame) -> None:
    heatmap(matrix, "pnl_correlation", "Corrélation des PnL quotidiens", "pnl_correlation_heatmap.png")
    heatmap(matrix, "downside_conditional_correlation", "Corrélation conditionnelle — au moins une sleeve négative", "downside_correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(13, 7))
    dates = pd.to_datetime(portfolios.index)
    ax.plot(dates, portfolios["equal_weight_equity_usd"], label="Equal weight", linewidth=2.0)
    ax.plot(dates, portfolios["equal_risk_equity_usd"], label="Equal risk inverse vol", linewidth=2.5)
    ax.axhline(INITIAL_CAPITAL, color="#888888", linewidth=0.8)
    ax.set_title("Portefeuilles equal weight et equal risk — fenêtre commune")
    ax.set_ylabel("Equity (USD)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HEATMAP_DIR / "equal_weight_vs_equal_risk_equity.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 7))
    for name in STRATEGIES:
        drawdown, _ = underwater(daily[name])
        ax.plot(pd.to_datetime(drawdown.index), drawdown, label=name, linewidth=1.3)
    ax.set_title("Drawdowns simultanés des sleeves")
    ax.set_ylabel("Drawdown (USD)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HEATMAP_DIR / "sleeve_drawdowns.png", dpi=170)
    plt.close(fig)

    weight_columns = [f"weight__{name}" for name in STRATEGIES]
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.stackplot(pd.to_datetime(portfolios.index), *[portfolios[column] for column in weight_columns], labels=STRATEGIES, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_title("Poids ex ante du portefeuille equal risk")
    ax.set_ylabel("Poids")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(HEATMAP_DIR / "equal_risk_weights.png", dpi=170)
    plt.close(fig)


def write_report(
    matrix: pd.DataFrame,
    orb_loss: pd.DataFrame,
    drawdown_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    risk: pd.DataFrame,
    portfolios: pd.DataFrame,
    config_hash: str,
) -> None:
    off_diag = matrix[matrix["strategy_a"] < matrix["strategy_b"]]
    lowest_corr = off_diag.loc[off_diag["pnl_correlation"].idxmin()]
    highest_corr = off_diag.loc[off_diag["pnl_correlation"].idxmax()]
    highest_overlap = off_diag.loc[off_diag["position_overlap_jaccard_pct"].idxmax()]
    lowest_overlap = off_diag.loc[off_diag["position_overlap_jaccard_pct"].idxmin()]
    all_four_dd = int(drawdown_summary.loc[drawdown_summary["number_of_sleeves_in_drawdown"].eq("all_4"), "day_count"].iloc[0])
    first_date = str(portfolios.index[0])[:10]
    lines = [
        "# Diversification et portefeuille equal risk", "", "## Protocole", "",
        "- Période d'analyse : **2019-05-06 → 2026-03-19**.",
        "- Corrélation conditionnelle : jours où au moins une des deux sleeves a un PnL négatif.",
        "- Overlap : intersection exacte des intervalles entrée-sortie, mesurée en minutes UTC.",
        "- Drawdown simultané : les deux courbes sont sous leur précédent plus-haut le même jour.",
        f"- Equal risk : inverse de la volatilité quotidienne sur {VOL_WINDOW} séances, avec {MIN_VOL_OBSERVATIONS} observations minimum, estimation décalée d'une séance.",
        "- Les poids sont longs, normalisés à 100 %, sans optimisation de Sharpe et sans levier.",
        f"- Première séance investissable equal risk : **{first_date}**.",
        f"- SHA-256 de la configuration verrouillée : `{config_hash}`.", "",
        "## Diversification", "",
        f"- Corrélation quotidienne hors diagonale : **{off_diag['pnl_correlation'].min():.2f} à {off_diag['pnl_correlation'].max():.2f}**.",
        f"- Corrélation conditionnelle négative hors diagonale : **{off_diag['downside_conditional_correlation'].min():.2f} à {off_diag['downside_conditional_correlation'].max():.2f}**.",
        f"- Corrélation la plus faible : `{lowest_corr['strategy_a']}` / `{lowest_corr['strategy_b']}` à **{lowest_corr['pnl_correlation']:.2f}**.",
        f"- Corrélation la plus élevée : `{highest_corr['strategy_a']}` / `{highest_corr['strategy_b']}` à **{highest_corr['pnl_correlation']:.2f}**.",
        f"- Overlap temporel maximal : `{highest_overlap['strategy_a']}` / `{highest_overlap['strategy_b']}` à **{highest_overlap['position_overlap_jaccard_pct']:.1f}%** du temps occupé combiné.",
        f"- Overlap temporel minimal : `{lowest_overlap['strategy_a']}` / `{lowest_overlap['strategy_b']}` à **{lowest_overlap['position_overlap_jaccard_pct']:.1f}%**.",
        f"- Jours où les quatre sleeves sont simultanément en drawdown : **{all_four_dd}**.", "",
        "### Performance pendant les jours de perte ORB", "",
        "| Sleeve | PnL total | PnL moyen/jour | Taux positif | Taux négatif |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in orb_loss.itertuples(index=False):
        lines.append(f"| `{row.sleeve}` | {row.total_pnl_on_orb_loss_days_usd:,.2f} | {row.mean_pnl_on_orb_loss_days_usd:,.2f} | {100*row.positive_rate_on_orb_loss_days:.1f}% | {100*row.negative_rate_on_orb_loss_days:.1f}% |")
    lines += ["", "## Equal weight contre equal risk", "",
              "| Portefeuille | PnL | Sharpe | PF journalier | Max DD | Max DD % | Vol annualisée |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in comparison.itertuples(index=False):
        lines.append(f"| `{row.portfolio}` | {row.net_pnl_usd:,.2f} | {row.sharpe_annualized:.2f} | {row.daily_profit_factor:.2f} | {row.max_drawdown_usd:,.2f} | {row.max_drawdown_pct:.2f}% | {row.annualized_volatility_pct:.2f}% |")
    lines += ["", "## Contributions au risque réalisées", "",
              "| Portefeuille | Sleeve | Poids moyen | Contribution au risque | Vol contribution |",
              "|---|---|---:|---:|---:|"]
    for row in risk.itertuples(index=False):
        lines.append(f"| `{row.portfolio}` | `{row.sleeve}` | {row.mean_portfolio_weight_pct:.1f}% | {row.realized_risk_contribution_pct:.1f}% | {row.annualized_component_volatility_pct:.2f}% |")
    lines += [
        "", "## Interprétation", "",
        "L'inverse-vol égalise ex ante `poids × volatilité propre`. Les contributions réalisées peuvent rester différentes à cause des corrélations entre sleeves.",
        "La comparaison principale utilise la même fenêtre post warm-up pour les deux portefeuilles. La ligne equal weight full-period est conservée comme raccord avec l'étude précédente.",
        "", "## Livrables", "",
        "- `diversification_matrix.csv`", "- `orb_loss_day_sleeve_performance.csv`", "- `simultaneous_drawdown_summary.csv`",
        "- `equal_risk_portfolio.csv`", "- `portfolio_comparison_metrics.csv`", "- `portfolio_risk_contributions.csv`",
        "- dossier `diversification_heatmaps/`", "- `run_diversification_equal_risk.py`", "",
    ]
    (OUTPUT_DIR / "DIVERSIFICATION_AND_EQUAL_RISK_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def execute() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    config_hash = sha256(CONFIG_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not config.get("locked_before_execution"):
        raise RuntimeError("Portfolio configuration is not locked")
    frozen_before = frozen_hashes()
    orb_hash_before = sha256(ORB_TRADES_PATH)
    daily, trades, checks = build_daily_and_trades(config)
    matrix, orb_loss, drawdown_summary = diversification_table(daily, trades)
    portfolios, comparison, risk = build_portfolios(daily)

    matrix.to_csv(OUTPUT_DIR / "diversification_matrix.csv", index=False)
    orb_loss.to_csv(OUTPUT_DIR / "orb_loss_day_sleeve_performance.csv", index=False)
    drawdown_summary.to_csv(OUTPUT_DIR / "simultaneous_drawdown_summary.csv", index=False)
    portfolios.reset_index().to_csv(OUTPUT_DIR / "equal_risk_portfolio.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "portfolio_comparison_metrics.csv", index=False)
    risk.to_csv(OUTPUT_DIR / "portfolio_risk_contributions.csv", index=False)
    make_plots(daily, matrix, portfolios)
    write_report(matrix, orb_loss, drawdown_summary, comparison, risk, portfolios, config_hash)

    if frozen_before != frozen_hashes():
        raise RuntimeError("A frozen baseline changed")
    if orb_hash_before != sha256(ORB_TRADES_PATH):
        raise RuntimeError("ORB source changed")
    if config_hash != sha256(CONFIG_PATH):
        raise RuntimeError("Locked portfolio config changed")
    metadata = {
        "config_sha256": config_hash,
        "data_sha256": sha256(DATA_PATH),
        "orb_trades_sha256": orb_hash_before,
        "start_date": config["start_date"],
        "end_date": config["end_date"],
        "volatility_window_sessions": VOL_WINDOW,
        "minimum_volatility_observations": MIN_VOL_OBSERVATIONS,
        "volatility_estimate_lag_sessions": 1,
        "weight_optimization": False,
        "leverage": False,
        "frozen_files_unchanged": True,
        "implementation_checks": checks,
    }
    (OUTPUT_DIR / "diversification_run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)
    print(risk.to_string(index=False), flush=True)


if __name__ == "__main__":
    execute()
