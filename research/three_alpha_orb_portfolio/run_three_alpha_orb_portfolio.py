"""Compare three locked alpha mechanisms, retained ORB, and a 25% equal-weight blend."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
MPL_CACHE = OUTPUT_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from research.baseline_backtests.baseline_engine import (  # noqa: E402
    load_frozen_specs,
    prepare_data,
    run_baseline_backtest,
)


CONFIG_PATH = OUTPUT_DIR / "locked_portfolio_config.json"
CONFIG_HASH_PATH = OUTPUT_DIR / "locked_portfolio_config.sha256"
FROZEN_DIR = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
DATA_PATH = REPO_ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"
ORB_TRADES_PATH = REPO_ROOT / "export" / "mnq_orb_vwap_atr_retained" / "trades.csv"
INITIAL_CAPITAL = 50_000.0
ANNUALIZATION = 252


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes() -> dict[str, str]:
    paths = sorted(FROZEN_DIR.glob("*.json")) + [FROZEN_MANIFEST]
    return {str(path.relative_to(REPO_ROOT)): sha256(path) for path in paths}


def load_locked_config() -> tuple[dict[str, Any], str]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = {"volatility_breakout", "compression_breakout", "atr_trend_flip"}
    if not config.get("locked_before_execution") or set(config["strategies"]) != expected:
        raise RuntimeError("Portfolio configuration is not locked or has unexpected strategies")
    weights = config["portfolio_construction"]["weights"]
    if set(weights) != expected | {"orb_vwap_atr"} or not math.isclose(sum(weights.values()), 1.0):
        raise RuntimeError("Equal weights must cover the four strategies and sum to one")
    digest = sha256(CONFIG_PATH)
    CONFIG_HASH_PATH.write_text(f"{digest}  {CONFIG_PATH.name}\n", encoding="ascii")
    return config, digest


def apply_structural_parameters(spec: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(spec)
    structural = updated["canonical_parameters"]["structural"]
    for name, value in values.items():
        if name not in structural:
            raise KeyError(f"Unknown parameter {updated['mechanism_id']}::{name}")
        structural[name] = value
    return updated


def daily_frame(session_dates: pd.Series, strategy: str, pnl_by_day: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"session_date": pd.to_datetime(session_dates).dt.date})
    frame["strategy"] = strategy
    mapping = {pd.Timestamp(day).date(): float(value) for day, value in pnl_by_day.items()}
    frame["daily_pnl_usd"] = frame["session_date"].map(mapping).fillna(0.0)
    frame["equity_usd"] = INITIAL_CAPITAL + frame["daily_pnl_usd"].cumsum()
    prior = frame["equity_usd"].shift(1).fillna(INITIAL_CAPITAL)
    frame["daily_return"] = frame["daily_pnl_usd"] / prior
    return frame


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 and np.isfinite(denominator) else np.nan


def metrics(strategy: str, daily: pd.DataFrame, trades: pd.DataFrame | None, pf_basis: str) -> dict[str, Any]:
    pnl = daily["daily_pnl_usd"].astype(float)
    returns = daily["daily_return"].astype(float)
    equity = daily["equity_usd"].astype(float)
    peak = pd.concat([pd.Series([INITIAL_CAPITAL]), equity], ignore_index=True).cummax().iloc[1:].reset_index(drop=True)
    drawdown = equity.reset_index(drop=True) - peak
    daily_gains = float(pnl[pnl > 0].sum())
    daily_losses = abs(float(pnl[pnl < 0].sum()))
    std = float(returns.std(ddof=1))
    row: dict[str, Any] = {
        "strategy": strategy,
        "start_date": daily["session_date"].iloc[0],
        "end_date": daily["session_date"].iloc[-1],
        "session_count": len(daily),
        "net_pnl_usd": float(pnl.sum()),
        "total_return_pct": float((equity.iloc[-1] / INITIAL_CAPITAL - 1.0) * 100.0),
        "sharpe_annualized": float(returns.mean() / std * math.sqrt(ANNUALIZATION)) if std > 0 else np.nan,
        "max_drawdown_usd": float(drawdown.min()),
        "max_drawdown_pct": float((drawdown / peak).min() * 100.0),
        "daily_profit_factor": safe_ratio(daily_gains, daily_losses),
        "positive_day_rate": float((pnl > 0).mean()),
        "pf_primary_basis": pf_basis,
    }
    if trades is None:
        row.update({"trade_count": np.nan, "trade_profit_factor": np.nan, "expectancy_per_trade_usd": np.nan, "win_rate": np.nan})
    else:
        trade_pnl = trades["net_pnl_usd"].astype(float)
        wins = float(trade_pnl[trade_pnl > 0].sum())
        losses = abs(float(trade_pnl[trade_pnl < 0].sum()))
        row.update({
            "trade_count": len(trades),
            "trade_profit_factor": safe_ratio(wins, losses),
            "expectancy_per_trade_usd": float(trade_pnl.mean()) if len(trades) else np.nan,
            "win_rate": float((trade_pnl > 0).mean()) if len(trades) else np.nan,
        })
    return row


def plot_equity(curves: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    styles = {
        "equal_weight_portfolio": {"linewidth": 3.0, "color": "black"},
        "orb_vwap_atr": {"linewidth": 1.8, "color": "#1f77b4"},
        "volatility_breakout": {"linewidth": 1.6, "color": "#2ca02c"},
        "compression_breakout": {"linewidth": 1.6, "color": "#ff7f0e"},
        "atr_trend_flip": {"linewidth": 1.6, "color": "#9467bd"},
    }
    for name, frame in curves.groupby("strategy", sort=False):
        ax.plot(pd.to_datetime(frame["session_date"]), frame["equity_usd"], label=name, **styles[name])
    ax.axhline(INITIAL_CAPITAL, color="#888888", linewidth=0.8)
    ax.set_title("MNQ — ORB, trois alphas verrouillés et portefeuille equal weight")
    ax.set_ylabel("Equity (USD, base 50 000)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "equity_curves.png", dpi=170)
    plt.close(fig)


def write_report(metric_frame: pd.DataFrame, config_hash: str) -> None:
    lines = [
        "# ORB + trois stratégies MNQ — portefeuille equal weight", "",
        "## Méthode", "",
        "- Période commune : **2019-05-06 → 2026-03-19**.",
        "- Les trois mécanismes utilisent exactement leurs paramètres verrouillés et le moteur baseline à 1 contrat MNQ.",
        "- L'ORB est la version retenue et corrigée `orb_vwap_atr`, avec son sizing historique à 0,5 % de risque nominal sur 50 000 USD.",
        "- Coûts communs : 1,25 USD par côté et 1 tick de slippage par côté.",
        "- Portefeuille : quatre sleeves fixes de 25 %. Le PnL journalier du portefeuille est la moyenne des quatre PnL journaliers.",
        "- Le PF principal du portefeuille est calculé sur le PnL quotidien agrégé, car plusieurs stratégies peuvent être actives simultanément.",
        f"- SHA-256 de la configuration : `{config_hash}`.", "",
        "## Métriques", "",
        "| Stratégie | Trades | PnL net | Rendement | Sharpe | PF trades | PF journalier | Max DD | Max DD % | Expectancy | Win rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_frame.itertuples(index=False):
        trades = "n/a" if pd.isna(row.trade_count) else f"{int(row.trade_count)}"
        trade_pf = "n/a" if pd.isna(row.trade_profit_factor) else f"{row.trade_profit_factor:.2f}"
        expectancy = "n/a" if pd.isna(row.expectancy_per_trade_usd) else f"{row.expectancy_per_trade_usd:,.2f}"
        win_rate = "n/a" if pd.isna(row.win_rate) else f"{100 * row.win_rate:.1f}%"
        lines.append(
            f"| `{row.strategy}` | {trades} | {row.net_pnl_usd:,.2f} | {row.total_return_pct:.2f}% | "
            f"{row.sharpe_annualized:.2f} | {trade_pf} | {row.daily_profit_factor:.2f} | "
            f"{row.max_drawdown_usd:,.2f} | {row.max_drawdown_pct:.2f}% | {expectancy} | {win_rate} |"
        )
    lines += [
        "", "## Lecture", "",
        "La courbe equal weight normalise les quatre stratégies par leur capital de référence. Elle ne simule pas quatre contrats simultanés ni un compte Topstep partagé avec ses limites intrajournalières.",
        "Les métriques individuelles permettent donc de comparer les moteurs ; la ligne portefeuille mesure leur combinaison quotidienne avec les poids fixés avant calcul.", "",
        "## Fichiers", "",
        "- `equity_curves.png` et `equity_curves.csv`", "- `strategy_metrics.csv`", "- `daily_pnl_and_contributions.csv`", "- `run_metadata.json`", "",
    ]
    (OUTPUT_DIR / "PORTFOLIO_COMPARISON_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def execute() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    config, config_hash = load_locked_config()
    frozen_before = frozen_hashes()
    orb_hash_before = sha256(ORB_TRADES_PATH)
    start = pd.Timestamp(config["start_date"]).date()
    end = pd.Timestamp(config["end_date"]).date()
    specs, _, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    spec_by_id = {spec["mechanism_id"]: spec for spec in specs}
    prepared = prepare_data(DATA_PATH, start_date=start, end_date=end)
    session_dates: pd.Series | None = None

    curves: list[pd.DataFrame] = []
    trades_by_name: dict[str, pd.DataFrame] = {}
    checks: dict[str, Any] = {}
    for mechanism_id, parameters in config["strategies"].items():
        spec = apply_structural_parameters(spec_by_id[mechanism_id], parameters)
        result = run_baseline_backtest(spec, prepared, evaluation_start=start, evaluation_end=end)
        if session_dates is None:
            session_dates = result.daily["session_date"].copy()
        violations = {key: value for key, value in result.checks.items() if key.endswith("violations") and value}
        if violations:
            raise RuntimeError(f"{mechanism_id} consistency failure: {violations}")
        daily = result.daily[["session_date", "daily_pnl_usd", "equity_usd", "daily_return"]].copy()
        daily.insert(1, "strategy", mechanism_id)
        curves.append(daily)
        trades_by_name[mechanism_id] = result.trades.copy()
        checks[mechanism_id] = result.checks

    if session_dates is None:
        raise RuntimeError("No intraday session calendar was produced")
    orb_trades = pd.read_csv(ORB_TRADES_PATH)
    orb_trades["session_date"] = pd.to_datetime(orb_trades["session_date"]).dt.date
    orb_trades = orb_trades[orb_trades["session_date"].between(start, end)].copy()
    orb_daily = daily_frame(session_dates, "orb_vwap_atr", orb_trades.groupby("session_date")["net_pnl_usd"].sum())
    curves.append(orb_daily)
    trades_by_name["orb_vwap_atr"] = orb_trades

    components = pd.concat(curves, ignore_index=True)
    wide = components.pivot(index="session_date", columns="strategy", values="daily_pnl_usd").fillna(0.0)
    weights = pd.Series(config["portfolio_construction"]["weights"], dtype=float)
    contributions = wide.mul(weights, axis=1)
    portfolio_pnl = contributions.sum(axis=1)
    portfolio_daily = daily_frame(pd.Series(portfolio_pnl.index), "equal_weight_portfolio", portfolio_pnl)
    all_curves = pd.concat([components, portfolio_daily], ignore_index=True)

    metric_rows = [metrics(name, frame, trades_by_name[name], "trade and daily") for name, frame in components.groupby("strategy", sort=False)]
    metric_rows.append(metrics("equal_weight_portfolio", portfolio_daily, None, "aggregated daily pnl"))
    metric_frame = pd.DataFrame(metric_rows)
    order = ["orb_vwap_atr", "volatility_breakout", "compression_breakout", "atr_trend_flip", "equal_weight_portfolio"]
    metric_frame["_order"] = metric_frame["strategy"].map({name: i for i, name in enumerate(order)})
    metric_frame = metric_frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    contribution_output = wide.copy()
    for name in weights.index:
        contribution_output[f"{name}__weighted_contribution"] = contributions[name]
    contribution_output["equal_weight_portfolio_daily_pnl"] = portfolio_pnl
    contribution_output.reset_index().to_csv(OUTPUT_DIR / "daily_pnl_and_contributions.csv", index=False)
    all_curves.to_csv(OUTPUT_DIR / "equity_curves.csv", index=False)
    metric_frame.to_csv(OUTPUT_DIR / "strategy_metrics.csv", index=False)
    plot_equity(all_curves)
    write_report(metric_frame, config_hash)

    if frozen_before != frozen_hashes():
        raise RuntimeError("A frozen baseline changed")
    if orb_hash_before != sha256(ORB_TRADES_PATH):
        raise RuntimeError("The retained ORB trade source changed")
    if config_hash != sha256(CONFIG_PATH):
        raise RuntimeError("The locked portfolio configuration changed")
    metadata = {
        "config_sha256": config_hash,
        "data_sha256": sha256(DATA_PATH),
        "orb_trades_sha256": orb_hash_before,
        "frozen_files_unchanged": True,
        "orb_source_unchanged": True,
        "strategy_count": 4,
        "portfolio_weight_sum": float(weights.sum()),
        "start_date": str(start),
        "end_date": str(end),
        "session_count": int(len(portfolio_daily)),
        "implementation_checks": checks,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(metric_frame.to_string(index=False), flush=True)
    print(f"Config SHA-256: {config_hash}", flush=True)


if __name__ == "__main__":
    execute()
