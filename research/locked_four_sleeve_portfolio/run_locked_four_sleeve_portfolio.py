"""Build the four locked MNQ sleeves, equal-weight reference, and ERC inputs."""

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
TRADES_DIR = OUTPUT_DIR / "sleeve_trades"
PLOTS_DIR = OUTPUT_DIR / "plots"
MPL_CACHE = OUTPUT_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from research.baseline_backtests.baseline_engine import (  # noqa: E402
    BacktestResult,
    load_frozen_specs,
    prepare_data,
    run_baseline_backtest,
)
from research.money_management_lab.mm_engine import (  # noqa: E402
    build_overlay_context,
    run_exit_overlay_backtest,
    slice_result,
)


CONFIG_PATH = OUTPUT_DIR / "locked_four_sleeve_config.json"
CONFIG_HASH_PATH = OUTPUT_DIR / "locked_four_sleeve_config.sha256"
MM_LOCK_PATH = REPO_ROOT / "research" / "money_management_lab" / "final_mm_locked_exit_rules.json"
MM_LOCK_HASH_PATH = REPO_ROOT / "research" / "money_management_lab" / "final_mm_locked_exit_rules.sha256"
FROZEN_DIR = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
DATA_PATH = REPO_ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"
INITIAL_CAPITAL = 50_000.0
ANNUALIZATION = 252
SLEEVES = ["orb_vwap_atr", "volatility_breakout", "compression_breakout", "atr_trend_flip"]
VOTE_SLEEVES = ("volatility_breakout", "compression_breakout", "atr_trend_flip")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_alpha_parameters(spec: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(spec)
    structural = updated["canonical_parameters"]["structural"]
    for name, value in parameters.items():
        if name not in structural:
            raise KeyError(f"Unknown structural parameter {updated['mechanism_id']}::{name}")
        structural[name] = value
    return updated


def load_and_validate_config() -> tuple[dict[str, Any], str, dict[str, Any], str]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not config.get("locked_before_execution") or list(config["sleeves"]) != SLEEVES:
        raise RuntimeError("The four-sleeve configuration is not locked or ordered as expected")
    weights = config["equal_weight_reference"]
    if set(weights) != set(SLEEVES) or not math.isclose(sum(weights.values()), 1.0):
        raise RuntimeError("Equal-weight reference must contain exactly four weights summing to one")
    if any(not math.isclose(float(weights[name]), 0.25) for name in SLEEVES):
        raise RuntimeError("Equal-weight reference must use 25 percent per sleeve")
    mm_locks = json.loads(MM_LOCK_PATH.read_text(encoding="utf-8"))
    stored_mm_hash = MM_LOCK_HASH_PATH.read_text(encoding="ascii").split()[0]
    mm_hash = sha256(MM_LOCK_PATH)
    if mm_hash != stored_mm_hash or not mm_locks.get("terminal_study"):
        raise RuntimeError("Money-management lock hash mismatch")
    if config["lock_sources"]["money_management_rules_sha256"] != mm_hash:
        raise RuntimeError("Configured money-management source hash mismatch")
    config_hash = sha256(CONFIG_PATH)
    CONFIG_HASH_PATH.write_text(f"{config_hash}  {CONFIG_PATH.name}\n", encoding="ascii")
    return config, config_hash, mm_locks, mm_hash


def validate_locks(config: dict[str, Any], mm_locks: dict[str, Any]) -> None:
    for name in VOTE_SLEEVES:
        sleeve = config["sleeves"][name]
        if sleeve.get("configuration_type") != "parameter_vote_ensemble":
            raise RuntimeError(f"{name} is not configured as its frozen parameter-vote ensemble")
        ensemble_path = REPO_ROOT / sleeve["ensemble_source"]
        if sha256(ensemble_path) != sleeve["ensemble_source_sha256"]:
            raise RuntimeError(f"{name} ensemble manifest hash mismatch")
        ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
        if ensemble.get("parameter_region") != sleeve.get("parameter_region"):
            raise RuntimeError(f"{name} ensemble region mismatch")
        if ensemble.get("vote", {}).get("threshold") != sleeve.get("vote", {}).get("threshold"):
            raise RuntimeError(f"{name} vote threshold mismatch")
    if mm_locks["strategies"]["compression_breakout"]["status"] != "MM_LOCKED_BASELINE":
        raise RuntimeError("Compression breakout is not baseline locked")
    atr_exit = mm_locks["strategies"]["atr_trend_flip"]["exit_rule"]
    if atr_exit.get("exit_family") != "atr_stop" or not math.isclose(float(atr_exit.get("stop_atr", -1)), 1.0):
        raise RuntimeError("ATR trend flip final exit lock mismatch")


def implementation_violations(checks: dict[str, Any]) -> dict[str, Any]:
    watched = {
        "lookahead_violations", "future_high_low_violations", "same_bar_policy_violations",
        "double_entry_violations", "overnight_position_violations",
        "entry_after_last_entry_violations", "post_hard_flatten_violations",
        "ambiguous_multi_order_cancellations", "missing_atr_entry_skips",
    }
    return {key: checks.get(key) for key in watched if checks.get(key, 0)}


def load_frozen_vote_result(name: str, sleeve: dict[str, Any]) -> BacktestResult:
    """Load and verify an executed vote ensemble instead of a legacy point."""
    manifest_path = REPO_ROOT / sleeve["ensemble_source"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    execution = manifest.get("execution_results")
    if not execution:
        raise RuntimeError(f"The frozen {name} ensemble has no executed result")
    result_dir = REPO_ROOT / execution["result_directory"]
    for filename, expected_hash in execution["files"].items():
        path = result_dir / filename
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"Frozen {name} result hash mismatch: {filename}")
    trades = pd.read_csv(result_dir / "trades.csv")
    daily = pd.read_csv(result_dir / "daily_results.csv")
    for column in ("signal_time", "entry_time", "exit_time"):
        trades[column] = pd.to_datetime(trades[column], utc=True).dt.tz_convert("America/New_York")
    trades["session_date"] = pd.to_datetime(trades["session_date"]).dt.date
    daily["session_date"] = pd.to_datetime(daily["session_date"]).dt.date
    equity = daily[["session_date", "mechanism_id", "daily_pnl_usd", "daily_return", "equity_usd"]].copy()
    positions = trades[["mechanism_id", "trade_id", "session_date", "position", "quantity", "entry_time", "exit_time"]].copy()
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    checks = summary["aggregate_checks"]
    if implementation_violations(checks):
        raise RuntimeError(f"Frozen {name} vote failed checks: {implementation_violations(checks)}")
    return BacktestResult(trades=trades, daily=daily, equity=equity, positions=positions, checks=checks)


def run_alpha_sleeves(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Any]]:
    start = pd.Timestamp(config["start_date"]).date()
    end = pd.Timestamp(config["end_date"]).date()
    specs, _, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    spec_by_id = {spec["mechanism_id"]: spec for spec in specs}
    prepared = prepare_data(DATA_PATH, start_date=start, end_date=end)
    runs: dict[str, Any] = {}
    trades: dict[str, pd.DataFrame] = {}
    checks: dict[str, Any] = {}
    for name in SLEEVES[1:]:
        sleeve = config["sleeves"][name]
        if sleeve.get("configuration_type") == "parameter_vote_ensemble":
            run = load_frozen_vote_result(name, sleeve)
        else:
            spec = apply_alpha_parameters(spec_by_id[name], sleeve["alpha_parameters"])
            if sleeve["exit"] == "locked_baseline":
                run = run_baseline_backtest(spec, prepared, evaluation_start=start, evaluation_end=end)
            else:
                policy = {"family": sleeve["exit"]["type"]}
                policy.update({k: v for k, v in sleeve["exit"].items() if k not in {"type", "atr_definition"}})
                timeframe = int(sleeve["signal_timeframe_minutes"])
                context = build_overlay_context(spec, prepared, timeframe)
                run = slice_result(run_exit_overlay_backtest(spec, prepared, policy, timeframe, context=context), start, end)
        violations = implementation_violations(run.checks)
        if violations:
            raise RuntimeError(f"Implementation checks failed for {name}: {violations}")
        runs[name] = run
        trades[name] = run.trades.copy()
        checks[name] = run.checks
    return runs, trades, checks


def equity_from_pnl(pnl: pd.Series) -> pd.DataFrame:
    equity = INITIAL_CAPITAL + pnl.cumsum()
    prior = equity.shift(1).fillna(INITIAL_CAPITAL)
    return pd.DataFrame({"daily_pnl_usd": pnl, "equity_usd": equity, "daily_return": pnl / prior})


def strategy_metrics(name: str, daily: pd.DataFrame, trades: pd.DataFrame | None) -> dict[str, Any]:
    pnl = daily["daily_pnl_usd"].astype(float)
    returns = daily["daily_return"].astype(float)
    equity = daily["equity_usd"].astype(float)
    peak = equity.cummax().clip(lower=INITIAL_CAPITAL)
    drawdown = equity - peak
    gains = float(pnl[pnl > 0].sum())
    losses = abs(float(pnl[pnl < 0].sum()))
    std = float(returns.std(ddof=1))
    row = {
        "strategy": name,
        "start_date": daily.index[0],
        "end_date": daily.index[-1],
        "session_count": len(daily),
        "net_pnl_usd": float(pnl.sum()),
        "sharpe_annualized": float(returns.mean() / std * math.sqrt(ANNUALIZATION)) if std > 0 else np.nan,
        "daily_profit_factor": gains / losses if losses > 0 else np.nan,
        "max_drawdown_usd": float(drawdown.min()),
        "max_drawdown_pct": float((drawdown / peak).min() * 100.0),
        "annualized_volatility_pct": float(std * math.sqrt(ANNUALIZATION) * 100.0),
    }
    if trades is None:
        row.update({"trade_count": np.nan, "trade_profit_factor": np.nan, "expectancy_per_trade_usd": np.nan})
    else:
        tpnl = trades["net_pnl_usd"].astype(float)
        wins, loss = float(tpnl[tpnl > 0].sum()), abs(float(tpnl[tpnl < 0].sum()))
        row.update({
            "trade_count": len(trades),
            "trade_profit_factor": wins / loss if loss > 0 else np.nan,
            "expectancy_per_trade_usd": float(tpnl.mean()) if len(tpnl) else np.nan,
        })
    return row


def covariance_outputs(
    daily_pnl: pd.DataFrame, weights: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    erc_returns = daily_pnl / INITIAL_CAPITAL
    daily_cov = erc_returns.cov(ddof=1).reindex(index=SLEEVES, columns=SLEEVES)
    annual_cov = daily_cov * ANNUALIZATION
    correlation = erc_returns.corr().reindex(index=SLEEVES, columns=SLEEVES)
    w = weights.reindex(SLEEVES).to_numpy(dtype=float)
    cov = daily_cov.to_numpy(dtype=float)
    variance = float(w @ cov @ w)
    marginal = cov @ w
    component = w * marginal
    rc = pd.DataFrame({
        "sleeve": SLEEVES,
        "equal_weight": w,
        "standalone_annualized_volatility_pct": np.sqrt(np.diag(cov) * ANNUALIZATION) * 100.0,
        "marginal_variance_contribution": marginal,
        "component_variance_contribution": component,
        "risk_contribution_pct": component / variance * 100.0,
    })
    return erc_returns, daily_cov, annual_cov, correlation, rc


def save_matrix(frame: pd.DataFrame, path: Path) -> None:
    frame.rename_axis("sleeve").reset_index().to_csv(path, index=False)


def make_plots(curves: pd.DataFrame, correlation: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    for name in SLEEVES + ["equal_weight_portfolio"]:
        part = curves[curves["strategy"] == name]
        ax.plot(pd.to_datetime(part["session_date"]), part["equity_usd"], label=name, linewidth=2.5 if name == "equal_weight_portfolio" else 1.4)
    ax.axhline(INITIAL_CAPITAL, color="#888888", linewidth=0.8)
    ax.set_title("MNQ — quatre sleeves verrouillées et référence equal weight")
    ax.set_ylabel("Equity (USD, base 50 000)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "equity_curves.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="RdYlGn", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Corrélation des rendements quotidiens des sleeves")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=170)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, risk: pd.DataFrame, config_hash: str, mm_hash: str) -> None:
    lines = [
        "# Portefeuille MNQ — quatre sleeves verrouillées", "", "## Construction", "",
        "- Période commune : **2019-05-06 → 2026-03-19**.",
        "- ORB : export retenu `baseline_fixed`, sizing historique de 0,5 % du nominal de 50 000 USD.",
        "- `volatility_breakout` : vote de 42 cellules, ATR 10–22 × multiple 0,5–0,6 × time stop 270–360 minutes, 1 MNQ.",
        "- `compression_breakout` : vote de 27 cellules, lookback 8–12 × buffer 0–2 ticks × cible 1,5–2,5 × range, 1 MNQ.",
        "- `atr_trend_flip` : vote de 9 cellules, ATR 8–12 × multiple 1,5–2,0, stop fixe 1 ATR par cellule, 1 MNQ.",
        "- Equal weight : 25 % de chaque PnL quotidien.",
        "- Les entrées ERC sont les rendements nets quotidiens calculés comme PnL / 50 000 USD. Aucune optimisation ERC n'est exécutée.",
        f"- SHA-256 configuration : `{config_hash}`.",
        f"- SHA-256 règles MM : `{mm_hash}`.", "", "## Métriques", "",
        "| Sleeve | Trades | PnL net | Sharpe | PF journalier | Max DD | Max DD % | Vol annualisée |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        trades = "n/a" if pd.isna(row.trade_count) else str(int(row.trade_count))
        lines.append(
            f"| `{row.strategy}` | {trades} | {row.net_pnl_usd:,.2f} | {row.sharpe_annualized:.3f} | "
            f"{row.daily_profit_factor:.3f} | {row.max_drawdown_usd:,.2f} | {row.max_drawdown_pct:.2f}% | {row.annualized_volatility_pct:.2f}% |"
        )
    lines += ["", "## Contributions au risque de l'equal weight", "",
              "| Sleeve | Poids | Contribution au risque | Vol annualisée individuelle |",
              "|---|---:|---:|---:|"]
    for row in risk.itertuples(index=False):
        lines.append(f"| `{row.sleeve}` | {100*row.equal_weight:.1f}% | {row.risk_contribution_pct:.2f}% | {row.standalone_annualized_volatility_pct:.2f}% |")
    lines += ["", "## Inputs ERC", "",
              "`erc_daily_returns.csv`, `erc_covariance_daily.csv`, `erc_covariance_annualized.csv` et `erc_correlation.csv` sont les entrées propres de l'étape suivante.",
              "Le script ne contient ni solveur, ni fonction objectif, ni poids ERC estimés.", ""]
    (OUTPUT_DIR / "FOUR_SLEEVE_PORTFOLIO_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def execute() -> None:
    for directory in (OUTPUT_DIR, TRADES_DIR, PLOTS_DIR, MPL_CACHE):
        directory.mkdir(parents=True, exist_ok=True)
    config, config_hash, mm_locks, mm_hash = load_and_validate_config()
    validate_locks(config, mm_locks)
    data_hash_before = sha256(DATA_PATH)
    orb_path = REPO_ROOT / config["sleeves"]["orb_vwap_atr"]["source"]
    orb_hash_before = sha256(orb_path)
    if data_hash_before != config["lock_sources"]["mnq_minute_data_sha256"]:
        raise RuntimeError("Configured MNQ minute-data hash mismatch")
    if orb_hash_before != config["lock_sources"]["orb_trades_sha256"]:
        raise RuntimeError("Configured retained ORB trades hash mismatch")
    start, end = pd.Timestamp(config["start_date"]).date(), pd.Timestamp(config["end_date"]).date()

    runs, trades, checks = run_alpha_sleeves(config)
    calendar = pd.Index(runs["compression_breakout"].daily["session_date"], name="session_date")
    daily_pnl = pd.DataFrame(index=calendar)
    for name in SLEEVES[1:]:
        daily_pnl[name] = runs[name].daily.set_index("session_date")["daily_pnl_usd"].reindex(calendar).fillna(0.0)
        trades[name].to_csv(TRADES_DIR / f"{name}.csv", index=False)

    orb = pd.read_csv(orb_path)
    orb["session_date"] = pd.to_datetime(orb["session_date"]).dt.date
    orb = orb[orb["session_date"].between(start, end)].copy()
    daily_pnl["orb_vwap_atr"] = orb.groupby("session_date")["net_pnl_usd"].sum().reindex(calendar).fillna(0.0)
    daily_pnl = daily_pnl[SLEEVES]
    trades["orb_vwap_atr"] = orb
    orb.to_csv(TRADES_DIR / "orb_vwap_atr.csv", index=False)

    weights = pd.Series(config["equal_weight_reference"], dtype=float).reindex(SLEEVES)
    equal_weight_pnl = daily_pnl.mul(weights, axis=1).sum(axis=1)
    daily_output = daily_pnl.copy()
    for name in SLEEVES:
        daily_output[f"equal_weight_contribution__{name}"] = daily_pnl[name] * weights[name]
    daily_output["equal_weight_portfolio"] = equal_weight_pnl
    daily_output.reset_index().to_csv(OUTPUT_DIR / "daily_pnl.csv", index=False)

    curve_parts, metric_rows = [], []
    for name in SLEEVES:
        frame = equity_from_pnl(daily_pnl[name])
        frame.insert(0, "session_date", frame.index)
        frame.insert(1, "strategy", name)
        curve_parts.append(frame.reset_index(drop=True))
        metric_rows.append(strategy_metrics(name, equity_from_pnl(daily_pnl[name]), trades[name]))
    equal_frame = equity_from_pnl(equal_weight_pnl)
    report_equal = equal_frame.copy()
    equal_frame.insert(0, "session_date", equal_frame.index)
    equal_frame.insert(1, "strategy", "equal_weight_portfolio")
    curve_parts.append(equal_frame.reset_index(drop=True))
    metric_rows.append(strategy_metrics("equal_weight_portfolio", report_equal, None))
    curves = pd.concat(curve_parts, ignore_index=True)
    metrics_frame = pd.DataFrame(metric_rows)
    curves.to_csv(OUTPUT_DIR / "equity_curves.csv", index=False)
    metrics_frame.to_csv(OUTPUT_DIR / "sleeve_metrics.csv", index=False)

    eq = report_equal.copy()
    eq.index.name = "session_date"
    eq.to_csv(OUTPUT_DIR / "equal_weight_portfolio.csv")
    erc_returns, daily_cov, annual_cov, correlation, risk = covariance_outputs(daily_pnl, weights)
    erc_returns.index.name = "session_date"
    erc_returns.to_csv(OUTPUT_DIR / "erc_daily_returns.csv")
    save_matrix(daily_cov, OUTPUT_DIR / "erc_covariance_daily.csv")
    save_matrix(annual_cov, OUTPUT_DIR / "erc_covariance_annualized.csv")
    save_matrix(correlation, OUTPUT_DIR / "erc_correlation.csv")
    risk.to_csv(OUTPUT_DIR / "equal_weight_risk_contributions.csv", index=False)
    make_plots(curves, correlation)
    write_report(metrics_frame, risk, config_hash, mm_hash)

    if sha256(CONFIG_PATH) != config_hash or sha256(MM_LOCK_PATH) != mm_hash:
        raise RuntimeError("A locked configuration changed during execution")
    if sha256(DATA_PATH) != data_hash_before or sha256(orb_path) != orb_hash_before:
        raise RuntimeError("An input changed during execution")
    metadata = {
        "config_sha256": config_hash,
        "money_management_lock_sha256": mm_hash,
        "data_sha256": data_hash_before,
        "orb_trades_sha256": orb_hash_before,
        "period": [str(start), str(end)],
        "session_count": len(calendar),
        "strategy_count": 4,
        "equal_weight_sum": float(weights.sum()),
        "erc_optimization_performed": False,
        "all_backtests_valid": not any(implementation_violations(value) for value in checks.values()),
        "implementation_checks": checks,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(metrics_frame.to_string(index=False), flush=True)
    print(risk.to_string(index=False), flush=True)
    print(f"Config SHA-256: {config_hash}", flush=True)


if __name__ == "__main__":
    execute()
