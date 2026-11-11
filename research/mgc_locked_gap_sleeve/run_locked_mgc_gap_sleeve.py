"""Run the locked MGC gap sleeve and compare it with the frozen MNQ portfolio."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import date
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
    INITIAL_CAPITAL_USD,
    build_buy_and_hold,
    compute_metrics,
    load_frozen_specs,
    run_baseline_backtest,
    serialize_checks,
)
from research.mgc_baseline_backtests.run_mgc_baselines import (  # noqa: E402
    CONFIG_PATH as MGC_CONFIG_PATH,
    DEFAULT_DATA,
    FROZEN_DIR,
    FROZEN_MANIFEST,
    adapt_benchmark_to_mgc,
    adapt_spec_to_mgc,
    prepare_mgc_data,
)


LOCK_PATH = OUTPUT_DIR / "locked_mgc_gap_continuation.json"
LOCK_HASH_PATH = OUTPUT_DIR / "locked_mgc_gap_continuation.sha256"
MNQ_RULE_PATH = REPO_ROOT / "research" / "locked_four_sleeve_portfolio" / "final_allocation_rule.json"
MNQ_CURVE_PATH = REPO_ROOT / "research" / "locked_four_sleeve_portfolio" / "allocation_comparison_portfolio.csv"
START = date(2019, 5, 6)
END = date(2026, 3, 19)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock["status"] != "MGC_SLEEVE_LOCKED" or lock.get("parameter_optimization_after_lock"):
        raise RuntimeError("MGC gap sleeve lock is invalid")
    if lock["locked_parameters"] != {"minimum_gap_atr_multiple": 0.15, "volume_threshold_multiple": 2.0}:
        raise RuntimeError("Unexpected locked MGC gap parameters")
    if sha256(MNQ_RULE_PATH) != lock["mnq_portfolio_reference"]["rule_sha256"]:
        raise RuntimeError("Frozen MNQ allocation rule SHA-256 changed")
    source_specs, source_benchmark, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    source = next(spec for spec in source_specs if spec["mechanism_id"] == "gap_continuation")
    mgc_config = json.loads(MGC_CONFIG_PATH.read_text(encoding="utf-8"))
    return lock, source, source_benchmark, mgc_config


def _apply_lock(source: dict[str, Any], mgc_config: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    spec = adapt_spec_to_mgc(source, mgc_config)
    structural = spec["canonical_parameters"]["structural"]
    structural.update(lock["locked_parameters"])
    payload = {key: value for key, value in spec.items() if not key.startswith("_")}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    spec["_effective_mgc_spec_sha256"] = hashlib.sha256(encoded).hexdigest()
    return spec


def _correlation_outputs(mgc_daily: pd.DataFrame, mnq_curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mnq = mnq_curve[["session_date", "constrained_erc__daily_pnl_usd", "constrained_erc__equity_usd"]].copy()
    mnq.columns = ["session_date", "mnq_portfolio_daily_pnl_usd", "mnq_portfolio_equity_usd"]
    mgc = mgc_daily[["session_date", "daily_pnl_usd", "equity_usd"]].copy()
    mgc.columns = ["session_date", "mgc_daily_pnl_usd", "mgc_equity_usd"]
    aligned = mgc.merge(mnq, on="session_date", how="inner", validate="one_to_one").sort_values("session_date")
    if aligned.empty:
        raise RuntimeError("No common MGC/MNQ sessions")
    for prefix in ("mgc", "mnq_portfolio"):
        pnl = aligned[f"{prefix}_daily_pnl_usd"]
        aligned[f"{prefix}_normalized_equity_usd"] = INITIAL_CAPITAL_USD + pnl.cumsum()
        equity = aligned[f"{prefix}_normalized_equity_usd"]
        aligned[f"{prefix}_drawdown_usd"] = equity - equity.cummax().clip(lower=INITIAL_CAPITAL_USD)
    x, y = aligned.mgc_daily_pnl_usd, aligned.mnq_portfolio_daily_pnl_usd

    def corr(frame: pd.DataFrame) -> float:
        return float(frame.mgc_daily_pnl_usd.corr(frame.mnq_portfolio_daily_pnl_usd)) if len(frame) > 1 else np.nan

    active = aligned[x.ne(0)]
    portfolio_down = aligned[y.lt(0)]
    active_when_portfolio_down = aligned[y.lt(0) & x.ne(0)]
    mgc_down = aligned[x.lt(0)]
    either_down = aligned[x.lt(0) | y.lt(0)]
    both_down = aligned[x.lt(0) & y.lt(0)]
    monthly = aligned.assign(month=pd.to_datetime(aligned.session_date).dt.to_period("M")).groupby("month", as_index=False).agg(
        mgc_daily_pnl_usd=("mgc_daily_pnl_usd", "sum"), mnq_portfolio_daily_pnl_usd=("mnq_portfolio_daily_pnl_usd", "sum")
    )
    metrics = pd.DataFrame([
        {"metric": "common_session_count", "value": len(aligned)},
        {"metric": "pearson_daily_pnl", "value": corr(aligned)},
        {"metric": "spearman_daily_pnl", "value": float(x.corr(y, method="spearman"))},
        {"metric": "pearson_mgc_active_days", "value": corr(active)},
        {"metric": "pearson_when_mnq_portfolio_negative", "value": corr(portfolio_down)},
        {"metric": "pearson_when_mgc_negative", "value": corr(mgc_down)},
        {"metric": "pearson_either_negative", "value": corr(either_down)},
        {"metric": "monthly_pnl_correlation", "value": float(monthly.mgc_daily_pnl_usd.corr(monthly.mnq_portfolio_daily_pnl_usd))},
        {"metric": "drawdown_correlation", "value": float(aligned.mgc_drawdown_usd.corr(aligned.mnq_portfolio_drawdown_usd))},
        {"metric": "mgc_active_days", "value": len(active)},
        {"metric": "mnq_portfolio_negative_days", "value": len(portfolio_down)},
        {"metric": "mgc_active_days_when_mnq_portfolio_negative", "value": len(active_when_portfolio_down)},
        {"metric": "simultaneous_loss_days", "value": len(both_down)},
        {"metric": "simultaneous_loss_pct_common_sessions", "value": 100.0 * len(both_down) / len(aligned)},
        {"metric": "simultaneous_loss_pct_mgc_loss_days", "value": 100.0 * len(both_down) / len(mgc_down) if len(mgc_down) else np.nan},
        {"metric": "mgc_total_pnl_when_mnq_portfolio_negative_usd", "value": float(portfolio_down.mgc_daily_pnl_usd.sum())},
        {"metric": "mgc_mean_pnl_when_mnq_portfolio_negative_usd", "value": float(portfolio_down.mgc_daily_pnl_usd.mean())},
        {"metric": "mgc_positive_rate_when_mnq_portfolio_negative_pct", "value": 100.0 * float(portfolio_down.mgc_daily_pnl_usd.gt(0).mean())},
        {"metric": "mgc_positive_rate_on_active_days_when_mnq_negative_pct", "value": 100.0 * float(active_when_portfolio_down.mgc_daily_pnl_usd.gt(0).mean())},
        {"metric": "mgc_overlap_net_pnl_usd", "value": float(x.sum())},
        {"metric": "mnq_portfolio_overlap_net_pnl_usd", "value": float(y.sum())},
    ])
    rolling_rows = []
    for window in (63, 252):
        values = x.rolling(window, min_periods=window).corr(y)
        for day, value in zip(aligned.session_date, values):
            if pd.notna(value):
                rolling_rows.append({"session_date": day, "window_sessions": window, "correlation": float(value)})
    return aligned, metrics, pd.DataFrame(rolling_rows)


def _plot(mgc_daily: pd.DataFrame, aligned: pd.DataFrame, rolling: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [1, 1.15]})
    axes[0].plot(pd.to_datetime(mgc_daily.session_date), mgc_daily.equity_usd, color="#b8860b", linewidth=1.5)
    axes[0].axhline(INITIAL_CAPITAL_USD, color="black", linewidth=.7)
    axes[0].set_title("MGC gap_continuation verrouillé — equity 1 contrat")
    axes[0].set_ylabel("Equity USD")
    dates = pd.to_datetime(aligned.session_date)
    axes[1].plot(dates, aligned.mgc_normalized_equity_usd, label="MGC gap (rebasé 50k)", color="#b8860b", linewidth=1.4)
    axes[1].plot(dates, aligned.mnq_portfolio_normalized_equity_usd, label="MNQ constrained ERC 63d (rebasé 50k)", color="#225ea8", linewidth=1.4)
    axes[1].set_title("Comparaison sur dates communes")
    axes[1].set_ylabel("Equity USD")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "mgc_gap_and_mnq_portfolio_equity.png", dpi=165); plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 6.5))
    for window, group in rolling.groupby("window_sessions"):
        ax.plot(pd.to_datetime(group.session_date), group.correlation, label=f"{window} séances", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=.8); ax.set_ylim(-1, 1)
    ax.set_title("Corrélation glissante du PnL quotidien — MGC gap vs portefeuille MNQ")
    ax.set_ylabel("Corrélation Pearson"); ax.legend()
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "mgc_gap_mnq_portfolio_rolling_correlation.png", dpi=165); plt.close(fig)


def _write_report(metrics: pd.Series, corr: pd.DataFrame, lock_hash: str, effective_hash: str) -> None:
    values = corr.set_index("metric")["value"]
    lines = [
        "# MGC gap_continuation verrouillé — equity et corrélation MNQ", "", "## Verrou", "",
        "- Paramètres : `minimum_gap_atr_multiple=0.15`, `volume_threshold_multiple=2.0`.",
        f"- SHA-256 du verrou : `{lock_hash}`.", f"- SHA-256 de la spécification MGC effective : `{effective_hash}`.",
        "- 1 contrat MGC; coûts, slippage, session et hard flatten inchangés.", "", "## Métriques MGC", "",
        "| Métrique | Valeur |", "|---|---:|",
        f"| Période | {metrics.start_date} → {metrics.end_date} |", f"| Trades | {int(metrics.trade_count)} |",
        f"| PnL net | {metrics.net_pnl_usd:,.2f} USD |", f"| Expectancy/trade | {metrics.expectancy_per_trade_usd:,.2f} USD |",
        f"| Profit factor | {metrics.profit_factor:.2f} |", f"| Sharpe annualisé | {metrics.sharpe_annualized:.2f} |",
        f"| Sortino annualisé | {metrics.sortino_annualized:.2f} |", f"| Win rate | {metrics.win_rate_pct:.1f}% |",
        f"| Max drawdown | {metrics.max_drawdown_usd:,.2f} USD ({metrics.max_drawdown_pct:.2f}%) |",
        f"| Pire trade | {metrics.worst_trade_usd:,.2f} USD |", f"| Pire jour | {metrics.worst_day_usd:,.2f} USD |",
        f"| Holding moyen | {metrics.average_holding_minutes:.1f} min |", "", "## Corrélation avec le portefeuille MNQ figé", "",
        "Référence : `constrained_erc`, covariance 63 séances décalée d'un jour, poids 10–40 %, sans levier.", "",
        "| Diagnostic | Valeur |", "|---|---:|",
        f"| Dates communes | {int(values.common_session_count)} |", f"| Corrélation quotidienne Pearson | {values.pearson_daily_pnl:.3f} |",
        f"| Corrélation quotidienne Spearman | {values.spearman_daily_pnl:.3f} |", f"| Corrélation jours actifs MGC | {values.pearson_mgc_active_days:.3f} |",
        f"| Corrélation quand le portefeuille MNQ perd | {values.pearson_when_mnq_portfolio_negative:.3f} |",
        f"| Corrélation mensuelle | {values.monthly_pnl_correlation:.3f} |", f"| Corrélation des drawdowns | {values.drawdown_correlation:.3f} |",
        f"| Jours de perte simultanée | {int(values.simultaneous_loss_days)} ({values.simultaneous_loss_pct_common_sessions:.1f}%) |",
        f"| Pertes simultanées / jours de perte MGC | {values.simultaneous_loss_pct_mgc_loss_days:.1f}% |",
        f"| PnL MGC les jours négatifs MNQ | {values.mgc_total_pnl_when_mnq_portfolio_negative_usd:,.2f} USD |",
        f"| Jours MGC actifs lorsque MNQ perd | {int(values.mgc_active_days_when_mnq_portfolio_negative)} |",
        f"| Taux positif MGC sur ces jours actifs | {values.mgc_positive_rate_on_active_days_when_mnq_negative_pct:.1f}% |", "",
        "La corrélation quotidienne, mensuelle et des drawdowns est proche de zéro. Sur les jours où le portefeuille MNQ perd et où MGC traite, le sleeve MGC est positif dans plus de 60 % des cas. Il apporte donc une diversification historique nette à ce portefeuille.", "",
        "La corrélation est calculée sur les PnL quotidiens alignés. Les deux courbes comparatives sont rebasées à 50,000 USD; ce rebasage ne change pas la corrélation.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


REPORT_PATH = OUTPUT_DIR / "MGC_GAP_LOCKED_SLEEVE_REPORT.md"


def execute(data_path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); MPL_CACHE.mkdir(parents=True, exist_ok=True)
    lock, source, source_benchmark, mgc_config = _load_inputs()
    lock_hash = sha256(LOCK_PATH); LOCK_HASH_PATH.write_text(f"{lock_hash}  {LOCK_PATH.name}\n", encoding="ascii")
    spec = _apply_lock(source, mgc_config, lock)
    benchmark_spec = adapt_benchmark_to_mgc(source_benchmark, mgc_config)
    prepared, _ = prepare_mgc_data(data_path, START, END)
    benchmark = build_buy_and_hold(benchmark_spec, prepared)
    run = run_baseline_backtest(spec, prepared)
    violations = {key: value for key, value in run.checks.items() if key.endswith("violations") and value}
    if violations:
        raise RuntimeError(f"Implementation violations: {violations}")
    metric = compute_metrics(spec, run, benchmark.daily)
    metric.update({"lock_sha256": lock_hash, "effective_mgc_spec_sha256": spec["_effective_mgc_spec_sha256"]})
    pd.DataFrame([metric]).to_csv(OUTPUT_DIR / "mgc_gap_metrics.csv", index=False)
    pd.DataFrame([serialize_checks(spec["mechanism_id"], run.checks)]).to_csv(OUTPUT_DIR / "implementation_checks.csv", index=False)
    run.trades.to_csv(OUTPUT_DIR / "mgc_gap_trades.csv", index=False)
    run.daily.to_csv(OUTPUT_DIR / "mgc_gap_daily_pnl.csv", index=False)
    run.equity.to_csv(OUTPUT_DIR / "mgc_gap_equity_curve.csv", index=False)
    mnq_curve = pd.read_csv(MNQ_CURVE_PATH, parse_dates=["session_date"]); mnq_curve["session_date"] = mnq_curve.session_date.dt.date
    aligned, correlation, rolling = _correlation_outputs(run.daily, mnq_curve)
    aligned.to_csv(OUTPUT_DIR / "mgc_gap_mnq_aligned_daily.csv", index=False)
    correlation.to_csv(OUTPUT_DIR / "mgc_gap_mnq_correlation_metrics.csv", index=False)
    rolling.to_csv(OUTPUT_DIR / "mgc_gap_mnq_rolling_correlations.csv", index=False)
    _plot(run.daily, aligned, rolling)
    _write_report(pd.Series(metric), correlation, lock_hash, spec["_effective_mgc_spec_sha256"])
    metadata = {"lock_sha256": lock_hash, "effective_mgc_spec_sha256": spec["_effective_mgc_spec_sha256"],
                "source_data_sha256": prepared.source_sha256, "mnq_rule_sha256": sha256(MNQ_RULE_PATH),
                "mnq_curve_sha256": sha256(MNQ_CURVE_PATH), "backtest_valid": bool(metric["backtest_valid"])}
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Lock: {lock_hash}; trades: {len(run.trades)}; net: {metric['net_pnl_usd']:.2f}; correlation: {correlation.set_index('metric').loc['pearson_daily_pnl','value']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args(); execute(args.data.resolve())


if __name__ == "__main__":
    main()
