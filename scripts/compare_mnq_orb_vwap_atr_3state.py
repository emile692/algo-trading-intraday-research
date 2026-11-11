"""Compare retained MNQ ORB trades with historical 15/60 volatility sizing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.metrics import compute_metrics  # noqa: E402
from src.analytics.mnq_orb_regime_filter_sizing_campaign import _scale_nominal_trades_by_multiplier  # noqa: E402
from src.analytics.orb_research.campaign import _build_incremental_experiments, _evaluate_experiment, _split_sessions  # noqa: E402
from src.analytics.orb_research.features import (  # noqa: E402
    attach_daily_reference, build_candidate_universe, build_daily_reference, prepare_minute_dataset,
)
from src.analytics.orb_research.types import BaselineEnsembleConfig, BaselineEntryConfig, CampaignContext  # noqa: E402
from src.config.settings import DEFAULT_TICK_VALUE_USD  # noqa: E402
from src.engine.execution_model import ExecutionModel  # noqa: E402

DATASET = ROOT / "data/processed/parquet/MNQ_c_0_1m_20260321_094501.parquet"
OUTPUT = ROOT / "export/mnq_orb_vwap_atr_3state_comparison"
LOW = 0.9429454367121718
MID = 1.1407799880685539


def summary(trades: pd.DataFrame, sessions: list, capital: float) -> dict:
    metrics = compute_metrics(trades, session_dates=sessions, initial_capital=capital)
    result = {key: metrics[key] for key in (
        "n_trades", "sharpe_ratio", "max_drawdown", "profit_factor", "win_rate", "expectancy"
    )}
    result["net_pnl"] = metrics["cumulative_pnl"]
    return result


def fmt(value: object, decimals: int = 2) -> str:
    return f"{float(value):,.{decimals}f}"


def main() -> None:
    entry = BaselineEntryConfig()
    ensemble = BaselineEnsembleConfig()
    minute = prepare_minute_dataset(DATASET, entry, (10, 14, 20, 30))
    minute = attach_daily_reference(minute, build_daily_reference(minute))
    candidates = build_candidate_universe(minute, baseline_entry=entry)
    sessions = sorted(pd.to_datetime(minute["session_date"]).dt.date.unique())
    is_sessions, oos_sessions = _split_sessions(sessions, 0.70)
    context = CampaignContext(sessions, is_sessions, oos_sessions, minute, candidates, None)
    experiment = _build_incremental_experiments(entry, ensemble)[0]
    row, detail = _evaluate_experiment(experiment, context, bootstrap_paths=0, random_seed=42, keep_details=True)
    if row["status"] != "ok" or detail is None:
        raise RuntimeError(f"Baseline failed: {row['status']}")
    nominal = detail["trades"].copy()
    prior = json.loads((ROOT / "export/mnq_orb_vwap_atr_retained/summary.json").read_text(encoding="utf-8"))
    if len(nominal) != prior["metrics"]["overall"]["n_trades"]:
        raise RuntimeError("Nominal trades differ from retained report")

    # The feature is causal and read on the breakout signal bar, before next-open execution.
    returns = minute["close"].pct_change()
    vol15 = returns.rolling(15).std()
    vol60 = returns.rolling(60).std()
    minute_features = minute[["session_date", "timestamp"]].copy()
    minute_features["ratio_15_60"] = vol15.div(vol60.where(vol60.ne(0)))
    signal_rows = detail["selected_final"][["session_date", "timestamp"]].copy()
    signal_rows = signal_rows.merge(minute_features, on=["session_date", "timestamp"], how="left", validate="one_to_one")
    signal_rows["bucket"] = np.select(
        [signal_rows["ratio_15_60"].le(LOW), signal_rows["ratio_15_60"].le(MID)],
        ["low", "mid"], default="high",
    )
    signal_rows.loc[signal_rows["ratio_15_60"].isna(), "bucket"] = "fallback_neutral"
    signal_rows["risk_multiplier"] = signal_rows["bucket"].map(
        {"low": 0.50, "mid": 1.00, "high": 0.25, "fallback_neutral": 1.00}
    )
    signal_rows["session_date"] = pd.to_datetime(signal_rows["session_date"]).dt.date
    nominal["session_date"] = pd.to_datetime(nominal["session_date"]).dt.date
    controls = nominal[["session_date"]].merge(
        signal_rows[["session_date", "ratio_15_60", "bucket", "risk_multiplier"]],
        on="session_date", how="left", validate="one_to_one",
    )
    if controls["risk_multiplier"].isna().any():
        raise RuntimeError("A traded session has no 3-state signal context")

    execution = ExecutionModel()
    scaled = _scale_nominal_trades_by_multiplier(
        nominal_trades=nominal,
        controls=controls,
        account_size_usd=entry.account_size_usd,
        base_risk_pct=entry.risk_per_trade_pct,
        tick_value_usd=DEFAULT_TICK_VALUE_USD,
        point_value_usd=DEFAULT_TICK_VALUE_USD / execution.tick_size,
        commission_per_side_usd=execution.commission_per_side_usd,
    )
    if scaled.empty:
        raise RuntimeError("No trades remain after sizing")
    both = {}
    for name, trades in (("sans_3state", nominal), ("avec_3state", scaled)):
        both[name] = {}
        for scope, scope_sessions in (("is", is_sessions), ("oos", oos_sessions), ("overall", sessions)):
            selected = trades.loc[trades["session_date"].isin(set(scope_sessions))]
            both[name][scope] = summary(selected, scope_sessions, entry.account_size_usd)
    for scope in ("is", "oos", "overall"):
        if abs(both["sans_3state"][scope]["net_pnl"] - prior["metrics"][scope]["net_pnl"]) > 1e-6:
            raise RuntimeError(f"Nominal PnL mismatch for {scope}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    controls.to_csv(OUTPUT / "signal_buckets.csv", index=False)
    scaled.to_csv(OUTPUT / "trades_3state.csv", index=False)
    (OUTPUT / "metrics.json").write_text(json.dumps(both, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        "# ORB + VWAP + ATR — avec et sans sizing 3-state",
        "",
        "Même signal et mêmes dates de marché que le rapport `mnq_orb_vwap_atr_retained`. Le sizing 3-state est calculé sur la barre de breakout, avant l'entrée à l'ouverture suivante. Les prix d'entrée, stops et sorties ne changent pas ; les quantités sont recalculées par contrat entier avec le moteur de sizing de recherche.",
        "",
        f"Ratio = écart-type glissant des rendements 15 barres / 60 barres. Seuils : low ≤ {LOW}, mid ≤ {MID}, high au-delà. Multiplicateurs historiques : low 0,50× ; mid 1,00× ; high 0,25×. Feature indisponible : repli neutre 1,00×.",
        f"Capital {fmt(entry.account_size_usd, 0)} $ ; budget nominal {fmt(entry.account_size_usd * entry.risk_per_trade_pct / 100)} $ par trade ; commissions {fmt(execution.commission_per_side_usd)} $ par côté ; slippage {fmt(execution.slippage_ticks, 0)} tick par côté.",
        f"IS {is_sessions[0]}–{is_sessions[-1]} ({len(is_sessions)} séances) ; OOS {oos_sessions[0]}–{oos_sessions[-1]} ({len(oos_sessions)} séances).",
        "",
        "| Période | Variante | Trades | PnL net ($) | Sharpe | Drawdown max ($) | Profit factor |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope, label in (("is", "IS"), ("oos", "OOS"), ("overall", "Ensemble")):
        for name, variant in (("sans_3state", "Sans"), ("avec_3state", "Avec")):
            m = both[name][scope]
            lines.append(f"| {label} | {variant} | {int(m['n_trades'])} | {fmt(m['net_pnl'])} | {fmt(m['sharpe_ratio'], 3)} | {fmt(m['max_drawdown'])} | {fmt(m['profit_factor'], 3)} |")
    counts = controls["bucket"].value_counts().to_dict()
    lines += [
        "", "## Distribution des états sur les trades nominaux", "",
        *[f"- {bucket} : {counts.get(bucket, 0)}" for bucket in ("low", "mid", "high", "fallback_neutral")],
        "", "## Lecture et limites", "",
        "- Comparaison sur une même stratégie de base. Une quantité inférieure à un contrat supprime le trade ; les trades conservés gardent les mêmes prix et motifs de sortie.",
        "- Les seuils et multiplicateurs proviennent d'une ancienne campagne sur une autre branche ORB. Ils ne sont pas recalibrés ici sur la stratégie simple : ce test est une évaluation historique de transfert, pas une validation indépendante du choix de l'overlay.",
        "- Le Sharpe inclut les séances sans trade selon le moteur de recherche. Les coûts restent fixes ; le comportement du broker réel et les limites de position ne sont pas simulés ici.",
        "- La base de sizing du dépôt d'exécution est de 150 000 $, contre 50 000 $ ici. Les PnL ne se transposent pas directement à cette configuration.",
        "", "## Reproduction", "",
        "`python scripts/compare_mnq_orb_vwap_atr_3state.py`", "",
    ]
    (OUTPUT / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT / "comparison_report.md")


if __name__ == "__main__":
    main()
