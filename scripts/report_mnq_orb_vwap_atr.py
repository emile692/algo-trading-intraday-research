"""Recompute and document the retained MNQ ORB + VWAP + ATR strategy."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.orb_research.campaign import (  # noqa: E402
    _build_incremental_experiments,
    _evaluate_experiment,
    _split_sessions,
)
from src.analytics.orb_research.features import (  # noqa: E402
    attach_daily_reference,
    build_candidate_universe,
    build_daily_reference,
    prepare_minute_dataset,
)
from src.analytics.orb_research.types import (  # noqa: E402
    BaselineEnsembleConfig,
    BaselineEntryConfig,
    CampaignContext,
)
from src.engine.execution_model import ExecutionModel  # noqa: E402


DATASET = ROOT / "data/processed/parquet/MNQ_c_0_1m_20260321_094501.parquet"
OUTPUT = ROOT / "export/mnq_orb_vwap_atr_retained"
AUDIT = ROOT / "strategy_registry/psf_mnq/orb_cutoff_audit/audit.json"
EXPECTED_DATA_SHA256 = "3d67f747ca7dfab84d97cdb5e8356610b0ca5ad0d97c4592e2d1b11b9eebf174"
EXPECTED_TRADES_SHA256 = "e4f5d74214127e4fa957bb40d035c08d9eb161796529b90a6f0a5322a997506c"
KEYS = ("n_trades", "net_pnl", "sharpe_ratio", "max_drawdown", "profit_factor", "win_rate", "expectancy")


def fmt(value: object, digits: int = 2) -> str:
    return f"{float(value):,.{digits}f}"


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)
    entry = BaselineEntryConfig()
    ensemble = BaselineEnsembleConfig()
    experiment = _build_incremental_experiments(entry, ensemble)[0]
    assert experiment.name == "baseline_fixed"

    minute = prepare_minute_dataset(DATASET, entry, (10, 14, 20, 30))
    minute = attach_daily_reference(minute, build_daily_reference(minute))
    candidates = build_candidate_universe(minute, baseline_entry=entry)
    sessions = sorted(pd.to_datetime(minute["session_date"]).dt.date.unique())
    is_sessions, oos_sessions = _split_sessions(sessions, 0.70)
    context = CampaignContext(
        all_sessions=sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        minute_df=minute,
        candidate_base_df=candidates,
        daily_patterns=None,
    )
    row, details = _evaluate_experiment(experiment, context, bootstrap_paths=300, random_seed=42, keep_details=True)
    if row["status"] != "ok" or details is None:
        raise RuntimeError(f"Baseline evaluation failed: {row['status']}")
    exit_times = pd.to_datetime(details["trades"]["exit_time"], utc=True).dt.tz_convert("America/New_York")
    if any(exit_time.time() > pd.Timestamp("16:00").time() for exit_time in exit_times):
        raise RuntimeError("Retained ORB has an exit after 16:00 New York")

    audit = json.loads(AUDIT.read_text(encoding="utf-8")) if AUDIT.exists() else None
    if audit is not None:
        differences = {}
        for scope in ("overall", "is", "oos"):
            for key in KEYS:
                column = f"{scope}_{key}"
                audit_key = "cumulative_pnl" if key == "net_pnl" else key
                delta = float(row[column]) - float(audit["corrected_metrics"][scope][audit_key])
                if abs(delta) > 1e-6:
                    differences[column] = delta
        if differences:
            raise RuntimeError(f"Recomputed baseline differs from cutoff audit: {differences}")

    execution = ExecutionModel()
    digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    if digest != EXPECTED_DATA_SHA256:
        raise RuntimeError(f"MNQ dataset hash mismatch: {digest}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    trades_path = OUTPUT / "trades.csv"
    details["trades"].to_csv(trades_path, index=False)
    trades_digest = hashlib.sha256(trades_path.read_bytes()).hexdigest()
    if trades_digest != EXPECTED_TRADES_SHA256:
        raise RuntimeError(f"Retained ORB trade hash mismatch: {trades_digest}")
    payload = {
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": digest,
        "strategy": "orb_vwap_atr",
        "campaign_name": experiment.name,
        "config": json.loads(row["config_json"]),
        "split": {"fraction_is": 0.70, "is_start": str(is_sessions[0]), "is_end": str(is_sessions[-1]),
                  "oos_start": str(oos_sessions[0]), "oos_end": str(oos_sessions[-1]),
                  "is_sessions": len(is_sessions), "oos_sessions": len(oos_sessions)},
        "execution_model": {"commission_per_side_usd": execution.commission_per_side_usd,
                            "slippage_ticks": execution.slippage_ticks, "tick_size": execution.tick_size},
        "metrics": {scope: {key: row[f"{scope}_{key}"] for key in KEYS} for scope in ("overall", "is", "oos")},
        "historical_export_reproduced": False,
        "cutoff_audit_reproduced": audit is not None,
        "corrected_trade_hash_reproduced": True,
        "trades_sha256": trades_digest,
        "archived_metrics_before_cutoff_fix": audit["archived_metrics"] if audit is not None else None,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    lines = [
        "# MNQ ORB + VWAP + ATR — rapport de performance",
        "",
        "Version corrigée le 18 septembre 2026 : stop et objectif sont évalués seulement avant la sortie de 16:00 New York. Voir l'[audit de la correction](../../strategy_registry/psf_mnq/orb_cutoff_audit/README.md).",
        "",
        "## Périmètre et méthode",
        "",
        "- Stratégie retenue : `orb_vwap_atr`, appelée `baseline_fixed` dans la campagne de recherche.",
        "- OR15, long-only, clôture de breakout, entrée à l'ouverture de la barre suivante, VWAP continu et ensemble ATR(14) avec vote 0,50.",
        "- Stop sous l'opening range avec tampon de 2 ticks ; objectif 2R ; sortie horaire à 16:00 (New York) ; au plus un trade par jour.",
        f"- Fichier MNQ 1 minute : `{payload['dataset']}` ; SHA-256 `{digest}`.",
        f"- IS : {is_sessions[0]} au {is_sessions[-1]} ({len(is_sessions)} séances) ; OOS : {oos_sessions[0]} au {oos_sessions[-1]} ({len(oos_sessions)} séances). Découpage chronologique 70/30.",
        f"- Capital de référence : {fmt(entry.account_size_usd, 0)} $ ; risque nominal : {fmt(entry.risk_per_trade_pct, 2)} % par trade.",
        f"- Coûts du backtest : {fmt(execution.commission_per_side_usd, 2)} $ de commission par côté et {fmt(execution.slippage_ticks, 0)} tick de slippage par côté ; tick de {fmt(execution.tick_size, 2)} point.",
        "- Même génération de signaux, jeu de données et séparation IS/OOS que la campagne ORB existante ; sorties recoupées avec l'audit du cutoff. L'export historique `baseline_fixed` précède cette correction.",
        "",
        "## Résultats nets du backtest",
        "",
        "| Période | Trades | PnL net ($) | Sharpe | Drawdown max ($) | Profit factor | Taux gagnant | Espérance/trade ($) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope, label in (("is", "IS"), ("oos", "OOS"), ("overall", "Ensemble")):
        m = payload["metrics"][scope]
        lines.append(f"| {label} | {int(m['n_trades'])} | {fmt(m['net_pnl'])} | {fmt(m['sharpe_ratio'], 3)} | {fmt(m['max_drawdown'])} | {fmt(m['profit_factor'], 3)} | {fmt(100 * m['win_rate'], 1)} % | {fmt(m['expectancy'])} |")

    lines += [
        "",
        "## Variantes historiques",
        "",
        "Les résultats de la variante complète, de la comparaison 3-state et des autres campagnes n'ont pas été rejoués après cette correction. Leur comparaison chiffrée avec la référence corrigée est suspendue.",
        "",
        "## Limites de lecture",
        "",
        "- Résultats historiques sur un seul jeu de données MNQ ; aucune performance future ou réussite de challenge n'en découle.",
        "- Les coûts sont fixes dans le moteur de recherche ; les fills, la latence, le spread et les incidents broker réels peuvent différer.",
        "- Le backtest risque 0,50 % de 50 000 $, soit 250 $ nominaux par trade. La configuration d'exécution risque 0,50 % d'une base nominale de 150 000 $, soit 750 $ avant limites et arrondis. Ces PnL ne sont donc pas une projection de la configuration d'exécution ; comparer signaux et ordres sur les mêmes barres avant toute extrapolation.",
        "- Le journal des trades et les métriques machine lisibles accompagnent ce rapport pour permettre l'audit.",
        "",
        "## Reproduction",
        "",
        "Depuis la racine du dépôt : `python scripts/report_mnq_orb_vwap_atr.py`.",
        "",
    ]
    (OUTPUT / "performance_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT / "performance_report.md")


if __name__ == "__main__":
    main()
