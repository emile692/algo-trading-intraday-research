"""Execute the 16 frozen MNQ baselines once, without parameter search."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.baseline_backtests.baseline_engine import (
    INITIAL_CAPITAL_USD,
    build_buy_and_hold,
    build_cme_equity_early_close_calendar,
    compute_metrics,
    load_frozen_specs,
    prepare_data,
    run_baseline_backtest,
    serialize_checks,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
TRADE_DIR = OUTPUT_DIR / "baseline_trade_level"
FROZEN_DIR = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.{digits}f}"
    return str(value)


def _markdown_table(results: pd.DataFrame) -> list[str]:
    lines = [
        "| Mécanisme | Grade | Trades | PnL net USD | Expectancy | Win rate | PF | Sharpe | Sortino | Max DD USD | Valide | Avertissement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results.sort_values("mechanism_id").itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.mechanism_id}`",
                    str(row.complexity_grade),
                    str(row.trade_count),
                    _fmt(row.net_pnl_usd),
                    _fmt(row.expectancy_per_trade_usd),
                    _fmt(row.win_rate_pct) + ("%" if pd.notna(row.win_rate_pct) else ""),
                    _fmt(row.profit_factor),
                    _fmt(row.sharpe_annualized),
                    _fmt(row.sortino_annualized),
                    _fmt(row.max_drawdown_usd),
                    "oui" if bool(row.backtest_valid) else "non",
                    str(row.implementation_warning).replace("|", "\\|") if pd.notna(row.implementation_warning) and row.implementation_warning else "—",
                ]
            )
            + " |"
        )
    return lines


def _write_report(
    specs: list[dict[str, Any]],
    benchmark_spec: dict[str, Any],
    results: pd.DataFrame,
    all_runs: dict[str, Any],
    prepared: Any,
    calendar_frame: pd.DataFrame,
) -> None:
    strategy_results = results[results["specification_type"] == "strategy_baseline"].copy()
    benchmark_row = results[results["mechanism_id"] == "MNQ_BUY_AND_HOLD"].iloc[0]
    session_start = prepared.sessions["session_date"].min()
    session_end = prepared.sessions["session_date"].max()
    lines = [
        "# Baseline backtest des 16 mécanismes MNQ",
        "",
        "## Cadre figé",
        "",
        "Ce tournoi exécute exactement une fois chacune des 16 spécifications `alpha_representative`. "
        "Aucun grid search, aucune variante et aucune modification de paramètre après observation des résultats n'ont été effectués. "
        "La stratégie ORB n'est ni chargée, ni exécutée, ni utilisée comme référence.",
        "",
        f"- Données : `{prepared.source_path.relative_to(REPO_ROOT).as_posix()}`",
        f"- SHA-256 des données : `{prepared.source_sha256}`",
        f"- Fenêtre commune : `{session_start}` à `{session_end}` ({len(prepared.sessions)} sessions RTH et {len(prepared.benchmark_daily)} closes CME)",
        "- Timestamps et DST : `America/New_York` avec timestamps timezone-aware",
        "- Exécution : données une minute, un contrat MNQ, sans pyramiding",
        "- Coûts : 1,25 USD par côté et un tick adverse par côté",
        "- Conflit stop/target dans une même minute : stop prioritaire",
        "- Benchmark unique : `MNQ_BUY_AND_HOLD`, courbe close-to-close continue sans daily flatten",
        "",
        "## Calendrier et sessions",
        "",
        "Les sessions régulières utilisent 09:30–16:00 et le hard flatten Topstep à 16:08. "
        "Les clôtures anticipées sont calculées par un calendrier CME equity-index déterministe : "
        "13:00 ET pour les séances de jours fériés, 13:15 ET pour la veille de l'Independence Day, "
        "Black Friday et Christmas Eve, avec les fermetures pré-RTH publiées pour Good Friday et le 9 janvier 2025. "
        "Le fichier `cme_equity_early_closes.csv` matérialise toutes les dates utilisées.",
        "",
        f"Le calendrier contient {len(calendar_frame)} dates spéciales dans la plage brute. "
        f"{int(prepared.sessions['early_close'].sum())} sessions raccourcies possèdent une ouverture RTH exploitable dans les données.",
        "",
        "## Résultats comparatifs",
        "",
        "La table est ordonnée par identifiant de mécanisme. Elle ne constitue pas un classement et n'attribue aucun statut `promising/dead`.",
        "",
        *_markdown_table(strategy_results),
        "",
        "## Benchmark MNQ_BUY_AND_HOLD",
        "",
        f"Sur la même fenêtre, la courbe de marché normalisée à {INITIAL_CAPITAL_USD:,.0f} USD termine avec un PnL de "
        f"{_fmt(benchmark_row.net_pnl_usd)} USD, un Sharpe annualisé de {_fmt(benchmark_row.sharpe_annualized)} "
        f"et un drawdown maximal de {_fmt(benchmark_row.max_drawdown_usd)} USD ({_fmt(benchmark_row.max_drawdown_pct)}%). "
        "Cette courbe conserve une exposition continue entre les closes et n'applique aucune règle Topstep.",
        "",
        "## Résultats individuels",
        "",
    ]

    spec_by_id = {spec["mechanism_id"]: spec for spec in specs}
    for row in strategy_results.sort_values("mechanism_id").itertuples(index=False):
        spec = spec_by_id[row.mechanism_id]
        run = all_runs[row.mechanism_id]
        exits = run.trades["exit_reason"].value_counts().to_dict() if len(run.trades) else {}
        exit_text = ", ".join(f"`{key}`={value}" for key, value in sorted(exits.items())) or "aucun trade"
        lines.extend(
            [
                f"### {row.mechanism_id}",
                "",
                f"- Représentant : `{row.strategy_id}` ; famille : `{row.research_family}` ; grade : `{row.complexity_grade}`",
                f"- Hypothèse : {spec['hypothesis']['statement']}",
                f"- Trades : **{row.trade_count}** ; PnL net : **{_fmt(row.net_pnl_usd)} USD** ; expectancy : **{_fmt(row.expectancy_per_trade_usd)} USD/trade**",
                f"- Win rate : {_fmt(row.win_rate_pct)}% ; payoff : {_fmt(row.payoff_ratio)} ; profit factor : {_fmt(row.profit_factor)}",
                f"- Sharpe : {_fmt(row.sharpe_annualized)} ; Sortino : {_fmt(row.sortino_annualized)} ; max drawdown : {_fmt(row.max_drawdown_usd)} USD ({_fmt(row.max_drawdown_pct)}%)",
                f"- MAE moyen : {_fmt(row.average_mae_usd)} USD ; MFE moyen : {_fmt(row.average_mfe_usd)} USD ; durée moyenne : {_fmt(row.average_holding_minutes)} minutes",
                f"- Commissions : {_fmt(row.commissions_usd)} USD ; slippage estimé : {_fmt(row.estimated_slippage_usd)} USD",
                f"- Sorties : {exit_text}",
                f"- Cohérence : **{'valide' if row.backtest_valid else 'invalide'}**. {row.implementation_warning or 'Aucun avertissement.'}",
                f"- Frozen SHA-256 : `{row.frozen_spec_sha256}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Contrôles de cohérence",
            "",
            "Pour chaque mécanisme, `implementation_checks.csv` contrôle explicitement : ordre causal signal/fill, "
            "absence de high/low futur, priorité du stop, unicité de position, absence d'overnight, respect du "
            "`last_entry_time`, absence de position après hard flatten et identité SHA-256 du frozen JSON. "
            "Les signaux sans minute d'exécution exacte sont ignorés conformément aux spécifications.",
            "",
            "## Limites de lecture",
            "",
            "- Les données sont une série continue MNQ fournie. Les rolls et ajustements éventuels peuvent affecter les signaux et le benchmark.",
            "- Les fills stop/target utilisent OHLC une minute. Quand les deux niveaux sont présents, le stop est appliqué en premier.",
            "- MAE et MFE sont des enveloppes calculées avec les extrêmes des minutes de détention, y compris les minutes d'entrée et de sortie; elles ne servent à aucune décision du moteur.",
            "- La comparaison au Buy & Hold décrit la même fenêtre, sans devenir une règle automatique de sélection.",
            "",
            "## Fichiers produits",
            "",
            "- `baseline_results.csv` : métriques agrégées des 16 stratégies et du benchmark",
            "- `baseline_daily_returns.csv` : PnL et rendements quotidiens",
            "- `baseline_equity_curves.csv` : courbes d'equity quotidiennes",
            "- `baseline_trade_level/` : trades détaillés, un CSV par stratégie",
            "- `baseline_position_intervals.csv` : intervalles de positions",
            "- `implementation_checks.csv` : contrôles et compteurs d'exécution",
            "- `run_metadata.json` : provenance, hashes et paramètres du run",
            "",
            "Aucune conclusion `promising/dead` n'est générée.",
        ]
    )
    (OUTPUT_DIR / "BASELINE_BACKTEST_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(data_path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRADE_DIR.mkdir(parents=True, exist_ok=True)
    frozen_paths = sorted(FROZEN_DIR.glob("*.json")) + [FROZEN_MANIFEST]
    frozen_before = _hashes(frozen_paths)
    specs, benchmark_spec, manifest = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    prepared = prepare_data(data_path)
    calendar_frame = build_cme_equity_early_close_calendar(
        prepared.minute.index.min().date(), prepared.minute.index.max().date()
    )
    calendar_frame.to_csv(OUTPUT_DIR / "cme_equity_early_closes.csv", index=False)

    benchmark_run = build_buy_and_hold(benchmark_spec, prepared)
    all_runs: dict[str, Any] = {"MNQ_BUY_AND_HOLD": benchmark_run}
    metric_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []

    for number, spec in enumerate(specs, start=1):
        print(f"[{number:02d}/16] {spec['mechanism_id']}", flush=True)
        run = run_baseline_backtest(spec, prepared)
        all_runs[spec["mechanism_id"]] = run
        metric_rows.append(compute_metrics(spec, run, benchmark_run.daily))
        check_rows.append(serialize_checks(spec["mechanism_id"], run.checks))
        run.trades.to_csv(TRADE_DIR / f"{spec['mechanism_id']}.csv", index=False)

    metric_rows.append(compute_metrics(benchmark_spec, benchmark_run, benchmark_run.daily))
    check_rows.append(serialize_checks("MNQ_BUY_AND_HOLD", benchmark_run.checks))
    results = pd.DataFrame(metric_rows)
    results.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False)
    pd.DataFrame(check_rows).to_csv(OUTPUT_DIR / "implementation_checks.csv", index=False)

    daily_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    identity = {spec["mechanism_id"]: spec for spec in [*specs, benchmark_spec]}
    for mechanism_id, run in all_runs.items():
        daily = run.daily.copy()
        daily.insert(1, "strategy_id", identity[mechanism_id]["strategy_id"])
        daily_frames.append(daily)
        equity = run.equity.copy()
        equity.insert(1, "strategy_id", identity[mechanism_id]["strategy_id"])
        equity_frames.append(equity)
        if not run.positions.empty:
            positions = run.positions.copy()
            positions.insert(1, "strategy_id", identity[mechanism_id]["strategy_id"])
            position_frames.append(positions)
    pd.concat(daily_frames, ignore_index=True).to_csv(OUTPUT_DIR / "baseline_daily_returns.csv", index=False)
    pd.concat(equity_frames, ignore_index=True).to_csv(OUTPUT_DIR / "baseline_equity_curves.csv", index=False)
    if position_frames:
        pd.concat(position_frames, ignore_index=True).to_csv(OUTPUT_DIR / "baseline_position_intervals.csv", index=False)
    else:
        pd.DataFrame(columns=["mechanism_id", "strategy_id", "trade_id", "session_date", "position", "quantity", "entry_time", "exit_time"]).to_csv(
            OUTPUT_DIR / "baseline_position_intervals.csv", index=False
        )

    _write_report(specs, benchmark_spec, results, all_runs, prepared, calendar_frame)
    frozen_after = _hashes(frozen_paths)
    if frozen_before != frozen_after:
        raise RuntimeError("A frozen baseline file changed during execution")

    metadata = {
        "run_type": "frozen_baseline_tournament",
        "optimization_performed": False,
        "grid_search_performed": False,
        "orb_loaded_or_used": False,
        "data_path": str(data_path.relative_to(REPO_ROOT)),
        "data_sha256": prepared.source_sha256,
        "data_rows_used": int(len(prepared.minute)),
        "raw_start_timestamp": str(prepared.minute.index.min()),
        "raw_end_timestamp": str(prepared.minute.index.max()),
        "session_start": str(prepared.sessions["session_date"].min()),
        "session_end": str(prepared.sessions["session_date"].max()),
        "session_count": int(len(prepared.sessions)),
        "benchmark_close_count": int(len(prepared.benchmark_daily)),
        "strategy_count": len(specs),
        "benchmark": "MNQ_BUY_AND_HOLD",
        "frozen_manifest_rows": int(len(manifest)),
        "frozen_files_unchanged": True,
        "frozen_sha256": {spec["mechanism_id"]: spec["_frozen_spec_sha256"] for spec in [*specs, benchmark_spec]},
        "calendar": "official_CME_equity_index_futures_calendar",
        "calendar_special_date_count": int(len(calendar_frame)),
        "early_close_sessions_used": int(prepared.sessions["early_close"].sum()),
        "initial_capital_usd": INITIAL_CAPITAL_USD,
        "backtest_valid_count": int(results.loc[results["specification_type"] == "strategy_baseline", "backtest_valid"].sum()),
        "data_warnings": prepared.data_warnings,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Strategies executed: {len(specs)}")
    print(f"Valid strategy backtests: {metadata['backtest_valid_count']}")
    print("Optimization/grid search: 0")
    print("Frozen files modified: 0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    execute(args.data.resolve())


if __name__ == "__main__":
    main()
