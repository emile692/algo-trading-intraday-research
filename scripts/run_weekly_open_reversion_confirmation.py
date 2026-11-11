"""Run the frozen final confirmation for weekly-open reversion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.weekly_open_reversion_confirmation import (
    ANCHORS,
    HORIZONS,
    STOP_MULTIPLES,
    ConfirmationConfig,
    add_stability,
    bootstrap_inference,
    build_confirmation_trades,
    classify_grid,
    combine_ledgers,
    deduplicate_underlying_weeks,
    summarize_grid,
)


INPUTS = {
    "MNQ": ROOT / "data/downloaded/MNQ_c_0_1m_20260321_094501.parquet",
    "NQ": ROOT / "data/downloaded/NQ_c_0_1m_20260321_095036.parquet",
}
DEFAULT_OUTPUT = ROOT / "research/weekly_open_reversion_confirmation_20261001"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: float, digits: int = 2) -> str:
    return "∞" if value == float("inf") else f"{value:.{digits}f}"


def _report(
    classified: pd.DataFrame,
    anchor_grid: pd.DataFrame,
    verdict: str,
    topstep_gate: bool,
    deduped: pd.DataFrame,
) -> str:
    lines = [
        "# Weekly Open reversion — confirmation finale bornée",
        "",
        f"## Verdict : `{verdict}`",
        "",
        f"Le gate Topstep est **{'ouvert' if topstep_gate else 'fermé'}**. "
        + ("Une zone 2×2 contiguë satisfait tous les critères confirmatoires." if topstep_gate else "Aucune zone 2×2 contiguë ne satisfait simultanément positivité nette, stabilité temporelle, significativité et domination des deux contrôles."),
        "",
        "## Grille dédupliquée par semaine sous-jacente",
        "",
        "NQ est conservé sur les semaines MNQ/NQ communes; MNQ ne peut contribuer que sur une semaine absente de NQ. Les six sleeves actives du portefeuille n'ont pas été lus, régénérés ou modifiés.",
        "",
        "| Horizon | Stop | n | Target avant stop | BE théorique | Exp. nette USD | PF | MaxDD USD | MAE méd. D | Années + | Sous-périodes + | IC95 exp. | vs placebo | vs Mon. close | Stable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    order = {name: i for i, name in enumerate(HORIZONS)}
    for row in classified.sort_values(["horizon", "stop_multiple"], key=lambda col: col.map(order) if col.name == "horizon" else col).itertuples():
        lines.append(
            f"| `{row.horizon}` | {row.stop_multiple:g}D | {row.n} | {row.target_before_stop_rate:.1%} | {row.theoretical_break_even_rate:.1%} | "
            f"{row.net_expectancy_usd:+.2f} | {_fmt(row.profit_factor)} | {row.max_drawdown_usd:.2f} | {row.mae_median_d:.2f} | "
            f"{row.positive_year_ratio:.0%} | {int(row.positive_subperiod_count)}/{int(row.subperiod_count)} | "
            f"[{row.expectancy_ci95_low:+.2f}; {row.expectancy_ci95_high:+.2f}] | {row.delta_vs_opposite_placebo_usd:+.2f} | "
            f"{row.delta_vs_monday_close_usd:+.2f} | {'oui' if row.temporally_stable else 'non'} |"
        )
    lines += [
        "",
        "Les taux target-before-stop et break-even ne sont pas directement comparables comme un test de rentabilité, car les sorties temporelles réalisent des gains ou pertes partiels.",
        "",
        "## Comparaison économique aux ancres de contrôle",
        "",
        "| Horizon | Stop | Weekly Open exp./PF (n) | Placebo opposé exp./PF (n) | Monday Close exp./PF (n) |",
        "|---|---:|---:|---:|---:|",
    ]
    indexed = anchor_grid.set_index(["anchor", "horizon", "stop_multiple"])
    for horizon in HORIZONS:
        for stop in STOP_MULTIPLES:
            cells = []
            for anchor in ANCHORS:
                row = indexed.loc[(anchor, horizon, stop)]
                cells.append(f"{row.net_expectancy_usd:+.2f} / {_fmt(row.profit_factor)} ({int(row.n)})")
            lines.append(f"| `{horizon}` | {stop:g}D | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Protocole verrouillé",
        "",
        "- Entrée : open mardi 09:30 ET; target principal : Weekly Open lundi 09:30 ET.",
        "- Stops : 0,5D / 1D / 1,5D / 2D / 3D; horizons : mardi 11:30, mardi 16:00, mercredi 16:00 ET; Globex inclus.",
        "- Exécution : 1 tick défavorable par côté et 1,25 USD de commission par côté, avec les multiplicateurs contractuels MNQ/NQ du dépôt.",
        "- Ordering : si stop et target sont touchés dans la même minute, le stop prévaut.",
        "- Contrôles : target placebo équidistante à l'opposé et ancre Monday Close (close de 15:59 ET).",
        "- Sous-périodes fixées ex ante : 2010–2014, 2015–2019, 2020–2026. Stabilité : les 3 sous-périodes positives et au moins 60 % des années positives.",
        "- Inférence : moving-block bootstrap de 5 000 tirages, blocs de 8 semaines, seed 20261001; différences aux contrôles appariées par semaine.",
        "- Gate robuste : rectangle contigu 2 stops × 2 horizons avec expectancy nette > 0, PF > 1, stabilité temporelle, borne IC95 d'expectancy > 0 et bornes IC95 des deux écarts aux contrôles > 0.",
        "",
        f"Semaines uniques utilisées pour l'inférence : **{deduped.loc[deduped['anchor'].eq('weekly_open'), 'week_monday'].nunique()}**.",
        "",
        "## Décision Topstep",
        "",
        "Le simulateur Topstep " + ("peut être lancé dans une phase séparée, sans retuning." if topstep_gate else "n'est pas lancé, conformément au gate pré-spécifié."),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = ConfirmationConfig()
    studies = []
    for symbol, path in INPUTS.items():
        studies.append(build_confirmation_trades(pd.read_parquet(path), symbol=symbol, config=config))
    trades, audit = combine_ledgers(studies)
    deduped = deduplicate_underlying_weeks(trades)

    instrument_keys = ["symbol", "anchor", "horizon", "stop_multiple"]
    instrument_grid = summarize_grid(trades, instrument_keys)
    dedup_keys = ["anchor", "horizon", "stop_multiple"]
    dedup_grid = summarize_grid(deduped, dedup_keys)
    dedup_grid, annual, subperiod = add_stability(deduped, dedup_grid, dedup_keys)
    inference = bootstrap_inference(deduped, config=config)
    weekly_grid = dedup_grid.loc[dedup_grid["anchor"] == "weekly_open"].drop(columns="anchor")
    classified, verdict, topstep_gate = classify_grid(weekly_grid, inference, config=config)
    sleeve_metrics = summarize_grid(
        deduped.loc[deduped["anchor"] == "weekly_open"],
        ["direction", "distance_bucket", "horizon", "stop_multiple"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.output_dir / "trade_ledger_all_instruments.csv", index=False)
    deduped.to_csv(args.output_dir / "trade_ledger_deduplicated.csv", index=False)
    audit.to_csv(args.output_dir / "exclusion_audit.csv", index=False)
    instrument_grid.to_csv(args.output_dir / "instrument_anchor_grid.csv", index=False)
    classified.to_csv(args.output_dir / "weekly_open_confirmatory_grid.csv", index=False)
    annual.to_csv(args.output_dir / "annual_stability.csv", index=False)
    subperiod.to_csv(args.output_dir / "subperiod_stability.csv", index=False)
    sleeve_metrics.to_csv(args.output_dir / "frozen_bucket_direction_metrics.csv", index=False)
    inference.to_csv(args.output_dir / "block_bootstrap_inference.csv", index=False)
    (args.output_dir / "REPORT.md").write_text(
        _report(classified, dedup_grid, verdict, topstep_gate, deduped), encoding="utf-8"
    )
    topstep_reason = None
    if not topstep_gate:
        topstep_reason = (
            "no positive and temporally stable contiguous 2x2 stop/horizon zone"
            if verdict == "REJECTED"
            else "positive stable zone did not satisfy full robust significance/control gate"
        )
    metadata = {
        "verdict": verdict,
        "topstep_gate_open": topstep_gate,
        "topstep_simulation_run": False,
        "reason_topstep_not_run": topstep_reason,
        "source_study": "research/weekly_open_reversion_20261001",
        "frozen_sleeves_modified": False,
        "stop_multiples": STOP_MULTIPLES,
        "horizons": HORIZONS,
        "anchors": ANCHORS,
        "execution": {"slippage_ticks_per_side": 1, "commission_per_side_usd": 1.25},
        "inference": {"deduplication": "prefer NQ per common underlying week", "bootstrap_samples": config.bootstrap_samples, "block_weeks": config.bootstrap_block_weeks, "seed": config.random_seed},
        "inputs": {symbol: {"path": str(path.resolve()), "sha256": _sha256(path)} for symbol, path in INPUTS.items()},
        "included_source_weeks": {symbol: int(audit.loc[(audit.symbol == symbol) & (audit.status == "included")].shape[0]) for symbol in INPUTS},
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "topstep_gate_open": topstep_gate, "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
