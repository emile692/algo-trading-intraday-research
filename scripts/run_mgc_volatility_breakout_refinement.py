"""Refine the MGC volatility-breakout landscape around its positive coarse zone."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from research.mgc_wide_parameter_landscape import run_mgc_wide_landscape as campaign


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "parquet" / "MGC_v_0_1m_20150101_20260923.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "research" / "mgc_volatility_breakout_refinement_v0_20260925"
COARSE_GRID = (
    REPO_ROOT
    / "research"
    / "mgc_wide_parameter_landscape_v0_20260924"
    / "mgc_wide_parameter_grids.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_grid() -> dict:
    source = json.loads(COARSE_GRID.read_text(encoding="utf-8"))
    definition = source["mechanisms"]["volatility_breakout"]
    definition["values_1"] = [3, 5, 7, 10, 14, 20, 30, 40, 50]
    definition["values_2"] = [0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    definition["rationale"] = (
        "Follow-up refinement selected after the coarse landscape: daily ATR periods retain the full "
        "coarse positive span and add intermediate horizons; entry multiples densely cover 0.55-1.30 "
        "around the coarse 0.6/0.8/1.2 positive region. Validation is therefore diagnostic, not pristine."
    )
    return {
        "schema_version": "1.0",
        "locked_before_execution": True,
        "selection_basis": (
            "Adaptive follow-up to the 2026-09-24 coarse MGC landscape. The grid was fixed before this "
            "refinement run, but its bounds were informed by prior IS and Validation results."
        ),
        "periods": source["periods"],
        "common_rules": source["common_rules"],
        "mechanisms": {"volatility_breakout": definition},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be >= 1")

    data_path = args.data.resolve()
    output_dir = args.output_dir.resolve()
    heatmap_dir = output_dir / "heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    config = build_grid()
    grid_path = output_dir / "volatility_breakout_refinement_grid.json"
    grid_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    grid_hash = sha256(grid_path)
    (output_dir / "volatility_breakout_refinement_grid.sha256").write_text(
        f"{grid_hash}  {grid_path.name}\n", encoding="ascii"
    )

    definition = config["mechanisms"]["volatility_breakout"]
    combinations = campaign._combinations(definition)
    periods = config["periods"]
    full_start = date.fromisoformat(periods["is"][0])
    full_end = date.fromisoformat(periods["validation"][1])
    source_hashes_before = campaign._source_hashes()

    tasks = [("volatility_breakout", definition, combo) for combo in combinations]
    rows: list[dict] = []
    print(f"Locked refinement grid SHA-256: {grid_hash}", flush=True)
    print(f"Cells to execute: {len(tasks)} with {args.workers} worker(s)", flush=True)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=campaign._init_worker,
        initargs=(str(data_path), full_start.isoformat(), full_end.isoformat(), periods),
    ) as pool:
        futures = {pool.submit(campaign._process_cell, task): task[2] for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            combo = futures[future]
            rows.extend(future.result())
            print(f"[{completed:03d}/{len(tasks)}] {combo[2]}/{combo[3]}", flush=True)

    if source_hashes_before != campaign._source_hashes():
        raise RuntimeError("A frozen source spec, baseline config, or coarse grid changed during execution")

    results = pd.DataFrame(rows).sort_values(
        ["parameter_1_index", "parameter_2_index", "period"]
    ).reset_index(drop=True)
    result_path = output_dir / "volatility_breakout_refinement_results.csv"
    results.to_csv(result_path, index=False)
    wide = campaign._wide_results(results)

    campaign.HEATMAP_DIR = heatmap_dir
    campaign._plot_mechanism(wide, "volatility_breakout", definition)

    wide["minimum_sharpe"] = wide[["sharpe_annualized_is", "sharpe_annualized_validation"]].min(axis=1)
    wide["mean_sharpe"] = wide[["sharpe_annualized_is", "sharpe_annualized_validation"]].mean(axis=1)
    ranking = wide.sort_values(
        ["minimum_sharpe", "mean_sharpe", "trade_count_validation"], ascending=False
    )
    ranking.to_csv(output_dir / "volatility_breakout_refinement_ranking.csv", index=False)
    best = ranking.iloc[0]
    positive = int(wide["positive_both"].sum())
    component = campaign._largest_component(wide)

    metadata = {
        "run_type": "mgc_volatility_breakout_adaptive_refinement_is_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grid_sha256": grid_hash,
        "data_path": str(data_path.relative_to(REPO_ROOT)),
        "data_sha256": sha256(data_path),
        "periods": periods,
        "combination_count": len(combinations),
        "result_row_count": len(results),
        "workers": args.workers,
        "parameter_selection_performed": True,
        "validation_status": "reused_for_adaptive_refinement_not_pristine_oos",
        "money_management_changed": False,
        "source_files_unchanged": True,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = f"""# MGC — Volatility Breakout, raffinement local IS / Validation

## Protocole

- IS : `{periods['is'][0]} → {periods['is'][1]}`.
- Validation : `{periods['validation'][0]} → {periods['validation'][1]}`.
- Grille : `{definition['values_1']}` × `{definition['values_2']}` ({len(combinations)} cellules).
- Signaux, sorties, coûts, fills, session, taille et money management inchangés.
- Ce raffinement a été défini après lecture de la campagne large : la Validation est réutilisée et ne constitue donc plus un OOS vierge.
- SHA-256 de la grille : `{grid_hash}`.

## Résultat descriptif

- Cellules positives simultanément : `{positive}/{len(wide)}`.
- Plus grande composante positive contiguë : `{len(component)}` cellules.
- Meilleure cellule selon le minimum des Sharpes IS/Validation : `daily_atr_period={best['parameter_1_value']:g}`, `entry_atr_multiple={best['parameter_2_value']:g}`.
- Sharpe IS / Validation : `{best['sharpe_annualized_is']:.3f}` / `{best['sharpe_annualized_validation']:.3f}`.
- Trades IS / Validation : `{int(best['trade_count_is'])}` / `{int(best['trade_count_validation'])}`.

Le classement complet utilise d'abord `min(Sharpe IS, Sharpe Validation)`, puis la moyenne des deux Sharpes. Il est descriptif et ne transforme pas cette Validation réutilisée en nouveau test hors-échantillon.
"""
    (output_dir / "VOLATILITY_BREAKOUT_REFINEMENT_REPORT.md").write_text(report, encoding="utf-8")
    print(f"Positive both: {positive}/{len(wide)}; largest component: {len(component)}", flush=True)
    print(
        "Best robust cell: "
        f"period={best['parameter_1_value']:g}, multiple={best['parameter_2_value']:g}, "
        f"Sharpe={best['sharpe_annualized_is']:.3f}/{best['sharpe_annualized_validation']:.3f}, "
        f"trades={int(best['trade_count_is'])}/{int(best['trade_count_validation'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
