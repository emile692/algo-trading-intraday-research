"""Replay the locked wide grids on temporal validation and compare them with IS."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import deque
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
TABLE_DIR = OUTPUT_DIR / "tables"
MPL_CACHE = OUTPUT_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from research.baseline_backtests.baseline_engine import (  # noqa: E402
    build_buy_and_hold,
    compute_metrics,
    load_frozen_specs,
    prepare_data,
    run_baseline_backtest,
    slice_prepared_data,
)
from research.wide_parameter_landscape.run_wide_parameter_landscape import (  # noqa: E402
    _apply_parameters,
    _combinations,
)


GRID_PATH = REPO_ROOT / "research" / "wide_parameter_landscape" / "wide_parameter_grids.json"
GRID_METADATA = REPO_ROOT / "research" / "wide_parameter_landscape" / "run_metadata.json"
IS_RESULTS_PATH = REPO_ROOT / "research" / "wide_parameter_landscape" / "wide_parameter_landscape_results.csv"
RULES_PATH = OUTPUT_DIR / "consensus_analysis_rules.json"
FROZEN_DIR = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"
VALIDATION_RESULTS_PATH = OUTPUT_DIR / "validation_landscape_results.csv"
CONSENSUS_RESULTS_PATH = OUTPUT_DIR / "is_validation_consensus_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "is_validation_consensus_summary.csv"
REPORT_PATH = OUTPUT_DIR / "IS_VALIDATION_CONSENSUS_REPORT.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_locked_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    actual_grid_hash = _sha256(GRID_PATH)
    prior_metadata = json.loads(GRID_METADATA.read_text(encoding="utf-8"))
    if actual_grid_hash != rules["locked_grid_sha256"] or actual_grid_hash != prior_metadata["grid_sha256"]:
        raise RuntimeError("The previously locked wide grid SHA-256 does not match")
    config = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    if set(config["mechanisms"]) != {
        "atr_trend_flip", "compression_breakout", "gap_continuation",
        "ma_crossover_trend", "session_range_breakout", "volatility_breakout",
    }:
        raise RuntimeError("Locked grid mechanism set changed")
    return config, rules


def _frozen_hashes() -> dict[str, str]:
    paths = sorted(FROZEN_DIR.glob("*.json")) + [FROZEN_MANIFEST]
    return {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in paths}


def _checkpoint_matches(path: Path, mechanism_id: str, definition: dict[str, Any], grid_hash: str) -> bool:
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    expected = _combinations(mechanism_id, definition)
    names = list(definition["parameters"])
    actual_pairs = {(str(row.parameter_1_value), str(row.parameter_2_value)) for row in frame.itertuples(index=False)}
    expected_pairs = {(str(item[names[0]]), str(item[names[1]])) for item in expected}
    return (
        len(frame) == len(expected)
        and actual_pairs == expected_pairs
        and frame["backtest_valid"].astype(bool).all()
        and (frame["locked_grid_sha256"].astype(str) == grid_hash).all()
    )


def _validation_frame(
    mechanism_id: str,
    definition: dict[str, Any],
    spec: dict[str, Any],
    prepared: Any,
    benchmark_daily: pd.DataFrame,
    validation_start: date,
    validation_end: date,
    grid_hash: str,
    completed: int,
    total: int,
) -> tuple[pd.DataFrame, int]:
    names = list(definition["parameters"])
    rows: list[dict[str, Any]] = []
    for number, values in enumerate(_combinations(mechanism_id, definition), 1):
        completed += 1
        print(f"[{completed:03d}/{total}] {mechanism_id} {values}", flush=True)
        selected = _apply_parameters(spec, values)
        run = run_baseline_backtest(
            selected,
            prepared,
            evaluation_start=validation_start,
            evaluation_end=validation_end,
        )
        metrics = compute_metrics(selected, run, benchmark_daily)
        if not metrics["backtest_valid"]:
            raise RuntimeError(f"Validation checks failed for {mechanism_id} {values}")
        rows.append({
            "mechanism_id": mechanism_id,
            "strategy_id": selected["strategy_id"],
            "combination_id": f"{mechanism_id}__{number:03d}",
            "parameter_1_name": names[0],
            "parameter_1_value": values[names[0]],
            "parameter_1_index": definition["parameters"][names[0]].index(values[names[0]]),
            "parameter_2_name": names[1],
            "parameter_2_value": values[names[1]],
            "parameter_2_index": definition["parameters"][names[1]].index(values[names[1]]),
            "validation_sharpe_annualized": metrics["sharpe_annualized"],
            "validation_expectancy_per_trade_usd": metrics["expectancy_per_trade_usd"],
            "validation_profit_factor": metrics["profit_factor"],
            "validation_max_drawdown_usd": metrics["max_drawdown_usd"],
            "validation_max_drawdown_pct": metrics["max_drawdown_pct"],
            "validation_trade_count": metrics["trade_count"],
            "validation_net_pnl_usd": metrics["net_pnl_usd"],
            "implementation_warning": metrics["implementation_warning"],
            "backtest_valid": metrics["backtest_valid"],
            "frozen_spec_sha256": metrics["frozen_spec_sha256"],
            "locked_grid_sha256": grid_hash,
            "validation_start": validation_start.isoformat(),
            "validation_end": validation_end.isoformat(),
        })
    return pd.DataFrame(rows), completed


def _merge_is_validation(is_frame: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    is_columns = [
        "combination_id", "sharpe_annualized", "expectancy_per_trade_usd", "profit_factor",
        "max_drawdown_usd", "max_drawdown_pct", "trade_count", "net_pnl_usd",
    ]
    renamed = is_frame[is_columns].rename(columns={
        "sharpe_annualized": "is_sharpe_annualized",
        "expectancy_per_trade_usd": "is_expectancy_per_trade_usd",
        "profit_factor": "is_profit_factor",
        "max_drawdown_usd": "is_max_drawdown_usd",
        "max_drawdown_pct": "is_max_drawdown_pct",
        "trade_count": "is_trade_count",
        "net_pnl_usd": "is_net_pnl_usd",
    })
    result = validation.merge(renamed, on="combination_id", how="left", validate="one_to_one")
    if result["is_sharpe_annualized"].isna().any():
        raise RuntimeError("Missing IS cell during validation merge")
    result["consensus_score"] = result[["is_sharpe_annualized", "validation_sharpe_annualized"]].min(axis=1)
    result["robust_cell"] = (
        (result["is_sharpe_annualized"] > 0)
        & (result["validation_sharpe_annualized"] > 0)
        & (result["is_profit_factor"] > 1)
        & (result["validation_profit_factor"] > 1)
    )
    result["is_positive_cell"] = (result["is_sharpe_annualized"] > 0) & (result["is_profit_factor"] > 1)
    result["sharpe_change_validation_minus_is"] = result["validation_sharpe_annualized"] - result["is_sharpe_annualized"]
    result["expectancy_change_validation_minus_is_usd"] = result["validation_expectancy_per_trade_usd"] - result["is_expectancy_per_trade_usd"]
    result["profit_factor_change_validation_minus_is"] = result["validation_profit_factor"] - result["is_profit_factor"]
    return result


def _components(frame: pd.DataFrame, column: str) -> list[pd.DataFrame]:
    selected = frame[frame[column].astype(bool)]
    coordinate_to_index = {
        (int(row.parameter_1_index), int(row.parameter_2_index)): index
        for index, row in selected.iterrows()
    }
    unseen = set(coordinate_to_index)
    components: list[pd.DataFrame] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = deque([start])
        indices: list[int] = []
        while queue:
            x, y = queue.popleft()
            indices.append(coordinate_to_index[(x, y)])
            for other in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if other in unseen:
                    unseen.remove(other)
                    queue.append(other)
        components.append(frame.loc[indices].copy())
    return sorted(components, key=lambda item: (-len(item), item["combination_id"].min()))


def _region(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "{}"
    return json.dumps({
        frame["parameter_1_name"].iloc[0]: sorted(frame["parameter_1_value"].unique().tolist()),
        frame["parameter_2_name"].iloc[0]: sorted(frame["parameter_2_value"].unique().tolist()),
    }, ensure_ascii=False, sort_keys=True)


def _medoid(component: pd.DataFrame) -> pd.Series | None:
    if component.empty:
        return None
    coordinates = component[["parameter_1_index", "parameter_2_index"]].to_numpy(dtype=int)
    distance_sum = np.abs(coordinates[:, None, :] - coordinates[None, :, :]).sum(axis=2).sum(axis=1)
    candidates = component.copy()
    candidates["distance_sum"] = distance_sum
    return candidates.sort_values(
        ["distance_sum", "consensus_score", "combination_id"], ascending=[True, False, True]
    ).iloc[0]


def _native(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    components = _components(frame, "robust_cell")
    largest = components[0] if components else frame.iloc[0:0]
    medoid = _medoid(largest)
    p1 = frame["parameter_1_name"].iloc[0]
    p2 = frame["parameter_2_name"].iloc[0]
    is_count = int(frame["is_positive_cell"].sum())
    robust_count = int(frame["robust_cell"].sum())
    return {
        "mechanism_id": frame["mechanism_id"].iloc[0],
        "strategy_id": frame["strategy_id"].iloc[0],
        "cell_count": len(frame),
        "is_positive_cell_count": is_count,
        "validation_positive_cell_count": int(((frame["validation_sharpe_annualized"] > 0) & (frame["validation_profit_factor"] > 1)).sum()),
        "robust_cell_count": robust_count,
        "robust_retention_from_is_pct": 100.0 * robust_count / is_count if is_count else np.nan,
        "robust_component_count": len(components),
        "largest_robust_region_size": len(largest),
        "largest_robust_region": _region(largest),
        "entire_is_zone_collapsed": bool(is_count > 0 and robust_count == 0),
        "medoid_combination_id": None if medoid is None else medoid["combination_id"],
        "medoid_parameters": None if medoid is None else json.dumps({
            p1: _native(medoid["parameter_1_value"]), p2: _native(medoid["parameter_2_value"])
        }, ensure_ascii=False, sort_keys=True),
        "medoid_consensus_score": np.nan if medoid is None else medoid["consensus_score"],
        "medoid_is_sharpe": np.nan if medoid is None else medoid["is_sharpe_annualized"],
        "medoid_validation_sharpe": np.nan if medoid is None else medoid["validation_sharpe_annualized"],
        "median_consensus_score_robust_region": np.nan if largest.empty else largest["consensus_score"].median(),
    }


def _paired_heatmap(frame: pd.DataFrame, mechanism_id: str, metric: str, title: str, fmt: str) -> None:
    p1 = frame["parameter_1_name"].iloc[0]
    p2 = frame["parameter_2_name"].iloc[0]
    is_column = f"is_{metric}"
    validation_column = f"validation_{metric}"
    left = frame.pivot(index="parameter_1_value", columns="parameter_2_value", values=is_column)
    right = frame.pivot(index="parameter_1_value", columns="parameter_2_value", values=validation_column)
    combined = np.concatenate([left.to_numpy().ravel(), right.to_numpy().ravel()])
    finite = combined[np.isfinite(combined)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if len(finite) else (None, None)
    center = 1.0 if metric == "profit_factor" else (0.0 if metric in {"sharpe_annualized", "expectancy_per_trade_usd", "max_drawdown_usd"} else None)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True)
    for ax, values, label in zip(axes, (left, right), ("IS 2019–2023", "Validation 2024–2026")):
        sns.heatmap(values, annot=True, fmt=fmt, cmap="RdYlGn", center=center, vmin=vmin, vmax=vmax,
                    linewidths=0.4, cbar=True, ax=ax)
        ax.set_title(label)
        ax.set_xlabel(p2)
        ax.set_ylabel(p1 if ax is axes[0] else "")
    fig.suptitle(f"{mechanism_id} — {title}")
    fig.tight_layout()
    fig.savefig(HEATMAP_DIR / f"{mechanism_id}_{metric}_is_vs_validation.png", dpi=160)
    plt.close(fig)


def _consensus_heatmaps(frame: pd.DataFrame, mechanism_id: str) -> None:
    p1 = frame["parameter_1_name"].iloc[0]
    p2 = frame["parameter_2_name"].iloc[0]
    score = frame.pivot(index="parameter_1_value", columns="parameter_2_value", values="consensus_score")
    robust = frame.pivot(index="parameter_1_value", columns="parameter_2_value", values="robust_cell").astype(float)
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    sns.heatmap(score, annot=True, fmt=".2f", cmap="RdYlGn", center=0, linewidths=0.4, ax=ax)
    ax.set_title(f"{mechanism_id}\nConsensus score = min(Sharpe IS, Sharpe Validation)")
    ax.set_xlabel(p2); ax.set_ylabel(p1); fig.tight_layout()
    fig.savefig(HEATMAP_DIR / f"{mechanism_id}_consensus_score.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    sns.heatmap(robust, annot=True, fmt=".0f", cmap=["#d9d9d9", "#238b45"], vmin=0, vmax=1,
                linewidths=0.4, cbar=False, ax=ax)
    ax.set_title(f"{mechanism_id}\nRobust cell commune IS / Validation")
    ax.set_xlabel(p2); ax.set_ylabel(p1); fig.tight_layout()
    fig.savefig(HEATMAP_DIR / f"{mechanism_id}_robust_mask.png", dpi=160); plt.close(fig)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value): return "n/a"
    if isinstance(value, (float, np.floating)): return f"{float(value):.{digits}f}"
    return str(value)


def _write_report(summary: pd.DataFrame, rules: dict[str, Any], metadata: dict[str, Any]) -> None:
    lines = [
        "# Consensus IS / Validation temporelle", "", "## Protocole", "",
        "Les wide grids déjà verrouillées ont été rejouées sans modification sur **2024-01-01 → 2026-03-19**, "
        "désormais appelée période de validation temporelle. Aucun axe, paramètre, coût, fill, sizing ou règle de "
        "session n'a été modifié. L'historique IS sert uniquement au warmup causal des indicateurs.", "",
        f"- SHA-256 de la grille réutilisée : `{metadata['locked_grid_sha256']}`",
        f"- Cellules : **{metadata['cell_count']}** ; valides : **{metadata['valid_cell_count']}**",
        "- `consensus_score = min(Sharpe_IS, Sharpe_VAL)`",
        "- `robust_cell = Sharpe_IS > 0 and Sharpe_VAL > 0 and PF_IS > 1 and PF_VAL > 1`",
        "- Contiguïté : quatre voisins orthogonaux dans la grille verrouillée", "",
        "## Synthèse", "",
        "| Mécanisme | Zone + IS | Zone + VAL | Cellules robustes | Plus grande zone | Rétention IS | Médoïde | Score consensus | Effondrement total |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in summary.sort_values("mechanism_id").itertuples(index=False):
        lines.append(
            f"| `{row.mechanism_id}` | {row.is_positive_cell_count} | {row.validation_positive_cell_count} | "
            f"{row.robust_cell_count} | {row.largest_robust_region_size} | {_fmt(row.robust_retention_from_is_pct)}% | "
            f"`{row.medoid_parameters if pd.notna(row.medoid_parameters) else 'n/a'}` | {_fmt(row.medoid_consensus_score)} | "
            f"{'oui' if row.entire_is_zone_collapsed else 'non'} |"
        )
    lines.extend(["", "## Détail", ""])
    for row in summary.sort_values("mechanism_id").itertuples(index=False):
        lines.extend([
            f"### {row.mechanism_id}", "",
            f"- Plus grande région robuste : `{row.largest_robust_region}` ({row.largest_robust_region_size} cellules)",
            f"- Médoïde : `{row.medoid_parameters if pd.notna(row.medoid_parameters) else 'n/a'}`",
            f"- Sharpe du médoïde : IS {_fmt(row.medoid_is_sharpe)} ; Validation {_fmt(row.medoid_validation_sharpe)} ; consensus {_fmt(row.medoid_consensus_score)}",
            f"- Rétention de la zone positive IS : {_fmt(row.robust_retention_from_is_pct)}%",
            f"- Effondrement complet : **{'oui' if row.entire_is_zone_collapsed else 'non'}**", "",
        ])
    lines.extend([
        "## Fichiers", "",
        "Les heatmaps `*_is_vs_validation.png` utilisent la même échelle de couleur dans les deux panneaux. Les cartes "
        "`*_consensus_score.png` et `*_robust_mask.png` montrent directement les régions communes. Les tables détaillées "
        "conservent les cinq métriques pour chaque cellule.", "",
        "Cette étape cartographie la stabilité temporelle des grilles. Elle ne sélectionne pas automatiquement un nouveau "
        "point et ne recrée pas de période OOS après observation.", "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def execute(data_path: Path, resume: bool = True) -> None:
    for directory in (OUTPUT_DIR, HEATMAP_DIR, TABLE_DIR, MPL_CACHE): directory.mkdir(parents=True, exist_ok=True)
    config, rules = _load_locked_inputs()
    grid_hash = rules["locked_grid_sha256"]
    frozen_before = _frozen_hashes()
    specs, benchmark_spec, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    spec_by_id = {spec["mechanism_id"]: spec for spec in specs}
    is_results = pd.read_csv(IS_RESULTS_PATH)
    validation_start = date.fromisoformat(rules["validation_start"])
    validation_end = date.fromisoformat(rules["validation_end"])
    prepared = prepare_data(data_path, start_date=date.fromisoformat(rules["is_start"]), end_date=validation_end)
    validation_data = slice_prepared_data(prepared, validation_start, validation_end)
    benchmark = build_buy_and_hold(benchmark_spec, validation_data)
    total = sum(len(_combinations(mid, definition)) for mid, definition in config["mechanisms"].items())
    completed = 0
    validation_frames: list[pd.DataFrame] = []
    consensus_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for mechanism_id, definition in config["mechanisms"].items():
        checkpoint = TABLE_DIR / f"{mechanism_id}_validation.csv"
        if resume and _checkpoint_matches(checkpoint, mechanism_id, definition, grid_hash):
            validation = pd.read_csv(checkpoint)
            completed += len(validation)
            print(f"[{completed:03d}/{total}] {mechanism_id} resumed from verified checkpoint", flush=True)
        else:
            validation, completed = _validation_frame(
                mechanism_id, definition, spec_by_id[mechanism_id], prepared, benchmark.daily,
                validation_start, validation_end, grid_hash, completed, total,
            )
            validation.to_csv(checkpoint, index=False)
        is_frame = is_results[is_results["mechanism_id"] == mechanism_id].copy()
        consensus = _merge_is_validation(is_frame, validation)
        consensus.to_csv(TABLE_DIR / f"{mechanism_id}_consensus.csv", index=False)
        for metric, title, fmt in [
            ("sharpe_annualized", "Sharpe annualisé", ".2f"),
            ("expectancy_per_trade_usd", "Expectancy par trade (USD)", ".1f"),
            ("profit_factor", "Profit factor", ".2f"),
            ("max_drawdown_usd", "Max drawdown (USD)", ".0f"),
            ("trade_count", "Nombre de trades", ".0f"),
        ]:
            _paired_heatmap(consensus, mechanism_id, metric, title, fmt)
        _consensus_heatmaps(consensus, mechanism_id)
        validation_frames.append(validation)
        consensus_frames.append(consensus)
        summaries.append(_summary(consensus))
    validation_results = pd.concat(validation_frames, ignore_index=True)
    consensus_results = pd.concat(consensus_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    validation_results.to_csv(VALIDATION_RESULTS_PATH, index=False)
    consensus_results.to_csv(CONSENSUS_RESULTS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    frozen_after = _frozen_hashes()
    if frozen_before != frozen_after: raise RuntimeError("A frozen baseline changed")
    if _sha256(GRID_PATH) != grid_hash: raise RuntimeError("The locked grid changed during validation")
    metadata = {
        "run_type": "is_temporal_validation_consensus", "locked_grid_sha256": grid_hash,
        "data_path": str(data_path.relative_to(REPO_ROOT)), "data_sha256": _sha256(data_path),
        "is_start": rules["is_start"], "is_end": rules["is_end"],
        "validation_start": rules["validation_start"], "validation_end": rules["validation_end"],
        "period_label": "temporal_validation_not_true_oos", "cell_count": len(consensus_results),
        "valid_cell_count": int(validation_results["backtest_valid"].astype(bool).sum()),
        "grid_expanded": False, "parameters_added": False, "post_observation_tuning": False,
        "frozen_files_unchanged": True, "frozen_sha256": frozen_before,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(summary, rules, metadata)
    print(f"Validation cells executed: {len(validation_results)}", flush=True)
    print(f"Valid validation cells: {int(validation_results['backtest_valid'].astype(bool).sum())}", flush=True)
    print(f"Mechanisms with complete IS-zone collapse: {int(summary['entire_is_zone_collapsed'].sum())}", flush=True)
    print("Grid expansions / added parameters / tuning: 0", flush=True)
    print("Frozen files modified: 0", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    execute(args.data.resolve(), resume=not args.no_resume)


if __name__ == "__main__":
    main()
