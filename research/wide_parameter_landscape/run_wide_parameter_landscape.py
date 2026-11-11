"""Run the predeclared wide IS landscapes for six frozen MNQ mechanisms."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
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
)


FROZEN_DIR = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
GRID_PATH = OUTPUT_DIR / "wide_parameter_grids.json"
DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"
RESULT_PATH = OUTPUT_DIR / "wide_parameter_landscape_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "wide_parameter_landscape_summary.csv"
BOUNDARY_PATH = OUTPUT_DIR / "boundary_transitions.csv"
SENSITIVITY_PATH = OUTPUT_DIR / "parameter_sensitivity.csv"
REPORT_PATH = OUTPUT_DIR / "WIDE_PARAMETER_LANDSCAPE_REPORT.md"
PENALTY = 0.5


def _native(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_hashes() -> dict[str, str]:
    paths = sorted(FROZEN_DIR.glob("*.json")) + [FROZEN_MANIFEST]
    return {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in paths}


def _load_config() -> dict[str, Any]:
    config = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    if not config.get("locked_before_execution"):
        raise ValueError("Wide grid must be locked before execution")
    if len(config["mechanisms"]) != 6:
        raise ValueError("Wide study must contain exactly six mechanisms")
    return config


def _valid_combination(mechanism_id: str, values: dict[str, Any]) -> bool:
    if mechanism_id == "ma_crossover_trend":
        return int(values["fast_ema_period"]) < int(values["slow_ema_period"])
    return True


def _combinations(mechanism_id: str, definition: dict[str, Any]) -> list[dict[str, Any]]:
    names = list(definition["parameters"])
    axes = [definition["parameters"][name] for name in names]
    values = [dict(zip(names, item)) for item in itertools.product(*axes)]
    return [item for item in values if _valid_combination(mechanism_id, item)]


def _apply_parameters(frozen_spec: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(frozen_spec)
    structural = spec["canonical_parameters"]["structural"]
    for name, value in values.items():
        structural[name] = value
    return spec


def _neighbors(frame: pd.DataFrame, row: pd.Series, chebyshev: bool) -> pd.DataFrame:
    dx = (frame["parameter_1_index"] - int(row["parameter_1_index"])).abs()
    dy = (frame["parameter_2_index"] - int(row["parameter_2_index"])).abs()
    mask = (np.maximum(dx, dy) <= 1) if chebyshev else ((dx + dy) == 1)
    return frame.loc[mask]


def _add_local_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    columns: dict[str, list[Any]] = {
        "neighborhood_size": [], "neighborhood_sharpe_median": [], "neighborhood_sharpe_std": [],
        "sharpe_stability_score": [], "neighborhood_expectancy_median": [],
        "positive_neighbor_fraction": [], "neighborhood_min_trade_count": [], "robust_plateau_cell": [],
    }
    for _, row in result.iterrows():
        neighbors = _neighbors(result, row, chebyshev=True)
        neighbors = neighbors[neighbors["backtest_valid"].astype(bool)]
        sharpes = pd.to_numeric(neighbors["sharpe_annualized"], errors="coerce").dropna()
        expectations = pd.to_numeric(neighbors["expectancy_per_trade_usd"], errors="coerce").dropna()
        median = float(sharpes.median()) if len(sharpes) else np.nan
        std = float(sharpes.std(ddof=0)) if len(sharpes) else np.nan
        score = median - PENALTY * std if len(sharpes) else np.nan
        positive_fraction = float(neighbors["positive_cell"].astype(bool).mean()) if len(neighbors) else np.nan
        min_trades = int(neighbors["trade_count"].min()) if len(neighbors) else 0
        robust = bool(
            row["positive_cell"] and len(neighbors) >= 4 and score > 0
            and positive_fraction >= (2.0 / 3.0) and min_trades >= 30
        )
        values = [len(neighbors), median, std, score,
                  float(expectations.median()) if len(expectations) else np.nan,
                  positive_fraction, min_trades, robust]
        for name, value in zip(columns, values):
            columns[name].append(value)
    for name, values in columns.items():
        result[name] = values
    return result


def _components(frame: pd.DataFrame, selector: str) -> list[list[int]]:
    selected = frame[frame[selector].astype(bool)]
    coordinate_to_index = {
        (int(row.parameter_1_index), int(row.parameter_2_index)): int(index)
        for index, row in selected.iterrows()
    }
    unseen = set(coordinate_to_index)
    components: list[list[int]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = deque([start])
        indices: list[int] = []
        while queue:
            coordinate = queue.popleft()
            indices.append(coordinate_to_index[coordinate])
            x, y = coordinate
            for other in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if other in unseen:
                    unseen.remove(other)
                    queue.append(other)
        components.append(indices)
    return sorted(components, key=lambda item: (-len(item), min(item)))


def _parameter_region(frame: pd.DataFrame, parameter_names: list[str]) -> str:
    region = {
        name: sorted(frame[f"parameter_{number}_value"].unique().tolist())
        for number, name in enumerate(parameter_names, 1)
    }
    return json.dumps(region, ensure_ascii=False, sort_keys=True)


def _medoid(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    coordinates = frame[["parameter_1_index", "parameter_2_index"]].to_numpy(dtype=int)
    distances = np.abs(coordinates[:, None, :] - coordinates[None, :, :]).sum(axis=2).sum(axis=1)
    candidates = frame.copy()
    candidates["distance_sum"] = distances
    candidates = candidates.sort_values(
        ["distance_sum", "sharpe_stability_score", "combination_id"],
        ascending=[True, False, True],
    )
    return candidates.iloc[0]


def _boundary_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_coordinate = {
        (int(row.parameter_1_index), int(row.parameter_2_index)): row
        for row in frame.itertuples(index=False)
    }
    for coordinate, source in sorted(by_coordinate.items()):
        if not bool(source.positive_cell):
            continue
        x, y = coordinate
        for target_coordinate in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            target = by_coordinate.get(target_coordinate)
            if target is None or bool(target.positive_cell):
                continue
            pair_key = tuple(sorted((coordinate, target_coordinate)))
            rows.append({
                "mechanism_id": source.mechanism_id,
                "transition_key": str(pair_key),
                "positive_combination_id": source.combination_id,
                "positive_parameter_1_value": source.parameter_1_value,
                "positive_parameter_2_value": source.parameter_2_value,
                "positive_sharpe": source.sharpe_annualized,
                "nonpositive_combination_id": target.combination_id,
                "nonpositive_parameter_1_value": target.parameter_1_value,
                "nonpositive_parameter_2_value": target.parameter_2_value,
                "nonpositive_sharpe": target.sharpe_annualized,
            })
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        unique[(row["mechanism_id"], row["transition_key"])] = row
    return list(unique.values())


def _sensitivity_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number in (1, 2):
        name = frame[f"parameter_{number}_name"].iloc[0]
        grouped = frame.groupby(f"parameter_{number}_value", sort=True)
        medians = grouped["sharpe_annualized"].median()
        for value, group in grouped:
            rows.append({
                "mechanism_id": frame["mechanism_id"].iloc[0],
                "parameter_name": name,
                "parameter_value": value,
                "cell_count": len(group),
                "median_sharpe": float(group["sharpe_annualized"].median()),
                "median_expectancy_per_trade_usd": float(group["expectancy_per_trade_usd"].median()),
                "positive_cell_fraction": float(group["positive_cell"].astype(bool).mean()),
                "conditional_median_sharpe_range": float(medians.max() - medians.min()),
            })
    return rows


def _summarize(frame: pd.DataFrame) -> dict[str, Any]:
    parameter_names = [frame["parameter_1_name"].iloc[0], frame["parameter_2_name"].iloc[0]]
    positive_components = _components(frame, "positive_cell")
    robust_components = _components(frame, "robust_plateau_cell")
    largest_positive = frame.loc[positive_components[0]] if positive_components else frame.iloc[0:0]
    largest_robust = frame.loc[robust_components[0]] if robust_components else frame.iloc[0:0]
    robust_detected = len(largest_robust) >= 2
    medoid = _medoid(largest_robust) if robust_detected else None
    maximum = frame.sort_values(["sharpe_annualized", "combination_id"], ascending=[False, True]).iloc[0]
    max_p1 = int(frame["parameter_1_index"].max())
    max_p2 = int(frame["parameter_2_index"].max())
    on_boundary_axes: list[str] = []
    if int(maximum["parameter_1_index"]) in (0, max_p1): on_boundary_axes.append(parameter_names[0])
    if int(maximum["parameter_2_index"]) in (0, max_p2): on_boundary_axes.append(parameter_names[1])
    sensitivity = _sensitivity_rows(frame)
    ranges = {}
    for row in sensitivity:
        ranges[row["parameter_name"]] = row["conditional_median_sharpe_range"]
    dominant = sorted(ranges, key=lambda name: (-ranges[name], name))[0]
    exterior_positive = frame[
        frame["positive_cell"].astype(bool)
        & ((frame["parameter_1_index"].isin([0, max_p1])) | (frame["parameter_2_index"].isin([0, max_p2])))
    ]
    return {
        "mechanism_id": frame["mechanism_id"].iloc[0],
        "strategy_id": frame["strategy_id"].iloc[0],
        "combination_count": len(frame),
        "valid_combination_count": int(frame["backtest_valid"].astype(bool).sum()),
        "positive_cell_count": int(frame["positive_cell"].astype(bool).sum()),
        "positive_component_count": len(positive_components),
        "largest_positive_component_size": len(largest_positive),
        "largest_positive_region": _parameter_region(largest_positive, parameter_names) if len(largest_positive) else "{}",
        "observed_edge_boundary_transition_count": len(_boundary_rows(frame)),
        "positive_cells_touch_grid_exterior": bool(len(exterior_positive)),
        "robust_plateau_detected": robust_detected,
        "robust_plateau_cell_count": int(frame["robust_plateau_cell"].astype(bool).sum()),
        "largest_robust_component_size": len(largest_robust),
        "robust_plateau_region": _parameter_region(largest_robust, parameter_names) if len(largest_robust) else "{}",
        "plateau_medoid_combination_id": None if medoid is None else medoid["combination_id"],
        "plateau_medoid_parameters": None if medoid is None else json.dumps({
            parameter_names[0]: _native(medoid["parameter_1_value"]),
            parameter_names[1]: _native(medoid["parameter_2_value"]),
        }, ensure_ascii=False, sort_keys=True),
        "plateau_medoid_sharpe": np.nan if medoid is None else medoid["sharpe_annualized"],
        "plateau_medoid_stability_score": np.nan if medoid is None else medoid["sharpe_stability_score"],
        "highest_sharpe_combination_id": maximum["combination_id"],
        "highest_sharpe": maximum["sharpe_annualized"],
        "highest_sharpe_parameters": json.dumps({
            parameter_names[0]: _native(maximum["parameter_1_value"]),
            parameter_names[1]: _native(maximum["parameter_2_value"]),
        }, ensure_ascii=False, sort_keys=True),
        "highest_sharpe_on_tested_boundary": bool(on_boundary_axes),
        "highest_sharpe_boundary_axes": ",".join(on_boundary_axes),
        "dominant_sensitivity_parameter": dominant,
        "parameter_1_median_sharpe_range": ranges[parameter_names[0]],
        "parameter_2_median_sharpe_range": ranges[parameter_names[1]],
    }


def _plot(frame: pd.DataFrame, mechanism_id: str, metric: str, title: str, fmt: str) -> None:
    p1 = frame["parameter_1_name"].iloc[0]
    p2 = frame["parameter_2_name"].iloc[0]
    pivot = frame.pivot(index="parameter_1_value", columns="parameter_2_value", values=metric)
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    center = 0 if metric in {"sharpe_annualized", "expectancy_per_trade_usd", "max_drawdown_usd"} else None
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap="RdYlGn", center=center, linewidths=0.4,
                cbar_kws={"label": title}, ax=ax)
    ax.set_title(f"{mechanism_id}\n{title} — wide IS 2019-05-06 à 2023-12-31")
    ax.set_xlabel(p2)
    ax.set_ylabel(p1)
    fig.tight_layout()
    fig.savefig(HEATMAP_DIR / f"{mechanism_id}_{metric}.png", dpi=160)
    plt.close(fig)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value): return "n/a"
    if isinstance(value, (float, np.floating)): return f"{float(value):.{digits}f}"
    return str(value)


def _write_report(config: dict[str, Any], results: pd.DataFrame, summary: pd.DataFrame, metadata: dict[str, Any]) -> None:
    lines = [
        "# Wide Parameter Landscape — six mécanismes MNQ", "",
        "## Cadre verrouillé", "",
        "Cette cartographie utilise uniquement l'IS **2019-05-06 → 2023-12-31**. La période "
        "**2024-01-01 → 2026-03-19** reste réservée. Les grilles sont exactement celles demandées et ont été "
        "enregistrées avant exécution dans `wide_parameter_grids.json`.", "",
        f"- Combinaisons : **{len(results)}** ; valides : **{int(results['backtest_valid'].astype(bool).sum())}**",
        f"- SHA-256 de la grille : `{metadata['grid_sha256']}`",
        "- Coûts, sizing, règles de risque, sessions et exécution : inchangés",
        "- Frozen baselines modifiés : aucun", "- OOS utilisée : non", "",
        "## Lecture des zones", "",
        "Une cellule positive exige simultanément Sharpe > 0, expectancy > 0, des checks valides et PF > 1; un PF "
        "indéfini faute de trade perdant est accepté comme positif, puis signalé par son faible nombre de trades. Le nombre "
        "de trades est affiché séparément afin de ne pas confondre frontière de performance et faible profondeur statistique. "
        "Les composantes utilisent une adjacence orthogonale. Une cellule de plateau robuste ajoute "
        "un voisinage local d'au moins quatre cellules, un score `médiane Sharpe − 0,5 × écart-type` positif, deux "
        "tiers de voisins positifs et au moins 30 trades dans chaque voisin. Le médoïde décrit le centre géométrique "
        "de la plus grande composante robuste; ce n'est pas une sélection par Sharpe.", "",
        "## Synthèse", "",
        "| Mécanisme | Cellules + | Plus grande zone + | Frontières observées | Plateau robuste | Médoïde | Maximum sur borne | Paramètre le plus sensible |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in summary.sort_values("mechanism_id").itertuples(index=False):
        lines.append(
            f"| `{row.mechanism_id}` | {row.positive_cell_count} | {row.largest_positive_component_size} | "
            f"{row.observed_edge_boundary_transition_count} | {'oui' if row.robust_plateau_detected else 'non'} | "
            f"`{row.plateau_medoid_parameters if pd.notna(row.plateau_medoid_parameters) else 'n/a'}` | "
            f"{'oui' if row.highest_sharpe_on_tested_boundary else 'non'} | `{row.dominant_sensitivity_parameter}` |"
        )
    lines.extend(["", "## Détail par mécanisme", ""])
    for mechanism_id, definition in config["mechanisms"].items():
        row = summary[summary["mechanism_id"] == mechanism_id].iloc[0]
        lines.extend([
            f"### {mechanism_id}", "",
            f"- Grille : `{json.dumps(definition['parameters'], ensure_ascii=False)}`",
            f"- Zone positive contiguë principale : `{row['largest_positive_region']}` ({int(row['largest_positive_component_size'])} cellules)",
            f"- Transitions observées vers Sharpe/expectancy/PF non positifs : **{int(row['observed_edge_boundary_transition_count'])}**",
            f"- La zone positive touche une limite de grille : **{'oui' if row['positive_cells_touch_grid_exterior'] else 'non'}**",
            f"- Plateau robuste : **{'oui' if row['robust_plateau_detected'] else 'non'}** ; région : `{row['robust_plateau_region']}`",
            f"- Médoïde : `{row['plateau_medoid_parameters'] if pd.notna(row['plateau_medoid_parameters']) else 'n/a'}` ; score local : {_fmt(row['plateau_medoid_stability_score'])}",
            f"- Maximum observé sur une borne : **{'oui' if row['highest_sharpe_on_tested_boundary'] else 'non'}** ({row['highest_sharpe_boundary_axes'] or 'aucun axe'})",
            f"- Sensibilité dominante : `{row['dominant_sensitivity_parameter']}`",
            f"- Table complète : `tables/{mechanism_id}.csv`", "",
        ])
    lines.extend([
        "## Fichiers d'audit", "",
        "- `boundary_transitions.csv` contient chaque passage orthogonal observé d'une cellule positive à une cellule non positive.",
        "- `parameter_sensitivity.csv` contient les médianes conditionnelles par valeur de paramètre.",
        "- Cinq heatmaps sont produites par mécanisme : Sharpe, expectancy, PF, max drawdown et nombre de trades.", "",
        "## Limite", "",
        "Une zone qui reste positive jusqu'à l'extérieur de la grille a une frontière non observée de ce côté. Le rapport "
        "la signale sans extrapoler. Aucun paramètre n'est validé pour utilisation future avant l'étape OOS réservée.", "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _checkpoint_matches(path: Path, mechanism_id: str, definition: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    expected = _combinations(mechanism_id, definition)
    names = list(definition["parameters"])
    actual_pairs = {
        (str(row.parameter_1_value), str(row.parameter_2_value))
        for row in frame.itertuples(index=False)
    }
    expected_pairs = {(str(item[names[0]]), str(item[names[1]])) for item in expected}
    return len(frame) == len(expected) and actual_pairs == expected_pairs and frame["backtest_valid"].astype(bool).all()


def execute(data_path: Path, resume: bool = True) -> None:
    for directory in (OUTPUT_DIR, HEATMAP_DIR, TABLE_DIR, MPL_CACHE): directory.mkdir(parents=True, exist_ok=True)
    config = _load_config()
    frozen_before = _frozen_hashes()
    specs, benchmark_spec, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    spec_by_id = {spec["mechanism_id"]: spec for spec in specs}
    prepared = prepare_data(
        data_path,
        start_date=date.fromisoformat(config["is_start"]),
        end_date=date.fromisoformat(config["is_end"]),
    )
    benchmark = build_buy_and_hold(benchmark_spec, prepared)
    total = sum(len(_combinations(mid, definition)) for mid, definition in config["mechanisms"].items())
    completed = 0
    frames: list[pd.DataFrame] = []
    boundary_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for mechanism_id, definition in config["mechanisms"].items():
        names = list(definition["parameters"])
        rows: list[dict[str, Any]] = []
        combinations = _combinations(mechanism_id, definition)
        checkpoint = TABLE_DIR / f"{mechanism_id}.csv"
        if resume and _checkpoint_matches(checkpoint, mechanism_id, definition):
            frame = pd.read_csv(checkpoint)
            completed += len(frame)
            print(f"[{completed:03d}/{total}] {mechanism_id} resumed from verified checkpoint", flush=True)
        else:
            for number, values in enumerate(combinations, 1):
                completed += 1
                print(f"[{completed:03d}/{total}] {mechanism_id} {values}", flush=True)
                spec = _apply_parameters(spec_by_id[mechanism_id], values)
                run = run_baseline_backtest(spec, prepared)
                metrics = compute_metrics(spec, run, benchmark.daily)
                if not metrics["backtest_valid"]:
                    raise RuntimeError(f"Invalid implementation checks: {mechanism_id} {values}")
                row = {
                "mechanism_id": mechanism_id, "strategy_id": spec["strategy_id"],
                "complexity_grade": spec["complexity_grade"], "combination_id": f"{mechanism_id}__{number:03d}",
                "parameter_1_name": names[0], "parameter_1_value": values[names[0]],
                "parameter_1_index": definition["parameters"][names[0]].index(values[names[0]]),
                "parameter_2_name": names[1], "parameter_2_value": values[names[1]],
                "parameter_2_index": definition["parameters"][names[1]].index(values[names[1]]),
                "sharpe_annualized": metrics["sharpe_annualized"],
                "expectancy_per_trade_usd": metrics["expectancy_per_trade_usd"],
                "profit_factor": metrics["profit_factor"], "max_drawdown_usd": metrics["max_drawdown_usd"],
                "max_drawdown_pct": metrics["max_drawdown_pct"], "trade_count": metrics["trade_count"],
                "net_pnl_usd": metrics["net_pnl_usd"], "implementation_warning": metrics["implementation_warning"],
                "backtest_valid": metrics["backtest_valid"], "frozen_spec_sha256": metrics["frozen_spec_sha256"],
                "is_start": config["is_start"], "is_end": config["is_end"],
                "reserved_oos_start": config["reserved_oos_start"], "reserved_oos_end": config["reserved_oos_end"],
                }
                row["positive_cell"] = bool(
                    row["sharpe_annualized"] > 0 and row["expectancy_per_trade_usd"] > 0
                    and (row["profit_factor"] > 1 or (pd.isna(row["profit_factor"]) and row["trade_count"] > 0))
                )
                rows.append(row)
            frame = pd.DataFrame(rows)
        frame["positive_cell"] = (
            (pd.to_numeric(frame["sharpe_annualized"], errors="coerce") > 0)
            & (pd.to_numeric(frame["expectancy_per_trade_usd"], errors="coerce") > 0)
            & ((pd.to_numeric(frame["profit_factor"], errors="coerce") > 1)
               | (pd.to_numeric(frame["profit_factor"], errors="coerce").isna()
                  & (pd.to_numeric(frame["trade_count"], errors="coerce") > 0)))
            & frame["backtest_valid"].astype(bool)
        )
        frame = _add_local_statistics(frame)
        frame.to_csv(TABLE_DIR / f"{mechanism_id}.csv", index=False)
        for metric, title, fmt in [
            ("sharpe_annualized", "Sharpe annualisé", ".2f"),
            ("expectancy_per_trade_usd", "Expectancy par trade (USD)", ".1f"),
            ("profit_factor", "Profit factor", ".2f"),
            ("max_drawdown_usd", "Max drawdown (USD)", ".0f"),
            ("trade_count", "Nombre de trades", ".0f"),
        ]:
            _plot(frame, mechanism_id, metric, title, fmt)
        frames.append(frame)
        boundary_rows.extend(_boundary_rows(frame))
        sensitivity_rows.extend(_sensitivity_rows(frame))
        summaries.append(_summarize(frame))
    results = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    results.to_csv(RESULT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    pd.DataFrame(boundary_rows).to_csv(BOUNDARY_PATH, index=False)
    pd.DataFrame(sensitivity_rows).to_csv(SENSITIVITY_PATH, index=False)
    frozen_after = _frozen_hashes()
    if frozen_before != frozen_after: raise RuntimeError("A frozen baseline changed during the wide study")
    metadata = {
        "run_type": "wide_parameter_landscape_is_only", "grid_sha256": _sha256(GRID_PATH),
        "data_path": str(data_path.relative_to(REPO_ROOT)), "data_sha256": _sha256(data_path),
        "is_start": config["is_start"], "is_end": config["is_end"],
        "reserved_oos_start": config["reserved_oos_start"], "reserved_oos_end": config["reserved_oos_end"],
        "oos_used": False, "parquet_filter_pushdown": True, "combination_count": len(results),
        "valid_combination_count": int(results["backtest_valid"].astype(bool).sum()),
        "frozen_files_unchanged": True, "frozen_sha256": frozen_before,
        "risk_execution_session_sizing_fixed": True, "variants_opened": 0, "orb_used": False,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(config, results, summary, metadata)
    print(f"Combinations executed: {len(results)}", flush=True)
    print(f"Valid combinations: {int(results['backtest_valid'].astype(bool).sum())}", flush=True)
    print(f"Robust plateaus: {int(summary['robust_plateau_detected'].astype(bool).sum())}", flush=True)
    print("OOS used: 0", flush=True)
    print("Frozen files modified: 0", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    execute(args.data.resolve(), resume=not args.no_resume)


if __name__ == "__main__": main()
