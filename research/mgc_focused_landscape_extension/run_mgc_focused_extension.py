"""Run three prelocked focused MGC IS/Validation landscape extensions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
MPL_CACHE = OUTPUT_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from research.baseline_backtests.baseline_engine import INITIAL_CAPITAL_USD, load_frozen_specs, run_baseline_backtest  # noqa: E402
from research.mgc_baseline_backtests.run_mgc_baselines import (  # noqa: E402
    DEFAULT_DATA,
    FROZEN_DIR,
    FROZEN_MANIFEST,
    adapt_spec_to_mgc,
    prepare_mgc_data,
)
from research.mgc_wide_parameter_landscape.run_mgc_wide_landscape import _apply_parameters  # noqa: E402


GRID_PATH = OUTPUT_DIR / "mgc_focused_extension_grids.json"
GRID_HASH_PATH = OUTPUT_DIR / "mgc_focused_extension_grids.sha256"
RESULT_PATH = OUTPUT_DIR / "mgc_focused_extension_results.csv"
REPORT_PATH = OUTPUT_DIR / "MGC_FOCUSED_EXTENSION_REPORT.md"
MGC_CONFIG_PATH = REPO_ROOT / "research" / "mgc_baseline_backtests" / "mgc_baseline_config.json"
PERIOD_ORDER = ("is", "validation")

_DATA: Any = None
_SPECS: dict[str, dict[str, Any]] = {}
_MGC_CONFIG: dict[str, Any] = {}
_PERIODS: dict[str, list[str]] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_grid() -> dict[str, Any]:
    config = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    if not config.get("locked_before_execution"):
        raise ValueError("Grid is not locked")
    expected = {"gap_continuation", "trend_pullback_ma_structure", "rolling_range_breakout"}
    if set(config["mechanisms"]) != expected:
        raise ValueError("Expected exactly the three requested mechanisms")
    for mechanism, definition in config["mechanisms"].items():
        if len(definition["values_1"]) != 7 or len(definition["values_2"]) != 7:
            raise ValueError(f"{mechanism}: expected 7x7")
    return config


def _metrics(run: Any, start: date, end: date) -> dict[str, Any]:
    daily = run.daily[run.daily["session_date"].between(start, end)].copy()
    trades = run.trades[run.trades["session_date"].between(start, end)].copy() if not run.trades.empty else run.trades.copy()
    pnl = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0).reset_index(drop=True)
    equity = INITIAL_CAPITAL_USD + pnl.cumsum()
    prior = equity.shift(1).fillna(INITIAL_CAPITAL_USD)
    returns = pnl / prior
    std = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
    sharpe = float(returns.mean() / std * math.sqrt(252)) if np.isfinite(std) and std > 0 else np.nan
    peak = pd.concat([pd.Series([INITIAL_CAPITAL_USD]), equity], ignore_index=True).cummax().iloc[1:].reset_index(drop=True)
    drawdown = equity - peak
    return {
        "sharpe_annualized": sharpe,
        "trade_count": int(len(trades)),
        "max_drawdown_usd": float(drawdown.min()) if len(drawdown) else 0.0,
        "session_count": int(len(daily)),
    }


def _init_worker(data_path: str, start: str, end: str, periods: dict[str, list[str]]) -> None:
    global _DATA, _SPECS, _MGC_CONFIG, _PERIODS
    specs, _, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    _SPECS = {spec["mechanism_id"]: spec for spec in specs}
    _MGC_CONFIG = json.loads(MGC_CONFIG_PATH.read_text(encoding="utf-8"))
    _PERIODS = periods
    _DATA, _ = prepare_mgc_data(Path(data_path), date.fromisoformat(start), date.fromisoformat(end))


def _worker(task: tuple[str, dict[str, Any], int, int, Any, Any]) -> list[dict[str, Any]]:
    mechanism, definition, i, j, value_1, value_2 = task
    spec = adapt_spec_to_mgc(_SPECS[mechanism], _MGC_CONFIG)
    spec = _apply_parameters(spec, definition, value_1, value_2)
    run = run_baseline_backtest(spec, _DATA)
    violation_keys = [key for key in run.checks if key.endswith("violations")]
    violations = {key: run.checks[key] for key in violation_keys if run.checks[key]}
    if violations:
        raise RuntimeError(f"{mechanism} {value_1}/{value_2}: {violations}")
    rows = []
    for period in PERIOD_ORDER:
        start, end = (date.fromisoformat(value) for value in _PERIODS[period])
        rows.append(
            {
                "mechanism_id": mechanism,
                "strategy_id": spec["strategy_id"],
                "combination_id": f"{mechanism}__{i:02d}_{j:02d}",
                "parameter_1_name": definition["parameter_1"],
                "parameter_1_value": value_1,
                "parameter_1_index": i,
                "parameter_2_name": definition["parameter_2"],
                "parameter_2_value": value_2,
                "parameter_2_index": j,
                "period": period,
                **_metrics(run, start, end),
                "backtest_valid": True,
            }
        )
    return rows


def _wide(results: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    identity = [
        "mechanism_id", "strategy_id", "combination_id", "parameter_1_name", "parameter_1_value",
        "parameter_1_index", "parameter_2_name", "parameter_2_value", "parameter_2_index",
    ]
    wide = results.pivot(
        index=identity, columns="period", values=["sharpe_annualized", "trade_count", "max_drawdown_usd"]
    ).reset_index()
    wide.columns = ["_".join(filter(None, item)).strip("_") if isinstance(item, tuple) else item for item in wide.columns]
    wide["insufficient_sample_is"] = wide["trade_count_is"].lt(min_trades)
    wide["insufficient_sample_validation"] = wide["trade_count_validation"].lt(min_trades)
    wide["insufficient_sample"] = wide["insufficient_sample_is"] | wide["insufficient_sample_validation"]
    wide["common_positive_eligible"] = (
        ~wide["insufficient_sample"]
        & wide["sharpe_annualized_is"].gt(0)
        & wide["sharpe_annualized_validation"].gt(0)
    )
    return wide


def _component(group: pd.DataFrame) -> list[tuple[int, int]]:
    cells = {
        (int(row.parameter_1_index), int(row.parameter_2_index))
        for row in group.itertuples(index=False) if row.common_positive_eligible
    }
    seen: set[tuple[int, int]] = set()
    best: list[tuple[int, int]] = []
    for start in cells:
        if start in seen:
            continue
        stack, current = [start], []
        seen.add(start)
        while stack:
            cell = stack.pop()
            current.append(cell)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (cell[0] + di, cell[1] + dj)
                if neighbor in cells and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if len(current) > len(best):
            best = current
    return best


def _region(group: pd.DataFrame, component: list[tuple[int, int]]) -> str:
    if not component:
        return "aucune"
    selected = group[group.apply(lambda row: (int(row.parameter_1_index), int(row.parameter_2_index)) in component, axis=1)]
    p1 = sorted(selected.parameter_1_value.unique().tolist())
    p2 = sorted(selected.parameter_2_value.unique().tolist())
    return f"{selected.parameter_1_name.iloc[0]}={p1}; {selected.parameter_2_name.iloc[0]}={p2}"


def _is_interior(component: list[tuple[int, int]], size: int = 7) -> bool:
    return bool(component) and all(0 < i < size - 1 and 0 < j < size - 1 for i, j in component)


def _pivot(group: pd.DataFrame, field: str, definition: dict[str, Any]) -> pd.DataFrame:
    return group.pivot(index="parameter_1_value", columns="parameter_2_value", values=field).reindex(
        index=definition["values_1"], columns=definition["values_2"]
    )


def _draw_pair(group: pd.DataFrame, definition: dict[str, Any], fields: tuple[str, str], titles: tuple[str, str], filename: str, fmt: str, center_zero: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.8), sharey=True)
    for ax, field, title in zip(axes, fields, titles):
        pivot = _pivot(group, field, definition)
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap="RdYlGn", center=0 if center_zero else None, linewidths=0.35, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(definition["parameter_2"])
        ax.set_ylabel(definition["parameter_1"])
    fig.suptitle(group.mechanism_id.iloc[0], fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HEATMAP_DIR / filename, dpi=155)
    plt.close(fig)


def _plot(group: pd.DataFrame, definition: dict[str, Any]) -> None:
    mechanism = group.mechanism_id.iloc[0]
    _draw_pair(group, definition, ("sharpe_annualized_is", "sharpe_annualized_validation"),
               ("IS — Sharpe annualisé", "Validation — Sharpe annualisé"), f"{mechanism}_sharpe_is_validation.png", ".2f", True)
    _draw_pair(group, definition, ("trade_count_is", "trade_count_validation"),
               ("IS — nombre de trades", "Validation — nombre de trades"), f"{mechanism}_trade_count_is_validation.png", ".0f", False)
    _draw_pair(group, definition, ("max_drawdown_usd_is", "max_drawdown_usd_validation"),
               ("IS — max drawdown USD", "Validation — max drawdown USD"), f"{mechanism}_max_drawdown_is_validation.png", ".0f", True)


def _write_report(config: dict[str, Any], wide: pd.DataFrame, grid_hash: str) -> None:
    min_trades = int(config["minimum_trades_each_period"])
    conclusions = {
        "gap_continuation": (
            "Une zone commune contiguë existe, mais elle touche la borne basse du seuil de gap. "
            "La hausse du seuil de volume raréfie rapidement les trades; la zone n'est donc pas encadrée par cette grille."
        ),
        "trend_pullback_ma_structure": (
            "La composante principale est entièrement intérieure à la grille: c'est la seule zone commune "
            "géométriquement encadrée de cette extension. Les Sharpe IS y restent cependant modestes."
        ),
        "rolling_range_breakout": (
            "La zone commune atteint la borne haute de longueur d'entrée et plusieurs longueurs de sortie "
            "produisent des résultats identiques. La frontière haute n'est donc pas observée et l'axe de sortie est peu discriminant."
        ),
    }
    lines = [
        "# MGC — extension ciblée des paysages IS / Validation",
        "",
        "## Protocole",
        "",
        "- IS : `2019-05-06 → 2023-12-31`; Validation : `2024-01-01 → 2026-03-19`.",
        "- Trois grilles 7×7 exactement conformes à la demande et verrouillées avant exécution.",
        "- Coûts, session MGC, fills, sizing, signaux et money management inchangés.",
        "- Une cellule faible en échantillon possède moins de 20 trades dans au moins une sous-période. Elle reste visible mais est exclue de la zone commune robuste.",
        "- Zone commune : Sharpe strictement positif dans les deux périodes et échantillon suffisant.",
        "- Aucun paramètre n'est sélectionné automatiquement.",
        f"- SHA-256 des grilles : `{grid_hash}`.",
        "",
        "## Résultats topologiques",
        "",
        "| Mécanisme | Cellules suffisantes | Positives communes éligibles | Plus grande zone | Région | Entièrement intérieure |",
        "|---|---:|---:|---:|---|---|",
    ]
    for mechanism in config["mechanisms"]:
        group = wide[wide.mechanism_id == mechanism]
        component = _component(group)
        lines.append(
            f"| `{mechanism}` | {int((~group.insufficient_sample).sum())}/49 | {int(group.common_positive_eligible.sum())} | "
            f"{len(component)} | `{_region(group, component)}` | {'oui' if _is_interior(component) else 'non'} |"
        )
    lines.extend(["", "## Lecture par mécanisme", ""])
    for mechanism, definition in config["mechanisms"].items():
        group = wide[wide.mechanism_id == mechanism]
        component = _component(group)
        boundary = bool(component) and not _is_interior(component)
        lines.extend(
            [
                f"### {mechanism}",
                "",
                f"- Grille : `{definition['parameter_1']}={definition['values_1']}` × `{definition['parameter_2']}={definition['values_2']}`.",
                f"- Cellules éligibles : `{int((~group.insufficient_sample).sum())}/49`; cellules positives communes : `{int(group.common_positive_eligible.sum())}`.",
                f"- Plus grande composante contiguë : `{len(component)}` cellules; région : `{_region(group, component)}`.",
                f"- Contact avec une borne de la grille : `{'oui' if boundary else 'non'}`.",
                f"- Conclusion : {conclusions[mechanism]}",
                "",
            ]
        )
    lines.extend(
        [
            "## Graphiques",
            "",
            "Trois images sont produites par mécanisme : Sharpe IS/Validation côte à côte, nombre de trades IS/Validation et max drawdown IS/Validation.",
            "",
            "## Limites",
            "",
            "Le masque commun mesure la persistance du signe du Sharpe et la suffisance de l'échantillon. Le max drawdown est affiché pour lecture du risque mais ne participe pas au masque. Une zone touchant une borne indique que sa frontière n'est pas entièrement observée; elle n'autorise aucun élargissement ou tuning supplémentaire dans cette étude.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(data_path: Path, workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    config = _load_grid()
    grid_hash = _sha256(GRID_PATH)
    GRID_HASH_PATH.write_text(f"{grid_hash}  {GRID_PATH.name}\n", encoding="ascii")
    start = config["periods"]["is"][0]
    end = config["periods"]["validation"][1]
    tasks = [
        (mechanism, definition, i, j, value_1, value_2)
        for mechanism, definition in config["mechanisms"].items()
        for i, value_1 in enumerate(definition["values_1"])
        for j, value_2 in enumerate(definition["values_2"])
    ]
    rows: list[dict[str, Any]] = []
    print(f"Grid SHA-256: {grid_hash}; cells: {len(tasks)}", flush=True)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(data_path), start, end, config["periods"]),
    ) as pool:
        futures = {pool.submit(_worker, task): task for task in tasks}
        for number, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            rows.extend(future.result())
            print(f"[{number:03d}/{len(tasks)}] {task[0]} {task[4]}/{task[5]}", flush=True)
    results = pd.DataFrame(rows).sort_values(["mechanism_id", "parameter_1_index", "parameter_2_index", "period"])
    wide = _wide(results, int(config["minimum_trades_each_period"]))
    flags = wide[[
        "combination_id", "insufficient_sample_is", "insufficient_sample_validation",
        "insufficient_sample", "common_positive_eligible",
    ]]
    results = results.merge(flags, on="combination_id", how="left", validate="many_to_one")
    results["insufficient_sample_this_period"] = results["trade_count"].lt(int(config["minimum_trades_each_period"]))
    results.to_csv(RESULT_PATH, index=False)
    for mechanism, definition in config["mechanisms"].items():
        _plot(wide[wide.mechanism_id == mechanism], definition)
    _write_report(config, wide, grid_hash)
    metadata = {
        "run_type": "mgc_focused_landscape_extension",
        "grid_sha256": grid_hash,
        "periods": config["periods"],
        "mechanism_count": 3,
        "combination_count": 147,
        "result_row_count": len(results),
        "minimum_trades_each_period": int(config["minimum_trades_each_period"]),
        "metrics": ["sharpe_annualized", "trade_count", "max_drawdown_usd"],
        "automatic_parameter_selection": False,
        "money_management_changed": False,
        "data_path": str(data_path.relative_to(REPO_ROOT)),
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Rows: {len(results)}; heatmaps: {len(list(HEATMAP_DIR.glob('*.png')))}; selected parameters: 0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    execute(args.data.resolve(), args.workers)


if __name__ == "__main__":
    main()
