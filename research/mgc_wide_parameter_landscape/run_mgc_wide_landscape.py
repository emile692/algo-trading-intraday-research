"""Execute prelocked wide IS/Validation landscapes for 16 MGC mechanisms."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
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

from research.baseline_backtests.baseline_engine import (  # noqa: E402
    INITIAL_CAPITAL_USD,
    load_frozen_specs,
    run_baseline_backtest,
)
from research.mgc_baseline_backtests.run_mgc_baselines import (  # noqa: E402
    DEFAULT_DATA,
    FROZEN_DIR,
    FROZEN_MANIFEST,
    adapt_spec_to_mgc,
    prepare_mgc_data,
)


GRID_PATH = OUTPUT_DIR / "mgc_wide_parameter_grids.json"
GRID_HASH_PATH = OUTPUT_DIR / "mgc_wide_parameter_grids.sha256"
MGC_CONFIG_PATH = REPO_ROOT / "research" / "mgc_baseline_backtests" / "mgc_baseline_config.json"
RESULT_PATH = OUTPUT_DIR / "mgc_wide_landscape_results.csv"
REPORT_PATH = OUTPUT_DIR / "MGC_WIDE_LANDSCAPE_REPORT.md"
PERIOD_ORDER = ("is", "validation")
PARTIAL_PATH = OUTPUT_DIR / "mgc_wide_landscape_results.partial.csv"

_WORKER_PREPARED: Any = None
_WORKER_SPECS: dict[str, dict[str, Any]] = {}
_WORKER_MGC_CONFIG: dict[str, Any] = {}
_WORKER_PERIODS: dict[str, list[str]] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_grid() -> dict[str, Any]:
    config = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    if not config.get("locked_before_execution"):
        raise ValueError("Grid must be locked before execution")
    expected = {
        "atr_trend_flip", "channel_extreme_reversion", "compression_breakout", "gap_continuation",
        "gap_fade", "ma_crossover_trend", "momentum_confirmed_breakout", "range_structure_reversion",
        "regression_equilibrium_reversion", "rolling_range_breakout", "session_range_breakout",
        "short_horizon_pullback_reversion", "statistical_zscore_reversion",
        "trend_pullback_ma_structure", "volatility_breakout", "vwap_equilibrium_reversion",
    }
    if set(config["mechanisms"]) != expected:
        raise ValueError("Grid must cover exactly the 16 representative mechanisms")
    for mechanism, definition in config["mechanisms"].items():
        if len(definition["values_1"]) != 7 or len(definition["values_2"]) != 7:
            raise ValueError(f"{mechanism}: expected a prelocked 7x7 grid")
        if len(set(map(str, definition["values_1"]))) != 7 or len(set(map(str, definition["values_2"]))) != 7:
            raise ValueError(f"{mechanism}: duplicate grid values")
    return config


def _source_hashes() -> dict[str, str]:
    paths = sorted(FROZEN_DIR.glob("*.json")) + [FROZEN_MANIFEST, MGC_CONFIG_PATH, GRID_PATH]
    return {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in paths}


def _combinations(definition: dict[str, Any]) -> list[tuple[int, int, Any, Any]]:
    combinations: list[tuple[int, int, Any, Any]] = []
    for index_1, value_1 in enumerate(definition["values_1"]):
        for index_2, value_2 in enumerate(definition["values_2"]):
            if definition.get("invalid_combination_rule") == "skip fast_ema_period >= slow_ema_period" and value_1 >= value_2:
                continue
            combinations.append((index_1, index_2, value_1, value_2))
    return combinations


def _apply_parameters(spec: dict[str, Any], definition: dict[str, Any], value_1: Any, value_2: Any) -> dict[str, Any]:
    updated = copy.deepcopy(spec)
    structural = updated["canonical_parameters"]["structural"]
    values = {definition["parameter_1"]: value_1, definition["parameter_2"]: value_2}
    for name, value in values.items():
        if name == "channel_period":
            structural["ema_period"] = int(value)
            structural["atr_period"] = int(value)
        elif name == "entry_rsi_extreme":
            structural["long_entry_rsi"] = float(value)
            structural["short_entry_rsi"] = 100.0 - float(value)
        elif name in {"ribbon_fast_anchor", "ribbon_slow_anchor"}:
            continue
        elif name not in structural:
            raise KeyError(f"Unknown parameter {updated['mechanism_id']}::{name}")
        else:
            structural[name] = value
    if "ribbon_fast_anchor" in values:
        fast = int(values["ribbon_fast_anchor"])
        slow = int(values["ribbon_slow_anchor"])
        periods = [
            fast,
            max(fast + 1, int(round(fast * 13.0 / 8.0))),
            max(fast + 2, int(round(fast * 21.0 / 8.0))),
            slow,
            max(slow + 1, int(round(slow * 55.0 / 34.0))),
        ]
        if periods != sorted(set(periods)):
            raise ValueError(f"Invalid EMA ribbon {periods}")
        structural["ema_periods"] = periods
    return updated


def _period_metrics(run: Any, start: date, end: date) -> dict[str, Any]:
    daily = run.daily[run.daily["session_date"].between(start, end)].copy()
    trades = run.trades[run.trades["session_date"].between(start, end)].copy() if not run.trades.empty else run.trades.copy()
    daily_pnl = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0)
    equity = INITIAL_CAPITAL_USD + daily_pnl.cumsum()
    prior_equity = equity.shift(1).fillna(INITIAL_CAPITAL_USD)
    returns = daily_pnl / prior_equity
    std = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
    sharpe = float(returns.mean() / std * math.sqrt(252)) if np.isfinite(std) and std > 0 else np.nan
    return {"sharpe_annualized": sharpe, "trade_count": int(len(trades)), "session_count": int(len(daily))}


def _violations(checks: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "lookahead_violations", "future_high_low_violations", "same_bar_policy_violations",
        "double_entry_violations", "overnight_position_violations",
        "entry_after_last_entry_violations", "post_hard_flatten_violations",
        "ambiguous_multi_order_cancellations",
    ]
    return {key: checks.get(key) for key in keys if checks.get(key, 0)}


def _run_cell(
    mechanism: str,
    source_spec: dict[str, Any],
    definition: dict[str, Any],
    combo: tuple[int, int, Any, Any],
    mgc_config: dict[str, Any],
    prepared: Any,
    periods: dict[str, list[str]],
) -> list[dict[str, Any]]:
    index_1, index_2, value_1, value_2 = combo
    adapted = adapt_spec_to_mgc(source_spec, mgc_config)
    spec = _apply_parameters(adapted, definition, value_1, value_2)
    run = run_baseline_backtest(spec, prepared)
    violations = _violations(run.checks)
    if violations:
        raise RuntimeError(f"{mechanism} {value_1}/{value_2}: {violations}")
    rows = []
    for period in PERIOD_ORDER:
        start, end = (date.fromisoformat(item) for item in periods[period])
        metrics = _period_metrics(run, start, end)
        rows.append(
            {
                "mechanism_id": mechanism,
                "strategy_id": spec["strategy_id"],
                "complexity_grade": spec["complexity_grade"],
                "combination_id": f"{mechanism}__{index_1:02d}_{index_2:02d}",
                "parameter_1_name": definition["parameter_1"],
                "parameter_1_value": value_1,
                "parameter_1_index": index_1,
                "parameter_2_name": definition["parameter_2"],
                "parameter_2_value": value_2,
                "parameter_2_index": index_2,
                "period": period,
                **metrics,
                "missing_execution_bar_skips_full_period": int(run.checks.get("missing_execution_bar_skips", 0)),
                "hard_flatten_fallbacks_full_period": int(run.checks.get("hard_flatten_fallbacks", 0)),
                "backtest_valid": True,
            }
        )
    return rows


def _init_worker(data_path: str, full_start: str, full_end: str, periods: dict[str, list[str]]) -> None:
    global _WORKER_PREPARED, _WORKER_SPECS, _WORKER_MGC_CONFIG, _WORKER_PERIODS
    specs, _, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    _WORKER_SPECS = {spec["mechanism_id"]: spec for spec in specs}
    _WORKER_MGC_CONFIG = json.loads(MGC_CONFIG_PATH.read_text(encoding="utf-8"))
    _WORKER_PERIODS = periods
    _WORKER_PREPARED, _ = prepare_mgc_data(
        Path(data_path), date.fromisoformat(full_start), date.fromisoformat(full_end)
    )


def _process_cell(task: tuple[str, dict[str, Any], tuple[int, int, Any, Any]]) -> list[dict[str, Any]]:
    mechanism, definition, combo = task
    return _run_cell(
        mechanism, _WORKER_SPECS[mechanism], definition, combo,
        _WORKER_MGC_CONFIG, _WORKER_PREPARED, _WORKER_PERIODS,
    )


def _wide_results(results: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "mechanism_id", "strategy_id", "complexity_grade", "combination_id",
        "parameter_1_name", "parameter_1_value", "parameter_1_index",
        "parameter_2_name", "parameter_2_value", "parameter_2_index",
    ]
    wide = results.pivot(index=identity, columns="period", values=["sharpe_annualized", "trade_count"]).reset_index()
    wide.columns = ["_".join(filter(None, item)).strip("_") if isinstance(item, tuple) else item for item in wide.columns]
    wide["positive_both"] = wide["sharpe_annualized_is"].gt(0) & wide["sharpe_annualized_validation"].gt(0)
    return wide


def _largest_component(group: pd.DataFrame) -> list[tuple[int, int]]:
    positive = {
        (int(row.parameter_1_index), int(row.parameter_2_index))
        for row in group.itertuples(index=False) if bool(row.positive_both)
    }
    seen: set[tuple[int, int]] = set()
    best: list[tuple[int, int]] = []
    for start in positive:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[tuple[int, int]] = []
        while stack:
            cell = stack.pop()
            component.append(cell)
            for delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (cell[0] + delta[0], cell[1] + delta[1])
                if neighbor in positive and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if len(component) > len(best):
            best = component
    return best


def _component_region(group: pd.DataFrame, component: list[tuple[int, int]]) -> str:
    if not component:
        return "aucune"
    cells = group[
        group.apply(
            lambda row: (int(row["parameter_1_index"]), int(row["parameter_2_index"])) in component,
            axis=1,
        )
    ]
    p1 = sorted(cells["parameter_1_value"].unique().tolist())
    p2 = sorted(cells["parameter_2_value"].unique().tolist())
    return f"{cells['parameter_1_name'].iloc[0]}={p1}; {cells['parameter_2_name'].iloc[0]}={p2}"


def _pivot(group: pd.DataFrame, field: str, definition: dict[str, Any]) -> pd.DataFrame:
    pivot = group.pivot(index="parameter_1_value", columns="parameter_2_value", values=field)
    return pivot.reindex(index=definition["values_1"], columns=definition["values_2"])


def _heatmap(ax: Any, pivot: pd.DataFrame, title: str, x_label: str, y_label: str, fmt: str, mask: bool = False) -> None:
    if mask:
        sns.heatmap(pivot.astype(float), annot=pivot.astype(int), fmt="d", cmap=["#f1f5f9", "#16a34a"], vmin=0, vmax=1,
                    cbar=False, linewidths=0.35, ax=ax)
    else:
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap="RdYlGn", center=0 if "Sharpe" in title else None,
                    linewidths=0.35, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def _plot_mechanism(group: pd.DataFrame, mechanism: str, definition: dict[str, Any]) -> None:
    p1, p2 = definition["parameter_1"], definition["parameter_2"]
    is_sharpe = _pivot(group, "sharpe_annualized_is", definition)
    val_sharpe = _pivot(group, "sharpe_annualized_validation", definition)
    positive = _pivot(group, "positive_both", definition).fillna(False)
    count_is = _pivot(group, "trade_count_is", definition)
    count_val = _pivot(group, "trade_count_validation", definition)

    for label, pivot, title in (
        ("sharpe_is", is_sharpe, "Sharpe annualisé — IS 2019-05-06 → 2023-12-31"),
        ("sharpe_validation", val_sharpe, "Sharpe annualisé — Validation 2024-01-01 → 2026-03-19"),
    ):
        fig, ax = plt.subplots(figsize=(9.2, 6.6))
        _heatmap(ax, pivot, f"{mechanism}\n{title}", p2, p1, ".2f")
        fig.tight_layout()
        fig.savefig(HEATMAP_DIR / f"{mechanism}_{label}.png", dpi=155)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.8), sharey=True)
    _heatmap(axes[0], is_sharpe, "IS — Sharpe annualisé", p2, p1, ".2f")
    _heatmap(axes[1], val_sharpe, "Validation — Sharpe annualisé", p2, p1, ".2f")
    fig.suptitle(f"{mechanism} — paysage large MGC", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HEATMAP_DIR / f"{mechanism}_sharpe_is_validation.png", dpi=155)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    _heatmap(ax, positive, f"{mechanism}\nSharpe positif en IS ET Validation", p2, p1, "d", mask=True)
    fig.tight_layout()
    fig.savefig(HEATMAP_DIR / f"{mechanism}_positive_both_mask.png", dpi=155)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.8), sharey=True)
    _heatmap(axes[0], count_is, "IS — nombre de trades", p2, p1, ".0f")
    _heatmap(axes[1], count_val, "Validation — nombre de trades", p2, p1, ".0f")
    fig.suptitle(f"{mechanism} — taille d'échantillon par cellule", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HEATMAP_DIR / f"{mechanism}_trade_count_is_validation.png", dpi=155)
    plt.close(fig)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}" if isinstance(value, (float, np.floating)) else str(value)


def _write_report(config: dict[str, Any], results: pd.DataFrame, wide: pd.DataFrame, metadata: dict[str, Any]) -> None:
    lines = [
        "# MGC — Wide Parameter Landscapes IS / Validation",
        "",
        "## Protocole verrouillé",
        "",
        "- IS : `2019-05-06 → 2023-12-31`.",
        "- Validation temporelle : `2024-01-01 → 2026-03-19`.",
        "- Les 16 grilles ont été définies et hashées avant exécution.",
        "- Chaque mécanisme conserve ses signaux, sorties, coûts, fills, session MGC, contrat unique et hard flatten du baseline brut.",
        "- Aucun money management ajouté, aucun objectif d'optimisation et aucune sélection automatique de paramètres.",
        "- Une cellule contient uniquement le Sharpe annualisé et le nombre de trades pour chaque sous-période.",
        f"- SHA-256 des grilles : `{metadata['grid_sha256']}`.",
        f"- Cellules structurelles exécutées : `{metadata['combination_count']}`; lignes période-cellule : `{len(results)}`.",
        "",
        "Le masque commun vaut 1 lorsque le Sharpe est strictement positif dans les deux périodes. Il ne contient aucun seuil caché de nombre de trades; le nombre de trades doit être lu sur la heatmap associée.",
        "",
        "## Vue descriptive des recouvrements",
        "",
        "Cette section décrit la topologie observée sans retenir de paramètres et sans classer les mécanismes.",
        "",
        "| Mécanisme | Grille valide | Positif IS | Positif Validation | Positif dans les deux | Plus grande zone commune | Étendue de cette zone | Trades IS min–max | Trades Validation min–max |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for mechanism, definition in config["mechanisms"].items():
        group = wide[wide["mechanism_id"] == mechanism]
        component = _largest_component(group)
        lines.append(
            f"| `{mechanism}` | {len(group)} | {int(group['sharpe_annualized_is'].gt(0).sum())} | "
            f"{int(group['sharpe_annualized_validation'].gt(0).sum())} | {int(group['positive_both'].sum())} | "
            f"{len(component)} | `{_component_region(group, component)}` | {int(group['trade_count_is'].min())}–{int(group['trade_count_is'].max())} | "
            f"{int(group['trade_count_validation'].min())}–{int(group['trade_count_validation'].max())} |"
        )
    lines.extend(["", "## Grilles et fichiers par mécanisme", ""])
    for mechanism, definition in config["mechanisms"].items():
        group = wide[wide["mechanism_id"] == mechanism]
        component = _largest_component(group)
        lines.extend(
            [
                f"### {mechanism}",
                "",
                f"- Axe 1 `{definition['parameter_1']}` : `{definition['values_1']}`.",
                f"- Axe 2 `{definition['parameter_2']}` : `{definition['values_2']}`.",
                f"- Justification pré-déclarée : {definition['rationale']}",
                f"- Cellules au Sharpe positif dans les deux périodes : `{int(group['positive_both'].sum())}/{len(group)}`; plus grande composante contiguë : `{len(component)}` cellules.",
                f"- Heatmaps : `{mechanism}_sharpe_is.png`, `{mechanism}_sharpe_validation.png`, "
                f"`{mechanism}_sharpe_is_validation.png`, `{mechanism}_positive_both_mask.png`, "
                f"`{mechanism}_trade_count_is_validation.png`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Lecture des mécanismes de gap",
            "",
            f"`gap_continuation` passe de `{int(wide.loc[wide['mechanism_id'].eq('gap_continuation'), 'trade_count_is'].min())}` à "
            f"`{int(wide.loc[wide['mechanism_id'].eq('gap_continuation'), 'trade_count_is'].max())}` trades IS selon les seuils et possède "
            f"`{int(wide.loc[wide['mechanism_id'].eq('gap_continuation'), 'positive_both'].sum())}` cellules positives dans les deux périodes. "
            "L'absence de trades du baseline provenait donc bien en partie de sa sélectivité, après correction de l'ancrage horaire MGC.",
            "",
            f"`gap_fade` varie de `{int(wide.loc[wide['mechanism_id'].eq('gap_fade'), 'trade_count_is'].min())}` à "
            f"`{int(wide.loc[wide['mechanism_id'].eq('gap_fade'), 'trade_count_is'].max())}` trades IS mais ne possède aucune cellule au Sharpe positif dans les deux périodes. "
            "Certaines cellules gardent des minutes de fill absentes, visibles dans le CSV; le paysage n'est toutefois plus vide.",
            "",
            "## Limites",
            "",
            "- Les cellules d'une même grille partagent les mêmes données et ne sont pas des expériences indépendantes.",
            "- Un Sharpe positif avec très peu de trades ne constitue pas une zone robuste; la heatmap de fréquence est indissociable de celle du Sharpe.",
            "- La série continue MGC présente des minutes sans transaction. Aucun fill n'est synthétisé sur une minute absente.",
            "- La Validation sert ici à comparer les paysages temporels. Aucun paramètre n'est sélectionné après sa lecture.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(data_path: Path, workers: int, mechanisms: list[str] | None = None, merge_existing: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    config = _load_grid()
    grid_hash = _sha256(GRID_PATH)
    GRID_HASH_PATH.write_text(f"{grid_hash}  {GRID_PATH.name}\n", encoding="ascii")
    before = _source_hashes()
    full_start = date.fromisoformat(config["periods"]["is"][0])
    full_end = date.fromisoformat(config["periods"]["validation"][1])
    # The parent prepares data once for provenance and plotting metadata. Each
    # process worker loads its own immutable copy once through the initializer.
    prepared, _ = prepare_mgc_data(data_path, full_start, full_end)

    selected = set(mechanisms or config["mechanisms"])
    unknown = selected - set(config["mechanisms"])
    if unknown:
        raise ValueError(f"Unknown mechanisms: {sorted(unknown)}")
    all_tasks = []
    for mechanism, definition in config["mechanisms"].items():
        for combo in _combinations(definition):
            all_tasks.append((mechanism, definition, combo))
    tasks = [task for task in all_tasks if task[0] in selected]
    rows: list[dict[str, Any]] = []
    completed = 0
    print(f"Locked grid SHA-256: {grid_hash}", flush=True)
    print(f"Cells to execute: {len(tasks)} with {workers} worker(s)", flush=True)
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(data_path), full_start.isoformat(), full_end.isoformat(), config["periods"]),
    ) as pool:
        futures = {
            pool.submit(_process_cell, task): (task[0], task[2])
            for task in tasks
        }
        for future in as_completed(futures):
            mechanism, combo = futures[future]
            rows.extend(future.result())
            completed += 1
            print(f"[{completed:03d}/{len(tasks)}] {mechanism} {combo[2]}/{combo[3]}", flush=True)
            if completed % 25 == 0:
                pd.DataFrame(rows).to_csv(PARTIAL_PATH, index=False)

    results = pd.DataFrame(rows)
    if merge_existing:
        if not RESULT_PATH.exists():
            raise FileNotFoundError("Cannot merge subset: existing result CSV is absent")
        existing = pd.read_csv(RESULT_PATH)
        existing = existing[~existing["mechanism_id"].isin(selected)]
        results = pd.concat([existing, results], ignore_index=True)
    results = results.sort_values(
        ["mechanism_id", "parameter_1_index", "parameter_2_index", "period"]
    ).reset_index(drop=True)
    results.to_csv(RESULT_PATH, index=False)
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()
    wide = _wide_results(results)
    for mechanism, definition in config["mechanisms"].items():
        _plot_mechanism(wide[wide["mechanism_id"] == mechanism], mechanism, definition)

    after = _source_hashes()
    if before != after:
        raise RuntimeError("A locked grid, frozen source spec, or MGC baseline config changed during execution")
    metadata = {
        "run_type": "mgc_wide_parameter_landscape_is_validation",
        "grid_sha256": grid_hash,
        "data_path": str(data_path.relative_to(REPO_ROOT)),
        "data_sha256": prepared.source_sha256,
        "periods": config["periods"],
        "mechanism_count": 16,
        "combination_count": len(all_tasks),
        "result_row_count": len(results),
        "workers": workers,
        "metrics": ["sharpe_annualized", "trade_count"],
        "parameter_selection_performed": False,
        "money_management_changed": False,
        "source_files_unchanged": True,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(config, results, wide, metadata)
    print(f"Mechanisms: 16; cells: {len(all_tasks)}; rows: {len(results)}; heatmaps: {len(list(HEATMAP_DIR.glob('*.png')))}")
    print("Automatic parameter selection: 0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mechanisms", nargs="*", default=None)
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    execute(args.data.resolve(), args.workers, args.mechanisms, args.merge_existing)


if __name__ == "__main__":
    main()
