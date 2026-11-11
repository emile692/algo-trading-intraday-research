"""Run locked exit-family grids over three frozen MNQ alpha mechanisms."""

from __future__ import annotations

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
    slice_prepared_data,
)
from research.money_management_lab.mm_engine import (  # noqa: E402
    build_overlay_context,
    run_exit_overlay_backtest,
    slice_result,
)


GRID_PATH = OUTPUT_DIR / "locked_mm_grid.json"
FROZEN_DIR = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
DATA_PATH = REPO_ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"
RESULT_PATH = OUTPUT_DIR / "money_management_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "money_management_family_summary.csv"
ZONE_PATH = OUTPUT_DIR / "money_management_robust_zones.csv"
REPORT_PATH = OUTPUT_DIR / "MONEY_MANAGEMENT_LAB_REPORT.md"
METRICS = [
    "sharpe_annualized", "expectancy_per_trade_usd", "profit_factor", "max_drawdown_usd", "trade_count",
    "average_win_usd", "average_loss_usd", "average_mae_usd", "average_mfe_usd", "average_holding_minutes",
    "net_pnl_usd",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes() -> dict[str, str]:
    paths = sorted(FROZEN_DIR.glob("*.json")) + [FROZEN_MANIFEST]
    return {str(path.relative_to(REPO_ROOT)): sha256(path) for path in paths}


def apply_alpha_parameters(spec: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(spec)
    structural = updated["canonical_parameters"]["structural"]
    for name, value in values.items():
        if name not in structural:
            raise KeyError(f"Unknown structural parameter {updated['mechanism_id']}::{name}")
        structural[name] = value
    if int(updated["canonical_parameters"]["risk_management"]["fixed_quantity"]) != 1:
        raise RuntimeError("Money Management Lab requires exactly one MNQ")
    return updated


def combinations(family: str, definition: dict[str, list[Any]]) -> list[dict[str, Any]]:
    names = list(definition)
    return [{"family": family, **dict(zip(names, values))} for values in itertools.product(*(definition[name] for name in names))]


def config_id(family: str, number: int) -> str:
    return "baseline" if family == "baseline" else f"{family}__{number:02d}"


def period_metrics(spec: dict[str, Any], run: Any, benchmark: pd.DataFrame, start: date, end: date) -> dict[str, Any]:
    return compute_metrics(spec, slice_result(run, start, end), benchmark)


def metric_row(
    mechanism_id: str, family: str, identifier: str, policy: dict[str, Any],
    is_metrics: dict[str, Any], val_metrics: dict[str, Any], grid_hash: str,
) -> dict[str, Any]:
    parameters = {key: value for key, value in policy.items() if key != "family"}
    names = list(parameters)
    row: dict[str, Any] = {
        "mechanism_id": mechanism_id,
        "exit_family": family,
        "configuration_id": f"{mechanism_id}__{identifier}",
        "parameters_json": json.dumps(parameters, sort_keys=True),
        "parameter_1_name": names[0] if names else "baseline",
        "parameter_1_value": parameters[names[0]] if names else "baseline",
        "parameter_1_index": 0,
        "parameter_2_name": names[1] if len(names) > 1 else "variant",
        "parameter_2_value": parameters[names[1]] if len(names) > 1 else "single",
        "parameter_2_index": 0,
        "grid_sha256": grid_hash,
        "frozen_spec_sha256": is_metrics["frozen_spec_sha256"],
    }
    for metric in METRICS:
        row[f"is_{metric}"] = is_metrics[metric]
        row[f"validation_{metric}"] = val_metrics[metric]
    row["is_backtest_valid"] = is_metrics["backtest_valid"]
    row["validation_backtest_valid"] = val_metrics["backtest_valid"]
    return row


def add_grid_indices(frame: pd.DataFrame, definition: dict[str, list[Any]]) -> pd.DataFrame:
    output = frame.copy()
    names = list(definition)
    for index, row in output.iterrows():
        parameters = json.loads(row["parameters_json"])
        output.loc[index, "parameter_1_index"] = definition[names[0]].index(parameters[names[0]])
        if len(names) > 1:
            output.loc[index, "parameter_2_index"] = definition[names[1]].index(parameters[names[1]])
    return output


def enrich_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    for mechanism_id, group in output.groupby("mechanism_id"):
        baseline = group[group["exit_family"] == "baseline"].iloc[0]
        mask = output["mechanism_id"] == mechanism_id
        for period in ("is", "validation"):
            for metric in ("expectancy_per_trade_usd", "max_drawdown_usd", "sharpe_annualized", "profit_factor", "net_pnl_usd"):
                output.loc[mask, f"{period}_{metric}_minus_baseline"] = output.loc[mask, f"{period}_{metric}"] - baseline[f"{period}_{metric}"]
            output.loc[mask, f"{period}_expectancy_ratio_to_baseline"] = output.loc[mask, f"{period}_expectancy_per_trade_usd"] / baseline[f"{period}_expectancy_per_trade_usd"]
            output.loc[mask, f"{period}_drawdown_improved"] = output.loc[mask, f"{period}_max_drawdown_usd"].abs() <= abs(float(baseline[f"{period}_max_drawdown_usd"]))
    output["common_positive"] = (
        output["is_backtest_valid"].astype(bool) & output["validation_backtest_valid"].astype(bool)
        & output["is_sharpe_annualized"].gt(0) & output["validation_sharpe_annualized"].gt(0)
        & output["is_profit_factor"].gt(1) & output["validation_profit_factor"].gt(1)
        & output["is_trade_count"].ge(20) & output["validation_trade_count"].ge(20)
    )
    output["expectancy_preserved_both"] = output["is_expectancy_ratio_to_baseline"].ge(0.8) & output["validation_expectancy_ratio_to_baseline"].ge(0.8)
    output["drawdown_improved_both"] = output["is_drawdown_improved"].astype(bool) & output["validation_drawdown_improved"].astype(bool)
    output["robust_dd_expectancy_cell"] = output["common_positive"] & output["expectancy_preserved_both"] & output["drawdown_improved_both"]
    return output


def components(frame: pd.DataFrame, flag: str) -> list[pd.DataFrame]:
    selected = frame[frame[flag].astype(bool)]
    lookup = {(int(row.parameter_1_index), int(row.parameter_2_index)): idx for idx, row in selected.iterrows()}
    unseen = set(lookup)
    found: list[pd.DataFrame] = []
    while unseen:
        start = min(unseen); unseen.remove(start); queue = deque([start]); indices = []
        while queue:
            x, y = queue.popleft(); indices.append(lookup[(x, y)])
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor); queue.append(neighbor)
        found.append(frame.loc[indices].copy())
    return sorted(found, key=lambda item: (-len(item), item["configuration_id"].min()))


def zone_rows(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    alternatives = results[results["exit_family"] != "baseline"]
    for (mechanism_id, family), group in alternatives.groupby(["mechanism_id", "exit_family"], sort=True):
        for flag, zone_type in (("common_positive", "common_positive"), ("robust_dd_expectancy_cell", "dd_expectancy_strict")):
            for zone_number, component in enumerate(components(group, flag), 1):
                rows.append({
                    "mechanism_id": mechanism_id,
                    "exit_family": family,
                    "zone_type": zone_type,
                    "zone_number": zone_number,
                    "zone_size": len(component),
                    "configuration_ids": "|".join(component["configuration_id"]),
                    "parameters": "|".join(component["parameters_json"]),
                    "median_min_sharpe": float(np.median(np.minimum(component["is_sharpe_annualized"], component["validation_sharpe_annualized"]))),
                    "median_is_drawdown_change_usd": float(component["is_max_drawdown_usd_minus_baseline"].median()),
                    "median_validation_drawdown_change_usd": float(component["validation_max_drawdown_usd_minus_baseline"].median()),
                })
    return pd.DataFrame(rows, columns=["mechanism_id", "exit_family", "zone_type", "zone_number", "zone_size", "configuration_ids", "parameters", "median_min_sharpe", "median_is_drawdown_change_usd", "median_validation_drawdown_change_usd"])


def family_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    alternatives = results[results["exit_family"] != "baseline"]
    for (mechanism_id, family), group in alternatives.groupby(["mechanism_id", "exit_family"], sort=True):
        comps = components(group, "robust_dd_expectancy_cell")
        positive_comps = components(group, "common_positive")
        rows.append({
            "mechanism_id": mechanism_id,
            "exit_family": family,
            "configuration_count": len(group),
            "common_positive_count": int(group["common_positive"].sum()),
            "dd_improved_both_count": int(group["drawdown_improved_both"].sum()),
            "expectancy_preserved_both_count": int(group["expectancy_preserved_both"].sum()),
            "robust_dd_expectancy_count": int(group["robust_dd_expectancy_cell"].sum()),
            "largest_robust_zone_size": len(comps[0]) if comps else 0,
            "largest_common_positive_zone_size": len(positive_comps[0]) if positive_comps else 0,
            "median_is_expectancy_delta_usd": float(group["is_expectancy_per_trade_usd_minus_baseline"].median()),
            "median_validation_expectancy_delta_usd": float(group["validation_expectancy_per_trade_usd_minus_baseline"].median()),
            "median_is_drawdown_delta_usd": float(group["is_max_drawdown_usd_minus_baseline"].median()),
            "median_validation_drawdown_delta_usd": float(group["validation_max_drawdown_usd_minus_baseline"].median()),
        })
    return pd.DataFrame(rows)


def matrix_for(group: pd.DataFrame, value: str, definition: dict[str, list[Any]]) -> pd.DataFrame:
    names = list(definition)
    temp = group.copy()
    temp["p1"] = temp["parameters_json"].map(lambda value_: json.loads(value_)[names[0]])
    if len(names) == 1:
        temp["p2"] = "value"
    else:
        temp["p2"] = temp["parameters_json"].map(lambda value_: json.loads(value_)[names[1]])
        temp["p2"] = temp["p2"].map(lambda value_: "None" if value_ is None else value_)
    return temp.pivot(index="p1", columns="p2", values=value).reindex(index=definition[names[0]])


def heatmap_panel(mechanism_id: str, family: str, group: pd.DataFrame, definition: dict[str, list[Any]]) -> None:
    metrics = [
        ("sharpe_annualized", "Sharpe", ".2f", 0.0),
        ("expectancy_per_trade_usd", "Expectancy USD", ".1f", 0.0),
        ("profit_factor", "Profit factor", ".2f", 1.0),
        ("max_drawdown_usd", "Max DD USD", ".0f", 0.0),
        ("trade_count", "Trades", ".0f", None),
    ]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(15, 4.0 * len(metrics)))
    for row_index, (metric, title, fmt, center) in enumerate(metrics):
        left = matrix_for(group, f"is_{metric}", definition)
        right = matrix_for(group, f"validation_{metric}", definition)
        values = np.concatenate([left.to_numpy(dtype=float).ravel(), right.to_numpy(dtype=float).ravel()])
        finite = values[np.isfinite(values)]
        vmin, vmax = (float(finite.min()), float(finite.max())) if len(finite) else (None, None)
        for col, (matrix, label) in enumerate(((left, "IS 2019–2023"), (right, "Validation 2024–2026"))):
            sns.heatmap(matrix, annot=True, fmt=fmt, cmap="RdYlGn", center=center, vmin=vmin, vmax=vmax, linewidths=0.4, ax=axes[row_index, col])
            axes[row_index, col].set_title(f"{title} — {label}")
            axes[row_index, col].set_xlabel(list(definition)[1] if len(definition) > 1 else "")
            axes[row_index, col].set_ylabel(list(definition)[0] if col == 0 else "")
    fig.suptitle(f"{mechanism_id} — {family}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(HEATMAP_DIR / f"{mechanism_id}__{family}__is_validation.png", dpi=150)
    plt.close(fig)


def write_report(results: pd.DataFrame, summary: pd.DataFrame, zones: pd.DataFrame, grid_hash: str) -> None:
    vol_240 = results[(results["mechanism_id"] == "volatility_breakout") & (results["exit_family"] == "time_stop") & (results["parameters_json"].str.contains("240"))].iloc[0]
    atr_stop_1 = results[(results["mechanism_id"] == "atr_trend_flip") & (results["exit_family"] == "atr_stop") & (results["parameters_json"] == '{"stop_atr": 1.0}')].iloc[0]
    lines = [
        "# Money Management Lab", "", "## Protocole", "",
        "- Alphas verrouillés : `volatility_breakout(14, 0.6)`, `compression_breakout(10, 1)` et `atr_trend_flip(10, 1.75)`.",
        "- Quantité fixe : **1 MNQ** ; aucun filtre, trigger ou paramètre alpha modifié.",
        "- IS : **2019-05-06 → 2023-12-31** ; Validation : **2024-01-01 → 2026-03-19**.",
        "- ATR de money management : Wilder ATR(14) sur le timeframe signal verrouillé, dernière barre entièrement terminée au fill.",
        "- Chaque famille remplace les sorties baseline ; le hard flatten et les règles d'early close restent inchangés.",
        "- Une cellule robuste exige Sharpe > 0, PF > 1 et au moins 20 trades dans chaque période.",
        "- Une zone DD/expectancy exige aussi un DD non supérieur à la baseline et au moins 80 % de son expectancy dans chaque période.",
        f"- SHA-256 de la grille : `{grid_hash}`.", "",
        "## Baselines", "",
        "| Mécanisme | Sharpe IS | Sharpe VAL | Exp. IS | Exp. VAL | PF IS | PF VAL | DD IS | DD VAL | Trades IS/VAL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results[results["exit_family"] == "baseline"].itertuples(index=False):
        lines.append(f"| `{row.mechanism_id}` | {row.is_sharpe_annualized:.2f} | {row.validation_sharpe_annualized:.2f} | {row.is_expectancy_per_trade_usd:.2f} | {row.validation_expectancy_per_trade_usd:.2f} | {row.is_profit_factor:.2f} | {row.validation_profit_factor:.2f} | {row.is_max_drawdown_usd:,.0f} | {row.validation_max_drawdown_usd:,.0f} | {int(row.is_trade_count)}/{int(row.validation_trade_count)} |")
    lines += ["", "## Baseline contre familles", "",
        "| Mécanisme | Famille | Configs | Positives communes | Plateau positif | DD amélioré | Expectancy préservée | Cellules strictes DD/Exp | Plateau strict | Δ DD médian IS/VAL | Δ Exp médiane IS/VAL |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| `{row.mechanism_id}` | `{row.exit_family}` | {row.configuration_count} | {row.common_positive_count} | {row.largest_common_positive_zone_size} | {row.dd_improved_both_count} | {row.expectancy_preserved_both_count} | {row.robust_dd_expectancy_count} | {row.largest_robust_zone_size} | {row.median_is_drawdown_delta_usd:,.0f}/{row.median_validation_drawdown_delta_usd:,.0f} | {row.median_is_expectancy_delta_usd:.2f}/{row.median_validation_expectancy_delta_usd:.2f} |")
    lines += ["", "## Plateaux positifs communs", ""]
    positive_zones = zones[(zones["zone_type"] == "common_positive") & (zones["zone_size"] >= 2)]
    if positive_zones.empty:
        lines.append("Aucun plateau positif contigu d'au moins deux cellules.")
    else:
        for row in positive_zones.itertuples(index=False):
            lines.append(f"- `{row.mechanism_id}` / `{row.exit_family}` — {row.zone_size} cellules, paramètres `{row.parameters}`.")
    lines += ["", "## Filtre strict drawdown + expectancy", ""]
    strict_zones = zones[zones["zone_type"] == "dd_expectancy_strict"]
    strict_plateaus = strict_zones[strict_zones["zone_size"] >= 2]
    if strict_plateaus.empty:
        lines.append("**Aucun plateau strict de plusieurs cellules.** Les cellules suivantes sont des points isolés, pas des zones robustes :")
    for row in strict_zones.itertuples(index=False):
        lines.append(f"- `{row.mechanism_id}` / `{row.exit_family}` — {row.zone_size} cellule(s), paramètres `{row.parameters}`.")
    lines.append("- `rr_target(stop_atr=1.0, target_R=None)` est mécaniquement identique à `atr_stop(stop_atr=1.0)` ; ce n'est pas une confirmation indépendante.")
    lines += [
        "", "## Lecture par mécanisme", "",
        f"- **volatility_breakout** : les sorties ATR, RR et breakeven détruisent largement l'expectancy. Les time stops 120 et 240 minutes sont les seuls points stricts. À 240 minutes, Sharpe IS/VAL = {vol_240['is_sharpe_annualized']:.2f}/{vol_240['validation_sharpe_annualized']:.2f}, PF = {vol_240['is_profit_factor']:.2f}/{vol_240['validation_profit_factor']:.2f}, expectancy = {vol_240['is_expectancy_per_trade_usd']:.2f}/{vol_240['validation_expectancy_per_trade_usd']:.2f} USD et DD = {vol_240['is_max_drawdown_usd']:,.0f}/{vol_240['validation_max_drawdown_usd']:,.0f} USD. Le point 180 minutes casse la continuité stricte.",
        "- **compression_breakout** : toutes les familles restent positives dans les deux périodes, formant de larges plateaux de positivité. Aucune cellule ne réduit toutefois le DD contre la baseline dans les deux périodes ; aucune zone stricte n'est retenue.",
        f"- **atr_trend_flip** : les quatre stops ATR forment un plateau positif, mais seul 1,0 ATR conserve au moins 80 % de l'expectancy et améliore le DD dans les deux périodes. Ses Sharpe IS/VAL sont {atr_stop_1['is_sharpe_annualized']:.2f}/{atr_stop_1['validation_sharpe_annualized']:.2f} et ses DD {atr_stop_1['is_max_drawdown_usd']:,.0f}/{atr_stop_1['validation_max_drawdown_usd']:,.0f} USD. Il reste un point isolé sous le filtre strict.",
        "- Conclusion de structure : aucun mécanisme ne présente encore un plateau multi-cellules combinant simultanément positivité, baisse du DD et préservation de l'expectancy.",
    ]
    lines += [
        "", "## Risques méthodologiques", "",
        "- Les familles modifient la durée des positions ; certains signaux ultérieurs peuvent devenir exécutables ou rester bloqués par la règle une-position.",
        "- Le breakeven n'est jamais déplacé rétroactivement dans la minute qui atteint son seuil.",
        "- Le trailing ignore la barre signal partiellement antérieure au fill.",
        "- Les zones sont décrites sans choisir automatiquement un point optimal.", "",
        "## Livrables", "", "- `money_management_results.csv`", "- `money_management_family_summary.csv`",
        "- `money_management_robust_zones.csv`", "- dossier `heatmaps/`", "- `run_money_management_lab.py` et `mm_engine.py`", "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def execute() -> None:
    for directory in (OUTPUT_DIR, HEATMAP_DIR, TABLE_DIR, MPL_CACHE):
        directory.mkdir(parents=True, exist_ok=True)
    config = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    if not config.get("locked_before_execution"):
        raise RuntimeError("Money management grid is not locked")
    grid_hash = sha256(GRID_PATH)
    (OUTPUT_DIR / "locked_mm_grid.sha256").write_text(f"{grid_hash}  {GRID_PATH.name}\n", encoding="ascii")
    frozen_before = frozen_hashes()
    specs, benchmark_spec, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    spec_by_id = {spec["mechanism_id"]: spec for spec in specs}
    is_start, is_end = date.fromisoformat(config["is_start"]), date.fromisoformat(config["is_end"])
    val_start, val_end = date.fromisoformat(config["validation_start"]), date.fromisoformat(config["validation_end"])
    prepared = prepare_data(DATA_PATH, start_date=is_start, end_date=val_end)
    is_benchmark = build_buy_and_hold(benchmark_spec, slice_prepared_data(prepared, is_start, is_end)).daily
    val_benchmark = build_buy_and_hold(benchmark_spec, slice_prepared_data(prepared, val_start, val_end)).daily
    all_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    total = len(config["mechanisms"]) * (1 + sum(len(combinations(name, definition)) for name, definition in config["exit_families"].items()))
    completed = 0
    for mechanism_id, alpha_values in config["mechanisms"].items():
        spec = apply_alpha_parameters(spec_by_id[mechanism_id], alpha_values)
        timeframe = int(config["signal_timeframes_minutes"][mechanism_id])
        baseline_run = run_baseline_backtest(spec, prepared)
        completed += 1
        print(f"[{completed:03d}/{total}] {mechanism_id} baseline", flush=True)
        is_metrics = period_metrics(spec, baseline_run, is_benchmark, is_start, is_end)
        val_metrics = period_metrics(spec, baseline_run, val_benchmark, val_start, val_end)
        all_rows.append(metric_row(mechanism_id, "baseline", "baseline", {"family": "baseline"}, is_metrics, val_metrics, grid_hash))
        check_rows.append({"mechanism_id": mechanism_id, "configuration_id": f"{mechanism_id}__baseline", **baseline_run.checks})
        context = build_overlay_context(spec, prepared, timeframe)
        for family, definition in config["exit_families"].items():
            policies = combinations(family, definition)
            family_rows = []
            for number, policy in enumerate(policies, 1):
                completed += 1
                identifier = config_id(family, number)
                print(f"[{completed:03d}/{total}] {mechanism_id} {identifier} {policy}", flush=True)
                run = run_exit_overlay_backtest(spec, prepared, policy, timeframe, context=context)
                if run.checks["missing_atr_entry_skips"]:
                    raise RuntimeError(f"Missing causal ATR at {mechanism_id} {identifier}")
                is_metrics = period_metrics(spec, run, is_benchmark, is_start, is_end)
                val_metrics = period_metrics(spec, run, val_benchmark, val_start, val_end)
                row = metric_row(mechanism_id, family, identifier, policy, is_metrics, val_metrics, grid_hash)
                family_rows.append(row); all_rows.append(row)
                check_rows.append({"mechanism_id": mechanism_id, "configuration_id": row["configuration_id"], **run.checks})
            indexed = add_grid_indices(pd.DataFrame(family_rows), definition)
            for row in indexed.to_dict("records"):
                for existing in reversed(all_rows):
                    if existing["configuration_id"] == row["configuration_id"]:
                        existing.update(row); break
            pd.DataFrame(indexed).to_csv(TABLE_DIR / f"{mechanism_id}__{family}.csv", index=False)
    results = enrich_comparisons(pd.DataFrame(all_rows))
    summary = family_summary(results)
    zones = zone_rows(results)
    results.to_csv(RESULT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    zones.to_csv(ZONE_PATH, index=False)
    pd.DataFrame(check_rows).to_csv(OUTPUT_DIR / "implementation_checks.csv", index=False)
    for mechanism_id, mechanism_group in results[results["exit_family"] != "baseline"].groupby("mechanism_id"):
        for family, family_group in mechanism_group.groupby("exit_family"):
            heatmap_panel(mechanism_id, family, family_group, config["exit_families"][family])
    write_report(results, summary, zones, grid_hash)
    if frozen_before != frozen_hashes():
        raise RuntimeError("A frozen baseline changed")
    if grid_hash != sha256(GRID_PATH):
        raise RuntimeError("Locked grid changed during execution")
    metadata = {
        "grid_sha256": grid_hash,
        "data_sha256": sha256(DATA_PATH),
        "configuration_count": len(results),
        "valid_is_count": int(results["is_backtest_valid"].sum()),
        "valid_validation_count": int(results["validation_backtest_valid"].sum()),
        "quantity_mnq": 1,
        "alpha_parameters_modified_during_run": False,
        "orb_used": False,
        "portfolio_used": False,
        "frozen_files_unchanged": True,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    execute()
