"""Terminal one-dimensional money-management studies and deterministic exit locks."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
CURVE_DIR = OUTPUT_DIR / "final_mm_curves"
MPL_CACHE = OUTPUT_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from research.baseline_backtests.baseline_engine import (  # noqa: E402
    build_buy_and_hold,
    compute_metrics,
    load_frozen_specs,
    prepare_data,
    run_baseline_backtest,
    slice_prepared_data,
)
from research.money_management_lab.mm_engine import build_overlay_context, run_exit_overlay_backtest, slice_result  # noqa: E402
from research.money_management_lab.run_money_management_lab import apply_alpha_parameters  # noqa: E402


GRID_PATH = OUTPUT_DIR / "final_mm_study_grid.json"
GRID_HASH_PATH = OUTPUT_DIR / "final_mm_study_grid.sha256"
RESULT_PATH = OUTPUT_DIR / "final_mm_results.csv"
LOCK_PATH = OUTPUT_DIR / "final_mm_locked_exit_rules.json"
LOCK_HASH_PATH = OUTPUT_DIR / "final_mm_locked_exit_rules.sha256"
LOCK_DECISIONS_PATH = OUTPUT_DIR / "final_mm_lock_decisions.json"
REPORT_PATH = OUTPUT_DIR / "FINAL_MM_REPORT.md"
FROZEN_DIR = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
DATA_PATH = REPO_ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"
METRICS = ["sharpe_annualized", "expectancy_per_trade_usd", "profit_factor", "max_drawdown_usd", "trade_count"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes() -> dict[str, str]:
    paths = sorted(FROZEN_DIR.glob("*.json")) + [FROZEN_MANIFEST]
    return {str(path.relative_to(REPO_ROOT)): sha256(path) for path in paths}


def metrics_for_period(spec: dict[str, Any], run: Any, benchmark: pd.DataFrame, start: date, end: date) -> dict[str, Any]:
    return compute_metrics(spec, slice_result(run, start, end), benchmark)


def result_row(
    mechanism_id: str, exit_family: str, parameter_name: str, parameter_value: Any,
    is_metrics: dict[str, Any], val_metrics: dict[str, Any], grid_hash: str,
) -> dict[str, Any]:
    row = {
        "mechanism_id": mechanism_id,
        "exit_family": exit_family,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "is_baseline": exit_family == "baseline",
        "grid_sha256": grid_hash,
        "frozen_spec_sha256": is_metrics["frozen_spec_sha256"],
        "is_backtest_valid": is_metrics["backtest_valid"],
        "validation_backtest_valid": val_metrics["backtest_valid"],
    }
    for metric in METRICS:
        row[f"is_{metric}"] = is_metrics[metric]
        row[f"validation_{metric}"] = val_metrics[metric]
    return row


def add_eligibility(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    output["expectancy_ratio_is"] = np.nan
    output["expectancy_ratio_validation"] = np.nan
    output["dd_improved_is"] = False
    output["dd_improved_validation"] = False
    for mechanism_id, group in output.groupby("mechanism_id"):
        baseline = group[group["is_baseline"]].iloc[0]
        mask = output["mechanism_id"] == mechanism_id
        output.loc[mask, "expectancy_ratio_is"] = output.loc[mask, "is_expectancy_per_trade_usd"] / baseline["is_expectancy_per_trade_usd"]
        output.loc[mask, "expectancy_ratio_validation"] = output.loc[mask, "validation_expectancy_per_trade_usd"] / baseline["validation_expectancy_per_trade_usd"]
        output.loc[mask, "dd_improved_is"] = output.loc[mask, "is_max_drawdown_usd"].abs() <= abs(float(baseline["is_max_drawdown_usd"]))
        output.loc[mask, "dd_improved_validation"] = output.loc[mask, "validation_max_drawdown_usd"].abs() <= abs(float(baseline["validation_max_drawdown_usd"]))
    output["eligible_for_stable_zone"] = (
        ~output["is_baseline"].astype(bool)
        & output["is_backtest_valid"].astype(bool) & output["validation_backtest_valid"].astype(bool)
        & output["is_sharpe_annualized"].gt(0) & output["validation_sharpe_annualized"].gt(0)
        & output["is_profit_factor"].gt(1) & output["validation_profit_factor"].gt(1)
        & output["is_trade_count"].ge(20) & output["validation_trade_count"].ge(20)
        & output["expectancy_ratio_is"].ge(0.8) & output["expectancy_ratio_validation"].ge(0.8)
        & output["dd_improved_is"].astype(bool) & output["dd_improved_validation"].astype(bool)
    )
    return output


def contiguous_zones(group: pd.DataFrame, ordered_values: list[float]) -> list[list[float]]:
    eligible = set(group.loc[group["eligible_for_stable_zone"], "parameter_value"].astype(float))
    zones: list[list[float]] = []
    current: list[float] = []
    for value in map(float, ordered_values):
        if value in eligible:
            current.append(value)
        elif current:
            zones.append(current); current = []
    if current:
        zones.append(current)
    return zones


def medoid(zone: list[float], ordered_values: list[float]) -> float:
    indices = {float(value): index for index, value in enumerate(ordered_values)}
    return min(zone, key=lambda value: (sum(abs(indices[value] - indices[other]) for other in zone), value))


def baseline_exit_payload(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "frozen_baseline",
        "exit_description": spec["exit"],
        "risk_management": spec["canonical_parameters"]["risk_management"],
    }


def build_locks(
    config: dict[str, Any], decisions: dict[str, Any], results: pd.DataFrame,
    spec_by_id: dict[str, dict[str, Any]], grid_hash: str,
) -> dict[str, Any]:
    locks: dict[str, Any] = {}
    for mechanism_id, definition in config["mechanisms"].items():
        group = results[results["mechanism_id"] == mechanism_id]
        baseline = group[group["is_baseline"]].iloc[0]
        decision = decisions["strategies"][mechanism_id]
        zones = [] if definition["study"] == "none" else contiguous_zones(
            group[~group["is_baseline"]], definition["values"]
        )
        if decision["exit_family"] == "baseline":
            status = "MM_LOCKED_BASELINE"
            rule = baseline_exit_payload(spec_by_id[mechanism_id])
            selected_metrics = baseline
        else:
            if definition["study"] != decision["exit_family"]:
                raise ValueError(f"Exit family mismatch for {mechanism_id}")
            parameter_name = definition["parameter_name"]
            selected = float(decision[parameter_name])
            if selected not in [float(value) for value in definition["values"]]:
                raise ValueError(f"Locked value outside tested grid for {mechanism_id}: {selected}")
            selected_rows = group[
                (~group["is_baseline"])
                & group["exit_family"].eq(decision["exit_family"])
                & group["parameter_value"].astype(float).eq(selected)
            ]
            if len(selected_rows) != 1:
                raise ValueError(f"Locked result is missing or duplicated for {mechanism_id}")
            selected_metrics = selected_rows.iloc[0]
            if not bool(selected_metrics["is_backtest_valid"] and selected_metrics["validation_backtest_valid"]):
                raise ValueError(f"Cannot lock invalid backtest for {mechanism_id}")
            status = "MM_LOCKED_MODIFIED"
            rule = {
                "source": "explicit_user_lock",
                "exit_family": decision["exit_family"],
                parameter_name: decision[parameter_name],
                "execution_semantics": decision["execution_semantics"],
                "hard_flatten": spec_by_id[mechanism_id]["exit"]["session_flatten"],
            }
        locks[mechanism_id] = {
            "status": status,
            "alpha_parameters": definition["alpha_parameters"],
            "quantity_mnq": 1,
            "exit_rule": rule,
            "justification": decision["justification"],
            "eligible_zones": zones,
            "selected_result": {
                "is_sharpe": selected_metrics["is_sharpe_annualized"],
                "validation_sharpe": selected_metrics["validation_sharpe_annualized"],
                "is_expectancy": selected_metrics["is_expectancy_per_trade_usd"],
                "validation_expectancy": selected_metrics["validation_expectancy_per_trade_usd"],
                "is_profit_factor": selected_metrics["is_profit_factor"],
                "validation_profit_factor": selected_metrics["validation_profit_factor"],
                "is_max_drawdown": selected_metrics["is_max_drawdown_usd"],
                "validation_max_drawdown": selected_metrics["validation_max_drawdown_usd"],
                "is_trade_count": int(selected_metrics["is_trade_count"]),
                "validation_trade_count": int(selected_metrics["validation_trade_count"]),
            },
            "baseline_reference": {
                "is_sharpe": baseline["is_sharpe_annualized"],
                "validation_sharpe": baseline["validation_sharpe_annualized"],
                "is_expectancy": baseline["is_expectancy_per_trade_usd"],
                "validation_expectancy": baseline["validation_expectancy_per_trade_usd"],
                "is_max_drawdown": baseline["is_max_drawdown_usd"],
                "validation_max_drawdown": baseline["validation_max_drawdown_usd"],
            },
        }
        continue
        if definition["study"] == "none":
            status = "MM_LOCKED_BASELINE"
            rule = baseline_exit_payload(spec_by_id[mechanism_id])
            reason = (
                "Sortie baseline verrouillée définitivement sur instruction utilisateur ; "
                "aucune autre variante de money management exécutée."
            )
            zones: list[list[float]] = []
        else:
            alternatives = group[~group["is_baseline"]]
            zones = contiguous_zones(alternatives, definition["values"])
            stable = [zone for zone in zones if len(zone) >= 3]
            if stable:
                selected_zone = sorted(stable, key=lambda zone: (-len(zone), ordered_start(zone, definition["values"])))[0]
                selected = medoid(selected_zone, definition["values"])
                status = "MM_LOCKED_MODIFIED"
                rule = {"exit_family": definition["study"], definition["parameter_name"]: selected, "stable_zone": selected_zone}
                reason = "Modified exit selected mechanically as the parameter medoid of the largest precommitted stable zone."
            else:
                status = "MM_LOCKED_BASELINE"
                rule = baseline_exit_payload(spec_by_id[mechanism_id])
                reason = (
                    "Aucune zone contiguë éligible d'au moins trois valeurs ; "
                    "sortie baseline conservée."
                )
        locks[mechanism_id] = {
            "status": status,
            "alpha_parameters": definition["alpha_parameters"],
            "quantity_mnq": 1,
            "exit_rule": rule,
            "justification": reason,
            "eligible_zones": zones,
            "baseline_reference": {
                "is_sharpe": baseline["is_sharpe_annualized"],
                "validation_sharpe": baseline["validation_sharpe_annualized"],
                "is_expectancy": baseline["is_expectancy_per_trade_usd"],
                "validation_expectancy": baseline["validation_expectancy_per_trade_usd"],
                "is_max_drawdown": baseline["is_max_drawdown_usd"],
                "validation_max_drawdown": baseline["validation_max_drawdown_usd"],
            },
        }
    return {
        "schema_version": "1.0",
        "status": "FINAL_MM_LOCKS",
        "terminal_study": True,
        "source_grid_sha256": grid_hash,
        "lock_decisions_sha256": sha256(LOCK_DECISIONS_PATH),
        "decision_source": decisions["decision_source"],
        "prior_mechanical_rule": config["precommitted_lock_rule"],
        "strategies": locks,
    }


def ordered_start(zone: list[float], values: list[float]) -> int:
    lookup = {float(value): index for index, value in enumerate(values)}
    return min(lookup[value] for value in zone)


def plot_study(mechanism_id: str, group: pd.DataFrame, baseline: pd.Series, parameter_name: str) -> None:
    group = group.sort_values("parameter_value")
    fig, axes = plt.subplots(5, 1, figsize=(11, 18), sharex=True)
    plot_metrics = [
        ("sharpe_annualized", "Sharpe annualisé"),
        ("expectancy_per_trade_usd", "Expectancy / trade (USD)"),
        ("profit_factor", "Profit factor"),
        ("max_drawdown_usd", "Max drawdown (USD)"),
        ("trade_count", "Nombre de trades"),
    ]
    x = group["parameter_value"].astype(float)
    for ax, (metric, label) in zip(axes, plot_metrics):
        ax.plot(x, group[f"is_{metric}"], marker="o", linewidth=2, label="IS")
        ax.plot(x, group[f"validation_{metric}"], marker="o", linewidth=2, label="Validation")
        ax.axhline(float(baseline[f"is_{metric}"]), color="#1f77b4", linestyle="--", alpha=0.45, label="Baseline IS")
        ax.axhline(float(baseline[f"validation_{metric}"]), color="#ff7f0e", linestyle="--", alpha=0.45, label="Baseline Validation")
        eligible = group["eligible_for_stable_zone"].astype(bool)
        ax.scatter(x[eligible], group.loc[eligible, f"is_{metric}"], s=70, facecolors="none", edgecolors="black", linewidths=1.5)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel(parameter_name)
    fig.suptitle(f"{mechanism_id} — étude finale {parameter_name}")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(CURVE_DIR / f"{mechanism_id}__{parameter_name}__is_validation.png", dpi=170)
    plt.close(fig)


def write_report(config: dict[str, Any], results: pd.DataFrame, locks: dict[str, Any], grid_hash: str, lock_hash: str) -> None:
    lines = [
        "# Final Money Management Report", "", "## Protocole final", "",
        "- Signaux d'entrée, paramètres alpha, coûts, exécution et quantité **1 MNQ** inchangés.",
        "- IS : **2019-05-06 → 2023-12-31** ; Validation : **2024-01-01 → 2026-03-19**.",
        "- Seuls `time_stop` pour volatility breakout et `stop_atr` pour ATR trend flip ont été rejoués.",
        "- Les choix finaux ci-dessous proviennent de l'instruction explicite de verrouillage donnée après revue des études IS/Validation.",
        "- Aucun nouveau paramètre et aucun nouveau backtest ne sont introduits par ce verrouillage.",
        f"- SHA-256 grille : `{grid_hash}`.",
        f"- SHA-256 décisions de verrouillage : `{sha256(LOCK_DECISIONS_PATH)}`.",
        f"- SHA-256 règles finales : `{lock_hash}`.", "",
    ]
    for mechanism_id in ("compression_breakout", "volatility_breakout", "atr_trend_flip"):
        lock = locks["strategies"][mechanism_id]
        lines += [f"## {mechanism_id}", "", f"Statut final : **`{lock['status']}`**.", "", lock["justification"], ""]
        if lock["status"] == "MM_LOCKED_MODIFIED":
            rule = lock["exit_rule"]
            parameter_name = config["mechanisms"][mechanism_id]["parameter_name"]
            lines += [
                f"Règle verrouillée : **`{rule['exit_family']}`**, `{parameter_name} = {rule[parameter_name]}`.",
                "",
            ]
        if mechanism_id == "compression_breakout":
            lines.append("La sortie baseline est conservée définitivement, conformément à la décision préalable. Aucun nouveau money management n'a été exécuté.")
        else:
            definition = config["mechanisms"][mechanism_id]
            group = results[(results["mechanism_id"] == mechanism_id) & (~results["is_baseline"])]
            lines += [
                f"Valeurs testées : `{definition['values']}`.",
                f"Zones éligibles détectées : `{lock['eligible_zones']}`.", "",
                "| Valeur | Sharpe IS/VAL | Expectancy IS/VAL | PF IS/VAL | DD IS/VAL | Trades IS/VAL | Éligible |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for row in group.sort_values("parameter_value").itertuples(index=False):
                lines.append(f"| {row.parameter_value:g} | {row.is_sharpe_annualized:.2f}/{row.validation_sharpe_annualized:.2f} | {row.is_expectancy_per_trade_usd:.2f}/{row.validation_expectancy_per_trade_usd:.2f} | {row.is_profit_factor:.2f}/{row.validation_profit_factor:.2f} | {row.is_max_drawdown_usd:,.0f}/{row.validation_max_drawdown_usd:,.0f} | {int(row.is_trade_count)}/{int(row.validation_trade_count)} | {'oui' if row.eligible_for_stable_zone else 'non'} |")
        lines.append("")
    lines += [
        "## Verrouillage terminal", "",
        "Cette étape clôt le tuning des sorties pour ces trois mécanismes. Toute étude ultérieure devra traiter les règles ci-dessus comme des spécifications figées.", "",
        "## Livrables", "", "- `final_mm_results.csv`", "- `final_mm_locked_exit_rules.json` et son SHA-256",
        "- dossier `final_mm_curves/`", "- `run_final_mm_study.py`", "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def execute() -> None:
    for directory in (OUTPUT_DIR, CURVE_DIR, MPL_CACHE):
        directory.mkdir(parents=True, exist_ok=True)
    config = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    decisions = json.loads(LOCK_DECISIONS_PATH.read_text(encoding="utf-8"))
    if not config.get("locked_before_execution") or not config.get("terminal_money_management_study"):
        raise RuntimeError("Final MM grid is not locked")
    if not decisions.get("terminal_lock"):
        raise RuntimeError("Final MM decisions are not locked")
    grid_hash = sha256(GRID_PATH)
    GRID_HASH_PATH.write_text(f"{grid_hash}  {GRID_PATH.name}\n", encoding="ascii")
    frozen_before = frozen_hashes()
    specs, benchmark_spec, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    spec_by_id = {spec["mechanism_id"]: spec for spec in specs}
    is_start, is_end = date.fromisoformat(config["is_start"]), date.fromisoformat(config["is_end"])
    val_start, val_end = date.fromisoformat(config["validation_start"]), date.fromisoformat(config["validation_end"])
    prepared = prepare_data(DATA_PATH, start_date=is_start, end_date=val_end)
    is_benchmark = build_buy_and_hold(benchmark_spec, slice_prepared_data(prepared, is_start, is_end)).daily
    val_benchmark = build_buy_and_hold(benchmark_spec, slice_prepared_data(prepared, val_start, val_end)).daily
    rows: list[dict[str, Any]] = []
    for mechanism_id, definition in config["mechanisms"].items():
        spec = apply_alpha_parameters(spec_by_id[mechanism_id], definition["alpha_parameters"])
        baseline_run = run_baseline_backtest(spec, prepared)
        is_metrics = metrics_for_period(spec, baseline_run, is_benchmark, is_start, is_end)
        val_metrics = metrics_for_period(spec, baseline_run, val_benchmark, val_start, val_end)
        rows.append(result_row(mechanism_id, "baseline", "baseline", np.nan, is_metrics, val_metrics, grid_hash))
        print(f"{mechanism_id}: baseline", flush=True)
        if definition["study"] == "none":
            continue
        timeframe = int(definition["signal_timeframe_minutes"])
        context = build_overlay_context(spec, prepared, timeframe)
        for value in definition["values"]:
            policy = {"family": definition["study"], definition["parameter_name"]: value}
            run = run_exit_overlay_backtest(spec, prepared, policy, timeframe, context=context)
            if run.checks["missing_atr_entry_skips"] or any(run.checks.get(key, 0) for key in ("lookahead_violations", "future_high_low_violations", "same_bar_policy_violations", "double_entry_violations", "overnight_position_violations", "entry_after_last_entry_violations", "post_hard_flatten_violations", "ambiguous_multi_order_cancellations")):
                raise RuntimeError(f"Execution check failed for {mechanism_id} {value}: {run.checks}")
            is_metrics = metrics_for_period(spec, run, is_benchmark, is_start, is_end)
            val_metrics = metrics_for_period(spec, run, val_benchmark, val_start, val_end)
            rows.append(result_row(mechanism_id, definition["study"], definition["parameter_name"], value, is_metrics, val_metrics, grid_hash))
            print(f"{mechanism_id}: {definition['parameter_name']}={value}", flush=True)
    results = add_eligibility(pd.DataFrame(rows))
    results.to_csv(RESULT_PATH, index=False)
    locks = build_locks(config, decisions, results, spec_by_id, grid_hash)
    LOCK_PATH.write_text(json.dumps(locks, indent=2, default=str) + "\n", encoding="utf-8")
    lock_hash = sha256(LOCK_PATH)
    LOCK_HASH_PATH.write_text(f"{lock_hash}  {LOCK_PATH.name}\n", encoding="ascii")
    for mechanism_id, definition in config["mechanisms"].items():
        if definition["study"] == "none":
            continue
        group = results[(results["mechanism_id"] == mechanism_id) & (~results["is_baseline"])]
        baseline = results[(results["mechanism_id"] == mechanism_id) & results["is_baseline"]].iloc[0]
        plot_study(mechanism_id, group, baseline, definition["parameter_name"])
    write_report(config, results, locks, grid_hash, lock_hash)
    if frozen_before != frozen_hashes():
        raise RuntimeError("A frozen baseline changed")
    if grid_hash != sha256(GRID_PATH) or lock_hash != sha256(LOCK_PATH):
        raise RuntimeError("A locked final artifact changed")
    metadata = {
        "grid_sha256": grid_hash,
        "locked_exit_rules_sha256": lock_hash,
        "data_sha256": sha256(DATA_PATH),
        "result_rows": len(results),
        "all_backtests_valid": bool(results["is_backtest_valid"].all() and results["validation_backtest_valid"].all()),
        "quantity_mnq": 1,
        "alpha_parameters_changed": False,
        "orb_used": False,
        "portfolio_used": False,
        "frozen_files_unchanged": True,
        "terminal_money_management_study": True,
    }
    (OUTPUT_DIR / "final_mm_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(locks, indent=2, default=str), flush=True)


if __name__ == "__main__":
    execute()
