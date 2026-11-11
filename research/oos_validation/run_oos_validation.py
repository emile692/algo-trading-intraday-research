"""Execute the six locked configurations once on the reserved OOS window."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.baseline_backtests.baseline_engine import (
    build_buy_and_hold,
    compute_metrics,
    load_frozen_specs,
    prepare_data,
    run_baseline_backtest,
    serialize_checks,
    slice_prepared_data,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
TRADE_DIR = OUTPUT_DIR / "trades"
PARAMETER_FILE = OUTPUT_DIR / "frozen_oos_parameters.json"
PARAMETER_SHA_FILE = OUTPUT_DIR / "frozen_oos_parameters.sha256"
RESULT_PATH = OUTPUT_DIR / "oos_results.csv"
REPORT_PATH = OUTPUT_DIR / "OOS_VALIDATION_REPORT.md"
RECEIPT_PATH = OUTPUT_DIR / "oos_execution_receipt.json"
FROZEN_DIR = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
IS_RESULTS = REPO_ROOT / "research" / "wide_parameter_landscape" / "wide_parameter_landscape_results.csv"
DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_locked_parameters() -> tuple[dict[str, Any], str]:
    expected = PARAMETER_SHA_FILE.read_text(encoding="ascii").split()[0].lower()
    actual = _sha256(PARAMETER_FILE)
    if actual != expected:
        raise RuntimeError(f"Frozen OOS parameter SHA-256 mismatch: {actual} != {expected}")
    config = json.loads(PARAMETER_FILE.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_oos_execution" or len(config.get("configurations", {})) != 6:
        raise RuntimeError("Frozen OOS parameter file is not valid")
    return config, actual


def _apply_parameters(spec: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    selected = copy.deepcopy(spec)
    structural = selected["canonical_parameters"]["structural"]
    for name, value in parameters.items():
        if name not in structural:
            raise KeyError(f"Unknown structural parameter {selected['mechanism_id']}::{name}")
        structural[name] = value
    return selected


def _is_result(frame: pd.DataFrame, mechanism_id: str, parameters: dict[str, Any]) -> pd.Series:
    candidates = frame[frame["mechanism_id"] == mechanism_id].copy()
    for name, value in parameters.items():
        matches = (
            ((candidates["parameter_1_name"] == name) & np.isclose(pd.to_numeric(candidates["parameter_1_value"]), float(value)))
            | ((candidates["parameter_2_name"] == name) & np.isclose(pd.to_numeric(candidates["parameter_2_value"]), float(value)))
        )
        candidates = candidates[matches]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one locked IS comparison row for {mechanism_id}; found {len(candidates)}")
    return candidates.iloc[0]


def _metric(value: Any) -> float:
    return float(value) if value is not None and pd.notna(value) else np.nan


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.{digits}f}"
    return str(value)


def _write_report(results: pd.DataFrame, config: dict[str, Any], parameter_sha: str, benchmark_metrics: dict[str, Any]) -> None:
    lines = [
        "# Validation OOS figée — six mécanismes MNQ", "",
        "## Protocole", "",
        "Les six configurations ont été écrites et leur SHA-256 calculé avant toute exécution OOS. Chaque configuration "
        "a ensuite été exécutée exactement une fois sur **2024-01-01 → 2026-03-19**, sans heatmap, voisinage, variante "
        "ou ajustement. L'historique antérieur initialise seulement les indicateurs; aucun trade ni rendement antérieur "
        "au 1er janvier 2024 n'entre dans les métriques OOS.", "",
        f"- SHA-256 des paramètres : `{parameter_sha}`",
        "- Coûts, fills, contrat unique, règles de session et de risque : hérités sans changement des frozen baselines",
        "- Benchmark OOS : MNQ Buy & Hold close-to-close, sans daily flatten",
        f"- Benchmark OOS : PnL {_fmt(benchmark_metrics['net_pnl_usd'])} USD ; Sharpe {_fmt(benchmark_metrics['sharpe_annualized'])} ; max DD {_fmt(benchmark_metrics['max_drawdown_usd'])} USD",
        "", "## Comparaison IS / OOS", "",
        "Une dégradation positive signifie que la métrique IS est supérieure à la métrique OOS. Pour le drawdown, "
        "une valeur positive signifie que la perte maximale absolue est plus élevée en OOS.", "",
        "| Mécanisme | Label | Trades OOS | PnL OOS | Exp. IS | Exp. OOS | PF IS | PF OOS | Sharpe IS | Sharpe OOS | Dégrad. Sharpe | DD IS | DD OOS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| `{row.mechanism_id}` | {row.sample_label} | {row.oos_trade_count} | {_fmt(row.oos_net_pnl_usd)} | "
            f"{_fmt(row.is_expectancy_per_trade_usd)} | {_fmt(row.oos_expectancy_per_trade_usd)} | "
            f"{_fmt(row.is_profit_factor)} | {_fmt(row.oos_profit_factor)} | {_fmt(row.is_sharpe_annualized)} | "
            f"{_fmt(row.oos_sharpe_annualized)} | {_fmt(row.sharpe_degradation_is_minus_oos)} | "
            f"{_fmt(row.is_max_drawdown_usd)} | {_fmt(row.oos_max_drawdown_usd)} |"
        )
    lines.extend(["", "## Résultats individuels", ""])
    for row in results.itertuples(index=False):
        parameters = config["configurations"][row.mechanism_id]
        lines.extend([
            f"### {row.mechanism_id}", "",
            f"- Paramètres figés : `{json.dumps(parameters, ensure_ascii=False, sort_keys=True)}`",
            f"- Label : **{row.sample_label}**",
            f"- OOS : {row.oos_trade_count} trades ; PnL {_fmt(row.oos_net_pnl_usd)} USD ; expectancy {_fmt(row.oos_expectancy_per_trade_usd)} USD ; PF {_fmt(row.oos_profit_factor)}",
            f"- OOS : Sharpe {_fmt(row.oos_sharpe_annualized)} ; max DD {_fmt(row.oos_max_drawdown_usd)} USD",
            f"- Dégradation IS→OOS : Sharpe {_fmt(row.sharpe_degradation_is_minus_oos)} ; expectancy {_fmt(row.expectancy_degradation_is_minus_oos_usd)} USD ; PF {_fmt(row.profit_factor_degradation_is_minus_oos)}",
            f"- Fréquence : {_fmt(row.is_trades_per_year)} trades/an IS → {_fmt(row.oos_trades_per_year)} trades/an OOS",
            f"- Checks : **{'valide' if row.backtest_valid else 'invalide'}**. {row.implementation_warning or 'Aucun avertissement.'}",
            f"- Trades : `trades/{row.mechanism_id}.csv`", "",
        ])
    lines.extend([
        "## Interprétation", "",
        "Ce rapport décrit la validation temporelle des configurations déjà figées. Il ne réouvre aucune décision de "
        "paramètre et n'élimine automatiquement aucun mécanisme. `gap_continuation` reste marqué `low_sample`, quelle "
        "que soit sa performance observée.", "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def execute(data_path: Path) -> None:
    if RECEIPT_PATH.exists() or RESULT_PATH.exists():
        raise RuntimeError("OOS validation output already exists; refusing a second execution")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRADE_DIR.mkdir(parents=True, exist_ok=True)
    config, parameter_sha = _load_locked_parameters()
    frozen_paths = sorted(FROZEN_DIR.glob("*.json")) + [FROZEN_MANIFEST]
    frozen_before = {str(path): _sha256(path) for path in frozen_paths}
    specs, benchmark_spec, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    spec_by_id = {spec["mechanism_id"]: spec for spec in specs}
    if set(config["configurations"]) != set(spec_by_id).intersection(config["configurations"]):
        raise RuntimeError("Unknown mechanism in locked OOS configurations")
    is_results = pd.read_csv(IS_RESULTS)
    oos_start = date.fromisoformat(config["oos_start"])
    oos_end = date.fromisoformat(config["oos_end"])
    warmup_start = date.fromisoformat(config["indicator_warmup"]["start"])
    prepared = prepare_data(data_path, start_date=warmup_start, end_date=oos_end)
    oos_data = slice_prepared_data(prepared, oos_start, oos_end)
    benchmark_run = build_buy_and_hold(benchmark_spec, oos_data)
    benchmark_metrics = compute_metrics(benchmark_spec, benchmark_run, benchmark_run.daily)
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for number, (mechanism_id, parameters) in enumerate(config["configurations"].items(), 1):
        print(f"[{number}/6] {mechanism_id}", flush=True)
        spec = _apply_parameters(spec_by_id[mechanism_id], parameters)
        run = run_baseline_backtest(spec, prepared, evaluation_start=oos_start, evaluation_end=oos_end)
        metrics = compute_metrics(spec, run, benchmark_run.daily)
        if not metrics["backtest_valid"]:
            raise RuntimeError(f"OOS implementation checks failed for {mechanism_id}")
        is_row = _is_result(is_results, mechanism_id, parameters)
        labels = config.get("labels", {}).get(mechanism_id, [])
        is_years = float(is_row["session_count"] / 252.0) if "session_count" in is_row else 4.65
        oos_years = float(metrics["session_count"] / 252.0)
        trade_frame = run.trades.copy()
        trade_frame.insert(2, "oos_parameter_file_sha256", parameter_sha)
        trade_frame.to_csv(TRADE_DIR / f"{mechanism_id}.csv", index=False)
        checks.append(serialize_checks(mechanism_id, run.checks))
        is_pf = _metric(is_row["profit_factor"])
        oos_pf = _metric(metrics["profit_factor"])
        rows.append({
            "mechanism_id": mechanism_id, "strategy_id": spec["strategy_id"],
            "parameters": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
            "sample_label": ",".join(labels) if labels else "standard",
            "oos_parameter_file_sha256": parameter_sha, "frozen_spec_sha256": spec["_frozen_spec_sha256"],
            "is_start": is_row["is_start"], "is_end": is_row["is_end"],
            "oos_start": config["oos_start"], "oos_end": config["oos_end"],
            "is_trade_count": int(is_row["trade_count"]), "oos_trade_count": int(metrics["trade_count"]),
            "is_net_pnl_usd": _metric(is_row["net_pnl_usd"]), "oos_net_pnl_usd": _metric(metrics["net_pnl_usd"]),
            "is_expectancy_per_trade_usd": _metric(is_row["expectancy_per_trade_usd"]),
            "oos_expectancy_per_trade_usd": _metric(metrics["expectancy_per_trade_usd"]),
            "is_profit_factor": is_pf, "oos_profit_factor": oos_pf,
            "is_sharpe_annualized": _metric(is_row["sharpe_annualized"]),
            "oos_sharpe_annualized": _metric(metrics["sharpe_annualized"]),
            "is_max_drawdown_usd": _metric(is_row["max_drawdown_usd"]),
            "oos_max_drawdown_usd": _metric(metrics["max_drawdown_usd"]),
            "sharpe_degradation_is_minus_oos": _metric(is_row["sharpe_annualized"]) - _metric(metrics["sharpe_annualized"]),
            "expectancy_degradation_is_minus_oos_usd": _metric(is_row["expectancy_per_trade_usd"]) - _metric(metrics["expectancy_per_trade_usd"]),
            "profit_factor_degradation_is_minus_oos": is_pf - oos_pf,
            "max_drawdown_abs_degradation_oos_minus_is_usd": abs(_metric(metrics["max_drawdown_usd"])) - abs(_metric(is_row["max_drawdown_usd"])),
            "is_trades_per_year": int(is_row["trade_count"]) / is_years,
            "oos_trades_per_year": int(metrics["trade_count"]) / oos_years,
            "implementation_warning": metrics["implementation_warning"], "backtest_valid": metrics["backtest_valid"],
        })
    results = pd.DataFrame(rows)
    results.to_csv(RESULT_PATH, index=False)
    pd.DataFrame(checks).to_csv(OUTPUT_DIR / "implementation_checks.csv", index=False)
    _write_report(results, config, parameter_sha, benchmark_metrics)
    frozen_after = {str(path): _sha256(path) for path in frozen_paths}
    if frozen_before != frozen_after:
        raise RuntimeError("A frozen baseline changed during OOS execution")
    if _sha256(PARAMETER_FILE) != parameter_sha:
        raise RuntimeError("The locked OOS parameter file changed during execution")
    receipt = {
        "status": "completed_once", "parameter_sha256": parameter_sha,
        "data_sha256": _sha256(data_path), "oos_start": config["oos_start"], "oos_end": config["oos_end"],
        "strategy_count": 6, "valid_strategy_count": int(results["backtest_valid"].astype(bool).sum()),
        "oos_execution_count_per_strategy": 1, "heatmaps_generated": 0, "tuning_performed": False,
        "neighborhood_analysis_performed": False, "frozen_files_unchanged": True,
    }
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("OOS strategies executed: 6", flush=True)
    print("Valid OOS backtests: 6", flush=True)
    print("OOS executions per strategy: 1", flush=True)
    print("Tuning/heatmaps/neighborhoods: 0", flush=True)
    print("Frozen files modified: 0", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    execute(args.data.resolve())


if __name__ == "__main__":
    main()
