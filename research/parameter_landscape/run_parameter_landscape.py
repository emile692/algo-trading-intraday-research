"""Run the predeclared in-sample parameter landscapes for 16 MNQ mechanisms."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
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
GRID_PATH = OUTPUT_DIR / "parameter_grids.json"
DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"
RESULT_PATH = OUTPUT_DIR / "parameter_landscape_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "parameter_landscape_summary.csv"
REPORT_PATH = OUTPUT_DIR / "PARAMETER_LANDSCAPE_REPORT.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_hashes() -> dict[str, str]:
    paths = sorted(FROZEN_DIR.glob("*.json")) + [FROZEN_MANIFEST]
    return {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in paths}


def _load_locked_grid() -> dict[str, Any]:
    config = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    if not config.get("locked_before_execution"):
        raise ValueError("Parameter grid is not marked as locked before execution")
    if set(config["mechanisms"]) != {
        "atr_trend_flip", "channel_extreme_reversion", "compression_breakout", "gap_continuation",
        "gap_fade", "ma_crossover_trend", "momentum_confirmed_breakout", "range_structure_reversion",
        "regression_equilibrium_reversion", "rolling_range_breakout", "session_range_breakout",
        "short_horizon_pullback_reversion", "statistical_zscore_reversion",
        "trend_pullback_ma_structure", "volatility_breakout", "vwap_equilibrium_reversion",
    }:
        raise ValueError("Locked grid does not cover exactly the 16 representative mechanisms")
    for mechanism_id, definition in config["mechanisms"].items():
        parameters = definition["parameters"]
        if len(parameters) not in (1, 2) or any(len(values) < 3 for values in parameters.values()):
            raise ValueError(f"Invalid small-grid definition for {mechanism_id}")
    return config


def _apply_parameters(frozen_spec: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(frozen_spec)
    structural = spec["canonical_parameters"]["structural"]
    for name, value in values.items():
        if name == "channel_period":
            structural["ema_period"] = int(value)
            structural["atr_period"] = int(value)
        elif name == "entry_rsi_extreme":
            structural["long_entry_rsi"] = float(value)
            structural["short_entry_rsi"] = 100.0 - float(value)
        elif name in {"short_ribbon_scale", "long_ribbon_scale"}:
            continue
        else:
            structural[name] = value
    if "short_ribbon_scale" in values:
        base = [8, 13, 21, 34, 55]
        short_scale = float(values["short_ribbon_scale"])
        long_scale = float(values["long_ribbon_scale"])
        structural["ema_periods"] = [
            *[max(2, int(round(item * short_scale))) for item in base[:3]],
            *[max(2, int(round(item * long_scale))) for item in base[3:]],
        ]
        if len(set(structural["ema_periods"])) != 5 or structural["ema_periods"] != sorted(structural["ema_periods"]):
            raise ValueError(f"Invalid scaled EMA ribbon: {structural['ema_periods']}")
    return spec


def _combinations(definition: dict[str, Any]) -> list[dict[str, Any]]:
    names = list(definition["parameters"])
    axes = [definition["parameters"][name] for name in names]
    return [dict(zip(names, values)) for values in itertools.product(*axes)]


def _is_canonical(values: dict[str, Any], definition: dict[str, Any]) -> bool:
    return all(value == definition["parameters"][name][len(definition["parameters"][name]) // 2] for name, value in values.items())


def _compute_neighborhoods(frame: pd.DataFrame, definition: dict[str, Any], penalty: float) -> pd.DataFrame:
    result = frame.copy()
    parameter_names = list(definition["parameters"])
    index_maps = {
        name: {str(value): index for index, value in enumerate(definition["parameters"][name])}
        for name in parameter_names
    }
    coordinates = [
        tuple(index_maps[name][str(row[f"parameter_{number}_value"])] for number, name in enumerate(parameter_names, 1))
        for _, row in result.iterrows()
    ]
    fields: dict[str, list[Any]] = {
        "neighborhood_size": [], "neighborhood_sharpe_median": [], "neighborhood_sharpe_std": [],
        "sharpe_stability_score": [], "neighborhood_expectancy_median": [], "neighborhood_expectancy_std": [],
        "expectancy_stability_score": [], "neighborhood_positive_sharpe_fraction": [],
        "neighborhood_min_trade_count": [], "plateau_node": [],
    }
    for coordinate in coordinates:
        mask = np.array([max(abs(a - b) for a, b in zip(coordinate, other)) <= 1 for other in coordinates])
        neighbors = result.loc[mask & result["backtest_valid"].astype(bool)]
        sharpe = pd.to_numeric(neighbors["sharpe_annualized"], errors="coerce").dropna()
        expectancy = pd.to_numeric(neighbors["expectancy_per_trade_usd"], errors="coerce").dropna()
        sharpe_std = float(sharpe.std(ddof=0)) if len(sharpe) else np.nan
        expectancy_std = float(expectancy.std(ddof=0)) if len(expectancy) else np.nan
        sharpe_median = float(sharpe.median()) if len(sharpe) else np.nan
        expectancy_median = float(expectancy.median()) if len(expectancy) else np.nan
        score = sharpe_median - penalty * sharpe_std if len(sharpe) else np.nan
        expectancy_score = expectancy_median - penalty * expectancy_std if len(expectancy) else np.nan
        positive_fraction = float((sharpe > 0).mean()) if len(sharpe) else np.nan
        min_trades = int(neighbors["trade_count"].min()) if len(neighbors) else 0
        plateau_node = bool(
            len(neighbors) >= 4 and np.isfinite(score) and sharpe_median > 0 and score > 0
            and positive_fraction >= (2.0 / 3.0) and min_trades >= 30
        )
        values = [len(neighbors), sharpe_median, sharpe_std, score, expectancy_median, expectancy_std,
                  expectancy_score, positive_fraction, min_trades, plateau_node]
        for key, value in zip(fields, values):
            fields[key].append(value)
    for key, values in fields.items():
        result[key] = values
    return result


def _plot_heatmap(frame: pd.DataFrame, mechanism_id: str, metric: str, title: str, fmt: str) -> None:
    parameter_1 = frame["parameter_1_name"].iloc[0]
    parameter_2 = frame["parameter_2_name"].iloc[0]
    pivot = frame.pivot(index="parameter_1_value", columns="parameter_2_value", values=metric)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap="RdYlGn", center=0 if metric != "trade_count" else None,
                linewidths=0.5, cbar_kws={"label": title}, ax=ax)
    ax.set_title(f"{mechanism_id}\n{title} — IS 2019-05-06 à 2023-12-31")
    ax.set_xlabel(parameter_2)
    ax.set_ylabel(parameter_1)
    fig.tight_layout()
    fig.savefig(HEATMAP_DIR / f"{mechanism_id}_{metric}.png", dpi=160)
    plt.close(fig)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _summarize(mechanism_frame: pd.DataFrame) -> dict[str, Any]:
    ordered = mechanism_frame.sort_values("sharpe_stability_score", ascending=False, na_position="last")
    reference = ordered.iloc[0]
    nodes = mechanism_frame[mechanism_frame["plateau_node"].astype(bool)].copy()
    parameter_names = [mechanism_frame["parameter_1_name"].iloc[0], mechanism_frame["parameter_2_name"].iloc[0]]
    plateau_detected = False
    if len(nodes) >= 2:
        coordinates = nodes[["parameter_1_index", "parameter_2_index"]].to_numpy(dtype=int)
        plateau_detected = any(
            max(abs(int(a[0] - b[0])), abs(int(a[1] - b[1]))) <= 1
            for i, a in enumerate(coordinates) for b in coordinates[i + 1 :]
        )
    region = {
        name: sorted(nodes[f"parameter_{index}_value"].unique().tolist()) if len(nodes) else []
        for index, name in enumerate(parameter_names, 1)
    }
    return {
        "mechanism_id": mechanism_frame["mechanism_id"].iloc[0],
        "strategy_id": mechanism_frame["strategy_id"].iloc[0],
        "complexity_grade": mechanism_frame["complexity_grade"].iloc[0],
        "parameter_1_name": parameter_names[0],
        "parameter_2_name": parameter_names[1],
        "combination_count": int(len(mechanism_frame)),
        "valid_combination_count": int(mechanism_frame["backtest_valid"].astype(bool).sum()),
        "plateau_detected": plateau_detected,
        "plateau_node_count": int(len(nodes)),
        "plateau_parameter_region": json.dumps(region, ensure_ascii=False, sort_keys=True),
        "highest_stability_combination_id": reference["combination_id"],
        "highest_stability_parameters": json.dumps({
            parameter_names[0]: reference["parameter_1_value"],
            parameter_names[1]: reference["parameter_2_value"],
        }, ensure_ascii=False, sort_keys=True),
        "highest_sharpe_stability_score": reference["sharpe_stability_score"],
        "highest_neighborhood_sharpe_median": reference["neighborhood_sharpe_median"],
        "canonical_sharpe": float(mechanism_frame.loc[mechanism_frame["is_canonical"], "sharpe_annualized"].iloc[0]),
        "canonical_expectancy_usd": float(mechanism_frame.loc[mechanism_frame["is_canonical"], "expectancy_per_trade_usd"].iloc[0]),
        "canonical_trade_count": int(mechanism_frame.loc[mechanism_frame["is_canonical"], "trade_count"].iloc[0]),
    }


def _write_report(config: dict[str, Any], results: pd.DataFrame, summary: pd.DataFrame, metadata: dict[str, Any]) -> None:
    lines = [
        "# Parameter Landscape Study — 16 mécanismes MNQ",
        "",
        "## Protocole pré-déclaré",
        "",
        "Cette étude utilise exclusivement la période de développement **2019-05-06 → 2023-12-31**. "
        "La période **2024-01-01 → 2026-03-19** reste réservée et n'a été ni chargée dans les calculs de signaux, "
        "ni utilisée pour définir les grilles ou qualifier les plateaux.",
        "",
        "Les grilles ont été verrouillées dans `parameter_grids.json` avant l'exécution. Elles proviennent des paramètres "
        "structurels des spécifications figées et de voisinages locaux conventionnels. Les coûts, le contrat unique, "
        "le slippage, les horaires, les règles Topstep et les formules de risque restent fixes.",
        "",
        f"- SHA-256 de la grille : `{metadata['grid_sha256']}`",
        f"- Combinaisons exécutées : **{len(results)}**",
        f"- Combinaisons valides : **{int(results['backtest_valid'].astype(bool).sum())}**",
        "- Pénalité de stabilité : **0,5 × écart-type du voisinage**",
        "- Voisinage : distance de Chebyshev ≤ 1 dans la grille, point central inclus",
        "- Aucun résultat OOS, aucune variante parmi les 19 et aucune stratégie ORB utilisés",
        "",
        "## Définition d'un plateau",
        "",
        "Un point de plateau exige au moins quatre combinaisons valides dans son voisinage, un Sharpe médian positif, "
        "un score `médiane − 0,5 × écart-type` positif, au moins deux tiers de Sharpes voisins positifs et au moins "
        "30 trades pour chaque voisin. Un mécanisme reçoit `plateau_detected=true` quand au moins deux points adjacents "
        "satisfont cette règle. Cette qualification ne sélectionne aucun paramètre pour l'OOS.",
        "",
        "## Synthèse",
        "",
        "| Mécanisme | Paramètres | Combinaisons | Plateau | Nœuds | Région | Sharpe canonique | Trades canoniques |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in summary.sort_values("mechanism_id").itertuples(index=False):
        lines.append(
            f"| `{row.mechanism_id}` | `{row.parameter_1_name}` × `{row.parameter_2_name}` | {row.combination_count} | "
            f"{'oui' if row.plateau_detected else 'non'} | {row.plateau_node_count} | `{row.plateau_parameter_region}` | "
            f"{_fmt(row.canonical_sharpe)} | {row.canonical_trade_count} |"
        )
    lines.extend(["", "## Grilles et résultats par mécanisme", ""])
    for mechanism_id, definition in config["mechanisms"].items():
        frame = results[results["mechanism_id"] == mechanism_id]
        item = summary[summary["mechanism_id"] == mechanism_id].iloc[0]
        lines.extend([
            f"### {mechanism_id}", "",
            f"- Grille : `{json.dumps(definition['parameters'], ensure_ascii=False)}`",
            f"- Justification pré-déclarée : {definition['rationale']}",
            f"- Plateau détecté : **{'oui' if item['plateau_detected'] else 'non'}** ; nœuds qualifiés : {int(item['plateau_node_count'])}",
            f"- Région qualifiée : `{item['plateau_parameter_region']}`",
            f"- Tableau complet : `tables/{mechanism_id}.csv`",
            f"- Heatmaps : `heatmaps/{mechanism_id}_sharpe_annualized.png`, "
            f"`heatmaps/{mechanism_id}_expectancy_per_trade_usd.png`, `heatmaps/{mechanism_id}_trade_count.png`",
            "",
        ])
    lines.extend([
        "## Limites méthodologiques", "",
        "Les paysages décrivent uniquement la stabilité locale sur l'IS. Les cellules partagent les mêmes observations "
        "et ne constituent pas des tests indépendants. Le score de voisinage réduit l'attrait d'un maximum isolé, mais "
        "ne remplace pas la validation temporelle réservée. Aucun mécanisme n'est éliminé à cette étape.", "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def execute(data_path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    config = _load_locked_grid()
    grid_hash = _sha256(GRID_PATH)
    frozen_before = _frozen_hashes()
    specs, benchmark_spec, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    spec_by_id = {spec["mechanism_id"]: spec for spec in specs}
    is_start = date.fromisoformat(config["is_start"])
    is_end = date.fromisoformat(config["is_end"])
    prepared = prepare_data(data_path, start_date=is_start, end_date=is_end)
    benchmark = build_buy_and_hold(benchmark_spec, prepared)
    penalty = float(config["stability"]["penalty"])
    all_frames: list[pd.DataFrame] = []
    completed = 0
    total = sum(len(_combinations(item)) for item in config["mechanisms"].values())

    for mechanism_id, definition in config["mechanisms"].items():
        frozen_spec = spec_by_id[mechanism_id]
        parameter_names = list(definition["parameters"])
        rows: list[dict[str, Any]] = []
        for local_number, values in enumerate(_combinations(definition), 1):
            completed += 1
            print(f"[{completed:03d}/{total}] {mechanism_id} {values}", flush=True)
            spec = _apply_parameters(frozen_spec, values)
            run = run_baseline_backtest(spec, prepared)
            metrics = compute_metrics(spec, run, benchmark.daily)
            if not metrics["backtest_valid"]:
                raise RuntimeError(f"Implementation checks failed for {mechanism_id} {values}")
            row = {
                "mechanism_id": mechanism_id,
                "strategy_id": spec["strategy_id"],
                "complexity_grade": spec["complexity_grade"],
                "combination_id": f"{mechanism_id}__{local_number:02d}",
                "parameter_1_name": parameter_names[0],
                "parameter_1_value": values[parameter_names[0]],
                "parameter_1_index": definition["parameters"][parameter_names[0]].index(values[parameter_names[0]]),
                "parameter_2_name": parameter_names[1],
                "parameter_2_value": values[parameter_names[1]],
                "parameter_2_index": definition["parameters"][parameter_names[1]].index(values[parameter_names[1]]),
                "is_canonical": _is_canonical(values, definition),
                "sharpe_annualized": metrics["sharpe_annualized"],
                "expectancy_per_trade_usd": metrics["expectancy_per_trade_usd"],
                "profit_factor": metrics["profit_factor"],
                "max_drawdown_usd": metrics["max_drawdown_usd"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "trade_count": metrics["trade_count"],
                "net_pnl_usd": metrics["net_pnl_usd"],
                "implementation_warning": metrics["implementation_warning"],
                "backtest_valid": metrics["backtest_valid"],
                "frozen_spec_sha256": metrics["frozen_spec_sha256"],
                "is_start": config["is_start"], "is_end": config["is_end"],
                "reserved_oos_start": config["reserved_oos_start"], "reserved_oos_end": config["reserved_oos_end"],
            }
            rows.append(row)
        mechanism_frame = _compute_neighborhoods(pd.DataFrame(rows), definition, penalty)
        mechanism_frame.to_csv(TABLE_DIR / f"{mechanism_id}.csv", index=False)
        for metric, title, fmt in [
            ("sharpe_annualized", "Sharpe annualisé", ".2f"),
            ("expectancy_per_trade_usd", "Expectancy par trade (USD)", ".1f"),
            ("trade_count", "Nombre de trades", ".0f"),
        ]:
            _plot_heatmap(mechanism_frame, mechanism_id, metric, title, fmt)
        all_frames.append(mechanism_frame)

    results = pd.concat(all_frames, ignore_index=True)
    results.to_csv(RESULT_PATH, index=False)
    summary = pd.DataFrame([_summarize(frame) for _, frame in results.groupby("mechanism_id", sort=True)])
    summary.to_csv(SUMMARY_PATH, index=False)
    frozen_after = _frozen_hashes()
    if frozen_before != frozen_after:
        raise RuntimeError("A frozen baseline file changed during the landscape study")
    metadata = {
        "run_type": "parameter_landscape_is_only", "grid_sha256": grid_hash,
        "data_path": str(data_path.relative_to(REPO_ROOT)), "data_sha256": _sha256(data_path),
        "is_start": config["is_start"], "is_end": config["is_end"],
        "reserved_oos_start": config["reserved_oos_start"], "reserved_oos_end": config["reserved_oos_end"],
        "oos_loaded": False, "combination_count": int(len(results)),
        "valid_combination_count": int(results["backtest_valid"].astype(bool).sum()),
        "frozen_files_unchanged": True, "frozen_sha256": frozen_before,
        "risk_execution_session_sizing_fixed": True, "variants_opened": 0, "orb_used": False,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(config, results, summary, metadata)
    print(f"Combinations executed: {len(results)}", flush=True)
    print(f"Valid combinations: {int(results['backtest_valid'].astype(bool).sum())}", flush=True)
    print(f"Plateaus detected: {int(summary['plateau_detected'].astype(bool).sum())}", flush=True)
    print("OOS loaded: 0", flush=True)
    print("Frozen files modified: 0", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    execute(args.data.resolve())


if __name__ == "__main__":
    main()
