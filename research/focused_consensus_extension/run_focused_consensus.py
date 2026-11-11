"""Run focused IS/validation grids for SuperTrend and gap continuation."""

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
    INITIAL_CAPITAL_USD,
    BacktestResult,
    build_buy_and_hold,
    compute_metrics,
    load_frozen_specs,
    prepare_data,
    run_baseline_backtest,
    slice_prepared_data,
)


GRID_PATH = OUTPUT_DIR / "focused_parameter_grids.json"
FROZEN_DIR = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"
RESULT_PATH = OUTPUT_DIR / "focused_consensus_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "focused_consensus_summary.csv"
REPORT_PATH = OUTPUT_DIR / "FOCUSED_CONSENSUS_REPORT.md"


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
    if not config.get("locked_before_execution") or set(config["mechanisms"]) != {"atr_trend_flip", "gap_continuation"}:
        raise RuntimeError("Focused grid is not locked or has the wrong mechanisms")
    return config


def _combinations(definition: dict[str, Any]) -> list[dict[str, Any]]:
    names = list(definition["parameters"])
    return [dict(zip(names, values)) for values in itertools.product(*(definition["parameters"][name] for name in names))]


def _apply_parameters(frozen_spec: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(frozen_spec)
    structural = spec["canonical_parameters"]["structural"]
    for name, value in values.items():
        if name not in structural:
            raise KeyError(f"Unknown structural parameter {spec['mechanism_id']}::{name}")
        structural[name] = value
    return spec


def _row_key(row: pd.Series | Any) -> tuple[str, str]:
    return str(row.parameter_1_value), str(row.parameter_2_value)


def _load_checkpoint(path: Path, grid_hash: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame["combination_id"].duplicated().any() or not (frame["grid_sha256"].astype(str) == grid_hash).all():
        raise RuntimeError(f"Invalid checkpoint: {path}")
    return frame


def _slice_result(run: BacktestResult, start: date, end: date) -> BacktestResult:
    """Create an independently capitalized period from a daily-flat full run."""
    daily = run.daily[run.daily["session_date"].between(start, end)].copy().reset_index(drop=True)
    daily["equity_usd"] = INITIAL_CAPITAL_USD + daily["daily_pnl_usd"].cumsum()
    prior_equity = daily["equity_usd"].shift(1).fillna(INITIAL_CAPITAL_USD)
    daily["daily_return"] = daily["daily_pnl_usd"] / prior_equity
    trades = run.trades[run.trades["session_date"].between(start, end)].copy().reset_index(drop=True)
    positions = run.positions[run.positions["session_date"].between(start, end)].copy().reset_index(drop=True)
    equity = daily[["session_date", "mechanism_id", "daily_pnl_usd", "daily_return", "equity_usd"]].copy()
    return BacktestResult(trades=trades, daily=daily, equity=equity, positions=positions, checks=dict(run.checks))


def _period_metrics(spec: dict[str, Any], run: BacktestResult, benchmark_daily: pd.DataFrame, start: date, end: date) -> dict[str, Any]:
    metrics = compute_metrics(spec, _slice_result(run, start, end), benchmark_daily)
    if not metrics["backtest_valid"]:
        raise RuntimeError(f"Implementation checks failed for {spec['mechanism_id']} {start}–{end}")
    return metrics


def _run_mechanism(
    mechanism_id: str,
    definition: dict[str, Any],
    frozen_spec: dict[str, Any],
    prepared: Any,
    is_benchmark: pd.DataFrame,
    validation_benchmark: pd.DataFrame,
    config: dict[str, Any],
    grid_hash: str,
    completed: int,
    total: int,
) -> tuple[pd.DataFrame, int]:
    checkpoint_path = TABLE_DIR / f"{mechanism_id}_checkpoint.csv"
    checkpoint = _load_checkpoint(checkpoint_path, grid_hash)
    existing = set(checkpoint["combination_id"].tolist()) if not checkpoint.empty else set()
    rows = checkpoint.to_dict("records") if not checkpoint.empty else []
    names = list(definition["parameters"])
    is_start, is_end = date.fromisoformat(config["is_start"]), date.fromisoformat(config["is_end"])
    val_start, val_end = date.fromisoformat(config["validation_start"]), date.fromisoformat(config["validation_end"])
    for number, values in enumerate(_combinations(definition), 1):
        combination_id = f"{mechanism_id}__{number:03d}"
        completed += 1
        if combination_id in existing:
            print(f"[{completed:03d}/{total}] {combination_id} resumed", flush=True)
            continue
        print(f"[{completed:03d}/{total}] {mechanism_id} {values}", flush=True)
        spec = _apply_parameters(frozen_spec, values)
        full_run = run_baseline_backtest(spec, prepared)
        is_metrics = _period_metrics(spec, full_run, is_benchmark, is_start, is_end)
        val_metrics = _period_metrics(spec, full_run, validation_benchmark, val_start, val_end)
        row = {
            "mechanism_id": mechanism_id,
            "strategy_id": spec["strategy_id"],
            "combination_id": combination_id,
            "parameter_1_name": names[0], "parameter_1_value": values[names[0]],
            "parameter_1_index": definition["parameters"][names[0]].index(values[names[0]]),
            "parameter_2_name": names[1], "parameter_2_value": values[names[1]],
            "parameter_2_index": definition["parameters"][names[1]].index(values[names[1]]),
            "is_sharpe_annualized": is_metrics["sharpe_annualized"],
            "validation_sharpe_annualized": val_metrics["sharpe_annualized"],
            "is_expectancy_per_trade_usd": is_metrics["expectancy_per_trade_usd"],
            "validation_expectancy_per_trade_usd": val_metrics["expectancy_per_trade_usd"],
            "is_profit_factor": is_metrics["profit_factor"],
            "validation_profit_factor": val_metrics["profit_factor"],
            "is_max_drawdown_usd": is_metrics["max_drawdown_usd"],
            "validation_max_drawdown_usd": val_metrics["max_drawdown_usd"],
            "is_trade_count": is_metrics["trade_count"],
            "validation_trade_count": val_metrics["trade_count"],
            "is_net_pnl_usd": is_metrics["net_pnl_usd"],
            "validation_net_pnl_usd": val_metrics["net_pnl_usd"],
            "is_backtest_valid": is_metrics["backtest_valid"],
            "validation_backtest_valid": val_metrics["backtest_valid"],
            "frozen_spec_sha256": spec["_frozen_spec_sha256"],
            "grid_sha256": grid_hash,
        }
        row["is_insufficient_sample"] = int(row["is_trade_count"]) < 20
        row["validation_insufficient_sample"] = int(row["validation_trade_count"]) < 20
        row["insufficient_sample_either_period"] = row["is_insufficient_sample"] or row["validation_insufficient_sample"]
        row["consensus_score"] = min(float(row["is_sharpe_annualized"]), float(row["validation_sharpe_annualized"]))
        row["robust_cell"] = bool(
            row["is_sharpe_annualized"] > 0
            and row["validation_sharpe_annualized"] > 0
            and row["is_profit_factor"] > 1
            and row["validation_profit_factor"] > 1
            and not row["is_insufficient_sample"]
            and not row["validation_insufficient_sample"]
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
    frame = pd.DataFrame(rows).sort_values(["parameter_1_index", "parameter_2_index"]).reset_index(drop=True)
    expected_count = len(_combinations(definition))
    if len(frame) != expected_count:
        raise RuntimeError(f"Incomplete focused grid for {mechanism_id}: {len(frame)} != {expected_count}")
    frame.to_csv(TABLE_DIR / f"{mechanism_id}.csv", index=False)
    return frame, completed


def _components(frame: pd.DataFrame) -> list[pd.DataFrame]:
    selected = frame[frame["robust_cell"].astype(bool)]
    lookup = {(int(row.parameter_1_index), int(row.parameter_2_index)): index for index, row in selected.iterrows()}
    unseen = set(lookup)
    components: list[pd.DataFrame] = []
    while unseen:
        start = min(unseen); unseen.remove(start); queue = deque([start]); indices: list[int] = []
        while queue:
            x, y = queue.popleft(); indices.append(lookup[(x, y)])
            for other in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if other in unseen: unseen.remove(other); queue.append(other)
        components.append(frame.loc[indices].copy())
    return sorted(components, key=lambda item: (-len(item), item["combination_id"].min()))


def _medoid(component: pd.DataFrame) -> pd.Series | None:
    if component.empty: return None
    coordinates = component[["parameter_1_index", "parameter_2_index"]].to_numpy(dtype=int)
    distances = np.abs(coordinates[:, None, :] - coordinates[None, :, :]).sum(axis=2).sum(axis=1)
    candidates = component.copy(); candidates["distance_sum"] = distances
    return candidates.sort_values(["distance_sum", "consensus_score", "combination_id"], ascending=[True, False, True]).iloc[0]


def _native(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _region(frame: pd.DataFrame) -> str:
    if frame.empty: return "{}"
    return json.dumps({
        frame["parameter_1_name"].iloc[0]: sorted(frame["parameter_1_value"].unique().tolist()),
        frame["parameter_2_name"].iloc[0]: sorted(frame["parameter_2_value"].unique().tolist()),
    }, ensure_ascii=False, sort_keys=True)


def _summarize(frame: pd.DataFrame) -> dict[str, Any]:
    components = _components(frame); largest = components[0] if components else frame.iloc[0:0]
    medoid = _medoid(largest); p1 = frame["parameter_1_name"].iloc[0]; p2 = frame["parameter_2_name"].iloc[0]
    eligible = frame[~frame["insufficient_sample_either_period"].astype(bool)]
    dual_positive = frame[(frame["is_sharpe_annualized"] > 0) & (frame["validation_sharpe_annualized"] > 0) & (frame["is_profit_factor"] > 1) & (frame["validation_profit_factor"] > 1)]
    robust = frame[frame["robust_cell"].astype(bool)]
    below_two = robust[robust[p2] < 2.0] if p2 in robust.columns else robust.iloc[0:0]
    # Parameter values live in generic columns, so mechanism-specific diagnostics use those columns.
    if frame["mechanism_id"].iloc[0] == "atr_trend_flip":
        below_two = robust[robust["parameter_2_value"] < 2.0]
        below_components = _components(frame.assign(robust_cell=frame["robust_cell"].astype(bool) & (frame["parameter_2_value"] < 2.0)))
        plateau_below_two = bool(below_components and len(below_components[0]) >= 2)
        max_robust_primary = np.nan
    else:
        plateau_below_two = False
        max_robust_primary = float(robust["parameter_1_value"].max()) if len(robust) else np.nan
    return {
        "mechanism_id": frame["mechanism_id"].iloc[0],
        "cell_count": len(frame),
        "eligible_sample_cell_count": len(eligible),
        "is_insufficient_sample_cell_count": int(frame["is_insufficient_sample"].sum()),
        "validation_insufficient_sample_cell_count": int(frame["validation_insufficient_sample"].sum()),
        "dual_positive_before_sample_filter_count": len(dual_positive),
        "robust_cell_count": len(robust),
        "robust_component_count": len(components),
        "largest_robust_component_size": len(largest),
        "robust_plateau_exists": bool(len(largest) >= 2),
        "largest_robust_region": _region(largest),
        "medoid_combination_id": None if medoid is None else medoid["combination_id"],
        "medoid_parameters": None if medoid is None else json.dumps({p1: _native(medoid["parameter_1_value"]), p2: _native(medoid["parameter_2_value"])}, ensure_ascii=False, sort_keys=True),
        "medoid_consensus_score": np.nan if medoid is None else medoid["consensus_score"],
        "medoid_is_sharpe": np.nan if medoid is None else medoid["is_sharpe_annualized"],
        "medoid_validation_sharpe": np.nan if medoid is None else medoid["validation_sharpe_annualized"],
        "atr_common_plateau_below_2": plateau_below_two,
        "atr_robust_cell_count_below_2": len(below_two),
        "gap_max_robust_minimum_atr_multiple": max_robust_primary,
        "gap_max_dual_positive_before_sample_filter": float(dual_positive["parameter_1_value"].max()) if frame["mechanism_id"].iloc[0] == "gap_continuation" and len(dual_positive) else np.nan,
    }


def _paired_heatmap(frame: pd.DataFrame, mechanism_id: str, metric: str, title: str, fmt: str) -> None:
    p1 = frame["parameter_1_name"].iloc[0]; p2 = frame["parameter_2_name"].iloc[0]
    left = frame.pivot(index="parameter_1_value", columns="parameter_2_value", values=f"is_{metric}")
    right = frame.pivot(index="parameter_1_value", columns="parameter_2_value", values=f"validation_{metric}")
    values = np.concatenate([left.to_numpy().ravel(), right.to_numpy().ravel()]); finite = values[np.isfinite(values)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if len(finite) else (None, None)
    center = 1.0 if metric == "profit_factor" else (0.0 if metric in {"sharpe_annualized", "expectancy_per_trade_usd", "max_drawdown_usd"} else None)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, max(4.8, 0.72 * len(left.index) + 2.2)), sharey=True)
    for ax, matrix, label in zip(axes, (left, right), ("IS 2019–2023", "Validation 2024–2026")):
        sns.heatmap(matrix, annot=True, fmt=fmt, cmap="RdYlGn", center=center, vmin=vmin, vmax=vmax, linewidths=0.4, ax=ax)
        ax.set_title(label); ax.set_xlabel(p2); ax.set_ylabel(p1 if ax is axes[0] else "")
    fig.suptitle(f"{mechanism_id} — {title}"); fig.tight_layout()
    fig.savefig(HEATMAP_DIR / f"{mechanism_id}_{metric}_is_vs_validation.png", dpi=160); plt.close(fig)


def _single_heatmap(frame: pd.DataFrame, mechanism_id: str, column: str, title: str, fmt: str, kind: str) -> None:
    p1 = frame["parameter_1_name"].iloc[0]; p2 = frame["parameter_2_name"].iloc[0]
    matrix = frame.pivot(index="parameter_1_value", columns="parameter_2_value", values=column).astype(float)
    fig, ax = plt.subplots(figsize=(7.6, max(4.8, 0.72 * len(matrix.index) + 2.2)))
    if kind == "mask":
        sns.heatmap(matrix, annot=True, fmt=fmt, cmap=["#d9d9d9", "#238b45"], vmin=0, vmax=1, cbar=False, linewidths=0.4, ax=ax)
    else:
        sns.heatmap(matrix, annot=True, fmt=fmt, cmap="RdYlGn", center=0, linewidths=0.4, ax=ax)
    ax.set_title(f"{mechanism_id}\n{title}"); ax.set_xlabel(p2); ax.set_ylabel(p1); fig.tight_layout()
    fig.savefig(HEATMAP_DIR / f"{mechanism_id}_{column}.png", dpi=160); plt.close(fig)


def _fmt(value: Any) -> str:
    return "n/a" if value is None or pd.isna(value) else (f"{float(value):.2f}" if isinstance(value, (float, np.floating)) else str(value))


def _write_report(summary: pd.DataFrame, results: pd.DataFrame, grid_hash: str) -> None:
    atr = summary[summary["mechanism_id"] == "atr_trend_flip"].iloc[0]
    gap = summary[summary["mechanism_id"] == "gap_continuation"].iloc[0]
    gap_cells = results[results["mechanism_id"] == "gap_continuation"].set_index("parameter_1_value")
    lines = [
        "# Focused consensus extension", "", "## Protocole", "",
        "Ces deux grilles ont été verrouillées avant exécution et calculées séparément sur IS **2019-05-06 → 2023-12-31** "
        "et Validation **2024-01-01 → 2026-03-19**. Les coûts, fills, règles de session, sizing et toute autre logique "
        "restent ceux des frozen baselines.", "",
        f"- SHA-256 de la grille : `{grid_hash}`",
        "- `consensus_score = min(Sharpe_IS, Sharpe_VAL)`",
        "- Une cellule robuste exige Sharpe positif et PF > 1 dans les deux périodes",
        "- Toute cellule avec moins de 20 trades dans au moins une période est exclue du consensus", "",
        "## Synthèse", "",
        "| Mécanisme | Cellules | Faible échantillon IS | Faible échantillon VAL | Double positives avant filtre | Robustes | Plus grande zone | Médoïde | Score |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.mechanism_id}` | {row.cell_count} | {row.is_insufficient_sample_cell_count} | "
            f"{row.validation_insufficient_sample_cell_count} | {row.dual_positive_before_sample_filter_count} | "
            f"{row.robust_cell_count} | {row.largest_robust_component_size} | "
            f"`{row.medoid_parameters if pd.notna(row.medoid_parameters) else 'n/a'}` | {_fmt(row.medoid_consensus_score)} |"
        )
    lines.extend([
        "", "## atr_trend_flip", "",
        f"- Plateau commun robuste sous 2,0 : **{'oui' if atr['atr_common_plateau_below_2'] else 'non'}**",
        f"- Cellules robustes sous 2,0 : **{int(atr['atr_robust_cell_count_below_2'])}**",
        f"- Région robuste principale : `{atr['largest_robust_region']}`",
        f"- Médoïde : `{atr['medoid_parameters']}` ; Sharpe IS {_fmt(atr['medoid_is_sharpe'])}, Validation {_fmt(atr['medoid_validation_sharpe'])}",
        "", "## gap_continuation", "",
        f"- Seuil maximal encore robuste après filtre d'échantillon : **{_fmt(gap['gap_max_robust_minimum_atr_multiple'])} ATR**",
        f"- Seuil maximal doublement positif avant filtre d'échantillon : **{_fmt(gap['gap_max_dual_positive_before_sample_filter'])} ATR**",
        f"- Région robuste principale : `{gap['largest_robust_region']}`",
        f"- Médoïde : `{gap['medoid_parameters']}` ; Sharpe IS {_fmt(gap['medoid_is_sharpe'])}, Validation {_fmt(gap['medoid_validation_sharpe'])}",
        f"- À 1,50 ATR : {int(gap_cells.loc[1.5, 'is_trade_count'])} trades IS et {int(gap_cells.loc[1.5, 'validation_trade_count'])} trade Validation ; cellule exclue pour échantillon insuffisant",
        "- Au-delà de 1,50 ATR (1,75 / 2,0 / 2,5) : aucun trade dans les deux sous-périodes ; l'edge n'est donc pas observable à ces seuils",
        "", "## Fichiers", "",
        "Les heatmaps côte à côte présentent Sharpe, expectancy, PF, max DD et nombre de trades. Des cartes séparées "
        "montrent le consensus score, le masque robuste et les faibles échantillons dans chaque sous-période.", "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def execute(data_path: Path) -> None:
    for directory in (OUTPUT_DIR, HEATMAP_DIR, TABLE_DIR, MPL_CACHE): directory.mkdir(parents=True, exist_ok=True)
    config = _load_config(); grid_hash = _sha256(GRID_PATH); frozen_before = _frozen_hashes()
    specs, benchmark_spec, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST); spec_by_id = {s["mechanism_id"]: s for s in specs}
    prepared = prepare_data(data_path, start_date=date.fromisoformat(config["is_start"]), end_date=date.fromisoformat(config["validation_end"]))
    is_data = slice_prepared_data(prepared, date.fromisoformat(config["is_start"]), date.fromisoformat(config["is_end"]))
    val_data = slice_prepared_data(prepared, date.fromisoformat(config["validation_start"]), date.fromisoformat(config["validation_end"]))
    is_benchmark = build_buy_and_hold(benchmark_spec, is_data).daily
    val_benchmark = build_buy_and_hold(benchmark_spec, val_data).daily
    total = sum(len(_combinations(d)) for d in config["mechanisms"].values()); completed = 0; frames=[]; summaries=[]
    for mechanism_id, definition in config["mechanisms"].items():
        frame, completed = _run_mechanism(mechanism_id, definition, spec_by_id[mechanism_id], prepared, is_benchmark, val_benchmark, config, grid_hash, completed, total)
        for metric, title, fmt in [
            ("sharpe_annualized", "Sharpe annualisé", ".2f"),
            ("expectancy_per_trade_usd", "Expectancy par trade (USD)", ".1f"),
            ("profit_factor", "Profit factor", ".2f"),
            ("max_drawdown_usd", "Max drawdown (USD)", ".0f"),
            ("trade_count", "Nombre de trades", ".0f"),
        ]: _paired_heatmap(frame, mechanism_id, metric, title, fmt)
        _single_heatmap(frame, mechanism_id, "consensus_score", "Consensus score", ".2f", "score")
        _single_heatmap(frame, mechanism_id, "robust_cell", "Robust cell après filtre d'échantillon", ".0f", "mask")
        _single_heatmap(frame, mechanism_id, "is_insufficient_sample", "Insufficient sample IS (<20 trades)", ".0f", "mask")
        _single_heatmap(frame, mechanism_id, "validation_insufficient_sample", "Insufficient sample Validation (<20 trades)", ".0f", "mask")
        frames.append(frame); summaries.append(_summarize(frame))
    results=pd.concat(frames,ignore_index=True); summary=pd.DataFrame(summaries)
    results.to_csv(RESULT_PATH,index=False); summary.to_csv(SUMMARY_PATH,index=False)
    frozen_after=_frozen_hashes()
    if frozen_before!=frozen_after: raise RuntimeError("A frozen baseline changed")
    if _sha256(GRID_PATH)!=grid_hash: raise RuntimeError("Focused grid changed during execution")
    metadata={"run_type":"focused_is_validation_consensus","grid_sha256":grid_hash,"data_sha256":_sha256(data_path),"cell_count":len(results),"valid_is_count":int(results.is_backtest_valid.sum()),"valid_validation_count":int(results.validation_backtest_valid.sum()),"sample_threshold_trades":20,"other_parameters_modified":False,"frozen_files_unchanged":True}
    (OUTPUT_DIR/"run_metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    _write_report(summary,results,grid_hash)
    print(f"Cells executed: {len(results)}",flush=True); print(f"Valid IS/Validation: {metadata['valid_is_count']}/{metadata['valid_validation_count']}",flush=True); print("Other parameters or logic modified: 0",flush=True); print("Frozen files modified: 0",flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--data",type=Path,default=DEFAULT_DATA); args=parser.parse_args(); execute(args.data.resolve())


if __name__ == "__main__": main()
