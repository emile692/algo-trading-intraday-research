"""Replicate four locked MNQ mechanisms on MES and M2K with small prelocked grids."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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

from research.baseline_backtests.baseline_engine import load_frozen_specs, prepare_data, run_baseline_backtest  # noqa: E402
from research.money_management_lab.mm_engine import build_overlay_context, run_exit_overlay_backtest, slice_result  # noqa: E402
from src.analytics.orb_research.exits import run_exit_variant_backtest  # noqa: E402
from src.analytics.orb_research.features import (  # noqa: E402
    apply_ensemble_selection,
    attach_daily_reference,
    build_candidate_universe,
    build_daily_reference,
    build_signal_frame_for_backtest,
    calibrate_ensemble_thresholds,
    first_pass_signal_rows,
    prepare_minute_dataset,
)
from src.analytics.orb_research.types import BaselineEnsembleConfig, BaselineEntryConfig, CompressionConfig, ExitConfig  # noqa: E402
from src.engine.execution_model import ExecutionModel  # noqa: E402


CONFIG_PATH = OUTPUT_DIR / "cross_asset_replication_grid.json"
CONFIG_HASH_PATH = OUTPUT_DIR / "cross_asset_replication_grid.sha256"
FROZEN_DIR = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
INITIAL_CAPITAL = 50_000.0
METRICS = ["sharpe", "expectancy", "profit_factor", "max_drawdown", "trade_count"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_from_trades(trades: pd.DataFrame, sessions: list[Any]) -> dict[str, float]:
    session_dates = pd.Index([pd.Timestamp(day).date() for day in sessions], name="session_date")
    if trades.empty:
        daily_pnl = pd.Series(0.0, index=session_dates)
        trade_pnl = pd.Series(dtype=float)
    else:
        frame = trades.copy()
        frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date
        trade_pnl = frame["net_pnl_usd"].astype(float)
        daily_pnl = frame.groupby("session_date")["net_pnl_usd"].sum().reindex(session_dates).fillna(0.0)
    equity = INITIAL_CAPITAL + daily_pnl.cumsum()
    prior = equity.shift(1).fillna(INITIAL_CAPITAL)
    returns = daily_pnl / prior
    std = float(returns.std(ddof=1))
    peak = equity.cummax().clip(lower=INITIAL_CAPITAL)
    gains = float(trade_pnl[trade_pnl > 0].sum())
    losses = abs(float(trade_pnl[trade_pnl < 0].sum()))
    return {
        "sharpe": float(returns.mean() / std * math.sqrt(252)) if std > 0 else np.nan,
        "expectancy": float(trade_pnl.mean()) if len(trade_pnl) else np.nan,
        "profit_factor": gains / losses if losses > 0 else np.nan,
        "max_drawdown": float((equity - peak).min()),
        "trade_count": int(len(trade_pnl)),
    }


def metrics_from_run(run: Any, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    sliced = slice_result(run, start.date(), end.date())
    return metrics_from_trades(sliced.trades, sliced.daily["session_date"].tolist())


def adapt_spec(spec: dict[str, Any], asset: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(spec)
    structural = updated["canonical_parameters"]["structural"]
    for name, value in parameters.items():
        if name not in structural:
            raise KeyError(f"Unknown structural parameter {updated['mechanism_id']}::{name}")
        structural[name] = value
    execution = updated["canonical_parameters"]["execution"]
    for name in ("tick_size", "tick_value_usd", "point_value_usd", "commission_per_side_usd", "slippage_ticks_per_side"):
        execution[name] = asset[name]
    updated["canonical_parameters"]["risk_management"]["fixed_quantity"] = 1
    return updated


def orb_grid(asset_name: str, asset: dict[str, Any], definition: dict[str, Any], periods: dict[str, list[str]]) -> list[dict[str, Any]]:
    dataset = REPO_ROOT / asset["data"]
    all_rows: list[dict[str, Any]] = []
    is_start, is_end = map(pd.Timestamp, periods["is"])
    val_start, val_end = map(pd.Timestamp, periods["validation"])
    execution = ExecutionModel(
        commission_per_side_usd=float(asset["commission_per_side_usd"]),
        slippage_ticks=float(asset["slippage_ticks_per_side"]),
        tick_size=float(asset["tick_size"]),
    )
    ensemble_base = BaselineEnsembleConfig()
    compression = CompressionConfig(mode="none", usage="hard_filter")
    for or_minutes in definition["values_1"]:
        entry = replace(
            BaselineEntryConfig(), or_minutes=int(or_minutes), tick_size=float(asset["tick_size"]),
            target_multiple=2.0, stop_buffer_ticks=2, risk_per_trade_pct=0.5, account_size_usd=INITIAL_CAPITAL,
        )
        minute = prepare_minute_dataset(dataset, entry, (14,))
        minute = attach_daily_reference(minute, build_daily_reference(minute))
        candidates = build_candidate_universe(minute, baseline_entry=entry)
        sessions = sorted(pd.to_datetime(minute["session_date"]).dt.date.unique())
        is_sessions = [day for day in sessions if is_start.date() <= day <= is_end.date()]
        pass_rows = first_pass_signal_rows(candidates, candidates["candidate_base_pass"].fillna(False).astype(bool))
        thresholds = calibrate_ensemble_thresholds(
            pass_rows, is_sessions, "atr_14", ensemble_base.q_lows_pct, ensemble_base.q_highs_pct,
        )
        for vote in definition["values_2"]:
            ensemble = replace(ensemble_base, vote_threshold=float(vote))
            scored = apply_ensemble_selection(
                pass_rows, "atr_14", thresholds, ensemble.vote_threshold, compression,
            )
            selected = scored.loc[scored.get("ensemble_selected", False)].copy()
            signal = build_signal_frame_for_backtest(minute, selected)
            trades = run_exit_variant_backtest(
                signal, execution, entry, ExitConfig(mode="baseline"),
                tick_value_usd=float(asset["tick_value_usd"]), point_value_usd=float(asset["point_value_usd"]),
            )
            for period_name, (start_text, end_text) in periods.items():
                start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
                period_sessions = [day for day in sessions if start.date() <= day <= end.date()]
                if trades.empty:
                    period_trades = trades.copy()
                else:
                    dates = pd.to_datetime(trades["session_date"]).dt.date
                    period_trades = trades[dates.between(start.date(), end.date())].copy()
                row = metrics_from_trades(period_trades, period_sessions)
                all_rows.append({"asset": asset_name, "mechanism": "orb_vwap_atr", "parameter_1": or_minutes, "parameter_2": vote, "period": period_name, **row})
    return all_rows


def alpha_grids(asset_name: str, asset: dict[str, Any], mechanisms: dict[str, Any], periods: dict[str, list[str]]) -> list[dict[str, Any]]:
    dataset = REPO_ROOT / asset["data"]
    start = pd.Timestamp(periods["is"][0])
    end = pd.Timestamp(periods["validation"][1])
    prepared = prepare_data(dataset, start_date=start.date(), end_date=end.date())
    specs, _, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    spec_by_id = {spec["mechanism_id"]: spec for spec in specs}
    rows: list[dict[str, Any]] = []
    for mechanism in ("volatility_breakout", "compression_breakout", "atr_trend_flip"):
        definition = mechanisms[mechanism]
        for value_1, value_2 in itertools.product(definition["values_1"], definition["values_2"]):
            spec = adapt_spec(spec_by_id[mechanism], asset, {definition["parameter_1"]: value_1, definition["parameter_2"]: value_2})
            if definition["fixed_exit"] == "locked_baseline":
                run = run_baseline_backtest(spec, prepared, evaluation_start=start.date(), evaluation_end=end.date())
            else:
                policy = definition["fixed_exit"]
                timeframe = int(definition["signal_timeframe_minutes"])
                context = build_overlay_context(spec, prepared, timeframe)
                run = run_exit_overlay_backtest(spec, prepared, policy, timeframe, context=context)
            violations = {k: v for k, v in run.checks.items() if (k.endswith("violations") or k == "missing_atr_entry_skips") and v}
            if violations:
                raise RuntimeError(f"{asset_name} {mechanism} {value_1}/{value_2}: {violations}")
            for period_name, (start_text, end_text) in periods.items():
                values = metrics_from_run(run, pd.Timestamp(start_text), pd.Timestamp(end_text))
                rows.append({"asset": asset_name, "mechanism": mechanism, "parameter_1": value_1, "parameter_2": value_2, "period": period_name, **values})
    return rows


def run_asset(asset_name: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    asset = config["assets"][asset_name]
    rows = orb_grid(asset_name, asset, config["mechanisms"]["orb_vwap_atr"], config["periods"])
    rows.extend(alpha_grids(asset_name, asset, config["mechanisms"], config["periods"]))
    return rows


def largest_component(mask: np.ndarray) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    best: list[tuple[int, int]] = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if not mask[i, j] or (i, j) in seen:
                continue
            stack, component = [(i, j)], []
            seen.add((i, j))
            while stack:
                cell = stack.pop(); component.append(cell)
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt = (cell[0] + di, cell[1] + dj)
                    if 0 <= nxt[0] < mask.shape[0] and 0 <= nxt[1] < mask.shape[1] and mask[nxt] and nxt not in seen:
                        seen.add(nxt); stack.append(nxt)
            if len(component) > len(best):
                best = component
    return best


def analyze(results: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_cols = ["asset", "mechanism", "parameter_1", "parameter_2"]
    wide = results.pivot(index=index_cols, columns="period", values=METRICS).reset_index()
    wide.columns = ["_".join(filter(None, col)).strip("_") if isinstance(col, tuple) else col for col in wide.columns]
    wide["robust_cell"] = (
        wide["sharpe_is"].gt(0) & wide["sharpe_validation"].gt(0)
        & wide["profit_factor_is"].gt(1) & wide["profit_factor_validation"].gt(1)
        & wide["trade_count_is"].ge(20) & wide["trade_count_validation"].ge(20)
    )
    summaries = []
    for (asset, mechanism), group in wide.groupby(["asset", "mechanism"]):
        definition = config["mechanisms"][mechanism]
        values_1, values_2 = list(map(float, definition["values_1"])), list(map(float, definition["values_2"]))
        robust = {(float(r.parameter_1), float(r.parameter_2)) for r in group.itertuples() if r.robust_cell}
        mask = np.array([[((v1, v2) in robust) for v2 in values_2] for v1 in values_1], dtype=bool)
        component = largest_component(mask)
        spans = len({i for i, _ in component}) >= 2 and len({j for _, j in component}) >= 2
        plateau = len(component) >= 3 and spans
        sufficient_share = float((group["trade_count_is"].ge(20) & group["trade_count_validation"].ge(20)).mean())
        if plateau:
            classification = "portable"
        elif sufficient_share >= 0.8 and not robust:
            classification = "asset_specific"
        else:
            classification = "inconclusive"
        summaries.append({
            "asset": asset, "mechanism": mechanism, "classification": classification,
            "robust_cell_count": int(mask.sum()), "largest_robust_component": len(component),
            "common_plateau": plateau, "sufficient_sample_cell_share": sufficient_share,
            "component_cells": json.dumps([[values_1[i], values_2[j]] for i, j in component]),
        })
    return wide, pd.DataFrame(summaries)


def plot_heatmaps(wide: pd.DataFrame, config: dict[str, Any]) -> None:
    labels = {"sharpe": "Sharpe", "expectancy": "Expectancy ($)", "profit_factor": "PF", "max_drawdown": "Max DD ($)", "trade_count": "Trades"}
    for (asset, mechanism), group in wide.groupby(["asset", "mechanism"]):
        definition = config["mechanisms"][mechanism]
        fig, axes = plt.subplots(5, 2, figsize=(12, 20))
        for row_idx, metric in enumerate(METRICS):
            for col_idx, period in enumerate(("is", "validation")):
                pivot = group.pivot(index="parameter_1", columns="parameter_2", values=f"{metric}_{period}").sort_index().sort_index(axis=1)
                sns.heatmap(pivot, annot=True, fmt=".2f" if metric != "trade_count" else ".0f", cmap="RdYlGn", center=0 if metric in {"sharpe", "expectancy", "max_drawdown"} else None, ax=axes[row_idx, col_idx])
                axes[row_idx, col_idx].set_title(f"{labels[metric]} — {period.upper()}")
                axes[row_idx, col_idx].set_xlabel(definition["parameter_2"])
                axes[row_idx, col_idx].set_ylabel(definition["parameter_1"])
        fig.suptitle(f"{asset} — {mechanism} — grille locale verrouillée", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(HEATMAP_DIR / f"{asset}__{mechanism}__is_validation.png", dpi=150)
        plt.close(fig)


def write_report(summary: pd.DataFrame, wide: pd.DataFrame, config: dict[str, Any], config_hash: str) -> None:
    lines = [
        "# Réplication MES / M2K des mécanismes MNQ", "", "## Protocole", "",
        "- IS : **2019-05-06 → 2023-12-31** ; Validation : **2024-01-01 → 2026-03-19**.",
        "- Grilles locales verrouillées avant exécution ; calibration et choix de seuils réalisés sur IS uniquement.",
        "- Un plateau commun exige au moins trois cellules robustes contiguës et une extension sur les deux axes.",
        "- Cellule robuste : Sharpe positif, PF > 1 et au moins 20 trades dans chaque sous-période.",
        "- Même logique de signal, sorties verrouillées, sessions et hard flatten ; spécifications de tick et valeur du point adaptées au contrat.",
        f"- SHA-256 de la grille : `{config_hash}`.", "", "## Classification", "",
        "| Actif | Mécanisme | Classe | Cellules robustes | Plus grande zone | Échantillon suffisant |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary.sort_values(["mechanism", "asset"]).itertuples(index=False):
        lines.append(f"| {row.asset} | `{row.mechanism}` | **`{row.classification}`** | {row.robust_cell_count} | {row.largest_robust_component} | {row.sufficient_sample_cell_share:.0%} |")
    lines += ["", "## Point central correspondant aux paramètres MNQ retenus", "",
              "| Actif | Mécanisme | Sharpe IS/VAL | Expectancy IS/VAL | PF IS/VAL | Max DD IS/VAL | Trades IS/VAL |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for mechanism, definition in config["mechanisms"].items():
        center_1, center_2 = float(definition["values_1"][1]), float(definition["values_2"][1])
        for asset in config["assets"]:
            row = wide[(wide["asset"] == asset) & (wide["mechanism"] == mechanism) & np.isclose(wide["parameter_1"], center_1) & np.isclose(wide["parameter_2"], center_2)].iloc[0]
            lines.append(
                f"| {asset} | `{mechanism}` | {row.sharpe_is:.2f}/{row.sharpe_validation:.2f} | "
                f"{row.expectancy_is:.2f}/{row.expectancy_validation:.2f} | {row.profit_factor_is:.2f}/{row.profit_factor_validation:.2f} | "
                f"{row.max_drawdown_is:,.0f}/{row.max_drawdown_validation:,.0f} | {int(row.trade_count_is)}/{int(row.trade_count_validation)} |"
            )
    lines += ["", "## Lecture par mécanisme", ""]
    for mechanism, group in summary.groupby("mechanism"):
        classes = dict(zip(group.asset, group.classification))
        if all(value == "portable" for value in classes.values()):
            conclusion = "portable sur MES et M2K : indication equity-index generic."
        elif any(value == "portable" for value in classes.values()):
            conclusion = "portabilité partielle selon l'actif."
        elif all(value == "asset_specific" for value in classes.values()):
            conclusion = "aucun plateau commun hors Nasdaq : indication Nasdaq-specific."
        else:
            conclusion = "preuve insuffisante pour conclure."
        lines.append(f"- `{mechanism}` : {conclusion}")
    lines += ["", "## Livrables", "", "- `cross_asset_results.csv`", "- `cross_asset_summary.csv`", "- dossier `heatmaps/`", "- `run_cross_asset_replication.py`", ""]
    (OUTPUT_DIR / "CROSS_ASSET_REPLICATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def execute() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not config.get("locked_before_execution"):
        raise RuntimeError("Cross-asset grids are not locked")
    config_hash = sha256(CONFIG_PATH)
    CONFIG_HASH_PATH.write_text(f"{config_hash}  {CONFIG_PATH.name}\n", encoding="ascii")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_asset, asset, config) for asset in config["assets"]]
        rows = [row for future in futures for row in future.result()]
    raw = pd.DataFrame(rows)
    wide, summary = analyze(raw, config)
    wide.to_csv(OUTPUT_DIR / "cross_asset_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "cross_asset_summary.csv", index=False)
    plot_heatmaps(wide, config)
    write_report(summary, wide, config, config_hash)
    metadata = {
        "grid_sha256": config_hash,
        "data_sha256": {asset: sha256(REPO_ROOT / definition["data"]) for asset, definition in config["assets"].items()},
        "cell_count": int(len(wide)),
        "period_result_count": int(len(raw)),
        "assets": list(config["assets"]),
        "mechanisms": list(config["mechanisms"]),
        "validation_used_for_tuning": False,
        "portfolio_constructed": False,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    execute()
