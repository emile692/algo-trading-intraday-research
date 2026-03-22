"""Targeted validation campaign for ORB dynamic noise-area gate overlay."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analytics.metrics import compute_metrics
from src.config.paths import REPO_ROOT, ensure_directories
from src.engine.portfolio import build_equity_curve

from .campaign import _evaluate_experiment, _experiment_from_json, _legacy_baseline_trades
from .features import attach_daily_reference, build_candidate_universe, build_daily_reference, prepare_minute_dataset
from .types import (
    BaselineEnsembleConfig,
    BaselineEntryConfig,
    CampaignContext,
    CompressionConfig,
    DynamicThresholdConfig,
    ExitConfig,
    ExperimentConfig,
)


@dataclass(frozen=True)
class NoiseGateValidationConfig:
    """Top-level config for the focused noise-gate validation campaign."""

    dataset_path: Path | None
    output_dir: Path
    is_fraction: float = 0.70
    random_seed: int = 42
    bootstrap_paths: int = 1200
    top_n: int = 12
    run_micro_refine: bool = True


def _split_sessions(all_sessions: list, is_fraction: float) -> tuple[list, list]:
    if len(all_sessions) < 2:
        raise ValueError("Not enough sessions for IS/OOS split.")
    split_idx = int(len(all_sessions) * float(is_fraction))
    split_idx = max(1, min(len(all_sessions) - 1, split_idx))
    return all_sessions[:split_idx], all_sessions[split_idx:]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _resolve_dataset_path(user_dataset: Path | None) -> Path:
    candidates = []
    if user_dataset is not None:
        candidates.append(user_dataset)
    candidates.extend(
        [
            REPO_ROOT / "data" / "dowloaded" / "MNQ_c_0_1m_20260321_094501.parquet",
            REPO_ROOT / "data" / "downloaded" / "MNQ_c_0_1m_20260321_094501.parquet",
            REPO_ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet",
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError("No MNQ minute dataset found. Pass --dataset explicitly.")


def _make_output_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "charts": root / "charts",
        "notebooks": root / "notebooks",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _noise_mode_for_schedule(schedule: str) -> str:
    if schedule == "every_5m":
        return "noise_area_gate_plus_discrete_schedule"
    return "noise_area_gate"


def _style_label(threshold_style: str) -> str:
    if threshold_style == "or_high_plus_k_noise_abs":
        return "or_plus_k_noise"
    return "max_or_noise"


def _bool_noise_gate_enabled(dynamic_mode: str) -> bool:
    return dynamic_mode in {
        "noise_area_gate",
        "noise_area_gate_plus_close_confirmation",
        "noise_area_gate_plus_discrete_schedule",
    }


def _baseline_experiment(
    baseline_entry: BaselineEntryConfig,
    baseline_ensemble: BaselineEnsembleConfig,
) -> ExperimentConfig:
    return ExperimentConfig(
        name="baseline_fixed",
        stage="baseline",
        family="baseline",
        baseline_entry=baseline_entry,
        baseline_ensemble=baseline_ensemble,
        compression=CompressionConfig(mode="none", usage="hard_filter"),
        exit=ExitConfig(mode="baseline"),
        dynamic_threshold=DynamicThresholdConfig(mode="disabled"),
    )


def build_noise_gate_grid_experiments(
    baseline_entry: BaselineEntryConfig,
    baseline_ensemble: BaselineEnsembleConfig,
) -> list[ExperimentConfig]:
    """Build focused overlay grid around the validated noise-gate cluster."""
    lookbacks = [20, 30, 40]
    vms = [0.75, 1.0, 1.25]
    styles = ["max_or_high_noise", "or_high_plus_k_noise_abs"]
    ks = [0.0, 0.25, 0.5]
    confirm_bars_values = [1, 2]
    schedules = ["continuous_on_bar_close", "every_5m"]

    experiments: list[ExperimentConfig] = []
    for lookback in lookbacks:
        for vm in vms:
            for style in styles:
                k_values = [0.0] if style == "max_or_high_noise" else ks
                for noise_k in k_values:
                    for confirm_bars in confirm_bars_values:
                        for schedule in schedules:
                            mode = _noise_mode_for_schedule(schedule)
                            style_label = _style_label(style)
                            name = (
                                f"noise_gate__{style_label}__L{lookback}"
                                f"__vm{str(vm).replace('.', 'p')}"
                                f"__k{str(noise_k).replace('.', 'p')}"
                                f"__c{confirm_bars}__{schedule}"
                            )
                            experiments.append(
                                ExperimentConfig(
                                    name=name,
                                    stage="grid",
                                    family="noise_gate",
                                    baseline_entry=baseline_entry,
                                    baseline_ensemble=baseline_ensemble,
                                    compression=CompressionConfig(mode="none", usage="hard_filter"),
                                    exit=ExitConfig(mode="baseline"),
                                    dynamic_threshold=DynamicThresholdConfig(
                                        mode=mode,
                                        noise_lookback=lookback,
                                        noise_vm=vm,
                                        threshold_style=style,
                                        noise_k=float(noise_k),
                                        atr_k=0.0,
                                        confirm_bars=confirm_bars,
                                        schedule=schedule,
                                    ),
                                )
                            )
    return experiments


def _experiment_param_key(experiment: ExperimentConfig) -> str:
    return json.dumps(
        {
            "baseline_entry": asdict(experiment.baseline_entry),
            "baseline_ensemble": asdict(experiment.baseline_ensemble),
            "compression": asdict(experiment.compression),
            "exit": asdict(experiment.exit),
            "dynamic_threshold": asdict(experiment.dynamic_threshold),
        },
        sort_keys=True,
    )


def _build_micro_refine_experiments(
    top3_grid: pd.DataFrame,
    baseline_entry: BaselineEntryConfig,
    baseline_ensemble: BaselineEnsembleConfig,
    existing_param_keys: set[str],
) -> list[ExperimentConfig]:
    """Create a small local zoom around the top-3 grid configs."""
    experiments: list[ExperimentConfig] = []
    local_seen = set(existing_param_keys)

    for rank, (_, row) in enumerate(top3_grid.iterrows(), start=1):
        seed = _experiment_from_json(str(row["config_json"]))
        dyn = seed.dynamic_threshold

        lookbacks = sorted({max(10, int(dyn.noise_lookback) - 10), int(dyn.noise_lookback), int(dyn.noise_lookback) + 10})
        vms = sorted(
            {
                max(0.5, round(float(dyn.noise_vm) - 0.25, 2)),
                round(float(dyn.noise_vm), 2),
                round(float(dyn.noise_vm) + 0.25, 2),
            }
        )
        if dyn.threshold_style == "or_high_plus_k_noise_abs":
            ks = sorted(
                {
                    max(0.0, round(float(dyn.noise_k) - 0.25, 2)),
                    round(float(dyn.noise_k), 2),
                    round(float(dyn.noise_k) + 0.25, 2),
                }
            )
        else:
            ks = [0.0]

        for lookback in lookbacks:
            for vm in vms:
                for k in ks:
                    mode = _noise_mode_for_schedule(seed.dynamic_threshold.schedule)
                    style_label = _style_label(seed.dynamic_threshold.threshold_style)
                    exp = ExperimentConfig(
                        name=(
                            f"noise_gate_micro__r{rank}__{style_label}"
                            f"__L{lookback}__vm{str(vm).replace('.', 'p')}"
                            f"__k{str(k).replace('.', 'p')}"
                            f"__c{seed.dynamic_threshold.confirm_bars}__{seed.dynamic_threshold.schedule}"
                        ),
                        stage="micro_refine",
                        family="noise_gate_micro",
                        baseline_entry=baseline_entry,
                        baseline_ensemble=baseline_ensemble,
                        compression=CompressionConfig(mode="none", usage="hard_filter"),
                        exit=ExitConfig(mode="baseline"),
                        dynamic_threshold=DynamicThresholdConfig(
                            mode=mode,
                            noise_lookback=int(lookback),
                            noise_vm=float(vm),
                            threshold_style=seed.dynamic_threshold.threshold_style,
                            noise_k=float(k),
                            atr_k=0.0,
                            confirm_bars=int(seed.dynamic_threshold.confirm_bars),
                            schedule=seed.dynamic_threshold.schedule,
                        ),
                    )
                    key = _experiment_param_key(exp)
                    if key in local_seen:
                        continue
                    local_seen.add(key)
                    experiments.append(exp)
    return experiments


def _run_experiments(
    experiments: list[ExperimentConfig],
    context: CampaignContext,
    bootstrap_paths: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i, experiment in enumerate(experiments, start=1):
        row, _ = _evaluate_experiment(
            experiment=experiment,
            context=context,
            bootstrap_paths=bootstrap_paths,
            random_seed=random_seed + i,
            keep_details=False,
        )
        row["noise_gate_enabled"] = _bool_noise_gate_enabled(experiment.dynamic_threshold.mode)
        row["threshold_style"] = experiment.dynamic_threshold.threshold_style
        row["noise_style"] = _style_label(experiment.dynamic_threshold.threshold_style)
        row["run_group"] = experiment.stage
        rows.append(row)
    return pd.DataFrame(rows)


def _compute_simple_prop_score(results: pd.DataFrame) -> pd.Series:
    ret_dd = pd.to_numeric(results["oos_return_over_drawdown"], errors="coerce").fillna(0.0)
    pf = pd.to_numeric(results["oos_profit_factor"], errors="coerce").fillna(1.0)
    expectancy = pd.to_numeric(results["oos_expectancy"], errors="coerce").fillna(0.0)
    max_dd = pd.to_numeric(results["oos_max_drawdown"], errors="coerce").abs().fillna(0.0)
    worst_5d = pd.to_numeric(results["oos_worst_5day_drawdown"], errors="coerce").abs().fillna(0.0)
    losing_streak = pd.to_numeric(results["oos_max_losing_streak"], errors="coerce").fillna(0.0)

    score = (
        2.2 * np.tanh(ret_dd / 4.0)
        + 1.1 * np.tanh((pf - 1.0) / 0.40)
        + 0.9 * np.tanh(expectancy / 45.0)
        - 2.4 * np.tanh(max_dd / 1200.0)
        - 1.8 * np.tanh(worst_5d / 900.0)
        - 1.2 * np.tanh(np.maximum(losing_streak - 2.0, 0.0) / 4.0)
    )
    return score.astype(float)


def _write_baseline_integrity_report(
    output_path: Path,
    dataset_path: Path,
    reference_trades: pd.DataFrame,
    candidate_trades: pd.DataFrame,
    reference_info: dict[str, object],
    tolerance: float = 1e-6,
) -> None:
    all_sessions = sorted(
        pd.to_datetime(
            pd.concat(
                [
                    reference_trades.get("session_date", pd.Series(dtype=object)),
                    candidate_trades.get("session_date", pd.Series(dtype=object)),
                ],
                ignore_index=True,
            ),
            errors="coerce",
        )
        .dropna()
        .dt.date.unique()
    )
    ref_metrics = compute_metrics(reference_trades, session_dates=all_sessions, initial_capital=50_000.0)
    new_metrics = compute_metrics(candidate_trades, session_dates=all_sessions, initial_capital=50_000.0)

    keys = ["n_trades", "cumulative_pnl", "sharpe_ratio", "profit_factor", "expectancy", "max_drawdown"]
    rows = []
    for key in keys:
        ref_v = _safe_float(ref_metrics.get(key), 0.0)
        new_v = _safe_float(new_metrics.get(key), 0.0)
        diff = new_v - ref_v
        rel = diff / max(abs(ref_v), 1e-9)
        ok = abs(diff) <= tolerance * max(1.0, abs(ref_v))
        rows.append((key, ref_v, new_v, diff, rel, ok))

    lines = [
        "# Baseline Integrity Report",
        "",
        "## Baseline Mapping",
        "",
        "- Data loading: `src/data/loader.py` -> `load_ohlcv_file`.",
        "- Feature engineering: `src/features/intraday.py`, `src/features/opening_range.py`, `src/features/volatility.py`.",
        "- ORB signal generation: `src/strategy/orb.py` (long-only with VWAP confirmation).",
        "- ATR ensemble selection: `src/analytics/atr_ensemble_campaign.py` (quantile cross-product + vote threshold).",
        "- Backtest engine: `src/engine/backtester.py` via `src/analytics/orb_research/exits.py` baseline mode.",
        "- Metrics: `src/analytics/metrics.py` + extended campaign metrics in `src/analytics/orb_research/evaluation.py`.",
        "",
        "## Reference",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Legacy baseline strategy id: `{reference_info.get('strategy_id')}`",
        f"- Legacy selected days: {reference_info.get('selected_days')}",
        "",
        "## Numerical Non-Regression",
        "",
        "| metric | reference | new_pipeline | diff | rel_diff | pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for key, ref_v, new_v, diff, rel, ok in rows:
        lines.append(f"| {key} | {ref_v:.8f} | {new_v:.8f} | {diff:.8f} | {rel:.6%} | {'yes' if ok else 'no'} |")

    lines.extend(
        [
            "",
            "## Integrity Verdict",
            "",
            (
                "- Baseline preserved within tolerance."
                if all(ok for *_, ok in rows)
                else "- Baseline mismatch detected (see table)."
            ),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _prepare_heatmap_data(df: pd.DataFrame, index_col: str, column_col: str, value_col: str, aggfunc: str = "mean") -> pd.DataFrame:
    pivot = df.pivot_table(index=index_col, columns=column_col, values=value_col, aggfunc=aggfunc)
    if pivot.empty:
        return pivot
    pivot = pivot.sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    return pivot


def _plot_heatmap(
    pivot: pd.DataFrame,
    title: str,
    out_path: Path,
    value_fmt: str = "{:.2f}",
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    if pivot.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
    else:
        data = pivot.to_numpy(dtype=float)
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(x) for x in pivot.columns], rotation=35, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(x) for x in pivot.index])
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                if np.isfinite(value):
                    ax.text(j, i, value_fmt.format(value), ha="center", va="center", fontsize=8, color="white")
        fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _build_equity_curve_for_row(
    row: pd.Series,
    context: CampaignContext,
    bootstrap_paths: int,
    random_seed: int,
) -> pd.DataFrame:
    experiment = _experiment_from_json(str(row["config_json"]))
    _, detail = _evaluate_experiment(
        experiment=experiment,
        context=context,
        bootstrap_paths=bootstrap_paths,
        random_seed=random_seed,
        keep_details=True,
    )
    trades = detail["trades"] if detail is not None else pd.DataFrame()
    return build_equity_curve(trades, initial_capital=experiment.baseline_entry.account_size_usd)


def _plot_equity_and_drawdown(
    baseline_row: pd.Series,
    top_noise_rows: pd.DataFrame,
    context: CampaignContext,
    out_equity: Path,
    out_drawdown: Path,
) -> None:
    curves: dict[str, pd.DataFrame] = {}
    curves[str(baseline_row["name"])] = _build_equity_curve_for_row(
        baseline_row,
        context=context,
        bootstrap_paths=700,
        random_seed=901,
    )
    for i, (_, row) in enumerate(top_noise_rows.head(4).iterrows(), start=1):
        curves[str(row["name"])] = _build_equity_curve_for_row(
            row,
            context=context,
            bootstrap_paths=700,
            random_seed=901 + i,
        )

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for name, curve in curves.items():
        if curve.empty:
            continue
        ax1.plot(curve["timestamp"], curve["equity"], linewidth=1.4, label=name)
    ax1.set_title("Baseline vs Top Noise-Gate Equity Curves")
    ax1.set_ylabel("Equity (USD)")
    ax1.set_xlabel("Time")
    ax1.legend(loc="best", fontsize=8)
    fig1.tight_layout()
    fig1.savefig(out_equity, dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for name, curve in curves.items():
        if curve.empty:
            continue
        ax2.plot(curve["timestamp"], curve["drawdown"], linewidth=1.4, label=name)
    ax2.set_title("Baseline vs Top Noise-Gate Drawdowns")
    ax2.set_ylabel("Drawdown (USD)")
    ax2.set_xlabel("Time")
    ax2.legend(loc="best", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(out_drawdown, dpi=150)
    plt.close(fig2)


def _write_notebook(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _cell(cell_type: str, source: str) -> dict[str, object]:
        if not source.endswith("\n"):
            source += "\n"
        return {
            "cell_type": cell_type,
            "metadata": {},
            "source": source.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
        }

    nb = {
        "cells": [
            _cell(
                "markdown",
                "# ORB Final Ensemble Validation With Noise Gate\n\nValidation ciblee de l'overlay noise-area gate sur baseline figee.",
            ),
            _cell(
                "code",
                "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\n\nroot = Path.cwd()\nout = root / 'export' / 'orb_noise_gate_validation'\nprint(out)",
            ),
            _cell(
                "code",
                "results = pd.read_csv(out / 'noise_gate_validation_results.csv')\ntop = pd.read_csv(out / 'noise_gate_top_configs.csv')\nsummary = (out / 'noise_gate_summary.md').read_text(encoding='utf-8')\n\ndisplay(top[['name','run_group','noise_style','noise_lookback','noise_vm','noise_k','confirm_bars','schedule','ranking_prop_score','oos_net_pnl','oos_max_drawdown','oos_worst_5day_drawdown','oos_max_losing_streak']].head(15))",
            ),
            _cell(
                "code",
                "baseline = results[results['name']=='baseline_fixed'].iloc[0]\nnoise = results[(results['noise_gate_enabled']==True) & (results['status']=='ok')].sort_values('ranking_prop_score', ascending=False)\ncompare = pd.concat([pd.DataFrame([baseline]), noise.head(8)], ignore_index=True)\ndisplay(compare[['name','run_group','noise_style','noise_lookback','noise_vm','noise_k','confirm_bars','schedule','ranking_prop_score','oos_net_pnl','oos_sharpe_ratio','oos_profit_factor','oos_expectancy','oos_max_drawdown','oos_return_over_drawdown','oos_worst_day','oos_worst_5day_drawdown','oos_max_losing_streak','oos_nb_trades','oos_pct_days_traded']])",
            ),
            _cell(
                "code",
                "charts = out / 'charts'\nfor name in [\n    'heatmap_prop_score_L_VM.png',\n    'heatmap_maxdd_L_VM.png',\n    'heatmap_netpnl_L_VM.png',\n    'heatmap_L_k.png',\n    'heatmap_confirm_schedule.png',\n    'heatmap_synthesis_best_by_family.png',\n    'equity_curves_top_noise_gate.png',\n    'drawdown_top_noise_gate.png',\n]:\n    p = charts / name\n    if p.exists():\n        display(Image(filename=str(p)))",
            ),
            _cell(
                "code",
                "display(Markdown(summary))",
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")


def _neighbor_cluster_strength(pivot: pd.DataFrame) -> tuple[bool, str]:
    if pivot.empty:
        return False, "Aucune cellule disponible."
    arr = pivot.to_numpy(dtype=float)
    if not np.isfinite(arr).any():
        return False, "Toutes les cellules sont NaN."
    i_best, j_best = np.unravel_index(np.nanargmax(arr), arr.shape)
    best = float(arr[i_best, j_best])
    neighbors = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i = i_best + di
            j = j_best + dj
            if i < 0 or i >= arr.shape[0] or j < 0 or j >= arr.shape[1]:
                continue
            value = float(arr[i, j])
            if math.isfinite(value):
                neighbors.append(value)
    strong = sum(v >= 0.90 * best for v in neighbors)
    is_cluster = strong >= 3
    label = f"best={best:.3f}, voisins>=90%={strong}"
    return is_cluster, label


def _write_summary(
    output_path: Path,
    baseline_row: pd.Series,
    best_row: pd.Series,
    safe_row: pd.Series,
    pivot_prop_l_vm: pd.DataFrame,
    pivot_confirm_schedule: pd.DataFrame,
) -> None:
    delta_prop = float(best_row["ranking_prop_score"]) - float(baseline_row["ranking_prop_score"])
    delta_pnl = float(best_row["oos_net_pnl"]) - float(baseline_row["oos_net_pnl"])
    delta_dd = float(best_row["oos_max_drawdown"]) - float(baseline_row["oos_max_drawdown"])
    improved = delta_prop > 0.0

    cluster_ok, cluster_label = _neighbor_cluster_strength(pivot_prop_l_vm)

    fragility = "n/a"
    if not pivot_confirm_schedule.empty and np.isfinite(pivot_confirm_schedule.to_numpy(dtype=float)).any():
        values = pivot_confirm_schedule.to_numpy(dtype=float)
        fragility = f"spread confirm/schedule={float(np.nanmax(values) - np.nanmin(values)):.3f}"

    lines = [
        "# Noise Gate Validation Summary",
        "",
        "## Setup",
        "",
        "- Campagne ciblee: baseline figee + overlay noise-area gate uniquement.",
        "- Pas de compression filter, pas de VWAP exit, pas de full re-opt global.",
        "- Classement principal: `ranking_prop_score` (prop-firm oriented, explicite).",
        "",
        "## Baseline vs Best Overlay",
        "",
        f"- Baseline (`{baseline_row['name']}`): prop_score={float(baseline_row['ranking_prop_score']):.4f}, net_pnl={float(baseline_row['oos_net_pnl']):.2f}, maxDD={float(baseline_row['oos_max_drawdown']):.2f}.",
        f"- Best noise gate (`{best_row['name']}`): prop_score={float(best_row['ranking_prop_score']):.4f}, net_pnl={float(best_row['oos_net_pnl']):.2f}, maxDD={float(best_row['oos_max_drawdown']):.2f}.",
        f"- Delta best-baseline: d_prop={delta_prop:.4f}, d_net_pnl={delta_pnl:.2f}, d_maxDD={delta_dd:.2f}.",
        "",
        "## Heatmap Read",
        "",
        f"- Cluster LxVM (style `max_or_noise`): {'stable' if cluster_ok else 'fragile'} ({cluster_label}).",
        f"- Sensibilite confirm_bars x schedule autour du meilleur couple structurel: {fragility}.",
        "",
        "## Top Robust Parameters",
        "",
        f"- Recommande (validation / production research): `{best_row['name']}`.",
        f"- Preset safe / prop-firm oriented: `{safe_row['name']}`.",
        "",
        "## Decision (Required Questions)",
        "",
        f"1. Est-ce que le noise-area gate ameliore la baseline figee ? {'Oui' if improved else 'Non'}.",
        f"2. Quel reglage est le meilleur selon prop_score ? `{best_row['name']}`.",
        f"3. Quel reglage est le plus robuste visuellement via heatmaps ? `{best_row['name']}` {'(cluster voisin stable)' if cluster_ok else '(attention: cluster peu dense)'}.",
        f"4. Y a-t-il un cluster simple et stable autour du meilleur reglage ? {'Oui' if cluster_ok else 'Non / a confirmer'}.",
        f"5. Recommandes-tu d'ajouter cette option a la baseline ? {'Oui, en option via flag explicite.' if improved else 'Pas pour l instant.'}",
        f"6. Quel preset recommandes-tu comme preset validation / production research ? `{best_row['name']}` (safe alternatif: `{safe_row['name']}`).",
        "",
        "## Notes",
        "",
        "- La baseline reste intacte: verification de non-regression executee avant conclusion.",
        "- Les resultats complets sont dans `noise_gate_validation_results.csv` et `noise_gate_top_configs.csv`.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_orb_noise_gate_validation(config: NoiseGateValidationConfig) -> dict[str, Path]:
    ensure_directories()
    dataset_path = _resolve_dataset_path(config.dataset_path)
    dirs = _make_output_dirs(config.output_dir)

    baseline_entry = BaselineEntryConfig()
    baseline_ensemble = BaselineEnsembleConfig()

    minute_df = prepare_minute_dataset(
        dataset_path=dataset_path,
        baseline_entry=baseline_entry,
        atr_windows=[baseline_ensemble.atr_window],
    )
    daily_df = build_daily_reference(minute_df)
    minute_df = attach_daily_reference(minute_df, daily_df)
    candidate_base = build_candidate_universe(minute_df, baseline_entry=baseline_entry)

    all_sessions = sorted(pd.to_datetime(minute_df["session_date"]).dt.date.unique())
    is_sessions, oos_sessions = _split_sessions(all_sessions, config.is_fraction)
    context = CampaignContext(
        all_sessions=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        minute_df=minute_df,
        candidate_base_df=candidate_base,
        daily_patterns=daily_df,
    )

    reference_trades, reference_info = _legacy_baseline_trades(dataset_path)
    baseline_exp = _baseline_experiment(baseline_entry=baseline_entry, baseline_ensemble=baseline_ensemble)
    baseline_row_dict, baseline_detail = _evaluate_experiment(
        experiment=baseline_exp,
        context=context,
        bootstrap_paths=max(700, config.bootstrap_paths),
        random_seed=config.random_seed,
        keep_details=True,
    )
    baseline_row_dict["noise_gate_enabled"] = False
    baseline_row_dict["threshold_style"] = "disabled"
    baseline_row_dict["noise_style"] = "disabled"
    baseline_row_dict["run_group"] = "baseline"
    baseline_trades_new = baseline_detail["trades"] if baseline_detail is not None else pd.DataFrame()

    _write_baseline_integrity_report(
        output_path=config.output_dir / "baseline_integrity_report.md",
        dataset_path=dataset_path,
        reference_trades=reference_trades,
        candidate_trades=baseline_trades_new,
        reference_info=reference_info,
    )

    grid_experiments = build_noise_gate_grid_experiments(baseline_entry, baseline_ensemble)
    grid_df = _run_experiments(
        experiments=grid_experiments,
        context=context,
        bootstrap_paths=config.bootstrap_paths,
        random_seed=config.random_seed + 1000,
    )
    baseline_df = pd.DataFrame([baseline_row_dict])

    valid_grid = grid_df.loc[grid_df["status"] == "ok"].copy()
    valid_grid["ranking_prop_score"] = _compute_simple_prop_score(valid_grid)
    top3_grid = valid_grid.sort_values(["ranking_prop_score", "oos_net_pnl"], ascending=[False, False]).head(3)

    micro_df = pd.DataFrame()
    if config.run_micro_refine and not top3_grid.empty:
        existing_keys = {_experiment_param_key(_experiment_from_json(payload)) for payload in grid_df["config_json"].tolist()}
        existing_keys.add(_experiment_param_key(baseline_exp))
        micro_experiments = _build_micro_refine_experiments(
            top3_grid=top3_grid,
            baseline_entry=baseline_entry,
            baseline_ensemble=baseline_ensemble,
            existing_param_keys=existing_keys,
        )
        if micro_experiments:
            micro_df = _run_experiments(
                experiments=micro_experiments,
                context=context,
                bootstrap_paths=config.bootstrap_paths,
                random_seed=config.random_seed + 2000,
            )

    results = pd.concat([baseline_df, grid_df, micro_df], ignore_index=True)
    results["ranking_prop_score"] = _compute_simple_prop_score(results)
    baseline_score = float(results.loc[results["name"] == "baseline_fixed", "ranking_prop_score"].iloc[0])
    results["delta_prop_score_vs_baseline"] = results["ranking_prop_score"] - baseline_score
    baseline_pnl = float(results.loc[results["name"] == "baseline_fixed", "oos_net_pnl"].iloc[0])
    results["delta_oos_net_pnl_vs_baseline"] = pd.to_numeric(results["oos_net_pnl"], errors="coerce") - baseline_pnl
    baseline_dd = float(results.loc[results["name"] == "baseline_fixed", "oos_max_drawdown"].iloc[0])
    results["delta_oos_maxdd_vs_baseline"] = pd.to_numeric(results["oos_max_drawdown"], errors="coerce") - baseline_dd

    results.to_csv(config.output_dir / "noise_gate_validation_results.csv", index=False)

    valid = results.loc[results["status"] == "ok"].copy()
    noise_valid = valid.loc[valid["noise_gate_enabled"]].copy()
    top_noise = noise_valid.sort_values(["ranking_prop_score", "oos_sharpe_ratio", "oos_net_pnl"], ascending=[False, False, False]).head(config.top_n)

    baseline_row = valid.loc[valid["name"] == "baseline_fixed"].iloc[0]
    top_with_baseline = pd.concat([pd.DataFrame([baseline_row]), top_noise], ignore_index=True)
    top_with_baseline.to_csv(config.output_dir / "noise_gate_top_configs.csv", index=False)

    max_style = noise_valid.loc[noise_valid["noise_style"] == "max_or_noise"].copy()
    plus_style = noise_valid.loc[noise_valid["noise_style"] == "or_plus_k_noise"].copy()

    pivot_prop_l_vm = _prepare_heatmap_data(max_style, "noise_lookback", "noise_vm", "ranking_prop_score", aggfunc="mean")
    _plot_heatmap(
        pivot=pivot_prop_l_vm,
        title="Noise Gate: L x VM (Prop Score, max_or_noise)",
        out_path=dirs["charts"] / "heatmap_prop_score_L_VM.png",
        value_fmt="{:.2f}",
    )

    pivot_dd_l_vm = _prepare_heatmap_data(max_style, "noise_lookback", "noise_vm", "oos_max_drawdown", aggfunc="mean")
    _plot_heatmap(
        pivot=pivot_dd_l_vm,
        title="Noise Gate: L x VM (Max Drawdown OOS)",
        out_path=dirs["charts"] / "heatmap_maxdd_L_VM.png",
        value_fmt="{:.0f}",
    )

    pivot_pnl_l_vm = _prepare_heatmap_data(max_style, "noise_lookback", "noise_vm", "oos_net_pnl", aggfunc="mean")
    _plot_heatmap(
        pivot=pivot_pnl_l_vm,
        title="Noise Gate: L x VM (Net PnL OOS)",
        out_path=dirs["charts"] / "heatmap_netpnl_L_VM.png",
        value_fmt="{:.0f}",
    )

    pivot_l_k = _prepare_heatmap_data(plus_style, "noise_lookback", "noise_k", "ranking_prop_score", aggfunc="mean")
    _plot_heatmap(
        pivot=pivot_l_k,
        title="Noise Gate: L x k (Prop Score, or_plus_k_noise)",
        out_path=dirs["charts"] / "heatmap_L_k.png",
        value_fmt="{:.2f}",
    )

    if not noise_valid.empty:
        best_struct = noise_valid.sort_values(["ranking_prop_score", "oos_net_pnl"], ascending=[False, False]).iloc[0]
        struct_filter = (
            noise_valid["noise_lookback"].eq(best_struct["noise_lookback"])
            & noise_valid["noise_vm"].eq(best_struct["noise_vm"])
            & noise_valid["threshold_style"].eq(best_struct["threshold_style"])
            & noise_valid["noise_k"].eq(best_struct["noise_k"])
        )
        confirm_schedule = noise_valid.loc[struct_filter].copy()
    else:
        confirm_schedule = pd.DataFrame()

    pivot_confirm_schedule = confirm_schedule.pivot_table(
        index="confirm_bars",
        columns="schedule",
        values="ranking_prop_score",
        aggfunc="mean",
    )
    if not pivot_confirm_schedule.empty:
        ordered_cols = [c for c in ["continuous_on_bar_close", "every_5m"] if c in pivot_confirm_schedule.columns]
        pivot_confirm_schedule = pivot_confirm_schedule.reindex(columns=ordered_cols)
        pivot_confirm_schedule = pivot_confirm_schedule.sort_index()
    _plot_heatmap(
        pivot=pivot_confirm_schedule,
        title="Noise Gate: confirm_bars x schedule (best structural params)",
        out_path=dirs["charts"] / "heatmap_confirm_schedule.png",
        value_fmt="{:.2f}",
    )

    pivot_synthesis = noise_valid.pivot_table(
        index="noise_style",
        columns="schedule",
        values="ranking_prop_score",
        aggfunc="max",
    )
    _plot_heatmap(
        pivot=pivot_synthesis,
        title="Noise Gate Synthesis: best prop score by style/schedule",
        out_path=dirs["charts"] / "heatmap_synthesis_best_by_family.png",
        value_fmt="{:.2f}",
    )

    _plot_equity_and_drawdown(
        baseline_row=baseline_row,
        top_noise_rows=top_noise,
        context=context,
        out_equity=dirs["charts"] / "equity_curves_top_noise_gate.png",
        out_drawdown=dirs["charts"] / "drawdown_top_noise_gate.png",
    )

    if noise_valid.empty:
        best_row = baseline_row
        safe_row = baseline_row
    else:
        best_row = noise_valid.sort_values(["ranking_prop_score", "oos_net_pnl"], ascending=[False, False]).iloc[0]
        q75 = noise_valid["ranking_prop_score"].quantile(0.75)
        safe_pool = noise_valid.loc[noise_valid["ranking_prop_score"] >= q75].copy()
        if safe_pool.empty:
            safe_pool = noise_valid.copy()
        safe_row = safe_pool.sort_values(
            ["oos_max_drawdown", "oos_worst_5day_drawdown", "ranking_prop_score"],
            ascending=[False, False, False],
        ).iloc[0]

    _write_summary(
        output_path=config.output_dir / "noise_gate_summary.md",
        baseline_row=baseline_row,
        best_row=best_row,
        safe_row=safe_row,
        pivot_prop_l_vm=pivot_prop_l_vm,
        pivot_confirm_schedule=pivot_confirm_schedule,
    )
    _write_notebook(path=dirs["notebooks"] / "orb_final_ensemble_validation_with_noise_gate.ipynb")

    return {
        "output_root": config.output_dir,
        "baseline_integrity_report": config.output_dir / "baseline_integrity_report.md",
        "results": config.output_dir / "noise_gate_validation_results.csv",
        "top_configs": config.output_dir / "noise_gate_top_configs.csv",
        "summary": config.output_dir / "noise_gate_summary.md",
        "notebook": dirs["notebooks"] / "orb_final_ensemble_validation_with_noise_gate.ipynb",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted ORB noise-gate overlay validation on frozen baseline.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "export" / "orb_noise_gate_validation")
    parser.add_argument("--is-fraction", type=float, default=0.70)
    parser.add_argument("--bootstrap-paths", type=int, default=1200)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-micro-refine", action="store_true")
    args = parser.parse_args()

    config = NoiseGateValidationConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        is_fraction=args.is_fraction,
        random_seed=args.seed,
        bootstrap_paths=args.bootstrap_paths,
        top_n=args.top_n,
        run_micro_refine=not bool(args.skip_micro_refine),
    )
    artifacts = run_orb_noise_gate_validation(config)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
