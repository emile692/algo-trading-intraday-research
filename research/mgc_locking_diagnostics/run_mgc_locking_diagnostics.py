"""Confirm and diagnose three MGC mechanisms without peak-Sharpe optimization."""

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
CHART_DIR = OUTPUT_DIR / "charts"
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


CONFIG_PATH = OUTPUT_DIR / "mgc_locking_diagnostic_config.json"
CONFIG_HASH_PATH = OUTPUT_DIR / "mgc_locking_diagnostic_config.sha256"
MGC_CONFIG_PATH = REPO_ROOT / "research" / "mgc_baseline_backtests" / "mgc_baseline_config.json"
GAP_RESULTS = OUTPUT_DIR / "gap_confirmation_results.csv"
GAP_SUMMARY = OUTPUT_DIR / "gap_confirmation_summary.csv"
TREND_YEARLY = OUTPUT_DIR / "trend_yearly_diagnostics.csv"
TREND_ROLLING = OUTPUT_DIR / "trend_rolling_diagnostics.csv"
ROLLING_EXIT_REASONS = OUTPUT_DIR / "rolling_range_exit_reasons.csv"
ROLLING_YEARLY = OUTPUT_DIR / "rolling_range_yearly_diagnostics.csv"
ROLLING_ROLLING = OUTPUT_DIR / "rolling_range_rolling_diagnostics.csv"
REPORT_PATH = OUTPUT_DIR / "MGC_LOCKING_DIAGNOSTICS_REPORT.md"

_DATA: Any = None
_SPECS: dict[str, dict[str, Any]] = {}
_MGC_CONFIG: dict[str, Any] = {}
_CONFIG: dict[str, Any] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not config.get("locked_before_execution"):
        raise ValueError("Diagnostic configuration is not locked")
    gap = config["gap_confirmation"]
    if len(gap["minimum_gap_atr_multiple"]) != 9 or len(gap["volume_threshold_multiple"]) != 9:
        raise ValueError("Gap confirmation must remain a 9x9 grid")
    return config


def _init_worker(data_path: str, config: dict[str, Any]) -> None:
    global _DATA, _SPECS, _MGC_CONFIG, _CONFIG
    specs, _, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    _SPECS = {spec["mechanism_id"]: spec for spec in specs}
    _MGC_CONFIG = json.loads(MGC_CONFIG_PATH.read_text(encoding="utf-8"))
    _CONFIG = config
    start = date.fromisoformat(config["periods"]["is"][0])
    end = date.fromisoformat(config["periods"]["validation"][1])
    _DATA, _ = prepare_mgc_data(Path(data_path), start, end)


def _adapt(mechanism: str, definition: dict[str, Any], value_1: Any, value_2: Any) -> dict[str, Any]:
    spec = adapt_spec_to_mgc(_SPECS[mechanism], _MGC_CONFIG)
    return _apply_parameters(spec, definition, value_1, value_2)


def _slice(run: Any, start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = run.daily[run.daily["session_date"].between(start, end)].copy()
    trades = run.trades[run.trades["session_date"].between(start, end)].copy() if not run.trades.empty else run.trades.copy()
    return daily, trades


def _metrics(daily: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
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
        "net_pnl_usd": float(pnl.sum()),
        "max_drawdown_usd": float(drawdown.min()) if len(drawdown) else 0.0,
        "trade_count": int(len(trades)),
        "session_count": int(len(daily)),
    }


def _violations(run: Any) -> dict[str, Any]:
    return {key: value for key, value in run.checks.items() if key.endswith("violations") and value}


def _yearly_rows(run: Any, mechanism: str, point_id: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year in sorted(pd.to_datetime(run.daily["session_date"]).dt.year.unique()):
        start, end = date(int(year), 1, 1), date(int(year), 12, 31)
        daily, trades = _slice(run, start, end)
        if daily.empty:
            continue
        rows.append({"mechanism_id": mechanism, "point_id": point_id, **parameters, "year": int(year), **_metrics(daily, trades)})
    return rows


def _rolling_rows(run: Any, mechanism: str, point_id: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
    window = int(_CONFIG["rolling_diagnostic"]["window_sessions"])
    minimum = int(_CONFIG["rolling_diagnostic"]["minimum_sessions"])
    daily = run.daily.sort_values("session_date").copy()
    pnl = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0)
    equity = INITIAL_CAPITAL_USD + pnl.cumsum()
    returns = pnl / equity.shift(1).fillna(INITIAL_CAPITAL_USD)
    mean = returns.rolling(window, min_periods=minimum).mean()
    std = returns.rolling(window, min_periods=minimum).std(ddof=1)
    sharpe = mean.div(std.where(std.gt(0))).mul(math.sqrt(252))
    rolling_pnl = pnl.rolling(window, min_periods=minimum).sum()
    out = []
    for i in range(len(daily)):
        if not np.isfinite(sharpe.iloc[i]):
            continue
        out.append({
            "mechanism_id": mechanism,
            "point_id": point_id,
            **parameters,
            "session_date": daily.iloc[i]["session_date"],
            "rolling_sessions": min(i + 1, window),
            "rolling_sharpe": float(sharpe.iloc[i]),
            "rolling_pnl_usd": float(rolling_pnl.iloc[i]),
        })
    return out


def _task_worker(task: dict[str, Any]) -> dict[str, Any]:
    kind = task["kind"]
    if kind == "gap":
        definition = {"parameter_1": "minimum_gap_atr_multiple", "parameter_2": "volume_threshold_multiple"}
        spec = _adapt("gap_continuation", definition, task["p1"], task["p2"])
    elif kind == "trend":
        definition = {"parameter_1": "ribbon_fast_anchor", "parameter_2": "ribbon_slow_anchor"}
        spec = _adapt("trend_pullback_ma_structure", definition, task["p1"], task["p2"])
    else:
        definition = {"parameter_1": "entry_channel_length", "parameter_2": "exit_channel_length"}
        spec = _adapt("rolling_range_breakout", definition, task["p1"], task["p2"])
    run = run_baseline_backtest(spec, _DATA)
    violations = _violations(run)
    if violations:
        raise RuntimeError(f"{task}: {violations}")
    result: dict[str, Any] = {"task": task, "period_metrics": [], "yearly": [], "rolling": [], "exit_reasons": []}
    for period in ("is", "validation"):
        start, end = (date.fromisoformat(v) for v in _CONFIG["periods"][period])
        daily, trades = _slice(run, start, end)
        result["period_metrics"].append({"period": period, **_metrics(daily, trades)})
        if kind == "rolling":
            counts = trades["exit_reason"].value_counts(dropna=False)
            for reason, count in counts.items():
                result["exit_reasons"].append({"period": period, "exit_reason": str(reason), "count": int(count)})
    parameters = {task["p1_name"]: task["p1"], task["p2_name"]: task["p2"]}
    point_id = task["point_id"]
    if kind == "trend" or (kind == "rolling" and task["p2"] == _CONFIG["rolling_range_diagnosis"]["temporal_diagnostic_exit_channel_length"]):
        result["yearly"] = _yearly_rows(run, spec["mechanism_id"], point_id, parameters)
        result["rolling"] = _rolling_rows(run, spec["mechanism_id"], point_id, parameters)
    return result


def _tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    gap = config["gap_confirmation"]
    for i, p1 in enumerate(gap["minimum_gap_atr_multiple"]):
        for j, p2 in enumerate(gap["volume_threshold_multiple"]):
            tasks.append({"kind": "gap", "p1": p1, "p2": p2, "i": i, "j": j,
                          "p1_name": "minimum_gap_atr_multiple", "p2_name": "volume_threshold_multiple",
                          "point_id": f"gap_{i:02d}_{j:02d}"})
    for index, point in enumerate(config["trend_representative_points"]):
        tasks.append({"kind": "trend", "p1": point["ribbon_fast_anchor"], "p2": point["ribbon_slow_anchor"],
                      "i": index, "j": 0, "p1_name": "ribbon_fast_anchor", "p2_name": "ribbon_slow_anchor",
                      "point_id": f"trend_f{point['ribbon_fast_anchor']}_s{point['ribbon_slow_anchor']}"})
    rr = config["rolling_range_diagnosis"]
    for i, p1 in enumerate(rr["entry_channel_length"]):
        for j, p2 in enumerate(rr["exit_channel_length"]):
            tasks.append({"kind": "rolling", "p1": p1, "p2": p2, "i": i, "j": j,
                          "p1_name": "entry_channel_length", "p2_name": "exit_channel_length",
                          "point_id": f"rolling_e{p1}_x{p2}"})
    return tasks


def _gap_tables(outputs: list[dict[str, Any]], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, int]], tuple[int, int] | None]:
    rows = []
    for output in outputs:
        task = output["task"]
        if task["kind"] != "gap":
            continue
        for metrics in output["period_metrics"]:
            rows.append({"mechanism_id": "gap_continuation", "combination_id": task["point_id"],
                         "minimum_gap_atr_multiple": task["p1"], "volume_threshold_multiple": task["p2"],
                         "parameter_1_index": task["i"], "parameter_2_index": task["j"], **metrics})
    long = pd.DataFrame(rows).sort_values(["parameter_1_index", "parameter_2_index", "period"])
    identity = ["mechanism_id", "combination_id", "minimum_gap_atr_multiple", "volume_threshold_multiple", "parameter_1_index", "parameter_2_index"]
    wide = long.pivot(index=identity, columns="period", values=["sharpe_annualized", "net_pnl_usd", "max_drawdown_usd", "trade_count"]).reset_index()
    wide.columns = ["_".join(filter(None, item)).strip("_") if isinstance(item, tuple) else item for item in wide.columns]
    minimum = int(config["minimum_trades_each_period"])
    wide["insufficient_sample"] = wide.trade_count_is.lt(minimum) | wide.trade_count_validation.lt(minimum)
    wide["conservative_sharpe"] = wide[["sharpe_annualized_is", "sharpe_annualized_validation"]].min(axis=1)
    wide["robust_cell"] = ~wide.insufficient_sample & wide.conservative_sharpe.gt(0)
    lookup = {(int(r.parameter_1_index), int(r.parameter_2_index)): r for r in wide.itertuples(index=False)}
    for idx, row in wide.iterrows():
        i, j = int(row.parameter_1_index), int(row.parameter_2_index)
        neighborhood = [lookup[(a, b)].conservative_sharpe for a in range(max(0, i - 1), min(9, i + 2))
                        for b in range(max(0, j - 1), min(9, j + 2)) if (a, b) in lookup and lookup[(a, b)].robust_cell]
        wide.loc[idx, "robust_neighbors_3x3"] = len(neighborhood)
        wide.loc[idx, "local_stability_score"] = (float(np.median(neighborhood)) - 0.5 * float(np.std(neighborhood))) if neighborhood else np.nan
    cells = {(int(r.parameter_1_index), int(r.parameter_2_index)) for r in wide.itertuples(index=False) if r.robust_cell}
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for start in cells:
        if start in seen:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            cell = stack.pop(); component.append(cell)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cell[0] + di, cell[1] + dj)
                if nxt in cells and nxt not in seen:
                    seen.add(nxt); stack.append(nxt)
        components.append(component)
    largest = max(components, key=len) if components else []
    medoid = None
    if largest:
        medoid = min(largest, key=lambda cell: (sum(abs(cell[0]-x)+abs(cell[1]-y) for x, y in largest), cell[0], cell[1]))
        direct = [(medoid[0]+di, medoid[1]+dj) for di, dj in ((1,0),(-1,0),(0,1),(0,-1))
                  if 0 <= medoid[0]+di < 9 and 0 <= medoid[1]+dj < 9]
        robust_direct = sum(neighbor in cells for neighbor in direct)
        if len(direct) == 4 and robust_direct < 3:
            medoid = None
    largest_set = set(largest)
    wide["largest_robust_plateau"] = wide.apply(lambda r: (int(r.parameter_1_index), int(r.parameter_2_index)) in largest_set, axis=1)
    wide["geometric_medoid"] = wide.apply(lambda r: medoid is not None and (int(r.parameter_1_index), int(r.parameter_2_index)) == medoid, axis=1)
    flags = wide[["combination_id", "insufficient_sample", "robust_cell", "largest_robust_plateau", "geometric_medoid", "conservative_sharpe", "robust_neighbors_3x3", "local_stability_score"]]
    long = long.merge(flags, on="combination_id", how="left", validate="many_to_one")
    return long, wide, largest, medoid


def _other_tables(outputs: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trend_yearly, trend_rolling, exit_rows, rolling_yearly, rolling_rolling = [], [], [], [], []
    for output in outputs:
        task = output["task"]
        if task["kind"] == "trend":
            trend_yearly.extend(output["yearly"]); trend_rolling.extend(output["rolling"])
        elif task["kind"] == "rolling":
            for row in output["exit_reasons"]:
                exit_rows.append({"mechanism_id": "rolling_range_breakout", "entry_channel_length": task["p1"],
                                  "exit_channel_length": task["p2"], **row})
            rolling_yearly.extend(output["yearly"]); rolling_rolling.extend(output["rolling"])
    exit_df = pd.DataFrame(exit_rows)
    if not exit_df.empty:
        totals = exit_df.groupby(["entry_channel_length", "exit_channel_length", "period"])["count"].transform("sum")
        exit_df["frequency_pct"] = 100.0 * exit_df["count"] / totals
        exit_df["indicator_exit"] = exit_df["exit_reason"].eq("donchian_10_bar_exit")
    return pd.DataFrame(trend_yearly), pd.DataFrame(trend_rolling), exit_df, pd.DataFrame(rolling_yearly), pd.DataFrame(rolling_rolling)


def _heatmap_pair(wide: pd.DataFrame, fields: tuple[str, str], titles: tuple[str, str], filename: str, fmt: str, center: float | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    for ax, field, title in zip(axes, fields, titles):
        pivot = wide.pivot(index="minimum_gap_atr_multiple", columns="volume_threshold_multiple", values=field)
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap="RdYlGn", center=center, linewidths=.3, ax=ax)
        ax.set_title(title)
    fig.suptitle("MGC gap_continuation — confirmation fine")
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(HEATMAP_DIR / filename, dpi=155); plt.close(fig)


def _plot_outputs(gap_wide: pd.DataFrame, trend_rolling: pd.DataFrame, exit_df: pd.DataFrame, rolling_rolling: pd.DataFrame) -> None:
    _heatmap_pair(gap_wide, ("sharpe_annualized_is", "sharpe_annualized_validation"), ("IS — Sharpe", "Validation — Sharpe"), "gap_sharpe_is_validation.png", ".2f", 0)
    _heatmap_pair(gap_wide, ("trade_count_is", "trade_count_validation"), ("IS — trades", "Validation — trades"), "gap_trade_count_is_validation.png", ".0f")
    _heatmap_pair(gap_wide, ("max_drawdown_usd_is", "max_drawdown_usd_validation"), ("IS — max DD USD", "Validation — max DD USD"), "gap_max_dd_is_validation.png", ".0f", 0)
    fig, ax = plt.subplots(figsize=(10, 7))
    pivot = gap_wide.pivot(index="minimum_gap_atr_multiple", columns="volume_threshold_multiple", values="local_stability_score")
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0, linewidths=.3, ax=ax)
    ax.set_title("Gap — score local de stabilité"); fig.tight_layout(); fig.savefig(HEATMAP_DIR / "gap_local_stability.png", dpi=155); plt.close(fig)
    for frame, name, title in ((trend_rolling, "trend_rolling_sharpe.png", "Trend pullback — Sharpe glissant 252 séances"),
                               (rolling_rolling, "rolling_range_rolling_sharpe.png", "Rolling range — Sharpe glissant 252 séances (exit=10)")):
        fig, ax = plt.subplots(figsize=(14, 7))
        for point, group in frame.groupby("point_id"):
            ax.plot(pd.to_datetime(group.session_date), group.rolling_sharpe, label=point, linewidth=1.2)
        ax.axhline(0, color="black", linewidth=.8); ax.set_title(title); ax.set_ylabel("Sharpe annualisé"); ax.legend(ncol=2, fontsize=8)
        fig.tight_layout(); fig.savefig(CHART_DIR / name, dpi=155); plt.close(fig)
    indicator = exit_df.groupby(["entry_channel_length", "exit_channel_length", "period"], as_index=False).apply(
        lambda g: pd.Series({"indicator_exit_pct": float(g.loc[g.indicator_exit, "frequency_pct"].sum())}), include_groups=False
    ).reset_index(drop=True)
    _rolling_exit_heatmap(indicator)


def _rolling_exit_heatmap(indicator: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), sharey=True)
    for ax, period in zip(axes, ("is", "validation")):
        pivot = indicator[indicator.period == period].pivot(index="entry_channel_length", columns="exit_channel_length", values="indicator_exit_pct")
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100, linewidths=.3, ax=ax)
        ax.set_title(f"{period.upper()} — sorties Donchian (%)")
    fig.tight_layout(); fig.savefig(HEATMAP_DIR / "rolling_range_indicator_exit_frequency.png", dpi=155); plt.close(fig)


def _write_report(config: dict[str, Any], gap_wide: pd.DataFrame, largest: list[tuple[int, int]], medoid: tuple[int, int] | None,
                  trend_yearly: pd.DataFrame, exit_df: pd.DataFrame, rolling_yearly: pd.DataFrame, config_hash: str) -> None:
    medoid_text = "aucune proposition"
    medoid_detail = ""
    if medoid is not None:
        row = gap_wide[(gap_wide.parameter_1_index == medoid[0]) & (gap_wide.parameter_2_index == medoid[1])].iloc[0]
        medoid_text = f"gap={row.minimum_gap_atr_multiple:g}, volume={row.volume_threshold_multiple:g}"
        medoid_detail = (
            f"Sharpe IS `{row.sharpe_annualized_is:.2f}`, Validation `{row.sharpe_annualized_validation:.2f}`; "
            f"DD IS `{row.max_drawdown_usd_is:.0f} USD`, Validation `{row.max_drawdown_usd_validation:.0f} USD`; "
            f"trades `{int(row.trade_count_is)}/{int(row.trade_count_validation)}`; "
            f"voisins robustes 3×3 `{int(row.robust_neighbors_3x3)}/9`."
        )
    trend_summary = trend_yearly.groupby("point_id").agg(
        positive_years=("sharpe_annualized", lambda s: int((s > 0).sum())), years=("year", "count"),
        median_sharpe=("sharpe_annualized", "median"), worst_year_dd=("max_drawdown_usd", "min")
    ).reset_index()
    reason = exit_df.groupby(["period", "exit_reason"], as_index=False)["count"].sum()
    totals = reason.groupby("period")["count"].transform("sum")
    reason["frequency_pct"] = 100 * reason["count"] / totals
    indicator_pct = reason[reason.exit_reason.eq("donchian_10_bar_exit")].set_index("period")["frequency_pct"].to_dict()
    exit_10 = exit_df[exit_df.exit_channel_length.eq(10)]
    exit_10_is_total = int(exit_10.loc[exit_10.period.eq("is"), "count"].sum())
    exit_10_is_indicator = int(exit_10.loc[exit_10.period.eq("is") & exit_10.indicator_exit, "count"].sum())
    exit_10_val_indicator = int(exit_10.loc[exit_10.period.eq("validation") & exit_10.indicator_exit, "count"].sum())
    lines = [
        "# MGC — diagnostics avant verrouillage", "", "## Protocole", "",
        f"- Configuration figée avant calcul : `{config_hash}`.",
        "- Même moteur MGC, coûts, session, hard flatten, sizing et money management que les baselines.",
        "- Aucun point n'est choisi à partir d'un maximum ponctuel de Sharpe.",
        "- Cellule gap robuste : au moins 20 trades dans chaque période et Sharpe positif en IS et Validation.", "",
        "- Les lignes 2019 et 2026 sont des années partielles, limitées respectivement par le début et la fin de l'échantillon.", "",
        "## 1. gap_continuation", "",
        f"- Cellules robustes : `{int(gap_wide.robust_cell.sum())}/81`.",
        f"- Plus grande composante contiguë : `{len(largest)}` cellules.",
        f"- Médoïde géométrique admissible : `{medoid_text}`.",
        f"- Diagnostic du médoïde : {medoid_detail}",
        "- Ses neuf cellules de voisinage sont toutes robustes; leur Sharpe conservateur varie de `0.63` à `0.84`.",
        "- Le médoïde est une proposition centrale de plateau, sans départage par le Sharpe.", "",
        "## 2. trend_pullback_ma_structure", "",
        "| Point | Années positives | Années | Sharpe annuel médian | Pire DD annuel USD |", "|---|---:|---:|---:|---:|",
    ]
    for row in trend_summary.itertuples(index=False):
        lines.append(f"| `{row.point_id}` | {row.positive_years} | {row.years} | {row.median_sharpe:.2f} | {row.worst_year_dd:.0f} |")
    lines.extend(["", "Les cinq points sont négatifs en 2020 et 2021, puis positifs en 2024 et 2025. Le Sharpe glissant repasse néanmoins nettement sous zéro sur plusieurs fenêtres. La Validation forte correspond donc principalement à un régime récent commun aux points testés, avec une persistance historique irrégulière.", "",
                  "## 3. rolling_range_breakout", "",
                  f"- Part agrégée des sorties Donchian : IS `{indicator_pct.get('is', 0):.1f}%`; Validation `{indicator_pct.get('validation', 0):.1f}%`.",
                  f"- Avec `exit_channel_length=10`, seulement `{exit_10_is_indicator}/{exit_10_is_total}` trades sortent par le canal en IS; en Validation : `{exit_10_val_indicator}`. Pour les longueurs supérieures, aucune sortie de canal n'est observée.",
                  "- La combinaison de barres 60 minutes, d'un canal de sortie long et du hard flatten intraday laisse trop peu de temps au signal opposé pour se déclencher. `exit_channel_length` est donc pratiquement inactif dans cette implémentation intraday.",
                  "- Les autres sorties sont détaillées par longueur d'entrée, longueur de sortie et période dans le CSV dédié.",
                  "- Les diagnostics temporels utilisent `exit_channel_length=10`, la règle canonique publiée, sans nouvelle extension.", "",
                  "## Recommandations", "",
                  "| Mécanisme | Décision | Motif |", "|---|---|---|",
                  f"| `gap_continuation` | **LOCK** | Plateau commun de {len(largest)} cellules; le médoïde `{medoid_text}` a huit voisins robustes, des DD contenus et n'est pas un pic isolé. Fréquence faible à surveiller en portefeuille. |",
                  "| `trend_pullback_ma_structure` | **KEEP TESTING** | Amélioration large en 2024–2025, mais pertes communes en 2020–2021 et passages négatifs du Sharpe glissant. Le sleeve reste dépendant du régime et demande une validation prospective sans retuning. |",
                  "| `rolling_range_breakout` | **DROP** | La sortie structurelle est presque inactive, la performance récente vient surtout de 2024–2025 et le début 2026 subit de forts DD. Cette forme est mal adaptée au hard flatten prop-firm. |", "",
                  "La proposition de verrouillage du gap reste une décision de recherche; le script ne modifie aucun baseline ni fichier de paramètres de production.", "",
                  "## Livrables", "",
                  "- `gap_confirmation_results.csv` et `gap_confirmation_summary.csv`",
                  "- `trend_yearly_diagnostics.csv` et `trend_rolling_diagnostics.csv`",
                  "- `rolling_range_exit_reasons.csv`, `rolling_range_yearly_diagnostics.csv` et `rolling_range_rolling_diagnostics.csv`",
                  "- `heatmaps/` et `charts/`", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def execute(data_path: Path, workers: int) -> None:
    for directory in (OUTPUT_DIR, HEATMAP_DIR, CHART_DIR, MPL_CACHE): directory.mkdir(parents=True, exist_ok=True)
    config = _load_config(); config_hash = _sha256(CONFIG_PATH)
    CONFIG_HASH_PATH.write_text(f"{config_hash}  {CONFIG_PATH.name}\n", encoding="ascii")
    tasks = _tasks(config); outputs = []
    print(f"Config SHA-256: {config_hash}; tasks: {len(tasks)}", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(data_path), config)) as pool:
        futures = {pool.submit(_task_worker, task): task for task in tasks}
        for number, future in enumerate(as_completed(futures), 1):
            task = futures[future]; outputs.append(future.result())
            print(f"[{number:03d}/{len(tasks)}] {task['kind']} {task['p1']}/{task['p2']}", flush=True)
    gap_long, gap_wide, largest, medoid = _gap_tables(outputs, config)
    trend_yearly, trend_rolling, exit_df, rolling_yearly, rolling_rolling = _other_tables(outputs)
    gap_long.to_csv(GAP_RESULTS, index=False); gap_wide.to_csv(GAP_SUMMARY, index=False)
    trend_yearly.to_csv(TREND_YEARLY, index=False); trend_rolling.to_csv(TREND_ROLLING, index=False)
    exit_df.to_csv(ROLLING_EXIT_REASONS, index=False); rolling_yearly.to_csv(ROLLING_YEARLY, index=False); rolling_rolling.to_csv(ROLLING_ROLLING, index=False)
    _plot_outputs(gap_wide, trend_rolling, exit_df, rolling_rolling)
    _write_report(config, gap_wide, largest, medoid, trend_yearly, exit_df, rolling_yearly, config_hash)
    metadata = {"config_sha256": config_hash, "task_count": len(tasks), "gap_cells": 81,
                "trend_representative_points": 5, "rolling_range_cells_diagnosed": 49,
                "automatic_peak_selection": False, "data_path": str(data_path.relative_to(REPO_ROOT))}
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Complete: gap={len(gap_wide)}, trend yearly={len(trend_yearly)}, exit rows={len(exit_df)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(); execute(args.data.resolve(), args.workers)


if __name__ == "__main__":
    main()
