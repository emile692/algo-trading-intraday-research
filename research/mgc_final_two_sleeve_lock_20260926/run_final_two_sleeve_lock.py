"""Run the final closed MGC diagnostic and freeze passing sleeve candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = OUTPUT_DIR / "final_lock_config.json"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
SPEC_DIR = OUTPUT_DIR / "frozen_candidate_specs"
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
    FROZEN_DIR,
    FROZEN_MANIFEST,
    CONFIG_PATH as MGC_CONFIG_PATH,
    adapt_spec_to_mgc,
    prepare_mgc_data,
)
from research.mgc_wide_parameter_landscape.run_mgc_wide_landscape import _apply_parameters  # noqa: E402


_DATA: Any = None
_SPECS: dict[str, dict[str, Any]] = {}
_MGC_CONFIG: dict[str, Any] = {}
_CONFIG: dict[str, Any] = {}
FINAL_DATA = REPO_ROOT / "data" / "processed" / "parquet" / "MGC_v_0_1m_20150101_20260923.parquet"
EXPECTED_DATA_SHA256 = "2bbc2d10d59409fd347779fc00a23633fa6ad6aec0cf4b52e7a6d7a278426e96"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not config.get("locked_before_execution"):
        raise ValueError("Final diagnostic config must be locked before execution")
    gap = config["gap_continuation_refinement"]
    if len(gap["minimum_gap_atr_multiple"]) != 9 or len(gap["volume_threshold_multiple"]) != 9:
        raise ValueError("Final gap refinement must remain 9x9")
    if len(config["volatility_breakout_representatives"]) != 3:
        raise ValueError("Exactly three volatility representatives are allowed")
    return config


def init_worker(data_path: str, config: dict[str, Any]) -> None:
    global _DATA, _SPECS, _MGC_CONFIG, _CONFIG
    specs, _, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    _SPECS = {spec["mechanism_id"]: spec for spec in specs}
    _MGC_CONFIG = json.loads(MGC_CONFIG_PATH.read_text(encoding="utf-8"))
    _CONFIG = config
    start = date.fromisoformat(config["periods"]["is"][0])
    end = date.fromisoformat(config["periods"]["validation"][1])
    _DATA, _ = prepare_mgc_data(Path(data_path), start, end)


def sliced(run: Any, start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = run.daily[run.daily["session_date"].between(start, end)].copy()
    trades = run.trades[run.trades["session_date"].between(start, end)].copy() if not run.trades.empty else run.trades.copy()
    return daily, trades


def metrics(daily: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    pnl = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0).reset_index(drop=True)
    equity = INITIAL_CAPITAL_USD + pnl.cumsum()
    prior = equity.shift(1).fillna(INITIAL_CAPITAL_USD)
    returns = pnl / prior
    std = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
    sharpe = float(returns.mean() / std * math.sqrt(252)) if np.isfinite(std) and std > 0 else np.nan
    peak = pd.concat([pd.Series([INITIAL_CAPITAL_USD]), equity], ignore_index=True).cummax().iloc[1:].reset_index(drop=True)
    drawdown = equity - peak
    trade_pnl = pd.to_numeric(trades.get("net_pnl_usd", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    gross_profit = float(trade_pnl[trade_pnl.gt(0)].sum())
    gross_loss = float(-trade_pnl[trade_pnl.lt(0)].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else np.nan)
    return {
        "sharpe_annualized": sharpe,
        "profit_factor": pf,
        "max_drawdown_usd": float(drawdown.min()) if len(drawdown) else 0.0,
        "net_pnl_usd": float(pnl.sum()),
        "trade_count": int(len(trades)),
        "session_count": int(len(daily)),
    }


def rolling_summary(run: Any) -> dict[str, Any]:
    cfg = _CONFIG["rolling_diagnostic"]
    daily = run.daily.sort_values("session_date").copy()
    pnl = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0)
    equity = INITIAL_CAPITAL_USD + pnl.cumsum()
    returns = pnl / equity.shift(1).fillna(INITIAL_CAPITAL_USD)
    mean = returns.rolling(int(cfg["window_sessions"]), min_periods=int(cfg["minimum_sessions"])).mean()
    std = returns.rolling(int(cfg["window_sessions"]), min_periods=int(cfg["minimum_sessions"])).std(ddof=1)
    values = mean.div(std.where(std.gt(0))).mul(math.sqrt(252)).dropna()
    return {
        "rolling_observation_count": int(len(values)),
        "rolling_sharpe_median": float(values.median()) if len(values) else np.nan,
        "rolling_sharpe_minimum": float(values.min()) if len(values) else np.nan,
        "rolling_sharpe_positive_pct": float(100.0 * values.gt(0).mean()) if len(values) else np.nan,
    }


def worker(task: dict[str, Any]) -> dict[str, Any]:
    mechanism = task["mechanism_id"]
    if mechanism == "volatility_breakout":
        definition = {"parameter_1": "daily_atr_period", "parameter_2": "entry_atr_multiple"}
    else:
        definition = {"parameter_1": "minimum_gap_atr_multiple", "parameter_2": "volume_threshold_multiple"}
    spec = adapt_spec_to_mgc(_SPECS[mechanism], _MGC_CONFIG)
    spec = _apply_parameters(spec, definition, task["p1"], task["p2"])
    run = run_baseline_backtest(spec, _DATA)
    violations = {key: value for key, value in run.checks.items() if key.endswith("violations") and value}
    if violations:
        raise RuntimeError(f"{task['point_id']}: {violations}")
    period_rows = []
    for period in ("is", "validation"):
        start, end = (date.fromisoformat(value) for value in _CONFIG["periods"][period])
        daily, trades = sliced(run, start, end)
        period_rows.append({"period": period, **metrics(daily, trades)})
    yearly_rows = []
    years = sorted(pd.to_datetime(run.daily["session_date"]).dt.year.unique())
    for year in years:
        daily, trades = sliced(run, date(int(year), 1, 1), date(int(year), 12, 31))
        if not daily.empty:
            yearly_rows.append({"year": int(year), **metrics(daily, trades)})
    return {"task": task, "period_metrics": period_rows, "yearly": yearly_rows, "rolling": rolling_summary(run)}


def build_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = []
    for index, point in enumerate(config["volatility_breakout_representatives"]):
        p1, p2 = point["daily_atr_period"], point["entry_atr_multiple"]
        tasks.append({"mechanism_id": "volatility_breakout", "point_id": f"volatility_{p1}_{p2:g}",
                      "p1": p1, "p2": p2, "i": index, "j": index})
    gap = config["gap_continuation_refinement"]
    for i, p1 in enumerate(gap["minimum_gap_atr_multiple"]):
        for j, p2 in enumerate(gap["volume_threshold_multiple"]):
            tasks.append({"mechanism_id": "gap_continuation", "point_id": f"gap_{i:02d}_{j:02d}",
                          "p1": p1, "p2": p2, "i": i, "j": j})
    return tasks


def tables(outputs: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    period_rows, yearly_rows, rolling_rows = [], [], []
    for output in outputs:
        task = output["task"]
        identity = {"mechanism_id": task["mechanism_id"], "point_id": task["point_id"],
                    "parameter_1_value": task["p1"], "parameter_2_value": task["p2"],
                    "parameter_1_index": task["i"], "parameter_2_index": task["j"]}
        period_rows.extend([{**identity, **row} for row in output["period_metrics"]])
        yearly_rows.extend([{**identity, **row} for row in output["yearly"]])
        rolling_rows.append({**identity, **output["rolling"]})
    long = pd.DataFrame(period_rows).sort_values(["mechanism_id", "parameter_1_index", "parameter_2_index", "period"])
    identity_cols = ["mechanism_id", "point_id", "parameter_1_value", "parameter_2_value", "parameter_1_index", "parameter_2_index"]
    value_cols = ["sharpe_annualized", "profit_factor", "max_drawdown_usd", "net_pnl_usd", "trade_count", "session_count"]
    wide = long.pivot(index=identity_cols, columns="period", values=value_cols).reset_index()
    wide.columns = ["_".join(filter(None, item)).strip("_") if isinstance(item, tuple) else item for item in wide.columns]
    return wide, pd.DataFrame(yearly_rows), pd.DataFrame(rolling_rows)


def robust_flags(wide: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = wide.copy()
    gates = config["freeze_gates"]
    out["conservative_sharpe"] = out[["sharpe_annualized_is", "sharpe_annualized_validation"]].min(axis=1)
    out["robust_cell"] = (
        out["trade_count_is"].ge(gates["minimum_trades_each_period"])
        & out["trade_count_validation"].ge(gates["minimum_trades_each_period"])
        & out["sharpe_annualized_is"].gt(gates["minimum_sharpe_each_period_exclusive"])
        & out["sharpe_annualized_validation"].gt(gates["minimum_sharpe_each_period_exclusive"])
        & out["profit_factor_is"].gt(gates["minimum_profit_factor_each_period_exclusive"])
        & out["profit_factor_validation"].gt(gates["minimum_profit_factor_each_period_exclusive"])
    )
    return out


def largest_component_and_medoid(gap: pd.DataFrame) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    cells = {(int(row.parameter_1_index), int(row.parameter_2_index)) for row in gap.itertuples(index=False) if row.robust_cell}
    seen: set[tuple[int, int]] = set(); components: list[list[tuple[int, int]]] = []
    for start in cells:
        if start in seen:
            continue
        stack, component = [start], []; seen.add(start)
        while stack:
            cell = stack.pop(); component.append(cell)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (cell[0] + di, cell[1] + dj)
                if neighbor in cells and neighbor not in seen:
                    seen.add(neighbor); stack.append(neighbor)
        components.append(component)
    largest = max(components, key=len) if components else []
    medoid = min(largest, key=lambda cell: (sum(abs(cell[0]-i)+abs(cell[1]-j) for i, j in largest), cell)) if largest else None
    return largest, medoid


def add_neighbor_diagnostics(wide: pd.DataFrame) -> pd.DataFrame:
    out = wide.copy()
    for mechanism in out.mechanism_id.unique():
        group = out[out.mechanism_id.eq(mechanism)]
        lookup = {(int(r.parameter_1_index), int(r.parameter_2_index)): r for r in group.itertuples(index=False)}
        for idx, row in group.iterrows():
            i, j = int(row.parameter_1_index), int(row.parameter_2_index)
            neighbors = [r for (a, b), r in lookup.items() if abs(a-i) <= 1 and abs(b-j) <= 1]
            out.loc[idx, "neighbor_count"] = len(neighbors)
            out.loc[idx, "robust_neighbors"] = sum(bool(r.robust_cell) for r in neighbors)
            out.loc[idx, "neighbor_conservative_sharpe_min"] = min(float(r.conservative_sharpe) for r in neighbors)
            out.loc[idx, "neighbor_conservative_sharpe_median"] = float(np.median([float(r.conservative_sharpe) for r in neighbors]))
    return out


def volatility_neighbor_table() -> pd.DataFrame:
    source = pd.read_csv(REPO_ROOT / "research" / "mgc_volatility_breakout_refinement_v0_20260925" / "volatility_breakout_refinement_ranking.csv")
    rows = []
    representatives = [(10, .8), (30, 1.1), (50, 1.3)]
    for period, threshold in representatives:
        center = source[np.isclose(source.parameter_1_value, period) & np.isclose(source.parameter_2_value, threshold)].iloc[0]
        neighbors = source[(source.parameter_1_index.sub(center.parameter_1_index).abs() <= 1)
                           & (source.parameter_2_index.sub(center.parameter_2_index).abs() <= 1)].copy()
        conservative = neighbors[["sharpe_annualized_is", "sharpe_annualized_validation"]].min(axis=1)
        rows.append({"point_id": f"volatility_{period}_{threshold:g}", "neighbor_count": len(neighbors),
                     "positive_both_neighbors": int(neighbors.positive_both.sum()),
                     "neighbor_conservative_sharpe_min": float(conservative.min()),
                     "neighbor_conservative_sharpe_median": float(conservative.median())})
    return pd.DataFrame(rows)


def plot_gap(gap: pd.DataFrame) -> None:
    for fields, titles, filename, fmt, center in (
        (("sharpe_annualized_is", "sharpe_annualized_validation"), ("IS — Sharpe", "Validation — Sharpe"), "gap_sharpe_is_validation.png", ".2f", 0),
        (("profit_factor_is", "profit_factor_validation"), ("IS — Profit factor", "Validation — Profit factor"), "gap_pf_is_validation.png", ".2f", 1),
        (("max_drawdown_usd_is", "max_drawdown_usd_validation"), ("IS — max DD USD", "Validation — max DD USD"), "gap_maxdd_is_validation.png", ".0f", 0),
        (("trade_count_is", "trade_count_validation"), ("IS — trades", "Validation — trades"), "gap_trades_is_validation.png", ".0f", None),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        for ax, field, title in zip(axes, fields, titles):
            pivot = gap.pivot(index="parameter_1_value", columns="parameter_2_value", values=field)
            sns.heatmap(pivot, annot=True, fmt=fmt, cmap="RdYlGn", center=center, linewidths=.3, ax=ax)
            ax.set_title(title); ax.set_xlabel("volume_threshold_multiple"); ax.set_ylabel("minimum_gap_atr_multiple")
        fig.suptitle("MGC gap_continuation — raffinement final fermé")
        fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(HEATMAP_DIR / filename, dpi=155); plt.close(fig)


def positive_complete_years(yearly: pd.DataFrame, point_id: str, config: dict[str, Any]) -> int:
    years = set(config["freeze_gates"]["complete_years"])
    frame = yearly[yearly.point_id.eq(point_id) & yearly.year.isin(years)]
    return int(frame.sharpe_annualized.gt(0).sum())


def passes_freeze(row: pd.Series, yearly: pd.DataFrame, config: dict[str, Any]) -> tuple[bool, int]:
    count = positive_complete_years(yearly, row.point_id, config)
    minimum = int(config["freeze_gates"]["minimum_positive_complete_years"])
    return bool(row.robust_cell and count >= minimum), count


def effective_spec(mechanism: str, p1: float, p2: float, decision: str, config_hash: str) -> dict[str, Any]:
    specs, _, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    source = next(spec for spec in specs if spec["mechanism_id"] == mechanism)
    mgc_config = json.loads(MGC_CONFIG_PATH.read_text(encoding="utf-8"))
    if mechanism == "volatility_breakout":
        definition = {"parameter_1": "daily_atr_period", "parameter_2": "entry_atr_multiple"}
    else:
        definition = {"parameter_1": "minimum_gap_atr_multiple", "parameter_2": "volume_threshold_multiple"}
    spec = _apply_parameters(adapt_spec_to_mgc(source, mgc_config), definition, p1, p2)
    payload = {key: value for key, value in spec.items() if not key.startswith("_")}
    payload["specification_status"] = "frozen_mgc_sleeve_candidate"
    payload["freeze_metadata"] = {
        "decision": decision,
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "final_lock_config_sha256": config_hash,
        "validation_status": "adaptive_reused_not_pristine_oos",
        "parameter_selection": "representative_of_plateau_not_peak_sharpe",
        "parameter_optimization_after_lock": False,
    }
    return payload


def write_freeze(rows: list[pd.Series], config_hash: str) -> tuple[list[dict[str, Any]], str]:
    manifest_rows = []
    for row in rows:
        mechanism = str(row.mechanism_id)
        spec = effective_spec(mechanism, float(row.parameter_1_value), float(row.parameter_2_value), "FREEZE_CANDIDATE", config_hash)
        path = SPEC_DIR / f"{mechanism}.json"
        path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest_rows.append({
            "mechanism_id": mechanism,
            "strategy_id": spec["strategy_id"],
            "status": "FREEZE_CANDIDATE",
            "parameter_1_value": float(row.parameter_1_value),
            "parameter_2_value": float(row.parameter_2_value),
            "specification_path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "specification_sha256": sha256(path),
            "validation_status": "adaptive_reused_not_pristine_oos",
            "no_further_parameter_optimization": True,
        })
    csv_path = OUTPUT_DIR / "frozen_candidate_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(csv_path, index=False)
    manifest = {
        "schema_version": "1.0",
        "status": "MGC_SLEEVE_CANDIDATES_FROZEN" if manifest_rows else "NO_MGC_SLEEVE_CANDIDATE_FROZEN",
        "final_lock_config_sha256": config_hash,
        "validation_status": "adaptive_reused_not_pristine_oos",
        "no_further_parameter_optimization": True,
        "candidates": manifest_rows,
    }
    json_path = OUTPUT_DIR / "frozen_candidate_manifest.json"
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_hash = sha256(json_path)
    (OUTPUT_DIR / "frozen_candidate_manifest.sha256").write_text(f"{manifest_hash}  {json_path.name}\n", encoding="ascii")
    return manifest_rows, manifest_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=FINAL_DATA)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for directory in (OUTPUT_DIR, HEATMAP_DIR, SPEC_DIR, MPL_CACHE):
        directory.mkdir(parents=True, exist_ok=True)
    config = load_config(); config_hash = sha256(CONFIG_PATH)
    data_path = args.data.resolve()
    data_hash = sha256(data_path)
    if data_hash != EXPECTED_DATA_SHA256:
        raise RuntimeError(f"Final lock requires the MGC v0 parquet {EXPECTED_DATA_SHA256}; got {data_hash}")
    (OUTPUT_DIR / "final_lock_config.sha256").write_text(f"{config_hash}  {CONFIG_PATH.name}\n", encoding="ascii")
    tasks = build_tasks(config); outputs = []
    print(f"Locked config: {config_hash}; tasks: {len(tasks)}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(str(data_path), config)) as pool:
        futures = {pool.submit(worker, task): task for task in tasks}
        for number, future in enumerate(as_completed(futures), 1):
            task = futures[future]; outputs.append(future.result())
            print(f"[{number:03d}/{len(tasks)}] {task['point_id']}", flush=True)
    wide, yearly, rolling = tables(outputs)
    wide = robust_flags(wide, config)
    gap = wide[wide.mechanism_id.eq("gap_continuation")].copy()
    gap = add_neighbor_diagnostics(gap)
    volatility = wide[wide.mechanism_id.eq("volatility_breakout")].copy()
    vol_neighbors = volatility_neighbor_table()
    volatility = volatility.merge(vol_neighbors, on="point_id", how="left", validate="one_to_one")
    largest, medoid = largest_component_and_medoid(gap)
    gap_rep = None if medoid is None else gap[
        gap.parameter_1_index.eq(medoid[0]) & gap.parameter_2_index.eq(medoid[1])
    ].iloc[0]
    vol_rep = volatility[volatility.point_id.eq("volatility_30_1.1")].iloc[0]
    vol_pass, vol_positive_years = passes_freeze(vol_rep, yearly, config)
    gap_pass, gap_positive_years = (False, 0) if gap_rep is None else passes_freeze(gap_rep, yearly, config)
    selected = [vol_rep] if vol_pass else []
    if gap_pass and gap_rep is not None:
        selected.append(gap_rep)
    manifest_rows, manifest_hash = write_freeze(selected, config_hash)
    volatility.to_csv(OUTPUT_DIR / "volatility_representative_diagnostics.csv", index=False)
    gap.to_csv(OUTPUT_DIR / "gap_refinement_diagnostics.csv", index=False)
    yearly.to_csv(OUTPUT_DIR / "yearly_diagnostics.csv", index=False)
    rolling.to_csv(OUTPUT_DIR / "rolling_diagnostics.csv", index=False)
    plot_gap(gap)

    def line(row: pd.Series, years: int, passed: bool) -> str:
        return (f"| `{row.point_id}` | {row.sharpe_annualized_is:.3f} | {row.sharpe_annualized_validation:.3f} | "
                f"{row.profit_factor_is:.3f} | {row.profit_factor_validation:.3f} | "
                f"{int(row.trade_count_is)} | {int(row.trade_count_validation)} | "
                f"{row.max_drawdown_usd_is:.0f} | {row.max_drawdown_usd_validation:.0f} | {years}/6 | "
                f"{'FREEZE' if passed else 'DO NOT FREEZE'} |")
    vol_lines = []
    for row in volatility.sort_values("parameter_1_index").itertuples(index=False):
        series = pd.Series(row._asdict()); passed, years = passes_freeze(series, yearly, config)
        vol_lines.append(line(series, years, passed))
    if gap_rep is None:
        gap_details = [
            "- Aucune cellule ne passe simultanément les gates préengagées de trades, Sharpe et PF dans les deux périodes.",
            "- Il n'existe donc ni composante robuste ni médoïde admissible dans cette zone; aucun fallback ni relâchement de seuil n'est appliqué.",
            "- Décision : **DO NOT FREEZE**.",
        ]
        gap_medoid_text = "aucune"
    else:
        gap_details = [
            f"- Médoïde géométrique : `minimum_gap_atr_multiple={gap_rep.parameter_1_value:g}`, `volume_threshold_multiple={gap_rep.parameter_2_value:g}`.",
            f"- Sharpe IS/Validation : `{gap_rep.sharpe_annualized_is:.3f}/{gap_rep.sharpe_annualized_validation:.3f}`; PF `{gap_rep.profit_factor_is:.3f}/{gap_rep.profit_factor_validation:.3f}`; trades `{int(gap_rep.trade_count_is)}/{int(gap_rep.trade_count_validation)}`; DD `{gap_rep.max_drawdown_usd_is:.0f}/{gap_rep.max_drawdown_usd_validation:.0f} USD`.",
            f"- Années complètes positives : `{gap_positive_years}/6`; voisinage robuste `{int(gap_rep.robust_neighbors)}/{int(gap_rep.neighbor_count)}`.",
            f"- Décision : **{'FREEZE CANDIDATE' if gap_pass else 'DO NOT FREEZE'}**.",
        ]
        gap_medoid_text = f"{gap_rep.parameter_1_value:g}/{gap_rep.parameter_2_value:g}"
    report = [
        "# MGC — verrouillage final de deux sleeves", "", "## Statut méthodologique", "",
        f"- Configuration verrouillée avant ce calcul : `{config_hash}`.",
        "- La Validation 2024-01-01 → 2026-03-19 a déjà servi à cibler ces tests : elle est **adaptative et non OOS vierge**.",
        "- Aucun nouveau paramètre ni nouvelle famille n'a été exploré après ce passage.",
        "- Les représentants sont choisis par règle de plateau, jamais par maximum de Sharpe.", "",
        "## Volatility breakout — trois représentants fermés", "",
        "| Point | Sharpe IS | Sharpe Val | PF IS | PF Val | Trades IS | Trades Val | DD IS USD | DD Val USD | Années complètes + | Décision brute |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|", *vol_lines, "",
        f"Représentant imposé par la règle centrale : `30 / 1.10`. Décision : **{'FREEZE CANDIDATE' if vol_pass else 'DO NOT FREEZE'}**. "
        f"Il compte {vol_positive_years}/6 années complètes positives; son voisinage fin contient {int(vol_rep.positive_both_neighbors)}/{int(vol_rep.neighbor_count)} cellules positives dans les deux périodes, avec Sharpe conservateur voisin médian {vol_rep.neighbor_conservative_sharpe_median:.2f}.", "",
        "## Gap continuation — raffinement local fermé", "",
        f"- Cellules robustes (trades, Sharpe et PF) : `{int(gap.robust_cell.sum())}/81`.",
        f"- Plus grande composante 4-connectée : `{len(largest)}` cellules.",
        *gap_details, "",
        "## Provenance des données", "",
        f"- Parquet MGC v0 imposé : `{data_path.relative_to(REPO_ROOT)}`.",
        f"- SHA-256 vérifié : `{data_hash}`.", "",
        "## Freeze", "",
        f"- Candidats écrits au manifeste : `{len(manifest_rows)}`.",
        f"- SHA-256 du manifeste : `{manifest_hash}`.",
        "- Les specs portent `parameter_optimization_after_lock=false` et remplacent uniquement le statut de candidat; aucun fichier baseline source ou ancien verrou n'est modifié.",
        "- Toute évaluation future doit être prospective ou sur un segment réellement vierge, sans retuning.", "",
    ]
    (OUTPUT_DIR / "FINAL_TWO_SLEEVE_LOCK_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {
        "run_type": "final_closed_two_sleeve_mgc_lock", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash, "manifest_sha256": manifest_hash, "task_count": len(tasks),
        "gap_cells": 81, "volatility_points": 3, "validation_status": "adaptive_reused_not_pristine_oos",
        "no_further_parameter_optimization": True, "data_path": str(data_path.relative_to(REPO_ROOT)),
        "data_sha256": data_hash,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Gap robust: {int(gap.robust_cell.sum())}/81; plateau: {len(largest)}; medoid: {gap_medoid_text}", flush=True)
    print(f"Freeze decisions: volatility={vol_pass}, gap={gap_pass}; manifest candidates={len(manifest_rows)}", flush=True)


if __name__ == "__main__":
    main()
