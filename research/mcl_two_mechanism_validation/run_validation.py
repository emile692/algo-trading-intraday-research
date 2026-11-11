"""Run the prelocked, restricted MCL robustness validation for two mechanisms."""
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

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CONFIG_PATH = OUT / "mcl_two_mechanism_validation_config.json"
MCL_CONFIG_PATH = ROOT / "research/mcl_baseline_backtests/mcl_baseline_config.json"
LANDSCAPE_PATH = ROOT / "research/mcl_wide_parameter_landscape/mcl_wide_landscape_results.csv"
DATA_PATH = ROOT / "data/downloaded/MCL_v_0_1m_20210712_20260924.parquet"
FROZEN_DIR = ROOT / "strategy_registry/psf_mnq/frozen_baselines"
FROZEN_MANIFEST = ROOT / "strategy_registry/psf_mnq/frozen_baselines_manifest.csv"
CHARTS = OUT / "charts"
TRADES = OUT / "trades"
MPL_CACHE = OUT / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from research.baseline_backtests.baseline_engine import (  # noqa: E402
    INITIAL_CAPITAL_USD,
    _rsi,
    load_frozen_specs,
    run_baseline_backtest,
)
from research.mcl_baseline_backtests.run_mcl_baselines import adapt, prepare  # noqa: E402
from research.mgc_wide_parameter_landscape.run_mgc_wide_landscape import _apply_parameters  # noqa: E402

_DATA: Any = None
_SPECS: dict[str, dict[str, Any]] = {}
_CONFIG: dict[str, Any] = {}
_MCL_CONFIG: dict[str, Any] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("status") != "LOCKED_BEFORE_DETAILED_VALIDATION":
        raise RuntimeError("Validation configuration is not locked")
    reps = config["representatives"]
    if set(reps) != {"short_horizon_pullback_reversion", "ma_crossover_trend"}:
        raise RuntimeError("Only the two authorized mechanisms may be evaluated")
    if any(not 3 <= len(points) <= 5 for points in reps.values()):
        raise RuntimeError("Each mechanism must have 3-5 predeclared representatives")
    return config


def init_worker(config: dict[str, Any]) -> None:
    global _DATA, _SPECS, _CONFIG, _MCL_CONFIG
    _CONFIG = config
    _MCL_CONFIG = json.loads(MCL_CONFIG_PATH.read_text(encoding="utf-8"))
    specs, _, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    _SPECS = {spec["mechanism_id"]: spec for spec in specs}
    start = date.fromisoformat(config["periods"]["FULL"][0])
    end = date.fromisoformat(config["periods"]["FULL"][1])
    _DATA = prepare(DATA_PATH, start, end, _MCL_CONFIG)


def metric_block(daily: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    pnl = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0).reset_index(drop=True)
    equity = INITIAL_CAPITAL_USD + pnl.cumsum()
    returns = pnl / equity.shift(1).fillna(INITIAL_CAPITAL_USD)
    std = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
    sharpe = float(returns.mean() / std * math.sqrt(252)) if np.isfinite(std) and std > 0 else np.nan
    peak = pd.concat([pd.Series([INITIAL_CAPITAL_USD]), equity], ignore_index=True).cummax().iloc[1:].reset_index(drop=True)
    drawdown = equity - peak
    trade_pnl = pd.to_numeric(trades.get("net_pnl_usd", pd.Series(dtype=float)), errors="coerce").dropna()
    wins, losses = trade_pnl[trade_pnl > 0], trade_pnl[trade_pnl < 0]
    gross_loss = abs(float(losses.sum()))
    pf = float(wins.sum()) / gross_loss if gross_loss > 0 else (np.inf if float(wins.sum()) > 0 else np.nan)
    return {
        "session_count": int(len(daily)),
        "trade_count": int(len(trades)),
        "net_pnl_usd": float(pnl.sum()),
        "profit_factor": pf,
        "sharpe_annualized": sharpe,
        "max_drawdown_usd": float(drawdown.min()) if len(drawdown) else 0.0,
        "win_rate_pct": float((trade_pnl > 0).mean() * 100.0) if len(trade_pnl) else np.nan,
    }


def slice_run(run: Any, start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = run.daily[run.daily["session_date"].between(start, end)].copy()
    trades = run.trades[run.trades["session_date"].between(start, end)].copy() if not run.trades.empty else run.trades.copy()
    return daily, trades


def spec_for(mechanism: str, representative: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    spec = adapt(_SPECS[mechanism], _MCL_CONFIG)
    if mechanism == "short_horizon_pullback_reversion":
        definition = {"parameter_1": "rsi_period", "parameter_2": "entry_rsi_extreme"}
        spec = _apply_parameters(spec, definition, representative["rsi_period"], representative["entry_rsi_extreme"])
    else:
        definition = {"parameter_1": "fast_ema_period", "parameter_2": "slow_ema_period"}
        spec = _apply_parameters(spec, definition, representative["fast_ema_period"], representative["slow_ema_period"])
    execution = spec["canonical_parameters"]["execution"]
    execution["slippage_ticks_per_side"] = int(scenario["slippage_ticks_per_side"])
    execution["commission_per_side_usd"] = float(scenario["commission_per_side_usd"])
    return spec


def violation_map(checks: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "lookahead_violations", "future_high_low_violations", "same_bar_policy_violations",
        "double_entry_violations", "overnight_position_violations",
        "entry_after_last_entry_violations", "post_hard_flatten_violations",
        "ambiguous_multi_order_cancellations",
    )
    return {key: checks.get(key) for key in keys if checks.get(key, 0)}


def rolling_rows(run: Any, mechanism: str, rep_id: str) -> list[dict[str, Any]]:
    window = int(_CONFIG["rolling_sharpe"]["window_sessions"])
    minimum = int(_CONFIG["rolling_sharpe"]["minimum_sessions"])
    daily = run.daily.sort_values("session_date").copy()
    pnl = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0)
    equity = INITIAL_CAPITAL_USD + pnl.cumsum()
    returns = pnl / equity.shift(1).fillna(INITIAL_CAPITAL_USD)
    sharpe = returns.rolling(window, min_periods=minimum).mean().div(
        returns.rolling(window, min_periods=minimum).std(ddof=1).replace(0.0, np.nan)
    ).mul(math.sqrt(252))
    return [
        {"mechanism_id": mechanism, "representative_id": rep_id,
         "session_date": daily.iloc[i]["session_date"], "rolling_sharpe_126": float(sharpe.iloc[i])}
        for i in range(len(daily)) if np.isfinite(sharpe.iloc[i])
    ]


def temporal_rows(run: Any, mechanism: str, rep_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    yearly, halfyear = [], []
    dates = pd.to_datetime(run.daily["session_date"])
    for year in sorted(dates.dt.year.unique()):
        start, end = date(int(year), 1, 1), date(int(year), 12, 31)
        daily, trades = slice_run(run, start, end)
        yearly.append({"mechanism_id": mechanism, "representative_id": rep_id, "year": int(year), **metric_block(daily, trades)})
        for half, (month_a, month_z, day_z) in (("H1", (1, 6, 30)), ("H2", (7, 12, 31))):
            start_h = date(int(year), month_a, 1)
            end_h = date(int(year), month_z, day_z)
            daily_h, trades_h = slice_run(run, start_h, end_h)
            if not daily_h.empty:
                halfyear.append({"mechanism_id": mechanism, "representative_id": rep_id,
                                 "subperiod": f"{year}-{half}", **metric_block(daily_h, trades_h)})
    return yearly, halfyear


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    mechanism, representative, scenario = task["mechanism"], task["representative"], task["scenario"]
    spec = spec_for(mechanism, representative, scenario)
    run = run_baseline_backtest(spec, _DATA)
    violations = violation_map(run.checks)
    if violations:
        raise RuntimeError(f"{representative['id']}::{scenario['id']} violations: {violations}")
    period_rows = []
    for period in ("IS", "OOS"):
        start, end = (date.fromisoformat(value) for value in _CONFIG["periods"][period])
        daily, trades = slice_run(run, start, end)
        period_rows.append({"mechanism_id": mechanism, "representative_id": representative["id"],
                            "role": representative["role"], "cost_scenario": scenario["id"],
                            **{k: v for k, v in representative.items() if k not in {"id", "role"}},
                            "period": period, **metric_block(daily, trades)})
    result: dict[str, Any] = {"period": period_rows, "yearly": [], "halfyear": [], "rolling": [], "trades": []}
    if scenario["id"] == "nominal":
        yearly, halfyear = temporal_rows(run, mechanism, representative["id"])
        result["yearly"], result["halfyear"] = yearly, halfyear
        result["rolling"] = rolling_rows(run, mechanism, representative["id"])
        result["trades"] = run.trades.to_dict("records")
    return result


def tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"mechanism": mechanism, "representative": representative, "scenario": scenario}
        for mechanism, representatives in config["representatives"].items()
        for representative in representatives
        for scenario in config["cost_scenarios"]
    ]


def landscape_wide() -> pd.DataFrame:
    raw = pd.read_csv(LANDSCAPE_PATH)
    identity = ["mechanism_id", "combination_id", "parameter_1_value", "parameter_1_index",
                "parameter_2_value", "parameter_2_index"]
    wide = raw.pivot(index=identity, columns="period", values=["sharpe_annualized", "trade_count"]).reset_index()
    wide.columns = ["_".join(str(x) for x in item if x) for item in wide.columns]
    wide["positive_both"] = wide["sharpe_annualized_is"].gt(0) & wide["sharpe_annualized_validation"].gt(0)
    return wide


def local_stability(config: dict[str, Any]) -> pd.DataFrame:
    wide = landscape_wide()
    rows = []
    for mechanism, representatives in config["representatives"].items():
        group = wide[wide["mechanism_id"] == mechanism]
        for representative in representatives:
            if mechanism == "short_horizon_pullback_reversion":
                p1, p2 = representative["rsi_period"], representative["entry_rsi_extreme"]
            else:
                p1, p2 = representative["fast_ema_period"], representative["slow_ema_period"]
            center = group[(group["parameter_1_value"] == p1) & (group["parameter_2_value"] == p2)].iloc[0]
            neighbors = group[
                group["parameter_1_index"].between(center.parameter_1_index - 1, center.parameter_1_index + 1)
                & group["parameter_2_index"].between(center.parameter_2_index - 1, center.parameter_2_index + 1)
            ]
            finite = neighbors.dropna(subset=["sharpe_annualized_is", "sharpe_annualized_validation"])
            rows.append({
                "mechanism_id": mechanism, "representative_id": representative["id"],
                "neighbor_cells": int(len(neighbors)), "finite_neighbor_cells": int(len(finite)),
                "positive_both_neighbors": int(finite["positive_both"].sum()),
                "positive_both_share": float(finite["positive_both"].mean()) if len(finite) else np.nan,
                "median_neighbor_sharpe_is": float(finite["sharpe_annualized_is"].median()) if len(finite) else np.nan,
                "median_neighbor_sharpe_oos": float(finite["sharpe_annualized_validation"].median()) if len(finite) else np.nan,
                "worst_neighbor_sharpe_is": float(finite["sharpe_annualized_is"].min()) if len(finite) else np.nan,
                "worst_neighbor_sharpe_oos": float(finite["sharpe_annualized_validation"].min()) if len(finite) else np.nan,
            })
    return pd.DataFrame(rows)


def rsi_quantiles(config: dict[str, Any]) -> pd.DataFrame:
    mcl_config = json.loads(MCL_CONFIG_PATH.read_text(encoding="utf-8"))
    start, end = (date.fromisoformat(value) for value in config["periods"]["FULL"])
    data = prepare(DATA_PATH, start, end, mcl_config)
    bars = data.bars[15].sort_values("end_time").copy()
    bars["sma200"] = bars["close"].rolling(200, min_periods=200).mean()
    rows = []
    for representative in config["representatives"]["short_horizon_pullback_reversion"]:
        period = int(representative["rsi_period"])
        threshold = float(representative["entry_rsi_extreme"])
        rsi = _rsi(bars["close"], period)
        for evaluation in ("IS", "OOS"):
            start_p, end_p = (date.fromisoformat(value) for value in config["periods"][evaluation])
            mask = pd.Series(bars["session_date"].between(start_p, end_p).to_numpy(), index=bars.index)
            long_eligible = mask & bars["sma200"].notna() & bars["close"].gt(bars["sma200"])
            short_eligible = mask & bars["sma200"].notna() & bars["close"].lt(bars["sma200"])
            long_tail = long_eligible & rsi.lt(threshold)
            short_tail = short_eligible & rsi.gt(100.0 - threshold)
            long_cross = long_tail & rsi.shift(1).ge(threshold)
            short_cross = short_tail & rsi.shift(1).le(100.0 - threshold)
            eligible_count = int(long_eligible.sum() + short_eligible.sum())
            rows.append({
                "representative_id": representative["id"], "period": evaluation,
                "rsi_period": period, "entry_rsi_extreme": threshold,
                "long_eligible_bars": int(long_eligible.sum()), "short_eligible_bars": int(short_eligible.sum()),
                "long_tail_percentile_pct": float(long_tail.sum() / long_eligible.sum() * 100.0),
                "short_tail_percentile_pct": float(short_tail.sum() / short_eligible.sum() * 100.0),
                "combined_tail_frequency_pct": float((long_tail.sum() + short_tail.sum()) / eligible_count * 100.0),
                "fresh_cross_count": int(long_cross.sum() + short_cross.sum()),
                "fresh_cross_frequency_pct": float((long_cross.sum() + short_cross.sum()) / eligible_count * 100.0),
            })
    return pd.DataFrame(rows)


def plot_outputs(period: pd.DataFrame, yearly: pd.DataFrame, rolling: pd.DataFrame, quantiles: pd.DataFrame) -> None:
    for mechanism, group in rolling.groupby("mechanism_id"):
        fig, ax = plt.subplots(figsize=(14, 7))
        for rep_id, rep in group.groupby("representative_id"):
            ax.plot(pd.to_datetime(rep["session_date"]), rep["rolling_sharpe_126"], label=rep_id, linewidth=1.1)
        ax.axhline(0, color="black", linewidth=.8)
        ax.axvspan(pd.Timestamp("2024-01-01"), pd.Timestamp("2026-03-19"), color="#dbeafe", alpha=.35, label="OOS")
        ax.set_title(f"MCL {mechanism} - Sharpe glissant 126 seances")
        ax.set_ylabel("Sharpe annualise")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout(); fig.savefig(CHARTS / f"{mechanism}_rolling_sharpe_126.png", dpi=155); plt.close(fig)

    nominal = period[(period["cost_scenario"] == "nominal")]
    for mechanism, group in yearly.groupby("mechanism_id"):
        pivot = group.pivot(index="year", columns="representative_id", values="net_pnl_usd")
        ax = pivot.plot(kind="bar", figsize=(14, 7), width=.82)
        ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("PnL USD")
        ax.set_title(f"MCL {mechanism} - PnL annuel nominal")
        ax.legend(ncol=2, fontsize=8); ax.figure.tight_layout()
        ax.figure.savefig(CHARTS / f"{mechanism}_yearly_pnl.png", dpi=155); plt.close(ax.figure)

    oos = period[period["period"] == "OOS"].pivot(index="representative_id", columns="cost_scenario", values="net_pnl_usd")
    ax = oos.plot(kind="bar", figsize=(15, 7), width=.82)
    ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("PnL OOS USD")
    ax.set_title("MCL - stress de couts/slippage OOS")
    ax.legend(fontsize=8); ax.figure.tight_layout()
    ax.figure.savefig(CHARTS / "oos_cost_stress_pnl.png", dpi=155); plt.close(ax.figure)

    pivot = quantiles.pivot(index="representative_id", columns="period", values="combined_tail_frequency_pct")
    ax = pivot.plot(kind="bar", figsize=(11, 6), width=.75)
    ax.set_ylabel("Frequence empirique de queue (%)")
    ax.set_title("Diagonale RSI - percentile implicite IS/OOS")
    ax.legend(); ax.figure.tight_layout()
    ax.figure.savefig(CHARTS / "rsi_diagonal_tail_percentiles.png", dpi=155); plt.close(ax.figure)


def execute(workers: int) -> None:
    for directory in (OUT, CHARTS, TRADES, MPL_CACHE):
        directory.mkdir(parents=True, exist_ok=True)
    config = load_config()
    config_hash = sha256(CONFIG_PATH)
    if sha256(DATA_PATH) != json.loads(MCL_CONFIG_PATH.read_text(encoding="utf-8"))["data"]["sha256"]:
        raise RuntimeError("MCL data hash mismatch")
    frozen_paths = [*sorted(FROZEN_DIR.glob("*.json")), FROZEN_MANIFEST]
    frozen_before = {str(path): sha256(path) for path in frozen_paths}
    task_list = tasks(config)
    outputs = []
    print(f"Locked config {config_hash}; {len(task_list)} restricted tasks", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(config,)) as pool:
        futures = {pool.submit(run_task, task): task for task in task_list}
        for number, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            outputs.append(future.result())
            print(f"[{number:02d}/{len(task_list)}] {task['representative']['id']} {task['scenario']['id']}", flush=True)
    if frozen_before != {str(path): sha256(path) for path in frozen_paths}:
        raise RuntimeError("Frozen source mechanisms changed during validation")

    period = pd.DataFrame(row for output in outputs for row in output["period"])
    yearly = pd.DataFrame(row for output in outputs for row in output["yearly"])
    halfyear = pd.DataFrame(row for output in outputs for row in output["halfyear"])
    rolling = pd.DataFrame(row for output in outputs for row in output["rolling"])
    period.sort_values(["mechanism_id", "representative_id", "cost_scenario", "period"]).to_csv(OUT / "period_metrics.csv", index=False)
    yearly.sort_values(["mechanism_id", "representative_id", "year"]).to_csv(OUT / "yearly_metrics.csv", index=False)
    halfyear.sort_values(["mechanism_id", "representative_id", "subperiod"]).to_csv(OUT / "halfyear_metrics.csv", index=False)
    rolling.sort_values(["mechanism_id", "representative_id", "session_date"]).to_csv(OUT / "rolling_sharpe_126.csv", index=False)
    for output in outputs:
        if output["trades"]:
            frame = pd.DataFrame(output["trades"])
            rep_id = next(row["representative_id"] for row in output["period"])
            frame.to_csv(TRADES / f"{rep_id}.csv", index=False)
    stability = local_stability(config)
    quantiles = rsi_quantiles(config)
    stability.to_csv(OUT / "local_stability.csv", index=False)
    quantiles.to_csv(OUT / "rsi_extreme_quantiles.csv", index=False)
    plot_outputs(period, yearly, rolling, quantiles)
    metadata = {
        "status": "COMPLETE", "config_sha256": config_hash, "data_sha256": sha256(DATA_PATH),
        "evaluated_mechanisms": list(config["representatives"]), "frozen_other_mechanisms": config["frozen_mechanisms"],
        "representative_count": 10, "cost_scenario_count": len(config["cost_scenarios"]),
        "task_count": len(task_list), "new_wide_grid": False, "automatic_best_sharpe_selection": False,
        "frozen_source_files_unchanged": True,
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "mcl_two_mechanism_validation_config.sha256").write_text(
        f"{config_hash}  {CONFIG_PATH.name}\n", encoding="ascii"
    )
    print(f"Complete: {len(period)} period rows; {len(yearly)} yearly rows; {len(rolling)} rolling rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    execute(args.workers)


if __name__ == "__main__":
    main()
