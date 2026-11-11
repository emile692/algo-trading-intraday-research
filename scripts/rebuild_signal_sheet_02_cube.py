"""Build the full 3-D robustness grid used by signal sheet 02.

The grid crosses the two alpha parameters with the locked time-stop study.  It
uses the frozen MNQ specification and the same causal execution engine, costs,
dates, and one-minute data as the terminal money-management research.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.baseline_backtests.baseline_engine import (
    build_buy_and_hold,
    compute_metrics,
    load_frozen_specs,
    prepare_data,
    slice_prepared_data,
)
from research.money_management_lab.mm_engine import (
    build_overlay_context,
    run_exit_overlay_backtest,
    slice_result,
)


OUTPUT_DIR = ROOT / "research" / "signal_sheet_02_cube"
RESULT_PATH = OUTPUT_DIR / "cube_metrics.csv"
METADATA_PATH = OUTPUT_DIR / "run_metadata.json"
FROZEN_DIR = ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
DATA_PATH = ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"

ATR_PERIODS = tuple(range(10, 31, 2))
ENTRY_MULTIPLES = tuple(round(value / 10, 1) for value in range(2, 9))
MAX_HOLDING_MINUTES = (60, 90, 120, 150, 180, 210, 240, 270, 300, 360)
RETAINED_ATR_PERIODS = tuple(range(10, 23, 2))
RETAINED_ENTRY_MULTIPLES = (0.5, 0.6)
RETAINED_MAX_HOLDING_MINUTES = (270, 300, 360)
IS_START, IS_END = date(2019, 5, 6), date(2023, 12, 31)
OOS_START, OOS_END = date(2024, 1, 1), date(2026, 3, 19)
METRICS = (
    "sharpe_annualized",
    "net_pnl_usd",
    "expectancy_per_trade_usd",
    "profit_factor",
    "max_drawdown_usd",
    "trade_count",
)
CHECK_KEYS = (
    "lookahead_violations",
    "future_high_low_violations",
    "same_bar_policy_violations",
    "double_entry_violations",
    "overnight_position_violations",
    "entry_after_last_entry_violations",
    "post_hard_flatten_violations",
    "ambiguous_multi_order_cancellations",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def period_metrics(spec: dict[str, Any], run: Any, benchmark: pd.DataFrame, start: date, end: date) -> dict[str, Any]:
    return compute_metrics(spec, slice_result(run, start, end), benchmark)


def row_key(atr_period: Any, entry_multiple: Any, duration: Any) -> tuple[int, float, int]:
    return int(atr_period), round(float(entry_multiple), 1), int(duration)


def reusable_rows(seed_path: Path | None, atr_periods: tuple[int, ...]) -> list[dict[str, Any]]:
    if seed_path is None or not seed_path.is_file():
        return []
    seed = pd.read_csv(seed_path)
    required = {"daily_atr_period", "entry_atr_multiple", "max_holding_minutes", "is_backtest_valid", "oos_backtest_valid"}
    if not required.issubset(seed.columns):
        return []
    selected = seed[
        seed["daily_atr_period"].isin(atr_periods)
        & seed["entry_atr_multiple"].round(1).isin(ENTRY_MULTIPLES)
        & seed["max_holding_minutes"].isin(MAX_HOLDING_MINUTES)
        & seed["is_backtest_valid"].astype(bool)
        & seed["oos_backtest_valid"].astype(bool)
    ].copy()
    selected = selected.drop_duplicates(
        ["daily_atr_period", "entry_atr_multiple", "max_holding_minutes"], keep="last"
    )
    return selected.to_dict("records")


def write_metadata(results: pd.DataFrame) -> None:
    metadata = {
        "created_for": "signal_sheet_02",
        "grid_rows": len(results),
        "daily_atr_periods": list(ATR_PERIODS),
        "entry_atr_multiples": list(ENTRY_MULTIPLES),
        "max_holding_minutes": list(MAX_HOLDING_MINUTES),
        "is_period": [IS_START.isoformat(), IS_END.isoformat()],
        "oos_validation_period": [OOS_START.isoformat(), OOS_END.isoformat()],
        "retained_parameter_region": {
            "daily_atr_period": list(RETAINED_ATR_PERIODS),
            "entry_atr_multiple": list(RETAINED_ENTRY_MULTIPLES),
            "max_holding_minutes": list(RETAINED_MAX_HOLDING_MINUTES),
            "submodel_count": 42,
            "vote_threshold": 0.5,
            "minimum_net_directional_votes": 21,
        },
        "data_sha256": sha256(DATA_PATH),
        "frozen_spec_sha256": sha256(FROZEN_DIR / "volatility_breakout.json"),
        "all_backtests_valid": True,
        "performance_maximum_used_for_lock": False,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def validate_results(results: pd.DataFrame, atr_periods: tuple[int, ...]) -> pd.DataFrame:
    results = results.drop_duplicates(
        ["daily_atr_period", "entry_atr_multiple", "max_holding_minutes"], keep="last"
    ).sort_values(["daily_atr_period", "entry_atr_multiple", "max_holding_minutes"])
    results["retained_parameters"] = (
        results["daily_atr_period"].isin(RETAINED_ATR_PERIODS)
        & results["entry_atr_multiple"].round(1).isin(RETAINED_ENTRY_MULTIPLES)
        & results["max_holding_minutes"].isin(RETAINED_MAX_HOLDING_MINUTES)
    )
    expected_rows = len(atr_periods) * len(ENTRY_MULTIPLES) * len(MAX_HOLDING_MINUTES)
    if len(results) != expected_rows or not results[["is_backtest_valid", "oos_backtest_valid"]].all().all():
        raise RuntimeError(
            f"The 3-D robustness grid is incomplete or invalid: {len(results)}/{expected_rows} rows"
        )
    return results


def execute(
    atr_periods: tuple[int, ...] = ATR_PERIODS,
    result_path: Path = RESULT_PATH,
    seed_path: Path | None = RESULT_PATH,
    metadata: bool = True,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    specs, benchmark_spec, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    frozen = next(spec for spec in specs if spec["mechanism_id"] == "volatility_breakout")
    prepared = prepare_data(DATA_PATH, start_date=IS_START, end_date=OOS_END)
    benchmarks = {
        "is": build_buy_and_hold(benchmark_spec, slice_prepared_data(prepared, IS_START, IS_END)).daily,
        "oos": build_buy_and_hold(benchmark_spec, slice_prepared_data(prepared, OOS_START, OOS_END)).daily,
    }

    rows = reusable_rows(seed_path, atr_periods)
    completed = {
        row_key(row["daily_atr_period"], row["entry_atr_multiple"], row["max_holding_minutes"])
        for row in rows
    }
    print(f"Reusing {len(completed)} valid rows from {seed_path}", flush=True)
    for atr_period in atr_periods:
        for entry_multiple in ENTRY_MULTIPLES:
            missing_durations = [
                duration
                for duration in MAX_HOLDING_MINUTES
                if row_key(atr_period, entry_multiple, duration) not in completed
            ]
            if not missing_durations:
                print(f"ATR={atr_period:>2} multiple={entry_multiple:.1f}: already complete", flush=True)
                continue
            spec = copy.deepcopy(frozen)
            structural = spec["canonical_parameters"]["structural"]
            structural["daily_atr_period"] = atr_period
            structural["entry_atr_multiple"] = entry_multiple
            context = build_overlay_context(spec, prepared, 5)
            for duration in missing_durations:
                policy = {"family": "time_stop", "max_holding_minutes": duration}
                run = run_exit_overlay_backtest(spec, prepared, policy, 5, context=context)
                if run.checks["missing_atr_entry_skips"] or any(run.checks.get(key, 0) for key in CHECK_KEYS):
                    raise RuntimeError(
                        f"Execution check failed for ATR={atr_period}, multiple={entry_multiple}, duration={duration}: {run.checks}"
                    )
                is_metrics = period_metrics(spec, run, benchmarks["is"], IS_START, IS_END)
                oos_metrics = period_metrics(spec, run, benchmarks["oos"], OOS_START, OOS_END)
                row: dict[str, Any] = {
                    "daily_atr_period": atr_period,
                    "entry_atr_multiple": entry_multiple,
                    "max_holding_minutes": duration,
                    "is_backtest_valid": bool(is_metrics["backtest_valid"]),
                    "oos_backtest_valid": bool(oos_metrics["backtest_valid"]),
                    "retained_parameters": (
                        atr_period in RETAINED_ATR_PERIODS
                        and entry_multiple in RETAINED_ENTRY_MULTIPLES
                        and duration in RETAINED_MAX_HOLDING_MINUTES
                    ),
                }
                for metric in METRICS:
                    row[f"is_{metric}"] = is_metrics[metric]
                    row[f"oos_{metric}"] = oos_metrics[metric]
                rows.append(row)
                completed.add(row_key(atr_period, entry_multiple, duration))
                print(
                    f"ATR={atr_period:>2} multiple={entry_multiple:.1f} duration={duration:>3} "
                    f"Sharpe IS/OOS={is_metrics['sharpe_annualized']:.3f}/{oos_metrics['sharpe_annualized']:.3f}",
                    flush=True,
                )
            checkpoint = pd.DataFrame(rows).drop_duplicates(
                ["daily_atr_period", "entry_atr_multiple", "max_holding_minutes"], keep="last"
            ).sort_values(["daily_atr_period", "entry_atr_multiple", "max_holding_minutes"])
            checkpoint.to_csv(result_path, index=False)

    results = validate_results(pd.DataFrame(rows), atr_periods)
    results.to_csv(result_path, index=False)
    if metadata and atr_periods == ATR_PERIODS and result_path.resolve() == RESULT_PATH.resolve():
        write_metadata(results)
    display_path = result_path.relative_to(ROOT) if result_path.is_relative_to(ROOT) else result_path
    print(f"Wrote {display_path} ({len(results)} rows)")
    retained = results[results["retained_parameters"]]
    print(f"Retained region: {len(retained)} submodels")


def merge_shards(paths: list[Path]) -> None:
    frames = [pd.read_csv(path) for path in paths]
    results = validate_results(pd.concat(frames, ignore_index=True), ATR_PERIODS)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_PATH, index=False)
    write_metadata(results)
    print(f"Merged {len(paths)} shards into {RESULT_PATH.relative_to(ROOT)} ({len(results)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--atr-periods", help="Comma-separated subset, used for parallel shards")
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--seed-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--merge-shards", nargs="+", type=Path)
    args = parser.parse_args()
    if args.merge_shards:
        merge_shards([path.resolve() for path in args.merge_shards])
    else:
        selected = ATR_PERIODS if not args.atr_periods else tuple(int(value) for value in args.atr_periods.split(","))
        unknown = set(selected) - set(ATR_PERIODS)
        if unknown:
            raise ValueError(f"ATR periods outside the locked display grid: {sorted(unknown)}")
        execute(
            selected,
            args.result_path.resolve(),
            args.seed_path.resolve() if args.seed_path else None,
            not args.no_metadata,
        )
