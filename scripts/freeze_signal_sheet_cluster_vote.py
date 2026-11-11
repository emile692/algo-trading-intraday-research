"""Freeze one S03-S09 parameter cluster and execute its aggregate vote.

The cluster is the lock neighbourhood selected by
``select_signal_sheet_clusters_cscv.py``.  Each cell is the exact cell of the
refinement grid (same engine, data and frozen parameters); the vote uses the
aggregator already executed for S02, so S02-S09 share one vote semantics.

Outputs: ``strategy_registry/psf_mnq/frozen_ensembles/<ensemble>.json`` and
``research/signal_sheet_<NN>_cluster/``.  Historical lock files are never
rewritten; the new manifest states what it supersedes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from rebuild_signal_sheet_02_cluster import aggregate_vote
from run_signal_sheet_parameter_refinement import (
    CAMPAIGNS,
    ROOT,
    apply_parameters,
    metric_block,
    prepare_campaign,
    run_candidate,
)
from select_signal_sheet_clusters_cscv import LOCK_NEIGHBOURHOOD


ENSEMBLE_DIR = ROOT / "strategy_registry" / "psf_mnq" / "frozen_ensembles"
SELECTION_EVIDENCE = ROOT / "research" / "signal_sheet_parameter_refinement" / "cluster_selection" / "cluster_selection_cscv.json"
FROZEN_ON = "2026-11-01"
WATCHED_CHECKS = (
    "lookahead_violations", "future_high_low_violations", "same_bar_policy_violations",
    "double_entry_violations", "overnight_position_violations", "entry_after_last_entry_violations",
    "post_hard_flatten_violations", "ambiguous_multi_order_cancellations", "missing_atr_entry_skips",
)

SUPERSEDES = {
    "s03": {"type": "single_point", "parameters": {"nr_lookback": 10, "entry_buffer_ticks": 1, "target_setup_range_multiple": 2.0},
            "source": "research/locked_four_sleeve_portfolio/locked_four_sleeve_config.json"},
    "s04": {"type": "single_point", "parameters": {"atr_period": 10, "atr_multiplier": 1.75, "stop_atr": 1.0},
            "source": "research/locked_four_sleeve_portfolio/locked_four_sleeve_config.json"},
    "s05": {"type": "single_point", "parameters": {"daily_atr_period": 30, "entry_atr_multiple": 1.1},
            "source": "research/mgc_final_two_sleeve_lock_20260926/final_lock_config.json"},
    "s06": {"type": "single_point", "parameters": {"minimum_gap_atr_multiple": 0.5, "volume_threshold_multiple": 0.6},
            "source": "research/mgc_final_two_sleeve_lock_20260926/gap_user_override_0_50_0_60.json"},
    "s07": {"type": "vote_region", "parameters": {"entry_channel_length": [20, 24], "exit_channel_length": [7, 10, 12, 15]},
            "source": "research/mbt_three_sleeve_freeze_20261004/vote_ensemble/ensemble_manifest.json",
            "change": "exit length 15 removed: identical to 12 for every entry length, it doubled two votes"},
    "s08": {"type": "vote_region", "parameters": {"fast_pair_scale": [0.5, 0.6, 0.75], "slow_tail_scale": [0.75, 1.0, 1.25]},
            "source": "research/mbt_three_sleeve_freeze_20261004/vote_ensemble/ensemble_manifest.json",
            "change": "region unchanged; re-executed with the common S02-S09 vote engine over the full sample"},
    "s09": {"type": "vote_region", "parameters": {"daily_atr_period": [10, 14, 20, 30], "entry_atr_multiple": [0.3, 0.4, 0.6]},
            "source": "research/mbt_three_sleeve_freeze_20261004/vote_ensemble/ensemble_manifest.json",
            "change": "region unchanged; re-executed with the common S02-S09 vote engine over the full sample"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def identifiers(key: str) -> tuple[str, str, int]:
    campaign = CAMPAIGNS[key]
    region = LOCK_NEIGHBOURHOOD[key]
    count = math.prod(len(values) for values in region.values())
    ensemble_id = f"{campaign['instrument'].lower()}_{campaign['mechanism']}_vote_{count}"
    return ensemble_id, ensemble_id.replace("_", "-"), count


def build_manifest(key: str, spec: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    campaign = CAMPAIGNS[key]
    region = LOCK_NEIGHBOURHOOD[key]
    ensemble_id, _, count = identifiers(key)
    execution = spec["canonical_parameters"]["execution"]
    directory = ROOT / "research" / "signal_sheet_parameter_refinement" / key
    return {
        "schema_version": "1.0",
        "configuration_type": "parameter_vote_ensemble",
        "configuration_status": "frozen_after_cscv_rule_validation",
        "frozen_on": FROZEN_ON,
        "instrument": campaign["instrument"],
        "mechanism_id": campaign["mechanism"],
        "sheet": campaign["sheet"],
        "ensemble_id": ensemble_id,
        "parameter_region": {**region, "cartesian_product": True, "submodel_count": count},
        "cell_definition": {
            "engine": "scripts/run_signal_sheet_parameter_refinement.py (apply_parameters + run_candidate), identical to the refinement grid cell",
            "other_parameters": "frozen at the values documented in the refinement parameter inventory",
            "internal_quantity": "virtual_state_only",
            "state_space": [-1, 0, 1],
        },
        "vote": {
            "score": f"arithmetic_mean_of_{count}_cell_position_states",
            "long_target": "score >= 0.50",
            "short_target": "score <= -0.50",
            "flat_target": "-0.50 < score < 0.50",
            "threshold": 0.5,
            "minimum_net_directional_votes": math.ceil(count / 2),
            "exact_threshold_included": True,
            "conflict_policy": "opposing cell states offset in the arithmetic mean",
            "target_space": [-1, 0, 1],
            "aggregator": "scripts/rebuild_signal_sheet_02_cluster.py::aggregate_vote (shared with S02)",
        },
        "aggregate_execution": {
            "quantity": 1,
            "target_change_execution": "open of the next available one-minute bar after the cell-state change becomes observable",
            "commission_per_side_usd": float(execution["commission_per_side_usd"]),
            "slippage_ticks_per_side": int(execution["slippage_ticks_per_side"]),
            "tick_size": float(execution["tick_size"]),
            "point_value_usd": float(execution["point_value_usd"]),
            "hard_flatten_time": "16:08:00 America/New_York",
            "cost_policy": "Costs apply only to changes in the aggregate target; virtual cell transitions are not charged as separate contracts.",
            "no_position_after_hard_flatten": True,
        },
        "historical_windows": {
            "is": [campaign["start"], campaign["is_end"]],
            "oos_validation_reused": [campaign["oos_start"], campaign["end"]],
        },
        "selection": {
            "method": "lock_neighbourhood: one grid step around the declared lock, or the frozen MBT region itself",
            "rationale": "Inside-IS CSCV showed that every performance-optimising cluster rule had a mean out-of-fold rank below 0.5 and PBO above 50 percent; no new degree of freedom is fitted.",
            "evidence_path": relative(SELECTION_EVIDENCE),
            "evidence_sha256": sha256(SELECTION_EVIDENCE),
        },
        "selection_disclosure": "The OOS window was read before this freeze and was consulted for earlier locks: it is reused validation, not untouched prospective OOS. Prospective evidence starts after 2026-10-02.",
        "research_evidence": {
            "grid_metrics_path": relative(directory / "grid_metrics.csv"),
            "grid_metrics_sha256": sha256(directory / "grid_metrics.csv"),
            "cell_daily_pnl_path": relative(directory / "cell_daily_pnl.parquet"),
            "cell_daily_pnl_sha256": sha256(directory / "cell_daily_pnl.parquet"),
            "minute_data_path": source["data_path"].replace("\\", "/"),
            "minute_data_sha256": source["data_sha256"],
        },
        "supersedes": SUPERSEDES[key],
    }


def build_cell_positions(key: str, base_spec: dict[str, Any], prepared: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    campaign = CAMPAIGNS[key]
    region = LOCK_NEIGHBOURHOOD[key]
    names = list(region)
    rows, checks = [], {}
    for values in itertools.product(*(region[name] for name in names)):
        params = dict(zip(names, values))
        identifier = "__".join(f"{name}={value:g}" for name, value in params.items())
        run = run_candidate(key, apply_parameters(base_spec, campaign, params), prepared, campaign)
        violations = {name: run.checks.get(name) for name in WATCHED_CHECKS if run.checks.get(name, 0)}
        if violations:
            raise RuntimeError(f"{key} {identifier}: {violations}")
        positions = run.positions.copy()
        positions["cell_id"] = identifier
        for name, value in params.items():
            positions[name] = value
        rows.append(positions)
        checks[identifier] = run.checks
        print(f"{key} {identifier}: {len(positions)} positions virtuelles", flush=True)
    return pd.concat(rows, ignore_index=True), checks


def audit_event_alignment(positions: pd.DataFrame, prepared: Any) -> None:
    """Every cell event must fall on a minute bar, otherwise the vote would silently miss it."""
    minutes = pd.DatetimeIndex(prepared.minute.index)
    flatten = set(pd.to_datetime(prepared.sessions["hard_flatten"]))
    for column in ("entry_time", "exit_time"):
        stamps = pd.to_datetime(positions[column])
        missing = ~(stamps.isin(minutes) | stamps.isin(flatten))
        if missing.any():
            raise RuntimeError(f"{int(missing.sum())} {column} values do not match a minute bar")


def relabel(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    ensemble_id, strategy_id, count = identifiers(key)
    mapping = {
        "volatility_breakout_vote_42": ensemble_id,
        "volatility-breakout-vote-42": strategy_id,
        "42_cell_vote": f"{count}_cell_vote",
    }
    frame = frame.copy()
    for column in ("mechanism_id", "strategy_id", "entry_reason", "setup_id"):
        if column in frame:
            frame[column] = frame[column].replace(mapping)
    return frame


def scoped(result_daily: pd.DataFrame, trades: pd.DataFrame, campaign: dict[str, Any]) -> dict[str, Any]:
    daily = result_daily.copy()
    daily["session_date"] = pd.to_datetime(daily["session_date"])
    trades = trades.copy()
    if not trades.empty:
        trades["session_date"] = pd.to_datetime(trades["session_date"])
    windows = {
        "full_sample": (campaign["start"], campaign["end"]),
        "is": (campaign["start"], campaign["is_end"]),
        "oos_validation_reused": (campaign["oos_start"], campaign["end"]),
    }
    metrics = {}
    for scope, (left, right) in windows.items():
        left, right = pd.Timestamp(left), pd.Timestamp(right)
        metrics[scope] = metric_block(
            daily[daily["session_date"].between(left, right)],
            trades[trades["session_date"].between(left, right)] if not trades.empty else trades,
        )
    return metrics


def freeze(key: str) -> None:
    campaign = CAMPAIGNS[key]
    ensemble_id, _, count = identifiers(key)
    manifest_path = ENSEMBLE_DIR / f"{ensemble_id}.json"
    output_dir = ROOT / "research" / f"signal_sheet_{campaign['sheet'][:2]}_cluster"
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared, base_spec, source = prepare_campaign(key, campaign)
    manifest = build_manifest(key, base_spec, source)
    positions, cell_checks = build_cell_positions(key, base_spec, prepared)
    if positions["cell_id"].nunique() != count:
        raise RuntimeError(f"{key}: {positions['cell_id'].nunique()} cells executed, {count} expected")
    audit_event_alignment(positions, prepared)

    result, vote_events = aggregate_vote(base_spec, prepared, positions, manifest)
    violations = {name: result.checks.get(name) for name in WATCHED_CHECKS if result.checks.get(name, 0)}
    if violations:
        raise RuntimeError(f"{key} aggregate vote: {violations}")
    trades, daily = relabel(result.trades, key), relabel(result.daily, key)

    entry_delay = (pd.to_datetime(trades["entry_time"]) - pd.to_datetime(trades["signal_time"])).dt.total_seconds() / 60.0
    ordered = trades.sort_values("entry_time")
    overlapping = int((pd.to_datetime(ordered["entry_time"]).iloc[1:].to_numpy()
                       < pd.to_datetime(ordered["exit_time"]).iloc[:-1].to_numpy()).sum()) if len(ordered) > 1 else 0
    if overlapping or (len(entry_delay) and entry_delay.min() < 1):
        raise RuntimeError(f"{key}: overlapping aggregate trades {overlapping}, minimum entry delay {entry_delay.min()}")

    positions.to_csv(output_dir / "cell_positions.csv", index=False)
    vote_events.to_csv(output_dir / "vote_events.csv", index=False)
    trades.to_csv(output_dir / "trades.csv", index=False)
    daily.to_csv(output_dir / "daily_results.csv", index=False)
    metrics = scoped(daily, trades, campaign)
    summary = {
        "ensemble_manifest": relative(manifest_path),
        "data_sha256": source["data_sha256"],
        "submodel_count": count,
        "vote_threshold": 0.5,
        "execution_semantics": manifest["aggregate_execution"],
        "aggregate_checks": result.checks,
        "cell_count": int(positions["cell_id"].nunique()),
        "cell_position_rows": int(len(positions)),
        "cell_checks_all_valid": True,
        "sharpe_definition": "sqrt(252) x mean / std of daily net PnL, as in the refinement grid",
        "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    (output_dir / "cell_checks.json").write_text(json.dumps(cell_checks, indent=2, default=str) + "\n", encoding="utf-8")

    files = ("summary.json", "trades.csv", "daily_results.csv", "vote_events.csv", "cell_positions.csv", "cell_checks.json")
    manifest["execution_results"] = {
        "executed_on": FROZEN_ON,
        "script": "scripts/freeze_signal_sheet_cluster_vote.py",
        "result_directory": relative(output_dir),
        "files": {name: sha256(output_dir / name) for name in files},
        "metrics": metrics,
        "audit": {
            "cell_count": count,
            "cell_position_rows": int(len(positions)),
            "vote_event_rows": int(len(vote_events)),
            "entry_delay_minutes_min": float(entry_delay.min()) if len(entry_delay) else None,
            "overlapping_aggregate_trades": overlapping,
            "post_hard_flatten_violations": int(result.checks.get("post_hard_flatten_violations", 0)),
            "all_cell_checks_valid": True,
            "event_alignment_verified": True,
        },
    }
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": relative(manifest_path), **{scope: {k: row[k] for k in ("trade_count", "net_pnl_usd", "sharpe_annualized", "profit_factor", "max_drawdown_usd")} for scope, row in metrics.items()}}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleeve", required=True, choices=sorted(LOCK_NEIGHBOURHOOD))
    args = parser.parse_args()
    freeze(args.sleeve)


if __name__ == "__main__":
    sys.exit(main())
