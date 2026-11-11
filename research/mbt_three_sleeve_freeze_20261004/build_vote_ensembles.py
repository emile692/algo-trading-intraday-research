"""Build executable one-contract MBT ensembles from the frozen parameter regions.

Every cell runs independently.  At each minute close, cell positions are
averaged.  The ensemble target for the next minute open is -1, 0, or +1:
mean <= -0.5, abs(mean) < 0.5, mean >= +0.5.  Exact ties are taken toward the
position.  This makes the aggregation causal and executable with one contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.baseline_backtests.baseline_engine import load_frozen_specs, run_baseline_backtest
from research.btc_baseline_backtests.run_btc_baselines import adapt_spec_to_btc, prepare_btc_data
from research.mbt_strategy_heatmaps_20261003.run_mbt_heatmaps import mbt_config
from research.mbt_three_sleeve_freeze_20261004.run_analysis import REGIONS, apply_cell


HERE = Path(__file__).resolve().parent
OUT = HERE / "vote_ensemble"
REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "downloaded" / "MBT_v_0_1m_20210425_20261003.parquet"
FROZEN_DIR = REPO / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = REPO / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
START = date(2021, 5, 3)
END = date(2026, 10, 2)
EXPORT_START = date(2024, 1, 1)
POINT_VALUE = 0.10
TICK_SIZE = 5.0
COMMISSION_SIDE = 1.41


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discretize(values: pd.Series) -> pd.Series:
    raw = values.to_numpy(float)
    result = np.where(raw >= 0.5, 1, np.where(raw <= -0.5, -1, 0))
    return pd.Series(result.astype(np.int8), index=values.index)


def position_from_trades(index: pd.DatetimeIndex, trades: pd.DataFrame) -> pd.Series:
    changes = np.zeros(len(index) + 1, dtype=np.int16)
    for row in trades.itertuples(index=False):
        entry = int(index.searchsorted(pd.Timestamp(row.entry_time), side="left"))
        exit_ = int(index.searchsorted(pd.Timestamp(row.exit_time), side="left"))
        if entry >= len(index) or exit_ <= 0:
            continue
        entry = max(entry, 0)
        exit_ = min(exit_, len(index))
        direction = int(row.position)
        changes[entry] += direction
        changes[exit_] -= direction
    return pd.Series(np.cumsum(changes[:-1]).astype(np.int8), index=index)


def causal_target(vote_at_close: pd.Series, sessions: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"vote": vote_at_close.astype(np.int8), "session_date": sessions}, index=vote_at_close.index)
    target = frame.groupby("session_date", sort=False)["vote"].shift(1).fillna(0).astype(np.int8)
    # The hard-flatten bar is an unconditional zero target.
    last = frame.groupby("session_date", sort=False).tail(1).index
    target.loc[last] = 0
    return target


def execute_target(
    bars: pd.DataFrame,
    target: pd.Series,
    sleeve: str,
    slippage_ticks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    minute_rows: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []
    for day, group in bars.groupby("session_date", sort=True):
        pos = 0
        cash = 0.0
        entry_time: pd.Timestamp | None = None
        entry_fill = np.nan
        entry_fees = 0.0
        trade_id = len(rows)
        path = []
        adverse_path = []
        position_path = []
        side_count = 0
        for stamp, bar in group.iterrows():
            wanted = int(target.loc[stamp])
            delta = wanted - pos
            if delta:
                # A flip closes one contract and opens the other: two sides.
                fill = float(bar.open) + np.sign(delta) * slippage_ticks * TICK_SIZE
                cash -= delta * fill * POINT_VALUE
                fee = abs(delta) * COMMISSION_SIDE
                cash -= fee
                side_count += abs(delta)
                if pos != 0:
                    exit_fill = fill
                    close_fee = COMMISSION_SIDE
                    trade_id += 1
                    net = (exit_fill - entry_fill) * pos * POINT_VALUE - entry_fees - close_fee
                    rows.append({
                        "sleeve": sleeve,
                        "trade_id": trade_id,
                        "session_date": day,
                        "position": pos,
                        "quantity": 1,
                        "entry_time": entry_time,
                        "exit_time": stamp,
                        "entry_price": entry_fill,
                        "exit_price": exit_fill,
                        "commissions_usd": entry_fees + close_fee,
                        "slippage_ticks_per_side": slippage_ticks,
                        "net_pnl_usd": net,
                    })
                    entry_time = None
                if wanted != 0:
                    entry_time = stamp
                    entry_fill = fill
                    entry_fees = COMMISSION_SIDE
                pos = wanted
            equity = cash + pos * float(bar.close) * POINT_VALUE
            adverse_mark = float(bar.low) if pos > 0 else float(bar.high) if pos < 0 else float(bar.close)
            adverse_equity = cash + pos * adverse_mark * POINT_VALUE
            path.append(equity)
            adverse_path.append(adverse_equity)
            position_path.append(pos)
        if pos != 0:
            raise RuntimeError(f"{sleeve} remained open after {day}")
        minute = pd.DataFrame({
            f"position__{sleeve}": position_path,
            f"equity_from_day_start__{sleeve}": path,
            f"adverse_equity_from_day_start__{sleeve}": adverse_path,
        }, index=group.index)
        minute_rows.append(minute)
        daily_rows.append({
            "session_date": day,
            "sleeve": sleeve,
            "daily_pnl_usd": cash,
            "trade_count": side_count // 2,
            "contract_sides": side_count,
            "worst_intraday_pnl_usd": float(np.min(adverse_path)),
        })
    return pd.DataFrame(rows), pd.DataFrame(daily_rows), pd.concat(minute_rows).sort_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prepared, roll_sessions = prepare_btc_data(DATA, START, END)
    specs, _, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    specs = {item["mechanism_id"]: item for item in specs}
    config = mbt_config()
    export_minute = prepared.minute.loc[pd.to_datetime(prepared.minute["session_date"]) >= pd.Timestamp(EXPORT_START)].copy()
    cell_positions: dict[str, pd.Series] = {}
    cell_checks: dict[str, dict[str, Any]] = {}

    for mechanism, region in REGIONS.items():
        keys = list(region)
        for first in region[keys[0]]:
            for second in region[keys[1]]:
                spec = copy.deepcopy(adapt_spec_to_btc(specs[mechanism], config))
                spec, params = apply_cell(spec, mechanism, first, second)
                run = run_baseline_backtest(spec, prepared)
                violations = {key: value for key, value in run.checks.items() if key.endswith("violations") and value}
                if violations:
                    raise RuntimeError(f"{mechanism} {params}: {violations}")
                cell_id = mechanism + "__" + "__".join(f"{key}={params[key]}" for key in keys)
                cell_positions[cell_id] = position_from_trades(export_minute.index, run.trades)
                cell_checks[cell_id] = run.checks

    state = pd.DataFrame(cell_positions, index=export_minute.index).astype(np.int8)
    votes = pd.DataFrame(index=state.index)
    targets = pd.DataFrame(index=state.index)
    diagnostics: list[dict[str, Any]] = []
    for mechanism in REGIONS:
        columns = [column for column in state if column.startswith(mechanism + "__")]
        mean_vote = state[columns].mean(axis=1)
        vote = discretize(mean_vote)
        target = causal_target(vote, export_minute["session_date"])
        votes[f"mean_vote__{mechanism}"] = mean_vote
        votes[f"vote__{mechanism}"] = vote
        targets[mechanism] = target
        diagnostics.append({
            "sleeve": mechanism,
            "cell_count": len(columns),
            "active_vote_minutes": int(vote.ne(0).sum()),
            "long_vote_minutes": int(vote.gt(0).sum()),
            "short_vote_minutes": int(vote.lt(0).sum()),
            "exact_positive_ties": int(np.isclose(mean_vote, 0.5).sum()),
            "exact_negative_ties": int(np.isclose(mean_vote, -0.5).sum()),
        })

    # Allocation-family helper: rolling/trend jointly consume at most one MBT contract.
    cluster_vote = discretize((targets["rolling_range_breakout"] + targets["trend_pullback_ma_structure"]) / 2.0)
    targets["rolling_trend_cluster_cap"] = cluster_vote.astype(np.int8)

    all_daily = []
    all_minute = []
    trade_hashes: dict[str, str] = {}
    for slippage in (1, 2):
        label = "nominal" if slippage == 1 else "two_ticks_per_side"
        for sleeve in targets:
            trades, daily, minute = execute_target(export_minute, targets[sleeve], sleeve, slippage)
            trades_path = OUT / f"trades__{sleeve}__{label}.csv"
            trades.to_csv(trades_path, index=False)
            trade_hashes[f"{sleeve}__{label}"] = sha256(trades_path)
            daily.insert(1, "cost_scenario", label)
            all_daily.append(daily)
            renamed = minute.rename(columns={column: column + f"__{label}" for column in minute.columns})
            all_minute.append(renamed)

    daily = pd.concat(all_daily, ignore_index=True)
    minute = pd.concat(all_minute, axis=1).sort_index()
    state.to_csv(OUT / "cell_position_states.csv", index_label="timestamp")
    pd.concat([votes, targets.add_prefix("target__")], axis=1).to_csv(OUT / "ensemble_votes_and_targets.csv", index_label="timestamp")
    pd.DataFrame(diagnostics).to_csv(OUT / "vote_diagnostics.csv", index=False)
    daily.to_csv(OUT / "daily_ensemble_results.csv", index=False)
    minute.to_csv(OUT / "minute_ensemble_results.csv", index_label="timestamp")
    manifest = {
        "schema_version": "1.0",
        "created": "2026-10-04",
        "source_freeze_manifest": str((HERE / "freeze_manifest.json").relative_to(REPO)),
        "source_freeze_manifest_sha256": sha256(HERE / "freeze_manifest.json"),
        "source_data": str(DATA.relative_to(REPO)),
        "source_data_sha256": sha256(DATA),
        "cell_regions": REGIONS,
        "aggregation": {
            "cell_signal": "independent realized position after each cell's own entries, stops and exits",
            "decision_time": "minute close",
            "execution_time": "next available minute open in the same session",
            "mean": "arithmetic mean across all cell positions in the sleeve",
            "discretization": {"short": "mean <= -0.5", "flat": "-0.5 < mean < 0.5", "long": "mean >= 0.5"},
            "tie_policy": "exact +/-0.5 rounds away from zero into the position",
            "quantity": "one MBT contract",
            "hard_flatten": "16:08 America/New_York",
        },
        "economics": {
            "point_value_usd": POINT_VALUE,
            "tick_size": TICK_SIZE,
            "tick_value_usd": POINT_VALUE * TICK_SIZE,
            "commission_per_side_usd": COMMISSION_SIDE,
            "nominal_slippage_ticks_per_side": 1,
            "stressed_slippage_ticks_per_side": 2,
        },
        "cluster_cap_rule": "discretize the mean of the already-causal rolling-range and trend-pullback targets; at most one contract for that pair",
        "export_range": [str(export_minute.session_date.min()), str(export_minute.session_date.max())],
        "intraday_roll_sessions_excluded": [str(item) for item in roll_sessions],
        "trade_file_sha256": trade_hashes,
        "frozen_source_strategies_modified": False,
        "pass_rate_used_for_selection": False,
    }
    (OUT / "ensemble_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"diagnostics": diagnostics, "daily_rows": len(daily), "minute_rows": len(minute), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
