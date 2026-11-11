"""Topstep 50K integration of the six frozen sleeves and three MBT vote ensembles."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SIX_DIR = REPO / "research" / "portfolio_integration_phase1"
MBT_DIR = REPO / "research" / "mbt_three_sleeve_freeze_20261004" / "vote_ensemble"
MNQ_DATA = REPO / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"
MGC_DATA = REPO / "data" / "processed" / "parquet" / "MGC_v_0_1m_20150101_20260923.parquet"
START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2026-03-19")
SEED = 20261004
BLOCKS = (3, 5, 10, 20)
PATHS_PER_BLOCK = 2_000
HORIZONS = (35, 60)
STARTING_BALANCE = 50_000.0
PROFIT_TARGET = 3_000.0
MLL_DISTANCE = 2_000.0
CONSISTENCY = 0.55
LOSS_VOL_STRESS = 1.25

OLD = [
    "mnq_orb_vwap_atr", "mnq_volatility_breakout", "mnq_compression_breakout",
    "mnq_atr_trend_flip", "mgc_volatility_breakout", "mgc_gap_continuation",
]
NEW = ["mbt_rolling_range_breakout", "mbt_trend_pullback_ma_structure", "mbt_volatility_breakout"]
ALL = OLD + NEW
INSTRUMENT = {name: name.split("_")[0].upper() for name in ALL}
POINT_VALUE = {"MNQ": 2.0, "MGC": 10.0, "MBT": 0.10}
TICK_VALUE = {"MNQ": 0.50, "MGC": 1.00, "MBT": 0.50}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_six() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    pnl = pd.read_csv(SIX_DIR / "daily_sleeve_pnl.csv", parse_dates=["session_date"]).set_index("session_date")[OLD].astype(float)
    counts = pd.read_csv(SIX_DIR / "daily_trade_counts.csv", parse_dates=["session_date"]).set_index("session_date")[OLD].astype(int)
    trades = {name: pd.read_csv(SIX_DIR / "sleeve_trades" / f"{name}.csv", parse_dates=["entry_ts_utc", "exit_ts_utc"]) for name in OLD}
    return pnl, counts, trades


def load_mbt(calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(MBT_DIR / "daily_ensemble_results.csv", parse_dates=["session_date"])
    name_map = {
        "rolling_range_breakout": NEW[0],
        "trend_pullback_ma_structure": NEW[1],
        "volatility_breakout": NEW[2],
        "rolling_trend_cluster_cap": "mbt_rolling_trend_cluster_cap",
    }
    pivots = {}
    count_pivots = {}
    for scenario in ("nominal", "two_ticks_per_side"):
        part = daily[daily.cost_scenario.eq(scenario)].copy()
        p = part.pivot(index="session_date", columns="sleeve", values="daily_pnl_usd").rename(columns=name_map)
        c = part.pivot(index="session_date", columns="sleeve", values="trade_count").rename(columns=name_map)
        pivots[scenario] = p.reindex(calendar).fillna(0.0)
        count_pivots[scenario] = c.reindex(calendar).fillna(0).astype(int)
    return pivots["nominal"], pivots["two_ticks_per_side"], count_pivots["nominal"], count_pivots["two_ticks_per_side"]


def read_prices(path: Path) -> pd.DataFrame:
    start = pd.Timestamp(START, tz="America/New_York")
    end = pd.Timestamp(END + pd.Timedelta(days=1), tz="America/New_York")
    frame = pd.read_parquet(path, columns=["open", "high", "low", "close"], filters=[("timestamp", ">=", start), ("timestamp", "<", end)])
    frame.index = frame.index.tz_convert("UTC").floor("min")
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def six_intraday_paths(
    timeline: pd.DatetimeIndex,
    trades: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = {"MNQ": read_prices(MNQ_DATA), "MGC": read_prices(MGC_DATA)}
    normal = pd.DataFrame(0.0, index=timeline, columns=OLD)
    adverse = pd.DataFrame(0.0, index=timeline, columns=OLD)
    extra_cost = pd.DataFrame(0.0, index=timeline, columns=OLD)
    timeline_dates = pd.Series(timeline.tz_convert("America/New_York").date, index=timeline)
    for sleeve in OLD:
        instrument = INSTRUMENT[sleeve]
        px = prices[instrument].reindex(timeline).groupby(timeline_dates, sort=False).ffill()
        pv = POINT_VALUE[instrument]
        for day, group in trades[sleeve].groupby(pd.to_datetime(trades[sleeve]["session_date"]).dt.date, sort=True):
            day_idx = timeline[timeline_dates.eq(day).to_numpy()]
            if not len(day_idx):
                continue
            base = normal.loc[day_idx, sleeve].to_numpy(copy=True)
            bad = adverse.loc[day_idx, sleeve].to_numpy(copy=True)
            cost = extra_cost.loc[day_idx, sleeve].to_numpy(copy=True)
            for row in group.itertuples(index=False):
                entry = pd.Timestamp(row.entry_ts_utc).floor("min")
                exit_ = pd.Timestamp(row.exit_ts_utc).floor("min")
                start_i = int(day_idx.searchsorted(entry, side="left"))
                end_i = int(day_idx.searchsorted(exit_, side="left"))
                direction = float(row.signed_quantity) / float(row.quantity)
                quantity = float(row.quantity)
                entry_price = float(row.entry_price)
                total_fees = float(row.fees) if hasattr(row, "fees") else float(row.commissions_usd)
                entry_fee = total_fees / 2.0
                if end_i > start_i:
                    marks = px.loc[day_idx[start_i:end_i], "close"].to_numpy(float)
                    adverse_marks = px.loc[day_idx[start_i:end_i], "low" if direction > 0 else "high"].to_numpy(float)
                    open_pnl = (marks - entry_price) * direction * quantity * pv - entry_fee
                    adverse_pnl = (adverse_marks - entry_price) * direction * quantity * pv - entry_fee
                    base[start_i:end_i] += open_pnl
                    bad[start_i:end_i] += adverse_pnl
                if end_i < len(day_idx):
                    net = float(row.net_pnl_usd)
                    base[end_i:] += net
                    bad[end_i:] += net
                # One additional adverse tick at each filled side in the cost stress.
                if start_i < len(day_idx):
                    cost[start_i:] += quantity * TICK_VALUE[instrument]
                if end_i < len(day_idx):
                    cost[end_i:] += quantity * TICK_VALUE[instrument]
            normal.loc[day_idx, sleeve] = base
            adverse.loc[day_idx, sleeve] = bad
            extra_cost.loc[day_idx, sleeve] = cost
    return normal, adverse, extra_cost


def load_mbt_minute(timeline: pd.DatetimeIndex, label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(MBT_DIR / "minute_ensemble_results.csv").set_index("timestamp")
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw.index, utc=True))
    map_base = {
        "rolling_range_breakout": NEW[0],
        "trend_pullback_ma_structure": NEW[1],
        "volatility_breakout": NEW[2],
        "rolling_trend_cluster_cap": "mbt_rolling_trend_cluster_cap",
    }
    positions, normal, adverse = {}, {}, {}
    for source, target in map_base.items():
        positions[target] = raw[f"position__{source}__{label}"].reindex(timeline).fillna(0.0)
        normal[target] = raw[f"equity_from_day_start__{source}__{label}"].reindex(timeline)
        adverse[target] = raw[f"adverse_equity_from_day_start__{source}__{label}"].reindex(timeline)
    normal_frame = pd.DataFrame(normal).groupby(pd.Series(timeline.tz_convert("America/New_York").date, index=timeline), sort=False).ffill().fillna(0.0)
    adverse_frame = pd.DataFrame(adverse).groupby(pd.Series(timeline.tz_convert("America/New_York").date, index=timeline), sort=False).ffill().fillna(0.0)
    return pd.DataFrame(positions), normal_frame, adverse_frame


def integer_equal_risk(vol: pd.Series) -> dict[str, int]:
    anchor = float(vol.max())
    # Round half up; at least one copy, hard guard of ten copies per sleeve.
    return {name: min(10, max(1, int(math.floor(anchor / float(value) + 0.5)))) for name, value in vol.items()}


def build_mappings(nominal: pd.DataFrame) -> dict[str, dict[str, Any]]:
    equal6 = {name: 1 for name in OLD}
    equal9 = {name: 1 for name in ALL}
    risk6 = integer_equal_risk(nominal[OLD].std(ddof=1))
    risk9 = integer_equal_risk(nominal[ALL].std(ddof=1))
    return {
        "equal_contract": {"six": equal6, "nine": equal9},
        "equal_risk_integer": {"six": risk6, "nine": risk9},
        "mbt_cluster_cap": {
            "six": equal6,
            "nine": {**equal6, "mbt_rolling_trend_cluster_cap": 1, NEW[2]: 1},
        },
    }


def portfolio_daily(matrix: pd.DataFrame, mapping: dict[str, int]) -> pd.Series:
    return sum(matrix[name] * copies for name, copies in mapping.items()).rename("daily_pnl_usd")


def portfolio_path(matrix: pd.DataFrame, mapping: dict[str, int]) -> pd.Series:
    return sum(matrix[name] * copies for name, copies in mapping.items()).rename("intraday_pnl_usd")


def apply_vol_stress(series: pd.Series) -> pd.Series:
    return series.where(series >= 0.0, series * LOSS_VOL_STRESS)


def max_drawdown_from_paths(daily: pd.Series, adverse_path: pd.Series, timeline_dates: pd.Series) -> float:
    prior = daily.cumsum().shift(1).fillna(0.0)
    stitched = adverse_path + timeline_dates.map(prior).to_numpy(float)
    peaks = stitched.cummax().clip(lower=0.0)
    return float((stitched - peaks).min())


def loss_cluster_metrics(daily: pd.Series, adverse_path: pd.Series, timeline_dates: pd.Series) -> dict[str, Any]:
    losing = daily.lt(0).astype(int)
    groups = losing.ne(losing.shift()).cumsum()
    longest = int(losing.groupby(groups).sum().max()) if len(losing) else 0
    return {
        "net_pnl_usd": float(daily.sum()),
        "daily_sharpe": float(daily.mean() / daily.std(ddof=1) * math.sqrt(252.0)) if daily.std(ddof=1) > 0 else np.nan,
        "worst_day_usd": float(daily.min()),
        "worst_5_days_usd": float(daily.nsmallest(5).sum()),
        "longest_losing_day_streak": longest,
        "max_intraday_drawdown_usd": max_drawdown_from_paths(daily, adverse_path, timeline_dates),
        "worst_intraday_pnl_from_day_start_usd": float(adverse_path.min()),
    }


def simulate_path(daily: np.ndarray, intraday_min: np.ndarray, traded: np.ndarray, horizon: int) -> dict[str, Any]:
    equity = STARTING_BALANCE
    floor = STARTING_BALANCE - MLL_DISTANCE
    best_day = 0.0
    traded_days = 0
    max_drawdown = 0.0
    peak = equity
    outcome = "EXPIRY"
    days_to_pass = np.nan
    used = 0
    for idx, pnl in enumerate(daily):
        used = idx + 1
        if bool(traded[idx]):
            traded_days += 1
        max_drawdown = min(max_drawdown, equity + float(intraday_min[idx]) - peak)
        if equity + float(intraday_min[idx]) <= floor:
            outcome = "MLL_FAIL"
            break
        equity += float(pnl)
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
        best_day = max(best_day, max(float(pnl), 0.0))
        cumulative = equity - STARTING_BALANCE
        if cumulative >= PROFIT_TARGET and best_day <= CONSISTENCY * cumulative:
            outcome = "PASS"
            days_to_pass = float(traded_days)
            break
        # Current Trading Combine: MLL updates after the day, never falls, and locks at start balance.
        floor = max(floor, min(STARTING_BALANCE, equity - MLL_DISTANCE))
        if traded_days >= horizon:
            outcome = "EXPIRY"
            break
    return {
        "outcome": outcome,
        "days_to_pass": days_to_pass,
        "terminal_pnl_usd": equity - STARTING_BALANCE,
        "max_drawdown_usd": max_drawdown,
        "traded_days": traded_days,
        "calendar_sessions": used,
    }


def valid_starts(traded: np.ndarray, horizon: int) -> np.ndarray:
    remaining = np.cumsum(traded[::-1])[::-1]
    return np.flatnonzero(remaining >= horizon)


def bootstrap_paths(traded: np.ndarray, horizon: int, block: int, rng: np.random.Generator) -> list[np.ndarray]:
    result = []
    for _ in range(PATHS_PER_BLOCK):
        sampled: list[int] = []
        count = 0
        while count < horizon:
            start = int(rng.integers(0, len(traded) - block + 1))
            for idx in range(start, start + block):
                sampled.append(idx)
                count += int(traded[idx])
                if count >= horizon:
                    break
        result.append(np.asarray(sampled, dtype=int))
    return result


def summarize(runs: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in runs.groupby(group, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group, keys))
        passed = part[part.outcome.eq("PASS")]
        row.update({
            "paths": len(part),
            "pass_probability": float(part.outcome.eq("PASS").mean()),
            "mll_probability": float(part.outcome.eq("MLL_FAIL").mean()),
            "expiry_probability": float(part.outcome.eq("EXPIRY").mean()),
            "median_days_to_pass": float(passed.days_to_pass.median()) if len(passed) else np.nan,
            "median_max_drawdown_usd": float(part.max_drawdown_usd.median()),
            "p05_max_drawdown_usd": float(part.max_drawdown_usd.quantile(0.05)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def topstep_campaign(variants: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []
    rng = np.random.default_rng(SEED)
    for horizon in HORIZONS:
        baseline_traded = variants["equal_contract__nominal__six"]["traded"]
        for start in valid_starts(baseline_traded, horizon):
            for name, values in variants.items():
                result = simulate_path(values["daily"][start:], values["intraday_min"][start:], values["traded"][start:], horizon)
                run_rows.append({"method": "historical_rolling_start", "block": np.nan, "path_id": int(start), "scenario": name, "horizon": horizon, **result})
        for block in BLOCKS:
            paths = bootstrap_paths(baseline_traded, horizon, block, rng)
            for path_id, indices in enumerate(paths, start=1):
                for name, values in variants.items():
                    result = simulate_path(values["daily"][indices], values["intraday_min"][indices], values["traded"][indices], horizon)
                    run_rows.append({"method": "moving_block_bootstrap", "block": block, "path_id": path_id, "scenario": name, "horizon": horizon, **result})
    runs = pd.DataFrame(run_rows)
    detail = summarize(runs, ["method", "scenario", "horizon", "block"])
    pooled = summarize(runs[runs.method.eq("moving_block_bootstrap")], ["method", "scenario", "horizon"])
    return runs, pd.concat([detail, pooled.assign(block="pooled")], ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    six_pnl, six_counts, six_trades = load_six()
    calendar = six_pnl.index
    mbt_nominal, mbt_cost, mbt_counts, mbt_cost_counts = load_mbt(calendar)
    nominal = six_pnl.join(mbt_nominal[NEW], how="left").fillna(0.0)
    nominal["mbt_rolling_trend_cluster_cap"] = mbt_nominal["mbt_rolling_trend_cluster_cap"]

    # Cost stress adds one adverse tick per filled side to the six frozen ledgers.
    extra_daily = pd.DataFrame(0.0, index=calendar, columns=OLD)
    for sleeve, trades in six_trades.items():
        dates = pd.to_datetime(trades.session_date)
        sides = pd.Series(2.0 * trades.quantity.to_numpy(float), index=dates).groupby(level=0).sum()
        extra_daily[sleeve] = sides.reindex(calendar).fillna(0.0) * TICK_VALUE[INSTRUMENT[sleeve]]
    cost_matrix = nominal.copy()
    cost_matrix[OLD] = six_pnl - extra_daily
    cost_matrix[NEW + ["mbt_rolling_trend_cluster_cap"]] = mbt_cost[NEW + ["mbt_rolling_trend_cluster_cap"]]

    timeline_source = pd.read_csv(SIX_DIR / "common_timestamp_portfolio.csv", usecols=["timestamp_utc", *[f"position__{name}" for name in OLD]], parse_dates=["timestamp_utc"])
    timeline_source = timeline_source.set_index("timestamp_utc")
    timeline = pd.DatetimeIndex(timeline_source.index).tz_convert("UTC")
    timeline_dates = pd.Series(pd.to_datetime(timeline.tz_convert("America/New_York").date), index=timeline)
    six_path, six_adverse, six_extra = six_intraday_paths(timeline, six_trades)
    mbt_pos, mbt_path, mbt_adverse = load_mbt_minute(timeline, "nominal")
    _, mbt_cost_path, mbt_cost_adverse = load_mbt_minute(timeline, "two_ticks_per_side")
    # Exact EOD reconciliation gates: intraday ledgers must reproduce daily PnL.
    last_mask = ~timeline_dates.duplicated(keep="last")
    last_dates = pd.DatetimeIndex(timeline_dates.loc[last_mask].to_numpy())
    six_eod = six_path.loc[last_mask].set_axis(last_dates)
    six_error = float((six_eod.reindex(calendar)[OLD] - six_pnl).abs().max().max())
    if six_error > 1e-7:
        raise RuntimeError(f"Six-sleeve intraday/daily reconciliation failed: {six_error}")
    mbt_eod = mbt_path.loc[last_mask].set_axis(last_dates)
    mbt_error = float((mbt_eod.reindex(calendar).fillna(0.0)[NEW] - mbt_nominal[NEW]).abs().max().max())
    if mbt_error > 1e-7:
        raise RuntimeError(f"MBT intraday/daily reconciliation failed: {mbt_error}")
    normal_path = six_path.join(mbt_path, how="left").fillna(0.0)
    normal_adverse = six_adverse.join(mbt_adverse, how="left").fillna(0.0)
    cost_path = (six_path - six_extra).join(mbt_cost_path, how="left").fillna(0.0)
    cost_adverse = (six_adverse - six_extra).join(mbt_cost_adverse, how="left").fillna(0.0)

    positions = timeline_source.rename(columns={f"position__{name}": name for name in OLD}).join(mbt_pos, how="left").fillna(0.0)
    mappings = build_mappings(nominal)
    mapping_rows = []
    for family, books in mappings.items():
        for book, mapping in books.items():
            for sleeve in sorted(set(ALL + ["mbt_rolling_trend_cluster_cap"])):
                mapping_rows.append({"allocation_family": family, "book": book, "sleeve": sleeve, "integer_copies": int(mapping.get(sleeve, 0))})
    pd.DataFrame(mapping_rows).to_csv(OUT / "integer_mapping_table.csv", index=False)

    variants: dict[str, dict[str, Any]] = {}
    risk_rows = []
    exposure_rows = []
    for family, books in mappings.items():
        for book, mapping in books.items():
            counts = six_counts.copy()
            for column in NEW:
                counts[column] = mbt_counts[column]
            if "mbt_rolling_trend_cluster_cap" in mapping:
                counts["mbt_rolling_trend_cluster_cap"] = mbt_counts["mbt_rolling_trend_cluster_cap"]
            traded = counts[list(mapping)].gt(0).any(axis=1).to_numpy(bool)
            for stress, matrix, paths, adverse_paths in (
                ("nominal", nominal, normal_path, normal_adverse),
                ("cost_2ticks", cost_matrix, cost_path, cost_adverse),
            ):
                daily_base = portfolio_daily(matrix, mapping)
                path_base = portfolio_path(paths, mapping)
                adverse_base = portfolio_path(adverse_paths, mapping)
                components = matrix[list(mapping)].mul(pd.Series(mapping), axis=1)
                losing_sleeves = components.lt(0).sum(axis=1)
                gross_negative = components.clip(upper=0).sum(axis=1)
                for vol_label in ("base_vol", "loss_vol_125pct"):
                    daily = daily_base if vol_label == "base_vol" else apply_vol_stress(daily_base)
                    path = path_base if vol_label == "base_vol" else apply_vol_stress(path_base)
                    adverse = adverse_base if vol_label == "base_vol" else apply_vol_stress(adverse_base)
                    scenario = f"{family}__{stress}{'__vol125' if vol_label != 'base_vol' else ''}__{book}"
                    intraday_min = adverse.groupby(timeline_dates, sort=False).min().reindex(calendar).fillna(0.0)
                    variants[scenario] = {"daily": daily.to_numpy(float), "intraday_min": intraday_min.to_numpy(float), "traded": traded}
                    portfolio_loss = daily.lt(0)
                    risk_rows.append({
                        "scenario": scenario,
                        **loss_cluster_metrics(daily, adverse, timeline_dates),
                        "max_simultaneous_losing_sleeves": int(losing_sleeves.max()),
                        "mean_losing_sleeves_on_portfolio_loss_days": float(losing_sleeves.loc[portfolio_loss].mean()),
                        "worst_same_day_gross_negative_contribution_usd": float(gross_negative.min()),
                    })

            pos = positions[list(mapping)].mul(pd.Series(mapping), axis=1)
            by_asset = {asset: pos[[name for name in mapping if INSTRUMENT.get(name, "MBT") == asset]] for asset in ("MNQ", "MGC", "MBT")}
            exposure_rows.append({
                "allocation_family": family,
                "book": book,
                "max_mnq_gross_contracts": float(by_asset["MNQ"].abs().sum(axis=1).max()) if len(by_asset["MNQ"].columns) else 0,
                "max_mgc_gross_contracts": float(by_asset["MGC"].abs().sum(axis=1).max()) if len(by_asset["MGC"].columns) else 0,
                "max_mbt_gross_contracts": float(by_asset["MBT"].abs().sum(axis=1).max()) if len(by_asset["MBT"].columns) else 0,
                "max_total_micro_contracts": float(pos.abs().sum(axis=1).max()),
                "topstep_50k_50_micro_limit_breached": bool(pos.abs().sum(axis=1).max() > 50),
            })

    runs, summary = topstep_campaign(variants)
    runs.to_csv(OUT / "topstep_path_results.csv", index=False)
    summary.to_csv(OUT / "topstep_summary.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(OUT / "drawdown_and_loss_clustering.csv", index=False)
    pd.DataFrame(exposure_rows).to_csv(OUT / "simultaneous_contract_exposure.csv", index=False)
    nominal.to_csv(OUT / "daily_nine_sleeves_nominal.csv", index_label="session_date")
    cost_matrix.to_csv(OUT / "daily_nine_sleeves_cost_stress.csv", index_label="session_date")

    pooled = summary[(summary.method == "moving_block_bootstrap") & (summary.block == "pooled")].copy()
    decisions = []
    for family in mappings:
        for horizon in HORIZONS:
            base = pooled[(pooled.scenario == f"{family}__nominal__six") & (pooled.horizon == horizon)].iloc[0]
            add = pooled[(pooled.scenario == f"{family}__nominal__nine") & (pooled.horizon == horizon)].iloc[0]
            decisions.append({
                "allocation_family": family,
                "horizon": horizon,
                "pass_delta": float(add.pass_probability - base.pass_probability),
                "mll_delta": float(add.mll_probability - base.mll_probability),
                "expiry_delta": float(add.expiry_probability - base.expiry_probability),
                "drawdown_delta_usd": float(add.median_max_drawdown_usd - base.median_max_drawdown_usd),
            })
    decision_frame = pd.DataFrame(decisions)
    decision_frame.to_csv(OUT / "six_vs_nine_deltas.csv", index=False)
    nominal60 = decision_frame[decision_frame.horizon.eq(60)]
    if (nominal60.pass_delta > 0.02).all() and (nominal60.mll_delta < 0).all():
        verdict = "IMPROVES_TOPSTEP_COMPATIBILITY"
    elif (nominal60.pass_delta < -0.02).any() or (nominal60.mll_delta > 0.02).any():
        verdict = "DEGRADES_TOPSTEP_COMPATIBILITY"
    else:
        verdict = "NEUTRAL"
    metadata = {
        "run_date": "2026-10-04",
        "verdict": verdict,
        "verdict_policy": "predeclared: improves only if every nominal 60d allocation family gains >2pp PASS and lowers MLL; degrades if any loses >2pp PASS or adds >2pp MLL; otherwise neutral",
        "rules": {
            "account": "Topstep Trading Combine 50K",
            "profit_target_usd": PROFIT_TARGET,
            "maximum_loss_limit_usd": MLL_DISTANCE,
            "mll_monitoring": "intraday one-minute conservative adverse OHLC mark; floor updates EOD and locks at starting balance",
            "consistency_target": CONSISTENCY,
            "horizons_traded_days": list(HORIZONS),
            "max_micro_contracts": 50,
        },
        "stress_definitions": {
            "cost_2ticks": "one additional adverse tick per filled side for every sleeve",
            "loss_vol_125pct": "multiply negative portfolio daily and intraday PnL by 1.25; positive PnL unchanged",
        },
        "bootstrap": {"seed": SEED, "blocks": list(BLOCKS), "paths_per_block": PATHS_PER_BLOCK},
        "input_hashes": {
            "six_daily": sha256(SIX_DIR / "daily_sleeve_pnl.csv"),
            "six_timestamp": sha256(SIX_DIR / "common_timestamp_portfolio.csv"),
            "mbt_ensemble_manifest": sha256(MBT_DIR / "ensemble_manifest.json"),
        },
        "frozen_strategies_modified": False,
        "pass_rate_optimization": False,
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(pd.DataFrame(mapping_rows).to_string(index=False))
    print(pooled.to_string(index=False))
    print(decision_frame.to_string(index=False))


if __name__ == "__main__":
    main()
