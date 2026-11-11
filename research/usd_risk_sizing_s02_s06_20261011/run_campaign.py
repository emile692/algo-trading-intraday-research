"""Isolated USD-risk sizing audit for frozen sleeves S02-S06."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CFG_PATH = OUT / "config.json"
SIX = ROOT / "research" / "portfolio_integration_phase1"
NINE = ROOT / "research" / "mbt_nine_sleeve_topstep_20261004"
MBT = ROOT / "research" / "mbt_three_sleeve_freeze_20261004" / "vote_ensemble"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from research.baseline_backtests.baseline_engine import (  # noqa: E402
    SIGNAL_BUILDERS,
    load_frozen_specs,
    prepare_data,
)
from research.locked_four_sleeve_portfolio.run_locked_four_sleeve_portfolio import (  # noqa: E402
    apply_alpha_parameters,
    load_and_validate_config,
)
from research.mgc_baseline_backtests.run_mgc_baselines import prepare_mgc_data  # noqa: E402
from research.mbt_nine_sleeve_topstep_20261004.run_topstep import (  # noqa: E402
    load_mbt,
    load_mbt_minute,
    six_intraday_paths,
)


OLD = [
    "mnq_orb_vwap_atr", "mnq_volatility_breakout", "mnq_compression_breakout",
    "mnq_atr_trend_flip", "mgc_volatility_breakout", "mgc_gap_continuation",
]
SIZED = OLD[1:]
NEW = ["mbt_rolling_range_breakout", "mbt_trend_pullback_ma_structure", "mbt_volatility_breakout"]
INSTRUMENT = {name: name.split("_")[0].upper() for name in OLD + NEW}
STARTING_BALANCE = 50_000.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config() -> dict[str, Any]:
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def load_trades() -> dict[str, pd.DataFrame]:
    result = {}
    for sleeve in OLD:
        frame = pd.read_csv(SIX / "sleeve_trades" / f"{sleeve}.csv")
        frame["session_date"] = pd.to_datetime(frame["session_date"])
        frame["entry_ts_utc"] = pd.to_datetime(frame["entry_ts_utc"], utc=True)
        frame["exit_ts_utc"] = pd.to_datetime(frame["exit_ts_utc"], utc=True)
        result[sleeve] = frame
    return result


def instruction_maps() -> dict[str, dict[str, Any]]:
    """Return frozen order instructions; levels exist before the fill."""
    mnq_cfg, _, _, _ = load_and_validate_config()
    specs, _, _ = load_frozen_specs(
        ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines",
        ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv",
    )
    by_id = {x["mechanism_id"]: x for x in specs}
    mnq_data = prepare_data(
        ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet",
        start_date=pd.Timestamp("2019-05-06").date(), end_date=pd.Timestamp("2026-03-19").date(),
    )
    maps: dict[str, dict[str, Any]] = {}
    for sleeve, source in ((OLD[1], "volatility_breakout"), (OLD[2], "compression_breakout")):
        spec = apply_alpha_parameters(by_id[source], mnq_cfg["sleeves"][source]["alpha_parameters"])
        entries = SIGNAL_BUILDERS[source](spec, mnq_data).entries
        maps[sleeve] = {e.setup_id: e for e in entries}

    mgc_data, _ = prepare_mgc_data(
        ROOT / "data" / "processed" / "parquet" / "MGC_v_0_1m_20150101_20260923.parquet",
        pd.Timestamp("2015-01-01").date(), pd.Timestamp("2026-03-19").date(),
    )
    spec = json.loads((ROOT / "research" / "mgc_final_two_sleeve_lock_20260926" / "frozen_candidate_specs" / "volatility_breakout.json").read_text(encoding="utf-8"))
    entries = SIGNAL_BUILDERS["volatility_breakout"](spec, mgc_data).entries
    maps[OLD[4]] = {e.setup_id: e for e in entries}
    maps["_minute_close_mnq"] = mnq_data.minute["close"].to_dict()
    maps["_minute_close_mgc"] = mgc_data.minute["close"].to_dict()
    return maps


def add_causal_risk_fields(trades: dict[str, pd.DataFrame], maps: dict[str, dict[str, Any]], cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for sleeve in SIZED:
        d = trades[sleeve].copy()
        inst = INSTRUMENT[sleeve]
        ex = cfg["execution"][inst]
        direction = np.sign(pd.to_numeric(d["signed_quantity"], errors="raise"))
        if sleeve in (OLD[1], OLD[2], OLD[4]):
            planned, structural = [], []
            for row, side in zip(d.itertuples(index=False), direction):
                ins = maps[sleeve][str(row.setup_id)]
                planned.append(float(ins.long_level if side > 0 else ins.short_level))
                structural.append(float(ins.long_stop if side > 0 else ins.short_stop))
            d["planned_entry_price"] = planned
            d["structural_stop_price"] = structural
        elif sleeve == OLD[3]:
            # The locked stop distance is ATR(14), known from the completed 15m bar.
            d["structural_stop_price"] = pd.to_numeric(d["initial_stop_price"], errors="raise")
            distance = (pd.to_numeric(d["entry_price"], errors="raise") - d["structural_stop_price"]).abs()
            d["planned_entry_price"] = d["structural_stop_price"] + direction * distance
        else:
            d["structural_stop_price"] = pd.to_numeric(d["initial_stop_price"], errors="raise")
            # S06 is a next-open market order. Use the last observable minute close (08:34), not the future 08:35 open.
            lookup = maps["_minute_close_mgc"]
            d["planned_entry_price"] = [float(lookup[pd.Timestamp(ts) - pd.Timedelta(minutes=1)]) for ts in d["entry_ts_utc"]]
        d["point_value"] = float(ex["point_value"])
        d["round_trip_commission"] = float(ex["round_trip_commission"])
        d["execution_allowance"] = 2.0 * float(ex["nominal_slippage_ticks_per_side"]) * float(ex["tick_value"])
        d["risk_per_contract_planned_usd"] = (
            (d["planned_entry_price"] - d["structural_stop_price"]).abs() * d["point_value"]
            + d["round_trip_commission"] + d["execution_allowance"]
        )
        d["risk_per_contract_fill_usd"] = (
            (pd.to_numeric(d["entry_price"], errors="raise") - d["structural_stop_price"]).abs() * d["point_value"]
            + d["round_trip_commission"] + d["execution_allowance"]
        )
        d["sleeve"] = sleeve
        rows.append(d)
    return pd.concat(rows, ignore_index=True, sort=False)


def size_and_reprice(all_risk: pd.DataFrame, original: dict[str, pd.DataFrame], budget: int, stress: int, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    output = {OLD[0]: original[OLD[0]].copy()}
    # S01 is unchanged, but book stress adds only the requested extra ticks.
    s01 = output[OLD[0]]
    q01 = pd.to_numeric(s01["quantity"], errors="raise")
    s01["net_pnl_usd"] = pd.to_numeric(s01["net_pnl_usd"], errors="raise") - 2.0 * stress * cfg["execution"]["MNQ"]["tick_value"] * q01
    for sleeve in SIZED:
        d = all_risk[all_risk["sleeve"].eq(sleeve)].copy()
        q = np.floor(float(budget) / d["risk_per_contract_planned_usd"]).astype(int)
        d["risk_budget_usd"] = budget
        d["desired_quantity"] = q
        d["quantity"] = q
        d["signed_quantity"] = np.sign(pd.to_numeric(d["signed_quantity"], errors="raise")) * q
        d["skipped"] = q.eq(0)
        d["actual_planned_risk_usd"] = q * d["risk_per_contract_planned_usd"]
        d["actual_fill_risk_usd"] = q * d["risk_per_contract_fill_usd"]
        d = d.loc[q.gt(0)].copy()
        inst = INSTRUMENT[sleeve]
        ex = cfg["execution"][inst]
        side = np.sign(pd.to_numeric(d["signed_quantity"], errors="raise"))
        gross_raw = (pd.to_numeric(d["raw_exit_price"], errors="raise") - pd.to_numeric(d["raw_entry_price"], errors="raise")) * side * float(ex["point_value"]) * d["quantity"]
        slip = 2.0 * (float(ex["nominal_slippage_ticks_per_side"]) + stress) * float(ex["tick_value"]) * d["quantity"]
        commissions = float(ex["round_trip_commission"]) * d["quantity"]
        d["gross_pnl_usd"] = gross_raw - slip
        d["commissions_usd"] = commissions
        d["slippage_usd"] = slip
        d["net_pnl_usd"] = gross_raw - slip - commissions
        d["fees"] = commissions
        output[sleeve] = d
    return output


def daily_from_trades(trades: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    pnl = pd.DataFrame(0.0, index=calendar, columns=OLD)
    counts = pd.DataFrame(0, index=calendar, columns=OLD, dtype=int)
    for sleeve, d in trades.items():
        dates = pd.to_datetime(d["session_date"])
        pnl[sleeve] = pd.Series(d["net_pnl_usd"].to_numpy(float), index=dates).groupby(level=0).sum().reindex(calendar).fillna(0.0)
        counts[sleeve] = pd.Series(1, index=dates).groupby(level=0).sum().reindex(calendar).fillna(0).astype(int)
    return pnl, counts


def metrics(pnl: pd.Series) -> dict[str, float]:
    equity = STARTING_BALANCE + pnl.cumsum()
    dd = equity - equity.cummax().clip(lower=STARTING_BALANCE)
    gains, losses = pnl.clip(lower=0).sum(), -pnl.clip(upper=0).sum()
    return {
        "net_pnl_usd": float(pnl.sum()),
        "sharpe": float(pnl.mean() / pnl.std(ddof=1) * math.sqrt(252)) if pnl.std(ddof=1) else np.nan,
        "profit_factor": float(gains / losses) if losses else np.nan,
        "max_drawdown_usd": float(dd.min()), "worst_day_usd": float(pnl.min()),
        "worst_5_days_usd": float(pnl.nsmallest(5).sum()),
        "p01_day_usd": float(pnl.quantile(0.01)),
    }


def simulate_path(daily: np.ndarray, intraday_min: np.ndarray, traded: np.ndarray, horizon: int, cfg: dict[str, Any]) -> dict[str, Any]:
    rules = cfg["topstep"]
    equity, peak = rules["starting_balance"], rules["starting_balance"]
    floor = equity - rules["mll_distance"]
    best_day, traded_days, max_dd = 0.0, 0, 0.0
    outcome, days_to_pass = "EXPIRY", np.nan
    for idx, value in enumerate(daily):
        if traded[idx]: traded_days += 1
        max_dd = min(max_dd, equity + float(intraday_min[idx]) - peak)
        if equity + float(intraday_min[idx]) <= floor:
            outcome = "MLL"; break
        equity += float(value); peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
        best_day = max(best_day, float(value), 0.0)
        cumulative = equity - rules["starting_balance"]
        if cumulative >= rules["profit_target"] and best_day <= rules["consistency_share"] * cumulative:
            outcome, days_to_pass = "PASS", float(traded_days); break
        floor = max(floor, min(rules["starting_balance"], equity - rules["mll_distance"]))
        if traded_days >= horizon: break
    return {"outcome": outcome, "days_to_pass": days_to_pass, "max_drawdown_usd": max_dd}


def valid_starts(traded: np.ndarray, horizon: int) -> np.ndarray:
    return np.flatnonzero(np.cumsum(traded[::-1])[::-1] >= horizon)


def sampled_paths(traded: np.ndarray, horizon: int, block: int, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    paths = []
    for _ in range(n):
        selected, count = [], 0
        while count < horizon:
            start = int(rng.integers(0, len(traded) - block + 1))
            for idx in range(start, start + block):
                selected.append(idx); count += int(traded[idx])
                if count >= horizon: break
        paths.append(np.asarray(selected))
    return paths


def topstep(variants: dict[str, dict[str, np.ndarray]], cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = variants["reference__six__slip0"]["traded"]
    rows = []
    rng = np.random.default_rng(int(cfg["topstep"]["seed"]))
    for horizon in cfg["topstep"]["horizons"]:
        for start in valid_starts(baseline, horizon):
            for name, v in variants.items():
                result = simulate_path(v["daily"][start:], v["intraday_min"][start:], v["traded"][start:], horizon, cfg)
                rows.append({"method": "historical_rolling_start", "block": np.nan, "path_id": int(start), "scenario": name, "horizon": horizon, **result})
        for block in cfg["topstep"]["blocks"]:
            paths = sampled_paths(baseline, horizon, block, int(cfg["topstep"]["paths_per_block"]), rng)
            for path_id, indices in enumerate(paths):
                for name, v in variants.items():
                    result = simulate_path(v["daily"][indices], v["intraday_min"][indices], v["traded"][indices], horizon, cfg)
                    rows.append({"method": "moving_block_bootstrap", "block": block, "path_id": path_id, "scenario": name, "horizon": horizon, **result})
    runs = pd.DataFrame(rows)
    summary_rows = []
    for keys, part in runs.groupby(["method", "scenario", "horizon"], sort=True):
        passed = part[part.outcome.eq("PASS")]
        summary_rows.append({"method": keys[0], "scenario": keys[1], "horizon": keys[2], "paths": len(part),
            "pass_probability": part.outcome.eq("PASS").mean(), "mll_probability": part.outcome.eq("MLL").mean(),
            "expiry_probability": part.outcome.eq("EXPIRY").mean(), "median_days_to_pass": passed.days_to_pass.median(),
            "median_max_drawdown_usd": part.max_drawdown_usd.median()})
    summary = pd.DataFrame(summary_rows)
    # Paired deltas and block-bootstrap uncertainty: the same path IDs feed reference and variant.
    paired_rows = []
    boot = runs[runs.method.eq("moving_block_bootstrap")].copy()
    for scenario in sorted(set(boot.scenario) - {"reference__six__slip0", "reference__nine__slip0"}):
        book = scenario.split("__")[1]
        ref = f"reference__{book}__slip0"
        for horizon in cfg["topstep"]["horizons"]:
            a = boot[(boot.scenario.eq(scenario)) & boot.horizon.eq(horizon)].sort_values(["block", "path_id"])
            b = boot[(boot.scenario.eq(ref)) & boot.horizon.eq(horizon)].sort_values(["block", "path_id"])
            if len(a) != len(b): continue
            dp = a.outcome.eq("PASS").to_numpy(int) - b.outcome.eq("PASS").to_numpy(int)
            dm = a.outcome.eq("MLL").to_numpy(int) - b.outcome.eq("MLL").to_numpy(int)
            dp_mean, dm_mean = float(dp.mean()), float(dm.mean())
            dp_half = 1.959963984540054 * float(dp.std(ddof=1)) / math.sqrt(len(dp))
            dm_half = 1.959963984540054 * float(dm.std(ddof=1)) / math.sqrt(len(dm))
            paired_rows.append({"scenario": scenario, "reference": ref, "horizon": horizon,
                "paired_pass_delta": dp_mean, "paired_pass_delta_ci_low": dp_mean - dp_half, "paired_pass_delta_ci_high": dp_mean + dp_half,
                "paired_mll_delta": dm_mean, "paired_mll_delta_ci_low": dm_mean - dm_half, "paired_mll_delta_ci_high": dm_mean + dm_half})
    return runs, summary, pd.DataFrame(paired_rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    cfg = config()
    original = load_trades()
    risk = add_causal_risk_fields(original, instruction_maps(), cfg)
    base_daily = pd.read_csv(SIX / "daily_sleeve_pnl.csv", parse_dates=["session_date"]).set_index("session_date")[OLD]
    calendar = base_daily.index
    reproduced, _ = daily_from_trades(original, calendar)
    errors = (reproduced - base_daily).abs().max()
    if float(errors.max()) > 1e-7:
        raise RuntimeError(f"Frozen six-sleeve reproduction failed: {errors.to_dict()}")
    nine_locked = pd.read_csv(ROOT / "research" / "locked_nine_sleeve_portfolio" / "daily_pnl.csv", parse_dates=["session_date"]).set_index("session_date")
    if not np.allclose(nine_locked[OLD].to_numpy(), base_daily.to_numpy(), atol=1e-7):
        raise RuntimeError("Nine-sleeve frozen source does not reproduce six-sleeve ledger")

    mbt_nominal, _, mbt_counts, _ = load_mbt(calendar)
    timeline_source = pd.read_csv(SIX / "common_timestamp_portfolio.csv", usecols=["timestamp_utc", *[f"position__{s}" for s in OLD]], parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
    timeline = pd.DatetimeIndex(timeline_source.index).tz_convert("UTC")
    timeline_dates = pd.Series(pd.to_datetime(timeline.tz_convert("America/New_York").date), index=timeline)
    mbt_pos, mbt_path0, mbt_adv0 = load_mbt_minute(timeline, "nominal")
    _, mbt_path1, mbt_adv1 = load_mbt_minute(timeline, "two_ticks_per_side")
    mbt_extra_path, mbt_extra_adv = mbt_path0 - mbt_path1, mbt_adv0 - mbt_adv1

    variant_records: dict[tuple[str, str, int], tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]] = {}
    sizing_rows, trade_exports, sleeve_metric_rows, yearly_rows, risk_contribution_rows = [], [], [], [], []
    variants: dict[str, dict[str, np.ndarray]] = {}
    path_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    budgets: list[int | None] = [None, *cfg["risk_budgets_usd"]]
    for budget in budgets:
        label = "reference" if budget is None else f"budget_{budget}"
        for stress in cfg["slippage_stress_extra_ticks_per_side"]:
            if budget is None:
                adjusted = {k: v.copy() for k, v in original.items()}
                for sleeve, d in adjusted.items():
                    inst = INSTRUMENT[sleeve]
                    d["net_pnl_usd"] = pd.to_numeric(d["net_pnl_usd"], errors="raise") - 2 * stress * cfg["execution"][inst]["tick_value"] * pd.to_numeric(d["quantity"], errors="raise")
            else:
                adjusted = size_and_reprice(risk, original, budget, stress, cfg)
            six_daily, six_counts = daily_from_trades(adjusted, calendar)
            if stress == 0:
                six_path, six_adv, six_extra = six_intraday_paths(timeline, adjusted)
                path_cache[label] = (six_path, six_adv, six_extra)
            else:
                base_path, base_adv, one_tick_extra = path_cache[label]
                six_path = base_path - stress * one_tick_extra
                six_adv = base_adv - stress * one_tick_extra
            for book in ("six", "nine"):
                daily_matrix = six_daily.copy()
                counts = six_counts.copy()
                path, adv = six_path.copy(), six_adv.copy()
                if book == "nine":
                    daily_matrix = daily_matrix.join(mbt_nominal[NEW], how="left").fillna(0.0)
                    if stress:
                        delta_daily = pd.DataFrame(index=calendar)
                        mbt_stressed_path = mbt_path0 - stress * mbt_extra_path
                        mbt_stressed_adv = mbt_adv0 - stress * mbt_extra_adv
                        last = ~timeline_dates.duplicated(keep="last")
                        eod = mbt_stressed_path.loc[last].set_axis(pd.DatetimeIndex(timeline_dates.loc[last].to_numpy())).reindex(calendar).fillna(0.0)
                        daily_matrix[NEW] = eod[NEW]
                        path, adv = path.join(mbt_stressed_path[NEW], how="left").fillna(0.0), adv.join(mbt_stressed_adv[NEW], how="left").fillna(0.0)
                    else:
                        path, adv = path.join(mbt_path0[NEW], how="left").fillna(0.0), adv.join(mbt_adv0[NEW], how="left").fillna(0.0)
                    counts = counts.join(mbt_counts[NEW], how="left").fillna(0).astype(int)
                portfolio = daily_matrix.sum(axis=1)
                portfolio_path, portfolio_adv = path.sum(axis=1), adv.sum(axis=1)
                intraday_min = portfolio_adv.groupby(timeline_dates, sort=False).min().reindex(calendar).fillna(0.0)
                traded = counts.gt(0).any(axis=1).to_numpy(bool)
                scenario = f"{label}__{book}__slip{stress}"
                variants[scenario] = {"daily": portfolio.to_numpy(float), "intraday_min": intraday_min.to_numpy(float), "traded": traded}
                sleeve_metric_rows.append({"scenario": scenario, "scope": "portfolio", **metrics(portfolio)})
                for sleeve in daily_matrix:
                    sleeve_metric_rows.append({"scenario": scenario, "scope": sleeve, **metrics(daily_matrix[sleeve])})
                for year in sorted(calendar.year.unique()):
                    part = portfolio[calendar.year == year]
                    yearly_rows.append({"scenario": scenario, "scope": "portfolio", "year": int(year), **metrics(part)})
                    for sleeve in daily_matrix:
                        yearly_rows.append({"scenario": scenario, "scope": sleeve, "year": int(year), **metrics(daily_matrix.loc[calendar.year == year, sleeve])})
                if stress == 0:
                    portfolio_var = float(portfolio.var(ddof=1))
                    for sleeve in daily_matrix:
                        component = float(daily_matrix[sleeve].cov(portfolio))
                        risk_contribution_rows.append({"scenario": scenario, "sleeve": sleeve, "instrument": INSTRUMENT[sleeve],
                            "component_daily_variance": component, "risk_contribution_pct": 100.0 * component / portfolio_var if portfolio_var else np.nan})
                variant_records[(label, book, stress)] = (daily_matrix, counts, adjusted)
            if budget is not None and stress == 0:
                for sleeve in SIZED:
                    source = risk[risk.sleeve.eq(sleeve)].copy()
                    q = np.floor(budget / source.risk_per_contract_planned_usd).astype(int)
                    vc = q.value_counts().sort_index()
                    executed = adjusted[sleeve]
                    losses = -pd.to_numeric(executed.loc[pd.to_numeric(executed["net_pnl_usd"], errors="raise").lt(0), "net_pnl_usd"], errors="raise")
                    exceed = (-pd.to_numeric(executed["net_pnl_usd"], errors="raise") > pd.to_numeric(executed["actual_planned_risk_usd"], errors="raise"))
                    sizing_rows.append({"sleeve": sleeve, "budget_usd": budget, "signals": len(source), "executed": int(q.gt(0).sum()), "skipped": int(q.eq(0).sum()),
                        "quantity_mean_executed": float(q[q.gt(0)].mean()), "quantity_max": int(q.max()), "quantity_distribution": ";".join(f"{int(k)}:{int(v)}" for k,v in vc.items()),
                        "planned_risk_mean_usd": float((q * source.risk_per_contract_planned_usd)[q.gt(0)].mean()),
                        "fill_risk_mean_usd": float((q * source.risk_per_contract_fill_usd)[q.gt(0)].mean()),
                        "fill_risk_over_budget_count": int(((q * source.risk_per_contract_fill_usd) > budget).sum()),
                        "realized_losing_trades": int(len(losses)), "realized_loss_mean_usd": float(losses.mean()) if len(losses) else 0.0,
                        "realized_loss_p95_usd": float(losses.quantile(.95)) if len(losses) else 0.0, "realized_loss_max_usd": float(losses.max()) if len(losses) else 0.0,
                        "realized_loss_over_planned_risk_count": int(exceed.sum())})
                exported = pd.concat([v.assign(campaign_variant=label) for k, v in adjusted.items() if k in SIZED], ignore_index=True, sort=False)
                trade_exports.append(exported)

    # Contract exposure from adjusted intervals plus unchanged MBT minute positions.
    exposure_rows = []
    for label in ["reference", *[f"budget_{x}" for x in cfg["risk_budgets_usd"]]]:
        adjusted = variant_records[(label, "six", 0)][2]
        pos = pd.DataFrame(0.0, index=timeline, columns=OLD)
        for sleeve, d in adjusted.items():
            changes = np.zeros(len(timeline) + 1, dtype=float)
            for row in d.itertuples(index=False):
                start_i = int(timeline.searchsorted(pd.Timestamp(row.entry_ts_utc), side="left"))
                end_i = int(timeline.searchsorted(pd.Timestamp(row.exit_ts_utc), side="left"))
                changes[start_i] += float(row.signed_quantity)
                changes[end_i] -= float(row.signed_quantity)
            pos[sleeve] = np.cumsum(changes[:-1])
        for book in ("six", "nine"):
            p = pos.copy()
            if book == "nine": p = p.join(mbt_pos[NEW], how="left").fillna(0.0)
            exposure_rows.append({"variant": label, "book": book,
                "max_total_micro_contracts": float(p.abs().sum(axis=1).max()),
                "max_mnq_gross_contracts": float(p[[c for c in p if INSTRUMENT[c] == "MNQ"]].abs().sum(axis=1).max()),
                "max_mgc_gross_contracts": float(p[[c for c in p if INSTRUMENT[c] == "MGC"]].abs().sum(axis=1).max()),
                "max_mbt_gross_contracts": float(p[[c for c in p if INSTRUMENT[c] == "MBT"]].abs().sum(axis=1).max()) if book == "nine" else 0.0,
                "limit_breached": bool(p.abs().sum(axis=1).max() > cfg["account"]["max_total_micro_contracts"])})

    runs, top_summary, paired = topstep(variants, cfg)
    sizing = pd.DataFrame(sizing_rows); sleeve_metrics = pd.DataFrame(sleeve_metric_rows); yearly = pd.DataFrame(yearly_rows); exposure = pd.DataFrame(exposure_rows)
    sizing.to_csv(OUT / "sizing_activity.csv", index=False)
    pd.concat(trade_exports, ignore_index=True, sort=False).to_csv(OUT / "resized_trades.csv", index=False)
    sleeve_metrics.to_csv(OUT / "performance_metrics.csv", index=False)
    yearly.to_csv(OUT / "yearly_results.csv", index=False)
    exposure.to_csv(OUT / "exposure_summary.csv", index=False)
    pd.DataFrame(risk_contribution_rows).to_csv(OUT / "risk_contributions.csv", index=False)
    runs.to_csv(OUT / "topstep_paths.csv", index=False)
    top_summary.to_csv(OUT / "topstep_summary.csv", index=False)
    paired.to_csv(OUT / "topstep_paired_deltas.csv", index=False)
    risk[["sleeve", "trade_id", "session_date", "planned_entry_price", "entry_price", "structural_stop_price", "risk_per_contract_planned_usd", "risk_per_contract_fill_usd"]].to_csv(OUT / "risk_audit.csv", index=False)

    # Compact comparison figures.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    view = sleeve_metrics[(sleeve_metrics.scope.eq("portfolio")) & sleeve_metrics.scenario.str.endswith("__slip0")].copy()
    view["book"] = view.scenario.str.split("__").str[1]; view["variant"] = view.scenario.str.split("__").str[0]
    for book, part in view.groupby("book"):
        axes[0].plot(part.variant, part.sharpe, marker="o", label=book)
        axes[1].plot(part.variant, part.max_drawdown_usd, marker="o", label=book)
    axes[0].set_title("Sharpe nominal"); axes[1].set_title("Max drawdown nominal (USD)")
    for ax in axes: ax.tick_params(axis="x", rotation=35); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "figures" / "performance_comparison.png", dpi=170); plt.close(fig)
    boot60 = top_summary[(top_summary.method.eq("moving_block_bootstrap")) & top_summary.horizon.eq(60) & top_summary.scenario.str.endswith("__slip0")].copy()
    fig, ax = plt.subplots(figsize=(10, 5)); ax.scatter(boot60.mll_probability, boot60.pass_probability)
    for r in boot60.itertuples(): ax.annotate(r.scenario.replace("__slip0", ""), (r.mll_probability, r.pass_probability), fontsize=8)
    ax.set(xlabel="P(MLL)", ylabel="P(PASS)", title="Topstep 60 séances — frontière appariée"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "topstep_frontier_60d.png", dpi=170); plt.close(fig)

    # Mechanical verdict across all budgets: improvement must survive stress and preserve PASS speed.
    ref = sleeve_metrics[(sleeve_metrics.scenario.eq("reference__nine__slip0")) & sleeve_metrics.scope.eq("portfolio")].iloc[0]
    candidates = sleeve_metrics[(sleeve_metrics.scope.eq("portfolio")) & sleeve_metrics.scenario.str.startswith("budget_")]
    dd_better_all = all(candidates.groupby(candidates.scenario.str.split("__").str[0]).max_drawdown_usd.min() > ref.max_drawdown_usd)
    ts60 = top_summary[(top_summary.method.eq("moving_block_bootstrap")) & top_summary.horizon.eq(60) & top_summary.scenario.str.contains("__nine__")]
    ref_pass = float(ts60[ts60.scenario.eq("reference__nine__slip0")].pass_probability.iloc[0])
    enough_speed = bool((ts60[ts60.scenario.str.startswith("budget_")].groupby(ts60.scenario.str.split("__").str[0]).pass_probability.min() >= ref_pass - .05).all())
    verdict = "ROBUST_IMPROVEMENT" if dd_better_all and enough_speed else ("TRADEOFF" if candidates.net_pnl_usd.max() > 0 else "NO_IMPROVEMENT")
    audit = {"campaign": cfg["campaign_id"], "reference_reproduction_max_abs_error": float(errors.max()),
        "s01_clarification": "S01 source trades carry risk_budget_usd=250 but the active six/nine books include one executable copy of that already-sized sleeve; comparison reference therefore keeps S01 exactly unchanged.",
        "s02_stop_discrepancy": "Published/frozen alpha stop is session open, but the locked 240-minute overlay ledger has blank initial_stop_price. The campaign uses the published session-open level for causal sizing and preserves the frozen ledger exits for reference comparability.",
        "account_limit_breached": bool(exposure.limit_breached.any()), "verdict": verdict,
        "input_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in [CFG_PATH, SIX / "daily_sleeve_pnl.csv", SIX / "common_timestamp_portfolio.csv", NINE / "daily_nine_sleeves_nominal.csv", MBT / "minute_ensemble_results.csv"]}}
    (OUT / "run_metadata.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    write_report(cfg, audit, sizing, sleeve_metrics, top_summary, exposure)
    print(json.dumps(audit, indent=2))


def write_report(cfg: dict[str, Any], audit: dict[str, Any], sizing: pd.DataFrame, perf: pd.DataFrame, top: pd.DataFrame, exposure: pd.DataFrame) -> None:
    nominal = perf[(perf.scope.eq("portfolio")) & perf.scenario.str.endswith("__slip0")].copy()
    t60 = top[(top.method.eq("moving_block_bootstrap")) & top.horizon.eq(60) & top.scenario.str.endswith("__slip0")]
    lines = ["# S02–S06 — sizing par budget de risque USD", "", f"**Verdict : `{audit['verdict']}`.** Campagne isolée; aucune référence gelée n'a été modifiée.", "",
        "## Audit et conventions", "", f"- Reproduction des six ledgers : erreur absolue maximale `{audit['reference_reproduction_max_abs_error']:.3g}` USD.",
        "- S01 est intrinsèquement risk-sized à 250 USD dans son fichier de trades, mais le book actif additionne une copie de cette sleeve déjà dimensionnée. La référence de cette campagne conserve donc S01 telle quelle dans toutes les branches.",
        "- S02 : le stop publié est l'open RTH; l'overlay 240 minutes gelé l'a effacé du ledger. Il sert ici d'ancre de sizing, sans réécrire les sorties gelées. Cette divergence empêche d'interpréter le risque prévu comme une perte garantie.",
        "- Quantité décidée avec le niveau d'ordre connu; les ordres au prochain open utilisent le dernier close observable ou une distance ATR déjà figée. Un gap de fill ne déclenche jamais de rescaling; son dépassement est mesuré dans `risk_audit.csv`.",
        "- Allocation: commission round-trip plus deux ticks nominaux round-trip. Les stress ajoutent 1 ou 2 ticks par côté. Quantité zéro = signal ignoré.",
        "- Les périodes 2024–2026 ont déjà servi aux décisions MNQ/MGC/MBT; aucune revendication de nouvel OOS indépendant.", "",
        "## Activité de sizing", "", "| Sleeve | Budget | Exécutés | Ignorés | Qté moyenne | Qté max | Risque prévu moyen | Risque fill moyen | Fill > budget |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sizing.itertuples():
        lines.append(f"| {r.sleeve} | {r.budget_usd} | {r.executed} | {r.skipped} | {r.quantity_mean_executed:.2f} | {r.quantity_max} | {r.planned_risk_mean_usd:.1f} | {r.fill_risk_mean_usd:.1f} | {r.fill_risk_over_budget_count} |")
    lines += ["", "## Books — nominal", "", "| Scénario | PnL net | Sharpe | PF | maxDD | Pire jour | Pires 5 jours |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in nominal.itertuples(): lines.append(f"| {r.scenario} | {r.net_pnl_usd:,.0f} | {r.sharpe:.2f} | {r.profit_factor:.2f} | {r.max_drawdown_usd:,.0f} | {r.worst_day_usd:,.0f} | {r.worst_5_days_usd:,.0f} |")
    lines += ["", "## Topstep — bootstrap mobile apparié, 60 séances", "", "| Scénario | PASS | MLL | EXPIRY | Jours vers PASS | DD médian |", "|---|---:|---:|---:|---:|---:|"]
    for r in t60.itertuples(): lines.append(f"| {r.scenario} | {r.pass_probability:.1%} | {r.mll_probability:.1%} | {r.expiry_probability:.1%} | {r.median_days_to_pass:.0f} | {r.median_max_drawdown_usd:,.0f} |")
    lines += ["", "## Exposition", "", "| Variante | Book | Total micro max | MNQ | MGC | MBT | Limite dépassée |", "|---|---|---:|---:|---:|---:|---|"]
    for r in exposure.itertuples(): lines.append(f"| {r.variant} | {r.book} | {r.max_total_micro_contracts:.0f} | {r.max_mnq_gross_contracts:.0f} | {r.max_mgc_gross_contracts:.0f} | {r.max_mbt_gross_contracts:.0f} | {r.limit_breached} |")
    lines += ["", "## Lecture", "", "Le verdict porte sur la grille commune entière, sans choisir le budget qui maximise le PASS. Les CSV contiennent les résultats annuels, les sleeves, les trois stress, les horizons 35/60 et les deltas appariés. Les intervalles utilisent des chemins par blocs communs aux variantes afin de préserver la dépendance sérielle et l'appariement; les fenêtres historiques chevauchantes restent descriptives.", "",
        "## Reproduction", "", "```powershell", ".\\.venv\\Scripts\\python.exe research/usd_risk_sizing_s02_s06_20261011/run_campaign.py", "```", "",
        "Figures: `figures/performance_comparison.png`, `figures/topstep_frontier_60d.png`. Tables principales: `sizing_activity.csv`, `performance_metrics.csv`, `yearly_results.csv`, `exposure_summary.csv`, `topstep_summary.csv`, `topstep_paired_deltas.csv`, `risk_audit.csv`, `resized_trades.csv`."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
