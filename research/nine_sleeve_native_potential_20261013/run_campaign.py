"""Nine frozen sleeves: native one-micro potential, sizing, and standalone prop fit."""
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
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research.baseline_backtests.baseline_engine import SIGNAL_BUILDERS, run_baseline_backtest
from research.locked_four_sleeve_portfolio.run_locked_four_sleeve_portfolio import (
    apply_alpha_parameters, load_and_validate_config, run_alpha_sleeves,
)
from research.mgc_baseline_backtests.run_mgc_baselines import prepare_mgc_data
from research.baseline_backtests.baseline_engine import load_frozen_specs, prepare_data

SLEEVES = [
    "mnq_orb_vwap_atr", "mnq_volatility_breakout", "mnq_compression_breakout",
    "mnq_atr_trend_flip", "mgc_volatility_breakout", "mgc_gap_continuation",
    "mbt_rolling_range_breakout", "mbt_trend_pullback_ma_structure", "mbt_volatility_breakout",
]
SHORT = {s: f"S{i:02d}" for i, s in enumerate(SLEEVES, 1)}
INST = {s: s.split("_")[0].upper() for s in SLEEVES}
MBT_NAMES = SLEEVES[6:]
NY = "America/New_York"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_cfg() -> dict[str, Any]:
    return json.loads((OUT / "config.json").read_text(encoding="utf-8"))


def to_ts(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True)


def standardize_engine(sleeve: str, d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["sleeve"] = sleeve
    x["instrument"] = INST[sleeve]
    x["session_date"] = pd.to_datetime(x["session_date"])
    x["entry_ts"] = to_ts(x["entry_time"])
    x["exit_ts"] = to_ts(x["exit_time"])
    x["direction_n"] = np.where(x["direction"].astype(str).str.lower().eq("long"), 1, -1)
    if pd.api.types.is_numeric_dtype(x["direction"]):
        x["direction_n"] = np.sign(pd.to_numeric(x["direction"]))
    x["quantity_native"] = 1
    x["net_1"] = pd.to_numeric(x["net_pnl_usd"])
    x["mae_1"] = pd.to_numeric(x.get("mae_usd", np.nan), errors="coerce")
    x["mfe_1"] = pd.to_numeric(x.get("mfe_usd", np.nan), errors="coerce")
    x["holding_minutes"] = (x["exit_ts"] - x["entry_ts"]).dt.total_seconds() / 60
    x["stop_price"] = pd.to_numeric(x.get("initial_stop_price", np.nan), errors="coerce")
    return x


def load_sources(cfg: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DatetimeIndex], dict[str, Any], dict[str, Any]]:
    mnq_cfg, _, _, _ = load_and_validate_config()
    mnq_runs, mnq_trades, mnq_checks = run_alpha_sleeves(mnq_cfg)
    frozen, _, _ = load_frozen_specs(
        ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines",
        ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv",
    )
    by_id = {z["mechanism_id"]: z for z in frozen}
    mnq_data = prepare_data(
        ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet",
        start_date=pd.Timestamp(mnq_cfg["start_date"]).date(), end_date=pd.Timestamp(mnq_cfg["end_date"]).date(),
    )
    maps: dict[str, Any] = {"mnq_close": mnq_data.minute["close"].to_dict()}
    for name in ("volatility_breakout", "compression_breakout"):
        spec = apply_alpha_parameters(by_id[name], mnq_cfg["sleeves"][name]["alpha_parameters"])
        maps[name] = {str(e.setup_id): e for e in SIGNAL_BUILDERS[name](spec, mnq_data).entries}

    result: dict[str, pd.DataFrame] = {}
    result[SLEEVES[1]] = standardize_engine(SLEEVES[1], mnq_trades["volatility_breakout"])
    result[SLEEVES[2]] = standardize_engine(SLEEVES[2], mnq_trades["compression_breakout"])
    result[SLEEVES[3]] = standardize_engine(SLEEVES[3], mnq_trades["atr_trend_flip"])
    calendars: dict[str, pd.DatetimeIndex] = {}
    for sleeve, key in zip(SLEEVES[1:4], ("volatility_breakout", "compression_breakout", "atr_trend_flip")):
        calendars[sleeve] = pd.DatetimeIndex(pd.to_datetime(mnq_runs[key].daily["session_date"]))

    orb = pd.read_csv(ROOT / "export" / "mnq_orb_vwap_atr_retained" / "trades.csv")
    q0 = pd.to_numeric(orb["quantity"])
    orb["sleeve"] = SLEEVES[0]; orb["instrument"] = "MNQ"
    orb["session_date"] = pd.to_datetime(orb["session_date"])
    orb["entry_ts"] = to_ts(orb["entry_time"]); orb["exit_ts"] = to_ts(orb["exit_time"])
    orb["direction_n"] = np.where(orb["direction"].eq("long"), 1, -1)
    orb["quantity_native"] = 1
    orb["net_1"] = pd.to_numeric(orb["net_pnl_usd"]) / q0
    orb["mae_1"] = np.nan; orb["mfe_1"] = np.nan
    orb["holding_minutes"] = (orb["exit_ts"] - orb["entry_ts"]).dt.total_seconds() / 60
    result[SLEEVES[0]] = orb
    calendars[SLEEVES[0]] = calendars[SLEEVES[1]]

    mgc_start, mgc_end = pd.Timestamp("2019-05-06").date(), pd.Timestamp("2026-03-19").date()
    mgc_data, _ = prepare_mgc_data(ROOT / "data" / "processed" / "parquet" / "MGC_v_0_1m_20150101_20260923.parquet", mgc_start, mgc_end)
    maps["mgc_close"] = mgc_data.minute["close"].to_dict()
    mgc_checks = {}
    for sleeve, mechanism in zip(SLEEVES[4:6], ("volatility_breakout", "gap_continuation")):
        spec = json.loads((ROOT / "research" / "mgc_final_two_sleeve_lock_20260926" / "frozen_candidate_specs" / f"{mechanism}.json").read_text(encoding="utf-8"))
        run = run_baseline_backtest(spec, mgc_data, evaluation_start=mgc_start, evaluation_end=mgc_end)
        result[sleeve] = standardize_engine(sleeve, run.trades)
        calendars[sleeve] = pd.DatetimeIndex(pd.to_datetime(run.daily["session_date"]))
        mgc_checks[mechanism] = run.checks
        maps[mechanism] = {str(e.setup_id): e for e in SIGNAL_BUILDERS[mechanism](spec, mgc_data).entries}

    mbt_dir = ROOT / "research" / "mbt_three_sleeve_freeze_20261004" / "vote_ensemble"
    daily_mbt = pd.read_csv(mbt_dir / "daily_ensemble_results.csv", parse_dates=["session_date"])
    for sleeve, mechanism in zip(MBT_NAMES, ("rolling_range_breakout", "trend_pullback_ma_structure", "volatility_breakout")):
        z = pd.read_csv(mbt_dir / f"trades__{mechanism}__nominal.csv")
        z["sleeve"] = sleeve; z["instrument"] = "MBT"; z["session_date"] = pd.to_datetime(z["session_date"])
        z["entry_ts"] = to_ts(z["entry_time"]); z["exit_ts"] = to_ts(z["exit_time"])
        z["direction_n"] = pd.to_numeric(z["position"]); z["quantity_native"] = 1
        z["net_1"] = pd.to_numeric(z["net_pnl_usd"]); z["mae_1"] = np.nan; z["mfe_1"] = np.nan
        z["holding_minutes"] = (z["exit_ts"] - z["entry_ts"]).dt.total_seconds() / 60
        z["stop_price"] = np.nan
        result[sleeve] = z
        dd = daily_mbt[(daily_mbt.sleeve.eq(mechanism)) & daily_mbt.cost_scenario.eq("nominal")]
        calendars[sleeve] = pd.DatetimeIndex(dd.session_date.drop_duplicates().sort_values())
    checks = {"mnq": mnq_checks, "mgc": mgc_checks}
    return result, calendars, maps, {"daily_mbt": daily_mbt, "checks": checks, "mnq_minute": mnq_data.minute}


def prior_close(lookup: dict, ts: pd.Timestamp) -> float:
    local = ts.tz_convert(NY)
    key = local - pd.Timedelta(minutes=1)
    if key in lookup:
        return float(lookup[key])
    key2 = ts - pd.Timedelta(minutes=1)
    if key2 in lookup:
        return float(lookup[key2])
    raise KeyError(f"Missing prior close for {ts}")


def attach_risk(trades: dict[str, pd.DataFrame], maps: dict[str, Any], cfg: dict[str, Any]) -> None:
    eligible = set(cfg["risk_sizing_eligible"])
    for sleeve, d in trades.items():
        d["risk_interpretable"] = sleeve in eligible
        d["planned_entry"] = np.nan; d["structural_stop"] = np.nan; d["risk_distance_points"] = np.nan
        if sleeve not in eligible:
            continue
        direction = d["direction_n"].to_numpy(float)
        if sleeve == SLEEVES[0]:
            d["structural_stop"] = pd.to_numeric(d["stop_price"])
            d["planned_entry"] = [prior_close(maps["mnq_close"], ts) for ts in d["entry_ts"]]
        elif sleeve in (SLEEVES[2], SLEEVES[4]):
            mechanism = "compression_breakout" if sleeve == SLEEVES[2] else "volatility_breakout"
            planned, stop = [], []
            for row, side in zip(d.itertuples(index=False), direction):
                ins = maps[mechanism][str(row.setup_id)]
                planned.append(float(ins.long_level if side > 0 else ins.short_level))
                stop.append(float(ins.long_stop if side > 0 else ins.short_stop))
            d["planned_entry"], d["structural_stop"] = planned, stop
        elif sleeve == SLEEVES[3]:
            # ATR distance is frozen from the latest completed 15-minute bar; its value does not depend on the fill.
            d["structural_stop"] = pd.to_numeric(d["initial_stop_price"])
            dist = (pd.to_numeric(d["entry_price"]) - d["structural_stop"]).abs()
            d["risk_distance_points"] = dist
            d["planned_entry"] = d["structural_stop"] + direction * dist
        else:
            d["structural_stop"] = pd.to_numeric(d["initial_stop_price"])
            d["planned_entry"] = [prior_close(maps["mgc_close"], ts) for ts in d["entry_ts"]]
        missing = d["risk_distance_points"].isna()
        d.loc[missing, "risk_distance_points"] = (d.loc[missing, "planned_entry"] - d.loc[missing, "structural_stop"]).abs()
        ex = cfg["execution"][INST[sleeve]]
        d["risk_per_contract_planned"] = d["risk_distance_points"] * ex["point_value"] + ex["commission_round_trip"] + 2 * ex["tick_value"]
        fill_col = "entry_price"
        d["risk_per_contract_fill"] = (pd.to_numeric(d[fill_col]) - d["structural_stop"]).abs() * ex["point_value"] + ex["commission_round_trip"] + 2 * ex["tick_value"]
        d["fill_risk_excess"] = d["risk_per_contract_fill"] - d["risk_per_contract_planned"]


def fill_orb_excursions(d: pd.DataFrame, minute: pd.DataFrame, ex: dict[str, Any]) -> None:
    maes, mfes = [], []
    for r in d.itertuples(index=False):
        start, end = r.entry_ts.tz_convert(NY), r.exit_ts.tz_convert(NY)
        bars = minute.loc[(minute.index >= start) & (minute.index <= end)]
        if bars.empty:
            maes.append(np.nan); mfes.append(np.nan); continue
        entry, side = float(r.entry_price), float(r.direction_n)
        adverse = ((bars["low"] - entry) if side > 0 else (entry - bars["high"])) * ex["point_value"]
        favorable = ((bars["high"] - entry) if side > 0 else (entry - bars["low"])) * ex["point_value"]
        maes.append(float(adverse.min()) - ex["commission_round_trip"] / 2)
        mfes.append(float(favorable.max()) - ex["commission_round_trip"] / 2)
    d["mae_1"], d["mfe_1"] = maes, mfes


def variants_for(sleeve: str, d: pd.DataFrame, cfg: dict[str, Any]) -> list[tuple[str, pd.Series, str]]:
    out: list[tuple[str, pd.Series, str]] = []
    for q in cfg["fixed_quantities"]:
        out.append((f"fixed_{q}", pd.Series(q, index=d.index, dtype=int), "fixed"))
    if sleeve in cfg["risk_sizing_eligible"]:
        rpc = pd.to_numeric(d["risk_per_contract_planned"])
        common_cap = min(int(v["max_micro_contracts"]) for v in cfg["topstep"]["accounts"].values())
        for budget in cfg["risk_budgets_usd"]:
            q = np.floor(float(budget) / rpc).astype(int).clip(upper=common_cap)
            out.append((f"budget_{budget}", q, "risk"))
            out.append((f"budget_{budget}_filter_q1", q.gt(0).astype(int), "filter"))
    return out


def max_dd_duration(pnl: pd.Series) -> int:
    eq = pnl.cumsum(); peak = eq.cummax(); underwater = eq < peak
    best = cur = 0
    for value in underwater:
        cur = cur + 1 if value else 0; best = max(best, cur)
    return best


def markdown_table(frame: pd.DataFrame, include_index: bool = True) -> str:
    """Small dependency-free Markdown renderer."""
    x = frame.copy()
    if include_index:
        x.insert(0, x.index.name or "index", x.index.astype(str))
    cols = [str(c) for c in x.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in x.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def metric_row(pnl: pd.Series, trade_pnl: pd.Series, q: pd.Series, d: pd.DataFrame) -> dict[str, Any]:
    sd = pnl.std(ddof=1); eq = pnl.cumsum(); dd = eq - eq.cummax().clip(lower=0)
    gains, losses = trade_pnl.clip(lower=0).sum(), -trade_pnl.clip(upper=0).sum()
    loss = trade_pnl[trade_pnl < 0]
    return {
        "sessions": len(pnl), "signals": len(d), "trades_executed": int(q.gt(0).sum()), "signals_skipped": int(q.eq(0).sum()),
        "retention_rate": float(q.gt(0).mean()), "mean_quantity": float(q[q.gt(0)].mean()) if q.gt(0).any() else 0,
        "max_quantity": int(q.max()), "net_pnl_usd": float(pnl.sum()),
        "sharpe_zero_days": float(pnl.mean() / sd * math.sqrt(252)) if sd else np.nan,
        "profit_factor": float(gains / losses) if losses else np.nan,
        "expectancy_usd": float(trade_pnl.mean()) if len(trade_pnl) else np.nan,
        "hit_rate": float(trade_pnl.gt(0).mean()) if len(trade_pnl) else np.nan,
        "max_drawdown_usd": float(dd.min()), "drawdown_duration_sessions": max_dd_duration(pnl),
        "worst_trade_usd": float(trade_pnl.min()) if len(trade_pnl) else np.nan,
        "worst_day_usd": float(pnl.min()), "worst_5_days_usd": float(pnl.nsmallest(5).sum()),
        "loss_p50_usd": float(loss.quantile(.5)) if len(loss) else np.nan,
        "loss_p95_magnitude_usd": float(-loss.quantile(.05)) if len(loss) else np.nan,
        "mean_holding_minutes_executed": float(d.loc[q.gt(0), "holding_minutes"].mean()) if q.gt(0).any() else np.nan,
        "exposure_contract_minutes": float((d["holding_minutes"] * q).sum()),
        "mean_mae_usd": float((d.loc[q.gt(0), "mae_1"] * q[q.gt(0)]).mean()) if q.gt(0).any() else np.nan,
        "mean_mfe_usd": float((d.loc[q.gt(0), "mfe_1"] * q[q.gt(0)]).mean()) if q.gt(0).any() else np.nan,
    }


def daily_arrays(sleeve: str, d: pd.DataFrame, calendar: pd.DatetimeIndex, q: pd.Series, stress: int, cfg: dict[str, Any], mbt_daily: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    ex = cfg["execution"][INST[sleeve]]
    trade_pnl = d["net_1"] * q - 2 * stress * ex["tick_value"] * q
    pnl = pd.Series(trade_pnl.to_numpy(), index=d["session_date"]).groupby(level=0).sum().reindex(calendar).fillna(0.0)
    if sleeve in MBT_NAMES:
        mechanism = sleeve.split("mbt_", 1)[1]
        base = mbt_daily[(mbt_daily.sleeve.eq(mechanism)) & mbt_daily.cost_scenario.eq("nominal")].set_index("session_date")
        # Exact frozen ensemble intraday path at nominal; scale by fixed q and subtract stressed entry-side allowance.
        q_const = int(q.max()) if len(q) else 0
        intraday = base["worst_intraday_pnl_usd"].reindex(calendar).fillna(0.0) * q_const - stress * ex["tick_value"] * q_const
    else:
        adverse = pd.Series(np.minimum(d["mae_1"].fillna(d["net_1"]).to_numpy() * q.to_numpy() - stress * ex["tick_value"] * q.to_numpy(), trade_pnl.to_numpy()), index=d["session_date"])
        intraday = adverse.groupby(level=0).min().reindex(calendar).fillna(0.0)
    traded = pd.Series(q.gt(0).to_numpy(dtype=bool), index=d["session_date"], dtype=bool).groupby(level=0).max().reindex(calendar, fill_value=False).astype(bool)
    return pnl, intraday, traded


def bootstrap_native(sleeve: str, pnl: pd.Series, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(cfg["bootstrap"]["seed"] + SLEEVES.index(sleeve))
    a = pnl.to_numpy(float); n = len(a); rows = []
    for block in cfg["bootstrap"]["blocks"]:
        vals = []
        for _ in range(cfg["bootstrap"]["paths_per_block"]):
            idx = []
            while len(idx) < n:
                start = int(rng.integers(0, max(1, n - block + 1)))
                idx.extend(range(start, min(start + block, n)))
            z = pd.Series(a[np.asarray(idx[:n])])
            sd = z.std(ddof=1); dd = z.cumsum() - z.cumsum().cummax().clip(lower=0)
            vals.append((z.sum(), z.mean() / sd * math.sqrt(252) if sd else np.nan, dd.min(), z.min()))
        ar = np.asarray(vals)
        for j, metric in enumerate(("net_pnl_usd", "sharpe_zero_days", "max_drawdown_usd", "worst_day_usd")):
            rows.append({"sleeve": sleeve, "block": block, "metric": metric, "ci_low": np.nanquantile(ar[:, j], .025), "median": np.nanmedian(ar[:, j]), "ci_high": np.nanquantile(ar[:, j], .975)})
    return rows


def simulate_topstep(pnl: np.ndarray, intra: np.ndarray, indices: np.ndarray, account: int, rules: dict[str, Any], horizon: int) -> tuple[str, float, float]:
    r = rules["accounts"][str(account)]; equity = peak = float(account); floor = account - r["mll_distance"]
    best_day = 0.0; max_dd = 0.0
    for day, i in enumerate(indices[:horizon], 1):
        max_dd = min(max_dd, equity + float(intra[i]) - peak)
        if equity + float(intra[i]) <= floor:
            return "MLL", np.nan, max_dd
        value = float(pnl[i]); equity += value; peak = max(peak, equity); max_dd = min(max_dd, equity - peak)
        best_day = max(best_day, value, 0.0); profit = equity - account
        if profit >= r["profit_target"] and best_day <= rules["consistency_share"] * profit:
            return "PASS", float(day), max_dd
        floor = max(floor, min(float(account), equity - r["mll_distance"]))
    return "UNRESOLVED", np.nan, max_dd


def prop_summary(sleeve: str, series: dict[str, tuple[pd.Series, pd.Series, pd.Series]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, paired = [], []
    n = len(next(iter(series.values()))[0]); rng = np.random.default_rng(cfg["bootstrap"]["seed"] + 100 + SLEEVES.index(sleeve))
    for horizon in cfg["topstep"]["horizons_sessions"]:
        paths: list[tuple[int, int, np.ndarray]] = []
        for block in cfg["bootstrap"]["blocks"]:
            for path_id in range(cfg["bootstrap"]["paths_per_block"]):
                idx = []
                while len(idx) < horizon:
                    start = int(rng.integers(0, max(1, n - block + 1)))
                    idx.extend(range(start, min(start + block, n)))
                paths.append((block, path_id, np.asarray(idx[:horizon])))
        for account in map(int, cfg["topstep"]["accounts"]):
            results: dict[str, list[tuple[str, float, float]]] = {}
            for variant, (pnl, intra, _) in series.items():
                results[variant] = [simulate_topstep(pnl.to_numpy(), intra.to_numpy(), ix, account, cfg["topstep"], horizon) for _, _, ix in paths]
                rr = results[variant]; outcomes = pd.Series([x[0] for x in rr]); passes = [x[1] for x in rr if x[0] == "PASS"]
                rows.append({"sleeve": sleeve, "variant": variant, "account": account, "horizon": horizon, "paths": len(rr),
                             "pass_probability": float(outcomes.eq("PASS").mean()), "mll_probability": float(outcomes.eq("MLL").mean()),
                             "unresolved_probability": float(outcomes.eq("UNRESOLVED").mean()), "median_days_to_pass": float(np.nanmedian(passes)) if passes else np.nan,
                             "median_max_drawdown_usd": float(np.median([x[2] for x in rr]))})
            ref = results["fixed_1"]
            for variant, rr in results.items():
                delta = np.asarray([int(a[0] == "PASS") - int(b[0] == "PASS") for a, b in zip(rr, ref)])
                # Cluster by bootstrap block: dispersion across the three block estimates preserves overlap dependence.
                block_means = []
                for block in cfg["bootstrap"]["blocks"]:
                    pos = [i for i, x in enumerate(paths) if x[0] == block]
                    block_means.append(float(delta[pos].mean()))
                paired.append({"sleeve": sleeve, "variant": variant, "reference": "fixed_1", "account": account, "horizon": horizon,
                               "paired_pass_delta": float(delta.mean()), "block_robust_low": float(min(block_means)), "block_robust_high": float(max(block_means))})
    return rows, paired


def main() -> None:
    OUT.mkdir(exist_ok=True); (OUT / "figures").mkdir(exist_ok=True); (OUT / "sleeve_reports").mkdir(exist_ok=True)
    cfg = read_cfg(); trades, calendars, maps, aux = load_sources(cfg)
    attach_risk(trades, maps, cfg)
    fill_orb_excursions(trades[SLEEVES[0]], aux["mnq_minute"], cfg["execution"]["MNQ"])

    metric_rows, yearly_rows, sub_rows, sizing_rows, risk_rows, boot_rows, trade_exports = [], [], [], [], [], [], []
    prop_rows, paired_rows = [], []
    series_by_sleeve: dict[str, dict[str, tuple[pd.Series, pd.Series, pd.Series]]] = {}
    for sleeve in SLEEVES:
        d = trades[sleeve].reset_index(drop=True); cal = calendars[sleeve]
        series_by_sleeve[sleeve] = {}
        for variant, q, family in variants_for(sleeve, d, cfg):
            q = q.reset_index(drop=True).astype(int)
            if q.lt(0).any(): raise RuntimeError("Negative quantity")
            if family == "risk":
                budget = int(variant.split("_")[1])
                if ((q * d["risk_per_contract_planned"] > budget + 1e-9) & q.gt(0)).any(): raise RuntimeError("Budget breach")
            for stress in cfg["extra_slippage_ticks_per_side"]:
                pnl, intra, traded = daily_arrays(sleeve, d, cal, q, stress, cfg, aux["daily_mbt"])
                tp = d["net_1"] * q - 2 * stress * cfg["execution"][INST[sleeve]]["tick_value"] * q
                row = {"sleeve": sleeve, "sleeve_id": SHORT[sleeve], "variant": variant, "family": family, "extra_slippage_ticks_per_side": stress}
                row.update(metric_row(pnl, tp[q.gt(0)], q, d)); metric_rows.append(row)
                for year in sorted(cal.year.unique()):
                    part = pnl[cal.year == year]
                    yearly_rows.append({"sleeve": sleeve, "variant": variant, "stress": stress, "year": int(year), "net_pnl_usd": part.sum(), "sharpe_zero_days": part.mean()/part.std(ddof=1)*math.sqrt(252) if part.std(ddof=1) else np.nan, "worst_day_usd": part.min()})
                splits = [("pre_2024", cal < "2024-01-01"), ("2024_plus", cal >= "2024-01-01")]
                if sleeve in MBT_NAMES:
                    splits = [("legacy_is", cal <= "2025-05-28"), ("inspected_post_split", cal >= "2025-05-29")]
                for label, mask in splits:
                    part = pnl[mask]
                    if len(part): sub_rows.append({"sleeve": sleeve, "variant": variant, "stress": stress, "subperiod": label, "sessions": len(part), "net_pnl_usd": part.sum(), "sharpe_zero_days": part.mean()/part.std(ddof=1)*math.sqrt(252) if part.std(ddof=1) else np.nan, "worst_day_usd": part.min()})
                if stress == 0:
                    series_by_sleeve[sleeve][variant] = (pnl, intra, traded)
            counts = q.value_counts().sort_index()
            for qty, count in counts.items(): sizing_rows.append({"sleeve": sleeve, "variant": variant, "quantity": int(qty), "signals": int(count)})
            if family == "risk":
                active = q.gt(0)
                risk_rows.append({"sleeve": sleeve, "variant": variant, "signals": len(d), "executed": int(active.sum()), "skipped": int((~active).sum()),
                                  "mean_planned_risk_usd": float((q[active]*d.loc[active,"risk_per_contract_planned"]).mean()) if active.any() else np.nan,
                                  "max_planned_risk_usd": float((q*d["risk_per_contract_planned"]).max()),
                                  "mean_fill_risk_usd": float((q[active]*d.loc[active,"risk_per_contract_fill"]).mean()) if active.any() else np.nan,
                                  "fill_exceedance_rate": float((d.loc[active,"risk_per_contract_fill"] > d.loc[active,"risk_per_contract_planned"]+1e-9).mean()) if active.any() else np.nan,
                                  "max_fill_over_plan_usd": float((q*(d["risk_per_contract_fill"]-d["risk_per_contract_planned"])).max())})
            export = pd.DataFrame({"sleeve": sleeve, "variant": variant, "trade_id": d.get("trade_id", pd.Series(np.arange(len(d))+1)), "session_date": d["session_date"], "quantity": q,
                                   "net_pnl_nominal_usd": d["net_1"]*q, "mae_usd": d["mae_1"]*q, "mfe_usd": d["mfe_1"]*q,
                                   "planned_risk_usd": q*d.get("risk_per_contract_planned", np.nan), "fill_risk_usd": q*d.get("risk_per_contract_fill", np.nan)})
            trade_exports.append(export)
        q1p = series_by_sleeve[sleeve]["fixed_1"][0]
        boot_rows.extend(bootstrap_native(sleeve, q1p, cfg))
        p, pair = prop_summary(sleeve, series_by_sleeve[sleeve], cfg); prop_rows.extend(p); paired_rows.extend(pair)

    metrics = pd.DataFrame(metric_rows); yearly = pd.DataFrame(yearly_rows); sub = pd.DataFrame(sub_rows)
    sizing = pd.DataFrame(sizing_rows); risks = pd.DataFrame(risk_rows); bootstrap = pd.DataFrame(boot_rows)
    prop = pd.DataFrame(prop_rows); paired = pd.DataFrame(paired_rows); exported = pd.concat(trade_exports, ignore_index=True)
    metrics.to_csv(OUT/"native_and_sizing_metrics.csv", index=False); yearly.to_csv(OUT/"yearly_results.csv", index=False)
    sub.to_csv(OUT/"chronological_subperiods.csv", index=False); sizing.to_csv(OUT/"quantity_distributions.csv", index=False)
    risks.to_csv(OUT/"risk_budget_audit.csv", index=False); bootstrap.to_csv(OUT/"native_block_bootstrap_ci.csv", index=False)
    prop.to_csv(OUT/"topstep_standalone_summary.csv", index=False); paired.to_csv(OUT/"topstep_paired_deltas.csv", index=False)
    exported.to_csv(OUT/"resized_trade_ledger.csv", index=False)

    audit = []
    for sleeve, d in trades.items():
        audit.append({"sleeve": sleeve, "sleeve_id": SHORT[sleeve], "instrument": INST[sleeve], "signals": len(d), "first_trade": d.session_date.min().date(), "last_trade": d.session_date.max().date(),
                      "calendar_sessions": len(calendars[sleeve]), "entry_model": "frozen", "exit_model": "time_stop_240" if sleeve==SLEEVES[1] else "frozen",
                      "native_source_quantity": "USD250 variable; reconstructed q1" if sleeve==SLEEVES[0] else "one executable micro",
                      "risk_sizing": "interpretable" if sleeve in cfg["risk_sizing_eligible"] else "non_interpretable_fixed_only"})
    pd.DataFrame(audit).to_csv(OUT/"sleeve_implementation_audit.csv", index=False)

    base = metrics[(metrics.variant.eq("fixed_1")) & metrics.extra_slippage_ticks_per_side.eq(0)].set_index("sleeve")
    fig, ax = plt.subplots(figsize=(10,6)); ax.scatter(-base.max_drawdown_usd, base.net_pnl_usd)
    for s,r in base.iterrows(): ax.annotate(SHORT[s], (-r.max_drawdown_usd,r.net_pnl_usd))
    ax.set(xlabel="Max drawdown magnitude (USD)", ylabel="Net PnL (USD)", title="Native one-micro potential"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT/"figures"/"native_pnl_drawdown.png",dpi=150); plt.close(fig)
    risk_plot = metrics[(metrics.family.eq("risk")) & metrics.extra_slippage_ticks_per_side.eq(0) & ~metrics.variant.str.contains("filter")]
    fig, axes = plt.subplots(3,2,figsize=(12,11),sharex=True); axes=axes.ravel()
    for ax,s in zip(axes,cfg["risk_sizing_eligible"]):
        z=risk_plot[risk_plot.sleeve.eq(s)].copy(); z["budget"]=z.variant.str.extract(r"(\d+)").astype(int)
        ax2=ax.twinx(); ax.plot(z.budget,z.net_pnl_usd,marker="o",label="PnL"); ax2.plot(z.budget,z.retention_rate,color="tab:red",marker="x",label="retention")
        ax.set_title(f"{SHORT[s]} {s}"); ax.set_ylabel("PnL USD"); ax2.set_ylabel("retention",color="tab:red")
    axes[-1].axis("off"); fig.tight_layout(); fig.savefig(OUT/"figures"/"risk_budget_retention.png",dpi=150); plt.close(fig)

    fixed_only = ", ".join(SHORT[s] for s in SLEEVES if s not in cfg["risk_sizing_eligible"])
    report = ["# Analyse individuelle des neuf sleeves gelées", "", "## Verdict", "",
              "**TRADEOFF — aucune modification du book actif.** Les variantes sont un diagnostic de capacité et de compatibilité prop, pas une sélection a posteriori.", "",
              "## Points d'audit déterminants", "",
              "- S01 était réellement sizée à **250 USD de risque nominal** avec quantité entière variable dans son ledger source. La référence native de cette campagne est sa reconstruction linéaire à un micro.",
              "- S02 exécute un **time-stop de 240 minutes**. Le niveau RTH open n'est pas un stop exécuté et le sizing USD est donc non interprétable.",
              "- S07–S09 sont les stratégies exécutables agrégées par vote, une position MBT maximum chacune ; les cellules sous-jacentes ne sont pas comptées comme ordres.",
              f"- Sizing USD non interprétable, quantités fixes seulement : {fixed_only}.",
              "- Les périodes postérieures aux anciens splits ont déjà été inspectées pendant la sélection : elles mesurent la stabilité, pas un nouvel OOS indépendant.", "",
              "## Référence native (un micro, coûts nominaux)", "",
              markdown_table(base[["signals","net_pnl_usd","sharpe_zero_days","profit_factor","max_drawdown_usd","worst_day_usd","worst_trade_usd","mean_holding_minutes_executed"]].round(3)), "",
              "## Lecture centrale", "",
              "La grille permet de distinguer explicitement la granularité (trades supprimés à quantité zéro) de l'effet d'échelle (variante `filter_q1`). Les budgets faibles ne sont acceptables que si leur stabilité, leurs pertes extrêmes et leur vitesse Topstep restent cohérentes dans les trois longueurs de blocs ; aucun budget n'est promu sur le seul PASS agrégé.", "",
              "## Topstep — règles vérifiées le 2026-10-13", "",
              "Comptes 50K/100K/150K : objectifs 3K/6K/9K, MLL 2K/3K/4.5K, limites 50/100/150 micros et règle de cohérence 55 %. Le MLL suit le solde EOD, est contrôlé en temps réel avec PnL latent et se verrouille au solde initial. Les fins de simulation sont `UNRESOLVED`, jamais `EXPIRY`.", "",
              "## Fichiers", "",
              "Les métriques, distributions de quantité, audit du risque prévu versus fill, années, sous-périodes, intervalles block-bootstrap et simulations Topstep appariées sont fournis dans les CSV voisins. Les hypothèses reproductibles sont dans `config.json` et `PROTOCOL.md`."]
    (OUT/"REPORT.md").write_text("\n".join(report),encoding="utf-8")
    for sleeve in SLEEVES:
        z=metrics[(metrics.sleeve.eq(sleeve)) & metrics.extra_slippage_ticks_per_side.eq(0)]
        lines=[f"# {SHORT[sleeve]} — {sleeve}","",f"Sizing USD : {'interprétable' if sleeve in cfg['risk_sizing_eligible'] else 'non interprétable; fixed-only'}.","",markdown_table(z[["variant","trades_executed","signals_skipped","mean_quantity","net_pnl_usd","sharpe_zero_days","profit_factor","max_drawdown_usd","worst_day_usd"]].round(3), include_index=False)]
        (OUT/"sleeve_reports"/f"{SHORT[sleeve]}_{sleeve}.md").write_text("\n".join(lines),encoding="utf-8")
    metadata={"campaign":cfg["campaign_id"],"completed_at":"2026-10-13","verdict":"TRADEOFF","frozen_sources_modified":False,"config_sha256":sha256(OUT/"config.json"),"rows":{"metrics":len(metrics),"trades":len(exported),"topstep":len(prop)},"validation":{"nonnegative_integer_quantities":True,"risk_budget_cap":True,"zero_days_in_sharpe":True,"terminal_label":"UNRESOLVED"}}
    (OUT/"run_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps(metadata,indent=2))

if __name__ == "__main__": main()
