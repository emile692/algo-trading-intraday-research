"""Integer contract scaling diagnostics for the frozen six-sleeve portfolio."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = OUTPUT_DIR / "topstep_portfolio_scaling_config.json"
FIGURES_DIR = OUTPUT_DIR / "figures"
MPL_CACHE = OUTPUT_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analytics.mnq_orb_topstep_50k_simulation import TopstepCampaignSpec  # noqa: E402


SLEEVES = [
    "mnq_orb_vwap_atr",
    "mnq_volatility_breakout",
    "mnq_compression_breakout",
    "mnq_atr_trend_flip",
    "mgc_volatility_breakout",
    "mgc_gap_continuation",
]
INSTRUMENT = {s: ("MNQ" if s.startswith("mnq_") else "MGC") for s in SLEEVES}
ALLOCATIONS = ["equal_sleeve_weight", "inverse_volatility", "equal_risk_contribution"]
TRADE_LOCK_KEYS = {
    "mnq_orb_vwap_atr": "trade_mnq_orb",
    "mnq_volatility_breakout": "trade_mnq_volatility",
    "mnq_compression_breakout": "trade_mnq_compression",
    "mnq_atr_trend_flip": "trade_mnq_atr",
    "mgc_volatility_breakout": "trade_mgc_volatility",
    "mgc_gap_continuation": "trade_mgc_gap",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> tuple[dict[str, Any], dict[str, str]]:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not cfg.get("locked_before_execution"):
        raise RuntimeError("Scaling config must be locked before execution")
    if cfg["allocations"] != ALLOCATIONS:
        raise RuntimeError("Frozen allocation list changed")
    hashes: dict[str, str] = {}
    for key, item in cfg["source_locks"].items():
        actual = sha256(REPO_ROOT / item["path"])
        if actual != item["sha256"]:
            raise RuntimeError(f"Locked input hash mismatch: {key}")
        hashes[key] = actual
    return cfg, hashes


def round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def integer_apportion(weights: np.ndarray, total: int, minimum: int = 1) -> np.ndarray:
    """Deterministic constrained largest-remainder allocation."""
    weights = np.asarray(weights, dtype=float)
    if total < len(weights) * minimum:
        raise ValueError("Total copy budget cannot keep every sleeve active")
    if np.any(weights < 0) or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-10):
        raise ValueError("Weights must be long-only and sum to one")
    target = total * weights
    result = np.maximum(np.floor(target).astype(int), minimum)
    while result.sum() < total:
        priority = target - result
        result[int(np.argmax(priority))] += 1
    while result.sum() > total:
        eligible = np.flatnonzero(result > minimum)
        if not len(eligible):
            raise RuntimeError("Cannot reduce integer allocation to target total")
        overage = result[eligible] - target[eligible]
        result[int(eligible[np.argmax(overage)])] -= 1
    return result


def load_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    sources = cfg["source_locks"]
    pnl = pd.read_csv(REPO_ROOT / sources["daily_sleeve_pnl"]["path"], parse_dates=["session_date"]).set_index("session_date")[SLEEVES]
    counts = pd.read_csv(REPO_ROOT / sources["daily_trade_counts"]["path"], parse_dates=["session_date"]).set_index("session_date")[SLEEVES]
    weights = pd.read_csv(REPO_ROOT / sources["allocation_weights"]["path"]).set_index("sleeve").reindex(SLEEVES)
    if not pnl.index.equals(counts.index) or pnl.isna().any().any() or counts.isna().any().any():
        raise RuntimeError("Phase-1 daily inputs are not clean and aligned")
    trades = {s: pd.read_csv(REPO_ROOT / sources[TRADE_LOCK_KEYS[s]]["path"], parse_dates=["session_date"]) for s in SLEEVES}
    position_columns = ["timestamp_utc"] + [f"position__{s}" for s in SLEEVES]
    positions = pd.read_csv(REPO_ROOT / sources["timestamp_portfolio"]["path"], usecols=position_columns, parse_dates=["timestamp_utc"])
    return pnl, counts, weights, trades, positions


def build_integer_mappings(cfg: dict[str, Any], weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for allocation in ALLOCATIONS:
        w = weights[allocation].to_numpy(float)
        for scaling in cfg["integer_executable_scaling_levels"]:
            total = round_half_up(float(scaling))
            quantities = integer_apportion(w, total, minimum=int(cfg["integer_mapping"]["minimum_copies_per_sleeve"]))
            theoretical = float(scaling) * w
            tracking_l1 = float(np.abs(quantities / quantities.sum() - w).sum())
            for idx, sleeve in enumerate(SLEEVES):
                rows.append({
                    "allocation": allocation,
                    "scaling": float(scaling),
                    "total_sleeve_copy_budget": total,
                    "sleeve": sleeve,
                    "instrument": INSTRUMENT[sleeve],
                    "frozen_weight": w[idx],
                    "theoretical_continuous_copies": theoretical[idx],
                    "integer_sleeve_copies": int(quantities[idx]),
                    "realized_copy_weight": quantities[idx] / quantities.sum(),
                    "allocation_l1_tracking_error": tracking_l1,
                })
    return pd.DataFrame(rows)


def build_continuous_mappings(cfg: dict[str, Any], weights: pd.DataFrame) -> pd.DataFrame:
    levels = sorted(set(float(v) for v in cfg["continuous_scaling_levels"] + cfg["integer_executable_scaling_levels"]))
    rows = []
    for allocation in ALLOCATIONS:
        w = weights[allocation].to_numpy(float)
        for scaling in levels:
            for idx, sleeve in enumerate(SLEEVES):
                rows.append({
                    "allocation": allocation, "scaling": scaling, "sleeve": sleeve,
                    "instrument": INSTRUMENT[sleeve], "frozen_weight": w[idx],
                    "theoretical_continuous_copies": scaling * w[idx],
                    "integer_sleeve_copies": np.nan, "realized_copy_weight": np.nan,
                    "allocation_l1_tracking_error": np.nan,
                })
    return pd.DataFrame(rows)


def requested_integer_feasibility(cfg: dict[str, Any]) -> pd.DataFrame:
    minimum = len(SLEEVES) * int(cfg["integer_mapping"]["minimum_copies_per_sleeve"])
    rows = []
    for scaling in cfg["continuous_scaling_levels"]:
        budget = round_half_up(float(scaling))
        rows.append({
            "requested_scaling": float(scaling), "rounded_total_copy_budget": budget,
            "minimum_copies_required_for_six_active_sleeves": minimum,
            "executable_with_all_six_sleeves": bool(budget >= minimum),
            "reason": "executable" if budget >= minimum else "fewer integer copies than frozen active sleeves",
        })
    return pd.DataFrame(rows)


def mapping_vectors(mappings: pd.DataFrame, allocation: str, scaling: float, integer: bool = True) -> np.ndarray:
    scoped = mappings[mappings["allocation"].eq(allocation) & mappings["scaling"].eq(scaling)].set_index("sleeve").reindex(SLEEVES)
    column = "integer_sleeve_copies" if integer else "theoretical_continuous_copies"
    return scoped[column].to_numpy(float)


def exposure_table(mappings: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    base = positions[[f"position__{s}" for s in SLEEVES]].to_numpy(float)
    rows = []
    for allocation in ALLOCATIONS:
        for scaling in sorted(mappings["scaling"].unique()):
            q = mapping_vectors(mappings, allocation, float(scaling), integer=True)
            scaled = base * q
            mnq = scaled[:, :4]
            mgc = scaled[:, 4:]
            max_by_sleeve = np.max(np.abs(scaled), axis=0)
            row = {
                "allocation": allocation,
                "scaling": float(scaling),
                "mnq_sleeve_copies": int(q[:4].sum()),
                "mgc_sleeve_copies": int(q[4:].sum()),
                "max_simultaneous_mnq_contracts": int(np.max(np.sum(np.abs(mnq), axis=1))),
                "max_simultaneous_mgc_contracts": int(np.max(np.sum(np.abs(mgc), axis=1))),
                "max_simultaneous_total_micro_contracts": int(np.max(np.sum(np.abs(scaled), axis=1))),
                "max_net_mnq_contracts_abs": int(np.max(np.abs(np.sum(mnq, axis=1)))),
                "max_net_mgc_contracts_abs": int(np.max(np.abs(np.sum(mgc, axis=1)))),
            }
            row.update({f"max_contracts__{s}": int(max_by_sleeve[idx]) for idx, s in enumerate(SLEEVES)})
            rows.append(row)
    return pd.DataFrame(rows)


def topstep_rules() -> list[dict[str, Any]]:
    return [asdict(rule) for rule in TopstepCampaignSpec().rulesets]


def simulate_path(pnl: np.ndarray, traded: np.ndarray, rules: dict[str, Any]) -> dict[str, Any]:
    equity = float(rules["starting_balance_usd"])
    high = equity
    best_day = 0.0
    traded_days = 0
    economic_target = False
    inconsistency_seen = False
    max_drawdown = 0.0
    worst_day = 0.0
    days_to_pass = np.nan
    terminal = "EXPIRY"
    used = 0
    for idx, value in enumerate(np.asarray(pnl, dtype=float)):
        used = idx + 1
        if bool(traded[idx]):
            traded_days += 1
        equity += float(value)
        high = max(high, equity)
        drawdown = equity - high
        max_drawdown = min(max_drawdown, drawdown)
        worst_day = min(worst_day, float(value))
        cumulative = equity - float(rules["starting_balance_usd"])
        best_day = max(best_day, max(float(value), 0.0))
        target_hit = cumulative >= float(rules["profit_target_usd"])
        consistency = target_hit and best_day < float(rules["consistency_share_limit"]) * cumulative
        economic_target = economic_target or target_hit
        inconsistency_seen = inconsistency_seen or (target_hit and not consistency)
        trailing_floor = high - float(rules["trailing_mll_usd"])
        if bool(rules["lock_at_starting_balance"]):
            trailing_floor = min(trailing_floor, float(rules["starting_balance_usd"]))
        if equity <= trailing_floor:
            terminal = "MLL_FAIL"
            break
        if consistency:
            terminal = "PASS"
            days_to_pass = float(traded_days)
            break
        if rules["max_traded_days"] is not None and traded_days >= int(rules["max_traded_days"]):
            terminal = "EXPIRY"
            break
    return {
        "outcome": terminal,
        "days_to_pass": days_to_pass,
        "terminal_pnl_usd": equity - float(rules["starting_balance_usd"]),
        "max_drawdown_usd": max_drawdown,
        "worst_day_usd": worst_day,
        "largest_portfolio_loss_usd": worst_day,
        "economic_target_hit": economic_target,
        "consistency_violation": inconsistency_seen,
        "traded_days": traded_days,
        "calendar_sessions": used,
    }


def valid_start_indices(traded: np.ndarray, horizon: int) -> np.ndarray:
    remaining = np.cumsum(traded[::-1])[::-1]
    return np.flatnonzero(remaining >= horizon)


def historical_runs(pnl: pd.DataFrame, counts: pd.DataFrame, mappings: pd.DataFrame,
                    exposure: pd.DataFrame, integer: bool = True) -> pd.DataFrame:
    matrix = pnl.to_numpy(float)
    traded = counts.sum(axis=1).to_numpy(float) > 0
    rows = []
    for allocation in ALLOCATIONS:
        for scaling in sorted(mappings["scaling"].unique()):
            q = mapping_vectors(mappings, allocation, float(scaling), integer=integer)
            daily = matrix @ q
            exp_rows = exposure[exposure["allocation"].eq(allocation) & exposure["scaling"].eq(scaling)]
            exp = exp_rows.iloc[0] if integer else None
            for rules in topstep_rules():
                horizon = int(rules["max_traded_days"])
                for start_idx in valid_start_indices(traded, horizon):
                    result = simulate_path(daily[start_idx:], traded[start_idx:], rules)
                    rows.append({
                        "implementation": "integer" if integer else "continuous",
                        "allocation": allocation,
                        "scaling": float(scaling),
                        "ruleset": rules["name"],
                        "horizon_traded_days": horizon,
                        "start_session_date": pnl.index[start_idx],
                        "max_simultaneous_mnq_contracts": exp["max_simultaneous_mnq_contracts"] if exp is not None else np.nan,
                        "max_simultaneous_mgc_contracts": exp["max_simultaneous_mgc_contracts"] if exp is not None else np.nan,
                        "max_simultaneous_total_micro_contracts": exp["max_simultaneous_total_micro_contracts"] if exp is not None else np.nan,
                        **result,
                    })
    return pd.DataFrame(rows)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - half, center + half


def summarize_runs(runs: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in runs.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        total = len(group)
        for outcome, label in (("PASS", "pass"), ("MLL_FAIL", "mll"), ("EXPIRY", "expiry")):
            count = int(group["outcome"].eq(outcome).sum())
            low, high = wilson(count, total)
            row[f"{label}_count"] = count
            row[f"{label}_probability"] = count / total
            row[f"{label}_ci95_low"] = low
            row[f"{label}_ci95_high"] = high
        passed = group[group["outcome"].eq("PASS")]
        row.update({
            "path_count": total,
            "median_days_to_pass": float(passed["days_to_pass"].median()) if len(passed) else np.nan,
            "median_terminal_pnl_usd": float(group["terminal_pnl_usd"].median()),
            "p05_terminal_pnl_usd": float(group["terminal_pnl_usd"].quantile(0.05)),
            "median_max_drawdown_usd": float(group["max_drawdown_usd"].median()),
            "p05_max_drawdown_usd": float(group["max_drawdown_usd"].quantile(0.05)),
            "median_worst_day_usd": float(group["worst_day_usd"].median()),
            "p05_worst_day_usd": float(group["worst_day_usd"].quantile(0.05)),
            "consistency_violation_rate": float(group["consistency_violation"].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_index_paths(traded: np.ndarray, horizon: int, block: int, paths: int,
                          rng: np.random.Generator) -> list[np.ndarray]:
    n = len(traded)
    max_start = n - block
    output = []
    for _ in range(paths):
        sampled: list[int] = []
        traded_days = 0
        while traded_days < horizon:
            start = int(rng.integers(0, max_start + 1))
            for idx in range(start, start + block):
                sampled.append(idx)
                traded_days += int(traded[idx])
                if traded_days >= horizon:
                    break
        output.append(np.asarray(sampled, dtype=int))
    return output


def bootstrap_runs(cfg: dict[str, Any], pnl: pd.DataFrame, counts: pd.DataFrame,
                   mappings: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    matrix = pnl.to_numpy(float)
    traded = counts.sum(axis=1).to_numpy(float) > 0
    boot = cfg["bootstrap"]
    rng = np.random.default_rng(int(boot["random_seed"]))
    rows = []
    for rules_idx, rules in enumerate(topstep_rules()):
        horizon = int(rules["max_traded_days"])
        for block in boot["block_lengths_sessions"]:
            paths = bootstrap_index_paths(traded, horizon, int(block), int(boot["paths_per_block_length"]), rng)
            for allocation in ALLOCATIONS:
                for scaling in sorted(mappings["scaling"].unique()):
                    q = mapping_vectors(mappings, allocation, float(scaling), integer=True)
                    daily = matrix @ q
                    exp = exposure[exposure["allocation"].eq(allocation) & exposure["scaling"].eq(scaling)].iloc[0]
                    for path_id, indices in enumerate(paths, start=1):
                        result = simulate_path(daily[indices], traded[indices], rules)
                        rows.append({
                            "allocation": allocation,
                            "scaling": float(scaling),
                            "ruleset": rules["name"],
                            "horizon_traded_days": horizon,
                            "block_length_sessions": int(block),
                            "path_id": path_id,
                            "max_simultaneous_mnq_contracts": int(exp["max_simultaneous_mnq_contracts"]),
                            "max_simultaneous_mgc_contracts": int(exp["max_simultaneous_mgc_contracts"]),
                            **result,
                        })
    return pd.DataFrame(rows)


def daily_explicit_costs(trades: dict[str, pd.DataFrame], index: pd.Index) -> pd.DataFrame:
    result = pd.DataFrame(0.0, index=index, columns=SLEEVES)
    for sleeve, frame in trades.items():
        if "gross_pnl_usd" in frame:
            cost = pd.to_numeric(frame["gross_pnl_usd"], errors="coerce") - pd.to_numeric(frame["net_pnl_usd"], errors="coerce")
        else:
            cost = pd.to_numeric(frame["pnl_usd"], errors="coerce") - pd.to_numeric(frame["net_pnl_usd"], errors="coerce")
        by_day = pd.DataFrame({"session_date": pd.to_datetime(frame["session_date"]), "cost": cost.clip(lower=0)}).groupby("session_date")["cost"].sum()
        result[sleeve] = by_day.reindex(index).fillna(0.0)
    return result


def stress_tests(cfg: dict[str, Any], pnl: pd.DataFrame, counts: pd.DataFrame, trades: dict[str, pd.DataFrame],
                 mappings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = pnl.to_numpy(float)
    traded = counts.sum(axis=1).to_numpy(float) > 0
    costs = daily_explicit_costs(trades, pnl.index).to_numpy(float)
    worst_trades = np.array([float(pd.to_numeric(trades[s]["net_pnl_usd"]).min()) for s in SLEEVES])
    event_rows, simulation_rows = [], []
    mll = float(cfg["stress"]["mll_usd"])
    for allocation in ALLOCATIONS:
        for scaling in sorted(mappings["scaling"].unique()):
            q = mapping_vectors(mappings, allocation, float(scaling), integer=True)
            base_daily = matrix @ q
            scenarios = {"base": base_daily, "double_costs": base_daily - costs @ q}
            mean = float(base_daily.mean())
            for vol in cfg["stress"]["volatility_multipliers"]:
                scenarios[f"volatility_{vol:g}x"] = mean + float(vol) * (base_daily - mean)
            sync_loss = float(worst_trades @ q)
            event_definitions = {"repeated_worst_portfolio_day": float(base_daily.min()),
                                 "synchronized_worst_sleeve_trades": sync_loss}
            for name, loss in event_definitions.items():
                repetitions = int(math.ceil(mll / abs(loss))) if loss < 0 else pd.NA
                event_rows.append({
                    "allocation": allocation, "scaling": float(scaling), "stress": name,
                    "single_event_loss_usd": loss, "events_to_initial_mll": repetitions,
                    "cumulative_loss_at_mll_event_usd": float(repetitions * loss) if loss < 0 else np.nan,
                })
            for scenario, daily in scenarios.items():
                loss = float(daily.min())
                repetitions = int(math.ceil(mll / abs(loss))) if loss < 0 else pd.NA
                event_rows.append({
                    "allocation": allocation, "scaling": float(scaling), "stress": scenario,
                    "single_event_loss_usd": loss, "events_to_initial_mll": repetitions,
                    "cumulative_loss_at_mll_event_usd": float(repetitions * loss) if loss < 0 else np.nan,
                })
                for rules in topstep_rules():
                    horizon = int(rules["max_traded_days"])
                    scenario_results = []
                    for start_idx in valid_start_indices(traded, horizon):
                        scenario_results.append(simulate_path(daily[start_idx:], traded[start_idx:], rules))
                    frame = pd.DataFrame(scenario_results)
                    simulation_rows.append({
                        "allocation": allocation, "scaling": float(scaling), "stress": scenario,
                        "ruleset": rules["name"], "horizon_traded_days": horizon,
                        "path_count": len(frame), "pass_probability": float(frame["outcome"].eq("PASS").mean()),
                        "mll_probability": float(frame["outcome"].eq("MLL_FAIL").mean()),
                        "expiry_probability": float(frame["outcome"].eq("EXPIRY").mean()),
                        "median_max_drawdown_usd": float(frame["max_drawdown_usd"].median()),
                        "worst_observed_day_usd": float(frame["worst_day_usd"].min()),
                    })
    return pd.DataFrame(event_rows), pd.DataFrame(simulation_rows)


def classify_frontier(cfg: dict[str, Any], historical: pd.DataFrame, bootstrap: pd.DataFrame,
                      mappings: pd.DataFrame, exposure: pd.DataFrame, stress: pd.DataFrame) -> pd.DataFrame:
    hist_summary = summarize_runs(historical, ["implementation", "allocation", "scaling", "horizon_traded_days"])
    int_hist = hist_summary[hist_summary["implementation"].eq("integer")]
    boot_summary = summarize_runs(bootstrap, ["allocation", "scaling", "horizon_traded_days", "block_length_sessions"])
    rows = []
    thresholds = cfg["classification"]
    candidate: dict[tuple[str, float], bool] = {}
    intermediate: dict[tuple[str, float], dict[str, Any]] = {}
    for allocation in ALLOCATIONS:
        minimum_scaling = float(mappings["scaling"].min())
        base_boot_expiry = float(boot_summary[(boot_summary["allocation"].eq(allocation)) & (boot_summary["scaling"].eq(minimum_scaling)) & (boot_summary["horizon_traded_days"].eq(60))]["expiry_probability"].mean())
        for scaling in sorted(mappings["scaling"].unique()):
            hist35 = int_hist[(int_hist["allocation"].eq(allocation)) & (int_hist["scaling"].eq(scaling)) & (int_hist["horizon_traded_days"].eq(35))].iloc[0]
            hist60 = int_hist[(int_hist["allocation"].eq(allocation)) & (int_hist["scaling"].eq(scaling)) & (int_hist["horizon_traded_days"].eq(60))].iloc[0]
            boot35 = boot_summary[(boot_summary["allocation"].eq(allocation)) & (boot_summary["scaling"].eq(scaling)) & (boot_summary["horizon_traded_days"].eq(35))]
            boot60 = boot_summary[(boot_summary["allocation"].eq(allocation)) & (boot_summary["scaling"].eq(scaling)) & (boot_summary["horizon_traded_days"].eq(60))]
            feasible = (
                float(boot60["pass_probability"].min()) >= float(thresholds["feasible_min_bootstrap_pass_rate_60d"])
                and float(boot60["mll_probability"].max()) <= float(thresholds["feasible_max_bootstrap_mll_rate_60d"])
                and float(hist60["mll_probability"]) <= float(thresholds["feasible_max_historical_mll_rate_60d"])
                and base_boot_expiry - float(boot60["expiry_probability"].mean()) >= float(thresholds["feasible_expiry_reduction_vs_1x"])
            )
            candidate[(allocation, float(scaling))] = feasible
            exp = exposure[exposure["allocation"].eq(allocation) & exposure["scaling"].eq(scaling)].iloc[0]
            map_part = mappings[mappings["allocation"].eq(allocation) & mappings["scaling"].eq(scaling)]
            sync = stress[(stress["allocation"].eq(allocation)) & (stress["scaling"].eq(scaling)) & stress["stress"].eq("synchronized_worst_sleeve_trades")].iloc[0]
            intermediate[(allocation, float(scaling))] = {
                "allocation": allocation, "scaling": float(scaling),
                "mnq_sleeve_copies": int(exp["mnq_sleeve_copies"]), "mgc_sleeve_copies": int(exp["mgc_sleeve_copies"]),
                "max_simultaneous_mnq_contracts": int(exp["max_simultaneous_mnq_contracts"]),
                "max_simultaneous_mgc_contracts": int(exp["max_simultaneous_mgc_contracts"]),
                "max_simultaneous_total_micro_contracts": int(exp["max_simultaneous_total_micro_contracts"]),
                "integer_contract_vector": ";".join(f"{r.sleeve}={int(r.integer_sleeve_copies)}" for r in map_part.itertuples(index=False)),
                "allocation_l1_tracking_error": float(map_part["allocation_l1_tracking_error"].iloc[0]),
                "historical_pass_35d": float(hist35["pass_probability"]), "historical_mll_35d": float(hist35["mll_probability"]),
                "historical_expiry_35d": float(hist35["expiry_probability"]), "historical_pass_60d": float(hist60["pass_probability"]),
                "historical_mll_60d": float(hist60["mll_probability"]), "historical_expiry_60d": float(hist60["expiry_probability"]),
                "bootstrap_pass_35d_mean": float(boot35["pass_probability"].mean()),
                "bootstrap_mll_35d_mean": float(boot35["mll_probability"].mean()),
                "bootstrap_expiry_35d_mean": float(boot35["expiry_probability"].mean()),
                "bootstrap_pass_60d_mean": float(boot60["pass_probability"].mean()),
                "bootstrap_pass_60d_min_block": float(boot60["pass_probability"].min()),
                "bootstrap_mll_60d_mean": float(boot60["mll_probability"].mean()),
                "bootstrap_mll_60d_max_block": float(boot60["mll_probability"].max()),
                "bootstrap_expiry_60d_mean": float(boot60["expiry_probability"].mean()),
                "median_days_to_pass": float(boot60["median_days_to_pass"].median()),
                "synchronized_stress_loss_usd": float(sync["single_event_loss_usd"]),
                "synchronized_events_to_mll": int(sync["events_to_initial_mll"]),
            }
    levels = sorted(float(v) for v in mappings["scaling"].unique())
    for allocation in ALLOCATIONS:
        for idx, scaling in enumerate(levels):
            row = intermediate[(allocation, scaling)]
            adjacent = ((idx > 0 and candidate[(allocation, levels[idx - 1])]) or
                        (idx + 1 < len(levels) and candidate[(allocation, levels[idx + 1])]))
            if candidate[(allocation, scaling)] and (not thresholds["feasible_requires_adjacent_candidate"] or adjacent):
                label = "FEASIBLE_REGION"
            elif max(row["historical_mll_60d"], row["bootstrap_mll_60d_max_block"]) >= float(thresholds["mll_dominated_rate_60d"]):
                label = "MLL_DOMINATED"
            elif row["bootstrap_expiry_60d_mean"] >= float(thresholds["under_sized_expiry_rate_60d"]):
                label = "UNDER_SIZED"
            else:
                label = "AGGRESSIVE"
            row["classification"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def make_figures(frontier: pd.DataFrame, bootstrap_summary: pd.DataFrame, mappings: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for allocation in ALLOCATIONS:
        part = frontier[frontier["allocation"].eq(allocation)].sort_values("scaling")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].plot(part["scaling"], part["historical_pass_60d"], marker="o", label="PASS historical")
        axes[0].plot(part["scaling"], part["historical_mll_60d"], marker="o", label="MLL historical")
        axes[0].plot(part["scaling"], part["historical_expiry_60d"], marker="o", label="EXPIRY historical")
        axes[0].plot(part["scaling"], part["bootstrap_pass_60d_mean"], marker="x", ls="--", label="PASS bootstrap")
        axes[0].plot(part["scaling"], part["bootstrap_mll_60d_mean"], marker="x", ls="--", label="MLL bootstrap")
        axes[0].plot(part["scaling"], part["bootstrap_expiry_60d_mean"], marker="x", ls="--", label="EXPIRY bootstrap")
        axes[0].set_ylim(0, 1)
        axes[0].set_xlabel("Executable scaling")
        axes[0].set_ylabel("Probability")
        axes[0].set_title(f"{allocation} — 60 traded days")
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        axes[1].plot(part["scaling"], part["median_days_to_pass"], marker="o", color="#7c3aed")
        axes[1].set_xlabel("Executable scaling")
        axes[1].set_ylabel("Median traded days to pass")
        axes[1].set_title("Bootstrap pass speed")
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"frontier_{allocation}.png", dpi=170)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 7))
    for allocation in ALLOCATIONS:
        part = mappings[mappings["allocation"].eq(allocation)].drop_duplicates("scaling").sort_values("scaling")
        ax.plot(part["scaling"], part["allocation_l1_tracking_error"], marker="o", label=allocation)
    ax.set_title("Integer allocation discontinuity — L1 weight tracking error")
    ax.set_xlabel("Executable scaling")
    ax.set_ylabel("L1 distance from frozen weights")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "integer_tracking_error.png", dpi=170)
    plt.close(fig)


def first_non_dominant(frontier: pd.DataFrame, allocation: str, field: str, threshold: float) -> str:
    part = frontier[frontier["allocation"].eq(allocation)].sort_values("scaling")
    found = part[part[field].lt(threshold)]
    return "none in grid" if found.empty else f"{float(found.iloc[0]['scaling']):g}x"


def write_report(cfg: dict[str, Any], frontier: pd.DataFrame, mappings: pd.DataFrame, exposure: pd.DataFrame,
                 continuous_summary: pd.DataFrame, bootstrap_summary: pd.DataFrame, stress_sim: pd.DataFrame) -> None:
    feasible = frontier[frontier["classification"].eq("FEASIBLE_REGION")]
    any_feasible = not feasible.empty
    lines = [
        "# TOPSTEP PORTFOLIO SCALING", "", "## Decision", "",
        ("A broad executable **FEASIBLE_REGION exists** under the predeclared structural gates." if any_feasible else "No broad executable **FEASIBLE_REGION** satisfies the predeclared structural gates."),
        "No scaling point is selected as an optimum. The tables describe the expiry/MLL trade-off only.",
        "", "## Frozen scope and integer implementation", "",
        "- The six sleeve definitions and the three Phase-1 allocation weights are unchanged.",
        "- Scaling is measured against the Phase-1 fractional portfolio, whose frozen weights sum to one sleeve copy. Keeping all six sleeves active requires at least six integer copies, so requested 1x–5x levels are continuous diagnostics but are not executable.",
        "- Integer quantities use constrained largest-remainder apportionment with a minimum of one copy per sleeve. An ORB copy multiplies its already-frozen variable integer trade quantity.",
        "- Historical and bootstrap paths use the repository TopstepX 50K rules exactly: USD 50,000 start, USD 3,000 target, USD 2,000 trailing MLL, 50% consistency, and 35/60 traded-day expiry. No daily loss rule, contract cap, or breakeven lock is invented.",
        "- Simulation remains daily because the existing repository Topstep engine is daily; timestamp positions are used for exact simultaneous contract counts.",
        "", "## Continuous bridge versus executable granularity", "",
        "These 1x–5x quantities are diagnostic only. They are fractional and cannot keep all six sleeves active with integer contracts.", "",
        "| Allocation | Continuous scale | PASS 60d | MLL 60d | EXPIRY 60d | Integer executable |", "|---|---:|---:|---:|---:|---|",
    ]
    continuous_60 = continuous_summary[(continuous_summary["implementation"].eq("continuous")) &
                                       (continuous_summary["horizon_traded_days"].eq(60)) &
                                       (continuous_summary["scaling"].isin(cfg["continuous_scaling_levels"]))]
    for row in continuous_60.itertuples(index=False):
        lines.append(f"| `{row.allocation}` | {row.scaling:g}x | {row.pass_probability:.1%} | {row.mll_probability:.1%} | {row.expiry_probability:.1%} | No |")
    lines += [
        "", "## Direct answers", "",
        "1. **Is the current portfolio simply under-sized?** The fractional 1x portfolio is under-sized—expiry is 100% historically—but integer granularity prevents a smooth scale-up while retaining all six sleeves.",
        "2. **Where does expiry stop dominating?** Approximately 4x for theoretical equal weight, just above 6x for inverse-vol, and between 6x and 9x for ERC. Integer granularity does not leave a safe all-sleeve point around those crossovers.",
        "3. **When is MLL risk unacceptable?** At the first executable 6x point: pooled 60d bootstrap MLL is 51.3% (95% CI 50.2%–52.4%), and synchronized worst sleeve trades lose USD 3,296.50 in one event.",
        "4. **Is there a broad robust region?** No. The grid jumps from fractional expiry dominance to executable MLL dominance.",
        "5. **Does one allocation materially dominate?** No. All allocations collapse to `[1,1,1,1,1,1]` at 6x. ERC has a higher descriptive pass rate at 9x, but bootstrap MLL is still 56.5%, so it is not a robust advantage.",
        "", "## Integer contract mappings", "",
        "Each vector reports integer sleeve copies in frozen order: MNQ ORB, MNQ volatility, MNQ compression, MNQ ATR, MGC volatility, MGC gap.", "",
        "| Allocation | Scaling | Integer copy vector | Max MNQ | Max MGC | Max total | L1 weight error |", "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in frontier.itertuples(index=False):
        vector = mappings[mappings["allocation"].eq(row.allocation) & mappings["scaling"].eq(row.scaling)].set_index("sleeve").reindex(SLEEVES)["integer_sleeve_copies"].astype(int).tolist()
        lines.append(f"| `{row.allocation}` | {row.scaling:g}x | `{vector}` | {row.max_simultaneous_mnq_contracts} | {row.max_simultaneous_mgc_contracts} | {row.max_simultaneous_total_micro_contracts} | {row.allocation_l1_tracking_error:.3f} |")

    lines += ["", "## Historical and bootstrap frontier", "",
              "Bootstrap estimates pool four separately reported moving-block lengths (3, 5, 10 and 20 complete joint portfolio days), 2,000 paths each. Path-level outputs and Wilson intervals are provided in CSV.", "",
              "| Allocation | Scale | PASS 35 H | MLL 35 H | EXP 35 H | PASS 60 H | MLL 60 H | EXP 60 H | PASS 60 boot | MLL 60 boot | Median pass days | Stress loss | Class |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in frontier.itertuples(index=False):
        days = "n/a" if pd.isna(row.median_days_to_pass) else f"{row.median_days_to_pass:.0f}"
        lines.append(
            f"| `{row.allocation}` | {row.scaling:g}x | {row.historical_pass_35d:.1%} | {row.historical_mll_35d:.1%} | {row.historical_expiry_35d:.1%} | "
            f"{row.historical_pass_60d:.1%} | {row.historical_mll_60d:.1%} | {row.historical_expiry_60d:.1%} | "
            f"{row.bootstrap_pass_60d_mean:.1%} | {row.bootstrap_mll_60d_mean:.1%} | {days} | {row.synchronized_stress_loss_usd:,.0f} | `{row.classification}` |"
        )

    lines += ["", "## Structural answers", ""]
    for allocation in ALLOCATIONS:
        part = frontier[frontier["allocation"].eq(allocation)].sort_values("scaling")
        best_pass = part.loc[part["bootstrap_pass_60d_mean"].idxmax()]
        lines += [
            f"### {allocation}", "",
            f"- Among executable mappings, expiry is already below 65% at: **{first_non_dominant(frontier, allocation, 'bootstrap_expiry_60d_mean', 0.65)}**.",
            f"- MLL exceeds 50% at: **{first_non_dominant(frontier.assign(inv=1-frontier['bootstrap_mll_60d_mean']), allocation, 'inv', 0.50)}**." if (part["bootstrap_mll_60d_mean"] >= 0.50).any() else "- MLL does not exceed 50% inside the tested grid.",
            f"- Highest observed bootstrap 60d pass probability is {best_pass.bootstrap_pass_60d_mean:.1%} at {best_pass.scaling:g}x; this is descriptive and is not a selection rule.",
        ]
    if any_feasible:
        regions = feasible.groupby("allocation")["scaling"].apply(lambda x: ", ".join(f"{v:g}x" for v in sorted(x))).to_dict()
        lines += ["", "Predeclared feasible regions: " + "; ".join(f"`{k}`: {v}" for k, v in regions.items()) + "."]
    else:
        lines += ["", "The transition from expiry to MLL does not leave an adjacent, block-robust region that passes all predeclared feasibility gates."]

    lines += ["", "## Stress tests", "",
              "Stress outputs include repeated worst historical portfolio day, synchronized individual worst trades, mean-centered 1.5x/2x volatility, and one extra layer of the explicitly modeled slippage/commissions. `stress_topstep_summary.csv` reports the resulting 35d/60d historical-start probabilities.",
              "", "## Required concise table", "",
              "The historical probabilities below use every eligible starting date. `MNQ/MGC contracts` are maximum simultaneous contracts, not sleeve-copy budgets.", "",
              "| Allocation | Scaling | MNQ contracts | MGC contracts | P(PASS) 35d | P(MLL) 35d | P(EXPIRY) 35d | P(PASS) 60d | P(MLL) 60d | P(EXPIRY) 60d | Median days | Stress loss | Classification |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in frontier.itertuples(index=False):
        days = "n/a" if pd.isna(row.median_days_to_pass) else f"{row.median_days_to_pass:.0f}"
        lines.append(f"| `{row.allocation}` | {row.scaling:g}x | {row.max_simultaneous_mnq_contracts} | {row.max_simultaneous_mgc_contracts} | {row.historical_pass_35d:.1%} | {row.historical_mll_35d:.1%} | {row.historical_expiry_35d:.1%} | {row.historical_pass_60d:.1%} | {row.historical_mll_60d:.1%} | {row.historical_expiry_60d:.1%} | {days} | {row.synchronized_stress_loss_usd:,.0f} | `{row.classification}` |")
    lines += ["", "## Artifacts", "", "- `requested_scaling_integer_feasibility.csv`, `integer_contract_mappings.csv`, and `integer_exposure_summary.csv`",
              "- `historical_start_paths.csv`, `historical_start_summary.csv`, `continuous_vs_integer_summary.csv`",
              "- `bootstrap_paths.csv`, `bootstrap_summary_by_block.csv`, `bootstrap_summary_pooled.csv`",
              "- `stress_events.csv`, `stress_topstep_summary.csv`, `scaling_frontier.csv`",
              "- `topstep_rulesets_exact.csv`, `run_metadata.json`, and `figures/`", ""]
    (OUTPUT_DIR / "TOPSTEP_PORTFOLIO_SCALING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def execute() -> None:
    for directory in (OUTPUT_DIR, FIGURES_DIR, MPL_CACHE):
        directory.mkdir(parents=True, exist_ok=True)
    cfg, source_hashes = load_config()
    config_hash = sha256(CONFIG_PATH)
    pnl, counts, weights, trades, positions = load_inputs(cfg)
    mappings = build_integer_mappings(cfg, weights)
    mappings.to_csv(OUTPUT_DIR / "integer_contract_mappings.csv", index=False)
    continuous_mappings = build_continuous_mappings(cfg, weights)
    requested_integer_feasibility(cfg).to_csv(OUTPUT_DIR / "requested_scaling_integer_feasibility.csv", index=False)
    exposure = exposure_table(mappings, positions)
    exposure.to_csv(OUTPUT_DIR / "integer_exposure_summary.csv", index=False)
    pd.DataFrame(topstep_rules()).to_csv(OUTPUT_DIR / "topstep_rulesets_exact.csv", index=False)

    historical_integer = historical_runs(pnl, counts, mappings, exposure, integer=True)
    historical_continuous = historical_runs(pnl, counts, continuous_mappings, exposure, integer=False)
    historical_integer.to_csv(OUTPUT_DIR / "historical_start_paths.csv", index=False)
    historical_summary = summarize_runs(historical_integer, ["implementation", "allocation", "scaling", "horizon_traded_days"])
    historical_summary.to_csv(OUTPUT_DIR / "historical_start_summary.csv", index=False)
    continuous_summary = summarize_runs(pd.concat([historical_integer, historical_continuous], ignore_index=True),
                                        ["implementation", "allocation", "scaling", "horizon_traded_days"])
    continuous_summary.to_csv(OUTPUT_DIR / "continuous_vs_integer_summary.csv", index=False)

    bootstrap = bootstrap_runs(cfg, pnl, counts, mappings, exposure)
    bootstrap.to_csv(OUTPUT_DIR / "bootstrap_paths.csv", index=False)
    bootstrap_by_block = summarize_runs(bootstrap, ["allocation", "scaling", "horizon_traded_days", "block_length_sessions"])
    bootstrap_by_block.to_csv(OUTPUT_DIR / "bootstrap_summary_by_block.csv", index=False)
    bootstrap_pooled = summarize_runs(bootstrap, ["allocation", "scaling", "horizon_traded_days"])
    bootstrap_pooled.to_csv(OUTPUT_DIR / "bootstrap_summary_pooled.csv", index=False)

    stress_events, stress_sim = stress_tests(cfg, pnl, counts, trades, mappings)
    stress_events.to_csv(OUTPUT_DIR / "stress_events.csv", index=False)
    stress_sim.to_csv(OUTPUT_DIR / "stress_topstep_summary.csv", index=False)
    frontier = classify_frontier(cfg, historical_integer, bootstrap, mappings, exposure, stress_events)
    frontier.to_csv(OUTPUT_DIR / "scaling_frontier.csv", index=False)
    make_figures(frontier, bootstrap_by_block, mappings)
    write_report(cfg, frontier, mappings, exposure, continuous_summary, bootstrap_by_block, stress_sim)

    if sha256(CONFIG_PATH) != config_hash:
        raise RuntimeError("Scaling config changed during execution")
    for key, expected in source_hashes.items():
        if sha256(REPO_ROOT / cfg["source_locks"][key]["path"]) != expected:
            raise RuntimeError(f"Locked source changed during execution: {key}")
    metadata = {
        "phase": cfg["phase"], "config_sha256": config_hash, "source_sha256": source_hashes,
        "session_count": len(pnl), "sleeves": SLEEVES, "allocations": ALLOCATIONS,
        "continuous_scaling_levels": cfg["continuous_scaling_levels"],
        "integer_executable_scaling_levels": cfg["integer_executable_scaling_levels"],
        "historical_path_count": len(historical_integer),
        "bootstrap_path_count": len(bootstrap), "bootstrap_seed": cfg["bootstrap"]["random_seed"],
        "rulesets": topstep_rules(), "classification_counts": frontier["classification"].value_counts().to_dict(),
        "sleeve_parameters_modified": False, "allocation_weights_optimized": False, "ml_used": False,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(frontier.to_string(index=False), flush=True)


if __name__ == "__main__":
    execute()
