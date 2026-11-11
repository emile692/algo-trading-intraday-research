"""Account-size compatibility analysis for the frozen six-sleeve portfolio."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = OUTPUT_DIR / "topstep_account_size_analysis_config.json"
FIGURES_DIR = OUTPUT_DIR / "figures"
MPL_CACHE = OUTPUT_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from research.topstep_portfolio_scaling.run_topstep_portfolio_scaling import (  # noqa: E402
    bootstrap_index_paths,
    daily_explicit_costs,
    valid_start_indices,
    wilson,
)


SLEEVES = [
    "mnq_orb_vwap_atr",
    "mnq_volatility_breakout",
    "mnq_compression_breakout",
    "mnq_atr_trend_flip",
    "mgc_volatility_breakout",
    "mgc_gap_continuation",
]
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
        raise RuntimeError("Configuration must be locked before execution")
    hashes: dict[str, str] = {}
    for key, item in cfg["source_locks"].items():
        actual = sha256(REPO_ROOT / item["path"])
        if actual != item["sha256"]:
            raise RuntimeError(f"Locked input hash mismatch: {key}")
        hashes[key] = actual
    return cfg, hashes


def load_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    sources = cfg["source_locks"]
    pnl = pd.read_csv(REPO_ROOT / sources["daily_sleeve_pnl"]["path"], parse_dates=["session_date"]).set_index("session_date")[SLEEVES]
    counts = pd.read_csv(REPO_ROOT / sources["daily_trade_counts"]["path"], parse_dates=["session_date"]).set_index("session_date")[SLEEVES]
    if not pnl.index.equals(counts.index) or pnl.isna().any().any() or counts.isna().any().any():
        raise RuntimeError("Frozen daily inputs are not clean and aligned")
    trades = {
        sleeve: pd.read_csv(REPO_ROOT / sources[TRADE_LOCK_KEYS[sleeve]]["path"], parse_dates=["session_date"])
        for sleeve in SLEEVES
    }
    positions = pd.read_csv(
        REPO_ROOT / sources["timestamp_portfolio"]["path"],
        usecols=["timestamp_utc"] + [f"position__{s}" for s in SLEEVES],
        parse_dates=["timestamp_utc"],
    )
    return pnl, counts, trades, positions


def account_rules(cfg: dict[str, Any], account: str, horizon: int) -> dict[str, Any]:
    base = next(row for row in cfg["account_rules"] if row["account"] == account)
    return {
        **base,
        "max_traded_days": int(horizon),
        "consistency_share_limit": float(cfg["topstep_protocol"]["consistency_share_limit"]),
        "lock_at_starting_balance": bool(cfg["topstep_protocol"]["lock_at_starting_balance"]),
    }


def simulate_path_diagnostics(pnl: np.ndarray, traded: np.ndarray, rules: dict[str, Any]) -> dict[str, Any]:
    equity = float(rules["starting_balance_usd"])
    high = equity
    best_day = 0.0
    traded_days = 0
    terminal = "EXPIRY"
    days_to_pass = np.nan
    days_to_mll = np.nan
    used = 0
    max_drawdown = 0.0
    minimum_mll_distance = float(rules["trailing_mll_usd"])
    for idx, value in enumerate(np.asarray(pnl, dtype=float)):
        used = idx + 1
        if bool(traded[idx]):
            traded_days += 1
        equity += float(value)
        high = max(high, equity)
        max_drawdown = min(max_drawdown, equity - high)
        best_day = max(best_day, max(float(value), 0.0))
        cumulative = equity - float(rules["starting_balance_usd"])
        floor = high - float(rules["trailing_mll_usd"])
        if bool(rules["lock_at_starting_balance"]):
            floor = min(floor, float(rules["starting_balance_usd"]))
        minimum_mll_distance = min(minimum_mll_distance, equity - floor)
        target_hit = cumulative >= float(rules["profit_target_usd"])
        consistent = target_hit and best_day < float(rules["consistency_share_limit"]) * cumulative
        if equity <= floor:
            terminal = "MLL_FAIL"
            days_to_mll = float(traded_days)
            break
        if consistent:
            terminal = "PASS"
            days_to_pass = float(traded_days)
            break
        if traded_days >= int(rules["max_traded_days"]):
            terminal = "EXPIRY"
            break
    return {
        "outcome": terminal,
        "days_to_pass": days_to_pass,
        "days_to_mll": days_to_mll,
        "traded_days": int(traded_days),
        "calendar_sessions": int(used),
        "terminal_pnl_usd": float(equity - float(rules["starting_balance_usd"])),
        "minimum_distance_to_mll_usd": float(minimum_mll_distance),
        "max_drawdown_usd": float(max_drawdown),
    }


def mapping_grid(cfg: dict[str, Any]) -> pd.DataFrame:
    maximum_increments = int(cfg["integer_neighbor_grid"]["maximum_sleeves_incremented_above_one"])
    rows = []
    for vector in itertools.product((1, 2), repeat=len(SLEEVES)):
        if sum(value > 1 for value in vector) > maximum_increments:
            continue
        rows.append({
            "mapping_id": "q_" + "".join(str(value) for value in vector),
            "copy_vector": ";".join(str(value) for value in vector),
            "total_sleeve_copies": int(sum(vector)),
            **{f"copies__{sleeve}": int(vector[idx]) for idx, sleeve in enumerate(SLEEVES)},
        })
    return pd.DataFrame(rows).sort_values(["total_sleeve_copies", "mapping_id"]).reset_index(drop=True)


def vector_from_mapping(row: Any) -> np.ndarray:
    return np.asarray([getattr(row, f"copies__{sleeve}") for sleeve in SLEEVES], dtype=float)


def exposure_summary(mappings: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    base = positions[[f"position__{s}" for s in SLEEVES]].to_numpy(float)
    rows = []
    for mapping in mappings.itertuples(index=False):
        q = vector_from_mapping(mapping)
        scaled = base * q
        rows.append({
            "mapping_id": mapping.mapping_id,
            "copy_vector": mapping.copy_vector,
            "max_simultaneous_mnq_contracts": int(np.abs(scaled[:, :4]).sum(axis=1).max()),
            "max_simultaneous_mgc_contracts": int(np.abs(scaled[:, 4:]).sum(axis=1).max()),
            "max_simultaneous_total_contracts": int(np.abs(scaled).sum(axis=1).max()),
            "max_net_mnq_contracts_abs": int(np.abs(scaled[:, :4].sum(axis=1)).max()),
            "max_net_mgc_contracts_abs": int(np.abs(scaled[:, 4:].sum(axis=1)).max()),
        })
    return pd.DataFrame(rows)


def daily_series(matrix: np.ndarray, costs: np.ndarray, q: np.ndarray, scenario: str) -> np.ndarray:
    base = matrix @ q
    if scenario == "base":
        return base
    if scenario == "double_costs":
        return base - costs @ q
    if scenario.startswith("volatility_"):
        multiplier = float(scenario.split("_")[1].removesuffix("x"))
        return float(base.mean()) + multiplier * (base - float(base.mean()))
    raise ValueError(f"Unknown scenario: {scenario}")


def historical_runs(cfg: dict[str, Any], pnl: pd.DataFrame, counts: pd.DataFrame, costs: np.ndarray,
                    mappings: pd.DataFrame, accounts: Iterable[str], scenarios: Iterable[str] = ("base",)) -> pd.DataFrame:
    matrix = pnl.to_numpy(float)
    traded = counts.sum(axis=1).to_numpy(float) > 0
    rows = []
    for mapping in mappings.itertuples(index=False):
        q = vector_from_mapping(mapping)
        for scenario in scenarios:
            daily = daily_series(matrix, costs, q, scenario)
            for account in accounts:
                for horizon in cfg["topstep_protocol"]["historical_horizons_traded_days"]:
                    rules = account_rules(cfg, account, int(horizon))
                    for start in valid_start_indices(traded, int(horizon)):
                        rows.append({
                            "mapping_id": mapping.mapping_id, "copy_vector": mapping.copy_vector,
                            "account": account, "scenario": scenario,
                            "horizon_traded_days": int(horizon), "start_session_date": pnl.index[start],
                            **simulate_path_diagnostics(daily[start:], traded[start:], rules),
                        })
    return pd.DataFrame(rows)


def common_bootstrap_paths(cfg: dict[str, Any], traded: np.ndarray) -> dict[int, list[np.ndarray]]:
    rng = np.random.default_rng(int(cfg["bootstrap"]["random_seed"]))
    horizon = int(cfg["topstep_protocol"]["bootstrap_horizon_traded_days"])
    return {
        int(block): bootstrap_index_paths(traded, horizon, int(block), int(cfg["bootstrap"]["paths_per_block_length"]), rng)
        for block in cfg["bootstrap"]["block_lengths_sessions"]
    }


def bootstrap_runs(cfg: dict[str, Any], pnl: pd.DataFrame, counts: pd.DataFrame, costs: np.ndarray,
                   mappings: pd.DataFrame, accounts: Iterable[str], paths: dict[int, list[np.ndarray]],
                   scenarios: Iterable[str] = ("base",)) -> pd.DataFrame:
    matrix = pnl.to_numpy(float)
    traded = counts.sum(axis=1).to_numpy(float) > 0
    horizon = int(cfg["topstep_protocol"]["bootstrap_horizon_traded_days"])
    rows = []
    for mapping in mappings.itertuples(index=False):
        q = vector_from_mapping(mapping)
        for scenario in scenarios:
            daily = daily_series(matrix, costs, q, scenario)
            for account in accounts:
                rules = account_rules(cfg, account, horizon)
                for block, block_paths in paths.items():
                    for path_id, indices in enumerate(block_paths, start=1):
                        rows.append({
                            "mapping_id": mapping.mapping_id, "copy_vector": mapping.copy_vector,
                            "account": account, "scenario": scenario, "horizon_traded_days": horizon,
                            "block_length_sessions": block, "path_id": path_id,
                            **simulate_path_diagnostics(daily[indices], traded[indices], rules),
                        })
    return pd.DataFrame(rows)


def summarize_runs(runs: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in runs.groupby(group_columns, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        total = len(group)
        for outcome, label in (("PASS", "pass"), ("MLL_FAIL", "mll"), ("EXPIRY", "expiry")):
            count = int(group["outcome"].eq(outcome).sum())
            low, high = wilson(count, total)
            row.update({f"{label}_count": count, f"{label}_probability": count / total,
                        f"{label}_ci95_low": low, f"{label}_ci95_high": high})
        passed = group[group["outcome"].eq("PASS")]
        failed = group[group["outcome"].eq("MLL_FAIL")]
        row.update({
            "path_count": total,
            "median_days_to_pass": float(passed["days_to_pass"].median()) if len(passed) else np.nan,
            "median_days_to_mll": float(failed["days_to_mll"].median()) if len(failed) else np.nan,
            "mean_attempt_traded_days": float(group["traded_days"].mean()),
            "median_attempt_traded_days": float(group["traded_days"].median()),
            "mean_attempt_sessions": float(group["calendar_sessions"].mean()),
            "median_attempt_sessions": float(group["calendar_sessions"].median()),
        })
        for column, prefix in (("terminal_pnl_usd", "terminal_pnl"),
                               ("minimum_distance_to_mll_usd", "minimum_mll_distance")):
            row.update({
                f"{prefix}_p05_usd": float(group[column].quantile(0.05)),
                f"{prefix}_p25_usd": float(group[column].quantile(0.25)),
                f"{prefix}_median_usd": float(group[column].median()),
                f"{prefix}_p75_usd": float(group[column].quantile(0.75)),
                f"{prefix}_p95_usd": float(group[column].quantile(0.95)),
                f"{prefix}_minimum_usd": float(group[column].min()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_bootstrap_summary(runs: pd.DataFrame) -> pd.DataFrame:
    return summarize_runs(runs, ["mapping_id", "copy_vector", "account", "scenario", "horizon_traded_days"])


def screen_accounts(cfg: dict[str, Any], baseline_historical_summary: pd.DataFrame,
                    baseline_bootstrap_summary: pd.DataFrame, baseline_stress_summary: pd.DataFrame) -> pd.DataFrame:
    gates = cfg["substantial_viability_screen"]
    hist = baseline_historical_summary[(baseline_historical_summary["scenario"].eq("base")) &
                                       (baseline_historical_summary["horizon_traded_days"].eq(60))].set_index("account")
    boot = baseline_bootstrap_summary[baseline_bootstrap_summary["scenario"].eq("base")].set_index("account")
    vol = baseline_stress_summary[baseline_stress_summary["scenario"].eq("volatility_1.5x")].set_index("account")
    base_h, base_b = hist.loc["50K"], boot.loc["50K"]
    rows = []
    for account in [row["account"] for row in cfg["account_rules"]]:
        qualifies = bool(
            account != "50K"
            and float(base_b.mll_probability - boot.loc[account].mll_probability) >= float(gates["minimum_bootstrap_mll_reduction_vs_50k"])
            and float(base_h.mll_probability - hist.loc[account].mll_probability) >= float(gates["minimum_historical_mll_reduction_vs_50k"])
            and float(boot.loc[account].mll_probability) <= float(gates["maximum_bootstrap_mll"])
            and float(boot.loc[account].expiry_probability) <= float(gates["maximum_bootstrap_expiry"])
        )
        rows.append({
            "account": account,
            "historical_mll_reduction_vs_50k": float(base_h.mll_probability - hist.loc[account].mll_probability),
            "bootstrap_mll_reduction_vs_50k": float(base_b.mll_probability - boot.loc[account].mll_probability),
            "bootstrap_pass_probability": float(boot.loc[account].pass_probability),
            "bootstrap_mll_probability": float(boot.loc[account].mll_probability),
            "bootstrap_expiry_probability": float(boot.loc[account].expiry_probability),
            "volatility_1_5x_mll_probability": float(vol.loc[account].mll_probability),
            "opens_integer_neighbor_grid": qualifies,
        })
    return pd.DataFrame(rows)


def classify_grid(cfg: dict[str, Any], historical_summary: pd.DataFrame, bootstrap_summary: pd.DataFrame,
                  bootstrap_by_block: pd.DataFrame, stress_summary: pd.DataFrame,
                  mappings: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    if bootstrap_summary.empty:
        return pd.DataFrame()
    gates = cfg["feasible_region_gates"]
    hist = historical_summary[(historical_summary["scenario"].eq("base")) &
                              (historical_summary["horizon_traded_days"].eq(60))].set_index(["account", "mapping_id"])
    boot = bootstrap_summary[bootstrap_summary["scenario"].eq("base")].set_index(["account", "mapping_id"])
    stress = stress_summary.set_index(["account", "mapping_id", "scenario"])
    candidates: dict[tuple[str, str], bool] = {}
    rows = []
    for account in sorted(bootstrap_summary["account"].unique()):
        for mapping in mappings.itertuples(index=False):
            key = (account, mapping.mapping_id)
            b, h = boot.loc[key], hist.loc[key]
            worst_block = float(bootstrap_by_block[
                bootstrap_by_block["account"].eq(account) &
                bootstrap_by_block["mapping_id"].eq(mapping.mapping_id) &
                bootstrap_by_block["scenario"].eq("base")
            ]["mll_probability"].max())
            candidate = bool(
                float(b.pass_probability) >= float(gates["minimum_bootstrap_pass"])
                and float(b.mll_probability) <= float(gates["maximum_bootstrap_mll"])
                and float(b.expiry_probability) <= float(gates["maximum_bootstrap_expiry"])
                and worst_block <= float(gates["maximum_worst_block_bootstrap_mll"])
                and float(h.mll_probability) <= float(gates["maximum_historical_mll"])
                and float(stress.loc[(account, mapping.mapping_id, "double_costs")].mll_probability) <= float(gates["maximum_double_costs_mll"])
                and float(stress.loc[(account, mapping.mapping_id, "volatility_1.5x")].mll_probability) <= float(gates["maximum_volatility_1_5x_mll"])
                and float(stress.loc[(account, mapping.mapping_id, "volatility_2x")].mll_probability) <= float(gates["maximum_volatility_2x_mll"])
            )
            candidates[key] = candidate
            exp = exposure[exposure["mapping_id"].eq(mapping.mapping_id)].iloc[0]
            rows.append({
                "account": account, "mapping_id": mapping.mapping_id, "copy_vector": mapping.copy_vector,
                "total_sleeve_copies": mapping.total_sleeve_copies,
                "historical_pass_60d": h.pass_probability, "historical_mll_60d": h.mll_probability,
                "historical_expiry_60d": h.expiry_probability,
                "bootstrap_pass_60d": b.pass_probability, "bootstrap_pass_ci95_low": b.pass_ci95_low,
                "bootstrap_pass_ci95_high": b.pass_ci95_high, "bootstrap_mll_60d": b.mll_probability,
                "bootstrap_expiry_60d": b.expiry_probability, "worst_block_bootstrap_mll_60d": worst_block,
                "median_days_to_pass": b.median_days_to_pass, "median_days_to_mll": b.median_days_to_mll,
                "double_costs_mll_60d": stress.loc[(account, mapping.mapping_id, "double_costs")].mll_probability,
                "volatility_1_5x_mll_60d": stress.loc[(account, mapping.mapping_id, "volatility_1.5x")].mll_probability,
                "volatility_2x_mll_60d": stress.loc[(account, mapping.mapping_id, "volatility_2x")].mll_probability,
                "max_simultaneous_mnq_contracts": exp.max_simultaneous_mnq_contracts,
                "max_simultaneous_mgc_contracts": exp.max_simultaneous_mgc_contracts,
                "max_simultaneous_total_contracts": exp.max_simultaneous_total_contracts,
                "gate_candidate": candidate,
            })
    result = pd.DataFrame(rows)
    vectors = {row.mapping_id: vector_from_mapping(row) for row in mappings.itertuples(index=False)}
    robust = []
    for row in result.itertuples(index=False):
        neighbor = any(
            other != row.mapping_id and candidates.get((row.account, other), False)
            and int(np.abs(vectors[row.mapping_id] - vector).sum()) == 1
            for other, vector in vectors.items()
        )
        robust.append(bool(row.gate_candidate and (neighbor or not gates["requires_l1_adjacent_candidate"])))
    result["feasible_region"] = robust
    result["classification"] = np.where(result["feasible_region"], "FEASIBLE_REGION",
        np.where(result["bootstrap_mll_60d"] > 0.50, "MLL_DOMINATED",
        np.where(result["bootstrap_expiry_60d"] > 0.50, "EXPIRY_DOMINATED", "TRADEOFF")))
    return result


def economics(cfg: dict[str, Any], baseline_bootstrap: pd.DataFrame) -> pd.DataFrame:
    econ = cfg["economics"]
    rows = []
    for account, group in baseline_bootstrap[baseline_bootstrap["scenario"].eq("base")].groupby("account"):
        observed_p = float(group["outcome"].eq("PASS").mean())
        billing_cycles = np.maximum(1, np.ceil(group["calendar_sessions"].to_numpy(float) /
                                               float(econ["trading_sessions_per_billing_month"])))
        extra_cycles = float(np.mean(billing_cycles - 1.0))
        monthly = float(econ["standard_path_monthly_usd"][account])
        reset = float(econ["standard_path_reset_usd"][account])
        for pass_multiplier in econ["pass_probability_sensitivity_multipliers"]:
            p = min(max(observed_p * float(pass_multiplier), 1e-9), 1.0)
            expected_attempts = 1.0 / p
            expected_failures = (1.0 - p) / p
            for reset_multiplier in econ["reset_price_sensitivity_multipliers"]:
                combine_cost = (monthly + expected_failures * reset * float(reset_multiplier)
                                + extra_cycles * monthly * expected_attempts)
                expected_credits = extra_cycles * expected_attempts
                paid_resets_after_credits = max(expected_failures - expected_credits, 0.0)
                credit_netted_cost = (monthly + paid_resets_after_credits * reset * float(reset_multiplier)
                                      + extra_cycles * monthly * expected_attempts)
                rows.append({
                    "account": account, "observed_pass_probability": observed_p,
                    "pass_probability_multiplier": float(pass_multiplier), "modeled_pass_probability": p,
                    "reset_price_multiplier": float(reset_multiplier), "monthly_fee_usd": monthly,
                    "reset_fee_usd": reset, "expected_attempts_to_pass": expected_attempts,
                    "expected_failures_before_pass": expected_failures,
                    "mean_attempt_traded_days": float(group["traded_days"].mean()),
                    "median_attempt_traded_days": float(group["traded_days"].median()),
                    "mean_attempt_sessions": float(group["calendar_sessions"].mean()),
                    "median_attempt_sessions": float(group["calendar_sessions"].median()),
                    "mean_billing_cycles_per_attempt": float(billing_cycles.mean()),
                    "expected_reset_credits_to_pass": expected_credits,
                    "expected_paid_resets_after_credits": paid_resets_after_credits,
                    "expected_combine_cost_to_pass_credit_netted_usd": credit_netted_cost,
                    "expected_combine_cost_to_pass_all_resets_paid_usd": combine_cost,
                    "activation_fee_after_pass_usd": float(econ["standard_path_activation_fee_after_pass_usd"]),
                    "expected_total_through_activation_credit_netted_usd": credit_netted_cost + float(econ["standard_path_activation_fee_after_pass_usd"]),
                    "expected_total_through_activation_all_resets_paid_usd": combine_cost + float(econ["standard_path_activation_fee_after_pass_usd"]),
                })
    return pd.DataFrame(rows)


def make_figures(baseline_bootstrap: pd.DataFrame, baseline_summary: pd.DataFrame,
                 frontier: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    base = baseline_summary[baseline_summary["scenario"].eq("base")].set_index("account").reindex(["50K", "100K", "150K"])
    fig, ax = plt.subplots(figsize=(10, 6))
    base[["pass_probability", "mll_probability", "expiry_probability"]].plot.bar(ax=ax)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Frozen [1,1,1,1,1,1] - bootstrap 60d")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_account_outcomes.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    data = [baseline_bootstrap[(baseline_bootstrap["account"].eq(account)) &
                               (baseline_bootstrap["scenario"].eq("base"))]["terminal_pnl_usd"]
            for account in ("50K", "100K", "150K")]
    ax.boxplot(data, tick_labels=["50K", "100K", "150K"], showfliers=False)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Terminal PnL (USD)")
    ax.set_title("Terminal PnL distribution - bootstrap 60d")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_terminal_pnl_distribution.png", dpi=170)
    plt.close(fig)

    if not frontier.empty:
        for account, group in frontier.groupby("account"):
            fig, ax = plt.subplots(figsize=(9, 6))
            colors = np.where(group["feasible_region"], "#16a34a", "#64748b")
            ax.scatter(group["bootstrap_expiry_60d"], group["bootstrap_mll_60d"],
                       c=colors, s=55, alpha=0.85)
            ax.set_xlabel("Bootstrap EXPIRY probability")
            ax.set_ylabel("Bootstrap MLL probability")
            ax.set_title(f"{account} local integer mappings: expiry / MLL")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / f"grid_tradeoff_{account}.png", dpi=170)
            plt.close(fig)


def write_report(cfg: dict[str, Any], baseline_h: pd.DataFrame, baseline_b: pd.DataFrame,
                 baseline_stress: pd.DataFrame, screen: pd.DataFrame, frontier: pd.DataFrame,
                 economics_table: pd.DataFrame) -> None:
    h35 = baseline_h[(baseline_h["scenario"].eq("base")) & (baseline_h["horizon_traded_days"].eq(35))].set_index("account")
    h60 = baseline_h[(baseline_h["scenario"].eq("base")) & (baseline_h["horizon_traded_days"].eq(60))].set_index("account")
    b60 = baseline_b[baseline_b["scenario"].eq("base")].set_index("account")
    stress = baseline_stress.set_index(["account", "scenario"])
    feasible = frontier[frontier["feasible_region"]] if not frontier.empty else frontier
    opened = screen[screen["opens_integer_neighbor_grid"]]["account"].tolist()
    lines = [
        "# TOPSTEP ACCOUNT SIZE COMPATIBILITY", "", "## Decision", "",
        ("Une FEASIBLE_REGION robuste existe dans la grille locale." if len(feasible) else
         "Aucune FEASIBLE_REGION robuste n'apparait sous les seuils predeclares."),
        "Aucun point n'est selectionne sur la base du meilleur PASS ponctuel.",
        "", "## Perimetre gele", "",
        "- Les six sleeves, leurs signaux, stops, filtres, entrees/sorties et sizings internes restent inchanges.",
        "- Le benchmark est `[1,1,1,1,1,1]`, sans governor. Les simulations restent journalieres, comme le moteur Topstep existant du repository.",
        "- Targets/MLL simules: 50K 3,000/2,000 USD; 100K 6,000/3,000 USD; 150K 9,000/4,500 USD. Le 4,500 USD du 150K est l'hypothese imposee par cette etude.",
        "- Historique: toutes les dates de depart eligibles sur 35/60 jours trades. Bootstrap 60j: 2,000 trajectoires communes pour chacun des blocs 3/5/10/20.",
        "- `EXPIRY` est ici le cutoff de recherche 35/60j du protocole precedent. L'abonnement Topstep actuel n'a pas de limite de temps pour passer et continue a rebiller; l'analyse economique traite donc ce cutoff comme une tentative/reinitialisation comparable, pas comme une cloture commerciale automatique.",
        "", "## Benchmark par taille de compte", "",
        "| Compte | PASS/MLL/EXP H35 | PASS/MLL/EXP H60 | PASS B60 [IC95%] | MLL B60 [IC95%] | EXP B60 [IC95%] | Med. PASS/MLL | PnL terminal median [p05,p95] | Distance MLL median [p05,p95] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for account in ("50K", "100K", "150K"):
        h_short, h, b = h35.loc[account], h60.loc[account], b60.loc[account]
        lines.append(
            f"| {account} | {h_short.pass_probability:.1%}/{h_short.mll_probability:.1%}/{h_short.expiry_probability:.1%} | "
            f"{h.pass_probability:.1%}/{h.mll_probability:.1%}/{h.expiry_probability:.1%} | "
            f"{b.pass_probability:.1%} [{b.pass_ci95_low:.1%},{b.pass_ci95_high:.1%}] | "
            f"{b.mll_probability:.1%} [{b.mll_ci95_low:.1%},{b.mll_ci95_high:.1%}] | "
            f"{b.expiry_probability:.1%} [{b.expiry_ci95_low:.1%},{b.expiry_ci95_high:.1%}] | "
            f"{b.median_days_to_pass:.0f}/{b.median_days_to_mll:.0f} | {b.terminal_pnl_median_usd:,.0f} [{b.terminal_pnl_p05_usd:,.0f},{b.terminal_pnl_p95_usd:,.0f}] | "
            f"{b.minimum_mll_distance_median_usd:,.0f} [{b.minimum_mll_distance_p05_usd:,.0f},{b.minimum_mll_distance_p95_usd:,.0f}] |"
        )
    lines += ["", "Une distance MLL negative signifie que le controle journalier a franchi la limite; sa valeur absolue mesure le depassement au constat."]
    lines += ["", "## Stress bootstrap 60j", "",
              "| Compte | Costs x2 PASS/MLL/EXP | Vol x1.5 PASS/MLL/EXP | Vol x2 PASS/MLL/EXP |",
              "|---|---:|---:|---:|"]
    for account in ("50K", "100K", "150K"):
        cells = []
        for scenario in ("double_costs", "volatility_1.5x", "volatility_2x"):
            r = stress.loc[(account, scenario)]
            cells.append(f"{r.pass_probability:.1%}/{r.mll_probability:.1%}/{r.expiry_probability:.1%}")
        lines.append(f"| {account} | " + " | ".join(cells) + " |")
    lines += ["", "## Ouverture conditionnelle de la grille", "",
              "Seuils nominaux predeclares: reduction MLL bootstrap >=10 points et historique >=5 points vs 50K; MLL bootstrap <=40%; EXPIRY <=60%. Les stress ne servent pas a ouvrir la grille; ils restent des vetos obligatoires de la FEASIBLE_REGION.", "",
              "| Compte | Reduction MLL H/B | PASS B | MLL B | EXP B | Vol1.5 MLL | Grille ouverte |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for row in screen.itertuples(index=False):
        lines.append(f"| {row.account} | {row.historical_mll_reduction_vs_50k:.1%}/{row.bootstrap_mll_reduction_vs_50k:.1%} | {row.bootstrap_pass_probability:.1%} | {row.bootstrap_mll_probability:.1%} | {row.bootstrap_expiry_probability:.1%} | {row.volatility_1_5x_mll_probability:.1%} | {'Oui' if row.opens_integer_neighbor_grid else 'Non'} |")
    lines += ["", "Grille ouverte pour: " + (", ".join(opened) if opened else "aucun compte") + "."]
    if not frontier.empty:
        lines += ["", "## Grille entiere locale", "",
                  "La grille contient 22 mappings [1,2]^6 avec au plus deux sleeves augmentees. Une FEASIBLE_REGION exige aussi un mapping L1-voisin admissible.", "",
                  f"Mappings en FEASIBLE_REGION: {len(feasible)}/{len(frontier)}.", "",
                  "| Compte | Mapping | PASS B | MLL B | EXP B | Costs2 MLL | Vol1.5 MLL | Vol2 MLL | Classe |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---|"]
        display = frontier.sort_values(["feasible_region", "bootstrap_pass_60d"], ascending=[False, False]).head(15)
        for row in display.itertuples(index=False):
            lines.append(f"| {row.account} | `{row.copy_vector}` | {row.bootstrap_pass_60d:.1%} | {row.bootstrap_mll_60d:.1%} | {row.bootstrap_expiry_60d:.1%} | {row.double_costs_mll_60d:.1%} | {row.volatility_1_5x_mll_60d:.1%} | {row.volatility_2x_mll_60d:.1%} | `{row.classification}` |")
    lines += ["", "## Economie du Combine", "",
              "Tarifs Standard Path verifies le 2026-09-28: 49/99/199 USD par mois et par reset pour 50K/100K/150K. L'abonnement rebille tous les 30 jours et chaque rebill credite un reset. Le modele est approximatif: 21 sessions par cycle, tentatives i.i.d., prix catalogue, hors promotions; il ajoute 149 USD d'activation seulement apres le PASS. La fourchette va de l'utilisation integrale des credits disponibles a l'hypothese conservatrice ou tous les resets sont payes.", "",
              "| Compte | Mensuel | P(PASS) | Duree tentative moyenne/mediane | Tentatives attendues | Echecs attendus | Cout Combine jusqu'au PASS credits/resets payes | Avec activation credits/resets payes |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    central = economics_table[(economics_table["pass_probability_multiplier"].eq(1.0)) &
                              (economics_table["reset_price_multiplier"].eq(1.0))]
    for account in ("50K", "100K", "150K"):
        row = central[central["account"].eq(account)].iloc[0]
        lines.append(f"| {row.account} | ${row.monthly_fee_usd:,.0f} | {row.observed_pass_probability:.1%} | {row.mean_attempt_traded_days:.1f}/{row.median_attempt_traded_days:.0f} j trades | {row.expected_attempts_to_pass:.2f} | {row.expected_failures_before_pass:.2f} | ${row.expected_combine_cost_to_pass_credit_netted_usd:,.0f}/${row.expected_combine_cost_to_pass_all_resets_paid_usd:,.0f} | ${row.expected_total_through_activation_credit_netted_usd:,.0f}/${row.expected_total_through_activation_all_resets_paid_usd:,.0f} |")
    lines += ["", "La sensibilite complete aux probabilites de PASS (x0.8/x1/x1.2) et prix de reset (x0.5/x1/x1.5) est dans `economic_sensitivity.csv`.",
              "", "Sources tarifaires: [pricing](https://help.topstep.com/en/articles/14289835-topstep-pricing-and-payment-questions), [subscriptions](https://help.topstep.com/en/articles/8284121-trading-combine-subscriptions), [resets](https://help.topstep.com/en/articles/8284128-what-is-a-reset).",
              "", "## Reponse finale", ""]
    if len(feasible):
        accounts = ", ".join(sorted(feasible["account"].unique()))
        lines.append(f"Le changement de compte cree une region localement robuste pour {accounts}; validation prospective encore requise.")
    else:
        lines.append("Le changement de compte ne transforme pas les six sleeves en portefeuille Topstep robuste sous ces criteres: il deplace principalement le risque du MLL vers l'expiry/profit target, ou la remise a l'echelle recree le MLL sous stress.")
    lines += ["", "## Artefacts", "",
              "- `baseline_historical_paths.csv`, `baseline_historical_summary.csv`",
              "- `baseline_bootstrap_paths.csv`, `baseline_bootstrap_summary_by_block.csv`, `baseline_bootstrap_summary_pooled.csv`",
              "- `baseline_stress_bootstrap_summary.csv`, `account_viability_screen.csv`",
              "- `integer_neighbor_mappings.csv`, `integer_neighbor_exposure.csv`, `grid_frontier.csv`",
              "- `grid_historical_summary.csv`, `grid_bootstrap_summary_by_block.csv`, `grid_bootstrap_summary_pooled.csv`, `grid_stress_bootstrap_summary.csv`",
              "- `economic_sensitivity.csv`, `run_metadata.json`, `figures/`", ""]
    (OUTPUT_DIR / "TOPSTEP_ACCOUNT_SIZE_COMPATIBILITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def execute() -> None:
    for directory in (OUTPUT_DIR, FIGURES_DIR, MPL_CACHE):
        directory.mkdir(parents=True, exist_ok=True)
    cfg, hashes = load_config()
    pnl, counts, trades, positions = load_inputs(cfg)
    costs = daily_explicit_costs(trades, pnl.index).to_numpy(float)
    traded = counts.sum(axis=1).to_numpy(float) > 0
    paths = common_bootstrap_paths(cfg, traded)
    mappings = mapping_grid(cfg)
    mappings.to_csv(OUTPUT_DIR / "integer_neighbor_mappings.csv", index=False)
    exposure = exposure_summary(mappings, positions)
    exposure.to_csv(OUTPUT_DIR / "integer_neighbor_exposure.csv", index=False)
    baseline_mapping = mappings[mappings["mapping_id"].eq("q_111111")]
    accounts = [row["account"] for row in cfg["account_rules"]]

    baseline_h = historical_runs(cfg, pnl, counts, costs, baseline_mapping, accounts)
    baseline_h.to_csv(OUTPUT_DIR / "baseline_historical_paths.csv", index=False)
    baseline_h_summary = summarize_runs(baseline_h, ["mapping_id", "copy_vector", "account", "scenario", "horizon_traded_days"])
    baseline_h_summary.to_csv(OUTPUT_DIR / "baseline_historical_summary.csv", index=False)

    baseline_b = bootstrap_runs(cfg, pnl, counts, costs, baseline_mapping, accounts, paths)
    baseline_b.to_csv(OUTPUT_DIR / "baseline_bootstrap_paths.csv", index=False)
    baseline_b_by_block = summarize_runs(baseline_b, ["mapping_id", "copy_vector", "account", "scenario", "horizon_traded_days", "block_length_sessions"])
    baseline_b_by_block.to_csv(OUTPUT_DIR / "baseline_bootstrap_summary_by_block.csv", index=False)
    baseline_b_summary = pooled_bootstrap_summary(baseline_b)
    baseline_b_summary.to_csv(OUTPUT_DIR / "baseline_bootstrap_summary_pooled.csv", index=False)

    stress_scenarios = ["double_costs", "volatility_1.5x", "volatility_2x"]
    baseline_stress_paths = bootstrap_runs(cfg, pnl, counts, costs, baseline_mapping, accounts, paths, stress_scenarios)
    baseline_stress_summary = pooled_bootstrap_summary(baseline_stress_paths)
    baseline_stress_summary.to_csv(OUTPUT_DIR / "baseline_stress_bootstrap_summary.csv", index=False)
    screen = screen_accounts(cfg, baseline_h_summary, baseline_b_summary, baseline_stress_summary)
    screen.to_csv(OUTPUT_DIR / "account_viability_screen.csv", index=False)
    qualified = screen[screen["opens_integer_neighbor_grid"]]["account"].tolist()

    if qualified:
        grid_h = historical_runs(cfg, pnl, counts, costs, mappings, qualified)
        grid_h_summary = summarize_runs(grid_h, ["mapping_id", "copy_vector", "account", "scenario", "horizon_traded_days"])
        grid_b = bootstrap_runs(cfg, pnl, counts, costs, mappings, qualified, paths)
        grid_b_by_block = summarize_runs(grid_b, ["mapping_id", "copy_vector", "account", "scenario", "horizon_traded_days", "block_length_sessions"])
        grid_b_summary = pooled_bootstrap_summary(grid_b)
        grid_stress_paths = bootstrap_runs(cfg, pnl, counts, costs, mappings, qualified, paths, stress_scenarios)
        grid_stress_summary = pooled_bootstrap_summary(grid_stress_paths)
        frontier = classify_grid(cfg, grid_h_summary, grid_b_summary, grid_b_by_block,
                                 grid_stress_summary, mappings, exposure)
    else:
        grid_h_summary = baseline_h_summary.iloc[0:0].copy()
        grid_b_by_block = baseline_b_by_block.iloc[0:0].copy()
        grid_b_summary = baseline_b_summary.iloc[0:0].copy()
        grid_stress_summary = baseline_stress_summary.iloc[0:0].copy()
        frontier = pd.DataFrame(columns=[
            "account", "mapping_id", "copy_vector", "total_sleeve_copies",
            "historical_pass_60d", "historical_mll_60d", "historical_expiry_60d",
            "bootstrap_pass_60d", "bootstrap_pass_ci95_low", "bootstrap_pass_ci95_high",
            "bootstrap_mll_60d", "bootstrap_expiry_60d", "worst_block_bootstrap_mll_60d",
            "median_days_to_pass", "median_days_to_mll", "double_costs_mll_60d",
            "volatility_1_5x_mll_60d", "volatility_2x_mll_60d",
            "max_simultaneous_mnq_contracts", "max_simultaneous_mgc_contracts",
            "max_simultaneous_total_contracts", "gate_candidate", "feasible_region", "classification",
        ])
    grid_h_summary.to_csv(OUTPUT_DIR / "grid_historical_summary.csv", index=False)
    grid_b_by_block.to_csv(OUTPUT_DIR / "grid_bootstrap_summary_by_block.csv", index=False)
    grid_b_summary.to_csv(OUTPUT_DIR / "grid_bootstrap_summary_pooled.csv", index=False)
    grid_stress_summary.to_csv(OUTPUT_DIR / "grid_stress_bootstrap_summary.csv", index=False)
    frontier.to_csv(OUTPUT_DIR / "grid_frontier.csv", index=False)

    econ = economics(cfg, baseline_b)
    econ.to_csv(OUTPUT_DIR / "economic_sensitivity.csv", index=False)
    make_figures(baseline_b, baseline_b_summary, frontier)
    write_report(cfg, baseline_h_summary, baseline_b_summary, baseline_stress_summary, screen, frontier, econ)
    metadata = {
        "phase": cfg["phase"], "config_sha256": sha256(CONFIG_PATH), "source_sha256": hashes,
        "account_count": len(accounts), "baseline_historical_path_count": len(baseline_h),
        "baseline_bootstrap_path_count": len(baseline_b), "qualified_grid_accounts": qualified,
        "integer_mapping_count": len(mappings), "grid_frontier_rows": len(frontier),
        "feasible_region_rows": int(frontier["feasible_region"].sum()) if not frontier.empty else 0,
        "bootstrap_seed": cfg["bootstrap"]["random_seed"], "sleeves_modified": False,
        "governor_used": False, "ml_used": False,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(screen.to_string(index=False), flush=True)
    if not frontier.empty:
        print(frontier.sort_values(["feasible_region", "bootstrap_pass_60d"], ascending=[False, False]).head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    execute()
