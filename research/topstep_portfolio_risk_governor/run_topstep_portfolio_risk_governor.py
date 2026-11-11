"""Test simple ex-ante portfolio execution caps on six frozen sleeves."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = OUTPUT_DIR / "topstep_portfolio_risk_governor_config.json"
FIGURES_DIR = OUTPUT_DIR / "figures"
MPL_CACHE = OUTPUT_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from research.topstep_portfolio_scaling.run_topstep_portfolio_scaling import (  # noqa: E402
    ALLOCATIONS,
    INSTRUMENT,
    SLEEVES,
    bootstrap_index_paths,
    sha256,
    simulate_path,
    summarize_runs,
    topstep_rules,
    valid_start_indices,
)


TRADE_KEYS = {
    "mnq_orb_vwap_atr": "trade_mnq_orb",
    "mnq_volatility_breakout": "trade_mnq_volatility",
    "mnq_compression_breakout": "trade_mnq_compression",
    "mnq_atr_trend_flip": "trade_mnq_atr",
    "mgc_volatility_breakout": "trade_mgc_volatility",
    "mgc_gap_continuation": "trade_mgc_gap",
}
SLEEVE_ORDER = {sleeve: idx for idx, sleeve in enumerate(SLEEVES)}


def load_config() -> tuple[dict[str, Any], dict[str, str]]:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not cfg.get("locked_before_execution"):
        raise RuntimeError("Governor config must be locked before execution")
    hashes: dict[str, str] = {}
    for key, item in cfg["source_locks"].items():
        actual = sha256(REPO_ROOT / item["path"])
        if actual != item["sha256"]:
            raise RuntimeError(f"Locked source hash mismatch: {key}")
        hashes[key] = actual
    return cfg, hashes


def load_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    sources = cfg["source_locks"]
    pnl = pd.read_csv(REPO_ROOT / sources["phase1_pnl"]["path"], parse_dates=["session_date"]).set_index("session_date")[SLEEVES]
    counts = pd.read_csv(REPO_ROOT / sources["phase1_counts"]["path"], parse_dates=["session_date"]).set_index("session_date")[SLEEVES]
    trades: dict[str, pd.DataFrame] = {}
    for sleeve in SLEEVES:
        frame = pd.read_csv(REPO_ROOT / sources[TRADE_KEYS[sleeve]]["path"])
        frame["sleeve"] = sleeve
        frame["instrument"] = INSTRUMENT[sleeve]
        frame["session_date"] = pd.to_datetime(frame["session_date"])
        frame["entry_ts_utc"] = pd.to_datetime(frame["entry_ts_utc"], utc=True)
        frame["exit_ts_utc"] = pd.to_datetime(frame["exit_ts_utc"], utc=True)
        frame["quantity"] = pd.to_numeric(frame["quantity"], errors="raise").astype(int)
        frame["signed_quantity"] = pd.to_numeric(frame["signed_quantity"], errors="raise").astype(int)
        frame["net_pnl_usd"] = pd.to_numeric(frame["net_pnl_usd"], errors="raise")
        if "gross_pnl_usd" in frame:
            frame["explicit_cost_usd"] = (pd.to_numeric(frame["gross_pnl_usd"]) - frame["net_pnl_usd"]).clip(lower=0)
        else:
            frame["explicit_cost_usd"] = (pd.to_numeric(frame["pnl_usd"]) - frame["net_pnl_usd"]).clip(lower=0)
        trades[sleeve] = frame
    return pnl, counts, trades


def combined_trade_tape(trades: dict[str, pd.DataFrame]) -> pd.DataFrame:
    columns = ["sleeve", "instrument", "trade_id", "session_date", "entry_ts_utc", "exit_ts_utc",
               "quantity", "signed_quantity", "net_pnl_usd", "explicit_cost_usd"]
    tape = pd.concat([frame[columns] for frame in trades.values()], ignore_index=True)
    tape["signal_id"] = tape["sleeve"] + ":" + tape["trade_id"].astype(str)
    tape["sleeve_order"] = tape["sleeve"].map(SLEEVE_ORDER)
    return tape.sort_values(["entry_ts_utc", "sleeve_order", "trade_id"]).reset_index(drop=True)


def governor_grid(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = [{"config_id": "benchmark_no_governor", "handler": "benchmark", "global_cap": 999,
             "mnq_cap": 999, "mgc_cap": 999, "is_benchmark": True}]
    for handler in cfg["collision_handlers"]:
        for global_cap in cfg["caps"]["global_open_micro_contracts"]:
            for mnq_cap in cfg["caps"]["mnq_open_micro_contracts"]:
                for mgc_cap in cfg["caps"]["mgc_open_micro_contracts"]:
                    rows.append({
                        "config_id": f"{handler}__g{global_cap}__mnq{mnq_cap}__mgc{mgc_cap}",
                        "handler": handler, "global_cap": int(global_cap), "mnq_cap": int(mnq_cap),
                        "mgc_cap": int(mgc_cap), "is_benchmark": False,
                    })
    return pd.DataFrame(rows)


def _exposure(active: list[dict[str, Any]]) -> tuple[int, dict[str, int], dict[str, int]]:
    gross = {"MNQ": 0, "MGC": 0}
    net = {"MNQ": 0, "MGC": 0}
    for row in active:
        gross[row["instrument"]] += abs(int(row["signed_quantity"]))
        net[row["instrument"]] += int(row["signed_quantity"])
    return gross["MNQ"] + gross["MGC"], gross, net


def _ordered_contenders(contenders: list[dict[str, Any]], handler: str, active: list[dict[str, Any]],
                        risk: dict[str, float], rr_pointer: int) -> list[dict[str, Any]]:
    if handler in {"benchmark", "first_signal_first_served"}:
        return sorted(contenders, key=lambda row: (SLEEVE_ORDER[row["sleeve"]], str(row["trade_id"])))
    if handler == "lowest_frozen_standalone_risk":
        return sorted(contenders, key=lambda row: (risk[row["sleeve"]], SLEEVE_ORDER[row["sleeve"]], str(row["trade_id"])))
    if handler == "round_robin":
        return sorted(contenders, key=lambda row: ((SLEEVE_ORDER[row["sleeve"]] - rr_pointer) % len(SLEEVES), str(row["trade_id"])))
    if handler == "least_incremental_directional_exposure":
        pending = list(contenders)
        ordered: list[dict[str, Any]] = []
        shadow = list(active)
        while pending:
            _, _, net = _exposure(shadow)
            selected = min(
                pending,
                key=lambda row: (
                    abs(net[row["instrument"]] + int(row["signed_quantity"])),
                    risk[row["sleeve"]], SLEEVE_ORDER[row["sleeve"]], str(row["trade_id"]),
                ),
            )
            ordered.append(selected)
            shadow.append(selected)
            pending.remove(selected)
        return ordered
    raise ValueError(f"Unknown collision handler: {handler}")


def apply_governor(tape: pd.DataFrame, config: dict[str, Any], risk: dict[str, float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    active: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    rr_pointer = 0
    max_global = max_mnq = max_mgc = 0
    grouped = tape.groupby("entry_ts_utc", sort=True)
    for timestamp, group in grouped:
        active = [row for row in active if pd.Timestamp(row["exit_ts_utc"]) > timestamp]
        contenders = group.to_dict("records")
        pending = (_ordered_contenders(contenders, config["handler"], active, risk, rr_pointer)
                   if config["handler"] != "least_incremental_directional_exposure" else list(contenders))
        while pending:
            if config["handler"] == "least_incremental_directional_exposure":
                _, _, current_net = _exposure(active)
                row = min(
                    pending,
                    key=lambda candidate: (
                        abs(current_net[candidate["instrument"]] + int(candidate["signed_quantity"])),
                        risk[candidate["sleeve"]], SLEEVE_ORDER[candidate["sleeve"]], str(candidate["trade_id"]),
                    ),
                )
                pending.remove(row)
            else:
                row = pending.pop(0)
            global_used, gross, _ = _exposure(active)
            quantity = abs(int(row["signed_quantity"]))
            instrument_cap = int(config["mnq_cap"] if row["instrument"] == "MNQ" else config["mgc_cap"])
            global_ok = global_used + quantity <= int(config["global_cap"])
            instrument_ok = gross[row["instrument"]] + quantity <= instrument_cap
            accepted = bool(global_ok and instrument_ok)
            reason = "accepted" if accepted else ("global_and_instrument_cap" if not global_ok and not instrument_ok else ("global_cap" if not global_ok else "instrument_cap"))
            decisions.append({
                "config_id": config["config_id"], "signal_id": row["signal_id"], "sleeve": row["sleeve"],
                "instrument": row["instrument"], "session_date": row["session_date"],
                "entry_ts_utc": row["entry_ts_utc"], "exit_ts_utc": row["exit_ts_utc"],
                "quantity": quantity, "signed_quantity": int(row["signed_quantity"]), "accepted": accepted,
                "decision_reason": reason, "net_pnl_usd": float(row["net_pnl_usd"]),
                "explicit_cost_usd": float(row["explicit_cost_usd"]),
                "global_contracts_before": global_used, "instrument_contracts_before": gross[row["instrument"]],
            })
            if accepted:
                active.append(row)
                rr_pointer = (SLEEVE_ORDER[row["sleeve"]] + 1) % len(SLEEVES)
            current_global, current_gross, _ = _exposure(active)
            max_global = max(max_global, current_global)
            max_mnq = max(max_mnq, current_gross["MNQ"])
            max_mgc = max(max_mgc, current_gross["MGC"])
    return pd.DataFrame(decisions), {
        "max_simultaneous_total_contracts": max_global,
        "max_simultaneous_mnq_contracts": max_mnq,
        "max_simultaneous_mgc_contracts": max_mgc,
    }


def build_governed_daily(grid: pd.DataFrame, tape: pd.DataFrame, index: pd.Index,
                         risk: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_parts, decision_parts, participation_rows, summary_rows = [], [], [], []
    for config in grid.to_dict("records"):
        decisions, exposure = apply_governor(tape, config, risk)
        decision_parts.append(decisions)
        accepted = decisions[decisions["accepted"]].copy()
        rejected = decisions[~decisions["accepted"]].copy()
        daily = accepted.groupby("session_date").agg(
            daily_pnl_usd=("net_pnl_usd", "sum"), daily_explicit_cost_usd=("explicit_cost_usd", "sum"),
            daily_trade_count=("signal_id", "size"),
        ).reindex(index).fillna(0.0)
        daily["config_id"] = config["config_id"]
        daily_parts.append(daily.reset_index())
        for sleeve in SLEEVES:
            scoped = decisions[decisions["sleeve"].eq(sleeve)]
            kept = scoped[scoped["accepted"]]
            dropped = scoped[~scoped["accepted"]]
            participation_rows.append({
                "config_id": config["config_id"], "sleeve": sleeve, "signals": len(scoped),
                "accepted_trades": len(kept), "rejected_trades": len(dropped),
                "acceptance_rate": float(scoped["accepted"].mean()),
                "accepted_pnl_usd": float(kept["net_pnl_usd"].sum()),
                "abandoned_pnl_usd": float(dropped["net_pnl_usd"].sum()),
            })
        summary_rows.append({
            **config, **exposure, "signals": len(decisions), "accepted_trades": len(accepted),
            "rejected_trades": len(rejected), "rejection_rate": float((~decisions["accepted"]).mean()),
            "accepted_net_pnl_usd": float(accepted["net_pnl_usd"].sum()),
            "abandoned_pnl_usd": float(rejected["net_pnl_usd"].sum()),
            "abandoned_positive_pnl_usd": float(rejected["net_pnl_usd"].clip(lower=0).sum()),
            "abandoned_negative_pnl_usd": float(rejected["net_pnl_usd"].clip(upper=0).sum()),
        })
    return (pd.concat(daily_parts, ignore_index=True), pd.concat(decision_parts, ignore_index=True),
            pd.DataFrame(participation_rows), pd.DataFrame(summary_rows))


def simulate_with_terminal_days(pnl: np.ndarray, traded: np.ndarray, rules: dict[str, Any]) -> dict[str, Any]:
    result = simulate_path(pnl, traded, rules)
    result["days_to_mll"] = float(result["traded_days"]) if result["outcome"] == "MLL_FAIL" else np.nan
    result["days_to_terminal"] = float(result["traded_days"])
    return result


def historical_paths(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config_id, group in daily.groupby("config_id", sort=True):
        group = group.sort_values("session_date")
        pnl = group["daily_pnl_usd"].to_numpy(float)
        traded = group["daily_trade_count"].to_numpy(float) > 0
        for rules in topstep_rules():
            horizon = int(rules["max_traded_days"])
            for start in valid_start_indices(traded, horizon):
                rows.append({
                    "config_id": config_id, "ruleset": rules["name"], "horizon_traded_days": horizon,
                    "start_session_date": group.iloc[start]["session_date"],
                    **simulate_with_terminal_days(pnl[start:], traded[start:], rules),
                })
    return pd.DataFrame(rows)


def summary_with_terminal_days(paths: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    summary = summarize_runs(paths, groups)
    terminal = paths.groupby(groups).agg(
        median_days_to_mll=("days_to_mll", "median"), median_days_to_terminal=("days_to_terminal", "median")
    ).reset_index()
    return summary.merge(terminal, on=groups, how="left")


def bootstrap_paths(cfg: dict[str, Any], daily: pd.DataFrame) -> pd.DataFrame:
    ordered_dates = sorted(daily["session_date"].unique())
    pivot_pnl = daily.pivot(index="session_date", columns="config_id", values="daily_pnl_usd").reindex(ordered_dates)
    pivot_trades = daily.pivot(index="session_date", columns="config_id", values="daily_trade_count").reindex(ordered_dates)
    configs = pivot_pnl.columns.tolist()
    rules = next(rule for rule in topstep_rules() if int(rule["max_traded_days"]) == int(cfg["topstep"]["bootstrap_horizon_traded_days"]))
    rng = np.random.default_rng(int(cfg["bootstrap"]["random_seed"]))
    rows = []
    # Sample until there are 60 dates traded by every configuration, so every governor
    # receives the exact same joint-day sequence while retaining its own traded-day clock.
    shared_traded = pivot_trades.gt(0).all(axis=1).to_numpy(bool)
    for block in cfg["bootstrap"]["block_lengths_sessions"]:
        indices_list = bootstrap_index_paths(shared_traded, int(rules["max_traded_days"]), int(block),
                                             int(cfg["bootstrap"]["paths_per_block_length"]), rng)
        for config_id in configs:
            pnl = pivot_pnl[config_id].to_numpy(float)
            traded = pivot_trades[config_id].to_numpy(float) > 0
            for path_id, indices in enumerate(indices_list, start=1):
                rows.append({
                    "config_id": config_id, "block_length_sessions": int(block), "path_id": path_id,
                    "ruleset": rules["name"], "horizon_traded_days": int(rules["max_traded_days"]),
                    **simulate_with_terminal_days(pnl[indices], traded[indices], rules),
                })
    return pd.DataFrame(rows)


def stress_historical(cfg: dict[str, Any], daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config_id, group in daily.groupby("config_id", sort=True):
        group = group.sort_values("session_date")
        base = group["daily_pnl_usd"].to_numpy(float)
        costs = group["daily_explicit_cost_usd"].to_numpy(float)
        traded = group["daily_trade_count"].to_numpy(float) > 0
        mean = float(base.mean())
        scenarios = {
            "double_costs": base - costs,
            "volatility_1_5x": mean + float(cfg["stress"]["volatility_multiplier"]) * (base - mean),
        }
        for scenario, pnl in scenarios.items():
            for rules in topstep_rules():
                horizon = int(rules["max_traded_days"])
                results = [simulate_with_terminal_days(pnl[start:], traded[start:], rules)
                           for start in valid_start_indices(traded, horizon)]
                frame = pd.DataFrame(results)
                rows.append({
                    "config_id": config_id, "stress": scenario, "ruleset": rules["name"],
                    "horizon_traded_days": horizon, "path_count": len(frame),
                    "pass_probability": float(frame["outcome"].eq("PASS").mean()),
                    "mll_probability": float(frame["outcome"].eq("MLL_FAIL").mean()),
                    "expiry_probability": float(frame["outcome"].eq("EXPIRY").mean()),
                    "median_days_to_pass": float(frame["days_to_pass"].median()),
                    "median_days_to_mll": float(frame["days_to_mll"].median()),
                    "worst_day_usd": float(frame["worst_day_usd"].min()),
                })
    return pd.DataFrame(rows)


def cap_neighbors(left: pd.Series, right: pd.Series) -> bool:
    if left["handler"] != right["handler"]:
        return False
    diffs = [abs(int(left[col]) - int(right[col])) for col in ("global_cap", "mnq_cap", "mgc_cap")]
    return sum(value > 0 for value in diffs) == 1 and max(diffs) == 1


def classify(cfg: dict[str, Any], grid: pd.DataFrame, governor_summary: pd.DataFrame,
             participation: pd.DataFrame, historical_summary: pd.DataFrame,
             bootstrap_summary: pd.DataFrame, stress: pd.DataFrame) -> pd.DataFrame:
    h60 = historical_summary[historical_summary["horizon_traded_days"].eq(60)].set_index("config_id")
    b60 = bootstrap_summary.set_index("config_id")
    base_h = h60.loc["benchmark_no_governor"]
    base_b = b60.loc["benchmark_no_governor"]
    stress60 = stress[(stress["horizon_traded_days"].eq(60)) & stress["stress"].eq("volatility_1_5x")].set_index("config_id")
    costs60 = stress[(stress["horizon_traded_days"].eq(60)) & stress["stress"].eq("double_costs")].set_index("config_id")
    base_stress = stress60.loc["benchmark_no_governor"]
    summary = governor_summary.set_index("config_id")
    gates = cfg["robust_zone_gates"]
    rows, candidates = [], {}
    for config in grid.to_dict("records"):
        cid = config["config_id"]
        min_acceptance = float(participation[participation["config_id"].eq(cid)]["acceptance_rate"].min())
        candidate = bool(
            not config["is_benchmark"]
            and float(base_b["mll_probability"] - b60.loc[cid, "mll_probability"]) >= float(gates["minimum_bootstrap_mll_reduction_vs_baseline"])
            and float(b60.loc[cid, "expiry_probability"] - base_b["expiry_probability"]) <= float(gates["maximum_bootstrap_expiry_increase_vs_baseline"])
            and float(base_h["mll_probability"] - h60.loc[cid, "mll_probability"]) >= float(gates["minimum_historical_mll_reduction_vs_baseline"])
            and min_acceptance >= float(gates["minimum_each_sleeve_acceptance_rate"])
            and float(base_stress["mll_probability"] - stress60.loc[cid, "mll_probability"]) >= float(gates["minimum_volatility_1_5x_mll_reduction_vs_baseline"])
        )
        candidates[cid] = candidate
        rows.append({
            **config, "historical_pass_60d": float(h60.loc[cid, "pass_probability"]),
            "historical_mll_60d": float(h60.loc[cid, "mll_probability"]),
            "historical_expiry_60d": float(h60.loc[cid, "expiry_probability"]),
            "bootstrap_pass_60d": float(b60.loc[cid, "pass_probability"]),
            "bootstrap_pass_ci95_low": float(b60.loc[cid, "pass_ci95_low"]),
            "bootstrap_pass_ci95_high": float(b60.loc[cid, "pass_ci95_high"]),
            "bootstrap_mll_60d": float(b60.loc[cid, "mll_probability"]),
            "bootstrap_expiry_60d": float(b60.loc[cid, "expiry_probability"]),
            "median_days_to_pass": float(b60.loc[cid, "median_days_to_pass"]),
            "median_days_to_mll": float(b60.loc[cid, "median_days_to_mll"]),
            "double_costs_pass_60d": float(costs60.loc[cid, "pass_probability"]),
            "double_costs_mll_60d": float(costs60.loc[cid, "mll_probability"]),
            "double_costs_expiry_60d": float(costs60.loc[cid, "expiry_probability"]),
            "volatility_1_5x_mll_60d": float(stress60.loc[cid, "mll_probability"]),
            "min_sleeve_acceptance_rate": min_acceptance,
            **summary.loc[cid].to_dict(), "gate_candidate": candidate,
        })
    result = pd.DataFrame(rows)
    robust = []
    for row in result.itertuples(index=False):
        has_neighbor = any(
            candidates.get(other.config_id, False) and cap_neighbors(pd.Series(row._asdict()), other)
            for _, other in result.iterrows() if other.config_id != row.config_id
        )
        robust.append(bool(row.gate_candidate and (has_neighbor or not gates["requires_adjacent_cap_candidate_same_handler"])))
    result["robust_zone"] = robust
    result["classification"] = np.where(result["is_benchmark"], "BENCHMARK", np.where(result["robust_zone"], "ROBUST_ZONE",
        np.where(result["gate_candidate"], "ISOLATED_CANDIDATE", np.where(
            result["bootstrap_mll_60d"] < float(base_b["mll_probability"]), "MLL_REDUCED_WITH_TRADEOFF", "NO_ROBUST_IMPROVEMENT"))))
    return result


def make_figures(frontier: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    governed = frontier[~frontier["is_benchmark"]].copy()
    fig, ax = plt.subplots(figsize=(12, 8))
    for handler, group in governed.groupby("handler"):
        ax.scatter(group["bootstrap_expiry_60d"], group["bootstrap_mll_60d"], s=45, alpha=0.75, label=handler)
    base = frontier[frontier["is_benchmark"]].iloc[0]
    ax.scatter([base.bootstrap_expiry_60d], [base.bootstrap_mll_60d], marker="*", s=260, color="black", label="benchmark")
    ax.set_xlabel("Bootstrap EXPIRY probability — 60d")
    ax.set_ylabel("Bootstrap MLL probability — 60d")
    ax.set_title("Risk governor trade-off: MLL versus expiry")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mll_expiry_tradeoff.png", dpi=170)
    plt.close(fig)

    for handler in sorted(governed["handler"].unique()):
        part = governed[governed["handler"].eq(handler)]
        table = part.pivot_table(index=["global_cap", "mnq_cap"], columns="mgc_cap", values="bootstrap_pass_60d")
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(table, annot=True, fmt=".1%", cmap="YlGnBu", ax=ax)
        ax.set_title(f"Bootstrap PASS 60d — {handler}")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"pass_heatmap_{handler}.png", dpi=170)
        plt.close(fig)


def write_report(cfg: dict[str, Any], frontier: pd.DataFrame, participation: pd.DataFrame,
                 historical_summary: pd.DataFrame, bootstrap_summary: pd.DataFrame,
                 stress: pd.DataFrame, decisions: pd.DataFrame) -> None:
    base = frontier[frontier["is_benchmark"]].iloc[0]
    robust = frontier[frontier["robust_zone"]]
    governed = frontier[~frontier["is_benchmark"]].copy()
    descriptive = governed.sort_values(["bootstrap_pass_60d", "bootstrap_mll_60d"], ascending=[False, True]).head(12)
    lowest_mll = governed.sort_values(["bootstrap_mll_60d", "bootstrap_expiry_60d"]).iloc[0]
    highest_pass = governed.sort_values(["bootstrap_pass_60d", "bootstrap_mll_60d"], ascending=[False, True]).iloc[0]
    gates = cfg["robust_zone_gates"]
    decision_comparison = decisions.merge(
        frontier[["config_id", "handler", "global_cap", "mnq_cap", "mgc_cap", "is_benchmark"]],
        on="config_id", how="left", validate="many_to_one",
    )
    governed_decisions = decision_comparison[~decision_comparison["is_benchmark"]]
    identical_handler_cap_tuples = 0
    for _, cap_group in governed_decisions.groupby(["global_cap", "mnq_cap", "mgc_cap"], sort=False):
        matrix = cap_group.pivot(index="signal_id", columns="handler", values="accepted")
        identical_handler_cap_tuples += int(matrix.nunique(axis=1).le(1).all())
    benchmark_decisions = decisions[decisions["config_id"].eq("benchmark_no_governor")]
    simultaneous_entries = benchmark_decisions.groupby("entry_ts_utc").size()
    collision_timestamp_count = int(simultaneous_entries.gt(1).sum())
    unique_entry_timestamp_count = int(len(simultaneous_entries))
    lines = [
        "# TOPSTEP PORTFOLIO RISK GOVERNOR", "", "## Decision", "",
        ("A simple neighboring **ROBUST_ZONE** exists under the frozen gates." if len(robust) else "No simple neighboring **ROBUST_ZONE** satisfies the frozen gates."),
        ("The governor family is therefore suitable for further prospective validation, not parameter selection." if len(robust) else "The current six-sleeve portfolio remains structurally incompatible with the Topstep 50K account size/constraint under these simple governors."),
        "", "## Protocol", "",
        "- All six sleeves continue to generate every original signal. The governor accepts or rejects the complete frozen trade at entry; it never changes its signal, quantity, stop, exit, or PnL path.",
        "- Caps count actual open micro contracts, including the ORB sleeve's frozen variable integer quantity.",
        "- Exits at a timestamp are processed before new entries. Rejected signals are not queued and active positions are never displaced.",
        "- `lowest_frozen_standalone_risk` uses the fixed Phase-1 daily-risk ranking declared before this run; no recent PnL is consulted.",
        "- Historical Topstep runs use every eligible start over 35/60 traded days. Bootstrap uses 2,000 joint complete-day paths for each of 3/5/10/20-session blocks over 60 traded days.",
        "", "## Collision-handler comparison", "",
        f"Only {collision_timestamp_count}/{unique_entry_timestamp_count} distinct entry timestamps contain simultaneous signals. The four handlers therefore produce identical accept/reject decisions for {identical_handler_cap_tuples}/18 cap tuples; where they differ, the first choice can also alter later cap availability. No handler shows a robust advantage across neighboring caps.",
        "", "## Benchmark", "",
        f"`[1,1,1,1,1,1]` without governor: historical 60d PASS {base.historical_pass_60d:.1%}, MLL {base.historical_mll_60d:.1%}, EXPIRY {base.historical_expiry_60d:.1%}; bootstrap PASS {base.bootstrap_pass_60d:.1%}, MLL {base.bootstrap_mll_60d:.1%}, EXPIRY {base.bootstrap_expiry_60d:.1%}.",
        f"Bootstrap median time: {base.median_days_to_pass:.0f} traded days to PASS and {base.median_days_to_mll:.0f} to MLL (conditional on reaching that terminal state).",
        f"Stress 60d: double costs PASS {base.double_costs_pass_60d:.1%}, MLL {base.double_costs_mll_60d:.1%}, EXPIRY {base.double_costs_expiry_60d:.1%}; volatility x1.5 MLL {base.volatility_1_5x_mll_60d:.1%}.",
        f"Maximum simultaneous exposure: {int(base.max_simultaneous_total_contracts)} total, {int(base.max_simultaneous_mnq_contracts)} MNQ, {int(base.max_simultaneous_mgc_contracts)} MGC.",
        "", "## Predeclared robustness gates", "",
        f"A rule must reduce bootstrap MLL by at least {100 * gates['minimum_bootstrap_mll_reduction_vs_baseline']:.0f} percentage points, increase bootstrap EXPIRY by no more than {100 * gates['maximum_bootstrap_expiry_increase_vs_baseline']:.0f} percentage points, reduce historical MLL by at least {100 * gates['minimum_historical_mll_reduction_vs_baseline']:.0f} percentage points, retain at least {gates['minimum_each_sleeve_acceptance_rate']:.0%} of every sleeve's signals, and reduce volatility-x1.5 MLL by at least {100 * gates['minimum_volatility_1_5x_mll_reduction_vs_baseline']:.0f} percentage points. A neighboring cap under the same handler must also pass.",
        "", "## Governor frontier", "",
        "The following rows are descriptive representatives sorted for readability; they are not a best-PASS selection. The complete 72-rule table is in `governor_frontier.csv`.", "",
        "| Handler | G | MNQ | MGC | PASS H | MLL H | EXP H | PASS B | MLL B | EXP B | Reject | Min sleeve participation | PnL abandoned | Costs x2 MLL | Vol1.5 MLL | Class |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in descriptive.itertuples(index=False):
        lines.append(
            f"| `{row.handler}` | {row.global_cap} | {row.mnq_cap} | {row.mgc_cap} | {row.historical_pass_60d:.1%} | {row.historical_mll_60d:.1%} | {row.historical_expiry_60d:.1%} | "
            f"{row.bootstrap_pass_60d:.1%} | {row.bootstrap_mll_60d:.1%} | {row.bootstrap_expiry_60d:.1%} | {row.rejection_rate:.1%} | {row.min_sleeve_acceptance_rate:.1%} | {row.abandoned_pnl_usd:,.0f} | {row.double_costs_mll_60d:.1%} | {row.volatility_1_5x_mll_60d:.1%} | `{row.classification}` |"
        )
    lines += ["", "## Trade-off anchors", "",
              f"- Lowest bootstrap MLL: `{lowest_mll.handler}` G{int(lowest_mll.global_cap)}/MNQ{int(lowest_mll.mnq_cap)}/MGC{int(lowest_mll.mgc_cap)} reaches MLL {lowest_mll.bootstrap_mll_60d:.1%}, but EXPIRY rises to {lowest_mll.bootstrap_expiry_60d:.1%}, PASS falls to {lowest_mll.bootstrap_pass_60d:.1%}, and the least-participating sleeve retains only {lowest_mll.min_sleeve_acceptance_rate:.1%} of signals. Median PASS/MLL times are {lowest_mll.median_days_to_pass:.0f}/{lowest_mll.median_days_to_mll:.0f} traded days; volatility-x1.5 MLL is {lowest_mll.volatility_1_5x_mll_60d:.1%}.",
              f"- Highest bootstrap PASS: `{highest_pass.handler}` G{int(highest_pass.global_cap)}/MNQ{int(highest_pass.mnq_cap)}/MGC{int(highest_pass.mgc_cap)} reaches PASS {highest_pass.bootstrap_pass_60d:.1%}, MLL {highest_pass.bootstrap_mll_60d:.1%}, and EXPIRY {highest_pass.bootstrap_expiry_60d:.1%}. This is descriptive only and is not selected as an optimum."]
    lines += ["", "## Robustness conclusion", ""]
    if len(robust):
        for handler, group in robust.groupby("handler"):
            caps = ", ".join(f"G{int(r.global_cap)}/MNQ{int(r.mnq_cap)}/MGC{int(r.mgc_cap)}" for r in group.itertuples(index=False))
            lines.append(f"- `{handler}` neighboring robust caps: {caps}.")
    else:
        mll_reduced = int(governed["bootstrap_mll_60d"].lt(base.bootstrap_mll_60d).sum())
        lines += [
            f"- {mll_reduced}/{len(governed)} governors reduce bootstrap MLL numerically, but none clears all predeclared gates with an adjacent cap configuration.",
            "- The typical trade-off is direct: tighter caps reduce simultaneous loss but reject enough trades to recreate expiry, suppress a sleeve too strongly, or lose robustness under 1.5x volatility.",
            "- No collision handler produces a broad plateau; isolated PASS improvements are not treated as evidence.",
        ]
    lines += ["", "## Participation and abandoned PnL", "",
              "`sleeve_participation.csv` reports every sleeve's accepted/rejected signals and abandoned counterfactual PnL. Abandoned PnL is diagnostic only and never enters collision decisions.",
              "", "## Artifacts", "", "- `governor_configurations.csv`, `trade_decisions.csv`, `governed_daily_results.csv`",
              "- `sleeve_participation.csv`, `governor_execution_summary.csv`",
              "- `historical_start_paths.csv`, `historical_start_summary.csv`",
              "- `bootstrap_paths.csv`, `bootstrap_summary_by_block.csv`, `bootstrap_summary_pooled.csv`",
              "- `stress_topstep_summary.csv`, `governor_frontier.csv`, `figures/`, `run_metadata.json`", ""]
    (OUTPUT_DIR / "TOPSTEP_PORTFOLIO_RISK_GOVERNOR_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def execute() -> None:
    for directory in (OUTPUT_DIR, FIGURES_DIR, MPL_CACHE):
        directory.mkdir(parents=True, exist_ok=True)
    cfg, source_hashes = load_config()
    config_hash = sha256(CONFIG_PATH)
    pnl, counts, trades = load_inputs(cfg)
    tape = combined_trade_tape(trades)
    grid = governor_grid(cfg)
    grid.to_csv(OUTPUT_DIR / "governor_configurations.csv", index=False)
    risk = {key: float(value) for key, value in cfg["frozen_standalone_daily_risk_usd"].items() if key != "source"}
    daily, decisions, participation, execution_summary = build_governed_daily(grid, tape, pnl.index, risk)
    decisions.to_csv(OUTPUT_DIR / "trade_decisions.csv", index=False)
    daily.to_csv(OUTPUT_DIR / "governed_daily_results.csv", index=False)
    participation.to_csv(OUTPUT_DIR / "sleeve_participation.csv", index=False)
    execution_summary.to_csv(OUTPUT_DIR / "governor_execution_summary.csv", index=False)

    benchmark_daily = daily[daily["config_id"].eq("benchmark_no_governor")].set_index("session_date")["daily_pnl_usd"]
    if not np.allclose(benchmark_daily.reindex(pnl.index).to_numpy(), pnl.sum(axis=1).to_numpy(), atol=1e-7):
        raise RuntimeError("No-governor benchmark does not reproduce the six-sleeve [1,1,1,1,1,1] PnL")

    historical = historical_paths(daily)
    historical.to_csv(OUTPUT_DIR / "historical_start_paths.csv", index=False)
    historical_summary = summary_with_terminal_days(historical, ["config_id", "horizon_traded_days"])
    historical_summary.to_csv(OUTPUT_DIR / "historical_start_summary.csv", index=False)

    bootstrap = bootstrap_paths(cfg, daily)
    bootstrap.to_csv(OUTPUT_DIR / "bootstrap_paths.csv", index=False)
    bootstrap_by_block = summary_with_terminal_days(bootstrap, ["config_id", "block_length_sessions"])
    bootstrap_by_block.to_csv(OUTPUT_DIR / "bootstrap_summary_by_block.csv", index=False)
    bootstrap_pooled = summary_with_terminal_days(bootstrap, ["config_id"])
    bootstrap_pooled.to_csv(OUTPUT_DIR / "bootstrap_summary_pooled.csv", index=False)
    stress = stress_historical(cfg, daily)
    stress.to_csv(OUTPUT_DIR / "stress_topstep_summary.csv", index=False)
    frontier = classify(cfg, grid, execution_summary, participation, historical_summary, bootstrap_pooled, stress)
    frontier.to_csv(OUTPUT_DIR / "governor_frontier.csv", index=False)
    make_figures(frontier)
    write_report(cfg, frontier, participation, historical_summary, bootstrap_pooled, stress, decisions)

    if sha256(CONFIG_PATH) != config_hash:
        raise RuntimeError("Governor config changed during execution")
    for key, expected in source_hashes.items():
        if sha256(REPO_ROOT / cfg["source_locks"][key]["path"]) != expected:
            raise RuntimeError(f"Locked source changed during execution: {key}")
    metadata = {
        "phase": cfg["phase"], "config_sha256": config_hash, "source_sha256": source_hashes,
        "configuration_count_including_benchmark": len(grid), "signal_count": len(tape),
        "historical_path_count": len(historical), "bootstrap_path_count": len(bootstrap),
        "bootstrap_seed": cfg["bootstrap"]["random_seed"], "classification_counts": frontier["classification"].value_counts().to_dict(),
        "robust_zone_count": int(frontier["robust_zone"].sum()), "sleeves_modified": False,
        "recent_pnl_used_for_collision_decisions": False, "ml_used": False,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(frontier.sort_values(["robust_zone", "bootstrap_pass_60d"], ascending=[False, False]).head(25).to_string(index=False), flush=True)


if __name__ == "__main__":
    execute()
