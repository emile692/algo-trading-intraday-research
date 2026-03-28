"""Aggregation helpers for Topstep business optimization runs."""

from __future__ import annotations

import math

import pandas as pd


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(float(denominator)):
        return float(default)
    value = float(numerator) / float(denominator)
    return float(value) if math.isfinite(value) else float(default)


def _nan_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else float("nan")


def summarize_business_run_frame(run_frame: pd.DataFrame) -> dict[str, float | str | int]:
    if run_frame.empty:
        return {
            "plan": "",
            "challenge_strategy": "",
            "challenge_source_variant_name": "",
            "challenge_leverage_factor": float("nan"),
            "funded_strategy": "",
            "funded_source_variant_name": "",
            "funded_leverage_factor": float("nan"),
            "run_count": 0,
            "pass_rate": 0.0,
            "first_attempt_pass_rate": 0.0,
            "avg_days_to_pass": float("nan"),
            "avg_resets": 0.0,
            "avg_payouts": 0.0,
            "at_least_one_payout_rate": 0.0,
            "avg_profit_post_pass": float("nan"),
            "avg_profit_post_pass_all_cycles": 0.0,
            "avg_subscription_cost_usd": 0.0,
            "avg_reset_cost_usd": 0.0,
            "avg_activation_fee_usd": 0.0,
            "total_cost": 0.0,
            "net_profit": 0.0,
            "expected_net_profit_per_cycle": 0.0,
            "expected_days_per_cycle": 0.0,
            "expected_net_profit_per_day": 0.0,
        }

    pass_mask = run_frame["eventual_pass"].astype(bool)
    expected_net_profit_per_cycle = _nan_mean(run_frame["net_profit_usd"])
    expected_days_per_cycle = _nan_mean(run_frame["total_days"])

    first = run_frame.iloc[0]
    return {
        "plan": str(first["plan"]),
        "challenge_strategy": str(first["challenge_strategy"]),
        "challenge_source_variant_name": str(first["challenge_source_variant_name"]),
        "challenge_leverage_factor": float(first["challenge_leverage_factor"]),
        "funded_strategy": str(first["funded_strategy"]),
        "funded_source_variant_name": str(first["funded_source_variant_name"]),
        "funded_leverage_factor": float(first["funded_leverage_factor"]),
        "run_count": int(len(run_frame)),
        "pass_rate": float(pass_mask.mean()),
        "first_attempt_pass_rate": float(run_frame["first_attempt_pass"].astype(bool).mean()),
        "avg_days_to_pass": _nan_mean(run_frame.loc[pass_mask, "challenge_days_to_pass"]),
        "avg_resets": _nan_mean(run_frame["resets"]),
        "avg_payouts": _nan_mean(run_frame["payouts"]),
        "at_least_one_payout_rate": float(run_frame["at_least_one_payout"].astype(bool).mean()),
        "avg_profit_post_pass": _nan_mean(run_frame.loc[pass_mask, "funded_profit_usd"]),
        "avg_profit_post_pass_all_cycles": _nan_mean(run_frame["funded_profit_usd"]),
        "avg_subscription_cost_usd": _nan_mean(run_frame["subscription_cost_usd"]),
        "avg_reset_cost_usd": _nan_mean(run_frame["reset_cost_usd"]),
        "avg_activation_fee_usd": _nan_mean(run_frame["activation_fee_usd"]),
        "total_cost": _nan_mean(run_frame["total_cost_usd"]),
        "net_profit": expected_net_profit_per_cycle,
        "expected_net_profit_per_cycle": expected_net_profit_per_cycle,
        "expected_days_per_cycle": expected_days_per_cycle,
        "expected_net_profit_per_day": _safe_div(
            expected_net_profit_per_cycle,
            expected_days_per_cycle,
            default=0.0,
        ),
    }


def summarize_business_runs(run_frame: pd.DataFrame) -> pd.DataFrame:
    if run_frame.empty:
        return pd.DataFrame(
            columns=[
                "plan",
                "challenge_strategy",
                "challenge_source_variant_name",
                "challenge_leverage_factor",
                "funded_strategy",
                "funded_source_variant_name",
                "funded_leverage_factor",
                "run_count",
                "pass_rate",
                "first_attempt_pass_rate",
                "avg_days_to_pass",
                "avg_resets",
                "avg_payouts",
                "at_least_one_payout_rate",
                "avg_profit_post_pass",
                "avg_profit_post_pass_all_cycles",
                "avg_subscription_cost_usd",
                "avg_reset_cost_usd",
                "avg_activation_fee_usd",
                "total_cost",
                "net_profit",
                "expected_net_profit_per_cycle",
                "expected_days_per_cycle",
                "expected_net_profit_per_day",
            ]
        )

    rows: list[dict[str, float | str | int]] = []
    group_columns = [
        "plan",
        "challenge_strategy",
        "challenge_source_variant_name",
        "challenge_leverage_factor",
        "funded_strategy",
        "funded_source_variant_name",
        "funded_leverage_factor",
    ]
    for _, group in run_frame.groupby(group_columns, dropna=False, sort=False):
        rows.append(summarize_business_run_frame(group))

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["plan", "expected_net_profit_per_day", "expected_net_profit_per_cycle", "pass_rate"],
        ascending=[True, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def build_ranking_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    ranked = summary.sort_values(
        ["expected_net_profit_per_day", "expected_net_profit_per_cycle", "pass_rate", "avg_payouts"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    ranked.insert(0, "overall_rank", range(1, len(ranked) + 1))
    ranked["plan_rank"] = ranked.groupby("plan").cumcount() + 1
    return ranked
