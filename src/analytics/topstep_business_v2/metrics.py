"""Summary tables and rankings for Topstep business v2 simulations."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.analytics.topstep_business_v2.simulator import AVERAGE_CALENDAR_DAYS_PER_MONTH


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(float(denominator)):
        return float(default)
    value = float(numerator) / float(denominator)
    return float(value) if math.isfinite(value) else float(default)


def _nan_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else float("nan")


def _summary_group_columns() -> list[str]:
    return [
        "simulation_method",
        "plan",
        "n_resets_per_month",
        "challenge_strategy",
        "challenge_source_variant_name",
        "challenge_leverage_factor",
        "funded_strategy",
        "funded_source_variant_name",
        "funded_leverage_factor",
    ]


def summarize_business_run_frame(run_frame: pd.DataFrame) -> dict[str, Any]:
    """Aggregate one configuration's simulation rows into decision-ready metrics."""
    if run_frame.empty:
        return {
            "simulation_method": "",
            "plan": "",
            "n_resets_per_month": 0,
            "challenge_strategy": "",
            "challenge_source_variant_name": "",
            "challenge_leverage_factor": float("nan"),
            "funded_strategy": "",
            "funded_source_variant_name": "",
            "funded_leverage_factor": float("nan"),
            "run_count": 0,
            "challenge_pass_rate": 0.0,
            "first_attempt_pass_rate": 0.0,
            "pass_within_month_rate": 0.0,
            "pass_within_2_attempts_same_month_rate": 0.0,
            "avg_days_to_raw_target": float("nan"),
            "avg_days_raw_target_to_true_pass": float("nan"),
            "consistency_block_rate": 0.0,
            "avg_max_day_share": float("nan"),
            "avg_consistency_penalty_usd": float("nan"),
            "avg_dll_hits_per_attempt": 0.0,
            "fail_mll_rate": 0.0,
            "timeout_rate": 0.0,
            "budget_exhausted_rate": 0.0,
            "avg_resets_per_month": 0.0,
            "probability_of_two_failures_same_month": 0.0,
            "funded_start_rate": 0.0,
            "funded_survival_rate_60d": 0.0,
            "first_payout_rate": 0.0,
            "avg_days_to_first_payout": float("nan"),
            "avg_payout_count": 0.0,
            "avg_funded_profit": 0.0,
            "funded_breach_rate": 0.0,
            "expected_net_profit_per_cycle": 0.0,
            "expected_net_profit_per_calendar_day": 0.0,
            "expected_net_profit_per_calendar_month": 0.0,
            "expected_cost_per_cycle": 0.0,
            "expected_cost_per_month": 0.0,
            "expected_resets_per_month": 0.0,
            "expected_activation_spend": 0.0,
            "avg_calendar_days_locked_out": 0.0,
            "ROI_on_cash_spent": 0.0,
        }

    first = run_frame.iloc[0]
    challenge_pass_mask = run_frame["challenge_passed"].astype(bool)
    funded_started_mask = run_frame["funded_started"].astype(bool)

    expected_net_profit_per_cycle = _nan_mean(run_frame["net_profit_usd"])
    expected_calendar_days = _nan_mean(run_frame["cycle_calendar_days"])
    expected_cost_per_cycle = _nan_mean(run_frame["total_cost_usd"])
    expected_net_profit_per_calendar_day = _safe_div(
        expected_net_profit_per_cycle,
        expected_calendar_days,
        default=0.0,
    )
    expected_cost_per_day = _safe_div(expected_cost_per_cycle, expected_calendar_days, default=0.0)
    expected_challenge_active_months = _nan_mean(run_frame["challenge_active_months"])

    return {
        "simulation_method": str(first["simulation_method"]),
        "plan": str(first["plan"]),
        "n_resets_per_month": int(first["n_resets_per_month"]),
        "challenge_strategy": str(first["challenge_strategy"]),
        "challenge_source_variant_name": str(first["challenge_source_variant_name"]),
        "challenge_leverage_factor": float(first["challenge_leverage_factor"]),
        "funded_strategy": str(first["funded_strategy"]),
        "funded_source_variant_name": str(first["funded_source_variant_name"]),
        "funded_leverage_factor": float(first["funded_leverage_factor"]),
        "run_count": int(len(run_frame)),
        "challenge_pass_rate": float(challenge_pass_mask.mean()),
        "first_attempt_pass_rate": float(run_frame["first_attempt_pass"].astype(bool).mean()),
        "pass_within_month_rate": float(run_frame["pass_within_month"].astype(bool).mean()),
        "pass_within_2_attempts_same_month_rate": float(
            run_frame["pass_within_2_attempts_same_month"].astype(bool).mean()
        ),
        "avg_days_to_raw_target": _nan_mean(run_frame["days_to_raw_target"]),
        "avg_days_raw_target_to_true_pass": _nan_mean(run_frame["days_raw_target_to_true_pass"]),
        "consistency_block_rate": float(run_frame["consistency_blocked"].astype(bool).mean()),
        "avg_max_day_share": _nan_mean(run_frame["avg_max_day_share"]),
        "avg_consistency_penalty_usd": _nan_mean(run_frame["avg_consistency_penalty_usd"]),
        "avg_dll_hits_per_attempt": _nan_mean(run_frame["avg_dll_hits_per_attempt"]),
        "fail_mll_rate": float(run_frame["challenge_fail_mll_count"].gt(0).mean()),
        "timeout_rate": float(run_frame["challenge_timeout"].astype(bool).mean()),
        "budget_exhausted_rate": float(run_frame["budget_exhausted"].astype(bool).mean()),
        "avg_resets_per_month": _nan_mean(run_frame["resets_per_active_month"]),
        "probability_of_two_failures_same_month": float(
            run_frame["probability_two_failures_same_month"].astype(bool).mean()
        ),
        "funded_start_rate": float(funded_started_mask.mean()),
        "funded_survival_rate_60d": float(run_frame["funded_survived_60d"].astype(bool).mean()),
        "first_payout_rate": float(run_frame["first_payout_achieved"].astype(bool).mean()),
        "avg_days_to_first_payout": _nan_mean(run_frame["days_to_first_payout"]),
        "avg_payout_count": _nan_mean(run_frame["payout_count"]),
        "avg_funded_profit": _nan_mean(run_frame["funded_profit_usd"]),
        "funded_breach_rate": float(run_frame["funded_breach"].astype(bool).mean()),
        "expected_net_profit_per_cycle": float(expected_net_profit_per_cycle),
        "expected_net_profit_per_calendar_day": float(expected_net_profit_per_calendar_day),
        "expected_net_profit_per_calendar_month": float(
            expected_net_profit_per_calendar_day * AVERAGE_CALENDAR_DAYS_PER_MONTH
        ),
        "expected_cost_per_cycle": float(expected_cost_per_cycle),
        "expected_cost_per_month": float(expected_cost_per_day * AVERAGE_CALENDAR_DAYS_PER_MONTH),
        "expected_resets_per_month": _safe_div(
            _nan_mean(run_frame["resets_used"]),
            expected_challenge_active_months,
            default=0.0,
        ),
        "expected_activation_spend": _nan_mean(run_frame["activation_fee_usd"]),
        "avg_calendar_days_locked_out": _nan_mean(run_frame["calendar_days_locked_out"]),
        "ROI_on_cash_spent": _safe_div(expected_net_profit_per_cycle, expected_cost_per_cycle, default=0.0),
        "conditional_first_payout_rate_given_pass": float(
            run_frame.loc[challenge_pass_mask, "first_payout_achieved"].astype(bool).mean()
        )
        if bool(challenge_pass_mask.any())
        else 0.0,
        "conditional_funded_survival_rate_given_pass": float(
            run_frame.loc[challenge_pass_mask, "funded_survived_60d"].astype(bool).mean()
        )
        if bool(challenge_pass_mask.any())
        else 0.0,
    }


def summarize_business_runs(run_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize all simulation rows by configuration."""
    if run_frame.empty:
        return pd.DataFrame()

    rows = [
        summarize_business_run_frame(group)
        for _, group in run_frame.groupby(_summary_group_columns(), sort=False, dropna=False)
    ]
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        [
            "simulation_method",
            "expected_net_profit_per_calendar_month",
            "expected_net_profit_per_calendar_day",
            "first_attempt_pass_rate",
            "first_payout_rate",
            "plan",
            "challenge_strategy",
            "funded_strategy",
        ],
        ascending=[True, False, False, False, False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_challenge_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the challenge-focused slice of the summary table."""
    if summary.empty:
        return summary.copy()
    columns = [
        "simulation_method",
        "plan",
        "n_resets_per_month",
        "challenge_strategy",
        "funded_strategy",
        "run_count",
        "challenge_pass_rate",
        "first_attempt_pass_rate",
        "pass_within_month_rate",
        "pass_within_2_attempts_same_month_rate",
        "avg_days_to_raw_target",
        "avg_days_raw_target_to_true_pass",
        "consistency_block_rate",
        "avg_max_day_share",
        "avg_consistency_penalty_usd",
        "avg_dll_hits_per_attempt",
        "fail_mll_rate",
        "timeout_rate",
        "budget_exhausted_rate",
        "avg_resets_per_month",
        "probability_of_two_failures_same_month",
        "avg_calendar_days_locked_out",
    ]
    return summary[columns].copy()


def build_funded_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the funded-focused slice of the summary table."""
    if summary.empty:
        return summary.copy()
    columns = [
        "simulation_method",
        "plan",
        "n_resets_per_month",
        "challenge_strategy",
        "funded_strategy",
        "funded_start_rate",
        "funded_survival_rate_60d",
        "first_payout_rate",
        "avg_days_to_first_payout",
        "avg_payout_count",
        "avg_funded_profit",
        "funded_breach_rate",
        "conditional_first_payout_rate_given_pass",
        "conditional_funded_survival_rate_given_pass",
    ]
    return summary[columns].copy()


def _apply_deterministic_ranking(
    frame: pd.DataFrame,
    primary_sort_columns: list[str],
    rank_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ranked = frame.sort_values(
        primary_sort_columns
        + [
            "plan",
            "simulation_method",
            "n_resets_per_month",
            "challenge_strategy",
            "funded_strategy",
            "challenge_source_variant_name",
            "funded_source_variant_name",
        ],
        ascending=[False, False, False, False] + [True, True, True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked.insert(0, rank_column, range(1, len(ranked) + 1))
    return ranked


def build_ranking_month(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank configurations by expected net profit per calendar month."""
    if summary.empty:
        return summary.copy()
    return _apply_deterministic_ranking(
        summary,
        primary_sort_columns=[
            "expected_net_profit_per_calendar_month",
            "expected_net_profit_per_calendar_day",
            "first_attempt_pass_rate",
            "first_payout_rate",
        ],
        rank_column="rank_month",
    )


def build_ranking_day(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank configurations by expected net profit per calendar day."""
    if summary.empty:
        return summary.copy()
    return _apply_deterministic_ranking(
        summary,
        primary_sort_columns=[
            "expected_net_profit_per_calendar_day",
            "expected_net_profit_per_calendar_month",
            "first_attempt_pass_rate",
            "first_payout_rate",
        ],
        rank_column="rank_day",
    )
