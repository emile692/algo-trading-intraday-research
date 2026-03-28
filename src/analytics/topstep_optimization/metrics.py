"""Economic metrics for Topstep optimization campaigns."""

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


def summarize_run_frame(
    run_frame: pd.DataFrame,
    payout_value: float,
    reset_cost: float,
) -> dict[str, float | str | int]:
    if run_frame.empty:
        return {
            "simulation_mode": "",
            "variant": "",
            "base_variant": "",
            "source_variant_name": "",
            "leverage_factor": float("nan"),
            "run_count": 0,
            "pass_rate": 0.0,
            "fail_rate": 0.0,
            "expire_rate": 0.0,
            "avg_time_to_pass": float("nan"),
            "avg_time_to_fail": float("nan"),
            "expected_profit_per_cycle": 0.0,
            "expected_days_per_cycle": 0.0,
            "expected_profit_per_day": 0.0,
            "probability_pass_within_20_days": 0.0,
            "probability_pass_within_30_days": 0.0,
        }

    pass_rate = float(pd.to_numeric(run_frame["pass"], errors="coerce").fillna(0.0).mean())
    fail_rate = float(pd.to_numeric(run_frame["fail"], errors="coerce").fillna(0.0).mean())
    expire_rate = float(pd.to_numeric(run_frame["expire"], errors="coerce").fillna(0.0).mean())
    avg_time_to_pass = _nan_mean(run_frame.loc[run_frame["pass"], "days_to_pass"])
    avg_time_to_fail = _nan_mean(run_frame.loc[run_frame["fail"], "days_to_fail"])
    expected_profit_per_cycle = float(pass_rate * float(payout_value) - fail_rate * float(reset_cost))
    expected_days_per_cycle = _nan_mean(run_frame["cycle_trading_days"])
    expected_profit_per_day = _safe_div(expected_profit_per_cycle, expected_days_per_cycle, default=0.0)
    probability_pass_within_20_days = float(
        ((pd.to_numeric(run_frame["pass"], errors="coerce").fillna(0.0) > 0.0) & (pd.to_numeric(run_frame["days_to_pass"], errors="coerce") <= 20)).mean()
    )
    probability_pass_within_30_days = float(
        ((pd.to_numeric(run_frame["pass"], errors="coerce").fillna(0.0) > 0.0) & (pd.to_numeric(run_frame["days_to_pass"], errors="coerce") <= 30)).mean()
    )

    first = run_frame.iloc[0]
    return {
        "simulation_mode": str(first["simulation_mode"]),
        "variant": str(first["variant"]),
        "base_variant": str(first["base_variant"]),
        "source_variant_name": str(first["source_variant_name"]),
        "leverage_factor": float(first["leverage_factor"]),
        "run_count": int(len(run_frame)),
        "pass_rate": pass_rate,
        "fail_rate": fail_rate,
        "expire_rate": expire_rate,
        "avg_time_to_pass": avg_time_to_pass,
        "avg_time_to_fail": avg_time_to_fail,
        "expected_profit_per_cycle": expected_profit_per_cycle,
        "expected_days_per_cycle": expected_days_per_cycle,
        "expected_profit_per_day": expected_profit_per_day,
        "probability_pass_within_20_days": probability_pass_within_20_days,
        "probability_pass_within_30_days": probability_pass_within_30_days,
    }


def summarize_simulation_results(
    run_frame: pd.DataFrame,
    payout_value: float,
    reset_cost: float,
) -> pd.DataFrame:
    if run_frame.empty:
        return pd.DataFrame(
            columns=[
                "simulation_mode",
                "variant",
                "base_variant",
                "source_variant_name",
                "leverage_factor",
                "run_count",
                "pass_rate",
                "fail_rate",
                "expire_rate",
                "avg_time_to_pass",
                "avg_time_to_fail",
                "expected_profit_per_cycle",
                "expected_days_per_cycle",
                "expected_profit_per_day",
                "probability_pass_within_20_days",
                "probability_pass_within_30_days",
            ]
        )

    rows: list[dict[str, float | str | int]] = []
    group_columns = ["simulation_mode", "variant", "base_variant", "source_variant_name", "leverage_factor"]
    for _, group in run_frame.groupby(group_columns, dropna=False, sort=False):
        rows.append(summarize_run_frame(group, payout_value=payout_value, reset_cost=reset_cost))

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["simulation_mode", "expected_profit_per_day", "expected_profit_per_cycle", "pass_rate"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def build_ranking_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    ranked = summary.sort_values(
        ["expected_profit_per_day", "expected_profit_per_cycle", "pass_rate", "avg_time_to_pass"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked
