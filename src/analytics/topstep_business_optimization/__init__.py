"""Topstep business optimization package."""

from __future__ import annotations

from src.analytics.topstep_business_optimization.metrics import build_ranking_table, summarize_business_runs
from src.analytics.topstep_business_optimization.simulator import (
    BusinessRules,
    StrategySeries,
    TopstepPlan,
    build_strategy_name,
    prepare_strategy_series,
    simulate_business_cycle,
    simulate_challenge_attempt,
    simulate_funded_phase,
)

__all__ = [
    "BusinessCampaignSpec",
    "BusinessRules",
    "StrategySeries",
    "TopstepPlan",
    "build_ranking_table",
    "build_strategy_name",
    "prepare_strategy_series",
    "run_campaign",
    "simulate_business_cycle",
    "simulate_challenge_attempt",
    "simulate_funded_phase",
    "summarize_business_runs",
]


def __getattr__(name: str):
    if name in {"BusinessCampaignSpec", "run_campaign"}:
        from src.analytics.topstep_business_optimization.campaign import BusinessCampaignSpec, run_campaign

        mapping = {
            "BusinessCampaignSpec": BusinessCampaignSpec,
            "run_campaign": run_campaign,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
