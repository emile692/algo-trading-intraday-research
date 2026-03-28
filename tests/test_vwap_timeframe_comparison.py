from __future__ import annotations

import pandas as pd

from src.analytics.vwap_timeframe_comparison import (
    _build_delta_summary,
    _build_global_verdict,
    build_default_timeframe_comparison_spec,
)
from src.config.vwap_campaign import adapt_vwap_variant_to_timeframe, build_default_vwap_timeframe_comparison_variants


def test_default_timeframe_comparison_universe_is_small_and_targeted() -> None:
    variants = build_default_vwap_timeframe_comparison_variants()
    names = [variant.name for variant in variants]

    assert names == ["paper_vwap_baseline", "baseline_futures_adapted", "vwap_reclaim"]


def test_adapt_vwap_variant_to_timeframe_rescales_bar_windows_without_retuning_thresholds() -> None:
    variant = next(v for v in build_default_vwap_timeframe_comparison_variants() if v.name == "vwap_reclaim")

    adapted = adapt_vwap_variant_to_timeframe(variant, bar_minutes=5, base_bar_minutes=1)

    assert adapted.slope_lookback == 1
    assert adapted.atr_period == 3
    assert adapted.compression_length == 2
    assert adapted.pullback_lookback == 2
    assert adapted.atr_buffer == variant.atr_buffer
    assert adapted.stop_buffer == variant.stop_buffer


def test_delta_summary_requires_broad_improvement_not_just_less_bad_pnl() -> None:
    spec = build_default_timeframe_comparison_spec()
    comparison_summary = pd.DataFrame(
        [
            {
                "timeframe": "1m",
                "bar_minutes": 1,
                "strategy_id": "vwap_reclaim",
                "role": "candidate",
                "oos_net_pnl": 100.0,
                "oos_profit_factor": 1.10,
                "oos_sharpe_ratio": 0.20,
                "oos_max_drawdown": -500.0,
                "oos_total_trades": 100,
                "oos_expectancy_per_trade": 1.0,
                "pnl_slip_x2": 80.0,
                "pf_slip_x2": 1.05,
                "positive_oos_splits": 2,
                "challenge_success_rate_standard": 0.10,
                "daily_loss_limit_breach_freq": 0.02,
                "trailing_drawdown_breach_freq": 0.05,
                "top_5_day_contribution_pct": 1.20,
                "pnl_excluding_top_5_days": -50.0,
            },
            {
                "timeframe": "5m",
                "bar_minutes": 5,
                "strategy_id": "vwap_reclaim",
                "role": "candidate",
                "oos_net_pnl": 120.0,
                "oos_profit_factor": 1.12,
                "oos_sharpe_ratio": 0.22,
                "oos_max_drawdown": -450.0,
                "oos_total_trades": 20,
                "oos_expectancy_per_trade": 6.0,
                "pnl_slip_x2": 90.0,
                "pf_slip_x2": 1.06,
                "positive_oos_splits": 2,
                "challenge_success_rate_standard": 0.10,
                "daily_loss_limit_breach_freq": 0.02,
                "trailing_drawdown_breach_freq": 0.05,
                "top_5_day_contribution_pct": 1.10,
                "pnl_excluding_top_5_days": -40.0,
            },
        ]
    )

    delta = _build_delta_summary(comparison_summary, spec)

    assert len(delta) == 1
    assert bool(delta.iloc[0]["credible_robustness_improvement"]) is False
    assert delta.iloc[0]["comparison_bucket"] == "amelioration partielle mais insuffisante"


def test_global_verdict_archives_family_when_no_candidate_shows_credible_5m_improvement() -> None:
    comparison_summary = pd.DataFrame(
        [
            {
                "timeframe": "1m",
                "strategy_id": "paper_vwap_baseline",
                "role": "paper_baseline_reference",
                "oos_net_pnl": -100.0,
                "oos_profit_factor": 0.95,
                "oos_sharpe_ratio": -0.1,
                "oos_max_drawdown": -500.0,
                "oos_total_trades": 100,
                "pnl_slip_x2": -120.0,
                "positive_oos_splits": 1,
                "challenge_success_rate_standard": 0.0,
                "top_5_day_contribution_pct": 1.5,
                "final_bucket": "reference_officielle",
            },
            {
                "timeframe": "5m",
                "strategy_id": "paper_vwap_baseline",
                "role": "paper_baseline_reference",
                "oos_net_pnl": -50.0,
                "oos_profit_factor": 0.98,
                "oos_sharpe_ratio": -0.05,
                "oos_max_drawdown": -350.0,
                "oos_total_trades": 30,
                "pnl_slip_x2": -70.0,
                "positive_oos_splits": 1,
                "challenge_success_rate_standard": 0.0,
                "top_5_day_contribution_pct": 1.2,
                "final_bucket": "reference_officielle",
            },
        ]
    )
    delta = pd.DataFrame(
        [
            {
                "strategy_id": "vwap_reclaim",
                "role": "candidate",
                "credible_robustness_improvement": False,
                "improvement_score": 2,
                "oos_profit_factor_5m": 1.01,
                "oos_sharpe_ratio_5m": 0.02,
            }
        ]
    )

    verdict = _build_global_verdict(comparison_summary, delta, runs=[])

    assert verdict["global_verdict"] == "5 minutes ne change pas le verdict, famille VWAP a archiver"
    assert verdict["top_candidate_if_any"] is None
