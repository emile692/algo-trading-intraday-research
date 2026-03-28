from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics.mnq_orb_prop_challenge_readiness_campaign import (
    FundedFollowupSpec,
    PropChallengeReadinessSpec,
    PropChallengeRules,
    RiskProfile,
    StressProfile,
    _build_final_verdict,
    _risk_scaled_trades,
    simulate_challenge_attempt,
    simulate_funded_followup,
)


def _daily_frame(rows: list[tuple[str, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["session_date", "daily_pnl_usd", "daily_trade_count"])


def test_simulate_challenge_attempt_caps_daily_loss_before_breach() -> None:
    daily = _daily_frame([("2024-01-02", -1500.0, 1), ("2024-01-03", 5000.0, 1)])
    rules = PropChallengeRules(
        profit_target_usd=3000.0,
        daily_loss_limit_usd=1000.0,
        trailing_drawdown_usd=2000.0,
        max_trading_days=5,
    )

    history, result = simulate_challenge_attempt(daily_results=daily, rules=rules)

    assert result["status"] == "breach"
    assert result["failure_reason"] == "daily_loss_limit"
    assert result["final_pnl_usd"] == -1000.0
    assert history.iloc[0]["effective_daily_pnl_usd"] == -1000.0


def test_risk_scaled_trades_applies_extra_slippage_cost() -> None:
    trades = pd.DataFrame(
        [
            {
                "trade_id": 1,
                "session_date": "2024-01-02",
                "quantity": 2.0,
                "pnl_usd": 100.0,
                "fees": 5.0,
                "net_pnl_usd": 95.0,
                "risk_multiplier": 1.0,
            }
        ]
    )

    stressed = _risk_scaled_trades(
        trades=trades,
        risk_multiplier=1.0,
        extra_slippage_ticks_per_side=1.0,
        tick_value_usd=0.5,
    )

    assert stressed.iloc[0]["stress_extra_slippage_usd"] == 2.0
    assert stressed.iloc[0]["pnl_usd"] == 98.0
    assert stressed.iloc[0]["net_pnl_usd"] == 93.0


def test_simulate_funded_followup_reports_breach_and_risk_ratio() -> None:
    pre_pass = pd.DataFrame(
        {
            "effective_daily_pnl_usd": [300.0, 350.0, 400.0],
        }
    )
    followup = _daily_frame(
        [
            ("2024-01-10", 200.0, 1),
            ("2024-01-11", -1001.0, 1),
        ]
    )
    rules = PropChallengeRules(
        daily_loss_limit_usd=1000.0,
        trailing_drawdown_usd=2000.0,
        max_trading_days=5,
    )

    result = simulate_funded_followup(
        daily_results=followup,
        rules=rules,
        spec=FundedFollowupSpec(enabled=True, trading_days=10),
        pre_pass_history=pre_pass,
    )

    assert result["followup_started"] is True
    assert result["followup_breach"] is True
    assert result["followup_breach_reason"] == "daily_loss_limit"
    assert result["followup_risk_intensity_ratio_post_vs_pre"] > 1.0


def test_build_final_verdict_detects_split_configuration() -> None:
    spec = PropChallengeReadinessSpec(
        risk_profiles=(RiskProfile(name="base", multiplier=1.0, description="base"),),
        stress_profiles=(
            StressProfile(name="slippage_nominal", slippage_multiplier=1.0, description="nominal"),
            StressProfile(name="slippage_x2", slippage_multiplier=2.0, description="x2"),
        ),
    )
    business_summary = pd.DataFrame(
        [
            {
                "variant_name": "baseline_3state_vvix_modulator",
                "risk_profile": "base",
                "stress_profile": "slippage_nominal",
                "pass_rate": 0.62,
                "breach_rate": 0.18,
                "median_days_to_pass": 12.0,
                "expected_net_profit_per_attempt": 900.0,
                "average_drawdown_before_pass": -1200.0,
                "followup_breach_rate": 0.30,
                "followup_complete_rate": 0.70,
                "followup_expected_profit_per_trading_day": 40.0,
            },
            {
                "variant_name": "baseline_3state",
                "risk_profile": "base",
                "stress_profile": "slippage_nominal",
                "pass_rate": 0.57,
                "breach_rate": 0.24,
                "median_days_to_pass": 11.0,
                "expected_net_profit_per_attempt": 850.0,
                "average_drawdown_before_pass": -1700.0,
                "followup_breach_rate": 0.15,
                "followup_complete_rate": 0.85,
                "followup_expected_profit_per_trading_day": 55.0,
            },
        ]
    )
    stress_summary = pd.DataFrame(
        [
            {
                "variant_name": "baseline_3state_vvix_modulator",
                "risk_profile": "base",
                "stress_profile": "slippage_x2",
                "pass_rate": 0.50,
                "breach_rate": 0.28,
                "median_days_to_pass": 14.0,
            },
            {
                "variant_name": "baseline_3state",
                "risk_profile": "base",
                "stress_profile": "slippage_x2",
                "pass_rate": 0.48,
                "breach_rate": 0.32,
                "median_days_to_pass": 13.0,
            },
        ]
    )

    verdict = _build_final_verdict(
        challenge_summary=business_summary.copy(),
        stress_summary=stress_summary,
        business_summary=business_summary,
        spec=spec,
        source_root=Path("dummy"),
        metadata={"run_timestamp": "2026-03-28T12:00:00"},
        common_start_dates=[pd.Timestamp("2024-01-02").date()],
    )

    assert verdict["challenge_best_variant"] == "baseline_3state_vvix_modulator"
    assert verdict["funded_best_variant"] == "baseline_3state"
    assert verdict["use_split_configuration"] is True
    assert verdict["vvix_modulator_improves_challenge_business"] is True
