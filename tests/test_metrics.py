import pandas as pd
import pytest

from src.analytics.metrics import compute_metrics
from src.config.orb_campaign import PropConstraintConfig


def test_compute_metrics_includes_prop_style_fields() -> None:
    trades = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"]).date,
            "entry_time": pd.to_datetime(
                [
                    "2024-01-02 09:35:00",
                    "2024-01-02 10:15:00",
                    "2024-01-03 09:40:00",
                    "2024-01-04 09:45:00",
                ]
            ),
            "exit_time": pd.to_datetime(
                [
                    "2024-01-02 10:00:00",
                    "2024-01-02 11:00:00",
                    "2024-01-03 12:00:00",
                    "2024-01-04 15:55:00",
                ]
            ),
            "net_pnl_usd": [-100.0, -100.0, 250.0, 50.0],
            "exit_reason": ["stop", "stop", "target", "time_exit"],
            "trade_risk_usd": [100.0, 100.0, 100.0, 100.0],
            "account_size_usd": [10_000.0, 10_000.0, 10_000.0, 10_000.0],
        }
    )
    signal_df = pd.DataFrame(
        {
            "session_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-05"]
            ).date,
            "raw_signal": [1, 1, 1, 1, 1, 0],
            "signal": [1, 0, 1, 0, 1, 0],
        }
    )

    metrics = compute_metrics(
        trades,
        signal_df=signal_df,
        session_dates=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]).date,
        initial_capital=10_000.0,
        prop_constraints=PropConstraintConfig(
            name="unit_test",
            account_size_usd=10_000.0,
            max_loss_limit_usd=300.0,
            daily_loss_limit_usd=150.0,
            profit_target_usd=50.0,
            monthly_subscription_cost_usd=50.0,
            trading_days_per_month=20.0,
        ),
    )

    assert metrics["n_trades"] == 4
    assert metrics["longest_loss_streak"] == 2
    assert metrics["number_of_loss_streaks_2_plus"] == 1
    assert metrics["worst_day"] == pytest.approx(-200.0)
    assert metrics["stop_hit_rate"] == pytest.approx(0.5)
    assert metrics["target_hit_rate"] == pytest.approx(0.25)
    assert metrics["eod_exit_rate"] == pytest.approx(0.25)
    assert metrics["proportion_filtered_out"] == pytest.approx(0.4)
    assert metrics["percent_of_days_traded"] == pytest.approx(0.75)
    assert metrics["percent_days_traded"] == pytest.approx(0.75)
    assert metrics["avg_R"] == pytest.approx(0.25)
    assert metrics["pnl_to_risk_ratio"] == pytest.approx(0.25)
    assert metrics["average_loss_streak_length"] == pytest.approx(2.0)
    assert metrics["count_loss_streaks_2_plus"] == 1
    assert metrics["profit_target_reached"] is True
    assert metrics["days_to_profit_target"] == 2
    assert metrics["days_to_reach_3000_profit_target"] == 2
    assert metrics["profit_target_reached_before_max_loss"] is True
    assert metrics["breaches_max_loss_limit"] is False
    assert metrics["any_daily_loss_limit_breach"] is True
    assert metrics["number_of_daily_loss_limit_breaches"] == 1
    assert metrics["minimum_equity_before_target_hit"] == pytest.approx(9800.0)
    assert metrics["max_adverse_drawdown_before_target_hit"] == pytest.approx(200.0)
    assert metrics["subscription_drag_estimate"] == pytest.approx(5.0)
    assert metrics["estimated_pnl_after_subscription"] == pytest.approx(95.0)
