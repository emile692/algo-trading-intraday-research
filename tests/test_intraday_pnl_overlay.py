from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.intraday_pnl_overlay import IntradayPnlOverlaySpec, run_intraday_pnl_overlay_backtest
from src.analytics.orb_multi_asset_campaign import BaselineSpec
from src.engine.backtester import run_backtest
from src.engine.execution_model import ExecutionModel


def _baseline_spec(time_exit: str = "09:35:00") -> BaselineSpec:
    return BaselineSpec(
        or_minutes=30,
        opening_time="09:30:00",
        direction="long",
        one_trade_per_day=False,
        entry_buffer_ticks=0,
        stop_buffer_ticks=0,
        target_multiple=2.0,
        vwap_confirmation=False,
        time_exit=time_exit,
        account_size_usd=50_000.0,
        risk_per_trade_pct=1.5,
        entry_on_next_open=True,
    )


def _single_trade_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
            "2024-01-02 09:33:00",
        ]
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "signal": [1, 0, 0],
            "or_high": [100.0, 100.0, 100.0],
            "or_low": [99.0, 99.0, 99.0],
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 102.1],
            "low": [100.0, 99.8, 100.0],
            "close": [100.0, 100.1, 102.0],
        }
    )


def _profit_lock_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
            "2024-01-02 09:33:00",
            "2024-01-02 09:34:00",
        ]
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "signal": [1, 0, 0, 0],
            "or_high": [100.0, 100.0, 100.0, 100.0],
            "or_low": [99.0, 99.0, 99.0, 99.0],
            "open": [100.0, 100.0, 100.5, 100.1],
            "high": [100.0, 100.6, 101.1, 100.3],
            "low": [100.0, 99.8, 100.8, 99.9],
            "close": [100.0, 100.5, 101.0, 100.0],
        }
    )


def _loss_cap_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
            "2024-01-02 09:33:00",
            "2024-01-02 09:34:00",
        ]
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "signal": [1, 0, 0, 0],
            "or_high": [100.0, 100.0, 100.0, 100.0],
            "or_low": [99.0, 99.0, 99.0, 99.0],
            "open": [100.0, 100.0, 99.8, 99.7],
            "high": [100.0, 100.1, 100.0, 99.8],
            "low": [100.0, 99.6, 99.4, 98.9],
            "close": [100.0, 99.7, 99.5, 99.0],
        }
    )


def _giveback_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
            "2024-01-02 09:33:00",
            "2024-01-02 09:34:00",
            "2024-01-02 09:35:00",
        ]
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "signal": [1, 0, 0, 0, 0],
            "or_high": [100.0, 100.0, 100.0, 100.0, 100.0],
            "or_low": [99.0, 99.0, 99.0, 99.0, 99.0],
            "open": [100.0, 100.0, 100.8, 101.2, 100.9],
            "high": [100.0, 100.7, 101.6, 101.5, 101.0],
            "low": [100.0, 99.9, 100.6, 100.8, 100.0],
            "close": [100.0, 100.6, 101.4, 100.9, 100.2],
        }
    )


def _two_trade_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
            "2024-01-02 09:33:00",
            "2024-01-02 09:34:00",
            "2024-01-02 09:35:00",
            "2024-01-02 09:36:00",
        ]
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.normalize().date,
            "signal": [1, 0, 0, 1, 0, 0],
            "or_high": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "or_low": [99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
            "open": [100.0, 100.0, 99.5, 100.0, 100.0, 100.6],
            "high": [100.0, 100.1, 99.6, 100.0, 100.8, 101.2],
            "low": [100.0, 98.9, 99.3, 100.0, 99.8, 100.4],
            "close": [100.0, 99.0, 99.5, 100.0, 100.7, 101.0],
        }
    )


def test_overlay_baseline_matches_plain_backtester_for_fixed_nominal() -> None:
    frame = _single_trade_frame()
    baseline = _baseline_spec(time_exit="09:33:00")
    execution = ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0.0, tick_size=0.25)

    expected = run_backtest(
        frame,
        execution_model=execution,
        tick_value_usd=0.5,
        time_exit=baseline.time_exit,
        stop_buffer_ticks=baseline.stop_buffer_ticks,
        target_multiple=baseline.target_multiple,
        account_size_usd=None,
        risk_per_trade_pct=None,
        entry_on_next_open=baseline.entry_on_next_open,
    )
    got = run_intraday_pnl_overlay_backtest(
        frame,
        execution_model=execution,
        baseline=baseline,
        overlay=IntradayPnlOverlaySpec(name="baseline", family="baseline", description="none"),
        fixed_contracts=1,
        tick_value_usd=0.5,
    )

    assert len(expected) == len(got.trades) == 1
    assert got.trades.iloc[0]["exit_reason"] == expected.iloc[0]["exit_reason"]
    assert got.trades.iloc[0]["net_pnl_usd"] == pytest.approx(expected.iloc[0]["net_pnl_usd"])


def test_hard_profit_lock_exits_trade_early() -> None:
    baseline = _baseline_spec(time_exit="09:34:00")
    execution = ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0.0, tick_size=0.25)

    result = run_intraday_pnl_overlay_backtest(
        _profit_lock_frame(),
        execution_model=execution,
        baseline=baseline,
        overlay=IntradayPnlOverlaySpec(
            name="profit_lock",
            family="hard_cap",
            description="profit lock",
            hard_profit_lock=1.0,
        ),
        fixed_contracts=1,
        tick_value_usd=0.5,
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "overlay_hard_profit_lock"
    assert trade["exit_time"] == pd.Timestamp("2024-01-02 09:33:00")
    assert trade["net_pnl_usd"] == pytest.approx(2.0)
    assert bool(result.daily_results.iloc[0]["hard_profit_lock_triggered"]) is True


def test_hard_loss_cap_cuts_trade_before_baseline_stop() -> None:
    baseline = _baseline_spec(time_exit="09:34:00")
    execution = ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0.0, tick_size=0.25)

    result = run_intraday_pnl_overlay_backtest(
        _loss_cap_frame(),
        execution_model=execution,
        baseline=baseline,
        overlay=IntradayPnlOverlaySpec(
            name="loss_cap",
            family="hard_cap",
            description="loss cap",
            hard_loss_cap=-0.5,
        ),
        fixed_contracts=1,
        tick_value_usd=0.5,
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "overlay_hard_loss_cap"
    assert trade["exit_time"] == pd.Timestamp("2024-01-02 09:33:00")
    assert trade["net_pnl_usd"] == pytest.approx(-1.0)
    assert bool(result.daily_results.iloc[0]["hard_loss_cap_triggered"]) is True


def test_giveback_rule_uses_intrabar_peak_then_exits_on_retrace() -> None:
    baseline = _baseline_spec(time_exit="09:35:00")
    execution = ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0.0, tick_size=0.25)

    result = run_intraday_pnl_overlay_backtest(
        _giveback_frame(),
        execution_model=execution,
        baseline=baseline,
        overlay=IntradayPnlOverlaySpec(
            name="giveback",
            family="giveback",
            description="giveback",
            giveback_activation=1.5,
            giveback_threshold=0.5,
        ),
        fixed_contracts=1,
        tick_value_usd=0.5,
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "overlay_peak_giveback_lock"
    assert trade["exit_time"] == pd.Timestamp("2024-01-02 09:34:00")
    assert trade["net_pnl_usd"] == pytest.approx(1.8)
    assert bool(result.daily_results.iloc[0]["giveback_triggered"]) is True


def test_first_trade_win_gate_blocks_second_trade_after_loss() -> None:
    baseline = _baseline_spec(time_exit="09:36:00")
    execution = ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0.0, tick_size=0.25)

    result = run_intraday_pnl_overlay_backtest(
        _two_trade_frame(),
        execution_model=execution,
        baseline=baseline,
        overlay=IntradayPnlOverlaySpec(
            name="first_trade_gate",
            family="sequence",
            description="gate",
            continue_only_if_first_trade_wins=True,
        ),
        fixed_contracts=1,
        tick_value_usd=0.5,
    )

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "stop"
    blocked = result.controls.loc[result.controls["allowed_to_trade"].eq(False)]
    assert len(blocked) == 1
    assert blocked.iloc[0]["blocked_reason"] == "state_halted"
    assert not result.state_transitions.empty
    assert result.state_transitions.iloc[0]["to_state"] == "halted"


def test_defensive_mode_reduces_quantity_on_second_trade_when_possible() -> None:
    baseline = _baseline_spec(time_exit="09:36:00")
    execution = ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0.0, tick_size=0.25)

    result = run_intraday_pnl_overlay_backtest(
        _two_trade_frame(),
        execution_model=execution,
        baseline=baseline,
        overlay=IntradayPnlOverlaySpec(
            name="defensive_after_loss",
            family="state_machine",
            description="defensive",
            defensive_after_total_losses=1,
            defensive_multiplier=0.5,
        ),
        fixed_contracts=2,
        tick_value_usd=0.5,
    )

    assert len(result.trades) == 2
    assert int(result.trades.iloc[0]["quantity"]) == 2
    assert int(result.trades.iloc[1]["quantity"]) == 1
    assert not result.state_transitions.empty
    assert "defensive" in result.state_transitions["to_state"].tolist()
