from __future__ import annotations

import pandas as pd

from src.analytics.volume_climax_pullback_hybrid_execution_diagnostics import (
    build_pnl_bridge,
    classify_divergence,
    compute_first_touch,
    run_recalibration_grid,
)


TZ = "America/New_York"


def _minute_path(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    data = []
    for timestamp, open_, high, low, close in rows:
        ts = pd.Timestamp(timestamp, tz=TZ)
        data.append(
            {
                "timestamp": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "session_date": ts.date(),
            }
        )
    return pd.DataFrame(data)


def test_first_touch_long_stop_before_target() -> None:
    path = _minute_path(
        [
            ("2024-01-02 10:31:00", 100.0, 100.2, 99.8, 100.0),
            ("2024-01-02 10:32:00", 100.0, 100.1, 98.9, 99.2),
            ("2024-01-02 10:33:00", 99.2, 101.5, 99.1, 101.0),
        ]
    )
    result = compute_first_touch(path, direction="long", stop_price=99.0, target_price=101.0)
    assert result["first_touch"] == "stop"
    assert pd.Timestamp(result["first_touch_time"]) == pd.Timestamp("2024-01-02 10:32:00", tz=TZ)


def test_first_touch_long_both_same_minute_stop_first() -> None:
    path = _minute_path(
        [
            ("2024-01-02 10:31:00", 100.0, 101.2, 98.8, 100.3),
        ]
    )
    result = compute_first_touch(path, direction="long", stop_price=99.0, target_price=101.0)
    assert result["first_touch"] == "both_same_minute"
    assert result["ambiguous_policy_applied"] == "stop_first"


def test_first_touch_short_target_before_stop() -> None:
    path = _minute_path(
        [
            ("2024-01-02 10:31:00", 100.0, 100.1, 98.8, 99.0),
            ("2024-01-02 10:32:00", 99.0, 101.5, 98.9, 101.0),
        ]
    )
    result = compute_first_touch(path, direction="short", stop_price=101.0, target_price=99.0)
    assert result["first_touch"] == "target"
    assert pd.Timestamp(result["first_touch_time"]) == pd.Timestamp("2024-01-02 10:31:00", tz=TZ)


def test_divergence_taxonomy_winner_to_loser() -> None:
    row = pd.Series(
        {
            "matched_status": "matched",
            "baseline_pnl": 25.0,
            "hybrid_after_entry_fill_pnl": -10.0,
            "first_touch": "stop",
            "baseline_exit_reason": "target",
            "hybrid_after_exit_reason": "stop_1m",
            "entry_price_delta_ticks": -3.0,
        }
    )
    result = classify_divergence(row)
    assert result["divergence_type"] == "winner_to_loser"


def test_pnl_bridge_sums_to_hybrid() -> None:
    diagnostic = pd.DataFrame(
        [
            {
                "baseline_pnl": 100.0,
                "hybrid_after_entry_fill_pnl": 70.0,
                "delta_pnl_after_entry_fill": -30.0,
                "primary_pnl_driver": "entry_price_effect",
            },
            {
                "baseline_pnl": 50.0,
                "hybrid_after_entry_fill_pnl": 20.0,
                "delta_pnl_after_entry_fill": -30.0,
                "primary_pnl_driver": "stop_before_target_effect",
            },
        ]
    )
    bridge = build_pnl_bridge(diagnostic)
    baseline = float(bridge.loc[bridge["bridge_component"] == "baseline_net_pnl", "amount"].iloc[0])
    adjustments = float(
        bridge.loc[~bridge["bridge_component"].isin(["baseline_net_pnl", "hybrid_net_pnl"]), "amount"].sum()
    )
    hybrid = float(bridge.loc[bridge["bridge_component"] == "hybrid_net_pnl", "amount"].iloc[0])
    assert abs((baseline + adjustments) - hybrid) < 1e-9


def test_recalibration_grid_runs() -> None:
    minute_df = _minute_path(
        [
            ("2024-01-02 10:31:00", 100.0, 100.4, 99.8, 100.2),
            ("2024-01-02 10:32:00", 100.2, 101.2, 100.1, 101.0),
            ("2024-01-02 10:33:00", 101.0, 101.4, 100.8, 101.2),
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "session_date": pd.Timestamp("2024-01-02").date(),
                "direction": "long",
                "quantity": 1,
                "entry_time": pd.Timestamp("2024-01-02 10:31:00", tz=TZ),
                "entry_price": 100.0,
                "stop_price": 99.0,
                "initial_stop_price": 99.0,
                "target_price": 101.0,
                "time_stop_at": pd.Timestamp("2024-01-02 10:40:00", tz=TZ),
                "exit_time": pd.Timestamp("2024-01-02 10:33:00", tz=TZ),
                "fees": 0.0,
            }
        ]
    )
    grid = run_recalibration_grid(
        trades,
        minute_df,
        tick_size=0.25,
        point_value_usd=1.0,
        stop_multipliers=[0.75, 1.0],
        target_multipliers=[0.75, 1.0, 1.25],
    )
    assert len(grid) == 6
