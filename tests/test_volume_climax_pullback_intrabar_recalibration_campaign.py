from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analytics import volume_climax_pullback_intrabar_recalibration_campaign as campaign
from src.engine.execution_model import ExecutionModel
from src.engine.vwap_backtester import InstrumentDetails
from src.strategy.volume_climax_pullback_v2 import VolumeClimaxPullbackV2Variant


TZ = "America/New_York"


def _variant(**overrides: object) -> VolumeClimaxPullbackV2Variant:
    payload: dict[str, object] = {
        "name": "intrabar_recalibration_test",
        "family": "dynamic_exit",
        "timeframe": "1h",
        "volume_quantile": 0.95,
        "volume_lookback": 50,
        "min_body_fraction": 0.5,
        "min_range_atr": 1.2,
        "trend_ema_window": None,
        "ema_slope_threshold": None,
        "atr_percentile_low": None,
        "atr_percentile_high": None,
        "compression_ratio_max": None,
        "entry_mode": "next_open",
        "pullback_fraction": None,
        "confirmation_window": None,
        "exit_mode": "fixed_rr",
        "rr_target": 1.0,
        "atr_target_multiple": None,
        "time_stop_bars": 2,
        "trailing_atr_multiple": 0.5,
        "session_overlay": "all_rth",
    }
    payload.update(overrides)
    return VolumeClimaxPullbackV2Variant(**payload)


def _config(
    *,
    config_id: str,
    stop_multiplier: float = 1.0,
    target_multiplier: float = 1.0,
    entry_delay_minutes: int = 0,
    filter_family: str = "none",
    filter_label: str = "none",
    filter_params: dict[str, object] | None = None,
) -> campaign.IntrabarRecalibrationConfig:
    return campaign.IntrabarRecalibrationConfig(
        config_id=config_id,
        symbol="MNQ",
        execution_timeframe="1min",
        entry_timing="next_execution_bar_open",
        protective_orders_active_from="next_execution_bar",
        ambiguous_policy="stop_first",
        stop_multiplier=stop_multiplier,
        target_multiplier=target_multiplier,
        entry_delay_minutes=entry_delay_minutes,
        filter_family=filter_family,
        filter_label=filter_label,
        filter_params={} if filter_params is None else dict(filter_params),
    )


def _minute_rows(start: str, end: str, default_open: float = 100.0) -> pd.DataFrame:
    timestamps = pd.date_range(start, end, freq="1min", tz=TZ)
    rows: list[dict[str, object]] = []
    for timestamp in timestamps:
        rows.append(
            {
                "timestamp": timestamp,
                "open": default_open,
                "high": default_open + 0.10,
                "low": default_open - 0.10,
                "close": default_open,
                "volume": 100.0,
                "session_date": timestamp.date(),
            }
        )
    return pd.DataFrame(rows)


def _fake_signal_df() -> pd.DataFrame:
    rows = []
    for date_str in ("2023-12-29", "2024-01-02"):
        actionable = pd.Timestamp(f"{date_str} 14:30:00", tz=TZ)
        setup = pd.Timestamp(f"{date_str} 13:30:00", tz=TZ)
        rows.append(
            {
                "timestamp": actionable,
                "session_date": actionable.date(),
                "open": 100.0,
                "high": 100.2,
                "low": 99.8,
                "close": 100.0,
                "signal": 1,
                "setup_signal_time": setup,
                "setup_reference_close": 100.0,
                "setup_reference_range": 1.0,
                "setup_reference_atr": 1.0,
                "setup_reference_vwap": 100.0,
                "setup_stop_reference_long": 99.0,
                "setup_stop_reference_short": pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _fake_execution_profile(symbol: str, profile_name: str) -> tuple[ExecutionModel, InstrumentDetails]:
    return (
        ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0, tick_size=0.25),
        InstrumentDetails(
            symbol=symbol,
            asset_class="futures",
            tick_size=0.25,
            tick_value_usd=0.5,
            point_value_usd=1.0,
            commission_per_side_usd=0.0,
            slippage_ticks=0,
        ),
    )


def test_stop_target_multiplier_changes_levels() -> None:
    long_stop, long_target = campaign.adjust_protective_levels(
        direction=1,
        entry_price=100.0,
        raw_stop_price=99.0,
        base_target_distance=2.0,
        stop_multiplier=1.5,
        target_multiplier=2.0,
    )
    short_stop, short_target = campaign.adjust_protective_levels(
        direction=-1,
        entry_price=100.0,
        raw_stop_price=101.0,
        base_target_distance=2.0,
        stop_multiplier=1.5,
        target_multiplier=2.0,
    )

    assert long_stop == 98.5
    assert long_target == 104.0
    assert short_stop == 101.5
    assert short_target == 96.0


def test_entry_delay_uses_future_minutes_only() -> None:
    session_minutes = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 14:31:00", periods=10, freq="1min", tz=TZ),
            "open": [100.0] * 10,
            "high": [100.0] * 10,
            "low": [100.0] * 10,
            "close": [100.0] * 10,
        }
    )

    base_idx, delayed_idx = campaign.resolve_delayed_entry_index(
        session_minutes,
        actionable_time=pd.Timestamp("2024-01-02 14:30:00", tz=TZ),
        entry_delay_minutes=5,
    )

    assert base_idx == 0
    assert delayed_idx == 5
    assert pd.Timestamp(session_minutes.iloc[delayed_idx]["timestamp"]) == pd.Timestamp("2024-01-02 14:36:00", tz=TZ)


def test_pre_entry_filter_skip_adverse_move() -> None:
    config = _config(
        config_id="adverse_skip",
        entry_delay_minutes=5,
        filter_family="avoid_immediate_adverse_move",
        filter_label="adverse_w5_ticks8",
        filter_params={"adverse_window_minutes": 5, "max_adverse_ticks": 8},
    )
    pre_entry_path = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2024-01-02 14:31:00", tz=TZ), "open": 100.0, "high": 100.1, "low": 99.8, "close": 99.9},
            {"timestamp": pd.Timestamp("2024-01-02 14:32:00", tz=TZ), "open": 99.9, "high": 100.0, "low": 97.9, "close": 98.2},
        ]
    )

    skip_setup, filter_reason = campaign.evaluate_pre_entry_filter(
        config,
        direction=1,
        pre_entry_path=pre_entry_path,
        actual_entry_price=100.0,
        adjusted_stop_price=99.0,
        tick_size=0.25,
    )

    assert skip_setup is True
    assert filter_reason == "skip_adverse_move_before_entry"


def test_is_only_selection_does_not_use_oos() -> None:
    metrics_with_scores = pd.DataFrame(
        [
            {
                "config_id": "good_is",
                "admissible_is": True,
                "robust_score_is": 0.85,
                "net_pnl": 500.0,
                "profit_factor": 1.30,
                "trades": 80,
                "stop_multiplier": 1.0,
                "target_multiplier": 1.5,
                "entry_delay_minutes": 5,
                "filter_family": "none",
                "net_pnl_oos": -500.0,
            },
            {
                "config_id": "good_oos_only",
                "admissible_is": False,
                "robust_score_is": 0.10,
                "net_pnl": -50.0,
                "profit_factor": 0.95,
                "trades": 20,
                "stop_multiplier": 0.75,
                "target_multiplier": 2.0,
                "entry_delay_minutes": 15,
                "filter_family": "avoid_high_noise_first_5min",
                "net_pnl_oos": 5000.0,
            },
        ]
    )

    selected = campaign.select_configs_is_only(metrics_with_scores, max_configs=1)

    assert list(selected["config_id"]) == ["good_is"]


def test_robustness_score_penalizes_one_year_dependency() -> None:
    metrics_is = pd.DataFrame(
        [
            {
                "config_id": "balanced",
                "net_pnl": 1000.0,
                "profit_factor": 1.30,
                "pnl_to_maxdd": 2.0,
                "trades": 80,
                "avg_trade": 20.0,
                "skip_rate": 0.10,
                "stop_multiplier": 1.0,
                "target_multiplier": 1.5,
                "entry_delay_minutes": 5,
                "filter_family": "none",
            },
            {
                "config_id": "concentrated",
                "net_pnl": 1000.0,
                "profit_factor": 1.30,
                "pnl_to_maxdd": 2.0,
                "trades": 80,
                "avg_trade": 20.0,
                "skip_rate": 0.10,
                "stop_multiplier": 1.25,
                "target_multiplier": 1.5,
                "entry_delay_minutes": 5,
                "filter_family": "none",
            },
        ]
    )
    subperiod_metrics = pd.DataFrame(
        [
            {"config_id": "balanced", "subperiod": "2020", "net_pnl": 250.0, "trades": 20},
            {"config_id": "balanced", "subperiod": "2021", "net_pnl": 250.0, "trades": 20},
            {"config_id": "balanced", "subperiod": "2022", "net_pnl": 250.0, "trades": 20},
            {"config_id": "balanced", "subperiod": "2023", "net_pnl": 250.0, "trades": 20},
            {"config_id": "concentrated", "subperiod": "2020", "net_pnl": 1000.0, "trades": 80},
            {"config_id": "concentrated", "subperiod": "2021", "net_pnl": 0.0, "trades": 0},
            {"config_id": "concentrated", "subperiod": "2022", "net_pnl": 0.0, "trades": 0},
            {"config_id": "concentrated", "subperiod": "2023", "net_pnl": 0.0, "trades": 0},
        ]
    )

    scored = campaign.build_robustness_scores(metrics_is, subperiod_metrics, estimated_cost_per_trade=2.0)
    balanced = scored.loc[scored["config_id"] == "balanced"].iloc[0]
    concentrated = scored.loc[scored["config_id"] == "concentrated"].iloc[0]

    assert bool(concentrated["one_year_dependency"]) is True
    assert float(concentrated["penalties"]) > float(balanced["penalties"])
    assert float(balanced["robust_score_is"]) > float(concentrated["robust_score_is"])


def test_campaign_outputs_required_files_smoke(tmp_path: Path, monkeypatch) -> None:
    minute_df = pd.concat(
        [
            _minute_rows("2023-12-29 13:30:00", "2023-12-29 16:00:00"),
            _minute_rows("2024-01-02 13:30:00", "2024-01-02 16:00:00"),
        ],
        ignore_index=True,
    )

    def _set_bar(timestamp: str, *, open_: float, high: float, low: float, close: float) -> None:
        ts = pd.Timestamp(timestamp, tz=TZ)
        mask = minute_df["timestamp"] == ts
        minute_df.loc[mask, ["open", "high", "low", "close"]] = [open_, high, low, close]

    _set_bar("2023-12-29 14:37:00", open_=100.0, high=101.6, low=99.9, close=101.3)
    _set_bar("2024-01-02 14:37:00", open_=100.0, high=100.1, low=98.4, close=98.8)

    signal_df = _fake_signal_df()
    benchmark_metrics = pd.DataFrame(
        [
            {"scenario": "baseline_1h", "trades": 2, "net_pnl_usd": 50.0, "profit_factor": 1.2, "sharpe": 0.5, "max_drawdown_usd": -10.0, "expectancy_usd": 25.0, "hit_rate": 0.5, "raw_signal_count": 2, "avg_minutes_held": None},
            {"scenario": "hybrid_after_entry_fill", "trades": 2, "net_pnl_usd": -5.0, "profit_factor": 0.9, "sharpe": -0.1, "max_drawdown_usd": -15.0, "expectancy_usd": -2.5, "hit_rate": 0.5, "raw_signal_count": 2, "avg_minutes_held": 5.0},
            {"scenario": "hybrid_next_execution_bar", "trades": 2, "net_pnl_usd": -4.0, "profit_factor": 0.95, "sharpe": -0.1, "max_drawdown_usd": -12.0, "expectancy_usd": -2.0, "hit_rate": 0.5, "raw_signal_count": 2, "avg_minutes_held": 5.0},
        ]
    )
    benchmark_context = {
        "metrics_comparison": benchmark_metrics,
        "baseline": benchmark_metrics.loc[benchmark_metrics["scenario"] == "baseline_1h"].iloc[0].to_dict(),
        "hybrid_after": benchmark_metrics.loc[benchmark_metrics["scenario"] == "hybrid_after_entry_fill"].iloc[0].to_dict(),
        "hybrid_next": benchmark_metrics.loc[benchmark_metrics["scenario"] == "hybrid_next_execution_bar"].iloc[0].to_dict(),
    }
    configs = [
        _config(config_id="current_none_sm1p00_tm1p00_d0", stop_multiplier=1.0, target_multiplier=1.0, entry_delay_minutes=0),
        _config(config_id="delay_none_sm0p75_tm1p50_d5", stop_multiplier=0.75, target_multiplier=1.5, entry_delay_minutes=5),
    ]

    monkeypatch.setattr(campaign, "prepare_volume_climax_pullback_v2_features", lambda frame: frame)
    monkeypatch.setattr(campaign, "build_volume_climax_pullback_v2_signal_frame", lambda features, variant: signal_df.copy())
    monkeypatch.setattr(campaign, "build_execution_model_for_profile", _fake_execution_profile)

    diagnostics_dir = tmp_path / "diagnostics"
    diagnostics_dir.mkdir()
    (diagnostics_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "verdict_context": {
                    "top_divergence_cause": "winner_to_loser",
                    "best_recalibration_net_pnl": 10.0,
                    "best_recalibration_stop_multiplier": 0.75,
                    "best_recalibration_target_multiplier": 2.0,
                },
                "matching_info": {"matching_confidence": "high"},
            }
        ),
        encoding="utf-8",
    )

    run_dir = campaign.run_campaign(
        symbol="MNQ",
        diagnostics_dir=diagnostics_dir,
        validation_dir=tmp_path / "validation",
        output_root=tmp_path,
        raw_minute_df_override=minute_df,
        variant_override=_variant(),
        configs_override=configs,
        benchmark_context_override=benchmark_context,
    )

    required_files = [
        "config_metrics_is.csv",
        "config_metrics_oos.csv",
        "config_metrics_full.csv",
        "config_subperiod_metrics.csv",
        "config_robustness_scores.csv",
        "selected_configs_is_only.csv",
        "selected_configs_oos_report.csv",
        "best_candidate_trade_audit.csv",
        "best_candidate_daily_returns.csv",
        "best_candidate_summary.json",
        "all_config_trades.csv",
        "final_report.md",
        "run_metadata.json",
    ]

    for filename in required_files:
        assert (run_dir / filename).exists(), filename

