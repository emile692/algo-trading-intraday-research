from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analytics import volume_climax_pullback_walkforward_intrabar_validation as wf
from src.strategy.volume_climax_pullback_v2 import VolumeClimaxPullbackV2Variant


TZ = "America/New_York"


def _variant() -> VolumeClimaxPullbackV2Variant:
    return VolumeClimaxPullbackV2Variant(
        name="walkforward_test",
        family="dynamic_exit",
        timeframe="1h",
        volume_quantile=0.95,
        volume_lookback=50,
        min_body_fraction=0.5,
        min_range_atr=1.2,
        trend_ema_window=None,
        ema_slope_threshold=None,
        atr_percentile_low=None,
        atr_percentile_high=None,
        compression_ratio_max=None,
        entry_mode="next_open",
        pullback_fraction=None,
        confirmation_window=None,
        exit_mode="fixed_rr",
        rr_target=1.0,
        atr_target_multiple=None,
        time_stop_bars=2,
        trailing_atr_multiple=0.5,
        session_overlay="all_rth",
    )


def _config(
    *,
    config_id: str,
    family: str,
    stop_multiplier: float,
    target_multiplier: float,
    entry_delay_minutes: int,
    filter_name: str = "none",
    stop_zone_fraction: float | None = None,
) -> wf.WalkforwardIntrabarConfig:
    return wf.WalkforwardIntrabarConfig(
        config_id=config_id,
        family=family,
        symbol="MNQ",
        stop_multiplier=stop_multiplier,
        target_multiplier=target_multiplier,
        entry_delay_minutes=entry_delay_minutes,
        filter_name=filter_name,
        stop_zone_fraction=stop_zone_fraction,
    )


def _events(dates: list[str], pnls: list[float]) -> pd.DataFrame:
    rows = []
    for raw_date, pnl in zip(dates, pnls, strict=True):
        ts = pd.Timestamp(f"{raw_date} 10:31:00", tz=TZ)
        rows.append(
            {
                "signal_time": ts - pd.Timedelta(hours=1),
                "signal_actionable_time": ts - pd.Timedelta(minutes=1),
                "direction": "long",
                "executed": True,
                "session_date": ts.date(),
                "entry_time": ts,
                "exit_time": ts + pd.Timedelta(minutes=5),
                "entry_price": 100.0,
                "stop_price": 99.0,
                "target_price": 102.0,
                "exit_price": 100.0 + pnl,
                "exit_reason": "target_1m" if pnl > 0 else "stop_1m",
                "holding_minutes": 5.0,
                "net_pnl_usd": float(pnl),
                "gross_pnl_usd": float(pnl),
                "pnl": float(pnl),
                "skipped_by_filter": False,
                "filter_reason": "none",
                "quantity": 1,
                "mfe_ticks_1m": 4.0,
                "mae_ticks_1m": 2.0,
            }
        )
    return pd.DataFrame(rows)


def _raw_minute_dates() -> pd.DataFrame:
    timestamps = [
        pd.Timestamp("2020-01-02 10:00:00", tz=TZ),
        pd.Timestamp("2021-06-01 10:00:00", tz=TZ),
        pd.Timestamp("2022-06-01 10:00:00", tz=TZ),
        pd.Timestamp("2023-06-01 10:00:00", tz=TZ),
    ]
    return pd.DataFrame(
        [
            {
                "timestamp": ts,
                "open": 100.0,
                "high": 100.2,
                "low": 99.8,
                "close": 100.0,
                "volume": 100.0,
                "session_date": ts.date(),
            }
            for ts in timestamps
        ]
    )


def _metadata_dirs(tmp_path: Path) -> tuple[Path, Path]:
    phase2_dir = tmp_path / "phase2"
    validation_dir = tmp_path / "validation"
    phase2_dir.mkdir()
    validation_dir.mkdir()
    (phase2_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "dataset_path": str(tmp_path / "unused.parquet"),
                "current_hybrid_config_id": "none_sm1p00_tm1p00_d0",
            }
        ),
        encoding="utf-8",
    )
    (validation_dir / "run_metadata.json").write_text(
        json.dumps({"dataset_path": str(tmp_path / "unused.parquet"), "variant": {}}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"scenario": "baseline_1h", "trades": 2, "net_pnl_usd": 10.0, "profit_factor": 1.1, "sharpe": 0.0, "max_drawdown_usd": -5.0, "expectancy_usd": 5.0, "hit_rate": 0.5, "raw_signal_count": 2, "avg_minutes_held": None},
            {"scenario": "hybrid_after_entry_fill", "trades": 2, "net_pnl_usd": -2.0, "profit_factor": 0.9, "sharpe": 0.0, "max_drawdown_usd": -5.0, "expectancy_usd": -1.0, "hit_rate": 0.5, "raw_signal_count": 2, "avg_minutes_held": 5.0},
            {"scenario": "hybrid_next_execution_bar", "trades": 2, "net_pnl_usd": -3.0, "profit_factor": 0.85, "sharpe": 0.0, "max_drawdown_usd": -5.0, "expectancy_usd": -1.5, "hit_rate": 0.5, "raw_signal_count": 2, "avg_minutes_held": 5.0},
        ]
    ).to_csv(validation_dir / "metrics_comparison.csv", index=False)
    return phase2_dir, validation_dir


def test_walkforward_folds_are_strictly_ordered() -> None:
    session_dates = [
        pd.Timestamp("2020-01-02").date(),
        pd.Timestamp("2021-06-01").date(),
        pd.Timestamp("2022-06-01").date(),
        pd.Timestamp("2023-06-01").date(),
        pd.Timestamp("2024-06-01").date(),
        pd.Timestamp("2025-06-01").date(),
        pd.Timestamp("2026-03-01").date(),
    ]
    folds = wf.build_walkforward_folds(session_dates)

    assert len(folds) == 5
    for fold in folds:
        assert fold.train_end < fold.test_start


def test_selection_uses_train_only() -> None:
    fold = wf.WalkforwardFold("fold_1", pd.Timestamp("2020-01-01").date(), pd.Timestamp("2021-12-31").date(), pd.Timestamp("2022-01-01").date(), pd.Timestamp("2022-12-31").date())
    config_universe_df = pd.DataFrame(
        [
            {"config_id": "config_a", "family": "delay_only", "filter_name": "none", "stop_multiplier": 1.0, "target_multiplier": 2.0, "entry_delay_minutes": 5, "stop_zone_fraction": pd.NA},
            {"config_id": "config_b", "family": "delay_only", "filter_name": "none", "stop_multiplier": 1.25, "target_multiplier": 2.0, "entry_delay_minutes": 5, "stop_zone_fraction": pd.NA},
        ]
    )
    train_dates = [f"2020-01-{day:02d}" for day in range(1, 21)] + [f"2021-01-{day:02d}" for day in range(1, 21)]
    test_dates = [f"2022-01-{day:02d}" for day in range(1, 13)]
    events_by_config = {
        "config_a": pd.concat([_events(train_dates, [-1.0] * len(train_dates)), _events(test_dates, [3.0] * len(test_dates))], ignore_index=True),
        "config_b": pd.concat([_events(train_dates, [2.0] * len(train_dates)), _events(test_dates, [-2.0] * len(test_dates))], ignore_index=True),
    }

    ranking = wf.compute_fold_train_ranking(fold, config_universe_df, events_by_config, estimated_cost_per_trade=0.25)
    winner = wf.select_fold_winner(ranking)

    assert str(winner["config_id"]) == "config_b"


def test_fixed_candidate_tracking_does_not_affect_selection(tmp_path: Path, monkeypatch) -> None:
    phase2_dir, validation_dir = _metadata_dirs(tmp_path)
    monkeypatch.setattr(wf, "prepare_volume_climax_pullback_v2_features", lambda frame: frame)
    monkeypatch.setattr(wf, "build_volume_climax_pullback_v2_signal_frame", lambda features, variant: pd.DataFrame())

    benchmark = _config(config_id="none_sm1p00_tm1p00_d0", family="benchmark_current_hybrid", stop_multiplier=1.0, target_multiplier=1.0, entry_delay_minutes=0)
    selected = _config(config_id="none_sm1p00_tm2p00_d5", family="delay_only", stop_multiplier=1.0, target_multiplier=2.0, entry_delay_minutes=5)
    fixed = _config(
        config_id=wf.FIXED_CANDIDATE_CONFIG_ID,
        family="sanity_anti_overfit",
        stop_multiplier=1.0,
        target_multiplier=2.5,
        entry_delay_minutes=5,
        filter_name="require_no_stop_zone_touch_before_entry",
        stop_zone_fraction=0.75,
    )
    train_dates = [f"2020-01-{day:02d}" for day in range(1, 21)] + [f"2021-01-{day:02d}" for day in range(1, 21)] + [f"2022-01-{day:02d}" for day in range(1, 21)]
    test_dates_2022 = [f"2022-06-{day:02d}" for day in range(1, 13)]
    test_dates_2023 = [f"2023-01-{day:02d}" for day in range(1, 13)]
    events_by_config = {
        "none_sm1p00_tm1p00_d0": pd.concat([_events(train_dates, [-1.0] * len(train_dates)), _events(test_dates_2022 + test_dates_2023, [-1.0] * (len(test_dates_2022) + len(test_dates_2023)))], ignore_index=True),
        "none_sm1p00_tm2p00_d5": pd.concat([_events(train_dates, [2.0] * len(train_dates)), _events(test_dates_2022 + test_dates_2023, [-0.5] * (len(test_dates_2022) + len(test_dates_2023)))], ignore_index=True),
        wf.FIXED_CANDIDATE_CONFIG_ID: pd.concat([_events(train_dates, [0.5] * len(train_dates)), _events(test_dates_2022 + test_dates_2023, [1.5] * (len(test_dates_2022) + len(test_dates_2023)))], ignore_index=True),
    }

    run_dir = wf.run_walkforward_validation(
        symbol="MNQ",
        phase2_dir=phase2_dir,
        validation_dir=validation_dir,
        output_root=tmp_path,
        raw_minute_df_override=_raw_minute_dates(),
        variant_override=_variant(),
        config_universe_override=[benchmark, selected, fixed],
        events_by_config_override=events_by_config,
    )
    selected_results = pd.read_csv(run_dir / "fold_selected_test_results.csv")
    fixed_tracking = pd.read_csv(run_dir / "fixed_candidate_oos_tracking.csv")

    assert not selected_results.empty
    assert set(selected_results["selected_config_id"]) == {"none_sm1p00_tm2p00_d5"}
    assert set(fixed_tracking["config_id"]) == {wf.FIXED_CANDIDATE_CONFIG_ID}


def test_family_universe_contains_phase2_candidate() -> None:
    universe = wf.build_phase3_config_universe("MNQ")
    universe_ids = {config.config_id for config in universe}

    assert wf.FIXED_CANDIDATE_CONFIG_ID in universe_ids


def test_stitched_trades_use_only_test_windows(tmp_path: Path, monkeypatch) -> None:
    phase2_dir, validation_dir = _metadata_dirs(tmp_path)
    monkeypatch.setattr(wf, "prepare_volume_climax_pullback_v2_features", lambda frame: frame)
    monkeypatch.setattr(wf, "build_volume_climax_pullback_v2_signal_frame", lambda features, variant: pd.DataFrame())

    benchmark = _config(config_id="none_sm1p00_tm1p00_d0", family="benchmark_current_hybrid", stop_multiplier=1.0, target_multiplier=1.0, entry_delay_minutes=0)
    selected = _config(config_id="none_sm1p00_tm2p00_d5", family="delay_only", stop_multiplier=1.0, target_multiplier=2.0, entry_delay_minutes=5)
    fixed = _config(
        config_id=wf.FIXED_CANDIDATE_CONFIG_ID,
        family="sanity_anti_overfit",
        stop_multiplier=1.0,
        target_multiplier=2.5,
        entry_delay_minutes=5,
        filter_name="require_no_stop_zone_touch_before_entry",
        stop_zone_fraction=0.75,
    )
    dates = [f"2020-01-{day:02d}" for day in range(1, 21)] + [f"2021-01-{day:02d}" for day in range(1, 21)] + [f"2022-01-{day:02d}" for day in range(1, 21)] + [f"2023-01-{day:02d}" for day in range(1, 21)]
    events_by_config = {
        "none_sm1p00_tm1p00_d0": _events(dates, [-1.0] * len(dates)),
        "none_sm1p00_tm2p00_d5": _events(dates, [2.0] * 60 + [-0.5] * 20),
        wf.FIXED_CANDIDATE_CONFIG_ID: _events(dates, [1.0] * len(dates)),
    }

    run_dir = wf.run_walkforward_validation(
        symbol="MNQ",
        phase2_dir=phase2_dir,
        validation_dir=validation_dir,
        output_root=tmp_path,
        raw_minute_df_override=_raw_minute_dates(),
        variant_override=_variant(),
        config_universe_override=[benchmark, selected, fixed],
        events_by_config_override=events_by_config,
    )
    stitched = pd.read_csv(run_dir / "walkforward_stitched_trades.csv")
    folds = pd.read_csv(run_dir / "walkforward_folds.csv")

    stitched["session_date"] = pd.to_datetime(stitched["session_date"]).dt.date
    for _, fold in folds.iterrows():
        test_start = pd.Timestamp(fold["test_start"]).date()
        test_end = pd.Timestamp(fold["test_end"]).date()
        fold_rows = stitched.loc[stitched["fold_id"] == fold["fold_id"]]
        assert not fold_rows.empty
        assert fold_rows["session_date"].between(test_start, test_end).all()


def test_walkforward_summary_consistency(tmp_path: Path, monkeypatch) -> None:
    phase2_dir, validation_dir = _metadata_dirs(tmp_path)
    monkeypatch.setattr(wf, "prepare_volume_climax_pullback_v2_features", lambda frame: frame)
    monkeypatch.setattr(wf, "build_volume_climax_pullback_v2_signal_frame", lambda features, variant: pd.DataFrame())

    benchmark = _config(config_id="none_sm1p00_tm1p00_d0", family="benchmark_current_hybrid", stop_multiplier=1.0, target_multiplier=1.0, entry_delay_minutes=0)
    selected = _config(config_id="none_sm1p00_tm2p00_d5", family="delay_only", stop_multiplier=1.0, target_multiplier=2.0, entry_delay_minutes=5)
    fixed = _config(
        config_id=wf.FIXED_CANDIDATE_CONFIG_ID,
        family="sanity_anti_overfit",
        stop_multiplier=1.0,
        target_multiplier=2.5,
        entry_delay_minutes=5,
        filter_name="require_no_stop_zone_touch_before_entry",
        stop_zone_fraction=0.75,
    )
    dates = [f"2020-01-{day:02d}" for day in range(1, 21)] + [f"2021-01-{day:02d}" for day in range(1, 21)] + [f"2022-01-{day:02d}" for day in range(1, 21)] + [f"2023-01-{day:02d}" for day in range(1, 21)]
    events_by_config = {
        "none_sm1p00_tm1p00_d0": _events(dates, [-1.0] * len(dates)),
        "none_sm1p00_tm2p00_d5": _events(dates, [2.0] * 60 + [-0.5] * 20),
        wf.FIXED_CANDIDATE_CONFIG_ID: _events(dates, [1.0] * len(dates)),
    }

    run_dir = wf.run_walkforward_validation(
        symbol="MNQ",
        phase2_dir=phase2_dir,
        validation_dir=validation_dir,
        output_root=tmp_path,
        raw_minute_df_override=_raw_minute_dates(),
        variant_override=_variant(),
        config_universe_override=[benchmark, selected, fixed],
        events_by_config_override=events_by_config,
    )
    stitched = pd.read_csv(run_dir / "walkforward_stitched_trades.csv")
    summary = pd.read_csv(run_dir / "walkforward_summary.csv")

    assert abs(float(summary.iloc[0]["total_test_net_pnl"]) - float(pd.to_numeric(stitched["net_pnl_usd"], errors="coerce").sum())) < 1e-9

