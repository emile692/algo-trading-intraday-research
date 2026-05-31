from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics.orb_opposite_breakout_invalidation_campaign import (
    CampaignConfig,
    OppositeBreakoutInvalidationSpec,
    _read_dataframe_with_fallback,
    _write_dataframe_with_fallback,
    apply_opposite_invalidation_filter,
    build_fast_policy_grid,
    classify_first_breakout,
    detect_opening_range,
    run_campaign,
    select_policy_grid,
)


def _minute_frame(
    timestamp_strings: list[str],
    *,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    timestamps = pd.to_datetime(timestamp_strings).tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes or [100.0] * len(timestamps),
        }
    )


def _policy_session_frame(
    *,
    highs: list[float],
    lows: list[float],
    closes: list[float],
    raw_signal: list[int],
    filter_pass: list[bool],
    vwap: list[float] | None = None,
) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:45:00", periods=len(highs), freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": timestamps.date,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100.0] * len(highs),
            "or_high": [100.0] * len(highs),
            "or_low": [95.0] * len(highs),
            "eligible_post_or": [True] * len(highs),
            "raw_signal": raw_signal,
            "filter_pass": filter_pass,
            "signal": [0] * len(highs),
            "continuous_session_vwap": vwap or [94.5] * len(highs),
        }
    )


def _smoke_dataset(path: Path, sessions: int = 8) -> None:
    rows: list[dict[str, object]] = []
    session_dates = pd.bdate_range("2024-01-02", periods=sessions)
    for day_idx, session_date in enumerate(session_dates):
        session_open = pd.Timestamp(session_date.date()).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        base = 100.0 + day_idx * 0.25
        prev_close = base
        for minute_idx in range(75):
            ts = session_open + pd.Timedelta(minutes=minute_idx)
            if minute_idx < 15:
                close = base + ((minute_idx % 5) - 2) * 0.08
            elif day_idx % 2 == 0 and minute_idx == 18:
                close = base - 0.90
            elif day_idx % 2 == 0 and minute_idx in (19, 20, 21):
                close = base - 0.20 + (minute_idx - 19) * 0.25
            elif minute_idx >= 22:
                close = base + 0.70 + (minute_idx - 22) * 0.02
            else:
                close = base + 0.15 + max(0, minute_idx - 15) * 0.03
            open_price = prev_close
            high = max(open_price, close) + 0.10
            low = min(open_price, close) - 0.10
            rows.append(
                {
                    "timestamp": ts,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 100.0 + day_idx,
                }
            )
            prev_close = close
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_opening_range_is_built_from_or_window_only() -> None:
    frame = _minute_frame(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
            "2024-01-02 09:33:00",
            "2024-01-02 09:34:00",
            "2024-01-02 09:35:00",
        ],
        opens=[100, 100, 100, 100, 100, 100],
        highs=[101, 102, 103, 104, 105, 120],
        lows=[99, 98, 97, 96, 95, 80],
        closes=[100, 100, 100, 100, 100, 100],
    )
    config = CampaignConfig(or_minutes=5)

    out = detect_opening_range(frame, config)

    assert float(out["or_high"].iloc[-1]) == 105.0
    assert float(out["or_low"].iloc[-1]) == 95.0
    assert bool(out["eligible_post_or"].iloc[4]) is False
    assert bool(out["eligible_post_or"].iloc[5]) is True


def test_first_breakout_downside_is_detected() -> None:
    frame = _minute_frame(
        [f"2024-01-02 09:{minute:02d}:00" for minute in range(30, 48)],
        opens=[100.0] * 18,
        highs=[100.2] * 15 + [99.5, 101.0, 101.2],
        lows=[99.8] * 15 + [94.8, 99.6, 99.7],
        closes=[100.0] * 18,
    )
    config = CampaignConfig(or_minutes=15, entry_buffer_ticks=2)
    detected = detect_opening_range(frame, config)

    info = classify_first_breakout(detected, or_high=100.0, or_low=95.0, config=config)

    assert info["first_breakout"] == "first_breakout_downside"


def test_first_breakout_upside_is_detected() -> None:
    frame = _minute_frame(
        [f"2024-01-02 09:{minute:02d}:00" for minute in range(30, 48)],
        opens=[100.0] * 18,
        highs=[100.2] * 15 + [100.7, 100.8, 100.9],
        lows=[99.8] * 15 + [95.2, 94.8, 94.7],
        closes=[100.0] * 18,
    )
    config = CampaignConfig(or_minutes=15, entry_buffer_ticks=2)
    detected = detect_opening_range(frame, config)

    info = classify_first_breakout(detected, or_high=100.0, or_low=95.0, config=config)

    assert info["first_breakout"] == "first_breakout_upside"


def test_invalidate_on_opposite_touch_blocks_day() -> None:
    session = _policy_session_frame(
        highs=[99.9, 99.9, 100.8],
        lows=[95.1, 94.9, 95.2],
        closes=[99.8, 99.4, 100.7],
        raw_signal=[0, 0, 1],
        filter_pass=[False, False, True],
    )
    spec = OppositeBreakoutInvalidationSpec(
        name="invalidate_touch",
        description="",
        policy_family="invalidate_for_day",
        opposite_confirmation="touch",
        opposite_breakout_buffer_ticks=0,
    )
    config = CampaignConfig(symbols=("MNQ",))

    filtered, summary = apply_opposite_invalidation_filter(session, spec, config, tick_size=0.25)

    assert int(filtered["signal"].sum()) == 0
    assert bool(summary["invalidated_for_day"].iloc[0]) is True


def test_invalidate_on_opposite_close_1m_ignores_wick_but_blocks_close_break() -> None:
    session_wick = _policy_session_frame(
        highs=[99.9, 100.8],
        lows=[94.8, 95.2],
        closes=[95.1, 100.7],
        raw_signal=[0, 1],
        filter_pass=[False, True],
    )
    session_close = _policy_session_frame(
        highs=[99.9, 100.8],
        lows=[94.8, 95.2],
        closes=[94.9, 100.7],
        raw_signal=[0, 1],
        filter_pass=[False, True],
    )
    spec = OppositeBreakoutInvalidationSpec(
        name="invalidate_close_1m",
        description="",
        policy_family="invalidate_for_day",
        opposite_confirmation="close_1m",
        opposite_breakout_buffer_ticks=0,
    )
    config = CampaignConfig(symbols=("MNQ",))

    filtered_wick, summary_wick = apply_opposite_invalidation_filter(session_wick, spec, config, tick_size=0.25)
    filtered_close, summary_close = apply_opposite_invalidation_filter(session_close, spec, config, tick_size=0.25)

    assert int(filtered_wick["signal"].sum()) == 1
    assert bool(summary_wick["invalidated_for_day"].iloc[0]) is False
    assert int(filtered_close["signal"].sum()) == 0
    assert bool(summary_close["invalidated_for_day"].iloc[0]) is True


def test_invalidate_on_opposite_n_closes_respects_consecutive_count() -> None:
    session = _policy_session_frame(
        highs=[99.9, 99.9, 99.9, 100.8],
        lows=[95.0, 95.0, 95.0, 95.2],
        closes=[94.9, 95.1, 94.8, 94.7],
        raw_signal=[0, 0, 0, 1],
        filter_pass=[False, False, False, True],
    )
    spec = OppositeBreakoutInvalidationSpec(
        name="invalidate_n_closes",
        description="",
        policy_family="invalidate_for_day",
        opposite_confirmation="n_closes_1m",
        opposite_breakout_buffer_ticks=0,
        opposite_breakout_confirm_bars=2,
    )
    config = CampaignConfig(symbols=("MNQ",))

    _, summary = apply_opposite_invalidation_filter(session, spec, config, tick_size=0.25)

    assert pd.Timestamp(summary["invalidated_at"].iloc[0]) == pd.Timestamp("2024-01-02 09:48:00", tz="America/New_York")


def test_reclaim_mode_requires_or_low_reclaim_before_long() -> None:
    session = _policy_session_frame(
        highs=[99.9, 100.8, 99.7, 100.9],
        lows=[94.8, 95.0, 95.2, 95.4],
        closes=[94.9, 100.7, 95.3, 100.8],
        raw_signal=[0, 1, 0, 1],
        filter_pass=[False, True, False, True],
    )
    spec = OppositeBreakoutInvalidationSpec(
        name="reclaim_conservative",
        description="",
        policy_family="reclaim_conservative",
        opposite_confirmation="touch",
        opposite_breakout_buffer_ticks=0,
        require_reclaim_or_low_close=True,
        strategy_tag="failed_breakdown_reclaim_long",
    )
    config = CampaignConfig(symbols=("MNQ",))

    filtered, summary = apply_opposite_invalidation_filter(session, spec, config, tick_size=0.25)

    assert int(filtered.loc[filtered["timestamp"] == pd.Timestamp("2024-01-02 09:46:00", tz="America/New_York"), "signal"].iloc[0]) == 0
    assert int(filtered.loc[filtered["timestamp"] == pd.Timestamp("2024-01-02 09:48:00", tz="America/New_York"), "signal"].iloc[0]) == 1
    assert bool(summary["is_reclaim_trade"].iloc[0]) is True


def test_reclaim_mode_with_vwap_requires_close_above_vwap() -> None:
    session = _policy_session_frame(
        highs=[99.9, 99.8, 100.0, 101.0],
        lows=[94.8, 95.1, 95.2, 95.3],
        closes=[94.9, 95.2, 95.6, 100.9],
        raw_signal=[0, 0, 0, 1],
        filter_pass=[False, False, False, True],
        vwap=[94.0, 95.4, 95.5, 95.4],
    )
    spec = OppositeBreakoutInvalidationSpec(
        name="reclaim_conservative_vwap",
        description="",
        policy_family="reclaim_conservative",
        opposite_confirmation="touch",
        opposite_breakout_buffer_ticks=0,
        require_reclaim_or_low_close=True,
        require_reclaim_vwap=True,
        strategy_tag="failed_breakdown_reclaim_long",
    )
    config = CampaignConfig(symbols=("MNQ",))

    filtered, summary = apply_opposite_invalidation_filter(session, spec, config, tick_size=0.25)

    assert int(filtered["signal"].sum()) == 1
    assert pd.Timestamp(summary["reclaim_vwap_ts"].iloc[0]) == pd.Timestamp("2024-01-02 09:47:00", tz="America/New_York")
    assert int(filtered.loc[filtered["timestamp"] == pd.Timestamp("2024-01-02 09:48:00", tz="America/New_York"), "signal"].iloc[0]) == 1


def test_no_signal_is_emitted_after_definitive_invalidation() -> None:
    session = _policy_session_frame(
        highs=[99.9, 99.9, 100.9, 101.1],
        lows=[94.8, 95.0, 95.2, 95.3],
        closes=[94.9, 95.1, 100.8, 101.0],
        raw_signal=[0, 0, 1, 1],
        filter_pass=[False, False, True, True],
    )
    spec = OppositeBreakoutInvalidationSpec(
        name="invalidate_close_1m",
        description="",
        policy_family="invalidate_for_day",
        opposite_confirmation="close_1m",
        opposite_breakout_buffer_ticks=0,
    )
    config = CampaignConfig(symbols=("MNQ",))

    filtered, _ = apply_opposite_invalidation_filter(session, spec, config, tick_size=0.25)

    assert int(filtered["signal"].sum()) == 0


def test_max_configs_limits_selected_grid() -> None:
    config = CampaignConfig(symbols=("MNQ",), max_configs=3)

    selected = select_policy_grid(config)

    assert len(selected) == 3
    assert selected[0].name == "baseline_no_opposite_invalidation"


def test_config_filter_keeps_matching_configs_only() -> None:
    config = CampaignConfig(symbols=("MNQ",), config_filter="invalidate_on_opposite_close_1m")

    selected = select_policy_grid(config)

    assert selected
    assert all("invalidate_on_opposite_close_1m" in spec.name for spec in selected)


def test_fast_grid_contains_baseline_and_key_families() -> None:
    names = [spec.name for spec in build_fast_policy_grid()]

    assert "baseline_no_opposite_invalidation" in names
    assert any("invalidate_on_opposite_touch" in name for name in names)
    assert any("invalidate_on_opposite_close_1m" in name for name in names)
    assert any("invalidate_on_opposite_n_closes_1m" in name for name in names)
    assert any("invalidate_on_opposite_close_5m" in name for name in names)
    assert any("allow_reclaim_after_opposite_breakout_conservative" in name for name in names)


def test_cache_frame_write_then_read(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    parquet_path = tmp_path / "cache.parquet"
    csv_path = tmp_path / "cache.csv"

    written_path = _write_dataframe_with_fallback(frame, parquet_path, csv_path)
    restored = _read_dataframe_with_fallback(parquet_path, csv_path)

    assert written_path.exists()
    pd.testing.assert_frame_equal(restored, frame)


def test_smoke_campaign_produces_expected_files(tmp_path: Path) -> None:
    dataset_path = tmp_path / "MNQ_c_0_1m_smoke.parquet"
    _smoke_dataset(dataset_path)
    output_root = tmp_path / "export"
    config = CampaignConfig(
        symbols=("MNQ",),
        dataset_paths={"MNQ": dataset_path},
        output_root=output_root,
        cache_root=tmp_path / "cache",
        smoke=True,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    result = run_campaign(config)
    export_dir = result["output_dir"]

    expected = [
        "config_grid.csv",
        "results_by_config.csv",
        "results_by_symbol.csv",
        "results_by_year.csv",
        "baseline_decomposition.csv",
        "opposite_breakout_trade_decomposition.csv",
        "daily_returns_by_config.csv",
        "trades_by_config.csv",
        "invalidated_days.csv",
        "ranking_robust.csv",
        "best_config_summary.json",
        "run_metadata.json",
        "final_report.md",
    ]

    for filename in expected:
        assert (export_dir / filename).exists(), filename


def test_resume_skips_completed_configs_and_reuses_export_dir(tmp_path: Path) -> None:
    dataset_path = tmp_path / "MNQ_c_0_1m_resume.parquet"
    _smoke_dataset(dataset_path, sessions=10)
    output_root = tmp_path / "export"
    cache_root = tmp_path / "cache"

    first = run_campaign(
        CampaignConfig(
            symbols=("MNQ",),
            dataset_paths={"MNQ": dataset_path},
            output_root=output_root,
            cache_root=cache_root,
            smoke=True,
            max_configs=2,
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
    )
    second = run_campaign(
        CampaignConfig(
            symbols=("MNQ",),
            dataset_paths={"MNQ": dataset_path},
            output_root=output_root,
            cache_root=cache_root,
            smoke=True,
            max_configs=2,
            resume=True,
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
    )

    assert first["output_dir"] == second["output_dir"]
    checkpoint = pd.read_csv(first["output_dir"] / "checkpoint_results_by_symbol.csv")
    assert len(checkpoint) == 2


def test_partial_report_mentions_run_status_and_checkpoints(tmp_path: Path) -> None:
    dataset_path = tmp_path / "MNQ_c_0_1m_partial.parquet"
    _smoke_dataset(dataset_path, sessions=10)
    result = run_campaign(
        CampaignConfig(
            symbols=("MNQ",),
            dataset_paths={"MNQ": dataset_path},
            output_root=tmp_path / "export",
            cache_root=tmp_path / "cache",
            smoke=False,
            max_configs=1,
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
    )

    report = (result["output_dir"] / "final_report.md").read_text(encoding="utf-8")
    assert "Run status: `partial`" in report
    assert "Checkpoint results by symbol" in report
