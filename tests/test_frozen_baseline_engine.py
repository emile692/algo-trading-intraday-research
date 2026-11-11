from __future__ import annotations

from datetime import date, time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from research.baseline_backtests.baseline_engine import (
    _intrabar_exit,
    _build_gap_continuation,
    _build_gap_fade,
    _round_tick,
    build_cme_equity_early_close_calendar,
    load_frozen_specs,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_registry_hashes_and_cardinality() -> None:
    frozen = ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
    specs, benchmark, manifest = load_frozen_specs(
        frozen,
        ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv",
    )
    assert len(specs) == 16
    assert len(manifest) == 17
    assert benchmark["mechanism_id"] == "MNQ_BUY_AND_HOLD"
    assert all(len(spec["_frozen_spec_sha256"]) == 64 for spec in specs)


def test_same_minute_stop_precedes_target_for_both_directions() -> None:
    long_position = {"direction_int": 1, "stop_price": 90.0, "target_price": 110.0}
    short_position = {"direction_int": -1, "stop_price": 110.0, "target_price": 90.0}
    assert _intrabar_exit(long_position, 100.0, 111.0, 89.0) == (90.0, "stop")
    assert _intrabar_exit(short_position, 100.0, 111.0, 89.0) == (110.0, "stop")


def test_adverse_tick_rounding_never_improves_fill() -> None:
    assert _round_tick(100.11, 0.25, 1, True) == 100.25
    assert _round_tick(100.11, 0.25, -1, True) == 100.0
    assert _round_tick(100.11, 0.25, 1, False) == 100.0
    assert _round_tick(100.11, 0.25, -1, False) == 100.25


def test_cme_early_close_rules_distinguish_1300_and_1315() -> None:
    frame = build_cme_equity_early_close_calendar(date(2019, 1, 1), date(2019, 12, 31)).set_index("session_date")
    assert frame.loc[date(2019, 7, 3), "early_close_time"] == "13:15:00"
    assert frame.loc[date(2019, 9, 2), "early_close_time"] == "13:00:00"
    assert frame.loc[date(2019, 11, 29), "early_close_time"] == "13:15:00"


def test_gap_builders_read_adapted_session_clocks() -> None:
    tz = "America/New_York"
    prior_day = date(2024, 1, 2)
    day = date(2024, 1, 3)
    bars = pd.DataFrame(
        [
            {"session_date": prior_day, "start_time": pd.Timestamp("2024-01-02 08:20", tz=tz), "end_time": pd.Timestamp("2024-01-02 08:35", tz=tz), "open": 99.0, "close": 99.5, "volume": 50.0},
            {"session_date": day, "start_time": pd.Timestamp("2024-01-03 08:20", tz=tz), "end_time": pd.Timestamp("2024-01-03 08:35", tz=tz), "open": 101.0, "close": 102.0, "volume": 100.0},
        ]
    )
    daily = pd.DataFrame(
        [
            {"session_date": prior_day, "open": 99.0, "prior_close": float("nan"), "prior_atr14": float("nan"), "true_range": 1.0},
            {"session_date": day, "open": 101.0, "prior_close": 100.0, "prior_atr14": 1.0, "true_range": 1.0},
        ]
    )
    continuation_spec = {
        "canonical_parameters": {
            "structural": {"minimum_gap_atr_multiple": 0.0, "volume_average_lookback": 1, "volume_threshold_multiple": 0.0},
            "execution": {"rth_start": "08:20:00"},
        }
    }
    continuation = _build_gap_continuation(continuation_spec, SimpleNamespace(bars={15: bars}, daily=daily))
    assert len(continuation.entries) == 1
    assert continuation.entries[0].activation_time.time() == time(8, 35)

    fade_spec = {
        "canonical_parameters": {
            "structural": {"daily_atr_period": 1, "maximum_gap_atr_multiple": 3.0, "minimum_gap_ticks": 0},
            "risk_management": {"midday_time_exit": "10:50:00"},
            "execution": {"tick_size": 0.1, "last_entry_time": "08:21:00"},
        }
    }
    fade = _build_gap_fade(fade_spec, SimpleNamespace(daily=daily))
    assert len(fade.entries) == 1
    assert fade.entries[0].activation_time.time() == time(8, 21)
    assert fade.time_exits[day][0] == time(10, 50)
