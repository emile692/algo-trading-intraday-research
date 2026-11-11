from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.treasury_auction_final import _add_returns, _control_events, _placebo_events
from src.data.process_treasury_continuous import process_continuous_parquet


def test_combined_price_effect_is_short_pre_plus_long_post() -> None:
    matrix = pd.DataFrame(
        [{-180: 0.0, -1: -2.0, 5: -1.5, 179: 1.5}]
    )
    details = pd.DataFrame([{"event_ts_utc": pd.Timestamp("2024-01-01 18:00", tz="UTC")}])
    result = _add_returns(matrix, details)
    assert result.loc[0, "pre_return_bps"] == -2.0
    assert result.loc[0, "post_return_bps"] == 3.0
    assert result.loc[0, "combined_effect_bps"] == 5.0
    assert bool(result.loc[0, "canonical_hit"])


def test_placebo_time_is_localized_in_new_york() -> None:
    events = pd.DataFrame(
        {
            "auction_date": [pd.Timestamp("2024-03-11")],
            "bid_deadline_ts_utc": [pd.Timestamp("2024-03-11 17:00", tz="UTC")],
        }
    )
    placebo = _placebo_events(events, "11:00", "test")
    assert placebo.loc[0, "event_ts_utc"] == pd.Timestamp("2024-03-11 15:00", tz="UTC")


def test_controls_do_not_land_on_another_coupon_auction() -> None:
    calendar = pd.DataFrame(
        {
            "auction_date": pd.to_datetime(["2024-01-03", "2024-01-10"]),
        }
    )
    events = pd.DataFrame(
        {
            "auction_date": pd.to_datetime(["2024-01-10"]),
            "bid_deadline_ts_utc": [pd.Timestamp("2024-01-10 18:00", tz="UTC")],
            "tenor": ["5-Year"],
            "instrument": ["ZF"],
            "mapping": ["5-Year->ZF"],
        }
    )
    controls = _control_events(calendar, events)
    assert pd.Timestamp(controls.loc[0, "control_date"]) == pd.Timestamp("2024-01-02")


def test_zf_processor_normalizes_index_and_writes_audit() -> None:
    index = pd.date_range("2024-01-02 13:20", periods=3, freq="min", tz="UTC", name="ts_event")
    frame = pd.DataFrame(
        {
            "rtype": np.array([33, 33, 33], dtype="uint8"),
            "publisher_id": np.array([1, 1, 1], dtype="uint16"),
            "instrument_id": np.array([1, 1, 2], dtype="uint32"),
            "open": [108.0, 108.01, 108.02],
            "high": [108.02, 108.03, 108.04],
            "low": [107.99, 108.0, 108.01],
            "close": [108.01, 108.02, 108.03],
            "volume": np.array([10, 20, 30], dtype="uint64"),
            "symbol": ["ZF.v.0"] * 3,
        },
        index=index,
    )
    root = Path("data/exports")
    root.mkdir(parents=True, exist_ok=True)
    source = root / "_test_zf_source.parquet"
    output = root / "_test_zf_output.parquet"
    coverage = root / "_test_zf_coverage.csv"
    metadata_path = root / "_test_zf_metadata.json"
    roll_path = root / "_test_zf_roll_transitions.csv"
    try:
        frame.to_parquet(source)
        metadata = process_continuous_parquet(
            source,
            output,
            coverage,
            metadata_path,
            "ZF",
        )
        processed = pd.read_parquet(output)
        assert processed.index.name == "timestamp"
        assert metadata["duplicate_timestamps"] == 0
        assert metadata["instrument_transitions"] == 1
    finally:
        for path in (source, output, coverage, metadata_path, roll_path):
            path.unlink(missing_ok=True)
