"""Normalize and audit a Databento Treasury continuous futures parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_continuous_parquet(
    source: Path,
    output: Path,
    coverage_csv: Path,
    metadata_json: Path,
    symbol: str,
) -> dict:
    frame = pd.read_parquet(source)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex from Databento")
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame.index.name = "timestamp"
    frame = frame.sort_index()

    required = ["open", "high", "low", "close", "volume", "instrument_id"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    duplicate_timestamps = int(frame.index.duplicated().sum())
    invalid_ohlc = int(
        (
            (frame["high"] < frame["low"])
            | (frame["open"] > frame["high"])
            | (frame["open"] < frame["low"])
            | (frame["close"] > frame["high"])
            | (frame["close"] < frame["low"])
        ).sum()
    )
    if duplicate_timestamps or invalid_ohlc or frame[required].isna().any().any():
        raise ValueError(
            f"Quality gate failed: duplicates={duplicate_timestamps}, "
            f"invalid_ohlc={invalid_ohlc}, missing={int(frame[required].isna().sum().sum())}"
        )

    local_index = frame.index.tz_convert("America/New_York")
    rth_mask = (local_index.time >= pd.Timestamp("08:20").time()) & (
        local_index.time < pd.Timestamp("15:00").time()
    )
    counts = (
        pd.Series(1, index=local_index[rth_mask])
        .groupby(local_index[rth_mask].date)
        .sum()
        .rename("bar_count")
        .to_frame()
    )
    counts.index.name = "session_date"
    counts["coverage_fraction"] = counts["bar_count"] / 400.0
    counts["below_50pct"] = counts["coverage_fraction"] < 0.50
    counts["below_80pct"] = counts["coverage_fraction"] < 0.80

    transitions = frame["instrument_id"].ne(frame["instrument_id"].shift())
    transitions.iloc[0] = False
    close_return = frame["close"].pct_change()
    roll_rows = pd.DataFrame(
        {
            "timestamp": frame.index[transitions],
            "previous_instrument_id": frame["instrument_id"].shift().loc[transitions].to_numpy(),
            "new_instrument_id": frame.loc[transitions, "instrument_id"].to_numpy(),
            "previous_close": frame["close"].shift().loc[transitions].to_numpy(),
            "new_close": frame.loc[transitions, "close"].to_numpy(),
            "close_return": close_return.loc[transitions].to_numpy(),
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output)
    counts.reset_index().to_csv(coverage_csv, index=False)
    roll_path = metadata_json.with_name(metadata_json.stem.replace("_metadata", "_roll_transitions") + ".csv")
    roll_rows.to_csv(roll_path, index=False)

    metadata = {
        "symbol": symbol,
        "source_path": str(source),
        "processed_path": str(output),
        "source_sha256": _sha256(source),
        "processed_sha256": _sha256(output),
        "row_count": len(frame),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "timezone": "UTC",
        "bar_resolution": "1 minute",
        "bar_timestamp_semantics": "bar start",
        "continuous_contract_methodology": "Databento volume-ranked front month (v.0)",
        "back_adjustment": False,
        "duplicate_timestamps": duplicate_timestamps,
        "invalid_ohlc_rows": invalid_ohlc,
        "missing_required_values": int(frame[required].isna().sum().sum()),
        "zero_volume_rows": int(frame["volume"].eq(0).sum()),
        "rth_session_timezone": "America/New_York",
        "rth_session": "08:20 inclusive to 15:00 exclusive",
        "rth_session_count": len(counts),
        "median_rth_bars": float(counts["bar_count"].median()),
        "mean_rth_bars": float(counts["bar_count"].mean()),
        "sessions_below_50pct": int(counts["below_50pct"].sum()),
        "sessions_below_80pct": int(counts["below_80pct"].sum()),
        "instrument_count": int(frame["instrument_id"].nunique()),
        "instrument_transitions": int(transitions.sum()),
        "roll_abs_return_median": float(roll_rows["close_return"].abs().median()),
        "roll_abs_return_p95": float(roll_rows["close_return"].abs().quantile(0.95)),
        "roll_abs_return_max": float(roll_rows["close_return"].abs().max()),
        "roll_transition_csv": str(roll_path),
    }
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()
    metadata = process_continuous_parquet(
        args.source, args.output, args.coverage_csv, args.metadata_json, args.symbol
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
