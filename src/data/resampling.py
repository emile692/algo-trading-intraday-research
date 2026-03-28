"""Resampling helpers for processed OHLCV datasets."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _normalize_timeframe_tag(rule: str) -> str:
    """Convert a pandas resample rule to a compact filename tag."""
    clean = str(rule).strip().lower().replace(" ", "")
    match = re.fullmatch(r"(\d+)(min|m|t)", clean)
    if match:
        return f"{match.group(1)}m"
    match = re.fullmatch(r"(\d+)(h|hour|hours)", clean)
    if match:
        return f"{match.group(1)}h"
    match = re.fullmatch(r"(\d+)(d|day|days)", clean)
    if match:
        return f"{match.group(1)}d"
    return re.sub(r"[^a-z0-9]+", "", clean)


def _ensure_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy indexed by timestamp."""
    if "timestamp" in df.columns:
        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        if out["timestamp"].isna().any():
            raise ValueError("Found unparsable timestamp values.")
        return out.set_index("timestamp").sort_index()

    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out.index = pd.to_datetime(out.index, errors="coerce")
        if out.index.isna().any():
            raise ValueError("Found unparsable timestamp index values.")
        out.index.name = "timestamp"
        return out.sort_index()

    raise ValueError("Expected a 'timestamp' column or a DatetimeIndex.")


def _build_aggregation_map(df: pd.DataFrame) -> dict[str, str]:
    """Choose sensible aggregations for OHLCV + processed metadata columns."""
    agg: dict[str, str] = {}
    for column in df.columns:
        lower = str(column).strip().lower()
        if lower == "open":
            agg[column] = "first"
        elif lower == "high":
            agg[column] = "max"
        elif lower == "low":
            agg[column] = "min"
        elif lower == "close":
            agg[column] = "last"
        elif lower == "volume":
            agg[column] = "sum"
        elif lower in {"open interest", "open_interest", "oi"}:
            agg[column] = "last"
        else:
            agg[column] = "last"
    return agg


def resample_ohlcv(
    df: pd.DataFrame,
    rule: str,
    label: str = "left",
    closed: str = "left",
    aggregation_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Resample OHLCV bars with proper aggregation and metadata preservation."""
    out = _ensure_timestamp_index(df)
    agg = _build_aggregation_map(out)
    if aggregation_overrides:
        for column, method in aggregation_overrides.items():
            if column in out.columns:
                agg[column] = str(method)
    result = out.resample(rule, label=label, closed=closed).agg(agg)
    required = [col for col in ["open", "high", "low", "close"] if col in result.columns]
    if required:
        result = result.dropna(subset=required)
    return result.reset_index()


def build_resampled_output_path(
    input_path: Path | str,
    rule: str,
    output_dir: Path | str | None = None,
) -> Path:
    """Build an output filename matching the processed dataset naming convention."""
    src = Path(input_path)
    tag = _normalize_timeframe_tag(rule)
    parts = src.stem.split("_")
    if len(parts) >= 6:
        parts[3] = tag
        name = "_".join(parts) + src.suffix
    else:
        name = f"{src.stem}_{tag}{src.suffix}"
    root = Path(output_dir) if output_dir is not None else src.parent
    return root / name


def resample_parquet_dataset(
    input_path: Path | str,
    rule: str,
    output_path: Path | str | None = None,
) -> Path:
    """Load a parquet dataset, resample it, and save it back to parquet."""
    src = Path(input_path)
    dst = Path(output_path) if output_path is not None else build_resampled_output_path(src, rule)
    dst.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(src)
    resampled = resample_ohlcv(raw, rule=rule)
    resampled = resampled.set_index("timestamp")
    resampled.index.name = "timestamp"
    resampled.to_parquet(dst)
    return dst


def batch_resample_parquet_datasets(
    input_paths: list[Path | str],
    rule: str,
    output_dir: Path | str | None = None,
) -> list[Path]:
    """Resample a list of parquet datasets and return their output paths."""
    outputs: list[Path] = []
    for path in input_paths:
        dst = build_resampled_output_path(path, rule=rule, output_dir=output_dir)
        outputs.append(resample_parquet_dataset(path, rule=rule, output_path=dst))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample processed parquet OHLCV datasets.")
    parser.add_argument("paths", nargs="+", help="One or more parquet file paths to resample.")
    parser.add_argument("--rule", required=True, help="Pandas resample rule, for example 5min or 15min.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory.")
    args = parser.parse_args()

    outputs = batch_resample_parquet_datasets(
        input_paths=[Path(path) for path in args.paths],
        rule=str(args.rule),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
