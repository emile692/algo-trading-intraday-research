import json
import os
import pathlib
import re
from argparse import ArgumentParser
from datetime import time

import databento as db
import pandas as pd


DATASET = "GLBX.MDP3"
API_KEY_ENV = "DATABENTO_API_KEY"
DEFAULT_OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "downloaded"


def _historical_client() -> db.Historical:
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set {API_KEY_ENV} before calling Databento")
    return db.Historical(api_key)


def download_data(
    ticker: str,
    start: str,
    end: str,
    schema: str = "ohlcv-1m",
    dataset: str = DATASET,
    stype_in: str = "continuous",
) -> pd.DataFrame:
    """Download historical OHLCV data for a given ticker and date range."""
    client = _historical_client()
    data = client.timeseries.get_range(
        dataset=dataset,
        symbols=ticker,
        stype_in=stype_in,
        schema=schema,
        start=start,
        end=end,
    )
    return data.to_df()


def estimate_cost(
    ticker: str,
    start: str,
    end: str,
    schema: str = "ohlcv-1m",
    dataset: str = DATASET,
    stype_in: str = "continuous",
) -> float:
    """Return Databento's pre-request USD cost estimate."""
    client = _historical_client()
    return float(
        client.metadata.get_cost(
            dataset=dataset,
            symbols=ticker,
            stype_in=stype_in,
            schema=schema,
            start=start,
            end=end,
        )
    )


def _sanitize_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned or "unknown"


def _default_timeframe_from_schema(schema: str) -> str:
    if schema in {"ohlcv1", "ohlcv-1m"}:
        return "1m"
    if schema == "ohlcv-1s":
        return "1s"
    if schema == "ohlcv-1h":
        return "1h"
    if schema == "ohlcv-1d":
        return "1d"
    return schema


def save_to_parquet(
    df: pd.DataFrame,
    filename: str | None = None,
    folder: str | pathlib.Path = DEFAULT_OUTPUT_DIR,
    ticker: str | None = None,
    timeframe: str | None = None,
    schema: str = "ohlcv-1m",
) -> str:
    """Save a DataFrame to the requested output folder."""
    target = pathlib.Path(folder)
    target.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ticker_part = _sanitize_filename_part(ticker or "unknown_ticker")
        timeframe_part = _sanitize_filename_part(timeframe or _default_timeframe_from_schema(schema))
        ts_part = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticker_part}_{timeframe_part}_{ts_part}.parquet"

    filepath = target / filename
    df.to_parquet(filepath)
    return str(filepath)


def coverage_diagnostics(
    df: pd.DataFrame,
    timezone: str = "America/New_York",
    session_start: str = "08:20",
    session_end: str = "15:00",
) -> dict[str, float | int | str]:
    """Measure minute-bar density during the requested local research session."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Databento frame must have a DatetimeIndex")
    timestamps = df.index
    timestamps = timestamps.tz_localize("UTC") if timestamps.tz is None else timestamps.tz_convert("UTC")
    local = timestamps.tz_convert(timezone)
    start_time = time.fromisoformat(session_start)
    end_time = time.fromisoformat(session_end)
    in_session = (local.time >= start_time) & (local.time < end_time)
    counts = pd.Series(1, index=local[in_session]).groupby(local[in_session].date).sum()
    expected = int((pd.Timedelta(session_end + ":00") - pd.Timedelta(session_start + ":00")).total_seconds() / 60)
    threshold = int(expected * 0.8)
    return {
        "timezone": timezone,
        "session_start": session_start,
        "session_end": session_end,
        "expected_minutes_per_full_session": expected,
        "observed_session_count": int(len(counts)),
        "minimum_bars": int(counts.min()) if len(counts) else 0,
        "median_bars": float(counts.median()) if len(counts) else 0.0,
        "mean_bars": float(counts.mean()) if len(counts) else 0.0,
        "sessions_below_80pct": int(counts.lt(threshold).sum()),
        "sessions_below_80pct_pct": float(100.0 * counts.lt(threshold).mean()) if len(counts) else 100.0,
    }


if __name__ == "__main__":
    parser = ArgumentParser(description="Download historical Databento data and store as parquet.")
    parser.add_argument(
        "--ticker",
        default="ZN.v.0",
        help="Continuous symbol. Prefer the volume-ranked front month, e.g. ZN.v.0.",
    )
    parser.add_argument("--start", default="2019-01-01T00:00:00Z", help="Start ISO datetime (UTC)")
    parser.add_argument("--end", default=None, help="End ISO datetime (UTC), defaults to now")
    parser.add_argument("--schema", default="ohlcv-1m", help="Databento schema (default: ohlcv-1m)")
    parser.add_argument("--dataset", default=DATASET, help=f"Databento dataset (default: {DATASET})")
    parser.add_argument("--stype-in", default="continuous", help="Input symbology type")
    parser.add_argument("--estimate-only", action="store_true", help="Print the estimated USD cost and do not download")
    parser.add_argument(
        "--timeframe",
        default="1m",
        help="Logical timeframe label to include in output filename (default: 1m)",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output folder for downloaded parquet")
    parser.add_argument("--filename", default=None, help="Optional explicit output filename")
    parser.add_argument("--qa-timezone", default="America/New_York")
    parser.add_argument("--qa-session-start", default="08:20", help="Local session start for coverage QA")
    parser.add_argument("--qa-session-end", default="15:00", help="Local session end for coverage QA")
    args = parser.parse_args()

    end_date = args.end or pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    cost = estimate_cost(args.ticker, args.start, end_date, args.schema, args.dataset, args.stype_in)
    print(f"Estimated Databento cost: ${cost:.6f}")
    if args.estimate_only:
        raise SystemExit(0)

    df = download_data(args.ticker, args.start, end_date, args.schema, args.dataset, args.stype_in)
    saved_path = save_to_parquet(
        df,
        filename=args.filename,
        folder=args.output_dir,
        ticker=args.ticker,
        timeframe=args.timeframe,
        schema=args.schema,
    )
    print(f"Saved parquet file to {saved_path}")
    diagnostics = coverage_diagnostics(df, args.qa_timezone, args.qa_session_start, args.qa_session_end)
    qa_path = pathlib.Path(saved_path).with_suffix(".coverage.json")
    qa_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))
    print(f"Saved coverage diagnostics to {qa_path}")
