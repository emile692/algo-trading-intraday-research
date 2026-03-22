import pathlib
import re
from argparse import ArgumentParser

import databento as db
import pandas as pd

api_key = 'db-wwB7CVfaGbRCgCmJULX5psuUH5Ajj'


def download_data(ticker: str, start: str, end: str, schema: str = "ohlcv-1m") -> pd.DataFrame:
    """Download historical OHLCV data for a given ticker and date range."""
    client = db.Historical(api_key)
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",      # CME Globex MDP 3.0
        symbols=ticker,        # MNQ continuous front month
        stype_in="continuous",
        schema=schema,
        start=start,
        end=end,
    )

    return data.to_df()


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
    folder: str = "data/dowloaded",
    ticker: str | None = None,
    timeframe: str | None = None,
    schema: str = "ohlcv-1m",
) -> str:
    """Save DataFrame to parquet in data/dowloaded.

    filename defaults to <ticker>_<timeframe>_<UTC timestamp>.parquet if not provided.
    """
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


if __name__ == "__main__":
    parser = ArgumentParser(description="Download historical Databento data and store as parquet.")
    parser.add_argument("--ticker", default="MNQ.c.0", help="Continuous symbol, e.g. MNQ.c.0 or NQ.c.0")
    parser.add_argument("--start", default="2019-01-01T00:00:00Z", help="Start ISO datetime (UTC)")
    parser.add_argument("--end", default=None, help="End ISO datetime (UTC), defaults to now")
    parser.add_argument("--schema", default="ohlcv-1m", help="Databento schema (default: ohlcv-1m)")
    parser.add_argument(
        "--timeframe",
        default="1m",
        help="Logical timeframe label to include in output filename (default: 1m)",
    )
    parser.add_argument("--output-dir", default="../../data/downloaded", help="Output folder for downloaded parquet")
    parser.add_argument("--filename", default=None, help="Optional explicit output filename")
    args = parser.parse_args()

    end_date = args.end or pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    df = download_data(args.ticker, args.start, end_date, args.schema)
    print(df.head())

    saved_path = save_to_parquet(
        df,
        filename=args.filename,
        folder=args.output_dir,
        ticker=args.ticker,
        timeframe=args.timeframe,
        schema=args.schema,
    )
    print(f"Saved parquet file to {saved_path}")
