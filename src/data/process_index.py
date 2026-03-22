from pathlib import Path

import pandas as pd


def process_index_rename_to_timestamp(df: pd.DataFrame, tz_from: str = "UTC", tz_to: str = "America/New_York") -> pd.DataFrame:
    """Process index: make DatetimeIndex, set name 'timestamp', localize/convert timezone to NY."""

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="raise")

    if df.index.tz is None:
        df.index = df.index.tz_localize(tz_from)

    df.index = df.index.tz_convert(tz_to)
    df.index.name = "timestamp"

    return df


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    return str(p)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process index and convert timezone to America/New_York")
    parser.add_argument("input", help="Input parquet file path")
    parser.add_argument("output", help="Output parquet file path")
    parser.add_argument("--from-tz", default="UTC", help="Source timezone for naive index (default UTC)")
    parser.add_argument("--to-tz", default="America/New_York", help="Target timezone (default America/New_York)")

    args = parser.parse_args()

    df = load_parquet(args.input)
    df = process_index_rename_to_timestamp(df, tz_from=args.from_tz, tz_to=args.to_tz)
    saved = save_parquet(df, args.output)
    print(f"Saved processed parquet to: {saved}")
