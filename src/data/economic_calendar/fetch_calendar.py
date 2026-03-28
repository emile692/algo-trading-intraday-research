"""Raw economic calendar ingestion helpers.

This module keeps the research pipeline reproducible by supporting two paths:

1. Optional structured JSON fetch from an external API.
2. Mandatory local CSV fallback stored inside the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.config.paths import DATA_DIR

DEFAULT_RAW_CALENDAR_PATH = DATA_DIR / "raw" / "economic_calendar_us.csv"

RAW_REQUIRED_COLUMNS = ("event_name", "country", "date", "time")
RAW_OPTIONAL_COLUMNS = (
    "impact",
    "actual",
    "forecast",
    "previous",
    "source_timezone",
    "currency",
    "timestamp",
    "event_timestamp",
)


def _to_snake_case(value: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_").lower()


def normalize_raw_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized lowercase snake-case columns."""
    out = df.copy()
    out.columns = [_to_snake_case(column) for column in out.columns]
    rename_map = {
        "event": "event_name",
        "name": "event_name",
        "event_title": "event_name",
        "calendar_event": "event_name",
        "calendar_date": "date",
        "event_date": "date",
        "calendar_time": "time",
        "event_time": "time",
        "priority": "impact",
    }
    out = out.rename(columns={key: value for key, value in rename_map.items() if key in out.columns})
    return out


def coerce_raw_calendar_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a predictable raw schema for downstream cleaning."""
    out = normalize_raw_calendar_columns(df)
    for column in RAW_REQUIRED_COLUMNS + RAW_OPTIONAL_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA

    required_minimum = {"event_name", "country"}
    missing = [column for column in required_minimum if column not in out.columns]
    if missing:
        raise ValueError(f"Missing required raw calendar columns: {missing}")

    if out["date"].isna().all() and out["timestamp"].isna().all() and out["event_timestamp"].isna().all():
        raise ValueError(
            "Raw calendar data must provide either a 'date' column or a timestamp column."
        )

    ordered_columns = list(RAW_REQUIRED_COLUMNS + RAW_OPTIONAL_COLUMNS)
    extras = [column for column in out.columns if column not in ordered_columns]
    return out[ordered_columns + extras]


def load_raw_calendar_csv(path: Path | str = DEFAULT_RAW_CALENDAR_PATH) -> pd.DataFrame:
    """Load a raw calendar CSV from disk and normalize its schema."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw economic calendar CSV not found: {csv_path}")
    return coerce_raw_calendar_schema(pd.read_csv(csv_path))


def write_raw_calendar_csv(df: pd.DataFrame, path: Path | str = DEFAULT_RAW_CALENDAR_PATH) -> Path:
    """Write a normalized raw calendar CSV to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coerce_raw_calendar_schema(df).to_csv(output_path, index=False)
    return output_path


def _extract_records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]

    if isinstance(payload, dict):
        for key in ("data", "results", "events", "calendar"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return [record for record in candidate if isinstance(record, dict)]

    raise ValueError("Unsupported economic calendar API payload shape.")


def fetch_calendar_from_api(
    api_url: str,
    api_key: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    timeout_seconds: int = 30,
    extra_params: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Fetch a structured economic calendar JSON payload from an API endpoint."""
    params: dict[str, str] = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    if extra_params:
        params.update({key: value for key, value in extra_params.items() if value is not None})

    query_string = urllib.parse.urlencode(params)
    request_url = api_url if not query_string else f"{api_url}?{query_string}"

    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(request_url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))

    records = _extract_records_from_payload(payload)
    if not records:
        return coerce_raw_calendar_schema(pd.DataFrame(columns=RAW_REQUIRED_COLUMNS))
    return coerce_raw_calendar_schema(pd.DataFrame.from_records(records))


def get_calendar_source_dataframe(
    raw_csv_path: Path | str = DEFAULT_RAW_CALENDAR_PATH,
    api_url: str | None = None,
    api_key: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    """Return raw calendar data from API when available, else from the local CSV fallback."""
    csv_path = Path(raw_csv_path)
    if api_url:
        try:
            return fetch_calendar_from_api(
                api_url=api_url,
                api_key=api_key,
                start_date=start_date,
                end_date=end_date,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            if csv_path.exists():
                warnings.warn(
                    f"API fetch failed, falling back to local calendar CSV at {csv_path}: {exc}",
                    stacklevel=2,
                )
            else:
                raise

    return load_raw_calendar_csv(csv_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch or validate raw economic calendar data.")
    parser.add_argument("--raw-path", default=str(DEFAULT_RAW_CALENDAR_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_RAW_CALENDAR_PATH))
    parser.add_argument("--api-url", default=os.getenv("ECON_CAL_API_URL"))
    parser.add_argument("--api-key-env-var", default="ECONOMIC_CALENDAR_API_KEY")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env_var) if args.api_url else None
    raw_df = get_calendar_source_dataframe(
        raw_csv_path=args.raw_path,
        api_url=args.api_url,
        api_key=api_key,
        start_date=args.start_date,
        end_date=args.end_date,
        timeout_seconds=args.timeout_seconds,
    )
    output_path = write_raw_calendar_csv(raw_df, args.output_path)
    print(output_path)


if __name__ == "__main__":
    main()
