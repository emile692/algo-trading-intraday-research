"""Ingest the official U.S. Treasury auction calendar without inventing times."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


API_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/"
    "fiscal_service/v1/accounting/od/auctions_query"
)
TENORS = ("2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "30-Year")
FIELDS = (
    "record_date",
    "cusip",
    "security_type",
    "security_term",
    "original_security_term",
    "auction_date",
    "announcemt_date",
    "issue_date",
    "closing_time_comp",
    "closing_time_noncomp",
    "reopening",
    "inflation_index_security",
    "pdf_filenm_announcemt",
    "pdf_filenm_comp_results",
)


def build_api_url(start: str, end: str, page_size: int = 10_000) -> str:
    """Build a reproducible official-API request for coupon securities."""
    params = {
        "fields": ",".join(FIELDS),
        "filter": (
            f"auction_date:gte:{start},auction_date:lte:{end},"
            "security_type:in:(Note,Bond)"
        ),
        "sort": "auction_date,original_security_term",
        "page[number]": "1",
        "page[size]": str(page_size),
    }
    return f"{API_URL}?{urllib.parse.urlencode(params, safe=':,()')}"


def fetch_official_auction_records(start: str, end: str) -> tuple[list[dict], dict, str]:
    """Fetch auction rows from FiscalData; no API key is required."""
    url = build_api_url(start, end)
    request = urllib.request.Request(url, headers={"User-Agent": "intraday-research/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS host
        payload = json.load(response)
    return payload.get("data", []), payload.get("meta", {}), url


def normalize_auction_records(records: list[dict]) -> pd.DataFrame:
    """Keep nominal coupon tenors and preserve unavailable result times as null."""
    frame = pd.DataFrame.from_records(records)
    output_columns = [
        "auction_date",
        "tenor",
        "security_type",
        "security_term",
        "cusip",
        "reopening",
        "announcement_date",
        "issue_date",
        "bid_deadline_local",
        "bid_deadline_timezone",
        "bid_deadline_ts_utc",
        "result_time_local",
        "result_time_timezone",
        "result_ts_utc",
        "result_time_status",
        "announcement_pdf",
        "results_pdf",
        "source",
    ]
    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    required = {
        "auction_date",
        "original_security_term",
        "inflation_index_security",
        "closing_time_comp",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"FiscalData response missing required fields: {sorted(missing)}")

    frame = frame.loc[
        frame["original_security_term"].isin(TENORS)
        & frame["inflation_index_security"].eq("No")
    ].copy()
    frame["auction_date"] = pd.to_datetime(frame["auction_date"], errors="raise")
    frame["announcement_date"] = pd.to_datetime(frame["announcemt_date"], errors="coerce")
    frame["issue_date"] = pd.to_datetime(frame["issue_date"], errors="coerce")

    local_naive = pd.to_datetime(
        frame["auction_date"].dt.strftime("%Y-%m-%d")
        + " "
        + frame["closing_time_comp"].astype(str),
        format="%Y-%m-%d %I:%M %p",
        errors="coerce",
    )
    local_ts = local_naive.dt.tz_localize(
        "America/New_York", ambiguous="raise", nonexistent="raise"
    )

    out = pd.DataFrame(
        {
            "auction_date": frame["auction_date"].dt.date,
            "tenor": frame["original_security_term"],
            "security_type": frame.get("security_type"),
            "security_term": frame.get("security_term"),
            "cusip": frame.get("cusip"),
            "reopening": frame.get("reopening"),
            "announcement_date": frame["announcement_date"].dt.date,
            "issue_date": frame["issue_date"].dt.date,
            "bid_deadline_local": frame["closing_time_comp"],
            "bid_deadline_timezone": "America/New_York",
            "bid_deadline_ts_utc": local_ts.dt.tz_convert("UTC"),
            # FiscalData does not expose the historical publication timestamp.
            "result_time_local": pd.NA,
            "result_time_timezone": pd.NA,
            "result_ts_utc": pd.NaT,
            "result_time_status": "not_available_from_fiscaldata_api",
            "announcement_pdf": frame.get("pdf_filenm_announcemt"),
            "results_pdf": frame.get("pdf_filenm_comp_results"),
            "source": "U.S. Treasury FiscalData auctions_query API",
        }
    )
    return out.sort_values(["auction_date", "tenor", "cusip"]).reset_index(drop=True)[
        output_columns
    ]


def ingest_auction_calendar(
    output_path: Path,
    metadata_path: Path,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch, normalize, and persist the canonical auction calendar and provenance."""
    records, api_meta, url = fetch_official_auction_records(start, end)
    calendar = normalize_auction_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calendar.to_csv(output_path, index=False)
    metadata = {
        "source_url": url,
        "source_name": "U.S. Treasury FiscalData Treasury Securities Auctions Data",
        "requested_start": start,
        "requested_end": end,
        "raw_record_count": len(records),
        "normalized_record_count": len(calendar),
        "tenors": list(TENORS),
        "result_timestamp_policy": (
            "Preserved as null because auctions_query does not expose an exact historical "
            "result-publication timestamp; no imputation is performed."
        ),
        "api_meta": api_meta,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    return calendar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2013-05-17")
    parser.add_argument("--end", default=pd.Timestamp.utcnow().date().isoformat())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/treasury_auction_calendar.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/raw/treasury_auction_calendar.metadata.json"),
    )
    args = parser.parse_args()
    calendar = ingest_auction_calendar(args.output, args.metadata, args.start, args.end)
    print(f"Saved {len(calendar)} nominal coupon auctions to {args.output}")


if __name__ == "__main__":
    main()
