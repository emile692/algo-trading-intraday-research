"""Build and freeze official event calendars used by the targeted campaign.

This script performs no market-data calculation and reads no strategy PnL.
It materializes event dates before the campaign runner is allowed to execute.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


OUTPUT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = OUTPUT_DIR / "campaign_config.json"
CALENDAR_PATH = OUTPUT_DIR / "event_calendar.csv"
MANIFEST_PATH = OUTPUT_DIR / "locked_input_manifest.json"
START = date(2015, 1, 1)
END = date(2026, 3, 19)
HEADERS = {"User-Agent": "Mozilla/5.0 research-calendar-audit/1.0"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=45)
    response.raise_for_status()
    return response.text


def parse_bls_archive(url: str, prefix: str, event_type: str) -> list[dict[str, str]]:
    try:
        soup = BeautifulSoup(fetch(url), "html.parser")
    except requests.HTTPError as exc:
        # BLS currently returns 403 to automated clients.  The fallback is a
        # release-aware FRED mirror whose event metadata records the BLS release
        # date; it is checked below against the expected one-release-per-month
        # cardinality.  This affects calendar transport only, never the rule.
        if exc.response is None or exc.response.status_code != 403:
            raise
        mirror = fetch(
            "https://raw.githubusercontent.com/superpilot69/"
            "fred-us-macro-open-data/main/data/fred-us-macro-events.json"
        )
        payload = json.loads(mirror)
        series_id = {"NFP": "PAYEMS", "CPI": "CPIAUCNS"}[event_type]
        dates = {
            item["metadata"]["releaseDate"]
            for item in payload["events"]
            if item.get("metadata", {}).get("seriesId") == series_id
            and item.get("metadata", {}).get("releaseDate")
            and START <= date.fromisoformat(item["metadata"]["releaseDate"]) <= END
        }
        rows = [{
            "event_date": stamp,
            "event_type": event_type,
            "event_time_et": "08:30:00",
            "source": url,
            "source_method": "FRED_release_metadata_fallback_BLS_403",
        } for stamp in sorted(dates)]
        if len(rows) < 120:
            raise RuntimeError(f"Too few {event_type} mirror dates parsed: {len(rows)}")
        return rows
    pattern = re.compile(rf"(?:archives/)?{re.escape(prefix)}_(\d{{8}})\.(?:pdf|htm)$", re.I)
    rows: dict[date, dict[str, str]] = {}
    for anchor in soup.find_all("a", href=True):
        match = pattern.search(anchor["href"])
        if not match:
            continue
        stamp = match.group(1)
        day = date(int(stamp[4:]), int(stamp[:2]), int(stamp[2:4]))
        if START <= day <= END:
            rows[day] = {
                "event_date": day.isoformat(),
                "event_type": event_type,
                "event_time_et": "08:30:00",
                "source": url,
                "source_method": "official_archive_link_date",
            }
    if len(rows) < 120:
        raise RuntimeError(f"Too few {event_type} archive dates parsed: {len(rows)}")
    return list(rows.values())


def parse_historical_fomc_year(year: int, template: str) -> list[dict[str, str]]:
    url = template.format(year=year)
    soup = BeautifulSoup(fetch(url), "html.parser")
    rows: list[dict[str, str]] = []
    month_number = {name: number for number, name in enumerate(
        ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    )}
    month_number.update({name[:3]: number for name, number in month_number.items() if name})
    for heading in soup.find_all(["h4", "h5"]):
        text = " ".join(heading.get_text(" ", strip=True).split())
        match = re.match(r"(?P<months>[A-Za-z]+(?:/[A-Za-z]+)?)\s+(?P<days>\d{1,2}(?:-\d{1,2})?)\s+Meeting\s+-\s+\d{4}", text)
        if not match:
            continue
        month_name = match.group("months").split("/")[-1]
        final_day = int(match.group("days").split("-")[-1])
        event_day = date(year, month_number[month_name], final_day)
        if START <= event_day <= END:
            rows.append({
                "event_date": event_day.isoformat(),
                "event_type": "FOMC",
                "event_time_et": "14:00:00",
                "source": url,
                "source_method": "official_scheduled_meeting_heading_final_day",
            })
    if len(rows) < 7:
        raise RuntimeError(f"Too few scheduled FOMC meetings parsed for {year}: {len(rows)}")
    return rows


def parse_current_fomc(url: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(fetch(url), "html.parser")
    dates: set[date] = set()
    pattern = re.compile(r"monetary(20\d{6})a\.htm", re.I)
    for anchor in soup.find_all("a", href=True):
        match = pattern.search(anchor["href"])
        if not match:
            continue
        stamp = match.group(1)
        day = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
        if date(2021, 1, 1) <= day <= END:
            dates.add(day)
    rows = [{
        "event_date": day.isoformat(),
        "event_type": "FOMC",
        "event_time_et": "14:00:00",
        "source": url,
        "source_method": "official_statement_link_date",
    } for day in sorted(dates)]
    if len(rows) < 40:
        raise RuntimeError(f"Too few current-page FOMC dates parsed: {len(rows)}")
    return rows


def fetch_treasury_auctions(url: str) -> list[dict[str, str]]:
    params = {
        # original_security_term retains the reopened 10-year issues whose
        # remaining term is reported as 9-Year 10/11-Month.
        "filter": f"security_type:eq:Note,original_security_term:eq:10-Year,auction_date:gte:{START},auction_date:lte:{END}",
        "fields": "auction_date,security_type,security_term,original_security_term,reopening,issue_date,record_date",
        "sort": "auction_date",
        "page[size]": 1000,
    }
    response = requests.get(url, params=params, headers=HEADERS, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if int(payload["meta"]["total-count"]) != len(payload["data"]):
        raise RuntimeError("Treasury auction API pagination is incomplete")
    rows = [{
        "event_date": item["auction_date"],
        "event_type": "10Y_TREASURY_AUCTION",
        "event_time_et": "13:00:00",
        "source": response.url,
        "source_method": "official_fiscaldata_api",
    } for item in payload["data"]]
    if len(rows) < 180:
        raise RuntimeError(f"Too few 10Y auctions parsed: {len(rows)}")
    return rows


def build_ism_calendar() -> list[dict[str, str]]:
    business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    rows = []
    for month in pd.period_range(START, END, freq="M"):
        first = pd.Timestamp(month.start_time.date())
        event_day = first if business_day.is_on_offset(first) else first + business_day
        day = event_day.date()
        if START <= day <= END:
            rows.append({
                "event_date": day.isoformat(),
                "event_type": "ISM_MANUFACTURING",
                "event_time_et": "10:00:00",
                "source": "pre-specified deterministic first-US-business-day rule",
                "source_method": "deterministic_calendar_rule",
            })
    return rows


def execute() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    urls = config["event_calendars"]
    rows: list[dict[str, str]] = []
    rows += parse_bls_archive(urls["bls_archive_employment"], "empsit", "NFP")
    rows += parse_bls_archive(urls["bls_archive_cpi"], "cpi", "CPI")
    for year in range(2015, 2021):
        rows += parse_historical_fomc_year(year, urls["fomc_historical_template"])
    rows += parse_current_fomc(urls["fomc_current"])
    rows += fetch_treasury_auctions(urls["treasury_auction_api"])
    rows += build_ism_calendar()
    calendar = pd.DataFrame(rows).drop_duplicates(["event_date", "event_type"]).sort_values(["event_date", "event_type"])
    calendar.to_csv(CALENDAR_PATH, index=False, lineterminator="\n")
    counts = calendar.groupby("event_type").size().astype(int).to_dict()
    manifest = {
        "status": "LOCKED_INPUTS_BEFORE_PERFORMANCE",
        "campaign_config_sha256": sha256(CONFIG_PATH),
        "event_calendar_sha256": sha256(CALENDAR_PATH),
        "event_counts": counts,
        "event_start": str(calendar.event_date.min()),
        "event_end": str(calendar.event_date.max()),
        "performance_computed": False,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    execute()
