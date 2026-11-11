"""Fetch and parse official BLS release calendars and Federal Reserve FOMC statements.

The parser deliberately derives every event date from cached official HTML.  It
does not contain a hand-written event-date fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "diagnostic_is" / "sources" / "macro"
OUT = ROOT / "diagnostic_is" / "macro_calendar_official.csv"
MANIFEST = SOURCE_DIR / "manifest.json"
USER_AGENT = "Mozilla/5.0 (compatible; reproducible-research/1.0; +https://openai.com/)"
IS_UNION_START = pd.Timestamp("2010-06-07")
IS_UNION_END = pd.Timestamp("2024-02-22")


def sources() -> list[tuple[str, str]]:
    items = [
        (f"bls_{year}.html", f"https://www.bls.gov/schedule/{year}/home.htm")
        for year in range(2010, 2025)
    ]
    items += [
        (f"fed_fomc_{year}.html", f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm")
        for year in range(2010, 2021)
    ]
    items.append(
        ("fed_fomc_current.html", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    )
    return items


def fetch() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for filename, url in sources():
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=45) as response:
            payload = response.read()
            status = response.status
        if status != 200 or len(payload) < 1000:
            raise RuntimeError(f"Bad response for {url}: status={status}, bytes={len(payload)}")
        path = SOURCE_DIR / filename
        path.write_bytes(payload)
        manifest.append({
            "file": filename,
            "url": url,
            "http_status": status,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def clean(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split())


def parse_bls(path: Path, url: str) -> list[dict[str, object]]:
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for tr in soup.find_all("tr"):
        direct_cells = tr.find_all(["th", "td"], recursive=False)
        date_cells = [cell for cell in direct_cells if "date-cell" in (cell.get("class") or [])]
        desc_cells = [cell for cell in direct_cells if "desc-cell" in (cell.get("class") or [])]
        if len(date_cells) != 1 or len(desc_cells) != 1:
            continue
        cells = [clean(cell.get_text(" ", strip=True)) for cell in direct_cells]
        if len(cells) < 2:
            continue
        joined = " | ".join(cells)
        event = None
        release_names = [clean(tag.get_text(" ", strip=True)) for tag in desc_cells[0].find_all("strong")]
        if any(re.fullmatch(r"Consumer Price Index(?: for All Urban Consumers)?", c, re.I) for c in release_names):
            event = "CPI"
        elif any(re.fullmatch(r"(?:The )?Employment Situation", c, re.I) for c in release_names):
            event = "NFP"
        if event is None:
            continue
        date_match = re.search(
            r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+"
            r"\d{1,2},?\s+\d{4}",
            joined,
            re.I,
        )
        if not date_match:
            raise RuntimeError(f"BLS release row has no parseable date in {path.name}: {joined}")
        raw_date = re.sub(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+", "", date_match.group(0), flags=re.I)
        date = pd.Timestamp(datetime.strptime(raw_date.replace(",", ""), "%B %d %Y")).date().isoformat()
        time_match = re.search(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b", joined, re.I)
        key = (date, event)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "date": date,
            "event": event,
            "release_time_et": clean(time_match.group(0).upper()) if time_match else None,
            "source_url": url,
            "source_file": path.name,
            "source_record": joined,
        })
    return rows


FOMC_HREF_DATE = re.compile(r"monetary(?:/|)(\d{8})a\.htm(?:l)?$", re.I)


def parse_fed(path: Path, url: str) -> list[dict[str, object]]:
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        label = clean(anchor.get_text(" ", strip=True))
        href = anchor.get("href", "")
        match = FOMC_HREF_DATE.search(href)
        # The official statement HTML has the stable monetaryYYYYMMDDa.htm
        # pattern. Exact suffix matching excludes implementation notes (a1),
        # minutes, projections and longer-run strategy statements (b).
        if not match:
            continue
        date = datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
        if date in seen:
            continue
        seen.add(date)
        rows.append({
            "date": date,
            "event": "FOMC",
            "release_time_et": None,
            "source_url": url,
            "source_file": path.name,
            "source_record": f"{label} | {href}",
        })
    return rows


def parse() -> pd.DataFrame:
    if not MANIFEST.exists():
        raise FileNotFoundError("Run with --fetch first; no official-source manifest exists")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    url_by_file = {row["file"]: row["url"] for row in manifest}
    rows: list[dict[str, object]] = []
    for filename, url in sources():
        path = SOURCE_DIR / filename
        if not path.exists() or url_by_file.get(filename) != url:
            raise RuntimeError(f"Missing or mismatched cached source: {filename}")
        rows.extend(parse_bls(path, url) if filename.startswith("bls_") else parse_fed(path, url))
    frame = pd.DataFrame(rows).drop_duplicates(["date", "event"]).sort_values(["date", "event"])
    frame["date"] = pd.to_datetime(frame.date)
    years = frame.date.dt.year
    for year in range(2010, 2025):
        expected = {"CPI", "NFP", "FOMC"}
        present = set(frame.loc[years.eq(year), "event"])
        if present != expected:
            raise RuntimeError(f"Incomplete parsed event types for {year}: {sorted(present)}")
    # Raw official pages are retained for audit, but the parsed table consumed
    # by diagnostics is hard-clipped to the union of the locked IS windows.
    frame = frame[frame.date.between(IS_UNION_START, IS_UNION_END)].copy()
    if frame.date.min() < IS_UNION_START or frame.date.max() > IS_UNION_END:
        raise AssertionError("Out-of-IS macro date entered the parsed analysis calendar")
    frame.to_csv(OUT, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="Refresh official HTML before parsing")
    args = parser.parse_args()
    if args.fetch:
        fetch()
    frame = parse()
    coverage = frame.assign(year=frame.date.dt.year).pivot_table(index="year", columns="event", values="date", aggfunc="count", fill_value=0)
    print(coverage.to_string())
    print(f"\nWrote {len(frame)} parsed events to {OUT}")


if __name__ == "__main__":
    main()
