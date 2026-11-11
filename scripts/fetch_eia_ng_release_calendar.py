"""Reconstruct WNGSR release dates from dated official EIA dashboard archives.

The dashboard is updated on the WNGSR release date and follows its holiday
schedule.  We probe business-day archive filenames and retain only HTTP 200s.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
import re
import subprocess


URL = (
    "https://www.eia.gov/naturalgas/storage/dashboard-api/archives/"
    "{date}_natural_gas_storage_dashboard.pdf"
)


def statuses_for(days: list[dt.date]) -> list[tuple[dt.date, int]]:
    """Use curl multiplexing; EIA throttles independent Python connections."""
    by_url = {URL.format(date=day.strftime("%Y%m%d")): day for day in days}
    found: dict[dt.date, int] = {}
    for year in sorted({day.year for day in days}):
        urls = [url for url, day in by_url.items() if day.year == year]
        command = [
            "curl.exe",
            "--parallel",
            "--parallel-max",
            "20",
            "--head",
            "--silent",
            "--output",
            "NUL",
            "--write-out",
            "%{http_code} %{url_effective}\\n",
            *urls,
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        for line in completed.stdout.splitlines():
            match = re.match(r"^(\d{3}) (https://\S+)$", line.strip())
            if match and match.group(2) in by_url:
                found[by_url[match.group(2)]] = int(match.group(1))
    return [(day, found.get(day, 0)) for day in days]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-10-03")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    thursdays = []
    day = start
    while day <= end:
        if day.weekday() == 3:
            thursdays.append(day)
        day += dt.timedelta(days=1)

    thursday_results = statuses_for(thursdays)

    # Only weeks without a Thursday archive need holiday-exception probes.
    exception_candidates = []
    for thursday, status in thursday_results:
        if status != 200:
            for offset in (-3, -2, -1, 1):
                candidate = thursday + dt.timedelta(days=offset)
                if start <= candidate <= end and candidate.weekday() < 5:
                    exception_candidates.append(candidate)
    exception_results = statuses_for(exception_candidates)
    results = thursday_results + exception_results

    failures = [day for day, status in results if status == 0]
    if failures:
        raise RuntimeError(f"Network failures for {len(failures)} dates; first={failures[0]}")

    releases = sorted((day, status) for day, status in results if status == 200)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["release_date", "release_time", "timezone", "source_url"],
        )
        writer.writeheader()
        for day, _ in releases:
            writer.writerow(
                {
                    "release_date": day.isoformat(),
                    # Historical exception notices use noon for releases
                    # advanced to Monday/Wednesday and 10:30 for Friday delays.
                    "release_time": "12:00:00" if day.weekday() in (0, 2) else "10:30:00",
                    "timezone": "America/New_York",
                    "source_url": URL.format(date=day.strftime("%Y%m%d")),
                }
            )
    print(
        f"probed={len(results)} thursdays={len(thursdays)} "
        f"releases={len(releases)} output={output}"
    )


if __name__ == "__main__":
    main()
