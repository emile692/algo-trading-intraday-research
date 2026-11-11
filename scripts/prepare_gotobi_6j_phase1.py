"""Prepare the frozen Gotobi calendar and fail clearly if 6J bars are absent."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from src.analytics.gotobi_6j_phase1 import (
    build_japan_business_calendar,
    japan_bank_holidays,
    load_official_japan_holidays,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "gotobi_6j_phase1_20261003"
HOLIDAYS = OUT / "japan_national_holidays_official.csv"
EXPECTED_DATA = ROOT / "data" / "downloaded" / "6J_v_0_1m_20150101_20261003.parquet"


def main() -> None:
    official = load_official_japan_holidays(HOLIDAYS)
    bank_holidays = japan_bank_holidays(official, 2015, 2026)
    calendar = build_japan_business_calendar(
        dt.date(2015, 1, 1), dt.date(2026, 10, 3), bank_holidays
    )
    calendar.to_csv(OUT / "japan_business_gotobi_calendar.csv", index=False)
    summary = {
        "calendar_start": str(calendar.japan_date.min()),
        "calendar_end": str(calendar.japan_date.max()),
        "japan_business_days": int(len(calendar)),
        "gotobi_days": int(calendar.is_gotobi.sum()),
        "gotobi_ex_eom_days": int(calendar.is_gotobi_ex_eom.sum()),
        "eom_days": int(calendar.is_eom.sum()),
        "all_windows_structurally_topstep_tradable": bool(
            calendar.all_requested_windows_structurally_topstep_tradable.all()
        ),
        "expected_6j_path": str(EXPECTED_DATA),
        "market_data_available": EXPECTED_DATA.exists(),
        "phase1_status": "READY_FOR_MARKET_DATA" if not EXPECTED_DATA.exists() else "READY_TO_RUN",
    }
    (OUT / "preflight_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
