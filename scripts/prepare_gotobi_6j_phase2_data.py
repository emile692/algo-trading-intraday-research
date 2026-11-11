"""Audit and acquire the frozen Gotobi Phase-2 event windows.

The script is deliberately event-only: it resolves the Phase-1 underlying
instrument ID on each event date, estimates every request, and refuses to
download above a hard total-cost guard.  The API key is read from the process
environment and is never written to disk.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

import databento as db
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "research" / "gotobi_6j_phase1_20261003"
OUT = ROOT / "research" / "gotobi_6j_phase2_20261003"
WINDOWS = OUT / "phase2_event_windows.csv"
ESTIMATE = OUT / "databento_cost_estimate.json"
MBP_DIR = OUT / "mbp1_events"
DEF_DIR = OUT / "definition_events"
DATASET = "GLBX.MDP3"
DEFAULT_GUARD_USD = 5.00


def frozen_events() -> pd.DataFrame:
    trades = pd.read_csv(PHASE1 / "trade_ledger.csv")
    mask = (
        trades["rule"].eq("pre_fix_short")
        & trades["variant"].eq("entry_0930")
        & trades["subset"].eq("eom")
        & trades["slippage_ticks_per_side"].eq(1)
    )
    events = trades.loc[mask].drop_duplicates("japan_date").copy()
    if len(events) != 136:
        raise RuntimeError(f"Phase-1 replication gate failed: expected 136 events, got {len(events)}")
    events["japan_date"] = pd.to_datetime(events["japan_date"])
    events["entry_utc"] = (
        events["japan_date"].dt.tz_localize("Asia/Tokyo") + pd.Timedelta(hours=9, minutes=31)
    ).dt.tz_convert("UTC")
    events["exit_utc"] = (
        events["japan_date"].dt.tz_localize("Asia/Tokyo") + pd.Timedelta(hours=9, minutes=54)
    ).dt.tz_convert("UTC")
    # A small lead establishes a current book before entry; the tail captures
    # the first observable quote immediately after the scheduled exit.
    events["request_start_utc"] = events["entry_utc"] - pd.Timedelta(seconds=30)
    events["request_end_utc"] = events["exit_utc"] + pd.Timedelta(seconds=30)
    return events


def resolve_asof(client: db.Historical, instrument_id: int, day: pd.Timestamp) -> str:
    start = day.date().isoformat()
    end = (day + pd.Timedelta(days=1)).date().isoformat()
    payload = client.symbology.resolve(
        dataset=DATASET,
        symbols=[int(instrument_id)],
        stype_in="instrument_id",
        stype_out="raw_symbol",
        start_date=start,
        end_date=end,
    )
    intervals = payload["result"].get(str(int(instrument_id)), [])
    choices = [row["s"] for row in intervals if row["d0"] <= start < row["d1"]]
    if len(choices) != 1:
        raise RuntimeError(f"Ambiguous as-of mapping for {instrument_id} on {start}: {choices}")
    return choices[0]


def request_kwargs(row: pd.Series, schema: str) -> dict:
    start = row.request_start_utc
    end = row.request_end_utc
    if schema == "definition":
        # Definition snapshots are emitted from UTC midnight.
        start = row.entry_utc.normalize()
        end = start + pd.Timedelta(days=1)
    return {
        "dataset": DATASET,
        "start": start,
        "end": end,
        "symbols": row.raw_symbol,
        "schema": schema,
        "stype_in": "raw_symbol",
    }


def estimate(client: db.Historical, events: pd.DataFrame) -> dict:
    totals: dict[str, dict[str, float | int]] = {}
    for schema in ("mbp-1", "definition"):
        costs: list[float] = []
        records: list[int] = []
        sizes: list[int] = []
        def one(row: pd.Series) -> tuple[float, int, int]:
            kwargs = request_kwargs(row, schema)
            return (
                float(client.metadata.get_cost(**kwargs)),
                int(client.metadata.get_record_count(**kwargs)),
                int(client.metadata.get_billable_size(**kwargs)),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            for cost, record_count, size in pool.map(one, [row for _, row in events.iterrows()]):
                costs.append(cost)
                records.append(record_count)
                sizes.append(size)
        totals[schema] = {
            "requests": len(events),
            "estimated_cost_usd": sum(costs),
            "record_count": sum(records),
            "billable_size_bytes": sum(sizes),
            "zero_record_windows": sum(value == 0 for value in records),
            "maximum_event_cost_usd": max(costs),
        }
    totals["combined_estimated_cost_usd"] = sum(
        float(totals[schema]["estimated_cost_usd"]) for schema in ("mbp-1", "definition")
    )
    return totals


def download(key: str, events: pd.DataFrame, guard_usd: float, estimate_payload: dict) -> None:
    total = float(estimate_payload["combined_estimated_cost_usd"])
    if total > guard_usd:
        raise RuntimeError(f"Estimated cost ${total:.6f} exceeds hard guard ${guard_usd:.2f}")
    MBP_DIR.mkdir(parents=True, exist_ok=True)
    DEF_DIR.mkdir(parents=True, exist_ok=True)
    def one(row: pd.Series) -> str:
        day = row.japan_date.strftime("%Y-%m-%d")
        target = MBP_DIR / f"{day}_{row.raw_symbol}_mbp1.parquet"
        if target.exists():
            return str(target)
        local_client = db.Historical(key)
        store = local_client.timeseries.get_range(
            **request_kwargs(row, "mbp-1"), stype_out="instrument_id"
        )
        frame = store.to_df()
        if frame.empty:
            raise RuntimeError(f"Empty MBP-1 response for {day} {row.raw_symbol}")
        frame.to_parquet(target)
        return str(target)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, [row for _, row in events.iterrows()]))

    def one_definition(row: pd.Series) -> str:
        day = row.japan_date.strftime("%Y-%m-%d")
        target = DEF_DIR / f"{day}_{row.raw_symbol}_definition.parquet"
        if target.exists():
            return str(target)
        local_client = db.Historical(key)
        store = local_client.timeseries.get_range(
            **request_kwargs(row, "definition"), stype_out="instrument_id"
        )
        frame = store.to_df()
        if frame.empty:
            raise RuntimeError(f"Empty definition response for {day} {row.raw_symbol}")
        frame.to_parquet(target)
        return str(target)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one_definition, [row for _, row in events.iterrows()]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--guard-usd", type=float, default=DEFAULT_GUARD_USD)
    args = parser.parse_args()
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY is required")
    OUT.mkdir(parents=True, exist_ok=True)
    client = db.Historical(key)
    if WINDOWS.exists():
        events = pd.read_csv(
            WINDOWS,
            parse_dates=["japan_date", "entry_utc", "exit_utc", "request_start_utc", "request_end_utc"],
        )
        if len(events) != 136 or "raw_symbol" not in events:
            raise RuntimeError("Cached Phase-2 event map does not match the frozen 136-event rule")
    else:
        events = frozen_events()
        events["raw_symbol"] = [
            resolve_asof(client, int(row.id_p0931), row.japan_date) for _, row in events.iterrows()
        ]
        events.to_csv(WINDOWS, index=False)
    if ESTIMATE.exists():
        payload = json.loads(ESTIMATE.read_text(encoding="utf-8"))
    else:
        payload = estimate(client, events)
        ESTIMATE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if args.download:
        download(key, events, args.guard_usd, payload)
        print(f"Downloaded {len(events)} event windows to {MBP_DIR}")
    os.environ.pop("DATABENTO_API_KEY", None)


if __name__ == "__main__":
    main()
