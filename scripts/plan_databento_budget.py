"""Find the earliest affordable Databento start date without downloading data."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta

import pandas as pd

from src.data.downloader import DATASET, _historical_client


def estimate(client, ticker: str, start: pd.Timestamp, end: pd.Timestamp, schema: str) -> float:
    return float(
        client.metadata.get_cost(
            dataset=DATASET,
            symbols=ticker,
            stype_in="continuous",
            schema=schema,
            start=start.isoformat(),
            end=end.isoformat(),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="ZN.v.0")
    parser.add_argument("--earliest", default="2010-06-06T00:00:00Z")
    parser.add_argument("--end", required=True)
    parser.add_argument("--schema", default="ohlcv-1m")
    parser.add_argument("--budget", type=float, default=15.75)
    parser.add_argument("--output", default="data/downloaded/ZN_v_0_budget_plan.json")
    args = parser.parse_args()

    client = _historical_client()
    earliest = pd.Timestamp(args.earliest)
    end = pd.Timestamp(args.end)
    if earliest.tzinfo is None:
        earliest = earliest.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    full_cost = estimate(client, args.ticker, earliest, end, args.schema)
    calls = [{"start": earliest.isoformat(), "estimated_cost_usd": full_cost}]
    if full_cost <= args.budget:
        selected, selected_cost = earliest, full_cost
    else:
        low = earliest
        high = end - timedelta(days=1)
        high_cost = estimate(client, args.ticker, high, end, args.schema)
        calls.append({"start": high.isoformat(), "estimated_cost_usd": high_cost})
        if high_cost > args.budget:
            raise RuntimeError("Even one day exceeds the configured budget")
        while (high - low).days > 1:
            middle = low + timedelta(days=(high - low).days // 2)
            middle_cost = estimate(client, args.ticker, middle, end, args.schema)
            calls.append({"start": middle.isoformat(), "estimated_cost_usd": middle_cost})
            if middle_cost <= args.budget:
                high = middle
            else:
                low = middle
        selected = high
        selected_cost = estimate(client, args.ticker, selected, end, args.schema)
        calls.append({"start": selected.isoformat(), "estimated_cost_usd": selected_cost})
    result = {
        "ticker": args.ticker,
        "dataset": DATASET,
        "schema": args.schema,
        "stype_in": "continuous",
        "budget_usd": args.budget,
        "selected_start": selected.isoformat(),
        "end": end.isoformat(),
        "estimated_cost_usd": selected_cost,
        "full_available_candidate_start": earliest.isoformat(),
        "full_candidate_cost_usd": full_cost,
        "metadata_calls": calls,
    }
    path = pd.io.common.stringify_path(args.output)
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("selected_start", "end", "estimated_cost_usd", "budget_usd", "full_candidate_cost_usd")}, indent=2))
    print(f"Saved plan to {target}")


if __name__ == "__main__":
    main()
