"""IS-only B0b: detect MGC climax events with aggregated GC volume ranks."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostic_is.run_state_diagnostics import random_thinning_is  # noqa: E402
from diagnostic_is.run_step1 import (  # noqa: E402
    B_OOS_START,
    B_PARAMS,
    B_SOURCE,
    TEMP,
    _assert_is_only_bars,
    _write_is_only_parquet,
    prior_daily_atr14,
)
from research.volume_auction_landscape_20261007.run_campaign import (  # noqa: E402
    INSTRUMENTS,
    add_causal_volume_baselines,
    add_price_features,
    backtest,
    load_bars,
    signals,
)


RAW = ROOT / "data" / "downloaded" / "GC_v_0_to_v_9_1m_20150102_20230317.parquet"
PROCESSED = ROOT / "data" / "processed" / "parquet" / "GC_v_0_to_v_9_aggregate_volume_5m_20150102_20230317.parquet"
OUT = ROOT / "diagnostic_is" / "gc_b0b"
STATE_TRADES = ROOT / "diagnostic_is" / "state_tables" / "signal_b_state_trades.csv"
SEED = 20261008


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_gc() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(RAW, columns=["volume", "instrument_id", "symbol"])
    idx = pd.DatetimeIndex(raw.index)
    if idx.tz is None:
        raise AssertionError("GC timestamps must be timezone-aware")
    if idx.max() >= pd.Timestamp("2023-03-17T00:00:00Z"):
        raise AssertionError("Post-IS GC bar entered B0b")
    dupes = raw.reset_index().duplicated([raw.index.name or "ts_event", "instrument_id"]).sum()
    if dupes:
        raise AssertionError(f"Duplicate GC timestamp/instrument rows: {dupes}")
    minute = raw.groupby(level=0).agg(
        volume=("volume", "sum"), contract_bars=("instrument_id", "size"),
        distinct_contracts=("instrument_id", "nunique"),
    )
    five = minute.resample("5min", label="left", closed="left").agg(
        volume=("volume", "sum"), contract_bars=("contract_bars", "sum"),
        max_distinct_contracts_in_minute=("distinct_contracts", "max"),
        minutes_with_trades=("volume", lambda s: int((s > 0).sum())),
    )
    five.index = five.index.tz_convert("America/New_York")
    five.index.name = "timestamp"
    five["source_present"] = five.contract_bars.gt(0)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    five.to_parquet(PROCESSED)

    v0 = raw[raw.symbol.eq("GC.v.0")].copy()
    v0.index = pd.DatetimeIndex(v0.index).tz_convert("America/New_York")
    tod = v0.index.strftime("%H:%M")
    v0 = v0[(tod >= "08:20") & (tod < "13:30") & (v0.index.dayofweek < 5)]
    v0["session_date"] = pd.to_datetime(v0.index.date)
    daily_id = v0.groupby("session_date").instrument_id.agg(
        lambda s: int(s.mode().iloc[0]) if len(s) else np.nan
    ).sort_index()
    roll_day = daily_id.ne(daily_id.shift())
    if len(roll_day):
        roll_day.iloc[0] = False
    roll = pd.DataFrame({"instrument_id_v0": daily_id, "mapping_change": roll_day})
    roll["roll_pm2"] = False
    positions = np.flatnonzero(roll.mapping_change.to_numpy())
    for pos in positions:
        roll.iloc[max(0, pos - 2):min(len(roll), pos + 3), roll.columns.get_loc("roll_pm2")] = True
    roll.index.name = "session_date"
    return five, roll


def add_r(trades: pd.DataFrame, daily_atr: pd.Series) -> pd.DataFrame:
    out = trades.copy()
    dates = pd.to_datetime(out.session_date)
    out["prior_daily_atr14"] = dates.map(daily_atr)
    out["cost"] = out.gross_pnl - out.net_pnl
    denominator = out.prior_daily_atr14 * INSTRUMENTS["MGC"].point_value
    out["gross_R"] = out.gross_pnl / denominator
    out["cost_R"] = out.cost / denominator
    out["net_R"] = out.net_pnl / denominator
    return out


def summary_row(name: str, trades: pd.DataFrame) -> dict[str, object]:
    return {
        "version": name,
        "trades": len(trades),
        "gross_usd": trades.gross_pnl.sum(),
        "cost_usd": trades.cost.sum(),
        "net_usd": trades.net_pnl.sum(),
        "gross_R": trades.gross_R.sum(),
        "cost_R": trades.cost_R.sum(),
        "net_R": trades.net_R.sum(),
        "mean_net_R": trades.net_R.mean(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    five, roll = aggregate_gc()
    temp = TEMP / "gc_b0b_MGC_is.parquet"
    TEMP.mkdir(exist_ok=True)
    try:
        _write_is_only_parquet(B_SOURCE, temp, B_OOS_START)
        bars = add_price_features(add_causal_volume_baselines(load_bars(temp, INSTRUMENTS["MGC"])))
        _assert_is_only_bars(bars, B_OOS_START)
    finally:
        temp.unlink(missing_ok=True)
        try:
            TEMP.rmdir()
        except OSError:
            pass

    gc = five.reindex(bars.index)
    # OHLCV emits no record for a zero-trade interval. A missing aggregate
    # record on an otherwise valid MGC bar is therefore a true zero volume.
    bars_gc = bars.copy()
    bars_gc["gc_volume_v0_v9"] = gc.volume.fillna(0).astype(float)
    bars_gc["gc_source_present"] = gc.source_present.fillna(False).astype(bool)
    bars_gc["gc_contract_bars"] = gc.contract_bars.fillna(0).astype(int)
    by_slot = bars_gc.groupby("slot", sort=False)["gc_volume_v0_v9"]
    bars_gc["gc_expected_median_20"] = by_slot.transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).median()
    )
    bars_gc["gc_rvol_20"] = bars_gc.gc_volume_v0_v9 / bars_gc.gc_expected_median_20.replace(0, np.nan)
    bars_gc["rvol_20"] = bars_gc.gc_rvol_20

    original_sig = signals(bars, B_PARAMS)
    gc_sig = signals(bars_gc, B_PARAMS)
    original = backtest(bars, original_sig, INSTRUMENTS["MGC"])
    gc_trades = backtest(bars_gc, gc_sig, INSTRUMENTS["MGC"])
    if pd.to_datetime(original.session_date).max() >= B_OOS_START or pd.to_datetime(gc_trades.session_date).max() >= B_OOS_START:
        raise AssertionError("Post-IS trade in B0b")
    stored = pd.read_csv(STATE_TRADES)
    if len(original) != len(stored) or not np.isclose(original.net_pnl.sum(), stored.net_pnl.sum(), atol=1e-8):
        raise AssertionError("Canonical MGC reproduction failed before B0b")
    atr = prior_daily_atr14(bars)
    original = add_r(original, atr)
    gc_trades = add_r(gc_trades, atr)

    original_events = set(bars.index[original_sig.ne(0)])
    gc_events = set(bars.index[gc_sig.ne(0)])
    original_exec = set(pd.to_datetime(original.signal_time, utc=True))
    gc_exec = set(pd.to_datetime(gc_trades.signal_time, utc=True))
    event_intersection = original_events & gc_events
    trade_intersection = original_exec & gc_exec
    comparison = pd.DataFrame([summary_row("MGC_volume_canonical", original), summary_row("GC_v0_v9_volume", gc_trades)])
    comparison["raw_signal_events"] = [len(original_events), len(gc_events)]
    comparison["event_overlap_count"] = len(event_intersection)
    comparison["event_overlap_vs_version"] = [
        len(event_intersection) / len(original_events) if original_events else np.nan,
        len(event_intersection) / len(gc_events) if gc_events else np.nan,
    ]
    comparison["executable_overlap_count"] = len(trade_intersection)
    comparison["executable_overlap_vs_version"] = [
        len(trade_intersection) / len(original_exec) if original_exec else np.nan,
        len(trade_intersection) / len(gc_exec) if gc_exec else np.nan,
    ]

    gc_trades["roll_pm2"] = np.where(
        pd.to_datetime(gc_trades.session_date).map(roll.roll_pm2).fillna(False), "roll_pm2", "outside_roll"
    )
    gc_trades.to_csv(OUT / "B0b_gc_volume_mgc_trades.csv", index=False)
    comparison.to_csv(OUT / "B0b_version_comparison.csv", index=False)
    gc_trades.assign(year=pd.to_datetime(gc_trades.session_date).dt.year).groupby("year", as_index=False).agg(
        trades=("net_R", "size"), gross_usd=("gross_pnl", "sum"), cost_usd=("cost", "sum"), net_usd=("net_pnl", "sum"),
        gross_R=("gross_R", "sum"), cost_R=("cost_R", "sum"), net_R=("net_R", "sum"), mean_net_R=("net_R", "mean"),
    ).to_csv(OUT / "B0b_gc_volume_yearly.csv", index=False)
    gc_trades.groupby("roll_pm2", as_index=False).agg(
        trades=("net_R", "size"), gross_usd=("gross_pnl", "sum"), cost_usd=("cost", "sum"), net_usd=("net_pnl", "sum"),
        gross_R=("gross_R", "sum"), cost_R=("cost_R", "sum"), net_R=("net_R", "sum"),
    ).to_csv(OUT / "B0b_gc_volume_roll_breakdown.csv", index=False)
    roll.reset_index().to_csv(OUT / "GC_v0_mapping_roll_days.csv", index=False)

    thinning, draws = random_thinning_is(bars, gc_trades)
    thinning["version"] = "B0b_GC_v0_v9_volume"
    thinning.to_csv(OUT / "B6_B0b_random_thinning_summary.csv", index=False)
    draws.to_csv(OUT / "B6_B0b_random_thinning_draws.csv", index=False)
    metadata = {
        "source": str(RAW.relative_to(ROOT)),
        "source_sha256": sha256(RAW),
        "processed": str(PROCESSED.relative_to(ROOT)),
        "processed_sha256": sha256(PROCESSED),
        "aggregation": "sum volume over distinct GC volume ranks 0..9 per minute, then sum to left-labelled 5-minute bars",
        "gc_rvol": "current GC aggregate 5m volume / same-slot median over prior 20 sessions",
        "price_and_execution": "canonical MGC bars and costs; only volume state replaced",
        "mGC_reproduction_trades": len(original),
        "mGC_reproduction_net_usd": float(original.net_pnl.sum()),
        "gc_b0b_trades": len(gc_trades),
        "gc_b0b_net_usd": float(gc_trades.net_pnl.sum()),
        "gc_b0b_net_R": float(gc_trades.net_R.sum()),
        "mGC_bars_with_any_GC_source_record": int(bars_gc.gc_source_present.sum()),
        "mGC_bars": len(bars_gc),
        "post_is_statistics_computed": False,
        "seed": SEED,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"comparison": comparison.to_dict("records"), "thinning": thinning.to_dict("records"), "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
