"""Validation campaign comparing baseline 1H execution vs hybrid 1m execution."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analytics.volume_climax_pullback_common import load_symbol_data, resample_rth_1h
from src.engine.execution_model import ExecutionModel
from src.engine.volume_climax_pullback_v2_backtester import (
    run_volume_climax_pullback_v2_backtest,
    run_volume_climax_pullback_v2_backtest_hybrid_1m,
)
from src.engine.vwap_backtester import resolve_instrument_details
from src.strategy.volume_climax_pullback_v2 import build_volume_climax_pullback_v2_signal_frame, build_volume_climax_pullback_v3_variants


def run_validation(symbol: str = "MNQ", output_dir: Path | None = None) -> Path:
    root = output_dir or (Path("export") / "volume_climax_pullback_hybrid_execution_validation")
    root.mkdir(parents=True, exist_ok=True)

    minute_df = load_symbol_data(symbol)
    signal_1h_df = resample_rth_1h(minute_df)
    variant = build_volume_climax_pullback_v3_variants(symbol)[0]
    signal_df = build_volume_climax_pullback_v2_signal_frame(signal_1h_df, variant)
    instrument = resolve_instrument_details(symbol)
    model = ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0.0, tick_size=instrument.tick_size)

    baseline = run_volume_climax_pullback_v2_backtest(signal_df, variant, model, instrument).trades
    hybrid_after = run_volume_climax_pullback_v2_backtest_hybrid_1m(
        signal_df, minute_df, variant, model, instrument, protective_orders_active_from="after_entry_fill"
    ).trades
    hybrid_next = run_volume_climax_pullback_v2_backtest_hybrid_1m(
        signal_df, minute_df, variant, model, instrument, protective_orders_active_from="next_execution_bar"
    ).trades

    baseline.to_csv(root / "trades_baseline_1h.csv", index=False)
    hybrid_after.to_csv(root / "trades_hybrid_after_entry_fill.csv", index=False)
    hybrid_next.to_csv(root / "trades_hybrid_next_execution_bar.csv", index=False)

    comparison = baseline[["entry_time", "exit_time", "entry_price", "exit_price", "exit_reason", "net_pnl_usd"]].copy()
    comparison = comparison.add_prefix("baseline_")
    for col in ["entry_time", "exit_time", "entry_price", "exit_price", "exit_reason", "net_pnl_usd", "setup_bar_label_time", "direction"]:
        if col in hybrid_after.columns:
            comparison[f"hybrid_{col}"] = hybrid_after[col].values[: len(comparison)]
    comparison["divergence_flag"] = (
        comparison.get("baseline_exit_reason", "") != comparison.get("hybrid_exit_reason", "")
    )
    comparison.to_csv(root / "trade_comparison.csv", index=False)

    metrics = pd.DataFrame(
        [
            {"model": "baseline_1h", "trades": len(baseline), "net_pnl_usd": float(baseline.get("net_pnl_usd", pd.Series(dtype=float)).sum())},
            {"model": "hybrid_after_entry_fill", "trades": len(hybrid_after), "net_pnl_usd": float(hybrid_after.get("net_pnl_usd", pd.Series(dtype=float)).sum())},
            {"model": "hybrid_next_execution_bar", "trades": len(hybrid_next), "net_pnl_usd": float(hybrid_next.get("net_pnl_usd", pd.Series(dtype=float)).sum())},
        ]
    )
    metrics.to_csv(root / "metrics_comparison.csv", index=False)
    (root / "final_report.md").write_text("# Hybrid execution validation\n\nDaily loss guard remains post-trade metrics only.", encoding="utf-8")
    (root / "run_metadata.json").write_text(json.dumps({"symbol": symbol}, indent=2), encoding="utf-8")
    return root
