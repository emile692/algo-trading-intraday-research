"""Validation campaign for hybrid 1H signal / 1min execution on volume climax pullback."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.volume_climax_pullback_common import (
    latest_path_for_symbol,
    load_symbol_data,
    resample_rth_1h,
    safe_float,
    summarize_scope,
)
from src.data.session import extract_rth
from src.engine.volume_climax_pullback_v2_backtester import run_volume_climax_pullback_v2_backtest
from src.engine.vwap_backtester import build_execution_model_for_profile
from src.strategy.volume_climax_pullback_v2 import (
    VolumeClimaxPullbackV2Variant,
    build_volume_climax_pullback_v2_signal_frame,
    build_volume_climax_pullback_v3_variants,
    prepare_volume_climax_pullback_v2_features,
)

DEFAULT_SYMBOL = "MNQ"
DEFAULT_OUTPUT_ROOT = Path("export")


def _resolve_variant(symbol: str, variant_name: str | None) -> VolumeClimaxPullbackV2Variant:
    catalog = {variant.name: variant for variant in build_volume_climax_pullback_v3_variants(symbol)}
    if variant_name is not None:
        if variant_name not in catalog:
            raise ValueError(f"Variant {variant_name!r} was not found for {symbol}.")
        return catalog[variant_name]
    if not catalog:
        raise ValueError(f"No V3 variants are available for {symbol}.")
    return next(iter(catalog.values()))


def _metrics_row(label: str, trades: pd.DataFrame, signal_df: pd.DataFrame) -> dict[str, Any]:
    sessions = sorted(pd.to_datetime(signal_df["session_date"]).dt.date.unique()) if not signal_df.empty else []
    metrics = summarize_scope(trades, signal_df, sessions)
    return {
        "scenario": label,
        "trades": int(len(trades)),
        "net_pnl_usd": float(metrics["net_pnl"]),
        "profit_factor": float(metrics["profit_factor"]),
        "sharpe": float(metrics["sharpe"]),
        "max_drawdown_usd": float(metrics["max_drawdown"]),
        "expectancy_usd": float(metrics["expectancy"]),
        "hit_rate": float(metrics["hit_rate"]),
        "raw_signal_count": int(metrics["raw_signal_count"]),
        "avg_minutes_held": float(pd.to_numeric(trades.get("minutes_held"), errors="coerce").mean()) if not trades.empty else 0.0,
    }


def _scenario_trade_view(trades: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep_cols = [
        "setup_bar_label_time",
        "direction",
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
        "exit_reason",
        "pnl_usd",
    ]
    available = [column for column in keep_cols if column in trades.columns]
    view = trades[available].copy() if not trades.empty else pd.DataFrame(columns=available)
    rename_map = {
        "entry_time": f"{prefix}_entry_time",
        "entry_price": f"{prefix}_entry_price",
        "exit_time": f"{prefix}_exit_time",
        "exit_price": f"{prefix}_exit_price",
        "exit_reason": f"{prefix}_exit_reason",
        "pnl_usd": f"{prefix}_pnl_usd",
    }
    return view.rename(columns=rename_map)


def _values_differ(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return False
    if pd.isna(left) != pd.isna(right):
        return True
    if isinstance(left, pd.Timestamp) or isinstance(right, pd.Timestamp):
        return pd.Timestamp(left) != pd.Timestamp(right)
    if isinstance(left, (int, float, np.integer, np.floating)) or isinstance(right, (int, float, np.integer, np.floating)):
        return not np.isclose(float(left), float(right), atol=1e-9, equal_nan=True)
    return left != right


def _add_divergence_flag(frame: pd.DataFrame, *, baseline_prefix: str, candidate_prefix: str) -> pd.DataFrame:
    out = frame.copy()
    fields = [
        ("entry_time", "entry_time"),
        ("entry_price", "entry_price"),
        ("exit_time", "exit_time"),
        ("exit_price", "exit_price"),
        ("exit_reason", "exit_reason"),
        ("pnl_usd", "pnl_usd"),
    ]
    divergence: list[bool] = []
    for row in out.itertuples(index=False):
        row_dict = row._asdict()
        row_has_divergence = False
        for left_field, right_field in fields:
            if _values_differ(
                row_dict.get(f"{baseline_prefix}_{left_field}"),
                row_dict.get(f"{candidate_prefix}_{right_field}"),
            ):
                row_has_divergence = True
                break
        divergence.append(row_has_divergence)
    out[f"{candidate_prefix}_divergence_flag"] = divergence
    return out


def _build_trade_comparison(
    *,
    baseline_trades: pd.DataFrame,
    hybrid_after_entry_fill: pd.DataFrame,
    hybrid_next_execution_bar: pd.DataFrame,
) -> pd.DataFrame:
    key_cols = ["setup_bar_label_time", "direction"]
    comparison = _scenario_trade_view(baseline_trades, "baseline_1h")
    comparison = comparison.merge(
        _scenario_trade_view(hybrid_after_entry_fill, "hybrid_after_entry_fill"),
        on=key_cols,
        how="outer",
    )
    comparison = comparison.merge(
        _scenario_trade_view(hybrid_next_execution_bar, "hybrid_next_execution_bar"),
        on=key_cols,
        how="outer",
    )
    comparison = _add_divergence_flag(
        comparison,
        baseline_prefix="baseline_1h",
        candidate_prefix="hybrid_after_entry_fill",
    )
    comparison = _add_divergence_flag(
        comparison,
        baseline_prefix="baseline_1h",
        candidate_prefix="hybrid_next_execution_bar",
    )
    return comparison.sort_values(["setup_bar_label_time", "direction"]).reset_index(drop=True)


def _write_report(
    *,
    output_dir: Path,
    symbol: str,
    dataset_path: Path,
    variant: VolumeClimaxPullbackV2Variant,
    signal_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    hybrid_after_entry_fill: pd.DataFrame,
    hybrid_next_execution_bar: pd.DataFrame,
    metrics_comparison: pd.DataFrame,
    trade_comparison: pd.DataFrame,
) -> None:
    divergence_after_fill = int(pd.to_numeric(trade_comparison.get("hybrid_after_entry_fill_divergence_flag"), errors="coerce").fillna(False).sum())
    divergence_next_bar = int(pd.to_numeric(trade_comparison.get("hybrid_next_execution_bar_divergence_flag"), errors="coerce").fillna(False).sum())
    last_minute = pd.Timestamp(minute_df["timestamp"].max()) if not minute_df.empty else pd.NaT
    report_lines = [
        "# Hybrid Execution Validation Report",
        "",
        "## A. Added",
        "- Added an explicit hybrid backtest mode on the V2 engine with `execution_timeframe=\"1min\"`.",
        "- Added `entry_timing` support for `next_execution_bar_open` and `same_timestamp_execution_open`.",
        "- Added `protective_orders_active_from` support for `after_entry_fill` and `next_execution_bar`.",
        "- Added a dedicated validation campaign exporting baseline and hybrid trade/metric comparisons.",
        "",
        "## B. Exact Semantics",
        f"- Signal timeframe remains `1H`; execution timeframe is `1min` for hybrid runs on `{symbol}`.",
        "- `signal_actionable_time` is the close of the setup bar, which equals the timestamp of the signal row built on left-labeled 1H bars.",
        "- `next_execution_bar_open` enters on the first 1min bar strictly after `signal_actionable_time`.",
        "- `same_timestamp_execution_open` enters on the 1min bar at the same timestamp when present, otherwise the next available 1min bar.",
        "- `after_entry_fill` activates stop/target logic on the entry minute bar itself after the fill; `next_execution_bar` activates them from the following 1min bar.",
        "- If stop and target are both touched inside the same 1min bar, the engine uses the pessimistic `stop_first` convention and logs `stop_ambiguous_first_1m`.",
        "- `time_stop_bars` keeps its 1H business meaning and is converted to `entry_time + N * 60 minutes`; the log stores the exact `time_stop_at` timestamp.",
        f"- EOD flat uses the last available RTH minute after the repo's inclusive `09:30 <= timestamp <= 16:00` filter. Current last minute in this run: `{last_minute}`.",
        "- Daily loss remains a post-trade research metric only; no active intraday daily-loss guard was added in this version.",
        "",
        "## C. Files",
        "- Modified: `src/engine/volume_climax_pullback_v2_backtester.py`.",
        "- Added: `src/analytics/volume_climax_pullback_hybrid_execution_validation.py`.",
        "- Added tests covering 1min entry timing, stop/target ambiguity, time stop conversion and EOD flattening.",
        "",
        "## D. Tests",
        "- Unit tests are executed separately in the implementation turn and reported from the pytest output.",
        "",
        "## E. Exports",
        "- `trades_baseline_1h.csv`",
        "- `trades_hybrid_after_entry_fill.csv`",
        "- `trades_hybrid_next_execution_bar.csv`",
        "- `trade_comparison.csv`",
        "- `metrics_comparison.csv`",
        "- `final_report.md`",
        "- `run_metadata.json`",
        "",
        "## F. First Conclusions",
        f"- Baseline trades: `{len(baseline_trades)}`.",
        f"- Hybrid `after_entry_fill` trades: `{len(hybrid_after_entry_fill)}` with `{divergence_after_fill}` divergences vs baseline.",
        f"- Hybrid `next_execution_bar` trades: `{len(hybrid_next_execution_bar)}` with `{divergence_next_bar}` divergences vs baseline.",
        f"- Baseline net PnL: `{safe_float(metrics_comparison.loc[metrics_comparison['scenario'] == 'baseline_1h', 'net_pnl_usd'].squeeze()):.2f}` USD.",
        f"- Hybrid `after_entry_fill` net PnL: `{safe_float(metrics_comparison.loc[metrics_comparison['scenario'] == 'hybrid_after_entry_fill', 'net_pnl_usd'].squeeze()):.2f}` USD.",
        f"- Hybrid `next_execution_bar` net PnL: `{safe_float(metrics_comparison.loc[metrics_comparison['scenario'] == 'hybrid_next_execution_bar', 'net_pnl_usd'].squeeze()):.2f}` USD.",
        "",
        "## Context",
        f"- Dataset: `{dataset_path}`.",
        f"- Variant: `{variant.name}`.",
        f"- Signal rows: `{len(signal_df)}`.",
        f"- RTH minute rows: `{len(minute_df)}`.",
    ]
    (output_dir / "final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def run_validation(
    *,
    symbol: str = DEFAULT_SYMBOL,
    input_path: Path | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    variant_name: str | None = None,
) -> Path:
    dataset_path = Path(input_path) if input_path is not None else latest_path_for_symbol(symbol)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / f"volume_climax_pullback_hybrid_execution_validation_{symbol.lower()}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    variant = _resolve_variant(symbol, variant_name)
    raw_minute_df = load_symbol_data(symbol, input_paths={symbol: dataset_path})
    minute_df = extract_rth(raw_minute_df.copy())
    minute_df["timestamp"] = pd.to_datetime(minute_df["timestamp"], errors="coerce")
    minute_df["session_date"] = minute_df["timestamp"].dt.date

    signal_1h_df = resample_rth_1h(raw_minute_df)
    signal_1h_df["timestamp"] = pd.to_datetime(signal_1h_df["timestamp"], errors="coerce")
    signal_1h_df["session_date"] = signal_1h_df["timestamp"].dt.date

    features = prepare_volume_climax_pullback_v2_features(signal_1h_df)
    signal_df = build_volume_climax_pullback_v2_signal_frame(features, variant)
    execution_model, instrument = build_execution_model_for_profile(symbol=symbol, profile_name="repo_realistic")

    baseline_result = run_volume_climax_pullback_v2_backtest(
        signal_df=signal_df,
        variant=variant,
        execution_model=execution_model,
        instrument=instrument,
        execution_timeframe="1h",
    )
    hybrid_after_fill_result = run_volume_climax_pullback_v2_backtest(
        signal_df=signal_df,
        variant=variant,
        execution_model=execution_model,
        instrument=instrument,
        execution_timeframe="1min",
        minute_df=minute_df,
        entry_timing="next_execution_bar_open",
        protective_orders_active_from="after_entry_fill",
    )
    hybrid_next_bar_result = run_volume_climax_pullback_v2_backtest(
        signal_df=signal_df,
        variant=variant,
        execution_model=execution_model,
        instrument=instrument,
        execution_timeframe="1min",
        minute_df=minute_df,
        entry_timing="next_execution_bar_open",
        protective_orders_active_from="next_execution_bar",
    )

    baseline_trades = baseline_result.trades.copy()
    hybrid_after_entry_fill = hybrid_after_fill_result.trades.copy()
    hybrid_next_execution_bar = hybrid_next_bar_result.trades.copy()

    baseline_trades.to_csv(output_dir / "trades_baseline_1h.csv", index=False)
    hybrid_after_entry_fill.to_csv(output_dir / "trades_hybrid_after_entry_fill.csv", index=False)
    hybrid_next_execution_bar.to_csv(output_dir / "trades_hybrid_next_execution_bar.csv", index=False)

    trade_comparison = _build_trade_comparison(
        baseline_trades=baseline_trades,
        hybrid_after_entry_fill=hybrid_after_entry_fill,
        hybrid_next_execution_bar=hybrid_next_execution_bar,
    )
    trade_comparison.to_csv(output_dir / "trade_comparison.csv", index=False)

    metrics_comparison = pd.DataFrame(
        [
            _metrics_row("baseline_1h", baseline_trades, signal_df),
            _metrics_row("hybrid_after_entry_fill", hybrid_after_entry_fill, signal_df),
            _metrics_row("hybrid_next_execution_bar", hybrid_next_execution_bar, signal_df),
        ]
    )
    metrics_comparison.to_csv(output_dir / "metrics_comparison.csv", index=False)

    run_metadata = {
        "symbol": symbol,
        "dataset_path": str(dataset_path),
        "generated_at": datetime.now().isoformat(),
        "variant_name": variant.name,
        "variant": asdict(variant),
        "execution_scenarios": [
            {
                "label": "baseline_1h",
                "execution_timeframe": "1h",
            },
            {
                "label": "hybrid_after_entry_fill",
                "execution_timeframe": "1min",
                "entry_timing": "next_execution_bar_open",
                "protective_orders_active_from": "after_entry_fill",
            },
            {
                "label": "hybrid_next_execution_bar",
                "execution_timeframe": "1min",
                "entry_timing": "next_execution_bar_open",
                "protective_orders_active_from": "next_execution_bar",
            },
        ],
        "signal_rows": int(len(signal_df)),
        "minute_rows_rth": int(len(minute_df)),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    _write_report(
        output_dir=output_dir,
        symbol=symbol,
        dataset_path=dataset_path,
        variant=variant,
        signal_df=signal_df,
        minute_df=minute_df,
        baseline_trades=baseline_trades,
        hybrid_after_entry_fill=hybrid_after_entry_fill,
        hybrid_next_execution_bar=hybrid_next_execution_bar,
        metrics_comparison=metrics_comparison,
        trade_comparison=trade_comparison,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate 1H baseline vs 1min hybrid execution for volume climax pullback.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Instrument symbol, default MNQ.")
    parser.add_argument("--input-path", default=None, help="Optional explicit parquet/csv dataset path.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root directory.")
    parser.add_argument("--variant-name", default=None, help="Optional explicit V3 variant name.")
    args = parser.parse_args()

    run_dir = run_validation(
        symbol=str(args.symbol).upper(),
        input_path=Path(args.input_path) if args.input_path else None,
        output_root=Path(args.output_root),
        variant_name=args.variant_name,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
