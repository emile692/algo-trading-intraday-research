"""Targeted V2 campaign for the volume climax pullback standalone strategy."""

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
    filter_trades_by_sessions,
    load_symbol_data,
    pf_for_ranking,
    resample_rth_1h,
    safe_float,
    split_sessions,
    summarize_scope,
)
from src.engine.volume_climax_pullback_backtester import run_volume_climax_pullback_backtest
from src.engine.volume_climax_pullback_v2_backtester import run_volume_climax_pullback_v2_backtest
from src.engine.vwap_backtester import build_execution_model_for_profile
from src.strategy.volume_climax_pullback import (
    VolumeClimaxPullbackVariant,
    build_signal_frame,
    prepare_volume_climax_features,
)
from src.strategy.volume_climax_pullback_v2 import (
    VolumeClimaxPullbackV2Variant,
    build_volume_climax_pullback_v2_signal_frame,
    build_volume_climax_pullback_v2_variants,
    prepare_volume_climax_pullback_v2_features,
)

SYMBOLS = ("MNQ", "MES", "M2K", "MGC")
RTH_TIMEFRAME = "1h"

V1_REFERENCE_VARIANT = VolumeClimaxPullbackVariant(
    name="baseline_v1_ref_climax_plus_bar_quality_1h_vq0p95_vl50_mb0p5_ra1p2_sb0_tick_rr1p0_ts2_all_rth",
    family="baseline_v1_ref",
    timeframe="1h",
    volume_quantile=0.95,
    volume_lookback=50,
    min_body_fraction=0.5,
    min_range_atr=1.2,
    stretch_ref=None,
    min_stretch_atr=None,
    wick_fraction=None,
    stop_buffer_mode="0_tick",
    rr_target=1.0,
    time_stop_bars=2,
    session_overlay="all_rth",
)


def _variant_descriptor(variant: VolumeClimaxPullbackV2Variant) -> dict[str, Any]:
    descriptor = asdict(variant)
    descriptor["generation"] = "v2"
    return descriptor


def _baseline_descriptor() -> dict[str, Any]:
    return {
        "name": V1_REFERENCE_VARIANT.name,
        "family": "baseline_v1_ref",
        "timeframe": "1h",
        "volume_quantile": V1_REFERENCE_VARIANT.volume_quantile,
        "volume_lookback": V1_REFERENCE_VARIANT.volume_lookback,
        "min_body_fraction": V1_REFERENCE_VARIANT.min_body_fraction,
        "min_range_atr": V1_REFERENCE_VARIANT.min_range_atr,
        "trend_ema_window": np.nan,
        "ema_slope_threshold": np.nan,
        "atr_percentile_low": np.nan,
        "atr_percentile_high": np.nan,
        "compression_ratio_max": np.nan,
        "entry_mode": "next_open",
        "pullback_fraction": np.nan,
        "confirmation_window": np.nan,
        "exit_mode": "fixed_rr",
        "rr_target": V1_REFERENCE_VARIANT.rr_target,
        "atr_target_multiple": np.nan,
        "time_stop_bars": V1_REFERENCE_VARIANT.time_stop_bars,
        "trailing_atr_multiple": np.nan,
        "session_overlay": "all_rth",
        "generation": "v1_ref",
    }


def _selection_score(row: pd.Series) -> float:
    if str(row.get("generation")) == "v1_ref":
        return np.nan
    if int(row.get("oos_nb_trades", 0)) <= 0:
        return -99.0
    return float(
        1.20 * np.tanh(safe_float(row.get("oos_sharpe")) / 1.5)
        + 0.55 * np.tanh(safe_float(row.get("delta_oos_sharpe_vs_v1")) / 0.75)
        + 0.50 * np.tanh(safe_float(row.get("oos_net_pnl")) / 300.0)
        + 0.35 * np.tanh(safe_float(row.get("oos_expectancy")) / 20.0)
        + 0.25 * np.tanh(safe_float(row.get("delta_oos_expectancy_vs_v1")) / 10.0)
        + 0.25 * np.tanh(pf_for_ranking(row.get("oos_profit_factor")) - 1.0)
        + 0.20 * np.tanh(safe_float(row.get("stability_is_oos_sharpe_ratio")) / 1.0)
        + 0.15 * min(safe_float(row.get("oos_nb_trades")) / 12.0, 1.0)
    )


def _status_label(row: pd.Series) -> str:
    if str(row.get("generation")) == "v1_ref":
        return "baseline_ref"
    if int(row.get("oos_nb_trades", 0)) == 0:
        return "dead"
    if (
        safe_float(row.get("oos_net_pnl")) <= 0.0
        or safe_float(row.get("oos_profit_factor")) <= 1.0
        or safe_float(row.get("oos_sharpe")) <= 0.0
    ):
        return "dead"
    if (
        safe_float(row.get("delta_oos_sharpe_vs_v1")) > 0.0
        and safe_float(row.get("delta_oos_expectancy_vs_v1")) > 0.0
    ):
        return "improved"
    return "mixed"


def _true_improvement(row: pd.Series) -> bool:
    return (
        str(row.get("variant_status")) == "improved"
        and int(row.get("oos_nb_trades", 0)) >= 4
        and safe_float(row.get("stability_is_oos_sharpe_ratio")) >= 0.0
    )


def _build_final_report(
    run_dir: Path,
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    comparison: pd.DataFrame,
    breakdown_by_asset: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> dict[str, Any]:
    non_baseline = summary.loc[summary["generation"] != "v1_ref"].copy()
    improved = non_baseline.loc[non_baseline["is_true_improvement"]].copy()
    alive = non_baseline.loc[non_baseline["variant_status"] != "dead"].copy()

    best_variant = ranking.iloc[0] if not ranking.empty else None
    best_asset_row = breakdown_by_asset.sort_values(
        ["true_improvement_count", "median_delta_oos_sharpe_vs_v1", "best_oos_sharpe"],
        ascending=False,
    ).iloc[0] if not breakdown_by_asset.empty else None

    regime_family = family_summary.loc[family_summary["family"] == "regime_filtered"].copy()
    entry_family = family_summary.loc[family_summary["family"] == "improved_entry"].copy()

    regime_improves_sharpe = bool(
        not regime_family.empty and safe_float(regime_family.iloc[0]["mean_delta_oos_sharpe_vs_v1"]) > 0.0
    )
    delayed_entry_improves_expectancy = bool(
        not entry_family.empty and safe_float(entry_family.iloc[0]["mean_delta_oos_expectancy_vs_v1"]) > 0.0
    )

    positive_assets = int((breakdown_by_asset["true_improvement_count"] > 0).sum()) if not breakdown_by_asset.empty else 0
    if best_variant is None:
        verdict = "non_exploitable"
        verdict_reason = "No V2 variant produced a defendable OOS profile."
    elif positive_assets >= 2 and len(improved) >= 4 and safe_float(best_variant["oos_sharpe"]) > 0.5:
        verdict = "exploitable_sous_conditions"
        verdict_reason = "A few V2 variants improve the V1 benchmark with positive OOS behavior across more than one asset."
    elif len(improved) >= 1 and safe_float(best_variant["oos_net_pnl"]) > 0.0:
        verdict = "amelioration_ciblee_mais_fragile"
        verdict_reason = "There are real improvements, but they are still too concentrated to claim a broad standalone strategy."
    else:
        verdict = "non_exploitable"
        verdict_reason = "The signal still fails to generalize after the targeted V2 work."

    lines = [
        "# Volume Climax Pullback V2 - Final Report",
        "",
        "## Verdict",
        f"- Final verdict: `{verdict}`.",
        f"- Reason: {verdict_reason}",
        f"- Total tested rows (including V1 refs): `{len(summary)}`.",
        f"- V2 live variants: `{len(alive)}`.",
        f"- V2 dead variants removed from OOS rankings: `{int((non_baseline['variant_status'] == 'dead').sum())}`.",
        f"- True V2 improvements vs V1: `{len(improved)}`.",
        "",
        "## V1 vs V2",
    ]

    if best_variant is not None:
        lines.extend(
            [
                f"- Best V2 variant: `{best_variant['variant_name']}` on `{best_variant['symbol']}`.",
                f"- Family / entry / exit: `{best_variant['family']}` / `{best_variant['entry_mode']}` / `{best_variant['exit_mode']}`.",
                f"- OOS Sharpe `{safe_float(best_variant['oos_sharpe']):.3f}` vs V1 `{safe_float(best_variant['baseline_oos_sharpe']):.3f}`.",
                f"- OOS net PnL `{safe_float(best_variant['oos_net_pnl']):.2f}` vs V1 `{safe_float(best_variant['baseline_oos_net_pnl']):.2f}`.",
                f"- OOS expectancy delta vs V1: `{safe_float(best_variant['delta_oos_expectancy_vs_v1']):.2f}`.",
                f"- Stability IS/OOS ratio: `{safe_float(best_variant['stability_is_oos_sharpe_ratio']):.3f}`.",
            ]
        )
    else:
        lines.append("- No V2 variant survived enough to produce a comparison.")

    lines.extend(
        [
            "",
            "## Research Answers",
            f"1. Regime filters improve Sharpe: `{'yes' if regime_improves_sharpe else 'no'}`.",
            f"2. Delayed entries improve expectancy: `{'yes' if delayed_entry_improves_expectancy else 'no'}`.",
            f"3. Standalone defendable strategy: `{'yes, under conditions' if verdict == 'exploitable_sous_conditions' else 'not yet'}`.",
            f"4. Most robust asset: `{best_asset_row['symbol']}`." if best_asset_row is not None else "4. Most robust asset: `n/a`.",
            f"5. IS/OOS stability improved: `{'yes' if not improved.empty and safe_float(improved['delta_stability_vs_v1'].median()) >= 0.0 else 'no'}`.",
            "",
            "## Artifacts",
            "- `summary_variants.csv`",
            "- `ranking_oos.csv`",
            "- `ranking_oos_by_asset.csv`",
            "- `comparison_vs_v1.csv`",
            "- `breakdown_by_asset.csv`",
            "- `family_research_summary.csv`",
        ]
    )

    report_path = run_dir / "final_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "verdict": verdict,
        "reason": verdict_reason,
        "best_variant_name": None if best_variant is None else str(best_variant["variant_name"]),
        "best_asset_symbol": None if best_asset_row is None else str(best_asset_row["symbol"]),
        "regime_filters_improve_sharpe": regime_improves_sharpe,
        "delayed_entries_improve_expectancy": delayed_entry_improves_expectancy,
        "positive_assets_with_true_improvement": positive_assets,
        "true_improvement_count": int(len(improved)),
    }
    (run_dir / "final_verdict.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_campaign(
    output_root: Path,
    *,
    symbols: tuple[str, ...] = SYMBOLS,
    variants: list[VolumeClimaxPullbackV2Variant] | None = None,
    input_paths: dict[str, Path] | None = None,
) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"volume_climax_pullback_v2_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    v2_variants = variants or build_volume_climax_pullback_v2_variants()
    rows: list[dict[str, Any]] = []

    for symbol in symbols:
        raw = load_symbol_data(symbol, input_paths=input_paths)
        bars = resample_rth_1h(raw)
        if bars.empty:
            raise ValueError(f"No RTH 1h bars available for {symbol}.")

        execution_model, instrument = build_execution_model_for_profile(symbol=symbol, profile_name="repo_realistic")
        session_frame = bars.copy()
        session_frame["timestamp"] = pd.to_datetime(session_frame["timestamp"], errors="coerce")
        session_frame["session_date"] = session_frame["timestamp"].dt.date
        is_sessions, oos_sessions = split_sessions(session_frame)

        baseline_features = prepare_volume_climax_features(bars)
        baseline_signal = build_signal_frame(baseline_features, V1_REFERENCE_VARIANT)
        baseline_trades = run_volume_climax_pullback_backtest(
            baseline_signal,
            V1_REFERENCE_VARIANT,
            execution_model,
            instrument,
        ).trades

        baseline_is_signal = baseline_signal.loc[pd.to_datetime(baseline_signal["session_date"]).dt.date.isin(is_sessions)].copy()
        baseline_oos_signal = baseline_signal.loc[pd.to_datetime(baseline_signal["session_date"]).dt.date.isin(oos_sessions)].copy()
        baseline_is_trades = filter_trades_by_sessions(baseline_trades, is_sessions)
        baseline_oos_trades = filter_trades_by_sessions(baseline_trades, oos_sessions)
        baseline_is = summarize_scope(baseline_is_trades, baseline_is_signal, is_sessions)
        baseline_oos = summarize_scope(baseline_oos_trades, baseline_oos_signal, oos_sessions)
        baseline_stability = (
            baseline_oos["sharpe"] / baseline_is["sharpe"] if abs(safe_float(baseline_is["sharpe"])) > 1e-9 else np.nan
        )
        rows.append(
            {
                "symbol": symbol,
                "variant_name": V1_REFERENCE_VARIANT.name,
                **_baseline_descriptor(),
                **{f"is_{key}": value for key, value in baseline_is.items()},
                **{f"oos_{key}": value for key, value in baseline_oos.items()},
                "stability_is_oos_sharpe_ratio": baseline_stability,
            }
        )

        v2_features = prepare_volume_climax_pullback_v2_features(bars)
        for variant in v2_variants:
            signal_df = build_volume_climax_pullback_v2_signal_frame(v2_features, variant)
            trades = run_volume_climax_pullback_v2_backtest(signal_df, variant, execution_model, instrument).trades

            is_signal = signal_df.loc[pd.to_datetime(signal_df["session_date"]).dt.date.isin(is_sessions)].copy()
            oos_signal = signal_df.loc[pd.to_datetime(signal_df["session_date"]).dt.date.isin(oos_sessions)].copy()
            is_trades = filter_trades_by_sessions(trades, is_sessions)
            oos_trades = filter_trades_by_sessions(trades, oos_sessions)
            is_metrics = summarize_scope(is_trades, is_signal, is_sessions)
            oos_metrics = summarize_scope(oos_trades, oos_signal, oos_sessions)
            stability = oos_metrics["sharpe"] / is_metrics["sharpe"] if abs(safe_float(is_metrics["sharpe"])) > 1e-9 else np.nan

            rows.append(
                {
                    "symbol": symbol,
                    "variant_name": variant.name,
                    **_variant_descriptor(variant),
                    **{f"is_{key}": value for key, value in is_metrics.items()},
                    **{f"oos_{key}": value for key, value in oos_metrics.items()},
                    "stability_is_oos_sharpe_ratio": stability,
                }
            )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["symbol", "generation", "family", "variant_name"]).reset_index(drop=True)

    baseline_cols = [
        "symbol",
        "oos_net_pnl",
        "oos_profit_factor",
        "oos_sharpe",
        "oos_expectancy",
        "oos_nb_trades",
        "stability_is_oos_sharpe_ratio",
    ]
    baseline_view = summary.loc[summary["generation"] == "v1_ref", baseline_cols].rename(
        columns={
            "oos_net_pnl": "baseline_oos_net_pnl",
            "oos_profit_factor": "baseline_oos_profit_factor",
            "oos_sharpe": "baseline_oos_sharpe",
            "oos_expectancy": "baseline_oos_expectancy",
            "oos_nb_trades": "baseline_oos_nb_trades",
            "stability_is_oos_sharpe_ratio": "baseline_stability_is_oos_sharpe_ratio",
        }
    )
    summary = summary.merge(baseline_view, on="symbol", how="left")

    summary["delta_oos_net_pnl_vs_v1"] = pd.to_numeric(summary["oos_net_pnl"], errors="coerce") - pd.to_numeric(summary["baseline_oos_net_pnl"], errors="coerce")
    summary["delta_oos_profit_factor_vs_v1"] = pd.to_numeric(summary["oos_profit_factor"], errors="coerce") - pd.to_numeric(summary["baseline_oos_profit_factor"], errors="coerce")
    summary["delta_oos_sharpe_vs_v1"] = pd.to_numeric(summary["oos_sharpe"], errors="coerce") - pd.to_numeric(summary["baseline_oos_sharpe"], errors="coerce")
    summary["delta_oos_expectancy_vs_v1"] = pd.to_numeric(summary["oos_expectancy"], errors="coerce") - pd.to_numeric(summary["baseline_oos_expectancy"], errors="coerce")
    summary["delta_stability_vs_v1"] = pd.to_numeric(summary["stability_is_oos_sharpe_ratio"], errors="coerce") - pd.to_numeric(summary["baseline_stability_is_oos_sharpe_ratio"], errors="coerce")
    summary["selection_score"] = summary.apply(_selection_score, axis=1)
    summary["variant_status"] = summary.apply(_status_label, axis=1)
    summary["is_true_improvement"] = summary.apply(_true_improvement, axis=1)

    summary.to_csv(run_dir / "summary_variants.csv", index=False)

    non_baseline = summary.loc[summary["generation"] != "v1_ref"].copy()
    ranking = non_baseline.loc[non_baseline["variant_status"] != "dead"].copy()
    ranking = ranking.sort_values(
        ["selection_score", "oos_sharpe", "delta_oos_sharpe_vs_v1", "oos_net_pnl"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    ranking.to_csv(run_dir / "ranking_oos.csv", index=False)
    ranking.sort_values(["symbol", "selection_score", "oos_sharpe"], ascending=[True, False, False]).to_csv(
        run_dir / "ranking_oos_by_asset.csv",
        index=False,
    )

    comparison = non_baseline.sort_values(
        ["is_true_improvement", "delta_oos_sharpe_vs_v1", "delta_oos_expectancy_vs_v1", "selection_score"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    comparison.to_csv(run_dir / "comparison_vs_v1.csv", index=False)

    breakdown_rows: list[dict[str, Any]] = []
    for symbol, frame in non_baseline.groupby("symbol", sort=True):
        live = frame.loc[frame["variant_status"] != "dead"].copy()
        ordered = live.sort_values(["selection_score", "oos_sharpe", "oos_net_pnl"], ascending=False) if not live.empty else live
        best = ordered.iloc[0] if not ordered.empty else None
        breakdown_rows.append(
            {
                "symbol": symbol,
                "total_v2_variants": int(len(frame)),
                "live_variant_count": int((frame["variant_status"] != "dead").sum()),
                "dead_variant_count": int((frame["variant_status"] == "dead").sum()),
                "true_improvement_count": int(frame["is_true_improvement"].sum()),
                "median_delta_oos_sharpe_vs_v1": float(pd.to_numeric(frame["delta_oos_sharpe_vs_v1"], errors="coerce").median()),
                "median_delta_oos_expectancy_vs_v1": float(pd.to_numeric(frame["delta_oos_expectancy_vs_v1"], errors="coerce").median()),
                "best_variant_name": None if best is None else str(best["variant_name"]),
                "best_family": None if best is None else str(best["family"]),
                "best_entry_mode": None if best is None else str(best["entry_mode"]),
                "best_exit_mode": None if best is None else str(best["exit_mode"]),
                "best_oos_sharpe": 0.0 if best is None else float(best["oos_sharpe"]),
                "best_oos_net_pnl": 0.0 if best is None else float(best["oos_net_pnl"]),
                "best_delta_oos_sharpe_vs_v1": 0.0 if best is None else float(best["delta_oos_sharpe_vs_v1"]),
            }
        )
    breakdown_by_asset = pd.DataFrame(breakdown_rows).sort_values(
        ["true_improvement_count", "median_delta_oos_sharpe_vs_v1", "best_oos_sharpe"],
        ascending=[False, False, False],
    )
    breakdown_by_asset.to_csv(run_dir / "breakdown_by_asset.csv", index=False)

    family_summary = (
        non_baseline.groupby("family", as_index=False)
        .agg(
            variants=("variant_name", "count"),
            live_variants=("variant_status", lambda s: int((pd.Series(s) != "dead").sum())),
            true_improvement_count=("is_true_improvement", "sum"),
            mean_oos_sharpe=("oos_sharpe", "mean"),
            mean_delta_oos_sharpe_vs_v1=("delta_oos_sharpe_vs_v1", "mean"),
            mean_delta_oos_expectancy_vs_v1=("delta_oos_expectancy_vs_v1", "mean"),
            mean_stability=("stability_is_oos_sharpe_ratio", "mean"),
        )
        .sort_values(["true_improvement_count", "mean_delta_oos_sharpe_vs_v1"], ascending=[False, False])
        .reset_index(drop=True)
    )
    family_summary.to_csv(run_dir / "family_research_summary.csv", index=False)

    verdict = _build_final_report(run_dir, summary, ranking, comparison, breakdown_by_asset, family_summary)
    metadata = {
        "run_dir": str(run_dir),
        "symbols": list(symbols),
        "variant_count_v2": int(len(v2_variants)),
        "generated_at_utc": datetime.utcnow().isoformat(),
        "verdict": verdict,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the targeted V2 volume climax pullback campaign.")
    parser.add_argument("--output-dir", default=None, help="Legacy output root flag.")
    parser.add_argument("--output-root", default=None, help="Output root directory.")
    args = parser.parse_args()
    output_root = Path(args.output_root or args.output_dir or "data/exports")
    out = run_campaign(output_root)
    print(out)


if __name__ == "__main__":
    main()
