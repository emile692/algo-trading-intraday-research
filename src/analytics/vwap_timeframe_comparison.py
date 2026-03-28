"""Strict leak-free 1m vs 5m comparison for a compact VWAP variant subset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.vwap_reranking import (
    PAPER_BASELINE_NAME,
    REALISTIC_BASELINE_NAME,
    DEFAULT_SPLIT_FRACTIONS,
    RerankingSpec,
    VariantEvaluation,
    _build_concentration_summary,
    _build_prop_summary,
    _build_stress_summary,
    _evaluate_variant,
    _export_baseline_reference,
    _final_verdict_payload,
    _json_dump,
    _merge_reranking_tables,
    _notebook_cell,
    _split_oos_rows,
    _what_changes_vs_baseline,
    _write_summary_outputs,
)
from src.config.paths import EXPORTS_DIR, NOTEBOOKS_DIR, ensure_directories
from src.config.vwap_campaign import (
    DEFAULT_IS_FRACTION,
    DEFAULT_PAPER_TIME_EXIT,
    DEFAULT_RTH_SESSION_END,
    DEFAULT_RTH_SESSION_START,
    PropFirmConstraintConfig,
    VWAPVariantConfig,
    adapt_vwap_variant_to_timeframe,
    build_default_prop_constraints,
    build_default_vwap_timeframe_comparison_variants,
    resolve_default_vwap_dataset,
    resolve_vwap_variant,
)
from src.data.cleaning import clean_ohlcv
from src.data.loader import load_ohlcv_file
from src.data.resampling import build_resampled_output_path, resample_ohlcv
from src.strategy.vwap import prepare_vwap_feature_frame


TIMEFRAME_COMPARISON_MODES = ("full", "notebook")
VWAP_PV_TYPICAL_COL = "vwap_pv_typical"
VWAP_PV_CLOSE_COL = "vwap_pv_close"


@dataclass(frozen=True)
class TimeframeDefinition:
    """Explicit bar interval used in the strict comparison."""

    label: str
    bar_minutes: int
    resample_rule: str | None = None


DEFAULT_TIMEFRAMES = (
    TimeframeDefinition(label="1m", bar_minutes=1, resample_rule=None),
    TimeframeDefinition(label="5m", bar_minutes=5, resample_rule="5min"),
)


@dataclass(frozen=True)
class TimeframeComparisonSpec:
    """Top-level settings for the compact 1m vs 5m comparison."""

    dataset_path: Path
    variant_names: tuple[str, ...]
    is_fraction: float = DEFAULT_IS_FRACTION
    split_fractions: tuple[float, ...] = DEFAULT_SPLIT_FRACTIONS
    session_start: str = DEFAULT_RTH_SESSION_START
    session_end: str = DEFAULT_RTH_SESSION_END
    paper_time_exit: str = DEFAULT_PAPER_TIME_EXIT
    prop_constraints: PropFirmConstraintConfig = build_default_prop_constraints()
    timeframes: tuple[TimeframeDefinition, ...] = DEFAULT_TIMEFRAMES
    vwap_price_mode: str = "typical"
    base_bar_minutes: int = 1
    min_trade_retention_ratio: float = 0.35


@dataclass
class TimeframeRunResult:
    """Collected outputs for one timeframe run."""

    timeframe: TimeframeDefinition
    dataset_path: Path
    variants: dict[str, VWAPVariantConfig]
    stress_df: pd.DataFrame
    split_details_df: pd.DataFrame
    split_summary_df: pd.DataFrame
    prop_df: pd.DataFrame
    concentration_df: pd.DataFrame
    reranking_df: pd.DataFrame
    verdict: dict[str, Any]


def build_default_timeframe_comparison_spec(dataset_path: Path | None = None) -> TimeframeComparisonSpec:
    """Return the compact timeframe comparison spec."""
    resolved = dataset_path or resolve_default_vwap_dataset("MNQ")
    variants = build_default_vwap_timeframe_comparison_variants()
    return TimeframeComparisonSpec(
        dataset_path=resolved,
        variant_names=tuple(variant.name for variant in variants),
        prop_constraints=build_default_prop_constraints(),
    )


def _variant_rows_for_timeframe(
    timeframe: TimeframeDefinition,
    variants: dict[str, VWAPVariantConfig],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, strategy_id in enumerate(variants, start=1):
        variant = variants[strategy_id]
        rows.append(
            {
                "timeframe": timeframe.label,
                "bar_minutes": timeframe.bar_minutes,
                "display_order": order,
                "strategy_id": strategy_id,
                "family": variant.family,
                "mode": variant.mode,
                "execution_profile": variant.execution_profile,
                "quantity_mode": variant.quantity_mode,
                "slope_lookback": variant.slope_lookback,
                "atr_period": variant.atr_period,
                "compression_length": variant.compression_length,
                "pullback_lookback": variant.pullback_lookback,
                "atr_buffer": variant.atr_buffer,
                "stop_buffer": variant.stop_buffer if variant.stop_buffer is not None else variant.atr_buffer,
                "max_trades_per_day": variant.max_trades_per_day,
                "time_windows": "|".join(f"{window.start}->{window.end}" for window in variant.time_windows) if variant.time_windows else "full_rth",
                "what_changes_vs_baseline": _what_changes_vs_baseline(variant),
            }
        )
    return rows


def _enrich_with_price_volume(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    volume = out["volume"].fillna(0.0)
    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    out[VWAP_PV_TYPICAL_COL] = typical * volume
    out[VWAP_PV_CLOSE_COL] = out["close"] * volume
    return out


def _price_volume_col_for_mode(price_mode: str) -> str:
    if price_mode == "typical":
        return VWAP_PV_TYPICAL_COL
    if price_mode == "close":
        return VWAP_PV_CLOSE_COL
    raise ValueError("price_mode must be 'close' or 'typical'.")


def _prepare_timeframe_input_frame(
    clean_1m: pd.DataFrame,
    spec: TimeframeComparisonSpec,
    timeframe: TimeframeDefinition,
    output_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    if timeframe.bar_minutes == spec.base_bar_minutes:
        return clean_1m.copy(), spec.dataset_path

    if timeframe.resample_rule is None:
        raise ValueError(f"Missing resample rule for timeframe '{timeframe.label}'.")

    enriched = _enrich_with_price_volume(clean_1m)
    resampled = resample_ohlcv(
        enriched,
        rule=timeframe.resample_rule,
        aggregation_overrides={
            VWAP_PV_TYPICAL_COL: "sum",
            VWAP_PV_CLOSE_COL: "sum",
        },
    )
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = build_resampled_output_path(spec.dataset_path, rule=timeframe.resample_rule, output_dir=data_dir)
    parquet_df = resampled.set_index("timestamp")
    parquet_df.index.name = "timestamp"
    parquet_df.to_parquet(dataset_path)
    return resampled, dataset_path


def _variants_for_timeframe(
    spec: TimeframeComparisonSpec,
    timeframe: TimeframeDefinition,
) -> dict[str, VWAPVariantConfig]:
    variants: dict[str, VWAPVariantConfig] = {}
    for strategy_id in spec.variant_names:
        base_variant = resolve_vwap_variant(strategy_id)
        variants[strategy_id] = adapt_vwap_variant_to_timeframe(
            base_variant,
            bar_minutes=timeframe.bar_minutes,
            base_bar_minutes=spec.base_bar_minutes,
        )
    return variants


def _prepare_feature_frame_for_timeframe(
    input_df: pd.DataFrame,
    spec: TimeframeComparisonSpec,
    variants: dict[str, VWAPVariantConfig],
    timeframe: TimeframeDefinition,
) -> pd.DataFrame:
    atr_windows = sorted({int(variant.atr_period) for variant in variants.values() if int(variant.atr_period) > 0})
    price_volume_col = None
    if timeframe.bar_minutes > spec.base_bar_minutes:
        price_volume_col = _price_volume_col_for_mode(spec.vwap_price_mode)
    return prepare_vwap_feature_frame(
        input_df,
        session_start=spec.session_start,
        session_end=spec.session_end,
        atr_windows=atr_windows,
        vwap_price_mode=spec.vwap_price_mode,
        vwap_price_volume_col=price_volume_col,
    )


def _run_single_timeframe(
    spec: TimeframeComparisonSpec,
    timeframe: TimeframeDefinition,
    clean_1m: pd.DataFrame,
    output_dir: Path,
) -> TimeframeRunResult:
    timeframe_dir = output_dir / timeframe.label
    timeframe_dir.mkdir(parents=True, exist_ok=True)

    input_df, dataset_path = _prepare_timeframe_input_frame(clean_1m, spec, timeframe, output_dir)
    variants = _variants_for_timeframe(spec, timeframe)
    feature_df = _prepare_feature_frame_for_timeframe(input_df, spec, variants, timeframe)
    reranking_spec = RerankingSpec(
        dataset_path=dataset_path,
        variant_names=spec.variant_names,
        is_fraction=spec.is_fraction,
        split_fractions=spec.split_fractions,
        session_start=spec.session_start,
        session_end=spec.session_end,
        paper_time_exit=spec.paper_time_exit,
        prop_constraints=spec.prop_constraints,
    )

    cache: dict[str, VariantEvaluation] = {}
    evaluations = {
        strategy_id: _evaluate_variant(feature_df, reranking_spec, variant, cache)
        for strategy_id, variant in variants.items()
    }
    stress_df, stress_map = _build_stress_summary(evaluations, reranking_spec)

    split_detail_frames: list[pd.DataFrame] = []
    split_summary_frames: list[pd.DataFrame] = []
    for evaluation in evaluations.values():
        detail_df, aggregate_df = _split_oos_rows(evaluation, reranking_spec)
        split_detail_frames.append(detail_df)
        split_summary_frames.append(aggregate_df)
    split_details_df = pd.concat(split_detail_frames, ignore_index=True) if split_detail_frames else pd.DataFrame()
    split_summary_df = pd.concat(split_summary_frames, ignore_index=True) if split_summary_frames else pd.DataFrame()

    prop_df = _build_prop_summary(evaluations, reranking_spec)
    concentration_df = _build_concentration_summary(evaluations, reranking_spec)
    reranking_df = _merge_reranking_tables(
        spec=reranking_spec,
        evaluations=evaluations,
        stress_df=stress_df,
        split_aggregate_df=split_summary_df,
        prop_df=prop_df,
        concentration_df=concentration_df,
    )
    verdict = _final_verdict_payload(reranking_df)

    baseline_eval = evaluations[PAPER_BASELINE_NAME]
    _export_baseline_reference(timeframe_dir, reranking_spec, baseline_eval, stress_map[PAPER_BASELINE_NAME])
    stress_df.to_csv(timeframe_dir / "stress_test_summary.csv", index=False)
    split_details_df.to_csv(timeframe_dir / "split_details.csv", index=False)
    split_summary_df.to_csv(timeframe_dir / "split_summary.csv", index=False)
    prop_df.to_csv(timeframe_dir / "prop_summary.csv", index=False)
    concentration_df.to_csv(timeframe_dir / "concentration_summary.csv", index=False)
    _write_summary_outputs(timeframe_dir, reranking_df, verdict)
    _json_dump(
        timeframe_dir / "run_metadata.json",
        {
            "run_timestamp": datetime.now().isoformat(),
            "timeframe": timeframe.label,
            "bar_minutes": timeframe.bar_minutes,
            "source_dataset_1m": spec.dataset_path,
            "dataset_path": dataset_path,
            "resample_rule": timeframe.resample_rule,
            "vwap_price_mode": spec.vwap_price_mode,
            "vwap_price_volume_col": None
            if timeframe.bar_minutes == spec.base_bar_minutes
            else _price_volume_col_for_mode(spec.vwap_price_mode),
            "variant_names": list(spec.variant_names),
        },
    )

    return TimeframeRunResult(
        timeframe=timeframe,
        dataset_path=dataset_path,
        variants=variants,
        stress_df=stress_df,
        split_details_df=split_details_df,
        split_summary_df=split_summary_df,
        prop_df=prop_df,
        concentration_df=concentration_df,
        reranking_df=reranking_df,
        verdict=verdict,
    )


def _merge_timeframe_frames(runs: list[TimeframeRunResult], attr: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs:
        frame = getattr(run, attr)
        if frame is None:
            continue
        tagged = frame.copy()
        tagged.insert(0, "timeframe", run.timeframe.label)
        tagged.insert(1, "bar_minutes", run.timeframe.bar_minutes)
        frames.append(tagged)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_delta_summary(
    comparison_summary: pd.DataFrame,
    spec: TimeframeComparisonSpec,
) -> pd.DataFrame:
    if comparison_summary.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    pivot = comparison_summary.set_index(["strategy_id", "timeframe"])
    for strategy_id in comparison_summary["strategy_id"].drop_duplicates():
        if (strategy_id, "1m") not in pivot.index or (strategy_id, "5m") not in pivot.index:
            continue
        row_1m = pivot.loc[(strategy_id, "1m")]
        row_5m = pivot.loc[(strategy_id, "5m")]
        trades_1m = float(row_1m.get("oos_total_trades", 0.0) or 0.0)
        trades_5m = float(row_5m.get("oos_total_trades", 0.0) or 0.0)
        trade_retention = np.nan if trades_1m <= 0 else trades_5m / trades_1m

        criteria = {
            "oos_and_costs_ok_5m": bool(row_5m["oos_net_pnl"] > 0 and row_5m["pf_slip_x2"] > 1.0),
            "split_stability_improves": bool(row_5m["positive_oos_splits"] > row_1m["positive_oos_splits"]),
            "prop_metrics_improve": bool(
                row_5m["challenge_success_rate_standard"] >= row_1m["challenge_success_rate_standard"]
                and row_5m["daily_loss_limit_breach_freq"] <= row_1m["daily_loss_limit_breach_freq"]
                and row_5m["trailing_drawdown_breach_freq"] <= row_1m["trailing_drawdown_breach_freq"]
            ),
            "concentration_improves": bool(
                row_5m["top_5_day_contribution_pct"] < row_1m["top_5_day_contribution_pct"]
                and row_5m["pnl_excluding_top_5_days"] >= row_1m["pnl_excluding_top_5_days"]
            ),
            "trade_count_retained": bool(np.isfinite(trade_retention) and trade_retention >= spec.min_trade_retention_ratio),
        }
        improvement_score = int(sum(int(value) for value in criteria.values()))
        credible_improvement = bool(all(criteria.values()))

        if credible_improvement:
            bucket = "amelioration credible"
            comment = "Le 5m ameliore simultanement la stabilite splits, les metriques prop, la concentration, et reste exploitable apres slippage x2."
        elif improvement_score >= 3:
            bucket = "amelioration partielle mais insuffisante"
            comment = "Le 5m lisse certains degats, mais pas assez pour renverser le verdict global."
        else:
            bucket = "pas d'amelioration exploitable"
            comment = "Le passage en 5m ne produit pas d'amelioration robuste suffisante."

        rows.append(
            {
                "strategy_id": strategy_id,
                "role": row_1m["role"],
                "oos_net_pnl_1m": row_1m["oos_net_pnl"],
                "oos_net_pnl_5m": row_5m["oos_net_pnl"],
                "oos_profit_factor_1m": row_1m["oos_profit_factor"],
                "oos_profit_factor_5m": row_5m["oos_profit_factor"],
                "oos_sharpe_ratio_1m": row_1m["oos_sharpe_ratio"],
                "oos_sharpe_ratio_5m": row_5m["oos_sharpe_ratio"],
                "oos_max_drawdown_1m": row_1m["oos_max_drawdown"],
                "oos_max_drawdown_5m": row_5m["oos_max_drawdown"],
                "oos_total_trades_1m": row_1m["oos_total_trades"],
                "oos_total_trades_5m": row_5m["oos_total_trades"],
                "expectancy_per_trade_1m": row_1m["oos_expectancy_per_trade"],
                "expectancy_per_trade_5m": row_5m["oos_expectancy_per_trade"],
                "pnl_slip_x2_1m": row_1m["pnl_slip_x2"],
                "pnl_slip_x2_5m": row_5m["pnl_slip_x2"],
                "positive_oos_splits_1m": row_1m["positive_oos_splits"],
                "positive_oos_splits_5m": row_5m["positive_oos_splits"],
                "challenge_success_rate_standard_1m": row_1m["challenge_success_rate_standard"],
                "challenge_success_rate_standard_5m": row_5m["challenge_success_rate_standard"],
                "top_5_day_contribution_pct_1m": row_1m["top_5_day_contribution_pct"],
                "top_5_day_contribution_pct_5m": row_5m["top_5_day_contribution_pct"],
                "pnl_excluding_top_5_days_1m": row_1m["pnl_excluding_top_5_days"],
                "pnl_excluding_top_5_days_5m": row_5m["pnl_excluding_top_5_days"],
                "trade_retention_ratio_5m_vs_1m": trade_retention,
                "improvement_score": improvement_score,
                "credible_robustness_improvement": credible_improvement,
                "comparison_bucket": bucket,
                "comparison_comment": comment,
                **criteria,
            }
        )
    return pd.DataFrame(rows)


def _paper_baseline_comparison(comparison_summary: pd.DataFrame) -> pd.DataFrame:
    if comparison_summary.empty:
        return pd.DataFrame()
    mask = comparison_summary["strategy_id"] == PAPER_BASELINE_NAME
    columns = [
        "timeframe",
        "bar_minutes",
        "strategy_id",
        "oos_net_pnl",
        "oos_profit_factor",
        "oos_sharpe_ratio",
        "oos_max_drawdown",
        "oos_total_trades",
        "pnl_slip_x2",
        "pf_slip_x2",
        "positive_oos_splits",
        "challenge_success_rate_standard",
        "top_5_day_contribution_pct",
        "final_bucket",
    ]
    return comparison_summary.loc[mask, columns].reset_index(drop=True)


def _build_global_verdict(
    comparison_summary: pd.DataFrame,
    delta_df: pd.DataFrame,
    runs: list[TimeframeRunResult],
) -> dict[str, Any]:
    credible_candidates = delta_df.loc[
        (delta_df["role"] == "candidate") & (delta_df["credible_robustness_improvement"] == True)  # noqa: E712
    ].copy()
    if credible_candidates.empty:
        global_verdict = "5 minutes ne change pas le verdict, famille VWAP a archiver"
        top_candidate = None
    else:
        credible_candidates = credible_candidates.sort_values(
            ["improvement_score", "oos_profit_factor_5m", "oos_sharpe_ratio_5m"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        top_candidate = str(credible_candidates.iloc[0]["strategy_id"])
        global_verdict = "5 minutes ameliore reellement la robustesse"

    timeframe_verdicts = {run.timeframe.label: run.verdict["global_verdict"] for run in runs}
    return {
        "global_verdict": global_verdict,
        "top_candidate_if_any": top_candidate,
        "credible_candidates": [] if credible_candidates.empty else credible_candidates["strategy_id"].tolist(),
        "timeframe_verdicts": timeframe_verdicts,
        "methodology_notes": [
            "Semantique leak-free conservee: signal sur information disponible au bar t-1, execution au next open du timeframe teste.",
            "Le 5m est derive du 1m par resampling left-closed/left-labeled, sans usage d'information future.",
            "Le VWAP 5m reutilise la somme des price*volume sous-jacents 1m pour eviter une distortion artificielle du calcul de VWAP.",
            "Seuls les parametres exprimes en nombre de barres ont ete rescalés minimalement pour conserver des horizons temporels comparables, sans retuning des seuils, buffers, fenetres horaires, ni couts.",
        ],
        "variant_comparisons": delta_df.to_dict(orient="records"),
        "baseline_rows": comparison_summary.loc[
            comparison_summary["role"].isin(["paper_baseline_reference", "realistic_baseline_reference"])
        ].to_dict(orient="records"),
    }


def _write_top_level_outputs(
    output_dir: Path,
    spec_rows: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    delta_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    split_details_df: pd.DataFrame,
    split_summary_df: pd.DataFrame,
    prop_df: pd.DataFrame,
    concentration_df: pd.DataFrame,
    verdict: dict[str, Any],
) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}

    spec_csv = output_dir / "variant_timeframe_specs.csv"
    comparison_csv = output_dir / "comparison_summary.csv"
    delta_csv = output_dir / "comparison_delta.csv"
    baseline_csv = output_dir / "paper_baseline_reference_comparison.csv"
    stress_csv = output_dir / "stress_test_summary.csv"
    split_details_csv = output_dir / "split_details.csv"
    split_summary_csv = output_dir / "split_summary.csv"
    prop_csv = output_dir / "prop_summary.csv"
    concentration_csv = output_dir / "concentration_summary.csv"
    summary_md = output_dir / "comparison_summary.md"
    verdict_json = output_dir / "final_verdict.json"

    spec_rows.to_csv(spec_csv, index=False)
    comparison_summary.to_csv(comparison_csv, index=False)
    delta_df.to_csv(delta_csv, index=False)
    _paper_baseline_comparison(comparison_summary).to_csv(baseline_csv, index=False)
    stress_df.to_csv(stress_csv, index=False)
    split_details_df.to_csv(split_details_csv, index=False)
    split_summary_df.to_csv(split_summary_csv, index=False)
    prop_df.to_csv(prop_csv, index=False)
    concentration_df.to_csv(concentration_csv, index=False)
    _json_dump(verdict_json, verdict)

    summary_md.write_text(
        "\n".join(
            [
                "# VWAP 1m vs 5m Comparison",
                "",
                f"- Global verdict: `{verdict['global_verdict']}`",
                f"- 1m verdict: `{verdict['timeframe_verdicts'].get('1m', 'n/a')}`",
                f"- 5m verdict: `{verdict['timeframe_verdicts'].get('5m', 'n/a')}`",
                f"- Credible candidates in 5m: `{', '.join(verdict['credible_candidates']) if verdict['credible_candidates'] else 'none'}`",
                "",
                "## Delta Table",
                "",
                "```text",
                delta_df.to_string(index=False),
                "```",
                "",
                "## Comparison Summary",
                "",
                "```text",
                comparison_summary[
                    [
                        "timeframe",
                        "strategy_id",
                        "role",
                        "oos_net_pnl",
                        "oos_profit_factor",
                        "oos_sharpe_ratio",
                        "oos_max_drawdown",
                        "oos_total_trades",
                        "pnl_slip_x2",
                        "positive_oos_splits",
                        "challenge_success_rate_standard",
                        "top_5_day_contribution_pct",
                        "final_bucket",
                    ]
                ].to_string(index=False),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    artifacts.update(
        {
            "variant_timeframe_specs_csv": spec_csv,
            "comparison_summary_csv": comparison_csv,
            "comparison_delta_csv": delta_csv,
            "paper_baseline_reference_comparison_csv": baseline_csv,
            "stress_test_summary_csv": stress_csv,
            "split_details_csv": split_details_csv,
            "split_summary_csv": split_summary_csv,
            "prop_summary_csv": prop_csv,
            "concentration_summary_csv": concentration_csv,
            "comparison_summary_md": summary_md,
            "final_verdict_json": verdict_json,
        }
    )
    return artifacts


def generate_timeframe_comparison_notebook(notebook_path: Path, output_dir: Path) -> Path:
    """Render the compact 1m vs 5m comparison notebook."""
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_dir = output_dir.resolve()
    setup_code = """from pathlib import Path
import json
import sys
from IPython.display import Markdown, display
import pandas as pd

root = Path.cwd().resolve()
while root != root.parent and not (root / "pyproject.toml").exists():
    root = root.parent
if not (root / "pyproject.toml").exists():
    raise RuntimeError("Could not locate repository root.")
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 160)
"""
    config_code = f"OUTPUT_DIR = Path(r\"{str(resolved_output_dir)}\")\nprint(OUTPUT_DIR)\n"
    methodology_code = """verdict = json.loads((OUTPUT_DIR / "final_verdict.json").read_text(encoding="utf-8"))
display(Markdown("\\n".join([f"- {note}" for note in verdict["methodology_notes"]])))
"""
    specs_code = """display(pd.read_csv(OUTPUT_DIR / "variant_timeframe_specs.csv"))"""
    comparison_code = """display(pd.read_csv(OUTPUT_DIR / "comparison_summary.csv"))
display(pd.read_csv(OUTPUT_DIR / "comparison_delta.csv"))
"""
    baseline_code = """display(pd.read_csv(OUTPUT_DIR / "paper_baseline_reference_comparison.csv"))"""
    stress_code = """display(pd.read_csv(OUTPUT_DIR / "stress_test_summary.csv"))"""
    split_code = """display(pd.read_csv(OUTPUT_DIR / "split_summary.csv"))"""
    prop_code = """display(pd.read_csv(OUTPUT_DIR / "prop_summary.csv"))"""
    concentration_code = """display(pd.read_csv(OUTPUT_DIR / "concentration_summary.csv"))"""
    verdict_code = """verdict = json.loads((OUTPUT_DIR / "final_verdict.json").read_text(encoding="utf-8"))
verdict
display(Markdown((OUTPUT_DIR / "comparison_summary.md").read_text(encoding="utf-8")))
"""
    notebook = {
        "cells": [
            _notebook_cell("markdown", "# VWAP 1m vs 5m Leak-Free Comparison"),
            _notebook_cell("code", setup_code),
            _notebook_cell("code", config_code),
            _notebook_cell("markdown", "## 1) Methodology"),
            _notebook_cell("code", methodology_code),
            _notebook_cell("markdown", "## 2) Timeframe-Specific Variant Specs"),
            _notebook_cell("code", specs_code),
            _notebook_cell("markdown", "## 3) Headline Comparison"),
            _notebook_cell("code", comparison_code),
            _notebook_cell("markdown", "## 4) Paper Baseline Reference"),
            _notebook_cell("code", baseline_code),
            _notebook_cell("markdown", "## 5) Stress x2"),
            _notebook_cell("code", stress_code),
            _notebook_cell("markdown", "## 6) Multi-Split"),
            _notebook_cell("code", split_code),
            _notebook_cell("markdown", "## 7) Prop Metrics"),
            _notebook_cell("code", prop_code),
            _notebook_cell("markdown", "## 8) Concentration"),
            _notebook_cell("code", concentration_code),
            _notebook_cell("markdown", "## 9) Final Verdict"),
            _notebook_cell("code", verdict_code),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return notebook_path


def run_vwap_timeframe_comparison(
    spec: TimeframeComparisonSpec,
    output_dir: Path,
    mode: str = "full",
    notebook_path: Path | None = None,
) -> dict[str, Path]:
    """Execute the strict 1m vs 5m comparison."""
    if mode not in TIMEFRAME_COMPARISON_MODES:
        raise ValueError(f"Unsupported timeframe comparison mode '{mode}'.")

    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_ohlcv_file(spec.dataset_path)
    clean_1m = clean_ohlcv(raw)

    runs = [_run_single_timeframe(spec, timeframe, clean_1m, output_dir) for timeframe in spec.timeframes]

    variant_rows = []
    for run in runs:
        variant_rows.extend(_variant_rows_for_timeframe(run.timeframe, run.variants))
    spec_rows_df = pd.DataFrame(variant_rows)
    comparison_summary = _merge_timeframe_frames(runs, "reranking_df")
    stress_df = _merge_timeframe_frames(runs, "stress_df")
    split_details_df = _merge_timeframe_frames(runs, "split_details_df")
    split_summary_df = _merge_timeframe_frames(runs, "split_summary_df")
    prop_df = _merge_timeframe_frames(runs, "prop_df")
    concentration_df = _merge_timeframe_frames(runs, "concentration_df")
    delta_df = _build_delta_summary(comparison_summary, spec)
    verdict = _build_global_verdict(comparison_summary, delta_df, runs)

    artifacts = {"output_dir": output_dir}
    artifacts.update(
        _write_top_level_outputs(
            output_dir=output_dir,
            spec_rows=spec_rows_df,
            comparison_summary=comparison_summary,
            delta_df=delta_df,
            stress_df=stress_df,
            split_details_df=split_details_df,
            split_summary_df=split_summary_df,
            prop_df=prop_df,
            concentration_df=concentration_df,
            verdict=verdict,
        )
    )

    if mode == "notebook" and notebook_path is not None:
        artifacts["validation_notebook"] = generate_timeframe_comparison_notebook(
            notebook_path=notebook_path,
            output_dir=output_dir,
        )

    _json_dump(
        output_dir / "run_metadata.json",
        {
            "run_timestamp": datetime.now().isoformat(),
            "mode": mode,
            "source_dataset_1m": spec.dataset_path,
            "variant_names": list(spec.variant_names),
            "timeframes": [timeframe.label for timeframe in spec.timeframes],
            "output_dir": output_dir,
        },
    )
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a strict leak-free VWAP 1m vs 5m comparison.")
    parser.add_argument("--dataset", type=Path, default=None, help="Optional 1-minute dataset path. Defaults to the latest MNQ 1-minute file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional export directory.")
    parser.add_argument("--mode", type=str, default="full", choices=TIMEFRAME_COMPARISON_MODES, help="Campaign mode: full or notebook.")
    parser.add_argument("--notebook-path", type=Path, default=None, help="Optional notebook output path.")
    args = parser.parse_args()

    spec = build_default_timeframe_comparison_spec(dataset_path=args.dataset)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (EXPORTS_DIR / f"vwap_timeframe_comparison_{timestamp}")
    notebook_path = args.notebook_path or (NOTEBOOKS_DIR / "vwap_timeframe_comparison.ipynb")
    artifacts = run_vwap_timeframe_comparison(spec=spec, output_dir=output_dir, mode=args.mode, notebook_path=notebook_path)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
