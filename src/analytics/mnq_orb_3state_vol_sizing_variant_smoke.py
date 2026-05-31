"""Targeted smoke test for MNQ ORB 3-state vol-sizing variants around the 15/60 plateau."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.mnq_orb_regime_filter_sizing_campaign import (
    RegimeFeatureSpec,
    _build_variant,
    _json_dump,
    _safe_series_div,
    _selected_ensemble_sessions,
    build_conditional_bucket_analysis,
    build_session_reference_features,
    build_static_regime_controls,
)
from src.analytics.orb_multi_asset_campaign import BaselineSpec, SearchGrid, SymbolAnalysis, analyze_symbol, resolve_processed_dataset
from src.config.orb_campaign import PropConstraintConfig
from src.config.paths import EXPORTS_DIR
from src.features.volatility import add_atr, add_rolling_std


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "export"
DEFAULT_EXPORT_GLOB = "mnq_orb_regime_filter_sizing_*"
OUTPUT_PREFIX = "mnq_orb_3state_vol_sizing_variant_smoke"
DEFAULT_LOW_MULTIPLIER = 0.50
DEFAULT_MID_MULTIPLIER = 1.00
DEFAULT_HIGH_MULTIPLIER = 0.25


@dataclass(frozen=True)
class VolRatioVariantSpec:
    name: str
    description: str
    components: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class MnqOrb3StateVolSizingVariantSmokeSpec:
    symbol: str = "MNQ"
    dataset_path: Path | None = None
    reference_export_root: Path | None = None
    output_root: Path | None = None
    is_fraction: float | None = None
    aggregation_rule: str | None = None
    baseline: BaselineSpec | None = None
    grid: SearchGrid | None = None
    prop_constraints: PropConstraintConfig | None = None
    min_bucket_obs_is: int = 50
    low_multiplier: float = DEFAULT_LOW_MULTIPLIER
    mid_multiplier: float = DEFAULT_MID_MULTIPLIER
    high_multiplier: float = DEFAULT_HIGH_MULTIPLIER


def build_variant_specs() -> tuple[VolRatioVariantSpec, ...]:
    return (
        VolRatioVariantSpec(
            name="single_15_60",
            description="Baseline retained ratio using fast=15 and slow=60.",
            components=((15, 60),),
        ),
        VolRatioVariantSpec(
            name="single_14_60",
            description="Single nearby alternative using fast=14 and slow=60.",
            components=((14, 60),),
        ),
        VolRatioVariantSpec(
            name="single_16_60",
            description="Single nearby alternative using fast=16 and slow=60.",
            components=((16, 60),),
        ),
        VolRatioVariantSpec(
            name="single_15_70",
            description="Single nearby alternative using fast=15 and slow=70.",
            components=((15, 70),),
        ),
        VolRatioVariantSpec(
            name="single_15_80",
            description="Single nearby alternative using fast=15 and slow=80.",
            components=((15, 80),),
        ),
        VolRatioVariantSpec(
            name="single_16_75",
            description="Single nearby alternative using fast=16 and slow=75.",
            components=((16, 75),),
        ),
        VolRatioVariantSpec(
            name="median_fast15_slow_60_70_80",
            description="Median of fast=15 vs slow in {60, 70, 80}.",
            components=((15, 60), (15, 70), (15, 80)),
        ),
        VolRatioVariantSpec(
            name="median_plateau_compact",
            description="Median of the compact plateau set: (14,60), (15,60), (16,60), (15,70), (16,75).",
            components=((14, 60), (15, 60), (16, 60), (15, 70), (16, 75)),
        ),
    )


def _find_latest_reference_export(exports_root: Path = EXPORTS_DIR, glob_pattern: str = DEFAULT_EXPORT_GLOB) -> Path:
    candidates = [path for path in exports_root.glob(glob_pattern) if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No reference export found for {glob_pattern!r} under {exports_root}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _baseline_from_payload(payload: dict[str, Any]) -> BaselineSpec:
    return BaselineSpec(**payload)


def _grid_from_payload(payload: dict[str, Any]) -> SearchGrid:
    return SearchGrid(
        atr_periods=tuple(int(value) for value in payload["atr_periods"]),
        q_lows_pct=tuple(int(value) for value in payload["q_lows_pct"]),
        q_highs_pct=tuple(int(value) for value in payload["q_highs_pct"]),
        aggregation_rules=tuple(str(value) for value in payload["aggregation_rules"]),
    )


def _constraints_from_payload(payload: dict[str, Any]) -> PropConstraintConfig:
    return PropConstraintConfig(**payload)


def _load_reference_metadata(reference_export_root: Path) -> dict[str, Any]:
    metadata_path = reference_export_root / "run_metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _resolve_runtime_spec(spec: MnqOrb3StateVolSizingVariantSmokeSpec) -> tuple[MnqOrb3StateVolSizingVariantSmokeSpec, dict[str, Any]]:
    reference_export_root = spec.reference_export_root or _find_latest_reference_export()
    metadata = _load_reference_metadata(reference_export_root)
    spec_payload = metadata.get("spec", {})

    baseline = spec.baseline or _baseline_from_payload(spec_payload["baseline"])
    grid = spec.grid or _grid_from_payload(spec_payload["grid"])
    constraints = spec.prop_constraints or _constraints_from_payload(spec_payload["prop_constraints"])
    dataset_path = spec.dataset_path or Path(metadata.get("dataset_path") or resolve_processed_dataset(spec.symbol))
    is_fraction = float(spec.is_fraction if spec.is_fraction is not None else spec_payload.get("is_fraction", 0.70))
    aggregation_rule = str(spec.aggregation_rule or metadata.get("selected_aggregation_rule") or spec_payload.get("aggregation_rule", "majority_50"))

    resolved = MnqOrb3StateVolSizingVariantSmokeSpec(
        symbol=str(spec.symbol).upper(),
        dataset_path=Path(dataset_path),
        reference_export_root=Path(reference_export_root),
        output_root=Path(spec.output_root) if spec.output_root is not None else None,
        is_fraction=is_fraction,
        aggregation_rule=aggregation_rule,
        baseline=baseline,
        grid=grid,
        prop_constraints=constraints,
        min_bucket_obs_is=int(spec.min_bucket_obs_is),
        low_multiplier=float(spec.low_multiplier),
        mid_multiplier=float(spec.mid_multiplier),
        high_multiplier=float(spec.high_multiplier),
    )
    return resolved, metadata


def _make_output_dir(base: Path | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base is None:
        root = DEFAULT_OUTPUT_ROOT / f"{OUTPUT_PREFIX}_{timestamp}"
    else:
        root = Path(base)
        if root.name.startswith(OUTPUT_PREFIX):
            root = root
        else:
            root = root / f"{OUTPUT_PREFIX}_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _required_vol_windows(variant_specs: tuple[VolRatioVariantSpec, ...]) -> tuple[int, ...]:
    windows = sorted({window for variant in variant_specs for component in variant.components for window in component})
    return tuple(int(window) for window in windows)


def build_regime_dataset_with_vol_windows(
    analysis: SymbolAnalysis,
    selected_sessions: set,
    vol_windows: tuple[int, ...],
) -> pd.DataFrame:
    signal_enriched = add_atr(analysis.signal_df.copy(), window=(10, 20))
    for window in vol_windows:
        signal_enriched = add_rolling_std(signal_enriched, window=int(window))
    signal_enriched["session_date"] = pd.to_datetime(signal_enriched["session_date"]).dt.date

    references = build_session_reference_features(
        signal_enriched,
        opening_time=analysis.baseline.opening_time,
        time_exit=analysis.baseline.time_exit,
    )

    selected_index = analysis.candidate_df.copy()
    selected_index["session_date"] = pd.to_datetime(selected_index["session_date"]).dt.date
    selected_index = selected_index.loc[
        selected_index["session_date"].isin(selected_sessions),
        ["session_date", "signal_index"],
    ].copy()

    signal_rows = signal_enriched.loc[selected_index["signal_index"].tolist()].copy()
    signal_rows = signal_rows.reset_index().rename(columns={"index": "signal_index"})
    signal_rows["session_date"] = pd.to_datetime(signal_rows["session_date"]).dt.date

    nominal_trades = analysis.baseline_trades.copy()
    nominal_trades["session_date"] = pd.to_datetime(nominal_trades["session_date"]).dt.date
    nominal_trades = nominal_trades.loc[nominal_trades["session_date"].isin(selected_sessions)].copy()

    regime = selected_index.merge(signal_rows, on=["session_date", "signal_index"], how="left")
    regime = regime.merge(references, on="session_date", how="left")
    regime = regime.merge(
        nominal_trades[
            [
                "session_date",
                "trade_id",
                "entry_time",
                "exit_time",
                "direction",
                "quantity",
                "net_pnl_usd",
                "trade_risk_usd",
                "risk_per_contract_usd",
                "fees",
                "exit_reason",
                "pnl_ticks",
                "entry_price",
            ]
        ],
        on="session_date",
        how="inner",
    )
    regime = regime.sort_values("session_date").reset_index(drop=True)

    is_set = set(pd.to_datetime(pd.Index(analysis.is_sessions)).date)
    regime["phase"] = np.where(regime["session_date"].isin(is_set), "is", "oos")
    return regime


def add_variant_ratio_columns(regime_df: pd.DataFrame, variant_specs: tuple[VolRatioVariantSpec, ...]) -> pd.DataFrame:
    enriched = regime_df.copy()
    component_cache: dict[tuple[int, int], pd.Series] = {}

    for variant in variant_specs:
        component_cols: list[pd.Series] = []
        for fast_window, slow_window in variant.components:
            key = (int(fast_window), int(slow_window))
            if key not in component_cache:
                component_cache[key] = _safe_series_div(
                    pd.to_numeric(enriched[f"vol_std_{fast_window}"], errors="coerce"),
                    pd.to_numeric(enriched[f"vol_std_{slow_window}"], errors="coerce"),
                )
            component_cols.append(component_cache[key].rename(f"ratio_{fast_window}_{slow_window}"))

        if len(component_cols) == 1:
            enriched[variant.name] = component_cols[0]
            continue

        component_frame = pd.concat(component_cols, axis=1)
        enriched[variant.name] = component_frame.median(axis=1, skipna=True)

    return enriched


def _selected_nominal_trades(analysis: SymbolAnalysis, selected_sessions: set) -> pd.DataFrame:
    nominal_trades = analysis.baseline_trades.copy()
    nominal_trades["session_date"] = pd.to_datetime(nominal_trades["session_date"]).dt.date
    nominal_trades = nominal_trades.loc[nominal_trades["session_date"].isin(selected_sessions)].copy()
    return nominal_trades.sort_values("entry_time").reset_index(drop=True)


def _bucket_multiplier_map(spec: MnqOrb3StateVolSizingVariantSmokeSpec) -> dict[str, float]:
    return {
        "low": float(spec.low_multiplier),
        "mid": float(spec.mid_multiplier),
        "high": float(spec.high_multiplier),
    }


def _build_variant_run(
    analysis: SymbolAnalysis,
    regime_df: pd.DataFrame,
    nominal_trades: pd.DataFrame,
    variant_spec: VolRatioVariantSpec,
    smoke_spec: MnqOrb3StateVolSizingVariantSmokeSpec,
):
    feature_spec = RegimeFeatureSpec(
        name=variant_spec.name,
        family="volatility",
        description=variant_spec.description,
        value_column=variant_spec.name,
    )
    conditional_df, _, assignments, calibrations = build_conditional_bucket_analysis(
        regime_df=regime_df,
        nominal_trades=nominal_trades,
        initial_capital=float(analysis.baseline.account_size_usd),
        feature_specs=(feature_spec,),
        min_bucket_obs_is=int(smoke_spec.min_bucket_obs_is),
    )
    if variant_spec.name not in assignments:
        raise ValueError(f"Unable to calibrate buckets for variant {variant_spec.name!r}.")

    controls = build_static_regime_controls(
        regime_df=regime_df,
        feature_name=variant_spec.name,
        bucket_labels=assignments[variant_spec.name],
        bucket_multipliers=_bucket_multiplier_map(smoke_spec),
    )
    variant_run = _build_variant(
        analysis=analysis,
        controls=controls,
        name=variant_spec.name,
        family="dynamic_sizing",
        feature_name=variant_spec.name,
        bucketing=f"{calibrations[variant_spec.name].bucket_kind}_{len(calibrations[variant_spec.name].labels)}",
        description=variant_spec.description,
        calibration_scope="is_only",
        parameters={
            "components": [list(component) for component in variant_spec.components],
            "bucket_multipliers": _bucket_multiplier_map(smoke_spec),
        },
        constraints=smoke_spec.prop_constraints,
        note="Signal ORB unchanged; only the 3-state realized-vol ratio construction changes.",
        rerun_with_sizing=True,
    )
    feature_conditional = conditional_df.loc[conditional_df["feature_name"] == variant_spec.name].copy().reset_index(drop=True)
    return variant_run, controls, feature_conditional


def _scope_phase_lookup(analysis: SymbolAnalysis) -> dict[Any, str]:
    is_set = set(pd.to_datetime(pd.Index(analysis.is_sessions)).date)
    return {session: ("is" if session in is_set else "oos") for session in analysis.all_sessions}


def _variant_bucket_contribution(
    variant_name: str,
    trades: pd.DataFrame,
    controls: pd.DataFrame,
    phase_lookup: dict[Any, str],
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "variant_name",
                "phase",
                "bucket_label",
                "risk_multiplier",
                "num_trades",
                "net_pnl",
                "avg_trade_pnl",
                "win_rate",
            ]
        )

    bucket_map = controls[["session_date", "bucket_label", "risk_multiplier"]].copy()
    bucket_map["session_date"] = pd.to_datetime(bucket_map["session_date"]).dt.date
    merged = trades.copy()
    merged["session_date"] = pd.to_datetime(merged["session_date"]).dt.date
    merged["phase"] = merged["session_date"].map(phase_lookup).fillna("unknown")
    merged = merged.merge(bucket_map, on="session_date", how="left", suffixes=("_trade", "_control"))
    risk_col = "risk_multiplier_control" if "risk_multiplier_control" in merged.columns else "risk_multiplier"

    rows: list[dict[str, Any]] = []
    for (phase, bucket_label), bucket_trades in merged.groupby(["phase", "bucket_label"], dropna=False):
        pnl = pd.to_numeric(bucket_trades["net_pnl_usd"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "variant_name": variant_name,
                "phase": phase,
                "bucket_label": bucket_label,
                "risk_multiplier": float(pd.to_numeric(bucket_trades[risk_col], errors="coerce").dropna().iloc[0]) if bucket_trades[risk_col].notna().any() else np.nan,
                "num_trades": int(len(bucket_trades)),
                "net_pnl": float(pnl.sum()),
                "avg_trade_pnl": float(pnl.mean()) if len(pnl) > 0 else 0.0,
                "win_rate": float((pnl > 0).mean()) if len(pnl) > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _variant_daily_export(variant_name: str, daily_results: pd.DataFrame, phase_lookup: dict[Any, str], capital: float) -> pd.DataFrame:
    out = daily_results.copy()
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.date
    out["variant_name"] = variant_name
    out["phase"] = out["session_date"].map(phase_lookup).fillna("unknown")
    out["daily_pnl_usd"] = pd.to_numeric(out["daily_pnl_usd"], errors="coerce").fillna(0.0)
    out["daily_return"] = out["daily_pnl_usd"] / float(capital) if capital > 0 else 0.0
    return out


def _variant_trade_export(variant_name: str, trades: pd.DataFrame, controls: pd.DataFrame, phase_lookup: dict[Any, str]) -> pd.DataFrame:
    out = trades.copy()
    if out.empty:
        out["variant_name"] = pd.Series(dtype="string")
        return out
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.date
    out["variant_name"] = variant_name
    out["phase"] = out["session_date"].map(phase_lookup).fillna("unknown")
    control_cols = controls[["session_date", "bucket_label", "risk_multiplier"]].copy()
    control_cols["session_date"] = pd.to_datetime(control_cols["session_date"]).dt.date
    merged = out.merge(control_cols, on="session_date", how="left", suffixes=("_trade", "_control"))
    if "risk_multiplier_control" in merged.columns:
        merged["risk_multiplier"] = pd.to_numeric(merged["risk_multiplier_control"], errors="coerce")
    elif "risk_multiplier" in merged.columns:
        merged["risk_multiplier"] = pd.to_numeric(merged["risk_multiplier"], errors="coerce")
    return merged


def _summary_row(variant_name: str, summary_by_scope: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    oos_row = summary_by_scope.loc[summary_by_scope["scope"] == "oos"].iloc[0].to_dict()
    is_row = summary_by_scope.loc[summary_by_scope["scope"] == "is"].iloc[0].to_dict()
    oos_trades = trades.copy()
    oos_trades["session_date"] = pd.to_datetime(oos_trades["session_date"]).dt.date
    avg_trade_pnl = float(pd.to_numeric(oos_trades["net_pnl_usd"], errors="coerce").mean()) if not oos_trades.empty else 0.0

    pass_prop = bool(oos_row.get("profit_target_reached_before_max_loss", False)) and float(
        oos_row.get("daily_loss_limit_breach_freq", 0.0)
    ) <= 0.0 and float(oos_row.get("max_loss_limit_buffer_usd", -1.0)) >= 0.0

    return {
        "variant_name": variant_name,
        "net_pnl": float(oos_row.get("net_pnl", 0.0)),
        "sharpe": float(oos_row.get("sharpe", 0.0)),
        "sortino": float(oos_row.get("sortino", 0.0)),
        "max_drawdown": float(oos_row.get("max_drawdown", 0.0)),
        "max_daily_loss": float(oos_row.get("worst_day", 0.0)),
        "profit_factor": float(oos_row.get("profit_factor", 0.0)),
        "win_rate": float((pd.to_numeric(oos_trades["net_pnl_usd"], errors="coerce").fillna(0.0) > 0).mean()) if not oos_trades.empty else 0.0,
        "num_trades": int(oos_row.get("n_trades", 0)),
        "avg_trade_pnl": avg_trade_pnl,
        "prop_pass": pass_prop,
        "pass_prop_constraints": pass_prop,
        "daily_loss_limit_breach_freq": float(oos_row.get("daily_loss_limit_breach_freq", 0.0)),
        "profit_target_reached_before_max_loss": bool(oos_row.get("profit_target_reached_before_max_loss", False)),
        "max_loss_limit_buffer_usd": float(oos_row.get("max_loss_limit_buffer_usd", 0.0)),
        "is_net_pnl": float(is_row.get("net_pnl", 0.0)),
        "is_sharpe": float(is_row.get("sharpe", 0.0)),
        "is_sortino": float(is_row.get("sortino", 0.0)),
    }


def _apply_baseline_deltas(summary_df: pd.DataFrame, baseline_name: str = "single_15_60") -> pd.DataFrame:
    out = summary_df.copy()
    baseline = out.loc[out["variant_name"] == baseline_name]
    if baseline.empty:
        raise ValueError(f"Baseline variant {baseline_name!r} is missing from summary.")
    baseline_row = baseline.iloc[0]
    out["delta_sharpe_vs_single_15_60"] = pd.to_numeric(out["sharpe"], errors="coerce") - float(baseline_row["sharpe"])
    out["delta_net_pnl_vs_single_15_60"] = pd.to_numeric(out["net_pnl"], errors="coerce") - float(baseline_row["net_pnl"])
    out["delta_maxdd_vs_single_15_60"] = pd.to_numeric(out["max_drawdown"], errors="coerce") - float(baseline_row["max_drawdown"])
    out["abs_max_drawdown"] = pd.to_numeric(out["max_drawdown"], errors="coerce").abs()
    return out


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    data = frame.loc[:, columns].copy()
    rendered = data.astype(object)
    for column in rendered.columns:
        if rendered[column].dtype == object:
            continue
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in data.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, (bool, np.bool_)):
                values.append("yes" if bool(value) else "no")
            elif pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _plateau_note(summary_df: pd.DataFrame) -> str:
    nearby = summary_df.loc[summary_df["variant_name"].str.startswith("single_")].copy()
    if nearby.empty:
        return "Single-parameter neighbors are unavailable."
    sharpe_span = float(pd.to_numeric(nearby["sharpe"], errors="coerce").max() - pd.to_numeric(nearby["sharpe"], errors="coerce").min())
    pnl_span = float(pd.to_numeric(nearby["net_pnl"], errors="coerce").max() - pd.to_numeric(nearby["net_pnl"], errors="coerce").min())
    return (
        f"Nearby single variants span about Sharpe `{sharpe_span:.3f}` and net PnL `{pnl_span:.1f}` OOS, "
        "which is a practical read on how flat the 15/60 neighborhood is."
    )


def _choose_verdict(summary_df: pd.DataFrame) -> tuple[str, str]:
    baseline = summary_df.loc[summary_df["variant_name"] == "single_15_60"].iloc[0]
    ensembles = summary_df.loc[
        summary_df["variant_name"].isin(["median_fast15_slow_60_70_80", "median_plateau_compact"])
    ].copy()
    if ensembles.empty:
        return "ne rien changer faute d'amélioration nette.", "No ensemble variant was available to compare."

    baseline_sharpe = float(baseline["sharpe"])
    baseline_dd_abs = abs(float(baseline["max_drawdown"]))
    baseline_prop = bool(baseline["pass_prop_constraints"])

    ensembles["qualifies"] = (
        (pd.to_numeric(ensembles["sharpe"], errors="coerce") >= baseline_sharpe - 0.05)
        & (
            (pd.to_numeric(ensembles["sharpe"], errors="coerce") > baseline_sharpe + 1e-9)
            | (pd.to_numeric(ensembles["max_drawdown"], errors="coerce").abs() <= baseline_dd_abs - 100.0)
            | (ensembles["pass_prop_constraints"].astype(bool) & ~baseline_prop)
        )
    )
    qualified = ensembles.loc[ensembles["qualifies"]].copy()
    if not qualified.empty:
        ranked = qualified.sort_values(
            ["sharpe", "pass_prop_constraints", "abs_max_drawdown", "net_pnl"],
            ascending=[False, False, True, False],
        ).reset_index(drop=True)
        winner = ranked.iloc[0]
        verdict = f"passer à `{winner['variant_name']}`"
        rationale = (
            f"{winner['variant_name']} keeps or improves OOS Sharpe versus single_15_60 "
            f"({float(winner['sharpe']):.3f} vs {baseline_sharpe:.3f}) while improving the risk/stability profile."
        )
        return verdict, rationale

    top_variant = summary_df.sort_values(["sharpe", "net_pnl"], ascending=[False, False]).iloc[0]
    if str(top_variant["variant_name"]) == "single_15_60":
        return "garder `single_15_60`", "single_15_60 remains at or near the top, and the ensemble variants do not deliver a cleaner OOS trade-off."
    return "ne rien changer faute d'amélioration nette.", "Another variant edges the ranking, but the gain is too small or not cleaner enough to justify a production change."


def _write_final_report(
    output_dir: Path,
    summary_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    smoke_spec: MnqOrb3StateVolSizingVariantSmokeSpec,
    reference_export_root: Path,
) -> tuple[Path, str]:
    by_sharpe = summary_df.sort_values(["sharpe", "net_pnl"], ascending=[False, False]).reset_index(drop=True)
    by_prop = summary_df.sort_values(
        ["pass_prop_constraints", "max_loss_limit_buffer_usd", "daily_loss_limit_breach_freq", "sharpe"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    baseline = summary_df.loc[summary_df["variant_name"] == "single_15_60"].iloc[0]
    verdict, rationale = _choose_verdict(summary_df)

    lines = [
        "# MNQ ORB 3-State Vol-Sizing Variant Smoke",
        "",
        "## Objective",
        "",
        "Quickly test whether the retained MNQ ORB 3-state vol sizing stays robust around `fast=15 / slow=60`,",
        "while changing only the vol-ratio construction and keeping the ORB signal, entries, exits, costs, sessions, filters, invalidation, and risk model unchanged.",
        "",
        f"- Reference export reused: `{reference_export_root}`",
        f"- Symbol: `{smoke_spec.symbol}`",
        f"- Dataset: `{smoke_spec.dataset_path}`",
        f"- Aggregation rule: `{smoke_spec.aggregation_rule}`",
        f"- Fixed three-state multipliers for every variant: `low={smoke_spec.low_multiplier:.2f}x`, `mid={smoke_spec.mid_multiplier:.2f}x`, `high={smoke_spec.high_multiplier:.2f}x`",
        "",
        "## Ranked By OOS Sharpe",
        "",
        _markdown_table(
            by_sharpe,
            [
                "variant_name",
                "sharpe",
                "net_pnl",
                "max_drawdown",
                "profit_factor",
                "num_trades",
                "delta_sharpe_vs_single_15_60",
            ],
        ),
        "",
        "## Ranked By Prop-Safe Robustness",
        "",
        _markdown_table(
            by_prop,
            [
                "variant_name",
                "pass_prop_constraints",
                "max_loss_limit_buffer_usd",
                "daily_loss_limit_breach_freq",
                "sharpe",
                "max_drawdown",
            ],
        ),
        "",
        "## Explicit Comparison Vs `single_15_60`",
        "",
        f"- Baseline `single_15_60`: Sharpe `{float(baseline['sharpe']):.3f}`, net PnL `{float(baseline['net_pnl']):.1f}`, max drawdown `{float(baseline['max_drawdown']):.1f}`.",
        f"- {_plateau_note(summary_df)}",
        f"- Final verdict: {verdict}.",
        f"- Rationale: {rationale}",
    ]
    if not bucket_df.empty:
        oos_buckets = bucket_df.loc[bucket_df["phase"] == "oos"].copy()
        if not oos_buckets.empty:
            lines.extend(
                [
                    "",
                    "## OOS Bucket Contribution Snapshot",
                    "",
                    _markdown_table(
                        oos_buckets.sort_values(["variant_name", "bucket_label"]).reset_index(drop=True),
                        ["variant_name", "bucket_label", "risk_multiplier", "num_trades", "net_pnl", "avg_trade_pnl"],
                    ),
                ]
            )

    report_path = output_dir / "final_report.md"
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path, verdict


def run_smoke_campaign(spec: MnqOrb3StateVolSizingVariantSmokeSpec) -> dict[str, Any]:
    resolved_spec, reference_metadata = _resolve_runtime_spec(spec)
    output_dir = _make_output_dir(resolved_spec.output_root)
    variant_specs = build_variant_specs()

    analysis = analyze_symbol(
        symbol=resolved_spec.symbol,
        baseline=resolved_spec.baseline,
        grid=resolved_spec.grid,
        is_fraction=float(resolved_spec.is_fraction),
        dataset_path=resolved_spec.dataset_path,
    )
    selected_sessions = _selected_ensemble_sessions(analysis, resolved_spec.aggregation_rule)
    if not selected_sessions:
        raise ValueError("No ensemble-selected sessions were found; cannot run the sizing smoke test.")

    regime_df = build_regime_dataset_with_vol_windows(
        analysis=analysis,
        selected_sessions=selected_sessions,
        vol_windows=_required_vol_windows(variant_specs),
    )
    regime_df = add_variant_ratio_columns(regime_df, variant_specs)
    nominal_trades = _selected_nominal_trades(analysis, selected_sessions)
    phase_lookup = _scope_phase_lookup(analysis)

    summary_rows: list[dict[str, Any]] = []
    daily_exports: list[pd.DataFrame] = []
    trade_exports: list[pd.DataFrame] = []
    bucket_exports: list[pd.DataFrame] = []

    for variant_spec in variant_specs:
        variant_run, controls, _ = _build_variant_run(
            analysis=analysis,
            regime_df=regime_df,
            nominal_trades=nominal_trades,
            variant_spec=variant_spec,
            smoke_spec=resolved_spec,
        )
        oos_sessions = set(pd.to_datetime(pd.Index(analysis.oos_sessions)).date)
        variant_oos_trades = variant_run.trades.copy()
        variant_oos_trades["session_date"] = pd.to_datetime(variant_oos_trades["session_date"]).dt.date
        variant_oos_trades = variant_oos_trades.loc[variant_oos_trades["session_date"].isin(oos_sessions)].copy().reset_index(drop=True)

        summary_rows.append(_summary_row(variant_run.name, variant_run.summary_by_scope, variant_oos_trades))
        daily_exports.append(
            _variant_daily_export(
                variant_run.name,
                variant_run.daily_results,
                phase_lookup=phase_lookup,
                capital=float(analysis.baseline.account_size_usd),
            )
        )
        trade_exports.append(_variant_trade_export(variant_run.name, variant_run.trades, controls, phase_lookup=phase_lookup))
        bucket_exports.append(_variant_bucket_contribution(variant_run.name, variant_run.trades, controls, phase_lookup=phase_lookup))

    summary_df = _apply_baseline_deltas(pd.DataFrame(summary_rows))
    summary_df = summary_df.sort_values(["sharpe", "net_pnl"], ascending=[False, False]).reset_index(drop=True)
    daily_df = pd.concat(daily_exports, ignore_index=True) if daily_exports else pd.DataFrame()
    trade_df = pd.concat(trade_exports, ignore_index=True) if trade_exports else pd.DataFrame()
    bucket_df = pd.concat(bucket_exports, ignore_index=True) if bucket_exports else pd.DataFrame()

    summary_path = output_dir / "variant_summary.csv"
    daily_path = output_dir / "variant_daily_returns.csv"
    trade_path = output_dir / "variant_trade_summary.csv"
    bucket_path = output_dir / "variant_bucket_contribution.csv"
    metadata_path = output_dir / "run_metadata.json"

    summary_df.to_csv(summary_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    trade_df.to_csv(trade_path, index=False)
    bucket_df.to_csv(bucket_path, index=False)
    report_path, verdict = _write_final_report(
        output_dir=output_dir,
        summary_df=summary_df,
        bucket_df=bucket_df,
        smoke_spec=resolved_spec,
        reference_export_root=resolved_spec.reference_export_root,
    )

    _json_dump(
        metadata_path,
        {
            "run_timestamp": datetime.now().isoformat(),
            "reference_export_root": resolved_spec.reference_export_root,
            "reference_export_metadata": reference_metadata,
            "dataset_path": resolved_spec.dataset_path,
            "selected_symbol": resolved_spec.symbol,
            "selected_aggregation_rule": resolved_spec.aggregation_rule,
            "selected_session_count": int(len(selected_sessions)),
            "spec": asdict(resolved_spec),
            "variants": [asdict(variant_spec) for variant_spec in variant_specs],
            "final_verdict": verdict,
        },
    )
    return {
        "output_dir": output_dir,
        "summary_path": summary_path,
        "daily_path": daily_path,
        "trade_path": trade_path,
        "bucket_path": bucket_path,
        "report_path": report_path,
        "metadata_path": metadata_path,
        "summary": summary_df,
    }


def build_default_spec(symbol: str = "MNQ", output_root: Path | None = None) -> MnqOrb3StateVolSizingVariantSmokeSpec:
    return MnqOrb3StateVolSizingVariantSmokeSpec(symbol=str(symbol).upper(), output_root=output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fast MNQ ORB 3-state vol-sizing plateau smoke test.")
    parser.add_argument("--symbol", type=str, default="MNQ")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--reference-export-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--is-fraction", type=float, default=None)
    args = parser.parse_args()

    artifacts = run_smoke_campaign(
        MnqOrb3StateVolSizingVariantSmokeSpec(
            symbol=args.symbol,
            dataset_path=Path(args.dataset_path) if args.dataset_path is not None else None,
            reference_export_root=Path(args.reference_export_root) if args.reference_export_root is not None else None,
            output_root=Path(args.output_root) if args.output_root is not None else None,
            is_fraction=float(args.is_fraction) if args.is_fraction is not None else None,
        )
    )
    print(f"output_dir: {artifacts['output_dir']}")
    print(f"summary: {artifacts['summary_path']}")
    print(f"report: {artifacts['report_path']}")


if __name__ == "__main__":
    main()
