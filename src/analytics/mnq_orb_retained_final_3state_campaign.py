"""Run the MNQ 3-state regime-sizing campaign on the retained-final ORB sleeve."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.audit_mnq_orb_retained_vs_3state import (
    RetainedConfig,
    _latest_mnq_dataset,
    _rebuild_retained_final,
)
from src.analytics.mnq_orb_regime_filter_sizing_campaign import (
    SUMMARY_COLUMNS,
    RegimeVariantRun,
    _best_feature_per_family,
    _build_mapping_rows,
    _build_variant,
    _conditional_rows_for_feature,
    _export_variant_artifacts,
    _feature_specs,
    _json_dump,
    _nominal_controls,
    _variant_row,
    build_conditional_bucket_analysis,
    build_session_reference_features,
    build_state_mapping_from_is_scores,
    build_static_regime_controls,
)
from src.analytics.orb_multi_asset_campaign import BaselineSpec
from src.config.orb_campaign import PropConstraintConfig, build_prop_constraints
from src.config.paths import EXPORTS_DIR, ensure_directories
from src.config.settings import get_instrument_spec
from src.features.volatility import add_atr, add_rolling_std


OUTPUT_PREFIX = "mnq_orb_retained_final_3state_campaign"


@dataclass(frozen=True)
class RetainedFinal3StateCampaignSpec:
    dataset_path: Path | None = None
    output_root: Path | None = None
    min_bucket_obs_is: int = 50
    multipliers_by_rank: tuple[float, float, float] = (0.50, 0.75, 1.00)
    prop_constraints: PropConstraintConfig = field(default_factory=build_prop_constraints)


def _make_output_dir(base: Path | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base is None:
        root = EXPORTS_DIR / f"{OUTPUT_PREFIX}_{timestamp}"
    else:
        root = Path(base)
        if not root.name.startswith(OUTPUT_PREFIX):
            root = root / f"{OUTPUT_PREFIX}_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _retained_baseline_spec(cfg: RetainedConfig) -> BaselineSpec:
    return BaselineSpec(
        or_minutes=int(cfg.or_minutes),
        opening_time=str(cfg.opening_time),
        direction=str(cfg.direction),
        one_trade_per_day=bool(cfg.one_trade_per_day),
        entry_buffer_ticks=int(cfg.entry_buffer_ticks),
        stop_buffer_ticks=int(cfg.stop_buffer_ticks),
        target_multiple=float(cfg.target_multiple),
        vwap_confirmation=bool(cfg.vwap_confirmation),
        vwap_column=str(cfg.vwap_column),
        time_exit=str(cfg.time_exit),
        account_size_usd=float(cfg.account_size_usd),
        risk_per_trade_pct=float(cfg.risk_per_trade_pct),
        entry_on_next_open=bool(cfg.entry_on_next_open),
    )


def _build_retained_regime_dataset(retained: dict[str, object], cfg: RetainedConfig) -> pd.DataFrame:
    minute_df = retained["minute_df"].copy()
    minute_df = add_atr(minute_df, window=(10, 20))
    minute_df = add_rolling_std(minute_df, window=15)
    minute_df = add_rolling_std(minute_df, window=60)
    minute_df["session_date"] = pd.to_datetime(minute_df["session_date"]).dt.date
    minute_df["timestamp"] = pd.to_datetime(minute_df["timestamp"], errors="coerce", utc=True)

    references = build_session_reference_features(
        minute_df,
        opening_time=str(cfg.opening_time),
        time_exit=str(cfg.time_exit),
    )

    selected_final = retained["selected_final"].copy()
    selected_final["session_date"] = pd.to_datetime(selected_final["session_date"]).dt.date
    selected_final["timestamp"] = pd.to_datetime(selected_final["timestamp"], errors="coerce", utc=True)

    signal_rows = selected_final.merge(
        minute_df[
            [
                "session_date",
                "timestamp",
                "atr_10",
                "atr_20",
                "vol_std_15",
                "vol_std_60",
            ]
        ],
        on=["session_date", "timestamp"],
        how="left",
    )

    nominal_trades = retained["trades"].copy()
    nominal_trades["session_date"] = pd.to_datetime(nominal_trades["session_date"]).dt.date

    regime = signal_rows.merge(references, on="session_date", how="left")
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
                "fees",
                "exit_reason",
            ]
        ],
        on="session_date",
        how="inner",
    )
    regime = regime.sort_values("session_date").reset_index(drop=True)

    is_set = set(pd.to_datetime(pd.Index(retained["is_sessions"])).date)
    regime["phase"] = np.where(regime["session_date"].isin(is_set), "is", "oos")
    regime["weekday_name"] = pd.to_numeric(regime["weekday"], errors="coerce").map(
        {
            0: "monday",
            1: "tuesday",
            2: "wednesday",
            3: "thursday",
            4: "friday",
            5: "saturday",
            6: "sunday",
        }
    )

    atr_20 = pd.to_numeric(regime["atr_20"], errors="coerce")
    atr_20_open = pd.to_numeric(regime["atr_20_open"], errors="coerce")
    atr_30 = pd.to_numeric(regime["atr_14"], errors="coerce")
    vol_15 = pd.to_numeric(regime["vol_std_15"], errors="coerce")
    vol_60 = pd.to_numeric(regime["vol_std_60"], errors="coerce")
    or_width = pd.to_numeric(regime["or_width"], errors="coerce")
    signal_close = pd.to_numeric(regime["close"], errors="coerce")
    or_high = pd.to_numeric(regime["or_high"], errors="coerce")
    or_low = pd.to_numeric(regime["or_low"], errors="coerce")
    signal_side = pd.to_numeric(regime["signal"], errors="coerce")

    regime["atr_ratio_10_30"] = pd.to_numeric(regime["atr_10"], errors="coerce").where(atr_30.ne(0)).divide(atr_30.where(atr_30.ne(0)))
    regime["opening_range_width_pts"] = or_width
    regime["realized_vol_ratio_15_60"] = vol_15.where(vol_60.ne(0)).divide(vol_60.where(vol_60.ne(0)))
    regime["gap_abs_atr20"] = (
        pd.to_numeric(regime["rth_open"], errors="coerce") - pd.to_numeric(regime["prev_rth_close"], errors="coerce")
    ).abs().where(atr_20_open.ne(0)).divide(atr_20_open.where(atr_20_open.ne(0)))
    regime["signal_vwap_distance_atr20"] = (
        signal_close - pd.to_numeric(regime["continuous_session_vwap"], errors="coerce")
    ).abs().where(atr_20.ne(0)).divide(atr_20.where(atr_20.ne(0)))
    extension = np.where(signal_side.eq(1), signal_close - or_high, or_low - signal_close)
    regime["signal_extension_over_or"] = pd.Series(extension, index=regime.index).where(or_width.ne(0)).divide(or_width.where(or_width.ne(0)))
    regime["nominal_selected"] = True
    return regime


def _analysis_bundle(retained: dict[str, object], dataset_path: Path, cfg: RetainedConfig) -> SimpleNamespace:
    return SimpleNamespace(
        symbol="MNQ",
        dataset_path=dataset_path,
        baseline=_retained_baseline_spec(cfg),
        instrument_spec=get_instrument_spec("MNQ"),
        baseline_trades=retained["trades"].copy(),
        all_sessions=list(retained["all_sessions"]),
        is_sessions=list(retained["is_sessions"]),
        oos_sessions=list(retained["oos_sessions"]),
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
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


def _write_campaign_markdown(
    output_path: Path,
    summary_df: pd.DataFrame,
    feature_scores: pd.DataFrame,
    dataset_path: Path,
    cfg: RetainedConfig,
) -> None:
    nominal = summary_df.loc[summary_df["variant_name"] == "nominal"].iloc[0]
    dynamic = summary_df.loc[summary_df["family"] == "dynamic_sizing"].copy()
    best_dynamic = dynamic.sort_values(["oos_sharpe", "oos_net_pnl"], ascending=[False, False]).head(1)
    top_features = feature_scores.sort_values(["feature_selection_score", "min_bucket_obs_is"], ascending=[False, False]).head(5)

    lines = [
        "# MNQ ORB retained-final 3-state campaign",
        "",
        "## Baseline",
        "",
        "- This campaign is run on the **retained final** sleeve, not on the earlier nominal ORB branch.",
        f"- Retained config: `{cfg.name}`",
        f"- OR window: `{cfg.or_minutes}` minutes",
        f"- Direction: `{cfg.direction}`",
        f"- VWAP confirmation: `{'enabled' if cfg.vwap_confirmation else 'disabled'}`",
        f"- ATR ensemble: `ATR({cfg.atr_window})`, vote threshold `{cfg.vote_threshold:.2f}`",
        f"- Compression / dynamic gate: `{cfg.compression_mode}` / `{cfg.dynamic_mode}`",
        f"- Base risk per trade: `{cfg.risk_per_trade_pct:.2f}%`",
        f"- Dataset: `{dataset_path}`",
        "",
        "## Nominal retained-final OOS",
        "",
        f"- Net PnL: `{float(nominal['oos_net_pnl']):.1f}`",
        f"- Sharpe: `{float(nominal['oos_sharpe']):.3f}`",
        f"- Max DD: `{float(nominal['oos_max_drawdown']):.1f}`",
        f"- Trades: `{int(nominal['oos_n_trades'])}`",
        "",
        "## Top feature candidates",
        "",
        _markdown_table(top_features) if not top_features.empty else "_No valid overlay feature survived the campaign filters._",
        "",
        "## Best 3-state overlay",
        "",
    ]
    if best_dynamic.empty:
        lines.append("_No dynamic-sizing variant was produced._")
    else:
        row = best_dynamic.iloc[0]
        lines.extend(
            [
                f"- Variant: `{row['variant_name']}`",
                f"- Feature: `{row['feature_name']}`",
                f"- OOS net PnL: `{float(row['oos_net_pnl']):.1f}`",
                f"- OOS Sharpe: `{float(row['oos_sharpe']):.3f}`",
                f"- OOS max DD: `{float(row['oos_max_drawdown']):.1f}`",
                f"- Net-PnL retention vs nominal: `{float(row['oos_net_pnl_retention_vs_nominal']):.3f}`",
                f"- Sharpe delta vs nominal: `{float(row['oos_sharpe_delta_vs_nominal']):.3f}`",
                f"- Max-DD improvement vs nominal: `{float(row['oos_max_drawdown_improvement_vs_nominal']):.3f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Full summary",
            "",
            _markdown_table(summary_df),
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_retained_final_3state_campaign(spec: RetainedFinal3StateCampaignSpec) -> dict[str, Path]:
    ensure_directories()
    output_root = _make_output_dir(spec.output_root)

    dataset_path = Path(spec.dataset_path) if spec.dataset_path is not None else _latest_mnq_dataset()
    retained = _rebuild_retained_final(dataset_path)
    cfg = retained.get("config", RetainedConfig())
    analysis = _analysis_bundle(retained, dataset_path, cfg)
    regime_df = _build_retained_regime_dataset(retained, cfg)
    regime_path = output_root / "selected_session_regimes.csv"
    regime_df.to_csv(regime_path, index=False)

    nominal_trades = analysis.baseline_trades.copy()
    nominal_trades["session_date"] = pd.to_datetime(nominal_trades["session_date"]).dt.date
    nominal_trades = nominal_trades.loc[nominal_trades["session_date"].isin(set(regime_df["session_date"]))].copy()

    conditional_df, feature_score_df, assignments, calibrations = build_conditional_bucket_analysis(
        regime_df=regime_df,
        nominal_trades=nominal_trades,
        initial_capital=float(analysis.baseline.account_size_usd),
        feature_specs=_feature_specs(),
        min_bucket_obs_is=int(spec.min_bucket_obs_is),
    )
    conditional_path = output_root / "conditional_bucket_analysis.csv"
    conditional_df.to_csv(conditional_path, index=False)

    feature_ranking_path = output_root / "feature_ranking.csv"
    feature_score_df.to_csv(feature_ranking_path, index=False)

    variants: list[RegimeVariantRun] = []
    mapping_frames: list[pd.DataFrame] = []

    nominal_variant = _build_variant(
        analysis=analysis,
        controls=_nominal_controls(regime_df),
        name="nominal",
        family="baseline",
        feature_name="nominal",
        bucketing="none",
        description="Retained-final original sleeve without extra 3-state overlay.",
        calibration_scope="none",
        parameters={},
        constraints=spec.prop_constraints,
        note="Reference retained-final sleeve for all coverage and improvement comparisons.",
        rerun_with_sizing=False,
    )
    variants.append(nominal_variant)

    top_by_family = _best_feature_per_family(feature_score_df)
    for row in top_by_family.itertuples():
        feature_name = str(row.feature_name)
        feature_rows = _conditional_rows_for_feature(conditional_df, feature_name)
        if feature_rows.empty:
            continue
        worst_bucket = str(row.worst_bucket_is)
        keep_multipliers = {label: (0.0 if str(label) == worst_bucket else 1.0) for label in calibrations[feature_name].labels}
        controls = build_static_regime_controls(regime_df, feature_name, assignments[feature_name], keep_multipliers)
        variants.append(
            _build_variant(
                analysis=analysis,
                controls=controls,
                name=f"filter_skip_worst_{feature_name}",
                family="regime_filter",
                feature_name=feature_name,
                bucketing=f"{calibrations[feature_name].bucket_kind}_{len(calibrations[feature_name].labels)}",
                description=f"Skip the weakest retained-final IS bucket for feature {feature_name}.",
                calibration_scope="is_only",
                parameters={"bucket_multipliers": keep_multipliers},
                constraints=spec.prop_constraints,
                note=f"Retained-final campaign: bucket {worst_bucket} removed using IS-only ranking.",
                rerun_with_sizing=False,
            )
        )
        mapping_frames.append(_build_mapping_rows(f"filter_skip_worst_{feature_name}", feature_name, feature_rows, keep_multipliers))

    continuous_candidates = feature_score_df.loc[
        feature_score_df["valid_for_overlay"] & feature_score_df["bucket_kind"].eq("quantile")
    ].copy()
    if not continuous_candidates.empty:
        for row in continuous_candidates.itertuples():
            feature_name = str(row.feature_name)
            feature_rows = _conditional_rows_for_feature(conditional_df, feature_name)
            if feature_rows.empty:
                continue
            sizing_3state = build_state_mapping_from_is_scores(
                feature_rows,
                multipliers_by_rank=tuple(float(x) for x in spec.multipliers_by_rank),
            )
            controls_3state = build_static_regime_controls(regime_df, feature_name, assignments[feature_name], sizing_3state)
            variants.append(
                _build_variant(
                    analysis=analysis,
                    controls=controls_3state,
                    name=f"sizing_3state_{feature_name}",
                    family="dynamic_sizing",
                    feature_name=feature_name,
                    bucketing=f"{calibrations[feature_name].bucket_kind}_{len(calibrations[feature_name].labels)}",
                    description=f"Three-state discrete sizing on retained-final feature {feature_name}.",
                    calibration_scope="is_only",
                    parameters={"bucket_multipliers": sizing_3state},
                    constraints=spec.prop_constraints,
                    note="Retained-final original trade set, with risk scaled by the calibrated 3-state map.",
                    rerun_with_sizing=True,
                )
            )
            mapping_frames.append(_build_mapping_rows(f"sizing_3state_{feature_name}", feature_name, feature_rows, sizing_3state))

    summary_rows = [_variant_row(variant, nominal_variant) for variant in variants]
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[[column for column in SUMMARY_COLUMNS if column in summary_df.columns]]
    summary_path = output_root / "summary_variants.csv"
    summary_df.to_csv(summary_path, index=False)

    mapping_df = pd.concat(mapping_frames, ignore_index=True) if mapping_frames else pd.DataFrame()
    mapping_path = output_root / "regime_state_mappings.csv"
    mapping_df.to_csv(mapping_path, index=False)

    for variant in variants:
        _export_variant_artifacts(output_root, variant)

    markdown_path = output_root / "campaign_summary.md"
    _write_campaign_markdown(markdown_path, summary_df, feature_score_df, dataset_path, cfg)

    metadata_path = output_root / "run_metadata.json"
    _json_dump(
        metadata_path,
        {
            "run_timestamp": datetime.now().isoformat(),
            "dataset_path": dataset_path,
            "campaign_type": "retained_final_3state",
            "retained_config": asdict(cfg),
            "spec": asdict(spec),
            "selected_session_count": int(regime_df["session_date"].nunique()),
            "selected_trade_count": int(len(nominal_trades)),
        },
    )
    return {
        "output_root": output_root,
        "summary": summary_path,
        "conditional": conditional_path,
        "feature_ranking": feature_ranking_path,
        "mappings": mapping_path,
        "regime_dataset": regime_path,
        "markdown": markdown_path,
        "metadata": metadata_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--min-bucket-obs-is", type=int, default=50)
    args = parser.parse_args()

    artifacts = run_retained_final_3state_campaign(
        RetainedFinal3StateCampaignSpec(
            dataset_path=Path(args.dataset_path) if args.dataset_path is not None else None,
            output_root=Path(args.output_root) if args.output_root is not None else None,
            min_bucket_obs_is=int(args.min_bucket_obs_is),
        )
    )
    for key, value in artifacts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
