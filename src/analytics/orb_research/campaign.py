"""Comprehensive ORB research campaign (compression, VWAP exits, dynamic thresholds)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analytics.atr_ensemble_campaign import (
    BaselineSpec as LegacyBaselineSpec,
)
from src.analytics.atr_ensemble_campaign import (
    _build_baseline_strategy as legacy_build_baseline_strategy,
)
from src.analytics.atr_ensemble_campaign import (
    _build_submodels as legacy_build_submodels,
)
from src.analytics.atr_ensemble_campaign import (
    _build_variant_configs as legacy_build_variant_configs,
)
from src.analytics.atr_ensemble_campaign import (
    _build_zone_definitions as legacy_build_zone_definitions,
)
from src.analytics.atr_ensemble_campaign import (
    _prepare_feature_dataset as legacy_prepare_feature_dataset,
)
from src.analytics.atr_ensemble_campaign import _run_backtest as legacy_run_backtest
from src.analytics.atr_ensemble_campaign import _select_signal_indices as legacy_select_signal_indices
from src.analytics.atr_ensemble_campaign import _selected_signal_rows as legacy_selected_signal_rows
from src.analytics.atr_ensemble_campaign import _signal_df_for_selected_indices as legacy_signal_df_for_selected_indices
from src.config.paths import REPO_ROOT, ensure_directories
from src.engine.execution_model import ExecutionModel
from src.engine.portfolio import build_equity_curve

from .evaluation import compute_extended_metrics, compute_scores
from .exits import run_exit_variant_backtest
from .features import (
    apply_ensemble_selection,
    attach_daily_reference,
    build_candidate_universe,
    build_daily_reference,
    build_signal_frame_for_backtest,
    calibrate_ensemble_thresholds,
    compression_mask,
    compute_noise_sigma,
    dynamic_gate_mask,
    first_pass_signal_rows,
    prepare_minute_dataset,
)
from .types import (
    BaselineEnsembleConfig,
    BaselineEntryConfig,
    CampaignConfig,
    CampaignContext,
    CompressionConfig,
    DynamicThresholdConfig,
    ExitConfig,
    ExperimentConfig,
)


def _split_sessions(all_sessions: list, is_fraction: float) -> tuple[list, list]:
    if len(all_sessions) < 2:
        raise ValueError("Not enough sessions for IS/OOS split.")
    split_idx = int(len(all_sessions) * float(is_fraction))
    split_idx = max(1, min(len(all_sessions) - 1, split_idx))
    return all_sessions[:split_idx], all_sessions[split_idx:]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _make_output_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "charts": root / "charts",
        "notebooks": root / "notebooks",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _resolve_dataset_path(user_dataset: Path | None) -> Path:
    candidates = []
    if user_dataset is not None:
        candidates.append(user_dataset)

    candidates.extend(
        [
            REPO_ROOT / "data" / "dowloaded" / "MNQ_c_0_1m_20260321_094501.parquet",
            REPO_ROOT / "data" / "downloaded" / "MNQ_c_0_1m_20260321_094501.parquet",
            REPO_ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet",
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError("No MNQ minute dataset found. Pass --dataset explicitly.")


def _equity_checksum(trades: pd.DataFrame, initial_capital: float) -> str:
    curve = build_equity_curve(trades, initial_capital=initial_capital)
    if curve.empty:
        return "empty"
    payload = np.round(curve["equity"].to_numpy(dtype=float), 6).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _serialize_config(config: ExperimentConfig) -> str:
    return json.dumps(
        {
            "name": config.name,
            "stage": config.stage,
            "family": config.family,
            "baseline_entry": asdict(config.baseline_entry),
            "baseline_ensemble": asdict(config.baseline_ensemble),
            "compression": asdict(config.compression),
            "exit": asdict(config.exit),
            "dynamic_threshold": asdict(config.dynamic_threshold),
        },
        sort_keys=True,
    )


def _experiment_from_json(payload: str) -> ExperimentConfig:
    data = json.loads(payload)
    return ExperimentConfig(
        name=data["name"],
        stage=data["stage"],
        family=data["family"],
        baseline_entry=BaselineEntryConfig(**data["baseline_entry"]),
        baseline_ensemble=BaselineEnsembleConfig(**data["baseline_ensemble"]),
        compression=CompressionConfig(**data["compression"]),
        exit=ExitConfig(**data["exit"]),
        dynamic_threshold=DynamicThresholdConfig(**data["dynamic_threshold"]),
    )


def _evaluate_experiment(
    experiment: ExperimentConfig,
    context: CampaignContext,
    bootstrap_paths: int,
    random_seed: int,
    keep_details: bool = False,
    max_leverage: float | None = None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    candidate_df = context.candidate_base_df
    ensemble = experiment.baseline_ensemble
    atr_col = f"atr_{ensemble.atr_window}"

    if atr_col not in candidate_df.columns:
        row = {
            "name": experiment.name,
            "stage": experiment.stage,
            "family": experiment.family,
            "config_json": _serialize_config(experiment),
            "status": "missing_atr_column",
        }
        return row, None

    comp_mask = compression_mask(candidate_df, experiment.compression)
    noise_sigma = None
    dyn_cfg = experiment.dynamic_threshold
    if dyn_cfg.mode in {
        "noise_area_gate",
        "noise_area_gate_plus_close_confirmation",
        "noise_area_gate_plus_discrete_schedule",
    }:
        if dyn_cfg.noise_lookback not in context.noise_cache:
            context.noise_cache[dyn_cfg.noise_lookback] = compute_noise_sigma(context.minute_df, dyn_cfg.noise_lookback)
        noise_sigma = context.noise_cache[dyn_cfg.noise_lookback]

    dyn_mask = dynamic_gate_mask(
        candidate_df=candidate_df,
        config=dyn_cfg,
        noise_sigma=noise_sigma,
        atr_col=atr_col,
    )

    pass_mask = candidate_df["candidate_base_pass"].fillna(False).astype(bool)
    if experiment.compression.usage == "hard_filter":
        pass_mask &= comp_mask.fillna(False).astype(bool)
    pass_mask &= dyn_mask.fillna(False).astype(bool)

    selected_pre_ensemble = first_pass_signal_rows(candidate_df, pass_mask)
    thresholds = calibrate_ensemble_thresholds(
        selected_signal_rows=selected_pre_ensemble,
        is_sessions=context.is_sessions,
        atr_col=atr_col,
        q_lows_pct=ensemble.q_lows_pct,
        q_highs_pct=ensemble.q_highs_pct,
    )

    scored = apply_ensemble_selection(
        selected_signal_rows=selected_pre_ensemble,
        atr_col=atr_col,
        thresholds=thresholds,
        vote_threshold=ensemble.vote_threshold,
        compression_config=experiment.compression,
    )
    selected_final = scored.loc[scored.get("ensemble_selected", False)].copy()

    signal_df = build_signal_frame_for_backtest(context.minute_df, selected_final)
    trades = run_exit_variant_backtest(
        signal_df=signal_df,
        execution_model=ExecutionModel(),
        baseline=experiment.baseline_entry,
        exit_cfg=experiment.exit,
        max_leverage=max_leverage,
    )

    overall = compute_extended_metrics(
        trades=trades,
        signal_df=None,
        sessions=context.all_sessions,
        initial_capital=experiment.baseline_entry.account_size_usd,
        bootstrap_paths=bootstrap_paths,
        random_seed=random_seed,
    )

    is_set = set(pd.to_datetime(pd.Index(context.is_sessions)).date)
    oos_set = set(pd.to_datetime(pd.Index(context.oos_sessions)).date)
    trade_dates = pd.to_datetime(trades["session_date"]).dt.date if not trades.empty else pd.Series(dtype=object)
    trades_is = trades.loc[trade_dates.isin(is_set)].copy() if not trades.empty else trades.copy()
    trades_oos = trades.loc[trade_dates.isin(oos_set)].copy() if not trades.empty else trades.copy()

    metrics_is = compute_extended_metrics(
        trades=trades_is,
        signal_df=None,
        sessions=context.is_sessions,
        initial_capital=experiment.baseline_entry.account_size_usd,
        bootstrap_paths=max(300, bootstrap_paths // 2),
        random_seed=random_seed,
    )
    metrics_oos = compute_extended_metrics(
        trades=trades_oos,
        signal_df=None,
        sessions=context.oos_sessions,
        initial_capital=experiment.baseline_entry.account_size_usd,
        bootstrap_paths=max(300, bootstrap_paths // 2),
        random_seed=random_seed + 17,
    )

    score_overall = compute_scores(pd.Series(overall))
    score_is = compute_scores(pd.Series(metrics_is))
    score_oos = compute_scores(pd.Series(metrics_oos))

    row: dict[str, object] = {
        "name": experiment.name,
        "stage": experiment.stage,
        "family": experiment.family,
        "config_json": _serialize_config(experiment),
        "status": "ok",
        "candidate_rows_raw": int(candidate_df["candidate_base_pass"].sum()),
        "candidate_rows_after_overlays": int(pass_mask.sum()),
        "candidate_days_pre_ensemble": int(selected_pre_ensemble["session_date"].nunique()) if not selected_pre_ensemble.empty else 0,
        "selected_days": int(selected_final["session_date"].nunique()) if not selected_final.empty else 0,
        "atr_window": int(ensemble.atr_window),
        "ensemble_vote_threshold": float(ensemble.vote_threshold),
        "compression_mode": experiment.compression.mode,
        "compression_usage": experiment.compression.usage,
        "exit_mode": experiment.exit.mode,
        "dynamic_mode": experiment.dynamic_threshold.mode,
        "noise_lookback": int(experiment.dynamic_threshold.noise_lookback),
        "noise_vm": float(experiment.dynamic_threshold.noise_vm),
        "noise_k": float(experiment.dynamic_threshold.noise_k),
        "atr_k": float(experiment.dynamic_threshold.atr_k),
        "confirm_bars": int(experiment.dynamic_threshold.confirm_bars),
        "schedule": experiment.dynamic_threshold.schedule,
    }

    for prefix, metrics in (("overall", overall), ("is", metrics_is), ("oos", metrics_oos)):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value

    row.update(
        {
            "overall_academic_score": score_overall["academic_score"],
            "overall_prop_score": score_overall["prop_score"],
            "is_academic_score": score_is["academic_score"],
            "is_prop_score": score_is["prop_score"],
            "oos_academic_score": score_oos["academic_score"],
            "oos_prop_score": score_oos["prop_score"],
        }
    )

    detail = None
    if keep_details:
        detail = {
            "trades": trades,
            "selected_final": selected_final,
            "signal_df": signal_df,
        }
    return row, detail


def _build_dynamic_overlay_configs() -> list[DynamicThresholdConfig]:
    configs: list[DynamicThresholdConfig] = []

    for lookback in (10, 14, 20, 30):
        for vm in (0.75, 1.0, 1.25, 1.5, 1.75):
            configs.append(
                DynamicThresholdConfig(
                    mode="noise_area_gate",
                    noise_lookback=lookback,
                    noise_vm=vm,
                    threshold_style="max_or_high_noise",
                    noise_k=0.0,
                )
            )

    for lookback in (10, 14, 20, 30):
        for vm in (0.75, 1.0, 1.25, 1.5, 1.75):
            for k in (0.25, 0.5, 0.75, 1.0):
                configs.append(
                    DynamicThresholdConfig(
                        mode="noise_area_gate",
                        noise_lookback=lookback,
                        noise_vm=vm,
                        threshold_style="or_high_plus_k_noise_abs",
                        noise_k=k,
                    )
                )

    for k in (0.0, 0.25, 0.5, 0.75, 1.0):
        configs.append(DynamicThresholdConfig(mode="atr_threshold_gate", atr_k=k))

    for confirm_bars in (1, 2, 3):
        configs.append(DynamicThresholdConfig(mode="close_confirmation_gate", confirm_bars=confirm_bars))

    for lookback in (14, 20, 30):
        for vm in (1.0, 1.25, 1.5):
            for confirm_bars in (2, 3):
                configs.append(
                    DynamicThresholdConfig(
                        mode="noise_area_gate_plus_close_confirmation",
                        noise_lookback=lookback,
                        noise_vm=vm,
                        threshold_style="max_or_high_noise",
                        confirm_bars=confirm_bars,
                    )
                )

    for lookback in (14, 20, 30):
        for vm in (1.0, 1.25, 1.5):
            for schedule in ("every_5m", "every_15m"):
                configs.append(
                    DynamicThresholdConfig(
                        mode="noise_area_gate_plus_discrete_schedule",
                        noise_lookback=lookback,
                        noise_vm=vm,
                        threshold_style="max_or_high_noise",
                        schedule=schedule,
                    )
                )

    return configs


def _build_incremental_experiments(
    baseline_entry: BaselineEntryConfig,
    baseline_ensemble: BaselineEnsembleConfig,
) -> list[ExperimentConfig]:
    experiments: list[ExperimentConfig] = [
        ExperimentConfig(
            name="baseline_fixed",
            stage="incremental_overlay",
            family="baseline",
            baseline_entry=baseline_entry,
            baseline_ensemble=baseline_ensemble,
            compression=CompressionConfig(mode="none", usage="hard_filter"),
            exit=ExitConfig(mode="baseline"),
            dynamic_threshold=DynamicThresholdConfig(mode="disabled"),
        )
    ]

    compression_modes = [
        "nr4",
        "nr7",
        "triangle",
        "nr4_or_nr7",
        "nr4_or_triangle",
        "nr4_or_nr7_or_triangle",
        "inside_day",
        "outside_day",
        "strong_close",
        "weak_close",
    ]
    for mode in compression_modes:
        experiments.append(
            ExperimentConfig(
                name=f"compression__{mode}__hard",
                stage="incremental_overlay",
                family="compression",
                baseline_entry=baseline_entry,
                baseline_ensemble=baseline_ensemble,
                compression=CompressionConfig(mode=mode, usage="hard_filter"),
                exit=ExitConfig(mode="baseline"),
                dynamic_threshold=DynamicThresholdConfig(mode="disabled"),
            )
        )
        experiments.append(
            ExperimentConfig(
                name=f"compression__{mode}__soft",
                stage="incremental_overlay",
                family="compression",
                baseline_entry=baseline_entry,
                baseline_ensemble=baseline_ensemble,
                compression=CompressionConfig(mode=mode, usage="soft_vote_bonus", soft_bonus_votes=1.0),
                exit=ExitConfig(mode="baseline"),
                dynamic_threshold=DynamicThresholdConfig(mode="disabled"),
            )
        )

    exit_variants = [
        ExitConfig(mode="baseline"),
        ExitConfig(mode="fail_fast_vwap"),
        ExitConfig(mode="trailing_vwap"),
        ExitConfig(mode="trailing_struct_plus_vwap"),
        ExitConfig(mode="partial_1R_then_vwap"),
        ExitConfig(mode="time_stop_plus_vwap", force_exit_time="15:30:00", stagnation_bars=20),
    ]
    for exit_cfg in exit_variants:
        if exit_cfg.mode == "baseline":
            continue
        experiments.append(
            ExperimentConfig(
                name=f"exit__{exit_cfg.mode}",
                stage="incremental_overlay",
                family="vwap_exit",
                baseline_entry=baseline_entry,
                baseline_ensemble=baseline_ensemble,
                compression=CompressionConfig(mode="none", usage="hard_filter"),
                exit=exit_cfg,
                dynamic_threshold=DynamicThresholdConfig(mode="disabled"),
            )
        )

    for dyn_cfg in _build_dynamic_overlay_configs():
        dyn_name = (
            f"dynamic__{dyn_cfg.mode}__L{dyn_cfg.noise_lookback}__vm{str(dyn_cfg.noise_vm).replace('.', 'p')}"
            f"__k{str(dyn_cfg.noise_k).replace('.', 'p')}__atrk{str(dyn_cfg.atr_k).replace('.', 'p')}"
            f"__c{dyn_cfg.confirm_bars}__{dyn_cfg.schedule}"
        )
        experiments.append(
            ExperimentConfig(
                name=dyn_name,
                stage="incremental_overlay",
                family="dynamic_threshold",
                baseline_entry=baseline_entry,
                baseline_ensemble=baseline_ensemble,
                compression=CompressionConfig(mode="none", usage="hard_filter"),
                exit=ExitConfig(mode="baseline"),
                dynamic_threshold=dyn_cfg,
            )
        )

    return experiments


def _top_configs_by_family(results: pd.DataFrame, family: str, n: int = 3) -> list[ExperimentConfig]:
    subset = results.loc[(results["family"] == family) & (results["status"] == "ok")].copy()
    if subset.empty:
        return []
    subset = subset.sort_values(["oos_prop_score", "oos_net_pnl"], ascending=[False, False]).head(n)
    return [_experiment_from_json(payload) for payload in subset["config_json"].tolist()]


def _build_pairwise_experiments(
    baseline_entry: BaselineEntryConfig,
    baseline_ensemble: BaselineEnsembleConfig,
    incremental_results: pd.DataFrame,
) -> list[ExperimentConfig]:
    top_compression = _top_configs_by_family(incremental_results, "compression", n=3)
    top_exit = _top_configs_by_family(incremental_results, "vwap_exit", n=3)
    top_dynamic = _top_configs_by_family(incremental_results, "dynamic_threshold", n=3)

    experiments: list[ExperimentConfig] = []

    for comp_cfg in top_compression:
        for exit_cfg in top_exit:
            experiments.append(
                ExperimentConfig(
                    name=f"pair__comp_exit__{comp_cfg.compression.mode}__{exit_cfg.exit.mode}",
                    stage="pairwise_overlay",
                    family="compression_plus_vwap",
                    baseline_entry=baseline_entry,
                    baseline_ensemble=baseline_ensemble,
                    compression=comp_cfg.compression,
                    exit=exit_cfg.exit,
                    dynamic_threshold=DynamicThresholdConfig(mode="disabled"),
                )
            )

    for comp_cfg in top_compression:
        for dyn_cfg in top_dynamic:
            experiments.append(
                ExperimentConfig(
                    name=f"pair__comp_dynamic__{comp_cfg.compression.mode}__{dyn_cfg.dynamic_threshold.mode}",
                    stage="pairwise_overlay",
                    family="compression_plus_dynamic",
                    baseline_entry=baseline_entry,
                    baseline_ensemble=baseline_ensemble,
                    compression=comp_cfg.compression,
                    exit=ExitConfig(mode="baseline"),
                    dynamic_threshold=dyn_cfg.dynamic_threshold,
                )
            )

    for exit_cfg in top_exit:
        for dyn_cfg in top_dynamic:
            experiments.append(
                ExperimentConfig(
                    name=f"pair__exit_dynamic__{exit_cfg.exit.mode}__{dyn_cfg.dynamic_threshold.mode}",
                    stage="pairwise_overlay",
                    family="vwap_plus_dynamic",
                    baseline_entry=baseline_entry,
                    baseline_ensemble=baseline_ensemble,
                    compression=CompressionConfig(mode="none", usage="hard_filter"),
                    exit=exit_cfg.exit,
                    dynamic_threshold=dyn_cfg.dynamic_threshold,
                )
            )

    return experiments


def _build_full_reopt_experiments(
    baseline_entry: BaselineEntryConfig,
    incremental_results: pd.DataFrame,
    pairwise_results: pd.DataFrame,
    max_trials: int,
    random_seed: int,
) -> list[ExperimentConfig]:
    rng = np.random.default_rng(random_seed)

    atr_windows = [10, 14, 20, 30]
    q_low_sets = [(15, 20, 25), (20, 25, 30), (25, 30, 35)]
    q_high_sets = [(85, 90), (90, 95), (85, 90, 95)]
    vote_thresholds = [0.50, 0.67, 0.75]

    compression_modes = [
        "none",
        "nr4",
        "nr7",
        "triangle",
        "nr4_or_nr7",
        "nr4_or_nr7_or_triangle",
        "inside_day",
        "strong_close",
    ]
    exit_options = [
        ExitConfig(mode="baseline"),
        ExitConfig(mode="fail_fast_vwap"),
        ExitConfig(mode="trailing_vwap"),
        ExitConfig(mode="trailing_struct_plus_vwap"),
        ExitConfig(mode="partial_1R_then_vwap"),
        ExitConfig(mode="time_stop_plus_vwap", force_exit_time="15:30:00", stagnation_bars=20),
    ]
    dynamic_options = [
        DynamicThresholdConfig(mode="disabled"),
        DynamicThresholdConfig(mode="noise_area_gate", noise_lookback=10, noise_vm=1.0),
        DynamicThresholdConfig(mode="noise_area_gate", noise_lookback=14, noise_vm=1.0),
        DynamicThresholdConfig(mode="noise_area_gate", noise_lookback=20, noise_vm=1.25),
        DynamicThresholdConfig(mode="noise_area_gate", noise_lookback=30, noise_vm=1.5),
        DynamicThresholdConfig(mode="noise_area_gate", noise_lookback=14, noise_vm=1.0, threshold_style="or_high_plus_k_noise_abs", noise_k=0.5),
        DynamicThresholdConfig(mode="noise_area_gate", noise_lookback=20, noise_vm=1.25, threshold_style="or_high_plus_k_noise_abs", noise_k=1.0),
        DynamicThresholdConfig(mode="atr_threshold_gate", atr_k=0.25),
        DynamicThresholdConfig(mode="atr_threshold_gate", atr_k=0.5),
        DynamicThresholdConfig(mode="atr_threshold_gate", atr_k=0.75),
        DynamicThresholdConfig(mode="close_confirmation_gate", confirm_bars=2),
        DynamicThresholdConfig(mode="close_confirmation_gate", confirm_bars=3),
        DynamicThresholdConfig(mode="noise_area_gate_plus_close_confirmation", noise_lookback=14, noise_vm=1.25, confirm_bars=2),
        DynamicThresholdConfig(mode="noise_area_gate_plus_close_confirmation", noise_lookback=20, noise_vm=1.5, confirm_bars=3),
        DynamicThresholdConfig(mode="noise_area_gate_plus_discrete_schedule", noise_lookback=14, noise_vm=1.25, schedule="every_5m"),
        DynamicThresholdConfig(mode="noise_area_gate_plus_discrete_schedule", noise_lookback=20, noise_vm=1.5, schedule="every_15m"),
    ]

    seeds: list[ExperimentConfig] = []
    for df in (incremental_results, pairwise_results):
        if df.empty:
            continue
        top = df.loc[df["status"] == "ok"].sort_values("oos_prop_score", ascending=False).head(12)
        seeds.extend([_experiment_from_json(payload) for payload in top["config_json"].tolist()])

    experiments: list[ExperimentConfig] = []
    seen: set[str] = set()

    def _append(exp: ExperimentConfig) -> None:
        key = _serialize_config(exp)
        if key in seen:
            return
        seen.add(key)
        experiments.append(exp)

    baseline = ExperimentConfig(
        name="full_reopt__baseline_ref",
        stage="full_reopt",
        family="full_reopt",
        baseline_entry=baseline_entry,
        baseline_ensemble=BaselineEnsembleConfig(
            atr_window=14,
            q_lows_pct=(20, 25, 30),
            q_highs_pct=(90, 95),
            vote_threshold=0.50,
        ),
        compression=CompressionConfig(mode="none", usage="hard_filter"),
        exit=ExitConfig(mode="baseline"),
        dynamic_threshold=DynamicThresholdConfig(mode="disabled"),
    )
    _append(baseline)

    for seed in seeds:
        _append(replace(seed, stage="full_reopt", family="full_reopt", name=f"full_reopt__seed__{seed.name}"))

    target_trials = max(50, int(max_trials))
    while len(experiments) < target_trials:
        compression_mode = str(rng.choice(compression_modes))
        compression_usage = "hard_filter"
        if compression_mode != "none" and float(rng.random()) < 0.45:
            compression_usage = "soft_vote_bonus"

        ensemble = BaselineEnsembleConfig(
            atr_window=int(rng.choice(atr_windows)),
            q_lows_pct=tuple(int(x) for x in rng.choice(np.array(q_low_sets, dtype=object))),
            q_highs_pct=tuple(int(x) for x in rng.choice(np.array(q_high_sets, dtype=object))),
            vote_threshold=float(rng.choice(vote_thresholds)),
        )
        exit_cfg = rng.choice(np.array(exit_options, dtype=object))
        dyn_cfg = rng.choice(np.array(dynamic_options, dtype=object))

        exp = ExperimentConfig(
            name=f"full_reopt__trial_{len(experiments)+1:04d}",
            stage="full_reopt",
            family="full_reopt",
            baseline_entry=baseline_entry,
            baseline_ensemble=ensemble,
            compression=CompressionConfig(mode=compression_mode, usage=compression_usage, soft_bonus_votes=1.0),
            exit=exit_cfg,
            dynamic_threshold=dyn_cfg,
        )
        _append(exp)

    return experiments[:target_trials]


def _run_experiments(
    experiments: list[ExperimentConfig],
    context: CampaignContext,
    bootstrap_paths: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i, experiment in enumerate(experiments, start=1):
        row, _ = _evaluate_experiment(
            experiment=experiment,
            context=context,
            bootstrap_paths=bootstrap_paths,
            random_seed=random_seed + i,
            keep_details=False,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _legacy_baseline_trades(dataset_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    legacy_base = LegacyBaselineSpec()
    feat = legacy_prepare_feature_dataset(dataset_path=dataset_path, baseline=legacy_base, atr_period=14)
    strategy = legacy_build_baseline_strategy(legacy_base)
    signal_df = strategy.generate_signals(feat)
    signal_rows = legacy_selected_signal_rows(signal_df, atr_col="atr_14")

    sessions = sorted(pd.to_datetime(feat["session_date"]).dt.date.unique())
    is_sessions, _ = _split_sessions(sessions, 0.70)
    zones = legacy_build_zone_definitions(include_expanded=True, q_low_anchor_pct=20)
    submodels_df, signal_scores, point_id, point_secondary_id = legacy_build_submodels(
        signal_rows=signal_rows,
        is_sessions=is_sessions,
        atr_col="atr_14",
        atr_period=14,
        zones=zones,
        point_q_low_pct=20.0,
        point_q_high_pct=90.0,
    )
    _ = submodels_df, point_id, point_secondary_id

    variants = legacy_build_variant_configs(
        submodels_df=submodels_df,
        zones=zones,
        point_model_id=point_id,
        point_secondary_model_id=point_secondary_id,
    )
    baseline_variant = variants.loc[
        variants["strategy_id"] == "ensemble__expanded_q20_25_30__q90_95__majority_50"
    ]
    if baseline_variant.empty:
        baseline_variant = variants.loc[variants["strategy_id"] == "baseline_no_atr"]
    variant = baseline_variant.iloc[0]

    selected_indices = legacy_select_signal_indices(variant=variant, signal_scores=signal_scores)
    variant_signal_df = legacy_signal_df_for_selected_indices(signal_df, selected_indices)
    trades = legacy_run_backtest(variant_signal_df, legacy_base, ExecutionModel())

    info = {
        "strategy_id": str(variant["strategy_id"]),
        "selected_days": int(len(selected_indices)),
    }
    return trades, info


def _new_baseline_experiment(
    baseline_entry: BaselineEntryConfig,
    baseline_ensemble: BaselineEnsembleConfig,
) -> ExperimentConfig:
    return ExperimentConfig(
        name="baseline_integrity_new_pipeline",
        stage="integrity",
        family="baseline",
        baseline_entry=baseline_entry,
        baseline_ensemble=baseline_ensemble,
        compression=CompressionConfig(mode="none", usage="hard_filter"),
        exit=ExitConfig(mode="baseline"),
        dynamic_threshold=DynamicThresholdConfig(mode="disabled"),
    )


def _write_baseline_integrity_report(
    output_path: Path,
    dataset_path: Path,
    reference_trades: pd.DataFrame,
    candidate_trades: pd.DataFrame,
    reference_info: dict[str, object],
    tolerance: float = 1e-6,
) -> None:
    from src.analytics.metrics import compute_metrics

    all_sessions = sorted(
        pd.to_datetime(
            pd.concat(
                [
                    reference_trades.get("session_date", pd.Series(dtype=object)),
                    candidate_trades.get("session_date", pd.Series(dtype=object)),
                ],
                ignore_index=True,
            ),
            errors="coerce",
        )
        .dropna()
        .dt.date.unique()
    )
    ref_metrics = compute_metrics(reference_trades, session_dates=all_sessions, initial_capital=50_000.0)
    new_metrics = compute_metrics(candidate_trades, session_dates=all_sessions, initial_capital=50_000.0)

    keys = ["n_trades", "cumulative_pnl", "sharpe_ratio", "profit_factor", "expectancy", "max_drawdown"]
    rows = []
    for key in keys:
        ref_v = _safe_float(ref_metrics.get(key), 0.0)
        new_v = _safe_float(new_metrics.get(key), 0.0)
        diff = new_v - ref_v
        rel = diff / max(abs(ref_v), 1e-9)
        ok = abs(diff) <= tolerance * max(1.0, abs(ref_v))
        rows.append((key, ref_v, new_v, diff, rel, ok))

    ref_checksum = _equity_checksum(reference_trades, 50_000.0)
    new_checksum = _equity_checksum(candidate_trades, 50_000.0)
    checksum_ok = ref_checksum == new_checksum

    lines = [
        "# Baseline Integrity Report",
        "",
        "## Baseline Mapping",
        "",
        "- Data loading: `src/data/loader.py` -> `load_ohlcv_file`.",
        "- Feature engineering: `src/features/intraday.py`, `src/features/opening_range.py`, `src/features/volatility.py`.",
        "- ORB generation: `src/strategy/orb.py` (long-only + VWAP confirmation).",
        "- ATR ensemble filter: `src/analytics/atr_ensemble_campaign.py` (cross-product quantiles + vote threshold).",
        "- Backtest engine: `src/engine/backtester.py` (costs/slippage/risk sizing).",
        "- Metrics: `src/analytics/metrics.py`.",
        "- Exports: campaign outputs under `export/orb_research_campaign/`.",
        "",
        "## Reference",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Legacy baseline strategy id: `{reference_info.get('strategy_id')}`",
        f"- Legacy selected days: {reference_info.get('selected_days')}",
        "",
        "## Numerical Non-Regression",
        "",
        "| metric | reference | new_pipeline | diff | rel_diff | pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]

    for key, ref_v, new_v, diff, rel, ok in rows:
        lines.append(f"| {key} | {ref_v:.8f} | {new_v:.8f} | {diff:.8f} | {rel:.6%} | {'yes' if ok else 'no'} |")

    lines.extend(
        [
            "",
            "## Equity Checksum",
            "",
            f"- reference checksum: `{ref_checksum}`",
            f"- new checksum: `{new_checksum}`",
            f"- checksum pass: `{'yes' if checksum_ok else 'no'}`",
            "",
            "## Integrity Verdict",
            "",
            (
                "- Baseline preserved within tolerance."
                if all(ok for *_, ok in rows) and checksum_ok
                else "- Baseline mismatch detected (see table/checksum)."
            ),
        ]
    )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _plot_equity_and_drawdown(
    experiments_df: pd.DataFrame,
    context: CampaignContext,
    baseline_entry: BaselineEntryConfig,
    output_charts: Path,
    n_top: int = 5,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    top = experiments_df.loc[experiments_df["status"] == "ok"].sort_values(
        ["oos_prop_score", "oos_net_pnl"], ascending=[False, False]
    ).head(n_top)

    curves: dict[str, pd.DataFrame] = {}
    for _, row in top.iterrows():
        exp = _experiment_from_json(row["config_json"])
        _, detail = _evaluate_experiment(
            experiment=exp,
            context=context,
            bootstrap_paths=800,
            random_seed=123,
            keep_details=True,
        )
        trades = detail["trades"] if detail is not None else pd.DataFrame()
        curve = build_equity_curve(trades, initial_capital=baseline_entry.account_size_usd)
        curves[row["name"]] = curve

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name, curve in curves.items():
        if curve.empty:
            continue
        axes[0].plot(curve["timestamp"], curve["equity"], linewidth=1.3, label=name)
        axes[1].plot(curve["timestamp"], curve["drawdown"], linewidth=1.3, label=name)

    axes[0].set_title("Top Equity Curves")
    axes[0].set_ylabel("Equity (USD)")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_title("Top Drawdowns")
    axes[1].set_ylabel("Drawdown (USD)")
    axes[1].set_xlabel("Time")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_charts / "equity_curves_top.png", dpi=150)
    fig.savefig(output_charts / "drawdown_comparison.png", dpi=150)
    plt.close(fig)

    for name, curve in curves.items():
        if curve.empty:
            continue
        curve.to_csv(output_charts / f"equity_curve__{name}.csv", index=False)

    return top, curves


def _plot_heatmap(df: pd.DataFrame, index_col: str, col_col: str, value_col: str, out_path: Path, title: str) -> None:
    pivot = df.pivot_table(index=index_col, columns=col_col, values=value_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    if pivot.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
    else:
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(x) for x in pivot.columns], rotation=35, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(x) for x in pivot.index])
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_rolling_oos_stability(
    top_rows: pd.DataFrame,
    context: CampaignContext,
    out_path: Path,
    rolling_window: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.5))

    oos_idx = pd.Index(pd.to_datetime(pd.Index(context.oos_sessions)).date)
    for _, row in top_rows.head(4).iterrows():
        exp = _experiment_from_json(row["config_json"])
        _, detail = _evaluate_experiment(
            experiment=exp,
            context=context,
            bootstrap_paths=600,
            random_seed=777,
            keep_details=True,
        )
        trades = detail["trades"] if detail is not None else pd.DataFrame()
        daily = pd.Series(0.0, index=oos_idx, dtype=float)
        if not trades.empty:
            grouped = trades.groupby(pd.to_datetime(trades["session_date"]).dt.date)["net_pnl_usd"].sum()
            daily = daily.add(grouped.reindex(oos_idx, fill_value=0.0), fill_value=0.0)
        rolling = daily.rolling(max(5, int(rolling_window))).sum()
        ax.plot(oos_idx, rolling, linewidth=1.2, label=row["name"])

    ax.set_title("Rolling OOS Stability (Rolling PnL)")
    ax.set_ylabel("Rolling PnL (USD)")
    ax.set_xlabel("Session")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_trade_distribution(top_df: pd.DataFrame, out_path: Path) -> None:
    view = top_df[["name", "oos_hit_ratio", "oos_avg_winner", "oos_avg_loser", "oos_expectancy"]].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    metrics = [
        ("oos_hit_ratio", "Hit Ratio"),
        ("oos_avg_winner", "Avg Winner"),
        ("oos_avg_loser", "Avg Loser"),
        ("oos_expectancy", "Expectancy"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(view["name"], view[col])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_parameter_clusters(full_reopt: pd.DataFrame, out_path: Path) -> None:
    view = full_reopt.loc[(full_reopt["status"] == "ok")].copy()
    if view.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No full reopt data", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    sc = ax.scatter(
        view["oos_return_over_drawdown"],
        view["oos_max_drawdown"].abs(),
        c=view["oos_prop_score"],
        cmap="plasma",
        alpha=0.75,
        s=28,
    )
    ax.set_xlabel("OOS Return/Drawdown")
    ax.set_ylabel("OOS |Max Drawdown|")
    ax.set_title("Parameter Landscape (Full Re-opt)")
    fig.colorbar(sc, ax=ax, label="OOS Prop Score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _build_robustness_summary(full_reopt: pd.DataFrame) -> pd.DataFrame:
    view = full_reopt.loc[(full_reopt["status"] == "ok")].copy()
    if view.empty:
        return pd.DataFrame()

    threshold = view["oos_prop_score"].quantile(0.70)
    top = view.loc[view["oos_prop_score"] >= threshold].copy()
    rows: list[dict[str, object]] = []

    def _append_group(group_name: str, col: str) -> None:
        for value, frame in top.groupby(col, sort=True):
            rows.append(
                {
                    "group": group_name,
                    "value": value,
                    "n": int(len(frame)),
                    "median_oos_prop_score": float(frame["oos_prop_score"].median()),
                    "median_oos_net_pnl": float(frame["oos_net_pnl"].median()),
                    "median_oos_max_dd": float(frame["oos_max_drawdown"].median()),
                    "median_oos_return_over_dd": float(frame["oos_return_over_drawdown"].median()),
                }
            )

    _append_group("atr_window", "atr_window")
    _append_group("vote_threshold", "ensemble_vote_threshold")
    _append_group("compression_mode", "compression_mode")
    _append_group("exit_mode", "exit_mode")
    _append_group("dynamic_mode", "dynamic_mode")

    return pd.DataFrame(rows).sort_values(["group", "median_oos_prop_score"], ascending=[True, False])


def _write_parameter_cluster_summary(path: Path, robustness_df: pd.DataFrame) -> None:
    lines = [
        "# Parameter Cluster Summary",
        "",
        "Clusters are built from the top 30% full re-opt configurations by OOS prop score.",
        "",
    ]
    if robustness_df.empty:
        lines.append("No robust cluster could be computed.")
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return

    for group, group_df in robustness_df.groupby("group", sort=True):
        best = group_df.iloc[0]
        lines.extend(
            [
                f"## {group}",
                "",
                f"- Best value: `{best['value']}`",
                f"- Median OOS prop score: {best['median_oos_prop_score']:.4f}",
                f"- Median OOS net pnl: {best['median_oos_net_pnl']:.2f}",
                f"- Median OOS max drawdown: {best['median_oos_max_dd']:.2f}",
                "",
            ]
        )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _report_effect_section(
    name: str,
    overlay_df: pd.DataFrame,
    full_df: pd.DataFrame,
    active_filter,
) -> list[str]:
    lines = [f"## {name}", ""]
    base_overlay = overlay_df.loc[overlay_df["family"] == "baseline"].head(1)
    base_row = base_overlay.iloc[0] if not base_overlay.empty else None

    overlay_active = overlay_df.loc[active_filter(overlay_df)].copy()
    overlay_best = overlay_active.sort_values("oos_prop_score", ascending=False).head(1)

    full_active = full_df.loc[active_filter(full_df)].copy()
    full_inactive = full_df.loc[~active_filter(full_df)].copy()
    full_best_active = full_active.sort_values("oos_prop_score", ascending=False).head(1)
    full_best_inactive = full_inactive.sort_values("oos_prop_score", ascending=False).head(1)

    lines.append("### Effet marginal pur")
    if base_row is None or overlay_best.empty:
        lines.append("- Donnees insuffisantes pour l'overlay pur.")
    else:
        b = base_row
        o = overlay_best.iloc[0]
        lines.append(
            f"- Baseline fixe OOS prop_score={b['oos_prop_score']:.4f}, net_pnl={b['oos_net_pnl']:.2f}, maxDD={b['oos_max_drawdown']:.2f}."
        )
        lines.append(
            f"- Meilleur overlay OOS: `{o['name']}` avec prop_score={o['oos_prop_score']:.4f}, net_pnl={o['oos_net_pnl']:.2f}, maxDD={o['oos_max_drawdown']:.2f}."
        )

    lines.append("")
    lines.append("### Effet apres re-optimisation")
    if full_best_active.empty or full_best_inactive.empty:
        lines.append("- Donnees insuffisantes pour comparer actif vs inactif en full re-opt.")
    else:
        a = full_best_active.iloc[0]
        n = full_best_inactive.iloc[0]
        lines.append(
            f"- Best actif: `{a['name']}` prop_score={a['oos_prop_score']:.4f}, net_pnl={a['oos_net_pnl']:.2f}, maxDD={a['oos_max_drawdown']:.2f}."
        )
        lines.append(
            f"- Best inactif: `{n['name']}` prop_score={n['oos_prop_score']:.4f}, net_pnl={n['oos_net_pnl']:.2f}, maxDD={n['oos_max_drawdown']:.2f}."
        )

    lines.append("")
    lines.append("### Conclusion robuste ou non")
    if full_best_active.empty or full_best_inactive.empty:
        lines.append("- Conclusion: evidence insuffisante.")
    else:
        a = full_best_active.iloc[0]
        n = full_best_inactive.iloc[0]
        robust = (
            float(a["oos_prop_score"]) > float(n["oos_prop_score"])
            and float(a["oos_net_pnl"]) >= float(n["oos_net_pnl"])
            and abs(float(a["oos_max_drawdown"])) <= 1.15 * abs(float(n["oos_max_drawdown"]))
        )
        lines.append(
            "- Conclusion: gain robuste detecte." if robust else "- Conclusion: gain non robuste ou principalement lie au deplacement d'optimum."
        )

    lines.append("")
    return lines


def _write_campaign_reports(
    output_root: Path,
    incremental: pd.DataFrame,
    full_reopt: pd.DataFrame,
    top_prop: pd.DataFrame,
    top_academic: pd.DataFrame,
) -> None:
    lines = [
        "# ORB Research Campaign Report",
        "",
        "This report enforces the required double reading:",
        "1) incremental overlay on frozen baseline,",
        "2) full re-optimization of baseline + new bricks.",
        "",
    ]

    lines.extend(
        _report_effect_section(
            "Compression Filter",
            incremental,
            full_reopt,
            lambda df: df["compression_mode"].ne("none"),
        )
    )
    lines.extend(
        _report_effect_section(
            "VWAP Exit / Trailing",
            incremental,
            full_reopt,
            lambda df: df["exit_mode"].ne("baseline"),
        )
    )
    lines.extend(
        _report_effect_section(
            "Dynamic Breakout Threshold",
            incremental,
            full_reopt,
            lambda df: df["dynamic_mode"].ne("disabled"),
        )
    )

    if not top_prop.empty:
        best = top_prop.iloc[0]
        lines.extend(
            [
                "## Final Answers",
                "",
                f"1. Compression en overlay pur: {'oui' if (incremental.loc[incremental['family']=='compression','oos_prop_score'].max() > incremental.loc[incremental['family']=='baseline','oos_prop_score'].max()) else 'non'}.",
                f"2. Compression apres full re-opt: {'oui' if (full_reopt.loc[full_reopt['compression_mode']!='none','oos_prop_score'].max() > full_reopt.loc[full_reopt['compression_mode']=='none','oos_prop_score'].max()) else 'non'}.",
                f"3. VWAP exits en overlay pur: {'oui' if (incremental.loc[incremental['family']=='vwap_exit','oos_prop_score'].max() > incremental.loc[incremental['family']=='baseline','oos_prop_score'].max()) else 'non'}.",
                f"4. VWAP exits apres full re-opt: {'oui' if (full_reopt.loc[full_reopt['exit_mode']!='baseline','oos_prop_score'].max() > full_reopt.loc[full_reopt['exit_mode']=='baseline','oos_prop_score'].max()) else 'non'}.",
                f"5. Dynamic threshold en overlay pur: {'oui' if (incremental.loc[incremental['family']=='dynamic_threshold','oos_prop_score'].max() > incremental.loc[incremental['family']=='baseline','oos_prop_score'].max()) else 'non'}.",
                f"6. Dynamic threshold apres full re-opt: {'oui' if (full_reopt.loc[full_reopt['dynamic_mode']!='disabled','oos_prop_score'].max() > full_reopt.loc[full_reopt['dynamic_mode']=='disabled','oos_prop_score'].max()) else 'non'}.",
                "7. Les gains persistent-ils apres re-optimisation complete: voir sections ci-dessus.",
                "8. Clusters robustes ou pics isoles: voir `parameter_cluster_summary.md`.",
                f"9. Config robuste recommandee: `{best['name']}`.",
                f"10. Config prop-firm oriented recommandee: `{best['name']}` (classement par prop_score).",
                "",
            ]
        )

    (output_root / "campaign_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    prop_lines = [
        "# ORB Campaign Report (Prop Firm Focus)",
        "",
        "Ranking priority: prop_score first, academic_score secondary.",
        "",
        "## Top Prop Configs",
        "",
        "```text",
        top_prop.head(15).to_string(index=False) if not top_prop.empty else "No data",
        "```",
        "",
        "## Top Academic Configs",
        "",
        "```text",
        top_academic.head(15).to_string(index=False) if not top_academic.empty else "No data",
        "```",
        "",
    ]
    (output_root / "campaign_report_prop_firm.md").write_text("\n".join(prop_lines).rstrip() + "\n", encoding="utf-8")


def _write_notebook(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _cell(cell_type: str, source: str) -> dict[str, object]:
        if not source.endswith("\n"):
            source += "\n"
        return {
            "cell_type": cell_type,
            "metadata": {},
            "source": source.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
        }

    nb = {
        "cells": [
            _cell("markdown", "# ORB Research Campaign Review\n\nNotebook de revue visuelle des artefacts exportes."),
            _cell(
                "code",
                "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\n\nroot = Path.cwd()\noutput_dir = root / 'export' / 'orb_research_campaign'\nprint(output_dir)",
            ),
            _cell(
                "code",
                "incremental = pd.read_csv(output_dir / 'incremental_overlay_results.csv')\npairwise = pd.read_csv(output_dir / 'pairwise_overlay_results.csv')\nfull = pd.read_csv(output_dir / 'full_reopt_results.csv')\nprop_top = pd.read_csv(output_dir / 'top_configs_prop_score.csv')\nacad_top = pd.read_csv(output_dir / 'top_configs_academic_score.csv')\ndisplay(prop_top.head(10))\ndisplay(acad_top.head(10))",
            ),
            _cell(
                "code",
                "charts = output_dir / 'charts'\nfor name in [\n    'equity_curves_top.png',\n    'heatmap_compression.png',\n    'heatmap_vwap_exit.png',\n    'heatmap_dynamic_threshold.png',\n    'heatmap_full_reopt.png',\n    'rolling_oos_stability.png',\n    'drawdown_comparison.png',\n    'trade_distribution_top.png',\n    'parameter_clusters.png',\n    'prop_firm_success_curves.png',\n    'worst_5day_dd_comparison.png',\n]:\n    p = charts / name\n    if p.exists():\n        display(Image(filename=str(p)))",
            ),
            _cell(
                "markdown",
                "## Conclusion visuelle attendue\n\n1. Ce qui ameliore reellement (overlay pur + full re-opt).\n2. Ce qui n'apporte rien.\n3. Ce qui est fragile.\n4. Meilleure config robuste.\n5. Meilleure config prop-firm oriented.",
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")


def _plot_prop_firm_success_curves(
    top_rows: pd.DataFrame,
    context: CampaignContext,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    oos_dates = pd.Index(pd.to_datetime(pd.Index(context.oos_sessions)).date)

    def _success_rate_by_horizon(daily: np.ndarray, target: float, dd_limit: float, max_horizon: int = 120) -> tuple[np.ndarray, np.ndarray]:
        horizons = np.arange(5, max_horizon + 1)
        rates = []
        for h in horizons:
            success = 0
            total = 0
            for start in range(len(daily)):
                path = daily[start : start + h]
                if len(path) < 5:
                    continue
                total += 1
                cum = 0.0
                for value in path:
                    cum += float(value)
                    if cum >= target:
                        success += 1
                        break
                    if cum <= -dd_limit:
                        break
            rates.append(success / total if total > 0 else 0.0)
        return horizons, np.array(rates, dtype=float)

    for _, row in top_rows.head(3).iterrows():
        exp = _experiment_from_json(row["config_json"])
        _, detail = _evaluate_experiment(
            experiment=exp,
            context=context,
            bootstrap_paths=400,
            random_seed=4242,
            keep_details=True,
        )
        trades = detail["trades"] if detail is not None else pd.DataFrame()
        daily = pd.Series(0.0, index=oos_dates, dtype=float)
        if not trades.empty:
            grouped = trades.groupby(pd.to_datetime(trades["session_date"]).dt.date)["net_pnl_usd"].sum()
            daily = daily.add(grouped.reindex(oos_dates, fill_value=0.0), fill_value=0.0)
        horizon, rates = _success_rate_by_horizon(
            daily=daily.to_numpy(dtype=float),
            target=0.06 * exp.baseline_entry.account_size_usd,
            dd_limit=0.04 * exp.baseline_entry.account_size_usd,
            max_horizon=min(120, max(20, len(daily))),
        )
        ax.plot(horizon, rates, linewidth=1.4, label=row["name"])

    ax.set_title("Prop-Firm Success Probability vs Horizon")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("P(success before -4% DD)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_worst_5day(top_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(top_df["name"], top_df["oos_worst_5day_drawdown"])
    ax.set_title("Worst 5-Day Drawdown (OOS)")
    ax.set_ylabel("USD")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_orb_research_campaign(config: CampaignConfig) -> dict[str, Path]:
    ensure_directories()
    dataset_path = _resolve_dataset_path(config.dataset_path)
    dirs = _make_output_dirs(config.output_dir)

    baseline_entry = BaselineEntryConfig()
    baseline_ensemble = BaselineEnsembleConfig()

    atr_windows = sorted({10, 14, 20, 30})
    minute_df = prepare_minute_dataset(dataset_path=dataset_path, baseline_entry=baseline_entry, atr_windows=atr_windows)
    daily_df = build_daily_reference(minute_df)
    minute_df = attach_daily_reference(minute_df, daily_df)

    candidate_base = build_candidate_universe(minute_df, baseline_entry=baseline_entry)

    all_sessions = sorted(pd.to_datetime(minute_df["session_date"]).dt.date.unique())
    is_sessions, oos_sessions = _split_sessions(all_sessions, config.is_fraction)
    context = CampaignContext(
        all_sessions=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        minute_df=minute_df,
        candidate_base_df=candidate_base,
        daily_patterns=daily_df,
    )

    reference_trades, reference_info = _legacy_baseline_trades(dataset_path)
    baseline_exp = _new_baseline_experiment(baseline_entry, baseline_ensemble)
    _, baseline_detail = _evaluate_experiment(
        experiment=baseline_exp,
        context=context,
        bootstrap_paths=1000,
        random_seed=config.random_seed,
        keep_details=True,
    )
    baseline_new_trades = baseline_detail["trades"] if baseline_detail is not None else pd.DataFrame()

    _write_baseline_integrity_report(
        output_path=config.output_dir / "baseline_integrity_report.md",
        dataset_path=dataset_path,
        reference_trades=reference_trades,
        candidate_trades=baseline_new_trades,
        reference_info=reference_info,
    )

    incremental_path = config.output_dir / "incremental_overlay_results.csv"
    if incremental_path.exists():
        incremental_df = pd.read_csv(incremental_path)
    else:
        incremental_experiments = _build_incremental_experiments(
            baseline_entry=baseline_entry,
            baseline_ensemble=baseline_ensemble,
        )
        incremental_df = _run_experiments(
            experiments=incremental_experiments,
            context=context,
            bootstrap_paths=config.bootstrap_paths,
            random_seed=config.random_seed,
        )
        incremental_df.to_csv(incremental_path, index=False)

    pairwise_path = config.output_dir / "pairwise_overlay_results.csv"
    if pairwise_path.exists():
        pairwise_df = pd.read_csv(pairwise_path)
    else:
        pairwise_experiments = _build_pairwise_experiments(
            baseline_entry=baseline_entry,
            baseline_ensemble=baseline_ensemble,
            incremental_results=incremental_df,
        )
        pairwise_df = _run_experiments(
            experiments=pairwise_experiments,
            context=context,
            bootstrap_paths=max(1000, config.bootstrap_paths // 2),
            random_seed=config.random_seed + 1000,
        )
        pairwise_df.to_csv(pairwise_path, index=False)

    full_reopt_path = config.output_dir / "full_reopt_results.csv"
    if full_reopt_path.exists():
        full_reopt_df = pd.read_csv(full_reopt_path)
    else:
        full_reopt_experiments = _build_full_reopt_experiments(
            baseline_entry=baseline_entry,
            incremental_results=incremental_df,
            pairwise_results=pairwise_df,
            max_trials=config.max_full_reopt_trials,
            random_seed=config.random_seed + 2000,
        )
        full_reopt_df = _run_experiments(
            experiments=full_reopt_experiments,
            context=context,
            bootstrap_paths=max(1000, config.bootstrap_paths // 2),
            random_seed=config.random_seed + 3000,
        )
        full_reopt_df.to_csv(full_reopt_path, index=False)

    all_results = pd.concat([incremental_df, pairwise_df, full_reopt_df], ignore_index=True)
    all_results.to_csv(config.output_dir / "experiment_registry.csv", index=False)

    valid = all_results.loc[all_results["status"] == "ok"].copy()
    top_oos = valid.sort_values(["oos_net_pnl", "oos_prop_score"], ascending=[False, False]).head(config.oos_top_n)
    top_prop = valid.sort_values(["oos_prop_score", "oos_net_pnl"], ascending=[False, False]).head(config.oos_top_n)
    top_academic = valid.sort_values(["oos_academic_score", "oos_net_pnl"], ascending=[False, False]).head(config.oos_top_n)

    top_oos.to_csv(config.output_dir / "top_configs_oos.csv", index=False)
    top_prop.to_csv(config.output_dir / "top_configs_prop_score.csv", index=False)
    top_academic.to_csv(config.output_dir / "top_configs_academic_score.csv", index=False)

    robustness = _build_robustness_summary(full_reopt_df)
    robustness.to_csv(config.output_dir / "robustness_summary.csv", index=False)
    _write_parameter_cluster_summary(config.output_dir / "parameter_cluster_summary.md", robustness)

    top_rows, _ = _plot_equity_and_drawdown(
        experiments_df=top_prop,
        context=context,
        baseline_entry=baseline_entry,
        output_charts=dirs["charts"],
        n_top=5,
    )

    compression_view = incremental_df.loc[incremental_df["family"] == "compression"].copy()
    if not compression_view.empty:
        _plot_heatmap(
            compression_view,
            index_col="compression_mode",
            col_col="compression_usage",
            value_col="oos_prop_score",
            out_path=dirs["charts"] / "heatmap_compression.png",
            title="Compression Overlay (OOS Prop Score)",
        )

    exit_view = incremental_df.loc[incremental_df["family"] == "vwap_exit"].copy()
    if not exit_view.empty:
        _plot_heatmap(
            exit_view,
            index_col="exit_mode",
            col_col="stage",
            value_col="oos_prop_score",
            out_path=dirs["charts"] / "heatmap_vwap_exit.png",
            title="VWAP Exit Overlay (OOS Prop Score)",
        )

    dyn_view = incremental_df.loc[incremental_df["family"] == "dynamic_threshold"].copy()
    if not dyn_view.empty:
        noise_only = dyn_view.loc[dyn_view["dynamic_mode"].str.contains("noise", na=False)].copy()
        if noise_only.empty:
            noise_only = dyn_view.copy()
        _plot_heatmap(
            noise_only,
            index_col="noise_lookback",
            col_col="noise_vm",
            value_col="oos_prop_score",
            out_path=dirs["charts"] / "heatmap_dynamic_threshold.png",
            title="Dynamic Threshold Heatmap (Noise Params)",
        )

    full_view = full_reopt_df.loc[full_reopt_df["status"] == "ok"].copy()
    if not full_view.empty:
        _plot_heatmap(
            full_view,
            index_col="atr_window",
            col_col="ensemble_vote_threshold",
            value_col="oos_prop_score",
            out_path=dirs["charts"] / "heatmap_full_reopt.png",
            title="Full Re-opt Heatmap (ATR Window x Vote Threshold)",
        )

    _plot_rolling_oos_stability(
        top_rows=top_prop,
        context=context,
        out_path=dirs["charts"] / "rolling_oos_stability.png",
        rolling_window=config.rolling_window_days_for_stability,
    )
    _plot_trade_distribution(top_prop.head(10), dirs["charts"] / "trade_distribution_top.png")
    _plot_parameter_clusters(full_reopt_df, dirs["charts"] / "parameter_clusters.png")
    _plot_prop_firm_success_curves(top_prop, context, dirs["charts"] / "prop_firm_success_curves.png")
    _plot_worst_5day(top_prop.head(10), dirs["charts"] / "worst_5day_dd_comparison.png")

    _write_campaign_reports(
        output_root=config.output_dir,
        incremental=incremental_df,
        full_reopt=full_reopt_df,
        top_prop=top_prop,
        top_academic=top_academic,
    )

    _write_notebook(path=dirs["notebooks"] / "orb_campaign_review.ipynb")

    return {
        "output_root": config.output_dir,
        "baseline_integrity_report": config.output_dir / "baseline_integrity_report.md",
        "experiment_registry": config.output_dir / "experiment_registry.csv",
        "incremental_overlay_results": config.output_dir / "incremental_overlay_results.csv",
        "pairwise_overlay_results": config.output_dir / "pairwise_overlay_results.csv",
        "full_reopt_results": config.output_dir / "full_reopt_results.csv",
        "robustness_summary": config.output_dir / "robustness_summary.csv",
        "top_oos": config.output_dir / "top_configs_oos.csv",
        "top_prop": config.output_dir / "top_configs_prop_score.csv",
        "top_academic": config.output_dir / "top_configs_academic_score.csv",
        "cluster_summary": config.output_dir / "parameter_cluster_summary.md",
        "campaign_report": config.output_dir / "campaign_report.md",
        "campaign_report_prop": config.output_dir / "campaign_report_prop_firm.md",
        "notebook": dirs["notebooks"] / "orb_campaign_review.ipynb",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ORB research campaign with compression/VWAP/dynamic thresholds.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "export" / "orb_research_campaign")
    parser.add_argument("--is-fraction", type=float, default=0.70)
    parser.add_argument("--max-full-reopt-trials", type=int, default=320)
    parser.add_argument("--bootstrap-paths", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = CampaignConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        is_fraction=args.is_fraction,
        random_seed=args.seed,
        max_full_reopt_trials=args.max_full_reopt_trials,
        bootstrap_paths=args.bootstrap_paths,
    )
    artifacts = run_orb_research_campaign(config)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
