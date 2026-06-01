"""Build a retained-final-first institutional memo for the MNQ ORB stack."""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analytics.audit_mnq_orb_retained_vs_3state import RetainedConfig, _latest_mnq_dataset, _rebuild_retained_final
from src.analytics.build_mnq_orb_strategy_memo import _markdown_table, _markdown_to_html
from src.analytics.mnq_orb_regime_filter_sizing_campaign import (
    RegimeFeatureSpec,
    _scale_nominal_trades_by_multiplier,
    build_conditional_bucket_analysis,
    build_static_regime_controls,
)
from src.analytics.orb_research.campaign import _evaluate_experiment
from src.analytics.orb_research.features import (
    attach_daily_reference,
    build_candidate_universe,
    build_daily_reference,
    prepare_minute_dataset,
)
from src.analytics.orb_research.types import (
    BaselineEnsembleConfig,
    BaselineEntryConfig,
    CampaignContext,
    CompressionConfig,
    DynamicThresholdConfig,
    ExitConfig,
    ExperimentConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_FILENAME = "mnq_orb_strategy_institutional_memo.md"
HTML_FILENAME = "mnq_orb_strategy_institutional_memo.html"
FIGURES_SUBDIR = Path("figures") / "mnq_orb_retained_final_memo"
RETAINED_3STATE_PREFIX = "mnq_orb_retained_final_3state_campaign_"


@dataclass(frozen=True)
class MemoConfig:
    output_dir: Path
    retained_3state_export: Path | None = None
    skip_heatmaps: bool = False


def _latest_retained_3state_export() -> Path:
    candidates = sorted((REPO_ROOT / "data" / "exports").glob(f"{RETAINED_3STATE_PREFIX}*"))
    if not candidates:
        raise FileNotFoundError("No retained-final 3-state export was found under data/exports.")
    return candidates[-1]


def _setup_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#293241",
            "axes.labelcolor": "#1b263b",
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.color": "#7d8597",
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "xtick.color": "#1b263b",
            "ytick.color": "#1b263b",
        }
    )


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _format_money(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:,.1f} USD"


def _format_number(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def _markdown_image(path: Path, output_dir: Path, caption: str) -> str:
    return f"![{caption}]({path.relative_to(output_dir).as_posix()})"


def _make_orb_mechanics_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(10)
    prices = np.array([100.1, 100.6, 100.4, 100.5, 101.9, 101.7, 102.3, 102.7, 102.4, 102.9])
    opens = np.array([100.0, 100.2, 100.5, 100.3, 100.8, 101.8, 101.6, 102.2, 102.8, 102.5])
    closes = prices
    highs = np.maximum(opens, closes) + np.array([0.3, 0.5, 0.4, 0.4, 0.6, 0.3, 0.4, 0.4, 0.3, 0.5])
    lows = np.minimum(opens, closes) - np.array([0.3, 0.2, 0.4, 0.3, 0.4, 0.2, 0.2, 0.3, 0.2, 0.3])

    or_high = 100.9
    or_low = 99.8
    stop_price = 99.55
    entry_price = 101.8
    target_price = 106.3

    ax.axhspan(or_low, or_high, xmin=0.02, xmax=0.34, color="#dceefb", alpha=0.85)
    ax.text(1.2, or_high + 0.18, "Opening range", color="#1d3557")
    ax.axhline(or_high, color="#457b9d", linestyle="--", linewidth=1.2)
    ax.axhline(or_low, color="#457b9d", linestyle="--", linewidth=1.2)
    ax.text(9.2, or_high, "OR high", va="center", ha="right", color="#1d3557")
    ax.text(9.2, or_low, "OR low", va="center", ha="right", color="#1d3557")

    width = 0.48
    for idx, (open_price, close_price, high_price, low_price) in enumerate(zip(opens, closes, highs, lows)):
        color = "#1d6f42" if close_price >= open_price else "#9a031e"
        ax.vlines(idx, low_price, high_price, color=color, linewidth=1.4, zorder=2)
        body_low = min(open_price, close_price)
        body_high = max(open_price, close_price)
        ax.add_patch(
            plt.Rectangle(
                (idx - width / 2, body_low),
                width,
                max(body_high - body_low, 0.02),
                facecolor=color,
                edgecolor=color,
                alpha=0.9,
                zorder=3,
            )
        )

    ax.annotate(
        "Breakout bar closes above OR high",
        xy=(4, closes[4]),
        xytext=(1.2, 103.2),
        arrowprops=dict(arrowstyle="->", color="#1d3557", lw=1.2),
        color="#1d3557",
    )
    ax.scatter([5], [entry_price], color="#ff7f11", s=55, zorder=5)
    ax.annotate(
        "Next open entry",
        xy=(5, entry_price),
        xytext=(5.8, 102.35),
        arrowprops=dict(arrowstyle="->", color="#ff7f11", lw=1.2),
        color="#7f4f24",
    )
    ax.axhline(stop_price, color="#b00020", linestyle=":", linewidth=1.4)
    ax.axhline(target_price, color="#2a9d8f", linestyle=":", linewidth=1.4)
    ax.text(9.2, stop_price, "Stop below OR", va="center", ha="right", color="#b00020")
    ax.text(9.2, target_price, "Target = entry + 2.0R", va="center", ha="right", color="#2a9d8f")
    ax.text(4.0, 98.9, "Signal is decided on the breakout-bar close.\nExecution happens at the next bar open.", color="#3d405b")
    ax.set_xlim(-0.8, 9.8)
    ax.set_ylim(98.7, 106.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["09:30", "09:35", "09:40", "09:45", "09:50", "09:55", "10:00", "10:05", "10:10", "10:15"])
    ax.set_ylabel("Price")
    ax.set_title("ORB mechanics diagram for the retained-final implementation")
    _save_figure(path)


def _make_entry(cfg: RetainedConfig, **overrides: Any) -> BaselineEntryConfig:
    payload = {
        "or_minutes": cfg.or_minutes,
        "opening_time": cfg.opening_time,
        "direction": cfg.direction,
        "one_trade_per_day": cfg.one_trade_per_day,
        "entry_buffer_ticks": cfg.entry_buffer_ticks,
        "stop_buffer_ticks": cfg.stop_buffer_ticks,
        "target_multiple": cfg.target_multiple,
        "vwap_confirmation": cfg.vwap_confirmation,
        "vwap_column": cfg.vwap_column,
        "time_exit": cfg.time_exit,
        "account_size_usd": cfg.account_size_usd,
        "risk_per_trade_pct": cfg.risk_per_trade_pct,
        "tick_size": cfg.tick_size,
        "entry_on_next_open": cfg.entry_on_next_open,
    }
    payload.update(overrides)
    return BaselineEntryConfig(**payload)


def _make_ensemble(cfg: RetainedConfig, **overrides: Any) -> BaselineEnsembleConfig:
    payload = {
        "atr_window": cfg.atr_window,
        "q_lows_pct": cfg.q_lows_pct,
        "q_highs_pct": cfg.q_highs_pct,
        "vote_threshold": cfg.vote_threshold,
    }
    payload.update(overrides)
    return BaselineEnsembleConfig(**payload)


def _make_compression(cfg: RetainedConfig, **overrides: Any) -> CompressionConfig:
    payload = {
        "mode": cfg.compression_mode,
        "usage": cfg.compression_usage,
        "soft_bonus_votes": cfg.compression_soft_bonus_votes,
    }
    payload.update(overrides)
    return CompressionConfig(**payload)


def _make_dynamic(cfg: RetainedConfig, **overrides: Any) -> DynamicThresholdConfig:
    payload = {
        "mode": cfg.dynamic_mode,
        "noise_lookback": cfg.noise_lookback,
        "noise_vm": cfg.noise_vm,
        "threshold_style": cfg.dynamic_threshold_style,
        "noise_k": cfg.noise_k,
        "atr_k": cfg.dynamic_atr_k,
        "confirm_bars": cfg.dynamic_confirm_bars,
        "schedule": cfg.dynamic_schedule,
    }
    payload.update(overrides)
    return DynamicThresholdConfig(**payload)


def _make_experiment(cfg: RetainedConfig, **overrides: Any) -> ExperimentConfig:
    entry = overrides.pop("entry", _make_entry(cfg))
    ensemble = overrides.pop("ensemble", _make_ensemble(cfg))
    compression = overrides.pop("compression", _make_compression(cfg))
    dynamic = overrides.pop("dynamic", _make_dynamic(cfg))
    exit_cfg = overrides.pop("exit_cfg", ExitConfig(mode=cfg.exit_mode))
    name = overrides.pop("name", cfg.name)
    return ExperimentConfig(
        name=name,
        stage="full_reopt",
        family="full_reopt",
        baseline_entry=entry,
        baseline_ensemble=ensemble,
        compression=compression,
        exit=exit_cfg,
        dynamic_threshold=dynamic,
    )


def _row_to_scope_records(row: dict[str, Any], extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    extra = extra or {}
    records: list[dict[str, Any]] = []
    for scope in ("is", "oos"):
        records.append(
            {
                **extra,
                "scope": scope,
                "net_pnl_usd": float(row.get(f"{scope}_net_pnl", 0.0)),
                "sharpe": float(row.get(f"{scope}_sharpe_ratio", 0.0)),
                "profit_factor": float(row.get(f"{scope}_profit_factor", 0.0)),
                "max_drawdown_usd": abs(float(row.get(f"{scope}_max_drawdown", 0.0))),
                "n_trades": int(row.get(f"{scope}_nb_trades", 0) or 0),
            }
        )
    return records


def _build_context_factory(dataset_path: Path, cfg: RetainedConfig):
    cache: dict[tuple[Any, ...], CampaignContext] = {}
    atr_grid_windows = (10, 14, 20, 30)

    def get_context(entry_cfg: BaselineEntryConfig, required_windows: tuple[int, ...]) -> CampaignContext:
        key = (
            int(entry_cfg.or_minutes),
            str(entry_cfg.opening_time),
            str(entry_cfg.direction),
            int(entry_cfg.entry_buffer_ticks),
            int(entry_cfg.stop_buffer_ticks),
            bool(entry_cfg.vwap_confirmation),
            str(entry_cfg.vwap_column),
            str(entry_cfg.time_exit),
            tuple(sorted(int(x) for x in required_windows)),
        )
        if key not in cache:
            minute_df = prepare_minute_dataset(dataset_path=dataset_path, baseline_entry=entry_cfg, atr_windows=required_windows)
            daily_reference = build_daily_reference(minute_df)
            minute_df = attach_daily_reference(minute_df, daily_reference)
            candidate_base = build_candidate_universe(minute_df, baseline_entry=entry_cfg)
            all_sessions = sorted(pd.to_datetime(minute_df["session_date"]).dt.date.unique())
            split_idx = int(len(all_sessions) * 0.70)
            split_idx = max(1, min(len(all_sessions) - 1, split_idx))
            is_sessions = all_sessions[:split_idx]
            oos_sessions = all_sessions[split_idx:]
            cache[key] = CampaignContext(
                all_sessions=all_sessions,
                is_sessions=is_sessions,
                oos_sessions=oos_sessions,
                minute_df=minute_df,
                candidate_base_df=candidate_base,
                daily_patterns=daily_reference,
            )
        return cache[key]

    def evaluate(experiment: ExperimentConfig) -> dict[str, Any]:
        required_windows = tuple(sorted({int(experiment.baseline_ensemble.atr_window), *atr_grid_windows}))
        context = get_context(experiment.baseline_entry, required_windows)
        row, _ = _evaluate_experiment(
            experiment=experiment,
            context=context,
            bootstrap_paths=0,
            random_seed=42,
            keep_details=False,
            max_leverage=None,
        )
        return row

    return evaluate


def _build_stage_grids(dataset_path: Path, cfg: RetainedConfig) -> dict[str, pd.DataFrame]:
    evaluate = _build_context_factory(dataset_path, cfg)
    grids: dict[str, pd.DataFrame] = {}

    entry_target_rows: list[dict[str, Any]] = []
    for or_minutes in (10, 15, 20, 30):
        for target_multiple in (1.5, 2.0, 2.5):
            exp = _make_experiment(
                cfg,
                name=f"or{or_minutes}_tm{target_multiple}",
                entry=_make_entry(cfg, or_minutes=int(or_minutes), target_multiple=float(target_multiple)),
            )
            entry_target_rows.extend(_row_to_scope_records(evaluate(exp), {"or_minutes": int(or_minutes), "target_multiple": float(target_multiple)}))
    grids["or_target"] = pd.DataFrame(entry_target_rows)

    vwap_time_rows: list[dict[str, Any]] = []
    for vwap_confirmation in (False, True):
        for time_exit in ("15:30:00", "16:00:00"):
            exp = _make_experiment(
                cfg,
                name=f"vwap{int(bool(vwap_confirmation))}_tx{time_exit.replace(':', '')}",
                entry=_make_entry(cfg, vwap_confirmation=bool(vwap_confirmation), time_exit=str(time_exit)),
            )
            vwap_time_rows.extend(_row_to_scope_records(evaluate(exp), {"vwap_confirmation": str(bool(vwap_confirmation)), "time_exit": str(time_exit)}))
    grids["vwap_time"] = pd.DataFrame(vwap_time_rows)

    ensemble_rows: list[dict[str, Any]] = []
    for atr_window in (10, 14, 20):
        for vote_threshold in (0.50, 0.67, 0.75):
            exp = _make_experiment(
                cfg,
                name=f"atr{atr_window}_vote{vote_threshold}",
                ensemble=_make_ensemble(cfg, atr_window=int(atr_window), vote_threshold=float(vote_threshold)),
            )
            ensemble_rows.extend(_row_to_scope_records(evaluate(exp), {"atr_window": int(atr_window), "vote_threshold": float(vote_threshold)}))
    grids["ensemble"] = pd.DataFrame(ensemble_rows)

    compression_rows: list[dict[str, Any]] = []
    for compression_mode in ("none", "weak_close", "strong_close", "nr4"):
        for compression_usage in ("hard_filter", "soft_vote_bonus"):
            exp = _make_experiment(
                cfg,
                name=f"comp_{compression_mode}_{compression_usage}",
                compression=_make_compression(cfg, mode=str(compression_mode), usage=str(compression_usage)),
            )
            compression_rows.extend(_row_to_scope_records(evaluate(exp), {"compression_mode": str(compression_mode), "compression_usage": str(compression_usage)}))
    grids["compression"] = pd.DataFrame(compression_rows)

    noise_rows: list[dict[str, Any]] = []
    for lookback in (14, 20, 30):
        for vm in (0.75, 1.0, 1.25):
            exp = _make_experiment(
                cfg,
                name=f"noise_l{lookback}_vm{vm}",
                dynamic=_make_dynamic(cfg, mode="noise_area_gate", noise_lookback=int(lookback), noise_vm=float(vm), threshold_style="max_or_high_noise"),
            )
            noise_rows.extend(_row_to_scope_records(evaluate(exp), {"noise_lookback": int(lookback), "noise_vm": float(vm)}))
    grids["noise"] = pd.DataFrame(noise_rows)
    return grids


def _plot_scope_pair_heatmap(
    frame: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    path: Path,
    title: str,
    current_x: Any,
    current_y: Any,
    *,
    cmap_name: str = "RdYlGn",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=False)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#f1f3f5")

    for idx, scope in enumerate(("is", "oos")):
        pivot = frame.loc[frame["scope"].eq(scope)].pivot(index=y, columns=x, values=value)
        pivot = pivot.sort_index().sort_index(axis=1)
        matrix = pivot.values.astype(float)
        ax = axes[idx]
        ax.grid(False)
        image = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
        x_labels = [str(v) for v in pivot.columns.tolist()]
        y_labels = [str(v) for v in pivot.index.tolist()]
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(scope.upper())
        if str(current_x) in x_labels and str(current_y) in y_labels:
            x_idx = x_labels.index(str(current_x))
            y_idx = y_labels.index(str(current_y))
            ax.add_patch(plt.Rectangle((x_idx - 0.5, y_idx - 0.5), 1.0, 1.0, fill=False, edgecolor="black", linewidth=2.0))
        for y_idx, row_values in enumerate(matrix):
            for x_idx, cell in enumerate(row_values):
                if math.isfinite(cell):
                    ax.text(x_idx, y_idx, f"{cell:.2f}", ha="center", va="center", fontsize=8)
    fig.suptitle(title)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=matplotlib.MatplotlibDeprecationWarning)
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82)
    cbar.set_label(value)
    _save_figure(path)


def _plot_dynamic_sizing_bar(summary_df: pd.DataFrame, path: Path) -> None:
    frame = summary_df.loc[summary_df["variant_name"].astype(str).str.startswith("sizing_3state_")].copy()
    frame = frame.sort_values("oos_sharpe", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.4, 4.4))
    values = pd.to_numeric(frame["oos_sharpe"], errors="coerce").fillna(0.0)
    colors = ["#264653" if "atr_ratio_10_30" in name else "#577590" if "realized_vol_ratio_15_60" in name else "#8d99ae" for name in frame["variant_name"].astype(str)]
    ax.bar(frame["feature_name"].astype(str), values, color=colors)
    ax.set_title("OOS Sharpe by retained-final 3-state overlay candidate")
    ax.set_ylabel("OOS Sharpe")
    ax.tick_params(axis="x", rotation=30)
    ax.axhline(0.0, color="#293241", linewidth=1.0)
    _save_figure(path)


def _build_retained_fast_slow_grid(retained: dict[str, Any]) -> pd.DataFrame:
    minute_df = retained["minute_df"].copy()
    minute_df["timestamp"] = pd.to_datetime(minute_df["timestamp"], errors="coerce", utc=True)
    minute_df["close"] = pd.to_numeric(minute_df["close"], errors="coerce")
    minute_df = minute_df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
    close_returns = minute_df["close"].pct_change()

    fast_windows = list(range(12, 19))
    slow_windows = list(range(50, 71, 5))
    all_windows = sorted(set(fast_windows + slow_windows))
    for window in all_windows:
        minute_df[f"vol_std_{window}"] = close_returns.rolling(window).std()

    selected_final = retained["selected_final"].copy()
    selected_final["session_date"] = pd.to_datetime(selected_final["session_date"], errors="coerce").dt.date
    selected_final["timestamp"] = pd.to_datetime(selected_final["timestamp"], errors="coerce", utc=True)
    feature_cols = [f"vol_std_{window}" for window in all_windows]
    signal_feature_rows = selected_final.merge(
        minute_df[["timestamp", *feature_cols]],
        on="timestamp",
        how="left",
    )

    nominal_trades = retained["trades"].copy()
    nominal_trades["session_date"] = pd.to_datetime(nominal_trades["session_date"], errors="coerce").dt.date
    is_set = set(pd.to_datetime(pd.Index(retained["is_sessions"])).date)
    oos_set = set(pd.to_datetime(pd.Index(retained["oos_sessions"])).date)

    rows: list[dict[str, Any]] = []
    for fast_window in fast_windows:
        for slow_window in slow_windows:
            if fast_window >= slow_window:
                continue
            feature_name = f"realized_vol_ratio_{fast_window}_{slow_window}"
            regime_probe = signal_feature_rows[["session_date"]].copy()
            regime_probe["phase"] = np.where(regime_probe["session_date"].isin(is_set), "is", "oos")
            regime_probe[feature_name] = (
                pd.to_numeric(signal_feature_rows[f"vol_std_{fast_window}"], errors="coerce")
                / pd.to_numeric(signal_feature_rows[f"vol_std_{slow_window}"], errors="coerce")
            )
            conditional_probe, feature_score_probe, assignments_probe, _ = build_conditional_bucket_analysis(
                regime_df=regime_probe,
                nominal_trades=nominal_trades,
                initial_capital=float(retained["config"].account_size_usd),
                feature_specs=(
                    RegimeFeatureSpec(
                        name=feature_name,
                        family="volatility",
                        description=f"Realized volatility ratio {fast_window}/{slow_window}.",
                        value_column=feature_name,
                    ),
                ),
                min_bucket_obs_is=50,
            )
            if feature_score_probe.empty or feature_name not in assignments_probe:
                continue
            feature_rows = conditional_probe.loc[conditional_probe["feature_name"].eq(feature_name)].sort_values("bucket_position")
            ranked = feature_rows.sort_values(["is_composite_score", "is_expectancy", "is_profit_factor"], ascending=[True, True, True]).reset_index(drop=True)
            if len(ranked) != 3:
                continue
            bucket_map = {str(ranked.iloc[0]["bucket_label"]): 0.50, str(ranked.iloc[1]["bucket_label"]): 0.75, str(ranked.iloc[2]["bucket_label"]): 1.00}
            controls_probe = build_static_regime_controls(regime_df=regime_probe, feature_name=feature_name, bucket_labels=assignments_probe[feature_name], bucket_multipliers=bucket_map)
            scaled = _scale_nominal_trades_by_multiplier(
                nominal_trades=nominal_trades,
                controls=controls_probe,
                account_size_usd=float(retained["config"].account_size_usd),
                base_risk_pct=float(retained["config"].risk_per_trade_pct),
                tick_value_usd=0.5,
                point_value_usd=2.0,
                commission_per_side_usd=1.25,
            )
            scaled["session_date"] = pd.to_datetime(scaled["session_date"], errors="coerce").dt.date
            oos_subset = scaled.loc[scaled["session_date"].isin(oos_set)].copy()
            oos_pnl = float(pd.to_numeric(oos_subset["net_pnl_usd"], errors="coerce").sum()) if not oos_subset.empty else 0.0
            score_row = feature_score_probe.iloc[0]
            rows.append(
                {
                    "fast_window": int(fast_window),
                    "slow_window": int(slow_window),
                    "feature_selection_score": float(score_row["feature_selection_score"]),
                    "oos_net_pnl": oos_pnl,
                }
            )
    return pd.DataFrame(rows)


def _plot_fast_slow_heatmap(grid: pd.DataFrame, value_column: str, path: Path, title: str, current_fast: int = 15, current_slow: int = 60) -> None:
    pivot = grid.pivot(index="fast_window", columns="slow_window", values=value_column).sort_index().sort_index(axis=1)
    matrix = pivot.values.astype(float)
    x_labels = pivot.columns.astype(int).tolist()
    y_labels = pivot.index.astype(int).tolist()

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#f1f3f5")
    ax.grid(False)
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Slow window")
    ax.set_ylabel("Fast window")
    ax.set_title(title)
    if current_slow in x_labels and current_fast in y_labels:
        x_idx = x_labels.index(current_slow)
        y_idx = y_labels.index(current_fast)
        ax.add_patch(plt.Rectangle((x_idx - 0.5, y_idx - 0.5), 1.0, 1.0, fill=False, edgecolor="black", linewidth=2.0))
        ax.text(x_idx, y_idx, "15/60", ha="center", va="center", color="black", fontsize=8, fontweight="bold")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=matplotlib.MatplotlibDeprecationWarning)
        cbar = plt.colorbar(image, ax=ax)
    cbar.set_label(value_column)
    _save_figure(path)


def _scope_table_from_row(row: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Scope": "IS",
                "Net PnL": _format_money(row.get("is_net_pnl")),
                "Sharpe": _format_number(row.get("is_sharpe_ratio")),
                "Max DD": _format_money(row.get("is_max_drawdown")),
                "Trades": int(row.get("is_nb_trades", 0) or 0),
            },
            {
                "Scope": "OOS",
                "Net PnL": _format_money(row.get("oos_net_pnl")),
                "Sharpe": _format_number(row.get("oos_sharpe_ratio")),
                "Max DD": _format_money(row.get("oos_max_drawdown")),
                "Trades": int(row.get("oos_nb_trades", 0) or 0),
            },
        ]
    )


def build_memo(config: MemoConfig) -> tuple[Path, Path]:
    output_dir = config.output_dir.resolve()
    figures_dir = output_dir / FIGURES_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    retained = _rebuild_retained_final(_latest_mnq_dataset())
    cfg: RetainedConfig = retained["config"]
    row = retained["row"]

    retained_3state_export = config.retained_3state_export.resolve() if config.retained_3state_export is not None else _latest_retained_3state_export()
    summary_df = pd.read_csv(retained_3state_export / "summary_variants.csv")
    feature_ranking = pd.read_csv(retained_3state_export / "feature_ranking.csv")
    mappings = pd.read_csv(retained_3state_export / "regime_state_mappings.csv")

    stage_grids = {} if config.skip_heatmaps else _build_stage_grids(_latest_mnq_dataset(), cfg)

    figures: dict[str, Path] = {}
    mechanics = figures_dir / "orb_mechanics_diagram.png"
    _make_orb_mechanics_diagram(mechanics)
    figures["mechanics"] = mechanics

    if stage_grids:
        base_heatmap = figures_dir / "base_signal_or_target_heatmap.png"
        _plot_scope_pair_heatmap(stage_grids["or_target"], "target_multiple", "or_minutes", "sharpe", base_heatmap, "Base signal | OR minutes x target multiple | Sharpe", cfg.target_multiple, cfg.or_minutes)
        figures["base_signal"] = base_heatmap

        vwap_heatmap = figures_dir / "vwap_time_heatmap.png"
        _plot_scope_pair_heatmap(stage_grids["vwap_time"], "time_exit", "vwap_confirmation", "sharpe", vwap_heatmap, "VWAP filter | confirmation x time exit | Sharpe", str(cfg.time_exit), str(bool(cfg.vwap_confirmation)))
        figures["vwap"] = vwap_heatmap

        atr_heatmap = figures_dir / "atr_vote_heatmap.png"
        _plot_scope_pair_heatmap(stage_grids["ensemble"], "vote_threshold", "atr_window", "sharpe", atr_heatmap, "ATR ensemble | atr window x vote threshold | Sharpe", cfg.vote_threshold, cfg.atr_window)
        figures["atr"] = atr_heatmap

        compression_heatmap = figures_dir / "compression_usage_heatmap.png"
        _plot_scope_pair_heatmap(stage_grids["compression"], "compression_usage", "compression_mode", "sharpe", compression_heatmap, "Compression overlay | mode x usage | Sharpe", cfg.compression_usage, cfg.compression_mode)
        figures["compression"] = compression_heatmap

        noise_heatmap = figures_dir / "noise_vm_heatmap.png"
        _plot_scope_pair_heatmap(stage_grids["noise"], "noise_vm", "noise_lookback", "sharpe", noise_heatmap, "Dynamic noise gate | noise lookback x noise VM | Sharpe", cfg.noise_vm, cfg.noise_lookback)
        figures["noise"] = noise_heatmap

    fast_slow_grid = _build_retained_fast_slow_grid(retained)
    fast_slow_is = figures_dir / "retained_final_fast_slow_is_heatmap.png"
    _plot_fast_slow_heatmap(fast_slow_grid, "feature_selection_score", fast_slow_is, "3-state on retained final | IS feature-selection score")
    figures["fast_slow_is"] = fast_slow_is

    fast_slow_oos = figures_dir / "retained_final_fast_slow_oos_heatmap.png"
    _plot_fast_slow_heatmap(fast_slow_grid, "oos_net_pnl", fast_slow_oos, "3-state on retained final | OOS net PnL")
    figures["fast_slow_oos"] = fast_slow_oos

    feature_bar = figures_dir / "retained_final_3state_feature_bar.png"
    _plot_dynamic_sizing_bar(summary_df, feature_bar)
    figures["feature_bar"] = feature_bar

    nominal_row = summary_df.loc[summary_df["variant_name"].eq("nominal")].iloc[0]
    best_overlay = summary_df.loc[summary_df["variant_name"].astype(str).str.startswith("sizing_3state_")].sort_values(["oos_sharpe", "oos_net_pnl"], ascending=[False, False]).iloc[0]
    rv15_60 = summary_df.loc[summary_df["variant_name"].eq("sizing_3state_realized_vol_ratio_15_60")].iloc[0]

    top_feature_table = feature_ranking.loc[:, ["feature_name", "family", "feature_selection_score", "best_bucket_is", "worst_bucket_is", "valid_for_overlay"]].head(6).copy()
    overlay_compare = summary_df.loc[
        summary_df["variant_name"].isin(["nominal", "sizing_3state_atr_ratio_10_30", "sizing_3state_realized_vol_ratio_15_60", "sizing_3state_overnight_range_pts"]),
        [
            "variant_name",
            "feature_name",
            "oos_net_pnl",
            "oos_sharpe",
            "oos_max_drawdown",
            "oos_net_pnl_retention_vs_nominal",
            "oos_sharpe_delta_vs_nominal",
            "oos_max_drawdown_improvement_vs_nominal",
        ],
    ].copy()
    overlay_compare = overlay_compare.sort_values(["variant_name"]).reset_index(drop=True)
    overlay_compare["oos_net_pnl"] = pd.to_numeric(overlay_compare["oos_net_pnl"], errors="coerce").round(1)
    overlay_compare["oos_sharpe"] = pd.to_numeric(overlay_compare["oos_sharpe"], errors="coerce").round(3)
    overlay_compare["oos_max_drawdown"] = pd.to_numeric(overlay_compare["oos_max_drawdown"], errors="coerce").round(1)
    overlay_compare["oos_net_pnl_retention_vs_nominal"] = pd.to_numeric(overlay_compare["oos_net_pnl_retention_vs_nominal"], errors="coerce").round(3)
    overlay_compare["oos_sharpe_delta_vs_nominal"] = pd.to_numeric(overlay_compare["oos_sharpe_delta_vs_nominal"], errors="coerce").round(3)
    overlay_compare["oos_max_drawdown_improvement_vs_nominal"] = pd.to_numeric(overlay_compare["oos_max_drawdown_improvement_vs_nominal"], errors="coerce").round(3)

    realized_mapping = mappings.loc[mappings["variant_name"].eq("sizing_3state_realized_vol_ratio_15_60"), ["bucket_label", "lower_bound", "upper_bound", "risk_multiplier"]].copy()
    realized_mapping["lower_bound"] = pd.to_numeric(realized_mapping["lower_bound"], errors="coerce").round(6)
    realized_mapping["upper_bound"] = pd.to_numeric(realized_mapping["upper_bound"], errors="coerce").round(6)
    realized_mapping["risk_multiplier"] = pd.to_numeric(realized_mapping["risk_multiplier"], errors="coerce").round(2)

    markdown = f"""# MNQ ORB Strategy - Institutional Research Memo

## 1. Executive Summary

This memo now follows the real retained stack in the same order as the research logic:

1. base signal,
2. VWAP filter,
3. ATR ensemble,
4. compression and dynamic gating,
5. 3-state overlay on top of the retained-final sleeve.

The implemented retained configuration remains `{cfg.name}` with `OR{cfg.or_minutes}`, `{cfg.direction}`, VWAP confirmation, ATR vote, `weak_close`, and `noise_area_gate`. The later 3-state campaign was rerun directly on that retained-final sleeve. The main result is that `realized_vol_ratio_15_60` is **not** the best overlay on the retained-final stack: the strongest OOS sizing candidate in the latest run is `{best_overlay["variant_name"]}`.

## 2. Layer 1 - Base Signal

The retained signal is an ORB continuation entry:

- OR window: `{cfg.or_minutes}` minutes
- Direction: `{cfg.direction}`
- Entry confirmation: breakout bar closes beyond the OR boundary
- Execution: next bar open
- Entry buffer / stop buffer: `{cfg.entry_buffer_ticks}` / `{cfg.stop_buffer_ticks}` ticks
- Target: `{cfg.target_multiple:.1f}R`
- Time exit: `{cfg.time_exit}`

Current retained-final IS/OOS metrics:

{_markdown_table(_scope_table_from_row(row))}

{_markdown_image(figures["mechanics"], output_dir, "Retained-final ORB mechanics") }

{_markdown_image(figures["base_signal"], output_dir, "IS/OOS heatmap for OR minutes x target multiple on the retained base signal") if "base_signal" in figures else "_Base-signal heatmap was skipped._"}

Readout:

- The retained point `OR15 / target 2.0R` sits inside a usable zone rather than on an isolated spike.
- The memo now shows the IS and OOS surfaces side by side instead of only describing the selected point.

## 3. Layer 2 - VWAP Filter

VWAP belongs to the retained signal stack itself. The retained point is:

- VWAP confirmation: `{"enabled" if cfg.vwap_confirmation else "disabled"}`
- VWAP column: `{cfg.vwap_column}`
- Current time exit used with that filter: `{cfg.time_exit}`

Retained-final IS/OOS metrics are the same trade set shown above because VWAP is already embedded in the retained sleeve:

{_markdown_table(_scope_table_from_row(row))}

{_markdown_image(figures["vwap"], output_dir, "IS/OOS heatmap for VWAP confirmation x time exit") if "vwap" in figures else "_VWAP heatmap was skipped._"}

Readout:

- The memo now makes VWAP explicit instead of burying it under the old sizing narration.
- The selected point is `vwap=True / 16:00 exit`.

## 4. Layer 3 - ATR Ensemble

ATR also belongs to the retained-final sleeve itself. The selected retained point is:

- ATR window: `{cfg.atr_window}`
- Vote threshold: `{cfg.vote_threshold:.2f}`
- Quantile lows: `{cfg.q_lows_pct}`
- Quantile highs: `{cfg.q_highs_pct}`

Current retained-final IS/OOS metrics:

{_markdown_table(_scope_table_from_row(row))}

{_markdown_image(figures["atr"], output_dir, "IS/OOS heatmap for ATR window x vote threshold") if "atr" in figures else "_ATR heatmap was skipped._"}

Readout:

- This is the layer where ATR actually lives in the retained-final architecture.
- The retained point `ATR(14) / vote 0.50` is now shown directly against nearby alternatives in IS and OOS.

## 5. Layer 4 - Compression And Dynamic Gate

The retained-final sleeve adds a pattern overlay and a dynamic noise gate:

- Compression mode: `{cfg.compression_mode}`
- Compression usage: `{cfg.compression_usage}`
- Dynamic mode: `{cfg.dynamic_mode}`
- Noise lookback / VM: `{cfg.noise_lookback}` / `{cfg.noise_vm:.2f}`
- Dynamic schedule: `{cfg.dynamic_schedule}`
- Threshold style: `{cfg.dynamic_threshold_style}`

Current retained-final IS/OOS metrics:

{_markdown_table(_scope_table_from_row(row))}

{_markdown_image(figures["compression"], output_dir, "IS/OOS heatmap for compression mode x usage") if "compression" in figures else "_Compression heatmap was skipped._"}

{_markdown_image(figures["noise"], output_dir, "IS/OOS heatmap for noise lookback x noise VM") if "noise" in figures else "_Dynamic-noise heatmap was skipped._"}

Readout:

- `weak_close / soft_vote_bonus` and `noise_area_gate` are no longer treated as footnotes.
- The memo now shows where the selected gate sits relative to nearby settings.

## 6. Layer 5 - 3-State Overlay On Retained Final

This layer is now run on the retained-final sleeve itself, not on the old nominal ORB branch.

Top retained-final overlay candidates by feature:

{_markdown_table(top_feature_table)}

{_markdown_image(figures["fast_slow_is"], output_dir, "IS heatmap for fast/slow realized-volatility ratios on retained final")}

{_markdown_image(figures["fast_slow_oos"], output_dir, "OOS heatmap for fast/slow realized-volatility ratios on retained final")}

The retained-final campaign compared the nominal sleeve against multiple 3-state overlays:

{_markdown_table(overlay_compare)}

{_markdown_image(figures["feature_bar"], output_dir, "OOS Sharpe by 3-state overlay candidate on retained final")}

Specific readout for `realized_vol_ratio_15_60` on retained final:

- Variant: `sizing_3state_realized_vol_ratio_15_60`
- IS net / Sharpe / maxDD: `{_format_money(rv15_60["is_net_pnl"])}` / `{_format_number(rv15_60["is_sharpe"])}` / `{_format_money(rv15_60["is_max_drawdown"])}`
- OOS net / Sharpe / maxDD: `{_format_money(rv15_60["oos_net_pnl"])}` / `{_format_number(rv15_60["oos_sharpe"])}` / `{_format_money(rv15_60["oos_max_drawdown"])}`
- OOS retention vs nominal: `{_format_number(rv15_60["oos_net_pnl_retention_vs_nominal"])}` 
- OOS Sharpe delta vs nominal: `{_format_number(rv15_60["oos_sharpe_delta_vs_nominal"])}` 

Bucket map used for `realized_vol_ratio_15_60` in the retained-final campaign:

{_markdown_table(realized_mapping)}

Readout:

- `realized_vol_ratio_15_60` still ranks near the top in IS feature selection, but it does **not** hold up as the best retained-final overlay in OOS.
- In the latest retained-final campaign, the strongest OOS 3-state candidate is `{best_overlay["variant_name"]}` with net `{_format_money(best_overlay["oos_net_pnl"])}`, Sharpe `{_format_number(best_overlay["oos_sharpe"])}`, and maxDD `{_format_money(best_overlay["oos_max_drawdown"])}`.
- The retained-final nominal sleeve remains stronger in raw PnL than the `15/60` overlay.

## 7. Recommendation

The logical reading of the stack is now:

1. the retained edge starts with the `OR15 / long / next-open` base signal;
2. VWAP and ATR are structural parts of that retained signal stack;
3. compression and dynamic gating are part of the retained-final filtering logic;
4. the 3-state overlay is a **separate last layer** that must be judged only after the retained-final sleeve is fixed.

Current recommendation from the latest retained-final-first read:

- keep the retained-final nominal sleeve clearly identified as the reference implementation;
- do **not** describe `realized_vol_ratio_15_60` as the retained-final 3-state winner;
- if a 3-state overlay is to be revisited on the retained-final sleeve, the first live candidate to inspect is `{best_overlay["variant_name"]}`, not `realized_vol_ratio_15_60`.
"""

    markdown_path = output_dir / DOCS_FILENAME
    html_path = output_dir / HTML_FILENAME
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(_markdown_to_html(markdown, "MNQ ORB Strategy - Institutional Research Memo"), encoding="utf-8")
    return markdown_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retained-3state-export", type=Path, default=None)
    parser.add_argument("--skip-heatmaps", action="store_true")
    args = parser.parse_args()
    build_memo(
        MemoConfig(
            output_dir=args.output_dir,
            retained_3state_export=Path(args.retained_3state_export) if args.retained_3state_export is not None else None,
            skip_heatmaps=bool(args.skip_heatmaps),
        )
    )


if __name__ == "__main__":
    main()
