"""Build an institutional research memo for the retained MNQ ORB strategy."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analytics.metrics import compute_metrics
from src.analytics.mnq_orb_regime_filter_sizing_campaign import (
    RegimeFeatureSpec,
    _scale_nominal_trades_by_multiplier,
    build_conditional_bucket_analysis,
    build_static_regime_controls,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
DOCS_FILENAME = "mnq_orb_strategy_institutional_memo.md"
HTML_FILENAME = "mnq_orb_strategy_institutional_memo.html"
FIGURES_SUBDIR = Path("figures") / "mnq_orb_strategy_memo"
TARGET_VARIANT = "single_15_60"
RESEARCH_WATCH_VARIANT = "median_plateau_compact"
DEFAULT_LOW_MULTIPLIER = 0.50
DEFAULT_MID_MULTIPLIER = 1.00
DEFAULT_HIGH_MULTIPLIER = 0.25
VARIANT_ORDER = [
    "single_15_60",
    "single_14_60",
    "single_16_60",
    "single_15_70",
    "single_15_80",
    "single_16_75",
    "median_fast15_slow_60_70_80",
    "median_plateau_compact",
]


@dataclass(frozen=True)
class MemoBuildConfig:
    output_dir: Path
    variant_export: Path
    audit_export: Path
    regime_export: Path | None = None
    include_stability_heatmaps: bool = True


@dataclass
class MemoArtifacts:
    markdown_path: Path
    html_path: Path
    figures_dir: Path
    figure_paths: list[Path]
    data_availability_notes: list[str]
    verdict_summary: str


@dataclass
class MemoBundle:
    variant_export: Path
    audit_export: Path
    regime_export: Path | None
    variant_metadata: dict[str, Any]
    audit_metadata: dict[str, Any]
    regime_metadata: dict[str, Any]
    variant_summary: pd.DataFrame
    variant_daily: pd.DataFrame
    variant_trades: pd.DataFrame
    variant_bucket_contribution: pd.DataFrame
    audit_summary: pd.DataFrame
    monthly_returns: pd.DataFrame
    daily_diff: pd.DataFrame
    bucket_distribution: pd.DataFrame
    worst_periods: pd.DataFrame
    multiplier_transitions: pd.DataFrame
    feature_ranking: pd.DataFrame
    conditional_bucket_analysis: pd.DataFrame
    regime_summary_variants: pd.DataFrame
    data_availability_notes: list[str] = field(default_factory=list)
    notebook_ipynb_path: Path | None = None
    notebook_html_path: Path | None = None


def _resolve_path(path_like: str | Path, base: Path | None = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    if base is None:
        return path.resolve()
    return (base / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path, *, parse_dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=parse_dates or [])


def _maybe_read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return _load_json(path)


def _coerce_numeric(series: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(pd.Series(series), errors="coerce")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _format_money(value: Any) -> str:
    numeric = _safe_float(value)
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:,.1f} USD"


def _format_number(value: Any, digits: int = 3) -> str:
    numeric = _safe_float(value)
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def _format_pct(value: Any, digits: int = 1, *, scale: float = 100.0) -> str:
    numeric = _safe_float(value)
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric * scale:.{digits}f}%"


def _format_date(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "n/a"
    return timestamp.strftime("%Y-%m-%d")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No data available._"

    clean = frame.copy()
    clean.columns = [str(column) for column in clean.columns]
    rows: list[list[str]] = []
    for _, row in clean.iterrows():
        rendered: list[str] = []
        for value in row:
            if isinstance(value, str):
                rendered.append(value)
                continue
            if pd.isna(value):
                rendered.append("n/a")
                continue
            if isinstance(value, (int, np.integer)):
                rendered.append(str(int(value)))
                continue
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.3f}")
                continue
            rendered.append(str(value))
        rows.append(rendered)

    header = "| " + " | ".join(clean.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(clean.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _variant_frame_by_order(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["variant_name"] = ordered["variant_name"].astype(str)
    ordered["variant_order"] = ordered["variant_name"].map({name: idx for idx, name in enumerate(VARIANT_ORDER)})
    ordered = ordered.sort_values(["variant_order", "variant_name"]).drop(columns="variant_order")
    return ordered.reset_index(drop=True)


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
            "grid.alpha": 0.25,
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


def _load_bundle(config: MemoBuildConfig) -> MemoBundle:
    output_dir = config.output_dir.resolve()
    variant_export = config.variant_export.resolve()
    audit_export = config.audit_export.resolve()

    variant_metadata = _load_json(variant_export / "run_metadata.json")
    audit_metadata = _load_json(audit_export / "run_metadata.json")

    regime_export = config.regime_export
    if regime_export is None:
        reference_root = variant_metadata.get("reference_export_root")
        regime_export = _resolve_path(reference_root, base=REPO_ROOT) if reference_root else None
    else:
        regime_export = regime_export.resolve()

    regime_metadata = _maybe_read_json(regime_export / "run_metadata.json" if regime_export else None)

    bundle = MemoBundle(
        variant_export=variant_export,
        audit_export=audit_export,
        regime_export=regime_export,
        variant_metadata=variant_metadata,
        audit_metadata=audit_metadata,
        regime_metadata=regime_metadata,
        variant_summary=_load_csv(variant_export / "variant_summary.csv"),
        variant_daily=_load_csv(variant_export / "variant_daily_returns.csv", parse_dates=["session_date"]),
        variant_trades=_load_csv(
            variant_export / "variant_trade_summary.csv",
            parse_dates=["session_date", "entry_time", "exit_time"],
        ),
        variant_bucket_contribution=_load_csv(variant_export / "variant_bucket_contribution.csv"),
        audit_summary=_load_csv(audit_export / "variant_audit_summary.csv"),
        monthly_returns=_load_csv(audit_export / "monthly_returns_by_variant.csv", parse_dates=["month_start", "month_end"]),
        daily_diff=_load_csv(audit_export / "baseline_vs_ensemble_daily_diff.csv", parse_dates=["session_date"]),
        bucket_distribution=_load_csv(audit_export / "bucket_distribution_by_variant.csv"),
        worst_periods=_load_csv(audit_export / "worst_periods_by_variant.csv"),
        multiplier_transitions=_load_csv(audit_export / "multiplier_transition_by_variant.csv"),
        feature_ranking=_load_csv(regime_export / "feature_ranking.csv") if regime_export else pd.DataFrame(),
        conditional_bucket_analysis=_load_csv(regime_export / "conditional_bucket_analysis.csv") if regime_export else pd.DataFrame(),
        regime_summary_variants=_load_csv(regime_export / "summary_variants.csv") if regime_export else pd.DataFrame(),
    )

    notebook_ipynb = NOTEBOOKS_DIR / "orb_MNQ_sizing_3state_client.executed.ipynb"
    notebook_html = NOTEBOOKS_DIR / "orb_MNQ_sizing_3state_client.executed.html"
    if notebook_ipynb.exists():
        bundle.notebook_ipynb_path = notebook_ipynb
    else:
        bundle.data_availability_notes.append(
            "Executed client notebook `notebooks/orb_MNQ_sizing_3state_client.executed.ipynb` was not found locally."
        )
    if notebook_html.exists():
        bundle.notebook_html_path = notebook_html
    else:
        bundle.data_availability_notes.append(
            "Standalone executed HTML notebook `notebooks/orb_MNQ_sizing_3state_client.executed.html` was not found locally; the memo uses the executed notebook and CSV exports instead."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    return bundle


def _target_variant_summary(bundle: MemoBundle) -> pd.Series:
    frame = bundle.variant_summary.loc[bundle.variant_summary["variant_name"].astype(str) == TARGET_VARIANT]
    if frame.empty:
        raise ValueError(f"Target variant {TARGET_VARIANT!r} not found in variant summary export.")
    return frame.iloc[0]


def _target_variant_trades(bundle: MemoBundle) -> pd.DataFrame:
    return (
        bundle.variant_trades.loc[bundle.variant_trades["variant_name"].astype(str) == TARGET_VARIANT]
        .copy()
        .sort_values("entry_time")
        .reset_index(drop=True)
    )


def _target_variant_daily(bundle: MemoBundle) -> pd.DataFrame:
    return (
        bundle.variant_daily.loc[bundle.variant_daily["variant_name"].astype(str) == TARGET_VARIANT]
        .copy()
        .sort_values("session_date")
        .reset_index(drop=True)
    )


def _feature_bucket_rows(bundle: MemoBundle) -> pd.DataFrame:
    if bundle.conditional_bucket_analysis.empty:
        return pd.DataFrame()
    rows = bundle.conditional_bucket_analysis.loc[
        bundle.conditional_bucket_analysis["feature_name"].astype(str) == "realized_vol_ratio_15_60"
    ].copy()
    return rows.sort_values("bucket_position").reset_index(drop=True)


def _feature_ranking_row(bundle: MemoBundle) -> pd.Series | None:
    if bundle.feature_ranking.empty:
        return None
    frame = bundle.feature_ranking.loc[
        bundle.feature_ranking["feature_name"].astype(str) == "realized_vol_ratio_15_60"
    ]
    return None if frame.empty else frame.iloc[0]


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
    ax.set_xticklabels(
        [
            "09:30",
            "09:35",
            "09:40",
            "09:45",
            "10:00\nsignal close",
            "10:05\nentry open",
            "10:10",
            "10:15",
            "10:20",
            "10:25",
        ]
    )
    ax.set_ylabel("Price")
    ax.set_title("ORB mechanics diagram for the retained MNQ implementation")
    _save_figure(path)


def _plot_variant_bar(
    frame: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    path: Path,
    *,
    invert_color_for_negative: bool = False,
) -> None:
    ordered = _variant_frame_by_order(frame)
    values = _coerce_numeric(ordered[value_col]).fillna(0.0)
    labels = ordered["variant_name"].astype(str).tolist()
    colors = []
    for name, value in zip(labels, values):
        if name == TARGET_VARIANT:
            colors.append("#264653")
        elif "median_" in name:
            colors.append("#7d8597")
        elif invert_color_for_negative and value < 0:
            colors.append("#9a031e")
        else:
            colors.append("#577590")

    fig, ax = plt.subplots(figsize=(11, 4.6))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    ax.axhline(0.0, color="#293241", linewidth=1.0)
    for bar, value in zip(bars, values):
        y = value + (max(abs(values.max()), abs(values.min()), 1.0) * 0.02 * (1 if value >= 0 else -1))
        va = "bottom" if value >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:,.2f}", ha="center", va=va, fontsize=8)
    _save_figure(path)


def _plot_monthly_returns(monthly_frame: pd.DataFrame, path: Path) -> None:
    ordered = monthly_frame.copy()
    ordered["month_start"] = pd.to_datetime(ordered["month_start"], errors="coerce")
    ordered = ordered.sort_values("month_start").reset_index(drop=True)
    values = _coerce_numeric(ordered["monthly_pnl_usd"]).fillna(0.0)
    colors = np.where(values >= 0.0, "#2a9d8f", "#b23a48")

    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.bar(ordered["month"].astype(str), values, color=colors)
    ax.set_title("Monthly PnL for retained production candidate `single_15_60`")
    ax.set_ylabel("Monthly PnL (USD)")
    ax.tick_params(axis="x", rotation=90)
    ax.axhline(0.0, color="#293241", linewidth=1.0)
    _save_figure(path)


def _plot_equity_curve(daily_frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(daily_frame["session_date"], _coerce_numeric(daily_frame["equity"]), color="#264653", linewidth=1.6)
    ax.set_title("Equity curve for retained production candidate `single_15_60`")
    ax.set_ylabel("Equity (USD)")
    _save_figure(path)


def _plot_drawdown_curve(daily_frame: pd.DataFrame, path: Path) -> None:
    drawdown = _coerce_numeric(daily_frame["drawdown_usd"]).fillna(0.0)
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.fill_between(daily_frame["session_date"], drawdown, 0.0, color="#b23a48", alpha=0.75)
    ax.plot(daily_frame["session_date"], drawdown, color="#8d0801", linewidth=1.2)
    ax.set_title("Drawdown curve for retained production candidate `single_15_60`")
    ax.set_ylabel("Drawdown (USD)")
    _save_figure(path)


def _plot_trade_distribution(trades: pd.DataFrame, path: Path) -> None:
    pnl = _coerce_numeric(trades["net_pnl_usd"]).dropna()
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.hist(pnl, bins=28, color="#577590", edgecolor="white")
    ax.axvline(float(pnl.mean()), color="#ff7f11", linestyle="--", linewidth=1.5, label="Average trade")
    ax.axvline(0.0, color="#293241", linewidth=1.0)
    ax.set_title("Trade-level PnL distribution for `single_15_60`")
    ax.set_xlabel("Net trade PnL (USD)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left")
    _save_figure(path)


def _plot_excess_contribution(diff_frame: pd.DataFrame, path: Path) -> None:
    plateau = diff_frame.loc[diff_frame["variant_name"].astype(str) == RESEARCH_WATCH_VARIANT].copy()
    plateau = plateau.sort_values("daily_pnl_diff", ascending=False).reset_index(drop=True)
    if plateau.empty:
        raise ValueError("No daily-diff rows were available for the plateau ensemble.")

    top_5 = float(plateau["daily_pnl_diff"].head(5).sum())
    total = float(plateau["daily_pnl_diff"].sum())
    remainder = total - top_5
    values = [top_5, remainder, total]
    labels = ["Top 5 positive days", "All remaining days", "Net excess PnL"]
    colors = ["#2a9d8f", "#6c757d" if remainder >= 0 else "#b23a48", "#264653" if total >= 0 else "#8d0801"]

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="#293241", linewidth=1.0)
    ax.set_ylabel("PnL vs `single_15_60` (USD)")
    ax.set_title("`median_plateau_compact` excess PnL is concentrated in a small set of outsized days")
    for bar, value in zip(bars, values):
        y = value + (max(abs(max(values)), abs(min(values)), 1.0) * 0.03 * (1 if value >= 0 else -1))
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:,.1f}", ha="center", va="bottom" if value >= 0 else "top")
    _save_figure(path)


def _plot_monthly_excess_bar(diff_frame: pd.DataFrame, path: Path) -> None:
    plateau = diff_frame.loc[diff_frame["variant_name"].astype(str) == RESEARCH_WATCH_VARIANT].copy()
    if plateau.empty:
        raise ValueError("No daily-diff rows were available for monthly excess aggregation.")
    plateau["month"] = plateau["session_date"].dt.to_period("M").astype(str)
    monthly = plateau.groupby("month", as_index=False)["daily_pnl_diff"].sum()
    values = _coerce_numeric(monthly["daily_pnl_diff"]).fillna(0.0)
    colors = np.where(values >= 0.0, "#4d908e", "#b23a48")

    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.bar(monthly["month"], values, color=colors)
    ax.axhline(0.0, color="#293241", linewidth=1.0)
    ax.set_title("Monthly excess PnL of `median_plateau_compact` versus `single_15_60`")
    ax.set_ylabel("Monthly excess PnL (USD)")
    ax.tick_params(axis="x", rotation=90)
    _save_figure(path)


def _plot_heatmap(
    grid: pd.DataFrame,
    value_column: str,
    path: Path,
    title: str,
    colorbar_label: str,
) -> None:
    pivot = (
        grid.pivot(index="fast_window", columns="slow_window", values=value_column)
        .sort_index()
        .sort_index(axis=1)
    )
    x_labels = pivot.columns.astype(int).tolist()
    y_labels = pivot.index.astype(int).tolist()
    matrix = pivot.values.astype(float)

    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#f1f3f5")
    ax.grid(False)
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Vol slow window")
    ax.set_ylabel("Vol fast window")
    ax.set_title(title)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=matplotlib.MatplotlibDeprecationWarning)
        cbar = plt.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)

    if 60 in x_labels and 15 in y_labels:
        x_idx = x_labels.index(60)
        y_idx = y_labels.index(15)
        ax.add_patch(plt.Rectangle((x_idx - 0.5, y_idx - 0.5), 1.0, 1.0, fill=False, edgecolor="black", linewidth=2.0))
        ax.text(x_idx, y_idx, "15/60", ha="center", va="center", color="black", fontsize=8, fontweight="bold")

    _save_figure(path)


def _build_stability_grid(bundle: MemoBundle) -> pd.DataFrame:
    if bundle.regime_export is None:
        raise FileNotFoundError("Reference regime export is unavailable.")

    regime_variant_controls_path = (
        bundle.regime_export / "variants" / "sizing_3state_realized_vol_ratio_15_60" / "controls.csv"
    )
    nominal_trades_path = bundle.regime_export / "variants" / "nominal" / "trades.csv"
    if not regime_variant_controls_path.exists() or not nominal_trades_path.exists():
        raise FileNotFoundError("Reference regime export is missing `controls.csv` or nominal `trades.csv`.")

    dataset_path_raw = (
        bundle.regime_metadata.get("dataset_path")
        or bundle.variant_metadata.get("dataset_path")
        or bundle.variant_metadata.get("reference_export_metadata", {}).get("dataset_path")
    )
    if dataset_path_raw is None:
        raise FileNotFoundError("Dataset path was not present in the export metadata.")

    dataset_path = _resolve_path(dataset_path_raw, base=REPO_ROOT)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed MNQ minute dataset was not found: {dataset_path}")

    controls = _load_csv(regime_variant_controls_path, parse_dates=["session_date"])
    baseline_trades = _load_csv(nominal_trades_path, parse_dates=["session_date", "entry_time", "exit_time"])
    minute_df = pd.read_parquet(dataset_path).copy()

    if "timestamp" not in minute_df.columns:
        reset = minute_df.reset_index()
        timestamp_col = "timestamp" if "timestamp" in reset.columns else reset.columns[0]
        minute_df = reset.rename(columns={timestamp_col: "timestamp"})

    minute_df["timestamp"] = pd.to_datetime(minute_df["timestamp"], errors="coerce", utc=True)
    minute_df["close"] = pd.to_numeric(minute_df["close"], errors="coerce")
    minute_df = minute_df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
    close_returns = minute_df["close"].pct_change()

    fast_windows = list(range(5, 31))
    slow_windows = list(range(30, 121, 5))
    all_windows = sorted(set(fast_windows + slow_windows))
    for window in all_windows:
        minute_df[f"vol_std_{window}"] = close_returns.rolling(window).std()

    phase_map = controls.loc[:, ["session_date", "phase"]].copy()
    phase_map["session_date"] = pd.to_datetime(phase_map["session_date"], errors="coerce").dt.date

    signal_rows = baseline_trades.copy()
    signal_rows["session_date"] = pd.to_datetime(signal_rows["session_date"], errors="coerce").dt.date
    signal_rows["entry_time"] = pd.to_datetime(signal_rows["entry_time"], errors="coerce", utc=True)
    signal_rows = signal_rows.loc[signal_rows["session_date"].isin(set(phase_map["session_date"]))].copy()
    signal_rows = (
        signal_rows.sort_values(["session_date", "entry_time"])
        .drop_duplicates(subset=["session_date"], keep="first")
        .reset_index(drop=True)
    )
    signal_rows["signal_timestamp"] = signal_rows["entry_time"] - pd.Timedelta(minutes=1)
    signal_rows = signal_rows.merge(phase_map, on="session_date", how="left", validate="one_to_one")

    feature_cols = [f"vol_std_{window}" for window in all_windows]
    signal_feature_rows = pd.merge_asof(
        signal_rows.sort_values("signal_timestamp"),
        minute_df[["timestamp", *feature_cols]].sort_values("timestamp"),
        left_on="signal_timestamp",
        right_on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(minutes=2),
    )
    signal_feature_rows = signal_feature_rows.dropna(subset=["timestamp"]).reset_index(drop=True)

    baseline_spec = bundle.variant_metadata.get("spec", {}).get("baseline", {})
    initial_capital = float(baseline_spec.get("account_size_usd", 50_000.0))
    base_risk_pct = float(baseline_spec.get("risk_per_trade_pct", 1.5))
    min_bucket_obs_is = int(bundle.regime_metadata.get("spec", {}).get("min_bucket_obs_is", 50))

    all_sessions = pd.Index(signal_rows["session_date"].dropna().tolist())
    is_sessions = pd.Index(phase_map.loc[phase_map["phase"].astype(str) == "is", "session_date"].dropna().tolist())
    oos_sessions = pd.Index(phase_map.loc[phase_map["phase"].astype(str) == "oos", "session_date"].dropna().tolist())

    stability_rows: list[dict[str, Any]] = []
    bucket_multiplier_map = {"low": DEFAULT_LOW_MULTIPLIER, "mid": DEFAULT_MID_MULTIPLIER, "high": DEFAULT_HIGH_MULTIPLIER}
    for fast_window in fast_windows:
        for slow_window in slow_windows:
            if fast_window >= slow_window:
                continue

            feature_name = f"realized_vol_ratio_{fast_window}_{slow_window}"
            regime_probe = signal_feature_rows[["session_date", "phase"]].copy()
            regime_probe[feature_name] = (
                pd.to_numeric(signal_feature_rows[f"vol_std_{fast_window}"], errors="coerce")
                / pd.to_numeric(signal_feature_rows[f"vol_std_{slow_window}"], errors="coerce")
            )

            conditional_probe, feature_score_probe, assignments_probe, _ = build_conditional_bucket_analysis(
                regime_df=regime_probe,
                nominal_trades=baseline_trades,
                initial_capital=initial_capital,
                feature_specs=(
                    RegimeFeatureSpec(
                        name=feature_name,
                        family="volatility",
                        description=f"Realized volatility ratio using rolling close-return stdev {fast_window} vs {slow_window} bars.",
                        value_column=feature_name,
                    ),
                ),
                min_bucket_obs_is=min_bucket_obs_is,
            )
            if feature_score_probe.empty or feature_name not in assignments_probe:
                continue

            controls_probe = build_static_regime_controls(
                regime_df=regime_probe,
                feature_name=feature_name,
                bucket_labels=assignments_probe[feature_name],
                bucket_multipliers=bucket_multiplier_map,
            )
            scaled_probe = _scale_nominal_trades_by_multiplier(
                nominal_trades=baseline_trades,
                controls=controls_probe,
                account_size_usd=initial_capital,
                base_risk_pct=base_risk_pct,
                tick_value_usd=0.5,
                point_value_usd=2.0,
                commission_per_side_usd=1.25,
            )
            oos_subset = scaled_probe.loc[
                pd.to_datetime(scaled_probe["session_date"], errors="coerce").dt.date.isin(set(oos_sessions))
            ].copy()
            oos_metrics = compute_metrics(oos_subset, session_dates=oos_sessions, initial_capital=initial_capital)
            score_row = feature_score_probe.iloc[0]
            stability_rows.append(
                {
                    "fast_window": int(fast_window),
                    "slow_window": int(slow_window),
                    "feature_selection_score": float(score_row["feature_selection_score"]),
                    "oos_sharpe_retained": float(oos_metrics.get("sharpe_ratio", 0.0)),
                    "oos_net_pnl_retained": float(oos_metrics.get("cumulative_pnl", 0.0)),
                    "all_session_count": int(len(all_sessions)),
                }
            )

    if not stability_rows:
        raise ValueError("Stability grid reconstruction did not produce any valid fast/slow combinations.")
    return pd.DataFrame(stability_rows).sort_values(["fast_window", "slow_window"]).reset_index(drop=True)


def _build_figures(bundle: MemoBundle, figures_dir: Path, *, include_stability_heatmaps: bool) -> tuple[list[Path], list[str]]:
    _setup_plot_style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    figure_paths: list[Path] = []

    summary = _variant_frame_by_order(bundle.variant_summary)
    daily = _target_variant_daily(bundle)
    trades = _target_variant_trades(bundle)
    monthly = bundle.monthly_returns.loc[bundle.monthly_returns["variant_name"].astype(str) == TARGET_VARIANT].copy()

    orb_diagram = figures_dir / "orb_mechanics_diagram.png"
    _make_orb_mechanics_diagram(orb_diagram)
    figure_paths.append(orb_diagram)

    sharpe_fig = figures_dir / "variant_sharpe_comparison.png"
    _plot_variant_bar(summary, "sharpe", "Sharpe by tested 3-state variant", "Sharpe", sharpe_fig)
    figure_paths.append(sharpe_fig)

    pnl_fig = figures_dir / "variant_net_pnl_comparison.png"
    _plot_variant_bar(summary, "net_pnl", "Net PnL by tested 3-state variant", "Net PnL (USD)", pnl_fig)
    figure_paths.append(pnl_fig)

    maxdd_fig = figures_dir / "variant_maxdd_comparison.png"
    _plot_variant_bar(summary, "max_drawdown", "Max drawdown by tested 3-state variant", "Max drawdown (USD)", maxdd_fig, invert_color_for_negative=True)
    figure_paths.append(maxdd_fig)

    if not monthly.empty:
        monthly_fig = figures_dir / "monthly_returns_single_15_60.png"
        _plot_monthly_returns(monthly, monthly_fig)
        figure_paths.append(monthly_fig)
    else:
        notes.append("Monthly return plot for `single_15_60` could not be generated because monthly returns were missing.")

    if not daily.empty and "equity" in daily.columns:
        equity_fig = figures_dir / "equity_curve_single_15_60.png"
        _plot_equity_curve(daily, equity_fig)
        figure_paths.append(equity_fig)
    else:
        notes.append("Equity curve for `single_15_60` could not be generated because daily equity data were missing.")

    if not daily.empty and "drawdown_usd" in daily.columns:
        drawdown_fig = figures_dir / "drawdown_curve_single_15_60.png"
        _plot_drawdown_curve(daily, drawdown_fig)
        figure_paths.append(drawdown_fig)
    else:
        notes.append("Drawdown curve for `single_15_60` could not be generated because drawdown data were missing.")

    if not trades.empty and "net_pnl_usd" in trades.columns:
        trade_dist_fig = figures_dir / "trade_pnl_distribution_single_15_60.png"
        _plot_trade_distribution(trades, trade_dist_fig)
        figure_paths.append(trade_dist_fig)
    else:
        notes.append("Trade PnL distribution for `single_15_60` could not be generated because trade-level data were missing.")

    if not bundle.daily_diff.empty:
        excess_fig = figures_dir / "baseline_vs_ensemble_excess_contribution.png"
        _plot_excess_contribution(bundle.daily_diff, excess_fig)
        figure_paths.append(excess_fig)

        excess_monthly_fig = figures_dir / "monthly_excess_pnl_plateau_vs_baseline.png"
        _plot_monthly_excess_bar(bundle.daily_diff, excess_monthly_fig)
        figure_paths.append(excess_monthly_fig)
    else:
        notes.append("Excess-contribution charts could not be generated because the audit daily-diff export was missing.")

    if include_stability_heatmaps:
        try:
            stability_grid = _build_stability_grid(bundle)
            heatmap_is = figures_dir / "vol_fast_slow_heatmap_is.png"
            _plot_heatmap(
                stability_grid,
                "feature_selection_score",
                heatmap_is,
                "IS feature-selection score around the retained 15/60 volatility ratio",
                "IS selection score",
            )
            figure_paths.append(heatmap_is)

            heatmap_oos = figures_dir / "vol_fast_slow_heatmap_oos.png"
            _plot_heatmap(
                stability_grid,
                "oos_sharpe_retained",
                heatmap_oos,
                "OOS Sharpe of retained 3-state policy around the retained 15/60 volatility ratio",
                "OOS Sharpe",
            )
            figure_paths.append(heatmap_oos)
        except Exception as exc:  # pragma: no cover - exercised on real data path
            notes.append(f"Vol fast/slow heatmaps were not reconstructed: {exc}")

    return figure_paths, notes


def _build_variant_comparison_table(bundle: MemoBundle) -> pd.DataFrame:
    frame = _variant_frame_by_order(bundle.variant_summary)
    cols = [
        "variant_name",
        "net_pnl",
        "sharpe",
        "max_drawdown",
        "delta_sharpe_vs_single_15_60",
        "delta_net_pnl_vs_single_15_60",
        "delta_maxdd_vs_single_15_60",
    ]
    out = frame.loc[:, cols].copy()
    out["net_pnl"] = _coerce_numeric(out["net_pnl"]).round(1)
    out["sharpe"] = _coerce_numeric(out["sharpe"]).round(3)
    out["max_drawdown"] = _coerce_numeric(out["max_drawdown"]).round(1)
    out["delta_sharpe_vs_single_15_60"] = _coerce_numeric(out["delta_sharpe_vs_single_15_60"]).round(3)
    out["delta_net_pnl_vs_single_15_60"] = _coerce_numeric(out["delta_net_pnl_vs_single_15_60"]).round(1)
    out["delta_maxdd_vs_single_15_60"] = _coerce_numeric(out["delta_maxdd_vs_single_15_60"]).round(1)
    return out


def _build_traceability_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Strategic component": "Retained final ORB sleeve definition",
                "Source file": "src/analytics/audit_mnq_orb_retained_vs_3state.py",
                "Functions / classes": "RetainedConfig; _rebuild_retained_final",
                "Associated tests": "Compared through retained-vs-3state audit outputs",
                "Status": "Source of truth for implemented retained sleeve",
            },
            {
                "Strategic component": "Retained final research notebook",
                "Source file": "src/analytics/build_mnq_retained_final_notebooks.py",
                "Functions / classes": "_orb_notebook",
                "Associated tests": "tests/test_build_mnq_retained_final_notebooks.py",
                "Status": "Notebook source for retained-final narrative",
            },
            {
                "Strategic component": "ORB signal definition",
                "Source file": "src/strategy/orb.py",
                "Functions / classes": "ORBStrategy.generate_signals",
                "Associated tests": "tests/test_orb_strategy.py; tests/test_opening_range.py",
                "Status": "Implemented and in use",
            },
            {
                "Strategic component": "Rolling volatility feature",
                "Source file": "src/features/volatility.py",
                "Functions / classes": "add_rolling_std",
                "Associated tests": "tests/test_volatility.py",
                "Status": "Implemented and causality-tested",
            },
            {
                "Strategic component": "Reference regime / sizing campaign",
                "Source file": "src/analytics/mnq_orb_regime_filter_sizing_campaign.py",
                "Functions / classes": "build_conditional_bucket_analysis; build_static_regime_controls",
                "Associated tests": "tests/test_mnq_orb_regime_filter_sizing_campaign.py",
                "Status": "Research reference",
            },
            {
                "Strategic component": "15/60 neighborhood smoke comparison",
                "Source file": "src/analytics/mnq_orb_3state_vol_sizing_variant_smoke.py",
                "Functions / classes": "run_smoke_campaign",
                "Associated tests": "tests/test_mnq_orb_3state_vol_sizing_variant_smoke.py",
                "Status": "Research reference",
            },
            {
                "Strategic component": "Variant robustness audit",
                "Source file": "src/analytics/mnq_orb_3state_vol_sizing_variant_audit.py",
                "Functions / classes": "run_audit; evaluate_diff_profile",
                "Associated tests": "tests/test_mnq_orb_3state_vol_sizing_variant_audit.py",
                "Status": "Research reference",
            },
            {
                "Strategic component": "Backtest execution timing and fills",
                "Source file": "src/engine/backtester.py",
                "Functions / classes": "run_backtest",
                "Associated tests": "tests/test_backtester.py",
                "Status": "Implemented and in use",
            },
            {
                "Strategic component": "Memo reconstruction",
                "Source file": "src/analytics/build_mnq_orb_strategy_memo.py",
                "Functions / classes": "build_strategy_memo",
                "Associated tests": "tests/test_build_mnq_orb_strategy_memo.py",
                "Status": "Generated from exports",
            },
        ]
    )


def _markdown_image(figure_path: Path, output_dir: Path, caption: str) -> str:
    relative_path = figure_path.relative_to(output_dir).as_posix()
    return f"![{caption}]({relative_path})"


def _load_retained_final_spec() -> dict[str, Any]:
    spec: dict[str, Any] = {
        "name": "full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate",
        "or_minutes": 15,
        "opening_time": "09:30:00",
        "direction": "long",
        "one_trade_per_day": True,
        "entry_buffer_ticks": 2,
        "stop_buffer_ticks": 2,
        "target_multiple": 2.0,
        "vwap_confirmation": True,
        "vwap_column": "continuous_session_vwap",
        "time_exit": "16:00:00",
        "account_size_usd": 50_000.0,
        "risk_per_trade_pct": 0.5,
        "entry_on_next_open": True,
        "atr_window": 14,
        "q_lows_pct": (20, 25, 30),
        "q_highs_pct": (90, 95),
        "vote_threshold": 0.5,
        "compression_mode": "weak_close",
        "compression_usage": "soft_vote_bonus",
        "dynamic_mode": "noise_area_gate",
        "noise_lookback": 30,
        "noise_vm": 1.0,
        "dynamic_schedule": "continuous_on_bar_close",
        "dynamic_threshold_style": "max_or_high_noise",
    }
    try:
        from src.analytics.audit_mnq_orb_retained_vs_3state import RetainedConfig

        cfg = RetainedConfig()
        spec.update(
            {
                "name": cfg.name,
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
                "entry_on_next_open": cfg.entry_on_next_open,
                "atr_window": cfg.atr_window,
                "q_lows_pct": cfg.q_lows_pct,
                "q_highs_pct": cfg.q_highs_pct,
                "vote_threshold": cfg.vote_threshold,
                "compression_mode": cfg.compression_mode,
                "compression_usage": cfg.compression_usage,
                "dynamic_mode": cfg.dynamic_mode,
                "noise_lookback": cfg.noise_lookback,
                "noise_vm": cfg.noise_vm,
                "dynamic_schedule": cfg.dynamic_schedule,
                "dynamic_threshold_style": cfg.dynamic_threshold_style,
            }
        )
    except Exception:
        pass
    return spec


def _markdown_link(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _inline_markdown(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
    return escaped


def _markdown_to_html(markdown_text: str, title: str) -> str:
    lines = markdown_text.splitlines()
    body_parts: list[str] = []
    paragraph_buffer: list[str] = []
    list_items: list[str] = []
    ordered_items: list[str] = []
    table_buffer: list[str] = []
    code_buffer: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        nonlocal paragraph_buffer
        if paragraph_buffer:
            body_parts.append(f"<p>{_inline_markdown(' '.join(paragraph_buffer).strip())}</p>")
            paragraph_buffer = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            body_parts.append("<ul>" + "".join(f"<li>{_inline_markdown(item)}</li>" for item in list_items) + "</ul>")
            list_items = []

    def flush_ordered() -> None:
        nonlocal ordered_items
        if ordered_items:
            body_parts.append("<ol>" + "".join(f"<li>{_inline_markdown(item)}</li>" for item in ordered_items) + "</ol>")
            ordered_items = []

    def flush_table() -> None:
        nonlocal table_buffer
        if not table_buffer:
            return
        rows = []
        for raw in table_buffer:
            cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
            rows.append(cells)
        if len(rows) >= 2:
            header = rows[0]
            body = rows[2:] if len(rows) > 2 else []
            head_html = "<thead><tr>" + "".join(f"<th>{_inline_markdown(cell)}</th>" for cell in header) + "</tr></thead>"
            body_html = "<tbody>" + "".join(
                "<tr>" + "".join(f"<td>{_inline_markdown(cell)}</td>" for cell in row) + "</tr>" for row in body
            ) + "</tbody>"
            body_parts.append(f"<table>{head_html}{body_html}</table>")
        table_buffer = []

    def flush_code() -> None:
        nonlocal code_buffer
        if code_buffer:
            code_text = "\n".join(code_buffer)
            body_parts.append(f"<pre><code>{html.escape(code_text)}</code></pre>")
            code_buffer = []

    for line in lines:
        stripped = line.rstrip("\n")
        if stripped.startswith("```"):
            flush_paragraph()
            flush_list()
            flush_ordered()
            flush_table()
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_buffer.append(stripped)
            continue

        if not stripped.strip():
            flush_paragraph()
            flush_list()
            flush_ordered()
            flush_table()
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            flush_list()
            flush_ordered()
            table_buffer.append(stripped)
            continue
        flush_table()

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_paragraph()
            flush_list()
            flush_ordered()
            level = len(heading_match.group(1))
            body_parts.append(f"<h{level}>{_inline_markdown(heading_match.group(2).strip())}</h{level}>")
            continue

        if stripped.startswith("![" ):
            flush_paragraph()
            flush_list()
            flush_ordered()
            image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
            if image_match:
                alt_text, src = image_match.groups()
                body_parts.append(
                    "<figure>"
                    f'<img src="{html.escape(src, quote=True)}" alt="{html.escape(alt_text, quote=True)}" />'
                    f"<figcaption>{_inline_markdown(alt_text)}</figcaption>"
                    "</figure>"
                )
                continue

        ordered_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if ordered_match:
            flush_paragraph()
            flush_list()
            ordered_items.append(ordered_match.group(1).strip())
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            flush_paragraph()
            flush_ordered()
            list_items.append(stripped[2:].strip())
            continue

        paragraph_buffer.append(stripped.strip())

    flush_paragraph()
    flush_list()
    flush_ordered()
    flush_table()
    flush_code()

    css = """
body { margin: 0; background: #f7f7f5; color: #14213d; font-family: Georgia, "Times New Roman", serif; line-height: 1.6; }
main { max-width: 1040px; margin: 0 auto; padding: 40px 48px 64px; background: white; box-shadow: 0 16px 50px rgba(20,33,61,0.08); }
h1, h2, h3, h4 { color: #0b2545; line-height: 1.25; }
h1 { margin-top: 0; border-bottom: 2px solid #d9e2ec; padding-bottom: 12px; }
h2 { margin-top: 40px; border-top: 1px solid #e5e7eb; padding-top: 24px; }
code { background: #edf2f7; padding: 0.12rem 0.3rem; border-radius: 4px; font-size: 0.95em; }
pre { background: #111827; color: #f8fafc; padding: 16px; border-radius: 8px; overflow-x: auto; }
table { width: 100%; border-collapse: collapse; margin: 18px 0 28px; font-size: 0.95rem; }
th, td { border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }
th { background: #f1f5f9; }
figure { margin: 24px 0 32px; }
img { max-width: 100%; height: auto; border: 1px solid #d9e2ec; }
figcaption { margin-top: 8px; color: #52606d; font-size: 0.95rem; }
ul, ol { padding-left: 24px; }
a { color: #14532d; }
"""
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\" />"
        f"<title>{html.escape(title, quote=True)}</title>"
        f"<style>{css}</style></head><body><main>{''.join(body_parts)}</main></body></html>"
    )


def _build_markdown(bundle: MemoBundle, output_dir: Path, figure_paths: list[Path], figure_notes: list[str]) -> tuple[str, str]:
    summary_row = _target_variant_summary(bundle)
    feature_row = _feature_ranking_row(bundle)
    bucket_rows = _feature_bucket_rows(bundle)
    trades = _target_variant_trades(bundle)
    daily = _target_variant_daily(bundle)
    monthly = bundle.monthly_returns.loc[bundle.monthly_returns["variant_name"].astype(str) == TARGET_VARIANT].copy()
    plateau_audit = bundle.audit_summary.loc[
        bundle.audit_summary["variant_name"].astype(str) == RESEARCH_WATCH_VARIANT
    ]
    plateau_audit_row = plateau_audit.iloc[0] if not plateau_audit.empty else pd.Series(dtype=object)
    baseline_audit = bundle.audit_summary.loc[bundle.audit_summary["variant_name"].astype(str) == TARGET_VARIANT]
    baseline_audit_row = baseline_audit.iloc[0] if not baseline_audit.empty else pd.Series(dtype=object)
    reference_variant = bundle.regime_summary_variants.loc[
        bundle.regime_summary_variants["variant_name"].astype(str) == "sizing_3state_realized_vol_ratio_15_60"
    ]
    reference_variant_row = reference_variant.iloc[0] if not reference_variant.empty else pd.Series(dtype=object)

    retained_spec = _load_retained_final_spec()
    retained_initial_capital = float(retained_spec.get("account_size_usd", 50_000.0))
    retained_risk_pct = float(retained_spec.get("risk_per_trade_pct", 0.5))
    retained_opening_time = str(retained_spec.get("opening_time", "09:30:00"))
    retained_or_minutes = int(retained_spec.get("or_minutes", 15))
    retained_target_multiple = float(retained_spec.get("target_multiple", 2.0))
    retained_entry_buffer_ticks = int(retained_spec.get("entry_buffer_ticks", 2))
    retained_stop_buffer_ticks = int(retained_spec.get("stop_buffer_ticks", 2))
    retained_time_exit = str(retained_spec.get("time_exit", "16:00:00"))
    retained_vwap_confirmation = bool(retained_spec.get("vwap_confirmation", True))
    retained_vwap_column = str(retained_spec.get("vwap_column", "continuous_session_vwap"))
    retained_direction = str(retained_spec.get("direction", "long"))
    retained_atr_window = int(retained_spec.get("atr_window", 14))
    retained_vote_threshold = float(retained_spec.get("vote_threshold", 0.5))
    retained_q_lows = tuple(retained_spec.get("q_lows_pct", (20, 25, 30)))
    retained_q_highs = tuple(retained_spec.get("q_highs_pct", (90, 95)))
    retained_compression_mode = str(retained_spec.get("compression_mode", "weak_close"))
    retained_compression_usage = str(retained_spec.get("compression_usage", "soft_vote_bonus"))
    retained_dynamic_mode = str(retained_spec.get("dynamic_mode", "noise_area_gate"))
    retained_noise_lookback = int(retained_spec.get("noise_lookback", 30))
    retained_noise_vm = float(retained_spec.get("noise_vm", 1.0))
    retained_dynamic_schedule = str(retained_spec.get("dynamic_schedule", "continuous_on_bar_close"))
    retained_threshold_style = str(retained_spec.get("dynamic_threshold_style", "max_or_high_noise"))

    three_state_baseline_spec = bundle.variant_metadata.get("spec", {}).get("baseline", {})
    three_state_initial_capital = float(three_state_baseline_spec.get("account_size_usd", 50_000.0))
    three_state_risk_pct = float(three_state_baseline_spec.get("risk_per_trade_pct", 1.5))
    three_state_opening_time = str(three_state_baseline_spec.get("opening_time", "09:30:00"))
    three_state_or_minutes = int(three_state_baseline_spec.get("or_minutes", 30))
    three_state_target_multiple = float(three_state_baseline_spec.get("target_multiple", 2.0))
    three_state_entry_buffer_ticks = int(three_state_baseline_spec.get("entry_buffer_ticks", 2))
    three_state_stop_buffer_ticks = int(three_state_baseline_spec.get("stop_buffer_ticks", 2))
    three_state_time_exit = str(three_state_baseline_spec.get("time_exit", "16:00:00"))
    three_state_vwap_confirmation = bool(three_state_baseline_spec.get("vwap_confirmation", True))
    three_state_direction = str(three_state_baseline_spec.get("direction", "both"))

    low_upper = _safe_float(bucket_rows.loc[bucket_rows["bucket_label"].astype(str) == "low", "upper_bound"].iloc[0]) if not bucket_rows.empty else float("nan")
    mid_upper = _safe_float(bucket_rows.loc[bucket_rows["bucket_label"].astype(str) == "mid", "upper_bound"].iloc[0]) if not bucket_rows.empty else float("nan")

    architecture_table = pd.DataFrame(
        [
            {
                "Research object": "retained final",
                "Role": "Implemented ORB sleeve retained from the general campaign",
                "OR window": f"{retained_or_minutes}m",
                "Direction": retained_direction,
                "VWAP": "yes" if retained_vwap_confirmation else "no",
                "ATR role": f"ATR({retained_atr_window}) vote",
                "Risk logic": f"fixed {_format_number(retained_risk_pct, 2)}%",
            },
            {
                "Research object": "3-state sizing branch",
                "Role": "Separate sizing study built on its own nominal ORB baseline",
                "OR window": f"{three_state_or_minutes}m",
                "Direction": three_state_direction,
                "VWAP": "yes" if three_state_vwap_confirmation else "no",
                "ATR role": "none in baseline; feature is realized-vol ratio",
                "Risk logic": f"base {_format_number(three_state_risk_pct, 2)}% x bucket multiplier",
            },
        ]
    )

    metrics_table = pd.DataFrame(
        [
            {
                "Metric": "Net PnL",
                "Value": _format_money(summary_row.get("net_pnl")),
            },
            {
                "Metric": "Sharpe",
                "Value": _format_number(summary_row.get("sharpe")),
            },
            {
                "Metric": "Sortino",
                "Value": _format_number(summary_row.get("sortino")),
            },
            {
                "Metric": "Max drawdown",
                "Value": _format_money(summary_row.get("max_drawdown")),
            },
            {
                "Metric": "Max daily loss",
                "Value": _format_money(summary_row.get("max_daily_loss")),
            },
            {
                "Metric": "Profit factor",
                "Value": _format_number(summary_row.get("profit_factor")),
            },
            {
                "Metric": "Win rate",
                "Value": _format_pct(summary_row.get("win_rate")),
            },
            {
                "Metric": "Trades",
                "Value": str(_safe_int(summary_row.get("num_trades"))),
            },
            {
                "Metric": "Average trade PnL",
                "Value": _format_money(summary_row.get("avg_trade_pnl")),
            },
            {
                "Metric": "Prop constraints pass",
                "Value": "Yes" if bool(summary_row.get("pass_prop_constraints", False)) else "No",
            },
        ]
    )

    bucket_table = pd.DataFrame()
    if not bucket_rows.empty:
        bucket_table = bucket_rows.loc[
            :,
            [
                "bucket_label",
                "lower_bound",
                "upper_bound",
                "is_n_obs",
                "is_net_pnl",
                "is_sharpe",
                "oos_n_obs",
                "oos_net_pnl",
                "oos_sharpe",
            ],
        ].copy()
        bucket_table["lower_bound"] = _coerce_numeric(bucket_table["lower_bound"]).round(6)
        bucket_table["upper_bound"] = _coerce_numeric(bucket_table["upper_bound"]).round(6)
        bucket_table["is_n_obs"] = _coerce_numeric(bucket_table["is_n_obs"]).astype("Int64")
        bucket_table["is_net_pnl"] = _coerce_numeric(bucket_table["is_net_pnl"]).round(1)
        bucket_table["is_sharpe"] = _coerce_numeric(bucket_table["is_sharpe"]).round(3)
        bucket_table["oos_n_obs"] = _coerce_numeric(bucket_table["oos_n_obs"]).astype("Int64")
        bucket_table["oos_net_pnl"] = _coerce_numeric(bucket_table["oos_net_pnl"]).round(1)
        bucket_table["oos_sharpe"] = _coerce_numeric(bucket_table["oos_sharpe"]).round(3)

    figures = {path.name: path for path in figure_paths}
    comparison_table = _build_variant_comparison_table(bundle)
    traceability_table = _build_traceability_table()

    data_availability_notes = [*bundle.data_availability_notes, *figure_notes]
    if not bundle.notebook_html_path and bundle.notebook_ipynb_path:
        data_availability_notes.append(
            "The client notebook narrative was sourced from the executed notebook `.ipynb` and not from a standalone HTML export."
        )

    appendix_commands = pd.DataFrame(
        [
            {"Command": "python -m py_compile src/analytics/build_mnq_orb_strategy_memo.py tests/test_build_mnq_orb_strategy_memo.py"},
            {"Command": "python -m pytest -q tests/test_build_mnq_orb_strategy_memo.py"},
            {
                "Command": "python -m src.analytics.build_mnq_orb_strategy_memo --output-dir docs --variant-export export/mnq_orb_3state_vol_sizing_variant_smoke_20260531_173822 --audit-export export/mnq_orb_3state_vol_sizing_variant_audit_20260531_195920",
            },
        ]
    )

    verdict_summary = (
        "The implemented retained ORB sleeve is the `retained final` OR15 long-only stack with VWAP and ATR gating; "
        "inside the separate 3-state sizing branch, the recommendation remains `single_15_60` with `low=0.50x`, `mid=1.00x`, and `high=0.25x`."
    )

    markdown = f"""# MNQ ORB Strategy - Institutional Research Memo

## 1. Executive Summary

The previous memo blended **two different MNQ ORB research objects**. That was the core confusion. This revised memo separates them explicitly:

- `retained final`: the implemented ORB sleeve retained from the main campaign, with **OR15**, **long-only**, **VWAP confirmation**, **ATR vote**, `weak_close`, and `noise_area_gate`.
- `single_15_60`: the retained candidate **inside a separate 3-state sizing branch**, built on a different nominal baseline (`OR30`, `both`, base risk `1.50%`).
- ATR and VWAP were not removed from the research stack; they belong to the `retained final` sleeve and were previously obscured by the sizing-branch narration.
- Branch-level recommendation: keep `single_15_60` over `median_plateau_compact` inside the 3-state sizing study because the latter's edge is outlier-driven.
- Reporting recommendation: do not describe `single_15_60` as the implemented retained signal itself.

## 2. Research Map

The strategy family trades **Micro Nasdaq-100 futures (`MNQ`)** on an intraday horizon and is designed to capture **post-opening-range continuation** after the US cash open. The important point is that the repo currently contains both a retained-final signal stack and a separate sizing study.

{_markdown_table(architecture_table)}

The clean read is:

- `retained final` answers: "what ORB sleeve was retained from the main research campaign?"
- `single_15_60` answers: "which 3-state realized-volatility sizing variant survived the later smoke-plus-audit process?"

Why MNQ:

- The contract is liquid around the US open.
- The instrument exhibits frequent directional bursts after the first regular-session range is defined.
- The micro contract supports controlled risk granularity while preserving the same underlying microstructure intuition as NQ.

The retained architecture separates four layers:

- **Signal**: detect a valid breakout of the opening range.
- **Execution**: wait for the signal bar to close, then enter at the next bar open.
- **Risk management**: stop tied to the opening range plus buffers, with a fixed target at `initial_risk * {retained_target_multiple:.1f}R`.
- **Overlay logic**: either the retained-final ATR/compression/dynamic filters, or the separate 3-state realized-volatility sizing study.

## 3. Implemented Retained ORB Sleeve

The implemented retained sleeve is **not** the 30-minute both-direction baseline. The retained sleeve in the main ORB campaign is the configuration labeled `{retained_spec.get("name")}` and corresponds to a **15-minute opening range**, **long-only**, beginning at `{retained_opening_time}`.

After the range expires, the strategy checks whether the **bar close** breaks above `or_high + {retained_entry_buffer_ticks}` ticks. Execution still happens on the **next bar open**. The retained configuration is:

- Instrument: `MNQ`
- OR window: `{retained_or_minutes}` minutes
- Direction: `{retained_direction}`
- One trade per day: `{retained_spec.get("one_trade_per_day")}`
- Entry buffer: `{retained_entry_buffer_ticks}` ticks
- Stop buffer: `{retained_stop_buffer_ticks}` ticks
- Target multiple: `{retained_target_multiple:.1f}R`
- VWAP confirmation: `{"enabled" if retained_vwap_confirmation else "disabled"}`
- VWAP column: `{retained_vwap_column}`
- Time exit: `{retained_time_exit}`
- Base risk per trade: `{retained_risk_pct:.2f}%`

The retained code path confirms the signal on the **breakout bar close** and then enters on the **next bar open**. The stop remains tied to the opening range, not to later intraday structure. This is the sleeve the user was remembering when referring to `15 min long`.

{_markdown_image(figures["orb_mechanics_diagram.png"], output_dir, "ORB mechanics diagram with OR high, OR low, breakout close, next-open entry, stop and target")}

## 4. Where VWAP and ATR Live

The missing VWAP and ATR pieces sit in the **retained-final signal stack**, not in the `single_15_60` name itself.

- **VWAP**: the retained-final entry requires confirmation against `{retained_vwap_column}`. This is a signal-quality filter attached to the ORB entry logic.
- **ATR**: the retained-final ensemble uses `ATR({retained_atr_window})` with vote threshold `{retained_vote_threshold:.2f}` and quantile bands `low={retained_q_lows}` / `high={retained_q_highs}` to decide whether a session is retained by the ensemble logic.
- **Compression overlay**: `{retained_compression_mode}` with usage `{retained_compression_usage}`.
- **Dynamic gate**: `{retained_dynamic_mode}` with `noise_lookback={retained_noise_lookback}`, `noise_vm={retained_noise_vm:.1f}`, schedule `{retained_dynamic_schedule}`, and threshold style `{retained_threshold_style}`.
- **Realized-vol ratio 15/60**: this is a different research object used in the separate sizing branch below. It is not the ATR ensemble itself.

This distinction matters operationally:

- ATR and VWAP belong to the **implemented retained sleeve**.
- `single_15_60` belongs to the **3-state sizing study**.
- Both can coexist in the repo without being the same live object.

## 5. Execution and Timing Convention

The timing convention remains explicit:

- `t-1 close -> signal bar close -> filters/overlay computed -> next open execution`

That timing is causal for both branches because the decision inputs are read only after the breakout bar closes.

Backtest assumptions in the retained-final sleeve:

- signal confirmed on the breakout-bar close;
- entry at next open;
- stop anchored to the opening range plus buffer;
- target at fixed `initial_risk * {retained_target_multiple:.1f}R`;
- intraday force-exit at `{retained_time_exit}`;
- fills and fees governed by the existing backtest execution model.

This avoids a look-ahead path where the strategy would size from information that did not yet exist at the moment of the decision. The main limitation is that live slippage can still differ from modeled slippage, especially around the US open and during abrupt microstructure transitions.

## 6. Separate 3-State Sizing Branch

The `single_15_60` branch is a **separate sizing study**. Its nominal baseline, as stored in the sizing exports used by this memo, is:

- OR window: `{three_state_or_minutes}` minutes
- Direction: `{three_state_direction}`
- Opening time: `{three_state_opening_time}`
- Entry buffer: `{three_state_entry_buffer_ticks}` ticks
- Stop buffer: `{three_state_stop_buffer_ticks}` ticks
- Target multiple: `{three_state_target_multiple:.1f}R`
- VWAP confirmation: `{"enabled" if three_state_vwap_confirmation else "disabled"}`
- Time exit: `{three_state_time_exit}`
- Base risk per trade: `{three_state_risk_pct:.2f}%`

This is where the previous memo drifted: it described this `OR30 / both` baseline as if it were the implemented retained-final sleeve. It is not.

Inside this branch, the sizing feature is based on realized volatility computed from minute-bar closes:

```python
returns = close.pct_change()
vol_std_fast = returns.rolling(15).std()
vol_std_slow = returns.rolling(60).std()
realized_vol_ratio_15_60 = vol_std_fast / vol_std_slow
```

The buckets are then mapped to sizing multipliers:

- `low -> 0.50x`
- `mid -> 1.00x`
- `high -> 0.25x`

The operational intuition is simple. The branch keeps full size in the **middle regime**, trims size in the **low regime**, and trims more aggressively in the **high regime** where realized volatility is less favorable for stable risk-adjusted execution.

Important causality note:

- `pandas.Series.rolling().std()` is trailing and inclusive of the current observation.
- In the retained timing convention, the current observation is the **signal bar close**.
- The feature is therefore available after the signal bar closes and before the next-open execution.

Observed bucket boundaries from the reference IS calibration for `realized_vol_ratio_15_60`:

{_markdown_table(bucket_table if not bucket_table.empty else pd.DataFrame([{{"bucket_label": "n/a", "lower_bound": "n/a", "upper_bound": "n/a", "is_n_obs": "n/a", "is_net_pnl": "n/a", "is_sharpe": "n/a", "oos_n_obs": "n/a", "oos_net_pnl": "n/a", "oos_sharpe": "n/a"}}]))}

Interpretation:

- `low` is the lower-volatility-ratio regime up to roughly `{_format_number(low_upper, 6)}`.
- `mid` spans roughly `{_format_number(low_upper, 6)} < ratio <= {_format_number(mid_upper, 6)}`.
- `high` is the upper tail beyond roughly `{_format_number(mid_upper, 6)}`.

## 7. Why `single_15_60` Was Retained Inside the Sizing Branch

The retained `15/60` pair ranks highly on the reference regime-sizing study and sits in a stable local neighborhood rather than standing alone as an isolated spike.

- IS feature-selection score for `realized_vol_ratio_15_60`: `{_format_number(feature_row.get("feature_selection_score") if feature_row is not None else np.nan)}`
- IS score spread across its three buckets: `{_format_number(feature_row.get("is_score_spread") if feature_row is not None else np.nan)}`
- Best IS bucket: `{feature_row.get("best_bucket_is", "n/a") if feature_row is not None else "n/a"}`
- Worst IS bucket: `{feature_row.get("worst_bucket_is", "n/a") if feature_row is not None else "n/a"}`

"15/60 is retained not as a fragile optimized point, but as a conservative representative of a robust local parameter region."

{_markdown_image(figures["vol_fast_slow_heatmap_is.png"], output_dir, "IS heatmap of feature-selection score around the retained 15/60 volatility ratio") if "vol_fast_slow_heatmap_is.png" in figures else "_IS heatmap could not be reconstructed from the available files._"}

{_markdown_image(figures["vol_fast_slow_heatmap_oos.png"], output_dir, "OOS heatmap of retained-policy Sharpe around the retained 15/60 volatility ratio") if "vol_fast_slow_heatmap_oos.png" in figures else "_OOS heatmap could not be reconstructed from the available files._"}

Why not promote the ensembles despite stronger headline Sharpe in the smoke comparison:

- the initial smoke comparison was only a local ranking pass;
- the later audit explicitly tested whether the improvement was broad-based or driven by a few outsized days;
- the later audit reversed the production recommendation back to `single_15_60`.

## 8. Sizing-Branch Robustness Audit

The local comparison set is:

{_markdown_table(comparison_table)}

Headline facts:

- `median_plateau_compact` posts the best headline Sharpe in the local smoke comparison.
- The later audit shows that the apparent edge is **not broad-based**.
- `positive_diff_day_share = {_format_number(plateau_audit_row.get("positive_diff_day_share"), 3)}`
- `positive_diff_month_share = {_format_number(plateau_audit_row.get("positive_diff_month_share"), 3)}`
- `excess_pnl_after_top_5 = {_format_money(plateau_audit_row.get("excess_pnl_after_top_5"))}`
- `maxDD` worsens to `{_format_money(plateau_audit_row.get("max_drawdown"))}` versus `{_format_money(baseline_audit_row.get("max_drawdown"))}` for `single_15_60`

Interpretation:

- only a small minority of months show positive excess versus the baseline;
- once the top positive outliers are removed, the excess PnL turns negative;
- the ensemble also carries a deeper drawdown profile.

Decision:

- do **not** promote `median_plateau_compact` to production at this stage;
- keep it as a research watchlist variant and revisit only if OOS breadth improves materially.

{_markdown_image(figures["variant_sharpe_comparison.png"], output_dir, "Sharpe comparison across tested 3-state variants")}

{_markdown_image(figures["variant_net_pnl_comparison.png"], output_dir, "Net PnL comparison across tested 3-state variants")}

{_markdown_image(figures["variant_maxdd_comparison.png"], output_dir, "Max drawdown comparison across tested 3-state variants")}

{_markdown_image(figures["baseline_vs_ensemble_excess_contribution.png"], output_dir, "Contribution of the top positive days to the apparent edge of `median_plateau_compact` versus `single_15_60`") if "baseline_vs_ensemble_excess_contribution.png" in figures else "_Excess-contribution figure was not generated._"}

{_markdown_image(figures["monthly_excess_pnl_plateau_vs_baseline.png"], output_dir, "Monthly excess PnL of `median_plateau_compact` versus `single_15_60`") if "monthly_excess_pnl_plateau_vs_baseline.png" in figures else "_Monthly excess-PnL figure was not generated._"}

## 9. Performance Summary for the Sizing Branch

The figures below quantify the **3-state sizing branch**, not the full retained-final ORB sleeve. Within that branch, the retained production candidate is the **single-pair 15/60 overlay** with the live-oriented `0.50x / 1.00x / 0.25x` policy.

{_markdown_table(metrics_table)}

Contextual comparisons:

- Current retained headline Sharpe for `single_15_60`: `{_format_number(baseline_audit_row.get("sharpe"))}`
- OOS headline of the earlier research reference `sizing_3state_realized_vol_ratio_15_60` with `high=0.75x`: Sharpe `{_format_number(reference_variant_row.get("oos_sharpe"))}`, max drawdown `{_format_money(reference_variant_row.get("oos_max_drawdown"))}`
- This memo documents the **current retained live-oriented sizing policy for that branch**, not the earlier `high=0.75x` reference variant.
- These charts should not be read as a full scorecard of the implemented `retained final` OR15 long-only sleeve.

{_markdown_image(figures["equity_curve_single_15_60.png"], output_dir, "Equity curve for `single_15_60`") if "equity_curve_single_15_60.png" in figures else "_Equity curve figure was not generated._"}

{_markdown_image(figures["drawdown_curve_single_15_60.png"], output_dir, "Drawdown curve for `single_15_60`") if "drawdown_curve_single_15_60.png" in figures else "_Drawdown curve figure was not generated._"}

{_markdown_image(figures["monthly_returns_single_15_60.png"], output_dir, "Monthly returns for `single_15_60`") if "monthly_returns_single_15_60.png" in figures else "_Monthly returns figure was not generated._"}

{_markdown_image(figures["trade_pnl_distribution_single_15_60.png"], output_dir, "Trade-level PnL distribution for `single_15_60`") if "trade_pnl_distribution_single_15_60.png" in figures else "_Trade-PnL distribution figure was not generated._"}

## 10. Risk and Failure Modes

- **Regime risk**: ORB continuation works best when the post-open flow is directional. It can degrade in choppy or reversal-heavy opens.
- **Parameter-selection risk**: even a locally robust region can drift as microstructure evolves.
- **Execution risk**: next-open fills can differ from backtest fills, especially after sharp breakout bars.
- **Slippage risk**: stop-outs near the open can be materially worse live than in a simplified execution model.
- **Latency risk**: any live delay between signal-close confirmation and next-open execution can alter entry quality.
- **Microstructure drift**: queue dynamics, participation mix, and opening behavior can change over time.
- **Prop-style constraint risk**: the strategy still needs monitoring against daily loss and max-loss rules despite passing the historical constraint summary.
- **Calendar and news sensitivity**: CPI, FOMC, payrolls, and large earnings clusters can change the open materially.
- **PnL concentration risk**: the audit already shows that some apparent improvements can be concentrated in a small set of favorable days.

## 11. Production Recommendation

Current recommendation:

- keep the implemented `retained final` sleeve clearly identified as the `OR15 / long-only / VWAP / ATR / weak_close / noise_area_gate` stack;
- inside the separate 3-state sizing branch, retain `single_15_60`;
- retain the sizing policy `low=0.50x`, `mid=1.00x`, `high=0.25x` for that branch;
- do not promote `median_plateau_compact` to production today;
- keep `median_plateau_compact` on the research watchlist;
- do not merge the retained-final signal narrative with the sizing-branch narrative in future reporting;
- repeat the audit if the OOS sample broadens materially.

This is a conservative decision. It favors **breadth and interpretability** over a modest headline uplift that is not yet supported by the month-level and outlier-removal diagnostics.

## 12. Implementation Traceability

{_markdown_table(traceability_table)}

## 13. Appendix

### Retained-final sleeve snapshot

- Research name: `{retained_spec.get("name")}`
- Opening time: `{retained_opening_time}`
- OR window: `{retained_or_minutes}` minutes
- Direction: `{retained_direction}`
- One trade per day: `{retained_spec.get("one_trade_per_day")}`
- Entry buffer: `{retained_entry_buffer_ticks}` ticks
- Stop buffer: `{retained_stop_buffer_ticks}` ticks
- Target multiple: `{retained_target_multiple:.1f}`
- VWAP confirmation: `{"enabled" if retained_vwap_confirmation else "disabled"}`
- ATR window / vote threshold: `{retained_atr_window}` / `{retained_vote_threshold:.2f}`
- Compression: `{retained_compression_mode}` via `{retained_compression_usage}`
- Dynamic gate: `{retained_dynamic_mode}`
- Time exit: `{retained_time_exit}`
- Account size: `{_format_money(retained_initial_capital)}`
- Base risk per trade: `{retained_risk_pct:.2f}%`

### 3-state sizing-branch baseline snapshot

- Opening time: `{three_state_opening_time}`
- OR window: `{three_state_or_minutes}` minutes
- Direction: `{three_state_direction}`
- Entry buffer: `{three_state_entry_buffer_ticks}` ticks
- Stop buffer: `{three_state_stop_buffer_ticks}` ticks
- Target multiple: `{three_state_target_multiple:.1f}`
- Time exit: `{three_state_time_exit}`
- Account size used in sizing export: `{_format_money(three_state_initial_capital)}`
- Base risk per trade: `{three_state_risk_pct:.2f}%`
- Retained branch policy: `low=0.50x`, `mid=1.00x`, `high=0.25x`

### Exports used

- Variant smoke export: `{bundle.variant_export}`
- Variant audit export: `{bundle.audit_export}`
- Reference regime export: `{bundle.regime_export if bundle.regime_export is not None else "n/a"}`
- Executed client notebook: `{bundle.notebook_ipynb_path if bundle.notebook_ipynb_path is not None else "n/a"}`
- Retained-final source of truth: `src/analytics/audit_mnq_orb_retained_vs_3state.py::RetainedConfig`
- Retained-final notebook source: `notebooks/finals/mnq_orb_retained_final.executed.ipynb`

### Reproduction commands

{_markdown_table(appendix_commands)}

### Known limitations

- The memo is only as good as the retained exports and current processed MNQ minute dataset.
- The standalone executed HTML notebook referenced in the instruction note was not present locally.
- Heatmaps were reconstructed from the available code and datasets rather than copied from notebook screenshots.
- The memo does not rerun the full historical research campaign; it rebuilds the report from the retained exports and a local fast/slow neighborhood reconstruction.
- The sizing-branch figures in this memo are not a substitute for a dedicated retained-final performance report.

### Data availability notes

{chr(10).join(f"- {note}" for note in data_availability_notes) if data_availability_notes else "- No missing-data issues were detected during memo construction."}

### Recommended next steps

- continue live-paper or shadow monitoring on `single_15_60`;
- keep collecting OOS evidence on whether the `median_plateau_compact` uplift broadens beyond a small set of outlier days;
- re-run the audit after a materially larger OOS sample rather than responding to isolated headline changes.
"""
    return markdown, verdict_summary


def build_strategy_memo(config: MemoBuildConfig) -> MemoArtifacts:
    bundle = _load_bundle(config)
    output_dir = config.output_dir.resolve()
    figures_dir = output_dir / FIGURES_SUBDIR

    figure_paths, figure_notes = _build_figures(
        bundle,
        figures_dir,
        include_stability_heatmaps=config.include_stability_heatmaps,
    )
    markdown_text, verdict_summary = _build_markdown(bundle, output_dir, figure_paths, figure_notes)
    html_text = _markdown_to_html(markdown_text, "MNQ ORB Strategy - Institutional Research Memo")

    markdown_path = output_dir / DOCS_FILENAME
    html_path = output_dir / HTML_FILENAME
    markdown_path.write_text(markdown_text, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")

    return MemoArtifacts(
        markdown_path=markdown_path,
        html_path=html_path,
        figures_dir=figures_dir,
        figure_paths=figure_paths,
        data_availability_notes=[*bundle.data_availability_notes, *figure_notes],
        verdict_summary=verdict_summary,
    )


def _parse_args() -> MemoBuildConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Target docs directory.")
    parser.add_argument("--variant-export", type=Path, required=True, help="Path to the variant smoke export.")
    parser.add_argument("--audit-export", type=Path, required=True, help="Path to the variant audit export.")
    parser.add_argument(
        "--regime-export",
        type=Path,
        default=None,
        help="Optional reference regime export. If omitted, it is inferred from the variant export metadata.",
    )
    parser.add_argument(
        "--skip-heatmaps",
        action="store_true",
        help="Skip the local fast/slow heatmap reconstruction.",
    )
    args = parser.parse_args()
    return MemoBuildConfig(
        output_dir=args.output_dir,
        variant_export=args.variant_export,
        audit_export=args.audit_export,
        regime_export=args.regime_export,
        include_stability_heatmaps=not args.skip_heatmaps,
    )


def main() -> None:
    config = _parse_args()
    artifacts = build_strategy_memo(config)
    print(json.dumps(
        {
            "markdown_path": str(artifacts.markdown_path),
            "html_path": str(artifacts.html_path),
            "figures_dir": str(artifacts.figures_dir),
            "figure_paths": [str(path) for path in artifacts.figure_paths],
            "data_availability_notes": artifacts.data_availability_notes,
            "verdict_summary": artifacts.verdict_summary,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
