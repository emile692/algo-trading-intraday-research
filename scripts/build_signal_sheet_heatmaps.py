"""Build the parameter-validation figures embedded in the nine signal sheets.

Every colour is computed from an existing research artefact or, for the MNQ
compression grid, from the frozen backtest engine.  No illustrative/synthetic
performance value is used.
"""

from __future__ import annotations

import copy
import io
import subprocess
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from research.baseline_backtests.baseline_engine import (
    build_buy_and_hold,
    load_frozen_specs,
    prepare_data,
    run_baseline_backtest,
    slice_prepared_data,
)
from research.money_management_lab.run_final_mm_study import metrics_for_period


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "signal_sheets" / "assets"
FROZEN_DIR = ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
MNQ_DATA = ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"
VOLATILITY_CUBE_DATA = ROOT / "research" / "signal_sheet_02_cube" / "cube_metrics.csv"
NINE_SLEEVE_DAILY = ROOT / "research" / "locked_nine_sleeve_portfolio_20261101" / "daily_pnl.csv"
# Since 2026-11-15 the book figures use the active lock (S01 = 18-model ATR ensemble, union calendar) plus S04
# (shadow) taken from the 2026-11-01 lock, at 0 on the union-only sessions (no MNQ bar on quarterly expiry Fridays).
EIGHT_SLEEVE_DAILY_20261115 = ROOT / "research" / "locked_eight_sleeve_portfolio_20261115" / "daily_pnl.csv"
BOOK_RESEARCH_START = pd.Timestamp("2024-01-02")
BOOK_RESEARCH_END = pd.Timestamp("2026-03-19")

SLEEVE_LABELS = {
    "mnq_orb_vwap_atr": "01 · MNQ ORB + VWAP + ATR",
    "mnq_volatility_breakout": "02 · MNQ volatility breakout",
    "mnq_compression_breakout": "03 · MNQ compression breakout",
    "mnq_atr_trend_flip": "04 · MNQ ATR trend flip",
    "mgc_volatility_breakout": "05 · MGC volatility breakout",
    "mgc_gap_continuation": "06 · MGC gap continuation",
    "mbt_rolling_range_breakout": "07 · MBT rolling range breakout",
    "mbt_trend_pullback_ma_structure": "08 · MBT trend pullback MA",
    "mbt_volatility_breakout": "09 · MBT volatility breakout",
}


def _plot_heatmap(
    matrix: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    selected: list[tuple[int, int]] | None = None,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    bound = max(0.5, float(np.max(np.abs(finite)))) if finite.size else 1.0
    fig, ax = plt.subplots(figsize=(max(7.2, len(xlabels) * 0.82 + 3.2), max(3.3, len(ylabels) * 0.62 + 2.3)))
    image = ax.imshow(matrix, cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, weight="bold")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, "—" if not np.isfinite(value) else f"{value:.2f}", ha="center", va="center", fontsize=8)
    for row, col in selected or []:
        ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#1669b2", linewidth=3))
    fig.colorbar(image, ax=ax, label="Sharpe annualisé")
    fig.tight_layout()
    fig.savefig(path, dpi=190, facecolor="white")
    plt.close(fig)


def _git_csv(path: str) -> pd.DataFrame:
    raw = subprocess.run(
        ["git", "show", f"HEAD:{path}"], cwd=ROOT, check=True, capture_output=True
    ).stdout.decode("utf-8")
    return pd.read_csv(io.StringIO(raw))


def build_orb() -> None:
    tables = [
        ("IS — construction", _git_csv("data/exports/atr_local_robustness_20260320_194355/tables/metrics_is.csv"), "is_sharpe_ratio"),
        ("OOS — évaluation", _git_csv("data/exports/atr_local_robustness_20260320_194355/tables/metrics_oos.csv"), "oos_sharpe_ratio"),
    ]
    periods = sorted(set().union(*(set(frame["atr_period"]) for _, frame, _ in tables)))
    lows = sorted(set().union(*(set(frame["q_low_pct"]) for _, frame, _ in tables)))
    highs = sorted(set().union(*(set(frame["q_high_pct"]) for _, frame, _ in tables)))
    period_pos = {value: index for index, value in enumerate(periods)}
    low_pos = {value: index for index, value in enumerate(lows)}
    high_pos = {value: index for index, value in enumerate(highs)}
    sharpe_values = pd.concat([frame[metric] for _, frame, metric in tables], ignore_index=True)
    color_min, color_max = float(sharpe_values.min()), float(sharpe_values.max())

    fig = plt.figure(figsize=(15.4, 7.1))
    axes = [fig.add_subplot(1, 2, index, projection="3d") for index in (1, 2)]
    for ax, (title, data, metric) in zip(axes, tables):
        cloud = ax.scatter(
            data["atr_period"].map(period_pos),
            data["q_low_pct"].map(low_pos),
            data["q_high_pct"].map(high_pos),
            c=data[metric],
            cmap="viridis",
            vmin=color_min,
            vmax=color_max,
            s=120,
            alpha=0.88,
            edgecolors="#26384d",
            linewidths=0.35,
            depthshade=False,
        )
        retained = data[
            data["atr_period"].isin([14, 16, 18])
            & data["q_low_pct"].isin([20, 25, 30])
            & data["q_high_pct"].isin([80, 85])
        ]
        ax.scatter(
            retained["atr_period"].map(period_pos),
            retained["q_low_pct"].map(low_pos),
            retained["q_high_pct"].map(high_pos),
            s=340,
            marker="s",
            facecolors="none",
            edgecolors="#0875d1",
            linewidths=2.8,
            label="18 sous-modèles retenus",
            depthshade=False,
        )
        ax.set_xlabel("Période ATR (séances)", labelpad=9)
        ax.set_ylabel("Quantile bas (%)", labelpad=9)
        ax.set_zlabel("Quantile haut (%)", labelpad=9)
        ax.set_xticks(range(len(periods)), [str(value) for value in periods])
        ax.set_yticks(range(len(lows)), [str(value) for value in lows])
        ax.set_zticks(range(len(highs)), [str(value) for value in highs])
        ax.view_init(elev=24, azim=-55)
        ax.set_title(title, weight="bold", pad=12)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.02))
    fig.suptitle("MNQ ORB — cubes de robustesse des trois paramètres ATR", weight="bold", fontsize=15, y=0.98)
    fig.subplots_adjust(left=0.01, right=0.89, bottom=0.07, top=0.90, wspace=0.02)
    color_axis = fig.add_axes([0.91, 0.20, 0.015, 0.58])
    fig.colorbar(cloud, cax=color_axis, label="Sharpe annualisé")
    fig.savefig(ASSET_DIR / "01_orb_is_heatmap.png", dpi=200, facecolor="white")
    plt.close(fig)


def build_money_management() -> None:
    data = pd.read_csv(ROOT / "research" / "money_management_lab" / "final_mm_results.csv")
    definitions = [
        ("atr_trend_flip", "stop_atr", 1.0, "04_atr_trend_flip_is_heatmap.png", "MNQ ATR trend flip — distance du stop", "Stop (multiple d'ATR14)"),
    ]
    for mechanism, parameter, lock, filename, title, xlabel in definitions:
        frame = data[(data["mechanism_id"] == mechanism) & (data["parameter_name"] == parameter)].sort_values("parameter_value")
        values = frame["parameter_value"].tolist()
        matrix = frame[["is_sharpe_annualized", "validation_sharpe_annualized"]].to_numpy().T
        selected = [(row, values.index(lock)) for row in range(2)]
        _plot_heatmap(matrix, [f"{v:g}" for v in values], ["IS", "Validation"], title, xlabel, "Période", ASSET_DIR / filename, selected)


def build_volatility_breakout_cube() -> None:
    data = pd.read_csv(VOLATILITY_CUBE_DATA)
    periods = sorted(data["daily_atr_period"].unique())
    multiples = sorted(data["entry_atr_multiple"].unique())
    durations = sorted(data["max_holding_minutes"].unique())
    period_pos = {value: index for index, value in enumerate(periods)}
    multiple_pos = {value: index for index, value in enumerate(multiples)}
    duration_pos = {value: index for index, value in enumerate(durations)}
    metrics = ("is_sharpe_annualized", "oos_sharpe_annualized")
    color_min = float(data[list(metrics)].min().min())
    color_max = float(data[list(metrics)].max().max())

    fig = plt.figure(figsize=(15.4, 7.1))
    axes = [fig.add_subplot(1, 2, index, projection="3d") for index in (1, 2)]
    for ax, title, metric in zip(axes, ("IS — construction", "OOS / validation"), metrics):
        cloud = ax.scatter(
            data["daily_atr_period"].map(period_pos),
            data["entry_atr_multiple"].map(multiple_pos),
            data["max_holding_minutes"].map(duration_pos),
            c=data[metric],
            cmap="viridis",
            vmin=color_min,
            vmax=color_max,
            s=125,
            alpha=0.88,
            edgecolors="#26384d",
            linewidths=0.35,
            depthshade=False,
        )
        retained = data[
            data["daily_atr_period"].isin(range(10, 23, 2))
            & data["entry_atr_multiple"].isin([0.5, 0.6])
            & data["max_holding_minutes"].isin([270, 300, 360])
        ]
        ax.scatter(
            retained["daily_atr_period"].map(period_pos),
            retained["entry_atr_multiple"].map(multiple_pos),
            retained["max_holding_minutes"].map(duration_pos),
            s=380,
            marker="s",
            facecolors="none",
            edgecolors="#0875d1",
            linewidths=3,
            label="42 sous-modèles retenus",
            depthshade=False,
        )
        ax.set_xlabel("Période ATR (séances)", labelpad=9)
        ax.set_ylabel("Multiple d’entrée (ATR)", labelpad=9)
        ax.set_zlabel("Durée max (minutes)", labelpad=9)
        ax.set_xticks(range(len(periods)), [f"{value:g}" for value in periods])
        ax.set_yticks(range(len(multiples)), [f"{value:.1f}" for value in multiples])
        ax.set_zticks(range(len(durations)), [f"{value:g}" for value in durations])
        ax.view_init(elev=23, azim=-56)
        ax.set_title(title, weight="bold", pad=12)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.02))
    fig.suptitle("MNQ volatility breakout — cubes de robustesse des trois paramètres", weight="bold", fontsize=15, y=0.98)
    fig.subplots_adjust(left=0.01, right=0.89, bottom=0.07, top=0.90, wspace=0.02)
    color_axis = fig.add_axes([0.91, 0.20, 0.015, 0.58])
    fig.colorbar(cloud, cax=color_axis, label="Sharpe annualisé")
    fig.savefig(ASSET_DIR / "02_volatility_breakout_is_heatmap.png", dpi=200, facecolor="white")
    plt.close(fig)


def build_compression() -> None:
    specs, benchmark_spec, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    spec = next(item for item in specs if item["mechanism_id"] == "compression_breakout")
    prepared = prepare_data(MNQ_DATA, start_date=date(2019, 5, 6), end_date=date(2026, 3, 19))
    periods = [
        ("IS", date(2019, 5, 6), date(2023, 12, 31)),
        ("Validation", date(2024, 1, 1), date(2026, 3, 19)),
    ]
    lookbacks, buffers = [7, 10, 14], [0, 1, 2]
    matrices = [np.full((len(buffers), len(lookbacks)), np.nan) for _ in periods]
    benchmarks = {
        label: build_buy_and_hold(benchmark_spec, slice_prepared_data(prepared, start, end)).daily
        for label, start, end in periods
    }
    for row, buffer_ticks in enumerate(buffers):
        for col, lookback in enumerate(lookbacks):
            candidate = copy.deepcopy(spec)
            structural = candidate["canonical_parameters"]["structural"]
            structural["nr_lookback"] = lookback
            structural["entry_buffer_ticks"] = buffer_ticks
            run = run_baseline_backtest(candidate, prepared)
            for matrix, (label, start, end) in zip(matrices, periods):
                metrics = metrics_for_period(candidate, run, benchmarks[label], start, end)
                matrix[row, col] = metrics["sharpe_annualized"]

    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    bound = max(0.5, float(np.max(np.abs(finite))))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7), sharey=True)
    for ax, matrix, (label, _, _) in zip(axes, matrices, periods):
        image = ax.imshow(matrix, cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto")
        ax.set_xticks(range(len(lookbacks)), [str(v) for v in lookbacks])
        ax.set_yticks(range(len(buffers)), [str(v) for v in buffers])
        ax.set_xlabel("Lookback NR (barres 1 h)")
        ax.set_title(label, weight="bold")
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center", fontsize=9)
        ax.add_patch(Rectangle((lookbacks.index(10) - 0.5, buffers.index(1) - 0.5), 1, 1, fill=False, edgecolor="#1669b2", linewidth=3))
    axes[0].set_ylabel("Buffer (ticks)")
    fig.suptitle("MNQ compression breakout — robustesse IS et validation", weight="bold")
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.16, top=0.82, wspace=0.12)
    color_axis = fig.add_axes([0.91, 0.19, 0.018, 0.58])
    fig.colorbar(image, cax=color_axis, label="Sharpe annualisé")
    fig.savefig(ASSET_DIR / "03_compression_breakout_is_heatmap.png", dpi=190, facecolor="white")
    plt.close(fig)


def build_mbt() -> None:
    data = pd.read_csv(ROOT / "research" / "mbt_three_sleeve_freeze_20261004" / "cell_metrics.csv")
    definitions = [
        ("rolling_range_breakout", "entry_channel_length", "exit_channel_length", "07_mbt_rolling_range_is_heatmap.png", "MBT rolling range — legacy IS"),
        ("trend_pullback_ma_structure", "fast_pair_scale", "slow_tail_scale", "08_mbt_trend_pullback_is_heatmap.png", "MBT trend pullback — legacy IS"),
        ("volatility_breakout", "daily_atr_period", "entry_atr_multiple", "09_mbt_volatility_breakout_is_heatmap.png", "MBT volatility breakout — legacy IS"),
    ]
    for mechanism, x, y, filename, title in definitions:
        frame = data[data["mechanism_id"] == mechanism].copy()
        xs, ys = sorted(frame[x].dropna().unique()), sorted(frame[y].dropna().unique())
        matrices = [
            frame[frame["scope"] == scope].pivot(index=y, columns=x, values="sharpe_annualized").reindex(index=ys, columns=xs).to_numpy()
            for scope in ("legacy_is", "legacy_oos")
        ]
        finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
        bound = max(0.5, float(np.max(np.abs(finite))))
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
        for ax, matrix, scope_title in zip(axes, matrices, ("Legacy IS", "Legacy OOS réutilisé")):
            image = ax.imshow(matrix, cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto")
            ax.set_xticks(range(len(xs)), [f"{v:g}" for v in xs])
            ax.set_yticks(range(len(ys)), [f"{v:g}" for v in ys])
            ax.set_xlabel(x.replace("_", " "))
            ax.set_title(scope_title, weight="bold")
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=7.5)
            for row in range(len(ys)):
                for col in range(len(xs)):
                    ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#1669b2", linewidth=1.5))
        axes[0].set_ylabel(y.replace("_", " "))
        fig.suptitle(title.replace("legacy IS", "robustesse sur les deux périodes"), weight="bold")
        fig.subplots_adjust(left=0.08, right=0.88, bottom=0.16, top=0.82, wspace=0.12)
        color_axis = fig.add_axes([0.91, 0.19, 0.018, 0.58])
        fig.colorbar(image, cax=color_axis, label="Sharpe annualisé")
        fig.savefig(ASSET_DIR / filename, dpi=190, facecolor="white")
        plt.close(fig)


def _book_metrics(daily_pnl: pd.Series) -> dict[str, float]:
    """Return comparable daily metrics, including the zero-PnL origin."""
    daily_pnl = daily_pnl.astype(float)
    equity_with_origin = np.r_[0.0, daily_pnl.cumsum().to_numpy()]
    drawdown = equity_with_origin - np.maximum.accumulate(equity_with_origin)
    volatility = float(daily_pnl.std(ddof=1))
    sharpe = float(np.sqrt(252.0) * daily_pnl.mean() / volatility) if volatility > 0 else np.nan
    return {
        "net_pnl_usd": float(daily_pnl.sum()),
        "daily_volatility_usd": volatility,
        "annualized_sharpe": sharpe,
        "worst_day_usd": float(daily_pnl.min()),
        "max_drawdown_usd": float(drawdown.min()),
    }


def build_equity_and_book() -> None:
    """Build book figures on the immutable common research window.

    The date filter prevents appended observations from entering the book
    diagnostics. Individual FS/OOS curves are generated by the dedicated
    ``build_equity_curves_fs_oos.py`` script and are not touched here.
    """
    eight = pd.read_csv(EIGHT_SLEEVE_DAILY_20261115, parse_dates=["session_date"]).set_index("session_date")
    s04 = pd.read_csv(NINE_SLEEVE_DAILY, parse_dates=["session_date"]).set_index("session_date")["mnq_atr_trend_flip"]
    if not set(s04.index) <= set(eight.index):
        raise RuntimeError("S04 has sessions outside the union calendar of the 2026-11-15 lock")
    data = eight.assign(mnq_atr_trend_flip=s04.reindex(eight.index).fillna(0.0)).reset_index()
    data = data.loc[
        data["session_date"].between(BOOK_RESEARCH_START, BOOK_RESEARCH_END, inclusive="both")
    ].copy()
    if data.empty or data["session_date"].min() != BOOK_RESEARCH_START or data["session_date"].max() != BOOK_RESEARCH_END:
        raise RuntimeError(
            "The frozen book research window must span exactly "
            f"{BOOK_RESEARCH_START.date()} to {BOOK_RESEARCH_END.date()}"
        )
    sleeves = list(SLEEVE_LABELS)
    pnl = data[sleeves].fillna(0.0)

    corr = pnl.corr()
    short = [f"S{i:02d}" for i in range(1, 10)]

    book_equity = pnl.sum(axis=1).cumsum()
    fig, ax = plt.subplots(figsize=(10.8, 4.1))
    ax.plot(data["session_date"], book_equity, color="#102b4e", linewidth=2.0)
    ax.fill_between(data["session_date"], 0, book_equity, color="#65c3ff", alpha=0.2)
    ax.axhline(0, color="#7b8794", linewidth=0.8)
    ax.set_title("Book equal-contract — equity curve nette", weight="bold")
    ax.set_ylabel("PnL cumulé (USD)")
    ax.set_xlabel("Fenêtre commune : 2 janv. 2024 — 19 mars 2026")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "book_equity_curve.png", dpi=190, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 7.7))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(9), short, rotation=45, ha="right")
    ax.set_yticks(range(9), short)
    for row in range(9):
        for col in range(9):
            value = corr.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7.7,
                    color="white" if abs(value) > 0.55 else "#172033")
    ax.set_title("Book — corrélation de Pearson des PnL quotidiens", weight="bold")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Corrélation")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "book_correlation_matrix.png", dpi=200, facecolor="white")
    plt.close(fig)

    vols = pnl.std(ddof=1)
    equity = pnl.cumsum()
    maxdds = (equity - equity.cummax()).min().abs()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    axes[0].bar(short, vols, color="#1769aa")
    axes[0].set_title("Volatilité quotidienne")
    axes[0].set_ylabel("USD / jour")
    axes[1].bar(short, maxdds, color="#c45a4a")
    axes[1].set_title("Drawdown maximal historique")
    axes[1].set_ylabel("USD")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Book risk — comparaison des neuf sleeves", weight="bold")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "book_risk_profile.png", dpi=190, facecolor="white")
    plt.close(fig)

    full_metrics = _book_metrics(pnl.sum(axis=1))
    leave_one_out_rows = []
    for code, omitted in zip(short, sleeves):
        metrics = _book_metrics(pnl.drop(columns=omitted).sum(axis=1))
        leave_one_out_rows.append(
            {
                "omitted_code": code,
                "omitted_sleeve": SLEEVE_LABELS[omitted],
                **metrics,
                "net_pnl_change_vs_full_usd": metrics["net_pnl_usd"] - full_metrics["net_pnl_usd"],
                "max_drawdown_change_vs_full_usd": metrics["max_drawdown_usd"] - full_metrics["max_drawdown_usd"],
                "research_start": BOOK_RESEARCH_START.date().isoformat(),
                "research_end": BOOK_RESEARCH_END.date().isoformat(),
                "sessions": len(data),
            }
        )
    leave_one_out = pd.DataFrame(leave_one_out_rows)
    leave_one_out.to_csv(ASSET_DIR / "book_leave_one_out.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.9))
    chart_specs = [
        ("net_pnl_usd", "PnL net final", "USD", full_metrics["net_pnl_usd"], "#1769aa"),
        ("annualized_sharpe", "Sharpe annualisé", "Ratio", full_metrics["annualized_sharpe"], "#2b8a6e"),
        ("max_drawdown_usd", "Drawdown maximal", "USD", full_metrics["max_drawdown_usd"], "#c45a4a"),
    ]
    for ax, (column, title, ylabel, full_value, color) in zip(axes, chart_specs):
        ax.bar(short, leave_one_out[column], color=color, alpha=0.88)
        ax.axhline(full_value, color="#172033", linewidth=1.4, linestyle="--", label="Book complet")
        ax.set_title(title, weight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Leave-one-out — book privé d’une sleeve à la fois", weight="bold")
    fig.text(0.5, 0.015, f"Fenêtre de recherche figée : 2 janv. 2024 — 19 mars 2026 ({len(data)} séances, calendrier union)",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(ASSET_DIR / "book_leave_one_out.png", dpi=190, facecolor="white")
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    build_orb()
    build_volatility_breakout_cube()
    build_money_management()
    build_compression()
    build_mbt()
    build_equity_and_book()
    for path in sorted(ASSET_DIR.glob("*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
