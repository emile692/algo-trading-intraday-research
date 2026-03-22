"""Wide Q-low / Q-high heatmaps for Ensemble ORB transfer review."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.orb_multi_asset_campaign import (  # noqa: E402
    BaselineSpec,
    SearchGrid,
    _build_session_sanity,
    _build_strategy,
    _evaluate_point_grid,
    _prepare_feature_dataset,
    _run_baseline_backtest,
    _selected_candidate_rows,
    _split_sessions,
    resolve_processed_dataset,
)
from src.config.settings import get_instrument_spec  # noqa: E402

REPO_ROOT = ROOT
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "export" / "orb_multi_asset_campaign"
DEFAULT_CHARTS_ROOT = DEFAULT_OUTPUT_ROOT / "charts" / "wide_q_heatmaps"
DEFAULT_ATR_WINDOWS = (10, 14, 20, 26, 30, 40, 60)


@dataclass(frozen=True)
class WideHeatmapConfig:
    symbols: tuple[str, ...] = ("MES", "M2K")
    atr_periods: tuple[int, ...] = DEFAULT_ATR_WINDOWS
    q_lows_pct: tuple[int, ...] = tuple(range(0, 31))
    q_highs_pct: tuple[int, ...] = tuple(range(70, 101))
    or_minutes: int = 15
    direction: str = "long"
    target_multiple: float = 2.0
    metric_scope: str = "oos"
    is_fraction: float = 0.70
    data_timeframe: str | None = None
    output_root: Path = DEFAULT_OUTPUT_ROOT


def _wide_grid(config: WideHeatmapConfig) -> SearchGrid:
    return SearchGrid(
        atr_periods=config.atr_periods,
        q_lows_pct=config.q_lows_pct,
        q_highs_pct=config.q_highs_pct,
        aggregation_rules=("majority_50",),
    )


def _tag(config: WideHeatmapConfig) -> str:
    base = f"or{int(config.or_minutes)}"
    if str(config.metric_scope).lower() != "oos":
        base = f"{base}_{str(config.metric_scope).lower()}"
    if config.data_timeframe:
        clean_timeframe = str(config.data_timeframe).strip().lower().replace(" ", "")
        clean_timeframe = clean_timeframe[:-3] + "m" if clean_timeframe.endswith("min") else clean_timeframe
        base = f"{base}_{clean_timeframe}"
    return base


def _metric_prefix(config: WideHeatmapConfig) -> str:
    scope = str(config.metric_scope).lower()
    if scope not in {"overall", "oos", "is"}:
        raise ValueError("metric_scope must be one of: overall, oos, is")
    return scope


def _analyze_points_only(symbol: str, config: WideHeatmapConfig, baseline: BaselineSpec) -> pd.DataFrame:
    dataset = resolve_processed_dataset(symbol, timeframe=config.data_timeframe)
    instrument_spec = get_instrument_spec(symbol)
    grid = _wide_grid(config)

    feat = _prepare_feature_dataset(dataset, baseline, grid)
    session_sanity = _build_session_sanity(feat, baseline)
    all_sessions = sorted(
        pd.to_datetime(session_sanity.loc[session_sanity["has_opening_range"], "session_date"]).dt.date.unique()
    )
    is_sessions, oos_sessions = _split_sessions(all_sessions, config.is_fraction)

    strategy = _build_strategy(baseline, tick_size=float(instrument_spec["tick_size"]))
    signal_df = strategy.generate_signals(feat)
    baseline_trades = _run_baseline_backtest(signal_df, baseline, instrument_spec)
    candidate_df = _selected_candidate_rows(signal_df, grid.atr_periods)
    point_results, _ = _evaluate_point_grid(
        candidate_df=candidate_df,
        baseline_trades=baseline_trades,
        baseline=baseline,
        grid=grid,
        all_sessions=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
    )
    point_results["symbol"] = symbol.upper()
    point_results["dataset_path"] = str(dataset)
    return point_results


def _build_aggregated_table(point_results: pd.DataFrame) -> pd.DataFrame:
    raise RuntimeError("Use _build_aggregated_table_for_scope instead.")


def _build_aggregated_table_for_scope(point_results: pd.DataFrame, prefix: str) -> pd.DataFrame:
    score_col = f"{prefix}_composite_score"
    sharpe_col = f"{prefix}_sharpe"
    pf_col = f"{prefix}_profit_factor"
    net_col = f"{prefix}_net_pnl"
    maxdd_col = f"{prefix}_max_drawdown_abs"

    aggregated = (
        point_results.groupby(["q_low_pct", "q_high_pct"], as_index=False)
        .agg(
            atr_count=("atr_period", "nunique"),
            **{
                f"median_{score_col}": (score_col, "median"),
                f"mean_{score_col}": (score_col, "mean"),
                f"median_{sharpe_col}": (sharpe_col, "median"),
                f"mean_{sharpe_col}": (sharpe_col, "mean"),
                f"median_{pf_col}": (pf_col, "median"),
                f"mean_{pf_col}": (pf_col, "mean"),
                f"median_{net_col}": (net_col, "median"),
                f"mean_{net_col}": (net_col, "mean"),
                f"median_{maxdd_col}": (maxdd_col, "median"),
                f"mean_{maxdd_col}": (maxdd_col, "mean"),
            },
        )
    )

    best_by_pair = (
        point_results.sort_values([score_col, pf_col], ascending=[False, False])
        .drop_duplicates(subset=["q_low_pct", "q_high_pct"])
        .loc[:, ["q_low_pct", "q_high_pct", "atr_period", score_col]]
        .rename(
            columns={
                "atr_period": "best_atr_by_score",
                score_col: f"best_{score_col}",
            }
        )
    )
    return aggregated.merge(best_by_pair, on=["q_low_pct", "q_high_pct"], how="left")


def _plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    center: float | None = None,
    cmap: str = "RdYlGn",
) -> None:
    pivot = (
        df.pivot_table(index="q_low_pct", columns="q_high_pct", values=value_col, aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(12.0, 8.8))
    if pivot.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
    else:
        values = pivot.to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        norm = None
        if center is not None and finite_values.size > 0:
            vmin = float(finite_values.min())
            vmax = float(finite_values.max())
            if vmin < center < vmax:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        im = ax.imshow(values, aspect="auto", origin="lower", cmap=cmap, norm=norm)
        x_idx = list(range(0, len(pivot.columns), 2))
        y_idx = list(range(0, len(pivot.index), 2))
        ax.set_xticks(x_idx)
        ax.set_xticklabels([str(int(pivot.columns[i])) for i in x_idx], rotation=45, ha="right")
        ax.set_yticks(y_idx)
        ax.set_yticklabels([str(int(pivot.index[i])) for i in y_idx])
        ax.set_xlabel("Q_HIGH")
        ax.set_ylabel("Q_LOW")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.92)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _summary_markdown(
    symbol_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    config: WideHeatmapConfig,
    output_path: Path,
) -> None:
    prefix = _metric_prefix(config)
    score_col = f"{prefix}_composite_score"
    sharpe_col = f"{prefix}_sharpe"
    pf_col = f"{prefix}_profit_factor"
    scope_label = "full period (IS+OOS)" if prefix == "overall" else prefix.upper()

    lines = [
        "# Wide Heatmaps Summary",
        "",
        f"- OR window: {int(config.or_minutes)} minutes.",
        f"- Direction: `{config.direction}`.",
        f"- Target multiple: `{config.target_multiple}`.",
        f"- Data timeframe: `{config.data_timeframe or 'latest available'}`.",
        f"- Objective: fixed ATR windows, Q_LOW on Y, Q_HIGH on X, scored on {scope_label}.",
        f"- ATR windows tested: {', '.join(str(v) for v in config.atr_periods)}.",
        "- Logic for ATR windows:",
        "  10/14 = short-term sensitivity, 20/26 = medium-term anchor, 30/40 = smoother swing-adapted range, 60 = slow regime filter.",
        f"- Q_LOW grid: {min(config.q_lows_pct)} to {max(config.q_lows_pct)}.",
        f"- Q_HIGH grid: {min(config.q_highs_pct)} to {max(config.q_highs_pct)}.",
        "",
    ]

    for symbol, (point_results, aggregated) in symbol_tables.items():
        lines.extend([f"## {symbol}", ""])
        top_per_atr = (
            point_results.sort_values([score_col, pf_col], ascending=[False, False])
            .groupby("atr_period", as_index=False)
            .head(1)
            .sort_values("atr_period")
        )
        best_agg = aggregated.sort_values(
            [f"median_{score_col}", f"median_{pf_col}"], ascending=[False, False]
        ).iloc[0]
        lines.append(f"- Best cell per ATR (by {prefix.upper()} composite score):")
        lines.extend(
            [
                f"  - ATR {int(row['atr_period'])}: q{int(row['q_low_pct'])}/q{int(row['q_high_pct'])}, "
                f"score={float(row[score_col]):.3f}, PF={float(row[pf_col]):.3f}, Sharpe={float(row[sharpe_col]):.3f}."
                for _, row in top_per_atr.iterrows()
            ]
        )
        lines.extend(
            [
                "- Aggregated robust pair (median across ATR windows):",
                f"  - q{int(best_agg['q_low_pct'])}/q{int(best_agg['q_high_pct'])}, "
                f"median score={float(best_agg[f'median_{score_col}']):.3f}, "
                f"median PF={float(best_agg[f'median_{pf_col}']):.3f}, "
                f"median Sharpe={float(best_agg[f'median_{sharpe_col}']):.3f}, "
                f"best ATR by score={int(best_agg['best_atr_by_score'])}.",
                "",
            ]
        )

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_wide_heatmaps(config: WideHeatmapConfig) -> dict[str, Path]:
    baseline = BaselineSpec(
        or_minutes=int(config.or_minutes),
        direction=str(config.direction),
        target_multiple=float(config.target_multiple),
    )
    prefix = _metric_prefix(config)
    run_tag = _tag(config)
    charts_root = Path(config.output_root) / "charts" / f"wide_q_heatmaps_{run_tag}"
    charts_root.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        (f"{prefix}_composite_score", "score", f"{prefix.upper()} Composite Score", 0.0),
        (f"{prefix}_sharpe", "sharpe", f"{prefix.upper()} Sharpe", 0.0),
        (f"{prefix}_profit_factor", "pf", f"{prefix.upper()} Profit Factor", 1.0),
        (f"{prefix}_net_pnl", "netpnl", f"{prefix.upper()} Net PnL", 0.0),
        (f"{prefix}_max_drawdown_abs", "maxdd", f"{prefix.upper()} |Max Drawdown|", 0.0),
    ]

    symbol_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for symbol in config.symbols:
        point_results = _analyze_points_only(symbol, config, baseline)
        aggregated = _build_aggregated_table_for_scope(point_results, prefix)
        symbol_lower = symbol.lower()

        point_path = Path(config.output_root) / f"{symbol_lower}_wide_heatmaps_{run_tag}_results.csv"
        aggregated_path = Path(config.output_root) / f"{symbol_lower}_wide_heatmaps_{run_tag}_aggregated.csv"
        point_results.to_csv(point_path, index=False)
        aggregated.to_csv(aggregated_path, index=False)

        for metric, suffix, label, center in metric_specs:
            for atr in config.atr_periods:
                frame = point_results.loc[point_results["atr_period"] == atr].copy()
                out_path = charts_root / f"{symbol_lower}_wide_{run_tag}_{suffix}_atr_{int(atr)}.png"
                _plot_heatmap(
                    frame,
                    value_col=metric,
                    output_path=out_path,
                    title=(
                        f"{symbol} {label} | OR {int(config.or_minutes)}m | Data {config.data_timeframe or 'auto'}"
                        f" | ATR {int(atr)} | Q_LOW x Q_HIGH"
                    ),
                    center=center,
                )

                agg_metric_col = f"median_{metric}"
            out_path = charts_root / f"{symbol_lower}_wide_{run_tag}_{suffix}_aggregated_median.png"
            _plot_heatmap(
                aggregated,
                value_col=agg_metric_col,
                output_path=out_path,
                title=(
                    f"{symbol} {label} | OR {int(config.or_minutes)}m | Data {config.data_timeframe or 'auto'}"
                    " | Aggregated Median Across ATR Windows"
                ),
                center=center,
            )

        symbol_tables[symbol.upper()] = (point_results, aggregated)

    summary_path = Path(config.output_root) / f"wide_q_heatmaps_{run_tag}_summary.md"
    _summary_markdown(symbol_tables, config, summary_path)

    metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "config": {
            **asdict(config),
            "output_root": str(config.output_root),
        },
    }
    metadata_path = Path(config.output_root) / f"wide_q_heatmaps_{run_tag}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return {
        "summary": summary_path,
        "metadata": metadata_path,
        "charts_root": charts_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wide Q_LOW/Q_HIGH heatmaps for Ensemble ORB symbols.")
    parser.add_argument("--symbols", nargs="*", default=["MES", "M2K"])
    parser.add_argument("--or-minutes", type=int, default=15)
    parser.add_argument("--direction", type=str, default="long")
    parser.add_argument("--target-multiple", type=float, default=2.0)
    parser.add_argument("--metric-scope", type=str, default="oos")
    parser.add_argument("--data-timeframe", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    config = WideHeatmapConfig(
        symbols=tuple(str(symbol).upper() for symbol in args.symbols),
        or_minutes=int(args.or_minutes),
        direction=str(args.direction),
        target_multiple=float(args.target_multiple),
        metric_scope=str(args.metric_scope).lower(),
        data_timeframe=str(args.data_timeframe).lower() if args.data_timeframe else None,
        output_root=Path(args.output_root),
    )
    artifacts = run_wide_heatmaps(config)
    print(f"summary: {artifacts['summary']}")
    print(f"charts_root: {artifacts['charts_root']}")


if __name__ == "__main__":
    main()
