"""Build paired IS/OOS local-robustness heatmaps for the 16 frozen strategies on MBT.

Each mechanism is evaluated on a predeclared 3x3 neighbourhood containing its
frozen nominal parameter cell.  Heatmap colour is annualised daily-PnL Sharpe;
cell labels also show net PnL and number of trades.  The chronological 70/30
split is determined once from the common MBT session calendar.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from research.baseline_backtests.baseline_engine import load_frozen_specs, run_baseline_backtest
from research.btc_baseline_backtests.run_btc_baselines import adapt_spec_to_btc, prepare_btc_data


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
CHART_DIR = OUTPUT_DIR / "charts"
FROZEN_DIR = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
DEFAULT_DATA = REPO_ROOT / "data" / "downloaded" / "MBT_v_0_1m_20210425_20261003.parquet"
START_DATE = date(2021, 5, 3)
END_DATE = date(2026, 10, 2)
IS_FRACTION = 0.70


@dataclass(frozen=True)
class GridDefinition:
    x_name: str
    x_values: tuple[Any, ...]
    y_name: str
    y_values: tuple[Any, ...]
    apply: Callable[[dict[str, Any], Any, Any], None]
    nominal_x: Any
    nominal_y: Any


def _set(structural: dict[str, Any], **updates: Any) -> None:
    structural.update(updates)


def _symmetric_rsi(structural: dict[str, Any], long_value: float) -> None:
    structural["long_entry_rsi"] = float(long_value)
    structural["short_entry_rsi"] = float(100.0 - long_value)


def _ribbon(structural: dict[str, Any], fast_scale: float, slow_scale: float) -> None:
    raw = [8 * fast_scale, 13 * fast_scale, 21, 34 * slow_scale, 55 * slow_scale]
    periods: list[int] = []
    for value in raw:
        periods.append(max(periods[-1] + 1 if periods else 2, int(round(value))))
    structural["ema_periods"] = periods


def grid_definitions() -> dict[str, GridDefinition]:
    """Return compact, predeclared local grids; every grid includes nominal (middle) cell."""
    return {
        "atr_trend_flip": GridDefinition("atr_period", (7, 10, 14), "atr_multiplier", (2.0, 3.0, 4.0), lambda s, x, y: _set(s, atr_period=int(x), atr_multiplier=float(y)), 10, 3.0),
        "channel_extreme_reversion": GridDefinition("ema_atr_period", (14, 20, 30), "keltner_atr_multiplier", (1.5, 2.0, 2.5), lambda s, x, y: _set(s, ema_period=int(x), atr_period=int(x), keltner_atr_multiplier=float(y)), 20, 2.0),
        "compression_breakout": GridDefinition("nr_lookback", (5, 7, 10), "entry_buffer_ticks", (0, 1, 2), lambda s, x, y: _set(s, nr_lookback=int(x), entry_buffer_ticks=int(y)), 7, 1),
        "gap_continuation": GridDefinition("minimum_gap_atr_multiple", (0.75, 1.0, 1.25), "volume_threshold_multiple", (0.75, 1.0, 1.25), lambda s, x, y: _set(s, minimum_gap_atr_multiple=float(x), volume_threshold_multiple=float(y)), 1.0, 1.0),
        "gap_fade": GridDefinition("daily_atr_period", (10, 14, 20), "maximum_gap_atr_multiple", (0.75, 1.0, 1.25), lambda s, x, y: _set(s, daily_atr_period=int(x), maximum_gap_atr_multiple=float(y)), 14, 1.0),
        "ma_crossover_trend": GridDefinition("fast_ema_period", (6, 9, 12), "slow_ema_period", (16, 21, 30), lambda s, x, y: _set(s, fast_ema_period=int(x), slow_ema_period=int(y)), 9, 21),
        "momentum_confirmed_breakout": GridDefinition("breakout_lookback", (14, 20, 30), "adx_threshold", (20.0, 25.0, 30.0), lambda s, x, y: _set(s, breakout_lookback=int(x), adx_threshold=float(y)), 20, 25.0),
        "range_structure_reversion": GridDefinition("range_lookback", (14, 20, 30), "entry_quartile_fraction", (0.20, 0.25, 0.30), lambda s, x, y: _set(s, range_lookback=int(x), entry_quartile_fraction=float(y)), 20, 0.25),
        "regression_equilibrium_reversion": GridDefinition("regression_lookback", (14, 20, 30), "entry_residual_sigma", (1.5, 2.0, 2.5), lambda s, x, y: _set(s, regression_lookback=int(x), entry_residual_sigma=float(y)), 20, 2.0),
        "rolling_range_breakout": GridDefinition("entry_channel_length", (14, 20, 30), "exit_channel_length", (7, 10, 15), lambda s, x, y: _set(s, entry_channel_length=int(x), exit_channel_length=int(y)), 20, 10),
        "session_range_breakout": GridDefinition("opening_range_minutes", (15, 30, 45), "breakout_buffer_ticks", (0, 1, 2), lambda s, x, y: _set(s, opening_range_minutes=int(x), breakout_buffer_ticks=int(y)), 30, 1),
        "short_horizon_pullback_reversion": GridDefinition("rsi_period", (2, 3, 4), "long_entry_rsi", (5.0, 10.0, 15.0), lambda s, x, y: (_set(s, rsi_period=int(x)), _symmetric_rsi(s, float(y))), 2, 10.0),
        "statistical_zscore_reversion": GridDefinition("mean_lookback", (14, 20, 30), "entry_z_abs", (1.5, 2.0, 2.5), lambda s, x, y: _set(s, mean_lookback=int(x), entry_z_abs=float(y)), 20, 2.0),
        "trend_pullback_ma_structure": GridDefinition("fast_pair_scale", (0.75, 1.0, 1.25), "slow_tail_scale", (0.75, 1.0, 1.25), _ribbon, 1.0, 1.0),
        "volatility_breakout": GridDefinition("daily_atr_period", (10, 14, 20), "entry_atr_multiple", (0.4, 0.6, 0.8), lambda s, x, y: _set(s, daily_atr_period=int(x), entry_atr_multiple=float(y)), 14, 0.6),
        "vwap_equilibrium_reversion": GridDefinition("minimum_completed_bars", (2, 4, 6), "entry_band_sigma", (1.5, 2.0, 2.5), lambda s, x, y: _set(s, minimum_completed_bars=int(x), entry_band_sigma=float(y)), 2, 2.0),
    }


def mbt_config() -> dict[str, Any]:
    return {
        "instrument": {
            "symbol": "MBT",
            "quantity": 1,
            "tick_size": 5.0,
            "tick_value_usd": 0.5,
            "point_value_usd": 0.1,
            "commission_per_side_usd": 1.41,
            "slippage_ticks_per_side": 1,
        },
        "session": {
            "timezone": "America/New_York",
            "rth_start": "09:30:00",
            "cash_close": "16:00:00",
            "last_entry_time": "15:55:00",
            "forced_flatten_time": "16:08:00",
            "calendar": "observed_full_US_window_only",
        },
        "relative_time_adaptations": {
            "gap_continuation_last_entry": "09:45:00",
            "gap_fade_last_entry": "09:31:00",
            "gap_fade_midday_time_exit": "12:00:00",
            "session_range_breakout_opening_range_start": "09:30:00",
        },
    }


def _fmt_axis(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _split_metrics(run: Any, split_start: date, split_end: date) -> dict[str, Any]:
    daily = run.daily.copy()
    mask = daily["session_date"].between(split_start, split_end)
    daily = daily.loc[mask]
    trades = run.trades.copy()
    if trades.empty:
        trade_count = 0
        win_rate = np.nan
        profit_factor = np.nan
    else:
        trades = trades.loc[trades["session_date"].between(split_start, split_end)]
        trade_count = int(len(trades))
        win_rate = float((trades["net_pnl_usd"] > 0).mean() * 100.0) if trade_count else np.nan
        gross_profit = float(trades.loc[trades["net_pnl_usd"] > 0, "net_pnl_usd"].sum())
        gross_loss = abs(float(trades.loc[trades["net_pnl_usd"] < 0, "net_pnl_usd"].sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
    pnl = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0)
    std = float(pnl.std(ddof=1))
    sharpe = float(pnl.mean() / std * math.sqrt(252.0)) if std > 0 and np.isfinite(std) else np.nan
    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax()
    return {
        "sessions": int(len(daily)),
        "trades": trade_count,
        "net_pnl_usd": float(pnl.sum()),
        "daily_pnl_sharpe": sharpe,
        "max_drawdown_usd": abs(float(drawdown.min())) if len(drawdown) else np.nan,
        "profit_factor": profit_factor,
        "win_rate_pct": win_rate,
    }


def _plot_pair(mechanism_id: str, strategy_id: str, grid: GridDefinition, rows: pd.DataFrame, output_path: Path) -> None:
    is_matrix = rows.pivot(index="y_index", columns="x_index", values="is_daily_pnl_sharpe").reindex(index=range(3), columns=range(3)).to_numpy(dtype=float)
    oos_matrix = rows.pivot(index="y_index", columns="x_index", values="oos_daily_pnl_sharpe").reindex(index=range(3), columns=range(3)).to_numpy(dtype=float)
    finite = np.concatenate([is_matrix[np.isfinite(is_matrix)], oos_matrix[np.isfinite(oos_matrix)]])
    bound = max(0.5, float(np.max(np.abs(finite)))) if finite.size else 1.0
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    image = None
    for axis, scope, matrix in zip(axes, ("IS", "OOS"), (is_matrix, oos_matrix)):
        image = axis.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto", origin="lower")
        axis.set_title(f"{scope} — Sharpe PnL quotidien", fontsize=12, weight="bold")
        axis.set_xlabel(grid.x_name)
        axis.set_ylabel(grid.y_name)
        axis.set_xticks(range(3), [_fmt_axis(v) for v in grid.x_values])
        axis.set_yticks(range(3), [_fmt_axis(v) for v in grid.y_values])
        for y_index in range(3):
            for x_index in range(3):
                cell = rows[(rows["x_index"] == x_index) & (rows["y_index"] == y_index)].iloc[0]
                sharpe = cell[f"{scope.lower()}_daily_pnl_sharpe"]
                pnl = cell[f"{scope.lower()}_net_pnl_usd"]
                trades = int(cell[f"{scope.lower()}_trades"])
                label = f"S {sharpe:.2f}\n${pnl:,.0f}\n{trades} tr" if np.isfinite(sharpe) else f"S n/a\n${pnl:,.0f}\n{trades} tr"
                axis.text(x_index, y_index, label, ha="center", va="center", fontsize=8, color="black")
        nx = list(grid.x_values).index(grid.nominal_x)
        ny = list(grid.y_values).index(grid.nominal_y)
        axis.add_patch(Rectangle((nx - 0.48, ny - 0.48), 0.96, 0.96, fill=False, edgecolor="#2457d6", linewidth=3.0))
    fig.colorbar(image, ax=axes, shrink=0.84, label="Sharpe annualisé du PnL quotidien")
    fig.suptitle(f"MBT · {strategy_id}\n{mechanism_id} · grille locale 3×3 · cellule nominale encadrée", fontsize=14, weight="bold")
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _build_overview(chart_paths: list[Path], output_path: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(24, 16), constrained_layout=True)
    for axis, chart_path in zip(axes.flat, chart_paths):
        axis.imshow(plt.imread(chart_path))
        axis.axis("off")
    fig.suptitle("MBT — 16 cartes de robustesse IS/OOS", fontsize=24, weight="bold")
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render the small summary table without pandas' optional tabulate dependency."""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        values: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("n/a" if not np.isfinite(value) else f"{float(value):.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_readme(summary: pd.DataFrame, metadata: dict[str, Any]) -> None:
    is_start, is_end = metadata["is_range"]
    oos_start, oos_end = metadata["oos_range"]
    lines = [
        "# MBT — 16 heatmaps de robustesse IS/OOS",
        "",
        f"- Données MBT: `{metadata['date_range'][0]}` → `{metadata['date_range'][1]}`; {metadata['session_count']} sessions complètes.",
        f"- IS: `{is_start}` → `{is_end}` ({metadata['is_sessions']} sessions).",
        f"- OOS: `{oos_start}` → `{oos_end}` ({metadata['oos_sessions']} sessions).",
        "- Coûts: 1 contrat MBT, 0,10 USD/point, tick 0,50 USD, commission 2,82 USD round-turn, slippage 1 tick/côté.",
        "- Chaque image juxtapose IS et OOS sur la même échelle de couleur. La cellule nominale gelée est encadrée en bleu.",
        "- Couleur: Sharpe annualisé du PnL quotidien; annotation: Sharpe, PnL net USD, nombre de trades.",
        "- Les grilles sont des voisinages locaux 3×3 déclarés avant lecture des résultats; aucune cellule n'est promue sur l'OOS.",
        "",
        "## Cellules nominales",
        "",
        _markdown_table(summary),
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(data_path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    source_specs, _, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    grids = grid_definitions()
    missing = {spec["mechanism_id"] for spec in source_specs} - set(grids)
    if missing:
        raise RuntimeError(f"Missing grid definitions: {sorted(missing)}")

    prepared, roll_sessions = prepare_btc_data(data_path, START_DATE, END_DATE)
    sessions = prepared.sessions["session_date"].tolist()
    split_index = max(1, min(len(sessions) - 1, int(len(sessions) * IS_FRACTION)))
    is_start, is_end = sessions[0], sessions[split_index - 1]
    oos_start, oos_end = sessions[split_index], sessions[-1]
    config = mbt_config()
    result_rows: list[dict[str, Any]] = []
    chart_paths: list[Path] = []

    for strategy_number, source_spec in enumerate(source_specs, start=1):
        mechanism_id = source_spec["mechanism_id"]
        strategy_id = source_spec["strategy_id"]
        grid = grids[mechanism_id]
        print(f"[{strategy_number:02d}/16] {mechanism_id}", flush=True)
        mechanism_rows: list[dict[str, Any]] = []
        for y_index, y_value in enumerate(grid.y_values):
            for x_index, x_value in enumerate(grid.x_values):
                spec = adapt_spec_to_btc(source_spec, config)
                spec = copy.deepcopy(spec)
                grid.apply(spec["canonical_parameters"]["structural"], x_value, y_value)
                run = run_baseline_backtest(spec, prepared)
                is_metrics = _split_metrics(run, is_start, is_end)
                oos_metrics = _split_metrics(run, oos_start, oos_end)
                row: dict[str, Any] = {
                    "mechanism_id": mechanism_id,
                    "strategy_id": strategy_id,
                    "x_parameter": grid.x_name,
                    "x_value": x_value,
                    "x_index": x_index,
                    "y_parameter": grid.y_name,
                    "y_value": y_value,
                    "y_index": y_index,
                    "is_nominal_cell": bool(x_value == grid.nominal_x and y_value == grid.nominal_y),
                }
                row.update({f"is_{key}": value for key, value in is_metrics.items()})
                row.update({f"oos_{key}": value for key, value in oos_metrics.items()})
                mechanism_rows.append(row)
                result_rows.append(row)
        mechanism_frame = pd.DataFrame(mechanism_rows)
        chart_path = CHART_DIR / f"{strategy_number:02d}_{mechanism_id}_is_oos.png"
        _plot_pair(mechanism_id, strategy_id, grid, mechanism_frame, chart_path)
        chart_paths.append(chart_path)
        pd.DataFrame(result_rows).to_csv(OUTPUT_DIR / "mbt_heatmap_metrics.csv", index=False)

    _build_overview(chart_paths, OUTPUT_DIR / "MBT_16_HEATMAPS_IS_OOS.png")
    result_frame = pd.DataFrame(result_rows)
    nominal = result_frame[result_frame["is_nominal_cell"]].copy()
    summary = nominal[[
        "mechanism_id", "strategy_id", "is_trades", "is_net_pnl_usd", "is_daily_pnl_sharpe",
        "oos_trades", "oos_net_pnl_usd", "oos_daily_pnl_sharpe", "oos_max_drawdown_usd",
    ]].sort_values("oos_daily_pnl_sharpe", ascending=False)
    summary.to_csv(OUTPUT_DIR / "mbt_nominal_is_oos_summary.csv", index=False)
    metadata = {
        "run_type": "mbt_frozen_strategy_local_robustness_heatmaps",
        "data_path": str(data_path.relative_to(REPO_ROOT)),
        "data_sha256": prepared.source_sha256,
        "date_range": [str(START_DATE), str(END_DATE)],
        "session_count": len(sessions),
        "is_fraction": IS_FRACTION,
        "is_range": [str(is_start), str(is_end)],
        "is_sessions": split_index,
        "oos_range": [str(oos_start), str(oos_end)],
        "oos_sessions": len(sessions) - split_index,
        "instrument": config["instrument"],
        "grid_shape_per_strategy": [3, 3],
        "strategy_count": 16,
        "total_backtests": 16 * 9,
        "intraday_roll_sessions_excluded": [str(day) for day in roll_sessions],
        "heatmap_metric": "annualized daily-PnL Sharpe",
        "selection_policy": "descriptive local robustness only; no OOS-based promotion",
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(summary, metadata)
    print(f"IS: {is_start} -> {is_end} ({split_index} sessions)")
    print(f"OOS: {oos_start} -> {oos_end} ({len(sessions) - split_index} sessions)")
    print(f"Charts: {len(chart_paths)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--finalize-only", action="store_true", help="Rebuild README from existing CSV/metadata without rerunning backtests.")
    args = parser.parse_args()
    if args.finalize_only:
        summary = pd.read_csv(OUTPUT_DIR / "mbt_nominal_is_oos_summary.csv")
        metadata = json.loads((OUTPUT_DIR / "run_metadata.json").read_text(encoding="utf-8"))
        _write_readme(summary, metadata)
        print("README rebuilt from existing outputs")
        return
    execute(args.data.resolve())


if __name__ == "__main__":
    main()
