"""Local robustness validation around the ATR-filtered ORB candidate.

This campaign is intentionally parsimonious and anti-lookahead:
- Baseline ORB signal logic is frozen.
- For each ATR(period, q_low, q_high) combination, ATR quantile thresholds are
  calibrated on IS only, then applied unchanged on OOS.
- Results are reported separately for IS and OOS.
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analytics.metrics import compute_metrics
from src.config.orb_campaign import DEFAULT_CAMPAIGN_DATASET
from src.config.paths import EXPORTS_DIR, ensure_directories
from src.config.settings import DEFAULT_INITIAL_CAPITAL_USD
from src.data.cleaning import clean_ohlcv
from src.data.loader import load_ohlcv_file
from src.engine.backtester import run_backtest
from src.engine.execution_model import ExecutionModel
from src.features.intraday import add_continuous_session_vwap, add_intraday_features, add_session_vwap
from src.features.opening_range import compute_opening_range
from src.features.volatility import add_atr
from src.strategy.orb import ORBStrategy


@dataclass(frozen=True)
class BaselineSpec:
    or_minutes: int = 15
    opening_time: str = "09:30:00"
    direction: str = "long"
    one_trade_per_day: bool = True
    entry_buffer_ticks: int = 2
    stop_buffer_ticks: int = 2
    target_multiple: float = 2.0
    vwap_confirmation: bool = True
    vwap_column: str = "continuous_session_vwap"
    time_exit: str = "16:00:00"
    account_size_usd: float = DEFAULT_INITIAL_CAPITAL_USD
    risk_per_trade_pct: float = 0.5
    tick_size: float = 0.25
    entry_on_next_open: bool = True


@dataclass(frozen=True)
class CampaignGrid:
    atr_periods: tuple[int, ...] = (10, 12, 14, 16, 18, 21, 30, 60)
    q_lows: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30)
    q_highs: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95)


@dataclass(frozen=True)
class CampaignSpec:
    dataset_path: Path = DEFAULT_CAMPAIGN_DATASET
    is_fraction: float = 0.70
    grid: CampaignGrid = CampaignGrid()
    candidate_period: int = 14
    candidate_q_low: float = 0.20
    candidate_q_high: float = 0.90


def _safe_rel(candidate: float, baseline: float, eps: float = 1e-9) -> float:
    if not math.isfinite(candidate) or not math.isfinite(baseline):
        return 0.0
    denom = max(abs(baseline), eps)
    return (candidate - baseline) / denom


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return default
    out = numerator / denominator
    return float(out) if math.isfinite(out) else default


def _quantile(series: pd.Series, q: float) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    value = float(clean.quantile(q))
    if not math.isfinite(value):
        return None
    return value


def _make_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "root": output_root,
        "data": output_root / "data",
        "tables": output_root / "tables",
        "heatmaps": output_root / "heatmaps",
        "summary": output_root / "summary",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    for split in ("is", "oos", "stability", "aggregated"):
        (dirs["heatmaps"] / split).mkdir(parents=True, exist_ok=True)
    return dirs


def _build_baseline_strategy(baseline: BaselineSpec) -> ORBStrategy:
    return ORBStrategy(
        or_minutes=baseline.or_minutes,
        direction=baseline.direction,
        one_trade_per_day=baseline.one_trade_per_day,
        entry_buffer_ticks=baseline.entry_buffer_ticks,
        stop_buffer_ticks=baseline.stop_buffer_ticks,
        target_multiple=baseline.target_multiple,
        opening_time=baseline.opening_time,
        time_exit=baseline.time_exit,
        account_size_usd=baseline.account_size_usd,
        risk_per_trade_pct=baseline.risk_per_trade_pct,
        tick_size=baseline.tick_size,
        vwap_confirmation=baseline.vwap_confirmation,
        vwap_column=baseline.vwap_column,
    )


def _prepare_feature_dataset(dataset_path: Path, baseline: BaselineSpec, atr_periods: tuple[int, ...]) -> pd.DataFrame:
    raw = load_ohlcv_file(dataset_path)
    raw = clean_ohlcv(raw)
    feat = add_intraday_features(raw)
    feat = add_session_vwap(feat)
    feat = add_continuous_session_vwap(feat, session_start_hour=18)
    feat = compute_opening_range(feat, or_minutes=baseline.or_minutes, opening_time=baseline.opening_time)
    for period in sorted(set(atr_periods)):
        feat = add_atr(feat, window=period)
    return feat


def _selected_signal_rows(signal_df: pd.DataFrame, atr_periods: tuple[int, ...]) -> pd.DataFrame:
    atr_cols = [f"atr_{period}" for period in atr_periods]
    keep_cols = ["session_date", "timestamp", "signal"] + [c for c in atr_cols if c in signal_df.columns]
    selected = signal_df.loc[signal_df["signal"].ne(0), keep_cols].copy()
    selected = selected.sort_values("timestamp")
    selected = selected.drop_duplicates(subset=["session_date"], keep="first")
    selected = selected.rename(columns={"timestamp": "signal_time"})
    return selected


def _run_backtest(signal_df: pd.DataFrame, baseline: BaselineSpec, execution_model: ExecutionModel) -> pd.DataFrame:
    return run_backtest(
        signal_df,
        execution_model=execution_model,
        time_exit=baseline.time_exit,
        stop_buffer_ticks=baseline.stop_buffer_ticks,
        target_multiple=baseline.target_multiple,
        account_size_usd=baseline.account_size_usd,
        risk_per_trade_pct=baseline.risk_per_trade_pct,
        entry_on_next_open=baseline.entry_on_next_open,
    )


def _compute_metrics_for_sessions(
    trades: pd.DataFrame,
    sessions: list,
    initial_capital: float,
) -> dict[str, float | int | bool | str]:
    subset = trades.loc[trades["session_date"].isin(set(sessions))].copy() if not trades.empty else trades.copy()
    return compute_metrics(subset, session_dates=sessions, initial_capital=initial_capital)


def _split_sessions(all_sessions: list, is_fraction: float) -> tuple[list, list]:
    n = len(all_sessions)
    if n < 2:
        raise ValueError("Not enough sessions to perform IS/OOS split.")
    split_idx = int(n * is_fraction)
    split_idx = max(1, min(n - 1, split_idx))
    return all_sessions[:split_idx], all_sessions[split_idx:]


def _prefix_metrics(metrics: dict[str, object], prefix: str) -> dict[str, object]:
    keys = [
        "n_trades",
        "win_rate",
        "expectancy",
        "profit_factor",
        "sharpe_ratio",
        "cumulative_pnl",
        "max_drawdown",
        "stop_hit_rate",
        "target_hit_rate",
        "time_exit_rate",
        "time_exit_win_rate",
    ]
    return {f"{prefix}_{key}": metrics.get(key, np.nan) for key in keys}


def _annotated_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    output_path: Path,
    cmap: str = "RdYlGn",
    fmt: str = ".2f",
    center: float | None = None,
) -> None:
    if df.empty:
        return
    pivot = (
        df.pivot_table(index="q_low_pct", columns="q_high_pct", values=value_col, aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )
    if pivot.empty:
        return

    rows = list(pivot.index)
    cols = list(pivot.columns)
    values = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    if center is not None and np.isfinite(values).any():
        vmax = float(np.nanmax(np.abs(values)))
        vmin = -vmax
        im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(values, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels([f"q{int(c)}" for c in cols])
    ax.set_yticklabels([f"q{int(r)}" for r in rows])
    ax.set_xlabel("ATR high quantile")
    ax.set_ylabel("ATR low quantile")
    ax.set_title(title)

    for i in range(len(rows)):
        for j in range(len(cols)):
            value = values[i, j]
            text = "nan" if not math.isfinite(value) else format(value, fmt)
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _combine_top_rows(df: pd.DataFrame, prefix: str, top_n: int = 10, min_trades: int = 40) -> pd.DataFrame:
    frame = df.loc[df[f"{prefix}_n_trades"] >= min_trades].copy()
    if frame.empty:
        frame = df.copy()
    sort_cols = [f"{prefix}_sharpe_ratio", f"{prefix}_profit_factor", f"{prefix}_expectancy", f"{prefix}_n_trades"]
    return frame.sort_values(sort_cols, ascending=[False, False, False, False]).head(top_n).reset_index(drop=True)

def _build_local_metrics(df: pd.DataFrame, baseline_oos_row: pd.Series) -> pd.DataFrame:
    out = df.copy()
    baseline_sharpe = float(baseline_oos_row["oos_sharpe_ratio"])
    baseline_pf = float(baseline_oos_row["oos_profit_factor"])
    baseline_exp = float(baseline_oos_row["oos_expectancy"])
    baseline_trades = float(baseline_oos_row["oos_n_trades"])

    out["oos_rel_sharpe_vs_baseline"] = out["oos_sharpe_ratio"].apply(lambda x: _safe_rel(float(x), baseline_sharpe))
    out["oos_rel_pf_vs_baseline"] = out["oos_profit_factor"].apply(lambda x: _safe_rel(float(x), baseline_pf))
    out["oos_rel_expectancy_vs_baseline"] = out["oos_expectancy"].apply(lambda x: _safe_rel(float(x), baseline_exp))

    out["delta_sharpe_oos_minus_is"] = out["oos_sharpe_ratio"] - out["is_sharpe_ratio"]
    out["delta_pf_oos_minus_is"] = out["oos_profit_factor"] - out["is_profit_factor"]
    out["delta_expectancy_oos_minus_is"] = out["oos_expectancy"] - out["is_expectancy"]

    out["ratio_oos_is_sharpe"] = out.apply(
        lambda r: _safe_div(float(r["oos_sharpe_ratio"]), float(r["is_sharpe_ratio"]), default=0.0),
        axis=1,
    )
    out["ratio_oos_is_pf"] = out.apply(
        lambda r: _safe_div(float(r["oos_profit_factor"]), float(r["is_profit_factor"]), default=0.0),
        axis=1,
    )
    out["ratio_oos_is_expectancy"] = out.apply(
        lambda r: _safe_div(float(r["oos_expectancy"]), float(r["is_expectancy"]), default=0.0),
        axis=1,
    )

    out["oos_good_metric_count"] = (
        (out["oos_sharpe_ratio"] >= baseline_sharpe).astype(int)
        + (out["oos_profit_factor"] >= baseline_pf).astype(int)
        + (out["oos_expectancy"] >= baseline_exp).astype(int)
    )
    trade_floor = max(40.0, 0.45 * baseline_trades)
    out["trade_floor_ok"] = out["oos_n_trades"] >= trade_floor
    out["robust_oos_flag"] = out["trade_floor_ok"] & (out["oos_good_metric_count"] >= 2)

    drift_sh = (out["delta_sharpe_oos_minus_is"].abs()) / out["is_sharpe_ratio"].abs().clip(lower=0.25)
    drift_pf = (out["delta_pf_oos_minus_is"].abs()) / out["is_profit_factor"].abs().clip(lower=0.25)
    drift_exp = (out["delta_expectancy_oos_minus_is"].abs()) / out["is_expectancy"].abs().clip(lower=1.0)
    drift_penalty = ((drift_sh + drift_pf + drift_exp) / 3.0).clip(upper=2.0)

    oos_edge = (
        out["oos_rel_sharpe_vs_baseline"]
        + out["oos_rel_pf_vs_baseline"]
        + out["oos_rel_expectancy_vs_baseline"]
    ) / 3.0
    participation = (out["oos_n_trades"] / max(baseline_trades, 1.0)).clip(lower=0.0, upper=1.0)
    stability = 1.0 - drift_penalty

    out["local_robustness_score"] = 0.55 * oos_edge + 0.25 * stability + 0.20 * participation
    return out


def _neighbors_table(
    df: pd.DataFrame,
    periods: list[int],
    lows: list[int],
    highs: list[int],
    candidate: tuple[int, int, int],
) -> pd.DataFrame:
    p, ql, qh = candidate
    p_idx = {value: i for i, value in enumerate(periods)}
    l_idx = {value: i for i, value in enumerate(lows)}
    h_idx = {value: i for i, value in enumerate(highs)}

    if p not in p_idx or ql not in l_idx or qh not in h_idx:
        return pd.DataFrame()

    neighbors = []
    candidate_row = df.loc[
        (df["atr_period"] == p) & (df["q_low_pct"] == ql) & (df["q_high_pct"] == qh)
    ].copy()
    if not candidate_row.empty:
        row = candidate_row.iloc[0].to_dict()
        row["neighbor_type"] = "candidate"
        row["manhattan_steps"] = 0
        neighbors.append(row)

    for dp, dl, dh in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
        ip = p_idx[p] + dp
        il = l_idx[ql] + dl
        ih = h_idx[qh] + dh
        if ip < 0 or ip >= len(periods) or il < 0 or il >= len(lows) or ih < 0 or ih >= len(highs):
            continue
        np_period = periods[ip]
        np_low = lows[il]
        np_high = highs[ih]
        match = df.loc[
            (df["atr_period"] == np_period)
            & (df["q_low_pct"] == np_low)
            & (df["q_high_pct"] == np_high)
        ]
        if match.empty:
            continue
        row = match.iloc[0].to_dict()
        row["neighbor_type"] = "neighbor"
        row["manhattan_steps"] = 1
        neighbors.append(row)

    if not neighbors:
        return pd.DataFrame()
    return pd.DataFrame(neighbors)


def _robust_clusters(df: pd.DataFrame, periods: list[int], lows: list[int], highs: list[int]) -> pd.DataFrame:
    robust = df.loc[df["robust_oos_flag"]].copy()
    if robust.empty:
        return pd.DataFrame()

    p_idx = {value: i for i, value in enumerate(periods)}
    l_idx = {value: i for i, value in enumerate(lows)}
    h_idx = {value: i for i, value in enumerate(highs)}

    keys = {
        (int(r["atr_period"]), int(r["q_low_pct"]), int(r["q_high_pct"])): i
        for i, r in robust.reset_index(drop=True).iterrows()
    }
    robust = robust.reset_index(drop=True)

    visited = set()
    rows: list[dict[str, object]] = []
    cluster_id = 0

    for key in keys:
        if key in visited:
            continue
        cluster_id += 1
        queue: deque[tuple[int, int, int]] = deque([key])
        component: list[tuple[int, int, int]] = []
        visited.add(key)

        while queue:
            cur = queue.popleft()
            component.append(cur)
            cp, cl, ch = cur
            for dp, dl, dh in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                ip = p_idx[cp] + dp
                il = l_idx[cl] + dl
                ih = h_idx[ch] + dh
                if ip < 0 or ip >= len(periods) or il < 0 or il >= len(lows) or ih < 0 or ih >= len(highs):
                    continue
                nxt = (periods[ip], lows[il], highs[ih])
                if nxt not in keys or nxt in visited:
                    continue
                visited.add(nxt)
                queue.append(nxt)

        comp_idx = [keys[item] for item in component]
        comp_df = robust.loc[comp_idx]
        rows.append(
            {
                "cluster_id": cluster_id,
                "n_cells": int(len(comp_df)),
                "period_min": int(comp_df["atr_period"].min()),
                "period_max": int(comp_df["atr_period"].max()),
                "q_low_min": int(comp_df["q_low_pct"].min()),
                "q_low_max": int(comp_df["q_low_pct"].max()),
                "q_high_min": int(comp_df["q_high_pct"].min()),
                "q_high_max": int(comp_df["q_high_pct"].max()),
                "avg_oos_sharpe": float(comp_df["oos_sharpe_ratio"].mean()),
                "avg_oos_pf": float(comp_df["oos_profit_factor"].mean()),
                "avg_oos_expectancy": float(comp_df["oos_expectancy"].mean()),
                "avg_oos_n_trades": float(comp_df["oos_n_trades"].mean()),
                "avg_local_robustness_score": float(comp_df["local_robustness_score"].mean()),
                "cells": ";".join([f"({a},{b},{c})" for a, b, c in sorted(component)]),
            }
        )

    return pd.DataFrame(rows).sort_values(["n_cells", "avg_local_robustness_score"], ascending=[False, False])


def run_atr_local_robustness(spec: CampaignSpec, output_dir: Path) -> dict[str, Path]:
    ensure_directories()
    dirs = _make_dirs(output_dir)

    baseline = BaselineSpec()
    feature_df = _prepare_feature_dataset(spec.dataset_path, baseline, spec.grid.atr_periods)
    all_sessions = sorted(pd.to_datetime(feature_df["session_date"]).dt.date.unique())
    is_sessions, oos_sessions = _split_sessions(all_sessions, spec.is_fraction)

    execution_model = ExecutionModel()
    strategy = _build_baseline_strategy(baseline)
    baseline_signal_df = strategy.generate_signals(feature_df)
    baseline_trades = _run_backtest(baseline_signal_df, baseline, execution_model)
    signal_rows = _selected_signal_rows(baseline_signal_df, spec.grid.atr_periods)

    baseline_is = _compute_metrics_for_sessions(baseline_trades, is_sessions, baseline.account_size_usd)
    baseline_oos = _compute_metrics_for_sessions(baseline_trades, oos_sessions, baseline.account_size_usd)

    baseline_split_table = pd.DataFrame(
        [
            {"split": "IS", **baseline_is},
            {"split": "OOS", **baseline_oos},
        ]
    )
    baseline_split_table.to_csv(dirs["tables"] / "baseline_is_oos_metrics.csv", index=False)

    rows: list[dict[str, object]] = []
    is_signal_rows = signal_rows.loc[signal_rows["session_date"].isin(set(is_sessions))].copy()

    for period in spec.grid.atr_periods:
        atr_col = f"atr_{period}"
        if atr_col not in signal_rows.columns:
            continue

        for q_low in spec.grid.q_lows:
            for q_high in spec.grid.q_highs:
                if q_low >= q_high:
                    continue

                low_thr = _quantile(is_signal_rows[atr_col], q_low)
                high_thr = _quantile(is_signal_rows[atr_col], q_high)
                if low_thr is None or high_thr is None or low_thr >= high_thr:
                    continue

                keep_mask_all = signal_rows[atr_col].between(low_thr, high_thr, inclusive="both")
                selected_signal_rows = signal_rows.loc[keep_mask_all]
                selected_sessions_all = set(selected_signal_rows["session_date"])
                selected_sessions_is = sorted(set(is_sessions).intersection(selected_sessions_all))
                selected_sessions_oos = sorted(set(oos_sessions).intersection(selected_sessions_all))

                is_metrics = _compute_metrics_for_sessions(
                    baseline_trades,
                    sessions=selected_sessions_is,
                    initial_capital=baseline.account_size_usd,
                )
                oos_metrics = _compute_metrics_for_sessions(
                    baseline_trades,
                    sessions=selected_sessions_oos,
                    initial_capital=baseline.account_size_usd,
                )

                row: dict[str, object] = {
                    "atr_period": int(period),
                    "q_low": float(q_low),
                    "q_high": float(q_high),
                    "q_low_pct": int(round(q_low * 100)),
                    "q_high_pct": int(round(q_high * 100)),
                    "atr_low_threshold_is": float(low_thr),
                    "atr_high_threshold_is": float(high_thr),
                    "n_signal_rows_total": int(len(signal_rows)),
                    "n_signal_rows_is": int(len(is_signal_rows)),
                    "n_selected_signals_total": int(len(selected_signal_rows)),
                    "n_selected_signals_is": int(len(selected_signal_rows.loc[selected_signal_rows["session_date"].isin(set(is_sessions))])),
                    "n_selected_signals_oos": int(len(selected_signal_rows.loc[selected_signal_rows["session_date"].isin(set(oos_sessions))])),
                }
                row.update(_prefix_metrics(is_metrics, "is"))
                row.update(_prefix_metrics(oos_metrics, "oos"))
                rows.append(row)

    combos = pd.DataFrame(rows)
    if combos.empty:
        raise RuntimeError("No valid ATR local combinations were generated.")

    baseline_row = pd.Series(
        {
            "oos_sharpe_ratio": baseline_oos.get("sharpe_ratio", 0.0),
            "oos_profit_factor": baseline_oos.get("profit_factor", 0.0),
            "oos_expectancy": baseline_oos.get("expectancy", 0.0),
            "oos_n_trades": baseline_oos.get("n_trades", 0.0),
        }
    )
    combos = _build_local_metrics(combos, baseline_row)
    combos = combos.sort_values(
        ["local_robustness_score", "oos_sharpe_ratio", "oos_profit_factor", "oos_expectancy"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    combos.to_csv(dirs["data"] / "atr_local_grid_all_combinations.csv", index=False)

    metrics_is = combos[
        [
            "atr_period",
            "q_low",
            "q_high",
            "q_low_pct",
            "q_high_pct",
            "atr_low_threshold_is",
            "atr_high_threshold_is",
            "is_n_trades",
            "is_win_rate",
            "is_expectancy",
            "is_profit_factor",
            "is_sharpe_ratio",
            "is_cumulative_pnl",
            "is_max_drawdown",
            "is_stop_hit_rate",
            "is_target_hit_rate",
            "is_time_exit_rate",
            "is_time_exit_win_rate",
        ]
    ].copy()
    metrics_oos = combos[
        [
            "atr_period",
            "q_low",
            "q_high",
            "q_low_pct",
            "q_high_pct",
            "atr_low_threshold_is",
            "atr_high_threshold_is",
            "oos_n_trades",
            "oos_win_rate",
            "oos_expectancy",
            "oos_profit_factor",
            "oos_sharpe_ratio",
            "oos_cumulative_pnl",
            "oos_max_drawdown",
            "oos_stop_hit_rate",
            "oos_target_hit_rate",
            "oos_time_exit_rate",
            "oos_time_exit_win_rate",
            "ratio_oos_is_sharpe",
            "ratio_oos_is_pf",
            "ratio_oos_is_expectancy",
            "delta_sharpe_oos_minus_is",
            "delta_pf_oos_minus_is",
            "delta_expectancy_oos_minus_is",
            "robust_oos_flag",
            "local_robustness_score",
        ]
    ].copy()

    metrics_is.to_csv(dirs["tables"] / "metrics_is.csv", index=False)
    metrics_oos.to_csv(dirs["tables"] / "metrics_oos.csv", index=False)

    top_is = _combine_top_rows(combos, prefix="is", top_n=12, min_trades=40)
    top_oos = _combine_top_rows(combos, prefix="oos", top_n=12, min_trades=40)
    top_robust = combos.sort_values(
        ["local_robustness_score", "oos_sharpe_ratio", "oos_profit_factor"],
        ascending=[False, False, False],
    ).head(20)

    top_is.to_csv(dirs["tables"] / "top_is.csv", index=False)
    top_oos.to_csv(dirs["tables"] / "top_oos.csv", index=False)
    top_robust.to_csv(dirs["tables"] / "top_local_robustness.csv", index=False)

    periods = sorted(combos["atr_period"].unique().tolist())
    q_lows_pct = sorted(combos["q_low_pct"].unique().tolist())
    q_highs_pct = sorted(combos["q_high_pct"].unique().tolist())

    for period in periods:
        frame = combos.loc[combos["atr_period"] == period].copy()
        _annotated_heatmap(
            frame,
            "oos_sharpe_ratio",
            f"OOS Sharpe | ATR({period})",
            dirs["heatmaps"] / "oos" / f"atr_{period}_oos_sharpe.png",
            cmap="RdYlGn",
            fmt=".2f",
        )
        _annotated_heatmap(
            frame,
            "oos_profit_factor",
            f"OOS Profit Factor | ATR({period})",
            dirs["heatmaps"] / "oos" / f"atr_{period}_oos_profit_factor.png",
            cmap="RdYlGn",
            fmt=".2f",
        )
        _annotated_heatmap(
            frame,
            "oos_expectancy",
            f"OOS Expectancy (USD) | ATR({period})",
            dirs["heatmaps"] / "oos" / f"atr_{period}_oos_expectancy.png",
            cmap="RdYlGn",
            fmt=".1f",
        )
        _annotated_heatmap(
            frame,
            "oos_n_trades",
            f"OOS Number of Trades | ATR({period})",
            dirs["heatmaps"] / "oos" / f"atr_{period}_oos_n_trades.png",
            cmap="Blues",
            fmt=".0f",
        )

        _annotated_heatmap(
            frame,
            "is_sharpe_ratio",
            f"IS Sharpe | ATR({period})",
            dirs["heatmaps"] / "is" / f"atr_{period}_is_sharpe.png",
            cmap="RdYlGn",
            fmt=".2f",
        )
        _annotated_heatmap(
            frame,
            "is_profit_factor",
            f"IS Profit Factor | ATR({period})",
            dirs["heatmaps"] / "is" / f"atr_{period}_is_profit_factor.png",
            cmap="RdYlGn",
            fmt=".2f",
        )

        _annotated_heatmap(
            frame,
            "delta_sharpe_oos_minus_is",
            f"Stability Delta Sharpe (OOS-IS) | ATR({period})",
            dirs["heatmaps"] / "stability" / f"atr_{period}_delta_sharpe.png",
            cmap="coolwarm",
            fmt=".2f",
            center=0.0,
        )
        _annotated_heatmap(
            frame,
            "delta_pf_oos_minus_is",
            f"Stability Delta PF (OOS-IS) | ATR({period})",
            dirs["heatmaps"] / "stability" / f"atr_{period}_delta_pf.png",
            cmap="coolwarm",
            fmt=".2f",
            center=0.0,
        )
        _annotated_heatmap(
            frame,
            "local_robustness_score",
            f"Local Robustness Score | ATR({period})",
            dirs["heatmaps"] / "stability" / f"atr_{period}_local_robustness_score.png",
            cmap="RdYlGn",
            fmt=".2f",
        )

    aggregated = (
        combos.groupby(["q_low_pct", "q_high_pct"], as_index=False)
        .agg(
            oos_sharpe_ratio=("oos_sharpe_ratio", "median"),
            oos_profit_factor=("oos_profit_factor", "median"),
            oos_expectancy=("oos_expectancy", "median"),
            oos_n_trades=("oos_n_trades", "median"),
            is_sharpe_ratio=("is_sharpe_ratio", "median"),
            is_profit_factor=("is_profit_factor", "median"),
            is_expectancy=("is_expectancy", "median"),
            delta_sharpe_oos_minus_is=("delta_sharpe_oos_minus_is", "median"),
            delta_pf_oos_minus_is=("delta_pf_oos_minus_is", "median"),
            delta_expectancy_oos_minus_is=("delta_expectancy_oos_minus_is", "median"),
            local_robustness_score=("local_robustness_score", "median"),
        )
        .sort_values(["q_low_pct", "q_high_pct"])
    )
    aggregated.to_csv(dirs["tables"] / "aggregated_by_quantiles_across_periods.csv", index=False)

    _annotated_heatmap(
        aggregated,
        "oos_sharpe_ratio",
        "Median OOS Sharpe Across ATR Periods",
        dirs["heatmaps"] / "aggregated" / "aggregated_oos_sharpe.png",
        cmap="RdYlGn",
        fmt=".2f",
    )
    _annotated_heatmap(
        aggregated,
        "oos_profit_factor",
        "Median OOS PF Across ATR Periods",
        dirs["heatmaps"] / "aggregated" / "aggregated_oos_profit_factor.png",
        cmap="RdYlGn",
        fmt=".2f",
    )
    _annotated_heatmap(
        aggregated,
        "oos_expectancy",
        "Median OOS Expectancy Across ATR Periods",
        dirs["heatmaps"] / "aggregated" / "aggregated_oos_expectancy.png",
        cmap="RdYlGn",
        fmt=".1f",
    )
    _annotated_heatmap(
        aggregated,
        "oos_n_trades",
        "Median OOS Trades Across ATR Periods",
        dirs["heatmaps"] / "aggregated" / "aggregated_oos_n_trades.png",
        cmap="Blues",
        fmt=".0f",
    )
    _annotated_heatmap(
        aggregated,
        "delta_sharpe_oos_minus_is",
        "Median Delta Sharpe (OOS-IS) Across ATR Periods",
        dirs["heatmaps"] / "aggregated" / "aggregated_delta_sharpe.png",
        cmap="coolwarm",
        fmt=".2f",
        center=0.0,
    )
    _annotated_heatmap(
        aggregated,
        "local_robustness_score",
        "Median Local Robustness Score Across ATR Periods",
        dirs["heatmaps"] / "aggregated" / "aggregated_local_robustness_score.png",
        cmap="RdYlGn",
        fmt=".2f",
    )

    cand = (
        int(spec.candidate_period),
        int(round(spec.candidate_q_low * 100)),
        int(round(spec.candidate_q_high * 100)),
    )
    neighbors = _neighbors_table(combos, periods, q_lows_pct, q_highs_pct, cand)
    neighbors.to_csv(dirs["tables"] / "candidate_neighbors.csv", index=False)

    clusters = _robust_clusters(combos, periods, q_lows_pct, q_highs_pct)
    clusters.to_csv(dirs["tables"] / "robust_clusters.csv", index=False)

    candidate_row_df = combos.loc[
        (combos["atr_period"] == cand[0])
        & (combos["q_low_pct"] == cand[1])
        & (combos["q_high_pct"] == cand[2])
    ]
    candidate_row = candidate_row_df.iloc[0] if not candidate_row_df.empty else None

    candidate_robust = bool(candidate_row["robust_oos_flag"]) if candidate_row is not None else False
    neighbor_count = int(len(neighbors.loc[neighbors["neighbor_type"] == "neighbor"])) if not neighbors.empty else 0
    robust_neighbors = (
        int(neighbors.loc[(neighbors["neighbor_type"] == "neighbor") & (neighbors["robust_oos_flag"])].shape[0])
        if not neighbors.empty
        else 0
    )
    neighbor_ratio = _safe_div(float(robust_neighbors), float(neighbor_count), default=0.0)

    candidate_cluster_size = 0
    candidate_cluster_id = None
    if not clusters.empty:
        cand_token = f"({cand[0]},{cand[1]},{cand[2]})"
        match = clusters.loc[clusters["cells"].str.contains(cand_token, regex=False)]
        if not match.empty:
            candidate_cluster_id = int(match.iloc[0]["cluster_id"])
            candidate_cluster_size = int(match.iloc[0]["n_cells"])

    if candidate_robust and candidate_cluster_size >= 6 and neighbor_ratio >= 0.50:
        verdict = "1. Le candidat ATR est localement robuste"
        recommendation = "Conserver le candidat actuel (14/q20/q90) ou un voisin proche dans le meme plateau."
    elif (candidate_robust and (candidate_cluster_size >= 3 or neighbor_ratio >= 0.33)) or (
        (not candidate_robust) and robust_neighbors >= 2
    ):
        verdict = "2. Le candidat ATR est prometteur mais la zone est etroite / fragile"
        recommendation = "Modifier legerement vers la zone voisine la plus stable plutot que garder un point exact."
    else:
        verdict = "3. Le candidat ATR semble surtout etre un optimum instable"
        recommendation = "Rejeter la parametrisation exacte et revenir a la baseline (ou reprendre une recherche plus large mais disciplinee)."

    local_zone_rows = combos.loc[
        combos["atr_period"].between(max(min(periods), cand[0] - 2), min(max(periods), cand[0] + 2))
        & combos["q_low_pct"].between(max(min(q_lows_pct), cand[1] - 5), min(max(q_lows_pct), cand[1] + 5))
        & combos["q_high_pct"].between(max(min(q_highs_pct), cand[2] - 5), min(max(q_highs_pct), cand[2] + 5))
    ].copy()
    local_zone_rows.to_csv(dirs["tables"] / "local_zone_around_candidate.csv", index=False)

    best_oos = top_oos.head(8)
    best_is = top_is.head(8)

    report_lines = [
        "# ATR Local Robustness Report",
        "",
        "## Candidat de depart",
        "",
        "- Strategie candidate: ORB baseline + filtre ATR borne.",
        f"- Point de reference: ATR({cand[0]}), q_low=q{cand[1]}, q_high=q{cand[2]}.",
        "- Objectif: verifier si la performance vient d'un plateau local stable ou d'un point fragile.",
        "",
        "## Protocole IS/OOS (anti-lookahead)",
        "",
        f"- Dataset: `{spec.dataset_path.name}`",
        f"- Sessions totales: {len(all_sessions)} ({all_sessions[0]} -> {all_sessions[-1]})",
        f"- Split chronologique: IS={len(is_sessions)} sessions, OOS={len(oos_sessions)} sessions",
        f"- Date de coupure: OOS commence le {oos_sessions[0]}",
        "- Calibration ATR: quantiles estimes uniquement sur IS, puis figes pour OOS.",
        "- Aucune recalibration sur OOS.",
        "",
        "## Grille testee",
        "",
        f"- ATR period: {list(spec.grid.atr_periods)}",
        f"- q_low: {[int(q*100) for q in spec.grid.q_lows]}",
        f"- q_high: {[int(q*100) for q in spec.grid.q_highs]}",
        f"- Combinaisons valides testees: {len(combos)}",
        "",
        "## Performance absolue",
        "",
        f"- Baseline IS: Sharpe={baseline_is.get('sharpe_ratio', 0.0):.3f}, PF={baseline_is.get('profit_factor', 0.0):.3f}, Expectancy={baseline_is.get('expectancy', 0.0):.2f}, Trades={int(baseline_is.get('n_trades', 0))}",
        f"- Baseline OOS: Sharpe={baseline_oos.get('sharpe_ratio', 0.0):.3f}, PF={baseline_oos.get('profit_factor', 0.0):.3f}, Expectancy={baseline_oos.get('expectancy', 0.0):.2f}, Trades={int(baseline_oos.get('n_trades', 0))}",
        "",
        "Top OOS (apres filtre trade floor):",
        "",
        "```text",
        best_oos[["atr_period", "q_low_pct", "q_high_pct", "oos_n_trades", "oos_sharpe_ratio", "oos_profit_factor", "oos_expectancy", "local_robustness_score", "robust_oos_flag"]].to_string(index=False),
        "```",
        "",
        "Top IS (apres filtre trade floor):",
        "",
        "```text",
        best_is[["atr_period", "q_low_pct", "q_high_pct", "is_n_trades", "is_sharpe_ratio", "is_profit_factor", "is_expectancy"]].to_string(index=False),
        "```",
        "",
        "## Robustesse locale autour de (14, q20, q90)",
        "",
    ]

    if candidate_row is not None:
        report_lines.extend(
            [
                f"- Candidat: OOS Sharpe={float(candidate_row['oos_sharpe_ratio']):.3f}, PF={float(candidate_row['oos_profit_factor']):.3f}, Expectancy={float(candidate_row['oos_expectancy']):.2f}, Trades={int(candidate_row['oos_n_trades'])}",
                f"- Candidat robuste OOS (regle 2/3 metriques + trade floor): {bool(candidate_row['robust_oos_flag'])}",
                f"- Neighbors robustes immediats: {robust_neighbors}/{neighbor_count} ({neighbor_ratio:.1%})",
                f"- Taille du cluster robuste contenant le candidat: {candidate_cluster_size}",
            ]
        )
    else:
        report_lines.extend(
            [
                "- Le point candidat exact n'existe pas dans la grille calculee.",
            ]
        )

    if candidate_cluster_id is not None and not clusters.empty:
        cluster_row = clusters.loc[clusters["cluster_id"] == candidate_cluster_id].iloc[0]
        report_lines.extend(
            [
                f"- Zone cluster candidat: ATR {int(cluster_row['period_min'])}-{int(cluster_row['period_max'])}, q_low q{int(cluster_row['q_low_min'])}-q{int(cluster_row['q_low_max'])}, q_high q{int(cluster_row['q_high_min'])}-q{int(cluster_row['q_high_max'])}",
                f"- Moyennes cluster (OOS): Sharpe={float(cluster_row['avg_oos_sharpe']):.3f}, PF={float(cluster_row['avg_oos_pf']):.3f}, Expectancy={float(cluster_row['avg_oos_expectancy']):.2f}",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Stabilite OOS",
            "",
            "- Analyse stabilite via deltas OOS-IS et score local_robustness_score.",
            "- Heatmaps de stabilite disponibles dans `heatmaps/stability/` et `heatmaps/aggregated/`.",
            "",
            "## Verdict",
            "",
            f"- {verdict}",
            f"- Recommandation: {recommendation}",
            "",
            "## Artefacts",
            "",
            f"- Table complete: `{(dirs['data'] / 'atr_local_grid_all_combinations.csv').as_posix()}`",
            f"- Metrics IS: `{(dirs['tables'] / 'metrics_is.csv').as_posix()}`",
            f"- Metrics OOS: `{(dirs['tables'] / 'metrics_oos.csv').as_posix()}`",
            f"- Clusters robustes: `{(dirs['tables'] / 'robust_clusters.csv').as_posix()}`",
            f"- Voisinage candidat: `{(dirs['tables'] / 'candidate_neighbors.csv').as_posix()}`",
            f"- Heatmaps: `{dirs['heatmaps'].as_posix()}`",
        ]
    )

    summary_path = dirs["summary"] / "atr_local_robustness_report.md"
    summary_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")

    metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "dataset": str(spec.dataset_path),
        "baseline": asdict(baseline),
        "grid": {
            "atr_periods": list(spec.grid.atr_periods),
            "q_lows": list(spec.grid.q_lows),
            "q_highs": list(spec.grid.q_highs),
        },
        "split": {
            "is_fraction": spec.is_fraction,
            "n_sessions_total": len(all_sessions),
            "n_sessions_is": len(is_sessions),
            "n_sessions_oos": len(oos_sessions),
            "is_start": str(is_sessions[0]),
            "is_end": str(is_sessions[-1]),
            "oos_start": str(oos_sessions[0]),
            "oos_end": str(oos_sessions[-1]),
        },
        "candidate": {
            "period": cand[0],
            "q_low_pct": cand[1],
            "q_high_pct": cand[2],
            "candidate_cluster_size": candidate_cluster_size,
            "robust_neighbors": robust_neighbors,
            "neighbor_count": neighbor_count,
            "neighbor_ratio": neighbor_ratio,
            "candidate_robust": candidate_robust,
            "verdict": verdict,
            "recommendation": recommendation,
        },
    }
    pd.Series(metadata).to_json(dirs["summary"] / "run_metadata.json", indent=2)

    return {
        "output_root": dirs["root"],
        "all_combinations": dirs["data"] / "atr_local_grid_all_combinations.csv",
        "metrics_is": dirs["tables"] / "metrics_is.csv",
        "metrics_oos": dirs["tables"] / "metrics_oos.csv",
        "clusters": dirs["tables"] / "robust_clusters.csv",
        "report": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local robustness validation around ATR candidate.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CAMPAIGN_DATASET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--is-fraction", type=float, default=0.70)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (EXPORTS_DIR / f"atr_local_robustness_{timestamp}")

    spec = CampaignSpec(dataset_path=args.dataset, is_fraction=args.is_fraction)
    artifacts = run_atr_local_robustness(spec=spec, output_dir=output_dir)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

