"""ATR ensemble campaign around robust quantile neighborhoods.

This runner compares:
- Baseline ORB (no ATR filter)
- Point ATR model (reference configured by campaign inputs)
- Ensemble ATR filters built from neighboring quantile pairs

Calibration protocol is anti-lookahead:
- quantile thresholds are estimated on IS only
- thresholds are frozen and applied unchanged on OOS
"""

from __future__ import annotations

import argparse
import json
import math
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
from src.engine.portfolio import build_equity_curve
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
class EnsembleZone:
    name: str
    q_lows_pct: tuple[int, ...]
    q_highs_pct: tuple[int, ...]


@dataclass(frozen=True)
class CampaignSpec:
    dataset_path: Path = DEFAULT_CAMPAIGN_DATASET
    is_fraction: float = 0.70
    atr_period: int = 14
    point_q_low_pct: float = 20.0
    point_q_high_pct: float = 90.0
    include_expanded_zone: bool = True


def _make_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "root": output_root,
        "baseline": output_root / "baseline",
        "point_model": output_root / "point_model",
        "ensemble_models": output_root / "ensemble_models",
        "diagnostics": output_root / "diagnostics",
        "summary": output_root / "summary",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    (dirs["diagnostics"] / "figures").mkdir(parents=True, exist_ok=True)
    return dirs


def _split_sessions(all_sessions: list, is_fraction: float) -> tuple[list, list]:
    if len(all_sessions) < 2:
        raise ValueError("Not enough sessions for IS/OOS split.")
    split_idx = int(len(all_sessions) * is_fraction)
    split_idx = max(1, min(len(all_sessions) - 1, split_idx))
    return all_sessions[:split_idx], all_sessions[split_idx:]


def _safe_rel(candidate: float, baseline: float, eps: float = 1e-9) -> float:
    if not math.isfinite(candidate) or not math.isfinite(baseline):
        return 0.0
    return (candidate - baseline) / max(abs(baseline), eps)


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


def _build_baseline_strategy(spec: BaselineSpec) -> ORBStrategy:
    return ORBStrategy(
        or_minutes=spec.or_minutes,
        direction=spec.direction,
        one_trade_per_day=spec.one_trade_per_day,
        entry_buffer_ticks=spec.entry_buffer_ticks,
        stop_buffer_ticks=spec.stop_buffer_ticks,
        target_multiple=spec.target_multiple,
        opening_time=spec.opening_time,
        time_exit=spec.time_exit,
        account_size_usd=spec.account_size_usd,
        risk_per_trade_pct=spec.risk_per_trade_pct,
        tick_size=spec.tick_size,
        vwap_confirmation=spec.vwap_confirmation,
        vwap_column=spec.vwap_column,
    )


def _prepare_feature_dataset(dataset_path: Path, baseline: BaselineSpec, atr_period: int) -> pd.DataFrame:
    raw = load_ohlcv_file(dataset_path)
    raw = clean_ohlcv(raw)
    feat = add_intraday_features(raw)
    feat = add_session_vwap(feat)
    feat = add_continuous_session_vwap(feat, session_start_hour=18)
    feat = compute_opening_range(feat, or_minutes=baseline.or_minutes, opening_time=baseline.opening_time)
    feat = add_atr(feat, window=atr_period)
    return feat


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


def _selected_signal_rows(signal_df: pd.DataFrame, atr_col: str) -> pd.DataFrame:
    selected = signal_df.loc[signal_df["signal"].ne(0)].copy()
    selected = selected.sort_values("timestamp")
    selected = selected.drop_duplicates(subset=["session_date"], keep="first")
    selected = selected.rename(columns={"timestamp": "signal_time"})
    selected["signal_index"] = selected.index
    keep_cols = ["signal_index", "session_date", "signal_time", "signal", atr_col]
    keep_cols = [c for c in keep_cols if c in selected.columns]
    return selected[keep_cols].copy()


def _metrics_for_sessions(trades: pd.DataFrame, sessions: list, initial_capital: float) -> dict[str, object]:
    subset = trades.loc[trades["session_date"].isin(set(sessions))].copy() if not trades.empty else trades.copy()
    return compute_metrics(subset, session_dates=sessions, initial_capital=initial_capital)


def _prefix_metrics(metrics: dict[str, object], prefix: str) -> dict[str, object]:
    keys = [
        "n_trades",
        "win_rate",
        "expectancy",
        "profit_factor",
        "sharpe_ratio",
        "cumulative_pnl",
        "max_drawdown",
        "percent_of_days_traded",
        "time_exit_rate",
        "time_exit_win_rate",
    ]
    return {f"{prefix}_{key}": metrics.get(key, np.nan) for key in keys}


def _pct_tag(value: float) -> str:
    value_f = float(value)
    if value_f.is_integer():
        return str(int(value_f))
    return str(value_f).replace(".", "p")


def _build_zone_definitions(include_expanded: bool, q_low_anchor_pct: float) -> list[EnsembleZone]:
    low_a = int(round(float(q_low_anchor_pct)))
    low_b = min(low_a + 5, 95)
    low_c = min(low_a + 10, 95)
    highs = (90, 95)

    zones = [
        EnsembleZone(
            name=f"narrow_q{low_a}_{low_b}__q{highs[0]}_{highs[1]}",
            q_lows_pct=(low_a, low_b),
            q_highs_pct=highs,
        ),
    ]
    if include_expanded:
        zones.append(
            EnsembleZone(
                name=f"expanded_q{low_a}_{low_b}_{low_c}__q{highs[0]}_{highs[1]}",
                q_lows_pct=(low_a, low_b, low_c),
                q_highs_pct=highs,
            )
        )
    return zones


def _serialize_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _build_submodels(
    signal_rows: pd.DataFrame,
    is_sessions: list,
    atr_col: str,
    atr_period: int,
    zones: list[EnsembleZone],
    point_q_low_pct: float,
    point_q_high_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    signal_scores = signal_rows.copy()
    submodels_rows: list[dict[str, object]] = []

    is_mask = signal_scores["session_date"].isin(set(is_sessions))
    is_values = signal_scores.loc[is_mask, atr_col]

    for zone in zones:
        zone_model_cols: list[str] = []
        for q_low in zone.q_lows_pct:
            for q_high in zone.q_highs_pct:
                if q_low >= q_high:
                    continue

                low_thr = _quantile(is_values, q_low / 100.0)
                high_thr = _quantile(is_values, q_high / 100.0)
                if low_thr is None or high_thr is None or low_thr >= high_thr:
                    continue

                model_id = f"{zone.name}__atr{atr_period}_q{q_low}_q{q_high}"
                pass_col = f"pass__{model_id}"
                signal_scores[pass_col] = signal_scores[atr_col].between(low_thr, high_thr, inclusive="both")
                zone_model_cols.append(pass_col)

                submodels_rows.append(
                    {
                        "zone": zone.name,
                        "model_id": model_id,
                        "atr_period": atr_period,
                        "q_low_pct": q_low,
                        "q_high_pct": q_high,
                        "low_threshold_is": float(low_thr),
                        "high_threshold_is": float(high_thr),
                        "pass_column": pass_col,
                    }
                )

        if zone_model_cols:
            signal_scores[f"score__{zone.name}"] = signal_scores[zone_model_cols].mean(axis=1)

    submodels_df = pd.DataFrame(submodels_rows)
    if submodels_df.empty:
        raise RuntimeError("No valid submodels were created for ensemble zones.")

    # Point reference model (user-provided reference).
    point_low_thr = _quantile(is_values, float(point_q_low_pct) / 100.0)
    point_high_thr = _quantile(is_values, float(point_q_high_pct) / 100.0)
    if point_low_thr is None or point_high_thr is None or point_low_thr >= point_high_thr:
        raise RuntimeError("Unable to calibrate point model thresholds on IS.")
    point_model_id = f"point_q{_pct_tag(point_q_low_pct)}_q{_pct_tag(point_q_high_pct)}"
    point_pass_col = f"pass__{point_model_id}"
    signal_scores[point_pass_col] = signal_scores[atr_col].between(point_low_thr, point_high_thr, inclusive="both")

    # Secondary point: +5 low quantile with same high quantile (if valid).
    point_secondary_q_low = float(point_q_low_pct) + 5.0
    point_secondary_q_high = float(point_q_high_pct)
    secondary_model_id = f"point_q{_pct_tag(point_secondary_q_low)}_q{_pct_tag(point_secondary_q_high)}"
    secondary_pass_col = f"pass__{secondary_model_id}"

    sec_low_thr = _quantile(is_values, point_secondary_q_low / 100.0)
    sec_high_thr = _quantile(is_values, point_secondary_q_high / 100.0)
    if sec_low_thr is not None and sec_high_thr is not None and sec_low_thr < sec_high_thr:
        signal_scores[secondary_pass_col] = signal_scores[atr_col].between(
            sec_low_thr,
            sec_high_thr,
            inclusive="both",
        )
    else:
        signal_scores[secondary_pass_col] = False

    submodels_df = pd.concat(
        [
            submodels_df,
            pd.DataFrame(
                [
                    {
                        "zone": "point_model",
                        "model_id": point_model_id,
                        "atr_period": atr_period,
                        "q_low_pct": float(point_q_low_pct),
                        "q_high_pct": float(point_q_high_pct),
                        "low_threshold_is": float(point_low_thr),
                        "high_threshold_is": float(point_high_thr),
                        "pass_column": point_pass_col,
                    },
                    {
                        "zone": "point_model_secondary",
                        "model_id": secondary_model_id,
                        "atr_period": atr_period,
                        "q_low_pct": point_secondary_q_low,
                        "q_high_pct": point_secondary_q_high,
                        "low_threshold_is": float(sec_low_thr) if sec_low_thr is not None else np.nan,
                        "high_threshold_is": float(sec_high_thr) if sec_high_thr is not None else np.nan,
                        "pass_column": secondary_pass_col,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    return submodels_df, signal_scores, point_model_id, secondary_model_id


def _build_variant_configs(
    submodels_df: pd.DataFrame,
    zones: list[EnsembleZone],
    point_model_id: str,
    point_secondary_model_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    rows.append(
        {
            "strategy_id": "baseline_no_atr",
            "strategy_group": "baseline",
            "zone": "none",
            "score_column": "",
            "threshold": np.nan,
            "aggregation": "none",
            "n_submodels": 0,
            "selection_column": "",
        }
    )
    rows.append(
        {
            "strategy_id": point_model_id,
            "strategy_group": "point_model",
            "zone": "point_model",
            "score_column": "",
            "threshold": np.nan,
            "aggregation": "single_point",
            "n_submodels": 1,
            "selection_column": f"pass__{point_model_id}",
        }
    )
    rows.append(
        {
            "strategy_id": point_secondary_model_id,
            "strategy_group": "point_model",
            "zone": "point_model_secondary",
            "score_column": "",
            "threshold": np.nan,
            "aggregation": "single_point_avg",
            "n_submodels": 1,
            "selection_column": f"pass__{point_secondary_model_id}",
        }
    )

    for zone in zones:
        zone_models = submodels_df.loc[submodels_df["zone"] == zone.name].copy()
        n_models = int(len(zone_models))
        if n_models == 0:
            continue
        score_col = f"score__{zone.name}"

        rows.extend(
            [
                {
                    "strategy_id": f"ensemble__{zone.name}__majority_50",
                    "strategy_group": "ensemble",
                    "zone": zone.name,
                    "score_column": score_col,
                    "threshold": 0.50,
                    "aggregation": "majority_50",
                    "n_submodels": n_models,
                    "selection_column": "",
                },
                {
                    "strategy_id": f"ensemble__{zone.name}__consensus_75",
                    "strategy_group": "ensemble",
                    "zone": zone.name,
                    "score_column": score_col,
                    "threshold": 0.75,
                    "aggregation": "consensus_75",
                    "n_submodels": n_models,
                    "selection_column": "",
                },
                {
                    "strategy_id": f"ensemble__{zone.name}__consensus_100",
                    "strategy_group": "ensemble",
                    "zone": zone.name,
                    "score_column": score_col,
                    "threshold": 1.00,
                    "aggregation": "consensus_100",
                    "n_submodels": n_models,
                    "selection_column": "",
                },
            ]
        )

        # One additional score-threshold test for the expanded zone only.
        if n_models >= 6:
            rows.append(
                {
                    "strategy_id": f"ensemble__{zone.name}__score_ge_0p67",
                    "strategy_group": "ensemble",
                    "zone": zone.name,
                    "score_column": score_col,
                    "threshold": 2.0 / 3.0,
                    "aggregation": "score_ge_0p67",
                    "n_submodels": n_models,
                    "selection_column": "",
                }
            )

    return pd.DataFrame(rows)


def _select_signal_indices(
    variant: pd.Series,
    signal_scores: pd.DataFrame,
) -> pd.Index:
    strategy_group = str(variant["strategy_group"])
    if strategy_group == "baseline":
        return pd.Index(signal_scores["signal_index"])

    selection_col = str(variant["selection_column"])
    if selection_col:
        mask = signal_scores[selection_col].fillna(False).astype(bool)
        return pd.Index(signal_scores.loc[mask, "signal_index"])

    score_col = str(variant["score_column"])
    threshold = float(variant["threshold"])
    if score_col not in signal_scores.columns:
        return pd.Index([])

    # Use strict ratio thresholds with exact pass proportion (e.g., 75% means >= 0.75).
    mask = pd.to_numeric(signal_scores[score_col], errors="coerce").fillna(0.0) >= threshold
    return pd.Index(signal_scores.loc[mask, "signal_index"])


def _signal_df_for_selected_indices(base_signal_df: pd.DataFrame, selected_indices: pd.Index) -> pd.DataFrame:
    out = base_signal_df.copy()
    signal_mask = out["signal"].ne(0)
    keep_index = set(pd.Index(selected_indices).tolist())
    drop_mask = signal_mask & (~out.index.isin(keep_index))
    out.loc[drop_mask, "signal"] = 0
    return out


def _plot_equity_drawdown(
    trades_map: dict[str, pd.DataFrame],
    initial_capital: float,
    out_equity: Path,
    out_drawdown: Path,
) -> None:
    curves = {name: build_equity_curve(trades, initial_capital=initial_capital) for name, trades in trades_map.items()}

    fig1, ax1 = plt.subplots(figsize=(11, 5.5))
    for name, curve in curves.items():
        if curve.empty:
            continue
        ax1.plot(curve["timestamp"], curve["equity"], label=name, linewidth=1.4)
    ax1.set_title("Equity Comparison")
    ax1.set_ylabel("Equity (USD)")
    ax1.legend(loc="best")
    fig1.tight_layout()
    fig1.savefig(out_equity, dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(11, 5.5))
    for name, curve in curves.items():
        if curve.empty:
            continue
        ax2.plot(curve["timestamp"], curve["drawdown"], label=name, linewidth=1.4)
    ax2.set_title("Drawdown Comparison")
    ax2.set_ylabel("Drawdown (USD)")
    ax2.legend(loc="best")
    fig2.tight_layout()
    fig2.savefig(out_drawdown, dpi=150)
    plt.close(fig2)

    for name, curve in curves.items():
        curve.to_csv(out_equity.parent / f"equity_curve__{name}.csv", index=False)


def _plot_main_metrics_bar(compare_df: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("oos_sharpe_ratio", "OOS Sharpe"),
        ("oos_profit_factor", "OOS PF"),
        ("oos_expectancy", "OOS Expectancy"),
        ("oos_cumulative_pnl", "OOS Cum PnL"),
        ("oos_n_trades", "OOS Trades"),
        ("oos_max_drawdown_abs", "OOS |Max DD|"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes_flat = axes.flatten()
    labels = compare_df["strategy_id"].tolist()

    for ax, (col, title) in zip(axes_flat, metrics):
        values = compare_df[col].astype(float).tolist()
        ax.bar(labels, values)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _consensus_distribution(
    signal_scores: pd.DataFrame,
    zones: list[EnsembleZone],
    is_sessions: list,
    oos_sessions: list,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    split_masks = {
        "total": pd.Series(True, index=signal_scores.index),
        "is": signal_scores["session_date"].isin(set(is_sessions)),
        "oos": signal_scores["session_date"].isin(set(oos_sessions)),
    }

    for zone in zones:
        score_col = f"score__{zone.name}"
        if score_col not in signal_scores.columns:
            continue
        for split_name, split_mask in split_masks.items():
            frame = signal_scores.loc[split_mask, ["session_date", score_col]].copy()
            if frame.empty:
                continue
            frame["score_level"] = frame[score_col].round(4)
            grouped = frame.groupby("score_level", dropna=False).size().reset_index(name="n_candidate_days")
            for _, row in grouped.iterrows():
                rows.append(
                    {
                        "zone": zone.name,
                        "split": split_name,
                        "score_level": float(row["score_level"]),
                        "n_candidate_days": int(row["n_candidate_days"]),
                    }
                )
    return pd.DataFrame(rows)


def _consensus_performance_by_score(
    signal_scores: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    zones: list[EnsembleZone],
    is_sessions: list,
    oos_sessions: list,
    initial_capital: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    split_sessions = {
        "total": sorted(set(is_sessions).union(set(oos_sessions))),
        "is": is_sessions,
        "oos": oos_sessions,
    }

    for zone in zones:
        score_col = f"score__{zone.name}"
        if score_col not in signal_scores.columns:
            continue

        for split_name, sessions in split_sessions.items():
            frame = signal_scores.loc[signal_scores["session_date"].isin(set(sessions)), ["session_date", score_col]].copy()
            if frame.empty:
                continue
            frame["score_level"] = frame[score_col].round(4)

            for score_level, bucket in frame.groupby("score_level", sort=True):
                bucket_days = sorted(bucket["session_date"].tolist())
                bucket_trades = baseline_trades.loc[baseline_trades["session_date"].isin(set(bucket_days))].copy()
                metrics = compute_metrics(bucket_trades, session_dates=bucket_days, initial_capital=initial_capital)
                rows.append(
                    {
                        "zone": zone.name,
                        "split": split_name,
                        "score_level": float(score_level),
                        "n_candidate_days": int(len(bucket_days)),
                        "n_trades": int(metrics.get("n_trades", 0)),
                        "win_rate": float(metrics.get("win_rate", 0.0)),
                        "expectancy": float(metrics.get("expectancy", 0.0)),
                        "profit_factor": float(metrics.get("profit_factor", 0.0)),
                        "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
                        "cumulative_pnl": float(metrics.get("cumulative_pnl", 0.0)),
                        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                    }
                )

    return pd.DataFrame(rows)


def _plot_consensus_diagnostics(
    distribution_df: pd.DataFrame,
    perf_by_score_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if not distribution_df.empty:
        oos_dist = distribution_df.loc[distribution_df["split"] == "oos"].copy()
        if not oos_dist.empty:
            zones = sorted(oos_dist["zone"].unique().tolist())
            fig, axes = plt.subplots(1, len(zones), figsize=(6 * len(zones), 4.2), squeeze=False)
            for ax, zone in zip(axes[0], zones):
                frame = oos_dist.loc[oos_dist["zone"] == zone].sort_values("score_level")
                ax.bar(frame["score_level"].astype(str), frame["n_candidate_days"])
                ax.set_title(f"OOS Consensus Score Distribution\n{zone}")
                ax.set_xlabel("Score")
                ax.set_ylabel("Candidate Days")
                ax.tick_params(axis="x", rotation=35)
            fig.tight_layout()
            fig.savefig(output_dir / "consensus_score_distribution_oos.png", dpi=150)
            plt.close(fig)

    if not perf_by_score_df.empty:
        oos_perf = perf_by_score_df.loc[perf_by_score_df["split"] == "oos"].copy()
        if not oos_perf.empty:
            zones = sorted(oos_perf["zone"].unique().tolist())
            fig, axes = plt.subplots(2, len(zones), figsize=(6 * len(zones), 7.5), squeeze=False)
            for i, zone in enumerate(zones):
                frame = oos_perf.loc[oos_perf["zone"] == zone].sort_values("score_level")

                axes[0, i].plot(frame["score_level"], frame["expectancy"], marker="o")
                axes[0, i].set_title(f"OOS Expectancy by Score\n{zone}")
                axes[0, i].set_xlabel("Score")
                axes[0, i].set_ylabel("Expectancy")

                axes[1, i].plot(frame["score_level"], frame["profit_factor"], marker="o", color="#b45309")
                axes[1, i].set_title(f"OOS PF by Score\n{zone}")
                axes[1, i].set_xlabel("Score")
                axes[1, i].set_ylabel("Profit Factor")

            fig.tight_layout()
            fig.savefig(output_dir / "consensus_expectancy_pf_by_score_oos.png", dpi=150)
            plt.close(fig)


def run_atr_ensemble_campaign(spec: CampaignSpec, output_dir: Path) -> dict[str, Path]:
    ensure_directories()
    dirs = _make_dirs(output_dir)

    baseline = BaselineSpec()
    zones = _build_zone_definitions(
        include_expanded=spec.include_expanded_zone,
        q_low_anchor_pct=spec.point_q_low_pct,
    )
    atr_col = f"atr_{spec.atr_period}"

    feat = _prepare_feature_dataset(spec.dataset_path, baseline, atr_period=spec.atr_period)
    all_sessions = sorted(pd.to_datetime(feat["session_date"]).dt.date.unique())
    is_sessions, oos_sessions = _split_sessions(all_sessions, spec.is_fraction)

    execution_model = ExecutionModel()
    strategy = _build_baseline_strategy(baseline)
    baseline_signal_df = strategy.generate_signals(feat)
    signal_rows = _selected_signal_rows(baseline_signal_df, atr_col=atr_col)
    baseline_trades = _run_backtest(baseline_signal_df, baseline, execution_model)

    submodels_df, signal_scores, point_model_id, point_secondary_model_id = _build_submodels(
        signal_rows=signal_rows,
        is_sessions=is_sessions,
        atr_col=atr_col,
        atr_period=spec.atr_period,
        zones=zones,
        point_q_low_pct=spec.point_q_low_pct,
        point_q_high_pct=spec.point_q_high_pct,
    )
    variant_configs = _build_variant_configs(
        submodels_df=submodels_df,
        zones=zones,
        point_model_id=point_model_id,
        point_secondary_model_id=point_secondary_model_id,
    )

    # Export building blocks.
    submodels_df.to_csv(dirs["ensemble_models"] / "submodels_definition.csv", index=False)
    signal_scores.to_csv(dirs["ensemble_models"] / "signal_scores_by_candidate_day.csv", index=False)
    variant_configs.to_csv(dirs["ensemble_models"] / "ensemble_variant_configs.csv", index=False)

    baseline_metrics_overall = compute_metrics(
        baseline_trades,
        signal_df=baseline_signal_df,
        session_dates=all_sessions,
        initial_capital=baseline.account_size_usd,
    )
    baseline_metrics_is = _metrics_for_sessions(baseline_trades, is_sessions, baseline.account_size_usd)
    baseline_metrics_oos = _metrics_for_sessions(baseline_trades, oos_sessions, baseline.account_size_usd)

    pd.DataFrame([baseline_metrics_overall]).to_csv(dirs["baseline"] / "baseline_metrics_overall.csv", index=False)
    pd.DataFrame([
        {"split": "is", **baseline_metrics_is},
        {"split": "oos", **baseline_metrics_oos},
    ]).to_csv(dirs["baseline"] / "baseline_metrics_is_oos.csv", index=False)
    baseline_trades.to_csv(dirs["baseline"] / "baseline_trades.csv", index=False)

    results_rows: list[dict[str, object]] = []
    trades_by_strategy: dict[str, pd.DataFrame] = {}

    for _, variant in variant_configs.iterrows():
        strategy_id = str(variant["strategy_id"])
        selected_indices = _select_signal_indices(variant=variant, signal_scores=signal_scores)
        variant_signal_df = _signal_df_for_selected_indices(baseline_signal_df, selected_indices)
        variant_trades = _run_backtest(variant_signal_df, baseline, execution_model)

        overall = compute_metrics(
            variant_trades,
            signal_df=variant_signal_df,
            session_dates=all_sessions,
            initial_capital=baseline.account_size_usd,
        )
        is_metrics = _metrics_for_sessions(variant_trades, is_sessions, baseline.account_size_usd)
        oos_metrics = _metrics_for_sessions(variant_trades, oos_sessions, baseline.account_size_usd)

        selected_signal_rows = signal_scores.loc[signal_scores["signal_index"].isin(set(selected_indices))]
        n_signals_total = int(len(selected_signal_rows))
        n_signals_is = int(len(selected_signal_rows.loc[selected_signal_rows["session_date"].isin(set(is_sessions))]))
        n_signals_oos = int(len(selected_signal_rows.loc[selected_signal_rows["session_date"].isin(set(oos_sessions))]))

        row = {
            **variant.to_dict(),
            "n_candidate_signal_days_total": int(len(signal_scores)),
            "n_selected_signal_days_total": n_signals_total,
            "n_selected_signal_days_is": n_signals_is,
            "n_selected_signal_days_oos": n_signals_oos,
            "selection_rate_total": _safe_div(n_signals_total, len(signal_scores), default=0.0),
        }
        row.update(_prefix_metrics(overall, "overall"))
        row.update(_prefix_metrics(is_metrics, "is"))
        row.update(_prefix_metrics(oos_metrics, "oos"))
        results_rows.append(row)

        trades_by_strategy[strategy_id] = variant_trades
        variant_trades.to_csv(dirs["ensemble_models"] / f"trades__{strategy_id}.csv", index=False)

    variants_df = pd.DataFrame(results_rows)
    if variants_df.empty:
        raise RuntimeError("No strategy variants were evaluated.")

    # Comparisons against baseline and point.
    baseline_row = variants_df.loc[variants_df["strategy_id"] == "baseline_no_atr"].iloc[0]
    point_row = variants_df.loc[variants_df["strategy_id"] == point_model_id].iloc[0]

    variants_df["oos_max_drawdown_abs"] = variants_df["oos_max_drawdown"].abs()
    variants_df["is_max_drawdown_abs"] = variants_df["is_max_drawdown"].abs()
    variants_df["delta_sharpe_oos_minus_is"] = variants_df["oos_sharpe_ratio"] - variants_df["is_sharpe_ratio"]
    variants_df["delta_pf_oos_minus_is"] = variants_df["oos_profit_factor"] - variants_df["is_profit_factor"]
    variants_df["delta_expectancy_oos_minus_is"] = variants_df["oos_expectancy"] - variants_df["is_expectancy"]

    for prefix, ref in (("baseline", baseline_row), ("point", point_row)):
        variants_df[f"oos_rel_sharpe_vs_{prefix}"] = variants_df["oos_sharpe_ratio"].apply(
            lambda x, b=float(ref["oos_sharpe_ratio"]): _safe_rel(float(x), b)
        )
        variants_df[f"oos_rel_pf_vs_{prefix}"] = variants_df["oos_profit_factor"].apply(
            lambda x, b=float(ref["oos_profit_factor"]): _safe_rel(float(x), b)
        )
        variants_df[f"oos_rel_expectancy_vs_{prefix}"] = variants_df["oos_expectancy"].apply(
            lambda x, b=float(ref["oos_expectancy"]): _safe_rel(float(x), b)
        )
        variants_df[f"oos_rel_drawdown_vs_{prefix}"] = variants_df["oos_max_drawdown_abs"].apply(
            lambda x, b=float(abs(ref["oos_max_drawdown"])): _safe_rel(b, float(x))
        )

    variants_df["stability_gap"] = (
        variants_df["delta_sharpe_oos_minus_is"].abs()
        + variants_df["delta_pf_oos_minus_is"].abs()
        + (variants_df["delta_expectancy_oos_minus_is"].abs() / variants_df["is_expectancy"].abs().clip(lower=1.0))
    )

    variants_df["ensemble_score_vs_point"] = (
        0.35 * variants_df["oos_rel_sharpe_vs_point"]
        + 0.30 * variants_df["oos_rel_pf_vs_point"]
        + 0.20 * variants_df["oos_rel_expectancy_vs_point"]
        + 0.15 * variants_df["oos_rel_drawdown_vs_point"]
    ) - 0.15 * variants_df["stability_gap"]

    variants_df = variants_df.sort_values(
        ["strategy_group", "ensemble_score_vs_point", "oos_sharpe_ratio", "oos_profit_factor"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    variants_df.to_csv(dirs["summary"] / "all_variants_metrics.csv", index=False)
    variants_df.to_csv(dirs["ensemble_models"] / "ensemble_variants_metrics.csv", index=False)

    point_export = variants_df.loc[variants_df["strategy_group"] == "point_model"].copy()
    point_export.to_csv(dirs["point_model"] / "point_models_metrics.csv", index=False)
    for strategy_id in point_export["strategy_id"].tolist():
        trades_by_strategy[strategy_id].to_csv(
            dirs["point_model"] / f"trades__{strategy_id}.csv",
            index=False,
        )

    # Refresh baseline/point rows after all derived columns are computed.
    baseline_row = variants_df.loc[variants_df["strategy_id"] == "baseline_no_atr"].iloc[0]
    point_row = variants_df.loc[variants_df["strategy_id"] == point_model_id].iloc[0]

    # Identify best ensemble with trade floor to avoid fragile low-activity picks.
    ensemble_rows = variants_df.loc[variants_df["strategy_group"] == "ensemble"].copy()
    point_oos_trades = float(point_row["oos_n_trades"])
    trade_floor = max(40.0, 0.60 * point_oos_trades)
    ensemble_valid = ensemble_rows.loc[ensemble_rows["oos_n_trades"] >= trade_floor].copy()
    if ensemble_valid.empty:
        ensemble_valid = ensemble_rows.copy()
    best_ensemble = ensemble_valid.sort_values(
        ["ensemble_score_vs_point", "oos_sharpe_ratio", "oos_profit_factor", "oos_expectancy"],
        ascending=[False, False, False, False],
    ).iloc[0]

    # Consensus diagnostics.
    dist_df = _consensus_distribution(signal_scores, zones, is_sessions, oos_sessions)
    perf_by_score_df = _consensus_performance_by_score(
        signal_scores=signal_scores,
        baseline_trades=baseline_trades,
        zones=zones,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        initial_capital=baseline.account_size_usd,
    )
    dist_df.to_csv(dirs["diagnostics"] / "consensus_score_distribution.csv", index=False)
    perf_by_score_df.to_csv(dirs["diagnostics"] / "consensus_performance_by_score.csv", index=False)
    _plot_consensus_diagnostics(dist_df, perf_by_score_df, dirs["diagnostics"] / "figures")

    consensus_summary_lines: list[str] = []
    oos_perf = perf_by_score_df.loc[(perf_by_score_df["split"] == "oos") & (perf_by_score_df["n_trades"] >= 8)].copy()
    for zone in sorted(oos_perf["zone"].unique().tolist()):
        frame = oos_perf.loc[oos_perf["zone"] == zone].sort_values("score_level")
        if frame.empty:
            continue
        best_exp = frame.sort_values("expectancy", ascending=False).iloc[0]
        best_pf = frame.sort_values("profit_factor", ascending=False).iloc[0]
        corr = frame["score_level"].rank(method="average").corr(
            frame["expectancy"].rank(method="average"),
            method="pearson",
        )
        corr_txt = f"{corr:.3f}" if math.isfinite(float(corr)) else "nan"
        consensus_summary_lines.append(
            f"- {zone}: best expectancy at score={float(best_exp['score_level']):.4f} "
            f"(exp={float(best_exp['expectancy']):.2f}, n_trades={int(best_exp['n_trades'])}); "
            f"best PF at score={float(best_pf['score_level']):.4f} "
            f"(PF={float(best_pf['profit_factor']):.3f}); "
            f"Spearman(score, expectancy)={corr_txt}."
        )

    # Visual comparisons: baseline vs point vs best ensemble.
    key_ids = ["baseline_no_atr", point_model_id, str(best_ensemble["strategy_id"])]
    compare_df = variants_df.loc[variants_df["strategy_id"].isin(key_ids)].copy()
    compare_df = compare_df.set_index("strategy_id").loc[key_ids].reset_index()

    compare_trades = {
        "baseline_no_atr": trades_by_strategy["baseline_no_atr"],
        point_model_id: trades_by_strategy[point_model_id],
        str(best_ensemble["strategy_id"]): trades_by_strategy[str(best_ensemble["strategy_id"])],
    }
    _plot_equity_drawdown(
        compare_trades,
        initial_capital=baseline.account_size_usd,
        out_equity=dirs["summary"] / "equity_comparison.png",
        out_drawdown=dirs["summary"] / "drawdown_comparison.png",
    )
    _plot_main_metrics_bar(compare_df, dirs["summary"] / "oos_metrics_bar_comparison.png")

    # Explicit final decision among 4 requested options.
    point = point_row
    ensemble = best_ensemble
    baseline_ref = baseline_row

    ensemble_vs_point_improve_count = sum(
        [
            float(ensemble["oos_sharpe_ratio"]) >= 1.03 * float(point["oos_sharpe_ratio"]),
            float(ensemble["oos_profit_factor"]) >= 1.02 * float(point["oos_profit_factor"]),
            float(ensemble["oos_expectancy"]) >= 1.02 * float(point["oos_expectancy"]),
            float(abs(ensemble["oos_max_drawdown"])) <= 0.97 * float(abs(point["oos_max_drawdown"])),
        ]
    )
    ensemble_vs_point_degrade = any(
        [
            float(ensemble["oos_sharpe_ratio"]) < 0.95 * float(point["oos_sharpe_ratio"]),
            float(ensemble["oos_profit_factor"]) < 0.95 * float(point["oos_profit_factor"]),
            float(ensemble["oos_expectancy"]) < 0.92 * float(point["oos_expectancy"]),
        ]
    )
    comparable_perf = all(
        [
            float(ensemble["oos_sharpe_ratio"]) >= 0.97 * float(point["oos_sharpe_ratio"]),
            float(ensemble["oos_profit_factor"]) >= 0.97 * float(point["oos_profit_factor"]),
            float(ensemble["oos_expectancy"]) >= 0.95 * float(point["oos_expectancy"]),
        ]
    )
    stability_better = float(ensemble["stability_gap"]) <= 0.95 * float(point["stability_gap"])

    ensemble_adds_value_vs_baseline = sum(
        [
            float(ensemble["oos_sharpe_ratio"]) > float(baseline_ref["oos_sharpe_ratio"]),
            float(ensemble["oos_profit_factor"]) > float(baseline_ref["oos_profit_factor"]),
            float(ensemble["oos_expectancy"]) > float(baseline_ref["oos_expectancy"]),
        ]
    ) >= 2

    if ensemble_vs_point_improve_count >= 3 and not ensemble_vs_point_degrade:
        final_option = "3. Le signal d'ensemble est clairement superieur"
    elif comparable_perf and stability_better:
        final_option = "2. Le signal d'ensemble est comparable mais plus robuste, donc preferable"
    elif not ensemble_adds_value_vs_baseline:
        final_option = "4. Le signal d'ensemble n'apporte pas de valeur"
    else:
        final_option = "1. Le point unique reste meilleur"

    report_lines = [
        "# ATR Ensemble Campaign Report",
        "",
        "## Rappel",
        "",
        "- Baseline: ORB long + VWAP continue, sans filtre ATR additionnel.",
        f"- Point unique de reference: ATR({spec.atr_period}), q{_pct_tag(spec.point_q_low_pct)}/q{_pct_tag(spec.point_q_high_pct)}.",
        "- Ensembles testes: agregations de sous-signaux ATR sur zones voisines.",
        "",
        "## Protocole IS/OOS",
        "",
        f"- Dataset: `{spec.dataset_path.name}`",
        f"- Sessions: {len(all_sessions)} ({all_sessions[0]} -> {all_sessions[-1]})",
        f"- Split: IS={len(is_sessions)} sessions, OOS={len(oos_sessions)} sessions",
        f"- OOS starts at: {oos_sessions[0]}",
        "- Calibration ATR: seuils quantiles calibres sur IS uniquement, puis figes sur OOS.",
        "",
        "## Sous-modeles composant les ensembles",
        "",
        "```text",
        submodels_df.to_string(index=False),
        "```",
        "",
        "## Tableau consolide des variantes",
        "",
        "```text",
        variants_df[
            [
                "strategy_id",
                "strategy_group",
                "zone",
                "aggregation",
                "n_submodels",
                "oos_n_trades",
                "oos_sharpe_ratio",
                "oos_profit_factor",
                "oos_expectancy",
                "oos_max_drawdown",
                "stability_gap",
                "ensemble_score_vs_point",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Diagnostics consensus",
        "",
        "- Distribution des scores exportee dans `diagnostics/consensus_score_distribution.csv`.",
        "- Performance conditionnelle par score exportee dans `diagnostics/consensus_performance_by_score.csv`.",
        "- Figures consensus dans `diagnostics/figures/`.",
        "",
        "## Comparaison cible (Baseline vs Point vs Meilleur Ensemble)",
        "",
        "```text",
        compare_df[
            [
                "strategy_id",
                "oos_n_trades",
                "oos_win_rate",
                "oos_expectancy",
                "oos_profit_factor",
                "oos_sharpe_ratio",
                "oos_cumulative_pnl",
                "oos_max_drawdown",
                "oos_time_exit_rate",
                "oos_time_exit_win_rate",
                "stability_gap",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Reponses aux hypotheses",
        "",
        f"- Q1 (ensemble vs point unique): meilleur ensemble = `{best_ensemble['strategy_id']}`.",
        f"- Q2 (majoritaire vs strict): voir classement `all_variants_metrics.csv`.",
        "- Q3 (consensus fort => meilleure qualite): verifier expectancy/PF par niveau de score dans diagnostics.",
        "- Q4 (robustesse OOS): evaluee via metrics OOS + stability_gap.",
        "",
        "## Conclusion finale",
        "",
        f"- {final_option}",
    ]

    insert_marker = "## Comparaison cible (Baseline vs Point vs Meilleur Ensemble)"
    insert_pos = report_lines.index(insert_marker)
    report_lines[insert_pos:insert_pos] = [
        "- Resume quantitatif OOS par niveau de score:",
        *(consensus_summary_lines if consensus_summary_lines else ["- Aucun resume consensus disponible."]),
        "",
    ]

    report_path = dirs["summary"] / "final_report.md"
    report_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")

    _serialize_json(
        dirs["summary"] / "run_metadata.json",
        {
            "run_timestamp": datetime.now().isoformat(),
            "dataset": str(spec.dataset_path),
            "baseline": asdict(baseline),
            "campaign_spec": asdict(spec),
            "final_option": final_option,
            "best_ensemble_strategy_id": str(best_ensemble["strategy_id"]),
            "output_root": str(dirs["root"]),
        },
    )

    return {
        "output_root": dirs["root"],
        "submodels": dirs["ensemble_models"] / "submodels_definition.csv",
        "variants": dirs["summary"] / "all_variants_metrics.csv",
        "consensus_distribution": dirs["diagnostics"] / "consensus_score_distribution.csv",
        "consensus_performance": dirs["diagnostics"] / "consensus_performance_by_score.csv",
        "report": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ATR ensemble campaign.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CAMPAIGN_DATASET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--is-fraction", type=float, default=0.70)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--point-q-low-pct", type=float, default=20.0)
    parser.add_argument("--point-q-high-pct", type=float, default=90.0)
    parser.add_argument("--no-expanded-zone", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (EXPORTS_DIR / f"atr_ensemble_campaign_{timestamp}")

    spec = CampaignSpec(
        dataset_path=args.dataset,
        is_fraction=args.is_fraction,
        atr_period=args.atr_period,
        point_q_low_pct=args.point_q_low_pct,
        point_q_high_pct=args.point_q_high_pct,
        include_expanded_zone=not args.no_expanded_zone,
    )
    artifacts = run_atr_ensemble_campaign(spec=spec, output_dir=output_dir)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
