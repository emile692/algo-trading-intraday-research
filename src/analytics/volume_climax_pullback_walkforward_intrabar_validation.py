"""Phase 3 strict walk-forward validation for intrabar-aware MNQ pullback execution."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analytics.volume_climax_pullback_common import load_symbol_data, resample_rth_1h, safe_float
from src.analytics.volume_climax_pullback_intrabar_recalibration_campaign import (
    IntrabarRecalibrationConfig,
    _compute_trade_metrics,
    _daily_returns,
    _file_metadata,
    _markdown_table,
    _simulate_config,
    _variant_from_validation_metadata,
)
from src.data.session import extract_rth
from src.engine.vwap_backtester import build_execution_model_for_profile
from src.strategy.volume_climax_pullback_v2 import (
    VolumeClimaxPullbackV2Variant,
    build_volume_climax_pullback_v2_signal_frame,
    prepare_volume_climax_pullback_v2_features,
)

DEFAULT_SYMBOL = "MNQ"
DEFAULT_PHASE2_DIR = Path("export/volume_climax_pullback_intrabar_recalibration_mnq_20260520_101154")
DEFAULT_VALIDATION_DIR = Path("export/volume_climax_pullback_hybrid_execution_validation_mnq_20260519_220109")
DEFAULT_OUTPUT_ROOT = Path("export")
FIXED_CANDIDATE_CONFIG_ID = "require_no_stop_zone_touch_before_entry_sm1p00_tm2p50_d5_stop_zone_0p75"
MIN_TRAIN_DAYS_DYNAMIC = 18 * 30
MIN_TEST_DAYS_DYNAMIC = 90


@dataclass(frozen=True)
class WalkforwardIntrabarConfig:
    config_id: str
    family: str
    symbol: str
    stop_multiplier: float
    target_multiplier: float
    entry_delay_minutes: int
    filter_name: str
    stop_zone_fraction: float | None
    execution_timeframe: str = "1min"
    entry_timing: str = "next_execution_bar_open"
    protective_orders_active_from: str = "next_execution_bar"
    ambiguous_policy: str = "stop_first"

    def to_recalibration_config(self) -> IntrabarRecalibrationConfig:
        params: dict[str, Any] = {}
        label = "none"
        if self.filter_name == "require_no_stop_zone_touch_before_entry":
            if self.stop_zone_fraction is None:
                raise ValueError(f"{self.config_id} is missing stop_zone_fraction.")
            params["stop_zone_fraction"] = float(self.stop_zone_fraction)
            label = f"stop_zone_{float(self.stop_zone_fraction):.2f}".replace(".", "p")
        return IntrabarRecalibrationConfig(
            config_id=self.config_id,
            symbol=self.symbol,
            execution_timeframe=self.execution_timeframe,
            entry_timing=self.entry_timing,
            protective_orders_active_from=self.protective_orders_active_from,
            ambiguous_policy=self.ambiguous_policy,
            stop_multiplier=float(self.stop_multiplier),
            target_multiplier=float(self.target_multiplier),
            entry_delay_minutes=int(self.entry_delay_minutes),
            filter_family=self.filter_name,
            filter_label=label,
            filter_params=params,
        )

    def parameters_json(self) -> str:
        payload = {
            "stop_multiplier": float(self.stop_multiplier),
            "target_multiplier": float(self.target_multiplier),
            "entry_delay_minutes": int(self.entry_delay_minutes),
            "filter_name": self.filter_name,
            "stop_zone_fraction": None if self.stop_zone_fraction is None else float(self.stop_zone_fraction),
            "entry_timing": self.entry_timing,
            "protective_orders_active_from": self.protective_orders_active_from,
            "ambiguous_policy": self.ambiguous_policy,
        }
        return json.dumps(payload, sort_keys=True)


@dataclass(frozen=True)
class WalkforwardFold:
    fold_id: str
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    @property
    def train_days(self) -> int:
        return (self.train_end - self.train_start).days + 1

    @property
    def test_days(self) -> int:
        return (self.test_end - self.test_start).days + 1


def build_phase3_config_universe(symbol: str) -> list[WalkforwardIntrabarConfig]:
    configs: list[WalkforwardIntrabarConfig] = []

    def _append(config: WalkforwardIntrabarConfig) -> None:
        if any(existing.config_id == config.config_id for existing in configs):
            raise ValueError(f"Duplicate config_id in phase 3 universe: {config.config_id}")
        configs.append(config)

    _append(
        WalkforwardIntrabarConfig(
            config_id="none_sm1p00_tm1p00_d0",
            family="benchmark_current_hybrid",
            symbol=symbol,
            stop_multiplier=1.0,
            target_multiplier=1.0,
            entry_delay_minutes=0,
            filter_name="none",
            stop_zone_fraction=None,
        )
    )

    for stop_multiplier in (0.75, 1.0, 1.25):
        for target_multiplier in (2.0, 2.5, 3.0):
            for entry_delay_minutes in (5, 10, 15, 30):
                config_id = f"none_sm{stop_multiplier:.2f}_tm{target_multiplier:.2f}_d{entry_delay_minutes}".replace(".", "p")
                _append(
                    WalkforwardIntrabarConfig(
                        config_id=config_id,
                        family="delay_only",
                        symbol=symbol,
                        stop_multiplier=float(stop_multiplier),
                        target_multiplier=float(target_multiplier),
                        entry_delay_minutes=int(entry_delay_minutes),
                        filter_name="none",
                        stop_zone_fraction=None,
                    )
                )

    sanity_subset = {
        (1.0, 2.0, 5, 0.75),
        (1.0, 2.0, 10, 0.75),
        (1.0, 2.5, 5, 0.75),
        (1.0, 2.5, 10, 0.75),
    }
    for stop_multiplier in (0.75, 1.0, 1.25):
        for target_multiplier in (2.0, 2.5, 3.0):
            for entry_delay_minutes in (5, 10, 15, 30):
                for stop_zone_fraction in (0.5, 0.75, 1.0):
                    family = "sanity_anti_overfit" if (
                        stop_multiplier,
                        target_multiplier,
                        entry_delay_minutes,
                        stop_zone_fraction,
                    ) in sanity_subset else "delay_stop_zone"
                    config_id = (
                        f"require_no_stop_zone_touch_before_entry_sm{stop_multiplier:.2f}_tm{target_multiplier:.2f}_"
                        f"d{entry_delay_minutes}_stop_zone_{stop_zone_fraction:.2f}"
                    ).replace(".", "p")
                    _append(
                        WalkforwardIntrabarConfig(
                            config_id=config_id,
                            family=family,
                            symbol=symbol,
                            stop_multiplier=float(stop_multiplier),
                            target_multiplier=float(target_multiplier),
                            entry_delay_minutes=int(entry_delay_minutes),
                            filter_name="require_no_stop_zone_touch_before_entry",
                            stop_zone_fraction=float(stop_zone_fraction),
                        )
                    )
    return configs


def build_walkforward_folds(session_dates: Sequence[date]) -> list[WalkforwardFold]:
    unique_dates = sorted({pd.Timestamp(value).date() for value in session_dates if pd.notna(value)})
    if not unique_dates:
        raise ValueError("No session dates available to build walk-forward folds.")

    min_date = unique_dates[0]
    max_date = unique_dates[-1]
    preferred = [
        ("fold_1", date(2020, 1, 1), date(2021, 12, 31), date(2022, 1, 1), date(2022, 12, 31)),
        ("fold_2", date(2020, 1, 1), date(2022, 12, 31), date(2023, 1, 1), date(2023, 12, 31)),
        ("fold_3", date(2020, 1, 1), date(2023, 12, 31), date(2024, 1, 1), date(2024, 12, 31)),
        ("fold_4", date(2020, 1, 1), date(2024, 12, 31), date(2025, 1, 1), date(2025, 12, 31)),
        ("fold_5", date(2020, 1, 1), date(2025, 12, 31), date(2026, 1, 1), max_date),
    ]
    folds: list[WalkforwardFold] = []
    for fold_id, train_start, train_end, test_start, test_end in preferred:
        actual_train_start = max(train_start, min_date)
        actual_test_end = min(test_end, max_date)
        if actual_train_start > train_end or test_start > max_date:
            continue
        if train_end >= test_start or actual_test_end < test_start:
            continue
        folds.append(
            WalkforwardFold(
                fold_id=fold_id,
                train_start=actual_train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=actual_test_end,
            )
        )
    if folds:
        return folds

    anchor = min_date
    dynamic_folds: list[WalkforwardFold] = []
    years = sorted({value.year for value in unique_dates})
    for year in years:
        test_start = date(year, 1, 1)
        if test_start <= anchor:
            continue
        train_end = test_start - pd.Timedelta(days=1)
        train_days = (train_end.date() - anchor).days + 1 if isinstance(train_end, pd.Timestamp) else (train_end - anchor).days + 1
        actual_train_end = train_end.date() if isinstance(train_end, pd.Timestamp) else train_end
        actual_test_end = min(date(year, 12, 31), max_date)
        test_days = (actual_test_end - test_start).days + 1
        if train_days < MIN_TRAIN_DAYS_DYNAMIC or test_days < MIN_TEST_DAYS_DYNAMIC:
            continue
        dynamic_folds.append(
            WalkforwardFold(
                fold_id=f"fold_{len(dynamic_folds) + 1}",
                train_start=anchor,
                train_end=actual_train_end,
                test_start=test_start,
                test_end=actual_test_end,
            )
        )
    if not dynamic_folds:
        raise ValueError("Unable to build dynamic walk-forward folds with minimum train/test lengths.")
    return dynamic_folds


def _period_mask(events: pd.DataFrame, start_date: date, end_date: date) -> pd.Series:
    session_dates = pd.to_datetime(events["session_date"], errors="coerce").dt.date
    return session_dates.between(start_date, end_date)


def _normalize(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.notna().sum() <= 1:
        return pd.Series(np.where(numeric.notna(), 1.0, np.nan), index=series.index, dtype=float)
    spread = numeric.max() - numeric.min()
    if spread == 0:
        return pd.Series(np.where(numeric.notna(), 1.0, np.nan), index=series.index, dtype=float)
    return (numeric - numeric.min()) / spread


def _config_neighbors(rank_frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    stop_zone = row.get("stop_zone_fraction")
    if pd.isna(stop_zone):
        zone_match = rank_frame["stop_zone_fraction"].isna()
    else:
        zone_match = pd.to_numeric(rank_frame["stop_zone_fraction"], errors="coerce").sub(float(stop_zone)).abs() < 1e-9
    return rank_frame.loc[
        (rank_frame["filter_name"] == row["filter_name"])
        & zone_match
        & (pd.to_numeric(rank_frame["stop_multiplier"], errors="coerce").sub(float(row["stop_multiplier"])).abs() <= 0.26)
        & (pd.to_numeric(rank_frame["target_multiplier"], errors="coerce").sub(float(row["target_multiplier"])).abs() <= 0.51)
        & (pd.to_numeric(rank_frame["entry_delay_minutes"], errors="coerce").sub(int(row["entry_delay_minutes"])).abs() <= 10)
    ].copy()


def compute_fold_train_ranking(
    fold: WalkforwardFold,
    config_universe_df: pd.DataFrame,
    events_by_config: dict[str, pd.DataFrame],
    *,
    estimated_cost_per_trade: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    years_in_train = list(range(fold.train_start.year, fold.train_end.year + 1))
    for _, config_row in config_universe_df.iterrows():
        config_id = str(config_row["config_id"])
        events = events_by_config[config_id].loc[_period_mask(events_by_config[config_id], fold.train_start, fold.train_end)].copy()
        metrics = _compute_trade_metrics(events, estimated_cost_per_trade=estimated_cost_per_trade)
        executed = events.loc[events["executed"]].copy()
        yearly_rows = []
        for year in years_in_train:
            year_slice = events.loc[pd.to_datetime(events["session_date"], errors="coerce").dt.year == year].copy()
            year_metrics = _compute_trade_metrics(year_slice, estimated_cost_per_trade=estimated_cost_per_trade)
            year_metrics["year"] = year
            yearly_rows.append(year_metrics)
        yearly_df = pd.DataFrame(yearly_rows)
        positive_years = int((pd.to_numeric(yearly_df["net_pnl"], errors="coerce") > 0).sum()) if not yearly_df.empty else 0
        years_with_trades = int((pd.to_numeric(yearly_df["trades"], errors="coerce") > 0).sum()) if not yearly_df.empty else 0
        pnl_by_year = pd.to_numeric(yearly_df["net_pnl"], errors="coerce").fillna(0.0)
        abs_total = float(pnl_by_year.abs().sum())
        max_year_contribution_pct = float(pnl_by_year.abs().max() / abs_total) if abs_total > 0 else 1.0
        rows.append(
            {
                "fold_id": fold.fold_id,
                "config_id": config_id,
                "family": config_row["family"],
                "filter_name": config_row["filter_name"],
                "stop_multiplier": config_row["stop_multiplier"],
                "target_multiplier": config_row["target_multiplier"],
                "entry_delay_minutes": config_row["entry_delay_minutes"],
                "stop_zone_fraction": config_row["stop_zone_fraction"],
                "train_trades": int(metrics["trades"]),
                "train_net_pnl": float(metrics["net_pnl"]),
                "train_profit_factor": float(metrics["profit_factor"]),
                "train_winrate": float(metrics["winrate"]),
                "train_avg_trade": float(metrics["avg_trade"]),
                "train_max_drawdown": float(metrics["max_drawdown"]),
                "train_pnl_to_maxdd": float(metrics["pnl_to_maxdd"]) if pd.notna(metrics["pnl_to_maxdd"]) else np.nan,
                "train_skip_rate": float(metrics["skip_rate"]),
                "train_blocked_setups": int(metrics["blocked_setups"]),
                "positive_years_train": positive_years,
                "years_with_trades_train": years_with_trades,
                "max_year_contribution_pct": max_year_contribution_pct,
                "train_executed_count": int(len(executed)),
            }
        )

    ranking = pd.DataFrame(rows)
    ranking["normalized_net_pnl"] = _normalize(ranking["train_net_pnl"])
    ranking["normalized_profit_factor"] = _normalize(ranking["train_profit_factor"].replace(np.inf, np.nan))
    ranking["normalized_pnl_to_maxdd"] = _normalize(ranking["train_pnl_to_maxdd"])
    ranking["trade_count_score"] = pd.to_numeric(ranking["train_trades"], errors="coerce").clip(lower=0).div(60.0).clip(upper=1.0)

    temporal_scores: list[float] = []
    neighborhood_scores: list[float] = []
    penalties_list: list[float] = []
    admissible_list: list[bool] = []
    for _, row in ranking.iterrows():
        years_with_trades = int(row["years_with_trades_train"])
        positive_years = int(row["positive_years_train"])
        max_year_contribution_pct = safe_float(row["max_year_contribution_pct"], 1.0)
        if years_with_trades > 0:
            temporal_stability = 0.5 * (positive_years / years_with_trades) + 0.5 * max(0.0, 1.0 - max_year_contribution_pct)
        else:
            temporal_stability = 0.0
        neighbors = _config_neighbors(ranking, row)
        if neighbors.empty:
            neighborhood_score = 0.0
        else:
            acceptable_neighbors = (
                (pd.to_numeric(neighbors["train_net_pnl"], errors="coerce") > 0)
                & (pd.to_numeric(neighbors["train_profit_factor"], errors="coerce") >= 1.0)
            )
            neighborhood_score = float(acceptable_neighbors.mean())

        penalties = 0.0
        if safe_float(row["train_trades"], 0.0) < 30:
            penalties += 0.20
        if safe_float(row["train_profit_factor"], 0.0) < 1.05:
            penalties += 0.15
        if max_year_contribution_pct > 0.70:
            penalties += 0.15
        if safe_float(row["train_skip_rate"], 0.0) > 0.70:
            penalties += 0.10
        if safe_float(row["train_avg_trade"], 0.0) <= estimated_cost_per_trade * 1.5:
            penalties += 0.15
        if safe_float(row["train_max_drawdown"], 0.0) < -abs(safe_float(row["train_net_pnl"], 0.0)) * 2.0 and safe_float(row["train_net_pnl"], 0.0) > 0:
            penalties += 0.10

        if years_with_trades >= 2:
            positive_years_requirement = positive_years >= 2
        else:
            positive_years_requirement = positive_years >= 1

        admissible = bool(
            safe_float(row["train_net_pnl"], 0.0) > 0
            and safe_float(row["train_trades"], 0.0) >= 30
            and safe_float(row["train_profit_factor"], 0.0) >= 1.10
            and safe_float(row["train_avg_trade"], 0.0) > estimated_cost_per_trade * 1.5
            and positive_years_requirement
        )

        temporal_scores.append(temporal_stability)
        neighborhood_scores.append(neighborhood_score)
        penalties_list.append(penalties)
        admissible_list.append(admissible)

    ranking["temporal_stability"] = temporal_scores
    ranking["neighborhood_stability"] = neighborhood_scores
    ranking["penalties"] = penalties_list
    ranking["admissible_train"] = admissible_list
    ranking["train_robust_score"] = (
        0.25 * ranking["normalized_net_pnl"].fillna(0.0)
        + 0.20 * ranking["normalized_profit_factor"].fillna(0.0)
        + 0.15 * ranking["normalized_pnl_to_maxdd"].fillna(0.0)
        + 0.15 * ranking["temporal_stability"].fillna(0.0)
        + 0.15 * ranking["neighborhood_stability"].fillna(0.0)
        + 0.10 * ranking["trade_count_score"].fillna(0.0)
        - ranking["penalties"].fillna(0.0)
    )
    ranking = ranking.sort_values(
        ["admissible_train", "train_robust_score", "train_net_pnl", "train_profit_factor", "train_skip_rate"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    ranking["train_rank"] = np.arange(1, len(ranking) + 1, dtype=int)
    ranking["selected_in_fold"] = False
    if not ranking.empty:
        ranking.loc[0, "selected_in_fold"] = True
    return ranking


def select_fold_winner(ranking: pd.DataFrame) -> pd.Series:
    if ranking.empty:
        raise ValueError("Cannot select a fold winner from an empty ranking.")
    return ranking.iloc[0].copy()


def _slice_events(events: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    return events.loc[_period_mask(events, start_date, end_date)].copy()


def build_walkforward_summary(
    stitched_trades: pd.DataFrame,
    fold_selected_test_results: pd.DataFrame,
    benchmark_events: pd.DataFrame,
    config_universe_df: pd.DataFrame,
    folds: Sequence[WalkforwardFold],
    *,
    estimated_cost_per_trade: float,
) -> pd.DataFrame:
    stitched_metrics = _compute_trade_metrics(stitched_trades, estimated_cost_per_trade=estimated_cost_per_trade)
    benchmark_slices = []
    for fold in folds:
        benchmark_slices.append(_slice_events(benchmark_events, fold.test_start, fold.test_end))
    benchmark_test_events = pd.concat(benchmark_slices, ignore_index=True) if benchmark_slices else pd.DataFrame()
    benchmark_metrics = _compute_trade_metrics(benchmark_test_events, estimated_cost_per_trade=estimated_cost_per_trade)
    family_counts = (
        fold_selected_test_results["selected_family"].value_counts().sort_index().to_dict()
        if not fold_selected_test_results.empty
        else {}
    )
    summary = pd.DataFrame(
        [
            {
                "total_test_trades": int(stitched_metrics["trades"]),
                "total_test_net_pnl": float(stitched_metrics["net_pnl"]),
                "test_profit_factor": float(stitched_metrics["profit_factor"]),
                "test_winrate": float(stitched_metrics["winrate"]),
                "avg_trade": float(stitched_metrics["avg_trade"]),
                "max_drawdown": float(stitched_metrics["max_drawdown"]),
                "pnl_to_maxdd": float(stitched_metrics["pnl_to_maxdd"]) if pd.notna(stitched_metrics["pnl_to_maxdd"]) else np.nan,
                "number_of_folds": int(len(folds)),
                "positive_folds": int((pd.to_numeric(fold_selected_test_results["test_net_pnl"], errors="coerce") > 0).sum()) if not fold_selected_test_results.empty else 0,
                "negative_folds": int((pd.to_numeric(fold_selected_test_results["test_net_pnl"], errors="coerce") <= 0).sum()) if not fold_selected_test_results.empty else 0,
                "pass_rate": float(pd.to_numeric(fold_selected_test_results["test_pass"], errors="coerce").fillna(False).astype(bool).mean()) if not fold_selected_test_results.empty else 0.0,
                "selected_config_diversity": int(fold_selected_test_results["selected_config_id"].nunique()) if not fold_selected_test_results.empty else 0,
                "selected_family_counts": json.dumps(family_counts, sort_keys=True),
                "benchmark_current_hybrid_net_pnl_over_same_test_windows": float(benchmark_metrics["net_pnl"]),
                "improvement_vs_current_hybrid": float(stitched_metrics["net_pnl"] - benchmark_metrics["net_pnl"]),
            }
        ]
    )
    return summary


def _folds_to_frame(folds: Sequence[WalkforwardFold]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold_id": fold.fold_id,
                "train_start": fold.train_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "train_days": fold.train_days,
                "test_days": fold.test_days,
            }
            for fold in folds
        ]
    )


def _plot_walkforward_outputs(
    *,
    output_dir: Path,
    stitched_daily: pd.DataFrame,
    fold_selected_test_results: pd.DataFrame,
    family_level_summary: pd.DataFrame,
    fixed_candidate_tracking: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
) -> list[str]:
    plot_paths: list[str] = []

    if not stitched_daily.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(pd.to_datetime(stitched_daily["session_date"]), stitched_daily["equity"], label="walkforward_stitched")
        ax.set_title("Walkforward Stitched Cumulative PnL")
        ax.legend()
        fig.tight_layout()
        path = output_dir / "walkforward_stitched_cumulative_pnl.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

    if not fold_selected_test_results.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(fold_selected_test_results["fold_id"], pd.to_numeric(fold_selected_test_results["test_net_pnl"], errors="coerce"))
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title("Fold Test PnL")
        fig.tight_layout()
        path = output_dir / "fold_test_pnl_bar.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(
            fold_selected_test_results["fold_id"],
            pd.to_numeric(fold_selected_test_results["train_robust_score"], errors="coerce"),
            s=80,
        )
        for _, row in fold_selected_test_results.iterrows():
            ax.annotate(str(row["selected_config_id"]), (row["fold_id"], safe_float(row["train_robust_score"], 0.0)), fontsize=8)
        ax.set_title("Selected Config By Fold")
        ax.set_ylabel("train_robust_score")
        fig.tight_layout()
        path = output_dir / "selected_config_by_fold.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            pd.to_numeric(fold_selected_test_results["train_robust_score"], errors="coerce"),
            pd.to_numeric(fold_selected_test_results["test_net_pnl"], errors="coerce"),
            s=70,
        )
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel("train_robust_score")
        ax.set_ylabel("test_net_pnl")
        ax.set_title("Train Score vs Test PnL")
        fig.tight_layout()
        path = output_dir / "train_score_vs_test_pnl_scatter.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

    if not family_level_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=family_level_summary, x="family", y="median_test_pnl_across_configs", ax=ax)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title("Family Test Distribution")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        path = output_dir / "family_test_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

    if not fixed_candidate_tracking.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(fixed_candidate_tracking["fold_id"], pd.to_numeric(fixed_candidate_tracking["test_net_pnl"], errors="coerce"))
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title("Fixed Candidate Fold PnL")
        fig.tight_layout()
        path = output_dir / "fixed_candidate_fold_pnl.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

    if not stitched_daily.empty and not benchmark_daily.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(pd.to_datetime(benchmark_daily["session_date"]), benchmark_daily["equity"], label="current_hybrid_same_test_windows")
        ax.plot(pd.to_datetime(stitched_daily["session_date"]), stitched_daily["equity"], label="walkforward_stitched")
        ax.set_title("Benchmark vs Walkforward Cumulative")
        ax.legend()
        fig.tight_layout()
        path = output_dir / "benchmark_vs_walkforward_cumulative.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

    return plot_paths


def _build_family_level_summary(
    fold_train_ranking: pd.DataFrame,
    all_test_metrics: pd.DataFrame,
    fold_selected_test_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, train_slice in fold_train_ranking.groupby("family", sort=True):
        test_slice = all_test_metrics.loc[all_test_metrics["family"] == family].copy()
        selected_slice = fold_selected_test_results.loc[fold_selected_test_results["selected_family"] == family].copy()
        rows.append(
            {
                "family": family,
                "avg_rank_train": float(pd.to_numeric(train_slice["train_rank"], errors="coerce").mean()),
                "times_selected": int(len(selected_slice)),
                "stitched_test_net_pnl_if_selected": float(pd.to_numeric(selected_slice["test_net_pnl"], errors="coerce").sum()) if not selected_slice.empty else 0.0,
                "fixed_oracle_warning": "descriptive_only_test_stats",
                "median_test_pnl_across_configs": float(pd.to_numeric(test_slice["test_net_pnl"], errors="coerce").median()) if not test_slice.empty else np.nan,
                "pct_configs_positive_test": float((pd.to_numeric(test_slice["test_net_pnl"], errors="coerce") > 0).mean()) if not test_slice.empty else np.nan,
                "pct_configs_pf_above_1": float((pd.to_numeric(test_slice["test_profit_factor"], errors="coerce") >= 1.0).mean()) if not test_slice.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["times_selected", "median_test_pnl_across_configs"], ascending=[False, False]).reset_index(drop=True)


def _compute_test_metrics_per_config(
    folds: Sequence[WalkforwardFold],
    config_universe_df: pd.DataFrame,
    events_by_config: dict[str, pd.DataFrame],
    *,
    estimated_cost_per_trade: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for _, config_row in config_universe_df.iterrows():
            config_id = str(config_row["config_id"])
            events = _slice_events(events_by_config[config_id], fold.test_start, fold.test_end)
            metrics = _compute_trade_metrics(events, estimated_cost_per_trade=estimated_cost_per_trade)
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "config_id": config_id,
                    "family": config_row["family"],
                    "test_trades": int(metrics["trades"]),
                    "test_net_pnl": float(metrics["net_pnl"]),
                    "test_profit_factor": float(metrics["profit_factor"]),
                    "test_winrate": float(metrics["winrate"]),
                    "test_avg_trade": float(metrics["avg_trade"]),
                    "test_max_drawdown": float(metrics["max_drawdown"]),
                    "test_pnl_to_maxdd": float(metrics["pnl_to_maxdd"]) if pd.notna(metrics["pnl_to_maxdd"]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _verdict_from_summary(summary_row: pd.Series) -> str:
    total_net = safe_float(summary_row.get("total_test_net_pnl"), 0.0)
    pass_rate = safe_float(summary_row.get("pass_rate"), 0.0)
    positive_folds = int(safe_float(summary_row.get("positive_folds"), 0.0))
    num_folds = max(int(safe_float(summary_row.get("number_of_folds"), 0.0)), 1)
    pf = safe_float(summary_row.get("test_profit_factor"), 0.0)

    if total_net <= 0 or pf < 1.0 or pass_rate < 0.40:
        return "Reject: no reliable walk-forward edge."
    if total_net > 0 and pf >= 1.15 and pass_rate >= 0.80 and positive_folds >= max(3, num_folds - 1):
        return "Strong candidate: robust across folds and families."
    if pass_rate < 0.60 or positive_folds < max(2, num_folds // 2):
        return "Weak watchlist: some positive pockets, not tradable."
    if total_net > 0 and pf >= 1.05:
        return "Candidate: walk-forward positive but requires multi-asset confirmation."
    return "Weak watchlist: some positive pockets, not tradable."


def _build_final_report(
    *,
    output_dir: Path,
    folds_frame: pd.DataFrame,
    config_universe_df: pd.DataFrame,
    fold_selected_test_results: pd.DataFrame,
    walkforward_summary: pd.DataFrame,
    fixed_candidate_tracking: pd.DataFrame,
    family_level_summary: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    phase2_context: dict[str, Any],
    verdict: str,
) -> None:
    summary_row = walkforward_summary.iloc[0] if not walkforward_summary.empty else pd.Series(dtype=object)
    lines = [
        "# Volume Climax Pullback Walk-Forward Intrabar Validation",
        "",
        "## 1. Executive Summary",
        f"- The delay + stop-zone family {'shows some persistence' if verdict != 'Reject: no reliable walk-forward edge.' else 'does not show a reliable walk-forward edge'} under strict train-only selection.",
        f"- Positive folds: `{int(safe_float(summary_row.get('positive_folds'), 0.0))}` / `{int(safe_float(summary_row.get('number_of_folds'), 0.0))}`.",
        f"- Stitched OOS net PnL: `{safe_float(summary_row.get('total_test_net_pnl'), 0.0):.2f} USD`.",
        f"- Improvement vs current hybrid over the same stitched test windows: `{safe_float(summary_row.get('improvement_vs_current_hybrid'), 0.0):.2f} USD`.",
        f"- Verdict: `{verdict}`.",
        "",
        "## 2. Context",
        "- The 1H baseline remains invalidated and is never used for selection.",
        f"- Current hybrid benchmark from phase 1 remained negative at `{safe_float(validation_metrics.loc[validation_metrics['scenario'] == 'hybrid_next_execution_bar', 'net_pnl_usd'].iloc[0], np.nan):.2f} USD` when executed realistically.",
        "- Phase 2 best IS-only config failed global OOS, while one delay + stop-zone config looked informative but non-authoritative.",
        "",
        "## 3. Walk-Forward Design",
        "- Signal timeframe stays 1H, execution timeframe stays 1min.",
        "- Each fold selects on train only using a deterministic robustness score, then applies the winner to the strictly future test window.",
        _markdown_table(folds_frame),
        "",
        "## 4. Config Universe",
        _markdown_table(
            config_universe_df.groupby("family", as_index=False)
            .agg(configs=("config_id", "count"))
            .sort_values("family")
        ),
        "",
        "## 5. Fold-by-Fold Selection",
        _markdown_table(
            fold_selected_test_results[
                [
                    "fold_id",
                    "selected_config_id",
                    "selected_family",
                    "train_robust_score",
                    "train_net_pnl",
                    "train_profit_factor",
                    "test_net_pnl",
                    "test_profit_factor",
                    "test_pass",
                ]
            ]
        ) if not fold_selected_test_results.empty else "No fold selection available.",
        "",
        "## 6. Stitched OOS Performance",
        _markdown_table(walkforward_summary),
        "",
        "## 7. Fixed Candidate Tracking",
        _markdown_table(fixed_candidate_tracking) if not fixed_candidate_tracking.empty else "Fixed candidate not available.",
        "",
        "## 8. Family-Level Diagnostics",
        _markdown_table(family_level_summary) if not family_level_summary.empty else "No family diagnostics available.",
        "",
        "## 9. Train Score vs Test Reality",
        f"- Correlation proxy between train score and test PnL is `{safe_float(fold_selected_test_results['train_robust_score'].corr(fold_selected_test_results['test_net_pnl']), np.nan):.4f}` across selected folds."
        if len(fold_selected_test_results) >= 2
        else "- Too few folds to estimate train-score vs test-PnL correlation robustly.",
        "",
        "## 10. Failure Modes",
        "- Typical failure modes remain fold concentration, low trade density in some windows, and parameter instability between adjacent delay / stop-zone variants.",
        "- If train winners repeatedly fail future tests, the phase 2 positive pocket should be treated as cherry-pick risk rather than surviving alpha.",
        "",
        "## 11. Verdict",
        f"- `{verdict}`",
        "",
        "## 12. Next Actions",
        "- Stop treating MNQ pullback standalone as a deployable alpha if the verdict remains reject.",
        "- Keep the signal only as a candidate feature or overlay inside broader intraday frameworks.",
        "- Focus forward effort on ORB / TopstepX-ready paths before spending more on standalone MNQ pullback.",
        "- If the verdict improves later, validate the same family on MES, M2K and MGC with the same walk-forward discipline.",
        "- Add a CI guardrail that blocks intraday research publication without walk-forward intrabar validation.",
    ]
    (output_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_walkforward_validation(
    *,
    symbol: str,
    phase2_dir: Path,
    validation_dir: Path,
    output_root: Path,
    raw_minute_df_override: pd.DataFrame | None = None,
    variant_override: VolumeClimaxPullbackV2Variant | None = None,
    config_universe_override: list[WalkforwardIntrabarConfig] | None = None,
    events_by_config_override: dict[str, pd.DataFrame] | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / f"volume_climax_pullback_walkforward_intrabar_validation_{symbol.lower()}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    phase2_dir = Path(phase2_dir)
    validation_dir = Path(validation_dir)
    phase2_metadata_path = phase2_dir / "run_metadata.json"
    if not phase2_metadata_path.exists():
        raise FileNotFoundError(f"Missing phase 2 run metadata: {phase2_metadata_path}")
    validation_metadata_path = validation_dir / "run_metadata.json"
    if not validation_metadata_path.exists():
        raise FileNotFoundError(f"Missing validation run metadata: {validation_metadata_path}")

    phase2_context = json.loads(phase2_metadata_path.read_text(encoding="utf-8"))
    variant, validation_metadata = (
        (variant_override, {"dataset_path": phase2_context.get("dataset_path"), "variant": asdict(variant_override)})
        if variant_override is not None
        else _variant_from_validation_metadata(validation_dir)
    )

    dataset_path = Path(validation_metadata.get("dataset_path") or phase2_context.get("dataset_path") or "")
    if raw_minute_df_override is not None:
        raw_minute_df = raw_minute_df_override.copy()
    else:
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Input dataset not found at {dataset_path}. "
                "Provide a valid validation-dir / phase2-dir pair with dataset_path in run_metadata.json."
            )
        raw_minute_df = load_symbol_data(symbol, input_paths={symbol: dataset_path})

    minute_df = extract_rth(raw_minute_df.copy())
    minute_df["timestamp"] = pd.to_datetime(minute_df["timestamp"], errors="coerce")
    minute_df["session_date"] = minute_df["timestamp"].dt.date
    session_dates = minute_df["session_date"].dropna().tolist()
    folds = build_walkforward_folds(session_dates)
    folds_frame = _folds_to_frame(folds)

    config_universe = config_universe_override or build_phase3_config_universe(symbol)
    config_universe_df = pd.DataFrame(
        [
            {
                "config_id": config.config_id,
                "family": config.family,
                "stop_multiplier": config.stop_multiplier,
                "target_multiplier": config.target_multiplier,
                "entry_delay_minutes": config.entry_delay_minutes,
                "filter_name": config.filter_name,
                "stop_zone_fraction": config.stop_zone_fraction,
                "parameters_json": config.parameters_json(),
            }
            for config in config_universe
        ]
    )
    if FIXED_CANDIDATE_CONFIG_ID not in set(config_universe_df["config_id"]):
        raise ValueError(
            f"The fixed candidate '{FIXED_CANDIDATE_CONFIG_ID}' is missing from the phase 3 config universe."
        )

    bars_1h = resample_rth_1h(raw_minute_df)
    bars_1h["timestamp"] = pd.to_datetime(bars_1h["timestamp"], errors="coerce")
    bars_1h["session_date"] = bars_1h["timestamp"].dt.date
    features = prepare_volume_climax_pullback_v2_features(bars_1h)
    signal_df = build_volume_climax_pullback_v2_signal_frame(features, variant)

    execution_model, instrument = build_execution_model_for_profile(symbol=symbol, profile_name="repo_realistic")
    estimated_cost_per_trade = float(execution_model.round_trip_fees(quantity=1))

    if events_by_config_override is not None:
        events_by_config = {key: value.copy() for key, value in events_by_config_override.items()}
    else:
        events_by_config = {}
        for config in config_universe:
            events = _simulate_config(
                config=config.to_recalibration_config(),
                signal_df=signal_df,
                minute_df=minute_df,
                variant=variant,
                execution_model=execution_model,
                point_value_usd=float(instrument.point_value_usd),
                tick_size=float(instrument.tick_size),
            )
            events["family"] = config.family
            events["filter_name"] = config.filter_name
            events["stop_zone_fraction"] = config.stop_zone_fraction
            events_by_config[config.config_id] = events

    fold_rankings: list[pd.DataFrame] = []
    selected_test_rows: list[dict[str, Any]] = []
    stitched_rows: list[pd.DataFrame] = []
    fixed_candidate_rows: list[dict[str, Any]] = []
    all_test_metrics = _compute_test_metrics_per_config(
        folds,
        config_universe_df,
        events_by_config,
        estimated_cost_per_trade=estimated_cost_per_trade,
    )

    for fold in folds:
        ranking = compute_fold_train_ranking(
            fold,
            config_universe_df,
            events_by_config,
            estimated_cost_per_trade=estimated_cost_per_trade,
        )
        fold_rankings.append(ranking)
        winner = select_fold_winner(ranking)
        selected_config_id = str(winner["config_id"])
        test_events = _slice_events(events_by_config[selected_config_id], fold.test_start, fold.test_end)
        test_metrics = _compute_trade_metrics(test_events, estimated_cost_per_trade=estimated_cost_per_trade)
        selected_test_rows.append(
            {
                "fold_id": fold.fold_id,
                "selected_config_id": selected_config_id,
                "selected_family": winner["family"],
                "train_robust_score": float(winner["train_robust_score"]),
                "train_net_pnl": float(winner["train_net_pnl"]),
                "train_profit_factor": float(winner["train_profit_factor"]),
                "test_trades": int(test_metrics["trades"]),
                "test_net_pnl": float(test_metrics["net_pnl"]),
                "test_profit_factor": float(test_metrics["profit_factor"]),
                "test_winrate": float(test_metrics["winrate"]),
                "test_avg_trade": float(test_metrics["avg_trade"]),
                "test_max_drawdown": float(test_metrics["max_drawdown"]),
                "test_pnl_to_maxdd": float(test_metrics["pnl_to_maxdd"]) if pd.notna(test_metrics["pnl_to_maxdd"]) else np.nan,
                "test_pass": bool(
                    safe_float(test_metrics["net_pnl"], 0.0) > 0
                    and safe_float(test_metrics["profit_factor"], 0.0) >= 1.05
                    and safe_float(test_metrics["trades"], 0.0) >= 10
                    and safe_float(test_metrics["avg_trade"], 0.0) > estimated_cost_per_trade
                ),
            }
        )

        fold_rows = test_events.copy()
        fold_rows["fold_id"] = fold.fold_id
        fold_rows["config_id"] = selected_config_id
        stitched_rows.append(fold_rows)

        fixed_events = _slice_events(events_by_config[FIXED_CANDIDATE_CONFIG_ID], fold.test_start, fold.test_end)
        fixed_metrics = _compute_trade_metrics(fixed_events, estimated_cost_per_trade=estimated_cost_per_trade)
        fixed_candidate_rows.append(
            {
                "fold_id": fold.fold_id,
                "config_id": FIXED_CANDIDATE_CONFIG_ID,
                "test_trades": int(fixed_metrics["trades"]),
                "test_net_pnl": float(fixed_metrics["net_pnl"]),
                "test_profit_factor": float(fixed_metrics["profit_factor"]),
                "test_winrate": float(fixed_metrics["winrate"]),
                "test_avg_trade": float(fixed_metrics["avg_trade"]),
                "test_max_drawdown": float(fixed_metrics["max_drawdown"]),
            }
        )

    fold_train_ranking = pd.concat(fold_rankings, ignore_index=True) if fold_rankings else pd.DataFrame()
    fold_selected_test_results = pd.DataFrame(selected_test_rows)
    stitched_trades = pd.concat(stitched_rows, ignore_index=True) if stitched_rows else pd.DataFrame()
    stitched_daily = _daily_returns(stitched_trades)
    fixed_candidate_tracking = pd.DataFrame(fixed_candidate_rows)

    current_hybrid_id = str(phase2_context.get("current_hybrid_config_id") or "none_sm1p00_tm1p00_d0")
    if current_hybrid_id not in events_by_config:
        raise ValueError(
            f"Current hybrid benchmark config '{current_hybrid_id}' was not simulated in phase 3. "
            "Keep benchmark_current_hybrid in the config universe."
        )
    benchmark_events = pd.concat(
        [_slice_events(events_by_config[current_hybrid_id], fold.test_start, fold.test_end) for fold in folds],
        ignore_index=True,
    )
    benchmark_daily = _daily_returns(benchmark_events)

    walkforward_summary = build_walkforward_summary(
        stitched_trades,
        fold_selected_test_results,
        events_by_config[current_hybrid_id],
        config_universe_df,
        folds,
        estimated_cost_per_trade=estimated_cost_per_trade,
    )
    family_level_summary = _build_family_level_summary(fold_train_ranking, all_test_metrics, fold_selected_test_results)

    validation_metrics = pd.read_csv(validation_dir / "metrics_comparison.csv")
    verdict = _verdict_from_summary(walkforward_summary.iloc[0]) if not walkforward_summary.empty else "Reject: no reliable walk-forward edge."

    folds_frame.to_csv(output_dir / "walkforward_folds.csv", index=False)
    config_universe_df.to_csv(output_dir / "config_universe.csv", index=False)
    fold_train_ranking.to_csv(output_dir / "fold_train_ranking.csv", index=False)
    fold_selected_test_results.to_csv(output_dir / "fold_selected_test_results.csv", index=False)
    stitched_trades.to_csv(output_dir / "walkforward_stitched_trades.csv", index=False)
    stitched_daily.to_csv(output_dir / "walkforward_stitched_daily_returns.csv", index=False)
    walkforward_summary.to_csv(output_dir / "walkforward_summary.csv", index=False)
    fixed_candidate_tracking.to_csv(output_dir / "fixed_candidate_oos_tracking.csv", index=False)
    family_level_summary.to_csv(output_dir / "family_level_walkforward_summary.csv", index=False)
    all_test_metrics.to_csv(output_dir / "fold_test_metrics_all_configs.csv", index=False)

    plot_paths = _plot_walkforward_outputs(
        output_dir=output_dir,
        stitched_daily=stitched_daily,
        fold_selected_test_results=fold_selected_test_results,
        family_level_summary=family_level_summary,
        fixed_candidate_tracking=fixed_candidate_tracking,
        benchmark_daily=benchmark_daily,
    )

    _build_final_report(
        output_dir=output_dir,
        folds_frame=folds_frame,
        config_universe_df=config_universe_df,
        fold_selected_test_results=fold_selected_test_results,
        walkforward_summary=walkforward_summary,
        fixed_candidate_tracking=fixed_candidate_tracking,
        family_level_summary=family_level_summary,
        validation_metrics=validation_metrics,
        phase2_context=phase2_context,
        verdict=verdict,
    )

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "symbol": symbol,
        "phase2_dir": str(phase2_dir),
        "validation_dir": str(validation_dir),
        "output_dir": str(output_dir),
        "dataset_path": str(dataset_path),
        "python_version": sys.version,
        "platform": platform.platform(),
        "variant": asdict(variant),
        "fold_count": int(len(folds)),
        "fixed_candidate_config_id": FIXED_CANDIDATE_CONFIG_ID,
        "current_hybrid_config_id": current_hybrid_id,
        "verdict": verdict,
        "plots": plot_paths,
        "input_files": {
            "phase2_run_metadata": _file_metadata(phase2_metadata_path),
            "validation_run_metadata": _file_metadata(validation_metadata_path),
            "validation_metrics_comparison": _file_metadata(validation_dir / "metrics_comparison.csv"),
        },
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict walk-forward validation for intrabar-aware MNQ pullback.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Symbol to evaluate, default MNQ.")
    parser.add_argument("--phase2-dir", default=str(DEFAULT_PHASE2_DIR), help="Phase 2 recalibration export directory.")
    parser.add_argument("--validation-dir", default=str(DEFAULT_VALIDATION_DIR), help="Validation directory from phase 1.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Export root.")
    args = parser.parse_args()

    run_dir = run_walkforward_validation(
        symbol=str(args.symbol).upper(),
        phase2_dir=Path(args.phase2_dir),
        validation_dir=Path(args.validation_dir),
        output_root=Path(args.output_root),
    )
    print(run_dir)


if __name__ == "__main__":
    main()
