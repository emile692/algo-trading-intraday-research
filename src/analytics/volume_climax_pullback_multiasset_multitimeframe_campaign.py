"""Multi-asset multi-timeframe realistic execution campaign for volume climax pullback."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analytics.volume_climax_pullback_common import (
    latest_path_for_symbol,
    load_symbol_data,
    resample_rth_1h,
    safe_float,
    split_sessions,
)
from src.analytics.volume_climax_pullback_hybrid_execution_validation import _resolve_variant
from src.analytics.volume_climax_pullback_intrabar_recalibration_campaign import (
    IntrabarRecalibrationConfig,
    _compute_trade_metrics,
    _daily_returns,
    _file_metadata,
    _markdown_table,
    _simulate_config,
)
from src.config.settings import DEFAULT_TIMEZONE
from src.data.session import extract_rth
from src.engine.vwap_backtester import build_execution_model_for_profile
from src.strategy.volume_climax_pullback_v2 import (
    VolumeClimaxPullbackV2Variant,
    build_volume_climax_pullback_v2_signal_frame,
    build_volume_climax_pullback_v3_variants,
    prepare_volume_climax_pullback_v2_features,
)

DEFAULT_SYMBOLS = ("MNQ", "MES", "M2K", "MGC")
DEFAULT_SIGNAL_TIMEFRAMES = ("15min", "30min", "1H")
DEFAULT_EXECUTION_TIMEFRAME = "1min"
DEFAULT_OUTPUT_ROOT = Path("export")

FIXED_SPLIT_IS_END = pd.Timestamp("2023-12-31").date()
FIXED_SPLIT_OOS_START = pd.Timestamp("2024-01-01").date()
SUPPORTED_TIMEFRAMES = {"15min": 15, "30min": 30, "1H": 60}

SEED_VARIANT_BY_SYMBOL = {
    "MNQ": "dynamic_exit_atr_target_1p0_ts2_vq0p95_bf0p5_ra1p2",
    "MES": "dynamic_exit_mixed_ts4_vq0p95_bf0p6_ra1p5",
    "M2K": "dynamic_exit_atr_target_1p0_ts4_vq0p95_bf0p5_ra1p2",
    "MGC": "regime_filtered_ema_mild_atr_20_80_compression_off_atr_target_1",
}


@dataclass(frozen=True)
class CampaignConfig:
    config_id: str
    symbol: str
    signal_timeframe: str
    execution_timeframe: str
    base_signal_variant: str
    stop_multiplier: float
    target_multiplier: float
    entry_delay_minutes: int
    filter_name: str
    filter_params: dict[str, Any]
    family: str

    def to_intrabar_config(self) -> IntrabarRecalibrationConfig:
        label = "none"
        if self.filter_name == "require_no_stop_zone_touch_before_entry":
            label = f"stop_zone_{float(self.filter_params['stop_zone_fraction']):.2f}".replace(".", "p")
        elif self.filter_name == "avoid_immediate_adverse_move":
            label = (
                f"adverse_w{int(self.filter_params['adverse_window_minutes'])}_"
                f"ticks{int(self.filter_params['max_adverse_ticks'])}"
            )
        return IntrabarRecalibrationConfig(
            config_id=self.config_id,
            symbol=self.symbol,
            execution_timeframe=self.execution_timeframe,
            entry_timing="next_execution_bar_open",
            protective_orders_active_from="next_execution_bar",
            ambiguous_policy="stop_first",
            stop_multiplier=float(self.stop_multiplier),
            target_multiplier=float(self.target_multiplier),
            entry_delay_minutes=int(self.entry_delay_minutes),
            filter_family=self.filter_name,
            filter_label=label,
            filter_params=dict(self.filter_params),
        )


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


def timeframe_to_minutes(timeframe: str) -> int:
    key = str(timeframe)
    if key not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {timeframe!r}. Supported: {sorted(SUPPORTED_TIMEFRAMES)}")
    return int(SUPPORTED_TIMEFRAMES[key])


def resample_rth_timeframe(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    minutes = timeframe_to_minutes(timeframe)
    scoped = extract_rth(df_1m.copy())
    if scoped.empty:
        return scoped
    scoped["timestamp"] = pd.to_datetime(scoped["timestamp"], errors="coerce")
    scoped = scoped.set_index("timestamp").sort_index()
    bars = scoped.resample(
        f"{minutes}min",
        label="left",
        closed="left",
        origin="start_day",
        offset="30min",
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return bars.dropna(subset=["open", "high", "low", "close"]).reset_index()


def _clone_variant_for_timeframe(variant: VolumeClimaxPullbackV2Variant, timeframe: str) -> VolumeClimaxPullbackV2Variant:
    payload = asdict(variant)
    payload["timeframe"] = str(timeframe)
    return VolumeClimaxPullbackV2Variant(**payload)


def _resolve_seed_variant(symbol: str) -> tuple[VolumeClimaxPullbackV2Variant, bool, str]:
    variants = build_volume_climax_pullback_v3_variants(symbol)
    if not variants:
        raise ValueError(f"No V3 variants available for {symbol}.")
    catalog = {variant.name: variant for variant in variants}
    desired = SEED_VARIANT_BY_SYMBOL.get(symbol.upper())
    if desired and desired in catalog:
        return catalog[desired], False, desired
    if desired and symbol.upper() == "MGC":
        fuzzy = [
            variant
            for variant in variants
            if variant.family == "regime_filtered"
            and variant.trend_ema_window == 50
            and abs(safe_float(variant.ema_slope_threshold, 0.0) - 0.06) <= 1e-9
            and abs(safe_float(variant.atr_percentile_low, 0.0) - 0.20) <= 1e-9
            and abs(safe_float(variant.atr_percentile_high, 0.0) - 0.80) <= 1e-9
            and variant.compression_ratio_max is None
            and variant.exit_mode == "atr_fraction"
        ]
        if fuzzy:
            return fuzzy[0], False, desired
    fallback = _resolve_variant(symbol, None)
    return fallback, True, desired or fallback.name


def build_config_universe_for_symbol_timeframe(
    *,
    symbol: str,
    signal_timeframe: str,
    execution_timeframe: str,
    base_signal_variant: str,
) -> list[CampaignConfig]:
    configs: list[CampaignConfig] = []

    def _add(
        *,
        stop_multiplier: float,
        target_multiplier: float,
        entry_delay_minutes: int,
        filter_name: str,
        filter_params: dict[str, Any],
        family: str,
    ) -> None:
        parts = [
            symbol.lower(),
            signal_timeframe.lower().replace("h", "h"),
            filter_name,
            f"sm{stop_multiplier:.2f}".replace(".", "p"),
            f"tm{target_multiplier:.2f}".replace(".", "p"),
            f"d{entry_delay_minutes}",
        ]
        if filter_name == "require_no_stop_zone_touch_before_entry":
            parts.append(f"sz{float(filter_params['stop_zone_fraction']):.2f}".replace(".", "p"))
        elif filter_name == "avoid_immediate_adverse_move":
            parts.append(
                f"aw{int(filter_params['adverse_window_minutes'])}_mt{int(filter_params['max_adverse_ticks'])}"
            )
        config_id = "_".join(parts)
        configs.append(
            CampaignConfig(
                config_id=config_id,
                symbol=symbol,
                signal_timeframe=signal_timeframe,
                execution_timeframe=execution_timeframe,
                base_signal_variant=base_signal_variant,
                stop_multiplier=float(stop_multiplier),
                target_multiplier=float(target_multiplier),
                entry_delay_minutes=int(entry_delay_minutes),
                filter_name=filter_name,
                filter_params=dict(filter_params),
                family=family,
            )
        )

    _add(
        stop_multiplier=1.0,
        target_multiplier=1.0,
        entry_delay_minutes=0,
        filter_name="none",
        filter_params={},
        family="raw_hybrid",
    )

    for entry_delay_minutes in (5, 15, 30):
        _add(
            stop_multiplier=1.0,
            target_multiplier=1.0,
            entry_delay_minutes=entry_delay_minutes,
            filter_name="none",
            filter_params={},
            family="delay_only",
        )

    for stop_multiplier in (0.75, 1.0, 1.25, 1.5):
        for target_multiplier in (1.5, 2.0, 2.5, 3.0):
            _add(
                stop_multiplier=stop_multiplier,
                target_multiplier=target_multiplier,
                entry_delay_minutes=0,
                filter_name="none",
                filter_params={},
                family="stop_target_recalibration",
            )
            for entry_delay_minutes in (5, 15, 30):
                _add(
                    stop_multiplier=stop_multiplier,
                    target_multiplier=target_multiplier,
                    entry_delay_minutes=entry_delay_minutes,
                    filter_name="none",
                    filter_params={},
                    family="delay_stop_target",
                )

    for stop_multiplier in (0.75, 1.0, 1.25):
        for target_multiplier in (2.0, 2.5, 3.0):
            for entry_delay_minutes in (5, 15, 30):
                for stop_zone_fraction in (0.5, 0.75, 1.0):
                    _add(
                        stop_multiplier=stop_multiplier,
                        target_multiplier=target_multiplier,
                        entry_delay_minutes=entry_delay_minutes,
                        filter_name="require_no_stop_zone_touch_before_entry",
                        filter_params={"stop_zone_fraction": float(stop_zone_fraction)},
                        family="delay_stop_zone_filter",
                    )
                for adverse_window_minutes in (5, 10):
                    for max_adverse_ticks in (8, 12, 16):
                        if entry_delay_minutes < adverse_window_minutes:
                            continue
                        _add(
                            stop_multiplier=stop_multiplier,
                            target_multiplier=target_multiplier,
                            entry_delay_minutes=entry_delay_minutes,
                            filter_name="avoid_immediate_adverse_move",
                            filter_params={
                                "adverse_window_minutes": int(adverse_window_minutes),
                                "max_adverse_ticks": int(max_adverse_ticks),
                            },
                            family="delay_adverse_filter",
                        )

    deduped: dict[str, CampaignConfig] = {config.config_id: config for config in configs}
    return list(deduped.values())


def build_config_universe(
    symbols: Sequence[str],
    signal_timeframes: Sequence[str],
    execution_timeframe: str,
) -> list[CampaignConfig]:
    all_configs: list[CampaignConfig] = []
    for symbol in symbols:
        base_variant, _, _ = _resolve_seed_variant(symbol)
        for timeframe in signal_timeframes:
            all_configs.extend(
                build_config_universe_for_symbol_timeframe(
                    symbol=symbol,
                    signal_timeframe=timeframe,
                    execution_timeframe=execution_timeframe,
                    base_signal_variant=base_variant.name,
                )
            )
    return all_configs


def _config_universe_frame(configs: Sequence[CampaignConfig]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "config_id": config.config_id,
                "symbol": config.symbol,
                "signal_timeframe": config.signal_timeframe,
                "execution_timeframe": config.execution_timeframe,
                "base_signal_variant": config.base_signal_variant,
                "stop_multiplier": config.stop_multiplier,
                "target_multiplier": config.target_multiplier,
                "entry_delay_minutes": config.entry_delay_minutes,
                "filter_name": config.filter_name,
                "filter_params_json": json.dumps(config.filter_params, sort_keys=True),
                "family": config.family,
            }
            for config in configs
        ]
    )


def _infer_split_mode(session_dates: Sequence[date]) -> dict[str, Any]:
    unique_dates = sorted({pd.Timestamp(value).date() for value in session_dates if pd.notna(value)})
    if not unique_dates:
        raise ValueError("No session dates available to infer IS/OOS split.")
    if unique_dates[0] <= FIXED_SPLIT_IS_END and unique_dates[-1] >= FIXED_SPLIT_OOS_START:
        return {
            "split_mode": "fixed_calendar",
            "is_start": unique_dates[0],
            "is_end": FIXED_SPLIT_IS_END,
            "oos_start": FIXED_SPLIT_OOS_START,
            "oos_end": unique_dates[-1],
        }
    cut = max(1, int(len(unique_dates) * 0.70))
    cut = min(cut, len(unique_dates) - 1)
    return {
        "split_mode": "fallback_70_30",
        "is_start": unique_dates[0],
        "is_end": unique_dates[cut - 1],
        "oos_start": unique_dates[cut],
        "oos_end": unique_dates[-1],
    }


def _period_mask(events: pd.DataFrame, start_date: date, end_date: date) -> pd.Series:
    session_dates = pd.to_datetime(events["session_date"], errors="coerce").dt.date
    return session_dates.between(start_date, end_date)


def _metrics_with_scope(
    *,
    events: pd.DataFrame,
    start_date: date,
    end_date: date,
    estimated_cost_per_trade: float,
) -> dict[str, Any]:
    scoped = events.loc[_period_mask(events, start_date, end_date)].copy()
    return _compute_trade_metrics(scoped, estimated_cost_per_trade=estimated_cost_per_trade)


def minimum_trades_for_timeframe(timeframe: str, *, scope: str) -> int:
    normalized = str(timeframe)
    if scope == "is":
        return {"15min": 80, "30min": 60, "1H": 40}[normalized]
    return {"15min": 30, "30min": 20, "1H": 10}[normalized]


def _simplicity_score(family: str) -> float:
    if family == "raw_hybrid":
        return 1.0
    if family == "delay_only":
        return 0.85
    if family in {"stop_target_recalibration", "delay_stop_target"}:
        return 0.65
    if family == "delay_stop_zone_filter":
        return 0.45
    return 0.35


def _normalize(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.notna().sum() <= 1:
        return pd.Series(np.where(numeric.notna(), 1.0, np.nan), index=series.index, dtype=float)
    spread = numeric.max() - numeric.min()
    if spread == 0:
        return pd.Series(np.where(numeric.notna(), 1.0, np.nan), index=series.index, dtype=float)
    return (numeric - numeric.min()) / spread


def _yearly_metrics(
    events: pd.DataFrame,
    *,
    start_date: date,
    end_date: date,
    estimated_cost_per_trade: float,
) -> pd.DataFrame:
    scoped = events.loc[_period_mask(events, start_date, end_date)].copy()
    if scoped.empty:
        return pd.DataFrame(columns=["year", "trades", "net_pnl"])
    years = sorted(pd.to_datetime(scoped["session_date"], errors="coerce").dt.year.dropna().astype(int).unique())
    rows: list[dict[str, Any]] = []
    for year in years:
        year_slice = scoped.loc[pd.to_datetime(scoped["session_date"], errors="coerce").dt.year == year].copy()
        metrics = _compute_trade_metrics(year_slice, estimated_cost_per_trade=estimated_cost_per_trade)
        rows.append({"year": year, "trades": int(metrics["trades"]), "net_pnl": float(metrics["net_pnl"])})
    return pd.DataFrame(rows)


def _neighborhood_score(frame: pd.DataFrame, row: pd.Series) -> float:
    neighbors = frame.loc[
        (frame["family"] == row["family"])
        & (frame["filter_name"] == row["filter_name"])
        & (pd.to_numeric(frame["stop_multiplier"], errors="coerce").sub(float(row["stop_multiplier"])).abs() <= 0.26)
        & (pd.to_numeric(frame["target_multiplier"], errors="coerce").sub(float(row["target_multiplier"])).abs() <= 0.51)
        & (pd.to_numeric(frame["entry_delay_minutes"], errors="coerce").sub(int(row["entry_delay_minutes"])).abs() <= 15)
    ].copy()
    if row["filter_name"] == "require_no_stop_zone_touch_before_entry":
        if "stop_zone_fraction" not in neighbors.columns:
            return 0.0
        target_zone = safe_float(row.get("stop_zone_fraction"), np.nan)
        neighbors = neighbors.loc[pd.to_numeric(neighbors["stop_zone_fraction"], errors="coerce").sub(target_zone).abs() < 1e-9]
    if row["filter_name"] == "avoid_immediate_adverse_move":
        if "adverse_window_minutes" not in neighbors.columns or "max_adverse_ticks" not in neighbors.columns:
            return 0.0
        neighbors = neighbors.loc[
            (pd.to_numeric(neighbors["adverse_window_minutes"], errors="coerce") == safe_float(row.get("adverse_window_minutes"), np.nan))
            & (pd.to_numeric(neighbors["max_adverse_ticks"], errors="coerce") == safe_float(row.get("max_adverse_ticks"), np.nan))
        ]
    if neighbors.empty:
        return 0.0
    acceptable = (
        (pd.to_numeric(neighbors["net_pnl"], errors="coerce") > 0)
        & (pd.to_numeric(neighbors["profit_factor"], errors="coerce") >= 1.0)
    )
    return float(acceptable.mean())


def compute_robustness_scores(
    metrics_is: pd.DataFrame,
    events_by_config: dict[str, pd.DataFrame],
    *,
    split_info: dict[str, Any],
    estimated_cost_per_trade: float,
) -> pd.DataFrame:
    frame = metrics_is.copy().reset_index(drop=True)
    frame["normalized_net_pnl"] = _normalize(frame["net_pnl"])
    frame["normalized_profit_factor"] = _normalize(frame["profit_factor"].replace(np.inf, np.nan))
    frame["normalized_pnl_to_maxdd"] = _normalize(frame["pnl_to_maxdd"])
    timeframe_requirements = frame["signal_timeframe"].map(lambda value: minimum_trades_for_timeframe(str(value), scope="is"))
    frame["trade_count_score"] = pd.to_numeric(frame["trades"], errors="coerce").div(timeframe_requirements).clip(upper=1.0)
    frame["simplicity_score"] = frame["family"].map(_simplicity_score)

    temporal_scores: list[float] = []
    neighborhood_scores: list[float] = []
    penalties_list: list[float] = []
    admissible_list: list[bool] = []
    positive_years_list: list[int] = []
    years_with_trades_list: list[int] = []
    max_year_contribution_list: list[float] = []
    for _, row in frame.iterrows():
        config_id = str(row["config_id"])
        yearly = _yearly_metrics(
            events_by_config[config_id],
            start_date=split_info["is_start"],
            end_date=split_info["is_end"],
            estimated_cost_per_trade=estimated_cost_per_trade,
        )
        positive_years = int((pd.to_numeric(yearly["net_pnl"], errors="coerce") > 0).sum()) if not yearly.empty else 0
        years_with_trades = int((pd.to_numeric(yearly["trades"], errors="coerce") > 0).sum()) if not yearly.empty else 0
        pnl_by_year = pd.to_numeric(yearly["net_pnl"], errors="coerce").fillna(0.0)
        abs_total = float(pnl_by_year.abs().sum())
        max_year_contribution_pct = float(pnl_by_year.abs().max() / abs_total) if abs_total > 0 else 1.0
        temporal_stability = (
            0.5 * (positive_years / max(years_with_trades, 1))
            + 0.5 * max(0.0, 1.0 - max_year_contribution_pct)
        )
        neighborhood = _neighborhood_score(frame, row)
        penalties = 0.0
        if safe_float(row["skip_rate"], 0.0) > 0.70:
            penalties += 0.10
        if safe_float(row["avg_holding_minutes"], 0.0) > 180.0:
            penalties += 0.10
        if safe_float(row["trades"], 0.0) < minimum_trades_for_timeframe(str(row["signal_timeframe"]), scope="is"):
            penalties += 0.20
        if safe_float(row["profit_factor"], 0.0) < 1.05:
            penalties += 0.15
        if max_year_contribution_pct > 0.70:
            penalties += 0.15
        if safe_float(row["avg_trade"], 0.0) <= estimated_cost_per_trade * 1.5:
            penalties += 0.15
        if years_with_trades > 0 and safe_float(row["trades_per_year"], 0.0) < minimum_trades_for_timeframe(str(row["signal_timeframe"]), scope="is") / 2.0:
            penalties += 0.10

        admissible = bool(
            safe_float(row["net_pnl"], 0.0) > 0
            and safe_float(row["trades"], 0.0) >= minimum_trades_for_timeframe(str(row["signal_timeframe"]), scope="is")
            and safe_float(row["profit_factor"], 0.0) >= 1.10
            and safe_float(row["avg_trade"], 0.0) > estimated_cost_per_trade * 1.5
            and (positive_years >= 2 if years_with_trades >= 2 else positive_years >= 1)
        )

        temporal_scores.append(temporal_stability)
        neighborhood_scores.append(neighborhood)
        penalties_list.append(penalties)
        admissible_list.append(admissible)
        positive_years_list.append(positive_years)
        years_with_trades_list.append(years_with_trades)
        max_year_contribution_list.append(max_year_contribution_pct)

    frame["temporal_stability_score"] = temporal_scores
    frame["parameter_neighborhood_score"] = neighborhood_scores
    frame["penalties"] = penalties_list
    frame["admissible_is"] = admissible_list
    frame["positive_years_is"] = positive_years_list
    frame["years_with_trades_is"] = years_with_trades_list
    frame["max_year_contribution_pct"] = max_year_contribution_list
    frame["robust_score_is"] = (
        0.20 * frame["normalized_net_pnl"].fillna(0.0)
        + 0.20 * frame["normalized_profit_factor"].fillna(0.0)
        + 0.15 * frame["normalized_pnl_to_maxdd"].fillna(0.0)
        + 0.15 * frame["temporal_stability_score"].fillna(0.0)
        + 0.15 * frame["parameter_neighborhood_score"].fillna(0.0)
        + 0.10 * frame["trade_count_score"].fillna(0.0)
        + 0.05 * frame["simplicity_score"].fillna(0.0)
        - frame["penalties"].fillna(0.0)
    )
    return frame


def select_top_configs_is_only(robustness: pd.DataFrame, *, max_configs: int = 3) -> pd.DataFrame:
    ranked = robustness.sort_values(
        ["admissible_is", "robust_score_is", "net_pnl", "profit_factor", "skip_rate"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    selected: list[pd.Series] = []
    used_families: set[str] = set()

    simple = ranked.loc[
        ranked["admissible_is"]
        & ranked["family"].isin(["raw_hybrid", "delay_only", "stop_target_recalibration", "delay_stop_target"])
        & (pd.to_numeric(ranked["profit_factor"], errors="coerce") >= 1.05)
    ].head(1)
    if not simple.empty:
        selected.append(simple.iloc[0])
        used_families.add(str(simple.iloc[0]["family"]))

    for row in ranked.itertuples(index=False):
        if len(selected) >= max_configs:
            break
        row_series = pd.Series(row._asdict())
        if any(str(existing["config_id"]) == str(row_series["config_id"]) for existing in selected):
            continue
        if str(row_series["family"]) in used_families and len({str(item["family"]) for item in selected}) < max_configs:
            continue
        if safe_float(row_series["profit_factor"], 0.0) < 1.05:
            continue
        if safe_float(row_series["trades"], 0.0) < minimum_trades_for_timeframe(str(row_series["signal_timeframe"]), scope="is"):
            continue
        selected.append(row_series)
        used_families.add(str(row_series["family"]))
    if not selected:
        fallback = ranked.head(max_configs).copy()
        fallback["rank_is"] = np.arange(1, len(fallback) + 1, dtype=int)
        return fallback
    out = pd.DataFrame(selected).sort_values("robust_score_is", ascending=False).reset_index(drop=True)
    out["rank_is"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def build_selected_oos_report(
    selected_is: pd.DataFrame,
    metrics_oos: pd.DataFrame,
    metrics_full: pd.DataFrame,
) -> pd.DataFrame:
    if selected_is.empty:
        return pd.DataFrame()
    report = selected_is.merge(metrics_oos.add_suffix("_oos"), left_on="config_id", right_on="config_id_oos", how="left")
    report = report.merge(metrics_full.add_suffix("_full"), left_on="config_id", right_on="config_id_full", how="left")
    min_trades = report["signal_timeframe"].map(lambda value: minimum_trades_for_timeframe(str(value), scope="oos"))
    report["degradation_ratio"] = (
        pd.to_numeric(report["net_pnl_oos"], errors="coerce")
        / pd.to_numeric(report["net_pnl"], errors="coerce").replace(0.0, np.nan)
    )
    report["oos_pass"] = (
        (pd.to_numeric(report["net_pnl_oos"], errors="coerce") > 0)
        & (pd.to_numeric(report["profit_factor_oos"], errors="coerce") >= 1.05)
        & (pd.to_numeric(report["trades_oos"], errors="coerce") >= min_trades)
        & (pd.to_numeric(report["avg_trade_oos"], errors="coerce") > pd.to_numeric(report["estimated_cost_per_trade_oos"], errors="coerce"))
    )
    verdicts: list[str] = []
    for row in report.itertuples(index=False):
        if bool(row.oos_pass) and safe_float(row.profit_factor_oos, 0.0) >= 1.20:
            verdicts.append("strong_candidate")
        elif bool(row.oos_pass):
            verdicts.append("candidate")
        elif safe_float(row.net_pnl_oos, 0.0) > 0 or safe_float(row.profit_factor_oos, 0.0) >= 1.0:
            verdicts.append("weak_watchlist")
        else:
            verdicts.append("reject")
    report["verdict"] = verdicts
    report["params"] = report.apply(
        lambda row: json.dumps(
            {
                "stop_multiplier": safe_float(row["stop_multiplier"], np.nan),
                "target_multiplier": safe_float(row["target_multiplier"], np.nan),
                "entry_delay_minutes": int(safe_float(row["entry_delay_minutes"], 0.0)),
                "filter_name": row["filter_name"],
                "filter_params_json": row["filter_params_json"],
            },
            sort_keys=True,
        ),
        axis=1,
    )
    return report


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
    dynamic: list[WalkforwardFold] = []
    years = sorted({value.year for value in unique_dates})
    anchor = min_date
    for year in years:
        test_start = date(year, 1, 1)
        if test_start <= anchor:
            continue
        train_end = test_start - pd.Timedelta(days=1)
        actual_train_end = train_end.date() if isinstance(train_end, pd.Timestamp) else train_end
        actual_test_end = min(date(year, 12, 31), max_date)
        train_days = (actual_train_end - anchor).days + 1
        test_days = (actual_test_end - test_start).days + 1
        if train_days < 540 or test_days < 90:
            continue
        dynamic.append(
            WalkforwardFold(
                fold_id=f"fold_{len(dynamic) + 1}",
                train_start=anchor,
                train_end=actual_train_end,
                test_start=test_start,
                test_end=actual_test_end,
            )
        )
    if not dynamic:
        raise ValueError("Unable to build walk-forward folds with minimum train/test length.")
    return dynamic


def compute_fold_train_ranking(
    *,
    symbol: str,
    signal_timeframe: str,
    fold: WalkforwardFold,
    config_frame: pd.DataFrame,
    events_by_config: dict[str, pd.DataFrame],
    estimated_cost_per_trade: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, config_row in config_frame.iterrows():
        config_id = str(config_row["config_id"])
        events = events_by_config[config_id].loc[_period_mask(events_by_config[config_id], fold.train_start, fold.train_end)].copy()
        metrics = _compute_trade_metrics(events, estimated_cost_per_trade=estimated_cost_per_trade)
        yearly = _yearly_metrics(
            events_by_config[config_id],
            start_date=fold.train_start,
            end_date=fold.train_end,
            estimated_cost_per_trade=estimated_cost_per_trade,
        )
        positive_years = int((pd.to_numeric(yearly["net_pnl"], errors="coerce") > 0).sum()) if not yearly.empty else 0
        years_with_trades = int((pd.to_numeric(yearly["trades"], errors="coerce") > 0).sum()) if not yearly.empty else 0
        pnl_by_year = pd.to_numeric(yearly["net_pnl"], errors="coerce").fillna(0.0)
        abs_total = float(pnl_by_year.abs().sum())
        max_year_contribution_pct = float(pnl_by_year.abs().max() / abs_total) if abs_total > 0 else 1.0
        rows.append(
            {
                "symbol": symbol,
                "signal_timeframe": signal_timeframe,
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
                "positive_years_train": positive_years,
                "years_with_trades_train": years_with_trades,
                "max_year_contribution_pct": max_year_contribution_pct,
            }
        )
    frame = pd.DataFrame(rows)
    frame["normalized_net_pnl"] = _normalize(frame["train_net_pnl"])
    frame["normalized_profit_factor"] = _normalize(frame["train_profit_factor"].replace(np.inf, np.nan))
    frame["normalized_pnl_to_maxdd"] = _normalize(frame["train_pnl_to_maxdd"])
    frame["trade_count_score"] = pd.to_numeric(frame["train_trades"], errors="coerce").div(
        minimum_trades_for_timeframe(signal_timeframe, scope="is")
    ).clip(upper=1.0)
    frame["simplicity_score"] = frame["family"].map(_simplicity_score)

    temporal_scores: list[float] = []
    neighborhood_scores: list[float] = []
    penalties_list: list[float] = []
    for _, row in frame.iterrows():
        temporal = 0.5 * (safe_float(row["positive_years_train"], 0.0) / max(safe_float(row["years_with_trades_train"], 1.0), 1.0)) + 0.5 * max(
            0.0, 1.0 - safe_float(row["max_year_contribution_pct"], 1.0)
        )
        neighborhood = _neighborhood_score(
            frame.rename(
                columns={
                    "train_net_pnl": "net_pnl",
                    "train_profit_factor": "profit_factor",
                }
            ),
            pd.Series(
                {
                    "family": row["family"],
                    "filter_name": row["filter_name"],
                    "stop_multiplier": row["stop_multiplier"],
                    "target_multiplier": row["target_multiplier"],
                    "entry_delay_minutes": row["entry_delay_minutes"],
                    "stop_zone_fraction": row["stop_zone_fraction"],
                }
            ),
        )
        penalties = 0.0
        if safe_float(row["train_trades"], 0.0) < max(30, minimum_trades_for_timeframe(signal_timeframe, scope="is") // 2):
            penalties += 0.20
        if safe_float(row["train_profit_factor"], 0.0) < 1.05:
            penalties += 0.15
        if safe_float(row["max_year_contribution_pct"], 1.0) > 0.70:
            penalties += 0.15
        if safe_float(row["train_skip_rate"], 0.0) > 0.70:
            penalties += 0.10
        if safe_float(row["train_avg_trade"], 0.0) <= estimated_cost_per_trade * 1.5:
            penalties += 0.15
        temporal_scores.append(temporal)
        neighborhood_scores.append(neighborhood)
        penalties_list.append(penalties)
    frame["temporal_stability"] = temporal_scores
    frame["neighborhood_stability"] = neighborhood_scores
    frame["penalties"] = penalties_list
    frame["train_robust_score"] = (
        0.25 * frame["normalized_net_pnl"].fillna(0.0)
        + 0.20 * frame["normalized_profit_factor"].fillna(0.0)
        + 0.15 * frame["normalized_pnl_to_maxdd"].fillna(0.0)
        + 0.15 * frame["temporal_stability"].fillna(0.0)
        + 0.15 * frame["neighborhood_stability"].fillna(0.0)
        + 0.10 * frame["trade_count_score"].fillna(0.0)
        - frame["penalties"].fillna(0.0)
    )
    frame = frame.sort_values(
        ["train_robust_score", "train_net_pnl", "train_profit_factor", "train_skip_rate"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    frame["selected_in_fold"] = False
    if not frame.empty:
        frame.loc[0, "selected_in_fold"] = True
    return frame


def _wfa_verdict(summary_row: pd.Series) -> str:
    if safe_float(summary_row["total_test_net_pnl"], 0.0) <= 0 or int(summary_row["positive_folds"]) < 3:
        return "reject"
    if safe_float(summary_row["test_profit_factor"], 0.0) < 1.10 or safe_float(summary_row["pass_rate"], 0.0) < 0.60:
        return "weak_watchlist"
    if (
        safe_float(summary_row["total_test_net_pnl"], 0.0) > 0
        and safe_float(summary_row["test_profit_factor"], 0.0) >= 1.20
        and int(summary_row["positive_folds"]) >= 4
        and safe_float(summary_row["train_score_test_corr"], np.nan) >= 0
        and safe_float(summary_row["improvement_vs_raw_hybrid"], 0.0) > 0
    ):
        return "strong_candidate"
    if safe_float(summary_row["improvement_vs_raw_hybrid"], 0.0) > 0:
        return "candidate"
    return "weak_watchlist"


def _evaluate_symbol_timeframe(
    *,
    symbol: str,
    signal_timeframe: str,
    execution_timeframe: str,
    output_dir: Path,
    raw_minute_df_override: pd.DataFrame | None = None,
) -> dict[str, Any]:
    raw_path = latest_path_for_symbol(symbol)
    raw_minute_df = raw_minute_df_override.copy() if raw_minute_df_override is not None else load_symbol_data(symbol, input_paths={symbol: raw_path})
    raw_minute_df["timestamp"] = pd.to_datetime(raw_minute_df["timestamp"], errors="coerce")
    minute_df = extract_rth(raw_minute_df.copy())
    minute_df["timestamp"] = pd.to_datetime(minute_df["timestamp"], errors="coerce")
    minute_df["session_date"] = minute_df["timestamp"].dt.date
    bars = resample_rth_timeframe(raw_minute_df, signal_timeframe)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")
    bars["session_date"] = bars["timestamp"].dt.date

    base_variant, used_fallback, desired_name = _resolve_seed_variant(symbol)
    signal_variant = _clone_variant_for_timeframe(base_variant, signal_timeframe)
    features = prepare_volume_climax_pullback_v2_features(bars)
    signal_df = build_volume_climax_pullback_v2_signal_frame(features, signal_variant)
    split_info = _infer_split_mode(minute_df["session_date"].dropna().tolist())

    config_universe = build_config_universe_for_symbol_timeframe(
        symbol=symbol,
        signal_timeframe=signal_timeframe,
        execution_timeframe=execution_timeframe,
        base_signal_variant=signal_variant.name,
    )
    config_frame = _config_universe_frame(config_universe)
    config_frame["stop_zone_fraction"] = config_frame["filter_params_json"].map(
        lambda value: safe_float(json.loads(value).get("stop_zone_fraction"), np.nan) if value and value != "{}" else np.nan
    )
    config_frame["adverse_window_minutes"] = config_frame["filter_params_json"].map(
        lambda value: safe_float(json.loads(value).get("adverse_window_minutes"), np.nan) if value and value != "{}" else np.nan
    )
    config_frame["max_adverse_ticks"] = config_frame["filter_params_json"].map(
        lambda value: safe_float(json.loads(value).get("max_adverse_ticks"), np.nan) if value and value != "{}" else np.nan
    )

    execution_model, instrument = build_execution_model_for_profile(symbol=symbol, profile_name="repo_realistic")
    estimated_cost_per_trade = float(execution_model.round_trip_fees(quantity=1))
    signal_bar_minutes = timeframe_to_minutes(signal_timeframe)
    events_by_config: dict[str, pd.DataFrame] = {}
    metrics_rows: list[dict[str, Any]] = []

    for config in config_universe:
        events = _simulate_config(
            config=config.to_intrabar_config(),
            signal_df=signal_df,
            minute_df=minute_df,
            variant=signal_variant,
            execution_model=execution_model,
            point_value_usd=float(instrument.point_value_usd),
            tick_size=float(instrument.tick_size),
            signal_bar_minutes=signal_bar_minutes,
        )
        events["symbol"] = symbol
        events["signal_timeframe"] = signal_timeframe
        events["config_id"] = config.config_id
        events_by_config[config.config_id] = events
        is_metrics = _metrics_with_scope(
            events=events,
            start_date=split_info["is_start"],
            end_date=split_info["is_end"],
            estimated_cost_per_trade=estimated_cost_per_trade,
        )
        oos_metrics = _metrics_with_scope(
            events=events,
            start_date=split_info["oos_start"],
            end_date=split_info["oos_end"],
            estimated_cost_per_trade=estimated_cost_per_trade,
        )
        full_metrics = _compute_trade_metrics(events, estimated_cost_per_trade=estimated_cost_per_trade)
        for scope, metrics in (("is", is_metrics), ("oos", oos_metrics), ("full", full_metrics)):
            metrics_rows.append(
                {
                    "symbol": symbol,
                    "signal_timeframe": signal_timeframe,
                    "config_id": config.config_id,
                    "family": config.family,
                    "scope": scope,
                    **metrics,
                }
            )

    metrics_frame = pd.DataFrame(metrics_rows).merge(config_frame, on=["symbol", "signal_timeframe", "config_id", "family"], how="left")
    metrics_frame["trades_per_year"] = pd.to_numeric(metrics_frame["trades"], errors="coerce") / max(
        ((pd.Timestamp(split_info["oos_end"]) - pd.Timestamp(split_info["is_start"])).days + 1) / 365.25,
        0.25,
    )
    metrics_frame["avg_trade_to_cost"] = pd.to_numeric(metrics_frame["avg_trade"], errors="coerce") / pd.to_numeric(
        metrics_frame["estimated_cost_per_trade"], errors="coerce"
    ).replace(0.0, np.nan)
    metrics_is = metrics_frame.loc[metrics_frame["scope"] == "is"].drop(columns=["scope"]).reset_index(drop=True)
    metrics_oos = metrics_frame.loc[metrics_frame["scope"] == "oos"].drop(columns=["scope"]).reset_index(drop=True)
    metrics_full = metrics_frame.loc[metrics_frame["scope"] == "full"].drop(columns=["scope"]).reset_index(drop=True)

    robustness = compute_robustness_scores(metrics_is, events_by_config, split_info=split_info, estimated_cost_per_trade=estimated_cost_per_trade)
    selected_is = select_top_configs_is_only(robustness, max_configs=3)
    selected_oos_report = build_selected_oos_report(selected_is, metrics_oos, metrics_full)

    folds = build_walkforward_folds(minute_df["session_date"].dropna().tolist())
    fold_train_rankings: list[pd.DataFrame] = []
    fold_selected_rows: list[dict[str, Any]] = []
    stitched_rows: list[pd.DataFrame] = []
    raw_benchmark_rows: list[pd.DataFrame] = []
    for fold in folds:
        ranking = compute_fold_train_ranking(
            symbol=symbol,
            signal_timeframe=signal_timeframe,
            fold=fold,
            config_frame=config_frame,
            events_by_config=events_by_config,
            estimated_cost_per_trade=estimated_cost_per_trade,
        )
        fold_train_rankings.append(ranking)
        winner = ranking.iloc[0]
        selected_config_id = str(winner["config_id"])
        test_events = events_by_config[selected_config_id].loc[_period_mask(events_by_config[selected_config_id], fold.test_start, fold.test_end)].copy()
        raw_events = events_by_config[
            config_frame.loc[config_frame["family"] == "raw_hybrid", "config_id"].iloc[0]
        ].loc[_period_mask(events_by_config[config_frame.loc[config_frame["family"] == "raw_hybrid", "config_id"].iloc[0]], fold.test_start, fold.test_end)].copy()
        test_metrics = _compute_trade_metrics(test_events, estimated_cost_per_trade=estimated_cost_per_trade)
        fold_selected_rows.append(
            {
                "symbol": symbol,
                "signal_timeframe": signal_timeframe,
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
                    and safe_float(test_metrics["trades"], 0.0) >= minimum_trades_for_timeframe(signal_timeframe, scope="oos")
                    and safe_float(test_metrics["avg_trade"], 0.0) > estimated_cost_per_trade
                ),
            }
        )
        test_events["fold_id"] = fold.fold_id
        stitched_rows.append(test_events)
        raw_events["fold_id"] = fold.fold_id
        raw_benchmark_rows.append(raw_events)

    fold_train_ranking = pd.concat(fold_train_rankings, ignore_index=True) if fold_train_rankings else pd.DataFrame()
    fold_selected_test_results = pd.DataFrame(fold_selected_rows)
    stitched_trades = pd.concat(stitched_rows, ignore_index=True) if stitched_rows else pd.DataFrame()
    benchmark_same_windows = pd.concat(raw_benchmark_rows, ignore_index=True) if raw_benchmark_rows else pd.DataFrame()
    stitched_metrics = _compute_trade_metrics(stitched_trades, estimated_cost_per_trade=estimated_cost_per_trade)
    benchmark_metrics_same_windows = _compute_trade_metrics(benchmark_same_windows, estimated_cost_per_trade=estimated_cost_per_trade)
    train_score_corr = (
        pd.to_numeric(fold_selected_test_results["train_robust_score"], errors="coerce").corr(
            pd.to_numeric(fold_selected_test_results["test_net_pnl"], errors="coerce")
        )
        if len(fold_selected_test_results) >= 2
        else np.nan
    )
    walkforward_summary = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "signal_timeframe": signal_timeframe,
                "total_test_trades": int(stitched_metrics["trades"]),
                "total_test_net_pnl": float(stitched_metrics["net_pnl"]),
                "test_profit_factor": float(stitched_metrics["profit_factor"]),
                "test_winrate": float(stitched_metrics["winrate"]),
                "avg_trade": float(stitched_metrics["avg_trade"]),
                "max_drawdown": float(stitched_metrics["max_drawdown"]),
                "pnl_to_maxdd": float(stitched_metrics["pnl_to_maxdd"]) if pd.notna(stitched_metrics["pnl_to_maxdd"]) else np.nan,
                "number_of_folds": int(len(folds)),
                "positive_folds": int((pd.to_numeric(fold_selected_test_results["test_net_pnl"], errors="coerce") > 0).sum()),
                "pass_rate": float(pd.to_numeric(fold_selected_test_results["test_pass"], errors="coerce").fillna(False).astype(bool).mean()) if not fold_selected_test_results.empty else 0.0,
                "selected_family_counts": json.dumps(fold_selected_test_results["selected_family"].value_counts().to_dict(), sort_keys=True) if not fold_selected_test_results.empty else "{}",
                "train_score_test_corr": train_score_corr,
                "benchmark_raw_hybrid_net_pnl_same_windows": float(benchmark_metrics_same_windows["net_pnl"]),
                "improvement_vs_raw_hybrid": float(stitched_metrics["net_pnl"] - benchmark_metrics_same_windows["net_pnl"]),
            }
        ]
    )
    walkforward_summary["verdict"] = walkforward_summary.apply(_wfa_verdict, axis=1)

    selected_daily_rows: list[pd.DataFrame] = []
    selected_is_daily_rows: list[pd.DataFrame] = []
    for _, row in selected_oos_report.iterrows():
        config_id = str(row["config_id"])
        oos_events = events_by_config[config_id].loc[_period_mask(events_by_config[config_id], split_info["oos_start"], split_info["oos_end"])].copy()
        is_events = events_by_config[config_id].loc[_period_mask(events_by_config[config_id], split_info["is_start"], split_info["is_end"])].copy()
        oos_daily = _daily_returns(oos_events)
        if not oos_daily.empty:
            oos_daily["sleeve_id"] = f"{symbol}_{signal_timeframe}_{config_id}"
            oos_daily["symbol"] = symbol
            oos_daily["signal_timeframe"] = signal_timeframe
            oos_daily["config_id"] = config_id
            oos_daily["selection_basis"] = "is_only"
            selected_daily_rows.append(oos_daily)
        is_daily = _daily_returns(is_events)
        if not is_daily.empty:
            is_daily["sleeve_id"] = f"{symbol}_{signal_timeframe}_{config_id}"
            is_daily["symbol"] = symbol
            is_daily["signal_timeframe"] = signal_timeframe
            is_daily["config_id"] = config_id
            selected_is_daily_rows.append(is_daily)

    stitched_daily = _daily_returns(stitched_trades)
    if not stitched_daily.empty:
        stitched_daily["sleeve_id"] = f"{symbol}_{signal_timeframe}_wfa"
        stitched_daily["symbol"] = symbol
        stitched_daily["signal_timeframe"] = signal_timeframe
        stitched_daily["config_id"] = "walkforward_stitched"
        stitched_daily["selection_basis"] = "walkforward_train_only"

    missing_days = []
    if not minute_df.empty:
        all_bdays = pd.date_range(minute_df["timestamp"].min().date(), minute_df["timestamp"].max().date(), freq="B")
        observed = set(pd.to_datetime(minute_df["session_date"]).dropna().astype(str))
        missing_days = [str(day.date()) for day in all_bdays if str(day.date()) not in observed][:50]
    data_audit = {
        "symbol": symbol,
        "signal_timeframe": signal_timeframe,
        "execution_timeframe": execution_timeframe,
        "source_path": str(raw_path),
        "first_timestamp": str(raw_minute_df["timestamp"].min()) if not raw_minute_df.empty else None,
        "last_timestamp": str(raw_minute_df["timestamp"].max()) if not raw_minute_df.empty else None,
        "number_of_1min_rows": int(len(raw_minute_df)),
        "number_of_signal_rows": int(len(signal_df)),
        "rth_rows": int(len(minute_df)),
        "timezone": DEFAULT_TIMEZONE,
        "session_convention": "RTH 09:30-16:00 inclusive local timestamp",
        "missing_days": missing_days,
        "split_mode": split_info["split_mode"],
        "variant_name": signal_variant.name,
        "variant_seed_fallback": bool(used_fallback),
        "variant_seed_requested": desired_name,
    }
    audit_path = output_dir / f"data_audit_{symbol}_{signal_timeframe}.json"
    audit_path.write_text(json.dumps(data_audit, indent=2), encoding="utf-8")

    return {
        "symbol": symbol,
        "signal_timeframe": signal_timeframe,
        "split_info": split_info,
        "data_audit": data_audit,
        "config_frame": config_frame,
        "metrics_is": metrics_is,
        "metrics_oos": metrics_oos,
        "metrics_full": metrics_full,
        "robustness": robustness,
        "selected_is": selected_is,
        "selected_oos_report": selected_oos_report,
        "walkforward_folds": pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "signal_timeframe": signal_timeframe,
                    "fold_id": fold.fold_id,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "train_days": fold.train_days,
                    "test_days": fold.test_days,
                }
                for fold in folds
            ]
        ),
        "fold_train_ranking": fold_train_ranking,
        "fold_selected_test_results": fold_selected_test_results,
        "walkforward_stitched_trades": stitched_trades,
        "walkforward_stitched_daily_returns": stitched_daily,
        "walkforward_summary": walkforward_summary,
        "selected_oos_daily": pd.concat(selected_daily_rows, ignore_index=True) if selected_daily_rows else pd.DataFrame(),
        "selected_is_daily": pd.concat(selected_is_daily_rows, ignore_index=True) if selected_is_daily_rows else pd.DataFrame(),
    }


def _portfolio_metrics_from_daily(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty:
        return {
            "net_pnl": 0.0,
            "annualized_pnl": np.nan,
            "volatility_daily_pnl": np.nan,
            "sharpe_like_daily_pnl": np.nan,
            "max_drawdown": 0.0,
            "pnl_to_maxdd": np.nan,
            "positive_days_pct": np.nan,
            "worst_day": np.nan,
            "worst_month": np.nan,
        }
    returns = pd.to_numeric(daily["portfolio_pnl"], errors="coerce").fillna(0.0)
    equity = returns.cumsum()
    drawdown = equity - equity.cummax()
    months = pd.to_datetime(daily["session_date"]).dt.to_period("M").astype(str)
    monthly = returns.groupby(months).sum()
    annualized = float(returns.mean() * 252.0) if len(returns) > 0 else np.nan
    vol = float(returns.std(ddof=0)) if len(returns) > 1 else np.nan
    sharpe = float(np.sqrt(252.0) * returns.mean() / returns.std(ddof=0)) if len(returns) > 1 and returns.std(ddof=0) > 0 else np.nan
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0
    return {
        "net_pnl": float(returns.sum()),
        "annualized_pnl": annualized,
        "volatility_daily_pnl": vol,
        "sharpe_like_daily_pnl": sharpe,
        "max_drawdown": max_dd,
        "pnl_to_maxdd": float(returns.sum() / abs(max_dd)) if max_dd < 0 else np.nan,
        "positive_days_pct": float((returns > 0).mean()),
        "worst_day": float(returns.min()) if len(returns) > 0 else np.nan,
        "worst_month": float(monthly.min()) if not monthly.empty else np.nan,
    }


def _build_portfolio_daily(
    sleeves_daily: pd.DataFrame,
    *,
    portfolio_name: str,
    weights: dict[str, float],
) -> pd.DataFrame:
    if sleeves_daily.empty or not weights:
        return pd.DataFrame(columns=["session_date", "portfolio_name", "portfolio_pnl", "equity"])
    frame = sleeves_daily.copy()
    frame["weight"] = frame["sleeve_id"].map(weights).fillna(0.0)
    grouped = (
        frame.groupby("session_date", as_index=False)
        .apply(lambda part: pd.Series({"portfolio_pnl": float((pd.to_numeric(part["daily_pnl"], errors="coerce") * pd.to_numeric(part["weight"], errors="coerce")).sum())}))
        .reset_index(drop=True)
    )
    grouped["portfolio_name"] = portfolio_name
    grouped["equity"] = pd.to_numeric(grouped["portfolio_pnl"], errors="coerce").cumsum()
    return grouped


def _equal_weights(sleeve_ids: Sequence[str]) -> dict[str, float]:
    unique = list(dict.fromkeys(sleeve_ids))
    if not unique:
        return {}
    weight = 1.0 / len(unique)
    return {sleeve_id: weight for sleeve_id in unique}


def _inverse_vol_weights(is_daily: pd.DataFrame, sleeve_ids: Sequence[str]) -> dict[str, float]:
    unique = list(dict.fromkeys(sleeve_ids))
    if not unique:
        return {}
    vol_map: dict[str, float] = {}
    for sleeve_id in unique:
        series = pd.to_numeric(is_daily.loc[is_daily["sleeve_id"] == sleeve_id, "daily_pnl"], errors="coerce")
        vol = float(series.std(ddof=0)) if len(series) > 1 and series.std(ddof=0) > 0 else np.nan
        vol_map[sleeve_id] = 1.0 / vol if np.isfinite(vol) and vol > 0 else 0.0
    total = sum(vol_map.values())
    if total <= 0:
        return _equal_weights(unique)
    return {sleeve_id: value / total for sleeve_id, value in vol_map.items()}


def _capped_equal_weights(sleeves_meta: pd.DataFrame, *, cap_per_asset: float) -> dict[str, float]:
    if sleeves_meta.empty:
        return {}
    assets = sorted(sleeves_meta["symbol"].astype(str).unique())
    if not assets:
        return {}
    asset_weight = min(cap_per_asset, 1.0 / len(assets))
    weights: dict[str, float] = {}
    remainder = 1.0 - asset_weight * len(assets)
    if remainder > 0:
        asset_weight += remainder / len(assets)
    for asset in assets:
        asset_rows = sleeves_meta.loc[sleeves_meta["symbol"] == asset].copy()
        if asset_rows.empty:
            continue
        per_sleeve = asset_weight / len(asset_rows)
        for sleeve_id in asset_rows["sleeve_id"]:
            weights[str(sleeve_id)] = per_sleeve
    return weights


def _build_asset_contributions(portfolio_daily_source: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    if portfolio_daily_source.empty or not weights:
        return pd.DataFrame(columns=["asset", "contribution_pnl"])
    frame = portfolio_daily_source.copy()
    frame["weight"] = frame["sleeve_id"].map(weights).fillna(0.0)
    frame["weighted_pnl"] = pd.to_numeric(frame["daily_pnl"], errors="coerce") * pd.to_numeric(frame["weight"], errors="coerce")
    return frame.groupby("symbol", as_index=False)["weighted_pnl"].sum().rename(columns={"symbol": "asset", "weighted_pnl": "contribution_pnl"})


def _build_portfolio_outputs(
    *,
    selected_oos_daily: pd.DataFrame,
    selected_is_daily: pd.DataFrame,
    walkforward_daily: pd.DataFrame,
    selected_oos_report: pd.DataFrame,
    walkforward_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oos_rows: list[dict[str, Any]] = []
    wfa_rows: list[dict[str, Any]] = []
    portfolio_daily_rows: list[pd.DataFrame] = []
    contribution_rows: list[pd.DataFrame] = []

    allowed_selected_ids = (
        set(
            selected_oos_report.apply(
                lambda row: f"{row['symbol']}_{row['signal_timeframe']}_{row['config_id']}",
                axis=1,
            )
        )
        if not selected_oos_report.empty
        else set()
    )
    if allowed_selected_ids and "sleeve_id" in selected_oos_daily.columns:
        selected_oos_daily = selected_oos_daily.loc[selected_oos_daily["sleeve_id"].isin(allowed_selected_ids)].copy()
    elif "sleeve_id" not in selected_oos_daily.columns:
        selected_oos_daily = pd.DataFrame()

    if allowed_selected_ids and "sleeve_id" in selected_is_daily.columns:
        selected_is_daily = selected_is_daily.loc[selected_is_daily["sleeve_id"].isin(allowed_selected_ids)].copy()
    elif "sleeve_id" not in selected_is_daily.columns:
        selected_is_daily = pd.DataFrame()

    selected_meta = (
        selected_oos_daily[["sleeve_id", "symbol", "signal_timeframe", "config_id"]].drop_duplicates().reset_index(drop=True)
        if not selected_oos_daily.empty
        else pd.DataFrame(columns=["sleeve_id", "symbol", "signal_timeframe", "config_id"])
    )
    all_candidate_ids = list(selected_meta["sleeve_id"]) if not selected_meta.empty else []
    portfolios_oos = [
        ("equal_weight_all_candidates", all_candidate_ids, _equal_weights(all_candidate_ids), "train_only_preselected", False),
        ("inverse_vol_daily_pnl", all_candidate_ids, _inverse_vol_weights(selected_is_daily, all_candidate_ids), "train_only_preselected", False),
        ("capped_equal_weight", all_candidate_ids, _capped_equal_weights(selected_meta, cap_per_asset=0.40), "train_only_preselected", False),
    ]
    watchlist_ids = (
        selected_oos_report.loc[selected_oos_report["verdict"].isin(["weak_watchlist", "candidate", "strong_candidate"])]
        .apply(lambda row: f"{row['symbol']}_{row['signal_timeframe']}_{row['config_id']}", axis=1)
        .tolist()
        if not selected_oos_report.empty
        else []
    )
    portfolios_oos.append(
        ("conservative_watchlist_book", watchlist_ids, _equal_weights(watchlist_ids), "diagnostic_post_oos_filter", True)
    )

    for portfolio_name, sleeve_ids, weights, selection_basis, diagnostic_flag in portfolios_oos:
        source = selected_oos_daily.loc[selected_oos_daily["sleeve_id"].isin(sleeve_ids)].copy()
        daily = _build_portfolio_daily(source, portfolio_name=portfolio_name, weights=weights)
        metrics = _portfolio_metrics_from_daily(daily)
        contributions = _build_asset_contributions(source, weights)
        contribution_rows.append(contributions.assign(portfolio_name=portfolio_name, selection_basis=selection_basis))
        oos_rows.append(
            {
                "portfolio_name": portfolio_name,
                "selection_basis": selection_basis,
                "diagnostic_post_oos_filter": diagnostic_flag,
                **metrics,
                "asset_contribution": json.dumps(
                    {row["asset"]: float(row["contribution_pnl"]) for _, row in contributions.iterrows()},
                    sort_keys=True,
                ),
            }
        )
        if not daily.empty:
            daily["selection_basis"] = selection_basis
            daily["diagnostic_post_oos_filter"] = diagnostic_flag
            portfolio_daily_rows.append(daily)

    allowed_wfa_ids = (
        set(
            walkforward_summary.apply(
                lambda row: f"{row['symbol']}_{row['signal_timeframe']}_wfa",
                axis=1,
            )
        )
        if not walkforward_summary.empty
        else set()
    )
    if allowed_wfa_ids and "sleeve_id" in walkforward_daily.columns:
        walkforward_daily = walkforward_daily.loc[walkforward_daily["sleeve_id"].isin(allowed_wfa_ids)].copy()
    elif "sleeve_id" not in walkforward_daily.columns:
        walkforward_daily = pd.DataFrame()

    wfa_meta = (
        walkforward_daily[["sleeve_id", "symbol", "signal_timeframe", "config_id"]].drop_duplicates().reset_index(drop=True)
        if not walkforward_daily.empty
        else pd.DataFrame(columns=["sleeve_id", "symbol", "signal_timeframe", "config_id"])
    )
    all_wfa_ids = list(wfa_meta["sleeve_id"]) if not wfa_meta.empty else []
    positive_wfa_ids = (
        walkforward_summary.loc[walkforward_summary["verdict"].isin(["weak_watchlist", "candidate", "strong_candidate"])]
        .apply(lambda row: f"{row['symbol']}_{row['signal_timeframe']}_wfa", axis=1)
        .tolist()
        if not walkforward_summary.empty
        else []
    )
    portfolios_wfa = [
        ("walkforward_equal_weight_all", all_wfa_ids, _equal_weights(all_wfa_ids), "walkforward_train_only", False),
        ("equal_weight_positive_wfa_only", positive_wfa_ids, _equal_weights(positive_wfa_ids), "diagnostic_post_wfa_filter", True),
    ]
    for portfolio_name, sleeve_ids, weights, selection_basis, diagnostic_flag in portfolios_wfa:
        if "sleeve_id" in walkforward_daily.columns:
            source = walkforward_daily.loc[walkforward_daily["sleeve_id"].isin(sleeve_ids)].copy()
        else:
            source = pd.DataFrame()
        daily = _build_portfolio_daily(source, portfolio_name=portfolio_name, weights=weights)
        metrics = _portfolio_metrics_from_daily(daily)
        contributions = _build_asset_contributions(source, weights)
        contribution_rows.append(contributions.assign(portfolio_name=portfolio_name, selection_basis=selection_basis))
        wfa_rows.append(
            {
                "portfolio_name": portfolio_name,
                "selection_basis": selection_basis,
                "diagnostic_post_wfa_filter": diagnostic_flag,
                **metrics,
                "asset_contribution": json.dumps(
                    {row["asset"]: float(row["contribution_pnl"]) for _, row in contributions.iterrows()},
                    sort_keys=True,
                ),
            }
        )
        if not daily.empty:
            daily["selection_basis"] = selection_basis
            daily["diagnostic_post_oos_filter"] = diagnostic_flag
            portfolio_daily_rows.append(daily)

    portfolio_daily = pd.concat(portfolio_daily_rows, ignore_index=True) if portfolio_daily_rows else pd.DataFrame()
    portfolio_oos_summary = pd.DataFrame(oos_rows)
    portfolio_walkforward_summary = pd.DataFrame(wfa_rows)
    portfolio_asset_contribution = pd.concat(contribution_rows, ignore_index=True) if contribution_rows else pd.DataFrame()

    if selected_oos_daily.empty:
        correlation = pd.DataFrame()
    else:
        pivot = selected_oos_daily.pivot_table(index="session_date", columns="sleeve_id", values="daily_pnl", aggfunc="sum").fillna(0.0)
        correlation = pivot.corr()
    return portfolio_oos_summary, portfolio_walkforward_summary, portfolio_daily, correlation, portfolio_asset_contribution


def build_timeframe_summary(
    metrics_is: pd.DataFrame,
    metrics_oos: pd.DataFrame,
    selected_oos_report: pd.DataFrame,
    walkforward_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for timeframe in sorted(metrics_is["signal_timeframe"].astype(str).unique()):
        is_slice = metrics_is.loc[metrics_is["signal_timeframe"] == timeframe].copy()
        oos_slice = metrics_oos.loc[metrics_oos["signal_timeframe"] == timeframe].copy()
        selected_slice = selected_oos_report.loc[selected_oos_report["signal_timeframe"] == timeframe].copy()
        wfa_slice = walkforward_summary.loc[walkforward_summary["signal_timeframe"] == timeframe].copy()
        if wfa_slice.empty:
            verdict = "reject"
            best_symbol = None
            median_wfa = np.nan
        else:
            best_idx = pd.to_numeric(wfa_slice["total_test_net_pnl"], errors="coerce").idxmax()
            best_symbol = wfa_slice.loc[best_idx, "symbol"]
            verdict_rank = {"strong_candidate": 3, "candidate": 2, "weak_watchlist": 1, "reject": 0}
            verdict = max(wfa_slice["verdict"], key=lambda value: verdict_rank.get(str(value), -1))
            median_wfa = float(pd.to_numeric(wfa_slice["total_test_net_pnl"], errors="coerce").median())
        rows.append(
            {
                "signal_timeframe": timeframe,
                "total_configs": int(len(is_slice)),
                "median_is_pf": float(pd.to_numeric(is_slice["profit_factor"], errors="coerce").median()) if not is_slice.empty else np.nan,
                "median_oos_pf": float(pd.to_numeric(oos_slice["profit_factor"], errors="coerce").median()) if not oos_slice.empty else np.nan,
                "pct_configs_is_positive": float((pd.to_numeric(is_slice["net_pnl"], errors="coerce") > 0).mean()) if not is_slice.empty else np.nan,
                "pct_configs_oos_positive": float((pd.to_numeric(oos_slice["net_pnl"], errors="coerce") > 0).mean()) if not oos_slice.empty else np.nan,
                "pct_selected_configs_oos_pass": float(pd.to_numeric(selected_slice["oos_pass"], errors="coerce").fillna(False).astype(bool).mean()) if not selected_slice.empty else np.nan,
                "wfa_stitched_median_pnl": median_wfa,
                "best_symbol": best_symbol,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def build_asset_summary(
    selected_oos_report: pd.DataFrame,
    walkforward_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol in sorted(set(selected_oos_report["symbol"].astype(str)).union(set(walkforward_summary["symbol"].astype(str)))):
        selected_slice = selected_oos_report.loc[selected_oos_report["symbol"] == symbol].copy()
        wfa_slice = walkforward_summary.loc[walkforward_summary["symbol"] == symbol].copy()
        best_is_timeframe = (
            selected_slice.sort_values(["robust_score_is", "net_pnl"], ascending=[False, False]).iloc[0]["signal_timeframe"]
            if not selected_slice.empty
            else None
        )
        best_wfa_timeframe = (
            wfa_slice.sort_values(["total_test_net_pnl", "test_profit_factor"], ascending=[False, False]).iloc[0]["signal_timeframe"]
            if not wfa_slice.empty
            else None
        )
        if wfa_slice.empty:
            verdict = "reject"
            stitched_net = np.nan
            stitched_pf = np.nan
            positive_folds = 0
            failure_mode = "inconclusive"
        else:
            best_wfa = wfa_slice.sort_values(["total_test_net_pnl", "test_profit_factor"], ascending=[False, False]).iloc[0]
            verdict = str(best_wfa["verdict"])
            stitched_net = float(best_wfa["total_test_net_pnl"])
            stitched_pf = float(best_wfa["test_profit_factor"])
            positive_folds = int(best_wfa["positive_folds"])
            if stitched_net <= 0 and safe_float(best_wfa["benchmark_raw_hybrid_net_pnl_same_windows"], 0.0) <= 0:
                failure_mode = "negative_raw_edge"
            elif stitched_net <= 0 and safe_float(best_wfa["improvement_vs_raw_hybrid"], 0.0) > 0:
                failure_mode = "poor_oos_transfer"
            elif safe_float(best_wfa["pass_rate"], 0.0) < 0.40:
                failure_mode = "unstable_params"
            elif safe_float(best_wfa.get("total_test_trades"), 0.0) < minimum_trades_for_timeframe(str(best_wfa["signal_timeframe"]), scope="oos"):
                failure_mode = "too_few_trades"
            elif stitched_pf < 1.0:
                failure_mode = "costs_too_high"
            else:
                failure_mode = "poor_oos_transfer"
        rows.append(
            {
                "symbol": symbol,
                "best_timeframe_is_only": best_is_timeframe,
                "best_timeframe_wfa": best_wfa_timeframe,
                "selected_configs_count": int(len(selected_slice)),
                "oos_pass_count": int(pd.to_numeric(selected_slice["oos_pass"], errors="coerce").fillna(False).astype(bool).sum()) if not selected_slice.empty else 0,
                "wfa_verdict": verdict,
                "net_pnl_stitched": stitched_net,
                "pf_stitched": stitched_pf,
                "positive_folds": positive_folds,
                "main_failure_mode": failure_mode,
            }
        )
    return pd.DataFrame(rows)


def _save_heatmap(frame: pd.DataFrame, index_col: str, column_col: str, value_col: str, title: str, path: Path) -> None:
    if frame.empty:
        return
    pivot = frame.pivot(index=index_col, columns=column_col, values=value_col)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0.0 if "pnl" in value_col.lower() else None, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _create_figures(
    *,
    output_dir: Path,
    selected_oos_report: pd.DataFrame,
    walkforward_summary: pd.DataFrame,
    portfolio_daily: pd.DataFrame,
    asset_correlation: pd.DataFrame,
    timeframe_summary: pd.DataFrame,
) -> list[str]:
    paths: list[str] = []
    if not selected_oos_report.empty:
        _save_heatmap(
            selected_oos_report.groupby(["symbol", "signal_timeframe"], as_index=False)["profit_factor_oos"].median(),
            "symbol",
            "signal_timeframe",
            "profit_factor_oos",
            "Median OOS PF by Symbol/Timeframe",
            output_dir / "heatmap_oos_pf_by_symbol_timeframe.png",
        )
        paths.append(str(output_dir / "heatmap_oos_pf_by_symbol_timeframe.png"))
        fig, ax = plt.subplots(figsize=(12, 5))
        top = selected_oos_report.copy()
        top["label"] = top["symbol"] + "_" + top["signal_timeframe"] + "_" + top["config_id"].astype(str)
        ax.bar(top["label"], pd.to_numeric(top["net_pnl_oos"], errors="coerce"))
        ax.tick_params(axis="x", rotation=90)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title("Selected Configs OOS")
        fig.tight_layout()
        fig.savefig(output_dir / "selected_configs_oos_bar.png", dpi=150)
        plt.close(fig)
        paths.append(str(output_dir / "selected_configs_oos_bar.png"))

    if not walkforward_summary.empty:
        _save_heatmap(
            walkforward_summary,
            "symbol",
            "signal_timeframe",
            "total_test_net_pnl",
            "Walkforward Stitched PnL by Symbol/Timeframe",
            output_dir / "heatmap_wfa_pnl_by_symbol_timeframe.png",
        )
        paths.append(str(output_dir / "heatmap_wfa_pnl_by_symbol_timeframe.png"))
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_frame = walkforward_summary.copy()
        plot_frame["label"] = plot_frame["symbol"] + "_" + plot_frame["signal_timeframe"]
        ax.bar(plot_frame["label"], pd.to_numeric(plot_frame["total_test_net_pnl"], errors="coerce"))
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title("Walkforward PnL by Fold Aggregate")
        fig.tight_layout()
        fig.savefig(output_dir / "walkforward_pnl_by_fold.png", dpi=150)
        plt.close(fig)
        paths.append(str(output_dir / "walkforward_pnl_by_fold.png"))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            pd.to_numeric(walkforward_summary["train_score_test_corr"], errors="coerce"),
            pd.to_numeric(walkforward_summary["total_test_net_pnl"], errors="coerce"),
            s=70,
        )
        for _, row in walkforward_summary.iterrows():
            ax.annotate(f"{row['symbol']}_{row['signal_timeframe']}", (safe_float(row["train_score_test_corr"], 0.0), safe_float(row["total_test_net_pnl"], 0.0)), fontsize=8)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title("Train Score vs Test PnL All")
        fig.tight_layout()
        fig.savefig(output_dir / "train_score_vs_test_pnl_scatter_all.png", dpi=150)
        plt.close(fig)
        paths.append(str(output_dir / "train_score_vs_test_pnl_scatter_all.png"))

    if not portfolio_daily.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for portfolio_name, part in portfolio_daily.groupby("portfolio_name", sort=True):
            ax.plot(pd.to_datetime(part["session_date"]), pd.to_numeric(part["equity"], errors="coerce"), label=portfolio_name)
        ax.legend(fontsize=8)
        ax.set_title("Portfolio Cumulative PnL")
        fig.tight_layout()
        fig.savefig(output_dir / "portfolio_cumulative_pnl.png", dpi=150)
        plt.close(fig)
        paths.append(str(output_dir / "portfolio_cumulative_pnl.png"))

    if not asset_correlation.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(asset_correlation, cmap="RdYlGn", center=0.0, ax=ax)
        ax.set_title("Asset Correlation Heatmap")
        fig.tight_layout()
        fig.savefig(output_dir / "asset_correlation_heatmap.png", dpi=150)
        plt.close(fig)
        paths.append(str(output_dir / "asset_correlation_heatmap.png"))

    if not timeframe_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=timeframe_summary, x="signal_timeframe", y="median_oos_pf", ax=ax)
        ax.axhline(1.0, color="black", linewidth=1)
        ax.set_title("Timeframe Comparison")
        fig.tight_layout()
        fig.savefig(output_dir / "timeframe_comparison_boxplot.png", dpi=150)
        plt.close(fig)
        paths.append(str(output_dir / "timeframe_comparison_boxplot.png"))
    return paths


def _build_final_report(
    *,
    output_dir: Path,
    data_audits: list[dict[str, Any]],
    config_universe: pd.DataFrame,
    selected_oos_report: pd.DataFrame,
    walkforward_summary: pd.DataFrame,
    timeframe_summary: pd.DataFrame,
    asset_summary: pd.DataFrame,
    portfolio_oos_summary: pd.DataFrame,
    portfolio_walkforward_summary: pd.DataFrame,
) -> None:
    best_oos = (
        selected_oos_report.sort_values(["verdict", "net_pnl_oos"], ascending=[True, False]).iloc[0]
        if not selected_oos_report.empty
        else None
    )
    best_wfa = (
        walkforward_summary.sort_values(["verdict", "total_test_net_pnl"], ascending=[True, False]).iloc[0]
        if not walkforward_summary.empty
        else None
    )
    best_wfa_label = None if best_wfa is None else f"{best_wfa['symbol']} {best_wfa['signal_timeframe']}"
    verdict_rank = {"strong_candidate": 4, "candidate": 3, "weak_watchlist": 2, "reject": 1}
    global_verdict = "Reject pullback family."
    if not walkforward_summary.empty:
        top_verdict = max(walkforward_summary["verdict"], key=lambda value: verdict_rank.get(str(value), 0))
        if top_verdict == "strong_candidate":
            global_verdict = "Strong candidate."
        elif top_verdict == "candidate":
            global_verdict = "Candidate on specific asset/timeframe."
        elif top_verdict == "weak_watchlist":
            global_verdict = "Keep as weak watchlist only."
    if not portfolio_walkforward_summary.empty:
        top_portfolio = portfolio_walkforward_summary.sort_values("net_pnl", ascending=False).iloc[0]
        if safe_float(top_portfolio["net_pnl"], 0.0) > 0 and safe_float(top_portfolio["pnl_to_maxdd"], 0.0) > 0:
            if global_verdict == "Keep as weak watchlist only.":
                global_verdict = "Candidate as diversified portfolio only."

    data_audit_frame = pd.DataFrame(data_audits)
    lines = [
        "# Volume Climax Pullback Multi-Asset Multi-Timeframe Campaign",
        "",
        "## 1. Executive Summary",
        f"- Pullback survival beyond MNQ is {'still not convincing' if global_verdict.startswith('Reject') else 'mixed but not dead'} under realistic 1min execution.",
        f"- Best asset/timeframe in walk-forward: `{best_wfa_label}`.",
        f"- Robust OOS support remains limited: `{int(pd.to_numeric(selected_oos_report['oos_pass'], errors='coerce').fillna(False).astype(bool).sum()) if not selected_oos_report.empty else 0}` selected configs passed OOS filters.",
        f"- Global verdict: `{global_verdict}`",
        "",
        "## 2. Context: MNQ 1H Was Rejected",
        "- The 1H hourly baseline was invalidated once intrabar stop/target sequencing was enforced on the 1min path.",
        "- MNQ hybrid 1H/1min stayed negative and its standalone walk-forward remained reject.",
        "",
        "## 3. Research Design",
        "- Assets: MNQ, MES, M2K, MGC.",
        "- Signal timeframes: 15min, 30min, 1H.",
        "- Execution timeframe: 1min only, with strict intrabar-aware exits.",
        "- Selection uses IS/train only; OOS and walk-forward are reporting and validation only.",
        "",
        "## 4. Data Audit",
        _markdown_table(
            data_audit_frame[
                [
                    "symbol",
                    "signal_timeframe",
                    "number_of_1min_rows",
                    "number_of_signal_rows",
                    "rth_rows",
                    "first_timestamp",
                    "last_timestamp",
                    "split_mode",
                    "variant_name",
                ]
            ]
        ),
        "",
        "## 5. Config Universe",
        _markdown_table(
            config_universe.groupby(["family"], as_index=False)
            .agg(configs=("config_id", "count"))
            .sort_values("family")
        ),
        "",
        "## 6. IS/OOS Results by Asset and Timeframe",
        _markdown_table(
            selected_oos_report[
                [
                    "symbol",
                    "signal_timeframe",
                    "rank_is",
                    "config_id",
                    "family",
                    "net_pnl",
                    "profit_factor",
                    "net_pnl_oos",
                    "profit_factor_oos",
                    "oos_pass",
                    "verdict",
                ]
            ]
        ) if not selected_oos_report.empty else "No selected OOS rows.",
        "",
        "## 7. Walk-Forward Results",
        _markdown_table(
            walkforward_summary[
                [
                    "symbol",
                    "signal_timeframe",
                    "total_test_net_pnl",
                    "test_profit_factor",
                    "positive_folds",
                    "pass_rate",
                    "improvement_vs_raw_hybrid",
                    "verdict",
                ]
            ]
        ) if not walkforward_summary.empty else "No walk-forward rows.",
        "",
        "## 8. Timeframe Analysis",
        _markdown_table(timeframe_summary) if not timeframe_summary.empty else "No timeframe summary.",
        "",
        "## 9. Asset Analysis",
        _markdown_table(asset_summary) if not asset_summary.empty else "No asset summary.",
        "",
        "## 10. Family Analysis",
        _markdown_table(
            selected_oos_report.groupby("family", as_index=False)
            .agg(
                selected=("config_id", "count"),
                oos_pass_rate=("oos_pass", lambda values: float(pd.Series(values).fillna(False).astype(bool).mean())),
                median_oos_pnl=("net_pnl_oos", "median"),
            )
            .sort_values("median_oos_pnl", ascending=False)
        ) if not selected_oos_report.empty else "No family analysis.",
        "",
        "## 11. Portfolio Test",
        _markdown_table(portfolio_oos_summary) if not portfolio_oos_summary.empty else "No OOS portfolio summary.",
        "",
        "## 12. Best Candidate Audit",
        _markdown_table(pd.DataFrame([best_wfa])) if best_wfa is not None else "No candidate passed walk-forward strongly enough.",
        "",
        "## 13. Failure Modes",
        "- The dominant failure modes remain poor OOS transfer, unstable parameter rankings, intrabar stop-before-target damage, and sparse trade counts on some sleeves.",
        "",
        "## 14. Verdict",
        f"- `{global_verdict}`",
        "",
        "## 15. Next Actions",
        "- Archive pullback standalone if the verdict stays reject across assets and timeframes.",
        "- Keep surviving sleeves only as watchlist overlays or feature candidates inside broader intraday books.",
        "- Move more effort toward ORB and execution-ready strategies if no cross-asset candidate emerges.",
        "- If one sleeve survives, stress it with prop-firm constraints, slippage shocks, and live shadow monitoring.",
        "- Add CI guardrails requiring realistic execution and walk-forward validation before publishing intraday alpha claims.",
    ]
    (output_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _task_payload(symbol: str, signal_timeframe: str, execution_timeframe: str, output_dir: Path) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "signal_timeframe": signal_timeframe,
        "execution_timeframe": execution_timeframe,
        "output_dir": str(output_dir),
    }


def _worker_run(payload: dict[str, Any]) -> dict[str, Any]:
    return _evaluate_symbol_timeframe(
        symbol=str(payload["symbol"]),
        signal_timeframe=str(payload["signal_timeframe"]),
        execution_timeframe=str(payload["execution_timeframe"]),
        output_dir=Path(payload["output_dir"]),
    )


def run_campaign(
    *,
    symbols: Sequence[str],
    signal_timeframes: Sequence[str],
    execution_timeframe: str,
    output_root: Path,
    max_workers: int | None = None,
    dataset_overrides: dict[str, pd.DataFrame] | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / f"volume_climax_pullback_multiasset_multitimeframe_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_universe = build_config_universe(symbols, signal_timeframes, execution_timeframe)
    config_universe_frame = _config_universe_frame(config_universe)
    results: list[dict[str, Any]] = []

    tasks = [
        _task_payload(symbol, signal_timeframe, execution_timeframe, output_dir)
        for symbol in symbols
        for signal_timeframe in signal_timeframes
    ]

    if dataset_overrides is not None or (max_workers or 1) <= 1:
        for payload in tasks:
            override = None if dataset_overrides is None else dataset_overrides.get(str(payload["symbol"]))
            results.append(
                _evaluate_symbol_timeframe(
                    symbol=str(payload["symbol"]),
                    signal_timeframe=str(payload["signal_timeframe"]),
                    execution_timeframe=str(payload["execution_timeframe"]),
                    output_dir=output_dir,
                    raw_minute_df_override=override,
                )
            )
    else:
        worker_count = min(max_workers or max(1, os.cpu_count() or 1), len(tasks))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(_worker_run, payload): payload for payload in tasks}
            for future in as_completed(future_map):
                results.append(future.result())

    data_audits = [result["data_audit"] for result in results]
    metrics_is = pd.concat([result["metrics_is"] for result in results], ignore_index=True) if results else pd.DataFrame()
    metrics_oos = pd.concat([result["metrics_oos"] for result in results], ignore_index=True) if results else pd.DataFrame()
    metrics_full = pd.concat([result["metrics_full"] for result in results], ignore_index=True) if results else pd.DataFrame()
    robustness = pd.concat([result["robustness"] for result in results], ignore_index=True) if results else pd.DataFrame()
    selected_is = pd.concat([result["selected_is"] for result in results], ignore_index=True) if results else pd.DataFrame()
    selected_oos_report = pd.concat([result["selected_oos_report"] for result in results], ignore_index=True) if results else pd.DataFrame()
    walkforward_folds = pd.concat([result["walkforward_folds"] for result in results], ignore_index=True) if results else pd.DataFrame()
    fold_train_ranking = pd.concat([result["fold_train_ranking"] for result in results], ignore_index=True) if results else pd.DataFrame()
    fold_selected_test_results = pd.concat([result["fold_selected_test_results"] for result in results], ignore_index=True) if results else pd.DataFrame()
    walkforward_stitched_trades = pd.concat([result["walkforward_stitched_trades"] for result in results], ignore_index=True) if results else pd.DataFrame()
    walkforward_daily = pd.concat([result["walkforward_stitched_daily_returns"] for result in results], ignore_index=True) if results else pd.DataFrame()
    walkforward_summary = pd.concat([result["walkforward_summary"] for result in results], ignore_index=True) if results else pd.DataFrame()
    selected_oos_daily = pd.concat([result["selected_oos_daily"] for result in results], ignore_index=True) if results else pd.DataFrame()
    selected_is_daily = pd.concat([result["selected_is_daily"] for result in results], ignore_index=True) if results else pd.DataFrame()

    portfolio_oos_summary, portfolio_walkforward_summary, portfolio_daily, asset_correlation, portfolio_asset_contribution = _build_portfolio_outputs(
        selected_oos_daily=selected_oos_daily,
        selected_is_daily=selected_is_daily,
        walkforward_daily=walkforward_daily,
        selected_oos_report=selected_oos_report,
        walkforward_summary=walkforward_summary,
    )

    timeframe_summary = build_timeframe_summary(metrics_is, metrics_oos, selected_oos_report, walkforward_summary)
    asset_summary = build_asset_summary(selected_oos_report, walkforward_summary)

    config_universe_frame.to_csv(output_dir / "config_universe.csv", index=False)
    metrics_is.to_csv(output_dir / "config_metrics_is.csv", index=False)
    metrics_oos.to_csv(output_dir / "config_metrics_oos.csv", index=False)
    metrics_full.to_csv(output_dir / "config_metrics_full.csv", index=False)
    robustness.to_csv(output_dir / "config_robustness_scores.csv", index=False)
    selected_is.to_csv(output_dir / "selected_configs_is_only.csv", index=False)
    selected_oos_report.to_csv(output_dir / "selected_configs_oos_report.csv", index=False)
    walkforward_folds.to_csv(output_dir / "walkforward_folds.csv", index=False)
    fold_train_ranking.to_csv(output_dir / "fold_train_ranking.csv", index=False)
    fold_selected_test_results.to_csv(output_dir / "fold_selected_test_results.csv", index=False)
    walkforward_stitched_trades.to_csv(output_dir / "walkforward_stitched_trades.csv", index=False)
    walkforward_summary.to_csv(output_dir / "walkforward_summary.csv", index=False)
    portfolio_oos_summary.to_csv(output_dir / "portfolio_oos_summary.csv", index=False)
    portfolio_walkforward_summary.to_csv(output_dir / "portfolio_walkforward_summary.csv", index=False)
    portfolio_daily.to_csv(output_dir / "portfolio_daily_pnl.csv", index=False)
    asset_correlation.to_csv(output_dir / "asset_correlation_matrix.csv")
    portfolio_asset_contribution.to_csv(output_dir / "portfolio_asset_contribution.csv", index=False)
    timeframe_summary.to_csv(output_dir / "timeframe_summary.csv", index=False)
    asset_summary.to_csv(output_dir / "asset_summary.csv", index=False)
    walkforward_daily.to_csv(output_dir / "walkforward_stitched_daily_returns.csv", index=False)

    plot_paths = _create_figures(
        output_dir=output_dir,
        selected_oos_report=selected_oos_report,
        walkforward_summary=walkforward_summary,
        portfolio_daily=portfolio_daily,
        asset_correlation=asset_correlation,
        timeframe_summary=timeframe_summary,
    )

    _build_final_report(
        output_dir=output_dir,
        data_audits=data_audits,
        config_universe=config_universe_frame,
        selected_oos_report=selected_oos_report,
        walkforward_summary=walkforward_summary,
        timeframe_summary=timeframe_summary,
        asset_summary=asset_summary,
        portfolio_oos_summary=portfolio_oos_summary,
        portfolio_walkforward_summary=portfolio_walkforward_summary,
    )

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "symbols": list(symbols),
        "signal_timeframes": list(signal_timeframes),
        "execution_timeframe": execution_timeframe,
        "output_dir": str(output_dir),
        "python_version": sys.version,
        "platform": platform.platform(),
        "plots": plot_paths,
        "input_files": {
            symbol: _file_metadata(latest_path_for_symbol(symbol))
            for symbol in symbols
        },
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-asset multi-timeframe realistic pullback campaign.")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS), help="Symbols to evaluate.")
    parser.add_argument("--signal-timeframes", nargs="+", default=list(DEFAULT_SIGNAL_TIMEFRAMES), help="Signal timeframes, e.g. 15min 30min 1H.")
    parser.add_argument("--execution-timeframe", default=DEFAULT_EXECUTION_TIMEFRAME, help="Execution timeframe, must remain 1min.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Export root.")
    parser.add_argument("--max-workers", type=int, default=min(4, max(1, os.cpu_count() or 1)), help="Parallel workers across asset/timeframe tasks.")
    args = parser.parse_args()

    run_dir = run_campaign(
        symbols=[str(symbol).upper() for symbol in args.symbols],
        signal_timeframes=[str(timeframe) for timeframe in args.signal_timeframes],
        execution_timeframe=str(args.execution_timeframe),
        output_root=Path(args.output_root),
        max_workers=int(args.max_workers),
    )
    print(run_dir)


if __name__ == "__main__":
    main()
