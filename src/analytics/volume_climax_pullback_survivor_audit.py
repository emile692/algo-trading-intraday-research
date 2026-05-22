"""Strict survivor audit for volume climax pullback after the multi-asset campaign."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.analytics.volume_climax_pullback_common import latest_path_for_symbol, load_symbol_data, safe_float
from src.analytics.volume_climax_pullback_intrabar_recalibration_campaign import (
    IntrabarRecalibrationConfig,
    _compute_trade_metrics,
    _daily_returns,
    _file_metadata,
    _markdown_table,
    _simulate_config,
)
from src.analytics.volume_climax_pullback_multiasset_multitimeframe_campaign import (
    FIXED_SPLIT_IS_END,
    FIXED_SPLIT_OOS_START,
    WalkforwardFold,
    _clone_variant_for_timeframe,
    _infer_split_mode,
    _period_mask,
    _resolve_seed_variant,
    build_walkforward_folds,
    minimum_trades_for_timeframe,
    resample_rth_timeframe,
    timeframe_to_minutes,
)
from src.config.settings import DEFAULT_TIMEZONE
from src.data.session import extract_rth
from src.engine.vwap_backtester import build_execution_model_for_profile
from src.strategy.volume_climax_pullback_v2 import (
    VolumeClimaxPullbackV2Variant,
    build_volume_climax_pullback_v2_signal_frame,
    prepare_volume_climax_pullback_v2_features,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ("M2K", "MGC", "MNQ")
DEFAULT_SIGNAL_TIMEFRAME = "1H"
DEFAULT_EXECUTION_TIMEFRAME = "1min"
DEFAULT_OUTPUT_ROOT = Path("export")


@dataclass(frozen=True)
class SurvivorAuditConfig:
    config_id: str
    symbol: str
    signal_timeframe: str
    execution_timeframe: str
    base_signal_variant: str
    family: str
    cluster_id: str
    stop_multiplier: float
    target_multiplier: float
    entry_delay_minutes: int
    time_stop_bars: int
    filter_name: str
    filter_params: dict[str, Any]

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


def _normalize(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.notna().sum() <= 1:
        return pd.Series(np.where(numeric.notna(), 1.0, np.nan), index=series.index, dtype=float)
    spread = numeric.max() - numeric.min()
    if spread == 0:
        return pd.Series(np.where(numeric.notna(), 1.0, np.nan), index=series.index, dtype=float)
    return (numeric - numeric.min()) / spread


def _config_sort_key(config: SurvivorAuditConfig) -> tuple[Any, ...]:
    payload = json.dumps(config.filter_params, sort_keys=True)
    return (
        str(config.family),
        str(config.cluster_id),
        int(config.time_stop_bars),
        float(config.stop_multiplier),
        float(config.target_multiplier),
        int(config.entry_delay_minutes),
        str(config.filter_name),
        payload,
        str(config.config_id),
    )


def build_candidate_universe(
    *,
    symbols: Sequence[str],
    signal_timeframe: str,
    execution_timeframe: str,
    include_negative_control: bool,
    max_configs_per_family: int,
) -> list[SurvivorAuditConfig]:
    configs: list[SurvivorAuditConfig] = []

    def _add(
        *,
        symbol: str,
        family: str,
        cluster_id: str,
        stop_multiplier: float,
        target_multiplier: float,
        entry_delay_minutes: int,
        time_stop_bars: int,
        filter_name: str,
        filter_params: dict[str, Any],
        base_signal_variant: str,
    ) -> None:
        parts = [
            symbol.lower(),
            signal_timeframe.lower(),
            family,
            filter_name,
            f"sm{stop_multiplier:.2f}".replace(".", "p"),
            f"tm{target_multiplier:.2f}".replace(".", "p"),
            f"d{entry_delay_minutes}",
            f"ts{int(time_stop_bars)}",
        ]
        if filter_name == "require_no_stop_zone_touch_before_entry":
            parts.append(f"sz{float(filter_params['stop_zone_fraction']):.2f}".replace(".", "p"))
        elif filter_name == "avoid_immediate_adverse_move":
            parts.append(
                f"aw{int(filter_params['adverse_window_minutes'])}_mt{int(filter_params['max_adverse_ticks'])}"
            )
        config_id = "_".join(parts)
        configs.append(
            SurvivorAuditConfig(
                config_id=config_id,
                symbol=symbol,
                signal_timeframe=signal_timeframe,
                execution_timeframe=execution_timeframe,
                base_signal_variant=base_signal_variant,
                family=family,
                cluster_id=cluster_id,
                stop_multiplier=float(stop_multiplier),
                target_multiplier=float(target_multiplier),
                entry_delay_minutes=int(entry_delay_minutes),
                time_stop_bars=int(time_stop_bars),
                filter_name=filter_name,
                filter_params=dict(filter_params),
            )
        )

    for symbol in symbols:
        if symbol.upper() == "MNQ" and not include_negative_control:
            continue

        base_variant, _, _ = _resolve_seed_variant(symbol)
        seed_name = base_variant.name
        if symbol.upper() == "M2K":
            for time_stop_bars in (3, 4):
                _add(
                    symbol=symbol,
                    family="raw_hybrid",
                    cluster_id="m2k_1h_raw_baseline",
                    stop_multiplier=1.0,
                    target_multiplier=1.0,
                    entry_delay_minutes=0,
                    time_stop_bars=time_stop_bars,
                    filter_name="none",
                    filter_params={},
                    base_signal_variant=seed_name,
                )
            for time_stop_bars in (3, 4):
                for stop_multiplier in (0.75, 1.0, 1.25):
                    for target_multiplier in (2.0, 2.5):
                        for entry_delay_minutes in (5, 15, 30):
                            _add(
                                symbol=symbol,
                                family="none_baseline",
                                cluster_id="m2k_1h_none_local",
                                stop_multiplier=stop_multiplier,
                                target_multiplier=target_multiplier,
                                entry_delay_minutes=entry_delay_minutes,
                                time_stop_bars=time_stop_bars,
                                filter_name="none",
                                filter_params={},
                                base_signal_variant=seed_name,
                            )
            for time_stop_bars in (3, 4):
                for stop_multiplier in (0.75, 1.0, 1.25, 1.5):
                    for target_multiplier in (2.0, 2.5, 3.0):
                        for entry_delay_minutes in (5, 15, 30):
                            for adverse_window_minutes in (5, 10):
                                if entry_delay_minutes < adverse_window_minutes:
                                    continue
                                for max_adverse_ticks in (8, 12):
                                    _add(
                                        symbol=symbol,
                                        family="delay_adverse_filter",
                                        cluster_id="m2k_1h_adverse_core",
                                        stop_multiplier=stop_multiplier,
                                        target_multiplier=target_multiplier,
                                        entry_delay_minutes=entry_delay_minutes,
                                        time_stop_bars=time_stop_bars,
                                        filter_name="avoid_immediate_adverse_move",
                                        filter_params={
                                            "adverse_window_minutes": int(adverse_window_minutes),
                                            "max_adverse_ticks": int(max_adverse_ticks),
                                        },
                                        base_signal_variant=seed_name,
                                    )
            for time_stop_bars in (3, 4):
                for stop_multiplier in (0.75, 1.0, 1.25):
                    for target_multiplier in (2.0, 2.5):
                        for entry_delay_minutes in (5, 15, 30):
                            for stop_zone_fraction in (0.5, 0.75, 1.0):
                                _add(
                                    symbol=symbol,
                                    family="delay_stop_zone_filter",
                                    cluster_id="m2k_1h_stop_zone_diag",
                                    stop_multiplier=stop_multiplier,
                                    target_multiplier=target_multiplier,
                                    entry_delay_minutes=entry_delay_minutes,
                                    time_stop_bars=time_stop_bars,
                                    filter_name="require_no_stop_zone_touch_before_entry",
                                    filter_params={"stop_zone_fraction": float(stop_zone_fraction)},
                                    base_signal_variant=seed_name,
                                )
        elif symbol.upper() == "MGC":
            for time_stop_bars in (2, 3, 4):
                _add(
                    symbol=symbol,
                    family="raw_hybrid",
                    cluster_id="mgc_1h_raw_baseline",
                    stop_multiplier=1.0,
                    target_multiplier=1.0,
                    entry_delay_minutes=0,
                    time_stop_bars=time_stop_bars,
                    filter_name="none",
                    filter_params={},
                    base_signal_variant=seed_name,
                )
            for time_stop_bars in (2, 3, 4):
                for stop_multiplier in (1.0, 1.25, 1.5):
                    for target_multiplier in (2.5, 3.0):
                        for entry_delay_minutes in (5, 15, 30):
                            _add(
                                symbol=symbol,
                                family="none_baseline",
                                cluster_id="mgc_1h_none_core",
                                stop_multiplier=stop_multiplier,
                                target_multiplier=target_multiplier,
                                entry_delay_minutes=entry_delay_minutes,
                                time_stop_bars=time_stop_bars,
                                filter_name="none",
                                filter_params={},
                                base_signal_variant=seed_name,
                            )
            for time_stop_bars in (2, 3, 4):
                for stop_multiplier in (1.0, 1.25, 1.5):
                    for target_multiplier in (2.5, 3.0):
                        for entry_delay_minutes in (5, 15, 30):
                            for stop_zone_fraction in (0.5, 0.75, 1.0):
                                _add(
                                    symbol=symbol,
                                    family="delay_stop_zone_filter",
                                    cluster_id="mgc_1h_stop_zone_core",
                                    stop_multiplier=stop_multiplier,
                                    target_multiplier=target_multiplier,
                                    entry_delay_minutes=entry_delay_minutes,
                                    time_stop_bars=time_stop_bars,
                                    filter_name="require_no_stop_zone_touch_before_entry",
                                    filter_params={"stop_zone_fraction": float(stop_zone_fraction)},
                                    base_signal_variant=seed_name,
                                )
            for time_stop_bars in (2, 3, 4):
                for stop_multiplier in (1.0, 1.25):
                    for target_multiplier in (2.5, 3.0):
                        for entry_delay_minutes in (15, 30):
                            for adverse_window_minutes in (5, 10):
                                if entry_delay_minutes < adverse_window_minutes:
                                    continue
                                for max_adverse_ticks in (8, 12):
                                    _add(
                                        symbol=symbol,
                                        family="delay_adverse_filter",
                                        cluster_id="mgc_1h_adverse_diag",
                                        stop_multiplier=stop_multiplier,
                                        target_multiplier=target_multiplier,
                                        entry_delay_minutes=entry_delay_minutes,
                                        time_stop_bars=time_stop_bars,
                                        filter_name="avoid_immediate_adverse_move",
                                        filter_params={
                                            "adverse_window_minutes": int(adverse_window_minutes),
                                            "max_adverse_ticks": int(max_adverse_ticks),
                                        },
                                        base_signal_variant=seed_name,
                                    )
        elif symbol.upper() == "MNQ":
            for time_stop_bars in (2, 3):
                _add(
                    symbol=symbol,
                    family="raw_hybrid",
                    cluster_id="mnq_1h_negative_control_raw",
                    stop_multiplier=1.0,
                    target_multiplier=1.0,
                    entry_delay_minutes=0,
                    time_stop_bars=time_stop_bars,
                    filter_name="none",
                    filter_params={},
                    base_signal_variant=seed_name,
                )
            for time_stop_bars in (2, 3):
                for target_multiplier in (2.0, 2.5):
                    for entry_delay_minutes in (15, 30):
                        _add(
                            symbol=symbol,
                            family="none_baseline",
                            cluster_id="mnq_1h_negative_control_none",
                            stop_multiplier=1.0,
                            target_multiplier=target_multiplier,
                            entry_delay_minutes=entry_delay_minutes,
                            time_stop_bars=time_stop_bars,
                            filter_name="none",
                            filter_params={},
                            base_signal_variant=seed_name,
                        )
            for time_stop_bars in (2, 3):
                for stop_multiplier in (0.75, 1.0):
                    for target_multiplier in (2.0, 2.5):
                        for entry_delay_minutes in (5, 15):
                            for max_adverse_ticks in (12, 16):
                                _add(
                                    symbol=symbol,
                                    family="delay_adverse_filter",
                                    cluster_id="mnq_1h_negative_control_adverse",
                                    stop_multiplier=stop_multiplier,
                                    target_multiplier=target_multiplier,
                                    entry_delay_minutes=entry_delay_minutes,
                                    time_stop_bars=time_stop_bars,
                                    filter_name="avoid_immediate_adverse_move",
                                    filter_params={
                                        "adverse_window_minutes": 5,
                                        "max_adverse_ticks": int(max_adverse_ticks),
                                    },
                                    base_signal_variant=seed_name,
                                )
                            for stop_zone_fraction in (0.75, 1.0):
                                _add(
                                    symbol=symbol,
                                    family="delay_stop_zone_filter",
                                    cluster_id="mnq_1h_negative_control_stop_zone",
                                    stop_multiplier=stop_multiplier,
                                    target_multiplier=target_multiplier,
                                    entry_delay_minutes=entry_delay_minutes,
                                    time_stop_bars=time_stop_bars,
                                    filter_name="require_no_stop_zone_touch_before_entry",
                                    filter_params={"stop_zone_fraction": float(stop_zone_fraction)},
                                    base_signal_variant=seed_name,
                                )

    deduped = {config.config_id: config for config in configs}
    capped: list[SurvivorAuditConfig] = []
    family_buckets: dict[tuple[str, str], list[SurvivorAuditConfig]] = {}
    for config in sorted(deduped.values(), key=_config_sort_key):
        family_buckets.setdefault((config.symbol, config.family), []).append(config)
    for bucket in family_buckets.values():
        capped.extend(bucket[: max(1, int(max_configs_per_family))])
    return sorted(capped, key=_config_sort_key)


def config_frame_from_universe(configs: Sequence[SurvivorAuditConfig]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "config_id": config.config_id,
                "symbol": config.symbol,
                "signal_timeframe": config.signal_timeframe,
                "execution_timeframe": config.execution_timeframe,
                "base_signal_variant": config.base_signal_variant,
                "family": config.family,
                "cluster_id": config.cluster_id,
                "stop_multiplier": config.stop_multiplier,
                "target_multiplier": config.target_multiplier,
                "entry_delay_minutes": config.entry_delay_minutes,
                "variant_time_stop_bars": config.time_stop_bars,
                "filter_name": config.filter_name,
                "filter_params_json": json.dumps(config.filter_params, sort_keys=True),
                "stop_zone_fraction": safe_float(config.filter_params.get("stop_zone_fraction"), np.nan),
                "adverse_window_minutes": safe_float(config.filter_params.get("adverse_window_minutes"), np.nan),
                "max_adverse_ticks": safe_float(config.filter_params.get("max_adverse_ticks"), np.nan),
            }
            for config in configs
        ]
    )


def _yearly_train_metrics(
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


def compute_extended_metrics(events: pd.DataFrame, *, estimated_cost_per_trade: float) -> dict[str, Any]:
    metrics = _compute_trade_metrics(events, estimated_cost_per_trade=estimated_cost_per_trade)
    executed = events.loc[events.get("executed", False)].copy()
    daily = _daily_returns(events)
    session_dates = pd.to_datetime(events.get("session_date"), errors="coerce").dt.date.dropna()
    total_days = len(pd.Index(session_dates).unique()) if not session_dates.empty else 0
    active_days = len(pd.Index(pd.to_datetime(daily.get("session_date"), errors="coerce").dt.date.dropna()).unique()) if not daily.empty else 0
    long_mask = executed.get("direction", pd.Series(dtype=str)).astype(str).str.lower() == "long"
    short_mask = executed.get("direction", pd.Series(dtype=str)).astype(str).str.lower() == "short"
    fold_daily = pd.to_numeric(daily.get("daily_pnl"), errors="coerce")
    fold_sharpe = float(np.sqrt(len(fold_daily)) * fold_daily.mean() / fold_daily.std(ddof=0)) if len(fold_daily) > 1 and fold_daily.std(ddof=0) > 0 else np.nan
    metrics.update(
        {
            "max_daily_drawdown": float(pd.to_numeric(daily.get("daily_pnl"), errors="coerce").min()) if not daily.empty else 0.0,
            "active_days": int(active_days),
            "calendar_days": int(total_days),
            "exposure_ratio": float(active_days / total_days) if total_days > 0 else 0.0,
            "fold_sharpe": fold_sharpe,
            "long_trades": int(long_mask.sum()) if not executed.empty else 0,
            "short_trades": int(short_mask.sum()) if not executed.empty else 0,
            "long_pnl": float(pd.to_numeric(executed.loc[long_mask, "net_pnl_usd"], errors="coerce").sum()) if not executed.empty else 0.0,
            "short_pnl": float(pd.to_numeric(executed.loc[short_mask, "net_pnl_usd"], errors="coerce").sum()) if not executed.empty else 0.0,
        }
    )
    return metrics


def compute_train_only_inverse_vol_weights(train_daily_by_symbol: dict[str, pd.DataFrame]) -> dict[str, float]:
    vol_scores: dict[str, float] = {}
    for symbol, daily in train_daily_by_symbol.items():
        if daily.empty:
            vol_scores[symbol] = 0.0
            continue
        series = pd.to_numeric(daily["daily_pnl"], errors="coerce")
        vol = float(series.std(ddof=0)) if len(series) > 1 and series.std(ddof=0) > 0 else np.nan
        vol_scores[symbol] = 1.0 / vol if np.isfinite(vol) and vol > 0 else 0.0
    total = sum(vol_scores.values())
    if total <= 0:
        keys = [symbol for symbol, daily in train_daily_by_symbol.items() if not daily.empty]
        if not keys:
            return {}
        equal = 1.0 / len(keys)
        return {symbol: equal for symbol in keys}
    return {symbol: value / total for symbol, value in vol_scores.items() if value > 0}


def build_portfolio_weights_from_train(
    train_daily_by_symbol: dict[str, pd.DataFrame],
    *,
    scheme: str,
) -> dict[str, float]:
    available = [symbol for symbol, daily in train_daily_by_symbol.items() if not daily.empty]
    if not available:
        return {}
    if scheme == "equal_weight":
        weight = 1.0 / len(available)
        return {symbol: weight for symbol in available}
    if scheme == "inverse_vol":
        return compute_train_only_inverse_vol_weights(train_daily_by_symbol)
    if scheme == "capped_equal_weight":
        base = 1.0 / len(available)
        capped = min(base, 0.60)
        weights = {symbol: capped for symbol in available}
        remainder = 1.0 - sum(weights.values())
        if remainder > 0:
            bonus = remainder / len(available)
            weights = {symbol: weight + bonus for symbol, weight in weights.items()}
        return weights
    raise ValueError(f"Unsupported portfolio scheme {scheme!r}")


def _cluster_neighbor_stats(frame: pd.DataFrame, row: pd.Series) -> tuple[float, float]:
    cluster = frame.loc[frame["cluster_id"] == row["cluster_id"]].copy()
    if cluster.empty:
        return np.nan, np.nan
    return (
        float(pd.to_numeric(cluster["net_pnl_oos"], errors="coerce").median()),
        float(pd.to_numeric(cluster["fixed_positive_fold_ratio"], errors="coerce").mean()),
    )


def compute_survivor_robustness_scores(
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
    frame["trade_count_score"] = pd.to_numeric(frame["trades"], errors="coerce").div(
        minimum_trades_for_timeframe("1H", scope="is")
    ).clip(upper=1.0)
    simplicity_map = {
        "raw_hybrid": 1.0,
        "none_baseline": 0.80,
        "delay_adverse_filter": 0.55,
        "delay_stop_zone_filter": 0.50,
    }
    frame["simplicity_score"] = frame["family"].map(lambda family: simplicity_map.get(str(family), 0.40))

    temporal_scores: list[float] = []
    penalties_list: list[float] = []
    admissible_list: list[bool] = []
    positive_years_list: list[int] = []
    years_with_trades_list: list[int] = []
    max_year_contribution_list: list[float] = []
    cluster_scores: list[float] = []

    for _, row in frame.iterrows():
        config_id = str(row["config_id"])
        yearly = _yearly_train_metrics(
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
        temporal = (
            0.50 * (positive_years / max(years_with_trades, 1))
            + 0.50 * max(0.0, 1.0 - max_year_contribution_pct)
        )
        cluster_slice = frame.loc[frame["cluster_id"] == row["cluster_id"]].copy()
        cluster_positive_ratio = float((pd.to_numeric(cluster_slice["net_pnl"], errors="coerce") > 0).mean()) if not cluster_slice.empty else 0.0
        penalties = 0.0
        if safe_float(row["trades"], 0.0) < minimum_trades_for_timeframe("1H", scope="is"):
            penalties += 0.20
        if safe_float(row["profit_factor"], 0.0) < 1.05:
            penalties += 0.15
        if safe_float(row["skip_rate"], 0.0) > 0.70:
            penalties += 0.10
        if safe_float(row["avg_trade"], 0.0) <= estimated_cost_per_trade * 1.50:
            penalties += 0.15
        if max_year_contribution_pct > 0.70:
            penalties += 0.10
        if safe_float(row["avg_holding_minutes"], 0.0) > 240.0:
            penalties += 0.05

        admissible = (
            safe_float(row["net_pnl"], 0.0) > 0
            and safe_float(row["profit_factor"], 0.0) >= 1.10
            and safe_float(row["trades"], 0.0) >= minimum_trades_for_timeframe("1H", scope="is")
            and safe_float(row["avg_trade"], 0.0) > estimated_cost_per_trade * 1.5
        )
        temporal_scores.append(temporal)
        penalties_list.append(penalties)
        admissible_list.append(bool(admissible))
        positive_years_list.append(positive_years)
        years_with_trades_list.append(years_with_trades)
        max_year_contribution_list.append(max_year_contribution_pct)
        cluster_scores.append(cluster_positive_ratio)

    frame["temporal_stability_score"] = temporal_scores
    frame["cluster_positive_ratio_is"] = cluster_scores
    frame["penalties"] = penalties_list
    frame["admissible_is"] = admissible_list
    frame["positive_years_is"] = positive_years_list
    frame["years_with_trades_is"] = years_with_trades_list
    frame["max_year_contribution_pct_is"] = max_year_contribution_list
    frame["robust_score_is"] = (
        0.20 * frame["normalized_net_pnl"].fillna(0.0)
        + 0.20 * frame["normalized_profit_factor"].fillna(0.0)
        + 0.15 * frame["normalized_pnl_to_maxdd"].fillna(0.0)
        + 0.15 * frame["temporal_stability_score"].fillna(0.0)
        + 0.15 * frame["cluster_positive_ratio_is"].fillna(0.0)
        + 0.10 * frame["trade_count_score"].fillna(0.0)
        + 0.05 * frame["simplicity_score"].fillna(0.0)
        - frame["penalties"].fillna(0.0)
    )
    return frame.sort_values(
        ["robust_score_is", "net_pnl", "profit_factor", "avg_trade"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def select_fold_winner(train_ranking: pd.DataFrame) -> pd.Series:
    if train_ranking.empty:
        raise ValueError("Cannot select fold winner from empty ranking.")
    ranked = train_ranking.sort_values(
        ["train_robust_score", "train_net_pnl", "train_profit_factor", "train_avg_trade"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def compute_survivor_fold_train_ranking(
    *,
    symbol: str,
    fold: WalkforwardFold,
    config_frame: pd.DataFrame,
    events_by_config: dict[str, pd.DataFrame],
    estimated_cost_per_trade: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, config_row in config_frame.iterrows():
        config_id = str(config_row["config_id"])
        events = events_by_config[config_id].loc[_period_mask(events_by_config[config_id], fold.train_start, fold.train_end)].copy()
        metrics = compute_extended_metrics(events, estimated_cost_per_trade=estimated_cost_per_trade)
        yearly = _yearly_train_metrics(
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
                "fold_id": fold.fold_id,
                "config_id": config_id,
                "family": config_row["family"],
                "cluster_id": config_row["cluster_id"],
                "filter_name": config_row["filter_name"],
                "stop_multiplier": config_row["stop_multiplier"],
                "target_multiplier": config_row["target_multiplier"],
                "entry_delay_minutes": config_row["entry_delay_minutes"],
                "variant_time_stop_bars": config_row["variant_time_stop_bars"],
                "stop_zone_fraction": config_row["stop_zone_fraction"],
                "adverse_window_minutes": config_row["adverse_window_minutes"],
                "max_adverse_ticks": config_row["max_adverse_ticks"],
                "train_trades": int(metrics["trades"]),
                "train_net_pnl": float(metrics["net_pnl"]),
                "train_gross_profit": float(metrics["gross_profit"]),
                "train_gross_loss": float(metrics["gross_loss"]),
                "train_profit_factor": float(metrics["profit_factor"]),
                "train_win_rate": float(metrics["winrate"]),
                "train_avg_trade": float(metrics["avg_trade"]),
                "train_median_trade": float(metrics["median_trade"]),
                "train_max_drawdown": float(metrics["max_drawdown"]),
                "train_max_daily_drawdown": float(metrics["max_daily_drawdown"]),
                "train_pnl_to_maxdd": float(metrics["pnl_to_maxdd"]) if pd.notna(metrics["pnl_to_maxdd"]) else np.nan,
                "train_skip_rate": float(metrics["skip_rate"]),
                "train_avg_holding_minutes": float(metrics["avg_holding_minutes"]),
                "positive_years_train": positive_years,
                "years_with_trades_train": years_with_trades,
                "max_year_contribution_pct_train": max_year_contribution_pct,
            }
        )
    frame = pd.DataFrame(rows)
    frame["normalized_net_pnl"] = _normalize(frame["train_net_pnl"])
    frame["normalized_profit_factor"] = _normalize(frame["train_profit_factor"].replace(np.inf, np.nan))
    frame["normalized_pnl_to_maxdd"] = _normalize(frame["train_pnl_to_maxdd"])
    frame["trade_count_score"] = pd.to_numeric(frame["train_trades"], errors="coerce").div(
        minimum_trades_for_timeframe("1H", scope="is")
    ).clip(upper=1.0)
    cluster_scores: list[float] = []
    temporal_scores: list[float] = []
    penalties: list[float] = []
    for _, row in frame.iterrows():
        cluster_slice = frame.loc[frame["cluster_id"] == row["cluster_id"]].copy()
        cluster_positive_ratio = float((pd.to_numeric(cluster_slice["train_net_pnl"], errors="coerce") > 0).mean()) if not cluster_slice.empty else 0.0
        temporal = (
            0.50 * (safe_float(row["positive_years_train"], 0.0) / max(safe_float(row["years_with_trades_train"], 1.0), 1.0))
            + 0.50 * max(0.0, 1.0 - safe_float(row["max_year_contribution_pct_train"], 1.0))
        )
        penalty = 0.0
        if safe_float(row["train_trades"], 0.0) < 30:
            penalty += 0.20
        if safe_float(row["train_profit_factor"], 0.0) < 1.05:
            penalty += 0.15
        if safe_float(row["train_skip_rate"], 0.0) > 0.70:
            penalty += 0.10
        if safe_float(row["train_avg_trade"], 0.0) <= estimated_cost_per_trade * 1.5:
            penalty += 0.15
        if safe_float(row["max_year_contribution_pct_train"], 1.0) > 0.70:
            penalty += 0.10
        cluster_scores.append(cluster_positive_ratio)
        temporal_scores.append(temporal)
        penalties.append(penalty)
    frame["cluster_positive_ratio_train"] = cluster_scores
    frame["temporal_stability_train"] = temporal_scores
    frame["penalties"] = penalties
    frame["train_robust_score"] = (
        0.25 * frame["normalized_net_pnl"].fillna(0.0)
        + 0.20 * frame["normalized_profit_factor"].fillna(0.0)
        + 0.15 * frame["normalized_pnl_to_maxdd"].fillna(0.0)
        + 0.15 * frame["temporal_stability_train"].fillna(0.0)
        + 0.15 * frame["cluster_positive_ratio_train"].fillna(0.0)
        + 0.10 * frame["trade_count_score"].fillna(0.0)
        - frame["penalties"].fillna(0.0)
    )
    frame = frame.sort_values(
        ["train_robust_score", "train_net_pnl", "train_profit_factor", "train_avg_trade"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    frame["rank_train"] = np.arange(1, len(frame) + 1, dtype=int)
    frame["selected_in_fold"] = False
    if not frame.empty:
        frame.loc[0, "selected_in_fold"] = True
    return frame


def derive_survivor_verdict(
    *,
    net_pnl: float,
    profit_factor: float,
    positive_folds: int,
    trades: int,
    cluster_positive_ratio: float,
    max_drawdown: float,
    monthly_positive_ratio: float,
    improvement_vs_m2k_only: float | None = None,
    is_portfolio: bool = False,
) -> str:
    if net_pnl <= 0 or profit_factor < 1.10 or positive_folds < 3:
        return "reject"
    if (
        is_portfolio
        and improvement_vs_m2k_only is not None
        and improvement_vs_m2k_only > 0
        and profit_factor >= 1.25
        and positive_folds >= 4
        and cluster_positive_ratio >= 0.55
        and monthly_positive_ratio >= 0.55
        and max_drawdown >= -500.0
    ):
        return "candidate"
    if (
        net_pnl > 0
        and profit_factor >= 1.20
        and positive_folds >= 4
        and cluster_positive_ratio >= 0.50
        and monthly_positive_ratio >= 0.50
        and trades >= 20
        and max_drawdown >= -750.0
    ):
        return "watchlist"
    if net_pnl > 0 and profit_factor > 1.15 and positive_folds >= 3:
        return "weak_watchlist"
    return "reject"


def _monthly_positive_ratio(daily: pd.DataFrame) -> float:
    if daily.empty:
        return 0.0
    frame = daily.copy()
    frame["month"] = pd.to_datetime(frame["session_date"], errors="coerce").dt.to_period("M").astype(str)
    monthly = frame.groupby("month", as_index=False)["daily_pnl"].sum()
    return float((pd.to_numeric(monthly["daily_pnl"], errors="coerce") > 0).mean()) if not monthly.empty else 0.0


def build_diagnostic_posthoc_rows(strict_wfa_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if strict_wfa_summary.empty:
        return pd.DataFrame(columns=["result_type", "name", "deployable", "reason", "net_pnl", "profit_factor", "positive_folds"])
    positive = strict_wfa_summary.loc[
        (pd.to_numeric(strict_wfa_summary["total_test_net_pnl"], errors="coerce") > 0)
        & (pd.to_numeric(strict_wfa_summary["test_profit_factor"], errors="coerce") >= 1.0)
    ].copy()
    for _, row in positive.iterrows():
        rows.append(
            {
                "result_type": "posthoc_positive_sleeve",
                "name": f"{row['symbol']}_{row['signal_timeframe']}",
                "deployable": False,
                "reason": "selected_after_full_walkforward_observation",
                "net_pnl": float(row["total_test_net_pnl"]),
                "profit_factor": float(row["test_profit_factor"]),
                "positive_folds": int(row["positive_folds"]),
            }
        )
    return pd.DataFrame(rows)


def _clone_variant_with_time_stop(base_variant: VolumeClimaxPullbackV2Variant, timeframe: str, time_stop_bars: int) -> VolumeClimaxPullbackV2Variant:
    variant = _clone_variant_for_timeframe(base_variant, timeframe)
    payload = asdict(variant)
    payload["time_stop_bars"] = int(time_stop_bars)
    return VolumeClimaxPullbackV2Variant(**payload)


def _entity_monthly_pnl(entity_type: str, entity_id: str, daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["entity_type", "entity_id", "month", "pnl"])
    frame = daily.copy()
    frame["month"] = pd.to_datetime(frame["session_date"], errors="coerce").dt.to_period("M").astype(str)
    grouped = frame.groupby("month", as_index=False)["daily_pnl"].sum().rename(columns={"daily_pnl": "pnl"})
    grouped["entity_type"] = entity_type
    grouped["entity_id"] = entity_id
    return grouped[["entity_type", "entity_id", "month", "pnl"]]


def _entity_yearly_pnl(entity_type: str, entity_id: str, daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["entity_type", "entity_id", "year", "pnl"])
    frame = daily.copy()
    frame["year"] = pd.to_datetime(frame["session_date"], errors="coerce").dt.year
    grouped = frame.groupby("year", as_index=False)["daily_pnl"].sum().rename(columns={"daily_pnl": "pnl"})
    grouped["entity_type"] = entity_type
    grouped["entity_id"] = entity_id
    return grouped[["entity_type", "entity_id", "year", "pnl"]]


def _entity_dayofweek_pnl(entity_type: str, entity_id: str, events: pd.DataFrame) -> pd.DataFrame:
    executed = events.loc[events.get("executed", False)].copy()
    if executed.empty:
        return pd.DataFrame(columns=["entity_type", "entity_id", "day_of_week", "pnl", "trades"])
    executed["day_of_week"] = pd.to_datetime(executed["entry_time"], errors="coerce").dt.day_name()
    grouped = executed.groupby("day_of_week", as_index=False).agg(
        pnl=("net_pnl_usd", "sum"),
        trades=("net_pnl_usd", "count"),
    )
    grouped["entity_type"] = entity_type
    grouped["entity_id"] = entity_id
    return grouped[["entity_type", "entity_id", "day_of_week", "pnl", "trades"]]


def _entity_entry_hour_pnl(entity_type: str, entity_id: str, events: pd.DataFrame) -> pd.DataFrame:
    executed = events.loc[events.get("executed", False)].copy()
    if executed.empty:
        return pd.DataFrame(columns=["entity_type", "entity_id", "entry_hour", "pnl", "trades"])
    executed["entry_hour"] = pd.to_datetime(executed["entry_time"], errors="coerce").dt.hour
    grouped = executed.groupby("entry_hour", as_index=False).agg(
        pnl=("net_pnl_usd", "sum"),
        trades=("net_pnl_usd", "count"),
    )
    grouped["entity_type"] = entity_type
    grouped["entity_id"] = entity_id
    return grouped[["entity_type", "entity_id", "entry_hour", "pnl", "trades"]]


def _trade_concentration(entity_id: str, events: pd.DataFrame) -> dict[str, Any]:
    executed = events.loc[events.get("executed", False)].copy()
    pnl = pd.to_numeric(executed.get("net_pnl_usd"), errors="coerce").dropna().sort_values(ascending=False)
    total = float(pnl.sum()) if not pnl.empty else 0.0
    worst = pnl.sort_values(ascending=True)
    def _share(series: pd.Series, n: int) -> float:
        if total == 0 or series.empty:
            return np.nan
        return float(series.head(n).sum() / total)
    return {
        "entity_id": entity_id,
        "trade_count": int(len(pnl)),
        "total_pnl": total,
        "top1_trade_pnl": float(pnl.head(1).sum()) if not pnl.empty else 0.0,
        "top3_trade_pnl": float(pnl.head(3).sum()) if not pnl.empty else 0.0,
        "top5_trade_pnl": float(pnl.head(5).sum()) if not pnl.empty else 0.0,
        "top1_contribution_pct": _share(pnl, 1),
        "top3_contribution_pct": _share(pnl, 3),
        "top5_contribution_pct": _share(pnl, 5),
        "worst1_trade_pnl": float(worst.head(1).sum()) if not worst.empty else 0.0,
        "worst3_trade_pnl": float(worst.head(3).sum()) if not worst.empty else 0.0,
        "worst5_trade_pnl": float(worst.head(5).sum()) if not worst.empty else 0.0,
        "worst1_contribution_pct": _share(worst, 1),
        "worst3_contribution_pct": _share(worst, 3),
        "worst5_contribution_pct": _share(worst, 5),
    }


def evaluate_symbol(
    *,
    symbol: str,
    signal_timeframe: str,
    execution_timeframe: str,
    configs: Sequence[SurvivorAuditConfig],
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
    execution_model, instrument = build_execution_model_for_profile(symbol=symbol, profile_name="repo_realistic")
    estimated_cost_per_trade = float(execution_model.round_trip_fees(quantity=1))
    signal_bar_minutes = timeframe_to_minutes(signal_timeframe)

    config_frame = config_frame_from_universe(configs)
    events_by_config: dict[str, pd.DataFrame] = {}
    metrics_rows: list[dict[str, Any]] = []

    for config in configs:
        variant = _clone_variant_with_time_stop(base_variant, signal_timeframe, config.time_stop_bars)
        events = _simulate_config(
            config=config.to_intrabar_config(),
            signal_df=signal_df,
            minute_df=minute_df,
            variant=variant,
            execution_model=execution_model,
            point_value_usd=float(instrument.point_value_usd),
            tick_size=float(instrument.tick_size),
            signal_bar_minutes=signal_bar_minutes,
        )
        events["symbol"] = symbol
        events["signal_timeframe"] = signal_timeframe
        events["config_id"] = config.config_id
        events["family"] = config.family
        events["cluster_id"] = config.cluster_id
        events["variant_time_stop_bars"] = int(config.time_stop_bars)
        events_by_config[config.config_id] = events

        for scope, start_date, end_date in (
            ("is", split_info["is_start"], split_info["is_end"]),
            ("oos", split_info["oos_start"], split_info["oos_end"]),
            ("full", split_info["is_start"], split_info["oos_end"]),
        ):
            scoped = events if scope == "full" else events.loc[_period_mask(events, start_date, end_date)].copy()
            metrics = compute_extended_metrics(scoped, estimated_cost_per_trade=estimated_cost_per_trade)
            metrics_rows.append(
                {
                    "symbol": symbol,
                    "signal_timeframe": signal_timeframe,
                    "config_id": config.config_id,
                    "family": config.family,
                    "cluster_id": config.cluster_id,
                    "scope": scope,
                    **metrics,
                }
            )

    metrics_frame = pd.DataFrame(metrics_rows).merge(
        config_frame,
        on=["symbol", "signal_timeframe", "config_id", "family", "cluster_id"],
        how="left",
    )
    metrics_is = metrics_frame.loc[metrics_frame["scope"] == "is"].drop(columns=["scope"]).reset_index(drop=True)
    metrics_oos = metrics_frame.loc[metrics_frame["scope"] == "oos"].drop(columns=["scope"]).reset_index(drop=True)
    metrics_full = metrics_frame.loc[metrics_frame["scope"] == "full"].drop(columns=["scope"]).reset_index(drop=True)
    robustness = compute_survivor_robustness_scores(
        metrics_is,
        events_by_config,
        split_info=split_info,
        estimated_cost_per_trade=estimated_cost_per_trade,
    )
    robustness["rank_is"] = np.arange(1, len(robustness) + 1, dtype=int)
    metrics_oos = metrics_oos.sort_values(["net_pnl", "profit_factor"], ascending=[False, False]).reset_index(drop=True)
    metrics_oos["rank_oos"] = np.arange(1, len(metrics_oos) + 1, dtype=int)

    folds = build_walkforward_folds(minute_df["session_date"].dropna().tolist())
    selection_rows: list[pd.DataFrame] = []
    fold_breakdown_rows: list[dict[str, Any]] = []
    stitched_rows: list[pd.DataFrame] = []
    benchmark_rows: list[pd.DataFrame] = []
    fixed_fold_rows: list[dict[str, Any]] = []
    train_daily_by_fold_symbol: dict[tuple[str, str], pd.DataFrame] = {}
    test_daily_by_fold_symbol: dict[tuple[str, str], pd.DataFrame] = {}

    raw_benchmark_id = str(config_frame.loc[config_frame["family"] == "raw_hybrid", "config_id"].iloc[0])
    for fold in folds:
        ranking = compute_survivor_fold_train_ranking(
            symbol=symbol,
            fold=fold,
            config_frame=config_frame,
            events_by_config=events_by_config,
            estimated_cost_per_trade=estimated_cost_per_trade,
        )
        selection_rows.append(ranking)
        winner = select_fold_winner(ranking)
        selected_config_id = str(winner["config_id"])
        train_events = events_by_config[selected_config_id].loc[_period_mask(events_by_config[selected_config_id], fold.train_start, fold.train_end)].copy()
        test_events = events_by_config[selected_config_id].loc[_period_mask(events_by_config[selected_config_id], fold.test_start, fold.test_end)].copy()
        benchmark_events = events_by_config[raw_benchmark_id].loc[_period_mask(events_by_config[raw_benchmark_id], fold.test_start, fold.test_end)].copy()
        train_metrics = compute_extended_metrics(train_events, estimated_cost_per_trade=estimated_cost_per_trade)
        test_metrics = compute_extended_metrics(test_events, estimated_cost_per_trade=estimated_cost_per_trade)
        benchmark_metrics = compute_extended_metrics(benchmark_events, estimated_cost_per_trade=estimated_cost_per_trade)
        train_daily_by_fold_symbol[(fold.fold_id, symbol)] = _daily_returns(train_events)
        test_daily_by_fold_symbol[(fold.fold_id, symbol)] = _daily_returns(test_events)
        test_daily_by_fold_symbol[(f"{fold.fold_id}_raw", symbol)] = _daily_returns(benchmark_events)
        test_events["fold_id"] = fold.fold_id
        benchmark_events["fold_id"] = fold.fold_id
        stitched_rows.append(test_events)
        benchmark_rows.append(benchmark_events)
        fold_breakdown_rows.append(
            {
                "symbol": symbol,
                "signal_timeframe": signal_timeframe,
                "fold_id": fold.fold_id,
                "selected_config_id": selected_config_id,
                "selected_family": winner["family"],
                "selected_cluster_id": winner["cluster_id"],
                "train_robust_score": float(winner["train_robust_score"]),
                "train_net_pnl": float(train_metrics["net_pnl"]),
                "train_profit_factor": float(train_metrics["profit_factor"]),
                "train_trades": int(train_metrics["trades"]),
                "test_net_pnl": float(test_metrics["net_pnl"]),
                "test_profit_factor": float(test_metrics["profit_factor"]),
                "test_trades": int(test_metrics["trades"]),
                "test_win_rate": float(test_metrics["winrate"]),
                "test_avg_trade": float(test_metrics["avg_trade"]),
                "test_median_trade": float(test_metrics["median_trade"]),
                "test_max_drawdown": float(test_metrics["max_drawdown"]),
                "test_max_daily_drawdown": float(test_metrics["max_daily_drawdown"]),
                "test_fold_sharpe": float(test_metrics["fold_sharpe"]) if pd.notna(test_metrics["fold_sharpe"]) else np.nan,
                "test_active_days": int(test_metrics["active_days"]),
                "test_exposure_ratio": float(test_metrics["exposure_ratio"]),
                "test_avg_holding_minutes": float(test_metrics["avg_holding_minutes"]),
                "raw_hybrid_test_net_pnl": float(benchmark_metrics["net_pnl"]),
                "strict_train_only": True,
            }
        )

        for _, config_row in config_frame.iterrows():
            cid = str(config_row["config_id"])
            fixed_test_events = events_by_config[cid].loc[_period_mask(events_by_config[cid], fold.test_start, fold.test_end)].copy()
            fixed_metrics = compute_extended_metrics(fixed_test_events, estimated_cost_per_trade=estimated_cost_per_trade)
            fixed_fold_rows.append(
                {
                    "symbol": symbol,
                    "signal_timeframe": signal_timeframe,
                    "fold_id": fold.fold_id,
                    "config_id": cid,
                    "family": config_row["family"],
                    "cluster_id": config_row["cluster_id"],
                    "test_net_pnl": float(fixed_metrics["net_pnl"]),
                    "test_profit_factor": float(fixed_metrics["profit_factor"]),
                    "test_trades": int(fixed_metrics["trades"]),
                    "positive_test_fold": bool(safe_float(fixed_metrics["net_pnl"], 0.0) > 0),
                }
            )

    selection_frame = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    stitched_trades = pd.concat(stitched_rows, ignore_index=True) if stitched_rows else pd.DataFrame()
    benchmark_trades = pd.concat(benchmark_rows, ignore_index=True) if benchmark_rows else pd.DataFrame()
    strict_fold_breakdown = pd.DataFrame(fold_breakdown_rows)
    fixed_fold_frame = pd.DataFrame(fixed_fold_rows)

    stitched_metrics = compute_extended_metrics(stitched_trades, estimated_cost_per_trade=estimated_cost_per_trade)
    benchmark_metrics_same_windows = compute_extended_metrics(benchmark_trades, estimated_cost_per_trade=estimated_cost_per_trade)
    stitched_daily = _daily_returns(stitched_trades)
    benchmark_daily = _daily_returns(benchmark_trades)
    positive_folds = int((pd.to_numeric(strict_fold_breakdown.get("test_net_pnl"), errors="coerce") > 0).sum()) if not strict_fold_breakdown.empty else 0
    train_score_test_corr = (
        pd.to_numeric(strict_fold_breakdown["train_robust_score"], errors="coerce").corr(pd.to_numeric(strict_fold_breakdown["test_net_pnl"], errors="coerce"))
        if len(strict_fold_breakdown) >= 2
        else np.nan
    )
    monthly_positive_ratio = _monthly_positive_ratio(stitched_daily)

    fixed_summary = (
        fixed_fold_frame.groupby(["config_id", "family", "cluster_id"], as_index=False)
        .agg(
            fixed_wfa_net_pnl=("test_net_pnl", "sum"),
            fixed_positive_fold_ratio=("positive_test_fold", "mean"),
            fixed_positive_folds=("positive_test_fold", "sum"),
        )
        if not fixed_fold_frame.empty
        else pd.DataFrame(columns=["config_id", "family", "cluster_id", "fixed_wfa_net_pnl", "fixed_positive_fold_ratio", "fixed_positive_folds"])
    )

    local_stability = robustness.merge(
        metrics_oos[["config_id", "net_pnl", "profit_factor", "rank_oos"]].rename(
            columns={"net_pnl": "net_pnl_oos", "profit_factor": "profit_factor_oos"}
        ),
        on="config_id",
        how="left",
    ).merge(fixed_summary, on=["config_id", "family", "cluster_id"], how="left")
    selected_counts = strict_fold_breakdown["selected_config_id"].value_counts().to_dict() if not strict_fold_breakdown.empty else {}
    local_stability["strict_selected_count"] = local_stability["config_id"].map(lambda value: int(selected_counts.get(str(value), 0)))
    local_stability["neighbor_median_oos_pnl"] = np.nan
    local_stability["neighbor_positive_fold_ratio"] = np.nan
    for idx, row in local_stability.iterrows():
        neighbor_median, neighbor_ratio = _cluster_neighbor_stats(local_stability, row)
        local_stability.loc[idx, "neighbor_median_oos_pnl"] = neighbor_median
        local_stability.loc[idx, "neighbor_positive_fold_ratio"] = neighbor_ratio

    cluster_stability_summary = (
        local_stability.groupby(["symbol", "cluster_id", "family"], as_index=False)
        .agg(
            configs=("config_id", "count"),
            median_is_net_pnl=("net_pnl", "median"),
            median_oos_net_pnl=("net_pnl_oos", "median"),
            median_fixed_wfa_net_pnl=("fixed_wfa_net_pnl", "median"),
            pct_configs_positive_oos=("net_pnl_oos", lambda values: float((pd.to_numeric(pd.Series(values), errors="coerce") > 0).mean())),
            pct_configs_positive_fixed_wfa=("fixed_wfa_net_pnl", lambda values: float((pd.to_numeric(pd.Series(values), errors="coerce") > 0).mean())),
            median_neighbor_oos_pnl=("neighbor_median_oos_pnl", "median"),
            selected_in_any_fold=("strict_selected_count", lambda values: int((pd.Series(values) > 0).sum())),
        )
        if not local_stability.empty
        else pd.DataFrame()
    )

    cluster_positive_ratio = float(pd.to_numeric(cluster_stability_summary["pct_configs_positive_fixed_wfa"], errors="coerce").max()) if not cluster_stability_summary.empty else 0.0
    strict_summary_row = {
        "symbol": symbol,
        "signal_timeframe": signal_timeframe,
        "total_test_trades": int(stitched_metrics["trades"]),
        "total_test_net_pnl": float(stitched_metrics["net_pnl"]),
        "gross_profit": float(stitched_metrics["gross_profit"]),
        "gross_loss": float(stitched_metrics["gross_loss"]),
        "test_profit_factor": float(stitched_metrics["profit_factor"]),
        "test_win_rate": float(stitched_metrics["winrate"]),
        "avg_trade": float(stitched_metrics["avg_trade"]),
        "median_trade": float(stitched_metrics["median_trade"]),
        "max_drawdown": float(stitched_metrics["max_drawdown"]),
        "max_daily_drawdown": float(stitched_metrics["max_daily_drawdown"]),
        "positive_folds": int(positive_folds),
        "fold_count": int(len(folds)),
        "fold_sharpe": float(stitched_metrics["fold_sharpe"]) if pd.notna(stitched_metrics["fold_sharpe"]) else np.nan,
        "active_days": int(stitched_metrics["active_days"]),
        "exposure_ratio": float(stitched_metrics["exposure_ratio"]),
        "avg_holding_minutes": float(stitched_metrics["avg_holding_minutes"]),
        "long_trades": int(stitched_metrics["long_trades"]),
        "short_trades": int(stitched_metrics["short_trades"]),
        "long_pnl": float(stitched_metrics["long_pnl"]),
        "short_pnl": float(stitched_metrics["short_pnl"]),
        "monthly_positive_ratio": monthly_positive_ratio,
        "cluster_positive_ratio": cluster_positive_ratio,
        "selected_family_counts": json.dumps(strict_fold_breakdown["selected_family"].value_counts().to_dict(), sort_keys=True) if not strict_fold_breakdown.empty else "{}",
        "selected_cluster_counts": json.dumps(strict_fold_breakdown["selected_cluster_id"].value_counts().to_dict(), sort_keys=True) if not strict_fold_breakdown.empty else "{}",
        "train_score_test_corr": train_score_test_corr,
        "benchmark_raw_hybrid_net_pnl_same_windows": float(benchmark_metrics_same_windows["net_pnl"]),
        "improvement_vs_raw_hybrid": float(stitched_metrics["net_pnl"] - benchmark_metrics_same_windows["net_pnl"]),
    }
    strict_summary_row["verdict"] = derive_survivor_verdict(
        net_pnl=float(strict_summary_row["total_test_net_pnl"]),
        profit_factor=float(strict_summary_row["test_profit_factor"]),
        positive_folds=int(strict_summary_row["positive_folds"]),
        trades=int(strict_summary_row["total_test_trades"]),
        cluster_positive_ratio=float(strict_summary_row["cluster_positive_ratio"]),
        max_drawdown=float(strict_summary_row["max_drawdown"]),
        monthly_positive_ratio=float(strict_summary_row["monthly_positive_ratio"]),
    )

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
        "split_mode": split_info["split_mode"],
        "variant_name": signal_variant.name,
        "variant_seed_fallback": bool(used_fallback),
        "variant_seed_requested": desired_name,
        "estimated_cost_per_trade": estimated_cost_per_trade,
    }
    return {
        "symbol": symbol,
        "config_frame": config_frame,
        "metrics_is": metrics_is,
        "metrics_oos": metrics_oos,
        "metrics_full": metrics_full,
        "robustness": robustness,
        "local_stability": local_stability,
        "cluster_stability_summary": cluster_stability_summary,
        "config_selection_by_fold": selection_frame,
        "strict_wfa_fold_breakdown": strict_fold_breakdown,
        "strict_wfa_summary": pd.DataFrame([strict_summary_row]),
        "strict_wfa_stitched_trades": stitched_trades,
        "strict_wfa_stitched_daily": stitched_daily,
        "raw_benchmark_daily": benchmark_daily,
        "train_daily_by_fold_symbol": train_daily_by_fold_symbol,
        "test_daily_by_fold_symbol": test_daily_by_fold_symbol,
        "fixed_fold_frame": fixed_fold_frame,
        "data_audit": data_audit,
    }


def build_strict_portfolios(
    *,
    symbol_results: list[dict[str, Any]],
    include_symbols: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_map = {str(result["symbol"]): result for result in symbol_results}
    target_symbols = [symbol for symbol in include_symbols if symbol in result_map and symbol in {"M2K", "MGC"}]
    if not target_symbols:
        return pd.DataFrame(), pd.DataFrame()

    folds = sorted(
        {
            str(fold_id)
            for result in symbol_results
            for fold_id, _symbol in result["train_daily_by_fold_symbol"].keys()
            if not str(fold_id).endswith("_raw")
        }
    )
    portfolio_daily_rows: list[dict[str, Any]] = []
    for fold_id in folds:
        train_daily = {
            symbol: result_map[symbol]["train_daily_by_fold_symbol"].get((fold_id, symbol), pd.DataFrame())
            for symbol in target_symbols
        }
        test_daily = {
            symbol: result_map[symbol]["test_daily_by_fold_symbol"].get((fold_id, symbol), pd.DataFrame())
            for symbol in target_symbols
        }
        schemes = {
            "m2k_only": {"M2K": 1.0} if "M2K" in target_symbols else {},
            "mgc_only": {"MGC": 1.0} if "MGC" in target_symbols else {},
            "m2k_mgc_equal_weight": build_portfolio_weights_from_train(train_daily, scheme="equal_weight"),
            "m2k_mgc_inverse_vol": build_portfolio_weights_from_train(train_daily, scheme="inverse_vol"),
            "m2k_mgc_capped_equal": build_portfolio_weights_from_train(train_daily, scheme="capped_equal_weight"),
        }
        for portfolio_name, weights in schemes.items():
            if not weights:
                continue
            session_dates = sorted(
                {
                    str(session_date)
                    for symbol in weights
                    for session_date in pd.to_datetime(test_daily[symbol].get("session_date"), errors="coerce").dt.date.dropna().astype(str).tolist()
                }
            )
            for raw_date in session_dates:
                session_date = pd.Timestamp(raw_date).date()
                pnl = 0.0
                for symbol, weight in weights.items():
                    daily = test_daily.get(symbol, pd.DataFrame())
                    if daily.empty:
                        continue
                    match = daily.loc[pd.to_datetime(daily["session_date"], errors="coerce").dt.date == session_date].copy()
                    pnl += float(pd.to_numeric(match.get("daily_pnl"), errors="coerce").sum()) * float(weight)
                portfolio_daily_rows.append(
                    {
                        "portfolio_name": portfolio_name,
                        "fold_id": fold_id,
                        "session_date": session_date,
                        "daily_pnl": pnl,
                        "weights_json": json.dumps(weights, sort_keys=True),
                        "selection_basis": "strict_train_only",
                        "deployable": True,
                    }
                )
    portfolio_daily = pd.DataFrame(portfolio_daily_rows)
    if portfolio_daily.empty:
        return pd.DataFrame(), pd.DataFrame()
    portfolio_daily = portfolio_daily.sort_values(["portfolio_name", "session_date"]).reset_index(drop=True)
    summaries: list[dict[str, Any]] = []
    m2k_net = np.nan
    grouped = portfolio_daily.groupby("portfolio_name", sort=True)
    for portfolio_name, part in grouped:
        daily = part[["session_date", "daily_pnl"]].copy()
        daily["equity"] = pd.to_numeric(daily["daily_pnl"], errors="coerce").cumsum()
        daily["drawdown"] = daily["equity"] - daily["equity"].cummax()
        fold_pnl = part.groupby("fold_id", as_index=False)["daily_pnl"].sum().rename(columns={"daily_pnl": "fold_pnl"})
        fold_sharpe = float(np.sqrt(len(fold_pnl)) * fold_pnl["fold_pnl"].mean() / fold_pnl["fold_pnl"].std(ddof=0)) if len(fold_pnl) > 1 and fold_pnl["fold_pnl"].std(ddof=0) > 0 else np.nan
        positive_folds = int((pd.to_numeric(fold_pnl["fold_pnl"], errors="coerce") > 0).sum())
        gross_profit = float(pd.to_numeric(daily["daily_pnl"], errors="coerce").loc[pd.to_numeric(daily["daily_pnl"], errors="coerce") > 0].sum())
        gross_loss = float(abs(pd.to_numeric(daily["daily_pnl"], errors="coerce").loc[pd.to_numeric(daily["daily_pnl"], errors="coerce") < 0].sum()))
        net_pnl = float(pd.to_numeric(daily["daily_pnl"], errors="coerce").sum())
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else np.inf
        max_drawdown = float(daily["drawdown"].min()) if not daily.empty else 0.0
        monthly_positive_ratio = _monthly_positive_ratio(daily)
        current_weights = json.loads(str(part.iloc[-1]["weights_json"])) if not part.empty else {}
        cluster_positive_ratio = float(
            np.mean(
                [
                    safe_float(result_map[symbol]["strict_wfa_summary"].iloc[0]["cluster_positive_ratio"], 0.0)
                    for symbol in current_weights.keys()
                    if symbol in result_map
                ]
            )
        ) if not part.empty else 0.0
        if portfolio_name == "m2k_only":
            m2k_net = net_pnl
        summaries.append(
            {
                "portfolio_name": portfolio_name,
                "selection_basis": "strict_train_only",
                "deployable": True,
                "net_pnl": net_pnl,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "profit_factor": profit_factor,
                "positive_folds": positive_folds,
                "fold_count": int(len(fold_pnl)),
                "fold_sharpe": fold_sharpe,
                "max_drawdown": max_drawdown,
                "monthly_positive_ratio": monthly_positive_ratio,
                "weights_last_json": json.dumps(current_weights, sort_keys=True),
                "improvement_vs_m2k_only": np.nan,
                "cluster_positive_ratio": cluster_positive_ratio,
            }
        )
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary["improvement_vs_m2k_only"] = pd.to_numeric(summary["net_pnl"], errors="coerce") - safe_float(m2k_net, 0.0)
        verdicts: list[str] = []
        for _, row in summary.iterrows():
            verdicts.append(
                derive_survivor_verdict(
                    net_pnl=float(row["net_pnl"]),
                    profit_factor=float(row["profit_factor"]),
                    positive_folds=int(row["positive_folds"]),
                    trades=999,
                    cluster_positive_ratio=float(row["cluster_positive_ratio"]),
                    max_drawdown=float(row["max_drawdown"]),
                    monthly_positive_ratio=float(row["monthly_positive_ratio"]),
                    improvement_vs_m2k_only=float(row["improvement_vs_m2k_only"]),
                    is_portfolio=True,
                )
            )
        summary["verdict"] = verdicts
    portfolio_daily["equity"] = portfolio_daily.groupby("portfolio_name")["daily_pnl"].cumsum()
    return summary, portfolio_daily


def _build_final_report(
    *,
    output_dir: Path,
    strict_wfa_summary: pd.DataFrame,
    strict_wfa_fold_breakdown: pd.DataFrame,
    strict_portfolio_summary: pd.DataFrame,
    cluster_stability_summary: pd.DataFrame,
    diagnostic_rows: pd.DataFrame,
) -> None:
    m2k_row = strict_wfa_summary.loc[strict_wfa_summary["symbol"] == "M2K"].copy()
    mgc_row = strict_wfa_summary.loc[strict_wfa_summary["symbol"] == "MGC"].copy()
    mnq_row = strict_wfa_summary.loc[strict_wfa_summary["symbol"] == "MNQ"].copy()
    best_portfolio = strict_portfolio_summary.sort_values(["verdict", "net_pnl"], ascending=[False, False]).iloc[0] if not strict_portfolio_summary.empty else None
    deployable = bool(best_portfolio is not None and str(best_portfolio["verdict"]) == "candidate")
    if deployable:
        global_verdict = "candidate"
    elif not strict_portfolio_summary.empty and (strict_portfolio_summary["verdict"] == "watchlist").any():
        global_verdict = "watchlist"
    elif (strict_wfa_summary["verdict"] == "watchlist").any() or (strict_wfa_summary["verdict"] == "weak_watchlist").any():
        global_verdict = "watchlist"
    else:
        global_verdict = "reject"

    lines = [
        "# Volume Climax Pullback Survivor Audit",
        "",
        "## 1. Executive Summary",
        f"- M2K 1H strict survivor status: `{m2k_row.iloc[0]['verdict']}`." if not m2k_row.empty else "- M2K 1H not evaluated.",
        f"- MGC 1H instability diagnosis: `{mgc_row.iloc[0]['verdict']}` with `{int(mgc_row.iloc[0]['positive_folds'])}` positive folds." if not mgc_row.empty else "- MGC 1H not evaluated.",
        f"- Strict train-only portfolio status: `{best_portfolio['portfolio_name']} -> {best_portfolio['verdict']}`." if best_portfolio is not None else "- No strict portfolio built.",
        f"- Deployable conclusion: `{'candidate' if deployable else 'watchlist_only_or_reject'}`.",
        "",
        "## 2. Strict Train-Only WFA Summary",
        _markdown_table(strict_wfa_summary) if not strict_wfa_summary.empty else "No strict WFA summary.",
        "",
        "## 3. Fold Breakdown",
        _markdown_table(strict_wfa_fold_breakdown) if not strict_wfa_fold_breakdown.empty else "No fold breakdown.",
        "",
        "## 4. Cluster Stability",
        _markdown_table(cluster_stability_summary) if not cluster_stability_summary.empty else "No cluster stability rows.",
        "",
        "## 5. Strict Portfolio",
        _markdown_table(strict_portfolio_summary) if not strict_portfolio_summary.empty else "No strict portfolio summary.",
        "",
        "## 6. Negative Control",
        _markdown_table(mnq_row) if not mnq_row.empty else "Negative control skipped.",
        "",
        "## 7. Diagnostic Posthoc Results",
        _markdown_table(diagnostic_rows) if not diagnostic_rows.empty else "No diagnostic-only rows.",
        "",
        "## 8. Verdict",
        f"- Final strict verdict: `{global_verdict}`",
        "- Diagnostic posthoc rows, if present, are explicitly non-deployable and excluded from the promotion decision.",
        "",
        "## 9. Next Actions",
        "- Keep M2K 1H as watchlist only unless a strict portfolio earns a candidate verdict.",
        "- Treat MGC 1H as regime-fragile until fold dispersion improves materially.",
        "- Keep MNQ 1H as a negative control rather than a promotion target.",
        "- If no strict candidate emerges, stop standalone pullback promotion and keep only as overlay/feature research.",
        "- Stress any future watchlist sleeve with slippage shocks and live shadow monitoring before escalation.",
    ]
    (output_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_campaign(
    *,
    symbols: Sequence[str],
    signal_timeframe: str,
    execution_timeframe: str,
    output_root: Path,
    smoke: bool = False,
    skip_negative_control: bool = False,
    max_configs_per_family: int = 64,
    dataset_overrides: dict[str, pd.DataFrame] | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / f"volume_climax_pullback_survivor_audit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    include_negative_control = not skip_negative_control
    universe = build_candidate_universe(
        symbols=symbols,
        signal_timeframe=signal_timeframe,
        execution_timeframe=execution_timeframe,
        include_negative_control=include_negative_control,
        max_configs_per_family=6 if smoke else max_configs_per_family,
    )
    config_universe = config_frame_from_universe(universe)

    symbol_groups: dict[str, list[SurvivorAuditConfig]] = {}
    for config in universe:
        symbol_groups.setdefault(config.symbol, []).append(config)

    symbol_results: list[dict[str, Any]] = []
    for symbol, configs in sorted(symbol_groups.items()):
        LOGGER.info("Evaluating %s %s with %d configs", symbol, signal_timeframe, len(configs))
        override = None if dataset_overrides is None else dataset_overrides.get(symbol)
        symbol_results.append(
            evaluate_symbol(
                symbol=symbol,
                signal_timeframe=signal_timeframe,
                execution_timeframe=execution_timeframe,
                configs=configs,
                output_dir=output_dir,
                raw_minute_df_override=override,
            )
        )

    strict_wfa_summary = pd.concat([result["strict_wfa_summary"] for result in symbol_results], ignore_index=True) if symbol_results else pd.DataFrame()
    strict_wfa_fold_breakdown = pd.concat([result["strict_wfa_fold_breakdown"] for result in symbol_results], ignore_index=True) if symbol_results else pd.DataFrame()
    config_selection_by_fold = pd.concat([result["config_selection_by_fold"] for result in symbol_results], ignore_index=True) if symbol_results else pd.DataFrame()
    local_parameter_stability = pd.concat([result["local_stability"] for result in symbol_results], ignore_index=True) if symbol_results else pd.DataFrame()
    cluster_stability_summary = pd.concat([result["cluster_stability_summary"] for result in symbol_results], ignore_index=True) if symbol_results else pd.DataFrame()
    strict_trades = pd.concat([result["strict_wfa_stitched_trades"] for result in symbol_results], ignore_index=True) if symbol_results else pd.DataFrame()
    fixed_fold_frame = pd.concat([result["fixed_fold_frame"] for result in symbol_results], ignore_index=True) if symbol_results else pd.DataFrame()

    strict_portfolio_summary, strict_portfolio_daily = build_strict_portfolios(
        symbol_results=symbol_results,
        include_symbols=[symbol for symbol in symbols if symbol in {"M2K", "MGC"}],
    )
    diagnostic_rows = build_diagnostic_posthoc_rows(strict_wfa_summary)

    trade_concentration = pd.DataFrame(
        [_trade_concentration(str(row["symbol"]), strict_trades.loc[strict_trades["symbol"] == row["symbol"]].copy()) for _, row in strict_wfa_summary.iterrows()]
    ) if not strict_wfa_summary.empty else pd.DataFrame()

    monthly_rows: list[pd.DataFrame] = []
    yearly_rows: list[pd.DataFrame] = []
    day_rows: list[pd.DataFrame] = []
    hour_rows: list[pd.DataFrame] = []
    for _, row in strict_wfa_summary.iterrows():
        symbol = str(row["symbol"])
        symbol_events = strict_trades.loc[strict_trades["symbol"] == symbol].copy()
        symbol_daily = _daily_returns(symbol_events)
        monthly_rows.append(_entity_monthly_pnl("sleeve", symbol, symbol_daily))
        yearly_rows.append(_entity_yearly_pnl("sleeve", symbol, symbol_daily))
        day_rows.append(_entity_dayofweek_pnl("sleeve", symbol, symbol_events))
        hour_rows.append(_entity_entry_hour_pnl("sleeve", symbol, symbol_events))
    if not strict_portfolio_daily.empty:
        for portfolio_name, part in strict_portfolio_daily.groupby("portfolio_name", sort=True):
            daily = part[["session_date", "daily_pnl"]].copy()
            monthly_rows.append(_entity_monthly_pnl("portfolio", str(portfolio_name), daily))
            yearly_rows.append(_entity_yearly_pnl("portfolio", str(portfolio_name), daily))

    monthly_pnl = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame(columns=["entity_type", "entity_id", "month", "pnl"])
    yearly_pnl = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame(columns=["entity_type", "entity_id", "year", "pnl"])
    dayofweek_pnl = pd.concat(day_rows, ignore_index=True) if day_rows else pd.DataFrame(columns=["entity_type", "entity_id", "day_of_week", "pnl", "trades"])
    entry_hour_pnl = pd.concat(hour_rows, ignore_index=True) if hour_rows else pd.DataFrame(columns=["entity_type", "entity_id", "entry_hour", "pnl", "trades"])

    run_metadata = {
        "timestamp": timestamp,
        "symbols": list(symbols),
        "signal_timeframe": signal_timeframe,
        "execution_timeframe": execution_timeframe,
        "strict_protocol": "train_only_fold_selection",
        "smoke": bool(smoke),
        "skip_negative_control": bool(skip_negative_control),
        "max_configs_per_family": int(6 if smoke else max_configs_per_family),
        "python_version": sys.version,
        "platform": platform.platform(),
        "fixed_split": {
            "is_end": str(FIXED_SPLIT_IS_END),
            "oos_start": str(FIXED_SPLIT_OOS_START),
        },
        "input_files": {
            result["symbol"]: _file_metadata(Path(result["data_audit"]["source_path"]))
            for result in symbol_results
            if result["data_audit"].get("source_path")
        },
        "data_audits": [result["data_audit"] for result in symbol_results],
    }

    config_universe.to_csv(output_dir / "config_universe.csv", index=False)
    strict_wfa_summary.to_csv(output_dir / "strict_wfa_summary.csv", index=False)
    strict_wfa_fold_breakdown.to_csv(output_dir / "strict_wfa_fold_breakdown.csv", index=False)
    strict_portfolio_summary.to_csv(output_dir / "strict_portfolio_summary.csv", index=False)
    strict_portfolio_daily.to_csv(output_dir / "strict_portfolio_daily_returns.csv", index=False)
    config_selection_by_fold.to_csv(output_dir / "config_selection_by_fold.csv", index=False)
    local_parameter_stability.to_csv(output_dir / "local_parameter_stability.csv", index=False)
    cluster_stability_summary.to_csv(output_dir / "cluster_stability_summary.csv", index=False)
    trade_concentration.to_csv(output_dir / "trade_concentration.csv", index=False)
    monthly_pnl.to_csv(output_dir / "monthly_pnl.csv", index=False)
    yearly_pnl.to_csv(output_dir / "yearly_pnl.csv", index=False)
    dayofweek_pnl.to_csv(output_dir / "dayofweek_pnl.csv", index=False)
    entry_hour_pnl.to_csv(output_dir / "entry_hour_pnl.csv", index=False)
    diagnostic_rows.to_csv(output_dir / "rejected_or_diagnostic_results.csv", index=False)
    strict_trades.to_csv(output_dir / "strict_wfa_stitched_trades.csv", index=False)
    fixed_fold_frame.to_csv(output_dir / "fixed_config_fold_test_results.csv", index=False)
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    _build_final_report(
        output_dir=output_dir,
        strict_wfa_summary=strict_wfa_summary,
        strict_wfa_fold_breakdown=strict_wfa_fold_breakdown,
        strict_portfolio_summary=strict_portfolio_summary,
        cluster_stability_summary=cluster_stability_summary,
        diagnostic_rows=diagnostic_rows,
    )
    return output_dir


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--signal-timeframe", default=DEFAULT_SIGNAL_TIMEFRAME)
    parser.add_argument("--execution-timeframe", default=DEFAULT_EXECUTION_TIMEFRAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-negative-control", action="store_true")
    parser.add_argument("--max-configs-per-family", type=int, default=64)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    output_dir = run_campaign(
        symbols=list(args.symbols),
        signal_timeframe=str(args.signal_timeframe),
        execution_timeframe=str(args.execution_timeframe),
        output_root=Path(args.output_root),
        smoke=bool(args.smoke),
        skip_negative_control=bool(args.skip_negative_control),
        max_configs_per_family=int(args.max_configs_per_family),
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
