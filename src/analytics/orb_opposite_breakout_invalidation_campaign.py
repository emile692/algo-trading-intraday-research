"""Opposite-breakout invalidation campaign for long-only ORB.

This module keeps the central ORB execution engine unchanged and builds
session-level signal policies on top of the existing 1-minute ORB baseline.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.metrics import compute_metrics
from src.config.orb_campaign import build_prop_constraints
from src.config.paths import PROCESSED_DATA_DIR
from src.config.settings import DEFAULT_INITIAL_CAPITAL_USD, get_instrument_spec
from src.data.cleaning import clean_ohlcv
from src.data.loader import load_ohlcv_file
from src.engine.backtester import run_backtest
from src.engine.execution_model import ExecutionModel
from src.features.intraday import add_continuous_session_vwap, add_intraday_features, add_session_vwap
from src.features.opening_range import compute_opening_range
from src.strategy.orb import ORBStrategy

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "export"
DEFAULT_PROCESSED_DIR = PROCESSED_DATA_DIR / "parquet"
DEFAULT_OPENING_TIME = "09:30:00"
DEFAULT_TIME_EXIT = "16:00:00"
FAILED_BREAKDOWN_RECLAIM_TAG = "failed_breakdown_reclaim_long"
BASELINE_TAG = "orb_long_only"


@dataclass(frozen=True)
class OppositeBreakoutInvalidationSpec:
    """One policy variant tested by the campaign."""

    name: str
    description: str
    policy_family: str
    opposite_confirmation: str
    opposite_breakout_buffer_ticks: int = 0
    opposite_breakout_confirm_bars: int = 1
    require_reclaim_vwap: bool = False
    require_reclaim_or_low_close: bool = False
    reclaim_confirm_bars: int = 1
    strategy_tag: str = BASELINE_TAG


@dataclass(frozen=True)
class CampaignConfig:
    """Top-level campaign configuration."""

    symbols: tuple[str, ...] = ("MNQ", "MES", "M2K", "MGC")
    start_date: str | None = "2018-01-01"
    end_date: str | None = "2026-12-31"
    output_root: Path = DEFAULT_OUTPUT_ROOT
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    dataset_paths: dict[str, Path] | None = None
    smoke: bool = False
    opening_time: str = DEFAULT_OPENING_TIME
    or_minutes: int = 15
    entry_buffer_ticks: int = 2
    stop_buffer_ticks: int = 2
    target_multiple: float = 2.0
    direction: str = "long"
    one_trade_per_day: bool = True
    vwap_confirmation: bool = True
    vwap_column: str = "continuous_session_vwap"
    time_exit: str = DEFAULT_TIME_EXIT
    account_size_usd: float = DEFAULT_INITIAL_CAPITAL_USD
    risk_per_trade_pct: float = 0.50
    entry_on_next_open: bool = True
    smoke_max_sessions: int = 40


@dataclass
class PreparedAssetData:
    """Prepared feature and candidate data for one symbol."""

    symbol: str
    dataset_path: Path
    instrument_spec: dict[str, Any]
    feature_df: pd.DataFrame
    candidate_signal_df: pd.DataFrame
    session_dates: list[pd.Timestamp]


def _normalize_timeframe_tag(timeframe: str | None) -> str | None:
    if timeframe is None:
        return None
    clean = str(timeframe).strip().lower().replace(" ", "")
    if clean.endswith("min"):
        clean = f"{clean[:-3]}m"
    return clean or None


def _dataset_timeframe_tag(path: Path) -> str | None:
    parts = path.stem.split("_")
    if len(parts) >= 6:
        return _normalize_timeframe_tag(parts[3])
    return None


def resolve_processed_dataset(
    symbol: str,
    processed_dir: Path | None = None,
    timeframe: str | None = "1m",
) -> Path:
    """Return the latest processed parquet for a symbol."""
    root = processed_dir or DEFAULT_PROCESSED_DIR
    matches = sorted(root.glob(f"{symbol.upper()}_*.parquet"))
    wanted_timeframe = _normalize_timeframe_tag(timeframe)
    if wanted_timeframe is not None:
        matches = [path for path in matches if _dataset_timeframe_tag(path) == wanted_timeframe]
    if not matches:
        if wanted_timeframe is None:
            raise FileNotFoundError(f"No processed parquet found for {symbol} in {root}")
        raise FileNotFoundError(
            f"No processed parquet found for {symbol} with timeframe {wanted_timeframe} in {root}"
        )
    return matches[-1]


def build_policy_grid(smoke: bool = False) -> list[OppositeBreakoutInvalidationSpec]:
    """Return the policy grid requested by the research brief."""
    specs: list[OppositeBreakoutInvalidationSpec] = [
        OppositeBreakoutInvalidationSpec(
            name="baseline_no_opposite_invalidation",
            description="Current behavior: downside break before long does not invalidate the day.",
            policy_family="baseline",
            opposite_confirmation="none",
        )
    ]

    touch_buffers = (0,) if smoke else (0, 1, 2, 4)
    close_buffers = (0,) if smoke else (0, 1, 2, 4)
    nclose_buffers = (0,) if smoke else (0, 1, 2)
    nclose_confirms = (2,) if smoke else (2, 3, 5)
    close5_buffers = (0,) if smoke else (0, 1, 2, 4)
    reclaim_buffers = (0,) if smoke else (0, 1, 2)
    reclaim_vwap_values = (True,) if smoke else (True, False)
    strict_buffers = (0,) if smoke else (0, 1, 2)
    strict_confirms = (1,) if smoke else (1, 2, 3)

    for buffer_ticks in touch_buffers:
        specs.append(
            OppositeBreakoutInvalidationSpec(
                name=f"invalidate_on_opposite_touch__buffer_{buffer_ticks}",
                description="Invalidate the day as soon as a 1m low breaks below OR low minus buffer.",
                policy_family="invalidate_for_day",
                opposite_confirmation="touch",
                opposite_breakout_buffer_ticks=buffer_ticks,
            )
        )
    for buffer_ticks in close_buffers:
        specs.append(
            OppositeBreakoutInvalidationSpec(
                name=f"invalidate_on_opposite_close_1m__buffer_{buffer_ticks}",
                description="Invalidate the day on the first 1m close below OR low minus buffer.",
                policy_family="invalidate_for_day",
                opposite_confirmation="close_1m",
                opposite_breakout_buffer_ticks=buffer_ticks,
            )
        )
    for buffer_ticks in nclose_buffers:
        for confirm_bars in nclose_confirms:
            specs.append(
                OppositeBreakoutInvalidationSpec(
                    name=(
                        "invalidate_on_opposite_n_closes_1m"
                        f"__buffer_{buffer_ticks}__confirm_{confirm_bars}"
                    ),
                    description="Invalidate after N consecutive 1m closes below OR low minus buffer.",
                    policy_family="invalidate_for_day",
                    opposite_confirmation="n_closes_1m",
                    opposite_breakout_buffer_ticks=buffer_ticks,
                    opposite_breakout_confirm_bars=confirm_bars,
                )
            )
    for buffer_ticks in close5_buffers:
        specs.append(
            OppositeBreakoutInvalidationSpec(
                name=f"invalidate_on_opposite_close_5m__buffer_{buffer_ticks}",
                description="Invalidate on a completed 5m close below OR low minus buffer.",
                policy_family="invalidate_for_day",
                opposite_confirmation="close_5m",
                opposite_breakout_buffer_ticks=buffer_ticks,
            )
        )
    for buffer_ticks in reclaim_buffers:
        for require_vwap in reclaim_vwap_values:
            specs.append(
                OppositeBreakoutInvalidationSpec(
                    name=(
                        "allow_reclaim_after_opposite_breakout_conservative"
                        f"__buffer_{buffer_ticks}__require_vwap_{str(require_vwap).lower()}"
                    ),
                    description=(
                        "Allow a later long only after reclaiming OR low on close, optionally VWAP, "
                        "then breaking the long trigger."
                    ),
                    policy_family="reclaim_conservative",
                    opposite_confirmation="touch",
                    opposite_breakout_buffer_ticks=buffer_ticks,
                    require_reclaim_vwap=require_vwap,
                    require_reclaim_or_low_close=True,
                    reclaim_confirm_bars=1,
                    strategy_tag=FAILED_BREAKDOWN_RECLAIM_TAG,
                )
            )
    for buffer_ticks in strict_buffers:
        for confirm_bars in strict_confirms:
            specs.append(
                OppositeBreakoutInvalidationSpec(
                    name=(
                        "allow_reclaim_after_opposite_breakout_strict"
                        f"__buffer_{buffer_ticks}__reclaim_{confirm_bars}"
                    ),
                    description=(
                        "After downside break, require closes back inside the OR, then a close above VWAP, "
                        "then the long trigger."
                    ),
                    policy_family="reclaim_strict",
                    opposite_confirmation="touch",
                    opposite_breakout_buffer_ticks=buffer_ticks,
                    reclaim_confirm_bars=confirm_bars,
                    strategy_tag=FAILED_BREAKDOWN_RECLAIM_TAG,
                )
            )
    return specs


def _make_output_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_root / f"orb_opposite_breakout_invalidation_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _serialize(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _serialize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def detect_opening_range(df_1m: pd.DataFrame, config: CampaignConfig) -> pd.DataFrame:
    """Return the 1m frame enriched with OR values and post-OR eligibility."""
    out = add_intraday_features(df_1m)
    out = compute_opening_range(out, or_minutes=config.or_minutes, opening_time=config.opening_time)
    open_minute = pd.Timestamp(config.opening_time).hour * 60 + pd.Timestamp(config.opening_time).minute
    or_end_minute = open_minute + int(config.or_minutes)
    out["or_built"] = out["minute_of_day"] >= or_end_minute
    out["has_opening_range"] = out["or_high"].notna() & out["or_low"].notna()
    out["eligible_post_or"] = out["or_built"] & out["has_opening_range"]
    return out


def classify_first_breakout(
    df_1m: pd.DataFrame,
    or_high: float,
    or_low: float,
    config: CampaignConfig,
) -> dict[str, Any]:
    """Classify the first post-OR breakout direction using 1m chronology."""
    tick_size = float(getattr(config, "tick_size", 0.25))
    upside_threshold = float(or_high) + float(config.entry_buffer_ticks) * tick_size
    downside_buffer_ticks = int(getattr(config, "opposite_breakout_buffer_ticks", 0))
    downside_threshold = float(or_low) - float(downside_buffer_ticks) * tick_size
    working = df_1m.loc[df_1m["eligible_post_or"]].sort_values("timestamp")

    for _, row in working.iterrows():
        upside = float(row["high"]) >= upside_threshold
        downside = float(row["low"]) <= downside_threshold
        if upside and downside:
            return {
                "first_breakout": "ambiguous_same_bar",
                "first_breakout_ts": row["timestamp"],
                "first_upside_ts": row["timestamp"],
                "first_downside_ts": row["timestamp"],
            }
        if downside:
            return {
                "first_breakout": "first_breakout_downside",
                "first_breakout_ts": row["timestamp"],
                "first_upside_ts": pd.NaT,
                "first_downside_ts": row["timestamp"],
            }
        if upside:
            return {
                "first_breakout": "first_breakout_upside",
                "first_breakout_ts": row["timestamp"],
                "first_upside_ts": row["timestamp"],
                "first_downside_ts": pd.NaT,
            }

    return {
        "first_breakout": "no_breakout",
        "first_breakout_ts": pd.NaT,
        "first_upside_ts": pd.NaT,
        "first_downside_ts": pd.NaT,
    }


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return 0.0
    out = numerator / denominator
    return float(out) if math.isfinite(out) else 0.0


def _daily_pnl(trades: pd.DataFrame, session_dates: list[pd.Timestamp]) -> pd.Series:
    index = pd.Index(pd.to_datetime(pd.Index(session_dates)).date)
    if index.empty:
        return pd.Series(dtype=float)
    if trades.empty:
        return pd.Series(0.0, index=index, dtype=float)
    daily = trades.groupby(pd.to_datetime(trades["session_date"]).dt.date)["net_pnl_usd"].sum()
    return daily.reindex(index, fill_value=0.0).astype(float)


def _daily_sharpe(daily: pd.Series, capital: float) -> float:
    if len(daily) <= 1 or capital <= 0:
        return 0.0
    returns = daily.astype(float) / capital
    sigma = float(returns.std(ddof=0))
    if sigma <= 0:
        return 0.0
    return float((returns.mean() / sigma) * math.sqrt(252.0))


def _daily_sortino(daily: pd.Series, capital: float) -> float:
    if len(daily) <= 1 or capital <= 0:
        return 0.0
    returns = daily.astype(float) / capital
    downside = returns[returns < 0.0]
    if downside.empty:
        return float(math.sqrt(252.0) * returns.mean()) if len(returns) > 0 else 0.0
    sigma = float(downside.std(ddof=0))
    if sigma <= 0:
        return 0.0
    return float((returns.mean() / sigma) * math.sqrt(252.0))


def _build_strategy(config: CampaignConfig, tick_size: float) -> ORBStrategy:
    return ORBStrategy(
        or_minutes=config.or_minutes,
        direction=config.direction,
        one_trade_per_day=False,
        entry_buffer_ticks=config.entry_buffer_ticks,
        stop_buffer_ticks=config.stop_buffer_ticks,
        target_multiple=config.target_multiple,
        opening_time=config.opening_time,
        time_exit=config.time_exit,
        account_size_usd=config.account_size_usd,
        risk_per_trade_pct=config.risk_per_trade_pct,
        tick_size=tick_size,
        vwap_confirmation=config.vwap_confirmation,
        vwap_column=config.vwap_column,
    )


def _prepare_asset_data(symbol: str, config: CampaignConfig) -> PreparedAssetData:
    if config.dataset_paths and symbol in config.dataset_paths:
        dataset_path = Path(config.dataset_paths[symbol])
    else:
        dataset_path = resolve_processed_dataset(symbol, processed_dir=config.processed_dir, timeframe="1m")

    raw = clean_ohlcv(load_ohlcv_file(dataset_path))
    if config.start_date:
        raw = raw.loc[raw["timestamp"] >= pd.Timestamp(config.start_date, tz="America/New_York")]
    if config.end_date:
        raw = raw.loc[raw["timestamp"] <= pd.Timestamp(config.end_date, tz="America/New_York") + pd.Timedelta(days=1)]
    raw = raw.reset_index(drop=True)
    if raw.empty:
        raise ValueError(f"No rows left for {symbol} after date filtering.")

    feat = detect_opening_range(raw, config)
    feat = add_session_vwap(feat)
    feat = add_continuous_session_vwap(feat, session_start_hour=18)

    if config.smoke:
        keep_sessions = (
            pd.Index(pd.to_datetime(feat["session_date"])).sort_values().unique()[-int(config.smoke_max_sessions) :]
        )
        feat = feat.loc[pd.to_datetime(feat["session_date"]).isin(keep_sessions)].copy()

    instrument_spec = get_instrument_spec(symbol)
    strategy = _build_strategy(config, tick_size=float(instrument_spec["tick_size"]))
    candidate_signal_df = strategy.generate_signals(feat)
    candidate_signal_df["signal"] = 0

    session_dates = sorted(pd.to_datetime(candidate_signal_df["session_date"]).unique().tolist())
    return PreparedAssetData(
        symbol=symbol,
        dataset_path=dataset_path,
        instrument_spec=instrument_spec,
        feature_df=feat,
        candidate_signal_df=candidate_signal_df,
        session_dates=session_dates,
    )


def _completed_5m_close_map(session_df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=session_df.index, dtype=float)
    buckets = pd.to_datetime(session_df["timestamp"]).dt.floor("5min")
    grouped = session_df.groupby(buckets, sort=True)
    for _, bucket_df in grouped:
        last_idx = bucket_df.index[-1]
        out.loc[last_idx] = float(bucket_df["close"].iloc[-1])
    return out


def _find_first_true(mask: pd.Series, timestamps: pd.Series) -> pd.Timestamp | pd.NaT:
    hits = timestamps.loc[mask]
    if hits.empty:
        return pd.NaT
    return pd.Timestamp(hits.iloc[0])


def _find_opposite_breakout_event(
    session_df: pd.DataFrame,
    spec: OppositeBreakoutInvalidationSpec,
    config: CampaignConfig,
    or_low: float,
    tick_size: float,
) -> pd.Timestamp | pd.NaT:
    threshold = float(or_low) - float(spec.opposite_breakout_buffer_ticks) * float(tick_size)
    post = session_df.loc[session_df["eligible_post_or"]].copy()
    if post.empty or spec.opposite_confirmation == "none":
        return pd.NaT

    if spec.opposite_confirmation == "touch":
        return _find_first_true(post["low"] <= threshold, post["timestamp"])
    if spec.opposite_confirmation == "close_1m":
        return _find_first_true(post["close"] <= threshold, post["timestamp"])
    if spec.opposite_confirmation == "n_closes_1m":
        below = (post["close"] <= threshold).astype(int)
        streak = below.groupby((below == 0).cumsum()).cumsum()
        return _find_first_true(streak >= int(spec.opposite_breakout_confirm_bars), post["timestamp"])
    if spec.opposite_confirmation == "close_5m":
        completed_close = _completed_5m_close_map(post)
        return _find_first_true(completed_close <= threshold, post["timestamp"])
    raise ValueError(f"Unsupported opposite confirmation mode '{spec.opposite_confirmation}'.")


def _find_reclaim_after_downside(
    session_df: pd.DataFrame,
    spec: OppositeBreakoutInvalidationSpec,
    config: CampaignConfig,
    or_high: float,
    or_low: float,
) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT]:
    post = session_df.loc[session_df["eligible_post_or"]].copy()
    if post.empty:
        return pd.NaT, pd.NaT

    vwap_col = config.vwap_column if config.vwap_column in post.columns else "session_vwap"
    if spec.policy_family == "reclaim_conservative":
        reclaim_mask = post["close"] >= float(or_low) if spec.require_reclaim_or_low_close else post["high"] >= float(or_low)
        reclaim_ts = _find_first_true(reclaim_mask, post["timestamp"])
        if pd.isna(reclaim_ts):
            return pd.NaT, pd.NaT
        if spec.require_reclaim_vwap:
            vwap_post = post.loc[post["timestamp"] > reclaim_ts]
            vwap_ts = _find_first_true(vwap_post["close"] > vwap_post[vwap_col], vwap_post["timestamp"])
            return reclaim_ts, vwap_ts
        return reclaim_ts, reclaim_ts

    if spec.policy_family == "reclaim_strict":
        inside_mask = (post["close"] >= float(or_low)) & (post["close"] <= float(or_high))
        inside_streak = inside_mask.astype(int).groupby((inside_mask == 0).cumsum()).cumsum()
        reclaim_ts = _find_first_true(inside_streak >= int(spec.reclaim_confirm_bars), post["timestamp"])
        if pd.isna(reclaim_ts):
            return pd.NaT, pd.NaT
        vwap_post = post.loc[post["timestamp"] > reclaim_ts]
        vwap_ts = _find_first_true(vwap_post["close"] > vwap_post[vwap_col], vwap_post["timestamp"])
        return reclaim_ts, vwap_ts

    return pd.NaT, pd.NaT


def apply_opposite_invalidation_filter(
    candidate_signal_df: pd.DataFrame,
    spec: OppositeBreakoutInvalidationSpec,
    config: CampaignConfig,
    tick_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one policy to a candidate ORB frame and return filtered signals + session diagnostics."""
    out = candidate_signal_df.copy()
    out["signal"] = 0
    out["opposite_policy_name"] = spec.name
    out["invalidated_for_day"] = False
    out["is_reclaim_trade"] = False
    out["reclaim_ready"] = False
    out["policy_blocked_candidate"] = False

    session_rows: list[dict[str, Any]] = []
    for session_date, session_df in out.groupby("session_date", sort=True):
        session_df = session_df.sort_values("timestamp")
        or_high = session_df["or_high"].dropna()
        or_low = session_df["or_low"].dropna()
        if or_high.empty or or_low.empty:
            session_rows.append(
                {
                    "session_date": session_date,
                    "policy_name": spec.name,
                    "strategy_tag": spec.strategy_tag,
                    "first_breakout_side": "no_opening_range",
                    "first_upside_touch_ts": pd.NaT,
                    "first_downside_break_ts": pd.NaT,
                    "invalidated_for_day": False,
                    "invalidated_at": pd.NaT,
                    "reclaim_or_low_ts": pd.NaT,
                    "reclaim_vwap_ts": pd.NaT,
                    "selected_signal_ts": pd.NaT,
                    "selected_signal_type": "none",
                    "trade_taken": False,
                    "is_reclaim_trade": False,
                }
            )
            continue

        session_or_high = float(or_high.iloc[0])
        session_or_low = float(or_low.iloc[0])
        post = session_df.loc[session_df["eligible_post_or"]].copy()
        long_trigger = session_or_high + float(config.entry_buffer_ticks) * float(tick_size)
        upside_touch_ts = _find_first_true(post["high"] >= long_trigger, post["timestamp"])
        downside_break_ts = _find_opposite_breakout_event(
            session_df=session_df,
            spec=spec,
            config=config,
            or_low=session_or_low,
            tick_size=tick_size,
        )

        if not pd.isna(downside_break_ts) and (pd.isna(upside_touch_ts) or downside_break_ts <= upside_touch_ts):
            first_breakout_side = "first_breakout_downside"
        elif not pd.isna(upside_touch_ts):
            first_breakout_side = "first_breakout_upside"
        else:
            first_breakout_side = "no_breakout"

        passing = session_df.loc[(session_df["raw_signal"] == 1) & (session_df["filter_pass"])].copy()
        selected_idx: int | None = None
        reclaim_or_low_ts = pd.NaT
        reclaim_vwap_ts = pd.NaT
        invalidated_for_day = False
        invalidated_at = pd.NaT
        selected_signal_type = "none"
        is_reclaim_trade = False

        if spec.policy_family == "baseline":
            if not passing.empty:
                selected_idx = int(passing.index[0])
                selected_signal_type = "baseline"
        elif spec.policy_family == "invalidate_for_day":
            if pd.notna(downside_break_ts):
                invalidated_for_day = True
                invalidated_at = downside_break_ts
            allowed = passing if pd.isna(downside_break_ts) else passing.loc[passing["timestamp"] < downside_break_ts]
            blocked = passing.loc[passing["timestamp"] >= downside_break_ts] if pd.notna(downside_break_ts) else passing.iloc[0:0]
            if not blocked.empty:
                out.loc[blocked.index, "policy_blocked_candidate"] = True
            if not allowed.empty:
                selected_idx = int(allowed.index[0])
                selected_signal_type = "pre_invalidation_long"
        elif spec.policy_family in {"reclaim_conservative", "reclaim_strict"}:
            if pd.isna(downside_break_ts):
                if not passing.empty:
                    selected_idx = int(passing.index[0])
                    selected_signal_type = "baseline_no_downside_break"
            else:
                pre_downside = passing.loc[passing["timestamp"] < downside_break_ts]
                if not pre_downside.empty:
                    selected_idx = int(pre_downside.index[0])
                    selected_signal_type = "upside_before_downside"
                else:
                    reclaim_or_low_ts, reclaim_vwap_ts = _find_reclaim_after_downside(
                        session_df=session_df,
                        spec=spec,
                        config=config,
                        or_high=session_or_high,
                        or_low=session_or_low,
                    )
                    ready_ts = reclaim_vwap_ts
                    if pd.notna(ready_ts):
                        ready_candidates = passing.loc[passing["timestamp"] > ready_ts]
                        blocked = passing.loc[passing["timestamp"] <= ready_ts]
                        if not blocked.empty:
                            out.loc[blocked.index, "policy_blocked_candidate"] = True
                        if not ready_candidates.empty:
                            selected_idx = int(ready_candidates.index[0])
                            selected_signal_type = "reclaim_long"
                            is_reclaim_trade = True
                    else:
                        if not passing.empty:
                            out.loc[passing.index, "policy_blocked_candidate"] = True
        else:
            raise ValueError(f"Unsupported policy family '{spec.policy_family}'.")

        if selected_idx is not None:
            out.at[selected_idx, "signal"] = 1
            out.at[selected_idx, "is_reclaim_trade"] = bool(is_reclaim_trade)
            if pd.notna(reclaim_or_low_ts):
                ready_rows = session_df.loc[session_df["timestamp"] >= reclaim_or_low_ts].index
                out.loc[ready_rows, "reclaim_ready"] = True

        if invalidated_for_day and pd.notna(invalidated_at):
            invalid_rows = session_df.loc[session_df["timestamp"] >= invalidated_at].index
            out.loc[invalid_rows, "invalidated_for_day"] = True

        selected_signal_ts = pd.NaT if selected_idx is None else pd.Timestamp(out.at[selected_idx, "timestamp"])
        session_rows.append(
            {
                "session_date": session_date,
                "policy_name": spec.name,
                "strategy_tag": spec.strategy_tag,
                "first_breakout_side": first_breakout_side,
                "first_upside_touch_ts": upside_touch_ts,
                "first_downside_break_ts": downside_break_ts,
                "invalidated_for_day": bool(invalidated_for_day),
                "invalidated_at": invalidated_at,
                "reclaim_or_low_ts": reclaim_or_low_ts,
                "reclaim_vwap_ts": reclaim_vwap_ts,
                "selected_signal_ts": selected_signal_ts,
                "selected_signal_type": selected_signal_type,
                "trade_taken": selected_idx is not None,
                "is_reclaim_trade": bool(is_reclaim_trade),
            }
        )

    session_summary = pd.DataFrame(session_rows)
    return out, session_summary


def _run_backtest_for_symbol(
    signal_df: pd.DataFrame,
    config: CampaignConfig,
    instrument_spec: dict[str, Any],
) -> pd.DataFrame:
    execution_model = ExecutionModel(
        commission_per_side_usd=float(instrument_spec["commission_per_side_usd"]),
        slippage_ticks=float(instrument_spec["slippage_ticks"]),
        tick_size=float(instrument_spec["tick_size"]),
    )
    return run_backtest(
        signal_df,
        execution_model=execution_model,
        tick_value_usd=float(instrument_spec["tick_value_usd"]),
        point_value_usd=float(instrument_spec["point_value_usd"]),
        time_exit=config.time_exit,
        stop_buffer_ticks=config.stop_buffer_ticks,
        target_multiple=config.target_multiple,
        account_size_usd=config.account_size_usd,
        risk_per_trade_pct=config.risk_per_trade_pct,
        entry_on_next_open=config.entry_on_next_open,
    )


def _extended_metrics(
    trades: pd.DataFrame,
    signal_df: pd.DataFrame,
    session_summary: pd.DataFrame,
    session_dates: list[pd.Timestamp],
    config: CampaignConfig,
) -> dict[str, Any]:
    prop_constraints = build_prop_constraints()
    base = compute_metrics(
        trades,
        signal_df=signal_df,
        session_dates=session_dates,
        initial_capital=config.account_size_usd,
        prop_constraints=prop_constraints,
    )
    daily = _daily_pnl(trades, session_dates)
    daily_cum = daily.cumsum()
    daily_drawdown = daily_cum - daily_cum.cummax()
    gross_pnl = float(trades["pnl_usd"].sum()) if "pnl_usd" in trades.columns and not trades.empty else float(
        base.get("cumulative_pnl", 0.0) + trades.get("fees", pd.Series(dtype=float)).sum()
    )
    downside_sessions = session_summary.loc[session_summary["first_breakout_side"] == "first_breakout_downside", "session_date"]
    downside_dates = pd.Index(pd.to_datetime(downside_sessions).dt.date)
    downside_trades = (
        trades.loc[pd.Index(pd.to_datetime(trades["session_date"]).dt.date).isin(downside_dates)].copy()
        if not trades.empty
        else trades.copy()
    )
    downside_daily = (
        downside_trades.groupby(pd.to_datetime(downside_trades["session_date"]).dt.date)["net_pnl_usd"].sum().reindex(
            downside_dates, fill_value=0.0
        )
        if len(downside_dates) > 0
        else pd.Series(dtype=float)
    )
    reclaim_trades = trades.loc[trades.get("is_reclaim_trade", False).fillna(False)].copy() if not trades.empty else trades.copy()
    reclaim_dates = pd.Index(pd.to_datetime(reclaim_trades["session_date"]).dt.date.unique()) if not reclaim_trades.empty else pd.Index([])
    reclaim_daily = (
        reclaim_trades.groupby(pd.to_datetime(reclaim_trades["session_date"]).dt.date)["net_pnl_usd"].sum().reindex(
            reclaim_dates, fill_value=0.0
        )
        if len(reclaim_dates) > 0
        else pd.Series(dtype=float)
    )

    return {
        "net_pnl": float(base.get("cumulative_pnl", 0.0)),
        "gross_pnl": gross_pnl,
        "number_of_trades": int(base.get("n_trades", 0)),
        "win_rate": float(base.get("win_rate", 0.0)),
        "avg_trade": float(base.get("expectancy", 0.0)),
        "profit_factor": float(base.get("profit_factor", 0.0)),
        "Sharpe": float(base.get("sharpe_ratio", 0.0)),
        "Sortino": _daily_sortino(daily, config.account_size_usd),
        "max_drawdown": float(base.get("max_drawdown", 0.0)),
        "max_daily_drawdown": float(daily_drawdown.min()) if not daily_drawdown.empty else 0.0,
        "average_daily_pnl": float(daily.mean()) if not daily.empty else 0.0,
        "median_daily_pnl": float(daily.median()) if not daily.empty else 0.0,
        "worst_day": float(daily.min()) if not daily.empty else 0.0,
        "best_day": float(daily.max()) if not daily.empty else 0.0,
        "trade_frequency": _safe_div(float(base.get("n_trades", 0)), float(len(session_dates))),
        "exposure_days": int(pd.Index(pd.to_datetime(trades["session_date"]).dt.date).nunique()) if not trades.empty else 0,
        "prop_pass": bool(base.get("profit_target_reached_before_max_loss", False)),
        "daily_loss_limit_breaches": int(base.get("number_of_daily_loss_limit_breaches", 0)),
        "first_breakout_downside_count": int((session_summary["first_breakout_side"] == "first_breakout_downside").sum()),
        "first_breakout_upside_count": int((session_summary["first_breakout_side"] == "first_breakout_upside").sum()),
        "invalidated_days_count": int(session_summary["invalidated_for_day"].sum()),
        "invalidated_days_pct": _safe_div(float(session_summary["invalidated_for_day"].sum()), float(len(session_summary))),
        "trades_after_downside_first_breakout_count": int(len(downside_trades)),
        "pnl_after_downside_first_breakout": float(downside_trades["net_pnl_usd"].sum()) if not downside_trades.empty else 0.0,
        "sharpe_after_downside_first_breakout": _daily_sharpe(downside_daily, config.account_size_usd)
        if not downside_daily.empty
        else 0.0,
        "reclaim_trade_count": int(len(reclaim_trades)),
        "reclaim_trade_pnl": float(reclaim_trades["net_pnl_usd"].sum()) if not reclaim_trades.empty else 0.0,
        "reclaim_trade_sharpe": _daily_sharpe(reclaim_daily, config.account_size_usd) if not reclaim_daily.empty else 0.0,
    }


def _metrics_row(
    trades: pd.DataFrame,
    signal_df: pd.DataFrame,
    session_summary: pd.DataFrame,
    session_dates: list[pd.Timestamp],
    config: CampaignConfig,
    symbol: str,
    spec: OppositeBreakoutInvalidationSpec,
) -> dict[str, Any]:
    metrics = _extended_metrics(trades, signal_df, session_summary, session_dates, config)
    return {
        "config_name": spec.name,
        "policy_family": spec.policy_family,
        "strategy_tag": spec.strategy_tag,
        "symbol": symbol,
        "opposite_confirmation": spec.opposite_confirmation,
        "opposite_breakout_buffer_ticks": spec.opposite_breakout_buffer_ticks,
        "opposite_breakout_confirm_bars": spec.opposite_breakout_confirm_bars,
        "require_reclaim_vwap": spec.require_reclaim_vwap,
        "require_reclaim_or_low_close": spec.require_reclaim_or_low_close,
        "reclaim_confirm_bars": spec.reclaim_confirm_bars,
        **metrics,
    }


def run_single_asset_config(
    prepared: PreparedAssetData,
    spec: OppositeBreakoutInvalidationSpec,
    config: CampaignConfig,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Run one asset / one policy pair."""
    tick_size = float(prepared.instrument_spec["tick_size"])
    filtered_signal_df, session_summary = apply_opposite_invalidation_filter(
        candidate_signal_df=prepared.candidate_signal_df,
        spec=spec,
        config=config,
        tick_size=tick_size,
    )
    trades = _run_backtest_for_symbol(filtered_signal_df, config=config, instrument_spec=prepared.instrument_spec)
    if not trades.empty:
        trades = trades.copy()
        trades["symbol"] = prepared.symbol
        trades["config_name"] = spec.name
        trades["policy_family"] = spec.policy_family
        trades["strategy_tag"] = spec.strategy_tag
        trades = trades.merge(
            session_summary[
                [
                    "session_date",
                    "first_breakout_side",
                    "invalidated_for_day",
                    "invalidated_at",
                    "selected_signal_type",
                    "is_reclaim_trade",
                ]
            ],
            on="session_date",
            how="left",
        )
    metrics_row = _metrics_row(
        trades=trades,
        signal_df=filtered_signal_df,
        session_summary=session_summary,
        session_dates=prepared.session_dates,
        config=config,
        symbol=prepared.symbol,
        spec=spec,
    )
    session_summary = session_summary.copy()
    session_summary["symbol"] = prepared.symbol
    session_summary["config_name"] = spec.name
    signal_daily = _daily_pnl(trades, prepared.session_dates).rename("daily_pnl").reset_index()
    signal_daily = signal_daily.rename(columns={"index": "session_date"})
    signal_daily["session_date"] = pd.to_datetime(signal_daily["session_date"])
    signal_daily["symbol"] = prepared.symbol
    signal_daily["config_name"] = spec.name
    return {
        "metrics": metrics_row,
        "trades": trades,
        "session_summary": session_summary,
        "daily": signal_daily,
        "signal_df": filtered_signal_df,
    }


def _baseline_trade_decomposition(
    trades_df: pd.DataFrame,
    session_df: pd.DataFrame,
) -> pd.DataFrame:
    baseline_sessions = session_df.loc[session_df["config_name"] == "baseline_no_opposite_invalidation"].copy()
    baseline_trades = trades_df.loc[trades_df["config_name"] == "baseline_no_opposite_invalidation"].copy()
    if baseline_trades.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "bucket",
                "trade_count",
                "net_pnl",
                "avg_trade",
                "win_rate",
                "sharpe",
            ]
        )

    if "first_breakout_side" in baseline_trades.columns:
        merged = baseline_trades.copy()
    else:
        merged = baseline_trades.merge(
            baseline_sessions[["symbol", "session_date", "first_breakout_side", "selected_signal_type"]],
            on=["symbol", "session_date"],
            how="left",
        )
    merged["bucket"] = np.where(
        merged["first_breakout_side"].eq("first_breakout_downside"),
        "first_breakout_downside_then_reclaim",
        "first_breakout_upside",
    )
    merged["has_prior_downside_breakout"] = merged["first_breakout_side"].eq("first_breakout_downside")

    rows: list[dict[str, Any]] = []
    for symbol, group in merged.groupby("symbol", sort=True):
        for bucket_name, bucket_df in group.groupby("bucket", sort=True):
            daily = bucket_df.groupby(pd.to_datetime(bucket_df["session_date"]).dt.date)["net_pnl_usd"].sum()
            rows.append(
                {
                    "symbol": symbol,
                    "bucket": bucket_name,
                    "trade_count": int(len(bucket_df)),
                    "net_pnl": float(bucket_df["net_pnl_usd"].sum()),
                    "avg_trade": float(bucket_df["net_pnl_usd"].mean()),
                    "win_rate": float((bucket_df["net_pnl_usd"] > 0).mean()),
                    "sharpe": _daily_sharpe(daily, DEFAULT_INITIAL_CAPITAL_USD),
                }
            )
        no_prior = group.loc[~group["has_prior_downside_breakout"]]
        daily = no_prior.groupby(pd.to_datetime(no_prior["session_date"]).dt.date)["net_pnl_usd"].sum()
        rows.append(
            {
                "symbol": symbol,
                "bucket": "no_downside_breakout_before_trade",
                "trade_count": int(len(no_prior)),
                "net_pnl": float(no_prior["net_pnl_usd"].sum()) if not no_prior.empty else 0.0,
                "avg_trade": float(no_prior["net_pnl_usd"].mean()) if not no_prior.empty else 0.0,
                "win_rate": float((no_prior["net_pnl_usd"] > 0).mean()) if not no_prior.empty else 0.0,
                "sharpe": _daily_sharpe(daily, DEFAULT_INITIAL_CAPITAL_USD) if not daily.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _trade_decomposition(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "config_name",
                "symbol",
                "bucket",
                "trade_count",
                "net_pnl",
                "avg_trade",
                "win_rate",
            ]
        )
    working = trades_df.copy()
    working["bucket"] = np.where(
        working["is_reclaim_trade"].fillna(False),
        "reclaim_trade",
        np.where(
            working["first_breakout_side"].eq("first_breakout_downside"),
            "trade_after_downside_first_breakout",
            "trade_without_prior_downside_breakout",
        ),
    )
    rows: list[dict[str, Any]] = []
    for (config_name, symbol, bucket), group in working.groupby(["config_name", "symbol", "bucket"], sort=True):
        rows.append(
            {
                "config_name": config_name,
                "symbol": symbol,
                "bucket": bucket,
                "trade_count": int(len(group)),
                "net_pnl": float(group["net_pnl_usd"].sum()),
                "avg_trade": float(group["net_pnl_usd"].mean()),
                "win_rate": float((group["net_pnl_usd"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _results_by_year(
    trades_df: pd.DataFrame,
    session_df: pd.DataFrame,
    config: CampaignConfig,
) -> pd.DataFrame:
    if session_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped_sessions = session_df.copy()
    grouped_sessions["year"] = pd.to_datetime(grouped_sessions["session_date"]).dt.year

    for (config_name, symbol, year), sess in grouped_sessions.groupby(["config_name", "symbol", "year"], sort=True):
        session_dates = sorted(pd.to_datetime(sess["session_date"]).unique().tolist())
        trades = trades_df.loc[
            (trades_df["config_name"] == config_name)
            & (trades_df["symbol"] == symbol)
            & (pd.to_datetime(trades_df["session_date"]).dt.year == int(year))
        ].copy()
        metrics = _extended_metrics(
            trades=trades,
            signal_df=pd.DataFrame(),
            session_summary=sess,
            session_dates=session_dates,
            config=config,
        )
        rows.append(
            {
                "config_name": config_name,
                "symbol": symbol,
                "year": int(year),
                "annual_trade_count": int(len(trades)),
                "annual_max_dd": float(metrics["max_drawdown"]),
                "annual_invalidation_count": int(sess["invalidated_for_day"].sum()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _split_label(session_date: pd.Timestamp) -> str:
    year = int(pd.Timestamp(session_date).year)
    if year <= 2021:
        return "is_2018_2021"
    if year <= 2023:
        return "validation_2022_2023"
    return "oos_2024_2026"


def _aggregate_config_results(
    results_by_symbol: pd.DataFrame,
    trades_df: pd.DataFrame,
    session_df: pd.DataFrame,
    config: CampaignConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if results_by_symbol.empty:
        return pd.DataFrame()

    for config_name, spec_rows in results_by_symbol.groupby("config_name", sort=True):
        sess = session_df.loc[session_df["config_name"] == config_name].copy()
        all_sessions = sorted(pd.to_datetime(sess["session_date"]).unique().tolist())
        trades = trades_df.loc[trades_df["config_name"] == config_name].copy()
        daily = (
            trades.groupby(pd.to_datetime(trades["session_date"]).dt.date)["net_pnl_usd"].sum().reindex(
                pd.Index(pd.to_datetime(pd.Index(all_sessions)).date),
                fill_value=0.0,
            )
            if all_sessions
            else pd.Series(dtype=float)
        )
        merged_metrics = _extended_metrics(
            trades=trades,
            signal_df=pd.DataFrame(),
            session_summary=sess,
            session_dates=all_sessions,
            config=config,
        )

        split_series = pd.Series({_split_label(ts): None for ts in all_sessions})
        split_metrics: dict[str, float] = {}
        for split in split_series.index:
            split_sessions = [ts for ts in all_sessions if _split_label(ts) == split]
            split_trades = trades.loc[pd.to_datetime(trades["session_date"]).isin(pd.to_datetime(split_sessions))].copy()
            split_sess = sess.loc[pd.to_datetime(sess["session_date"]).isin(pd.to_datetime(split_sessions))].copy()
            if not split_sessions:
                continue
            metrics = _extended_metrics(
                trades=split_trades,
                signal_df=pd.DataFrame(),
                session_summary=split_sess,
                session_dates=split_sessions,
                config=config,
            )
            split_metrics[f"{split}_net_pnl"] = float(metrics["net_pnl"])
            split_metrics[f"{split}_Sharpe"] = float(metrics["Sharpe"])
            split_metrics[f"{split}_max_drawdown"] = float(metrics["max_drawdown"])
            split_metrics[f"{split}_number_of_trades"] = int(metrics["number_of_trades"])

        rows.append(
            {
                "config_name": config_name,
                "policy_family": str(spec_rows["policy_family"].iloc[0]),
                "strategy_tag": str(spec_rows["strategy_tag"].iloc[0]),
                "opposite_confirmation": str(spec_rows["opposite_confirmation"].iloc[0]),
                "opposite_breakout_buffer_ticks": int(spec_rows["opposite_breakout_buffer_ticks"].iloc[0]),
                "opposite_breakout_confirm_bars": int(spec_rows["opposite_breakout_confirm_bars"].iloc[0]),
                "require_reclaim_vwap": bool(spec_rows["require_reclaim_vwap"].iloc[0]),
                "require_reclaim_or_low_close": bool(spec_rows["require_reclaim_or_low_close"].iloc[0]),
                "reclaim_confirm_bars": int(spec_rows["reclaim_confirm_bars"].iloc[0]),
                "symbols_covered": int(spec_rows["symbol"].nunique()),
                "assets_profitable_count": int((spec_rows["net_pnl"] > 0).sum()),
                "asset_stability_score": float((spec_rows["Sharpe"] > 0).mean()),
                "annual_stability_score": float((results_by_symbol.loc[results_by_symbol["config_name"] == config_name, "net_pnl"] > 0).mean()),
                "trade_count_score": min(1.0, _safe_div(float(len(trades)), 120.0)),
                **merged_metrics,
                **split_metrics,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    baseline_row = out.loc[out["config_name"] == "baseline_no_opposite_invalidation"]
    baseline_oos_sharpe = float(baseline_row["oos_2024_2026_Sharpe"].iloc[0]) if not baseline_row.empty else 0.0
    baseline_oos_pnl = float(baseline_row["oos_2024_2026_net_pnl"].iloc[0]) if not baseline_row.empty else 0.0
    baseline_pf = float(baseline_row["profit_factor"].iloc[0]) if not baseline_row.empty else 0.0
    baseline_dd = float(baseline_row["max_drawdown"].iloc[0]) if not baseline_row.empty else 0.0
    baseline_breaches = int(baseline_row["daily_loss_limit_breaches"].iloc[0]) if not baseline_row.empty else 0
    baseline_trades = trades_df.loc[trades_df["config_name"] == "baseline_no_opposite_invalidation"].copy()

    delta_rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        cfg_name = str(row["config_name"])
        cfg_trades = trades_df.loc[trades_df["config_name"] == cfg_name].copy()
        baseline_only = baseline_trades.merge(
            cfg_trades[["symbol", "session_date"]],
            on=["symbol", "session_date"],
            how="left",
            indicator=True,
        )
        removed = baseline_only.loc[baseline_only["_merge"] == "left_only"]
        pnl_delta_vs_baseline_score = math.tanh(
            _safe_div(float(row.get("oos_2024_2026_net_pnl", 0.0)) - baseline_oos_pnl, max(abs(baseline_oos_pnl), 1.0))
        )
        max_dd_norm = _safe_div(abs(float(row.get("oos_2024_2026_max_drawdown", row["max_drawdown"]))), max(abs(float(row.get("oos_2024_2026_net_pnl", row["net_pnl"]))), 1.0))
        robust_score = (
            float(row.get("oos_2024_2026_Sharpe", row["Sharpe"]))
            - 0.5 * max_dd_norm
            + 0.25 * float(row["annual_stability_score"])
            + 0.25 * float(row["trade_count_score"])
            + 0.25 * pnl_delta_vs_baseline_score
            + 0.25 * float(row["asset_stability_score"])
        )
        delta_rows.append(
            {
                "config_name": cfg_name,
                "baseline_pnl": baseline_oos_pnl,
                "filtered_pnl": float(row.get("oos_2024_2026_net_pnl", 0.0)),
                "pnl_removed_by_invalidation": float(removed["net_pnl_usd"].sum()) if not removed.empty else 0.0,
                "trades_removed": int(len(removed)),
                "Sharpe_delta": float(row.get("oos_2024_2026_Sharpe", 0.0) - baseline_oos_sharpe),
                "DD_delta": float(row["max_drawdown"] - baseline_dd),
                "daily_loss_breach_delta": int(row["daily_loss_limit_breaches"] - baseline_breaches),
                "profit_factor_delta": float(row["profit_factor"] - baseline_pf),
                "pnl_delta_vs_baseline_score": pnl_delta_vs_baseline_score,
                "robust_score": robust_score,
            }
        )

    deltas = pd.DataFrame(delta_rows)
    return out.merge(deltas, on="config_name", how="left")


def build_summary_tables(campaign_result: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Build all export tables from raw campaign outputs."""
    return {
        "config_grid": campaign_result["config_grid"],
        "results_by_symbol": campaign_result["results_by_symbol"],
        "results_by_config": campaign_result["results_by_config"],
        "results_by_year": campaign_result["results_by_year"],
        "baseline_decomposition": campaign_result["baseline_decomposition"],
        "opposite_breakout_trade_decomposition": campaign_result["opposite_breakout_trade_decomposition"],
        "daily_returns_by_config": campaign_result["daily_returns_by_config"],
        "trades_by_config": campaign_result["trades_by_config"],
        "invalidated_days": campaign_result["invalidated_days"],
        "ranking_robust": campaign_result["ranking_robust"],
    }


def _yaml_policy_snippet(row: pd.Series) -> str:
    if str(row["strategy_tag"]) == FAILED_BREAKDOWN_RECLAIM_TAG:
        return (
            "opposite_breakout_policy:\n"
            "  enabled: true\n"
            "  mode: reclaim_required\n"
            f"  confirmation: {row['opposite_confirmation']}\n"
            f"  buffer_ticks: {int(row['opposite_breakout_buffer_ticks'])}\n"
            f"  require_reclaim_or_low: {str(bool(row['require_reclaim_or_low_close'])).lower()}\n"
            f"  require_reclaim_vwap: {str(bool(row['require_reclaim_vwap'])).lower()}\n"
            f"  reclaim_confirm_bars: {int(row['reclaim_confirm_bars'])}\n"
        )
    return (
        "opposite_breakout_policy:\n"
        "  enabled: true\n"
        "  mode: invalidate_for_day\n"
        f"  confirmation: {row['opposite_confirmation']}\n"
        f"  buffer_ticks: {int(row['opposite_breakout_buffer_ticks'])}\n"
        f"  confirm_bars: {int(row['opposite_breakout_confirm_bars'])}\n"
        "  allow_reclaim: false\n"
    )


def write_markdown_report(
    output_dir: Path,
    campaign_result: dict[str, Any],
) -> Path:
    """Write the final markdown report."""
    ranking = campaign_result["ranking_robust"]
    results_by_symbol = campaign_result["results_by_symbol"]
    baseline_decomp = campaign_result["baseline_decomposition"]
    results_by_year = campaign_result["results_by_year"]
    best_row = ranking.iloc[0] if not ranking.empty else None
    best_reclaim = ranking.loc[ranking["strategy_tag"] == FAILED_BREAKDOWN_RECLAIM_TAG].head(1)
    best_invalidation = ranking.loc[
        (ranking["policy_family"] == "invalidate_for_day")
        & (ranking["config_name"] != "baseline_no_opposite_invalidation")
    ].head(1)

    verdict = "inconclusive"
    if best_row is not None:
        if str(best_row["config_name"]) == "baseline_no_opposite_invalidation":
            verdict = "keep baseline"
        elif str(best_row["strategy_tag"]) == FAILED_BREAKDOWN_RECLAIM_TAG:
            verdict = "use reclaim strategy separately"
        elif str(best_row["policy_family"]) == "invalidate_for_day":
            verdict = "invalidate on opposite breakout"

    best_asset_lines = []
    if not results_by_symbol.empty:
        for symbol, group in results_by_symbol.groupby("symbol", sort=True):
            top = group.sort_values(["Sharpe", "net_pnl"], ascending=[False, False]).iloc[0]
            best_asset_lines.append(
                f"- {symbol}: `{top['config_name']}` | net `{top['net_pnl']:.2f}` | Sharpe `{top['Sharpe']:.3f}` | DD `{top['max_drawdown']:.2f}`"
            )

    baseline_lines = ["- No baseline trades to decompose."] if baseline_decomp.empty else [
        f"- {row['symbol']} / {row['bucket']}: trades {int(row['trade_count'])}, net {row['net_pnl']:.2f}, Sharpe {row['sharpe']:.3f}"
        for _, row in baseline_decomp.iterrows()
    ]

    impact_lines = ["- Comparison unavailable."] if ranking.empty else [
        f"- {row['config_name']}: baseline pnl {row['baseline_pnl']:.2f}, filtered pnl {row['filtered_pnl']:.2f}, "
        f"removed pnl {row['pnl_removed_by_invalidation']:.2f}, trades removed {int(row['trades_removed'])}, "
        f"Sharpe delta {row['Sharpe_delta']:.3f}, DD delta {row['DD_delta']:.2f}, "
        f"daily loss breach delta {int(row['daily_loss_breach_delta'])}, PF delta {row['profit_factor_delta']:.3f}"
        for _, row in ranking.head(5).iterrows()
    ]

    yearly_lines = ["- No yearly rows available."] if results_by_year.empty else [
        f"- {row['config_name']} / {row['symbol']} / {int(row['year'])}: net {row['net_pnl']:.2f}, trades {int(row['annual_trade_count'])}, "
        f"maxDD {row['annual_max_dd']:.2f}, invalidations {int(row['annual_invalidation_count'])}"
        for _, row in results_by_year.head(20).iterrows()
    ]

    prop_lines = ["- No prop summary available."] if ranking.empty else [
        f"- {row['config_name']}: prop_pass={bool(row['prop_pass'])}, daily_loss_limit_breaches={int(row['daily_loss_limit_breaches'])}"
        for _, row in ranking.head(5).iterrows()
    ]

    recommendation_lines = [
        "- Keep the execution repo ORB baseline unchanged unless the selected policy improves OOS Sharpe and drawdown together.",
        "- If the best reclaim variant remains competitive, wire it as a separate strategy family rather than mixing it into the ORB quality filter.",
    ]

    best_yaml = "No recommendation available."
    if best_row is not None:
        best_yaml = _yaml_policy_snippet(best_row)

    reclaim_yaml = "No reclaim candidate available."
    if not best_reclaim.empty:
        reclaim_yaml = _yaml_policy_snippet(best_reclaim.iloc[0])

    lines = [
        "# ORB Opposite Breakout Invalidation Campaign",
        "",
        "## 1. Executive summary",
        "",
        f"- Symbols tested: `{', '.join(campaign_result['config'].symbols)}`",
        f"- Period requested: `{campaign_result['config'].start_date}` -> `{campaign_result['config'].end_date}`",
        f"- Policies tested: {len(campaign_result['config_grid'])}",
        f"- Verdict: **{verdict}**",
        "",
        "## 2. Verdict clair",
        "",
        f"- {verdict}",
        "",
        "## 3. Best config overall",
        "",
    ]
    if best_row is not None:
        lines.extend(
            [
                f"- Config: `{best_row['config_name']}`",
                f"- Family: `{best_row['policy_family']}`",
                f"- Strategy tag: `{best_row['strategy_tag']}`",
                f"- OOS Sharpe: {float(best_row.get('oos_2024_2026_Sharpe', 0.0)):.3f}",
                f"- OOS net pnl: {float(best_row.get('oos_2024_2026_net_pnl', 0.0)):.2f}",
                f"- Robust score: {float(best_row.get('robust_score', 0.0)):.3f}",
            ]
        )
    else:
        lines.append("- No best config available.")

    lines.extend(
        [
            "",
            "## 4. Best config per asset",
            "",
            *best_asset_lines,
            "",
            "## 5. Baseline decomposition",
            "",
            *baseline_lines,
            "",
            "## 6. Impact of removing trades after downside first breakout",
            "",
            *impact_lines,
            "",
            "## 7. Robustness by year",
            "",
            *yearly_lines,
            "",
            "## 8. Prop-firm risk view",
            "",
            *prop_lines,
            "",
            "## 9. Recommendation for live execution repo",
            "",
            *recommendation_lines,
            "",
            "## 10. Suggested YAML fields for execution config",
            "",
            "### Best overall policy",
            "",
            "```yaml",
            best_yaml.rstrip(),
            "```",
            "",
            "### Best reclaim policy",
            "",
            "```yaml",
            reclaim_yaml.rstrip(),
            "```",
        ]
    )
    report_path = output_dir / "final_report.md"
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def run_campaign(config: CampaignConfig) -> dict[str, Any]:
    """Execute the full campaign and export all required artifacts."""
    output_dir = _make_output_dir(Path(config.output_root))
    policy_grid = build_policy_grid(smoke=config.smoke)
    prepared_assets = [_prepare_asset_data(symbol, config) for symbol in config.symbols]

    metrics_rows: list[dict[str, Any]] = []
    trades_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    session_frames: list[pd.DataFrame] = []

    for spec in policy_grid:
        for prepared in prepared_assets:
            result = run_single_asset_config(prepared=prepared, spec=spec, config=config)
            metrics_rows.append(result["metrics"])
            trades_frames.append(result["trades"])
            daily_frames.append(result["daily"])
            session_frames.append(result["session_summary"])

    results_by_symbol = pd.DataFrame(metrics_rows)
    trades_by_config = pd.concat(trades_frames, ignore_index=True) if trades_frames else pd.DataFrame()
    daily_returns_by_config = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    session_summary = pd.concat(session_frames, ignore_index=True) if session_frames else pd.DataFrame()
    invalidated_days = session_summary.loc[session_summary["invalidated_for_day"]].copy()
    results_by_year = _results_by_year(trades_by_config, session_summary, config)
    results_by_config = _aggregate_config_results(results_by_symbol, trades_by_config, session_summary, config)
    ranking_robust = results_by_config.sort_values(
        ["robust_score", "oos_2024_2026_Sharpe", "oos_2024_2026_net_pnl"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    baseline_decomposition = _baseline_trade_decomposition(trades_by_config, session_summary)
    opposite_breakout_trade_decomposition = _trade_decomposition(trades_by_config)

    tables = {
        "config_grid": pd.DataFrame([asdict(spec) for spec in policy_grid]),
        "results_by_symbol": results_by_symbol,
        "results_by_config": results_by_config,
        "results_by_year": results_by_year,
        "baseline_decomposition": baseline_decomposition,
        "opposite_breakout_trade_decomposition": opposite_breakout_trade_decomposition,
        "daily_returns_by_config": daily_returns_by_config,
        "trades_by_config": trades_by_config,
        "invalidated_days": invalidated_days,
        "ranking_robust": ranking_robust,
    }

    for name, df in tables.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)

    best_config_summary = ranking_robust.iloc[0].to_dict() if not ranking_robust.empty else {}
    best_config_path = output_dir / "best_config_summary.json"
    best_config_path.write_text(json.dumps(_serialize(best_config_summary), indent=2), encoding="utf-8")

    metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "config": _serialize(asdict(config)),
        "datasets": {prepared.symbol: str(prepared.dataset_path) for prepared in prepared_assets},
        "export_dir": str(output_dir),
        "policy_count": len(policy_grid),
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(_serialize(metadata), indent=2), encoding="utf-8")

    campaign_result = {
        **tables,
        "output_dir": output_dir,
        "config": config,
        "best_config_summary": best_config_summary,
    }
    report_path = write_markdown_report(output_dir, campaign_result)

    return {
        **campaign_result,
        "best_config_summary_path": best_config_path,
        "run_metadata_path": metadata_path,
        "final_report_path": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ORB opposite-breakout invalidation campaign.")
    parser.add_argument("--symbols", nargs="+", default=["MNQ", "MES", "M2K", "MGC"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", type=str, default="2018-01-01")
    parser.add_argument("--end-date", type=str, default="2026-12-31")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CampaignConfig(
        symbols=tuple(str(symbol).upper() for symbol in args.symbols),
        start_date=str(args.start_date) if args.start_date else None,
        end_date=str(args.end_date) if args.end_date else None,
        output_root=Path(args.output_root),
        smoke=bool(args.smoke),
    )
    result = run_campaign(config)
    print(f"export_dir: {result['output_dir']}")
    print(f"final_report: {result['final_report_path']}")
    print(f"ranking_robust: {result['output_dir'] / 'ranking_robust.csv'}")


if __name__ == "__main__":
    main()
