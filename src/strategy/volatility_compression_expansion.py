"""Volatility Compression -> Expansion Breakout signal construction."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.config.settings import DEFAULT_INITIAL_CAPITAL_USD
from src.data.session import add_session_date
from src.features.volatility import add_atr


DEFAULT_SESSION_START = "09:30:00"
DEFAULT_SESSION_END = "16:00:00"
DEFAULT_ENTRY_START = "09:45:00"
DEFAULT_ENTRY_END = "15:30:00"
DEFAULT_BAR_MINUTES = 5
DEFAULT_BARS_PER_SESSION = 78


@dataclass(frozen=True)
class VCEBVariantConfig:
    """Fixed V1 strategy parameters for one VCEB variant."""

    name: str
    box_lookback: int
    compression_threshold: float
    target_r: float
    atr_window: int = 48
    expansion_threshold: float = 1.20
    breakout_buffer_atr: float = 0.10
    time_stop_bars: int = 12
    fixed_quantity: int = 1
    initial_capital_usd: float = DEFAULT_INITIAL_CAPITAL_USD
    session_start: str = DEFAULT_SESSION_START
    session_end: str = DEFAULT_SESSION_END
    entry_start: str = DEFAULT_ENTRY_START
    entry_end: str = DEFAULT_ENTRY_END


def build_default_vceb_variants() -> list[VCEBVariantConfig]:
    """Return the disciplined V1 parameter grid requested by the user."""
    variants: list[VCEBVariantConfig] = []
    for box_lookback in (8, 12, 16):
        for compression_threshold in (15.0, 20.0):
            for target_r in (1.8, 2.2):
                variants.append(
                    VCEBVariantConfig(
                        name=(
                            f"vceb_n{int(box_lookback)}"
                            f"_ct{int(compression_threshold)}"
                            f"_tr{str(target_r).replace('.', 'p')}"
                        ),
                        box_lookback=int(box_lookback),
                        compression_threshold=float(compression_threshold),
                        target_r=float(target_r),
                    )
                )
    return variants


def _left_closed_time_mask(timestamps: pd.Series, start_time: str, end_time: str) -> pd.Series:
    start = dt.time.fromisoformat(start_time)
    end = dt.time.fromisoformat(end_time)
    times = pd.to_datetime(timestamps, errors="coerce").dt.time
    if start <= end:
        return (times >= start) & (times < end)
    return (times >= start) | (times < end)


def _inclusive_time_mask(timestamps: pd.Series, start_time: str, end_time: str) -> pd.Series:
    start = dt.time.fromisoformat(start_time)
    end = dt.time.fromisoformat(end_time)
    times = pd.to_datetime(timestamps, errors="coerce").dt.time
    if start <= end:
        return (times >= start) & (times <= end)
    return (times >= start) | (times <= end)


def filter_rth_bar_starts(
    df: pd.DataFrame,
    *,
    session_start: str = DEFAULT_SESSION_START,
    session_end: str = DEFAULT_SESSION_END,
) -> pd.DataFrame:
    """Filter a start-aligned intraday frame to RTH bars in [start, end)."""
    mask = _left_closed_time_mask(df["timestamp"], session_start, session_end)
    return df.loc[mask].copy().reset_index(drop=True)


class _FenwickTree:
    """Compact Fenwick tree for exact rolling percentile ranks."""

    def __init__(self, size: int) -> None:
        self._size = int(size)
        self._tree = np.zeros(self._size + 1, dtype=np.int64)

    def add(self, index: int, delta: int) -> None:
        idx = int(index)
        while idx <= self._size:
            self._tree[idx] += int(delta)
            idx += idx & -idx

    def prefix_sum(self, index: int) -> int:
        idx = int(index)
        total = 0
        while idx > 0:
            total += int(self._tree[idx])
            idx -= idx & -idx
        return total


def rolling_percentile_rank_exact(
    values: pd.Series,
    *,
    lookback: int,
    min_history: int = 1,
) -> pd.Series:
    """Return an exact, strict no-lookahead rolling percentile rank."""
    if lookback <= 0:
        raise ValueError("lookback must be strictly positive.")
    if min_history < 1:
        raise ValueError("min_history must be >= 1.")

    clean = pd.to_numeric(values, errors="coerce")
    array = clean.to_numpy(dtype=float)
    out = np.full(len(array), np.nan, dtype=float)

    finite_mask = np.isfinite(array)
    if not finite_mask.any():
        return pd.Series(out, index=values.index, dtype=float)

    unique_values = np.unique(array[finite_mask])
    ranks = np.zeros(len(array), dtype=np.int32)
    ranks[finite_mask] = np.searchsorted(unique_values, array[finite_mask], side="left").astype(np.int32) + 1

    tree = _FenwickTree(size=len(unique_values))
    history_count = 0

    for idx, rank in enumerate(ranks):
        old_idx = idx - int(lookback)
        if old_idx >= 0 and ranks[old_idx] > 0:
            tree.add(int(ranks[old_idx]), -1)
            history_count -= 1

        if rank > 0 and history_count >= int(min_history):
            lt = tree.prefix_sum(int(rank) - 1)
            le = tree.prefix_sum(int(rank))
            eq = le - lt
            out[idx] = (float(lt) + 0.5 * float(eq)) / float(history_count)

        if rank > 0:
            tree.add(int(rank), 1)
            history_count += 1

    return pd.Series(out, index=values.index, dtype=float)


def _normalize_box_lengths(values: Iterable[int]) -> list[int]:
    lengths = sorted({int(value) for value in values if int(value) > 0})
    if not lengths:
        raise ValueError("box_lengths must contain at least one positive integer.")
    return lengths


def prepare_vceb_feature_frame(
    df: pd.DataFrame,
    *,
    session_start: str = DEFAULT_SESSION_START,
    session_end: str = DEFAULT_SESSION_END,
    box_lengths: Iterable[int] = (8, 12, 16),
    atr_window: int = 48,
    compression_percentile_lookback_bars: int = 20 * DEFAULT_BARS_PER_SESSION,
    compression_percentile_min_history_bars: int = 5 * DEFAULT_BARS_PER_SESSION,
) -> pd.DataFrame:
    """Build the leak-free 5-minute VCEB feature frame."""
    if int(atr_window) <= 0:
        raise ValueError("atr_window must be strictly positive.")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if out["timestamp"].isna().any():
        raise ValueError("Found unparsable timestamps while preparing the VCEB feature frame.")

    out = out.sort_values("timestamp").reset_index(drop=True)
    out = filter_rth_bar_starts(out, session_start=session_start, session_end=session_end)
    out = add_session_date(out)
    out = add_atr(out, window=int(atr_window))
    out = out.rename(columns={f"atr_{int(atr_window)}": "atr_48" if int(atr_window) == 48 else f"atr_{int(atr_window)}"})

    atr_col = "atr_48" if int(atr_window) == 48 else f"atr_{int(atr_window)}"
    prev_close = pd.to_numeric(out["close"], errors="coerce").shift(1)
    out["true_range"] = np.maximum(
        pd.to_numeric(out["high"], errors="coerce") - pd.to_numeric(out["low"], errors="coerce"),
        np.maximum(
            (pd.to_numeric(out["high"], errors="coerce") - prev_close).abs(),
            (pd.to_numeric(out["low"], errors="coerce") - prev_close).abs(),
        ),
    )
    out["expansion_ratio"] = out["true_range"] / pd.to_numeric(out[atr_col], errors="coerce").replace(0.0, np.nan)

    box_lengths_clean = _normalize_box_lengths(box_lengths)
    for box_length in box_lengths_clean:
        high_col = f"box_high_{int(box_length)}"
        low_col = f"box_low_{int(box_length)}"
        width_col = f"box_width_{int(box_length)}"
        ratio_col = f"compression_ratio_{int(box_length)}"
        pct_col = f"compression_pct_{int(box_length)}"

        out[high_col] = pd.to_numeric(out["high"], errors="coerce").shift(1).rolling(int(box_length), min_periods=int(box_length)).max()
        out[low_col] = pd.to_numeric(out["low"], errors="coerce").shift(1).rolling(int(box_length), min_periods=int(box_length)).min()
        out[width_col] = out[high_col] - out[low_col]
        out[ratio_col] = out[width_col] / pd.to_numeric(out[atr_col], errors="coerce").replace(0.0, np.nan)
        out[pct_col] = rolling_percentile_rank_exact(
            out[ratio_col],
            lookback=int(compression_percentile_lookback_bars),
            min_history=int(compression_percentile_min_history_bars),
        )

    group = out.groupby("session_date", sort=True)
    out["is_first_bar_of_session"] = group.cumcount().eq(0)
    out["is_last_bar_of_session"] = group.cumcount(ascending=False).eq(0)
    out["bar_index_in_session"] = group.cumcount()
    return out.reset_index(drop=True)


def build_vceb_signal_frame(
    feature_df: pd.DataFrame,
    variant: VCEBVariantConfig,
) -> pd.DataFrame:
    """Generate next-open VCEB entry signals and shifted stop metadata."""
    required = {
        "timestamp",
        "session_date",
        "open",
        "high",
        "low",
        "close",
        "is_last_bar_of_session",
        "expansion_ratio",
        "true_range",
    }
    missing = sorted(required - set(feature_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for VCEB signal generation: {missing}")

    out = feature_df.copy().sort_values("timestamp").reset_index(drop=True)
    atr_col = "atr_48" if "atr_48" in out.columns else f"atr_{int(variant.atr_window)}"
    box_suffix = int(variant.box_lookback)
    pct_col = f"compression_pct_{box_suffix}"
    high_col = f"box_high_{box_suffix}"
    low_col = f"box_low_{box_suffix}"
    width_col = f"box_width_{box_suffix}"
    ratio_col = f"compression_ratio_{box_suffix}"
    required_variant_columns = [atr_col, pct_col, high_col, low_col, width_col, ratio_col]
    missing_variant = [column for column in required_variant_columns if column not in out.columns]
    if missing_variant:
        raise ValueError(f"Missing VCEB feature columns for variant '{variant.name}': {missing_variant}")

    atr = pd.to_numeric(out[atr_col], errors="coerce")
    box_width = pd.to_numeric(out[width_col], errors="coerce")
    buffer_points = float(variant.breakout_buffer_atr) * atr

    out["compression_valid"] = pd.to_numeric(out[pct_col], errors="coerce") <= (float(variant.compression_threshold) / 100.0)
    out["breakout_buffer_points"] = buffer_points
    out["stop_distance_signal"] = np.maximum(1.0 * atr, 0.75 * box_width)
    out["target_r_signal"] = float(variant.target_r)
    out["signal_box_width"] = box_width
    out["signal_atr"] = atr
    out["trade_allowed"] = _inclusive_time_mask(out["timestamp"], variant.entry_start, variant.entry_end)

    raw_long = (
        out["compression_valid"]
        & (pd.to_numeric(out["close"], errors="coerce") > pd.to_numeric(out[high_col], errors="coerce") + buffer_points)
        & (pd.to_numeric(out["expansion_ratio"], errors="coerce") >= float(variant.expansion_threshold))
    )
    raw_short = (
        out["compression_valid"]
        & (pd.to_numeric(out["close"], errors="coerce") < pd.to_numeric(out[low_col], errors="coerce") - buffer_points)
        & (pd.to_numeric(out["expansion_ratio"], errors="coerce") >= float(variant.expansion_threshold))
    )

    raw_signal = np.where(raw_long, 1, np.where(raw_short, -1, 0))
    out["raw_signal"] = pd.Series(raw_signal, index=out.index, dtype=int)

    group = out.groupby("session_date", sort=True)
    out["entry_signal_time"] = group["timestamp"].shift(1)
    out["entry_stop_distance"] = group["stop_distance_signal"].shift(1)
    out["entry_target_r"] = group["target_r_signal"].shift(1)
    out["entry_signal_box_width"] = group["signal_box_width"].shift(1)
    out["entry_signal_atr"] = group["signal_atr"].shift(1)

    out["entry_long"] = (
        pd.Series(raw_long, index=out.index, dtype="boolean")
        .groupby(out["session_date"], sort=True)
        .shift(1)
        .fillna(False)
        .astype(bool)
        & out["trade_allowed"].fillna(False)
    )
    out["entry_short"] = (
        pd.Series(raw_short, index=out.index, dtype="boolean")
        .groupby(out["session_date"], sort=True)
        .shift(1)
        .fillna(False)
        .astype(bool)
        & out["trade_allowed"].fillna(False)
    )

    out["signal"] = np.where(out["entry_long"], 1, np.where(out["entry_short"], -1, 0))
    out["variant_name"] = variant.name
    return out.reset_index(drop=True)
