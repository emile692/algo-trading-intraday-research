"""Volume climax pullback strategy primitives (strict next-open, leak-free)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from itertools import islice, product
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.session import add_session_date


@dataclass(frozen=True)
class VolumeClimaxPullbackVariant:
    name: str
    family: str
    timeframe: str
    volume_quantile: float
    volume_lookback: int
    min_body_fraction: float | None
    min_range_atr: float | None
    stretch_ref: str | None
    min_stretch_atr: float | None
    wick_fraction: float | None
    stop_buffer_mode: str
    rr_target: float
    time_stop_bars: int
    session_overlay: str


def _mask_between(times: pd.Series, start: str, end: str) -> pd.Series:
    s = dt.time.fromisoformat(start)
    e = dt.time.fromisoformat(end)
    t = pd.to_datetime(times, errors="coerce").dt.time
    if s <= e:
        return (t >= s) & (t < e)
    return (t >= s) | (t < e)


def apply_session_overlay(df: pd.DataFrame, overlay: str) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    base = _mask_between(ts, "09:30:00", "16:00:00")
    if overlay == "all_rth":
        mask = base
    elif overlay == "exclude_first_10m":
        mask = base & (~_mask_between(ts, "09:30:00", "09:40:00"))
    elif overlay == "open_to_midday_only":
        mask = _mask_between(ts, "09:30:00", "12:00:00")
    elif overlay == "exclude_lunch":
        mask = base & (~_mask_between(ts, "12:00:00", "13:30:00"))
    else:
        raise ValueError(f"Unknown overlay '{overlay}'.")
    out = out.loc[mask].copy().reset_index(drop=True)
    return add_session_date(out)


def prepare_volume_climax_features(df: pd.DataFrame, *, atr_window: int = 20, ema_window: int = 20) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.sort_values("timestamp").reset_index(drop=True)
    out = add_session_date(out)

    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    open_ = pd.to_numeric(out["open"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(int(atr_window), min_periods=int(atr_window)).mean()

    typical = (high + low + close) / 3.0
    out["bar_range"] = high - low
    out["body"] = (close - open_).abs()
    out["body_fraction"] = out["body"] / out["bar_range"].replace(0.0, np.nan)
    out["upper_wick_fraction"] = (high - np.maximum(open_, close)) / out["bar_range"].replace(0.0, np.nan)
    out["lower_wick_fraction"] = (np.minimum(open_, close) - low) / out["bar_range"].replace(0.0, np.nan)

    session_key = pd.to_datetime(out["session_date"])
    out["session_vwap"] = (typical * volume).groupby(session_key, sort=True).cumsum() / volume.groupby(session_key, sort=True).cumsum().replace(0.0, np.nan)
    out["ema20"] = close.ewm(span=int(ema_window), adjust=False).mean()
    out["range_atr"] = out["bar_range"] / out["atr"].replace(0.0, np.nan)
    out["stretch_vwap_atr"] = (close - out["session_vwap"]).abs() / out["atr"].replace(0.0, np.nan)
    out["stretch_ema20_atr"] = (close - out["ema20"]).abs() / out["atr"].replace(0.0, np.nan)
    return out


def build_signal_frame(features: pd.DataFrame, variant: VolumeClimaxPullbackVariant) -> pd.DataFrame:
    out = features.copy()
    vol = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    q = vol.shift(1).rolling(int(variant.volume_lookback), min_periods=int(variant.volume_lookback)).quantile(float(variant.volume_quantile))
    out["volume_threshold_hist"] = q

    prev_open = pd.to_numeric(out["open"], errors="coerce").shift(1)
    prev_close = pd.to_numeric(out["close"], errors="coerce").shift(1)
    prev_high = pd.to_numeric(out["high"], errors="coerce").shift(1)
    prev_low = pd.to_numeric(out["low"], errors="coerce").shift(1)

    climax = vol.shift(1) > q
    bullish_prev = prev_close > prev_open
    bearish_prev = prev_close < prev_open

    quality = pd.Series(True, index=out.index)
    if variant.min_body_fraction is not None:
        quality &= pd.to_numeric(out["body_fraction"], errors="coerce").shift(1) >= float(variant.min_body_fraction)
    if variant.min_range_atr is not None:
        quality &= pd.to_numeric(out["range_atr"], errors="coerce").shift(1) >= float(variant.min_range_atr)

    stretch = pd.Series(True, index=out.index)
    if variant.stretch_ref is not None and variant.min_stretch_atr is not None:
        col = "stretch_vwap_atr" if variant.stretch_ref == "vwap" else "stretch_ema20_atr"
        stretch &= pd.to_numeric(out[col], errors="coerce").shift(1) >= float(variant.min_stretch_atr)

    rejection_short = pd.Series(True, index=out.index)
    rejection_long = pd.Series(True, index=out.index)
    if variant.wick_fraction is not None:
        rejection_short &= pd.to_numeric(out["upper_wick_fraction"], errors="coerce").shift(1) >= float(variant.wick_fraction)
        rejection_long &= pd.to_numeric(out["lower_wick_fraction"], errors="coerce").shift(1) >= float(variant.wick_fraction)

    out["entry_short"] = climax & bullish_prev & quality & stretch & rejection_short
    out["entry_long"] = climax & bearish_prev & quality & stretch & rejection_long

    atr_prev = pd.to_numeric(out["atr"], errors="coerce").shift(1)
    if variant.stop_buffer_mode == "0_tick":
        stop_buffer = 0.0
    elif variant.stop_buffer_mode == "1_tick":
        stop_buffer = np.nan  # resolved in backtester with instrument tick size
    elif variant.stop_buffer_mode == "0.1_atr":
        stop_buffer = 0.1 * atr_prev
    else:
        raise ValueError(f"Unknown stop_buffer_mode '{variant.stop_buffer_mode}'.")

    out["entry_stop_reference_long"] = prev_low - stop_buffer
    out["entry_stop_reference_short"] = prev_high + stop_buffer
    out["entry_target_r"] = float(variant.rr_target)
    out["entry_time_stop_bars"] = int(variant.time_stop_bars)
    return out


def _take(iterable: Iterable[VolumeClimaxPullbackVariant], n: int) -> list[VolumeClimaxPullbackVariant]:
    return list(islice(iterable, max(int(n), 0)))


def build_compact_variants(timeframes: Iterable[str]) -> list[VolumeClimaxPullbackVariant]:
    families = {
        "pure_climax": dict(use_quality=False, use_stretch=False, use_rejection=False),
        "climax_plus_bar_quality": dict(use_quality=True, use_stretch=False, use_rejection=False),
        "climax_plus_stretch": dict(use_quality=False, use_stretch=True, use_rejection=False),
        "climax_plus_rejection": dict(use_quality=False, use_stretch=False, use_rejection=True),
        "combined_qs": dict(use_quality=True, use_stretch=True, use_rejection=False),
        "combined_qsr": dict(use_quality=True, use_stretch=True, use_rejection=True),
    }
    variants: list[VolumeClimaxPullbackVariant] = []

    core = list(product([0.95, 0.975, 0.99], [50, 100, 200]))
    quality_pairs = [(0.5, 1.2), (0.6, 1.5), (0.7, 2.0)]
    stretch_pairs = list(product(["vwap", "ema20"], [0.5, 0.75, 1.0]))
    rejection = [0.2, 0.3, 0.4]
    exits = list(product(["0_tick", "1_tick", "0.1_atr"], [0.75, 1.0, 1.25], [2, 4, 6, 8]))
    overlays = ["all_rth", "exclude_first_10m", "open_to_midday_only", "exclude_lunch"]

    for timeframe in timeframes:
        for family, flags in families.items():
            pool: list[VolumeClimaxPullbackVariant] = []
            for vq, vlb in core:
                q_opts = quality_pairs if flags["use_quality"] else [(None, None)]
                s_opts = stretch_pairs if flags["use_stretch"] else [(None, None)]
                r_opts = rejection if flags["use_rejection"] else [None]
                for (mbf, mra), (sref, msa), wick, (sbm, rr, ts), overlay in product(q_opts, s_opts, r_opts, exits, overlays):
                    name = (
                        f"{family}_{timeframe}_vq{vq}_vl{vlb}_mb{mbf}_ra{mra}_sr{sref}_ms{msa}"
                        f"_wk{wick}_sb{sbm}_rr{rr}_ts{ts}_ov{overlay}"
                    ).replace(".", "p")
                    pool.append(
                        VolumeClimaxPullbackVariant(
                            name=name,
                            family=family,
                            timeframe=str(timeframe),
                            volume_quantile=float(vq),
                            volume_lookback=int(vlb),
                            min_body_fraction=mbf,
                            min_range_atr=mra,
                            stretch_ref=sref,
                            min_stretch_atr=msa,
                            wick_fraction=wick,
                            stop_buffer_mode=sbm,
                            rr_target=float(rr),
                            time_stop_bars=int(ts),
                            session_overlay=overlay,
                        )
                    )
            variants.extend(_take(iter(pool), 24))
    return variants
