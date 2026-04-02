"""Leak-free V2 strategy primitives for the volume climax pullback family."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from src.data.session import add_session_date


@dataclass(frozen=True)
class VolumeClimaxPullbackV2Variant:
    """Compact research config for the V2 standalone campaign."""

    name: str
    family: str
    timeframe: str
    volume_quantile: float
    volume_lookback: int
    min_body_fraction: float
    min_range_atr: float
    trend_ema_window: int | None
    ema_slope_threshold: float | None
    atr_percentile_low: float | None
    atr_percentile_high: float | None
    compression_ratio_max: float | None
    entry_mode: str
    pullback_fraction: float | None
    confirmation_window: int | None
    exit_mode: str
    rr_target: float
    atr_target_multiple: float | None
    time_stop_bars: int
    trailing_atr_multiple: float
    session_overlay: str = "all_rth"


CORE_SIGNAL_GRID: tuple[tuple[float, float, float], ...] = (
    (0.95, 0.5, 1.2),
    (0.95, 0.5, 1.5),
    (0.95, 0.6, 1.2),
    (0.95, 0.6, 1.5),
    (0.975, 0.5, 1.2),
    (0.975, 0.5, 1.5),
    (0.975, 0.6, 1.2),
    (0.975, 0.6, 1.5),
)

ANCHOR_SIGNAL_GRID: tuple[tuple[float, float, float], ...] = (
    (0.95, 0.5, 1.2),
    (0.95, 0.6, 1.2),
    (0.975, 0.5, 1.2),
    (0.975, 0.6, 1.5),
)

V3_DYNAMIC_EXIT_SIGNAL_GRID: tuple[tuple[float, float, float], ...] = (
    (0.95, 0.5, 1.2),
    (0.95, 0.5, 1.5),
    (0.95, 0.6, 1.2),
    (0.95, 0.6, 1.5),
    (0.975, 0.5, 1.2),
    (0.975, 0.5, 1.5),
    (0.975, 0.6, 1.2),
    (0.975, 0.6, 1.5),
)

V3_DYNAMIC_EXIT_SPECS: tuple[tuple[str, dict[str, float | int | str | None]], ...] = (
    ("atr_target_1p0_ts2", dict(exit_mode="atr_fraction", atr_target_multiple=1.0, time_stop_bars=2)),
    ("atr_target_1p0_ts3", dict(exit_mode="atr_fraction", atr_target_multiple=1.0, time_stop_bars=3)),
    ("atr_target_1p0_ts4", dict(exit_mode="atr_fraction", atr_target_multiple=1.0, time_stop_bars=4)),
    ("mixed_ts2", dict(exit_mode="mixed", atr_target_multiple=None, time_stop_bars=2)),
    ("mixed_ts3", dict(exit_mode="mixed", atr_target_multiple=None, time_stop_bars=3)),
    ("mixed_ts4", dict(exit_mode="mixed", atr_target_multiple=None, time_stop_bars=4)),
)

V3_MGC_REGIME_SIGNAL_GRID: tuple[tuple[float, float, float], ...] = (
    (0.95, 0.5, 1.2),
    (0.975, 0.5, 1.2),
)

V3_MGC_REGIME_EXIT_SPECS: tuple[tuple[str, dict[str, float | int | str | None]], ...] = (
    ("atr_target_1p0_ts3", dict(exit_mode="atr_fraction", atr_target_multiple=1.0, time_stop_bars=3)),
    ("mixed_ts3", dict(exit_mode="mixed", atr_target_multiple=None, time_stop_bars=3)),
    ("mixed_ts4", dict(exit_mode="mixed", atr_target_multiple=None, time_stop_bars=4)),
)

V3_MGC_EMA_SLOPE_FILTERS: tuple[tuple[str, dict[str, float | int | None]], ...] = (
    ("off", dict(trend_ema_window=None, ema_slope_threshold=None)),
    ("mild", dict(trend_ema_window=50, ema_slope_threshold=0.06)),
)

V3_MGC_ATR_PERCENTILE_FILTERS: tuple[tuple[str, dict[str, float | None]], ...] = (
    ("off", dict(atr_percentile_low=None, atr_percentile_high=None)),
    ("20_80", dict(atr_percentile_low=0.20, atr_percentile_high=0.80)),
    ("30_70", dict(atr_percentile_low=0.30, atr_percentile_high=0.70)),
)

V3_MGC_COMPRESSION_FILTERS: tuple[tuple[str, dict[str, float | None]], ...] = (
    ("off", dict(compression_ratio_max=None)),
    ("mild", dict(compression_ratio_max=0.90)),
)


def _rolling_percent_rank(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if window <= 1:
        return pd.Series(out, index=series.index, dtype=float)

    for idx in range(window - 1, len(values)):
        window_values = values[idx - window + 1 : idx + 1]
        if np.isnan(window_values).any():
            continue
        out[idx] = float(np.searchsorted(np.sort(window_values), window_values[-1], side="right") / window)
    return pd.Series(out, index=series.index, dtype=float)


def prepare_volume_climax_pullback_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build strict historical features for the 1h standalone campaign."""

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.sort_values("timestamp").reset_index(drop=True)
    out = add_session_date(out)

    open_ = pd.to_numeric(out["open"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    out["atr_5"] = true_range.rolling(5, min_periods=5).mean()
    out["atr_20"] = true_range.rolling(20, min_periods=20).mean()
    out["atr_50"] = true_range.rolling(50, min_periods=50).mean()
    out["bar_range"] = high - low
    out["body"] = (close - open_).abs()
    out["body_fraction"] = out["body"] / out["bar_range"].replace(0.0, np.nan)
    out["range_atr"] = out["bar_range"] / out["atr_20"].replace(0.0, np.nan)
    out["atr_ratio_5_20"] = out["atr_5"] / out["atr_20"].replace(0.0, np.nan)

    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["ema20_slope_3_atr"] = (out["ema20"] - out["ema20"].shift(3)) / out["atr_20"].replace(0.0, np.nan)
    out["ema50_slope_3_atr"] = (out["ema50"] - out["ema50"].shift(3)) / out["atr_20"].replace(0.0, np.nan)

    typical = (high + low + close) / 3.0
    session_key = pd.to_datetime(out["session_date"])
    cum_volume = volume.groupby(session_key, sort=True).cumsum().replace(0.0, np.nan)
    out["session_vwap"] = (typical * volume).groupby(session_key, sort=True).cumsum() / cum_volume
    out["atr_percentile_100"] = _rolling_percent_rank(out["atr_20"], 100)
    out["is_last_bar_of_session"] = out["session_date"] != out["session_date"].shift(-1)
    return out


def build_volume_climax_pullback_v2_signal_frame(
    features: pd.DataFrame,
    variant: VolumeClimaxPullbackV2Variant,
) -> pd.DataFrame:
    """Project a raw climax setup into a strict t-1 signal frame."""

    out = features.copy()
    volume = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    volume_threshold = volume.shift(1).rolling(
        int(variant.volume_lookback),
        min_periods=int(variant.volume_lookback),
    ).quantile(float(variant.volume_quantile))

    prev_open = pd.to_numeric(out["open"], errors="coerce").shift(1)
    prev_high = pd.to_numeric(out["high"], errors="coerce").shift(1)
    prev_low = pd.to_numeric(out["low"], errors="coerce").shift(1)
    prev_close = pd.to_numeric(out["close"], errors="coerce").shift(1)
    prev_body_fraction = pd.to_numeric(out["body_fraction"], errors="coerce").shift(1)
    prev_range_atr = pd.to_numeric(out["range_atr"], errors="coerce").shift(1)

    bullish_prev = prev_close > prev_open
    bearish_prev = prev_close < prev_open
    climax_prev = volume.shift(1) > volume_threshold
    quality_prev = (prev_body_fraction >= float(variant.min_body_fraction)) & (
        prev_range_atr >= float(variant.min_range_atr)
    )

    raw_short = climax_prev & bullish_prev & quality_prev
    raw_long = climax_prev & bearish_prev & quality_prev

    short_regime = pd.Series(True, index=out.index, dtype=bool)
    long_regime = pd.Series(True, index=out.index, dtype=bool)

    if variant.trend_ema_window is not None and variant.ema_slope_threshold is not None:
        slope_col = f"ema{int(variant.trend_ema_window)}_slope_3_atr"
        slope_prev = pd.to_numeric(out[slope_col], errors="coerce").shift(1)
        threshold = float(variant.ema_slope_threshold)
        short_regime &= slope_prev <= threshold
        long_regime &= slope_prev >= -threshold

    if variant.atr_percentile_low is not None and variant.atr_percentile_high is not None:
        atr_pct_prev = pd.to_numeric(out["atr_percentile_100"], errors="coerce").shift(1)
        short_regime &= atr_pct_prev.between(float(variant.atr_percentile_low), float(variant.atr_percentile_high))
        long_regime &= atr_pct_prev.between(float(variant.atr_percentile_low), float(variant.atr_percentile_high))

    if variant.compression_ratio_max is not None:
        compression_prev = pd.to_numeric(out["atr_ratio_5_20"], errors="coerce").shift(1)
        short_regime &= compression_prev <= float(variant.compression_ratio_max)
        long_regime &= compression_prev <= float(variant.compression_ratio_max)

    signal = pd.Series(0, index=out.index, dtype=int)
    signal.loc[raw_short & short_regime] = -1
    signal.loc[raw_long & long_regime] = 1

    raw_signal = pd.Series(0, index=out.index, dtype=int)
    raw_signal.loc[raw_short] = -1
    raw_signal.loc[raw_long] = 1

    out["volume_threshold_hist"] = volume_threshold
    out["raw_signal"] = raw_signal
    out["signal"] = signal
    out["entry_short"] = signal.eq(-1)
    out["entry_long"] = signal.eq(1)
    out["setup_signal_time"] = out["timestamp"].shift(1)
    out["setup_reference_open"] = prev_open
    out["setup_reference_high"] = prev_high
    out["setup_reference_low"] = prev_low
    out["setup_reference_close"] = prev_close
    out["setup_reference_range"] = (prev_high - prev_low).clip(lower=0.0)
    out["setup_reference_atr"] = pd.to_numeric(out["atr_20"], errors="coerce").shift(1)
    out["setup_reference_vwap"] = pd.to_numeric(out["session_vwap"], errors="coerce").shift(1)
    out["setup_stop_reference_long"] = prev_low
    out["setup_stop_reference_short"] = prev_high
    return out


def build_volume_climax_pullback_v2_variants() -> list[VolumeClimaxPullbackV2Variant]:
    """Return a targeted but meaningful V2 research grid."""

    variants: list[VolumeClimaxPullbackV2Variant] = []

    for volume_quantile, body_fraction, range_atr in CORE_SIGNAL_GRID:
        core_name = f"vq{volume_quantile}_bf{body_fraction}_ra{range_atr}".replace(".", "p")
        variants.append(
            VolumeClimaxPullbackV2Variant(
                name=f"signal_core_{core_name}_next_open_rr1p0_ts2",
                family="signal_core",
                timeframe="1h",
                volume_quantile=volume_quantile,
                volume_lookback=50,
                min_body_fraction=body_fraction,
                min_range_atr=range_atr,
                trend_ema_window=None,
                ema_slope_threshold=None,
                atr_percentile_low=None,
                atr_percentile_high=None,
                compression_ratio_max=None,
                entry_mode="next_open",
                pullback_fraction=None,
                confirmation_window=None,
                exit_mode="fixed_rr",
                rr_target=1.0,
                atr_target_multiple=None,
                time_stop_bars=2,
                trailing_atr_multiple=0.5,
            )
        )

    regime_specs = (
        ("trend_ema20_low", dict(trend_ema_window=20, ema_slope_threshold=0.03)),
        ("trend_ema50_medium", dict(trend_ema_window=50, ema_slope_threshold=0.06)),
        ("vol_20_80", dict(atr_percentile_low=0.20, atr_percentile_high=0.80)),
        ("vol_30_70", dict(atr_percentile_low=0.30, atr_percentile_high=0.70)),
        ("compression_0p8", dict(compression_ratio_max=0.80)),
        ("compression_0p9", dict(compression_ratio_max=0.90)),
        (
            "stack_ema20_low_vol20_80_comp0p9",
            dict(
                trend_ema_window=20,
                ema_slope_threshold=0.03,
                atr_percentile_low=0.20,
                atr_percentile_high=0.80,
                compression_ratio_max=0.90,
            ),
        ),
        (
            "stack_ema50_medium_vol30_70_comp0p8",
            dict(
                trend_ema_window=50,
                ema_slope_threshold=0.06,
                atr_percentile_low=0.30,
                atr_percentile_high=0.70,
                compression_ratio_max=0.80,
            ),
        ),
    )

    for volume_quantile, body_fraction, range_atr in CORE_SIGNAL_GRID:
        core_name = f"vq{volume_quantile}_bf{body_fraction}_ra{range_atr}".replace(".", "p")
        for spec_name, regime_kwargs in regime_specs:
            variants.append(
                VolumeClimaxPullbackV2Variant(
                    name=f"regime_filtered_{spec_name}_{core_name}",
                    family="regime_filtered",
                    timeframe="1h",
                    volume_quantile=volume_quantile,
                    volume_lookback=50,
                    min_body_fraction=body_fraction,
                    min_range_atr=range_atr,
                    trend_ema_window=regime_kwargs.get("trend_ema_window"),
                    ema_slope_threshold=regime_kwargs.get("ema_slope_threshold"),
                    atr_percentile_low=regime_kwargs.get("atr_percentile_low"),
                    atr_percentile_high=regime_kwargs.get("atr_percentile_high"),
                    compression_ratio_max=regime_kwargs.get("compression_ratio_max"),
                    entry_mode="next_open",
                    pullback_fraction=None,
                    confirmation_window=None,
                    exit_mode="fixed_rr",
                    rr_target=1.0,
                    atr_target_multiple=None,
                    time_stop_bars=2,
                    trailing_atr_multiple=0.5,
                )
            )

    entry_specs = (
        ("pullback_limit_0p25", dict(entry_mode="pullback_limit", pullback_fraction=0.25, confirmation_window=None)),
        ("pullback_limit_0p5", dict(entry_mode="pullback_limit", pullback_fraction=0.50, confirmation_window=None)),
        ("confirmation_0p25_w1", dict(entry_mode="confirmation", pullback_fraction=0.25, confirmation_window=1)),
        ("confirmation_0p25_w2", dict(entry_mode="confirmation", pullback_fraction=0.25, confirmation_window=2)),
        ("confirmation_0p5_w2", dict(entry_mode="confirmation", pullback_fraction=0.50, confirmation_window=2)),
    )

    for volume_quantile, body_fraction, range_atr in ANCHOR_SIGNAL_GRID:
        core_name = f"vq{volume_quantile}_bf{body_fraction}_ra{range_atr}".replace(".", "p")
        for spec_name, entry_kwargs in entry_specs:
            variants.append(
                VolumeClimaxPullbackV2Variant(
                    name=f"improved_entry_{spec_name}_{core_name}",
                    family="improved_entry",
                    timeframe="1h",
                    volume_quantile=volume_quantile,
                    volume_lookback=50,
                    min_body_fraction=body_fraction,
                    min_range_atr=range_atr,
                    trend_ema_window=None,
                    ema_slope_threshold=None,
                    atr_percentile_low=None,
                    atr_percentile_high=None,
                    compression_ratio_max=None,
                    entry_mode=str(entry_kwargs["entry_mode"]),
                    pullback_fraction=entry_kwargs.get("pullback_fraction"),
                    confirmation_window=entry_kwargs.get("confirmation_window"),
                    exit_mode="fixed_rr",
                    rr_target=1.0,
                    atr_target_multiple=None,
                    time_stop_bars=2,
                    trailing_atr_multiple=0.5,
                )
            )

    exit_specs = (
        ("target_vwap_ts2", dict(exit_mode="target_vwap", time_stop_bars=2)),
        ("target_vwap_ts3", dict(exit_mode="target_vwap", time_stop_bars=3)),
        ("atr_target_0p5_ts2", dict(exit_mode="atr_fraction", atr_target_multiple=0.5, time_stop_bars=2)),
        ("atr_target_1p0_ts3", dict(exit_mode="atr_fraction", atr_target_multiple=1.0, time_stop_bars=3)),
        ("mixed_ts3", dict(exit_mode="mixed", time_stop_bars=3)),
        ("mixed_ts4", dict(exit_mode="mixed", time_stop_bars=4)),
    )

    for volume_quantile, body_fraction, range_atr in ANCHOR_SIGNAL_GRID:
        core_name = f"vq{volume_quantile}_bf{body_fraction}_ra{range_atr}".replace(".", "p")
        for spec_name, exit_kwargs in exit_specs:
            variants.append(
                VolumeClimaxPullbackV2Variant(
                    name=f"dynamic_exit_{spec_name}_{core_name}",
                    family="dynamic_exit",
                    timeframe="1h",
                    volume_quantile=volume_quantile,
                    volume_lookback=50,
                    min_body_fraction=body_fraction,
                    min_range_atr=range_atr,
                    trend_ema_window=None,
                    ema_slope_threshold=None,
                    atr_percentile_low=None,
                    atr_percentile_high=None,
                    compression_ratio_max=None,
                    entry_mode="next_open",
                    pullback_fraction=None,
                    confirmation_window=None,
                    exit_mode=str(exit_kwargs["exit_mode"]),
                    rr_target=1.0,
                    atr_target_multiple=exit_kwargs.get("atr_target_multiple"),
                    time_stop_bars=int(exit_kwargs["time_stop_bars"]),
                    trailing_atr_multiple=0.5,
                )
            )

    deduped: list[VolumeClimaxPullbackV2Variant] = []
    seen_names: set[str] = set()
    for variant in variants:
        if variant.name in seen_names:
            continue
        deduped.append(variant)
        seen_names.add(variant.name)
    return deduped


def _dedupe_variants(variants: list[VolumeClimaxPullbackV2Variant]) -> list[VolumeClimaxPullbackV2Variant]:
    deduped: list[VolumeClimaxPullbackV2Variant] = []
    seen_names: set[str] = set()
    for variant in variants:
        if variant.name in seen_names:
            continue
        deduped.append(variant)
        seen_names.add(variant.name)
    return deduped


def _build_dynamic_exit_variant(
    *,
    family: str,
    spec_name: str,
    core_name: str,
    volume_quantile: float,
    body_fraction: float,
    range_atr: float,
    trend_ema_window: int | None = None,
    ema_slope_threshold: float | None = None,
    atr_percentile_low: float | None = None,
    atr_percentile_high: float | None = None,
    compression_ratio_max: float | None = None,
    exit_mode: str,
    atr_target_multiple: float | None,
    time_stop_bars: int,
) -> VolumeClimaxPullbackV2Variant:
    return VolumeClimaxPullbackV2Variant(
        name=f"{family}_{spec_name}_{core_name}",
        family=family,
        timeframe="1h",
        volume_quantile=volume_quantile,
        volume_lookback=50,
        min_body_fraction=body_fraction,
        min_range_atr=range_atr,
        trend_ema_window=trend_ema_window,
        ema_slope_threshold=ema_slope_threshold,
        atr_percentile_low=atr_percentile_low,
        atr_percentile_high=atr_percentile_high,
        compression_ratio_max=compression_ratio_max,
        entry_mode="next_open",
        pullback_fraction=None,
        confirmation_window=None,
        exit_mode=exit_mode,
        rr_target=1.0,
        atr_target_multiple=atr_target_multiple,
        time_stop_bars=time_stop_bars,
        trailing_atr_multiple=0.5,
    )


def build_volume_climax_pullback_v3_variants(symbol: str) -> list[VolumeClimaxPullbackV2Variant]:
    """Return the compact V3 asset-aware grid."""

    normalized_symbol = str(symbol).upper()
    variants: list[VolumeClimaxPullbackV2Variant] = []

    for volume_quantile, body_fraction, range_atr in V3_DYNAMIC_EXIT_SIGNAL_GRID:
        core_name = f"vq{volume_quantile}_bf{body_fraction}_ra{range_atr}".replace(".", "p")
        for spec_name, exit_kwargs in V3_DYNAMIC_EXIT_SPECS:
            variants.append(
                _build_dynamic_exit_variant(
                    family="dynamic_exit",
                    spec_name=spec_name,
                    core_name=core_name,
                    volume_quantile=volume_quantile,
                    body_fraction=body_fraction,
                    range_atr=range_atr,
                    exit_mode=str(exit_kwargs["exit_mode"]),
                    atr_target_multiple=exit_kwargs.get("atr_target_multiple"),
                    time_stop_bars=int(exit_kwargs["time_stop_bars"]),
                )
            )

    if normalized_symbol != "MGC":
        return _dedupe_variants(variants)

    for volume_quantile, body_fraction, range_atr in V3_MGC_REGIME_SIGNAL_GRID:
        core_name = f"vq{volume_quantile}_bf{body_fraction}_ra{range_atr}".replace(".", "p")
        for exit_spec_name, exit_kwargs in V3_MGC_REGIME_EXIT_SPECS:
            for ema_label, ema_kwargs in V3_MGC_EMA_SLOPE_FILTERS:
                for atr_label, atr_kwargs in V3_MGC_ATR_PERCENTILE_FILTERS:
                    for compression_label, compression_kwargs in V3_MGC_COMPRESSION_FILTERS:
                        if ema_label == "off" and atr_label == "off" and compression_label == "off":
                            continue
                        filter_name = f"ema_{ema_label}_atr_{atr_label}_compression_{compression_label}"
                        variants.append(
                            _build_dynamic_exit_variant(
                                family="regime_filtered",
                                spec_name=f"{filter_name}_{exit_spec_name}",
                                core_name=core_name,
                                volume_quantile=volume_quantile,
                                body_fraction=body_fraction,
                                range_atr=range_atr,
                                trend_ema_window=ema_kwargs.get("trend_ema_window"),
                                ema_slope_threshold=ema_kwargs.get("ema_slope_threshold"),
                                atr_percentile_low=atr_kwargs.get("atr_percentile_low"),
                                atr_percentile_high=atr_kwargs.get("atr_percentile_high"),
                                compression_ratio_max=compression_kwargs.get("compression_ratio_max"),
                                exit_mode=str(exit_kwargs["exit_mode"]),
                                atr_target_multiple=exit_kwargs.get("atr_target_multiple"),
                                time_stop_bars=int(exit_kwargs["time_stop_bars"]),
                            )
                        )

    return _dedupe_variants(variants)
