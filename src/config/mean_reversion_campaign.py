"""Configuration for the intraday mean reversion research campaign."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.paths import EXPORTS_DIR
from src.config.vwap_campaign import (
    DEFAULT_IS_FRACTION,
    DEFAULT_RTH_SESSION_END,
    DEFAULT_RTH_SESSION_START,
    PropFirmConstraintConfig,
    TimeWindow,
    build_default_prop_constraints,
    resolve_default_vwap_dataset,
)


@dataclass(frozen=True)
class TimeframeDefinition:
    """Explicit bar interval used by the campaign."""

    label: str
    bar_minutes: int
    resample_rule: str


DEFAULT_TIMEFRAMES = (
    TimeframeDefinition(label="5m", bar_minutes=5, resample_rule="5min"),
    TimeframeDefinition(label="15m", bar_minutes=15, resample_rule="15min"),
)


@dataclass(frozen=True)
class MeanReversionVariantConfig:
    """Leak-free discrete mean reversion strategy definition."""

    name: str
    family: str
    symbol: str
    timeframe: str
    notes: str = ""
    fixed_quantity: int = 1
    max_trades_per_day: int = 2
    min_oos_trades: int = 20
    entry_start: str = "09:45:00"
    entry_end: str = "15:30:00"
    excluded_windows: tuple[TimeWindow, ...] = ()
    skip_first_minutes: int = 15
    skip_last_minutes: int = 15
    atr_period: int = 14
    stop_atr_multiple: float = 1.0
    timeout_bars: int = 6
    use_partial_exit: bool = False
    partial_target_fraction: float = 0.50
    target_source: str = "session_vwap"
    extension_mode: str | None = None
    extension_threshold: float | None = None
    zscore_window: int | None = None
    bollinger_window: int | None = None
    bollinger_std: float | None = None
    bollinger_confirmation: str = "immediate"
    oscillator_kind: str | None = None
    oscillator_period: int | None = None
    oscillator_period_fast: int | None = None
    oscillator_period_slow: int | None = None
    oscillator_smoothing: int | None = None
    oversold_level: float | None = None
    overbought_level: float | None = None
    oscillator_trigger: str = "extreme"
    opening_window_minutes: int | None = None
    stretch_reference: str | None = None
    stretch_threshold_atr: float | None = None
    stretch_threshold_or_multiple: float | None = None
    stretch_volume_z_max: float | None = None
    ema_window: int | None = None
    band_width_atr: float | None = None
    require_closes_outside: int = 1
    streak_length: int | None = None
    streak_extension_atr: float | None = None
    require_exhaustion_bar: bool = False
    anchor_distance_source: str = "session_vwap"
    anchor_distance_max_atr: float | None = None
    adx_period: int = 14
    adx_max: float | None = None
    ema_filter_window: int = 20
    ema_slope_lookback: int = 3
    ema_slope_max_atr: float | None = None
    vwap_slope_lookback: int = 3
    vwap_slope_max_atr: float | None = None
    anti_trend_day_max: float | None = None
    session_range_max_atr: float | None = None
    persistent_vwap_distance_max: float | None = None
    persistent_lookback: int = 4
    opening_impulse_max_atr: float | None = None


@dataclass(frozen=True)
class ORBReferenceConfig:
    """Reference ORB configuration used in portfolio phase."""

    or_minutes: int = 30
    direction: str = "both"
    one_trade_per_day: bool = True
    entry_buffer_ticks: int = 1
    stop_buffer_ticks: int = 1
    target_multiple: float = 1.5
    opening_time: str = "09:30:00"
    time_exit: str = "15:55:00"
    atr_period: int = 14
    vwap_confirmation: bool = True
    vwap_column: str = "session_vwap"


@dataclass(frozen=True)
class MeanReversionCampaignSpec:
    """Top-level mean reversion campaign settings."""

    datasets_by_symbol: dict[str, Path]
    timeframes: tuple[TimeframeDefinition, ...] = DEFAULT_TIMEFRAMES
    is_fraction: float = DEFAULT_IS_FRACTION
    split_fractions: tuple[float, ...] = (0.60, 0.70, 0.80)
    session_start: str = DEFAULT_RTH_SESSION_START
    session_end: str = DEFAULT_RTH_SESSION_END
    prop_constraints: PropFirmConstraintConfig = build_default_prop_constraints()
    include_orb_baseline: bool = True
    orb_reference: ORBReferenceConfig = ORBReferenceConfig()
    max_validation_survivors: int = 12
    max_portfolio_candidates: int = 8
    output_root: Path = EXPORTS_DIR / "mean_reversion_intraday"


def _variant(
    name: str,
    family: str,
    symbol: str,
    timeframe: str,
    **kwargs: object,
) -> MeanReversionVariantConfig:
    return MeanReversionVariantConfig(
        name=name,
        family=family,
        symbol=symbol,
        timeframe=timeframe,
        **kwargs,
    )


def build_default_mean_reversion_variants() -> list[MeanReversionVariantConfig]:
    """Return the curated, asset-aware research universe."""
    variants: list[MeanReversionVariantConfig] = []

    for timeframe in ("5m", "15m"):
        variants.extend(
            [
                _variant(
                    name=f"mnq_{timeframe}_vwap_ext_atr_filtered",
                    family="vwap_extension_reversion",
                    symbol="MNQ",
                    timeframe=timeframe,
                    extension_mode="atr",
                    extension_threshold=1.60 if timeframe == "5m" else 1.80,
                    target_source="session_vwap",
                    stop_atr_multiple=1.10,
                    timeout_bars=5 if timeframe == "5m" else 4,
                    adx_max=18.0,
                    ema_slope_max_atr=0.22,
                    vwap_slope_max_atr=0.12,
                    anti_trend_day_max=2.60,
                    session_range_max_atr=4.60,
                    persistent_vwap_distance_max=1.80,
                    min_oos_trades=25 if timeframe == "5m" else 18,
                    max_trades_per_day=2,
                    notes="MNQ excess fade only when extension is large but intraday trend pressure stays moderate.",
                ),
                _variant(
                    name=f"mnq_{timeframe}_opening_stretch_or_mid_fade",
                    family="opening_stretch_fade",
                    symbol="MNQ",
                    timeframe=timeframe,
                    opening_window_minutes=45 if timeframe == "5m" else 60,
                    stretch_reference="or_midpoint",
                    stretch_threshold_atr=1.50 if timeframe == "5m" else 1.80,
                    stretch_threshold_or_multiple=1.20,
                    stretch_volume_z_max=1.75,
                    target_source="or_midpoint",
                    stop_atr_multiple=1.10,
                    timeout_bars=4,
                    adx_max=18.0,
                    ema_slope_max_atr=0.24,
                    anti_trend_day_max=2.40,
                    opening_impulse_max_atr=2.80,
                    max_trades_per_day=1,
                    min_oos_trades=18 if timeframe == "5m" else 14,
                    notes="Preferred MNQ fade: first impulse after the open is faded only when trend-day evidence stays weak.",
                ),
                _variant(
                    name=f"mnq_{timeframe}_streak4_vwap_snapback",
                    family="streak_exhaustion_reversion",
                    symbol="MNQ",
                    timeframe=timeframe,
                    streak_length=4 if timeframe == "5m" else 3,
                    streak_extension_atr=1.40 if timeframe == "5m" else 1.20,
                    require_exhaustion_bar=True,
                    anchor_distance_max_atr=1.20,
                    target_source="ema_20",
                    stop_atr_multiple=0.95,
                    timeout_bars=3,
                    adx_max=20.0,
                    ema_slope_max_atr=0.28,
                    anti_trend_day_max=2.80,
                    persistent_vwap_distance_max=1.90,
                    max_trades_per_day=2,
                    min_oos_trades=20 if timeframe == "5m" else 16,
                    notes="MNQ short-run exhaustion scalp, only when already extended away from session anchor.",
                ),
            ]
        )

    for timeframe in ("5m", "15m"):
        variants.extend(
            [
                _variant(
                    name=f"mes_{timeframe}_vwap_ext_zscore",
                    family="vwap_extension_reversion",
                    symbol="MES",
                    timeframe=timeframe,
                    extension_mode="zscore",
                    extension_threshold=2.10 if timeframe == "5m" else 1.90,
                    zscore_window=24 if timeframe == "5m" else 20,
                    target_source="session_vwap",
                    stop_atr_multiple=1.00,
                    timeout_bars=6 if timeframe == "5m" else 5,
                    use_partial_exit=timeframe == "5m",
                    fixed_quantity=2 if timeframe == "5m" else 1,
                    adx_max=21.0,
                    ema_slope_max_atr=0.24,
                    vwap_slope_max_atr=0.14,
                    anti_trend_day_max=2.80,
                    session_range_max_atr=5.00,
                    persistent_vwap_distance_max=2.00,
                    min_oos_trades=30 if timeframe == "5m" else 20,
                    notes="MES anchor reversion with z-score threshold and mild anti-trend guardrails.",
                ),
                _variant(
                    name=f"mes_{timeframe}_bollinger_20x2_immediate",
                    family="bollinger_zscore_reversion",
                    symbol="MES",
                    timeframe=timeframe,
                    bollinger_window=20,
                    bollinger_std=2.0,
                    bollinger_confirmation="immediate",
                    target_source="rolling_mean_20",
                    stop_atr_multiple=1.00,
                    timeout_bars=6 if timeframe == "5m" else 5,
                    adx_max=20.0,
                    ema_slope_max_atr=0.24,
                    anti_trend_day_max=2.70,
                    session_range_max_atr=4.80,
                    persistent_vwap_distance_max=1.80,
                    min_oos_trades=28 if timeframe == "5m" else 18,
                    notes="Classic MES Bollinger fade retained only in low directional pressure regimes.",
                ),
                _variant(
                    name=f"mes_{timeframe}_bollinger_30x2p5_reentry",
                    family="bollinger_zscore_reversion",
                    symbol="MES",
                    timeframe=timeframe,
                    bollinger_window=30,
                    bollinger_std=2.5,
                    bollinger_confirmation="reentry",
                    target_source="rolling_mean_30",
                    stop_atr_multiple=1.10,
                    timeout_bars=7 if timeframe == "5m" else 6,
                    adx_max=18.0,
                    ema_slope_max_atr=0.22,
                    anti_trend_day_max=2.60,
                    persistent_vwap_distance_max=1.70,
                    min_oos_trades=24 if timeframe == "5m" else 16,
                    notes="MES reentry inside wide bands, favored over immediate catch-the-falling-knife variants.",
                ),
                _variant(
                    name=f"mes_{timeframe}_rsi3_exit_extreme",
                    family="rsi_stochastic_contrarian",
                    symbol="MES",
                    timeframe=timeframe,
                    oscillator_kind="rsi",
                    oscillator_period=3,
                    oversold_level=12.0,
                    overbought_level=88.0,
                    oscillator_trigger="exit_extreme",
                    target_source="ema_20",
                    stop_atr_multiple=1.00,
                    timeout_bars=4,
                    anchor_distance_max_atr=1.10,
                    adx_max=18.0,
                    ema_slope_max_atr=0.20,
                    anti_trend_day_max=2.50,
                    min_oos_trades=18,
                    notes="Fast RSI fade around session anchor, only after the oscillator exits the extreme zone.",
                ),
            ]
        )

    for timeframe in ("5m", "15m"):
        variants.extend(
            [
                _variant(
                    name=f"m2k_{timeframe}_keltner_ema20_1p5",
                    family="keltner_band_snapback",
                    symbol="M2K",
                    timeframe=timeframe,
                    ema_window=20,
                    band_width_atr=1.50,
                    require_closes_outside=1,
                    target_source="ema_20",
                    stop_atr_multiple=1.00,
                    timeout_bars=5,
                    adx_max=21.0,
                    ema_slope_max_atr=0.28,
                    anti_trend_day_max=2.80,
                    persistent_vwap_distance_max=1.90,
                    min_oos_trades=20,
                    notes="M2K snapback after first close outside the adaptive Keltner envelope.",
                ),
                _variant(
                    name=f"m2k_{timeframe}_keltner_ema30_2p0_2closes",
                    family="keltner_band_snapback",
                    symbol="M2K",
                    timeframe=timeframe,
                    ema_window=30,
                    band_width_atr=2.00,
                    require_closes_outside=2,
                    target_source="ema_30",
                    stop_atr_multiple=1.10,
                    timeout_bars=6,
                    adx_max=19.0,
                    ema_slope_max_atr=0.25,
                    anti_trend_day_max=2.60,
                    persistent_vwap_distance_max=1.75,
                    min_oos_trades=16,
                    notes="M2K two-close outside-band snapback to avoid overtrading weak single-bar pokes.",
                ),
                _variant(
                    name=f"m2k_{timeframe}_opening_stretch_open_fade",
                    family="opening_stretch_fade",
                    symbol="M2K",
                    timeframe=timeframe,
                    opening_window_minutes=30 if timeframe == "5m" else 45,
                    stretch_reference="session_open",
                    stretch_threshold_atr=1.40 if timeframe == "5m" else 1.60,
                    stretch_threshold_or_multiple=1.10,
                    stretch_volume_z_max=1.80,
                    target_source="session_open",
                    stop_atr_multiple=1.05,
                    timeout_bars=4,
                    adx_max=18.0,
                    ema_slope_max_atr=0.24,
                    anti_trend_day_max=2.40,
                    opening_impulse_max_atr=2.60,
                    max_trades_per_day=1,
                    min_oos_trades=16,
                    notes="M2K opening stretch fade kept only when follow-through volume is not explosive.",
                ),
                _variant(
                    name=f"m2k_{timeframe}_streak3_local_mean",
                    family="streak_exhaustion_reversion",
                    symbol="M2K",
                    timeframe=timeframe,
                    streak_length=3,
                    streak_extension_atr=1.20,
                    require_exhaustion_bar=timeframe == "5m",
                    anchor_distance_max_atr=1.20,
                    target_source="rolling_mean_20",
                    stop_atr_multiple=0.95,
                    timeout_bars=3,
                    adx_max=20.0,
                    ema_slope_max_atr=0.28,
                    anti_trend_day_max=2.70,
                    min_oos_trades=18,
                    notes="M2K streak fade toward local mean, especially after a short impulsive run.",
                ),
            ]
        )

    variants.extend(
        [
            _variant(
                name="mgc_5m_vwap_ext_structural",
                family="vwap_extension_reversion",
                symbol="MGC",
                timeframe="5m",
                extension_mode="atr",
                extension_threshold=1.80,
                target_source="session_vwap",
                stop_atr_multiple=1.15,
                timeout_bars=6,
                adx_max=17.0,
                ema_slope_max_atr=0.18,
                vwap_slope_max_atr=0.10,
                anti_trend_day_max=2.20,
                persistent_vwap_distance_max=1.60,
                min_oos_trades=18,
                notes="MGC 5m structural excess fade kept very selective because gold trends cleanly when it moves.",
            ),
            _variant(
                name="mgc_5m_keltner_ema30_2p0",
                family="keltner_band_snapback",
                symbol="MGC",
                timeframe="5m",
                ema_window=30,
                band_width_atr=2.00,
                require_closes_outside=2,
                target_source="ema_30",
                stop_atr_multiple=1.10,
                timeout_bars=6,
                adx_max=17.0,
                ema_slope_max_atr=0.18,
                anti_trend_day_max=2.20,
                min_oos_trades=16,
                notes="Gold 5m snapback only after a cleaner two-close excursion beyond the outer Keltner band.",
            ),
            _variant(
                name="mgc_15m_rsi5_reversal",
                family="rsi_stochastic_contrarian",
                symbol="MGC",
                timeframe="15m",
                oscillator_kind="rsi",
                oscillator_period=5,
                oversold_level=18.0,
                overbought_level=82.0,
                oscillator_trigger="extreme_reversal",
                target_source="ema_30",
                stop_atr_multiple=1.15,
                timeout_bars=5,
                anchor_distance_max_atr=1.10,
                adx_max=18.0,
                ema_slope_max_atr=0.18,
                anti_trend_day_max=2.20,
                min_oos_trades=14,
                max_trades_per_day=1,
                notes="Gold 15m contrarian RSI reversal around a slow anchor, aligned with more structural mean reversion.",
            ),
            _variant(
                name="mgc_15m_stoch_8_3_3_exit_extreme",
                family="rsi_stochastic_contrarian",
                symbol="MGC",
                timeframe="15m",
                oscillator_kind="stochastic",
                oscillator_period_fast=8,
                oscillator_period_slow=3,
                oscillator_smoothing=3,
                oversold_level=20.0,
                overbought_level=80.0,
                oscillator_trigger="exit_extreme",
                target_source="session_vwap",
                stop_atr_multiple=1.10,
                timeout_bars=5,
                anchor_distance_max_atr=1.25,
                adx_max=18.0,
                ema_slope_max_atr=0.18,
                anti_trend_day_max=2.30,
                min_oos_trades=14,
                notes="Gold 15m stochastic pullback fade after the oscillator leaves the extreme zone.",
            ),
            _variant(
                name="mgc_15m_keltner_ema30_2p5",
                family="keltner_band_snapback",
                symbol="MGC",
                timeframe="15m",
                ema_window=30,
                band_width_atr=2.50,
                require_closes_outside=1,
                target_source="ema_30",
                stop_atr_multiple=1.15,
                timeout_bars=5,
                adx_max=18.0,
                ema_slope_max_atr=0.16,
                anti_trend_day_max=2.10,
                min_oos_trades=12,
                notes="Gold 15m snapback after a structural displacement beyond a wide ATR envelope.",
            ),
            _variant(
                name="mgc_15m_bollinger_30x2p5_reentry",
                family="bollinger_zscore_reversion",
                symbol="MGC",
                timeframe="15m",
                bollinger_window=30,
                bollinger_std=2.5,
                bollinger_confirmation="reentry",
                target_source="rolling_mean_30",
                stop_atr_multiple=1.15,
                timeout_bars=6,
                adx_max=17.0,
                ema_slope_max_atr=0.16,
                anti_trend_day_max=2.10,
                min_oos_trades=12,
                notes="Wide-band MGC 15m reentry used as a structural gold-style mean reversion block.",
            ),
        ]
    )

    return variants


def build_default_mean_reversion_campaign_spec() -> MeanReversionCampaignSpec:
    """Return the default campaign spec for the four requested micro futures."""
    datasets_by_symbol = {
        symbol: resolve_default_vwap_dataset(symbol)
        for symbol in ("MNQ", "MES", "M2K", "MGC")
    }
    return MeanReversionCampaignSpec(datasets_by_symbol=datasets_by_symbol)
