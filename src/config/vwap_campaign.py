"""Configuration objects for the VWAP research campaign."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

from src.config.paths import DOWNLOADED_DATA_DIR, PROCESSED_DATA_DIR
from src.config.settings import DEFAULT_INITIAL_CAPITAL_USD, DEFAULT_SYMBOL

DEFAULT_PAPER_INITIAL_CAPITAL_USD = 25_000.0
DEFAULT_IS_FRACTION = 0.70
DEFAULT_PAPER_TIME_EXIT = "16:00:00"
DEFAULT_RTH_SESSION_START = "09:30:00"
DEFAULT_RTH_SESSION_END = "16:00:00"


@dataclass(frozen=True)
class TimeWindow:
    """Intraday trading window in local market time."""

    start: str
    end: str


@dataclass(frozen=True)
class PropFirmConstraintConfig:
    """Practical guardrails used to evaluate prop-style viability."""

    name: str = "generic_prop_reference"
    account_size_usd: float = DEFAULT_INITIAL_CAPITAL_USD
    profit_target_pct: float = 0.06
    daily_loss_limit_usd: float = 1_000.0
    trailing_drawdown_limit_usd: float = 2_000.0
    consecutive_red_days_threshold: int = 3
    trading_days_per_month: float = 21.0


@dataclass(frozen=True)
class VWAPVariantConfig:
    """Explicit strategy variant definition used by the campaign."""

    name: str
    family: str
    mode: str
    execution_profile: str
    initial_capital_usd: float
    quantity_mode: str = "fixed_quantity"
    fixed_quantity: int = 1
    time_windows: tuple[TimeWindow, ...] = ()
    slope_lookback: int = 5
    slope_threshold: float = 0.0
    require_vwap_slope_alignment: bool = False
    max_vwap_distance_atr: float | None = None
    atr_period: int = 14
    atr_buffer: float = 0.25
    stop_buffer: float | None = None
    compression_length: int = 3
    pullback_lookback: int = 6
    confirmation_threshold: float = 0.0
    max_trades_per_day: int | None = None
    max_losses_per_day: int | None = None
    daily_stop_threshold_usd: float | None = None
    consecutive_losses_threshold: int | None = None
    deleverage_after_losing_streak: float = 1.0
    risk_per_trade_pct: float | None = None
    exit_on_vwap_recross: bool = True
    use_partial_exit: bool = False
    partial_exit_r_multiple: float = 1.0
    keep_runner_until_close: bool = True
    notes: str = ""


@dataclass(frozen=True)
class VWAPCampaignSpec:
    """Top-level campaign settings."""

    dataset_path: Path
    is_fraction: float = DEFAULT_IS_FRACTION
    session_start: str = DEFAULT_RTH_SESSION_START
    session_end: str = DEFAULT_RTH_SESSION_END
    paper_time_exit: str = DEFAULT_PAPER_TIME_EXIT
    prop_constraints: PropFirmConstraintConfig = PropFirmConstraintConfig()
    rolling_window_days: int = 20
    sensitivity_cost_multipliers: tuple[float, ...] = (0.75, 1.0, 1.25)
    sensitivity_atr_buffers: tuple[float, ...] = (0.15, 0.25, 0.40)
    sensitivity_slope_thresholds: tuple[float, ...] = (0.0, 0.01, 0.02)
    sensitivity_max_trades_per_day: tuple[int, ...] = (1, 2)


def infer_symbol_from_dataset_path(dataset_path: Path | str) -> str:
    """Infer the traded symbol from the dataset filename."""
    path = Path(dataset_path)
    stem = path.stem
    token = stem.split("_")[0].strip().upper()
    if not token:
        raise ValueError(f"Unable to infer symbol from dataset path: {path}")
    return token


def resolve_default_vwap_dataset(symbol: str = DEFAULT_SYMBOL) -> Path:
    """Return the latest available 1-minute dataset for the requested symbol."""
    upper_symbol = symbol.upper()
    candidates: list[Path] = []

    processed_parquet_dir = PROCESSED_DATA_DIR / "parquet"
    if processed_parquet_dir.exists():
        candidates.extend(sorted(processed_parquet_dir.glob(f"{upper_symbol}_*_1m_*.parquet")))

    if DOWNLOADED_DATA_DIR.exists():
        candidates.extend(sorted(DOWNLOADED_DATA_DIR.glob(f"{upper_symbol}_*_1m*.parquet")))

    if not candidates:
        raise FileNotFoundError(
            f"No 1-minute dataset found for symbol '{upper_symbol}' in "
            f"{processed_parquet_dir} or {DOWNLOADED_DATA_DIR}."
        )

    return sorted(candidates)[-1]


def build_default_prop_constraints(account_size_usd: float = DEFAULT_INITIAL_CAPITAL_USD) -> PropFirmConstraintConfig:
    """Return the default prop-style research constraints."""
    return PropFirmConstraintConfig(account_size_usd=account_size_usd)


def build_default_vwap_variants() -> list[VWAPVariantConfig]:
    """Return the paper baseline plus the requested prop-oriented variants."""
    return [
        VWAPVariantConfig(
            name="paper_vwap_baseline",
            family="baseline",
            mode="target_position",
            execution_profile="paper_reference",
            initial_capital_usd=DEFAULT_PAPER_INITIAL_CAPITAL_USD,
            quantity_mode="paper_full_notional",
            fixed_quantity=1,
            notes=(
                "Exact paper logic: previous close relative to session VWAP drives the next bar "
                "position, always in market during RTH, flat overnight."
            ),
        ),
        VWAPVariantConfig(
            name="baseline_futures_adapted",
            family="baseline",
            mode="target_position",
            execution_profile="repo_realistic",
            initial_capital_usd=DEFAULT_INITIAL_CAPITAL_USD,
            quantity_mode="fixed_quantity",
            fixed_quantity=1,
            notes="Same signal logic as the paper baseline, but priced with realistic futures costs.",
        ),
        VWAPVariantConfig(
            name="vwap_time_filtered_baseline",
            family="prop_variant",
            mode="target_position",
            execution_profile="repo_realistic",
            initial_capital_usd=DEFAULT_INITIAL_CAPITAL_USD,
            quantity_mode="fixed_quantity",
            fixed_quantity=1,
            time_windows=(
                TimeWindow("09:35:00", "11:30:00"),
                TimeWindow("15:00:00", "15:50:00"),
            ),
            notes="Paper signal with allowed entries and reversals only in the profitable time buckets.",
        ),
        VWAPVariantConfig(
            name="vwap_baseline_trade_capped",
            family="prop_variant",
            mode="target_position",
            execution_profile="repo_realistic",
            initial_capital_usd=DEFAULT_INITIAL_CAPITAL_USD,
            quantity_mode="fixed_quantity",
            fixed_quantity=1,
            max_trades_per_day=6,
            notes="Paper signal with a hard cap on the number of daily flips/trades.",
        ),
        VWAPVariantConfig(
            name="vwap_baseline_regime_filtered",
            family="prop_variant",
            mode="target_position",
            execution_profile="repo_realistic",
            initial_capital_usd=DEFAULT_INITIAL_CAPITAL_USD,
            quantity_mode="fixed_quantity",
            fixed_quantity=1,
            slope_lookback=5,
            slope_threshold=0.0,
            require_vwap_slope_alignment=True,
            max_vwap_distance_atr=1.0,
            atr_period=14,
            max_trades_per_day=8,
            notes="Paper signal gated by previous-bar VWAP slope alignment and a simple distance-to-VWAP filter.",
        ),
        VWAPVariantConfig(
            name="vwap_baseline_with_killswitch",
            family="prop_variant",
            mode="target_position",
            execution_profile="repo_realistic",
            initial_capital_usd=DEFAULT_INITIAL_CAPITAL_USD,
            quantity_mode="fixed_quantity",
            fixed_quantity=1,
            max_losses_per_day=3,
            daily_stop_threshold_usd=750.0,
            max_trades_per_day=12,
            notes="Paper signal with daily loss and trade-count controls.",
        ),
        VWAPVariantConfig(
            name="vwap_reclaim",
            family="prop_variant",
            mode="discrete",
            execution_profile="repo_realistic",
            initial_capital_usd=DEFAULT_INITIAL_CAPITAL_USD,
            quantity_mode="fixed_quantity",
            fixed_quantity=1,
            slope_lookback=5,
            slope_threshold=0.0,
            atr_period=14,
            atr_buffer=0.25,
            stop_buffer=0.25,
            compression_length=4,
            pullback_lookback=6,
            max_trades_per_day=3,
            exit_on_vwap_recross=True,
            notes="VWAP reclaim after compression, confirmed by positive or negative VWAP slope.",
        ),
        VWAPVariantConfig(
            name="vwap_pullback_continuation",
            family="prop_variant",
            mode="discrete",
            execution_profile="repo_realistic",
            initial_capital_usd=DEFAULT_INITIAL_CAPITAL_USD,
            quantity_mode="fixed_quantity",
            fixed_quantity=1,
            slope_lookback=5,
            slope_threshold=0.0,
            atr_period=14,
            atr_buffer=0.30,
            stop_buffer=0.30,
            compression_length=3,
            pullback_lookback=8,
            confirmation_threshold=0.0,
            max_trades_per_day=3,
            exit_on_vwap_recross=True,
            notes="Trend continuation after a contained pullback inside a valid VWAP regime.",
        ),
        VWAPVariantConfig(
            name="vwap_reclaim_with_prop_overlay",
            family="prop_variant",
            mode="discrete",
            execution_profile="repo_realistic",
            initial_capital_usd=DEFAULT_INITIAL_CAPITAL_USD,
            quantity_mode="fixed_quantity",
            fixed_quantity=1,
            time_windows=(
                TimeWindow("09:35:00", "11:30:00"),
                TimeWindow("15:00:00", "15:50:00"),
            ),
            slope_lookback=5,
            slope_threshold=0.0,
            atr_period=14,
            atr_buffer=0.25,
            stop_buffer=0.25,
            compression_length=4,
            pullback_lookback=6,
            max_trades_per_day=2,
            max_losses_per_day=2,
            daily_stop_threshold_usd=500.0,
            consecutive_losses_threshold=2,
            deleverage_after_losing_streak=0.5,
            exit_on_vwap_recross=True,
            notes=(
                "Reclaim variant with time filter, hard daily stop, trade cap, and position "
                "deleveraging after a losing streak."
            ),
        ),
    ]


def resolve_vwap_variant(variant_name: str) -> VWAPVariantConfig:
    """Resolve a VWAP variant by name."""
    for variant in build_default_vwap_variants():
        if variant.name == variant_name:
            return variant
    raise ValueError(f"Unknown VWAP variant '{variant_name}'.")


def build_default_vwap_reranking_variants() -> list[VWAPVariantConfig]:
    """Return the compact, leak-free reranking universe used for variant selection."""
    variant_map = {variant.name: variant for variant in build_default_vwap_variants()}
    ordered_names = (
        "paper_vwap_baseline",
        "baseline_futures_adapted",
        "vwap_time_filtered_baseline",
        "vwap_baseline_trade_capped",
        "vwap_baseline_regime_filtered",
        "vwap_baseline_with_killswitch",
        "vwap_reclaim",
        "vwap_reclaim_with_prop_overlay",
    )
    return [variant_map[name] for name in ordered_names]


def build_default_vwap_timeframe_comparison_variants() -> list[VWAPVariantConfig]:
    """Return the small 1m vs 5m comparison universe."""
    variant_map = {variant.name: variant for variant in build_default_vwap_variants()}
    ordered_names = (
        "paper_vwap_baseline",
        "baseline_futures_adapted",
        "vwap_reclaim",
    )
    return [variant_map[name] for name in ordered_names]


def _scaled_bar_parameter(value: int, ratio: float, minimum: int) -> int:
    if value <= 0:
        return value
    return max(minimum, int(math.ceil(float(value) / float(ratio))))


def adapt_vwap_variant_to_timeframe(
    variant: VWAPVariantConfig,
    bar_minutes: int,
    base_bar_minutes: int = 1,
) -> VWAPVariantConfig:
    """Rescale bar-count parameters to preserve approximate elapsed-time horizons."""
    if bar_minutes <= base_bar_minutes:
        return variant

    ratio = float(bar_minutes) / float(base_bar_minutes)
    adapted = replace(
        variant,
        slope_lookback=_scaled_bar_parameter(variant.slope_lookback, ratio=ratio, minimum=1),
        atr_period=_scaled_bar_parameter(variant.atr_period, ratio=ratio, minimum=2),
        compression_length=_scaled_bar_parameter(variant.compression_length, ratio=ratio, minimum=2),
        pullback_lookback=_scaled_bar_parameter(variant.pullback_lookback, ratio=ratio, minimum=2),
        notes=(
            f"{variant.notes} Minimal timeframe adaptation: bar-count lookbacks were rescaled from "
            f"{base_bar_minutes}m to {bar_minutes}m to keep approximately comparable elapsed-time horizons, "
            "without retuning thresholds, buffers, session windows, or costs."
        ).strip(),
    )
    return adapted
