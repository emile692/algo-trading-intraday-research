"""Configuration objects for the ORB research campaign."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BaselineEntryConfig:
    """Baseline ORB entry settings (must remain backward compatible)."""

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
    account_size_usd: float = 50_000.0
    risk_per_trade_pct: float = 0.5
    tick_size: float = 0.25
    entry_on_next_open: bool = True


@dataclass(frozen=True)
class BaselineEnsembleConfig:
    """ATR-ensemble day-selection settings layered on top of entry signals."""

    atr_window: int = 14
    q_lows_pct: tuple[int, ...] = (20, 25, 30)
    q_highs_pct: tuple[int, ...] = (90, 95)
    vote_threshold: float = 0.5


@dataclass(frozen=True)
class CompressionConfig:
    """Compression overlay configuration."""

    mode: str = "none"
    usage: str = "hard_filter"  # hard_filter | soft_vote_bonus
    soft_bonus_votes: float = 1.0


@dataclass(frozen=True)
class ExitConfig:
    """Exit/trailing overlay configuration."""

    mode: str = "baseline"
    force_exit_time: str | None = None
    stagnation_bars: int | None = None
    stagnation_min_r_multiple: float = 0.15
    partial_fraction: float = 0.5


@dataclass(frozen=True)
class DynamicThresholdConfig:
    """Dynamic breakout/noise-gate configuration."""

    mode: str = "disabled"
    noise_lookback: int = 14
    noise_vm: float = 1.0
    threshold_style: str = "max_or_high_noise"  # max_or_high_noise | or_high_plus_k_noise_abs
    noise_k: float = 0.0
    atr_k: float = 0.0
    confirm_bars: int = 1
    schedule: str = "continuous_on_bar_close"  # continuous_on_bar_close | every_5m | every_15m


@dataclass(frozen=True)
class ExperimentConfig:
    """Full experiment definition used by the campaign executor."""

    name: str
    stage: str
    family: str
    baseline_entry: BaselineEntryConfig
    baseline_ensemble: BaselineEnsembleConfig
    compression: CompressionConfig
    exit: ExitConfig
    dynamic_threshold: DynamicThresholdConfig


@dataclass(frozen=True)
class CampaignConfig:
    """Top-level campaign configuration."""

    dataset_path: Path
    output_dir: Path
    is_fraction: float = 0.70
    random_seed: int = 42
    max_full_reopt_trials: int = 500
    rolling_window_days_for_stability: int = 60
    oos_top_n: int = 20
    bootstrap_paths: int = 3000


@dataclass
class CampaignContext:
    """Runtime cache shared by experiment evaluators."""

    all_sessions: list
    is_sessions: list
    oos_sessions: list
    minute_df: object
    candidate_base_df: object
    daily_patterns: object
    noise_cache: dict[int, object] = field(default_factory=dict)
