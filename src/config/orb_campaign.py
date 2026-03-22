"""Experiment definitions for the ORB research campaign."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.paths import DOWNLOADED_DATA_DIR
from src.config.settings import (
    DEFAULT_COMMISSION_PER_SIDE_USD,
    DEFAULT_INITIAL_CAPITAL_USD,
    DEFAULT_POINT_VALUE_USD,
    DEFAULT_SLIPPAGE_TICKS,
    DEFAULT_TICK_SIZE,
    DEFAULT_TICK_VALUE_USD,
)

DEFAULT_CAMPAIGN_DATASET = DOWNLOADED_DATA_DIR / "MNQ_1mim.parquet"
DEFAULT_OPENING_TIME = "09:30:00"
DEFAULT_TIME_EXIT = "16:00:00"
DEFAULT_STOP_BUFFER_TICKS = 2
DEFAULT_PROP_RISK_PCT = 0.25
DEFAULT_PAPER_RISK_PCT = 1.0
DEFAULT_PAPER_INITIAL_CAPITAL_USD = 25_000.0
DEFAULT_MONTHLY_SUBSCRIPTION_COST_USD = 150.0


@dataclass(frozen=True)
class ExecutionProfile:
    """Execution and cost assumptions used by an experiment."""

    name: str
    commission_per_side_usd: float
    slippage_ticks: int
    tick_size: float = DEFAULT_TICK_SIZE


@dataclass(frozen=True)
class AtrRegimeDefinition:
    """Named ATR regime resolved from dataset quantiles."""

    name: str
    lower_quantile: float | None
    upper_quantile: float | None


@dataclass(frozen=True)
class RankingConfig:
    """Weights and guardrails for the robustness leaderboard."""

    min_trades: int = 40
    target_days_traded: float = 0.18
    avg_r_weight: float = 35.0
    profit_factor_weight: float = 18.0
    expectancy_weight: float = 12.0
    drawdown_weight: float = 30.0
    loss_streak_weight: float = 4.0
    participation_weight: float = 12.0
    insufficient_trades_penalty: float = 30.0


@dataclass(frozen=True)
class PropConstraintConfig:
    """Topstep-style research constraints used for practical ranking."""

    name: str = "topstep_50k_reference"
    account_size_usd: float = DEFAULT_INITIAL_CAPITAL_USD
    max_loss_limit_usd: float = 2_000.0
    daily_loss_limit_usd: float | None = 1_000.0
    profit_target_usd: float = 3_000.0
    monthly_subscription_cost_usd: float = DEFAULT_MONTHLY_SUBSCRIPTION_COST_USD
    trading_days_per_month: float = 21.0
    daily_loss_limit_basis: str = "realized_daily_pnl"


@dataclass(frozen=True)
class FocusedRankingConfig:
    """Prop-oriented weights for the focused robustness campaign."""

    min_trades: int = 120
    target_days_traded: float = 0.10
    target_months_to_goal: float = 3.0
    acceptable_loss_streak: float = 7.0
    profit_factor_weight: float = 18.0
    expectancy_weight: float = 16.0
    trade_count_weight: float = 10.0
    participation_weight: float = 8.0
    target_reached_weight: float = 20.0
    target_speed_weight: float = 12.0
    drawdown_penalty_weight: float = 24.0
    loss_streak_penalty_weight: float = 10.0
    max_loss_breach_penalty_weight: float = 18.0
    daily_loss_breach_penalty_weight: float = 6.0
    subscription_drag_penalty_weight: float = 8.0
    insufficient_trades_penalty_weight: float = 12.0


@dataclass(frozen=True)
class ORBExperiment:
    """Explicit experiment definition for a single ORB run."""

    name: str
    axis: str
    group: str
    strategy_variant: str
    dataset_key: str
    execution_profile: str
    or_minutes: int
    target_multiple: float
    side_mode: str = "both"
    entry_buffer_ticks: int = 0
    stop_buffer_ticks: int = DEFAULT_STOP_BUFFER_TICKS
    opening_time: str = DEFAULT_OPENING_TIME
    time_exit: str = DEFAULT_TIME_EXIT
    one_trade_per_day: bool = True
    risk_per_trade_pct: float | None = None
    initial_capital_usd: float = DEFAULT_INITIAL_CAPITAL_USD
    tick_value_usd: float = DEFAULT_TICK_VALUE_USD
    point_value_usd: float = DEFAULT_POINT_VALUE_USD
    max_leverage: float | None = None
    atr_period: int = 14
    atr_regime: str = "none"
    atr_min: float | None = None
    atr_max: float | None = None
    direction_filter_mode: str = "none"
    ema_length: int | None = None
    entry_on_next_open: bool = True
    notes: str = ""


def build_execution_profiles() -> dict[str, ExecutionProfile]:
    """Return reusable execution/cost profiles."""
    return {
        "repo_realistic": ExecutionProfile(
            name="repo_realistic",
            commission_per_side_usd=DEFAULT_COMMISSION_PER_SIDE_USD,
            slippage_ticks=DEFAULT_SLIPPAGE_TICKS,
            tick_size=DEFAULT_TICK_SIZE,
        ),
        "paper_reference": ExecutionProfile(
            name="paper_reference",
            commission_per_side_usd=0.0005,
            slippage_ticks=0,
            tick_size=DEFAULT_TICK_SIZE,
        ),
    }


def build_atr_regimes() -> dict[str, AtrRegimeDefinition]:
    """Return structured ATR regime bands resolved at runtime from quantiles."""
    return {
        "none": AtrRegimeDefinition("none", None, None),
        "band_1": AtrRegimeDefinition("band_1", 0.0, 1.0 / 3.0),
        "band_2": AtrRegimeDefinition("band_2", 1.0 / 3.0, 2.0 / 3.0),
        "band_3": AtrRegimeDefinition("band_3", 2.0 / 3.0, 1.0),
    }


def build_ranking_config() -> RankingConfig:
    """Return the default robustness-scoring configuration."""
    return RankingConfig()


def build_prop_constraints() -> PropConstraintConfig:
    """Return the default Topstep-style research constraints."""
    return PropConstraintConfig()


def build_focused_ranking_config() -> FocusedRankingConfig:
    """Return the default ranking config for the focused prop campaign."""
    return FocusedRankingConfig()


def _risk_suffix(risk_per_trade_pct: float) -> str:
    return str(risk_per_trade_pct).replace(".", "p")


def build_orb_experiments(dataset_path: Path = DEFAULT_CAMPAIGN_DATASET) -> list[ORBExperiment]:
    """Build the structured paper, baseline, and filter-campaign experiment list."""
    experiments: list[ORBExperiment] = []

    paper_notes = (
        f"Default dataset is {dataset_path.name}; paper logic is replicated on 5-minute resampled bars, "
        "but instrument and cost assumptions may still differ from the original paper."
    )
    paper_base = dict(
        axis="axis_a_paper_replication",
        group="group_1_paper_replication",
        strategy_variant="paper_exact",
        dataset_key="paper_5m_rth",
        or_minutes=5,
        target_multiple=10.0,
        side_mode="both",
        stop_buffer_ticks=0,
        opening_time=DEFAULT_OPENING_TIME,
        time_exit=DEFAULT_TIME_EXIT,
        one_trade_per_day=True,
        risk_per_trade_pct=DEFAULT_PAPER_RISK_PCT,
        initial_capital_usd=DEFAULT_PAPER_INITIAL_CAPITAL_USD,
        notes=paper_notes,
    )
    experiments.extend(
        [
            ORBExperiment(
                name="paper_exact_reference_costs_cap4",
                execution_profile="paper_reference",
                max_leverage=4.0,
                **paper_base,
            ),
            ORBExperiment(
                name="paper_exact_repo_costs_cap4",
                execution_profile="repo_realistic",
                max_leverage=4.0,
                **paper_base,
            ),
            ORBExperiment(
                name="paper_exact_reference_costs_no_leverage_cap",
                execution_profile="paper_reference",
                max_leverage=None,
                **paper_base,
            ),
            ORBExperiment(
                name="paper_exact_repo_costs_no_leverage_cap",
                execution_profile="repo_realistic",
                max_leverage=None,
                **paper_base,
            ),
        ]
    )

    baseline_base = dict(
        axis="axis_b_current_logic_baselines",
        group="group_2_current_logic_baselines",
        strategy_variant="current_orb",
        dataset_key="current_1m_rth",
        execution_profile="repo_realistic",
        or_minutes=15,
        entry_buffer_ticks=0,
        stop_buffer_ticks=DEFAULT_STOP_BUFFER_TICKS,
        opening_time=DEFAULT_OPENING_TIME,
        time_exit=DEFAULT_TIME_EXIT,
        one_trade_per_day=True,
        risk_per_trade_pct=DEFAULT_PROP_RISK_PCT,
        initial_capital_usd=DEFAULT_INITIAL_CAPITAL_USD,
    )
    for side_mode in ("both", "long_only"):
        for target_multiple in (10.0, 5.0, 3.0):
            experiments.append(
                ORBExperiment(
                    name=f"current_15m_{side_mode}_rr{int(target_multiple)}",
                    target_multiple=target_multiple,
                    side_mode=side_mode,
                    **baseline_base,
                )
            )

    prop_base = dict(
        axis="axis_b_prop_viable",
        group="group_2_prop_risk_grid",
        strategy_variant="current_orb",
        dataset_key="current_1m_rth",
        execution_profile="repo_realistic",
        or_minutes=15,
        entry_buffer_ticks=0,
        stop_buffer_ticks=DEFAULT_STOP_BUFFER_TICKS,
        opening_time=DEFAULT_OPENING_TIME,
        time_exit=DEFAULT_TIME_EXIT,
        one_trade_per_day=True,
        initial_capital_usd=DEFAULT_INITIAL_CAPITAL_USD,
    )
    for side_mode in ("both", "long_only"):
        for target_multiple in (5.0, 3.0):
            for risk_per_trade_pct in (0.10, 0.15, 0.25, 0.50):
                name = f"prop_baseline_{side_mode}_rr{int(target_multiple)}"
                if risk_per_trade_pct != DEFAULT_PROP_RISK_PCT:
                    name = f"{name}_risk_{_risk_suffix(risk_per_trade_pct)}"

                experiments.append(
                    ORBExperiment(
                        name=name,
                        target_multiple=target_multiple,
                        side_mode=side_mode,
                        risk_per_trade_pct=risk_per_trade_pct,
                        **prop_base,
                    )
                )

    filter_base = dict(
        axis="axis_c_filter_campaign",
        group="group_3_filter_campaign",
        strategy_variant="current_orb",
        dataset_key="current_1m_rth",
        execution_profile="repo_realistic",
        or_minutes=15,
        entry_buffer_ticks=0,
        stop_buffer_ticks=DEFAULT_STOP_BUFFER_TICKS,
        opening_time=DEFAULT_OPENING_TIME,
        time_exit=DEFAULT_TIME_EXIT,
        one_trade_per_day=True,
        target_multiple=5.0,
        risk_per_trade_pct=DEFAULT_PROP_RISK_PCT,
        initial_capital_usd=DEFAULT_INITIAL_CAPITAL_USD,
        atr_period=14,
    )
    for side_mode in ("both", "long_only"):
        for atr_regime in ("none", "band_1", "band_2", "band_3"):
            for direction_filter_mode in ("none", "vwap_only", "ema_only", "vwap_and_ema"):
                ema_lengths = (None,)
                if direction_filter_mode in ("ema_only", "vwap_and_ema"):
                    ema_lengths = (20, 50)

                for ema_length in ema_lengths:
                    name = f"filter_current_{side_mode}_{atr_regime}_{direction_filter_mode}"
                    if ema_length is not None:
                        name = f"{name}_ema{ema_length}"

                    experiments.append(
                        ORBExperiment(
                            name=name,
                            side_mode=side_mode,
                            atr_regime=atr_regime,
                            direction_filter_mode=direction_filter_mode,
                            ema_length=ema_length,
                            **filter_base,
                        )
                    )

    return experiments


def build_focused_atr_regimes() -> dict[str, AtrRegimeDefinition]:
    """Return monotonic ATR filters for the focused practical campaign."""
    return {
        "none": AtrRegimeDefinition("none", None, None),
        "moderate_band": AtrRegimeDefinition("moderate_band", 0.50, 1.0),
        "restrictive_band": AtrRegimeDefinition("restrictive_band", 2.0 / 3.0, 1.0),
    }


def build_focused_orb_experiments(dataset_path: Path = DEFAULT_CAMPAIGN_DATASET) -> list[ORBExperiment]:
    """Build the explicit focused grid around the best long-only EMA branch."""
    experiments: list[ORBExperiment] = []

    base = dict(
        axis="focused_prop_branch",
        group="focused_long_only_ema_atr_prop",
        strategy_variant="current_orb",
        dataset_key="current_1m_rth",
        execution_profile="repo_realistic",
        or_minutes=15,
        entry_buffer_ticks=0,
        stop_buffer_ticks=DEFAULT_STOP_BUFFER_TICKS,
        opening_time=DEFAULT_OPENING_TIME,
        time_exit=DEFAULT_TIME_EXIT,
        one_trade_per_day=True,
        side_mode="long_only",
        initial_capital_usd=DEFAULT_INITIAL_CAPITAL_USD,
        atr_period=14,
        direction_filter_mode="ema_only",
        notes=(
            f"Focused practical campaign on {dataset_path.name}: long-only current ORB, "
            "EMA directional filter, optional ATR regime filter, and prop-style sizing."
        ),
    )

    for target_multiple in (3.0, 4.0, 5.0):
        for ema_length in (30, 50, 70, 100):
            for atr_regime in ("none", "moderate_band", "restrictive_band"):
                for risk_per_trade_pct in (0.10, 0.15, 0.20, 0.25):
                    experiments.append(
                        ORBExperiment(
                            name=(
                                f"focused_long_only_rr{int(target_multiple)}_ema{ema_length}_"
                                f"{atr_regime}_risk_{_risk_suffix(risk_per_trade_pct)}"
                            ),
                            target_multiple=target_multiple,
                            ema_length=ema_length,
                            atr_regime=atr_regime,
                            risk_per_trade_pct=risk_per_trade_pct,
                            **base,
                        )
                    )

    return experiments
