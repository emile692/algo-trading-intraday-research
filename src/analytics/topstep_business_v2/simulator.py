"""Topstep business v2 simulator with calendar-aware combine and funded phases."""

from __future__ import annotations

import calendar
import datetime as dt
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.mnq_orb_prop_challenge_simulation import _normalize_daily_results

AVERAGE_CALENDAR_DAYS_PER_MONTH = 365.25 / 12.0


@dataclass(frozen=True)
class TopstepBusinessPlan:
    """Fee schedule for a Topstep-style plan."""

    name: str
    monthly_subscription_usd: float
    reset_cost_usd: float
    activation_fee_usd: float


@dataclass(frozen=True)
class BusinessV2Rules:
    """Core Topstep business-model assumptions."""

    starting_balance_usd: float = 50_000.0
    challenge_profit_target_usd: float = 3_000.0
    trailing_mll_usd: float = 2_000.0
    daily_loss_limit_usd: float = 1_000.0
    consistency_share_limit: float = 0.50
    consistency_epsilon_usd: float = 0.01
    funded_trading_days: int = 60
    payout_threshold_usd: float = 1_000.0
    n_resets_per_month: int = 1
    challenge_timeout_calendar_days: int = 365
    challenge_timeout_trading_days: int = 252
    bootstrap_block_size: int = 5
    bootstrap_min_buffer_sessions: int = 40
    max_total_budget_usd: float | None = None


@dataclass(frozen=True)
class StrategySeries:
    """Normalized daily audited series for one strategy variant."""

    strategy_name: str
    source_variant_name: str
    leverage_factor: float
    session_dates: tuple[dt.date, ...]
    daily_pnl: np.ndarray
    daily_trade_count: np.ndarray
    reference_account_size_usd: float = 50_000.0

    @property
    def session_ordinals(self) -> np.ndarray:
        return np.fromiter((value.toordinal() for value in self.session_dates), dtype=np.int64)

    @property
    def trade_day_fraction(self) -> float:
        if len(self.daily_trade_count) == 0:
            return 0.0
        return float(np.mean(self.daily_trade_count > 0.0))


@dataclass(frozen=True)
class DailyPath:
    """Concrete date-aligned path used by one simulation cycle."""

    session_dates: tuple[dt.date, ...]
    daily_pnl: np.ndarray
    daily_trade_count: np.ndarray

    @property
    def session_ordinals(self) -> np.ndarray:
        return np.fromiter((value.toordinal() for value in self.session_dates), dtype=np.int64)


@dataclass
class BusinessCycleArtifacts:
    """Per-cycle outputs including diagnostics rows."""

    cycle_row: dict[str, Any]
    challenge_attempt_rows: list[dict[str, Any]]
    funded_rows: list[dict[str, Any]]


STANDARD_PLAN = TopstepBusinessPlan(
    name="STANDARD",
    monthly_subscription_usd=49.0,
    reset_cost_usd=49.0,
    activation_fee_usd=149.0,
)

NO_ACTIVATION_FEE_PLAN = TopstepBusinessPlan(
    name="NO_ACTIVATION_FEE",
    monthly_subscription_usd=109.0,
    reset_cost_usd=109.0,
    activation_fee_usd=0.0,
)

DEFAULT_PLANS = (STANDARD_PLAN, NO_ACTIVATION_FEE_PLAN)


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(float(denominator)):
        return float(default)
    value = float(numerator) / float(denominator)
    return float(value) if math.isfinite(value) else float(default)


def build_strategy_name(base_name: str, leverage_factor: float) -> str:
    """Return a stable strategy label with leverage suffix when needed."""
    if abs(float(leverage_factor) - 1.0) < 1e-12:
        return str(base_name)
    return f"{base_name}_x{float(leverage_factor):.1f}"


def prepare_strategy_series(
    daily_results: pd.DataFrame,
    strategy_name: str,
    source_variant_name: str,
    leverage_factor: float = 1.0,
    reference_account_size_usd: float = 50_000.0,
) -> StrategySeries:
    """Normalize a daily audited frame into a compact array-backed series."""
    ordered = _normalize_daily_results(daily_results)
    factor = float(leverage_factor)
    pnl = pd.to_numeric(ordered["daily_pnl_usd"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if abs(factor - 1.0) > 1e-12:
        pnl = pnl * factor
    daily_trade_count = (
        pd.to_numeric(ordered["daily_trade_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    )
    session_dates = tuple(pd.to_datetime(ordered["session_date"]).dt.date.tolist())
    return StrategySeries(
        strategy_name=build_strategy_name(strategy_name, leverage_factor),
        source_variant_name=source_variant_name,
        leverage_factor=factor,
        session_dates=session_dates,
        daily_pnl=pnl,
        daily_trade_count=daily_trade_count,
        reference_account_size_usd=float(reference_account_size_usd),
    )


def build_historical_path(series: StrategySeries, start_index: int) -> DailyPath:
    """Return the realized path starting from one common OOS index."""
    return DailyPath(
        session_dates=series.session_dates[int(start_index) :],
        daily_pnl=series.daily_pnl[int(start_index) :],
        daily_trade_count=series.daily_trade_count[int(start_index) :],
    )


def required_profit_to_pass(
    best_day_profit_usd: float,
    profit_target_usd: float,
    epsilon_usd: float,
) -> float:
    """Dynamic required profit enforcing strict best-day share below 50%."""
    return float(max(float(profit_target_usd), 2.0 * max(float(best_day_profit_usd), 0.0) + float(epsilon_usd)))


def _month_key(value: dt.date) -> str:
    return f"{value.year:04d}-{value.month:02d}"


def _month_end(value: dt.date) -> dt.date:
    last_day = calendar.monthrange(value.year, value.month)[1]
    return dt.date(value.year, value.month, last_day)


def _days_in_month(value: dt.date) -> int:
    return calendar.monthrange(value.year, value.month)[1]


def _shift_days(value: dt.date, days: int) -> dt.date:
    return value + dt.timedelta(days=int(days))


def _count_months_between(start_date: dt.date, end_date: dt.date) -> int:
    if end_date < start_date:
        return 0
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1


def _trailing_floor(high_watermark_usd: float, rules: BusinessV2Rules) -> float:
    floor_value = float(high_watermark_usd) - float(rules.trailing_mll_usd)
    return float(min(float(rules.starting_balance_usd), floor_value))


def _find_next_index_after(path: DailyPath, date_value: dt.date, start_index: int) -> int:
    ordinals = path.session_ordinals
    target = int(date_value.toordinal())
    return int(np.searchsorted(ordinals, target, side="right"))


def _prorated_range_cost(start_date: dt.date, end_date: dt.date, monthly_fee_usd: float) -> float:
    if end_date < start_date:
        return 0.0
    total = 0.0
    cursor = start_date
    while cursor <= end_date:
        segment_end = min(_month_end(cursor), end_date)
        nb_days = (segment_end - cursor).days + 1
        total += float(monthly_fee_usd) * float(nb_days) / float(_days_in_month(cursor))
        cursor = _shift_days(segment_end, 1)
    return float(total)


def _accrue_subscription_cost(
    billed_through_date: dt.date | None,
    target_end_date: dt.date,
    monthly_fee_usd: float,
    total_spend_before_accrual_usd: float,
    current_subscription_cost_usd: float,
    max_total_budget_usd: float | None,
) -> dict[str, Any]:
    start_date = target_end_date if billed_through_date is None else _shift_days(billed_through_date, 1)
    if billed_through_date is not None and target_end_date <= billed_through_date:
        return {
            "subscription_cost_usd": float(current_subscription_cost_usd),
            "billed_through_date": billed_through_date,
            "budget_exhausted": False,
            "budget_stop_date": None,
        }

    total_cost = float(current_subscription_cost_usd)
    cursor = start_date
    last_billed = billed_through_date

    while cursor <= target_end_date:
        segment_end = min(_month_end(cursor), target_end_date)
        daily_fee = float(monthly_fee_usd) / float(_days_in_month(cursor))
        nb_days = (segment_end - cursor).days + 1

        if max_total_budget_usd is not None:
            already_spent = float(total_spend_before_accrual_usd) + float(total_cost)
            remaining_budget = float(max_total_budget_usd) - already_spent
            affordable_days = int(math.floor((remaining_budget + 1e-12) / daily_fee)) if daily_fee > 0 else nb_days
            if affordable_days < nb_days:
                if affordable_days > 0:
                    total_cost += float(affordable_days) * daily_fee
                    last_billed = _shift_days(cursor, affordable_days - 1)
                return {
                    "subscription_cost_usd": float(total_cost),
                    "billed_through_date": last_billed,
                    "budget_exhausted": True,
                    "budget_stop_date": last_billed,
                }

        total_cost += float(nb_days) * daily_fee
        last_billed = segment_end
        cursor = _shift_days(segment_end, 1)

    return {
        "subscription_cost_usd": float(total_cost),
        "billed_through_date": last_billed,
        "budget_exhausted": False,
        "budget_stop_date": None,
    }


def _charge_fixed_cost(
    current_total_spend_usd: float,
    fixed_cost_usd: float,
    max_total_budget_usd: float | None,
) -> tuple[float, bool]:
    proposed = float(current_total_spend_usd) + float(fixed_cost_usd)
    if max_total_budget_usd is not None and proposed > float(max_total_budget_usd) + 1e-12:
        return float(current_total_spend_usd), True
    return proposed, False


def _calendar_template_start_index(rng: np.random.Generator, nb_dates: int) -> int:
    if nb_dates <= 1:
        return 0
    return int(rng.integers(0, nb_dates))


def _build_synthetic_trading_dates(
    template_dates: tuple[dt.date, ...],
    length: int,
    start_index: int,
) -> tuple[dt.date, ...]:
    if length <= 0:
        return tuple()
    if not template_dates:
        return tuple()
    if len(template_dates) == 1:
        start_date = template_dates[0]
        return tuple(_shift_days(start_date, idx) for idx in range(length))

    gaps = [
        max(1, (template_dates[idx + 1] - template_dates[idx]).days)
        for idx in range(len(template_dates) - 1)
    ]
    if not gaps:
        gaps = [1]

    dates = [template_dates[int(start_index) % len(template_dates)]]
    gap_index = int(start_index) % len(gaps)
    while len(dates) < int(length):
        gap_days = gaps[gap_index % len(gaps)]
        dates.append(_shift_days(dates[-1], gap_days))
        gap_index += 1
    return tuple(dates)


def _bootstrap_target_sessions(series: StrategySeries, rules: BusinessV2Rules) -> int:
    trade_rate = max(series.trade_day_fraction, 0.25)
    required_traded = int(rules.challenge_timeout_trading_days + rules.funded_trading_days)
    required_sessions = int(math.ceil(required_traded / trade_rate))
    base_length = max(len(series.session_dates), required_sessions + int(rules.bootstrap_min_buffer_sessions))
    return int(max(base_length, rules.bootstrap_block_size * 10))


def sample_block_bootstrap_path(
    series: StrategySeries,
    rules: BusinessV2Rules,
    rng: np.random.Generator,
    calendar_anchor_index: int | None = None,
) -> DailyPath:
    """Sample a deterministic block-bootstrap path and assign a coherent synthetic calendar."""
    n_obs = len(series.session_dates)
    if n_obs == 0:
        return DailyPath(session_dates=tuple(), daily_pnl=np.array([], dtype=float), daily_trade_count=np.array([], dtype=float))

    target_sessions = _bootstrap_target_sessions(series, rules)
    effective_block = max(1, min(int(rules.bootstrap_block_size), n_obs))
    max_start = max(n_obs - effective_block, 0)

    sampled_indices: list[int] = []
    while len(sampled_indices) < target_sessions:
        block_start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        for index in range(block_start, min(block_start + effective_block, n_obs)):
            sampled_indices.append(index)
            if len(sampled_indices) >= target_sessions:
                break

    anchor_index = (
        int(calendar_anchor_index)
        if calendar_anchor_index is not None
        else _calendar_template_start_index(rng, len(series.session_dates))
    )
    sampled_dates = _build_synthetic_trading_dates(
        template_dates=series.session_dates,
        length=len(sampled_indices),
        start_index=anchor_index,
    )
    sampled_indices_np = np.asarray(sampled_indices, dtype=int)
    return DailyPath(
        session_dates=sampled_dates,
        daily_pnl=series.daily_pnl[sampled_indices_np],
        daily_trade_count=series.daily_trade_count[sampled_indices_np],
    )


def _challenge_timeout_row(start_date: dt.date | None, attempt_index: int, start_index: int) -> dict[str, Any]:
    return {
        "attempt_index": int(attempt_index),
        "attempt_status": "TIMEOUT",
        "attempt_start_date": start_date,
        "attempt_end_date": start_date,
        "attempt_start_index": int(start_index),
        "attempt_next_index": int(start_index),
        "attempt_trading_days": 0,
        "attempt_calendar_days": 0,
        "dll_hit_count": 0,
        "raw_target_hit": False,
        "days_to_raw_target": float("nan"),
        "days_raw_target_to_true_pass": float("nan"),
        "consistency_blocked": False,
        "delayed_pass_after_consistency": False,
        "best_day_profit_usd": 0.0,
        "required_profit_to_pass_usd": float("nan"),
        "consistency_penalty_usd": float("nan"),
        "final_cumulative_profit_usd": 0.0,
        "best_day_share_at_end": float("nan"),
        "mll_threshold_before_final_day_usd": float("nan"),
        "daily_trace_dates": tuple(),
    }


def simulate_challenge_attempt(
    path: DailyPath,
    rules: BusinessV2Rules,
    attempt_index: int,
    start_index: int = 0,
    include_daily_trace: bool = False,
) -> dict[str, Any]:
    """Simulate a single combine attempt until pass, MLL failure, or timeout."""
    if start_index >= len(path.session_dates):
        return _challenge_timeout_row(start_date=None, attempt_index=attempt_index, start_index=start_index)

    start_date = path.session_dates[int(start_index)]
    start_ordinal = int(start_date.toordinal())
    equity = float(rules.starting_balance_usd)
    high_watermark = float(rules.starting_balance_usd)
    best_day_profit_usd = 0.0
    trading_days = 0
    dll_hit_count = 0
    raw_target_hit = False
    raw_target_day = float("nan")
    consistency_blocked = False
    delayed_pass = False
    final_status = "TIMEOUT"
    final_index = int(start_index)
    required_profit = float(rules.challenge_profit_target_usd)
    best_day_share_at_end = float("nan")
    final_cumulative_profit_usd = 0.0
    mll_threshold_before_day = _trailing_floor(high_watermark, rules)
    daily_trace_dates: list[dt.date] = []

    for index in range(int(start_index), len(path.session_dates)):
        session_date = path.session_dates[index]
        raw_daily_pnl = float(path.daily_pnl[index])
        traded = bool(float(path.daily_trade_count[index]) > 0.0)
        if traded:
            trading_days += 1

        capped_daily_pnl = raw_daily_pnl
        if traded and raw_daily_pnl <= -float(rules.daily_loss_limit_usd):
            capped_daily_pnl = -float(rules.daily_loss_limit_usd)
            dll_hit_count += 1

        mll_threshold_before_day = _trailing_floor(high_watermark, rules)
        equity += capped_daily_pnl
        final_cumulative_profit_usd = float(equity - float(rules.starting_balance_usd))
        best_day_profit_usd = max(best_day_profit_usd, max(capped_daily_pnl, 0.0))
        required_profit = required_profit_to_pass(
            best_day_profit_usd=best_day_profit_usd,
            profit_target_usd=rules.challenge_profit_target_usd,
            epsilon_usd=rules.consistency_epsilon_usd,
        )
        if final_cumulative_profit_usd > 0 and best_day_profit_usd > 0:
            best_day_share_at_end = float(best_day_profit_usd / final_cumulative_profit_usd)

        if not raw_target_hit and final_cumulative_profit_usd >= float(rules.challenge_profit_target_usd):
            raw_target_hit = True
            raw_target_day = float(trading_days)

        if raw_target_hit and final_cumulative_profit_usd < required_profit:
            consistency_blocked = True

        if include_daily_trace:
            daily_trace_dates.append(session_date)

        calendar_days = int(session_date.toordinal() - start_ordinal + 1)
        final_index = int(index + 1)
        mll_failed = bool(equity <= mll_threshold_before_day + 1e-12)
        passed = bool(final_cumulative_profit_usd >= required_profit - 1e-12)

        if mll_failed:
            final_status = "FAIL_MLL"
            break

        high_watermark = max(high_watermark, equity)

        if passed:
            final_status = "PASS"
            delayed_pass = bool(raw_target_hit and raw_target_day < float(trading_days))
            break

        if trading_days >= int(rules.challenge_timeout_trading_days):
            final_status = "TIMEOUT"
            break
        if calendar_days >= int(rules.challenge_timeout_calendar_days):
            final_status = "TIMEOUT"
            break
    else:
        session_date = path.session_dates[-1]
        calendar_days = int(session_date.toordinal() - start_ordinal + 1)

    final_end_date = path.session_dates[max(int(final_index) - 1, int(start_index))]
    days_raw_target_to_true_pass = float("nan")
    if raw_target_hit and final_status == "PASS":
        days_raw_target_to_true_pass = float(float(trading_days) - float(raw_target_day))

    return {
        "attempt_index": int(attempt_index),
        "attempt_status": str(final_status),
        "attempt_start_date": start_date,
        "attempt_end_date": final_end_date,
        "attempt_start_index": int(start_index),
        "attempt_next_index": int(final_index),
        "attempt_trading_days": int(trading_days),
        "attempt_calendar_days": int(final_end_date.toordinal() - start_ordinal + 1),
        "dll_hit_count": int(dll_hit_count),
        "raw_target_hit": bool(raw_target_hit),
        "days_to_raw_target": float(raw_target_day),
        "days_raw_target_to_true_pass": float(days_raw_target_to_true_pass),
        "consistency_blocked": bool(consistency_blocked),
        "delayed_pass_after_consistency": bool(delayed_pass),
        "best_day_profit_usd": float(best_day_profit_usd),
        "required_profit_to_pass_usd": float(required_profit),
        "consistency_penalty_usd": float(max(required_profit - float(rules.challenge_profit_target_usd), 0.0)),
        "final_cumulative_profit_usd": float(final_cumulative_profit_usd),
        "best_day_share_at_end": float(best_day_share_at_end),
        "mll_threshold_before_final_day_usd": float(mll_threshold_before_day),
        "daily_trace_dates": tuple(daily_trace_dates) if include_daily_trace else tuple(),
    }


def simulate_funded_phase(
    path: DailyPath,
    rules: BusinessV2Rules,
    start_index: int = 0,
) -> dict[str, Any]:
    """Simulate the funded phase with DLL day capping, MLL breach, and payout units."""
    if start_index >= len(path.session_dates):
        return {
            "funded_started": False,
            "funded_status": "TIMEOUT_HISTORY",
            "funded_start_date": None,
            "funded_end_date": None,
            "funded_trading_days": 0,
            "funded_calendar_days": 0,
            "funded_survival_days": 0,
            "funded_breach": False,
            "funded_breach_reason": "",
            "funded_dll_hit_count": 0,
            "funded_profit_usd": 0.0,
            "payout_count": 0,
            "realized_payout_value_usd": 0.0,
            "first_payout_achieved": False,
            "days_to_first_payout": float("nan"),
            "survived_60d": False,
        }

    start_date = path.session_dates[int(start_index)]
    start_ordinal = int(start_date.toordinal())
    equity = float(rules.starting_balance_usd)
    high_watermark = float(rules.starting_balance_usd)
    trading_days = 0
    dll_hit_count = 0
    payout_count = 0
    next_payout_threshold = float(rules.payout_threshold_usd)
    first_payout_day = float("nan")
    final_index = int(start_index)
    funded_status = "TIMEOUT_HISTORY"
    breach_reason = ""

    for index in range(int(start_index), len(path.session_dates)):
        traded = bool(float(path.daily_trade_count[index]) > 0.0)
        if traded:
            trading_days += 1

        raw_daily_pnl = float(path.daily_pnl[index])
        capped_daily_pnl = raw_daily_pnl
        if traded and raw_daily_pnl <= -float(rules.daily_loss_limit_usd):
            capped_daily_pnl = -float(rules.daily_loss_limit_usd)
            dll_hit_count += 1

        mll_threshold_before_day = _trailing_floor(high_watermark, rules)
        equity += capped_daily_pnl
        funded_profit_usd = float(equity - float(rules.starting_balance_usd))

        while funded_profit_usd >= next_payout_threshold - 1e-12:
            payout_count += 1
            if payout_count == 1:
                first_payout_day = float(trading_days)
            next_payout_threshold += float(rules.payout_threshold_usd)

        final_index = int(index + 1)
        mll_failed = bool(equity <= mll_threshold_before_day + 1e-12)
        if mll_failed:
            funded_status = "BREACH_MLL"
            breach_reason = "MLL"
            break

        high_watermark = max(high_watermark, equity)

        if trading_days >= int(rules.funded_trading_days):
            funded_status = "SURVIVED_60D"
            break

    end_date = path.session_dates[max(int(final_index) - 1, int(start_index))]
    funded_profit_usd = float(equity - float(rules.starting_balance_usd))
    return {
        "funded_started": True,
        "funded_status": str(funded_status),
        "funded_start_date": start_date,
        "funded_end_date": end_date,
        "funded_trading_days": int(trading_days),
        "funded_calendar_days": int(end_date.toordinal() - start_ordinal + 1),
        "funded_survival_days": int(trading_days),
        "funded_breach": bool(funded_status == "BREACH_MLL"),
        "funded_breach_reason": breach_reason,
        "funded_dll_hit_count": int(dll_hit_count),
        "funded_profit_usd": float(funded_profit_usd),
        "payout_count": int(payout_count),
        "realized_payout_value_usd": float(payout_count * float(rules.payout_threshold_usd)),
        "first_payout_achieved": bool(payout_count > 0),
        "days_to_first_payout": float(first_payout_day),
        "survived_60d": bool(funded_status == "SURVIVED_60D"),
    }


def simulate_business_cycle(
    plan: TopstepBusinessPlan,
    challenge_path: DailyPath,
    challenge_strategy: StrategySeries,
    funded_path: DailyPath,
    funded_strategy: StrategySeries,
    rules: BusinessV2Rules,
    simulation_method: str,
    simulation_id: int,
    config_id: str,
    start_session_date: dt.date | None,
) -> BusinessCycleArtifacts:
    """Simulate one full combine-to-funded business cycle."""
    if not challenge_path.session_dates:
        cycle_row = {
            "simulation_method": simulation_method,
            "simulation_id": int(simulation_id),
            "config_id": config_id,
            "start_session_date": start_session_date,
            "plan": plan.name,
            "n_resets_per_month": int(rules.n_resets_per_month),
            "challenge_strategy": challenge_strategy.strategy_name,
            "challenge_source_variant_name": challenge_strategy.source_variant_name,
            "challenge_leverage_factor": float(challenge_strategy.leverage_factor),
            "funded_strategy": funded_strategy.strategy_name,
            "funded_source_variant_name": funded_strategy.source_variant_name,
            "funded_leverage_factor": float(funded_strategy.leverage_factor),
            "challenge_status": "TIMEOUT",
            "challenge_passed": False,
            "first_attempt_pass": False,
            "pass_within_month": False,
            "pass_within_2_attempts_same_month": False,
            "challenge_attempt_count": 0,
            "challenge_fail_mll_count": 0,
            "challenge_timeout": True,
            "budget_exhausted": False,
            "challenge_calendar_days": 0,
            "challenge_active_months": 0,
            "calendar_days_locked_out": 0,
            "resets_used": 0,
            "resets_per_active_month": 0.0,
            "probability_two_failures_same_month": False,
            "raw_target_hit": False,
            "days_to_raw_target": float("nan"),
            "days_raw_target_to_true_pass": float("nan"),
            "consistency_blocked": False,
            "avg_dll_hits_per_attempt": 0.0,
            "avg_max_day_share": float("nan"),
            "avg_consistency_penalty_usd": float("nan"),
            "subscription_cost_usd": 0.0,
            "reset_cost_usd": 0.0,
            "activation_fee_usd": 0.0,
            "total_cost_usd": 0.0,
            "funded_started": False,
            "funded_status": "TIMEOUT_HISTORY",
            "funded_survived_60d": False,
            "funded_breach": False,
            "funded_profit_usd": 0.0,
            "payout_count": 0,
            "realized_payout_value_usd": 0.0,
            "first_payout_achieved": False,
            "days_to_first_payout": float("nan"),
            "cycle_calendar_days": 0,
            "net_profit_usd": 0.0,
            "roi_on_cash_spent": 0.0,
        }
        return BusinessCycleArtifacts(cycle_row=cycle_row, challenge_attempt_rows=[], funded_rows=[])

    cycle_start_date = challenge_path.session_dates[0]
    cycle_start_month_key = _month_key(cycle_start_date)

    billed_through_date: dt.date | None = _shift_days(cycle_start_date, -1)
    subscription_cost_usd = 0.0
    reset_cost_usd = 0.0
    activation_fee_usd = 0.0
    challenge_attempt_count = 0
    challenge_fail_mll_count = 0
    resets_used = 0
    calendar_days_locked_out = 0
    challenge_attempt_rows: list[dict[str, Any]] = []
    funded_rows: list[dict[str, Any]] = []
    resets_used_by_month: dict[str, int] = defaultdict(int)
    failure_count_by_month: dict[str, int] = defaultdict(int)
    attempts_started_by_month: dict[str, int] = defaultdict(int)
    challenge_passed = False
    challenge_status = "TIMEOUT"
    budget_exhausted = False
    challenge_end_date = cycle_start_date
    current_index = 0
    successful_attempt: dict[str, Any] | None = None

    while current_index < len(challenge_path.session_dates):
        attempt_start_date = challenge_path.session_dates[current_index]
        challenge_attempt_count += 1
        attempts_started_by_month[_month_key(attempt_start_date)] += 1
        include_trace = rules.max_total_budget_usd is not None
        attempt = simulate_challenge_attempt(
            path=challenge_path,
            rules=rules,
            attempt_index=challenge_attempt_count,
            start_index=current_index,
            include_daily_trace=include_trace,
        )

        if include_trace and attempt["daily_trace_dates"]:
            for trace_date in attempt["daily_trace_dates"]:
                accrual = _accrue_subscription_cost(
                    billed_through_date=billed_through_date,
                    target_end_date=trace_date,
                    monthly_fee_usd=plan.monthly_subscription_usd,
                    total_spend_before_accrual_usd=reset_cost_usd + activation_fee_usd,
                    current_subscription_cost_usd=subscription_cost_usd,
                    max_total_budget_usd=rules.max_total_budget_usd,
                )
                subscription_cost_usd = float(accrual["subscription_cost_usd"])
                billed_through_date = accrual["billed_through_date"]
                if accrual["budget_exhausted"]:
                    budget_exhausted = True
                    challenge_status = "BUDGET_EXHAUSTED"
                    challenge_end_date = billed_through_date or trace_date
                    break
            if budget_exhausted:
                break
        else:
            accrual = _accrue_subscription_cost(
                billed_through_date=billed_through_date,
                target_end_date=attempt["attempt_end_date"],
                monthly_fee_usd=plan.monthly_subscription_usd,
                total_spend_before_accrual_usd=reset_cost_usd + activation_fee_usd,
                current_subscription_cost_usd=subscription_cost_usd,
                max_total_budget_usd=rules.max_total_budget_usd,
            )
            subscription_cost_usd = float(accrual["subscription_cost_usd"])
            billed_through_date = accrual["billed_through_date"]
            if accrual["budget_exhausted"]:
                budget_exhausted = True
                challenge_status = "BUDGET_EXHAUSTED"
                challenge_end_date = billed_through_date or attempt["attempt_end_date"]
                break

        challenge_end_date = attempt["attempt_end_date"]
        current_index = int(attempt["attempt_next_index"])
        challenge_attempt_rows.append(
            {
                "simulation_method": simulation_method,
                "simulation_id": int(simulation_id),
                "config_id": config_id,
                "plan": plan.name,
                "n_resets_per_month": int(rules.n_resets_per_month),
                "challenge_strategy": challenge_strategy.strategy_name,
                "challenge_source_variant_name": challenge_strategy.source_variant_name,
                "challenge_leverage_factor": float(challenge_strategy.leverage_factor),
                **{key: value for key, value in attempt.items() if key != "daily_trace_dates"},
            }
        )

        if attempt["attempt_status"] == "PASS":
            challenge_passed = True
            challenge_status = "PASS"
            successful_attempt = attempt
            break

        if attempt["attempt_status"] == "TIMEOUT":
            challenge_status = "TIMEOUT"
            break

        challenge_fail_mll_count += 1
        failure_month_key = _month_key(attempt["attempt_end_date"])
        failure_count_by_month[failure_month_key] += 1
        if resets_used_by_month[failure_month_key] < int(rules.n_resets_per_month):
            proposed_reset_total, reset_budget_exhausted = _charge_fixed_cost(
                current_total_spend_usd=subscription_cost_usd + reset_cost_usd + activation_fee_usd,
                fixed_cost_usd=plan.reset_cost_usd,
                max_total_budget_usd=rules.max_total_budget_usd,
            )
            if reset_budget_exhausted:
                budget_exhausted = True
                challenge_status = "BUDGET_EXHAUSTED"
                challenge_end_date = attempt["attempt_end_date"]
                break
            reset_cost_usd = float(proposed_reset_total - subscription_cost_usd - activation_fee_usd)
            resets_used_by_month[failure_month_key] += 1
            resets_used += 1
            continue

        lock_start_date = _shift_days(attempt["attempt_end_date"], 1)
        month_end_date = _month_end(attempt["attempt_end_date"])
        if lock_start_date <= month_end_date:
            locked_days_here = int((month_end_date - lock_start_date).days + 1)
            calendar_days_locked_out += max(locked_days_here, 0)
            accrual = _accrue_subscription_cost(
                billed_through_date=billed_through_date,
                target_end_date=month_end_date,
                monthly_fee_usd=plan.monthly_subscription_usd,
                total_spend_before_accrual_usd=reset_cost_usd + activation_fee_usd,
                current_subscription_cost_usd=subscription_cost_usd,
                max_total_budget_usd=rules.max_total_budget_usd,
            )
            subscription_cost_usd = float(accrual["subscription_cost_usd"])
            billed_through_date = accrual["billed_through_date"]
            challenge_end_date = billed_through_date or month_end_date
            if accrual["budget_exhausted"]:
                budget_exhausted = True
                challenge_status = "BUDGET_EXHAUSTED"
                break
        current_index = _find_next_index_after(challenge_path, month_end_date, current_index)
        if current_index >= len(challenge_path.session_dates):
            challenge_status = "TIMEOUT"
            challenge_end_date = month_end_date
            break

    if budget_exhausted:
        challenge_status = "BUDGET_EXHAUSTED"
    elif not challenge_passed and challenge_status == "TIMEOUT" and billed_through_date is not None:
        challenge_end_date = billed_through_date

    challenge_active_months = _count_months_between(cycle_start_date, challenge_end_date)
    resets_per_active_month = _safe_div(resets_used, challenge_active_months, default=0.0)
    any_two_failures_same_month = any(count >= 2 for count in failure_count_by_month.values())
    raw_target_hit = any(bool(row["raw_target_hit"]) for row in challenge_attempt_rows)
    consistency_blocked = any(bool(row["consistency_blocked"]) for row in challenge_attempt_rows)
    best_day_share_values = [
        float(row["best_day_share_at_end"])
        for row in challenge_attempt_rows
        if pd.notna(row["best_day_share_at_end"])
    ]
    max_day_share_cycle = max(best_day_share_values) if best_day_share_values else float("nan")
    avg_dll_hits_per_attempt = _safe_div(
        sum(int(row["dll_hit_count"]) for row in challenge_attempt_rows),
        challenge_attempt_count,
        default=0.0,
    )
    consistency_penalties = [
        float(row["consistency_penalty_usd"])
        for row in challenge_attempt_rows
        if pd.notna(row["consistency_penalty_usd"]) and float(row["consistency_penalty_usd"]) > 0.0
    ]
    avg_consistency_penalty_usd = (
        float(np.mean(consistency_penalties)) if consistency_penalties else float("nan")
    )

    first_attempt_pass = bool(challenge_passed and challenge_attempt_count == 1)
    pass_within_month = bool(challenge_passed and _month_key(challenge_end_date) == cycle_start_month_key)
    pass_within_2_attempts_same_month = bool(
        challenge_passed
        and _month_key(challenge_end_date) == cycle_start_month_key
        and attempts_started_by_month[cycle_start_month_key] <= 2
    )

    days_to_raw_target = float("nan")
    days_raw_target_to_true_pass = float("nan")
    if successful_attempt is not None:
        days_to_raw_target = float(successful_attempt["days_to_raw_target"])
        days_raw_target_to_true_pass = float(successful_attempt["days_raw_target_to_true_pass"])

    total_cost_before_activation = float(subscription_cost_usd + reset_cost_usd)
    funded_result = {
        "funded_started": False,
        "funded_status": "NOT_STARTED",
        "funded_start_date": None,
        "funded_end_date": None,
        "funded_trading_days": 0,
        "funded_calendar_days": 0,
        "funded_survival_days": 0,
        "funded_breach": False,
        "funded_breach_reason": "",
        "funded_dll_hit_count": 0,
        "funded_profit_usd": 0.0,
        "payout_count": 0,
        "realized_payout_value_usd": 0.0,
        "first_payout_achieved": False,
        "days_to_first_payout": float("nan"),
        "survived_60d": False,
    }

    if challenge_passed:
        _, activation_budget_exhausted = _charge_fixed_cost(
            current_total_spend_usd=total_cost_before_activation,
            fixed_cost_usd=plan.activation_fee_usd,
            max_total_budget_usd=rules.max_total_budget_usd,
        )
        if activation_budget_exhausted:
            budget_exhausted = True
            funded_result["funded_status"] = "BUDGET_EXHAUSTED_BEFORE_FUNDED"
        else:
            activation_fee_usd = float(plan.activation_fee_usd)
            funded_start_index = _find_next_index_after(
                funded_path,
                challenge_end_date,
                0,
            )
            funded_result = simulate_funded_phase(
                path=funded_path,
                rules=rules,
                start_index=funded_start_index,
            )
            if funded_result["funded_started"]:
                funded_rows.append(
                    {
                        "simulation_method": simulation_method,
                        "simulation_id": int(simulation_id),
                        "config_id": config_id,
                        "plan": plan.name,
                        "n_resets_per_month": int(rules.n_resets_per_month),
                        "challenge_strategy": challenge_strategy.strategy_name,
                        "funded_strategy": funded_strategy.strategy_name,
                        **funded_result,
                    }
                )

    cycle_end_date = funded_result["funded_end_date"] or challenge_end_date
    cycle_calendar_days = int(cycle_end_date.toordinal() - cycle_start_date.toordinal() + 1)
    total_cost_usd = float(subscription_cost_usd + reset_cost_usd + activation_fee_usd)
    realized_payout_value_usd = float(funded_result["realized_payout_value_usd"])
    net_profit_usd = float(realized_payout_value_usd - total_cost_usd)
    roi_on_cash_spent = _safe_div(net_profit_usd, total_cost_usd, default=0.0)

    cycle_row = {
        "simulation_method": simulation_method,
        "simulation_id": int(simulation_id),
        "config_id": config_id,
        "start_session_date": start_session_date,
        "plan": plan.name,
        "n_resets_per_month": int(rules.n_resets_per_month),
        "challenge_strategy": challenge_strategy.strategy_name,
        "challenge_source_variant_name": challenge_strategy.source_variant_name,
        "challenge_leverage_factor": float(challenge_strategy.leverage_factor),
        "funded_strategy": funded_strategy.strategy_name,
        "funded_source_variant_name": funded_strategy.source_variant_name,
        "funded_leverage_factor": float(funded_strategy.leverage_factor),
        "challenge_status": challenge_status,
        "challenge_passed": bool(challenge_passed),
        "first_attempt_pass": bool(first_attempt_pass),
        "pass_within_month": bool(pass_within_month),
        "pass_within_2_attempts_same_month": bool(pass_within_2_attempts_same_month),
        "challenge_attempt_count": int(challenge_attempt_count),
        "challenge_fail_mll_count": int(challenge_fail_mll_count),
        "challenge_timeout": bool(challenge_status == "TIMEOUT"),
        "budget_exhausted": bool(budget_exhausted),
        "challenge_calendar_days": int(challenge_end_date.toordinal() - cycle_start_date.toordinal() + 1),
        "challenge_active_months": int(challenge_active_months),
        "calendar_days_locked_out": int(calendar_days_locked_out),
        "resets_used": int(resets_used),
        "resets_per_active_month": float(resets_per_active_month),
        "probability_two_failures_same_month": bool(any_two_failures_same_month),
        "raw_target_hit": bool(raw_target_hit),
        "days_to_raw_target": float(days_to_raw_target),
        "days_raw_target_to_true_pass": float(days_raw_target_to_true_pass),
        "consistency_blocked": bool(consistency_blocked),
        "avg_dll_hits_per_attempt": float(avg_dll_hits_per_attempt),
        "avg_max_day_share": float(max_day_share_cycle),
        "avg_consistency_penalty_usd": float(avg_consistency_penalty_usd),
        "subscription_cost_usd": float(subscription_cost_usd),
        "reset_cost_usd": float(reset_cost_usd),
        "activation_fee_usd": float(activation_fee_usd),
        "total_cost_usd": float(total_cost_usd),
        "funded_started": bool(funded_result["funded_started"]),
        "funded_status": funded_result["funded_status"],
        "funded_survived_60d": bool(funded_result["survived_60d"]),
        "funded_breach": bool(funded_result["funded_breach"]),
        "funded_profit_usd": float(funded_result["funded_profit_usd"]),
        "payout_count": int(funded_result["payout_count"]),
        "realized_payout_value_usd": float(realized_payout_value_usd),
        "first_payout_achieved": bool(funded_result["first_payout_achieved"]),
        "days_to_first_payout": float(funded_result["days_to_first_payout"]),
        "cycle_calendar_days": int(cycle_calendar_days),
        "net_profit_usd": float(net_profit_usd),
        "roi_on_cash_spent": float(roi_on_cash_spent),
    }
    return BusinessCycleArtifacts(
        cycle_row=cycle_row,
        challenge_attempt_rows=challenge_attempt_rows,
        funded_rows=funded_rows,
    )
