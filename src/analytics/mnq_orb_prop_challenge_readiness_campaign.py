"""Business-oriented prop challenge readiness campaign for validated MNQ ORB variants.

This campaign is intentionally narrow:
- it reuses the audited export produced by the VVIX + 3-state validation stack,
- it compares already-sanctioned variants on the exact common session universe,
- it does not touch the ORB signal or re-optimize overlays,
- it focuses on challenge-style readiness and a light funded follow-up lens.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.mnq_orb_prop_challenge_simulation import (
    DEFAULT_PRIMARY_SCOPE,
    _json_dump,
    _normalize_daily_results,
    _read_run_metadata,
    _safe_div,
    _scope_daily_results,
    _source_is_fraction,
)
from src.analytics.mnq_orb_prop_survivability_campaign import _rebuild_daily_results_from_trades
from src.config.paths import EXPORTS_DIR, ensure_directories
from src.config.settings import get_instrument_spec


DEFAULT_SOURCE_RUN_GLOB = "mnq_orb_vvix_sizing_modulation_*"
DEFAULT_CORE_VARIANTS = ("baseline_3state", "baseline_3state_vvix_modulator")
DEFAULT_BOUND_VARIANTS = ("baseline_nominal", "baseline_vvix_modulator")
DEFAULT_VARIANT_ORDER = DEFAULT_CORE_VARIANTS + DEFAULT_BOUND_VARIANTS
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class PropChallengeRules:
    """Centralized challenge rules."""

    name: str = "prop_50k_main"
    description: str = "50K-style prop challenge with profit target, daily loss, and trailing drawdown."
    account_size_usd: float = 50_000.0
    profit_target_usd: float = 3_000.0
    max_trading_days: int | None = 35
    daily_loss_limit_usd: float | None = 1_000.0
    static_max_loss_usd: float | None = None
    trailing_drawdown_usd: float | None = 2_000.0
    trailing_drawdown_locks_at_start_balance: bool = False
    cut_day_on_daily_loss: bool = True
    near_daily_limit_buffer_frac: float = 0.10
    near_global_limit_buffer_frac: float = 0.10
    notes: str = (
        "Daily loss is modeled with end-of-day audited PnL and capped at the limit when the limit is hit. "
        "Trailing drawdown uses prior closed-equity high-watermark logic."
    )


@dataclass(frozen=True)
class RiskProfile:
    name: str
    multiplier: float
    description: str


@dataclass(frozen=True)
class StressProfile:
    name: str
    slippage_multiplier: float
    description: str


@dataclass(frozen=True)
class FundedFollowupSpec:
    enabled: bool = True
    trading_days: int = 60
    big_negative_day_frac_of_daily_limit: float = 0.80


@dataclass
class VariantInput:
    variant_name: str
    label: str
    source_root: Path
    trades: pd.DataFrame
    daily_results: pd.DataFrame
    controls: pd.DataFrame
    reference_account_size_usd: float
    source_summary_row: dict[str, Any]


@dataclass(frozen=True)
class PropChallengeReadinessSpec:
    source_run_root: Path | None = None
    primary_scope: str = DEFAULT_PRIMARY_SCOPE
    variant_names: tuple[str, ...] = DEFAULT_VARIANT_ORDER
    core_variant_names: tuple[str, ...] = DEFAULT_CORE_VARIANTS
    rules: PropChallengeRules = field(default_factory=PropChallengeRules)
    risk_profiles: tuple[RiskProfile, ...] = field(
        default_factory=lambda: (
            RiskProfile(
                name="conservative",
                multiplier=0.75,
                description="25% less gross risk than the validated base overlay sizing.",
            ),
            RiskProfile(
                name="base",
                multiplier=1.0,
                description="Exact validated overlay sizing as exported.",
            ),
            RiskProfile(
                name="assertive",
                multiplier=1.20,
                description="20% more gross risk, kept compact on purpose.",
            ),
        )
    )
    stress_profiles: tuple[StressProfile, ...] = field(
        default_factory=lambda: (
            StressProfile(
                name="slippage_nominal",
                slippage_multiplier=1.0,
                description="Repo nominal execution assumptions.",
            ),
            StressProfile(
                name="slippage_x2",
                slippage_multiplier=2.0,
                description="Execution stress with 2x nominal slippage.",
            ),
            StressProfile(
                name="slippage_x3",
                slippage_multiplier=3.0,
                description="Execution stress with 3x nominal slippage.",
            ),
        )
    )
    funded_followup: FundedFollowupSpec = field(default_factory=FundedFollowupSpec)
    random_seed: int = DEFAULT_RANDOM_SEED
    output_root: Path | None = None


def _find_latest_source_run(root: Path = EXPORTS_DIR) -> Path:
    required = set(DEFAULT_CORE_VARIANTS)
    candidates: list[Path] = []
    for path in sorted(root.glob(DEFAULT_SOURCE_RUN_GLOB)):
        if not path.is_dir():
            continue
        if not (path / "run_metadata.json").exists():
            continue
        variants_root = path / "variants"
        if not variants_root.exists():
            continue
        available = {item.name for item in variants_root.iterdir() if item.is_dir()}
        if required.issubset(available):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No compatible VVIX sizing modulation export was found.")
    return candidates[-1]


def _summary_row_map(source_root: Path) -> dict[str, dict[str, Any]]:
    for filename in ("validation_summary.csv", "full_variant_results.csv"):
        path = source_root / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "variant_name" not in df.columns:
            continue
        return {
            str(row["variant_name"]): {str(key): value for key, value in row.items()}
            for _, row in df.iterrows()
        }
    return {}


def _normalize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    out = trades.copy()
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.date
    numeric_columns = [
        "trade_id",
        "quantity",
        "risk_budget_usd",
        "risk_per_contract_usd",
        "actual_risk_usd",
        "trade_risk_usd",
        "notional_usd",
        "leverage_used",
        "pnl_points",
        "pnl_ticks",
        "pnl_usd",
        "fees",
        "net_pnl_usd",
        "risk_multiplier",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.reset_index(drop=True)


def _load_variant_input(
    source_root: Path,
    variant_name: str,
    summary_rows: dict[str, dict[str, Any]],
    default_account_size_usd: float,
) -> VariantInput:
    variant_root = source_root / "variants" / variant_name
    if not variant_root.exists():
        raise FileNotFoundError(f"Variant directory not found: {variant_root}")

    trades_path = variant_root / "trades.csv"
    daily_path = variant_root / "daily_results.csv"
    controls_path = variant_root / "controls.csv"

    trades = _normalize_trades(pd.read_csv(trades_path)) if trades_path.exists() else pd.DataFrame()
    daily_results = _normalize_daily_results(pd.read_csv(daily_path)) if daily_path.exists() else pd.DataFrame()
    controls = pd.read_csv(controls_path) if controls_path.exists() else pd.DataFrame()

    if not trades.empty and "account_size_usd" in trades.columns:
        account_series = pd.to_numeric(trades["account_size_usd"], errors="coerce").dropna()
        reference_account_size = float(account_series.iloc[0]) if not account_series.empty else float(default_account_size_usd)
    else:
        reference_account_size = float(default_account_size_usd)

    return VariantInput(
        variant_name=variant_name,
        label=variant_name,
        source_root=variant_root,
        trades=trades,
        daily_results=daily_results,
        controls=controls,
        reference_account_size_usd=reference_account_size,
        source_summary_row=summary_rows.get(variant_name, {}),
    )


def _subset_sessions(frame: pd.DataFrame, sessions: list) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    session_set = set(pd.to_datetime(pd.Index(sessions)).date)
    out = frame.copy()
    out_dates = pd.to_datetime(out["session_date"]).dt.date
    return out.loc[out_dates.isin(session_set)].copy().reset_index(drop=True)


def _common_scope_sessions(variant_inputs: list[VariantInput], is_fraction: float, scope: str) -> list:
    scoped_sets: list[set] = []
    for variant in variant_inputs:
        scoped_daily = _scope_daily_results(variant.daily_results, is_fraction=is_fraction, scope=scope)
        scoped_sets.append(set(pd.to_datetime(scoped_daily["session_date"]).dt.date.tolist()))
    if not scoped_sets:
        return []
    return sorted(set.intersection(*scoped_sets))


def _trailing_floor(start_equity: float, high_watermark: float, rules: PropChallengeRules) -> float | None:
    if rules.trailing_drawdown_usd is None:
        return None
    floor = float(high_watermark) - float(rules.trailing_drawdown_usd)
    if rules.trailing_drawdown_locks_at_start_balance:
        floor = min(floor, float(start_equity))
    return float(floor)


def _active_global_floor(
    start_equity: float,
    high_watermark: float,
    rules: PropChallengeRules,
) -> tuple[float, str | None, float | None]:
    candidates: list[tuple[float, str, float]] = []
    if rules.static_max_loss_usd is not None:
        candidates.append(
            (
                float(start_equity) - float(rules.static_max_loss_usd),
                "static_max_loss",
                float(rules.static_max_loss_usd),
            )
        )
    trailing_floor = _trailing_floor(start_equity=start_equity, high_watermark=high_watermark, rules=rules)
    if trailing_floor is not None:
        candidates.append(
            (
                float(trailing_floor),
                "trailing_drawdown",
                float(rules.trailing_drawdown_usd or 0.0),
            )
        )
    if not candidates:
        return float("-inf"), None, None
    floor_value, floor_name, allowance = max(candidates, key=lambda item: item[0])
    return float(floor_value), floor_name, float(allowance)


def _scale_daily_results(daily_results: pd.DataFrame, scale: float) -> pd.DataFrame:
    if abs(float(scale) - 1.0) < 1e-12:
        return _normalize_daily_results(daily_results)
    scaled = _normalize_daily_results(daily_results).copy()
    for column in ("daily_pnl_usd", "daily_gross_pnl_usd", "daily_fees_usd"):
        if column in scaled.columns:
            scaled[column] = pd.to_numeric(scaled[column], errors="coerce").fillna(0.0) * float(scale)
    return scaled


def _eligible_rolling_start_indices(daily_results: pd.DataFrame, max_trading_days: int | None) -> list[int]:
    ordered = _normalize_daily_results(daily_results)
    if max_trading_days is None:
        return list(range(len(ordered)))
    traded_mask = pd.to_numeric(ordered["daily_trade_count"], errors="coerce").fillna(0.0).gt(0.0).astype(int)
    remaining_traded = traded_mask.iloc[::-1].cumsum().iloc[::-1]
    return remaining_traded.loc[remaining_traded >= int(max_trading_days)].index.tolist()


def _common_rolling_start_dates(
    scenario_daily_map: dict[str, pd.DataFrame],
    rules: PropChallengeRules,
) -> list:
    common_dates: set | None = None
    for daily_results in scenario_daily_map.values():
        ordered = _normalize_daily_results(daily_results)
        dates = {
            ordered.iloc[idx]["session_date"]
            for idx in _eligible_rolling_start_indices(ordered, max_trading_days=rules.max_trading_days)
        }
        common_dates = dates if common_dates is None else (common_dates & dates)
    return sorted(common_dates) if common_dates else []


def _risk_scaled_trades(
    trades: pd.DataFrame,
    risk_multiplier: float,
    extra_slippage_ticks_per_side: float,
    tick_value_usd: float,
) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()

    out = _normalize_trades(trades).copy()
    factor = float(risk_multiplier)

    if "quantity" in out.columns:
        out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce").fillna(0.0) * factor
    else:
        out["quantity"] = 0.0

    for column in ("risk_budget_usd", "actual_risk_usd", "trade_risk_usd", "notional_usd", "leverage_used", "pnl_usd", "fees", "net_pnl_usd"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0) * factor

    if "risk_multiplier" in out.columns:
        out["effective_risk_multiplier"] = pd.to_numeric(out["risk_multiplier"], errors="coerce").fillna(0.0) * factor
    else:
        out["effective_risk_multiplier"] = factor
    out["global_risk_multiplier"] = factor

    extra_cost = 2.0 * float(extra_slippage_ticks_per_side) * float(tick_value_usd) * pd.to_numeric(out["quantity"], errors="coerce").fillna(0.0)
    out["stress_extra_slippage_usd"] = extra_cost
    out["pnl_usd"] = pd.to_numeric(out["pnl_usd"], errors="coerce").fillna(0.0) - extra_cost
    out["net_pnl_usd"] = pd.to_numeric(out["pnl_usd"], errors="coerce").fillna(0.0) - pd.to_numeric(out["fees"], errors="coerce").fillna(0.0)
    return out


def _scenario_daily_results(
    variant: VariantInput,
    sessions: list,
    risk_profile: RiskProfile,
    stress_profile: StressProfile,
    tick_value_usd: float,
    base_slippage_ticks: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset_trades = _subset_sessions(variant.trades, sessions)
    extra_slippage_ticks = max(float(stress_profile.slippage_multiplier) - 1.0, 0.0) * float(base_slippage_ticks)
    scenario_trades = _risk_scaled_trades(
        trades=subset_trades,
        risk_multiplier=float(risk_profile.multiplier),
        extra_slippage_ticks_per_side=extra_slippage_ticks,
        tick_value_usd=float(tick_value_usd),
    )
    scenario_daily = _rebuild_daily_results_from_trades(
        trades=scenario_trades,
        all_sessions=sessions,
        initial_capital=float(variant.reference_account_size_usd),
    )
    return scenario_trades, _normalize_daily_results(scenario_daily)


def simulate_challenge_attempt(
    daily_results: pd.DataFrame,
    rules: PropChallengeRules,
    reference_account_size_usd: float = 50_000.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered = _normalize_daily_results(daily_results)
    scale = _safe_div(rules.account_size_usd, reference_account_size_usd, default=1.0)
    ordered = _scale_daily_results(ordered, scale=scale)

    start_equity = float(rules.account_size_usd)
    equity = float(start_equity)
    high_watermark = float(start_equity)
    traded_days = 0
    calendar_days = 0
    max_drawdown_usd = 0.0
    near_limit_day_count = 0
    near_daily_limit_day_count = 0
    near_global_limit_day_count = 0
    daily_limit_hit_day_count = 0
    history_rows: list[dict[str, Any]] = []

    status = "open"
    failure_reason = ""
    days_to_pass = float("nan")
    days_to_breach = float("nan")

    for _, row in ordered.iterrows():
        calendar_days += 1
        session_date = pd.to_datetime(row["session_date"]).date()
        raw_daily_pnl = float(pd.to_numeric(pd.Series([row.get("daily_pnl_usd", 0.0)]), errors="coerce").iloc[0])
        traded = bool(float(pd.to_numeric(pd.Series([row.get("daily_trade_count", 0.0)]), errors="coerce").iloc[0]) > 0.0)
        if traded:
            traded_days += 1

        prior_high_watermark = float(high_watermark)
        global_floor_before_day, floor_name, allowance = _active_global_floor(
            start_equity=start_equity,
            high_watermark=prior_high_watermark,
            rules=rules,
        )

        daily_limit_hit = bool(
            rules.daily_loss_limit_usd is not None
            and raw_daily_pnl <= -float(rules.daily_loss_limit_usd)
        )
        effective_daily_pnl = float(raw_daily_pnl)
        if daily_limit_hit and rules.cut_day_on_daily_loss and rules.daily_loss_limit_usd is not None:
            effective_daily_pnl = -float(rules.daily_loss_limit_usd)

        equity += effective_daily_pnl
        high_watermark = max(high_watermark, equity)
        current_drawdown = float(equity - high_watermark)
        max_drawdown_usd = min(max_drawdown_usd, current_drawdown)
        cumulative_profit = float(equity - start_equity)

        global_limit_breached = bool(math.isfinite(global_floor_before_day) and equity <= global_floor_before_day + 1e-12)
        global_buffer_usd = float("inf") if not math.isfinite(global_floor_before_day) else float(equity - global_floor_before_day)
        near_daily = bool(
            rules.daily_loss_limit_usd is not None
            and min(effective_daily_pnl, 0.0) <= -float(rules.daily_loss_limit_usd) * (1.0 - float(rules.near_daily_limit_buffer_frac))
        )
        near_global = bool(
            allowance is not None
            and global_buffer_usd <= float(allowance) * float(rules.near_global_limit_buffer_frac)
        )
        near_any = bool(near_daily or near_global)
        near_limit_day_count += int(near_any)
        near_daily_limit_day_count += int(near_daily)
        near_global_limit_day_count += int(near_global)
        daily_limit_hit_day_count += int(daily_limit_hit)

        history_rows.append(
            {
                "session_date": session_date,
                "raw_daily_pnl_usd": raw_daily_pnl,
                "effective_daily_pnl_usd": effective_daily_pnl,
                "equity": equity,
                "high_watermark": high_watermark,
                "global_floor_usd": global_floor_before_day if math.isfinite(global_floor_before_day) else np.nan,
                "global_floor_name": floor_name,
                "global_buffer_usd": global_buffer_usd if math.isfinite(global_buffer_usd) else np.nan,
                "drawdown_usd": current_drawdown,
                "daily_limit_hit": daily_limit_hit,
                "global_limit_breached": global_limit_breached,
                "near_daily_limit": near_daily,
                "near_global_limit": near_global,
                "near_any_limit": near_any,
                "cumulative_profit_usd": cumulative_profit,
                "traded_days_elapsed": traded_days,
                "calendar_days_elapsed": calendar_days,
            }
        )

        if daily_limit_hit:
            status = "breach"
            failure_reason = "daily_loss_limit"
            days_to_breach = float(traded_days)
            break

        if global_limit_breached:
            status = "breach"
            failure_reason = str(floor_name or "global_limit")
            days_to_breach = float(traded_days)
            break

        if cumulative_profit >= float(rules.profit_target_usd):
            status = "pass"
            days_to_pass = float(traded_days)
            break

        if rules.max_trading_days is not None and traded_days >= int(rules.max_trading_days):
            status = "expire"
            break

    if status == "open":
        status = "expire" if rules.max_trading_days is not None else "breach"
        if status == "breach":
            failure_reason = "insufficient_history"
            days_to_breach = float(traded_days)

    history = pd.DataFrame(history_rows)
    trading_days_for_share = max(int(traded_days), 1)
    result = {
        "status": status,
        "pass": bool(status == "pass"),
        "breach": bool(status == "breach"),
        "expire": bool(status == "expire"),
        "failure_reason": failure_reason,
        "days_to_pass": days_to_pass,
        "days_to_breach": days_to_breach,
        "days_traded": int(traded_days),
        "calendar_days": int(calendar_days),
        "final_pnl_usd": float(equity - start_equity),
        "final_equity_usd": float(equity),
        "profit_per_trading_day_usd": _safe_div(float(equity - start_equity), float(max(traded_days, 1)), default=0.0),
        "max_drawdown_usd": float(max_drawdown_usd),
        "near_limit_day_count": int(near_limit_day_count),
        "near_limit_day_share": _safe_div(float(near_limit_day_count), float(trading_days_for_share), default=0.0),
        "near_daily_limit_day_count": int(near_daily_limit_day_count),
        "near_global_limit_day_count": int(near_global_limit_day_count),
        "daily_limit_hit_day_count": int(daily_limit_hit_day_count),
        "daily_limit_hit_or_near_day_share": _safe_div(
            float(daily_limit_hit_day_count + near_daily_limit_day_count),
            float(trading_days_for_share),
            default=0.0,
        ),
        "goal_before_breach": bool(status == "pass"),
    }
    return history, result


def simulate_funded_followup(
    daily_results: pd.DataFrame,
    rules: PropChallengeRules,
    spec: FundedFollowupSpec,
    pre_pass_history: pd.DataFrame | None = None,
    reference_account_size_usd: float = 50_000.0,
) -> dict[str, Any]:
    if not spec.enabled:
        return {
            "followup_started": False,
            "followup_status": "disabled",
        }

    ordered = _normalize_daily_results(daily_results)
    scale = _safe_div(rules.account_size_usd, reference_account_size_usd, default=1.0)
    ordered = _scale_daily_results(ordered, scale=scale)
    if ordered.empty:
        return {
            "followup_started": False,
            "followup_status": "no_history_after_pass",
        }

    start_equity = float(rules.account_size_usd)
    equity = float(start_equity)
    high_watermark = float(start_equity)
    trading_days = 0
    calendar_days = 0
    max_drawdown_usd = 0.0
    daily_loss_hits = 0
    near_daily_limit_days = 0
    big_negative_days = 0
    challenge_like_breach = False
    breach_reason = ""
    collected_pnls: list[float] = []

    daily_limit = float(rules.daily_loss_limit_usd) if rules.daily_loss_limit_usd is not None else float("nan")
    big_negative_threshold = (
        -float(rules.daily_loss_limit_usd) * float(spec.big_negative_day_frac_of_daily_limit)
        if rules.daily_loss_limit_usd is not None
        else float("-inf")
    )

    for _, row in ordered.iterrows():
        raw_daily_pnl = float(pd.to_numeric(pd.Series([row.get("daily_pnl_usd", 0.0)]), errors="coerce").iloc[0])
        traded = bool(float(pd.to_numeric(pd.Series([row.get("daily_trade_count", 0.0)]), errors="coerce").iloc[0]) > 0.0)
        calendar_days += 1
        if not traded:
            continue

        trading_days += 1
        prior_high_watermark = float(high_watermark)
        global_floor_before_day, floor_name, _ = _active_global_floor(
            start_equity=start_equity,
            high_watermark=prior_high_watermark,
            rules=rules,
        )
        daily_limit_hit = bool(rules.daily_loss_limit_usd is not None and raw_daily_pnl <= -float(rules.daily_loss_limit_usd))
        effective_daily_pnl = float(raw_daily_pnl)
        if daily_limit_hit and rules.cut_day_on_daily_loss and rules.daily_loss_limit_usd is not None:
            effective_daily_pnl = -float(rules.daily_loss_limit_usd)

        equity += effective_daily_pnl
        high_watermark = max(high_watermark, equity)
        current_drawdown = float(equity - high_watermark)
        max_drawdown_usd = min(max_drawdown_usd, current_drawdown)
        collected_pnls.append(effective_daily_pnl)

        daily_loss_hits += int(daily_limit_hit)
        near_daily_limit_days += int(
            rules.daily_loss_limit_usd is not None
            and effective_daily_pnl <= -float(rules.daily_loss_limit_usd) * (1.0 - float(rules.near_daily_limit_buffer_frac))
        )
        big_negative_days += int(effective_daily_pnl <= float(big_negative_threshold))

        if daily_limit_hit:
            challenge_like_breach = True
            breach_reason = "daily_loss_limit"
            break

        if math.isfinite(global_floor_before_day) and equity <= global_floor_before_day + 1e-12:
            challenge_like_breach = True
            breach_reason = str(floor_name or "global_limit")
            break

        if trading_days >= int(spec.trading_days):
            break

    pnl_array = np.asarray(collected_pnls, dtype=float)
    pre_risk_abs = float("nan")
    if pre_pass_history is not None and not pre_pass_history.empty:
        pre_risk_abs = float(
            pd.to_numeric(pre_pass_history["effective_daily_pnl_usd"], errors="coerce").abs().mean()
        )
    followup_risk_abs = float(np.mean(np.abs(pnl_array))) if pnl_array.size else float("nan")

    return {
        "followup_started": bool(trading_days > 0),
        "followup_status": "breach" if challenge_like_breach else "complete",
        "followup_breach": bool(challenge_like_breach),
        "followup_breach_reason": breach_reason,
        "followup_trading_days": int(trading_days),
        "followup_calendar_days": int(calendar_days),
        "followup_net_pnl_usd": float(np.sum(pnl_array)) if pnl_array.size else 0.0,
        "followup_profit_per_trading_day_usd": float(np.mean(pnl_array)) if pnl_array.size else 0.0,
        "followup_daily_pnl_std_usd": float(np.std(pnl_array, ddof=0)) if pnl_array.size else float("nan"),
        "followup_positive_day_rate": float(np.mean(pnl_array > 0.0)) if pnl_array.size else float("nan"),
        "followup_max_drawdown_usd": float(max_drawdown_usd),
        "followup_daily_loss_hit_rate": _safe_div(float(daily_loss_hits), float(max(trading_days, 1)), default=0.0),
        "followup_near_daily_limit_rate": _safe_div(float(near_daily_limit_days), float(max(trading_days, 1)), default=0.0),
        "followup_big_negative_day_rate": _safe_div(float(big_negative_days), float(max(trading_days, 1)), default=0.0),
        "followup_big_negative_threshold_usd": float(big_negative_threshold) if math.isfinite(big_negative_threshold) else float("nan"),
        "followup_risk_intensity_ratio_post_vs_pre": (
            _safe_div(followup_risk_abs, pre_risk_abs, default=float("nan")) if pd.notna(pre_risk_abs) else float("nan")
        ),
        "followup_large_loss_limit_usd": daily_limit if math.isfinite(daily_limit) else float("nan"),
    }


def _nan_median(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.median()) if not clean.empty else float("nan")


def _nan_mean(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else float("nan")


def aggregate_challenge_attempts(attempts: pd.DataFrame) -> dict[str, Any]:
    if attempts.empty:
        return {
            "attempt_count": 0,
            "pass_rate": 0.0,
            "breach_rate": 0.0,
            "expire_rate": 0.0,
            "goal_before_breach_rate": 0.0,
            "median_days_to_pass": float("nan"),
            "mean_days_to_pass": float("nan"),
            "median_days_to_breach": float("nan"),
            "mean_days_to_breach": float("nan"),
            "expected_profit_per_trading_day": 0.0,
            "expected_net_profit_per_attempt": 0.0,
            "average_drawdown_before_pass": float("nan"),
            "worst_drawdown_path_usd": float("nan"),
            "worst_drawdown_start_session_date": pd.NA,
            "daily_loss_hit_or_near_day_rate": 0.0,
            "near_limit_day_rate": 0.0,
            "p25_final_pnl_usd": float("nan"),
            "median_final_pnl_usd": float("nan"),
            "p75_final_pnl_usd": float("nan"),
        }

    pass_mask = attempts["pass"].astype(bool)
    breach_mask = attempts["breach"].astype(bool)
    expire_mask = attempts["expire"].astype(bool)
    worst_idx = pd.to_numeric(attempts["max_drawdown_usd"], errors="coerce").idxmin()
    worst_row = attempts.loc[worst_idx] if pd.notna(worst_idx) else pd.Series(dtype="object")
    total_trading_days = float(max(pd.to_numeric(attempts["days_traded"], errors="coerce").fillna(0.0).sum(), 1.0))
    return {
        "attempt_count": int(len(attempts)),
        "pass_rate": float(pass_mask.mean()),
        "breach_rate": float(breach_mask.mean()),
        "expire_rate": float(expire_mask.mean()),
        "goal_before_breach_rate": float(attempts["goal_before_breach"].astype(bool).mean()),
        "median_days_to_pass": _nan_median(attempts.loc[pass_mask, "days_to_pass"]),
        "mean_days_to_pass": _nan_mean(attempts.loc[pass_mask, "days_to_pass"]),
        "median_days_to_breach": _nan_median(attempts.loc[breach_mask, "days_to_breach"]),
        "mean_days_to_breach": _nan_mean(attempts.loc[breach_mask, "days_to_breach"]),
        "expected_profit_per_trading_day": _nan_mean(attempts["profit_per_trading_day_usd"]),
        "expected_net_profit_per_attempt": _nan_mean(attempts["final_pnl_usd"]),
        "average_drawdown_before_pass": _nan_mean(attempts.loc[pass_mask, "max_drawdown_usd"]),
        "worst_drawdown_path_usd": float(pd.to_numeric(attempts["max_drawdown_usd"], errors="coerce").min()),
        "worst_drawdown_start_session_date": worst_row.get("start_session_date", pd.NA),
        "daily_loss_hit_or_near_day_rate": _safe_div(
            float(pd.to_numeric(attempts["daily_limit_hit_or_near_day_count"], errors="coerce").fillna(0.0).sum()),
            total_trading_days,
            default=0.0,
        ),
        "near_limit_day_rate": _safe_div(
            float(pd.to_numeric(attempts["near_limit_day_count"], errors="coerce").fillna(0.0).sum()),
            total_trading_days,
            default=0.0,
        ),
        "p25_final_pnl_usd": float(pd.to_numeric(attempts["final_pnl_usd"], errors="coerce").quantile(0.25)),
        "median_final_pnl_usd": _nan_median(attempts["final_pnl_usd"]),
        "p75_final_pnl_usd": float(pd.to_numeric(attempts["final_pnl_usd"], errors="coerce").quantile(0.75)),
    }


def aggregate_funded_followups(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "followup_path_count": 0,
            "followup_started_rate": 0.0,
            "followup_breach_rate": float("nan"),
            "followup_complete_rate": float("nan"),
            "followup_expected_profit_per_trading_day": float("nan"),
            "followup_expected_net_pnl": float("nan"),
            "followup_median_max_drawdown_usd": float("nan"),
            "followup_big_negative_day_rate": float("nan"),
            "followup_daily_loss_hit_rate": float("nan"),
            "followup_median_risk_intensity_ratio_post_vs_pre": float("nan"),
        }

    started_mask = rows["followup_started"].astype(bool)
    scoped = rows.loc[started_mask].copy()
    if scoped.empty:
        return {
            "followup_path_count": int(len(rows)),
            "followup_started_rate": 0.0,
            "followup_breach_rate": float("nan"),
            "followup_complete_rate": float("nan"),
            "followup_expected_profit_per_trading_day": float("nan"),
            "followup_expected_net_pnl": float("nan"),
            "followup_median_max_drawdown_usd": float("nan"),
            "followup_big_negative_day_rate": float("nan"),
            "followup_daily_loss_hit_rate": float("nan"),
            "followup_median_risk_intensity_ratio_post_vs_pre": float("nan"),
        }

    return {
        "followup_path_count": int(len(rows)),
        "followup_started_rate": float(started_mask.mean()),
        "followup_breach_rate": float(scoped["followup_breach"].astype(bool).mean()),
        "followup_complete_rate": float((~scoped["followup_breach"].astype(bool)).mean()),
        "followup_expected_profit_per_trading_day": _nan_mean(scoped["followup_profit_per_trading_day_usd"]),
        "followup_expected_net_pnl": _nan_mean(scoped["followup_net_pnl_usd"]),
        "followup_median_max_drawdown_usd": _nan_median(scoped["followup_max_drawdown_usd"]),
        "followup_big_negative_day_rate": _nan_mean(scoped["followup_big_negative_day_rate"]),
        "followup_daily_loss_hit_rate": _nan_mean(scoped["followup_daily_loss_hit_rate"]),
        "followup_median_risk_intensity_ratio_post_vs_pre": _nan_median(
            scoped["followup_risk_intensity_ratio_post_vs_pre"]
        ),
    }


def _core_primary_business_summary(business_summary: pd.DataFrame, spec: PropChallengeReadinessSpec) -> pd.DataFrame:
    core = business_summary.loc[
        business_summary["variant_name"].isin(spec.core_variant_names)
        & business_summary["risk_profile"].eq("base")
        & business_summary["stress_profile"].eq("slippage_nominal")
    ].copy()
    return core.sort_values(
        [
            "pass_rate",
            "breach_rate",
            "median_days_to_pass",
            "followup_breach_rate",
            "followup_expected_profit_per_trading_day",
            "expected_net_profit_per_attempt",
        ],
        ascending=[False, True, True, True, False, False],
        na_position="last",
    ).reset_index(drop=True)


def _primary_challenge_row(business_summary: pd.DataFrame, spec: PropChallengeReadinessSpec) -> pd.Series:
    core = _core_primary_business_summary(business_summary, spec)
    return core.iloc[0] if not core.empty else pd.Series(dtype="object")


def _primary_funded_row(business_summary: pd.DataFrame, spec: PropChallengeReadinessSpec) -> pd.Series:
    core = business_summary.loc[
        business_summary["variant_name"].isin(spec.core_variant_names)
        & business_summary["risk_profile"].eq("base")
        & business_summary["stress_profile"].eq("slippage_nominal")
    ].copy()
    core = core.sort_values(
        [
            "followup_expected_profit_per_trading_day",
            "followup_complete_rate",
            "followup_breach_rate",
            "pass_rate",
        ],
        ascending=[False, False, True, False],
        na_position="last",
    )
    return core.iloc[0] if not core.empty else pd.Series(dtype="object")


def _choose_recommended_risk_profile(
    challenge_summary: pd.DataFrame,
    business_summary: pd.DataFrame,
    spec: PropChallengeReadinessSpec,
) -> pd.Series:
    challenge_row = _primary_challenge_row(business_summary, spec)
    if challenge_row.empty:
        return pd.Series(dtype="object")
    variant_name = str(challenge_row["variant_name"])
    risk_rows = challenge_summary.loc[
        challenge_summary["variant_name"].eq(variant_name)
        & challenge_summary["stress_profile"].eq("slippage_nominal")
    ].copy()
    if risk_rows.empty:
        return pd.Series(dtype="object")

    x2 = challenge_summary.loc[
        challenge_summary["variant_name"].eq(variant_name)
        & challenge_summary["stress_profile"].eq("slippage_x2")
    ][["risk_profile", "pass_rate", "breach_rate"]].rename(
        columns={"pass_rate": "x2_pass_rate", "breach_rate": "x2_breach_rate"}
    )
    merged = risk_rows.merge(x2, on="risk_profile", how="left")
    merged["stress_penalty"] = (
        (pd.to_numeric(merged["pass_rate"], errors="coerce") - pd.to_numeric(merged["x2_pass_rate"], errors="coerce")).abs().fillna(0.0)
        + (pd.to_numeric(merged["x2_breach_rate"], errors="coerce") - pd.to_numeric(merged["breach_rate"], errors="coerce")).clip(lower=0.0).fillna(0.0)
    )
    merged = merged.sort_values(
        ["pass_rate", "stress_penalty", "breach_rate", "median_days_to_pass", "expected_net_profit_per_attempt"],
        ascending=[False, True, True, True, False],
        na_position="last",
    )
    return merged.iloc[0]


def _build_final_verdict(
    challenge_summary: pd.DataFrame,
    stress_summary: pd.DataFrame,
    business_summary: pd.DataFrame,
    spec: PropChallengeReadinessSpec,
    source_root: Path,
    metadata: dict[str, Any],
    common_start_dates: list,
) -> dict[str, Any]:
    primary_challenge = _primary_challenge_row(business_summary, spec)
    primary_funded = _primary_funded_row(business_summary, spec)
    recommended_risk = _choose_recommended_risk_profile(challenge_summary, business_summary, spec)

    challenge_variant = str(primary_challenge.get("variant_name", "")) if not primary_challenge.empty else ""
    funded_variant = str(primary_funded.get("variant_name", "")) if not primary_funded.empty else ""
    split_configuration = bool(challenge_variant and funded_variant and challenge_variant != funded_variant)

    core_nominal = _core_primary_business_summary(business_summary, spec)
    combined_row = core_nominal.loc[core_nominal["variant_name"].eq("baseline_3state_vvix_modulator")]
    three_state_row = core_nominal.loc[core_nominal["variant_name"].eq("baseline_3state")]
    combined = combined_row.iloc[0] if not combined_row.empty else pd.Series(dtype="object")
    three_state = three_state_row.iloc[0] if not three_state_row.empty else pd.Series(dtype="object")

    vvix_improves_challenge = bool(
        not combined.empty
        and not three_state.empty
        and (
            float(combined.get("pass_rate", 0.0)) > float(three_state.get("pass_rate", 0.0)) + 0.02
            or float(combined.get("breach_rate", 1.0)) < float(three_state.get("breach_rate", 1.0)) - 0.03
            or float(combined.get("average_drawdown_before_pass", float("-inf")))
            > float(three_state.get("average_drawdown_before_pass", float("-inf"))) + 250.0
        )
    )

    x2_core = stress_summary.loc[
        stress_summary["variant_name"].isin(spec.core_variant_names)
        & stress_summary["risk_profile"].eq("base")
        & stress_summary["stress_profile"].eq("slippage_x2")
    ].copy()
    x2_best = x2_core.sort_values(
        ["pass_rate", "breach_rate", "median_days_to_pass"],
        ascending=[False, True, True],
        na_position="last",
    ).iloc[0] if not x2_core.empty else pd.Series(dtype="object")
    stress_flip = bool(
        not primary_challenge.empty
        and not x2_best.empty
        and str(primary_challenge.get("variant_name")) != str(x2_best.get("variant_name"))
    )

    launch_ready = bool(
        not primary_challenge.empty
        and float(primary_challenge.get("pass_rate", 0.0)) >= 0.50
        and float(primary_challenge.get("breach_rate", 1.0)) <= 0.35
        and (
            x2_best.empty
            or (
                float(x2_best.get("pass_rate", 0.0)) >= 0.40
                and float(x2_best.get("breach_rate", 1.0)) <= 0.45
            )
        )
    )

    notes = [
        "Challenge readiness uses common OOS rolling starts only; no new alpha search is performed.",
        "Risk profiles use a compact global exposure multiplier layered on top of the already-validated overlay behavior.",
        "Slippage stress is applied from audited trade logs using the repo instrument tick value and nominal slippage baseline.",
        "Funded follow-up is a light post-pass lens, not a full commercial funded-account model.",
    ]
    if split_configuration:
        notes.append("The best challenge configuration and the best funded configuration are different.")
    if stress_flip:
        notes.append("The challenge verdict is not perfectly stable under slippage stress.")

    return {
        "run_type": "mnq_orb_prop_challenge_readiness",
        "source_run_root": str(source_root),
        "source_run_timestamp": metadata.get("run_timestamp"),
        "primary_scope": spec.primary_scope,
        "common_rolling_start_count": int(len(common_start_dates)),
        "challenge_rules": asdict(spec.rules),
        "challenge_best_variant": challenge_variant,
        "challenge_best_risk_profile": str(recommended_risk.get("risk_profile", primary_challenge.get("risk_profile", ""))),
        "funded_best_variant": funded_variant,
        "funded_best_risk_profile": str(primary_funded.get("risk_profile", "")),
        "vvix_modulator_improves_challenge_business": bool(vvix_improves_challenge),
        "use_split_configuration": bool(split_configuration),
        "recommended_launch_risk_profile": str(recommended_risk.get("risk_profile", "")),
        "recommended_launch_variant": challenge_variant,
        "stress_flip_detected": bool(stress_flip),
        "launch_readiness": "defendable" if launch_ready else "not_defendable_yet",
        "launch_defendable": bool(launch_ready),
        "assumptions": notes,
    }


def _build_report(
    output_path: Path,
    source_root: Path,
    spec: PropChallengeReadinessSpec,
    common_start_dates: list,
    risk_profile_summary: pd.DataFrame,
    stress_test_summary: pd.DataFrame,
    funded_summary: pd.DataFrame,
    business_summary: pd.DataFrame,
    verdict: dict[str, Any],
) -> None:
    primary_core = _core_primary_business_summary(business_summary, spec)
    challenge_best = primary_core.iloc[0] if not primary_core.empty else pd.Series(dtype="object")
    challenge_second = primary_core.iloc[1] if len(primary_core) > 1 else pd.Series(dtype="object")

    funded_primary = business_summary.loc[
        business_summary["variant_name"].isin(spec.core_variant_names)
        & business_summary["risk_profile"].eq("base")
        & business_summary["stress_profile"].eq("slippage_nominal")
    ].copy().sort_values(
        ["followup_expected_profit_per_trading_day", "followup_complete_rate", "followup_breach_rate"],
        ascending=[False, False, True],
        na_position="last",
    )
    funded_best = funded_primary.iloc[0] if not funded_primary.empty else pd.Series(dtype="object")

    lines = [
        "# MNQ ORB Prop Challenge Readiness",
        "",
        "## Scope",
        f"- Source export: `{source_root}`",
        f"- Scope used: `{spec.primary_scope}`",
        f"- Common rolling starts across included variants: `{len(common_start_dates)}`",
        f"- Core comparison: `{spec.core_variant_names[0]}` vs `{spec.core_variant_names[1]}`",
        f"- Additional bounds included: `{[name for name in spec.variant_names if name not in spec.core_variant_names]}`",
        "",
        "## Central Rules",
        f"- Account size: `{spec.rules.account_size_usd:,.0f} USD`",
        f"- Profit target: `{spec.rules.profit_target_usd:,.0f} USD`",
        f"- Daily loss limit: `{spec.rules.daily_loss_limit_usd}`",
        f"- Static max loss: `{spec.rules.static_max_loss_usd}`",
        f"- Trailing drawdown: `{spec.rules.trailing_drawdown_usd}`",
        f"- Max trading days: `{spec.rules.max_trading_days}`",
        f"- Daily cut on DLL hit: `{spec.rules.cut_day_on_daily_loss}`",
        "",
        "## Primary Challenge Readout",
    ]

    if not challenge_best.empty:
        lines.append(
            f"- Best challenge row on primary settings: `{challenge_best['variant_name']}` | pass `{challenge_best['pass_rate']:.1%}` | breach `{challenge_best['breach_rate']:.1%}` | median days `{challenge_best['median_days_to_pass']:.1f}` | expected attempt pnl `{challenge_best['expected_net_profit_per_attempt']:.2f}`."
        )
    if not challenge_second.empty:
        lines.append(
            f"- Runner-up core row: `{challenge_second['variant_name']}` | pass `{challenge_second['pass_rate']:.1%}` | breach `{challenge_second['breach_rate']:.1%}` | median days `{challenge_second['median_days_to_pass']:.1f}`."
        )
    if not funded_best.empty:
        lines.append(
            f"- Best funded lens row on primary settings: `{funded_best['variant_name']}` | complete `{funded_best['followup_complete_rate']:.1%}` | breach `{funded_best['followup_breach_rate']:.1%}` | follow-up profit/day `{funded_best['followup_expected_profit_per_trading_day']:.2f}`."
        )

    lines.extend(
        [
            "",
            "## Risk Profile Summary",
            "",
            "```text",
            risk_profile_summary[
                [
                    "variant_name",
                    "risk_profile",
                    "pass_rate",
                    "breach_rate",
                    "median_days_to_pass",
                    "expected_net_profit_per_attempt",
                ]
            ].to_string(index=False)
            if not risk_profile_summary.empty
            else "No risk profile rows.",
            "```",
            "",
            "## Stress Summary",
            "",
            "```text",
            stress_test_summary[
                [
                    "variant_name",
                    "stress_profile",
                    "pass_rate",
                    "breach_rate",
                    "median_days_to_pass",
                    "expected_net_profit_per_attempt",
                ]
            ].to_string(index=False)
            if not stress_test_summary.empty
            else "No stress rows.",
            "```",
            "",
            "## Direct Answers",
            f"- Best configuration to pass the challenge: `{verdict.get('challenge_best_variant')}` with risk profile `{verdict.get('challenge_best_risk_profile')}`.",
            f"- Best configuration for longer-term / funded offense: `{verdict.get('funded_best_variant')}`.",
            f"- Does the VVIX modulator improve challenge business value even if it does not beat raw 3-state performance? `{'Yes' if verdict.get('vvix_modulator_improves_challenge_business') else 'No'}`.",
            f"- Single universal configuration or split challenge/funded setup? `{'Split configuration recommended' if verdict.get('use_split_configuration') else 'A single configuration is sufficient on current evidence'}`.",
            f"- Most defendable live launch risk profile: `{verdict.get('recommended_launch_risk_profile')}` on `{verdict.get('recommended_launch_variant')}`.",
            "",
            "## Funded Lens",
            "",
            "```text",
            funded_summary[
                [
                    "variant_name",
                    "risk_profile",
                    "stress_profile",
                    "followup_complete_rate",
                    "followup_breach_rate",
                    "followup_expected_profit_per_trading_day",
                    "followup_big_negative_day_rate",
                ]
            ].to_string(index=False)
            if not funded_summary.empty
            else "No funded follow-up rows.",
            "```",
            "",
            "## Business Verdict",
            f"- Launch readiness: `{verdict.get('launch_readiness')}`.",
            f"- Stress flip detected: `{verdict.get('stress_flip_detected')}`.",
            "- This report is decision-oriented and does not claim discovery of a new alpha source.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_campaign(spec: PropChallengeReadinessSpec) -> dict[str, Path]:
    ensure_directories()
    source_root = spec.source_run_root or _find_latest_source_run()
    metadata = _read_run_metadata(source_root)
    source_summary_rows = _summary_row_map(source_root)
    source_spec = metadata.get("spec", {})
    baseline_payload = source_spec.get("baseline", {})
    default_account_size_usd = float(baseline_payload.get("account_size_usd", spec.rules.account_size_usd))
    symbol = str(metadata.get("selected_symbol") or source_spec.get("symbol") or "MNQ")
    instrument = get_instrument_spec(symbol)
    base_slippage_ticks = float(instrument["slippage_ticks"])
    tick_value_usd = float(instrument["tick_value_usd"])
    is_fraction = _source_is_fraction(metadata)

    variant_inputs = [
        _load_variant_input(
            source_root=source_root,
            variant_name=variant_name,
            summary_rows=source_summary_rows,
            default_account_size_usd=default_account_size_usd,
        )
        for variant_name in spec.variant_names
    ]
    common_sessions = _common_scope_sessions(variant_inputs, is_fraction=is_fraction, scope=spec.primary_scope)
    if not common_sessions:
        raise RuntimeError("No common scoped sessions were available across the requested variants.")

    scoped_variants: list[VariantInput] = []
    for variant in variant_inputs:
        scoped_variants.append(
            VariantInput(
                variant_name=variant.variant_name,
                label=variant.label,
                source_root=variant.source_root,
                trades=_subset_sessions(variant.trades, common_sessions),
                daily_results=_subset_sessions(variant.daily_results, common_sessions),
                controls=_subset_sessions(variant.controls, common_sessions) if not variant.controls.empty and "session_date" in variant.controls.columns else variant.controls.copy(),
                reference_account_size_usd=variant.reference_account_size_usd,
                source_summary_row=variant.source_summary_row,
            )
        )

    scenario_daily_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    for variant in scoped_variants:
        for risk_profile in spec.risk_profiles:
            for stress_profile in spec.stress_profiles:
                _, scenario_daily = _scenario_daily_results(
                    variant=variant,
                    sessions=common_sessions,
                    risk_profile=risk_profile,
                    stress_profile=stress_profile,
                    tick_value_usd=tick_value_usd,
                    base_slippage_ticks=base_slippage_ticks,
                )
                scenario_daily_cache[(variant.variant_name, risk_profile.name, stress_profile.name)] = scenario_daily

    common_start_dates = _common_rolling_start_dates(
        scenario_daily_map={
            variant.variant_name: scenario_daily_cache[(variant.variant_name, "base", "slippage_nominal")]
            for variant in scoped_variants
        },
        rules=spec.rules,
    )
    if not common_start_dates:
        raise RuntimeError("No common rolling starts satisfy the challenge horizon across requested variants.")

    attempt_rows: list[dict[str, Any]] = []
    funded_rows: list[dict[str, Any]] = []

    for variant in scoped_variants:
        for risk_profile in spec.risk_profiles:
            for stress_profile in spec.stress_profiles:
                scenario_daily = scenario_daily_cache[(variant.variant_name, risk_profile.name, stress_profile.name)]
                ordered = _normalize_daily_results(scenario_daily)
                start_date_set = set(pd.to_datetime(pd.Index(common_start_dates)).date)

                for idx in _eligible_rolling_start_indices(ordered, max_trading_days=spec.rules.max_trading_days):
                    start_session_date = ordered.iloc[idx]["session_date"]
                    if start_session_date not in start_date_set:
                        continue
                    subset = ordered.iloc[idx:].copy().reset_index(drop=True)
                    history, result = simulate_challenge_attempt(
                        daily_results=subset,
                        rules=spec.rules,
                        reference_account_size_usd=variant.reference_account_size_usd,
                    )
                    attempt_rows.append(
                        {
                            "variant_name": variant.variant_name,
                            "risk_profile": risk_profile.name,
                            "risk_multiplier": float(risk_profile.multiplier),
                            "stress_profile": stress_profile.name,
                            "stress_slippage_multiplier": float(stress_profile.slippage_multiplier),
                            "start_session_date": start_session_date,
                            "source_session_count": int(len(subset)),
                            "daily_limit_hit_or_near_day_count": int(
                                result["daily_limit_hit_day_count"] + result["near_daily_limit_day_count"]
                            ),
                            **result,
                        }
                    )

                    if spec.funded_followup.enabled and bool(result["pass"]):
                        pass_traded_days = int(float(result["days_to_pass"]))
                        pass_index = 0
                        traded_counter = 0
                        for pass_index, (_, pass_row) in enumerate(subset.iterrows()):
                            traded = bool(float(pd.to_numeric(pd.Series([pass_row.get("daily_trade_count", 0.0)]), errors="coerce").iloc[0]) > 0.0)
                            if traded:
                                traded_counter += 1
                            if traded_counter >= pass_traded_days:
                                break
                        followup_subset = subset.iloc[pass_index + 1 :].copy().reset_index(drop=True)
                        funded_rows.append(
                            {
                                "variant_name": variant.variant_name,
                                "risk_profile": risk_profile.name,
                                "risk_multiplier": float(risk_profile.multiplier),
                                "stress_profile": stress_profile.name,
                                "stress_slippage_multiplier": float(stress_profile.slippage_multiplier),
                                "start_session_date": start_session_date,
                                "pass_session_date": history.iloc[-1]["session_date"] if not history.empty else pd.NA,
                                **simulate_funded_followup(
                                    daily_results=followup_subset,
                                    rules=spec.rules,
                                    spec=spec.funded_followup,
                                    pre_pass_history=history,
                                    reference_account_size_usd=variant.reference_account_size_usd,
                                ),
                            }
                        )

    rolling_start_summary = pd.DataFrame(attempt_rows)
    funded_followup_paths = pd.DataFrame(funded_rows)

    challenge_outcome_summary = pd.DataFrame(
        [
            {
                "variant_name": variant_name,
                "risk_profile": risk_profile_name,
                "stress_profile": stress_profile_name,
                **aggregate_challenge_attempts(group),
            }
            for (variant_name, risk_profile_name, stress_profile_name), group in rolling_start_summary.groupby(
                ["variant_name", "risk_profile", "stress_profile"],
                sort=True,
            )
        ]
    )
    funded_followup_summary = pd.DataFrame(
        [
            {
                "variant_name": variant_name,
                "risk_profile": risk_profile_name,
                "stress_profile": stress_profile_name,
                **aggregate_funded_followups(group),
            }
            for (variant_name, risk_profile_name, stress_profile_name), group in funded_followup_paths.groupby(
                ["variant_name", "risk_profile", "stress_profile"],
                sort=True,
            )
        ]
    )

    risk_profile_summary = challenge_outcome_summary.loc[
        challenge_outcome_summary["stress_profile"].eq("slippage_nominal")
    ].copy().sort_values(
        ["variant_name", "pass_rate", "breach_rate", "median_days_to_pass"],
        ascending=[True, False, True, True],
        na_position="last",
    )
    stress_test_summary = challenge_outcome_summary.loc[
        challenge_outcome_summary["risk_profile"].eq("base")
    ].copy().sort_values(
        ["variant_name", "pass_rate", "breach_rate", "median_days_to_pass"],
        ascending=[True, False, True, True],
        na_position="last",
    )

    business_summary = challenge_outcome_summary.merge(
        funded_followup_summary,
        on=["variant_name", "risk_profile", "stress_profile"],
        how="left",
    )
    business_summary["is_core_variant"] = business_summary["variant_name"].isin(spec.core_variant_names)
    business_summary["is_primary_business_scenario"] = (
        business_summary["risk_profile"].eq("base") & business_summary["stress_profile"].eq("slippage_nominal")
    )
    business_summary = business_summary.sort_values(
        [
            "is_primary_business_scenario",
            "is_core_variant",
            "pass_rate",
            "breach_rate",
            "median_days_to_pass",
            "followup_complete_rate",
            "followup_expected_profit_per_trading_day",
        ],
        ascending=[False, False, False, True, True, False, False],
        na_position="last",
    ).reset_index(drop=True)

    verdict = _build_final_verdict(
        challenge_summary=challenge_outcome_summary,
        stress_summary=stress_test_summary,
        business_summary=business_summary,
        spec=spec,
        source_root=source_root,
        metadata=metadata,
        common_start_dates=common_start_dates,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = spec.output_root or (EXPORTS_DIR / f"mnq_orb_prop_challenge_readiness_{timestamp}")
    output_root.mkdir(parents=True, exist_ok=True)

    rules_path = output_root / "challenge_rules.csv"
    challenge_outcome_path = output_root / "challenge_outcome_summary.csv"
    rolling_start_path = output_root / "rolling_start_summary.csv"
    risk_profile_path = output_root / "risk_profile_summary.csv"
    stress_test_path = output_root / "stress_test_summary.csv"
    funded_summary_path = output_root / "funded_followup_summary.csv"
    funded_paths_path = output_root / "funded_followup_paths.csv"
    business_summary_path = output_root / "business_summary.csv"
    report_path = output_root / "final_report.md"
    verdict_path = output_root / "final_verdict.json"
    metadata_path = output_root / "run_metadata.json"

    pd.DataFrame([asdict(spec.rules)]).to_csv(rules_path, index=False)
    challenge_outcome_summary.to_csv(challenge_outcome_path, index=False)
    rolling_start_summary.to_csv(rolling_start_path, index=False)
    risk_profile_summary.to_csv(risk_profile_path, index=False)
    stress_test_summary.to_csv(stress_test_path, index=False)
    funded_followup_summary.to_csv(funded_summary_path, index=False)
    funded_followup_paths.to_csv(funded_paths_path, index=False)
    business_summary.to_csv(business_summary_path, index=False)
    _json_dump(verdict_path, verdict)
    _json_dump(
        metadata_path,
        {
            "run_timestamp": datetime.now().isoformat(),
            "source_run_root": str(source_root),
            "source_metadata": metadata,
            "primary_scope": spec.primary_scope,
            "variant_names": list(spec.variant_names),
            "core_variant_names": list(spec.core_variant_names),
            "common_scope_session_count": int(len(common_sessions)),
            "common_rolling_start_count": int(len(common_start_dates)),
            "rules": asdict(spec.rules),
            "risk_profiles": [asdict(profile) for profile in spec.risk_profiles],
            "stress_profiles": [asdict(profile) for profile in spec.stress_profiles],
            "funded_followup": asdict(spec.funded_followup),
            "symbol": symbol,
            "instrument": instrument,
            "random_seed": spec.random_seed,
        },
    )
    _build_report(
        output_path=report_path,
        source_root=source_root,
        spec=spec,
        common_start_dates=common_start_dates,
        risk_profile_summary=risk_profile_summary,
        stress_test_summary=stress_test_summary,
        funded_summary=funded_followup_summary,
        business_summary=business_summary,
        verdict=verdict,
    )

    return {
        "output_root": output_root,
        "challenge_rules": rules_path,
        "business_summary": business_summary_path,
        "challenge_outcome_summary": challenge_outcome_path,
        "rolling_start_summary": rolling_start_path,
        "risk_profile_summary": risk_profile_path,
        "stress_test_summary": stress_test_path,
        "funded_followup_summary": funded_summary_path,
        "funded_followup_paths": funded_paths_path,
        "final_report": report_path,
        "final_verdict": verdict_path,
        "run_metadata": metadata_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, default=None)
    parser.add_argument("--primary-scope", type=str, default=DEFAULT_PRIMARY_SCOPE, choices=("overall", "oos"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--profit-target-usd", type=float, default=3_000.0)
    parser.add_argument("--daily-loss-limit-usd", type=float, default=1_000.0)
    parser.add_argument("--static-max-loss-usd", type=float, default=None)
    parser.add_argument("--trailing-drawdown-usd", type=float, default=2_000.0)
    parser.add_argument("--max-trading-days", type=int, default=35)
    parser.add_argument("--funded-trading-days", type=int, default=60)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    artifacts = run_campaign(
        PropChallengeReadinessSpec(
            source_run_root=args.source_run_root,
            primary_scope=str(args.primary_scope),
            rules=PropChallengeRules(
                profit_target_usd=float(args.profit_target_usd),
                daily_loss_limit_usd=(None if args.daily_loss_limit_usd is None else float(args.daily_loss_limit_usd)),
                static_max_loss_usd=(None if args.static_max_loss_usd is None else float(args.static_max_loss_usd)),
                trailing_drawdown_usd=(None if args.trailing_drawdown_usd is None else float(args.trailing_drawdown_usd)),
                max_trading_days=None if args.max_trading_days is None else int(args.max_trading_days),
            ),
            funded_followup=FundedFollowupSpec(
                enabled=True,
                trading_days=int(args.funded_trading_days),
            ),
            output_root=args.output_root,
        )
    )
    print(f"output_root: {artifacts['output_root']}")
    print(f"business_summary: {artifacts['business_summary']}")
    print(f"final_report: {artifacts['final_report']}")


if __name__ == "__main__":
    main()
