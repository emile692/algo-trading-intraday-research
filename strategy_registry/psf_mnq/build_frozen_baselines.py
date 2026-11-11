"""Build frozen, executable baseline specifications for the 16 mechanisms.

This script is specification-only. It reads registry metadata and writes JSON,
CSV, and Markdown files. It never imports or invokes a backtest engine and it
never reads performance results.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "frozen_baselines"
AUDIT_PATH = ROOT / "alpha_engine_logical_audit.csv"
MECHANISMS_PATH = ROOT / "alpha_engine_mechanisms.csv"
MANIFEST_PATH = ROOT / "frozen_baselines_manifest.csv"
REVIEW_PATH = ROOT / "FROZEN_BASELINES_REVIEW.md"

SCHEMA_VERSION = "1.0"
TIMEZONE = "America/New_York"
RTH_START = "09:30:00"
CASH_CLOSE = "16:00:00"
DEFAULT_LAST_ENTRY_TIME = "15:55:00"
FORCED_FLATTEN_TIME = "16:08:00"
EARLY_CLOSE_LAST_ENTRY_BUFFER_MINUTES = 5
EARLY_CLOSE_FLATTEN_BUFFER_MINUTES = 1
TICK_SIZE = 0.25
TICK_VALUE_USD = 0.50
POINT_VALUE_USD = 2.00
COMMISSION_PER_SIDE_USD = 1.25
SLIPPAGE_TICKS_PER_SIDE = 1


def parameter(
    name: str,
    value: Any,
    unit: str,
    origin: str,
    description: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "origin": origin,
        "description": description,
    }


def assumption(assumption_id: str, description: str) -> dict[str, str]:
    return {
        "type": "research_assumption",
        "id": assumption_id,
        "description": description,
    }


def risk_control(risk: str, control: str) -> dict[str, str]:
    return {"risk": risk, "control": control}


def canonical_parameter_map(parameters: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return {
        category: {item["name"]: item["value"] for item in items}
        for category, items in parameters.items()
    }


def common_execution_parameters(last_entry_time: str) -> list[dict[str, Any]]:
    return [
        parameter("tick_size", TICK_SIZE, "index_points", "repository_execution_default", "MNQ minimum price increment."),
        parameter("tick_value_usd", TICK_VALUE_USD, "USD_per_tick", "repository_execution_default", "MNQ tick value per contract."),
        parameter("point_value_usd", POINT_VALUE_USD, "USD_per_point", "repository_execution_default", "MNQ point value per contract."),
        parameter("commission_per_side_usd", COMMISSION_PER_SIDE_USD, "USD_per_contract", "repository_execution_default", "Fixed commission on entry and exit."),
        parameter("slippage_ticks_per_side", SLIPPAGE_TICKS_PER_SIDE, "ticks", "repository_execution_default", "Adverse slippage applied on entry and exit."),
        parameter("session_timezone", TIMEZONE, "IANA_timezone", "repository_session_default", "All session times use New York civil time and DST."),
        parameter("rth_start", RTH_START, "local_time", "repository_session_default", "Start of the research session."),
        parameter("cash_close", CASH_CLOSE, "local_time", "user_topstep_rule", "Regular-day cash-session close; it is not an automatic flatten unless a strategy explicitly exits there."),
        parameter("last_entry_time", last_entry_time, "local_time", "user_topstep_rule", "No new position may be opened after this time."),
        parameter("forced_flatten_time", FORCED_FLATTEN_TIME, "local_time", "user_topstep_rule", "Regular-day Topstep hard flatten; no position may remain afterward."),
        parameter("early_close_calendar", "official_CME_equity_index_futures_calendar", "calendar", "research_assumption", "Required source for early-close timestamps; never infer them from missing bars."),
        parameter("early_close_last_entry_buffer_minutes", EARLY_CLOSE_LAST_ENTRY_BUFFER_MINUTES, "minutes", "research_assumption", "On an early-close day, stop new entries five minutes before the official close."),
        parameter("early_close_flatten_buffer_minutes", EARLY_CLOSE_FLATTEN_BUFFER_MINUTES, "minutes", "research_assumption", "On an early-close day, hard-flatten one minute before the official close."),
        parameter("position_policy", "one_position_no_pyramiding", "policy", "research_assumption", "At most one open position; no scaling or pyramiding."),
        parameter("bar_conflict_policy", "stop_before_target", "policy", "research_assumption", "Conservative ordering when stop and target occur in one execution bar."),
        parameter("missing_execution_bar_policy", "skip_signal", "policy", "research_assumption", "Skip a fill if the exact required next one-minute bar is absent."),
        parameter("price_rounding_policy", "adverse_to_strategy_to_valid_tick", "policy", "research_assumption", "Order levels are rounded to a valid tick without improving the assumed fill."),
    ]


def common_risk_parameters() -> list[dict[str, Any]]:
    return [
        parameter("fixed_quantity", 1, "MNQ_contract", "research_assumption", "One contract keeps baseline sizing independent of signal logic."),
        parameter("hard_flatten_time", FORCED_FLATTEN_TIME, "local_time", "user_topstep_rule", "No strategy position may remain open after the Topstep hard flatten."),
    ]


def strategy_spec(
    *,
    mechanism_id: str,
    strategy_id: str,
    research_family: str,
    complexity_grade: str,
    timeframe: str,
    hypothesis: str,
    economic_rationale: str,
    long_direction: str,
    short_direction: str,
    calculation: str,
    signal_timestamp: str,
    entry_condition: str,
    entry_price: str,
    execution_delay: str,
    target: str,
    stop: str,
    time_exit: str,
    structural_parameters: list[dict[str, Any]],
    risk_parameters: list[dict[str, Any]],
    execution_parameters: list[dict[str, Any]] | None = None,
    data_notes: list[str] | None = None,
    lookahead_risks: list[dict[str, str]] | None = None,
    same_bar_risks: list[dict[str, str]] | None = None,
    execution_ambiguities: list[dict[str, str]] | None = None,
    remaining_assumptions: list[dict[str, str]] | None = None,
    last_entry_time: str = DEFAULT_LAST_ENTRY_TIME,
) -> dict[str, Any]:
    parameters = {
        "structural": [
            parameter("signal_timeframe", timeframe, "bar_interval", "published_horizon", "Frozen signal timeframe selected without performance evidence."),
            *structural_parameters,
        ],
        "risk_management": [*common_risk_parameters(), *risk_parameters],
        "execution": [*common_execution_parameters(last_entry_time), *(execution_parameters or [])],
    }
    default_lookahead = [
        risk_control(
            "Using a signal bar's final high, low, close, volume, or indicator value before that bar closes.",
            "All signal calculations use completed bars only; no order based on that bar can fill before its close.",
        ),
        risk_control(
            "Rolling windows accidentally include future bars or the current breakout observation in their reference level.",
            "Windows are trailing and causal; reference highs/lows explicitly exclude the bar being evaluated when stated by the strategy.",
        ),
    ]
    default_same_bar = [
        risk_control(
            "A target and stop can both lie inside the same one-minute bar.",
            "Record the stop first, the conservative path, because intraminute ordering is unavailable.",
        ),
        risk_control(
            "A close-confirmed signal cannot be filled at that same close.",
            "Use the exact next one-minute open and apply adverse slippage; skip the signal when that bar is missing.",
        ),
    ]
    default_ambiguities = [
        risk_control(
            "The source cards do not define a universal execution and cost model.",
            "Use one MNQ contract, one tick adverse slippage per side, and USD 1.25 commission per side from repository defaults.",
        ),
        risk_control(
            "The source cards do not consistently define end-of-session handling.",
            "Treat 16:00 cash close, the last-entry cutoff, and the 16:08 Topstep hard flatten as distinct events. Continue risk exits after cash close, but never retain a position after hard flatten.",
        ),
        risk_control(
            "CME holiday sessions can close before the regular cash close and hard-flatten times.",
            "Require the official CME early-close timestamp; set last entry to min(strategy cutoff, early close minus five minutes) and hard flatten to min(16:08, early close minus one minute). Reject the session if the calendar value is unavailable.",
        ),
    ]
    assumptions = [
        assumption("rth_only_signal_bars", "Indicators and signal bars use only 09:30–16:00 RTH observations; completed prior sessions may remain in rolling history."),
        assumption("frozen_signal_timeframe", "The signal timeframe is selected from the horizons published on the source card for mechanical compatibility with its rule, without performance comparison."),
        assumption("one_contract_baseline", "All strategy baselines use exactly one MNQ contract and no pyramiding."),
        assumption("common_execution_model", "All strategies share the repository's fixed MNQ commission, tick size, and slippage defaults."),
        assumption("early_close_buffers", "On official CME early-close days, last entry is five minutes before the early close and hard flatten is one minute before it."),
        *(remaining_assumptions or []),
    ]
    spec = {
        "schema_version": SCHEMA_VERSION,
        "specification_type": "strategy_baseline",
        "specification_status": "frozen_ready_with_research_assumptions",
        "mechanism_id": mechanism_id,
        "strategy_id": strategy_id,
        "research_family": research_family,
        "complexity_grade": complexity_grade,
        "source": {
            "card_path": f"cards/{strategy_id}.md",
            "logical_audit_path": "alpha_engine_logical_audit.csv",
            "performance_information_used": False,
            "parameter_optimization_used": False,
        },
        "hypothesis": {
            "statement": hypothesis,
            "economic_rationale": economic_rationale,
        },
        "direction": {"long": long_direction, "short": short_direction},
        "signal": {
            "timeframe": timeframe,
            "calculation": calculation,
            "signal_timestamp": signal_timestamp,
        },
        "entry": {
            "condition": entry_condition,
            "execution_price": entry_price,
            "execution_delay": execution_delay,
            "last_entry_time": last_entry_time,
            "last_entry_policy": "No new position may be opened after this timestamp. Strategy-specific entry rules may impose an earlier effective cutoff.",
        },
        "exit": {
            "target": target,
            "stop": stop,
            "time_exit": time_exit,
            "session_flatten": {
                "enabled": True,
                "daily_flatten": True,
                "conditional_session_flatten": True,
                "topstep_rules_applied": True,
                "cash_close": CASH_CLOSE,
                "cash_close_action": "No automatic flatten unless exit.time_exit explicitly says cash close.",
                "last_entry_time": last_entry_time,
                "forced_flatten_time": FORCED_FLATTEN_TIME,
                "no_position_after_forced_flatten": True,
                "timezone": TIMEZONE,
                "execution": "Exit at the open of the 16:08 one-minute bar. If that exact bar is absent, use the last available close strictly before 16:08. No position may remain afterward.",
                "post_cash_close_monitoring": "From 16:00 through hard flatten, accept no new entries or strategy signals; continue monitoring already-known stops and targets on one-minute bars.",
                "early_close_exception": {
                    "calendar": "official_CME_equity_index_futures_calendar",
                    "cash_close": "official early-close timestamp",
                    "last_entry_time": "min(strategy last_entry_time, early close minus 5 minutes)",
                    "forced_flatten_time": "min(16:08, early close minus 1 minute)",
                    "missing_calendar_policy": "reject_session",
                },
            },
        },
        "data_requirements": {
            "instrument": "MNQ",
            "contract_series": "provided_continuous_front_contract",
            "source_granularity": "1min",
            "signal_timeframe": timeframe,
            "execution_timeframe": "1min",
            "required_fields": ["timestamp", "open", "high", "low", "close", "volume", "session_date"],
            "timezone": TIMEZONE,
            "session": {
                "start": RTH_START,
                "cash_close": CASH_CLOSE,
                "last_entry_time": last_entry_time,
                "forced_flatten_time": FORCED_FLATTEN_TIME,
                "signal_window_end": CASH_CLOSE,
                "position_monitoring_end": FORCED_FLATTEN_TIME,
                "early_close_calendar_required": True,
            },
            "bar_timestamp_semantics": "Input timestamp is bar open; resampled signal intervals are left-closed/right-open and become observable at interval end.",
            "availability": True,
            "notes": [
                "One-minute execution data must extend through the applicable hard-flatten timestamp.",
                "An official CME equity-index-futures calendar with early-close timestamps is mandatory; do not infer an early close from missing bars.",
                *(data_notes or []),
            ],
        },
        "parameters": parameters,
        "canonical_parameters": canonical_parameter_map(parameters),
        "lookahead_risks": [*default_lookahead, *(lookahead_risks or [])],
        "same_bar_risks": [*default_same_bar, *(same_bar_risks or [])],
        "execution_ambiguities": [*default_ambiguities, *(execution_ambiguities or [])],
        "remaining_assumptions": assumptions,
        "unresolved_ambiguities": [],
        "ready_for_baseline_backtest": True,
    }
    return spec


def build_strategy_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    specs.append(strategy_spec(
        mechanism_id="trend_pullback_ma_structure",
        strategy_id="ema-ribbon-trend",
        research_family="trend_momentum",
        complexity_grade="C",
        timeframe="15min",
        hypothesis="An ordered EMA structure identifies persistent trend, and a pullback into the fast ribbon offers continuation entry.",
        economic_rationale="Persistent order flow can keep returns directionally dependent while temporary profit taking pulls price toward fast moving averages.",
        long_direction="EMA 8 > 13 > 21 > 34 > 55; buy a confirmed pullback into the EMA 8–13 zone.",
        short_direction="EMA 8 < 13 < 21 < 34 < 55; sell a confirmed pullback into the EMA 8–13 zone.",
        calculation="Compute close-based EMA 8, 13, 21, 34, and 55 on completed RTH 15-minute bars. Long pullback: the bar overlaps the EMA 8–13 zone and closes at or above EMA 13. Short is symmetric and closes at or below EMA 13.",
        signal_timestamp="Close of the completed 15-minute pullback-confirmation bar.",
        entry_condition="Enter on the first qualifying pullback bar while the five EMAs are strictly ordered; require no open position.",
        entry_price="Next one-minute bar open after the signal, rounded adversely and adjusted by one tick of entry slippage.",
        execution_delay="One one-minute execution bar after signal close.",
        target="No fixed profit target; remain in the trend until crossover exit or session flatten.",
        stop="Initial stop at EMA 55 from the signal bar. Thereafter update on completed 15-minute bars without loosening: max for long, min for short.",
        time_exit="No bar-count exit; the 16:08 Topstep hard flatten applies.",
        structural_parameters=[
            parameter("ema_periods", [8, 13, 21, 34, 55], "bars", "published_rule", "Published ribbon periods."),
            parameter("pullback_zone", "between_ema_8_and_ema_13", "price_zone", "published_rule", "Published fast-ribbon pullback area."),
            parameter("confirmation_close", "long_at_or_above_ema13_short_at_or_below_ema13", "rule", "research_assumption", "Minimal mechanical definition of a confirmed pullback."),
        ],
        risk_parameters=[
            parameter("trend_exit", "ema8_crosses_ema21_against_position", "rule", "published_rule", "Published crossover exit."),
            parameter("protective_stop", "ema55_monotonic", "dynamic_price", "published_rule_plus_research_assumption", "Published EMA 55 stop with a no-loosening convention."),
        ],
        execution_ambiguities=[risk_control("The card does not define the confirmation candle or whether the EMA 55 stop may loosen.", "Require a close back on the trend side of EMA 13 and never loosen the stop.")],
        remaining_assumptions=[assumption("ema_pullback_confirmation", "A bar confirms a long pullback by overlapping the EMA 8–13 zone and closing at or above EMA 13; short is symmetric.")],
    ))

    specs.append(strategy_spec(
        mechanism_id="vwap_equilibrium_reversion",
        strategy_id="vwap-deviation",
        research_family="mean_reversion",
        complexity_grade="B",
        timeframe="5min",
        hypothesis="A two-standard-deviation displacement from session VWAP is a temporary intraday imbalance that reverts toward VWAP.",
        economic_rationale="Session VWAP proxies traded equilibrium; liquidity shocks can push price away from it before two-sided flow restores balance.",
        long_direction="Buy a bullish confirmation after price reaches the -2 sigma VWAP band.",
        short_direction="Sell a bearish confirmation after price reaches the +2 sigma VWAP band.",
        calculation="Reset each RTH session. Use typical price (high+low+close)/3, cumulative volume-weighted VWAP, and cumulative volume-weighted population variance. Bands are VWAP plus or minus 2 and 3 standard deviations.",
        signal_timestamp="Close of a completed 5-minute bar after at least two completed session bars.",
        entry_condition="Long when low touches or breaches -2 sigma and close > open; short when high touches or breaches +2 sigma and close < open.",
        entry_price="Next one-minute bar open after the confirmation bar, with adverse tick rounding and one tick slippage.",
        execution_delay="One one-minute execution bar after confirmation close.",
        target="Fixed VWAP value observed on the signal bar; full exit because the baseline holds one contract.",
        stop="Fixed -3 sigma signal-bar band for long and +3 sigma signal-bar band for short.",
        time_exit="No additional time stop; the 16:08 Topstep hard flatten applies.",
        structural_parameters=[
            parameter("vwap_price_source", "hlc3", "formula", "research_assumption", "Typical price used in the VWAP calculation."),
            parameter("band_variance", "cumulative_volume_weighted_population", "formula", "research_assumption", "Mechanical definition of the published VWAP standard-deviation bands."),
            parameter("entry_band_sigma", 2.0, "standard_deviations", "published_rule", "Published entry band."),
            parameter("minimum_completed_bars", 2, "signal_bars", "research_assumption", "Minimum needed for nonzero session variance."),
            parameter("confirmation_candle", "directional_real_body", "rule", "research_assumption", "Bullish close for long and bearish close for short."),
        ],
        risk_parameters=[
            parameter("target", "signal_bar_vwap", "price", "published_rule", "Published full-reversion target."),
            parameter("stop_band_sigma", 3.0, "standard_deviations", "published_rule", "Published invalidation band."),
            parameter("partial_exit", "disabled_one_contract", "policy", "research_assumption", "Published +/-1 sigma partial is disabled for a one-contract baseline."),
        ],
        lookahead_risks=[risk_control("VWAP variance can be recomputed with future session volume.", "Freeze all VWAP and band values at each completed bar; never use end-of-session totals retrospectively.")],
        same_bar_risks=[risk_control("The signal bar can touch both -2 and -3 sigma before its close.", "No entry occurs in the signal bar; use fixed signal-close bands and enter on the next minute only if entry remains between stop and target.")],
        execution_ambiguities=[risk_control("The card names a confirmation candle without defining it and does not say whether VWAP target and bands move after entry.", "Use candle-body direction and freeze target and stop at signal time.")],
        remaining_assumptions=[
            assumption("vwap_band_formula", "VWAP bands use cumulative volume-weighted population variance of HLC3 from the RTH open."),
            assumption("vwap_confirmation", "Confirmation means a bullish real body for long and bearish real body for short."),
            assumption("fixed_vwap_levels_after_entry", "VWAP target and 3-sigma stop are frozen from the signal bar."),
        ],
    ))

    specs.append(strategy_spec(
        mechanism_id="rolling_range_breakout",
        strategy_id="donchian-breakout",
        research_family="breakout_expansion",
        complexity_grade="B",
        timeframe="1H",
        hypothesis="A close beyond a recent rolling price range starts a directional continuation that persists until a shorter opposite range breaks.",
        economic_rationale="A range escape can reveal information-driven order imbalance; the shorter exit channel responds when that imbalance reverses.",
        long_direction="Buy a close above the highest high of the previous 20 completed one-hour bars.",
        short_direction="Sell a close below the lowest low of the previous 20 completed one-hour bars.",
        calculation="On RTH-aligned one-hour bars, compute the entry Donchian channel from the previous 20 bars and exit channel from the previous 10 bars, always excluding the current bar.",
        signal_timestamp="Close of the completed one-hour breakout bar.",
        entry_condition="Close strictly beyond the corresponding prior 20-bar channel and no open position; simultaneous long and short signals are invalid.",
        entry_price="Next one-minute bar open after the one-hour close, with adverse rounding and one tick slippage.",
        execution_delay="One one-minute execution bar after signal close.",
        target="No fixed target.",
        stop="Channel exit: long exits after a completed one-hour close below the prior 10-bar low; short exits above the prior 10-bar high. Execute at the next one-minute open.",
        time_exit="No multi-session holding; the 16:08 Topstep hard flatten overrides the channel exit.",
        structural_parameters=[
            parameter("entry_channel_length", 20, "1H_bars", "published_rule", "Published Donchian entry length."),
            parameter("exit_channel_length", 10, "1H_bars", "published_rule", "Published Donchian exit length."),
            parameter("breakout_confirmation", "close_beyond_prior_channel", "rule", "research_assumption", "Close confirmation avoids intrabar ordering ambiguity."),
        ],
        risk_parameters=[
            parameter("exit_rule", "opposite_10_bar_channel_close", "rule", "published_rule", "Published shorter-channel exit."),
            parameter("atr_position_sizing", "disabled_fixed_one_contract", "policy", "research_assumption", "Published ATR sizing is separated from alpha for the common baseline."),
        ],
        data_notes=["One-hour bars are aligned from 09:30; only complete bars are eligible for signals."],
        lookahead_risks=[risk_control("Including the breakout bar in its own 20-bar channel makes a strict breakout impossible or biased.", "Shift both Donchian channels by one completed signal bar.")],
        remaining_assumptions=[assumption("donchian_close_confirmation", "A breakout requires the completed one-hour close beyond the prior channel, rather than an intrabar touch.")],
    ))

    specs.append(strategy_spec(
        mechanism_id="atr_trend_flip",
        strategy_id="supertrend",
        research_family="trend_momentum",
        complexity_grade="A",
        timeframe="15min",
        hypothesis="A volatility-adjusted trend state persists after SuperTrend flips until an opposite flip occurs.",
        economic_rationale="ATR-scaled bands adapt the reversal threshold to current noise, allowing persistent moves to remain active while filtering smaller counter-moves.",
        long_direction="Long when the completed-bar SuperTrend state flips from bearish to bullish.",
        short_direction="Short when the completed-bar SuperTrend state flips from bullish to bearish.",
        calculation="Use standard recursive SuperTrend on completed 15-minute RTH bars: Wilder ATR(10), HL2 basic bands at 3 ATR, final bands carried forward, and state flips only when close crosses the prior final band.",
        signal_timestamp="Close of the completed 15-minute bar that confirms the state flip.",
        entry_condition="Enter or reverse in the new SuperTrend direction on a confirmed flip.",
        entry_price="Next one-minute bar open after the flip, with adverse rounding and one tick slippage.",
        execution_delay="One one-minute execution bar after signal close; an opposite flip may close and reverse at that fill.",
        target="No fixed target.",
        stop="Exit or reverse on the next confirmed opposite SuperTrend flip; no intrabar stop derived from an unfinished band.",
        time_exit="Forced session flatten; state may be recomputed from history next session but positions do not carry overnight.",
        structural_parameters=[
            parameter("atr_period", 10, "15min_bars", "research_assumption", "Card omits the SuperTrend ATR period; 10 is frozen without performance testing."),
            parameter("atr_multiplier", 3.0, "ATR", "research_assumption", "Card omits the band multiplier; 3 is frozen without performance testing."),
            parameter("atr_method", "wilder_rma", "formula", "research_assumption", "Explicit ATR smoothing convention."),
        ],
        risk_parameters=[
            parameter("exit_rule", "opposite_confirmed_supertrend_flip", "rule", "published_rule", "Published exit."),
            parameter("optional_tightening", "disabled", "policy", "research_assumption", "Optional faster volatile-market exit is excluded from the canonical baseline."),
        ],
        lookahead_risks=[risk_control("Using a current unfinished bar to flip SuperTrend repaints the signal.", "Only completed 15-minute closes can update final bands and state.")],
        execution_ambiguities=[risk_control("The card does not publish the SuperTrend formula, ATR period, or multiplier.", "Freeze the standard recursive formulation with ATR(10) and multiplier 3.0 as research assumptions.")],
        remaining_assumptions=[assumption("supertrend_defaults", "Use Wilder ATR(10), multiplier 3.0, HL2 basic bands, and the standard carried final-band recursion.")],
    ))

    specs.append(strategy_spec(
        mechanism_id="ma_crossover_trend",
        strategy_id="dual-ma-crossover",
        research_family="trend_momentum",
        complexity_grade="B",
        timeframe="15min",
        hypothesis="A fast EMA crossing a slow EMA during sufficient directional strength identifies a trend likely to persist.",
        economic_rationale="Smoothed price acceleration can proxy persistent order flow, while ADX suppresses low-directionality crossovers.",
        long_direction="Long when EMA 9 crosses above EMA 21 and ADX(14) > 20.",
        short_direction="Short when EMA 9 crosses below EMA 21 and ADX(14) > 20.",
        calculation="Compute close-based EMA 9 and EMA 21 plus Wilder DMI/ADX(14) on completed RTH 15-minute bars.",
        signal_timestamp="Close of the completed 15-minute crossover bar.",
        entry_condition="A strict EMA crossover occurs on the current completed bar and ADX is greater than 20.",
        entry_price="Next one-minute bar open after the signal, with adverse rounding and one tick slippage.",
        execution_delay="One one-minute execution bar; an opposite qualified crossover may close and reverse at that fill.",
        target="No fixed target.",
        stop="Exit on the next completed-bar opposing EMA crossover, whether or not ADX confirms the exit.",
        time_exit="Forced session flatten.",
        structural_parameters=[
            parameter("fast_ema_period", 9, "15min_bars", "published_rule", "Published fast EMA."),
            parameter("slow_ema_period", 21, "15min_bars", "published_rule", "Published slow EMA."),
            parameter("adx_period", 14, "15min_bars", "research_assumption", "Card specifies ADX but not its period."),
            parameter("adx_entry_threshold", 20.0, "ADX", "published_rule", "Published trend-strength threshold."),
        ],
        risk_parameters=[
            parameter("exit_rule", "opposing_ema_crossover", "rule", "published_rule", "Published primary exit."),
            parameter("optional_adx_early_exit", "disabled", "policy", "research_assumption", "Optional ADX < 15 exit is excluded."),
        ],
        execution_ambiguities=[risk_control("The ADX period is absent and the ADX < 15 exit is optional.", "Freeze Wilder ADX(14) and omit the optional early exit.")],
        remaining_assumptions=[assumption("dual_ma_adx_period", "Use Wilder DMI/ADX with a 14-bar period.")],
    ))

    specs.append(strategy_spec(
        mechanism_id="session_range_breakout",
        strategy_id="opening-range-breakout",
        research_family="breakout_expansion",
        complexity_grade="B",
        timeframe="5min",
        hypothesis="A confirmed close outside the first 30-minute range continues as early-session price discovery resolves directionally.",
        economic_rationale="The opening range aggregates overnight repricing and initial liquidity; a volume-expanding escape can indicate persistent imbalance.",
        long_direction="Buy a close at least one tick above the opening-range high with expanding volume.",
        short_direction="Sell a close at least one tick below the opening-range low with expanding volume.",
        calculation="From 09:30 inclusive to 10:00 exclusive, freeze the high, low, midpoint, and range. After 10:00, evaluate completed 5-minute closes. Expanding volume means current 5-minute volume is greater than the immediately preceding completed 5-minute bar.",
        signal_timestamp="Close of the first qualifying 5-minute bar after 10:00.",
        entry_condition="Close beyond the corresponding range boundary by at least one MNQ tick and current bar volume > previous bar volume.",
        entry_price="Next one-minute bar open after the qualifying close, with adverse rounding and one tick slippage; skip if entry is already beyond the target or stop.",
        execution_delay="One one-minute execution bar after signal close.",
        target="Opening-range high plus one full opening-range width for long; opening-range low minus one width for short.",
        stop="Fixed opening-range midpoint.",
        time_exit="Forced session flatten; no optional post-target trail in the one-contract baseline.",
        structural_parameters=[
            parameter("opening_range_start", "09:30:00", "local_time", "published_rule", "Start of the first 30 minutes."),
            parameter("opening_range_minutes", 30, "minutes", "published_rule", "Published opening-range duration."),
            parameter("breakout_buffer_ticks", 1, "ticks", "published_rule_plus_research_assumption", "Published long buffer applied symmetrically to short."),
            parameter("volume_expansion", "current_5m_volume_gt_previous_5m_volume", "rule", "research_assumption", "Minimal definition of expanding volume."),
        ],
        risk_parameters=[
            parameter("target_range_multiple", 1.0, "opening_range", "published_rule", "Published measured target."),
            parameter("stop_location", "opening_range_midpoint", "price", "published_rule", "Published stop inside the range."),
            parameter("optional_trailing_stop", "disabled", "policy", "research_assumption", "Optional trail is excluded for one contract."),
        ],
        lookahead_risks=[risk_control("Using the full opening range before 10:00 or updating it later leaks future highs/lows.", "Freeze the range only after the 09:55–10:00 bar is complete; never update it afterward.")],
        execution_ambiguities=[risk_control("Expanding volume and the short-side tick buffer are not fully specified.", "Compare with the immediately prior 5-minute volume and use a symmetric one-tick buffer.")],
        remaining_assumptions=[
            assumption("opening_range_volume_expansion", "Expanding volume means current completed 5-minute volume exceeds the immediately prior completed bar."),
            assumption("opening_range_symmetric_buffer", "Apply the published one-tick close buffer to both long and short signals."),
            assumption("opening_range_five_minute_bars", "Use the published 5-minute horizon so the 30-minute opening range contains six complete bars."),
        ],
    ))

    specs.append(strategy_spec(
        mechanism_id="compression_breakout",
        strategy_id="range-compression-breakout",
        research_family="breakout_expansion",
        complexity_grade="C",
        timeframe="1H",
        hypothesis="An inside NR7 bar represents volatility compression whose eventual range break initiates expansion.",
        economic_rationale="Volatility clusters; unusually narrow balance can precede repricing when resting liquidity is exhausted on one side.",
        long_direction="Place a buy stop one tick above the completed inside NR7 bar high.",
        short_direction="Place a sell stop one tick below the completed inside NR7 bar low.",
        calculation="A setup bar has the smallest high-low range among itself and the previous six completed one-hour bars, and its high is <= the previous bar high while its low is >= the previous bar low. Ties qualify only if no earlier bar in the seven-bar window has the same minimum range.",
        signal_timestamp="Close of the completed one-hour setup bar; entry orders become active on the next one-minute bar.",
        entry_condition="First side of the setup range touched after activation. If both entry levels trade in the same one-minute bar, cancel the setup without a trade.",
        entry_price="Stop-order level, or the adverse one-minute open if price gaps beyond it, plus one tick adverse slippage.",
        execution_delay="Resting stop entries are active from the first one-minute bar after setup close until session flatten.",
        target="Two setup-bar ranges projected from the triggered breakout level.",
        stop="Opposite side of the setup bar.",
        time_exit="Cancel unfilled entry orders after the applicable last-entry cutoff; hard-flatten any open position at 16:08. Setups do not carry to the next session.",
        structural_parameters=[
            parameter("nr_lookback", 7, "1H_bars", "published_rule", "Published NR7 lookback."),
            parameter("inside_bar_required", True, "boolean", "published_rule", "Published conjunction with inside bar."),
            parameter("entry_buffer_ticks", 1, "ticks", "research_assumption", "Minimal executable buffer beyond the setup range."),
            parameter("bb_width_confirmation", "disabled_optional", "policy", "research_assumption", "Optional Bollinger-width confirmation is excluded."),
            parameter("nr_tie_policy", "earliest_minimum_only", "policy", "research_assumption", "Avoid repeated setups from tied ranges."),
        ],
        risk_parameters=[
            parameter("target_setup_range_multiple", 2.0, "setup_range", "published_rule", "Published target."),
            parameter("stop_location", "opposite_setup_bar_extreme", "price", "published_rule", "Published stop."),
            parameter("optional_trail", "disabled_one_contract", "policy", "research_assumption", "Ambiguous target-1 trail is excluded."),
        ],
        same_bar_risks=[
            risk_control("Both long and short entry levels can trade inside one one-minute bar.", "Cancel the setup because entry order is unknowable without intraminute sequencing."),
            risk_control("Entry and opposite-side stop can both trade in the entry minute.", "Assume entry then stop, the conservative executable path."),
        ],
        execution_ambiguities=[risk_control("The source does not define buffers, tie handling, order lifetime, or the optional trail's first target.", "Use a one-tick buffer, earliest tie only, same-session orders, and disable the optional trail.")],
        remaining_assumptions=[
            assumption("nr7_tie_rule", "A tied minimum qualifies only when the setup bar is the first occurrence of that minimum in the seven-bar window."),
            assumption("nr7_entry_buffer", "Entry stops sit one MNQ tick outside the setup bar."),
        ],
    ))

    specs.append(strategy_spec(
        mechanism_id="statistical_zscore_reversion",
        strategy_id="z-score-mean-reversion",
        research_family="mean_reversion",
        complexity_grade="B",
        timeframe="1H",
        hypothesis="A close more than two rolling standard deviations from its mean reverts toward that mean.",
        economic_rationale="Temporary order imbalance can move price beyond its recent statistical equilibrium before liquidity provision restores it.",
        long_direction="Buy when z-score crosses from >= -2 to < -2.",
        short_direction="Sell when z-score crosses from <= +2 to > +2.",
        calculation="On completed RTH one-hour closes, compute SMA(20), population standard deviation with ddof=0, and z=(close-SMA)/sigma. Require 20 valid observations and sigma > 0.",
        signal_timestamp="Close of the completed one-hour threshold-crossing bar.",
        entry_condition="First threshold crossing while flat; persistent occupancy beyond the threshold does not create repeated entries.",
        entry_price="Next one-minute bar open, with adverse rounding and one tick slippage.",
        execution_delay="One one-minute execution bar after signal close.",
        target="Indicator exit: long when completed-bar z >= -0.5; short when z <= +0.5. Execute next one-minute open.",
        stop="Indicator stop: long when z <= -3.0; short when z >= +3.0. If target and stop conditions conflict, stop prevails.",
        time_exit="Forced session flatten.",
        structural_parameters=[
            parameter("mean_lookback", 20, "1H_bars", "published_rule", "Published SMA period."),
            parameter("standard_deviation_ddof", 0, "degrees_of_freedom", "research_assumption", "Population standard deviation convention."),
            parameter("entry_z_abs", 2.0, "z_score", "published_rule", "Published entry threshold."),
            parameter("entry_mode", "threshold_cross", "rule", "research_assumption", "Prevents repeated signals while already extreme."),
        ],
        risk_parameters=[
            parameter("target_z_abs", 0.5, "z_score", "published_rule", "Published reversion exit."),
            parameter("stop_z_abs", 3.0, "z_score", "published_rule", "Published invalidation."),
            parameter("partial_exit_z_abs", "disabled_one_contract", "policy", "research_assumption", "Published +/-1 partial is disabled."),
        ],
        lookahead_risks=[risk_control("A centered rolling mean or future-normalized sigma would leak future observations.", "Use trailing close observations ending at the completed signal bar only.")],
        execution_ambiguities=[risk_control("The standard-deviation convention and repeated-signal behavior are unspecified.", "Use population sigma and threshold crossings only.")],
        remaining_assumptions=[
            assumption("zscore_population_sigma", "Use rolling population standard deviation with ddof=0."),
            assumption("zscore_cross_entries", "Open only on a fresh crossing of +/-2, not on every bar beyond the threshold."),
        ],
    ))

    specs.append(strategy_spec(
        mechanism_id="regression_equilibrium_reversion",
        strategy_id="regression-to-mean",
        research_family="mean_reversion",
        complexity_grade="C",
        timeframe="1H",
        hypothesis="Price deviations beyond two residual standard deviations from a local linear trend revert to the fitted equilibrium while the trend direction remains intact.",
        economic_rationale="A local trend can persist while temporary liquidity shocks create residual excursions around that trend, allowing reversion without requiring a flat market.",
        long_direction="Buy when close crosses below the regression endpoint minus 2 residual sigma and fitted slope is positive.",
        short_direction="Sell when close crosses above the regression endpoint plus 2 residual sigma and fitted slope is negative.",
        calculation="Fit ordinary least squares close = intercept + slope*x on the last 20 completed RTH one-hour bars with x=0..19. The current equilibrium is the fitted value at x=19; sigma is the population standard deviation of in-window residuals.",
        signal_timestamp="Close of the completed one-hour deviation bar.",
        entry_condition="A fresh crossing beyond the corresponding 2-sigma residual band while the fitted slope has the required sign.",
        entry_price="Next one-minute bar open, with adverse rounding and one tick slippage.",
        execution_delay="One one-minute execution bar after signal close.",
        target="Exit after a completed one-hour close reaches or crosses the contemporaneous fitted regression endpoint; execute next one-minute open.",
        stop="Exit after a completed one-hour close reaches or crosses the contemporaneous 3-sigma residual band against the position.",
        time_exit="Forced session flatten.",
        structural_parameters=[
            parameter("regression_lookback", 20, "1H_bars", "research_assumption", "Card omits the regression window; 20 matches its cited rolling-band context without tuning."),
            parameter("regression_method", "ordinary_least_squares_on_close", "formula", "research_assumption", "Explicit causal regression definition."),
            parameter("residual_standard_deviation_ddof", 0, "degrees_of_freedom", "research_assumption", "Population residual sigma."),
            parameter("entry_residual_sigma", 2.0, "standard_deviations", "published_rule", "Published entry deviation."),
            parameter("slope_filter", "positive_long_negative_short", "rule", "published_rule", "Published trend-direction filter."),
            parameter("r_squared_filter", "disabled_not_specified", "policy", "research_assumption", "R-squared is cited but no threshold appears in the entry rule."),
        ],
        risk_parameters=[
            parameter("target", "current_regression_endpoint", "dynamic_price", "published_rule", "Published regression-line target."),
            parameter("stop_residual_sigma", 3.0, "standard_deviations", "published_rule", "Published stop."),
            parameter("partial_exit", "disabled_one_contract", "policy", "research_assumption", "Published 1-sigma partial is disabled."),
        ],
        lookahead_risks=[risk_control("A centered regression or a line projected with future closes would leak information.", "Fit only the trailing 20 completed bars and evaluate the fitted endpoint at the current completed bar.")],
        execution_ambiguities=[risk_control("Regression length, residual sigma convention, R-squared use, and target updating are unspecified.", "Freeze OLS(20), population residual sigma, no R-squared filter, and update exit levels only after completed one-hour bars.")],
        remaining_assumptions=[
            assumption("regression_window", "Use a 20-bar trailing OLS on close."),
            assumption("regression_dynamic_exit", "Recompute the regression line and residual bands only at completed one-hour closes while a trade is open."),
        ],
    ))

    specs.append(strategy_spec(
        mechanism_id="volatility_breakout",
        strategy_id="volatility-breakout",
        research_family="breakout_expansion",
        complexity_grade="B",
        timeframe="5min",
        hypothesis="A majority vote across a compact region of prior-daily-ATR breakout thresholds identifies directional volatility expansion more robustly than one isolated parameter point.",
        economic_rationale="A move large relative to recent daily range can indicate a trend-day information shock rather than ordinary opening noise.",
        long_direction="Target long when the mean state of the 42 frozen ATR-breakout cells is at least +0.50.",
        short_direction="Target short when the mean state of the 42 frozen ATR-breakout cells is at most -0.50.",
        calculation="At 09:30, freeze the RTH session open and the prior completed Wilder daily ATR for periods 10, 12, 14, 16, 18, 20 and 22. Run the Cartesian product with entry multiples 0.5/0.6 and time stops 270/300/360 minutes, then average the 42 virtual position states.",
        signal_timestamp="Entry levels are known immediately after the 09:30 open; the triggering one-minute bar is recorded when its range first reaches a level.",
        entry_condition="First touched threshold wins. If both thresholds trade within the same one-minute bar, cancel the session because order is unknown.",
        entry_price="Threshold price, or adverse one-minute open when it gaps beyond the threshold, plus one tick adverse slippage.",
        execution_delay="Each virtual cell uses its causal resting-stop execution. Aggregate target changes are executed at the next available one-minute open.",
        target="No fixed target. Each virtual cell exits at its session-open stop or frozen 270/300/360-minute time stop.",
        stop="Fixed RTH session open price.",
        time_exit="Each virtual cell uses its frozen 270, 300 or 360-minute maximum holding time. The aggregate target follows the 42-cell vote and is flat no later than the 16:08 Topstep hard flatten.",
        structural_parameters=[
            parameter("daily_atr_period", 14, "RTH_sessions", "research_assumption", "Card cites previous-day ATR but not its period."),
            parameter("daily_atr_method", "wilder_rma", "formula", "research_assumption", "Explicit ATR convention."),
            parameter("entry_atr_multiple", 0.6, "prior_daily_ATR", "published_rule", "Published threshold from session open."),
            parameter("entry_order_type", "resting_stop", "order_type", "research_assumption", "Makes the published price-move entry executable without close look-ahead."),
        ],
        risk_parameters=[
            parameter("stop", "session_open", "price", "published_rule", "Published invalidation."),
            parameter("optional_atr_trail", "disabled", "policy", "research_assumption", "Card offers close exit or 1-ATR trail; canonical baseline uses the simpler close exit."),
        ],
        lookahead_risks=[risk_control("Using current-session daily ATR would include future range.", "Freeze every Wilder ATR(10,12,14,16,18,20,22) from the last completed RTH session only.")],
        same_bar_risks=[
            risk_control("Both long and short thresholds may trade in the same minute.", "Cancel the session without a trade."),
            risk_control("An entry threshold and the session-open stop may trade in the same minute.", "Assume entry followed by stop, the conservative path."),
        ],
        execution_ambiguities=[risk_control("Previous-day ATR can mean true range or a smoothed ATR, and the source offers two exits.", "Use prior completed Wilder RTH daily ATR for every frozen cell and the cell-specific time stop.")],
        remaining_assumptions=[assumption("volatility_breakout_atr", "Previous-day ATR means the prior completed Wilder RTH daily ATR value for each frozen period.")],
    ))
    specs[-1]["source"].update({
        "performance_information_used": True,
        "parameter_optimization_used": True,
        "selection_note": "The original single-point baseline was specified without optimization; the 42-cell ensemble region was frozen on 2026-10-29 after review of the historical IS and reused OOS/validation cubes.",
    })
    specs[-1]["ensemble_configuration"] = {
        "configuration_type": "parameter_vote_ensemble",
        "configuration_status": "frozen_after_historical_is_oos_review",
        "manifest_path": "frozen_ensembles/volatility_breakout_42.json",
        "ensemble_id": "mnq_volatility_breakout_atr10_22_mult05_06_duration270_360",
        "daily_atr_period": [10, 12, 14, 16, 18, 20, 22],
        "entry_atr_multiple": [0.5, 0.6],
        "max_holding_minutes": [270, 300, 360],
        "submodel_count": 42,
        "cell_state_space": [-1, 0, 1],
        "vote_score": "arithmetic_mean_of_42_cell_position_states",
        "vote_threshold": 0.5,
        "minimum_net_directional_votes": 21,
        "aggregate_quantity_mnq": 1,
        "aggregate_target_execution": "next_one_minute_open",
        "supersedes_single_point": {
            "daily_atr_period": 14,
            "entry_atr_multiple": 0.6,
            "max_holding_minutes": 240,
        },
    }

    specs.append(strategy_spec(
        mechanism_id="short_horizon_pullback_reversion",
        strategy_id="rsi-2-pullback",
        research_family="mean_reversion",
        complexity_grade="C",
        timeframe="15min",
        hypothesis="An extreme two-period RSI move against a 200-SMA trend is a short-lived pullback that reverts in the trend direction.",
        economic_rationale="Very short-horizon selling or buying pressure can become exhausted while the slower trend remains intact.",
        long_direction="Buy when RSI(2) crosses below 10 while close is above SMA 200.",
        short_direction="Sell when RSI(2) crosses above 90 while close is below SMA 200.",
        calculation="Compute Wilder RSI(2) and close-based SMA(200) on completed RTH 15-minute bars. Require a fresh RSI threshold crossing and a fully initialized SMA.",
        signal_timestamp="Close of the completed 15-minute threshold-crossing bar.",
        entry_condition="Long: prior RSI >= 10, current RSI < 10, close > SMA200. Short is symmetric at 90 and below SMA200.",
        entry_price="Next one-minute bar open, with adverse rounding and one tick slippage.",
        execution_delay="One one-minute execution bar after signal close.",
        target="Indicator exit: long when completed-bar RSI(2) crosses above 65; short when it crosses below 35.",
        stop="No hard price stop is published; session flatten is the terminal risk limit.",
        time_exit="No four-bar forced exit: the card describes 1–4 bars as typical, not as a rule. Forced session flatten applies.",
        structural_parameters=[
            parameter("rsi_period", 2, "15min_bars", "published_rule", "Published RSI length."),
            parameter("rsi_method", "wilder_rma", "formula", "research_assumption", "Explicit RSI smoothing convention."),
            parameter("long_entry_rsi", 10.0, "RSI", "published_rule", "Published long threshold."),
            parameter("short_entry_rsi", 90.0, "RSI", "published_rule", "Published short threshold."),
            parameter("trend_sma_period", 200, "15min_bars", "published_rule", "Published trend filter."),
            parameter("entry_mode", "threshold_cross", "rule", "research_assumption", "Avoid repeated entry signals at a persistent extreme."),
        ],
        risk_parameters=[
            parameter("long_exit_rsi", 65.0, "RSI", "published_rule", "Published long exit."),
            parameter("short_exit_rsi", 35.0, "RSI", "published_rule", "Published short exit."),
            parameter("hard_price_stop", None, "price", "published_rule", "No hard stop is published."),
        ],
        execution_ambiguities=[risk_control("RSI smoothing and whether an extreme creates one or repeated entries are unspecified.", "Use Wilder RSI and fresh threshold crossings only.")],
        remaining_assumptions=[
            assumption("rsi2_wilder", "Use Wilder's RSI calculation."),
            assumption("rsi2_cross_entries", "Only fresh crossings of 10 or 90 create entries."),
        ],
    ))

    specs.append(strategy_spec(
        mechanism_id="momentum_confirmed_breakout",
        strategy_id="momentum-breakout",
        research_family="breakout_expansion",
        complexity_grade="C",
        timeframe="1H",
        hypothesis="A 20-bar range escape accompanied by directional RSI and strong ADX is more likely to continue than an unconfirmed breakout.",
        economic_rationale="Price discovery accompanied by momentum and directional strength can reflect broad participation rather than a transient stop run.",
        long_direction="Buy a close above the prior 20-bar high when RSI(14) > 60 and ADX(14) > 25.",
        short_direction="Sell a close below the prior 20-bar low when RSI(14) < 40 and ADX(14) > 25.",
        calculation="On completed RTH one-hour bars, shift the rolling 20-bar high/low by one bar and compute Wilder RSI(14), DMI/ADX(14), and Wilder ATR(14).",
        signal_timestamp="Close of the completed one-hour confirmed breakout bar.",
        entry_condition="Close beyond the prior 20-bar extreme with the corresponding RSI threshold and ADX strictly above 25.",
        entry_price="Next one-minute bar open, with adverse rounding and one tick slippage.",
        execution_delay="One one-minute execution bar after signal close.",
        target="No fixed target.",
        stop="Initial stop two ATR(14) from entry. On each completed one-hour bar, trail long stop to max(previous stop, highest completed high since entry - 2*current ATR); short is symmetric. Never loosen.",
        time_exit="Exit after 20 completed one-hour bars if still open, or at the 16:08 hard flatten each session; no position carries overnight.",
        structural_parameters=[
            parameter("breakout_lookback", 20, "1H_bars", "published_rule", "Published price breakout length."),
            parameter("rsi_period", 14, "1H_bars", "published_rule", "Published RSI period."),
            parameter("long_rsi_threshold", 60.0, "RSI", "published_rule", "Published long confirmation."),
            parameter("short_rsi_threshold", 40.0, "RSI", "published_rule", "Published short confirmation."),
            parameter("adx_period", 14, "1H_bars", "published_rule", "Published ADX period."),
            parameter("adx_threshold", 25.0, "ADX", "published_rule", "Published strength threshold."),
            parameter("atr_period", 14, "1H_bars", "research_assumption", "ATR period absent from the exit rule."),
            parameter("breakout_confirmation", "completed_close", "rule", "research_assumption", "Avoids intrabar breakout look-ahead."),
        ],
        risk_parameters=[
            parameter("trailing_stop_atr_multiple", 2.0, "ATR", "published_rule", "Published trail distance."),
            parameter("time_stop_bars", 20, "1H_bars", "published_rule", "Published time stop."),
            parameter("rsi_divergence_exit", "disabled_undefined", "policy", "research_assumption", "Divergence has no causal definition in the card."),
        ],
        lookahead_risks=[
            risk_control("The current bar must not be included in the 20-bar breakout reference.", "Shift the rolling high/low by one completed bar."),
            risk_control("RSI divergence can be detected retrospectively with future pivots.", "Do not use the undefined divergence exit in the canonical baseline."),
        ],
        execution_ambiguities=[risk_control("The trail anchor, ATR period, and RSI-divergence exit are not defined.", "Use ATR(14), highest/lowest completed high/low since entry, and omit divergence.")],
        remaining_assumptions=[
            assumption("momentum_breakout_close", "Require a completed close beyond the shifted 20-bar range."),
            assumption("momentum_trail_definition", "Trail from completed extremes using current Wilder ATR(14), never loosening the stop."),
        ],
    ))

    specs.append(strategy_spec(
        mechanism_id="gap_fade",
        strategy_id="gap-fill",
        research_family="session_gap",
        complexity_grade="B",
        timeframe="5min",
        hypothesis="A small RTH opening gap relative to recent daily volatility tends to close during the same session.",
        economic_rationale="Small overnight imbalances may reflect thin liquidity rather than durable information and can be arbitraged back toward the prior RTH close.",
        long_direction="If current RTH open is below prior RTH close and absolute gap is between one tick and one prior daily ATR, buy toward prior close.",
        short_direction="If current RTH open is above prior RTH close and absolute gap is between one tick and one prior daily ATR, sell toward prior close.",
        calculation="Prior close is the last one-minute close before 16:00 of the previous valid RTH session. Current open is the 09:30 one-minute open. Prior ATR is the previous completed value of Wilder RTH daily ATR(14).",
        signal_timestamp="09:30 session open becomes observable at the start of the 09:30 one-minute bar.",
        entry_condition="One-tick minimum gap and absolute gap < prior ATR(14). Direction is opposite the gap.",
        entry_price="09:31 one-minute open, with adverse rounding and one tick slippage; this delay avoids assuming a fill at an open whose gap was not known beforehand.",
        execution_delay="One full one-minute observation after the 09:30 open.",
        target="Previous valid RTH session close.",
        stop="Fifty percent of the gap size beyond the current session open, away from the target.",
        time_exit="Open of the 12:00 one-minute bar, defining the published 'midday' exit; the 16:08 hard flatten is a safety fallback only.",
        structural_parameters=[
            parameter("gap_reference", "current_rth_open_minus_prior_rth_close", "formula", "published_rule", "Published overnight gap definition made session-explicit."),
            parameter("daily_atr_period", 14, "RTH_sessions", "research_assumption", "ATR period absent from the card."),
            parameter("daily_atr_method", "wilder_rma", "formula", "research_assumption", "Explicit ATR convention."),
            parameter("maximum_gap_atr_multiple", 1.0, "prior_daily_ATR", "published_rule", "Published small-gap ceiling."),
            parameter("minimum_gap_ticks", 1, "ticks", "research_assumption", "Avoid zero-gap trades."),
        ],
        risk_parameters=[
            parameter("target", "prior_rth_close", "price", "published_rule", "Published full-fill target."),
            parameter("stop_gap_extension", 0.5, "gap_size", "published_rule", "Published extension stop."),
            parameter("midday_time_exit", "12:00:00", "local_time", "research_assumption", "Mechanical definition of midday."),
        ],
        lookahead_risks=[risk_control("Using the current session range in ATR or a later opening print would leak information.", "Use only prior completed daily ATR and the first 09:30 open.")],
        same_bar_risks=[risk_control("The prior close target or extension stop may trade during the 09:30 minute before entry.", "Entry is delayed to 09:31; pre-entry touches do not count and levels must remain valid at entry.")],
        execution_ambiguities=[risk_control("The source does not define ATR period, minimum gap, exact open fill, or 'midday'.", "Use prior Wilder ATR(14), one-tick minimum, 09:31 entry, and 12:00 exit.")],
        remaining_assumptions=[
            assumption("gap_fade_entry_delay", "Observe the 09:30 open and enter at 09:31 rather than claiming the same opening print."),
            assumption("gap_fade_midday", "Midday means 12:00 America/New_York."),
        ],
        last_entry_time="09:31:00",
    ))

    specs.append(strategy_spec(
        mechanism_id="gap_continuation",
        strategy_id="gap-and-go",
        research_family="session_gap",
        complexity_grade="C",
        timeframe="15min",
        hypothesis="A large opening gap confirmed by a directional first 15-minute candle and elevated volume continues in the gap direction.",
        economic_rationale="A large, well-participated gap can encode durable new information that attracts trend-following flow rather than immediate liquidity-driven reversion.",
        long_direction="For a gap up greater than prior ATR, buy when the 09:30–09:45 candle closes above its open and its volume is above average.",
        short_direction="For a gap down greater than prior ATR, sell when the 09:30–09:45 candle closes below its open and its volume is above average.",
        calculation="Compute gap from current RTH open to prior RTH close and prior Wilder daily ATR(14). The first 15-minute bar is 09:30–09:45. Average volume is the mean of the previous 20 completed RTH 15-minute bars.",
        signal_timestamp="09:45, after the first 15-minute candle is complete.",
        entry_condition="Absolute gap > prior ATR, first candle direction agrees with gap, and first-candle volume > trailing 20 completed 15-minute-bar mean.",
        entry_price="09:45 one-minute open immediately after the first 15-minute interval, with adverse rounding and one tick slippage.",
        execution_delay="One minute-bar open after the 15-minute confirmation interval; no fill at the confirmation close.",
        target="Entry price plus two absolute gap sizes for long, or minus two gap sizes for short.",
        stop="Gap midpoint: halfway between prior RTH close and current RTH open, implementing 50% gap re-entry.",
        time_exit="Topstep hard flatten at 16:08 if neither target nor stop has exited the position.",
        structural_parameters=[
            parameter("gap_reference", "current_rth_open_minus_prior_rth_close", "formula", "published_rule", "Published gap direction and size."),
            parameter("daily_atr_period", 14, "RTH_sessions", "research_assumption", "ATR period absent from the card."),
            parameter("minimum_gap_atr_multiple", 1.0, "prior_daily_ATR", "published_rule", "Published large-gap threshold."),
            parameter("confirmation_bar_minutes", 15, "minutes", "published_rule", "Published first-candle confirmation."),
            parameter("volume_average_lookback", 20, "15min_bars", "research_assumption", "Mechanical definition of above-average volume."),
            parameter("volume_threshold_multiple", 1.0, "trailing_mean_volume", "published_rule_plus_research_assumption", "Above-average means strictly greater than trailing mean."),
        ],
        risk_parameters=[
            parameter("target_gap_multiple", 2.0, "gap_size", "published_rule", "Published measured move."),
            parameter("stop_gap_reentry_fraction", 0.5, "gap_size", "published_rule", "Published 50% re-entry stop."),
            parameter("swing_trail", "disabled_undefined", "policy", "research_assumption", "Swing identification is not defined and is omitted."),
        ],
        lookahead_risks=[risk_control("Using final first-bar volume or direction before 09:45 would leak the confirmation.", "Do not signal or fill until the first 15-minute bar is complete.")],
        execution_ambiguities=[risk_control("Average volume, measured-move anchor, and swing trail are unspecified.", "Use trailing 20-bar mean, project two gap sizes from entry, and omit the swing trail.")],
        remaining_assumptions=[
            assumption("gap_go_volume_average", "Above-average volume means first-bar volume exceeds the prior 20 completed 15-minute bars' mean."),
            assumption("gap_go_target_anchor", "Project the two-gap target from actual entry price."),
        ],
        last_entry_time="09:45:00",
    ))

    specs.append(strategy_spec(
        mechanism_id="range_structure_reversion",
        strategy_id="range-midpoint-reversion",
        research_family="mean_reversion",
        complexity_grade="B",
        timeframe="15min",
        hypothesis="Price entering an outer quartile of an established 20-bar range with an RSI extreme reverts toward the range midpoint.",
        economic_rationale="Balanced auction conditions can attract liquidity near established boundaries and pull price back toward the center of accepted value.",
        long_direction="Buy a close inside the bottom quartile of the prior 20-bar range when RSI(14) <= 30.",
        short_direction="Sell a close inside the top quartile of the prior 20-bar range when RSI(14) >= 70.",
        calculation="From the previous 20 completed RTH 15-minute bars, compute high, low, width, midpoint, lower quartile, and upper quartile. Exclude the current signal bar. Compute Wilder RSI(14).",
        signal_timestamp="Close of the completed 15-minute bar that first enters an outer quartile with the RSI extreme.",
        entry_condition="Long close is between prior range low and lower quartile and prior close was above the lower quartile; short is symmetric. Reject closes already outside the prior range.",
        entry_price="Next one-minute bar open, with adverse rounding and one tick slippage; skip if it is beyond stop or target.",
        execution_delay="One one-minute execution bar after signal close.",
        target="Fixed midpoint of the prior 20-bar range observed at signal time.",
        stop="One MNQ tick beyond the corresponding prior range boundary.",
        time_exit="Forced session flatten.",
        structural_parameters=[
            parameter("range_lookback", 20, "15min_bars", "published_rule", "Published range length."),
            parameter("entry_quartile_fraction", 0.25, "range_width", "published_rule", "Published outer 25% entry zone."),
            parameter("rsi_period", 14, "15min_bars", "research_assumption", "RSI period not specified."),
            parameter("long_rsi_threshold", 30.0, "RSI", "research_assumption", "Mechanical oversold definition."),
            parameter("short_rsi_threshold", 70.0, "RSI", "research_assumption", "Mechanical overbought definition."),
            parameter("entry_mode", "fresh_quartile_entry", "rule", "research_assumption", "Avoid repeated signals within the same zone."),
        ],
        risk_parameters=[
            parameter("target", "signal_range_midpoint", "price", "published_rule", "Published midpoint target."),
            parameter("stop_buffer_ticks", 1, "ticks_beyond_range", "published_rule", "Published one-tick invalidation."),
            parameter("midpoint_trail", "not_applicable_after_full_exit", "policy", "research_assumption", "One-contract baseline exits fully at midpoint."),
        ],
        lookahead_risks=[risk_control("Including the signal bar in its own range changes boundaries after the move.", "Compute and freeze the 20-bar range from bars ending immediately before the signal bar.")],
        execution_ambiguities=[risk_control("RSI extreme and repeated occupancy in the outer quartile are undefined.", "Use Wilder RSI(14), 30/70, and require a fresh zone entry.")],
        remaining_assumptions=[
            assumption("range_midpoint_rsi", "RSI extreme means Wilder RSI(14) <= 30 for long and >= 70 for short."),
            assumption("range_midpoint_fresh_entry", "Only the first close entering an outer quartile creates a signal."),
        ],
    ))

    specs.append(strategy_spec(
        mechanism_id="channel_extreme_reversion",
        strategy_id="keltner-mean-reversion",
        research_family="mean_reversion",
        complexity_grade="C",
        timeframe="15min",
        hypothesis="A volatility-adjusted Keltner channel extreme accompanied by an RSI extreme reverts toward EMA 20.",
        economic_rationale="Moves beyond a volatility-scaled envelope can represent temporary exhaustion; the EMA midline proxies the local equilibrium.",
        long_direction="Buy after a completed bar touches the lower Keltner band and RSI(14) < 35.",
        short_direction="Sell after a completed bar touches the upper Keltner band and RSI(14) > 65.",
        calculation="On completed RTH 15-minute bars, compute EMA(20), Wilder ATR(20), lower band EMA-2*ATR, upper band EMA+2*ATR, and Wilder RSI(14). Long touch uses low <= lower band; short uses high >= upper band.",
        signal_timestamp="Close of the completed 15-minute touch-and-RSI bar.",
        entry_condition="First qualifying band touch while flat; require next execution price to remain strictly between fixed stop and fixed target.",
        entry_price="Next one-minute bar open, with adverse rounding and one tick slippage.",
        execution_delay="One one-minute execution bar after signal close.",
        target="EMA 20 value frozen from the signal bar.",
        stop="Long: signal lower band minus one signal ATR. Short: signal upper band plus one signal ATR.",
        time_exit="Forced session flatten.",
        structural_parameters=[
            parameter("ema_period", 20, "15min_bars", "published_rule", "Published channel midline."),
            parameter("atr_period", 20, "15min_bars", "research_assumption", "ATR period absent; matched to EMA length without tuning."),
            parameter("keltner_atr_multiplier", 2.0, "ATR", "research_assumption", "Channel width absent from the card."),
            parameter("rsi_period", 14, "15min_bars", "research_assumption", "RSI period absent from the card."),
            parameter("long_rsi_threshold", 35.0, "RSI", "published_rule", "Published long filter."),
            parameter("short_rsi_threshold", 65.0, "RSI", "published_rule", "Published short filter."),
            parameter("band_touch_source", "bar_low_long_bar_high_short", "rule", "research_assumption", "Mechanical touch definition."),
        ],
        risk_parameters=[
            parameter("target", "signal_ema20", "price", "published_rule", "Published midline target, frozen at signal."),
            parameter("stop_atr_beyond_channel", 1.0, "ATR", "published_rule", "Published stop distance."),
            parameter("partial_exit", "disabled_one_contract", "policy", "research_assumption", "Published half-distance partial is disabled."),
        ],
        lookahead_risks=[risk_control("A Keltner band using current unfinished high/low or ATR can repaint during the bar.", "Evaluate touch and indicators only after the 15-minute bar closes; enter next minute.")],
        same_bar_risks=[risk_control("The signal bar can touch both band and stop extension before entry.", "No signal-bar fill is allowed; fixed levels are validated against the next open before entry.")],
        execution_ambiguities=[risk_control("ATR length, band multiplier, RSI length, touch price, and dynamic-versus-fixed exits are unspecified.", "Freeze ATR(20), 2 ATR bands, RSI(14), high/low touch, and fixed signal-bar target/stop.")],
        remaining_assumptions=[
            assumption("keltner_definition", "Use EMA(20) +/- 2 times Wilder ATR(20)."),
            assumption("keltner_touch", "A long touch is bar low <= lower band; a short touch is bar high >= upper band."),
            assumption("keltner_fixed_exit_levels", "Freeze EMA target and outer stop from the signal bar."),
        ],
    ))

    return specs


def build_benchmark_spec() -> dict[str, Any]:
    parameters = {
        "structural": [
            parameter("price_column", "close", "field", "benchmark_convention", "Use the MNQ continuous-series close."),
            parameter("sampling", "last_close_per_session", "frequency", "repository_benchmark_convention", "Existing repository benchmark convention."),
            parameter("cash_close", CASH_CLOSE, "local_time", "benchmark_convention", "Regular-session close used to sample the close-to-close market curve."),
            parameter("early_close_calendar", "official_CME_equity_index_futures_calendar", "calendar", "research_assumption", "On shortened sessions, sample the official CME early close."),
            parameter("exposure", 1.0, "long_fraction", "benchmark_convention", "Fully invested long market exposure."),
            parameter("initial_capital_usd", 50000.0, "USD", "repository_execution_default", "Normalization base for the benchmark equity curve."),
            parameter("equity_formula", "initial_capital * close_t / close_0", "formula", "repository_benchmark_convention", "Price-relative market benchmark."),
        ],
        "risk_management": [],
        "execution": [
            parameter("transaction_costs", "not_applied_price_index_benchmark", "policy", "benchmark_convention", "Benchmark measures market price performance rather than a futures trading implementation."),
            parameter("roll_handling", "use_provided_continuous_series_unchanged", "policy", "research_assumption", "No synthetic roll reconstruction is introduced."),
            parameter("session_timezone", TIMEZONE, "IANA_timezone", "repository_session_default", "Session closes are identified in New York time."),
            parameter("daily_flatten", False, "boolean", "benchmark_convention", "Keep continuous close-to-close exposure; do not flatten each day."),
            parameter("topstep_rules_applied", False, "boolean", "benchmark_convention", "The market benchmark is independent of Topstep entry and flatten rules."),
        ],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "specification_type": "market_benchmark",
        "specification_status": "frozen_benchmark_ready",
        "mechanism_id": "MNQ_BUY_AND_HOLD",
        "strategy_id": "MNQ_BUY_AND_HOLD",
        "research_family": "market_benchmark",
        "complexity_grade": "N/A",
        "source": {
            "card_path": None,
            "repository_reference": "src/analytics/volume_climax_pullback_notebook_utils.py::build_buy_hold_benchmark_curve",
            "performance_information_used": False,
            "parameter_optimization_used": False,
        },
        "hypothesis": {
            "statement": "Continuous long exposure to MNQ represents the passive market return over the fixed evaluation window.",
            "economic_rationale": "The benchmark measures broad Nasdaq futures price appreciation and drawdown independently of any active strategy signal.",
        },
        "direction": {
            "long": "Continuous normalized long exposure of 1.0 from the first eligible session close.",
            "short": "Never short.",
        },
        "signal": {
            "timeframe": "1_session",
            "calculation": "Take the last valid MNQ close at the 16:00 cash close, or at the official CME early close on a shortened session, and compute equity_t = initial_capital * close_t / first_close.",
            "signal_timestamp": "Each applicable session close after that close is observed.",
        },
        "entry": {
            "condition": "First valid session close in the fixed evaluation window.",
            "execution_price": "First valid session close, used only as the normalization denominator.",
            "execution_delay": "None; this is a price-index benchmark, not an executable strategy fill model.",
        },
        "exit": {
            "target": None,
            "stop": None,
            "time_exit": "Last valid session close of the fixed evaluation window.",
            "session_flatten": {
                "enabled": False,
                "daily_flatten": False,
                "conditional_session_flatten": False,
                "topstep_rules_applied": False,
                "cash_close": CASH_CLOSE,
                "cash_close_action": "Sample the close for the continuous close-to-close curve; do not liquidate or reset exposure.",
                "last_entry_time": None,
                "forced_flatten_time": None,
                "no_position_after_forced_flatten": None,
                "timezone": TIMEZONE,
                "early_close_exception": {
                    "calendar": "official_CME_equity_index_futures_calendar",
                    "cash_close": "official early-close timestamp",
                    "effect": "Sample the shortened session's official close without flattening or resetting benchmark exposure.",
                    "missing_calendar_policy": "reject_session",
                },
                "reason": "MNQ_BUY_AND_HOLD is a continuous close-to-close market benchmark independent of Topstep rules.",
            },
        },
        "data_requirements": {
            "instrument": "MNQ",
            "contract_series": "provided_continuous_front_contract",
            "source_granularity": "1min_or_session_close",
            "signal_timeframe": "1_session",
            "execution_timeframe": None,
            "required_fields": ["timestamp", "close", "session_date"],
            "timezone": TIMEZONE,
            "session": {
                "cash_close": CASH_CLOSE,
                "early_close_calendar_required": True,
                "daily_flatten": False,
                "last_entry_time": None,
                "forced_flatten_time": None,
            },
            "availability": True,
            "notes": [
                "The continuous-series construction and any back adjustment must remain fixed across all future reports.",
                "The official CME equity-index-futures calendar supplies shortened-session close timestamps; do not infer them from missing bars.",
                "Topstep entry cutoffs and hard-flatten rules never alter this benchmark curve.",
            ],
        },
        "parameters": parameters,
        "canonical_parameters": canonical_parameter_map(parameters),
        "lookahead_risks": [
            risk_control("Selecting the first or last date after inspecting strategy performance changes the benchmark window.", "Use exactly the same predeclared calendar window as the strategy report being compared."),
            risk_control("A continuous futures series may contain roll adjustment artifacts.", "Use the same frozen continuous MNQ series for every comparison and disclose its roll construction when available."),
        ],
        "same_bar_risks": [],
        "execution_ambiguities": [
            risk_control("A normalized continuous-futures price curve is not a directly tradable cash buy-and-hold position.", "Label it as a market price benchmark and do not apply active-strategy commissions, slippage, or session flattening."),
        ],
        "remaining_assumptions": [
            assumption("continuous_series_as_market_proxy", "The provided continuous MNQ close series is accepted unchanged as the passive market proxy."),
            assumption("benchmark_window_lock", "The comparison window is fixed before active-strategy metrics are computed."),
        ],
        "unresolved_ambiguities": [],
        "ready_for_baseline_backtest": True,
    }


REQUIRED_SPEC_KEYS = {
    "mechanism_id",
    "strategy_id",
    "research_family",
    "complexity_grade",
    "hypothesis",
    "direction",
    "signal",
    "entry",
    "exit",
    "data_requirements",
    "parameters",
    "canonical_parameters",
    "lookahead_risks",
    "same_bar_risks",
    "execution_ambiguities",
    "remaining_assumptions",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_specs(strategy_specs: list[dict[str, Any]], benchmark: dict[str, Any]) -> None:
    audit_rows = read_csv(AUDIT_PATH)
    expected_representatives = {
        row["slug"]: row
        for row in audit_rows
        if row["logical_status"] == "alpha_representative"
    }
    mechanism_rows = read_csv(MECHANISMS_PATH)
    expected_mechanisms = {
        row["mechanism_cluster"]: row["representative_slug"]
        for row in mechanism_rows
    }

    if len(strategy_specs) != 16:
        raise ValueError(f"Expected 16 strategy specifications, got {len(strategy_specs)}")
    if len(expected_representatives) != 16 or len(expected_mechanisms) != 16:
        raise ValueError("The logical audit must expose exactly 16 representatives and mechanisms")

    ids = [spec["mechanism_id"] for spec in strategy_specs]
    strategy_ids = [spec["strategy_id"] for spec in strategy_specs]
    if len(set(ids)) != 16 or len(set(strategy_ids)) != 16:
        raise ValueError("Mechanism IDs and representative strategy IDs must be unique")
    if set(strategy_ids) != set(expected_representatives):
        raise ValueError(
            f"Representative mismatch; missing={sorted(set(expected_representatives) - set(strategy_ids))}, "
            f"unexpected={sorted(set(strategy_ids) - set(expected_representatives))}"
        )
    actual_mapping = {spec["mechanism_id"]: spec["strategy_id"] for spec in strategy_specs}
    if actual_mapping != expected_mechanisms:
        raise ValueError("Mechanism-to-representative mapping differs from alpha_engine_mechanisms.csv")

    forbidden_key_fragments = ("sharpe", "profit_factor", "ranking", "observed_performance")
    for spec in [*strategy_specs, benchmark]:
        missing = REQUIRED_SPEC_KEYS - set(spec)
        if missing:
            raise ValueError(f"{spec.get('mechanism_id')} missing fields: {sorted(missing)}")
        serialized_keys = json.dumps(list(spec.keys())).lower()
        if any(fragment in serialized_keys for fragment in forbidden_key_fragments):
            raise ValueError(f"Performance field found in {spec['mechanism_id']}")
        for category in ("structural", "risk_management", "execution"):
            items = spec["parameters"][category]
            names = [item["name"] for item in items]
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate {category} parameter in {spec['mechanism_id']}")
        if spec["canonical_parameters"] != canonical_parameter_map(spec["parameters"]):
            raise ValueError(f"Canonical parameter map mismatch in {spec['mechanism_id']}")
        if spec["unresolved_ambiguities"]:
            raise ValueError(f"Frozen executable spec still has unresolved ambiguities: {spec['mechanism_id']}")
        if not spec["ready_for_baseline_backtest"]:
            raise ValueError(f"Frozen spec is not marked ready: {spec['mechanism_id']}")

    for spec in strategy_specs:
        audit = expected_representatives[spec["strategy_id"]]
        if spec["research_family"] != audit["reviewed_family"]:
            raise ValueError(f"Research family mismatch in {spec['mechanism_id']}")
        if spec["complexity_grade"] != audit["complexity_grade"]:
            raise ValueError(f"Complexity mismatch in {spec['mechanism_id']}")
        session_rules = spec["exit"]["session_flatten"]
        if not session_rules["enabled"] or not session_rules["daily_flatten"]:
            raise ValueError(f"Strategy must flatten intraday: {spec['mechanism_id']}")
        if not session_rules["topstep_rules_applied"]:
            raise ValueError(f"Strategy must apply Topstep timing rules: {spec['mechanism_id']}")
        if session_rules["timezone"] != TIMEZONE:
            raise ValueError(f"Strategy timezone mismatch: {spec['mechanism_id']}")
        if session_rules["cash_close"] != CASH_CLOSE:
            raise ValueError(f"Strategy cash close mismatch: {spec['mechanism_id']}")
        if session_rules["forced_flatten_time"] != FORCED_FLATTEN_TIME:
            raise ValueError(f"Strategy hard-flatten mismatch: {spec['mechanism_id']}")
        if not session_rules["no_position_after_forced_flatten"]:
            raise ValueError(f"Strategy can retain a position after hard flatten: {spec['mechanism_id']}")
        if session_rules["last_entry_time"] != spec["entry"]["last_entry_time"]:
            raise ValueError(f"Last-entry time mismatch: {spec['mechanism_id']}")
        if session_rules["early_close_exception"]["calendar"] != "official_CME_equity_index_futures_calendar":
            raise ValueError(f"Early-close calendar missing: {spec['mechanism_id']}")
        card_path = ROOT / spec["source"]["card_path"]
        if not card_path.is_file():
            raise ValueError(f"Missing representative source card: {card_path}")

    expected_last_entry_times = {
        "gap_fade": "09:31:00",
        "gap_continuation": "09:45:00",
    }
    for spec in strategy_specs:
        expected_time = expected_last_entry_times.get(spec["mechanism_id"], DEFAULT_LAST_ENTRY_TIME)
        if spec["entry"]["last_entry_time"] != expected_time:
            raise ValueError(f"Unexpected last-entry cutoff in {spec['mechanism_id']}")

    if benchmark["mechanism_id"] != "MNQ_BUY_AND_HOLD":
        raise ValueError("Benchmark ID must be MNQ_BUY_AND_HOLD")
    if benchmark["specification_type"] != "market_benchmark":
        raise ValueError("Benchmark must remain separate from strategy baselines")
    if benchmark["exit"]["session_flatten"]["enabled"]:
        raise ValueError("Continuous market benchmark must not inherit strategy session flattening")
    benchmark_session = benchmark["exit"]["session_flatten"]
    if benchmark_session["daily_flatten"] or benchmark_session["topstep_rules_applied"]:
        raise ValueError("Continuous market benchmark must remain independent of daily Topstep rules")
    if benchmark_session["last_entry_time"] is not None or benchmark_session["forced_flatten_time"] is not None:
        raise ValueError("Continuous market benchmark must not define entry or hard-flatten cutoffs")
    if benchmark_session["cash_close"] != CASH_CLOSE or benchmark_session["timezone"] != TIMEZONE:
        raise ValueError("Benchmark close sampling convention mismatch")
    if benchmark_session["early_close_exception"]["calendar"] != "official_CME_equity_index_futures_calendar":
        raise ValueError("Benchmark early-close calendar missing")


def write_json_specs(strategy_specs: list[dict[str, Any]], benchmark: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_specs = [*strategy_specs, benchmark]
    expected_names = {
        f"{spec['mechanism_id']}.json" for spec in all_specs
    }
    existing_names = {path.name for path in OUTPUT_DIR.glob("*.json")}
    unexpected = existing_names - expected_names
    if unexpected:
        raise ValueError(f"Unexpected frozen specification files: {sorted(unexpected)}")

    for spec in all_specs:
        path = OUTPUT_DIR / f"{spec['mechanism_id']}.json"
        path.write_text(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def manifest_row(spec: dict[str, Any]) -> dict[str, Any]:
    specification_path = OUTPUT_DIR / f"{spec['mechanism_id']}.json"
    if not specification_path.is_file():
        raise ValueError(f"Specification must be written before manifest hashing: {specification_path}")
    session_rules = spec["exit"]["session_flatten"]
    return {
        "mechanism_id": spec["mechanism_id"],
        "strategy_id": spec["strategy_id"],
        "specification_type": spec["specification_type"],
        "research_family": spec["research_family"],
        "complexity_grade": spec["complexity_grade"],
        "specification_status": spec["specification_status"],
        "number_structural_parameters": len(spec["parameters"]["structural"]),
        "number_risk_parameters": len(spec["parameters"]["risk_management"]),
        "number_execution_parameters": len(spec["parameters"]["execution"]),
        "research_assumptions": len(spec["remaining_assumptions"]),
        "unresolved_ambiguities": len(spec["unresolved_ambiguities"]),
        "lookahead_risk": "identified_and_controlled" if spec["lookahead_risks"] else "none_identified",
        "data_available": spec["data_requirements"]["availability"],
        "timezone": session_rules["timezone"],
        "cash_close": session_rules["cash_close"],
        "last_entry_time": session_rules["last_entry_time"],
        "forced_flatten_time": session_rules["forced_flatten_time"],
        "daily_flatten": session_rules["daily_flatten"],
        "conditional_session_flatten": session_rules["conditional_session_flatten"],
        "topstep_rules_applied": session_rules["topstep_rules_applied"],
        "no_position_after_forced_flatten": session_rules["no_position_after_forced_flatten"],
        "early_close_calendar": session_rules["early_close_exception"]["calendar"],
        "early_close_missing_calendar_policy": session_rules["early_close_exception"]["missing_calendar_policy"],
        "ready_for_baseline_backtest": spec["ready_for_baseline_backtest"],
        "specification_path": f"frozen_baselines/{spec['mechanism_id']}.json",
        "specification_sha256": hashlib.sha256(specification_path.read_bytes()).hexdigest(),
    }


def write_manifest(strategy_specs: list[dict[str, Any]], benchmark: dict[str, Any]) -> None:
    rows = [manifest_row(spec) for spec in [*strategy_specs, benchmark]]
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_review(strategy_specs: list[dict[str, Any]], benchmark: dict[str, Any]) -> None:
    common_ids = {
        "rth_only_signal_bars",
        "frozen_signal_timeframe",
        "one_contract_baseline",
        "common_execution_model",
        "early_close_buffers",
    }
    lines = [
        "# Revue des baselines canoniques figées",
        "",
        "Cette étape transforme les 16 mécanismes représentatifs en 16 spécifications exécutables. "
        "Aucun résultat de backtest, classement, Sharpe ou Profit Factor publié n'a servi à choisir "
        "une règle ou un paramètre. Aucun backtest n'est lancé par le script de construction.",
        "",
        "Le seul benchmark préparé est `MNQ_BUY_AND_HOLD`. Il est défini séparément des stratégies "
        "et ne participe ni au choix des mécanismes ni au gel de leurs paramètres.",
        "",
        "Le détail de gel et les empreintes SHA-256 sont dans "
        "[`frozen_baselines_manifest.csv`](frozen_baselines_manifest.csv).",
        "",
        "## Les 16 mécanismes figés",
        "",
        "| Mécanisme | Représentant | Famille | Grade | Horizon | Hypothèse économique |",
        "|---|---|---|---:|---|---|",
    ]
    for spec in strategy_specs:
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(value)
                for value in (
                    f"[`{spec['mechanism_id']}`](frozen_baselines/{spec['mechanism_id']}.json)",
                    f"[`{spec['strategy_id']}`]({spec['source']['card_path']})",
                    spec["research_family"],
                    spec["complexity_grade"],
                    spec["signal"]["timeframe"],
                    spec["hypothesis"]["economic_rationale"],
                )
            )
            + " |"
        )

    lines.extend([
        "",
        "## Conventions communes ajoutées",
        "",
        "- Les indicateurs utilisent seulement des barres RTH closes, en heure `America/New_York`.",
        "- Chaque horizon est choisi dans la liste publiée pour rendre la règle mécanique, sans "
        "  comparaison de performance; ce choix est ensuite figé dans le JSON.",
        "- Les barres signal sont construites à partir des données MNQ une minute; aucune barre "
        "  incomplète ne peut produire un signal.",
        "- Un signal confirmé à la clôture est exécuté à l'ouverture exacte de la minute suivante. "
        "  Les ordres stop déjà connus avant la minute suivent leur règle de déclenchement propre.",
        "- Chaque baseline utilise un contrat MNQ, sans pyramiding ni sizing conditionnel.",
        f"- La clôture cash régulière est `{CASH_CLOSE}` en heure `{TIMEZONE}`. Elle termine la "
        "  fenêtre de nouveaux signaux, sans imposer à elle seule une sortie sauf règle spécifique.",
        f"- La limite de nouvelle entrée est `{DEFAULT_LAST_ENTRY_TIME}` par défaut. `gap_fade` "
        "  conserve sa limite spécifique à `09:31:00` et `gap_continuation` à `09:45:00`.",
        f"- Le hard flatten Topstep est `{FORCED_FLATTEN_TIME}` : aucune position de stratégie ne "
        "  peut subsister après cette heure. Entre 16:00 et 16:08, seuls les stops et cibles déjà "
        "  définis restent surveillés; aucun nouveau signal ni aucune nouvelle entrée n'est accepté.",
        "- Lors d'une clôture anticipée officielle du CME, la clôture cash devient l'heure publiée, "
        f"  la limite d'entrée est le minimum entre la règle de la stratégie et {EARLY_CLOSE_LAST_ENTRY_BUFFER_MINUTES} minutes avant cette clôture, "
        f"  et le hard flatten intervient {EARLY_CLOSE_FLATTEN_BUFFER_MINUTES} minute avant. La session est rejetée si le calendrier manque.",
        f"- Coûts figés: `{COMMISSION_PER_SIDE_USD:.2f} USD` par côté et "
        f"`{SLIPPAGE_TICKS_PER_SIDE}` tick adverse par côté; tick MNQ `{TICK_SIZE}` point.",
        "- Si stop et cible sont touchés dans la même minute, le stop est prioritaire.",
        "- Les sorties partielles et options facultatives sont désactivées avec un contrat; leur "
        "  omission est explicitement marquée comme `research_assumption`.",
        "",
        "## Hypothèses spécifiques ajoutées",
        "",
        "| Mécanisme | Conventions spécifiques |",
        "|---|---|",
    ])
    for spec in strategy_specs:
        specific = [
            item["id"]
            for item in spec["remaining_assumptions"]
            if item["id"] not in common_ids
        ]
        lines.append(f"| `{spec['mechanism_id']}` | {', '.join(f'`{item}`' for item in specific) or 'Aucune'} |")

    lines.extend([
        "",
        "## Ambiguïtés restantes et exécutabilité",
        "",
        "Les 16 fichiers ont `unresolved_ambiguities = []`. Les règles absentes des fiches ont été "
        "résolues par la convention mécanique la plus simple et restent visibles dans "
        "`remaining_assumptions`. Les 16 spécifications sont donc exécutables sans paramètre libre, "
        "mais ces conventions devront être conservées telles quelles pendant le tournoi baseline.",
        "Le manifeste contient le SHA-256 de chaque JSON pour détecter toute modification ultérieure.",
        "",
        "**Stratégies non exécutables à ce stade : aucune parmi les 16 représentantes.**",
        "",
        "## Risques méthodologiques contrôlés",
        "",
        "- **Look-ahead** : les rolling highs/lows excluent la barre testée; VWAP, ATR, RSI, ADX, "
        "  régression et bandes ne sont mis à jour qu'après clôture de leur barre.",
        "- **High/low non terminé** : les touches intrabar servent seulement aux ordres dont le "
        "  niveau était connu avant la minute; les autres entrées attendent la clôture du signal.",
        "- **Same bar** : stop avant cible; les doubles déclenchements directionnels impossibles à "
        "  ordonner annulent le setup.",
        "- **Fills impossibles** : pas de fill au close qui crée le signal; absence de la minute "
        "  d'exécution attendue = signal ignoré; les gaps sont remplis au prix ouvert défavorable.",
        "- **Données** : toutes les stratégies utilisent OHLCV, timestamp et `session_date`. Aucune "
        "  donnée tick, carnet, volume delta ou volume-at-price n'est reconstruite. Le calendrier "
        "  officiel CME est requis pour identifier les clôtures anticipées.",
        "- **Série continue** : les changements de contrat et ajustements de roll peuvent affecter "
        "  les signaux et le benchmark; la même série gelée doit être utilisée partout.",
        "",
        "## Benchmark MNQ_BUY_AND_HOLD",
        "",
        "Le benchmark reprend la convention existante du dépôt: dernier close valide de chaque "
        "session et courbe normalisée `capital_initial × close_t / close_0`. Il représente une "
        "exposition marché continue close-to-close. La clôture cash de 16:00, ou la clôture CME "
        "anticipée officielle, sert uniquement à échantillonner le close. Le benchmark n'applique "
        "ni limite d'entrée, ni daily flatten, ni hard flatten Topstep, ni coûts d'exécution des "
        "stratégies. La fenêtre de comparaison doit être fixée avant le calcul des performances.",
        "",
        f"Spécification: [`{benchmark['mechanism_id']}.json`](frozen_baselines/{benchmark['mechanism_id']}.json).",
        "",
        "## Reproduction et prochaine étape",
        "",
        "```powershell",
        "python strategy_registry/psf_mnq/build_frozen_baselines.py",
        "```",
        "",
        "La prochaine étape autorisée par cette préparation est:",
        "",
        "```text",
        "frozen specification -> baseline backtest -> robustness -> OOS",
        "```",
        "",
        "Ce document et le script de construction ne lancent aucune de ces simulations.",
    ])
    REVIEW_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    strategy_specs = build_strategy_specs()
    benchmark = build_benchmark_spec()
    validate_specs(strategy_specs, benchmark)
    write_json_specs(strategy_specs, benchmark)
    write_manifest(strategy_specs, benchmark)
    write_review(strategy_specs, benchmark)
    print(f"Frozen strategy specifications: {len(strategy_specs)}")
    print("Market benchmarks: 1")
    print(f"Manifest rows: {len(strategy_specs) + 1}")
    print("Backtests executed: 0")


if __name__ == "__main__":
    main()
