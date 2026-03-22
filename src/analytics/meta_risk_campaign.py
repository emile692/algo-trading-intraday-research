"""Meta-risk overlay campaign on top of the existing ORB strategy.

This campaign keeps entry signals unchanged and only applies a daily
position-risk overlay (size scaling or skip-day) with strictly
chronological state updates.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analytics.metrics import compute_metrics
from src.config.orb_campaign import DEFAULT_CAMPAIGN_DATASET, build_execution_profiles
from src.config.paths import EXPORTS_DIR, NOTEBOOKS_DIR, ensure_directories
from src.config.settings import DEFAULT_INITIAL_CAPITAL_USD
from src.data.cleaning import clean_ohlcv
from src.data.loader import load_ohlcv_file
from src.data.session import add_session_date, extract_rth
from src.engine.backtester import run_backtest
from src.engine.execution_model import ExecutionModel
from src.engine.trade_log import empty_trade_log
from src.features.intraday import add_ema, add_intraday_features
from src.features.opening_range import compute_opening_range
from src.features.volatility import add_atr
from src.strategy.orb import ORBStrategy

VARIANT_BASELINE = "baseline"
VARIANT_HALF_AFTER_2_LOSSES = "half_after_2_losses"
VARIANT_SKIP_AFTER_3_LOSSES = "skip_day_after_3_losses"
VARIANT_LOCAL_DD_SCALING = "local_drawdown_scaling"

VARIANT_ORDER = [
    VARIANT_BASELINE,
    VARIANT_HALF_AFTER_2_LOSSES,
    VARIANT_SKIP_AFTER_3_LOSSES,
    VARIANT_LOCAL_DD_SCALING,
]

VARIANT_LABELS = {
    VARIANT_BASELINE: "1_baseline",
    VARIANT_HALF_AFTER_2_LOSSES: "2_half_after_2_losses",
    VARIANT_SKIP_AFTER_3_LOSSES: "3_skip_after_3_losses",
    VARIANT_LOCAL_DD_SCALING: "4_local_drawdown_scaling",
}


@dataclass(frozen=True)
class BaseStrategySpec:
    """Reference strategy setup used by the campaign."""

    dataset_path: Path = DEFAULT_CAMPAIGN_DATASET
    is_fraction: float = 0.70
    initial_capital_usd: float = DEFAULT_INITIAL_CAPITAL_USD
    risk_per_trade_pct: float = 0.25
    or_minutes: int = 15
    opening_time: str = "09:30:00"
    time_exit: str = "16:00:00"
    side_mode: str = "long_only"
    entry_buffer_ticks: int = 0
    stop_buffer_ticks: int = 2
    target_multiple: float = 5.0
    entry_on_next_open: bool = True
    atr_period: int = 14
    atr_q_low: float = 0.50
    atr_q_high: float = 1.00
    ema_length: int = 30
    direction_filter_mode: str = "ema_only"
    execution_profile: str = "repo_realistic"


@dataclass(frozen=True)
class ChallengeSimulationConfig:
    """Bootstrap challenge simulation settings."""

    target_return_pct: float = 0.06
    max_drawdown_pct: float = 0.04
    n_bootstrap_paths: int = 5000
    horizon_days: int | None = None
    random_seed: int = 42


@dataclass(frozen=True)
class MetaRiskCampaignSpec:
    """Campaign spec for strategy + challenge simulation."""

    strategy: BaseStrategySpec = field(default_factory=BaseStrategySpec)
    challenge: ChallengeSimulationConfig = field(default_factory=ChallengeSimulationConfig)


def _split_sessions(all_sessions: list, is_fraction: float) -> tuple[list, list]:
    if len(all_sessions) < 2:
        raise ValueError("Not enough sessions for IS/OOS split.")
    split_idx = int(len(all_sessions) * is_fraction)
    split_idx = max(1, min(len(all_sessions) - 1, split_idx))
    return all_sessions[:split_idx], all_sessions[split_idx:]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _variant_from_name(variant: str) -> str:
    if variant not in VARIANT_ORDER:
        valid = ", ".join(VARIANT_ORDER)
        raise ValueError(f"Unknown meta-risk variant '{variant}'. Valid values: {valid}.")
    return variant


def _init_policy_state(variant: str, initial_capital: float) -> dict[str, Any]:
    variant = _variant_from_name(variant)
    if variant == VARIANT_HALF_AFTER_2_LOSSES:
        return {"mode": "full", "loss_streak": 0}
    if variant == VARIANT_SKIP_AFTER_3_LOSSES:
        return {"loss_streak": 0, "skip_next_session": False}
    if variant == VARIANT_LOCAL_DD_SCALING:
        return {"equity": float(initial_capital), "peak_equity": float(initial_capital)}
    return {}


def _policy_multiplier(variant: str, state: dict[str, Any]) -> float:
    variant = _variant_from_name(variant)
    if variant == VARIANT_BASELINE:
        return 1.0
    if variant == VARIANT_HALF_AFTER_2_LOSSES:
        return 0.5 if state.get("mode") == "half" else 1.0
    if variant == VARIANT_SKIP_AFTER_3_LOSSES:
        return 0.0 if bool(state.get("skip_next_session", False)) else 1.0
    if variant == VARIANT_LOCAL_DD_SCALING:
        peak = _safe_float(state.get("peak_equity"), default=0.0)
        equity = _safe_float(state.get("equity"), default=peak)
        if peak <= 0:
            return 1.0
        drawdown_pct = max(0.0, (peak - equity) / peak)
        if drawdown_pct <= 0.01:
            return 1.0
        if drawdown_pct <= 0.02:
            return 0.5
        return 0.0
    return 1.0


def _update_policy_state(variant: str, state: dict[str, Any], daily_pnl_usd: float) -> None:
    variant = _variant_from_name(variant)
    pnl = _safe_float(daily_pnl_usd, default=0.0)

    if variant == VARIANT_BASELINE:
        return

    if variant == VARIANT_HALF_AFTER_2_LOSSES:
        if pnl < 0:
            state["loss_streak"] = int(state.get("loss_streak", 0)) + 1
        else:
            state["loss_streak"] = 0

        if state.get("mode") == "full" and int(state.get("loss_streak", 0)) >= 2:
            state["mode"] = "half"
        elif state.get("mode") == "half" and pnl > 0:
            state["mode"] = "full"
            state["loss_streak"] = 0
        return

    if variant == VARIANT_SKIP_AFTER_3_LOSSES:
        if bool(state.get("skip_next_session", False)):
            state["skip_next_session"] = False
            state["loss_streak"] = 0
            return

        if pnl < 0:
            state["loss_streak"] = int(state.get("loss_streak", 0)) + 1
        else:
            state["loss_streak"] = 0

        if int(state.get("loss_streak", 0)) >= 3:
            state["skip_next_session"] = True
            state["loss_streak"] = 0
        return

    if variant == VARIANT_LOCAL_DD_SCALING:
        equity = _safe_float(state.get("equity"), default=0.0) + pnl
        peak = _safe_float(state.get("peak_equity"), default=equity)
        state["equity"] = equity
        state["peak_equity"] = max(peak, equity)


def compute_policy_multipliers(
    variant: str,
    daily_pnl_sequence: list[float] | pd.Series,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL_USD,
) -> list[float]:
    """Return the sequence of pre-session multipliers for a realized daily PnL path.

    This helper is mainly used for deterministic unit tests of policy logic.
    """

    state = _init_policy_state(variant, initial_capital=initial_capital)
    multipliers: list[float] = []
    for pnl in pd.Series(daily_pnl_sequence, dtype=float).fillna(0.0):
        multipliers.append(_policy_multiplier(variant, state))
        _update_policy_state(variant, state, float(pnl))
    return multipliers


def _prepare_strategy_inputs(
    strategy_spec: BaseStrategySpec,
) -> tuple[pd.DataFrame, ORBStrategy, list, list, list, dict[str, Any]]:
    raw = load_ohlcv_file(strategy_spec.dataset_path)
    clean = clean_ohlcv(raw)
    rth = extract_rth(clean)
    rth = add_session_date(rth)
    feat = add_intraday_features(rth)
    feat = add_atr(feat, window=strategy_spec.atr_period)
    feat = add_ema(feat, window=strategy_spec.ema_length)
    feat = compute_opening_range(
        feat,
        or_minutes=strategy_spec.or_minutes,
        opening_time=strategy_spec.opening_time,
    )

    all_sessions = sorted(pd.to_datetime(feat["session_date"]).dt.date.unique())
    is_sessions, oos_sessions = _split_sessions(all_sessions, strategy_spec.is_fraction)

    atr_col = f"atr_{strategy_spec.atr_period}"
    is_mask = feat["session_date"].isin(set(is_sessions))
    atr_is = pd.to_numeric(feat.loc[is_mask, atr_col], errors="coerce").dropna()

    atr_min = None
    atr_max = None
    atr_regime = "none"
    if not atr_is.empty:
        low = float(atr_is.quantile(strategy_spec.atr_q_low))
        high = float(atr_is.quantile(strategy_spec.atr_q_high))
        if math.isfinite(low) and math.isfinite(high) and low < high:
            atr_min = low
            atr_max = high
            atr_regime = "is_quantile_band"

    strategy = ORBStrategy(
        or_minutes=strategy_spec.or_minutes,
        direction=strategy_spec.side_mode,
        one_trade_per_day=True,
        entry_buffer_ticks=strategy_spec.entry_buffer_ticks,
        stop_buffer_ticks=strategy_spec.stop_buffer_ticks,
        target_multiple=strategy_spec.target_multiple,
        opening_time=strategy_spec.opening_time,
        time_exit=strategy_spec.time_exit,
        account_size_usd=strategy_spec.initial_capital_usd,
        risk_per_trade_pct=strategy_spec.risk_per_trade_pct,
        tick_size=0.25,
        atr_period=strategy_spec.atr_period,
        atr_min=atr_min,
        atr_max=atr_max,
        atr_regime=atr_regime,
        direction_filter_mode=strategy_spec.direction_filter_mode,
        ema_length=strategy_spec.ema_length,
    )

    signal_df = strategy.generate_signals(feat)
    context = {
        "atr_column": atr_col,
        "atr_min_is": atr_min,
        "atr_max_is": atr_max,
        "atr_regime_mode": atr_regime,
    }
    return signal_df, strategy, all_sessions, is_sessions, oos_sessions, context


def _run_variant_sequential_backtest(
    signal_df: pd.DataFrame,
    all_sessions: list,
    execution_model: ExecutionModel,
    strategy_spec: BaseStrategySpec,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variant = _variant_from_name(variant)
    session_frames = {
        pd.to_datetime(session_date).date(): frame.sort_values("timestamp").copy()
        for session_date, frame in signal_df.groupby("session_date", sort=True)
    }

    policy_state = _init_policy_state(variant, initial_capital=strategy_spec.initial_capital_usd)
    trade_parts: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []

    for session_date in all_sessions:
        multiplier = float(_policy_multiplier(variant, policy_state))
        session_df = session_frames.get(pd.to_datetime(session_date).date())
        daily_pnl = 0.0
        base_risk_pct = _safe_float(strategy_spec.risk_per_trade_pct, default=0.0)

        if session_df is not None and not session_df.empty and multiplier > 0:
            scaled_risk_pct = base_risk_pct * multiplier
            if scaled_risk_pct > 0:
                day_trades = run_backtest(
                    session_df,
                    execution_model=execution_model,
                    time_exit=strategy_spec.time_exit,
                    stop_buffer_ticks=strategy_spec.stop_buffer_ticks,
                    target_multiple=strategy_spec.target_multiple,
                    account_size_usd=strategy_spec.initial_capital_usd,
                    risk_per_trade_pct=scaled_risk_pct,
                    entry_on_next_open=strategy_spec.entry_on_next_open,
                )
                if not day_trades.empty:
                    day_trades = day_trades.copy()
                    day_trades["meta_risk_variant"] = variant
                    day_trades["meta_risk_multiplier"] = multiplier
                    trade_parts.append(day_trades)
                    daily_pnl = float(day_trades["net_pnl_usd"].sum())

        _update_policy_state(variant, policy_state, daily_pnl_usd=daily_pnl)
        daily_rows.append(
            {
                "session_date": pd.to_datetime(session_date).date(),
                "meta_risk_variant": variant,
                "risk_multiplier": multiplier,
                "daily_pnl_usd": daily_pnl,
            }
        )

    if trade_parts:
        trades = pd.concat(trade_parts, ignore_index=True)
        trades = trades.sort_values("exit_time").reset_index(drop=True)
        trades["trade_id"] = np.arange(1, len(trades) + 1)
    else:
        trades = empty_trade_log()
        trades["meta_risk_variant"] = pd.Series(dtype=str)
        trades["meta_risk_multiplier"] = pd.Series(dtype=float)

    daily = pd.DataFrame(daily_rows).sort_values("session_date").reset_index(drop=True)
    daily["equity"] = strategy_spec.initial_capital_usd + daily["daily_pnl_usd"].cumsum()
    daily["peak_equity"] = daily["equity"].cummax()
    daily["drawdown_usd"] = daily["equity"] - daily["peak_equity"]
    daily["drawdown_pct"] = np.where(
        daily["peak_equity"] > 0,
        (daily["peak_equity"] - daily["equity"]) / daily["peak_equity"],
        0.0,
    )
    return trades, daily


def _metrics_for_sessions(
    trades: pd.DataFrame,
    sessions: list,
    initial_capital: float,
) -> dict[str, Any]:
    if trades.empty:
        subset = trades.copy()
    else:
        session_set = set(pd.to_datetime(pd.Index(sessions)).date)
        subset = trades.loc[pd.to_datetime(trades["session_date"]).dt.date.isin(session_set)].copy()
    metrics = compute_metrics(
        subset,
        session_dates=sessions,
        initial_capital=initial_capital,
    )
    daily = pd.Series(0.0, index=pd.Index(pd.to_datetime(pd.Index(sessions)).date), dtype=float)
    if not subset.empty:
        grouped = subset.groupby(pd.to_datetime(subset["session_date"]).dt.date)["net_pnl_usd"].sum()
        daily = daily.add(grouped, fill_value=0.0)

    cum = daily.cumsum()
    cum_with_start = pd.concat([pd.Series([0.0]), cum], ignore_index=True)
    dd = cum_with_start - cum_with_start.cummax()
    max_dd = float(dd.min()) if not dd.empty else 0.0
    metrics["max_drawdown"] = max_dd
    metrics["max_drawdown_pct"] = float(abs(max_dd) / initial_capital) if initial_capital > 0 else 0.0
    return metrics


def simulate_prop_challenge(
    daily_pnl: pd.Series,
    account_size_usd: float,
    config: ChallengeSimulationConfig,
) -> dict[str, Any]:
    """Bootstrap the probability of reaching target return before drawdown breach."""

    series = pd.to_numeric(pd.Series(daily_pnl), errors="coerce").fillna(0.0)
    if series.empty or config.n_bootstrap_paths <= 0:
        return {
            "challenge_target_return_pct": config.target_return_pct,
            "challenge_drawdown_limit_pct": config.max_drawdown_pct,
            "challenge_drawdown_limit_usd": account_size_usd * config.max_drawdown_pct,
            "challenge_target_usd": account_size_usd * config.target_return_pct,
            "challenge_horizon_days": 0,
            "challenge_bootstrap_paths": config.n_bootstrap_paths,
            "challenge_pass_rate": 0.0,
            "challenge_success_count": 0,
            "challenge_median_days_to_target": np.nan,
            "challenge_mean_days_to_target": np.nan,
            "challenge_p25_days_to_target": np.nan,
            "challenge_p75_days_to_target": np.nan,
            "historical_path_pass": False,
            "historical_days_to_target": np.nan,
        }

    returns = series.to_numpy(dtype=float)
    horizon = int(config.horizon_days) if config.horizon_days is not None else int(len(returns))
    horizon = max(1, horizon)

    target_usd = float(account_size_usd * config.target_return_pct)
    drawdown_limit_usd = float(account_size_usd * config.max_drawdown_pct)

    rng = np.random.default_rng(config.random_seed)
    success_days: list[int] = []
    for _ in range(int(config.n_bootstrap_paths)):
        sampled = rng.choice(returns, size=horizon, replace=True)
        cumulative = np.cumsum(sampled)

        target_hit = np.flatnonzero(cumulative >= target_usd)
        dd_hit = np.flatnonzero(cumulative <= -drawdown_limit_usd)

        target_idx = int(target_hit[0]) if len(target_hit) > 0 else None
        dd_idx = int(dd_hit[0]) if len(dd_hit) > 0 else None

        if target_idx is not None and (dd_idx is None or target_idx < dd_idx):
            success_days.append(target_idx + 1)

    cumulative_realized = np.cumsum(returns)
    target_hit_real = np.flatnonzero(cumulative_realized >= target_usd)
    dd_hit_real = np.flatnonzero(cumulative_realized <= -drawdown_limit_usd)
    target_idx_real = int(target_hit_real[0]) if len(target_hit_real) > 0 else None
    dd_idx_real = int(dd_hit_real[0]) if len(dd_hit_real) > 0 else None
    historical_pass = target_idx_real is not None and (dd_idx_real is None or target_idx_real < dd_idx_real)

    success_count = len(success_days)
    pass_rate = float(success_count / config.n_bootstrap_paths) if config.n_bootstrap_paths > 0 else 0.0
    return {
        "challenge_target_return_pct": config.target_return_pct,
        "challenge_drawdown_limit_pct": config.max_drawdown_pct,
        "challenge_drawdown_limit_usd": drawdown_limit_usd,
        "challenge_target_usd": target_usd,
        "challenge_horizon_days": horizon,
        "challenge_bootstrap_paths": int(config.n_bootstrap_paths),
        "challenge_pass_rate": pass_rate,
        "challenge_success_count": int(success_count),
        "challenge_median_days_to_target": float(np.median(success_days)) if success_days else np.nan,
        "challenge_mean_days_to_target": float(np.mean(success_days)) if success_days else np.nan,
        "challenge_p25_days_to_target": float(np.quantile(success_days, 0.25)) if success_days else np.nan,
        "challenge_p75_days_to_target": float(np.quantile(success_days, 0.75)) if success_days else np.nan,
        "historical_path_pass": bool(historical_pass),
        "historical_days_to_target": (target_idx_real + 1) if historical_pass else np.nan,
    }


def _comparison_metrics_row(
    variant: str,
    overall: dict[str, Any],
    is_metrics: dict[str, Any],
    oos_metrics: dict[str, Any],
    challenge_metrics: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
    }
    metric_map = {
        "pnl": "cumulative_pnl",
        "trades": "n_trades",
        "win_rate": "win_rate",
        "profit_factor": "profit_factor",
        "expectancy": "expectancy",
        "sharpe": "sharpe_ratio",
        "max_drawdown": "max_drawdown",
    }
    for split_name, split_metrics in (("overall", overall), ("is", is_metrics), ("oos", oos_metrics)):
        for out_name, key in metric_map.items():
            row[f"{split_name}_{out_name}"] = split_metrics.get(key, np.nan)

    row.update(challenge_metrics)
    return row


def _rank_variants(compare_df: pd.DataFrame) -> pd.DataFrame:
    ranked = compare_df.copy()
    ranked["oos_abs_max_drawdown"] = ranked["oos_max_drawdown"].abs()
    ranked["challenge_days_for_sort"] = pd.to_numeric(
        ranked["challenge_median_days_to_target"],
        errors="coerce",
    ).fillna(np.inf)

    ranked = ranked.sort_values(
        by=[
            "challenge_pass_rate",
            "challenge_days_for_sort",
            "oos_abs_max_drawdown",
            "oos_sharpe",
            "oos_profit_factor",
            "oos_expectancy",
        ],
        ascending=[False, True, True, False, False, False],
    ).reset_index(drop=True)
    ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def _plot_equity_curves(
    daily_curves: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for variant in VARIANT_ORDER:
        curve = daily_curves.get(variant)
        if curve is None or curve.empty:
            continue
        x = pd.to_datetime(curve["session_date"])
        axes[0].plot(x, curve["equity"], linewidth=1.3, label=VARIANT_LABELS[variant])
        axes[1].plot(x, 100.0 * curve["drawdown_pct"], linewidth=1.3, label=VARIANT_LABELS[variant])

    axes[0].set_title("Equity Curves (Daily, Chronological IS+OOS)")
    axes[0].set_ylabel("Equity (USD)")
    axes[0].legend(loc="best")
    axes[1].set_title("Drawdown Curves")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Session")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_report(
    spec: MetaRiskCampaignSpec,
    ranking_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    all_sessions: list,
    is_sessions: list,
    oos_sessions: list,
    context: dict[str, Any],
) -> str:
    winner = ranking_df.iloc[0]
    top_summary = ranking_df[
        [
            "selection_rank",
            "variant_label",
            "challenge_pass_rate",
            "challenge_median_days_to_target",
            "oos_max_drawdown",
            "oos_sharpe",
            "oos_profit_factor",
            "oos_expectancy",
            "oos_pnl",
            "oos_trades",
        ]
    ].copy()

    caveat_lines: list[str] = []
    max_pass_rate = float(top_summary["challenge_pass_rate"].max())
    if max_pass_rate < 0.30:
        caveat_lines.append(
            "- Le pass rate challenge reste faible sur toutes les variantes: amélioration relative oui, robustesse absolue encore limitée."
        )
    if top_summary["challenge_median_days_to_target"].isna().all():
        caveat_lines.append("- Aucun scénario gagnant dans le bootstrap: objectif +6% trop ambitieux pour ce profil.")
    if not caveat_lines:
        caveat_lines.append("- Le gagnant est relatif à ce dataset et ces hypothèses de coûts; il faut revalider hors-échantillon complémentaire.")

    atr_line = "- ATR filter disabled (not enough IS information)."
    if context.get("atr_min_is") is not None and context.get("atr_max_is") is not None:
        atr_line = (
            f"- ATR IS band: `{context['atr_column']}` in "
            f"[{float(context['atr_min_is']):.4f}, {float(context['atr_max_is']):.4f}]"
        )

    return "\n".join(
        [
            "# Meta-Risk Control Campaign",
            "",
            "## Scope",
            "",
            "- Entry signal logic unchanged (overlay only on risk size / trading permission).",
            "- No look-ahead in control state updates (strict chronological daily updates).",
            "- IS/OOS split preserved; ATR filter thresholds calibrated on IS and frozen on OOS.",
            "",
            "## Dataset & Split",
            "",
            f"- Dataset: `{spec.strategy.dataset_path.name}`",
            f"- Sessions total: {len(all_sessions)} ({all_sessions[0]} -> {all_sessions[-1]})",
            f"- IS sessions: {len(is_sessions)}",
            f"- OOS sessions: {len(oos_sessions)} (from {oos_sessions[0]})",
            "",
            "## Baseline Strategy (unchanged signal)",
            "",
            f"- OR minutes: `{spec.strategy.or_minutes}`",
            f"- Side mode: `{spec.strategy.side_mode}`",
            f"- Target multiple: `{spec.strategy.target_multiple}`",
            f"- Stop buffer ticks: `{spec.strategy.stop_buffer_ticks}`",
            f"- Risk per trade (base): `{spec.strategy.risk_per_trade_pct}%`",
            f"- EMA filter: `{spec.strategy.direction_filter_mode}` / EMA{spec.strategy.ema_length}",
            atr_line,
            "",
            "## Variants",
            "",
            "- 1_baseline",
            "- 2_half_after_2_losses",
            "- 3_skip_after_3_losses",
            "- 4_local_drawdown_scaling",
            "",
            "## Ranking Priority",
            "",
            "1) challenge pass rate",
            "2) median days to +6%",
            "3) drawdown",
            "4) Sharpe/PF/expectancy",
            "",
            "## Final Ranking",
            "",
            "```text",
            top_summary.to_string(index=False),
            "```",
            "",
            "## Winner",
            "",
            (
                f"- Selected: `{winner['variant_label']}` "
                f"(pass_rate={float(winner['challenge_pass_rate']):.2%}, "
                f"median_days={winner['challenge_median_days_to_target']}, "
                f"OOS_DD={float(winner['oos_max_drawdown']):.2f})."
            ),
            "",
            "## Honest Conclusion",
            "",
            *caveat_lines,
            "",
            "## Full Comparative Table",
            "",
            "```text",
            compare_df[
                [
                    "variant_label",
                    "overall_pnl",
                    "overall_trades",
                    "overall_win_rate",
                    "overall_profit_factor",
                    "overall_expectancy",
                    "overall_sharpe",
                    "overall_max_drawdown",
                    "oos_pnl",
                    "oos_trades",
                    "oos_win_rate",
                    "oos_profit_factor",
                    "oos_expectancy",
                    "oos_sharpe",
                    "oos_max_drawdown",
                    "challenge_pass_rate",
                    "challenge_median_days_to_target",
                ]
            ].to_string(index=False),
            "```",
            "",
        ]
    )


def _build_notebook_cell(cell_type: str, source: str) -> dict[str, Any]:
    if not source.endswith("\n"):
        source = source + "\n"
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def generate_meta_risk_notebook(
    notebook_path: Path,
    output_dir: Path,
    spec: MetaRiskCampaignSpec,
) -> Path:
    """Create an executable notebook to rerun the full meta-risk campaign."""

    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    setup_code = """from pathlib import Path
import sys

root = Path.cwd().resolve()
while root != root.parent and not (root / "pyproject.toml").exists():
    root = root.parent

if not (root / "pyproject.toml").exists():
    raise RuntimeError("Could not locate repository root from current working directory.")

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

print(f"Project root: {root}")
"""

    imports_code = """import pandas as pd
import matplotlib.pyplot as plt

from src.analytics.meta_risk_campaign import (
    BaseStrategySpec,
    ChallengeSimulationConfig,
    MetaRiskCampaignSpec,
    run_meta_risk_campaign,
)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 120)
"""

    config_code = f"""OUTPUT_DIR = root / "data" / "exports" / "{output_dir.name}"

spec = MetaRiskCampaignSpec(
    strategy=BaseStrategySpec(
        dataset_path=root / "data" / "raw" / "{spec.strategy.dataset_path.name}",
        is_fraction={spec.strategy.is_fraction},
        initial_capital_usd={spec.strategy.initial_capital_usd},
        risk_per_trade_pct={spec.strategy.risk_per_trade_pct},
        or_minutes={spec.strategy.or_minutes},
        opening_time="{spec.strategy.opening_time}",
        time_exit="{spec.strategy.time_exit}",
        side_mode="{spec.strategy.side_mode}",
        entry_buffer_ticks={spec.strategy.entry_buffer_ticks},
        stop_buffer_ticks={spec.strategy.stop_buffer_ticks},
        target_multiple={spec.strategy.target_multiple},
        entry_on_next_open={spec.strategy.entry_on_next_open},
        atr_period={spec.strategy.atr_period},
        atr_q_low={spec.strategy.atr_q_low},
        atr_q_high={spec.strategy.atr_q_high},
        ema_length={spec.strategy.ema_length},
        direction_filter_mode="{spec.strategy.direction_filter_mode}",
        execution_profile="{spec.strategy.execution_profile}",
    ),
    challenge=ChallengeSimulationConfig(
        target_return_pct={spec.challenge.target_return_pct},
        max_drawdown_pct={spec.challenge.max_drawdown_pct},
        n_bootstrap_paths={spec.challenge.n_bootstrap_paths},
        horizon_days={spec.challenge.horizon_days},
        random_seed={spec.challenge.random_seed},
    ),
)
"""

    run_code = """artifacts = run_meta_risk_campaign(spec=spec, output_dir=OUTPUT_DIR, notebook_path=None)
for name, path in artifacts.items():
    print(f"{name}: {path}")
"""

    display_code = """ranking = pd.read_csv(artifacts["ranking_csv"])
compare = pd.read_csv(artifacts["comparative_csv"])
display(ranking)
display(compare)
"""

    plot_code = """from pathlib import Path

curves_dir = Path(artifacts["curves_dir"])
variant_files = sorted(curves_dir.glob("equity_curve__*.csv"))

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for path in variant_files:
    curve = pd.read_csv(path)
    label = path.stem.replace("equity_curve__", "")
    x = pd.to_datetime(curve["session_date"])
    axes[0].plot(x, curve["equity"], label=label, linewidth=1.2)
    axes[1].plot(x, 100.0 * curve["drawdown_pct"], label=label, linewidth=1.2)

axes[0].set_title("Equity Curves")
axes[0].set_ylabel("Equity (USD)")
axes[0].legend(loc="best")
axes[1].set_title("Drawdown Curves (%)")
axes[1].set_ylabel("Drawdown (%)")
axes[1].set_xlabel("Session")
axes[1].legend(loc="best")
plt.tight_layout()
plt.show()
"""

    summary_md = """## Notes

- Ranking priority is: challenge pass rate, then time to +6%, then drawdown, then Sharpe/PF/expectancy.
- Overlay controls do not alter entry signal generation.
- Read `meta_risk_report.md` for the written conclusion and caveats.
"""

    notebook = {
        "cells": [
            _build_notebook_cell(
                "markdown",
                "# Meta-Risk Campaign Notebook\n\nExecutable notebook for the 4-variant meta-risk overlay campaign.",
            ),
            _build_notebook_cell("code", setup_code),
            _build_notebook_cell("code", imports_code),
            _build_notebook_cell("code", config_code),
            _build_notebook_cell("code", run_code),
            _build_notebook_cell("code", display_code),
            _build_notebook_cell("code", plot_code),
            _build_notebook_cell("markdown", summary_md),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return notebook_path


def run_meta_risk_campaign(
    spec: MetaRiskCampaignSpec,
    output_dir: Path,
    notebook_path: Path | None = None,
) -> dict[str, Path]:
    """Run the full 4-variant meta-risk campaign and export artifacts."""

    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    curves_dir = output_dir / "curves"
    summary_dir = output_dir / "summary"
    tables_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    signal_df, _, all_sessions, is_sessions, oos_sessions, context = _prepare_strategy_inputs(spec.strategy)

    profiles = build_execution_profiles()
    if spec.strategy.execution_profile not in profiles:
        available = ", ".join(sorted(profiles.keys()))
        raise ValueError(
            f"Unknown execution_profile '{spec.strategy.execution_profile}'. Available: {available}."
        )
    profile = profiles[spec.strategy.execution_profile]
    execution_model = ExecutionModel(
        commission_per_side_usd=profile.commission_per_side_usd,
        slippage_ticks=profile.slippage_ticks,
        tick_size=profile.tick_size,
    )

    compare_rows: list[dict[str, Any]] = []
    daily_curves: dict[str, pd.DataFrame] = {}
    trades_by_variant: dict[str, pd.DataFrame] = {}
    controls_by_variant: dict[str, pd.DataFrame] = {}

    for variant in VARIANT_ORDER:
        trades, daily = _run_variant_sequential_backtest(
            signal_df=signal_df,
            all_sessions=all_sessions,
            execution_model=execution_model,
            strategy_spec=spec.strategy,
            variant=variant,
        )
        daily_curves[variant] = daily
        trades_by_variant[variant] = trades
        controls_by_variant[variant] = daily[["session_date", "risk_multiplier", "daily_pnl_usd"]].copy()

        overall_metrics = _metrics_for_sessions(trades, all_sessions, spec.strategy.initial_capital_usd)
        is_metrics = _metrics_for_sessions(trades, is_sessions, spec.strategy.initial_capital_usd)
        oos_metrics = _metrics_for_sessions(trades, oos_sessions, spec.strategy.initial_capital_usd)

        oos_daily = daily.loc[pd.to_datetime(daily["session_date"]).dt.date.isin(set(oos_sessions)), "daily_pnl_usd"]
        challenge_metrics = simulate_prop_challenge(
            daily_pnl=oos_daily,
            account_size_usd=spec.strategy.initial_capital_usd,
            config=spec.challenge,
        )
        compare_rows.append(
            _comparison_metrics_row(
                variant=variant,
                overall=overall_metrics,
                is_metrics=is_metrics,
                oos_metrics=oos_metrics,
                challenge_metrics=challenge_metrics,
            )
        )

    compare_df = pd.DataFrame(compare_rows)
    compare_df = compare_df.set_index("variant").loc[VARIANT_ORDER].reset_index()
    ranking_df = _rank_variants(compare_df)

    compare_csv = tables_dir / "meta_risk_comparative_metrics.csv"
    ranking_csv = tables_dir / "meta_risk_ranking.csv"
    challenge_csv = tables_dir / "meta_risk_challenge_simulation.csv"
    compare_df.to_csv(compare_csv, index=False)
    ranking_df.to_csv(ranking_csv, index=False)
    compare_df[
        [
            "variant",
            "variant_label",
            "challenge_target_return_pct",
            "challenge_drawdown_limit_pct",
            "challenge_target_usd",
            "challenge_drawdown_limit_usd",
            "challenge_horizon_days",
            "challenge_bootstrap_paths",
            "challenge_pass_rate",
            "challenge_median_days_to_target",
            "challenge_mean_days_to_target",
            "challenge_p25_days_to_target",
            "challenge_p75_days_to_target",
            "historical_path_pass",
            "historical_days_to_target",
        ]
    ].to_csv(challenge_csv, index=False)

    for variant, curve in daily_curves.items():
        curve.to_csv(curves_dir / f"equity_curve__{variant}.csv", index=False)
    for variant, trades in trades_by_variant.items():
        trades.to_csv(tables_dir / f"trades__{variant}.csv", index=False)
    for variant, control in controls_by_variant.items():
        control.to_csv(tables_dir / f"daily_controls__{variant}.csv", index=False)

    curves_png = curves_dir / "equity_drawdown_curves.png"
    _plot_equity_curves(daily_curves=daily_curves, output_path=curves_png)

    report_path = summary_dir / "meta_risk_report.md"
    report_path.write_text(
        _build_report(
            spec=spec,
            ranking_df=ranking_df,
            compare_df=compare_df,
            all_sessions=all_sessions,
            is_sessions=is_sessions,
            oos_sessions=oos_sessions,
            context=context,
        ),
        encoding="utf-8",
    )

    metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "spec": asdict(spec),
        "session_count_total": len(all_sessions),
        "session_count_is": len(is_sessions),
        "session_count_oos": len(oos_sessions),
        "oos_start_date": str(oos_sessions[0]) if oos_sessions else None,
        "winner_variant": str(ranking_df.iloc[0]["variant"]) if not ranking_df.empty else None,
        "output_dir": str(output_dir),
    }
    (summary_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    generated_notebook = None
    if notebook_path is not None:
        generated_notebook = generate_meta_risk_notebook(
            notebook_path=notebook_path,
            output_dir=output_dir,
            spec=spec,
        )

    artifacts: dict[str, Path] = {
        "output_dir": output_dir,
        "comparative_csv": compare_csv,
        "ranking_csv": ranking_csv,
        "challenge_csv": challenge_csv,
        "curves_png": curves_png,
        "report_md": report_path,
        "curves_dir": curves_dir,
        "tables_dir": tables_dir,
    }
    if generated_notebook is not None:
        artifacts["notebook"] = generated_notebook
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run meta-risk overlay campaign.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CAMPAIGN_DATASET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--notebook-path", type=Path, default=None)
    parser.add_argument("--is-fraction", type=float, default=0.70)
    parser.add_argument("--challenge-target-pct", type=float, default=0.06)
    parser.add_argument("--challenge-dd-pct", type=float, default=0.04)
    parser.add_argument("--challenge-paths", type=int, default=5000)
    parser.add_argument("--challenge-horizon-days", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (EXPORTS_DIR / f"meta_risk_campaign_{timestamp}")
    notebook_path = args.notebook_path or (NOTEBOOKS_DIR / "meta_risk_campaign.ipynb")

    spec = MetaRiskCampaignSpec(
        strategy=BaseStrategySpec(
            dataset_path=args.dataset,
            is_fraction=args.is_fraction,
        ),
        challenge=ChallengeSimulationConfig(
            target_return_pct=args.challenge_target_pct,
            max_drawdown_pct=args.challenge_dd_pct,
            n_bootstrap_paths=args.challenge_paths,
            horizon_days=args.challenge_horizon_days,
            random_seed=args.seed,
        ),
    )

    artifacts = run_meta_risk_campaign(spec=spec, output_dir=output_dir, notebook_path=notebook_path)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
