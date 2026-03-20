"""Performance metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config.orb_campaign import PropConstraintConfig
from src.config.settings import DEFAULT_INITIAL_CAPITAL_USD


def _loss_streak_lengths(pnl: pd.Series) -> list[int]:
    """Return lengths of consecutive losing-trade streaks."""
    streaks: list[int] = []
    current = 0

    for value in pnl:
        if value < 0:
            current += 1
            continue
        if current > 0:
            streaks.append(current)
            current = 0

    if current > 0:
        streaks.append(current)
    return streaks


def _resolve_session_dates(
    trades: pd.DataFrame,
    signal_df: pd.DataFrame | None,
    session_dates: pd.Series | pd.Index | list | None,
) -> pd.Index:
    """Resolve the session-date index used for daily metrics."""
    if session_dates is not None:
        return pd.Index(pd.to_datetime(pd.Index(session_dates)).date)
    if signal_df is not None and "session_date" in signal_df.columns:
        return pd.Index(pd.to_datetime(signal_df["session_date"]).dt.date.unique())
    if not trades.empty and "session_date" in trades.columns:
        return pd.Index(pd.to_datetime(trades["session_date"]).dt.date.unique())
    return pd.Index([])


def _daily_pnl(trades: pd.DataFrame, resolved_session_dates: pd.Index) -> pd.Series:
    """Aggregate daily PnL, including flat days when session dates are provided."""
    if trades.empty:
        if resolved_session_dates.empty:
            return pd.Series(dtype=float)
        return pd.Series(0.0, index=resolved_session_dates, dtype=float)

    daily = trades.groupby("session_date")["net_pnl_usd"].sum()
    daily.index = pd.Index(pd.to_datetime(daily.index).date)
    if resolved_session_dates.empty:
        return daily.sort_index()
    return daily.reindex(resolved_session_dates, fill_value=0.0).sort_index()


def _prop_empty_metrics(
    resolved_session_dates: pd.Index,
    capital: float,
    prop_constraints: PropConstraintConfig | None,
) -> dict[str, float | int | bool | str]:
    """Return prop-style defaults for empty trade logs."""
    if prop_constraints is None:
        return {}

    observed_months = (
        float(len(resolved_session_dates) / prop_constraints.trading_days_per_month)
        if prop_constraints.trading_days_per_month > 0
        else 0.0
    )
    subscription_drag = observed_months * prop_constraints.monthly_subscription_cost_usd

    return {
        "prop_constraint_profile": prop_constraints.name,
        "profit_target_usd": prop_constraints.profit_target_usd,
        "max_loss_limit_usd": prop_constraints.max_loss_limit_usd,
        "daily_loss_limit_usd": prop_constraints.daily_loss_limit_usd or 0.0,
        "daily_loss_limit_active": bool(prop_constraints.daily_loss_limit_usd is not None),
        "profit_target_reached": False,
        "days_to_profit_target": np.nan,
        "days_to_reach_3000_profit_target": np.nan,
        "estimated_months_to_profit_target": np.nan,
        "profit_target_reached_before_max_loss": False,
        "breaches_max_loss_limit": False,
        "max_loss_limit_buffer_usd": prop_constraints.max_loss_limit_usd,
        "any_daily_loss_limit_breach": False,
        "number_of_daily_loss_limit_breaches": 0,
        "minimum_equity_before_target_hit": np.nan,
        "minimum_equity_observed": capital,
        "max_adverse_drawdown_before_target_hit": np.nan,
        "subscription_drag_estimate": float(subscription_drag),
        "estimated_pnl_after_subscription": float(-subscription_drag),
        "profit_target_progress_pct": 0.0,
    }


def _compute_prop_metrics(
    trades: pd.DataFrame,
    daily: pd.Series,
    resolved_session_dates: pd.Index,
    capital: float,
    prop_constraints: PropConstraintConfig | None,
) -> dict[str, float | int | bool | str]:
    """Estimate practical prop-style survival metrics from the realized path."""
    if prop_constraints is None:
        return {}

    if trades.empty:
        return _prop_empty_metrics(resolved_session_dates, capital, prop_constraints)

    ordered = trades.sort_values("exit_time").reset_index(drop=True).copy()
    ordered["cum_pnl"] = ordered["net_pnl_usd"].astype(float).cumsum()
    ordered["equity"] = capital + ordered["cum_pnl"]

    cumulative = ordered["cum_pnl"]
    minimum_equity_observed = float(ordered["equity"].min())
    worst_cumulative_pnl = float(cumulative.min())

    target_positions = np.flatnonzero(cumulative >= prop_constraints.profit_target_usd)
    target_position = int(target_positions[0]) if len(target_positions) > 0 else None
    profit_target_reached = target_position is not None

    max_loss_positions = np.flatnonzero(cumulative <= -prop_constraints.max_loss_limit_usd)
    max_loss_position = int(max_loss_positions[0]) if len(max_loss_positions) > 0 else None
    breaches_max_loss_limit = max_loss_position is not None

    days_to_profit_target = np.nan
    estimated_months_to_profit_target = np.nan
    minimum_equity_before_target_hit = np.nan
    max_adverse_drawdown_before_target_hit = np.nan

    if profit_target_reached:
        target_session_date = pd.to_datetime(ordered.loc[target_position, "session_date"]).date()
        if not resolved_session_dates.empty:
            days_to_profit_target = int((resolved_session_dates <= target_session_date).sum())
        else:
            days_to_profit_target = int(
                pd.Index(pd.to_datetime(ordered.loc[:target_position, "session_date"]).date).nunique()
            )
        if prop_constraints.trading_days_per_month > 0:
            estimated_months_to_profit_target = float(
                days_to_profit_target / prop_constraints.trading_days_per_month
            )

        path_before_target = ordered.loc[:target_position]
        minimum_equity_before_target_hit = float(path_before_target["equity"].min())
        max_adverse_drawdown_before_target_hit = float(capital - minimum_equity_before_target_hit)

    daily_loss_limit = prop_constraints.daily_loss_limit_usd
    if daily_loss_limit is not None:
        daily_breach_mask = daily <= -daily_loss_limit
        number_of_daily_breaches = int(daily_breach_mask.sum())
    else:
        number_of_daily_breaches = 0
    any_daily_loss_limit_breach = bool(number_of_daily_breaches > 0)

    observed_months = (
        float(len(resolved_session_dates) / prop_constraints.trading_days_per_month)
        if prop_constraints.trading_days_per_month > 0
        else 0.0
    )
    subscription_months = (
        float(estimated_months_to_profit_target) if profit_target_reached else observed_months
    )
    subscription_drag = float(subscription_months * prop_constraints.monthly_subscription_cost_usd)

    return {
        "prop_constraint_profile": prop_constraints.name,
        "profit_target_usd": prop_constraints.profit_target_usd,
        "max_loss_limit_usd": prop_constraints.max_loss_limit_usd,
        "daily_loss_limit_usd": prop_constraints.daily_loss_limit_usd or 0.0,
        "daily_loss_limit_active": bool(prop_constraints.daily_loss_limit_usd is not None),
        "profit_target_reached": bool(profit_target_reached),
        "days_to_profit_target": days_to_profit_target,
        "days_to_reach_3000_profit_target": days_to_profit_target,
        "estimated_months_to_profit_target": estimated_months_to_profit_target,
        "profit_target_reached_before_max_loss": bool(
            profit_target_reached and (max_loss_position is None or target_position <= max_loss_position)
        ),
        "breaches_max_loss_limit": bool(breaches_max_loss_limit),
        "max_loss_limit_buffer_usd": float(prop_constraints.max_loss_limit_usd - abs(min(worst_cumulative_pnl, 0.0))),
        "any_daily_loss_limit_breach": any_daily_loss_limit_breach,
        "number_of_daily_loss_limit_breaches": number_of_daily_breaches,
        "minimum_equity_before_target_hit": minimum_equity_before_target_hit,
        "minimum_equity_observed": minimum_equity_observed,
        "max_adverse_drawdown_before_target_hit": max_adverse_drawdown_before_target_hit,
        "subscription_drag_estimate": subscription_drag,
        "estimated_pnl_after_subscription": float(cumulative.iloc[-1] - subscription_drag),
        "profit_target_progress_pct": float(cumulative.iloc[-1] / prop_constraints.profit_target_usd)
        if prop_constraints.profit_target_usd > 0
        else 0.0,
    }


def compute_metrics(
    trades: pd.DataFrame,
    signal_df: pd.DataFrame | None = None,
    session_dates: pd.Series | pd.Index | list | None = None,
    initial_capital: float | None = None,
    prop_constraints: PropConstraintConfig | None = None,
) -> dict[str, float | int | bool | str]:
    """Compute key strategy metrics from trade log."""
    resolved_session_dates = _resolve_session_dates(trades, signal_df, session_dates)
    n_sessions = int(len(resolved_session_dates))

    if trades.empty:
        empty_metrics: dict[str, float | int | bool | str] = {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "average_losing_trade": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "cumulative_pnl": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_consecutive_losses": 0,
            "avg_consecutive_losses": 0.0,
            "longest_loss_streak": 0,
            "number_of_loss_streaks_2_plus": 0,
            "worst_trade": 0.0,
            "worst_day": 0.0,
            "stop_hit_rate": 0.0,
            "target_hit_rate": 0.0,
            "eod_exit_rate": 0.0,
            "proportion_filtered_out": 0.0,
            "trade_count_after_filters": 0,
            "percent_of_days_traded": 0.0,
            "percent_days_traded": 0.0,
            "avg_R": 0.0,
            "median_R": 0.0,
            "pnl_to_risk_ratio": 0.0,
            "average_loss_streak_length": 0.0,
            "count_loss_streaks_2_plus": 0,
            "daily_sharpe_basis": "daily_pnl_over_static_capital",
            "n_sessions": n_sessions,
            "raw_signal_count": 0,
        }
        empty_metrics.update(_prop_empty_metrics(resolved_session_dates, initial_capital or DEFAULT_INITIAL_CAPITAL_USD, prop_constraints))
        return empty_metrics

    pnl = trades["net_pnl_usd"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax()

    gross_profit = wins.sum()
    gross_loss_abs = abs(losses.sum())
    profit_factor = float(gross_profit / gross_loss_abs) if gross_loss_abs > 0 else np.inf

    capital = initial_capital
    if capital is None:
        account_sizes = trades["account_size_usd"].dropna() if "account_size_usd" in trades.columns else pd.Series()
        capital = float(account_sizes.iloc[0]) if not account_sizes.empty else DEFAULT_INITIAL_CAPITAL_USD

    daily = _daily_pnl(trades, resolved_session_dates)
    if len(daily) > 1 and capital > 0:
        daily_returns = daily / capital
        daily_std = daily_returns.std(ddof=0)
        sharpe = (daily_returns.mean() / daily_std) * math.sqrt(252.0) if daily_std > 0 else 0.0
    else:
        sharpe = 0.0

    loss_streaks = _loss_streak_lengths(pnl)
    longest_loss_streak = max(loss_streaks, default=0)
    avg_loss_streak = float(np.mean(loss_streaks)) if loss_streaks else 0.0
    streaks_2_plus = int(sum(length >= 2 for length in loss_streaks))

    raw_signal_count = 0
    filtered_out_count = 0
    if signal_df is not None and "raw_signal" in signal_df.columns:
        raw_signal_count = int(signal_df["raw_signal"].ne(0).sum())
        filtered_out_count = int((signal_df["raw_signal"].ne(0) & signal_df["signal"].eq(0)).sum())

    trade_risk = trades["trade_risk_usd"] if "trade_risk_usd" in trades.columns else pd.Series(dtype=float)
    valid_trade_risk = trade_risk[(trade_risk.notna()) & (trade_risk > 0)]
    if len(valid_trade_risk) == len(trades):
        r_multiple = pnl / valid_trade_risk
        avg_r = float(r_multiple.mean())
        median_r = float(r_multiple.median())
        pnl_to_risk = float(pnl.sum() / valid_trade_risk.sum()) if valid_trade_risk.sum() > 0 else 0.0
    else:
        avg_r = 0.0
        median_r = 0.0
        pnl_to_risk = 0.0

    traded_days = int(pd.Index(pd.to_datetime(trades["session_date"]).dt.date).nunique())
    percent_of_days_traded = float(traded_days / n_sessions) if n_sessions > 0 else 0.0

    metrics: dict[str, float | int | bool | str] = {
        "n_trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
        "average_losing_trade": float(losses.mean()) if not losses.empty else 0.0,
        "expectancy": float(pnl.mean()),
        "profit_factor": float(profit_factor),
        "cumulative_pnl": float(pnl.sum()),
        "max_drawdown": float(drawdown.min()),
        "max_drawdown_pct": float(abs(drawdown.min()) / capital) if capital > 0 else 0.0,
        "sharpe_ratio": float(sharpe),
        "max_consecutive_losses": int(longest_loss_streak),
        "avg_consecutive_losses": avg_loss_streak,
        "longest_loss_streak": int(longest_loss_streak),
        "number_of_loss_streaks_2_plus": streaks_2_plus,
        "worst_trade": float(pnl.min()),
        "worst_day": float(daily.min()) if not daily.empty else 0.0,
        "stop_hit_rate": float((trades["exit_reason"] == "stop").mean()),
        "target_hit_rate": float((trades["exit_reason"] == "target").mean()),
        "eod_exit_rate": float(trades["exit_reason"].isin(["time_exit", "eod_exit"]).mean()),
        "proportion_filtered_out": float(filtered_out_count / raw_signal_count) if raw_signal_count > 0 else 0.0,
        "trade_count_after_filters": int(len(trades)),
        "percent_of_days_traded": percent_of_days_traded,
        "percent_days_traded": percent_of_days_traded,
        "avg_R": avg_r,
        "median_R": median_r,
        "pnl_to_risk_ratio": pnl_to_risk,
        "average_loss_streak_length": avg_loss_streak,
        "count_loss_streaks_2_plus": streaks_2_plus,
        "daily_sharpe_basis": "daily_pnl_over_static_capital",
        "n_sessions": n_sessions,
        "raw_signal_count": raw_signal_count,
    }
    metrics.update(
        _compute_prop_metrics(
            trades=trades,
            daily=daily,
            resolved_session_dates=resolved_session_dates,
            capital=capital,
            prop_constraints=prop_constraints,
        )
    )
    return metrics
