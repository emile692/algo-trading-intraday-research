"""Extended metrics and research breakdowns for the VWAP campaign."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.metrics import compute_metrics
from src.config.vwap_campaign import PropFirmConstraintConfig
from src.engine.portfolio import build_equity_curve


def _negative_streak_lengths(values: pd.Series) -> list[int]:
    streaks: list[int] = []
    current = 0
    for value in pd.Series(values, dtype=float).fillna(0.0):
        if value < 0:
            current += 1
        elif current > 0:
            streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return streaks


def _sortino_ratio(daily_pnl: pd.Series, capital: float) -> float:
    if capital <= 0:
        return 0.0
    daily_returns = pd.Series(daily_pnl, dtype=float) / capital
    downside = daily_returns[daily_returns < 0]
    if len(daily_returns) < 2 or downside.empty:
        return 0.0
    downside_std = float(np.sqrt(np.mean(np.square(downside))))
    if downside_std == 0:
        return 0.0
    return float((daily_returns.mean() / downside_std) * math.sqrt(252.0))


def _session_intraday_drawdown(bar_results: pd.DataFrame) -> pd.Series:
    if bar_results.empty:
        return pd.Series(dtype=float)

    out = bar_results.copy()
    out["bar_net"] = pd.to_numeric(out["net_bar_pnl_usd"], errors="coerce").fillna(0.0)
    out["cum_intra"] = out.groupby("session_date")["bar_net"].cumsum()
    out["peak_intra"] = out.groupby("session_date")["cum_intra"].cummax()
    out["dd_intra"] = out["cum_intra"] - out["peak_intra"]
    session_dd = out.groupby("session_date")["dd_intra"].min()
    session_dd.index = pd.Index(pd.to_datetime(session_dd.index).date)
    return session_dd


def build_pnl_by_hour_table(bar_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday PnL by hourly bucket."""
    if bar_results.empty:
        return pd.DataFrame(columns=["hour", "bars", "net_bar_pnl_usd", "avg_bar_pnl_usd"])

    out = bar_results.copy()
    out["hour"] = pd.to_datetime(out["timestamp"]).dt.strftime("%H:00")
    grouped = (
        out.groupby("hour", as_index=False)["net_bar_pnl_usd"]
        .agg(bars="count", net_bar_pnl_usd="sum", avg_bar_pnl_usd="mean")
        .sort_values("hour")
        .reset_index(drop=True)
    )
    return grouped


def build_trade_hour_table(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade outcomes by entry hour."""
    if trades.empty:
        return pd.DataFrame(columns=["hour", "trades", "net_pnl_usd", "avg_trade_pnl_usd", "win_rate"])

    out = trades.copy()
    out["hour"] = pd.to_datetime(out["entry_time"]).dt.strftime("%H:00")
    out["is_win"] = pd.to_numeric(out["net_pnl_usd"], errors="coerce").gt(0)
    grouped = (
        out.groupby("hour", as_index=False)
        .agg(
            trades=("trade_id", "count"),
            net_pnl_usd=("net_pnl_usd", "sum"),
            avg_trade_pnl_usd=("net_pnl_usd", "mean"),
            win_rate=("is_win", "mean"),
        )
        .sort_values("hour")
        .reset_index(drop=True)
    )
    return grouped


def build_long_short_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """Split trade performance by direction."""
    if trades.empty:
        return pd.DataFrame(
            columns=["direction", "trades", "net_pnl_usd", "avg_trade_pnl_usd", "win_rate", "profit_factor"]
        )

    rows: list[dict[str, Any]] = []
    for direction, frame in trades.groupby("direction", sort=True):
        pnl = pd.to_numeric(frame["net_pnl_usd"], errors="coerce").fillna(0.0)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        gross_loss_abs = abs(float(losses.sum()))
        profit_factor = float(wins.sum() / gross_loss_abs) if gross_loss_abs > 0 else np.inf
        rows.append(
            {
                "direction": direction,
                "trades": int(len(frame)),
                "net_pnl_usd": float(pnl.sum()),
                "avg_trade_pnl_usd": float(pnl.mean()) if len(frame) > 0 else 0.0,
                "win_rate": float((pnl > 0).mean()) if len(frame) > 0 else 0.0,
                "profit_factor": profit_factor,
            }
        )
    return pd.DataFrame(rows).sort_values("direction").reset_index(drop=True)


def build_weekday_pnl_table(daily_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily PnL by weekday."""
    if daily_results.empty:
        return pd.DataFrame(columns=["weekday", "days", "net_pnl_usd", "avg_day_pnl_usd", "green_day_pct"])

    out = daily_results.copy()
    out["weekday"] = pd.to_datetime(out["session_date"]).dt.day_name()
    grouped = (
        out.groupby("weekday", as_index=False)
        .agg(
            days=("session_date", "count"),
            net_pnl_usd=("daily_pnl_usd", "sum"),
            avg_day_pnl_usd=("daily_pnl_usd", "mean"),
            green_day_pct=("green_day", "mean"),
        )
        .reset_index(drop=True)
    )
    ordered = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    grouped["weekday_order"] = grouped["weekday"].map({name: idx for idx, name in enumerate(ordered)})
    grouped = grouped.sort_values("weekday_order").drop(columns=["weekday_order"]).reset_index(drop=True)
    return grouped


def build_rolling_metric_table(
    daily_results: pd.DataFrame,
    initial_capital: float,
    window_days: int = 20,
) -> pd.DataFrame:
    """Compute rolling daily Sharpe and expectancy series."""
    if daily_results.empty:
        return pd.DataFrame(columns=["session_date", "rolling_sharpe", "rolling_expectancy"])

    out = daily_results[["session_date", "daily_pnl_usd"]].copy()
    out = out.sort_values("session_date").reset_index(drop=True)
    if initial_capital > 0:
        daily_returns = out["daily_pnl_usd"] / initial_capital
        rolling_mean = daily_returns.rolling(window_days).mean()
        rolling_std = daily_returns.rolling(window_days).std(ddof=0)
        out["rolling_sharpe"] = np.where(
            rolling_std > 0,
            (rolling_mean / rolling_std) * math.sqrt(252.0),
            np.nan,
        )
    else:
        out["rolling_sharpe"] = np.nan
    out["rolling_expectancy"] = out["daily_pnl_usd"].rolling(window_days).mean()
    return out


def compute_prop_viability_metrics(
    daily_results: pd.DataFrame,
    cumulative_pnl: float,
    max_drawdown: float,
    constraints: PropFirmConstraintConfig,
) -> dict[str, Any]:
    """Estimate prop-firm viability from the realized daily path."""
    if daily_results.empty:
        return {
            "days_to_target_pct": np.nan,
            "daily_loss_limit_breach_freq": 0.0,
            "trailing_drawdown_breach_freq": 0.0,
            "empirical_prob_red_streak_ge_threshold": 0.0,
            "profit_to_drawdown_ratio": 0.0,
        }

    out = daily_results.copy().sort_values("session_date").reset_index(drop=True)
    target_usd = constraints.account_size_usd * constraints.profit_target_pct
    cumulative = out["daily_pnl_usd"].cumsum()
    target_hits = np.flatnonzero(cumulative >= target_usd)
    days_to_target = int(target_hits[0] + 1) if len(target_hits) > 0 else np.nan

    equity = constraints.account_size_usd + cumulative
    peak = equity.cummax()
    trailing_drawdown = peak - equity

    red_streaks = _negative_streak_lengths(out["daily_pnl_usd"])
    threshold = int(constraints.consecutive_red_days_threshold)
    prob_red_streak = float(sum(length >= threshold for length in red_streaks) / len(red_streaks)) if red_streaks else 0.0

    profit_to_drawdown = float(cumulative_pnl / abs(max_drawdown)) if max_drawdown < 0 else np.inf

    return {
        "days_to_target_pct": days_to_target,
        "daily_loss_limit_breach_freq": float(
            (pd.to_numeric(out["daily_pnl_usd"], errors="coerce") <= -constraints.daily_loss_limit_usd).mean()
        ),
        "trailing_drawdown_breach_freq": float(
            (pd.to_numeric(trailing_drawdown, errors="coerce") >= constraints.trailing_drawdown_limit_usd).mean()
        ),
        "empirical_prob_red_streak_ge_threshold": prob_red_streak,
        "profit_to_drawdown_ratio": profit_to_drawdown,
    }


def compute_extended_vwap_metrics(
    trades: pd.DataFrame,
    daily_results: pd.DataFrame,
    bar_results: pd.DataFrame,
    signal_df: pd.DataFrame | None,
    initial_capital: float,
    prop_constraints: PropFirmConstraintConfig,
    rolling_window_days: int = 20,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute the full metrics dictionary plus the rolling 20-day table."""
    session_dates = daily_results["session_date"] if not daily_results.empty else None
    metrics = compute_metrics(
        trades,
        signal_df=signal_df,
        session_dates=session_dates,
        initial_capital=initial_capital,
    )

    daily_pnl = pd.to_numeric(daily_results.get("daily_pnl_usd"), errors="coerce").fillna(0.0) if not daily_results.empty else pd.Series(dtype=float)
    daily_trade_count = (
        pd.to_numeric(daily_results.get("daily_trade_count"), errors="coerce").fillna(0.0)
        if not daily_results.empty
        else pd.Series(dtype=float)
    )
    intraday_dd = _session_intraday_drawdown(bar_results)
    trade_holding = pd.to_numeric(trades.get("holding_minutes"), errors="coerce") if not trades.empty else pd.Series(dtype=float)

    rolling_table = build_rolling_metric_table(
        daily_results=daily_results,
        initial_capital=initial_capital,
        window_days=rolling_window_days,
    )

    worst_trade_streak = max(_negative_streak_lengths(pd.to_numeric(trades.get("net_pnl_usd"), errors="coerce")), default=0)
    worst_day_streak = max(_negative_streak_lengths(daily_pnl), default=0)
    avg_gain_loss_ratio = (
        float(metrics["avg_win"] / abs(metrics["avg_loss"]))
        if float(metrics.get("avg_loss", 0.0)) < 0
        else np.inf
    )

    extended = {
        **metrics,
        "net_pnl": float(metrics.get("cumulative_pnl", 0.0)),
        "sortino_ratio": _sortino_ratio(daily_pnl, initial_capital),
        "max_daily_drawdown": float(intraday_dd.min()) if not intraday_dd.empty else 0.0,
        "worst_day": float(daily_pnl.min()) if not daily_pnl.empty else 0.0,
        "worst_losing_days_streak": int(worst_day_streak),
        "worst_losing_trades_streak": int(worst_trade_streak),
        "avg_trades_per_day": float(daily_trade_count.mean()) if not daily_trade_count.empty else 0.0,
        "max_trades_per_day": int(daily_trade_count.max()) if not daily_trade_count.empty else 0,
        "expectancy_per_trade": float(metrics.get("expectancy", 0.0)),
        "expectancy_per_day": float(daily_pnl.mean()) if not daily_pnl.empty else 0.0,
        "pct_green_days": float((daily_pnl > 0).mean()) if not daily_pnl.empty else 0.0,
        "avg_gain_loss_ratio": avg_gain_loss_ratio,
        "hit_rate": float(metrics.get("win_rate", 0.0)),
        "avg_time_in_position_min": float(trade_holding.mean()) if not trade_holding.empty else 0.0,
        "rolling_20d_sharpe_median": float(pd.to_numeric(rolling_table["rolling_sharpe"], errors="coerce").median())
        if not rolling_table.empty
        else np.nan,
        "rolling_20d_expectancy_median": float(pd.to_numeric(rolling_table["rolling_expectancy"], errors="coerce").median())
        if not rolling_table.empty
        else np.nan,
    }
    extended.update(
        compute_prop_viability_metrics(
            daily_results=daily_results,
            cumulative_pnl=float(metrics.get("cumulative_pnl", 0.0)),
            max_drawdown=float(metrics.get("max_drawdown", 0.0)),
            constraints=prop_constraints,
        )
    )
    return extended, rolling_table


def build_export_tables(
    trades: pd.DataFrame,
    daily_results: pd.DataFrame,
    bar_results: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return the standard breakdown tables exported by the campaign."""
    equity_curve = build_equity_curve(trades, initial_capital=float(trades["account_size_usd"].dropna().iloc[0])) if not trades.empty and "account_size_usd" in trades.columns and trades["account_size_usd"].dropna().any() else build_equity_curve(trades)
    return {
        "equity_curve": equity_curve,
        "hourly_pnl": build_pnl_by_hour_table(bar_results),
        "trade_hourly": build_trade_hour_table(trades),
        "long_short": build_long_short_stats(trades),
        "weekday_pnl": build_weekday_pnl_table(daily_results),
    }
