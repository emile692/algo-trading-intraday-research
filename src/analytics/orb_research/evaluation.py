"""Extended evaluation metrics and ranking helpers for ORB research."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.analytics.metrics import compute_metrics


def _daily_pnl_from_trades(trades: pd.DataFrame, sessions: list) -> pd.Series:
    idx = pd.Index(pd.to_datetime(pd.Index(sessions)).date)
    if idx.empty:
        return pd.Series(dtype=float)
    if trades.empty:
        return pd.Series(0.0, index=idx, dtype=float)

    grouped = trades.groupby(pd.to_datetime(trades["session_date"]).dt.date)["net_pnl_usd"].sum()
    return grouped.reindex(idx, fill_value=0.0).astype(float)


def _run_lengths(mask: pd.Series) -> list[int]:
    lengths: list[int] = []
    current = 0
    for flag in mask.astype(bool).tolist():
        if flag:
            current += 1
        elif current > 0:
            lengths.append(current)
            current = 0
    if current > 0:
        lengths.append(current)
    return lengths


def _challenge_outcome(
    pnl_path: np.ndarray,
    target_usd: float,
    max_dd_usd: float,
) -> tuple[str, int]:
    cum = 0.0
    for i, value in enumerate(pnl_path, start=1):
        cum += float(value)
        if cum >= target_usd:
            return "success", i
        if cum <= -max_dd_usd:
            return "fail", i
    return "open", len(pnl_path)


def simulate_prop_challenge(
    daily_pnl: pd.Series,
    initial_capital: float,
    target_return_pct: float = 0.06,
    max_drawdown_pct: float = 0.04,
    bootstrap_paths: int = 3000,
    random_seed: int = 42,
) -> dict[str, float]:
    series = pd.to_numeric(daily_pnl, errors="coerce").fillna(0.0)
    if series.empty:
        return {
            "challenge_target_return_pct": target_return_pct,
            "challenge_drawdown_limit_pct": max_drawdown_pct,
            "challenge_target_usd": initial_capital * target_return_pct,
            "challenge_drawdown_limit_usd": initial_capital * max_drawdown_pct,
            "challenge_hit_probability": 0.0,
            "challenge_hit_probability_bootstrap": 0.0,
            "challenge_hit_probability_rolling": 0.0,
            "challenge_fail_fast_probability": 0.0,
            "challenge_median_days_to_target": np.nan,
            "challenge_median_days_to_target_bootstrap": np.nan,
            "challenge_median_days_to_target_rolling": np.nan,
            "challenge_success_paths_bootstrap": 0.0,
            "challenge_success_paths_rolling": 0.0,
        }

    target_usd = float(initial_capital * target_return_pct)
    max_dd_usd = float(initial_capital * max_drawdown_pct)
    values = series.to_numpy(dtype=float)

    rolling_days: list[int] = []
    rolling_success = 0
    rolling_fail_fast = 0
    for start in range(len(values)):
        outcome, days = _challenge_outcome(values[start:], target_usd=target_usd, max_dd_usd=max_dd_usd)
        if outcome == "success":
            rolling_success += 1
            rolling_days.append(days)
        if outcome == "fail" and days <= 5:
            rolling_fail_fast += 1

    rolling_total = max(1, len(values))
    rolling_rate = rolling_success / rolling_total
    fail_fast_rate = rolling_fail_fast / rolling_total

    rng = np.random.default_rng(random_seed)
    bootstrap_days: list[int] = []
    bootstrap_success = 0
    horizon = len(values)
    for _ in range(max(0, int(bootstrap_paths))):
        sampled = rng.choice(values, size=horizon, replace=True)
        outcome, days = _challenge_outcome(sampled, target_usd=target_usd, max_dd_usd=max_dd_usd)
        if outcome == "success":
            bootstrap_success += 1
            bootstrap_days.append(days)

    bootstrap_rate = bootstrap_success / bootstrap_paths if bootstrap_paths > 0 else 0.0
    combined_rate = 0.5 * rolling_rate + 0.5 * bootstrap_rate

    return {
        "challenge_target_return_pct": target_return_pct,
        "challenge_drawdown_limit_pct": max_drawdown_pct,
        "challenge_target_usd": target_usd,
        "challenge_drawdown_limit_usd": max_dd_usd,
        "challenge_hit_probability": float(combined_rate),
        "challenge_hit_probability_bootstrap": float(bootstrap_rate),
        "challenge_hit_probability_rolling": float(rolling_rate),
        "challenge_fail_fast_probability": float(fail_fast_rate),
        "challenge_median_days_to_target": float(np.median(bootstrap_days)) if bootstrap_days else np.nan,
        "challenge_median_days_to_target_bootstrap": float(np.median(bootstrap_days)) if bootstrap_days else np.nan,
        "challenge_median_days_to_target_rolling": float(np.median(rolling_days)) if rolling_days else np.nan,
        "challenge_success_paths_bootstrap": float(bootstrap_success),
        "challenge_success_paths_rolling": float(rolling_success),
    }


def compute_extended_metrics(
    trades: pd.DataFrame,
    signal_df: pd.DataFrame | None,
    sessions: list,
    initial_capital: float,
    bootstrap_paths: int,
    random_seed: int,
) -> dict[str, float | int | str | bool]:
    base = compute_metrics(
        trades,
        signal_df=signal_df,
        session_dates=sessions,
        initial_capital=initial_capital,
    )

    daily = _daily_pnl_from_trades(trades, sessions)
    cumulative_daily = daily.cumsum()
    equity = initial_capital + cumulative_daily
    peak = equity.cummax()
    drawdown = equity - peak

    if len(daily) >= 3:
        worst_3day_run = float(daily.rolling(3).sum().min())
    else:
        worst_3day_run = float(daily.sum()) if not daily.empty else 0.0

    if len(daily) >= 5:
        worst_5day_drawdown = float(daily.rolling(5).sum().min())
    else:
        worst_5day_drawdown = float(daily.sum()) if not daily.empty else 0.0

    avg_daily_pnl = float(daily.mean()) if not daily.empty else 0.0
    worst_day = float(daily.min()) if not daily.empty else 0.0

    losing_streaks = _run_lengths((daily < 0).astype(bool))
    max_losing_streak_days = max(losing_streaks, default=0)

    if not trades.empty:
        hold_minutes = (
            (pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"]))
            .dt.total_seconds()
            .div(60.0)
        )
        avg_hold_minutes = float(hold_minutes.mean())
    else:
        avg_hold_minutes = 0.0

    return_over_drawdown = float(base.get("cumulative_pnl", 0.0)) / max(abs(float(base.get("max_drawdown", 0.0))), 1.0)

    underwater = drawdown < 0
    underwater_pct = float(underwater.mean()) if len(underwater) > 0 else 0.0
    ulcer_index = float(np.sqrt(np.mean(np.square((drawdown / initial_capital) * 100.0)))) if len(drawdown) > 0 else 0.0

    challenge = simulate_prop_challenge(
        daily_pnl=daily,
        initial_capital=initial_capital,
        target_return_pct=0.06,
        max_drawdown_pct=0.04,
        bootstrap_paths=bootstrap_paths,
        random_seed=random_seed,
    )

    out: dict[str, float | int | str | bool] = dict(base)
    out.update(
        {
            "net_pnl": float(base.get("cumulative_pnl", 0.0)),
            "return_over_drawdown": return_over_drawdown,
            "avg_daily_pnl": avg_daily_pnl,
            "worst_day": worst_day,
            "worst_3day_run": worst_3day_run,
            "worst_5day_drawdown": worst_5day_drawdown,
            "max_losing_streak": int(base.get("longest_loss_streak", 0)),
            "max_losing_streak_days": int(max_losing_streak_days),
            "nb_trades": int(base.get("n_trades", 0)),
            "pct_days_traded": float(base.get("percent_of_days_traded", 0.0)),
            "avg_time_in_position_min": avg_hold_minutes,
            "hit_ratio": float(base.get("win_rate", 0.0)),
            "avg_winner": float(base.get("avg_win", 0.0)),
            "avg_loser": float(base.get("avg_loss", 0.0)),
            "time_underwater_pct": underwater_pct,
            "ulcer_index": ulcer_index,
        }
    )
    out.update(challenge)
    return out


def compute_scores(metrics_row: pd.Series) -> dict[str, float]:
    """Compute academic and prop-oriented ranking scores."""
    sharpe = float(metrics_row.get("sharpe_ratio", 0.0))
    pf = float(metrics_row.get("profit_factor", 0.0))
    expectancy = float(metrics_row.get("expectancy", 0.0))
    ret_dd = float(metrics_row.get("return_over_drawdown", 0.0))

    max_dd = abs(float(metrics_row.get("max_drawdown", 0.0)))
    worst_5d = abs(float(metrics_row.get("worst_5day_drawdown", 0.0)))
    losing_streak = float(metrics_row.get("max_losing_streak", 0.0))
    fail_fast = float(metrics_row.get("challenge_fail_fast_probability", 0.0))
    challenge_prob = float(metrics_row.get("challenge_hit_probability", 0.0))
    median_days = pd.to_numeric(pd.Series([metrics_row.get("challenge_median_days_to_target")]), errors="coerce").iloc[0]
    pnl_oos = float(metrics_row.get("net_pnl", 0.0))

    pf_capped = pf if math.isfinite(pf) else 3.0

    academic_score = (
        0.45 * sharpe
        + 0.25 * (pf_capped - 1.0)
        + 0.20 * np.tanh(expectancy / 50.0)
        + 0.10 * np.tanh(ret_dd / 4.0)
    )

    speed_bonus = 0.0
    if pd.notna(median_days) and float(median_days) > 0:
        speed_bonus = float(np.clip(40.0 / float(median_days), 0.0, 2.0))

    prop_score = (
        2.2 * np.tanh(ret_dd / 3.0)
        + 2.4 * challenge_prob
        + 0.6 * np.tanh((pf_capped - 1.0) / 0.4)
        + 0.5 * np.tanh(expectancy / 40.0)
        + 0.3 * np.tanh(pnl_oos / 4000.0)
        + 0.7 * speed_bonus
        - 1.8 * np.tanh(max_dd / 2500.0)
        - 1.4 * np.tanh(worst_5d / 1800.0)
        - 0.9 * np.tanh(max(losing_streak - 2.0, 0.0) / 5.0)
        - 1.2 * fail_fast
    )

    return {
        "academic_score": float(academic_score),
        "prop_score": float(prop_score),
    }
