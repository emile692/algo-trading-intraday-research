"""Grid search and heatmap preparation."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.analytics.metrics import compute_metrics
from src.engine.backtester import run_backtest
from src.engine.execution_model import ExecutionModel
from src.strategy.orb import ORBStrategy


def run_orb_grid_search(
    df: pd.DataFrame,
    or_minutes_values: Iterable[int],
    target_multiple_values: Iterable[float],
) -> pd.DataFrame:
    """Run simple grid search over ORB parameters."""
    rows = []
    for or_minutes in or_minutes_values:
        for target_mult in target_multiple_values:
            strategy = ORBStrategy(or_minutes=or_minutes, target_multiple=target_mult)
            signals = strategy.generate_signals(df)
            trades = run_backtest(
                signals,
                execution_model=ExecutionModel(),
                target_multiple=target_mult,
                stop_multiple=strategy.stop_multiple,
                time_exit=strategy.time_exit,
            )
            metrics = compute_metrics(trades)
            rows.append(
                {
                    "or_minutes": or_minutes,
                    "target_multiple": target_mult,
                    "cumulative_pnl": metrics["cumulative_pnl"],
                    "win_rate": metrics["win_rate"],
                    "n_trades": metrics["n_trades"],
                }
            )
    return pd.DataFrame(rows)


def pivot_heatmap(df: pd.DataFrame, index: str, columns: str, values: str) -> pd.DataFrame:
    """Build pivot table suitable for seaborn heatmap."""
    return df.pivot(index=index, columns=columns, values=values)
