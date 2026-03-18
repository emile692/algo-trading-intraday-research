"""Performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(trades: pd.DataFrame) -> dict[str, float]:
    """Compute key strategy metrics from trade log."""
    if trades.empty:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "cumulative_pnl": 0.0,
            "max_drawdown": 0.0,
        }

    pnl = trades["net_pnl_usd"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax()

    gross_profit = wins.sum()
    gross_loss_abs = abs(losses.sum())
    profit_factor = float(gross_profit / gross_loss_abs) if gross_loss_abs > 0 else np.inf

    n_years = (trades["exit_time"].max() - trades["entry_time"].min()).days / 365.25
    trades_per_year = len(trades) / n_years
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(trades_per_year) if pnl.std() > 0 else 0.0

    return {
        "n_trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
        "expectancy": float(pnl.mean()),
        "profit_factor": float(profit_factor),
        "cumulative_pnl": float(pnl.sum()),
        "max_drawdown": float(drawdown.min()),
        "sharpe_ratio": float(sharpe),
    }
