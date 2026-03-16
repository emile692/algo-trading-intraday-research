"""Portfolio and equity computations."""

from __future__ import annotations

import pandas as pd

from src.config.settings import DEFAULT_INITIAL_CAPITAL_USD


def build_equity_curve(trades: pd.DataFrame, initial_capital: float = DEFAULT_INITIAL_CAPITAL_USD) -> pd.DataFrame:
    """Build cumulative equity and drawdown series from trades."""
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "drawdown"]) 

    out = trades.sort_values("exit_time").copy()
    out["cum_pnl"] = out["net_pnl_usd"].cumsum()
    out["equity"] = initial_capital + out["cum_pnl"]
    out["peak_equity"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] - out["peak_equity"]
    return out[["exit_time", "equity", "drawdown"]].rename(columns={"exit_time": "timestamp"})
