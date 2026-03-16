"""Diagnostics grouped by calendar dimensions."""

from __future__ import annotations

import pandas as pd


def performance_by_weekday(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate performance by exit weekday."""
    out = trades.copy()
    out["weekday"] = pd.to_datetime(out["exit_time"]).dt.day_name()
    return out.groupby("weekday", as_index=False)["net_pnl_usd"].agg(["count", "sum", "mean"]).reset_index()


def performance_by_month(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate performance by exit month."""
    out = trades.copy()
    out["month"] = pd.to_datetime(out["exit_time"]).dt.to_period("M").astype(str)
    return out.groupby("month", as_index=False)["net_pnl_usd"].agg(["count", "sum", "mean"]).reset_index()


def performance_by_year(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate performance by exit year."""
    out = trades.copy()
    out["year"] = pd.to_datetime(out["exit_time"]).dt.year
    return out.groupby("year", as_index=False)["net_pnl_usd"].agg(["count", "sum", "mean"]).reset_index()
