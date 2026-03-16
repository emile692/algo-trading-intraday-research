"""General strategy plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_trade_histogram(trades: pd.DataFrame) -> None:
    """Plot histogram of net trade PnL."""
    plt.figure(figsize=(8, 4))
    plt.hist(trades["net_pnl_usd"], bins=30)
    plt.title("Trade PnL Histogram")
    plt.xlabel("Net PnL (USD)")
    plt.ylabel("Count")
    plt.tight_layout()


def plot_pnl_distribution(trades: pd.DataFrame) -> None:
    """Plot KDE-like cumulative distribution proxy for PnL."""
    plt.figure(figsize=(8, 4))
    trades["net_pnl_usd"].sort_values().reset_index(drop=True).plot()
    plt.title("Sorted Trade PnL")
    plt.ylabel("Net PnL (USD)")
    plt.tight_layout()


def plot_cumulative_pnl(trades: pd.DataFrame) -> None:
    """Plot cumulative net PnL over exits."""
    plt.figure(figsize=(10, 4))
    trades = trades.sort_values("exit_time")
    trades["net_pnl_usd"].cumsum().plot()
    plt.title("Cumulative Net PnL")
    plt.ylabel("USD")
    plt.tight_layout()
