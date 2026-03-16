"""Equity and drawdown visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(equity_df: pd.DataFrame) -> None:
    """Plot equity curve."""
    plt.figure(figsize=(10, 4))
    plt.plot(equity_df["timestamp"], equity_df["equity"])
    plt.title("Equity Curve")
    plt.ylabel("Equity (USD)")
    plt.tight_layout()


def plot_drawdown_curve(equity_df: pd.DataFrame) -> None:
    """Plot drawdown curve."""
    plt.figure(figsize=(10, 3))
    plt.plot(equity_df["timestamp"], equity_df["drawdown"])
    plt.title("Drawdown")
    plt.ylabel("USD")
    plt.tight_layout()
