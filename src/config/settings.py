"""Global settings for the research framework."""

from __future__ import annotations

DEFAULT_TIMEZONE = "America/New_York"

# Session defaults (US Eastern Time)
RTH_START = "09:30"
RTH_END = "16:00"
ETH_START = "18:00"
ETH_END = "17:00"

# Market / instrument defaults
NQ_TICK_SIZE = 0.25
DEFAULT_TICK_VALUE_USD = 5.0

# Execution costs
DEFAULT_COMMISSION_PER_SIDE_USD = 1.25
DEFAULT_SLIPPAGE_TICKS = 1

# Backtest defaults
DEFAULT_INITIAL_CAPITAL_USD = 100_000.0
