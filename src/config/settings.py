"""Global settings for the research framework."""

from __future__ import annotations

# =========================
# Timezone
# =========================
DEFAULT_TIMEZONE = "America/New_York"

# =========================
# Sessions (US Eastern Time)
# =========================
RTH_START = "09:30"
RTH_END = "16:00"
ETH_START = "18:00"
ETH_END = "17:00"

# =========================
# Instrument specifications
# =========================

# --- E-mini Nasdaq (NQ) ---
NQ_TICK_SIZE = 0.25
NQ_TICK_VALUE_USD = 5.0
NQ_POINT_VALUE_USD = 20.0

# --- Micro E-mini Nasdaq (MNQ) ---
MNQ_TICK_SIZE = 0.25
MNQ_TICK_VALUE_USD = 0.5
MNQ_POINT_VALUE_USD = 2.0

# =========================
# Default instrument (change here depending on your backtest)
# =========================
DEFAULT_TICK_SIZE = MNQ_TICK_SIZE
DEFAULT_TICK_VALUE_USD = MNQ_TICK_VALUE_USD
DEFAULT_POINT_VALUE_USD = MNQ_POINT_VALUE_USD

# =========================
# Execution costs
# =========================
DEFAULT_COMMISSION_PER_SIDE_USD = 1.25
DEFAULT_SLIPPAGE_TICKS = 1

# =========================
# Backtest defaults
# =========================
DEFAULT_INITIAL_CAPITAL_USD = 50_000.0