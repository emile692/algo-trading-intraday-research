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

INSTRUMENT_SPECS = {
    # --- E-mini Nasdaq (NQ) ---
    "NQ": {
        "tick_size": 0.25,
        "tick_value_usd": 5.0,
        "point_value_usd": 20.0,
        "commission_per_side_usd": 1.25,
        "slippage_ticks": 1,
    },
    # --- Micro E-mini Nasdaq (MNQ) ---
    "MNQ": {
        "tick_size": 0.25,
        "tick_value_usd": 0.5,
        "point_value_usd": 2.0,
        "commission_per_side_usd": 1.25,
        "slippage_ticks": 1,
    },
    # --- E-mini S&P 500 (ES) ---
    "ES": {
        "tick_size": 0.25,
        "tick_value_usd": 12.5,
        "point_value_usd": 50.0,
        "commission_per_side_usd": 1.25,
        "slippage_ticks": 1,
    },
    # --- Micro E-mini S&P 500 (MES) ---
    "MES": {
        "tick_size": 0.25,
        "tick_value_usd": 1.25,
        "point_value_usd": 5.0,
        "commission_per_side_usd": 1.25,
        "slippage_ticks": 1,
    },
    # --- E-mini Russell 2000 (RTY) ---
    "RTY": {
        "tick_size": 0.10,
        "tick_value_usd": 5.0,
        "point_value_usd": 50.0,
        "commission_per_side_usd": 1.25,
        "slippage_ticks": 1,
    },
    # --- Micro E-mini Russell 2000 (M2K) ---
    "M2K": {
        "tick_size": 0.10,
        "tick_value_usd": 0.5,
        "point_value_usd": 5.0,
        "commission_per_side_usd": 1.25,
        "slippage_ticks": 1,
    },
    # --- Micro Gold (MGC) ---
    "MGC": {
        "tick_size": 0.10,
        "tick_value_usd": 1.0,
        "point_value_usd": 10.0,
        "commission_per_side_usd": 1.25,
        "slippage_ticks": 1,
    },
}

# =========================
# Default instrument
# =========================
DEFAULT_SYMBOL = "MNQ"

# Keep backward-compatible defaults
DEFAULT_TICK_SIZE = INSTRUMENT_SPECS[DEFAULT_SYMBOL]["tick_size"]
DEFAULT_TICK_VALUE_USD = INSTRUMENT_SPECS[DEFAULT_SYMBOL]["tick_value_usd"]
DEFAULT_POINT_VALUE_USD = INSTRUMENT_SPECS[DEFAULT_SYMBOL]["point_value_usd"]

# =========================
# Execution costs
# =========================
DEFAULT_COMMISSION_PER_SIDE_USD = INSTRUMENT_SPECS[DEFAULT_SYMBOL]["commission_per_side_usd"]
DEFAULT_SLIPPAGE_TICKS = INSTRUMENT_SPECS[DEFAULT_SYMBOL]["slippage_ticks"]

# =========================
# Backtest defaults
# =========================
DEFAULT_INITIAL_CAPITAL_USD = 50_000.0


def get_instrument_spec(symbol: str) -> dict:
    """Return execution / contract specs for a given futures symbol."""
    key = symbol.upper()
    if key not in INSTRUMENT_SPECS:
        raise ValueError(
            f"Unknown instrument '{symbol}'. "
            f"Available: {', '.join(sorted(INSTRUMENT_SPECS))}"
        )
    return INSTRUMENT_SPECS[key].copy()