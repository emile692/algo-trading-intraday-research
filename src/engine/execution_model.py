"""Execution and cost model."""

from __future__ import annotations

from dataclasses import dataclass

from src.config.settings import DEFAULT_COMMISSION_PER_SIDE_USD, DEFAULT_SLIPPAGE_TICKS, NQ_TICK_SIZE


@dataclass
class ExecutionModel:
    """Simple fixed commission + fixed slippage model."""

    commission_per_side_usd: float = DEFAULT_COMMISSION_PER_SIDE_USD
    slippage_ticks: int = DEFAULT_SLIPPAGE_TICKS
    tick_size: float = NQ_TICK_SIZE

    def apply_slippage(self, price: float, direction: int, is_entry: bool) -> float:
        """Adjust price by slippage in unfavorable direction."""
        slippage_points = self.slippage_ticks * self.tick_size
        if direction == 1:
            return price + slippage_points if is_entry else price - slippage_points
        return price - slippage_points if is_entry else price + slippage_points

    def round_trip_fees(self) -> float:
        """Return total round-trip fees in USD."""
        return 2.0 * self.commission_per_side_usd
