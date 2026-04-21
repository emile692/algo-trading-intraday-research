"""Shared wrapper helpers for symbol-specific risk sizing campaigns."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.analytics import volume_climax_pullback_mnq_risk_sizing_campaign as core


@contextmanager
def temporary_symbol_defaults(*, symbol: str, output_prefix: str) -> Iterator[None]:
    """Run the shared campaign engine with symbol-scoped defaults."""
    original_symbol = core.DEFAULT_SYMBOL
    original_prefix = core.DEFAULT_OUTPUT_PREFIX
    try:
        core.DEFAULT_SYMBOL = str(symbol).upper()
        core.DEFAULT_OUTPUT_PREFIX = str(output_prefix)
        yield
    finally:
        core.DEFAULT_SYMBOL = original_symbol
        core.DEFAULT_OUTPUT_PREFIX = original_prefix


def run_symbol_campaign(
    *,
    symbol: str,
    output_prefix: str,
    output_root: Path | None = None,
    input_path: Path | None = None,
    initial_capital_usd: float = core.DEFAULT_INITIAL_CAPITAL_USD,
    risk_pcts: tuple[float, ...] = core.DEFAULT_RISK_PCTS,
    max_contracts_grid: tuple[int, ...] = core.DEFAULT_MAX_CONTRACTS,
    skip_flags: tuple[bool, ...] = core.DEFAULT_SKIP_FLAGS,
    base_variant_name: str | None = None,
    reference_v3_dir: Path | None = None,
) -> Path:
    """Delegate to the shared risk-sizing runner with a symbol override."""
    with temporary_symbol_defaults(symbol=symbol, output_prefix=output_prefix):
        return core.run_campaign(
            output_root=output_root,
            input_path=input_path,
            initial_capital_usd=initial_capital_usd,
            risk_pcts=risk_pcts,
            max_contracts_grid=max_contracts_grid,
            skip_flags=skip_flags,
            base_variant_name=base_variant_name,
            reference_v3_dir=reference_v3_dir,
        )


def add_common_campaign_args(parser: Any) -> Any:
    """Attach the shared CLI arguments used by every symbol wrapper."""
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--base-variant-name", type=str, default=None)
    parser.add_argument("--reference-v3-dir", type=Path, default=None)
    return parser
