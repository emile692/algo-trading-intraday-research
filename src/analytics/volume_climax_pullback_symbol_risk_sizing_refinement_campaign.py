"""Shared wrapper helpers for symbol-specific risk sizing refinement campaigns."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.analytics import volume_climax_pullback_mnq_risk_sizing_campaign as core
from src.analytics import volume_climax_pullback_mnq_risk_sizing_refinement_campaign as refinement


@dataclass(frozen=True)
class SymbolRefinementConfig:
    symbol: str
    output_prefix: str
    risk_pcts: tuple[float, ...]
    max_contracts_grid: tuple[int, ...]
    best_previous_winner_name: str
    best_previous_winner_risk_pct: float
    best_previous_winner_max_contracts: int


@contextmanager
def temporary_refinement_defaults(config: SymbolRefinementConfig) -> Iterator[None]:
    """Run the shared refinement engine with symbol-scoped defaults."""
    original_core_symbol = core.DEFAULT_SYMBOL
    original_ref_symbol = refinement.DEFAULT_SYMBOL
    original_output_prefix = refinement.DEFAULT_OUTPUT_PREFIX
    original_risk_pcts = refinement.DEFAULT_RISK_PCTS
    original_max_contracts = refinement.DEFAULT_MAX_CONTRACTS
    original_winner_name = refinement.BEST_PREVIOUS_WINNER_NAME
    original_winner_risk = refinement.BEST_PREVIOUS_WINNER_RISK_PCT
    original_winner_cap = refinement.BEST_PREVIOUS_WINNER_MAX_CONTRACTS

    try:
        core.DEFAULT_SYMBOL = str(config.symbol).upper()
        refinement.DEFAULT_SYMBOL = str(config.symbol).upper()
        refinement.DEFAULT_OUTPUT_PREFIX = str(config.output_prefix)
        refinement.DEFAULT_RISK_PCTS = tuple(float(value) for value in config.risk_pcts)
        refinement.DEFAULT_MAX_CONTRACTS = tuple(int(value) for value in config.max_contracts_grid)
        refinement.BEST_PREVIOUS_WINNER_NAME = str(config.best_previous_winner_name)
        refinement.BEST_PREVIOUS_WINNER_RISK_PCT = float(config.best_previous_winner_risk_pct)
        refinement.BEST_PREVIOUS_WINNER_MAX_CONTRACTS = int(config.best_previous_winner_max_contracts)
        yield
    finally:
        core.DEFAULT_SYMBOL = original_core_symbol
        refinement.DEFAULT_SYMBOL = original_ref_symbol
        refinement.DEFAULT_OUTPUT_PREFIX = original_output_prefix
        refinement.DEFAULT_RISK_PCTS = original_risk_pcts
        refinement.DEFAULT_MAX_CONTRACTS = original_max_contracts
        refinement.BEST_PREVIOUS_WINNER_NAME = original_winner_name
        refinement.BEST_PREVIOUS_WINNER_RISK_PCT = original_winner_risk
        refinement.BEST_PREVIOUS_WINNER_MAX_CONTRACTS = original_winner_cap


def run_symbol_refinement_campaign(
    *,
    config: SymbolRefinementConfig,
    output_root: Path | None = None,
    input_path: Path | None = None,
    initial_capital_usd: float = core.DEFAULT_INITIAL_CAPITAL_USD,
    base_variant_name: str | None = None,
    reference_v3_dir: Path | None = None,
) -> Path:
    """Delegate to the shared refinement runner with symbol overrides."""
    with temporary_refinement_defaults(config):
        return refinement.run_campaign(
            output_root=output_root,
            input_path=input_path,
            initial_capital_usd=initial_capital_usd,
            risk_pcts=config.risk_pcts,
            max_contracts_grid=config.max_contracts_grid,
            base_variant_name=base_variant_name,
            reference_v3_dir=reference_v3_dir,
        )


def add_common_refinement_args(parser: Any) -> Any:
    """Attach the shared CLI args for symbol refinement wrappers."""
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--base-variant-name", type=str, default=None)
    parser.add_argument("--reference-v3-dir", type=Path, default=None)
    return parser
