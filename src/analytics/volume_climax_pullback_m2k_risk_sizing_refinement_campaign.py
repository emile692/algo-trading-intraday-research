"""Local refinement campaign around the best M2K risk-sizing zone."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analytics.volume_climax_pullback_symbol_risk_sizing_refinement_campaign import (
    SymbolRefinementConfig,
    add_common_refinement_args,
    run_symbol_refinement_campaign,
)

CONFIG = SymbolRefinementConfig(
    symbol="M2K",
    output_prefix="volume_climax_pullback_m2k_risk_sizing_refinement_",
    risk_pcts=(0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040),
    max_contracts_grid=(2, 3, 4, 5, 6),
    best_previous_winner_name="risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true",
    best_previous_winner_risk_pct=0.0025,
    best_previous_winner_max_contracts=3,
)


def run_campaign(
    *,
    output_root: Path | None = None,
    input_path: Path | None = None,
    initial_capital_usd: float = 50_000.0,
    base_variant_name: str | None = None,
    reference_v3_dir: Path | None = None,
) -> Path:
    return run_symbol_refinement_campaign(
        config=CONFIG,
        output_root=output_root,
        input_path=input_path,
        initial_capital_usd=initial_capital_usd,
        base_variant_name=base_variant_name,
        reference_v3_dir=reference_v3_dir,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    return add_common_refinement_args(parser)


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    output_dir = run_campaign(
        output_root=args.output_root,
        input_path=args.input_path,
        base_variant_name=args.base_variant_name,
        reference_v3_dir=args.reference_v3_dir,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
