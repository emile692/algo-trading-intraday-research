"""Dedicated M2K risk-sizing campaign for Volume Climax Pullback."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analytics.volume_climax_pullback_symbol_risk_sizing_campaign import (
    add_common_campaign_args,
    run_symbol_campaign,
)

DEFAULT_SYMBOL = "M2K"
DEFAULT_OUTPUT_PREFIX = "volume_climax_pullback_m2k_risk_sizing_"


def run_campaign(
    *,
    output_root: Path | None = None,
    input_path: Path | None = None,
    initial_capital_usd: float = 50_000.0,
    risk_pcts: tuple[float, ...] = (0.0025, 0.0050, 0.0075, 0.0100),
    max_contracts_grid: tuple[int, ...] = (3, 5, 10, 15),
    skip_flags: tuple[bool, ...] = (True, False),
    base_variant_name: str | None = None,
    reference_v3_dir: Path | None = None,
) -> Path:
    return run_symbol_campaign(
        symbol=DEFAULT_SYMBOL,
        output_prefix=DEFAULT_OUTPUT_PREFIX,
        output_root=output_root,
        input_path=input_path,
        initial_capital_usd=initial_capital_usd,
        risk_pcts=risk_pcts,
        max_contracts_grid=max_contracts_grid,
        skip_flags=skip_flags,
        base_variant_name=base_variant_name,
        reference_v3_dir=reference_v3_dir,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    return add_common_campaign_args(parser)


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
