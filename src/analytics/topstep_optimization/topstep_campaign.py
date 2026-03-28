"""Topstep 50K optimization campaign focused on expected economic return."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics.mnq_orb_prop_challenge_simulation import (
    DEFAULT_PRIMARY_SCOPE,
    DEFAULT_RANDOM_SEED,
    _find_latest_source_run,
    _json_dump,
    _load_variant_input,
    _read_run_metadata,
    _scope_daily_results,
    _source_is_fraction,
    _summary_row_map,
)
from src.analytics.topstep_optimization.metrics import build_ranking_table, summarize_simulation_results
from src.analytics.topstep_optimization.topstep_simulator import (
    ScaledVariantSeries,
    TopstepRules,
    build_variant_name,
    eligible_start_dates,
    run_block_bootstrap_simulations,
    run_historical_rolling_simulations,
    scale_daily_results,
)
from src.config.paths import EXPORTS_DIR, ensure_directories


DEFAULT_SOURCE_VARIANTS = (
    ("nominal", "nominal"),
    ("sizing_3state_realized_vol_ratio_15_60", "sizing_3state"),
)
DEFAULT_LEVERAGE_FACTORS = (1.0, 1.2, 1.5, 1.8)
DEFAULT_BOOTSTRAP_PATHS = 2000
DEFAULT_BOOTSTRAP_BLOCK_SIZE = 5
DEFAULT_RESET_COST = 100.0
DEFAULT_PAYOUT_VALUE = 3000.0


@dataclass(frozen=True)
class TopstepOptimizationSpec:
    source_run_root: Path | None = None
    primary_scope: str = DEFAULT_PRIMARY_SCOPE
    leverage_factors: tuple[float, ...] = DEFAULT_LEVERAGE_FACTORS
    bootstrap_paths: int = DEFAULT_BOOTSTRAP_PATHS
    bootstrap_block_size: int = DEFAULT_BOOTSTRAP_BLOCK_SIZE
    random_seed: int = DEFAULT_RANDOM_SEED
    reset_cost: float = DEFAULT_RESET_COST
    payout_value: float = DEFAULT_PAYOUT_VALUE
    rules: TopstepRules = TopstepRules()
    output_root: Path | None = None


def _variant_seed(base_seed: int, variant_idx: int) -> int:
    return int(base_seed + 10_000 * (variant_idx + 1))


def _load_scaled_variant_series(
    spec: TopstepOptimizationSpec,
) -> tuple[Path, dict[str, Any], float, list[ScaledVariantSeries], list]:
    source_root = spec.source_run_root or _find_latest_source_run()
    metadata = _read_run_metadata(source_root)
    is_fraction = _source_is_fraction(metadata)
    summary_rows = _summary_row_map(source_root)

    base_inputs: list[tuple[str, str, Any, pd.DataFrame]] = []
    common_start_dates: set | None = None

    for source_variant_name, base_variant in DEFAULT_SOURCE_VARIANTS:
        variant_input = _load_variant_input(source_root, source_variant_name, summary_rows)
        scoped_daily = _scope_daily_results(
            variant_input.daily_results,
            is_fraction=is_fraction,
            scope=spec.primary_scope,
        )
        base_inputs.append((source_variant_name, base_variant, variant_input, scoped_daily))
        starts = set(eligible_start_dates(scoped_daily, max_trading_days=spec.rules.max_trading_days))
        common_start_dates = starts if common_start_dates is None else common_start_dates.intersection(starts)

    scaled_series: list[ScaledVariantSeries] = []
    for source_variant_name, base_variant, variant_input, scoped_daily in base_inputs:
        for leverage_factor in spec.leverage_factors:
            scaled_series.append(
                ScaledVariantSeries(
                    variant=build_variant_name(base_variant, leverage_factor),
                    base_variant=base_variant,
                    source_variant_name=source_variant_name,
                    leverage_factor=float(leverage_factor),
                    daily_results=scale_daily_results(scoped_daily, leverage_factor),
                    reference_account_size_usd=variant_input.reference_account_size_usd,
                )
            )

    common_starts_sorted = sorted(pd.to_datetime(pd.Index(common_start_dates or [])).date)
    return source_root, metadata, is_fraction, scaled_series, common_starts_sorted


def _build_report(
    output_path: Path,
    spec: TopstepOptimizationSpec,
    source_root: Path,
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    common_start_dates: list,
) -> None:
    if summary.empty:
        output_path.write_text("# Topstep Optimization Report\n\nNo simulation rows were produced.\n", encoding="utf-8")
        return

    rolling = summary.loc[summary["simulation_mode"].eq("historical_rolling")].copy()
    bootstrap = summary.loc[summary["simulation_mode"].eq("block_bootstrap")].copy()

    rolling_best = rolling.sort_values("expected_profit_per_day", ascending=False).iloc[0] if not rolling.empty else None
    bootstrap_best = bootstrap.sort_values("expected_profit_per_day", ascending=False).iloc[0] if not bootstrap.empty else None
    overall_best = ranking.iloc[0]
    recommendation = bootstrap_best if bootstrap_best is not None and float(bootstrap_best["expected_profit_per_day"]) > 0.0 else overall_best

    lines = [
        "# Topstep Optimization Report",
        "",
        "## Scope",
        f"- Source run: `{source_root}`",
        f"- Primary scope: `{spec.primary_scope}`",
        f"- OOS rolling common start dates: `{len(common_start_dates)}`",
        f"- Rules: start `{spec.rules.starting_balance_usd:,.0f}` | target `{spec.rules.profit_target_usd:,.0f}` | trailing DD `{spec.rules.trailing_drawdown_usd:,.0f}` | daily loss `{spec.rules.daily_loss_limit_usd:,.0f}` | max trading days `{spec.rules.max_trading_days}`",
        f"- Economic objective: `pass_rate * payout_value - fail_rate * reset_cost` with payout `{spec.payout_value:,.0f}` and reset `{spec.reset_cost:,.0f}`",
        "",
        "## Best Variant",
        f"- Overall top row by expected profit per day: **{overall_best['simulation_mode']} / {overall_best['variant']}**",
        f"- Historical rolling winner: **{rolling_best['variant']}** | expected profit/day `{rolling_best['expected_profit_per_day']:.2f}` | pass `{rolling_best['pass_rate']:.1%}` | fail `{rolling_best['fail_rate']:.1%}`" if rolling_best is not None else "- Historical rolling winner: n/a",
        f"- Block bootstrap winner: **{bootstrap_best['variant']}** | expected profit/day `{bootstrap_best['expected_profit_per_day']:.2f}` | pass `{bootstrap_best['pass_rate']:.1%}` | fail `{bootstrap_best['fail_rate']:.1%}`" if bootstrap_best is not None else "- Block bootstrap winner: n/a",
        "",
        "## Trade-Off",
        "- Higher leverage only helps if the extra pass-rate offsets the additional daily-loss and trailing-DD failures; this campaign ranks variants on that economic trade-off rather than raw pnl.",
        f"- The recommended row has pass<=20d probability `{recommendation['probability_pass_within_20_days']:.1%}` and pass<=30d probability `{recommendation['probability_pass_within_30_days']:.1%}`.",
        f"- Recommended live configuration: **{recommendation['variant']}** under **{recommendation['simulation_mode']}** emphasis, because it delivers expected profit/day `{recommendation['expected_profit_per_day']:.2f}` with pass `{recommendation['pass_rate']:.1%}` and fail `{recommendation['fail_rate']:.1%}`.",
        "",
        "## Top Ranking",
        "",
        "```text",
        ranking[
            [
                "rank",
                "simulation_mode",
                "variant",
                "pass_rate",
                "fail_rate",
                "avg_time_to_pass",
                "avg_time_to_fail",
                "expected_profit_per_cycle",
                "expected_days_per_cycle",
                "expected_profit_per_day",
                "probability_pass_within_20_days",
                "probability_pass_within_30_days",
            ]
        ]
        .head(8)
        .to_string(index=False),
        "```",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _serialize_dates(values: list) -> list[str]:
    return [pd.Timestamp(value).date().isoformat() for value in values]


def run_campaign(spec: TopstepOptimizationSpec) -> Path:
    ensure_directories()
    source_root, metadata, is_fraction, scaled_series, common_start_dates = _load_scaled_variant_series(spec)

    rolling_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    for variant_idx, series in enumerate(scaled_series):
        rolling_frames.append(
            run_historical_rolling_simulations(
                series=series,
                rules=spec.rules,
                common_start_dates=common_start_dates,
            )
        )
        bootstrap_frames.append(
            run_block_bootstrap_simulations(
                series=series,
                rules=spec.rules,
                bootstrap_paths=spec.bootstrap_paths,
                block_size=spec.bootstrap_block_size,
                random_seed=_variant_seed(spec.random_seed, variant_idx),
            )
        )

    rolling_results = pd.concat(rolling_frames, ignore_index=True) if rolling_frames else pd.DataFrame()
    bootstrap_results = pd.concat(bootstrap_frames, ignore_index=True) if bootstrap_frames else pd.DataFrame()
    combined_results = pd.concat([rolling_results, bootstrap_results], ignore_index=True)

    summary = summarize_simulation_results(
        combined_results,
        payout_value=spec.payout_value,
        reset_cost=spec.reset_cost,
    )
    ranking = build_ranking_table(summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = spec.output_root or (EXPORTS_DIR / f"topstep_optimization_{timestamp}")
    output_root.mkdir(parents=True, exist_ok=True)

    rolling_results.to_csv(output_root / "rolling_results.csv", index=False)
    bootstrap_results.to_csv(output_root / "bootstrap_results.csv", index=False)
    summary.to_csv(output_root / "summary_variants.csv", index=False)
    ranking.to_csv(output_root / "ranking_table.csv", index=False)
    _build_report(
        output_path=output_root / "topstep_optimization_report.md",
        spec=spec,
        source_root=source_root,
        summary=summary,
        ranking=ranking,
        common_start_dates=common_start_dates,
    )
    _json_dump(
        output_root / "run_metadata.json",
        {
            "run_timestamp": datetime.now().isoformat(),
            "source_run_root": source_root,
            "source_is_fraction": is_fraction,
            "source_metadata": metadata,
            "primary_scope": spec.primary_scope,
            "rules": asdict(spec.rules),
            "leverage_factors": list(spec.leverage_factors),
            "bootstrap_paths": spec.bootstrap_paths,
            "bootstrap_block_size": spec.bootstrap_block_size,
            "random_seed": spec.random_seed,
            "reset_cost": spec.reset_cost,
            "payout_value": spec.payout_value,
            "common_start_dates": _serialize_dates(common_start_dates),
        },
    )
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, default=None)
    parser.add_argument("--primary-scope", default=DEFAULT_PRIMARY_SCOPE, choices=("overall", "oos"))
    parser.add_argument("--bootstrap-paths", type=int, default=DEFAULT_BOOTSTRAP_PATHS)
    parser.add_argument("--bootstrap-block-size", type=int, default=DEFAULT_BOOTSTRAP_BLOCK_SIZE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--reset-cost", type=float, default=DEFAULT_RESET_COST)
    parser.add_argument("--payout-value", type=float, default=DEFAULT_PAYOUT_VALUE)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = TopstepOptimizationSpec(
        source_run_root=args.source_run_root,
        primary_scope=args.primary_scope,
        bootstrap_paths=args.bootstrap_paths,
        bootstrap_block_size=args.bootstrap_block_size,
        random_seed=args.random_seed,
        reset_cost=args.reset_cost,
        payout_value=args.payout_value,
        output_root=args.output_root,
    )
    output_root = run_campaign(spec)
    print(f"Topstep optimization export written to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
