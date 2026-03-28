"""Full Topstep business optimization campaign across challenge and funded phases."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
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
from src.analytics.topstep_business_optimization.metrics import build_ranking_table, summarize_business_runs
from src.analytics.topstep_business_optimization.simulator import (
    BusinessRules,
    StrategySeries,
    TopstepPlan,
    prepare_strategy_series,
    simulate_business_cycle,
)
from src.config.paths import EXPORTS_DIR, ensure_directories


DEFAULT_BOOTSTRAP_PATHS = 2000
DEFAULT_BOOTSTRAP_BLOCK_SIZE = 5
DEFAULT_CHALLENGE_VARIANTS = (
    ("nominal", "nominal", 1.0),
    ("nominal", "nominal", 1.2),
    ("nominal", "nominal", 1.5),
    ("nominal", "nominal", 1.8),
)
DEFAULT_FUNDED_VARIANTS = (
    ("sizing_3state_realized_vol_ratio_15_60", "sizing_3state", 1.0),
    ("sizing_3state_realized_vol_ratio_15_60", "sizing_3state", 1.2),
    ("nominal", "nominal", 1.0),
)
DEFAULT_PLANS = (
    TopstepPlan(
        name="standard",
        subscription_monthly_usd=49.0,
        reset_cost_usd=49.0,
        activation_fee_usd=149.0,
    ),
    TopstepPlan(
        name="no_activation_fee",
        subscription_monthly_usd=109.0,
        reset_cost_usd=109.0,
        activation_fee_usd=0.0,
    ),
)


@dataclass(frozen=True)
class BusinessCampaignSpec:
    source_run_root: Path | None = None
    primary_scope: str = DEFAULT_PRIMARY_SCOPE
    bootstrap_paths: int = DEFAULT_BOOTSTRAP_PATHS
    bootstrap_block_size: int = DEFAULT_BOOTSTRAP_BLOCK_SIZE
    random_seed: int = DEFAULT_RANDOM_SEED
    rules: BusinessRules = BusinessRules()
    plans: tuple[TopstepPlan, ...] = DEFAULT_PLANS
    output_root: Path | None = None


def _config_seed(base_seed: int, plan_idx: int, challenge_idx: int, funded_idx: int) -> int:
    return int(base_seed + (plan_idx + 1) * 100_000 + (challenge_idx + 1) * 10_000 + (funded_idx + 1) * 1_000)


def _load_strategy_universe(
    spec: BusinessCampaignSpec,
) -> tuple[Path, dict[str, Any], list[StrategySeries], list[StrategySeries]]:
    source_root = spec.source_run_root or _find_latest_source_run()
    metadata = _read_run_metadata(source_root)
    is_fraction = _source_is_fraction(metadata)
    summary_rows = _summary_row_map(source_root)

    required_variants = {
        variant_name for variant_name, _, _ in DEFAULT_CHALLENGE_VARIANTS + DEFAULT_FUNDED_VARIANTS
    }
    variant_inputs = {
        variant_name: _load_variant_input(source_root, variant_name, summary_rows)
        for variant_name in sorted(required_variants)
    }
    scoped_inputs = {
        variant_name: _scope_daily_results(
            variant_input.daily_results,
            is_fraction=is_fraction,
            scope=spec.primary_scope,
        )
        for variant_name, variant_input in variant_inputs.items()
    }

    challenge_series = [
        prepare_strategy_series(
            daily_results=scoped_inputs[source_variant_name],
            strategy_name=strategy_name,
            source_variant_name=source_variant_name,
            leverage_factor=leverage_factor,
        )
        for source_variant_name, strategy_name, leverage_factor in DEFAULT_CHALLENGE_VARIANTS
    ]
    funded_series = [
        prepare_strategy_series(
            daily_results=scoped_inputs[source_variant_name],
            strategy_name=strategy_name,
            source_variant_name=source_variant_name,
            leverage_factor=leverage_factor,
        )
        for source_variant_name, strategy_name, leverage_factor in DEFAULT_FUNDED_VARIANTS
    ]

    return source_root, metadata, challenge_series, funded_series


def _build_report(
    output_path: Path,
    source_root: Path,
    spec: BusinessCampaignSpec,
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
) -> None:
    if summary.empty:
        output_path.write_text("# Topstep Business Optimization\n\nNo simulation rows were produced.\n", encoding="utf-8")
        return

    best_by_plan = {
        plan_name: group.sort_values("expected_net_profit_per_day", ascending=False).iloc[0]
        for plan_name, group in summary.groupby("plan", sort=True)
        if not group.empty
    }
    overall_best = ranking.iloc[0]

    sensitivity_text = "- Reset-cost sensitivity could not be computed cleanly."
    paired = summary.pivot_table(
        index=["challenge_strategy", "funded_strategy"],
        columns="plan",
        values=["avg_resets", "expected_net_profit_per_day"],
    )
    if (
        not paired.empty
        and ("expected_net_profit_per_day", "standard") in paired.columns
        and ("expected_net_profit_per_day", "no_activation_fee") in paired.columns
    ):
        compare = paired.copy()
        compare["profit_day_delta_no_activation_minus_standard"] = (
            compare[("expected_net_profit_per_day", "no_activation_fee")]
            - compare[("expected_net_profit_per_day", "standard")]
        )
        most_sensitive_idx = compare["profit_day_delta_no_activation_minus_standard"].abs().idxmax()
        delta_value = float(compare.loc[most_sensitive_idx, "profit_day_delta_no_activation_minus_standard"])
        sensitivity_text = (
            f"- Most fee-sensitive pairing: **{most_sensitive_idx[0]} -> {most_sensitive_idx[1]}** | "
            f"delta profit/day `{delta_value:.2f}` for `no_activation_fee - standard`."
        )

    lines = [
        "# Topstep Business Optimization",
        "",
        "## Scope",
        f"- Source export: `{source_root}`",
        f"- Scope used: `{spec.primary_scope}`",
        f"- Bootstrap paths per configuration: `{spec.bootstrap_paths}`",
        f"- Block size: `{spec.bootstrap_block_size}`",
        f"- Seed: `{spec.random_seed}`",
        f"- Challenge rules: target `{spec.rules.challenge_profit_target_usd:,.0f}` | trailing DD `{spec.rules.trailing_drawdown_usd:,.0f}` | daily loss `{spec.rules.daily_loss_limit_usd:,.0f}`",
        f"- Funded phase: `{spec.rules.funded_trading_days}` traded days max, payout every `{spec.rules.payout_threshold_usd:,.0f}` cumulative funded profit",
        "",
        "## Best Configuration By Plan",
    ]

    for plan_name, row in best_by_plan.items():
        lines.append(
            f"- **{plan_name}**: `{row['challenge_strategy']}` -> `{row['funded_strategy']}` | expected net/day `{row['expected_net_profit_per_day']:.2f}` | pass `{row['pass_rate']:.1%}` | avg resets `{row['avg_resets']:.2f}` | avg payouts `{row['avg_payouts']:.2f}`."
        )

    lines.extend(
        [
            "",
            "## Trade-Off",
            f"- Overall best row: **{overall_best['plan']} / {overall_best['challenge_strategy']} -> {overall_best['funded_strategy']}** with expected net/day `{overall_best['expected_net_profit_per_day']:.2f}` and expected net/cycle `{overall_best['expected_net_profit_per_cycle']:.2f}`.",
            "- Aggressive challenge leverage helps only if faster passes offset more resets and the extra subscription drag from longer retry loops.",
            "- Funded-side value is captured through realized payout counts, while raw funded mark-to-market is still reported separately in the summary table.",
            "",
            "## Reset-Cost Sensitivity",
            sensitivity_text,
            "- Standard has cheaper monthly/reset costs but pays activation; No Activation Fee removes the pass fee but taxes every challenge day and every reset more heavily.",
            "",
            "## Top Ranking",
            "",
            "```text",
            ranking[
                [
                    "overall_rank",
                    "plan_rank",
                    "plan",
                    "challenge_strategy",
                    "funded_strategy",
                    "pass_rate",
                    "avg_days_to_pass",
                    "avg_resets",
                    "avg_payouts",
                    "total_cost",
                    "expected_net_profit_per_cycle",
                    "expected_net_profit_per_day",
                ]
            ]
            .head(10)
            .to_string(index=False),
            "```",
            "",
            "## Assumptions",
            "- Challenge attempts are repeated with resets until first pass or a technical cap is hit (`max_challenge_attempts`, `challenge_safety_max_days`).",
            "- Subscription cost is prorated on challenge trading days only because the available audited input is daily strategy PnL.",
            "- Funded payouts are counted each time cumulative funded profit crosses another +1,000 USD threshold; payouts do not reduce simulated account equity in this stylized model.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_campaign(spec: BusinessCampaignSpec) -> Path:
    ensure_directories()
    source_root, metadata, challenge_series, funded_series = _load_strategy_universe(spec)

    rows: list[dict[str, Any]] = []
    for plan_idx, plan in enumerate(spec.plans):
        for challenge_idx, challenge_strategy in enumerate(challenge_series):
            for funded_idx, funded_strategy in enumerate(funded_series):
                rng = np.random.default_rng(
                    _config_seed(spec.random_seed, plan_idx=plan_idx, challenge_idx=challenge_idx, funded_idx=funded_idx)
                )
                for simulation_id in range(int(spec.bootstrap_paths)):
                    result = simulate_business_cycle(
                        plan=plan,
                        challenge_strategy=challenge_strategy,
                        funded_strategy=funded_strategy,
                        rules=spec.rules,
                        block_size=spec.bootstrap_block_size,
                        rng=rng,
                    )
                    rows.append(
                        {
                            "simulation_mode": "block_bootstrap_business",
                            "simulation_id": int(simulation_id),
                            "config_id": f"{plan.name}__{challenge_strategy.strategy_name}__{funded_strategy.strategy_name}",
                            **result,
                        }
                    )

    detailed = pd.DataFrame(rows)
    summary = summarize_business_runs(detailed)
    ranking = build_ranking_table(summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = spec.output_root or (EXPORTS_DIR / f"topstep_business_optimization_{timestamp}")
    output_root.mkdir(parents=True, exist_ok=True)

    detailed.to_csv(output_root / "detailed_simulations.csv", index=False)
    summary.to_csv(output_root / "business_summary.csv", index=False)
    ranking.to_csv(output_root / "ranking_by_profit_per_day.csv", index=False)
    _build_report(
        output_path=output_root / "business_report.md",
        source_root=source_root,
        spec=spec,
        summary=summary,
        ranking=ranking,
    )
    _json_dump(
        output_root / "run_metadata.json",
        {
            "run_timestamp": datetime.now().isoformat(),
            "source_run_root": str(source_root),
            "source_metadata": metadata,
            "primary_scope": spec.primary_scope,
            "bootstrap_paths": spec.bootstrap_paths,
            "bootstrap_block_size": spec.bootstrap_block_size,
            "random_seed": spec.random_seed,
            "plans": [asdict(plan) for plan in spec.plans],
            "rules": asdict(spec.rules),
        },
    )
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, default=None)
    parser.add_argument("--primary-scope", type=str, default=DEFAULT_PRIMARY_SCOPE, choices=("overall", "oos"))
    parser.add_argument("--bootstrap-paths", type=int, default=DEFAULT_BOOTSTRAP_PATHS)
    parser.add_argument("--bootstrap-block-size", type=int, default=DEFAULT_BOOTSTRAP_BLOCK_SIZE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = run_campaign(
        BusinessCampaignSpec(
            source_run_root=args.source_run_root,
            primary_scope=args.primary_scope,
            bootstrap_paths=args.bootstrap_paths,
            bootstrap_block_size=args.bootstrap_block_size,
            random_seed=args.random_seed,
            output_root=args.output_root,
        )
    )
    print(f"Topstep business optimization export written to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
