"""Topstep business v2 campaign with combine, funded, and calendar-aware economics."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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
from src.analytics.topstep_business_v2.metrics import (
    build_challenge_diagnostics,
    build_funded_diagnostics,
    build_ranking_day,
    build_ranking_month,
    summarize_business_runs,
)
from src.analytics.topstep_business_v2.simulator import (
    DEFAULT_PLANS,
    BusinessCycleArtifacts,
    BusinessV2Rules,
    StrategySeries,
    TopstepBusinessPlan,
    build_historical_path,
    prepare_strategy_series,
    sample_block_bootstrap_path,
    simulate_business_cycle,
)
from src.config.paths import EXPORTS_DIR, ensure_directories


DEFAULT_BOOTSTRAP_PATHS = 2000
DEFAULT_CHALLENGE_VARIANTS = (
    ("nominal", "nominal", 1.2),
    ("nominal", "nominal", 1.5),
    ("nominal", "nominal", 1.8),
    ("sizing_3state_realized_vol_ratio_15_60", "sizing_3state", 1.2),
    ("sizing_3state_realized_vol_ratio_15_60", "sizing_3state", 1.5),
)
DEFAULT_FUNDED_VARIANTS = (
    ("nominal", "nominal", 1.0),
    ("sizing_3state_realized_vol_ratio_15_60", "sizing_3state", 1.0),
    ("sizing_3state_realized_vol_ratio_15_60", "sizing_3state", 1.2),
)


@dataclass(frozen=True)
class TopstepBusinessV2Spec:
    """Configuration for the Topstep business v2 campaign."""

    source_run_root: Path | None = None
    primary_scope: str = DEFAULT_PRIMARY_SCOPE
    bootstrap_paths: int = DEFAULT_BOOTSTRAP_PATHS
    bootstrap_block_size: int = 5
    random_seed: int = DEFAULT_RANDOM_SEED
    n_resets_per_month_grid: tuple[int, ...] = (1,)
    max_total_budget_usd: float | None = None
    challenge_variants: tuple[tuple[str, str, float], ...] = DEFAULT_CHALLENGE_VARIANTS
    funded_variants: tuple[tuple[str, str, float], ...] = DEFAULT_FUNDED_VARIANTS
    plans: tuple[TopstepBusinessPlan, ...] = DEFAULT_PLANS
    rules: BusinessV2Rules = BusinessV2Rules()
    output_root: Path | None = None


def _config_seed(base_seed: int, reset_idx: int, plan_idx: int, challenge_idx: int, funded_idx: int) -> int:
    return int(
        base_seed
        + (reset_idx + 1) * 1_000_000
        + (plan_idx + 1) * 100_000
        + (challenge_idx + 1) * 10_000
        + (funded_idx + 1) * 1_000
    )


def _load_strategy_universe(
    spec: TopstepBusinessV2Spec,
) -> tuple[Path, dict[str, Any], list[StrategySeries], list[StrategySeries], list[dt.date]]:
    source_root = spec.source_run_root or _find_latest_source_run()
    metadata = _read_run_metadata(source_root)
    is_fraction = _source_is_fraction(metadata)
    summary_rows = _summary_row_map(source_root)

    requested_source_variants = {
        source_variant_name
        for source_variant_name, _, _ in spec.challenge_variants + spec.funded_variants
    }
    scoped_daily_map: dict[str, pd.DataFrame] = {}
    reference_account_size_map: dict[str, float] = {}
    date_sets: list[set[dt.date]] = []

    for source_variant_name in sorted(requested_source_variants):
        try:
            variant_input = _load_variant_input(source_root, source_variant_name, summary_rows)
        except FileNotFoundError:
            continue
        scoped_daily = _scope_daily_results(
            variant_input.daily_results,
            is_fraction=is_fraction,
            scope=spec.primary_scope,
        )
        if scoped_daily.empty:
            continue
        scoped_daily_map[source_variant_name] = scoped_daily
        reference_account_size_map[source_variant_name] = float(variant_input.reference_account_size_usd)
        date_sets.append(set(pd.to_datetime(scoped_daily["session_date"]).dt.date.tolist()))

    if not scoped_daily_map:
        raise FileNotFoundError("No requested source variants were available for topstep_business_v2.")

    challenge_series: list[StrategySeries] = []
    funded_series: list[StrategySeries] = []
    for source_variant_name, strategy_name, leverage_factor in spec.challenge_variants:
        if source_variant_name not in scoped_daily_map:
            continue
        challenge_series.append(
            prepare_strategy_series(
                daily_results=scoped_daily_map[source_variant_name],
                strategy_name=strategy_name,
                source_variant_name=source_variant_name,
                leverage_factor=leverage_factor,
                reference_account_size_usd=reference_account_size_map[source_variant_name],
            )
        )
    for source_variant_name, strategy_name, leverage_factor in spec.funded_variants:
        if source_variant_name not in scoped_daily_map:
            continue
        funded_series.append(
            prepare_strategy_series(
                daily_results=scoped_daily_map[source_variant_name],
                strategy_name=strategy_name,
                source_variant_name=source_variant_name,
                leverage_factor=leverage_factor,
                reference_account_size_usd=reference_account_size_map[source_variant_name],
            )
        )

    if not challenge_series:
        raise ValueError("No challenge strategy variants were available.")
    if not funded_series:
        raise ValueError("No funded strategy variants were available.")

    common_dates = sorted(set.intersection(*date_sets)) if date_sets else []
    return source_root, metadata, challenge_series, funded_series, common_dates


def _series_index_map(series_list: list[StrategySeries]) -> dict[str, dict[dt.date, int]]:
    return {
        series.strategy_name: {date_value: index for index, date_value in enumerate(series.session_dates)}
        for series in series_list
    }


def _config_id(
    simulation_method: str,
    plan: TopstepBusinessPlan,
    n_resets_per_month: int,
    challenge_strategy: StrategySeries,
    funded_strategy: StrategySeries,
) -> str:
    return (
        f"{simulation_method}__{plan.name}__resets_{int(n_resets_per_month)}__"
        f"{challenge_strategy.strategy_name}__{funded_strategy.strategy_name}"
    )


def _collect_artifacts(
    artifacts: BusinessCycleArtifacts,
    cycle_rows: list[dict[str, Any]],
    challenge_attempt_rows: list[dict[str, Any]],
    funded_rows: list[dict[str, Any]],
) -> None:
    cycle_rows.append(artifacts.cycle_row)
    challenge_attempt_rows.extend(artifacts.challenge_attempt_rows)
    funded_rows.extend(artifacts.funded_rows)


def _best_row(summary: pd.DataFrame, reset_budget: int, simulation_method: str) -> pd.Series | None:
    scoped = summary.loc[
        summary["n_resets_per_month"].eq(int(reset_budget))
        & summary["simulation_method"].eq(simulation_method)
    ].copy()
    if scoped.empty:
        return None
    scoped = build_ranking_month(scoped)
    return scoped.iloc[0]


def _challenge_leader(summary: pd.DataFrame, reset_budget: int, simulation_method: str) -> pd.Series | None:
    scoped = summary.loc[
        summary["n_resets_per_month"].eq(int(reset_budget))
        & summary["simulation_method"].eq(simulation_method)
    ].copy()
    if scoped.empty:
        return None
    grouped = (
        scoped.groupby("challenge_strategy", dropna=False)
        .agg(
            expected_net_profit_per_calendar_month=("expected_net_profit_per_calendar_month", "mean"),
            consistency_block_rate=("consistency_block_rate", "mean"),
            avg_days_raw_target_to_true_pass=("avg_days_raw_target_to_true_pass", "mean"),
            avg_consistency_penalty_usd=("avg_consistency_penalty_usd", "mean"),
        )
        .reset_index()
        .sort_values(
            [
                "expected_net_profit_per_calendar_month",
                "consistency_block_rate",
                "avg_days_raw_target_to_true_pass",
            ],
            ascending=[False, True, True],
            na_position="last",
        )
    )
    return grouped.iloc[0] if not grouped.empty else None


def _cash_constrained_row(summary: pd.DataFrame, reset_budget: int, simulation_method: str) -> pd.Series | None:
    scoped = summary.loc[
        summary["n_resets_per_month"].eq(int(reset_budget))
        & summary["simulation_method"].eq(simulation_method)
    ].copy()
    if scoped.empty:
        return None
    profitable = scoped.loc[scoped["expected_net_profit_per_calendar_month"] > 0.0].copy()
    candidate = profitable if not profitable.empty else scoped
    candidate = candidate.sort_values(
        [
            "ROI_on_cash_spent",
            "expected_cost_per_month",
            "expected_net_profit_per_calendar_month",
            "first_attempt_pass_rate",
        ],
        ascending=[False, True, False, False],
        na_position="last",
    )
    return candidate.iloc[0] if not candidate.empty else None


def _build_reset_sensitivity_table(summary: pd.DataFrame, simulation_method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for reset_budget in sorted(summary["n_resets_per_month"].dropna().unique()):
        best_row = _best_row(summary, int(reset_budget), simulation_method)
        if best_row is None:
            continue
        rows.append(
            {
                "simulation_method": simulation_method,
                "n_resets_per_month": int(reset_budget),
                "plan": best_row["plan"],
                "challenge_strategy": best_row["challenge_strategy"],
                "funded_strategy": best_row["funded_strategy"],
                "expected_net_profit_per_calendar_month": best_row["expected_net_profit_per_calendar_month"],
                "expected_net_profit_per_calendar_day": best_row["expected_net_profit_per_calendar_day"],
                "first_attempt_pass_rate": best_row["first_attempt_pass_rate"],
                "first_payout_rate": best_row["first_payout_rate"],
            }
        )
    return pd.DataFrame(rows)


def _build_report(
    output_path: Path,
    source_root: Path,
    spec: TopstepBusinessV2Spec,
    summary: pd.DataFrame,
    ranking_month: pd.DataFrame,
    reset_sensitivity_bootstrap: pd.DataFrame,
) -> None:
    if summary.empty:
        output_path.write_text("# Topstep Business V2\n\nNo simulation rows were produced.\n", encoding="utf-8")
        return

    bootstrap_best_1 = _best_row(summary, reset_budget=1, simulation_method="block_bootstrap")
    bootstrap_best_0 = _best_row(summary, reset_budget=0, simulation_method="block_bootstrap")
    rolling_best_1 = _best_row(summary, reset_budget=1, simulation_method="historical_rolling")
    challenge_leader_1 = _challenge_leader(summary, reset_budget=1, simulation_method="block_bootstrap")
    cash_row_1 = _cash_constrained_row(summary, reset_budget=1, simulation_method="block_bootstrap")

    nominal_x18_rows = summary.loc[
        summary["simulation_method"].eq("block_bootstrap")
        & summary["challenge_strategy"].eq("nominal_x1.8")
        & summary["n_resets_per_month"].eq(1)
    ].copy()
    nominal_x18_best = build_ranking_month(nominal_x18_rows).iloc[0] if not nominal_x18_rows.empty else None
    nominal_x18_optimal = bool(
        bootstrap_best_1 is not None
        and str(bootstrap_best_1["challenge_strategy"]) == "nominal_x1.8"
    )

    aggressive_penalty = (
        summary.loc[
            summary["simulation_method"].eq("block_bootstrap")
            & summary["n_resets_per_month"].eq(1)
            & summary["challenge_strategy"].isin(["nominal_x1.8", "nominal_x1.5"])
        ]
        .groupby("challenge_strategy", dropna=False)
        .agg(
            consistency_block_rate=("consistency_block_rate", "mean"),
            avg_days_raw_target_to_true_pass=("avg_days_raw_target_to_true_pass", "mean"),
            avg_consistency_penalty_usd=("avg_consistency_penalty_usd", "mean"),
        )
        .reset_index()
    )

    limited_reset_rows = summary.loc[
        summary["simulation_method"].eq("block_bootstrap") & summary["n_resets_per_month"].eq(1)
    ].copy()
    plan_compare = (
        limited_reset_rows.groupby("plan", dropna=False)
        .agg(
            expected_net_profit_per_calendar_month=("expected_net_profit_per_calendar_month", "max"),
            expected_net_profit_per_calendar_day=("expected_net_profit_per_calendar_day", "max"),
        )
        .reset_index()
        .sort_values(
            ["expected_net_profit_per_calendar_month", "expected_net_profit_per_calendar_day"],
            ascending=[False, False],
        )
    )
    best_pair_1 = (
        limited_reset_rows.sort_values(
            [
                "expected_net_profit_per_calendar_month",
                "expected_net_profit_per_calendar_day",
                "first_attempt_pass_rate",
                "first_payout_rate",
            ],
            ascending=[False, False, False, False],
        ).iloc[0]
        if not limited_reset_rows.empty
        else None
    )

    lines = [
        "# Topstep Business V2",
        "",
        "## Scope",
        f"- Source export: `{source_root}`",
        f"- Scope used: `{spec.primary_scope}`",
        f"- Bootstrap paths per configuration: `{spec.bootstrap_paths}`",
        f"- Bootstrap block size: `{spec.bootstrap_block_size}`",
        f"- Seed: `{spec.random_seed}`",
        f"- Reset-budget grid: `{list(spec.n_resets_per_month_grid)}`",
        f"- Max total budget: `{spec.max_total_budget_usd}`",
        "",
        "## Best Rows",
        f"- Bootstrap best under `n_resets_per_month=1`: **{bootstrap_best_1['plan']} | {bootstrap_best_1['challenge_strategy']} -> {bootstrap_best_1['funded_strategy']}** | net/month `{bootstrap_best_1['expected_net_profit_per_calendar_month']:.2f}`." if bootstrap_best_1 is not None else "- Bootstrap best under `n_resets_per_month=1`: n/a",
        f"- Bootstrap best under `n_resets_per_month=0`: **{bootstrap_best_0['plan']} | {bootstrap_best_0['challenge_strategy']} -> {bootstrap_best_0['funded_strategy']}** | net/month `{bootstrap_best_0['expected_net_profit_per_calendar_month']:.2f}`." if bootstrap_best_0 is not None else "- Bootstrap best under `n_resets_per_month=0`: n/a",
        f"- Rolling best under `n_resets_per_month=1`: **{rolling_best_1['plan']} | {rolling_best_1['challenge_strategy']} -> {rolling_best_1['funded_strategy']}** | net/month `{rolling_best_1['expected_net_profit_per_calendar_month']:.2f}`." if rolling_best_1 is not None else "- Rolling best under `n_resets_per_month=1`: n/a",
        "",
        "## Reset Sensitivity",
        "",
        "```text",
        reset_sensitivity_bootstrap.to_string(index=False) if not reset_sensitivity_bootstrap.empty else "No sensitivity rows.",
        "```",
        "",
        "## Required Diagnostics",
        f"1. nominal_x1.8 remains optimal after correct consistency modeling: **{'YES' if nominal_x18_optimal else 'NO'}**."
        + (
            f" Best bootstrap row at `n_resets_per_month=1` is `{bootstrap_best_1['challenge_strategy']} -> {bootstrap_best_1['funded_strategy']}`."
            if bootstrap_best_1 is not None
            else ""
        ),
        "2. Consistency penalty on aggressive challenge variants:",
        "",
        "```text",
        aggressive_penalty.to_string(index=False) if not aggressive_penalty.empty else "No aggressive rows available.",
        "```",
        f"3. Under `n_resets_per_month=1`, the best challenge strategy is **{challenge_leader_1['challenge_strategy']}** on the bootstrap decision surface." if challenge_leader_1 is not None else "3. Under `n_resets_per_month=1`, no bootstrap challenge leader was available.",
        f"4. Under limited resets, **{plan_compare.iloc[0]['plan']}** beats the other plan on best achievable net/month." if not plan_compare.empty else "4. Plan comparison unavailable.",
        f"5. Best pair `challenge -> funded`: **{best_pair_1['challenge_strategy']} -> {best_pair_1['funded_strategy']}**." if best_pair_1 is not None else "5. Best pair unavailable.",
        f"6. Best setup for a trader with limited cash and only 1 reset/month: **{cash_row_1['plan']} | {cash_row_1['challenge_strategy']} -> {cash_row_1['funded_strategy']}** with ROI `{cash_row_1['ROI_on_cash_spent']:.2f}` and expected cost/month `{cash_row_1['expected_cost_per_month']:.2f}`." if cash_row_1 is not None else "6. Cash-constrained recommendation unavailable.",
        "",
        "## Top Ranking By Month",
        "",
        "```text",
        ranking_month[
            [
                "rank_month",
                "simulation_method",
                "plan",
                "n_resets_per_month",
                "challenge_strategy",
                "funded_strategy",
                "expected_net_profit_per_calendar_month",
                "expected_net_profit_per_calendar_day",
                "first_attempt_pass_rate",
                "first_payout_rate",
            ]
        ]
        .head(12)
        .to_string(index=False),
        "```",
        "",
        "## Approximations",
        "- Challenge daily loss is modeled as an auto-liquidation cap on the day PnL at `-1,000 USD` because only audited daily results are available.",
        "- The trailing MLL uses prior end-of-day high-watermark logic and locks at the starting balance once reached.",
        "- Historical rolling uses real OOS dates; bootstrap assigns sampled sessions onto a synthetic calendar built from repeated real trading-day gaps so monthly reset refreshes and subscription accrual remain calendar-aware.",
        "- Challenge timeout is a research safety cap, not an official Topstep expiry rule.",
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_campaign(spec: TopstepBusinessV2Spec) -> Path:
    ensure_directories()
    source_root, metadata, challenge_series, funded_series, common_start_dates = _load_strategy_universe(spec)
    challenge_index_map = _series_index_map(challenge_series)
    funded_index_map = _series_index_map(funded_series)

    cycle_rows: list[dict[str, Any]] = []
    challenge_attempt_rows: list[dict[str, Any]] = []
    funded_rows: list[dict[str, Any]] = []

    for reset_idx, n_resets_per_month in enumerate(spec.n_resets_per_month_grid):
        rules = replace(
            spec.rules,
            n_resets_per_month=int(n_resets_per_month),
            bootstrap_block_size=int(spec.bootstrap_block_size),
            max_total_budget_usd=spec.max_total_budget_usd,
        )
        for plan_idx, plan in enumerate(spec.plans):
            for challenge_idx, challenge_strategy in enumerate(challenge_series):
                for funded_idx, funded_strategy in enumerate(funded_series):
                    config_seed = _config_seed(
                        spec.random_seed,
                        reset_idx=reset_idx,
                        plan_idx=plan_idx,
                        challenge_idx=challenge_idx,
                        funded_idx=funded_idx,
                    )

                    for simulation_id, start_date in enumerate(common_start_dates):
                        if start_date not in challenge_index_map[challenge_strategy.strategy_name]:
                            continue
                        if start_date not in funded_index_map[funded_strategy.strategy_name]:
                            continue
                        artifacts = simulate_business_cycle(
                            plan=plan,
                            challenge_path=build_historical_path(
                                challenge_strategy,
                                challenge_index_map[challenge_strategy.strategy_name][start_date],
                            ),
                            challenge_strategy=challenge_strategy,
                            funded_path=build_historical_path(
                                funded_strategy,
                                funded_index_map[funded_strategy.strategy_name][start_date],
                            ),
                            funded_strategy=funded_strategy,
                            rules=rules,
                            simulation_method="historical_rolling",
                            simulation_id=int(simulation_id),
                            config_id=_config_id(
                                "historical_rolling",
                                plan,
                                int(n_resets_per_month),
                                challenge_strategy,
                                funded_strategy,
                            ),
                            start_session_date=start_date,
                        )
                        _collect_artifacts(artifacts, cycle_rows, challenge_attempt_rows, funded_rows)

                    rng = np.random.default_rng(config_seed)
                    for simulation_id in range(int(spec.bootstrap_paths)):
                        calendar_anchor_index = int(rng.integers(0, max(len(challenge_strategy.session_dates), 1)))
                        artifacts = simulate_business_cycle(
                            plan=plan,
                            challenge_path=sample_block_bootstrap_path(
                                challenge_strategy,
                                rules=rules,
                                rng=rng,
                                calendar_anchor_index=calendar_anchor_index,
                            ),
                            challenge_strategy=challenge_strategy,
                            funded_path=sample_block_bootstrap_path(
                                funded_strategy,
                                rules=rules,
                                rng=rng,
                                calendar_anchor_index=calendar_anchor_index,
                            ),
                            funded_strategy=funded_strategy,
                            rules=rules,
                            simulation_method="block_bootstrap",
                            simulation_id=int(simulation_id),
                            config_id=_config_id(
                                "block_bootstrap",
                                plan,
                                int(n_resets_per_month),
                                challenge_strategy,
                                funded_strategy,
                            ),
                            start_session_date=None,
                        )
                        _collect_artifacts(artifacts, cycle_rows, challenge_attempt_rows, funded_rows)

    detailed = pd.DataFrame(cycle_rows)
    summary = summarize_business_runs(detailed)
    challenge_diag = build_challenge_diagnostics(summary)
    funded_diag = build_funded_diagnostics(summary)
    ranking_month = build_ranking_month(summary)
    ranking_day = build_ranking_day(summary)
    reset_sensitivity_bootstrap = _build_reset_sensitivity_table(summary, simulation_method="block_bootstrap")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = spec.output_root or (EXPORTS_DIR / f"topstep_business_v2_{timestamp}")
    output_root.mkdir(parents=True, exist_ok=True)

    detailed.to_csv(output_root / "business_v2_detailed_simulations.csv", index=False)
    summary.to_csv(output_root / "business_v2_summary.csv", index=False)
    ranking_month.to_csv(output_root / "business_v2_ranking_month.csv", index=False)
    ranking_day.to_csv(output_root / "business_v2_ranking_day.csv", index=False)
    challenge_diag.to_csv(output_root / "business_v2_challenge_diagnostics.csv", index=False)
    funded_diag.to_csv(output_root / "business_v2_funded_diagnostics.csv", index=False)
    _build_report(
        output_path=output_root / "business_v2_report.md",
        source_root=source_root,
        spec=spec,
        summary=summary,
        ranking_month=ranking_month,
        reset_sensitivity_bootstrap=reset_sensitivity_bootstrap,
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
            "n_resets_per_month_grid": list(spec.n_resets_per_month_grid),
            "max_total_budget_usd": spec.max_total_budget_usd,
            "plans": [asdict(plan) for plan in spec.plans],
            "rules": asdict(spec.rules),
        },
    )
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, default=None)
    parser.add_argument("--primary-scope", choices=("overall", "oos"), default=DEFAULT_PRIMARY_SCOPE)
    parser.add_argument("--bootstrap-paths", type=int, default=DEFAULT_BOOTSTRAP_PATHS)
    parser.add_argument("--bootstrap-block-size", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--n-resets-per-month", type=int, nargs="+", default=[1])
    parser.add_argument("--max-total-budget-usd", type=float, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = run_campaign(
        TopstepBusinessV2Spec(
            source_run_root=args.source_run_root,
            primary_scope=str(args.primary_scope),
            bootstrap_paths=int(args.bootstrap_paths),
            bootstrap_block_size=int(args.bootstrap_block_size),
            random_seed=int(args.random_seed),
            n_resets_per_month_grid=tuple(int(value) for value in args.n_resets_per_month),
            max_total_budget_usd=args.max_total_budget_usd,
            output_root=args.output_root,
        )
    )
    print(f"Topstep business v2 export written to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
