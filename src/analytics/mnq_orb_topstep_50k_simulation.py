"""TopstepX 50K challenge simulation for MNQ ORB nominal vs sizing_3state."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.mnq_orb_prop_challenge_simulation import (
    DEFAULT_PRIMARY_SCOPE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_VARIANT_ORDER,
    VariantInput,
    _find_latest_source_run,
    _json_dump,
    _load_variant_input,
    _nan_mean,
    _nan_median,
    _normalize_daily_results,
    _read_run_metadata,
    _safe_div,
    _scope_daily_results,
    _source_is_fraction,
    _summary_row_map,
)
from src.config.paths import EXPORTS_DIR, ensure_directories


DEFAULT_BOOTSTRAP_PATHS = 2000
DEFAULT_BOOTSTRAP_BLOCK_SIZE = 5


@dataclass(frozen=True)
class TopstepRuleset:
    name: str
    description: str
    starting_balance_usd: float = 50_000.0
    profit_target_usd: float = 3_000.0
    trailing_mll_usd: float = 2_000.0
    consistency_share_limit: float = 0.50
    max_traded_days: int | None = None
    lock_at_starting_balance: bool = False
    notes: str = ""


@dataclass(frozen=True)
class TopstepCampaignSpec:
    source_run_root: Path | None = None
    primary_scope: str = DEFAULT_PRIMARY_SCOPE
    bootstrap_paths: int = DEFAULT_BOOTSTRAP_PATHS
    bootstrap_block_size: int = DEFAULT_BOOTSTRAP_BLOCK_SIZE
    random_seed: int = DEFAULT_RANDOM_SEED
    variant_names: tuple[str, ...] = DEFAULT_VARIANT_ORDER
    rulesets: tuple[TopstepRuleset, ...] = field(
        default_factory=lambda: (
            TopstepRuleset(
                name="topstepx_50k_main_35d",
                description="Main scenario with 35 traded-day expiry.",
                max_traded_days=35,
                lock_at_starting_balance=False,
                notes="Trailing MLL stays 2,000 below the running high-watermark; no breakeven lock is added.",
            ),
            TopstepRuleset(
                name="topstepx_50k_extended_60d",
                description="Longer horizon to separate risk geometry from speed / expiry pressure.",
                max_traded_days=60,
                lock_at_starting_balance=False,
                notes="Same Topstep-style risk rules, but with a longer traded-day horizon.",
            ),
        )
    )
    output_root: Path | None = None


def _eligible_start_dates(daily_results: pd.DataFrame, max_traded_days: int | None) -> list:
    ordered = _normalize_daily_results(daily_results)
    if max_traded_days is None:
        return ordered["session_date"].tolist()
    traded_mask = pd.to_numeric(ordered["daily_trade_count"], errors="coerce").fillna(0.0).gt(0).astype(int)
    remaining_traded = traded_mask.iloc[::-1].cumsum().iloc[::-1]
    return ordered.loc[remaining_traded >= int(max_traded_days), "session_date"].tolist()


def _sample_block_bootstrap_daily(
    daily_results: pd.DataFrame,
    ruleset: TopstepRuleset,
    block_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    ordered = _normalize_daily_results(daily_results)
    if ordered.empty:
        return ordered.copy()

    effective_block = max(1, min(int(block_size), len(ordered)))
    max_start = max(len(ordered) - effective_block, 0)
    sampled_rows: list[dict[str, Any]] = []
    traded_days = 0
    synthetic_idx = 0
    max_rows = len(ordered) * 4

    while True:
        start_idx = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        block = ordered.iloc[start_idx : start_idx + effective_block]
        for _, row in block.iterrows():
            synthetic_idx += 1
            row_dict = row.to_dict()
            row_dict["source_session_date"] = row_dict["session_date"]
            row_dict["session_date"] = synthetic_idx
            sampled_rows.append(row_dict)
            if float(row.get("daily_trade_count", 0.0)) > 0:
                traded_days += 1
            if ruleset.max_traded_days is not None and traded_days >= int(ruleset.max_traded_days):
                return pd.DataFrame(sampled_rows)
            if ruleset.max_traded_days is None and len(sampled_rows) >= len(ordered):
                return pd.DataFrame(sampled_rows)
            if len(sampled_rows) >= max_rows:
                return pd.DataFrame(sampled_rows)


def _trailing_floor(high_watermark: float, ruleset: TopstepRuleset) -> float:
    floor = float(high_watermark) - float(ruleset.trailing_mll_usd)
    if ruleset.lock_at_starting_balance:
        floor = min(floor, float(ruleset.starting_balance_usd))
    return float(floor)


def simulate_topstep_path(
    daily_results: pd.DataFrame,
    ruleset: TopstepRuleset,
    reference_account_size_usd: float = 50_000.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered = _normalize_daily_results(daily_results)
    scale = _safe_div(ruleset.starting_balance_usd, reference_account_size_usd, default=1.0)
    if abs(scale - 1.0) > 1e-12:
        ordered = ordered.copy()
        for column in ("daily_pnl_usd", "daily_gross_pnl_usd", "daily_fees_usd"):
            if column in ordered.columns:
                ordered[column] = pd.to_numeric(ordered[column], errors="coerce").fillna(0.0) * float(scale)

    equity = float(ruleset.starting_balance_usd)
    high_watermark = float(ruleset.starting_balance_usd)
    best_winning_day = 0.0
    traded_days = 0
    calendar_days = 0
    history_rows: list[dict[str, Any]] = []

    status = "open"
    failure_reason = ""
    days_to_pass = float("nan")
    days_to_fail = float("nan")
    economic_target_hit = False
    economic_target_day = float("nan")
    economic_target_immediate_validation = False
    delayed_pass_after_inconsistency = False
    failed_after_economic_target = False
    extra_traded_days_to_consistency = float("nan")
    trailing_mll_breached = False
    max_favorable_excursion = 0.0
    max_adverse_excursion = 0.0
    max_drawdown_usd = 0.0

    for _, row in ordered.iterrows():
        calendar_days += 1
        session_date = pd.to_datetime(row["session_date"]).date()
        daily_pnl = float(pd.to_numeric(pd.Series([row.get("daily_pnl_usd", 0.0)]), errors="coerce").iloc[0])
        traded = bool(float(pd.to_numeric(pd.Series([row.get("daily_trade_count", 0.0)]), errors="coerce").iloc[0]) > 0.0)
        if traded:
            traded_days += 1

        equity += daily_pnl
        high_watermark = max(high_watermark, equity)
        trailing_floor = _trailing_floor(high_watermark=high_watermark, ruleset=ruleset)
        cumulative_profit = equity - float(ruleset.starting_balance_usd)
        best_winning_day = max(best_winning_day, max(daily_pnl, 0.0))
        best_day_share = float("nan")
        if cumulative_profit > 0 and best_winning_day > 0:
            best_day_share = float(best_winning_day / cumulative_profit)
        consistency_satisfied = bool(
            cumulative_profit >= float(ruleset.profit_target_usd)
            and best_winning_day < float(ruleset.consistency_share_limit) * cumulative_profit
        )
        if not economic_target_hit and cumulative_profit >= float(ruleset.profit_target_usd):
            economic_target_hit = True
            economic_target_day = float(traded_days)
            economic_target_immediate_validation = consistency_satisfied

        if economic_target_hit and not economic_target_immediate_validation and consistency_satisfied:
            delayed_pass_after_inconsistency = True
            extra_traded_days_to_consistency = float(traded_days) - float(economic_target_day)

        trailing_mll_breached = bool(equity <= trailing_floor)
        if economic_target_hit and trailing_mll_breached and not consistency_satisfied:
            failed_after_economic_target = True

        max_favorable_excursion = max(max_favorable_excursion, cumulative_profit)
        max_adverse_excursion = min(max_adverse_excursion, cumulative_profit)
        current_drawdown = float(equity - high_watermark)
        max_drawdown_usd = min(max_drawdown_usd, current_drawdown)

        history_rows.append(
            {
                "session_date": session_date,
                "daily_pnl_usd": daily_pnl,
                "equity": equity,
                "high_watermark": high_watermark,
                "trailing_floor_usd": trailing_floor,
                "cumulative_profit_usd": cumulative_profit,
                "best_winning_day_usd": best_winning_day,
                "best_day_share": best_day_share,
                "economic_target_hit": economic_target_hit,
                "consistency_satisfied": consistency_satisfied,
                "trailing_mll_breached": trailing_mll_breached,
                "traded_days_elapsed": traded_days,
                "calendar_days_elapsed": calendar_days,
            }
        )

        if trailing_mll_breached:
            status = "fail"
            failure_reason = "trailing_mll"
            days_to_fail = float(traded_days)
            break

        if consistency_satisfied:
            status = "pass"
            days_to_pass = float(traded_days)
            break

        if ruleset.max_traded_days is not None and traded_days >= int(ruleset.max_traded_days):
            status = "expire"
            break

    if status == "open":
        status = "expire" if ruleset.max_traded_days is not None else "fail"
        if status == "fail":
            failure_reason = "insufficient_history"
            days_to_fail = float(traded_days)

    history = pd.DataFrame(history_rows)
    return history, {
        "status": status,
        "pass": bool(status == "pass"),
        "fail": bool(status == "fail"),
        "expire": bool(status == "expire"),
        "failure_reason": failure_reason,
        "days_to_pass": days_to_pass,
        "days_to_fail": days_to_fail,
        "days_traded": int(traded_days),
        "calendar_days": int(calendar_days),
        "final_pnl_usd": float(equity - float(ruleset.starting_balance_usd)),
        "trailing_mll_breached": bool(trailing_mll_breached),
        "economic_target_hit": bool(economic_target_hit),
        "economic_target_day": economic_target_day,
        "economic_target_immediate_validation": bool(economic_target_immediate_validation),
        "economic_target_without_immediate_validation": bool(economic_target_hit and not economic_target_immediate_validation),
        "delayed_pass_after_inconsistency": bool(delayed_pass_after_inconsistency),
        "failed_after_economic_target": bool(failed_after_economic_target),
        "near_validation_then_fail": bool(failed_after_economic_target),
        "extra_traded_days_to_consistency": extra_traded_days_to_consistency,
        "best_winning_day_usd": float(best_winning_day),
        "best_day_share_at_end": float(best_winning_day / (equity - float(ruleset.starting_balance_usd)))
        if (equity - float(ruleset.starting_balance_usd)) > 0 and best_winning_day > 0
        else float("nan"),
        "max_favorable_excursion_usd": float(max_favorable_excursion),
        "max_adverse_excursion_usd": float(max_adverse_excursion),
        "max_drawdown_usd": float(max_drawdown_usd),
    }


def run_rolling_start_simulations(
    daily_results: pd.DataFrame,
    variant: VariantInput,
    ruleset: TopstepRuleset,
    common_start_dates: list,
) -> pd.DataFrame:
    ordered = _normalize_daily_results(daily_results)
    allowed_dates = set(pd.to_datetime(pd.Index(common_start_dates)).date)
    rows: list[dict[str, Any]] = []
    for idx, session_date in enumerate(ordered["session_date"]):
        if session_date not in allowed_dates:
            continue
        subset = ordered.iloc[idx:].copy().reset_index(drop=True)
        _, result = simulate_topstep_path(
            daily_results=subset,
            ruleset=ruleset,
            reference_account_size_usd=variant.reference_account_size_usd,
        )
        rows.append(
            {
                "simulation_method": "rolling_start",
                "variant_name": variant.variant_name,
                "variant_label": variant.label,
                "ruleset_name": ruleset.name,
                "run_id": f"rolling_{idx}",
                "start_session_date": session_date,
                **result,
            }
        )
    return pd.DataFrame(rows)


def run_bootstrap_simulations(
    daily_results: pd.DataFrame,
    variant: VariantInput,
    ruleset: TopstepRuleset,
    bootstrap_paths: int,
    block_size: int,
    random_seed: int,
) -> pd.DataFrame:
    ordered = _normalize_daily_results(daily_results)
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    for path_idx in range(int(bootstrap_paths)):
        sampled = _sample_block_bootstrap_daily(
            daily_results=ordered,
            ruleset=ruleset,
            block_size=block_size,
            rng=rng,
        )
        _, result = simulate_topstep_path(
            daily_results=sampled,
            ruleset=ruleset,
            reference_account_size_usd=variant.reference_account_size_usd,
        )
        rows.append(
            {
                "simulation_method": f"bootstrap_block_{int(block_size)}",
                "variant_name": variant.variant_name,
                "variant_label": variant.label,
                "ruleset_name": ruleset.name,
                "run_id": f"bootstrap_{path_idx + 1}",
                "start_session_date": pd.NA,
                **result,
            }
        )
    return pd.DataFrame(rows)


def aggregate_topstep_runs(runs: pd.DataFrame) -> dict[str, Any]:
    if runs.empty:
        return {
            "run_count": 0,
            "pass_rate": 0.0,
            "fail_rate": 0.0,
            "expire_rate": 0.0,
            "median_days_to_pass": float("nan"),
            "mean_days_to_pass": float("nan"),
            "median_days_to_fail": float("nan"),
            "mean_days_to_fail": float("nan"),
            "trailing_mll_breach_rate": 0.0,
            "economic_target_hit_rate": 0.0,
            "economic_target_without_immediate_validation_rate": 0.0,
            "delayed_pass_after_inconsistency_rate": 0.0,
            "failed_after_economic_target_rate": 0.0,
            "near_validation_then_fail_rate": 0.0,
            "median_extra_traded_days_to_consistency": float("nan"),
            "mean_extra_traded_days_to_consistency": float("nan"),
            "median_final_pnl_usd": float("nan"),
            "mean_final_pnl_usd": float("nan"),
            "median_max_drawdown_usd": float("nan"),
            "worst_max_drawdown_usd": float("nan"),
        }

    pass_mask = runs["pass"].astype(bool)
    fail_mask = runs["fail"].astype(bool)
    expire_mask = runs["expire"].astype(bool)
    delayed_mask = runs["delayed_pass_after_inconsistency"].astype(bool)
    return {
        "run_count": int(len(runs)),
        "pass_rate": float(pass_mask.mean()),
        "fail_rate": float(fail_mask.mean()),
        "expire_rate": float(expire_mask.mean()),
        "median_days_to_pass": _nan_median(runs.loc[pass_mask, "days_to_pass"]),
        "mean_days_to_pass": _nan_mean(runs.loc[pass_mask, "days_to_pass"]),
        "median_days_to_fail": _nan_median(runs.loc[fail_mask, "days_to_fail"]),
        "mean_days_to_fail": _nan_mean(runs.loc[fail_mask, "days_to_fail"]),
        "trailing_mll_breach_rate": float(runs["trailing_mll_breached"].astype(bool).mean()),
        "economic_target_hit_rate": float(runs["economic_target_hit"].astype(bool).mean()),
        "economic_target_without_immediate_validation_rate": float(
            runs["economic_target_without_immediate_validation"].astype(bool).mean()
        ),
        "delayed_pass_after_inconsistency_rate": float(runs["delayed_pass_after_inconsistency"].astype(bool).mean()),
        "failed_after_economic_target_rate": float(runs["failed_after_economic_target"].astype(bool).mean()),
        "near_validation_then_fail_rate": float(runs["near_validation_then_fail"].astype(bool).mean()),
        "median_extra_traded_days_to_consistency": _nan_median(runs.loc[delayed_mask, "extra_traded_days_to_consistency"]),
        "mean_extra_traded_days_to_consistency": _nan_mean(runs.loc[delayed_mask, "extra_traded_days_to_consistency"]),
        "median_final_pnl_usd": _nan_median(runs["final_pnl_usd"]),
        "mean_final_pnl_usd": _nan_mean(runs["final_pnl_usd"]),
        "median_max_drawdown_usd": _nan_median(runs["max_drawdown_usd"]),
        "worst_max_drawdown_usd": float(pd.to_numeric(runs["max_drawdown_usd"], errors="coerce").min()),
    }


def _prefixed(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def build_comparison_table(summary_variants: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ruleset_name, df in summary_variants.groupby("ruleset_name", sort=True):
        nominal = df.loc[df["variant_name"] == "nominal"]
        sizing = df.loc[df["variant_name"] == "sizing_3state_realized_vol_ratio_15_60"]
        if nominal.empty or sizing.empty:
            continue
        nominal_row = nominal.iloc[0]
        sizing_row = sizing.iloc[0]

        rolling_pass_edge = float(sizing_row["rolling_pass_rate"] - nominal_row["rolling_pass_rate"])
        rolling_fail_edge = float(sizing_row["rolling_fail_rate"] - nominal_row["rolling_fail_rate"])
        pass_edge = float(sizing_row["bootstrap_pass_rate"] - nominal_row["bootstrap_pass_rate"])
        fail_edge = float(sizing_row["bootstrap_fail_rate"] - nominal_row["bootstrap_fail_rate"])
        speed_edge = float(
            pd.to_numeric(pd.Series([nominal_row["bootstrap_median_days_to_pass"]]), errors="coerce").fillna(np.nan).iloc[0]
            - pd.to_numeric(pd.Series([sizing_row["bootstrap_median_days_to_pass"]]), errors="coerce").fillna(np.nan).iloc[0]
        )
        consistency_edge = float(
            sizing_row["bootstrap_economic_target_without_immediate_validation_rate"]
            - nominal_row["bootstrap_economic_target_without_immediate_validation_rate"]
        )
        consistency_inactive = bool(
            max(
                float(nominal_row["bootstrap_economic_target_without_immediate_validation_rate"]),
                float(sizing_row["bootstrap_economic_target_without_immediate_validation_rate"]),
                float(nominal_row["rolling_economic_target_without_immediate_validation_rate"]),
                float(sizing_row["rolling_economic_target_without_immediate_validation_rate"]),
            )
            == 0.0
        )

        if consistency_inactive and rolling_pass_edge <= -0.03 and pass_edge >= 0.03:
            verdict = "consistency neutre; nominal meilleur sur rolling reel, sizing meilleur en bootstrap"
        elif consistency_inactive and rolling_pass_edge <= -0.03 and pass_edge <= -0.03:
            verdict = "consistency neutre; TopstepX 50K favorise encore le nominal"
        elif consistency_inactive and rolling_pass_edge >= 0.03 and pass_edge >= 0.03 and fail_edge <= -0.05:
            verdict = "consistency neutre; TopstepX 50K favorise sizing_3state"
        elif pass_edge >= 0.03 and fail_edge <= -0.05:
            verdict = "TopstepX 50K favorise sizing_3state"
        elif pass_edge <= -0.03 and fail_edge >= -0.05:
            verdict = "TopstepX 50K favorise encore le nominal"
        elif consistency_edge <= -0.05 and fail_edge <= -0.05:
            verdict = "la consistency target favorise sizing_3state"
        else:
            verdict = "la hierarchie depend de l'horizon"

        rows.append(
            {
                "ruleset_name": ruleset_name,
                "nominal_rolling_pass_rate": nominal_row["rolling_pass_rate"],
                "sizing_rolling_pass_rate": sizing_row["rolling_pass_rate"],
                "nominal_rolling_fail_rate": nominal_row["rolling_fail_rate"],
                "sizing_rolling_fail_rate": sizing_row["rolling_fail_rate"],
                "nominal_bootstrap_pass_rate": nominal_row["bootstrap_pass_rate"],
                "sizing_bootstrap_pass_rate": sizing_row["bootstrap_pass_rate"],
                "nominal_bootstrap_fail_rate": nominal_row["bootstrap_fail_rate"],
                "sizing_bootstrap_fail_rate": sizing_row["bootstrap_fail_rate"],
                "nominal_bootstrap_inconsistency_rate": nominal_row["bootstrap_economic_target_without_immediate_validation_rate"],
                "sizing_bootstrap_inconsistency_rate": sizing_row["bootstrap_economic_target_without_immediate_validation_rate"],
                "nominal_bootstrap_delayed_pass_rate": nominal_row["bootstrap_delayed_pass_after_inconsistency_rate"],
                "sizing_bootstrap_delayed_pass_rate": sizing_row["bootstrap_delayed_pass_after_inconsistency_rate"],
                "nominal_bootstrap_failed_after_target_rate": nominal_row["bootstrap_failed_after_economic_target_rate"],
                "sizing_bootstrap_failed_after_target_rate": sizing_row["bootstrap_failed_after_economic_target_rate"],
                "rolling_pass_rate_edge_sizing_minus_nominal": rolling_pass_edge,
                "rolling_fail_rate_edge_sizing_minus_nominal": rolling_fail_edge,
                "pass_rate_edge_sizing_minus_nominal": pass_edge,
                "fail_rate_edge_sizing_minus_nominal": fail_edge,
                "speed_edge_nominal_minus_sizing_days_to_pass": speed_edge,
                "inconsistency_edge_sizing_minus_nominal": consistency_edge,
                "consistency_inactive_in_sample": consistency_inactive,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def _overall_verdict(comparison_table: pd.DataFrame) -> str:
    if comparison_table.empty:
        return "aucune conclusion exploitable"
    verdicts = comparison_table["verdict"].tolist()
    if all(bool(value) for value in comparison_table.get("consistency_inactive_in_sample", pd.Series(dtype=bool)).tolist()):
        if all("nominal meilleur sur rolling reel, sizing meilleur en bootstrap" in verdict for verdict in verdicts):
            return "la consistency target ne change pas la hierarchie; nominal reste meilleur sur l'historique reel, sizing_3state domine le bootstrap et la survivabilite simulee"
        if any("nominal meilleur sur rolling reel, sizing meilleur en bootstrap" in verdict for verdict in verdicts):
            return "la consistency target ne change pas la hierarchie; nominal reste meilleur sur l'historique reel, sizing_3state est meilleur en bootstrap, donc la conclusion depend surtout de l'horizon et de la confiance accordee au resampling"
        if all("nominal" in verdict for verdict in verdicts):
            return "TopstepX 50K favorise encore le nominal et la consistency target ne change rien"
        if all("sizing_3state" in verdict for verdict in verdicts):
            return "TopstepX 50K favorise sizing_3state et la consistency target ne change rien"
    if all("nominal" in verdict for verdict in verdicts):
        return "TopstepX 50K favorise encore le nominal"
    if all("sizing_3state" in verdict for verdict in verdicts):
        return "la consistency target fait basculer l'avantage vers sizing_3state"
    if any("sizing meilleur pour survivre" in verdict for verdict in verdicts):
        return "nominal meilleur pour passer vite, sizing meilleur pour survivre, hierarchie finale depend de l'horizon"
    return "la hierarchie depend fortement de l'horizon"


def _ruleset_summary_table(rulesets: tuple[TopstepRuleset, ...]) -> pd.DataFrame:
    return pd.DataFrame([asdict(ruleset) for ruleset in rulesets])


def _build_markdown_summary(
    spec: TopstepCampaignSpec,
    source_root: Path,
    metadata: dict[str, Any],
    comparison_table: pd.DataFrame,
    summary_variants: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# MNQ ORB TopstepX 50K Simulation",
            "",
            "## Scope",
            "",
            f"- Source export: `{source_root}`",
            f"- Scope principal: `{spec.primary_scope}`",
            f"- Bootstrap: `{spec.bootstrap_paths}` paths, block size `{spec.bootstrap_block_size}`",
            f"- Dataset: `{metadata.get('dataset_path')}`",
            "- Source of truth rules: 50K start, +3K target, trailing MLL 2K, no daily loss limit, consistency 50%.",
            "- Assumption retained: no breakeven lock is added to the trailing floor; it remains 2,000 below the running high-watermark.",
            "",
            "## Comparison",
            "",
            "```text",
            comparison_table.to_string(index=False),
            "```",
            "",
            "## Variant Summary",
            "",
            "```text",
            summary_variants[
                [
                    "ruleset_name",
                    "variant_name",
                    "rolling_pass_rate",
                    "rolling_fail_rate",
                    "rolling_expire_rate",
                    "rolling_economic_target_without_immediate_validation_rate",
                    "rolling_delayed_pass_after_inconsistency_rate",
                    "bootstrap_pass_rate",
                    "bootstrap_fail_rate",
                    "bootstrap_expire_rate",
                    "bootstrap_economic_target_without_immediate_validation_rate",
                    "bootstrap_delayed_pass_after_inconsistency_rate",
                ]
            ].to_string(index=False),
            "```",
            "",
            "## Verdict",
            "",
            f"- `{_overall_verdict(comparison_table)}`",
            "",
        ]
    )


def run_campaign(spec: TopstepCampaignSpec) -> dict[str, Path]:
    ensure_directories()
    source_root = spec.source_run_root or _find_latest_source_run()
    metadata = _read_run_metadata(source_root)
    is_fraction = _source_is_fraction(metadata)
    summary_rows = _summary_row_map(source_root)
    variant_inputs = [
        _load_variant_input(source_root=source_root, variant_name=variant_name, summary_rows=summary_rows)
        for variant_name in spec.variant_names
    ]
    scoped_daily_map = {
        variant.variant_name: _scope_daily_results(variant.daily_results, is_fraction=is_fraction, scope=spec.primary_scope)
        for variant in variant_inputs
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = spec.output_root or (EXPORTS_DIR / f"mnq_orb_topstep_50k_simulation_{timestamp}")
    output_root.mkdir(parents=True, exist_ok=True)

    rolling_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    summary_rows_combined: list[dict[str, Any]] = []
    all_runs: list[pd.DataFrame] = []

    for ruleset_idx, ruleset in enumerate(spec.rulesets):
        common_dates: set | None = None
        for variant in variant_inputs:
            variant_dates = set(_eligible_start_dates(scoped_daily_map[variant.variant_name], max_traded_days=ruleset.max_traded_days))
            common_dates = variant_dates if common_dates is None else (common_dates & variant_dates)
        ordered_common_dates = sorted(common_dates) if common_dates else []

        for variant in variant_inputs:
            scoped_daily = scoped_daily_map[variant.variant_name]
            rolling_runs = run_rolling_start_simulations(
                daily_results=scoped_daily,
                variant=variant,
                ruleset=ruleset,
                common_start_dates=ordered_common_dates,
            )
            bootstrap_runs = run_bootstrap_simulations(
                daily_results=scoped_daily,
                variant=variant,
                ruleset=ruleset,
                bootstrap_paths=spec.bootstrap_paths,
                block_size=spec.bootstrap_block_size,
                random_seed=spec.random_seed + (ruleset_idx * 1000),
            )
            all_runs.extend([rolling_runs, bootstrap_runs])

            rolling_metrics = aggregate_topstep_runs(rolling_runs)
            bootstrap_metrics = aggregate_topstep_runs(bootstrap_runs)
            rolling_rows.append({"ruleset_name": ruleset.name, "variant_name": variant.variant_name, **rolling_metrics})
            bootstrap_rows.append({"ruleset_name": ruleset.name, "variant_name": variant.variant_name, **bootstrap_metrics})
            summary_rows_combined.append(
                {
                    "ruleset_name": ruleset.name,
                    "variant_name": variant.variant_name,
                    "variant_label": variant.label,
                    **_prefixed("rolling", rolling_metrics),
                    **_prefixed("bootstrap", bootstrap_metrics),
                }
            )

    rolling_summary = pd.DataFrame(rolling_rows)
    bootstrap_summary = pd.DataFrame(bootstrap_rows)
    summary_variants = pd.DataFrame(summary_rows_combined)
    comparison_table = build_comparison_table(summary_variants)
    simulation_runs = pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame()
    ruleset_summary = _ruleset_summary_table(spec.rulesets)

    ruleset_path = output_root / "topstep_ruleset_summary.csv"
    summary_variants_path = output_root / "summary_variants.csv"
    comparison_path = output_root / "comparison_table.csv"
    runs_path = output_root / "simulation_runs.csv"
    rolling_path = output_root / "rolling_start_summary.csv"
    bootstrap_path = output_root / "bootstrap_summary.csv"
    markdown_path = output_root / "topstep_50k_summary.md"
    metadata_path = output_root / "run_metadata.json"

    ruleset_summary.to_csv(ruleset_path, index=False)
    summary_variants.to_csv(summary_variants_path, index=False)
    comparison_table.to_csv(comparison_path, index=False)
    simulation_runs.to_csv(runs_path, index=False)
    rolling_summary.to_csv(rolling_path, index=False)
    bootstrap_summary.to_csv(bootstrap_path, index=False)
    markdown_path.write_text(
        _build_markdown_summary(
            spec=spec,
            source_root=source_root,
            metadata=metadata,
            comparison_table=comparison_table,
            summary_variants=summary_variants,
        ),
        encoding="utf-8",
    )
    _json_dump(
        metadata_path,
        {
            "run_timestamp": datetime.now().isoformat(),
            "source_run_root": str(source_root),
            "primary_scope": spec.primary_scope,
            "bootstrap_paths": spec.bootstrap_paths,
            "bootstrap_block_size": spec.bootstrap_block_size,
            "overall_verdict": _overall_verdict(comparison_table),
            "rulesets": [asdict(ruleset) for ruleset in spec.rulesets],
        },
    )

    return {
        "topstep_ruleset_summary_csv": ruleset_path,
        "summary_variants_csv": summary_variants_path,
        "comparison_table_csv": comparison_path,
        "simulation_runs_csv": runs_path,
        "rolling_start_summary_csv": rolling_path,
        "bootstrap_summary_csv": bootstrap_path,
        "summary_markdown": markdown_path,
        "run_metadata_json": metadata_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MNQ ORB TopstepX 50K simulation.")
    parser.add_argument("--source-run-root", type=Path, default=None)
    parser.add_argument("--primary-scope", choices=("overall", "oos"), default=DEFAULT_PRIMARY_SCOPE)
    parser.add_argument("--bootstrap-paths", type=int, default=DEFAULT_BOOTSTRAP_PATHS)
    parser.add_argument("--bootstrap-block-size", type=int, default=DEFAULT_BOOTSTRAP_BLOCK_SIZE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_campaign(
        TopstepCampaignSpec(
            source_run_root=args.source_run_root,
            primary_scope=str(args.primary_scope),
            bootstrap_paths=int(args.bootstrap_paths),
            bootstrap_block_size=int(args.bootstrap_block_size),
            random_seed=int(args.random_seed),
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
