"""Local refinement campaign around the best symbol-specific risk-sizing zone."""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analytics.volume_climax_pullback_common import load_symbol_data, resample_rth_1h, split_sessions
from src.analytics.volume_climax_pullback_mnq_risk_sizing_campaign import (
    DEFAULT_INITIAL_CAPITAL_USD,
    DEFAULT_SYMBOL,
    _concat_non_empty_frames,
    _daily_results_from_trades,
    _json_dump,
    _phase_map,
    _resolve_base_alpha_variant,
    _risk_label,
    _subset_by_sessions,
    _summarize_scope,
)
from src.config.paths import EXPORTS_DIR, ensure_directories
from src.engine.volume_climax_pullback_v2_backtester import run_volume_climax_pullback_v2_backtest
from src.engine.vwap_backtester import build_execution_model_for_profile
from src.risk.position_sizing import FixedContractPositionSizing, PositionSizingConfig, RiskPercentPositionSizing
from src.strategy.volume_climax_pullback_v2 import (
    build_volume_climax_pullback_v2_signal_frame,
    prepare_volume_climax_pullback_v2_features,
)


DEFAULT_OUTPUT_PREFIX = "volume_climax_pullback_mnq_risk_sizing_refinement_"
DEFAULT_RISK_PCTS = (0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040)
DEFAULT_MAX_CONTRACTS = (2, 3, 4, 5, 6)
DEFAULT_SKIP_TRADE_IF_TOO_SMALL = True
BEST_PREVIOUS_WINNER_NAME = "risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true"
BEST_PREVIOUS_WINNER_ALIAS = "best_previous_winner"
BEST_PREVIOUS_WINNER_RISK_PCT = 0.0025
BEST_PREVIOUS_WINNER_MAX_CONTRACTS = 3


@dataclass(frozen=True)
class RefinementVariantSpec:
    campaign_variant_name: str
    variant_role: str
    sizing_mode: str
    fixed_contracts: int | None
    risk_pct: float | None
    max_contracts: int | None
    skip_trade_if_too_small: bool | None
    initial_capital_usd: float
    position_sizing: PositionSizingConfig


def build_refinement_variants(
    *,
    initial_capital_usd: float,
    risk_pcts: tuple[float, ...] = DEFAULT_RISK_PCTS,
    max_contracts_grid: tuple[int, ...] = DEFAULT_MAX_CONTRACTS,
    include_best_previous_winner_alias: bool = True,
) -> list[RefinementVariantSpec]:
    variants = [
        RefinementVariantSpec(
            campaign_variant_name="fixed_1_contract",
            variant_role="baseline",
            sizing_mode="fixed_contracts",
            fixed_contracts=1,
            risk_pct=None,
            max_contracts=None,
            skip_trade_if_too_small=None,
            initial_capital_usd=float(initial_capital_usd),
            position_sizing=FixedContractPositionSizing(fixed_contracts=1),
        )
    ]
    for risk_pct in risk_pcts:
        for max_contracts in max_contracts_grid:
            variants.append(
                RefinementVariantSpec(
                    campaign_variant_name=(
                        f"risk_pct_{_risk_label(risk_pct)}"
                        f"__max_contracts_{int(max_contracts)}"
                        "__skip_trade_if_too_small_true"
                    ),
                    variant_role="grid",
                    sizing_mode="risk_percent",
                    fixed_contracts=None,
                    risk_pct=float(risk_pct),
                    max_contracts=int(max_contracts),
                    skip_trade_if_too_small=True,
                    initial_capital_usd=float(initial_capital_usd),
                    position_sizing=RiskPercentPositionSizing(
                        initial_capital_usd=float(initial_capital_usd),
                        risk_pct=float(risk_pct),
                        max_contracts=int(max_contracts),
                        skip_trade_if_too_small=True,
                    ),
                )
            )

    if include_best_previous_winner_alias:
        variants.append(
            RefinementVariantSpec(
                campaign_variant_name=BEST_PREVIOUS_WINNER_ALIAS,
                variant_role="best_previous_winner",
                sizing_mode="risk_percent",
                fixed_contracts=None,
                risk_pct=float(BEST_PREVIOUS_WINNER_RISK_PCT),
                max_contracts=int(BEST_PREVIOUS_WINNER_MAX_CONTRACTS),
                skip_trade_if_too_small=True,
                initial_capital_usd=float(initial_capital_usd),
                position_sizing=RiskPercentPositionSizing(
                    initial_capital_usd=float(initial_capital_usd),
                    risk_pct=float(BEST_PREVIOUS_WINNER_RISK_PCT),
                    max_contracts=int(BEST_PREVIOUS_WINNER_MAX_CONTRACTS),
                    skip_trade_if_too_small=True,
                ),
            )
        )
    return variants


def compute_prop_score(row: pd.Series | dict[str, Any], *, prefix: str = "oos") -> float:
    series = row if isinstance(row, pd.Series) else pd.Series(row)
    base = f"{prefix}_" if prefix and not str(prefix).endswith("_") else str(prefix)

    def _float(name: str) -> float:
        value = pd.to_numeric(series.get(f"{base}{name}"), errors="coerce")
        return float(value) if pd.notna(value) else 0.0

    pass_flag = bool(series.get(f"{base}pass_target_3000_usd_without_breaching_2000_dd", False))
    net_bonus = min(max(_float("net_pnl_usd"), 0.0) / 3_000.0, 2.0) * 4.0
    sharpe_bonus = min(max(_float("sharpe"), 0.0), 2.5) * 3.0
    maxdd_penalty = min(max(_float("max_drawdown_usd"), 0.0) / 2_000.0, 3.0) * 3.0
    daily_dd_penalty = min(max(_float("max_daily_drawdown_usd"), 0.0) / 1_000.0, 3.0) * 5.0
    worst_trade_penalty = min(max(_float("worst_trade_loss_usd"), 0.0) / 200.0, 3.0) * 2.0
    below_500_penalty = min(max(_float("nb_days_below_minus_500"), 0.0), 5.0) * 0.5
    pass_penalty = 6.0 if not pass_flag else 0.0
    return float(
        net_bonus
        + sharpe_bonus
        - maxdd_penalty
        - daily_dd_penalty
        - worst_trade_penalty
        - below_500_penalty
        - pass_penalty
    )


def identify_connected_clusters(
    frame: pd.DataFrame,
    *,
    eligible_column: str = "is_top_quartile_prop_score",
    connectivity: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    if work.empty:
        return work.assign(cluster_id=pd.Series(dtype="Int64")), pd.DataFrame()

    risk_values = sorted(pd.to_numeric(work["risk_pct"], errors="coerce").dropna().unique().tolist())
    cap_values = sorted(pd.to_numeric(work["max_contracts"], errors="coerce").dropna().unique().tolist())
    risk_pos = {float(value): index for index, value in enumerate(risk_values)}
    cap_pos = {int(value): index for index, value in enumerate(cap_values)}

    work["risk_idx"] = pd.to_numeric(work["risk_pct"], errors="coerce").map(
        lambda value: risk_pos.get(float(value)) if pd.notna(value) else np.nan
    )
    work["cap_idx"] = pd.to_numeric(work["max_contracts"], errors="coerce").map(
        lambda value: cap_pos.get(int(value)) if pd.notna(value) else np.nan
    )
    work["cluster_id"] = pd.Series(pd.NA, index=work.index, dtype="Int64")

    eligible = pd.Series(work.get(eligible_column), dtype="boolean").fillna(False)
    node_lookup: dict[tuple[int, int], int] = {}
    for row in work.loc[eligible].itertuples():
        if pd.isna(row.risk_idx) or pd.isna(row.cap_idx):
            continue
        node_lookup[(int(row.risk_idx), int(row.cap_idx))] = int(row.Index)

    if connectivity == 4:
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        neighbors = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        )

    visited: set[tuple[int, int]] = set()
    cluster_rows: list[dict[str, Any]] = []
    cluster_id = 0

    for node in sorted(node_lookup):
        if node in visited:
            continue
        cluster_id += 1
        queue: deque[tuple[int, int]] = deque([node])
        members: list[int] = []
        visited.add(node)

        while queue:
            current = queue.popleft()
            members.append(node_lookup[current])
            for dr, dc in neighbors:
                candidate = (current[0] + dr, current[1] + dc)
                if candidate in node_lookup and candidate not in visited:
                    visited.add(candidate)
                    queue.append(candidate)

        work.loc[members, "cluster_id"] = cluster_id
        cluster_view = work.loc[members].copy()
        best_row = cluster_view.sort_values(
            ["oos_prop_score", "oos_net_pnl_usd", "oos_sharpe", "oos_max_drawdown_usd"],
            ascending=[False, False, False, True],
        ).iloc[0]
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": int(len(cluster_view)),
                "center_risk_pct": float(pd.to_numeric(cluster_view["risk_pct"], errors="coerce").mean()),
                "center_max_contracts": float(pd.to_numeric(cluster_view["max_contracts"], errors="coerce").mean()),
                "risk_pct_min": float(pd.to_numeric(cluster_view["risk_pct"], errors="coerce").min()),
                "risk_pct_max": float(pd.to_numeric(cluster_view["risk_pct"], errors="coerce").max()),
                "max_contracts_min": int(pd.to_numeric(cluster_view["max_contracts"], errors="coerce").min()),
                "max_contracts_max": int(pd.to_numeric(cluster_view["max_contracts"], errors="coerce").max()),
                "mean_oos_net_pnl_usd": float(pd.to_numeric(cluster_view["oos_net_pnl_usd"], errors="coerce").mean()),
                "mean_oos_sharpe": float(pd.to_numeric(cluster_view["oos_sharpe"], errors="coerce").mean()),
                "mean_oos_max_drawdown_usd": float(pd.to_numeric(cluster_view["oos_max_drawdown_usd"], errors="coerce").mean()),
                "mean_oos_prop_score": float(pd.to_numeric(cluster_view["oos_prop_score"], errors="coerce").mean()),
                "best_variant_name": str(best_row["campaign_variant_name"]),
                "best_variant_prop_score": float(best_row["oos_prop_score"]),
            }
        )

    cluster_summary = pd.DataFrame(cluster_rows)
    if not cluster_summary.empty:
        cluster_summary = cluster_summary.sort_values(
            ["cluster_size", "mean_oos_prop_score", "mean_oos_net_pnl_usd"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return work, cluster_summary


def _variant_cache_key(spec: RefinementVariantSpec) -> tuple[Any, ...]:
    if spec.sizing_mode == "fixed_contracts":
        return ("fixed_contracts", int(spec.fixed_contracts or 1))
    return (
        "risk_percent",
        float(spec.risk_pct or 0.0),
        int(spec.max_contracts or 0),
        bool(spec.skip_trade_if_too_small),
        float(spec.initial_capital_usd),
    )


def _contract_mix_metrics(trades: pd.DataFrame, sizing_decisions: pd.DataFrame) -> dict[str, float | int]:
    quantities = pd.to_numeric(trades.get("quantity"), errors="coerce").dropna() if not trades.empty else pd.Series(dtype=float)
    attempts = int(len(sizing_decisions))
    skipped = int(pd.Series(sizing_decisions.get("skipped"), dtype="boolean").fillna(False).sum()) if not sizing_decisions.empty else 0
    return {
        "avg_contracts": float(quantities.mean()) if not quantities.empty else 0.0,
        "median_contracts": float(quantities.median()) if not quantities.empty else 0.0,
        "pct_trades_at_1_contract": float(quantities.eq(1).mean() * 100.0) if not quantities.empty else 0.0,
        "pct_trades_at_2_contracts": float(quantities.eq(2).mean() * 100.0) if not quantities.empty else 0.0,
        "pct_trades_at_3_plus_contracts": float(quantities.ge(3).mean() * 100.0) if not quantities.empty else 0.0,
        "nb_skipped_trades": skipped,
        "pct_skipped_trades": float(skipped / attempts * 100.0) if attempts > 0 else 0.0,
    }


def _extra_prop_metrics(trades: pd.DataFrame, daily_results: pd.DataFrame, base_summary: dict[str, Any]) -> dict[str, float | int]:
    pnl = pd.to_numeric(trades.get("net_pnl_usd"), errors="coerce").fillna(0.0) if not trades.empty else pd.Series(dtype=float)
    daily_pnl = pd.to_numeric(daily_results.get("daily_pnl_usd"), errors="coerce").fillna(0.0) if not daily_results.empty else pd.Series(dtype=float)
    return {
        "nb_days_below_minus_250": int((daily_pnl <= -250.0).sum()) if not daily_pnl.empty else 0,
        "nb_days_below_minus_500": int((daily_pnl <= -500.0).sum()) if not daily_pnl.empty else 0,
        "nb_days_below_minus_1000": int((daily_pnl <= -1_000.0).sum()) if not daily_pnl.empty else 0,
        "nb_trades_below_minus_150": int((pnl <= -150.0).sum()) if not pnl.empty else 0,
        "nb_trades_below_minus_250": int((pnl <= -250.0).sum()) if not pnl.empty else 0,
        "nb_trades_below_minus_500": int((pnl <= -500.0).sum()) if not pnl.empty else 0,
        "days_to_3000_usd_if_reached": base_summary.get("days_to_target_3000_usd", np.nan),
    }


def _summarize_scope_refinement(
    *,
    trades: pd.DataFrame,
    signal_df: pd.DataFrame,
    sessions: list,
    daily_results: pd.DataFrame,
    sizing_decisions: pd.DataFrame,
    initial_capital: float,
) -> dict[str, Any]:
    summary = _summarize_scope(
        trades=trades,
        signal_df=signal_df,
        sessions=sessions,
        daily_results=daily_results,
        sizing_decisions=sizing_decisions,
        initial_capital=initial_capital,
    )
    summary.update(_contract_mix_metrics(trades, sizing_decisions))
    summary.update(_extra_prop_metrics(trades, daily_results, summary))
    return summary


def _top_grid_oos_table(summary: pd.DataFrame, *, top_n: int = 10) -> str:
    cols = [
        "campaign_variant_name",
        "oos_net_pnl_usd",
        "oos_sharpe",
        "oos_max_drawdown_usd",
        "oos_max_daily_drawdown_usd",
        "oos_prop_score",
        "oos_pass_target_3000_usd_without_breaching_2000_dd",
    ]
    view = summary.loc[summary["variant_role"] == "grid", cols].copy()
    view = view.sort_values(
        ["oos_prop_score", "oos_net_pnl_usd", "oos_sharpe", "oos_max_drawdown_usd"],
        ascending=[False, False, False, True],
    ).head(top_n)
    return view.to_string(index=False)


def _guardrail_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.Series(frame["oos_pass_target_3000_usd_without_breaching_2000_dd"], dtype="boolean").fillna(False)
        & pd.to_numeric(frame["oos_max_daily_drawdown_usd"], errors="coerce").lt(800.0)
        & pd.to_numeric(frame["oos_worst_trade_loss_usd"], errors="coerce").lt(200.0)
    )


def _select_point(rows: pd.DataFrame) -> pd.Series | None:
    if rows.empty:
        return None
    ordered = rows.sort_values(
        ["oos_prop_score", "oos_net_pnl_usd", "oos_sharpe", "oos_max_drawdown_usd"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return ordered.iloc[0]


def _build_heatmap_metrics(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int | None]:
    heatmap = summary.loc[summary["variant_role"] == "grid"].copy().reset_index(drop=True)
    if heatmap.empty:
        return heatmap, pd.DataFrame(), None

    heatmap["is_guardrail_prop_safe"] = _guardrail_mask(heatmap)
    eligible_scores = pd.to_numeric(
        heatmap.loc[heatmap["is_guardrail_prop_safe"], "oos_prop_score"],
        errors="coerce",
    ).dropna()
    top_quartile_threshold = float(eligible_scores.quantile(0.75)) if not eligible_scores.empty else math.nan
    heatmap["top_quartile_prop_score_threshold"] = top_quartile_threshold
    heatmap["is_top_quartile_prop_score"] = (
        heatmap["is_guardrail_prop_safe"]
        & pd.to_numeric(heatmap["oos_prop_score"], errors="coerce").ge(top_quartile_threshold if math.isfinite(top_quartile_threshold) else math.inf)
    )

    heatmap, clusters = identify_connected_clusters(heatmap, eligible_column="is_top_quartile_prop_score", connectivity=8)
    recommended_cluster_id: int | None = None
    if not clusters.empty:
        recommended_cluster_id = int(clusters.iloc[0]["cluster_id"])
        clusters["is_recommended_cluster"] = clusters["cluster_id"].eq(recommended_cluster_id)
    else:
        clusters["is_recommended_cluster"] = pd.Series(dtype=bool)

    heatmap["is_recommended_cluster"] = (
        pd.Series(heatmap.get("cluster_id"), dtype="Int64").eq(recommended_cluster_id)
        if recommended_cluster_id is not None
        else False
    )
    return heatmap, clusters, recommended_cluster_id


def _save_heatmap_png(
    *,
    frame: pd.DataFrame,
    value_column: str,
    title: str,
    output_path: Path,
    cmap: str,
    value_format: str,
) -> None:
    risk_values = sorted(pd.to_numeric(frame["risk_pct"], errors="coerce").dropna().unique().tolist())
    max_values = sorted(pd.to_numeric(frame["max_contracts"], errors="coerce").dropna().unique().tolist())
    pivot = (
        frame.pivot_table(index="risk_pct", columns="max_contracts", values=value_column, aggfunc="mean")
        .reindex(index=risk_values, columns=max_values)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(max_values)))
    ax.set_xticklabels([str(int(value)) for value in max_values])
    ax.set_yticks(np.arange(len(risk_values)))
    ax.set_yticklabels([f"{float(value):.4f}" for value in risk_values])
    ax.set_xlabel("max_contracts")
    ax.set_ylabel("risk_pct")
    ax.set_title(title)

    for row_index in range(pivot.shape[0]):
        for col_index in range(pivot.shape[1]):
            value = pivot.iat[row_index, col_index]
            if pd.isna(value):
                continue
            ax.text(col_index, row_index, format(float(value), value_format), ha="center", va="center", color="black", fontsize=9)

    fig.colorbar(image, ax=ax, shrink=0.92)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_cluster_map_png(*, heatmap_metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    base = heatmap_metrics.copy()
    base["risk_pct"] = pd.to_numeric(base["risk_pct"], errors="coerce")
    base["max_contracts"] = pd.to_numeric(base["max_contracts"], errors="coerce")

    non_safe = base.loc[~pd.Series(base["is_guardrail_prop_safe"], dtype="boolean").fillna(False)]
    safe = base.loc[pd.Series(base["is_guardrail_prop_safe"], dtype="boolean").fillna(False)]
    top_quartile = base.loc[pd.Series(base["is_top_quartile_prop_score"], dtype="boolean").fillna(False)]
    recommended = base.loc[pd.Series(base["is_recommended_cluster"], dtype="boolean").fillna(False)]

    if not non_safe.empty:
        ax.scatter(non_safe["max_contracts"], non_safe["risk_pct"], label="non prop-safe", color="#9ca3af", s=90, marker="x")
    if not safe.empty:
        ax.scatter(safe["max_contracts"], safe["risk_pct"], label="prop-safe", color="#22c55e", s=70, alpha=0.7)
    if not top_quartile.empty:
        ax.scatter(top_quartile["max_contracts"], top_quartile["risk_pct"], label="top quartile prop_score", color="#f59e0b", s=150, facecolors="none", linewidths=1.8)
    if not recommended.empty:
        ax.scatter(recommended["max_contracts"], recommended["risk_pct"], label="recommended cluster", color="#ef4444", s=220, facecolors="none", linewidths=2.2)

    ax.set_xlabel("max_contracts")
    ax.set_ylabel("risk_pct")
    ax.set_title(f"{DEFAULT_SYMBOL} risk sizing refinement - robustness cluster map")
    ax.set_xticks(sorted(base["max_contracts"].dropna().astype(int).unique().tolist()))
    ax.set_yticks(sorted(base["risk_pct"].dropna().unique().tolist()))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _robustness_scale_text(cluster_size: int, total_grid_points: int, risk_span_count: int, cap_span_count: int) -> str:
    coverage = float(cluster_size / total_grid_points) if total_grid_points > 0 else 0.0
    if cluster_size <= 1:
        return "point_isole"
    if coverage >= 0.25 and risk_span_count >= 3 and cap_span_count >= 2:
        return "zone_large"
    if coverage >= 0.15 and risk_span_count >= 2 and cap_span_count >= 2:
        return "zone_locale_robuste"
    return "zone_etruite_mais_non_isolee"


def _slice_table(summary: pd.DataFrame, *, risk_pct: float | None = None, max_contracts: int | None = None) -> str:
    scoped = summary.loc[summary["variant_role"] == "grid"].copy()
    if risk_pct is not None:
        scoped = scoped.loc[np.isclose(pd.to_numeric(scoped["risk_pct"], errors="coerce"), float(risk_pct))]
        scoped = scoped.sort_values("max_contracts")
    if max_contracts is not None:
        scoped = scoped.loc[pd.to_numeric(scoped["max_contracts"], errors="coerce").astype("Int64").eq(int(max_contracts))]
        scoped = scoped.sort_values("risk_pct")
    cols = [
        "campaign_variant_name",
        "risk_pct",
        "max_contracts",
        "oos_net_pnl_usd",
        "oos_sharpe",
        "oos_max_drawdown_usd",
        "oos_max_daily_drawdown_usd",
        "oos_prop_score",
    ]
    return scoped[cols].to_string(index=False)


def _build_final_report(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    heatmap_metrics: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    base_variant_name: str,
    reference_v3_dir: Path | None,
    dataset_path: Path | None,
    is_sessions: list,
    oos_sessions: list,
) -> dict[str, Any]:
    baseline = summary.loc[summary["variant_role"] == "baseline"].iloc[0]
    point_winner = _select_point(heatmap_metrics)
    prop_safe_pool = heatmap_metrics.loc[heatmap_metrics["is_guardrail_prop_safe"]].copy()
    prop_safe_variant = _select_point(prop_safe_pool)
    aggressive_pool = summary.loc[
        (summary["variant_role"] == "grid")
        & pd.Series(summary["oos_pass_target_3000_usd_without_breaching_2000_dd"], dtype="boolean").fillna(False)
        & pd.to_numeric(summary["oos_max_daily_drawdown_usd"], errors="coerce").lt(1_000.0)
        & pd.to_numeric(summary["oos_worst_trade_loss_usd"], errors="coerce").lt(250.0)
    ].copy()
    if aggressive_pool.empty:
        aggressive_pool = summary.loc[
            (summary["variant_role"] == "grid")
            & pd.Series(summary["oos_pass_target_3000_usd_without_breaching_2000_dd"], dtype="boolean").fillna(False)
        ].copy()
    aggressive_variant = None
    if not aggressive_pool.empty:
        aggressive_variant = aggressive_pool.sort_values(
            ["oos_net_pnl_usd", "oos_prop_score", "oos_sharpe"],
            ascending=[False, False, False],
        ).iloc[0]

    recommended_cluster = cluster_summary.iloc[0] if not cluster_summary.empty else None
    recommended_variant = None
    verdict = "retenir_une_unique_variante"
    if recommended_cluster is not None:
        cluster_rows = heatmap_metrics.loc[
            pd.Series(heatmap_metrics["cluster_id"], dtype="Int64").eq(int(recommended_cluster["cluster_id"]))
        ].copy()
        recommended_variant = _select_point(cluster_rows)
        risk_span_count = int(cluster_rows["risk_pct"].nunique())
        cap_span_count = int(cluster_rows["max_contracts"].nunique())
        robustness_scale = _robustness_scale_text(
            int(recommended_cluster["cluster_size"]),
            int(len(heatmap_metrics)),
            risk_span_count,
            cap_span_count,
        )
        verdict = "retenir_une_zone_parametrique" if int(recommended_cluster["cluster_size"]) > 1 else "retenir_une_unique_variante"
    else:
        robustness_scale = "point_isole"
        recommended_variant = point_winner
        if recommended_variant is None:
            verdict = "point_precedent_trop_isole"

    best_previous = summary.loc[summary["campaign_variant_name"] == BEST_PREVIOUS_WINNER_ALIAS].iloc[0]

    lines = [
        f"# Volume Climax Pullback {DEFAULT_SYMBOL} Risk Sizing - Refinement Report",
        "",
        "## Scope",
        f"- Symbol: `{DEFAULT_SYMBOL}` only.",
        f"- Base alpha reused unchanged: `{base_variant_name}`.",
        f"- Reference V3 run: `{reference_v3_dir}`." if reference_v3_dir is not None else "- Reference V3 run: `not used`.",
        f"- Dataset: `{dataset_path}`." if dataset_path is not None else f"- Dataset: `repo latest {DEFAULT_SYMBOL} source`.",
        f"- Sessions: full `{len(is_sessions) + len(oos_sessions)}` | IS `{len(is_sessions)}` | OOS `{len(oos_sessions)}`.",
        "- Sizing logic unchanged from the prior campaign. Only the local grid around the previous winner was refined.",
        "",
        "## Prop Score",
        "- Formula on OOS: `prop_score = 4 * min(net_pnl/3000, 2) + 3 * min(sharpe, 2.5) - 3 * min(maxDD/2000, 3) - 5 * min(max_daily_DD/1000, 3) - 2 * min(worst_trade/200, 3) - 0.5 * nb_days_below_-500 - 6 * 1[pass=False]`.",
        "- Interpretation: reward enough OOS profit to clear a 50k target, reward clean Sharpe, penalize drawdown, penalize daily drawdown strongly, and penalize any non-pass configuration heavily.",
        "",
        "## OOS Winner",
    ]

    if point_winner is None:
        lines.append("- No valid OOS grid point was available.")
    else:
        lines.extend(
            [
                f"- Punctual OOS winner by `prop_score`: `{point_winner['campaign_variant_name']}`.",
                f"- OOS metrics: net `{float(point_winner['oos_net_pnl_usd']):.2f}` | CAGR `{float(point_winner['oos_cagr_pct']):.2f}%` | Sharpe `{float(point_winner['oos_sharpe']):.3f}` | maxDD `{float(point_winner['oos_max_drawdown_usd']):.2f}` | max daily DD `{float(point_winner['oos_max_daily_drawdown_usd']):.2f}` | prop_score `{float(point_winner['oos_prop_score']):.2f}`.",
                f"- Versus previous winner tag: net `{float(point_winner['oos_net_pnl_usd']) - float(best_previous['oos_net_pnl_usd']):+.2f}` | Sharpe `{float(point_winner['oos_sharpe']) - float(best_previous['oos_sharpe']):+.3f}` | maxDD `{float(point_winner['oos_max_drawdown_usd']) - float(best_previous['oos_max_drawdown_usd']):+.2f}`.",
            ]
        )

    lines.extend(["", "```text", _top_grid_oos_table(summary), "```", "", "## Robust Zone"])
    if recommended_cluster is None or recommended_variant is None:
        lines.append("- No multi-point robust cluster passed the guardrails plus top-quartile filter. The previous point looks isolated.")
    else:
        lines.extend(
            [
                f"- Recommended cluster id: `{int(recommended_cluster['cluster_id'])}` | size `{int(recommended_cluster['cluster_size'])}` | scale `{robustness_scale}`.",
                f"- Zone range: risk_pct `{float(recommended_cluster['risk_pct_min']):.4f}` -> `{float(recommended_cluster['risk_pct_max']):.4f}` | max_contracts `{int(recommended_cluster['max_contracts_min'])}` -> `{int(recommended_cluster['max_contracts_max'])}`.",
                f"- Zone center: risk_pct `{float(recommended_cluster['center_risk_pct']):.4f}` | max_contracts `{float(recommended_cluster['center_max_contracts']):.2f}`.",
                f"- Zone means: net `{float(recommended_cluster['mean_oos_net_pnl_usd']):.2f}` | Sharpe `{float(recommended_cluster['mean_oos_sharpe']):.3f}` | maxDD `{float(recommended_cluster['mean_oos_max_drawdown_usd']):.2f}` | prop_score `{float(recommended_cluster['mean_oos_prop_score']):.2f}`.",
                f"- Best point inside the zone: `{recommended_variant['campaign_variant_name']}`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Requested Readout",
            f"1. Le gagnant ponctuel OOS: `{point_winner['campaign_variant_name']}`." if point_winner is not None else "1. Le gagnant ponctuel OOS: `n/a`.",
            (
                f"2. La zone robuste OOS: cluster `{int(recommended_cluster['cluster_id'])}` couvrant risk_pct `{float(recommended_cluster['risk_pct_min']):.4f}` -> `{float(recommended_cluster['risk_pct_max']):.4f}` et cap `{int(recommended_cluster['max_contracts_min'])}` -> `{int(recommended_cluster['max_contracts_max'])}`."
                if recommended_cluster is not None
                else "2. La zone robuste OOS: aucune zone multi-point claire, le signal de robustesse reste local."
            ),
            f"3. Impact marginal de risk_pct autour de `{BEST_PREVIOUS_WINNER_RISK_PCT:.4f}`: voir la coupe `max_contracts={BEST_PREVIOUS_WINNER_MAX_CONTRACTS}` ci-dessous. La question utile est de savoir si le score reste propre quand on s'eloigne de `{BEST_PREVIOUS_WINNER_RISK_PCT:.4f}`.",
            f"4. Impact marginal du cap `max_contracts` autour de `{BEST_PREVIOUS_WINNER_MAX_CONTRACTS}`: voir la coupe `risk_pct={BEST_PREVIOUS_WINNER_RISK_PCT:.4f}` ci-dessous. La lecture utile est la vitesse de deterioration du drawdown et du `prop_score` quand on desserre le cap.",
            (
                f"5. Distribution des tailles pour la meilleure variante: avg `{float(recommended_variant['oos_avg_contracts']):.2f}` | median `{float(recommended_variant['oos_median_contracts']):.2f}` | 1c `{float(recommended_variant['oos_pct_trades_at_1_contract']):.1f}%` | 2c `{float(recommended_variant['oos_pct_trades_at_2_contracts']):.1f}%` | 3+c `{float(recommended_variant['oos_pct_trades_at_3_plus_contracts']):.1f}%`."
                if recommended_variant is not None
                else "5. Distribution des tailles de position: `n/a`."
            ),
            (
                f"6. Trades skippes pour la meilleure variante: `{int(recommended_variant['oos_nb_skipped_trades'])}` soit `{float(recommended_variant['oos_pct_skipped_trades']):.1f}%` des tentatives."
                if recommended_variant is not None
                else "6. Nombre et pourcentage de trades skippes: `n/a`."
            ),
            (
                f"7. Robustesse: `{robustness_scale}`."
                if recommended_cluster is not None
                else "7. Robustesse: locale et trop etroite pour parler d'une vraie zone."
            ),
            (
                f"8. Reference de recherche: `{recommended_variant['campaign_variant_name']}` | mode prop-safe: `{prop_safe_variant['campaign_variant_name']}` | mode plus agressif mais defendable: `{aggressive_variant['campaign_variant_name']}`."
                if recommended_variant is not None and prop_safe_variant is not None and aggressive_variant is not None
                else "8. Reference / prop-safe / agressif: voir les tableaux, aucun trio complet ne ressort proprement."
            ),
            f"9. Verdict final: `{verdict}`.",
            "",
            f"### Slice: risk_pct around {BEST_PREVIOUS_WINNER_RISK_PCT:.4f} with max_contracts={BEST_PREVIOUS_WINNER_MAX_CONTRACTS}",
            "```text",
            _slice_table(summary, max_contracts=BEST_PREVIOUS_WINNER_MAX_CONTRACTS),
            "```",
            "",
            f"### Slice: max_contracts around {BEST_PREVIOUS_WINNER_MAX_CONTRACTS} with risk_pct={BEST_PREVIOUS_WINNER_RISK_PCT:.4f}",
            "```text",
            _slice_table(summary, risk_pct=BEST_PREVIOUS_WINNER_RISK_PCT),
            "```",
        ]
    )

    report_path = output_dir / "final_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    verdict_payload = {
        "final_verdict": verdict,
        "recommended_variant": None if recommended_variant is None else str(recommended_variant["campaign_variant_name"]),
        "punctual_oos_winner": None if point_winner is None else str(point_winner["campaign_variant_name"]),
        "prop_safe_variant": None if prop_safe_variant is None else str(prop_safe_variant["campaign_variant_name"]),
        "aggressive_variant": None if aggressive_variant is None else str(aggressive_variant["campaign_variant_name"]),
        "recommended_cluster_id": None if recommended_cluster is None else int(recommended_cluster["cluster_id"]),
    }
    _json_dump(output_dir / "final_verdict.json", verdict_payload)
    return verdict_payload


def run_campaign(
    *,
    output_root: Path | None = None,
    input_path: Path | None = None,
    initial_capital_usd: float = DEFAULT_INITIAL_CAPITAL_USD,
    risk_pcts: tuple[float, ...] = DEFAULT_RISK_PCTS,
    max_contracts_grid: tuple[int, ...] = DEFAULT_MAX_CONTRACTS,
    base_variant_name: str | None = None,
    reference_v3_dir: Path | None = None,
) -> Path:
    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) if output_root is not None else EXPORTS_DIR / f"{DEFAULT_OUTPUT_PREFIX}{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_alpha_variant, resolved_reference_dir, resolved_reference_name = _resolve_base_alpha_variant(
        base_variant_name=base_variant_name,
        reference_v3_dir=reference_v3_dir,
    )
    dataset_path = Path(input_path) if input_path is not None else None
    raw = load_symbol_data(DEFAULT_SYMBOL, input_paths=None if dataset_path is None else {DEFAULT_SYMBOL: dataset_path})
    bars = resample_rth_1h(raw)
    if bars.empty:
        raise ValueError(f"No RTH 1h bars are available for the {DEFAULT_SYMBOL} refinement campaign.")

    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")
    bars["session_date"] = bars["timestamp"].dt.date
    is_sessions, oos_sessions = split_sessions(bars[["session_date"]].copy())
    phase_lookup = _phase_map(is_sessions, oos_sessions)

    features = prepare_volume_climax_pullback_v2_features(bars)
    signal_df = build_volume_climax_pullback_v2_signal_frame(features, base_alpha_variant)
    oos_signal_df = _subset_by_sessions(signal_df, oos_sessions)

    execution_model, instrument = build_execution_model_for_profile(symbol=DEFAULT_SYMBOL, profile_name="repo_realistic")
    variants = build_refinement_variants(
        initial_capital_usd=float(initial_capital_usd),
        risk_pcts=tuple(risk_pcts),
        max_contracts_grid=tuple(max_contracts_grid),
        include_best_previous_winner_alias=True,
    )

    result_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    trade_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []
    prop_rows: list[dict[str, Any]] = []
    sizing_rows: list[pd.DataFrame] = []

    full_sessions = list(pd.to_datetime(signal_df["session_date"]).dt.date.unique())

    for order_index, campaign_variant in enumerate(variants):
        cache_key = _variant_cache_key(campaign_variant)
        if cache_key not in result_cache:
            result_cache[cache_key] = {
                "full": run_volume_climax_pullback_v2_backtest(
                    signal_df=signal_df,
                    variant=base_alpha_variant,
                    execution_model=execution_model,
                    instrument=instrument,
                    position_sizing=campaign_variant.position_sizing,
                ),
                "oos_only": run_volume_climax_pullback_v2_backtest(
                    signal_df=oos_signal_df,
                    variant=base_alpha_variant,
                    execution_model=execution_model,
                    instrument=instrument,
                    position_sizing=campaign_variant.position_sizing,
                ),
            }
        cached = result_cache[cache_key]

        evaluations: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]] = {}
        for scope_name, result, scoped_signal_df, scoped_sessions in (
            ("full", cached["full"], signal_df, full_sessions),
            ("oos_only", cached["oos_only"], oos_signal_df, oos_sessions),
        ):
            scoped_trades = result.trades.copy()
            scoped_trades["campaign_variant_name"] = campaign_variant.campaign_variant_name
            scoped_trades["variant_role"] = campaign_variant.variant_role
            scoped_trades["scope"] = scope_name
            scoped_trades["alpha_variant_name"] = base_alpha_variant.name
            if not scoped_trades.empty:
                scoped_trades["phase"] = pd.to_datetime(scoped_trades["session_date"]).dt.date.map(phase_lookup).fillna(
                    "oos" if scope_name == "oos_only" else "unknown"
                )

            scoped_decisions = result.sizing_decisions.copy()
            scoped_decisions["campaign_variant_name"] = campaign_variant.campaign_variant_name
            scoped_decisions["variant_role"] = campaign_variant.variant_role
            scoped_decisions["scope"] = scope_name
            scoped_decisions["alpha_variant_name"] = base_alpha_variant.name
            if not scoped_decisions.empty:
                scoped_decisions["phase"] = pd.to_datetime(scoped_decisions["session_date"]).dt.date.map(phase_lookup).fillna(
                    "oos" if scope_name == "oos_only" else "unknown"
                )

            daily_results = _daily_results_from_trades(
                trades=scoped_trades,
                sessions=scoped_sessions,
                initial_capital=float(campaign_variant.initial_capital_usd),
            )
            daily_results["campaign_variant_name"] = campaign_variant.campaign_variant_name
            daily_results["variant_role"] = campaign_variant.variant_role
            daily_results["scope"] = scope_name
            daily_results["alpha_variant_name"] = base_alpha_variant.name
            daily_results["phase"] = pd.to_datetime(daily_results["session_date"]).dt.date.map(phase_lookup).fillna(
                "oos" if scope_name == "oos_only" else "unknown"
            )

            evaluations[scope_name] = (scoped_trades, scoped_decisions, daily_results, scoped_sessions)
            trade_rows.append(scoped_trades)
            daily_rows.append(daily_results)
            sizing_rows.append(scoped_decisions)

        full_summary = _summarize_scope_refinement(
            trades=evaluations["full"][0],
            signal_df=signal_df,
            sessions=evaluations["full"][3],
            daily_results=evaluations["full"][2],
            sizing_decisions=evaluations["full"][1],
            initial_capital=float(campaign_variant.initial_capital_usd),
        )
        oos_summary = _summarize_scope_refinement(
            trades=evaluations["oos_only"][0],
            signal_df=oos_signal_df,
            sessions=evaluations["oos_only"][3],
            daily_results=evaluations["oos_only"][2],
            sizing_decisions=evaluations["oos_only"][1],
            initial_capital=float(campaign_variant.initial_capital_usd),
        )

        for scope_name, scope_summary in (("full", full_summary), ("oos", oos_summary)):
            prop_rows.append(
                {
                    "campaign_variant_name": campaign_variant.campaign_variant_name,
                    "variant_role": campaign_variant.variant_role,
                    "scope": scope_name,
                    "sizing_mode": campaign_variant.sizing_mode,
                    "risk_pct": campaign_variant.risk_pct,
                    "max_contracts": campaign_variant.max_contracts,
                    "skip_trade_if_too_small": campaign_variant.skip_trade_if_too_small,
                    "fixed_contracts": campaign_variant.fixed_contracts,
                    "initial_capital_usd": campaign_variant.initial_capital_usd,
                    "worst_trade_loss_usd": scope_summary["worst_trade_loss_usd"],
                    "worst_day_pnl_usd": scope_summary["worst_day_pnl_usd"],
                    "max_daily_drawdown_usd": scope_summary["max_daily_drawdown_usd"],
                    "nb_days_below_minus_250": scope_summary["nb_days_below_minus_250"],
                    "nb_days_below_minus_500": scope_summary["nb_days_below_minus_500"],
                    "nb_days_below_minus_1000": scope_summary["nb_days_below_minus_1000"],
                    "nb_trades_below_minus_150": scope_summary["nb_trades_below_minus_150"],
                    "nb_trades_below_minus_250": scope_summary["nb_trades_below_minus_250"],
                    "nb_trades_below_minus_500": scope_summary["nb_trades_below_minus_500"],
                    "pass_target_3000_usd_without_breaching_2000_dd": scope_summary["pass_target_3000_usd_without_breaching_2000_dd"],
                    "days_to_3000_usd_if_reached": scope_summary["days_to_3000_usd_if_reached"],
                    "max_trailing_drawdown_observed_usd": scope_summary["max_trailing_drawdown_observed_usd"],
                    "max_static_drawdown_observed_usd": scope_summary["max_static_drawdown_observed_usd"],
                }
            )

        summary_rows.append(
            {
                "order_index": order_index,
                "campaign_variant_name": campaign_variant.campaign_variant_name,
                "variant_role": campaign_variant.variant_role,
                "alpha_variant_name": base_alpha_variant.name,
                "sizing_mode": campaign_variant.sizing_mode,
                "fixed_contracts": campaign_variant.fixed_contracts,
                "risk_pct": campaign_variant.risk_pct,
                "max_contracts": campaign_variant.max_contracts,
                "skip_trade_if_too_small": campaign_variant.skip_trade_if_too_small,
                "initial_capital_usd": campaign_variant.initial_capital_usd,
                **{f"full_{key}": value for key, value in full_summary.items()},
                **{f"oos_{key}": value for key, value in oos_summary.items()},
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["order_index", "campaign_variant_name"]).reset_index(drop=True)
    baseline = summary.loc[summary["campaign_variant_name"] == "fixed_1_contract"].iloc[0]
    for prefix in ("full", "oos"):
        summary[f"{prefix}_prop_score"] = summary.apply(lambda row: compute_prop_score(row, prefix=prefix), axis=1)
        summary[f"{prefix}_net_pnl_delta_vs_baseline_usd"] = pd.to_numeric(summary[f"{prefix}_net_pnl_usd"], errors="coerce") - float(
            baseline[f"{prefix}_net_pnl_usd"]
        )
        summary[f"{prefix}_cagr_delta_vs_baseline_pct"] = pd.to_numeric(summary[f"{prefix}_cagr_pct"], errors="coerce") - float(
            baseline[f"{prefix}_cagr_pct"]
        )
        summary[f"{prefix}_sharpe_delta_vs_baseline"] = pd.to_numeric(summary[f"{prefix}_sharpe"], errors="coerce") - float(
            baseline[f"{prefix}_sharpe"]
        )
        summary[f"{prefix}_max_drawdown_delta_vs_baseline_usd"] = pd.to_numeric(
            summary[f"{prefix}_max_drawdown_usd"], errors="coerce"
        ) - float(baseline[f"{prefix}_max_drawdown_usd"])

    heatmap_metrics, cluster_summary, recommended_cluster_id = _build_heatmap_metrics(summary)
    summary["recommended_cluster_id"] = recommended_cluster_id
    if not cluster_summary.empty:
        summary["is_recommended_cluster"] = summary["campaign_variant_name"].isin(
            heatmap_metrics.loc[heatmap_metrics["is_recommended_cluster"], "campaign_variant_name"].tolist()
        )
    else:
        summary["is_recommended_cluster"] = False

    summary.to_csv(output_dir / "summary_by_variant.csv", index=False)
    oos_only = summary[
        [
            "campaign_variant_name",
            "variant_role",
            "alpha_variant_name",
            "sizing_mode",
            "fixed_contracts",
            "risk_pct",
            "max_contracts",
            "skip_trade_if_too_small",
            "initial_capital_usd",
            *[column for column in summary.columns if column.startswith("oos_")],
        ]
    ].copy()
    oos_only = oos_only.rename(columns={column: column[4:] for column in oos_only.columns if column.startswith("oos_")})
    oos_only.to_csv(output_dir / "summary_oos_only.csv", index=False)

    trade_export = _concat_non_empty_frames(trade_rows)
    daily_export = _concat_non_empty_frames(daily_rows)
    sizing_export = _concat_non_empty_frames(sizing_rows)
    trade_export.to_csv(output_dir / "trades_by_variant.csv", index=False)
    daily_export.to_csv(output_dir / "daily_equity_by_variant.csv", index=False)
    pd.DataFrame(prop_rows).to_csv(output_dir / "prop_constraints_summary.csv", index=False)
    sizing_export.to_csv(output_dir / "sizing_decisions_by_variant.csv", index=False)
    heatmap_metrics.to_csv(output_dir / "heatmap_metrics.csv", index=False)
    cluster_summary.to_csv(output_dir / "robustness_zone_summary.csv", index=False)

    _save_heatmap_png(
        frame=heatmap_metrics,
        value_column="oos_net_pnl_usd",
        title="OOS Net PnL",
        output_path=output_dir / "heatmap_oos_net_pnl.png",
        cmap="RdYlGn",
        value_format=".0f",
    )
    _save_heatmap_png(
        frame=heatmap_metrics,
        value_column="oos_sharpe",
        title="OOS Sharpe",
        output_path=output_dir / "heatmap_oos_sharpe.png",
        cmap="RdYlGn",
        value_format=".2f",
    )
    _save_heatmap_png(
        frame=heatmap_metrics,
        value_column="oos_max_drawdown_usd",
        title="OOS Max Drawdown USD",
        output_path=output_dir / "heatmap_oos_maxdd_usd.png",
        cmap="RdYlGn_r",
        value_format=".0f",
    )
    _save_heatmap_png(
        frame=heatmap_metrics,
        value_column="oos_prop_score",
        title="OOS Prop Score",
        output_path=output_dir / "heatmap_oos_prop_score.png",
        cmap="RdYlGn",
        value_format=".2f",
    )
    _save_cluster_map_png(
        heatmap_metrics=heatmap_metrics,
        output_path=output_dir / "robustness_cluster_map.png",
    )

    verdict = _build_final_report(
        output_dir=output_dir,
        summary=summary,
        heatmap_metrics=heatmap_metrics,
        cluster_summary=cluster_summary,
        base_variant_name=resolved_reference_name or base_alpha_variant.name,
        reference_v3_dir=resolved_reference_dir,
        dataset_path=dataset_path,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
    )

    _json_dump(
        output_dir / "run_metadata.json",
        {
            "generated_at": datetime.now().isoformat(),
            "symbol": DEFAULT_SYMBOL,
            "dataset_path": dataset_path,
            "reference_v3_dir": resolved_reference_dir,
            "resolved_base_alpha_variant_name": resolved_reference_name or base_alpha_variant.name,
            "resolved_base_alpha_variant": asdict(base_alpha_variant),
            "initial_capital_usd": initial_capital_usd,
            "risk_pcts": list(risk_pcts),
            "max_contracts_grid": list(max_contracts_grid),
            "skip_trade_if_too_small": DEFAULT_SKIP_TRADE_IF_TOO_SMALL,
            "session_count_full": int(len(is_sessions) + len(oos_sessions)),
            "session_count_is": int(len(is_sessions)),
            "session_count_oos": int(len(oos_sessions)),
            "variant_count": int(len(summary)),
            "recommended_cluster_id": recommended_cluster_id,
            "best_previous_winner_alias": BEST_PREVIOUS_WINNER_ALIAS,
            "best_previous_winner_grid_name": BEST_PREVIOUS_WINNER_NAME,
            "verdict": verdict,
        },
    )
    return output_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--base-variant-name", type=str, default=None)
    parser.add_argument("--reference-v3-dir", type=Path, default=None)
    return parser


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
