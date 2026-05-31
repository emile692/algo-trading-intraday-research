"""Audit why MNQ ORB 3-state vol-sizing ensemble variants beat the 15/60 baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "export"
OUTPUT_PREFIX = "mnq_orb_3state_vol_sizing_variant_audit"
TARGET_VARIANTS = (
    "single_15_60",
    "median_fast15_slow_60_70_80",
    "median_plateau_compact",
)
BASELINE_VARIANT = "single_15_60"
PERFORMANCE_VARIANT = "median_plateau_compact"


@dataclass(frozen=True)
class AuditPaths:
    variant_export_root: Path
    output_dir: Path


def _make_output_dir(base: Path | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base is None:
        root = DEFAULT_OUTPUT_ROOT / f"{OUTPUT_PREFIX}_{timestamp}"
    else:
        base_path = Path(base)
        if base_path.name.startswith(OUTPUT_PREFIX):
            root = base_path
        else:
            root = base_path / f"{OUTPUT_PREFIX}_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required export file: {path}")
    return pd.read_csv(path, parse_dates=parse_dates or [])


def load_variant_export(variant_export_root: Path) -> dict[str, Any]:
    root = Path(variant_export_root)
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    summary = _load_csv(root / "variant_summary.csv")
    daily = _load_csv(root / "variant_daily_returns.csv", parse_dates=["session_date"])
    trades = _load_csv(root / "variant_trade_summary.csv", parse_dates=["session_date", "entry_time", "exit_time"])
    bucket_contrib = _load_csv(root / "variant_bucket_contribution.csv")
    return {
        "root": root,
        "metadata": metadata,
        "summary": summary,
        "daily": daily,
        "trades": trades,
        "bucket_contrib": bucket_contrib,
    }


def _filter_variants(frame: pd.DataFrame, variants: tuple[str, ...] = TARGET_VARIANTS) -> pd.DataFrame:
    out = frame.loc[frame["variant_name"].isin(set(variants))].copy()
    if out.empty:
        raise ValueError("No target variants were found in the provided export.")
    return out


def prepare_daily_frame(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    daily_frame = daily.copy()
    daily_frame = _filter_variants(daily_frame)
    daily_frame["session_date"] = pd.to_datetime(daily_frame["session_date"], errors="coerce").dt.normalize()
    daily_frame = daily_frame.sort_values(["variant_name", "session_date"]).reset_index(drop=True)
    daily_frame["daily_pnl_usd"] = pd.to_numeric(daily_frame["daily_pnl_usd"], errors="coerce").fillna(0.0)
    daily_frame["daily_return"] = pd.to_numeric(daily_frame["daily_return"], errors="coerce").fillna(0.0)

    trade_cols = trades.copy()
    trade_cols = _filter_variants(trade_cols)
    trade_cols["session_date"] = pd.to_datetime(trade_cols["session_date"], errors="coerce").dt.normalize()
    trade_cols["risk_multiplier"] = pd.to_numeric(trade_cols.get("risk_multiplier"), errors="coerce")
    trade_cols["net_pnl_usd"] = pd.to_numeric(trade_cols["net_pnl_usd"], errors="coerce").fillna(0.0)

    daily_multiplier = (
        trade_cols.groupby(["variant_name", "session_date"], as_index=False)
        .agg(
            risk_multiplier=("risk_multiplier", "median"),
            bucket_label=("bucket_label", "last"),
            traded=("trade_id", "size"),
            daily_trade_pnl=("net_pnl_usd", "sum"),
        )
        .rename(columns={"traded": "trade_count"})
    )

    merged = daily_frame.merge(daily_multiplier, on=["variant_name", "session_date"], how="left")
    merged["risk_multiplier"] = pd.to_numeric(merged["risk_multiplier"], errors="coerce").fillna(0.0)
    merged["bucket_label"] = merged["bucket_label"].fillna("no_trade").astype(str)
    merged["trade_count"] = pd.to_numeric(merged["trade_count"], errors="coerce").fillna(0).astype(int)
    merged["traded"] = merged["trade_count"].gt(0)
    merged["month"] = merged["session_date"].dt.to_period("M").astype(str)
    merged["week_start"] = merged["session_date"] - pd.to_timedelta(merged["session_date"].dt.weekday, unit="D")
    return merged


def build_monthly_returns(daily_frame: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        daily_frame.groupby(["variant_name", "month"], as_index=False)
        .agg(
            month_start=("session_date", "min"),
            month_end=("session_date", "max"),
            monthly_pnl_usd=("daily_pnl_usd", "sum"),
            monthly_return=("daily_return", "sum"),
            positive_days=("daily_pnl_usd", lambda values: int((pd.Series(values) > 0).sum())),
            negative_days=("daily_pnl_usd", lambda values: int((pd.Series(values) < 0).sum())),
            trading_days=("session_date", "size"),
            traded_days=("traded", lambda values: int(pd.Series(values).sum())),
        )
    )
    monthly["positive_month"] = monthly["monthly_pnl_usd"].gt(0.0)
    return monthly.sort_values(["variant_name", "month"]).reset_index(drop=True)


def _rolling_window_drawdown(daily_pnl: pd.Series, window: int) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    pnl = pd.to_numeric(daily_pnl, errors="coerce").fillna(0.0).reset_index(drop=True)
    if pnl.empty:
        return 0.0, None, None
    rolling_values: list[tuple[float, int, int]] = []
    for end_idx in range(len(pnl)):
        start_idx = max(0, end_idx - window + 1)
        window_pnl = pnl.iloc[start_idx : end_idx + 1]
        cumulative = window_pnl.cumsum()
        drawdown = cumulative - cumulative.cummax()
        rolling_values.append((float(drawdown.min()), start_idx, end_idx))
    worst = min(rolling_values, key=lambda item: item[0])
    return worst


def build_worst_periods(daily_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant_name, variant_daily in daily_frame.groupby("variant_name", sort=True):
        ordered = variant_daily.sort_values("session_date").reset_index(drop=True)

        worst_day = ordered.loc[ordered["daily_pnl_usd"].idxmin()]
        rows.append(
            {
                "variant_name": variant_name,
                "period_type": "day",
                "period_label": str(pd.Timestamp(worst_day["session_date"]).date()),
                "period_start": pd.Timestamp(worst_day["session_date"]).date(),
                "period_end": pd.Timestamp(worst_day["session_date"]).date(),
                "period_value": float(worst_day["daily_pnl_usd"]),
            }
        )

        weekly = (
            ordered.groupby("week_start", as_index=False)
            .agg(
                period_start=("session_date", "min"),
                period_end=("session_date", "max"),
                period_value=("daily_pnl_usd", "sum"),
            )
            .sort_values("period_value", ascending=True)
            .reset_index(drop=True)
        )
        if not weekly.empty:
            row = weekly.iloc[0]
            rows.append(
                {
                    "variant_name": variant_name,
                    "period_type": "week",
                    "period_label": str(pd.Timestamp(row["period_start"]).date()),
                    "period_start": pd.Timestamp(row["period_start"]).date(),
                    "period_end": pd.Timestamp(row["period_end"]).date(),
                    "period_value": float(row["period_value"]),
                }
            )

        monthly = (
            ordered.groupby("month", as_index=False)
            .agg(
                period_start=("session_date", "min"),
                period_end=("session_date", "max"),
                period_value=("daily_pnl_usd", "sum"),
            )
            .sort_values("period_value", ascending=True)
            .reset_index(drop=True)
        )
        if not monthly.empty:
            row = monthly.iloc[0]
            rows.append(
                {
                    "variant_name": variant_name,
                    "period_type": "month",
                    "period_label": str(row["month"]),
                    "period_start": pd.Timestamp(row["period_start"]).date(),
                    "period_end": pd.Timestamp(row["period_end"]).date(),
                    "period_value": float(row["period_value"]),
                }
            )

        for window in (20, 60):
            worst_dd, start_idx, end_idx = _rolling_window_drawdown(ordered["daily_pnl_usd"], window=window)
            if start_idx is None or end_idx is None:
                continue
            rows.append(
                {
                    "variant_name": variant_name,
                    "period_type": f"rolling_{window}d_maxdd",
                    "period_label": f"{window}d",
                    "period_start": pd.Timestamp(ordered.iloc[start_idx]["session_date"]).date(),
                    "period_end": pd.Timestamp(ordered.iloc[end_idx]["session_date"]).date(),
                    "period_value": float(worst_dd),
                }
            )
    return pd.DataFrame(rows)


def build_bucket_distribution(daily_frame: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    target_buckets = ("low", "mid", "high")
    rows: list[dict[str, Any]] = []
    filtered_trades = _filter_variants(trades)
    for variant_name, variant_daily in daily_frame.groupby("variant_name", sort=True):
        active_days = variant_daily.loc[variant_daily["traded"]].copy()
        day_total = max(len(active_days), 1)
        variant_trades = filtered_trades.loc[filtered_trades["variant_name"] == variant_name].copy()
        trade_total = max(len(variant_trades), 1)
        for bucket in target_buckets:
            day_count = int(active_days["bucket_label"].eq(bucket).sum())
            trade_count = int(variant_trades["bucket_label"].eq(bucket).sum())
            rows.append(
                {
                    "variant_name": variant_name,
                    "bucket_label": bucket,
                    "day_count": day_count,
                    "pct_days": float(day_count / day_total),
                    "trade_count": trade_count,
                    "pct_trades": float(trade_count / trade_total),
                }
            )
    return pd.DataFrame(rows)


def build_multiplier_transitions(daily_frame: pd.DataFrame, baseline_variant: str = BASELINE_VARIANT) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline = daily_frame.loc[daily_frame["variant_name"] == baseline_variant, ["session_date", "risk_multiplier"]].copy()
    baseline = baseline.rename(columns={"risk_multiplier": "baseline_multiplier"})
    for variant_name, variant_daily in daily_frame.groupby("variant_name", sort=True):
        active = variant_daily.loc[variant_daily["traded"]].sort_values("session_date").reset_index(drop=True)
        if active.empty:
            continue
        transition_pairs = active["bucket_label"].astype(str).shift(1).fillna("start") + "->" + active["bucket_label"].astype(str)
        counts = transition_pairs.value_counts().sort_index()
        switch_count = int(active["bucket_label"].ne(active["bucket_label"].shift(1)).sum() - 1) if len(active) > 1 else 0
        switch_rate = float(switch_count / max(len(active) - 1, 1))

        merged = variant_daily[["session_date", "risk_multiplier"]].merge(baseline, on="session_date", how="inner")
        corr = float(
            pd.to_numeric(merged["risk_multiplier"], errors="coerce").corr(
                pd.to_numeric(merged["baseline_multiplier"], errors="coerce")
            )
        ) if len(merged) >= 2 else np.nan

        avg_multiplier = float(active["risk_multiplier"].mean())
        median_multiplier = float(active["risk_multiplier"].median())

        for transition_name, count in counts.items():
            from_bucket, to_bucket = transition_name.split("->", 1)
            rows.append(
                {
                    "variant_name": variant_name,
                    "transition_from": from_bucket,
                    "transition_to": to_bucket,
                    "transition_count": int(count),
                    "transition_pct": float(count / max(counts.sum(), 1)),
                    "bucket_switches": switch_count,
                    "bucket_switch_rate": switch_rate,
                    "avg_multiplier": avg_multiplier,
                    "median_multiplier": median_multiplier,
                    "multiplier_corr_vs_single_15_60": corr,
                }
            )
    return pd.DataFrame(rows)


def build_daily_diff_frame(daily_frame: pd.DataFrame, baseline_variant: str = BASELINE_VARIANT) -> pd.DataFrame:
    baseline = daily_frame.loc[
        daily_frame["variant_name"] == baseline_variant,
        ["session_date", "daily_pnl_usd", "daily_return", "risk_multiplier", "bucket_label"],
    ].copy()
    baseline = baseline.rename(
        columns={
            "daily_pnl_usd": "baseline_daily_pnl_usd",
            "daily_return": "baseline_daily_return",
            "risk_multiplier": "baseline_risk_multiplier",
            "bucket_label": "baseline_bucket_label",
        }
    )
    rows: list[pd.DataFrame] = []
    for variant_name in TARGET_VARIANTS:
        if variant_name == baseline_variant:
            continue
        variant = daily_frame.loc[
            daily_frame["variant_name"] == variant_name,
            ["session_date", "daily_pnl_usd", "daily_return", "risk_multiplier", "bucket_label"],
        ].copy()
        variant = variant.rename(
            columns={
                "daily_pnl_usd": "variant_daily_pnl_usd",
                "daily_return": "variant_daily_return",
                "risk_multiplier": "variant_risk_multiplier",
                "bucket_label": "variant_bucket_label",
            }
        )
        merged = baseline.merge(variant, on="session_date", how="inner")
        merged["variant_name"] = variant_name
        merged["daily_pnl_diff"] = pd.to_numeric(merged["variant_daily_pnl_usd"], errors="coerce") - pd.to_numeric(
            merged["baseline_daily_pnl_usd"], errors="coerce"
        )
        merged["daily_return_diff"] = pd.to_numeric(merged["variant_daily_return"], errors="coerce") - pd.to_numeric(
            merged["baseline_daily_return"], errors="coerce"
        )
        rows.append(merged)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out["abs_daily_pnl_diff"] = pd.to_numeric(out["daily_pnl_diff"], errors="coerce").abs()
    return out.sort_values(["variant_name", "session_date"]).reset_index(drop=True)


def _top_share(values: pd.Series, top_n: int) -> float:
    clean = pd.to_numeric(values, errors="coerce").fillna(0.0)
    total = float(clean.sum())
    if total <= 0:
        return 0.0
    return float(clean.nlargest(top_n).sum() / total)


def evaluate_diff_profile(diff_frame: pd.DataFrame, variant_name: str) -> dict[str, Any]:
    variant_diff = diff_frame.loc[diff_frame["variant_name"] == variant_name].copy()
    if variant_diff.empty:
        return {
            "variant_name": variant_name,
            "excess_pnl": 0.0,
            "positive_diff_day_share": 0.0,
            "positive_diff_month_share": 0.0,
            "top_3_positive_share": 0.0,
            "top_10_positive_share": 0.0,
            "excess_pnl_after_top_5": 0.0,
            "broad_based_improvement": False,
            "outlier_driven": True,
        }

    pnl_diff = pd.to_numeric(variant_diff["daily_pnl_diff"], errors="coerce").fillna(0.0)
    non_zero = pnl_diff[pnl_diff.ne(0.0)]
    positive_share = float((non_zero > 0).mean()) if not non_zero.empty else 0.0
    monthly_diff = (
        variant_diff.assign(month=variant_diff["session_date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)["daily_pnl_diff"]
        .sum()
    )
    positive_month_share = float((pd.to_numeric(monthly_diff["daily_pnl_diff"], errors="coerce") > 0).mean()) if not monthly_diff.empty else 0.0
    positive_pnl = pnl_diff[pnl_diff > 0]
    excess_pnl = float(pnl_diff.sum())
    top_3_share = _top_share(positive_pnl, top_n=3)
    top_10_share = _top_share(positive_pnl, top_n=10)
    top_5_sum = float(positive_pnl.nlargest(5).sum()) if not positive_pnl.empty else 0.0
    excess_after_top_5 = float(excess_pnl - top_5_sum)
    broad_based = bool(
        excess_pnl > 0
        and positive_share >= 0.50
        and positive_month_share >= 0.50
        and top_10_share <= 0.70
        and excess_after_top_5 > 0
    )
    outlier_driven = bool(
        not broad_based
        and (
            top_10_share > 0.70
            or excess_after_top_5 <= 0.0
            or positive_month_share < 0.35
        )
    )
    return {
        "variant_name": variant_name,
        "excess_pnl": excess_pnl,
        "positive_diff_day_share": positive_share,
        "positive_diff_month_share": positive_month_share,
        "top_3_positive_share": top_3_share,
        "top_10_positive_share": top_10_share,
        "excess_pnl_after_top_5": excess_after_top_5,
        "broad_based_improvement": broad_based,
        "outlier_driven": outlier_driven,
    }


def build_audit_summary(
    smoke_summary: pd.DataFrame,
    daily_frame: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    worst_periods: pd.DataFrame,
    bucket_distribution: pd.DataFrame,
    transitions: pd.DataFrame,
    diff_frame: pd.DataFrame,
) -> pd.DataFrame:
    base_summary = _filter_variants(smoke_summary).copy()
    base_summary = base_summary.set_index("variant_name")
    rows: list[dict[str, Any]] = []

    transition_summary = (
        transitions.groupby("variant_name", as_index=False)
        .agg(
            bucket_switches=("bucket_switches", "max"),
            bucket_switch_rate=("bucket_switch_rate", "max"),
            avg_multiplier=("avg_multiplier", "max"),
            median_multiplier=("median_multiplier", "max"),
            multiplier_corr_vs_single_15_60=("multiplier_corr_vs_single_15_60", "max"),
        )
        if not transitions.empty
        else pd.DataFrame(columns=["variant_name"])
    )
    transition_lookup = transition_summary.set_index("variant_name") if not transition_summary.empty else pd.DataFrame()

    for variant_name in TARGET_VARIANTS:
        if variant_name not in base_summary.index:
            continue
        summary_row = base_summary.loc[variant_name]
        monthly = monthly_returns.loc[monthly_returns["variant_name"] == variant_name].copy()
        worst = worst_periods.loc[worst_periods["variant_name"] == variant_name].copy()
        buckets = bucket_distribution.loc[bucket_distribution["variant_name"] == variant_name].copy()
        diff_profile = evaluate_diff_profile(diff_frame, variant_name)

        bucket_map = buckets.set_index("bucket_label") if not buckets.empty else pd.DataFrame()
        trans_row = transition_lookup.loc[variant_name] if variant_name in getattr(transition_lookup, "index", []) else pd.Series(dtype=object)

        rows.append(
            {
                "variant_name": variant_name,
                "net_pnl": float(summary_row["net_pnl"]),
                "sharpe": float(summary_row["sharpe"]),
                "sortino": float(summary_row["sortino"]),
                "max_drawdown": float(summary_row["max_drawdown"]),
                "monthly_mean_return": float(pd.to_numeric(monthly["monthly_return"], errors="coerce").mean()) if not monthly.empty else 0.0,
                "monthly_median_return": float(pd.to_numeric(monthly["monthly_return"], errors="coerce").median()) if not monthly.empty else 0.0,
                "monthly_hit_rate": float(pd.to_numeric(monthly["monthly_pnl_usd"], errors="coerce").gt(0.0).mean()) if not monthly.empty else 0.0,
                "worst_month": float(
                    pd.to_numeric(worst.loc[worst["period_type"] == "month", "period_value"], errors="coerce").min()
                ) if not worst.empty else 0.0,
                "worst_week": float(
                    pd.to_numeric(worst.loc[worst["period_type"] == "week", "period_value"], errors="coerce").min()
                ) if not worst.empty else 0.0,
                "worst_day": float(
                    pd.to_numeric(worst.loc[worst["period_type"] == "day", "period_value"], errors="coerce").min()
                ) if not worst.empty else 0.0,
                "rolling_20d_max_drawdown": float(
                    pd.to_numeric(worst.loc[worst["period_type"] == "rolling_20d_maxdd", "period_value"], errors="coerce").min()
                ) if not worst.empty else 0.0,
                "rolling_60d_max_drawdown": float(
                    pd.to_numeric(worst.loc[worst["period_type"] == "rolling_60d_maxdd", "period_value"], errors="coerce").min()
                ) if not worst.empty else 0.0,
                "avg_multiplier": float(trans_row.get("avg_multiplier", np.nan)),
                "median_multiplier": float(trans_row.get("median_multiplier", np.nan)),
                "pct_days_low": float(bucket_map.loc["low", "pct_days"]) if "low" in getattr(bucket_map, "index", []) else 0.0,
                "pct_days_mid": float(bucket_map.loc["mid", "pct_days"]) if "mid" in getattr(bucket_map, "index", []) else 0.0,
                "pct_days_high": float(bucket_map.loc["high", "pct_days"]) if "high" in getattr(bucket_map, "index", []) else 0.0,
                "pct_trades_low": float(bucket_map.loc["low", "pct_trades"]) if "low" in getattr(bucket_map, "index", []) else 0.0,
                "pct_trades_mid": float(bucket_map.loc["mid", "pct_trades"]) if "mid" in getattr(bucket_map, "index", []) else 0.0,
                "pct_trades_high": float(bucket_map.loc["high", "pct_trades"]) if "high" in getattr(bucket_map, "index", []) else 0.0,
                "bucket_switches": int(trans_row.get("bucket_switches", 0)) if not trans_row.empty else 0,
                "bucket_switch_rate": float(trans_row.get("bucket_switch_rate", 0.0)) if not trans_row.empty else 0.0,
                "multiplier_corr_vs_single_15_60": float(trans_row.get("multiplier_corr_vs_single_15_60", np.nan)) if not trans_row.empty else np.nan,
                "delta_sharpe_vs_single_15_60": float(summary_row.get("delta_sharpe_vs_single_15_60", 0.0)),
                "delta_net_pnl_vs_single_15_60": float(summary_row.get("delta_net_pnl_vs_single_15_60", 0.0)),
                "delta_maxdd_vs_single_15_60": float(summary_row.get("delta_maxdd_vs_single_15_60", 0.0)),
                "positive_diff_day_share": float(diff_profile["positive_diff_day_share"]),
                "positive_diff_month_share": float(diff_profile["positive_diff_month_share"]),
                "top_3_positive_share": float(diff_profile["top_3_positive_share"]),
                "top_10_positive_share": float(diff_profile["top_10_positive_share"]),
                "excess_pnl_after_top_5": float(diff_profile["excess_pnl_after_top_5"]),
                "broad_based_improvement": bool(diff_profile["broad_based_improvement"]),
                "outlier_driven": bool(diff_profile["outlier_driven"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["sharpe", "net_pnl"], ascending=[False, False]).reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.loc[:, columns].iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, (bool, np.bool_)):
                values.append("yes" if bool(value) else "no")
            elif pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _recommendation_text(summary: pd.DataFrame) -> tuple[str, str]:
    baseline = summary.loc[summary["variant_name"] == BASELINE_VARIANT].iloc[0]
    candidate = summary.loc[summary["variant_name"] == PERFORMANCE_VARIANT].iloc[0]
    dd_worse = abs(float(candidate["max_drawdown"])) > abs(float(baseline["max_drawdown"]))
    broad_based = bool(candidate["broad_based_improvement"])

    if broad_based and not dd_worse:
        return (
            "Recommend median_plateau_compact for production.",
            "The improvement is broad-based and the drawdown profile is not worse than the baseline.",
        )
    if not broad_based and dd_worse:
        return (
            "Stay with single_15_60.",
            "The gain looks outlier-driven and comes with a worse max drawdown profile.",
        )
    if broad_based and dd_worse:
        return (
            "Use median_plateau_compact as the performance version and single_15_60 as the conservative version.",
            "The improvement is broad-based, but the drawdown profile remains worse than the baseline.",
        )
    return (
        "Stay with single_15_60 until a cleaner ensemble edge is confirmed.",
        "The ensemble edge is not clean enough to offset the additional complexity.",
    )


def _write_report(
    output_dir: Path,
    audit_summary: pd.DataFrame,
    diff_frame: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    variant_export_root: Path,
) -> tuple[Path, str]:
    baseline = audit_summary.loc[audit_summary["variant_name"] == BASELINE_VARIANT].iloc[0]
    candidate = audit_summary.loc[audit_summary["variant_name"] == PERFORMANCE_VARIANT].iloc[0]
    verdict, rationale = _recommendation_text(audit_summary)

    candidate_diff = diff_frame.loc[diff_frame["variant_name"] == PERFORMANCE_VARIANT].copy()
    top_positive = candidate_diff.sort_values("daily_pnl_diff", ascending=False).head(10).copy()
    top_negative = candidate_diff.sort_values("daily_pnl_diff", ascending=True).head(10).copy()
    monthly_ranked = monthly_returns.loc[monthly_returns["variant_name"].isin(TARGET_VARIANTS)].copy()

    lines = [
        "# MNQ ORB 3-State Variant Audit",
        "",
        "## Objective",
        "",
        "Audit why `median_plateau_compact` beats `single_15_60` in the existing smoke run,",
        "without relaunching a broad campaign, and decide whether the edge looks broad-based or outlier-driven.",
        "",
        f"- Source run: `{variant_export_root}`",
        "",
        "## Summary",
        "",
        _markdown_table(
            audit_summary,
            [
                "variant_name",
                "sharpe",
                "net_pnl",
                "max_drawdown",
                "monthly_hit_rate",
                "rolling_20d_max_drawdown",
                "rolling_60d_max_drawdown",
                "avg_multiplier",
                "bucket_switches",
            ],
        ),
        "",
        "## Broad-Based Or Outlier-Driven",
        "",
        f"- median_plateau_compact broad-based improvement: `{'yes' if bool(candidate['broad_based_improvement']) else 'no'}`",
        f"- Positive diff day share vs single_15_60: `{float(candidate['positive_diff_day_share']):.3f}`",
        f"- Positive diff month share vs single_15_60: `{float(candidate['positive_diff_month_share']):.3f}`",
        f"- Top 3 positive contribution share: `{float(candidate['top_3_positive_share']):.3f}`",
        f"- Top 10 positive contribution share: `{float(candidate['top_10_positive_share']):.3f}`",
        f"- Excess PnL after removing top 5 positive diff days: `{float(candidate['excess_pnl_after_top_5']):.1f}`",
        "",
        "## Baseline Vs Ensemble",
        "",
        f"- single_15_60: Sharpe `{float(baseline['sharpe']):.3f}`, net PnL `{float(baseline['net_pnl']):.1f}`, maxDD `{float(baseline['max_drawdown']):.1f}`",
        f"- median_plateau_compact: Sharpe `{float(candidate['sharpe']):.3f}`, net PnL `{float(candidate['net_pnl']):.1f}`, maxDD `{float(candidate['max_drawdown']):.1f}`",
        f"- Verdict: {verdict}",
        f"- Rationale: {rationale}",
        "",
        "## Top Positive Contribution Days",
        "",
        _markdown_table(
            top_positive,
            ["session_date", "daily_pnl_diff", "baseline_daily_pnl_usd", "variant_daily_pnl_usd", "baseline_bucket_label", "variant_bucket_label"],
        ),
        "",
        "## Top Negative Contribution Days",
        "",
        _markdown_table(
            top_negative,
            ["session_date", "daily_pnl_diff", "baseline_daily_pnl_usd", "variant_daily_pnl_usd", "baseline_bucket_label", "variant_bucket_label"],
        ),
        "",
        "## Monthly Return Snapshot",
        "",
        _markdown_table(
            monthly_ranked.head(18),
            ["variant_name", "month", "monthly_pnl_usd", "monthly_return", "positive_month"],
        ),
    ]

    report_path = output_dir / "final_audit_report.md"
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path, verdict


def run_audit(variant_export_root: Path, output_root: Path | None = None) -> dict[str, Any]:
    bundle = load_variant_export(variant_export_root)
    output_dir = _make_output_dir(output_root)

    summary = _filter_variants(bundle["summary"])
    daily_frame = prepare_daily_frame(bundle["daily"], bundle["trades"])
    monthly_returns = build_monthly_returns(daily_frame)
    worst_periods = build_worst_periods(daily_frame)
    bucket_distribution = build_bucket_distribution(daily_frame, bundle["trades"])
    transitions = build_multiplier_transitions(daily_frame)
    diff_frame = build_daily_diff_frame(daily_frame)
    audit_summary = build_audit_summary(
        smoke_summary=summary,
        daily_frame=daily_frame,
        monthly_returns=monthly_returns,
        worst_periods=worst_periods,
        bucket_distribution=bucket_distribution,
        transitions=transitions,
        diff_frame=diff_frame,
    )

    summary_path = output_dir / "variant_audit_summary.csv"
    monthly_path = output_dir / "monthly_returns_by_variant.csv"
    worst_path = output_dir / "worst_periods_by_variant.csv"
    bucket_path = output_dir / "bucket_distribution_by_variant.csv"
    transition_path = output_dir / "multiplier_transition_by_variant.csv"
    diff_path = output_dir / "baseline_vs_ensemble_daily_diff.csv"
    metadata_path = output_dir / "run_metadata.json"

    audit_summary.to_csv(summary_path, index=False)
    monthly_returns.to_csv(monthly_path, index=False)
    worst_periods.to_csv(worst_path, index=False)
    bucket_distribution.to_csv(bucket_path, index=False)
    transitions.to_csv(transition_path, index=False)
    diff_frame.to_csv(diff_path, index=False)
    report_path, verdict = _write_report(
        output_dir=output_dir,
        audit_summary=audit_summary,
        diff_frame=diff_frame,
        monthly_returns=monthly_returns,
        variant_export_root=Path(variant_export_root),
    )

    metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "variant_export_root": str(Path(variant_export_root)),
        "target_variants": list(TARGET_VARIANTS),
        "baseline_variant": BASELINE_VARIANT,
        "performance_variant": PERFORMANCE_VARIANT,
        "final_recommendation": verdict,
    }
    (metadata_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "output_dir": output_dir,
        "summary_path": summary_path,
        "monthly_path": monthly_path,
        "worst_path": worst_path,
        "bucket_path": bucket_path,
        "transition_path": transition_path,
        "diff_path": diff_path,
        "report_path": report_path,
        "metadata_path": metadata_path,
        "audit_summary": audit_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the MNQ ORB 3-state vol-sizing smoke export.")
    parser.add_argument("--variant-export", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    artifacts = run_audit(args.variant_export, output_root=args.output_root)
    print(f"output_dir: {artifacts['output_dir']}")
    print(f"summary: {artifacts['summary_path']}")
    print(f"report: {artifacts['report_path']}")


if __name__ == "__main__":
    main()
