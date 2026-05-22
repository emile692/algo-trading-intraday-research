"""Integrate the fixed pullback sleeve into the existing prop-firm portfolio baseline."""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.topstep_optimization.topstep_simulator import TopstepRules, simulate_account_path

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path("export")
DEFAULT_PULLBACK_PORTFOLIO = "m2k_mgc_equal_weight"
DEFAULT_PULLBACK_M2K_ONLY = "m2k_only"
DEFAULT_PROP_CAPITAL = 50_000.0
DEFAULT_BOOTSTRAP_PATHS = 500
DEFAULT_BOOTSTRAP_BLOCK = 5
DEFAULT_RANDOM_SEED = 7
DEFAULT_PULLBACK_EXPORT = Path("export/volume_climax_pullback_survivor_audit_20260521_091448")
DEFAULT_BASELINE_CANDIDATES = (
    Path("data/exports/mnq_orb_vvix_sizing_modulation_20260328_run/variants/baseline_3state/daily_results.csv"),
    Path("data/exports/mnq_orb_vix_vvix_validation_20260327_run/variants/baseline_fixed_nominal_atr/daily_results.csv"),
)
TRAIN_END = pd.Timestamp("2023-12-31")
OOS_START = pd.Timestamp("2024-01-01")


@dataclass(frozen=True)
class PortfolioSpec:
    portfolio_name: str
    baseline_weight: float
    pullback_weight: float
    pullback_source: str
    scale_to_baseline_risk: bool = False
    flat_day_only: bool = False
    capped_pullback_weight: float | None = None
    deployable: bool = True
    notes: str = ""


def _file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return float(numeric)


def _resolve_baseline_daily_pnl_path(user_path: str | None) -> Path:
    if user_path:
        path = Path(user_path)
        if not path.exists():
            raise FileNotFoundError(f"Baseline daily pnl path not found: {path}")
        return path

    readiness = Path("data/exports/mnq_orb_prop_challenge_readiness_20260328_run/final_verdict.json")
    if readiness.exists():
        payload = json.loads(readiness.read_text(encoding="utf-8"))
        source_run_root = payload.get("source_run_root")
        variant_name = payload.get("challenge_best_variant") or payload.get("recommended_launch_variant")
        if source_run_root and variant_name:
            candidate = Path(source_run_root) / "variants" / str(variant_name) / "daily_results.csv"
            if candidate.exists():
                return candidate

    for candidate in DEFAULT_BASELINE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not infer a baseline daily pnl path. Pass --baseline-daily-pnl-path explicitly.")


def load_baseline_daily_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["session_date"])
    required = {"session_date", "daily_pnl_usd"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Baseline daily results missing columns: {sorted(missing)}")
    if "daily_trade_count" not in frame.columns:
        frame["daily_trade_count"] = np.where(pd.to_numeric(frame["daily_pnl_usd"], errors="coerce").fillna(0.0) != 0.0, 1.0, 0.0)
    out = frame.loc[:, ["session_date", "daily_pnl_usd", "daily_trade_count"]].copy()
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.normalize()
    out["daily_pnl_usd"] = pd.to_numeric(out["daily_pnl_usd"], errors="coerce").fillna(0.0)
    out["daily_trade_count"] = pd.to_numeric(out["daily_trade_count"], errors="coerce").fillna(0.0)
    out["defined"] = True
    out = out.sort_values("session_date").drop_duplicates("session_date", keep="last").reset_index(drop=True)
    return out


def load_pullback_portfolios(export_dir: Path) -> dict[str, pd.DataFrame]:
    daily_path = export_dir / "strict_portfolio_daily_returns.csv"
    summary_path = export_dir / "strict_portfolio_summary.csv"
    if not daily_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            "Pullback survivor audit export must contain strict_portfolio_daily_returns.csv and strict_portfolio_summary.csv."
        )

    daily = pd.read_csv(daily_path, parse_dates=["session_date"])
    summary = pd.read_csv(summary_path)
    allowed = set(summary.loc[summary["selection_basis"].eq("strict_train_only") & summary["deployable"].astype(bool), "portfolio_name"])
    selected = daily.loc[daily["portfolio_name"].isin(allowed)].copy()
    selected["session_date"] = pd.to_datetime(selected["session_date"]).dt.normalize()
    selected["daily_pnl"] = pd.to_numeric(selected["daily_pnl"], errors="coerce").fillna(0.0)
    by_name: dict[str, pd.DataFrame] = {}
    for portfolio_name, part in selected.groupby("portfolio_name", sort=True):
        by_name[str(portfolio_name)] = (
            part.loc[:, ["portfolio_name", "fold_id", "session_date", "daily_pnl", "selection_basis", "deployable"]]
            .sort_values("session_date")
            .reset_index(drop=True)
        )
    return by_name


def expand_pullback_series(events: pd.DataFrame, baseline_sessions: pd.Series) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["session_date", "daily_pnl_usd", "daily_trade_count", "defined", "fold_id"])
    start = pd.to_datetime(events["session_date"]).min().normalize()
    end = pd.to_datetime(events["session_date"]).max().normalize()
    calendar = pd.DataFrame({"session_date": pd.Series(pd.to_datetime(baseline_sessions).dt.normalize().unique())})
    calendar = calendar.loc[calendar["session_date"].between(start, end)].sort_values("session_date").reset_index(drop=True)
    aggregated = (
        events.groupby("session_date", as_index=False)
        .agg(
            daily_pnl_usd=("daily_pnl", "sum"),
            daily_trade_count=("daily_pnl", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) != 0.0).sum())),
            fold_id=("fold_id", "last"),
        )
        .sort_values("session_date")
    )
    merged = calendar.merge(aggregated, on="session_date", how="left")
    merged["daily_pnl_usd"] = pd.to_numeric(merged["daily_pnl_usd"], errors="coerce").fillna(0.0)
    merged["daily_trade_count"] = pd.to_numeric(merged["daily_trade_count"], errors="coerce").fillna(0.0)
    merged["defined"] = True
    return merged


def align_daily_pnl(
    baseline: pd.DataFrame,
    pullback_equal: pd.DataFrame,
    m2k_only: pd.DataFrame,
) -> pd.DataFrame:
    base = baseline.rename(
        columns={
            "daily_pnl_usd": "baseline_daily_pnl_usd",
            "daily_trade_count": "baseline_daily_trade_count",
            "defined": "baseline_defined",
        }
    )
    frame = base.loc[:, ["session_date", "baseline_daily_pnl_usd", "baseline_daily_trade_count", "baseline_defined"]].copy()
    for prefix, source in (("pullback", pullback_equal), ("m2k_only", m2k_only)):
        renamed = source.rename(
            columns={
                "daily_pnl_usd": f"{prefix}_daily_pnl_usd",
                "daily_trade_count": f"{prefix}_daily_trade_count",
                "defined": f"{prefix}_defined",
            }
        )
        keep = ["session_date", f"{prefix}_daily_pnl_usd", f"{prefix}_daily_trade_count", f"{prefix}_defined"]
        if "fold_id" in renamed.columns:
            keep.append("fold_id")
            renamed = renamed.rename(columns={"fold_id": f"{prefix}_fold_id"})
            keep[-1] = f"{prefix}_fold_id"
        frame = frame.merge(renamed.loc[:, keep], on="session_date", how="left")

    for prefix in ("pullback", "m2k_only"):
        frame[f"{prefix}_daily_pnl_usd"] = pd.to_numeric(frame[f"{prefix}_daily_pnl_usd"], errors="coerce").fillna(0.0)
        frame[f"{prefix}_daily_trade_count"] = pd.to_numeric(frame[f"{prefix}_daily_trade_count"], errors="coerce").fillna(0.0)
        frame[f"{prefix}_defined"] = frame[f"{prefix}_defined"].fillna(False)
    frame["overlap_defined"] = frame["baseline_defined"].astype(bool) & frame["pullback_defined"].astype(bool)
    frame["train_mask"] = frame["session_date"] <= TRAIN_END
    frame["oos_mask"] = frame["session_date"] >= OOS_START
    return frame.sort_values("session_date").reset_index(drop=True)


def generate_portfolio_specs() -> list[PortfolioSpec]:
    return [
        PortfolioSpec("baseline_only", 1.0, 0.0, "pullback", notes="Current prop-firm baseline only."),
        PortfolioSpec("pullback_m2k_mgc_only", 0.0, 1.0, "pullback", notes="Fixed pullback sleeve only."),
        PortfolioSpec("pullback_m2k_only", 0.0, 1.0, "m2k_only", notes="Conservative pullback alternative."),
        PortfolioSpec("baseline_plus_pullback_equal_notional", 1.0, 1.0, "pullback", notes="Raw fixed sleeve added 1:1 to baseline."),
        PortfolioSpec(
            "baseline_plus_pullback_scaled_to_baseline_risk",
            1.0,
            1.0,
            "pullback",
            scale_to_baseline_risk=True,
            notes="Raw 1:1 combo scaled with train-only factor to baseline daily volatility.",
        ),
        PortfolioSpec(
            "baseline_plus_pullback_capped_50pct",
            1.0,
            0.5,
            "pullback",
            capped_pullback_weight=0.5,
            notes="Pullback sleeve capped at half the baseline notional.",
        ),
        PortfolioSpec(
            "baseline_plus_pullback_when_baseline_flat",
            1.0,
            1.0,
            "pullback",
            flat_day_only=True,
            notes="Pullback contributes only when baseline has zero trades that day.",
        ),
        PortfolioSpec(
            "baseline_plus_m2k_only_equal_notional",
            1.0,
            1.0,
            "m2k_only",
            notes="Conservative overlay with only the M2K strict survivor.",
        ),
    ]


def _train_only_scale_factor(aligned: pd.DataFrame, baseline_col: str, pullback_col: str, flat_day_only: bool) -> float:
    train = aligned.loc[aligned["overlap_defined"] & aligned["train_mask"]].copy()
    if train.empty:
        return 1.0
    baseline = pd.to_numeric(train[baseline_col], errors="coerce").fillna(0.0)
    pullback = pd.to_numeric(train[pullback_col], errors="coerce").fillna(0.0)
    if flat_day_only:
        active = train["baseline_daily_trade_count"].eq(0.0)
        pullback = pullback.where(active, 0.0)
    raw = baseline + pullback
    baseline_vol = float(baseline.std(ddof=0))
    raw_vol = float(raw.std(ddof=0))
    if baseline_vol <= 0 or raw_vol <= 0:
        return 1.0
    return float(baseline_vol / raw_vol)


def build_combined_portfolio_series(aligned: pd.DataFrame, spec: PortfolioSpec) -> pd.DataFrame:
    source_prefix = "pullback" if spec.pullback_source == "pullback" else "m2k_only"
    pullback_col = f"{source_prefix}_daily_pnl_usd"
    pullback_trade_col = f"{source_prefix}_daily_trade_count"
    base = aligned.copy()
    baseline = pd.to_numeric(base["baseline_daily_pnl_usd"], errors="coerce").fillna(0.0) * float(spec.baseline_weight)
    pullback = pd.to_numeric(base[pullback_col], errors="coerce").fillna(0.0) * float(spec.pullback_weight)
    if spec.flat_day_only:
        pullback = pullback.where(base["baseline_daily_trade_count"].eq(0.0), 0.0)
    if spec.scale_to_baseline_risk:
        factor = _train_only_scale_factor(base, "baseline_daily_pnl_usd", pullback_col, spec.flat_day_only)
        combined = (baseline + pullback) * float(factor)
    else:
        factor = 1.0
        combined = baseline + pullback

    defined = base["baseline_defined"].astype(bool)
    if spec.pullback_weight != 0.0:
        defined &= base[f"{source_prefix}_defined"].astype(bool)

    out = pd.DataFrame(
        {
            "session_date": base["session_date"],
            "portfolio_name": spec.portfolio_name,
            "daily_pnl_usd": combined,
            "baseline_component_usd": baseline,
            "pullback_component_usd": pullback,
            "daily_trade_count": (
                pd.to_numeric(base["baseline_daily_trade_count"], errors="coerce").fillna(0.0)
                + pd.to_numeric(base[pullback_trade_col], errors="coerce").fillna(0.0)
            ),
            "baseline_trade_count": pd.to_numeric(base["baseline_daily_trade_count"], errors="coerce").fillna(0.0),
            "pullback_trade_count": pd.to_numeric(base[pullback_trade_col], errors="coerce").fillna(0.0),
            "pullback_source": spec.pullback_source,
            "deployable": spec.deployable,
            "defined": defined,
            "overlap_defined": base["overlap_defined"],
            "train_mask": base["train_mask"],
            "oos_mask": base["oos_mask"],
            "scaling_factor": float(factor),
            "notes": spec.notes,
        }
    )
    out["equity"] = DEFAULT_PROP_CAPITAL + out["daily_pnl_usd"].cumsum()
    return out


def build_scope_slices(portfolio_daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    overlap = portfolio_daily.loc[portfolio_daily["defined"]].copy()
    train = overlap.loc[overlap["train_mask"]].copy()
    oos = overlap.loc[overlap["oos_mask"]].copy()
    return {
        "defined_full": overlap,
        "defined_train": train,
        "defined_oos": oos,
    }


def _profit_factor(values: pd.Series) -> float:
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity - peak
    return float(drawdown.min()) if not drawdown.empty else 0.0


def compute_portfolio_metrics(series: pd.DataFrame, baseline_reference: pd.Series | None = None) -> dict[str, Any]:
    pnl = pd.to_numeric(series["daily_pnl_usd"], errors="coerce").fillna(0.0)
    if pnl.empty:
        return {
            "days": 0,
            "net_pnl": 0.0,
            "annualized_return_proxy": 0.0,
            "daily_sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "max_daily_loss": 0.0,
            "profit_factor": 0.0,
            "day_win_rate": 0.0,
            "avg_winning_day": 0.0,
            "avg_losing_day": 0.0,
            "monthly_hit_rate": 0.0,
            "active_days": 0,
            "trade_days": 0,
            "avg_trade_count_per_day": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "median_day": 0.0,
            "beta_to_baseline": 0.0,
            "corr_to_baseline": 0.0,
        }
    equity = DEFAULT_PROP_CAPITAL + pnl.cumsum()
    returns = pnl / DEFAULT_PROP_CAPITAL
    std = float(returns.std(ddof=0))
    downside = returns[returns < 0]
    downside_std = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    sharpe = float(returns.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0
    sortino = float(returns.mean() / downside_std * math.sqrt(252.0)) if downside_std > 0 else 0.0
    years = max(len(series) / 252.0, 1.0 / 252.0)
    final_equity = float(equity.iloc[-1]) if len(equity) else DEFAULT_PROP_CAPITAL
    ann = float(((final_equity / DEFAULT_PROP_CAPITAL) ** (1.0 / years) - 1.0) * 100.0) if final_equity > 0 else float("nan")
    monthly = series.assign(month=series["session_date"].dt.to_period("M")).groupby("month", as_index=False)["daily_pnl_usd"].sum()
    corr = 0.0
    beta = 0.0
    if baseline_reference is not None and len(series) == len(baseline_reference):
        base = pd.to_numeric(baseline_reference, errors="coerce").fillna(0.0)
        if pnl.std(ddof=0) > 0 and base.std(ddof=0) > 0:
            corr = float(pnl.corr(base))
            beta = float(np.cov(pnl, base, ddof=0)[0, 1] / np.var(base, ddof=0))
    return {
        "days": int(len(series)),
        "net_pnl": float(pnl.sum()),
        "annualized_return_proxy": ann,
        "daily_sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": _max_drawdown(equity),
        "max_daily_loss": float(pnl.min()),
        "profit_factor": _profit_factor(pnl),
        "day_win_rate": float((pnl > 0).mean()),
        "avg_winning_day": float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0,
        "avg_losing_day": float(pnl[pnl < 0].mean()) if (pnl < 0).any() else 0.0,
        "monthly_hit_rate": float((monthly["daily_pnl_usd"] > 0).mean()) if not monthly.empty else 0.0,
        "active_days": int((pnl != 0).sum()),
        "trade_days": int((pd.to_numeric(series["daily_trade_count"], errors="coerce").fillna(0.0) > 0).sum()),
        "avg_trade_count_per_day": float(pd.to_numeric(series["daily_trade_count"], errors="coerce").fillna(0.0).mean()),
        "gross_profit": float(pnl[pnl > 0].sum()),
        "gross_loss": float(-pnl[pnl < 0].sum()),
        "median_day": float(pnl.median()),
        "corr_to_baseline": corr,
        "beta_to_baseline": beta,
    }


def build_portfolio_summary(portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    baseline_oos = build_scope_slices(portfolios["baseline_only"])["defined_oos"]
    baseline_ref = baseline_oos["daily_pnl_usd"].reset_index(drop=True) if not baseline_oos.empty else None
    rows: list[dict[str, Any]] = []
    for name, frame in portfolios.items():
        for scope_name, scope_frame in build_scope_slices(frame).items():
            reference = baseline_ref if scope_name == "defined_oos" and baseline_ref is not None and len(scope_frame) == len(baseline_ref) else None
            metrics = compute_portfolio_metrics(scope_frame, baseline_reference=reference)
            rows.append({"portfolio_name": name, "scope": scope_name, **metrics, "deployable": bool(frame["deployable"].iloc[0])})
    return pd.DataFrame(rows)


def build_pairwise_correlation(portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scoped = {
        name: build_scope_slices(frame)["defined_oos"].set_index("session_date")["daily_pnl_usd"]
        for name, frame in portfolios.items()
        if not build_scope_slices(frame)["defined_oos"].empty
    }
    names = sorted(scoped)
    for left in names:
        for right in names:
            merged = pd.concat([scoped[left], scoped[right]], axis=1, join="inner")
            merged.columns = ["left", "right"]
            corr = float(merged["left"].corr(merged["right"])) if len(merged) > 1 else 0.0
            rows.append({"left_portfolio": left, "right_portfolio": right, "correlation": corr, "overlap_days": int(len(merged))})
    return pd.DataFrame(rows)


def build_incremental_metrics(portfolio_summary: pd.DataFrame) -> pd.DataFrame:
    oos = portfolio_summary.loc[portfolio_summary["scope"].eq("defined_oos")].copy()
    baseline = oos.loc[oos["portfolio_name"].eq("baseline_only")].iloc[0]
    rows: list[dict[str, Any]] = []
    for _, row in oos.iterrows():
        rows.append(
            {
                "portfolio_name": row["portfolio_name"],
                "incremental_net_pnl_vs_baseline": float(row["net_pnl"] - baseline["net_pnl"]),
                "incremental_sharpe_vs_baseline": float(row["daily_sharpe"] - baseline["daily_sharpe"]),
                "incremental_sortino_vs_baseline": float(row["sortino"] - baseline["sortino"]),
                "incremental_max_drawdown_vs_baseline": float(row["max_drawdown"] - baseline["max_drawdown"]),
                "incremental_monthly_hit_rate_vs_baseline": float(row["monthly_hit_rate"] - baseline["monthly_hit_rate"]),
            }
        )
    return pd.DataFrame(rows)


def build_monthly_yearly(portfolios: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    for name, frame in portfolios.items():
        defined = build_scope_slices(frame)["defined_oos"]
        if defined.empty:
            continue
        monthly = defined.assign(month=defined["session_date"].dt.to_period("M")).groupby("month", as_index=False)["daily_pnl_usd"].sum()
        for _, row in monthly.iterrows():
            monthly_rows.append({"portfolio_name": name, "month": str(row["month"]), "net_pnl": float(row["daily_pnl_usd"])})
        yearly = defined.assign(year=defined["session_date"].dt.year).groupby("year", as_index=False)["daily_pnl_usd"].sum()
        for _, row in yearly.iterrows():
            yearly_rows.append({"portfolio_name": name, "year": int(row["year"]), "net_pnl": float(row["daily_pnl_usd"])})
    return pd.DataFrame(monthly_rows), pd.DataFrame(yearly_rows)


def build_worst_days(portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, frame in portfolios.items():
        defined = build_scope_slices(frame)["defined_oos"]
        if defined.empty:
            continue
        worst = defined.nsmallest(5, "daily_pnl_usd")
        for rank, (_, row) in enumerate(worst.iterrows(), start=1):
            rows.append({"portfolio_name": name, "rank": rank, "session_date": row["session_date"].date().isoformat(), "daily_pnl_usd": float(row["daily_pnl_usd"])})
    return pd.DataFrame(rows)


def build_drawdown_comparison(portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name, frame in portfolios.items():
        defined = build_scope_slices(frame)["defined_oos"].copy()
        if defined.empty:
            continue
        defined["equity"] = DEFAULT_PROP_CAPITAL + defined["daily_pnl_usd"].cumsum()
        peak = defined["equity"].cummax()
        defined["drawdown_usd"] = defined["equity"] - peak
        defined["portfolio_name"] = name
        frames.append(defined.loc[:, ["portfolio_name", "session_date", "daily_pnl_usd", "equity", "drawdown_usd"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_trade_concentration(portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, frame in portfolios.items():
        defined = build_scope_slices(frame)["defined_oos"]
        pnl = pd.to_numeric(defined["daily_pnl_usd"], errors="coerce").fillna(0.0)
        net = float(pnl.sum())
        if pnl.empty:
            continue
        sorted_pos = pnl.sort_values(ascending=False).reset_index(drop=True)
        sorted_neg = pnl.sort_values().reset_index(drop=True)
        for count in (1, 3, 5):
            rows.append(
                {
                    "portfolio_name": name,
                    "bucket": f"top_{count}_days",
                    "pnl_sum": float(sorted_pos.head(count).sum()),
                    "contribution_pct_of_net": float(sorted_pos.head(count).sum() / net) if net != 0 else 0.0,
                }
            )
            rows.append(
                {
                    "portfolio_name": name,
                    "bucket": f"worst_{count}_days",
                    "pnl_sum": float(sorted_neg.head(count).sum()),
                    "contribution_pct_of_net": float(sorted_neg.head(count).sum() / net) if net != 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_portfolios(
    portfolios: dict[str, pd.DataFrame],
    *,
    bootstrap_paths: int,
    block_size: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    rules = TopstepRules()
    for name, frame in portfolios.items():
        oos = build_scope_slices(frame)["defined_oos"].copy()
        if oos.empty:
            continue
        pnl = pd.to_numeric(oos["daily_pnl_usd"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        trades = pd.to_numeric(oos["daily_trade_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if len(pnl) == 0:
            continue
        path_metrics: list[dict[str, Any]] = []
        for path_idx in range(int(bootstrap_paths)):
            blocks: list[int] = []
            while len(blocks) < len(pnl):
                start = int(rng.integers(0, max(len(pnl) - block_size + 1, 1)))
                blocks.extend(range(start, min(start + block_size, len(pnl))))
            indices = np.asarray(blocks[: len(pnl)], dtype=int)
            sample_pnl = pnl[indices]
            sample_trades = trades[indices]
            equity = DEFAULT_PROP_CAPITAL + pd.Series(sample_pnl).cumsum()
            sample_frame = pd.DataFrame(
                {
                    "session_date": pd.RangeIndex(1, len(sample_pnl) + 1),
                    "daily_pnl_usd": sample_pnl,
                    "daily_trade_count": sample_trades,
                }
            )
            _, prop = simulate_account_path(sample_frame, rules=rules)
            path_metrics.append(
                {
                    "net_pnl": float(sample_pnl.sum()),
                    "max_drawdown": _max_drawdown(equity),
                    "daily_loss_breach": bool(np.min(sample_pnl) <= -rules.daily_loss_limit_usd),
                    "prop_pass": bool(prop["pass"]),
                }
            )
        sample = pd.DataFrame(path_metrics)
        rows.append(
            {
                "portfolio_name": name,
                "bootstrap_paths": int(bootstrap_paths),
                "block_size": int(block_size),
                "median_net_pnl": float(sample["net_pnl"].median()),
                "p05_net_pnl": float(sample["net_pnl"].quantile(0.05)),
                "p95_net_pnl": float(sample["net_pnl"].quantile(0.95)),
                "probability_positive": float((sample["net_pnl"] > 0).mean()),
                "probability_drawdown_breach_2k": float((sample["max_drawdown"] <= -2000.0).mean()),
                "probability_daily_loss_breach": float(sample["daily_loss_breach"].mean()),
                "probability_prop_pass": float(sample["prop_pass"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_prop_constraints(portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rules = TopstepRules()
    for name, frame in portfolios.items():
        oos = build_scope_slices(frame)["defined_oos"].copy()
        if oos.empty:
            continue
        sample = pd.DataFrame(
            {
                "session_date": oos["session_date"],
                "daily_pnl_usd": oos["daily_pnl_usd"],
                "daily_trade_count": oos["daily_trade_count"],
            }
        )
        history, result = simulate_account_path(sample, rules=rules)
        rows.append(
            {
                "portfolio_name": name,
                "daily_loss_limit_breaches": int((pd.to_numeric(oos["daily_pnl_usd"], errors="coerce").fillna(0.0) <= -rules.daily_loss_limit_usd).sum()),
                "max_daily_loss": float(pd.to_numeric(oos["daily_pnl_usd"], errors="coerce").fillna(0.0).min()),
                "max_drawdown": float(history["equity"].sub(history["high_watermark"]).min()) if not history.empty else 0.0,
                "historical_prop_status": result["status"],
                "historical_days_to_pass": _safe_float(result["days_to_pass"], float("nan")),
                "historical_days_to_fail": _safe_float(result["days_to_fail"], float("nan")),
                "historical_final_profit_usd": _safe_float(result["final_profit_usd"]),
            }
        )
    return pd.DataFrame(rows)


def build_baseline_comparison(portfolio_summary: pd.DataFrame, correlation: pd.DataFrame, bootstrap: pd.DataFrame, prop_summary: pd.DataFrame) -> pd.DataFrame:
    oos = portfolio_summary.loc[portfolio_summary["scope"].eq("defined_oos")].copy()
    baseline = oos.loc[oos["portfolio_name"].eq("baseline_only")].iloc[0]
    rows: list[dict[str, Any]] = []
    for _, row in oos.iterrows():
        corr_row = correlation.loc[
            correlation["left_portfolio"].eq(row["portfolio_name"]) & correlation["right_portfolio"].eq("baseline_only")
        ]
        bootstrap_row = bootstrap.loc[bootstrap["portfolio_name"].eq(row["portfolio_name"])]
        prop_row = prop_summary.loc[prop_summary["portfolio_name"].eq(row["portfolio_name"])]
        rows.append(
            {
                "portfolio_name": row["portfolio_name"],
                "net_pnl_vs_baseline": float(row["net_pnl"] - baseline["net_pnl"]),
                "sharpe_vs_baseline": float(row["daily_sharpe"] - baseline["daily_sharpe"]),
                "sortino_vs_baseline": float(row["sortino"] - baseline["sortino"]),
                "max_drawdown_vs_baseline": float(row["max_drawdown"] - baseline["max_drawdown"]),
                "correlation_to_baseline": float(corr_row["correlation"].iloc[0]) if not corr_row.empty else 0.0,
                "bootstrap_p05_vs_baseline": (
                    float(bootstrap_row["p05_net_pnl"].iloc[0] - bootstrap.loc[bootstrap["portfolio_name"].eq("baseline_only"), "p05_net_pnl"].iloc[0])
                    if not bootstrap_row.empty and not bootstrap.loc[bootstrap["portfolio_name"].eq("baseline_only")].empty
                    else 0.0
                ),
                "daily_loss_breaches_vs_baseline": (
                    int(prop_row["daily_loss_limit_breaches"].iloc[0] - prop_summary.loc[prop_summary["portfolio_name"].eq("baseline_only"), "daily_loss_limit_breaches"].iloc[0])
                    if not prop_row.empty and not prop_summary.loc[prop_summary["portfolio_name"].eq("baseline_only")].empty
                    else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_verdict(
    *,
    baseline_net_pnl: float,
    net_pnl: float,
    profit_factor: float,
    max_drawdown_delta: float,
    sleeve_correlation_to_baseline: float,
    sharpe_delta: float,
    sortino_delta: float,
    p05_delta: float,
    daily_loss_breach_delta: int,
) -> str:
    net_delta = float(net_pnl - baseline_net_pnl)
    if net_pnl <= 0:
        return "reject"
    if daily_loss_breach_delta > 2 or max_drawdown_delta < -250.0:
        return "reject"
    if net_delta <= 0 and sharpe_delta <= 0 and sortino_delta <= 0 and p05_delta < 0:
        return "reject"
    if (
        net_delta > 0
        and sharpe_delta > 0
        and sortino_delta > 0
        and p05_delta >= 0
        and profit_factor > 1.3
        and sleeve_correlation_to_baseline <= 0.25
        and max_drawdown_delta >= -50.0
        and daily_loss_breach_delta <= 0
    ):
        if net_delta > 250.0 and sharpe_delta > 0.05:
            return "portfolio_candidate"
        return "diversifier_watchlist"
    return "diversifier_watchlist"


def summarize_final_verdict(portfolio_summary: pd.DataFrame, baseline_comparison: pd.DataFrame, yearly_pnl: pd.DataFrame) -> pd.DataFrame:
    oos = portfolio_summary.loc[portfolio_summary["scope"].eq("defined_oos")].copy()
    oos_indexed = oos.set_index("portfolio_name")
    baseline_net = float(oos_indexed.loc["baseline_only", "net_pnl"])
    sleeve_corr = float(oos_indexed.loc["pullback_m2k_mgc_only", "corr_to_baseline"])
    rows: list[dict[str, Any]] = []
    for _, row in oos.iterrows():
        comparison = baseline_comparison.loc[baseline_comparison["portfolio_name"].eq(row["portfolio_name"])].iloc[0]
        verdict = classify_verdict(
            baseline_net_pnl=baseline_net,
            net_pnl=float(row["net_pnl"]),
            profit_factor=float(row["profit_factor"]),
            max_drawdown_delta=float(comparison["max_drawdown_vs_baseline"]),
            sleeve_correlation_to_baseline=sleeve_corr,
            sharpe_delta=float(comparison["sharpe_vs_baseline"]),
            sortino_delta=float(comparison["sortino_vs_baseline"]),
            p05_delta=float(comparison["bootstrap_p05_vs_baseline"]),
            daily_loss_breach_delta=int(comparison["daily_loss_breaches_vs_baseline"]),
        )
        rows.append({"portfolio_name": row["portfolio_name"], "verdict": verdict})
    return pd.DataFrame(rows)


def build_report(
    *,
    output_dir: Path,
    baseline_path: Path,
    pullback_export: Path,
    portfolio_summary: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    prop_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    verdicts: pd.DataFrame,
) -> None:
    oos = portfolio_summary.loc[portfolio_summary["scope"].eq("defined_oos")].copy().set_index("portfolio_name")
    baseline = oos.loc["baseline_only"]
    pullback = oos.loc["pullback_m2k_mgc_only"]
    main_combo = oos.loc["baseline_plus_pullback_equal_notional"]
    verdict = verdicts.set_index("portfolio_name").loc["baseline_plus_pullback_equal_notional", "verdict"]
    combo_corr = float(
        baseline_comparison.loc[baseline_comparison["portfolio_name"].eq("baseline_plus_pullback_equal_notional"), "correlation_to_baseline"].iloc[0]
    )
    sleeve_corr = float(pullback["corr_to_baseline"])
    prop_row = prop_summary.loc[prop_summary["portfolio_name"].eq("baseline_plus_pullback_equal_notional")].iloc[0]
    bootstrap_row = bootstrap_summary.loc[bootstrap_summary["portfolio_name"].eq("baseline_plus_pullback_equal_notional")].iloc[0]
    lines = [
        "# Volume Climax Pullback Portfolio Integration",
        "",
        "## Executive Summary",
        f"- Baseline only OOS net PnL: `{baseline['net_pnl']:.2f} USD`, Sharpe `{baseline['daily_sharpe']:.2f}`, PF `{baseline['profit_factor']:.2f}`.",
        f"- Pullback sleeve only OOS net PnL: `{pullback['net_pnl']:.2f} USD`, Sharpe `{pullback['daily_sharpe']:.2f}`, PF `{pullback['profit_factor']:.2f}`.",
        f"- Main strict integrated portfolio `baseline_plus_pullback_equal_notional`: `{main_combo['net_pnl']:.2f} USD`, Sharpe `{main_combo['daily_sharpe']:.2f}`, PF `{main_combo['profit_factor']:.2f}`.",
        f"- Pullback sleeve vs baseline daily correlation on OOS overlap: `{sleeve_corr:.3f}`.",
        f"- Integrated combo vs baseline daily correlation on OOS overlap: `{combo_corr:.3f}`.",
        f"- Prop constraint impact for the main combo: `{int(prop_row['daily_loss_limit_breaches'])}` daily loss breaches, historical status `{prop_row['historical_prop_status']}`.",
        f"- Bootstrap for the main combo: median `{bootstrap_row['median_net_pnl']:.2f} USD`, p05 `{bootstrap_row['p05_net_pnl']:.2f} USD`, probability positive `{bootstrap_row['probability_positive']:.1%}`.",
        f"- Final verdict: `{verdict}`.",
        "",
        "## Inputs",
        f"- Baseline daily pnl path: `{baseline_path}`",
        f"- Pullback survivor export: `{pullback_export}`",
        "",
        "## Notes",
        "- The pullback sleeve is fixed from the survivor audit strict train-only output.",
        "- No pullback re-optimization, no posthoc asset filtering, and no new gating were used here.",
        "- Risk scaling, when present, is calibrated on the integration train window only (`<= 2023-12-31`).",
    ]
    (output_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_campaign(
    *,
    pullback_export: Path,
    baseline_daily_pnl_path: str | None,
    output_root: Path,
    smoke: bool,
) -> Path:
    baseline_path = _resolve_baseline_daily_pnl_path(baseline_daily_pnl_path)
    baseline = load_baseline_daily_results(baseline_path)
    pullback_portfolios = load_pullback_portfolios(pullback_export)
    required = {DEFAULT_PULLBACK_PORTFOLIO, DEFAULT_PULLBACK_M2K_ONLY}
    missing = sorted(required.difference(pullback_portfolios))
    if missing:
        raise ValueError(f"Pullback export missing required strict portfolios: {missing}")

    pullback_equal = expand_pullback_series(pullback_portfolios[DEFAULT_PULLBACK_PORTFOLIO], baseline["session_date"])
    m2k_only = expand_pullback_series(pullback_portfolios[DEFAULT_PULLBACK_M2K_ONLY], baseline["session_date"])
    aligned = align_daily_pnl(baseline, pullback_equal, m2k_only)

    specs = generate_portfolio_specs()
    portfolios = {spec.portfolio_name: build_combined_portfolio_series(aligned, spec) for spec in specs}

    portfolio_summary = build_portfolio_summary(portfolios)
    portfolio_correlation = build_pairwise_correlation(portfolios)
    incremental_metrics = build_incremental_metrics(portfolio_summary)
    monthly_pnl, yearly_pnl = build_monthly_yearly(portfolios)
    worst_days = build_worst_days(portfolios)
    drawdown_comparison = build_drawdown_comparison(portfolios)
    trade_concentration = build_trade_concentration(portfolios)
    bootstrap_summary = bootstrap_portfolios(
        portfolios,
        bootstrap_paths=64 if smoke else DEFAULT_BOOTSTRAP_PATHS,
        block_size=3 if smoke else DEFAULT_BOOTSTRAP_BLOCK,
        seed=DEFAULT_RANDOM_SEED,
    )
    prop_constraint_summary = summarize_prop_constraints(portfolios)
    baseline_comparison = build_baseline_comparison(portfolio_summary, portfolio_correlation, bootstrap_summary, prop_constraint_summary)
    verdicts = summarize_final_verdict(portfolio_summary, baseline_comparison, yearly_pnl)
    rejected_or_diagnostic = pd.DataFrame(
        [
            {
                "portfolio_name": "baseline_plus_pullback_when_baseline_flat",
                "label": "diagnostic_only_if_timestamp_precision_required",
                "deployable": True,
                "notes": "This rule is safe on days where baseline trade_count == 0, but does not attempt intraday mutual exclusion beyond that.",
            }
        ]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"volume_climax_pullback_portfolio_integration_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    aligned.to_csv(output_dir / "daily_pnl_aligned.csv", index=False)
    portfolio_summary.merge(verdicts, on="portfolio_name", how="left").to_csv(output_dir / "portfolio_summary.csv", index=False)
    portfolio_correlation.to_csv(output_dir / "portfolio_correlation.csv", index=False)
    incremental_metrics.to_csv(output_dir / "incremental_metrics.csv", index=False)
    monthly_pnl.to_csv(output_dir / "monthly_pnl.csv", index=False)
    yearly_pnl.to_csv(output_dir / "yearly_pnl.csv", index=False)
    worst_days.to_csv(output_dir / "worst_days.csv", index=False)
    drawdown_comparison.to_csv(output_dir / "drawdown_comparison.csv", index=False)
    bootstrap_summary.to_csv(output_dir / "bootstrap_summary.csv", index=False)
    prop_constraint_summary.to_csv(output_dir / "prop_constraint_summary.csv", index=False)
    trade_concentration.to_csv(output_dir / "trade_concentration.csv", index=False)
    baseline_comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)
    rejected_or_diagnostic.to_csv(output_dir / "rejected_or_diagnostic_results.csv", index=False)

    run_metadata = {
        "timestamp": timestamp,
        "pullback_export": _file_metadata(pullback_export),
        "baseline_daily_pnl_path": _file_metadata(baseline_path),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "smoke": bool(smoke),
        "train_window_end": str(TRAIN_END.date()),
        "oos_window_start": str(OOS_START.date()),
        "fixed_pullback_portfolio": DEFAULT_PULLBACK_PORTFOLIO,
        "fixed_pullback_m2k_only_portfolio": DEFAULT_PULLBACK_M2K_ONLY,
        "bootstrap_paths": 64 if smoke else DEFAULT_BOOTSTRAP_PATHS,
        "bootstrap_block_size": 3 if smoke else DEFAULT_BOOTSTRAP_BLOCK,
        "random_seed": DEFAULT_RANDOM_SEED,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    build_report(
        output_dir=output_dir,
        baseline_path=baseline_path,
        pullback_export=pullback_export,
        portfolio_summary=portfolio_summary,
        baseline_comparison=baseline_comparison,
        prop_summary=prop_constraint_summary,
        bootstrap_summary=bootstrap_summary,
        verdicts=verdicts,
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrate the fixed pullback sleeve into the prop-firm baseline portfolio.")
    parser.add_argument("--pullback-export", type=Path, default=DEFAULT_PULLBACK_EXPORT)
    parser.add_argument("--baseline-daily-pnl-path", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(levelname)s %(name)s: %(message)s")
    output_dir = run_campaign(
        pullback_export=args.pullback_export,
        baseline_daily_pnl_path=args.baseline_daily_pnl_path,
        output_root=args.output_root,
        smoke=bool(args.smoke),
    )
    print(output_dir)


if __name__ == "__main__":
    main()
