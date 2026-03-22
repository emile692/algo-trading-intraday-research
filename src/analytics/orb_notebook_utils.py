"""Shared helpers for the final ORB client notebooks.

These utilities keep the notebook displays honest by:
- computing returns from the configured initial capital,
- making the full-sample / IS / OOS scope explicit,
- standardizing notebook KPI tables and curve formatting.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd


def to_naive_utc(series: pd.Series) -> pd.Series:
    """Convert timestamps to naive UTC for stable notebook plotting."""
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.tz_convert(None)


def curve_drawdown_pct(equity: pd.Series) -> pd.Series:
    """Return percentage drawdown from an equity series."""
    eq = pd.to_numeric(equity, errors="coerce")
    peak = eq.cummax()
    return (eq / peak - 1.0) * 100.0


def normalize_curve(curve: pd.DataFrame) -> pd.DataFrame:
    """Standardize curve timestamps, ordering, and drawdown fields."""
    if curve.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "drawdown", "drawdown_pct"])

    out = curve.copy()
    out["timestamp"] = to_naive_utc(out["timestamp"])
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if "drawdown_pct" not in out.columns and "equity" in out.columns:
        out["drawdown_pct"] = curve_drawdown_pct(out["equity"])
    return out


def curve_end_equity(curve: pd.DataFrame) -> float | None:
    """Return the last equity value, or None when unavailable."""
    if curve.empty or "equity" not in curve.columns:
        return None
    value = float(pd.to_numeric(curve["equity"], errors="coerce").iloc[-1])
    return value if math.isfinite(value) else None


def curve_total_return_pct(curve: pd.DataFrame, initial_capital: float) -> float:
    """Return total performance from the configured initial capital."""
    end_equity = curve_end_equity(curve)
    if end_equity is None or not math.isfinite(initial_capital) or initial_capital == 0:
        return 0.0
    return float((end_equity / float(initial_capital) - 1.0) * 100.0)


def curve_max_drawdown_pct(curve: pd.DataFrame) -> float:
    """Return the worst drawdown percentage from a normalized curve."""
    if curve.empty or "drawdown_pct" not in curve.columns:
        return 0.0
    value = float(pd.to_numeric(curve["drawdown_pct"], errors="coerce").min())
    return value if math.isfinite(value) else 0.0


def curve_daily_sharpe(curve: pd.DataFrame) -> float:
    """Compute a daily Sharpe ratio from the curve equity series."""
    if curve.empty:
        return 0.0
    daily = curve.set_index("timestamp")["equity"].resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    if len(rets) < 2 or float(rets.std(ddof=0)) == 0.0:
        return 0.0
    return float((rets.mean() / rets.std(ddof=0)) * math.sqrt(252.0))


def curve_annualized_return(curve: pd.DataFrame, initial_capital: float) -> float:
    """Compute CAGR-like annualized return from initial capital to last equity."""
    if curve.empty or len(curve) < 2:
        return 0.0
    end_equity = curve_end_equity(curve)
    if end_equity is None or initial_capital <= 0 or end_equity <= 0:
        return 0.0
    n_days = max((curve["timestamp"].iloc[-1] - curve["timestamp"].iloc[0]).days, 1)
    years = n_days / 365.25
    if years <= 0:
        return 0.0
    return float((end_equity / float(initial_capital)) ** (1.0 / years) - 1.0)


def curve_daily_vol(curve: pd.DataFrame) -> float:
    """Compute annualized daily volatility from the curve equity series."""
    if curve.empty:
        return 0.0
    daily = curve.set_index("timestamp")["equity"].resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    if len(rets) < 2:
        return 0.0
    return float(rets.std(ddof=0) * math.sqrt(252.0))


def format_curve_stats_line(
    name: str,
    sharpe: float,
    ret_pct: float,
    cagr_pct: float,
    vol_pct: float,
    dd_pct: float,
    pf: float | None = None,
    exp: float | None = None,
) -> str:
    """Create a compact HTML-like metrics line for notebook annotations."""
    parts = [f"<b>{name}</b>", f"Sharpe {sharpe:.2f}"]
    if pf is not None:
        parts.append(f"PF {pf:.2f}")
    if exp is not None:
        parts.append(f"Exp {exp:.1f}")
    parts.extend(
        [
            f"Ret {ret_pct:.1f}%",
            f"CAGR {cagr_pct:.1f}%",
            f"Vol {vol_pct:.1f}%",
            f"MaxDD {dd_pct:.1f}%",
        ]
    )
    return " | ".join(parts)


def build_selected_ensemble_kpi_frame(selected_ensemble: Mapping[str, Any]) -> pd.DataFrame:
    """Return a standard KPI table for the selected ensemble row."""
    row = {
        "model": "selected_ensemble",
        "aggregation_rule": selected_ensemble.get("aggregation_rule"),
        "overall_score": float(selected_ensemble.get("overall_composite_score", 0.0)),
        "overall_sharpe": float(selected_ensemble.get("overall_sharpe", 0.0)),
        "overall_profit_factor": float(selected_ensemble.get("overall_profit_factor", 0.0)),
        "overall_expectancy": float(selected_ensemble.get("overall_expectancy", 0.0)),
        "overall_net_pnl": float(selected_ensemble.get("overall_net_pnl", 0.0)),
        "overall_return_over_drawdown": float(selected_ensemble.get("overall_return_over_drawdown", 0.0)),
        "overall_trades": int(selected_ensemble.get("overall_nb_trades", 0)),
        "oos_score": float(selected_ensemble.get("oos_composite_score", 0.0)),
        "oos_sharpe": float(selected_ensemble.get("oos_sharpe", 0.0)),
        "oos_profit_factor": float(selected_ensemble.get("oos_profit_factor", 0.0)),
        "oos_expectancy": float(selected_ensemble.get("oos_expectancy", 0.0)),
        "oos_net_pnl": float(selected_ensemble.get("oos_net_pnl", 0.0)),
        "oos_return_over_drawdown": float(selected_ensemble.get("oos_return_over_drawdown", 0.0)),
        "oos_trades": int(selected_ensemble.get("oos_nb_trades", 0)),
        "oos_pct_days_traded": float(selected_ensemble.get("oos_pct_days_traded", 0.0)),
    }
    return pd.DataFrame([row])


def build_scope_readout_markdown(
    full_curve: pd.DataFrame,
    oos_curve: pd.DataFrame,
    initial_capital: float,
    full_label: str = "Full-sample ensemble curve",
    oos_label: str = "OOS-only curve",
) -> str:
    """Explain curve scope so notebook readers do not confuse OOS and full sample."""
    full_end = curve_end_equity(full_curve)
    oos_end = curve_end_equity(oos_curve)
    full_ret = curve_total_return_pct(full_curve, initial_capital)
    oos_ret = curve_total_return_pct(oos_curve, initial_capital)
    full_dd = curve_max_drawdown_pct(full_curve)
    oos_dd = curve_max_drawdown_pct(oos_curve)

    lines = [
        "### Scope check",
        f"- **{full_label}** includes every selected trade across IS and OOS. Final equity: **{_fmt_money(full_end)} USD** | Return: **{full_ret:.1f}%** | MaxDD: **{full_dd:.1f}%**.",
        f"- **{oos_label}** keeps only the OOS trades. Final equity: **{_fmt_money(oos_end)} USD** | Return: **{oos_ret:.1f}%** | MaxDD: **{oos_dd:.1f}%**.",
        "- Read `overall_*` metrics against the full-sample curve and `oos_*` metrics against the OOS-only curve.",
    ]
    if math.isfinite(full_ret) and math.isfinite(oos_ret) and (full_ret * oos_ret) < 0:
        lines.append(
            "- Full sample and OOS point in opposite directions here; that is possible because the full curve still contains the IS history."
        )
    return "\n".join(lines)


def _fmt_money(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:,.0f}"
