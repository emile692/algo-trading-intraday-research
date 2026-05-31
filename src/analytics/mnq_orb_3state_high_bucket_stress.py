"""Minimal OOS stress-test for the MNQ ORB 3-state high-vol bucket."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analytics.mnq_orb_prop_survivability_campaign import _rebuild_daily_results_from_trades
from src.analytics.mnq_orb_regime_filter_sizing_campaign import _scale_nominal_trades_by_multiplier


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORTS_ROOT = REPO_ROOT / "data" / "exports"
OUTPUT_ROOT = REPO_ROOT / "export"
DEFAULT_EXPORT_GLOB = "mnq_orb_regime_filter_sizing_*"
VARIANT_NAME = "sizing_3state_realized_vol_ratio_15_60"
BASELINE_NAME = "nominal"
HIGH_BUCKET_MULTIPLIERS = [0.75, 0.50, 0.25, 0.00]
LOW_MULTIPLIER = 0.50
MID_MULTIPLIER = 1.00


@dataclass(frozen=True)
class VariantSpec:
    name: str
    high_multiplier: float


def _latest_export_root() -> Path:
    candidates = [path for path in EXPORTS_ROOT.glob(DEFAULT_EXPORT_GLOB) if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No export folder found for {DEFAULT_EXPORT_GLOB!r} under {EXPORTS_ROOT}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _make_output_dir(base: Path | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = base or (OUTPUT_ROOT / f"mnq_orb_3state_high_bucket_stress_{timestamp}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_inputs(export_root: Path) -> dict[str, object]:
    metadata = json.loads((export_root / "run_metadata.json").read_text(encoding="utf-8"))
    summary_variants = pd.read_csv(export_root / "summary_variants.csv")
    nominal_trades = pd.read_csv(
        export_root / "variants" / BASELINE_NAME / "trades.csv",
        parse_dates=["session_date", "entry_time", "exit_time"],
    )
    variant_controls = pd.read_csv(
        export_root / "variants" / VARIANT_NAME / "controls.csv",
        parse_dates=["session_date"],
    )
    nominal_daily = pd.read_csv(
        export_root / "variants" / BASELINE_NAME / "daily_results.csv",
        parse_dates=["session_date"],
    )
    mappings = pd.read_csv(export_root / "regime_state_mappings.csv")

    baseline_cfg = metadata["spec"]["baseline"]
    all_sessions = pd.to_datetime(nominal_daily["session_date"], errors="coerce").dt.date.tolist()
    oos_sessions = (
        pd.to_datetime(variant_controls.loc[variant_controls["phase"] == "oos", "session_date"], errors="coerce")
        .dt.date.dropna().tolist()
    )
    bucket_map = (
        mappings.loc[
            (mappings["variant_name"] == VARIANT_NAME)
            & (mappings["feature_name"] == "realized_vol_ratio_15_60"),
            ["bucket_label", "lower_bound", "upper_bound", "risk_multiplier", "is_composite_score"],
        ]
        .drop_duplicates()
        .sort_values("bucket_label")
        .reset_index(drop=True)
    )
    return {
        "metadata": metadata,
        "summary_variants": summary_variants,
        "nominal_trades": nominal_trades,
        "variant_controls": variant_controls,
        "nominal_daily": nominal_daily,
        "bucket_map": bucket_map,
        "baseline_cfg": baseline_cfg,
        "all_sessions": all_sessions,
        "oos_sessions": oos_sessions,
    }


def _build_variant_controls(base_controls: pd.DataFrame, high_multiplier: float) -> pd.DataFrame:
    controls = base_controls.copy()
    controls["bucket_label"] = controls["bucket_label"].astype(str)
    controls["risk_multiplier"] = controls["bucket_label"].map(
        {
            "low": LOW_MULTIPLIER,
            "mid": MID_MULTIPLIER,
            "high": float(high_multiplier),
        }
    ).fillna(0.0)
    controls["skip_trade"] = pd.to_numeric(controls["risk_multiplier"], errors="coerce").fillna(0.0).le(0.0)
    return controls


def _oos_daily_curve(trades: pd.DataFrame, oos_sessions: list, initial_capital: float) -> pd.DataFrame:
    daily = _rebuild_daily_results_from_trades(trades, all_sessions=oos_sessions, initial_capital=initial_capital)
    daily = daily.sort_values("session_date").reset_index(drop=True)
    daily["session_date"] = pd.to_datetime(daily["session_date"], errors="coerce")
    return daily


def _sortino_from_daily_pnl(daily_pnl: pd.Series, capital: float) -> float:
    if capital <= 0:
        return 0.0
    rets = pd.to_numeric(daily_pnl, errors="coerce").fillna(0.0) / float(capital)
    downside = rets[rets < 0]
    if len(rets) < 2 or downside.empty:
        return 0.0
    downside_std = float(np.sqrt(np.mean(np.square(downside))))
    if downside_std <= 0:
        return 0.0
    return float(rets.mean() / downside_std * np.sqrt(252.0))


def _compute_variant_metrics(
    trades: pd.DataFrame,
    controls: pd.DataFrame,
    oos_sessions: list,
    initial_capital: float,
    variant_name: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    oos_set = set(pd.to_datetime(pd.Index(oos_sessions), errors="coerce").date)
    oos_trades = trades.loc[pd.to_datetime(trades["session_date"], errors="coerce").dt.date.isin(oos_set)].copy()
    oos_trades = oos_trades.sort_values("entry_time").reset_index(drop=True)
    oos_daily = _oos_daily_curve(oos_trades, oos_sessions=oos_sessions, initial_capital=initial_capital)

    pnl = pd.to_numeric(oos_trades["net_pnl_usd"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss_abs = float(abs(losses.sum()))
    profit_factor = float(gross_profit / gross_loss_abs) if gross_loss_abs > 0 else float("inf")

    daily_pnl = pd.to_numeric(oos_daily["daily_pnl_usd"], errors="coerce").fillna(0.0)
    daily_ret = daily_pnl / float(initial_capital) if initial_capital > 0 else daily_pnl * 0.0
    daily_std = float(daily_ret.std(ddof=0)) if len(daily_ret) > 1 else 0.0
    sharpe = float(daily_ret.mean() / daily_std * np.sqrt(252.0)) if daily_std > 0 else 0.0
    sortino = _sortino_from_daily_pnl(daily_pnl, capital=initial_capital)

    bucket_lookup = controls[["session_date", "bucket_label", "risk_multiplier"]].copy()
    bucket_lookup["session_date"] = pd.to_datetime(bucket_lookup["session_date"], errors="coerce").dt.date
    oos_trades["session_date_key"] = pd.to_datetime(oos_trades["session_date"], errors="coerce").dt.date
    if "bucket_label" not in oos_trades.columns:
        oos_trades = oos_trades.merge(
            bucket_lookup[["session_date", "bucket_label"]],
            left_on="session_date_key",
            right_on="session_date",
            how="left",
        )

    bucket_contrib = (
        oos_trades.groupby("bucket_label", dropna=False)["net_pnl_usd"]
        .sum()
        .reindex(["low", "mid", "high"], fill_value=0.0)
        .reset_index()
        .rename(columns={"net_pnl_usd": "oos_bucket_net_pnl_usd"})
    )
    bucket_contrib["variant_name"] = variant_name
    bucket_contrib["high_multiplier"] = float(variant_name.split("_")[-1].replace("p", "."))

    non_zero_days = oos_daily.loc[pd.to_numeric(oos_daily["daily_trade_count"], errors="coerce").fillna(0.0) > 0].copy()
    avg_daily_non_zero = float(pd.to_numeric(non_zero_days["daily_pnl_usd"], errors="coerce").mean()) if not non_zero_days.empty else 0.0
    worst_daily = float(daily_pnl.min()) if not oos_daily.empty else 0.0
    max_dd_usd = float(pd.to_numeric(oos_daily["drawdown_usd"], errors="coerce").min()) if not oos_daily.empty else 0.0
    max_dd_pct = float(pd.to_numeric(oos_daily["drawdown_pct"], errors="coerce").max()) * 100.0 if not oos_daily.empty else 0.0

    metrics = {
        "variant_name": variant_name,
        "high_bucket_multiplier": float(variant_name.split("_")[-1].replace("p", ".")),
        "oos_net_pnl_usd": float(pnl.sum()),
        "oos_annualized_sharpe": sharpe,
        "oos_sortino": sortino,
        "oos_max_drawdown_usd": max_dd_usd,
        "oos_max_drawdown_pct": max_dd_pct,
        "oos_win_rate": float((pnl > 0).mean()) if len(pnl) > 0 else 0.0,
        "oos_profit_factor": profit_factor,
        "oos_avg_daily_pnl_non_zero_usd": avg_daily_non_zero,
        "oos_worst_daily_pnl_usd": worst_daily,
        "oos_n_trades": int(len(oos_trades)),
        "oos_avg_risk_multiplier": float(pd.to_numeric(oos_trades["risk_multiplier"], errors="coerce").mean()) if len(oos_trades) > 0 else 0.0,
        "oos_bucket_low_pnl_usd": float(bucket_contrib.loc[bucket_contrib["bucket_label"] == "low", "oos_bucket_net_pnl_usd"].iloc[0]),
        "oos_bucket_mid_pnl_usd": float(bucket_contrib.loc[bucket_contrib["bucket_label"] == "mid", "oos_bucket_net_pnl_usd"].iloc[0]),
        "oos_bucket_high_pnl_usd": float(bucket_contrib.loc[bucket_contrib["bucket_label"] == "high", "oos_bucket_net_pnl_usd"].iloc[0]),
    }
    return metrics, oos_daily, bucket_contrib


def _variant_label(multiplier: float) -> str:
    return f"high_{str(multiplier).replace('.', 'p')}"


def _rank_variants(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    ranked = ranked.sort_values(
        by=["oos_annualized_sharpe", "oos_max_drawdown_usd", "oos_net_pnl_usd"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def _select_recommendation(summary: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    baseline = summary.loc[np.isclose(summary["high_bucket_multiplier"], 0.75)].iloc[0]
    view = summary.copy()
    view["pnl_retention_vs_075"] = np.where(
        float(baseline["oos_net_pnl_usd"]) != 0.0,
        pd.to_numeric(view["oos_net_pnl_usd"], errors="coerce") / float(baseline["oos_net_pnl_usd"]),
        np.nan,
    )
    view["dd_improvement_vs_075_usd"] = abs(float(baseline["oos_max_drawdown_usd"])) - abs(
        pd.to_numeric(view["oos_max_drawdown_usd"], errors="coerce")
    )
    view["worst_day_improvement_vs_075_usd"] = abs(float(baseline["oos_worst_daily_pnl_usd"])) - abs(
        pd.to_numeric(view["oos_worst_daily_pnl_usd"], errors="coerce")
    )
    view["trade_retention_vs_075"] = np.where(
        int(baseline["oos_n_trades"]) > 0,
        pd.to_numeric(view["oos_n_trades"], errors="coerce") / int(baseline["oos_n_trades"]),
        np.nan,
    )

    preferred = view.loc[
        (view["pnl_retention_vs_075"] >= 0.70)
        & (view["dd_improvement_vs_075_usd"] >= max(250.0, 0.10 * abs(float(baseline["oos_max_drawdown_usd"]))))
        & (view["worst_day_improvement_vs_075_usd"] >= 0.0)
    ].copy()
    if not preferred.empty:
        preferred = preferred.sort_values(
            by=["oos_annualized_sharpe", "oos_max_drawdown_usd", "oos_net_pnl_usd"],
            ascending=[False, False, False],
        )
        return preferred.iloc[0], view

    fallback = view.sort_values(
        by=["oos_annualized_sharpe", "oos_max_drawdown_usd", "oos_net_pnl_usd"],
        ascending=[False, False, False],
    ).iloc[0]
    return fallback, view


def _write_report(
    output_dir: Path,
    export_root: Path,
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    bucket_contrib: pd.DataFrame,
    bucket_map: pd.DataFrame,
    recommendation: pd.Series,
    decision_view: pd.DataFrame,
) -> Path:
    baseline = summary.loc[np.isclose(summary["high_bucket_multiplier"], 0.75)].iloc[0]
    rec = recommendation
    bucket_map_lines = "\n".join(
        f"- `{row.bucket_label}`: [{float(row.lower_bound):.6f}, {float(row.upper_bound):.6f}] -> default `{float(row.risk_multiplier):.2f}x`"
        for row in bucket_map.itertuples(index=False)
    )
    top_table = ranking[
        ["rank", "variant_name", "high_bucket_multiplier", "oos_annualized_sharpe", "oos_max_drawdown_usd", "oos_net_pnl_usd", "oos_worst_daily_pnl_usd", "oos_n_trades"]
    ].to_string(index=False)
    report = f"""# MNQ ORB 3-state high-bucket stress

Source export:
- `{export_root}`

Tested variants:
- `low={LOW_MULTIPLIER:.2f}x`, `mid={MID_MULTIPLIER:.2f}x`, `high in {HIGH_BUCKET_MULTIPLIERS}`

Reference bucket map:
{bucket_map_lines}

## Recommendation

Recommended live-oriented variant: **`{rec['variant_name']}`** with `high={float(rec['high_bucket_multiplier']):.2f}x`.

Why:
- OOS Sharpe: `{float(rec['oos_annualized_sharpe']):.3f}` vs current `0.75x` at `{float(baseline['oos_annualized_sharpe']):.3f}`
- OOS net PnL: `{float(rec['oos_net_pnl_usd']):,.1f} USD` vs current `0.75x` at `{float(baseline['oos_net_pnl_usd']):,.1f} USD`
- OOS max drawdown: `{float(rec['oos_max_drawdown_usd']):,.1f} USD` vs current `0.75x` at `{float(baseline['oos_max_drawdown_usd']):,.1f} USD`
- OOS worst daily PnL: `{float(rec['oos_worst_daily_pnl_usd']):,.1f} USD` vs current `0.75x` at `{float(baseline['oos_worst_daily_pnl_usd']):,.1f} USD`
- OOS trades: `{int(rec['oos_n_trades'])}` vs current `0.75x` at `{int(baseline['oos_n_trades'])}`

Live interpretation:
- The objective here is not to maximize raw PnL.
- The preferred variant is the one that preserves a useful share of OOS PnL while reducing path risk, especially drawdown and worst day.
- `high=0.00x` should not be selected blindly if it removes too much PnL or too many trades.

## Ranking

{top_table}

## Conservative takeaway

- If the `high` bucket contributes weakly or negatively while worsening drawdown, it should be cut aggressively.
- If `high=0.00x` improves risk but leaves too little PnL or diversification, prefer an intermediate setting.
- The final choice should bias toward smoother live behavior rather than the highest backtest gross.
"""
    path = output_dir / "high_bucket_stress_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def run_campaign(export_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Path]:
    export_root = export_root or _latest_export_root()
    output_dir = _make_output_dir(output_dir)
    loaded = _load_inputs(export_root)

    nominal_trades = loaded["nominal_trades"].copy()
    base_controls = loaded["variant_controls"].copy()
    baseline_cfg = loaded["baseline_cfg"]
    initial_capital = float(baseline_cfg["account_size_usd"])
    base_risk_pct = float(baseline_cfg["risk_per_trade_pct"])
    oos_sessions = loaded["oos_sessions"]

    summary_rows: list[dict[str, object]] = []
    bucket_contrib_rows: list[pd.DataFrame] = []
    curve_rows: list[pd.DataFrame] = []
    dd_rows: list[pd.DataFrame] = []

    for high_multiplier in HIGH_BUCKET_MULTIPLIERS:
        spec = VariantSpec(name=_variant_label(high_multiplier), high_multiplier=float(high_multiplier))
        controls = _build_variant_controls(base_controls, high_multiplier=spec.high_multiplier)
        trades = _scale_nominal_trades_by_multiplier(
            nominal_trades=nominal_trades,
            controls=controls,
            account_size_usd=initial_capital,
            base_risk_pct=base_risk_pct,
            tick_value_usd=0.5,
            point_value_usd=2.0,
            commission_per_side_usd=1.25,
        )
        metrics, oos_daily, bucket_contrib = _compute_variant_metrics(
            trades=trades,
            controls=controls,
            oos_sessions=oos_sessions,
            initial_capital=initial_capital,
            variant_name=spec.name,
        )
        summary_rows.append(metrics)
        bucket_contrib_rows.append(bucket_contrib)

        curve_rows.append(
            oos_daily.assign(variant_name=spec.name)[["session_date", "equity", "variant_name"]].copy()
        )
        dd_rows.append(
            oos_daily.assign(variant_name=spec.name)[["session_date", "drawdown_usd", "variant_name"]].copy()
        )

    summary = pd.DataFrame(summary_rows).sort_values("high_bucket_multiplier", ascending=False).reset_index(drop=True)
    bucket_contrib = pd.concat(bucket_contrib_rows, ignore_index=True)
    ranking = _rank_variants(summary)
    recommendation, decision_view = _select_recommendation(summary)

    summary_path = output_dir / "summary_metrics.csv"
    bucket_path = output_dir / "bucket_contribution.csv"
    ranking_path = output_dir / "variant_ranking.csv"
    summary.to_csv(summary_path, index=False)
    bucket_contrib.to_csv(bucket_path, index=False)
    ranking.to_csv(ranking_path, index=False)

    curves = pd.concat(curve_rows, ignore_index=True)
    fig = px.line(
        curves,
        x="session_date",
        y="equity",
        color="variant_name",
        title="MNQ ORB 3-state high-bucket stress - OOS equity curves",
        labels={"session_date": "Session date", "equity": "Equity (USD)", "variant_name": "Variant"},
    )
    equity_html = output_dir / "equity_curves.html"
    fig.write_html(equity_html)

    drawdowns = pd.concat(dd_rows, ignore_index=True)
    fig = px.line(
        drawdowns,
        x="session_date",
        y="drawdown_usd",
        color="variant_name",
        title="MNQ ORB 3-state high-bucket stress - OOS drawdowns",
        labels={"session_date": "Session date", "drawdown_usd": "Drawdown (USD)", "variant_name": "Variant"},
    )
    drawdown_html = output_dir / "drawdowns.html"
    fig.write_html(drawdown_html)

    fig = px.bar(
        bucket_contrib,
        x="variant_name",
        y="oos_bucket_net_pnl_usd",
        color="bucket_label",
        barmode="group",
        title="MNQ ORB 3-state high-bucket stress - OOS bucket PnL contribution",
        labels={"variant_name": "Variant", "oos_bucket_net_pnl_usd": "OOS bucket PnL (USD)", "bucket_label": "Bucket"},
    )
    bucket_html = output_dir / "bucket_contributions.html"
    fig.write_html(bucket_html)

    report_path = _write_report(
        output_dir=output_dir,
        export_root=export_root,
        summary=summary,
        ranking=ranking,
        bucket_contrib=bucket_contrib,
        bucket_map=loaded["bucket_map"],
        recommendation=recommendation,
        decision_view=decision_view,
    )

    return {
        "summary_metrics": summary_path,
        "bucket_contribution": bucket_path,
        "variant_ranking": ranking_path,
        "report": report_path,
        "equity_curves": equity_html,
        "drawdowns": drawdown_html,
        "bucket_contributions": bucket_html,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test MNQ ORB sizing_3state high bucket.")
    parser.add_argument("--export-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    outputs = run_campaign(export_root=args.export_root, output_dir=args.output_dir)
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
