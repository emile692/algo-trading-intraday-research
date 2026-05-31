"""Build a client-facing notebook for the retained MNQ ORB 3-state sizing overlay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_EXPORTS_ROOT = REPO_ROOT / "data" / "exports"
REPORT_EXPORTS_ROOT = REPO_ROOT / "export"
NOTEBOOKS_ROOT = REPO_ROOT / "notebooks"

DEFAULT_VARIANT_NAME = "sizing_3state_realized_vol_ratio_15_60"
DEFAULT_BASELINE_NAME = "nominal"
DEFAULT_HIGH_BUCKET_MULTIPLIER = 0.25
DEFAULT_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "orb_MNQ_sizing_3state_client.ipynb"
DEFAULT_EXECUTED_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "orb_MNQ_sizing_3state_client.executed.ipynb"


def find_latest_export(prefix: str, exports_root: Path = DATA_EXPORTS_ROOT) -> Path:
    candidates = [path for path in exports_root.glob(f"{prefix}_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No export folder found for prefix {prefix!r} under {exports_root}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _title_cell() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """# MNQ ORB Client Notebook - `sizing_3state` retenu

Ce notebook presente la version client **solo** du `3-state sizing` retenu pour le MNQ ORB.

- **Signal ORB inchange**: on repart du `nominal` officiel.
- **Overlay seulement**: le sizing varie par bucket sur `realized_vol_ratio_15_60`.
- **Version retenue**: `low = 0.50x`, `mid = 1.00x`, `high = 0.25x`.
- **But du document**: montrer les parametres, la calibration IS, puis la lecture performance/equity de la variante retenue.
"""
    )


def _imports_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """import json
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent

if not (ROOT / "pyproject.toml").exists():
    raise RuntimeError("Impossible de retrouver la racine du repo depuis le notebook.")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import Markdown, display
from plotly.subplots import make_subplots

from src.analytics.metrics import compute_metrics
from src.analytics.mnq_orb_prop_survivability_campaign import _rebuild_daily_results_from_trades
from src.analytics.mnq_orb_regime_filter_sizing_campaign import (
    RegimeFeatureSpec,
    _scale_nominal_trades_by_multiplier,
    build_conditional_bucket_analysis,
    build_static_regime_controls,
)
from src.analytics.orb_opposite_breakout_invalidation_campaign import (
    CampaignConfig as OppositeInvalidationCampaignConfig,
    OppositeBreakoutInvalidationSpec,
    _prepare_asset_data as prepare_opposite_invalidation_asset,
    run_single_asset_config,
)

pd.set_option("display.max_columns", 300)
pd.set_option("display.width", 240)

PLOT_TEMPLATE = "plotly_white"


def fmt_money(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.1f} USD"


def fmt_float(value, digits=3):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def fmt_pct(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100.0:.1f}%"


def build_curve_from_daily(daily, initial_balance):
    out = daily.copy()
    out["session_date"] = pd.to_datetime(out["session_date"], errors="coerce")
    out = out.sort_values("session_date").reset_index(drop=True)
    out["daily_pnl_usd"] = pd.to_numeric(out["daily_pnl_usd"], errors="coerce").fillna(0.0)
    out["equity"] = initial_balance + out["daily_pnl_usd"].cumsum()
    out["peak_equity"] = out["equity"].cummax()
    out["drawdown_usd"] = out["equity"] - out["peak_equity"]
    out["drawdown_pct"] = np.where(out["peak_equity"] > 0, (out["equity"] / out["peak_equity"] - 1.0) * 100.0, 0.0)
    return out


def rebase_oos_curve(curve_df, start_date, initial_balance):
    curve = curve_df.loc[curve_df["session_date"] >= pd.Timestamp(start_date)].copy()
    curve = curve.sort_values("session_date").reset_index(drop=True)
    curve["equity"] = initial_balance + curve["daily_pnl_usd"].cumsum()
    curve["peak_equity"] = curve["equity"].cummax()
    curve["drawdown_usd"] = curve["equity"] - curve["peak_equity"]
    curve["drawdown_pct"] = np.where(curve["peak_equity"] > 0, (curve["equity"] / curve["peak_equity"] - 1.0) * 100.0, 0.0)
    return curve


def scope_metrics(trades, sessions, initial_capital):
    metrics = compute_metrics(trades, session_dates=sessions, initial_capital=initial_capital)
    return {
        "net_pnl": float(metrics.get("cumulative_pnl", 0.0)),
        "sharpe": float(metrics.get("sharpe_ratio", 0.0)),
        "sortino": float(metrics.get("sortino_ratio", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "expectancy": float(metrics.get("expectancy", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "n_trades": int(metrics.get("n_trades", 0)),
        "pct_days_traded": float(metrics.get("percent_of_days_traded", 0.0)),
        "worst_day": float(metrics.get("worst_day", 0.0)),
        "win_rate": float(metrics.get("win_rate", 0.0)),
    }


def build_scope_frame(trades, all_sessions, is_sessions, oos_sessions, initial_capital):
    return pd.DataFrame(
        [
            {"scope": "overall", **scope_metrics(trades, all_sessions, initial_capital)},
            {
                "scope": "is",
                **scope_metrics(
                    trades.loc[pd.to_datetime(trades["session_date"], errors="coerce").dt.date.isin(set(is_sessions))].copy(),
                    is_sessions,
                    initial_capital,
                ),
            },
            {
                "scope": "oos",
                **scope_metrics(
                    trades.loc[pd.to_datetime(trades["session_date"], errors="coerce").dt.date.isin(set(oos_sessions))].copy(),
                    oos_sessions,
                    initial_capital,
                ),
            },
        ]
    )


def compact_metric_frame(frame):
    out = frame.copy()
    for col in ["net_pnl", "expectancy", "max_drawdown", "worst_day"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(1)
    for col in ["sharpe", "sortino", "profit_factor", "pct_days_traded", "win_rate"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    return out
"""
    )


def _parameter_cell(regime_export_root: Path, stress_export_root: Path) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        f"""REGIME_EXPORT_ROOT = ROOT / r"{regime_export_root.relative_to(REPO_ROOT)}"
STRESS_EXPORT_ROOT = ROOT / r"{stress_export_root.relative_to(REPO_ROOT)}"
VARIANT_NAME = "{DEFAULT_VARIANT_NAME}"
BASELINE_NAME = "{DEFAULT_BASELINE_NAME}"
LOW_BUCKET_MULTIPLIER = 0.50
MID_BUCKET_MULTIPLIER = 1.00
HIGH_BUCKET_MULTIPLIER = {DEFAULT_HIGH_BUCKET_MULTIPLIER}

required_paths = {{
    "regime_export_root": REGIME_EXPORT_ROOT,
    "summary_variants": REGIME_EXPORT_ROOT / "summary_variants.csv",
    "baseline_daily": REGIME_EXPORT_ROOT / "variants" / BASELINE_NAME / "daily_results.csv",
    "baseline_trades": REGIME_EXPORT_ROOT / "variants" / BASELINE_NAME / "trades.csv",
    "baseline_controls": REGIME_EXPORT_ROOT / "variants" / BASELINE_NAME / "controls.csv",
    "variant_controls": REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "controls.csv",
    "feature_ranking": REGIME_EXPORT_ROOT / "feature_ranking.csv",
    "conditional_bucket_analysis": REGIME_EXPORT_ROOT / "conditional_bucket_analysis.csv",
    "regime_mapping": REGIME_EXPORT_ROOT / "regime_state_mappings.csv",
    "stress_summary_metrics": STRESS_EXPORT_ROOT / "summary_metrics.csv",
    "stress_variant_ranking": STRESS_EXPORT_ROOT / "variant_ranking.csv",
}}

missing = [name for name, path in required_paths.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"Fichiers manquants pour le notebook: {{missing}}")

print("REGIME_EXPORT_ROOT =", REGIME_EXPORT_ROOT)
print("STRESS_EXPORT_ROOT =", STRESS_EXPORT_ROOT)
print("VARIANT_NAME       =", VARIANT_NAME)
print("BASELINE_NAME      =", BASELINE_NAME)
print("RETAINED MULTIPLIERS =", {{"low": LOW_BUCKET_MULTIPLIER, "mid": MID_BUCKET_MULTIPLIER, "high": HIGH_BUCKET_MULTIPLIER}})
"""
    )


def _load_data_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """regime_metadata = json.loads((REGIME_EXPORT_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
summary_variants = pd.read_csv(REGIME_EXPORT_ROOT / "summary_variants.csv")
baseline_daily = pd.read_csv(REGIME_EXPORT_ROOT / "variants" / BASELINE_NAME / "daily_results.csv", parse_dates=["session_date"])
baseline_trades = pd.read_csv(
    REGIME_EXPORT_ROOT / "variants" / BASELINE_NAME / "trades.csv",
    parse_dates=["session_date", "entry_time", "exit_time"],
)
baseline_controls = pd.read_csv(
    REGIME_EXPORT_ROOT / "variants" / BASELINE_NAME / "controls.csv",
    parse_dates=["session_date"],
)
variant_controls = pd.read_csv(
    REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "controls.csv",
    parse_dates=["session_date"],
)
feature_ranking = pd.read_csv(REGIME_EXPORT_ROOT / "feature_ranking.csv")
conditional_bucket_analysis = pd.read_csv(REGIME_EXPORT_ROOT / "conditional_bucket_analysis.csv")
regime_mapping = pd.read_csv(REGIME_EXPORT_ROOT / "regime_state_mappings.csv")
stress_summary = pd.read_csv(STRESS_EXPORT_ROOT / "summary_metrics.csv")
stress_ranking = pd.read_csv(STRESS_EXPORT_ROOT / "variant_ranking.csv")

baseline_trades["session_date"] = pd.to_datetime(baseline_trades["session_date"], errors="coerce")
baseline_trades["entry_time"] = pd.to_datetime(baseline_trades["entry_time"], errors="coerce", utc=True)
baseline_trades["exit_time"] = pd.to_datetime(baseline_trades["exit_time"], errors="coerce", utc=True)
baseline_controls["session_date"] = pd.to_datetime(baseline_controls["session_date"], errors="coerce")
variant_controls["session_date"] = pd.to_datetime(variant_controls["session_date"], errors="coerce")

baseline_config = regime_metadata["spec"]["baseline"]
min_bucket_obs_is_threshold = int(regime_metadata["spec"].get("min_bucket_obs_is", 50))
initial_capital = float(baseline_config["account_size_usd"])
base_risk_pct = float(baseline_config["risk_per_trade_pct"])
dataset_path_candidate = Path(str(regime_metadata["dataset_path"]))
if not dataset_path_candidate.exists():
    dataset_path_candidate = ROOT / "data" / "processed" / "parquet" / dataset_path_candidate.name
all_sessions = pd.to_datetime(baseline_daily["session_date"], errors="coerce").dt.date.tolist()
is_sessions = pd.to_datetime(variant_controls.loc[variant_controls["phase"] == "is", "session_date"], errors="coerce").dt.date.tolist()
oos_sessions = pd.to_datetime(variant_controls.loc[variant_controls["phase"] == "oos", "session_date"], errors="coerce").dt.date.tolist()
oos_start_date = pd.to_datetime(variant_controls.loc[variant_controls["phase"] == "oos", "session_date"].min())

baseline_scope_df = build_scope_frame(baseline_trades, all_sessions, is_sessions, oos_sessions, initial_capital)

selected_feature = feature_ranking.loc[feature_ranking["feature_name"] == "realized_vol_ratio_15_60"].iloc[0]

bucket_map = (
    regime_mapping.loc[
        (regime_mapping["variant_name"] == VARIANT_NAME)
        & (regime_mapping["feature_name"] == "realized_vol_ratio_15_60"),
        ["bucket_label", "bucket_position", "lower_bound", "upper_bound", "is_n_obs", "is_sharpe", "is_profit_factor", "is_expectancy", "oos_n_obs"],
    ]
    .drop_duplicates()
    .sort_values("bucket_position")
    .reset_index(drop=True)
)

bucket_multiplier_map = {"low": LOW_BUCKET_MULTIPLIER, "mid": MID_BUCKET_MULTIPLIER, "high": HIGH_BUCKET_MULTIPLIER}
bucket_map["risk_multiplier"] = bucket_map["bucket_label"].map(bucket_multiplier_map)
bucket_map["effective_risk_per_trade_pct"] = bucket_map["risk_multiplier"] * base_risk_pct

variant_controls["bucket_label"] = variant_controls["bucket_label"].astype(str)
variant_controls["risk_multiplier"] = variant_controls["bucket_label"].map(bucket_multiplier_map).fillna(0.0)
variant_controls["skip_trade"] = pd.to_numeric(variant_controls["risk_multiplier"], errors="coerce").fillna(0.0).le(0.0)

variant_trades = _scale_nominal_trades_by_multiplier(
    nominal_trades=baseline_trades,
    controls=variant_controls,
    account_size_usd=initial_capital,
    base_risk_pct=base_risk_pct,
    tick_value_usd=0.5,
    point_value_usd=2.0,
    commission_per_side_usd=1.25,
)
variant_trades["session_date"] = pd.to_datetime(variant_trades["session_date"], errors="coerce")
variant_trades["entry_time"] = pd.to_datetime(variant_trades["entry_time"], errors="coerce", utc=True)
variant_trades["exit_time"] = pd.to_datetime(variant_trades["exit_time"], errors="coerce", utc=True)

variant_daily = _rebuild_daily_results_from_trades(variant_trades, all_sessions=all_sessions, initial_capital=initial_capital)
variant_daily["session_date"] = pd.to_datetime(variant_daily["session_date"], errors="coerce")
variant_scope_df = build_scope_frame(variant_trades, all_sessions, is_sessions, oos_sessions, initial_capital)

sizing_only_trades = variant_trades.copy()
sizing_only_daily = variant_daily.copy()
sizing_only_scope_df = variant_scope_df.copy()

opposite_invalidation_config = OppositeInvalidationCampaignConfig(
    symbols=("MNQ",),
    dataset_paths={"MNQ": dataset_path_candidate},
    start_date=None,
    end_date=None,
    opening_time=str(baseline_config["opening_time"]),
    or_minutes=int(baseline_config["or_minutes"]),
    entry_buffer_ticks=int(baseline_config["entry_buffer_ticks"]),
    stop_buffer_ticks=int(baseline_config["stop_buffer_ticks"]),
    target_multiple=float(baseline_config["target_multiple"]),
    direction=str(baseline_config["direction"]),
    one_trade_per_day=bool(baseline_config["one_trade_per_day"]),
    vwap_confirmation=bool(baseline_config["vwap_confirmation"]),
    vwap_column=str(baseline_config["vwap_column"]),
    time_exit=str(baseline_config["time_exit"]),
    account_size_usd=float(baseline_config["account_size_usd"]),
    risk_per_trade_pct=float(baseline_config["risk_per_trade_pct"]),
    entry_on_next_open=bool(baseline_config.get("entry_on_next_open", True)),
    session_selection_path=REGIME_EXPORT_ROOT / "variants" / BASELINE_NAME / "controls.csv",
    cache_namespace="mnq_orb_sizing_client_pre_sizing_invalidation",
)
opposite_invalidation_spec = OppositeBreakoutInvalidationSpec(
    name="invalidate_on_opposite_n_closes_1m__buffer_2__confirm_3",
    description="Invalidate long setup after 3 consecutive 1m closes below OR low with 2 ticks buffer.",
    policy_family="invalidate_for_day",
    opposite_confirmation="n_closes_1m",
    opposite_breakout_buffer_ticks=2,
    opposite_breakout_confirm_bars=3,
)
opposite_prepared = prepare_opposite_invalidation_asset("MNQ", opposite_invalidation_config)
opposite_result = run_single_asset_config(opposite_prepared, opposite_invalidation_spec, opposite_invalidation_config)
opposite_session_summary = pd.DataFrame(opposite_result["session_summary"]).copy()
opposite_session_summary["session_date"] = pd.to_datetime(opposite_session_summary["session_date"], errors="coerce")
surviving_long_sessions = set(
    pd.to_datetime(
        opposite_session_summary.loc[opposite_session_summary["selected_signal_ts"].notna(), "session_date"],
        errors="coerce",
    ).dt.date.tolist()
)

invalidation_trades = baseline_trades.loc[
    (~baseline_trades["direction"].astype(str).str.lower().eq("long"))
    | (pd.to_datetime(baseline_trades["session_date"], errors="coerce").dt.date.isin(surviving_long_sessions))
].copy()
invalidation_daily = _rebuild_daily_results_from_trades(invalidation_trades, all_sessions=all_sessions, initial_capital=initial_capital)
invalidation_daily["session_date"] = pd.to_datetime(invalidation_daily["session_date"], errors="coerce")
invalidation_scope_df = build_scope_frame(invalidation_trades, all_sessions, is_sessions, oos_sessions, initial_capital)

final_trades = _scale_nominal_trades_by_multiplier(
    nominal_trades=invalidation_trades,
    controls=variant_controls,
    account_size_usd=initial_capital,
    base_risk_pct=base_risk_pct,
    tick_value_usd=0.5,
    point_value_usd=2.0,
    commission_per_side_usd=1.25,
)
final_trades["session_date"] = pd.to_datetime(final_trades["session_date"], errors="coerce")
final_trades["entry_time"] = pd.to_datetime(final_trades["entry_time"], errors="coerce", utc=True)
final_trades["exit_time"] = pd.to_datetime(final_trades["exit_time"], errors="coerce", utc=True)
final_daily = _rebuild_daily_results_from_trades(final_trades, all_sessions=all_sessions, initial_capital=initial_capital)
final_daily["session_date"] = pd.to_datetime(final_daily["session_date"], errors="coerce")
final_scope_df = build_scope_frame(final_trades, all_sessions, is_sessions, oos_sessions, initial_capital)

baseline_curve = build_curve_from_daily(baseline_daily, initial_capital)
variant_curve = build_curve_from_daily(final_daily, initial_capital)
invalidation_curve = build_curve_from_daily(invalidation_daily, initial_capital)
sizing_only_curve = build_curve_from_daily(sizing_only_daily, initial_capital)
baseline_curve_oos = rebase_oos_curve(baseline_curve, oos_start_date, initial_capital)
variant_curve_oos = rebase_oos_curve(variant_curve, oos_start_date, initial_capital)
invalidation_curve_oos = rebase_oos_curve(invalidation_curve, oos_start_date, initial_capital)
sizing_only_curve_oos = rebase_oos_curve(sizing_only_curve, oos_start_date, initial_capital)

trade_buckets = final_trades.merge(
    variant_controls[["session_date", "phase", "bucket_label", "risk_multiplier"]],
    on="session_date",
    how="left",
    suffixes=("", "_control"),
)
if "risk_multiplier_control" in trade_buckets.columns:
    trade_buckets["risk_multiplier"] = pd.to_numeric(trade_buckets["risk_multiplier_control"], errors="coerce").fillna(
        pd.to_numeric(trade_buckets.get("risk_multiplier"), errors="coerce")
    )
daily_buckets = final_daily.merge(
    variant_controls[["session_date", "phase", "bucket_label", "risk_multiplier"]],
    on="session_date",
    how="left",
)

comparison_df = pd.DataFrame(
    [
        {
            "variant": "nominal",
            "overall_net_pnl": float(baseline_scope_df.loc[baseline_scope_df["scope"] == "overall", "net_pnl"].iloc[0]),
            "overall_sharpe": float(baseline_scope_df.loc[baseline_scope_df["scope"] == "overall", "sharpe"].iloc[0]),
            "overall_profit_factor": float(baseline_scope_df.loc[baseline_scope_df["scope"] == "overall", "profit_factor"].iloc[0]),
            "overall_max_drawdown": float(baseline_scope_df.loc[baseline_scope_df["scope"] == "overall", "max_drawdown"].iloc[0]),
            "oos_net_pnl": float(baseline_scope_df.loc[baseline_scope_df["scope"] == "oos", "net_pnl"].iloc[0]),
            "oos_sharpe": float(baseline_scope_df.loc[baseline_scope_df["scope"] == "oos", "sharpe"].iloc[0]),
            "oos_profit_factor": float(baseline_scope_df.loc[baseline_scope_df["scope"] == "oos", "profit_factor"].iloc[0]),
            "oos_max_drawdown": float(baseline_scope_df.loc[baseline_scope_df["scope"] == "oos", "max_drawdown"].iloc[0]),
            "oos_n_trades": int(baseline_scope_df.loc[baseline_scope_df["scope"] == "oos", "n_trades"].iloc[0]),
        },
        {
            "variant": "pre_sizing_invalidation_best_campaign",
            "overall_net_pnl": float(invalidation_scope_df.loc[invalidation_scope_df["scope"] == "overall", "net_pnl"].iloc[0]),
            "overall_sharpe": float(invalidation_scope_df.loc[invalidation_scope_df["scope"] == "overall", "sharpe"].iloc[0]),
            "overall_profit_factor": float(invalidation_scope_df.loc[invalidation_scope_df["scope"] == "overall", "profit_factor"].iloc[0]),
            "overall_max_drawdown": float(invalidation_scope_df.loc[invalidation_scope_df["scope"] == "overall", "max_drawdown"].iloc[0]),
            "oos_net_pnl": float(invalidation_scope_df.loc[invalidation_scope_df["scope"] == "oos", "net_pnl"].iloc[0]),
            "oos_sharpe": float(invalidation_scope_df.loc[invalidation_scope_df["scope"] == "oos", "sharpe"].iloc[0]),
            "oos_profit_factor": float(invalidation_scope_df.loc[invalidation_scope_df["scope"] == "oos", "profit_factor"].iloc[0]),
            "oos_max_drawdown": float(invalidation_scope_df.loc[invalidation_scope_df["scope"] == "oos", "max_drawdown"].iloc[0]),
            "oos_n_trades": int(invalidation_scope_df.loc[invalidation_scope_df["scope"] == "oos", "n_trades"].iloc[0]),
        },
        {
            "variant": "sizing_3state_retained_high_0p25",
            "overall_net_pnl": float(sizing_only_scope_df.loc[sizing_only_scope_df["scope"] == "overall", "net_pnl"].iloc[0]),
            "overall_sharpe": float(sizing_only_scope_df.loc[sizing_only_scope_df["scope"] == "overall", "sharpe"].iloc[0]),
            "overall_profit_factor": float(sizing_only_scope_df.loc[sizing_only_scope_df["scope"] == "overall", "profit_factor"].iloc[0]),
            "overall_max_drawdown": float(sizing_only_scope_df.loc[sizing_only_scope_df["scope"] == "overall", "max_drawdown"].iloc[0]),
            "oos_net_pnl": float(sizing_only_scope_df.loc[sizing_only_scope_df["scope"] == "oos", "net_pnl"].iloc[0]),
            "oos_sharpe": float(sizing_only_scope_df.loc[sizing_only_scope_df["scope"] == "oos", "sharpe"].iloc[0]),
            "oos_profit_factor": float(sizing_only_scope_df.loc[sizing_only_scope_df["scope"] == "oos", "profit_factor"].iloc[0]),
            "oos_max_drawdown": float(sizing_only_scope_df.loc[sizing_only_scope_df["scope"] == "oos", "max_drawdown"].iloc[0]),
            "oos_n_trades": int(sizing_only_scope_df.loc[sizing_only_scope_df["scope"] == "oos", "n_trades"].iloc[0]),
        },
        {
            "variant": "best_campaign_invalidation_plus_sizing_high_0p25",
            "overall_net_pnl": float(final_scope_df.loc[final_scope_df["scope"] == "overall", "net_pnl"].iloc[0]),
            "overall_sharpe": float(final_scope_df.loc[final_scope_df["scope"] == "overall", "sharpe"].iloc[0]),
            "overall_profit_factor": float(final_scope_df.loc[final_scope_df["scope"] == "overall", "profit_factor"].iloc[0]),
            "overall_max_drawdown": float(final_scope_df.loc[final_scope_df["scope"] == "overall", "max_drawdown"].iloc[0]),
            "oos_net_pnl": float(final_scope_df.loc[final_scope_df["scope"] == "oos", "net_pnl"].iloc[0]),
            "oos_sharpe": float(final_scope_df.loc[final_scope_df["scope"] == "oos", "sharpe"].iloc[0]),
            "oos_profit_factor": float(final_scope_df.loc[final_scope_df["scope"] == "oos", "profit_factor"].iloc[0]),
            "oos_max_drawdown": float(final_scope_df.loc[final_scope_df["scope"] == "oos", "max_drawdown"].iloc[0]),
            "oos_n_trades": int(final_scope_df.loc[final_scope_df["scope"] == "oos", "n_trades"].iloc[0]),
        },
    ]
)

comparison_df["oos_net_pnl_retention_vs_nominal"] = comparison_df["oos_net_pnl"] / float(comparison_df.loc[comparison_df["variant"] == "nominal", "oos_net_pnl"].iloc[0])
comparison_df["oos_sharpe_delta_vs_nominal"] = comparison_df["oos_sharpe"] - float(comparison_df.loc[comparison_df["variant"] == "nominal", "oos_sharpe"].iloc[0])
comparison_df["oos_max_drawdown_improvement_vs_nominal"] = abs(float(comparison_df.loc[comparison_df["variant"] == "nominal", "oos_max_drawdown"].iloc[0])) - abs(comparison_df["oos_max_drawdown"])

chosen_stress_row = stress_ranking.loc[stress_ranking["high_bucket_multiplier"] == HIGH_BUCKET_MULTIPLIER].iloc[0]
invalidation_removed_long_trades = int(
    baseline_trades["direction"].astype(str).str.lower().eq("long").sum()
    - invalidation_trades["direction"].astype(str).str.lower().eq("long").sum()
)
invalidation_blocked_sessions = int(opposite_session_summary["invalidated_for_day"].fillna(False).astype(bool).sum())

display(Markdown(f"**OOS start date:** `{oos_start_date.date()}`"))
display(Markdown(f"**Feature retenu pour l'overlay:** `{selected_feature['feature_name']}`"))
display(Markdown(f"**Choix high bucket retenu par la campagne de stress:** `{HIGH_BUCKET_MULTIPLIER:.2f}x`"))
display(Markdown(f"**Pre-sizing invalidation retenue pour test:** `3 closes 1m sous OR low avec buffer 2 ticks` | source `{opposite_invalidation_spec.name}` | sessions invalidées `{invalidation_blocked_sessions}` | longs retirés `{invalidation_removed_long_trades}`"))
"""
    )


def _quick_read_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 1. Lecture Rapide

Ce notebook raconte la version retenue du `3-state sizing` en lecture client:

- le `mid` reste le moteur principal du PnL,
- le `high` est conserve mais **fortement coupe** a `0.25x`,
- on ajoute aussi le test **pre-sizing**: annuler le setup long apres `3` closes 1m sous `OR low` avec `2 ticks` de buffer,
- la calibration regime garde un cadre simple a relire et a expliquer.
"""
    )


def _quick_read_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """nominal_oos = comparison_df.loc[comparison_df["variant"] == "nominal"].iloc[0]
invalidation_oos = comparison_df.loc[comparison_df["variant"] == "pre_sizing_invalidation_best_campaign"].iloc[0]
sizing_only_oos = comparison_df.loc[comparison_df["variant"] == "sizing_3state_retained_high_0p25"].iloc[0]
retained_oos = comparison_df.loc[comparison_df["variant"] == "best_campaign_invalidation_plus_sizing_high_0p25"].iloc[0]

quick_lines = [
    "### Synthese executive",
    f"- Le filtre pre-sizing `3 closes sous OR low avec buffer 2 ticks` retire **{invalidation_removed_long_trades}** longs et amene un Sharpe OOS de **{invalidation_oos['oos_sharpe']:.3f}**.",
    f"- Le `3-state` seul (`high=0.25`) monte le Sharpe OOS a **{sizing_only_oos['oos_sharpe']:.3f}**.",
    f"- La version finale chainee `best invalidation de campagne + sizing` garde **{retained_oos['oos_net_pnl_retention_vs_nominal'] * 100.0:.1f}%** du net PnL OOS du nominal.",
    f"- Le max drawdown OOS passe de **{fmt_money(nominal_oos['oos_max_drawdown'])}** a **{fmt_money(retained_oos['oos_max_drawdown'])}** sur la version finale.",
    f"- La campagne de stress classe `high=0.25` **rang #{int(chosen_stress_row['rank'])}** avec Sharpe OOS **{chosen_stress_row['oos_annualized_sharpe']:.3f}** et maxDD **{fmt_money(chosen_stress_row['oos_max_drawdown_usd'])}**.",
]

display(Markdown("\\n".join(quick_lines)))
"""
    )


def _parameters_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 2. Parametres Exacts

Ici on fige ce qui est vraiment utilise par la version retenue:

- baseline ORB,
- feature de regime,
- bornes de buckets,
- multiplicateurs de risque.
"""
    )


def _parameters_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """baseline_df = pd.DataFrame([{"parameter": key, "value": value} for key, value in baseline_config.items()])

overlay_df = pd.DataFrame(
    [
        {"parameter": "variant_name_source", "value": VARIANT_NAME},
        {"parameter": "feature_name", "value": selected_feature["feature_name"]},
        {"parameter": "feature_selection_score", "value": round(float(selected_feature["feature_selection_score"]), 3)},
        {"parameter": "best_bucket_is", "value": selected_feature["best_bucket_is"]},
        {"parameter": "worst_bucket_is", "value": selected_feature["worst_bucket_is"]},
        {"parameter": "low_multiplier", "value": LOW_BUCKET_MULTIPLIER},
        {"parameter": "mid_multiplier", "value": MID_BUCKET_MULTIPLIER},
        {"parameter": "high_multiplier_retained", "value": HIGH_BUCKET_MULTIPLIER},
    ]
)

bucket_display = bucket_map.copy()
for col in ["lower_bound", "upper_bound", "is_sharpe", "is_profit_factor", "is_expectancy", "effective_risk_per_trade_pct"]:
    if col in bucket_display.columns:
        bucket_display[col] = pd.to_numeric(bucket_display[col], errors="coerce").round(3)

display(Markdown("### Baseline ORB"))
display(baseline_df)

display(Markdown("### Overlay retenu"))
display(overlay_df)

display(Markdown("### Buckets et exposition effective"))
display(bucket_display)
"""
    )


def _calibration_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 3. Calibration IS et Heatmaps

La calibration ci-dessous reste volontairement lisible:

- pourquoi `realized_vol_ratio_15_60` a ete retenu,
- comment les buckets se comportent en in-sample,
- ou se trouve le vrai moteur du `3-state`.
"""
    )


def _calibration_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """ranking_display = feature_ranking.copy().sort_values("feature_selection_score", ascending=False).reset_index(drop=True)
ranking_display["feature_selection_score"] = ranking_display["feature_selection_score"].round(3)
ranking_display["is_score_spread"] = ranking_display["is_score_spread"].round(3)
ranking_display["skip_coverage_is"] = ranking_display["skip_coverage_is"].round(3)
display(Markdown("### Ranking des features overlay candidats"))
display(ranking_display)

feature_heat_src = ranking_display.loc[:, ["feature_name", "feature_selection_score", "is_score_spread", "skip_coverage_is"]].copy()
feature_heat = feature_heat_src.set_index("feature_name").T

fig_feature = go.Figure(
    data=go.Heatmap(
        z=feature_heat.values,
        x=feature_heat.columns.tolist(),
        y=feature_heat.index.tolist(),
        colorscale="RdYlGn",
        text=np.round(feature_heat.values, 3),
        texttemplate="%{text}",
    )
)
fig_feature.update_layout(
    template=PLOT_TEMPLATE,
    width=1200,
    height=420,
    title="Heatmap IS - qualite des features candidats pour l'overlay",
)
fig_feature.show()

bucket_is = (
    conditional_bucket_analysis.loc[conditional_bucket_analysis["feature_name"] == "realized_vol_ratio_15_60"]
    .sort_values("bucket_position")
    .copy()
)
bucket_is["effective_multiplier_retained"] = bucket_is["bucket_label"].map(bucket_multiplier_map)

bucket_heat = pd.DataFrame(
    {
        bucket: [
            float(bucket_is.loc[bucket_is["bucket_label"] == bucket, "is_sharpe"].iloc[0]),
            float(bucket_is.loc[bucket_is["bucket_label"] == bucket, "is_profit_factor"].iloc[0]),
            float(bucket_is.loc[bucket_is["bucket_label"] == bucket, "is_expectancy"].iloc[0]),
            float(bucket_is.loc[bucket_is["bucket_label"] == bucket, "is_net_pnl"].iloc[0]),
            float(bucket_is.loc[bucket_is["bucket_label"] == bucket, "effective_multiplier_retained"].iloc[0]),
        ]
        for bucket in ["low", "mid", "high"]
    },
    index=["is_sharpe", "is_profit_factor", "is_expectancy", "is_net_pnl_usd", "retained_multiplier"],
)

display(Markdown("### Bucket analysis IS du feature retenu"))
display(
    bucket_is.loc[
        :,
        [
            "bucket_label",
            "lower_bound",
            "upper_bound",
            "is_n_obs",
            "is_net_pnl",
            "is_sharpe",
            "is_profit_factor",
            "is_expectancy",
            "is_max_drawdown",
            "effective_multiplier_retained",
        ],
    ].round(3)
)

fig_bucket = go.Figure(
    data=go.Heatmap(
        z=bucket_heat.values,
        x=bucket_heat.columns.tolist(),
        y=bucket_heat.index.tolist(),
        colorscale="RdYlGn",
        text=np.round(bucket_heat.values, 2),
        texttemplate="%{text}",
    )
)
fig_bucket.update_layout(
    template=PLOT_TEMPLATE,
    width=950,
    height=430,
    title="Heatmap IS - buckets du feature retenu",
)
fig_bucket.show()
"""
    )


def _stability_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 4. Stabilité Large du Couple `vol_fast x vol_slow`

Cette section repond directement a la question:

- est-ce que `15/60` est un point isole,
- ou bien est-ce qu'il vit dans une vraie zone stable,
- et que racontent les couples voisins en IS puis en OOS.

Lecture:

- axe `x` = fenetre **slow**
- axe `y` = fenetre **fast**
- la case `15/60` est marquee explicitement
- on regarde a la fois le **score IS de selection** et le **Sharpe OOS** de l'overlay retenu
"""
    )


def _stability_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
"""stability_fast_windows = list(range(5, 31))
stability_slow_windows = list(range(30, 121, 5))
stability_all_windows = sorted(set(stability_fast_windows + stability_slow_windows))

minute_df = pd.read_parquet(dataset_path_candidate).copy()
if "timestamp" not in minute_df.columns:
    if getattr(minute_df.index, "name", None) == "timestamp":
        minute_df = minute_df.reset_index()
    else:
        minute_df = minute_df.reset_index().rename(columns={minute_df.reset_index().columns[0]: "timestamp"})
minute_df["timestamp"] = pd.to_datetime(minute_df["timestamp"], errors="coerce", utc=True)
minute_df["close"] = pd.to_numeric(minute_df["close"], errors="coerce")
minute_df = minute_df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
close_returns = minute_df["close"].pct_change()

for window in stability_all_windows:
    minute_df[f"vol_std_{window}"] = close_returns.rolling(window).std()

phase_map = variant_controls[["session_date", "phase"]].copy()
phase_map["session_date"] = pd.to_datetime(phase_map["session_date"], errors="coerce").dt.date

signal_rows = baseline_trades.copy()
signal_rows["session_date"] = pd.to_datetime(signal_rows["session_date"], errors="coerce").dt.date
signal_rows["entry_time"] = pd.to_datetime(signal_rows["entry_time"], errors="coerce", utc=True)
signal_rows = signal_rows.loc[signal_rows["session_date"].isin(set(phase_map["session_date"]))].copy()
signal_rows = signal_rows.sort_values(["session_date", "entry_time"]).drop_duplicates(subset=["session_date"], keep="first").reset_index(drop=True)
signal_rows["signal_timestamp"] = signal_rows["entry_time"] - pd.Timedelta(minutes=1)
signal_rows = signal_rows.merge(phase_map, on="session_date", how="left", validate="one_to_one")

feature_cols = [f"vol_std_{window}" for window in stability_all_windows]
signal_feature_rows = pd.merge_asof(
    signal_rows.sort_values("signal_timestamp"),
    minute_df[["timestamp", *feature_cols]].sort_values("timestamp"),
    left_on="signal_timestamp",
    right_on="timestamp",
    direction="backward",
    tolerance=pd.Timedelta(minutes=2),
)
signal_feature_rows["match_gap_seconds"] = (
    signal_feature_rows["signal_timestamp"] - signal_feature_rows["timestamp"]
).dt.total_seconds()

match_rate = float(signal_feature_rows["timestamp"].notna().mean()) if len(signal_feature_rows) else 0.0
display(Markdown(f"### Couverture des lignes signal pour la heatmap large\\n- Match signal bar: **{match_rate * 100.0:.1f}%**"))

bucket_multiplier_map_retained = {
    "low": LOW_BUCKET_MULTIPLIER,
    "mid": MID_BUCKET_MULTIPLIER,
    "high": HIGH_BUCKET_MULTIPLIER,
}

stability_rows = []
for fast_window in stability_fast_windows:
    for slow_window in stability_slow_windows:
        if fast_window >= slow_window:
            continue

        feature_name = f"realized_vol_ratio_{fast_window}_{slow_window}"
        regime_probe = signal_feature_rows[["session_date", "phase"]].copy()
        regime_probe[feature_name] = (
            pd.to_numeric(signal_feature_rows[f"vol_std_{fast_window}"], errors="coerce")
            / pd.to_numeric(signal_feature_rows[f"vol_std_{slow_window}"], errors="coerce")
        )

        conditional_probe, feature_score_probe, assignments_probe, _ = build_conditional_bucket_analysis(
            regime_df=regime_probe,
            nominal_trades=baseline_trades,
            initial_capital=initial_capital,
            feature_specs=(
                RegimeFeatureSpec(
                    name=feature_name,
                    family="volatility",
                    description=f"Realized volatility ratio using rolling close-return stdev {fast_window} vs {slow_window} bars.",
                    value_column=feature_name,
                ),
            ),
            min_bucket_obs_is=min_bucket_obs_is_threshold,
        )
        if feature_score_probe.empty or feature_name not in assignments_probe:
            continue

        controls_probe = build_static_regime_controls(
            regime_df=regime_probe,
            feature_name=feature_name,
            bucket_labels=assignments_probe[feature_name],
            bucket_multipliers=bucket_multiplier_map_retained,
        )
        scaled_probe = _scale_nominal_trades_by_multiplier(
            nominal_trades=baseline_trades,
            controls=controls_probe,
            account_size_usd=initial_capital,
            base_risk_pct=base_risk_pct,
            tick_value_usd=0.5,
            point_value_usd=2.0,
            commission_per_side_usd=1.25,
        )
        scaled_probe["session_date"] = pd.to_datetime(scaled_probe["session_date"], errors="coerce")
        scaled_probe["entry_time"] = pd.to_datetime(scaled_probe["entry_time"], errors="coerce", utc=True)
        scaled_probe["exit_time"] = pd.to_datetime(scaled_probe["exit_time"], errors="coerce", utc=True)
        scaled_scope = build_scope_frame(scaled_probe, all_sessions, is_sessions, oos_sessions, initial_capital)
        oos_probe = scaled_scope.loc[scaled_scope["scope"] == "oos"].iloc[0]

        selected_probe = feature_score_probe.iloc[0]
        stability_rows.append(
            {
                "fast_window": int(fast_window),
                "slow_window": int(slow_window),
                "feature_name": feature_name,
                "feature_selection_score": float(selected_probe["feature_selection_score"]),
                "is_score_spread": float(selected_probe["is_score_spread"]),
                "best_bucket_is": str(selected_probe["best_bucket_is"]),
                "worst_bucket_is": str(selected_probe["worst_bucket_is"]),
                "valid_for_overlay": bool(selected_probe["valid_for_overlay"]),
                "min_bucket_obs_is": int(selected_probe["min_bucket_obs_is"]),
                "oos_sharpe_retained": float(oos_probe["sharpe"]),
                "oos_profit_factor_retained": float(oos_probe["profit_factor"]),
                "oos_net_pnl_retained": float(oos_probe["net_pnl"]),
                "oos_max_drawdown_retained": float(oos_probe["max_drawdown"]),
                "oos_trade_count_retained": int(oos_probe["n_trades"]),
            }
        )

stability_grid = pd.DataFrame(stability_rows).sort_values(["fast_window", "slow_window"]).reset_index(drop=True)

score_pivot = stability_grid.pivot(index="fast_window", columns="slow_window", values="feature_selection_score").sort_index().sort_index(axis=1)
sharpe_pivot = stability_grid.pivot(index="fast_window", columns="slow_window", values="oos_sharpe_retained").sort_index().sort_index(axis=1)

fig_stability = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Score IS de selection", "Sharpe OOS du sizing retenu"),
    horizontal_spacing=0.10,
)
fig_stability.add_trace(
    go.Heatmap(
        z=score_pivot.values,
        x=score_pivot.columns.tolist(),
        y=score_pivot.index.tolist(),
        colorscale="RdYlGn",
        colorbar=dict(title="IS score", x=0.46),
        hovertemplate="slow=%{x}<br>fast=%{y}<br>IS score=%{z:.3f}<extra></extra>",
    ),
    row=1,
    col=1,
)
fig_stability.add_trace(
    go.Heatmap(
        z=sharpe_pivot.values,
        x=sharpe_pivot.columns.tolist(),
        y=sharpe_pivot.index.tolist(),
        colorscale="RdYlGn",
        colorbar=dict(title="OOS Sharpe", x=1.02),
        hovertemplate="slow=%{x}<br>fast=%{y}<br>OOS Sharpe=%{z:.3f}<extra></extra>",
    ),
    row=1,
    col=2,
)
for subplot_col in (1, 2):
    fig_stability.add_trace(
        go.Scatter(
            x=[60],
            y=[15],
            mode="markers+text",
            text=["15/60"],
            textposition="middle center",
            marker=dict(symbol="x", size=14, color="black", line=dict(width=2, color="white")),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=subplot_col,
    )
fig_stability.update_layout(
    template=PLOT_TEMPLATE,
    width=1350,
    height=620,
    title="Heatmap large vol slow x vol fast autour du ratio retenu 15/60",
)
fig_stability.update_xaxes(title_text="Vol slow window", row=1, col=1)
fig_stability.update_xaxes(title_text="Vol slow window", row=1, col=2)
fig_stability.update_yaxes(title_text="Vol fast window", row=1, col=1)
fig_stability.update_yaxes(title_text="Vol fast window", row=1, col=2)
fig_stability.show()

local_neighborhood = (
    stability_grid.loc[
        stability_grid["fast_window"].between(10, 20)
        & stability_grid["slow_window"].between(45, 75)
    ]
    .sort_values(["oos_sharpe_retained", "feature_selection_score"], ascending=[False, False])
    .reset_index(drop=True)
)

anchor_row = stability_grid.loc[
    (stability_grid["fast_window"] == 15) & (stability_grid["slow_window"] == 60)
].iloc[0]

display(Markdown("### Zoom local autour de `15/60`"))
display(
    local_neighborhood.loc[
        :,
        [
            "fast_window",
            "slow_window",
            "feature_selection_score",
            "oos_sharpe_retained",
            "oos_net_pnl_retained",
            "oos_max_drawdown_retained",
            "best_bucket_is",
            "worst_bucket_is",
        ],
    ].head(18).round(3)
)

display(
    Markdown(
        "\\n".join(
            [
                "### Lecture rapide de la zone `15/60`",
                f"- Ancre `15/60`: score IS **{anchor_row['feature_selection_score']:.3f}**, Sharpe OOS **{anchor_row['oos_sharpe_retained']:.3f}**, maxDD OOS **{fmt_money(anchor_row['oos_max_drawdown_retained'])}**.",
                f"- Mediane du voisinage local `(fast 10:20, slow 45:75)`: score IS **{local_neighborhood['feature_selection_score'].median():.3f}**, Sharpe OOS **{local_neighborhood['oos_sharpe_retained'].median():.3f}**.",
                f"- Rang local de `15/60` sur le Sharpe OOS: **#{1 + int((local_neighborhood['oos_sharpe_retained'] > anchor_row['oos_sharpe_retained']).sum())}** sur **{len(local_neighborhood)}** couples.",
            ]
        )
    )
)
"""
    )


def _performance_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 5. Performance Audit

On compare ici quatre etages du meme moteur:

- `nominal`,
- `nominal + best invalidation de campagne`,
- `nominal + sizing 3-state`,
- `best invalidation de campagne + sizing`.
"""
    )


def _performance_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """comparison_display = comparison_df.copy()
for col in ["overall_net_pnl", "overall_max_drawdown", "oos_net_pnl", "oos_max_drawdown"]:
    comparison_display[col] = pd.to_numeric(comparison_display[col], errors="coerce").round(1)
for col in ["overall_sharpe", "overall_profit_factor", "oos_sharpe", "oos_profit_factor", "oos_net_pnl_retention_vs_nominal", "oos_sharpe_delta_vs_nominal", "oos_max_drawdown_improvement_vs_nominal"]:
    comparison_display[col] = pd.to_numeric(comparison_display[col], errors="coerce").round(3)

display(Markdown("### Comparatif incremental"))
display(comparison_display)

scope_stack = pd.concat(
    [
        compact_metric_frame(baseline_scope_df).assign(variant="nominal"),
        compact_metric_frame(invalidation_scope_df).assign(variant="pre_sizing_invalidation_best_campaign"),
        compact_metric_frame(sizing_only_scope_df).assign(variant="sizing_3state_retained_high_0p25"),
        compact_metric_frame(final_scope_df).assign(variant="best_campaign_invalidation_plus_sizing_high_0p25"),
    ],
    ignore_index=True,
)
display(Markdown("### Metrics par scope"))
display(scope_stack)
"""
    )


def _equity_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 6. Courbes de Capital

La lecture client doit voir:

- la courbe historique complete,
- la courbe OOS seule,
- le drawdown associe.
"""
    )


def _equity_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=("Full sample - Equity", "OOS only - Equity", "Full sample - Drawdown", "OOS only - Drawdown"),
    horizontal_spacing=0.1,
    vertical_spacing=0.12,
)

for label, curve, color in [
    ("Nominal", baseline_curve, "#2563eb"),
    ("Invalidation best campaign", invalidation_curve, "#d97706"),
    ("3-state seul", sizing_only_curve, "#7c3aed"),
    ("Best invalidation + 3-state", variant_curve, "#15803d"),
]:
    fig.add_trace(go.Scatter(x=curve["session_date"], y=curve["equity"], mode="lines", name=f"{label} full", line=dict(width=2.6, color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=curve["session_date"], y=curve["drawdown_usd"], mode="lines", name=f"{label} full DD", showlegend=False, line=dict(width=1.8, color=color, dash="dot")), row=2, col=1)

for label, curve, color in [
    ("Nominal", baseline_curve_oos, "#2563eb"),
    ("Invalidation best campaign", invalidation_curve_oos, "#d97706"),
    ("3-state seul", sizing_only_curve_oos, "#7c3aed"),
    ("Best invalidation + 3-state", variant_curve_oos, "#15803d"),
]:
    fig.add_trace(go.Scatter(x=curve["session_date"], y=curve["equity"], mode="lines", name=f"{label} oos", line=dict(width=2.6, color=color)), row=1, col=2)
    fig.add_trace(go.Scatter(x=curve["session_date"], y=curve["drawdown_usd"], mode="lines", name=f"{label} oos DD", showlegend=False, line=dict(width=1.8, color=color, dash="dot")), row=2, col=2)

fig.update_layout(template=PLOT_TEMPLATE, width=1250, height=820, title="Comparatif incremental - equity et drawdown", legend=dict(orientation="h", y=1.08))
fig.update_yaxes(title_text="USD", row=1, col=1)
fig.update_yaxes(title_text="USD", row=1, col=2)
fig.update_yaxes(title_text="DD USD", row=2, col=1)
fig.update_yaxes(title_text="DD USD", row=2, col=2)
fig.show()
"""
    )


def _stress_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 7. Stress Test du High Bucket

Le `high=0.25` n'est pas arbitraire: il vient de la mini-campagne de stress sur le bucket haute vol.
"""
    )


def _stress_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """stress_display = stress_summary.copy().sort_values("high_bucket_multiplier", ascending=False).reset_index(drop=True)
for col in [
    "oos_net_pnl_usd",
    "oos_max_drawdown_usd",
    "oos_avg_daily_pnl_non_zero_usd",
    "oos_worst_daily_pnl_usd",
    "oos_bucket_low_pnl_usd",
    "oos_bucket_mid_pnl_usd",
    "oos_bucket_high_pnl_usd",
]:
    stress_display[col] = pd.to_numeric(stress_display[col], errors="coerce").round(1)
for col in [
    "oos_annualized_sharpe",
    "oos_sortino",
    "oos_max_drawdown_pct",
    "oos_win_rate",
    "oos_profit_factor",
    "oos_avg_risk_multiplier",
]:
    stress_display[col] = pd.to_numeric(stress_display[col], errors="coerce").round(3)

display(Markdown("### Tableau de stress OOS"))
display(stress_display)

fig_stress = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Sharpe vs maxDD", "PnL bucket high vs multiplicateur"),
    horizontal_spacing=0.12,
)
fig_stress.add_trace(
    go.Scatter(
        x=stress_summary["high_bucket_multiplier"],
        y=stress_summary["oos_annualized_sharpe"],
        mode="lines+markers+text",
        text=stress_summary["variant_name"],
        textposition="top center",
        line=dict(color="#15803d", width=2.5),
        name="OOS Sharpe",
    ),
    row=1,
    col=1,
)
fig_stress.add_trace(
    go.Bar(
        x=stress_summary["high_bucket_multiplier"],
        y=stress_summary["oos_bucket_high_pnl_usd"],
        marker_color="#b45309",
        name="High bucket pnl",
    ),
    row=1,
    col=2,
)
fig_stress.update_layout(template=PLOT_TEMPLATE, width=1200, height=450, title="Stress test - impact du high bucket")
fig_stress.update_xaxes(title_text="High bucket multiplier", row=1, col=1)
fig_stress.update_xaxes(title_text="High bucket multiplier", row=1, col=2)
fig_stress.update_yaxes(title_text="Sharpe", row=1, col=1)
fig_stress.update_yaxes(title_text="High bucket pnl (USD)", row=1, col=2)
fig_stress.show()
"""
    )


def _sizing_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 8. Logique de Sizing en Pratique

Cette section montre comment l'overlay retenu se traduit trade par trade et bucket par bucket.
"""
    )


def _sizing_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """bucket_trade_stats = (
    trade_buckets.groupby(["bucket_label", "risk_multiplier"], dropna=False)
    .agg(
        n_trades=("trade_id", "count"),
        avg_quantity=("quantity", "mean"),
        avg_risk_per_trade_pct=("risk_per_trade_pct", "mean"),
        avg_actual_risk_usd=("actual_risk_usd", "mean"),
        total_net_pnl_usd=("net_pnl_usd", "sum"),
        avg_net_pnl_usd=("net_pnl_usd", "mean"),
    )
    .reset_index()
    .sort_values(["risk_multiplier", "bucket_label"])
)

for col in ["avg_quantity", "avg_risk_per_trade_pct", "avg_actual_risk_usd", "total_net_pnl_usd", "avg_net_pnl_usd"]:
    bucket_trade_stats[col] = pd.to_numeric(bucket_trade_stats[col], errors="coerce").round(2)

display(bucket_trade_stats)

fig_bucket_bar = px.bar(
    bucket_trade_stats,
    x="bucket_label",
    y="total_net_pnl_usd",
    color="risk_multiplier",
    barmode="group",
    title="Contribution PnL par bucket - variante retenue",
    labels={"bucket_label": "Bucket", "total_net_pnl_usd": "Net PnL total (USD)", "risk_multiplier": "Risk multiplier"},
)
fig_bucket_bar.update_layout(template=PLOT_TEMPLATE, width=980, height=430)
fig_bucket_bar.show()
"""
    )


def _distribution_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 9. Distribution Journaliere OOS

On finit avec la geometrie journaliere de la variante retenue.
"""
    )


def _distribution_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """hist_df = pd.concat(
    [
        baseline_daily.loc[baseline_daily["session_date"] >= oos_start_date, ["session_date", "daily_pnl_usd"]].assign(variant="nominal"),
        final_daily.loc[final_daily["session_date"] >= oos_start_date, ["session_date", "daily_pnl_usd"]].assign(variant="invalidation_plus_3state"),
    ],
    ignore_index=True,
)
hist_df = hist_df.loc[hist_df["daily_pnl_usd"] != 0].copy()

fig_hist = px.histogram(
    hist_df,
    x="daily_pnl_usd",
    color="variant",
    nbins=60,
    marginal="box",
    opacity=0.58,
    barmode="overlay",
    title="Distribution OOS des daily pnl non nuls",
    labels={"daily_pnl_usd": "Daily pnl (USD)", "variant": "Variant"},
)
fig_hist.update_layout(template=PLOT_TEMPLATE, width=1100, height=460)
fig_hist.show()

best_worst_days = pd.concat(
    [
        daily_buckets.loc[daily_buckets["session_date"] >= oos_start_date].nlargest(5, "daily_pnl_usd"),
        daily_buckets.loc[daily_buckets["session_date"] >= oos_start_date].nsmallest(5, "daily_pnl_usd"),
    ],
    ignore_index=True,
)
best_worst_days = best_worst_days.loc[:, ["session_date", "daily_pnl_usd", "bucket_label", "risk_multiplier", "daily_trade_count"]]
best_worst_days["session_date"] = pd.to_datetime(best_worst_days["session_date"]).dt.date
best_worst_days["daily_pnl_usd"] = pd.to_numeric(best_worst_days["daily_pnl_usd"], errors="coerce").round(1)

display(Markdown("### Meilleurs / pires jours OOS"))
display(best_worst_days)
"""
    )


def _conclusion_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 10. Conclusion Client

La lecture finale doit rester simple:

- le `3-state` ne change pas l'alpha,
- la meilleure invalidation de campagne agit avant le sizing,
- le `mid` reste le coeur du moteur,
- `high=0.25` est la version prudente retenue.
"""
    )


def _conclusion_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """final_lines = [
    "### Verdict simple",
    f"- La version finale ajoute d'abord la meilleure invalidation de la campagne: **3 closes 1m sous OR low avec buffer 2 ticks annulent le long**.",
    f"- Ensuite seulement vient le `3-state`, donc la sequence devient: **signal ORB -> best invalidation campaign -> sizing regime**.",
    f"- La calibration IS montre bien que `realized_vol_ratio_15_60` est le feature overlay le plus lisible et que le bucket `mid` est le vrai moteur.",
    f"- Le bucket `high` reste present mais coupe a **{HIGH_BUCKET_MULTIPLIER:.2f}x**, ce qui preserve l'exposition tactique sans laisser la haute vol dominer le drawdown.",
    f"- En OOS, la variante retenue garde **{retained_oos['oos_net_pnl_retention_vs_nominal'] * 100.0:.1f}%** du PnL du nominal avec un drawdown materially plus propre.",
    f"- Lecture client recommandee: **version live-oriented prudente, plus defendable que l'ancien `high=0.75`, avec la meilleure hygiene supplementaire identifiee par la campagne d'invalidation avant l'overlay**.",
]

display(Markdown("\\n".join(final_lines)))
"""
    )


def _appendix_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 11. Appendice - Sources

Le notebook reste raccorde a des exports explicites pour rester auditable.
"""
    )


def _appendix_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """source_paths = pd.DataFrame(
    [
        {"name": "regime_export_root", "path": str(REGIME_EXPORT_ROOT)},
        {"name": "stress_export_root", "path": str(STRESS_EXPORT_ROOT)},
        {"name": "dataset_path", "path": regime_metadata["dataset_path"]},
        {"name": "summary_variants", "path": str(REGIME_EXPORT_ROOT / "summary_variants.csv")},
        {"name": "feature_ranking", "path": str(REGIME_EXPORT_ROOT / "feature_ranking.csv")},
        {"name": "conditional_bucket_analysis", "path": str(REGIME_EXPORT_ROOT / "conditional_bucket_analysis.csv")},
    ]
)
display(source_paths)
"""
    )


def build_notebook(regime_export_root: Path, stress_export_root: Path) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
    }
    notebook.cells = [
        _title_cell(),
        _imports_cell(),
        _parameter_cell(regime_export_root, stress_export_root),
        _load_data_cell(),
        _quick_read_markdown(),
        _quick_read_cell(),
        _parameters_markdown(),
        _parameters_cell(),
        _calibration_markdown(),
        _calibration_cell(),
        _stability_markdown(),
        _stability_cell(),
        _performance_markdown(),
        _performance_cell(),
        _equity_markdown(),
        _equity_cell(),
        _stress_markdown(),
        _stress_cell(),
        _sizing_markdown(),
        _sizing_cell(),
        _distribution_markdown(),
        _distribution_cell(),
        _conclusion_markdown(),
        _conclusion_cell(),
        _appendix_markdown(),
        _appendix_cell(),
    ]
    return notebook


def write_notebook(notebook: nbf.NotebookNode, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(nbf.writes(notebook), encoding="utf-8")
    return output_path


def execute_notebook(input_path: Path, output_path: Path, timeout_seconds: int = 600) -> Path:
    notebook = nbf.read(input_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout_seconds,
        kernel_name="python3",
        resources={"metadata": {"path": str(input_path.parent)}},
    )
    client.execute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(nbf.writes(notebook), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regime-export-root",
        type=Path,
        default=find_latest_export("mnq_orb_regime_filter_sizing"),
        help="Audited regime/sizing export root to load.",
    )
    parser.add_argument(
        "--stress-export-root",
        type=Path,
        default=find_latest_export("mnq_orb_3state_high_bucket_stress", exports_root=REPORT_EXPORTS_ROOT),
        help="High-bucket stress campaign export root to load.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_NOTEBOOK_PATH,
        help="Notebook output path.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the notebook after generation and save the executed version.",
    )
    parser.add_argument(
        "--executed-output",
        type=Path,
        default=DEFAULT_EXECUTED_NOTEBOOK_PATH,
        help="Executed notebook output path.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="Notebook execution timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebook = build_notebook(args.regime_export_root, args.stress_export_root)
    output_path = write_notebook(notebook, args.output)
    print(f"Notebook written to {output_path}")

    if args.execute:
        executed_path = execute_notebook(output_path, args.executed_output, timeout_seconds=args.timeout_seconds)
        print(f"Executed notebook written to {executed_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
