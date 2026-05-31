"""Build a comparison notebook for MNQ ORB nominal vs retained 3-state overlay."""

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
NOTEBOOKS_ROOT = REPO_ROOT / "notebooks"

DEFAULT_VARIANT_NAME = "sizing_3state_realized_vol_ratio_15_60"
DEFAULT_BASELINE_NAME = "nominal"
DEFAULT_HIGH_BUCKET_MULTIPLIER = 0.25
DEFAULT_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "orb_MNQ_nominal_vs_3state_overlay_comparison.ipynb"
DEFAULT_EXECUTED_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "orb_MNQ_nominal_vs_3state_overlay_comparison.executed.ipynb"


def find_latest_export(prefix: str) -> Path:
    candidates = [path for path in DATA_EXPORTS_ROOT.glob(f"{prefix}_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No export folder found for prefix {prefix!r} under {DATA_EXPORTS_ROOT}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _title_cell() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """# MNQ ORB - Comparatif propre `nominal` vs `sizing_3state` retenu

Ce notebook compare le baseline `nominal` au **3-state retenu**:

- `low = 0.50x`
- `mid = 1.00x`
- `high = 0.25x`

La comparaison reste saine:

- même signal ORB,
- mêmes entrées / sorties,
- même stop / target,
- même logique de coûts,
- seule l'exposition varie par bucket.
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
from src.analytics.mnq_orb_regime_filter_sizing_campaign import _scale_nominal_trades_by_multiplier

pd.set_option("display.max_columns", 300)
pd.set_option("display.width", 240)


def fmt_money(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.1f} USD"


def fmt_float(value, digits=3):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_curve_from_daily(daily, initial_balance):
    out = daily.copy()
    out["session_date"] = pd.to_datetime(out["session_date"], errors="coerce")
    out = out.sort_values("session_date").reset_index(drop=True)
    out["daily_pnl_usd"] = pd.to_numeric(out["daily_pnl_usd"], errors="coerce").fillna(0.0)
    out["equity"] = initial_balance + out["daily_pnl_usd"].cumsum()
    out["peak_equity"] = out["equity"].cummax()
    out["drawdown_usd"] = out["equity"] - out["peak_equity"]
    out["drawdown_pct"] = (out["equity"] / out["peak_equity"] - 1.0) * 100.0
    return out


def rebase_oos_curve(curve_df, start_date, initial_balance):
    curve = curve_df.loc[curve_df["session_date"] >= pd.Timestamp(start_date)].copy()
    curve = curve.sort_values("session_date").reset_index(drop=True)
    curve["equity"] = initial_balance + curve["daily_pnl_usd"].cumsum()
    curve["peak_equity"] = curve["equity"].cummax()
    curve["drawdown_usd"] = curve["equity"] - curve["peak_equity"]
    curve["drawdown_pct"] = (curve["equity"] / curve["peak_equity"] - 1.0) * 100.0
    return curve


def scope_metrics(trades, sessions, initial_capital):
    metrics = compute_metrics(trades, session_dates=sessions, initial_capital=initial_capital)
    return {
        "net_pnl": float(metrics.get("cumulative_pnl", 0.0)),
        "sharpe": float(metrics.get("sharpe_ratio", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "n_trades": int(metrics.get("n_trades", 0)),
        "pct_days_traded": float(metrics.get("percent_of_days_traded", 0.0)),
        "worst_day": float(metrics.get("worst_day", 0.0)),
        "win_rate": float(metrics.get("win_rate", 0.0)),
    }
"""
    )


def _parameter_cell(regime_export_root: Path) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        f"""REGIME_EXPORT_ROOT = ROOT / r"{regime_export_root.relative_to(REPO_ROOT)}"
VARIANT_NAME = "{DEFAULT_VARIANT_NAME}"
BASELINE_NAME = "{DEFAULT_BASELINE_NAME}"
HIGH_BUCKET_MULTIPLIER = {DEFAULT_HIGH_BUCKET_MULTIPLIER}
LOW_BUCKET_MULTIPLIER = 0.50
MID_BUCKET_MULTIPLIER = 1.00
INITIAL_BALANCE_USD = 50_000.0

required_paths = {{
    "regime_export_root": REGIME_EXPORT_ROOT,
    "summary_variants": REGIME_EXPORT_ROOT / "summary_variants.csv",
    "baseline_metrics": REGIME_EXPORT_ROOT / "variants" / BASELINE_NAME / "metrics_by_scope.csv",
    "baseline_daily": REGIME_EXPORT_ROOT / "variants" / BASELINE_NAME / "daily_results.csv",
    "baseline_trades": REGIME_EXPORT_ROOT / "variants" / BASELINE_NAME / "trades.csv",
    "variant_controls": REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "controls.csv",
    "regime_mapping": REGIME_EXPORT_ROOT / "regime_state_mappings.csv",
}}

missing = [name for name, path in required_paths.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"Fichiers manquants pour le notebook: {{missing}}")

print("REGIME_EXPORT_ROOT =", REGIME_EXPORT_ROOT)
print("BASELINE_NAME      =", BASELINE_NAME)
print("VARIANT_NAME       =", VARIANT_NAME)
print("HIGH_MULTIPLIER    =", HIGH_BUCKET_MULTIPLIER)
"""
    )


def _load_data_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """regime_metadata = json.loads((REGIME_EXPORT_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
summary_variants = pd.read_csv(REGIME_EXPORT_ROOT / "summary_variants.csv")
baseline_metrics = pd.read_csv(REGIME_EXPORT_ROOT / "variants" / BASELINE_NAME / "metrics_by_scope.csv")
baseline_daily = pd.read_csv(REGIME_EXPORT_ROOT / "variants" / BASELINE_NAME / "daily_results.csv", parse_dates=["session_date"])
baseline_trades = pd.read_csv(
    REGIME_EXPORT_ROOT / "variants" / BASELINE_NAME / "trades.csv",
    parse_dates=["session_date", "entry_time", "exit_time"],
)
variant_controls = pd.read_csv(
    REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "controls.csv",
    parse_dates=["session_date"],
)
regime_mapping = pd.read_csv(REGIME_EXPORT_ROOT / "regime_state_mappings.csv")

baseline_trades["session_date"] = pd.to_datetime(baseline_trades["session_date"], errors="coerce")
baseline_trades["entry_time"] = pd.to_datetime(baseline_trades["entry_time"], errors="coerce", utc=True)
baseline_trades["exit_time"] = pd.to_datetime(baseline_trades["exit_time"], errors="coerce", utc=True)
variant_controls["session_date"] = pd.to_datetime(variant_controls["session_date"], errors="coerce")

baseline_row = summary_variants.loc[summary_variants["variant_name"] == BASELINE_NAME].iloc[0]
baseline_config = regime_metadata["spec"]["baseline"]
initial_capital = float(baseline_config["account_size_usd"])
base_risk_pct = float(baseline_config["risk_per_trade_pct"])
all_sessions = pd.to_datetime(baseline_daily["session_date"], errors="coerce").dt.date.tolist()
is_sessions = pd.to_datetime(variant_controls.loc[variant_controls["phase"] == "is", "session_date"], errors="coerce").dt.date.tolist()
oos_sessions = pd.to_datetime(variant_controls.loc[variant_controls["phase"] == "oos", "session_date"], errors="coerce").dt.date.tolist()
oos_start_date = pd.to_datetime(variant_controls.loc[variant_controls["phase"] == "oos", "session_date"].min())

bucket_map = (
    regime_mapping.loc[
        (regime_mapping["variant_name"] == VARIANT_NAME)
        & (regime_mapping["feature_name"] == "realized_vol_ratio_15_60"),
        ["bucket_label", "bucket_position", "lower_bound", "upper_bound", "risk_multiplier", "is_composite_score", "oos_n_obs", "oos_net_pnl", "oos_sharpe", "oos_max_drawdown"],
    ]
    .drop_duplicates()
    .sort_values("bucket_position")
    .reset_index(drop=True)
)

variant_controls["bucket_label"] = variant_controls["bucket_label"].astype(str)
variant_controls["risk_multiplier"] = variant_controls["bucket_label"].map(
    {
        "low": LOW_BUCKET_MULTIPLIER,
        "mid": MID_BUCKET_MULTIPLIER,
        "high": HIGH_BUCKET_MULTIPLIER,
    }
).fillna(0.0)
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

variant_metrics = pd.DataFrame(
    [
        {"scope": "overall", **scope_metrics(variant_trades, all_sessions, initial_capital)},
        {"scope": "is", **scope_metrics(variant_trades.loc[pd.to_datetime(variant_trades["session_date"], errors="coerce").dt.date.isin(set(is_sessions))].copy(), is_sessions, initial_capital)},
        {"scope": "oos", **scope_metrics(variant_trades.loc[pd.to_datetime(variant_trades["session_date"], errors="coerce").dt.date.isin(set(oos_sessions))].copy(), oos_sessions, initial_capital)},
    ]
)

variant_row = pd.Series(
    {
        "variant_name": f"sizing_3state_high_{str(HIGH_BUCKET_MULTIPLIER).replace('.', 'p')}",
        "overall_net_pnl": float(variant_metrics.loc[variant_metrics["scope"] == "overall", "net_pnl"].iloc[0]),
        "overall_sharpe": float(variant_metrics.loc[variant_metrics["scope"] == "overall", "sharpe"].iloc[0]),
        "overall_profit_factor": float(variant_metrics.loc[variant_metrics["scope"] == "overall", "profit_factor"].iloc[0]),
        "overall_max_drawdown": float(variant_metrics.loc[variant_metrics["scope"] == "overall", "max_drawdown"].iloc[0]),
        "oos_net_pnl": float(variant_metrics.loc[variant_metrics["scope"] == "oos", "net_pnl"].iloc[0]),
        "oos_sharpe": float(variant_metrics.loc[variant_metrics["scope"] == "oos", "sharpe"].iloc[0]),
        "oos_profit_factor": float(variant_metrics.loc[variant_metrics["scope"] == "oos", "profit_factor"].iloc[0]),
        "oos_max_drawdown": float(variant_metrics.loc[variant_metrics["scope"] == "oos", "max_drawdown"].iloc[0]),
    }
)
variant_row["oos_net_pnl_retention_vs_nominal"] = variant_row["oos_net_pnl"] / float(baseline_row["oos_net_pnl"]) if float(baseline_row["oos_net_pnl"]) != 0 else np.nan
variant_row["oos_sharpe_delta_vs_nominal"] = variant_row["oos_sharpe"] - float(baseline_row["oos_sharpe"])
variant_row["oos_max_drawdown_improvement_vs_nominal"] = abs(float(baseline_row["oos_max_drawdown"])) - abs(float(variant_row["oos_max_drawdown"]))

bucket_map["effective_risk_per_trade_pct"] = bucket_map["bucket_label"].map(
    {"low": LOW_BUCKET_MULTIPLIER, "mid": MID_BUCKET_MULTIPLIER, "high": HIGH_BUCKET_MULTIPLIER}
) * base_risk_pct

baseline_curve = build_curve_from_daily(baseline_daily, INITIAL_BALANCE_USD)
variant_curve = build_curve_from_daily(variant_daily, INITIAL_BALANCE_USD)
baseline_curve_oos = rebase_oos_curve(baseline_daily, oos_start_date, INITIAL_BALANCE_USD)
variant_curve_oos = rebase_oos_curve(variant_daily, oos_start_date, INITIAL_BALANCE_USD)

baseline_trades["trade_key"] = baseline_trades["entry_time"].dt.strftime("%Y-%m-%d %H:%M:%S%z") + "|" + baseline_trades["direction"].astype(str)
variant_trades["trade_key"] = variant_trades["entry_time"].dt.strftime("%Y-%m-%d %H:%M:%S%z") + "|" + variant_trades["direction"].astype(str)

trade_comparison = (
    baseline_trades.rename(
        columns={
            "quantity": "quantity_nominal",
            "net_pnl_usd": "net_pnl_nominal",
            "risk_per_trade_pct": "risk_pct_nominal",
            "actual_risk_usd": "actual_risk_nominal",
            "fees": "fees_nominal",
            "exit_reason": "exit_reason_nominal",
            "exit_time": "exit_time_nominal",
            "stop_price": "stop_price_nominal",
            "target_price": "target_price_nominal",
        }
    )
    .merge(
        variant_trades.rename(
            columns={
                "quantity": "quantity_3state",
                "net_pnl_usd": "net_pnl_3state",
                "risk_per_trade_pct": "risk_pct_3state",
                "actual_risk_usd": "actual_risk_3state",
                "fees": "fees_3state",
                "exit_reason": "exit_reason_3state",
                "exit_time": "exit_time_3state",
                "stop_price": "stop_price_3state",
                "target_price": "target_price_3state",
            }
        )[
            [
                "trade_key",
                "quantity_3state",
                "net_pnl_3state",
                "risk_pct_3state",
                "actual_risk_3state",
                "fees_3state",
                "risk_multiplier",
                "exit_reason_3state",
                "exit_time_3state",
                "stop_price_3state",
                "target_price_3state",
            ]
        ],
        on="trade_key",
        how="outer",
    )
    .sort_values("entry_time")
    .reset_index(drop=True)
)

trade_comparison["same_trade"] = trade_comparison["quantity_nominal"].notna() & trade_comparison["quantity_3state"].notna()
trade_comparison["same_exit_time"] = trade_comparison["exit_time_nominal"].eq(trade_comparison["exit_time_3state"])
trade_comparison["same_exit_reason"] = trade_comparison["exit_reason_nominal"].eq(trade_comparison["exit_reason_3state"])
trade_comparison["same_stop_price"] = np.isclose(trade_comparison["stop_price_nominal"], trade_comparison["stop_price_3state"], equal_nan=True)
trade_comparison["same_target_price"] = np.isclose(trade_comparison["target_price_nominal"], trade_comparison["target_price_3state"], equal_nan=True)
trade_comparison["size_ratio_3state_vs_nominal"] = pd.to_numeric(trade_comparison["quantity_3state"], errors="coerce") / pd.to_numeric(trade_comparison["quantity_nominal"], errors="coerce")
trade_comparison["pnl_ratio_3state_vs_nominal"] = pd.to_numeric(trade_comparison["net_pnl_3state"], errors="coerce") / pd.to_numeric(trade_comparison["net_pnl_nominal"], errors="coerce")
trade_comparison["abs_pnl_ratio_3state_vs_nominal"] = pd.to_numeric(trade_comparison["net_pnl_3state"], errors="coerce").abs() / pd.to_numeric(trade_comparison["net_pnl_nominal"], errors="coerce").abs()

same_trade_rows = trade_comparison.loc[trade_comparison["same_trade"]].copy()
display(Markdown(f"**OOS start date:** `{oos_start_date.date()}`"))
"""
    )


def _summary_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 1. Lecture Rapide

Cette version correspond au 3-state retenu:

- `low = 0.50x`
- `mid = 1.00x`
- `high = 0.25x`
"""
    )


def _summary_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """quick_lines = [
    "### Synthese executive",
    f"- Le nominal OOS fait `{fmt_money(baseline_row['oos_net_pnl'])}` avec Sharpe `{fmt_float(baseline_row['oos_sharpe'])}` et maxDD `{fmt_money(baseline_row['oos_max_drawdown'])}`.",
    f"- Le 3-state retenu OOS fait `{fmt_money(variant_row['oos_net_pnl'])}` avec Sharpe `{fmt_float(variant_row['oos_sharpe'])}` et maxDD `{fmt_money(variant_row['oos_max_drawdown'])}`.",
    f"- La variante retenue conserve **{variant_row['oos_net_pnl_retention_vs_nominal'] * 100.0:.1f}%** du pnl OOS du nominal.",
    f"- Le bucket `high` est maintenant coupé à **{HIGH_BUCKET_MULTIPLIER:.2f}x**.",
]
display(Markdown("\\n".join(quick_lines)))
"""
    )


def _integrity_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 2. Intégrité de la Comparaison

On vérifie ici qu’on compare bien le même trade set de base, avec seulement un changement d’exposition.
"""
    )


def _integrity_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """integrity_table = pd.DataFrame(
    [
        {"check": "baseline trade count", "value": int(len(baseline_trades))},
        {"check": "retained 3state trade count", "value": int(len(variant_trades))},
        {"check": "matched trades", "value": int(same_trade_rows.shape[0])},
        {"check": "baseline-only rows", "value": int(trade_comparison["quantity_nominal"].notna().sum() - same_trade_rows.shape[0])},
        {"check": "3state-only rows", "value": int(trade_comparison["quantity_3state"].notna().sum() - same_trade_rows.shape[0])},
        {"check": "same exit time", "value": int(same_trade_rows["same_exit_time"].sum())},
        {"check": "same exit reason", "value": int(same_trade_rows["same_exit_reason"].sum())},
        {"check": "same stop price", "value": int(same_trade_rows["same_stop_price"].sum())},
        {"check": "same target price", "value": int(same_trade_rows["same_target_price"].sum())},
    ]
)
display(integrity_table)
"""
    )


def _parameters_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 3. Paramètres Exacts

On remet côte à côte le baseline nominal et le mapping effectif du 3-state retenu.
"""
    )


def _parameters_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """baseline_df = pd.DataFrame([{"parameter": key, "value": value} for key, value in baseline_config.items()])

bucket_display = bucket_map.copy()
for col in ["lower_bound", "upper_bound", "is_composite_score", "effective_risk_per_trade_pct", "oos_net_pnl", "oos_sharpe", "oos_max_drawdown"]:
    if col in bucket_display.columns:
        bucket_display[col] = pd.to_numeric(bucket_display[col], errors="coerce").round(3)

display(Markdown("### Baseline nominal"))
display(baseline_df)
display(Markdown("### Bucket mapping retenu"))
display(bucket_display)
"""
    )


def _metrics_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 4. Tableau Comparatif

On compare ici `nominal` vs `3-state retenu`.
"""
    )


def _metrics_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """comparison_metrics = pd.DataFrame(
    [
        {
            "variant_name": BASELINE_NAME,
            "overall_net_pnl": baseline_row["overall_net_pnl"],
            "overall_sharpe": baseline_row["overall_sharpe"],
            "overall_profit_factor": baseline_row["overall_profit_factor"],
            "overall_max_drawdown": baseline_row["overall_max_drawdown"],
            "oos_net_pnl": baseline_row["oos_net_pnl"],
            "oos_sharpe": baseline_row["oos_sharpe"],
            "oos_profit_factor": baseline_row["oos_profit_factor"],
            "oos_max_drawdown": baseline_row["oos_max_drawdown"],
        },
        {
            "variant_name": variant_row["variant_name"],
            "overall_net_pnl": variant_row["overall_net_pnl"],
            "overall_sharpe": variant_row["overall_sharpe"],
            "overall_profit_factor": variant_row["overall_profit_factor"],
            "overall_max_drawdown": variant_row["overall_max_drawdown"],
            "oos_net_pnl": variant_row["oos_net_pnl"],
            "oos_sharpe": variant_row["oos_sharpe"],
            "oos_profit_factor": variant_row["oos_profit_factor"],
            "oos_max_drawdown": variant_row["oos_max_drawdown"],
        },
    ]
)
display(comparison_metrics)
display(variant_metrics)
"""
    )


def _equity_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 5. Courbes de Capital

On regarde les courbes full sample et OOS only pour la version retenue.
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

for curve_name, curve_df, color in [
    ("Nominal", baseline_curve, "#2563eb"),
    ("3-state retained", variant_curve, "#16a34a"),
]:
    fig.add_trace(go.Scatter(x=curve_df["session_date"], y=curve_df["equity"], mode="lines", name=f"{curve_name} full", line=dict(width=2.5, color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=curve_df["session_date"], y=curve_df["drawdown_usd"], mode="lines", name=f"{curve_name} full DD", showlegend=False, line=dict(width=1.8, color=color, dash="dot")), row=2, col=1)

for curve_name, curve_df, color in [
    ("Nominal", baseline_curve_oos, "#2563eb"),
    ("3-state retained", variant_curve_oos, "#16a34a"),
]:
    fig.add_trace(go.Scatter(x=curve_df["session_date"], y=curve_df["equity"], mode="lines", name=f"{curve_name} oos", line=dict(width=2.5, color=color)), row=1, col=2)
    fig.add_trace(go.Scatter(x=curve_df["session_date"], y=curve_df["drawdown_usd"], mode="lines", name=f"{curve_name} oos DD", showlegend=False, line=dict(width=1.8, color=color, dash="dot")), row=2, col=2)

fig.update_layout(height=820, width=1200, title="MNQ ORB comparison - nominal vs retained sizing_3state (high=0.25)", legend=dict(orientation="h", y=1.08))
fig.show()
"""
    )


def _trade_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 6. Table Trade par Trade

Cette table montre directement ce que change l’overlay retenu sur les trades matchés.
"""
    )


def _trade_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """trade_table = same_trade_rows[
    [
        "session_date",
        "entry_time",
        "direction",
        "quantity_nominal",
        "quantity_3state",
        "risk_multiplier",
        "risk_pct_nominal",
        "risk_pct_3state",
        "net_pnl_nominal",
        "net_pnl_3state",
        "size_ratio_3state_vs_nominal",
        "abs_pnl_ratio_3state_vs_nominal",
    ]
].copy()

for col in ["risk_multiplier", "risk_pct_nominal", "risk_pct_3state", "net_pnl_nominal", "net_pnl_3state", "size_ratio_3state_vs_nominal", "abs_pnl_ratio_3state_vs_nominal"]:
    trade_table[col] = pd.to_numeric(trade_table[col], errors="coerce").round(3)

display(trade_table.head(40))
"""
    )


def _distribution_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 7. Distribution et Buckets

On regarde comment la version retenue redistribue l’exposition par bucket.
"""
    )


def _distribution_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """trade_buckets = variant_trades.copy()
if "bucket_label" not in trade_buckets.columns:
    trade_buckets = trade_buckets.merge(
        variant_controls[["session_date", "phase", "bucket_label"]],
        on="session_date",
        how="left",
    )

bucket_trade_stats = (
    trade_buckets.groupby(["bucket_label", "risk_multiplier"], dropna=False)
    .agg(
        n_trades=("trade_id", "count"),
        avg_quantity=("quantity", "mean"),
        avg_risk_pct=("risk_per_trade_pct", "mean"),
        total_net_pnl_usd=("net_pnl_usd", "sum"),
        avg_net_pnl_usd=("net_pnl_usd", "mean"),
    )
    .reset_index()
    .sort_values(["risk_multiplier", "bucket_label"])
)
for col in ["avg_quantity", "avg_risk_pct", "total_net_pnl_usd", "avg_net_pnl_usd"]:
    bucket_trade_stats[col] = pd.to_numeric(bucket_trade_stats[col], errors="coerce").round(2)
display(bucket_trade_stats)

fig = px.bar(
    bucket_trade_stats,
    x="bucket_label",
    y="total_net_pnl_usd",
    color="risk_multiplier",
    barmode="group",
    title="Contribution PnL par bucket - version retenue",
)
fig.show()
"""
    )


def _conclusion_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 8. Conclusion

Le notebook reflète maintenant la version retenue `high=0.25`.
"""
    )


def _conclusion_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """final_lines = [
    "### Verdict simple",
    f"- Le comparatif affiche maintenant la version retenue: **high = {HIGH_BUCKET_MULTIPLIER:.2f}x**.",
    f"- Le nominal garde la pleine exposition: OOS net `{fmt_money(baseline_row['oos_net_pnl'])}`, Sharpe `{fmt_float(baseline_row['oos_sharpe'])}`.",
    f"- La version retenue réduit plus agressivement le bucket `high`: OOS net `{fmt_money(variant_row['oos_net_pnl'])}`, Sharpe `{fmt_float(variant_row['oos_sharpe'])}`.",
    f"- La lecture correcte reste: **même alpha ORB, exposition plus conservative et plus live-oriented**.",
]
display(Markdown("\\n".join(final_lines)))
"""
    )


def build_notebook(regime_export_root: Path) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"},
    }
    notebook.cells = [
        _title_cell(),
        _imports_cell(),
        _parameter_cell(regime_export_root),
        _load_data_cell(),
        _summary_markdown(),
        _summary_cell(),
        _integrity_markdown(),
        _integrity_cell(),
        _parameters_markdown(),
        _parameters_cell(),
        _metrics_markdown(),
        _metrics_cell(),
        _equity_markdown(),
        _equity_cell(),
        _trade_markdown(),
        _trade_cell(),
        _distribution_markdown(),
        _distribution_cell(),
        _conclusion_markdown(),
        _conclusion_cell(),
    ]
    return notebook


def write_notebook(path: Path, notebook: nbf.NotebookNode) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)


def execute_notebook(notebook: nbf.NotebookNode) -> nbf.NotebookNode:
    client = NotebookClient(notebook, timeout=1200, kernel_name="python3")
    client.execute()
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MNQ nominal vs retained sizing_3state overlay comparison notebook.")
    parser.add_argument("--notebook-path", type=Path, default=DEFAULT_NOTEBOOK_PATH)
    parser.add_argument("--executed-path", type=Path, default=DEFAULT_EXECUTED_NOTEBOOK_PATH)
    args = parser.parse_args()

    regime_export_root = find_latest_export("mnq_orb_regime_filter_sizing")
    notebook = build_notebook(regime_export_root)
    write_notebook(args.notebook_path, notebook)

    executed = execute_notebook(nbf.from_dict(notebook))
    write_notebook(args.executed_path, executed)

    print(f"Notebook written to {args.notebook_path}")
    print(f"Executed notebook written to {args.executed_path}")


if __name__ == "__main__":
    main()
