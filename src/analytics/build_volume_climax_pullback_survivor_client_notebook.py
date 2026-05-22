"""Build a single client notebook for the latest volume climax pullback survivors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytics.volume_climax_pullback_common import load_latest_reference_run

EXPORTS_ROOT = REPO_ROOT / "export"
NOTEBOOKS_ROOT = REPO_ROOT / "notebooks"

SURVIVOR_PREFIX = "volume_climax_pullback_survivor_audit_"
REGIME_PREFIX = "volume_climax_pullback_regime_gated_portfolio_"
INTEGRATION_PREFIX = "volume_climax_pullback_portfolio_integration_"

DEFAULT_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "volume_climax_pullback_survivors_client.ipynb"
DEFAULT_EXECUTED_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "volume_climax_pullback_survivors_client.executed.ipynb"


def find_latest_export(prefix: str) -> Path:
    return load_latest_reference_run(EXPORTS_ROOT, prefix)


def _title_cell() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """# Volume Climax Pullback Survivors - Client Notebook

Ce notebook client regroupe en une seule vue les deux signaux pullback encore utiles apres les campagnes strictes :

- `M2K 1H` : survivor faible mais propre, verdict `weak_watchlist`
- `MGC 1H` : diversifiant opportuniste, mais instable en standalone, verdict `reject`

Le notebook recharge trois briques deja auditees :

1. le **survivor audit** pour les signaux stricts `M2K/MGC` ;
2. le **regime-gated audit** pour montrer pourquoi le gating MGC n'a pas ete retenu ;
3. l'**integration portefeuille** pour voir si le sleeve fixe `M2K + MGC` apporte quelque chose au book prop.

Objectif :

- garder un seul support client,
- montrer les verdicts stricts sans cherry-picking,
- rendre lisibles les deux signaux, le sleeve combine, et l'impact portefeuille.
"""
    )


def _imports_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """import json
import math
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

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 240)


def fmt_money(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.1f} USD"


def fmt_pct(value, digits=1):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}%"


def fmt_float(value, digits=3):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def safe_corr(left, right):
    left = pd.to_numeric(left, errors="coerce")
    right = pd.to_numeric(right, errors="coerce")
    if left.count() < 2 or right.count() < 2:
        return float("nan")
    return float(left.corr(right))


def build_equity(frame, pnl_col):
    out = frame.copy()
    out[pnl_col] = pd.to_numeric(out[pnl_col], errors="coerce").fillna(0.0)
    out["equity"] = out[pnl_col].cumsum()
    out["peak"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] - out["peak"]
    return out
"""
    )


def _parameter_cell(survivor_root: Path, regime_root: Path, integration_root: Path) -> nbf.NotebookNode:
    survivor_root = survivor_root if survivor_root.is_absolute() else (REPO_ROOT / survivor_root)
    regime_root = regime_root if regime_root.is_absolute() else (REPO_ROOT / regime_root)
    integration_root = integration_root if integration_root.is_absolute() else (REPO_ROOT / integration_root)
    survivor_root = survivor_root.resolve()
    regime_root = regime_root.resolve()
    integration_root = integration_root.resolve()
    return nbf.v4.new_code_cell(
        f"""SURVIVOR_EXPORT_ROOT = ROOT / r"{survivor_root.relative_to(REPO_ROOT)}"
REGIME_EXPORT_ROOT = ROOT / r"{regime_root.relative_to(REPO_ROOT)}"
INTEGRATION_EXPORT_ROOT = ROOT / r"{integration_root.relative_to(REPO_ROOT)}"

PRIMARY_SYMBOLS = ["M2K", "MGC"]
NEGATIVE_CONTROL = "MNQ"
SIGNAL_TIMEFRAME = "1H"

STRICT_SIGNAL_LABELS = {{
    "M2K": "M2K 1H survivor",
    "MGC": "MGC 1H opportunistic diversifier",
    "MNQ": "MNQ 1H negative control",
}}

STRICT_SLEEVE_NAME = "m2k_mgc_equal_weight"
STRICT_SLEEVE_FALLBACK = "m2k_mgc_capped_equal"
M2K_ONLY_PORTFOLIO = "m2k_only"
MGC_ONLY_PORTFOLIO = "mgc_only"
BEST_REGIME_PORTFOLIO = "strict_best_regime_gated"

BASELINE_PORTFOLIO = "baseline_only"
PULLBACK_PORTFOLIO = "pullback_m2k_mgc_only"
INTEGRATED_PORTFOLIO = "baseline_plus_pullback_equal_notional"
INTEGRATED_M2K_ONLY = "baseline_plus_m2k_only_equal_notional"

PLOT_TEMPLATE = "plotly_white"

required_paths = {{
    "survivor_summary": SURVIVOR_EXPORT_ROOT / "strict_wfa_summary.csv",
    "survivor_fold_breakdown": SURVIVOR_EXPORT_ROOT / "strict_wfa_fold_breakdown.csv",
    "survivor_portfolio_summary": SURVIVOR_EXPORT_ROOT / "strict_portfolio_summary.csv",
    "survivor_portfolio_daily": SURVIVOR_EXPORT_ROOT / "strict_portfolio_daily_returns.csv",
    "survivor_selection": SURVIVOR_EXPORT_ROOT / "config_selection_by_fold.csv",
    "survivor_cluster_stability": SURVIVOR_EXPORT_ROOT / "cluster_stability_summary.csv",
    "survivor_local_stability": SURVIVOR_EXPORT_ROOT / "local_parameter_stability.csv",
    "survivor_monthly": SURVIVOR_EXPORT_ROOT / "monthly_pnl.csv",
    "survivor_yearly": SURVIVOR_EXPORT_ROOT / "yearly_pnl.csv",
    "survivor_trade_concentration": SURVIVOR_EXPORT_ROOT / "trade_concentration.csv",
    "regime_summary": REGIME_EXPORT_ROOT / "strict_regime_wfa_summary.csv",
    "regime_rules": REGIME_EXPORT_ROOT / "selected_regime_rule_by_fold.csv",
    "regime_retention": REGIME_EXPORT_ROOT / "mgc_regime_retention_summary.csv",
    "integration_summary": INTEGRATION_EXPORT_ROOT / "portfolio_summary.csv",
    "integration_daily": INTEGRATION_EXPORT_ROOT / "daily_pnl_aligned.csv",
    "integration_corr": INTEGRATION_EXPORT_ROOT / "portfolio_correlation.csv",
    "integration_incremental": INTEGRATION_EXPORT_ROOT / "incremental_metrics.csv",
    "integration_bootstrap": INTEGRATION_EXPORT_ROOT / "bootstrap_summary.csv",
    "integration_prop": INTEGRATION_EXPORT_ROOT / "prop_constraint_summary.csv",
}}

missing = [name for name, path in required_paths.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"Fichiers manquants pour le notebook: {{missing}}")

display(Markdown("### Parametrage client"))
display(pd.DataFrame(
    [
        {{"parameter": "SURVIVOR_EXPORT_ROOT", "value": str(SURVIVOR_EXPORT_ROOT)}},
        {{"parameter": "REGIME_EXPORT_ROOT", "value": str(REGIME_EXPORT_ROOT)}},
        {{"parameter": "INTEGRATION_EXPORT_ROOT", "value": str(INTEGRATION_EXPORT_ROOT)}},
        {{"parameter": "PRIMARY_SYMBOLS", "value": ", ".join(PRIMARY_SYMBOLS)}},
        {{"parameter": "NEGATIVE_CONTROL", "value": NEGATIVE_CONTROL}},
        {{"parameter": "SIGNAL_TIMEFRAME", "value": SIGNAL_TIMEFRAME}},
        {{"parameter": "STRICT_SLEEVE_NAME", "value": STRICT_SLEEVE_NAME}},
        {{"parameter": "BASELINE_PORTFOLIO", "value": BASELINE_PORTFOLIO}},
        {{"parameter": "INTEGRATED_PORTFOLIO", "value": INTEGRATED_PORTFOLIO}},
        {{"parameter": "PLOT_TEMPLATE", "value": PLOT_TEMPLATE}},
    ]
))
"""
    )


def _load_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """strict_wfa_summary = pd.read_csv(SURVIVOR_EXPORT_ROOT / "strict_wfa_summary.csv")
strict_wfa_fold_breakdown = pd.read_csv(SURVIVOR_EXPORT_ROOT / "strict_wfa_fold_breakdown.csv")
strict_portfolio_summary = pd.read_csv(SURVIVOR_EXPORT_ROOT / "strict_portfolio_summary.csv")
strict_portfolio_daily = pd.read_csv(
    SURVIVOR_EXPORT_ROOT / "strict_portfolio_daily_returns.csv",
    parse_dates=["session_date"],
)
config_selection_by_fold = pd.read_csv(SURVIVOR_EXPORT_ROOT / "config_selection_by_fold.csv")
cluster_stability_summary = pd.read_csv(SURVIVOR_EXPORT_ROOT / "cluster_stability_summary.csv")
local_parameter_stability = pd.read_csv(SURVIVOR_EXPORT_ROOT / "local_parameter_stability.csv")
survivor_monthly_pnl = pd.read_csv(SURVIVOR_EXPORT_ROOT / "monthly_pnl.csv")
survivor_yearly_pnl = pd.read_csv(SURVIVOR_EXPORT_ROOT / "yearly_pnl.csv")
survivor_trade_concentration = pd.read_csv(SURVIVOR_EXPORT_ROOT / "trade_concentration.csv")

strict_regime_wfa_summary = pd.read_csv(REGIME_EXPORT_ROOT / "strict_regime_wfa_summary.csv")
selected_regime_rule_by_fold = pd.read_csv(REGIME_EXPORT_ROOT / "selected_regime_rule_by_fold.csv")
mgc_regime_retention_summary = pd.read_csv(REGIME_EXPORT_ROOT / "mgc_regime_retention_summary.csv")

portfolio_summary = pd.read_csv(INTEGRATION_EXPORT_ROOT / "portfolio_summary.csv")
daily_pnl_aligned = pd.read_csv(INTEGRATION_EXPORT_ROOT / "daily_pnl_aligned.csv", parse_dates=["session_date"])
portfolio_correlation = pd.read_csv(INTEGRATION_EXPORT_ROOT / "portfolio_correlation.csv")
incremental_metrics = pd.read_csv(INTEGRATION_EXPORT_ROOT / "incremental_metrics.csv")
bootstrap_summary = pd.read_csv(INTEGRATION_EXPORT_ROOT / "bootstrap_summary.csv")
prop_constraint_summary = pd.read_csv(INTEGRATION_EXPORT_ROOT / "prop_constraint_summary.csv")

strict_signal_view = strict_wfa_summary.loc[
    (strict_wfa_summary["signal_timeframe"].astype(str) == SIGNAL_TIMEFRAME)
    & (strict_wfa_summary["symbol"].astype(str).isin(PRIMARY_SYMBOLS + [NEGATIVE_CONTROL]))
].copy()
strict_signal_view["display_name"] = strict_signal_view["symbol"].astype(str).map(STRICT_SIGNAL_LABELS)

strict_fold_view = strict_wfa_fold_breakdown.loc[
    (strict_wfa_fold_breakdown["signal_timeframe"].astype(str) == SIGNAL_TIMEFRAME)
    & (strict_wfa_fold_breakdown["symbol"].astype(str).isin(PRIMARY_SYMBOLS + [NEGATIVE_CONTROL]))
].copy()
strict_fold_view["display_name"] = strict_fold_view["symbol"].astype(str).map(STRICT_SIGNAL_LABELS)

selection_view = config_selection_by_fold.loc[
    config_selection_by_fold["symbol"].astype(str).isin(PRIMARY_SYMBOLS + [NEGATIVE_CONTROL])
].copy()

cluster_view = cluster_stability_summary.loc[
    cluster_stability_summary["symbol"].astype(str).isin(PRIMARY_SYMBOLS + [NEGATIVE_CONTROL])
].copy()

local_view = local_parameter_stability.loc[
    local_parameter_stability["symbol"].astype(str).isin(PRIMARY_SYMBOLS + [NEGATIVE_CONTROL])
].copy()

strict_portfolios = strict_portfolio_summary.copy()
strict_portfolio_daily["session_date"] = pd.to_datetime(strict_portfolio_daily["session_date"], errors="coerce").dt.normalize()
daily_pnl_aligned["session_date"] = pd.to_datetime(daily_pnl_aligned["session_date"], errors="coerce").dt.normalize()

regime_entity_col = "entity_id" if "entity_id" in strict_regime_wfa_summary.columns else "entity_name"

strict_sleeve_row = strict_portfolios.loc[strict_portfolios["portfolio_name"].astype(str) == STRICT_SLEEVE_NAME]
if strict_sleeve_row.empty:
    strict_sleeve_row = strict_portfolios.loc[strict_portfolios["portfolio_name"].astype(str) == STRICT_SLEEVE_FALLBACK]
if strict_sleeve_row.empty:
    raise ValueError("Impossible de retrouver le sleeve strict M2K+MGC dans strict_portfolio_summary.csv")
strict_sleeve_row = strict_sleeve_row.iloc[0]

m2k_row = strict_portfolios.loc[strict_portfolios["portfolio_name"].astype(str) == M2K_ONLY_PORTFOLIO].iloc[0]
mgc_row = strict_portfolios.loc[strict_portfolios["portfolio_name"].astype(str) == MGC_ONLY_PORTFOLIO].iloc[0]
best_regime_row = strict_regime_wfa_summary.loc[
    strict_regime_wfa_summary[regime_entity_col].astype(str) == BEST_REGIME_PORTFOLIO
].iloc[0]

baseline_row = portfolio_summary.loc[
    (portfolio_summary["portfolio_name"].astype(str) == BASELINE_PORTFOLIO)
    & (portfolio_summary["scope"].astype(str) == "defined_oos")
].iloc[0]
pullback_row = portfolio_summary.loc[
    (portfolio_summary["portfolio_name"].astype(str) == PULLBACK_PORTFOLIO)
    & (portfolio_summary["scope"].astype(str) == "defined_oos")
].iloc[0]
integrated_row = portfolio_summary.loc[
    (portfolio_summary["portfolio_name"].astype(str) == INTEGRATED_PORTFOLIO)
    & (portfolio_summary["scope"].astype(str) == "defined_oos")
].iloc[0]
integrated_m2k_row = portfolio_summary.loc[
    (portfolio_summary["portfolio_name"].astype(str) == INTEGRATED_M2K_ONLY)
    & (portfolio_summary["scope"].astype(str) == "defined_oos")
].iloc[0]

incremental_equal = incremental_metrics.loc[incremental_metrics["portfolio_name"].astype(str) == INTEGRATED_PORTFOLIO].iloc[0]
bootstrap_equal = bootstrap_summary.loc[bootstrap_summary["portfolio_name"].astype(str) == INTEGRATED_PORTFOLIO].iloc[0]
bootstrap_baseline = bootstrap_summary.loc[bootstrap_summary["portfolio_name"].astype(str) == BASELINE_PORTFOLIO].iloc[0]
bootstrap_pullback = bootstrap_summary.loc[bootstrap_summary["portfolio_name"].astype(str) == PULLBACK_PORTFOLIO].iloc[0]
prop_equal = prop_constraint_summary.loc[prop_constraint_summary["portfolio_name"].astype(str) == INTEGRATED_PORTFOLIO].iloc[0]
prop_baseline = prop_constraint_summary.loc[prop_constraint_summary["portfolio_name"].astype(str) == BASELINE_PORTFOLIO].iloc[0]

correlation_equal = portfolio_correlation.loc[
    (portfolio_correlation["left_portfolio"].astype(str) == BASELINE_PORTFOLIO)
    & (portfolio_correlation["right_portfolio"].astype(str) == PULLBACK_PORTFOLIO)
].iloc[0]

strict_sleeve_daily = strict_portfolio_daily.loc[
    strict_portfolio_daily["portfolio_name"].astype(str).isin([STRICT_SLEEVE_NAME, STRICT_SLEEVE_FALLBACK])
].copy()
strict_sleeve_daily = strict_sleeve_daily.sort_values("session_date").drop_duplicates("session_date", keep="last")

integration_curve = daily_pnl_aligned.loc[daily_pnl_aligned["oos_mask"].fillna(False)].copy()
integration_curve["baseline_equity"] = pd.to_numeric(integration_curve["baseline_daily_pnl_usd"], errors="coerce").fillna(0.0).cumsum()
integration_curve["pullback_equity"] = pd.to_numeric(integration_curve["pullback_daily_pnl_usd"], errors="coerce").fillna(0.0).cumsum()
integration_curve["integrated_equity"] = (
    pd.to_numeric(integration_curve["baseline_daily_pnl_usd"], errors="coerce").fillna(0.0)
    + pd.to_numeric(integration_curve["pullback_daily_pnl_usd"], errors="coerce").fillna(0.0)
).cumsum()

display(Markdown(f"**Survivor export**: `{SURVIVOR_EXPORT_ROOT}`"))
display(Markdown(f"**Regime export**: `{REGIME_EXPORT_ROOT}`"))
display(Markdown(f"**Integration export**: `{INTEGRATION_EXPORT_ROOT}`"))
"""
    )


def _summary_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 1. Executive Summary"))

summary_lines = [
    f"- `M2K 1H` reste le seul survivor standalone non rejete : net strict WFA `{fmt_money(strict_signal_view.loc[strict_signal_view['symbol'] == 'M2K', 'total_test_net_pnl'].iloc[0])}` | PF `{fmt_float(strict_signal_view.loc[strict_signal_view['symbol'] == 'M2K', 'test_profit_factor'].iloc[0])}` | folds positifs `{int(strict_signal_view.loc[strict_signal_view['symbol'] == 'M2K', 'positive_folds'].iloc[0])}/{int(strict_signal_view.loc[strict_signal_view['symbol'] == 'M2K', 'fold_count'].iloc[0])}` | verdict `{strict_signal_view.loc[strict_signal_view['symbol'] == 'M2K', 'verdict'].iloc[0]}`.",
    f"- `MGC 1H` garde un net eleve mais trop concentre regime/folds : net `{fmt_money(strict_signal_view.loc[strict_signal_view['symbol'] == 'MGC', 'total_test_net_pnl'].iloc[0])}` | PF `{fmt_float(strict_signal_view.loc[strict_signal_view['symbol'] == 'MGC', 'test_profit_factor'].iloc[0])}` | folds positifs `{int(strict_signal_view.loc[strict_signal_view['symbol'] == 'MGC', 'positive_folds'].iloc[0])}/{int(strict_signal_view.loc[strict_signal_view['symbol'] == 'MGC', 'fold_count'].iloc[0])}` | verdict `{strict_signal_view.loc[strict_signal_view['symbol'] == 'MGC', 'verdict'].iloc[0]}`.",
    f"- Sleeve strict `M2K + MGC` equal-weight : net `{fmt_money(strict_sleeve_row['net_pnl'])}` | PF `{fmt_float(strict_sleeve_row['profit_factor'])}` | maxDD `{fmt_money(strict_sleeve_row['max_drawdown'])}` | verdict `{strict_sleeve_row['verdict']}`.",
    f"- Regime gating MGC non retenu : meilleur portefeuille gate net `{fmt_money(best_regime_row['net_pnl'])}` versus sleeve raw `{fmt_money(strict_sleeve_row['net_pnl'])}`.",
    f"- Integration au book baseline : baseline `{fmt_money(baseline_row['net_pnl'])}` -> baseline + sleeve `{fmt_money(integrated_row['net_pnl'])}` | incremental `{fmt_money(incremental_equal['incremental_net_pnl_vs_baseline'])}` | corr pullback vs baseline `{fmt_float(correlation_equal['correlation'])}`.",
]
display(Markdown("\\n".join(summary_lines)))

executive_table = strict_signal_view[
    [
        "display_name",
        "total_test_net_pnl",
        "test_profit_factor",
        "positive_folds",
        "fold_count",
        "avg_trade",
        "max_drawdown",
        "monthly_positive_ratio",
        "verdict",
    ]
].copy()
display(executive_table.round(3))
"""
    )


def _signals_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 2. The Two Signals"))

signal_card = strict_signal_view[
    [
        "symbol",
        "display_name",
        "total_test_trades",
        "total_test_net_pnl",
        "test_profit_factor",
        "avg_trade",
        "max_drawdown",
        "positive_folds",
        "fold_count",
        "train_score_test_corr",
        "selected_family_counts",
        "selected_cluster_counts",
        "verdict",
    ]
].copy()
display(signal_card.round(3))

selected_columns = [
    "symbol",
    "fold_id",
    "config_id",
    "family",
    "cluster_id",
    "filter_name",
    "stop_multiplier",
    "target_multiplier",
    "entry_delay_minutes",
    "train_robust_score",
    "train_net_pnl",
    "train_profit_factor",
    "train_trades",
    "selected_in_fold",
]
signal_selection = selection_view.loc[
    selection_view["symbol"].astype(str).isin(PRIMARY_SYMBOLS),
    selected_columns,
].copy()
display(Markdown("### Fold selections actually used"))
display(signal_selection.round(3))
"""
    )


def _folds_and_curves_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 3. Fold-by-Fold and Sleeve Curves"))

fold_fig = px.bar(
    strict_fold_view.loc[strict_fold_view["symbol"].astype(str).isin(PRIMARY_SYMBOLS)],
    x="fold_id",
    y="test_net_pnl",
    color="display_name",
    barmode="group",
    title="Strict WFA test PnL by fold",
)
fold_fig.update_layout(template=PLOT_TEMPLATE, width=1150, height=450)
fold_fig.show()

curve_fig = make_subplots(
    rows=2,
    cols=1,
    vertical_spacing=0.12,
    subplot_titles=("Strict sleeve stitched equity", "Portfolio integration OOS equity"),
)

for portfolio_name, label, color in [
    (M2K_ONLY_PORTFOLIO, "M2K only", "#2563eb"),
    (MGC_ONLY_PORTFOLIO, "MGC only", "#a855f7"),
    (STRICT_SLEEVE_NAME, "M2K + MGC strict sleeve", "#f59e0b"),
    (STRICT_SLEEVE_FALLBACK, "M2K + MGC strict sleeve", "#f59e0b"),
]:
    frame = strict_portfolio_daily.loc[strict_portfolio_daily["portfolio_name"].astype(str) == portfolio_name].copy()
    if frame.empty:
        continue
    frame = frame.sort_values("session_date")
    curve_fig.add_trace(
        go.Scatter(
            x=frame["session_date"],
            y=frame["equity"],
            mode="lines",
            name=label,
            line=dict(color=color, width=2.5 if "strict sleeve" in label else 1.9),
        ),
        row=1,
        col=1,
    )

for name, y_col, color in [
    ("Baseline only", "baseline_equity", "#475569"),
    ("Pullback sleeve only", "pullback_equity", "#f59e0b"),
    ("Baseline + pullback", "integrated_equity", "#16a34a"),
]:
    curve_fig.add_trace(
        go.Scatter(
            x=integration_curve["session_date"],
            y=integration_curve[y_col],
            mode="lines",
            name=name,
            line=dict(color=color, width=2.6 if name == "Baseline + pullback" else 2.0),
        ),
        row=2,
        col=1,
    )

curve_fig.update_yaxes(title_text="Equity (USD)", row=1, col=1)
curve_fig.update_yaxes(title_text="OOS cumulative pnl (USD)", row=2, col=1)
curve_fig.update_xaxes(title_text="Session date", row=2, col=1)
curve_fig.update_layout(template=PLOT_TEMPLATE, height=950, width=1450, legend=dict(orientation="h", y=-0.10, x=0.0))
curve_fig.show()
"""
    )


def _stability_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 4. Stability Around Parameters"))

cluster_table = cluster_view[
    [
        "symbol",
        "cluster_id",
        "family",
        "configs",
        "median_is_net_pnl",
        "median_oos_net_pnl",
        "median_fixed_wfa_net_pnl",
        "pct_configs_positive_oos",
        "pct_configs_positive_fixed_wfa",
        "selected_in_any_fold",
    ]
].copy()
display(cluster_table.round(3))

local_focus = local_view.loc[
    local_view["symbol"].astype(str).isin(PRIMARY_SYMBOLS)
    & local_view["rank_is"].fillna(999999).astype(int).le(15)
].copy()
display(Markdown("### Local neighborhood around the best ranks"))
display(
    local_focus[
        [
            "symbol",
            "config_id",
            "family",
            "cluster_id",
            "stop_multiplier",
            "target_multiplier",
            "entry_delay_minutes",
            "variant_time_stop_bars",
            "stop_zone_fraction",
            "adverse_window_minutes",
            "max_adverse_ticks",
            "robust_score_is",
            "net_pnl_oos",
            "fixed_wfa_net_pnl",
            "neighbor_median_oos_pnl",
            "neighbor_positive_fold_ratio",
        ]
    ].round(3)
)

scatter = px.scatter(
    local_focus,
    x="robust_score_is",
    y="fixed_wfa_net_pnl",
    color="symbol",
    hover_name="config_id",
    symbol="family",
    title="Train robustness vs fixed WFA net pnl",
)
scatter.update_layout(template=PLOT_TEMPLATE, width=1200, height=550)
scatter.show()
"""
    )


def _regime_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 5. Why Regime Gating Was Not Retained"))

regime_compare = strict_regime_wfa_summary.copy()
regime_compare["entity_label"] = regime_compare[regime_entity_col].astype(str)
display(regime_compare.round(3))

display(Markdown("### Selected regime rule by fold"))
display(
    selected_regime_rule_by_fold[
        [
            "fold_id",
            "rule_id",
            "family",
            "allocation_scheme",
            "train_score",
            "train_net_pnl",
            "train_profit_factor",
            "train_trades",
            "mgc_retention_rate_train",
        ]
    ].round(3)
)

retention_fig = px.bar(
    mgc_regime_retention_summary,
    x="fold_id",
    y=["mgc_test_retention_rate", "mgc_train_retention_rate"],
    barmode="group",
    title="MGC retention under selected regime rules",
)
retention_fig.update_layout(template=PLOT_TEMPLATE, width=1100, height=450)
retention_fig.show()
"""
    )


def _portfolio_integration_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 6. Portfolio Integration Readout"))

integration_view = portfolio_summary.loc[
    (portfolio_summary["scope"].astype(str) == "defined_oos")
    & (
        portfolio_summary["portfolio_name"].astype(str).isin(
            [
                BASELINE_PORTFOLIO,
                PULLBACK_PORTFOLIO,
                INTEGRATED_PORTFOLIO,
                INTEGRATED_M2K_ONLY,
            ]
        )
    )
].copy()
display(
    integration_view[
        [
            "portfolio_name",
            "net_pnl",
            "daily_sharpe",
            "sortino",
            "max_drawdown",
            "max_daily_loss",
            "profit_factor",
            "day_win_rate",
            "monthly_hit_rate",
            "verdict",
        ]
    ].round(3)
)

display(Markdown("### Incremental impact vs baseline"))
display(incremental_metrics.round(3))

display(Markdown("### Bootstrap robustness"))
display(
    bootstrap_summary[
        [
            "portfolio_name",
            "median_net_pnl",
            "p05_net_pnl",
            "p95_net_pnl",
            "probability_positive",
            "probability_drawdown_breach_2k",
            "probability_prop_pass",
        ]
    ].round(3)
)

display(Markdown("### Prop-firm constraint summary"))
display(prop_constraint_summary.round(3))
"""
    )


def _seasonality_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 7. Trade Concentration and Seasonality"))

display(Markdown("### Trade concentration"))
display(survivor_trade_concentration.round(3))

monthly_focus = survivor_monthly_pnl.loc[
    survivor_monthly_pnl["entity_id"].astype(str).isin(PRIMARY_SYMBOLS)
].copy()
monthly_focus["month"] = pd.to_datetime(monthly_focus["month"], errors="coerce")

monthly_fig = px.bar(
    monthly_focus,
    x="month",
    y="pnl",
    color="entity_id",
    barmode="group",
    title="Monthly pnl by signal",
)
monthly_fig.update_layout(template=PLOT_TEMPLATE, width=1350, height=450)
monthly_fig.show()

if "year" in survivor_yearly_pnl.columns:
    yearly_focus = survivor_yearly_pnl.loc[survivor_yearly_pnl["entity_id"].astype(str).isin(PRIMARY_SYMBOLS)].copy()
    yearly_fig = px.bar(
        yearly_focus,
        x="year",
        y="pnl",
        color="entity_id",
        barmode="group",
        title="Yearly pnl by signal",
    )
    yearly_fig.update_layout(template=PLOT_TEMPLATE, width=1000, height=420)
    yearly_fig.show()
"""
    )


def _conclusion_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 8. Conclusion"))

lines = [
    f"- `M2K 1H` reste le signal coeur a conserver en watchlist : net `{fmt_money(m2k_row['net_pnl'])}` | PF `{fmt_float(m2k_row['profit_factor'])}` | verdict `{m2k_row['verdict']}`.",
    f"- `MGC 1H` n'est pas promu seul : net `{fmt_money(mgc_row['net_pnl'])}` mais seulement `{int(mgc_row['positive_folds'])}/{int(mgc_row['fold_count'])}` folds positifs, donc verdict `{mgc_row['verdict']}`.",
    f"- Le meilleur sleeve strict a garder comme reference est `M2K + MGC` equal-weight : net `{fmt_money(strict_sleeve_row['net_pnl'])}` | PF `{fmt_float(strict_sleeve_row['profit_factor'])}` | verdict `{strict_sleeve_row['verdict']}`.",
    f"- Le gating regime n'a pas amene de gain robuste : `{fmt_money(best_regime_row['net_pnl'])}` versus raw strict `{fmt_money(strict_sleeve_row['net_pnl'])}`.",
    f"- Comme overlay portefeuille, le sleeve ajoute `{fmt_money(incremental_equal['incremental_net_pnl_vs_baseline'])}` au baseline avec une correlation faible `{fmt_float(correlation_equal['correlation'])}`, donc `diversifier_watchlist` plutot que candidat deployable.",
    "- Si tu veux la suite naturelle, on peut maintenant te construire le replay scene-by-scene des trades selectionnes avec bougies 1min, niveaux stop/target, ordres et etat regime.",
]
display(Markdown("\\n".join(lines)))
"""
    )


def build_notebook(survivor_root: Path, regime_root: Path, integration_root: Path) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"},
    }
    notebook.cells = [
        _title_cell(),
        _imports_cell(),
        _parameter_cell(survivor_root, regime_root, integration_root),
        _load_cell(),
        _summary_cell(),
        _signals_cell(),
        _folds_and_curves_cell(),
        _stability_cell(),
        _regime_cell(),
        _portfolio_integration_cell(),
        _seasonality_cell(),
        _conclusion_cell(),
    ]
    return notebook


def write_notebook(notebook: nbf.NotebookNode, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(nbf.writes(notebook), encoding="utf-8")
    return output_path


def execute_notebook(input_path: Path, output_path: Path, timeout_seconds: int = 1800) -> Path:
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
    parser.add_argument("--survivor-export-root", type=Path, default=find_latest_export(SURVIVOR_PREFIX))
    parser.add_argument("--regime-export-root", type=Path, default=find_latest_export(REGIME_PREFIX))
    parser.add_argument("--integration-export-root", type=Path, default=find_latest_export(INTEGRATION_PREFIX))
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK_PATH)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--executed-output", type=Path, default=DEFAULT_EXECUTED_NOTEBOOK_PATH)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebook = build_notebook(
        survivor_root=args.survivor_export_root,
        regime_root=args.regime_export_root,
        integration_root=args.integration_export_root,
    )
    output_path = write_notebook(notebook, args.output)
    print(f"Notebook written to {output_path}")
    if args.execute:
        executed_path = execute_notebook(output_path, args.executed_output, timeout_seconds=args.timeout_seconds)
        print(f"Executed notebook written to {executed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
