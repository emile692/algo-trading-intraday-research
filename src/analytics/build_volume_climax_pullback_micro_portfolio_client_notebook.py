"""Build a client notebook for the micro portfolio campaign."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPORTS_ROOT = REPO_ROOT / "data" / "exports"
NOTEBOOKS_ROOT = REPO_ROOT / "notebooks"

DEFAULT_EXPORT_ROOT = EXPORTS_ROOT / "volume_climax_pullback_micro_portfolio_20260409_000331"
DEFAULT_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "volume_climax_pullback_micro_portfolio_client.ipynb"
DEFAULT_EXECUTED_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "volume_climax_pullback_micro_portfolio_client.executed.ipynb"


def _title_cell() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """# Volume Climax Pullback Micro Portfolio - Client Notebook

Ce notebook relit l'export portefeuille audite et met en avant la hierarchie figee :

- Recherche / reference : `MNQ_M2K_MES__equal_weight_notional__core_default`
- Live prop-safe : `MNQ_M2K_MES__equal_weight_risk_budget__conservative_mix`
- Mode agressif controle : `MNQ_M2K_MES__capped_overlay__core_default`

Les courbes des singles peuvent rester affiches en reference dans la meme vue.
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

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 240)


def fmt_money(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.1f} USD"


def fmt_pct(value, digits=2):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}%"


def fmt_float(value, digits=3):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"
"""
    )


def _parameter_cell(export_root: Path) -> nbf.NotebookNode:
    export_root = export_root if export_root.is_absolute() else (REPO_ROOT / export_root)
    export_root = export_root.resolve()
    return nbf.v4.new_code_cell(
        f"""EXPORT_ROOT = ROOT / r"{export_root.relative_to(REPO_ROOT)}"

# Frozen hierarchy requested by the user
RESEARCH_VARIANT = "MNQ_M2K_MES__equal_weight_notional__core_default"
PROP_SAFE_VARIANT = "MNQ_M2K_MES__equal_weight_risk_budget__conservative_mix"
AGGRESSIVE_VARIANT = "MNQ_M2K_MES__capped_overlay__core_default"

# Optional reference curves on the same chart
INCLUDE_SINGLE_REFERENCES = True
REFERENCE_VARIANTS = [
    "MNQ_only__standalone__core_default",
    "M2K_only__standalone__core_default",
    "MES_only__standalone__core_default",
]

PLOT_TEMPLATE = "plotly_dark"

required_paths = {{
    "summary": EXPORT_ROOT / "summary_by_portfolio.csv",
    "daily_equity": EXPORT_ROOT / "daily_equity_by_portfolio.csv",
    "daily_motor": EXPORT_ROOT / "daily_pnl_by_motor.csv",
    "diversification": EXPORT_ROOT / "diversification_summary.csv",
    "corr_daily": EXPORT_ROOT / "correlation_matrix_daily.csv",
    "final_verdict": EXPORT_ROOT / "final_verdict.json",
}}

missing = [name for name, path in required_paths.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"Fichiers manquants pour le notebook: {{missing}}")

print("EXPORT_ROOT =", EXPORT_ROOT)
"""
    )


def _load_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """summary = pd.read_csv(EXPORT_ROOT / "summary_by_portfolio.csv")
daily_equity = pd.read_csv(EXPORT_ROOT / "daily_equity_by_portfolio.csv", parse_dates=["session_date"])
daily_motor = pd.read_csv(EXPORT_ROOT / "daily_pnl_by_motor.csv", parse_dates=["session_date"])
diversification = pd.read_csv(EXPORT_ROOT / "diversification_summary.csv")
correlation_matrix_daily = pd.read_csv(EXPORT_ROOT / "correlation_matrix_daily.csv", index_col=0)
final_verdict = json.loads((EXPORT_ROOT / "final_verdict.json").read_text(encoding="utf-8"))

daily_equity["session_date"] = daily_equity["session_date"].dt.normalize()
daily_motor["session_date"] = daily_motor["session_date"].dt.normalize()

selected_portfolios = [RESEARCH_VARIANT, PROP_SAFE_VARIANT, AGGRESSIVE_VARIANT]
if INCLUDE_SINGLE_REFERENCES:
    selected_portfolios.extend(REFERENCE_VARIANTS)
selected_portfolios = list(dict.fromkeys(selected_portfolios))

available_names = set(summary["portfolio_variant_name"].astype(str))
missing_names = [name for name in selected_portfolios if name not in available_names]
if missing_names:
    raise ValueError(f"Portfolio variants introuvables dans l'export: {missing_names}")

display_names = {
    RESEARCH_VARIANT: "Recherche / reference",
    PROP_SAFE_VARIANT: "Live prop-safe",
    AGGRESSIVE_VARIANT: "Agressif controle",
    "MNQ_only__standalone__core_default": "MNQ seul",
    "M2K_only__standalone__core_default": "M2K seul",
    "MES_only__standalone__core_default": "MES seul",
}

selected_summary = summary.loc[summary["portfolio_variant_name"].astype(str).isin(selected_portfolios)].copy()
selected_summary["display_name"] = selected_summary["portfolio_variant_name"].astype(str).map(display_names).fillna(selected_summary["portfolio_variant_name"].astype(str))

selected_daily = daily_equity.loc[daily_equity["portfolio_variant_name"].astype(str).isin(selected_portfolios)].copy()
selected_full = selected_daily.loc[selected_daily["scope"].astype(str) == "full"].copy()
selected_oos = selected_daily.loc[selected_daily["scope"].astype(str) == "oos"].copy()
common_oos_start = selected_oos["session_date"].min()

metrics_view = selected_summary[
    [
        "display_name",
        "portfolio_variant_name",
        "allocation_scheme",
        "config_bundle",
        "oos_net_pnl_usd",
        "oos_cagr_pct",
        "oos_sharpe",
        "oos_max_drawdown_usd",
        "oos_max_daily_drawdown_usd",
        "oos_portfolio_score",
        "oos_pass_target_3000_usd_without_breaching_2000_dd",
    ]
].sort_values(["oos_portfolio_score", "oos_sharpe"], ascending=[False, False]).reset_index(drop=True)

research_div = diversification.loc[
    (diversification["portfolio_variant_name"].astype(str) == RESEARCH_VARIANT)
    & (diversification["scope"].astype(str) == "oos")
].copy()
research_div = research_div.iloc[0] if not research_div.empty else None

research_motor_oos = daily_motor.loc[
    (daily_motor["portfolio_variant_name"].astype(str) == RESEARCH_VARIANT)
    & (daily_motor["scope"].astype(str) == "oos")
].copy()
research_motor_oos["symbol"] = research_motor_oos["symbol"].astype(str)
research_motor_oos["cumulative_pnl_usd"] = research_motor_oos.groupby("symbol")["daily_pnl_usd"].cumsum()

research_motor_monthly = research_motor_oos.copy()
research_motor_monthly["month"] = research_motor_monthly["session_date"].dt.to_period("M").dt.to_timestamp()
research_motor_monthly = (
    research_motor_monthly.groupby(["month", "symbol"], as_index=False)["daily_pnl_usd"]
    .sum()
    .rename(columns={"daily_pnl_usd": "monthly_pnl_usd"})
)

display(Markdown(f"**Export root:** `{EXPORT_ROOT}`"))
display(Markdown(f"**Common OOS start:** `{common_oos_start.date()}`"))
"""
    )


def _summary_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 1. Executive Summary"))

research_row = selected_summary.loc[selected_summary["portfolio_variant_name"].astype(str) == RESEARCH_VARIANT].iloc[0]
prop_safe_row = selected_summary.loc[selected_summary["portfolio_variant_name"].astype(str) == PROP_SAFE_VARIANT].iloc[0]
aggressive_row = selected_summary.loc[selected_summary["portfolio_variant_name"].astype(str) == AGGRESSIVE_VARIANT].iloc[0]

lines = [
    f"- Recherche / reference: `{RESEARCH_VARIANT}` | net `{fmt_money(research_row['oos_net_pnl_usd'])}` | Sharpe `{fmt_float(research_row['oos_sharpe'])}` | maxDD `{fmt_money(research_row['oos_max_drawdown_usd'])}`.",
    f"- Live prop-safe: `{PROP_SAFE_VARIANT}` | net `{fmt_money(prop_safe_row['oos_net_pnl_usd'])}` | Sharpe `{fmt_float(prop_safe_row['oos_sharpe'])}` | maxDD `{fmt_money(prop_safe_row['oos_max_drawdown_usd'])}`.",
    f"- Agressif controle: `{AGGRESSIVE_VARIANT}` | net `{fmt_money(aggressive_row['oos_net_pnl_usd'])}` | Sharpe `{fmt_float(aggressive_row['oos_sharpe'])}` | maxDD `{fmt_money(aggressive_row['oos_max_drawdown_usd'])}`.",
    f"- Verdict export: `{final_verdict['final_verdict']}`.",
]
display(Markdown("\\n".join(lines)))

display(Markdown("## 2. Selected Metrics"))
display(metrics_view.round(3))
"""
    )


def _equity_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 3. Equity Curves"))

color_map = {
    "Recherche / reference": "#f59e0b",
    "Live prop-safe": "#22c55e",
    "Agressif controle": "#ef4444",
    "MNQ seul": "#60a5fa",
    "M2K seul": "#a78bfa",
    "MES seul": "#14b8a6",
}

fig = make_subplots(
    rows=2,
    cols=1,
    vertical_spacing=0.12,
    subplot_titles=("Full Sample Equity", "OOS Equity"),
)

for portfolio_name in selected_portfolios:
    label = display_names.get(portfolio_name, portfolio_name)
    color = color_map.get(label, "#d4d4d4")

    full_frame = selected_full.loc[selected_full["portfolio_variant_name"].astype(str) == portfolio_name].copy()
    if not full_frame.empty:
        fig.add_trace(
            go.Scatter(
                x=full_frame["session_date"],
                y=full_frame["equity"],
                mode="lines",
                name=label,
                legendgroup=label,
                line=dict(color=color, width=2.8 if portfolio_name == RESEARCH_VARIANT else 2.0),
            ),
            row=1,
            col=1,
        )

    oos_frame = selected_oos.loc[selected_oos["portfolio_variant_name"].astype(str) == portfolio_name].copy()
    if not oos_frame.empty:
        fig.add_trace(
            go.Scatter(
                x=oos_frame["session_date"],
                y=oos_frame["equity"],
                mode="lines",
                name=f"{label} OOS",
                legendgroup=label,
                showlegend=False,
                line=dict(color=color, width=2.8 if portfolio_name == RESEARCH_VARIANT else 2.0),
            ),
            row=2,
            col=1,
        )

fig.update_yaxes(title_text="Equity (USD)", row=1, col=1)
fig.update_yaxes(title_text="Equity (USD)", row=2, col=1)
fig.update_xaxes(title_text="Session Date", row=2, col=1)
fig.update_layout(template=PLOT_TEMPLATE, height=950, width=1550, legend=dict(orientation="h", y=-0.12, x=0.0))
fig.show()
"""
    )


def _drawdown_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 4. Drawdown Curves"))

fig = make_subplots(
    rows=2,
    cols=1,
    vertical_spacing=0.12,
    subplot_titles=("Full Sample Drawdown", "OOS Drawdown"),
)

for portfolio_name in selected_portfolios:
    label = display_names.get(portfolio_name, portfolio_name)
    color = color_map.get(label, "#d4d4d4")

    full_frame = selected_full.loc[selected_full["portfolio_variant_name"].astype(str) == portfolio_name].copy()
    if not full_frame.empty:
        fig.add_trace(
            go.Scatter(
                x=full_frame["session_date"],
                y=full_frame["drawdown_usd"],
                mode="lines",
                name=label,
                legendgroup=label,
                line=dict(color=color, width=2.8 if portfolio_name == RESEARCH_VARIANT else 2.0),
            ),
            row=1,
            col=1,
        )

    oos_frame = selected_oos.loc[selected_oos["portfolio_variant_name"].astype(str) == portfolio_name].copy()
    if not oos_frame.empty:
        fig.add_trace(
            go.Scatter(
                x=oos_frame["session_date"],
                y=oos_frame["drawdown_usd"],
                mode="lines",
                name=f"{label} OOS",
                legendgroup=label,
                showlegend=False,
                line=dict(color=color, width=2.8 if portfolio_name == RESEARCH_VARIANT else 2.0),
            ),
            row=2,
            col=1,
        )

fig.update_yaxes(title_text="Drawdown (USD)", row=1, col=1)
fig.update_yaxes(title_text="Drawdown (USD)", row=2, col=1)
fig.update_xaxes(title_text="Session Date", row=2, col=1)
fig.update_layout(template=PLOT_TEMPLATE, height=950, width=1550, legend=dict(orientation="h", y=-0.12, x=0.0))
fig.show()
"""
    )


def _diversification_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 5. Diversification"))

corr_fig = px.imshow(
    correlation_matrix_daily.astype(float),
    text_auto=".2f",
    color_continuous_scale="RdYlGn_r",
    zmin=-1.0,
    zmax=1.0,
    title="Daily PnL Correlation Heatmap (OOS, default motors)",
)
corr_fig.update_layout(template=PLOT_TEMPLATE, width=700, height=550)
corr_fig.show()

contrib_fig = px.line(
    research_motor_oos,
    x="session_date",
    y="cumulative_pnl_usd",
    color="symbol",
    title="Recherche / reference - OOS cumulative PnL contribution by motor",
)
contrib_fig.update_layout(template=PLOT_TEMPLATE, width=1350, height=500)
contrib_fig.show()

monthly_fig = px.bar(
    research_motor_monthly,
    x="month",
    y="monthly_pnl_usd",
    color="symbol",
    barmode="group",
    title="Recherche / reference - monthly motor contribution",
)
monthly_fig.update_layout(template=PLOT_TEMPLATE, width=1400, height=500)
monthly_fig.show()

if research_div is not None:
    pnl_dd = pd.DataFrame(
        {
            "symbol": ["MNQ", "M2K", "MES"],
            "pnl_contribution_pct": [
                research_div.get("pnl_contribution_pct_MNQ"),
                research_div.get("pnl_contribution_pct_M2K"),
                research_div.get("pnl_contribution_pct_MES"),
            ],
            "drawdown_contribution_pct": [
                research_div.get("drawdown_contribution_pct_MNQ"),
                research_div.get("drawdown_contribution_pct_M2K"),
                research_div.get("drawdown_contribution_pct_MES"),
            ],
        }
    )
    contrib_bar = px.bar(
        pnl_dd.melt(id_vars="symbol", var_name="metric", value_name="value"),
        x="symbol",
        y="value",
        color="metric",
        barmode="group",
        title="Recherche / reference - OOS contribution mix",
    )
    contrib_bar.update_layout(template=PLOT_TEMPLATE, width=1000, height=450)
    contrib_bar.show()
"""
    )


def _conclusion_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 6. Conclusion"))

research_row = selected_summary.loc[selected_summary["portfolio_variant_name"].astype(str) == RESEARCH_VARIANT].iloc[0]
mnq_single = selected_summary.loc[selected_summary["portfolio_variant_name"].astype(str) == "MNQ_only__standalone__core_default"]
mnq_row = mnq_single.iloc[0] if not mnq_single.empty else None

lines = [
    f"- Reference de recherche conservee: `{RESEARCH_VARIANT}`.",
    f"- Portefeuille prop-safe conserve: `{PROP_SAFE_VARIANT}`.",
    f"- Portefeuille agressif conserve: `{AGGRESSIVE_VARIANT}`.",
]
if mnq_row is not None:
    lines.append(
        f"- Versus MNQ seul: net `{float(research_row['oos_net_pnl_usd']) - float(mnq_row['oos_net_pnl_usd']):+.2f}` | Sharpe `{float(research_row['oos_sharpe']) - float(mnq_row['oos_sharpe']):+.3f}` | maxDD `{float(research_row['oos_max_drawdown_usd']) - float(mnq_row['oos_max_drawdown_usd']):+.2f}`."
    )
display(Markdown("\\n".join(lines)))
"""
    )


def build_notebook(export_root: Path) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"},
    }
    notebook.cells = [
        _title_cell(),
        _imports_cell(),
        _parameter_cell(export_root),
        _load_cell(),
        _summary_cell(),
        _equity_cell(),
        _drawdown_cell(),
        _diversification_cell(),
        _conclusion_cell(),
    ]
    return notebook


def write_notebook(notebook: nbf.NotebookNode, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(nbf.writes(notebook), encoding="utf-8")
    return output_path


def execute_notebook(input_path: Path, output_path: Path, timeout_seconds: int = 900) -> Path:
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
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK_PATH)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--executed-output", type=Path, default=DEFAULT_EXECUTED_NOTEBOOK_PATH)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebook = build_notebook(args.export_root)
    output_path = write_notebook(notebook, args.output)
    print(f"Notebook written to {output_path}")
    if args.execute:
        executed_path = execute_notebook(output_path, args.executed_output, timeout_seconds=args.timeout_seconds)
        print(f"Executed notebook written to {executed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
