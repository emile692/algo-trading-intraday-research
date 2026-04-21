"""Build a client-facing notebook for the MNQ risk-sizing refinement export."""

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
DEFAULT_EXPORT_PREFIX = "volume_climax_pullback_mnq_risk_sizing_refinement_"
DEFAULT_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "volume_climax_pullback_mnq_risk_sizing_client.ipynb"
DEFAULT_EXECUTED_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "volume_climax_pullback_mnq_risk_sizing_client.executed.ipynb"


def find_latest_export(prefix: str = DEFAULT_EXPORT_PREFIX) -> Path:
    candidates = [path for path in EXPORTS_ROOT.glob(f"{prefix}*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No export folder found for prefix {prefix!r} under {EXPORTS_ROOT}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _title_cell() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """# MNQ Volume Climax Pullback - Risk Sizing Client Notebook

Ce notebook relit la campagne locale de refinement du sizing a risque fixe sur `MNQ`.

- alpha, signaux, horaires et exits restent ceux de la strategie VCP retenue,
- seul le sizing varie dans la grille locale autour de la zone gagnante,
- tous les parametres utiles du notebook sont regroupes dans une seule cellule,
- les graphiques principaux sont en Plotly pour une lecture client directe.
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

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import Markdown, display
from plotly.subplots import make_subplots

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)


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


def resolve_variant_name(summary_frame, active_variant_name, risk_pct, max_contracts, skip_trade_if_too_small):
    if active_variant_name:
        match = summary_frame.loc[summary_frame["campaign_variant_name"].astype(str) == str(active_variant_name)].copy()
        if match.empty:
            raise ValueError(f"Variant {active_variant_name!r} not found in export.")
        return str(match.iloc[0]["campaign_variant_name"])

    mask = (
        pd.to_numeric(summary_frame["risk_pct"], errors="coerce").round(6).eq(round(float(risk_pct), 6))
        & pd.to_numeric(summary_frame["max_contracts"], errors="coerce").astype("Int64").eq(int(max_contracts))
        & pd.Series(summary_frame["skip_trade_if_too_small"], dtype="boolean").fillna(False).eq(bool(skip_trade_if_too_small))
    )
    match = summary_frame.loc[mask].copy()
    if match.empty:
        raise ValueError("No variant matches the requested notebook knobs.")
    ordered = match.sort_values(["oos_prop_score", "oos_net_pnl_usd"], ascending=[False, False]).reset_index(drop=True)
    return str(ordered.iloc[0]["campaign_variant_name"])
"""
    )


def _parameter_cell(export_root: Path) -> nbf.NotebookNode:
    export_root = export_root if export_root.is_absolute() else (REPO_ROOT / export_root)
    export_root = export_root.resolve()
    return nbf.v4.new_code_cell(
        f"""EXPORT_ROOT = ROOT / r"{export_root.relative_to(REPO_ROOT)}"

# Main notebook knobs
ACTIVE_VARIANT_NAME = None
ACTIVE_RISK_PCT = 0.0025
ACTIVE_MAX_CONTRACTS = 3
ACTIVE_SKIP_TRADE_IF_TOO_SMALL = True
COMPARE_BASELINE = True
COMPARE_PREVIOUS_WINNER = True
PLOT_TEMPLATE = "plotly_dark"

required_paths = {{
    "summary_by_variant": EXPORT_ROOT / "summary_by_variant.csv",
    "summary_oos_only": EXPORT_ROOT / "summary_oos_only.csv",
    "trades_by_variant": EXPORT_ROOT / "trades_by_variant.csv",
    "daily_equity_by_variant": EXPORT_ROOT / "daily_equity_by_variant.csv",
    "prop_constraints_summary": EXPORT_ROOT / "prop_constraints_summary.csv",
    "heatmap_metrics": EXPORT_ROOT / "heatmap_metrics.csv",
    "robustness_zone_summary": EXPORT_ROOT / "robustness_zone_summary.csv",
    "run_metadata": EXPORT_ROOT / "run_metadata.json",
    "final_verdict": EXPORT_ROOT / "final_verdict.json",
    "final_report": EXPORT_ROOT / "final_report.md",
}}

missing = [name for name, path in required_paths.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"Fichiers manquants pour le notebook: {{missing}}")

print("EXPORT_ROOT =", EXPORT_ROOT)
"""
    )


def _load_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """run_metadata = json.loads((EXPORT_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
final_verdict = json.loads((EXPORT_ROOT / "final_verdict.json").read_text(encoding="utf-8"))
final_report_text = (EXPORT_ROOT / "final_report.md").read_text(encoding="utf-8")

summary = pd.read_csv(EXPORT_ROOT / "summary_by_variant.csv")
summary_oos = pd.read_csv(EXPORT_ROOT / "summary_oos_only.csv")
trades = pd.read_csv(EXPORT_ROOT / "trades_by_variant.csv", parse_dates=["entry_time", "exit_time"], low_memory=False)
daily = pd.read_csv(EXPORT_ROOT / "daily_equity_by_variant.csv", parse_dates=["session_date"], low_memory=False)
prop_summary = pd.read_csv(EXPORT_ROOT / "prop_constraints_summary.csv")
heatmap_metrics = pd.read_csv(EXPORT_ROOT / "heatmap_metrics.csv")
cluster_summary = pd.read_csv(EXPORT_ROOT / "robustness_zone_summary.csv")

active_variant = resolve_variant_name(
    summary,
    active_variant_name=ACTIVE_VARIANT_NAME,
    risk_pct=ACTIVE_RISK_PCT,
    max_contracts=ACTIVE_MAX_CONTRACTS,
    skip_trade_if_too_small=ACTIVE_SKIP_TRADE_IF_TOO_SMALL,
)
baseline_variant = "fixed_1_contract"
previous_variant = final_verdict.get("recommended_variant") or final_verdict.get("punctual_oos_winner")
best_previous_tag = run_metadata.get("best_previous_winner_alias", "best_previous_winner")

variant_rows = {
    name: summary.loc[summary["campaign_variant_name"].astype(str) == str(name)].iloc[0]
    for name in {active_variant, baseline_variant, best_previous_tag, previous_variant}
    if summary["campaign_variant_name"].astype(str).eq(str(name)).any()
}

compare_names = [active_variant]
if COMPARE_BASELINE and baseline_variant not in compare_names:
    compare_names.append(baseline_variant)
if COMPARE_PREVIOUS_WINNER and best_previous_tag in variant_rows and best_previous_tag not in compare_names:
    compare_names.append(best_previous_tag)

display(Markdown(f"**Active variant:** `{active_variant}`"))
display(Markdown(f"**Final verdict:** `{final_verdict['final_verdict']}`"))
"""
    )


def _snapshot_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """snapshot_cols = [
    "campaign_variant_name",
    "variant_role",
    "oos_net_pnl_usd",
    "oos_cagr_pct",
    "oos_sharpe",
    "oos_max_drawdown_usd",
    "oos_max_daily_drawdown_usd",
    "oos_prop_score",
    "oos_pass_target_3000_usd_without_breaching_2000_dd",
    "oos_avg_contracts",
    "oos_median_contracts",
    "oos_pct_trades_at_1_contract",
    "oos_pct_trades_at_2_contracts",
    "oos_pct_trades_at_3_plus_contracts",
    "oos_nb_skipped_trades",
    "oos_pct_skipped_trades",
]
display(Markdown("## 1. Snapshot OOS"))
display(summary.loc[summary["campaign_variant_name"].isin(compare_names), snapshot_cols].round(3))

if not cluster_summary.empty:
    display(Markdown("## 2. Zone robuste"))
    display(cluster_summary.round(4))
"""
    )


def _curve_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 3. Equity Curves"))

fig = make_subplots(
    rows=2,
    cols=2,
    shared_xaxes=False,
    vertical_spacing=0.10,
    horizontal_spacing=0.10,
    subplot_titles=("Full Sample Equity", "OOS Only Equity", "Full Sample Drawdown USD", "OOS Only Drawdown USD"),
)

colors = ["#16a34a", "#2563eb", "#f59e0b", "#7c3aed"]
for color, name in zip(colors, compare_names):
    full_curve = daily.loc[(daily["campaign_variant_name"] == name) & (daily["scope"] == "full")].copy()
    oos_curve = daily.loc[(daily["campaign_variant_name"] == name) & (daily["scope"] == "oos_only")].copy()
    fig.add_trace(go.Scatter(x=full_curve["session_date"], y=full_curve["equity"], mode="lines", name=f"{name} full", line=dict(color=color, width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=oos_curve["session_date"], y=oos_curve["equity"], mode="lines", name=f"{name} oos", line=dict(color=color, width=2.5), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=full_curve["session_date"], y=full_curve["drawdown_usd"], mode="lines", name=f"{name} full dd", line=dict(color=color, width=1.6, dash="dot"), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=oos_curve["session_date"], y=oos_curve["drawdown_usd"], mode="lines", name=f"{name} oos dd", line=dict(color=color, width=1.6, dash="dot"), showlegend=False), row=2, col=2)

fig.update_layout(template=PLOT_TEMPLATE, height=850, width=1500, legend=dict(orientation="h", y=-0.12))
fig.show()
"""
    )


def _heatmap_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 4. Heatmaps OOS"))

metric_specs = [
    ("oos_net_pnl_usd", "OOS Net PnL", "RdYlGn"),
    ("oos_sharpe", "OOS Sharpe", "RdYlGn"),
    ("oos_max_drawdown_usd", "OOS MaxDD USD", "RdYlGn_r"),
    ("oos_prop_score", "OOS Prop Score", "RdYlGn"),
]

fig = make_subplots(rows=2, cols=2, subplot_titles=[title for _, title, _ in metric_specs], horizontal_spacing=0.10, vertical_spacing=0.14)
for index, (metric, title, colorscale) in enumerate(metric_specs, start=1):
    pivot = heatmap_metrics.pivot_table(index="risk_pct", columns="max_contracts", values=metric, aggfunc="mean").sort_index()
    row = 1 if index <= 2 else 2
    col = 1 if index in (1, 3) else 2
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=[str(int(value)) for value in pivot.columns],
            y=[f"{float(value):.4f}" for value in pivot.index],
            colorscale=colorscale,
            text=pivot.round(2).astype(str).values,
            texttemplate="%{text}",
            hovertemplate="risk_pct=%{y}<br>max_contracts=%{x}<br>value=%{z}<extra></extra>",
        ),
        row=row,
        col=col,
    )

fig.update_layout(
    template=PLOT_TEMPLATE,
    height=900,
    width=1500,
)
fig.show()
"""
    )


def _distribution_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 5. Active Variant Trade Profile"))

active_trades_oos = trades.loc[(trades["campaign_variant_name"] == active_variant) & (trades["scope"] == "oos_only")].copy()
baseline_trades_oos = trades.loc[(trades["campaign_variant_name"] == baseline_variant) & (trades["scope"] == "oos_only")].copy()

if not active_trades_oos.empty:
    quantity_mix = (
        active_trades_oos.assign(bucket=active_trades_oos["quantity"].map(lambda q: "3+" if q >= 3 else str(int(q))))
        .groupby("bucket", as_index=False)
        .agg(n_trades=("trade_id", "count"), net_pnl_usd=("net_pnl_usd", "sum"))
    )
    quantity_mix["pct_trades"] = quantity_mix["n_trades"] / quantity_mix["n_trades"].sum() * 100.0
    display(quantity_mix.round(2))

    mix_fig = px.bar(quantity_mix, x="bucket", y="pct_trades", text_auto=".1f", title="Active OOS position-size mix", labels={"bucket": "contracts bucket", "pct_trades": "% trades"})
    mix_fig.update_layout(template=PLOT_TEMPLATE, width=900, height=450)
    mix_fig.show()

hist_source = pd.concat(
    [
        baseline_trades_oos[["net_pnl_usd"]].assign(label="baseline"),
        active_trades_oos[["net_pnl_usd"]].assign(label="active"),
    ],
    ignore_index=True,
)
if not hist_source.empty:
    hist_fig = px.histogram(hist_source, x="net_pnl_usd", color="label", marginal="box", barmode="overlay", opacity=0.55, nbins=50, title="OOS trade PnL distribution")
    hist_fig.update_layout(template=PLOT_TEMPLATE, width=1100, height=500)
    hist_fig.show()
"""
    )


def _conclusion_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 6. Final Readout"))
display(Markdown(final_report_text))
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
        _snapshot_cell(),
        _curve_cell(),
        _heatmap_cell(),
        _distribution_cell(),
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
    parser.add_argument("--export-root", type=Path, default=find_latest_export())
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
