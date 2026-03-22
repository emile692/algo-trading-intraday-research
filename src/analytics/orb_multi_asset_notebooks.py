"""Generate client notebooks for the multi-asset Ensemble ORB campaign."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.orb_multi_asset_campaign import resolve_processed_dataset

REPO_ROOT = ROOT
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

DEFAULT_NOTEBOOK_CONFIG = {
    "is_fraction": 0.70,
    "or_minutes": 15,
    "atr_periods": [25, 26, 27, 28, 29, 30],
    "q_low_values": [25, 26, 27, 28, 29, 30],
    "q_high_values": [90, 91, 92, 93, 94, 95],
    "aggregation_rule": "majority_50",
}

SYMBOL_NOTEBOOK_OVERRIDES = {
    # Focused OR30 defaults aligned with the recent MGC heatmap review.
    "MGC": {
        "or_minutes": 30,
        "atr_periods": [10, 14, 20, 26, 30, 40, 60],
        "q_low_values": [20, 22, 24, 26, 28, 30],
        "q_high_values": [75, 80, 85, 90, 95, 99, 100],
        "aggregation_rule": "majority_50",
    },
}


def _title(symbol: str) -> str:
    return f"# ORB {symbol} Final Ensemble Validation"


def _notebook_config(symbol: str) -> dict[str, object]:
    config = dict(DEFAULT_NOTEBOOK_CONFIG)
    config.update(SYMBOL_NOTEBOOK_OVERRIDES.get(symbol.upper(), {}))
    return config


def _imports_cell() -> str:
    return """import math
import sys
from pathlib import Path

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import Markdown, display

from src.analytics.orb_multi_asset_campaign import (
    BaselineSpec,
    SearchGrid,
    build_notebook_bundle,
    resolve_processed_dataset,
)
from src.analytics.orb_notebook_utils import (
    build_scope_readout_markdown,
    build_selected_ensemble_kpi_frame,
    curve_max_drawdown_pct,
    curve_total_return_pct,
)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)
"""


def _parameter_cell(symbol: str) -> str:
    config = _notebook_config(symbol)
    return f"""SYMBOL = "{symbol}"
DATASET_PATH = resolve_processed_dataset(SYMBOL)
IS_FRACTION = {config["is_fraction"]}

ATR_PERIODS = {config["atr_periods"]}
Q_LOW_VALUES = {config["q_low_values"]}
Q_HIGH_VALUES = {config["q_high_values"]}
AGGREGATION_RULE = "{config["aggregation_rule"]}"  # majority_50 | consensus_75 | unanimity_100

BASELINE = BaselineSpec(
    or_minutes={config["or_minutes"]},
    opening_time="09:30:00",
    direction="long",
    one_trade_per_day=True,
    entry_buffer_ticks=2,
    stop_buffer_ticks=2,
    target_multiple=2.0,
    vwap_confirmation=True,
    vwap_column="continuous_session_vwap",
    time_exit="16:00:00",
    account_size_usd=50_000.0,
    risk_per_trade_pct=1.5,
    entry_on_next_open=True,
)

GRID = SearchGrid(
    atr_periods=tuple(ATR_PERIODS),
    q_lows_pct=tuple(Q_LOW_VALUES),
    q_highs_pct=tuple(Q_HIGH_VALUES),
    aggregation_rules=("majority_50", "consensus_75", "unanimity_100"),
)

print("SYMBOL =", SYMBOL)
print("DATASET_PATH =", DATASET_PATH)
print("AGGREGATION_RULE =", AGGREGATION_RULE)
print("N_SUBSIGNALS_MAX =", len(ATR_PERIODS) * len(Q_LOW_VALUES) * len(Q_HIGH_VALUES))
"""


def _analysis_cell() -> str:
    return """bundle = build_notebook_bundle(
    symbol=SYMBOL,
    baseline=BASELINE,
    grid=GRID,
    aggregation_rule=AGGREGATION_RULE,
    is_fraction=IS_FRACTION,
    dataset_path=DATASET_PATH,
)

analysis = bundle["analysis"]
point_results = analysis.point_results.copy()
ensemble_results = analysis.ensemble_results.copy()
ensemble_curve = bundle["ensemble_curve"].copy()
ensemble_curve_is = bundle["ensemble_curve_is"].copy()
ensemble_curve_oos = bundle["ensemble_curve_oos"].copy()
benchmark_curve = bundle["benchmark_curve"].copy()
selected_ensemble = bundle["selected_ensemble"]

best_cell = analysis.best_cell
best_ensemble = analysis.best_ensemble
robust_cell = analysis.robust_cell

print("Instrument spec from config =", analysis.instrument_spec)
print("Sessions analysed =", len(analysis.all_sessions))
print("IS sessions =", len(analysis.is_sessions))
print("OOS sessions =", len(analysis.oos_sessions))
print("Best cell =", f"ATR {int(best_cell['atr_period'])}, q{int(best_cell['q_low_pct'])}/q{int(best_cell['q_high_pct'])}")
print("Best ensemble =", best_ensemble["aggregation_rule"])
print("Selected ensemble overall Sharpe =", round(float(selected_ensemble.get("overall_sharpe", 0.0)), 3))
print("Selected ensemble OOS Sharpe =", round(float(selected_ensemble.get("oos_sharpe", 0.0)), 3))
"""


def _equity_cell() -> str:
    return """initial_capital = float(BASELINE.account_size_usd)

ens_ret = curve_total_return_pct(ensemble_curve, initial_capital)
ens_dd_pct = curve_max_drawdown_pct(ensemble_curve)
ens_pf = float(selected_ensemble.get("overall_profit_factor", 0.0))
ens_sh = float(selected_ensemble.get("overall_sharpe", 0.0))
ens_exp = float(selected_ensemble.get("overall_expectancy", 0.0))

oos_ret = curve_total_return_pct(ensemble_curve_oos, initial_capital)
oos_dd_pct = curve_max_drawdown_pct(ensemble_curve_oos)
oos_pf = float(selected_ensemble.get("oos_profit_factor", 0.0))
oos_sh = float(selected_ensemble.get("oos_sharpe", 0.0))
oos_exp = float(selected_ensemble.get("oos_expectancy", 0.0))

bench_ret = curve_total_return_pct(benchmark_curve, initial_capital)
bench_dd_pct = curve_max_drawdown_pct(benchmark_curve)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.70, 0.30],
    subplot_titles=("Equity Curve (USD)", "Drawdown (%)"),
)

fig.add_trace(
    go.Scatter(
        x=ensemble_curve["timestamp"],
        y=ensemble_curve["equity"],
        mode="lines",
        name=f"Ensemble Full Sample | Ret {ens_ret:.1f}% | Sharpe {ens_sh:.2f} | PF {ens_pf:.2f} | Exp {ens_exp:.1f}",
        line=dict(width=3.0, color="#22c55e"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=ensemble_curve["timestamp"],
        y=ensemble_curve["drawdown_pct"],
        mode="lines",
        name="DD Ensemble",
        showlegend=False,
        line=dict(width=1.7, color="#22c55e", dash="dot"),
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=ensemble_curve_oos["timestamp"],
        y=ensemble_curve_oos["equity"],
        mode="lines",
        name=f"Ensemble OOS Only | Ret {oos_ret:.1f}% | Sharpe {oos_sh:.2f} | PF {oos_pf:.2f} | Exp {oos_exp:.1f}",
        line=dict(width=2.4, color="#f59e0b"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=ensemble_curve_oos["timestamp"],
        y=ensemble_curve_oos["drawdown_pct"],
        mode="lines",
        name="DD Ensemble OOS",
        showlegend=False,
        line=dict(width=1.5, color="#f59e0b", dash="dot"),
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=benchmark_curve["timestamp"],
        y=benchmark_curve["equity"],
        mode="lines",
        name=f"Buy&Hold | Ret {bench_ret:.1f}% | MaxDD {bench_dd_pct:.1f}%",
        line=dict(width=2.6, color="#38bdf8"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=benchmark_curve["timestamp"],
        y=benchmark_curve["drawdown_pct"],
        mode="lines",
        name="DD Buy&Hold",
        showlegend=False,
        line=dict(width=1.5, color="#38bdf8", dash="dot"),
    ),
    row=2,
    col=1,
)

fig.update_layout(
    template="plotly_dark",
    width=1800,
    height=950,
    title=f"{SYMBOL} Ensemble vs Buy&Hold",
    legend=dict(orientation="h", yanchor="bottom", y=-0.24, xanchor="left", x=0.0),
    margin=dict(l=70, r=40, t=90, b=140),
)
fig.update_yaxes(title_text="Equity (USD)", row=1, col=1)
fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
fig.update_xaxes(title_text="Time", row=2, col=1)
fig.show()

display(Markdown(build_scope_readout_markdown(
    full_curve=ensemble_curve,
    oos_curve=ensemble_curve_oos,
    initial_capital=initial_capital,
    full_label="Full-sample ensemble curve",
    oos_label="OOS-only ensemble curve",
)))
"""


def _heatmap_cell() -> str:
    return """heat_src = point_results.copy()

def _pivot_metric(metric: str) -> pd.DataFrame:
    return (
        heat_src.pivot_table(index="atr_period", columns="pair", values=metric, aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )

heatmaps = [
    ("oos_sharpe", "OOS Sharpe", "RdYlGn"),
    ("oos_profit_factor", "OOS Profit Factor", "RdYlGn"),
    ("oos_net_pnl", "OOS Net PnL", "RdYlGn"),
    ("oos_max_drawdown_abs", "OOS |Max Drawdown|", "RdYlGn"),
    ("oos_composite_score", "OOS Composite Score", "RdYlGn"),
]

fig_heat = make_subplots(
    rows=3,
    cols=2,
    horizontal_spacing=0.07,
    vertical_spacing=0.11,
    subplot_titles=[item[1] for item in heatmaps] + [""],
)

positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)]
for (metric, title, scale), (row, col) in zip(heatmaps, positions):
    pivot = _pivot_metric(metric)
    fig_heat.add_trace(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=scale,
            colorbar=dict(title=title, len=0.22),
        ),
        row=row,
        col=col,
    )

fig_heat.update_layout(
    title=f"{SYMBOL} Parameter Heatmaps",
    width=1850,
    height=1250,
    template="plotly_dark",
)
fig_heat.update_xaxes(title_text="Quantile pair", tickangle=45)
fig_heat.update_yaxes(title_text="ATR period")
fig_heat.show()

display(Markdown(
    "### Quick summary\\n"
    f"- Best ensemble: **{best_ensemble['aggregation_rule']}**\\n"
    f"- Best point cell: **ATR {int(best_cell['atr_period'])} | q{int(best_cell['q_low_pct'])}/q{int(best_cell['q_high_pct'])}**\\n"
    f"- Most robust point cell: **ATR {int(robust_cell['atr_period'])} | q{int(robust_cell['q_low_pct'])}/q{int(robust_cell['q_high_pct'])}**"
))
"""


def _summary_cell() -> str:
    return """kpi = build_selected_ensemble_kpi_frame(selected_ensemble)
display(kpi)

display(Markdown("### Aggregation rules"))
display(
    ensemble_results[
        [
            "aggregation_rule",
            "oos_composite_score",
            "oos_sharpe",
            "oos_profit_factor",
            "oos_expectancy",
            "oos_return_over_drawdown",
            "oos_nb_trades",
            "ensemble_robustness_score",
        ]
    ]
)

display(Markdown("### Top point cells (OOS)"))
display(
    point_results[
        [
            "atr_period",
            "q_low_pct",
            "q_high_pct",
            "oos_composite_score",
            "oos_sharpe",
            "oos_profit_factor",
            "oos_expectancy",
            "oos_return_over_drawdown",
            "local_robustness_score",
        ]
    ].head(15)
)

display(Markdown("### Final readout"))
display(Markdown(
    f"- **Best ensemble**: `{best_ensemble['aggregation_rule']}`\\n"
    f"- **Best cell**: `ATR {int(best_cell['atr_period'])} | q{int(best_cell['q_low_pct'])}/q{int(best_cell['q_high_pct'])}`\\n"
    f"- **Selected ensemble overall Sharpe / OOS Sharpe**: `{float(selected_ensemble.get('overall_sharpe', 0.0)):.2f} / {float(selected_ensemble.get('oos_sharpe', 0.0)):.2f}`\\n"
    f"- **Robust cluster count**: `{len(analysis.robust_clusters)}`"
))
"""


def build_notebook(symbol: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(_title(symbol)),
        nbf.v4.new_code_cell(_imports_cell()),
        nbf.v4.new_markdown_cell("## 1) Parameters (edit here)"),
        nbf.v4.new_code_cell(_parameter_cell(symbol)),
        nbf.v4.new_markdown_cell("## 2) Build data, baseline signals, and ensemble bundle"),
        nbf.v4.new_code_cell(_analysis_cell()),
        nbf.v4.new_markdown_cell("## 3) Backtest ensemble + Buy and Hold benchmark"),
        nbf.v4.new_code_cell(_equity_cell()),
        nbf.v4.new_markdown_cell("## 4) Heatmaps"),
        nbf.v4.new_code_cell(_heatmap_cell()),
        nbf.v4.new_markdown_cell("## 5) KPI table and final summary"),
        nbf.v4.new_code_cell(_summary_cell()),
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.10"}
    return nb


def write_notebook(symbol: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook(symbol)
    with output_path.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ORB ensemble client notebooks.")
    parser.add_argument("--symbols", nargs="*", default=["MES", "M2K", "MGC"])
    args = parser.parse_args()

    for symbol in args.symbols:
        name = f"orb_{str(symbol).upper()}_final_ensemble_validation.ipynb"
        output = NOTEBOOKS_DIR / name
        write_notebook(str(symbol).upper(), output)
        dataset = resolve_processed_dataset(str(symbol).upper())
        print(f"{output} <- {dataset.name}")


if __name__ == "__main__":
    main()
