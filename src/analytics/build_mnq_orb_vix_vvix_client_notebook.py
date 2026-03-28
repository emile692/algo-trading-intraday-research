"""Build a client-facing notebook for the MNQ ORB VIX/VVIX validation campaign."""

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

DEFAULT_EXPORT_PREFIX = "mnq_orb_vix_vvix_validation"
DEFAULT_VARIANT_NAME = "filter_drop_low__vvix_pct_63_t1"
DEFAULT_BASELINE_NAME = "baseline_fixed_nominal_atr"
DEFAULT_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "orb_MNQ_vix_vvix_client.ipynb"
DEFAULT_EXECUTED_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "orb_MNQ_vix_vvix_client.executed.ipynb"


def find_latest_export(prefix: str) -> Path:
    candidates = [path for path in EXPORTS_ROOT.glob(f"{prefix}_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No export folder found for prefix {prefix!r} under {EXPORTS_ROOT}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _title_cell() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """# MNQ ORB Client Notebook - filtre `filter_drop_low__vvix_pct_63_t1`

Ce notebook est un support client centre sur la variante `filter_drop_low__vvix_pct_63_t1`.

- **Baseline inchangee** : on garde l'ORB MNQ actuel, son filtre ATR structurel, et les hypotheses d'execution/couts.
- **Nouveau bloc uniquement** : le seul ajout est un veto regime base sur `vvix_pct_63_t1`.
- **Objectif** : visualiser rapidement ce que le filtre change sur la courbe, le drawdown et la selection des jours.
- **Philosophie** : le notebook relit les exports audites de la campagne, il ne refait pas la recherche.
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
from plotly.subplots import make_subplots
from IPython.display import Markdown, display

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)


def coerce_bool(series: pd.Series) -> pd.Series:
    if str(series.dtype) == "bool":
        return series
    mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    return mapped.fillna(False)


def fmt_money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.1f} USD"


def fmt_pct_ratio(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100.0:.{digits}f}%"


def fmt_pct_points(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100.0:.{digits}f} pts"


def fmt_float(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_curve(frame: pd.DataFrame, pnl_column: str, initial_balance: float = 50_000.0) -> pd.DataFrame:
    out = frame.copy()
    out["session_date"] = pd.to_datetime(out["session_date"])
    out = out.sort_values("session_date").reset_index(drop=True)
    out[pnl_column] = pd.to_numeric(out[pnl_column], errors="coerce").fillna(0.0)
    out["equity"] = initial_balance + out[pnl_column].cumsum()
    out["peak_equity"] = out["equity"].cummax()
    out["drawdown_usd"] = out["equity"] - out["peak_equity"]
    out["drawdown_pct"] = (out["equity"] / out["peak_equity"] - 1.0) * 100.0
    return out
"""
    )


def _parameter_cell(export_root: Path) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        f"""EXPORT_ROOT = ROOT / r"{export_root.relative_to(REPO_ROOT)}"
VARIANT_NAME = "{DEFAULT_VARIANT_NAME}"
BASELINE_NAME = "{DEFAULT_BASELINE_NAME}"

required_paths = {{
    "export_root": EXPORT_ROOT,
    "run_metadata": EXPORT_ROOT / "run_metadata.json",
    "final_verdict": EXPORT_ROOT / "final_verdict.json",
    "validation_summary": EXPORT_ROOT / "validation_summary.csv",
    "regime_summary": EXPORT_ROOT / "regime_summary.csv",
    "interaction_summary": EXPORT_ROOT / "interaction_summary.csv",
    "trade_features": EXPORT_ROOT / "selected_trade_vix_vvix_features.csv",
    "baseline_metrics": EXPORT_ROOT / "variants" / BASELINE_NAME / "metrics_by_scope.csv",
    "baseline_daily": EXPORT_ROOT / "variants" / BASELINE_NAME / "daily_results.csv",
    "variant_metrics": EXPORT_ROOT / "variants" / VARIANT_NAME / "metrics_by_scope.csv",
    "variant_daily": EXPORT_ROOT / "variants" / VARIANT_NAME / "daily_results.csv",
    "variant_controls": EXPORT_ROOT / "variants" / VARIANT_NAME / "controls.csv",
}}

missing = [name for name, path in required_paths.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"Fichiers manquants pour le notebook: {{missing}}")

print("EXPORT_ROOT =", EXPORT_ROOT)
print("VARIANT_NAME =", VARIANT_NAME)
print("BASELINE_NAME =", BASELINE_NAME)
"""
    )


def _load_data_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """run_metadata = json.loads((EXPORT_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
final_verdict = json.loads((EXPORT_ROOT / "final_verdict.json").read_text(encoding="utf-8"))

validation_summary = pd.read_csv(EXPORT_ROOT / "validation_summary.csv")
regime_summary = pd.read_csv(EXPORT_ROOT / "regime_summary.csv")
interaction_summary = pd.read_csv(EXPORT_ROOT / "interaction_summary.csv")
trade_features = pd.read_csv(EXPORT_ROOT / "selected_trade_vix_vvix_features.csv", parse_dates=["session_date"])
baseline_metrics = pd.read_csv(EXPORT_ROOT / "variants" / BASELINE_NAME / "metrics_by_scope.csv")
baseline_daily = pd.read_csv(EXPORT_ROOT / "variants" / BASELINE_NAME / "daily_results.csv", parse_dates=["session_date"])
variant_metrics = pd.read_csv(EXPORT_ROOT / "variants" / VARIANT_NAME / "metrics_by_scope.csv")
variant_daily = pd.read_csv(EXPORT_ROOT / "variants" / VARIANT_NAME / "daily_results.csv", parse_dates=["session_date"])
variant_controls = pd.read_csv(EXPORT_ROOT / "variants" / VARIANT_NAME / "controls.csv", parse_dates=["session_date"])

variant_controls["selected"] = coerce_bool(variant_controls["selected"])
variant_controls["selected_by_baseline_atr"] = coerce_bool(variant_controls["selected_by_baseline_atr"])
variant_controls["skip_trade"] = coerce_bool(variant_controls["skip_trade"])
trade_features["selected_by_baseline_atr"] = coerce_bool(trade_features["selected_by_baseline_atr"])

baseline_row = validation_summary.loc[validation_summary["variant_name"] == BASELINE_NAME].iloc[0]
variant_row = validation_summary.loc[validation_summary["variant_name"] == VARIANT_NAME].iloc[0]
baseline_metrics_by_scope = baseline_metrics.set_index("scope")
variant_metrics_by_scope = variant_metrics.set_index("scope")

initial_balance = float(run_metadata["spec"]["initial_capital_usd"])

session_frame = (
    variant_controls[
        [
            "session_date",
            "phase",
            "breakout_side",
            "breakout_timing_bucket",
            "feature_value",
            "bucket_label",
            "selected",
            "skip_trade",
        ]
    ]
    .drop_duplicates()
    .sort_values("session_date")
    .reset_index(drop=True)
)
session_frame = session_frame.merge(
    baseline_daily[["session_date", "daily_pnl_usd"]].rename(columns={"daily_pnl_usd": "baseline_pnl_usd"}),
    on="session_date",
    how="left",
)
session_frame = session_frame.merge(
    variant_daily[["session_date", "daily_pnl_usd"]].rename(columns={"daily_pnl_usd": "variant_pnl_usd"}),
    on="session_date",
    how="left",
)
session_frame["selected"] = session_frame["selected"].fillna(False)
session_frame["skip_trade"] = session_frame["skip_trade"].fillna(~session_frame["selected"])
session_frame["baseline_pnl_usd"] = pd.to_numeric(session_frame["baseline_pnl_usd"], errors="coerce").fillna(0.0)
session_frame["variant_pnl_usd"] = pd.to_numeric(session_frame["variant_pnl_usd"], errors="coerce").fillna(0.0)
session_frame["selection_label"] = session_frame["selected"].map({True: "kept_by_filter", False: "skipped_by_filter"})

oos_start_date = session_frame.loc[session_frame["phase"] == "oos", "session_date"].min()

baseline_curve_full = build_curve(
    session_frame[["session_date", "baseline_pnl_usd"]].rename(columns={"baseline_pnl_usd": "daily_pnl_usd"}),
    pnl_column="daily_pnl_usd",
    initial_balance=initial_balance,
)
variant_curve_full = build_curve(
    session_frame[["session_date", "variant_pnl_usd"]].rename(columns={"variant_pnl_usd": "daily_pnl_usd"}),
    pnl_column="daily_pnl_usd",
    initial_balance=initial_balance,
)
baseline_curve_oos = build_curve(
    session_frame.loc[session_frame["phase"] == "oos", ["session_date", "baseline_pnl_usd"]].rename(columns={"baseline_pnl_usd": "daily_pnl_usd"}),
    pnl_column="daily_pnl_usd",
    initial_balance=initial_balance,
)
variant_curve_oos = build_curve(
    session_frame.loc[session_frame["phase"] == "oos", ["session_date", "variant_pnl_usd"]].rename(columns={"variant_pnl_usd": "daily_pnl_usd"}),
    pnl_column="daily_pnl_usd",
    initial_balance=initial_balance,
)

bucket_view = (
    regime_summary.loc[regime_summary["feature_name"] == "vvix_pct_63_t1"]
    .sort_values("bucket_position")
    .reset_index(drop=True)
)
low_upper = float(bucket_view.loc[bucket_view["bucket_label"] == "low", "upper_bound"].iloc[0])
mid_upper = float(bucket_view.loc[bucket_view["bucket_label"] == "mid", "upper_bound"].iloc[0])

interaction_view = (
    interaction_summary.loc[
        (interaction_summary["variant_name"] == VARIANT_NAME)
        & (interaction_summary["dimension"].isin(["breakout_side", "breakout_timing_bucket"]))
    ]
    .copy()
    .reset_index(drop=True)
)

coverage_summary = final_verdict["coverage_summary"]

display(Markdown(f"**OOS start date:** `{oos_start_date.date()}`"))
display(Markdown(f"**Campaign verdict:** `{final_verdict['best_variant_verdict']}`"))
"""
    )


def _quick_read_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 1. Lecture rapide

Avant les graphiques, voici la lecture la plus compacte du filtre.

- Il ne change pas l'alpha ORB.
- Il ne change pas les couts, le slippage, ni la logique ATR de reference.
- Il agit comme un **veto defensif** : on garde les jours `mid/high` sur `vvix_pct_63_t1`, on coupe les jours `low`.
- Le point important a visualiser est donc moins "plus de trades gagnants" que "meilleure selection des jours" et "courbe plus propre".
"""
    )


def _quick_read_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """summary_lines = [
    "### Synthese executive",
    f"- Variante suivie : `{VARIANT_NAME}`.",
    f"- Regle exacte : skip bucket `low` sur `vvix_pct_63_t1`, keep `mid/high`.",
    f"- Frontiere low/mid IS : `{fmt_pct_ratio(low_upper, digits=2)}` du percentile 63 jours.",
    f"- Frontiere mid/high IS : `{fmt_pct_ratio(mid_upper, digits=2)}` du percentile 63 jours.",
    f"- Trade coverage OOS : **{fmt_pct_ratio(final_verdict['best_variant_oos_trade_coverage_vs_baseline'])}**.",
    f"- PnL retention OOS : **{fmt_pct_ratio(final_verdict['best_variant_oos_net_pnl_retention_vs_baseline'])}**.",
    f"- Delta Sharpe OOS : **{fmt_float(final_verdict['best_variant_oos_sharpe_delta_vs_baseline'])}**.",
    f"- Delta Profit Factor OOS : **{fmt_float(final_verdict['best_variant_oos_profit_factor_delta_vs_baseline'])}**.",
    f"- Amelioration MaxDD OOS : **{fmt_pct_ratio(final_verdict['best_variant_oos_max_drawdown_improvement_vs_baseline'])}**.",
    f"- Lecture recherche : **{final_verdict['edge_character']}** via **{final_verdict['primary_mechanism']}**.",
]
display(Markdown("\\n".join(summary_lines)))
"""
    )


def _rule_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 2. Regle exact et comparaison de perimetre

On compare ici deux objets tres simples:

1. `baseline_fixed_nominal_atr`
2. `filter_drop_low__vvix_pct_63_t1`

La baseline garde son ORB, son ATR et son execution. La variante applique seulement un veto journalier base sur le bucket `vvix_pct_63_t1`.
"""
    )


def _comparison_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """scope_rows = []
for scope in ("overall", "is", "oos"):
    base = baseline_metrics_by_scope.loc[scope]
    variant = variant_metrics_by_scope.loc[scope]
    scope_rows.append(
        {
            "scope": scope.upper(),
            "baseline_net_pnl_usd": round(float(base["net_pnl"]), 1),
            "variant_net_pnl_usd": round(float(variant["net_pnl"]), 1),
            "baseline_sharpe": round(float(base["sharpe"]), 3),
            "variant_sharpe": round(float(variant["sharpe"]), 3),
            "baseline_profit_factor": round(float(base["profit_factor"]), 3),
            "variant_profit_factor": round(float(variant["profit_factor"]), 3),
            "baseline_expectancy_usd": round(float(base["expectancy"]), 1),
            "variant_expectancy_usd": round(float(variant["expectancy"]), 1),
            "baseline_max_drawdown_usd": round(float(base["max_drawdown"]), 1),
            "variant_max_drawdown_usd": round(float(variant["max_drawdown"]), 1),
            "baseline_n_trades": int(base["n_trades"]),
            "variant_n_trades": int(variant["n_trades"]),
        }
    )

scope_table = pd.DataFrame(scope_rows)
display(scope_table)

validation_snapshot = pd.DataFrame(
    [
        {
            "variant_name": VARIANT_NAME,
            "verdict": variant_row["verdict"],
            "block": variant_row["block"],
            "feature_name": variant_row["feature_name"],
            "oos_trade_coverage_pct": round(float(variant_row["oos_trade_coverage_vs_baseline"]) * 100.0, 2),
            "oos_day_coverage_pct": round(float(variant_row["oos_day_coverage_vs_baseline"]) * 100.0, 2),
            "oos_pnl_retention_pct": round(float(variant_row["oos_net_pnl_retention_vs_baseline"]) * 100.0, 2),
            "oos_sharpe_delta": round(float(variant_row["oos_sharpe_delta_vs_baseline"]), 3),
            "oos_profit_factor_delta": round(float(variant_row["oos_profit_factor_delta_vs_baseline"]), 3),
            "oos_expectancy_delta_usd": round(float(variant_row["oos_expectancy_delta_vs_baseline"]), 2),
            "oos_max_drawdown_improvement_pct": round(float(variant_row["oos_max_drawdown_improvement_vs_baseline"]) * 100.0, 2),
        }
    ]
)
display(validation_snapshot)
"""
    )


def _equity_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 3. Courbe de capital et drawdown

La lecture ici doit repondre a une question simple: est-ce que le filtre nettoie la courbe sans casser le PnL?

- La courbe **full sample** est calculee sur la fenetre de couverture de la campagne.
- La courbe **OOS only** permet de voir si le benefice reste visible hors calibration.
- Les jours sautes par le filtre sont comptabilises a `0` sur la variante, ce qui rend la comparaison visuelle directe.
"""
    )


def _equity_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """vline_x = oos_start_date.to_pydatetime()

full_fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("Equity curve - full sample", "Drawdown - full sample"),
)
full_fig.add_trace(
    go.Scatter(x=baseline_curve_full["session_date"], y=baseline_curve_full["equity"], name="baseline", line=dict(width=2)),
    row=1,
    col=1,
)
full_fig.add_trace(
    go.Scatter(x=variant_curve_full["session_date"], y=variant_curve_full["equity"], name=VARIANT_NAME, line=dict(width=2)),
    row=1,
    col=1,
)
full_fig.add_trace(
    go.Scatter(x=baseline_curve_full["session_date"], y=baseline_curve_full["drawdown_usd"], name="baseline_drawdown", line=dict(width=2), showlegend=False),
    row=2,
    col=1,
)
full_fig.add_trace(
    go.Scatter(x=variant_curve_full["session_date"], y=variant_curve_full["drawdown_usd"], name="variant_drawdown", line=dict(width=2), showlegend=False),
    row=2,
    col=1,
)
full_fig.add_vline(x=vline_x, line_dash="dash", line_color="firebrick")
full_fig.add_annotation(x=vline_x, y=1.02, xref="x", yref="paper", text="OOS start", showarrow=False, xanchor="left")
full_fig.update_layout(height=850, template="plotly_white", title="Baseline vs filtre VVIX - full sample")
full_fig.update_yaxes(title_text="Equity (USD)", row=1, col=1)
full_fig.update_yaxes(title_text="Drawdown (USD)", row=2, col=1)
full_fig.show()

oos_fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("Equity curve - OOS only", "Drawdown - OOS only"),
)
oos_fig.add_trace(
    go.Scatter(x=baseline_curve_oos["session_date"], y=baseline_curve_oos["equity"], name="baseline", line=dict(width=2)),
    row=1,
    col=1,
)
oos_fig.add_trace(
    go.Scatter(x=variant_curve_oos["session_date"], y=variant_curve_oos["equity"], name=VARIANT_NAME, line=dict(width=2)),
    row=1,
    col=1,
)
oos_fig.add_trace(
    go.Scatter(x=baseline_curve_oos["session_date"], y=baseline_curve_oos["drawdown_usd"], name="baseline_drawdown", line=dict(width=2), showlegend=False),
    row=2,
    col=1,
)
oos_fig.add_trace(
    go.Scatter(x=variant_curve_oos["session_date"], y=variant_curve_oos["drawdown_usd"], name="variant_drawdown", line=dict(width=2), showlegend=False),
    row=2,
    col=1,
)
oos_fig.update_layout(height=850, template="plotly_white", title="Baseline vs filtre VVIX - OOS")
oos_fig.update_yaxes(title_text="Equity (USD)", row=1, col=1)
oos_fig.update_yaxes(title_text="Drawdown (USD)", row=2, col=1)
oos_fig.show()
"""
    )


def _selection_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 4. Mecanique de selection des jours

Le filtre ne fait qu'une chose: il coupe les jours ou `vvix_pct_63_t1` est dans le bucket `low`.

Cette section sert a voir:

- combien de jours sont gardes vs coupes,
- ou se situe la frontiere de bucket,
- ce que representent ces jours en PnL baseline.
"""
    )


def _selection_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """selection_summary = (
    session_frame.groupby(["phase", "bucket_label", "selection_label"], dropna=False)
    .agg(
        n_sessions=("session_date", "count"),
        total_baseline_pnl_usd=("baseline_pnl_usd", "sum"),
        avg_baseline_pnl_usd=("baseline_pnl_usd", "mean"),
    )
    .reset_index()
)
selection_summary["total_baseline_pnl_usd"] = selection_summary["total_baseline_pnl_usd"].round(1)
selection_summary["avg_baseline_pnl_usd"] = selection_summary["avg_baseline_pnl_usd"].round(2)
display(selection_summary)

counts_fig = px.bar(
    selection_summary,
    x="bucket_label",
    y="n_sessions",
    color="selection_label",
    facet_col="phase",
    barmode="group",
    category_orders={"bucket_label": ["low", "mid", "high"], "phase": ["is", "oos"]},
    title="Sessions gardees/coupees par bucket VVIX percentile 63j",
    template="plotly_white",
)
counts_fig.show()

pnl_rollup = (
    session_frame.groupby(["phase", "selection_label"], dropna=False)
    .agg(
        n_sessions=("session_date", "count"),
        total_baseline_pnl_usd=("baseline_pnl_usd", "sum"),
        avg_baseline_pnl_usd=("baseline_pnl_usd", "mean"),
    )
    .reset_index()
)
pnl_rollup["total_baseline_pnl_usd"] = pnl_rollup["total_baseline_pnl_usd"].round(1)
pnl_rollup["avg_baseline_pnl_usd"] = pnl_rollup["avg_baseline_pnl_usd"].round(2)
display(pnl_rollup)

pnl_fig = px.bar(
    pnl_rollup,
    x="selection_label",
    y="total_baseline_pnl_usd",
    color="phase",
    barmode="group",
    title="PnL baseline represente par les jours gardes vs coupes",
    template="plotly_white",
)
pnl_fig.show()

hist_fig = px.histogram(
    session_frame,
    x="feature_value",
    color="selection_label",
    facet_col="phase",
    nbins=24,
    barmode="overlay",
    opacity=0.65,
    category_orders={"phase": ["is", "oos"]},
    title="Distribution de `vvix_pct_63_t1` et zone coupee par le filtre",
    template="plotly_white",
)
for col_index in (1, 2):
    hist_fig.add_vline(x=low_upper, line_dash="dash", line_color="firebrick", row=1, col=col_index)
    hist_fig.add_vline(x=mid_upper, line_dash="dot", line_color="gray", row=1, col=col_index)
hist_fig.update_xaxes(title_text="vvix_pct_63_t1")
hist_fig.show()
"""
    )


def _bucket_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 5. Diagnostic par bucket `vvix_pct_63_t1`

Le filtre utilise uniquement les buckets calibres en IS. On regarde ici si la hierarchie reste lisible en OOS.

L'idee n'est pas de sur-interpreter un seul bucket, mais de verifier que le bucket `low` est bien la zone la moins utile a conserver pour cette campagne.
"""
    )


def _bucket_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """bucket_display = bucket_view[
    [
        "bucket_label",
        "lower_bound",
        "upper_bound",
        "is_n_obs",
        "oos_n_obs",
        "is_expectancy",
        "oos_expectancy",
        "is_profit_factor",
        "oos_profit_factor",
        "is_hit_rate",
        "oos_hit_rate",
        "oos_net_pnl",
    ]
].copy()
bucket_display["lower_bound"] = bucket_display["lower_bound"].round(4)
bucket_display["upper_bound"] = bucket_display["upper_bound"].round(4)
bucket_display["is_expectancy"] = bucket_display["is_expectancy"].round(2)
bucket_display["oos_expectancy"] = bucket_display["oos_expectancy"].round(2)
bucket_display["is_profit_factor"] = bucket_display["is_profit_factor"].round(3)
bucket_display["oos_profit_factor"] = bucket_display["oos_profit_factor"].round(3)
bucket_display["is_hit_rate"] = (bucket_display["is_hit_rate"] * 100.0).round(2)
bucket_display["oos_hit_rate"] = (bucket_display["oos_hit_rate"] * 100.0).round(2)
bucket_display["oos_net_pnl"] = bucket_display["oos_net_pnl"].round(1)
display(bucket_display)

expectancy_long = bucket_view.melt(
    id_vars=["bucket_label"],
    value_vars=["is_expectancy", "oos_expectancy"],
    var_name="scope",
    value_name="expectancy",
)
expectancy_fig = px.bar(
    expectancy_long,
    x="bucket_label",
    y="expectancy",
    color="scope",
    barmode="group",
    category_orders={"bucket_label": ["low", "mid", "high"]},
    title="Expectancy par bucket VVIX percentile 63j",
    template="plotly_white",
)
expectancy_fig.show()

quality_fig = make_subplots(specs=[[{"secondary_y": True}]])
quality_fig.add_trace(
    go.Bar(
        x=bucket_view["bucket_label"],
        y=bucket_view["oos_net_pnl"],
        name="OOS net pnl",
    ),
    secondary_y=False,
)
quality_fig.add_trace(
    go.Scatter(
        x=bucket_view["bucket_label"],
        y=bucket_view["oos_profit_factor"],
        name="OOS profit factor",
        mode="lines+markers",
        line=dict(width=3),
    ),
    secondary_y=True,
)
quality_fig.update_layout(
    height=500,
    template="plotly_white",
    title="OOS net pnl et profit factor par bucket VVIX percentile 63j",
)
quality_fig.update_yaxes(title_text="OOS net pnl (USD)", secondary_y=False)
quality_fig.update_yaxes(title_text="OOS profit factor", secondary_y=True)
quality_fig.show()
"""
    )


def _interaction_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 6. Breakdown du filtre

Le notebook pousse un peu plus loin la lecture pour voir si le filtre aide surtout:

- les cassures haussieres ou baissieres,
- les timings `early/mid/late`.

Le but n'est pas de lancer une nouvelle optimisation, juste de voir ou le filtre semble le plus utile.
"""
    )


def _interaction_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """interaction_display = interaction_view[
    [
        "dimension",
        "bucket",
        "trade_coverage_vs_baseline",
        "expectancy_delta",
        "hit_rate_delta",
        "stop_hit_rate_delta",
    ]
].copy()
interaction_display["trade_coverage_vs_baseline"] = (interaction_display["trade_coverage_vs_baseline"] * 100.0).round(2)
interaction_display["expectancy_delta"] = interaction_display["expectancy_delta"].round(2)
interaction_display["hit_rate_delta"] = (interaction_display["hit_rate_delta"] * 100.0).round(2)
interaction_display["stop_hit_rate_delta"] = (interaction_display["stop_hit_rate_delta"] * 100.0).round(2)
display(interaction_display)

expectancy_delta_fig = px.bar(
    interaction_view,
    x="bucket",
    y="expectancy_delta",
    color="bucket",
    facet_col="dimension",
    title="Delta expectancy du filtre par breakout side et timing bucket",
    template="plotly_white",
)
expectancy_delta_fig.show()

coverage_fig = px.bar(
    interaction_view,
    x="bucket",
    y="trade_coverage_vs_baseline",
    color="bucket",
    facet_col="dimension",
    title="Trade coverage du filtre par breakout side et timing bucket",
    template="plotly_white",
)
coverage_fig.update_yaxes(tickformat=".0%")
coverage_fig.show()
"""
    )


def _conclusion_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 7. Conclusion client

La bonne lecture de ce filtre est plutot **controle de regime / filtrage defensif** qu'alpha autonome.

Si la visualisation confirme ta lecture, la suite logique reste bien la phase deja prevue:

`baseline ORB + ATR + meilleur filtre VIX/VVIX + sizing 3-state`
"""
    )


def _conclusion_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """conclusion_lines = [
    "### Verdict client net",
    f"- Le notebook confirme une lecture **{final_verdict['edge_character']}**.",
    f"- Le mecanisme principal est **{final_verdict['primary_mechanism']}**.",
    f"- OOS, la variante garde **{fmt_pct_ratio(final_verdict['best_variant_oos_net_pnl_retention_vs_baseline'])}** du PnL baseline pour **{fmt_pct_ratio(final_verdict['best_variant_oos_trade_coverage_vs_baseline'])}** des trades.",
    f"- Le Sharpe OOS gagne **{fmt_float(final_verdict['best_variant_oos_sharpe_delta_vs_baseline'])}** et le MaxDD s'ameliore de **{fmt_pct_ratio(final_verdict['best_variant_oos_max_drawdown_improvement_vs_baseline'])}**.",
    f"- Business read : le filtre semble utile comme surcouche defensive, pas comme re-ecriture de l'alpha.",
]
display(Markdown("\\n".join(conclusion_lines)))
"""
    )


def build_notebook(export_root: Path) -> nbf.NotebookNode:
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
        _parameter_cell(export_root),
        _load_data_cell(),
        _quick_read_markdown(),
        _quick_read_cell(),
        _rule_markdown(),
        _comparison_cell(),
        _equity_markdown(),
        _equity_cell(),
        _selection_markdown(),
        _selection_cell(),
        _bucket_markdown(),
        _bucket_cell(),
        _interaction_markdown(),
        _interaction_cell(),
        _conclusion_markdown(),
        _conclusion_cell(),
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
        "--export-root",
        type=Path,
        default=find_latest_export(DEFAULT_EXPORT_PREFIX),
        help="Audited VIX/VVIX validation export root to load.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_NOTEBOOK_PATH,
        help="Path where the notebook should be written.",
    )
    parser.add_argument(
        "--executed-output",
        type=Path,
        default=DEFAULT_EXECUTED_NOTEBOOK_PATH,
        help="Path where the executed notebook should be written if --execute is set.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the notebook after writing it.",
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
    notebook = build_notebook(args.export_root)
    output_path = write_notebook(notebook, args.output)
    print(f"Notebook written to {output_path}")

    if args.execute:
        executed_path = execute_notebook(output_path, args.executed_output, timeout_seconds=args.timeout_seconds)
        print(f"Executed notebook written to {executed_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
