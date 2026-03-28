"""Build a client-facing notebook for the MNQ ORB sizing_3state variant."""

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

DEFAULT_VARIANT_NAME = "sizing_3state_realized_vol_ratio_15_60"
DEFAULT_BASELINE_NAME = "nominal"
DEFAULT_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "orb_MNQ_sizing_3state_client.ipynb"
DEFAULT_EXECUTED_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "orb_MNQ_sizing_3state_client.executed.ipynb"


def find_latest_export(prefix: str) -> Path:
    candidates = [path for path in EXPORTS_ROOT.glob(f"{prefix}_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No export folder found for prefix {prefix!r} under {EXPORTS_ROOT}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _title_cell() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """# MNQ ORB Client Notebook - Variante `sizing_3state`

Ce notebook est un support client centré sur la variante `sizing_3state_realized_vol_ratio_15_60`.

- **Alpha inchangé** : la brique alpha reste le baseline MNQ ORB ensemble officiel.
- **Overlay ajouté** : seul le sizing varie, via un découpage en 3 états sur `realized_vol_ratio_15_60`.
- **Objectif** : documenter les paramètres exacts, la logique de sizing, les métriques historiques auditées et la lecture TopstepX 50K.
- **Philosophie** : ce notebook ne refait pas la recherche. Il relit et met en forme des exports déjà audités du repo.
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

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import Markdown, display

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)


def fmt_money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.1f} USD"


def fmt_pct_from_ratio(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100.0:.1f}%"


def fmt_float(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_curve(daily: pd.DataFrame, initial_balance: float = 50_000.0) -> pd.DataFrame:
    out = daily.copy()
    out["session_date"] = pd.to_datetime(out["session_date"])
    out = out.sort_values("session_date").reset_index(drop=True)
    out["daily_pnl_usd"] = pd.to_numeric(out["daily_pnl_usd"], errors="coerce").fillna(0.0)
    out["equity"] = initial_balance + out["daily_pnl_usd"].cumsum()
    out["peak_equity"] = out["equity"].cummax()
    out["drawdown_usd"] = out["equity"] - out["peak_equity"]
    out["drawdown_pct"] = (out["equity"] / out["peak_equity"] - 1.0) * 100.0
    return out


def compact_metrics_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.loc[:, columns].copy()
    money_cols = [col for col in out.columns if "pnl" in col or "drawdown" in col or "worst_day" in col]
    pct_cols = [col for col in out.columns if "retention" in col or "coverage" in col]
    float_cols = [col for col in out.columns if col not in money_cols and col not in pct_cols and col != "variant_name"]
    for col in money_cols:
        out[col] = out[col].map(lambda value: round(float(value), 1) if pd.notna(value) else value)
    for col in pct_cols:
        out[col] = out[col].map(lambda value: round(float(value) * 100.0, 2) if pd.notna(value) else value)
    for col in float_cols:
        if col == "variant_name":
            continue
        out[col] = out[col].map(lambda value: round(float(value), 3) if pd.notna(value) else value)
    return out
"""
    )


def _parameter_cell(regime_export_root: Path, topstep_export_root: Path) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        f"""REGIME_EXPORT_ROOT = ROOT / r"{regime_export_root.relative_to(REPO_ROOT)}"
TOPSTEP_EXPORT_ROOT = ROOT / r"{topstep_export_root.relative_to(REPO_ROOT)}"
VARIANT_NAME = "{DEFAULT_VARIANT_NAME}"
BASELINE_NAME = "{DEFAULT_BASELINE_NAME}"
TOPSTEP_RULESETS = ("topstepx_50k_main_35d", "topstepx_50k_extended_60d")

required_paths = {{
    "regime_export_root": REGIME_EXPORT_ROOT,
    "topstep_export_root": TOPSTEP_EXPORT_ROOT,
    "summary_variants": REGIME_EXPORT_ROOT / "summary_variants.csv",
    "variant_metrics": REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "metrics_by_scope.csv",
    "variant_daily": REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "daily_results.csv",
    "variant_trades": REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "trades.csv",
    "variant_controls": REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "controls.csv",
    "baseline_daily": REGIME_EXPORT_ROOT / "variants" / BASELINE_NAME / "daily_results.csv",
    "regime_mapping": REGIME_EXPORT_ROOT / "regime_state_mappings.csv",
    "topstep_comparison": TOPSTEP_EXPORT_ROOT / "comparison_table.csv",
    "topstep_bootstrap": TOPSTEP_EXPORT_ROOT / "bootstrap_summary.csv",
    "topstep_rolling": TOPSTEP_EXPORT_ROOT / "rolling_start_summary.csv",
}}

missing = [name for name, path in required_paths.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"Fichiers manquants pour le notebook: {{missing}}")

print("REGIME_EXPORT_ROOT =", REGIME_EXPORT_ROOT)
print("TOPSTEP_EXPORT_ROOT =", TOPSTEP_EXPORT_ROOT)
print("VARIANT_NAME =", VARIANT_NAME)
print("BASELINE_NAME =", BASELINE_NAME)
"""
    )


def _load_data_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """regime_metadata = json.loads((REGIME_EXPORT_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
topstep_metadata = json.loads((TOPSTEP_EXPORT_ROOT / "run_metadata.json").read_text(encoding="utf-8"))

summary_variants = pd.read_csv(REGIME_EXPORT_ROOT / "summary_variants.csv")
variant_metrics = pd.read_csv(REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "metrics_by_scope.csv")
variant_daily = pd.read_csv(REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "daily_results.csv", parse_dates=["session_date"])
variant_trades = pd.read_csv(
    REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "trades.csv",
    parse_dates=["session_date", "entry_time", "exit_time"],
)
variant_controls = pd.read_csv(
    REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "controls.csv",
    parse_dates=["session_date"],
)
baseline_daily = pd.read_csv(
    REGIME_EXPORT_ROOT / "variants" / BASELINE_NAME / "daily_results.csv",
    parse_dates=["session_date"],
)
regime_mapping = pd.read_csv(REGIME_EXPORT_ROOT / "regime_state_mappings.csv")
topstep_comparison = pd.read_csv(TOPSTEP_EXPORT_ROOT / "comparison_table.csv")
topstep_bootstrap = pd.read_csv(TOPSTEP_EXPORT_ROOT / "bootstrap_summary.csv")
topstep_rolling = pd.read_csv(TOPSTEP_EXPORT_ROOT / "rolling_start_summary.csv")

variant_row = summary_variants.loc[summary_variants["variant_name"] == VARIANT_NAME].iloc[0]
baseline_row = summary_variants.loc[summary_variants["variant_name"] == BASELINE_NAME].iloc[0]
variant_params = json.loads(variant_row["parameters_json"])
baseline_config = regime_metadata["spec"]["baseline"]
initial_balance = float(baseline_config["account_size_usd"])
oos_start_date = variant_controls.loc[variant_controls["phase"] == "oos", "session_date"].min()

bucket_map = (
    regime_mapping.loc[
        (regime_mapping["variant_name"] == VARIANT_NAME)
        & (regime_mapping["feature_name"] == "realized_vol_ratio_15_60"),
        ["bucket_label", "lower_bound", "upper_bound", "risk_multiplier", "oos_n_obs", "oos_net_pnl", "oos_sharpe", "oos_max_drawdown"],
    ]
    .drop_duplicates()
    .sort_values(["risk_multiplier", "bucket_label"])
    .reset_index(drop=True)
)
bucket_map["effective_risk_per_trade_pct"] = bucket_map["risk_multiplier"] * float(baseline_config["risk_per_trade_pct"])

variant_curve = build_curve(variant_daily, initial_balance)
baseline_curve = build_curve(baseline_daily, initial_balance)
variant_curve_oos = build_curve(variant_daily.loc[variant_daily["session_date"] >= oos_start_date], initial_balance)
baseline_curve_oos = build_curve(baseline_daily.loc[baseline_daily["session_date"] >= oos_start_date], initial_balance)

trade_buckets = variant_trades.merge(
    variant_controls[["session_date", "phase", "bucket_label"]],
    on="session_date",
    how="left",
)
daily_buckets = variant_daily.merge(
    variant_controls[["session_date", "phase", "bucket_label", "risk_multiplier"]],
    on="session_date",
    how="left",
)

topstep_tidy = pd.concat(
    [
        topstep_rolling.assign(method="historical_rolling"),
        topstep_bootstrap.assign(method="bootstrap"),
    ],
    ignore_index=True,
)
topstep_tidy = topstep_tidy.loc[topstep_tidy["variant_name"].isin([BASELINE_NAME, VARIANT_NAME])].copy()

display(Markdown(f"**OOS start date:** `{oos_start_date.date()}`"))
display(Markdown(f"**Primary verdict from Topstep export:** `{topstep_metadata['overall_verdict']}`"))
"""
    )


def _quick_read_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 1. Lecture Rapide

Cette section donne la lecture la plus courte possible de la variante avant d'entrer dans les tableaux.

- `sizing_3state` ne change pas le signal ORB. Il change seulement l'exposition.
- Le feature de contrôle est `realized_vol_ratio_15_60`.
- La logique est discrète et lisible : **0.50x / 1.00x / 0.75x** selon le bucket.
- Le bon usage client de cette variante est **prop-oriented / risk-shaped**, pas “nouvelle alpha”.
"""
    )


def _quick_read_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """variant_oos_retention = float(variant_row["oos_net_pnl_retention_vs_nominal"])
variant_oos_sharpe = float(variant_row["oos_sharpe"])
baseline_oos_sharpe = float(baseline_row["oos_sharpe"])
variant_oos_dd = float(variant_row["oos_max_drawdown"])
baseline_oos_dd = float(baseline_row["oos_max_drawdown"])

quick_lines = [
    "### Synthese executive",
    f"- La variante conserve **{variant_oos_retention * 100.0:.1f}%** du pnl OOS du nominal.",
    f"- Le Sharpe OOS passe de **{baseline_oos_sharpe:.3f}** a **{variant_oos_sharpe:.3f}**.",
    f"- Le max drawdown OOS passe de **{baseline_oos_dd:,.1f} USD** a **{variant_oos_dd:,.1f} USD**.",
    f"- Le notebook Topstep reference conclut: **{topstep_metadata['overall_verdict']}**.",
]

display(Markdown("\\n".join(quick_lines)))
"""
    )


def _parameters_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 2. Paramètres Exactes

Le but ici est que le lecteur puisse relier sans ambiguïté la version présentée à ses paramètres exacts:

- baseline alpha,
- overlay de sizing,
- buckets de régime,
- cadre TopstepX 50K utilisé ensuite.
"""
    )


def _parameters_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """baseline_df = pd.DataFrame(
    [{"parameter": key, "value": value} for key, value in baseline_config.items()]
)

variant_overview_df = pd.DataFrame(
    [
        {"parameter": "variant_name", "value": VARIANT_NAME},
        {"parameter": "family", "value": variant_row["family"]},
        {"parameter": "feature_name", "value": variant_row["feature_name"]},
        {"parameter": "bucketing", "value": variant_row["bucketing"]},
        {"parameter": "calibration_scope", "value": variant_row["calibration_scope"]},
        {"parameter": "description", "value": variant_row["description"]},
        {"parameter": "note", "value": variant_row["note"]},
        {"parameter": "parameters_json", "value": variant_row["parameters_json"]},
    ]
)

bucket_display = bucket_map.copy()
bucket_display["lower_bound"] = bucket_display["lower_bound"].round(6)
bucket_display["upper_bound"] = bucket_display["upper_bound"].round(6)
bucket_display["effective_risk_per_trade_pct"] = bucket_display["effective_risk_per_trade_pct"].round(3)
bucket_display["oos_net_pnl"] = bucket_display["oos_net_pnl"].round(1)
bucket_display["oos_sharpe"] = bucket_display["oos_sharpe"].round(3)
bucket_display["oos_max_drawdown"] = bucket_display["oos_max_drawdown"].round(1)

topstep_rules_df = pd.DataFrame(topstep_metadata["rulesets"])

display(Markdown("### Baseline ORB officielle"))
display(baseline_df)

display(Markdown("### Overlay sizing_3state"))
display(variant_overview_df)

display(Markdown("### Bucket mapping et exposition effective"))
display(bucket_display)

display(Markdown("### Rulesets Topstep charges dans le notebook"))
display(topstep_rules_df)
"""
    )


def _performance_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 3. Performance Auditée

La lecture ci-dessous compare le nominal officiel au `sizing_3state` avec le minimum de bruit nécessaire:

- lecture globale,
- lecture IS / OOS,
- retention vs baseline,
- improvement de risque.
"""
    )


def _performance_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """comparison_columns = [
    "variant_name",
    "overall_net_pnl",
    "overall_sharpe",
    "overall_profit_factor",
    "overall_max_drawdown",
    "oos_net_pnl",
    "oos_sharpe",
    "oos_profit_factor",
    "oos_max_drawdown",
    "oos_net_pnl_retention_vs_nominal",
    "oos_sharpe_delta_vs_nominal",
    "oos_max_drawdown_improvement_vs_nominal",
]

variant_comparison = compact_metrics_table(
    summary_variants.loc[summary_variants["variant_name"].isin([BASELINE_NAME, VARIANT_NAME])].copy(),
    comparison_columns,
)

metrics_scope_display = variant_metrics.copy()
for col in ["net_pnl", "max_drawdown", "worst_day", "max_loss_limit_buffer_usd"]:
    if col in metrics_scope_display.columns:
        metrics_scope_display[col] = metrics_scope_display[col].round(1)
for col in ["sharpe", "sortino", "profit_factor", "expectancy"]:
    if col in metrics_scope_display.columns:
        metrics_scope_display[col] = metrics_scope_display[col].round(3)

display(Markdown("### Nominal vs sizing_3state"))
display(variant_comparison)

display(Markdown("### Metrics par scope pour sizing_3state"))
display(metrics_scope_display)
"""
    )


def _equity_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 4. Courbes de Capital

On regarde ici deux choses:

- la courbe full sample pour garder la continuité historique,
- la courbe OOS-only pour juger la variante là où la décision compte le plus.
"""
    )


def _equity_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """curve_summary = pd.DataFrame(
    [
        {
            "curve": "nominal_full",
            "final_pnl_usd": baseline_curve["equity"].iloc[-1] - initial_balance,
            "max_drawdown_usd": baseline_curve["drawdown_usd"].min(),
        },
        {
            "curve": "sizing_full",
            "final_pnl_usd": variant_curve["equity"].iloc[-1] - initial_balance,
            "max_drawdown_usd": variant_curve["drawdown_usd"].min(),
        },
        {
            "curve": "nominal_oos_only",
            "final_pnl_usd": baseline_curve_oos["equity"].iloc[-1] - initial_balance,
            "max_drawdown_usd": baseline_curve_oos["drawdown_usd"].min(),
        },
        {
            "curve": "sizing_oos_only",
            "final_pnl_usd": variant_curve_oos["equity"].iloc[-1] - initial_balance,
            "max_drawdown_usd": variant_curve_oos["drawdown_usd"].min(),
        },
    ]
)
curve_summary["final_pnl_usd"] = curve_summary["final_pnl_usd"].round(1)
curve_summary["max_drawdown_usd"] = curve_summary["max_drawdown_usd"].round(1)
display(curve_summary)

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Full sample - Equity",
        "OOS only - Equity",
        "Full sample - Drawdown",
        "OOS only - Drawdown",
    ),
    horizontal_spacing=0.1,
    vertical_spacing=0.12,
)

for curve_name, curve_df, color in [
    ("Nominal", baseline_curve, "#2563eb"),
    ("Sizing 3-state", variant_curve, "#16a34a"),
]:
    fig.add_trace(
        go.Scatter(x=curve_df["session_date"], y=curve_df["equity"], mode="lines", name=f"{curve_name} full", line=dict(width=2.5, color=color)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=curve_df["session_date"], y=curve_df["drawdown_usd"], mode="lines", name=f"{curve_name} full DD", showlegend=False, line=dict(width=1.8, color=color, dash="dot")),
        row=2,
        col=1,
    )

for curve_name, curve_df, color in [
    ("Nominal", baseline_curve_oos, "#2563eb"),
    ("Sizing 3-state", variant_curve_oos, "#16a34a"),
]:
    fig.add_trace(
        go.Scatter(x=curve_df["session_date"], y=curve_df["equity"], mode="lines", name=f"{curve_name} oos", line=dict(width=2.5, color=color)),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=curve_df["session_date"], y=curve_df["drawdown_usd"], mode="lines", name=f"{curve_name} oos DD", showlegend=False, line=dict(width=1.8, color=color, dash="dot")),
        row=2,
        col=2,
    )

fig.update_layout(height=820, width=1200, title="Nominal vs sizing_3state - equity et drawdown", legend=dict(orientation="h", y=1.08))
fig.update_yaxes(title_text="USD", row=1, col=1)
fig.update_yaxes(title_text="USD", row=1, col=2)
fig.update_yaxes(title_text="DD USD", row=2, col=1)
fig.update_yaxes(title_text="DD USD", row=2, col=2)
fig.show()
"""
    )


def _sizing_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 5. Logique de Sizing

Cette partie montre comment la variante se comporte en pratique:

- combien de trades passent dans chaque bucket,
- quelle exposition effective est prise,
- quelle contribution PnL vient de chaque bucket.
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

bucket_trade_stats["avg_quantity"] = bucket_trade_stats["avg_quantity"].round(2)
bucket_trade_stats["avg_risk_per_trade_pct"] = bucket_trade_stats["avg_risk_per_trade_pct"].round(3)
bucket_trade_stats["avg_actual_risk_usd"] = bucket_trade_stats["avg_actual_risk_usd"].round(1)
bucket_trade_stats["total_net_pnl_usd"] = bucket_trade_stats["total_net_pnl_usd"].round(1)
bucket_trade_stats["avg_net_pnl_usd"] = bucket_trade_stats["avg_net_pnl_usd"].round(1)

display(bucket_trade_stats)

fig = px.bar(
    bucket_trade_stats,
    x="bucket_label",
    y="total_net_pnl_usd",
    color="risk_multiplier",
    barmode="group",
    title="Contribution PnL par bucket de sizing",
    labels={"total_net_pnl_usd": "Net PnL total (USD)", "bucket_label": "Bucket regime", "risk_multiplier": "Risk multiplier"},
)
fig.show()
"""
    )


def _distribution_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 6. Distribution de PnL

Ici, l'idée est de montrer la géométrie quotidienne de la variante:

- dispersion des journées tradées,
- quantiles simples,
- meilleurs et pires jours.
"""
    )


def _distribution_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """traded_daily = daily_buckets.loc[daily_buckets["daily_trade_count"] > 0].copy()
traded_daily_oos = traded_daily.loc[traded_daily["session_date"] >= oos_start_date].copy()

daily_quantiles = traded_daily_oos["daily_pnl_usd"].quantile([0.05, 0.25, 0.50, 0.75, 0.95]).rename("daily_pnl_usd")
display(Markdown("### Quantiles OOS des journees tradees"))
display(daily_quantiles.to_frame())

best_worst_days = pd.concat(
    [
        traded_daily_oos.nlargest(5, "daily_pnl_usd"),
        traded_daily_oos.nsmallest(5, "daily_pnl_usd"),
    ],
    ignore_index=True,
)
best_worst_days = best_worst_days.loc[:, ["session_date", "daily_pnl_usd", "bucket_label", "risk_multiplier", "daily_trade_count"]]
best_worst_days["session_date"] = best_worst_days["session_date"].dt.date
best_worst_days["daily_pnl_usd"] = best_worst_days["daily_pnl_usd"].round(1)
display(Markdown("### Meilleurs / pires jours OOS"))
display(best_worst_days)

hist_df = pd.concat(
    [
        baseline_daily.loc[baseline_daily["session_date"] >= oos_start_date, ["session_date", "daily_pnl_usd"]].assign(variant="nominal"),
        variant_daily.loc[variant_daily["session_date"] >= oos_start_date, ["session_date", "daily_pnl_usd"]].assign(variant="sizing_3state"),
    ],
    ignore_index=True,
)
hist_df = hist_df.loc[hist_df["daily_pnl_usd"] != 0].copy()

fig = px.histogram(
    hist_df,
    x="daily_pnl_usd",
    color="variant",
    marginal="box",
    nbins=60,
    barmode="overlay",
    opacity=0.55,
    title="Distribution OOS des daily pnl non nuls",
    labels={"daily_pnl_usd": "Daily pnl (USD)", "variant": "Variant"},
)
fig.show()
"""
    )


def _topstep_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 7. Lecture TopstepX 50K

Le notebook recharge la campagne Topstep ciblée déjà faite dans le repo:

- Starting balance: `50,000`
- Profit target: `+3,000`
- Trailing Maximum Loss Limit: `2,000`
- Consistency target: meilleur jour gagnant strictement `< 50%` des profits au moment du pass

Le point à retenir ici est la hiérarchie **nominal vs sizing_3state** dans ce ruleset précis.
"""
    )


def _topstep_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """topstep_display = topstep_tidy.loc[
    topstep_tidy["ruleset_name"].isin(TOPSTEP_RULESETS),
    [
        "method",
        "ruleset_name",
        "variant_name",
        "pass_rate",
        "fail_rate",
        "expire_rate",
        "median_days_to_pass",
        "median_days_to_fail",
        "trailing_mll_breach_rate",
        "economic_target_without_immediate_validation_rate",
        "delayed_pass_after_inconsistency_rate",
        "failed_after_economic_target_rate",
    ],
].copy()

for col in [
    "pass_rate",
    "fail_rate",
    "expire_rate",
    "trailing_mll_breach_rate",
    "economic_target_without_immediate_validation_rate",
    "delayed_pass_after_inconsistency_rate",
    "failed_after_economic_target_rate",
]:
    topstep_display[col] = (topstep_display[col] * 100.0).round(2)

display(Markdown("### Topstep summary tidy"))
display(topstep_display.sort_values(["method", "ruleset_name", "variant_name"]))

display(Markdown("### Table de comparaison nominal vs sizing_3state"))
display(topstep_comparison)

consistency_active = bool(topstep_tidy["economic_target_without_immediate_validation_rate"].fillna(0.0).max() > 0.0)
display(Markdown(f"**Consistency active dans l'echantillon:** `{consistency_active}`"))

fig = px.bar(
    topstep_tidy,
    x="ruleset_name",
    y="pass_rate",
    color="variant_name",
    barmode="group",
    facet_col="method",
    title="TopstepX 50K - pass rate par methode et horizon",
    labels={"pass_rate": "Pass rate", "ruleset_name": "Ruleset", "variant_name": "Variant"},
)
fig.update_yaxes(tickformat=".0%")
fig.for_each_annotation(lambda ann: ann.update(text=ann.text.replace("method=", "")))
fig.show()
"""
    )


def _conclusion_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 8. Conclusion Client

Cette dernière cellule reformule la décision en langage simple, en séparant bien:

- la lecture historique observée,
- la lecture prop / survivabilité,
- la vraie portée de la consistency target.
"""
    )


def _conclusion_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """main_rolling = topstep_tidy.loc[
    (topstep_tidy["method"] == "historical_rolling") & (topstep_tidy["ruleset_name"] == "topstepx_50k_main_35d")
].set_index("variant_name")
main_bootstrap = topstep_tidy.loc[
    (topstep_tidy["method"] == "bootstrap") & (topstep_tidy["ruleset_name"] == "topstepx_50k_main_35d")
].set_index("variant_name")

final_lines = [
    "### Verdict simple",
    f"- `sizing_3state` conserve environ **{variant_oos_retention * 100.0:.1f}%** du pnl OOS du nominal avec une amelioration nette du profil de drawdown.",
    f"- En **historical rolling Topstep 35 jours**, le nominal garde l'avantage en vitesse / pass rate: **{main_rolling.loc[BASELINE_NAME, 'pass_rate'] * 100.0:.1f}%** vs **{main_rolling.loc[VARIANT_NAME, 'pass_rate'] * 100.0:.1f}%**.",
    f"- En **bootstrap Topstep 35 jours**, `sizing_3state` devient plus defendable en survivabilite: **{main_bootstrap.loc[VARIANT_NAME, 'pass_rate'] * 100.0:.1f}%** de pass pour **{main_bootstrap.loc[VARIANT_NAME, 'fail_rate'] * 100.0:.1f}%** de fail.",
    f"- Dans l'echantillon recharge ici, la **consistency target ne bind pas**: le changement de hierarchie ne vient donc pas de la consistency, mais de la path dependency entre chemins historiques reels et chemins resamples.",
    f"- Lecture client recommandee: **nominal pour passer vite sur trajectoire historique observee, sizing_3state pour une version plus prudente et plus prop-oriented quand on privilegie la survivabilite**.",
]

display(Markdown("\\n".join(final_lines)))
"""
    )


def _appendix_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 9. Appendice - Sources

Le notebook est volontairement raccordé à des sources explicites pour rester auditable.
"""
    )


def _appendix_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """source_paths = pd.DataFrame(
    [
        {"name": "regime_export_root", "path": str(REGIME_EXPORT_ROOT)},
        {"name": "topstep_export_root", "path": str(TOPSTEP_EXPORT_ROOT)},
        {"name": "dataset_path", "path": regime_metadata["dataset_path"]},
        {"name": "campaign_summary_md", "path": str(REGIME_EXPORT_ROOT / "campaign_summary.md")},
        {"name": "topstep_summary_md", "path": str(TOPSTEP_EXPORT_ROOT / "topstep_50k_summary.md")},
    ]
)
display(source_paths)
"""
    )


def build_notebook(regime_export_root: Path, topstep_export_root: Path) -> nbf.NotebookNode:
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
        _parameter_cell(regime_export_root, topstep_export_root),
        _load_data_cell(),
        _quick_read_markdown(),
        _quick_read_cell(),
        _parameters_markdown(),
        _parameters_cell(),
        _performance_markdown(),
        _performance_cell(),
        _equity_markdown(),
        _equity_cell(),
        _sizing_markdown(),
        _sizing_cell(),
        _distribution_markdown(),
        _distribution_cell(),
        _topstep_markdown(),
        _topstep_cell(),
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
        "--topstep-export-root",
        type=Path,
        default=find_latest_export("mnq_orb_topstep_50k_simulation"),
        help="Topstep 50K simulation export root to load.",
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
    notebook = build_notebook(args.regime_export_root, args.topstep_export_root)
    output_path = write_notebook(notebook, args.output)
    print(f"Notebook written to {output_path}")

    if args.execute:
        executed_path = execute_notebook(output_path, args.executed_output, timeout_seconds=args.timeout_seconds)
        print(f"Executed notebook written to {executed_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
