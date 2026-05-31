"""Build a comparison notebook for MNQ ORB sizing_3state vs retained final ORB."""

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
RESEARCH_EXPORT_ROOT = REPO_ROOT / "export" / "orb_research_campaign"
NOTEBOOKS_ROOT = REPO_ROOT / "notebooks"

DEFAULT_VARIANT_NAME = "sizing_3state_realized_vol_ratio_15_60"
DEFAULT_RETAINED_CONFIG_NAME = "full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate"
DEFAULT_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "orb_MNQ_sizing_vs_retained_comparison.ipynb"
DEFAULT_EXECUTED_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "orb_MNQ_sizing_vs_retained_comparison.executed.ipynb"


def find_latest_export(prefix: str) -> Path:
    candidates = [path for path in DATA_EXPORTS_ROOT.glob(f"{prefix}_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No export folder found for prefix {prefix!r} under {DATA_EXPORTS_ROOT}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _title_cell() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """# MNQ ORB - Comparatif `sizing_3state` vs ORB retenue

Ce notebook compare les deux versions ORB qui ont servi a deux lectures differentes du repo:

- `sizing_3state`: la version la plus poussee pour la lecture prop / survivabilite.
- `retained final`: la version ORB retenue dans la campagne de recherche generale.

Le but ici est simple: remettre les deux sur des graphiques comparables, avec le meme esprit de charting que le notebook d'ensemble.
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


def fmt_money(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.1f} USD"


def fmt_pct_from_ratio(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100.0:.1f}%"


def fmt_float(value, digits=3):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_curve_from_daily(daily, initial_balance):
    out = daily.copy()
    out["session_date"] = pd.to_datetime(out["session_date"])
    out = out.sort_values("session_date").reset_index(drop=True)
    out["daily_pnl_usd"] = pd.to_numeric(out["daily_pnl_usd"], errors="coerce").fillna(0.0)
    out["equity"] = initial_balance + out["daily_pnl_usd"].cumsum()
    out["peak_equity"] = out["equity"].cummax()
    out["drawdown_usd"] = out["equity"] - out["peak_equity"]
    out["drawdown_pct"] = (out["equity"] / out["peak_equity"] - 1.0) * 100.0
    out["timestamp"] = out["session_date"]
    return out


def build_curve_from_equity_points(curve_df):
    out = curve_df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out["equity"] = pd.to_numeric(out["equity"], errors="coerce")
    out = out.dropna(subset=["equity"]).reset_index(drop=True)
    out["peak_equity"] = out["equity"].cummax()
    out["drawdown_usd"] = out["equity"] - out["peak_equity"]
    out["drawdown_pct"] = (out["equity"] / out["peak_equity"] - 1.0) * 100.0
    out["session_date"] = out["timestamp"].dt.normalize()
    return out


def rebase_curve_segment(curve_df, start_timestamp, initial_balance):
    curve = curve_df.sort_values("timestamp").reset_index(drop=True).copy()
    start_ts = pd.Timestamp(start_timestamp)
    curve = curve.loc[curve["timestamp"] >= start_ts].copy()
    if curve.empty:
        raise ValueError("No rows available after requested start timestamp.")

    previous_rows = curve_df.loc[curve_df["timestamp"] < start_ts].sort_values("timestamp")
    prior_equity = float(previous_rows["equity"].iloc[-1]) if not previous_rows.empty else initial_balance

    curve["equity"] = initial_balance + (curve["equity"] - prior_equity)
    curve["peak_equity"] = curve["equity"].cummax()
    curve["drawdown_usd"] = curve["equity"] - curve["peak_equity"]
    curve["drawdown_pct"] = (curve["equity"] / curve["peak_equity"] - 1.0) * 100.0
    curve["session_date"] = curve["timestamp"].dt.normalize()
    return curve.reset_index(drop=True)


def curve_to_daily_pnl(curve_df):
    curve = curve_df.sort_values("timestamp").reset_index(drop=True).copy()
    curve["prev_equity"] = curve["equity"].shift(1)
    curve["pnl_increment"] = curve["equity"] - curve["prev_equity"]
    curve.loc[curve["prev_equity"].isna(), "pnl_increment"] = curve.loc[curve["prev_equity"].isna(), "equity"] - 50000.0
    curve["session_date"] = curve["timestamp"].dt.normalize()
    daily = curve.groupby("session_date", as_index=False)["pnl_increment"].sum().rename(columns={"pnl_increment": "daily_pnl_usd"})
    daily["session_date"] = pd.to_datetime(daily["session_date"])
    return daily
"""
    )


def _parameter_cell(
    regime_export_root: Path,
    prop_export_root: Path,
    retained_config_name: str,
) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        f"""REGIME_EXPORT_ROOT = ROOT / r"{regime_export_root.relative_to(REPO_ROOT)}"
PROP_EXPORT_ROOT = ROOT / r"{prop_export_root.relative_to(REPO_ROOT)}"
RESEARCH_EXPORT_ROOT = ROOT / r"{RESEARCH_EXPORT_ROOT.relative_to(REPO_ROOT)}"
VARIANT_NAME = "{DEFAULT_VARIANT_NAME}"
RETAINED_CONFIG_NAME = "{retained_config_name}"
INITIAL_BALANCE_USD = 50_000.0

required_paths = {{
    "regime_export_root": REGIME_EXPORT_ROOT,
    "prop_export_root": PROP_EXPORT_ROOT,
    "research_export_root": RESEARCH_EXPORT_ROOT,
    "summary_variants": REGIME_EXPORT_ROOT / "summary_variants.csv",
    "variant_metrics": REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "metrics_by_scope.csv",
    "variant_daily": REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "daily_results.csv",
    "variant_controls": REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "controls.csv",
    "regime_mapping": REGIME_EXPORT_ROOT / "regime_state_mappings.csv",
    "retained_results": RESEARCH_EXPORT_ROOT / "full_reopt_results.csv",
    "retained_curve": RESEARCH_EXPORT_ROOT / "charts" / "equity_curve__full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate.csv",
    "campaign_report": RESEARCH_EXPORT_ROOT / "campaign_report.md",
    "prop_final_report": PROP_EXPORT_ROOT / "final_report.md",
    "prop_final_verdict": PROP_EXPORT_ROOT / "final_verdict.json",
}}

missing = [name for name, path in required_paths.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"Fichiers manquants pour le notebook: {{missing}}")

print("REGIME_EXPORT_ROOT =", REGIME_EXPORT_ROOT)
print("PROP_EXPORT_ROOT   =", PROP_EXPORT_ROOT)
print("RESEARCH_EXPORT    =", RESEARCH_EXPORT_ROOT)
print("VARIANT_NAME       =", VARIANT_NAME)
print("RETAINED_CONFIG    =", RETAINED_CONFIG_NAME)
"""
    )


def _load_data_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """regime_metadata = json.loads((REGIME_EXPORT_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
prop_verdict = json.loads((PROP_EXPORT_ROOT / "final_verdict.json").read_text(encoding="utf-8"))

summary_variants = pd.read_csv(REGIME_EXPORT_ROOT / "summary_variants.csv")
variant_metrics = pd.read_csv(REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "metrics_by_scope.csv")
variant_daily = pd.read_csv(REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "daily_results.csv", parse_dates=["session_date"])
variant_controls = pd.read_csv(REGIME_EXPORT_ROOT / "variants" / VARIANT_NAME / "controls.csv", parse_dates=["session_date"])
regime_mapping = pd.read_csv(REGIME_EXPORT_ROOT / "regime_state_mappings.csv")

retained_results = pd.read_csv(RESEARCH_EXPORT_ROOT / "full_reopt_results.csv")
retained_curve_raw = pd.read_csv(RESEARCH_EXPORT_ROOT / "charts" / "equity_curve__full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate.csv")

variant_row = summary_variants.loc[summary_variants["variant_name"] == VARIANT_NAME].iloc[0]
oos_start_date = pd.to_datetime(variant_controls.loc[variant_controls["phase"] == "oos", "session_date"].min())

retained_row = retained_results.loc[
    (retained_results["name"] == RETAINED_CONFIG_NAME)
    & (retained_results["compression_mode"] == "weak_close")
    & (retained_results["dynamic_mode"] == "noise_area_gate")
    & (retained_results["exit_mode"] == "baseline")
    & (retained_results["noise_lookback"] == 30)
].iloc[0]

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
bucket_map["effective_risk_per_trade_pct"] = bucket_map["risk_multiplier"] * float(regime_metadata["spec"]["baseline"]["risk_per_trade_pct"])

variant_curve = build_curve_from_daily(variant_daily, INITIAL_BALANCE_USD)
variant_curve_oos = build_curve_from_daily(variant_daily.loc[variant_daily["session_date"] >= oos_start_date], INITIAL_BALANCE_USD)

retained_curve = build_curve_from_equity_points(retained_curve_raw)
retained_curve_oos = rebase_curve_segment(retained_curve, oos_start_date, INITIAL_BALANCE_USD)

variant_daily_oos = variant_daily.loc[variant_daily["session_date"] >= oos_start_date, ["session_date", "daily_pnl_usd"]].copy()
retained_daily_full = curve_to_daily_pnl(retained_curve)
retained_daily_oos = retained_daily_full.loc[retained_daily_full["session_date"] >= oos_start_date].copy()

retained_config = json.loads(retained_row["config_json"])
display(Markdown(f"**OOS start date shared for the comparison:** `{oos_start_date.date()}`"))
"""
    )


def _summary_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 1. Lecture Rapide

Avant de lire les courbes, il faut garder la logique de decision en tete:

- `sizing_3state` = overlay de risque / survivabilite sur une base ORB nominale.
- `retained final` = config ORB finalement retenue dans la campagne de recherche generale.
- Les deux ne repondent donc pas exactement au meme objectif, meme si la comparaison visuelle reste utile.
"""
    )


def _summary_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """summary_lines = [
    "### Resume net",
    f"- `sizing_3state` lit le repo sous un angle prop: OOS net `{fmt_money(variant_row['oos_net_pnl'])}`, Sharpe `{fmt_float(variant_row['oos_sharpe'])}`, maxDD `{fmt_money(variant_row['oos_max_drawdown'])}`.",
    f"- `retained final` est la config ORB retenue dans la campagne generale: OOS net `{fmt_money(retained_row['oos_net_pnl'])}`, Sharpe `{fmt_float(retained_row['oos_sharpe_ratio'])}`, maxDD `{fmt_money(retained_row['oos_max_drawdown'])}`.",
    f"- Verdict prop recharge: meilleure variante de challenge = `{prop_verdict['recommended_launch_variant']}` avec profil `{prop_verdict['recommended_launch_risk_profile']}`.",
    "- Lecture pratique: `sizing_3state` si tu privilegies la survivabilite prop, `retained final` si tu veux relire la config ORB officiellement retenue dans la recherche du repo.",
]

display(Markdown("\\n".join(summary_lines)))
"""
    )


def _metrics_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 2. Tableau Comparatif

On met les scopes `full / is / oos` cote a cote avec les memes colonnes simples.
"""
    )


def _metrics_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """variant_metrics_view = variant_metrics.rename(
    columns={
        "scope": "scope",
        "net_pnl": "net_pnl_usd",
        "sharpe": "sharpe",
        "profit_factor": "profit_factor",
        "max_drawdown": "max_drawdown_usd",
        "n_trades": "n_trades",
        "pct_days_traded": "pct_days_traded",
        "worst_day": "worst_day_usd",
    }
)[["scope", "net_pnl_usd", "sharpe", "profit_factor", "max_drawdown_usd", "n_trades", "pct_days_traded", "worst_day_usd"]].copy()
variant_metrics_view["strategy"] = "sizing_3state"

retained_metrics_view = pd.DataFrame(
    [
        {
            "strategy": "retained_final",
            "scope": "full",
            "net_pnl_usd": retained_row["overall_net_pnl"],
            "sharpe": retained_row["overall_sharpe_ratio"],
            "profit_factor": retained_row["overall_profit_factor"],
            "max_drawdown_usd": retained_row["overall_max_drawdown"],
            "n_trades": retained_row["overall_nb_trades"],
            "pct_days_traded": retained_row["overall_pct_days_traded"],
            "worst_day_usd": retained_row["overall_worst_day"],
        },
        {
            "strategy": "retained_final",
            "scope": "is",
            "net_pnl_usd": retained_row["is_net_pnl"],
            "sharpe": retained_row["is_sharpe_ratio"],
            "profit_factor": retained_row["is_profit_factor"],
            "max_drawdown_usd": retained_row["is_max_drawdown"],
            "n_trades": retained_row["is_nb_trades"],
            "pct_days_traded": retained_row["is_pct_days_traded"],
            "worst_day_usd": retained_row["is_worst_day"],
        },
        {
            "strategy": "retained_final",
            "scope": "oos",
            "net_pnl_usd": retained_row["oos_net_pnl"],
            "sharpe": retained_row["oos_sharpe_ratio"],
            "profit_factor": retained_row["oos_profit_factor"],
            "max_drawdown_usd": retained_row["oos_max_drawdown"],
            "n_trades": retained_row["oos_nb_trades"],
            "pct_days_traded": retained_row["oos_pct_days_traded"],
            "worst_day_usd": retained_row["oos_worst_day"],
        },
    ]
)

comparison_metrics = pd.concat([variant_metrics_view, retained_metrics_view], ignore_index=True)
comparison_metrics["net_pnl_usd"] = comparison_metrics["net_pnl_usd"].map(lambda v: round(float(v), 1))
comparison_metrics["sharpe"] = comparison_metrics["sharpe"].map(lambda v: round(float(v), 3))
comparison_metrics["profit_factor"] = comparison_metrics["profit_factor"].map(lambda v: round(float(v), 3))
comparison_metrics["max_drawdown_usd"] = comparison_metrics["max_drawdown_usd"].map(lambda v: round(float(v), 1))
comparison_metrics["pct_days_traded"] = comparison_metrics["pct_days_traded"].map(lambda v: round(float(v) * 100.0, 2))
comparison_metrics["worst_day_usd"] = comparison_metrics["worst_day_usd"].map(lambda v: round(float(v), 1))

display(comparison_metrics.sort_values(["scope", "strategy"]).reset_index(drop=True))
"""
    )


def _equity_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 3. Equity / Drawdown

On reprend le meme genre de lecture que dans le notebook d'ensemble:

- courbe full sample,
- drawdown full sample,
- courbe OOS only,
- drawdown OOS only.
"""
    )


def _equity_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Full sample equity",
        "Full sample drawdown %",
        "OOS only equity",
        "OOS only drawdown %",
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.10,
)

for name, curve, color in [
    ("Sizing 3-state full", variant_curve, "#16a34a"),
    ("Retained final full", retained_curve, "#2563eb"),
]:
    fig.add_trace(
        go.Scatter(x=curve["timestamp"], y=curve["equity"], mode="lines", name=name, line=dict(width=2, color=color)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=curve["timestamp"], y=curve["drawdown_pct"], mode="lines", name=f"{name} DD", showlegend=False, line=dict(width=1.8, color=color, dash="dot")),
        row=1,
        col=2,
    )

for name, curve, color in [
    ("Sizing 3-state oos", variant_curve_oos, "#16a34a"),
    ("Retained final oos", retained_curve_oos, "#2563eb"),
]:
    fig.add_trace(
        go.Scatter(x=curve["timestamp"], y=curve["equity"], mode="lines", name=name, line=dict(width=2, color=color)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=curve["timestamp"], y=curve["drawdown_pct"], mode="lines", name=f"{name} DD", showlegend=False, line=dict(width=1.8, color=color, dash="dot")),
        row=2,
        col=2,
    )

fig.update_yaxes(title_text="Equity (USD)", row=1, col=1)
fig.update_yaxes(title_text="Drawdown %", row=1, col=2)
fig.update_yaxes(title_text="Equity (USD)", row=2, col=1)
fig.update_yaxes(title_text="Drawdown %", row=2, col=2)
fig.update_layout(
    height=900,
    width=1400,
    title="MNQ ORB comparison - sizing_3state vs retained final",
    legend=dict(orientation="h", y=1.08),
)
fig.show()
"""
    )


def _params_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 4. Parametrage

Ici on regarde rapidement ce que chaque version change vraiment.
"""
    )


def _params_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("### Bucket map - sizing_3state"))
display(bucket_map)

retained_param_rows = [
    {"section": "entry", "parameter": "or_minutes", "value": retained_config["baseline_entry"]["or_minutes"]},
    {"section": "entry", "parameter": "direction", "value": retained_config["baseline_entry"]["direction"]},
    {"section": "entry", "parameter": "entry_buffer_ticks", "value": retained_config["baseline_entry"]["entry_buffer_ticks"]},
    {"section": "entry", "parameter": "stop_buffer_ticks", "value": retained_config["baseline_entry"]["stop_buffer_ticks"]},
    {"section": "entry", "parameter": "target_multiple", "value": retained_config["baseline_entry"]["target_multiple"]},
    {"section": "entry", "parameter": "vwap_confirmation", "value": retained_config["baseline_entry"]["vwap_confirmation"]},
    {"section": "ensemble", "parameter": "atr_window", "value": retained_config["baseline_ensemble"]["atr_window"]},
    {"section": "ensemble", "parameter": "vote_threshold", "value": retained_config["baseline_ensemble"]["vote_threshold"]},
    {"section": "overlay", "parameter": "compression_mode", "value": retained_config["compression"]["mode"]},
    {"section": "overlay", "parameter": "compression_usage", "value": retained_config["compression"]["usage"]},
    {"section": "overlay", "parameter": "dynamic_mode", "value": retained_config["dynamic_threshold"]["mode"]},
    {"section": "overlay", "parameter": "noise_lookback", "value": retained_config["dynamic_threshold"]["noise_lookback"]},
    {"section": "overlay", "parameter": "noise_vm", "value": retained_config["dynamic_threshold"]["noise_vm"]},
]

display(Markdown("### Param snapshot - retained final"))
display(pd.DataFrame(retained_param_rows))
"""
    )


def _distribution_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 5. Distribution OOS des daily pnl

Ce chart donne une lecture rapide du profil de jour OOS sur les deux variantes.
"""
    )


def _distribution_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """oos_distribution = pd.concat(
    [
        variant_daily_oos.assign(strategy="sizing_3state"),
        retained_daily_oos.assign(strategy="retained_final"),
    ],
    ignore_index=True,
)
oos_distribution = oos_distribution.loc[oos_distribution["daily_pnl_usd"].fillna(0.0) != 0.0].copy()

fig = px.histogram(
    oos_distribution,
    x="daily_pnl_usd",
    color="strategy",
    nbins=60,
    barmode="overlay",
    opacity=0.55,
    title="Distribution OOS des daily pnl non nuls",
    labels={"daily_pnl_usd": "Daily pnl (USD)", "strategy": "Strategy"},
)
fig.show()
"""
    )


def _conclusion_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 6. Conclusion

La derniere cellule reformule la difference de role entre les deux notebooks.
"""
    )


def _conclusion_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """conclusion_lines = [
    "### Verdict simple",
    f"- `sizing_3state` est le plus fort ici si ton filtre principal est la survivabilite prop: OOS Sharpe `{fmt_float(variant_row['oos_sharpe'])}`, net `{fmt_money(variant_row['oos_net_pnl'])}`, maxDD `{fmt_money(variant_row['oos_max_drawdown'])}`.",
    f"- `retained final` est la version ORB officiellement retenue dans la recherche repo: OOS Sharpe `{fmt_float(retained_row['oos_sharpe_ratio'])}`, net `{fmt_money(retained_row['oos_net_pnl'])}`, maxDD `{fmt_money(retained_row['oos_max_drawdown'])}`.",
    "- Si tu veux une lecture client prop / challenge: repars du notebook `sizing_3state`.",
    "- Si tu veux la sleeve ORB finale du portefeuille recherche: repars du notebook `mnq_orb_retained_final`.",
]

display(Markdown("\\n".join(conclusion_lines)))
"""
    )


def _appendix_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 7. Sources

Le notebook reste branche sur des exports et notebooks explicites.
"""
    )


def _appendix_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """source_paths = pd.DataFrame(
    [
        {"name": "regime_export_root", "path": str(REGIME_EXPORT_ROOT)},
        {"name": "prop_export_root", "path": str(PROP_EXPORT_ROOT)},
        {"name": "research_export_root", "path": str(RESEARCH_EXPORT_ROOT)},
        {"name": "sizing_notebook", "path": str(ROOT / "notebooks" / "orb_MNQ_sizing_3state_client.ipynb")},
        {"name": "retained_notebook", "path": str(ROOT / "notebooks" / "finals" / "mnq_orb_retained_final.ipynb")},
    ]
)
display(source_paths)
"""
    )


def build_notebook(regime_export_root: Path, prop_export_root: Path) -> nbf.NotebookNode:
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
        _parameter_cell(regime_export_root, prop_export_root, DEFAULT_RETAINED_CONFIG_NAME),
        _load_data_cell(),
        _summary_markdown(),
        _summary_cell(),
        _metrics_markdown(),
        _metrics_cell(),
        _equity_markdown(),
        _equity_cell(),
        _params_markdown(),
        _params_cell(),
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
        help="Audited sizing/regime export root to load.",
    )
    parser.add_argument(
        "--prop-export-root",
        type=Path,
        default=find_latest_export("mnq_orb_prop_challenge_readiness"),
        help="Prop challenge readiness export root to load.",
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
    notebook = build_notebook(args.regime_export_root, args.prop_export_root)
    output_path = write_notebook(notebook, args.output)
    print(f"Notebook written to {output_path}")

    if args.execute:
        executed_path = execute_notebook(output_path, args.executed_output, timeout_seconds=args.timeout_seconds)
        print(f"Executed notebook written to {executed_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
