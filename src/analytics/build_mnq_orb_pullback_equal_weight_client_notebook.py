"""Build a client notebook for equal-weight ORB + pullback MNQ blend."""

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

DEFAULT_ORB_EXPORT_ROOT = EXPORTS_ROOT / "mnq_orb_vix_vvix_validation_20260327_run"
DEFAULT_PULLBACK_EXPORT_ROOT = EXPORTS_ROOT / "volume_climax_pullback_mnq_risk_sizing_refinement_20260406_231223"
DEFAULT_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "mnq_orb_pullback_equal_weight_client.ipynb"
DEFAULT_EXECUTED_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "mnq_orb_pullback_equal_weight_client.executed.ipynb"


def _title_cell() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """# Full MNQ Client Notebook - ORB vs Pullback vs Benchmark

Ce notebook compare, sur MNQ uniquement:

- l'ORB MNQ de reference du repo en standalone,
- le pullback MNQ en standalone,
- le portefeuille equal weight ORB + pullback,
- le benchmark MNQ buy & hold.

Le parametrage client est centralise dans la cellule suivante: chemins d'exports, variantes, poids, capital initial et template graphique.
Les multiplicateurs de risque/levier sont egalement disponibles dans cette cellule pour tester rapidement un profil plus agressif sans relancer les campagnes.

Le blend est calcule de facon transparente:

- sur l'overlap commun audite entre les deux strategies,
- au niveau des rendements journaliers,
- avec rebalancing quotidien a poids fixes.
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
pd.set_option("display.width", 220)


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


def normalize_weights(weights):
    total = sum(float(weight) for weight in weights)
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return [float(weight) / total for weight in weights]


def validate_multiplier(value, name):
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")
    return value


def scale_daily_return(daily_return, multiplier, label):
    scaled = pd.to_numeric(daily_return, errors="coerce").fillna(0.0) * float(multiplier)
    if (scaled <= -1.0).any():
        worst = float(scaled.min())
        raise ValueError(
            f"{label}: leverage creates a daily return <= -100% (worst={worst:.2%}). "
            "Lower the relevant leverage multiplier."
        )
    return scaled


def build_sleeve_path(daily_pnl, initial_capital):
    prev = float(initial_capital)
    rets = []
    equities = []
    for pnl in pd.to_numeric(daily_pnl, errors="coerce").fillna(0.0):
        ret = float(pnl) / prev if prev else 0.0
        prev += float(pnl)
        rets.append(ret)
        equities.append(prev)
    return pd.DataFrame({"daily_return": rets, "equity": equities})


def build_return_path(daily_return, initial_capital, label="curve"):
    daily_return = pd.to_numeric(daily_return, errors="coerce").fillna(0.0).reset_index(drop=True)
    if (daily_return <= -1.0).any():
        worst = float(daily_return.min())
        raise ValueError(f"{label}: daily return <= -100% (worst={worst:.2%}).")
    equity = float(initial_capital) * (1.0 + daily_return).cumprod()
    peak = equity.cummax()
    drawdown_usd = equity - peak
    drawdown_pct = np.where(peak > 0, (equity / peak - 1.0) * 100.0, 0.0)
    daily_pnl_usd = equity.diff().fillna(equity.iloc[0] - float(initial_capital))
    return pd.DataFrame(
        {
            "daily_return": daily_return,
            "equity": equity,
            "daily_pnl_usd": daily_pnl_usd,
            "drawdown_usd": drawdown_usd,
            "drawdown_pct": drawdown_pct,
        }
    )


def curve_metrics(curve, initial_capital):
    ordered = curve.sort_values("session_date").reset_index(drop=True)
    daily_ret = pd.to_numeric(ordered["daily_return"], errors="coerce").fillna(0.0)
    equity = pd.to_numeric(ordered["equity"], errors="coerce").fillna(float(initial_capital))
    peak = equity.cummax()
    drawdown = equity - peak
    drawdown_pct = np.where(peak > 0, (equity / peak - 1.0) * 100.0, 0.0)
    start = ordered["session_date"].iloc[0]
    end = ordered["session_date"].iloc[-1]
    years = max(((end - start).days + 1) / 365.25, 1 / 365.25)
    vol = float(daily_ret.std(ddof=0) * math.sqrt(252.0) * 100.0) if len(daily_ret) > 1 else 0.0
    sharpe = float(daily_ret.mean() / daily_ret.std(ddof=0) * math.sqrt(252.0)) if len(daily_ret) > 1 and daily_ret.std(ddof=0) > 0 else 0.0
    downside = daily_ret[daily_ret < 0]
    downside_std = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) > 0 else 0.0
    sortino = float(daily_ret.mean() / downside_std * math.sqrt(252.0)) if downside_std > 0 else 0.0
    final_equity = float(equity.iloc[-1])
    cagr = float(((final_equity / float(initial_capital)) ** (1.0 / years) - 1.0) * 100.0) if final_equity > 0 else float("nan")
    pnl = ordered.get("daily_pnl_usd", equity.diff().fillna(equity.iloc[0] - float(initial_capital)))
    profit_factor = float(pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())) if (pnl < 0).any() else float("inf")
    return {
        "net_pnl_usd": float(final_equity - float(initial_capital)),
        "return_pct": float((final_equity / float(initial_capital) - 1.0) * 100.0),
        "cagr_pct": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "annualized_vol_pct": vol,
        "profit_factor_daily": profit_factor,
        "max_drawdown_usd": float(abs(drawdown.min())),
        "max_drawdown_pct": float(abs(drawdown_pct.min())),
        "worst_day_usd": float(pd.to_numeric(pnl, errors="coerce").fillna(0.0).min()),
    }
"""
    )


def _parameter_cell(orb_export_root: Path, pullback_export_root: Path) -> nbf.NotebookNode:
    orb_export_root = orb_export_root if orb_export_root.is_absolute() else (REPO_ROOT / orb_export_root)
    pullback_export_root = pullback_export_root if pullback_export_root.is_absolute() else (REPO_ROOT / pullback_export_root)
    orb_export_root = orb_export_root.resolve()
    pullback_export_root = pullback_export_root.resolve()
    return nbf.v4.new_code_cell(
        f"""ORB_EXPORT_ROOT = ROOT / r"{orb_export_root.relative_to(REPO_ROOT)}"
PULLBACK_EXPORT_ROOT = ROOT / r"{pullback_export_root.relative_to(REPO_ROOT)}"

# Parametrage client
ORB_VARIANT_NAME = "baseline_fixed_nominal_atr"
PULLBACK_VARIANT_NAME = "risk_pct_0p0025__max_contracts_6__skip_trade_if_too_small_true"
ORB_WEIGHT = 0.50
PULLBACK_WEIGHT = 0.50
INITIAL_CAPITAL_USD = 50_000.0
BENCHMARK_LABEL = "MNQ buy & hold"

# Multiplicateurs de risque / levier post-backtest.
# 1.0 = risque original; 1.5 = +50% de risque; 2.0 = risque double.
ORB_LEVERAGE = 1.00
PULLBACK_LEVERAGE = 1.00
BLEND_LEVERAGE = 1.00
BENCHMARK_LEVERAGE = 1.00
LEVERAGE_GRID = [1.00, 1.25, 1.50, 2.00]

PLOT_TEMPLATE = "plotly_dark"

required_paths = {{
    "orb_metrics": ORB_EXPORT_ROOT / "variants" / ORB_VARIANT_NAME / "metrics_by_scope.csv",
    "orb_daily": ORB_EXPORT_ROOT / "variants" / ORB_VARIANT_NAME / "daily_results.csv",
    "orb_controls": ORB_EXPORT_ROOT / "variants" / ORB_VARIANT_NAME / "controls.csv",
    "orb_run_metadata": ORB_EXPORT_ROOT / "run_metadata.json",
    "pullback_daily": PULLBACK_EXPORT_ROOT / "daily_equity_by_variant.csv",
    "pullback_summary": PULLBACK_EXPORT_ROOT / "summary_by_variant.csv",
    "pullback_run_metadata": PULLBACK_EXPORT_ROOT / "run_metadata.json",
}}

missing = [name for name, path in required_paths.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"Fichiers manquants pour le notebook: {{missing}}")

print("ORB_EXPORT_ROOT =", ORB_EXPORT_ROOT)
print("PULLBACK_EXPORT_ROOT =", PULLBACK_EXPORT_ROOT)

client_parameters = pd.DataFrame(
    [
        {{"parameter": "ORB_VARIANT_NAME", "value": ORB_VARIANT_NAME}},
        {{"parameter": "PULLBACK_VARIANT_NAME", "value": PULLBACK_VARIANT_NAME}},
        {{"parameter": "ORB_WEIGHT", "value": ORB_WEIGHT}},
        {{"parameter": "PULLBACK_WEIGHT", "value": PULLBACK_WEIGHT}},
        {{"parameter": "INITIAL_CAPITAL_USD", "value": INITIAL_CAPITAL_USD}},
        {{"parameter": "BENCHMARK_LABEL", "value": BENCHMARK_LABEL}},
        {{"parameter": "ORB_LEVERAGE", "value": ORB_LEVERAGE}},
        {{"parameter": "PULLBACK_LEVERAGE", "value": PULLBACK_LEVERAGE}},
        {{"parameter": "BLEND_LEVERAGE", "value": BLEND_LEVERAGE}},
        {{"parameter": "BENCHMARK_LEVERAGE", "value": BENCHMARK_LEVERAGE}},
        {{"parameter": "LEVERAGE_GRID", "value": LEVERAGE_GRID}},
        {{"parameter": "PLOT_TEMPLATE", "value": PLOT_TEMPLATE}},
    ]
)
display(Markdown("### Parametrage client"))
display(client_parameters)
"""
    )


def _load_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """orb_run_metadata = json.loads((ORB_EXPORT_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
pullback_run_metadata = json.loads((PULLBACK_EXPORT_ROOT / "run_metadata.json").read_text(encoding="utf-8"))

orb_metrics = pd.read_csv(ORB_EXPORT_ROOT / "variants" / ORB_VARIANT_NAME / "metrics_by_scope.csv")
orb_daily = pd.read_csv(ORB_EXPORT_ROOT / "variants" / ORB_VARIANT_NAME / "daily_results.csv", parse_dates=["session_date"])
orb_controls = pd.read_csv(ORB_EXPORT_ROOT / "variants" / ORB_VARIANT_NAME / "controls.csv", parse_dates=["session_date"])
pullback_daily_all = pd.read_csv(PULLBACK_EXPORT_ROOT / "daily_equity_by_variant.csv", parse_dates=["session_date"])
pullback_summary = pd.read_csv(PULLBACK_EXPORT_ROOT / "summary_by_variant.csv")

weights = normalize_weights([ORB_WEIGHT, PULLBACK_WEIGHT])
orb_weight, pullback_weight = weights
orb_leverage = validate_multiplier(ORB_LEVERAGE, "ORB_LEVERAGE")
pullback_leverage = validate_multiplier(PULLBACK_LEVERAGE, "PULLBACK_LEVERAGE")
blend_leverage = validate_multiplier(BLEND_LEVERAGE, "BLEND_LEVERAGE")
benchmark_leverage = validate_multiplier(BENCHMARK_LEVERAGE, "BENCHMARK_LEVERAGE")

effective_exposure = pd.DataFrame(
    [
        {"sleeve": "ORB", "portfolio_weight": orb_weight, "sleeve_leverage": orb_leverage, "blend_leverage": blend_leverage, "effective_blend_exposure": orb_weight * orb_leverage * blend_leverage},
        {"sleeve": "Pullback", "portfolio_weight": pullback_weight, "sleeve_leverage": pullback_leverage, "blend_leverage": blend_leverage, "effective_blend_exposure": pullback_weight * pullback_leverage * blend_leverage},
        {"sleeve": BENCHMARK_LABEL, "portfolio_weight": 1.0, "sleeve_leverage": benchmark_leverage, "blend_leverage": 1.0, "effective_blend_exposure": benchmark_leverage},
    ]
)

orb_daily["session_date"] = orb_daily["session_date"].dt.normalize()
orb_controls["session_date"] = orb_controls["session_date"].dt.normalize()
pullback_daily_all["session_date"] = pullback_daily_all["session_date"].dt.normalize()

orb_calendar = orb_controls[["session_date", "phase"]].drop_duplicates().sort_values("session_date").reset_index(drop=True)
orb_frame = orb_calendar.merge(orb_daily[["session_date", "daily_pnl_usd"]], on="session_date", how="left")
orb_frame["daily_pnl_usd"] = orb_frame["daily_pnl_usd"].fillna(0.0)

pullback_frame = pullback_daily_all.loc[
    (pullback_daily_all["campaign_variant_name"].astype(str) == PULLBACK_VARIANT_NAME)
    & (pullback_daily_all["scope"].astype(str) == "full")
].copy()

overlap_start = max(orb_frame["session_date"].min(), pullback_frame["session_date"].min())
overlap_end = min(orb_frame["session_date"].max(), pullback_frame["session_date"].max())

orb_overlap = orb_frame.loc[(orb_frame["session_date"] >= overlap_start) & (orb_frame["session_date"] <= overlap_end)].copy()
pullback_overlap = pullback_frame.loc[(pullback_frame["session_date"] >= overlap_start) & (pullback_frame["session_date"] <= overlap_end)].copy()

blend = orb_overlap.merge(
    pullback_overlap[["session_date", "daily_pnl_usd", "phase"]],
    on="session_date",
    how="inner",
    suffixes=("_orb", "_pullback"),
)

blend["orb_raw_daily_pnl_usd"] = blend["daily_pnl_usd_orb"]
blend["pullback_raw_daily_pnl_usd"] = blend["daily_pnl_usd_pullback"]

orb_unlevered_path = build_sleeve_path(blend["orb_raw_daily_pnl_usd"], INITIAL_CAPITAL_USD)
pullback_unlevered_path = build_sleeve_path(blend["pullback_raw_daily_pnl_usd"], INITIAL_CAPITAL_USD)
blend["orb_unlevered_daily_return"] = orb_unlevered_path["daily_return"]
blend["pullback_unlevered_daily_return"] = pullback_unlevered_path["daily_return"]
blend["orb_daily_return"] = scale_daily_return(blend["orb_unlevered_daily_return"], orb_leverage, "ORB")
blend["pullback_daily_return"] = scale_daily_return(blend["pullback_unlevered_daily_return"], pullback_leverage, "Pullback")

orb_path = build_return_path(blend["orb_daily_return"], INITIAL_CAPITAL_USD, label="ORB")
pullback_path = build_return_path(blend["pullback_daily_return"], INITIAL_CAPITAL_USD, label="Pullback")
blend["orb_equity"] = orb_path["equity"]
blend["pullback_equity"] = pullback_path["equity"]
blend["daily_pnl_usd_orb"] = orb_path["daily_pnl_usd"]
blend["daily_pnl_usd_pullback"] = pullback_path["daily_pnl_usd"]

blend["blend_pre_leverage_daily_return"] = orb_weight * blend["orb_daily_return"] + pullback_weight * blend["pullback_daily_return"]
blend["blend_daily_return"] = scale_daily_return(blend["blend_pre_leverage_daily_return"], blend_leverage, "Blend")
blend_path = build_return_path(blend["blend_daily_return"], INITIAL_CAPITAL_USD, label="Blend")
blend["blend_equity"] = blend_path["equity"]
blend["blend_drawdown_usd"] = blend_path["drawdown_usd"]
blend["blend_drawdown_pct"] = blend_path["drawdown_pct"]
blend["blend_daily_pnl_usd"] = blend_path["daily_pnl_usd"]

common_oos_start = max(
    blend.loc[blend["phase_orb"].astype(str) == "oos", "session_date"].min(),
    blend.loc[blend["phase_pullback"].astype(str) == "oos", "session_date"].min(),
)

blend_full = blend[["session_date", "blend_daily_return", "blend_equity", "blend_drawdown_usd", "blend_drawdown_pct", "blend_daily_pnl_usd"]].rename(
    columns={"blend_daily_return": "daily_return", "blend_equity": "equity", "blend_drawdown_usd": "drawdown_usd", "blend_drawdown_pct": "drawdown_pct", "blend_daily_pnl_usd": "daily_pnl_usd"}
)
blend_oos = blend.loc[blend["session_date"] >= common_oos_start].copy().reset_index(drop=True)
blend_oos["equity_rebased"] = INITIAL_CAPITAL_USD * (1.0 + blend_oos["blend_daily_return"]).cumprod()
blend_oos["peak_rebased"] = blend_oos["equity_rebased"].cummax()
blend_oos["drawdown_rebased_usd"] = blend_oos["equity_rebased"] - blend_oos["peak_rebased"]
blend_oos["drawdown_rebased_pct"] = np.where(blend_oos["peak_rebased"] > 0, (blend_oos["equity_rebased"] / blend_oos["peak_rebased"] - 1.0) * 100.0, 0.0)
blend_oos["daily_pnl_rebased_usd"] = blend_oos["equity_rebased"].diff().fillna(blend_oos["equity_rebased"].iloc[0] - INITIAL_CAPITAL_USD)

metrics_rows = []
for label, ret_col, pnl_col, eq_col, dd_col, phase_filter in [
    ("orb_full", "orb_daily_return", "daily_pnl_usd_orb", None, None, slice(None)),
    ("pullback_full", "pullback_daily_return", "daily_pnl_usd_pullback", None, None, slice(None)),
    ("blend_full", "blend_daily_return", "blend_daily_pnl_usd", "blend_equity", "blend_drawdown_usd", slice(None)),
]:
    scoped = blend.loc[phase_filter].copy()
    curve = pd.DataFrame({"session_date": scoped["session_date"], "daily_return": scoped[ret_col]})
    if eq_col is None:
        sleeve = build_sleeve_path(scoped[pnl_col], INITIAL_CAPITAL_USD)
        curve["equity"] = sleeve["equity"]
        peak = curve["equity"].cummax()
        curve["drawdown_usd"] = curve["equity"] - peak
        curve["drawdown_pct"] = np.where(peak > 0, (curve["equity"] / peak - 1.0) * 100.0, 0.0)
        curve["daily_pnl_usd"] = pd.to_numeric(scoped[pnl_col], errors="coerce").fillna(0.0).values
    else:
        curve["equity"] = scoped[eq_col].values
        curve["drawdown_usd"] = scoped[dd_col].values
        curve["drawdown_pct"] = scoped["blend_drawdown_pct"].values
        curve["daily_pnl_usd"] = scoped[pnl_col].values
    metrics_rows.append({"portfolio": label, **curve_metrics(curve, INITIAL_CAPITAL_USD)})

for label, ret_col, pnl_col in [
    ("orb_oos", "orb_daily_return", "daily_pnl_usd_orb"),
    ("pullback_oos", "pullback_daily_return", "daily_pnl_usd_pullback"),
]:
    scoped = blend.loc[blend["session_date"] >= common_oos_start].copy().reset_index(drop=True)
    curve = pd.DataFrame({"session_date": scoped["session_date"], "daily_return": scoped[ret_col]})
    sleeve = build_sleeve_path(scoped[pnl_col], INITIAL_CAPITAL_USD)
    curve["equity"] = sleeve["equity"]
    peak = curve["equity"].cummax()
    curve["drawdown_usd"] = curve["equity"] - peak
    curve["drawdown_pct"] = np.where(peak > 0, (curve["equity"] / peak - 1.0) * 100.0, 0.0)
    curve["daily_pnl_usd"] = pd.to_numeric(scoped[pnl_col], errors="coerce").fillna(0.0).values
    metrics_rows.append({"portfolio": label, **curve_metrics(curve, INITIAL_CAPITAL_USD)})

blend_oos_curve = pd.DataFrame({
    "session_date": blend_oos["session_date"],
    "daily_return": blend_oos["blend_daily_return"],
    "equity": blend_oos["equity_rebased"],
    "drawdown_usd": blend_oos["drawdown_rebased_usd"],
    "drawdown_pct": blend_oos["drawdown_rebased_pct"],
    "daily_pnl_usd": blend_oos["daily_pnl_rebased_usd"],
})
metrics_rows.append({"portfolio": "blend_oos", **curve_metrics(blend_oos_curve, INITIAL_CAPITAL_USD)})
portfolio_metrics = pd.DataFrame(metrics_rows)

correlation_full = float(blend[["orb_daily_return", "pullback_daily_return"]].corr().iloc[0, 1])
correlation_oos = float(blend.loc[blend["session_date"] >= common_oos_start, ["orb_daily_return", "pullback_daily_return"]].corr().iloc[0, 1])

dataset_path = Path(str(orb_run_metadata["dataset_path"]))
raw_benchmark = pd.read_parquet(dataset_path)
if "timestamp" not in raw_benchmark.columns:
    if raw_benchmark.index.name:
        raw_benchmark = raw_benchmark.reset_index()
    else:
        raise KeyError("Benchmark dataset must expose a timestamp column or timestamp index.")
raw_benchmark = raw_benchmark.loc[:, [col for col in ["timestamp", "close"] if col in raw_benchmark.columns]].copy()
raw_benchmark["timestamp"] = pd.to_datetime(raw_benchmark["timestamp"], errors="coerce")
raw_benchmark = raw_benchmark.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
raw_benchmark["session_date"] = raw_benchmark["timestamp"].dt.tz_localize(None).dt.normalize()
daily_close = raw_benchmark.groupby("session_date", as_index=False)["close"].last()
benchmark = blend[["session_date"]].merge(daily_close, on="session_date", how="left").sort_values("session_date").reset_index(drop=True)
benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce").ffill()
benchmark["unlevered_daily_return"] = benchmark["close"].pct_change().fillna(0.0)
benchmark["daily_return"] = scale_daily_return(benchmark["unlevered_daily_return"], benchmark_leverage, BENCHMARK_LABEL)
benchmark_path = build_return_path(benchmark["daily_return"], INITIAL_CAPITAL_USD, label=BENCHMARK_LABEL)
benchmark["equity"] = benchmark_path["equity"]
benchmark["drawdown_usd"] = benchmark_path["drawdown_usd"]
benchmark["drawdown_pct"] = benchmark_path["drawdown_pct"]
benchmark["daily_pnl_usd"] = benchmark_path["daily_pnl_usd"]
benchmark_oos = benchmark.loc[benchmark["session_date"] >= common_oos_start].copy().reset_index(drop=True)
benchmark_oos_path = build_return_path(benchmark_oos["daily_return"], INITIAL_CAPITAL_USD, label=f"{BENCHMARK_LABEL} OOS")
benchmark_oos["equity_rebased"] = benchmark_oos_path["equity"]
benchmark_oos["drawdown_rebased_usd"] = benchmark_oos_path["drawdown_usd"]
benchmark_oos["drawdown_rebased_pct"] = benchmark_oos_path["drawdown_pct"]
benchmark_oos["daily_pnl_rebased_usd"] = benchmark_oos_path["daily_pnl_usd"]

benchmark_full_curve = benchmark[["session_date", "daily_return", "equity", "drawdown_usd", "drawdown_pct", "daily_pnl_usd"]].copy()
benchmark_oos_curve = pd.DataFrame({
    "session_date": benchmark_oos["session_date"],
    "daily_return": benchmark_oos["daily_return"],
    "equity": benchmark_oos["equity_rebased"],
    "drawdown_usd": benchmark_oos["drawdown_rebased_usd"],
    "drawdown_pct": benchmark_oos["drawdown_rebased_pct"],
    "daily_pnl_usd": benchmark_oos["daily_pnl_rebased_usd"],
})
benchmark_metrics = pd.DataFrame(
    [
        {"portfolio": "benchmark_full", **curve_metrics(benchmark_full_curve, INITIAL_CAPITAL_USD)},
        {"portfolio": "benchmark_oos", **curve_metrics(benchmark_oos_curve, INITIAL_CAPITAL_USD)},
    ]
)
portfolio_metrics = pd.concat([portfolio_metrics, benchmark_metrics], ignore_index=True)

client_scorecard = (
    portfolio_metrics.loc[
        portfolio_metrics["portfolio"].isin(
            ["orb_oos", "pullback_oos", "blend_oos", "benchmark_oos"]
        ),
        [
            "portfolio",
            "net_pnl_usd",
            "return_pct",
            "cagr_pct",
            "sharpe",
            "sortino",
            "profit_factor_daily",
            "max_drawdown_usd",
            "max_drawdown_pct",
            "worst_day_usd",
        ],
    ]
    .assign(
        portfolio=lambda frame: pd.Categorical(
            frame["portfolio"],
            categories=["orb_oos", "pullback_oos", "blend_oos", "benchmark_oos"],
            ordered=True,
        )
    )
    .sort_values("portfolio")
    .reset_index(drop=True)
)
client_scorecard["portfolio"] = client_scorecard["portfolio"].astype(str)

display(Markdown(f"**Overlap sample:** `{overlap_start.date()}` -> `{overlap_end.date()}`"))
display(Markdown(f"**Common OOS start:** `{common_oos_start.date()}`"))
display(Markdown("**Risk / leverage multipliers used in this run:**"))
display(effective_exposure.round(3))
"""
    )


def _summary_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 1. Executive Summary"))

blend_oos_row = portfolio_metrics.loc[portfolio_metrics["portfolio"] == "blend_oos"].iloc[0]
orb_oos_row = portfolio_metrics.loc[portfolio_metrics["portfolio"] == "orb_oos"].iloc[0]
pullback_oos_row = portfolio_metrics.loc[portfolio_metrics["portfolio"] == "pullback_oos"].iloc[0]
benchmark_oos_row = portfolio_metrics.loc[portfolio_metrics["portfolio"] == "benchmark_oos"].iloc[0]
blend_vs_benchmark_net = float(blend_oos_row["net_pnl_usd"] - benchmark_oos_row["net_pnl_usd"])
blend_vs_benchmark_dd = float(blend_oos_row["max_drawdown_usd"] - benchmark_oos_row["max_drawdown_usd"])

summary_lines = [
    f"- ORB variant par defaut: `{ORB_VARIANT_NAME}`.",
    f"- Pullback variant: `{PULLBACK_VARIANT_NAME}`.",
    f"- Poids utilises: ORB `{orb_weight:.0%}` / Pullback `{pullback_weight:.0%}`.",
    f"- Levier applique: ORB `{orb_leverage:.2f}x` | Pullback `{pullback_leverage:.2f}x` | Blend `{blend_leverage:.2f}x` | Benchmark `{benchmark_leverage:.2f}x`.",
    f"- Correlation journaliere full: `{fmt_float(correlation_full, 3)}` | OOS: `{fmt_float(correlation_oos, 3)}`.",
    f"- Blend OOS: net `{fmt_money(blend_oos_row['net_pnl_usd'])}` | Sharpe `{fmt_float(blend_oos_row['sharpe'])}` | maxDD `{fmt_money(blend_oos_row['max_drawdown_usd'])}`.",
    f"- Benchmark OOS: net `{fmt_money(benchmark_oos_row['net_pnl_usd'])}` | Sharpe `{fmt_float(benchmark_oos_row['sharpe'])}` | maxDD `{fmt_money(benchmark_oos_row['max_drawdown_usd'])}`.",
    f"- Blend OOS vs benchmark: delta net `{fmt_money(blend_vs_benchmark_net)}` | delta maxDD `{fmt_money(blend_vs_benchmark_dd)}`.",
    f"- ORB OOS seul: net `{fmt_money(orb_oos_row['net_pnl_usd'])}` | Sharpe `{fmt_float(orb_oos_row['sharpe'])}` | maxDD `{fmt_money(orb_oos_row['max_drawdown_usd'])}`.",
    f"- Pullback OOS seul: net `{fmt_money(pullback_oos_row['net_pnl_usd'])}` | Sharpe `{fmt_float(pullback_oos_row['sharpe'])}` | maxDD `{fmt_money(pullback_oos_row['max_drawdown_usd'])}`.",
]
display(Markdown("\\n".join(summary_lines)))

display(Markdown("## 2. Client Scorecard - OOS"))
display(client_scorecard.round(3))

display(Markdown("## 2b. Metrics Table - Full and OOS"))
display(portfolio_metrics.round(3))
"""
    )


def _leverage_sensitivity_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 2c. Leverage Sensitivity - Blend"))

sensitivity_rows = []
for leverage in LEVERAGE_GRID:
    leverage = validate_multiplier(leverage, "LEVERAGE_GRID item")
    for scope_name, scoped in [
        ("full", blend),
        ("oos", blend.loc[blend["session_date"] >= common_oos_start].copy().reset_index(drop=True)),
    ]:
        try:
            tested_return = scale_daily_return(scoped["blend_pre_leverage_daily_return"], leverage, f"Blend sensitivity {leverage:.2f}x {scope_name}")
            tested_path = build_return_path(tested_return, INITIAL_CAPITAL_USD, label=f"Blend sensitivity {leverage:.2f}x {scope_name}")
            tested_curve = pd.DataFrame(
                {
                    "session_date": scoped["session_date"].reset_index(drop=True),
                    "daily_return": tested_path["daily_return"],
                    "equity": tested_path["equity"],
                    "drawdown_usd": tested_path["drawdown_usd"],
                    "drawdown_pct": tested_path["drawdown_pct"],
                    "daily_pnl_usd": tested_path["daily_pnl_usd"],
                }
            )
            sensitivity_rows.append(
                {
                    "scope": scope_name,
                    "blend_leverage_tested": leverage,
                    "status": "ok",
                    **curve_metrics(tested_curve, INITIAL_CAPITAL_USD),
                }
            )
        except ValueError as exc:
            sensitivity_rows.append(
                {
                    "scope": scope_name,
                    "blend_leverage_tested": leverage,
                    "status": str(exc),
                }
            )

leverage_sensitivity = pd.DataFrame(sensitivity_rows)
display(leverage_sensitivity.round(3))
"""
    )


def _curve_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 3. Equity Curves With Benchmark"))

fig = make_subplots(
    rows=2,
    cols=1,
    vertical_spacing=0.12,
    subplot_titles=("Full Sample Equity", "OOS Equity Rebased"),
)

full_series = [
    ("ORB", "#2563eb", blend["orb_equity"]),
    ("Pullback", "#16a34a", blend["pullback_equity"]),
    ("Blend", "#f59e0b", blend["blend_equity"]),
    (BENCHMARK_LABEL, "#a3a3a3", benchmark["equity"]),
]
for name, color, series in full_series:
    fig.add_trace(
        go.Scatter(
            x=blend["session_date"],
            y=series,
            mode="lines",
            name=name,
            line=dict(color=color, width=2.8 if name == "Blend" else 2.0, dash="solid" if name != BENCHMARK_LABEL else "dash"),
        ),
        row=1,
        col=1,
    )

oos_scope = blend.loc[blend["session_date"] >= common_oos_start].copy().reset_index(drop=True)
oos_orb = build_sleeve_path(oos_scope["daily_pnl_usd_orb"], INITIAL_CAPITAL_USD)
oos_pullback = build_sleeve_path(oos_scope["daily_pnl_usd_pullback"], INITIAL_CAPITAL_USD)

oos_series = [
    ("ORB", "#2563eb", oos_orb["equity"]),
    ("Pullback", "#16a34a", oos_pullback["equity"]),
    ("Blend", "#f59e0b", blend_oos["equity_rebased"]),
    (BENCHMARK_LABEL, "#a3a3a3", benchmark_oos["equity_rebased"]),
]
for name, color, series in oos_series:
    fig.add_trace(
        go.Scatter(
            x=oos_scope["session_date"],
            y=series,
            mode="lines",
            name=f"{name} OOS",
            legendgroup=name,
            showlegend=False,
            line=dict(color=color, width=2.8 if name == "Blend" else 2.0, dash="solid" if name != BENCHMARK_LABEL else "dash"),
        ),
        row=2,
        col=1,
    )

fig.update_yaxes(title_text="Equity (USD)", row=1, col=1)
fig.update_yaxes(title_text="Equity Rebased (USD)", row=2, col=1)
fig.update_xaxes(title_text="Session Date", row=2, col=1)
fig.update_layout(
    template=PLOT_TEMPLATE,
    height=950,
    width=1550,
    legend=dict(orientation="h", y=-0.10, x=0.0),
)
fig.show()
"""
    )


def _diversification_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 4. Diversification Readout"))

scatter = px.scatter(
    blend,
    x="orb_daily_return",
    y="pullback_daily_return",
    color="phase_orb",
    title="Daily return scatter: ORB vs Pullback",
)
scatter.update_layout(template=PLOT_TEMPLATE, width=900, height=550)
scatter.show()

roll = blend[["session_date", "orb_daily_return", "pullback_daily_return"]].copy()
roll["rolling_corr_63d"] = roll["orb_daily_return"].rolling(63).corr(roll["pullback_daily_return"])
roll_fig = px.line(roll, x="session_date", y="rolling_corr_63d", title="63-day rolling correlation")
roll_fig.update_layout(template=PLOT_TEMPLATE, width=1200, height=450)
roll_fig.show()

monthly = blend.copy()
monthly["month"] = monthly["session_date"].dt.to_period("M").dt.to_timestamp()
monthly_rollup = monthly.groupby("month", as_index=False).agg(
    orb_pnl_usd=("daily_pnl_usd_orb", "sum"),
    pullback_pnl_usd=("daily_pnl_usd_pullback", "sum"),
    blend_pnl_usd=("blend_daily_pnl_usd", "sum"),
)
monthly_long = monthly_rollup.melt(id_vars="month", var_name="sleeve", value_name="pnl_usd")
bar = px.bar(monthly_long, x="month", y="pnl_usd", color="sleeve", barmode="group", title="Monthly PnL contribution")
bar.update_layout(template=PLOT_TEMPLATE, width=1400, height=500)
bar.show()
"""
    )


def _conclusion_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 5. Conclusion"))

blend_oos_row = portfolio_metrics.loc[portfolio_metrics["portfolio"] == "blend_oos"].iloc[0]
benchmark_oos_row = portfolio_metrics.loc[portfolio_metrics["portfolio"] == "benchmark_oos"].iloc[0]
if float(blend_oos_row["sharpe"]) >= 1.6:
    sharpe_read = "au-dessus"
else:
    sharpe_read = "en-dessous"

lines = [
    f"- Sur l'overlap commun audite, le Sharpe OOS du blend est `{fmt_float(blend_oos_row['sharpe'])}`, donc `{sharpe_read}` de ton hypothese `1.6`.",
    f"- Le point cle est la correlation journaliere faible / negative entre les sleeves (`{fmt_float(correlation_oos, 3)}` en OOS).",
    f"- Face au benchmark `{BENCHMARK_LABEL}`, le blend OOS fait `{fmt_money(blend_oos_row['net_pnl_usd'] - benchmark_oos_row['net_pnl_usd'])}` de net PnL differentiel.",
    f"- Si tu veux, le notebook te permet aussi de remplacer l'ORB baseline par une variante filtree simplement en changeant `ORB_VARIANT_NAME` dans la cellule de parametres.",
]
display(Markdown("\\n".join(lines)))
"""
    )


def build_notebook(orb_export_root: Path, pullback_export_root: Path) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"},
    }
    notebook.cells = [
        _title_cell(),
        _imports_cell(),
        _parameter_cell(orb_export_root, pullback_export_root),
        _load_cell(),
        _summary_cell(),
        _leverage_sensitivity_cell(),
        _curve_cell(),
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
    parser.add_argument("--orb-export-root", type=Path, default=DEFAULT_ORB_EXPORT_ROOT)
    parser.add_argument("--pullback-export-root", type=Path, default=DEFAULT_PULLBACK_EXPORT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK_PATH)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--executed-output", type=Path, default=DEFAULT_EXECUTED_NOTEBOOK_PATH)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebook = build_notebook(args.orb_export_root, args.pullback_export_root)
    output_path = write_notebook(notebook, args.output)
    print(f"Notebook written to {output_path}")
    if args.execute:
        executed_path = execute_notebook(output_path, args.executed_output, timeout_seconds=args.timeout_seconds)
        print(f"Executed notebook written to {executed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
