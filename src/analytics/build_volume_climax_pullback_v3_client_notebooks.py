"""Build client-facing notebooks for the four frozen Volume Climax Pullback V3 specs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import nbformat as nbf
import pandas as pd
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytics.volume_climax_pullback_common import load_latest_reference_run

EXPORTS_ROOT = REPO_ROOT / "data" / "exports" / "volume_climax_pullback_v3_run"
DEFAULT_EXPORT_PREFIX = "volume_climax_pullback_v3_"
NOTEBOOKS_ROOT = REPO_ROOT / "notebooks"
SYMBOLS = ("MNQ", "MES", "M2K", "MGC")


@dataclass(frozen=True)
class NotebookContext:
    symbol: str
    export_root: Path
    v2_reference_root: Path
    frozen_variant_name: str
    v2_reference_variant_name: str
    active_family: str
    active_exit_mode: str
    active_time_stop_bars: int
    active_volume_quantile: float
    active_body_fraction: float
    active_range_atr: float
    active_ema_slope_filter: str
    active_atr_percentile_band: str
    active_compression_filter: str
    regime_heatmap_core: str
    dynamic_compare_variant_name: str | None
    dynamic_compare_label: str | None


def find_latest_export(prefix: str = DEFAULT_EXPORT_PREFIX) -> Path:
    return load_latest_reference_run(EXPORTS_ROOT, prefix)


def _core_label(row: pd.Series) -> str:
    return (
        f"vq{float(row['volume_quantile']):.3f}"
        f" | bf{float(row['min_body_fraction']):.1f}"
        f" | ra{float(row['min_range_atr']):.1f}"
    )


def _build_contexts(export_root: Path) -> dict[str, NotebookContext]:
    export_root = export_root if export_root.is_absolute() else (REPO_ROOT / export_root)
    export_root = export_root.resolve()
    summary = pd.read_csv(export_root / "summary_variants.csv")
    ranking = pd.read_csv(export_root / "ranking_oos_by_asset.csv")
    final_verdict = json.loads((export_root / "final_verdict.json").read_text(encoding="utf-8"))
    run_metadata = json.loads((export_root / "run_metadata.json").read_text(encoding="utf-8"))
    v2_reference_root = (REPO_ROOT / Path(run_metadata["v2_reference_dir"])).resolve()

    contexts: dict[str, NotebookContext] = {}
    for symbol in SYMBOLS:
        frozen_variant_name = str(final_verdict["verdict_by_symbol"][symbol]["recommended_variant_name"])
        asset_summary = summary.loc[summary["symbol"] == symbol].copy()
        frozen_row = asset_summary.loc[asset_summary["variant_name"] == frozen_variant_name].iloc[0]
        dynamic_compare_variant_name = None
        dynamic_compare_label = None
        if symbol == "MGC":
            dynamic_compare = (
                ranking.loc[(ranking["symbol"] == symbol) & (ranking["family"] == "dynamic_exit")]
                .sort_values(
                    ["is_clean_survivor", "selection_score", "oos_sharpe", "oos_net_pnl"],
                    ascending=[False, False, False, False],
                )
                .iloc[0]
            )
            dynamic_compare_variant_name = str(dynamic_compare["variant_name"])
            dynamic_compare_label = "best_v3_dynamic_exit"

        contexts[symbol] = NotebookContext(
            symbol=symbol,
            export_root=export_root,
            v2_reference_root=v2_reference_root,
            frozen_variant_name=frozen_variant_name,
            v2_reference_variant_name=str(frozen_row["v2_reference_variant_name"]),
            active_family=str(frozen_row["family"]),
            active_exit_mode=str(frozen_row["exit_mode"]),
            active_time_stop_bars=int(frozen_row["time_stop_bars"]),
            active_volume_quantile=float(frozen_row["volume_quantile"]),
            active_body_fraction=float(frozen_row["min_body_fraction"]),
            active_range_atr=float(frozen_row["min_range_atr"]),
            active_ema_slope_filter=str(frozen_row.get("ema_slope_filter", "off")),
            active_atr_percentile_band=str(frozen_row.get("atr_percentile_band", "off")),
            active_compression_filter=str(frozen_row.get("compression_filter", "off")),
            regime_heatmap_core=_core_label(frozen_row),
            dynamic_compare_variant_name=dynamic_compare_variant_name,
            dynamic_compare_label=dynamic_compare_label,
        )
    return contexts


def _title_cell(ctx: NotebookContext) -> nbf.NotebookNode:
    dynamic_note = (
        ""
        if ctx.dynamic_compare_variant_name is None
        else "\n- **Comparatif MGC en plus** : le notebook recharge aussi le meilleur `dynamic_exit` pur V3 pour isoler la valeur nette du filtre de regime."
    )
    return nbf.v4.new_markdown_cell(
        f"""# Volume Climax Pullback V3 Client Notebook - {ctx.symbol}

Ce notebook client gele par defaut la spec V3 retenue pour **{ctx.symbol}** :

- **Spec recommandee par defaut** : `{ctx.frozen_variant_name}`
- **Reference V2 explicite** : `{ctx.v2_reference_variant_name}`
- **Benchmark de marche** : buy & hold daily close-to-close sur le meme sous-jacent RTH 1h
{dynamic_note}

Philosophie du notebook :

- il relit la campagne V3 auditee,
- il laisse les parametres de la spec modifiables dans le premier bloc,
- il recharge seulement les specs de comparaison utiles,
- il montre des heatmaps compactes de robustesse sans relancer une recherche large.
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
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from IPython.display import Markdown, display
from plotly.subplots import make_subplots

from src.analytics.orb_notebook_utils import (
    build_scope_readout_markdown,
    curve_annualized_return,
    curve_daily_sharpe,
    curve_daily_vol,
    curve_max_drawdown_pct,
    curve_total_return_pct,
    format_curve_stats_line,
)
from src.analytics.volume_climax_pullback_notebook_utils import (
    core_label,
    evaluate_variant,
    exit_profile_label,
    find_variant_row,
    regime_signature,
    variant_from_summary_row,
)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)
sns.set_theme(style="whitegrid")


def fmt_money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.1f} USD"


def fmt_float(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}%"


def scope_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("scope")


def rounded_view(frame: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = frame.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].round(digits)
    return out


def curve_stats_line(name: str, curve: pd.DataFrame, metrics_row: pd.Series | None = None) -> str:
    return format_curve_stats_line(
        name=name,
        sharpe=curve_daily_sharpe(curve),
        ret_pct=curve_total_return_pct(curve, INITIAL_CAPITAL_USD),
        cagr_pct=curve_annualized_return(curve, INITIAL_CAPITAL_USD) * 100.0,
        vol_pct=curve_daily_vol(curve) * 100.0,
        dd_pct=abs(curve_max_drawdown_pct(curve)),
        pf=None if metrics_row is None else float(metrics_row.get("profit_factor", 0.0)),
        exp=None if metrics_row is None else float(metrics_row.get("expectancy", 0.0)),
    )


def metrics_snapshot(bundle: dict, label: str) -> dict:
    metrics = scope_lookup(bundle["metrics_by_scope"])
    overall = metrics.loc["overall"]
    oos = metrics.loc["oos"]
    return {
        "label": label,
        "overall_net_pnl": round(float(overall["net_pnl"]), 1),
        "overall_sharpe": round(float(overall["sharpe"]), 3),
        "overall_profit_factor": round(float(overall["profit_factor"]), 3),
        "overall_expectancy": round(float(overall["expectancy"]), 2),
        "overall_trades": int(overall["nb_trades"]),
        "oos_net_pnl": round(float(oos["net_pnl"]), 1),
        "oos_sharpe": round(float(oos["sharpe"]), 3),
        "oos_profit_factor": round(float(oos["profit_factor"]), 3),
        "oos_expectancy": round(float(oos["expectancy"]), 2),
        "oos_trades": int(oos["nb_trades"]),
    }
"""
    )


def _parameter_cell(ctx: NotebookContext) -> nbf.NotebookNode:
    export_rel = ctx.export_root.relative_to(REPO_ROOT)
    v2_rel = ctx.v2_reference_root.relative_to(REPO_ROOT)
    dynamic_compare_value = "None" if ctx.dynamic_compare_variant_name is None else repr(ctx.dynamic_compare_variant_name)
    dynamic_compare_label = "None" if ctx.dynamic_compare_label is None else repr(ctx.dynamic_compare_label)
    return nbf.v4.new_code_cell(
        f"""EXPORT_ROOT = ROOT / r"{export_rel}"
V2_REFERENCE_ROOT = ROOT / r"{v2_rel}"
SYMBOL = "{ctx.symbol}"
INITIAL_CAPITAL_USD = 50_000.0

FROZEN_VARIANT_NAME = "{ctx.frozen_variant_name}"
FROZEN_V2_REFERENCE_VARIANT_NAME = "{ctx.v2_reference_variant_name}"
FROZEN_V3_DYNAMIC_COMPARE_NAME = {dynamic_compare_value}
FROZEN_V3_DYNAMIC_COMPARE_LABEL = {dynamic_compare_label}

# Leave ACTIVE_VARIANT_NAME to None to resolve the active spec from the explicit knobs below.
ACTIVE_VARIANT_NAME = None
ACTIVE_FAMILY = "{ctx.active_family}"
ACTIVE_EXIT_MODE = "{ctx.active_exit_mode}"
ACTIVE_TIME_STOP_BARS = {ctx.active_time_stop_bars}
ACTIVE_VOLUME_QUANTILE = {ctx.active_volume_quantile!r}
ACTIVE_BODY_FRACTION = {ctx.active_body_fraction!r}
ACTIVE_RANGE_ATR = {ctx.active_range_atr!r}
ACTIVE_EMA_SLOPE_FILTER = "{ctx.active_ema_slope_filter}"
ACTIVE_ATR_PERCENTILE_BAND = "{ctx.active_atr_percentile_band}"
ACTIVE_COMPRESSION_FILTER = "{ctx.active_compression_filter}"

COMPARE_TO_V2_REFERENCE = True
COMPARE_TO_BUY_HOLD = True
SHOW_TOP_COMPETITORS = 10
REGIME_HEATMAP_CORE = "{ctx.regime_heatmap_core}"

required_paths = {{
    "export_root": EXPORT_ROOT,
    "run_metadata": EXPORT_ROOT / "run_metadata.json",
    "final_verdict": EXPORT_ROOT / "final_verdict.json",
    "summary_variants": EXPORT_ROOT / "summary_variants.csv",
    "ranking_oos_by_asset": EXPORT_ROOT / "ranking_oos_by_asset.csv",
    "v2_reference_root": V2_REFERENCE_ROOT,
    "v2_summary_variants": V2_REFERENCE_ROOT / "summary_variants.csv",
}}

missing = [name for name, path in required_paths.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"Fichiers manquants pour le notebook: {{missing}}")

print("EXPORT_ROOT =", EXPORT_ROOT)
print("SYMBOL =", SYMBOL)
print("FROZEN_VARIANT_NAME =", FROZEN_VARIANT_NAME)
print("ACTIVE_VARIANT_NAME =", ACTIVE_VARIANT_NAME)
"""
    )


def _load_and_replay_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """run_metadata = json.loads((EXPORT_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
final_verdict = json.loads((EXPORT_ROOT / "final_verdict.json").read_text(encoding="utf-8"))

summary = pd.read_csv(EXPORT_ROOT / "summary_variants.csv")
ranking_by_asset = pd.read_csv(EXPORT_ROOT / "ranking_oos_by_asset.csv")
v2_summary = pd.read_csv(V2_REFERENCE_ROOT / "summary_variants.csv")

asset_summary = summary.loc[summary["symbol"] == SYMBOL].copy().reset_index(drop=True)
asset_ranking = ranking_by_asset.loc[ranking_by_asset["symbol"] == SYMBOL].copy().reset_index(drop=True)
frozen_row = asset_summary.loc[asset_summary["variant_name"] == FROZEN_VARIANT_NAME].iloc[0]

active_row = find_variant_row(
    asset_summary,
    symbol=SYMBOL,
    family=ACTIVE_FAMILY,
    exit_mode=ACTIVE_EXIT_MODE,
    time_stop_bars=ACTIVE_TIME_STOP_BARS,
    volume_quantile=ACTIVE_VOLUME_QUANTILE,
    min_body_fraction=ACTIVE_BODY_FRACTION,
    min_range_atr=ACTIVE_RANGE_ATR,
    ema_slope_filter=ACTIVE_EMA_SLOPE_FILTER,
    atr_percentile_band=ACTIVE_ATR_PERCENTILE_BAND,
    compression_filter=ACTIVE_COMPRESSION_FILTER,
    variant_name=ACTIVE_VARIANT_NAME,
)
active_variant_changed = str(active_row["variant_name"]) != FROZEN_VARIANT_NAME

v2_reference_row = v2_summary.loc[
    (v2_summary["symbol"] == SYMBOL)
    & (v2_summary["variant_name"] == FROZEN_V2_REFERENCE_VARIANT_NAME)
].iloc[0]

dynamic_compare_row = None
if FROZEN_V3_DYNAMIC_COMPARE_NAME is not None:
    dynamic_compare_row = asset_summary.loc[asset_summary["variant_name"] == FROZEN_V3_DYNAMIC_COMPARE_NAME].iloc[0]

top_competitors = asset_ranking.head(SHOW_TOP_COMPETITORS).copy()
top_competitors["core_label"] = top_competitors.apply(core_label, axis=1)
top_competitors["regime_signature"] = top_competitors.apply(regime_signature, axis=1)

active_bundle = evaluate_variant(
    symbol=SYMBOL,
    variant=variant_from_summary_row(active_row),
    initial_capital=INITIAL_CAPITAL_USD,
)
v2_reference_bundle = evaluate_variant(
    symbol=SYMBOL,
    variant=variant_from_summary_row(v2_reference_row),
    initial_capital=INITIAL_CAPITAL_USD,
)
dynamic_compare_bundle = None
if dynamic_compare_row is not None:
    dynamic_compare_bundle = evaluate_variant(
        symbol=SYMBOL,
        variant=variant_from_summary_row(dynamic_compare_row),
        initial_capital=INITIAL_CAPITAL_USD,
    )

active_metrics = scope_lookup(active_bundle["metrics_by_scope"])
v2_reference_metrics = scope_lookup(v2_reference_bundle["metrics_by_scope"])
dynamic_compare_metrics = None if dynamic_compare_bundle is None else scope_lookup(dynamic_compare_bundle["metrics_by_scope"])

oos_start_date = pd.to_datetime(active_bundle["oos_sessions"][0]) if active_bundle["oos_sessions"] else None
asset_verdict = final_verdict["verdict_by_symbol"][SYMBOL]["verdict"]

display(Markdown(f"**Asset verdict from V3 export:** `{asset_verdict}`"))
if active_variant_changed:
    display(Markdown(f"**Notebook override active:** `{active_row['variant_name']}` instead of frozen `{FROZEN_VARIANT_NAME}`."))
"""
    )


def _quick_read_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 1. Lecture rapide

On veut repondre vite a quatre questions:

1. la spec gelee tient-elle toujours une fois rejouee seule?
2. gagne-t-elle encore face a la reference V2 retenue?
3. que dit le benchmark buy & hold sur la periode complete?
4. le paysage de variants reste-t-il lisible sans concurrence excessive?
"""
    )


def _quick_read_cell(ctx: NotebookContext) -> nbf.NotebookNode:
    extra_line = (
        ""
        if ctx.dynamic_compare_variant_name is None
        else """quick_lines.append(
    f"- MGC compare regime vs best dynamic_exit pur: `{dynamic_compare_row['variant_name']}`."
)"""
    )
    return nbf.v4.new_code_cell(
        f"""quick_lines = [
    "### Synthese executive",
    f"- Spec gelee: `{{FROZEN_VARIANT_NAME}}`.",
    f"- Spec actuellement chargee: `{{active_row['variant_name']}}`.",
    f"- Reference V2: `{{FROZEN_V2_REFERENCE_VARIANT_NAME}}`.",
    f"- OOS active: Sharpe **{{fmt_float(active_metrics.loc['oos', 'sharpe'])}}** | PF **{{fmt_float(active_metrics.loc['oos', 'profit_factor'])}}** | Net PnL **{{fmt_money(active_metrics.loc['oos', 'net_pnl'])}}** | Trades **{{int(active_metrics.loc['oos', 'nb_trades'])}}**.",
    f"- Delta OOS vs V2: Sharpe **{{fmt_float(active_metrics.loc['oos', 'sharpe'] - v2_reference_metrics.loc['oos', 'sharpe'])}}** | Net PnL **{{fmt_money(active_metrics.loc['oos', 'net_pnl'] - v2_reference_metrics.loc['oos', 'net_pnl'])}}**.",
    f"- Buy & hold full sample: **{{fmt_pct(curve_total_return_pct(active_bundle['benchmark_curve_full'], INITIAL_CAPITAL_USD))}}** de retour avec **{{fmt_pct(abs(curve_max_drawdown_pct(active_bundle['benchmark_curve_full'])))}}** de max drawdown.",
    f"- Nombre de concurrents affiches dans le notebook: **{{len(top_competitors)}}**.",
]
{extra_line}
display(Markdown("\\n".join([line for line in quick_lines if line])))
"""
    )


def _parameters_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 2. Parametres exacts et modifiables

Le bloc precedent fixe un notebook client, pas une recherche ouverte.

- `FROZEN_*` documente la version recommandee du run V3.
- `ACTIVE_*` permet de permuter proprement a un autre variant **de la meme grille V3 exportee**.
- si `ACTIVE_VARIANT_NAME` reste `None`, le notebook reconstruit le variant a partir des parametres explicites.
"""
    )


def _parameters_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """parameter_snapshot = pd.DataFrame(
    [
        {"parameter": "SYMBOL", "value": SYMBOL},
        {"parameter": "INITIAL_CAPITAL_USD", "value": INITIAL_CAPITAL_USD},
        {"parameter": "FROZEN_VARIANT_NAME", "value": FROZEN_VARIANT_NAME},
        {"parameter": "ACTIVE_VARIANT_NAME", "value": ACTIVE_VARIANT_NAME},
        {"parameter": "ACTIVE_FAMILY", "value": ACTIVE_FAMILY},
        {"parameter": "ACTIVE_EXIT_MODE", "value": ACTIVE_EXIT_MODE},
        {"parameter": "ACTIVE_TIME_STOP_BARS", "value": ACTIVE_TIME_STOP_BARS},
        {"parameter": "ACTIVE_VOLUME_QUANTILE", "value": ACTIVE_VOLUME_QUANTILE},
        {"parameter": "ACTIVE_BODY_FRACTION", "value": ACTIVE_BODY_FRACTION},
        {"parameter": "ACTIVE_RANGE_ATR", "value": ACTIVE_RANGE_ATR},
        {"parameter": "ACTIVE_EMA_SLOPE_FILTER", "value": ACTIVE_EMA_SLOPE_FILTER},
        {"parameter": "ACTIVE_ATR_PERCENTILE_BAND", "value": ACTIVE_ATR_PERCENTILE_BAND},
        {"parameter": "ACTIVE_COMPRESSION_FILTER", "value": ACTIVE_COMPRESSION_FILTER},
        {"parameter": "REGIME_HEATMAP_CORE", "value": REGIME_HEATMAP_CORE},
    ]
)

active_parameter_table = pd.DataFrame(
    [
        {"field": "variant_name", "active": active_row["variant_name"], "frozen_default": frozen_row["variant_name"]},
        {"field": "family", "active": active_row["family"], "frozen_default": frozen_row["family"]},
        {"field": "exit_mode", "active": active_row["exit_mode"], "frozen_default": frozen_row["exit_mode"]},
        {"field": "time_stop_bars", "active": int(active_row["time_stop_bars"]), "frozen_default": int(frozen_row["time_stop_bars"])},
        {"field": "volume_quantile", "active": float(active_row["volume_quantile"]), "frozen_default": float(frozen_row["volume_quantile"])},
        {"field": "min_body_fraction", "active": float(active_row["min_body_fraction"]), "frozen_default": float(frozen_row["min_body_fraction"])},
        {"field": "min_range_atr", "active": float(active_row["min_range_atr"]), "frozen_default": float(frozen_row["min_range_atr"])},
        {"field": "ema_slope_filter", "active": active_row.get("ema_slope_filter", "off"), "frozen_default": frozen_row.get("ema_slope_filter", "off")},
        {"field": "atr_percentile_band", "active": active_row.get("atr_percentile_band", "off"), "frozen_default": frozen_row.get("atr_percentile_band", "off")},
        {"field": "compression_filter", "active": active_row.get("compression_filter", "off"), "frozen_default": frozen_row.get("compression_filter", "off")},
    ]
)

reference_parameter_table = pd.DataFrame(
    [
        {"field": "variant_name", "v2_reference": v2_reference_row["variant_name"]},
        {"field": "family", "v2_reference": v2_reference_row["family"]},
        {"field": "exit_mode", "v2_reference": v2_reference_row["exit_mode"]},
        {"field": "time_stop_bars", "v2_reference": int(v2_reference_row["time_stop_bars"])},
        {"field": "volume_quantile", "v2_reference": float(v2_reference_row["volume_quantile"])},
        {"field": "min_body_fraction", "v2_reference": float(v2_reference_row["min_body_fraction"])},
        {"field": "min_range_atr", "v2_reference": float(v2_reference_row["min_range_atr"])},
        {"field": "ema_slope_filter", "v2_reference": v2_reference_row.get("ema_slope_filter", "off")},
        {"field": "atr_percentile_band", "v2_reference": v2_reference_row.get("atr_percentile_band", "off")},
        {"field": "compression_filter", "v2_reference": v2_reference_row.get("compression_filter", "off")},
    ]
)

display(Markdown("### Parametres editables du notebook"))
display(parameter_snapshot)

display(Markdown("### Spec active vs spec gelee"))
display(active_parameter_table)

display(Markdown("### Reference V2 rechargee dans le notebook"))
display(reference_parameter_table)
"""
    )


def _research_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 3. Snapshot recherche et references

Cette section ne refait pas le classement V3. Elle le remet en forme pour un lecteur client:

- meilleur candidat actif,
- concurrents directs,
- reference V2 rechargee dans le notebook,
- optionnellement meilleur `dynamic_exit` pur pour MGC.
"""
    )


def _research_cell(ctx: NotebookContext) -> nbf.NotebookNode:
    extra_dynamic = (
        ""
        if ctx.dynamic_compare_variant_name is None
        else """dynamic_snapshot = metrics_snapshot(dynamic_compare_bundle, FROZEN_V3_DYNAMIC_COMPARE_LABEL)
display(Markdown("### MGC compare branch"))
display(pd.DataFrame([dynamic_snapshot]))"""
    )
    return nbf.v4.new_code_cell(
        f"""bundle_snapshot = pd.DataFrame(
    [
        metrics_snapshot(active_bundle, "active_v3"),
        metrics_snapshot(v2_reference_bundle, "v2_reference"),
    ]
)
display(Markdown("### Replay snapshot"))
display(bundle_snapshot)

{extra_dynamic}

display(Markdown("### Top competitors from the audited V3 ranking"))
display(
    rounded_view(
        top_competitors[
            [
                "variant_name",
                "family",
                "exit_mode",
                "time_stop_bars",
                "core_label",
                "regime_signature",
                "oos_sharpe",
                "oos_profit_factor",
                "oos_net_pnl",
                "oos_nb_trades",
                "variant_status",
                "selection_score",
            ]
        ]
    )
)
"""
    )


def _curves_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 4. Courbes, drawdown et benchmark

Lecture attendue:

- la courbe active doit rester lisible vs reference V2,
- l OOS only sert a verifier que le signal ne tient pas seulement par l IS,
- le benchmark buy & hold sert de repere de marche, pas de baseline metier.
"""
    )


def _curves_cell(ctx: NotebookContext) -> nbf.NotebookNode:
    extra_trace_full = (
        ""
        if ctx.dynamic_compare_variant_name is None
        else """full_traces.append(
    {
        "name": FROZEN_V3_DYNAMIC_COMPARE_LABEL,
        "curve": dynamic_compare_bundle["curve_full"],
        "metrics_row": dynamic_compare_metrics.loc["overall"],
        "color": "#7c3aed",
    }
)
oos_traces.append(
    {
        "name": FROZEN_V3_DYNAMIC_COMPARE_LABEL,
        "curve": dynamic_compare_bundle["curve_oos"],
        "metrics_row": dynamic_compare_metrics.loc["oos"],
        "color": "#7c3aed",
    }
)"""
    )
    return nbf.v4.new_code_cell(
        f"""full_traces = [
    {{
        "name": "active_v3",
        "curve": active_bundle["curve_full"],
        "metrics_row": active_metrics.loc["overall"],
        "color": "#059669",
    }},
    {{
        "name": "v2_reference",
        "curve": v2_reference_bundle["curve_full"],
        "metrics_row": v2_reference_metrics.loc["overall"],
        "color": "#d97706",
    }},
]
oos_traces = [
    {{
        "name": "active_v3",
        "curve": active_bundle["curve_oos"],
        "metrics_row": active_metrics.loc["oos"],
        "color": "#059669",
    }},
    {{
        "name": "v2_reference",
        "curve": v2_reference_bundle["curve_oos"],
        "metrics_row": v2_reference_metrics.loc["oos"],
        "color": "#d97706",
    }},
]
{extra_trace_full}

if COMPARE_TO_BUY_HOLD:
    full_traces.append(
        {{
            "name": "buy_hold",
            "curve": active_bundle["benchmark_curve_full"],
            "metrics_row": None,
            "color": "#0ea5e9",
        }}
    )
    oos_traces.append(
        {{
            "name": "buy_hold",
            "curve": active_bundle["benchmark_curve_oos"],
            "metrics_row": None,
        "color": "#0ea5e9",
        }}
    )

vline_x = None if oos_start_date is None else pd.to_datetime(oos_start_date).to_pydatetime()

def build_curve_figure_plotly(title: str, traces: list[dict]) -> None:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.70, 0.30],
        subplot_titles=("Equity Curve (USD)", "Drawdown (%)"),
    )
    for trace in traces:
        fig.add_trace(
            go.Scatter(
                x=trace["curve"]["timestamp"],
                y=trace["curve"]["equity"],
                mode="lines",
                name=curve_stats_line(trace["name"], trace["curve"], trace["metrics_row"]),
                line=dict(width=2.6, color=trace["color"]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=trace["curve"]["timestamp"],
                y=trace["curve"]["drawdown_pct"],
                mode="lines",
                name=f"DD {{trace['name']}}",
                showlegend=False,
                line=dict(width=1.5, color=trace["color"], dash="dot"),
            ),
            row=2,
            col=1,
        )
    if vline_x is not None:
        fig.add_vline(x=vline_x, line_dash="dash", line_color="firebrick", line_width=1.2, row=1, col=1)
        fig.add_vline(x=vline_x, line_dash="dash", line_color="firebrick", line_width=1.2, row=2, col=1)
    fig.update_layout(
        template="plotly_dark",
        width=1800,
        height=950,
        title=title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.24, xanchor="left", x=0.0),
        margin=dict(l=70, r=40, t=90, b=140),
    )
    fig.update_yaxes(title_text="Equity (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.show()

build_curve_figure_plotly(f"{{SYMBOL}} - full sample", full_traces)
build_curve_figure_plotly(f"{{SYMBOL}} - OOS only", oos_traces)

display(Markdown(build_scope_readout_markdown(
    full_curve=active_bundle["curve_full"],
    oos_curve=active_bundle["curve_oos"],
    initial_capital=INITIAL_CAPITAL_USD,
    full_label="Active full-sample curve",
    oos_label="Active OOS-only curve",
)))
"""
    )


def _heatmaps_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 5. Heatmaps de robustesse

Les heatmaps ne sont pas une nouvelle optimisation. Elles servent a verifier que la spec gelee est placee dans un voisinage credible.

- pour tous les actifs: heatmaps `dynamic_exit` sur la grille compacte V3,
- pour MGC: heatmaps de branche `regime_filtered` autour du core choisi.
"""
    )


def _heatmaps_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """dynamic_src = asset_summary.loc[asset_summary["family"] == "dynamic_exit"].copy()
dynamic_src["core_label"] = dynamic_src.apply(core_label, axis=1)
dynamic_src["exit_profile"] = dynamic_src.apply(exit_profile_label, axis=1)

core_order = (
    dynamic_src[["core_label", "volume_quantile", "min_body_fraction", "min_range_atr"]]
    .drop_duplicates()
    .sort_values(["volume_quantile", "min_body_fraction", "min_range_atr"])
    ["core_label"]
    .tolist()
)
exit_mode_order = ["atr_fraction", "mixed"]
metric_specs = [
    ("oos_sharpe", "OOS Sharpe", "RdYlGn"),
    ("oos_profit_factor", "OOS Profit Factor", "RdYlGn"),
    ("oos_net_pnl", "OOS Net PnL", "RdYlGn"),
    ("stability_is_oos_sharpe_ratio", "IS/OOS Stability", "RdYlGn"),
]

for metric, title, scale in metric_specs:
    fig, axes = plt.subplots(1, len(exit_mode_order), figsize=(18, 5), squeeze=False)
    for col_index, exit_mode in enumerate(exit_mode_order):
        pivot = (
            dynamic_src.loc[dynamic_src["exit_mode"] == exit_mode]
            .pivot_table(index="time_stop_bars", columns="core_label", values=metric, aggfunc="mean")
            .reindex(index=sorted(dynamic_src["time_stop_bars"].unique()), columns=core_order)
        )
        sns.heatmap(
            pivot,
            ax=axes[0, col_index],
            cmap=scale,
            annot=True,
            fmt=".2f",
            cbar=True,
        )
        axes[0, col_index].set_title(f"{title} | {exit_mode}")
        axes[0, col_index].set_xlabel("Core signal")
        axes[0, col_index].set_ylabel("time_stop_bars")
        axes[0, col_index].tick_params(axis="x", rotation=45)
    fig.suptitle(f"{SYMBOL} dynamic_exit heatmap - {title}")
    plt.tight_layout()
    plt.show()

regime_src = asset_summary.loc[asset_summary["family"] == "regime_filtered"].copy()
if not regime_src.empty:
    regime_src["core_label"] = regime_src.apply(core_label, axis=1)
    regime_src["regime_signature"] = regime_src.apply(regime_signature, axis=1)
    regime_src["exit_profile"] = regime_src.apply(exit_profile_label, axis=1)
    scoped_regime = regime_src.loc[regime_src["core_label"] == REGIME_HEATMAP_CORE].copy()

    if not scoped_regime.empty:
        regime_metric_specs = [
            ("oos_sharpe", "MGC regime branch - OOS Sharpe", "RdYlGn"),
            ("oos_profit_factor", "MGC regime branch - OOS Profit Factor", "RdYlGn"),
            ("oos_net_pnl", "MGC regime branch - OOS Net PnL", "RdYlGn"),
        ]
        regime_order = (
            scoped_regime[["regime_signature", "ema_slope_filter", "atr_percentile_band", "compression_filter"]]
            .drop_duplicates()
            .sort_values(["ema_slope_filter", "atr_percentile_band", "compression_filter"])
            ["regime_signature"]
            .tolist()
        )
        exit_profile_order = (
            scoped_regime[["exit_profile", "exit_mode", "time_stop_bars"]]
            .drop_duplicates()
            .sort_values(["exit_mode", "time_stop_bars"])
            ["exit_profile"]
            .tolist()
        )

        for metric, title, scale in regime_metric_specs:
            pivot = (
                scoped_regime.pivot_table(index="regime_signature", columns="exit_profile", values=metric, aggfunc="mean")
                .reindex(index=regime_order, columns=exit_profile_order)
            )
            plt.figure(figsize=(14, 7))
            sns.heatmap(
                pivot,
                cmap=scale,
                annot=True,
                fmt=".2f",
                cbar=True,
            )
            plt.title(f"{title} | core={REGIME_HEATMAP_CORE}")
            plt.xlabel("Exit profile")
            plt.ylabel("Regime signature")
            plt.tight_layout()
            plt.show()
"""
    )


def _profile_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 6. Profil de trades et de jours

Ce bloc est plus narratif:

- repartition des raisons de sortie,
- PnL mensuel,
- cadence de trading,
- biais long/short.
"""
    )


def _profile_cell(ctx: NotebookContext) -> nbf.NotebookNode:
    extra_bundle_rows = (
        ""
        if ctx.dynamic_compare_variant_name is None
        else """if dynamic_compare_bundle is not None:
    named_bundles[FROZEN_V3_DYNAMIC_COMPARE_LABEL] = dynamic_compare_bundle"""
    )
    return nbf.v4.new_code_cell(
        f"""named_bundles = {{
    "active_v3": active_bundle,
    "v2_reference": v2_reference_bundle,
}}
{extra_bundle_rows}

exit_reason_rows = []
direction_rows = []
monthly_rows = []
for label, bundle in named_bundles.items():
    trades = bundle["trades"].copy()
    if not trades.empty:
        exit_summary = (
            trades.groupby("exit_reason", as_index=False)
            .agg(
                n_trades=("exit_reason", "count"),
                net_pnl_usd=("net_pnl_usd", "sum"),
                avg_pnl_usd=("net_pnl_usd", "mean"),
            )
        )
        exit_summary["label"] = label
        exit_reason_rows.append(exit_summary)

        direction_summary = (
            trades.groupby("direction", as_index=False)
            .agg(
                n_trades=("direction", "count"),
                net_pnl_usd=("net_pnl_usd", "sum"),
                avg_pnl_usd=("net_pnl_usd", "mean"),
            )
        )
        direction_summary["label"] = label
        direction_rows.append(direction_summary)

    daily = bundle["daily_results"].copy()
    daily["month"] = pd.to_datetime(daily["session_date"]).dt.to_period("M").dt.to_timestamp()
    monthly = daily.groupby("month", as_index=False)["daily_pnl_usd"].sum()
    monthly["label"] = label
    monthly_rows.append(monthly)

if exit_reason_rows:
    exit_reason_summary = rounded_view(pd.concat(exit_reason_rows, ignore_index=True), digits=2)
    display(Markdown("### Exit reasons"))
    display(exit_reason_summary)

if direction_rows:
    direction_summary = rounded_view(pd.concat(direction_rows, ignore_index=True), digits=2)
    display(Markdown("### Direction mix"))
    display(direction_summary)

monthly_summary = pd.concat(monthly_rows, ignore_index=True)
plt.figure(figsize=(16, 6))
sns.barplot(data=monthly_summary, x="month", y="daily_pnl_usd", hue="label")
plt.title(f"{{SYMBOL}} monthly PnL by compared specification")
plt.ylabel("Monthly net PnL (USD)")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
    )


def _conclusion_markdown() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 7. Conclusion client

Le bon usage de ce notebook:

- valider la spec gelee telle quelle,
- verifier ou elle se place face a V2 et au benchmark,
- documenter la lisibilite du voisinage de robustesse,
- modifier le premier bloc seulement si on veut tester un autre point **de la meme grille V3 compacte**.
"""
    )


def _conclusion_cell(ctx: NotebookContext) -> nbf.NotebookNode:
    mgc_extra = (
        ""
        if ctx.dynamic_compare_variant_name is None
        else """conclusion_lines.append(
    f"- Lecture MGC regime vs dynamic_exit pur: active OOS Sharpe {fmt_float(active_metrics.loc['oos', 'sharpe'])} vs dynamic {fmt_float(dynamic_compare_metrics.loc['oos', 'sharpe'])}."
)"""
    )
    return nbf.v4.new_code_cell(
        f"""conclusion_lines = [
    "### Verdict client net",
    f"- Spec gelee par defaut: `{{FROZEN_VARIANT_NAME}}`.",
    f"- Si le notebook reste sur la spec gelee, la recommandation exportee pour {{SYMBOL}} tient toujours.",
    f"- La comparaison V2 rechargee ici reste `{{FROZEN_V2_REFERENCE_VARIANT_NAME}}`.",
    f"- OOS active vs V2: Sharpe `{{fmt_float(active_metrics.loc['oos', 'sharpe'])}}` vs `{{fmt_float(v2_reference_metrics.loc['oos', 'sharpe'])}}`, Net PnL `{{fmt_money(active_metrics.loc['oos', 'net_pnl'])}}` vs `{{fmt_money(v2_reference_metrics.loc['oos', 'net_pnl'])}}`.",
    f"- Le benchmark buy & hold reste un repere de marche, pas de baseline metier principal.",
    f"- Si tu modifies les knobs du premier bloc et qu un variant correspondant existe dans l export V3, le notebook se rejoue proprement sans elargir la campagne.",
]
{mgc_extra}
display(Markdown("\\n".join(conclusion_lines)))
"""
    )


def build_notebook(ctx: NotebookContext) -> nbf.NotebookNode:
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
        _title_cell(ctx),
        _imports_cell(),
        _parameter_cell(ctx),
        _load_and_replay_cell(),
        _quick_read_markdown(),
        _quick_read_cell(ctx),
        _parameters_markdown(),
        _parameters_cell(),
        _research_markdown(),
        _research_cell(ctx),
        _curves_markdown(),
        _curves_cell(ctx),
        _heatmaps_markdown(),
        _heatmaps_cell(),
        _profile_markdown(),
        _profile_cell(ctx),
        _conclusion_markdown(),
        _conclusion_cell(ctx),
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


def _output_path(symbol: str) -> Path:
    return NOTEBOOKS_ROOT / f"volume_climax_pullback_v3_{symbol}_client.ipynb"


def _executed_output_path(symbol: str) -> Path:
    return NOTEBOOKS_ROOT / f"volume_climax_pullback_v3_{symbol}_client.executed.ipynb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-root",
        type=Path,
        default=find_latest_export(),
        help="Audited V3 export root to load.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=list(SYMBOLS),
        help="Subset of symbols to build. Defaults to all four.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute each generated notebook after writing it.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Notebook execution timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contexts = _build_contexts(args.export_root)
    for raw_symbol in args.symbols:
        symbol = str(raw_symbol).upper()
        if symbol not in contexts:
            raise ValueError(f"Unsupported symbol {symbol!r}.")
        ctx = contexts[symbol]
        notebook = build_notebook(ctx)
        output_path = write_notebook(notebook, _output_path(symbol))
        print(f"Notebook written to {output_path}")
        if args.execute:
            executed_path = execute_notebook(output_path, _executed_output_path(symbol), timeout_seconds=args.timeout_seconds)
            print(f"Executed notebook written to {executed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
