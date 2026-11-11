"""Build the interactive IS/OOS Sharpe view from an executed refinement grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from export_signal_sheets_pdf import ROOT, _interactive_plot_html


RESULT_ROOT = ROOT / "research" / "signal_sheet_parameter_refinement"
ENSEMBLE_DIR = ROOT / "strategy_registry" / "psf_mnq" / "frozen_ensembles"
RETAINED_MARKER = {"size": 17, "symbol": "square-open", "color": "#0875d1", "line": {"width": 4, "color": "#0875d1"}}
LABELS = {
    "nr_lookback": "Lookback NR",
    "entry_buffer_ticks": "Buffer d’entrée (ticks)",
    "target_setup_range_multiple": "Cible (× range)",
    "atr_period": "Période ATR",
    "atr_multiplier": "Multiplicateur ATR",
    "stop_atr": "Stop (× ATR)",
    "daily_atr_period": "Période ATR journalière",
    "entry_atr_multiple": "Multiple d’entrée ATR",
    "minimum_gap_atr_multiple": "Seuil de gap (× ATR)",
    "volume_threshold_multiple": "Seuil de volume",
    "entry_channel_length": "Canal d’entrée",
    "exit_channel_length": "Canal de sortie",
    "fast_pair_scale": "Échelle EMA rapide",
    "slow_tail_scale": "Échelle EMA lente",
}


def frozen_region(sheet: str) -> dict[str, list[float]] | None:
    """Parameter region of the frozen vote ensemble built for this sheet, if any."""
    for path in sorted(ENSEMBLE_DIR.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("sheet") == sheet:
            region = manifest["parameter_region"]
            return {axis: values for axis, values in region.items() if isinstance(values, list)}
    return None


def retained(frame: pd.DataFrame, region: dict[str, list[float]] | None) -> pd.DataFrame:
    if region is None:
        return frame.iloc[0:0]
    mask = pd.Series(True, index=frame.index)
    for axis, values in region.items():
        mask &= frame[axis].round(9).isin([round(float(v), 9) for v in values])
    return frame[mask]


def cube(frame: pd.DataFrame, params: list[str], title: str, region: dict[str, list[float]] | None = None) -> go.Figure:
    figure = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                           subplot_titles=("IS", "OOS / validation"), horizontal_spacing=.03)
    cmin, cmax = float(frame["sharpe_annualized"].min()), float(frame["sharpe_annualized"].max())
    for column, scope in enumerate(("is", "oos"), 1):
        scoped = frame[frame["scope"].eq(scope)]
        custom = scoped[["profit_factor", "max_drawdown_usd", "trade_count", "worst_day_usd", "net_pnl_usd"]].to_numpy(float)
        figure.add_trace(go.Scatter3d(
            x=scoped[params[0]], y=scoped[params[1]], z=scoped[params[2]], mode="markers",
            name=scope.upper(), showlegend=column == 1, customdata=custom,
            marker={"size": 9, "color": scoped["sharpe_annualized"], "colorscale": "Viridis", "cmin": cmin, "cmax": cmax,
                    "showscale": column == 2, "colorbar": {"title": "Sharpe", "x": 1.01, "thickness": 18},
                    "line": {"width": .7, "color": "#26384d"}},
            hovertemplate=(f"{LABELS[params[0]]} : %{{x}}<br>{LABELS[params[1]]} : %{{y}}<br>{LABELS[params[2]]} : %{{z}}"
                           "<br>Sharpe : %{marker.color:.3f}<br>PF : %{customdata[0]:.3f}<br>maxDD : %{customdata[1]:,.2f} USD"
                           "<br>Trades : %{customdata[2]:.0f}<br>Pire journée : %{customdata[3]:,.2f} USD<br>PnL net : %{customdata[4]:,.2f} USD<extra></extra>"),
        ), row=1, col=column)
        kept = retained(scoped, region)
        if not kept.empty:
            figure.add_trace(go.Scatter3d(
                x=kept[params[0]], y=kept[params[1]], z=kept[params[2]], mode="markers",
                name=f"{len(kept)} sous-modèles retenus", legendgroup="retained", showlegend=column == 1,
                marker=RETAINED_MARKER, customdata=kept[["sharpe_annualized"]].to_numpy(float),
                hovertemplate="Sous-modèle retenu<br>Sharpe : %{customdata[0]:.3f}<extra></extra>",
            ), row=1, col=column)
    axis = {"backgroundcolor": "#f4f7fa", "gridcolor": "#cbd6e2", "showbackground": True}
    scene = {"xaxis": {**axis, "title": LABELS[params[0]]}, "yaxis": {**axis, "title": LABELS[params[1]]},
             "zaxis": {**axis, "title": LABELS[params[2]]}, "aspectmode": "cube"}
    figure.update_layout(title={"text": title, "x": .5}, scene=scene, scene2=scene, height=620,
                         margin={"l": 0, "r": 75, "t": 85, "b": 20}, paper_bgcolor="white")
    return figure


def response_surface(frame: pd.DataFrame, params: list[str], title: str, region: dict[str, list[float]] | None = None) -> go.Figure:
    figure = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "surface"}]],
                           subplot_titles=("IS", "OOS / validation"), horizontal_spacing=.05)
    cmin, cmax = float(frame["sharpe_annualized"].min()), float(frame["sharpe_annualized"].max())
    metrics = ["profit_factor", "max_drawdown_usd", "trade_count", "worst_day_usd"]
    for column, scope in enumerate(("is", "oos"), 1):
        scoped = frame[frame["scope"].eq(scope)]
        xs, ys = sorted(scoped[params[0]].unique()), sorted(scoped[params[1]].unique())
        pivot = {
            name: scoped.pivot(index=params[1], columns=params[0], values=name).reindex(index=ys, columns=xs)
            for name in ("sharpe_annualized", *metrics)
        }
        z = pivot["sharpe_annualized"].to_numpy(float)
        custom = [[[pivot[name].to_numpy(float)[row][col] for name in metrics] for col in range(len(xs))] for row in range(len(ys))]
        figure.add_trace(go.Surface(
            x=xs, y=ys, z=z, surfacecolor=z, cmin=cmin, cmax=cmax, colorscale="Viridis",
            showscale=column == 2, colorbar={"title": "Sharpe", "x": 1.01, "thickness": 18},
            customdata=custom,
            hovertemplate=(f"{LABELS[params[0]]} : %{{x}}<br>{LABELS[params[1]]} : %{{y}}<br>Sharpe : %{{z:.3f}}"
                           "<br>PF : %{customdata[0]:.3f}<br>maxDD : %{customdata[1]:,.2f} USD"
                           "<br>Trades : %{customdata[2]:.0f}<br>Pire journée : %{customdata[3]:,.2f} USD<extra></extra>"),
        ), row=1, col=column)
        kept = retained(scoped, region)
        if not kept.empty:
            figure.add_trace(go.Scatter3d(
                x=kept[params[0]], y=kept[params[1]], z=kept["sharpe_annualized"], mode="markers",
                name=f"{len(kept)} sous-modèles retenus", legendgroup="retained", showlegend=column == 1,
                marker=RETAINED_MARKER,
                hovertemplate="Sous-modèle retenu<br>Sharpe : %{z:.3f}<extra></extra>",
            ), row=1, col=column)
    scene = {"xaxis": {"title": LABELS[params[0]]}, "yaxis": {"title": LABELS[params[1]]}, "zaxis": {"title": "Sharpe"}}
    figure.update_layout(title={"text": title, "x": .5}, scene=scene, scene2=scene, height=610,
                         margin={"l": 0, "r": 75, "t": 85, "b": 25}, paper_bgcolor="white")
    return figure


def build(sleeve: str) -> Path:
    directory = RESULT_ROOT / sleeve
    frame = pd.read_csv(directory / "grid_metrics.csv")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    params = list(manifest["grids"])
    title = manifest["sheet"].replace("_", " ")
    region = frozen_region(manifest["sheet"])
    note = "Tourner, zoomer et survoler chaque point pour lire Sharpe, PF, maxDD, trades et pire journée."
    if region is not None:
        note += " Les carrés bleus sont les sous-modèles de l’ensemble gelé ; la couleur reste une métrique par cellule."
    if len(params) == 3:
        block = _interactive_plot_html(f"{sleeve}-cube", cube(frame, params, f"{title} — cube du Sharpe IS/OOS", region), note)
    else:
        block = _interactive_plot_html(
            f"{sleeve}-surface", response_surface(frame, params, f"{title} — surface du Sharpe IS/OOS", region),
            note + " La surface relie uniquement les cellules testées et ne crée aucun backtest intermédiaire.",
        )
    output = directory / "sheet_visuals.html"
    output.write_text(block, encoding="utf-8")

    pivot = frame.pivot(index=["cell_id", *params], columns="scope", values="sharpe_annualized").reset_index()
    positive = pivot[(pivot["is"] > 0) & (pivot["oos"] > 0)]
    summary = {
        "cell_count": int(frame["cell_id"].nunique()), "positive_sharpe_is_and_oos": int(len(positive)),
        "axis_median_sharpe": {
            param: frame[frame["scope"].isin(["is", "oos"])].groupby(["scope", param])["sharpe_annualized"].median().unstack(0).reset_index().to_dict("records")
            for param in params
        },
        "selection_performed": False,
    }
    (directory / "visual_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleeve", required=True, choices=[f"s{index:02d}" for index in range(3, 10)])
    args = parser.parse_args()
    build(args.sleeve)


if __name__ == "__main__":
    main()
