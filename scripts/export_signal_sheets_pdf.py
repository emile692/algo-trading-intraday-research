"""Build print-ready and interactive HTML for the nine client signal sheets.

The HTML files are rendered to PDF with a Chromium-compatible browser. Mermaid
is deliberately supplied as a local script so the final render is reproducible
and does not depend on network access.  On screen, the static ORB robustness
image is replaced by manipulable, side-by-side IS/OOS Plotly cubes; print uses
the corresponding static IS/OOS figure.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import markdown
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE_DIR = ROOT / "docs" / "signal_sheets"
SHEET_PATTERN = re.compile(r"^\d{2}_.+\.md$")
MERMAID_BLOCK = re.compile(
    r'<pre><code class="language-mermaid">(.*?)</code></pre>', re.DOTALL
)
IMAGE_SRC = re.compile(r'(<img\s+[^>]*src=")([^"]+)(")')
ORB_CUBE_IMAGE = re.compile(
    r'<p><img(?P<attrs>[^>]*\bsrc="assets/01_orb_is_heatmap\.png"[^>]*)\s*/?></p>'
)
VOLATILITY_CUBE_IMAGE = re.compile(
    r'<p><img(?P<attrs>[^>]*\bsrc="assets/02_volatility_breakout_is_heatmap\.png"[^>]*)\s*/?></p>'
)
INTERACTIVE_IMAGE_PATTERNS = {
    "orb_indicator": re.compile(r'<p><img(?P<attrs>[^>]*\bsrc="assets/01_indicator_orb_vwap\.png"[^>]*)\s*/?></p>'),
    "atr_indicator": re.compile(r'<p><img(?P<attrs>[^>]*\bsrc="assets/indicator_atr\.png"[^>]*)\s*/?></p>'),
    "volatility_indicator": re.compile(r'<p><img(?P<attrs>[^>]*\bsrc="assets/02_indicator_atr_breakout\.png"[^>]*)\s*/?></p>'),
    "equity_01": re.compile(r'<p><img(?P<attrs>[^>]*\bsrc="assets/01_equity_curve\.png"[^>]*)\s*/?></p>'),
    "equity_02": re.compile(r'<p><img(?P<attrs>[^>]*\bsrc="assets/02_equity_curve\.png"[^>]*)\s*/?></p>'),
}
for _number, _names in {
    3: ("03_indicator_compression.png", "03_equity_curve.png"),
    4: ("04_indicator_trend_flip.png", "04_equity_curve.png"),
    5: ("05_indicator_atr_breakout.png", "05_equity_curve.png"),
    6: ("06_indicator_gap.png", "06_equity_curve.png"),
    7: ("07_indicator_donchian.png", "07_equity_curve.png"),
    8: ("08_indicator_ema_pullback.png", "08_equity_curve.png"),
    9: ("09_indicator_atr_breakout.png", "09_equity_curve.png"),
}.items():
    for _kind, _filename in zip(("indicator", "equity"), _names):
        INTERACTIVE_IMAGE_PATTERNS[f"s{_number:02d}_{_kind}"] = re.compile(
            rf'<p><img(?P<attrs>[^>]*\bsrc="[^"]*{re.escape(_filename)}"[^>]*)\s*/?></p>'
        )
ORB_TABLE_DIR = "data/exports/atr_local_robustness_20260320_194355/tables"
VOLATILITY_CUBE_DATA = ROOT / "research" / "signal_sheet_02_cube" / "cube_metrics.csv"
EQUITY_DATA = {
    1: ROOT / "research" / "signal_sheet_01_atr18" / "daily_results.csv",
    **{number: ROOT / "research" / f"signal_sheet_{number:02d}_cluster" / "daily_results.csv" for number in range(2, 10)},
}
SHEET_LABELS = {
    1: "01 · MNQ ORB + VWAP + ATR",
    2: "02 · MNQ volatility breakout",
    3: "03 · MNQ compression breakout",
    4: "04 · MNQ ATR trend flip",
    5: "05 · MGC volatility breakout",
    6: "06 · MGC gap continuation",
    7: "07 · MBT rolling range breakout",
    8: "08 · MBT trend pullback MA",
    9: "09 · MBT volatility breakout",
}
OOS_START = {
    1: "2024-02-28", 2: "2024-01-01", 3: "2024-01-01", 4: "2024-01-01",
    5: "2024-01-01", 6: "2024-01-01", 7: "2025-05-29", 8: "2025-05-29", 9: "2025-05-29",
}
OOS_LABEL = {
    1: "Historique OOS réutilisé",
    2: "OOS / validation réutilisée",
    3: "OOS / validation réutilisée",
    4: "OOS / validation réutilisée",
    5: "OOS / validation réutilisée",
    6: "OOS / validation réutilisée",
    7: "Legacy OOS réutilisé",
    8: "Legacy OOS réutilisé",
    9: "Legacy OOS réutilisé",
}

CSS = r"""
@page { size: A4; margin: 15mm 14mm 17mm; }
* { box-sizing: border-box; }
html { color: #172033; background: #fff; font-family: "Segoe UI", Arial, sans-serif; }
body { margin: 0; font-size: 9.45pt; line-height: 1.48; }
h1, h2, h3 { color: #102b4e; page-break-after: avoid; }
h1 { margin: 0 0 5mm; font-size: 22pt; line-height: 1.1; letter-spacing: -.02em; }
h2 { margin: 6mm 0 2.2mm; padding-bottom: 1.1mm; border-bottom: 1px solid #b9c8d9; font-size: 14pt; }
h3 { margin: 4mm 0 1.5mm; font-size: 11.2pt; }
p { margin: 0 0 2.6mm; text-align: justify; hyphens: auto; }
ul, ol { margin: 1.5mm 0 3mm; padding-left: 6mm; }
li { margin-bottom: .8mm; text-align: justify; hyphens: auto; }
strong { color: #102b4e; }
a { color: #1769aa; text-decoration: none; }
code { padding: .15em .35em; border-radius: 3px; background: #edf3f8; color: #17375e; font-family: Consolas, monospace; font-size: 8.6pt; }
pre:not(.mermaid) { overflow-wrap: anywhere; white-space: pre-wrap; padding: 3.5mm; border: 1px solid #c9d5e2; border-radius: 5px; background: #f6f8fb; page-break-inside: avoid; }
pre:not(.mermaid) code { padding: 0; background: transparent; }
table { width: 100%; margin: 2.5mm 0 4mm; border-collapse: collapse; font-size: 8.25pt; page-break-inside: avoid; }
thead { display: table-header-group; }
th { background: #17375e; color: white; text-align: left; }
th, td { padding: 1.8mm 2mm; border: 1px solid #c5d0dc; vertical-align: top; }
tr:nth-child(even) td { background: #f4f7fa; }
blockquote { margin: 3mm 0; padding: 2.5mm 4mm; border-left: 4px solid #3a80b9; background: #eef5fb; }
img { display: block; max-width: 100%; max-height: 126mm; width: auto; height: auto; margin: 3mm auto 4mm; object-fit: contain; page-break-inside: avoid; }
.interactive-cube { width: 100%; min-height: 620px; margin: 4mm auto 5mm; border: 1px solid #c9d5e2; border-radius: 8px; background: #fff; overflow: hidden; }
.interactive-figure { width: 100%; min-height: 450px; margin: 4mm auto 5mm; border: 1px solid #c9d5e2; border-radius: 8px; background: #fff; overflow: hidden; }
.interactive-cube-note { margin: -2mm 0 4mm; color: #526174; font-size: 8.5pt; text-align: center; }
.interactive-print { display: none; }
.document-meta { margin: 0 0 6mm; padding: 2.3mm 3mm; border-radius: 4px; background: #102b4e; color: #fff; font-size: 8.5pt; letter-spacing: .02em; }
.signal-sheet { page-break-before: always; }
.signal-sheet.first { page-break-before: auto; }
.book-section { page-break-before: always; }
.mermaid { margin: 4mm auto 5mm; padding: 3mm; border: 1px solid #c9d5e2; border-radius: 6px; background: #fbfcfe; text-align: center; page-break-inside: avoid; }
.mermaid svg { max-width: 100% !important; max-height: 215mm !important; width: auto !important; height: auto !important; }
.cover { min-height: 255mm; display: flex; flex-direction: column; justify-content: center; page-break-after: always; color: #fff; background: linear-gradient(145deg, #0d2747, #1769aa); margin: -15mm -14mm -17mm; padding: 30mm 24mm; }
.cover .eyebrow { font-size: 11pt; text-transform: uppercase; letter-spacing: .16em; opacity: .8; }
.cover h1 { margin: 8mm 0 5mm; color: #fff; font-size: 34pt; }
.cover p { max-width: 145mm; font-size: 14pt; }
.cover .rule { width: 42mm; height: 1.5mm; margin: 8mm 0; background: #65c3ff; }
.cover .date { margin-top: 18mm; font-size: 10pt; opacity: .78; }
.toc { page-break-after: always; }
.toc h1 { margin-bottom: 9mm; }
.toc ol { padding-left: 8mm; font-size: 12pt; }
.toc li { margin: 3mm 0; padding-left: 2mm; }
.toc a { color: #102b4e; }
.client-note { margin: 7mm 0 0; padding: 4mm; border: 1px solid #d5a744; background: #fff8e8; page-break-inside: avoid; }
@media (max-width: 900px) { .interactive-cube { min-height: 760px; } }
@media print {
  .interactive-only { display: none !important; }
  .interactive-print { display: block; }
}
"""


def _git_csv(path: str) -> pd.DataFrame:
    """Read a tracked research table without restoring deleted bulk exports."""
    raw = subprocess.run(
        ["git", "show", f"HEAD:{path}"], cwd=ROOT, check=True, capture_output=True
    ).stdout.decode("utf-8")
    return pd.read_csv(io.StringIO(raw))


def orb_cube_figure() -> go.Figure:
    """Return comparable IS and OOS cubes using the frozen research tables."""
    scopes = [
        ("IS", _git_csv(f"{ORB_TABLE_DIR}/metrics_is.csv"), "is_sharpe_ratio"),
        ("OOS", _git_csv(f"{ORB_TABLE_DIR}/metrics_oos.csv"), "oos_sharpe_ratio"),
    ]
    periods = sorted(set().union(*(set(frame["atr_period"]) for _, frame, _ in scopes)))
    lows = sorted(set().union(*(set(frame["q_low_pct"]) for _, frame, _ in scopes)))
    highs = sorted(set().union(*(set(frame["q_high_pct"]) for _, frame, _ in scopes)))
    period_pos = {value: index for index, value in enumerate(periods)}
    low_pos = {value: index for index, value in enumerate(lows)}
    high_pos = {value: index for index, value in enumerate(highs)}
    values = pd.concat([frame[column] for _, frame, column in scopes], ignore_index=True)
    cmin, cmax = float(values.min()), float(values.max())

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("IS — construction", "OOS — évaluation"),
        horizontal_spacing=0.03,
    )
    for column_index, (scope, frame, metric) in enumerate(scopes, 1):
        hover = [
            (
                f"Période ATR : {row.atr_period:g}<br>"
                f"Quantile bas : {row.q_low_pct:g} %<br>"
                f"Quantile haut : {row.q_high_pct:g} %<br>"
                f"Sharpe {scope} : {getattr(row, metric):.3f}"
            )
            for row in frame.itertuples(index=False)
        ]
        figure.add_trace(
            go.Scatter3d(
                x=frame["atr_period"].map(period_pos),
                y=frame["q_low_pct"].map(low_pos),
                z=frame["q_high_pct"].map(high_pos),
                mode="markers",
                name=f"Combinaisons {scope}",
                legendgroup="grid",
                showlegend=column_index == 1,
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                marker={
                    "size": 10,
                    "opacity": 0.88,
                    "color": frame[metric],
                    "colorscale": "Viridis",
                    "cmin": cmin,
                    "cmax": cmax,
                    "showscale": column_index == 2,
                    "colorbar": {"title": "Sharpe<br>annualisé", "x": 1.01, "thickness": 18},
                    "line": {"width": 0.6, "color": "rgba(30,45,65,.55)"},
                },
            ),
            row=1,
            col=column_index,
        )
        retained = frame[
            frame["atr_period"].isin([14, 16, 18])
            & frame["q_low_pct"].isin([20, 25, 30])
            & frame["q_high_pct"].isin([80, 85])
        ]
        figure.add_trace(
            go.Scatter3d(
                x=retained["atr_period"].map(period_pos),
                y=retained["q_low_pct"].map(low_pos),
                z=retained["q_high_pct"].map(high_pos),
                mode="markers",
                name="18 sous-modèles retenus",
                legendgroup="retained",
                showlegend=column_index == 1,
                text=[
                    f"Retenu · ATR {row.atr_period:g} · {row.q_low_pct:g}/{row.q_high_pct:g}<br>Sharpe {scope} : {getattr(row, metric):.3f}"
                    for row in retained.itertuples(index=False)
                ],
                hovertemplate="%{text}<extra></extra>",
                marker={
                    "size": 16,
                    "symbol": "square-open",
                    "color": "#0875d1",
                    "line": {"width": 4, "color": "#0875d1"},
                },
            ),
            row=1,
            col=column_index,
        )

    axis = {
        "backgroundcolor": "#f4f7fa",
        "gridcolor": "#cbd6e2",
        "showbackground": True,
        "zeroline": False,
    }
    scene = {
        "xaxis": {**axis, "title": "Période ATR", "tickvals": list(period_pos.values()), "ticktext": [str(v) for v in periods]},
        "yaxis": {**axis, "title": "Quantile bas (%)", "tickvals": list(low_pos.values()), "ticktext": [str(v) for v in lows]},
        "zaxis": {**axis, "title": "Quantile haut (%)", "tickvals": list(high_pos.values()), "ticktext": [str(v) for v in highs]},
        "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.15}},
        "aspectmode": "cube",
    }
    figure.update_layout(
        title={"text": "MNQ ORB — robustesse des trois paramètres ATR", "x": 0.5},
        scene=scene,
        scene2=scene,
        margin={"l": 0, "r": 70, "t": 85, "b": 20},
        paper_bgcolor="white",
        font={"family": "Segoe UI, Arial, sans-serif", "color": "#172033"},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.04, "yanchor": "bottom"},
        height=620,
    )
    return figure


def orb_cube_html() -> str:
    payload = json.dumps(orb_cube_figure().to_plotly_json(), ensure_ascii=False)
    return f"""<div class="interactive-only">
<div id="orb-is-oos-cube" class="interactive-cube"></div>
<p class="interactive-cube-note">Glisser pour tourner les deux vues ensemble · molette pour zoomer · double-clic pour réinitialiser · survoler un point pour lire sa valeur</p>
<script>
window.addEventListener("load", () => {{
  const figure = {payload};
  Plotly.newPlot("orb-is-oos-cube", figure.data, figure.layout, {{responsive:true, displaylogo:false, scrollZoom:true}})
    .then((plot) => {{
      let syncing = false;
      plot.on("plotly_relayout", (change) => {{
        if (syncing) return;
        const source = change["scene.camera"] ? "scene" : (change["scene2.camera"] ? "scene2" : null);
        if (!source) return;
        syncing = true;
        const target = source === "scene" ? "scene2.camera" : "scene.camera";
        Plotly.relayout(plot, {{[target]: change[source + ".camera"]}}).finally(() => {{ syncing = false; }});
      }});
    }});
}});
</script>
</div>"""


def volatility_breakout_cube_figure() -> go.Figure:
    """Return synchronized IS/OOS cubes for the three volatility-breakout parameters."""
    frame = pd.read_csv(VOLATILITY_CUBE_DATA)
    periods = sorted(frame["daily_atr_period"].unique())
    multiples = sorted(frame["entry_atr_multiple"].unique())
    durations = sorted(frame["max_holding_minutes"].unique())
    period_pos = {value: index for index, value in enumerate(periods)}
    multiple_pos = {value: index for index, value in enumerate(multiples)}
    duration_pos = {value: index for index, value in enumerate(durations)}
    scopes = (("IS", "is"), ("OOS / validation", "oos"))
    metric_columns = [f"{prefix}_sharpe_annualized" for _, prefix in scopes]
    cmin = float(frame[metric_columns].min().min())
    cmax = float(frame[metric_columns].max().max())

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("IS — construction", "OOS / validation"),
        horizontal_spacing=0.03,
    )
    for column_index, (scope, prefix) in enumerate(scopes, 1):
        metric = f"{prefix}_sharpe_annualized"
        hover = [
            (
                f"Période ATR : {row.daily_atr_period:g} séances<br>"
                f"Multiple d’entrée : {row.entry_atr_multiple:.1f} ATR<br>"
                f"Durée maximale : {row.max_holding_minutes:g} min<br>"
                f"Sharpe {scope} : {getattr(row, metric):.3f}<br>"
                f"PF : {getattr(row, prefix + '_profit_factor'):.3f}<br>"
                f"Trades : {getattr(row, prefix + '_trade_count'):g}"
            )
            for row in frame.itertuples(index=False)
        ]
        figure.add_trace(
            go.Scatter3d(
                x=frame["daily_atr_period"].map(period_pos),
                y=frame["entry_atr_multiple"].map(multiple_pos),
                z=frame["max_holding_minutes"].map(duration_pos),
                mode="markers",
                name=f"Combinaisons {scope}",
                legendgroup="grid",
                showlegend=column_index == 1,
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                marker={
                    "size": 10,
                    "opacity": 0.88,
                    "color": frame[metric],
                    "colorscale": "Viridis",
                    "cmin": cmin,
                    "cmax": cmax,
                    "showscale": column_index == 2,
                    "colorbar": {"title": "Sharpe<br>annualisé", "x": 1.01, "thickness": 18},
                    "line": {"width": 0.6, "color": "rgba(30,45,65,.55)"},
                },
            ),
            row=1,
            col=column_index,
        )
        retained = frame[
            frame["daily_atr_period"].isin(range(10, 23, 2))
            & frame["entry_atr_multiple"].isin([0.5, 0.6])
            & frame["max_holding_minutes"].isin([270, 300, 360])
        ]
        figure.add_trace(
            go.Scatter3d(
                x=retained["daily_atr_period"].map(period_pos),
                y=retained["entry_atr_multiple"].map(multiple_pos),
                z=retained["max_holding_minutes"].map(duration_pos),
                mode="markers",
                name="42 sous-modèles retenus",
                legendgroup="retained",
                showlegend=column_index == 1,
                text=[
                    (
                        f"Sous-modèle retenu · ATR {row.daily_atr_period:g} · multiple {row.entry_atr_multiple:.1f} · "
                        f"{row.max_holding_minutes:g} min<br>Sharpe {scope} : {getattr(row, metric):.3f}"
                    )
                    for row in retained.itertuples(index=False)
                ],
                hovertemplate="%{text}<extra></extra>",
                marker={"size": 17, "symbol": "square-open", "color": "#0875d1", "line": {"width": 4, "color": "#0875d1"}},
            ),
            row=1,
            col=column_index,
        )

    axis = {"backgroundcolor": "#f4f7fa", "gridcolor": "#cbd6e2", "showbackground": True, "zeroline": False}
    scene = {
        "xaxis": {**axis, "title": "Période ATR", "tickvals": list(period_pos.values()), "ticktext": [str(v) for v in periods]},
        "yaxis": {**axis, "title": "Multiple d’entrée", "tickvals": list(multiple_pos.values()), "ticktext": [f"{v:.1f}" for v in multiples]},
        "zaxis": {**axis, "title": "Durée max (min)", "tickvals": list(duration_pos.values()), "ticktext": [str(v) for v in durations]},
        "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.15}},
        "aspectmode": "cube",
    }
    figure.update_layout(
        title={"text": "MNQ volatility breakout — robustesse des trois paramètres", "x": 0.5},
        scene=scene,
        scene2=scene,
        margin={"l": 0, "r": 70, "t": 85, "b": 20},
        paper_bgcolor="white",
        font={"family": "Segoe UI, Arial, sans-serif", "color": "#172033"},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.04, "yanchor": "bottom"},
        height=620,
    )
    return figure


def volatility_breakout_cube_html() -> str:
    payload = json.dumps(volatility_breakout_cube_figure().to_plotly_json(), ensure_ascii=False)
    return f"""<div class="interactive-only">
<div id="volatility-breakout-is-oos-cube" class="interactive-cube"></div>
<p class="interactive-cube-note">Glisser pour tourner les deux vues ensemble · molette pour zoomer · double-clic pour réinitialiser · survoler un point pour lire Sharpe, PF et nombre de trades</p>
<script>
window.addEventListener("load", () => {{
  const figure = {payload};
  Plotly.newPlot("volatility-breakout-is-oos-cube", figure.data, figure.layout, {{responsive:true, displaylogo:false, scrollZoom:true}})
    .then((plot) => {{
      let syncing = false;
      plot.on("plotly_relayout", (change) => {{
        if (syncing) return;
        const source = change["scene.camera"] ? "scene" : (change["scene2.camera"] ? "scene2" : null);
        if (!source) return;
        syncing = true;
        const target = source === "scene" ? "scene2.camera" : "scene.camera";
        Plotly.relayout(plot, {{[target]: change[source + ".camera"]}}).finally(() => {{ syncing = false; }});
      }});
    }});
}});
</script>
</div>"""


def _interactive_plot_html(div_id: str, figure: go.Figure, note: str) -> str:
    """Embed a responsive Plotly figure without any network dependency."""
    payload = json.dumps(figure.to_plotly_json(), ensure_ascii=False, cls=PlotlyJSONEncoder)
    return f"""<div class="interactive-only">
<div id="{div_id}" class="interactive-figure"></div>
<p class="interactive-cube-note">{html.escape(note)}</p>
<script>
window.addEventListener("load", () => {{
  const figure = {payload};
  Plotly.newPlot("{div_id}", figure.data, figure.layout, {{
    responsive:true, displaylogo:false, scrollZoom:true,
    modeBarButtonsToRemove:["lasso2d", "select2d"]
  }});
}});
</script>
</div>"""


def orb_indicator_figure() -> go.Figure:
    closes = [100.2, 100.7, 100.1, 101.0, 100.6, 101.1, 101.3, 101.7, 102.1, 102.8, 103.2, 103.8]
    opens = [100.0, *closes[:-1]]
    high_offsets = [.4, .3, .25, .35, .2, .25, .2, .3, .25, .3, .25, .3]
    low_offsets = [.2, .25, .3, .2, .25, .2, .15, .2, .2, .15, .2, .2]
    highs = [max(o, c) + d for o, c, d in zip(opens, closes, high_offsets)]
    lows = [min(o, c) - d for o, c, d in zip(opens, closes, low_offsets)]
    volumes = [12, 15, 11, 17, 10, 9, 10, 12, 16, 20, 18, 17]
    typical = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    cumulative_pv, cumulative_volume, vwap = 0.0, 0.0, []
    for price, volume in zip(typical, volumes):
        cumulative_pv += price * volume
        cumulative_volume += volume
        vwap.append(cumulative_pv / cumulative_volume)
    x = list(range(len(closes)))
    or_high, or_low = max(highs[:5]), min(lows[:5])
    figure = go.Figure()
    figure.add_trace(go.Candlestick(
        x=x, open=opens, high=highs, low=lows, close=closes,
        name="Prix", increasing_line_color="#25855a", decreasing_line_color="#c45a4a",
        hovertemplate="Minute %{x}<br>O %{open:.2f}<br>H %{high:.2f}<br>L %{low:.2f}<br>C %{close:.2f}<extra></extra>",
    ))
    figure.add_trace(go.Scatter(x=x, y=vwap, mode="lines+markers", name="VWAP cumulée",
                                line={"color": "#d9a21b", "width": 3},
                                hovertemplate="Minute %{x}<br>VWAP %{y:.2f}<extra></extra>"))
    figure.add_hrect(y0=or_low, y1=or_high, x0=-.5, x1=4.5, fillcolor="#dbe8f4", opacity=.7,
                     line_width=0, annotation_text="Opening range 09:30–09:45", annotation_position="top left")
    figure.add_hline(y=or_high, line_dash="dash", line_color="#7b8794")
    figure.add_hline(y=or_low, line_dash="dash", line_color="#7b8794")
    figure.add_trace(go.Scatter(x=[9], y=[closes[9]], mode="markers", name="Cassure validée",
                                marker={"size": 14, "color": "#1769aa"},
                                hovertemplate="Cassure validée<br>Clôture %{y:.2f}<extra></extra>"))
    figure.add_annotation(x=9, y=closes[9], text="Cassure clôturée au-dessus de l’OR<br>+ prix au-dessus de la VWAP<br>+ filtre ATR favorable",
                          showarrow=True, ax=-135, ay=-55, arrowcolor="#102b4e", bgcolor="rgba(255,255,255,.9)")
    figure.update_layout(title={"text": "ORB + VWAP + filtre de régime ATR", "x": .5},
                         xaxis_title="Minutes successives (schéma)", yaxis_title="Prix",
                         xaxis_rangeslider_visible=False, hovermode="x unified", height=470,
                         margin={"l": 60, "r": 30, "t": 70, "b": 55}, paper_bgcolor="white")
    return figure


def atr_indicator_figure() -> go.Figure:
    figure = make_subplots(rows=1, cols=2, subplot_titles=("True Range d’une séance", "Lissage ATR : amplitude, pas direction"), horizontal_spacing=.1)
    figure.add_trace(go.Scatter(x=[.1, 1.0], y=[100, 100], mode="lines", name="Clôture J−1",
                                line={"color": "#7b8794", "dash": "dash"}), row=1, col=1)
    figure.add_trace(go.Scatter(x=[1.5, 1.5], y=[103, 108], mode="lines+markers", name="Range J",
                                line={"color": "#25855a", "width": 7}, marker={"size": 9, "color": "#102b4e"},
                                hovertemplate="%{y:.1f} points<extra>Range J</extra>"), row=1, col=1)
    figure.add_annotation(x=1.5, y=105.5, text="High − Low", showarrow=True, ax=75, ay=0, row=1, col=1)
    figure.add_annotation(x=.8, y=101.5, text="Gap pris en compte", showarrow=True, ax=70, ay=35, row=1, col=1)
    tr = [72, 84, 76, 91, 110, 98, 121, 105, 95, 112, 126, 118, 109, 102]
    atr = [float(tr[0])]
    for value in tr[1:]:
        atr.append((13 * atr[-1] + value) / 14)
    sessions = list(range(1, len(tr) + 1))
    figure.add_trace(go.Bar(x=sessions, y=tr, name="True Range quotidien", marker_color="#b8cee2",
                            hovertemplate="Séance %{x}<br>TR %{y:.1f}<extra></extra>"), row=1, col=2)
    figure.add_trace(go.Scatter(x=sessions, y=atr, mode="lines+markers", name="ATR14 Wilder",
                                line={"color": "#1769aa", "width": 3},
                                hovertemplate="Séance %{x}<br>ATR14 %{y:.2f}<extra></extra>"), row=1, col=2)
    figure.update_xaxes(title_text="J−1 / J", range=[0, 3.4], row=1, col=1)
    figure.update_yaxes(title_text="Points de cours", range=[98.5, 109.5], row=1, col=1)
    figure.update_xaxes(title_text="Séances terminées", row=1, col=2)
    figure.update_yaxes(title_text="Points de cours", row=1, col=2)
    figure.update_layout(title={"text": "Comment lire l’ATR", "x": .5}, height=470, barmode="overlay",
                         hovermode="closest", legend={"orientation": "h", "x": .5, "xanchor": "center", "y": -0.14},
                         margin={"l": 60, "r": 30, "t": 80, "b": 90}, paper_bgcolor="white")
    return figure


def volatility_indicator_figure() -> go.Figure:
    open_price, atr, multiple = 20000.0, 200.0, .60
    upper, lower = open_price + multiple * atr, open_price - multiple * atr
    path = [open_price + fraction * atr for fraction in [0, .08, -.04, .18, .34, .58, .72, .55]]
    x = list(range(len(path)))
    hit = next(index for index, value in enumerate(path) if value >= upper)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=x, y=path, mode="lines+markers", name="Trajectoire intrajournalière",
                                line={"color": "#102b4e", "width": 3}, marker={"size": 8},
                                hovertemplate="Étape %{x}<br>Prix %{y:,.2f}<extra></extra>"))
    for value, label, color, dash in (
        (upper, "U = O + 0,60 × ATR", "#25855a", "dash"),
        (lower, "L = O − 0,60 × ATR", "#c45a4a", "dash"),
        (open_price, "Open O = stop d’invalidation", "#7b8794", "solid"),
    ):
        figure.add_trace(go.Scatter(x=[x[0], x[-1]], y=[value, value], mode="lines", name=label,
                                    line={"color": color, "dash": dash, "width": 2},
                                    hovertemplate=f"{html.escape(label)}<br>Prix %{{y:,.2f}}<extra></extra>"))
    figure.add_trace(go.Scatter(x=[hit], y=[path[hit]], mode="markers", name="Premier seuil touché → long",
                                marker={"size": 15, "color": "#25855a"},
                                hovertemplate="Premier seuil touché → long<br>Prix %{y:,.2f}<extra></extra>"))
    figure.add_annotation(x=hit, y=path[hit], text="Premier seuil touché → long", showarrow=True,
                          ax=-110, ay=-45, arrowcolor="#25855a", font={"color": "#25855a"})
    figure.add_annotation(x=.02, y=.05, xref="paper", yref="paper", xanchor="left", yanchor="bottom",
                          text="Exemple : ATR = 200 points<br>Distance entrée–stop ≈ 240 USD/contrat",
                          showarrow=False, bgcolor="rgba(255,255,255,.92)", bordercolor="#c5d0dc")
    figure.update_layout(title={"text": "MNQ — breakout normalisé par l’ATR", "x": .5},
                         xaxis_title="Temps intrajournalier (schéma)", yaxis_title="Prix",
                         hovermode="x unified", height=460,
                         margin={"l": 75, "r": 30, "t": 70, "b": 55}, paper_bgcolor="white")
    return figure


def compression_indicator_figure() -> go.Figure:
    opens = [100, 102, 101, 103, 102, 102.5]
    closes = [102, 101, 103, 102, 102.4, 104.5]
    highs = [103, 103.2, 104, 104.2, 103.0, 105]
    lows = [99, 100, 100.5, 100, 101.4, 102.2]
    labels = ["barre −4", "−3", "−2", "Mother bar", "Setup", "Breakout"]
    figure = go.Figure(go.Candlestick(
        x=labels, open=opens, high=highs, low=lows, close=closes, name="Prix",
        increasing_line_color="#25855a", decreasing_line_color="#c45a4a",
        hovertemplate="%{x}<br>O %{open:.2f}<br>H %{high:.2f}<br>L %{low:.2f}<br>C %{close:.2f}<extra></extra>",
    ))
    figure.add_shape(type="rect", x0=3.65, x1=4.35, y0=lows[4], y1=highs[4], xref="x", yref="y",
                     line={"color": "#1769aa", "width": 4}, fillcolor="rgba(0,0,0,0)")
    figure.add_hline(y=highs[4] + .25, line_dash="dash", line_color="#25855a", annotation_text="Buy stop + 1 tick")
    figure.add_hline(y=lows[4] - .25, line_dash="dash", line_color="#c45a4a", annotation_text="Sell stop − 1 tick")
    figure.add_annotation(x="Setup", y=highs[4], text="Inside bar + plus petite range (NR10)", showarrow=True, ax=-130, ay=-55)
    figure.update_layout(title={"text": "Compression : mother bar, inside bar et range de setup", "x": .5},
                         xaxis_title="Barres successives", yaxis_title="Prix", xaxis_rangeslider_visible=False,
                         height=460, margin={"l": 60, "r": 35, "t": 70, "b": 55}, paper_bgcolor="white")
    return figure


def trend_flip_indicator_figure() -> go.Figure:
    price = [100,100.5,101,101.4,101.1,100.7,100.2,99.6,99.2,99.5,100,100.6,101.3,102,102.6,103,103.5,104]
    padded = [price[0], price[0], *price]
    center = [sum(padded[i:i + 3]) / 3 for i in range(len(price))]
    upper, lower = [[value + delta for value in center] for delta in (1.15, -1.15)]
    x = list(range(len(price)))
    figure = go.Figure()
    for y, name, color, dash in (
        (price, "Clôture 15 min", "#102b4e", "solid"),
        (upper, "Bande haute ajustée ATR", "#25855a", "dash"),
        (lower, "Bande basse ajustée ATR", "#c45a4a", "dash"),
    ):
        figure.add_trace(go.Scatter(x=x, y=y, mode="lines+markers" if name == "Clôture 15 min" else "lines", name=name,
                                    line={"color": color, "width": 3 if name == "Clôture 15 min" else 2, "dash": dash},
                                    hovertemplate=f"Barre %{{x}}<br>{html.escape(name)} %{{y:.2f}}<extra></extra>"))
    figure.add_trace(go.Scatter(x=[12], y=[price[12]], mode="markers", name="Flip haussier",
                                marker={"size": 15, "color": "#1769aa"}, hovertemplate="Flip haussier confirmé<extra></extra>"))
    figure.add_annotation(x=12, y=price[12], text="Flip haussier confirmé à la clôture<br>→ entrée minute suivante", showarrow=True, ax=-120, ay=-55)
    figure.update_layout(title={"text": "ATR trend flip — bandes adaptatives et changement d’état", "x": .5},
                         xaxis_title="Barres 15 min terminées", yaxis_title="Prix", hovermode="x unified",
                         height=460, margin={"l": 60, "r": 35, "t": 70, "b": 55}, paper_bgcolor="white")
    return figure


def normalized_breakout_figure(instrument: str, open_price: float, atr: float, multiple: float, point_value: float) -> go.Figure:
    upper, lower = open_price + multiple * atr, open_price - multiple * atr
    path = [open_price + fraction * multiple * atr for fraction in [0, .12, -.06, .30, .58, .96, 1.18, .90]]
    hit = next(index for index, value in enumerate(path) if value >= upper)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=list(range(len(path))), y=path, mode="lines+markers", name="Trajectoire",
                                line={"color": "#102b4e", "width": 3}, hovertemplate="Étape %{x}<br>Prix %{y:,.2f}<extra></extra>"))
    for value, label, color, dash in (
        (upper, f"U = O + {multiple:.2f} × ATR", "#25855a", "dash"),
        (lower, f"L = O − {multiple:.2f} × ATR", "#c45a4a", "dash"),
        (open_price, "Open O = stop d’invalidation", "#7b8794", "solid"),
    ):
        figure.add_trace(go.Scatter(x=[0, len(path) - 1], y=[value, value], mode="lines", name=label,
                                    line={"color": color, "dash": dash, "width": 2}, hovertemplate=f"{html.escape(label)}<br>%{{y:,.2f}}<extra></extra>"))
    figure.add_trace(go.Scatter(x=[hit], y=[path[hit]], mode="markers", name="Premier seuil touché → long",
                                marker={"size": 15, "color": "#25855a"}, hovertemplate="Premier seuil touché → long<extra></extra>"))
    risk_usd = multiple * atr * point_value
    figure.add_annotation(x=.02, y=.05, xref="paper", yref="paper", xanchor="left", yanchor="bottom",
                          text=f"ATR = {atr:g} points<br>Distance entrée–stop ≈ {risk_usd:,.0f} USD/contrat",
                          showarrow=False, bgcolor="rgba(255,255,255,.92)", bordercolor="#c5d0dc")
    figure.update_layout(title={"text": f"{instrument} — breakout normalisé par l’ATR", "x": .5},
                         xaxis_title="Temps intrajournalier (schéma)", yaxis_title="Prix", hovermode="x unified",
                         height=460, margin={"l": 75, "r": 30, "t": 70, "b": 55}, paper_bgcolor="white")
    return figure


def gap_indicator_figure() -> go.Figure:
    figure = go.Figure()
    lines = [
        ([0, 2.2], [2480, 2480], "Clôture RTH J−1", "#7b8794", "solid"),
        ([2.3, 6.8], [2490, 2490], "Stop : milieu du gap", "#c45a4a", "dash"),
        ([4.0, 6.8], [2544, 2544], "Cible : entrée + 2 × gap", "#25855a", "dash"),
    ]
    for x, y, name, color, dash in lines:
        figure.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line={"color": color, "width": 3, "dash": dash},
                                    hovertemplate=f"{html.escape(name)}<br>%{{y:,.2f}}<extra></extra>"))
    figure.add_trace(go.Scatter(x=[2.8, 4.2], y=[2500, 2504], mode="markers", name="Open / entrée",
                                text=["Open 08:20", "Entrée 08:35"], marker={"size": 14, "color": ["#1769aa", "#102b4e"]},
                                hovertemplate="%{text}<br>%{y:,.2f}<extra></extra>"))
    figure.add_trace(go.Candlestick(x=[3.6], open=[2498], high=[2508], low=[2496], close=[2506], name="Bougie 08:20–08:35",
                                    increasing_line_color="#25855a", hovertemplate="Confirmation<br>O %{open}<br>H %{high}<br>L %{low}<br>C %{close}<extra></extra>"))
    figure.add_annotation(x=3.6, y=2507, text="Bougie de confirmation", showarrow=True, ax=85, ay=-45)
    figure.update_layout(title={"text": "Gap continuation — confirmation, invalidation et cible", "x": .5},
                         xaxis_title="Séquence intrajournalière", yaxis_title="Prix MGC", xaxis_rangeslider_visible=False,
                         height=460, margin={"l": 70, "r": 30, "t": 70, "b": 55}, paper_bgcolor="white")
    return figure


def donchian_indicator_figure() -> go.Figure:
    price = [100,101,100.4,102,101.5,103,102.2,103.4,104,103.5,104.2,105,106,106.5,107,108,107.2,106.4,105.5,104.8,103.8,103]
    upper = [None if i < 5 else max(price[max(0, i - 5):i]) for i in range(len(price))]
    lower = [None if i < 3 else min(price[max(0, i - 3):i]) for i in range(len(price))]
    x = list(range(len(price)))
    figure = go.Figure()
    for y, name, color, shape in (
        (price, "Clôture 1 h", "#102b4e", "linear"),
        (upper, "Canal haut d’entrée N", "#25855a", "hv"),
        (lower, "Canal bas de sortie M", "#c45a4a", "hv"),
    ):
        figure.add_trace(go.Scatter(x=x, y=y, mode="lines+markers" if name == "Clôture 1 h" else "lines", name=name,
                                    line={"color": color, "width": 2.5, "shape": shape}, hovertemplate=f"Barre %{{x}}<br>{html.escape(name)} %{{y:.2f}}<extra></extra>"))
    figure.add_trace(go.Scatter(x=[11, 20], y=[price[11], price[20]], mode="markers", name="Événements",
                                text=["Cassure → long", "Canal de sortie → flat"], marker={"size": 15, "color": ["#25855a", "#c45a4a"]},
                                hovertemplate="%{text}<br>Prix %{y:.2f}<extra></extra>"))
    figure.update_layout(title={"text": "Canal de Donchian — cassure d’entrée et canal opposé de sortie", "x": .5},
                         xaxis_title="Barres 1 h terminées", yaxis_title="Prix", hovermode="x unified",
                         height=460, margin={"l": 60, "r": 30, "t": 70, "b": 55}, paper_bgcolor="white")
    return figure


def ema_pullback_indicator_figure() -> go.Figure:
    price = [100,100.5,101,101.8,102.2,103,103.6,104.2,105,105.8,106.3,106.8,107.4,107,106.5,106.1,106.6,107.3,108.1,109,109.6,110.2,110.8,111.2,111.8,112.5,113,113.4]
    def ema(values: list[float], span: int) -> list[float]:
        alpha, result = 2 / (span + 1), [float(values[0])]
        for value in values[1:]:
            result.append(alpha * value + (1 - alpha) * result[-1])
        return result
    x = list(range(len(price)))
    figure = go.Figure(go.Scatter(x=x, y=price, mode="lines+markers", name="Prix", line={"color": "#102b4e", "width": 3},
                                  hovertemplate="Barre %{x}<br>Prix %{y:.2f}<extra></extra>"))
    for span, color in zip([3, 5, 8, 13, 18], ["#d9a21b", "#e07a3f", "#1769aa", "#7367a8", "#7b8794"]):
        figure.add_trace(go.Scatter(x=x, y=ema(price, span), mode="lines", name=f"EMA {span}", line={"color": color, "width": 2},
                                    hovertemplate=f"Barre %{{x}}<br>EMA {span} %{{y:.2f}}<extra></extra>"))
    figure.add_vrect(x0=13, x1=16.5, fillcolor="#dbe8f4", opacity=.6, line_width=0, annotation_text="Pullback")
    figure.add_annotation(x=17, y=price[17], text="Reprise → cellule longue", showarrow=True, ax=105, ay=35)
    figure.update_layout(title={"text": "Ruban d’EMA — tendance, pullback et reprise", "x": .5},
                         xaxis_title="Barres 15 min terminées", yaxis_title="Prix", hovermode="x unified",
                         height=470, margin={"l": 60, "r": 30, "t": 70, "b": 55}, paper_bgcolor="white")
    return figure


def equity_figure(number: int) -> go.Figure:
    daily = pd.read_csv(EQUITY_DATA[number])
    daily["session_date"] = pd.to_datetime(daily["session_date"])
    daily["daily_pnl_usd"] = pd.to_numeric(daily["daily_pnl_usd"], errors="raise").fillna(0.0)
    split = pd.Timestamp(OOS_START[number])
    status = OOS_LABEL[number]
    title = SHEET_LABELS[number]
    oos = daily[daily["session_date"].ge(split)].copy()
    daily["equity"] = daily["daily_pnl_usd"].cumsum()
    oos["equity"] = oos["daily_pnl_usd"].cumsum()
    figure = make_subplots(rows=1, cols=2, subplot_titles=("FS — Full Sample", status), horizontal_spacing=.08)
    figure.add_trace(go.Scatter(x=daily["session_date"], y=daily["equity"], mode="lines", name="Equity FS",
                                line={"color": "#1769aa", "width": 2, "shape": "hv"},
                                customdata=daily["daily_pnl_usd"],
                                hovertemplate="%{x|%d/%m/%Y}<br>PnL jour %{customdata:,.2f} USD<br>PnL cumulé %{y:,.2f} USD<extra></extra>"), row=1, col=1)
    figure.add_trace(go.Scatter(x=oos["session_date"], y=oos["equity"], mode="lines", name=status,
                                line={"color": "#25855a", "width": 2, "shape": "hv"},
                                customdata=oos["daily_pnl_usd"],
                                hovertemplate="%{x|%d/%m/%Y}<br>PnL jour %{customdata:,.2f} USD<br>PnL cumulé %{y:,.2f} USD<extra></extra>"), row=1, col=2)
    figure.add_vrect(x0=split, x1=daily["session_date"].max(), fillcolor="#dbe8f4", opacity=.35, line_width=0, row=1, col=1)
    figure.add_vline(x=split, line_dash="dash", line_color="#c45a4a", row=1, col=1)
    figure.add_annotation(x=split, y=1, xref="x", yref="paper", text=f"Début : {split:%d/%m/%Y}",
                          showarrow=False, xanchor="left", yanchor="top", font={"color": "#c45a4a"})
    figure.add_annotation(x=.55, y=.05, xref="paper", yref="paper", text=f"PnL de période : {oos['daily_pnl_usd'].sum():,.2f} USD".replace(",", " "),
                          showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,.9)", bordercolor="#c5d0dc")
    figure.update_yaxes(title_text="PnL cumulé (USD)", zeroline=True, zerolinecolor="#7b8794")
    figure.update_xaxes(title_text="Date", rangeslider_visible=True, row=1, col=1)
    figure.update_xaxes(title_text="Date", rangeslider_visible=True, row=1, col=2)
    figure.update_layout(title={"text": f"Equity curves nettes — {title}", "x": .5}, height=520,
                         hovermode="x", legend={"orientation": "h", "x": .5, "xanchor": "center", "y": -0.18},
                         margin={"l": 65, "r": 25, "t": 80, "b": 105}, paper_bgcolor="white")
    return figure


def interactive_sheet_figures(index: int) -> dict[str, str]:
    note = "Survoler pour lire les valeurs · glisser pour zoomer · double-cliquer pour réinitialiser"
    if index == 1:
        return {
            "orb_indicator": _interactive_plot_html("s01-orb-indicator", orb_indicator_figure(), note),
            "atr_indicator": _interactive_plot_html("s01-atr-indicator", atr_indicator_figure(), note),
            "equity_01": _interactive_plot_html("s01-equity", equity_figure(1), "Survoler pour lire le PnL quotidien et cumulé · utiliser les curseurs sous les panneaux pour choisir la période"),
        }
    if index == 2:
        return {
            "volatility_indicator": _interactive_plot_html("s02-volatility-indicator", volatility_indicator_figure(), note),
            "equity_02": _interactive_plot_html("s02-equity", equity_figure(2), "Survoler pour lire le PnL quotidien et cumulé · utiliser les curseurs sous les panneaux pour choisir la période"),
        }
    if index == 3:
        indicator = compression_indicator_figure()
    elif index == 4:
        indicator = trend_flip_indicator_figure()
    elif index == 5:
        indicator = normalized_breakout_figure("MGC", 2500, 20, 1.10, 10.0)
    elif index == 6:
        indicator = gap_indicator_figure()
    elif index == 7:
        indicator = donchian_indicator_figure()
    elif index == 8:
        indicator = ema_pullback_indicator_figure()
    elif index == 9:
        indicator = normalized_breakout_figure("MBT", 100000, 4000, .40, .10)
    else:
        return {}
    return {
        f"s{index:02d}_indicator": _interactive_plot_html(f"s{index:02d}-indicator", indicator, note),
        f"s{index:02d}_equity": _interactive_plot_html(f"s{index:02d}-equity", equity_figure(index), "Survoler pour lire le PnL quotidien et cumulé · utiliser les curseurs sous les panneaux pour choisir la période"),
    }


def markdown_to_body(
    text: str,
    source: Path,
    interactive_orb: str | None = None,
    interactive_volatility: str | None = None,
    interactive_figures: dict[str, str] | None = None,
) -> str:
    rendered = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
        output_format="html5",
    )
    rendered = MERMAID_BLOCK.sub(
        lambda match: f'<pre class="mermaid">{html.unescape(match.group(1))}</pre>',
        rendered,
    )
    if interactive_orb:
        rendered, replacements = ORB_CUBE_IMAGE.subn(
            lambda match: (
                interactive_orb
                + f'<p class="interactive-print"><img{match.group("attrs")}></p>'
            ),
            rendered,
        )
        if replacements != 1:
            raise RuntimeError(f"Expected one ORB cube image, replaced {replacements}")
    if interactive_volatility:
        rendered, replacements = VOLATILITY_CUBE_IMAGE.subn(
            lambda match: interactive_volatility + f'<p class="interactive-print"><img{match.group("attrs")}></p>',
            rendered,
        )
        if replacements != 1:
            raise RuntimeError(f"Expected one volatility cube image, replaced {replacements}")
    for key, interactive_figure in (interactive_figures or {}).items():
        pattern = INTERACTIVE_IMAGE_PATTERNS[key]
        rendered, replacements = pattern.subn(
            lambda match, figure=interactive_figure: figure + f'<p class="interactive-print"><img{match.group("attrs")}></p>',
            rendered,
        )
        if replacements != 1:
            raise RuntimeError(f"Expected one {key} image, replaced {replacements}")
    sheet_match = re.match(r"^(\d{2})_", source.name)
    if sheet_match and int(sheet_match.group(1)) >= 3:
        refinement = ROOT / "research" / "signal_sheet_parameter_refinement" / f"s{sheet_match.group(1)}" / "sheet_visuals.html"
        marker = "<!-- PARAMETER_REFINEMENT_VISUALS -->"
        if marker in rendered and refinement.is_file():
            rendered = rendered.replace(marker, refinement.read_text(encoding="utf-8"))
    return IMAGE_SRC.sub(
        lambda match: (
            match.group(1)
            + ((source.parent / html.unescape(match.group(2))).resolve().as_uri() if "://" not in match.group(2) else match.group(2))
            + match.group(3)
        ),
        rendered,
    )


def page(title: str, body: str, mermaid_src: str, plotly_src: str) -> str:
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{CSS}</style>
</head>
<body>
{body}
<script src="{html.escape(plotly_src, quote=True)}"></script>
<script src="{html.escape(mermaid_src, quote=True)}"></script>
<script>
mermaid.initialize({{startOnLoad:false, theme:"neutral", securityLevel:"loose", flowchart:{{useMaxWidth:true, htmlLabels:true}}}});
window.addEventListener("load", async () => {{
  try {{ await mermaid.run({{querySelector:".mermaid"}}); document.body.dataset.rendered = "true"; }}
  catch (error) {{ document.body.dataset.rendered = "error"; console.error(error); }}
}});
</script>
</body>
</html>
"""


def title_from_markdown(text: str) -> str:
    match = re.search(r"(?m)^#\s+(.+)$", text)
    if not match:
        raise ValueError("Every signal sheet must have an H1 title")
    return match.group(1).strip()


def build(output_dir: Path, mermaid_js: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets = sorted(path for path in SOURCE_DIR.glob("*.md") if SHEET_PATTERN.match(path.name))
    if len(sheets) != 9:
        raise RuntimeError(f"Expected 9 signal sheets, found {len(sheets)}")
    if not mermaid_js.is_file():
        raise FileNotFoundError(mermaid_js)

    local_mermaid = output_dir / "mermaid.min.js"
    local_mermaid.write_bytes(mermaid_js.read_bytes())
    local_plotly = output_dir / "plotly.min.js"
    local_plotly.write_text(get_plotlyjs(), encoding="utf-8")
    interactive_orb = orb_cube_html()
    interactive_volatility = volatility_breakout_cube_html()
    outputs: list[Path] = []
    rendered_sheets: list[tuple[str, str, str]] = []

    for index, source in enumerate(sheets, 1):
        text = source.read_text(encoding="utf-8")
        title = title_from_markdown(text)
        body = markdown_to_body(
            text,
            source,
            interactive_orb if index == 1 else None,
            interactive_volatility if index == 2 else None,
            interactive_sheet_figures(index),
        )
        anchor = source.stem
        rendered_sheets.append((anchor, title, body))
        edition = (
            "22 OCTOBRE 2026"
            if index == 1
            else "30 OCTOBRE 2026"
            if index == 2
            else "1 NOVEMBRE 2026"
        )
        decorated = (
            f'<div class="document-meta">FICHE SIGNAL {index:02d}/09 · DOCUMENT CLIENT · ÉDITION {edition}</div>'
            + body
        )
        destination = output_dir / f"{source.stem}.html"
        destination.write_text(page(title, decorated, "mermaid.min.js", "plotly.min.js"), encoding="utf-8")
        outputs.append(destination)

    toc = "".join(
        f'<li><a href="#{html.escape(anchor)}">{html.escape(title)}</a></li>'
        for anchor, title, _ in rendered_sheets
    )
    book_source = SOURCE_DIR / "BOOK.md"
    book_text = book_source.read_text(encoding="utf-8")
    book_title = title_from_markdown(book_text)
    book_body = markdown_to_body(book_text, book_source)
    toc += f'<li><a href="#book">{html.escape(book_title)}</a></li>'
    sections = "".join(
        f'<section class="signal-sheet{" first" if index == 0 else ""}" id="{html.escape(anchor)}">'
        f'<div class="document-meta">FICHE SIGNAL {index + 1:02d}/09 · DOCUMENT CLIENT</div>{body}</section>'
        for index, (anchor, _, body) in enumerate(rendered_sheets)
    )
    sections += f'<section class="book-section" id="book"><div class="document-meta">BOOK · CORRÉLATIONS ET RISQUES</div>{book_body}</section>'
    consolidated_body = f"""
<section class="cover">
  <div class="eyebrow">Portefeuille intraday · Référence verrouillée</div>
  <h1>Les 9 sleeves<br>Fiches signal</h1>
  <div class="rule"></div>
  <p>Guide pédagogique complet : intuition, données, paramètres, construction, exécution, risques et procédure de réimplémentation.</p>
  <div class="date">Édition du 1er novembre 2026 · Equal-contract [1,1,1,1,1,1,1,1,1]</div>
</section>
<section class="toc">
  <h1>Sommaire</h1>
  <ol>{toc}</ol>
  <div class="client-note"><strong>Note importante.</strong> Les performances historiques et simulations présentées ne constituent ni une garantie de résultat futur ni un conseil d'investissement. Toute réimplémentation doit préserver les conventions causales, les coûts et les règles d'exécution documentées.</div>
</section>
{sections}
"""
    consolidated = output_dir / "00_portefeuille_9_sleeves_complet.html"
    consolidated.write_text(
        page("Portefeuille intraday — 9 fiches signal", consolidated_body, "mermaid.min.js", "plotly.min.js"),
        encoding="utf-8",
    )
    outputs.append(consolidated)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mermaid-js", type=Path, required=True)
    args = parser.parse_args()
    for output in build(args.output_dir.resolve(), args.mermaid_js.resolve()):
        print(output)


if __name__ == "__main__":
    main()
