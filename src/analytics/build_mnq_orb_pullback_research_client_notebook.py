"""Build a full MNQ ORB + pullback research notebook from source data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NOTEBOOKS_ROOT = REPO_ROOT / "notebooks"
DEFAULT_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "mnq_orb_pullback_research_client.ipynb"
DEFAULT_EXECUTED_NOTEBOOK_PATH = NOTEBOOKS_ROOT / "mnq_orb_pullback_research_client.executed.ipynb"


def _title_cell() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """# MNQ ORB + Pullback - Notebook de recherche complet

Objectif: reconstruire les deux signaux depuis zero, expliquer les parametres, visualiser les zones de robustesse, puis comparer:

- ORB standalone,
- Volume climax pullback standalone,
- portefeuille equal weight ORB + pullback,
- benchmark MNQ buy & hold.

Le notebook ne charge pas les resultats d'exports pour calculer les performances. Il relit les donnees MNQ traitees, reconstruit les features, regenere les signaux, relance les backtests et reconstruit les heatmaps dans le notebook.
"""
    )


def _imports_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """import math
import sys
from dataclasses import asdict, replace
from itertools import product
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

from src.analytics.orb_multi_asset_campaign import resolve_processed_dataset
from src.analytics.orb_research.campaign import _evaluate_experiment, _split_sessions
from src.analytics.orb_research.features import (
    attach_daily_reference,
    build_candidate_universe,
    build_daily_reference,
    prepare_minute_dataset,
)
from src.analytics.orb_research.types import (
    BaselineEnsembleConfig,
    BaselineEntryConfig,
    CampaignContext,
    CompressionConfig,
    DynamicThresholdConfig,
    ExitConfig,
    ExperimentConfig,
)
from src.analytics.volume_climax_pullback_common import (
    filter_trades_by_sessions,
    load_symbol_data,
    resample_rth_1h,
    split_sessions,
)
from src.config.settings import get_instrument_spec
from src.engine.execution_model import ExecutionModel
from src.engine.vwap_backtester import build_execution_model_for_profile
from src.risk.position_sizing import FixedContractPositionSizing, RiskPercentPositionSizing
from src.strategy.volume_climax_pullback_v2 import (
    VolumeClimaxPullbackV2Variant,
    build_volume_climax_pullback_v2_signal_frame,
    prepare_volume_climax_pullback_v2_features,
)
from src.engine.volume_climax_pullback_v2_backtester import run_volume_climax_pullback_v2_backtest

pd.set_option("display.max_columns", 220)
pd.set_option("display.width", 240)
"""
    )


def _helpers_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """def fmt_money(value):
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


def normalize_sessions(sessions):
    return pd.to_datetime(pd.Index(sessions)).normalize().tolist()


def session_set(sessions):
    return set(pd.to_datetime(pd.Index(sessions)).date)


def subset_trades(trades, sessions):
    if trades.empty:
        return trades.copy()
    allowed = session_set(sessions)
    out = trades.copy()
    out["_session_key"] = pd.to_datetime(out["session_date"], errors="coerce").dt.date
    return out.loc[out["_session_key"].isin(allowed)].drop(columns=["_session_key"]).copy().reset_index(drop=True)


def daily_results_from_trades(trades, sessions, initial_capital):
    calendar = pd.DataFrame({"session_date": normalize_sessions(sessions)})
    calendar = calendar.drop_duplicates().sort_values("session_date").reset_index(drop=True)
    if calendar.empty:
        return pd.DataFrame(columns=["session_date", "daily_pnl_usd", "daily_trade_count", "equity", "peak_equity", "drawdown_usd", "drawdown_pct", "daily_return"])

    if trades.empty:
        daily = calendar.copy()
        daily["daily_pnl_usd"] = 0.0
        daily["daily_trade_count"] = 0
    else:
        view = trades.copy()
        view["session_date"] = pd.to_datetime(view["session_date"], errors="coerce").dt.normalize()
        grouped = (
            view.groupby("session_date", as_index=False)
            .agg(daily_pnl_usd=("net_pnl_usd", "sum"), daily_trade_count=("trade_id", "count"))
        )
        daily = calendar.merge(grouped, on="session_date", how="left")
        daily["daily_pnl_usd"] = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0)
        daily["daily_trade_count"] = pd.to_numeric(daily["daily_trade_count"], errors="coerce").fillna(0).astype(int)

    daily = daily.sort_values("session_date").reset_index(drop=True)
    prev_equity = float(initial_capital)
    returns = []
    equities = []
    for pnl in pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0):
        returns.append(float(pnl) / prev_equity if prev_equity else 0.0)
        prev_equity += float(pnl)
        equities.append(prev_equity)
    daily["daily_return"] = returns
    daily["equity"] = equities
    daily["peak_equity"] = daily["equity"].cummax()
    daily["drawdown_usd"] = daily["equity"] - daily["peak_equity"]
    daily["drawdown_pct"] = np.where(daily["peak_equity"] > 0, (daily["equity"] / daily["peak_equity"] - 1.0) * 100.0, 0.0)
    return daily


def curve_from_returns(session_dates, daily_return, initial_capital, label="curve"):
    out = pd.DataFrame({"session_date": pd.to_datetime(pd.Series(session_dates).reset_index(drop=True), errors="coerce").dt.normalize()})
    out["daily_return"] = pd.to_numeric(pd.Series(daily_return).reset_index(drop=True), errors="coerce").fillna(0.0)
    if (out["daily_return"] <= -1.0).any():
        worst = float(out["daily_return"].min())
        raise ValueError(f"{label}: daily return <= -100% after scaling ({worst:.2%}).")
    out["equity"] = float(initial_capital) * (1.0 + out["daily_return"]).cumprod()
    out["daily_pnl_usd"] = out["equity"].diff().fillna(out["equity"].iloc[0] - float(initial_capital))
    out["peak_equity"] = out["equity"].cummax()
    out["drawdown_usd"] = out["equity"] - out["peak_equity"]
    out["drawdown_pct"] = np.where(out["peak_equity"] > 0, (out["equity"] / out["peak_equity"] - 1.0) * 100.0, 0.0)
    out["daily_trade_count"] = 0
    return out


def scale_daily_returns(daily_return, leverage, label):
    scaled = pd.to_numeric(pd.Series(daily_return), errors="coerce").fillna(0.0) * float(leverage)
    if (scaled <= -1.0).any():
        worst = float(scaled.min())
        raise ValueError(f"{label}: leverage creates a daily return <= -100% ({worst:.2%}).")
    return scaled


def curve_metrics(daily, trades=None, initial_capital=50_000.0):
    if daily.empty:
        return {
            "net_pnl_usd": 0.0,
            "return_pct": 0.0,
            "cagr_pct": np.nan,
            "sharpe": 0.0,
            "sortino": 0.0,
            "annualized_vol_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
            "worst_day_usd": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
        }
    ordered = daily.sort_values("session_date").reset_index(drop=True)
    rets = pd.to_numeric(ordered["daily_return"], errors="coerce").fillna(0.0)
    equity = pd.to_numeric(ordered["equity"], errors="coerce").fillna(float(initial_capital))
    pnl = pd.to_numeric(ordered["daily_pnl_usd"], errors="coerce").fillna(0.0)
    start = pd.Timestamp(ordered["session_date"].iloc[0])
    end = pd.Timestamp(ordered["session_date"].iloc[-1])
    years = max(((end - start).days + 1) / 365.25, 1 / 365.25)
    final_equity = float(equity.iloc[-1])
    sharpe = float(rets.mean() / rets.std(ddof=0) * math.sqrt(252.0)) if len(rets) > 1 and rets.std(ddof=0) > 0 else 0.0
    downside = rets[rets < 0]
    downside_std = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    sortino = float(rets.mean() / downside_std * math.sqrt(252.0)) if downside_std > 0 else 0.0
    cagr = float(((final_equity / float(initial_capital)) ** (1.0 / years) - 1.0) * 100.0) if final_equity > 0 else np.nan
    if trades is not None and not trades.empty:
        trade_pnl = pd.to_numeric(trades["net_pnl_usd"], errors="coerce").fillna(0.0)
        wins = trade_pnl[trade_pnl > 0]
        losses = trade_pnl[trade_pnl < 0]
        profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else float("inf")
        n_trades = int(len(trade_pnl))
        win_rate = float((trade_pnl > 0).mean())
    else:
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else float("inf")
        n_trades = int(pd.to_numeric(ordered.get("daily_trade_count", 0), errors="coerce").fillna(0).sum())
        win_rate = 0.0
    return {
        "net_pnl_usd": float(final_equity - float(initial_capital)),
        "return_pct": float((final_equity / float(initial_capital) - 1.0) * 100.0),
        "cagr_pct": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "annualized_vol_pct": float(rets.std(ddof=0) * math.sqrt(252.0) * 100.0) if len(rets) > 1 else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown_usd": float(abs(pd.to_numeric(ordered["drawdown_usd"], errors="coerce").min())),
        "max_drawdown_pct": float(abs(pd.to_numeric(ordered["drawdown_pct"], errors="coerce").min())),
        "worst_day_usd": float(pnl.min()),
        "n_trades": n_trades,
        "win_rate": win_rate,
    }


def summarize_scopes(label, daily_full, trades_full, is_sessions, oos_sessions, initial_capital, daily_oos_rebased=None, trades_oos=None):
    rows = []
    for scope_name, sessions in [("full", daily_full["session_date"].tolist()), ("is", is_sessions), ("oos", oos_sessions)]:
        if scope_name == "oos" and daily_oos_rebased is not None:
            scoped_daily = daily_oos_rebased.copy()
            scoped_trades = trades_oos if trades_oos is not None else subset_trades(trades_full, sessions)
        else:
            allowed = set(pd.to_datetime(pd.Index(sessions)).normalize())
            scoped_daily = daily_full.loc[daily_full["session_date"].isin(allowed)].copy()
            scoped_trades = subset_trades(trades_full, sessions)
        rows.append({"strategy": label, "scope": scope_name, **curve_metrics(scoped_daily, scoped_trades, initial_capital)})
    return pd.DataFrame(rows)


def scale_trade_log_to_fixed_contracts(trades, quantity, initial_capital, point_value_usd):
    out = trades.copy()
    quantity = int(quantity)
    if out.empty or quantity == 1:
        if not out.empty:
            out["quantity"] = quantity
        return out
    scale = float(quantity)
    for column in ["risk_budget_usd", "actual_risk_usd", "trade_risk_usd", "notional_usd", "pnl_usd", "fees", "net_pnl_usd"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce") * scale
    out["quantity"] = quantity
    if "entry_price" in out.columns:
        entry_price = pd.to_numeric(out["entry_price"], errors="coerce")
        out["notional_usd"] = entry_price * float(point_value_usd) * float(quantity)
        out["leverage_used"] = out["notional_usd"] / float(max(initial_capital, 1.0))
    return out


def parameter_table(rows):
    frame = pd.DataFrame(rows)
    display(frame[["section", "parameter", "value", "meaning"]])
    return frame


def plot_heatmap(frame, x, y, z, title, color_continuous_scale="RdYlGn", text_auto=".2f"):
    pivot = frame.pivot_table(index=y, columns=x, values=z, aggfunc="mean").sort_index()
    fig = px.imshow(
        pivot,
        aspect="auto",
        text_auto=text_auto,
        color_continuous_scale=color_continuous_scale,
        title=title,
    )
    fig.update_layout(template=PLOT_TEMPLATE, width=850, height=520)
    fig.update_xaxes(title=x)
    fig.update_yaxes(title=y)
    fig.show()
"""
    )


def _parameter_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """SYMBOL = "MNQ"
INITIAL_CAPITAL_USD = 50_000.0
IS_FRACTION = 0.70
PLOT_TEMPLATE = "plotly_dark"

# Data
MNQ_1M_DATASET_PATH = None  # None = dernier parquet MNQ 1m dans data/processed/parquet

# ORB signal - meilleure variante issue de `export/orb_research_campaign/campaign_report.md`
ORB_RESEARCH_CONFIG_NAME = "full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate"
ORB_OR_MINUTES = 15
ORB_OPENING_TIME = "09:30:00"
ORB_DIRECTION = "long"
ORB_ONE_TRADE_PER_DAY = True
ORB_ENTRY_BUFFER_TICKS = 2
ORB_STOP_BUFFER_TICKS = 2
ORB_TARGET_MULTIPLE = 2.0
ORB_VWAP_CONFIRMATION = True
ORB_VWAP_COLUMN = "continuous_session_vwap"
ORB_TIME_EXIT = "16:00:00"
ORB_RISK_PER_TRADE_PCT = 0.50
ORB_MAX_LEVERAGE = None  # None = pas de cap notionnel supplementaire

# ORB overlays / ensemble
ORB_ATR_WINDOW = 14
ORB_Q_LOW_PCTS = (20, 25, 30)
ORB_Q_HIGH_PCTS = (90, 95)
ORB_ENSEMBLE_VOTE_THRESHOLD = 0.50
ORB_COMPRESSION_MODE = "weak_close"
ORB_COMPRESSION_USAGE = "soft_vote_bonus"
ORB_COMPRESSION_SOFT_BONUS_VOTES = 1.0
ORB_EXIT_MODE = "baseline"
ORB_DYNAMIC_MODE = "noise_area_gate"
ORB_NOISE_LOOKBACK = 30
ORB_NOISE_VM = 1.0
ORB_NOISE_K = 0.0
ORB_DYNAMIC_ATR_K = 0.0
ORB_DYNAMIC_CONFIRM_BARS = 1
ORB_DYNAMIC_SCHEDULE = "continuous_on_bar_close"
ORB_DYNAMIC_THRESHOLD_STYLE = "max_or_high_noise"

# ORB local robustness grids. Ces grilles recalculent les performances depuis les donnees.
ORB_NOISE_LOOKBACK_GRID = (10, 14, 20, 30)
ORB_NOISE_VM_GRID = (0.75, 1.0, 1.25, 1.5)
ORB_ATR_WINDOW_GRID = (10, 14, 20, 30)
ORB_VOTE_THRESHOLD_GRID = (0.50, 0.67, 0.75)
ORB_GRID_BOOTSTRAP_PATHS = 300
ORB_FINAL_BOOTSTRAP_PATHS = 1000

# Pullback alpha signal
PULLBACK_VOLUME_QUANTILE = 0.95
PULLBACK_VOLUME_LOOKBACK = 50
PULLBACK_MIN_BODY_FRACTION = 0.50
PULLBACK_MIN_RANGE_ATR = 1.20
PULLBACK_ENTRY_MODE = "next_open"
PULLBACK_EXIT_MODE = "atr_fraction"
PULLBACK_ATR_TARGET_MULTIPLE = 1.00
PULLBACK_TIME_STOP_BARS = 2
PULLBACK_TRAILING_ATR_MULTIPLE = 0.50

# Pullback sizing
PULLBACK_RISK_PCT = 0.0025
PULLBACK_MAX_CONTRACTS = 6
PULLBACK_SKIP_TRADE_IF_TOO_SMALL = True
PULLBACK_COMPOUND_REALIZED_PNL = False
PULLBACK_RISK_PCT_GRID = (0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040)
PULLBACK_MAX_CONTRACTS_GRID = (2, 3, 4, 5, 6)

# Pullback alpha robustness grid
PULLBACK_ALPHA_VOLUME_QUANTILES = (0.95, 0.975)
PULLBACK_ALPHA_BODY_FRACTIONS = (0.50, 0.60)
PULLBACK_ALPHA_RANGE_ATRS = (1.20, 1.50)

# Portfolio
ORB_WEIGHT = 0.50
PULLBACK_WEIGHT = 0.50
ORB_LEVERAGE = 1.00
PULLBACK_LEVERAGE = 1.00
BLEND_LEVERAGE = 1.00
BENCHMARK_LEVERAGE = 1.00

dataset_path = Path(MNQ_1M_DATASET_PATH) if MNQ_1M_DATASET_PATH is not None else resolve_processed_dataset(SYMBOL, timeframe="1m")
instrument_spec = get_instrument_spec(SYMBOL)

parameter_rows = [
    {"section": "global", "parameter": "SYMBOL", "value": SYMBOL, "meaning": "Contrat et univers de recherche."},
    {"section": "global", "parameter": "INITIAL_CAPITAL_USD", "value": INITIAL_CAPITAL_USD, "meaning": "Capital de depart pour toutes les courbes comparables."},
    {"section": "global", "parameter": "IS_FRACTION", "value": IS_FRACTION, "meaning": "Part historique utilisee comme in-sample; le reste est out-of-sample."},
    {"section": "data", "parameter": "MNQ_1M_DATASET_PATH", "value": str(dataset_path), "meaning": "Source de donnees MNQ 1 minute; les deux signaux repartent de ce fichier."},
    {"section": "orb", "parameter": "ORB_RESEARCH_CONFIG_NAME", "value": ORB_RESEARCH_CONFIG_NAME, "meaning": "Config recommandee par la campagne ORB research du repo; les performances sont recalculees ici."},
    {"section": "orb", "parameter": "ORB_OR_MINUTES", "value": ORB_OR_MINUTES, "meaning": "Duree de l'opening range. OR15 signifie high/low des 15 premieres minutes RTH."},
    {"section": "orb", "parameter": "ORB_DIRECTION", "value": ORB_DIRECTION, "meaning": "Direction tradee. La meilleure config de recherche retenue ici est long-only."},
    {"section": "orb", "parameter": "ORB_ENTRY_BUFFER_TICKS", "value": ORB_ENTRY_BUFFER_TICKS, "meaning": "Distance au-dessus/dessous de l'opening range exigee pour valider le breakout."},
    {"section": "orb", "parameter": "ORB_STOP_BUFFER_TICKS", "value": ORB_STOP_BUFFER_TICKS, "meaning": "Buffer ajoute au stop place de l'autre cote de l'opening range."},
    {"section": "orb", "parameter": "ORB_TARGET_MULTIPLE", "value": ORB_TARGET_MULTIPLE, "meaning": "Target en multiple du risque initial."},
    {"section": "orb", "parameter": "ORB_VWAP_CONFIRMATION", "value": ORB_VWAP_CONFIRMATION, "meaning": "Le breakout long doit etre au-dessus du VWAP continu."},
    {"section": "orb", "parameter": "ORB_ATR_WINDOW", "value": ORB_ATR_WINDOW, "meaning": "Fenetre ATR utilisee pour scorer les regimes de volatilite."},
    {"section": "orb", "parameter": "ORB_Q_LOW_PCTS / ORB_Q_HIGH_PCTS", "value": f"{ORB_Q_LOW_PCTS} / {ORB_Q_HIGH_PCTS}", "meaning": "Bandes de quantiles ATR de l'ensemble; chaque bande vote pour ou contre la session."},
    {"section": "orb", "parameter": "ORB_ENSEMBLE_VOTE_THRESHOLD", "value": ORB_ENSEMBLE_VOTE_THRESHOLD, "meaning": "Score minimum de votes ATR pour garder le signal. 0.50 = majorite simple."},
    {"section": "orb", "parameter": "ORB_COMPRESSION_MODE", "value": ORB_COMPRESSION_MODE, "meaning": "Overlay de contexte daily: weak_close = la veille a cloture dans le bas de son range RTH."},
    {"section": "orb", "parameter": "ORB_COMPRESSION_USAGE", "value": ORB_COMPRESSION_USAGE, "meaning": "soft_vote_bonus ajoute un vote bonus au filtre ATR au lieu de filtrer dur."},
    {"section": "orb", "parameter": "ORB_DYNAMIC_MODE", "value": ORB_DYNAMIC_MODE, "meaning": "Noise-area gate: le breakout doit depasser le plus haut d'OR et une zone de bruit intraday."},
    {"section": "orb", "parameter": "ORB_NOISE_LOOKBACK / ORB_NOISE_VM", "value": f"{ORB_NOISE_LOOKBACK} / {ORB_NOISE_VM}", "meaning": "Lookback sessions pour le bruit minute par minute, puis multiplicateur de cette zone de bruit."},
    {"section": "orb", "parameter": "ORB_RISK_PER_TRADE_PCT", "value": ORB_RISK_PER_TRADE_PCT, "meaning": "Risque ORB par trade en pourcentage du capital. 0.50 = 0.5%; les contrats sont calcules depuis la distance au stop."},
    {"section": "orb", "parameter": "ORB_MAX_LEVERAGE", "value": ORB_MAX_LEVERAGE, "meaning": "Cap optionnel de levier notionnel pour l'ORB. None = pas de cap supplementaire."},
    {"section": "pullback", "parameter": "PULLBACK_VOLUME_QUANTILE", "value": PULLBACK_VOLUME_QUANTILE, "meaning": "Seuil de volume historique. 0.95 = volume de la bougie setup > 95e percentile roulant."},
    {"section": "pullback", "parameter": "PULLBACK_VOLUME_LOOKBACK", "value": PULLBACK_VOLUME_LOOKBACK, "meaning": "Nombre de barres 1h utilisees pour calibrer le seuil de volume."},
    {"section": "pullback", "parameter": "PULLBACK_MIN_BODY_FRACTION", "value": PULLBACK_MIN_BODY_FRACTION, "meaning": "Qualite de bougie: corps / range minimum sur la bougie climax."},
    {"section": "pullback", "parameter": "PULLBACK_MIN_RANGE_ATR", "value": PULLBACK_MIN_RANGE_ATR, "meaning": "Range minimum de la bougie climax en multiple de l'ATR 20 barres."},
    {"section": "pullback", "parameter": "PULLBACK_EXIT_MODE", "value": PULLBACK_EXIT_MODE, "meaning": "Mode de sortie. atr_fraction cible un deplacement de X ATR depuis l'entree."},
    {"section": "pullback", "parameter": "PULLBACK_RISK_PCT", "value": PULLBACK_RISK_PCT, "meaning": "Fraction du capital risquee par trade pullback. 0.0025 = 0.25%."},
    {"section": "pullback", "parameter": "PULLBACK_MAX_CONTRACTS", "value": PULLBACK_MAX_CONTRACTS, "meaning": "Cap de contrats pour eviter que le sizing grossisse trop quand le stop est court."},
    {"section": "pullback", "parameter": "PULLBACK_COMPOUND_REALIZED_PNL", "value": PULLBACK_COMPOUND_REALIZED_PNL, "meaning": "False = sizing calcule sur le capital initial constant, sans ajouter le PnL realise."},
    {"section": "portfolio", "parameter": "ORB_WEIGHT / PULLBACK_WEIGHT", "value": f"{ORB_WEIGHT} / {PULLBACK_WEIGHT}", "meaning": "Poids journaliers avant normalisation dans le portefeuille blend."},
    {"section": "portfolio", "parameter": "LEVERAGE", "value": f"ORB {ORB_LEVERAGE}x / Pullback {PULLBACK_LEVERAGE}x / Blend {BLEND_LEVERAGE}x", "meaning": "Multiplicateurs post-backtest pour simuler plus ou moins de risque."},
]

display(Markdown("## 0. Parametrage client"))
params = parameter_table(parameter_rows)
"""
    )


def _data_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 1. Data et sanity checks"))

raw_1m = load_symbol_data(SYMBOL, input_paths={SYMBOL: dataset_path})
raw_1m["timestamp"] = pd.to_datetime(raw_1m["timestamp"], errors="coerce")

data_sanity = pd.DataFrame(
    [
        {"item": "dataset_path", "value": str(dataset_path)},
        {"item": "rows_1m", "value": f"{len(raw_1m):,}"},
        {"item": "first_timestamp", "value": str(raw_1m["timestamp"].min())},
        {"item": "last_timestamp", "value": str(raw_1m["timestamp"].max())},
        {"item": "duplicate_timestamps", "value": int(raw_1m["timestamp"].duplicated().sum())},
        {"item": "instrument_tick_size", "value": instrument_spec["tick_size"]},
        {"item": "instrument_tick_value_usd", "value": instrument_spec["tick_value_usd"]},
        {"item": "instrument_point_value_usd", "value": instrument_spec["point_value_usd"]},
        {"item": "commission_per_side_usd", "value": instrument_spec["commission_per_side_usd"]},
        {"item": "slippage_ticks", "value": instrument_spec["slippage_ticks"]},
    ]
)
display(data_sanity)
"""
    )


def _orb_markdown_cell() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 2. ORB avance: reconstruction du signal

Logique de recherche:

Cette section reprend la meilleure configuration ORB de la campagne de recherche du repo: `full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate`.

1. On construit les features MNQ 1 minute: session, VWAP, opening range OR15, ATR et references daily sans look-ahead.
2. Le signal brut apparait quand le close casse le haut de l'opening range apres la fenetre OR. Cette meilleure config est long-only.
3. Le filtre VWAP exige que le breakout soit au-dessus du VWAP continu.
4. L'overlay `weak_close` regarde la veille: cloture faible dans le range RTH. Ici il sert de bonus de vote, pas de filtre dur.
5. Le `noise_area_gate` exige que le breakout depasse aussi une zone de bruit intraday estimee sur les sessions precedentes.
6. L'ATR ensemble retient les sessions dont la volatilite tombe dans des bandes de quantiles stables.
7. Le backtest final est relance avec `ORB_RISK_PER_TRADE_PCT`: le nombre de contrats est recalcule trade par trade depuis le risque stop.
"""
    )


def _orb_run_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """orb_entry = BaselineEntryConfig(
    or_minutes=ORB_OR_MINUTES,
    opening_time=ORB_OPENING_TIME,
    direction=ORB_DIRECTION,
    one_trade_per_day=ORB_ONE_TRADE_PER_DAY,
    entry_buffer_ticks=ORB_ENTRY_BUFFER_TICKS,
    stop_buffer_ticks=ORB_STOP_BUFFER_TICKS,
    target_multiple=ORB_TARGET_MULTIPLE,
    vwap_confirmation=ORB_VWAP_CONFIRMATION,
    vwap_column=ORB_VWAP_COLUMN,
    time_exit=ORB_TIME_EXIT,
    account_size_usd=INITIAL_CAPITAL_USD,
    risk_per_trade_pct=ORB_RISK_PER_TRADE_PCT,
    tick_size=float(instrument_spec["tick_size"]),
    entry_on_next_open=True,
)
orb_ensemble = BaselineEnsembleConfig(
    atr_window=ORB_ATR_WINDOW,
    q_lows_pct=tuple(ORB_Q_LOW_PCTS),
    q_highs_pct=tuple(ORB_Q_HIGH_PCTS),
    vote_threshold=ORB_ENSEMBLE_VOTE_THRESHOLD,
)
orb_compression = CompressionConfig(
    mode=ORB_COMPRESSION_MODE,
    usage=ORB_COMPRESSION_USAGE,
    soft_bonus_votes=ORB_COMPRESSION_SOFT_BONUS_VOTES,
)
orb_exit = ExitConfig(mode=ORB_EXIT_MODE)
orb_dynamic = DynamicThresholdConfig(
    mode=ORB_DYNAMIC_MODE,
    noise_lookback=ORB_NOISE_LOOKBACK,
    noise_vm=ORB_NOISE_VM,
    threshold_style=ORB_DYNAMIC_THRESHOLD_STYLE,
    noise_k=ORB_NOISE_K,
    atr_k=ORB_DYNAMIC_ATR_K,
    confirm_bars=ORB_DYNAMIC_CONFIRM_BARS,
    schedule=ORB_DYNAMIC_SCHEDULE,
)
orb_experiment = ExperimentConfig(
    name=ORB_RESEARCH_CONFIG_NAME,
    stage="full_reopt",
    family="full_reopt",
    baseline_entry=orb_entry,
    baseline_ensemble=orb_ensemble,
    compression=orb_compression,
    exit=orb_exit,
    dynamic_threshold=orb_dynamic,
)

orb_atr_windows = sorted({int(ORB_ATR_WINDOW), *[int(x) for x in ORB_ATR_WINDOW_GRID]})
orb_minute_df = prepare_minute_dataset(dataset_path=dataset_path, baseline_entry=orb_entry, atr_windows=orb_atr_windows)
orb_daily_reference = build_daily_reference(orb_minute_df)
orb_minute_df = attach_daily_reference(orb_minute_df, orb_daily_reference)
orb_candidate_base = build_candidate_universe(orb_minute_df, baseline_entry=orb_entry)

orb_all_session_dates = sorted(pd.to_datetime(orb_minute_df["session_date"]).dt.date.unique())
orb_is_raw, orb_oos_raw = _split_sessions(orb_all_session_dates, IS_FRACTION)
orb_context = CampaignContext(
    all_sessions=orb_all_session_dates,
    is_sessions=orb_is_raw,
    oos_sessions=orb_oos_raw,
    minute_df=orb_minute_df,
    candidate_base_df=orb_candidate_base,
    daily_patterns=orb_daily_reference,
)

orb_row, orb_detail = _evaluate_experiment(
    experiment=orb_experiment,
    context=orb_context,
    bootstrap_paths=ORB_FINAL_BOOTSTRAP_PATHS,
    random_seed=42,
    keep_details=True,
    max_leverage=ORB_MAX_LEVERAGE,
)
if orb_detail is None:
    raise RuntimeError(f"ORB experiment failed: {orb_row}")

orb_trades = orb_detail["trades"].copy()
orb_signal_df = orb_detail["signal_df"].copy()
orb_selected_final = orb_detail["selected_final"].copy()
orb_all_sessions = normalize_sessions(orb_context.all_sessions)
orb_is_sessions = normalize_sessions(orb_context.is_sessions)
orb_oos_sessions = normalize_sessions(orb_context.oos_sessions)
orb_daily_full = daily_results_from_trades(orb_trades, orb_all_sessions, INITIAL_CAPITAL_USD)
orb_daily_oos = daily_results_from_trades(subset_trades(orb_trades, orb_oos_sessions), orb_oos_sessions, INITIAL_CAPITAL_USD)
orb_summary = summarize_scopes("ORB", orb_daily_full, orb_trades, orb_is_sessions, orb_oos_sessions, INITIAL_CAPITAL_USD, daily_oos_rebased=orb_daily_oos, trades_oos=subset_trades(orb_trades, orb_oos_sessions))

display(Markdown("### ORB run summary"))
display(
    pd.DataFrame(
        [
            {"item": "sessions_all", "value": len(orb_all_sessions)},
            {"item": "sessions_is", "value": len(orb_is_sessions)},
            {"item": "sessions_oos", "value": len(orb_oos_sessions)},
            {"item": "candidate_rows_raw", "value": int(orb_row.get("candidate_rows_raw", len(orb_candidate_base)))},
            {"item": "candidate_rows_after_overlays", "value": int(orb_row.get("candidate_rows_after_overlays", 0))},
            {"item": "candidate_days_pre_ensemble", "value": int(orb_row.get("candidate_days_pre_ensemble", 0))},
            {"item": "selected_days_after_atr_ensemble", "value": int(orb_row.get("selected_days", len(orb_selected_final)))},
            {"item": "trades_after_all_filters", "value": int(len(orb_trades))},
            {"item": "oos_trades", "value": int(len(subset_trades(orb_trades, orb_oos_sessions)))},
        ]
    )
)
display(Markdown("### ORB config effective"))
orb_config_rows = []
for block, payload in [
    ("entry", asdict(orb_entry)),
    ("ensemble", asdict(orb_ensemble)),
    ("compression", asdict(orb_compression)),
    ("dynamic_threshold", asdict(orb_dynamic)),
    ("exit", asdict(orb_exit)),
]:
    for parameter, value in payload.items():
        orb_config_rows.append({"block": block, "parameter": parameter, "value": value})
display(pd.DataFrame(orb_config_rows))
display(Markdown("### ORB research metrics recalculees"))
display(
    pd.DataFrame(
        [
            {
                "scope": scope,
                "net_pnl_usd": orb_row.get(f"{prefix}_net_pnl"),
                "sharpe": orb_row.get(f"{prefix}_sharpe_ratio"),
                "profit_factor": orb_row.get(f"{prefix}_profit_factor"),
                "max_drawdown_usd": abs(float(orb_row.get(f"{prefix}_max_drawdown", 0.0))),
                "prop_score": orb_row.get(f"{prefix}_prop_score"),
                "trades": orb_row.get(f"{prefix}_nb_trades"),
                "pct_days_traded": orb_row.get(f"{prefix}_pct_days_traded"),
            }
            for scope, prefix in [("full", "overall"), ("is", "is"), ("oos", "oos")]
        ]
    ).round(3)
)
display(orb_summary.round(3))
"""
    )


def _orb_heatmap_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 3. ORB heatmaps et lecture de robustesse"))

def orb_experiment_with(**changes):
    exp = orb_experiment
    if "baseline_ensemble" in changes:
        exp = replace(exp, baseline_ensemble=changes["baseline_ensemble"])
    if "compression" in changes:
        exp = replace(exp, compression=changes["compression"])
    if "dynamic_threshold" in changes:
        exp = replace(exp, dynamic_threshold=changes["dynamic_threshold"])
    if "exit" in changes:
        exp = replace(exp, exit=changes["exit"])
    if "name" in changes:
        exp = replace(exp, name=changes["name"])
    return exp


def evaluate_orb_experiment(exp, seed_offset=0):
    row, _ = _evaluate_experiment(
        experiment=exp,
        context=orb_context,
        bootstrap_paths=ORB_GRID_BOOTSTRAP_PATHS,
        random_seed=1000 + int(seed_offset),
        keep_details=False,
        max_leverage=ORB_MAX_LEVERAGE,
    )
    return row


noise_rows = []
counter = 0
for lookback, vm in product(ORB_NOISE_LOOKBACK_GRID, ORB_NOISE_VM_GRID):
    counter += 1
    dyn = replace(orb_dynamic, noise_lookback=int(lookback), noise_vm=float(vm))
    row = evaluate_orb_experiment(
        orb_experiment_with(name=f"local_noise_L{lookback}_VM{vm}", dynamic_threshold=dyn),
        seed_offset=counter,
    )
    row.update({"noise_lookback": int(lookback), "noise_vm": float(vm)})
    noise_rows.append(row)

orb_noise_grid = pd.DataFrame(noise_rows)
orb_noise_ok = orb_noise_grid.loc[orb_noise_grid["status"].eq("ok")].copy()
orb_noise_ok["oos_max_drawdown_abs"] = orb_noise_ok["oos_max_drawdown"].abs()

display(Markdown("### Noise-area gate local grid"))
display(
    orb_noise_ok.sort_values(["oos_prop_score", "oos_sharpe_ratio", "oos_net_pnl"], ascending=[False, False, False])
    [["noise_lookback", "noise_vm", "oos_net_pnl", "oos_sharpe_ratio", "oos_profit_factor", "oos_max_drawdown_abs", "oos_prop_score", "oos_nb_trades"]]
    .round(3)
    .head(12)
)
plot_heatmap(orb_noise_ok, "noise_vm", "noise_lookback", "oos_sharpe_ratio", "ORB avance OOS Sharpe | noise gate", text_auto=".2f")
plot_heatmap(orb_noise_ok, "noise_vm", "noise_lookback", "oos_net_pnl", "ORB avance OOS Net PnL | noise gate", text_auto=".0f")
plot_heatmap(orb_noise_ok, "noise_vm", "noise_lookback", "oos_max_drawdown_abs", "ORB avance OOS Max Drawdown Abs | noise gate", color_continuous_scale="RdYlGn_r", text_auto=".0f")
plot_heatmap(orb_noise_ok, "noise_vm", "noise_lookback", "oos_prop_score", "ORB avance OOS Prop Score | noise gate", text_auto=".2f")

ensemble_rows = []
counter = 100
for atr_window, vote_threshold in product(ORB_ATR_WINDOW_GRID, ORB_VOTE_THRESHOLD_GRID):
    counter += 1
    ensemble = replace(orb_ensemble, atr_window=int(atr_window), vote_threshold=float(vote_threshold))
    row = evaluate_orb_experiment(
        orb_experiment_with(name=f"local_ensemble_atr{atr_window}_vote{vote_threshold}", baseline_ensemble=ensemble),
        seed_offset=counter,
    )
    row.update({"atr_window": int(atr_window), "vote_threshold": float(vote_threshold)})
    ensemble_rows.append(row)

orb_ensemble_grid = pd.DataFrame(ensemble_rows)
orb_ensemble_ok = orb_ensemble_grid.loc[orb_ensemble_grid["status"].eq("ok")].copy()
orb_ensemble_ok["oos_max_drawdown_abs"] = orb_ensemble_ok["oos_max_drawdown"].abs()

display(Markdown("### ATR ensemble local grid"))
display(
    orb_ensemble_ok.sort_values(["oos_prop_score", "oos_sharpe_ratio", "oos_net_pnl"], ascending=[False, False, False])
    [["atr_window", "vote_threshold", "selected_days", "oos_net_pnl", "oos_sharpe_ratio", "oos_profit_factor", "oos_max_drawdown_abs", "oos_prop_score", "oos_nb_trades"]]
    .round(3)
    .head(12)
)
plot_heatmap(orb_ensemble_ok, "vote_threshold", "atr_window", "oos_sharpe_ratio", "ORB avance OOS Sharpe | ATR ensemble", text_auto=".2f")
plot_heatmap(orb_ensemble_ok, "vote_threshold", "atr_window", "oos_net_pnl", "ORB avance OOS Net PnL | ATR ensemble", text_auto=".0f")
plot_heatmap(orb_ensemble_ok, "vote_threshold", "atr_window", "oos_max_drawdown_abs", "ORB avance OOS Max Drawdown Abs | ATR ensemble", color_continuous_scale="RdYlGn_r", text_auto=".0f")

display(Markdown("### Lecture rapide"))
display(Markdown(
    f"- Config chargee: `{ORB_RESEARCH_CONFIG_NAME}`.\\n"
    f"- Le point par defaut est `noise_lookback={ORB_NOISE_LOOKBACK}`, `noise_vm={ORB_NOISE_VM}`, `atr_window={ORB_ATR_WINDOW}`, `vote_threshold={ORB_ENSEMBLE_VOTE_THRESHOLD}`.\\n"
    "- Les heatmaps ci-dessus recalculent chaque point depuis les donnees minute, elles ne relisent pas les exports de performance."
))
"""
    )


def _pullback_markdown_cell() -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        """## 4. Volume climax pullback: reconstruction du signal

Logique de recherche:

1. On part des memes barres MNQ 1 minute, puis on extrait la session RTH et on resample en barres 1h.
2. Le setup detecte une bougie de volume extreme par rapport a son historique roulant.
3. La bougie setup doit avoir un corps suffisant et un range suffisant en ATR.
4. Le trade est pris sur la barre suivante (`next_open`): short apres climax haussier, long apres climax baissier.
5. Le stop vient de l'extreme de la bougie setup. Le target ici est `1.0 ATR`, avec time stop a 2 barres.
6. Le sizing pullback est un sizing risque a capital constant: `risk_pct` est applique sur `INITIAL_CAPITAL_USD`, cap `max_contracts`, et possibilite de skipper un trade trop petit.
"""
    )


def _pullback_run_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """pullback_variant = VolumeClimaxPullbackV2Variant(
    name="client_pullback_core",
    family="dynamic_exit",
    timeframe="1h",
    volume_quantile=PULLBACK_VOLUME_QUANTILE,
    volume_lookback=PULLBACK_VOLUME_LOOKBACK,
    min_body_fraction=PULLBACK_MIN_BODY_FRACTION,
    min_range_atr=PULLBACK_MIN_RANGE_ATR,
    trend_ema_window=None,
    ema_slope_threshold=None,
    atr_percentile_low=None,
    atr_percentile_high=None,
    compression_ratio_max=None,
    entry_mode=PULLBACK_ENTRY_MODE,
    pullback_fraction=None,
    confirmation_window=None,
    exit_mode=PULLBACK_EXIT_MODE,
    rr_target=1.0,
    atr_target_multiple=PULLBACK_ATR_TARGET_MULTIPLE,
    time_stop_bars=PULLBACK_TIME_STOP_BARS,
    trailing_atr_multiple=PULLBACK_TRAILING_ATR_MULTIPLE,
    session_overlay="all_rth",
)

pullback_bars_1h = resample_rth_1h(raw_1m)
pullback_bars_1h["timestamp"] = pd.to_datetime(pullback_bars_1h["timestamp"], errors="coerce")
pullback_bars_1h["session_date"] = pullback_bars_1h["timestamp"].dt.date
pullback_is_sessions, pullback_oos_sessions = split_sessions(pullback_bars_1h[["session_date"]].copy(), ratio=IS_FRACTION)
pullback_all_sessions = normalize_sessions(sorted(pd.to_datetime(pullback_bars_1h["session_date"]).dt.date.unique()))
pullback_is_sessions = normalize_sessions(pullback_is_sessions)
pullback_oos_sessions = normalize_sessions(pullback_oos_sessions)

pullback_features = prepare_volume_climax_pullback_v2_features(pullback_bars_1h)
pullback_signal_df = build_volume_climax_pullback_v2_signal_frame(pullback_features, pullback_variant)
pullback_oos_signal_df = pullback_signal_df.loc[pd.to_datetime(pullback_signal_df["session_date"]).dt.date.isin(session_set(pullback_oos_sessions))].copy()

pullback_execution, pullback_instrument = build_execution_model_for_profile(symbol=SYMBOL, profile_name="repo_realistic")
pullback_sizing = RiskPercentPositionSizing(
    initial_capital_usd=INITIAL_CAPITAL_USD,
    risk_pct=PULLBACK_RISK_PCT,
    max_contracts=PULLBACK_MAX_CONTRACTS,
    skip_trade_if_too_small=PULLBACK_SKIP_TRADE_IF_TOO_SMALL,
    compound_realized_pnl=PULLBACK_COMPOUND_REALIZED_PNL,
)

pullback_full_result = run_volume_climax_pullback_v2_backtest(
    signal_df=pullback_signal_df,
    variant=pullback_variant,
    execution_model=pullback_execution,
    instrument=pullback_instrument,
    position_sizing=pullback_sizing,
)
pullback_oos_result = run_volume_climax_pullback_v2_backtest(
    signal_df=pullback_oos_signal_df,
    variant=pullback_variant,
    execution_model=pullback_execution,
    instrument=pullback_instrument,
    position_sizing=pullback_sizing,
)

pullback_trades = pullback_full_result.trades.copy()
pullback_trades_oos = pullback_oos_result.trades.copy()
pullback_daily_full = daily_results_from_trades(pullback_trades, pullback_all_sessions, INITIAL_CAPITAL_USD)
pullback_daily_oos = daily_results_from_trades(pullback_trades_oos, pullback_oos_sessions, INITIAL_CAPITAL_USD)
pullback_summary = summarize_scopes("Pullback", pullback_daily_full, pullback_trades, pullback_is_sessions, pullback_oos_sessions, INITIAL_CAPITAL_USD, daily_oos_rebased=pullback_daily_oos, trades_oos=pullback_trades_oos)

display(Markdown("### Pullback run summary"))
display(
    pd.DataFrame(
        [
            {"item": "bars_1h", "value": int(len(pullback_bars_1h))},
            {"item": "sessions_all", "value": len(pullback_all_sessions)},
            {"item": "sessions_is", "value": len(pullback_is_sessions)},
            {"item": "sessions_oos", "value": len(pullback_oos_sessions)},
            {"item": "raw_signal_count", "value": int(pd.to_numeric(pullback_signal_df["raw_signal"], errors="coerce").ne(0).sum())},
            {"item": "filtered_signal_count", "value": int(pd.to_numeric(pullback_signal_df["signal"], errors="coerce").ne(0).sum())},
            {"item": "full_trades", "value": int(len(pullback_trades))},
            {"item": "oos_trades_rebased", "value": int(len(pullback_trades_oos))},
        ]
    )
)
display(pd.DataFrame([asdict(pullback_variant)]).T.rename(columns={0: "value"}))
display(pullback_summary.round(3))
"""
    )


def _pullback_grid_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 5. Pullback heatmaps: alpha puis sizing"))

alpha_rows = []
for volume_quantile, body_fraction, range_atr in product(PULLBACK_ALPHA_VOLUME_QUANTILES, PULLBACK_ALPHA_BODY_FRACTIONS, PULLBACK_ALPHA_RANGE_ATRS):
    alpha_variant = VolumeClimaxPullbackV2Variant(
        name=f"alpha_vq{volume_quantile}_bf{body_fraction}_ra{range_atr}",
        family="alpha_grid",
        timeframe="1h",
        volume_quantile=float(volume_quantile),
        volume_lookback=PULLBACK_VOLUME_LOOKBACK,
        min_body_fraction=float(body_fraction),
        min_range_atr=float(range_atr),
        trend_ema_window=None,
        ema_slope_threshold=None,
        atr_percentile_low=None,
        atr_percentile_high=None,
        compression_ratio_max=None,
        entry_mode=PULLBACK_ENTRY_MODE,
        pullback_fraction=None,
        confirmation_window=None,
        exit_mode=PULLBACK_EXIT_MODE,
        rr_target=1.0,
        atr_target_multiple=PULLBACK_ATR_TARGET_MULTIPLE,
        time_stop_bars=PULLBACK_TIME_STOP_BARS,
        trailing_atr_multiple=PULLBACK_TRAILING_ATR_MULTIPLE,
        session_overlay="all_rth",
    )
    alpha_signal = build_volume_climax_pullback_v2_signal_frame(pullback_features, alpha_variant)
    alpha_oos_signal = alpha_signal.loc[pd.to_datetime(alpha_signal["session_date"]).dt.date.isin(session_set(pullback_oos_sessions))].copy()
    alpha_result = run_volume_climax_pullback_v2_backtest(
        signal_df=alpha_oos_signal,
        variant=alpha_variant,
        execution_model=pullback_execution,
        instrument=pullback_instrument,
        position_sizing=pullback_sizing,
    )
    alpha_daily = daily_results_from_trades(alpha_result.trades, pullback_oos_sessions, INITIAL_CAPITAL_USD)
    alpha_rows.append(
        {
            "volume_quantile": float(volume_quantile),
            "min_body_fraction": float(body_fraction),
            "min_range_atr": float(range_atr),
            "signal_count_full": int(pd.to_numeric(alpha_signal["signal"], errors="coerce").ne(0).sum()),
            **curve_metrics(alpha_daily, alpha_result.trades, INITIAL_CAPITAL_USD),
        }
    )

pullback_alpha_grid = pd.DataFrame(alpha_rows)
display(Markdown("### Alpha grid OOS"))
display(pullback_alpha_grid.sort_values(["sharpe", "net_pnl_usd"], ascending=[False, False]).round(3))

for range_atr in sorted(pullback_alpha_grid["min_range_atr"].unique()):
    alpha_slice = pullback_alpha_grid.loc[pullback_alpha_grid["min_range_atr"].eq(range_atr)].copy()
    plot_heatmap(alpha_slice, "volume_quantile", "min_body_fraction", "sharpe", f"Pullback alpha OOS Sharpe | min_range_atr={range_atr}", text_auto=".2f")
    plot_heatmap(alpha_slice, "volume_quantile", "min_body_fraction", "net_pnl_usd", f"Pullback alpha OOS Net PnL | min_range_atr={range_atr}", text_auto=".0f")

sizing_rows = []
for risk_pct, max_contracts in product(PULLBACK_RISK_PCT_GRID, PULLBACK_MAX_CONTRACTS_GRID):
    sizing = RiskPercentPositionSizing(
        initial_capital_usd=INITIAL_CAPITAL_USD,
        risk_pct=float(risk_pct),
        max_contracts=int(max_contracts),
        skip_trade_if_too_small=PULLBACK_SKIP_TRADE_IF_TOO_SMALL,
        compound_realized_pnl=PULLBACK_COMPOUND_REALIZED_PNL,
    )
    result = run_volume_climax_pullback_v2_backtest(
        signal_df=pullback_oos_signal_df,
        variant=pullback_variant,
        execution_model=pullback_execution,
        instrument=pullback_instrument,
        position_sizing=sizing,
    )
    daily = daily_results_from_trades(result.trades, pullback_oos_sessions, INITIAL_CAPITAL_USD)
    row = {
        "risk_pct": float(risk_pct),
        "max_contracts": int(max_contracts),
        "trades_entered": int(len(result.trades)),
        "trades_skipped": int(pd.Series(result.sizing_decisions.get("skipped"), dtype="boolean").fillna(False).sum()) if not result.sizing_decisions.empty else 0,
        **curve_metrics(daily, result.trades, INITIAL_CAPITAL_USD),
    }
    sizing_rows.append(row)

pullback_sizing_grid = pd.DataFrame(sizing_rows)
display(Markdown("### Risk sizing grid OOS"))
display(pullback_sizing_grid.sort_values(["sharpe", "net_pnl_usd"], ascending=[False, False]).round(3).head(15))

plot_heatmap(pullback_sizing_grid, "max_contracts", "risk_pct", "net_pnl_usd", "Pullback sizing OOS Net PnL", text_auto=".0f")
plot_heatmap(pullback_sizing_grid, "max_contracts", "risk_pct", "sharpe", "Pullback sizing OOS Sharpe", text_auto=".2f")
plot_heatmap(pullback_sizing_grid, "max_contracts", "risk_pct", "max_drawdown_usd", "Pullback sizing OOS Max Drawdown", color_continuous_scale="RdYlGn_r", text_auto=".0f")
"""
    )


def _portfolio_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 6. Comparaison standalone, blend equal weight et benchmark"))

weights = np.array([float(ORB_WEIGHT), float(PULLBACK_WEIGHT)], dtype=float)
weights = weights / weights.sum()
orb_weight, pullback_weight = float(weights[0]), float(weights[1])

comparison = (
    orb_daily_full[["session_date", "daily_return", "equity", "daily_pnl_usd"]]
    .rename(columns={"daily_return": "orb_return", "equity": "orb_equity", "daily_pnl_usd": "orb_pnl"})
    .merge(
        pullback_daily_full[["session_date", "daily_return", "equity", "daily_pnl_usd"]].rename(
            columns={"daily_return": "pullback_return", "equity": "pullback_equity", "daily_pnl_usd": "pullback_pnl"}
        ),
        on="session_date",
        how="inner",
    )
    .sort_values("session_date")
    .reset_index(drop=True)
)

comparison["orb_return_scaled"] = scale_daily_returns(comparison["orb_return"], ORB_LEVERAGE, "ORB")
comparison["pullback_return_scaled"] = scale_daily_returns(comparison["pullback_return"], PULLBACK_LEVERAGE, "Pullback")
comparison["blend_pre_leverage_return"] = orb_weight * comparison["orb_return_scaled"] + pullback_weight * comparison["pullback_return_scaled"]
comparison["blend_return"] = scale_daily_returns(comparison["blend_pre_leverage_return"], BLEND_LEVERAGE, "Blend")

orb_curve_common = curve_from_returns(comparison["session_date"], comparison["orb_return_scaled"], INITIAL_CAPITAL_USD, "ORB common")
pullback_curve_common = curve_from_returns(comparison["session_date"], comparison["pullback_return_scaled"], INITIAL_CAPITAL_USD, "Pullback common")
blend_curve_common = curve_from_returns(comparison["session_date"], comparison["blend_return"], INITIAL_CAPITAL_USD, "Blend common")

daily_close = raw_1m.copy()
daily_close_ts = pd.to_datetime(daily_close["timestamp"], errors="coerce")
if getattr(daily_close_ts.dt, "tz", None) is not None:
    daily_close_ts = daily_close_ts.dt.tz_localize(None)
daily_close["session_date"] = daily_close_ts.dt.normalize()
daily_close = daily_close.groupby("session_date", as_index=False)["close"].last()
benchmark = comparison[["session_date"]].merge(daily_close, on="session_date", how="left").sort_values("session_date").reset_index(drop=True)
benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce").ffill()
benchmark["benchmark_return"] = benchmark["close"].pct_change().fillna(0.0)
benchmark["benchmark_return"] = scale_daily_returns(benchmark["benchmark_return"], BENCHMARK_LEVERAGE, "Benchmark")
benchmark_curve_common = curve_from_returns(benchmark["session_date"], benchmark["benchmark_return"], INITIAL_CAPITAL_USD, "Benchmark common")

common_oos_start = max(pd.to_datetime(orb_oos_sessions).min(), pd.to_datetime(pullback_oos_sessions).min())
oos_mask = comparison["session_date"] >= common_oos_start
comparison_oos = comparison.loc[oos_mask].copy().reset_index(drop=True)
benchmark_oos = benchmark.loc[benchmark["session_date"] >= common_oos_start].copy().reset_index(drop=True)

orb_curve_oos = curve_from_returns(comparison_oos["session_date"], comparison_oos["orb_return_scaled"], INITIAL_CAPITAL_USD, "ORB OOS common")
pullback_curve_oos = curve_from_returns(comparison_oos["session_date"], comparison_oos["pullback_return_scaled"], INITIAL_CAPITAL_USD, "Pullback OOS common")
blend_curve_oos = curve_from_returns(comparison_oos["session_date"], comparison_oos["blend_return"], INITIAL_CAPITAL_USD, "Blend OOS common")
benchmark_curve_oos = curve_from_returns(benchmark_oos["session_date"], benchmark_oos["benchmark_return"], INITIAL_CAPITAL_USD, "Benchmark OOS common")

scorecard_rows = []
for scope, curves in [
    ("full_common", {
        "ORB": orb_curve_common,
        "Pullback": pullback_curve_common,
        "Blend": blend_curve_common,
        "Benchmark": benchmark_curve_common,
    }),
    ("oos_common", {
        "ORB": orb_curve_oos,
        "Pullback": pullback_curve_oos,
        "Blend": blend_curve_oos,
        "Benchmark": benchmark_curve_oos,
    }),
]:
    for name, curve in curves.items():
        scorecard_rows.append({"strategy": name, "scope": scope, **curve_metrics(curve, None, INITIAL_CAPITAL_USD)})
portfolio_scorecard = pd.DataFrame(scorecard_rows)

display(Markdown(f"Common sample: `{comparison['session_date'].min().date()}` -> `{comparison['session_date'].max().date()}`"))
display(Markdown(f"Common OOS start: `{common_oos_start.date()}`"))
display(portfolio_scorecard.round(3))

fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12, subplot_titles=("Full common sample", "OOS rebased"))
series_full = [
    ("ORB", "#2563eb", orb_curve_common),
    ("Pullback", "#16a34a", pullback_curve_common),
    ("Blend", "#f59e0b", blend_curve_common),
    ("Benchmark", "#a3a3a3", benchmark_curve_common),
]
for name, color, curve in series_full:
    fig.add_trace(
        go.Scatter(x=curve["session_date"], y=curve["equity"], mode="lines", name=name, line=dict(color=color, width=2.8 if name == "Blend" else 2.0, dash="dash" if name == "Benchmark" else "solid")),
        row=1,
        col=1,
    )

series_oos = [
    ("ORB OOS", "#2563eb", orb_curve_oos),
    ("Pullback OOS", "#16a34a", pullback_curve_oos),
    ("Blend OOS", "#f59e0b", blend_curve_oos),
    ("Benchmark OOS", "#a3a3a3", benchmark_curve_oos),
]
for name, color, curve in series_oos:
    fig.add_trace(
        go.Scatter(x=curve["session_date"], y=curve["equity"], mode="lines", name=name, showlegend=False, line=dict(color=color, width=2.8 if name == "Blend OOS" else 2.0, dash="dash" if name == "Benchmark OOS" else "solid")),
        row=2,
        col=1,
    )

fig.update_yaxes(title_text="Equity USD", row=1, col=1)
fig.update_yaxes(title_text="Equity USD", row=2, col=1)
fig.update_layout(template=PLOT_TEMPLATE, width=1450, height=900, legend=dict(orientation="h", y=-0.08))
fig.show()
"""
    )


def _diagnostics_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 7. Diagnostics de diversification"))

ret_diag = comparison[["session_date", "orb_return_scaled", "pullback_return_scaled", "blend_return"]].copy()
correlation_full = float(ret_diag[["orb_return_scaled", "pullback_return_scaled"]].corr().iloc[0, 1])
correlation_oos = float(ret_diag.loc[ret_diag["session_date"] >= common_oos_start, ["orb_return_scaled", "pullback_return_scaled"]].corr().iloc[0, 1])
display(Markdown(f"Correlation journaliere ORB / Pullback: full `{correlation_full:.3f}` | OOS `{correlation_oos:.3f}`."))

scatter = px.scatter(
    ret_diag,
    x="orb_return_scaled",
    y="pullback_return_scaled",
    color=ret_diag["session_date"].ge(common_oos_start).map({True: "oos", False: "is"}),
    title="Daily returns: ORB vs Pullback",
)
scatter.update_layout(template=PLOT_TEMPLATE, width=900, height=550)
scatter.show()

ret_diag["rolling_corr_63d"] = ret_diag["orb_return_scaled"].rolling(63).corr(ret_diag["pullback_return_scaled"])
roll_fig = px.line(ret_diag, x="session_date", y="rolling_corr_63d", title="Rolling correlation 63 sessions")
roll_fig.update_layout(template=PLOT_TEMPLATE, width=1250, height=450)
roll_fig.show()

monthly = comparison.copy()
monthly["month"] = monthly["session_date"].dt.to_period("M").dt.to_timestamp()
monthly_rollup = monthly.groupby("month", as_index=False).agg(
    orb_pnl=("orb_pnl", "sum"),
    pullback_pnl=("pullback_pnl", "sum"),
)
monthly_rollup["blend_pnl_proxy"] = orb_weight * monthly_rollup["orb_pnl"] + pullback_weight * monthly_rollup["pullback_pnl"]
monthly_long = monthly_rollup.melt(id_vars="month", var_name="sleeve", value_name="pnl_usd")
bar = px.bar(monthly_long, x="month", y="pnl_usd", color="sleeve", barmode="group", title="Monthly PnL contribution")
bar.update_layout(template=PLOT_TEMPLATE, width=1400, height=520)
bar.show()
"""
    )


def _conclusion_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        """display(Markdown("## 8. Lecture finale"))

oos = portfolio_scorecard.loc[portfolio_scorecard["scope"].eq("oos_common")].set_index("strategy")
blend = oos.loc["Blend"]
bench = oos.loc["Benchmark"]
orb = oos.loc["ORB"]
pullback = oos.loc["Pullback"]

lines = [
    f"- ORB OOS common: net `{fmt_money(orb['net_pnl_usd'])}`, Sharpe `{fmt_float(orb['sharpe'])}`, maxDD `{fmt_money(orb['max_drawdown_usd'])}`.",
    f"- Pullback OOS common: net `{fmt_money(pullback['net_pnl_usd'])}`, Sharpe `{fmt_float(pullback['sharpe'])}`, maxDD `{fmt_money(pullback['max_drawdown_usd'])}`.",
    f"- Blend OOS common: net `{fmt_money(blend['net_pnl_usd'])}`, Sharpe `{fmt_float(blend['sharpe'])}`, maxDD `{fmt_money(blend['max_drawdown_usd'])}`.",
    f"- Benchmark OOS common: net `{fmt_money(bench['net_pnl_usd'])}`, Sharpe `{fmt_float(bench['sharpe'])}`, maxDD `{fmt_money(bench['max_drawdown_usd'])}`.",
    f"- Delta blend vs benchmark: net `{fmt_money(blend['net_pnl_usd'] - bench['net_pnl_usd'])}`, maxDD `{fmt_money(blend['max_drawdown_usd'] - bench['max_drawdown_usd'])}`.",
    f"- Les parametres a challenger en premier sont visibles dans les heatmaps: ORB `noise_lookback/noise_vm` puis `atr_window/vote_threshold`, pullback `volume_quantile/body/range_atr`, puis sizing `risk_pct/max_contracts`.",
]
display(Markdown("\\n".join(lines)))
"""
    )


def build_notebook() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"},
    }
    notebook.cells = [
        _title_cell(),
        _imports_cell(),
        _helpers_cell(),
        _parameter_cell(),
        _data_cell(),
        _orb_markdown_cell(),
        _orb_run_cell(),
        _orb_heatmap_cell(),
        _pullback_markdown_cell(),
        _pullback_run_cell(),
        _pullback_grid_cell(),
        _portfolio_cell(),
        _diagnostics_cell(),
        _conclusion_cell(),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK_PATH)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--executed-output", type=Path, default=DEFAULT_EXECUTED_NOTEBOOK_PATH)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebook = build_notebook()
    output_path = write_notebook(notebook, args.output)
    print(f"Notebook written to {output_path}")
    if args.execute:
        executed_path = execute_notebook(output_path, args.executed_output, timeout_seconds=args.timeout_seconds)
        print(f"Executed notebook written to {executed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
