"""Patch the MNQ final ensemble notebook to add a dynamic 3-state sizing overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "orb_MNQ_final_ensemble_validation.ipynb"

LEGACY_INTRO_APPENDIX = """

Le notebook peut maintenant superposer une variante **3-state sizing** sur les memes jours selectionnes par l'ensemble nominal. Cette couche reste separable et parametrique: tu peux changer le `SIZING_FEATURE_NAME`, les multiplicateurs de risque par rang ou par bucket, puis comparer directement la courbe nominale, la courbe 3-state et le buy and hold sur le meme graph.
""".strip()

INTRO_APPENDIX = """

Le notebook peut maintenant comparer plusieurs surcouches sur les memes jours selectionnes par l'ensemble nominal. La premiere reste la variante **3-state sizing** parametrique via `SIZING_FEATURE_NAME` et ses multiplicateurs de risque. La seconde ajoute un **filtre VVIX** audite, branche sur l'export `filter_drop_low__vvix_pct_63_t1`, afin de comparer directement la courbe nominale, la courbe 3-state, la courbe VVIX-filtered et le buy and hold sur le meme graph.
""".strip()

IMPORTS_CELL = """import math
import sys
from pathlib import Path
from types import SimpleNamespace

# Make `src` imports work whether notebook is launched from repo root or /notebooks
ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, Markdown

from src.data.loader import load_ohlcv_file
from src.data.cleaning import clean_ohlcv
from src.features.intraday import add_intraday_features, add_session_vwap, add_continuous_session_vwap
from src.features.opening_range import compute_opening_range
from src.features.volatility import add_atr
from src.strategy.orb import ORBStrategy
from src.engine.execution_model import ExecutionModel
from src.engine.backtester import run_backtest
from src.engine.portfolio import build_equity_curve
from src.analytics.metrics import compute_metrics
from src.analytics.mnq_orb_regime_filter_sizing_campaign import (
    _conditional_rows_for_feature,
    _feature_specs,
    _scale_nominal_trades_by_multiplier,
    build_conditional_bucket_analysis,
    build_regime_dataset,
    build_state_mapping_from_is_scores,
    build_static_regime_controls,
)
from src.analytics.orb_vvix_overlay import build_vvix_filter_controls, find_latest_export
from src.analytics.orb_notebook_utils import (
    build_scope_readout_markdown,
    curve_annualized_return,
    curve_daily_sharpe,
    curve_daily_vol,
    curve_drawdown_pct,
    curve_max_drawdown_pct,
    curve_total_return_pct,
    format_curve_stats_line,
    normalize_curve,
)
from src.config.settings import get_instrument_spec

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)
"""

PARAMETERS_CELL = """ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent

# ----------------------
# Data / split settings
# ----------------------
DATASET_PATH = ROOT / 'data' / 'processed' / 'parquet' / 'MNQ_c_0_1m_20260321_094501.parquet'
IS_FRACTION = 0.70

# ----------------------
# Ensemble settings
# ----------------------
# OR15LONG
# ATR_PERIODS = list(range(30, 60 + 1, 1))
# Q_LOW_VALUES = list(range(25, 30 + 1, 1))
# Q_HIGH_VALUES = list(range(70, 75 + 1, 1))

# OR30LONG
ATR_PERIODS = list(range(15, 20 + 1, 1))
Q_LOW_VALUES = list(range(25, 30 + 1, 1))
Q_HIGH_VALUES = list(range(70, 75 + 1, 1))

# OR30BOTH
# ATR_PERIODS = list(range(10, 30 + 1, 1))
# Q_LOW_VALUES = list(range(20, 25 + 1, 1))
# Q_HIGH_VALUES = list(range(85, 90 + 1, 1))

AGGREGATION_RULE = 'majority_50'      # majority_50 | consensus_75 | consensus_100 | custom
CUSTOM_THRESHOLD = 0.90               # used only if AGGREGATION_RULE == 'custom'

# ----------------------
# Baseline strategy settings
# ----------------------
BASELINE = {
    'or_minutes': 30,
    'opening_time': '09:30:00',
    'direction': 'long',
    'one_trade_per_day': True,
    'entry_buffer_ticks': 2,
    'stop_buffer_ticks': 2,
    'target_multiple': 2.0,
    'vwap_confirmation': True,
    'vwap_column': 'continuous_session_vwap',
    'time_exit': '16:00:00',
    'account_size_usd': 50000.0,
    'risk_per_trade_pct': 1.5,
    'tick_size': 0.25,
    'entry_on_next_open': True,
}

# ----------------------
# Execution / costs
# ----------------------
EXECUTION = {
    'commission_per_side_usd': 0.62,
    'slippage_ticks': 1,
    'tick_size': 0.25,
}

# ----------------------
# 3-state sizing comparison
# ----------------------
ENABLE_3STATE_COMPARISON = True
SIZING_FEATURE_NAME = 'realized_vol_ratio_15_60'
SIZING_BUCKET_MULTIPLIERS_BY_RANK = (0.50, 0.75, 1.00)   # worst IS bucket -> best IS bucket
SIZING_BUCKET_MULTIPLIERS_OVERRIDE = {}                   # example: {'low': 0.50, 'mid': 1.00, 'high': 0.75}
SIZING_MIN_BUCKET_OBS_IS = 50

# ----------------------
# VVIX regime filter comparison
# ----------------------
ENABLE_VVIX_FILTER_COMPARISON = True
VVIX_VALIDATION_EXPORT_ROOT = find_latest_export('mnq_orb_vix_vvix_validation')
VVIX_FILTER_VARIANT_NAME = 'filter_drop_low__vvix_pct_63_t1'
VVIX_FILTER_FEATURE_NAME_OVERRIDE = ''
VVIX_FILTER_KEEP_BUCKETS_OVERRIDE = ()                    # example: ('mid', 'high')
VVIX_VIX_DAILY_PATH = ROOT / 'data' / 'raw' / 'vix-daily.csv'
VVIX_VVIX_DAILY_PATH = ROOT / 'data' / 'raw' / 'vvix-daily.csv'

print('DATASET_PATH      =', DATASET_PATH)
print('IS_FRACTION       =', IS_FRACTION)
print('ATR_PERIODS       =', ATR_PERIODS)
print('Q_LOW_VALUES      =', Q_LOW_VALUES)
print('Q_HIGH_VALUES     =', Q_HIGH_VALUES)
print('N_SUBSIGNALS_MAX  =', len(ATR_PERIODS) * len(Q_LOW_VALUES) * len(Q_HIGH_VALUES))
print('AGG_RULE          =', AGGREGATION_RULE)
print('ENABLE_3STATE     =', ENABLE_3STATE_COMPARISON)
if ENABLE_3STATE_COMPARISON:
    print('SIZING_FEATURE    =', SIZING_FEATURE_NAME)
    print('SIZING_RANK_MULT  =', SIZING_BUCKET_MULTIPLIERS_BY_RANK)
    print('SIZING_OVERRIDE   =', SIZING_BUCKET_MULTIPLIERS_OVERRIDE)
    print('SIZING_MIN_OBS_IS =', SIZING_MIN_BUCKET_OBS_IS)
print('ENABLE_VVIX       =', ENABLE_VVIX_FILTER_COMPARISON)
if ENABLE_VVIX_FILTER_COMPARISON:
    print('VVIX_EXPORT_ROOT  =', VVIX_VALIDATION_EXPORT_ROOT)
    print('VVIX_VARIANT      =', VVIX_FILTER_VARIANT_NAME)
    print('VVIX_FEATURE_OVR  =', VVIX_FILTER_FEATURE_NAME_OVERRIDE or '<export-default>')
    print('VVIX_KEEP_OVR     =', VVIX_FILTER_KEEP_BUCKETS_OVERRIDE or '<export-default>')
"""

BACKTEST_CELL = """def _to_naive_utc(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors='coerce')
    return ts.dt.tz_convert(None)


initial_capital = float(BASELINE['account_size_usd'])


def drawdown_pct_from_equity(eq: pd.Series) -> pd.Series:
    return curve_drawdown_pct(eq)


def sharpe_daily_from_equity(df: pd.DataFrame) -> float:
    return curve_daily_sharpe(df)


def annualized_return_from_equity(df: pd.DataFrame) -> float:
    return curve_annualized_return(df, initial_capital)


def vol_daily_from_equity(df: pd.DataFrame) -> float:
    return curve_daily_vol(df)


def total_return_pct(df: pd.DataFrame) -> float:
    return curve_total_return_pct(df, initial_capital)


def max_drawdown_pct(df: pd.DataFrame) -> float:
    return curve_max_drawdown_pct(df)


def format_stats_line(name: str, sharpe: float, ret_pct: float, cagr_pct: float, vol_pct: float, dd_pct: float, pf: float | None = None, exp: float | None = None) -> str:
    return format_curve_stats_line(
        name=name,
        sharpe=sharpe,
        ret_pct=ret_pct,
        cagr_pct=cagr_pct,
        vol_pct=vol_pct,
        dd_pct=dd_pct,
        pf=pf,
        exp=exp,
    )


# Run ensemble backtest with explicit execution settings
exec_model = ExecutionModel(
    commission_per_side_usd=float(EXECUTION['commission_per_side_usd']),
    slippage_ticks=int(EXECUTION['slippage_ticks']),
    tick_size=float(EXECUTION['tick_size']),
)

ensemble_trades = run_backtest(
    ensemble_signal_df,
    execution_model=exec_model,
    time_exit=str(BASELINE['time_exit']),
    stop_buffer_ticks=int(BASELINE['stop_buffer_ticks']),
    target_multiple=float(BASELINE['target_multiple']),
    account_size_usd=float(BASELINE['account_size_usd']),
    risk_per_trade_pct=float(BASELINE['risk_per_trade_pct']),
    entry_on_next_open=bool(BASELINE['entry_on_next_open']),
)

ensemble_eq = normalize_curve(
    build_equity_curve(ensemble_trades, initial_capital=float(BASELINE['account_size_usd']))
)

ensemble_sessions = set(pd.to_datetime(candidates.loc[candidates['ensemble_pass'], 'session_date']).dt.date)

selected_nominal_trades = baseline_trades.copy()
selected_nominal_trades['session_date'] = pd.to_datetime(selected_nominal_trades['session_date']).dt.date
selected_nominal_trades = selected_nominal_trades.loc[selected_nominal_trades['session_date'].isin(ensemble_sessions)].copy()

# Buy and hold from daily close
bench_src = feat[['timestamp', 'close', 'session_date']].copy()
bench_src['timestamp'] = _to_naive_utc(bench_src['timestamp'])
bench_src['close'] = pd.to_numeric(bench_src['close'], errors='coerce')
bench_src = bench_src.dropna(subset=['timestamp', 'close'])
daily_close = bench_src.groupby('session_date', as_index=True)['close'].last().dropna()

bench = normalize_curve(pd.DataFrame({
    'timestamp': pd.to_datetime(daily_close.index),
    'equity': float(BASELINE['account_size_usd']) * (daily_close / daily_close.iloc[0]),
}).sort_values('timestamp').reset_index(drop=True))

# Ensemble metrics
ensemble_overall = compute_metrics(
    ensemble_trades,
    signal_df=ensemble_signal_df,
    session_dates=all_sessions,
    initial_capital=float(BASELINE['account_size_usd']),
)
ensemble_is = compute_metrics(
    ensemble_trades.loc[ensemble_trades['session_date'].isin(set(is_sessions))].copy(),
    session_dates=is_sessions,
    initial_capital=float(BASELINE['account_size_usd']),
)
ensemble_oos = compute_metrics(
    ensemble_trades.loc[ensemble_trades['session_date'].isin(set(oos_sessions))].copy(),
    session_dates=oos_sessions,
    initial_capital=float(BASELINE['account_size_usd']),
)

# Dynamic 3-state sizing overlay on the same ensemble-selected sessions
sizing_enabled = bool(ENABLE_3STATE_COMPARISON)
sizing_state_map = {}
state_map_text = 'disabled'
sizing_feature_rows = pd.DataFrame()
sizing_feature_scores = pd.DataFrame()
sizing_controls = pd.DataFrame()
sizing_trades = pd.DataFrame()
sizing_eq = pd.DataFrame(columns=['timestamp', 'equity', 'drawdown', 'drawdown_pct'])
sizing_selected_days = 0

sizing_overall = compute_metrics(
    pd.DataFrame(),
    session_dates=all_sessions,
    initial_capital=float(BASELINE['account_size_usd']),
)
sizing_is = compute_metrics(
    pd.DataFrame(),
    session_dates=is_sessions,
    initial_capital=float(BASELINE['account_size_usd']),
)
sizing_oos = compute_metrics(
    pd.DataFrame(),
    session_dates=oos_sessions,
    initial_capital=float(BASELINE['account_size_usd']),
)

# Audited VVIX regime filter overlay on the same ensemble-selected sessions
vvix_filter_enabled = bool(ENABLE_VVIX_FILTER_COMPARISON)
vvix_overlay_spec = None
vvix_feature_name = ''
vvix_feature_rows = pd.DataFrame()
vvix_filter_controls = pd.DataFrame()
vvix_filter_trades = pd.DataFrame()
vvix_filter_eq = pd.DataFrame(columns=['timestamp', 'equity', 'drawdown', 'drawdown_pct'])
vvix_selected_days = 0
vvix_state_text = 'disabled'
vvix_kept_buckets = ()

vvix_filter_overall = compute_metrics(
    pd.DataFrame(),
    session_dates=all_sessions,
    initial_capital=float(BASELINE['account_size_usd']),
)
vvix_filter_is = compute_metrics(
    pd.DataFrame(),
    session_dates=is_sessions,
    initial_capital=float(BASELINE['account_size_usd']),
)
vvix_filter_oos = compute_metrics(
    pd.DataFrame(),
    session_dates=oos_sessions,
    initial_capital=float(BASELINE['account_size_usd']),
)

if sizing_enabled:
    regime_feature_specs = tuple(spec for spec in _feature_specs() if spec.name == SIZING_FEATURE_NAME)
    if not regime_feature_specs:
        available = [spec.name for spec in _feature_specs()]
        raise ValueError(f'Unknown SIZING_FEATURE_NAME: {SIZING_FEATURE_NAME}. Available: {available}')

    analysis_proxy = SimpleNamespace(
        signal_df=add_atr(signal_df.copy(), window=(30,)),
        baseline=SimpleNamespace(
            opening_time=str(BASELINE['opening_time']),
            time_exit=str(BASELINE['time_exit']),
            account_size_usd=float(BASELINE['account_size_usd']),
            risk_per_trade_pct=float(BASELINE['risk_per_trade_pct']),
        ),
        candidate_df=candidates[['session_date', 'signal_index']].copy(),
        baseline_trades=baseline_trades.copy(),
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
    )

    regime_df = build_regime_dataset(analysis_proxy, ensemble_sessions)
    conditional_df, feature_score_df, assignments, calibrations = build_conditional_bucket_analysis(
        regime_df=regime_df,
        nominal_trades=selected_nominal_trades,
        initial_capital=float(BASELINE['account_size_usd']),
        feature_specs=regime_feature_specs,
        min_bucket_obs_is=int(SIZING_MIN_BUCKET_OBS_IS),
    )

    sizing_feature_rows = _conditional_rows_for_feature(conditional_df, SIZING_FEATURE_NAME)
    sizing_feature_scores = feature_score_df.loc[feature_score_df['feature_name'].eq(SIZING_FEATURE_NAME)].copy()
    if sizing_feature_rows.empty:
        raise RuntimeError(f'No calibrated bucket rows available for {SIZING_FEATURE_NAME}.')

    if SIZING_BUCKET_MULTIPLIERS_OVERRIDE:
        sizing_state_map = {str(key): float(value) for key, value in SIZING_BUCKET_MULTIPLIERS_OVERRIDE.items()}
    else:
        sizing_state_map = build_state_mapping_from_is_scores(
            sizing_feature_rows,
            tuple(float(value) for value in SIZING_BUCKET_MULTIPLIERS_BY_RANK),
        )

    sizing_controls = build_static_regime_controls(
        regime_df=regime_df,
        feature_name=SIZING_FEATURE_NAME,
        bucket_labels=assignments[SIZING_FEATURE_NAME],
        bucket_multipliers=sizing_state_map,
    )

    instrument_spec = get_instrument_spec('MNQ')
    sizing_trades = _scale_nominal_trades_by_multiplier(
        nominal_trades=selected_nominal_trades,
        controls=sizing_controls,
        account_size_usd=float(BASELINE['account_size_usd']),
        base_risk_pct=float(BASELINE['risk_per_trade_pct']),
        tick_value_usd=float(instrument_spec['tick_value_usd']),
        point_value_usd=float(instrument_spec['point_value_usd']),
        commission_per_side_usd=float(EXECUTION['commission_per_side_usd']),
    )
    sizing_eq = normalize_curve(
        build_equity_curve(sizing_trades, initial_capital=float(BASELINE['account_size_usd']))
    )

    sizing_overall = compute_metrics(
        sizing_trades,
        session_dates=all_sessions,
        initial_capital=float(BASELINE['account_size_usd']),
    )
    sizing_is = compute_metrics(
        sizing_trades.loc[pd.to_datetime(sizing_trades['session_date']).dt.date.isin(set(is_sessions))].copy(),
        session_dates=is_sessions,
        initial_capital=float(BASELINE['account_size_usd']),
    )
    sizing_oos = compute_metrics(
        sizing_trades.loc[pd.to_datetime(sizing_trades['session_date']).dt.date.isin(set(oos_sessions))].copy(),
        session_dates=oos_sessions,
        initial_capital=float(BASELINE['account_size_usd']),
    )

    ordered_labels = sizing_feature_rows['bucket_label'].astype(str).tolist()
    state_map_text = ', '.join(
        f"{label}={float(sizing_state_map.get(label, 0.0)):.2f}x"
        for label in ordered_labels
    )
    sizing_selected_days = int(
        pd.to_numeric(sizing_controls['risk_multiplier'], errors='coerce').fillna(0.0).gt(0.0).sum()
    )

if vvix_filter_enabled:
    override_feature_name = str(VVIX_FILTER_FEATURE_NAME_OVERRIDE).strip() or None
    kept_override = tuple(str(value) for value in VVIX_FILTER_KEEP_BUCKETS_OVERRIDE) if VVIX_FILTER_KEEP_BUCKETS_OVERRIDE else ()
    vvix_overlay_spec, vvix_filter_controls = build_vvix_filter_controls(
        session_dates=sorted(ensemble_sessions),
        export_root=VVIX_VALIDATION_EXPORT_ROOT,
        variant_name=VVIX_FILTER_VARIANT_NAME,
        feature_name=override_feature_name,
        kept_buckets_override=kept_override,
        vix_path=VVIX_VIX_DAILY_PATH,
        vvix_path=VVIX_VVIX_DAILY_PATH,
    )
    vvix_feature_name = vvix_overlay_spec.feature_name
    vvix_kept_buckets = vvix_overlay_spec.kept_buckets
    vvix_feature_rows = vvix_overlay_spec.bucket_rows.copy()
    vvix_filter_controls['phase'] = np.where(
        pd.to_datetime(vvix_filter_controls['session_date']).dt.date.isin(set(is_sessions)),
        'is',
        'oos',
    )

    vvix_selected_sessions = set(
        pd.to_datetime(vvix_filter_controls.loc[vvix_filter_controls['selected'], 'session_date']).dt.date
    )
    vvix_filter_trades = selected_nominal_trades.loc[
        pd.to_datetime(selected_nominal_trades['session_date']).dt.date.isin(vvix_selected_sessions)
    ].copy()
    vvix_filter_eq = normalize_curve(
        build_equity_curve(vvix_filter_trades, initial_capital=float(BASELINE['account_size_usd']))
    )

    vvix_filter_overall = compute_metrics(
        vvix_filter_trades,
        session_dates=all_sessions,
        initial_capital=float(BASELINE['account_size_usd']),
    )
    vvix_filter_is = compute_metrics(
        vvix_filter_trades.loc[pd.to_datetime(vvix_filter_trades['session_date']).dt.date.isin(set(is_sessions))].copy(),
        session_dates=is_sessions,
        initial_capital=float(BASELINE['account_size_usd']),
    )
    vvix_filter_oos = compute_metrics(
        vvix_filter_trades.loc[pd.to_datetime(vvix_filter_trades['session_date']).dt.date.isin(set(oos_sessions))].copy(),
        session_dates=oos_sessions,
        initial_capital=float(BASELINE['account_size_usd']),
    )

    vvix_selected_days = int(vvix_filter_controls['selected'].sum())
    vvix_state_text = (
        f"feature {vvix_feature_name} | keep {', '.join(vvix_kept_buckets)} | "
        f"variant {VVIX_FILTER_VARIANT_NAME}"
    )

print('ensemble_trades =', len(ensemble_trades))
print('ensemble_is_sharpe =', float(ensemble_is.get('sharpe_ratio', 0.0)))
print('ensemble_oos_sharpe =', float(ensemble_oos.get('sharpe_ratio', 0.0)))
print('ensemble_overall_sharpe =', float(ensemble_overall.get('sharpe_ratio', 0.0)))
print('buy_hold_total_return_pct =', total_return_pct(bench))
if sizing_enabled:
    print('sizing_feature =', SIZING_FEATURE_NAME)
    print('sizing_state_map =', sizing_state_map)
    print('sizing_selected_days =', sizing_selected_days)
    print('sizing_trades =', len(sizing_trades))
    print('sizing_is_sharpe =', float(sizing_is.get('sharpe_ratio', 0.0)))
    print('sizing_oos_sharpe =', float(sizing_oos.get('sharpe_ratio', 0.0)))
if vvix_filter_enabled:
    print('vvix_variant =', VVIX_FILTER_VARIANT_NAME)
    print('vvix_feature =', vvix_feature_name)
    print('vvix_kept_buckets =', vvix_kept_buckets)
    print('vvix_selected_days =', vvix_selected_days)
    print('vvix_trades =', len(vvix_filter_trades))
    print('vvix_is_sharpe =', float(vvix_filter_is.get('sharpe_ratio', 0.0)))
    print('vvix_oos_sharpe =', float(vvix_filter_oos.get('sharpe_ratio', 0.0)))

display(Markdown(build_scope_readout_markdown(
    full_curve=ensemble_eq,
    oos_curve=normalize_curve(build_equity_curve(
        ensemble_trades.loc[pd.to_datetime(ensemble_trades['session_date']).dt.date.isin(set(oos_sessions))].copy(),
        initial_capital=float(BASELINE['account_size_usd']),
    )),
    initial_capital=float(BASELINE['account_size_usd']),
    full_label='Full-sample nominal ensemble curve',
    oos_label='OOS-only nominal ensemble curve',
)))

if sizing_enabled and not sizing_trades.empty:
    display(Markdown(build_scope_readout_markdown(
        full_curve=sizing_eq,
        oos_curve=normalize_curve(build_equity_curve(
            sizing_trades.loc[pd.to_datetime(sizing_trades['session_date']).dt.date.isin(set(oos_sessions))].copy(),
            initial_capital=float(BASELINE['account_size_usd']),
        )),
        initial_capital=float(BASELINE['account_size_usd']),
        full_label='Full-sample 3-state curve',
        oos_label='OOS-only 3-state curve',
    )))
if vvix_filter_enabled and not vvix_filter_trades.empty:
    display(Markdown(build_scope_readout_markdown(
        full_curve=vvix_filter_eq,
        oos_curve=normalize_curve(build_equity_curve(
            vvix_filter_trades.loc[pd.to_datetime(vvix_filter_trades['session_date']).dt.date.isin(set(oos_sessions))].copy(),
            initial_capital=float(BASELINE['account_size_usd']),
        )),
        initial_capital=float(BASELINE['account_size_usd']),
        full_label='Full-sample VVIX-filter curve',
        oos_label='OOS-only VVIX-filter curve',
    )))
"""

PLOT_CELL = """heat_src = sweep_df.copy()
heat_src['pair'] = 'q' + heat_src['q_low_pct'].astype(int).astype(str) + '/q' + heat_src['q_high_pct'].astype(int).astype(str)

heat_sh = heat_src.pivot_table(index='atr_period', columns='pair', values='oos_sharpe_ratio', aggfunc='mean').sort_index()
heat_pf = heat_src.pivot_table(index='atr_period', columns='pair', values='oos_profit_factor', aggfunc='mean').sort_index()

fig_heat = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.08,
    subplot_titles=('OOS Sharpe (full OOS calendar)', 'OOS Profit Factor (full OOS calendar)')
)
fig_heat.add_trace(
    go.Heatmap(z=heat_sh.to_numpy(), x=heat_sh.columns.tolist(), y=heat_sh.index.tolist(), colorscale='RdYlGn', colorbar=dict(title='Sharpe', x=0.46), zmin=0.5, zmax=1.5),
    row=1,
    col=1,
)
fig_heat.add_trace(
    go.Heatmap(z=heat_pf.to_numpy(), x=heat_pf.columns.tolist(), y=heat_pf.index.tolist(), colorscale='RdYlGn', colorbar=dict(title='PF', x=1.01), zmin=0.5, zmax=1.5),
    row=1,
    col=2,
)
fig_heat.update_layout(title='Parameter Heatmaps', width=1850, height=650, template='plotly_dark')
fig_heat.update_xaxes(title_text='Quantile pair', tickangle=45)
fig_heat.update_yaxes(title_text='ATR period')
fig_heat.show()

trend_df = (
    sweep_df.groupby('atr_period', as_index=False)
    .agg(
        median_oos_sharpe=('oos_sharpe_ratio', 'median'),
        median_oos_pf=('oos_profit_factor', 'median'),
        median_oos_expectancy=('oos_expectancy', 'median'),
        median_oos_selected_days=('oos_selected_days', 'median'),
    )
    .sort_values('atr_period')
)

fig_trend = make_subplots(rows=1, cols=3, subplot_titles=('Median OOS Sharpe', 'Median OOS PF', 'Median OOS Expectancy'))
fig_trend.add_trace(go.Scatter(x=trend_df['atr_period'], y=trend_df['median_oos_sharpe'], mode='lines+markers', line=dict(color='#22c55e')), row=1, col=1)
fig_trend.add_trace(go.Scatter(x=trend_df['atr_period'], y=trend_df['median_oos_pf'], mode='lines+markers', line=dict(color='#38bdf8')), row=1, col=2)
fig_trend.add_trace(go.Scatter(x=trend_df['atr_period'], y=trend_df['median_oos_expectancy'], mode='lines+markers', line=dict(color='#f59e0b')), row=1, col=3)
fig_trend.update_layout(template='plotly_dark', width=1850, height=520, title='Metrics Evolution by ATR')
fig_trend.update_xaxes(title_text='ATR period')
fig_trend.show()

best_atr_row = trend_df.sort_values(['median_oos_sharpe', 'median_oos_pf'], ascending=[False, False]).iloc[0]
display(Markdown(
    '### Quick summary\\n'
    f"- Best single cell (full OOS): **ATR {int(best_row['atr_period'])} | q{int(best_row['q_low_pct'])}/q{int(best_row['q_high_pct'])}**\\n"
    f"- Best ATR by median full-OOS behavior: **ATR {int(best_atr_row['atr_period'])}**\\n"
    f"- Median selected OOS days at best ATR: **{best_atr_row['median_oos_selected_days']:.0f}** / **{len(oos_sessions)}**\\n"
    f"- Trade floor used for ranking: **{trade_floor}**"
))

# -----------------------------
# Nominal ensemble vs 3-state vs Buy and Hold plot
# -----------------------------
ens_ret = total_return_pct(ensemble_eq)
ens_dd = max_drawdown_pct(ensemble_eq)
ens_cagr = annualized_return_from_equity(ensemble_eq) * 100.0
ens_vol = vol_daily_from_equity(ensemble_eq) * 100.0
ens_overall_sh = float(ensemble_overall.get('sharpe_ratio', 0.0))
ens_overall_pf = float(ensemble_overall.get('profit_factor', 0.0))
ens_overall_exp = float(ensemble_overall.get('expectancy', 0.0))
ens_is_sh = float(ensemble_is.get('sharpe_ratio', 0.0))
ens_oos_sh = float(ensemble_oos.get('sharpe_ratio', 0.0))
ens_oos_pf = float(ensemble_oos.get('profit_factor', 0.0))
ens_oos_exp = float(ensemble_oos.get('expectancy', 0.0))

siz_ret = total_return_pct(sizing_eq)
siz_dd = max_drawdown_pct(sizing_eq)
siz_cagr = annualized_return_from_equity(sizing_eq) * 100.0
siz_vol = vol_daily_from_equity(sizing_eq) * 100.0
siz_overall_sh = float(sizing_overall.get('sharpe_ratio', 0.0))
siz_overall_pf = float(sizing_overall.get('profit_factor', 0.0))
siz_overall_exp = float(sizing_overall.get('expectancy', 0.0))
siz_is_sh = float(sizing_is.get('sharpe_ratio', 0.0))
siz_oos_sh = float(sizing_oos.get('sharpe_ratio', 0.0))
siz_oos_pf = float(sizing_oos.get('profit_factor', 0.0))
siz_oos_exp = float(sizing_oos.get('expectancy', 0.0))

vvix_ret = total_return_pct(vvix_filter_eq)
vvix_dd = max_drawdown_pct(vvix_filter_eq)
vvix_cagr = annualized_return_from_equity(vvix_filter_eq) * 100.0
vvix_vol = vol_daily_from_equity(vvix_filter_eq) * 100.0
vvix_overall_sh = float(vvix_filter_overall.get('sharpe_ratio', 0.0))
vvix_overall_pf = float(vvix_filter_overall.get('profit_factor', 0.0))
vvix_overall_exp = float(vvix_filter_overall.get('expectancy', 0.0))
vvix_is_sh = float(vvix_filter_is.get('sharpe_ratio', 0.0))
vvix_oos_sh = float(vvix_filter_oos.get('sharpe_ratio', 0.0))
vvix_oos_pf = float(vvix_filter_oos.get('profit_factor', 0.0))
vvix_oos_exp = float(vvix_filter_oos.get('expectancy', 0.0))

bench_ret = total_return_pct(bench)
bench_dd = max_drawdown_pct(bench)
bench_sh = sharpe_daily_from_equity(bench)
bench_cagr = annualized_return_from_equity(bench) * 100.0
bench_vol = vol_daily_from_equity(bench) * 100.0
bench_is = bench.loc[bench['timestamp'].dt.date.isin(set(is_sessions))].copy()
bench_oos = bench.loc[bench['timestamp'].dt.date.isin(set(oos_sessions))].copy()
bench_is_sh = sharpe_daily_from_equity(bench_is)
bench_oos_sh = sharpe_daily_from_equity(bench_oos)

legend_ens = (
    f"Nominal ensemble | Overall Sharpe {ens_overall_sh:.2f} | PF {ens_overall_pf:.2f} | "
    f"Exp {ens_overall_exp:.1f} | DD {float(ensemble_overall.get('max_drawdown', 0.0)):.0f}"
)
legend_siz = (
    f"3-state sizing | Overall Sharpe {siz_overall_sh:.2f} | PF {siz_overall_pf:.2f} | "
    f"Exp {siz_overall_exp:.1f} | DD {float(sizing_overall.get('max_drawdown', 0.0)):.0f}"
)
legend_vvix = (
    f"VVIX filter | Overall Sharpe {vvix_overall_sh:.2f} | PF {vvix_overall_pf:.2f} | "
    f"Exp {vvix_overall_exp:.1f} | DD {float(vvix_filter_overall.get('max_drawdown', 0.0)):.0f}"
)
legend_bh = (
    f"Buy&Hold | Overall Sharpe {bench_sh:.2f} | Ret {bench_ret:.1f}% | "
    f"CAGR {bench_cagr:.1f}% | Vol {bench_vol:.1f}% | MaxDD {bench_dd:.1f}%"
)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.075,
    row_heights=[0.70, 0.30],
    subplot_titles=('Equity Curve (USD)', 'Drawdown (%)')
)

fig.add_trace(
    go.Scatter(x=ensemble_eq['timestamp'], y=ensemble_eq['equity'], mode='lines', name=legend_ens, line=dict(width=3.0, color='#22c55e')),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=ensemble_eq['timestamp'], y=ensemble_eq['drawdown_pct'], mode='lines', name='DD Nominal ensemble', showlegend=False, line=dict(width=1.7, color='#22c55e', dash='dot')),
    row=2,
    col=1,
)

if sizing_enabled and not sizing_eq.empty:
    fig.add_trace(
        go.Scatter(x=sizing_eq['timestamp'], y=sizing_eq['equity'], mode='lines', name=legend_siz, line=dict(width=2.7, color='#f59e0b')),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=sizing_eq['timestamp'], y=sizing_eq['drawdown_pct'], mode='lines', name='DD 3-state sizing', showlegend=False, line=dict(width=1.5, color='#f59e0b', dash='dot')),
        row=2,
        col=1,
    )

if vvix_filter_enabled and not vvix_filter_eq.empty:
    fig.add_trace(
        go.Scatter(x=vvix_filter_eq['timestamp'], y=vvix_filter_eq['equity'], mode='lines', name=legend_vvix, line=dict(width=2.7, color='#ef4444')),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=vvix_filter_eq['timestamp'], y=vvix_filter_eq['drawdown_pct'], mode='lines', name='DD VVIX filter', showlegend=False, line=dict(width=1.5, color='#ef4444', dash='dot')),
        row=2,
        col=1,
    )

fig.add_trace(
    go.Scatter(x=bench['timestamp'], y=bench['equity'], mode='lines', name=legend_bh, line=dict(width=2.6, color='#38bdf8')),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=bench['timestamp'], y=bench['drawdown_pct'], mode='lines', name='DD Buy&Hold', showlegend=False, line=dict(width=1.5, color='#38bdf8', dash='dot')),
    row=2,
    col=1,
)

summary_lines = [
    format_stats_line('Nominal ensemble', ens_overall_sh, ens_ret, ens_cagr, ens_vol, ens_dd, pf=ens_overall_pf, exp=ens_overall_exp),
    f"<b>Nominal split</b> | IS Sharpe {ens_is_sh:.2f} | OOS Sharpe {ens_oos_sh:.2f} | OOS PF {ens_oos_pf:.2f} | OOS Exp {ens_oos_exp:.1f}",
]
if sizing_enabled:
    summary_lines.extend(
        [
            format_stats_line('3-state sizing', siz_overall_sh, siz_ret, siz_cagr, siz_vol, siz_dd, pf=siz_overall_pf, exp=siz_overall_exp),
            f"<b>3-state split</b> | IS Sharpe {siz_is_sh:.2f} | OOS Sharpe {siz_oos_sh:.2f} | OOS PF {siz_oos_pf:.2f} | OOS Exp {siz_oos_exp:.1f}",
            f"<b>3-state params</b> | feature {SIZING_FEATURE_NAME} | map {state_map_text} | active days {sizing_selected_days}",
        ]
    )
if vvix_filter_enabled:
    summary_lines.extend(
        [
            format_stats_line('VVIX filter', vvix_overall_sh, vvix_ret, vvix_cagr, vvix_vol, vvix_dd, pf=vvix_overall_pf, exp=vvix_overall_exp),
            f"<b>VVIX split</b> | IS Sharpe {vvix_is_sh:.2f} | OOS Sharpe {vvix_oos_sh:.2f} | OOS PF {vvix_oos_pf:.2f} | OOS Exp {vvix_oos_exp:.1f}",
            f"<b>VVIX params</b> | {vvix_state_text} | active days {vvix_selected_days}",
        ]
    )
summary_lines.extend(
    [
        f"<b>Buy&Hold</b> | Overall Sharpe {bench_sh:.2f} | IS Sharpe {bench_is_sh:.2f} | OOS Sharpe {bench_oos_sh:.2f} | Ret {bench_ret:.1f}% | CAGR {bench_cagr:.1f}% | Vol {bench_vol:.1f}% | MaxDD {bench_dd:.1f}%",
        f"<b>Params</b> | ATR {ATR_PERIODS} | Qlow {Q_LOW_VALUES} | Qhigh {Q_HIGH_VALUES} | agg {AGGREGATION_RULE} | thr {consensus_threshold:.2f} | Nsub {n_subsignals}",
        f"<b>Best cell</b> | ATR {int(best_row['atr_period'])} | q{int(best_row['q_low_pct'])}/q{int(best_row['q_high_pct'])} | OOS days {int(best_row['oos_selected_days'])}/{len(oos_sessions)}",
    ]
)

annotation_y = 1.34 if sizing_enabled and vvix_filter_enabled else (1.28 if sizing_enabled or vvix_filter_enabled else 1.22)
comparison_parts = ['Nominal ensemble']
if sizing_enabled and not sizing_eq.empty:
    comparison_parts.append('3-state sizing')
if vvix_filter_enabled and not vvix_filter_eq.empty:
    comparison_parts.append('VVIX filter')
comparison_parts.append('Buy&Hold')

fig.add_annotation(
    xref='paper', yref='paper', x=0.01, y=annotation_y,
    text='<br>'.join(summary_lines),
    showarrow=False, align='left',
    bordercolor='rgba(148,163,184,0.40)', borderwidth=1, borderpad=10,
    bgcolor='rgba(15,23,42,0.94)', font=dict(size=13, color='#e5e7eb'),
)

fig.update_layout(
    template='plotly_dark',
    width=1850,
    height=1080,
    title=dict(
        text=' vs '.join(comparison_parts),
        x=0.5,
        xanchor='center',
        font=dict(size=24),
    ),
    legend=dict(orientation='h', yanchor='bottom', y=-0.24, xanchor='left', x=0.0, font=dict(size=11), bgcolor='rgba(15,23,42,0.82)', bordercolor='rgba(148,163,184,0.25)', borderwidth=1),
    margin=dict(l=70, r=40, t=275 if sizing_enabled and vvix_filter_enabled else (240 if sizing_enabled or vvix_filter_enabled else 195), b=140),
    paper_bgcolor='#020617',
    plot_bgcolor='#020617',
)
fig.update_annotations(font=dict(size=16, color='#e5e7eb'))
fig.update_yaxes(title_text='Equity (USD)', row=1, col=1)
fig.update_yaxes(title_text='Drawdown %', row=2, col=1)
fig.update_xaxes(title_text='Time', row=2, col=1)

fig.show()
"""

KPI_CELL = """kpi_rows = [
    {
        'model': 'ensemble_nominal',
        'overall_sharpe': ens_overall_sh,
        'is_sharpe': ens_is_sh,
        'oos_sharpe': ens_oos_sh,
        'overall_profit_factor': ens_overall_pf,
        'oos_profit_factor': ens_oos_pf,
        'overall_expectancy': ens_overall_exp,
        'oos_expectancy': ens_oos_exp,
        'total_return_pct': ens_ret,
        'cagr_pct': ens_cagr,
        'vol_pct': ens_vol,
        'max_drawdown_pct': ens_dd,
        'n_subsignals': n_subsignals,
        'selected_signal_days': int(candidates['ensemble_pass'].sum()),
    }
]

if sizing_enabled:
    kpi_rows.append(
        {
            'model': 'ensemble_3state',
            'overall_sharpe': siz_overall_sh,
            'is_sharpe': siz_is_sh,
            'oos_sharpe': siz_oos_sh,
            'overall_profit_factor': siz_overall_pf,
            'oos_profit_factor': siz_oos_pf,
            'overall_expectancy': siz_overall_exp,
            'oos_expectancy': siz_oos_exp,
            'total_return_pct': siz_ret,
            'cagr_pct': siz_cagr,
            'vol_pct': siz_vol,
            'max_drawdown_pct': siz_dd,
            'n_subsignals': n_subsignals,
            'selected_signal_days': sizing_selected_days,
        }
    )

if vvix_filter_enabled:
    kpi_rows.append(
        {
            'model': 'ensemble_vvix_filter',
            'overall_sharpe': vvix_overall_sh,
            'is_sharpe': vvix_is_sh,
            'oos_sharpe': vvix_oos_sh,
            'overall_profit_factor': vvix_overall_pf,
            'oos_profit_factor': vvix_oos_pf,
            'overall_expectancy': vvix_overall_exp,
            'oos_expectancy': vvix_oos_exp,
            'total_return_pct': vvix_ret,
            'cagr_pct': vvix_cagr,
            'vol_pct': vvix_vol,
            'max_drawdown_pct': vvix_dd,
            'n_subsignals': n_subsignals,
            'selected_signal_days': vvix_selected_days,
        }
    )

kpi_rows.append(
    {
        'model': 'buy_and_hold',
        'overall_sharpe': bench_sh,
        'is_sharpe': bench_is_sh,
        'oos_sharpe': bench_oos_sh,
        'overall_profit_factor': np.nan,
        'oos_profit_factor': np.nan,
        'overall_expectancy': np.nan,
        'oos_expectancy': np.nan,
        'total_return_pct': bench_ret,
        'cagr_pct': bench_cagr,
        'vol_pct': bench_vol,
        'max_drawdown_pct': bench_dd,
        'n_subsignals': np.nan,
        'selected_signal_days': np.nan,
    }
)

kpi = pd.DataFrame(kpi_rows)
display(kpi)

if sizing_enabled:
    display(Markdown(
        '### 3-state sizing diagnostics\\n'
        f"- Feature: **{SIZING_FEATURE_NAME}**\\n"
        f"- Mapping: **{state_map_text}**\\n"
        f"- Override active: **{bool(SIZING_BUCKET_MULTIPLIERS_OVERRIDE)}**\\n"
        f"- Active selected days after sizing map: **{sizing_selected_days}**"
    ))

    if not sizing_feature_scores.empty:
        score_cols = [
            'feature_name',
            'family',
            'bucket_count',
            'min_bucket_obs_is',
            'balance_is',
            'is_score_spread',
            'feature_selection_score',
            'valid_for_overlay',
        ]
        display(sizing_feature_scores[score_cols])

    bucket_cols = [
        'bucket_label',
        'lower_bound',
        'upper_bound',
        'is_n_obs',
        'is_expectancy',
        'is_profit_factor',
        'oos_n_obs',
        'oos_expectancy',
        'oos_profit_factor',
    ]
    bucket_display = sizing_feature_rows[bucket_cols].copy()
    bucket_display['risk_multiplier'] = bucket_display['bucket_label'].map(sizing_state_map).astype(float)
    for col in ['lower_bound', 'upper_bound', 'is_expectancy', 'is_profit_factor', 'oos_expectancy', 'oos_profit_factor', 'risk_multiplier']:
        bucket_display[col] = pd.to_numeric(bucket_display[col], errors='coerce').round(3)
    display(bucket_display)

if vvix_filter_enabled:
    display(Markdown(
        '### VVIX filter diagnostics\\n'
        f"- Export root: **{VVIX_VALIDATION_EXPORT_ROOT}**\\n"
        f"- Variant: **{VVIX_FILTER_VARIANT_NAME}**\\n"
        f"- Feature used: **{vvix_feature_name}**\\n"
        f"- Kept buckets: **{', '.join(vvix_kept_buckets)}**\\n"
        f"- Active selected days after filter: **{vvix_selected_days}**"
    ))

    vvix_bucket_cols = [
        'bucket_label',
        'lower_bound',
        'upper_bound',
        'is_n_obs',
        'oos_n_obs',
    ]
    vvix_bucket_display = vvix_feature_rows[vvix_bucket_cols].copy()
    vvix_bucket_counts = (
        vvix_filter_controls.groupby(['phase', 'bucket_label'], as_index=False)
        .agg(
            notebook_signal_days=('session_date', 'count'),
            kept_signal_days=('selected', 'sum'),
        )
    )
    vvix_bucket_counts = vvix_bucket_counts.pivot_table(
        index='bucket_label',
        columns='phase',
        values=['notebook_signal_days', 'kept_signal_days'],
        aggfunc='sum',
        fill_value=0,
    )
    vvix_bucket_counts.columns = [
        f"{value}_{phase}" for value, phase in vvix_bucket_counts.columns.to_flat_index()
    ]
    vvix_bucket_counts = vvix_bucket_counts.reset_index()
    vvix_bucket_display = vvix_bucket_display.merge(vvix_bucket_counts, on='bucket_label', how='left')
    vvix_bucket_display['kept_by_variant'] = vvix_bucket_display['bucket_label'].isin(set(vvix_kept_buckets))
    for col in ['lower_bound', 'upper_bound']:
        vvix_bucket_display[col] = pd.to_numeric(vvix_bucket_display[col], errors='coerce').round(3)
    display(vvix_bucket_display)

display(Markdown(
    '### Top parameter cells (full-calendar OOS)\\n'
    'Heatmaps and ranking below use the entire OOS window so they stay comparable to the ensemble OOS metrics.'
))
display(
    ranked[
        [
            'atr_period',
            'q_low_pct',
            'q_high_pct',
            'oos_selected_days',
            'oos_n_trades',
            'oos_sharpe_ratio',
            'oos_profit_factor',
            'oos_expectancy',
        ]
    ].head(12)
)
"""


def _replace_once(source: str, old: str, new: str) -> str:
    if old in source:
        return source.replace(old, new, 1)
    return source


def patch_notebook(path: Path) -> None:
    nb = nbf.read(path, as_version=4)
    if len(nb.cells) < 12:
        raise RuntimeError(f"Unexpected notebook structure in {path}.")

    if LEGACY_INTRO_APPENDIX in nb.cells[0].source:
        nb.cells[0].source = nb.cells[0].source.replace(LEGACY_INTRO_APPENDIX, INTRO_APPENDIX)
    if INTRO_APPENDIX not in nb.cells[0].source:
        nb.cells[0].source = nb.cells[0].source.rstrip() + "\n\n" + INTRO_APPENDIX + "\n"

    nb.cells[1].source = IMPORTS_CELL
    nb.cells[2].source = "## 1) Parameters (edit here, including the 3-state and VVIX overlays)"
    nb.cells[3].source = PARAMETERS_CELL
    nb.cells[6].source = "## 3) Backtest nominal ensemble, overlays, and benchmark"
    nb.cells[7].source = BACKTEST_CELL
    nb.cells[8].source = "## 4) Dark Plotly chart (Nominal ensemble vs overlays vs Buy and Hold)"
    nb.cells[9].source = PLOT_CELL
    nb.cells[10].source = "## 5) KPI table + overlay diagnostics"
    nb.cells[11].source = KPI_CELL

    for idx in (1, 3, 5, 7, 9, 11):
        cell = nb.cells[idx]
        if cell.cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    nbf.write(nb, path)


def execute_notebook(path: Path, timeout_seconds: int) -> None:
    nb = nbf.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout_seconds,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    nbf.write(nb, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=NOTEBOOK_PATH)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patch_notebook(args.path)
    print(f"patched {args.path}")
    if args.execute:
        execute_notebook(args.path, timeout_seconds=args.timeout_seconds)
        print(f"executed {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
